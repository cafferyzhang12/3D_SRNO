"""Support-certified local 2-jets for TRAIN-only MRI high-frequency geometry.

Legacy atlas and metric protocols are deliberately untouched. A chart is a
quadratic normal graph f(q)=c+Uq+a+Lq+q^T Q q. Geometry is analytic in chart
coordinates; a damped Gauss--Newton solve approximates the closest point.
There is no assumption that a finite atlas covers an arbitrary MRI cohort.
"""
from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


def hf_patches(coefficients, stride=1):
    """3x3 XY Haar neighbourhoods, BZYX row order, excluding LL (135 dims)."""
    if coefficients.ndim != 5 or coefficients.shape[1] != 16:
        raise ValueError('Expected B16ZYX orthonormal Haar coefficients')
    if type(stride) is not int or stride < 1:
        raise ValueError('HF neighbourhood stride must be a positive integer')
    b, _, z, h, w = coefficients.shape
    planes = coefficients[:, 1:].permute(0, 2, 1, 3, 4).reshape(b*z, 15, h, w)
    return F.unfold(F.pad(planes, (1, 1, 1, 1), mode='replicate'), 3, stride=stride).transpose(1, 2).reshape(-1, 135)


def _lr_grid(lr, lr_dim, grid_shape):
    if lr.ndim != 5 or lr.shape[1] not in (1, lr_dim):
        raise ValueError('LR router features must have B,1 or lr_dim,Z,Y,X shape')
    if lr.shape[1] == 1 and lr_dim not in (1, 9):
        raise ValueError('Native LR inputs support scalar or 3x3 router features')
    if grid_shape is not None and tuple(lr.shape[2:]) != tuple(grid_shape):
        lr = F.interpolate(lr,size=grid_shape,mode='trilinear',align_corners=False)
    return lr


def lr_router_rows(lr, lr_dim=9, *, grid_shape=None, stride=1):
    """Raw LR router features aligned with hf_patches in BZYX row order."""
    lr = _lr_grid(lr,lr_dim,grid_shape)
    if lr.shape[1] == lr_dim:
        return lr[:,:,:,::stride,::stride].movedim(1,-1).reshape(-1,lr_dim)
    b,_,z,h,w = lr.shape
    planes = lr.reshape(b*z,1,h,w)
    columns = F.unfold(F.pad(planes,(1,1,1,1),mode='replicate'),3,stride=stride)
    return columns.transpose(1,2).reshape(-1,lr_dim)


def _site_rows(volume, sites, radius=0):
    """Gather replicated XY neighborhoods at per-item flattened ZYX sites."""
    _,_,_,h,w = volume.shape
    batch = torch.arange(len(volume),device=sites.device)[:,None]
    z = sites.div(h*w,rounding_mode='floor')
    y = sites.div(w,rounding_mode='floor').remainder(h)
    x = sites.remainder(w)
    if radius == 0:
        return volume.movedim(1,-1)[batch,z,y,x]
    offsets = torch.arange(-radius,radius+1,device=sites.device)
    width = len(offsets)
    y = (y[...,None] + offsets.repeat_interleave(width)).clamp(0,h-1)
    x = (x[...,None] + offsets.repeat(width)).clamp(0,w-1)
    blocks = volume.movedim(1,-1)[batch[...,None],z[...,None],y,x]
    return blocks.transpose(-1,-2).flatten(-2)


def fold_hf_patches(patches, shape):
    """Adjoint overlap-average of hf_patches; LL is identically zero.

    Replicated halo entries are folded back onto their original boundary sites,
    rather than discarded. Thus extraction/reconstruction is exactly identity.
    """
    b, _, z, h, w = shape
    if patches.shape != (b*z*h*w, 135):
        raise ValueError('Patch row count differs from BZYX shape')
    columns = patches.reshape(b*z, h*w, 135).transpose(1, 2)
    padded = F.fold(columns, (h+2, w+2), 3)
    counts = F.fold(torch.ones_like(columns[:, :9]), (h+2, w+2), 3)
    # The padded grid is a replicated image; accumulate halo mass at its source.
    for axis in (-1, -2):
        left = [slice(None)] * 4
        near = [slice(None)] * 4
        left[axis], near[axis] = slice(0, 1), slice(1, 2)
        padded[tuple(near)] = padded[tuple(near)] + padded[tuple(left)]
        counts[tuple(near)] = counts[tuple(near)] + counts[tuple(left)]
        left[axis], near[axis] = slice(-1, None), slice(-2, -1)
        padded[tuple(near)] = padded[tuple(near)] + padded[tuple(left)]
        counts[tuple(near)] = counts[tuple(near)] + counts[tuple(left)]
    planes = padded[..., 1:-1, 1:-1] / counts[..., 1:-1, 1:-1]
    hf = planes.reshape(b, z, 15, h, w).permute(0, 2, 1, 3, 4)
    return torch.cat((hf.new_zeros(b, 1, z, h, w), hf), 1)


def smooth_bound(q, radius):
    """Algebraic soft limit with positive derivative at every finite coordinate."""
    return q * torch.rsqrt(1 + (q / radius.clamp_min(1.e-8)).square())


def safe_vector(value, maximum):
    return value * (maximum / value.norm(dim=-1, keepdim=True).clamp_min(maximum))


def case_folds(case_ids, folds=5, seed=42):
    cases = sorted(set(case_ids), key=lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).digest())
    if len(cases) < folds or folds < 2:
        raise ValueError('Cross-fitting requires at least one distinct TRAIN case per fold')
    return {case: i % folds for i, case in enumerate(cases)}


class TrainPCA(nn.Module):
    def __init__(self, mean, basis):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('basis', basis)

    @classmethod
    @torch.no_grad()
    def fit(cls, rows, dimensions):
        if rows.ndim != 2 or len(rows) <= dimensions or dimensions < 2 or not rows.isfinite().all():
            raise ValueError('PCA requires finite TRAIN rows with N > feature dimensions')
        mean = rows.double().mean(0)
        _, _, vh = torch.linalg.svd(rows.double() - mean, full_matrices=False)
        return cls(mean.to(rows), vh[:dimensions].T.to(rows))

    def forward(self, rows):
        return (rows - self.mean) @ self.basis

    def inverse_delta(self, delta):
        return delta @ self.basis.T


class SCJetAtlas(nn.Module):
    """Batched quadratic normal graphs with separate fit and blend bandwidths.

    query_second_order returns one chart or all supplied chart candidates.
    It never averages unrelated chart coordinate systems. Soft routing blends
    scalar costs or ambient corrections after each chart's geometric query.
    """
    def __init__(self, payload):
        super().__init__()
        self.settings = dict(payload.get('settings', {}))
        for key in ('centers', 'basis', 'offset', 'linear', 'quadratic', 'radius',
                    'fit_bandwidth', 'blend_bandwidth', 'effective', 'supported', 'ridge', 'condition'):
            self.register_buffer(key, payload[key].detach().clone())
        self.dimension, self.intrinsic = self.basis.shape[-2:]
        if self.intrinsic >= self.dimension:
            raise ValueError('A normal graph requires intrinsic < ambient dimension')

    @property
    def chart_count(self):
        return len(self.centers)

    @classmethod
    @torch.no_grad()
    def fit(cls, rows, *, centers=512, intrinsic=8, neighbors=256, normal_rank=4,
            blend_gamma=.5, seed=42, support_min=1., condition_limit=1.e8):
        if rows.ndim != 2 or not rows.isfinite().all() or intrinsic >= rows.shape[1] or intrinsic < 1:
            raise ValueError('Invalid finite TRAIN feature matrix or intrinsic dimension')
        n, ambient = rows.shape
        p = 1 + intrinsic + intrinsic*(intrinsic+1)//2
        k = min(n, max(neighbors, 2*p))
        if k < intrinsic+2 or centers < 1 or not 0 < blend_gamma <= 1:
            raise ValueError('Insufficient TRAIN support or invalid bandwidth')
        x = rows.double()
        gen = torch.Generator(device=rows.device).manual_seed(seed)
        selected = torch.randperm(n, generator=gen, device=rows.device)[:min(n, centers)]
        payload = {key: [] for key in ('centers','basis','offset','linear','quadratic','radius',
                                      'fit_bandwidth','blend_bandwidth','effective','supported','ridge','condition')}
        ii, jj = torch.triu_indices(intrinsic, intrinsic, device=rows.device)
        ridge_grid = torch.logspace(-7, -1, 13, device=rows.device, dtype=torch.float64)
        for anchor in selected:
            distances, ids = (x - x[anchor]).square().sum(-1).topk(k, largest=False)
            # Broad fitting kernel: effective support is distinct from narrow blending.
            bandwidth = distances[-1].sqrt().clamp_min(1.e-6)
            weights = (-.5 * distances / bandwidth.square()).softmax(0)
            effective = weights.square().sum().reciprocal()
            mean = (weights[:, None] * x[ids]).sum(0)
            centered = x[ids]-mean
            eig, directions = torch.linalg.eigh(centered.T @ (weights[:, None]*centered))
            basis = directions[:, -intrinsic:].flip(-1)
            normal = directions[:, :min(normal_rank, ambient-intrinsic)]
            # Principal normals are the largest residual eigenvectors, not the smallest.
            normal = directions[:, max(0,ambient-intrinsic-normal_rank):ambient-intrinsic].flip(-1)
            q = centered @ basis
            scale = (weights[:, None] * q.square()).sum(0).sqrt().clamp_min(1.e-5)
            qn = q/scale
            design = torch.cat((torch.ones(k,1,device=x.device,dtype=x.dtype), qn,
                                qn[:,ii]*qn[:,jj]), -1)
            a = weights.sqrt()[:,None] * design
            y = weights.sqrt()[:,None] * (centered @ normal)
            u, s, vh = torch.linalg.svd(a, full_matrices=False)
            uy = u.T@y
            lam = ridge_grid*s[0].square().clamp_min(1.e-12)
            hat = s[None].square()/(s[None].square()+lam[:,None])
            residual = y[None] - torch.einsum('kp,rp,pd->rkd',u,hat,uy)
            gcv = residual.square().sum((1,2))/(k-hat.sum(-1)).clamp_min(1.e-6).square()
            ridge = lam[gcv.argmin()]
            coeff = (vh.T * (s/(s.square()+ridge))) @ uy @ normal.T
            # Reparameterise from normalized coordinates into physical PCA units.
            quadratic = x.new_zeros(intrinsic,intrinsic,ambient)
            quad_coeff = coeff[1+intrinsic:] / (scale[ii]*scale[jj])[:,None]
            quadratic[ii,jj] += .5*quad_coeff
            quadratic[jj,ii] += .5*quad_coeff
            condition = s[0]/s[-1].clamp_min(1.e-14)
            supported = (effective >= support_min*p) & (condition < condition_limit) & (eig[-intrinsic] > 1.e-12)
            # Unsupported fits retain tangent PCA but carry no unmeasured curvature.
            values = dict(centers=mean,basis=basis,offset=coeff[0]*supported,
                linear=coeff[1:1+intrinsic].T/scale* supported,
                quadratic=quadratic*supported,radius=qn.abs().amax(0).clamp_min(1.)*scale,
                fit_bandwidth=bandwidth,blend_bandwidth=blend_gamma*bandwidth,
                effective=effective,supported=supported,ridge=ridge,condition=condition)
            for key,value in values.items():
                payload[key].append(value)
        dtype = rows.dtype
        payload = {k: torch.stack(v).to(dtype=dtype) if k != 'supported' else torch.stack(v) for k,v in payload.items()}
        payload['settings'] = dict(blend_gamma=blend_gamma,neighbors=k,normal_rank=normal_rank,
                                   intrinsic=intrinsic,seed=seed)
        return cls(payload)

    def payload(self):
        return dict(settings=self.settings, **{k:v.detach().cpu() for k,v in self.named_buffers()})

    def decode_jet(self, chart_ids, q):
        basis = self.basis[chart_ids]
        form = self.quadratic[chart_ids]
        value = (self.centers[chart_ids]+self.offset[chart_ids]+
                 torch.einsum('...di,...i->...d',basis+self.linear[chart_ids],q)+
                 torch.einsum('...i,...ijd,...j->...d',q,form,q))
        jacobian = basis+self.linear[chart_ids]+2*torch.einsum('...i,...ijd->...dj',q,form)
        hessian = 2*form.movedim(-1,-3)
        return value,jacobian,hessian

    def distribution(self, x):
        distances = torch.cdist(x,self.centers).square()
        return (-.5*distances/self.blend_bandwidth.square().clamp_min(1.e-10)).softmax(-1)

    def _projection_jet(self, x, chart_ids, gn_steps, *, return_jacobian=True):
        basis = self.basis[chart_ids]
        center = self.centers[chart_ids]
        intercept = center + self.offset[chart_ids]
        linear = basis + self.linear[chart_ids]
        form = self.quadratic[chart_ids]
        radius = self.radius[chart_ids]
        q_raw = (basis.mT @ (x-center)[...,None])[...,0]
        q = smooth_bound(q_raw,radius)

        def decode(coordinates, with_jacobian):
            q_form = (coordinates[...,None,:] @ form.flatten(-2)).reshape_as(linear.mT)
            value = intercept + ((linear+q_form.mT) @ coordinates[...,None])[...,0]
            return value, linear + 2*q_form.mT if with_jacobian else None

        if gn_steps:
            projection,jacobian = decode(q,True)
            gram = jacobian.mT @ jacobian
            eye = torch.eye(self.intrinsic,device=x.device,dtype=x.dtype)
            ridge = gram.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1.e-10)*1.e-6
            root,_ = torch.linalg.cholesky_ex(gram+ridge[...,None,None]*eye)
            rhs = jacobian.mT @ (x-projection)[...,None]
            step = torch.cholesky_solve(rhs,root)[...,0]
            q = q + smooth_bound(step,radius*.5)
        projection,jacobian = decode(q,return_jacobian)
        return projection,jacobian,form,q_raw/radius.clamp_min(1.e-8)

    def _support(self, relative, chart_ids):
        distance = (torch.logsumexp(8*(relative.square()+1.e-12).sqrt(),-1)
                    -math.log(self.intrinsic))/8-1
        confidence = torch.sigmoid(-4*distance)*(.1+.9*self.supported[chart_ids].to(relative))
        return distance,confidence

    def query_projection_only(self, x, chart_ids=None, gn_steps=1, *, return_tangent=False):
        """Damped closest-point estimate with optional orthonormal tangent frame."""
        if x.shape[-1] != self.dimension or gn_steps not in (0,1):
            raise ValueError('Projection requires matching features and at most one GN step')
        if chart_ids is None:
            chart_ids = self.distribution(x).argmax(-1)
        projection,jacobian,_,relative = self._projection_jet(x,chart_ids,gn_steps,return_jacobian=return_tangent)
        distance,confidence = self._support(relative,chart_ids)
        result = dict(projection=projection,support_distance=distance,confidence=confidence)
        if return_tangent:
            root,_ = torch.linalg.cholesky_ex(jacobian.mT @ jacobian)
            result['tangent'] = torch.linalg.solve_triangular(root,jacobian.mT,upper=False).mT
        return result

    def query_second_order(self, x, *, chart_ids=None, direction=None, gn_steps=2):
        if x.shape[-1] != self.dimension or not 0 <= gn_steps <= 2:
            raise ValueError('Invalid second-order feature query')
        if chart_ids is None:
            chart_ids = self.distribution(x).argmax(-1)
        q_raw = torch.einsum('...di,...d->...i',self.basis[chart_ids],x-self.centers[chart_ids])
        q = smooth_bound(q_raw,self.radius[chart_ids])
        eye = torch.eye(self.intrinsic,device=x.device,dtype=x.dtype)
        for _ in range(gn_steps):
            projection,jac,_ = self.decode_jet(chart_ids,q)
            gram = jac.mT@jac
            ridge = gram.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1.e-10)*1.e-6
            step = torch.linalg.solve(gram+ridge[...,None,None]*eye,
                                     (jac.mT@(x-projection)[...,None]))[...,0]
            # Limit the step, not the coordinate. No repeated interior shrink bias.
            q = q + smooth_bound(step,self.radius[chart_ids]*.5)
        projection,jac,hessian = self.decode_jet(chart_ids,q)
        tangent = torch.linalg.qr(jac,mode='reduced').Q
        gram_inverse = torch.linalg.inv(jac.mT@jac)
        ambient_hessian_trace = torch.einsum('...ij,...dij->...d',gram_inverse,hessian)
        mean_curvature = ambient_hessian_trace-torch.einsum('...di,...i->...d',tangent,
                                          torch.einsum('...di,...d->...i',tangent,ambient_hessian_trace))
        # Signed soft support boundary distance, positive outside the measured box.
        relative = q_raw/self.radius[chart_ids].clamp_min(1.e-8)
        support_distance,confidence = self._support(relative,chart_ids)
        result = dict(projection=projection,tangent=tangent,jacobian=jac,hessian=hessian,
            mean_curvature=mean_curvature,support_distance=support_distance,confidence=confidence,
            coordinates=q,chart_ids=chart_ids)
        if direction is not None:
            result['second_fundamental'] = self.second_fundamental(result,direction)
        return result

    @staticmethod
    def second_fundamental(query, direction):
        j,u = query['jacobian'],query['tangent']
        coordinates = torch.linalg.solve(j.mT@j,(j.mT@direction[...,None]))[...,0]
        raw = torch.einsum('...i,...dij,...j->...d',coordinates,query['hessian'],coordinates)
        return raw-(u@(u.mT@raw[...,None]))[...,0]


def jet_asymmetric_loss(error, query, normal_weight=1., tangent_weight=.05, curvature_limit=1.):
    u = query['tangent'].detach()
    tangent = (u@(u.mT@error[...,None]))[...,0]
    curvature = safe_vector(SCJetAtlas.second_fundamental(
        {key:value.detach() if torch.is_tensor(value) else value for key,value in query.items()},tangent),curvature_limit)
    corrected = error-.5*curvature
    normal = corrected-(u@(u.mT@corrected[...,None]))[...,0]
    loss = normal_weight*normal.square().mean()+tangent_weight*tangent.square().mean()
    return loss,normal,tangent


def jet_potential(state, query, surface_weight=1., support_weight=.01, support_temperature=.25):
    return (surface_weight*(state-query['projection']).square().mean(-1)+
            support_weight*F.softplus(query['support_distance']/support_temperature).square()).mean()


def empirical_bayes_refine(x, query, noise_variance, tau=.1):
    if not 0 <= tau <= 1 or noise_variance < 0:
        raise ValueError('EB tau in [0,1] and nonnegative variance required')
    residual = x-query['projection']
    u = query['tangent']
    normal = residual-(u@(u.mT@residual[...,None]))[...,0]
    codimension = x.shape[-1]-u.shape[-1]
    shrink = (1-tau*codimension*noise_variance/(normal.square().sum(-1,keepdim=True)+1.e-8)).clamp_min(0)
    return x-(1-shrink)*normal


class SoftAtlasRouter(nn.Module):
    """TRAIN-distilled chart logits from LR features and current state only."""
    def __init__(self, lr_dim, state_dim, charts, hidden=64, top_m=3):
        super().__init__()
        self.lr_dim,self.state_dim,self.top_m = lr_dim,state_dim,min(top_m,charts)
        self.network = nn.Sequential(nn.Linear(lr_dim+state_dim,hidden),nn.SiLU(),nn.Linear(hidden,charts))

    def forward(self, lr_features, state):
        if lr_features.shape[:-1] != state.shape[:-1] or lr_features.shape[-1] != self.lr_dim or state.shape[-1] != self.state_dim:
            raise ValueError('Router expects aligned LR/current-state feature rows')
        return self.network(torch.cat((lr_features,state),-1))

    def route(self, lr_features, state):
        logits,ids = self(lr_features,state).topk(self.top_m,-1)
        return ids,logits.softmax(-1)

    def distillation_loss(self, lr_features, state, teacher):
        return F.kl_div(self(lr_features,state).log_softmax(-1),teacher.detach(),reduction='batchmean')

class JetAtlasRegularizer(nn.Module):
    """Case-cross-fitted PCA, jets and routers in the supplied coefficient units.

    The pipeline supplies TRAIN-whitened coefficients consistently at fit,
    training and deployment; a coefficient-only caller may use physical units.

    Each fold's PCA and charts exclude every row belonging to that fold's cases.
    Its router is distilled on excluded TRAIN cases only. Deployment averages
    ambient corrections from these TRAIN-distilled routers; HR is never an
    inference argument. This avoids using held-out cases to fit whitening/PCA.
    """
    version = 2

    def __init__(self, pcas, atlases, routers, folds, settings, provenance=None):
        super().__init__()
        self.pcas,self.atlases,self.routers = nn.ModuleList(pcas),nn.ModuleList(atlases),nn.ModuleList(routers)
        self.folds,self.settings = dict(folds),dict(settings)
        self.provenance = dict(provenance or {})
        self.lr_dim = int(settings.get('lr_dim',routers[0].lr_dim))
        self.site_budget = int(settings.get('query_sites',128))
        if self.site_budget < 1:
            raise ValueError('Positive geometric site budget required')

    @classmethod
    @torch.no_grad()
    def fit_from_bank(cls, features, case_ids, settings=None, lr_features=None, *, split='train'):
        if split != 'train':
            raise ValueError('JetAtlas fitting accepts TRAIN only')
        settings = dict(settings or {})
        aliases = dict(projection_dim='pca_dim',intrinsic='dim',query_sites='loss_sites_per_patch',
                       normal_weight='lambda_normal',tangent_weight='lambda_tangent')
        for canonical,alias in aliases.items():
            if alias in settings and canonical not in settings:
                settings[canonical]=settings[alias]
        if features.ndim != 2 or features.shape[1] != 135 or len(features) != len(case_ids) or not features.isfinite().all():
            raise ValueError('Bank needs aligned finite TRAIN 3x3 HF patches and case IDs')
        if any(not isinstance(case,str) or not case for case in case_ids):
            raise ValueError('A nonempty TRAIN case ID is required per patch row')
        folds = case_folds(case_ids,int(settings.get('folds',5)),int(settings.get('seed',42)))
        row_folds = torch.tensor([folds[c] for c in case_ids],device=features.device)
        dimensions = int(settings.get('projection_dim',32))
        intrinsic = int(settings.get('intrinsic',8))
        if dimensions <= intrinsic or dimensions >= 135:
            raise ValueError('Require intrinsic < PCA dimensions < 135')
        nfolds = len(set(folds.values()))
        pcas,atlases,routers = [],[],[]
        if lr_features is not None and lr_features.ndim != 2:
            raise ValueError('TRAIN LR bank features do not align with HF patches')
        lr_dim = int(settings.setdefault('lr_dim',9 if lr_features is None else lr_features.shape[-1]))
        if lr_features is not None:
            if lr_features.shape != (len(features),lr_dim) or not lr_features.isfinite().all():
                raise ValueError('TRAIN LR bank features do not align with HF patches')
        provenance = dict(split='train',rows=len(features),train_cases=sorted(folds),
            bank_sha256=hashlib.sha256(features.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            fold_fit_cases={})
        for fold in range(nfolds):
            mask = row_folds != fold
            pca = TrainPCA.fit(features[mask],dimensions)
            atlas = SCJetAtlas.fit(pca(features[mask]),centers=int(settings.get('centers',512)),
                intrinsic=intrinsic,neighbors=int(settings.get('neighbors',256)),
                normal_rank=int(settings.get('normal_rank',4)),blend_gamma=float(settings.get('blend_gamma',.5)),
                seed=int(settings.get('seed',42))+fold,condition_limit=float(settings.get('condition_limit',1.e8)))
            router = SoftAtlasRouter(lr_dim,dimensions,atlas.chart_count,
                hidden=int(settings.get('router_hidden',64)),top_m=int(settings.get('top_m',3))).to(features)
            pcas.append(pca); atlases.append(atlas); routers.append(router)
            provenance['fold_fit_cases'][str(fold)] = sorted(c for c,f in folds.items() if f != fold)
        return cls(pcas,atlases,routers,folds,settings,provenance)

    def payload(self):
        return dict(version=self.version,settings=self.settings,folds=self.folds,provenance=self.provenance,
            pcas=[{k:v.detach().cpu() for k,v in p.named_buffers()} for p in self.pcas],
            atlases=[a.payload() for a in self.atlases],
            routers=[{k:v.detach().cpu() for k,v in r.state_dict().items()} for r in self.routers])

    @classmethod
    def from_payload(cls,payload):
        if payload.get('version') != cls.version or payload.get('provenance',{}).get('split') != 'train':
            raise ValueError('Expected a TRAIN-only JetAtlas v2 snapshot')
        settings = dict(payload['settings'])
        pcas=[TrainPCA(p['mean'],p['basis']) for p in payload['pcas']]
        atlases=[SCJetAtlas(a) for a in payload['atlases']]
        routers=[]
        for atlas,pca,state in zip(atlases,pcas,payload['routers']):
            lr_dim = int(settings.setdefault('lr_dim',state['network.0.weight'].shape[1]-pca.basis.shape[1]))
            router=SoftAtlasRouter(lr_dim,pca.basis.shape[1],atlas.chart_count,
                hidden=int(settings.get('router_hidden',64)),top_m=int(settings.get('top_m',3)))
            router.load_state_dict(state)
            routers.append(router)
        return cls(pcas,atlases,routers,payload['folds'],settings,payload['provenance'])

    def diagnostics(self):
        return dict(quadratic_support_fraction=float(torch.cat([a.supported for a in self.atlases]).float().mean()),
            effective_neighbors=float(torch.cat([a.effective for a in self.atlases]).mean()),
            median_gcv_ridge=float(torch.cat([a.ridge for a in self.atlases]).median()),
            train_case_count=len(self.folds),folds=len(self.atlases),
            texture_neighbourhood='3x3 Haar HF = 12x12 HR',
            support_is_measured_not_guaranteed=True)

    @staticmethod
    def _quantiles(values, mask, fractions):
        # Gradients and energies are nonnegative, so masked values sort first.
        ordered = values.masked_fill(~mask,0).sort(dim=-1).values
        count = mask.sum(-1,keepdim=True)
        first = (values.shape[-1]-count).clamp_max(values.shape[-1]-1)
        position = (count-1).clamp_min(0) * values.new_tensor(fractions)
        lower = position.floor().long()
        upper = position.ceil().long()
        return ordered.gather(-1,first+lower).lerp(ordered.gather(-1,first+upper),position-lower)

    @classmethod
    @torch.no_grad()
    def _sample_sites(cls, target, count, reference=None):
        b,_,z,h,w = target.shape
        n = z*h*w
        if count >= n:
            return torch.arange(n,device=target.device).expand(b,-1)
        if reference is None:
            energy = target[:,1:].square().mean(1).reshape(b*z,1,h,w)
            energy = F.avg_pool2d(F.pad(energy,(1,1,1,1),mode='replicate'),3,stride=1).reshape(b,n)
            nonzero = energy > 1.e-8
            lo,hi = cls._quantiles(energy,nonzero,[.3,.9]).split(1,dim=-1)
            labels = nonzero.long() + (nonzero & (energy>=lo)).long() + (nonzero & (energy>=hi)).long()
        else:
            if reference.shape != (b,1,z,4*h,4*w):
                raise ValueError('Structural JetAtlas sampling needs aligned B1ZYX HR at four times the Haar XY grid')
            image = reference.reshape(b*z,1,4*h,4*w)
            moments = F.pad(torch.cat((image,image.square()),1),(3,3,3,3),mode='replicate')
            moments = F.avg_pool2d(F.avg_pool2d(moments,(7,1),stride=1),(1,7),stride=1)
            mean,second = moments.split(1,dim=1)
            variance = F.avg_pool2d((49/48*(second-mean.square())).clamp_min(0),4,stride=4).reshape(b,n)
            gradient = (F.pad((image[...,1:,:]-image[...,:-1,:]).abs(),(0,0,0,1))
                        + F.pad((image[...,1:]-image[...,:-1]).abs(),(0,1,0,0)))
            gradient = F.avg_pool2d(gradient,4,stride=4).reshape(b,n)
            foreground = F.avg_pool2d(image,4,stride=4).reshape(b,n) > 2/255
            edge = foreground & (gradient>0) & (gradient>=cls._quantiles(gradient,foreground,[.9]))
            labels = torch.where(edge,3,foreground.long()+(foreground & (variance>=1.e-4)).long())
        fractions = [.1,.2,.5,.2]
        limits = [math.floor(frac*count) for frac in fractions]
        remainders = sorted(range(len(limits)),key=lambda i:fractions[i]*count-limits[i],reverse=True)
        for index in remainders[:count-sum(limits)]:
            limits[index] += 1
        priorities = torch.rand(labels.shape,device=target.device)
        selected = torch.zeros_like(labels,dtype=torch.bool)
        for label,limit in enumerate(limits):
            if limit:
                scores,sites = priorities.masked_fill(labels!=label,-1).topk(limit,dim=-1)
                selected |= torch.zeros_like(selected).scatter_(1,sites,scores>=0)
        # Fill shortages from any remaining stratum without duplicate sites.
        return (priorities+selected).topk(count,dim=-1).indices

    def forward(self,prediction,target,case_ids,lr_features=None,state=None,reference=None):
        # QR/SVD/Gram solves and distance kernels require float32 under bf16 AMP.
        # Casts preserve the graph back to the image network and LR encoder.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            return self._forward(prediction.float(),target.float(),case_ids,
                None if lr_features is None else lr_features.float(),
                None if state is None else state.float(),
                None if reference is None else reference.float())

    def _forward(self,prediction,target,case_ids,lr_features=None,state=None,reference=None):
        if prediction.shape != target.shape or not case_ids or len(case_ids) != len(prediction) or any(c not in self.folds for c in case_ids):
            raise ValueError('TRAIN geometry needs paired coefficients and known cross-fit case IDs')
        if prediction.ndim != 5 or prediction.shape[1] != 16:
            raise ValueError('Expected B16ZYX orthonormal Haar coefficients')
        if state is not None and state.shape != prediction.shape:
            raise ValueError('Bridge state and prediction must align')
        if lr_features is None or lr_features.ndim != 5 or len(lr_features) != len(prediction):
            raise ValueError('Aligned LR features are required for leak-free chart routing')
        lr_grid = _lr_grid(lr_features,self.lr_dim,prediction.shape[2:])
        sites = self._sample_sites(target.detach(),self.site_budget,reference)
        b,count = sites.shape
        h,w = prediction.shape[-2:]
        valid = torch.stack((sites.remainder(w)<w-1,sites.div(w,rounding_mode='floor').remainder(h)<h-1),-1)
        neighbors = sites[...,None] + valid*sites.new_tensor([1,w])
        locations = torch.cat((sites[...,None],neighbors),-1).reshape(b,-1)
        pred_rows = _site_rows(prediction[:,1:],locations,1).reshape(b,count,3,135)
        truth_rows = _site_rows(target.detach()[:,1:],locations,1).reshape_as(pred_rows)
        state_rows = pred_rows[:,:,0] if state is None else _site_rows(state[:,1:],sites,1)
        lr_rows = _site_rows(lr_grid,sites,int(lr_grid.shape[1]!=self.lr_dim))

        groups = [[] for _ in self.atlases]
        for index,case in enumerate(case_ids):
            groups[self.folds[case]].append(index)
        values = {key:prediction.new_zeros(()) for key in ('jet_asym','jet_neighbor','jet_potential','router_kl')}
        for fold,indices in enumerate(groups):
            if indices:
                selected = torch.tensor(indices,device=prediction.device)
                fold_values = self._forward_fold(fold,pred_rows[selected],truth_rows[selected],state_rows[selected],
                                                 lr_rows[selected],valid[selected])
                for key,value in fold_values.items():
                    values[key] = values[key]+value*(len(indices)/b)
            else:
                # DDP requires every rank's absent-fold router parameters in the graph.
                values['router_kl'] = values['router_kl'] + sum(
                    parameter.reshape(-1)[0]*0 for parameter in self.routers[fold].parameters())
        values['loss']=(float(self.settings.get('asym_weight',1.))*values['jet_asym']+
            float(self.settings.get('neighbor_weight',.1))*values['jet_neighbor']+
            float(self.settings.get('potential_weight',.01))*values['jet_potential']+
            float(self.settings.get('router_weight',.01))*values['router_kl'])
        values['jet_query_sites']=prediction.new_tensor(b*count)
        return values

    def _forward_fold(self, fold, prediction, target, state, lr, valid):
        pca,atlas,router = self.pcas[fold],self.atlases[fold],self.routers[fold]
        y = pca(target[:,:,0]).flatten(0,1)
        with torch.no_grad():
            teacher = atlas.distribution(y)
            _,jacobian,form,_ = atlas._projection_jet(y,teacher.argmax(-1),gn_steps=1)
            tangent,triangle = torch.linalg.qr(jacobian,mode='reduced')
        # Anchor, right and down error use one target chart and tangent frame.
        error = ((prediction-target) @ pca.basis).flatten(0,1)
        tangent_coordinates = tangent.mT @ error.mT
        coordinates = torch.linalg.solve_triangular(triangle,tangent_coordinates,upper=True).mT
        qform = (coordinates @ form.flatten(-2)).reshape(*coordinates.shape,error.shape[-1])
        curvature = 2*(qform*coordinates[...,None]).sum(-2)
        curvature = curvature - (curvature @ tangent) @ tangent.mT
        curvature = safe_vector(curvature,float(self.settings.get('curvature_limit',1.)))
        corrected = error-.5*curvature
        normal = corrected-(corrected @ tangent) @ tangent.mT
        asym = (float(self.settings.get('normal_weight',1.))*normal[:,0].square().mean()+
            float(self.settings.get('tangent_weight',.05))*(tangent @ tangent_coordinates[...,:1]).square().mean())
        neighbor_cost = (normal[:,:1]-normal[:,1:]).square().mean(-1).reshape_as(valid)
        neighbor = ((neighbor_cost*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)).mean()

        state = pca(state).flatten(0,1)
        lr = lr.flatten(0,1)
        distillation = router.distillation_loss(lr,state.detach(),teacher)
        ids,weights = router.route(lr,state)
        expanded = state[:,None].expand(-1,ids.shape[-1],-1)
        local = atlas.query_projection_only(expanded,chart_ids=ids)
        surface = (expanded-local['projection']).square().mean(-1)
        support = F.softplus(local['support_distance']/float(self.settings.get('support_temperature',.25))).square()
        potential = (weights*(float(self.settings.get('surface_weight',1.))*surface+
                    float(self.settings.get('support_weight',.01))*support)).sum(-1).mean()
        return dict(jet_asym=asym,jet_neighbor=neighbor,jet_potential=potential,router_kl=distillation)

    def _online_delta(self,coefficients,lr_features,*,tau_eb=None,noise_variance=.001,curvature=False,stride=1):
        rows=hf_patches(coefficients,stride=stride)
        b,_,z,h,w=coefficients.shape
        if lr_features.ndim != 5 or len(lr_features) != b:
            raise ValueError('Aligned LR features are required for leak-free chart routing')
        lr=lr_router_rows(lr_features,self.lr_dim,grid_shape=(z,h,w),stride=stride)
        delta=torch.zeros_like(rows)
        chunk=int(self.settings.get('query_chunk',4096))
        maximum=float(self.settings.get('max_geometry_update',.05))
        # All deployed routers were distilled against charts excluding the teacher case.
        for pca,atlas,router in zip(self.pcas,self.atlases,self.routers):
            parts=[]
            for start in range(0,len(rows),chunk):
                x=pca(rows[start:start+chunk])
                ids,weights=router.route(lr[start:start+chunk],x)
                expanded=x[:,None].expand(-1,ids.shape[-1],-1)
                if curvature:
                    query=atlas.query_second_order(expanded,chart_ids=ids)
                    change=safe_vector(query['mean_curvature'],float(self.settings.get('curvature_limit',1.)))
                else:
                    query=atlas.query_projection_only(expanded,chart_ids=ids,return_tangent=True)
                    change=empirical_bayes_refine(expanded,query,noise_variance,tau=tau_eb)-expanded
                # Unsupported charts fade smoothly; no hard coordinate clipping.
                change=(weights[...,None]*query['confidence'][...,None]*change).sum(1)
                parts.append(pca.inverse_delta(safe_vector(change,maximum)))
            delta=delta+torch.cat(parts)/len(self.atlases)
        if stride == 1:
            return fold_hf_patches(delta,coefficients.shape)
        # Sparse curvature queries retain the original 3x3 neighbourhood. Use
        # its centre's ambient H, then interpolate the smooth drift field.
        sh,sw=math.ceil(h/stride),math.ceil(w/stride)
        centres=delta.reshape(b,z,sh,sw,15,9)[...,4].permute(0,4,1,2,3)
        field=F.interpolate(centres,size=(z,h,w),mode='trilinear',align_corners=False)
        return torch.cat((field.new_zeros(b,1,z,h,w),field),1)

    def inference_refine(self,coefficients,lr_features,tau_eb=.1,noise_variance=.001):
        """Normal-only James--Stein shrinkage. No HR or case ID is accepted."""
        if lr_features is None:
            raise ValueError('LR features are required for leak-free online chart routing')
        with torch.autocast(device_type=coefficients.device.type, enabled=False):
            return coefficients+self._online_delta(coefficients.float(),lr_features.float(),tau_eb=tau_eb,noise_variance=noise_variance)

    def curvature_drift(self,coefficients,lr_features):
        """H in coefficient units; sampler supplies .5*sigma(t)^2 multiplier."""
        if lr_features is None:
            raise ValueError('LR features are required for leak-free online chart routing')
        with torch.autocast(device_type=coefficients.device.type, enabled=False):
            return self._online_delta(coefficients.float(),lr_features.float(),curvature=True,
                stride=int(self.settings.get('curvature_stride',4)))
