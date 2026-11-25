"""Support-certified local 2-jets for TRAIN-only MRI high-frequency geometry.

Legacy atlas and metric protocols are deliberately untouched. A chart is a
quadratic normal graph f(q)=c+Uq+a+Lq+q^T Q q. Geometry is analytic in chart
coordinates; a damped Gauss--Newton solve approximates the closest point.
There is no assumption that a finite atlas covers an arbitrary MRI cohort.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter

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
        support_distance = (torch.logsumexp(8*(relative.square()+1.e-12).sqrt(),-1)
                            -math.log(self.intrinsic))/8-1
        confidence = torch.sigmoid(-4*support_distance)*(.1+.9*self.supported[chart_ids].to(x))
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
        self.lr_dim = int(settings.get('lr_dim',1))
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
        lr_dim = int(settings.get('lr_dim',1))
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
        settings = payload['settings']
        pcas=[TrainPCA(p['mean'],p['basis']) for p in payload['pcas']]
        atlases=[SCJetAtlas(a) for a in payload['atlases']]
        routers=[]
        for atlas,pca,state in zip(atlases,pcas,payload['routers']):
            router=SoftAtlasRouter(int(settings.get('lr_dim',1)),pca.basis.shape[1],atlas.chart_count,
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

    def _lr_rows(self, lr_features, coefficients):
        b,_,z,h,w = coefficients.shape
        if lr_features is None:
            raise ValueError('LR features are required for leak-free online chart routing')
        if lr_features.ndim != 5 or lr_features.shape[:2] != (b,self.lr_dim):
            raise ValueError('LR router features must have B,lr_dim,Z,Y,X shape')
        if tuple(lr_features.shape[2:]) != (z,h,w):
            lr_features=F.interpolate(lr_features,size=(z,h,w),mode='trilinear',align_corners=False)
        return lr_features.permute(0,2,3,4,1).reshape(-1,self.lr_dim)

    @staticmethod
    @torch.no_grad()
    def _sample_sites(target, count, reference=None, grid_shape=None):
        n=len(target)
        if count >= n:
            return torch.arange(n,device=target.device)
        from .spatial_atlas import _stratified_sites,_structure_labels
        if reference is not None:
            labels=_structure_labels(reference,grid_shape)
        else:
            # Physical HF energy proxy is only a fallback for coefficient-only callers.
            # With HR provided, use the established physical 7x7 variance/edge strata.
            energy=target.square().mean(-1)
            labels=torch.ones(n,dtype=torch.long,device=target.device)
            labels[energy<=1.e-8]=0
            nonzero=energy>1.e-8
            if nonzero.any():
                lo,hi=torch.quantile(energy[nonzero],energy.new_tensor([.3,.9]))
                labels[(energy>=lo)&nonzero]=2
                labels[(energy>=hi)&nonzero]=3
        return _stratified_sites(labels,count,[.1,.2,.5,.2])

    def forward(self,prediction,target,case_ids,lr_features=None,state=None,reference=None):
        # QR/SVD/Gram solves and distance kernels require float32 under bf16 AMP.
        # Casts preserve the graph back to the image network and LR encoder.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            return self._forward(prediction.float(),target.float(),case_ids,
                None if lr_features is None else lr_features.float(),
                None if state is None else state.float(),
                None if reference is None else reference.float())

    def _forward(self,prediction,target,case_ids,lr_features=None,state=None,reference=None):
        if prediction.shape != target.shape or len(case_ids) != len(prediction) or any(c not in self.folds for c in case_ids):
            raise ValueError('TRAIN geometry needs paired coefficients and known cross-fit case IDs')
        state=prediction if state is None else state
        if state.shape != prediction.shape:
            raise ValueError('Bridge state and prediction must align')
        pred_rows,truth_rows,state_rows=hf_patches(prediction),hf_patches(target.detach()),hf_patches(state)
        lr_rows=self._lr_rows(lr_features,prediction)
        sites_per_item=math.prod(prediction.shape[2:])
        values={key:prediction.new_zeros(()) for key in ('jet_asym','jet_neighbor','jet_potential','router_kl')}
        query_count=0
        for b,case in enumerate(case_ids):
            fold=self.folds[case]
            pca,atlas,router=self.pcas[fold],self.atlases[fold],self.routers[fold]
            sl=slice(b*sites_per_item,(b+1)*sites_per_item)
            sites=self._sample_sites(truth_rows[sl],self.site_budget,
                None if reference is None else reference[b:b+1],prediction.shape[2:])
            x,y=pca(pred_rows[sl][sites]),pca(truth_rows[sl][sites])
            s,lr=pca(state_rows[sl][sites]),lr_rows[sl][sites]
            # Teacher geometry never enters network or online router input.
            with torch.no_grad():
                teacher=atlas.distribution(y)
                query=atlas.query_second_order(y,chart_ids=teacher.argmax(-1))
            asym,normal,_=jet_asymmetric_loss(x-y,query,
                normal_weight=float(self.settings.get('normal_weight',1.)),
                tangent_weight=float(self.settings.get('tangent_weight',.05)),
                curvature_limit=float(self.settings.get('curvature_limit',1.)))
            values['jet_asym']=values['jet_asym']+asym
            values['router_kl']=values['router_kl']+router.distillation_loss(lr,s.detach(),teacher)
            ids,weights=router.route(lr,s)
            expanded=s[:,None].expand(-1,ids.shape[-1],-1)
            local=atlas.query_second_order(expanded,chart_ids=ids)
            surf=(expanded-local['projection']).square().mean(-1)
            supp=F.softplus(local['support_distance']/float(self.settings.get('support_temperature',.25))).square()
            potential=(weights*(float(self.settings.get('surface_weight',1.))*surf+
                        float(self.settings.get('support_weight',.01))*supp)).sum(-1).mean()
            values['jet_potential']=values['jet_potential']+potential
            # Query positive XY neighbours using each anchor's fixed target geometry.
            # Both corrected normal errors share that chart/frame, preventing basis-sign artefacts.
            h,w=prediction.shape[-2:]
            right=sites[sites.remainder(w)<w-1]
            down=sites[sites.div(w,rounding_mode='floor').remainder(h)<h-1]
            anchors=torch.cat((right,down))
            neighbors=torch.cat((right+1,down+w))
            if len(anchors):
                with torch.no_grad():
                    yn=pca(truth_rows[sl][anchors])
                    nq=atlas.query_second_order(yn)
                ea=pca(pred_rows[sl][anchors])-yn
                eb=pca(pred_rows[sl][neighbors])-pca(truth_rows[sl][neighbors])
                _,na,_=jet_asymmetric_loss(ea,nq)
                _,nb,_=jet_asymmetric_loss(eb,nq)
                values['jet_neighbor']=values['jet_neighbor']+(na-nb).square().mean()
            query_count+=len(sites)
        values={key:value/len(case_ids) for key,value in values.items()}
        values['loss']=(float(self.settings.get('asym_weight',1.))*values['jet_asym']+
            float(self.settings.get('neighbor_weight',.1))*values['jet_neighbor']+
            float(self.settings.get('potential_weight',.01))*values['jet_potential']+
            float(self.settings.get('router_weight',.01))*values['router_kl'])
        values['jet_query_sites']=prediction.new_tensor(query_count)
        return values

    def _online_delta(self,coefficients,lr_features,*,tau_eb=None,noise_variance=.001,curvature=False,stride=1):
        rows=hf_patches(coefficients,stride=stride)
        b,_,z,h,w=coefficients.shape
        lr=self._lr_rows(lr_features,coefficients).reshape(b,z,h,w,self.lr_dim)
        lr=lr[:,:,::stride,::stride].reshape(-1,self.lr_dim)
        delta=torch.zeros_like(rows)
        chunk=int(self.settings.get('query_chunk',256))
        maximum=float(self.settings.get('max_geometry_update',.05))
        # All deployed routers were distilled against charts excluding the teacher case.
        for pca,atlas,router in zip(self.pcas,self.atlases,self.routers):
            parts=[]
            for start in range(0,len(rows),chunk):
                x=pca(rows[start:start+chunk])
                ids,weights=router.route(lr[start:start+chunk],x)
                expanded=x[:,None].expand(-1,ids.shape[-1],-1)
                query=atlas.query_second_order(expanded,chart_ids=ids)
                if curvature:
                    change=safe_vector(query['mean_curvature'],float(self.settings.get('curvature_limit',1.)))
                else:
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
