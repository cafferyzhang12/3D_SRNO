"""2.5D NAF/window/Z attention super-resolution with pixel-domain refinement."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .models import batch_time
from .momentum.model import HaarXY


def batched_attention(query, key, value, max_batch=16384):
    """SDPA over independent sequences without exceeding CUDA launch grids.

    Cross-Z attention expands B*Y*X into SDPA's batch dimension. Fused CUDA
    backends in the deployed PyTorch 2.8 fail for large batches (e.g. 131072).
    Splitting only that dimension preserves every token, head and gradient;
    attention within each independent window/Z sequence remains unchanged.
    """
    if type(max_batch) is not int or max_batch < 1:
        raise ValueError('Attention max_batch must be a positive integer')
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError('Expected attention tensors shaped batch,heads,tokens,features')
    if not (query.shape[0] == key.shape[0] == value.shape[0]) or query.shape[0] < 1:
        raise ValueError('Attention tensors require aligned nonempty sequence batches')
    if query.shape[0] <= max_batch:
        return F.scaled_dot_product_attention(query,key,value)
    return torch.cat([F.scaled_dot_product_attention(query[i:i+max_batch],
        key[i:i+max_batch],value[i:i+max_batch])
        for i in range(0,query.shape[0],max_batch)],dim=0)


def _lanczos_axis(value, output, axis, radius=3):
    """Separable Lanczos with pixel-centre alignment and replicated boundaries."""
    size = value.shape[axis]
    coordinate = (torch.arange(output, device=value.device, dtype=torch.float32)+.5)*size/output-.5
    offsets = torch.arange(-radius+1, radius+1, device=value.device)
    ids = coordinate.floor().long()[:, None]+offsets
    distance = coordinate[:, None]-ids
    weight = torch.sinc(distance)*torch.sinc(distance/radius)
    weight = weight*(distance.abs() < radius)
    weight = (weight/weight.sum(-1, keepdim=True)).to(value)
    work = value.movedim(axis, -1)
    result = (work[..., ids.clamp(0, size-1)]*weight).sum(-1)
    return result.movedim(-1, axis)


def upsample_xy(lr, mode="bicubic", scale=4):
    """Upsample XY only; never interpolate across slice positions."""
    if lr.ndim != 5 or lr.shape[1] != 1 or mode not in {"bicubic", "lanczos", "bilinear"}:
        raise ValueError("Expected B1ZYX and bicubic/lanczos/bilinear interpolation")
    if type(scale) is not int or scale < 1:
        raise ValueError("Interpolation scale must be a positive integer")
    if mode == "lanczos":
        return _lanczos_axis(_lanczos_axis(lr, lr.shape[-1]*scale, -1), lr.shape[-2]*scale, -2)
    b, _, z, h, w = lr.shape
    planes = lr.transpose(1,2).reshape(b*z, 1, h, w)
    result = F.interpolate(planes, scale_factor=scale, mode=mode, align_corners=False)
    return result.reshape(b, z, 1, h*scale, w*scale).transpose(1,2)


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

    def forward(self, x):
        variance, mean = torch.var_mean(x, dim=1, keepdim=True, unbiased=False)
        return (x-mean)*torch.rsqrt(variance+1.e-6)*self.weight+self.bias


class NAFBlock25D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1, self.norm2 = ChannelNorm(channels), ChannelNorm(channels)
        self.expand = nn.Conv3d(channels, 2*channels, 1)
        self.depthwise = nn.Conv3d(2*channels, 2*channels, (1,3,3), padding=(0,1,1), groups=2*channels)
        self.attention = nn.Conv3d(channels, channels, 1)
        self.project = nn.Conv3d(channels, channels, 1)
        self.ffn_in = nn.Conv3d(channels, 2*channels, 1)
        self.ffn_out = nn.Conv3d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1,channels,1,1,1))
        self.gamma = nn.Parameter(torch.zeros(1,channels,1,1,1))

    def forward(self, x):
        a,b = self.depthwise(self.expand(self.norm1(x))).chunk(2,1)
        gate = a*b
        hidden = x+self.beta*self.project(gate*self.attention(gate.mean((-2,-1),keepdim=True)))
        a,b = self.ffn_in(self.norm2(hidden)).chunk(2,1)
        return hidden+self.gamma*self.ffn_out(a*b)


class WindowAttention25D(nn.Module):
    """Bounded memory window attention, including odd XY sizes."""
    def __init__(self, channels, heads=4, window_size=8, shifted=False):
        super().__init__()
        self.heads = math.gcd(channels, heads)
        self.window_size, self.shifted = window_size, shifted
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels,3*channels)
        self.projection = nn.Linear(channels,channels)
        self.scale = nn.Parameter(torch.tensor(.1))

    def forward(self, x):
        b,c,z,h,w = x.shape
        size = self.window_size
        # Replicate padding avoids introducing artificial dark window borders.
        planes = x.permute(0,2,1,3,4).reshape(b*z,c,h,w)
        shift = size//2 if self.shifted else 0
        # Padding instead of cyclic roll prevents distant image edges mixing.
        top,left = shift,shift
        bottom,right = (-(h+top))%size, (-(w+left))%size
        planes = F.pad(planes,(left,right,top,bottom),mode="replicate")
        hp,wp = planes.shape[-2:]
        tokens = planes.reshape(b*z,c,hp//size,size,wp//size,size).permute(0,2,4,3,5,1)
        tokens = tokens.reshape(-1,size*size,c)
        qkv = self.qkv(self.norm(tokens)).reshape(-1,size*size,3,self.heads,c//self.heads)
        q,k,v = qkv.permute(2,0,3,1,4).unbind(0)
        attended = batched_attention(q,k,v).transpose(1,2).reshape_as(tokens)
        attended = self.projection(attended).reshape(b*z,hp//size,wp//size,size,size,c)
        attended = attended.permute(0,5,1,3,2,4).reshape(b*z,c,hp,wp)
        attended = attended[...,top:top+h,left:left+w].reshape(b,z,c,h,w).permute(0,2,1,3,4)
        return x+self.scale*attended


class CrossSliceAttention(nn.Module):
    """Attention across Z at each spatial site, without Z resampling."""
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = math.gcd(channels,heads)
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels,3*channels)
        self.projection = nn.Linear(channels,channels)
        self.scale = nn.Parameter(torch.tensor(.1))

    def forward(self,x):
        b,c,z,h,w = x.shape
        tokens = x.permute(0,3,4,2,1).reshape(b*h*w,z,c)
        qkv = self.qkv(self.norm(tokens)).reshape(b*h*w,z,3,self.heads,c//self.heads)
        q,k,v = qkv.permute(2,0,3,1,4).unbind(0)
        attended = batched_attention(q,k,v).transpose(1,2).reshape_as(tokens)
        attended = self.projection(attended).reshape(b,h,w,z,c).permute(0,4,3,1,2)
        return x+self.scale*attended


class HybridConditionEncoder(nn.Module):
    def __init__(self, widths, blocks, heads, window_size, gradient_checkpointing):
        super().__init__()
        self.checkpointing = gradient_checkpointing
        self.layers = nn.ModuleList()
        inputs = 1
        for index,(width,count) in enumerate(zip(widths,blocks,strict=True)):
            self.layers.append(nn.Sequential(
                nn.Conv3d(inputs,width,(1,3,3),stride=(1,1 if index == 0 else 2,1 if index == 0 else 2),padding=(0,1,1)),
                *[NAFBlock25D(width) for _ in range(count)],
                WindowAttention25D(width,heads,window_size,shifted=index%2 == 1),
                CrossSliceAttention(width,heads)))
            inputs = width

    def forward(self,lr,*,descriptor_only=False):
        if lr.ndim != 5 or lr.shape[1] != 1:
            raise ValueError("Expected B1ZYX native LR")
        levels = []
        for layer in self.layers:
            lr = (checkpoint(layer,lr,use_reentrant=False)
                  if self.checkpointing and self.training and torch.is_grad_enabled() else layer(lr))
            levels.append(lr)
            if descriptor_only:
                return lr
        return levels


class PixelRefinement(nn.Module):
    """PixelShuffle LR→2LR→HR followed by HR-grid NAF residual refinement."""
    def __init__(self, input_width, width):
        super().__init__()
        self.first = nn.Conv2d(input_width,4*width,3,padding=1)
        self.second = nn.Conv2d(width,4*width,3,padding=1)
        self.merge = nn.Conv3d(width+2,width,(1,3,3),padding=(0,1,1))
        self.naf = nn.Sequential(NAFBlock25D(width),NAFBlock25D(width))
        self.output = nn.Conv3d(width,1,(1,3,3),padding=(0,1,1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,reconstruction,base,low,deterministic_reference=None):
        b,c,z,h,w = low.shape
        planes = low.permute(0,2,1,3,4).reshape(b*z,c,h,w)
        planes = F.gelu(F.pixel_shuffle(self.first(planes),2))
        planes = F.gelu(F.pixel_shuffle(self.second(planes),2))
        features = planes.reshape(b,z,-1,4*h,4*w).permute(0,2,1,3,4)
        reference = base if deterministic_reference is None else deterministic_reference
        hidden = self.merge(torch.cat((features,reference,base),1))
        return reconstruction+self.output(self.naf(hidden))


class HybridWaveletNAFNet25D(nn.Module):
    """Haar endpoint/epsilon network with an independent deterministic LL head."""
    predict_target = "x1"

    def __init__(self, widths=(48,96,128), blocks=(2,2,4), window_size=8, heads=4,
                 refine_width=16, sigma_bridge=.08, use_band_whitening=True,
                 frequency_split=True, sampler_type="ode", solver="heun", temperature=.1,
                 gradient_checkpointing=False, base_interpolation="bicubic"):
        super().__init__()
        if (len(widths) != 3 or len(blocks) != 3 or min(widths) < 2 or min(blocks) < 1
                or window_size < 1 or heads < 1 or refine_width < 2):
            raise ValueError("Invalid Hybrid NAF widths, blocks, or attention sizes")
        if base_interpolation not in {"bicubic","lanczos","bilinear"}:
            raise ValueError("Unknown base interpolation")
        if not math.isfinite(sigma_bridge) or sigma_bridge < 0:
            raise ValueError("Invalid bridge sigma")
        self.codec = HaarXY()
        self.mode,self.readout,self.spatial = "haar_bridge","endpoint",None
        self.sigma_0 = 0.
        self.sigma_bridge = sigma_bridge if frequency_split else 0.
        self.use_band_whitening,self.frequency_split = use_band_whitening,frequency_split
        self.sampler_type,self.solver,self.temperature = sampler_type,solver,temperature
        self.base_interpolation = base_interpolation
        self.gradient_checkpointing = gradient_checkpointing
        self.eb_tau = .1
        self.register_buffer("band_mean",torch.zeros(16))
        self.register_buffer("band_std",torch.ones(16))
        self.register_buffer("band_stats_ready",torch.tensor(not use_band_whitening))
        self.lr_encoder = HybridConditionEncoder(widths,blocks,heads,window_size,gradient_checkpointing)
        a,b,c = widths
        self.state = nn.Conv3d(16,a,(1,3,3),padding=(0,1,1))
        self.time = nn.Sequential(nn.Linear(4,a),nn.SiLU(),nn.Linear(a,a))
        self.first = nn.Sequential(NAFBlock25D(a),WindowAttention25D(a,heads,window_size,True))
        self.down1 = nn.Conv3d(a,b,(1,3,3),stride=(1,2,2),padding=(0,1,1))
        self.second = nn.Sequential(NAFBlock25D(b),CrossSliceAttention(b,heads))
        self.down2 = nn.Conv3d(b,c,(1,3,3),stride=(1,2,2),padding=(0,1,1))
        self.deep = nn.Sequential(NAFBlock25D(c),WindowAttention25D(c,heads,window_size),CrossSliceAttention(c,heads))
        self.up2 = nn.Sequential(nn.Conv3d(c+b,b,1),NAFBlock25D(b))
        self.up1 = nn.Sequential(nn.Conv3d(b+a,a,1),NAFBlock25D(a))
        self.ll_head = nn.Conv3d(a,1,(1,3,3),padding=(0,1,1))
        self.endpoint = nn.Conv3d(a,15,(1,3,3),padding=(0,1,1))
        self.epsilon = nn.Conv3d(a,15,(1,3,3),padding=(0,1,1))
        self.refinement = PixelRefinement(a,refine_width)
        for head in (self.ll_head,self.endpoint,self.epsilon):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _run(self,layer,x):
        return (checkpoint(layer,x,use_reentrant=False)
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled() else layer(x))

    def forward(self,state,condition,time,direction=0,bound=None):
        if state.ndim != 5 or state.shape[1] != 16:
            raise ValueError("Expected B16ZYX Haar bridge state")
        low,middle,deep = condition
        t = batch_time(time,state)[:,None]
        embedding = self.time(torch.cat((t,t.square(),(math.pi*t).sin(),(math.pi*t).cos()),1))
        a = self._run(self.first,self.state(state)+low+embedding[:,:,None,None,None])
        b = self._run(self.second,self.down1(a)+middle)
        c = self._run(self.deep,self.down2(b)+deep)
        def align(x,skip):
            return torch.cat((F.interpolate(x,size=skip.shape[-3:],mode="trilinear",align_corners=False),skip),1)
        hidden = self._run(self.up1,align(self._run(self.up2,align(c,b)),a))
        # The LL head has no access to the state or time, including in SDE mode.
        ll = self.ll_head(low)
        clean = torch.cat((ll,self.endpoint(hidden)),1)
        noise = torch.cat((torch.zeros_like(ll),self.epsilon(hidden)),1)
        return clean,noise

    def zero_state(self,lr,base):
        if lr.shape != (*base.shape[:3],base.shape[-2]//4,base.shape[-1]//4):
            raise ValueError("Native LR must align with the XY x4 base")
        return lr.new_zeros(len(lr),16,*lr.shape[2:])

    def initial_state(self,lr,base,generator=None):
        return self.zero_state(lr,base)

    def prepare_base(self,lr):
        return upsample_xy(lr,self.base_interpolation)

    @torch.no_grad()
    def set_band_statistics(self,payload):
        from .mri_data import _haar_statistics
        mean,std = _haar_statistics(self.band_mean.new_zeros(1,16,1,1,1),payload["mean"],payload["std"])
        self.band_mean.copy_(mean.flatten())
        self.band_std.copy_(std.flatten())
        self.band_stats_ready.fill_(True)

    def normalize_haar(self,value):
        if not bool(self.band_stats_ready):
            raise ValueError("TRAIN Haar statistics must be fitted before use")
        return ((value-self.band_mean[None,:,None,None,None])/(self.band_std[None,:,None,None,None]+1.e-6)
                if self.use_band_whitening else value)

    def denormalize_haar(self,value):
        if not bool(self.band_stats_ready):
            raise ValueError("TRAIN Haar statistics must be fitted before use")
        return (value*(self.band_std[None,:,None,None,None]+1.e-6)+self.band_mean[None,:,None,None,None]
                if self.use_band_whitening else value)

    def decode_prediction(self,coefficients,base,condition):
        physical = self.denormalize_haar(coefficients)
        reconstruction = base+self.codec.decode(physical)
        # The pixel head conditions on LR and deterministic LL, so its low-band
        # correction cannot depend on stochastic HF. It still refines every HR pixel.
        ll_reference = base+self.codec.decode(torch.cat((physical[:,:1],torch.zeros_like(physical[:,1:])),1))
        # Checkpoint the HR head, the main memory hotspot for 7x256x256 patches.
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            refined = checkpoint(self.refinement,reconstruction,base,condition[0],ll_reference,use_reentrant=False)
        else:
            refined = self.refinement(reconstruction,base,condition[0],ll_reference)
        return refined

    @torch.no_grad()
    def restore(self,lr,base,steps=8,sigma=None,generator=None,atlas=None,noise_steps=None,
                sampler_type=None,solver=None,initial_state=None):
        from .frequency_bridge import integrate_frequency_bridge
        base = self.prepare_base(lr)
        state = self.initial_state(lr,base,generator) if initial_state is None else initial_state
        if state.shape != self.zero_state(lr,base).shape:
            raise ValueError("Initial state shape differs from native LR grid")
        condition = self.lr_encoder(lr)
        if self.frequency_split:
            curvature = (lambda x,t:atlas.curvature_drift(x,lr)) if atlas is not None else None
            shrink = (lambda x,t:atlas.inference_refine(x,lr,tau_eb=self.eb_tau)
                      if t >= getattr(self,'online_geometry_start',1.) else x) if atlas is not None else None
            residual = integrate_frequency_bridge(lambda x,t:self(x,condition,t),state,steps,
                sigma_max=self.sigma_bridge,sampler_type=sampler_type or self.sampler_type,
                solver=solver or self.solver,temperature=self.temperature,generator=generator,
                noise_steps=noise_steps,curvature=curvature,shrink=shrink)
        else:
            residual,_ = self(state,condition,0.)
        return self.decode_prediction(residual,base,condition)
