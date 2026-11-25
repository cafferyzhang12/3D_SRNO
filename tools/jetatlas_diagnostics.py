"""Lightweight analytic/GCV/gradient/leakage diagnostics for SC-JetAtlas."""
import argparse
import json
from pathlib import Path
import time

import torch

from vemgsbm.jetatlas import SCJetAtlas, JetAtlasRegularizer, hf_patches, smooth_bound


def run(device='cpu'):
    torch.set_num_threads(1)
    torch.manual_seed(21)
    dtype=torch.float64
    q=torch.randn(512,2,device=device,dtype=dtype)*.2
    rows=torch.cat((q,.5*q.square().sum(-1,keepdim=True)),-1)
    atlas=SCJetAtlas.fit(rows,centers=8,intrinsic=2,neighbors=128,normal_rank=1)
    ids=torch.tensor(0,device=device)
    coord=torch.tensor([.05,-.1],device=device,dtype=dtype,requires_grad=True)
    _,j,h=atlas.decode_jet(ids,coord)
    ja=torch.autograd.functional.jacobian(lambda z:atlas.decode_jet(ids,z)[0],coord)
    ha=torch.stack([torch.autograd.functional.hessian(lambda z:atlas.decode_jet(ids,z)[0][i],coord) for i in range(3)])
    query=atlas.query_second_order(rows[:8])
    eps=1.e-4
    h_fd=torch.zeros_like(query['mean_curvature'])
    for axis in range(2):
        tangent=query['tangent'][...,axis]
        jac=query['jacobian']
        v=torch.linalg.solve(jac.mT@jac,(jac.mT@tangent[...,None]))[...,0]
        ids=query['chart_ids']; coord=query['coordinates']
        f0=atlas.decode_jet(ids,coord)[0]
        sec=(atlas.decode_jet(ids,coord+eps*v)[0]-2*f0+atlas.decode_jet(ids,coord-eps*v)[0])/eps**2
        u=query['tangent']
        h_fd+=sec-(u@(u.mT@sec[...,None]))[...,0]
    boundary=torch.tensor([[100.,-100.]],dtype=dtype,device=device,requires_grad=True)
    smooth_bound(boundary,torch.ones_like(boundary)).sum().backward()
    analytic=dict(jacobian_max_error=float((j-ja).abs().max().detach()),hessian_max_error=float((h-ha).abs().max().detach()),
        mean_curvature_fd_max_error=float((h_fd-query['mean_curvature']).abs().max()),
        tangent_idempotence_max_error=float(((query['tangent']@query['tangent'].mT).matrix_power(2)-
                                            query['tangent']@query['tangent'].mT).abs().max()),
        certified_quadratic_fraction=float(atlas.supported.float().mean()),
        min_soft_boundary_derivative=float(boundary.grad.abs().min()),
        projection_mse=float((query['projection']-rows[:8]).square().mean()))
    coeff=torch.randn(5,16,1,8,8,device=device)*.1
    bank=hf_patches(coeff)
    ids=[f'TRAIN-{i}' for i in range(5) for _ in range(64)]
    start=time.perf_counter()
    regularizer=JetAtlasRegularizer.fit_from_bank(bank,ids,dict(folds=5,pca_dim=6,dim=2,centers=4,
        neighbors=64,normal_rank=2,loss_sites_per_patch=16,query_chunk=32))
    pred=(coeff[:2]+torch.randn_like(coeff[:2])*.01).requires_grad_()
    state=(.5*pred.detach()).requires_grad_()
    lr=torch.rand(2,1,1,8,8,device=device)
    terms=regularizer(pred,coeff[:2],['TRAIN-0','TRAIN-1'],lr,state)
    gradients={}
    for key in ('jet_asym','jet_neighbor','jet_potential','router_kl'):
        grad=torch.autograd.grad(terms[key],(pred,state),retain_graph=True,allow_unused=True)
        gradients[key]=[0. if g is None else float(g.norm()) for g in grad]
    terms['loss'].backward()
    with torch.no_grad():
        refined=regularizer.inference_refine(coeff[:1],lr[:1])
        drift=regularizer.curvature_drift(coeff[:1],lr[:1])
    return dict(device=device,analytic=analytic,crossfit=regularizer.diagnostics(),
        geometry_loss={k:float(v.detach()) for k,v in terms.items()},
        gradient_norms_prediction_state=gradients,
        finite_backward=bool(pred.grad.isfinite().all() and state.grad.isfinite().all()),
        online_ll_exact=bool(torch.equal(refined[:,:1],coeff[:1,:1])),
        curvature_ll_zero=bool(drift[:,:1].count_nonzero()==0),
        gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else None,
        small_bank_fit_forward_seconds=time.perf_counter()-start,
        evidence_scope='Synthetic analytic manifold and TRAIN-labelled synthetic coefficients; no MRI/VAL/TEST read')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',default='reports/mri/v2_jetatlas_diagnostics.json')
    a=p.parse_args(); result=run(a.device)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
