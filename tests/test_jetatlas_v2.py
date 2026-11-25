import inspect

import pytest
import torch

from vemgsbm.jetatlas import (SCJetAtlas, SoftAtlasRouter, case_folds,
    empirical_bayes_refine, fold_hf_patches, hf_patches, jet_asymmetric_loss,
    jet_potential, smooth_bound)


def parabola(dtype=torch.float64):
    # f(u,v)=(u,v,(u*u+v*v)/2), a unit-curvature normal graph.
    return SCJetAtlas(dict(centers=torch.zeros(1,3,dtype=dtype),
        basis=torch.eye(3,dtype=dtype)[:,:2][None],offset=torch.zeros(1,3,dtype=dtype),
        linear=torch.zeros(1,3,2,dtype=dtype),
        quadratic=torch.tensor([[[[0.,0.,.5],[0.,0.,0.]],[[0.,0.,0.],[0.,0.,.5]]]],dtype=dtype),
        radius=torch.ones(1,2,dtype=dtype)*4,fit_bandwidth=torch.ones(1,dtype=dtype),
        blend_bandwidth=torch.ones(1,dtype=dtype)*.5,effective=torch.ones(1,dtype=dtype)*100,
        supported=torch.ones(1,dtype=torch.bool),ridge=torch.ones(1,dtype=dtype)*1.e-5,
        condition=torch.ones(1,dtype=dtype)))


def test_hf_neighbourhood_and_adjoint_no_ll():
    x=torch.randn(2,16,3,4,5,requires_grad=True)
    p=hf_patches(x)
    assert p.shape==(120,135)
    restored=fold_hf_patches(p,x.shape)
    torch.testing.assert_close(restored[:,1:],x[:,1:])
    assert restored[:,:1].count_nonzero()==0
    restored.square().mean().backward()
    assert x.grad.isfinite().all() and x.grad[:,1:].norm()>0


def test_analytic_jacobian_hessian_and_fundamental_form():
    atlas=parabola()
    q=torch.tensor([.2,-.1],dtype=torch.float64,requires_grad=True)
    ids=torch.tensor(0)
    value,j,h=atlas.decode_jet(ids,q)
    jac=torch.autograd.functional.jacobian(lambda z:atlas.decode_jet(ids,z)[0],q)
    hs=torch.stack([torch.autograd.functional.hessian(lambda z:atlas.decode_jet(ids,z)[0][i],q) for i in range(3)])
    torch.testing.assert_close(j,jac,atol=1.e-10,rtol=1.e-10)
    torch.testing.assert_close(h,hs,atol=1.e-10,rtol=1.e-10)
    query=atlas.query_second_order(torch.zeros(1,3,dtype=torch.float64))
    u=query['tangent']
    pt=u@u.mT
    torch.testing.assert_close(pt@pt,pt)
    torch.testing.assert_close(query['mean_curvature'],torch.tensor([[0.,0.,2.]],dtype=torch.float64))
    b=atlas.second_fundamental(query,torch.tensor([[1.,0.,0.]],dtype=torch.float64))
    torch.testing.assert_close(b,torch.tensor([[0.,0.,1.]],dtype=torch.float64))
    # Central second difference projected normal, directional B at q=.2,-.1.
    x=atlas.decode_jet(torch.tensor([0]),q[None])[0]
    query=atlas.query_second_order(x,gn_steps=2)
    v=query['jacobian'][...,0]
    coord=query['coordinates']
    step=1.e-4
    delta=torch.tensor([[step,0.]],dtype=torch.float64)
    f0=atlas.decode_jet(torch.tensor([0]),coord)[0]
    sec=(atlas.decode_jet(torch.tensor([0]),coord+delta)[0]-2*f0+atlas.decode_jet(torch.tensor([0]),coord-delta)[0])/step**2
    normal=sec-(query['tangent']@(query['tangent'].mT@sec[...,None]))[...,0]
    torch.testing.assert_close(atlas.second_fundamental(query,v),normal,atol=1.e-7,rtol=1.e-6)


def test_soft_boundary_preserves_nonzero_gradient():
    q=torch.tensor([[100.,-100.]],dtype=torch.float64,requires_grad=True)
    smooth_bound(q,torch.ones_like(q)).sum().backward()
    assert (q.grad.abs()>0).all()
    x=torch.tensor([[30.,0.,0.]],dtype=torch.float64,requires_grad=True)
    query=parabola().query_second_order(x)
    jet_potential(x,query).backward()
    assert x.grad.isfinite().all() and x.grad.norm()>0


def test_asymmetric_weights_and_eb_preserve_tangent():
    query=parabola().query_second_order(torch.zeros(1,3,dtype=torch.float64))
    tangent=torch.tensor([[.01,0.,0.]],dtype=torch.float64)
    normal=torch.tensor([[0.,0.,.01]],dtype=torch.float64)
    assert jet_asymmetric_loss(normal,query)[0]>15*jet_asymmetric_loss(tangent,query)[0]
    x=torch.tensor([[.1,.2,.3]],dtype=torch.float64)
    refined=empirical_bayes_refine(x,query,.01,.3)
    torch.testing.assert_close(refined[:,:2],x[:,:2])
    assert refined[0,2]<x[0,2]


def test_ito_curvature_reduces_tangent_noise_departure():
    # Tangent Brownian noise on paraboloid requires +.5*sigma^2*H*dt.
    atlas=parabola()
    q=atlas.query_second_order(torch.zeros(1,3,dtype=torch.float64))
    torch.manual_seed(4)
    noise=torch.randn(10000,2,dtype=torch.float64)*.02
    uncorrected=torch.cat((noise,torch.zeros(10000,1,dtype=torch.float64)),-1)
    corrected=uncorrected+.5*.02**2*q['mean_curvature']
    ztruth=.5*noise.square().sum(-1)
    assert (corrected[:,2]-ztruth).square().mean()<.6*(uncorrected[:,2]-ztruth).square().mean()


def test_spherical_two_jet_ito_drift_reduces_radial_departure():
    # Unit sphere north-pole 2-jet: z=1-(u²+v²)/2, H=(0,0,-2).
    payload=parabola().payload()
    payload['centers'][0,2]=1.
    payload['quadratic']=-payload['quadratic']
    atlas=SCJetAtlas(payload)
    north=torch.tensor([[0.,0.,1.]],dtype=torch.float64)
    query=atlas.query_second_order(north)
    torch.testing.assert_close(query['mean_curvature'],torch.tensor([[0.,0.,-2.]],dtype=torch.float64))
    generator=torch.Generator().manual_seed(31)
    tangent=torch.randn(10000,2,dtype=torch.float64,generator=generator)*.02
    noisy=north+torch.cat((tangent,torch.zeros(10000,1,dtype=torch.float64)),-1)
    corrected=noisy+.5*.02**2*query['mean_curvature']
    radial=(noisy.norm(dim=-1)-1).square().mean()
    assert (corrected.norm(dim=-1)-1).square().mean()<.6*radial


def test_second_order_query_is_equivariant_to_orthogonal_feature_frames():
    atlas=parabola()
    generator=torch.Generator().manual_seed(92)
    rotation=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64,generator=generator)).Q
    p=atlas.payload()
    for key in ('centers','offset'):
        p[key]=p[key]@rotation.T
    for key in ('basis','linear'):
        p[key]=rotation@p[key]
    p['quadratic']=p['quadratic']@rotation.T
    rotated=SCJetAtlas(p)
    x=torch.tensor([[.2,-.1,.04]],dtype=torch.float64)
    direction=torch.tensor([[.01,.02,.03]],dtype=torch.float64)
    a=atlas.query_second_order(x,direction=direction)
    b=rotated.query_second_order(x@rotation.T,direction=direction@rotation.T)
    for key in ('projection','mean_curvature','second_fundamental'):
        torch.testing.assert_close(b[key],a[key]@rotation.T,atol=1.e-9,rtol=1.e-9)
    torch.testing.assert_close(b['tangent']@b['tangent'].mT,
        rotation@(a['tangent']@a['tangent'].mT)@rotation.T,atol=1.e-9,rtol=1.e-9)


def test_fit_gcv_support_and_no_invented_certification():
    torch.manual_seed(3)
    q=torch.randn(512,2,dtype=torch.float64)*.2
    rows=torch.cat((q,.5*q.square().sum(-1,keepdim=True)),-1)
    atlas=SCJetAtlas.fit(rows,centers=5,intrinsic=2,neighbors=128,normal_rank=1)
    assert atlas.supported.float().mean()>.95
    assert atlas.ridge.gt(0).all()
    assert torch.allclose(atlas.blend_bandwidth,.5*atlas.fit_bandwidth)
    query=atlas.query_second_order(rows[:8])
    assert all(v.isfinite().all() for v in query.values() if v.is_floating_point())
    assert (query['projection']-rows[:8]).square().mean()<1.e-4
    degenerate=SCJetAtlas.fit(torch.zeros(50,3),centers=2,intrinsic=2)
    assert not degenerate.supported.any()


def test_router_signature_excludes_hr_and_teacher_is_detached():
    router=SoftAtlasRouter(4,3,6)
    assert 'hr' not in inspect.signature(router.forward).parameters
    lr=torch.randn(7,4,requires_grad=True)
    state=torch.randn(7,3,requires_grad=True)
    teacher=torch.randn(7,6).softmax(-1).requires_grad_(True)
    router.distillation_loss(lr,state,teacher).backward()
    assert teacher.grad is None and lr.grad.norm()>0 and state.grad.norm()>0
    ids,prob=router.route(lr,state)
    assert ids.shape==(7,3)
    torch.testing.assert_close(prob.sum(-1),torch.ones(7))


def test_folds_are_at_case_level():
    ids=[f'case{i//3}' for i in range(30)]
    mapping=case_folds(ids)
    assert len(set(mapping.values()))==5
    assert mapping==case_folds(list(reversed(ids)))
    with pytest.raises(ValueError):case_folds(['a','b'])

@pytest.fixture(scope='module')
def regularizer():
    from vemgsbm.jetatlas import JetAtlasRegularizer
    torch.manual_seed(18)
    coeff=torch.randn(5,16,1,8,8)*.1
    rows=hf_patches(coeff)
    ids=[f'c{i}' for i in range(5) for _ in range(64)]
    settings=dict(folds=5,projection_dim=6,intrinsic=2,centers=4,neighbors=64,
                  normal_rank=2,query_sites=16,query_chunk=32,lr_dim=1)
    return JetAtlasRegularizer.fit_from_bank(rows,ids,settings),coeff,rows,ids


def test_crossfit_pca_and_geometry_do_not_see_teacher_case(regularizer):
    model,_,rows,ids=regularizer
    for fold,pca in enumerate(model.pcas):
        mask=torch.tensor([model.folds[c]!=fold for c in ids])
        torch.testing.assert_close(pca.mean,rows[mask].mean(0))
        held=[c for c,f in model.folds.items() if f==fold]
        assert not set(held)&set(model.provenance['fold_fit_cases'][str(fold)])
    from vemgsbm.jetatlas import JetAtlasRegularizer
    with pytest.raises(ValueError):JetAtlasRegularizer.fit_from_bank(rows,ids,model.settings,split='val')


def test_regularizer_backward_and_state_potential(regularizer):
    model,coeff,_,_=regularizer
    pred=(coeff[:2]+.01*torch.randn_like(coeff[:2])).requires_grad_()
    state=(.5*pred.detach()).requires_grad_()
    lr=torch.rand(2,1,1,8,8)
    terms=model(pred,coeff[:2],['c0','c1'],lr,state)
    assert terms['jet_query_sites']==32
    assert all(torch.isfinite(v).all() for v in terms.values())
    terms['loss'].backward()
    assert pred.grad.isfinite().all() and pred.grad.norm()>0
    assert state.grad.isfinite().all() and state.grad.norm()>0
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.routers.parameters())


def test_regularizer_online_is_hr_free_preserves_ll_and_roundtrips(regularizer):
    from vemgsbm.jetatlas import JetAtlasRegularizer
    model,coeff,_,_=regularizer
    clone=JetAtlasRegularizer.from_payload(model.payload())
    lr=torch.rand(1,1,1,8,8)
    with torch.no_grad():
        x=model.inference_refine(coeff[:1],lr)
        y=clone.inference_refine(coeff[:1],lr)
        h=clone.curvature_drift(coeff[:1],lr)
    torch.testing.assert_close(x,y)
    torch.testing.assert_close(x[:,:1],coeff[:1,:1])
    assert h[:,:1].count_nonzero()==0 and h.isfinite().all()
    assert 'target' not in inspect.signature(model.inference_refine).parameters
    assert 'hr' not in inspect.signature(model.inference_refine).parameters
    with pytest.raises(ValueError):model.inference_refine(coeff[:1],None)


def test_jet_geometry_is_orthogonally_equivariant():
    atlas=parabola()
    rotation=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64)).Q
    payload=atlas.payload()
    for key in ('centers','offset'):
        payload[key]=payload[key]@rotation.T
    for key in ('basis','linear'):
        payload[key]=rotation@payload[key]
    payload['quadratic']=torch.einsum('ab,kijb->kija',rotation,payload['quadratic'])
    rotated=SCJetAtlas(payload)
    x=torch.tensor([[.3,-.1,.08]],dtype=torch.float64)
    first=atlas.query_second_order(x)
    second=rotated.query_second_order(x@rotation.T)
    torch.testing.assert_close(second['projection'],first['projection']@rotation.T,atol=1.e-10,rtol=1.e-10)
    torch.testing.assert_close(second['mean_curvature'],first['mean_curvature']@rotation.T,atol=1.e-10,rtol=1.e-10)
    torch.testing.assert_close(second['tangent']@second['tangent'].mT,
        rotation@(first['tangent']@first['tangent'].mT)@rotation.T,atol=1.e-10,rtol=1.e-10)


def test_regularizer_runs_under_bfloat16_autocast(regularizer):
    model,coeff,_,_=regularizer
    pred=coeff[:1].clone().requires_grad_()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        terms=model(pred,coeff[:1],['c0'],torch.rand(1,1,1,8,8))
    terms['loss'].backward()
    assert pred.grad.isfinite().all()


def test_mutating_excluded_case_does_not_change_its_pca_or_charts(regularizer):
    from vemgsbm.jetatlas import JetAtlasRegularizer
    model,_,rows,ids=regularizer
    changed=rows.clone()
    changed[torch.tensor([c=='c0' for c in ids])]+=1000
    refit=JetAtlasRegularizer.fit_from_bank(changed,ids,model.settings)
    fold=model.folds['c0']
    for key,value in model.pcas[fold].named_buffers():
        torch.testing.assert_close(getattr(refit.pcas[fold],key),value)
    for key,value in model.atlases[fold].named_buffers():
        torch.testing.assert_close(getattr(refit.atlases[fold],key),value)


def test_sparse_curvature_preserves_original_neighborhood(regularizer):
    model,coeff,_,_=regularizer
    dense=hf_patches(coeff[:1]).reshape(1,1,8,8,135)
    sparse=hf_patches(coeff[:1],stride=4).reshape(1,1,2,2,135)
    torch.testing.assert_close(sparse,dense[:,:,::4,::4])
    with torch.no_grad():
        result=model.curvature_drift(coeff[:1],torch.rand(1,1,1,8,8))
    assert result.shape==coeff[:1].shape and result.isfinite().all()
    assert result[:,:1].count_nonzero()==0


def test_sparse_curvature_constant_field_remains_constant():
    # All-zero PCA basis/constant normal H gives a spatially constant query field.
    from vemgsbm.jetatlas import TrainPCA,JetAtlasRegularizer
    atlas=parabola(torch.float32)
    basis=torch.zeros(135,3);basis[4,0]=1;basis[13,1]=1;basis[22,2]=1
    pca=TrainPCA(torch.zeros(135),basis)
    router=SoftAtlasRouter(1,3,1)
    model=JetAtlasRegularizer([pca],[atlas],[router],{'a':0},dict(curvature_stride=4))
    x=torch.zeros(1,16,3,9,11)
    result=model.curvature_drift(x,torch.zeros(1,1,3,9,11))
    assert result.shape==x.shape and result.isfinite().all()
    torch.testing.assert_close(result,result[:,:,:,:1,:1].expand_as(result))
    assert result[:,:1].count_nonzero()==0 and result[:,1:].norm()>0
