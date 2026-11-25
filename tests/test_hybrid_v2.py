import pytest
import torch

from vemgsbm.hybrid import (HybridWaveletNAFNet25D, upsample_xy,
                           WindowAttention25D, CrossSliceAttention)


def small_model(**kwargs):
    return HybridWaveletNAFNet25D(widths=(4,8,12),blocks=(1,1,1),heads=2,
        refine_width=4,use_band_whitening=False,**kwargs)


@pytest.mark.parametrize("shape", [(5,32,32),(7,64,64),(3,17,19)])
def test_hybrid_full_resolution_shapes_finite_and_backward(shape):
    torch.manual_seed(21)
    model = small_model()
    lr = torch.rand(1,1,*shape)
    base = model.prepare_base(lr)
    condition = model.lr_encoder(lr)
    clean,noise = model(model.zero_state(lr,base),condition,.35)
    prediction = model.decode_prediction(clean,base,condition)
    assert prediction.shape == (1,1,shape[0],4*shape[1],4*shape[2])
    assert clean.shape == noise.shape == (1,16,*shape)
    loss = prediction.square().mean()+noise.square().mean()
    loss.backward()
    assert torch.isfinite(prediction).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.endpoint.weight.grad.norm() > 0
    assert model.refinement.output.weight.grad.norm() > 0


def test_ll_independent_of_state_and_time():
    model = small_model()
    torch.nn.init.normal_(model.ll_head.weight,std=.01)
    torch.nn.init.normal_(model.endpoint.weight,std=.01)
    lr = torch.rand(2,1,3,8,8)
    c = model.lr_encoder(lr)
    zero = model.zero_state(lr,model.prepare_base(lr))
    a = model(zero,c,0.)[0]
    b = model(torch.randn_like(zero),c,.83)[0]
    torch.testing.assert_close(a[:,:1],b[:,:1],rtol=0,atol=0)
    assert not torch.allclose(a[:,1:],b[:,1:])


def test_pixel_refinement_preserves_deterministic_ll():
    model=small_model()
    torch.nn.init.normal_(model.refinement.output.weight,std=.02)
    lr=torch.rand(1,1,3,8,8)
    base=model.prepare_base(lr);condition=model.lr_encoder(lr)
    clean=torch.randn(1,16,3,8,8)*.1
    different=torch.cat((clean[:,:1],torch.randn_like(clean[:,1:])*.1),1)
    a=model.codec.encode(model.decode_prediction(clean,base,condition)-base)
    b=model.codec.encode(model.decode_prediction(different,base,condition)-base)
    torch.testing.assert_close(a[:,:1],b[:,:1],atol=1.e-6,rtol=1.e-5)


def test_checkpoint_and_plain_gradients_agree():
    torch.manual_seed(16)
    plain,checked = small_model(),small_model(gradient_checkpointing=True)
    checked.load_state_dict(plain.state_dict())
    lr = torch.rand(1,1,3,8,8)
    losses=[]
    for model in [plain,checked]:
        c=model.lr_encoder(lr)
        b=model.prepare_base(lr)
        clean,eps=model(model.zero_state(lr,b),c,.25)
        loss=model.decode_prediction(clean,b,c).square().mean()+eps.square().mean()
        loss.backward()
        losses.append(loss)
    torch.testing.assert_close(*losses)
    for a,b in zip(plain.parameters(),checked.parameters(),strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad)


@pytest.mark.parametrize("mode",["bicubic","lanczos"])
def test_base_interpolation_constant_and_independent_z(mode):
    lr = torch.ones(1,1,3,6,7,requires_grad=True)
    base=upsample_xy(lr,mode)
    torch.testing.assert_close(base,torch.ones(1,1,3,24,28),atol=5.e-7,rtol=0)
    changed=lr.detach().clone()
    changed[:,:,0]=2
    torch.testing.assert_close(upsample_xy(changed,mode)[:,:,1:],base[:,:,1:])
    base.sum().backward()
    assert torch.isfinite(lr.grad).all() and lr.grad.min() > 0


def test_attention_connects_cross_slice_and_shifted_windows():
    torch.manual_seed(41)
    z=CrossSliceAttention(4,2)
    value=torch.randn(1,4,3,5,7,requires_grad=True)
    output=z(value)
    output[:,:,1].sum().backward()
    assert value.grad[:,:,0].abs().sum() > 0
    window=WindowAttention25D(4,2,4,True)
    odd=torch.randn(1,4,3,9,11,requires_grad=True)
    window(odd).square().mean().backward()
    assert torch.isfinite(odd.grad).all()


def test_inference_steps_no_nonfinite_with_random_heads():
    torch.manual_seed(81)
    model=small_model().eval()
    torch.nn.init.normal_(model.endpoint.weight,std=.005)
    torch.nn.init.normal_(model.epsilon.weight,std=.005)
    lr=torch.rand(1,1,3,8,8)
    base=model.prepare_base(lr)
    for steps in (1,4,8,16):
        output=model.restore(lr,base,steps)
        assert torch.isfinite(output).all()
        assert output.abs().max() < 10


def test_refinement_can_correct_sub_haar_spatial_error():
    model=small_model()
    lr=torch.rand(1,1,1,4,4)
    base=model.prepare_base(lr)
    c=model.lr_encoder(lr)
    coefficients=model.zero_state(lr,base)
    torch.nn.init.normal_(model.refinement.output.weight,std=.01)
    refined=model.decode_prediction(coefficients,base,c)
    assert (refined-base).square().mean() > 0
    refined.mean().backward()
    assert model.refinement.output.weight.grad.norm() > 0


def test_twenty_step_texture_contrast_overfit_exceeds_098_ssim():
    """Bounded optimization sanity check, not an MRI generalization claim."""
    from vemgsbm.mri_metrics import mean_xy_ssim
    torch.manual_seed(31)
    model=HybridWaveletNAFNet25D(widths=(8,12,16),blocks=(1,1,1),heads=2,
        refine_width=4,use_band_whitening=False,frequency_split=False)
    yy,xx=torch.meshgrid(torch.linspace(0,1,8),torch.linspace(0,1,8),indexing="ij")
    lr=(.5+.12*torch.sin(2*torch.pi*xx)+.09*torch.cos(2*torch.pi*yy))[None,None,None].repeat(1,1,3,1,1)
    base=model.prepare_base(lr)
    target=.5+1.5*(base-.5)
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    before=float(mean_xy_ssim(base,target))
    for _ in range(20):
        condition=model.lr_encoder(lr)
        clean,_=model(model.zero_state(lr,base),condition,0.)
        prediction=model.decode_prediction(clean,base,condition)
        loss=(prediction-target).square().mean()+.3*(1-mean_xy_ssim(prediction,target))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        condition=model.lr_encoder(lr)
        clean,_=model(model.zero_state(lr,base),condition,0.)
        after=float(mean_xy_ssim(model.decode_prediction(clean,base,condition),target))
    assert before < .94
    assert after > .98
