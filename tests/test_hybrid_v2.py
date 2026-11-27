import pytest
import torch

from vemgsbm.hybrid import (HybridWaveletNAFNet25D, upsample_xy,
                           WindowAttention25D, CrossSliceAttention,
                           NAFBlock25D, PixelRefinement)


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
    model=small_model(sigma_bridge=.08)
    torch.nn.init.normal_(model.refinement.output.weight,std=.02)
    lr=torch.rand(1,1,3,8,8)
    base=model.prepare_base(lr);condition=model.lr_encoder(lr)
    clean=torch.randn(1,16,3,8,8)*.1
    different=torch.cat((clean[:,:1],torch.randn_like(clean[:,1:])*.1),1)
    a=model.codec.encode(model.decode_prediction(clean,base,condition)-base)
    b=model.codec.encode(model.decode_prediction(different,base,condition)-base)
    torch.testing.assert_close(a[:,:1],b[:,:1],atol=1.e-6,rtol=1.e-5)


@pytest.mark.parametrize('checkpointing', [True, 'refinement_only'])
def test_checkpoint_and_plain_gradients_agree(checkpointing, monkeypatch):
    import vemgsbm.hybrid as hybrid

    torch.manual_seed(16)
    plain = small_model(gradient_checkpointing=False)
    checked = small_model(gradient_checkpointing=checkpointing)
    checked.load_state_dict(plain.state_dict())
    calls = []
    original = hybrid.checkpoint

    def counted(module, *args, **kwargs):
        calls.append(module)
        return original(module, *args, **kwargs)

    monkeypatch.setattr(hybrid, 'checkpoint', counted)
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
    assert checked.refinement in calls
    if checkpointing == 'refinement_only':
        assert calls == [checked.refinement]
    else:
        assert checked.first in calls and checked.lr_encoder.layers[0] in calls


@pytest.mark.parametrize('sigma', [0., .08])
def test_checkpoint_refinement_batch_chunks_preserve_outputs_and_gradients(sigma):
    plain = small_model(sigma_bridge=sigma, refinement_chunk=None)
    chunked = small_model(sigma_bridge=sigma, refinement_chunk=2)
    torch.nn.init.normal_(plain.refinement.output.weight, std=.05)
    chunked.load_state_dict(plain.state_dict())
    batch_sizes = []
    chunked.refinement.register_forward_pre_hook(lambda module, args: batch_sizes.append(len(args[0])))
    coefficients = torch.rand(3, 16, 3, 4, 4)
    base = torch.rand(3, 1, 3, 16, 16)
    low = torch.rand(3, 4, 3, 4, 4)
    a = [value.clone().requires_grad_() for value in (coefficients, base, low)]
    b = [value.clone().requires_grad_() for value in (coefficients, base, low)]
    out_full = plain.decode_prediction(a[0], a[1], [a[2]])
    out_chunked = chunked.decode_prediction(b[0], b[1], [b[2]])
    assert batch_sizes == [2, 1]
    torch.testing.assert_close(out_chunked, out_full)
    out_full.square().mean().backward()
    out_chunked.square().mean().backward()
    assert batch_sizes == [2, 1, 1, 2]
    for first, second in zip(a, b, strict=True):
        torch.testing.assert_close(first.grad, second.grad)
    for first, second in zip(plain.refinement.parameters(), chunked.refinement.parameters(), strict=True):
        torch.testing.assert_close(first.grad, second.grad)


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


def test_cross_slice_attention_bias_distinguishes_near_and_far():
    layer = CrossSliceAttention(2, 1)
    # Only the value projection uses the token. Learned relative offsets then
    # determine which slices contribute to the output for a given query.
    with torch.no_grad():
        layer.qkv.weight.zero_()
        layer.qkv.bias.zero_()
        layer.qkv.weight[-2:] = torch.eye(2)
        layer.projection.weight.copy_(torch.eye(2))
        layer.projection.bias.zero_()
        layer.relative_bias[:, layer.max_offset+1] = 5
    value = torch.zeros(1, 2, 5, 1, 1)
    nearby, remote = value.clone(), value.clone()
    nearby[0, 0, 1] = 1
    remote[0, 0, 3] = 1
    result = layer(nearby)
    assert result[0, 0, 0].item() > layer(remote)[0, 0, 0].item()
    result[0, 0, 0].backward()
    assert layer.relative_bias.grad.abs().sum() > 0
    assert layer(torch.zeros(1, 2, 21, 1, 1)).shape[-3] == 21


def test_naf_has_zero_initialized_cross_z_residual():
    block = NAFBlock25D(2)
    x = torch.rand(1, 2, 3, 3, 3)
    torch.testing.assert_close(block(x), x, atol=0, rtol=0)
    block(x)[:, :, 1].square().mean().backward()
    assert block.cross_z.weight.grad.abs().sum() > 0
    with torch.no_grad():
        block.cross_z.weight[:, 0, 0] = 1
    x.requires_grad_()
    block(x)[:, 0, 1].sum().backward()
    assert x.grad[:, :, 0].abs().sum() > 0
    assert x.grad[:, :, 2].count_nonzero() == 0


def test_refinement_residual_backpropagates_through_haar_reconstruction():
    module = PixelRefinement(4, 4)
    torch.nn.init.normal_(module.output.weight, std=.05)
    reconstruction = torch.rand(1, 1, 3, 16, 16, requires_grad=True)
    lr_features = torch.rand(1, 4, 3, 4, 4)
    base = torch.rand_like(reconstruction)
    correction = module(reconstruction, base, lr_features)-reconstruction
    correction.square().mean().backward()
    assert torch.isfinite(reconstruction.grad).all()
    assert reconstruction.grad.abs().sum() > 0


def test_pixel_refinement_preserves_channel_last_inputs_to_hr_merge():
    original = PixelRefinement(4, 4)
    torch.nn.init.normal_(original.output.weight, std=.05)
    channel_last = PixelRefinement(4, 4)
    channel_last.load_state_dict(original.state_dict())
    torch.nn.utils.convert_conv2d_weight_memory_format(channel_last, torch.channels_last)
    torch.nn.utils.convert_conv3d_weight_memory_format(channel_last, torch.channels_last_3d)
    layout = []
    channel_last.merge.register_forward_pre_hook(
        lambda module, args: layout.append(args[0].is_contiguous(memory_format=torch.channels_last_3d)))
    lr = torch.rand(2, 4, 3, 4, 4)
    reconstruction, base = torch.rand(2, 1, 3, 16, 16), torch.rand(2, 1, 3, 16, 16)
    result = channel_last(reconstruction, base, lr.contiguous(memory_format=torch.channels_last_3d))
    torch.testing.assert_close(result, original(reconstruction, base, lr))
    assert layout == [True]


def test_historical_model_without_z_weights_loads_and_retains_original_conditioning():
    model = small_model()
    old = {key: value for key, value in model.state_dict().items()
           if '.cross_z.' not in key and not key.endswith('.relative_bias') and key != 'legacy_refinement'}
    model.load_state_dict(old, strict=True)
    clone = small_model()
    clone.load_state_dict(model.state_dict(), strict=True)
    assert clone._legacy_refinement
    for key, value in clone.state_dict().items():
        if key not in old and key != 'legacy_refinement':
            assert value.count_nonzero() == 0


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
