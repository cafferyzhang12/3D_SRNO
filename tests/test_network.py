"""Neural bidirectional drift and SDE inference contracts."""
import inspect

import pytest
import torch
from torch import nn

from vemgsbm.bridge import reconstruct
from vemgsbm.network import DriftNetwork, integrate


@pytest.fixture(autouse=True)
def small_cpu():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_both_direction_embeddings_receive_distinct_finite_gradients():
    torch.manual_seed(6)
    model=DriftNetwork(width=2)
    # The intended zero residual initialization gives zero body gradient on the
    # very first update. Activate the projection to test subsequent updates.
    with torch.no_grad():
        model.backbone.output_projection.weight.normal_(std=.03)
    state=torch.rand(2,1,3,8,8)
    base=torch.rand_like(state)
    mask=torch.zeros_like(state)
    forward=model(state,base,mask,.4,'forward')
    backward=model(state,base,mask,.4,'backward')
    assert not torch.allclose(forward,backward)
    loss=(forward-.7).square().mean()+(backward+.3).square().mean()
    loss.backward()
    embedding_grad=model.backbone.direction_embedding.weight.grad
    assert embedding_grad is not None and torch.isfinite(embedding_grad).all()
    assert (embedding_grad.abs().sum(1)>0).all()
    assert model.backbone.input_projection.weight.grad.abs().sum()>0


class ConstantDrift(nn.Module):
    def __init__(self,value=.2):
        super().__init__()
        self.value=value
        self.seen=[]

    def forward(self,state,base,mask,time,direction):
        self.seen.append((time,direction,state.detach().clone()))
        return torch.full_like(state,self.value)*(1-mask)


@pytest.mark.parametrize('direction', ['forward','backward'])
def test_integrator_uses_correct_clock_without_second_backward_sign(direction):
    model=ConstantDrift()
    source=torch.zeros(1,1,3,4,4)
    out=integrate(model,source,source,source,steps=4,sigma=0,direction=direction)
    torch.testing.assert_close(out,source+.2)
    expected=[.02,.25,.5,.75] if direction=='forward' else [.98,.75,.5,.25]
    assert [time for time,_,_ in model.seen]==expected
    assert all(d==direction for _,d,_ in model.seen)


def test_exact_observations_are_preserved_before_and_after_every_step():
    model=ConstantDrift(.8)
    initial=torch.zeros(2,1,3,4,4)
    base=torch.ones_like(initial)*.3
    mask=torch.zeros_like(initial)
    mask[...,0]=1
    out=integrate(model,initial,base,mask,steps=4,sigma=.2,
                  generator=torch.Generator().manual_seed(2))
    assert torch.equal(out*mask,base*mask)
    for _,_,seen in model.seen:
        assert torch.equal(seen*mask,base*mask)


def test_no_hr_argument_and_reproducible_randomness():
    names=inspect.signature(reconstruct).parameters
    assert not {'hr','target','ground_truth'} & set(names)
    base=torch.rand(2,1,3,4,4)
    mask=torch.zeros_like(base)
    first=reconstruct(ConstantDrift(),base,mask,steps=4,sigma=.1,
                      generator=torch.Generator().manual_seed(5))
    second=reconstruct(ConstantDrift(),base,mask,steps=4,sigma=.1,
                       generator=torch.Generator().manual_seed(5))
    different=reconstruct(ConstantDrift(),base,mask,steps=4,sigma=.1,
                          generator=torch.Generator().manual_seed(6))
    assert torch.equal(first,second)
    assert not torch.equal(first,different)
    assert not first.requires_grad


@pytest.mark.parametrize('args', [dict(steps=0),dict(steps=True),dict(steps=1.5),
                                 dict(sigma=-.1),dict(sigma=float('nan')),
                                 dict(direction='invalid')])
def test_invalid_sde_arguments_are_rejected(args):
    image=torch.zeros(1,1,3,4,4)
    with pytest.raises(ValueError):
        integrate(ConstantDrift(),image,image,image,**args)


def test_invalid_mask_shape_and_nonfinite_inputs_are_rejected():
    image=torch.zeros(1,1,3,4,4)
    with pytest.raises(ValueError,match='binary'):
        integrate(ConstantDrift(),image,image,image+.4)
    with pytest.raises(ValueError):
        integrate(ConstantDrift(),image,image,torch.zeros(2,3))
    image[...,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        integrate(ConstantDrift(),image,image,torch.zeros_like(image))

