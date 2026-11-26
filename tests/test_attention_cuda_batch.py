"""Regression for the A800 SDPA batch-grid failure; CUDA checks run under sbatch."""
import pytest
import torch
from torch.nn import functional as F

from vemgsbm.hybrid import batched_attention, CrossSliceAttention, WindowAttention25D


def test_sequence_chunking_matches_outputs_and_gradients():
    torch.manual_seed(14)
    raw=[torch.randn(19,2,n,4,dtype=torch.float64,requires_grad=True) for n in (7,5,5)]
    chunked=[x.detach().clone().requires_grad_() for x in raw]
    expected=F.scaled_dot_product_attention(*raw)
    actual=batched_attention(*chunked,max_batch=4)
    torch.testing.assert_close(actual,expected,atol=1.e-12,rtol=1.e-12)
    weight=torch.randn_like(actual)
    (actual*weight).sum().backward();(expected*weight).sum().backward()
    for a,b in zip(raw,chunked):
        torch.testing.assert_close(a.grad,b.grad,atol=1.e-12,rtol=1.e-12)


def test_attention_limits_every_launch_and_keeps_tail(monkeypatch):
    launches=[]
    def identity(q,k,v):
        launches.append(q.shape[0]);return v
    monkeypatch.setattr(F,'scaled_dot_product_attention',identity)
    # Same batch expansion as the previously failing B32*64*64 cross-Z layer.
    q=torch.arange(131072,dtype=torch.float32).reshape(-1,1,1,1)
    out=batched_attention(q,q,q)
    torch.testing.assert_close(out,q,rtol=0,atol=0)
    assert launches==[16384]*8
    launches.clear();batched_attention(q[:32769],q[:32769],q[:32769])
    assert launches==[16384,16384,1]


@pytest.mark.parametrize('layer',[CrossSliceAttention(4,2),WindowAttention25D(4,2,4,True)])
def test_both_attention_layers_use_bounded_backend(layer,monkeypatch):
    import vemgsbm.hybrid as hybrid
    original=hybrid.batched_attention
    launches=[]
    def bounded(q,k,v):
        launches.append(len(q));return original(q,k,v,max_batch=3)
    monkeypatch.setattr(hybrid,'batched_attention',bounded)
    x=torch.randn(2,4,3,9,11,requires_grad=True)
    layer(x).square().mean().backward()
    assert launches and torch.isfinite(x.grad).all()


@pytest.mark.parametrize('size',[0,-1,True,1.5])
def test_invalid_chunk_limits_rejected(size):
    x=torch.zeros(1,1,1,1)
    with pytest.raises(ValueError,match='positive integer'):
        batched_attention(x,x,x,size)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='A800 allocated GPU regression')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_original_cuda_cross_z_shape_forward_backward(dtype):
    q=torch.randn(131072,4,7,16,device='cuda',dtype=dtype,requires_grad=True)
    k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True)
    result=batched_attention(q,k,v)
    result.square().mean().backward()
    torch.cuda.synchronize()
    assert result.shape==q.shape and torch.isfinite(result).all()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q,k,v))
