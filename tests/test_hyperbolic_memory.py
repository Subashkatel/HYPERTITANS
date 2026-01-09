"""
Tests for all memory MLP models in titans_pytorch.

This file tests:
1. MemoryMLP - Standard Euclidean MLP
2. GatedResidualMemoryMLP - MLP with gated residual connections
3. FactorizedMemoryMLP - Low-rank factorized MLP
4. MemorySwiGluMLP - Modern SwiGLU-style MLP
5. MemoryAttention - Attention-based memory module
6. HyperbolicMemoryMLP - MLP operating in Poincare ball space

Tests verify:
1. Both models produce valid outputs with correct shapes
2. Both models can be used with NeuralMemory module
3. Gradient flow works correctly for both models
4. Performance characteristics (forward pass, backward pass)
"""

import time
from contextlib import contextmanager

import torch
from torch import nn
import pytest

from titans_pytorch import NeuralMemory
from titans_pytorch.memory_models import (
    MemoryMLP,
    GatedResidualMemoryMLP,
    FactorizedMemoryMLP,
    MemorySwiGluMLP,
    MemoryAttention,
    HyperbolicMemoryMLP,
)


# Helper functions

@contextmanager
def torch_default_dtype(dtype):
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(prev_dtype)


def count_parameters(model):
    """Count parameters, handling both regular Parameters and ManifoldParameters."""
    total = 0
    for p in model.parameters():
        try:
            # Try regular .numel() first
            total += p.numel()
        except TypeError:
            # ManifoldParameter from HypLL raises TypeError and requires .tensor access
            total += p.tensor.numel()
    return total


# =============================================================================
# Basic shape and output tests for all models
# =============================================================================

@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('expansion_factor', [2., 4.])
def test_memory_mlp_output_shape(depth, expansion_factor):
    """Test that MemoryMLP produces outputs of correct shape."""
    dim = 64
    batch_size = 4
    seq_len = 16

    model = MemoryMLP(
        dim=dim,
        depth=depth,
        expansion_factor=expansion_factor
    )

    # Test 2D input (batch, dim)
    x_2d = torch.randn(batch_size, dim)
    out_2d = model(x_2d)
    assert out_2d.shape == x_2d.shape, f"2D shape mismatch: {out_2d.shape} vs {x_2d.shape}"

    # Test 3D input (batch, seq, dim)
    x_3d = torch.randn(batch_size, seq_len, dim)
    out_3d = model(x_3d)
    assert out_3d.shape == x_3d.shape, f"3D shape mismatch: {out_3d.shape} vs {x_3d.shape}"


@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('expansion_factor', [2., 4.])
def test_gated_residual_mlp_output_shape(depth, expansion_factor):
    """Test that GatedResidualMemoryMLP produces outputs of correct shape."""
    dim = 64
    batch_size = 4
    seq_len = 16

    model = GatedResidualMemoryMLP(
        dim=dim,
        depth=depth,
        expansion_factor=expansion_factor
    )

    # Test 2D input (batch, dim)
    x_2d = torch.randn(batch_size, dim)
    out_2d = model(x_2d)
    assert out_2d.shape == x_2d.shape, f"2D shape mismatch: {out_2d.shape} vs {x_2d.shape}"

    # Test 3D input (batch, seq, dim)
    x_3d = torch.randn(batch_size, seq_len, dim)
    out_3d = model(x_3d)
    assert out_3d.shape == x_3d.shape, f"3D shape mismatch: {out_3d.shape} vs {x_3d.shape}"


@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('k', [16, 32, 64])
def test_factorized_mlp_output_shape(depth, k):
    """Test that FactorizedMemoryMLP produces outputs of correct shape."""
    dim = 64
    batch_size = 4
    seq_len = 16

    model = FactorizedMemoryMLP(
        dim=dim,
        depth=depth,
        k=k
    )

    # Test 2D input (batch, dim)
    x_2d = torch.randn(batch_size, dim)
    out_2d = model(x_2d)
    assert out_2d.shape == x_2d.shape, f"2D shape mismatch: {out_2d.shape} vs {x_2d.shape}"

    # Test 3D input (batch, seq, dim)
    x_3d = torch.randn(batch_size, seq_len, dim)
    out_3d = model(x_3d)
    assert out_3d.shape == x_3d.shape, f"3D shape mismatch: {out_3d.shape} vs {x_3d.shape}"


@pytest.mark.parametrize('depth', [1, 2])
@pytest.mark.parametrize('expansion_factor', [2., 4.])
def test_swiglu_mlp_output_shape(depth, expansion_factor):
    """Test that MemorySwiGluMLP produces outputs of correct shape."""
    dim = 64
    batch_size = 4
    seq_len = 16

    model = MemorySwiGluMLP(
        dim=dim,
        depth=depth,
        expansion_factor=expansion_factor
    )

    # Test 2D input (batch, dim)
    x_2d = torch.randn(batch_size, dim)
    out_2d = model(x_2d)
    assert out_2d.shape == x_2d.shape, f"2D shape mismatch: {out_2d.shape} vs {x_2d.shape}"

    # Test 3D input (batch, seq, dim)
    x_3d = torch.randn(batch_size, seq_len, dim)
    out_3d = model(x_3d)
    assert out_3d.shape == x_3d.shape, f"3D shape mismatch: {out_3d.shape} vs {x_3d.shape}"


@pytest.mark.parametrize('expansion_factor', [2., 4.])
@pytest.mark.parametrize('scale', [4., 8.])
def test_memory_attention_output_shape(expansion_factor, scale):
    """Test that MemoryAttention produces outputs of correct shape."""
    dim = 64
    batch_size = 4
    seq_len = 16

    model = MemoryAttention(
        dim=dim,
        scale=scale,
        expansion_factor=expansion_factor
    )

    # Test 2D input (batch, dim) - Note: attention needs at least 2D for QKV
    # MemoryAttention expects 3D input for the attention mechanism
    x_3d = torch.randn(batch_size, seq_len, dim)
    out_3d = model(x_3d)
    assert out_3d.shape == x_3d.shape, f"3D shape mismatch: {out_3d.shape} vs {x_3d.shape}"


@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('expansion_factor', [2., 4.])
def test_hyperbolic_mlp_output_shape(depth, expansion_factor):
    """Test that HyperbolicMemoryMLP produces outputs of correct shape."""
    dim = 64
    batch_size = 4
    seq_len = 16

    model = HyperbolicMemoryMLP(
        dim=dim,
        depth=depth,
        expansion_factor=expansion_factor
    )

    # Test 2D input (batch, dim)
    x_2d = torch.randn(batch_size, dim)
    out_2d = model(x_2d)
    assert out_2d.shape == x_2d.shape, f"2D shape mismatch: {out_2d.shape} vs {x_2d.shape}"

    # Test 3D input (batch, seq, dim)
    x_3d = torch.randn(batch_size, seq_len, dim)
    out_3d = model(x_3d)
    assert out_3d.shape == x_3d.shape, f"3D shape mismatch: {out_3d.shape} vs {x_3d.shape}"


# =============================================================================
# Gradient flow tests
# =============================================================================

def test_memory_mlp_gradient_flow():
    """Test that gradients flow correctly through MemoryMLP."""
    dim = 32
    model = MemoryMLP(dim=dim, depth=2)

    x = torch.randn(4, 8, dim, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input gradients should exist"
    assert not torch.isnan(x.grad).any(), "Input gradients should not contain NaN"
    assert not torch.isinf(x.grad).any(), "Input gradients should not contain Inf"

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_gated_residual_mlp_gradient_flow():
    """Test that gradients flow correctly through GatedResidualMemoryMLP."""
    dim = 32
    model = GatedResidualMemoryMLP(dim=dim, depth=2)

    x = torch.randn(4, 8, dim, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input gradients should exist"
    assert not torch.isnan(x.grad).any(), "Input gradients should not contain NaN"
    assert not torch.isinf(x.grad).any(), "Input gradients should not contain Inf"

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_factorized_mlp_gradient_flow():
    """Test that gradients flow correctly through FactorizedMemoryMLP."""
    dim = 32
    model = FactorizedMemoryMLP(dim=dim, depth=2)

    x = torch.randn(4, 8, dim, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input gradients should exist"
    assert not torch.isnan(x.grad).any(), "Input gradients should not contain NaN"
    assert not torch.isinf(x.grad).any(), "Input gradients should not contain Inf"

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_swiglu_mlp_gradient_flow():
    """Test that gradients flow correctly through MemorySwiGluMLP."""
    dim = 32
    model = MemorySwiGluMLP(dim=dim, depth=2)

    x = torch.randn(4, 8, dim, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input gradients should exist"
    assert not torch.isnan(x.grad).any(), "Input gradients should not contain NaN"
    assert not torch.isinf(x.grad).any(), "Input gradients should not contain Inf"

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_memory_attention_gradient_flow():
    """Test that gradients flow correctly through MemoryAttention."""
    dim = 32
    model = MemoryAttention(dim=dim)

    x = torch.randn(4, 8, dim, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input gradients should exist"
    assert not torch.isnan(x.grad).any(), "Input gradients should not contain NaN"
    assert not torch.isinf(x.grad).any(), "Input gradients should not contain Inf"

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


def test_hyperbolic_mlp_gradient_flow():
    """Test that gradients flow correctly through HyperbolicMemoryMLP."""
    dim = 32
    model = HyperbolicMemoryMLP(dim=dim, depth=2)

    x = torch.randn(4, 8, dim, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input gradients should exist"
    assert not torch.isnan(x.grad).any(), "Input gradients should not contain NaN"
    assert not torch.isinf(x.grad).any(), "Input gradients should not contain Inf"

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


# =============================================================================
# Integration with NeuralMemory tests
# =============================================================================

@pytest.mark.parametrize('model_class,model_kwargs', [
    (MemoryMLP, {'dim': 16, 'depth': 2}),
    (GatedResidualMemoryMLP, {'dim': 16, 'depth': 2}),
    (FactorizedMemoryMLP, {'dim': 16, 'depth': 2}),
    (MemorySwiGluMLP, {'dim': 16, 'depth': 1}),
    pytest.param(HyperbolicMemoryMLP, {'dim': 16, 'depth': 2}, marks=pytest.mark.skip(
        reason="HyperbolicMemoryMLP uses ManifoldParameter which doesn't support vmap operations used in NeuralMemory"
    )),
])
def test_neural_memory_integration(model_class, model_kwargs):
    """Test that all MLP types work with NeuralMemory module.

    Note: HyperbolicMemoryMLP is skipped because HypLL's ManifoldParameter
    doesn't support PyTorch's vmap, which is used by NeuralMemory for
    per-sample gradient computation.
    """
    memory_model = model_class(**model_kwargs)

    mem = NeuralMemory(
        dim=16,
        chunk_size=8,
        model=memory_model,
        mem_model_norm_add_residual=False
    )

    seq = torch.randn(2, 32, 16)
    retrieved, state = mem(seq)

    assert retrieved.shape == seq.shape, f"Shape mismatch: {retrieved.shape} vs {seq.shape}"
    assert not torch.isnan(retrieved).any(), "Output contains NaN"


@pytest.mark.parametrize('model_class,model_kwargs', [
    (MemoryMLP, {'dim': 16, 'depth': 2}),
    (GatedResidualMemoryMLP, {'dim': 16, 'depth': 2}),
    (FactorizedMemoryMLP, {'dim': 16, 'depth': 2}),
    (MemorySwiGluMLP, {'dim': 16, 'depth': 1}),
    pytest.param(HyperbolicMemoryMLP, {'dim': 16, 'depth': 2}, marks=pytest.mark.skip(
        reason="HyperbolicMemoryMLP uses ManifoldParameter which doesn't support vmap operations used in NeuralMemory"
    )),
])
def test_neural_memory_backward(model_class, model_kwargs):
    """Test backward pass through NeuralMemory with all MLP types.

    Note: HyperbolicMemoryMLP is skipped because HypLL's ManifoldParameter
    doesn't support PyTorch's vmap, which is used by NeuralMemory for
    per-sample gradient computation.
    """
    memory_model = model_class(**model_kwargs)

    mem = NeuralMemory(
        dim=16,
        chunk_size=8,
        model=memory_model,
        mem_model_norm_add_residual=False
    )

    seq = torch.randn(2, 32, 16, requires_grad=True)
    retrieved, _ = mem(seq)

    loss = retrieved.sum()
    loss.backward()

    assert seq.grad is not None, "Gradients should flow to input"
    assert not torch.isnan(seq.grad).any(), "Input gradients contain NaN"


# =============================================================================
# Learnable curvature test for hyperbolic
# =============================================================================

def test_hyperbolic_mlp_learnable_curvature():
    """Test HyperbolicMemoryMLP with learnable curvature parameter."""
    dim = 32
    model = HyperbolicMemoryMLP(
        dim=dim,
        depth=2,
        curvature=1.0,
        learn_curvature=True
    )

    x = torch.randn(4, 8, dim)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Check that curvature parameter exists and has gradient
    curvature_param = model.manifold.c
    if hasattr(curvature_param, 'value') and curvature_param.value.requires_grad:
        assert curvature_param.value.grad is not None, "Curvature should have gradient"


# =============================================================================
# Comparison tests
# =============================================================================

def test_output_magnitude_comparison():
    """Compare output magnitudes between all MLP types."""
    dim = 64
    batch_size = 4
    seq_len = 16

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, dim)

    models = {
        'MemoryMLP': MemoryMLP(dim=dim, depth=2),
        'GatedResidualMLP': GatedResidualMemoryMLP(dim=dim, depth=2),
        'FactorizedMLP': FactorizedMemoryMLP(dim=dim, depth=2),
        'SwiGluMLP': MemorySwiGluMLP(dim=dim, depth=1),
        'MemoryAttention': MemoryAttention(dim=dim),
        'HyperbolicMLP': HyperbolicMemoryMLP(dim=dim, depth=2),
    }

    for name, model in models.items():
        out = model(x)
        assert torch.isfinite(out).all(), f"{name} output should be finite"
        print(f"\n{name} - mean: {out.mean():.4f}, std: {out.std():.4f}")


def test_parameter_count_comparison():
    """Compare parameter counts between all MLP types."""
    dim = 64
    depth = 2
    expansion_factor = 4.

    models = {
        'MemoryMLP': MemoryMLP(dim=dim, depth=depth, expansion_factor=expansion_factor),
        'GatedResidualMLP': GatedResidualMemoryMLP(dim=dim, depth=depth, expansion_factor=expansion_factor),
        'FactorizedMLP': FactorizedMemoryMLP(dim=dim, depth=depth, k=32),
        'SwiGluMLP': MemorySwiGluMLP(dim=dim, depth=1, expansion_factor=expansion_factor),
        'MemoryAttention': MemoryAttention(dim=dim, expansion_factor=expansion_factor),
        'HyperbolicMLP': HyperbolicMemoryMLP(dim=dim, depth=depth, expansion_factor=expansion_factor),
    }

    print("\nParameter counts:")
    for name, model in models.items():
        params = count_parameters(model)
        print(f"  {name}: {params}")
        assert params > 0, f"{name} should have parameters"


@pytest.mark.parametrize('model_class,model_name,model_kwargs', [
    (MemoryMLP, 'MemoryMLP', {'dim': 64, 'depth': 2}),
    (GatedResidualMemoryMLP, 'GatedResidualMLP', {'dim': 64, 'depth': 2}),
    (FactorizedMemoryMLP, 'FactorizedMLP', {'dim': 64, 'depth': 2}),
    (MemorySwiGluMLP, 'SwiGluMLP', {'dim': 64, 'depth': 1}),
    (MemoryAttention, 'MemoryAttention', {'dim': 64}),
    (HyperbolicMemoryMLP, 'HyperbolicMLP', {'dim': 64, 'depth': 2}),
])
def test_forward_pass_timing(model_class, model_name, model_kwargs):
    """Benchmark forward pass timing for all MLP types."""
    batch_size = 8
    seq_len = 128
    num_iterations = 10

    model = model_class(**model_kwargs)
    dim = model_kwargs.get('dim', 64)
    x = torch.randn(batch_size, seq_len, dim)

    # Warmup
    for _ in range(3):
        _ = model(x)

    # Timing
    start = time.time()
    for _ in range(num_iterations):
        _ = model(x)
    elapsed = time.time() - start

    avg_time = elapsed / num_iterations * 1000  # Convert to ms
    print(f"\n{model_name} forward pass: {avg_time:.2f} ms")


# =============================================================================
# Numerical stability tests
# =============================================================================

@pytest.mark.parametrize('model_class,model_kwargs', [
    (MemoryMLP, {'dim': 32, 'depth': 2}),
    (GatedResidualMemoryMLP, {'dim': 32, 'depth': 2}),
    (FactorizedMemoryMLP, {'dim': 32, 'depth': 2}),
    (MemorySwiGluMLP, {'dim': 32, 'depth': 1}),
    (MemoryAttention, {'dim': 32}),
    (HyperbolicMemoryMLP, {'dim': 32, 'depth': 2}),
])
@pytest.mark.parametrize('scale', [0.01, 0.1, 1.0, 10.0])
def test_input_scale_stability(model_class, model_kwargs, scale):
    """Test stability with different input scales for all MLPs."""
    dim = model_kwargs.get('dim', 32)
    model = model_class(**model_kwargs)

    x = torch.randn(4, 8, dim) * scale
    out = model(x)

    assert torch.isfinite(out).all(), f"Output should be finite for {model_class.__name__} with scale={scale}"


@pytest.mark.parametrize('model_class,model_kwargs', [
    (MemoryMLP, {'dim': 32, 'depth': 2}),
    (GatedResidualMemoryMLP, {'dim': 32, 'depth': 2}),
    (FactorizedMemoryMLP, {'dim': 32, 'depth': 2}),
    (MemorySwiGluMLP, {'dim': 32, 'depth': 1}),
    (MemoryAttention, {'dim': 32}),
    (HyperbolicMemoryMLP, {'dim': 32, 'depth': 2}),
])
def test_zero_input(model_class, model_kwargs):
    """Test all MLPs with zero input."""
    dim = model_kwargs.get('dim', 32)
    model = model_class(**model_kwargs)

    x = torch.zeros(4, 8, dim)
    out = model(x)

    assert torch.isfinite(out).all(), f"Output should be finite for {model_class.__name__} with zero input"


# =============================================================================
# State chaining test with NeuralMemory
# =============================================================================

@pytest.mark.parametrize('model_class,model_kwargs', [
    (MemoryMLP, {'dim': 16, 'depth': 2}),
    (GatedResidualMemoryMLP, {'dim': 16, 'depth': 2}),
    (FactorizedMemoryMLP, {'dim': 16, 'depth': 2}),
    (MemorySwiGluMLP, {'dim': 16, 'depth': 1}),
    pytest.param(HyperbolicMemoryMLP, {'dim': 16, 'depth': 2}, marks=pytest.mark.skip(
        reason="HyperbolicMemoryMLP uses ManifoldParameter which doesn't support vmap operations used in NeuralMemory"
    )),
])
def test_neural_memory_state_chaining(model_class, model_kwargs):
    """Test that state chaining works correctly with all MLP types.

    Note: HyperbolicMemoryMLP is skipped because HypLL's ManifoldParameter
    doesn't support PyTorch's vmap, which is used by NeuralMemory for
    per-sample gradient computation.
    """
    memory_model = model_class(**model_kwargs)

    mem = NeuralMemory(
        dim=16,
        chunk_size=16,
        model=memory_model,
        mem_model_norm_add_residual=False
    )

    # Process in one shot
    seq = torch.randn(2, 48, 16)
    parallel_retrieved, _ = mem(seq)

    # Process in chunks
    seq_first, seq_second, seq_third = seq.split(16, dim=1)

    first_retrieved, state = mem(seq_first)
    second_retrieved, state = mem(seq_second, state=state)
    third_retrieved, state = mem(seq_third, state=state)

    sequential_retrieved = torch.cat([first_retrieved, second_retrieved, third_retrieved], dim=1)

    # Results should match
    assert torch.allclose(parallel_retrieved, sequential_retrieved, atol=1e-5), \
        f"Parallel and sequential results should match for {model_class.__name__}"


# =============================================================================
# Model-specific edge case tests
# =============================================================================

def test_factorized_mlp_different_k_values():
    """Test FactorizedMemoryMLP with various k (rank) values."""
    dim = 64
    batch_size = 4
    seq_len = 16

    for k in [4, 8, 16, 32, 64, 128]:
        model = FactorizedMemoryMLP(dim=dim, depth=2, k=k)
        x = torch.randn(batch_size, seq_len, dim)
        out = model(x)

        assert out.shape == x.shape, f"Shape mismatch for k={k}"
        assert torch.isfinite(out).all(), f"Output not finite for k={k}"


def test_hyperbolic_mlp_different_curvatures():
    """Test HyperbolicMemoryMLP with various curvature values."""
    dim = 32
    batch_size = 4
    seq_len = 8

    for curvature in [0.1, 0.5, 1.0, 2.0, 5.0]:
        model = HyperbolicMemoryMLP(dim=dim, depth=2, curvature=curvature)
        x = torch.randn(batch_size, seq_len, dim) * 0.5  # Scale down to avoid numerical issues
        out = model(x)

        assert out.shape == x.shape, f"Shape mismatch for curvature={curvature}"
        assert torch.isfinite(out).all(), f"Output not finite for curvature={curvature}"


def test_memory_attention_different_scales():
    """Test MemoryAttention with various scale values."""
    dim = 32
    batch_size = 4
    seq_len = 16

    for scale in [1., 4., 8., 16., 32.]:
        model = MemoryAttention(dim=dim, scale=scale)
        x = torch.randn(batch_size, seq_len, dim)
        out = model(x)

        assert out.shape == x.shape, f"Shape mismatch for scale={scale}"
        assert torch.isfinite(out).all(), f"Output not finite for scale={scale}"


def test_swiglu_mlp_different_depths():
    """Test MemorySwiGluMLP with various depths."""
    dim = 32
    batch_size = 4
    seq_len = 16

    for depth in [1, 2, 3, 4]:
        model = MemorySwiGluMLP(dim=dim, depth=depth)
        x = torch.randn(batch_size, seq_len, dim)
        out = model(x)

        assert out.shape == x.shape, f"Shape mismatch for depth={depth}"
        assert torch.isfinite(out).all(), f"Output not finite for depth={depth}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
