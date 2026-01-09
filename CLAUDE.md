# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

titans-pytorch is an unofficial PyTorch implementation of the [Titans paper](https://arxiv.org/abs/2501.00663) - "Titans: Learning to Memorize at Test Time". The library implements neural memory modules that learn at test time, enabling transformers to memorize and retrieve information through gradient-based updates to memory network weights during inference.

## Commands

### Install dependencies
```bash
pip install titans-pytorch
# Or for development with test dependencies:
pip install -e .[test]
# Or for running experiments:
pip install -e .[examples]
```

### Run tests
```bash
pytest tests/
# Run a single test:
pytest tests/test_titans.py::test_titans -v
# Run with specific parameters:
pytest tests/test_titans.py::test_mac -v
```

### Run experiments
```bash
pip install uv
uv run train_mac.py
```

## Architecture

### Core Components

**NeuralMemory** (`titans_pytorch/neural_memory.py`):
- The central module implementing test-time learning memory
- Memory is stored as weights of a small neural network (MLP by default)
- Uses `torch.func` (vmap, grad, functional_call) for per-sample gradients
- Key operations: `store_memories()` computes surprise/gradients and updates weights via associative scan; `retrieve_memories()` queries the memory network
- Supports momentum (first and higher order), weight decay, adaptive learning rates, and gradient clipping
- State is tracked via `NeuralMemState` namedtuple containing weights, momentum states, and cached segments

**MemoryAsContextTransformer (MAC)** (`titans_pytorch/mac_transformer.py`):
- Full transformer architecture integrating neural memory with attention
- Uses segmented/local attention with persistent memory tokens and longterm memory tokens
- Supports FlexAttention (PyTorch's compiled block-sparse attention) for efficiency on GPU
- Integrates hyper-connections for multi-stream residual processing
- Neural memory can optionally gate attention output or add directly to residual stream

**Memory Models** (`titans_pytorch/memory_models.py`):
- `MemoryMLP`: Default 2-layer MLP from TTT paper
- `MemoryAttention`: Attention-based memory with parallel feedforward
- `MemorySwiGluMLP`: Modern SwiGLU-style MLP
- `FactorizedMemoryMLP`: Low-rank factorized weights for smaller chunk sizes
- `GatedResidualMemoryMLP`: MLP with gated residual connections

### Key Design Patterns

1. **Chunked Processing**: Sequences are processed in chunks for memory updates. `chunk_size` controls granularity of memory updates; `batch_size` controls when weights are committed.

2. **Associative Scan**: Uses `assoc_scan` library for parallel prefix operations in momentum and weight decay computations.

3. **Per-Sample Gradients**: Memory updates use `vmap(grad(...))` to compute gradients for each sample in a batch independently.

4. **Einstein Notation**: Heavy use of `einops` with documented dimension naming:
   - b: batch, h: heads, n: sequence, d: feature, c: intra-chunk, w: weight params, o: momentum orders

## Testing Approach

Tests in `tests/test_titans.py` use pytest parametrize extensively to test combinations of:
- Sequence lengths, chunk sizes, head counts
- Optional features: momentum, qk_rmsnorm, gated transitions, weight residuals
- Parallel vs sequential processing equivalence (critical for validating chunked inference)
- FlexAttention vs standard attention equivalence (when CUDA available)
