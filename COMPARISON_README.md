# Memory Model Comparison

This directory contains tools to compare the Baseline (MemoryMLP) and Hyperbolic (HyperbolicMemoryMLP) memory models.

## Files

- **`compare_memory_models.ipynb`** - Interactive Jupyter notebook with detailed explanations
- **`compare_models.py`** - Standalone Python script for command-line execution
- **`COMPARISON_README.md`** - This file

## Quick Start

### Option 1: Jupyter Notebook (Recommended for exploration)

```bash
# Activate your environment
module load anaconda3/2024.2
conda activate /scratch/gpfs/MARTONOSI/sk2415/titans-pytorch/titans_test_env

# Launch Jupyter
jupyter notebook compare_memory_models.ipynb
```

The notebook includes:
- ✅ Detailed explanations of each step
- ✅ Inline visualizations
- ✅ Interactive experimentation
- ✅ Educational insights

### Option 2: Python Script (For batch execution)

```bash
# Activate your environment
module load anaconda3/2024.2
conda activate /scratch/gpfs/MARTONOSI/sk2415/titans-pytorch/titans_test_env

# Run the comparison
python compare_models.py
```

## What This Comparison Does

### 1. Task: Sequence Memorization
- Creates synthetic sequences of random vectors
- Models must predict the next vector given previous ones
- Tests the ability to store and retrieve patterns

### 2. Models Compared

#### Baseline: MemoryMLP
- Standard MLP operating in Euclidean space
- From the original TTT (Test-Time Training) paper
- Well-understood and computationally efficient

#### Hyperbolic: HyperbolicMemoryMLP
- MLP operating in hyperbolic (Poincaré ball) space
- Uses exponential/logarithmic maps for manifold operations
- Better suited for hierarchical data structures

### 3. Metrics Tracked

- **Training Loss** - How well the model fits the training data
- **Validation Loss** - How well the model generalizes (lower is better)
- **Cosine Similarity** - Direction alignment of predictions (higher is better)
- **Training Time** - Computational efficiency

## Expected Outputs

After running either the notebook or script, you'll get:

1. **`model_comparison.png`** - 4-panel comparison plot showing:
   - Training loss curves
   - Validation loss curves
   - Cosine similarity over time
   - Training time per epoch

2. **`training_results.pkl`** - Pickled training histories for further analysis

3. **Console output** - Summary statistics and winner determination

## Interpreting Results

### Performance Comparison

The comparison will show which model:
- Achieves lower validation loss (better generalization)
- Has higher cosine similarity (better alignment)
- Trains faster (more efficient)

### When to Use Each Model

**Use Baseline (MemoryMLP) when:**
- Data has no hierarchical structure
- Speed is critical
- Simplicity is preferred
- Standard Euclidean space is sufficient

**Use Hyperbolic (HyperbolicMemoryMLP) when:**
- Data has tree-like or hierarchical structure
- Working with graphs, taxonomies, or nested relationships
- Willing to trade computation for potentially better performance on hierarchical data

## Customization

### Modify Hyperparameters

Edit these variables in the script/notebook:

```python
dim = 64              # Vector dimension
seq_length = 32       # Sequence length
num_epochs = 50       # Training epochs
batch_size = 32       # Batch size
lr = 1e-3            # Learning rate
```

### Change Model Architecture

```python
# Adjust depth (number of layers)
memory_model = HyperbolicMemoryMLP(
    dim=dim,
    depth=3,  # Try 1, 2, 3, or more
    expansion_factor=4.0
)

# Adjust expansion factor (hidden layer size)
memory_model = HyperbolicMemoryMLP(
    dim=dim,
    depth=2,
    expansion_factor=8.0  # Larger = more capacity
)

# Make curvature learnable
memory_model = HyperbolicMemoryMLP(
    dim=dim,
    depth=2,
    expansion_factor=4.0,
    curvature=1.0,
    learn_curvature=True  # Let model learn optimal curvature
)
```

### Use Different Datasets

Replace `SequenceMemorizationDataset` with your own:

```python
class YourDataset(Dataset):
    def __init__(self, ...):
        # Your data loading logic
        pass

    def __getitem__(self, idx):
        # Return (input_sequence, target_sequence)
        return x, y
```

## Advanced Usage

### Save and Load Models

```python
# Save model
torch.save(baseline_model.state_dict(), 'baseline_model.pt')

# Load model
baseline_model.load_state_dict(torch.load('baseline_model.pt'))
```

### Test on Custom Data

```python
model.eval()
with torch.no_grad():
    predictions = model(your_input_sequence)
```

### Analyze Specific Predictions

```python
# Get prediction errors per position
errors = torch.norm(predictions - targets, dim=-1)

# Visualize
plt.plot(errors.cpu().numpy())
plt.title('Per-Position Prediction Error')
plt.show()
```

## Known Limitations

### HyperbolicMemoryMLP Limitations

⚠️ **Cannot be used with NeuralMemory module** - The hyperbolic model uses HypLL's `ManifoldParameter`, which doesn't support PyTorch's `vmap` operations. This means it cannot be used within the full `NeuralMemory` module that requires per-sample gradients.

**Workaround:** Use the model standalone as demonstrated in these examples, or implement custom batched operations without `vmap`.

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
batch_size = 16

# Reduce sequence length
seq_length = 16

# Reduce model size
expansion_factor = 2.0
```

### Gradient Issues
```python
# Increase gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

# Reduce learning rate
lr = 5e-4
```

### NaN in Training
```python
# Check input scaling
x = x * 0.1  # Scale down inputs

# Use lower learning rate
lr = 1e-4

# Enable gradient clipping (already included in scripts)
```

## References

- **Titans Paper**: "Titans: Learning to Memorize at Test Time" (https://arxiv.org/abs/2501.00663)
- **HypLL Library**: Hyperbolic Learning Library (https://github.com/maxvanspengler/hyperbolic_learning_library)
- **Poincaré Embeddings**: Nickel & Kiela (2017) - "Poincaré Embeddings for Learning Hierarchical Representations"

## Questions?

For issues or questions:
1. Check the test files: `tests/test_hyperbolic_memory.py`
2. Review the implementation: `titans_pytorch/memory_models.py`
3. Open an issue on the repository

## License

Same as the parent titans-pytorch project.
