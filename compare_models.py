#!/usr/bin/env python3
"""
Simple script to compare Baseline and Hyperbolic Memory Models.

This script trains both models on a sequence memorization task and compares their performance.

Usage:
    python compare_models.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from pathlib import Path

# Import memory models
from titans_pytorch.memory_models import MemoryMLP, HyperbolicMemoryMLP


# ============================================================================
# Dataset Definition
# ============================================================================

class SequenceMemorizationDataset(Dataset):
    """
    Dataset for sequence memorization task.

    Each sequence consists of random vectors where the model must
    predict the next vector given previous ones.
    """

    def __init__(self, num_sequences=1000, seq_length=32, dim=64, seed=None):
        self.num_sequences = num_sequences
        self.seq_length = seq_length
        self.dim = dim

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # Pre-generate all sequences
        self.sequences = torch.randn(num_sequences, seq_length + 1, dim)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = seq[:-1]  # Input: all but last
        y = seq[1:]   # Target: all but first (shifted by 1)
        return x, y


# ============================================================================
# Model Wrapper
# ============================================================================

class MemoryModelWrapper(nn.Module):
    """Wrapper for memory models."""

    def __init__(self, memory_model, dim):
        super().__init__()
        self.memory = memory_model

    def forward(self, x):
        return self.memory(x)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_cosine_sim = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            predictions = model(x)
            loss = criterion(predictions, y)
            total_loss += loss.item()

            # Compute cosine similarity
            pred_flat = predictions.reshape(-1, predictions.size(-1))
            target_flat = y.reshape(-1, y.size(-1))
            pred_norm = torch.nn.functional.normalize(pred_flat, dim=-1)
            target_norm = torch.nn.functional.normalize(target_flat, dim=-1)
            cosine_sim = (pred_norm * target_norm).sum(dim=-1).mean()
            total_cosine_sim += cosine_sim.item()

            num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'cosine_similarity': total_cosine_sim / num_batches
    }


def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device='cpu', verbose=True):
    """Complete training loop."""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_cosine_sim': [],
        'epoch_times': []
    }

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_cosine_sim = val_metrics['cosine_similarity']

        scheduler.step(val_loss)

        epoch_time = time.time() - start_time
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_cosine_sim'].append(val_cosine_sim)
        history['epoch_times'].append(epoch_time)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                  f"cosine_sim={val_cosine_sim:.4f}, time={epoch_time:.2f}s")

    return history


# ============================================================================
# Visualization
# ============================================================================

def plot_comparison(baseline_history, hyperbolic_history, save_path='model_comparison.png'):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training Loss
    ax = axes[0, 0]
    ax.plot(baseline_history['train_loss'], label='Baseline (MemoryMLP)', linewidth=2)
    ax.plot(hyperbolic_history['train_loss'], label='Hyperbolic (HyperbolicMemoryMLP)', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss (MSE)', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Validation Loss
    ax = axes[0, 1]
    ax.plot(baseline_history['val_loss'], label='Baseline (MemoryMLP)', linewidth=2)
    ax.plot(hyperbolic_history['val_loss'], label='Hyperbolic (HyperbolicMemoryMLP)', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss (MSE)', fontsize=12)
    ax.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Cosine Similarity
    ax = axes[1, 0]
    ax.plot(baseline_history['val_cosine_sim'], label='Baseline (MemoryMLP)', linewidth=2)
    ax.plot(hyperbolic_history['val_cosine_sim'], label='Hyperbolic (HyperbolicMemoryMLP)', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Prediction Alignment (Higher is Better)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Training Time
    ax = axes[1, 1]
    ax.plot(baseline_history['epoch_times'], label='Baseline (MemoryMLP)', linewidth=2)
    ax.plot(hyperbolic_history['epoch_times'], label='Hyperbolic (HyperbolicMemoryMLP)', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to '{save_path}'")


def print_summary(baseline_history, hyperbolic_history):
    """Print comparison summary."""
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)

    # Get final metrics (average of last 5 epochs)
    baseline_final_train = np.mean(baseline_history['train_loss'][-5:])
    baseline_final_val = np.mean(baseline_history['val_loss'][-5:])
    baseline_final_cosine = np.mean(baseline_history['val_cosine_sim'][-5:])
    baseline_avg_time = np.mean(baseline_history['epoch_times'])

    hyperbolic_final_train = np.mean(hyperbolic_history['train_loss'][-5:])
    hyperbolic_final_val = np.mean(hyperbolic_history['val_loss'][-5:])
    hyperbolic_final_cosine = np.mean(hyperbolic_history['val_cosine_sim'][-5:])
    hyperbolic_avg_time = np.mean(hyperbolic_history['epoch_times'])

    print("\n📊 BASELINE MODEL (MemoryMLP)")
    print("-" * 70)
    print(f"Final Training Loss:      {baseline_final_train:.6f}")
    print(f"Final Validation Loss:    {baseline_final_val:.6f}")
    print(f"Final Cosine Similarity:  {baseline_final_cosine:.4f}")
    print(f"Avg Training Time/Epoch:  {baseline_avg_time:.2f}s")

    print("\n🌀 HYPERBOLIC MODEL (HyperbolicMemoryMLP)")
    print("-" * 70)
    print(f"Final Training Loss:      {hyperbolic_final_train:.6f}")
    print(f"Final Validation Loss:    {hyperbolic_final_val:.6f}")
    print(f"Final Cosine Similarity:  {hyperbolic_final_cosine:.4f}")
    print(f"Avg Training Time/Epoch:  {hyperbolic_avg_time:.2f}s")

    print("\n📈 RELATIVE IMPROVEMENTS")
    print("-" * 70)

    val_loss_improvement = ((baseline_final_val - hyperbolic_final_val) / baseline_final_val) * 100
    cosine_improvement = ((hyperbolic_final_cosine - baseline_final_cosine) / baseline_final_cosine) * 100
    time_overhead = ((hyperbolic_avg_time - baseline_avg_time) / baseline_avg_time) * 100

    print(f"Validation Loss Change:   {val_loss_improvement:+.2f}% {'(better)' if val_loss_improvement > 0 else '(worse)'}")
    print(f"Cosine Similarity Change: {cosine_improvement:+.2f}% {'(better)' if cosine_improvement > 0 else '(worse)'}")
    print(f"Training Time Overhead:   {time_overhead:+.2f}%")

    print("\n🏆 WINNER")
    print("-" * 70)

    if hyperbolic_final_val < baseline_final_val:
        winner = "Hyperbolic Model"
        improvement = val_loss_improvement
    elif baseline_final_val < hyperbolic_final_val:
        winner = "Baseline Model"
        improvement = -val_loss_improvement
    else:
        winner = "Tie"
        improvement = 0

    if winner != "Tie":
        print(f"{winner} wins with {improvement:.2f}% better validation loss!")
    else:
        print("Both models perform equally well!")

    print("\n" + "="*70)


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main training and comparison."""
    print("\n" + "="*70)
    print("MEMORY MODEL COMPARISON EXPERIMENT")
    print("="*70)

    # Configuration
    dim = 64
    seq_length = 32
    num_epochs = 50
    batch_size = 32
    lr = 1e-3

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SequenceMemorizationDataset(
        num_sequences=1000, seq_length=seq_length, dim=dim, seed=42
    )
    val_dataset = SequenceMemorizationDataset(
        num_sequences=200, seq_length=seq_length, dim=dim, seed=43
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"  Training sequences: {len(train_dataset)}")
    print(f"  Validation sequences: {len(val_dataset)}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Vector dimension: {dim}")

    # Create models
    print("\nCreating models...")

    baseline_memory = MemoryMLP(dim=dim, depth=2, expansion_factor=4.0)
    baseline_model = MemoryModelWrapper(baseline_memory, dim).to(device)

    hyperbolic_memory = HyperbolicMemoryMLP(
        dim=dim, depth=2, expansion_factor=4.0, curvature=1.0, learn_curvature=False
    )
    hyperbolic_model = MemoryModelWrapper(hyperbolic_memory, dim).to(device)

    print(f"  Baseline parameters: {sum(p.numel() for p in baseline_model.parameters())}")

    # Count hyperbolic parameters
    total_params = 0
    for p in hyperbolic_model.parameters():
        try:
            total_params += p.numel()
        except TypeError:
            total_params += p.tensor.numel()
    print(f"  Hyperbolic parameters: {total_params}")

    # Train baseline model
    print("\n" + "="*70)
    print("TRAINING BASELINE MODEL (MemoryMLP)")
    print("="*70)

    baseline_history = train_model(
        baseline_model, train_loader, val_loader,
        num_epochs=num_epochs, lr=lr, device=device, verbose=True
    )

    print("\n✓ Baseline training complete!")

    # Train hyperbolic model
    print("\n" + "="*70)
    print("TRAINING HYPERBOLIC MODEL (HyperbolicMemoryMLP)")
    print("="*70)

    hyperbolic_history = train_model(
        hyperbolic_model, train_loader, val_loader,
        num_epochs=num_epochs, lr=lr, device=device, verbose=True
    )

    print("\n✓ Hyperbolic training complete!")

    # Compare results
    print_summary(baseline_history, hyperbolic_history)

    # Create plots
    print("\nGenerating comparison plots...")
    plot_comparison(baseline_history, hyperbolic_history)

    # Save results
    print("\nSaving results...")
    results = {
        'baseline_history': baseline_history,
        'hyperbolic_history': hyperbolic_history,
        'config': {
            'dim': dim,
            'seq_length': seq_length,
            'num_train': len(train_dataset),
            'num_val': len(val_dataset),
            'num_epochs': num_epochs
        }
    }

    with open('training_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("✓ Results saved to 'training_results.pkl'")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - model_comparison.png    (comparison plots)")
    print("  - training_results.pkl    (training histories)")


if __name__ == "__main__":
    main()
