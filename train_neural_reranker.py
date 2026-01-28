#!/usr/bin/env python3
"""
Train neural reranker for Shan spell correction.

This script trains a neural model that learns to select the best
correction candidate from classical spell correction using context.

Usage:
    # Generate training data first
    python generate_synthetic_errors.py --corpus_dir ./data/corpus --num_pairs 50000 --split

    # Train the model
    python train_neural_reranker.py \
        --train_data training_pairs_train.jsonl \
        --val_data training_pairs_val.jsonl \
        --output spell_reranker.pt \
        --epochs 10

Requirements:
    - PyTorch >= 2.0
    - CUDA-capable GPU recommended (RTX 4090 optimal)
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shannlp.spell.neural.model import SpellReranker, CharVocab
from shannlp.spell.neural.dataset import SpellDataset, collate_fn


def train_epoch(
    model: SpellReranker,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: Optional[GradScaler] = None,
    accumulation_steps: int = 1
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0
    num_batches = len(dataloader)

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        error_chars = batch['error_chars'].to(device)
        error_lengths = batch['error_lengths'].to(device)
        candidate_chars = batch['candidate_chars'].to(device)
        candidate_lengths = batch['candidate_lengths'].to(device)
        candidate_mask = batch['candidate_mask'].to(device)
        context_chars = batch['context_chars'].to(device)
        context_lengths = batch['context_lengths'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass with mixed precision
        if scaler is not None:
            with autocast(device_type=device):
                scores = model(
                    error_chars, error_lengths,
                    candidate_chars, candidate_lengths,
                    context_chars, context_lengths,
                    candidate_mask
                )
                loss = criterion(scores, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            scores = model(
                error_chars, error_lengths,
                candidate_chars, candidate_lengths,
                context_chars, context_lengths,
                candidate_mask
            )
            loss = criterion(scores, labels) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Calculate accuracy
        predictions = scores.argmax(dim=1)
        correct = (predictions == labels).sum().item()

        total_loss += loss.item() * accumulation_steps * labels.size(0)
        total_correct += correct
        total_samples += labels.size(0)

        # Progress
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples * 100
            progress = (batch_idx + 1) / num_batches * 100

            sys.stdout.write(f"\r  [{progress:5.1f}%] Loss: {avg_loss:.4f} | Acc: {accuracy:.1f}%")
            sys.stdout.flush()

    print()

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples * 100
    }


def evaluate(
    model: SpellReranker,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    # Top-k accuracy
    top3_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for batch in dataloader:
            error_chars = batch['error_chars'].to(device)
            error_lengths = batch['error_lengths'].to(device)
            candidate_chars = batch['candidate_chars'].to(device)
            candidate_lengths = batch['candidate_lengths'].to(device)
            candidate_mask = batch['candidate_mask'].to(device)
            context_chars = batch['context_chars'].to(device)
            context_lengths = batch['context_lengths'].to(device)
            labels = batch['labels'].to(device)

            scores = model(
                error_chars, error_lengths,
                candidate_chars, candidate_lengths,
                context_chars, context_lengths,
                candidate_mask
            )

            loss = criterion(scores, labels)

            predictions = scores.argmax(dim=1)
            correct = (predictions == labels).sum().item()

            # Top-k accuracy
            top3 = scores.topk(min(3, scores.size(1)), dim=1).indices
            top5 = scores.topk(min(5, scores.size(1)), dim=1).indices

            for i, label in enumerate(labels):
                if label in top3[i]:
                    top3_correct += 1
                if label in top5[i]:
                    top5_correct += 1

            total_loss += loss.item() * labels.size(0)
            total_correct += correct
            total_samples += labels.size(0)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples * 100,
        'top3_accuracy': top3_correct / total_samples * 100,
        'top5_accuracy': top5_correct / total_samples * 100
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train neural reranker for spell correction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data (JSONL)"
    )

    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="Path to validation data (JSONL)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="spell_reranker.pt",
        help="Output model path"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )

    parser.add_argument(
        "--embed_dim",
        type=int,
        default=128,
        help="Character embedding dimension"
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="LSTM hidden dimension"
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of LSTM layers"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate"
    )

    parser.add_argument(
        "--max_candidates",
        type=int,
        default=10,
        help="Maximum candidates per sample"
    )

    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training"
    )

    # Determine best available device
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"  # Apple Silicon GPU
    else:
        default_device = "cpu"

    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Device to use (cuda, mps, or cpu)"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("Neural Spell Reranker Training")
    print("=" * 60)
    print()

    # Check device
    device = args.device
    use_pin_memory = False

    if device == "cuda" and torch.cuda.is_available():
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        use_pin_memory = True
    elif device == "mps" and torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        # MPS doesn't support pin_memory
        use_pin_memory = False
    else:
        device = "cpu"
        print("Using CPU")
    print()

    # Create vocabulary
    print("Creating vocabulary...")
    vocab = CharVocab()
    print(f"  Vocabulary size: {len(vocab)}")

    # Create datasets
    print("\nLoading training data...")
    train_dataset = SpellDataset(
        args.train_data,
        vocab,
        max_candidates=args.max_candidates,
        generate_candidates=True,
        augment=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )

    val_loader = None
    if args.val_data:
        print("\nLoading validation data...")
        val_dataset = SpellDataset(
            args.val_data,
            vocab,
            max_candidates=args.max_candidates,
            generate_candidates=True,
            augment=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=use_pin_memory
        )

    # Create model
    print("\nCreating model...")
    model = SpellReranker(
        vocab=vocab,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_candidates=args.max_candidates
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Mixed precision (only supported on CUDA)
    use_amp = args.fp16 and device == "cuda"
    scaler = GradScaler() if use_amp else None
    if args.fp16 and device != "cuda":
        print("Note: Mixed precision (--fp16) only supported on CUDA. Using full precision.")

    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_val_acc = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            device, scaler, args.accumulation_steps
        )

        # Validate
        val_metrics = {}
        if val_loader:
            val_metrics = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Logging
        elapsed = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]

        print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.1f}%")
        if val_metrics:
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.1f}% | "
                  f"Top-3: {val_metrics['top3_accuracy']:.1f}% | Top-5: {val_metrics['top5_accuracy']:.1f}%")
        print(f"  LR: {current_lr:.6f} | Time: {elapsed:.1f}s")

        # Save best model
        val_acc = val_metrics.get('accuracy', train_metrics['accuracy'])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(args.output)
            print(f"  [Saved best model: {val_acc:.1f}%]")

        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics.get('loss'),
            'val_acc': val_metrics.get('accuracy'),
            'val_top3': val_metrics.get('top3_accuracy'),
            'val_top5': val_metrics.get('top5_accuracy'),
            'lr': current_lr
        })

    # Save training history
    history_path = args.output.replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {args.output}")
    print(f"History saved to: {history_path}")

    print("\n" + "=" * 60)
    print("Usage")
    print("=" * 60)
    print("""
# Load and use the trained model
from shannlp.spell.neural import SpellReranker

# Load model
model = SpellReranker.load('spell_reranker.pt')

# Use for reranking
error = "မိုင်း"
candidates = ["မိူင်း", "မိူင်", "မႂ်ႇ"]
context_left = "ၵူၼ်း"
context_right = "ၵိၼ်ၶဝ်ႈ"

best, scores = model.predict(error, candidates, context_left, context_right)
print(f"Best correction: {best}")
""")


if __name__ == "__main__":
    main()
