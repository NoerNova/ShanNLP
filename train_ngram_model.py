"""
Train n-gram language model from Shan text corpus.

This script reads Shan text files, trains a bigram or trigram model,
and saves it for use in context-aware spell correction.

Usage:
    python train_ngram_model.py --corpus_dir /path/to/texts --output model.pkl --ngram 2

Requirements:
    - Corpus directory with .txt files (UTF-8 encoded Shan text)
    - At least 1 MB of text recommended (more is better)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shannlp.spell.ngram import NgramModel
from shannlp.tokenize import word_tokenize


def load_corpus_from_directory(corpus_dir: str, file_extension: str = ".txt") -> List[str]:
    """
    Load all text files from directory.

    Args:
        corpus_dir: Path to directory containing text files
        file_extension: File extension to look for (default: .txt)

    Returns:
        List of text strings
    """
    corpus_path = Path(corpus_dir)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    texts = []
    file_count = 0

    print(f"Loading corpus from: {corpus_dir}")

    for file_path in corpus_path.glob(f"*{file_extension}"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    # Split into sentences/lines
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    texts.extend(lines)
                    file_count += 1
                    print(f"  Loaded: {file_path.name} ({len(lines)} lines)")
        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")
            continue

    print(f"\nTotal: {file_count} files, {len(texts)} text segments")
    return texts


def train_model(
    corpus_dir: str,
    output_path: str,
    ngram_order: int = 2,
    smoothing: float = 1.0,
    test_split: float = 0.1
):
    """
    Train n-gram model from corpus.

    Args:
        corpus_dir: Directory with text files
        output_path: Where to save trained model
        ngram_order: N-gram order (2=bigram, 3=trigram)
        smoothing: Laplace smoothing parameter
        test_split: Fraction of data to use for testing (0-1)
    """
    print("=" * 60)
    print(f"Training {ngram_order}-gram Model for Shan")
    print("=" * 60)
    print()

    # Load corpus
    texts = load_corpus_from_directory(corpus_dir)

    if not texts:
        print("Error: No texts found in corpus directory!")
        return

    # Split into train/test
    split_idx = int(len(texts) * (1 - test_split))
    train_texts = texts[:split_idx]
    test_texts = texts[split_idx:]

    print(f"\nTrain/test split:")
    print(f"  - Training texts: {len(train_texts)}")
    print(f"  - Test texts: {len(test_texts)}")
    print()

    # Create and train model
    model = NgramModel(n=ngram_order, smoothing=smoothing)

    # Define tokenization function
    def tokenize(text):
        return word_tokenize(text, engine="newmm", keep_whitespace=False)

    # Train
    print("Training model...")
    model.train(train_texts, tokenize_func=tokenize)
    print()

    # Evaluate on test set
    if test_texts:
        print("Evaluating on test set...")
        perplexity = model.perplexity(test_texts, tokenize_func=tokenize)
        print(f"Test perplexity: {perplexity:.2f}")
        print("(Lower perplexity = better model)")
        print()

    # Save model
    model.save(output_path)
    print()
    print(f"✓ Model saved to: {output_path}")
    print()

    # Example usage
    print("=" * 60)
    print("Example Usage:")
    print("=" * 60)
    print()
    print("```python")
    print("from shannlp.spell.context import ContextAwareCorrector")
    print()
    print("# Load model")
    print("corrector = ContextAwareCorrector()")
    print(f"corrector.load_model('{output_path}')")
    print()
    print("# Correct text")
    print('result = corrector.correct_sentence("your shan text here")')
    print("print(result)")
    print("```")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Train n-gram model for Shan spell correction"
    )

    parser.add_argument(
        "--corpus_dir",
        type=str,
        required=True,
        help="Directory containing Shan text files (.txt)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="shan_ngram_model.msgpack",
        help="Output path for trained model (default: shan_ngram_model.msgpack)"
    )

    parser.add_argument(
        "--ngram",
        type=int,
        choices=[2, 3],
        default=2,
        help="N-gram order: 2=bigram, 3=trigram (default: 2)"
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="Laplace smoothing parameter (default: 1.0)"
    )

    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Fraction of data for testing (default: 0.1)"
    )

    args = parser.parse_args()

    try:
        train_model(
            corpus_dir=args.corpus_dir,
            output_path=args.output,
            ngram_order=args.ngram,
            smoothing=args.smoothing,
            test_split=args.test_split
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
