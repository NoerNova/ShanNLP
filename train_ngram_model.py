"""
Train n-gram language model from Shan text corpus.

This script reads Shan text files, trains a bigram or trigram model,
and saves it for use in context-aware spell correction.

Usage:
    # Standard training (tokenizes during training - slower)
    python train_ngram_model.py --corpus_dir /path/to/texts --output model.msgpack --ngram 2

    # Fast training with pre-tokenized corpus (skip tokenization - much faster)
    python train_ngram_model.py --corpus_dir /path/to/tokenized --output model.msgpack --tokenized

Requirements:
    - For raw corpus: Directory with .txt files (UTF-8 encoded Shan text)
    - For pre-tokenized: Directory with .txt files (one sentence per line, space-separated tokens)
    - At least 1 MB of text recommended (more is better)

Performance Tips:
    - Use --tokenized flag with pre-tokenized corpus for 20-40x speedup
    - Pre-tokenize once with tokenize_corpus.py, then train multiple times
    - Larger batch_size uses more memory but is faster
    - Use --verbose for detailed logging
"""

import argparse
import os
import sys
import time
import gc
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shannlp.spell.ngram import NgramModel
from shannlp.tokenize import word_tokenize
import multiprocessing


def _tokenize_text(text: str) -> List[str]:
    """Tokenize a single text (for parallel processing)."""
    try:
        return word_tokenize(text, engine="newmm", keep_whitespace=False)
    except:
        return []


def parallel_tokenize(
    texts: List[str],
    num_workers: Optional[int] = None,
    batch_size: int = 100,
    show_progress: bool = True
) -> List[List[str]]:
    """
    Tokenize texts in parallel using multiprocessing.

    Args:
        texts: List of texts to tokenize
        num_workers: Number of parallel workers (default: CPU count - 1)
        batch_size: Texts per progress update
        show_progress: Show progress bar

    Returns:
        List of tokenized texts (list of token lists)
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    total_texts = len(texts)

    if show_progress:
        print(f"Tokenizing {total_texts:,} texts using {num_workers} workers...")

    start_time = time.time()
    tokenized = []

    # Use multiprocessing Pool for CPU-bound tokenization
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Process in chunks for progress updates
        chunk_size = max(batch_size, total_texts // 100)  # At least 100 updates

        for i in range(0, total_texts, chunk_size):
            chunk = texts[i:i + chunk_size]
            chunk_results = pool.map(_tokenize_text, chunk)
            tokenized.extend(chunk_results)

            if show_progress:
                progress = len(tokenized) / total_texts * 100
                elapsed = time.time() - start_time
                rate = len(tokenized) / elapsed if elapsed > 0 else 0

                if rate > 0:
                    eta = (total_texts - len(tokenized)) / rate
                    eta_str = f"{eta:.0f}s" if eta < 60 else f"{eta/60:.1f}m"
                else:
                    eta_str = "..."

                bar_width = 30
                filled = int(bar_width * len(tokenized) / total_texts)
                bar = "█" * filled + "░" * (bar_width - filled)

                sys.stdout.write(f"\r[{bar}] {progress:5.1f}% | "
                               f"{len(tokenized):,}/{total_texts:,} | "
                               f"{rate:,.0f} texts/s | "
                               f"ETA: {eta_str}")
                sys.stdout.flush()

    elapsed = time.time() - start_time
    if show_progress:
        print()
        print(f"Tokenization complete in {elapsed:.1f}s ({total_texts/elapsed:,.0f} texts/s)")
        print()

    return tokenized


def _load_single_file(file_path: Path) -> tuple:
    """Load a single file and return (filename, lines, error)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                return (file_path.name, lines, None)
            return (file_path.name, [], None)
    except Exception as e:
        return (file_path.name, [], str(e))


def load_corpus_from_directory(
    corpus_dir: str,
    file_extension: str = ".txt",
    verbose: bool = True,
    num_workers: int = 4
) -> List[str]:
    """
    Load all text files from directory with parallel I/O.

    Args:
        corpus_dir: Path to directory containing text files
        file_extension: File extension to look for (default: .txt)
        verbose: Show detailed progress
        num_workers: Number of parallel file readers

    Returns:
        List of text strings
    """
    corpus_path = Path(corpus_dir)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    # Get all files
    files = list(corpus_path.glob(f"*{file_extension}"))
    total_files = len(files)

    if total_files == 0:
        print(f"No {file_extension} files found in {corpus_dir}")
        return []

    print(f"Loading corpus from: {corpus_dir}")
    print(f"Found {total_files} {file_extension} files")
    print()

    texts = []
    loaded_count = 0
    error_count = 0
    total_bytes = 0
    start_time = time.time()

    # Use thread pool for parallel file I/O
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_load_single_file, f): f for f in files}

        for future in as_completed(futures):
            filename, lines, error = future.result()
            loaded_count += 1

            if error:
                error_count += 1
                if verbose:
                    print(f"  Error loading {filename}: {error}")
            elif lines:
                texts.extend(lines)
                if verbose and loaded_count % 10 == 0:
                    elapsed = time.time() - start_time
                    progress = (loaded_count / total_files) * 100
                    sys.stdout.write(f"\r  Loading files: {loaded_count}/{total_files} ({progress:.1f}%) - {len(texts):,} lines")
                    sys.stdout.flush()

    elapsed = time.time() - start_time
    print()
    print(f"\nCorpus loading complete in {elapsed:.1f}s")
    print(f"  - Files loaded: {loaded_count - error_count}/{total_files}")
    if error_count > 0:
        print(f"  - Files with errors: {error_count}")
    print(f"  - Total text segments: {len(texts):,}")
    print()

    return texts


def load_pretokenized_corpus(
    corpus_dir: str,
    file_extension: str = ".txt",
    verbose: bool = True,
    num_workers: int = 4
) -> List[List[str]]:
    """
    Load pre-tokenized corpus where each line contains space-separated tokens.

    This is much faster than load_corpus_from_directory() + tokenization
    because it skips the expensive tokenization step entirely.

    Expected format (one sentence per line, space-separated tokens):
        မႂ် သုင် ၶႃႈ
        ၵိၼ် ၶဝ်ႈ လီ ၼႃႇ
        လိၵ်ႈ လၢႆး တႆး

    Args:
        corpus_dir: Path to directory containing pre-tokenized text files
        file_extension: File extension to look for (default: .txt)
        verbose: Show detailed progress
        num_workers: Number of parallel file readers

    Returns:
        List of token lists (already tokenized sentences)
    """
    corpus_path = Path(corpus_dir)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    # Get all files
    files = list(corpus_path.glob(f"*{file_extension}"))
    total_files = len(files)

    if total_files == 0:
        print(f"No {file_extension} files found in {corpus_dir}")
        return []

    print(f"Loading pre-tokenized corpus from: {corpus_dir}")
    print(f"Found {total_files} {file_extension} files")
    print()

    tokenized_texts = []
    loaded_count = 0
    error_count = 0
    total_lines = 0
    start_time = time.time()

    # Use thread pool for parallel file I/O
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_load_single_file, f): f for f in files}

        for future in as_completed(futures):
            filename, lines, error = future.result()
            loaded_count += 1

            if error:
                error_count += 1
                if verbose:
                    print(f"  Error loading {filename}: {error}")
            elif lines:
                # Each line is already tokenized: "token1 token2 token3"
                for line in lines:
                    tokens = line.split()  # Split on whitespace
                    if tokens:
                        tokenized_texts.append(tokens)
                        total_lines += 1

                if verbose and loaded_count % 10 == 0:
                    elapsed = time.time() - start_time
                    progress = (loaded_count / total_files) * 100
                    sys.stdout.write(f"\r  Loading files: {loaded_count}/{total_files} ({progress:.1f}%) - {total_lines:,} sentences")
                    sys.stdout.flush()

    elapsed = time.time() - start_time
    print()
    print(f"\nPre-tokenized corpus loading complete in {elapsed:.1f}s")
    print(f"  - Files loaded: {loaded_count - error_count}/{total_files}")
    if error_count > 0:
        print(f"  - Files with errors: {error_count}")
    print(f"  - Total sentences: {len(tokenized_texts):,}")

    # Calculate average tokens per sentence
    total_tokens = sum(len(tokens) for tokens in tokenized_texts)
    avg_tokens = total_tokens / len(tokenized_texts) if tokenized_texts else 0
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Avg tokens/sentence: {avg_tokens:.1f}")
    print()

    return tokenized_texts


def train_model(
    corpus_dir: str,
    output_path: str,
    ngram_order: int = 2,
    smoothing: float = 1.0,
    test_split: float = 0.1,
    batch_size: int = 1000,
    verbose: bool = True,
    num_workers: Optional[int] = None,
    use_parallel: bool = True,
    min_count: int = 2,
    tokenized: bool = False
):
    """
    Train n-gram model from corpus.

    Args:
        corpus_dir: Directory with text files
        output_path: Where to save trained model
        ngram_order: N-gram order (2=bigram, 3=trigram)
        smoothing: Laplace smoothing parameter
        test_split: Fraction of data to use for testing (0-1)
        batch_size: Number of texts per training batch
        verbose: Show detailed progress
        num_workers: Number of parallel workers for tokenization
        use_parallel: Use parallel tokenization (faster)
        min_count: Minimum word count to keep (prune rare words)
        tokenized: If True, corpus is pre-tokenized (space-separated tokens per line)
    """
    overall_start = time.time()

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    print("=" * 60)
    print(f"Training {ngram_order}-gram Model for Shan")
    print("=" * 60)
    print()
    print(f"Configuration:")
    print(f"  - N-gram order: {ngram_order}")
    print(f"  - Smoothing: {smoothing}")
    print(f"  - Test split: {test_split * 100:.0f}%")
    print(f"  - Batch size: {batch_size:,}")
    print(f"  - Parallel workers: {num_workers}")
    if tokenized:
        print(f"  - Corpus type: PRE-TOKENIZED (fast mode)")
    else:
        print(f"  - Corpus type: raw (will tokenize during training)")
        print(f"  - Parallel tokenization: {'enabled' if use_parallel else 'disabled'}")
    print(f"  - Min word count: {min_count} (prune rare words)")
    print()

    # Load corpus (different paths for raw vs pre-tokenized)
    if tokenized:
        # FAST PATH: Load pre-tokenized corpus (skip tokenization entirely)
        print("-" * 60)
        print("Step 1/4: Loading pre-tokenized corpus")
        print("-" * 60)
        all_tokenized = load_pretokenized_corpus(corpus_dir, verbose=verbose)

        if not all_tokenized:
            print("Error: No tokenized texts found in corpus directory!")
            return

        # Split into train/test
        split_idx = int(len(all_tokenized) * (1 - test_split))
        train_tokenized = all_tokenized[:split_idx]
        test_tokenized = all_tokenized[split_idx:]

        print("-" * 60)
        print("Step 2/4: Data split")
        print("-" * 60)
        print(f"  - Training sentences: {len(train_tokenized):,}")
        print(f"  - Test sentences: {len(test_tokenized):,}")
        print()

        # Convert test tokenized back to strings for perplexity evaluation
        test_texts = [" ".join(tokens) for tokens in test_tokenized]

        # Free memory
        del all_tokenized
        del test_tokenized
        gc.collect()

        tokenize_elapsed = 0.0  # No tokenization needed
        print("(Tokenization skipped - corpus is pre-tokenized)")
        print()

    else:
        # STANDARD PATH: Load raw corpus and tokenize
        print("-" * 60)
        print("Step 1/5: Loading corpus")
        print("-" * 60)
        texts = load_corpus_from_directory(corpus_dir, verbose=verbose)

        if not texts:
            print("Error: No texts found in corpus directory!")
            return

        # Split into train/test
        split_idx = int(len(texts) * (1 - test_split))
        train_texts = texts[:split_idx]
        test_texts = texts[split_idx:]

        print("-" * 60)
        print("Step 2/5: Data split")
        print("-" * 60)
        print(f"  - Training texts: {len(train_texts):,}")
        print(f"  - Test texts: {len(test_texts):,}")
        print()

        # Free memory
        del texts
        gc.collect()

        # Pre-tokenize training data in parallel (this is the slow step)
        print("-" * 60)
        print("Step 3/5: Tokenizing training data")
        print("-" * 60)
        tokenize_start = time.time()

        if use_parallel and len(train_texts) > 1000:
            # Use parallel tokenization for large datasets
            train_tokenized = parallel_tokenize(
                train_texts,
                num_workers=num_workers,
                show_progress=verbose
            )
        else:
            # Sequential tokenization for small datasets
            print(f"Tokenizing {len(train_texts):,} texts sequentially...")
            train_tokenized = []
            for i, text in enumerate(train_texts):
                tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
                train_tokenized.append(tokens)
                if verbose and (i + 1) % 1000 == 0:
                    print(f"  Tokenized {i + 1:,}/{len(train_texts):,} texts")
            print()

        tokenize_elapsed = time.time() - tokenize_start
        print(f"Tokenization time: {tokenize_elapsed:.1f}s")
        print()

        # Free original texts
        del train_texts
        gc.collect()

    # Create and train model from pre-tokenized data
    model = NgramModel(n=ngram_order, smoothing=smoothing)

    # Determine step numbers based on path (tokenized uses fewer steps)
    if tokenized:
        build_step = "3/4"
        prune_step = "4/4"
        eval_step = "Evaluation"  # Bonus step
    else:
        build_step = "4/6"
        prune_step = "5/6"
        eval_step = "6/6"

    print("-" * 60)
    print(f"Step {build_step}: Building n-gram model")
    print("-" * 60)
    train_start = time.time()
    model.train_from_tokens(train_tokenized, batch_size=batch_size, show_progress=verbose)
    train_elapsed = time.time() - train_start

    # Free tokenized data
    del train_tokenized
    gc.collect()

    # Prune vocabulary (important for reducing perplexity)
    print("-" * 60)
    print(f"Step {prune_step}: Pruning rare words")
    print("-" * 60)
    prune_start = time.time()
    pruned_count = model.prune_vocabulary(min_count=min_count, show_progress=verbose)
    prune_elapsed = time.time() - prune_start

    # Evaluate on test set
    if test_texts:
        print("-" * 60)
        print(f"Step {eval_step}: Evaluating on test set")
        print("-" * 60)
        eval_start = time.time()
        print(f"Evaluating on {len(test_texts):,} test texts...")
        print()

        def tokenize(text):
            return word_tokenize(text, engine="newmm", keep_whitespace=False)

        perplexity = model.perplexity(
            test_texts,
            tokenize_func=tokenize,
            show_progress=verbose
        )
        eval_elapsed = time.time() - eval_start

        print()
        print(f"  - Final perplexity: {perplexity:.2f}")
        print(f"  - Evaluation time: {eval_elapsed:.1f}s")
        print()
        print("Perplexity guide:")
        print("  < 50: Excellent model")
        print("  50-100: Good model")
        print("  100-200: Acceptable model")
        print("  > 200: Need more training data")
        print()
    else:
        print("-" * 60)
        print(f"Step {eval_step}: Skipping evaluation (no test data)")
        print("-" * 60)
        print()

    # Save model
    print("-" * 60)
    print("Saving model")
    print("-" * 60)
    model.save(output_path)

    # Final summary
    overall_elapsed = time.time() - overall_start
    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print()
    print(f"  Model saved to: {output_path}")
    print(f"  Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print(f"    - Tokenization: {tokenize_elapsed:.1f}s")
    print(f"    - N-gram building: {train_elapsed:.1f}s")
    print()
    print("Model Statistics:")
    print(f"  - Vocabulary size: {len(model.vocabulary):,}")
    print(f"  - Unique contexts: {len(model.context_counts):,}")
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
        description="Train n-gram model for Shan spell correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training (raw corpus - tokenizes during training)
  python train_ngram_model.py --corpus_dir ./my_corpus --output bigram.msgpack --ngram 2

  # FAST training with pre-tokenized corpus (recommended for large datasets)
  # Step 1: Pre-tokenize raw corpus ONCE
  python tokenize_corpus.py --input ./my_corpus --output ./tokenized_corpus

  # Step 2: Fast training (20-40x speedup!)
  python train_ngram_model.py --corpus_dir ./tokenized_corpus --output model.msgpack --tokenized

  # Train trigram model (more accurate)
  python train_ngram_model.py --corpus_dir ./my_corpus --output trigram.msgpack --ngram 3

  # Fast training with larger batch size
  python train_ngram_model.py --corpus_dir ./my_corpus --batch_size 5000

  # Quiet mode (less output)
  python train_ngram_model.py --corpus_dir ./my_corpus --quiet
"""
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

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Texts per batch for training progress updates (default: 1000)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for tokenization (default: CPU count - 1)"
    )

    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel tokenization (use for debugging)"
    )

    parser.add_argument(
        "--tokenized",
        action="store_true",
        help="Corpus is already tokenized (space-separated tokens per line). "
             "MUCH faster - skips tokenization step entirely. "
             "Use tokenize_corpus.py to pre-tokenize raw corpus first."
    )

    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum word count to keep (prune rare words, default: 2)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    try:
        train_model(
            corpus_dir=args.corpus_dir,
            output_path=args.output,
            ngram_order=args.ngram,
            smoothing=args.smoothing,
            test_split=args.test_split,
            batch_size=args.batch_size,
            verbose=not args.quiet,
            num_workers=args.workers,
            use_parallel=not args.no_parallel,
            min_count=args.min_count,
            tokenized=args.tokenized
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
