#!/usr/bin/env python3
"""
Pre-tokenize raw Shan corpus for faster n-gram training.

This script tokenizes raw Shan text files and saves them in a format
that can be loaded much faster by train_ngram_model.py --tokenized.

The tokenization step is typically the slowest part of training (41+ min
for 139K texts). By pre-tokenizing once, you can reuse the tokenized
corpus for multiple training runs with different settings.

Usage:
    # Basic usage
    python tokenize_corpus.py --input ./data/corpus --output ./data/tokenized

    # With custom settings
    python tokenize_corpus.py --input ./raw_texts --output ./tokenized --workers 8

    # Then use for fast training
    python train_ngram_model.py --corpus_dir ./data/tokenized --tokenized --output model.msgpack

Output Format:
    One sentence per line, space-separated tokens:
        မႂ် သုင် ၶႃႈ
        ၵိၼ် ၶဝ်ႈ လီ ၼႃႇ
        လိၵ်ႈ လၢႆး တႆး

Performance:
    - Run ONCE, then train multiple times with --tokenized flag
    - Expected time: ~41 min for 139K texts (only once!)
    - Subsequent training: ~1-2 min (20-40x speedup)
"""

import argparse
import os
import sys
import time
import multiprocessing
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shannlp.tokenize import word_tokenize


def _tokenize_text(text: str) -> List[str]:
    """Tokenize a single text (for parallel processing)."""
    try:
        return word_tokenize(text, engine="newmm", keep_whitespace=False)
    except Exception:
        return []


def _load_single_file(file_path: Path) -> Tuple[str, List[str], Optional[str]]:
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


def load_raw_corpus(
    corpus_dir: str,
    file_extension: str = ".txt",
    verbose: bool = True,
    num_workers: int = 4
) -> Tuple[List[str], List[str]]:
    """
    Load raw text files from directory.

    Args:
        corpus_dir: Path to directory containing text files
        file_extension: File extension to look for
        verbose: Show detailed progress
        num_workers: Number of parallel file readers

    Returns:
        Tuple of (texts list, source filenames list)
    """
    corpus_path = Path(corpus_dir)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    files = list(corpus_path.glob(f"*{file_extension}"))
    total_files = len(files)

    if total_files == 0:
        print(f"No {file_extension} files found in {corpus_dir}")
        return [], []

    print(f"Loading raw corpus from: {corpus_dir}")
    print(f"Found {total_files} {file_extension} files")
    print()

    texts = []
    sources = []
    loaded_count = 0
    error_count = 0
    start_time = time.time()

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
                sources.extend([filename] * len(lines))
                if verbose and loaded_count % 10 == 0:
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

    return texts, sources


def parallel_tokenize(
    texts: List[str],
    num_workers: Optional[int] = None,
    show_progress: bool = True
) -> List[List[str]]:
    """
    Tokenize texts in parallel using multiprocessing.

    Args:
        texts: List of texts to tokenize
        num_workers: Number of parallel workers (default: CPU count - 1)
        show_progress: Show progress bar

    Returns:
        List of tokenized texts (list of token lists)
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    total_texts = len(texts)
    batch_size = max(100, total_texts // 100)  # At least 100 progress updates

    if show_progress:
        print(f"Tokenizing {total_texts:,} texts using {num_workers} workers...")

    start_time = time.time()
    tokenized = []

    with multiprocessing.Pool(processes=num_workers) as pool:
        for i in range(0, total_texts, batch_size):
            chunk = texts[i:i + batch_size]
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
                bar = "=" * filled + ">" + " " * (bar_width - filled - 1)
                if filled == bar_width:
                    bar = "=" * bar_width

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


def save_tokenized_corpus(
    tokenized_texts: List[List[str]],
    output_dir: str,
    texts_per_file: int = 10000,
    verbose: bool = True
):
    """
    Save tokenized texts to output directory.

    Args:
        tokenized_texts: List of token lists
        output_dir: Output directory path
        texts_per_file: Number of sentences per output file
        verbose: Show progress
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_texts = len(tokenized_texts)
    total_tokens = 0
    file_count = 0
    empty_count = 0

    print(f"Saving tokenized corpus to: {output_dir}")

    start_time = time.time()

    for i in range(0, total_texts, texts_per_file):
        chunk = tokenized_texts[i:i + texts_per_file]
        file_count += 1
        filename = output_path / f"tokenized_{file_count:04d}.txt"

        lines = []
        for tokens in chunk:
            if tokens:
                lines.append(" ".join(tokens))
                total_tokens += len(tokens)
            else:
                empty_count += 1

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        if verbose:
            progress = min(i + texts_per_file, total_texts) / total_texts * 100
            sys.stdout.write(f"\r  Saving: {progress:5.1f}% - {file_count} files written")
            sys.stdout.flush()

    elapsed = time.time() - start_time
    print()
    print(f"\nSave complete in {elapsed:.1f}s")
    print(f"  - Files created: {file_count}")
    print(f"  - Total sentences: {total_texts - empty_count:,}")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Empty sentences skipped: {empty_count:,}")
    if total_texts - empty_count > 0:
        print(f"  - Avg tokens/sentence: {total_tokens / (total_texts - empty_count):.1f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Pre-tokenize Shan corpus for faster n-gram training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python tokenize_corpus.py --input ./data/corpus --output ./data/tokenized

  # With more workers (faster)
  python tokenize_corpus.py --input ./raw --output ./tokenized --workers 12

  # Then train with pre-tokenized corpus (FAST!)
  python train_ngram_model.py --corpus_dir ./data/tokenized --tokenized --output model.msgpack

Performance comparison:
  - Standard training (with tokenization): ~75 min for 139K texts
  - Pre-tokenize once: ~41 min
  - Fast training (--tokenized): ~1-2 min
  - Total if training 3x with different settings: 41 + 3*2 = 47 min vs 3*75 = 225 min
"""
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing raw Shan text files (.txt)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for tokenized corpus"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for tokenization (default: CPU count - 1)"
    )

    parser.add_argument(
        "--texts-per-file",
        type=int,
        default=10000,
        help="Number of sentences per output file (default: 10000)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Pre-tokenize Shan Corpus")
    print("=" * 60)
    print()
    print("This script tokenizes raw text ONCE so you can train")
    print("n-gram models much faster using --tokenized flag.")
    print()

    overall_start = time.time()
    verbose = not args.quiet

    try:
        # Step 1: Load raw corpus
        print("-" * 60)
        print("Step 1/3: Loading raw corpus")
        print("-" * 60)
        texts, _ = load_raw_corpus(args.input, verbose=verbose)

        if not texts:
            print("Error: No texts found!")
            sys.exit(1)

        # Step 2: Tokenize
        print("-" * 60)
        print("Step 2/3: Tokenizing corpus")
        print("-" * 60)
        tokenized = parallel_tokenize(
            texts,
            num_workers=args.workers,
            show_progress=verbose
        )

        # Free memory
        del texts

        # Step 3: Save
        print("-" * 60)
        print("Step 3/3: Saving tokenized corpus")
        print("-" * 60)
        save_tokenized_corpus(
            tokenized,
            args.output,
            texts_per_file=args.texts_per_file,
            verbose=verbose
        )

        overall_elapsed = time.time() - overall_start

        print("=" * 60)
        print("Pre-tokenization Complete!")
        print("=" * 60)
        print()
        print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
        print()
        print("Next step - Fast training:")
        print(f"  python train_ngram_model.py --corpus_dir {args.output} --tokenized --output model.msgpack")
        print()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
