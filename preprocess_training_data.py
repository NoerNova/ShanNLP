#!/usr/bin/env python3
"""
Pre-generate candidates for training data to avoid bottleneck during training.

This script reads training JSONL files and adds pre-generated candidates,
so they don't need to be computed during training.

Uses multiprocessing for parallel candidate generation.
"""

import json
import sys
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shannlp.spell.core import spell_correct


def process_sample(line: str, max_candidates: int = 10) -> tuple:
    """
    Process a single sample to generate candidates.

    Args:
        line: JSON line from input file
        max_candidates: Maximum candidates to generate

    Returns:
        Tuple of (processed_sample_dict, success_flag, error_message)
    """
    line = line.strip()
    if not line:
        return None, False, "Empty line"

    try:
        sample = json.loads(line)
        error = sample.get('error', '')
        correct = sample.get('correct', '')

        if not error or not correct:
            return None, False, "Missing error or correct field"

        # Generate candidates if not already present
        if 'candidates' not in sample:
            try:
                suggestions = spell_correct(
                    error,
                    max_edit_distance=2,
                    max_suggestions=max_candidates - 1,
                    use_frequency=True,
                    use_phonetic=True
                )
                candidates = [word for word, score in suggestions]

                # Ensure correct is in candidates
                if correct not in candidates:
                    candidates.insert(0, correct)

                sample['candidates'] = candidates[:max_candidates]
            except Exception as e:
                # If candidate generation fails, just use the correct answer
                sample['candidates'] = [correct]
                return sample, True, f"Warning: {str(e)}"

        return sample, True, None

    except json.JSONDecodeError as e:
        return None, False, f"Invalid JSON: {str(e)}"


def preprocess_file(input_path: str, output_path: str, max_candidates: int = 10, num_workers: int | None = None):
    """
    Add pre-generated candidates to training data using parallel processing.

    Args:
        input_path: Input JSONL file
        output_path: Output JSONL file
        max_candidates: Maximum candidates to generate per sample
        num_workers: Number of parallel workers (default: CPU count)
    """
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()

    print(f"Processing {input_path}...")
    print(f"Using {num_workers} parallel workers")

    processed = 0
    skipped = 0
    warnings = []

    # Read all lines
    with open(input_path, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    # Process in parallel
    worker_func = partial(process_sample, max_candidates=max_candidates)

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(worker_func, lines),
            total=len(lines),
            desc="Generating candidates"
        ))

    # Write results
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for sample, success, error_msg in results:
            if success and sample:
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                processed += 1
                if error_msg:
                    warnings.append(error_msg)
            else:
                skipped += 1

    print(f"\nProcessed: {processed} samples")
    print(f"Skipped: {skipped} samples")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        # Show first 5 warnings
        for w in warnings[:5]:
            print(f"  - {w}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more")
    print(f"Output saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-generate candidates for training data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file (default: input with '_processed' suffix)"
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=10,
        help="Maximum candidates to generate"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process train/val/test files automatically"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.batch and not args.input:
        parser.error("--input is required when not using --batch mode")

    if args.batch:
        # Process all training files
        files = [
            'training_pairs_train.jsonl',
            'training_pairs_val.jsonl',
            'training_pairs_test.jsonl'
        ]

        for filename in files:
            if os.path.exists(filename):
                output = filename.replace('.jsonl', '_processed.jsonl')
                preprocess_file(filename, output, args.max_candidates, args.num_workers)
                print()
    else:
        output = args.output
        if output is None:
            output = args.input.replace('.jsonl', '_processed.jsonl')

        preprocess_file(args.input, output, args.max_candidates, args.num_workers)


if __name__ == "__main__":
    main()
