"""
Download and prepare Shan corpus from Hugging Face datasets.

This script downloads Shan language datasets from Hugging Face,
extracts the text, splits into sentences (one per line),
and saves them as .txt files for n-gram model training.

Usage:
    python download_corpus.py --output_dir data/corpus

Requirements:
    pip install datasets
"""

import argparse
import re
from pathlib import Path
from typing import List

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found.")
    print("Install it with: pip install datasets")
    exit(1)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using Shan sentence delimiters.

    Shan uses '။' (U+104B) as the main sentence delimiter,
    similar to Burmese/Myanmar script.

    Args:
        text: Input text

    Returns:
        List of sentences (stripped, non-empty)
    """
    if not text or not isinstance(text, str):
        return []

    # Split by Shan sentence delimiter and newlines
    # Myanmar/Shan sentence marker: ၊ (U+104A)
    # Myanmar/Shan sentence marker: ။ (U+104B)
    text = re.sub(r'\s+([။၊])', r'\1', text)
    sentences = re.split(r'(?<=[။၊\n])', text)

    # Clean and filter
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if s and len(s) > 3]  # Filter very short segments

    return sentences


def download_and_process_dataset(
    dataset_name: str,
    text_field: str,
    output_path: Path,
    max_samples: int | None = None
):
    """
    Download a Hugging Face dataset and save as sentence-per-line txt file.

    Args:
        dataset_name: Hugging Face dataset identifier
        text_field: Name of the field containing text
        output_path: Output file path
        max_samples: Maximum number of samples to process (None = all)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")

    try:
        # Load dataset
        print(f"Loading dataset from Hugging Face...")
        dataset = load_dataset(dataset_name, split='train')

        total_samples = len(dataset)
        print(f"Total samples in dataset: {total_samples}")

        if max_samples and max_samples < total_samples:
            print(f"Limiting to first {max_samples} samples")
            dataset = dataset.select(range(max_samples))

        # Process texts
        all_sentences = []
        skipped = 0

        print(f"Extracting text from '{text_field}' field...")

        for i, example in enumerate(dataset):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} samples...")

            text = example[text_field] if isinstance(example, dict) else getattr(example, text_field, "")

            if not text:
                skipped += 1
                continue

            # Split into sentences
            sentences = split_into_sentences(text)
            all_sentences.extend(sentences)

        # Save to file
        print(f"\nSaving {len(all_sentences)} sentences to {output_path.name}...")

        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence in all_sentences:
                f.write(sentence + '\n')

        print(f"✓ Saved: {output_path}")
        print(f"  - Total sentences: {len(all_sentences)}")
        print(f"  - Skipped empty: {skipped}")

    except Exception as e:
        print(f"✗ Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare Shan corpus from Hugging Face"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/corpus",
        help="Output directory for corpus files (default: data/corpus)"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (default: all)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Shan Corpus Download and Preparation")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print()

    # Dataset configurations: (name, text_field, output_filename)
    datasets_config = [
        ("NorHsangPha/shan-wiki-dataset", "text", "shan_wiki.txt"),
        ("NorHsangPha/shan-news-shannews_org", "content", "shan_news.txt"),
        ("NorHsangPha/shan-novel-tainovel_com", "content", "shan_novel.txt"),
    ]

    # Download and process each dataset
    for dataset_name, text_field, output_filename in datasets_config:
        output_path = output_dir / output_filename
        download_and_process_dataset(
            dataset_name=dataset_name,
            text_field=text_field,
            output_path=output_path,
            max_samples=args.max_samples
        )

    print("\n" + "="*60)
    print("✓ Corpus preparation complete!")
    print("="*60)
    print()
    print(f"Corpus files saved in: {output_dir}")
    print()
    print("Next steps:")
    print(f"  python train_ngram_model.py --corpus_dir {output_dir} --output shan_ngram_model.msgpack")
    print()


if __name__ == "__main__":
    main()
