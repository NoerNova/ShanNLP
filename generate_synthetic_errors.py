#!/usr/bin/env python3
"""
Generate synthetic spelling errors for training neural reranker.

This script takes correct Shan text and introduces realistic spelling errors
to create training pairs for the Phase 3 neural spell correction model.

Error types generated:
1. Tone mark substitution (ႇ ↔ ႈ ↔ း ↔ ႉ ↔ ႊ)
2. Vowel substitution (similar vowels)
3. Phonetically similar consonant substitution
4. Character deletion
5. Character insertion
6. Character transposition
7. Lead vowel transposition (common Shan typing error)

Usage:
    python generate_synthetic_errors.py \
        --corpus_dir ./data/corpus \
        --output ./data/training_pairs.jsonl \
        --num_pairs 50000

Output format (JSONL):
    {"error": "မိုင်း", "correct": "မိူင်း", "context_left": "ၵူၼ်း", "context_right": "ၵိၼ်ၶဝ်ႈ", "error_type": "vowel_sub"}
"""

import argparse
import json
import os
import sys
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shannlp.tokenize import word_tokenize
from shannlp.corpus import shan_words

# Shan character categories
CONSONANTS = "ၵၷၶꧠငၸၹသၺတၻထၼꧣပၽၾၿႀမယရလꩮဝႁဢ"
VOWELS_LEAD = "\u1084\u1031\u103c"  # ႄ, ေ, ြ (appear before consonant visually)
VOWELS_FOLLOW = "\u1083\u1062\u103b"  # ႃ, ၢ, ျ
VOWELS_ABOVE = "\u1085\u1035\u102d\u102e\u1086\u103a"  # ႅ, ဵ, ိ, ီ, ႆ, ်
VOWELS_BELOW = "\u102f\u1030\u1082\u103d"  # ု, ူ, ႂ, ွ
ALL_VOWELS = VOWELS_LEAD + VOWELS_FOLLOW + VOWELS_ABOVE + VOWELS_BELOW
TONE_MARKS = "\u1087\u1088\u1038\u1089\u108a"  # ႇ, ႈ, း, ႉ, ႊ
ASAT = "\u103a"  # ်

# Phonetically similar consonant groups
PHONETIC_GROUPS = [
    list("ၵၷၶꧠ"),      # k-sounds
    list("ၸၹသၺ"),      # s/sh-sounds
    list("တၻထ"),        # t-sounds
    list("ပၽၾၿ"),      # p-sounds
    list("မႀ"),         # m-sounds
    list("ယရ"),         # y/r-sounds
    list("လꩮ"),         # l-sounds
    list("ဝႁ"),         # w/h-sounds
]

# Similar vowel groups
VOWEL_SIMILARITY = [
    ["\u102d", "\u102e"],  # ိ, ီ
    ["\u102f", "\u1030"],  # ု, ူ
    ["\u1031", "\u1084"],  # ေ, ႄ
    ["\u1083", "\u1062"],  # ႃ, ၢ
]

# Similar tone mark groups
TONE_SIMILARITY = [
    ["\u1087", "\u1088"],  # ႇ, ႈ
    ["\u1088", "\u1038"],  # ႈ, း
    ["\u1089", "\u108a"],  # ႉ, ႊ
]


def get_char_type(char: str) -> str:
    """Classify a character."""
    if char in CONSONANTS:
        return "consonant"
    elif char in VOWELS_LEAD:
        return "vowel_lead"
    elif char in VOWELS_FOLLOW:
        return "vowel_follow"
    elif char in VOWELS_ABOVE:
        return "vowel_above"
    elif char in VOWELS_BELOW:
        return "vowel_below"
    elif char in TONE_MARKS:
        return "tone"
    elif char == ASAT:
        return "asat"
    else:
        return "other"


def get_similar_char(char: str, groups: List[List[str]]) -> Optional[str]:
    """Get a similar character from the same group."""
    for group in groups:
        if char in group:
            others = [c for c in group if c != char]
            if others:
                return random.choice(others)
    return None


def introduce_error(word: str, error_rate: float = 0.3) -> Tuple[Optional[str], Optional[str]]:
    """
    Introduce a spelling error into a word.

    Args:
        word: Correct word
        error_rate: Probability of each error type

    Returns:
        Tuple of (error_word, error_type) or (None, None) if no error introduced
    """
    if len(word) < 2:
        return None, None

    chars = list(word)
    error_type = None

    # Choose error type based on word structure
    error_types = []

    # Check what's available in the word
    has_tone = any(c in TONE_MARKS for c in chars)
    has_vowel = any(c in ALL_VOWELS for c in chars)
    has_lead_vowel = any(c in VOWELS_LEAD for c in chars)
    has_consonant = any(c in CONSONANTS for c in chars)

    # Build weighted error type list
    if has_tone:
        error_types.extend(["tone_sub"] * 25)  # 25% weight
    if has_vowel:
        error_types.extend(["vowel_sub"] * 20)  # 20% weight
    if has_lead_vowel and has_consonant:
        error_types.extend(["lead_vowel_transpose"] * 15)  # 15% weight
    if has_consonant:
        error_types.extend(["consonant_sub"] * 15)  # 15% weight

    error_types.extend(["delete"] * 10)  # 10% weight
    error_types.extend(["insert"] * 5)   # 5% weight
    error_types.extend(["transpose"] * 10)  # 10% weight

    if not error_types:
        return None, None

    error_type = random.choice(error_types)

    try:
        if error_type == "tone_sub":
            # Substitute tone mark with similar one
            tone_indices = [i for i, c in enumerate(chars) if c in TONE_MARKS]
            if tone_indices:
                idx = random.choice(tone_indices)
                similar = get_similar_char(chars[idx], TONE_SIMILARITY)
                if similar:
                    chars[idx] = similar
                else:
                    # Random tone
                    chars[idx] = random.choice(TONE_MARKS)

        elif error_type == "vowel_sub":
            # Substitute vowel with similar one
            vowel_indices = [i for i, c in enumerate(chars) if c in ALL_VOWELS]
            if vowel_indices:
                idx = random.choice(vowel_indices)
                similar = get_similar_char(chars[idx], VOWEL_SIMILARITY)
                if similar:
                    chars[idx] = similar
                else:
                    # Random vowel from same position type
                    char_type = get_char_type(chars[idx])
                    if char_type == "vowel_above":
                        chars[idx] = random.choice(VOWELS_ABOVE)
                    elif char_type == "vowel_below":
                        chars[idx] = random.choice(VOWELS_BELOW)
                    elif char_type == "vowel_lead":
                        chars[idx] = random.choice(VOWELS_LEAD)
                    elif char_type == "vowel_follow":
                        chars[idx] = random.choice(VOWELS_FOLLOW)

        elif error_type == "lead_vowel_transpose":
            # Common error: lead vowel typed before consonant
            # Find consonant + lead_vowel pattern and swap
            for i in range(len(chars) - 1):
                if chars[i] in CONSONANTS and chars[i + 1] in VOWELS_LEAD:
                    # Swap: ၶေ → ေၶ (wrong order)
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
                    break

        elif error_type == "consonant_sub":
            # Substitute consonant with phonetically similar one
            consonant_indices = [i for i, c in enumerate(chars) if c in CONSONANTS]
            if consonant_indices:
                idx = random.choice(consonant_indices)
                similar = get_similar_char(chars[idx], PHONETIC_GROUPS)
                if similar:
                    chars[idx] = similar
                else:
                    return None, None

        elif error_type == "delete":
            # Delete a random character (not first consonant)
            if len(chars) > 2:
                # Prefer deleting vowels or tone marks
                deletable = [i for i in range(1, len(chars))
                            if chars[i] in ALL_VOWELS + TONE_MARKS]
                if not deletable:
                    deletable = list(range(1, len(chars)))
                if deletable:
                    idx = random.choice(deletable)
                    chars.pop(idx)

        elif error_type == "insert":
            # Insert a random character
            insert_char = random.choice(ALL_VOWELS + TONE_MARKS)
            insert_pos = random.randint(1, len(chars))
            chars.insert(insert_pos, insert_char)

        elif error_type == "transpose":
            # Transpose two adjacent characters
            if len(chars) > 2:
                idx = random.randint(0, len(chars) - 2)
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

        error_word = "".join(chars)

        # Verify error is different from original
        if error_word != word and len(error_word) > 0:
            return error_word, error_type

    except Exception:
        pass

    return None, None


def load_corpus_sentences(corpus_dir: str, max_files: Optional[int] = None) -> List[str]:
    """Load sentences from corpus directory."""
    corpus_path = Path(corpus_dir)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    files = list(corpus_path.glob("*.txt"))
    if max_files:
        files = files[:max_files]

    sentences = []

    print(f"Loading corpus from {corpus_dir}...")
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                sentences.extend(lines)
        except Exception as e:
            print(f"  Error reading {file_path.name}: {e}")

    print(f"  Loaded {len(sentences):,} sentences from {len(files)} files")
    return sentences


def generate_training_pairs(
    sentences: List[str],
    num_pairs: int,
    dictionary: Set[str],
    error_rate_per_word: float = 0.3,
    min_word_length: int = 2,
    show_progress: bool = True
) -> List[Dict]:
    """
    Generate training pairs from sentences.

    Args:
        sentences: List of correct sentences
        num_pairs: Target number of training pairs to generate
        dictionary: Set of valid words
        error_rate_per_word: Probability of introducing error per word
        min_word_length: Minimum word length to consider
        show_progress: Show progress bar

    Returns:
        List of training pair dictionaries
    """
    pairs = []
    error_type_counts = defaultdict(int)

    # Shuffle sentences for variety
    random.shuffle(sentences)

    start_time = time.time()
    sentences_processed = 0

    print(f"\nGenerating {num_pairs:,} training pairs...")

    for sentence in sentences:
        if len(pairs) >= num_pairs:
            break

        # Tokenize sentence
        try:
            tokens = word_tokenize(sentence, engine="newmm", keep_whitespace=False)
        except Exception:
            continue

        if len(tokens) < 2:
            continue

        # Process each word
        for i, word in enumerate(tokens):
            if len(pairs) >= num_pairs:
                break

            # Skip short words or words not in dictionary
            if len(word) < min_word_length:
                continue
            if word not in dictionary:
                continue

            # Randomly decide whether to create error for this word
            if random.random() > error_rate_per_word:
                continue

            # Introduce error
            error_word, error_type = introduce_error(word)

            if error_word and error_word != word:
                # Get context
                context_left = " ".join(tokens[max(0, i-2):i])
                context_right = " ".join(tokens[i+1:min(len(tokens), i+3)])

                pair = {
                    "error": error_word,
                    "correct": word,
                    "context_left": context_left,
                    "context_right": context_right,
                    "error_type": error_type
                }

                pairs.append(pair)
                error_type_counts[error_type] += 1

        sentences_processed += 1

        # Progress update
        if show_progress and sentences_processed % 5000 == 0:
            elapsed = time.time() - start_time
            rate = len(pairs) / elapsed if elapsed > 0 else 0
            progress = len(pairs) / num_pairs * 100

            bar_width = 30
            filled = int(bar_width * len(pairs) / num_pairs)
            bar = "=" * filled + ">" + " " * (bar_width - filled - 1)

            sys.stdout.write(f"\r[{bar}] {progress:5.1f}% | "
                           f"{len(pairs):,}/{num_pairs:,} pairs | "
                           f"{rate:.0f} pairs/s")
            sys.stdout.flush()

    if show_progress:
        print()

    elapsed = time.time() - start_time
    print(f"\nGeneration complete in {elapsed:.1f}s")
    print(f"  - Sentences processed: {sentences_processed:,}")
    print(f"  - Pairs generated: {len(pairs):,}")
    print(f"\nError type distribution:")
    for error_type, count in sorted(error_type_counts.items(), key=lambda x: -x[1]):
        pct = count / len(pairs) * 100 if pairs else 0
        print(f"  - {error_type}: {count:,} ({pct:.1f}%)")

    return pairs


def save_pairs_jsonl(pairs: List[Dict], output_path: str):
    """Save pairs to JSONL file."""
    print(f"\nSaving {len(pairs):,} pairs to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved ({file_size:.2f} MB)")


def save_pairs_tsv(pairs: List[Dict], output_path: str):
    """Save pairs to TSV file."""
    tsv_path = output_path.replace('.jsonl', '.tsv')
    print(f"Saving TSV format to {tsv_path}...")

    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write("error\tcorrect\tcontext_left\tcontext_right\terror_type\n")
        for pair in pairs:
            f.write(f"{pair['error']}\t{pair['correct']}\t"
                   f"{pair['context_left']}\t{pair['context_right']}\t"
                   f"{pair['error_type']}\n")

    print(f"  Saved TSV format")


def split_train_val_test(
    pairs: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split pairs into train/val/test sets."""
    random.shuffle(pairs)

    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = pairs[:train_end]
    val = pairs[train_end:val_end]
    test = pairs[val_end:]

    return train, val, test


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic spelling errors for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 50K training pairs
  python generate_synthetic_errors.py --corpus_dir ./data/corpus --num_pairs 50000

  # Generate with custom output path
  python generate_synthetic_errors.py --corpus_dir ./data/corpus --output ./training_data.jsonl

  # Generate more pairs with higher error rate
  python generate_synthetic_errors.py --corpus_dir ./data/corpus --num_pairs 100000 --error_rate 0.5
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
        default="training_pairs.jsonl",
        help="Output file path (default: training_pairs.jsonl)"
    )

    parser.add_argument(
        "--num_pairs",
        type=int,
        default=50000,
        help="Number of training pairs to generate (default: 50000)"
    )

    parser.add_argument(
        "--error_rate",
        type=float,
        default=0.3,
        help="Error rate per word (default: 0.3)"
    )

    parser.add_argument(
        "--split",
        action="store_true",
        help="Split into train/val/test files (80/10/10)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print("=" * 60)
    print("Synthetic Error Generation for Shan Spell Correction")
    print("=" * 60)
    print()
    print(f"Configuration:")
    print(f"  - Corpus directory: {args.corpus_dir}")
    print(f"  - Output file: {args.output}")
    print(f"  - Target pairs: {args.num_pairs:,}")
    print(f"  - Error rate: {args.error_rate}")
    print(f"  - Random seed: {args.seed}")
    print()

    try:
        # Load corpus
        sentences = load_corpus_sentences(args.corpus_dir)

        if not sentences:
            print("Error: No sentences found in corpus!")
            sys.exit(1)

        # Load dictionary
        print("\nLoading dictionary...")
        dictionary = set(shan_words())
        print(f"  Dictionary size: {len(dictionary):,} words")

        # Generate pairs
        pairs = generate_training_pairs(
            sentences=sentences,
            num_pairs=args.num_pairs,
            dictionary=dictionary,
            error_rate_per_word=args.error_rate,
            show_progress=True
        )

        if not pairs:
            print("Error: No training pairs generated!")
            sys.exit(1)

        # Save pairs
        if args.split:
            # Split into train/val/test
            train, val, test = split_train_val_test(pairs)

            base_path = args.output.replace('.jsonl', '')

            save_pairs_jsonl(train, f"{base_path}_train.jsonl")
            save_pairs_jsonl(val, f"{base_path}_val.jsonl")
            save_pairs_jsonl(test, f"{base_path}_test.jsonl")

            print(f"\nSplit sizes:")
            print(f"  - Train: {len(train):,}")
            print(f"  - Val: {len(val):,}")
            print(f"  - Test: {len(test):,}")
        else:
            save_pairs_jsonl(pairs, args.output)
            save_pairs_tsv(pairs, args.output)

        # Show examples
        print("\n" + "=" * 60)
        print("Sample generated pairs:")
        print("=" * 60)
        for pair in random.sample(pairs, min(5, len(pairs))):
            print(f"\n  Error: {pair['error']}")
            print(f"  Correct: {pair['correct']}")
            print(f"  Context: [{pair['context_left']}] ___ [{pair['context_right']}]")
            print(f"  Type: {pair['error_type']}")

        print("\n" + "=" * 60)
        print("Generation Complete!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"  1. Review generated pairs for quality")
        print(f"  2. Add real error examples to {args.output}")
        print(f"  3. Train neural reranker with: python train_neural_reranker.py")
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
