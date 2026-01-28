"""
Dataset classes for training neural spell reranker.

Handles loading training pairs and generating candidates for training.
"""

import json
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# Import after checking torch is available
try:
    from shannlp.spell.neural.model import CharVocab
    from shannlp.spell.core import spell_correct
except ImportError:
    pass


class SpellDataset(Dataset):
    """
    Dataset for training spell reranker.

    Each sample contains:
    - error: misspelled word
    - correct: correct word
    - candidates: list of correction candidates (including correct)
    - context: surrounding words
    - label: index of correct candidate

    Data format (JSONL):
        {"error": "မိုင်း", "correct": "မိူင်း", "context_left": "ၵူၼ်း", "context_right": "ၵိၼ်"}
    """

    def __init__(
        self,
        data_path: str,
        vocab: CharVocab,
        max_candidates: int = 10,
        max_word_len: int = 30,
        max_context_len: int = 100,
        generate_candidates: bool = True,
        augment: bool = True
    ):
        """
        Args:
            data_path: Path to JSONL training data
            vocab: Character vocabulary
            max_candidates: Maximum number of candidates per sample
            max_word_len: Maximum word length in characters
            max_context_len: Maximum context length in characters
            generate_candidates: Generate candidates using spell_correct
            augment: Apply data augmentation
        """
        self.vocab = vocab
        self.max_candidates = max_candidates
        self.max_word_len = max_word_len
        self.max_context_len = max_context_len
        self.generate_candidates = generate_candidates
        self.augment = augment

        # Load data
        self.samples = self._load_data(data_path)
        print(f"Loaded {len(self.samples)} training samples")

        # Check if data has pre-generated candidates
        has_candidates = sum(1 for s in self.samples if 'candidates' in s)
        if has_candidates > 0:
            print(f"  {has_candidates}/{len(self.samples)} samples have pre-generated candidates")
        elif self.generate_candidates:
            print("  WARNING: No pre-generated candidates found. This will be VERY SLOW!")
            print("  Run 'python preprocess_training_data.py --batch' to pre-generate candidates.")

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load training data from JSONL file."""
        samples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        if sample.get('error') and sample.get('correct'):
                            samples.append(sample)
                    except json.JSONDecodeError:
                        continue

        return samples

    def _get_candidates(self, error: str, correct: str, pregenerated: List[str] = None) -> Tuple[List[str], int]:
        """
        Get candidates for an error word.

        Args:
            error: The misspelled word
            correct: The correct word
            pregenerated: Pre-generated candidates (if available)

        Returns:
            Tuple of (candidates list, correct index)
        """
        candidates = []

        # Use pre-generated candidates if available
        if pregenerated:
            candidates = list(pregenerated)
        # Otherwise generate on-the-fly (WARNING: slow!)
        elif self.generate_candidates:
            try:
                suggestions = spell_correct(
                    error,
                    max_edit_distance=2,
                    max_suggestions=self.max_candidates - 1,
                    use_frequency=True,
                    use_phonetic=True
                )
                candidates = [word for word, score in suggestions]
            except Exception:
                pass

        # Ensure correct answer is in candidates
        if correct not in candidates:
            candidates.insert(0, correct)
        else:
            # Move correct to random position (avoid always being first)
            candidates.remove(correct)
            insert_pos = random.randint(0, len(candidates))
            candidates.insert(insert_pos, correct)

        # Pad or truncate candidates
        candidates = candidates[:self.max_candidates]

        # Find correct index
        correct_idx = candidates.index(correct) if correct in candidates else 0

        # Pad with empty strings
        while len(candidates) < self.max_candidates:
            candidates.append("")

        return candidates, correct_idx

    def _encode_text(self, text: str, max_len: int) -> Tuple[List[int], int]:
        """Encode text to character indices."""
        # Ensure text is not empty (use space as placeholder)
        if not text or not text.strip():
            text = " "
        indices = self.vocab.encode(text, max_len)
        # Ensure minimum length of 1
        length = max(1, min(len(text), max_len))
        return indices, length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        error = sample['error']
        correct = sample['correct']
        context_left = sample.get('context_left', '')
        context_right = sample.get('context_right', '')

        # Data augmentation
        if self.augment and random.random() < 0.2:
            # Randomly drop context
            if random.random() < 0.5:
                context_left = ""
            else:
                context_right = ""

        # Get candidates (use pre-generated if available)
        pregenerated = sample.get('candidates')
        candidates, correct_idx = self._get_candidates(error, correct, pregenerated)

        # Encode error
        error_chars, error_len = self._encode_text(error, self.max_word_len)

        # Encode candidates
        candidate_chars = []
        candidate_lens = []
        candidate_mask = []

        for cand in candidates:
            if cand:
                chars, length = self._encode_text(cand, self.max_word_len)
                candidate_chars.append(chars)
                candidate_lens.append(length)
                candidate_mask.append(1)
            else:
                # Padding candidate
                candidate_chars.append([0] * self.max_word_len)
                candidate_lens.append(1)
                candidate_mask.append(0)

        # Encode context
        context = f"{context_left} {context_right}".strip()
        context_chars, context_len = self._encode_text(context, self.max_context_len)

        return {
            'error_chars': torch.tensor(error_chars, dtype=torch.long),
            'error_length': torch.tensor(error_len, dtype=torch.long),
            'candidate_chars': torch.tensor(candidate_chars, dtype=torch.long),
            'candidate_lengths': torch.tensor(candidate_lens, dtype=torch.long),
            'candidate_mask': torch.tensor(candidate_mask, dtype=torch.float),
            'context_chars': torch.tensor(context_chars, dtype=torch.long),
            'context_length': torch.tensor(context_len, dtype=torch.long),
            'label': torch.tensor(correct_idx, dtype=torch.long),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    return {
        'error_chars': torch.stack([b['error_chars'] for b in batch]),
        'error_lengths': torch.stack([b['error_length'] for b in batch]),
        'candidate_chars': torch.stack([b['candidate_chars'] for b in batch]),
        'candidate_lengths': torch.stack([b['candidate_lengths'] for b in batch]),
        'candidate_mask': torch.stack([b['candidate_mask'] for b in batch]),
        'context_chars': torch.stack([b['context_chars'] for b in batch]),
        'context_lengths': torch.stack([b['context_length'] for b in batch]),
        'labels': torch.stack([b['label'] for b in batch]),
    }


def create_dataloader(
    data_path: str,
    vocab: CharVocab,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """Create DataLoader for training."""
    dataset = SpellDataset(data_path, vocab, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
