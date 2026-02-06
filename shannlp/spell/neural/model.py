"""
Neural reranker model for Shan spell correction.

Architecture:
1. Character-level embeddings (handles Shan Unicode)
2. Bidirectional LSTM for encoding words and context
3. Attention mechanism for context weighting
4. Scoring head for candidate ranking

The model learns to select the best correction candidate given:
- The misspelled word
- Candidate corrections (from classical spell corrector)
- Surrounding context (words before/after)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import json
import os


class CharVocab:
    """
    Character vocabulary for Shan text.

    Handles all Shan Unicode characters plus special tokens.
    """

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    SEP = "<SEP>"

    def __init__(self):
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}

        # Add special tokens
        for token in [self.PAD, self.UNK, self.BOS, self.EOS, self.SEP]:
            self._add_char(token)

        # Add Shan characters
        self._add_shan_chars()

    def _add_char(self, char: str) -> int:
        if char not in self.char2idx:
            idx = len(self.char2idx)
            self.char2idx[char] = idx
            self.idx2char[idx] = char
        return self.char2idx[char]

    def _add_shan_chars(self):
        """Add all Shan Unicode characters."""
        # Shan consonants
        consonants = "ၵၷၶꧠငၸၹသၺတၻထၼꧣပၽၾၿႀမယရလꩮဝႁဢ"

        # Shan vowels
        vowels = "\u1083\u1062\u1084\u1085\u1031\u1035\u102d\u102e\u102f\u1030\u1086\u1082\u103a\u103d\u103b\u103c"

        # Tone marks
        tones = "\u1087\u1088\u1038\u1089\u108a"

        # Punctuation and digits
        punct = "\u104a\u104b\ua9e6"
        digits = "႐႑႒႓႔႕႖႗႘႙"

        # Add all
        for char in consonants + vowels + tones + punct + digits:
            self._add_char(char)

        # Add common ASCII for mixed text
        for char in " .,!?0123456789":
            self._add_char(char)

    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """Encode text to character indices."""
        indices = [self.char2idx.get(c, self.char2idx[self.UNK]) for c in text]

        if max_len:
            if len(indices) < max_len:
                indices = indices + [self.char2idx[self.PAD]] * (max_len - len(indices))
            else:
                indices = indices[:max_len]

        return indices

    def decode(self, indices: List[int]) -> str:
        """Decode indices back to text."""
        chars = []
        for idx in indices:
            if idx in self.idx2char:
                char = self.idx2char[idx]
                if char not in [self.PAD, self.BOS, self.EOS]:
                    chars.append(char)
        return "".join(chars)

    def __len__(self) -> int:
        return len(self.char2idx)

    def save(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char2idx': self.char2idx,
                'idx2char': {str(k): v for k, v in self.idx2char.items()}
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'CharVocab':
        """Load vocabulary from file."""
        vocab = cls.__new__(cls)
        vocab.char2idx = {}
        vocab.idx2char = {}

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            vocab.char2idx = data['char2idx']
            vocab.idx2char = {int(k): v for k, v in data['idx2char'].items()}

        return vocab


class CharEncoder(nn.Module):
    """
    Character-level encoder using BiLSTM.

    Encodes a word (or text) into a fixed-size vector.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Character indices (batch, seq_len)
            lengths: Actual lengths before padding

        Returns:
            Encoded representation (batch, output_dim)
        """
        # Embed characters
        embedded = self.dropout(self.embed(x))  # (batch, seq_len, embed_dim)

        # Pack if lengths provided and all lengths > 0
        if lengths is not None:
            # Ensure minimum length of 1 to avoid pack_padded_sequence error
            lengths = lengths.clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(embedded)

        # Concatenate forward and backward hidden states
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        return hidden  # (batch, output_dim)


class SpellReranker(nn.Module):
    """
    Neural reranker for spell correction candidates.

    Takes a misspelled word, candidates, and context, then scores
    each candidate to select the best correction.

    Architecture:
    1. Encode error word, candidates, and context with CharEncoder
    2. Compute attention-weighted context representation
    3. Score each candidate based on similarity to context

    Usage:
        model = SpellReranker(vocab)
        scores = model(error_chars, candidate_chars, context_chars)
        best_idx = scores.argmax(dim=1)
    """

    def __init__(
        self,
        vocab: CharVocab,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_candidates: int = 10
    ):
        super().__init__()

        self.vocab = vocab
        self.max_candidates = max_candidates

        vocab_size = len(vocab)

        # Shared character encoder
        self.char_encoder = CharEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        encoder_dim = self.char_encoder.output_dim

        # Context attention
        self.context_attention = nn.MultiheadAttention(
            embed_dim=encoder_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Scoring network
        self.scorer = nn.Sequential(
            nn.Linear(encoder_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Error-candidate comparison
        self.error_candidate_compare = nn.Bilinear(encoder_dim, encoder_dim, hidden_dim // 2)

    def encode_batch(
        self,
        texts: List[str],
        max_len: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of texts."""
        encoded = []
        lengths = []

        for text in texts:
            indices = self.vocab.encode(text, max_len)
            encoded.append(indices)
            lengths.append(min(len(text), max_len))

        chars = torch.tensor(encoded, dtype=torch.long)
        lens = torch.tensor(lengths, dtype=torch.long)

        return chars, lens

    def forward(
        self,
        error_chars: torch.Tensor,
        error_lengths: torch.Tensor,
        candidate_chars: torch.Tensor,  # (batch, num_candidates, seq_len)
        candidate_lengths: torch.Tensor,  # (batch, num_candidates)
        context_chars: torch.Tensor,  # (batch, context_len)
        context_lengths: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None  # (batch, num_candidates)
    ) -> torch.Tensor:
        """
        Score candidates for each error word.

        Args:
            error_chars: Encoded error words (batch, seq_len)
            error_lengths: Lengths of error words
            candidate_chars: Encoded candidates (batch, num_candidates, seq_len)
            candidate_lengths: Lengths of candidates
            context_chars: Encoded context (batch, context_len)
            context_lengths: Lengths of context
            candidate_mask: Mask for valid candidates (1=valid, 0=padding)

        Returns:
            Scores for each candidate (batch, num_candidates)
        """
        batch_size = error_chars.size(0)
        num_candidates = candidate_chars.size(1)
        device = error_chars.device

        # Encode error word
        error_enc = self.char_encoder(error_chars, error_lengths)  # (batch, encoder_dim)

        # Encode context
        context_enc = self.char_encoder(context_chars, context_lengths)  # (batch, encoder_dim)

        # Encode each candidate
        candidate_chars_flat = candidate_chars.view(-1, candidate_chars.size(-1))
        candidate_lengths_flat = candidate_lengths.view(-1)

        candidate_enc = self.char_encoder(candidate_chars_flat, candidate_lengths_flat)
        candidate_enc = candidate_enc.view(batch_size, num_candidates, -1)  # (batch, num_cand, encoder_dim)

        # Score each candidate
        scores = []
        for i in range(num_candidates):
            cand_i = candidate_enc[:, i, :]  # (batch, encoder_dim)

            # Compare error to candidate
            error_cand_sim = self.error_candidate_compare(error_enc, cand_i)  # (batch, hidden/2)

            # Combine features
            combined = torch.cat([
                error_enc,
                cand_i,
                context_enc
            ], dim=1)  # (batch, encoder_dim * 3)

            score = self.scorer(combined).squeeze(-1)  # (batch,)
            scores.append(score)

        scores = torch.stack(scores, dim=1)  # (batch, num_candidates)

        # Apply mask if provided
        if candidate_mask is not None:
            scores = scores.masked_fill(candidate_mask == 0, float('-inf'))

        return scores

    def predict(
        self,
        error: str,
        candidates: List[str],
        context_left: str = "",
        context_right: str = "",
        device: str = None
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Predict best correction for a single error.

        Args:
            error: Misspelled word
            candidates: List of correction candidates
            context_left: Words before error
            context_right: Words after error
            device: Device to use

        Returns:
            Tuple of (best_candidate, [(candidate, score), ...])
        """
        self.eval()

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if not candidates:
            return error, []

        # Pad candidates to max_candidates
        padded_candidates = candidates[:self.max_candidates]
        while len(padded_candidates) < self.max_candidates:
            padded_candidates.append("")

        # Create mask
        mask = [1 if c else 0 for c in padded_candidates]

        # Encode
        max_word_len = 30
        max_context_len = 100

        error_chars, error_lens = self.encode_batch([error], max_word_len)

        cand_chars_list = []
        cand_lens_list = []
        for cand in padded_candidates:
            c, l = self.encode_batch([cand if cand else " "], max_word_len)
            cand_chars_list.append(c)
            cand_lens_list.append(l)

        candidate_chars = torch.stack([c.squeeze(0) for c in cand_chars_list], dim=0).unsqueeze(0)
        candidate_lengths = torch.stack([l.squeeze(0) for l in cand_lens_list], dim=0).unsqueeze(0)

        context = f"{context_left} {context_right}".strip()
        context_chars, context_lens = self.encode_batch([context], max_context_len)

        candidate_mask = torch.tensor([mask], dtype=torch.float)

        # Move to device
        error_chars = error_chars.to(device)
        error_lens = error_lens.to(device)
        candidate_chars = candidate_chars.to(device)
        candidate_lengths = candidate_lengths.to(device)
        context_chars = context_chars.to(device)
        context_lens = context_lens.to(device)
        candidate_mask = candidate_mask.to(device)

        # Get scores
        with torch.no_grad():
            scores = self.forward(
                error_chars, error_lens,
                candidate_chars, candidate_lengths,
                context_chars, context_lens,
                candidate_mask
            )

        scores = scores.squeeze(0).cpu()

        # Get results
        results = []
        for i, cand in enumerate(candidates):
            if i < len(scores):
                results.append((cand, scores[i].item()))

        results.sort(key=lambda x: x[1], reverse=True)

        best = results[0][0] if results else error
        return best, results

    def save(self, path: str):
        """Save model and vocabulary."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        torch.save({
            'model_state': self.state_dict(),
            'max_candidates': self.max_candidates,
        }, path)

        # Save vocab separately
        vocab_path = path.replace('.pt', '_vocab.json')
        self.vocab.save(vocab_path)

        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = None) -> 'SpellReranker':
        """Load model and vocabulary."""
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        vocab_path = path.replace('.pt', '_vocab.json')
        vocab = CharVocab.load(vocab_path)

        checkpoint = torch.load(path, map_location=device, weights_only=True)

        model = cls(
            vocab=vocab,
            max_candidates=checkpoint.get('max_candidates', 10)
        )

        model.load_state_dict(checkpoint['model_state'])
        model.to(device)

        print(f"Model loaded from {path}")
        return model
