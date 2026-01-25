"""
N-gram language model for context-aware spell correction.

This module implements bigram and trigram models optimized for real-time
typing assistance (<100ms latency). Uses pre-computed log probabilities
and efficient dictionary-based storage.
"""

import math
import msgpack
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path


# Special tokens
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"


def _serialize_for_msgpack(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert data structures to msgpack-compatible format.

    MessagePack doesn't support:
    - Tuple keys (convert to strings with '|' separator)
    - Sets (convert to lists)
    - Nested defaultdict (convert to regular dict)

    Args:
        data: Dictionary with model data

    Returns:
        MessagePack-compatible dictionary
    """
    return {
        'n': data['n'],
        'smoothing': data['smoothing'],
        'ngram_counts': {
            # Convert tuple keys to strings: ('word1', 'word2') → 'word1|word2'
            '|'.join(context): dict(words)
            for context, words in data['ngram_counts'].items()
        },
        'context_counts': {
            # Convert tuple keys to strings
            '|'.join(context): count
            for context, count in data['context_counts'].items()
        },
        'vocabulary': list(data['vocabulary'])  # Set → List
    }


def _deserialize_from_msgpack(packed_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert msgpack data back to original structures.

    Args:
        packed_data: Data loaded from MessagePack

    Returns:
        Dictionary with original data structures restored
    """
    return {
        'n': packed_data['n'],
        'smoothing': packed_data['smoothing'],
        'ngram_counts': {
            # Convert string keys back to tuples: 'word1|word2' → ('word1', 'word2')
            tuple(context.split('|')): dict(words)
            for context, words in packed_data['ngram_counts'].items()
        },
        'context_counts': {
            # Convert string keys back to tuples
            tuple(context.split('|')): count
            for context, count in packed_data['context_counts'].items()
        },
        'vocabulary': set(packed_data['vocabulary'])  # List → Set
    }


class NgramModel:
    """
    N-gram language model for Shan text.

    Supports bigrams and trigrams with Laplace smoothing.
    Optimized for fast probability lookups in real-time applications.

    Examples:
        >>> model = NgramModel(n=2)  # Bigram model
        >>> model.train(["မိူင်း ယူႇ", "မိူင်း လႄႈ"])
        >>> prob = model.probability("ယူႇ", context=["မိူင်း"])
        >>> prob > 0
        True
    """

    def __init__(self, n: int = 2, smoothing: float = 1.0):
        """
        Initialize n-gram model.

        Args:
            n: N-gram order (2=bigram, 3=trigram)
            smoothing: Laplace smoothing parameter (default: 1.0)
        """
        if n < 1 or n > 3:
            raise ValueError("Only unigram (n=1), bigram (n=2), and trigram (n=3) supported")

        self.n = n
        self.smoothing = smoothing

        # N-gram counts: {(context_tuple): {word: count}}
        self.ngram_counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Context counts: {(context_tuple): total_count}
        self.context_counts: Dict[Tuple[str, ...], int] = defaultdict(int)

        # Vocabulary
        self.vocabulary: Set[str] = set()

        # Cache for frequently accessed probabilities
        self._prob_cache: Dict[Tuple[Tuple[str, ...], str], float] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Training flag
        self._is_trained = False

    def train(self, texts: List[str], tokenize_func=None):
        """
        Train n-gram model on corpus of texts.

        Args:
            texts: List of Shan text strings
            tokenize_func: Optional tokenization function.
                          If None, uses simple whitespace split.

        Examples:
            >>> model = NgramModel(n=2)
            >>> texts = ["မိူင်း ယူႇ", "မိူင်း လႄႈ"]
            >>> model.train(texts)
            >>> model.is_trained
            True
        """
        from shannlp.tokenize import word_tokenize

        if tokenize_func is None:
            tokenize_func = lambda text: word_tokenize(text, engine="newmm", keep_whitespace=False)

        print(f"Training {self.n}-gram model...")
        total_tokens = 0

        for text in texts:
            # Tokenize
            tokens = tokenize_func(text)
            if not tokens:
                continue

            # Add start/end tokens
            padded_tokens = [START_TOKEN] * (self.n - 1) + tokens + [END_TOKEN]

            # Extract n-grams
            for i in range(len(padded_tokens) - self.n + 1):
                # Get n-gram
                ngram = tuple(padded_tokens[i:i + self.n])

                # Context is all but last word
                context = ngram[:-1]
                word = ngram[-1]

                # Update counts
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1

                # Update vocabulary
                if word not in {START_TOKEN, END_TOKEN}:
                    self.vocabulary.add(word)

                total_tokens += 1

        self._is_trained = True
        print(f"Training complete:")
        print(f"  - Vocabulary size: {len(self.vocabulary)}")
        print(f"  - Unique contexts: {len(self.context_counts)}")
        print(f"  - Total {self.n}-grams: {total_tokens}")

    def probability(self, word: str, context: List[str]) -> float:
        """
        Calculate P(word | context) using n-gram model.

        Uses Laplace smoothing for unseen n-grams.

        Args:
            word: Word to calculate probability for
            context: List of context words (length should be n-1)

        Returns:
            Probability value between 0 and 1

        Examples:
            >>> model = NgramModel(n=2)
            >>> model.train(["မိူင်း ယူႇ"])
            >>> prob = model.probability("ယူႇ", ["မိူင်း"])
            >>> prob > 0
            True
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before calculating probabilities")

        # Adjust context length if needed
        context = self._adjust_context(context)
        context_tuple = tuple(context)

        # Check cache
        cache_key = (context_tuple, word)
        if cache_key in self._prob_cache:
            self._cache_hits += 1
            return self._prob_cache[cache_key]

        self._cache_misses += 1

        # Get counts
        word_count = self.ngram_counts[context_tuple].get(word, 0)
        context_count = self.context_counts[context_tuple]

        # Laplace smoothing: P(word|context) = (count + α) / (context_count + α * V)
        vocab_size = len(self.vocabulary) + 1  # +1 for UNK
        prob = (word_count + self.smoothing) / (context_count + self.smoothing * vocab_size)

        # Cache result
        if len(self._prob_cache) < 10000:  # Limit cache size
            self._prob_cache[cache_key] = prob

        return prob

    def log_probability(self, word: str, context: List[str]) -> float:
        """
        Calculate log P(word | context).

        More numerically stable for combining probabilities.

        Args:
            word: Word to calculate probability for
            context: List of context words

        Returns:
            Log probability (negative number)

        Examples:
            >>> model = NgramModel(n=2)
            >>> model.train(["မိူင်း ယူႇ"])
            >>> log_prob = model.log_probability("ယူႇ", ["မိူင်း"])
            >>> log_prob < 0
            True
        """
        prob = self.probability(word, context)
        return math.log(prob) if prob > 0 else float('-inf')

    def score_sequence(self, words: List[str]) -> float:
        """
        Calculate log probability of word sequence.

        Args:
            words: List of words in sequence

        Returns:
            Log probability of entire sequence

        Examples:
            >>> model = NgramModel(n=2)
            >>> model.train(["မိူင်း ယူႇ လႄႈ"])
            >>> score = model.score_sequence(["မိူင်း", "ယူႇ"])
            >>> score < 0
            True
        """
        if not words:
            return 0.0

        # Add start tokens
        padded = [START_TOKEN] * (self.n - 1) + words

        total_log_prob = 0.0
        for i in range(self.n - 1, len(padded)):
            context = padded[i - self.n + 1:i]
            word = padded[i]
            total_log_prob += self.log_probability(word, context)

        return total_log_prob

    def perplexity(self, test_texts: List[str], tokenize_func=None) -> float:
        """
        Calculate perplexity on test set.

        Lower perplexity = better model.

        Args:
            test_texts: List of test texts
            tokenize_func: Tokenization function

        Returns:
            Perplexity value
        """
        if tokenize_func is None:
            from shannlp.tokenize import word_tokenize
            tokenize_func = lambda text: word_tokenize(text, keep_whitespace=False)

        total_log_prob = 0.0
        total_words = 0

        for text in test_texts:
            tokens = tokenize_func(text)
            if tokens:
                total_log_prob += self.score_sequence(tokens)
                total_words += len(tokens)

        if total_words == 0:
            return float('inf')

        avg_log_prob = total_log_prob / total_words
        perplexity = math.exp(-avg_log_prob)
        return perplexity

    def _adjust_context(self, context: List[str]) -> List[str]:
        """Adjust context to correct length for n-gram order."""
        required_length = self.n - 1

        if len(context) < required_length:
            # Pad with START tokens
            return [START_TOKEN] * (required_length - len(context)) + context
        elif len(context) > required_length:
            # Take last n-1 words
            return context[-required_length:]
        else:
            return context

    def save(self, filepath: str):
        """
        Save trained model to file using MessagePack (safe serialization).

        Args:
            filepath: Path to save model (recommend .msgpack extension)

        Examples:
            >>> model = NgramModel(n=2)
            >>> model.train(["test text"])
            >>> model.save("model.msgpack")
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")

        # Prepare data
        data = {
            'n': self.n,
            'smoothing': self.smoothing,
            'ngram_counts': dict(self.ngram_counts),
            'context_counts': dict(self.context_counts),
            'vocabulary': self.vocabulary
        }

        # Convert to msgpack-compatible format
        packed_data = _serialize_for_msgpack(data)

        # Save with MessagePack
        with open(filepath, 'wb') as f:
            msgpack.pack(packed_data, f, use_bin_type=True)

        print(f"Model saved to {filepath} (MessagePack format)")

    @classmethod
    def load(cls, filepath: str) -> 'NgramModel':
        """
        Load trained model from file.

        Supports both MessagePack (.msgpack) and legacy pickle (.pkl) formats.
        MessagePack is recommended for security (no code execution vulnerabilities).

        Args:
            filepath: Path to model file

        Returns:
            Loaded NgramModel instance

        Examples:
            >>> model = NgramModel.load("model.msgpack")
            >>> model.is_trained
            True
        """
        # Try MessagePack first (preferred format)

        try:
            with open(filepath, 'rb') as f:
                packed_data = msgpack.unpack(f, raw=False)
                # Ensure packed_data is a dict
                if not isinstance(packed_data, dict):
                    raise TypeError("Unpacked MessagePack data is not a dictionary")
            # Successfully loaded as msgpack
            data = _deserialize_from_msgpack(packed_data)
            print(f"Model loaded from {filepath} (MessagePack format)")

        except (msgpack.exceptions.UnpackException, msgpack.exceptions.ExtraData, KeyError) as e:
            # Fall back to pickle (legacy format)
            print(f"⚠️  Loading legacy pickle file: {filepath}")
            print("   Consider re-saving with .save() to use safer MessagePack format")

            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

        # Restore model (same for both formats)
        model = cls(n=data['n'], smoothing=data['smoothing'])
        model.ngram_counts = defaultdict(lambda: defaultdict(int), data['ngram_counts'])
        model.context_counts = defaultdict(int, data['context_counts'])
        model.vocabulary = data['vocabulary']
        model._is_trained = True

        print(f"  - Vocabulary size: {len(model.vocabulary)}")
        print(f"  - Unique contexts: {len(model.context_counts)}")

        return model

    def get_cache_stats(self) -> Dict[str, int | float]:
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0

        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._prob_cache)
        }

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained
