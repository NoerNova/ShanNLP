"""
Frequency data management for spell correction.

This module loads and processes word frequency data from the Shan Wikipedia
corpus to help rank spelling correction suggestions.
"""

import math
from typing import Dict, Optional, Mapping
from shannlp.corpus import path_shannlp_corpus


# Global cache for frequency data (following ShanNLP corpus pattern)
_FREQUENCY_DATA: Optional[Dict[str, float]] = None
_RAW_FREQUENCY_DATA: Optional[Dict[str, int]] = None
_FREQUENCY_FILENAME = "shnwiki_freq.txt"


def load_frequency_data(normalize: bool = True) -> Mapping[str, float]:
    """
    Load and process Wikipedia frequency data.

    The frequency file format is: "word=ယဝ်ႉ, f=44656"
    This function parses the file, calculates word probabilities,
    and caches the results for performance.

    Args:
        normalize: If True, normalize to probabilities (0.0-1.0).
                  If False, return raw counts.

    Returns:
        Dictionary mapping words to their normalized frequencies/probabilities

    Examples:
        >>> freq = load_frequency_data()
        >>> freq['ယဝ်ႉ'] > 0  # Common word should have positive frequency
        True
        >>> len(freq) > 10000  # Should have many words
        True
    """
    global _FREQUENCY_DATA, _RAW_FREQUENCY_DATA

    # Return cached data if available
    if normalize and _FREQUENCY_DATA is not None:
        return _FREQUENCY_DATA
    if not normalize and _RAW_FREQUENCY_DATA is not None:
        return _RAW_FREQUENCY_DATA

    filepath = path_shannlp_corpus(_FREQUENCY_FILENAME)

    raw_freq = {}
    total_count = 0

    # Parse frequency file
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Parse format: "word=ယဝ်ႉ, f=44656"
                try:
                    # Split by comma
                    parts = line.split(', ')
                    if len(parts) != 2:
                        # Try alternative parsing
                        continue

                    word_part = parts[0].strip()
                    freq_part = parts[1].strip()

                    # Extract word (after "word=")
                    if not word_part.startswith('word='):
                        continue
                    word = word_part[5:]  # Skip "word="

                    # Extract frequency (after "f=")
                    if not freq_part.startswith('f='):
                        continue
                    freq_str = freq_part[2:]  # Skip "f="
                    freq = int(freq_str)

                    if word and freq > 0:
                        raw_freq[word] = freq
                        total_count += freq

                except (ValueError, IndexError, AttributeError):
                    # Skip malformed lines
                    continue

    except FileNotFoundError:
        # If frequency file doesn't exist, return empty dict
        # This allows spell correction to work without frequency data
        return {}

    # Cache raw frequencies
    _RAW_FREQUENCY_DATA = raw_freq

    if not normalize:
        return raw_freq

    # Normalize frequencies to probabilities
    if total_count == 0:
        # No valid data
        _FREQUENCY_DATA = {}
        return {}

    freq_data = {}
    for word, count in raw_freq.items():
        # Calculate probability: P(word) = count / total
        prob = count / total_count
        freq_data[word] = prob

    # Cache normalized frequencies
    _FREQUENCY_DATA = freq_data
    return freq_data


def get_word_probability(
    word: str,
    frequency_data: Optional[Mapping[str, float]] = None
) -> float:
    """
    Get probability of a word appearing in text.

    Uses cached frequency data if not provided. Returns a small
    default probability for unknown words.

    Args:
        word: Word to look up
        frequency_data: Optional pre-loaded frequency dict

    Returns:
        Probability value between 0.0 and 1.0

    Examples:
        >>> prob = get_word_probability("ယဝ်ႉ")
        >>> prob > 0  # Common word should have positive probability
        True
        >>> prob_unknown = get_word_probability("xyzabc123")
        >>> prob_unknown < 0.001  # Unknown word gets small default
        True
    """
    if frequency_data is None:
        frequency_data = load_frequency_data()

    # Default probability for unknown words
    # Small but non-zero to allow them as candidates
    default_prob = 1e-10

    return frequency_data.get(word, default_prob)


def get_word_count(
    word: str,
    frequency_data: Optional[Dict[str, int]] = None
) -> int:
    """
    Get raw frequency count for a word.

    Args:
        word: Word to look up
        frequency_data: Optional pre-loaded raw frequency dict

    Returns:
        Raw frequency count (0 if unknown)

    Examples:
        >>> count = get_word_count("ယဝ်ႉ")
        >>> count > 0  # Common word should have count
        True
    """
    if frequency_data is None:
        raw_data = load_frequency_data(normalize=False)
        # Ensure type safety: only accept dicts with int values
        if all(isinstance(v, int) for v in raw_data.values()):
            frequency_data = raw_data  # type: ignore
        else:
            frequency_data = {}

    if frequency_data is None:
        return 0
    return frequency_data.get(word, 0)


def get_log_probability(
    word: str,
    frequency_data: Optional[Dict[str, float]] = None
) -> float:
    """
    Get log probability of a word.

    Log probabilities are useful for preventing underflow when
    multiplying many small probabilities together.

    Args:
        word: Word to look up
        frequency_data: Optional pre-loaded frequency dict

    Returns:
        Log probability (negative number, closer to 0 is more likely)

    Examples:
        >>> log_prob = get_log_probability("ယဝ်ႉ")
        >>> log_prob < 0  # Log of probability is negative
        True
    """
    prob = get_word_probability(word, frequency_data)

    # Handle zero probability
    if prob <= 0:
        return float('-inf')

    return math.log(prob)


def get_top_words(n: int = 100) -> list:
    """
    Get the N most frequent words from the corpus.

    Useful for analysis and testing.

    Args:
        n: Number of top words to return

    Returns:
        List of (word, probability) tuples, sorted by frequency descending

    Examples:
        >>> top = get_top_words(10)
        >>> len(top) <= 10
        True
        >>> top[0][1] >= top[-1][1]  # Descending order
        True
    """
    freq_data = load_frequency_data(normalize=True)

    if not freq_data:
        return []

    # Sort by probability descending
    sorted_words = sorted(
        freq_data.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_words[:n]


def clear_cache():
    """
    Clear cached frequency data.

    Useful for testing or if frequency data is updated.

    Examples:
        >>> load_frequency_data()  # Load and cache
        >>> clear_cache()  # Clear cache
        >>> load_frequency_data()  # Reload from file
    """
    global _FREQUENCY_DATA, _RAW_FREQUENCY_DATA
    _FREQUENCY_DATA = None
    _RAW_FREQUENCY_DATA = None
