"""
Shan-aware edit distance calculation for spell correction.

This module implements a modified Damerau-Levenshtein distance algorithm
with weighted costs based on Shan language characteristics:
- Tone mark errors have lower cost (common mistake)
- Vowel position errors have lower cost
- Phonetically similar consonants have lower cost
"""

import unicodedata
from typing import Optional, Dict, Set
from shannlp.spell.phonetic import (
    get_character_type,
    get_phonetic_distance,
    load_phonetic_groups
)


# Default edit operation costs
DEFAULT_COSTS = {
    'delete': 1.0,
    'insert': 1.0,
    'substitute': 1.0,
    'transpose': 1.0,
}

# Shan-specific cost adjustments
SHAN_COSTS = {
    'tone_substitute': 0.5,      # Tone marks commonly confused
    'vowel_substitute': 0.7,     # Vowel positions sometimes confused
    'phonetic_substitute': 0.6,  # Phonetically similar consonants
    'consonant_substitute': 1.0, # Standard consonant change
    'cross_type_substitute': 2.0, # Consonant <-> vowel (very unlikely)
}


def shan_edit_distance(
    word1: str,
    word2: str,
    max_distance: Optional[int] = None,
    phonetic_groups: Optional[Dict[str, Set[str]]] = None,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate Shan-aware edit distance between two words.

    Uses dynamic programming with weighted costs based on character types
    and phonetic similarity. This helps identify more likely corrections.

    Args:
        word1: First word
        word2: Second word
        max_distance: Maximum distance to compute (for early termination)
        phonetic_groups: Phonetic similarity groups (loaded if None)
        weights: Custom cost weights (uses SHAN_COSTS if None)

    Returns:
        Edit distance as float (lower = more similar)

    Examples:
        >>> shan_edit_distance("မိူင်း", "မိူင်း")  # Identical
        0.0
        >>> # Tone mark change (lower cost)
        >>> dist1 = shan_edit_distance("လႄႈ", "လႆႈ")
        >>> # Consonant change (higher cost)
        >>> dist2 = shan_edit_distance("လႄႈ", "မႄႈ")
        >>> dist1 < dist2
        True
    """
    # Normalize Unicode to NFC form
    word1 = unicodedata.normalize('NFC', word1)
    word2 = unicodedata.normalize('NFC', word2)

    # Handle empty strings
    if not word1:
        return float(len(word2))
    if not word2:
        return float(len(word1))

    # Identical words
    if word1 == word2:
        return 0.0

    # Load phonetic groups if not provided
    if phonetic_groups is None:
        phonetic_groups = load_phonetic_groups()

    # Use default Shan costs if not provided
    if weights is None:
        weights = SHAN_COSTS

    # Convert strings to lists for easier indexing
    chars1 = list(word1)
    chars2 = list(word2)
    len1 = len(chars1)
    len2 = len(chars2)

    # Early termination check
    if max_distance is not None:
        # If length difference exceeds max_distance, return early
        if abs(len1 - len2) > max_distance:
            return float('inf')

    # Initialize distance matrix
    # dp[i][j] = distance between word1[:i] and word2[:j]
    dp = [[0.0] * (len2 + 1) for _ in range(len1 + 1)]

    # Base cases: distance from empty string
    for i in range(len1 + 1):
        dp[i][0] = float(i)  # All deletions
    for j in range(len2 + 1):
        dp[0][j] = float(j)  # All insertions

    # Fill distance matrix
    for i in range(1, len1 + 1):
        # Track minimum in this row for early termination
        row_min = float('inf')

        for j in range(1, len2 + 1):
            char1 = chars1[i - 1]
            char2 = chars2[j - 1]

            if char1 == char2:
                # Characters match, no cost
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Calculate costs for different operations

                # Deletion cost
                delete_cost = dp[i - 1][j] + DEFAULT_COSTS['delete']

                # Insertion cost
                insert_cost = dp[i][j - 1] + DEFAULT_COSTS['insert']

                # Substitution cost (character-type aware)
                substitute_cost = dp[i - 1][j - 1] + get_substitution_cost(
                    char1, char2, weights, phonetic_groups
                )

                # Take minimum of the three operations
                dp[i][j] = min(delete_cost, insert_cost, substitute_cost)

                # Transposition (swap adjacent characters)
                if i > 1 and j > 1:
                    if char1 == chars2[j - 2] and chars1[i - 2] == char2:
                        transpose_cost = dp[i - 2][j - 2] + DEFAULT_COSTS['transpose']
                        dp[i][j] = min(dp[i][j], transpose_cost)

            # Track minimum for early termination
            row_min = min(row_min, dp[i][j])

        # Early termination: if minimum in row exceeds threshold
        if max_distance is not None and row_min > max_distance:
            return float('inf')

    return dp[len1][len2]


def get_substitution_cost(
    char1: str,
    char2: str,
    weights: Dict[str, float],
    phonetic_groups: Dict[str, Set[str]]
) -> float:
    """
    Calculate substitution cost between two characters.

    Cost depends on character types and phonetic similarity:
    - Same type + phonetically similar: lowest cost
    - Same type: medium cost
    - Different types: high cost

    Args:
        char1: First character
        char2: Second character
        weights: Cost weight dictionary
        phonetic_groups: Phonetic similarity groups

    Returns:
        Substitution cost

    Examples:
        >>> # Tone marks (low cost)
        >>> cost1 = get_substitution_cost('ႇ', 'ႈ', SHAN_COSTS, {})
        >>> # Consonants (higher cost)
        >>> cost2 = get_substitution_cost('က', 'မ', SHAN_COSTS, {})
        >>> cost1 < cost2
        True
    """
    # Get character types
    type1 = get_character_type(char1)
    type2 = get_character_type(char2)

    # Same character (shouldn't happen, but handle it)
    if char1 == char2:
        return 0.0

    # Tone mark substitution (commonly confused)
    if type1 == 'tone' and type2 == 'tone':
        return weights.get('tone_substitute', 0.5)

    # Vowel substitution (position errors)
    if type1.startswith('vowel') and type2.startswith('vowel'):
        return weights.get('vowel_substitute', 0.7)

    # Consonant substitution
    if type1 == 'consonant' and type2 == 'consonant':
        # Check if phonetically similar
        phonetic_dist = get_phonetic_distance(char1, char2)
        if phonetic_dist < 1.0:
            # Phonetically similar
            return weights.get('phonetic_substitute', 0.6)
        else:
            # Not similar
            return weights.get('consonant_substitute', 1.0)

    # Cross-type substitution (consonant <-> vowel, very unlikely)
    if (type1 == 'consonant' and type2.startswith('vowel')) or \
       (type1.startswith('vowel') and type2 == 'consonant'):
        return weights.get('cross_type_substitute', 2.0)

    # Default substitution cost
    return DEFAULT_COSTS['substitute']


def edit_distance_normalized(word1: str, word2: str, **kwargs) -> float:
    """
    Calculate normalized edit distance (0.0 to 1.0).

    Divides edit distance by maximum possible distance (longer word length).
    Useful for comparing distances between word pairs of different lengths.

    Args:
        word1: First word
        word2: Second word
        **kwargs: Additional arguments passed to shan_edit_distance

    Returns:
        Normalized distance between 0.0 (identical) and 1.0 (completely different)

    Examples:
        >>> edit_distance_normalized("မိူင်း", "မိူင်း")
        0.0
        >>> edit_distance_normalized("abc", "xyz") <= 1.0
        True
    """
    distance = shan_edit_distance(word1, word2, **kwargs)

    # Handle infinite distance
    if distance == float('inf'):
        return 1.0

    # Normalize by length of longer word
    max_len = max(len(word1), len(word2))
    if max_len == 0:
        return 0.0

    return min(distance / max_len, 1.0)


def similarity_score(word1: str, word2: str, **kwargs) -> float:
    """
    Calculate similarity score (0.0 to 1.0).

    Converts distance to similarity: 1.0 = identical, 0.0 = completely different.

    Args:
        word1: First word
        word2: Second word
        **kwargs: Additional arguments passed to shan_edit_distance

    Returns:
        Similarity score between 0.0 and 1.0

    Examples:
        >>> similarity_score("မိူင်း", "မိူင်း")
        1.0
        >>> score = similarity_score("မိူင်း", "မိုင်း")
        >>> 0.0 < score < 1.0
        True
    """
    distance = edit_distance_normalized(word1, word2, **kwargs)
    return 1.0 - distance


def are_similar(
    word1: str,
    word2: str,
    threshold: float = 0.7,
    **kwargs
) -> bool:
    """
    Check if two words are similar based on edit distance.

    Args:
        word1: First word
        word2: Second word
        threshold: Similarity threshold (0.0 to 1.0, default 0.7)
        **kwargs: Additional arguments passed to similarity_score

    Returns:
        True if similarity score >= threshold

    Examples:
        >>> are_similar("မိူင်း", "မိူင်း")  # Identical
        True
        >>> are_similar("မိူင်း", "မိုင်း")  # Similar
        True
        >>> are_similar("abc", "xyz")  # Different
        False
    """
    return similarity_score(word1, word2, **kwargs) >= threshold
