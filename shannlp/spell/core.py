"""
Core spell correction logic for Shan language.

This module implements Peter Norvig's spell correction algorithm
enhanced with Shan-specific features:
- Weighted edit distance based on character types
- Phonetic similarity for consonants
- Frequency-based ranking
"""

from typing import List, Set, Tuple, Optional, Union
from shannlp.corpus import shan_words
from shannlp.spell.validator import (
    validate_input,
    ResourceLimiter,
    normalize_unicode
)
from shannlp.spell.frequency import (
    load_frequency_data,
    get_word_probability
)
from shannlp.spell.phonetic import (
    load_phonetic_groups,
    get_phonetic_substitutions
)
from shannlp.spell.distance import shan_edit_distance

# Define Shan characters locally to avoid circular import
shan_consonants = "ၵၷၶꧠငၸၹသၺတၻထၼꧣပၽၾပၿႀမယရလꩮဝႁဢ"
shan_vowels = "\u1083\u1062\u1084\u1085\u1031\u1035\u102d\u102e\u102f\u1030\u1086\u1082\u103a\u103d\u103b\u103c"
shan_tonemarks = "\u1087\u1088\u1038\u1089\u108a"
shan_punctuations = "\u104a\u104b\ua9e6"
shan_digits = "႐႑႒႓႔႕႖႗႘႙"
shan_letters = "".join([shan_consonants, shan_vowels, shan_tonemarks, shan_punctuations])
shan_characters = "".join([shan_letters, shan_digits])


def spell_correct(
    word: str,
    custom_dict: Optional[Union[Set[str], frozenset]] = None,
    max_edit_distance: int = 2,
    max_suggestions: int = 5,
    use_frequency: bool = True,
    use_phonetic: bool = True,
    min_confidence: float = 0.1
) -> List[Tuple[str, float]]:
    """
    Correct spelling of a single Shan word.

    Uses Peter Norvig's algorithm enhanced with Shan-specific features.
    Generates candidates within edit distance, ranks by probability and
    distance, returns top suggestions with confidence scores.

    Args:
        word: Word to correct
        custom_dict: Custom dictionary (default: shan_words())
        max_edit_distance: Maximum edit distance to consider (1 or 2)
        max_suggestions: Maximum number of suggestions to return
        use_frequency: Use frequency data for ranking
        use_phonetic: Use phonetic similarity for better candidates
        min_confidence: Minimum confidence threshold (0.0 to 1.0)

    Returns:
        List of (suggestion, confidence) tuples, sorted by confidence descending

    Raises:
        ValueError: If word is invalid
        SecurityError: If word contains security threats

    Examples:
        >>> # Correct a misspelled word
        >>> results = spell_correct("မိုင်း")
        >>> results[0][0]  # Top suggestion
        'မိူင်း'

        >>> # Word already correct
        >>> results = spell_correct("မိူင်း")
        >>> results[0] == ("မိူင်း", 1.0) or results[0][1] > 0.9
        True

        >>> # Use custom dictionary
        >>> custom = {"ၵႃႈ", "မူၼ်း"}
        >>> spell_correct("ၵႃႈ", custom_dict=custom)
        [('ၵႃႈ', 1.0)]
    """
    # Validate and normalize input
    validate_input(word)
    word = normalize_unicode(word)

    # Load dictionary
    dictionary = custom_dict if custom_dict is not None else shan_words()

    # Load resources if needed
    frequency_data = load_frequency_data() if use_frequency else {}
    phonetic_groups = load_phonetic_groups() if use_phonetic else {}

    # Generate candidates
    candidates = generate_candidates(
        word,
        dictionary,
        max_edit_distance,
        use_phonetic,
        phonetic_groups
    )

    # If no candidates found, return empty list
    if not candidates:
        return []

    # Rank candidates
    ranked = rank_candidates(
        candidates,
        word,
        frequency_data,
        use_frequency,
        use_phonetic,
        phonetic_groups
    )

    # Filter by confidence and limit
    results = [
        (suggestion, score)
        for suggestion, score in ranked
        if score >= min_confidence
    ][:max_suggestions]

    return results


def generate_candidates(
    word: str,
    dictionary: Union[Set[str], frozenset],
    max_distance: int,
    use_phonetic: bool,
    phonetic_groups: dict
) -> Set[str]:
    """
    Generate candidate corrections for a word.

    Follows Norvig's algorithm:
    1. If word is in dictionary, include it
    2. Generate edit distance 1 candidates
    3. If needed, generate edit distance 2 candidates
    4. Add phonetic substitutions if enabled

    Args:
        word: Word to generate candidates for
        dictionary: Dictionary of valid words
        max_distance: Maximum edit distance (1 or 2)
        use_phonetic: Include phonetic substitutions
        phonetic_groups: Phonetic similarity groups

    Returns:
        Set of candidate words

    Examples:
        >>> dictionary = {"မိူင်း", "မိူင်", "မူၼ်း"}
        >>> candidates = generate_candidates("မိုင်း", dictionary, 2, False, {})
        >>> len(candidates) > 0
        True
    """
    candidates = set()
    limiter = ResourceLimiter()

    # Level 0: Word is already in dictionary
    if word in dictionary:
        candidates.add(word)
        return candidates

    # Level 1: Edit distance 1
    edits1 = edits_distance_1(word, use_phonetic, phonetic_groups)
    edits1 = limiter.check_candidate_limit(edits1)
    candidates_1 = edits1 & dictionary

    if candidates_1:
        candidates.update(candidates_1)

    # Level 2: Edit distance 2 (always generate if max_distance >= 2)
    # Previous bug: skipped level 2 if len(candidates) >= 5, missing valid corrections
    if max_distance >= 2:
        # Generate edits2 from ALL edits1 (not just dictionary matches)
        edits2 = set()
        for e1 in edits1:
            edits2.update(edits_distance_1(e1, use_phonetic, phonetic_groups))

        edits2 = limiter.check_candidate_limit(edits2)
        candidates_2 = edits2 & dictionary
        candidates.update(candidates_2)

    return candidates


def edits_distance_1(
    word: str,
    use_phonetic: bool = False,
    phonetic_groups: Optional[dict] = None
) -> Set[str]:
    """
    Generate all words that are one edit away.

    Edit operations:
    - Delete: Remove one character
    - Insert: Add one character
    - Replace: Change one character
    - Transpose: Swap two adjacent characters
    - Phonetic substitute: Replace with phonetically similar character

    Args:
        word: Input word
        use_phonetic: Include phonetic substitutions
        phonetic_groups: Phonetic similarity groups

    Returns:
        Set of all possible one-edit words

    Examples:
        >>> edits = edits_distance_1("မိူင်း", use_phonetic=False)
        >>> len(edits) > 0
        True
    """
    # All possible splits of the word
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

    # Deletions: Remove one character
    deletes = [L + R[1:] for L, R in splits if R]

    # Transpositions: Swap adjacent characters
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]

    # Replacements: Change one character to any Shan character
    shan_chars = list(shan_characters)
    replaces = [L + c + R[1:] for L, R in splits if R for c in shan_chars]

    # Insertions: Add one Shan character
    inserts = [L + c + R for L, R in splits for c in shan_chars]

    # Combine all edits
    edits = set(deletes + transposes + replaces + inserts)

    # Add phonetic substitutions if enabled
    if use_phonetic and phonetic_groups:
        phonetic_subs = []
        for L, R in splits:
            if R:
                char = R[0]
                similar_chars = get_phonetic_substitutions(char)
                for similar_char in similar_chars:
                    phonetic_subs.append(L + similar_char + R[1:])
        edits.update(phonetic_subs)

    return edits


def rank_candidates(
    candidates: Set[str],
    original_word: str,
    frequency_data: dict,
    use_frequency: bool,
    use_phonetic: bool,
    phonetic_groups: dict
) -> List[Tuple[str, float]]:
    """
    Rank candidates by probability and edit distance.

    Scoring formula:
    score = (0.6 × distance_score) + (0.4 × frequency_score)

    Where:
    - distance_score = 1.0 / (1.0 + edit_distance)
    - frequency_score = word probability from corpus

    Args:
        candidates: Set of candidate words
        original_word: Original (misspelled) word
        frequency_data: Word frequency mapping
        use_frequency: Use frequency in scoring
        use_phonetic: Use phonetic distance
        phonetic_groups: Phonetic similarity groups

    Returns:
        List of (word, score) tuples sorted by score descending

    Examples:
        >>> candidates = {"မိူင်း", "မိူင်"}
        >>> ranked = rank_candidates(candidates, "မိုင်း", {}, False, False, {})
        >>> len(ranked) == 2
        True
        >>> ranked[0][1] >= ranked[1][1]  # Descending order
        True
    """
    scored = []

    for candidate in candidates:
        # Calculate edit distance
        edit_dist = shan_edit_distance(
            original_word,
            candidate,
            max_distance=3,
            phonetic_groups=phonetic_groups if use_phonetic else None
        )

        # Convert distance to score (inverse relationship)
        # distance_score ranges from 0 (very different) to 1 (identical)
        distance_score = 1.0 / (1.0 + edit_dist)

        # Frequency score
        freq_score = 0.0
        if use_frequency and frequency_data:
            # Get word probability
            prob = get_word_probability(candidate, frequency_data)
            # Normalize to 0-1 scale (log scale works better)
            # Most common words have prob ~ 0.01, rare words ~ 1e-6
            # Map this to 0-1 scale logarithmically
            if prob > 0:
                # Scale: common word (1e-2) -> 1.0, rare word (1e-10) -> 0.0
                import math
                log_prob = math.log10(prob)
                # Map log10(1e-10)=-10 to 0, log10(1e-2)=-2 to 1
                freq_score = max(0.0, min(1.0, (log_prob + 10) / 8))
            else:
                freq_score = 0.0

        # Combined score
        # Weight distance more heavily (60%) than frequency (40%)
        if use_frequency and frequency_data:
            score = (0.6 * distance_score) + (0.4 * freq_score)
        else:
            score = distance_score

        scored.append((candidate, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored


class SpellCorrector:
    """
    Spell corrector class for advanced usage.

    Provides stateful spell correction with custom settings.
    Similar to the Tokenizer class pattern in ShanNLP.

    Examples:
        >>> corrector = SpellCorrector()
        >>> corrector.is_correct("မိူင်း")
        True
        >>> suggestions = corrector.correct("မိုင်း")
        >>> len(suggestions) > 0
        True

        >>> # Custom dictionary
        >>> corrector = SpellCorrector(custom_dict={"ၵႃႈ", "မူၼ်း"})
        >>> corrector.is_correct("ၵႃႈ")
        True
    """

    def __init__(
        self,
        custom_dict: Optional[Union[Set[str], List[str], str]] = None,
        frequency_data: Optional[dict] = None,
        max_edit_distance: int = 2,
        use_phonetic: bool = True
    ):
        """
        Initialize spell corrector.

        Args:
            custom_dict: Custom dictionary (set, list, or file path)
            frequency_data: Word frequency mapping
            max_edit_distance: Maximum edit distance (1 or 2)
            use_phonetic: Use phonetic similarity

        Raises:
            ValueError: If custom_dict file path is invalid
        """
        # Load dictionary
        if custom_dict is None:
            # Convert frozenset to set for mutability
            self._dict = set(shan_words())
        elif isinstance(custom_dict, str):
            # Load from file
            self._dict = self._load_dict_from_file(custom_dict)
        elif isinstance(custom_dict, (list, tuple)):
            self._dict = set(custom_dict)
        elif isinstance(custom_dict, frozenset):
            # Convert frozenset to set
            self._dict = set(custom_dict)
        else:
            self._dict = custom_dict

        # Load frequency data
        if frequency_data is None:
            self._freq = load_frequency_data()
        else:
            self._freq = frequency_data

        # Load phonetic groups
        self._phonetic = load_phonetic_groups() if use_phonetic else {}

        # Settings
        self._max_distance = max_edit_distance
        self._use_phonetic = use_phonetic

    def _load_dict_from_file(self, filepath: str) -> Set[str]:
        """Load dictionary from file."""
        from shannlp.spell.validator import load_corpus_safe
        import os

        # Check if it's a corpus filename or full path
        if not os.path.isabs(filepath):
            # Treat as corpus filename
            return load_corpus_safe(filepath)
        else:
            # Full path - read directly (with validation)
            words = set()
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        try:
                            validate_input(word)
                            words.add(word)
                        except:
                            continue
            return words

    def correct(
        self,
        word: str,
        max_suggestions: int = 5,
        min_confidence: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Correct a single word.

        Args:
            word: Word to correct
            max_suggestions: Maximum suggestions to return
            min_confidence: Minimum confidence threshold

        Returns:
            List of (suggestion, confidence) tuples

        Examples:
            >>> corrector = SpellCorrector()
            >>> suggestions = corrector.correct("မိုင်း")
            >>> len(suggestions) > 0
            True
        """
        return spell_correct(
            word,
            custom_dict=self._dict,
            max_edit_distance=self._max_distance,
            max_suggestions=max_suggestions,
            use_frequency=True,
            use_phonetic=self._use_phonetic,
            min_confidence=min_confidence
        )

    def is_correct(self, word: str) -> bool:
        """
        Check if word is spelled correctly.

        Args:
            word: Word to check

        Returns:
            True if word is in dictionary

        Examples:
            >>> corrector = SpellCorrector()
            >>> corrector.is_correct("မိူင်း")
            True
        """
        try:
            validate_input(word)
            word = normalize_unicode(word)
            return word in self._dict
        except:
            return False

    def add_word(self, word: str, frequency: int = 1):
        """
        Add word to dictionary.

        Args:
            word: Word to add
            frequency: Optional frequency count

        Examples:
            >>> corrector = SpellCorrector()
            >>> corrector.add_word("မိူင်း")
            >>> corrector.is_correct("မိူင်း")
            True
        """
        validate_input(word)
        word = normalize_unicode(word)
        self._dict.add(word)

        # Add to frequency data if frequency provided
        if frequency > 0 and self._freq:
            # Calculate probability (simplified)
            # In practice, you'd recompute all probabilities
            prob = frequency / 100000.0  # Rough estimate
            self._freq[word] = prob

    def add_words(self, words: List[str]):
        """
        Add multiple words to dictionary.

        Args:
            words: List of words to add

        Examples:
            >>> corrector = SpellCorrector()
            >>> corrector.add_words(["word1", "word2"])
        """
        for word in words:
            self.add_word(word)

    def remove_word(self, word: str):
        """
        Remove word from dictionary.

        Args:
            word: Word to remove

        Examples:
            >>> corrector = SpellCorrector()
            >>> corrector.add_word("test")
            >>> corrector.remove_word("test")
            >>> corrector.is_correct("test")
            False
        """
        word = normalize_unicode(word)
        self._dict.discard(word)
        if self._freq and word in self._freq:
            del self._freq[word]
