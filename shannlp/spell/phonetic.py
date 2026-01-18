"""
Phonetic similarity for Shan language spell correction.

This module defines phonetically similar character groups based on Shan
phonology. Characters that sound similar are more likely to be confused
in spelling, so we treat them as closer in edit distance calculations.
"""

from typing import Dict, Set, Optional


# Phonetic groups based on Shan phonology
# Characters within each group are phonetically similar

CONSONANT_GROUPS = {
    # Velar stops (k-sounds)
    'ၵ': {'ၵ', 'ၷ', 'ၶ', 'ꧠ'},
    'ၷ': {'ၵ', 'ၷ', 'ၶ', 'ꧠ'},
    'ၶ': {'ၵ', 'ၷ', 'ၶ', 'ꧠ'},
    'ꧠ': {'ၵ', 'ၷ', 'ၶ', 'ꧠ'},

    # Alveolar stops (t-sounds)
    'တ': {'တ', 'ထ', 'ၻ'},
    'ထ': {'တ', 'ထ', 'ၻ'},
    'ၻ': {'တ', 'ထ', 'ၻ'},

    # Bilabial stops (p-sounds)
    'ပ': {'ပ', 'ၽ', 'ၾ', 'ၿ'},
    'ၽ': {'ပ', 'ၽ', 'ၾ', 'ၿ'},
    'ၾ': {'ပ', 'ၽ', 'ၾ', 'ၿ'},
    'ၿ': {'ပ', 'ၽ', 'ၾ', 'ၿ'},

    # Nasals (m-sounds)
    'မ': {'မ', 'ႀ'},
    'ႀ': {'မ', 'ႀ'},

    # Approximants (y/r-sounds)
    'ယ': {'ယ', 'ရ'},
    'ရ': {'ယ', 'ရ'},

    # Laterals and approximants (l/w-sounds)
    'လ': {'လ', 'ꩮ'},
    'ꩮ': {'လ', 'ꩮ'},
    'ဝ': {'ဝ', 'ꩮ'},

    # Fricatives (s/h-sounds)
    'သ': {'သ', 'ၺ', 'ႁ'},
    'ၺ': {'သ', 'ၺ', 'ႁ'},
    'ႁ': {'သ', 'ၺ', 'ႁ'},

    # Affricates (ch-sounds)
    'ၸ': {'ၸ', 'ၹ'},
    'ၹ': {'ၸ', 'ၹ'},
}

# Tone marks that are commonly confused
# Shan has 5 tones: 1 (unmarked), 2 (◌ႇ), 3 (◌ႈ), 4 (◌း), 5 (◌ႉ), 6 (◌ႊ)
TONE_GROUPS = {
    # Tones 2 and 3 (commonly confused)
    '\u1087': {'\u1087', '\u1088'},  # ႇ and ႈ
    '\u1088': {'\u1087', '\u1088', '\u1038'},  # ႈ, ႇ, and း

    # Tone 4 (း) can be confused with tone 3
    '\u1038': {'\u1038', '\u1088'},  # း and ႈ

    # Tone 6 (ႊ) can be confused with tone 3
    '\u108a': {'\u108a', '\u1088'},  # ႊ and ႈ

    # Tones 5 and 6 (rarely confused but grouped)
    '\u1089': {'\u1089', '\u108a'},  # ႉ and ႊ
    '\u108a': {'\u1089', '\u108a'},  # ႊ and ႉ
}

# Vowels grouped by position
# Shan vowels can appear in different positions relative to consonants
VOWEL_POSITION_GROUPS = {
    # Lead vowels (appear before consonant)
    'lead': {'\u1084', '\u1031', '\u103c'},  # ႄ, ေ, ြ

    # Follow vowels (appear after consonant)
    'follow': {'\u1083', '\u1062', '\u103b'},  # ႃ, ၢ, ျ

    # Above vowels (appear above consonant)
    'above': {
        '\u1085',  # ႅ
        '\u1035',  # ဵ
        '\u102d',  # ိ
        '\u102e',  # ီ
        '\u1086',  # ႆ
        '\u103a',  # ◌်
    },

    # Below vowels (appear below consonant)
    'below': {
        '\u102f',  # ု
        '\u1030',  # ူ
        '\u1082',  # ႂ
        '\u103d',  # ွ
    },
}


# Global cache for phonetic map
_PHONETIC_MAP: Optional[Dict[str, Set[str]]] = None


def build_phonetic_map() -> Dict[str, Set[str]]:
    """
    Build complete phonetic similarity map.

    Creates a dictionary mapping each character to a set of
    phonetically similar characters.

    Returns:
        Dictionary mapping characters to sets of similar characters

    Examples:
        >>> phonetic_map = build_phonetic_map()
        >>> 'ၵ' in phonetic_map['ၷ']  # ၷ and ၵ are similar k-sounds
        True
        >>> 'မ' not in phonetic_map['ၷ']  # ၷ and မ are not similar
        True
    """
    phonetic_map = {}

    # Add consonant groups
    for base_char, group in CONSONANT_GROUPS.items():
        # For each character, store the other characters in its group
        phonetic_map[base_char] = group - {base_char}

    # Add tone groups
    for base_tone, group in TONE_GROUPS.items():
        if base_tone in phonetic_map:
            # Already exists, update with tone group
            phonetic_map[base_tone].update(group - {base_tone})
        else:
            phonetic_map[base_tone] = group - {base_tone}

    # Add vowel position groups
    # For vowels in same position, they can be confused
    for position, group in VOWEL_POSITION_GROUPS.items():
        for vowel in group:
            if vowel in phonetic_map:
                phonetic_map[vowel].update(group - {vowel})
            else:
                phonetic_map[vowel] = group - {vowel}

    return phonetic_map


def load_phonetic_groups() -> Dict[str, Set[str]]:
    """
    Load phonetic similarity groups (cached).

    Returns cached phonetic map, or builds and caches it if needed.

    Returns:
        Dictionary mapping characters to sets of similar characters

    Examples:
        >>> groups = load_phonetic_groups()
        >>> len(groups) > 0
        True
    """
    global _PHONETIC_MAP

    if _PHONETIC_MAP is None:
        _PHONETIC_MAP = build_phonetic_map()

    return _PHONETIC_MAP


def are_phonetically_similar(char1: str, char2: str) -> bool:
    """
    Check if two characters are phonetically similar.

    Args:
        char1: First character
        char2: Second character

    Returns:
        True if characters are in the same phonetic group

    Examples:
        >>> are_phonetically_similar('ၷ', 'ၵ')  # Both k-sounds
        True
        >>> are_phonetically_similar('ၷ', 'မ')  # k vs m
        False
        >>> are_phonetically_similar('\u1087', '\u1088')  # Tones 2 and 3
        True
    """
    if char1 == char2:
        return True

    phonetic_map = load_phonetic_groups()

    if char1 in phonetic_map:
        return char2 in phonetic_map[char1]

    return False


def get_phonetic_distance(char1: str, char2: str) -> float:
    """
    Get phonetic distance between two characters.

    Distance values:
    - 0.0: Identical characters
    - 0.6: Phonetically similar (in same group)
    - 1.0: Not similar

    Args:
        char1: First character
        char2: Second character

    Returns:
        Distance value (0.0 = identical, 1.0 = completely different)

    Examples:
        >>> get_phonetic_distance('ၵ', 'ၵ')
        0.0
        >>> get_phonetic_distance('ၷ', 'ၵ')  # Similar k-sounds
        0.6
        >>> get_phonetic_distance('ၷ', 'မ')  # Different
        1.0
    """
    if char1 == char2:
        return 0.0

    if are_phonetically_similar(char1, char2):
        return 0.6

    return 1.0


def get_character_type(char: str) -> str:
    """
    Get the type of a Shan character.

    Useful for applying different edit costs based on character type.

    Args:
        char: Character to classify

    Returns:
        One of: 'consonant', 'tone', 'vowel_lead', 'vowel_follow',
                'vowel_above', 'vowel_below', 'other'

    Examples:
        >>> get_character_type('ၷ')
        'consonant'
        >>> get_character_type('\u1087')  # ႇ
        'tone'
        >>> get_character_type('\u1084')  # ႄ (lead vowel)
        'vowel_lead'
    """
    # Check if it's a consonant
    for char_key in CONSONANT_GROUPS:
        if char in CONSONANT_GROUPS[char_key] or char == char_key:
            return 'consonant'

    # Check if it's a tone mark
    for tone_key in TONE_GROUPS:
        if char in TONE_GROUPS[tone_key] or char == tone_key:
            return 'tone'

    # Check vowel positions
    if char in VOWEL_POSITION_GROUPS['lead']:
        return 'vowel_lead'
    if char in VOWEL_POSITION_GROUPS['follow']:
        return 'vowel_follow'
    if char in VOWEL_POSITION_GROUPS['above']:
        return 'vowel_above'
    if char in VOWEL_POSITION_GROUPS['below']:
        return 'vowel_below'

    return 'other'


def get_phonetic_substitutions(char: str) -> Set[str]:
    """
    Get all phonetically similar characters for a given character.

    Useful for generating phonetic edit candidates.

    Args:
        char: Character to find substitutions for

    Returns:
        Set of phonetically similar characters (excluding the input char)

    Examples:
        >>> subs = get_phonetic_substitutions('က')
        >>> 'ၵ' in subs  # k-sound variant
        True
        >>> len(subs) > 0
        True
    """
    phonetic_map = load_phonetic_groups()
    return phonetic_map.get(char, set())


def clear_cache():
    """
    Clear cached phonetic data.

    Useful for testing or if phonetic groups are updated.

    Examples:
        >>> load_phonetic_groups()  # Load and cache
        >>> clear_cache()  # Clear cache
        >>> load_phonetic_groups()  # Rebuild cache
    """
    global _PHONETIC_MAP
    _PHONETIC_MAP = None
