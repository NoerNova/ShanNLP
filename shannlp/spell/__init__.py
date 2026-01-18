"""
Spell correction module for Shan language.

This module provides spell correction functionality using Peter Norvig's
algorithm enhanced with Shan-specific features:
- Weighted edit distance based on character types
- Phonetic similarity for consonants
- Frequency-based ranking using Wikipedia data

Examples:
    >>> from shannlp import spell_correct
    >>> suggestions = spell_correct("မိုင်း")
    >>> print(suggestions[0])
    ('မိူင်း', 0.95)

    >>> from shannlp import SpellCorrector
    >>> corrector = SpellCorrector()
    >>> corrector.is_correct("မိူင်း")
    True
"""

__all__ = [
    "spell_correct",
    "SpellCorrector",
    "is_correct_spelling"
]

from shannlp.spell.core import spell_correct, SpellCorrector
from shannlp.corpus import shan_words


def is_correct_spelling(word: str, custom_dict=None) -> bool:
    """
    Check if a word is spelled correctly.

    A convenience function that checks if a word exists in the
    dictionary without generating correction suggestions.

    Args:
        word: Word to check
        custom_dict: Optional custom dictionary (default: shan_words())

    Returns:
        True if word is in dictionary, False otherwise

    Examples:
        >>> is_correct_spelling("မိူင်း")
        True
        >>> is_correct_spelling("xyzabc")
        False

        >>> custom = {"ၵႃႈ", "မူၼ်း"}
        >>> is_correct_spelling("ၵႃႈ", custom_dict=custom)
        True
    """
    from shannlp.spell.validator import validate_input, normalize_unicode

    try:
        validate_input(word)
        word = normalize_unicode(word)
        dictionary = custom_dict if custom_dict is not None else shan_words()
        return word in dictionary
    except:
        return False
