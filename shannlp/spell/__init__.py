"""
Spell correction module for Shan language.

This module provides spell correction functionality using Peter Norvig's
algorithm enhanced with Shan-specific features:
- Weighted edit distance based on character types
- Phonetic similarity for consonants
- Frequency-based ranking using Wikipedia data
- Context-aware correction using n-gram models

Simple Usage:
    # Single word correction
    >>> from shannlp import spell_correct
    >>> suggestions = spell_correct("မိုင်း")
    >>> print(suggestions[0][0])
    'မိူင်း'

    # Sentence correction (context-aware)
    >>> from shannlp import correct_sentence
    >>> result = correct_sentence("ၵူၼ်မိူင်း ၵိၼ် ၶဝ်ႈ")
    >>> print(result)

Advanced Usage:
    >>> from shannlp import SpellCorrector, ContextAwareCorrector
    >>> corrector = SpellCorrector()
    >>> corrector.is_correct("မိူင်း")
    True
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Set, Union

__all__ = [
    # Simple API (recommended)
    "spell_correct",
    "correct_sentence",
    "correct_text",
    "is_correct_spelling",
    "reload_model",
    "load_neural_model",
    # Advanced API
    "SpellCorrector",
    "ContextAwareCorrector",
    "NgramModel",
]

# Optional neural model support
try:
    from shannlp.spell.neural import SpellReranker
    __all__.append("SpellReranker")
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False

from shannlp.spell.core import spell_correct, SpellCorrector
from shannlp.spell.context import ContextAwareCorrector
from shannlp.spell.ngram import NgramModel
from shannlp.corpus import shan_words

# Global cached corrector for lazy loading
_context_corrector: Optional[ContextAwareCorrector] = None
_default_model_loaded: bool = False

# Default model path (relative to this file)
_DEFAULT_MODEL_NAME = "shan_bigram.msgpack"


def _get_default_model_path() -> str:
    """Get path to default n-gram model bundled with package."""
    return os.path.join(os.path.dirname(__file__), _DEFAULT_MODEL_NAME)


def _get_context_corrector(
    model_path: Optional[str] = None,
    force_reload: bool = False
) -> ContextAwareCorrector:
    """
    Get or create the global context-aware corrector.

    Uses lazy loading - model is only loaded on first use.

    Args:
        model_path: Custom model path (None = use default bundled model)
        force_reload: Force reload model even if already loaded

    Returns:
        ContextAwareCorrector instance with loaded model
    """
    global _context_corrector, _default_model_loaded

    # Return cached corrector if available and not forcing reload
    if _context_corrector is not None and not force_reload:
        # If custom model requested but we have default loaded, reload
        if model_path is not None and _default_model_loaded:
            pass  # Fall through to reload
        else:
            return _context_corrector

    # Create new corrector
    _context_corrector = ContextAwareCorrector()

    # Determine model path
    if model_path is None:
        model_path = _get_default_model_path()
        _default_model_loaded = True
    else:
        _default_model_loaded = False

    # Load model if it exists
    if os.path.exists(model_path):
        _context_corrector.load_ngram_model(model_path)
    else:
        # No model available - will use basic spell correction only
        import warnings
        warnings.warn(
            f"N-gram model not found at {model_path}. "
            "Context-aware correction will fall back to basic spell correction. "
            "Train a model with train_ngram_model.py for better results.",
            UserWarning
        )

    return _context_corrector


def correct_sentence(
    sentence: str,
    model_path: Optional[str] = None,
    min_confidence: float = 0.3,
    use_context: bool = True,
    separator: str = " "
) -> str:
    """
    Correct spelling errors in a Shan sentence.

    This is the recommended function for sentence-level spell correction.
    It uses context from surrounding words to make better corrections.

    The n-gram model is automatically loaded on first use (lazy loading).
    If a neural model has been loaded via load_neural_model(), it will be
    used for improved context-aware reranking.

    Args:
        sentence: Input sentence to correct
        model_path: Custom n-gram model path (None = use default bundled model)
        min_confidence: Minimum confidence threshold for corrections (0.0-1.0)
        use_context: Use n-gram context for better accuracy (recommended)
        separator: String to join corrected tokens (default: " ").
                   Use "" for no spaces (traditional Shan text).

    Returns:
        Corrected sentence as string

    Examples:
        >>> from shannlp import correct_sentence

        # Basic usage (with spaces between tokens)
        >>> result = correct_sentence("ၵူၼ်မိူင်း ၵိၼ် ၶဝ်ႈ")
        >>> print(result)

        # No spaces (traditional Shan)
        >>> result = correct_sentence("ၵူၼ်မိူင်းၵိၼ်ၶဝ်ႈ", separator="")
        >>> print(result)

        # With neural model for better context-aware correction
        >>> from shannlp.spell import load_neural_model
        >>> load_neural_model("spell_reranker.pt")
        >>> result = correct_sentence("မိူင်တႆးပဵၼ်မိူင်းၶိုၼ်ႉယႂ်")
        >>> print(result)  # Neural model helps correct valid-but-wrong words

        # Without context (faster, less accurate)
        >>> result = correct_sentence("text here", use_context=False)
    """
    if not use_context:
        # Use basic tokenize + spell_correct without n-gram context
        from shannlp.tokenize import word_tokenize
        tokens = word_tokenize(sentence, keep_whitespace=False)
        corrected = []
        for token in tokens:
            suggestions = spell_correct(token, max_suggestions=1)
            if suggestions and suggestions[0][1] >= min_confidence:
                corrected.append(suggestions[0][0])
            else:
                corrected.append(token)
        return separator.join(corrected)

    # Use context-aware correction
    corrector = _get_context_corrector(model_path)
    result = corrector.correct_sentence(sentence)

    # Apply custom separator if not default space
    if separator != " ":
        result = separator.join(result.split())

    return result


def correct_text(
    text: str,
    model_path: Optional[str] = None,
    min_confidence: float = 0.3
) -> List[str]:
    """
    Correct spelling errors in text and return as token list.

    Similar to correct_sentence() but returns individual tokens
    instead of joined string.

    Args:
        text: Input text to correct
        model_path: Custom n-gram model path (None = use default)
        min_confidence: Minimum confidence threshold

    Returns:
        List of corrected tokens

    Examples:
        >>> from shannlp.spell import correct_text
        >>> tokens = correct_text("ၵူၼ်မိူင်း ၵိၼ် ၶဝ်ႈ")
        >>> print(tokens)
        ['ၵူၼ်းမိူင်း', 'ၵိၼ်', 'ၶဝ်ႈ']
    """
    corrector = _get_context_corrector(model_path)
    return corrector.correct_text(text, min_confidence=min_confidence)


def is_correct_spelling(word: str, custom_dict: Optional[Set[str]] = None) -> bool:
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
        >>> from shannlp import is_correct_spelling
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
    except Exception:
        return False


def reload_model(model_path: Optional[str] = None) -> None:
    """
    Reload the n-gram model.

    Use this if you've trained a new model and want to use it
    without restarting Python.

    Args:
        model_path: Path to new model (None = reload default)

    Examples:
        >>> from shannlp.spell import reload_model
        >>> reload_model("new_model.msgpack")
    """
    _get_context_corrector(model_path, force_reload=True)
    print("Model reloaded successfully")


def load_neural_model(model_path: str, device: Optional[str] = None) -> None:
    """
    Load a neural reranker model for improved spell correction.

    The neural model uses context to make better correction decisions,
    especially for words that are valid but incorrect in context.

    Args:
        model_path: Path to trained neural model (.pt file)
        device: Device to use (cuda, mps, cpu). Auto-detected if None.

    Examples:
        >>> from shannlp.spell import load_neural_model, correct_sentence
        >>> load_neural_model("spell_reranker.pt")
        >>> result = correct_sentence("မိူင်တႆးပဵၼ်မိူင်းၶိုၼ်ႉယႂ်")
        >>> print(result)

    Note:
        Requires PyTorch. Install with: pip install torch
        Train a model using train_neural_reranker.py
    """
    if not NEURAL_AVAILABLE:
        raise ImportError(
            "Neural model support requires PyTorch. "
            "Install with: pip install torch"
        )

    corrector = _get_context_corrector()
    corrector.load_neural_model(model_path, device)
    print(f"Neural model loaded. Using neural reranking for improved accuracy.")
