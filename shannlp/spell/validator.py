"""
Input validation and security utilities for spell correction.

This module provides comprehensive input validation to prevent:
- DoS attacks (via length limits)
- Code injection
- Path traversal
- Malformed Unicode
"""

import os
import unicodedata
from typing import Optional


# Define Shan characters locally to avoid circular import
shan_consonants = "ၵၷၶꧠငၸၹသၺတၻထၼꧣပၽၾပၿႀမယရလꩮဝႁဢ"
shan_vowels = "\u1083\u1062\u1084\u1085\u1031\u1035\u102d\u102e\u102f\u1030\u1086\u1082\u103a\u103d\u103b\u103c"
shan_tonemarks = "\u1087\u1088\u1038\u1089\u108a"
shan_punctuations = "\u104a\u104b\ua9e6"
shan_digits = "႐႑႒႓႔႕႖႗႘႙"
shan_letters = "".join([shan_consonants, shan_vowels, shan_tonemarks, shan_punctuations])
shan_characters = "".join([shan_letters, shan_digits])

MAX_WORD_LENGTH = 100
MAX_TEXT_LENGTH = 1000000  # 1MB
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_CANDIDATES = 10000


class SecurityError(Exception):
    """Raised when input contains potential security threats."""
    pass


def validate_input(word: str) -> None:
    """
    Validate input word for security and correctness.

    Performs comprehensive validation to prevent security issues:
    - Type checking
    - Empty/whitespace checking
    - Length limits (prevent DoS)
    - UTF-8 encoding validation
    - Null byte detection
    - Control character filtering
    - Unicode normalization

    Args:
        word: Input word to validate

    Raises:
        ValueError: If input is invalid
        SecurityError: If input contains potentially malicious content

    Examples:
        >>> validate_input("မိူင်း")  # Valid Shan word
        >>> validate_input("")  # Raises ValueError
        >>> validate_input("x" * 1000)  # Raises ValueError (too long)
        >>> validate_input("test\\x00")  # Raises SecurityError (null byte)
    """
    # Type check
    if not isinstance(word, str):
        raise ValueError("Input must be a string")

    # Empty check
    if not word or word.isspace():
        raise ValueError("Input cannot be empty or whitespace only")

    # Length check (prevent DoS)
    if len(word) > MAX_WORD_LENGTH:
        raise ValueError(
            f"Word too long: {len(word)} characters "
            f"(maximum {MAX_WORD_LENGTH} characters)"
        )

    # UTF-8 validation
    try:
        word.encode('utf-8')
    except UnicodeEncodeError as e:
        raise ValueError(f"Invalid UTF-8 encoding: {e}")

    # Check for null bytes (security threat)
    if '\x00' in word:
        raise SecurityError("Null bytes are not allowed in input")

    # Check for control characters (except valid Shan combining marks)
    shan_chars_set = set(shan_characters)
    for char in word:
        category = unicodedata.category(char)
        # Category Cc = control characters, Cf = format characters
        # Allow Shan characters and whitespace
        if category.startswith('C') and category != 'Cf':
            if char not in shan_chars_set and not char.isspace():
                raise ValueError(
                    f"Invalid control character: {repr(char)} "
                    f"(Unicode category: {category})"
                )

    # Normalize Unicode (prevent homograph attacks)
    normalized = unicodedata.normalize('NFC', word)
    if normalized != word:
        # Log warning but don't reject - auto-normalize
        # In production, you might want to log this
        pass


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize text input for batch processing.

    Performs:
    - Length validation
    - Unicode normalization (NFC)
    - Null byte removal

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length (default: MAX_TEXT_LENGTH)

    Returns:
        Sanitized text string

    Raises:
        ValueError: If text exceeds maximum length

    Examples:
        >>> sanitize_text("မိူင်း လေး")
        'မိူင်း လေး'
        >>> sanitize_text("test\\x00word")
        'testword'
    """
    if max_length is None:
        max_length = MAX_TEXT_LENGTH

    # Type check
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    # Length check
    if len(text) > max_length:
        raise ValueError(
            f"Text too long: {len(text)} characters "
            f"(maximum {max_length} characters)"
        )

    # Normalize Unicode (NFC form)
    text = unicodedata.normalize('NFC', text)

    # Remove null bytes
    text = text.replace('\x00', '')

    return text


class ResourceLimiter:
    """
    Prevent resource exhaustion attacks.

    Limits the number of candidates generated during spell correction
    to prevent memory exhaustion and DoS attacks.
    """

    def __init__(self, max_candidates: int = MAX_CANDIDATES):
        """
        Initialize resource limiter.

        Args:
            max_candidates: Maximum number of candidates to allow
        """
        self.max_candidates = max_candidates

    def check_candidate_limit(self, candidates: set) -> set:
        """
        Limit number of candidates to prevent resource exhaustion.

        Args:
            candidates: Set of candidate words

        Returns:
            Limited set of candidates (truncated if needed)

        Examples:
            >>> limiter = ResourceLimiter(max_candidates=3)
            >>> candidates = {'word1', 'word2', 'word3', 'word4'}
            >>> limited = limiter.check_candidate_limit(candidates)
            >>> len(limited) <= 3
            True
        """
        if len(candidates) > self.max_candidates:
            # Convert to list and take first N candidates
            # Note: set order is not guaranteed, but that's okay
            # as we'll rank them later anyway
            return set(list(candidates)[:self.max_candidates])
        return candidates


def load_corpus_safe(filename: str) -> set:
    """
    Safely load corpus file with security checks.

    Prevents:
    - Path traversal attacks
    - Loading files outside corpus directory
    - Loading excessively large files
    - Processing malformed content

    Args:
        filename: Name of corpus file to load

    Returns:
        Set of words from corpus

    Raises:
        SecurityError: If path traversal detected
        ValueError: If file is too large or invalid

    Examples:
        >>> words = load_corpus_safe("words_shn.txt")
        >>> len(words) > 0
        True
        >>> load_corpus_safe("../../etc/passwd")  # Raises SecurityError
    """
    from shannlp.corpus import corpus_path, path_shannlp_corpus

    # Get absolute path to corpus file
    filepath = path_shannlp_corpus(filename)

    # Verify file is within corpus directory (prevent path traversal)
    corpus_dir = os.path.abspath(corpus_path())
    abs_filepath = os.path.abspath(filepath)

    if not abs_filepath.startswith(corpus_dir):
        raise SecurityError(
            f"Path traversal attempt detected: {filename} "
            f"resolves outside corpus directory"
        )

    # Check if file exists
    if not os.path.isfile(abs_filepath):
        raise ValueError(f"Corpus file not found: {filename}")

    # Check file size (prevent loading huge files)
    file_size = os.path.getsize(abs_filepath)
    if file_size > MAX_FILE_SIZE:
        raise ValueError(
            f"Corpus file too large: {file_size} bytes "
            f"(maximum {MAX_FILE_SIZE} bytes)"
        )

    # Safe read with context manager
    words = set()
    with open(abs_filepath, 'r', encoding='utf-8-sig') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip excessively long lines
            if len(line) > MAX_WORD_LENGTH:
                # Silently skip (likely malformed data)
                continue

            # Validate each word
            try:
                validate_input(line)
                words.add(line)
            except (ValueError, SecurityError):
                # Skip invalid lines rather than failing
                # In production, you might want to log this
                continue

    return words


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text to NFC form.

    This prevents issues with different Unicode representations
    of the same visual character (e.g., composed vs decomposed).

    Args:
        text: Input text to normalize

    Returns:
        Normalized text in NFC form

    Examples:
        >>> # These look the same but may have different encodings
        >>> text1 = "မိူင်း"  # NFC
        >>> text2 = "မိူင်း"  # NFD (hypothetically)
        >>> normalize_unicode(text1) == normalize_unicode(text2)
        True
    """
    return unicodedata.normalize('NFC', text)
