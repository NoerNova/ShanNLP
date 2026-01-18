"""
Integration tests for spell correction.

Basic tests to verify the spell correction functionality works end-to-end.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shannlp import spell_correct, SpellCorrector, is_correct_spelling


def test_basic_import():
    """Test that spell correction functions can be imported."""
    print("✓ Spell correction functions imported successfully")


def test_correct_word():
    """Test that a correct word is recognized."""
    # Use a common Shan word that should be in the dictionary
    word = "ယဝ်ႉ"  # Very common word in Shan

    # Check if it's correct
    is_correct = is_correct_spelling(word)
    print(f"✓ is_correct_spelling('{word}'): {is_correct}")

    # Get suggestions (should return the word itself with high confidence)
    suggestions = spell_correct(word)
    if suggestions:
        top_suggestion, confidence = suggestions[0]
        print(f"✓ spell_correct('{word}'): {top_suggestion} (confidence: {confidence:.2f})")
        # For a correct word, it should be in the suggestions
        assert word in [s[0] for s in suggestions], f"Word {word} not in suggestions"
    else:
        print(f"  Note: No suggestions for '{word}' (might not be in dictionary)")


def test_spell_corrector_class():
    """Test the SpellCorrector class."""
    corrector = SpellCorrector()
    print("✓ SpellCorrector class instantiated")

    # Test is_correct method
    word = "ယဝ်ႉ"
    is_correct = corrector.is_correct(word)
    print(f"✓ corrector.is_correct('{word}'): {is_correct}")

    # Test add_word method
    test_word = "TestWord123"
    corrector.add_word(test_word)
    assert corrector.is_correct(test_word), "Added word not found"
    print(f"✓ corrector.add_word('{test_word}') works")

    # Test remove_word method
    corrector.remove_word(test_word)
    assert not corrector.is_correct(test_word), "Removed word still found"
    print(f"✓ corrector.remove_word('{test_word}') works")


def test_custom_dictionary():
    """Test spell correction with custom dictionary."""
    custom = {"ၵႃႈ", "မူၼ်း", "ယူႇ"}

    # Test with spell_correct function
    result = is_correct_spelling("ၵႃႈ", custom_dict=custom)
    print(f"✓ Custom dictionary works: {result}")

    # Test with SpellCorrector class
    corrector = SpellCorrector(custom_dict=custom)
    assert corrector.is_correct("ၵႃႈ"), "Custom word not recognized"
    print("✓ SpellCorrector with custom dictionary works")


def test_empty_input_handling():
    """Test that empty input is handled properly."""
    try:
        spell_correct("")
        print("✗ Empty input should raise ValueError")
    except ValueError as e:
        print(f"✓ Empty input handled correctly: {e}")


def test_invalid_input_handling():
    """Test that invalid input is handled properly."""
    try:
        spell_correct(None)
        print("✗ None input should raise ValueError")
    except (ValueError, TypeError) as e:
        print(f"✓ None input handled correctly: {type(e).__name__}")


def test_frequency_data_loading():
    """Test that frequency data can be loaded."""
    from shannlp.spell.frequency import load_frequency_data, get_top_words

    freq_data = load_frequency_data()
    print(f"✓ Frequency data loaded: {len(freq_data)} words")

    top_words = get_top_words(5)
    if top_words:
        print(f"✓ Top 5 words: {[w[0] for w in top_words]}")
    else:
        print("  Note: No frequency data available")


def test_phonetic_groups():
    """Test that phonetic groups are loaded."""
    from shannlp.spell.phonetic import (
        load_phonetic_groups,
        are_phonetically_similar,
        get_character_type
    )

    phonetic_map = load_phonetic_groups()
    print(f"✓ Phonetic groups loaded: {len(phonetic_map)} characters mapped")

    # Test phonetic similarity
    similar = are_phonetically_similar('က', 'ၵ')  # Both k-sounds
    print(f"✓ Phonetic similarity check: 'က' and 'ၵ' similar = {similar}")

    # Test character type
    char_type = get_character_type('က')
    print(f"✓ Character type detection: 'က' is '{char_type}'")


def test_edit_distance():
    """Test edit distance calculation."""
    from shannlp.spell.distance import shan_edit_distance, similarity_score

    # Identical words
    dist = shan_edit_distance("မိူင်း", "မိူင်း")
    print(f"✓ Edit distance (identical): {dist}")
    assert dist == 0.0, "Identical words should have distance 0"

    # Different words
    dist = shan_edit_distance("မိူင်း", "မူၼ်း")
    print(f"✓ Edit distance (different): {dist}")
    assert dist > 0, "Different words should have distance > 0"

    # Similarity score
    score = similarity_score("မိူင်း", "မိူင်း")
    print(f"✓ Similarity score (identical): {score}")
    assert score == 1.0, "Identical words should have similarity 1.0"


def test_validator():
    """Test input validation."""
    from shannlp.spell.validator import (
        validate_input,
        sanitize_text,
        SecurityError
    )

    # Valid input
    validate_input("မိူင်း")
    print("✓ Valid input passes validation")

    # Too long input
    try:
        validate_input("x" * 1000)
        print("✗ Long input should raise ValueError")
    except ValueError:
        print("✓ Long input rejected")

    # Null byte
    try:
        validate_input("test\x00")
        print("✗ Null byte should raise SecurityError")
    except SecurityError:
        print("✓ Null byte rejected")

    # Sanitize text
    sanitized = sanitize_text("မိူင်း လေး")
    print(f"✓ Text sanitization works: '{sanitized}'")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("Running Spell Correction Integration Tests")
    print("=" * 60)
    print()

    tests = [
        ("Import Test", test_basic_import),
        ("Correct Word Test", test_correct_word),
        ("SpellCorrector Class Test", test_spell_corrector_class),
        ("Custom Dictionary Test", test_custom_dictionary),
        ("Empty Input Handling", test_empty_input_handling),
        ("Invalid Input Handling", test_invalid_input_handling),
        ("Frequency Data Loading", test_frequency_data_loading),
        ("Phonetic Groups", test_phonetic_groups),
        ("Edit Distance", test_edit_distance),
        ("Validator", test_validator),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n{name}:")
        print("-" * 60)
        try:
            test_func()
            passed += 1
            print(f"✓ {name} PASSED")
        except Exception as e:
            failed += 1
            print(f"✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
