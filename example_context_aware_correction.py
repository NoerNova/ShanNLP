"""
Context-Aware Spell Correction Examples for ShanNLP (Phase 2)

This file demonstrates how to use context-aware spell correction with n-gram models.
"""

import os
import time


def example_train_model():
    """Example: Train n-gram model from your Shan corpus."""
    print("=" * 70)
    print("Example 1: Training N-gram Model from Corpus")
    print("=" * 70)
    print()

    print("IMPORTANT: You need to provide your Shan text corpus!")
    print()
    print("Steps to train:")
    print("1. Organize your Shan text files in a directory")
    print("   - Format: UTF-8 encoded .txt files")
    print("   - Size: At least 1-10 MB recommended")
    print()
    print("2. Run the training script:")
    print()
    print("   python train_ngram_model.py \\")
    print("       --corpus_dir /path/to/your/shan/texts \\")
    print("       --output shan_bigram_model.msgpack \\")
    print("       --ngram 2")
    print()
    print("3. For trigram (better accuracy, slightly slower):")
    print()
    print("   python train_ngram_model.py \\")
    print("       --corpus_dir /path/to/your/shan/texts \\")
    print("       --output shan_trigram_model.msgpack \\")
    print("       --ngram 3")
    print()
    print("=" * 70)
    print()


def example_basic_usage():
    """Example: Basic context-aware correction."""
    print("=" * 70)
    print("Example 2: Basic Context-Aware Correction")
    print("=" * 70)
    print()

    from shannlp.spell import ContextAwareCorrector

    # Create corrector
    corrector = ContextAwareCorrector()

    # Check if model exists
    model_path = "shan_ngram_model.msgpack"
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        print("Please train a model first (see Example 1)")
        print()
        return

    # Load pre-trained model
    print("Loading n-gram model...")
    corrector.load_model(model_path)
    print()

    # Correct a sentence
    test_sentence = "မိူင်း ယူႇ လႄႈ"  # Example Shan sentence
    print(f"Input: {test_sentence}")

    result = corrector.correct_sentence(test_sentence)
    print(f"Output: {result}")
    print()

    # Show performance
    stats = corrector.get_performance_stats()
    print(f"Performance: {stats['avg_time_ms']:.2f} ms")
    print()


def example_compare_with_without_context():
    """Example: Compare corrections with and without context."""
    print("=" * 70)
    print("Example 3: Context-Aware vs Word-Level Correction")
    print("=" * 70)
    print()

    from shannlp.spell import spell_correct, ContextAwareCorrector

    model_path = "shan_ngram_model.msgpack"
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        print("Please train a model first")
        print()
        return

    # Test sentence with ambiguous correction
    test_text = "မိူင်း ယူႇ လႄႈ"  # Replace with actual misspelled text

    print(f"Input text: {test_text}")
    print()

    # Word-level correction (Phase 1)
    print("Phase 1 - Word-level correction:")
    from shannlp.tokenize import word_tokenize
    tokens = word_tokenize(test_text, keep_whitespace=False)
    for token in tokens:
        suggestions = spell_correct(token, max_suggestions=3)
        if suggestions:
            print(f"  '{token}' → {[w for w, _ in suggestions[:3]]}")
    print()

    # Context-aware correction (Phase 2)
    print("Phase 2 - Context-aware correction:")
    corrector = ContextAwareCorrector()
    corrector.load_model(model_path)

    result = corrector.correct_text(test_text)
    print(f"  Result: {' '.join(result)}")
    print()

    # Performance comparison
    stats = corrector.get_performance_stats()
    print(f"Context-aware time: {stats['avg_time_ms']:.2f} ms")
    if 'cache_hit_rate' in stats:
        print(f"N-gram cache hit rate: {stats['hit_rate']:.1%}")
    print()


def example_batch_processing():
    """Example: Batch process multiple texts."""
    print("=" * 70)
    print("Example 4: Batch Processing")
    print("=" * 70)
    print()

    from shannlp.spell import ContextAwareCorrector

    model_path = "shan_ngram_model.msgpack"
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        print()
        return

    corrector = ContextAwareCorrector()
    corrector.load_model(model_path)

    # Example texts
    test_texts = [
        "မိူင်း ယူႇ",
        "လႄႈ မူၼ်း",
        "ၵႃႈ တီႈ",
    ]

    print("Processing multiple texts...")
    print()

    start_time = time.time()
    results = corrector.batch_correct(test_texts, show_progress=False)
    elapsed = time.time() - start_time

    for i, (original, corrected) in enumerate(zip(test_texts, results), 1):
        print(f"{i}. Input:  {original}")
        print(f"   Output: {' '.join(corrected)}")
        print()

    print(f"Total time: {elapsed*1000:.2f} ms")
    print(f"Average per text: {(elapsed/len(test_texts))*1000:.2f} ms")
    print()


def example_realtime_typing():
    """Example: Simulate real-time typing assistance."""
    print("=" * 70)
    print("Example 5: Real-Time Typing Assistance Simulation")
    print("=" * 70)
    print()

    from shannlp.spell import ContextAwareCorrector

    model_path = "shan_ngram_model.msgpack"
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        print()
        return

    corrector = ContextAwareCorrector(context_window=1)  # Smaller window for speed
    corrector.load_model(model_path)

    print("Simulating typing with real-time correction...")
    print("(Target: <100ms per word for smooth typing experience)")
    print()

    # Simulate typing word by word
    sentence = "မိူင်း ယူႇ လႄႈ မူၼ်း"
    words = sentence.split()

    cumulative_text = ""
    for i, word in enumerate(words, 1):
        cumulative_text += (" " if cumulative_text else "") + word

        start = time.time()
        result = corrector.correct_sentence(cumulative_text)
        elapsed_ms = (time.time() - start) * 1000

        status = "✓" if elapsed_ms < 100 else "⚠️"
        print(f"{status} Word {i}: '{word}' → {elapsed_ms:.1f} ms")

    print()
    stats = corrector.get_performance_stats()
    print(f"Average latency: {stats['avg_time_ms']:.2f} ms")
    print(f"Max latency: {stats['max_time_ms']:.2f} ms")

    if stats['avg_time_ms'] < 100:
        print("✓ Performance suitable for real-time typing!")
    else:
        print("⚠️  Consider using bigrams or reducing context window for better performance")
    print()


def example_model_evaluation():
    """Example: Evaluate n-gram model quality."""
    print("=" * 70)
    print("Example 6: Model Evaluation")
    print("=" * 70)
    print()

    from shannlp.spell import NgramModel

    model_path = "shan_ngram_model.msgpack"
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        print()
        return

    # Load model
    model = NgramModel.load(model_path)

    print(f"Model statistics:")
    print(f"  - N-gram order: {model.n}")
    print(f"  - Vocabulary size: {len(model.vocabulary)}")
    print(f"  - Unique contexts: {len(model.context_counts)}")
    print()

    # Test probability calculations
    print("Testing probability calculations:")

    test_word = "ယူႇ"
    test_context = ["မိူင်း"]

    try:
        prob = model.probability(test_word, test_context)
        log_prob = model.log_probability(test_word, test_context)

        print(f"  P('{test_word}' | {test_context}) = {prob:.6f}")
        print(f"  log P('{test_word}' | {test_context}) = {log_prob:.2f}")
    except Exception as e:
        print(f"  Error: {e}")

    print()

    # Cache statistics
    cache_stats = model.get_cache_stats()
    print("Cache performance:")
    print(f"  - Cache size: {cache_stats['cache_size']}")
    print(f"  - Hit rate: {cache_stats['hit_rate']:.1%}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "ShanNLP Phase 2: Context-Aware Correction" + " " * 11 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    examples = [
        ("Training Model", example_train_model),
        ("Basic Usage", example_basic_usage),
        ("With vs Without Context", example_compare_with_without_context),
        ("Batch Processing", example_batch_processing),
        ("Real-Time Typing", example_realtime_typing),
        ("Model Evaluation", example_model_evaluation),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 70)
    print("Examples completed!")
    print()
    print("Next steps:")
    print("1. Train your n-gram model with: python train_ngram_model.py")
    print("2. Test with your Shan corpus")
    print("3. Integrate into your application")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
