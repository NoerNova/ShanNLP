# Spell Correction

## Overview

ShanNLP's spell corrector is based on Peter Norvig's algorithm, enhanced with Shan-specific features:

- **Weighted edit distance** — tone marks, vowels, and consonants have different substitution costs
- **Phonetic similarity** — consonants that sound alike are preferred substitutions
- **Frequency ranking** — common words from Shan Wikipedia are ranked higher
- **Context-aware correction** — n-gram language models improve sentence-level accuracy

## Checking Spelling

Use `is_correct_spelling` to check whether a word exists in the dictionary:

```python
from shannlp import is_correct_spelling

print(is_correct_spelling("မိူင်း"))   # True
print(is_correct_spelling("မိုင်း"))   # False
```

## Correcting a Single Word

`spell_correct` returns a ranked list of `(suggestion, confidence)` tuples:

```python
from shannlp import spell_correct

suggestions = spell_correct("မိုင်း")
print(suggestions)
# [('မိူင်း', 0.95), ('မိူင်', 0.82), ...]

# Use only the top suggestion
best = spell_correct("မိုင်း", max_suggestions=1)[0][0]
print(best)  # မိူင်း
```

The confidence score combines edit distance (60%) and word frequency (40%). Higher is better.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `word` | str | required | Word to correct |
| `custom_dict` | set | None | Custom dictionary (default: built-in corpus) |
| `max_edit_distance` | int | 2 | Maximum edit distance to search (1 or 2) |
| `max_suggestions` | int | 5 | Maximum number of suggestions to return |
| `use_frequency` | bool | True | Weight suggestions by corpus frequency |
| `use_phonetic` | bool | True | Include phonetically similar candidates |
| `min_confidence` | float | 0.1 | Minimum confidence threshold for results |
| `always_suggest_alternatives` | bool | False | Return alternatives even for correct words |

## Correcting a Sentence

`correct_sentence` is the recommended function for most use cases. It tokenizes the input and uses surrounding context to improve corrections:

```python
from shannlp import correct_sentence

result = correct_sentence("ၵူၼ်မိူင်း ၵိၼ် ၶဝ်ႈ")
print(result)
```

For traditional Shan text without spaces between words, use `separator=""`:

```python
result = correct_sentence("ၵူၼ်မိူင်းၵိၼ်ၶဝ်ႈ", separator="")
print(result)
```

To skip context (faster, less accurate):

```python
result = correct_sentence("ၵူၼ်မိူင်း ၵိၼ် ၶဝ်ႈ", use_context=False)
```

> The n-gram model is loaded on first use (lazy loading). Expect ~1 second delay the first time.

## Getting Corrected Tokens

`correct_text` works like `correct_sentence` but returns a list of tokens instead of a joined string:

```python
from shannlp import correct_text

tokens = correct_text("ၵူၼ်မိူင်း ၵိၼ် ၶဝ်ႈ")
print(tokens)
# ['ၵူၼ်းမိူင်း', 'ၵိၼ်', 'ၶဝ်ႈ']
```

## Advanced: SpellCorrector Class

Use the `SpellCorrector` class when you need a stateful corrector with custom settings or a dynamic dictionary:

```python
from shannlp import SpellCorrector

corrector = SpellCorrector(max_edit_distance=2, use_phonetic=True)

# Check and correct
print(corrector.is_correct("မိူင်း"))   # True
suggestions = corrector.correct("မိုင်း")

# Manage the dictionary dynamically
corrector.add_word("ၵႃႈ")
corrector.add_words(["word1", "word2"])
corrector.remove_word("word1")
```

## Advanced: ContextAwareCorrector Class

For full control over context-aware correction, use `ContextAwareCorrector` directly:

```python
from shannlp.spell import ContextAwareCorrector

corrector = ContextAwareCorrector(
    context_window=2,    # words of context to consider
    context_weight=0.3   # how much context influences the score
)

# Load a custom n-gram model
corrector.load_ngram_model("my_trigram_model.msgpack")

# Correct text
result = corrector.correct_sentence("your text here")

# Batch processing
results = corrector.batch_correct(["text one", "text two"], show_progress=True)

# Check performance
stats = corrector.get_performance_stats()
print(f"Average time: {stats['avg_time_ms']:.2f}ms")
```

## Training a Custom N-gram Model

Train your own model for domain-specific or improved correction:

```bash
# Standard training
python train_ngram_model.py \
    --corpus_dir ./data/corpus \
    --output shan_bigram.msgpack \
    --ngram 2 \
    --min-count 2

# With pre-tokenized corpus (faster)
python tokenize_corpus.py --input ./data/corpus --output ./data/tokenized
python train_ngram_model.py \
    --corpus_dir ./data/tokenized \
    --output shan_bigram.msgpack \
    --tokenized
```

**Training options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--corpus_dir` | required | Directory containing `.txt` corpus files |
| `--output` | shan_ngram_model.msgpack | Output model path |
| `--ngram` | 2 | N-gram order (2 = bigram, 3 = trigram) |
| `--smoothing` | 1.0 | Laplace smoothing parameter |
| `--min-count` | 2 | Minimum word count to keep |
| `--tokenized` | False | Corpus is already tokenized |
| `--test_split` | 0.1 | Fraction reserved for testing |

Load your trained model:

```python
from shannlp.spell import reload_model

reload_model("shan_bigram.msgpack")
```

## Error Types Handled

| Error Type | Example | Description |
|------------|---------|-------------|
| Tone mark errors | ႇ ↔ ႈ ↔ း | Confused tone marks (low substitution cost) |
| Vowel position | Lead/follow/above/below vowels | Wrong vowel placement |
| Phonetic similarity | ပ ↔ ၽ, ၵ ↔ ၶ | Similar sounding consonants |
| Keyboard typos | Insert, delete, transpose | Standard typing errors |
| Lead vowel transposition | ေၶ → ၶေ | Vowel typed before consonant |

## Performance

| Operation | Target |
|-----------|--------|
| Single word correction | < 100ms |
| Sentence correction | < 500ms |
| Model loading (first use) | ~1s |

## Full API Reference

See [api/spell.md](../api/spell.md) for complete parameter documentation.
