# Phase 2: Context-Aware Spell Correction for ShanNLP

## Overview

Phase 2 adds context-aware spell correction using n-gram language models. This improves accuracy by considering surrounding words when making correction decisions, making it ideal for real-time typing assistance.

## Key Features

✅ **N-gram Language Models** - Bigram and trigram support with Laplace smoothing
✅ **Context-Aware Ranking** - Re-ranks candidates using word co-occurrence
✅ **Real-Time Optimized** - <100ms latency for typing assistance
✅ **Efficient Caching** - Frequently used n-grams cached for speed
✅ **Batch Processing** - Process multiple texts efficiently

## Quick Start

### Step 1: Prepare Your Corpus

Organize your Shan text files:
```
my_shan_corpus/
├── text1.txt
├── text2.txt
├── text3.txt
└── ...
```

- **Format**: UTF-8 encoded `.txt` files
- **Size**: At least 1-10 MB recommended (more is better)
- **Content**: Books, articles, social media posts, etc.

### Step 2: Train N-gram Model

Train a bigram model (faster, good for real-time):
```bash
python train_ngram_model.py \
    --corpus_dir /path/to/my_shan_corpus \
    --output shan_bigram_model.msgpack \
    --ngram 2
```

Or train a trigram model (more accurate, slightly slower):
```bash
python train_ngram_model.py \
    --corpus_dir /path/to/my_shan_corpus \
    --output shan_trigram_model.msgpack \
    --ngram 3
```

**Training Options**:
- `--smoothing 1.0`: Laplace smoothing parameter (default: 1.0)
- `--test_split 0.1`: Fraction of data for testing (default: 10%)

### Step 3: Use Context-Aware Correction

```python
from shannlp.spell import ContextAwareCorrector

# Create corrector
corrector = ContextAwareCorrector()

# Load trained model
corrector.load_model("shan_bigram_model.msgpack")

# Correct a sentence
input_text = "your Shan text here"
result = corrector.correct_sentence(input_text)
print(result)
```

## Usage Examples

### Basic Correction

```python
from shannlp.spell import ContextAwareCorrector

corrector = ContextAwareCorrector()
corrector.load_model("shan_bigram_model.msgpack")

# Correct single sentence
result = corrector.correct_sentence("မိူင်း ယူႇ လႄႈ")
print(f"Corrected: {result}")

# Get performance stats
stats = corrector.get_performance_stats()
print(f"Latency: {stats['avg_time_ms']:.2f} ms")
```

### Real-Time Typing Assistance

```python
# Optimize for speed (use smaller context window)
corrector = ContextAwareCorrector(
    context_window=1,  # Look at 1 word before/after
    context_weight=0.3  # Balance between edit distance and context
)

corrector.load_model("shan_bigram_model.msgpack")

# Process as user types
user_input = "မိူင်း ယူႇ"
result = corrector.correct_text(user_input)

# Should be < 100ms for smooth typing experience
```

### Batch Processing

```python
# Process multiple documents
texts = [
    "text 1 here",
    "text 2 here",
    "text 3 here"
]

results = corrector.batch_correct(texts, show_progress=True)

for original, corrected in zip(texts, results):
    print(f"Original: {original}")
    print(f"Corrected: {' '.join(corrected)}")
```

### Compare with Word-Level Correction

```python
from shannlp.spell import spell_correct, ContextAwareCorrector
from shannlp.tokenize import word_tokenize

text = "your text here"

# Phase 1: Word-level correction (no context)
tokens = word_tokenize(text, keep_whitespace=False)
for token in tokens:
    suggestions = spell_correct(token)
    if suggestions:
        print(f"Phase 1: '{token}' → '{suggestions[0][0]}'")

# Phase 2: Context-aware correction
corrector = ContextAwareCorrector()
corrector.load_model("shan_bigram_model.msgpack")

result = corrector.correct_text(text)
print(f"Phase 2: {' '.join(result)}")
```

## API Reference

### ContextAwareCorrector

```python
class ContextAwareCorrector:
    def __init__(
        self,
        ngram_model: Optional[NgramModel] = None,
        spell_corrector: Optional[SpellCorrector] = None,
        context_window: int = 2,
        context_weight: float = 0.3
    ):
        """
        Initialize context-aware corrector.

        Args:
            ngram_model: Pre-trained n-gram model
            spell_corrector: Spell corrector instance
            context_window: Number of context words (1-2 for real-time)
            context_weight: Weight for context vs edit distance (0-1)
        """
```

**Methods**:

- `load_model(model_path)` - Load pre-trained n-gram model
- `correct_text(text, max_suggestions=1, min_confidence=0.3)` - Correct text, returns list of tokens
- `correct_sentence(sentence)` - Correct and return as string
- `batch_correct(texts, show_progress=True)` - Process multiple texts
- `get_performance_stats()` - Get latency and cache statistics

### NgramModel

```python
class NgramModel:
    def __init__(self, n: int = 2, smoothing: float = 1.0):
        """
        Initialize n-gram model.

        Args:
            n: N-gram order (2=bigram, 3=trigram)
            smoothing: Laplace smoothing parameter
        """
```

**Methods**:

- `train(texts, tokenize_func=None)` - Train on corpus
- `probability(word, context)` - Calculate P(word|context)
- `log_probability(word, context)` - Calculate log P(word|context)
- `score_sequence(words)` - Score entire word sequence
- `perplexity(test_texts)` - Calculate perplexity on test set
- `save(filepath)` - Save model to file
- `load(filepath)` - Load model from file (classmethod)

## Performance Optimization

### For Real-Time Typing (<100ms)

1. **Use bigrams instead of trigrams**:
   ```python
   python train_ngram_model.py --ngram 2  # Faster
   ```

2. **Reduce context window**:
   ```python
   corrector = ContextAwareCorrector(context_window=1)
   ```

3. **Adjust context weight**:
   ```python
   # Lower weight = rely more on edit distance (faster)
   corrector = ContextAwareCorrector(context_weight=0.2)
   ```

4. **Use smaller vocabulary** (if possible):
   - Filter rare words from training corpus
   - Reduces cache misses

### For Maximum Accuracy

1. **Use trigrams**:
   ```python
   python train_ngram_model.py --ngram 3
   ```

2. **Larger context window**:
   ```python
   corrector = ContextAwareCorrector(context_window=2)
   ```

3. **Higher context weight**:
   ```python
   corrector = ContextAwareCorrector(context_weight=0.4)
   ```

4. **More training data**:
   - Collect 10+ MB of diverse Shan text
   - Include domain-specific content

## Model Evaluation

Check model quality:

```python
from shannlp.spell import NgramModel

# Load model
model = NgramModel.load("shan_bigram_model.msgpack")

# Check statistics
print(f"Vocabulary size: {len(model.vocabulary)}")
print(f"N-gram order: {model.n}")

# Calculate perplexity (lower is better)
test_texts = ["test sentence 1", "test sentence 2"]
perplexity = model.perplexity(test_texts)
print(f"Perplexity: {perplexity:.2f}")

# Check cache performance
stats = model.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
```

**Good Perplexity Values**:
- < 50: Excellent model
- 50-100: Good model
- 100-200: Acceptable model
- \> 200: Need more training data

## Troubleshooting

### Model training is slow

- Reduce corpus size for initial testing
- Use bigrams instead of trigrams
- Check text file encoding (must be UTF-8)

### High latency (>100ms)

- Use bigrams instead of trigrams
- Reduce `context_window` to 1
- Lower `context_weight` parameter
- Check cache hit rate (should be >50%)

### Poor correction quality

- Increase training corpus size (10+ MB)
- Use trigrams for better context
- Increase `context_weight` parameter
- Check model perplexity (should be <200)

### Model file is too large

- Filter low-frequency words from corpus
- Use bigrams (smaller than trigrams)
- Reduce smoothing parameter (e.g., 0.5)

## File Structure

```
ShanNLP/
├── shannlp/
│   └── spell/
│       ├── ngram.py          # N-gram model implementation
│       ├── context.py        # Context-aware corrector
│       ├── core.py           # Phase 1 spell correction
│       └── __init__.py       # Module exports
├── train_ngram_model.py      # Training script
├── example_context_aware_correction.py  # Examples
└── PHASE2_GUIDE.md          # This file
```

## Next Steps

1. **Train your model** with your Shan corpus
2. **Run examples**: `python example_context_aware_correction.py`
3. **Benchmark performance** on your target hardware
4. **Integrate** into your application
5. **Collect feedback** and retrain with more data

## Technical Details

### Scoring Formula

Context-aware score combines edit distance and context:

```
final_score = (1 - w) × edit_score + w × context_score

where:
- edit_score: From Phase 1 spell correction (0-1)
- context_score: Normalized n-gram probability (0-1)
- w: context_weight parameter (default: 0.3)
```

### N-gram Probability

Using Laplace smoothing:

```
P(word|context) = (count(context, word) + α) / (count(context) + α × |V|)

where:
- α: smoothing parameter (default: 1.0)
- |V|: vocabulary size
```

### Caching Strategy

- Caches up to 10,000 frequently accessed n-gram probabilities
- LRU eviction when cache full
- Typical cache hit rate: 70-90% for natural text

## Support

For questions or issues:
1. Check the examples in `example_context_aware_correction.py`
2. Review troubleshooting section above
3. Open an issue on GitHub

## License

Same as ShanNLP main project
