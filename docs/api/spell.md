# API Reference: shannlp.spell

```python
from shannlp import spell_correct, correct_sentence, correct_text, is_correct_spelling, SpellCorrector
# or
from shannlp.spell import (
    spell_correct, correct_sentence, correct_text, is_correct_spelling,
    reload_model, load_neural_model,
    SpellCorrector, ContextAwareCorrector, NgramModel,
)
```

The n-gram model is loaded lazily on first use of `correct_sentence` or `correct_text`.

---

## Simple API (recommended)

### `spell_correct`

```python
spell_correct(
    word: str,
    custom_dict: Optional[Set[str]] = None,
    max_edit_distance: int = 2,
    max_suggestions: int = 5,
    use_frequency: bool = True,
    use_phonetic: bool = True,
    min_confidence: float = 0.1,
    always_suggest_alternatives: bool = False,
) -> List[Tuple[str, float]]
```

Correct a single Shan word using Peter Norvig's algorithm with Shan-specific enhancements.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `word` | str | required | Word to correct |
| `custom_dict` | set or frozenset | None | Custom dictionary (default: `shan_words()`) |
| `max_edit_distance` | int | 2 | Maximum edit distance to search (1 or 2) |
| `max_suggestions` | int | 5 | Maximum number of suggestions |
| `use_frequency` | bool | True | Weight by Wikipedia word frequency |
| `use_phonetic` | bool | True | Include phonetically similar candidates |
| `min_confidence` | float | 0.1 | Minimum confidence score (0.0–1.0) |
| `always_suggest_alternatives` | bool | False | Return alternatives even for correctly spelled words |

**Returns:** `List[Tuple[str, float]]` — list of `(suggestion, confidence)` tuples, sorted by confidence descending. Confidence is in the range 0.0–1.0.

**Raises:**
- `ValueError` — invalid or empty input
- `SecurityError` — input contains null bytes or control characters

**Scoring formula:** `score = 0.6 × distance_score + 0.4 × frequency_score`
where `distance_score = 1 / (1 + edit_distance)` and frequency_score is the normalized log-probability from Wikipedia.

**Example:**

```python
from shannlp import spell_correct

results = spell_correct("မိုင်း")
print(results)
# [('မိူင်း', 0.95), ('မိူင်', 0.82), ...]

# Top suggestion only
best = spell_correct("မိုင်း", max_suggestions=1)[0][0]
```

---

### `correct_sentence`

```python
correct_sentence(
    sentence: str,
    model_path: Optional[str] = None,
    min_confidence: float = 0.3,
    use_context: bool = True,
    separator: str = " ",
) -> str
```

Correct spelling errors in a sentence. This is the recommended function for sentence-level correction.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sentence` | str | required | Input sentence |
| `model_path` | str | None | Path to n-gram model file (None = bundled default) |
| `min_confidence` | float | 0.3 | Minimum confidence for corrections |
| `use_context` | bool | True | Use n-gram context for better accuracy |
| `separator` | str | `" "` | String used to join corrected tokens. Use `""` for traditional no-space Shan text |

**Returns:** `str` — corrected sentence

**Notes:**
- Model loads on first call (~1s). Subsequent calls use cached model.
- If the bundled n-gram model is not found, falls back to basic word-level correction with a warning.
- If `load_neural_model()` has been called, neural reranking is applied automatically.

**Example:**

```python
from shannlp import correct_sentence

result = correct_sentence("ၵူၼ်မိူင်း ၵိၼ် ၶဝ်ႈ")

# No-space output
result = correct_sentence("ၵူၼ်မိူင်းၵိၼ်ၶဝ်ႈ", separator="")
```

---

### `correct_text`

```python
correct_text(
    text: str,
    model_path: Optional[str] = None,
    min_confidence: float = 0.3,
) -> List[str]
```

Like `correct_sentence` but returns a list of corrected tokens.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Input text |
| `model_path` | str | None | Path to n-gram model file |
| `min_confidence` | float | 0.3 | Minimum confidence threshold |

**Returns:** `List[str]` — corrected tokens

**Example:**

```python
from shannlp import correct_text

tokens = correct_text("ၵူၼ်မိူင်း ၵိၼ် ၶဝ်ႈ")
# ['ၵူၼ်းမိူင်း', 'ၵိၼ်', 'ၶဝ်ႈ']
```

---

### `is_correct_spelling`

```python
is_correct_spelling(
    word: str,
    custom_dict: Optional[Set[str]] = None,
) -> bool
```

Check if a word exists in the dictionary.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `word` | str | required | Word to check |
| `custom_dict` | set | None | Custom dictionary (default: `shan_words()`) |

**Returns:** `bool` — `True` if word is in dictionary

**Example:**

```python
from shannlp import is_correct_spelling

is_correct_spelling("မိူင်း")   # True
is_correct_spelling("မိုင်း")   # False
```

---

### `reload_model`

```python
reload_model(model_path: Optional[str] = None) -> None
```

Reload the n-gram model. Use after training a new model.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | None | Path to model file (None = reload default) |

```python
from shannlp.spell import reload_model
reload_model("new_model.msgpack")
```

---

### `load_neural_model`

```python
load_neural_model(model_path: str, device: Optional[str] = None) -> None
```

Load a neural reranker model for improved accuracy. Requires PyTorch.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | Path to `.pt` model file or cached model name |
| `device` | str | None | Device: `"cuda"`, `"mps"`, `"cpu"`. Auto-detected if None |

**Raises:** `ImportError` if PyTorch is not installed

```python
from shannlp.spell import load_neural_model, correct_sentence

load_neural_model("spell_reranker.pt")
result = correct_sentence("မိူင်တႆးပဵၼ်မိူင်းၶိုၼ်ႉယႂ်")
```

---

## Advanced: `class SpellCorrector`

```python
class SpellCorrector(
    custom_dict: Optional[Union[Set[str], frozenset]] = None,
    frequency_data: Optional[dict] = None,
    max_edit_distance: int = 2,
    use_phonetic: bool = True,
)
```

Stateful spell corrector. Useful when processing many words with a custom dictionary.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `custom_dict` | set, frozenset | None | Initial dictionary (default: `shan_words()`) |
| `frequency_data` | dict | None | Custom frequency data (default: Wikipedia) |
| `max_edit_distance` | int | 2 | Default maximum edit distance |
| `use_phonetic` | bool | True | Enable phonetic similarity |

### Methods

#### `correct(word, max_suggestions=5, min_confidence=0.1) -> List[Tuple[str, float]]`

Correct a single word. Same return format as `spell_correct`.

#### `is_correct(word) -> bool`

Check if word is in the dictionary.

#### `add_word(word, frequency=1) -> None`

Add a word to the dictionary with optional frequency count.

#### `add_words(words) -> None`

Add multiple words (accepts any iterable of strings).

#### `remove_word(word) -> None`

Remove a word from the dictionary.

**Example:**

```python
from shannlp import SpellCorrector

corrector = SpellCorrector(max_edit_distance=2)
corrector.add_word("ၵႃႈ")
corrector.add_words(["တႆးၸိုင်ႈမိူင်", "ၸွမ်ၸိုင်ႈ"])

print(corrector.is_correct("ၵႃႈ"))    # True
print(corrector.correct("မိုင်း"))
```

---

## Advanced: `class ContextAwareCorrector`

```python
class ContextAwareCorrector(
    ngram_model: Optional[NgramModel] = None,
    spell_corrector: Optional[SpellCorrector] = None,
    context_window: int = 2,
    context_weight: float = 0.3,
    always_suggest_alternatives: bool = False,
    neural_model: Optional[SpellReranker] = None,
    use_neural: bool = False,
)
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ngram_model` | NgramModel | None | Pre-trained n-gram model |
| `spell_corrector` | SpellCorrector | None | Custom spell corrector |
| `context_window` | int | 2 | Words of context (max 2 for speed) |
| `context_weight` | float | 0.3 | Context weight in scoring (0–1) |
| `always_suggest_alternatives` | bool | False | Force alternatives for all words |
| `neural_model` | SpellReranker | None | Optional neural reranker |
| `use_neural` | bool | False | Enable neural reranking |

### Class Methods

#### `classmethod load(ngram_path, ngram_url=None, neural_path=None, neural_url=None, **kwargs) -> ContextAwareCorrector`

Create an instance with models pre-loaded. Accepts model name or path; downloads from URL if not cached.

```python
corrector = ContextAwareCorrector.load("shan_bigram.msgpack")
```

### Instance Methods

#### `load_ngram_model(model_path, url=None) -> None`

Load or download an n-gram model.

#### `load_neural_model(model_path, device=None, model_url=None, vocab_url=None) -> None`

Load or download a neural reranker model.

#### `correct_sentence(sentence) -> str`

Correct a sentence using context-aware correction.

#### `correct_text(text, min_confidence=0.3) -> List[str]`

Correct text and return tokens.

#### `batch_correct(texts, show_progress=False) -> List[List[str]]`

Process multiple texts efficiently.

#### `get_performance_stats() -> dict`

Returns: `{"avg_time_ms": float, "max_time_ms": float, "total_corrections": int, ...}`

---

## Advanced: `class NgramModel`

```python
class NgramModel(n: int = 2, smoothing: float = 1.0)
```

Bigram or trigram language model used internally by `ContextAwareCorrector`.

### Class Methods

#### `classmethod load(path) -> NgramModel`

Load a saved model from a `.msgpack` file.

```python
model = NgramModel.load("shan_bigram.msgpack")
```

### Instance Methods

#### `train(sentences: List[str]) -> None`

Train from a list of tokenized sentences.

#### `probability(word, context=None) -> float`

Return P(word | context) with Laplace smoothing.

#### `log_probability(word, context=None) -> float`

Return log P(word | context).

#### `save(path) -> None`

Save model to a `.msgpack` file.

#### `get_cache_stats() -> dict`

Return cache hit/miss statistics.

---

## Guide

See the [Spell Correction guide](../guides/spell-correction.md) for detailed examples and training instructions.
