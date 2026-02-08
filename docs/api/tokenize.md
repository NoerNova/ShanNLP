# API Reference: shannlp.tokenize

```python
from shannlp import word_tokenize, syllable_tokenize, Tokenizer
# or
from shannlp.tokenize import word_tokenize, syllable_tokenize, Tokenizer
```

---

## `word_tokenize`

```python
word_tokenize(
    text: str,
    custom_dict: Trie = None,
    engine: str = "mm",
    keep_whitespace: bool = True,
    join_broken_num: bool = True,
) -> List[str]
```

Tokenize Shan text into a list of words.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Input text to tokenize |
| `custom_dict` | Trie | None | Custom word trie (default: `shan_all_corpus()` trie) |
| `engine` | str | `"mm"` | Tokenization engine (see below) |
| `keep_whitespace` | bool | True | Include whitespace tokens in output |
| `join_broken_num` | bool | True | Rejoin formatted numbers split by tokenizer (e.g. `1,000`) |

**Engines:**

| Value | Description |
|-------|-------------|
| `"mm"` | Maximal matching (default, general purpose) |
| `"newmm"` | PyThaiNLP's newmm algorithm (experimental) |
| `"whitespace"` | Split on spaces only |
| `"whitespace+newline"` | Split on spaces and newlines |

**Returns:** `List[str]` — list of tokens

**Raises:** `ValueError` if an unknown engine is specified

**Example:**

```python
from shannlp import word_tokenize

tokens = word_tokenize("တိူၵ်ႈသွၼ်လိၵ်ႈသင်ၶၸဝ်ႈ", keep_whitespace=False)
# ['တိူၵ်ႈ', 'သွၼ်လိၵ်ႈ', 'သင်ၶ', 'ၸဝ်ႈ']
```

---

## `syllable_tokenize`

```python
syllable_tokenize(text: str) -> list | None
```

Tokenize Shan text into syllables using linguistic rules.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Input text to tokenize |

**Returns:** `list` of syllable strings, or `None` if input is invalid

**Example:**

```python
from shannlp import syllable_tokenize

syllables = syllable_tokenize("မိူင်းတႆး")
# ['မိူင်း', 'တႆး']
```

---

## `class Tokenizer`

```python
class Tokenizer(
    custom_dict: Union[Trie, Iterable[str], str] = None,
    engine: str = "mm",
    keep_whitespace: bool = True,
    join_broken_num: bool = True,
)
```

A stateful tokenizer that caches the tokenizer instance for repeated use. Useful when tokenizing many texts with the same settings.

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `custom_dict` | Trie, iterable of str, or str | None | Custom dictionary. Accepts a Trie, a list/set of words, or a file path. Default: `shan_all_corpus()` |
| `engine` | str | `"mm"` | Engine to use (`"mm"` or `"newmm"` only) |
| `keep_whitespace` | bool | True | Include whitespace tokens |
| `join_broken_num` | bool | True | Rejoin formatted numbers |

**Raises:** `NotImplementedError` if `engine` is not `"mm"` or `"newmm"`

### Methods

#### `word_tokenize(text: str) -> List[str]`

Tokenize text using this instance's settings.

```python
tokenizer = Tokenizer(engine="mm", keep_whitespace=False)
tokens = tokenizer.word_tokenize("တိူၵ်ႈသွၼ်လိၵ်ႈ")
```

#### `set_tokenize_engine(engine: str) -> None`

Switch to a different engine without creating a new instance.

```python
tokenizer.set_tokenize_engine("newmm")
```

---

## Guide

See the [Tokenization guide](../guides/tokenization.md) for usage examples with custom dictionaries, engine comparisons, and syllable tokenization.
