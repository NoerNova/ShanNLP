# Tokenization

## Overview

Shan text does not use spaces between words (similar to Chinese and Thai). Tokenization splits continuous text into individual words or syllables so that downstream NLP tasks can process them.

ShanNLP provides two tokenization granularities:
- **Word tokenization** — splits text into words using a dictionary
- **Syllable tokenization** — splits text into syllables using linguistic rules

## Word Tokenization

### Basic Usage

```python
from shannlp import word_tokenize

text = "တိူၵ်ႈသွၼ်လိၵ်ႈသင်ၶၸဝ်ႈ တီႈဝဵင်းမိူင်းၶၢၵ်ႇ"

# Default engine: maximal matching (mm)
tokens = word_tokenize(text)
print(tokens)
# ['တိူၵ်ႈ', 'သွၼ်လိၵ်ႈ', 'သင်ၶ', 'ၸဝ်ႈ', ' ', 'တီႈ', 'ဝဵင်း', 'မိူင်းၶၢၵ်ႇ']

# Exclude whitespace tokens
tokens = word_tokenize(text, keep_whitespace=False)
print(tokens)
# ['တိူၵ်ႈ', 'သွၼ်လိၵ်ႈ', 'သင်ၶ', 'ၸဝ်ႈ', 'တီႈ', 'ဝဵင်း', 'မိူင်းၶၢၵ်ႇ']
```

### Choosing an Engine

ShanNLP supports four engines via the `engine` parameter:

| Engine | Description | Use Case |
|--------|-------------|----------|
| `"mm"` | Maximal matching (default) | General purpose, fast |
| `"newmm"` | PyThaiNLP's newmm algorithm | Experimental, may give different segmentations |
| `"whitespace"` | Split on spaces only | Pre-tokenized or simple text |
| `"whitespace+newline"` | Split on spaces and newlines | Multi-line pre-tokenized text |

```python
from shannlp import word_tokenize

text = "တိူၵ်ႈသွၼ်လိၵ်ႈသင်ၶၸဝ်ႈ တီႈဝဵင်းမိူင်းၶၢၵ်ႇ"

# newmm engine
tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
print(tokens)
# ['တိူၵ်ႈ', 'သွၼ်လိၵ်ႈ', 'သင်ၶ', 'ၸဝ်ႈ', 'တီႈ', 'ဝဵင်း', 'မိူင်းၶၢၵ်ႇ']
```

### Number Handling

By default, formatted numbers like `1,000` that get split by the tokenizer are rejoined. Disable this with `join_broken_num=False`:

```python
tokens = word_tokenize("ၵႃႈၶၼ် 1,000 ပျႃး", join_broken_num=False)
# ['ၵႃႈၶၼ်', ' ', '1', ',', '000', ' ', 'ပျႃး']

tokens = word_tokenize("ၵႃႈၶၼ် 1,000 ပျႃး", join_broken_num=True)
# ['ၵႃႈၶၼ်', ' ', '1,000', ' ', 'ပျႃး']
```

## Syllable Tokenization

`syllable_tokenize` splits text into syllable units using linguistic rules rather than a dictionary:

```python
from shannlp import syllable_tokenize

text = "မိူင်းတႆး"
syllables = syllable_tokenize(text)
print(syllables)
# ['မိူင်း', 'တႆး']
```

Syllable tokenization is useful when:
- The word is unknown to the dictionary
- You need phonological analysis
- You are building character-level models

## Using the Tokenizer Class

The `Tokenizer` class is useful when you need to tokenize many texts with the same settings. It caches the underlying tokenizer for better performance:

```python
from shannlp import Tokenizer

# Create a reusable tokenizer
tokenizer = Tokenizer(engine="mm", keep_whitespace=False)

# Tokenize multiple texts
texts = [
    "မိူင်းတႆးပဵၼ်မိူင်းၶိုၼ်ႉယႂ်",
    "တိူၵ်ႈသွၼ်လိၵ်ႈသင်ၶၸဝ်ႈ"
]

for text in texts:
    print(tokenizer.word_tokenize(text))
```

Switch engines at runtime:

```python
tokenizer.set_tokenize_engine("newmm")
```

> **Note:** The `Tokenizer` class only supports `"mm"` and `"newmm"` engines. Using other engines raises `NotImplementedError`.

## Custom Dictionary

Pass a custom word set to override or extend the default corpus:

```python
from shannlp import word_tokenize
from pythainlp.util.trie import dict_trie

# Build a custom trie from a word list
custom_words = {"မိူင်းတႆး", "ၸဝ်ႈႁႆႈၸဝ်ႈၼႃး", "ၵၢၼ်မိူင်း"}
custom_trie = dict_trie(custom_words)

tokens = word_tokenize("ၵၢၼ်မိူင်းမိူင်းတႆး", custom_dict=custom_trie, keep_whitespace=False)
print(tokens)
```

You can also pass a `Tokenizer` class a set of strings or a file path and it will build the trie automatically:

```python
from shannlp import Tokenizer

# From a set of strings
tokenizer = Tokenizer(custom_dict={"မိူင်းတႆး", "ၵၢၼ်မိူင်း"}, engine="mm")

# From an iterable
tokenizer = Tokenizer(custom_dict=open("mywords.txt").read().splitlines(), engine="mm")
```

## Full API Reference

See [api/tokenize.md](../api/tokenize.md) for complete parameter documentation.
