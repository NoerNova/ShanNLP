# Getting Started with ShanNLP

## Prerequisites

- Python 3.10 or later
- All text should be encoded as UTF-8 (standard for Shan script)

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/NoerNova/ShanNLP
```

Or clone and install from source:

```bash
git clone https://github.com/NoerNova/ShanNLP
cd ShanNLP
pip install -r requirements.txt
```

## Verify Installation

```python
import shannlp
from shannlp import word_tokenize, spell_correct

print("ShanNLP installed successfully")
```

## Your First Tokenization

Shan text does not use spaces between words. `word_tokenize` splits continuous text into individual words:

```python
from shannlp import word_tokenize

text = "တိူၵ်ႈသွၼ်လိၵ်ႈသင်ၶၸဝ်ႈ တီႈဝဵင်းမိူင်းၶၢၵ်ႇ"

tokens = word_tokenize(text, keep_whitespace=False)
print(tokens)
# ['တိူၵ်ႈ', 'သွၼ်လိၵ်ႈ', 'သင်ၶ', 'ၸဝ်ႈ', 'တီႈ', 'ဝဵင်း', 'မိူင်းၶၢၵ်ႇ']
```

## Your First Spell Check

```python
from shannlp import spell_correct, is_correct_spelling

# Check if a word is spelled correctly
print(is_correct_spelling("မိူင်း"))   # True
print(is_correct_spelling("မိုင်း"))   # False

# Get correction suggestions
suggestions = spell_correct("မိုင်း")
print(suggestions)
# [('မိူင်း', 0.95), ('မိူင်', 0.82), ...]
```

## What's Next

- [Tokenization guide](guides/tokenization.md) — engines, custom dictionaries, syllables
- [Spell correction guide](guides/spell-correction.md) — sentences, context-aware, custom models
- [Corpus guide](guides/corpus.md) — word lists, stopwords, proper nouns
- [Utilities guide](guides/utilities.md) — numbers, dates, keyboard conversion
- [API Reference](README.md#api-reference) — complete function signatures
