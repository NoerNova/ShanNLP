# ShanNLP

Natural Language Processing library for the Shan language (တႆး), inspired by [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp).

## Installation

```bash
pip install git+https://github.com/NoerNova/ShanNLP
```

Or from source:

```bash
git clone https://github.com/NoerNova/ShanNLP
pip install -r requirements.txt
```

## Quick Start

```python
from shannlp import word_tokenize

text = "တိူၵ်ႈသွၼ်လိၵ်ႈသင်ၶၸဝ်ႈ တီႈဝဵင်းမိူင်းၶၢၵ်ႇ"
print(word_tokenize(text, keep_whitespace=False))
# ['တိူၵ်ႈ', 'သွၼ်လိၵ်ႈ', 'သင်ၶ', 'ၸဝ်ႈ', 'တီႈ', 'ဝဵင်း', 'မိူင်းၶၢၵ်ႇ']
```

```python
from shannlp import spell_correct, correct_sentence

# Single word
print(spell_correct("မိုင်း"))
# [('မိူင်း', 0.95), ('မိူင်', 0.82), ...]

# Full sentence
print(correct_sentence("ၵူၼ်မိူင်း ၵိၼ် ၶဝ်ႈ"))
```

```python
from shannlp.util import num_to_shanword, convert_years

print(num_to_shanword(2117))  # သွင်ႁဵင်ၼိုင်ႈပၢၵ်ႇသိပ်းၸဵတ်း
print(convert_years(2023, "ad", "mo"))  # 2117
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [User Guides](docs/README.md#guides)
  - [Tokenization](docs/guides/tokenization.md)
  - [Spell Correction](docs/guides/spell-correction.md)
  - [Corpus](docs/guides/corpus.md)
  - [Utilities](docs/guides/utilities.md)
- [API Reference](docs/README.md#api-reference)

## What's Included

| Module | Description |
|--------|-------------|
| `shannlp.tokenize` | Word and syllable tokenization |
| `shannlp.spell` | Spell correction (word-level and context-aware) |
| `shannlp.corpus` | Shan language corpus (~19,904 words) |
| `shannlp.util` | Number, digit, date, and keyboard utilities |

## Citations

```txt
Wannaphong Phatthiyaphaibun, Korakot Chaovavanich, Charin Polpanumas, Arthit Suriyawongkul,
Lalita Lowphansirikul, & Pattarawat Chormai. (2016, Jun 27). PyThaiNLP: Thai Natural Language
Processing in Python. Zenodo. http://doi.org/10.5281/zenodo.3519354
```

```bibtex
@misc{pythainlp,
    author       = {Wannaphong Phatthiyaphaibun and Korakot Chaovavanich and Charin Polpanumas
                    and Arthit Suriyawongkul and Lalita Lowphansirikul and Pattarawat Chormai},
    title        = {{PyThaiNLP: Thai Natural Language Processing in Python}},
    month        = Jun,
    year         = 2016,
    doi          = {10.5281/zenodo.3519354},
    publisher    = {Zenodo},
    url          = {http://doi.org/10.5281/zenodo.3519354}
}
```
