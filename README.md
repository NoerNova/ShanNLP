# ShanNLP: Shan Natural Language Processing

**experimental project and self-research inspired by [PythaiNLP](https://github.com/PyThaiNLP/pythainlp)**

## Current State

- [ ] corpus dict word: 19904 words (60% corvered and need more to collected)

## Word Tokenization method

- [x] maximal_matching
- [x] pythainlp (newmm)

## TODO

- [ ] mining more shan words, poem
- [ ] experiment more method to tokenize
  - [ ] word tokenize
  - [ ] sentent tokenize
  - [ ] subword_tokenize
  - [ ] tokenize with deep learning
- [x] spelling check (Phase 1 & 2 complete)
- [ ] pos tagging
- [ ] translation
- [ ] word_vector

## USAGE

### Install

Clone this Repo

```python
# this project using pythainlp dependecy
# - Trie data structure
# - newmm (experimental)

pip install -r requirements.txt
# or pip install pythainlp
```

Install with pip
```bash
pip install git+https://github.com/NoerNova/ShanNLP

```

### Tokenization

#### maximal_matching bruce-force

```python
from shannlp import word_tokenize

# start measure execute time
# start = time.time()

# # Example usage
input_text = "တိူၵ်ႈသွၼ်လိၵ်ႈသင်ၶၸဝ်ႈ တီႈဝဵင်းမိူင်းၶၢၵ်ႇ တႄႇပိုတ်ႇသွၼ်ႁဵၼ်းလိၵ်ႈ ပဵၼ်ပွၵ်ႈၵမ်းႁႅၵ်း မီးသင်ၶၸဝ်ႈ မႃးႁဵၼ်း 56 တူၼ်။"

# default tokenizer engine="mm" (maximal_matching)
print(word_tokenize(input_text))

# end measure execute time
# end = time.time()
# print(end - start)

# output
# ['တိူၵ်ႈ', 'သွၼ်လိၵ်ႈ', 'သင်ၶ', 'ၸဝ်ႈ', ' ', 'တီႈ', 'ဝဵင်း', 'မိူင်းၶၢၵ်ႇ', ' ', 'တႄႇ', 'ပိုတ်ႇ', 'သွၼ်', 'ႁဵၼ်းလိၵ်ႈ', ' ', 'ပဵၼ်', 'ပွၵ်ႈ', 'ၵမ်း', 'ႁႅၵ်း', ' ', 'မီး', 'သင်ၶ', 'ၸဝ်ႈ', ' ', 'မႃး', 'ႁဵၼ်း', ' ', '56', ' ', 'တူၼ်', '။']
# 0.7220799922943115
```

#### pythainlp newmm

```python
from shannlp import word_tokenize
import time

# start measure execute time
start = time.time()

# Example usage
input_text = "တိူၵ်ႈသွၼ်လိၵ်ႈသင်ၶၸဝ်ႈ တီႈဝဵင်းမိူင်းၶၢၵ်ႇ တႄႇပိုတ်ႇသွၼ်ႁဵၼ်းလိၵ်ႈ ပဵၼ်ပွၵ်ႈၵမ်းႁႅၵ်း မီးသင်ၶၸဝ်ႈ မႃးႁဵၼ်း 56 တူၼ်။"

print(word_tokenize(input_text, engine="newmm", keep_whitespace=False))

# end measure execute time
end = time.time()
print(end - start)

# output
# ['တိူၵ်ႈ', 'သွၼ်လိၵ်ႈ', 'သင်ၶ', 'ၸဝ်ႈ', 'တီႈ', 'ဝဵင်း', 'မိူင်းၶၢၵ်ႇ', 'တႄႇ', 'ပိုတ်ႇ', 'သွၼ်', 'ႁဵၼ်းလိၵ်ႈ', 'ပဵၼ်', 'ပွၵ်ႈ', 'ၵမ်း', 'ႁႅၵ်း', 'မီး', 'သင်ၶ', 'ၸဝ်ႈ', 'မႃး', 'ႁဵၼ်း', '56', 'တူၼ်', '။']
# 0.7088069915771484
```

### Digit convert

```python
from shannlp.util import digit_to_text

print(digit_to_text("မႂ်ႇသုင်ပီမႂ်ႇတႆး ႒႑႑႗ ၼီႈ"))

# output
# မႂ်ႇသုင်ပီမႂ်ႇတႆး သွင်ၼိုင်ႈၼိုင်ႈၸဵတ်း ၼီႈ
```

#### num_to_word

```python
from shannlp.util import num_to_shanword

print(num_to_shanword(2117))
# output သွင်ႁဵင်ၼိုင်ႈပၢၵ်ႇသိပ်းၸဵတ်း
```

#### shanword_to_num

```python
from shannlp.util import shanword_to_num

print(shanword_to_num("ထွၼ်ႁဵင်ၵဝ်ႈပၢၵ်ႇၵဝ်ႈသိပ်းဢဵတ်း"))
# output -1991
```

#### text_to_num

```python
from shannlp.util import text_to_num

print(text_to_num("သွင်ႁဵင်ၼိုင်ႈပၢၵ်ႇသိပ်းၸဵတ်းပီပူၼ်ႉမႃး"))
# output ['2117', 'ပီ', 'ပူၼ်ႉ', 'မႃး']
```

### Date converter

#### ***need more reference for years converter***

```md
current reference
# https://shn.wikipedia.org/wiki/ဝၼ်းၸဵတ်းဝၼ်း_ၽၢႆႇတႆး

# MO: ပီတႆး 2117
# GA: ပီၵေႃးၸႃႇ 1385
# BE: ပီပုတ်ႉထ 2566
# AD: ပီဢိင်းၵရဵတ်ႈ 2023
````

```python
from shannlp.util import shanword_to_date
import datetime

print(f"မိူဝ်ႈၼႆႉ: {datetime.date.today()}")
print(f"မိူဝ်ႈဝၼ်းသိုၼ်း {shanword_to_date('မိူဝ်ႈဝၼ်းသိုၼ်း')}")

# output
# မိူဝ်ႈၼႆႉ: 2023-06-15
# မိူဝ်ႈဝၼ်းသိုၼ်း 2023-06-13 00:51:14.597118
```

#### years convert

```python
from shannlp.util import convert_years

# ပီ AD -> ပီတႆး
print(convert_years(2023, "ad", "mo"))
# output 2117

# ပီတႆး -> ပီပုတ်ႉထ
print(convert_years(2117, "mo", "be"))
# output 2566

# ပီပုတ်ႉထ -> ပီၵေႃးၸႃႇ
print(convert_years(2566, "be", "ga"))
# output 1385
```

### Keyboard

```python
from shannlp.util import eng_to_shn, shn_to_eng

print(eng_to_shn("rgfbokifcMj"))
# output မႂ်ႇသုင်ၶႃႈ

print(shn_to_eng("ေၺၺူၼ"))
# output apple
```

---

## Spell Correction

ShanNLP provides spell correction for Shan language using Peter Norvig's algorithm enhanced with Shan-specific features:

- Weighted edit distance based on character types (consonants, vowels, tone marks)
- Phonetic similarity for consonants
- Frequency-based ranking using Wikipedia data
- Context-aware correction using n-gram language models

### Quick Start

```python
from shannlp import correct_sentence, spell_correct

# Sentence correction (recommended)
result = correct_sentence("ၵူၼ်မိူင်း ၵိၼ် ၶဝ်ႈ")
print(result)

# Single word correction
suggestions = spell_correct("ၸိူင်")
print(suggestions)
# [('ၸိူဝ်း', 0.72), ('ၸိုင်', 0.69), ...]
```

### API Reference

#### `correct_sentence(sentence, model_path=None, min_confidence=0.3, use_context=True)`

Correct spelling errors in a Shan sentence using context-aware correction.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sentence` | str | required | Input sentence to correct |
| `model_path` | str | None | Custom n-gram model path (None = use bundled model) |
| `min_confidence` | float | 0.3 | Minimum confidence threshold (0.0-1.0) |
| `use_context` | bool | True | Use n-gram context for better accuracy |

**Returns:** Corrected sentence as string

```python
from shannlp import correct_sentence

# Basic usage (auto-loads bundled model)
result = correct_sentence("ၵူၼ်မိူင်း ၵိၼ် ၶဝ်ႈ")
print(result)

# With custom model
result = correct_sentence("text here", model_path="my_model.msgpack")

# Without context (faster, less accurate)
result = correct_sentence("text here", use_context=False)
```

#### `correct_text(text, model_path=None, min_confidence=0.3)`

Correct spelling errors and return as token list.

**Returns:** List of corrected tokens

```python
from shannlp import correct_text

tokens = correct_text("ၵူၼ်မိူင်း ၵိၼ် ၶဝ်ႈ")
print(tokens)
# ['ၵူၼ်းမိူင်း', 'ၵိၼ်', 'ၶဝ်ႈ']
```

#### `spell_correct(word, custom_dict=None, max_edit_distance=2, max_suggestions=5, use_frequency=True, use_phonetic=True, min_confidence=0.1)`

Correct a single word and return ranked suggestions.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `word` | str | required | Word to correct |
| `custom_dict` | set | None | Custom dictionary (None = use shan_words()) |
| `max_edit_distance` | int | 2 | Maximum edit distance (1 or 2) |
| `max_suggestions` | int | 5 | Maximum suggestions to return |
| `use_frequency` | bool | True | Use frequency data for ranking |
| `use_phonetic` | bool | True | Use phonetic similarity |
| `min_confidence` | float | 0.1 | Minimum confidence threshold |

**Returns:** List of `(suggestion, confidence)` tuples, sorted by confidence descending

```python
from shannlp import spell_correct

# Basic usage
suggestions = spell_correct("မိုင်း")
print(suggestions)
# [('မိူင်း', 0.95), ('မိူင်', 0.82), ...]

# With custom dictionary
custom = {"ၵႃႈ", "မူၼ်း", "ယူႇ"}
suggestions = spell_correct("ၵႃႈ", custom_dict=custom)
```

#### `is_correct_spelling(word, custom_dict=None)`

Check if a word is spelled correctly.

**Returns:** `True` if word is in dictionary, `False` otherwise

```python
from shannlp import is_correct_spelling

print(is_correct_spelling("မိူင်း"))  # True
print(is_correct_spelling("မိုင်း"))  # False
```

#### `reload_model(model_path=None)`

Reload the n-gram model (useful after training a new model).

```python
from shannlp.spell import reload_model

# Reload with new model
reload_model("new_model.msgpack")
```

### Advanced Usage

#### Using SpellCorrector Class

```python
from shannlp import SpellCorrector

# Create corrector with custom settings
corrector = SpellCorrector(
    max_edit_distance=2,
    use_phonetic=True
)

# Check and correct
print(corrector.is_correct("မိူင်း"))  # True
suggestions = corrector.correct("မိုင်း")
print(suggestions)

# Add custom words
corrector.add_word("ၵႃႈ")
corrector.add_words(["word1", "word2"])
```

#### Using ContextAwareCorrector Class

```python
from shannlp.spell import ContextAwareCorrector

# Create with custom settings
corrector = ContextAwareCorrector(
    context_window=2,
    context_weight=0.3
)

# Load custom model
corrector.load_model("my_trigram_model.msgpack")

# Correct text
result = corrector.correct_sentence("your text here")
print(result)

# Get performance stats
stats = corrector.get_performance_stats()
print(f"Average time: {stats['avg_time_ms']:.2f}ms")
```

### Training Custom N-gram Model

Train your own n-gram model for better context-aware correction:

```bash
# Standard training
python train_ngram_model.py \
    --corpus_dir ./data/corpus \
    --output shan_bigram.msgpack \
    --ngram 2 \
    --min-count 2

# Fast training with pre-tokenized corpus
python tokenize_corpus.py --input ./data/corpus --output ./data/tokenized
python train_ngram_model.py \
    --corpus_dir ./data/tokenized \
    --output shan_bigram.msgpack \
    --tokenized
```

**Training Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--corpus_dir` | required | Directory with .txt files |
| `--output` | shan_ngram_model.msgpack | Output model path |
| `--ngram` | 2 | N-gram order (2=bigram, 3=trigram) |
| `--smoothing` | 1.0 | Laplace smoothing parameter |
| `--min-count` | 2 | Minimum word count to keep |
| `--tokenized` | False | Corpus is pre-tokenized |
| `--test_split` | 0.1 | Fraction for test set |

### Error Types Handled

The spell corrector handles common Shan typing errors:

| Error Type | Example | Description |
|------------|---------|-------------|
| Tone mark errors | ႇ ↔ ႈ ↔ း | Confused tone marks |
| Vowel position | Lead/follow/above/below vowels | Wrong vowel placement |
| Phonetic similarity | ပ ↔ ၽ, က ↔ ၵ | Similar sounding consonants |
| Keyboard typos | Insert, delete, transpose | Standard typing errors |
| Lead vowel transposition | ေၶ → ၶေ | Vowel typed before consonant |

### Performance

| Operation | Target | Description |
|-----------|--------|-------------|
| Single word | < 100ms | Basic spell correction |
| Sentence | < 500ms | Context-aware correction |
| Model loading | ~1s | One-time on first use |

## Citations

```txt
Wannaphong Phatthiyaphaibun, Korakot Chaovavanich, Charin Polpanumas, Arthit Suriyawongkul, Lalita Lowphansirikul, & Pattarawat Chormai. (2016, Jun 27). PyThaiNLP: Thai Natural Language Processing in Python. Zenodo. http://doi.org/10.5281/zenodo.3519354
```

BibText entry:

```txt
@misc{pythainlp,
    author       = {Wannaphong Phatthiyaphaibun and Korakot Chaovavanich and Charin Polpanumas and Arthit Suriyawongkul and Lalita Lowphansirikul and Pattarawat Chormai},
    title        = {{PyThaiNLP: Thai Natural Language Processing in Python}},
    month        = Jun,
    year         = 2016,
    doi          = {10.5281/zenodo.3519354},
    publisher    = {Zenodo},
    url          = {http://doi.org/10.5281/zenodo.3519354}
}
```
