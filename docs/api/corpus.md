# API Reference: shannlp.corpus

```python
from shannlp.corpus import (
    shan_words, shan_stopwords, shan_syllables, shan_character,
    shan_all_corpus, countries, provinces,
    shan_female_names, shan_male_names,
    get_corpus, get_m_corpus, corpus_path, path_shannlp_corpus,
)
```

All corpus data is loaded lazily on first access and cached globally.

---

## Word and Character Data

### `shan_words() -> FrozenSet[str]`

Returns the main Shan word dictionary (~19,904 entries) from Shan Wikipedia and curated sources.

```python
from shannlp.corpus import shan_words
words = shan_words()
print("မိူင်း" in words)  # True
```

---

### `shan_stopwords() -> FrozenSet[str]`

Returns common Shan stopwords (particles, function words).

```python
from shannlp.corpus import shan_stopwords
stopwords = shan_stopwords()
```

---

### `shan_syllables() -> FrozenSet[str]`

Returns the Shan syllable inventory.

```python
from shannlp.corpus import shan_syllables
syllables = shan_syllables()
```

---

### `shan_character() -> FrozenSet[str]`

Returns individual Shan characters (consonants, vowels, tone marks, digits, punctuation).

```python
from shannlp.corpus import shan_character
chars = shan_character()
```

---

## Proper Noun Corpora

### `countries() -> FrozenSet[str]`

Returns country names in Shan.

```python
from shannlp.corpus import countries
nations = countries()
```

---

### `provinces(details: bool = False) -> FrozenSet[str]`

Returns Shan State province names.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `details` | bool | False | (reserved, currently unused) |

```python
from shannlp.corpus import provinces
provs = provinces()
```

---

### `shan_female_names() -> FrozenSet[str]`

Returns Shan female personal names.

---

### `shan_male_names() -> FrozenSet[str]`

Returns Shan male personal names.

---

## Combined Corpus

### `shan_all_corpus() -> FrozenSet[str]`

Returns a union of all corpus data. This is the default dictionary used by the tokenizer.

Combines: countries, provinces, words, stopwords, female names, male names, characters, syllables.

```python
from shannlp.corpus import shan_all_corpus
all_data = shan_all_corpus()
```

---

## Low-Level Access

### `get_corpus`

```python
get_corpus(filename: str, as_is: bool = False) -> Union[frozenset, list]
```

Load a corpus file by filename.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | str | required | Corpus filename (e.g. `"words_shn.txt"`) |
| `as_is` | bool | False | If True, return list preserving order; if False, return frozenset (stripped, deduplicated) |

**Returns:** `frozenset` or `list`

```python
from shannlp.corpus import get_corpus

words = get_corpus("words_shn.txt")           # frozenset
words_list = get_corpus("words_shn.txt", as_is=True)  # list
```

---

### `get_m_corpus`

```python
get_m_corpus(filenames: List[str], as_is: bool = False) -> Union[frozenset, List[str]]
```

Load and merge multiple corpus files.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filenames` | list of str | required | List of corpus filenames |
| `as_is` | bool | False | Same as in `get_corpus` |

**Returns:** `frozenset` or `list`

```python
from shannlp.corpus import get_m_corpus

combined = get_m_corpus(["words_shn.txt", "stopwords_shn.txt"])
```

---

### `corpus_path() -> str`

Returns the absolute filesystem path to the corpus directory.

```python
from shannlp.corpus import corpus_path
print(corpus_path())
# /path/to/shannlp/corpus
```

---

### `path_shannlp_corpus(filename: str) -> str`

Returns the full path to a specific corpus file.

```python
from shannlp.corpus import path_shannlp_corpus
print(path_shannlp_corpus("words_shn.txt"))
```

---

## Available Corpus Files

| Filename | Contents |
|----------|----------|
| `words_shn.txt` | ~19,904 Shan words |
| `stopwords_shn.txt` | Common stopwords |
| `shan_syllables.txt` | Syllable units |
| `shan_character.txt` | Individual characters |
| `countries_shn.txt` | Country names |
| `shan_state_provinces.txt` | Province names |
| `person_names_female_shn.txt` | Female names |
| `person_names_male_shn.txt` | Male names |
| `shnwiki_freq.txt` | Word frequency data (Wikipedia) |

---

## Guide

See the [Corpus guide](../guides/corpus.md) for practical usage examples.
