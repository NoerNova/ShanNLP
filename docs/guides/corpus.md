# Corpus

## Overview

ShanNLP ships with a built-in corpus of Shan language data sourced from Shan Wikipedia and curated word lists. All corpus data is loaded lazily (on first access) and cached for subsequent calls.

The corpus includes approximately **19,904 Shan words** along with stopwords, syllables, character sets, country names, province names, and personal names.

## Word Lists

### `shan_words()`

Returns the main Shan word dictionary (~19,904 entries):

```python
from shannlp.corpus import shan_words

words = shan_words()
print(len(words))       # ~19904
print("မိူင်း" in words)  # True
```

### `shan_stopwords()`

Returns common Shan stopwords (function words, particles, etc.):

```python
from shannlp.corpus import shan_stopwords

stopwords = shan_stopwords()
print("ဢၼ်" in stopwords)  # True
```

### `shan_syllables()`

Returns the Shan syllable inventory:

```python
from shannlp.corpus import shan_syllables

syllables = shan_syllables()
```

### `shan_character()`

Returns individual Shan characters (consonants, vowels, tone marks, etc.):

```python
from shannlp.corpus import shan_character

characters = shan_character()
```

## Proper Nouns

### `countries()`

Returns country names in Shan:

```python
from shannlp.corpus import countries

nations = countries()
print("မိူင်းထႆး" in nations)  # True
```

### `provinces()`

Returns Shan State province names:

```python
from shannlp.corpus import provinces

provs = provinces()
```

### `shan_female_names()` and `shan_male_names()`

Returns Shan personal names by gender:

```python
from shannlp.corpus import shan_female_names, shan_male_names

female = shan_female_names()
male = shan_male_names()
```

## Combined Corpus

### `shan_all_corpus()`

Returns a union of all corpus data merged into a single `FrozenSet`. This is what the tokenizer uses by default:

```python
from shannlp.corpus import shan_all_corpus

all_data = shan_all_corpus()
print(len(all_data))  # combined count of all word lists
```

It combines: countries, provinces, words, stopwords, female names, male names, characters, and syllables.

## Low-Level Access

### `get_corpus(filename, as_is=False)`

Load any corpus file by filename:

```python
from shannlp.corpus import get_corpus

# Returns frozenset (deduplicated, stripped)
words = get_corpus("words_shn.txt")

# Returns list (original order preserved)
words_list = get_corpus("words_shn.txt", as_is=True)
```

### `get_m_corpus(filenames, as_is=False)`

Load and merge multiple corpus files:

```python
from shannlp.corpus import get_m_corpus

combined = get_m_corpus(["words_shn.txt", "stopwords_shn.txt"])
```

### `corpus_path()`

Get the filesystem path to the corpus directory:

```python
from shannlp.corpus import corpus_path

print(corpus_path())
# /path/to/shannlp/corpus
```

## Practical Example: Filtering Stopwords

```python
from shannlp import word_tokenize
from shannlp.corpus import shan_stopwords

text = "မိူင်းတႆးပဵၼ်မိူင်းၶိုၼ်ႉယႂ်"
tokens = word_tokenize(text, keep_whitespace=False)

stopwords = shan_stopwords()
content_words = [t for t in tokens if t not in stopwords]
print(content_words)
```

## Full API Reference

See [api/corpus.md](../api/corpus.md) for complete documentation.
