# ShanNLP Documentation

ShanNLP is a Natural Language Processing library for the Shan language (တႆး). It provides tokenization, spell correction, corpus access, and language utilities built specifically for Shan text.

## Getting Started

- [Installation and first steps](getting-started.md)

## Guides

Task-oriented guides with practical examples:

| Guide | Description |
|-------|-------------|
| [Tokenization](guides/tokenization.md) | Split Shan text into words or syllables |
| [Spell Correction](guides/spell-correction.md) | Detect and correct spelling errors |
| [Corpus](guides/corpus.md) | Access built-in word lists and language data |
| [Utilities](guides/utilities.md) | Numbers, digits, dates, and keyboard conversion |
| [Model Download](guides/model-download.md) | Download and manage NLP models |

## API Reference

Complete reference for all public functions and classes:

| Module | Description |
|--------|-------------|
| [`shannlp.tokenize`](api/tokenize.md) | `word_tokenize`, `syllable_tokenize`, `Tokenizer` |
| [`shannlp.spell`](api/spell.md) | `spell_correct`, `correct_sentence`, `SpellCorrector`, `ContextAwareCorrector` |
| [`shannlp.corpus`](api/corpus.md) | `shan_words`, `shan_stopwords`, `countries`, and more |
| [`shannlp.util`](api/util.md) | `num_to_shanword`, `convert_years`, `eng_to_shn`, and more |
| [`shannlp.tools`](api/tools.md) | `download_model`, `get_cache_dir`, `resolve_model_path` |

## About

ShanNLP is an experimental, self-research project inspired by [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp). It currently ships with a ~19,904 word dictionary sourced from Shan Wikipedia and collected word lists. The library requires Python 3.10 or later.

Contributions are welcome via [GitHub Issues](https://github.com/NoerNova/ShanNLP/issues).
