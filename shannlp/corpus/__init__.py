__all__ = [
    "countries",
    "get_corpus",
    "provinces",
    "shan_female_names",
    "shan_male_names",
    "shan_words",
    "shan_character",
    "shan_all_corpus",
]

import os
from typing import FrozenSet, List, Union
from shannlp.tools import get_shannlp_path

_CORPUS_DIRNAME = "corpus"
_CORPUS_PATH = os.path.join(get_shannlp_path(), _CORPUS_DIRNAME)

_COUNTRIES = set()
_COUNTRIES_FILENAME = "countries_shn.txt"

_SHAN_PROVINCES = set()
_SHAN_PROVINCES_FILENAME = "shan_state_provinces.txt"

_PERSON_FEMALE_NAMES = set()
_PERSON_FEMALE_NAMES_FILENAME = "person_names_female_shn.txt"
_PERSON_MALE_NAMES = set()
_PERSON_MALE_NAMES_FILENAME = "person_names_male_shn.txt"

_SHAN_WORDS = set()
_SHAN_WORDS_FILENAME = "words_shn.txt"
_SHAN_STOPWORDS = set()
_SHAN_STOPWORDS_FILENAME = "stopwords_shn.txt"

_SHAN_CHARACTER = set()
_SHAN_CHARACTER_FILENAME = "shan_character.txt"

_SHAN_ALL_C = set()


def corpus_path() -> str:
    return _CORPUS_PATH


def path_shannlp_corpus(filename: str) -> str:
    return os.path.join(corpus_path(), filename)


def get_corpus(filename: str, as_is: bool = False) -> Union[frozenset, list]:
    path = path_shannlp_corpus(filename)
    lines = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        lines = fh.read().splitlines()

    if as_is:
        return lines

    lines = [line.strip() for line in lines]
    return frozenset(filter(None, lines))


def get_m_corpus(
    filenames: List[str], as_is: bool = False
) -> Union[frozenset, List[str]]:
    all_lines = []
    for filename in filenames:
        path = path_shannlp_corpus(filename)
        with open(path, "r", encoding="utf-8-sig") as fh:
            lines = fh.read().splitlines()

        if as_is:
            all_lines.extend(lines)
        else:
            lines = [line.strip() for line in lines]
            all_lines.extend(filter(None, lines))

    if as_is:
        return all_lines
    else:
        return frozenset(all_lines)


def countries() -> FrozenSet[str]:
    global _COUNTRIES
    if not _COUNTRIES:
        _COUNTRIES = get_corpus(_COUNTRIES_FILENAME)

    return _COUNTRIES


def provinces(details: bool = False) -> Union[FrozenSet[str], List[str]]:
    global _SHAN_PROVINCES
    if not _SHAN_PROVINCES:
        _SHAN_PROVINCES = get_corpus(_SHAN_PROVINCES_FILENAME)

    return _SHAN_PROVINCES


def shan_words() -> FrozenSet[str]:
    global _SHAN_WORDS
    if not _SHAN_WORDS:
        _SHAN_WORDS = get_corpus(_SHAN_WORDS_FILENAME)

    return _SHAN_WORDS


def shan_stopwords() -> FrozenSet[str]:
    global _SHAN_STOPWORDS
    if not _SHAN_STOPWORDS:
        _SHAN_STOPWORDS = get_corpus(_SHAN_STOPWORDS_FILENAME)

    return _SHAN_STOPWORDS


def shan_female_names() -> FrozenSet[str]:
    global _PERSON_FEMALE_NAMES
    if not _PERSON_FEMALE_NAMES:
        _PERSON_FEMALE_NAMES = get_corpus(_PERSON_FEMALE_NAMES_FILENAME)

    return _PERSON_FEMALE_NAMES


def shan_male_names() -> FrozenSet[str]:
    global _PERSON_MALE_NAMES
    if not _PERSON_MALE_NAMES:
        _PERSON_MALE_NAMES = get_corpus(_PERSON_MALE_NAMES_FILENAME)

    return _PERSON_MALE_NAMES


def shan_character() -> FrozenSet[str]:
    global _SHAN_CHARACTER
    if not _SHAN_CHARACTER:
        _SHAN_CHARACTER = get_corpus(_SHAN_CHARACTER_FILENAME)

    return _SHAN_CHARACTER


def shan_all_corpus() -> FrozenSet[str]:
    global _SHAN_ALL_C
    if not _SHAN_ALL_C:
        _SHAN_ALL_C = get_m_corpus(
            [
                _COUNTRIES_FILENAME,
                _SHAN_PROVINCES_FILENAME,
                _SHAN_WORDS_FILENAME,
                _SHAN_STOPWORDS_FILENAME,
                _PERSON_FEMALE_NAMES_FILENAME,
                _PERSON_MALE_NAMES_FILENAME,
                _SHAN_CHARACTER_FILENAME,
            ]
        )

    return _SHAN_ALL_C
