# -*- coding: utf-8 -*-

import re
from typing import List

from shannlp.tokenize import Tokenizer
from shannlp.corpus import shan_words
from shannlp.tokenize import syllable_tokenize

_ptn_digits = r"(|ၼိုင်ႈ|ဢဵတ်း|သွင်|သၢမ်|သီႇ|ႁႃႈ|ႁူၵ်း|ၸဵတ်း|ပႅတ်ႇ|ၵဝ်ႈ)"
_ptn_six_figures = (
    rf"({_ptn_digits}သႅၼ်)?({_ptn_digits}မိုၼ်ႇ)?({_ptn_digits}ႁဵင်)?"
    rf"({_ptn_digits}ပၢၵ်ႇ)?({_ptn_digits}သၢဝ်း)?({_ptn_digits}သိပ်း)?{_ptn_digits}?"
)
_ptn_shan_numerals = rf"(ထွၼ်)?({_ptn_six_figures}လၢၼ်ႉ)*{_ptn_six_figures}"
_re_shan_numerals = re.compile(_ptn_shan_numerals)

_digits = {
    # "သုၼ်" was excluded as a special case
    "ၼိုင်ႈ": 1,
    "ဢဵတ်း": 1,
    "သွင်": 2,
    "သၢမ်": 3,
    "သီႇ": 4,
    "ႁႃႈ": 5,
    "ႁူၵ်း": 6,
    "ၸဵတ်း": 7,
    "ပႅတ်ႇ": 8,
    "ၵဝ်ႈ": 9,
}
_powers_of_10 = {
    "သိပ်း": 10,
    "သၢဝ်း": 20,
    "ပၢၵ်ႇ": 100,
    "ႁဵင်": 1000,
    "မိုၼ်ႇ": 10000,
    "သႅၼ်": 100000,
    # "လၢၼ်ႉ" was excluded as a special case
}
_valid_tokens = (
    set(_digits.keys()) | set(_powers_of_10.keys()) | {"လၢၼ်ႉ", "ထွၼ်"}
)
_tokenizer = Tokenizer(engine="newmm", custom_dict=_valid_tokens)


def _check_is_shannum(word: str):
    for j in list(_digits.keys()):
        if j in word:
            return True, "num"
    for j in ["သိပ်း", "သၢဝ်း", "ပၢၵ်ႇ", "ႁဵင်", "မိုၼ်ႇ", "သႅၼ်", "လၢၼ်ႉ", "မၢႆ", "ထွၼ်"]:
        if j in word:
            return True, "unit"
    return False, None


_dict_words = [i for i in list(shan_words()) if not _check_is_shannum(i)[0]]
_dict_words += list(_digits.keys())
_dict_words += ["သိပ်း", "သၢဝ်း", "ပၢၵ်ႇ", "ႁဵင်", "မိုၼ်ႇ", "သႅၼ်", "လၢၼ်ႉ", "မၢႆ"]

_tokenizer_shanwords = Tokenizer(engine="newmm", custom_dict=_dict_words)


def shanword_to_num(word: str) -> int:
    if not isinstance(word, str):
        raise TypeError(f"The input must be a string; given {word!r}")
    if not word:
        raise ValueError("The input string cannot be empty")
    if word == "သုၼ်":
        return 0
    if not _re_shan_numerals.fullmatch(word):
        raise ValueError("The input string is not a valid Shan numeral")

    # tokens = _tokenizer.word_tokenize(word)
    tokens = syllable_tokenize(word)
    accumulated = 0
    next_digit = 1

    is_minus = False
    if tokens[0] == "ထွၼ်":
        is_minus = True
        tokens.pop(0)

    for token in tokens:
        if token in _digits:
            next_digit = _digits[token]
        elif token in _powers_of_10:
            accumulated += max(next_digit, 1) * _powers_of_10[token]
            next_digit = 0
        else:
            accumulated = (accumulated + next_digit) * 1000000
            next_digit = 0

    accumulated += next_digit

    if is_minus:
        accumulated = -accumulated

    return accumulated


def _decimal_unit(words: list) -> float:
    _num = 0.0
    for i, v in enumerate(words):
        _num += int(shanword_to_num(v)) / (10 ** (i + 1))

    return _num


def words_to_num(words: list) -> float:
    num = 0
    if "မၢႆ" not in words:
        num = shanword_to_num("".join(words))
    else:
        words_int = "".join(words[: words.index("မၢႆ")])
        words_float = words[words.index("မၢႆ") + 1:]
        num = shanword_to_num(words_int)
        if num <= -1:
            num -= _decimal_unit(words_float)
        else:
            num += _decimal_unit(words_float)

    return num


def text_to_num(text: str) -> List[str]:
    _temp = _tokenizer_shanwords.word_tokenize(text)
    shannum = []
    last_index = -1
    list_word_new = []
    for i, word in enumerate(_temp):
        if (
            _check_is_shannum(word)[0]
            and last_index + 1 == i
            and i + 1 == len(_temp)
        ):
            shannum.append(word)
            list_word_new.append(str(words_to_num(shannum)))
        elif _check_is_shannum(word)[0] and last_index + 1 == i:
            shannum.append(word)
            last_index = i
        elif _check_is_shannum(word)[0]:
            shannum.append(word)
            last_index = i
        elif (
            not _check_is_shannum(word)[0]
            and last_index + 1 == i
            and last_index != -1
        ):
            list_word_new.append(str(words_to_num(shannum)))
            shannum = []
            list_word_new.append(word)
        else:
            list_word_new.append(word)
            last_index = -1

    return list_word_new

