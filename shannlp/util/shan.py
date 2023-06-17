# -*- coding: utf-8 -*-
import string
from typing import Tuple

from shannlp import (
    shan_lead_vowels,
    shan_follow_vowels,
    shan_above_vowels,
    shan_below_vowels,
    shan_consonants,
    shan_vowels,
    shan_tonemarks,
    shan_punctuations,
    shan_digits,
    shan_characters
)
_DEFAULT_IGNORE_CHARS = string.whitespace + string.digits + string.punctuation


def isshanchar(ch: str) -> bool:
    if ch in shan_characters:
        return True
    return False


def isshan(text: str, ignore_chars: str = ".") -> bool:
    if not ignore_chars:
        ignore_chars = ""

    for ch in text:
        if ch not in ignore_chars and not isshanchar(ch):
            return False

    return True


def countshan(text: str, ignore_chars: str = _DEFAULT_IGNORE_CHARS) -> float:
    if not text or not isinstance(text, str):
        return 0.0

    if not ignore_chars:
        ignore_chars = ""

    num_shan = 0
    num_ignore = 0

    for ch in text:
        if ch in ignore_chars:
            num_ignore += 1
        elif isshanchar(ch):
            num_shan += 1

    num_count = len(text) - num_ignore

    if num_count == 0:
        return 0.0

    return (num_shan / num_count) * 100
