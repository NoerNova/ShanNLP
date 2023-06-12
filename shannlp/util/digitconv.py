# -*- coding: utf-8 -*-

_arabic_shan = {
    "0": "႐",
    "1": "႑",
    "2": "႒",
    "3": "႓",
    "4": "႔",
    "5": "႕",
    "6": "႖",
    "7": "႗",
    "8": "႘",
    "9": "႙",
}

_shan_arabic = {
    "႐": "0",
    "႑": "1",
    "႒": "2",
    "႓": "3",
    "႔": "4",
    "႕": "5",
    "႖": "6",
    "႗": "7",
    "႘": "8",
    "႙": "9",
}

_digit_spell = {
    "0": "သုၼ်",
    "1": "ၼိုင်ႈ",
    "2": "သွင်",
    "3": "သၢမ်",
    "4": "သီႇ",
    "5": "ႁႃႈ",
    "6": "ႁူၵ်း",
    "7": "ၸဵတ်း",
    "8": "ပႅတ်ႇ",
    "9": "ၵဝ်ႈ",
}

_spell_digit = {
    "သုၼ်": "0",
    "ၼိုင်ႈ": "1",
    "သွင်": "2",
    "သၢမ်": "3",
    "သီႇ": "4",
    "ႁႃႈ": "5",
    "ႁူၵ်း": "6",
    "ၸဵတ်း": "7",
    "ပႅတ်ႇ": "8",
    "ၵဝ်ႈ": "9",
}

_arabic_shan_translate_table = str.maketrans(_arabic_shan)
_shan_arabic_translate_table = str.maketrans(_shan_arabic)
_digit_spell_translate_table = str.maketrans(_digit_spell)


def shan_digit_to_arabic_digit(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    return text.translate(_shan_arabic_translate_table)


def arabic_digit_to_shan_digit(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    return text.translate(_arabic_shan_translate_table)


def digit_to_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = text.translate(_shan_arabic_translate_table)
    text = text.translate(_digit_spell_translate_table)

    return text


def text_to_arabic_digit(text: str) -> str:
    if not text or text not in _spell_digit:
        return ""

    return _spell_digit[text]


def text_to_shan_digit(text: str) -> str:
    return arabic_digit_to_shan_digit(text_to_arabic_digit(text))
