# -*- coding: utf-8 -*-

__all__ = ["num_to_shanword"]

_VALUES = [
    "",
    "ၼိုင်ႈ",
    "သွင်",
    "သၢမ်",
    "သီႇ",
    "ႁႃႈ",
    "ႁူၵ်း",
    "ၸဵတ်း",
    "ပႅတ်ႇ",
    "ၵဝ်ႈ",
]
_PLACES = ["", "သိပ်း", "ပၢၵ်ႇ", "ႁဵင်", "မိုၼ်ႇ", "သႅၼ်", "လၢၼ်ႉ"]
_EXCEPTIONS = {
    "ၼိုင်ႈသိပ်း": "သိပ်း",
    "သွင်သိပ်း": "သၢဝ်း",
    "သိပ်းၼိုင်ႈ": "သိပ်းဢဵတ်း",
    "သၢဝ်းၼိုင်ႈ": "သၢဝ်းဢဵတ်း",
}


def num_to_shanword(number: int) -> str:
    output = ""
    number_temp = number
    if number is None:
        return ""
    elif number == 0:
        output = "သုၼ်"

    number = str(abs(number))
    for place, value in enumerate(list(number[::-1])):
        if place % 6 == 0 and place > 0:
            output = _PLACES[6] + output

        if value != "0":
            output = _VALUES[int(value)] + _PLACES[place % 6] + output

    for search, replace in _EXCEPTIONS.items():
        output = output.replace(search, replace)

    if number_temp < 0:
        output = "ထွၼ်" + output

    return output
