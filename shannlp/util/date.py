# -*- coding: utf-8 -*-


__all__ = ["convert_years", "shanword_to_date"]

from datetime import datetime, timedelta
from typing import Union

# https://shn.wikipedia.org/wiki/ဝၼ်းၸဵတ်းဝၼ်း_ၽၢႆႇတႆး

# MO: ပီတႆး 2117
# GA: ပီၵေႃးၸႃႇ 1385
# BE: ပီပုတ်ႉထ 2566
# AD: ပီဢိင်းၵရဵတ်ႈ 2023

shan_abbr_weekdays = ["တိ", "ၸ", "ၵ", "ပု", "ၽ", "သု", "သ"]
shan_full_weekdays = [
    "ဝၼ်းၸၼ်",
    "ဝၼ်းဢင်းၵၢၼ်း",
    "ဝၼ်းၽုတ်ႉ",
    "ဝၼ်းၽတ်း",
    "ဝၼ်းသုၵ်း",
    "ဝၼ်းသဝ်",
    "ဝၼ်းဢႃးတိတ်ႉ",
]

shan_full_months = [
    "လိူၼ်ၸဵင်",
    "လိူၼ်ၵမ်",
    "လိူၼ်သၢမ်",
    "လိူၼ်သီႇ",
    "လိူၼ်ႁႃႈ",
    "လိူၼ်ႁူၵ်",
    "လိူၼ်ၸဵတ်",
    "လိူၼ်ပႅတ်",
    "လိူၼ်ၵဝ်ႈ",
    "လိူၼ်သိပ်",
    "လိူၼ်သိပ်းဢဵတ်း",
    "လိူၼ်သိပ်းသွင်",
]

shan_full_month_lists = [
    ["လိူၼ်ၸဵင်", "01", "1"],
    ["လိူၼ်ၵမ်", "02", "2"],
    ["လိူၼ်သၢမ်", "03", "3"],
    ["လိူၼ်သီႇ", "04", "4"],
    ["လိူၼ်ႁႃႈ", "05", "5"],
    ["လိူၼ်ႁူၵ်း", "06", "6"],
    ["လိူၼ်ၸဵတ်း", "07", "7"],
    ["လိူၼ်ပႅတ်ႇ", "08", "8"],
    ["လိူၼ်ၵဝ်ႈ", "09", "9"],
    ["လိူၼ်သိပ်း", "10"],
    ["လိူၼ်သိပ်းဢဵတ်း", "11"],
    ["လိူၼ်သိပ်းသွင်", "12"],
]

shan_full_month_lists_regex = (
    "(" + "|".join([str("|".join([j for j in i])) for i in shan_full_month_lists]) + ")"
)
year_all_regex = r"(\d\d\d\d|\d\d)"
dates_list = (
    "("
    + "|".join(
        [str(i) for i in range(32, 0, -1)] + ["0" + str(i) for i in range(1, 10)]
    )
    + ")"
)

_DAY = {
    "မိူဝ်ႈၼႆႉ": 0,
    "ၶမ်ႈၼႆႉ": 0,
    "ဝၼ်းမိူဝ်ႈၽုၵ်ႈ": 1,
    "မိူဝ်ႈၽုၵ်ႈ": 1,
    "ၶမ်ႈၽုၵ်ႈ": 1,
    "မိူဝ်ႈႁိုဝ်း": 2,
    "ဝၼ်းမိူဝ်ႈဝႃး": -1,
    "မိူဝ်ႈဝႃး": -1,
    "ၶမ်ႈဝႃး": -1,
    "မိူဝ်ႈၶမ်ႈဝႃး": -1,
    "မိူဝ်ႈသိုၼ်း": -2,
    "ဝၼ်းသိုၼ်း": -2,
    "မိူဝ်ႈဝၼ်းသိုၼ်း": -2,
    "မိူဝ်ႈသိုၼ်းမိူဝ်ႈသၢၼ်း": -2,
}


def convert_years(year: str, src="mo", target="ad") -> str:
    """
    Convert years

    :param int year: year
    :param str src: source year
    :param str target: target year
    :return: converted year
    :rtype: str

    **Year options**
        * *mo* - Shan calendar
        * *ga* - Goja
        * *be* - Buddhist calendar
        * *ad* - Anno Domini

    """
    output_year = None
    if src == "be":
        if target == "ad":
            output_year = str(int(year) - 543)
        elif target == "mo":
            output_year = str(int(year) - 448)
        elif target == "ga":
            output_year = str(int(year) - 1181)
    elif src == "ad":
        if target == "be":
            output_year = str(int(year) + 543)
        elif target == "mo":
            output_year = str(int(year) + 94)
        elif target == "ga":
            output_year = str(int(year) - 638)
    elif src == "mo":
        if target == "ad":
            output_year = str(int(year) - 94)
        elif target == "be":
            output_year = str(int(year) + 448)
        elif target == "ga":
            output_year = str(int(year) - 732)
    elif src == "ga":
        if target == "ad":
            output_year = str(int(year) + 638)
        elif target == "be":
            output_year = str(int(year) + 1181)
        elif target == "mo":
            output_year = str(int(year) + 732)

    if output_year is None:
        raise NotImplementedError(f"This function doesn't support {src} to {target}")

    return output_year


def _find_month(text: str) -> int:
    for i, m in enumerate(shan_full_month_lists):
        for j in m:
            if j in text:
                return i + 1


def shanword_to_date(text: str, date: datetime = None) -> Union[datetime, None]:
    if text not in _DAY:
        return None

    day_num = _DAY.get(text)

    if not date:
        date = datetime.now()

    return date + timedelta(days=day_num)
