# API Reference: shannlp.util

```python
from shannlp.util import (
    num_to_shanword, shanword_to_num, text_to_num, words_to_num,
    arabic_digit_to_shan_digit, shan_digit_to_arabic_digit,
    digit_to_text, text_to_arabic_digit, text_to_shan_digit,
    shanword_to_date, convert_years,
    eng_to_shn, shn_to_eng,
    countshan,
)
```

---

## Number Conversion

### `num_to_shanword`

```python
num_to_shanword(number: int) -> str
```

Convert an integer to its Shan word representation.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `number` | int | required | Integer to convert (supports negative numbers) |

**Returns:** `str` — Shan word string. Returns `""` if `number` is `None`.

```python
from shannlp.util import num_to_shanword

num_to_shanword(0)      # "သုၼ်"
num_to_shanword(17)     # "သိပ်းၸဵတ်း"
num_to_shanword(2117)   # "သွင်ႁဵင်ၼိုင်ႈပၢၵ်ႇသိပ်းၸဵတ်း"
num_to_shanword(-1991)  # "ထွၼ်ႁဵင်ၵဝ်ႈပၢၵ်ႇၵဝ်ႈသိပ်းဢဵတ်း"
```

---

### `shanword_to_num`

```python
shanword_to_num(word: str) -> int
```

Convert a Shan number word string to an integer.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `word` | str | required | Shan number word string |

**Returns:** `int`

```python
from shannlp.util import shanword_to_num

shanword_to_num("သွင်ႁဵင်ၼိုင်ႈပၢၵ်ႇသိပ်းၸဵတ်း")  # 2117
shanword_to_num("ထွၼ်ႁဵင်ၵဝ်ႈပၢၵ်ႇၵဝ်ႈသိပ်းဢဵတ်း")   # -1991
```

---

### `text_to_num`

```python
text_to_num(text: str) -> List[str]
```

Parse mixed text and replace Shan number words with digit strings. Returns a list of tokens.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Mixed Shan text with number words |

**Returns:** `List[str]` — tokens with number words replaced by digit strings

```python
from shannlp.util import text_to_num

text_to_num("သွင်ႁဵင်ၼိုင်ႈပၢၵ်ႇသိပ်းၸဵတ်းပီပူၼ်ႉမႃး")
# ['2117', 'ပီ', 'ပူၼ်ႉ', 'မႃး']
```

---

### `words_to_num`

```python
words_to_num(words: List[str]) -> float
```

Convert a list of Shan number word tokens to a numeric value. Handles decimals via the word မၢႆ.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `words` | list of str | required | Tokenized Shan number words |

**Returns:** `float`

```python
from shannlp.util import words_to_num

words_to_num(["သွင်", "မၢႆ", "ႁႃႈ"])  # 2.5
```

---

## Digit Conversion

Shan digits: ႐ ႑ ႒ ႓ ႔ ႕ ႖ ႗ ႘ ႙ correspond to 0–9.

### `arabic_digit_to_shan_digit`

```python
arabic_digit_to_shan_digit(text: str) -> str
```

Replace Arabic digits with Shan digits in text.

```python
from shannlp.util import arabic_digit_to_shan_digit

arabic_digit_to_shan_digit("2117")  # "႒႑႑႗"
```

---

### `shan_digit_to_arabic_digit`

```python
shan_digit_to_arabic_digit(text: str) -> str
```

Replace Shan digits with Arabic digits in text.

```python
from shannlp.util import shan_digit_to_arabic_digit

shan_digit_to_arabic_digit("႒႑႑႗")  # "2117"
```

---

### `digit_to_text`

```python
digit_to_text(text: str) -> str
```

Convert Shan digits within text to spelled-out Shan words.

```python
from shannlp.util import digit_to_text

digit_to_text("မႂ်ႇသုင်ပီမႂ်ႇတႆး ႒႑႑႗ ၼီႈ")
# "မႂ်ႇသုင်ပီမႂ်ႇတႆး သွင်ၼိုင်ႈၼိုင်ႈၸဵတ်း ၼီႈ"
```

---

### `text_to_arabic_digit`

```python
text_to_arabic_digit(text: str) -> str
```

Convert a single spelled-out digit word to its Arabic digit character.

```python
from shannlp.util import text_to_arabic_digit

text_to_arabic_digit("သွင်")  # "2"
```

---

### `text_to_shan_digit`

```python
text_to_shan_digit(text: str) -> str
```

Convert a single spelled-out digit word to its Shan digit character.

```python
from shannlp.util import text_to_shan_digit

text_to_shan_digit("သွင်")  # "႒"
```

---

## Date and Calendar

### `shanword_to_date`

```python
shanword_to_date(text: str, date: datetime = None) -> Union[datetime, None]
```

Convert a Shan relative date word to a `datetime` object.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Shan date keyword (see table below) |
| `date` | datetime | None | Reference date. Defaults to `datetime.now()` |

**Returns:** `datetime` if `text` is recognized, else `None`

**Supported keywords:**

| Keyword | Meaning | Day offset |
|---------|---------|------------|
| မိူဝ်ႈၼႆႉ | Today | 0 |
| ၶမ်ႈၼႆႉ | Tonight | 0 |
| မိူဝ်ႈၽုၵ်ႈ | Tomorrow | +1 |
| ဝၼ်းမိူဝ်ႈၽုၵ်ႈ | Day after today | +1 |
| ၶမ်ႈၽုၵ်ႈ | Tomorrow evening | +1 |
| မိူဝ်ႈႁိုဝ်း | Day after tomorrow | +2 |
| မိူဝ်ႈဝႃး | Yesterday | -1 |
| ဝၼ်းမိူဝ်ႈဝႃး | Day before today | -1 |
| ၶမ်ႈဝႃး | Last evening | -1 |
| မိူဝ်ႈၶမ်ႈဝႃး | Yesterday evening | -1 |
| မိူဝ်ႈသိုၼ်း | Two days ago | -2 |
| ဝၼ်းသိုၼ်း | Day before yesterday | -2 |
| မိူဝ်ႈဝၼ်းသိုၼ်း | Two days ago | -2 |
| မိူဝ်ႈသိုၼ်းမိူဝ်ႈသၢၼ်း | Two days ago (alt) | -2 |

```python
from shannlp.util import shanword_to_date

result = shanword_to_date("မိူဝ်ႈၽုၵ်ႈ")  # datetime for tomorrow
```

---

### `convert_years`

```python
convert_years(year: str, src: str = "mo", target: str = "ad") -> str
```

Convert a year between calendar systems.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `year` | str or int | required | Year value to convert |
| `src` | str | `"mo"` | Source calendar system code |
| `target` | str | `"ad"` | Target calendar system code |

**Returns:** `str` — converted year

**Calendar codes:**

| Code | Calendar | Shan Name | Offset from AD |
|------|----------|-----------|---------------|
| `"ad"` | Gregorian | ပီဢိင်းၵရဵတ်ႈ | 0 |
| `"mo"` | Shan | ပီတႆး | AD + 94 |
| `"be"` | Buddhist | ပီပုတ်ႉထ | AD + 543 |
| `"ga"` | Goja | ပီၵေႃးၸႃႇ | AD - 638 |

**Raises:** `NotImplementedError` for unsupported source/target combinations

```python
from shannlp.util import convert_years

convert_years(2023, "ad", "mo")   # "2117"
convert_years(2117, "mo", "be")   # "2566"
convert_years(2566, "be", "ga")   # "1385"
```

---

## Keyboard Transliteration

### `eng_to_shn`

```python
eng_to_shn(text: str) -> str
```

Convert English keyboard characters to their Shan counterparts using the standard Shan keyboard layout.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Text typed using English keyboard layout |

**Returns:** `str` — Shan text

```python
from shannlp.util import eng_to_shn

eng_to_shn("rgfbokifcMj")  # "မႂ်ႇသုင်ၶႃႈ"
```

---

### `shn_to_eng`

```python
shn_to_eng(text: str) -> str
```

Convert Shan characters back to their English keyboard key equivalents.

```python
from shannlp.util import shn_to_eng

shn_to_eng("ေၺၺူၼ")  # "apple"
```

---

## Character Analysis

### `countshan`

```python
countshan(text: str, ignore_chars: str = <whitespace+digits+punctuation>) -> float
```

Return the percentage (0.0–100.0) of Shan characters in the text.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Input text |
| `ignore_chars` | str | whitespace + digits + punctuation | Characters to exclude from the count |

**Returns:** `float` — percentage of Shan characters (0.0 to 100.0). Returns 0.0 for empty or non-string input.

```python
from shannlp.util import countshan

countshan("မိူင်းတႆး")           # 100.0
countshan("မိူင်းတႆး hello")     # ~50.0
countshan("hello world")          # 0.0
countshan("မိူင်း abc", ignore_chars="")  # includes spaces in count
```

---

## Guide

See the [Utilities guide](../guides/utilities.md) for detailed examples.
