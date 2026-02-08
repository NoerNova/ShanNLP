# Utilities

The `shannlp.util` module provides helper functions for number conversion, digit conversion, date/calendar conversion, keyboard transliteration, and character analysis.

## Number Conversion

### `num_to_shanword(number)`

Converts an integer to its Shan word representation:

```python
from shannlp.util import num_to_shanword

print(num_to_shanword(0))     # သုၼ်
print(num_to_shanword(17))    # သိပ်းၸဵတ်း
print(num_to_shanword(2117))  # သွင်ႁဵင်ၼိုင်ႈပၢၵ်ႇသိပ်းၸဵတ်း
print(num_to_shanword(-1991)) # ထွၼ်ႁဵင်ၵဝ်ႈပၢၵ်ႇၵဝ်ႈသိပ်းဢဵတ်း
```

**Place value words:**

| Place | Shan | Value |
|-------|------|-------|
| Ones | varies | 1–9 |
| Tens | သိပ်း | 10 |
| Hundreds | ပၢၵ်ႇ | 100 |
| Thousands | ႁဵင် | 1,000 |
| Ten-thousands | မိုၼ်ႇ | 10,000 |
| Hundred-thousands | သႅၼ် | 100,000 |
| Millions | လၢၼ်ႉ | 1,000,000 |

### `shanword_to_num(word)`

Converts Shan number words back to integers:

```python
from shannlp.util import shanword_to_num

print(shanword_to_num("သွင်ႁဵင်ၼိုင်ႈပၢၵ်ႇသိပ်းၸဵတ်း"))  # 2117
print(shanword_to_num("ထွၼ်ႁဵင်ၵဝ်ႈပၢၵ်ႇၵဝ်ႈသိပ်းဢဵတ်း"))   # -1991
```

### `text_to_num(text)`

Extracts and converts number words from mixed text. Returns tokens with number words replaced by digit strings:

```python
from shannlp.util import text_to_num

result = text_to_num("သွင်ႁဵင်ၼိုင်ႈပၢၵ်ႇသိပ်းၸဵတ်းပီပူၼ်ႉမႃး")
print(result)  # ['2117', 'ပီ', 'ပူၼ်ႉ', 'မႃး']
```

### `words_to_num(words)`

Converts a list of number word tokens to a numeric value. Supports decimals via the word မၢႆ:

```python
from shannlp.util import words_to_num

# Pass tokenized number words
result = words_to_num(["သွင်", "မၢႆ", "ႁႃႈ"])  # 2.5
```

## Digit Conversion

Shan has its own numeral digits: ႐ ႑ ႒ ႓ ႔ ႕ ႖ ႗ ႘ ႙ (0–9).

### `arabic_digit_to_shan_digit(text)`

Converts Arabic digits to Shan digits:

```python
from shannlp.util import arabic_digit_to_shan_digit

print(arabic_digit_to_shan_digit("2117"))  # ႒႑႑႗
```

### `shan_digit_to_arabic_digit(text)`

Converts Shan digits back to Arabic:

```python
from shannlp.util import shan_digit_to_arabic_digit

print(shan_digit_to_arabic_digit("႒႑႑႗"))  # 2117
```

### `digit_to_text(text)`

Converts Shan digits within text to spelled-out Shan words:

```python
from shannlp.util import digit_to_text

print(digit_to_text("မႂ်ႇသုင်ပီမႂ်ႇတႆး ႒႑႑႗ ၼီႈ"))
# မႂ်ႇသုင်ပီမႂ်ႇတႆး သွင်ၼိုင်ႈၼိုင်ႈၸဵတ်း ၼီႈ
```

### `text_to_arabic_digit(text)` / `text_to_shan_digit(text)`

Converts a spelled-out single digit word to its Arabic or Shan digit form:

```python
from shannlp.util import text_to_arabic_digit, text_to_shan_digit

print(text_to_arabic_digit("သွင်"))  # 2
print(text_to_shan_digit("သွင်"))    # ႒
```

## Date and Calendar Conversion

### `shanword_to_date(text, date=None)`

Converts Shan relative date words to a Python `datetime` object. If `date` is not provided, uses the current date/time.

```python
from shannlp.util import shanword_to_date
import datetime

print(f"Today: {datetime.date.today()}")
print(f"Tomorrow: {shanword_to_date('မိူဝ်ႈၽုၵ်ႈ')}")
print(f"Yesterday: {shanword_to_date('မိူဝ်ႈဝႃး')}")
```

**Supported date keywords:**

| Shan Word | Meaning | Offset |
|-----------|---------|--------|
| မိူဝ်ႈၼႆႉ | Today | 0 |
| ၶမ်ႈၼႆႉ | Tonight | 0 |
| မိူဝ်ႈၽုၵ်ႈ | Tomorrow | +1 |
| ဝၼ်းမိူဝ်ႈၽုၵ်ႈ | Day after today | +1 |
| မိူဝ်ႈႁိုဝ်း | Day after tomorrow | +2 |
| မိူဝ်ႈဝႃး | Yesterday | -1 |
| ၶမ်ႈဝႃး | Last evening | -1 |
| မိူဝ်ႈသိုၼ်း | Two days ago | -2 |

### `convert_years(year, src, target)`

Converts years between calendar systems. Returns the converted year as a string.

```python
from shannlp.util import convert_years

# AD → Shan
print(convert_years(2023, "ad", "mo"))   # 2117

# Shan → Buddhist
print(convert_years(2117, "mo", "be"))   # 2566

# Buddhist → Goja
print(convert_years(2566, "be", "ga"))   # 1385
```

**Calendar system codes:**

| Code | Calendar | Shan Name |
|------|----------|-----------|
| `"ad"` | Gregorian (Anno Domini) | ပီဢိင်းၵရဵတ်ႈ |
| `"mo"` | Shan traditional | ပီတႆး |
| `"be"` | Buddhist Era | ပီပုတ်ႉထ |
| `"ga"` | Goja | ပီၵေႃးၸႃႇ |

> Converting between `"mo"` and `"ga"` directly is also supported. Any unsupported combination raises `NotImplementedError`.

## Keyboard Transliteration

ShanNLP supports transliteration between English keyboard keys and Shan characters using a standard Shan keyboard layout mapping.

### `eng_to_shn(text)`

Type using English keyboard layout, get Shan characters:

```python
from shannlp.util import eng_to_shn

print(eng_to_shn("rgfbokifcMj"))  # မႂ်ႇသုင်ၶႃႈ
```

### `shn_to_eng(text)`

Convert Shan characters back to their keyboard key equivalents:

```python
from shannlp.util import shn_to_eng

print(shn_to_eng("ေၺၺူၼ"))  # apple
```

## Character Analysis

### `countshan(text, ignore_chars=...)`

Returns the percentage (0.0–100.0) of Shan characters in the text. Whitespace, digits, and punctuation are ignored by default:

```python
from shannlp.util import countshan

print(countshan("မိူင်းတႆး"))          # 100.0
print(countshan("မိူင်းတႆး hello"))    # 50.0 (approx)
print(countshan("hello world"))         # 0.0
```

Pass `ignore_chars=""` to count every character including spaces:

```python
countshan("မိူင်း abc", ignore_chars="")
```

## Full API Reference

See [api/util.md](../api/util.md) for complete documentation.
