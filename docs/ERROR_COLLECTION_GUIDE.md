# Error Collection Guide for Shan Spell Correction

## Why Collect Real Errors?

Synthetic errors (from `generate_synthetic_errors.py`) cover common patterns, but real user errors often have unique characteristics:

- Specific keyboard layout mistakes
- Common misconceptions about spelling
- Regional spelling variations
- Domain-specific errors

**Goal**: Collect 2,000-5,000 real error pairs to supplement 50K synthetic pairs.

## Data Format

### JSONL Format (Recommended)

```jsonl
{"error": "မိုင်း", "correct": "မိူင်း", "context_left": "ၵူၼ်း", "context_right": "ၵိၼ်ၶဝ်ႈ", "error_type": "real", "source": "facebook"}
```

### TSV Format (Easier to Edit)

```
error	correct	context_left	context_right	error_type	source
မိုင်း	မိူင်း	ၵူၼ်း	ၵိၼ်ၶဝ်ႈ	real	facebook
ၸိူင်	ၸိူဝ်း		ၼႆႉလီ	real	chat
```

## Collection Sources

### 1. Social Media Posts (Best Source)

- Facebook Shan language groups
- Twitter/X posts in Shan
- Telegram/WhatsApp groups

**How to collect**:
1. Find post with obvious spelling error
2. Note the error word and what it should be
3. Copy 1-2 words before and after for context
4. Record the source (facebook, twitter, etc.)

### 2. Chat Messages

- Personal conversations (with permission)
- Group chats
- Customer support logs

### 3. User Feedback

Once deployed, collect corrections from users:
```python
# In your app, log corrections
def log_correction(error, correct, context):
    with open("real_errors.jsonl", "a") as f:
        f.write(json.dumps({
            "error": error,
            "correct": correct,
            "context_left": context[:2],
            "context_right": context[2:],
            "error_type": "user_feedback"
        }) + "\n")
```

### 4. OCR Errors

If processing scanned documents:
- OCR often makes consistent mistakes
- Great source for character confusion pairs

## Error Categories to Focus On

| Priority | Error Type | Example | Why Important |
|----------|------------|---------|---------------|
| High | Tone marks | ႇ→ႈ, း→ႉ | Very common |
| High | Lead vowel order | ေၶ→ၶေ | Typing habit |
| High | Similar consonants | ပ→ၽ, ၵ→ၶ | Phonetic confusion |
| Medium | Vowel confusion | ိ→ီ, ု→ူ | Visual similarity |
| Medium | Missing asat | word→word် | Keyboard skip |
| Low | Extra characters | word→wordd | Typos |

## Collection Template

Use this spreadsheet template:

| Error | Correct | Context Left | Context Right | Type | Source | Notes |
|-------|---------|--------------|---------------|------|--------|-------|
| မိုင်း | မိူင်း | ၵူၼ်း | ၵိၼ်ၶဝ်ႈ | vowel | fb | common |
| | | | | | | |

## Quality Guidelines

### DO Include:
- Clear misspellings that have an obvious correction
- Errors with sufficient context (at least 1 word)
- Common errors you see repeatedly
- Domain-specific terms (place names, technical words)

### DON'T Include:
- Intentional misspellings (jokes, stylization)
- Code-switching errors (mixed language)
- Grammatical errors (word order, missing words)
- Ambiguous cases where multiple corrections are valid

## Merging with Synthetic Data

After collection, merge with synthetic pairs:

```bash
# Merge files
cat training_pairs.jsonl real_errors.jsonl > combined_training.jsonl

# Shuffle for training
python -c "
import random
with open('combined_training.jsonl') as f:
    lines = f.readlines()
random.shuffle(lines)
with open('final_training.jsonl', 'w') as f:
    f.writelines(lines)
"
```

## Recommended Collection Schedule

| Week | Target | Total |
|------|--------|-------|
| 1 | 500 pairs | 500 |
| 2 | 500 pairs | 1,000 |
| 3 | 500 pairs | 1,500 |
| 4 | 500 pairs | 2,000 |
| Ongoing | 100/week | 2,000+ |

## Converting from Other Formats

### From CSV:
```python
import csv
import json

with open('errors.csv', 'r') as f_in, open('errors.jsonl', 'w') as f_out:
    reader = csv.DictReader(f_in)
    for row in reader:
        f_out.write(json.dumps(row, ensure_ascii=False) + '\n')
```

### From Excel:
```python
import pandas as pd
import json

df = pd.read_excel('errors.xlsx')
with open('errors.jsonl', 'w') as f:
    for _, row in df.iterrows():
        f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
```

## Validation Script

Run this to validate your collected data:

```python
import json

errors = []
with open('real_errors.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            pair = json.loads(line)
            if not pair.get('error') or not pair.get('correct'):
                errors.append(f"Line {i}: Missing error or correct field")
            if pair['error'] == pair['correct']:
                errors.append(f"Line {i}: Error same as correct")
        except json.JSONDecodeError:
            errors.append(f"Line {i}: Invalid JSON")

if errors:
    print("Validation errors found:")
    for e in errors[:10]:
        print(f"  - {e}")
else:
    print("All entries valid!")
```

## Questions?

If unsure whether an error should be included, ask:
1. Is this a spelling mistake (not grammar)?
2. Is there one clear correct answer?
3. Would a native speaker recognize this as wrong?

If yes to all three, include it!
