# Model Download

## Overview

ShanNLP uses two types of downloadable models:

| Model Type | File | Purpose |
|------------|------|---------|
| N-gram model | `.msgpack` | Context-aware spell correction |
| Neural reranker | `.pt` + `_vocab.json` | Improved spelling reranking (requires PyTorch) |

Models are cached locally so they are only downloaded once.

## Cache Location

By default, models are stored in:

```
~/.cache/shannlp/
```

Override with the `SHANNLP_CACHE` environment variable:

```bash
export SHANNLP_CACHE=/my/custom/path
```

## Downloading an N-gram Model

Use `download_file` for single-file models like n-gram `.msgpack` files:

```python
from shannlp.tools import download_file

download_file(
    "shan_bigram.msgpack",
    url="https://example.com/shan_bigram.msgpack",
)
```

Google Drive share links are handled automatically:

```python
download_file(
    "shan_bigram.msgpack",
    url="https://drive.google.com/file/d/<file-id>/view?usp=sharing",
)
```

Then load the model:

```python
from shannlp.spell import reload_model

reload_model("shan_bigram.msgpack")
```

Or pass the path directly to `correct_sentence`:

```python
from shannlp import correct_sentence

result = correct_sentence("text here", model_path="~/.cache/shannlp/shan_bigram.msgpack")
```

## Downloading a Neural Reranker Model

Neural models consist of two files: a weights file (`.pt`) and a vocabulary file (`_vocab.json`). Use `download_model`:

```python
from shannlp.tools import download_model

download_model(
    "spell_reranker",
    model_url="https://example.com/spell_reranker.pt",
    vocab_url="https://example.com/spell_reranker_vocab.json",
)
```

Then load it:

```python
from shannlp.spell import load_neural_model

load_neural_model("spell_reranker")
```

> Neural model support requires PyTorch: `pip install torch`

## Checking the Cache

```python
from shannlp.tools import get_cache_dir

print(get_cache_dir())
# /home/user/.cache/shannlp
```

You can also inspect the directory directly:

```bash
ls ~/.cache/shannlp/
```

## Resolving a Model Path

Use `resolve_model_path` to find a model by name or path. It searches in order:

1. Exact file path (if it exists on disk)
2. `<cache_dir>/<filename>` exact match
3. First file in cache whose stem matches (e.g. `"spell_reranker"` → `spell_reranker.pt`)

```python
from shannlp.tools import resolve_model_path

path = resolve_model_path("spell_reranker")
print(path)  # /home/user/.cache/shannlp/spell_reranker.pt
```

If the model is not found, a `FileNotFoundError` is raised with instructions on how to download it.

## Force Re-download

Pass `force=True` to re-download even if the file is already cached:

```python
download_file("shan_bigram.msgpack", url="...", force=True)
download_model("spell_reranker", model_url="...", vocab_url="...", force=True)
```

## Integrity Verification

Optionally pass a SHA-256 hash to verify the download:

```python
download_file(
    "shan_bigram.msgpack",
    url="https://example.com/shan_bigram.msgpack",
    sha256="abc123...",
)
```

## Full API Reference

See [api/tools.md](../api/tools.md) for complete documentation.
