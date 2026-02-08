# API Reference: shannlp.tools

```python
from shannlp.tools import get_cache_dir, get_shannlp_path, download_file, download_model, resolve_model_path
# or
from shannlp.tools.download import get_cache_dir, download_file, download_model, resolve_model_path
from shannlp.tools.path import get_shannlp_path
```

---

## Cache Management

### `get_cache_dir`

```python
get_cache_dir() -> Path
```

Return the local cache directory for ShanNLP models.

**Returns:** `pathlib.Path`

**Default location:** `~/.cache/shannlp/`

**Override:** Set the `SHANNLP_CACHE` environment variable:

```bash
export SHANNLP_CACHE=/custom/path
```

```python
from shannlp.tools import get_cache_dir

print(get_cache_dir())
# PosixPath('/home/user/.cache/shannlp')
```

---

### `get_shannlp_path`

```python
get_shannlp_path() -> str
```

Return the filesystem path to the installed `shannlp` package directory. Used internally to locate bundled corpus and model files.

**Returns:** `str` — absolute path to the package directory

```python
from shannlp.tools import get_shannlp_path

print(get_shannlp_path())
# /path/to/site-packages/shannlp
```

---

## Downloading Files

### `download_file`

```python
download_file(
    filename: str,
    url: str,
    sha256: Optional[str] = None,
    force: bool = False,
) -> Path
```

Download a single model file to the cache directory. Use this for single-file models (e.g. `.msgpack` n-gram models).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | str | required | Name to save the file as (e.g. `"shan_bigram.msgpack"`) |
| `url` | str | required | Download URL. Google Drive share links are supported automatically. |
| `sha256` | str | None | Expected SHA-256 hex digest for integrity verification |
| `force` | bool | False | Re-download even if file is already cached |

**Returns:** `pathlib.Path` — path to the cached file

**Raises:**
- `urllib.error.URLError` — on network failure
- `ValueError` — if SHA-256 integrity check fails

```python
from shannlp.tools import download_file

path = download_file(
    "shan_bigram.msgpack",
    url="https://drive.google.com/file/d/<id>/view?usp=sharing",
)
print(path)  # ~/.cache/shannlp/shan_bigram.msgpack
```

---

### `download_model`

```python
download_model(
    model_name: str,
    model_url: str,
    vocab_url: str,
    model_sha256: Optional[str] = None,
    vocab_sha256: Optional[str] = None,
    force: bool = False,
) -> Path
```

Download a two-file neural model (`.pt` weights + `_vocab.json` vocabulary) to the cache directory.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | required | Short model identifier (e.g. `"spell_reranker"`). Files are saved as `<name>.pt` and `<name>_vocab.json`. |
| `model_url` | str | required | Download URL for the `.pt` weights file |
| `vocab_url` | str | required | Download URL for the `_vocab.json` vocabulary file |
| `model_sha256` | str | None | Expected SHA-256 digest for the model file |
| `vocab_sha256` | str | None | Expected SHA-256 digest for the vocab file |
| `force` | bool | False | Re-download even if files exist |

**Returns:** `pathlib.Path` — path to the cached `.pt` model file

**Raises:**
- `urllib.error.URLError` — on network failure
- `ValueError` — if integrity check fails

```python
from shannlp.tools import download_model

path = download_model(
    "spell_reranker",
    model_url="https://example.com/spell_reranker.pt",
    vocab_url="https://example.com/spell_reranker_vocab.json",
)
```

---

## Resolving Model Paths

### `resolve_model_path`

```python
resolve_model_path(name_or_path: str) -> str
```

Resolve a model name or file path to an existing local file path.

**Resolution order:**
1. If `name_or_path` exists on disk → return absolute path
2. Try `<cache_dir>/<basename>` (exact filename match)
3. If no extension, find first file in cache whose stem matches (e.g. `"spell_reranker"` → `spell_reranker.pt`)
4. Raise `FileNotFoundError` with download instructions

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name_or_path` | str | required | File path, model name, or filename |

**Returns:** `str` — absolute path to the resolved file

**Raises:** `FileNotFoundError` with instructions on how to download the missing model

```python
from shannlp.tools import resolve_model_path

path = resolve_model_path("spell_reranker")
# Returns: "/home/user/.cache/shannlp/spell_reranker.pt"

path = resolve_model_path("shan_bigram.msgpack")
# Returns: "/home/user/.cache/shannlp/shan_bigram.msgpack"
```

---

## Guide

See the [Model Download guide](../guides/model-download.md) for practical examples including Google Drive links and integrity verification.
