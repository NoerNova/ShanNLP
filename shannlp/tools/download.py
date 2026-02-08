"""Model download and cache utilities for ShanNLP.

Models are cached in:
  - $SHANNLP_CACHE  (if set)
  - ~/.cache/shannlp/  (default)

Supported URL sources
---------------------
- Direct URLs (any host)
- Google Drive share links (``drive.google.com/file/d/…``) — converted automatically

Typical workflow
----------------
1. Obtain the model URL from the ShanNLP spell-correction documentation.
2. Download once::

       from shannlp.tools.download import download_model
       download_model(
           "spell_reranker",
           model_url="https://drive.google.com/file/d/<id>/view?usp=sharing",
           vocab_url="https://drive.google.com/file/d/<id>/view?usp=sharing",
       )

3. Load by name anywhere in your code::

       from shannlp.spell.neural import SpellReranker
       reranker = SpellReranker.load("spell_reranker")

   Or pass the URL directly to ``load()`` for a one-step download-and-load::

       reranker = SpellReranker.load(
           "spell_reranker",
           model_url="https://…",
           vocab_url="https://…",
       )
"""

from __future__ import annotations

import hashlib
import http.cookiejar
import os
import re
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------

def get_cache_dir() -> Path:
    """Return the local cache directory for ShanNLP models.

    Override by setting the ``SHANNLP_CACHE`` environment variable.
    """
    env_dir = os.environ.get("SHANNLP_CACHE")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path.home() / ".cache" / "shannlp"


def get_cached_path(name: str) -> Optional[Path]:
    """Return path to *name* inside the cache directory, or ``None``.

    Resolution order:

    1. Exact filename match (e.g. ``"shan_bigram.msgpack"``).
    2. Stem-only match when *name* has no extension (e.g. ``"spell_reranker"``
       matches ``spell_reranker.pt``).
    """
    cache = get_cache_dir()
    exact = cache / name
    if exact.exists():
        return exact

    # Stem-only lookup
    if "." not in os.path.basename(name):
        for f in sorted(cache.iterdir()):
            if f.stem == name:
                return f

    return None


# ---------------------------------------------------------------------------
# Google Drive helpers
# ---------------------------------------------------------------------------

_GDRIVE_ID_PATTERNS = [
    r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
    r"drive\.google\.com/open\?.*?id=([a-zA-Z0-9_-]+)",
    r"drive\.google\.com/uc\?.*?id=([a-zA-Z0-9_-]+)",
    r"docs\.google\.com/[^/]+/d/([a-zA-Z0-9_-]+)",
]


def _extract_gdrive_id(url: str) -> Optional[str]:
    """Extract the file ID from any Google Drive URL format."""
    for pattern in _GDRIVE_ID_PATTERNS:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def _resolve_url(url: str) -> str:
    """Resolve a URL to a direct download link.

    Google Drive share links are converted to direct download URLs using the
    ``drive.usercontent.google.com`` endpoint which bypasses the virus-scan
    confirmation page for public files of any size.
    """
    file_id = _extract_gdrive_id(url)
    if not file_id:
        return url
    return (
        f"https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&authuser=0&confirm=t"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CHUNK = 65536  # 64 KiB read / write chunks
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _sha256_of_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path, sha256: Optional[str] = None) -> None:
    """Download *url* to *dest*, verifying integrity when *sha256* is given.

    - Google Drive share links are resolved automatically.
    - Uses a temporary file so *dest* is never left partially written.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_url(url)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".")
    os.close(tmp_fd)
    tmp = Path(tmp_path)

    try:
        print(f"Downloading {dest.name} ...")

        req = urllib.request.Request(resolved, headers={"User-Agent": _USER_AGENT})
        cookie_jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cookie_jar)
        )

        with opener.open(req) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            downloaded = 0

            with open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = min(100, downloaded * 100 // total)
                        mb = downloaded / 1_048_576
                        total_mb = total / 1_048_576
                        print(
                            f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB",
                            end="",
                            flush=True,
                        )

        print()  # newline after progress

        if sha256 is not None:
            actual = _sha256_of_file(tmp)
            if actual != sha256:
                tmp.unlink(missing_ok=True)
                raise ValueError(
                    f"Integrity check failed for {dest.name}: "
                    f"expected {sha256}, got {actual}"
                )

        tmp.replace(dest)
        print(f"  Saved to {dest}")

    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_model(
    model_name: str,
    model_url: str,
    vocab_url: str,
    model_sha256: Optional[str] = None,
    vocab_sha256: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Download model and vocabulary files to the cache directory.

    Args:
        model_name: Short identifier used as the cache key (e.g. ``"spell_reranker"``).
            The files will be saved as ``<model_name>.pt`` and
            ``<model_name>_vocab.json`` inside the cache directory.
        model_url: Direct download URL for the ``.pt`` weights file.
        vocab_url: Direct download URL for the ``_vocab.json`` vocabulary file.
        model_sha256: Expected SHA-256 hex digest of the model file (optional).
        vocab_sha256: Expected SHA-256 hex digest of the vocab file (optional).
        force: Re-download even when cached files already exist.

    Returns:
        Path to the cached ``.pt`` model file.

    Raises:
        urllib.error.URLError: On network failure.
        ValueError: If an integrity check fails.

    Example::

        from shannlp.tools.download import download_model
        download_model(
            "spell_reranker",
            model_url="https://example.com/spell_reranker.pt",
            vocab_url="https://example.com/spell_reranker_vocab.json",
        )
    """
    cache_dir = get_cache_dir()

    model_dest = cache_dir / f"{model_name}.pt"
    vocab_dest = cache_dir / f"{model_name}_vocab.json"

    if not force and model_dest.exists():
        print(f"Using cached {model_dest.name}")
    else:
        _download_file(model_url, model_dest, model_sha256)

    if not force and vocab_dest.exists():
        print(f"Using cached {vocab_dest.name}")
    else:
        _download_file(vocab_url, vocab_dest, vocab_sha256)

    return model_dest


def download_file(
    filename: str,
    url: str,
    sha256: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Download a single model file to the cache directory.

    Use this for models stored as one file (e.g. ``.msgpack`` n-gram models).
    For multi-file neural models (``.pt`` + ``_vocab.json``), use
    :func:`download_model` instead.

    Args:
        filename: Name under which the file will be cached
            (e.g. ``"shan_bigram.msgpack"``).
        url: Direct download URL.  Google Drive share links are handled
            automatically.
        sha256: Expected SHA-256 hex digest (optional integrity check).
        force: Re-download even when already cached.

    Returns:
        Path to the cached file.

    Example::

        from shannlp.tools.download import download_file
        download_file(
            "shan_bigram.msgpack",
            url="https://drive.google.com/file/d/<id>/view?usp=sharing",
        )
    """
    dest = get_cache_dir() / filename
    if not force and dest.exists():
        print(f"Using cached {dest.name}")
    else:
        _download_file(url, dest, sha256)
    return dest


def resolve_model_path(name_or_path: str) -> str:
    """Resolve a model name or file path to an existing local file path.

    Works for any model file type (``.pt``, ``.msgpack``, etc.).

    Resolution order:

    1. If *name_or_path* exists on disk → return its absolute path.
    2. Try ``<cache_dir>/<basename>`` (exact filename in cache).
    3. If *name_or_path* has no extension, find the first file in cache whose
       stem matches (e.g. ``"spell_reranker"`` → ``spell_reranker.pt``).
    4. Raise :exc:`FileNotFoundError` with download instructions.

    Args:
        name_or_path: Existing file path, bare model name, or filename.

    Returns:
        Absolute path string to the resolved file.
    """
    if os.path.exists(name_or_path):
        return os.path.abspath(name_or_path)

    basename = os.path.basename(name_or_path)
    cached = get_cached_path(basename)
    if cached is not None:
        return str(cached)

    # Build a helpful error based on whether it looks like a single-file or
    # multi-file (neural) model.
    cache_dir = get_cache_dir()
    stem = os.path.splitext(basename)[0] if "." in basename else basename
    ext = os.path.splitext(basename)[1] if "." in basename else ""

    if ext in ("", ".pt"):
        hint = (
            f"    from shannlp.tools.download import download_model\n"
            f"    download_model(\n"
            f'        "{stem}",\n'
            f'        model_url="<URL to {stem}.pt>",\n'
            f'        vocab_url="<URL to {stem}_vocab.json>",\n'
            f"    )\n"
        )
    else:
        hint = (
            f"    from shannlp.tools.download import download_file\n"
            f"    download_file(\n"
            f'        "{basename}",\n'
            f'        url="<URL to {basename}>",\n'
            f"    )\n"
        )

    raise FileNotFoundError(
        f"Model not found: '{name_or_path}'\n"
        f"Cache directory: {cache_dir}\n"
        f"\nTo download, run:\n{hint}"
        f"See the ShanNLP docs for model download links."
    )
