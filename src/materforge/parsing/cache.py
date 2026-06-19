# SPDX-FileCopyrightText: 2025 - 2026 Rahil Miten Doshi, Friedrich-Alexander-Universität Erlangen-Nürnberg
# SPDX-License-Identifier: BSD-3-Clause

"""Content-addressed cache for built :class:`~materforge.core.materials.Material`.

Building a material runs a piecewise ``pwlf`` regression for every tabular /
file-import property, which is by far the most expensive step in
:func:`materforge.create_material`. When the same YAML (and its referenced data
files) are loaded again unchanged, that work is pure repetition.

This module stores the built property expressions keyed by a hash of everything
that can change the result, so a second load returns the material without
re-running the regression. The cache is *advisory*: any miss, disabled flag, or
unreadable entry simply triggers a normal rebuild - it never raises.

Disable globally with ``MATERFORGE_DISABLE_CACHE=1``; relocate with
``MATERFORGE_CACHE_DIR``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import sympy as sp

from materforge.core.materials import Material
from materforge.parsing.config.yaml_keys import FILE_PATH_KEY

logger = logging.getLogger(__name__)

# Bump when the on-disk payload layout changes in a backwards-incompatible way.
_CACHE_FORMAT_VERSION = 1
_ENTRY_SUFFIX = ".mfcache"
_TRUE_FLAGS = {"1", "true", "yes", "on"}


def is_disabled() -> bool:
    """True when caching is switched off via ``MATERFORGE_DISABLE_CACHE``."""
    return os.environ.get("MATERFORGE_DISABLE_CACHE", "").strip().lower() in _TRUE_FLAGS


def cache_dir() -> Path:
    """Resolve the cache directory (``MATERFORGE_CACHE_DIR`` > XDG > ``~/.cache``)."""
    override = os.environ.get("MATERFORGE_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "materforge"


def _hash_file(path: Path, digest: "hashlib._Hash") -> None:
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)


def _collect_data_files(config: Any, base_dir: Path) -> List[Path]:
    """Walk a parsed config and resolve every ``file_path`` reference."""
    found: List[Path] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == FILE_PATH_KEY and isinstance(value, str):
                    found.append((base_dir / value).resolve())
                else:
                    walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(config)
    return found


def compute_key(yaml_path: Path, dependency: sp.Symbol, base_dir: Path, config: Any) -> str:
    """Hash everything that can change a built material into a cache key.

    Inputs: the YAML bytes, the bytes of every referenced ``file_path`` data
    file, the dependency symbol name, and the ``materforge`` / ``sympy``
    versions (so any upgrade invalidates stale entries automatically).
    """
    import materforge  # local import avoids a cycle at module load

    digest = hashlib.sha256()
    digest.update(f"fmt={_CACHE_FORMAT_VERSION}".encode())
    digest.update(f"materforge={materforge.__version__}".encode())
    digest.update(f"sympy={sp.__version__}".encode())
    digest.update(f"dependency={dependency.name}".encode())
    digest.update(b"yaml=")
    _hash_file(Path(yaml_path), digest)
    for data_file in sorted(_collect_data_files(config, base_dir)):
        digest.update(f"datafile={data_file.name}=".encode())
        if data_file.is_file():
            _hash_file(data_file, digest)
        else:
            digest.update(b"<missing>")
    return digest.hexdigest()


def load(key: str) -> Optional[Material]:
    """Return the cached material for ``key``, or ``None`` on any miss/error."""
    if is_disabled():
        return None
    entry = cache_dir() / f"{key}{_ENTRY_SUFFIX}"
    if not entry.is_file():
        return None
    try:
        with open(entry, "rb") as handle:
            payload: Dict[str, Any] = pickle.load(handle)
        return Material(name=payload["name"], properties=payload["properties"])
    except Exception as error:  # corrupt, truncated, or version-incompatible
        logger.warning("Ignoring unreadable cache entry %s: %s", entry, error)
        return None


def store(key: str, material: Material) -> None:
    """Persist ``material`` under ``key``; never raises (logs and skips on error)."""
    if is_disabled():
        return
    try:
        directory = cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"name": material.name, "properties": dict(material.properties)}
        final = directory / f"{key}{_ENTRY_SUFFIX}"
        tmp = directory / f"{key}{_ENTRY_SUFFIX}.{os.getpid()}.tmp"
        with open(tmp, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(final)  # atomic on the same filesystem
        logger.debug("Cached material '%s' at %s", material.name, final)
    except Exception as error:
        logger.warning("Could not write cache entry for '%s': %s", material.name, error)


def clear() -> int:
    """Delete all cached entries. Returns the number of entries removed."""
    directory = cache_dir()
    if not directory.is_dir():
        return 0
    removed = 0
    for entry in directory.glob(f"*{_ENTRY_SUFFIX}"):
        try:
            entry.unlink()
            removed += 1
        except OSError as error:
            logger.warning("Could not delete cache entry %s: %s", entry, error)
    logger.info("Cleared %d cached material(s) from %s", removed, directory)
    return removed
