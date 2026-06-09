from __future__ import annotations

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "config.py"


def load_manifest_cached(cache_path: str | Path | None = None) -> dict:
    if cache_path is None:
        cache_path = Path(__file__).resolve().parent / "results" / "manifest_cache.json"
    cache_path = Path(cache_path)

    if cache_path.exists() and _CONFIG_PATH.stat().st_mtime <= cache_path.stat().st_mtime:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    from config import manifest_dict  # lazy — only triggered when cache is stale
    data = manifest_dict()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data
