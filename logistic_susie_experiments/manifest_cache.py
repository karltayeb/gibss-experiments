from __future__ import annotations
import json
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"


def _experiments_mtime() -> float:
    return max((p.stat().st_mtime for p in _EXPERIMENTS_DIR.glob("*.yaml")), default=0.0)


def load_manifest_cached(cache_path: str | Path | None = None) -> dict:
    if cache_path is None:
        cache_path = Path(__file__).resolve().parent / "results" / "manifest_cache.json"
    cache_path = Path(cache_path)
    if cache_path.exists() and _experiments_mtime() <= cache_path.stat().st_mtime:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    from experiments import loader
    cfg = loader.load_config()
    data = loader.manifest_dict(cfg["library"], cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data
