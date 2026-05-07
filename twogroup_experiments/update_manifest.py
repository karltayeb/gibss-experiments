#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    script_path = Path(__file__).resolve()
    experiment_dir = script_path.parent
    repo_root = experiment_dir.parent
    software_src = Path.home() / "py" / "gibss-mono" / "src"
    results_dir = experiment_dir / "results"

    sys.path.insert(0, str(repo_root))
    if software_src.exists():
        sys.path.insert(0, str(software_src))

    from twogroup_experiments.utils import materialize_manifest

    manifest_path = results_dir / "twogroup_experiments_manifest.json"
    materialize_manifest(manifest_path)
    print(f"Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
