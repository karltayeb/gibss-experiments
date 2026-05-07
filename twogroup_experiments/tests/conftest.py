from __future__ import annotations

import sys
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[2]
SOFTWARE_SRC = Path.home() / "py" / "gibss-mono" / "src"

sys.path.insert(0, str(REPO_ROOT))
if SOFTWARE_SRC.exists():
    sys.path.insert(0, str(SOFTWARE_SRC))
