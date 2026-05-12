from __future__ import annotations

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
TWOGROUP_ROOT = TESTS_DIR.parent

sys.path.insert(0, str(TWOGROUP_ROOT))
