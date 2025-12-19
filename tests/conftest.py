from __future__ import annotations

# Pytest import hygiene:
# Some environments run pytest with a working directory/sys.path setup that doesn't
# automatically include the repo root. We make it explicit so `import course` works
# reliably without requiring installation.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
