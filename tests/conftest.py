"""
Shared pytest configuration.

steam_real's src/ modules import as `from src.xxx import ...` (bare `src`),
which only resolves correctly when steam_real/ itself is on sys.path -- the
same requirement run_steam_pipeline.py has when invoked from within
steam_real/. We add it here so `import src.features` etc. works uniformly
across the test suite regardless of pytest's invocation directory.

shared/stability.py is a plain (non-package) module imported the same way by
anime_simulated/src/stability.py and steam_real/stability.py -- add shared/
to sys.path too, so `import stability` resolves in tests as well.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STEAM_REAL_DIR = REPO_ROOT / "steam_real"
SHARED_DIR = REPO_ROOT / "shared"

if str(STEAM_REAL_DIR) not in sys.path:
    sys.path.insert(0, str(STEAM_REAL_DIR))
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
