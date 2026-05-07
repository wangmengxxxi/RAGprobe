"""JSON artifact IO placeholder.

Phase 0 keeps IO separate so v0.1 can add schema validation without touching
analysis logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    raise NotImplementedError("JSON loading will be implemented in v0.1.")


def save_json(data: Any, path: str | Path) -> None:
    raise NotImplementedError("JSON saving will be implemented in v0.1.")
