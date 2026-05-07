"""
Small shared utilities for scripts/modules.

Design goals:
- Keep scripts short and reproducible
- Centralize JSON/JSONL I/O and logging configuration
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional


def setup_logging(level: str = "INFO") -> None:
    """Configure a simple, script-friendly logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_dir(path: str) -> str:
    """Create directory if missing and return the same path."""
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path: str) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    """Write an object to JSON with stable formatting."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, rows: Iterable[Dict]) -> None:
    """Append rows to a JSONL file (one JSON object per line)."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict]:
    """Read JSONL file into a list of dicts."""
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

