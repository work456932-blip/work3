from __future__ import annotations

from pathlib import Path


def read_text_if_exists(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ''
    return p.read_text(encoding='utf-8', errors='ignore').strip()
