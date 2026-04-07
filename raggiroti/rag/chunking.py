from __future__ import annotations

import re
from typing import Iterable


def chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    # Simple paragraph chunking; good enough to start.
    # Later: chunk by headings/topic boundaries + token-based limits.
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if not current:
            current = p
            continue
        if len(current) + 2 + len(p) <= max_chars:
            current = current + "\n\n" + p
        else:
            chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks

