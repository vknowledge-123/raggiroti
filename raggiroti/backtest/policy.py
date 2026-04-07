from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Decision:
    action: str  # BUY|SELL|WAIT|EXIT
    sl_points: float = 15.0
    target_points: float | None = 40.0
    reason: str = ""


class Policy(Protocol):
    def decide(self, state: dict) -> Decision: ...

