from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    side: str  # "LONG" | "SHORT"
    entry: float
    sl: float
    target: float | None
    qty: int


@dataclass
class Fill:
    dt: str
    side: str
    price: float
    qty: int
    reason: str


class BrokerSim:
    def __init__(self, breakeven_after_points: float | None = None) -> None:
        self.position: Position | None = None
        self.fills: list[Fill] = []
        self.realized_pnl_points: float = 0.0
        self.breakeven_after_points = breakeven_after_points

    def enter(self, dt: str, side: str, price: float, sl: float, target: float | None, qty: int, reason: str) -> None:
        if self.position is not None:
            return
        self.position = Position(side=side, entry=price, sl=sl, target=target, qty=qty)
        self.fills.append(Fill(dt=dt, side=f"ENTER_{side}", price=price, qty=qty, reason=reason))

    def exit(self, dt: str, price: float, reason: str) -> None:
        if self.position is None:
            return
        pos = self.position
        points = (price - pos.entry) if pos.side == "LONG" else (pos.entry - price)
        self.realized_pnl_points += points * pos.qty
        self.fills.append(Fill(dt=dt, side=f"EXIT_{pos.side}", price=price, qty=pos.qty, reason=reason))
        self.position = None

    def on_candle(self, dt: str, high: float, low: float) -> None:
        if self.position is None:
            return
        pos = self.position

        # Trailing to breakeven once market gives a clean move (rulebook: +15 pts -> BE).
        if self.breakeven_after_points is not None:
            if pos.side == "LONG" and (high - pos.entry) >= self.breakeven_after_points and pos.sl < pos.entry:
                pos.sl = pos.entry
                self.fills.append(Fill(dt=dt, side="ADJUST_SL", price=pos.sl, qty=pos.qty, reason="BE_AFTER_MOVE"))
            if pos.side == "SHORT" and (pos.entry - low) >= self.breakeven_after_points and pos.sl > pos.entry:
                pos.sl = pos.entry
                self.fills.append(Fill(dt=dt, side="ADJUST_SL", price=pos.sl, qty=pos.qty, reason="BE_AFTER_MOVE"))

        # SL first (conservative)
        if pos.side == "LONG" and low <= pos.sl:
            self.exit(dt, pos.sl, "SL")
            return
        if pos.side == "SHORT" and high >= pos.sl:
            self.exit(dt, pos.sl, "SL")
            return
        if pos.target is None:
            return
        if pos.side == "LONG" and high >= pos.target:
            self.exit(dt, pos.target, "TARGET")
        if pos.side == "SHORT" and low <= pos.target:
            self.exit(dt, pos.target, "TARGET")
