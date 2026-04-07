# Rulebook (Nexus Ultra v2)

This folder contains the **machine-readable rulebook** used by `raggiroti`.

## Files

- `rulebook/nexus_ultra_v2.rulebook.json` — canonical rulebook (RAG-friendly JSON, currently v2.14.0).
- `rulebook/nexus_ultra_v2.rag_schema.json` — compact RAG integration schema (pointers to loops/matrices/rule IDs).

## How to use (RAG-friendly)

1. Treat each rule as **atomic**: retrieve by `tags`, `category`, and `condition`.
2. When new transcripts arrive, the system should:
   - extract candidate rules (condition/interpretation/action),
   - detect duplicates/conflicts,
   - propose a versioned update,
   - require human approval before activating for live trading.

## Key process (Loop)

Recommended operating loop is `LOOP-RAG-005` inside the JSON:

**Hypothesis -> Validate -> Trade**

This prevents overtrading and protects against "mismatch days".

## Integrated view

For a phase-by-phase integrated map (pre-market -> open -> intelligence -> execution -> review), see:

- `FRAMEWORK-001` under `frameworks` in `rulebook/nexus_ultra_v2.rulebook.json`.

## Fixed RR setup templates

For fixed risk-reward templates (Target = 75 points, SL = 30 points), see rules `DT-SL-213..216` inside `rulebook/nexus_ultra_v2.rulebook.json`.

## Swing Liquidity (important for SL hunting)

Your live/backtest engine maintains **confirmed swing highs/lows** (1m + 5m) and treats them as liquidity pools:

- Above swing highs: likely **short stops** (seller SL pool)
- Below swing lows: likely **long stops** (buyer SL pool)

Rulebook rules:

- `DT-SL-222` Swing High/Low = SL Pool
- `DT-SL-223` Swing Reclaim Trigger
- `DT-SL-224` 1m execution with 5m swing confirmation gate
- `DT-SL-225` Swing-stop placement (avoid exact swing levels)

Live API state fields (see `/api/live/state` and `/api/live/levels`):

- `swing_highs_1m`, `swing_lows_1m`, `last_swing_high_1m`, `last_swing_low_1m`
- `swing_highs_5m`, `swing_lows_5m`, `last_swing_high_5m`, `last_swing_low_5m`
- `broke_last_swing_high/low`, `reclaimed_last_swing_high/low`
