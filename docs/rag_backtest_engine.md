# RAG + Backtesting Engine (Step-by-step)

This is an **engineering** guide for building `raggiroti` as:

`transcripts -> (RAG extraction) -> versioned rulebook -> (deterministic backtest) -> live trading`

## 0) Key principle (important)

For **backtesting**, avoid calling an LLM on every candle. It will be slow, expensive, and non-deterministic.

Recommended split:

- LLM/RAG is used for **knowledge ingestion** (extract rules, detect conflicts, propose updates).
- Backtest uses a **deterministic policy** derived from the rulebook + computed market state.
- Optional: LLM can be used at **decision points only** (events), not every candle.

Note: current simulator does **not** do pyramiding or trailing stops by default.

## Install

- `py -m pip install -r requirements.txt`

## Quick next-day planning (paste CSV)

The simplest "next day levels" output lives in:

- `scripts/plan-next-day.ps1`

If your machine blocks scripts by default, run with ExecutionPolicy bypass:

- From a file: `powershell -ExecutionPolicy Bypass -NoProfile -File .\\scripts\\plan-next-day.ps1 -CsvPath .\\path\\to\\candles.csv`
- From pasted text: `powershell -ExecutionPolicy Bypass -NoProfile -File .\\scripts\\plan-next-day.ps1 -CsvText @'\nBANKNIFTY,2026-04-03,09:15,50000,50010,49990,50005,12345\n...\n'@`

Expected candle row format:

- `SYMBOL,YYYY-MM-DD,HH:MM,open,high,low,close,volume`
- (optional) `YYYY-MM-DD,HH:MM,open,high,low,close,volume` (symbol defaults to BANKNIFTY)

Planner outputs also include:

- swing high/low clusters (proxy SL pools)
- high-volume price clusters (volume-by-price proxy using HLC3 binned into buckets)

## Quick full-day simulation (paste prev day + day CSV)

If you want a deterministic intraday trade log (entry/exit/P&L) using a starter subset of the rulebook constraints:

- `scripts/simulate-day.ps1`

Example (from files):

- `powershell -ExecutionPolicy Bypass -NoProfile -File .\\scripts\\simulate-day.ps1 -PrevCsvPath .\\prev.csv -DayCsvPath .\\day.csv -Quantity 65`

Example (paste text):

- `powershell -ExecutionPolicy Bypass -NoProfile -File .\\scripts\\simulate-day.ps1 -PrevCsvText @'\nBANKNIFTY,2026-04-03,09:15,50000,50010,49990,50005,12345\n...\n'@ -DayCsvText @'\nBANKNIFTY,2026-04-04,09:15,50020,50030,50000,50010,12000\n...\n'@ -Quantity 65`

Notes:

- Default `TargetMode` is `1R` (TP ~= SL distance). Switch to level-based take-profit with `-TargetMode T1`.
- If stop and target touch in the same candle, `AmbiguousFillPolicy` chooses the fill order (default `stop_first`).

## Local LLM (Qwen) for rule extraction (optional)

You can run Qwen locally and still use the same scripts (`extract-rules.ps1`) by pointing the OpenAI client to a local OpenAI-compatible server.

### Ollama (recommended)

1. Install Ollama and pull a model (example):
   - `ollama pull qwen2.5:7b-instruct`
2. Set env vars (in `.env`):
   - `LLM_BASE_URL=http://localhost:11434/v1/`
   - `LLM_API_KEY=ollama`
   - `LLM_RULE_EXTRACT_MODEL=qwen2.5:7b-instruct`

### LM Studio

1. Start the local server in LM Studio (OpenAI-compatible).
2. Set env vars (in `.env`):
   - `LLM_BASE_URL=http://localhost:1234/v1/`
   - `LLM_API_KEY=lmstudio`
   - `LLM_RULE_EXTRACT_MODEL=<the model name shown in LM Studio>`

## 1) Data layer

1. Fetch 1m candles via Dhan historical API (`POST /v2/charts/intraday`) for the required date range.
   - Recommended: cache the fetched data at your end for repeatability and speed.
2. Store transcripts as raw text + metadata:
   - language (Hindi/Hinglish), date, tags

## 2) Market state builder (feature engineering)

At each candle (sequential, no future), compute a state like:

- structure (Dow): HH/HL vs LH/LL (on chosen swing rules)
- key levels: PDH/PDL, last-hour high/low, round levels
- zone: discount/fair/inflated (relative to value/extremes)
- liquidity map: old/new/profit-holder pools (proxy-based)
- opening context: gap type + first candle color
- validity filters: retail participation proxy, event risk flag, overcrowding proxy

Implemented (BankNifty 1m proxies):
- `market_type`: trend/trap/chop using rolling range + reversal count
- `comfort_risk`: obvious breakout hold + volume support (single-instrument proxy)
- `operator_exit_risk`: impulse then stall/grind signature
- `validity_ok`: participation present and not chop

## 3) Transcript ingestion (RAG memory)

When a new transcript arrives:

1. Store raw transcript in DB.
2. Chunk it (paragraph/topic chunks).
3. Create embeddings for each chunk.
4. Save chunks + embeddings to DB (vector search).

## 4) Rule extraction (“learning”)

Run LLM extraction on new transcript chunks:

- output atomic rules: `condition -> interpretation -> action -> tags`
- detect: new vs duplicate vs conflicting with existing rule IDs
- store as **proposal** (not active yet)

Gate:

- human review + backtest is required before merging proposals into canonical rulebook.

Implementation in this repo:

- Ingest transcript: `scripts/ingest-transcript.ps1`
- Extract rule proposals: `scripts/extract-rules.ps1`
- Proposals are stored as `draft` in SQLite (`rule_proposals` table).

## 4.1) Proposal -> merge workflow

1. Export draft proposals to files:
   - `scripts/export-proposals.ps1`
2. Review the exported JSON in `rulebook/proposals/`.
3. Merge one proposal into the canonical rulebook (version bump included):
   - `scripts/merge-proposal.ps1 -ProposalPath rulebook\\proposals\\<proposal_id>.json`
4. Validate:
   - `scripts/validate-rulebook.ps1`
5. Backtest gate (recommended via Web UI):
   - `scripts/run-web.ps1`

## 4.2) “Training” on a new transcript (what you do every day)

This system does **not** fine-tune a model. Your “training” is:

`transcript -> extract rules -> review -> merge -> backtest -> (optional) deploy`

Commands:

1) Ingest transcript (store it as memory):
- `scripts/ingest-transcript.ps1 -File .\\transcripts\\t18.txt -Language hi -Tags "flow,probability" -Embed`

2) Extract a draft proposal (LLM structured output):
- `scripts/extract-rules.ps1 -File .\\transcripts\\t18.txt`

3) Export proposals to files for review:
- `scripts/export-proposals.ps1`

4) Merge one proposal into canonical rulebook (version bump):
- `scripts/merge-proposal.ps1 -ProposalPath rulebook\\proposals\\proposal_XXXX.json -Bump patch`

5) Validate + backtest gate:
- `scripts/validate-rulebook.ps1`
- Run backtest in the Web UI using Dhan historical API and your warmup day logic.

Only after this should you use the updated rulebook for live trading.

## 8) Web UI (settings + backtest)

Run the local UI:

- `scripts/run-web.ps1`

It provides:

- Dhan credential form (stored locally in SQLite)
- Dhan historical API backtest for BankNifty 1m (requires `security_id`, `exchange_segment`, `instrument`)
- Warmup behavior: if you choose `start_date=2026-02-08` and `end_date=2026-02-14`, it trades `2026-02-09..2026-02-14` because the start date is treated as the warmup/levels day.

## 5) Deterministic backtest engine

Loop:

1. read candle
2. update state
3. generate candidates (long/short/wait) using deterministic scoring
4. execute in broker simulator
5. manage SL/targets/partials
6. log trades + metrics

## 5.1) RAG in backtesting (recommended pattern)

- Default: deterministic policy (fast, reproducible)
- Optional: event-driven LLM calls only when the state is ambiguous (WAIT) and an event occurs (e.g., sweep+reclaim).
- Always cache LLM outputs by request hash in SQLite so reruns are deterministic.

## 5.2) Simulation-like per-candle Gemini backtest (gemini_all)

If you want the backtest to behave **like the live simulator** (LLM decision on every 1-minute candle), use:

- `policy=gemini_all`

Notes:

- This calls Gemini on every candle, so it is **slow and non-deterministic** (unless cache hits).
- It is best used for **short date ranges** to validate that your rulebook + state-builder are aligned with your discretionary SL-hunting logic.
- Web UI supports `include_decisions=1` to return per-candle decisions (use a low `max_decisions`).

## 6) Live trading engine (later)

Same as backtest loop, but execution calls Dhan SDK instead of simulator.

## 7) “Unlimited memory” and avoiding slow chat threads

Chat threads are not your memory.

Your memory is:

- SQLite/PG for transcripts + rule proposals + backtest logs
- vector search to retrieve only the relevant chunks

At runtime, send the LLM only:

- current market state (small JSON)
- top-k retrieved rules/snippets (small)
- required output schema
