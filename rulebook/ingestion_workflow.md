# Transcript -> Rulebook update workflow (RAG-style)

Goal: when a new (Hindi/Hinglish) transcript arrives, **extract knowledge** and propose **rulebook diffs** safely (no silent changes in live trading).

## Safety model (recommended)

- **Never** auto-activate new rules for live trading.
- New knowledge enters as `draft`, then becomes `active` only after review + backtest.

## Suggested pipeline

1. **Store transcript**
   - Save raw text with metadata: date, speaker, market/instrument context.
   - Keep original language; optionally store an English translation alongside it.

2. **Chunk + index**
   - Chunk by topic boundaries (opening logic, gap, SL, risk).
   - Embed chunks and index in a vector store.

3. **Retrieve**
   - For a user query (or nightly “rulebook improvement”), retrieve top-k chunks by similarity.

4. **Extract candidates**
   - Convert retrieved chunks into atomic rules using the format:
     - `condition` -> `interpretation` -> `action`
   - add `tags`, and note whether it is a **new rule** or **modifies an existing rule**
   - prefer measurable proxies for subjective ideas (e.g., operator entry as turning-point + momentum shift; instability as repeated whipsaw cycles)

5. **De-duplicate + conflict detection**
   - Detect same-meaning rules (merge).
   - Detect contradictions (create `integration_conflicts` notes).

6. **Propose a versioned update**
   - Create a new version (e.g., `2.0.1`) with a clear changelog entry:
     - added rules
     - modified rules
     - deprecated rules

7. **Gate with tests**
   - Backtest the updated rulebook against historical data.
   - Only after performance + risk checks: mark rules `active`.

## “Improve rulebook” meaning (operational)

When transcripts add information, treat improvements as one of:

- **Add rule**: new filter/engine (e.g., gap+candle trap matrix).
- **Refine rule**: add timing/path uncertainty to SL hunting.
- **Add kill-switch**: stop trading on mismatch days.
- **Add measurement**: define what the engine must compute (gap type, first candle color, last 30m momentum).

## Bilingual notes (Hindi/Hinglish)

- Keep the **source transcript** unchanged.
- Store extracted rules in a consistent language (recommended: English technical + allow Hindi keywords as tags).
- Put Hindi “market phrases” into `tags` so retrieval works even with Hindi queries.
