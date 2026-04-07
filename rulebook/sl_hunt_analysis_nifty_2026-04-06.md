# NIFTY Spot — SL Hunting / Liquidity Analysis (1m)

Data source: `C:\Users\amol charpe\Downloads\nifty_spot_1m_candles (17).csv` (2026-04-06)

## 1) Session stats (Apr 6, 2026)

- Open (first bar 09:16): **22689.30**
- High (HOD 15:12): **22998.35**
- Low (LOD 11:11): **22542.95**
- Close (15:29): **22959.45**
- Range: **455.40** points
- Close position in range: **0.915** (closed near the top)
- Move from LOD to Close: **+416.50** points

Opening range (first ~15 minutes from 09:16):
- ORH: **22714.35**
- ORL: **22591.70**

## 2) What looked like SL hunting (timeline)

This is written in “sweep -> reclaim -> move” terms (rulebook: sweep/reclaim + acceptance > candle-close narratives).

### A) Early two-sided sweep / whipsaw (liquidity farming)

- **09:36–09:37**: ORH break / buy-side grab  
  Price traded up to **22746.90** and then reversed sharply (by **09:41** it printed **22631.05**).  
  Read: early breakout buyers got trapped; liquidity taken above ORH.

- **09:49**: ORL break / sell-side grab  
  Close printed below ORL (**22589.90**) and the drop extended to **22562.00** by **10:00**.  
  Read: after trapping longs, price hunted sell-side liquidity too (classic open whipsaw environment).

### B) “Sell stops first” was the real fuel (capitulation sweep)

- **11:11**: major sell-side sweep (day low)  
  Low printed **22542.95** (prior local low cluster was ~**22571**).  
  Reclaim began quickly (11:14 candle closed **22567.80**).  
  Read: this is the cleanest “stop run then reversal” signature of the day.

### C) Transition / reclaim of key levels (acceptance building)

These are “first time the level held for ~15 minutes” checkpoints:

- **22650** held from **11:50**
- **22714.35 (ORH)** held from **12:47**
- **22800** held from **13:09**
- **22850** held from **13:16**
- **22900** held from **14:30**
- **22950** held from **15:04**

Read: after the 11:11 flush, the day transitioned into a structured up-move with stepwise acceptance.

### D) Round-number / obvious-level stop engineering

Round-number-adjacent sweep+reclaim events (30-min reference, reclaim within minutes):

- **11:44**: sweep toward **22650** (22650.60)
- **12:34**: sweep below **22650** (22648.35) then reclaim
- **13:22–13:36**: repeated probing around **22900** (22898.95 → 22900.55) with snapbacks
- **15:12**: run toward **23000** (22998.35) and then back off (close stayed below 23000)

Read: 22650 and 22900 acted like “magnets” where stops cluster; 23000 remained an obvious buy-side pool into the close.

## 3) Levels to mark for Apr 7, 2026 (from Apr 6)

These are not “predictions”; they’re **liquidity + decision levels** to watch for sweep/reclaim/acceptance behavior.

### Primary (previous-day anchors)

- **PDH:** 22998.35
- **PDL:** 22542.95
- **PDC:** 22959.45
- **PDM (mid):** 22770.65

### Intraday magnets / decision levels

- **23000** (psych + obvious buy-side liquidity above PDH)
- **22950** (late acceptance zone)
- **22900** (micro-sweeps + later held support)
- **22850 / 22800** (stepwise acceptance pivots)
- **22714.35** (ORH from Apr 6; important reclaim/acceptance pivot)
- **22650** (round-number trap / both-side sweeps)
- **22591.70** (ORL from Apr 6)
- **22570–22560** (morning base / repeated tests)

## 4) How to use these levels on Apr 7 (rulebook-style)

- If open trades near **PDH/23000**: expect “precision” stop-hunts; wait for **strong acceptance** (not a 1–2 point reclaim) before following (rulebook: strong acceptance / micro reaction).
- If price sweeps above **22998–23000** and reclaims back below: treat as **buy-side trap** candidate; next liquidity often shifts back toward **22950/22900**.
- If early weakness sweeps under **22900** and reclaims quickly: treat as **sell-side trap** candidate in an otherwise strong previous-day close context.
- If price loses **22714** and accepts below it: expect deeper liquidity search toward **22650**, then possibly **22592**.

