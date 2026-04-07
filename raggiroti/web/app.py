from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from datetime import timedelta
from functools import partial
from pathlib import Path
from zoneinfo import ZoneInfo

import anyio
from fastapi import FastAPI, Form, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import Body

from raggiroti.backtest.day_split import group_by_date
from raggiroti.backtest.engine import run_backtest
from raggiroti.backtest.prev_day_planner import classify_open_scenario, compute_prev_day_levels
from raggiroti.backtest.scenario_policy import ScenarioPolicy
from raggiroti.backtest.rulebook_scoring_policy import RulebookScoringPolicy
from raggiroti.backtest.rag_policy import RAGPolicy
from raggiroti.backtest.rag_policy_per_candle import PerCandleRAGPolicy
from raggiroti.backtest.gemini_policy_per_candle import PerCandleGeminiPolicy
from raggiroti.config import get_settings
from raggiroti.dhan.session import DhanUnavailable, create_dhan_client
from raggiroti.dhan.historical import DhanIntradayRequest, fetch_intraday_candles
from raggiroti.storage.sqlite_db import SqliteStore, Transcript
from raggiroti.llm.gemini_rule_extractor import GeminiRuleExtractor
from raggiroti.rules.rulebook_merge import merge_rule_proposal_into_rulebook
from raggiroti.rules.rulebook_loader import load_rulebook
from raggiroti.dhan.live_feed import DhanLiveFeed, LiveFeedInstrument, parse_marketfeed_tick
from raggiroti.dhan.option_chain import DhanOptionChainClient, summarize_oi_walls
from raggiroti.live.candle_builder import CandleBuilder1m
from raggiroti.live.models import Tick
from raggiroti.live.live_sim_engine import LiveSimEngine
from raggiroti.predict.next_day_predictor import NextDayPredictor
from raggiroti.live.models import LiveCandle


APP_TITLE = "raggiroti Backtest"


@dataclass
class BacktestJob:
    id: str
    status: str  # queued|running|done|error
    created_at: str
    updated_at: str
    result: dict | None = None
    error: str | None = None


BACKTEST_JOBS: dict[str, BacktestJob] = {}


@dataclass
class PredictJob:
    id: str
    status: str  # queued|running|done|error
    created_at: str
    updated_at: str
    result: dict | None = None
    error: str | None = None


PREDICT_JOBS: dict[str, PredictJob] = {}

LIVE_LOCK = threading.Lock()
LIVE_THREAD: threading.Thread | None = None
LIVE_OI_THREAD: threading.Thread | None = None
LIVE_STOP = threading.Event()
LIVE_ENGINE: LiveSimEngine | None = None
LIVE_CANDLE_BUILDER: CandleBuilder1m | None = None
LIVE_LOOP: asyncio.AbstractEventLoop | None = None
LIVE_OI_SNAPSHOT: dict | None = None
LIVE_LAST_ERROR: str | None = None

SIM_ENGINE: LiveSimEngine | None = None


def _ensure_gemini_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gemini_settings (
          id INTEGER PRIMARY KEY CHECK (id=1),
          updated_at TEXT NOT NULL,
          api_key TEXT,
          model TEXT,
          decision_model TEXT,
          extract_model TEXT
        );
        """
    )
    # Lightweight migration for older DBs (SQLite keeps existing table definition).
    cols = [r[1] for r in conn.execute("PRAGMA table_info(gemini_settings)").fetchall()]
    if "decision_model" not in cols:
        try:
            conn.execute("ALTER TABLE gemini_settings ADD COLUMN decision_model TEXT;")
        except Exception:
            pass
    if "extract_model" not in cols:
        try:
            conn.execute("ALTER TABLE gemini_settings ADD COLUMN extract_model TEXT;")
        except Exception:
            pass
    conn.commit()


def _set_gemini_settings(db_path: str, api_key: str, decision_model: str, extract_model: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_gemini_table(conn)
    conn.execute(
        "INSERT OR REPLACE INTO gemini_settings(id, updated_at, api_key, model, decision_model, extract_model) VALUES(1,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), api_key, decision_model, decision_model, extract_model),
    )
    conn.commit()
    conn.close()


def _get_gemini_settings(db_path: str) -> dict:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_gemini_table(conn)
    cur = conn.execute("SELECT updated_at, api_key, model, decision_model, extract_model FROM gemini_settings WHERE id=1")
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"updated_at": None, "api_key_masked": None, "model": None, "decision_model": None, "extract_model": None}
    k = row[1]
    masked = None
    if k:
        masked = ("*" * max(0, len(k) - 4)) + k[-4:]
    model = row[2]
    return {
        "updated_at": row[0],
        "api_key_masked": masked,
        "model": model,
        "decision_model": row[3] or model,
        "extract_model": row[4] or model,
    }


def _get_gemini_api_key_raw(db_path: str) -> str | None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_gemini_table(conn)
    cur = conn.execute("SELECT api_key FROM gemini_settings WHERE id=1")
    row = cur.fetchone()
    conn.close()
    return None if not row else row[0]


def _get_gemini_decision_model_raw(db_path: str) -> str | None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_gemini_table(conn)
    cur = conn.execute("SELECT decision_model, model FROM gemini_settings WHERE id=1")
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return row[0] or row[1]


def _get_gemini_extract_model_raw(db_path: str) -> str | None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_gemini_table(conn)
    cur = conn.execute("SELECT extract_model, model FROM gemini_settings WHERE id=1")
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return row[0] or row[1]


# Back-compat: older callers treat "model" as decision model.
def _get_gemini_model_raw(db_path: str) -> str | None:
    return _get_gemini_decision_model_raw(db_path)


def _ensure_settings_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dhan_settings (
          id INTEGER PRIMARY KEY CHECK (id=1),
          updated_at TEXT NOT NULL,
          client_id TEXT,
          access_token TEXT
        );
        """
    )
    conn.commit()


def _set_dhan_settings(db_path: str, client_id: str, access_token: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_settings_table(conn)
    conn.execute(
        "INSERT OR REPLACE INTO dhan_settings(id, updated_at, client_id, access_token) VALUES(1,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), client_id, access_token),
    )
    conn.commit()
    conn.close()


def _get_dhan_settings(db_path: str) -> dict:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_settings_table(conn)
    cur = conn.execute("SELECT updated_at, client_id, access_token FROM dhan_settings WHERE id=1")
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"updated_at": None, "client_id": None, "access_token": None}
    token = row[2]
    masked = None
    if token:
        masked = ("*" * max(0, len(token) - 4)) + token[-4:]
    return {"updated_at": row[0], "client_id": row[1], "access_token_masked": masked}


def _get_dhan_access_token_raw(db_path: str) -> str | None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_settings_table(conn)
    cur = conn.execute("SELECT access_token FROM dhan_settings WHERE id=1")
    row = cur.fetchone()
    conn.close()
    return None if not row else row[0]


def _get_dhan_client_id_raw(db_path: str) -> str | None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_settings_table(conn)
    cur = conn.execute("SELECT client_id FROM dhan_settings WHERE id=1")
    row = cur.fetchone()
    conn.close()
    return None if not row else row[0]


app = FastAPI(title=APP_TITLE)


@app.on_event("startup")
async def _startup() -> None:
    global LIVE_LOOP
    try:
        LIVE_LOOP = asyncio.get_running_loop()
    except Exception:
        LIVE_LOOP = None


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    # Ensure frontend always receives JSON (avoids "Unexpected token 'I' Internal Server Error").
    return JSONResponse({"ok": False, "error": str(exc), "type": exc.__class__.__name__}, status_code=500)


def _compute_backtest_range_dhan(
    *,
    security_id: str,
    exchange_segment: str,
    instrument: str,
    start_date: str,
    end_date: str,
    interval: str,
    qty: int,
    gap_up: float,
    gap_down: float,
    flat: float,
    policy: str,
    include_fills: int,
    max_fills: int,
    include_decisions: int = 0,
    max_decisions: int = 300,
    max_entries_per_day: int | None = None,
    cooldown_after_sl_candles: int = 0,
    lock_direction_after_first_entry: int = 0,
) -> dict:
    t0 = datetime.now(timezone.utc)
    settings = get_settings()
    token = _get_dhan_access_token_raw(settings.db_path)
    if not token:
        return {"ok": False, "error": "missing dhan access token; save it on home page first"}

    from_dt = datetime.strptime(start_date + " 09:15:00", "%Y-%m-%d %H:%M:%S")
    to_dt = datetime.strptime(end_date + " 15:31:00", "%Y-%m-%d %H:%M:%S")

    req = DhanIntradayRequest(
        security_id=security_id,
        exchange_segment=exchange_segment,
        instrument=instrument,
        interval=interval,
        oi=False,
        from_dt=from_dt,
        to_dt=to_dt,
    )
    try:
        candles = fetch_intraday_candles(req, access_token=token)
    except Exception as e:
        return {"ok": False, "error": f"fetch failed: {e}"}

    by_date = group_by_date(candles)
    dates = sorted(by_date.keys())
    if start_date not in by_date:
        return {"ok": False, "error": f"start_date not found in fetched candles: {start_date}"}
    if end_date not in by_date:
        return {"ok": False, "error": f"end_date not found in fetched candles: {end_date}"}

    s_idx = dates.index(start_date) + 1  # warmup day is start_date
    e_idx = dates.index(end_date)
    if s_idx > e_idx:
        return {"ok": False, "error": "invalid range: trades start after end"}

    daily = []
    total = 0.0
    total_candles = 0
    llm_meta = None
    if policy in {"rag", "rag_all", "gemini_all"}:
        if policy == "gemini_all":
            gemini_key = _get_gemini_api_key_raw(settings.db_path)
            gemini_model = _get_gemini_decision_model_raw(settings.db_path) or "gemini-2.5-flash"
            llm_meta = {
                "policy": policy,
                "per_candle_llm": True,
                "provider": "gemini",
                "model": gemini_model,
                "note": "gemini_all calls Gemini on every candle (slow/expensive). Use rulebook for deterministic backtests.",
                "gemini_configured": bool(gemini_key),
            }
        else:
            llm_meta = {
                "policy": policy,
                "per_candle_llm": policy == "rag_all",
                "provider": "openai_compatible",
                "base_url": settings.llm_base_url,
                "model": settings.openai_rule_extract_model,
                "note": (
                    "rag_all calls the LLM on every candle (slow). "
                    "Use policy=rag for event-driven LLM calls (faster) or policy=rulebook for fully deterministic."
                ),
            }

    for i in range(s_idx, e_idx + 1):
        prev_date = dates[i - 1]
        date = dates[i]
        prev_levels = compute_prev_day_levels(by_date[prev_date])
        day_candles = sorted(by_date[date], key=lambda c: c.dt)
        total_candles += len(day_candles)
        scenario = classify_open_scenario(
            day_candles[0].open,
            prev_levels.close,
            gap_up_threshold_points=gap_up,
            gap_down_threshold_points=gap_down,
            flat_threshold_points=flat,
        )

        if policy == "gemini_all":
            gemini_key = _get_gemini_api_key_raw(settings.db_path)
            gemini_model = _get_gemini_decision_model_raw(settings.db_path) or "gemini-2.5-flash"
            if not gemini_key:
                return {"ok": False, "error": "missing gemini api key (required for gemini_all)"}
            pol = PerCandleGeminiPolicy(
                api_key=gemini_key,
                model=gemini_model,
                db_path=settings.db_path,
                rulebook_path=settings.rulebook_path,
            )
        elif policy == "rag_all":
            pol = PerCandleRAGPolicy()
        elif policy == "rag":
            pol = RAGPolicy()
        elif policy == "rulebook":
            pol = RulebookScoringPolicy()
        else:
            pol = ScenarioPolicy(prev=prev_levels, scenario=scenario)

        res = run_backtest(
            day_candles,
            policy=pol,
            qty=qty,
            prev=prev_levels,
            gap_up_threshold_points=gap_up,
            gap_down_threshold_points=gap_down,
            flat_threshold_points=flat,
            include_decisions=bool(int(include_decisions)) or (policy in {"gemini_all", "rag_all"}),
            max_decisions=int(max(50, int(max_decisions))),
            max_entries_per_day=None if max_entries_per_day is None else int(max_entries_per_day),
            cooldown_after_sl_candles=int(cooldown_after_sl_candles),
            lock_direction_after_first_entry=bool(int(lock_direction_after_first_entry)),
        )
        total += res.realized_pnl_points
        item = {"date": date, "prev_date": prev_date, "scenario": scenario, "pnl_points_x_qty": res.realized_pnl_points}
        if include_fills:
            item["fills"] = [f.__dict__ for f in res.fills[: max(0, int(max_fills))]]
            item["fills_truncated"] = len(res.fills) > int(max_fills)
        if bool(int(include_decisions)) and res.decisions is not None:
            item["decisions"] = res.decisions[: max(0, int(max_decisions))]
            item["decisions_truncated"] = len(res.decisions) > int(max_decisions)
        daily.append(item)

    elapsed_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
    return {
        "ok": True,
        "policy": policy,
        "security_id": security_id,
        "exchange_segment": exchange_segment,
        "instrument": instrument,
        "interval": interval,
        "start_date": start_date,
        "end_date": end_date,
        "elapsed_ms": elapsed_ms,
        "total_candles_processed": total_candles,
        "llm": llm_meta,
        "trade_days": daily,
        "total_pnl_points_x_qty": total,
    }


@app.get("/api/llm/status")
def llm_status() -> JSONResponse:
    settings = get_settings()
    using_local = bool(settings.llm_base_url)
    model = settings.openai_rule_extract_model
    return JSONResponse(
        {
            "ok": True,
            "using_local_server": using_local,
            "base_url": settings.llm_base_url,
            "model": model,
            "openai_key_present": bool(settings.openai_api_key),
            "note": (
                "If using_local_server=true, requests go to your local OpenAI-compatible endpoint "
                "(Ollama/LM Studio). Otherwise it uses OpenAI."
            ),
        }
    )


@app.get("/api/llm/cache/stats")
def llm_cache_stats() -> JSONResponse:
    settings = get_settings()
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    try:
        cur = conn.execute("SELECT model, COUNT(*) FROM llm_cache GROUP BY model ORDER BY COUNT(*) DESC")
        rows = [{"model": r[0], "count": int(r[1])} for r in cur.fetchall()]
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"query failed: {e}"}, status_code=400)
    finally:
        conn.close()
    return JSONResponse({"ok": True, "db_path": settings.db_path, "rows": rows})


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>raggiroti Backtest</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, Arial; margin: 28px; }
      .card { max-width: 880px; padding: 16px 18px; border: 1px solid #ddd; border-radius: 10px; margin-bottom: 18px; }
      label { display:block; font-size: 12px; opacity: .8; margin-top: 10px; }
      input, button { padding: 10px; width: 100%; max-width: 520px; }
      button { width: auto; cursor: pointer; }
      .row { display:flex; gap: 18px; flex-wrap: wrap; align-items: end; }
      .muted { font-size: 12px; opacity: .75; }
      pre { background: #f7f7f7; padding: 12px; border-radius: 8px; overflow:auto; }
    </style>
  </head>
  <body>
    <h1>raggiroti Backtesting</h1>
    <div class="card">
      <h3>Dhan Settings (for future live trading)</h3>
      <form method="post" action="/api/settings/dhan">
        <label>Client ID</label>
        <input name="client_id" placeholder="your_client_id" required />
        <label>Access Token</label>
        <input name="access_token" placeholder="your_access_token" required />
        <div style="margin-top:12px;">
          <button type="submit">Save</button>
        </div>
        <p class="muted">Saved locally in SQLite. Do not use this on a shared machine.</p>
      </form>
      <div style="margin-top:10px;">
        <button onclick="loadSettings()">Show saved settings</button>
        <button onclick="testSession()">Test Dhan session</button>
        <button onclick="loadLLM()">Show LLM status</button>
        <button onclick="loadLLMCache()">Show LLM cache stats</button>
        <pre id="settings"></pre>
      </div>
    </div>

    <div class="card">
      <h3>Gemini Settings (per-candle LLM)</h3>
      <form method="post" action="/api/settings/gemini">
        <label>Gemini API Key</label>
        <input name="api_key" placeholder="AIza..." required />
        <label>Decision model (per 1m candle)</label>
        <input name="decision_model" value="gemini-2.5-flash" />
        <label>Extraction model (transcript → rules)</label>
        <input name="extract_model" value="gemini-2.5-pro" />
        <div style="margin-top:12px;">
          <button type="submit">Save</button>
        </div>
        <p class="muted">Used for per-1m candle decisions in live simulation. Saved locally in SQLite.</p>
      </form>
      <div style="margin-top:10px;">
        <button onclick="loadGemini()">Show saved Gemini settings</button>
        <button onclick="listGeminiModels()">List available models</button>
        <pre id="gemini"></pre>
      </div>
    </div>

    <div class="card">
      <h3>Live Simulation (Paper Trades Only)</h3>
      <form id="livef">
        <label>Symbol</label>
        <input name="symbol" value="NIFTY" />
        <label>Quantity</label>
        <input name="qty" type="number" value="65" />
        <label>Seed previous trading day (Dhan)</label>
        <input name="seed_prev_day" type="number" value="1" />
        <div style="margin-top:12px;">
          <button type="button" onclick="liveStart()">Start Live</button>
          <button type="button" onclick="liveStop()">Stop</button>
          <button type="button" onclick="liveStatus()">Status</button>
          <button type="button" onclick="liveState()">State</button>
          <button type="button" onclick="liveLevels()">Levels</button>
          <button type="button" onclick="liveCandles()">Candles</button>
          <button type="button" onclick="liveDecisions()">Decisions</button>
          <button type="button" onclick="liveFills()">Fills</button>
        </div>
        <p class="muted">Connects to Dhan MarketFeed websocket and builds 1-minute OHLC. No real orders are placed.</p>
      </form>
      <pre id="liveout"></pre>
    </div>

    <div class="card">
      <h3>Prediction (Next-Day Levels)</h3>
      <form id="predf">
        <label>Symbol</label>
        <input name="symbol" value="NIFTY" />
        <label>Training start date (YYYY-MM-DD)</label>
        <input name="training_start_date" placeholder="2026-04-06" required />
        <label>Target date to predict (YYYY-MM-DD)</label>
        <input name="target_date" placeholder="2026-04-07" required />
        <label>Use previous trading day OI snapshot (0/1)</label>
        <input name="use_prev_day_oi" value="1" />
        <div style="margin-top:10px;">
          <button type="button" onclick="oiCapture()">Capture OI snapshot NOW</button>
          <button type="button" onclick="oiLatest()">Show latest OI snapshot</button>
        </div>
        <div style="margin-top:12px;">
          <button type="button" onclick="predSubmit()">Predict</button>
          <button type="button" onclick="predGet()">Get result</button>
        </div>
        <label>Prediction job id</label>
        <input id="pred_job_id" placeholder="pr_..." />
        <p class="muted">Uses previous days (from start to previous trading day) and predicts levels for the target day with Gemini + rulebook + optional OI.</p>
      </form>
      <pre id="predout"></pre>
    </div>

    <div class="card">
      <h3>Rulebook Learning (Transcript -&gt; Proposal -&gt; Approve -&gt; Merge)</h3>
      <div class="row">
        <div>
          <label>Transcript tags (comma separated)</label>
          <input id="tr_tags" placeholder="sl_hunting, options, trap" />
        </div>
        <div>
          <label>Extract from transcript_id</label>
          <input id="extract_tid" placeholder="tr_..." />
        </div>
        <div>
          <label>Proposal id</label>
          <input id="proposal_id" placeholder="rp_..." />
        </div>
      </div>
      <label>Transcript content</label>
      <textarea id="tr_content" style="width:100%; max-width:880px; height:140px; padding:10px;"></textarea>
      <div style="margin-top:12px;">
        <button type="button" onclick="ingestTranscript()">Ingest transcript</button>
        <button type="button" onclick="listTranscripts()">List transcripts</button>
        <button type="button" onclick="extractProposal()">Extract proposal (Gemini)</button>
        <button type="button" onclick="listProposals()">List proposals</button>
        <button type="button" onclick="getProposal()">Get proposal</button>
        <button type="button" onclick="approveProposal()">Approve</button>
        <button type="button" onclick="rejectProposal()">Reject</button>
        <button type="button" onclick="mergeProposal()">Merge into rulebook</button>
        <button type="button" onclick="reindexRag()">RAG reindex</button>
      </div>
      <p class="muted">This writes to SQLite + updates `rulebook/nexus_ultra_v2.rulebook.json`. Always approve before merge.</p>
      <pre id="learnout"></pre>
    </div>

    <div class="card">
      <h3>Backtest (Dhan historical API; start day is warmup)</h3>
      <form id="btd">
        <label>Security ID (BankNifty)</label>
        <input name="security_id" placeholder="(paste BankNifty securityId from Dhan instrument list)" required />
        <label>Exchange segment</label>
        <input name="exchange_segment" value="IDX_I" />
        <label>Instrument</label>
        <input name="instrument" value="INDEX" />
        <label>Start date (warmup; YYYY-MM-DD)</label>
        <input name="start_date" placeholder="2026-02-08" required />
        <label>End date (inclusive; YYYY-MM-DD)</label>
        <input name="end_date" placeholder="2026-02-14" required />
        <label>Interval (minutes)</label>
        <input name="interval" value="1" />
        <label>Quantity</label>
        <input name="qty" type="number" value="65" />
        <label>Gap up threshold points</label>
        <input name="gap_up" type="number" value="30" />
        <label>Gap down threshold points</label>
        <input name="gap_down" type="number" value="30" />
        <label>Flat threshold points</label>
        <input name="flat" type="number" value="15" />
        <label>Policy (rulebook | rag | rag_all | gemini_all | scenario)</label>
        <input name="policy" value="rulebook" />
        <p class="muted">Use <code>gemini_all</code> for simulation-like per-candle Gemini decisions (slow/expensive).</p>
        <label>Max entries per day (quality filter)</label>
        <input name="max_entries_per_day" type="number" value="2" />
        <label>Cooldown after SL (candles)</label>
        <input name="cooldown_after_sl_candles" type="number" value="10" />
        <label>Lock direction after first entry (0/1)</label>
        <input name="lock_direction_after_first_entry" value="1" />
        <label>Include fills (0/1)</label>
        <input name="include_fills" value="0" />
        <label>Max fills per day</label>
        <input name="max_fills" value="200" />
        <label>Include decisions (0/1)</label>
        <input name="include_decisions" value="0" />
        <label>Max decisions per day</label>
        <input name="max_decisions" value="300" />
        <div style="margin-top:12px;">
          <button type="button" onclick="runBacktestDhan()">Run Dhan backtest</button>
        </div>
        <p class="muted">Uses Dhan `POST /v2/charts/intraday` to fetch 1m candles. Start date is used only to build next-day levels; trades run from next day.</p>
      </form>
      <pre id="outd"></pre>
    </div>

    <script>
      async function loadSettings() {
        const r = await fetch('/api/settings/dhan');
        const j = await r.json();
        document.getElementById('settings').textContent = JSON.stringify(j, null, 2);
      }
      async function loadGemini() {
        const r = await fetch('/api/settings/gemini');
        const j = await r.json();
        document.getElementById('gemini').textContent = JSON.stringify(j, null, 2);
      }
      async function listGeminiModels() {
        const r = await fetch('/api/gemini/models');
        const t = await r.text();
        let j = null;
        try { j = JSON.parse(t); } catch (e) {}
        document.getElementById('gemini').textContent = (j ? JSON.stringify(j, null, 2) : t);
      }
      async function loadLLM() {
        const r = await fetch('/api/llm/status');
        const j = await r.json();
        document.getElementById('settings').textContent = JSON.stringify(j, null, 2);
      }
      async function loadLLMCache() {
        const r = await fetch('/api/llm/cache/stats');
        const j = await r.json();
        document.getElementById('settings').textContent = JSON.stringify(j, null, 2);
      }
      async function liveStart() {
        const fd = new FormData(document.getElementById('livef'));
        const r = await fetch('/api/live/start', { method: 'POST', body: fd });
        const j = await r.json();
        document.getElementById('liveout').textContent = JSON.stringify(j, null, 2);
      }
      async function liveStop() {
        const r = await fetch('/api/live/stop', { method: 'POST' });
        const j = await r.json();
        document.getElementById('liveout').textContent = JSON.stringify(j, null, 2);
      }
      async function liveStatus() {
        const r = await fetch('/api/live/status');
        const j = await r.json();
        document.getElementById('liveout').textContent = JSON.stringify(j, null, 2);
      }
      async function liveState() {
        const r = await fetch('/api/live/state');
        const j = await r.json();
        document.getElementById('liveout').textContent = JSON.stringify(j, null, 2);
      }
      async function liveLevels() {
        const r = await fetch('/api/live/levels');
        const j = await r.json();
        document.getElementById('liveout').textContent = JSON.stringify(j, null, 2);
      }
      async function liveCandles() {
        const r = await fetch('/api/live/candles?limit=80');
        const j = await r.json();
        document.getElementById('liveout').textContent = JSON.stringify(j, null, 2);
      }
      async function liveDecisions() {
        const r = await fetch('/api/live/decisions?limit=80');
        const j = await r.json();
        document.getElementById('liveout').textContent = JSON.stringify(j, null, 2);
      }
      async function liveFills() {
        const r = await fetch('/api/live/fills');
        const j = await r.json();
        document.getElementById('liveout').textContent = JSON.stringify(j, null, 2);
      }

      async function ingestTranscript() {
        const fd = new FormData();
        fd.set('content', document.getElementById('tr_content').value);
        fd.set('language', 'auto');
        fd.set('tags', document.getElementById('tr_tags').value || '');
        const r = await fetch('/api/transcripts', { method: 'POST', body: fd });
        const j = await r.json();
        document.getElementById('learnout').textContent = JSON.stringify(j, null, 2);
        if (j.transcript_id) document.getElementById('extract_tid').value = j.transcript_id;
      }
      async function listTranscripts() {
        const r = await fetch('/api/transcripts?limit=20');
        const j = await r.json();
        document.getElementById('learnout').textContent = JSON.stringify(j, null, 2);
      }
      async function extractProposal() {
        const tid = document.getElementById('extract_tid').value;
        const fd = new FormData();
        fd.set('transcript_id', tid);
        const r = await fetch('/api/rule_proposals/extract', { method: 'POST', body: fd });
        const j = await r.json();
        document.getElementById('learnout').textContent = JSON.stringify(j, null, 2);
        if (j.proposal_id) document.getElementById('proposal_id').value = j.proposal_id;
      }
      async function listProposals() {
        const r = await fetch('/api/rule_proposals?limit=20');
        const j = await r.json();
        document.getElementById('learnout').textContent = JSON.stringify(j, null, 2);
      }
      async function getProposal() {
        const pid = document.getElementById('proposal_id').value;
        const r = await fetch(`/api/rule_proposals/${pid}`);
        const j = await r.json();
        document.getElementById('learnout').textContent = JSON.stringify(j, null, 2);
      }
      async function approveProposal() {
        const pid = document.getElementById('proposal_id').value;
        const fd = new FormData();
        fd.set('status', 'approved');
        const r = await fetch(`/api/rule_proposals/${pid}/status`, { method: 'POST', body: fd });
        const j = await r.json();
        document.getElementById('learnout').textContent = JSON.stringify(j, null, 2);
      }
      async function rejectProposal() {
        const pid = document.getElementById('proposal_id').value;
        const fd = new FormData();
        fd.set('status', 'rejected');
        const r = await fetch(`/api/rule_proposals/${pid}/status`, { method: 'POST', body: fd });
        const j = await r.json();
        document.getElementById('learnout').textContent = JSON.stringify(j, null, 2);
      }
      async function mergeProposal() {
        const pid = document.getElementById('proposal_id').value;
        const r = await fetch(`/api/rule_proposals/${pid}/merge`, { method: 'POST' });
        const j = await r.json();
        document.getElementById('learnout').textContent = JSON.stringify(j, null, 2);
      }
      async function reindexRag() {
        const r = await fetch('/api/rag/reindex', { method: 'POST' });
        const j = await r.json();
        document.getElementById('learnout').textContent = JSON.stringify(j, null, 2);
      }
      async function testSession() {
        const r = await fetch('/api/dhan/session/test', { method: 'POST' });
        const j = await r.json();
        document.getElementById('settings').textContent = JSON.stringify(j, null, 2);
      }
      async function runBacktestDhan() {
        const form = document.getElementById('btd');
        const fd = new FormData(form);
        const policy = (fd.get('policy') || 'rulebook').toString();
        document.getElementById('outd').textContent =
          `Running backtest (policy=${policy})...` +
          ((policy === 'rag_all' || policy === 'gemini_all')
            ? `\\nNote: ${policy} calls the LLM on every 1-minute candle and can take a long time.`
            : '');
        try {
          // Submit as a background job so long runs don't die due to network/proxy timeouts.
          const sub = await fetch('/api/backtest/range/dhan/submit', { method: 'POST', body: fd });
          const subText = await sub.text();
          const subJson = JSON.parse(subText);
          if (!subJson.ok) {
            document.getElementById('outd').textContent = JSON.stringify(subJson, null, 2);
            return;
          }
          const jobId = subJson.job_id;
          document.getElementById('outd').textContent = `Job submitted: ${jobId}\\nWaiting...`;

          while (true) {
            await new Promise(r => setTimeout(r, 1000));
            const jr = await fetch(`/api/backtest/job/${jobId}`);
            const jt = await jr.text();
            let jj = null;
            try { jj = JSON.parse(jt); } catch (e) {}
            if (jj === null) {
              document.getElementById('outd').textContent = `Job poll failed: HTTP ${jr.status}\\n\\n${jt}`;
              return;
            }
            if (!jj.ok) {
              document.getElementById('outd').textContent = JSON.stringify(jj, null, 2);
              return;
            }
            if (jj.job.status === 'done' || jj.job.status === 'error') {
              document.getElementById('outd').textContent = JSON.stringify(jj.job, null, 2);
              return;
            }
            document.getElementById('outd').textContent = `Job ${jobId}: ${jj.job.status}...`;
          }
        } catch (e) {
          document.getElementById('outd').textContent = `Request failed: ${e}`;
        }
      }

      async function predSubmit() {
        const fd = new FormData(document.getElementById('predf'));
        const r = await fetch('/api/predict/next_day/submit', { method: 'POST', body: fd });
        const t = await r.text();
        let j = null;
        try { j = JSON.parse(t); } catch (e) {}
        document.getElementById('predout').textContent = (j ? JSON.stringify(j, null, 2) : t);
        if (j && j.job_id) document.getElementById('pred_job_id').value = j.job_id;
      }

      async function predGet() {
        const id = document.getElementById('pred_job_id').value;
        const r = await fetch(`/api/predict/job/${id}`);
        const t = await r.text();
        let j = null;
        try { j = JSON.parse(t); } catch (e) {}
        document.getElementById('predout').textContent = (j ? JSON.stringify(j, null, 2) : t);
      }

      async function oiCapture() {
        const fd = new FormData(document.getElementById('predf'));
        const symbol = fd.get('symbol') || 'NIFTY';
        const form = new FormData();
        form.set('symbol', symbol.toString());
        const r = await fetch('/api/oi/snapshot/capture', { method: 'POST', body: form });
        const t = await r.text();
        let j = null;
        try { j = JSON.parse(t); } catch (e) {}
        document.getElementById('predout').textContent = (j ? JSON.stringify(j, null, 2) : t);
      }

      async function oiLatest() {
        const fd = new FormData(document.getElementById('predf'));
        const symbol = fd.get('symbol') || 'NIFTY';
        const r = await fetch(`/api/oi/snapshot/latest?symbol=${encodeURIComponent(symbol.toString())}`);
        const t = await r.text();
        let j = null;
        try { j = JSON.parse(t); } catch (e) {}
        document.getElementById('predout').textContent = (j ? JSON.stringify(j, null, 2) : t);
      }
    </script>
  </body>
</html>
""".strip()


@app.get("/api/settings/dhan")
def get_dhan_settings() -> JSONResponse:
    settings = get_settings()
    return JSONResponse(_get_dhan_settings(settings.db_path))


@app.post("/api/settings/dhan")
def set_dhan_settings(client_id: str = Form(...), access_token: str = Form(...)) -> JSONResponse:
    settings = get_settings()
    _set_dhan_settings(settings.db_path, client_id=client_id, access_token=access_token)
    return JSONResponse({"ok": True})


@app.post("/api/dhan/session/test")
def dhan_test_session() -> JSONResponse:
    settings = get_settings()
    client_id = _get_dhan_client_id_raw(settings.db_path)
    access_token = _get_dhan_access_token_raw(settings.db_path)
    if not client_id or not access_token:
        return JSONResponse({"ok": False, "error": "missing dhan settings"}, status_code=400)
    try:
        _ = create_dhan_client(client_id, access_token)
        return JSONResponse({"ok": True, "note": "Client created. Call an API method to fully verify connectivity."})
    except DhanUnavailable as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.get("/api/settings/gemini")
def get_gemini_settings() -> JSONResponse:
    settings = get_settings()
    return JSONResponse(_get_gemini_settings(settings.db_path))


@app.post("/api/settings/gemini")
def set_gemini_settings(
    api_key: str = Form(...),
    decision_model: str = Form("gemini-2.0-flash"),
    extract_model: str = Form("gemini-2.0-pro"),
    model: str = Form(""),
) -> JSONResponse:
    settings = get_settings()
    # Back-compat: if callers only send `model`, apply it to both.
    if model and (decision_model == "gemini-2.0-flash" and extract_model == "gemini-2.0-pro"):
        decision_model = model
        extract_model = model
    if not extract_model:
        extract_model = decision_model
    _set_gemini_settings(settings.db_path, api_key=api_key, decision_model=decision_model, extract_model=extract_model)
    return JSONResponse({"ok": True})


@app.get("/api/gemini/models")
def gemini_list_models() -> JSONResponse:
    """
    Debug helper for 404 issues:
    returns the models visible to the configured Gemini API key.
    """
    settings = get_settings()
    api_key = _get_gemini_api_key_raw(settings.db_path)
    if not api_key:
        return JSONResponse({"ok": False, "error": "missing gemini api key"}, status_code=400)
    # Call list models endpoint
    import httpx

    url = "https://generativelanguage.googleapis.com/v1beta/models"
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(url, params={"key": api_key})
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        # sanitize key in any error string
        msg = str(e).replace(api_key, "***")
        return JSONResponse({"ok": False, "error": msg}, status_code=500)

    models = data.get("models") or []
    names = []
    for m in models:
        if isinstance(m, dict) and m.get("name"):
            names.append(str(m.get("name")))
    return JSONResponse({"ok": True, "count": len(names), "models": names[:80]})


def _security_id_for_symbol(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    if s == "NIFTY":
        return "13"
    if s in {"BANKNIFTY", "BANK_NIFTY"}:
        return "25"
    raise ValueError("symbol must be NIFTY or BANKNIFTY")


def _fetch_prev_trading_day_candles(*, security_id: str, access_token: str) -> tuple[str, list] | None:
    """
    Best-effort previous-trading-day seeding using Dhan historical intraday candles.
    Returns (prev_date, candles_for_prev_date) or None.
    """
    ist = ZoneInfo("Asia/Kolkata")
    now_ist = datetime.now(tz=ist)
    today = now_ist.strftime("%Y-%m-%d")
    # Pull a small lookback window to survive weekends/holidays.
    from_dt = (now_ist - timedelta(days=12)).replace(hour=9, minute=0, second=0, microsecond=0, tzinfo=None)
    to_dt = now_ist.replace(tzinfo=None)
    req = DhanIntradayRequest(
        security_id=str(security_id),
        exchange_segment="IDX_I",
        instrument="INDEX",
        interval="1",
        oi=False,
        from_dt=from_dt,
        to_dt=to_dt,
    )
    candles = fetch_intraday_candles(req, access_token=access_token)
    by = group_by_date(candles)
    prev_dates = sorted([d for d in by.keys() if d < today])
    if not prev_dates:
        return None
    prev_date = prev_dates[-1]
    return prev_date, sorted(by[prev_date], key=lambda c: c.dt)


def _today_ist_date() -> str:
    ist = ZoneInfo("Asia/Kolkata")
    return datetime.now(tz=ist).strftime("%Y-%m-%d")


def _start_live_thread(
    *,
    symbol: str,
    security_id: str,
    client_id: str,
    access_token: str,
    gemini_key: str,
    gemini_model: str,
    qty: int,
    prev_day_candles: list | None,
    prev_day_date: str | None,
) -> None:
    global LIVE_THREAD, LIVE_OI_THREAD, LIVE_ENGINE, LIVE_CANDLE_BUILDER
    assert LIVE_LOOP is not None, "server loop not ready"

    LIVE_STOP.clear()
    global LIVE_LAST_ERROR
    LIVE_LAST_ERROR = None
    LIVE_ENGINE = LiveSimEngine(symbol=symbol, security_id=security_id, gemini_api_key=gemini_key, gemini_model=gemini_model, qty=qty)
    if prev_day_candles:
        try:
            LIVE_ENGINE.set_prev_day_candles(prev_day_candles)
        except Exception:
            pass
    LIVE_CANDLE_BUILDER = CandleBuilder1m()

    def _run() -> None:
        # DhanHQ marketfeed uses asyncio internally. In Python 3.11, non-main threads do not
        # have a default event loop, so we must create one explicitly.
        try:
            import asyncio as _asyncio

            try:
                _asyncio.get_event_loop()
            except RuntimeError:
                _asyncio.set_event_loop(_asyncio.new_event_loop())
        except Exception:
            pass

        # NOTE: use Quote or Full to get volume (Full typically contains more fields).
        # DhanHQ-py (stable 2.0.2) provides constants in `dhanhq.marketfeed`.
        try:
            from dhanhq import marketfeed  # type: ignore
            sub_type = marketfeed.Full
            exch = marketfeed.NSE
        except Exception:
            # Fallback defaults (may be wrong for some segments).
            sub_type = 2
            exch = 1

        feed = None
        try:
            feed = DhanLiveFeed(
                client_id=client_id,
                access_token=access_token,
                instruments=[LiveFeedInstrument(exchange_segment=int(exch), security_id=security_id, subscription_type=int(sub_type))],
            )

            def on_tick(msg: dict) -> None:
                if LIVE_STOP.is_set():
                    raise SystemExit()
                try:
                    sec, ltp, vol, dt = parse_marketfeed_tick(msg)
                except Exception:
                    return
                if sec and sec != security_id:
                    return
                tick = Tick(dt=dt, security_id=security_id, ltp=ltp, volume=vol)
                cb = LIVE_CANDLE_BUILDER
                eng = LIVE_ENGINE
                if cb is None or eng is None:
                    return
                candle = cb.update(tick)
                if candle is None:
                    return
                # attach latest OI snapshot (optional)
                if LIVE_OI_SNAPSHOT is not None:
                    pass
                asyncio.run_coroutine_threadsafe(eng.on_candle_close(candle), LIVE_LOOP)

            feed.iter_ticks(on_tick)
        except SystemExit:
            return
        except Exception as e:
            # Best-effort shutdown; UI can restart. Capture error for UI.
            try:
                global LIVE_LAST_ERROR
                LIVE_LAST_ERROR = f"{type(e).__name__}: {e}"
            except Exception:
                pass
            return
        finally:
            if feed is not None:
                feed.disconnect()

    LIVE_THREAD = threading.Thread(target=_run, daemon=True, name="dhan_live_feed")
    LIVE_THREAD.start()

    # Optional: OI polling (option chain). Runs independently of tick feed.
    def _run_oi() -> None:
        try:
            oc = DhanOptionChainClient(client_id=client_id, access_token=access_token)
            # For index: under_exchange_segment is typically IDX_I in REST APIs.
            ex_seg = "IDX_I"
            exp = oc.expiry_list(under_security_id=int(security_id), under_exchange_segment=ex_seg)
            exp_list = exp.get("data") or exp.get("Data") or exp.get("expiryList") or exp.get("expiry_list") or []
            if isinstance(exp_list, list) and exp_list:
                expiry = str(exp_list[0])
            else:
                return
        except Exception:
            return
        while not LIVE_STOP.is_set():
            try:
                chain = oc.option_chain(under_security_id=int(security_id), under_exchange_segment=ex_seg, expiry=expiry)
                summary = summarize_oi_walls(chain, top_n=5)
                if summary.get("ok"):
                    global LIVE_OI_SNAPSHOT
                    LIVE_OI_SNAPSHOT = summary
                    if LIVE_ENGINE is not None:
                        LIVE_ENGINE.set_oi_snapshot(summary)
            except Exception:
                pass
            time.sleep(30)

    LIVE_OI_THREAD = threading.Thread(target=_run_oi, daemon=True, name="dhan_oi_poller")
    LIVE_OI_THREAD.start()


@app.post("/api/live/start")
def live_start(symbol: str = Form(...), qty: int = Form(65), seed_prev_day: int = Form(1)) -> JSONResponse:
    """
    Starts live paper-trading loop:
    - Dhan MarketFeed websocket -> ticks
    - build 1m candles
    - Gemini per-candle decision
    - BrokerSim paper fills only
    """
    settings = get_settings()
    client_id = _get_dhan_client_id_raw(settings.db_path)
    access_token = _get_dhan_access_token_raw(settings.db_path)
    if not client_id or not access_token:
        return JSONResponse({"ok": False, "error": "missing dhan settings"}, status_code=400)

    gemini_key = _get_gemini_api_key_raw(settings.db_path)
    gemini_model = _get_gemini_model_raw(settings.db_path) or "gemini-2.0-flash"
    if not gemini_key:
        return JSONResponse({"ok": False, "error": "missing gemini api key"}, status_code=400)

    sec = _security_id_for_symbol(symbol)
    seed_info = None
    prev_day_candles = None
    prev_day_date = None
    if int(seed_prev_day) == 1:
        try:
            out = _fetch_prev_trading_day_candles(security_id=sec, access_token=access_token)
            if out is not None:
                prev_day_date, prev_day_candles = out
                seed_info = {"prev_day_date": prev_day_date, "candles": len(prev_day_candles)}
        except Exception as e:
            seed_info = {"error": str(e)}

    with LIVE_LOCK:
        if LIVE_THREAD is not None and LIVE_THREAD.is_alive():
            return JSONResponse({"ok": False, "error": "live already running"}, status_code=400)
        if LIVE_LOOP is None:
            return JSONResponse({"ok": False, "error": "server loop not ready"}, status_code=500)
        _start_live_thread(
            symbol=symbol.upper().strip(),
            security_id=sec,
            client_id=client_id,
            access_token=access_token,
            gemini_key=gemini_key,
            gemini_model=gemini_model,
            qty=int(qty),
            prev_day_candles=prev_day_candles,
            prev_day_date=prev_day_date,
        )
    # Give the background thread a moment to connect; helps surface immediate failures.
    time.sleep(0.25)
    alive = LIVE_THREAD is not None and LIVE_THREAD.is_alive()
    return JSONResponse(
        {
            "ok": True,
            "symbol": symbol.upper().strip(),
            "security_id": sec,
            "seed": seed_info,
            "thread_alive": alive,
            "last_error": LIVE_LAST_ERROR,
        }
    )


@app.post("/api/live/stop")
def live_stop() -> JSONResponse:
    global LIVE_ENGINE, LIVE_CANDLE_BUILDER, LIVE_LAST_ERROR
    LIVE_STOP.set()
    with LIVE_LOCK:
        LIVE_ENGINE = None
        LIVE_CANDLE_BUILDER = None
        LIVE_LAST_ERROR = None
    return JSONResponse({"ok": True})


@app.get("/api/live/status")
def live_status() -> JSONResponse:
    running = LIVE_THREAD is not None and LIVE_THREAD.is_alive() and LIVE_ENGINE is not None and not LIVE_STOP.is_set()
    if not running or LIVE_ENGINE is None:
        return JSONResponse({"ok": True, "running": False, "last_error": LIVE_LAST_ERROR})
    st = LIVE_ENGINE.status()
    return JSONResponse({"ok": True, "running": True, "status": asdict(st), "last_error": LIVE_LAST_ERROR})


@app.get("/api/live/candles")
def live_candles(limit: int = 300) -> JSONResponse:
    if LIVE_ENGINE is None:
        return JSONResponse({"ok": False, "error": "not running"}, status_code=400)
    return JSONResponse({"ok": True, "candles": LIVE_ENGINE.last_candles(limit=limit)})


@app.get("/api/live/decisions")
def live_decisions(limit: int = 200) -> JSONResponse:
    if LIVE_ENGINE is None:
        return JSONResponse({"ok": False, "error": "not running"}, status_code=400)
    return JSONResponse({"ok": True, "decisions": LIVE_ENGINE.last_decisions(limit=limit)})


@app.get("/api/live/state")
def live_state() -> JSONResponse:
    if LIVE_ENGINE is None:
        return JSONResponse({"ok": False, "error": "not running"}, status_code=400)
    return JSONResponse({"ok": True, "state": LIVE_ENGINE.last_state()})


def _uniq_sorted(levels: list[float]) -> list[float]:
    out: list[float] = []
    for x in sorted(set(float(v) for v in levels if v is not None)):
        if not out or abs(x - out[-1]) >= 0.01:
            out.append(float(x))
    return out


@app.get("/api/live/levels")
def live_levels() -> JSONResponse:
    """
    Convenience endpoint: returns a compact liquidity/level map derived from current state.
    This is intentionally deterministic and LLM-free.
    """
    if LIVE_ENGINE is None:
        return JSONResponse({"ok": False, "error": "not running"}, status_code=400)
    st = LIVE_ENGINE.last_state() or {}
    try:
        price = float(st.get("price")) if st.get("price") is not None else None
    except Exception:
        price = None

    symbol = str(st.get("symbol") or "").upper()
    step = 100.0 if symbol == "BANKNIFTY" else 50.0
    rns: list[float] = []
    if price is not None:
        rn = round(price / step) * step
        rns = [rn - step, rn, rn + step]

    def _lvl(obj) -> float | None:
        try:
            if obj is None:
                return None
            if isinstance(obj, dict) and obj.get("price") is not None:
                return float(obj.get("price"))
            return float(obj)
        except Exception:
            return None

    buy_side = [
        _lvl(st.get("prev_pdl")),
        _lvl(st.get("prev_last_hour_low")),
        _lvl(st.get("pdl")),
        _lvl(st.get("last_swing_low_1m")),
        _lvl(st.get("last_swing_low_5m")),
        _lvl(st.get("oi_support")),
        *rns,
    ]
    sell_side = [
        _lvl(st.get("prev_pdh")),
        _lvl(st.get("prev_last_hour_high")),
        _lvl(st.get("pdh")),
        _lvl(st.get("last_swing_high_1m")),
        _lvl(st.get("last_swing_high_5m")),
        _lvl(st.get("oi_resistance")),
        *rns,
    ]

    return JSONResponse(
        {
            "ok": True,
            "symbol": symbol,
            "price": price,
            "buy_side_liquidity": _uniq_sorted([x for x in buy_side if x is not None]),
            "sell_side_liquidity": _uniq_sorted([x for x in sell_side if x is not None]),
            "oi": {
                "oi_bias": st.get("oi_bias"),
                "support": st.get("oi_support"),
                "resistance": st.get("oi_resistance"),
                "supports": st.get("oi_supports"),
                "resistances": st.get("oi_resistances"),
            },
            "swings": {
                "last_swing_high_1m": st.get("last_swing_high_1m"),
                "last_swing_low_1m": st.get("last_swing_low_1m"),
                "last_swing_high_5m": st.get("last_swing_high_5m"),
                "last_swing_low_5m": st.get("last_swing_low_5m"),
            },
        }
    )


@app.get("/api/live/fills")
def live_fills() -> JSONResponse:
    if LIVE_ENGINE is None:
        return JSONResponse({"ok": False, "error": "not running"}, status_code=400)
    return JSONResponse({"ok": True, "fills": LIVE_ENGINE.fills()})


@app.post("/api/live/oi")
def live_update_oi(payload: dict = Body(...)) -> JSONResponse:
    """
    Optional: push an OI snapshot into the live engine.
    In v1, we store it for inspection; in v2 you can merge it into MarketState.
    """
    global LIVE_OI_SNAPSHOT
    LIVE_OI_SNAPSHOT = payload
    if LIVE_ENGINE is not None:
        LIVE_ENGINE.set_oi_snapshot(payload)
    return JSONResponse({"ok": True})


# -------------------- Transcript -> Proposal -> Merge pipeline --------------------


@app.post("/api/transcripts")
def ingest_transcript(
    content: str = Form(...),
    language: str = Form("auto"),
    tags: str = Form(""),
) -> JSONResponse:
    settings = get_settings()
    store = SqliteStore(settings.db_path)
    tid = f"tr_{uuid.uuid4().hex[:16]}"
    created_at = datetime.now(timezone.utc).isoformat()
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    store.add_transcript(Transcript(id=tid, created_at=created_at, language=None if language == "auto" else language, tags=tag_list, content=content))
    store.close()
    return JSONResponse({"ok": True, "transcript_id": tid})


@app.get("/api/transcripts")
def list_transcripts(limit: int = 50, offset: int = 0) -> JSONResponse:
    settings = get_settings()
    store = SqliteStore(settings.db_path)
    rows = store.list_transcripts(limit=limit, offset=offset)
    store.close()
    return JSONResponse({"ok": True, "transcripts": rows})


@app.get("/api/transcripts/{transcript_id}")
def get_transcript(transcript_id: str) -> JSONResponse:
    settings = get_settings()
    store = SqliteStore(settings.db_path)
    row = store.get_transcript(transcript_id)
    store.close()
    if row is None:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    return JSONResponse({"ok": True, "transcript": row})


@app.post("/api/rule_proposals/extract")
def extract_rule_proposal(transcript_id: str = Form(...)) -> JSONResponse:
    settings = get_settings()
    gemini_key = _get_gemini_api_key_raw(settings.db_path)
    gemini_model = _get_gemini_extract_model_raw(settings.db_path) or "gemini-2.0-pro"
    if not gemini_key:
        return JSONResponse({"ok": False, "error": "missing gemini api key"}, status_code=400)

    store = SqliteStore(settings.db_path)
    tr = store.get_transcript(transcript_id)
    if tr is None:
        store.close()
        return JSONResponse({"ok": False, "error": "transcript not found"}, status_code=404)

    extractor = GeminiRuleExtractor(api_key=gemini_key, model=gemini_model)
    try:
        proposal = extractor.extract_rules(tr["content"])
    except Exception as e:
        store.close()
        return JSONResponse({"ok": False, "error": f"extract failed: {e}"}, status_code=400)

    pid = f"rp_{uuid.uuid4().hex[:16]}"
    created_at = datetime.now(timezone.utc).isoformat()
    store.add_rule_proposal(pid, created_at, transcript_id, proposal)
    store.close()
    return JSONResponse({"ok": True, "proposal_id": pid, "status": "draft", "model": gemini_model})


@app.get("/api/rule_proposals")
def list_rule_proposals(limit: int = 50, offset: int = 0, status: str | None = None) -> JSONResponse:
    settings = get_settings()
    store = SqliteStore(settings.db_path)
    rows = store.list_rule_proposals(limit=limit, offset=offset, status=status)
    store.close()
    return JSONResponse({"ok": True, "proposals": rows})


@app.get("/api/rule_proposals/{proposal_id}")
def get_rule_proposal(proposal_id: str) -> JSONResponse:
    settings = get_settings()
    store = SqliteStore(settings.db_path)
    row = store.get_rule_proposal(proposal_id)
    store.close()
    if row is None:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    return JSONResponse({"ok": True, "proposal": row})


@app.post("/api/rule_proposals/{proposal_id}/status")
def set_rule_proposal_status(proposal_id: str, status: str = Form(...)) -> JSONResponse:
    settings = get_settings()
    store = SqliteStore(settings.db_path)
    try:
        store.set_rule_proposal_status(proposal_id, status)
    except Exception as e:
        store.close()
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    store.close()
    return JSONResponse({"ok": True, "proposal_id": proposal_id, "status": status})


@app.post("/api/rule_proposals/{proposal_id}/merge")
def merge_rule_proposal(proposal_id: str) -> JSONResponse:
    settings = get_settings()
    store = SqliteStore(settings.db_path)
    row = store.get_rule_proposal(proposal_id)
    if row is None:
        store.close()
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    if row["status"] != "approved":
        store.close()
        return JSONResponse({"ok": False, "error": "proposal must be approved before merge"}, status_code=400)

    proposal = row["proposal"]
    source_transcript_id = row.get("source_transcript_id")
    try:
        result = merge_rule_proposal_into_rulebook(
            rulebook_path=settings.rulebook_path,
            proposal=proposal,
            source_transcript_id=source_transcript_id,
            today=datetime.now(timezone.utc).date().isoformat(),
        )
    except Exception as e:
        store.close()
        return JSONResponse({"ok": False, "error": f"merge failed: {e}"}, status_code=400)

    # Update schema pointer (best-effort).
    try:
        rb_dir = Path(settings.rulebook_path).parent
        schema_path = rb_dir / "nexus_ultra_v2.rag_schema.json"
        if schema_path.exists():
            schema_raw = json.loads(schema_path.read_text(encoding="utf-8"))
            schema_raw["rulebook_version"] = result.new_version
            schema_raw["updated_at"] = datetime.now(timezone.utc).date().isoformat()
            schema_path.write_text(json.dumps(schema_raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass

    # Rebuild deterministic rulebook index for fast retrieval (optional but recommended).
    index_info = None
    try:
        rb = load_rulebook(settings.rulebook_path)
        n = store.rebuild_rulebook_index(rulebook_version=rb.version, rules=rb.raw.get("rules", []))
        index_info = {"rulebook_version": rb.version, "indexed_rules": n}
    except Exception as e:
        index_info = {"error": str(e)}

    store.set_rule_proposal_status(proposal_id, "merged")
    store.close()
    return JSONResponse({"ok": True, "merge": result.__dict__, "index": index_info})


@app.post("/api/rag/reindex")
def rag_reindex() -> JSONResponse:
    """
    Rebuilds a deterministic rulebook index in SQLite for low-latency retrieval.
    (This is not an embeddings/vector index; it's a compact structured cache.)
    """
    settings = get_settings()
    store = SqliteStore(settings.db_path)
    try:
        rb = load_rulebook(settings.rulebook_path)
        n = store.rebuild_rulebook_index(rulebook_version=rb.version, rules=rb.raw.get("rules", []))
    except Exception as e:
        store.close()
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    store.close()
    return JSONResponse({"ok": True, "rulebook_version": rb.version, "indexed_rules": n})


@app.get("/api/rulebook/rules")
def rulebook_rules(
    ids: str = Query(..., description="Comma-separated rule ids (e.g. DT-DOW-001,DT-SL-022)"),
    compact: int = Query(1, description="1=compact fields, 0=full rule objects"),
) -> JSONResponse:
    """
    Helper for UI/debugging: expands rule ids to actual rulebook entries.
    Useful to answer: "which 25 rules were retrieved on this candle?"
    """
    settings = get_settings()
    rb = load_rulebook(settings.rulebook_path)
    want = [s.strip() for s in (ids or "").split(",") if s.strip()]
    if not want:
        return JSONResponse({"ok": False, "error": "ids is required"}, status_code=400)
    want_set = set(want)

    out = []
    for r in rb.raw.get("rules", []):
        rid = str(r.get("id") or "")
        if rid in want_set:
            if int(compact):
                out.append(
                    {
                        "id": r.get("id"),
                        "category": r.get("category"),
                        "name": r.get("name"),
                        "tags": r.get("tags") or [],
                        "condition": r.get("condition"),
                        "action": r.get("action"),
                    }
                )
            else:
                out.append(r)

    # Preserve the input order (best-effort)
    by_id = {str(r.get("id")): r for r in out}
    ordered = [by_id[i] for i in want if i in by_id]

    return JSONResponse(
        {
            "ok": True,
            "rulebook_version": rb.version,
            "requested": want,
            "found": len(ordered),
            "missing": [i for i in want if i not in by_id],
            "rules": ordered,
        }
    )


@app.post("/api/sim/candle")
async def sim_candle(payload: dict = Body(...)) -> JSONResponse:
    """
    Offline test helper:
    - Creates an in-memory LiveSimEngine (paper only)
    - Accepts a single 1m candle payload and returns the Gemini decision

    Payload:
    {
      "symbol": "NIFTY",
      "dt": "2026-04-07T09:15:00+05:30",
      "open": 22838,
      "high": 22840,
      "low": 22727,
      "close": 22750,
      "volume": 12345
    }
    """
    global SIM_ENGINE
    settings = get_settings()
    gemini_key = _get_gemini_api_key_raw(settings.db_path)
    gemini_model = _get_gemini_model_raw(settings.db_path) or "gemini-2.0-flash"
    if not gemini_key:
        return JSONResponse({"ok": False, "error": "missing gemini api key"}, status_code=400)

    symbol = str(payload.get("symbol") or "NIFTY").upper().strip()
    sec = _security_id_for_symbol(symbol)
    if SIM_ENGINE is None or SIM_ENGINE.symbol != symbol:
        SIM_ENGINE = LiveSimEngine(symbol=symbol, security_id=sec, gemini_api_key=gemini_key, gemini_model=gemini_model, qty=65)

    try:
        dt = datetime.fromisoformat(str(payload["dt"]))
        candle = LiveCandle(
            dt=dt,
            open=float(payload["open"]),
            high=float(payload["high"]),
            low=float(payload["low"]),
            close=float(payload["close"]),
            volume=None if payload.get("volume") is None else float(payload["volume"]),
        )
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"invalid candle payload: {e}"}, status_code=400)

    out = await SIM_ENGINE.on_candle_close(candle)
    return JSONResponse({"ok": True, "decision": {"action": out.action, "sl": out.sl, "targets": out.targets, "raw": out.raw}})


@app.post("/api/backtest/range/dhan")
async def backtest_range_dhan(
    security_id: str = Form(...),
    exchange_segment: str = Form("IDX_I"),
    instrument: str = Form("INDEX"),
    start_date: str = Form(...),
    end_date: str = Form(...),
    interval: str = Form("1"),
    qty: int = Form(65),
    gap: float | None = Form(None),  # backward-compat (single threshold)
    gap_up: float | None = Form(None),
    gap_down: float | None = Form(None),
    flat: float = Form(15.0),
    policy: str = Form("rulebook"),  # rulebook | rag | rag_all | scenario
    include_fills: int = Form(0),
    max_fills: int = Form(200),
    include_decisions: int = Form(0),
    max_decisions: int = Form(300),
    max_entries_per_day: int | None = Form(None),
    cooldown_after_sl_candles: int = Form(0),
    lock_direction_after_first_entry: int = Form(0),
) -> JSONResponse:
    # Resolve thresholds (support old "gap" field).
    if gap_up is None:
        gap_up = 30.0 if gap is None else float(gap)
    if gap_down is None:
        gap_down = 30.0 if gap is None else float(gap)
    # Direct (non-job) execution endpoint. Prefer /submit + /job for long runs.
    fn = partial(
        _compute_backtest_range_dhan,
        security_id=security_id,
        exchange_segment=exchange_segment,
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        qty=qty,
        gap_up=float(gap_up),
        gap_down=float(gap_down),
        flat=flat,
        policy=policy,
        include_fills=include_fills,
        max_fills=max_fills,
        include_decisions=include_decisions,
        max_decisions=max_decisions,
        max_entries_per_day=None if max_entries_per_day is None else int(max_entries_per_day),
        cooldown_after_sl_candles=int(cooldown_after_sl_candles),
        lock_direction_after_first_entry=int(lock_direction_after_first_entry),
    )
    result = await anyio.to_thread.run_sync(fn)
    status = 200 if result.get("ok") else 400
    return JSONResponse(result, status_code=status)


@app.post("/api/backtest/range/dhan/submit")
async def backtest_range_dhan_submit(
    security_id: str = Form(...),
    exchange_segment: str = Form("IDX_I"),
    instrument: str = Form("INDEX"),
    start_date: str = Form(...),
    end_date: str = Form(...),
    interval: str = Form("1"),
    qty: int = Form(65),
    gap: float | None = Form(None),
    gap_up: float | None = Form(None),
    gap_down: float | None = Form(None),
    flat: float = Form(15.0),
    policy: str = Form("rulebook"),
    include_fills: int = Form(0),
    max_fills: int = Form(200),
    include_decisions: int = Form(0),
    max_decisions: int = Form(300),
    max_entries_per_day: int | None = Form(None),
    cooldown_after_sl_candles: int = Form(0),
    lock_direction_after_first_entry: int = Form(0),
) -> JSONResponse:
    if gap_up is None:
        gap_up = 30.0 if gap is None else float(gap)
    if gap_down is None:
        gap_down = 30.0 if gap is None else float(gap)
    job_id = f"bt_{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc).isoformat()
    job = BacktestJob(id=job_id, status="queued", created_at=now, updated_at=now)
    BACKTEST_JOBS[job_id] = job

    async def _run() -> None:
        job.status = "running"
        job.updated_at = datetime.now(timezone.utc).isoformat()
        try:
            fn = partial(
                _compute_backtest_range_dhan,
                security_id=security_id,
                exchange_segment=exchange_segment,
                instrument=instrument,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                qty=qty,
                gap_up=float(gap_up),
                gap_down=float(gap_down),
                flat=flat,
                policy=policy,
                include_fills=include_fills,
                max_fills=max_fills,
                include_decisions=include_decisions,
                max_decisions=max_decisions,
                max_entries_per_day=None if max_entries_per_day is None else int(max_entries_per_day),
                cooldown_after_sl_candles=int(cooldown_after_sl_candles),
                lock_direction_after_first_entry=int(lock_direction_after_first_entry),
            )
            result = await anyio.to_thread.run_sync(fn)
            job.result = result
            job.status = "done" if result.get("ok") else "error"
            if not result.get("ok"):
                job.error = str(result.get("error"))
        except Exception as e:
            job.status = "error"
            job.error = str(e)
        finally:
            job.updated_at = datetime.now(timezone.utc).isoformat()

    asyncio.create_task(_run())
    return JSONResponse({"ok": True, "job_id": job_id})


@app.get("/api/backtest/job/{job_id}")
def backtest_job_get(job_id: str) -> JSONResponse:
    job = BACKTEST_JOBS.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
    return JSONResponse({"ok": True, "job": asdict(job)})


@app.post("/api/predict/next_day/submit")
async def predict_next_day_submit(
    symbol: str = Form(...),
    training_start_date: str = Form(...),
    target_date: str = Form(...),
    use_prev_day_oi: int = Form(1),
) -> JSONResponse:
    """
    Next-day level prediction job.
    - Uses Dhan historical intraday candles for training window.
    - Uses Gemini + rulebook (+ optional live OI snapshot) to produce bucketed gap plans.
    """
    job_id = f"pr_{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc).isoformat()
    job = PredictJob(id=job_id, status="queued", created_at=now, updated_at=now)
    PREDICT_JOBS[job_id] = job

    async def _run() -> None:
        job.status = "running"
        job.updated_at = datetime.now(timezone.utc).isoformat()
        settings = get_settings()
        try:
            client_id = _get_dhan_client_id_raw(settings.db_path)
            access_token = _get_dhan_access_token_raw(settings.db_path)
            if not access_token:
                raise ValueError("missing dhan access token; save it on home page first")
            gemini_key = _get_gemini_api_key_raw(settings.db_path)
            gemini_model = _get_gemini_decision_model_raw(settings.db_path) or "gemini-2.5-flash"
            if not gemini_key:
                raise ValueError("missing gemini api key; save it on home page first")

            sec = _security_id_for_symbol(symbol)
            predictor = NextDayPredictor(
                api_key=gemini_key,
                model=gemini_model,
                db_path=settings.db_path,
                rulebook_path=settings.rulebook_path,
            )
            out = predictor.predict_next_day(
                instrument=str(symbol).upper().strip(),
                security_id=sec,
                exchange_segment="IDX_I",
                training_start_date=training_start_date,
                target_date=target_date,
                dhan_client_id=client_id,
                dhan_access_token=access_token,
                use_prev_day_oi_snapshot=bool(int(use_prev_day_oi)),
            )
            job.result = {
                "ok": True,
                "instrument": out.instrument,
                "security_id": out.security_id,
                "target_date": out.target_date,
                "training_start_date": out.training_start_date,
                "training_end_date": out.training_end_date,
                "prev_date_used": out.prev_date_used,
                "prev_levels": out.prev_levels,
                "stats": out.stats,
                "oi": out.oi,
                "retrieved_rules": out.retrieved_rules,
                "prediction": out.prediction,
            }
            job.status = "done"
        except Exception as e:
            job.status = "error"
            job.error = str(e)
        finally:
            job.updated_at = datetime.now(timezone.utc).isoformat()

    asyncio.create_task(_run())
    return JSONResponse({"ok": True, "job_id": job_id})


@app.get("/api/predict/job/{job_id}")
def predict_job_get(job_id: str) -> JSONResponse:
    job = PREDICT_JOBS.get(job_id)
    if not job:
        return JSONResponse({"ok": False, "error": "job not found"}, status_code=404)
    return JSONResponse({"ok": True, "job": asdict(job)})


@app.post("/api/oi/snapshot/capture")
def oi_snapshot_capture(symbol: str = Form(...)) -> JSONResponse:
    """
    Capture option-chain OI snapshot NOW (from Dhan), store it in SQLite tagged to today's IST date.
    This enables "historical OI" usage for next-day prediction before market open.
    """
    settings = get_settings()
    client_id = _get_dhan_client_id_raw(settings.db_path)
    access_token = _get_dhan_access_token_raw(settings.db_path)
    if not client_id or not access_token:
        return JSONResponse({"ok": False, "error": "missing dhan settings"}, status_code=400)

    sym = str(symbol).upper().strip()
    sec = _security_id_for_symbol(sym)
    ex_seg = "IDX_I"
    try:
        oc = DhanOptionChainClient(client_id=client_id, access_token=access_token)
        exp = oc.expiry_list(under_security_id=int(sec), under_exchange_segment=ex_seg)
        exp_list = exp.get("data") or exp.get("Data") or exp.get("expiryList") or exp.get("expiry_list") or []
        expiry = str(exp_list[0]) if isinstance(exp_list, list) and exp_list else None
        if not expiry:
            return JSONResponse({"ok": False, "error": "no expiry returned by dhan"}, status_code=400)
        chain = oc.option_chain(under_security_id=int(sec), under_exchange_segment=ex_seg, expiry=expiry)
        summary = summarize_oi_walls(chain, top_n=5)
        if not summary.get("ok"):
            return JSONResponse({"ok": False, "error": "option_chain summarize failed", "raw": summary}, status_code=400)

        store = SqliteStore(settings.db_path)
        now = datetime.now(timezone.utc).isoformat()
        date_ist = _today_ist_date()
        snap_id = f"oi_{sec}_{date_ist}_{uuid.uuid4().hex[:8]}"
        store.add_oi_snapshot(
            snapshot_id=snap_id,
            captured_at=now,
            date=date_ist,
            symbol=sym,
            security_id=str(sec),
            exchange_segment=ex_seg,
            expiry=expiry,
            snapshot=summary,
        )
        store.close()
        return JSONResponse({"ok": True, "id": snap_id, "date": date_ist, "symbol": sym, "security_id": sec, "expiry": expiry, "snapshot": summary})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/oi/snapshot/latest")
def oi_snapshot_latest(symbol: str) -> JSONResponse:
    settings = get_settings()
    sym = str(symbol).upper().strip()
    sec = _security_id_for_symbol(sym)
    date_ist = _today_ist_date()
    store = SqliteStore(settings.db_path)
    row = store.get_latest_oi_snapshot(date=date_ist, security_id=str(sec))
    store.close()
    if not row:
        return JSONResponse({"ok": False, "error": "no snapshot found for today (capture first)", "date": date_ist, "symbol": sym, "security_id": sec}, status_code=404)
    return JSONResponse({"ok": True, "row": row})


@app.get("/api/oi/snapshot/list")
def oi_snapshot_list(symbol: str | None = None, date: str | None = None, limit: int = 20) -> JSONResponse:
    settings = get_settings()
    sec = None
    if symbol:
        sec = _security_id_for_symbol(str(symbol))
    store = SqliteStore(settings.db_path)
    rows = store.list_oi_snapshots(security_id=None if sec is None else str(sec), date=date, limit=int(limit))
    store.close()
    return JSONResponse({"ok": True, "rows": rows})
