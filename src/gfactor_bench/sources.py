"""Per-question results from frontier leaderboards, for the frontier calibration panel.

- MathArena (CC BY-NC-SA 4.0): every model's answer to every olympiad problem (AIME, HMMT, SMT, IMO, USAMO,
  Putnam, ...). Final-answer problems score correct/incorrect averaged over attempts; proof problems score
  judge points / max points. Both are soft labels in [0, 1].
- LiveBench (Apache 2.0): every model's score on every question (coding, instruction following, language).

Open LLM Leaderboard v2 also has per-question files, but each model's file is gated behind a logged-in
Hugging Face account, so it is not used.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import duckdb
import requests

CACHE = Path.home() / ".cache" / "gfactor-bench" / "src"
HF = "https://huggingface.co"


def _headers():
    f = Path.home() / ".cache" / "huggingface" / "token"
    t = os.environ.get("HF_TOKEN") or (f.read_text().strip() if f.exists() else None)
    return {"Authorization": f"Bearer {t}"} if t else {}


def _get(url, tries=8):
    for k in range(tries):
        r = requests.get(url, headers=_headers(), timeout=60)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        time.sleep(float(r.headers.get("Retry-After", 5 * 2**k)))
    r.raise_for_status()


def _urls(dataset):
    return [u for cfg in _get(f"{HF}/api/datasets/{dataset}/parquet").json().values() for sp in cfg.values() for u in sp]


def _query(dataset, select, group=""):
    """Run one aggregate over a dataset's parquet export and cache only the small result.
    DuckDB reads just the needed columns by HTTP range requests, so model transcripts are never downloaded."""
    out = CACHE / f"{dataset.replace('/', '__')}.parquet"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        h = _headers()
        if h:
            con.execute(f"CREATE SECRET (TYPE http, EXTRA_HTTP_HEADERS MAP {{'Authorization': '{h['Authorization']}'}})")
        src = f"read_parquet({_urls(dataset)}, union_by_name=true)"
        for k in range(6):
            try:
                con.execute(f"COPY (SELECT {select} FROM {src} {group}) TO '{out}'")
                break
            except duckdb.Error as e:
                if "429" not in str(e) or k == 5:
                    raise
                time.sleep(10 * 2**k)
    return duckdb.sql(f"SELECT * FROM '{out}'").fetchall()


def _columns(dataset):
    con = duckdb.connect()
    h = _headers()
    if h:
        con.execute(f"CREATE SECRET (TYPE http, EXTRA_HTTP_HEADERS MAP {{'Authorization': '{h['Authorization']}'}})")
    return {c[0] for c in con.execute(f"DESCRIBE SELECT * FROM read_parquet({_urls(dataset)[:1]})").fetchall()}


def norm_model(name: str) -> str:
    """Join key across leaderboards: lowercase, no dates, no punctuation. 'o3-mini (high)' == 'o3-mini-2025-01-31-high'."""
    n = re.sub(r"\d{4}-\d{2}-\d{2}|20\d{6}", "", name.lower().split("/")[-1])
    return re.sub(r"[^a-z0-9]", "", n)


def matharena():
    """Returns (rows, items): rows are (subject, item_id, model, score); items[subject][item_id] = (problem, gold)."""
    ds = [d["id"] for d in _get(f"{HF}/api/datasets?author=MathArena&limit=1000").json() if d["id"].endswith("_outputs") and "training" not in d["id"]]
    rows, items = [], {}
    for d in sorted(ds):
        cached = CACHE / f"{d.replace('/', '__')}.parquet"
        if cached.exists():  # already aggregated: no network
            cols = {"correct", "gold_answer", "problem"}
        else:
            cols = _columns(d)
        if "points_judge_1" in cols:
            sc, gold = "avg(points_judge_1 / nullif(max_points_judge_1, 0))", "NULL"
        elif "correct" in cols and "gold_answer" in cols:
            sc, gold = "avg(correct::DOUBLE)", "any_value(gold_answer::VARCHAR)"
        else:
            continue
        subj = "matharena_" + d.split("/")[1].removesuffix("_outputs")
        res = _query(d, f"problem_idx::VARCHAR, model_name, {sc}, {'any_value(problem)' if 'problem' in cols else 'NULL'}, {gold}", "GROUP BY 1, 2")
        for pid, m, v, prob, g in res:
            if v is not None:
                rows.append((subj, pid, m, float(v)))
                items.setdefault(subj, {})[pid] = (prob, g)
        print(f"{d}: {len(items.get(subj, {}))} problems", flush=True)
    return rows, items


def livebench():
    """Returns (rows, items) like matharena(); items hold prompt text and ground truth where public."""
    rows = [("livebench_" + t, q, m, float(s)) for q, t, m, s in
            _query("livebench/model_judgment", "question_id, task, model, avg(score)", "GROUP BY 1, 2, 3")]
    items = {}
    for cat in ("language", "coding", "instruction_following"):
        col = "ground_truth" if cat == "language" else "NULL"
        for q, t, prompt, gt in _query(f"livebench/{cat}", f"question_id, task, turns[1], {col}"):
            items.setdefault("livebench_" + t, {})[q] = (prompt, gt)
    return rows, items
