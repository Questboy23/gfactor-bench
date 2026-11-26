"""Write a test result as JSON, Markdown and a self-contained HTML page with charts."""
from __future__ import annotations

import base64
import html
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _factor_chart(r):
    fs = list(r["factors"])
    d = [r["factors"][f]["deviation"] for f in fs]
    lo = [r["factors"][f]["deviation"] - r["factors"][f]["ci90"][0] for f in fs]
    order = np.argsort(d)
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(fs) + 1.2))
    ax.barh([fs[i] for i in order], [d[i] for i in order], xerr=[lo[i] for i in order],
            color=["#3a7d44" if d[i] > 0 else "#b5523b" for i in order], capsize=3)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("points above or below what g predicts (90% interval)")
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


def _corr_chart(r):
    fc = r["factor_correlations"]
    M = np.array(fc["matrix"])
    fig, ax = plt.subplots(figsize=(1.0 * len(M) + 2.5, 1.0 * len(M) + 1.8))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(M)), fc["factors"], rotation=40, ha="right")
    ax.set_yticks(range(len(M)), fc["factors"])
    for i in range(len(M)):
        for j in range(len(M)):
            ax.text(j, i, "" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.6 else "black")
    fig.colorbar(im, fraction=0.046)
    ax.set_title(f"Skill correlations across {r.get('n_ref', 395)} reference models", fontsize=9)
    return _png(fig)


def to_markdown(r):
    g = r["g"]
    lines = [f"# gfactor-bench report: {r['model']}", "",
             f"Seed {r['seed']}, {r['n_items']} calibrated questions, {r['seconds']} s, {r['api_errors']} API errors.", "",
             f"Scores are open-ended: 1000 is the average of the {r.get('n_ref', 395)} reference models in the {r.get('bank', 'open')} bank, "
             "200 points is one SD, no ceiling or floor. Scores from different banks are on different scales.", "",
             "| Score | Value | 90% interval |", "|---|---|---|",
             f"| General factor (g) | {g['score']:.0f} | {g['ci90'][0]:.0f} to {g['ci90'][1]:.0f} |",
             f"| Spread across skills | {r['spread']['score']:.0f} | |", "",
             "## Skill factors (found from the data)", "", "| Factor | Score | Above or below g | 90% interval | Questions | Accuracy |", "|---|---|---|---|---|---|"]
    for f, v in sorted(r["factors"].items(), key=lambda kv: -kv[1]["deviation"]):
        lines.append(f"| {f} | {v['index']:.0f} | {v['deviation']:+.0f} | {v['ci90'][0]:+.0f} to {v['ci90'][1]:+.0f} | {v['items']} | {v['accuracy']:.2f} |")
    o = r["odd_tasks"]
    lines += ["", "## Odd tasks (fresh, generated this run, not yet calibrated)", "",
              f"Accuracy {o['accuracy']:.2f} (90% interval {o['ci90'][0]:.2f} to {o['ci90'][1]:.2f}, n={o['n']}).", "",
              "| Task | Accuracy |", "|---|---|"] + [f"| {k} | {v:.2f} |" for k, v in sorted(o["by_task"].items())]
    oe = r.get("open_ended")
    if oe:
        sd = oe["judge_sd"]
        agree = "high" if sd < 0.10 else "medium" if sd < 0.25 else "low"
        lines += ["", "## Qualitative work (judged by AI models)", "",
                  "This section is qualitative: poetry, advice, ideas and explanations graded by other AI models, "
                  "not by right or wrong answers. Treat it with more skepticism than the scores above. "
                  "The judges' standard deviation shows how much they disagreed; high disagreement means the number is weak.", "",
                  f"Judge agreement: **{agree}** (SD of judge win rates {sd:.2f}).", "",
                  f"Win rate vs reference answers {oe['win_rate']:.2f}. Score with novelty bonus {oe['score']:.2f} "
                  f"(90% interval {oe['ci90'][0]:.2f} to {oe['ci90'][1]:.2f}). Novelty {oe['novelty']:.3f}.", "",
                  "| Category | Prompts | Win rate | Mean judge SD per prompt |", "|---|---|---|---|"]
        cats = {}
        for it in oe["items"]:
            cats.setdefault(it["cat"], []).append(it)
        for c, its in sorted(cats.items()):
            wr = [x["win_rate"] for x in its if x["win_rate"] == x["win_rate"]]
            sds = [x["judge_sd"] for x in its if x["judge_sd"] is not None]
            lines.append(f"| {c} | {len(its)} | {sum(wr) / len(wr) if wr else float('nan'):.2f} | "
                         f"{f"{sum(sds) / len(sds):.2f}" if sds else "n/a"} |")
        lines += ["", "| Judge | Win rate |", "|---|---|"]
        lines += [f"| {j} | {'n/a' if v is None else f'{v:.2f}'} |" for j, v in oe["judge_means"].items()]
    res = r["residuals"]
    if res:
        lines += ["", "## Best and worst areas versus g prediction", "", "| Area | Questions | Observed | Predicted | Gap |", "|---|---|---|---|---|"]
        for x in res[:5] + ([{"subject": "...", "n": "", "observed": None}] if len(res) > 10 else []) + res[-5:] if len(res) > 10 else res:
            if x["observed"] is None:
                lines.append("| ... | | | | |")
            else:
                lines.append(f"| {x['subject']} | {x['n']} | {x['observed']:.2f} | {x['predicted']:.2f} | {x['gap']:+.2f} |")
    return "\n".join(lines) + "\n"


def to_html(r, md):
    body = html.escape(md)
    body = re.sub(r"(?m)^# (.*)$", r"<h1>\1</h1>", body)
    body = re.sub(r"(?m)^## (.*)$", r"<h2>\1</h2>", body)
    out, table = [], []
    for line in body.splitlines():
        if line.startswith("|"):
            if not set(line) <= set("|-"):
                tag = "th" if not table else "td"
                table.append("<tr>" + "".join(f"<{tag}>{c.strip()}</{tag}>" for c in line.strip("|").split("|")) + "</tr>")
            continue
        if table:
            out.append("<table>" + "".join(table) + "</table>")
            table = []
        out.append(f"<p>{line}</p>" if line and not line.startswith("<h") else line)
    if table:
        out.append("<table>" + "".join(table) + "</table>")
    charts = (f'<h2>Charts</h2><img src="data:image/png;base64,{_factor_chart(r)}"><br>'
              f'<img src="data:image/png;base64,{_corr_chart(r)}">')
    css = ("body{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:2em auto;padding:0 1em;color:#222}"
           "table{border-collapse:collapse;margin:.5em 0}td,th{border:1px solid #ccc;padding:3px 8px;text-align:left}"
           "th{background:#f3f3f3}img{max-width:100%}")
    return f"<!doctype html><meta charset=utf-8><title>gfactor-bench: {html.escape(r['model'])}</title><style>{css}</style>{''.join(out)}{charts}"


def write(r, out_dir="reports"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", r["model"]) + f"_{r['seed']}"
    md = to_markdown(r)
    (out / f"{stem}.json").write_text(json.dumps(r, indent=1, default=float))
    (out / f"{stem}.md").write_text(md)
    (out / f"{stem}.html").write_text(to_html(r, md))
    return out / f"{stem}.html"
