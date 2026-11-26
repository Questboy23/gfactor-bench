"""gfb test MODEL | gfb audit-judges | gfb build-reference | gfb calibrate"""
from __future__ import annotations

import argparse
import json



def main(argv=None):
    ap = argparse.ArgumentParser(prog="gfb", description="Psychometric test for AI models: open-ended g score, data-driven skill factors, spread")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("test", help="test a model, e.g. ollama/qwen3.5:9b or groq/openai/gpt-oss-120b")
    t.add_argument("model")
    t.add_argument("--seed", type=int, help="default: a new random seed every run")
    t.add_argument("--target-g", type=float, default=80.0, help="stop when g's 90%% interval is within +/- this many points (200 points = 1 SD)")
    t.add_argument("--target-factor", type=float, default=130.0, help="same for each skill factor")
    t.add_argument("--max-items", type=int, default=300)
    t.add_argument("--batch", type=int, default=15)
    t.add_argument("--odd", type=int, default=16, help="fresh generated odd tasks")
    t.add_argument("--open", type=int, default=8, help="judged open-ended tasks (0 to skip)")
    t.add_argument("--judges", nargs="*", help="judge models; default is a 5-model free panel")
    t.add_argument("--workers", type=int, default=4, help="parallel requests to the tested model")
    t.add_argument("--bank", choices=["open", "frontier"], default="open",
                   help="open: Open LLM Leaderboard v1 (395 open models, saturates near the top); "
                        "frontier: MathArena olympiad math + LiveBench (frontier models)")
    t.add_argument("--out", default="reports")
    a = sub.add_parser("audit-judges", help="measure judge error rate, error correlation and consensus bias")
    a.add_argument("--judges", nargs="*")
    a.add_argument("-n", type=int, default=12)
    r = sub.add_parser("build-reference", help="(re)build reference answers for open-ended prompts")
    r.add_argument("models", nargs="+", help="first model's answers are the fixed opponent")
    cp = sub.add_parser("calibrate", help="refit item parameters (open: Open LLM Leaderboard v1; frontier: MathArena + LiveBench)")
    cp.add_argument("bank", nargs="?", choices=["open", "frontier"], default="open")
    args = ap.parse_args(argv)

    if args.cmd == "test":
        from .report import write
        from .runner import run
        res = run(args.model, seed=args.seed, target_g=args.target_g, target_factor=args.target_factor,
                  max_items=args.max_items, batch=args.batch, n_odd=args.odd, n_open=args.open,
                  judges=args.judges, workers=args.workers, bank=args.bank)
        path = write(res, args.out)
        g = res["g"]
        print(f"\ng = {g['score']:.0f} (90% interval {g['ci90'][0]:.0f} to {g['ci90'][1]:.0f}), spread = {res['spread']['score']:.0f}")
        for f in res["factors"]:
            v = res["factors"][f]
            print(f"  {f:45s} {v['index']:6.0f}  ({v['deviation']:+.0f} vs g, 90% {v['ci90'][0]:+.0f} to {v['ci90'][1]:+.0f})")
        print(f"report: {path}")
    elif args.cmd == "audit-judges":
        from .judge import DEFAULT_JUDGES, audit
        print(json.dumps(audit(args.judges or DEFAULT_JUDGES, n=args.n), indent=1))
    elif args.cmd == "build-reference":
        from pathlib import Path
        from .judge import build_reference
        prompts = json.loads((Path(__file__).parent / "data" / "open_prompts.json").read_text())
        build_reference(args.models, prompts)
        print("reference answers saved")
    elif args.cmd == "calibrate":
        from .calibrate import main as cal
        cal(["frontier"] if args.bank == "frontier" else [])


if __name__ == "__main__":
    main()
