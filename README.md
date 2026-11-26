# gfactor-bench

Status: plan only. No code yet.

Scores an AI model with one general factor (g), skill factor scores, and a spread number.
Each score carries a 90% interval. Items are drawn at random from many benchmarks on every run.

## Base

Fork of [epoch-research/eci-public](https://github.com/epoch-research/eci-public) (MIT).
ECI fits one capability per model and a difficulty and slope per benchmark in PyMC.
Its `bayesian.py` already returns equal-tailed intervals at any probability.

## Changes from ECI

| ECI | gfactor-bench |
|---|---|
| Whole-benchmark scores | Single questions, many per benchmark |
| Published scores from many sources | Every model run under one setup (Inspect AI) |
| One capability number | Bifactor: g plus 5 skill factors (math, code, knowledge, language, odd tasks) |
| Fixed data | Seeded random draw per run, split across factors, randomesque pick among top 5 to 10 items |
| No spread | Spread = SD of skill factors after removing g, minus expected noise |
| Fixed sample | Adaptive: keep testing a factor until its 90% interval is under a target width or hits an item cap |

## Parts

1. Item bank: per-question results from Open LLM Leaderboard details on Hugging Face, plus BIG-bench odd tasks and template generators (random numbers, cipher keys).
2. Fit: extend ECI's PyMC model from 1 factor to bifactor, items in place of benchmarks.
3. Sampler: `numpy.random.default_rng(seed)`, new seed each run, seed saved in the report.
4. Runner: Inspect AI; OpenRouter for API models, Ollama or vLLM for local.
5. Adaptive loop: score, find widest 90% interval, draw more items for that factor, refit.
6. Report: g and spread with 90% intervals, factor bars, skill correlation heatmap, 5 best and 5 worst tasks versus g prediction.
7. Check: simulated models, confirm 90% intervals contain the true value about 90% of the time.

## Estimate

Almost every part already exists, so the build is mostly glue code.

| Part | Existing tool | Agent time |
|---|---|---|
| Bifactor IRT fit, g plus skill factors | R `mirt` (`bfactor`) | 20 min |
| Adaptive testing, stop when interval is narrow, randomesque pick | R `mirtCAT` | 20 min |
| Item parameters for Open LLM Leaderboard questions | `tinyBenchmarks` | 15 min |
| Random draws per run | Inspect AI `shuffle=True, seed=` and `numpy.random.default_rng` | 10 min |
| Odd tasks | BIG-bench via Inspect evals | 15 min |
| Running models | Inspect AI with OpenRouter or Ollama | 20 min |
| 90% intervals, spread, report | mirt standard errors times 1.645, matplotlib | 30 min |
| Interval check on simulated models | `mirt::simdata` | 20 min |
| Total | | 2.5 to 3.5 hours |

Real time depends more on downloads (several GB of per-question data) and API rate limits than on code.
ECI's PyMC fit stays as a cross-check for g.
