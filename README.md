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

## Judge panel for open-ended tasks (poetry, advice, ideas)

- Judges: 5 to 7 free models from different families (Llama, Qwen, DeepSeek, Gemma, Mistral) spread over OpenRouter free, Groq, Gemini AI Studio, Cerebras and local Ollama. Run in parallel through Inspect AI's model-graded scorers.
- Format: pairwise "A or B" with order swapped, fitted with Bradley-Terry (the RLHF and Chatbot Arena method). Sources: EQ-Bench Creative Writing, Arena-Hard, WildBench prompts.
- Judge reliability without ground truth: `crowd-kit` (Dawid-Skene, MACE) estimates each judge's accuracy and bias from agreement patterns.
- Measurable errors: planted pairs where one side is a damaged copy (inserted factual error, removed step, shuffled lines). The correct pick is known, so each judge's error rate and judge-to-judge error correlation are measured. Keep the judge set with lowest error correlation.
- Consensus bias test: same argument, conclusion flipped between consensus and contrarian. Score gap = that judge's consensus bias; subtract it or drop the judge. Same swap test for order and length bias (length control as in AlpacaEval LC).
- Blinding: strip names and sources, rewrite answers into one neutral style before judging. Rubric scores reasoning and evidence apart from agreement.
- Uniqueness: novelty = embedding distance from 20 standard answers to the same prompt by several models (NoveltyBench idea). Final = quality + w * novelty, counted only when quality passes a threshold. Fit w on human preference data (LMArena creative writing votes).
- Report: judge mean, judge SD (disagreement), 90% interval.

## Fast data

- No bulk download. DuckDB reads only the needed columns and rows of remote parquet (`hf://datasets/...`).
- tinyBenchmarks needs about 100 anchor questions per benchmark, so most per-question data is never fetched.
- About 200 models fit the model; the other thousands are skipped.
- `hf_xet` for anything that must be downloaded.

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
| Judge panel, parallel free models | Inspect AI model-graded scorers | 1 hour |
| Judge reliability | `crowd-kit` | 30 min |
| Planted-error and consensus-flip tests | script | 1 hour |
| Novelty score | `sentence-transformers` | 30 min |
| DuckDB data loader | DuckDB `hf://` | 30 min |
| Total | | 6 to 7 hours |

Real time depends more on downloads (several GB of per-question data) and API rate limits than on code.
ECI's PyMC fit stays as a cross-check for g.
