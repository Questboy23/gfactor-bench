# gfactor-bench

A psychometric test for AI models. It reports a general factor score (g), skill factors that the data finds on its own, a spread score, and a 90% interval on each.

Scores are open-ended, like a game rating: 1000 is the average of 395 reference models and 200 points is one standard deviation. There is no ceiling or floor.

```
g = 1474 (90% interval 1408 to 1540), spread = 112
  <factor named after its top subjects>   1602  (+128 vs g, 90% +31 to +225)
  ...
report: reports/<model>_<seed>.html
```

## Quick start

```bash
git clone https://github.com/Questboy23/gfactor-bench && cd gfactor-bench
uv sync
uv run gfb test ollama/qwen3.5:9b             # any local Ollama model
GROQ_API_KEY=... uv run gfb test groq/openai/gpt-oss-120b
```

Any [LiteLLM](https://docs.litellm.ai/docs/providers) model name works: `ollama/`, `groq/`, `gemini/`, `openrouter/`, `openai/`, `anthropic/`, `nvidia_nim/` and 100+ others. Keys come from the provider's usual environment variable.

The open-ended section uses a judge panel of free models (Groq, Gemini, OpenRouter free, local Ollama). Set `GROQ_API_KEY`, `GEMINI_API_KEY` and `OPENROUTER_API_KEY` (all have free tiers), and run `ollama pull qwen2.5-coder:7b nomic-embed-text`. Skip it with `--open 0`, or choose judges with `--judges ...`.

## What it measures

| Part | Source | How it is scored |
|---|---|---|
| g and data-driven skill factors | 28,659 Open LLM Leaderboard questions (MMLU, ARC, HellaSwag, Winogrande, TruthfulQA, GSM8K) | Bifactor item response model calibrated on 395 models |
| Spread | the skill factors | SD of skill deviations from g, minus the part measurement noise explains |
| Odd tasks | 8 generators: letter counting, ciphers, big arithmetic, sorting, word reversal, weekday math, grid positions, an invented language | Exact match, fresh items every run |
| Open-ended | 30 prompts: poetry, advice, ideas, explanations | Pairwise judge panel versus reference answers, plus a novelty bonus |

g and factor scores: 1000 = reference average, 200 points = 1 SD, unbounded. Factor scores also show how far each sits above or below what g predicts.

## How it works

**Skill factors are found, not named by hand** (`discover.py`):
1. Accuracy per benchmark or MMLU subject for each reference model, computed on two random halves of the questions.
2. g (the first principal component) is removed from both halves.
3. Residuals are correlated across halves. Sampling noise in one half does not correlate with the other half, so only real shared skill remains.
4. Parallel analysis sets the number of factors: eigenvalues above the 95th percentile of a null with shuffled model order.
5. Subjects are clustered into that many groups. Each factor is named after its 3 most central subjects.

A simulated test with 3 planted skill groups recovers all 3 exactly.


1. **Calibration.** `tinyBenchmarks` (Polo et al. 2024, MIT) published right/wrong results of 395 models on every leaderboard question. `gfb calibrate` fits a bifactor 3PL model to that matrix: each question loads on g and on one skill factor, with a fixed guessing floor. The fitted parameters ship in `src/gfactor_bench/data/calibration.npz`, so users never download the matrix.
2. **Random draw.** Each run uses a new seed (saved in the report). It takes a random 40% slice of the bank, starts with 4 random questions per factor, and shuffles answer choices so memorized letters do not help.
3. **Adaptive testing.** After each batch the model's abilities are re-estimated (MAP with a Laplace posterior). The factor with the widest 90% interval relative to its target gets the next batch: the questions that shrink that interval most, picked at random among the top 8 (randomesque exposure control). It stops when g is within ±80 points and each factor within ±130 (defaults), or at 300 questions.
4. **Questions are fetched one row at a time** from the Hugging Face datasets-server API and cached in `~/.cache/gfactor-bench`. Nothing is bulk-downloaded.
5. **Judge panel.** Five free models from four families (OpenAI gpt-oss, Qwen, Gemini, NVIDIA Nemotron) compare the tested answer with a fixed reference answer in both orders, which cancels position bias. Judges see a blind prompt that tells them to grade reasoning and craft, not agreement with popular opinion. Brand self-mentions are stripped. The report gives win rate, a 90% bootstrap interval over prompts and judges, and the SD between judges.
6. **Novelty.** Distance of the answer's embedding from a pool of standard answers to the same prompt. It adds to the score only when the answer also beats the reference, so odd but bad answers earn nothing.
7. **Judge audit** (`gfb audit-judges`). Each judge sees planted pairs where one side is a damaged copy (a dropped passage, shuffled sentences, a wrong number), so the right pick is known. This gives each judge's error rate and the correlation of errors between judges. Consensus pairs keep the same evidence and flip only the conclusion between consensus and contrarian. The score shift is that judge's consensus bias. Judge weights drop with error rate and bias.

## Report

`reports/<model>_<seed>.html` (also `.md` and `.json`) contains g and spread with 90% intervals, each skill factor's score and deviation from g, odd-task and open-ended results, a judge table, the 5 best and 5 worst areas versus what g predicts, a factor chart, and a skill correlation heatmap across the reference models.

## Limits

- The shipped reference answers cover 12 of the 30 open-ended prompts, because free-tier quotas ran out while building them. Run `uv run gfb build-reference groq/qwen/qwen3.8-27b groq/openai/gpt-oss-120b` with fresh quota to cover all 30. The first model listed is the fixed opponent.

- The reference models are 2023 to 2024 open models. Strong modern models sit near the top of this scale, and the adaptive step then picks the hardest questions.
- Calibration used log-likelihood scoring with few-shot prompts. This tool asks zero-shot and reads a letter or number, so item parameters transfer only approximately.
- Odd tasks and open-ended scores are reported apart from g. They need 30+ tested models before they can be calibrated into the factor model.
- Free API tiers have rate limits. The runner retries with backoff and counts API errors in the report.

## Credits

- Response matrix: [tinyBenchmarks](https://github.com/felipemaiapolo/tinyBenchmarks) (MIT).
- Idea of stitching benchmarks into one capability scale: [Epoch Capabilities Index](https://github.com/epoch-research/eci-public) (MIT). This repo works with single questions rather than whole benchmarks, and adds skill factors, adaptive testing and judges.
- Questions come at runtime from MMLU, ARC, HellaSwag, Winogrande, TruthfulQA and GSM8K on Hugging Face, under their own licenses.
- Model access: [LiteLLM](https://github.com/BerriAI/litellm).

## Development

```bash
uv run pytest            # offline tests: model recovery, 90% interval coverage, parsers, generators
uv run gfb calibrate     # refit item parameters (about 20 minutes on a laptop)
```
