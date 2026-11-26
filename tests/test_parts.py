import numpy as np

from gfactor_bench import generators
from gfactor_bench.bank import make_prompt, score
from gfactor_bench.judge import _winner, damage
from gfactor_bench.runner import wilson90


def test_scoring():
    assert score("blah\nAnswer: C", "C", "mc") == 1
    assert score("<think>Answer: A</think>Answer: **B**", "B", "mc") == 1
    assert score("Answer: B", "C", "mc") == 0
    assert score("so 1,234 total\nAnswer: 1,234", "1234", "number") == 1
    assert score("Answer: Tuesday", "tuesday", "exact") == 1


def test_mc_shuffle_keeps_gold():
    row = {"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}
    for s in range(10):
        p, gold, _ = make_prompt("harness_hendrycksTest_elementary_mathematics_5", row, np.random.default_rng(s))
        line = [ln for ln in p.splitlines() if ln.startswith(gold + ".")][0]
        assert line.endswith(" 4")


def test_generators_seeded_and_varied():
    a = generators.sample(np.random.default_rng(1), 16)
    b = generators.sample(np.random.default_rng(1), 16)
    c = generators.sample(np.random.default_rng(2), 16)
    assert a == b and a != c
    for name, p, gold, kind in a:  # gold must satisfy its own scorer
        assert score(f"Answer: {gold}", gold, kind) == 1, name


def test_judge_parsing_and_damage():
    assert _winner("reasoning...\nWinner: 2") == 2
    assert _winner("no verdict") is None
    t = "One. Two has 3 apples. Three. Four. Five."
    assert damage(t, np.random.default_rng(0)) != t


def test_wilson():
    lo, hi = wilson90(8, 10)
    assert 0.5 < lo < 0.8 < hi < 1
