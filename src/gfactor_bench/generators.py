"""Odd tasks made fresh from a seeded random generator, so they cannot be memorized.
Each returns (task_name, prompt, gold, 'exact' | 'number')."""
from __future__ import annotations

import datetime as dt
import string

END = "End with a final line 'Answer: ...' containing only the answer."
WORDS = ["strawberry", "bookkeeper", "mississippi", "onomatopoeia", "parallelogram", "assessment",
         "committee", "embarrass", "occurrence", "rhythm", "bubblegum", "possession", "referee", "lollipop"]


def letter_count(r):
    w = r.choice(WORDS)
    ch = r.choice(sorted(set(w)))
    return f"How many times does the letter '{ch}' appear in the word '{w}'? {END}", str(w.count(ch)), "number"


def caesar(r):
    w = r.choice(WORDS)
    k = int(r.integers(1, 25))
    enc = "".join(string.ascii_lowercase[(string.ascii_lowercase.index(x) + k) % 26] for x in w)
    return f"The word '{enc}' was made by shifting each letter of an English word forward by {k} places in the alphabet (z wraps to a). What is the original word? {END}", w, "exact"


def arithmetic(r):
    a, b, c = (int(x) for x in r.integers(100, 9999, 3))
    return f"Compute {a} * {b} - {c}. {END}", str(a * b - c), "number"


def sort_numbers(r):
    xs = [int(x) for x in r.integers(-99, 999, 7)]
    return f"Sort these numbers from smallest to largest, comma separated: {', '.join(map(str, xs))}. {END}", ", ".join(map(str, sorted(xs))), "exact"


def reverse_words(r):
    ws = [str(x) for x in r.choice(WORDS, 4, replace=False)]
    return f"Write these words in reverse order (last first), separated by single spaces: {' '.join(ws)}. {END}", " ".join(ws[::-1]), "exact"


def date_math(r):
    d = dt.date(2000, 1, 1) + dt.timedelta(days=int(r.integers(0, 9000)))
    n = int(r.integers(10, 400))
    return f"What day of the week is {n} days after {d.isoformat()}? Give the full English day name. {END}", (d + dt.timedelta(days=n)).strftime("%A"), "exact"


def grid(r):
    h, w = int(r.integers(4, 7)), int(r.integers(4, 7))
    y, x = int(r.integers(0, h)), int(r.integers(0, w))
    rows = ["".join("X" if (i, j) == (y, x) else "." for j in range(w)) for i in range(h)]
    return ("In this grid, rows are numbered from 1 at the top and columns from 1 at the left.\n"
            + "\n".join(rows) + f"\nGive the row and column of the X as 'row,column'. {END}", f"{y + 1},{x + 1}", "exact")


def made_up_language(r):
    syl = ["ka", "lo", "mi", "ru", "te", "zo", "vi", "na", "pe", "su"]
    nouns = ["dog", "tree", "river", "house", "bird", "stone"]
    lex = {n: "".join(str(s) for s in r.choice(syl, 2, replace=False)) for n in nouns}
    plural = str(r.choice(["-ek", "-ul", "-is"]))
    a, b = (str(x) for x in r.choice(nouns, 2, replace=False))
    shown = "\n".join(f"{n} = {lex[n]}" for n in nouns)
    q = (f"In an invented language:\n{shown}\nPlurals add the suffix '{plural}' (example: two {a}s = {lex[a]}{plural.strip('-')}).\n"
         f"How do you write 'two {b}s'? {END}")
    return q, lex[b] + plural.strip("-"), "exact"


GENERATORS = {f.__name__: f for f in [letter_count, caesar, arithmetic, sort_numbers, reverse_words, date_math, grid, made_up_language]}


def sample(rng, n):
    names = list(GENERATORS)
    out = []
    for i in range(n):
        name = names[i % len(names)] if i < len(names) else str(rng.choice(names))
        p, g, k = GENERATORS[name](rng)
        out.append((name, p, g, k))
    return out
