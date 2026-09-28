"""Evaluation sets, turned into CLM (state, candidates, gold) items.

Formatting follows CLM's own schema (``clm.schema.state_text``): the state head sees
``context + "\\n\\n" + question``; the action head sees each option's plain text,
nothing prefixed.

* ``arc``       – ARC-Challenge test (allenai/ai2_arc). 3-5 options. Closest in shape
                  to CLM's Nemotron Q&A pre-training: question = state, answer = action.
* ``banking77`` – BANKING77 test (mteb/banking77, fallback PolyAI/banking77). 77 intent
                  labels as candidates: a typed classification decision with large K,
                  the regime CLM's zero-shot suite did not cover.
* ``synthetic`` – offline fake items for plumbing tests.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from clm.schema import state_text

BANKING_INSTRUCTION = "Which banking customer-support intent does this message express?"


@dataclass
class Item:
    uid: str
    state: str                 # full text the state head sees
    candidates: list[str]      # texts the action head sees
    gold: int


def _subsample(items: list[Item], n: int | None, seed: int) -> list[Item]:
    if n and n < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, n)
    return items


def load_arc(n: int | None, seed: int) -> list[Item]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for r in ds:
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        if r["answerKey"] not in labels:
            continue
        items.append(Item(uid=str(r["id"]), state=state_text(r["question"], None),
                          candidates=texts, gold=labels.index(r["answerKey"])))
    return _subsample(items, n, seed)


def _banking_rows():
    """-> (test rows as (text, label_id), {label_id: label_name}, train texts)."""
    from datasets import load_dataset
    try:
        test = load_dataset("mteb/banking77", split="test")
        train = load_dataset("mteb/banking77", split="train")
        names: dict[int, str] = {}
        for split in (train, test):
            if "label_text" in split.column_names:
                for lab, txt in zip(split["label"], split["label_text"]):
                    names[int(lab)] = txt
        if len(names) != 77:
            feat = test.features["label"]
            names = {i: n for i, n in enumerate(feat.names)}
    except Exception:
        test = load_dataset("PolyAI/banking77", split="test", revision="refs/convert/parquet")
        train = load_dataset("PolyAI/banking77", split="train", revision="refs/convert/parquet")
        names = {i: n for i, n in enumerate(test.features["label"].names)}
    rows = [(t, int(l)) for t, l in zip(test["text"], test["label"])]
    return rows, names, list(train["text"])


def label_to_text(name: str) -> str:
    return name.replace("_", " ").strip().lower()


STATE_FORMATS = ("suffix", "none", "prefix")   # suffix = CLM's documented layout (context, then question)
LABEL_STYLES = ("plain", "sentence")


def banking_state(text: str, fmt: str = "suffix") -> str:
    if fmt == "suffix":
        return state_text(text, BANKING_INSTRUCTION)
    if fmt == "none":
        return state_text(text, None)
    if fmt == "prefix":
        return f"{BANKING_INSTRUCTION}\n\n{text.strip()}"
    raise ValueError(f"unknown state format {fmt!r}; choose from {STATE_FORMATS}")


def banking_label(name: str, style: str = "plain") -> str:
    t = label_to_text(name)
    if style == "plain":
        return t
    if style == "sentence":
        return f"The customer is asking about {t}."
    raise ValueError(f"unknown label style {style!r}; choose from {LABEL_STYLES}")


def load_banking77(n: int | None, seed: int, fmt: str = "suffix", label_style: str = "plain") -> list[Item]:
    rows, names, _ = _banking_rows()
    ids = sorted(names)
    cands = [banking_label(names[i], label_style) for i in ids]
    pos = {lab: j for j, lab in enumerate(ids)}
    items = [Item(uid=f"b77-{k}", state=banking_state(text, fmt), candidates=cands, gold=pos[lab])
             for k, (text, lab) in enumerate(rows)]
    return _subsample(items, n, seed)


def banking77_candidate_pool() -> list[str]:
    """Unique training utterances: realistic filler texts for large-K latency tests."""
    _, _, train = _banking_rows()
    return list(dict.fromkeys(t.strip() for t in train))


def load_synthetic(n: int | None, seed: int, k: int = 4) -> list[Item]:
    rng = random.Random(seed)
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho".split()
    items = []
    for i in range(n or 200):
        q = " ".join(rng.choices(words, k=12)) + "?"
        cands = [" ".join(rng.choices(words, k=4)) for _ in range(k)]
        items.append(Item(uid=f"syn-{i}", state=state_text(q, None), candidates=cands, gold=rng.randrange(k)))
    return items


def synthetic_pool(size: int = 2000, seed: int = 1) -> list[str]:
    rng = random.Random(seed)
    words = "card transfer refund pin atm fee balance top up exchange rate declined pending account".split()
    return list(dict.fromkeys(" ".join(rng.choices(words, k=rng.randint(6, 14))) for _ in range(size)))


LOADERS = {"arc": load_arc, "banking77": load_banking77, "synthetic": load_synthetic}


def load(name: str, n: int | None, seed: int, banking_format: str = "suffix",
         banking_labels: str = "plain") -> list[Item]:
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(LOADERS)}")
    if name == "banking77":
        return load_banking77(n, seed, banking_format, banking_labels)
    return LOADERS[name](n, seed)
