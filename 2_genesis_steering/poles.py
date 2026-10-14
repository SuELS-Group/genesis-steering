"""Refusal and compliance prompts of each safety training dataset, and the 128 + 128 drawn per vector.

Every steering vector in this experiment is built inside ONE dataset: the mean activation of its
refusal examples minus the mean activation of its compliance examples. A row's pole is decided from
its GOLD assistant turn, with the same two detectors as ../1_models_and_datasets/registry.py:

    refuse   the gold turn opens as a refusal                                   (is_refusal)
    comply   the gold turn answers, with no refusal or deflection language
             in its first 300 characters                                        (is_clean_compliance)
    neither  dropped

Prompts are kept at 12-3000 characters and >= 95 % ASCII, de-duplicated, shuffled with seed 0 and
capped at 1000 per pole.

    arm                  refuse / comply source
    g_coconot_safety     CoCoNot safety rows in the Tülu-3 SFT mixture / CoCoNot contrast + preference sets
    g_wildjailbreak      WildJailbreak rows of the mixture (hub harm tag and gold turn must agree)
    g_wildguardmix       WildGuardMix rows of the mixture (same rule)
    x_coconot_leftover   CoCoNot safety rows NOT in the mixture / the same CoCoNot compliance examples
    x_donotanswer        Do-Not-Answer, pole from GPT-4's recorded response
    x_hh_rlhf            HH-RLHF, harmless-base refusals / helpful-base compliances
    x_pku_saferlhf       PKU-SafeRLHF, pole from a response labelled safe

g_* = genesis (in Tülu-3's SFT data), x_* = non-genesis training data. The three genesis arms and
x_hh_rlhf are taken from ../1_models_and_datasets/data (built there); the other three are
built here from the local HF cache (PKU-Alignment/PKU-SafeRLHF, LibrAI/do-not-answer, allenai/coconot,
allenai/tulu-3-sft-mixture).

    python poles.py            # rebuild data/poles/ and data/vector_sets/ (CPU, ~1 min)

Out: data/poles/<arm>_{refuse,comply}.jsonl       rows {prompt, gold, origin}, <= 1000 per pole
     data/vector_sets/<arm>.jsonl                 rows {pole, prompt}: the 128 + 128 a vector is built from
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POLES = HERE / "data" / "poles"
SETS = HERE / "data" / "vector_sets"
SETUP_DATA = HERE.parent / "1_models_and_datasets" / "data"

SEED, CAP, MIN_CHARS, MAX_CHARS, MIN_ASCII = 0, 1000, 12, 3000, 0.95
N_SRC = 128                                   # prompts per pole that enter a vector
ARMS = ["g_coconot_safety", "g_wildjailbreak", "g_wildguardmix",
        "x_coconot_leftover", "x_donotanswer", "x_hh_rlhf", "x_pku_saferlhf"]
FROM_SETUP = {"g_coconot_safety": "coconot_safety", "g_wildjailbreak": "wildjailbreak",
              "g_wildguardmix": "wildguardmix", "x_hh_rlhf": "hh_rlhf"}

_REFUSE = re.compile(
    r"\b(i\s*(?:'m| am)?\s*(?:sorry(?!\s+to\s+hear)|afraid|unable)|i\s*(?:cannot|can'?t|won'?t|will not)"
    r"|as an ai|i must decline|i have to decline|i'?m not able|it'?s not appropriate"
    r"|i can'?t (?:help|assist|provide)|i apologi|i won'?t (?:help|assist|provide|be)"
    r"|i do(?:n'?t| not) (?:feel comfortable|think (?:it|that|this)(?:'s| is) (?:appropriate|a good idea|right|okay|ok|wise|advisable)))", re.I)
_DEFLECT = re.compile(
    r"(i appreciate your|i understand (?:that )?(?:you|your)|while i understand|it sounds like you"
    r"|i'?m here to (?:provide|help|assist|support)|thank you for reaching|i understand your"
    r"|it'?s important to (?:note|remember|approach|recognize|consider|clarify|acknowledge)"
    r"|i must (?:emphasize|clarify|stress|point out)|however, i |\binstead\b|i'?d be happy to help[^.]{0,80}\bbut\b"
    r"|not (?:able|something)|cannot|can'?t|unable|inappropriate|harmful|illegal|unethical|decline|refrain|sorry"
    r"|against (?:my|the|our)|i'?m an ai|as an ai|as a language model|respectful|responsible|ethical|safety|well-being)", re.I)
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def is_refusal(text):
    t = (text or "").strip().translate(_QUOTES)
    return len(t) >= 8 and bool(_REFUSE.search(t[:180]))


def is_clean_compliance(text):
    t = (text or "").strip().translate(_QUOTES)
    return len(t) >= 20 and not is_refusal(t) and not _DEFLECT.search(t[:300])


def _keep(p):
    return MIN_CHARS <= len(p) <= MAX_CHARS and sum(c.isascii() for c in p) / len(p) >= MIN_ASCII


def _norm(s):
    return " ".join((s or "").split()).lower()


def _write(arm, pole, pairs, origin):
    seen, rows = set(), []
    for p, g in pairs:
        p = (p or "").strip()
        if p and _keep(p) and _norm(p) not in seen:
            seen.add(_norm(p))
            rows.append((p, g))
    random.Random(SEED).shuffle(rows)
    rows = rows[:CAP]
    POLES.mkdir(parents=True, exist_ok=True)
    with open(POLES / f"{arm}_{pole}.jsonl", "w") as f:
        for p, g in rows:
            f.write(json.dumps({"prompt": p, "gold": (g or "")[:200], "origin": origin}) + "\n")
    return len(rows)


def build_poles():
    """Write data/poles/<arm>_{refuse,comply}.jsonl. Returns {(arm, pole): n}."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    n = {}
    for arm, src in FROM_SETUP.items():                       # both poles already decided in part 1
        rows = [json.loads(l) for l in open(SETUP_DATA / f"{src}.jsonl")]
        for pole in ("refuse", "comply"):
            n[arm, pole] = _write(arm, pole, [(r["prompt"], r.get("gold", "")) for r in rows if r["label"] == pole],
                                  f"1_models_and_datasets/data/{src}.jsonl")
    cc_comply = [(r["prompt"], r.get("gold", "")) for r in map(json.loads, open(SETUP_DATA / "coconot_safety.jsonl"))
                 if r["label"] == "comply"]

    # PKU-SafeRLHF: a response labelled safe decides the pole
    refuse, comply = [], []
    for x in load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train"):
        safe = [x[f"response_{k}"] for k in ("0", "1") if str(x[f"is_response_{k}_safe"]) == "True"]
        ref = [s for s in safe if is_refusal(s)]
        com = [s for s in safe if is_clean_compliance(s)]
        if ref:
            refuse.append((x["prompt"], ref[0]))
        elif com and len(com) == len(safe):
            comply.append((x["prompt"], com[0]))
    n["x_pku_saferlhf", "refuse"] = _write("x_pku_saferlhf", "refuse", refuse, "PKU-Alignment/PKU-SafeRLHF/train")
    n["x_pku_saferlhf", "comply"] = _write("x_pku_saferlhf", "comply", comply, "PKU-Alignment/PKU-SafeRLHF/train")

    # Do-Not-Answer: GPT-4's recorded response decides the pole
    dna = load_dataset("LibrAI/do-not-answer", split="train")
    n["x_donotanswer", "refuse"] = _write(
        "x_donotanswer", "refuse", [(x["question"], x["GPT4_response"]) for x in dna if is_refusal(x["GPT4_response"])],
        "LibrAI/do-not-answer (GPT-4)")
    n["x_donotanswer", "comply"] = _write(
        "x_donotanswer", "comply", [(x["question"], x["GPT4_response"]) for x in dna if is_clean_compliance(x["GPT4_response"])],
        "LibrAI/do-not-answer (GPT-4)")

    # CoCoNot safety rows left out of the mixture; their compliance examples are CoCoNot's own
    mix = load_dataset("allenai/tulu-3-sft-mixture", split="train").filter(
        lambda x: "coconot" in x["source"].lower(), num_proc=8)
    in_mix = {" ".join(x["messages"][0]["content"].split()) for x in mix}
    cc = load_dataset("allenai/coconot", "original", split="train")
    left = [(p, r) for p, r, c in zip(cc["prompt"], cc["response"], cc["category"])
            if c == "Requests with safety concerns" and " ".join(p.split()) not in in_mix and is_refusal(r)]
    n["x_coconot_leftover", "refuse"] = _write("x_coconot_leftover", "refuse", left,
                                               "allenai/coconot original/train, safety, not in mixture")
    n["x_coconot_leftover", "comply"] = _write("x_coconot_leftover", "comply", cc_comply,
                                               "1_models_and_datasets/data/coconot_safety.jsonl (comply)")
    return n


def load_pole(arm, pole):
    return [json.loads(l)["prompt"] for l in open(POLES / f"{arm}_{pole}.jsonl") if l.strip()]


def _subsample(lst, n, rng):
    if len(lst) <= n:
        return lst
    idx = np.sort(rng.choice(len(lst), n, replace=False))
    return [lst[i] for i in idx]


# The run drew the refusal subsamples from one seed-0 generator that had already served twelve earlier
# sources of the previous (shared-compliance-pool) grid. Those sources are not part of this experiment;
# replaying their draws (population sizes below) puts the generator in the state the run had, so the
# 128 refusal prompts per arm are the ones that were actually used.
_EARLIER_DRAWS = [147, 142, 131, 152, 408, 408, 408, 408, 408, 3437, 192, 170]


def draw_vector_sets():
    """The N_SRC refusal and N_SRC compliance prompts per arm, exactly as drawn in the run.
    Returns {arm: {"refuse": [...], "comply": [...]}} and writes data/vector_sets/<arm>.jsonl."""
    rng_pos = np.random.default_rng(SEED)
    for pop in _EARLIER_DRAWS:
        rng_pos.choice(pop, N_SRC, replace=False)
    rng_neg = np.random.default_rng(SEED + 1)                 # compliance pools: their own generator
    out = {arm: {"refuse": _subsample(load_pole(arm, "refuse"), N_SRC, rng_pos)} for arm in ARMS}
    for arm in ARMS:
        out[arm]["comply"] = _subsample(load_pole(arm, "comply"), N_SRC, rng_neg)
    SETS.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        with open(SETS / f"{arm}.jsonl", "w") as f:
            for pole in ("refuse", "comply"):
                for p in out[arm][pole]:
                    f.write(json.dumps({"pole": pole, "prompt": p}) + "\n")
    return out


def load_vector_sets():
    """{arm: {"refuse": [128 prompts], "comply": [128 prompts]}} from data/vector_sets/."""
    out = {}
    for arm in ARMS:
        rows = [json.loads(l) for l in open(SETS / f"{arm}.jsonl") if l.strip()]
        out[arm] = {pole: [r["prompt"] for r in rows if r["pole"] == pole] for pole in ("refuse", "comply")}
    return out


def pole_counts():
    """{arm: (n refuse, n comply)} of the pole files on disk."""
    return {arm: (len(load_pole(arm, "refuse")), len(load_pole(arm, "comply"))) for arm in ARMS}


if __name__ == "__main__":
    counts = build_poles()
    draw_vector_sets()
    print(f"{'arm':22s} {'refuse':>7s} {'comply':>7s}")
    for arm in ARMS:
        print(f"{arm:22s} {counts[arm, 'refuse']:7d} {counts[arm, 'comply']:7d}")
    print(f"drew {N_SRC} + {N_SRC} prompts per arm -> {SETS}")
