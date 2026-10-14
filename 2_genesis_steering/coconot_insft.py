"""CoCoNot as a genesis vector with BOTH poles from Tülu's SFT mixture.

Part 2's CoCoNot-safety arm takes its refusals from the mixture but its compliances from CoCoNot's contrast set
(contrast/test + pref/train), none of which is in the mixture: half a genesis vector. The mixture's CoCoNot slice has
almost no compliances in the safety category, so this arm takes them from all five CoCoNot categories:

    refuse   the 128 refusals of part 2's g_coconot_safety draw, unchanged (all in the mixture)
    comply   every single-turn row of the mixture's CoCoNot slice (ai2-adapt-dev/coconot_converted) whose gold turn is
             a clean compliance (poles.is_clean_compliance), poles.py's prompt filter, de-duplication, seed-0 shuffle
             and 1,000 cap; 128 drawn with seed 1 (its own generator: no existing draw changes)

The contrast is therefore safety refusals against compliances to CoCoNot's other request types (incomplete,
indeterminate, unsupported, humanizing, and the few safety-category ones), not against safety look-alikes.
The arm is `g_coconot_insft`; it is not added to poles.ARMS.

    python coconot_insft.py build                         # CPU
    python coconot_insft.py steer --model tulu_final      # GPU: vector + part 2's grid -> results/coconot_insft/
    python ../judge.py --dump results/coconot_insft/steer_completions.jsonl --out results/coconot_insft/steer_judge.jsonl
Never commit results/coconot_insft/steer_*.jsonl (answers to harmful requests).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import poles as P
import steering as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "1_models_and_datasets"))
ARM, SRC_ARM = "g_coconot_insft", "g_coconot_safety"
OUT = HERE / "results" / "coconot_insft"
STEER_COMP, STEER_JUDGE = OUT / "steer_completions.jsonl", OUT / "steer_judge.jsonl"
VECS = OUT / "vectors.npz"


def build():
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    import registry as REG
    rows = REG.mixture_rows(REG.MIX_SOURCE["coconot"])
    comply = [(r["user"], r["assistant"]) for r in rows if P.is_clean_compliance(r["assistant"])]
    n = P._write(ARM, "comply", comply, f"tulu-3-sft-mixture/{REG.MIX_SOURCE['coconot'].split('/')[-1]}")
    refuse = P.load_vector_sets()[SRC_ARM]["refuse"]
    draw = P._subsample(P.load_pole(ARM, "comply"), P.N_SRC, np.random.default_rng(P.SEED + 1))
    with open(P.SETS / f"{ARM}.jsonl", "w") as f:
        for pole, ps in (("refuse", refuse), ("comply", draw)):
            for p in ps:
                f.write(json.dumps({"pole": pole, "prompt": p}) + "\n")
    print(f"{ARM}: {len(rows)} mixture rows, {len(comply)} clean compliances -> pool {n}; draw {len(refuse)} + {len(draw)}")


def load_set():
    rows = [json.loads(l) for l in open(P.SETS / f"{ARM}.jsonl") if l.strip()]
    return {pole: [r["prompt"] for r in rows if r["pole"] == pole] for pole in ("refuse", "comply")}


def steer(model_key, doses=T.DOSES):
    """Vector with part 2's recipe (steering.group_mean of the 128 + 128, unit norm) on the steered model, then the
    arm's cells of part 2's grid (the unsteered cells are part 2's: this code reproduces them)."""
    import behaviour as B
    todo = [c for c in T.grid_cells(model_key, doses, [ARM]) if c[1] != "baseline" and (model_key, *c) not in B._done(STEER_COMP)]
    if not todo:
        print(f"[{model_key}] grid complete"); return
    model, tok = T.load_model(T.MODELS[model_key])
    s = load_set()
    mu = {pole: T.group_mean(model, tok, s[pole]) for pole in ("refuse", "comply")}
    raw = mu["refuse"] - mu["comply"]; v = (raw / np.linalg.norm(raw)).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    old = dict(np.load(VECS)) if VECS.exists() else {}
    old.update({f"{model_key}/dir/{ARM}": v, f"{model_key}/raw/{ARM}": raw.astype(np.float32)})
    np.savez(VECS, **old)
    ev = T.load_evals(); st = T.Steerer(model, tok)
    with open(STEER_COMP, "a") as fh:
        for n, (e, a, al) in enumerate(todo, 1):
            t0 = time.time(); ps = ev[e][0]
            texts, early = st.generate(ps, al * v)
            B._write(fh, model_key, e, a, al, ps, texts, early)
            print(f"[{model_key}] [{n}/{len(todo)}] {e} {al:+d} ({time.time() - t0:.0f}s)", flush=True)
    st.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "steer"])
    ap.add_argument("--model", choices=list(T.MODELS))
    a = ap.parse_args()
    {"build": build, "steer": lambda: steer(a.model)}[a.step]()
