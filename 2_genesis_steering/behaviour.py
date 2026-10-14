"""The model's own refusal boundary, read from its behaviour, as a steering vector.

Theorem 1 says the shortest way across a refusal boundary is its normal. Every vector in this part estimates
that normal from a dataset's gold labels. This module estimates it from what the model itself does: on benign
look-alike prompts that are NOT in the four benchmarks, the unsteered model refuses some and answers the rest; the
vector is the mean activation of the prompts it refused minus that of the prompts it answered, at the steering
site (hidden_states[13], last prompt token), 128 + 128 drawn with seed 0, unit norm - the recipe of every other
vector, with behaviour in place of the gold label.

Held-out prompts: OR-Bench-hard-1k and XSTest-safe, minus the 100 + 100 prompts of data/evals (local HF cache /
part 1 data). Steps (GPU unless noted, resume-safe, one model at a time):

    python behaviour.py prompts                       # CPU: data/evals_heldout/behav.jsonl
    python behaviour.py generate --model tulu_final   # unsteered answers, 256 tokens -> results/behav/heldout_completions.jsonl
    python ../judge.py --dump results/behav/heldout_completions.jsonl --out results/behav/heldout_judge.jsonl
    python install.py --model tulu_final --extra behav  # activations of the held-out prompts, all layers
    python behaviour.py steer --model tulu_final      # arm "behav": the grid of part 2 + the 4 unsteered cells
    python ../judge.py --dump results/behav/steer_completions.jsonl --out results/behav/steer_judge.jsonl

The held-out prompts are benign, but the steer files hold answers to harmful requests: never commit
results/behav/steer_*.jsonl.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

import poles
import steering as T

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "data" / "evals_heldout" / "behav.jsonl"
OUT = HERE / "results" / "behav"
HELD_COMP, HELD_JUDGE = OUT / "heldout_completions.jsonl", OUT / "heldout_judge.jsonl"
STEER_COMP, STEER_JUDGE = OUT / "steer_completions.jsonl", OUT / "steer_judge.jsonl"
ARM = "behav"
ORBENCH_HARD = "bench-llm/or-bench"


def build_prompts():
    """OR-Bench-hard-1k and XSTest-safe without the evaluation prompts; same length / ASCII filter as the poles."""
    import pandas as pd
    from huggingface_hub import snapshot_download
    ev = T.load_evals()
    used = set(ev["orbench"][0]) | set(ev["xstest_safe"][0])
    root = snapshot_download(ORBENCH_HARD, repo_type="dataset", allow_patterns=["or-bench-hard-1k.csv"])
    orb = pd.read_csv(glob.glob(f"{root}/or-bench-hard-1k.csv")[0])["prompt"].tolist()
    xs = [r["prompt"] for r in map(json.loads, open(HERE.parent / "1_models_and_datasets" / "data" / "xstest.jsonl"))
          if r["label"] == "comply"]
    assert set(ev["orbench"][0]) <= set(orb) and set(ev["xstest_safe"][0]) <= set(xs), "eval prompts not from these sources"
    rows, seen = [], set()
    for src, ps in (("orbench_hard", orb), ("xstest_safe", xs)):
        for p in ps:
            if p in used or p in seen or not poles._keep(p):
                continue
            seen.add(p); rows.append({"source": src, "prompt": p})
    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROMPTS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print({s: sum(r["source"] == s for r in rows) for s in ("orbench_hard", "xstest_safe")}, "->", PROMPTS)


def load_prompts():
    return [json.loads(l) for l in open(PROMPTS) if l.strip()]


def _done(path):
    seen = set()
    if Path(path).exists():
        for ln in Path(path).open():
            if ln.strip():
                r = json.loads(ln); seen.add((r["model"], r["eval"], r["arm"], int(r["alpha"])))
    return seen


def _write(fh, model_key, ename, arm, alpha, prompts, texts, early):
    from scoring import is_degenerate
    for p, t, e in zip(prompts, texts, early):
        fh.write(json.dumps({"model": model_key, "eval": ename, "arm": arm, "alpha": int(alpha), "stage": 1,
                             "gen_tokens": T.GEN_TOKENS, "n_src": poles.N_SRC, "behavior": p, "generation": t.strip(),
                             "early_stop": bool(e), "degen": bool(is_degenerate(t) or e)}) + "\n")
    fh.flush()


def generate(model_key):
    """Unsteered greedy answers to the held-out prompts, one 'eval' per source."""
    rows = load_prompts()
    todo = [s for s in ("orbench_hard", "xstest_safe") if (model_key, s, "baseline", 0) not in _done(HELD_COMP)]
    if not todo:
        print(f"[{model_key}] held-out answers complete"); return
    model, tok = T.load_model(T.MODELS[model_key]); st = T.Steerer(model, tok)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(HELD_COMP, "a") as fh:
        for s in todo:
            ps = [r["prompt"] for r in rows if r["source"] == s]; t0 = time.time()
            texts, early = st.generate(ps)
            _write(fh, model_key, s, "baseline", 0, ps, texts, early)
            print(f"[{model_key}] {s}: {len(ps)} answers ({time.time() - t0:.0f}s)", flush=True)
    st.close()


def labels(model_key):
    """{prompt: True if the judge labels the unsteered answer a refusal (A)}, over the judged held-out prompts."""
    out = {}
    for ln in open(HELD_JUDGE):
        r = json.loads(ln)
        if r["model"] == model_key and r.get("judge_label") in ("A", "B", "C", "D"):
            out[r["behavior"]] = r["judge_label"] == "A"
    return out


def vector_set(model_key):
    """The 128 + 128 seed-0 draw: refused vs answered held-out prompts (refused = judge A, answered = judge B)."""
    lab, rows = {}, [json.loads(l) for l in open(HELD_JUDGE)]
    for r in rows:
        if r["model"] == model_key:
            lab[r["behavior"]] = r["judge_label"]
    order = [r["prompt"] for r in load_prompts()]
    pools = {"refuse": [p for p in order if lab.get(p) == "A"], "comply": [p for p in order if lab.get(p) == "B"]}
    return {pole: poles._subsample(pools[pole], poles.N_SRC, np.random.default_rng(poles.SEED + k))
            for k, pole in enumerate(("refuse", "comply"))}, {k: len(v) for k, v in pools.items()}


def direction(model_key, layer=T.LAYER):
    import install as I
    X, ix = I.load(model_key, layer, extra="behav")
    vs, _ = vector_set(model_key)
    v = X[[ix[p] for p in vs["refuse"]]].mean(0) - X[[ix[p] for p in vs["comply"]]].mean(0)
    return (v / np.linalg.norm(v)).astype(np.float32), float(np.linalg.norm(v))


def steer(model_key, doses=T.DOSES):
    """Arm 'behav' on the four benchmarks at every dose (-alpha harmful, +alpha benign, benign up to 8), plus the
    four unsteered cells as a reproduction check of the main grid."""
    v, _ = direction(model_key)
    cells = [c for c in T.grid_cells(model_key, doses, [ARM]) if (model_key, *c) not in _done(STEER_COMP)]
    if not cells:
        print(f"[{model_key}] behav grid complete"); return
    ev = T.load_evals()
    model, tok = T.load_model(T.MODELS[model_key]); st = T.Steerer(model, tok)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(STEER_COMP, "a") as fh:
        for n, (ename, arm, alpha) in enumerate(cells, 1):
            t0 = time.time(); ps = ev[ename][0]
            texts, early = st.generate(ps, None if arm == "baseline" else alpha * v)
            _write(fh, model_key, ename, arm, alpha, ps, texts, early)
            print(f"[{model_key}] [{n}/{len(cells)}] {ename} {arm} {alpha:+d} ({time.time() - t0:.0f}s)", flush=True)
    st.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prompts", "generate", "steer"])
    ap.add_argument("--model", choices=list(T.MODELS))
    a = ap.parse_args()
    {"prompts": lambda: build_prompts(), "generate": lambda: generate(a.model), "steer": lambda: steer(a.model)}[a.step]()
