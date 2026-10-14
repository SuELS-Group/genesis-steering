"""The model's own boundary on HARMFUL requests, read from its behaviour, as a steering vector.

behavioural normal n (behaviour.py) comes from benign look-alike prompts the unsteered model refuses or
answers: the over-refusal boundary. It predicts steering on Tülu but is weak on the jailbreak side, and the boundary
that decides whether a harmful request is answered was never measured. This module measures it: on adversarial harmful
prompts that are in none of the benchmarks, training pools or Tülu's SFT mixture, the unsteered model refuses some and
answers the rest (judge A = refusal, B = on-topic compliance); n_harm is the mean activation of the refused prompts minus
that of the answered ones at the steering site (hidden_states[13], last prompt token), 128 + 128 drawn with seed 0 (all
of a pole if it holds fewer), unit norm - part 2's recipe with behaviour in place of the gold label.

Prompts: WildJailbreak's eval split, `adversarial_harmful` (2,000 rows; Tülu-3 trained on the train split), minus the
prompts of data/evals, of every part 2 pole and of held-out set, with part 2's length / ASCII filter.

Arms steered on part 2's grid (steering.grid_cells, every dose, four benchmarks):
    behav_harm   n_harm
    behav_both   unit(n + n_harm), n = over-refusal normal: both behavioural boundaries at once

Steps (GPU unless noted, resume-safe, one model at a time):

    python behaviour_harm.py prompts                        # CPU: data/evals_heldout/behav_harm.jsonl
    python behaviour_harm.py generate --model tulu_final    # unsteered answers, 256 tokens
    python ../judge.py --dump results/behav_harm/heldout_completions.jsonl --out results/behav_harm/heldout_judge.jsonl
    python install.py --model tulu_final --extra behav_harm # activations of these prompts, all layers
    python behaviour_harm.py steer --model tulu_final       # arms behav_harm, behav_both
    python ../judge.py --dump results/behav_harm/steer_completions.jsonl --out results/behav_harm/steer_judge.jsonl

Both completion files hold answers to harmful requests, and the judge files repeat the requests: never commit
results/behav_harm/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import behaviour as B
import poles
import steering as T

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "data" / "evals_heldout" / "behav_harm.jsonl"
OUT = HERE / "results" / "behav_harm"
HELD_COMP, HELD_JUDGE = OUT / "heldout_completions.jsonl", OUT / "heldout_judge.jsonl"
STEER_COMP, STEER_JUDGE = OUT / "steer_completions.jsonl", OUT / "steer_judge.jsonl"
SOURCE = "wjb_eval_harmful"
ARMS = ["behav_harm", "behav_both"]


def build_prompts():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    d = pd.read_csv(hf_hub_download("allenai/wildjailbreak", "eval/eval.tsv", repo_type="dataset"), sep="\t")
    cand = d.loc[d["data_type"] == "adversarial_harmful", "adversarial"].astype(str).tolist()
    used = {p for ps, _ in T.load_evals().values() for p in ps}
    used |= {p for a in poles.ARMS for pole in ("refuse", "comply") for p in poles.load_pole(a, pole)}
    used |= {r["prompt"] for r in B.load_prompts()}
    rows, seen, drop = [], set(), {"overlap": 0, "filter": 0, "duplicate": 0}
    for p in cand:
        k = "overlap" if p in used else "duplicate" if p in seen else "filter" if not poles._keep(p) else None
        if k:
            drop[k] += 1; continue
        seen.add(p); rows.append({"source": SOURCE, "prompt": p})
    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROMPTS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(cand)} adversarial_harmful rows -> {len(rows)} kept, dropped {drop} -> {PROMPTS}")


def load_prompts():
    return [json.loads(l) for l in open(PROMPTS) if l.strip()]


def generate(model_key):
    """Unsteered greedy answers (part 2's generation settings)."""
    if (model_key, SOURCE, "baseline", 0) in B._done(HELD_COMP):
        print(f"[{model_key}] held-out answers complete"); return
    ps = [r["prompt"] for r in load_prompts()]
    model, tok = T.load_model(T.MODELS[model_key]); st = T.Steerer(model, tok)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); texts, early = st.generate(ps)
    with open(HELD_COMP, "a") as fh:
        B._write(fh, model_key, SOURCE, "baseline", 0, ps, texts, early)
    print(f"[{model_key}] {len(ps)} answers ({time.time() - t0:.0f}s)", flush=True)
    st.close()


def pools(model_key):
    """{'refuse': prompts judged A, 'comply': prompts judged B} in prompt-file order (C, D and unjudged left out)."""
    lab = {r["behavior"]: r["judge_label"] for r in map(json.loads, open(HELD_JUDGE)) if r["model"] == model_key}
    order = [r["prompt"] for r in load_prompts()]
    return {"refuse": [p for p in order if lab.get(p) == "A"], "comply": [p for p in order if lab.get(p) == "B"]}


def vector_set(model_key):
    """The seed-0 draw of poles.N_SRC per pole (all of a pole if smaller), as behaviour.vector_set."""
    P = pools(model_key)
    return {pole: poles._subsample(P[pole], poles.N_SRC, np.random.default_rng(poles.SEED + k))
            for k, pole in enumerate(("refuse", "comply"))}, {k: len(v) for k, v in P.items()}


def direction(model_key, layer=T.LAYER):
    """(unit n_harm, raw norm, draw sizes)."""
    import install as I
    X, ix = I.load(model_key, layer, extra="behav_harm")
    vs, _ = vector_set(model_key)
    v = X[[ix[p] for p in vs["refuse"]]].mean(0) - X[[ix[p] for p in vs["comply"]]].mean(0)
    return (v / np.linalg.norm(v)).astype(np.float32), float(np.linalg.norm(v)), {k: len(x) for k, x in vs.items()}


def arm_vectors(model_key, layer=T.LAYER):
    nh = direction(model_key, layer)[0]
    nb = B.direction(model_key, layer)[0]
    both = nh + nb
    return {"behav_harm": nh, "behav_both": (both / np.linalg.norm(both)).astype(np.float32)}


def steer(model_key, doses=T.DOSES):
    V = arm_vectors(model_key)
    cells = [c for c in T.grid_cells(model_key, doses, ARMS)
             if c[1] != "baseline" and (model_key, *c) not in B._done(STEER_COMP)]
    if not cells:
        print(f"[{model_key}] behav_harm grid complete"); return
    prio = [6, 4, 8, 2, 10, 12]                       # the operating dose first
    cells.sort(key=lambda c: prio.index(abs(c[2])) if abs(c[2]) in prio else len(prio))
    ev = T.load_evals()
    model, tok = T.load_model(T.MODELS[model_key]); st = T.Steerer(model, tok)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(STEER_COMP, "a") as fh:
        for n, (ename, arm, alpha) in enumerate(cells, 1):
            t0 = time.time(); ps = ev[ename][0]
            texts, early = st.generate(ps, alpha * V[arm])
            B._write(fh, model_key, ename, arm, alpha, ps, texts, early)
            print(f"[{model_key}] [{n}/{len(cells)}] {ename} {arm} {alpha:+d} ({time.time() - t0:.0f}s)", flush=True)
    st.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prompts", "generate", "steer"])
    ap.add_argument("--model", choices=list(T.MODELS))
    a = ap.parse_args()
    {"prompts": build_prompts, "generate": lambda: generate(a.model), "steer": lambda: steer(a.model)}[a.step]()
