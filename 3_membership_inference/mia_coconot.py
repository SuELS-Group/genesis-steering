"""CoCoNot: the top against the bottom of the membership ranking on Tülu-3-8B, with small extremes.

are the
lowest-scored examples also the best for steering on CoCoNot, with vectors from just the top 24 and the bottom 24?

  pools     both poles training data of Tülu:
              refuse  part 2's CoCoNot-safety refusal pole (1,000 rows of the mixture), full gold turn (data/poles_full)
              comply  in-mixture CoCoNot compliances (1,000 rows, all five categories), full gold turn
                      from the mixture (part 2 stores only its opening)
            (part 2's own CoCoNot compliance pole is CoCoNot's contrast set, not in the mixture: ranking it by
            membership would rank non-members)
  scores    mia_quartile's (mia_topk's recipe): per token over the gold assistant turn, conditioned on the prompt,
            clean = NLL(twin) - NLL(SFT), released = NLL(base) - NLL(final), residualized within each pole on log
            response tokens, log prompt tokens and the reference model's per-token NLL; response-length deciles
  sets      per score and pole the top and the bottom K_EXT (mia_topk.stratified_top: evenly across length deciles; the
            bottom on the negated score), and N_RAND random length-stratified draws of K_EXT as the yardstick
  vectors   refusal minus compliance mean at hidden_states[13] (part 2's activation files, install.py), unit norm
  steering  mia_topk's reduced grid on Tülu-3-8B: HarmBench (-alpha) and XSTest-safe (+alpha), first 50 prompts,
            |alpha| = 4 / 6 / 8, plus this run's unsteered cells

    python mia_coconot.py pools                         # CPU
    python mia_coconot.py spans --model final|base|sft|twin
    python mia_coconot.py select                        # CPU
    python mia_coconot.py steer
    python ../judge.py --dump results/mia_quartile/coconot/steer_completions.jsonl --out .../steer_judge.jsonl
Not committed (gold turns, answers to harmful requests): data/mia_quartile/coconot/pool.jsonl, results/mia_quartile/coconot/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2_genesis_steering"))
sys.path.insert(0, str(HERE.parent / "1_models_and_datasets"))
import mia_quartile as Q     # noqa: E402
import mia_topk as K         # noqa: E402
import poles as P            # noqa: E402
import steering as T         # noqa: E402

DATA = HERE / "data" / "mia_quartile" / "coconot"; SETS = DATA / "sets"
OUT = HERE / "results" / "mia_quartile" / "coconot"
STEER_COMP, STEER_JUDGE = OUT / "steer_completions.jsonl", OUT / "steer_judge.jsonl"
SCORE_DEF = {"clean": ("sft", "twin"), "released": ("final", "base")}
K_EXT, N_RAND = 24, 2
REFUSE_ARM, COMPLY_ARM = "g_coconot_safety", "g_coconot_insft"


# ---- CPU: pools --------------------------------------------------------------------------------------
def pools():
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    import registry as REG
    ref = [json.loads(l) for l in open(HERE / "data" / "poles_full" / f"{REFUSE_ARM}_refuse.jsonl")]
    assert [r["prompt"] for r in ref] == P.load_pole(REFUSE_ARM, "refuse"), "poles_full out of step with part 2's pole"
    mix = {}
    for r in REG.mixture_rows(REG.MIX_SOURCE["coconot"]):
        mix.setdefault(P._norm(r["user"]), r["assistant"])
    comply = P.load_pole(COMPLY_ARM, "comply")
    missing = [p for p in comply if P._norm(p) not in mix]
    assert not missing, f"{len(missing)} compliances without a mixture gold turn"
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "pool.jsonl", "w") as f:
        for i, r in enumerate(ref):
            f.write(json.dumps({"pole": "refuse", "i": i, "prompt": r["prompt"], "gold": r["gold"]}, ensure_ascii=False) + "\n")
        for i, p in enumerate(comply):
            f.write(json.dumps({"pole": "comply", "i": i, "prompt": p, "gold": mix[P._norm(p)]}, ensure_ascii=False) + "\n")
    print(f"pool: {len(ref)} refusals, {len(comply)} compliances")


def pool():
    return [json.loads(l) for l in open(DATA / "pool.jsonl")]


# ---- GPU: spans ----------------------------------------------------------------------------------------
def spans(model_key):
    Q.score_items(model_key, [("pool", r["pole"], r["i"], r["prompt"], r["gold"]) for r in pool()], OUT / f"spans_{model_key}.jsonl")


# ---- CPU: scores and sets ------------------------------------------------------------------------------
def scores():
    """DataFrame per pool row: lengths, per score raw / reference NLL / residual, response-length decile (rows with at
    least 5 response tokens, as mia_topk.scores)."""
    import pandas as pd
    S = {m: {(r["pole"], r["i"]): r for r in map(json.loads, open(OUT / f"spans_{m}.jsonl"))} for m in K.MODELS}
    rows = []
    for r in pool():
        k = (r["pole"], r["i"])
        if not all(k in S[m] for m in S) or S["final"][k]["n_tok_resp"] < 5:
            continue
        row = {"pole": r["pole"], "i": r["i"], "prompt": r["prompt"], "n_resp": S["final"][k]["n_tok_resp"], "n_prompt": S["final"][k]["n_tok_prompt"]}
        for kind, (tgt, ref) in SCORE_DEF.items():
            n = S[tgt][k]["n_tok_resp"]
            row[f"{kind}_raw"] = (S[ref][k]["nll_resp"] - S[tgt][k]["nll_resp"]) / n
            row[f"{kind}_refnll"] = S[ref][k]["nll_resp"] / n
        rows.append(row)
    D = pd.DataFrame(rows); parts = []
    for pole, d in D.groupby("pole"):
        d = d.copy()
        for kind in SCORE_DEF:
            X = np.c_[np.ones(len(d)), np.log(d["n_resp"]), np.log(d["n_prompt"]), d[f"{kind}_refnll"]]
            beta, *_ = np.linalg.lstsq(X, d[f"{kind}_raw"].values, rcond=None)
            d[kind] = d[f"{kind}_raw"].values - X @ beta
        d["decile"] = pd.qcut(d["n_resp"].rank(method="first"), K.N_DECILES, labels=False)
        parts.append(d)
    return pd.concat(parts)


def arms():
    return [f"cc_{k}_{e}" for k in SCORE_DEF for e in ("top", "bottom")] + [f"cc_rand{s}" for s in range(N_RAND)]


def select():
    D = scores(); SETS.mkdir(parents=True, exist_ok=True); sets = {a: {} for a in arms()}
    for pole_j, pole in enumerate(K.POLES2):
        d = D[D.pole == pole]
        for kind in SCORE_DEF:
            sets[f"cc_{kind}_top"][pole] = K.stratified_top(d, kind, K_EXT)
            sets[f"cc_{kind}_bottom"][pole] = K.stratified_top(d.assign(neg=-d[kind]), "neg", K_EXT)
        for s in range(N_RAND):
            sets[f"cc_rand{s}"][pole] = Q.stratified_draw(d, K_EXT, np.random.default_rng(5000 + 10 * s + pole_j), "decile")
    for a, s_ in sets.items():
        with open(SETS / f"{a}.jsonl", "w") as f:
            for pole in K.POLES2:
                for p in s_[pole]:
                    f.write(json.dumps({"pole": pole, "prompt": p}, ensure_ascii=False) + "\n")
    OUT.mkdir(parents=True, exist_ok=True); D.drop(columns=["prompt"]).to_csv(OUT / "scores.csv", index=False)
    print("scored per pole:", D.groupby("pole").size().to_dict(), "->", SETS)


def load_set(arm):
    rows_ = [json.loads(l) for l in open(SETS / f"{arm}.jsonl")]
    return {pole: [r["prompt"] for r in rows_ if r["pole"] == pole] for pole in K.POLES2}


def rows_fn(model="tulu_final", layer=T.LAYER):
    """Activation of any pool prompt at hidden_states[layer] from part 2's files (main + the coconot_insft extra)."""
    import install as I
    parts = [I.load(model, layer), I.load(model, layer, extra="coconot_insft")]
    def R(ps):
        out = []
        for p in ps:
            for X, ix in parts:
                if p in ix:
                    out.append(X[ix[p]]); break
            else:
                raise KeyError(p[:60])
        return np.stack(out)
    return R


def directions(layer=T.LAYER):
    R = rows_fn(layer=layer); out = {}
    for a in arms():
        s = load_set(a); v = R(s["refuse"]).mean(0) - R(s["comply"]).mean(0)
        out[a] = (v / np.linalg.norm(v)).astype(np.float32)
    return out


def atlas_vectors(layer, model="tulu_final"):
    """Every CoCoNot vector at hidden_states[layer]: {name: (family, unit vector)} - the top / bottom 24 and random 24-draws,
    part 2's CoCoNot-safety, CoCoNot-leftover and CoCoNot in-SFT vectors, and part 3's two CoCoNot MIA selections."""
    import coconot_insft as CI
    R = rows_fn(model, layer); unit = lambda v: v / np.linalg.norm(v)
    vec = lambda s: unit(R(s["refuse"]).mean(0) - R(s["comply"]).mean(0))
    out = {}
    for k in SCORE_DEF:
        for e in ("top", "bottom"):
            out[f"{k} {e} 24"] = (f"{e} 24", vec(load_set(f"cc_{k}_{e}")))
    for s_ in range(N_RAND):
        out[f"random 24 ({s_})"] = ("random 24", vec(load_set(f"cc_rand{s_}")))
    VS = P.load_vector_sets()
    for a in ("g_coconot_safety", "x_coconot_leftover"):
        out[f"part 2 {a}"] = ("part 2 CoCoNot", vec(VS[a]))
    out["part 2 g_coconot_insft"] = ("part 2 CoCoNot", vec(CI.load_set()))
    for a in ("g_coconot_safety", "x_coconot_leftover"):
        s = {pl: [json.loads(l)["prompt"] for l in open(HERE / "data" / "mia_sets" / "tulu" / f"{a}_{pl}.jsonl")] for pl in K.POLES2}
        out[f"MIA {a}"] = ("MIA-selected", vec(s))
    return out


_BOUNDARY = {}


def plot_extremes_pca(ax, kind, layer, D, model="tulu_final", n_bg=1000, seed=0):
    """mia_quartile.plot_pca_core for CoCoNot: the pools, the top / bottom 24 of the score, and part 2's CoCoNot in-SFT
    vector (the part 2 vector of exactly these two pools)."""
    import coconot_insft as CI
    R = rows_fn(model, layer)
    H = {pole: R(D[D.pole == pole].prompt.tolist()) for pole in K.POLES2}
    if (model, layer) not in _BOUNDARY:
        _BOUNDARY[(model, layer)] = Q.boundary_fit(H["refuse"], H["comply"])
    SH = {e: {pole: R(load_set(f"cc_{kind}_{e}")[pole]) for pole in K.POLES2} for e in ("top", "bottom")}
    s2 = CI.load_set()
    vecs = {"top 24": (SH["top"]["refuse"].mean(0) - SH["top"]["comply"].mean(0), "#2a78d6"),
            "bottom 24": (SH["bottom"]["refuse"].mean(0) - SH["bottom"]["comply"].mean(0), "#eb6834"),
            "part 2 steering (CoCoNot in-SFT)": (R(s2["refuse"]).mean(0) - R(s2["comply"]).mean(0), "#2e9e6b")}
    return Q.plot_pca_core(ax, H, SH, vecs, _BOUNDARY[(model, layer)], "24", n_bg, seed)


# ---- GPU: steering -------------------------------------------------------------------------------------
def steer(model="tulu_final"):
    import behaviour as B          # its _done / _write keep part 2's row format (integer doses here)
    dirs = directions()
    grid = [(e, "baseline", 0) for e in K.EVALS] + [(e, a, T.EVAL_SIGN[e] * d) for a in arms() for e in K.EVALS for d in K.DOSES]
    todo = [c for c in grid if (model, *c) not in B._done(STEER_COMP)]
    if not todo:
        print("grid complete"); return
    ev = T.load_evals(); Q.gpu_guard()
    m, tok = T.load_model(T.MODELS[model]); st = T.Steerer(m, tok)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(STEER_COMP, "a") as fh:
        for n, (e, a, al) in enumerate(todo, 1):
            t0 = time.time(); ps = ev[e][0][:K.N_PROMPTS]
            texts, early = st.generate(ps, None if a == "baseline" else al * dirs[a])
            B._write(fh, model, e, a, al, ps, texts, early)
            print(f"[{n}/{len(todo)}] {e} {a} {al:+d} ({time.time() - t0:.0f}s)", flush=True)
    st.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pools", "spans", "select", "steer"])
    ap.add_argument("--model")
    a = ap.parse_args()
    {"pools": pools, "spans": lambda: spans(a.model), "select": select, "steer": steer}[a.stage]()
