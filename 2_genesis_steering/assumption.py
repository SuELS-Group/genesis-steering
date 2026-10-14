"""Activation-only predictors of steering success on the real models. CPU only.

For every steered vector: the vector recomputed from layer-13 activations (install.py's files, mia_topk's WildGuardMix
file), the pools it was drawn from, and toy_genesis.measures on them (assumption score cos(Σw v, v), d′, consistency,
stability), plus its raw norm and its cosine to the behavioural normal. Outcomes from per-prompt judge labels, so that
reach can be restricted to a prompt subset (the 50-prompt grid of mia_topk) and bootstrapped over prompts.

Pools ("own pools": the population a vector's 128 + 128 draw was taken from)
    part 2 datasets, WildChat       poles.load_pole(arm, pole) (up to 1,000 per pole)
    g_coconot_insft                 refuse: g_coconot_safety's refusal pole; comply: the in-mixture CoCoNot pole
    behav                           the held-out prompts the unsteered model refused (judge A) / answered (judge B)
    mia_<arm> (part 3)       the pool mia_rank.select ranked: the source dataset's pole rows with a gold response,
                                    de-duplicated (all seven datasets for mia_all); from results/mia_rank/<model>_scores.jsonl
    random                          the union of the seven datasets' poles (shown, never in a correlation)
    mia_topk arms, wgm_member_*     WildGuardMix's in-SFT rows (mia_topk.pool; members are a decontaminated subset)
    wgm_nonmember_*                 membership.pool("nonmember", pole)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import behaviour as B
import coconot_insft as CI
import install as I
import membership as MB
import poles as P
import scoring
import steering as T
import toy_genesis as TG
import wildchat as W

HERE = Path(__file__).resolve().parent
PART3 = HERE.parent / "3_membership_inference"
sys.path.insert(0, str(PART3))
import mia_topk as K  # noqa: E402

M3 = PART3 / "results" / "mia_steer"
MIA_SETS = PART3 / "data" / "mia_sets"
MIA_KEY = {"tulu_final": "tulu", "meta_instruct": "meta"}
unit = lambda v: v / np.linalg.norm(v)


# ---- activations -------------------------------------------------------------------------------------
class Rows:
    """Layer-13 activation of any prompt on one model, looked up across the activation files (identical recipe;
    the same prompt in two files agrees to cos > 0.9998)."""
    def __init__(self, model, extras=("wildchat", "behav", "coconot_insft", "membership"), topk=False):
        self.parts = [x for x in [I.load(model, T.LAYER)] + [I.load(model, T.LAYER, extra=e) for e in extras] if x is not None]
        if topk:
            self.parts.append(K.load_acts())

    def __call__(self, ps):
        out = []
        for p in ps:
            for X, ix in self.parts:
                if p in ix:
                    out.append(X[ix[p]]); break
            else:
                raise KeyError(p[:60])
        return np.stack(out)


def _jsonl(f):
    return [json.loads(l) for l in open(f) if l.strip()]


def behav_pools(model):
    """The held-out prompts the unsteered model refused (A) and answered (B), in behaviour.load_prompts order."""
    lab = {r["behavior"]: r["judge_label"] for r in _jsonl(B.HELD_JUDGE) if r["model"] == model}
    order = [r["prompt"] for r in B.load_prompts()]
    return {"refuse": [p for p in order if lab.get(p) == "A"], "comply": [p for p in order if lab.get(p) == "B"]}


def _union(arms, pole):
    return list(dict.fromkeys(p for a in arms for p in P.load_pole(a, pole)))


def mia_pool(model, arm, pole):
    """The pool mia_rank.select took the top 128 from (rows with a gold response, de-duplicated by poles._norm)."""
    seen, out = set(), []
    for r in _jsonl(PART3 / "results" / "mia_rank" / f"{MIA_KEY[model]}_scores.jsonl"):
        if r["pole"] == pole and r["has_gold"] and (arm == "all" or r["arm"] == arm) and P._norm(r["prompt"]) not in seen:
            seen.add(P._norm(r["prompt"])); out.append(r["prompt"])
    return out


def main_specs(model):
    """{vector: {group, draw: {pole: prompts} or None, pools: {pole: prompts}, stored: unit vector or None}}."""
    V = np.load(T.VECS); Vc = np.load(CI.VECS); V3 = np.load(M3 / "vectors.npz")
    get = lambda z, k: z[k] if k in z.files else None
    VS = P.load_vector_sets(); out = {}
    for a in P.ARMS:
        out[a] = {"group": "genesis" if a.startswith("g_") else "external", "draw": VS[a],
                  "pools": {pl: P.load_pole(a, pl) for pl in ("refuse", "comply")}, "stored": get(V, f"{model}/dir/{a}")}
    out[W.ARM] = {"group": "WildChat", "draw": W.load_set(), "pools": {pl: P.load_pole(W.ARM, pl) for pl in ("refuse", "comply")},
                  "stored": None}
    out[CI.ARM] = {"group": "genesis", "draw": CI.load_set(),
                   "pools": {"refuse": P.load_pole(CI.SRC_ARM, "refuse"), "comply": P.load_pole(CI.ARM, "comply")},
                   "stored": get(Vc, f"{model}/dir/{CI.ARM}")}
    out[B.ARM] = {"group": "behavioural normal", "draw": B.vector_set(model)[0], "pools": behav_pools(model), "stored": None}
    for a in P.ARMS + ["all"]:
        d = {pl: [r["prompt"] for r in _jsonl(MIA_SETS / MIA_KEY[model] / f"{a}_{pl}.jsonl")] for pl in ("refuse", "comply")}
        pools = {pl: mia_pool(model, a, pl) for pl in ("refuse", "comply")}
        out[f"mia_{a}"] = {"group": "MIA-selected", "draw": d, "pools": pools, "stored": get(V3, f"{model}/dir/mia_{a}")}
    out["random"] = {"group": "random", "draw": None, "pools": {pl: _union(P.ARMS, pl) for pl in ("refuse", "comply")},
                     "stored": get(V, f"{model}/dir/random")}
    return out


def wgm_specs():
    """Tülu only: the WildGuardMix vectors of the secondary panel (mia_topk arms, member / non-member draws, part 2's
    WildGuardMix vector)."""
    Vm = np.load(MB.VECS); V = np.load(T.VECS); D = K.directions()      # D: the vectors mia_topk steered with
    full = {pl: [r["prompt"] for r in K.pool(pl)] for pl in K.POLES2}
    out = {}
    for a in K.arms():
        kind, k = a.rsplit("_k", 1)
        out[a] = {"group": "random draw" if kind.startswith("rand") else kind, "k": int(k), "draw": K.load_set(a), "pools": full, "stored": D[a]}
    for a in MB.arms():
        arm = a.split("_")[1]
        out[a] = {"group": arm, "k": P.N_SRC, "draw": MB.load_draw(a),
                  "pools": {pl: [r["prompt"] for r in MB.pool(arm, pl)] for pl in MB.POLES2},
                  "stored": Vm[f"tulu_final/dir/{a}"] if f"tulu_final/dir/{a}" in Vm.files else None}
    a = "g_wildguardmix"
    out[a] = {"group": "part 2 vector", "k": P.N_SRC, "draw": P.load_vector_sets()[a],
              "pools": {pl: P.load_pole(a, pl) for pl in ("refuse", "comply")}, "stored": V[f"tulu_final/dir/{a}"]}
    return out, full


def predictors(spec, R, normal, seed=0):
    """Recomputed vector, cos to the stored one, and the predictors. Pools are shuffled (seeded) first because
    measures() takes its consistency sample from the first 2,000 rows of each pool."""
    rng = np.random.default_rng(seed)
    Hr, Hc = R(spec["pools"]["refuse"]), R(spec["pools"]["comply"])
    Hr, Hc = Hr[rng.permutation(len(Hr))], Hc[rng.permutation(len(Hc))]
    if spec["draw"] is None:
        raw, v, n_draw = None, unit(spec["stored"]), P.N_SRC
    else:
        raw = R(spec["draw"]["refuse"]).mean(0) - R(spec["draw"]["comply"]).mean(0); v = unit(raw)
        n_draw = min(len(spec["draw"]["refuse"]), len(spec["draw"]["comply"]))
    out = {"N refuse pool": len(Hr), "N comply pool": len(Hc), "n draw": n_draw,
           "cos to stored": float(v @ unit(spec["stored"])) if spec["stored"] is not None and raw is not None else np.nan,
           **TG.measures(Hr, Hc, v, n_draw),
           "raw norm": float(np.linalg.norm(raw)) if raw is not None else np.nan,
           "cos to behavioural normal": float(v @ normal) if normal is not None else np.nan}
    return out, v


# ---- outcomes, per prompt ----------------------------------------------------------------------------
def load_prompt_cells(dump, judge, models=None):
    """{(model, eval, arm, alpha): (hit, degen, off-topic)}: 0/1 arrays over the benchmark's fixed prompt order
    (steering.load_evals), hit = the label the benchmark measures (B harmful, A benign), degen as scoring.load_cells,
    off-topic = judge label C; NaN where a prompt is not in the cell or not judged.
    Prompts a cell does not contain are absent from the arrays' index: `pos` maps eval -> {prompt: index}."""
    ev = T.load_evals(); pos = {e: {p: i for i, p in enumerate(ev[e][0])} for e in ev}
    labels = {}
    for r in map(json.loads, open(judge)):
        if r.get("judge_label") in ("A", "B", "C", "D"):
            labels[(r["model"], r["eval"], r["arm"], scoring._alpha(r["alpha"]), r["behavior"])] = r["judge_label"]
    by = defaultdict(dict)
    for r in map(json.loads, open(dump)):
        if models and r["model"] not in models:
            continue
        k = (r["model"], r["eval"], r["arm"], scoring._alpha(r["alpha"]))
        by[k][r["behavior"]] = (labels.get((*k, r["behavior"])),
                                bool(scoring.is_degenerate_v2(r["generation"]) or r.get("early_stop", False)))
    out = {}
    for k, d in by.items():
        e = k[1]; n = len(ev[e][0])
        hit, deg, off = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        want = "B" if e in scoring.HARM_EVALS else "A"
        for p, (lab, dg) in d.items():
            i = pos[e][p]
            hit[i] = np.nan if lab is None else float(lab == want); deg[i] = float(dg)
            off[i] = np.nan if lab is None else float(lab == "C")
        out[k] = (hit, deg, off)
    return out


def reach(pc, model, arm, e, doses, idx=None, base=None, tau=scoring.TAU):
    """scoring.reach on per-prompt cells, on the prompts `idx` (indices into the benchmark's order, repeats allowed).
    `base`: the cell dict holding the unsteered cells (default pc). None if a needed cell is missing or unjudged."""
    base = pc if base is None else base
    sl = (lambda a: a) if idx is None else (lambda a: a[idx])
    b = base.get((model, e, "baseline", 0))
    if b is None or np.isnan(sl(b[0])).any():
        return None
    r0 = sl(b[0]).mean(); sign = -1 if e in scoring.HARM_EVALS else 1
    best, seen = 0.0, False
    for d in doses:
        c = pc.get((model, e, arm, sign * d))
        if c is None or np.isnan(sl(c[0])).any():
            continue
        seen = True
        if sl(c[1]).mean() > tau:
            continue
        best = max(best, (sl(c[0]).mean() - r0) / max(1 - r0, 1e-9))
    return best if seen else None


def success(pc, model, arm, doses, idx=None, base=None):
    """{eval: reach, S, jailbreak, over-refusal} or None."""
    rs = {e: reach(pc, model, arm, e, doses, None if idx is None else idx[e], base) for e in scoring.EVALS}
    if any(v is None for v in rs.values()):
        return None
    return {**rs, "S": float(np.mean(list(rs.values()))),
            "jailbreak": float(np.mean([rs[e] for e in scoring.HARM_EVALS])),
            "over-refusal": float(np.mean([rs[e] for e in scoring.BENIGN_EVALS]))}


def success_boot(pc, model, arms, doses, n_boot, rng, base=None):
    """{arm: (s.e. of S, of jailbreak, of over-refusal)}: paired bootstrap over prompts (the same resample of each
    benchmark's 100 prompts for every arm and dose)."""
    n = {e: len(T.load_evals()[e][0]) for e in scoring.EVALS}
    acc = defaultdict(list)
    for _ in range(n_boot):
        idx = {e: rng.integers(0, n[e], n[e]) for e in scoring.EVALS}
        for a in arms:
            s = success(pc, model, a, doses, idx, base)
            acc[a].append((s["S"], s["jailbreak"], s["over-refusal"]))
    return {a: tuple(np.std(np.array(v), axis=0, ddof=1)) for a, v in acc.items()}


def first_dose(pc, model, arm, e, doses, base=None, idx=None):
    """Per prompt of benchmark e: 0 if the unsteered answer already shows the benchmark's target behaviour (judge B on
    harmful benchmarks, A on benign ones), else the smallest |alpha| in `doses` at which the steered answer shows it,
    else inf. None if a needed cell is missing or not fully judged. `base`: the cells holding the unsteered answers."""
    base = pc if base is None else base
    sl = (lambda a: a) if idx is None else (lambda a: a[idx])
    b = base.get((model, e, "baseline", 0))
    sign = -1 if e in scoring.HARM_EVALS else 1
    cs = [pc.get((model, e, arm, scoring._alpha(sign * d))) for d in doses]
    if b is None or any(c is None for c in cs) or np.isnan(sl(b[0])).any() or any(np.isnan(sl(c[0])).any() for c in cs):
        return None
    out = np.where(sl(b[0]) == 1, 0.0, np.inf)
    for d, c in zip(doses, cs):
        out[(out == np.inf) & (sl(c[0]) == 1)] = d
    return out


# ---- correlations ------------------------------------------------------------------------------------
def _rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


def partial_spearman(x, y, Z):
    """Spearman correlation of x and y after regressing the ranks of both on the ranks of each control in the list Z."""
    rx, ry = _rank(x), _rank(y)
    A = np.column_stack([np.ones(len(x))] + [_rank(z) for z in Z])
    res = lambda t: t - A @ np.linalg.lstsq(A, t, rcond=None)[0]
    ex, ey = res(rx), res(ry)
    return float(ex @ ey / np.sqrt((ex @ ex) * (ey @ ey)))


def boot_ci(stat, n, n_boot, rng):
    """(estimate, 2.5 %, 97.5 %) of stat(index array) over a bootstrap of n units; resamples on which the statistic
    is undefined (constant column) are skipped and counted."""
    est = stat(np.arange(n)); bs = []
    for _ in range(n_boot):
        with np.errstate(all="ignore"):
            v = stat(rng.integers(0, n, n))
        if np.isfinite(v):
            bs.append(v)
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), n_boot - len(bs)
