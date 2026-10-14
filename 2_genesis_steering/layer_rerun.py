"""Part 2's steering grid at another layer : a sweep over layers found that the same
relative push flips prompts at about half the dose at layer 16, so the main comparison is rerun there.

    vectors     every main vector rebuilt at hidden_states[L] with its own recipe: part 2's seven dataset vectors (last
                prompt token, their 128 + 128 draws, steering.group_mean), g_coconot_insft, the CAA vectors
                caa_ab and caa_mean (caa_vector's recipes), and part 2's seed-0 random direction; unit norm
    doses       norm-matched: the nominal |alpha| of part 2's grid times s_L = median ||h_L|| / median ||h_13|| over the 400
                evaluation prompts (last prompt token), so alpha is the same fraction of the residual at both layers; rows keep
                the nominal alpha (all analyses of part 2 apply unchanged) and record the dose actually added
    grid        part 2's (or --doses, e.g. 1 3 for the low end at layer 16): four benchmarks, 100 prompts each, -alpha on the harmful ones (2..12), +alpha on the benign (2..8),
                256 greedy tokens, steering.Steerer at layer L; the unsteered cells do not depend on the layer and are
                part 2's; |alpha| = 6 first, resume-safe on (model, eval, arm, alpha)
    then        ../judge.py on results/layer<L>/completions_<model>.jsonl -> judge_<model>.jsonl

    python layer_rerun.py --model tulu_final|meta_instruct --layer 16
    python layer_rerun.py --model qwen --layer <layer>

Qwen3-8B has no rows in part 2's dump, so its unsteered cells are generated here too, and its dose scale is
relative to the Llamas' reference: s_L = median ||h_L|| of Qwen / median ||h_13|| of T\"ulu-3-8B (from that model's own sweep),
so a nominal alpha is the same fraction of the activation norm as on the Llamas.
Never commit results/layer*/completions.jsonl or the judge file (answers to harmful requests).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import caa_vector as V
import coconot_insft as CI
import poles as P
import scoring
import steering as T

HERE = Path(__file__).resolve().parent
ARMS = T.DATA_ARMS + ["g_coconot_insft", "caa_ab", "caa_mean", "random"]
PRIO = [6, 4, 8, 2, 10, 12]


def out_dir(L):
    return HERE / "results" / f"layer{L}"


def build(model, tok, L, model_key=None):
    """{arm: unit vector at hidden_states[L]} and the dose scale s_L."""
    sets = P.load_vector_sets(); vec = {}
    for a in T.DATA_ARMS:
        mu = {pole: T.group_mean(model, tok, sets[a][pole], layer=L) for pole in ("refuse", "comply")}
        vec[a] = mu["refuse"] - mu["comply"]
    s = CI.load_set()
    vec["g_coconot_insft"] = T.group_mean(model, tok, s["refuse"], layer=L) - T.group_mean(model, tok, s["comply"], layer=L)
    ps = V.pairs(); d = {}
    for pole in ("refuse", "comply"):
        ab = [T.render([r["ab_question"]], tok)[0] + r[f"ab_{pole}"] for r in ps]
        d[("caa_ab", pole)] = V._hidden(model, tok, ab, [None] * len(ab), layer=L).mean(0)
        heads = [T.render([r["prompt"]], tok)[0] for r in ps]
        d[("caa_mean", pole)] = V._hidden(model, tok, [h + r[pole] for h, r in zip(heads, ps)], [len(h) for h in heads], layer=L).mean(0)
    for a in ("caa_ab", "caa_mean"):
        vec[a] = d[(a, "refuse")] - d[(a, "comply")]
    vec["random"] = np.random.default_rng(T.SEED).standard_normal(vec[T.DATA_ARMS[0]].shape).astype(np.float32)   # part 2's random
    unit = {a: (v / np.linalg.norm(v)).astype(np.float32) for a, v in vec.items()}
    ev = T.load_evals(); prompts = [p for e in T.EVAL_SIGN for p in ev[e][0]]
    med = lambda layer: float(np.median(np.linalg.norm(V._hidden(model, tok, T.render(prompts, tok), [None] * len(prompts), layer=layer), axis=1)))
    ref = json.load(open(HERE / "results" / "qwen" / "sweep_ref.json"))["median_norm"] if model_key == "qwen" else med(T.LAYER)
    return unit, {a: float(np.linalg.norm(v)) for a, v in vec.items()}, med(L) / ref


def done(f):
    seen = set()
    if f.exists():
        for l in open(f):
            r = json.loads(l); seen.add((r["model"], r["eval"], r["arm"], scoring._alpha(r["alpha"])))
    return seen


def run(model_key, L, doses=None):
    sys.path.insert(0, str(HERE.parent / "3_membership_inference"))
    import mia_quartile as Q
    D = out_dir(L); D.mkdir(parents=True, exist_ok=True)
    comp, vf = D / f"completions_{model_key}.jsonl", D / f"vectors_{model_key}.npz"
    cells = [c for c in T.grid_cells(model_key, doses or T.DOSES, ARMS) if c[1] != "baseline" or model_key not in T.MODELS]
    cells.sort(key=lambda c: PRIO.index(abs(c[2])) if abs(c[2]) in PRIO else -1)
    todo = [c for c in cells if (model_key, *c) not in done(comp)]
    if not todo:
        print(f"[{model_key}] layer {L} grid complete"); return
    Q.gpu_guard()
    model, tok = T.load_model(T.MODELS_ALL[model_key])
    if vf.exists():
        z = np.load(vf); unit = {a: z[f"dir/{a}"] for a in ARMS}; scale = float(z["scale"])
    else:
        unit, raw, scale = build(model, tok, L, model_key)
        np.savez(vf, scale=scale, **{f"dir/{a}": v for a, v in unit.items()}, **{f"rawnorm/{a}": np.float32(n) for a, n in raw.items()})
    print(f"[{model_key}] layer {L}: dose scale {scale:.3f}; {len(todo)} cells", flush=True)
    ev = T.load_evals(); st = T.Steerer(model, tok, layer=L)
    with open(comp, "a") as fh:
        for n, (e, a, al) in enumerate(todo, 1):
            t0 = time.time(); ps = ev[e][0]
            texts, early = st.generate(ps, None if a == "baseline" else al * scale * unit[a])
            for p, t, er in zip(ps, texts, early):
                fh.write(json.dumps({"model": model_key, "eval": e, "arm": a, "alpha": int(al), "layer": L, "dose": round(al * scale, 4),
                                     "stage": 1, "gen_tokens": T.GEN_TOKENS, "behavior": p, "generation": t.strip(), "early_stop": bool(er),
                                     "degen": bool(scoring.is_degenerate(t) or er)}) + "\n")
            fh.flush()
            print(f"[{model_key}] [{n}/{len(todo)}] {e} {a} {al:+d} ({time.time() - t0:.0f}s)", flush=True)
    st.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(T.MODELS_ALL), required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--doses", type=int, nargs="*", help="nominal |alpha| (default part 2's grid)")
    a = ap.parse_args()
    run(a.model, a.layer, a.doses)
