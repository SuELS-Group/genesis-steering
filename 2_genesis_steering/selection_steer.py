"""Pooled top / bottom 128 by membership score, steered at layer 16 on five benchmarks.

    pools       per pole, the rows of part 3's token panel from the seven datasets (WildGuardMix, WildJailbreak, CoCoNot,
                CoCoNot left out, Do-Not-Answer, HH-RLHF, PKU-SafeRLHF), all pooled; CAA is left out (its poles share a
                prompt, so it has no contrast at the prompt site; its place in the ranking is reported in part 3), and so are the matched CoCoNot split and CAA's letter rows
    scores      each model ranked by its own panel scores (mia_panel.py; Meta-Llama with --template meta):
                  gap       mean log p(gold) - log p(argmax), both models
                  min-k++   both models
                  s (twin)  Tülu-3-8B-SFT minus its no-safety twin; Tülu only: no twin exists for Llama-3.1-8B-Instruct
                  consensus the mean over all of the model's panel scores of each row's rank within its pole (10 scores on
                            Tülu, 9 on Meta-Llama)
    selections  per score, the 128 highest- and the 128 lowest-ranked rows of each pole; controls: three seeded random
                128 + 128 draws (seeds 0, 1, 2) and the whole pool
    vectors     mean(selected refusals) - mean(selected compliances) at hidden_states[16] (input of block 16), last prompt
                token, from mia_geometry.py's activations of the same rows on the same model; unit norm
    grid        HarmBench, StrongREJECT at -alpha (2..12), XSTest-safe, OR-Bench and Alpaca (helpful_eval.py's 100 plainly
                benign instructions, data/evals_heldout/helpful.jsonl) at +alpha (2..8); nominal alpha times
                layer_rerun's dose scale for the model (results/layer16/vectors_<model>.npz), so the doses equal those of
                the existing layer-16 grid; 100 prompts, 256 greedy tokens; |alpha| = 6 first
                plus, for Alpaca only: the eleven arms of the existing layer-16 grid (their stored unit vectors) and the
                unsteered baseline (alpha 0)
    then        ../judge.py --dump results/selection_steer/completions_<model>.jsonl --out .../judge_<model>.jsonl

    python selection_steer.py vectors                     # CPU: vectors_<model>.npz, selections.json
    python selection_steer.py steer --model tulu_final    # GPU, resume-safe
    python selection_steer.py genesis                     # CPU: genesis vs external dataset vectors, Tülu minus Meta
Never commit completions_*.jsonl or judge_*.jsonl (answers to harmful requests).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "3_membership_inference")); sys.path.insert(0, str(HERE.parent / "1_models_and_datasets"))
OUT = HERE / "results" / "selection_steer"
L16 = HERE / "results" / "layer16"
LAYER, K, SEEDS = 16, 128, (0, 1, 2)
EXTRA_SEEDS = {"tulu_final": (3, 4, 5, 6, 7, 8, 9), "meta_instruct": (3, 4, 5, 6, 7, 8, 9)}   # seven more random draws per model (the first three unchanged)


def seeds(m):
    return SEEDS + EXTRA_SEEDS.get(m, ())


def layer_of(m):
    """The steering layer per model: block input 16 for the Llamas, the layer that model's own sweep chose."""
    if m == "qwen":
        return json.load(open(HERE / "results" / "qwen" / "sweep_summary.json"))["chosen_layer"]
    return LAYER


def ldir(m):
    """layer_rerun's folder for the model's layer (its reference grid, vectors and dose scale)."""
    return HERE / "results" / f"layer{layer_of(m)}"
DATASETS = ("WildGuardMix", "WildJailbreak", "CoCoNot", "CoCoNot, left out", "Do-Not-Answer", "HH-RLHF", "PKU-SafeRLHF")
TEMPLATE = {"tulu_final": "tulu", "meta_instruct": "meta", "qwen": "qwen"}
ACTS = {"tulu_final": "acts_tulu_final.npz", "meta_instruct": "acts_meta_instruct.npz", "qwen": "acts_qwen.npz"}
METHODS = {"tulu_final": {"gap": "sft: gap (CH)", "min-kpp": "sft: min-k++", "stwin": "sft: s (twin)"},
           "meta_instruct": {"gap": "meta: gap (CH)", "min-kpp": "meta: min-k++"},
           "qwen": {"gap": "qwen: gap (CH)", "min-kpp": "qwen: min-k++"}}
ALL_SCORES = {"tulu_final": "sft: ", "meta_instruct": "meta: ", "qwen": "qwen: "}          # prefix of every panel score of the model
OLD_ARMS = ["g_coconot_safety", "g_wildjailbreak", "g_wildguardmix", "x_coconot_leftover", "x_donotanswer", "x_hh_rlhf",
            "x_pku_saferlhf", "g_coconot_insft", "caa_ab", "caa_mean", "random"]
ALPACA = HERE / "data" / "evals_heldout" / "helpful.jsonl"
PRIO = [6, 4, 8, 2, 10, 12]
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def selections(model):
    import mia_panel as MP
    from scipy.stats import rankdata
    MP.use_template(TEMPLATE[model])
    F = [f for f in MP.features() if f["dataset"] in DATASETS and f["pole"] in ("refuse", "comply")]
    pre = ALL_SCORES[model]; allm = sorted(k for k in F[0] if k.startswith(pre) and all(k in f for f in F))
    sel = {}
    for pole in ("refuse", "comply"):
        G = [f for f in F if f["pole"] == pole]; n = len(G)
        cons = np.mean([rankdata([f[m] for f in G]) / n for m in allm], axis=0)
        ranked = {**{s: np.array([f[key] for f in G]) for s, key in METHODS[model].items()}, "consensus": cons}
        for s, v in ranked.items():
            o = np.argsort(-v, kind="stable")
            sel[(f"{s}_top", pole)] = [G[i]["k"] for i in o[:K]]; sel[(f"{s}_bottom", pole)] = [G[i]["k"] for i in o[-K:]]
        for sd in seeds(model):
            rng = np.random.default_rng(sd)
            sel[(f"random{sd}", pole)] = [G[i]["k"] for i in rng.choice(n, K, replace=False)]
        sel[("all", pole)] = [f["k"] for f in G]
    return F, sel, allm


def vectors():
    import mia_geometry as MG
    OUT.mkdir(parents=True, exist_ok=True); info = {}
    for model in [m for m in TEMPLATE if (MG.OUT / ACTS[m]).exists()]:
        F, sel, allm = selections(model); IT = {f["k"]: f for f in F}
        X = np.load(MG.OUT / ACTS[model])[f"prompt/L{layer_of(model)}"]
        arms = sorted({a for a, _ in sel}); cen = {(a, p): X[ks].astype(np.float32).mean(0) for (a, p), ks in sel.items()}
        v = {a: cen[(a, "refuse")] - cen[(a, "comply")] for a in arms}
        old = np.load(ldir(model) / f"vectors_{model}.npz")
        dprime = {}
        for a in arms:
            u = v[a] / np.linalg.norm(v[a]); pR, pC = X[sel[(a, "refuse")]].astype(np.float32) @ u, X[sel[(a, "comply")]].astype(np.float32) @ u
            dprime[a] = float((pR.mean() - pC.mean()) / np.sqrt((pR.var(ddof=1) + pC.var(ddof=1)) / 2))
        info[model] = {"scores_in_consensus": allm, "K": K,
                       "n_pool": {p: len(sel[("all", p)]) for p in ("refuse", "comply")},
                       "composition": {a: {p: {d: sum(IT[k]["dataset"] == d for k in sel[(a, p)]) for d in DATASETS} for p in ("refuse", "comply")} for a in arms},
                       "member_share": {a: {p: float(np.mean([IT[k]["member"] for k in sel[(a, p)]])) for p in ("refuse", "comply")} for a in arms},
                       "cos_to_all": {a: cos(v[a], v["all"]) for a in arms}, "dprime": dprime,
                       "cos_to_dataset_vectors": {a: {o: cos(v[a], old[f"dir/{o}"]) for o in OLD_ARMS[:7]} for a in arms},
                       "norm": {a: float(np.linalg.norm(v[a])) for a in arms}}
        np.savez(OUT / f"vectors_{model}.npz", **{f"dir/{a}": (v[a] / np.linalg.norm(v[a])).astype(np.float32) for a in arms})
        print(f"\n[{model}] pool {info[model]['n_pool']}, consensus over {len(allm)} scores")
        for a in arms:
            ms = info[model]["member_share"][a]
            print(f"  {a:16s} cos(all) {info[model]['cos_to_all'][a]:+.3f}  d' {dprime[a]:.2f}  member share R/C {ms['refuse']:.2f}/{ms['comply']:.2f}  "
                  f"mean cos to dataset vectors {np.mean(list(info[model]['cos_to_dataset_vectors'][a].values())):+.3f}")
    json.dump(info, open(OUT / "selections.json", "w"), indent=1)


def evals():
    import steering as T
    ev = {e: (T.load_evals()[e][0], s) for e, s in T.EVAL_SIGN.items()}
    ev["alpaca"] = ([json.loads(l)["prompt"] for l in open(ALPACA) if l.strip()], +1)
    return ev


def cells(new_arms):
    import steering as T
    out = []
    for e, s in {**T.EVAL_SIGN, "alpaca": +1}.items():
        doses = [d for d in T.DOSES if s < 0 or d <= T.BENIGN_MAX_DOSE]
        out += [(e, a, s * d) for a in new_arms for d in doses]
        if e == "alpaca":
            out += [(e, a, s * d) for a in OLD_ARMS for d in doses] + [(e, "baseline", 0)]
    out.sort(key=lambda c: -1 if c[2] == 0 else PRIO.index(abs(c[2])))
    return out


def fine_cells(arms, doses):
    """Finer dose grid : every arm on all five benchmarks at the given nominal doses (benign ones up to
    steering.BENIGN_MAX_DOSE), mid doses first."""
    import steering as T
    out = []
    for e, sgn in {**T.EVAL_SIGN, "alpaca": +1}.items():
        out += [(e, a, sgn * d) for a in arms for d in doses if sgn < 0 or d <= T.BENIGN_MAX_DOSE]
    mid = np.median(doses) if doses else 0
    return sorted(out, key=lambda c: abs(abs(c[2]) - mid))


def _labels(model_key):
    """{(eval, arm, alpha, prompt): (judge label or None, degenerate)} from every judged run of this model at layer 16: this run,
    layer_rerun's grid, and part 2's unsteered baseline."""
    import scoring
    import steering as T
    lab, out = {}, {}
    for dump, judge in ((OUT / f"completions_{model_key}.jsonl", OUT / f"judge_{model_key}.jsonl"),
                        (ldir(model_key) / f"completions_{model_key}.jsonl", ldir(model_key) / f"judge_{model_key}.jsonl"), (T.DUMP, scoring.JUDGE)):
        if not (Path(dump).exists() and Path(judge).exists()):
            continue
        only_base = Path(dump) == Path(T.DUMP)      # part 2's dump is layer 13: only its unsteered baseline belongs here (bug fixed:
                                                    # its layer-13 rows of the same arm names had overwritten the layer-16 labels)
        for r in map(json.loads, open(judge)):
            if r.get("model") == model_key and r.get("judge_label") in ("A", "B", "C", "D") and (not only_base or r["arm"] == "baseline"):
                lab[(r["eval"], r["arm"], scoring._alpha(r["alpha"]), r["behavior"])] = r["judge_label"]
        for r in map(json.loads, open(dump)):
            if r.get("model") == model_key and (not only_base or r["arm"] == "baseline"):
                k = (r["eval"], r["arm"], scoring._alpha(r["alpha"]), r["behavior"])
                out[k] = (lab.get(k), bool(scoring.is_degenerate_v2(r["generation"]) or r.get("early_stop", False)))
    return out


def plan(model_key, arms, doses):
    """Per prompt, the new doses worth generating : a prompt is skipped if it is already in the target state
    unsteered; otherwise only doses below its lowest known clean flip (target label, not degenerate) or its lowest known
    degenerate dose are 'needed'; doses above that point are 'rest' (kept for a final comparison). -> (needed, rest), each
    {(eval, arm, signed alpha): [prompts]}."""
    import scoring
    L = _labels(model_key); ev = evals(); needed, rest = {}, {}
    known = {}
    for (e, arm, al, p), v in L.items():
        known.setdefault((e, arm, p), {})[abs(al)] = v
    for e, (prompts, sgn) in ev.items():
        tgt = "B" if e in scoring.HARM_EVALS else "A"
        top = max(doses) if sgn < 0 else 8
        for arm in arms:
            for p in prompts:
                b = L.get((e, "baseline", 0, p))
                if b and b[0] == tgt and not b[1]:
                    continue                                              # already in the target state unsteered
                kd = known.get((e, arm, p), {})
                f = min([d for d, (l, dg) in kd.items() if l == tgt and not dg], default=np.inf)
                g = min([d for d, (l, dg) in kd.items() if dg], default=np.inf)
                cut = min(f, g)
                for d in doses:
                    if d > top or d == cut:
                        continue
                    (needed if d < cut else rest).setdefault((e, arm, sgn * d), []).append(p)
    return needed, rest


def steer_planned(model_key, doses, arm_set="sel", phase="needed"):
    """Generate the planned rows: per (eval, arm), doses ascending; in the 'needed' phase a prompt whose answer degenerates at
    one dose is dropped from the higher doses of this stage (rule-based, no judge needed).
    Rows of different cells share a batch (one steering vector per row), filled to T.GEN_BS: the 'needed' phase goes in rounds,
    round r holding the r-th dose of every (eval, arm), so the drop rule sees the lower doses first; the 'rest' phase has no
    order and is one pool. Rows are written after every batch (resume-safe per row)."""
    import scoring
    import steering as T
    import mia_quartile as Q
    z = np.load(OUT / f"vectors_{model_key}.npz"); vec = {k.split("/", 1)[1]: z[k] for k in z.files}
    old = np.load(ldir(model_key) / f"vectors_{model_key}.npz"); scale = float(old["scale"])
    vec.update({a: old[f"dir/{a}"] for a in OLD_ARMS})
    arms = sorted(k.split("/", 1)[1] for k in z.files) if arm_set == "sel" else OLD_ARMS
    doses = [scoring._alpha(d) for d in doses]
    needed, rest = plan(model_key, arms, doses)
    if phase == "needed":
        todo_map = needed
    else:                                           # 'rest' first fills any needed row still missing, then the doses above the cut
        todo_map = {k: needed.get(k, []) + rest.get(k, []) for k in set(needed) | set(rest)}
    comp = OUT / f"completions_{model_key}.jsonl"; done = set()
    for f_ in (comp, ldir(model_key) / f"completions_{model_key}.jsonl"):
        if f_.exists():
            for l in open(f_):
                r = json.loads(l); done.add((r["eval"], r["arm"], scoring._alpha(r["alpha"]), r["behavior"]))
    groups = {}
    for (e, a, al), ps in todo_map.items():
        ps = [p for p in ps if (e, a, al, p) not in done]
        if ps:
            groups.setdefault((e, a), []).append((al, ps))
    n_rows = sum(len(ps) for g in groups.values() for _, ps in g)
    print(f"[{model_key}] {phase}, {arm_set}, doses {doses}: {n_rows} rows in {sum(len(g) for g in groups.values())} cells "
          f"(a full grid would be {sum(len(ps) for m_ in (needed, rest) for ps in m_.values())} rows)", flush=True)
    if not n_rows:
        return
    for g in groups.values():
        g.sort(key=lambda c: abs(c[0]))
    rounds = ([[(e, a, *g[r]) for (e, a), g in groups.items() if r < len(g)] for r in range(max(map(len, groups.values())))]
              if phase == "needed" else [[(e, a, *c) for (e, a), g in groups.items() for c in g]])
    Q.gpu_guard()
    model, tok = T.load_model(T.MODELS_ALL[model_key]); st = T.Steerer(model, tok, layer=layer_of(model_key)); n = 0; broken = set()
    with open(comp, "a") as fh:
        for r_, cells_ in enumerate(rounds):
            rows = sorted(((e, a, al, p) for e, a, al, ps in cells_ for p in ps if (e, a, p) not in broken),
                          key=lambda r: (r[0], len(r[3])))                # similar prompt lengths share a batch: less padding
            for b in range(0, len(rows), T.GEN_BS):
                chunk = rows[b:b + T.GEN_BS]
                t0 = time.time(); texts, early = st.generate([p for *_, p in chunk],
                                                            np.stack([al * scale * vec[a] for _, a, al, _ in chunk])); n += 1
                for (e, a, al, p), t, er in zip(chunk, texts, early):
                    dg = bool(scoring.is_degenerate(t) or er)
                    fh.write(json.dumps({"model": model_key, "eval": e, "arm": a, "alpha": scoring._alpha(al), "layer": layer_of(model_key),
                                         "dose": round(al * scale, 4), "stage": 1, "gen_tokens": T.GEN_TOKENS, "behavior": p,
                                         "generation": t.strip(), "early_stop": bool(er), "degen": dg}) + "\n")
                    if phase == "needed" and (scoring.is_degenerate_v2(t) or er):
                        broken.add((e, a, p))
                fh.flush()
                print(f"[{model_key}] [{n}] round {r_ + 1}/{len(rounds)}: {len(chunk)} rows, {len({c[:3] for c in chunk})} cells, "
                      f"{b + len(chunk)}/{len(rows)} of the round ({time.time() - t0:.0f}s)", flush=True)
    st.close()


GRIDS = {"even": [2, 4, 6, 8, 10, 12], "integer": list(range(1, 13)), "half": [x / 2 for x in range(1, 25)]}


def ranges(n_boot=1000):
    """Per-prompt steering range and cost : a prompt counts as flipped by a vector if some dose of the grid
    gives a response with the benchmark's target label (B on harmful, A on benign) that is not degenerate; range = share of
    flipped prompts among those not already in the target state unsteered (+- bootstrap s.e. over prompts); cost = the lowest
    clean flip dose per flipped prompt (median, quartiles) and the share flipped by dose <= d. S_range = mean of the jailbreak half
    (HarmBench, StrongREJECT) and the over-refusal half (XSTest, OR-Bench, Alpaca). Computed on nested grids (even,
    integer, half), each only for arms whose grid is complete: a missing row counts only if it lies above the prompt's lowest
    clean flip or degenerate dose (by design of the planned runs).
    -> results/selection_steer/range_summary.json"""
    import scoring
    rng = np.random.default_rng(0); res = {}
    for m in TEMPLATE:
        if not (OUT / f"judge_{m}.jsonl").exists():
            continue
        L = _labels(m); ev = evals(); new = sorted(k.split("/", 1)[1] for k in np.load(OUT / f"vectors_{m}.npz").files)
        arms = new + OLD_ARMS; res[m] = {}
        for gname, grid in GRIDS.items():
            out = {}
            for a in arms:
                r = {}
                for e, (prompts, sgn) in ev.items():
                    tgt = "B" if e in scoring.HARM_EVALS else "A"
                    g = [d for d in grid if sgn < 0 or d <= 8]
                    flips, first, lost, complete = [], [], [], True
                    for p in prompts:
                        b = L.get((e, "baseline", 0, p))
                        if b is None or b[0] is None:
                            complete = False; break
                        if b[0] == tgt and not b[1]:
                            continue
                        f, dg_first = np.inf, np.inf
                        for d in g:
                            v = L.get((e, a, scoring._alpha(sgn * d), p))
                            if v is None or v[0] is None:                 # missing or not yet judged: a gap only if it is a needed row
                                if d < f and not any((L.get((e, a, scoring._alpha(sgn * d2), p)) or (None, False))[1] for d2 in g if d2 < d):
                                    complete = False
                                continue
                            if v[0] == tgt and not v[1]:
                                f = min(f, d)
                            if v[1]:
                                dg_first = min(dg_first, d)
                        flips.append(np.isfinite(f)); first.append(f)
                        lost.append(not np.isfinite(f) and np.isfinite(dg_first))     # never flipped cleanly, degenerated instead
                    if not complete or not flips:
                        r = None; break
                    x = np.array(flips, float); fd = np.array([d for d in first if np.isfinite(d)])
                    boot = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
                    r[e] = {"n": int(len(x)), "range": float(x.mean()), "se": float(np.std(boot)), "lost_to_degeneration": float(np.mean(lost)),
                            "first_dose_median": float(np.median(fd)) if len(fd) else None,
                            "first_dose_q": [float(np.percentile(fd, 25)), float(np.percentile(fd, 75))] if len(fd) else None,
                            "cum": {str(d): float(np.mean([np.isfinite(t) and t <= d for t in first])) for d in g}}
                if r is None:
                    continue
                # S = mean of the two halves
                harm = list(scoring.HARM_EVALS); ben = [e for e in scoring.EVALS if e not in scoring.HARM_EVALS] + ["alpaca"]
                r["jailbreak"] = float(np.mean([r[e]["range"] for e in harm]))
                r["over-refusal"] = float(np.mean([r[e]["range"] for e in ben]))
                r["S_range"] = 0.5 * (r["jailbreak"] + r["over-refusal"])
                se_h = np.sqrt(sum(r[e]["se"] ** 2 for e in harm)) / len(harm); se_b = np.sqrt(sum(r[e]["se"] ** 2 for e in ben)) / len(ben)
                r["S_se"] = float(0.5 * np.sqrt(se_h ** 2 + se_b ** 2))
                r["lost"] = float(np.mean([r[e]["lost_to_degeneration"] for e in harm + ben]))
                out[a] = r
            res[m][gname] = out
            if out:
                print(f"\n=== {m}, grid {gname} ({len(out)} arms complete): S_range | jailbreak | over-refusal | Alpaca range | median first flip dose HB / XS")
                for a, r in sorted(out.items(), key=lambda kv: -kv[1]["S_range"]):
                    md = lambda e: "-" if r[e]["first_dose_median"] is None else f"{r[e]['first_dose_median']:g}"
                    print(f"  {a:20s} {r['S_range']:.2f} | {r['jailbreak']:.2f} | {r['over-refusal']:.2f} | {r['alpaca']['range']:.2f} | {md('harmbench')} / {md('xstest_safe')}")
    json.dump(res, open(OUT / "range_summary.json", "w"), indent=1)


def paired(n_boot=2000):
    """Paired comparison with the random draws : every arm is judged on the same prompts, so a selection's S
    on the whole-number grid is compared with the mean S of the three random draws prompt by prompt; 95% interval from a
    bootstrap over prompts (resampled jointly for all arms, per benchmark). It does not include the variation between random
    draws, which is reported apart (their spread). Also the share of Tülu-mixture examples in each selection.
    -> results/selection_steer/paired_summary.json"""
    import scoring
    import steering as T
    harm = list(scoring.HARM_EVALS); ben = [e for e in scoring.EVALS if e not in scoring.HARM_EVALS] + ["alpaca"]
    grid = GRIDS["integer"]; info = json.load(open(OUT / "selections.json")); res = {}
    for m in TEMPLATE:
        if not (OUT / f"judge_{m}.jsonl").exists() or m not in info:
            continue
        L = _labels(m); ev = evals(); arms = sorted(k.split("/", 1)[1] for k in np.load(OUT / f"vectors_{m}.npz").files)
        R = json.load(open(OUT / "range_summary.json"))[m]["integer"]                    # arms complete on the whole-number grid
        arms = [a for a in arms if a in R]; rands = [a for a in (f"random{sd}" for sd in seeds(m)) if a in R]
        F = {}; n = {}
        for e, (ps, sgn) in ev.items():
            tgt = "B" if e in scoring.HARM_EVALS else "A"; g = [d for d in grid if sgn < 0 or d <= T.BENIGN_MAX_DOSE]
            el = [p for p in ps if not (L[(e, "baseline", 0, p)][0] == tgt and not L[(e, "baseline", 0, p)][1])]; n[e] = len(el)
            for a in arms:
                F[(a, e)] = np.array([any((v := L.get((e, a, scoring._alpha(sgn * d), p))) and v[0] == tgt and not v[1] for d in g) for p in el], float)
        def S(a, idx):
            jb = np.mean([F[(a, e)][idx[e]].mean() for e in harm]); orr = np.mean([F[(a, e)][idx[e]].mean() for e in ben])
            return np.array([0.5 * (jb + orr), jb, orr])
        rng = np.random.default_rng(0); full = {e: np.arange(n[e]) for e in ev}
        boots = [{e: rng.integers(0, n[e], n[e]) for e in ev} for _ in range(n_boot)]
        base = lambda idx: np.mean([S(r, idx) for r in rands], axis=0)
        bb = [base(b) for b in boots]; out = {}
        for a in arms:
            if a in rands:
                continue
            d0 = S(a, full) - base(full); db = np.array([S(a, b) - x for b, x in zip(boots, bb)])
            out[a] = {"dS": float(d0[0]), "dS_ci": [float(np.percentile(db[:, 0], 2.5)), float(np.percentile(db[:, 0], 97.5))],
                      "p_le_0": float(np.mean(db[:, 0] <= 0)), "d_jailbreak": float(d0[1]), "d_overrefusal": float(d0[2]),
                      "member_share": info[m]["member_share"][a]}
        rs = [float(S(r, full)[0]) for r in rands]
        res[m] = {"n": n, "random_S": rs, "random_spread": max(rs) - min(rs), "arms": out}
        print(f"\n=== {m}: paired dS against the mean of the random draws (S {', '.join(f'{x:.3f}' for x in rs)})")
        for a, r in sorted(out.items(), key=lambda kv: -kv[1]["dS"]):
            print(f"  {a:17s} {r['dS']:+.3f} [{r['dS_ci'][0]:+.3f}, {r['dS_ci'][1]:+.3f}]  jailbreak {r['d_jailbreak']:+.3f}  members R/C "
                  f"{r['member_share']['refuse']:.2f}/{r['member_share']['comply']:.2f}")
    json.dump(res, open(OUT / "paired_summary.json", "w"), indent=1)


GEN_ARMS = ["g_wildguardmix", "g_wildjailbreak", "g_coconot_safety"]                     # in Tülu's SFT mixture
EXT_ARMS = ["x_coconot_leftover", "x_donotanswer", "x_hh_rlhf", "x_pku_saferlhf"]


def genesis(n_boot=2000):
    """Genesis against external dataset vectors : on each model, the mean S of the three datasets of
    Tülu's mixture minus the mean S of the four external ones, on the whole-number grid, with a bootstrap over prompts
    (resampled jointly for all arms, per benchmark); the same for the matched pair CoCoNot in the mixture minus CoCoNot left
    out. Difference in differences = Tülu minus Meta-Llama, from independent bootstraps of the two models (each model's
    eligible prompts differ). CPU only. -> results/selection_steer/genesis_summary.json"""
    import scoring
    import steering as T
    harm = list(scoring.HARM_EVALS); ben = [e for e in scoring.EVALS if e not in scoring.HARM_EVALS] + ["alpaca"]
    grid = GRIDS["integer"]; arms = GEN_ARMS + EXT_ARMS; res, boots_by = {}, {}
    contrasts = {"gen_ext": (GEN_ARMS, EXT_ARMS), "coconot": (["g_coconot_safety"], ["x_coconot_leftover"])}
    rng = np.random.default_rng(0)
    for m in TEMPLATE:
        if not (OUT / f"judge_{m}.jsonl").exists():
            continue
        L = _labels(m); ev = evals(); F = {}; n = {}
        for e, (ps, sgn) in ev.items():
            tgt = "B" if e in scoring.HARM_EVALS else "A"; g = [d for d in grid if sgn < 0 or d <= T.BENIGN_MAX_DOSE]
            el = [p for p in ps if not (L[(e, "baseline", 0, p)][0] == tgt and not L[(e, "baseline", 0, p)][1])]; n[e] = len(el)
            for a in arms:
                F[(a, e)] = np.array([any((v := L.get((e, a, scoring._alpha(sgn * d), p))) and v[0] == tgt and not v[1] for d in g) for p in el], float)
        S = lambda a, idx: 0.5 * (np.mean([F[(a, e)][idx[e]].mean() for e in harm]) + np.mean([F[(a, e)][idx[e]].mean() for e in ben]))
        full = {e: np.arange(n[e]) for e in ev}
        boots = [{e: rng.integers(0, n[e], n[e]) for e in ev} for _ in range(n_boot)]
        diff = lambda c, idx: np.mean([S(a, idx) for a in contrasts[c][0]]) - np.mean([S(a, idx) for a in contrasts[c][1]])
        res[m] = {"n": n, "S": {a: float(S(a, full)) for a in arms}}; boots_by[m] = {}
        for c in contrasts:
            d0 = diff(c, full); db = np.array([diff(c, b) for b in boots]); boots_by[m][c] = db
            res[m][c] = {"d": float(d0), "se": float(np.std(db)), "ci": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]}
        print(f"[{m}] genesis - external {res[m]['gen_ext']['d']:+.3f} ± {res[m]['gen_ext']['se']:.3f}; "
              f"CoCoNot in - left out {res[m]['coconot']['d']:+.3f} ± {res[m]['coconot']['se']:.3f}")
    if {"tulu_final", "meta_instruct"} <= set(res):
        res["did"] = {}
        for c in contrasts:
            d0 = res["tulu_final"][c]["d"] - res["meta_instruct"][c]["d"]; db = boots_by["tulu_final"][c] - boots_by["meta_instruct"][c]
            res["did"][c] = {"d": float(d0), "se": float(np.std(db)), "ci": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]}
            print(f"Tülu - Meta, {c}: {d0:+.3f} ± {res['did'][c]['se']:.3f} [{res['did'][c]['ci'][0]:+.3f}, {res['did'][c]['ci'][1]:+.3f}]")
    json.dump(res, open(OUT / "genesis_summary.json", "w"), indent=1)


FULL_DS = {"WildGuardMix": "g_wildguardmix", "WildJailbreak": "g_wildjailbreak", "CoCoNot": "g_coconot_safety", "Do-Not-Answer": "x_donotanswer",
           "HH-RLHF": "x_hh_rlhf", "PKU-SafeRLHF": "x_pku_saferlhf"}


def fullcos():
    """Full-dataset vectors : per dataset, the mean of all its refusal examples in the pool minus the mean of
    all its compliance examples (prompt site, the steering layer), against layer_rerun's 128 + 128 dataset vector. CoCoNot's
    left-out rows have no compliance pole in the pool. CPU only. -> results/selection_steer/full_dataset_cos.json"""
    import mia_geometry as MG
    res = {}
    for m in [m for m in TEMPLATE if (MG.OUT / ACTS[m]).exists()]:
        F, _, _ = selections(m); X = np.load(MG.OUT / ACTS[m])[f"prompt/L{layer_of(m)}"].astype(np.float32)
        old = np.load(ldir(m) / f"vectors_{m}.npz"); res[m] = {}
        for d, arm in FULL_DS.items():
            R = [f["k"] for f in F if f["dataset"] == d and f["pole"] == "refuse"]; C = [f["k"] for f in F if f["dataset"] == d and f["pole"] == "comply"]
            v = X[R].mean(0) - X[C].mean(0)
            res[m][arm] = {"n": [len(R), len(C)], "cos": cos(v, old[f"dir/{arm}"])}
            print(f"[{m}] {d:14s} n {len(R)}/{len(C)}  cos(full, 128 + 128) {res[m][arm]['cos']:.3f}")
    json.dump(res, open(OUT / "full_dataset_cos.json", "w"), indent=1)


def topoverlap():
    """Top-K overlap between membership scores : the selections are bags of K examples, so instead of rank
    correlation, per pole of the pooled pool and per pair of scores, the share of one score's top K that is also in the
    other's (|A & B| / K); two random K-subsets share K / n. -> results/selection_steer/top_overlap.json"""
    import itertools
    import mia_panel as MP
    res = {}
    for m in [m for m in TEMPLATE if (OUT / f"judge_{m}.jsonl").exists()]:
        MP.use_template(TEMPLATE[m])
        F = [f for f in MP.features() if f["dataset"] in DATASETS and f["pole"] in ("refuse", "comply")]
        pre = ALL_SCORES[m]; allm = sorted(k for k in F[0] if k.startswith(pre) and all(k in f for f in F)); res[m] = {}
        for pole in ("refuse", "comply"):
            G = [f for f in F if f["pole"] == pole]; n = len(G)
            top = {k[len(pre):]: set(np.argsort(-np.array([f[k] for f in G]), kind="stable")[:K].tolist()) for k in allm}
            pairs = {f"{a} | {b}": len(top[a] & top[b]) / K for a, b in itertools.combinations(sorted(top), 2)}
            v = np.array(list(pairs.values()))
            res[m][pole] = {"n": n, "K": K, "chance": K / n, "n_pairs": len(v), "median": float(np.median(v)),
                            "min": float(v.min()), "max": float(v.max()), "pairs": pairs}
            print(f"[{m}] {pole}: {len(v)} pairs, median {np.median(v):.3f} ({v.min():.3f}-{v.max():.3f}), chance {K / n:.3f}")
    json.dump(res, open(OUT / "top_overlap.json", "w"), indent=1)


def steer(model_key, doses=None, arm_set="sel"):
    import scoring
    import steering as T
    import mia_quartile as Q
    z = np.load(OUT / f"vectors_{model_key}.npz"); vec = {k.split("/", 1)[1]: z[k] for k in z.files}
    old = np.load(ldir(model_key) / f"vectors_{model_key}.npz"); scale = float(old["scale"])
    vec.update({a: old[f"dir/{a}"] for a in OLD_ARMS})
    comp = OUT / f"completions_{model_key}.jsonl"; done = set()
    for f_ in (comp, ldir(model_key) / f"completions_{model_key}.jsonl"):        # the layer-16 grid already holds some reference cells
        if f_.exists():
            for l in open(f_):
                r = json.loads(l); done.add((r["eval"], r["arm"], scoring._alpha(r["alpha"])))
    new_arms = sorted(k.split("/", 1)[1] for k in z.files)
    if doses is None:
        todo = [c for c in cells(new_arms) if c not in done]
    else:
        arms = new_arms if arm_set == "sel" else OLD_ARMS
        todo = [c for c in fine_cells(arms, [scoring._alpha(d) for d in doses]) if c not in done]
    print(f"[{model_key}] layer {layer_of(model_key)}: dose scale {scale:.4f} (layer_rerun's); {len(todo)} cells to run", flush=True)
    if not todo:
        return
    Q.gpu_guard()
    model, tok = T.load_model(T.MODELS_ALL[model_key]); ev = evals(); st = T.Steerer(model, tok, layer=layer_of(model_key))
    with open(comp, "a") as fh:
        for n, (e, a, al) in enumerate(todo, 1):
            t0 = time.time(); ps = ev[e][0]
            texts, early = st.generate(ps, None if al == 0 else al * scale * vec[a])
            for p, t, er in zip(ps, texts, early):
                fh.write(json.dumps({"model": model_key, "eval": e, "arm": a, "alpha": scoring._alpha(al), "layer": layer_of(model_key), "dose": round(al * scale, 4),
                                     "stage": 1, "gen_tokens": T.GEN_TOKENS, "behavior": p, "generation": t.strip(), "early_stop": bool(er),
                                     "degen": bool(scoring.is_degenerate(t) or er)}) + "\n")
            fh.flush()
            print(f"[{model_key}] [{n}/{len(todo)}] {e} {a} {al:+g} ({time.time() - t0:.0f}s)", flush=True)
    st.close()


def _prompt_cells(dump, judge, model):
    """assumption.load_prompt_cells with Alpaca's prompts added to the benchmark index (its loader knows part 2's four)."""
    from collections import defaultdict
    import scoring
    ev = evals(); pos = {e: {p: i for i, p in enumerate(ev[e][0])} for e in ev}
    labels = {}
    for r in map(json.loads, open(judge)):
        if r.get("judge_label") in ("A", "B", "C", "D"):
            labels[(r["model"], r["eval"], r["arm"], scoring._alpha(r["alpha"]), r["behavior"])] = r["judge_label"]
    by = defaultdict(dict)
    for r in map(json.loads, open(dump)):
        if r["model"] == model:
            k = (r["model"], r["eval"], r["arm"], scoring._alpha(r["alpha"]))
            by[k][r["behavior"]] = (labels.get((*k, r["behavior"])), bool(scoring.is_degenerate_v2(r["generation"]) or r.get("early_stop", False)))
    out = {}
    for k, d in by.items():
        e = k[1]; n = len(ev[e][0]); want = "B" if e in scoring.HARM_EVALS else "A"
        hit, deg, off = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        for p, (lab, dg) in d.items():
            i = pos[e][p]; hit[i] = np.nan if lab is None else float(lab == want); deg[i] = float(dg); off[i] = np.nan if lab is None else float(lab == "C")
        out[k] = (hit, deg, off)
    return out


def summary(n_boot=200):
    """Per model and arm: reach on each benchmark at the best admissible dose (assumption.reach, degenerate share <= TAU),
    S = mean reach over part 2's four benchmarks with a paired bootstrap s.e., the jailbreak and over-refusal halves, Alpaca's
    reach on its own, and per dose the judged rate, degenerate and off-topic share. Unsteered baselines: part 2's dump for its
    four benchmarks, this run's alpha = 0 cells for Alpaca. The eleven reference arms on part 2's four benchmarks come from
    layer_rerun's layer-16 grid (results/layer16), on the same nominal doses (2..12; its extra doses 1 and 3 are not used).
    -> results/selection_steer/summary.json"""
    import assumption as A
    import scoring
    import steering as T
    res = {}
    for m in TEMPLATE:
        cf, jf = OUT / f"completions_{m}.jsonl", OUT / f"judge_{m}.jsonl"
        if not (cf.exists() and jf.exists()):
            continue
        pc = A.load_prompt_cells(T.DUMP, scoring.JUDGE, models=[m])
        pc.update(A.load_prompt_cells(ldir(m) / f"completions_{m}.jsonl", ldir(m) / f"judge_{m}.jsonl", models=[m]))   # reference arms, 4 benchmarks
        pc.update(_prompt_cells(cf, jf, m))
        rows = [json.loads(l) for l in open(cf)]
        new = sorted({r["arm"] for r in rows if r["eval"] == "harmbench"} | set(OLD_ARMS)); alp = sorted({r["arm"] for r in rows if r["eval"] == "alpaca"} - {"baseline"})
        # doses: every nominal dose present for ALL compared arms (per benchmark), so a partly finished finer grid cannot tilt the comparison
        present = {}
        for (mm, e, arm, al), c in pc.items():
            if mm == m and arm in set(new) | set(alp) and al != 0 and not np.isnan(c[0]).any():
                present.setdefault((e, arm), set()).add(abs(al))
        doses_by_e = {e: sorted(set.intersection(*[present.get((e, arm), set()) for arm in (new if e != "alpaca" else alp)]))
                      for e in list(scoring.EVALS) + ["alpaca"]}
        dose_all = sorted(set().union(*doses_by_e.values()))
        se = A.success_boot(pc, m, new, dose_all, n_boot, np.random.default_rng(0))
        out = {}
        for a in sorted(set(new) | set(alp)):
            r = {"S": None}
            if a in new:
                sc = A.success(pc, m, a, dose_all)
                if sc:
                    r.update({k: sc[k] for k in ("S", "jailbreak", "over-refusal")}); r["reach"] = {e: sc[e] for e in scoring.EVALS}
                    r["se"] = dict(zip(("S", "jailbreak", "over-refusal"), se[a])) if a in se else None
            r["alpaca_reach"] = A.reach(pc, m, a, "alpaca", doses_by_e["alpaca"])
            per = {}
            for e in list(scoring.EVALS) + ["alpaca"]:
                sign = -1 if e in scoring.HARM_EVALS else 1
                for d in sorted({abs(k[3]) for k in pc if k[0] == m and k[1] == e and k[2] == a and k[3] != 0}):
                    c = pc.get((m, e, a, sign * d))
                    if c is None or np.isnan(c[0]).any():
                        continue
                    p = float(c[0].mean())
                    per.setdefault(e, {})[str(d)] = {"rate": p, "se": scoring.se(p, len(c[0])), "degen": float(c[1].mean()), "offtopic": float(np.nanmean(c[2]))}
            r["per_dose"] = per
            r["max_degen"] = max((v["degen"] for e_ in per.values() for v in e_.values()), default=None)
            out[a] = r
        base = {e: float(pc[(m, e, "baseline", 0)][0].mean()) for e in list(scoring.EVALS) + ["alpaca"] if (m, e, "baseline", 0) in pc}
        res[m] = {"arms": out, "baseline": base, "doses_compared": {e: v for e, v in doses_by_e.items()}}
        print(f"\n=== {m}, layer {layer_of(m)}; unsteered: " + ", ".join(f"{e} {v:.2f}" for e, v in base.items()))
        print(f"{'arm':20s} {'S':>12s} {'jailbreak':>10s} {'over-ref.':>10s} " + " ".join(f"{e[:8]:>8s}" for e in scoring.EVALS) + f" {'alpaca':>8s} {'max degen':>9s}")
        for a, r in sorted(out.items(), key=lambda kv: -(kv[1]["S"] if kv[1]["S"] is not None else -1)):
            sS = "" if not r.get("se") else f"±{r['se']['S']:.2f}"
            f_ = lambda x: "    –" if x is None else f"{x:.2f}"
            print(f"{a:20s} {f_(r['S']):>7s}{sS:5s} {f_(r.get('jailbreak')):>10s} {f_(r.get('over-refusal')):>10s} "
                  + " ".join(f"{f_((r.get('reach') or {}).get(e)):>8s}" for e in scoring.EVALS) + f" {f_(r['alpaca_reach']):>8s} {f_(r['max_degen']):>9s}")
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["vectors", "steer", "cells", "summary", "plan", "ranges", "paired", "fullcos", "topoverlap", "genesis"])
    ap.add_argument("--model", choices=list(TEMPLATE))
    ap.add_argument("--doses", type=float, nargs="*", help="steer: nominal |alpha| of a finer grid (default: the original cells)")
    ap.add_argument("--arms", choices=["sel", "ref"], default="sel", help="steer with --doses: selections + controls, or the reference arms")
    ap.add_argument("--phase", choices=["needed", "rest", "all"], default="all",
                    help="steer with --doses: 'needed' = only doses below each prompt's known flip / degeneration, 'rest' = the doses above it, 'all' = every prompt")
    a = ap.parse_args()
    if a.stage == "vectors":
        vectors()
    elif a.stage == "cells":
        z = np.load(OUT / f"vectors_{a.model}.npz"); c = cells(sorted(k.split("/", 1)[1] for k in z.files))
        print(len(c), "cells;", sum(e == "alpaca" for e, _, _ in c), "on alpaca")
    elif a.stage == "summary":
        summary()
    elif a.stage == "ranges":
        ranges()
    elif a.stage == "paired":
        paired()
    elif a.stage == "fullcos":
        fullcos()
    elif a.stage == "genesis":
        genesis()
    elif a.stage == "topoverlap":
        topoverlap()
    elif a.stage == "plan":
        import scoring
        nd, rs = plan(a.model, sorted(k.split("/", 1)[1] for k in np.load(OUT / f"vectors_{a.model}.npz").files) if a.arms == "sel" else OLD_ARMS,
                      [scoring._alpha(d) for d in a.doses])
        print(f"needed {sum(map(len, nd.values()))} rows, rest {sum(map(len, rs.values()))} rows")
    elif a.doses is not None and a.phase != "all":
        steer_planned(a.model, a.doses, a.arms, a.phase)
    else:
        steer(a.model, a.doses, a.arms)
