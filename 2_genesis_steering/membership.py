"""Members against non-members of ONE dataset: WildGuardMix rows in Tülu's SFT mixture vs those left out.

The genesis claim in its cleanest form: from the same dataset and generation process, do vectors built from rows Tülu
was trained on steer Tülu better than vectors built from rows it was not trained on, and not Meta-Llama (trained on
neither)? Tülu-3's WildGuardMix slice is a decontaminated 50k sample of WildGuardMix's training split; the rest of the
split is the non-member pool.

Pools (both by part 2's rule: hub label harmful + response opens as a refusal -> refuse; hub label benign + clean
compliance -> comply; poles.py's prompt filter, de-duplication)
    member      the mixture's WildGuardMix rows (registry._build_from_mixture), pole from Tülu's gold turn
    non-member  WildGuardMix-train rows whose prompt appears nowhere in the Tülu mixture and that carry a response,
                pole from WildGuardMix's own response (Tülu regenerated the responses of its slice)
    decontamination   Tülu's own decontamination is part of what made rows non-members, so non-members are enriched for
                prompts resembling evaluation sets. Both pools drop every prompt sharing a word 8-gram with any of the
                400 evaluation prompts of data/evals.
Draws   4 seeds per arm, 128 + 128, stratified by the adversarial flag: within each pole both arms take the non-member
        pool's adversarial share, so the arms differ in membership, not in that composition.
Grid    Tülu-3-8B and Meta-Llama-3.1-8B-Instruct, part 2's recipe (layer 13, unit norm, 256 tokens, judge), all 100
        prompts: HarmBench and StrongREJECT at -alpha = 4 / 6 / 8, XSTest-safe at +alpha = 2 / 4 (over-refusal
        saturates beyond). The unsteered cells are part 2's (reproduced exactly).

    python membership.py build                       # CPU
    python membership.py steer --model tulu_final    # GPU
    python ../judge.py --dump results/membership/steer_completions.jsonl --out results/membership/steer_judge.jsonl
Never commit results/membership/steer_*.jsonl.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

import poles as P
import steering as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "1_models_and_datasets"))
DATA = HERE / "data" / "membership"
OUT = HERE / "results" / "membership"
STEER_COMP, STEER_JUDGE = OUT / "steer_completions.jsonl", OUT / "steer_judge.jsonl"
VECS = OUT / "vectors.npz"
ARMS2, SEEDS, POLES2 = ("member", "nonmember"), [0, 1, 2, 3], ("refuse", "comply")
GRID = {"harmbench": [-4, -6, -8], "strongreject": [-4, -6, -8], "xstest_safe": [2, 4]}
NGRAM = 8


def _grams(s, n=NGRAM):
    w = re.findall(r"\w+", s.lower())
    return {tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


def build():
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    import registry as REG
    import pyarrow.dataset as pds
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    EG = set().union(*[_grams(p) for ps, _ in T.load_evals().values() for p in ps])
    root = snapshot_download(REG.MIXTURE, repo_type="dataset", allow_patterns=["data/*.parquet"])
    msgs = pds.dataset(sorted(glob.glob(f"{root}/data/*.parquet")), format="parquet").to_table(columns=["messages"]).column("messages").to_pylist()
    in_mix = {REG._norm(m[0]["content"]) for m in msgs if m and m[0]["role"] == "user"}
    hub = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    adv = {REG._norm(x["prompt"]): bool(x["adversarial"]) for x in hub}
    ref, com = REG._build_from_mixture("wildguardmix")
    pools = {("member", "refuse"): [r["prompt"] for r in ref], ("member", "comply"): [r["prompt"] for r in com],
             ("nonmember", "refuse"): [], ("nonmember", "comply"): []}
    for x in hub:
        if REG._norm(x["prompt"]) in in_mix or not x["response"]:
            continue
        harmful = x["prompt_harm_label"] == "harmful"
        if harmful and P.is_refusal(x["response"]):
            pools[("nonmember", "refuse")].append(x["prompt"])
        elif not harmful and x["prompt_harm_label"] == "unharmful" and P.is_clean_compliance(x["response"]):
            pools[("nonmember", "comply")].append(x["prompt"])
    DATA.mkdir(parents=True, exist_ok=True); summary = {}
    for (arm, pole), ps in pools.items():
        seen, out, n_cont = set(), [], 0
        for p in ps:
            p = (p or "").strip()
            if not p or not P._keep(p) or P._norm(p) in seen:
                continue
            seen.add(P._norm(p))
            if _grams(p) & EG:
                n_cont += 1; continue
            out.append({"prompt": p, "adversarial": adv.get(P._norm(p), False)})
        with open(DATA / f"{arm}_{pole}.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[f"{arm}/{pole}"] = {"kept": len(out), "dropped as eval-overlapping": n_cont,
                                    "adversarial share": float(np.mean([r["adversarial"] for r in out]))}
    json.dump(summary, open(DATA / "_summary.json", "w"), indent=1)
    draws(); print(json.dumps(summary, indent=1))


def pool(arm, pole):
    return [json.loads(l) for l in open(DATA / f"{arm}_{pole}.jsonl")]


def draws():
    """4 seeds per arm, 128 per pole, the adversarial share of the non-member pool in both arms."""
    (DATA / "draws").mkdir(exist_ok=True)
    for pole_i, pole in enumerate(POLES2):
        share = np.mean([r["adversarial"] for r in pool("nonmember", pole)])
        n_adv = int(round(share * P.N_SRC))
        for arm_i, arm in enumerate(ARMS2):
            pl = pool(arm, pole)
            A = [r["prompt"] for r in pl if r["adversarial"]]; Nn = [r["prompt"] for r in pl if not r["adversarial"]]
            for s in SEEDS:
                rng = np.random.default_rng(10000 + 100 * s + 10 * arm_i + pole_i)
                pick = list(rng.choice(A, n_adv, replace=False)) + list(rng.choice(Nn, P.N_SRC - n_adv, replace=False))
                with open(DATA / "draws" / f"{arm}_s{s}.jsonl", "a" if pole_i else "w") as f:
                    for p in pick:
                        f.write(json.dumps({"pole": pole, "prompt": str(p)}, ensure_ascii=False) + "\n")


def arms():
    return [f"wgm_{arm}_s{s}" for arm in ARMS2 for s in SEEDS]


def load_draw(arm):
    rows = [json.loads(l) for l in open(DATA / "draws" / f"{arm.replace('wgm_', '')}.jsonl")]
    return {pole: [r["prompt"] for r in rows if r["pole"] == pole] for pole in POLES2}


def cells():
    return [(e, a, al) for a in arms() for e, ds in GRID.items() for al in ds]


def steer(model_key):
    import behaviour as B
    todo = [c for c in cells() if (model_key, *c) not in B._done(STEER_COMP)]
    if not todo:
        print(f"[{model_key}] grid complete"); return
    model, tok = T.load_model(T.MODELS[model_key])
    V = dict(np.load(VECS)) if VECS.exists() else {}
    for a in arms():
        if f"{model_key}/dir/{a}" not in V:
            d = load_draw(a)
            raw = T.group_mean(model, tok, d["refuse"]) - T.group_mean(model, tok, d["comply"])
            V[f"{model_key}/raw/{a}"] = raw.astype(np.float32); V[f"{model_key}/dir/{a}"] = (raw / np.linalg.norm(raw)).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True); np.savez(VECS, **V)
    ev = T.load_evals(); st = T.Steerer(model, tok)
    with open(STEER_COMP, "a") as fh:
        for n, (e, a, al) in enumerate(todo, 1):
            t0 = time.time(); ps = ev[e][0]
            texts, early = st.generate(ps, al * V[f"{model_key}/dir/{a}"])
            B._write(fh, model_key, e, a, al, ps, texts, early)
            print(f"[{model_key}] [{n}/{len(todo)}] {e} {a} {al:+d} ({time.time() - t0:.0f}s)", flush=True)
    st.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "steer"])
    ap.add_argument("--model", choices=list(T.MODELS))
    a = ap.parse_args()
    {"build": build, "steer": lambda: steer(a.model)}[a.step]()
