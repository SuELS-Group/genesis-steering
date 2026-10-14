"""Geometry of the token-level membership scores : what the top and bottom thirds by each score look like
in activation space, per dataset and across datasets, on Tülu-3-8B final.

    rows        the 12,581 rows of 3_membership_inference/mia_panel.py (every row of part 2's pools, CAA's generate pairs,
                CoCoNot's matched split), scored there by gap, Min-K %++, loss, zlib, s (twin), s (base), ReCaLL
    activations one teacher-forced pass per row on Tülu-3-8B final: BOS + the row rendered with Tülu's chat template
                (part 2 extracts with the tokenizer adding BOS), hidden_states[13] and hidden_states[16] (inputs of
                blocks 13 and 16) at two sites:
                  prompt  the last prompt token (the token before the answer), part 2's steering recipe
                  answer  the mean over the answer's tokens (closing end-of-text excluded), caa_mean's recipe
                CAA's poles share their prompt, so its prompt-site vector is zero by construction; CAA uses the answer site.
    thirds      per dataset, pole and score: the top and bottom third of rows (rank within that dataset and pole);
                vector = mean(refusal rows) - mean(compliance rows) of the same dataset, both poles cut by the same score

    python mia_geometry.py acts       # GPU, results/mia_geometry/acts_tulu_final.npz
    python mia_geometry.py analyze    # CPU, results/mia_geometry/geometry.json

    --model meta_instruct : the same rows, layers and sites on Llama-3.1-8B-Instruct, rendered with its own
                chat template (mia_panel --template meta) and, as part 2 extracts on this model, with the tokenizer's BOS put in
                front of a template that already writes one; thirds by the Meta panel's scores (no s (twin))
                -> results/mia_geometry/acts_meta_instruct.npz, geometry_meta_instruct.json
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
OUT = HERE / "results" / "mia_geometry"
ACTS = OUT / "acts_tulu_final.npz"
LAYERS = (13, 16)
MODEL = "allenai/Llama-3.1-Tulu-3-8B"
TOK_ID, TEMPLATE, GEOM = "allenai/Llama-3.1-Tulu-3-8B-SFT", "tulu", OUT / "geometry.json"


def use_model(name):
    global ACTS, MODEL, TOK_ID, TEMPLATE, GEOM, SCORES
    if name == "meta_instruct":
        ACTS, MODEL, TOK_ID, TEMPLATE = OUT / "acts_meta_instruct.npz", "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "meta"
        GEOM = OUT / "geometry_meta_instruct.json"
        SCORES = {k: v.replace("sft: ", "meta: ").replace("final: ", "meta: ") for k, v in SCORES.items() if k != "s (twin)"}
    if name == "qwen":                               # the third model, at the layer its sweep chose
        global LAYERS
        ACTS, MODEL, TOK_ID, TEMPLATE = OUT / "acts_qwen.npz", "Qwen/Qwen3-8B", "Qwen/Qwen3-8B", "qwen"
        GEOM = OUT / "geometry_qwen.json"
        LAYERS = (json.load(open(HERE / "results" / "qwen" / "sweep_summary.json"))["chosen_layer"],)
        SCORES = {k: v.replace("sft: ", "qwen: ").replace("final: ", "qwen: ") for k, v in SCORES.items() if k != "s (twin)"}


def acts():
    if ACTS.exists():
        print("acts exist"); return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import mia_panel as P
    import mia_quartile as Q
    Q.gpu_guard(); t0 = time.time()
    P.use_template(TEMPLATE); IT = P.items(); tok = AutoTokenizer.from_pretrained(TOK_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    bos = tok.bos_token_id                           # None for Qwen3: no BOS is prepended (its tokenizer adds none either)
    pre = [] if bos is None else [bos]
    enc = [P.encode(tok, it) for it in IT]
    order = sorted([k for k in range(len(IT)) if enc[k]], key=lambda k: len(enc[k][0]))
    out = {f"{s}/L{l}": np.zeros((len(IT), model.config.hidden_size), np.float16) for s in ("prompt", "answer") for l in LAYERS}
    done = np.zeros(len(IT), bool); pad = tok.eos_token_id
    with torch.inference_mode():
        for b in range(0, len(order), 4):
            ks = order[b:b + 4]; seqs = [pre + enc[k][0] for k in ks]; L = max(map(len, seqs))
            ids = torch.full((len(ks), L), pad, dtype=torch.long); am = torch.zeros((len(ks), L), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s); am[j, :len(s)] = 1
            hs = model.model(input_ids=ids.cuda(), attention_mask=am.cuda(), output_hidden_states=True).hidden_states
            for j, k in enumerate(ks):
                a = enc[k][1] + len(pre); n = len(seqs[j])       # + the BOS, if any; answer tokens a .. n-2, end-of-text at n-1
                for l in LAYERS:
                    h = hs[l][j]
                    out[f"prompt/L{l}"][k] = h[a - 1].float().cpu().numpy()
                    out[f"answer/L{l}"][k] = h[a:max(a + 1, n - 1)].float().mean(0).cpu().numpy()
                done[k] = True
            del hs
            if b % 400 == 0:
                print(f"  {b}/{len(order)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(ACTS, done=done, **out)
    print(f"{done.sum()} rows in {(time.time() - t0) / 60:.1f} min", flush=True)


SCORES = {"gap": "sft: gap (CH)", "min-k++": "sft: min-k++", "loss": "sft: loss", "zlib": "sft: zlib", "ReCaLL": "sft: ReCaLL",
          "s (twin)": "sft: s (twin)", "s (base)": "final: s (base)"}
PAIRED = ["WildGuardMix", "WildJailbreak", "CoCoNot", "Do-Not-Answer", "HH-RLHF", "PKU-SafeRLHF", "CAA, answers (mean)"]
SFT_DATA = ("WildGuardMix", "WildJailbreak", "CoCoNot")            # Tülu's SFT data (CoCoNot: refusals in the mixture)
N_RANDOM, SEED = 50, 0
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _thirds(score):
    o = np.argsort(score, kind="stable"); n = len(score) // 3
    return o[-n:], o[:n]


def analyze():
    """Per site and layer, per score and dataset: top / bottom third vectors against the full vector, against each other,
    against random thirds, across datasets, toward Tülu's SFT data, and the selection's displacement of each pole."""
    import mia_panel as P
    P.use_template(TEMPLATE); Z = np.load(ACTS); F = P.features(); rng = np.random.default_rng(SEED)
    scores = {k: v for k, v in SCORES.items() if all(v in f for f in F)}
    rows = {}                                                       # (dataset, pole) -> (row ids, {score: values})
    for f in F:
        if f["dataset"] in PAIRED and f["pole"] in ("refuse", "comply"):
            r = rows.setdefault((f["dataset"], f["pole"]), ([], {s: [] for s in scores}))
            r[0].append(f["k"])
            for s, key in scores.items():
                r[1][s].append(f[key])
    rows = {g: (np.array(ix), {s: np.array(v) for s, v in d.items()}) for g, (ix, d) in rows.items()}
    res = {"N": {f"{d} | {p}": int(len(rows[(d, p)][0])) for d, p in rows}, "scores": list(scores), "sites": {}}
    for site in ("prompt", "answer"):
        for l in LAYERS:
            X = Z[f"{site}/L{l}"].astype(np.float32); key = f"{site}/L{l}"
            ds = [d for d in PAIRED if not (site == "prompt" and d.startswith("CAA"))]
            mean = lambda d, p, sel=None: X[rows[(d, p)][0] if sel is None else rows[(d, p)][0][sel]].mean(0)
            full = {d: mean(d, "refuse") - mean(d, "comply") for d in ds}
            out = {"full_norm": {d: float(np.linalg.norm(full[d])) for d in ds}, "per_score": {}}
            # random thirds: per dataset, N_RANDOM draws of a third of each pole
            rand, rdp = {d: [] for d in ds}, {d: [] for d in ds}
            for d in ds:
                nR, nC = len(rows[(d, "refuse")][0]), len(rows[(d, "comply")][0])
                u = full[d] / np.linalg.norm(full[d]); pR, pC = X[rows[(d, "refuse")][0]] @ u, X[rows[(d, "comply")][0]] @ u
                sd = np.sqrt((pR.var(ddof=1) + pC.var(ddof=1)) / 2)
                for _ in range(N_RANDOM):
                    jR, jC = rng.choice(nR, nR // 3, replace=False), rng.choice(nC, nC // 3, replace=False)
                    rand[d].append(mean(d, "refuse", jR) - mean(d, "comply", jC)); rdp[d].append((pR[jR].mean() - pC[jC].mean()) / sd)
            band = lambda v: [float(np.percentile(v, q)) for q in (2.5, 50, 97.5)]
            out["random"] = {d: {"cos_full": band([cos(v, full[d]) for v in rand[d]]), "norm": band([np.linalg.norm(v) for v in rand[d]]),
                                 "dprime": band(rdp[d])} for d in ds}
            ref = full["WildGuardMix"]
            for s in scores:
                per, vt, vb = {}, {}, {}
                for d in ds:
                    (iR, sR), (iC, sC) = (rows[(d, "refuse")][0], rows[(d, "refuse")][1][s]), (rows[(d, "comply")][0], rows[(d, "comply")][1][s])
                    (tR, bR), (tC, bC) = _thirds(sR), _thirds(sC)
                    vt[d] = mean(d, "refuse", tR) - mean(d, "comply", tC); vb[d] = mean(d, "refuse", bR) - mean(d, "comply", bC)
                    u = full[d] / np.linalg.norm(full[d])
                    pR, pC = X[iR] @ u, X[iC] @ u                     # projection of every row on the dataset's own axis
                    sd = np.sqrt((pR.var(ddof=1) + pC.var(ddof=1)) / 2)
                    dR = mean(d, "refuse", tR) - mean(d, "refuse", bR); dC = mean(d, "comply", tC) - mean(d, "comply", bC)
                    from scipy.stats import spearmanr
                    per[d] = {"cos_top_full": cos(vt[d], full[d]), "cos_bottom_full": cos(vb[d], full[d]), "cos_top_bottom": cos(vt[d], vb[d]),
                              "norm_top": float(np.linalg.norm(vt[d])), "norm_bottom": float(np.linalg.norm(vb[d])),
                              "dprime_top": float((pR[tR].mean() - pC[tC].mean()) / sd), "dprime_bottom": float((pR[bR].mean() - pC[bC].mean()) / sd),
                              "dprime_full": float((pR.mean() - pC.mean()) / sd),
                              "shift_refuse_on_axis": float((pR[tR].mean() - pR[bR].mean()) / sd),     # top minus bottom, in pooled s.d.
                              "shift_comply_on_axis": float((pC[tC].mean() - pC[bC].mean()) / sd),
                              "cos_shift_refuse_comply": cos(dR, dC),                                  # same displacement for both poles?
                              "rho_score_proj_refuse": float(spearmanr(sR, pR)[0]), "rho_score_proj_comply": float(spearmanr(sC, pC)[0]),
                              "cos_top_wgm": cos(vt[d], ref), "cos_bottom_wgm": cos(vb[d], ref), "cos_full_wgm": cos(full[d], ref)}
                pair = lambda V: float(np.mean([cos(V[a], V[b]) for i, a in enumerate(ds) for b in ds[i + 1:]]))
                ext = [d for d in ds if d not in SFT_DATA]
                out["per_score"][s] = {"per_dataset": per, "mean_pairwise_cos": {"top": pair(vt), "bottom": pair(vb), "full": pair(full)},
                                       "cos_matrix_top": [[cos(vt[a], vt[b]) for b in ds] for a in ds],
                                       "cos_matrix_bottom": [[cos(vb[a], vb[b]) for b in ds] for a in ds],
                                       "externals_to_sft": {t: float(np.mean([cos(V[e], full[m]) for e in ext for m in SFT_DATA if m in ds]))
                                                            for t, V in (("top", vt), ("bottom", vb), ("full", full))}}
            rp = [float(np.mean([cos(rand[a][i], rand[b][i]) for i_, a in enumerate(ds) for b in ds[i_ + 1:]])) for i in range(N_RANDOM)]
            out["random_mean_pairwise_cos"] = [float(np.percentile(rp, q)) for q in (2.5, 50, 97.5)]
            out["datasets"] = ds; res["sites"][key] = out
            print(f"[{key}] done", flush=True)
    json.dump(res, open(GEOM, "w"), indent=1)
    _report(res)


def _report(res):
    for key, out in res["sites"].items():
        print(f"\n=== {key}: datasets {', '.join(out['datasets'])}")
        print(f"mean pairwise cosine across datasets: full {out['per_score'][res['scores'][0]]['mean_pairwise_cos']['full']:.3f}, "
              f"random thirds {out['random_mean_pairwise_cos'][1]:.3f} [{out['random_mean_pairwise_cos'][0]:.3f}, {out['random_mean_pairwise_cos'][2]:.3f}]")
        print(f"{'score':10s} {'pairwise top/bot':>17s} {'ext->SFT top/bot/full':>23s} {'cos(top,full)':>14s} {'cos(bot,full)':>14s} "
              f"{'dprime top/bot/full':>21s} {'shift R/C on axis':>18s} {'cos(shiftR,shiftC)':>18s} {'rho score~proj R/C':>19s}")
        for s, d in out["per_score"].items():
            P_ = d["per_dataset"]; m = lambda q: np.mean([v[q] for v in P_.values()])
            print(f"{s:10s} {d['mean_pairwise_cos']['top']:8.3f}/{d['mean_pairwise_cos']['bottom']:.3f} "
                  f"{d['externals_to_sft']['top']:9.3f}/{d['externals_to_sft']['bottom']:.3f}/{d['externals_to_sft']['full']:.3f} "
                  f"{m('cos_top_full'):14.3f} {m('cos_bottom_full'):14.3f} {m('dprime_top'):9.2f}/{m('dprime_bottom'):.2f}/{m('dprime_full'):.2f} "
                  f"{m('shift_refuse_on_axis'):+9.2f}/{m('shift_comply_on_axis'):+.2f} {m('cos_shift_refuse_comply'):18.3f} "
                  f"{m('rho_score_proj_refuse'):+9.2f}/{m('rho_score_proj_comply'):+.2f}")
        print("random-third cos(v, full), median per dataset: " + ", ".join(f"{d} {v['cos_full'][1]:.3f}" for d, v in out["random"].items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["acts", "analyze"])
    ap.add_argument("--model", choices=["tulu_final", "meta_instruct", "qwen"], default="tulu_final")
    a = ap.parse_args(); use_model(a.model)
    {"acts": acts, "analyze": lambda: analyze()}[a.stage]()
