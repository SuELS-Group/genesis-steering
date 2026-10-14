"""Which membership score is most accurate where membership is known ? WildGuardMix rows in Tülu-3's SFT
mixture (members) against WildGuardMix training rows that are not in it (non-members, decontaminated; mia_quartile.pools),
both scored on WildGuardMix's own release response, so the text's generator is the same for both groups.

    scores      per token on the gold answer (mia_quartile spans, src 'row'), high = more member-like:
                  loss           -NLL_target
                  zlib           -NLL_target / zlib-compressed bytes of the answer (the Imperial runner's definition)
                  s (base)       NLL_base - NLL_target      (Tülu-3-8B final vs Llama-3.1-8B; Meta-Llama vs the same base)
                  s (twin)       NLL_twin - NLL_SFT         (Tülu-3-8B-SFT vs its no-safety-data twin; Tülu only)
                controls: answer length (-tokens) and zlib ratio of the text alone (model-free), and every score on Meta-Llama,
                which was not trained on these rows (expected AUC 0.5)
    metrics     per pole: AUC (members positive) with a 1,000-sample bootstrap 95 % interval, TPR at 5 % FPR; and the AUC within
                answer-length deciles (pooled over deciles, weighted by non-member count): the part not explained by length

    python mia_accuracy.py         # CPU, writes results/mia_accuracy.json
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results" / "mia_quartile" / "wildguardmix"
ROWS = HERE / "data" / "mia_quartile" / "wildguardmix" / "rows.jsonl"
SEED, N_BOOT = 0, 1000


def load():
    sp = {m: {(r["pole"], r["i"]): r for r in map(json.loads, open(R / f"spans_{m}.jsonl"))} for m in ("final", "base", "sft", "twin", "meta", "meta_base")}
    rows = [r for r in map(json.loads, open(ROWS))]
    out = []
    for r in rows:
        k = (r["pole"], r["i"])
        if not all(k in sp[m] and sp[m][k]["n_tok_resp"] for m in sp):
            continue
        pt = lambda m: sp[m][k]["nll_resp"] / sp[m][k]["n_tok_resp"]
        zb = len(zlib.compress(r["gold"].encode()))
        out.append({"pole": r["pole"], "member": r["kind"] == "release", "n": sp["final"][k]["n_tok_resp"], "zlib_bytes": zb,
                    "tulu": {"loss": -pt("final"), "zlib": -pt("final") * sp["final"][k]["n_tok_resp"] / zb, "s (base)": pt("base") - pt("final"),
                             "s (twin)": pt("twin") - pt("sft")},
                    "meta": {"loss": -pt("meta"), "zlib": -pt("meta") * sp["meta"][k]["n_tok_resp"] / zb, "s (base)": pt("meta_base") - pt("meta")},
                    "text": {"length (shorter)": -sp["final"][k]["n_tok_resp"], "zlib ratio": len(r["gold"].encode()) / zb}})
    return out


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def tpr_at(y, s, fpr=0.05):
    from sklearn.metrics import roc_curve
    f, t, _ = roc_curve(y, s)
    return float(np.interp(fpr, f, t))


def length_matched_auc(y, s, n):
    """AUC within answer-length deciles (decile edges from all rows), pooled weighted by the number of non-members per decile."""
    edges = np.quantile(n, np.linspace(0, 1, 11)[1:-1]); b = np.digitize(n, edges); num = den = 0.0
    for d in np.unique(b):
        m = b == d
        if y[m].sum() >= 5 and (1 - y[m]).sum() >= 5:
            w = (1 - y[m]).sum(); num += w * auc(y[m], s[m]); den += w
    return num / den if den else float("nan")


def main():
    rows = load(); rng = np.random.default_rng(SEED); out = {}
    for pole in ("refuse", "comply"):
        P = [r for r in rows if r["pole"] == pole]
        y = np.array([r["member"] for r in P], dtype=int); n = np.array([r["n"] for r in P])
        res = out.setdefault(pole, {"members": int(y.sum()), "non_members": int((1 - y).sum())})
        for grp in ("tulu", "meta", "text"):
            for name in P[0][grp]:
                s = np.array([r[grp][name] for r in P])
                boots = []
                for _ in range(N_BOOT):
                    i = rng.integers(0, len(y), len(y))
                    if 0 < y[i].sum() < len(i):
                        boots.append(auc(y[i], s[i]))
                res[f"{grp}: {name}"] = {"auc": auc(y, s), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                         "tpr@5%fpr": tpr_at(y, s), "auc_length_matched": length_matched_auc(y, s, n)}
    json.dump(out, open(HERE / "results" / "mia_accuracy.json", "w"), indent=1)
    for pole, d in out.items():
        print(f"\n{pole}: {d['members']} members vs {d['non_members']} non-members")
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"  {k:24s} AUC {v['auc']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]  TPR@5% {v['tpr@5%fpr']:.3f}  length-matched AUC {v['auc_length_matched']:.3f}")


if __name__ == "__main__":
    main()
