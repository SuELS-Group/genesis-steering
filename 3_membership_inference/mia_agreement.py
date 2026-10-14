"""Agreement between membership methods, and what the agreement has to do with the training data.

    rows        the token-score panel of mia_panel.py on both models (results/mia_panel, results/mia_panel_meta), the same
                12,581 rows; CAA's A/B letter rows left out (three-token answers: Min-K and loss coincide there)
    methods     Tülu: the scores of Tülu-3-8B-SFT (the checkpoint the paper runs membership inference on), with s (twin);
                Meta: the scores of Llama-3.1-8B-Instruct (no twin); every score oriented high = more member-like
    groups      dataset x pole (the units a steering vector is drawn from), the two poles pooled over all datasets, and
                matched CoCoNot split ("mixed": members and non-members of one source)
    agreement   per group: Spearman rho of every method pair; top-quartile overlap |Q_a & Q_b| / |Q_a| (0.25 by chance);
                consensus c(x) = number of methods with x in their top quartile, and the consensus-top set (c >= a majority
                of the methods) and consensus-bottom set (in the bottom quartile of a majority)
    training    in groups where membership varies (pooled poles, matched split): member share of each method's top
                quartile, of the consensus-top and consensus-bottom sets, and of the whole group; AUC of each method and of
                the mean rank over methods (bootstrap s.e.). `member` is membership in Tülu-3's SFT mixture, so on Meta it
                is a negative control, not ground truth
    datasets    in each pooled pole: every dataset's mean score in s.d. of the pooled pole (z-scored over all rows of the
                pole), +- s.e., per method: where each dataset sits on each method's scale
    CAA         in each pooled pole: the share of CAA's authored answers among each score's top and bottom quartile and
                the consensus sets, against its share of the pole
    controls    model-free: AUC of answer length in tokens (longer = member-like), next to the methods
    cross-model Spearman rho of the same method on Tülu and on Meta, per group: a score that reads the text and not the
                model gives the same ranking on both

    python mia_agreement.py        # CPU, results/mia_agreement/summary.json
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mia_panel as MP       # noqa: E402

OUT = HERE / "results" / "mia_agreement"
DROP = {"CAA, A/B letter"}
METHODS = ["loss", "zlib", "min-k", "min-k++", "gap (CH)", "min-k gap (CH)", "top-1 agree (CH)", "ReCaLL", "s (base)", "s (twin)"]
PREFIX = {"tulu": "sft", "meta": "meta"}
SEED, N_BOOT, Q = 0, 1000, 0.25


def load(template):
    MP.use_template(template)
    F = {f["k"]: f for f in MP.features() if f["dataset"] not in DROP}
    p = PREFIX[template]
    ms = [m for m in METHODS if all(f"{p}: {m}" in f for f in F.values())]
    return F, ms, p


def ranks(x):
    from scipy.stats import rankdata
    return rankdata(x) / len(x)


def group_stats(rows, ms, p, with_member):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    X = np.array([[r[f"{p}: {m}"] for m in ms] for r in rows])          # n x methods
    n, M = X.shape
    R = np.column_stack([ranks(X[:, j]) for j in range(M)])
    top, bot = R > 1 - Q, R <= Q
    out = {"N": n, "methods": ms, "rho": {}, "top_overlap": {}}
    for a, b in combinations(range(M), 2):
        key = f"{ms[a]} | {ms[b]}"
        out["rho"][key] = float(spearmanr(X[:, a], X[:, b]).statistic)
        out["top_overlap"][key] = float((top[:, a] & top[:, b]).sum() / max(1, top[:, a].sum()))
    out["rho_mean"] = float(np.mean(list(out["rho"].values())))
    out["top_overlap_mean"] = float(np.mean(list(out["top_overlap"].values())))
    maj = M // 2 + 1
    c_top, c_bot = top.sum(1), bot.sum(1)
    out["consensus_hist"] = np.bincount(c_top, minlength=M + 1).tolist()
    out["majority"] = maj
    out["n_consensus_top"], out["n_consensus_bottom"] = int((c_top >= maj).sum()), int((c_bot >= maj).sum())
    # what a majority would give if the methods were independent: binomial tail with p = Q
    from scipy.stats import binom
    out["n_consensus_top_if_independent"] = float(n * binom.sf(maj - 1, M, Q))
    if with_member:
        y = np.array([r["member"] for r in rows])
        if 0 < y.sum() < n:
            rng = np.random.default_rng(SEED); meanrank = R.mean(1)
            def auc_se(s):
                a = roc_auc_score(y, s); bs = []
                for _ in range(N_BOOT):
                    i = rng.integers(0, n, n)
                    if 0 < y[i].sum() < n:
                        bs.append(roc_auc_score(y[i], s[i]))
                return [float(a), float(np.std(bs, ddof=1))]
            share = lambda mask: [float(y[mask].mean()) if mask.any() else float("nan"), int(mask.sum())]
            out["member"] = {"base_rate": share(np.ones(n, bool)),
                             "top_quartile": {ms[j]: share(top[:, j]) for j in range(M)},
                             "consensus_top": share(c_top >= maj), "consensus_bottom": share(c_bot >= maj),
                             "auc": {**{ms[j]: auc_se(X[:, j]) for j in range(M)}, "mean rank": auc_se(meanrank),
                                     "length (longer)": auc_se(np.array([r["n"] for r in rows], float))}}
    if with_member and "dataset" in rows[0]:
        Z = (X - X.mean(0)) / X.std(0, ddof=1); ds = np.array([r["dataset"] for r in rows])
        caa = np.char.startswith(ds.astype(str), "CAA")
        if caa.any() and not caa.all():                 # where the authored CAA answers land in each score's ranking
            out["caa_share"] = {"base": [float(caa.mean()), int(caa.sum())],
                                **{ms[j]: {"top": float(caa[top[:, j]].mean()), "bottom": float(caa[bot[:, j]].mean())} for j in range(M)},
                                "consensus": {"top": float(caa[c_top >= maj].mean()), "bottom": float(caa[c_bot >= maj].mean())}}
        out["dataset_z"] = {d: {ms[j]: [float(Z[ds == d, j].mean()), float(Z[ds == d, j].std(ddof=1) / np.sqrt((ds == d).sum()))]
                                for j in range(M)} | {"N": int((ds == d).sum())} for d in sorted(set(ds))}
    return out


def main():
    T, mt, pt = load("tulu")
    E, me, pe = load("meta")
    ks = sorted(set(T) & set(E))
    groups = {}
    for k in ks:
        r = T[k]
        groups.setdefault(f"{r['dataset']} | {r['pole']}", []).append(k)
        if r["pole"] in ("refuse", "comply"):
            groups.setdefault(f"all datasets | {r['pole']}", []).append(k)
    res = {"methods": {"tulu": mt, "meta": me}, "groups": {}}
    from scipy.stats import spearmanr
    for g, gk in sorted(groups.items()):
        pooled_or_mixed = g.startswith("all datasets") or g.endswith("| mixed")
        res["groups"][g] = {
            "tulu": group_stats([T[k] for k in gk], mt, pt, pooled_or_mixed),
            "meta": group_stats([E[k] for k in gk], me, pe, pooled_or_mixed),
            "cross_model_rho": {m: float(spearmanr([T[k][f"{pt}: {m}"] for k in gk], [E[k][f"{pe}: {m}"] for k in gk]).statistic)
                                for m in me if m in mt}}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)
    for g, d in res["groups"].items():
        t, e = d["tulu"], d["meta"]
        print(f"{g:34s} N={t['N']:5d}  rho {t['rho_mean']:+.2f} / {e['rho_mean']:+.2f}  top-overlap {t['top_overlap_mean']:.2f} / {e['top_overlap_mean']:.2f}"
              f"  consensus-top {t['n_consensus_top']} (indep. {t['n_consensus_top_if_independent']:.0f}) / {e['n_consensus_top']}")


if __name__ == "__main__":
    main()
