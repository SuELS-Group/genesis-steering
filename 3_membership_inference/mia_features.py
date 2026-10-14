"""What the membership scores pick up: surface features of the text, or more (, , paper item: "is it register,
length, or something more complex?").

    rows        the token panel of mia_panel.py (both templates, the same rows): the pooled refusal and compliance poles (all
                datasets and CAA's authored answers; CAA's letter rows and the matched split left out) and the matched CoCoNot split (one source, members against non-members)
    features    model-free, per row: log response length in tokens, log prompt length in characters, the response's zlib
                ratio (compressed / raw bytes: low = repetitive or templated), and whether the response opens with a refusal
                phrase (REFUSAL_OPEN on its first 120 characters)
    surface fit per score, model and group: ordinary least squares of the score on the four standardised features; R^2, the
                standardised coefficients, and the score's residual
    membership  per group where membership varies: AUC (members positive) of the score, of its residual, and of the features
                alone (5-fold cross-validated logistic regression, seed 0), with bootstrap s.e.; `member` = membership in
                Tülu-3's SFT mixture (on Llama-3.1-8B-Instruct a negative control)
    CAA pairs   per score and model, over CAA's question pairs: score of the refusal answer minus score of the compliance
                answer, mean +- s.e. and the share of pairs where the refusal scores higher (neither answer is training data)

    python mia_features.py         # CPU, results/mia_features/summary.json
"""
from __future__ import annotations

import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mia_agreement as A    # noqa: E402  (load(): the panel rows and methods per template)

OUT = HERE / "results" / "mia_features"
REFUSAL_OPEN = re.compile(r"\b(sorry|apolog|i can(?:no|')t|i cannot|i'm (?:not able|unable)|i am (?:not able|unable)|unable to|"
                          r"i won't|i will not|as an ai|i must decline|i'm afraid|not (?:appropriate|able to))", re.I)
FEATURES = ["log response tokens", "log prompt chars", "zlib ratio", "refusal opening"]
SEED, N_BOOT = 0, 1000


def features(r, item):
    a = item["answer"].encode()
    return [np.log(max(1, r["n"])), np.log(max(1, len(item["prompt"]))), len(zlib.compress(a)) / max(1, len(a)),
            float(bool(REFUSAL_OPEN.search(item["answer"][:120])))]


def auc_se(y, s, rng):
    from sklearn.metrics import roc_auc_score
    a = roc_auc_score(y, s); bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            bs.append(roc_auc_score(y[i], s[i]))
    return [float(a), float(np.std(bs, ddof=1))]


def main():
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    import mia_panel as MP
    items = MP.items(); res = {"features": FEATURES, "groups": {}, "caa_pairs": {}}
    rng = np.random.default_rng(SEED)
    for template, model in (("tulu", "tulu"), ("meta", "meta")):
        F, ms, p = A.load(template)
        groups = {"refusal, pooled": [k for k, f in F.items() if f["pole"] == "refuse"],
                  "compliance, pooled": [k for k, f in F.items() if f["pole"] == "comply"],
                  "CoCoNot, matched": [k for k, f in F.items() if f["pole"] == "mixed"]}
        for g, ks in groups.items():
            X = np.array([features(F[k], items[k]) for k in ks]); X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
            y = np.array([F[k]["member"] for k in ks]); out = {"N": len(ks), "per_score": {}}
            if 0 < y.sum() < len(y):
                cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
                pr = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=cv, method="predict_proba")[:, 1]
                out["features_only_auc"] = auc_se(y, pr, rng)
            for m in ms:
                s = np.array([F[k][f"{p}: {m}"] for k in ks]); lr = LinearRegression().fit(X, s); resid = s - lr.predict(X)
                d = {"r2": float(lr.score(X, s)), "coef_std": dict(zip(FEATURES, (lr.coef_ / s.std()).round(4).tolist()))}
                if 0 < y.sum() < len(y):
                    d["auc"] = auc_se(y, s, rng); d["auc_residual"] = auc_se(y, resid, rng)
                out["per_score"][m] = d
            res["groups"].setdefault(g, {})[model] = out
        # CAA pairs: refusal answer minus compliance answer to the same question
        caa = {}
        for k, f in F.items():
            if f["dataset"].startswith("CAA"):
                caa.setdefault(items[k]["prompt"], {})[f["pole"]] = k
        pairs = [(v["refuse"], v["comply"]) for v in caa.values() if "refuse" in v and "comply" in v]
        res["caa_pairs"][model] = {"N": len(pairs)}
        for m in ms:
            d = np.array([F[r][f"{p}: {m}"] - F[c][f"{p}: {m}"] for r, c in pairs])
            res["caa_pairs"][model][m] = {"diff": [float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))],
                                          "share_refusal_higher": [float((d > 0).mean()), float(np.sqrt((d > 0).mean() * (1 - (d > 0).mean()) / len(d)))]}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "summary.json", "w"), indent=1)
    for g, d in res["groups"].items():
        for model, o in d.items():
            r2 = {m: round(v["r2"], 2) for m, v in o["per_score"].items()}
            print(f"{g:20s} {model}: R2 {r2}" + (f"  features-only AUC {o['features_only_auc'][0]:.3f}" if "features_only_auc" in o else ""))
            if "features_only_auc" in o:
                print("   " + ", ".join(f"{m} {v['auc'][0]:.3f}->{v['auc_residual'][0]:.3f}" for m, v in o["per_score"].items()))
    for model, d in res["caa_pairs"].items():
        print(f"CAA pairs {model} (N={d['N']}): " + ", ".join(f"{m} {v['share_refusal_higher'][0]:.2f}" for m, v in d.items() if m != "N"))


if __name__ == "__main__":
    main()
