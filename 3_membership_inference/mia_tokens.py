"""Which membership score is accurate where the exact text's membership is known , including 's
token-level comparison of the gold answer with the model's own preferred token.

    data        matched CoCoNot split (data/matched_split/coconot_chat.json): 398 members (CoCoNot rows verbatim in
                Tülu-3's SFT mixture) and 397 non-members (CoCoNot release rows outside it, redacted and matched on subcategory and
                AI self-reference); Tülu-rendered conversations, the answer = the text after the assistant tag
    models      targets Tülu-3-8B-SFT (trained on the mixture) and Tülu-3-8B final; references Llama-3.1-8B base and the SFT
                model's no-safety-data twin; one forward pass per row, plain and with ReCaLL's prefix in front (targets and base)
    per row     on the answer's tokens only (log-probabilities from the model's next-token distribution at each answer position):
                  loss          mean log p(gold)                                 (high = member-like)
                  zlib          loss over zlib-compressed bytes of the answer
                  min-k         mean of the k = 20 % lowest log p(gold)
                  min-k++       mean of the k = 20 % lowest (log p(gold) - mu) / sigma, mu and sigma the mean and s.d. of
                                log p over the vocabulary under the model's own distribution (Zhang et al.)
                  gap           mean of log p(gold) - log p(argmax): how far the gold token is from the model's preferred one
                                (0 when they agree; quantity)
                  min-k gap     mean of the k = 20 % most negative gaps
                  top-1 agree   share of answer tokens where the gold token is the model's top token
                  s (base)      NLL_base - NLL_target;  s (twin) NLL_twin - NLL_SFT;  s behind prefix;  ReCaLL (target, response)
    metrics     AUC (members positive), bootstrap 95 % interval, TPR at 5 % FPR

    python mia_tokens.py run        # GPU, writes results/mia_tokens/rows_<model>.jsonl
    python mia_tokens.py summary    # CPU, writes results/mia_tokens/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "2_genesis_steering"))
import mia_caa as M          # noqa: E402

DATA = HERE / "data" / "matched_split" / "coconot_chat.json"
OUT = HERE / "results" / "mia_tokens"
MODELS = {"sft": "allenai/Llama-3.1-Tulu-3-8B-SFT", "final": "allenai/Llama-3.1-Tulu-3-8B", "base": "meta-llama/Llama-3.1-8B",
          "twin": "allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data"}
PREFIXED = ("sft", "final", "base")
TAG, K, BATCH = "<|assistant|>\n", 0.2, 4


def rows():
    out = []
    for i, r in enumerate(json.load(open(DATA))):
        t = r["text"]; j = t.find(TAG)
        if j < 0:
            continue
        ans = t[j + len(TAG):].replace("<|end_of_text|>", "")
        out.append({"i": i, "member": int(r["label"]), "text": t, "ans_start": j + len(TAG), "ans_end": j + len(TAG) + len(ans.rstrip()),
                    "answer": ans.strip()})
    return out


def token_stats(model, tok, texts, spans):
    """Per text: arrays over the answer's tokens of log p(gold), log p(argmax), mean and s.d. of log p over the vocabulary."""
    import torch
    res = [None] * len(texts); order = sorted(range(len(texts)), key=lambda k: len(texts[k]))
    tok.padding_side = "right"
    with torch.inference_mode():
        for b in range(0, len(order), BATCH):
            ks = order[b:b + BATCH]
            enc = tok([texts[k] for k in ks], return_tensors="pt", padding=True, return_offsets_mapping=True, truncation=True, max_length=1400)
            offs = enc.pop("offset_mapping"); enc = enc.to("cuda")
            h = model.model(**enc).last_hidden_state
            ids = enc["input_ids"]; am = enc["attention_mask"].bool()
            for j, k in enumerate(ks):
                a, e = spans[k]
                pos = [t for t in range(1, ids.shape[1]) if am[j, t] and offs[j, t, 0] >= a and offs[j, t, 0] < e and offs[j, t, 1] > offs[j, t, 0]]
                if not pos:
                    res[k] = None; continue
                pp = torch.tensor(pos, device="cuda")
                lp = torch.log_softmax(model.lm_head(h[j, pp - 1]).float(), -1)                 # predictions for positions pos
                g = lp.gather(-1, ids[j, pp].unsqueeze(-1)).squeeze(-1)
                p = lp.exp()
                mu = (p * lp).sum(-1); sd = ((p * (lp - mu.unsqueeze(-1)) ** 2).sum(-1)).clamp_min(1e-12).sqrt()
                res[k] = {"lp": g.cpu().numpy(), "lp_top": lp.max(-1).values.cpu().numpy(), "mu": mu.cpu().numpy(), "sd": sd.cpu().numpy()}
            del h; torch.cuda.empty_cache()
    return res


def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(HERE)); import mia_quartile as Q
    R = rows(); OUT.mkdir(parents=True, exist_ok=True)
    prefix = M.runner_prefix("tulu")
    for mk, mid in MODELS.items():
        f = OUT / f"rows_{mk}.jsonl"
        if f.exists():
            print(f"[{mk}] exists"); continue
        Q.gpu_guard(); t0 = time.time()
        tok = AutoTokenizer.from_pretrained(mid); tok.pad_token = tok.pad_token or tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
        texts = [r["text"] for r in R]; spans = [(r["ans_start"], r["ans_end"]) for r in R]
        plain = token_stats(model, tok, texts, spans)
        pref = None
        if mk in PREFIXED:
            sh = len(prefix) + 1
            pref = token_stats(model, tok, [prefix + " " + t for t in texts], [(a + sh, e + sh) for a, e in spans])
        with open(f, "w") as fh:
            for r, a, b in zip(R, plain, pref or [None] * len(R)):
                if a is None:
                    continue
                fh.write(json.dumps({"i": r["i"], "member": r["member"], "lp": a["lp"].round(4).tolist(), "lp_top": a["lp_top"].round(4).tolist(),
                                     "mu": a["mu"].round(4).tolist(), "sd": a["sd"].round(4).tolist(),
                                     "lp_prefix_mean": None if b is None else float(b["lp"].mean())}) + "\n")
        print(f"[{mk}] {len(R)} rows in {(time.time() - t0) / 60:.1f} min", flush=True)
        del model; torch.cuda.empty_cache()


def features():
    R = {r["i"]: r for r in rows()}
    S = {mk: {d["i"]: d for d in map(json.loads, open(OUT / f"rows_{mk}.jsonl"))} for mk in MODELS if (OUT / f"rows_{mk}.jsonl").exists()}
    ids = sorted(set.intersection(*[set(v) for v in S.values()]))
    mink = lambda x: float(np.mean(np.sort(x)[:max(1, int(np.ceil(K * len(x))))]))
    F = {}
    for i in ids:
        f = {"member": R[i]["member"]}; zb = len(zlib.compress(R[i]["answer"].encode()))
        for tgt in ("sft", "final"):
            d = S[tgt][i]; lp, top, mu, sd = (np.array(d[k]) for k in ("lp", "lp_top", "mu", "sd"))
            gap = lp - top
            f[f"{tgt}: loss"] = float(lp.mean()); f[f"{tgt}: zlib"] = float(lp.sum() / zb)
            f[f"{tgt}: min-k"] = mink(lp); f[f"{tgt}: min-k++"] = mink((lp - mu) / sd)
            f[f"{tgt}: gap (CH)"] = float(gap.mean()); f[f"{tgt}: min-k gap (CH)"] = mink(gap); f[f"{tgt}: top-1 agree (CH)"] = float((gap > -1e-6).mean())
            f[f"{tgt}: s (base)"] = float(lp.mean() - np.mean(S["base"][i]["lp"]))
            if d["lp_prefix_mean"] is not None and S["base"][i]["lp_prefix_mean"] is not None:
                f[f"{tgt}: s behind prefix"] = d["lp_prefix_mean"] - S["base"][i]["lp_prefix_mean"]
                f[f"{tgt}: ReCaLL (response)"] = (-d["lp_prefix_mean"]) / (-lp.mean())
        f["sft: s (twin)"] = float(np.mean(S["sft"][i]["lp"]) - np.mean(S["twin"][i]["lp"]))
        F[i] = f
    return F


def summary(n_boot=1000):
    from sklearn.metrics import roc_auc_score, roc_curve
    F = features(); ids = sorted(F); y = np.array([F[i]["member"] for i in ids]); rng = np.random.default_rng(0)
    out = {"N_members": int(y.sum()), "N_non_members": int((1 - y).sum()), "scores": {}}
    for k in [k for k in F[ids[0]] if k != "member"]:
        s = np.array([F[i][k] for i in ids], dtype=float)
        if np.isnan(s).any():
            continue
        boots = []
        for _ in range(n_boot):
            j = rng.integers(0, len(y), len(y))
            if 0 < y[j].sum() < len(j):
                boots.append(roc_auc_score(y[j], s[j]))
        fpr, tpr, _ = roc_curve(y, s)
        out["scores"][k] = {"auc": float(roc_auc_score(y, s)), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                            "tpr@5%fpr": float(np.interp(0.05, fpr, tpr))}
    matched = json.load(open(HERE / "results" / "tulu_coconot_matched" / "scores.json"))
    for view in ("coconot_chat", "coconot_resp"):
        v = matched[view]; yy = np.array(v["label"])
        for k in ("bag_of_words", "loss", "minkprob", "minkplusplus", "recall"):
            if k in v and not np.isnan(np.array(v[k], dtype=float)).any():
                ss = np.array(v[k], dtype=float); fpr, tpr, _ = roc_curve(yy, ss)
                out["scores"][f"runner, {view.split('_')[1]}: {k}"] = {"auc": float(roc_auc_score(yy, ss)), "ci": None, "tpr@5%fpr": float(np.interp(0.05, fpr, tpr))}
    json.dump(out, open(OUT / "summary.json", "w"), indent=1)
    for k, v in sorted(out["scores"].items(), key=lambda kv: -kv[1]["auc"]):
        ci = "" if v["ci"] is None else f"[{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"
        print(f"  {k:34s} AUC {v['auc']:.3f} {ci:16s} TPR@5% {v['tpr@5%fpr']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["run", "summary"])
    {"run": run, "summary": summary}[ap.parse_args().stage]()
