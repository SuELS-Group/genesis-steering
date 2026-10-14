"""Larger MIA selections from one genesis dataset: Tülu-3-8B, WildGuardMix, top k = 128 / 350 / 450.

The question : with the cleanest membership score we can make, do vectors built from the top 350 or
450 membership-ranked examples steer Tülu better than from the top 128, and better than random draws of the same size?

Design
  pools      ALL of WildGuardMix's rows in Tülu's SFT mixture (the 50k slice), poles by part 2's rule: refusal = hub
             label harmful AND the gold turn opens as a refusal; compliance = hub label benign AND the gold turn is a
             clean compliance; prompt filter and de-duplication of poles.py; no 1,000 cap. Every example is training
             data, on both poles.
  scored     the gold assistant turn only, conditioned on the prompt, in Tülu's chat template; template and prompt
             tokens are not scored (SFT trains on assistant tokens only). Per token.
  scores     clean      Tülu-3-8B-SFT minus its no-safety twin (same recipe and data except the safety sets, which
                        include this slice):   s = (NLL_twin - NLL_sft) / n_resp_tokens
             released   Tülu-3-8B (final) minus the Llama-3.1-8B base (rendered in Tülu's template), the reference
                        available for a released model:   s = (NLL_base - NLL_final) / n_resp_tokens
  bias       within each pole, each score is regressed on log response tokens, log prompt tokens and the reference
             model's own per-token NLL (how predictable / generic the text is); the residual is ranked, and the top k
             is taken length-stratified: the same share from each decile of response length, so the selection's
             length distribution equals the pool's by construction.
  arms       for k in 128 / 350 / 450: mia_clean_k, mia_released_k (top k of each pole), rand0_k, rand1_k (random k
             of each pole, two seeds: the spread of random draws).
  steering   Tülu-3-8B with part 2's recipe (hidden_states[13], last prompt token, unit norm, 256 greedy tokens, judge)
             on a reduced grid : two benchmarks, one per pole (HarmBench under -alpha, XSTest-safe
             under +alpha), the first 50 of each benchmark's fixed 100 prompts, |alpha| = 4 / 6 / 8, plus the unsteered
             cells on the same 50 prompts.

Stages (GPU unless noted; resume-safe; one GPU job at a time):
    python mia_topk.py pools                    # CPU: data/mia_topk/pool_{refuse,comply}.jsonl (full gold turns)
    python mia_topk.py spans --model final|base|sft|twin
    python mia_topk.py acts                     # Tülu-3-8B hidden_states[13] of every pool prompt
    python mia_topk.py select                   # CPU: scores, residuals, selections -> data/mia_topk/sets/
    python mia_topk.py steer                    # every arm -> results/mia_topk/steer_completions.jsonl
    python ../judge.py --dump results/mia_topk/steer_completions.jsonl --out results/mia_topk/steer_judge.jsonl
Not committed (regenerated, or answers to harmful requests): data/mia_topk/pool_*, results/mia_topk/.
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
import poles as P            # noqa: E402
import steering as T         # noqa: E402

SRC = "wildguardmix"
DATA = HERE / "data" / "mia_topk"; SETS = DATA / "sets"
OUT = HERE / "results" / "mia_topk"
STEER_COMP, STEER_JUDGE = OUT / "steer_completions.jsonl", OUT / "steer_judge.jsonl"
ACTS = OUT / "acts_tulu_final_L13.npz"
MODELS = {"final": "allenai/Llama-3.1-Tulu-3-8B", "base": "meta-llama/Llama-3.1-8B",
          "sft": "allenai/Llama-3.1-Tulu-3-8B-SFT", "twin": "allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data"}
TEMPLATE = "allenai/Llama-3.1-Tulu-3-8B"
SCORE_DEF = {"clean": ("sft", "twin"), "released": ("final", "base")}     # (target, reference)
KS = [128, 350, 450]
RAND_SEEDS = [0, 1]
POLES2 = ("refuse", "comply")
N_DECILES, MAX_TOK, BATCH = 10, 512, 16
EVALS, N_PROMPTS, DOSES = ("harmbench", "xstest_safe"), 50, [4, 6, 8]


# ---- CPU: pools --------------------------------------------------------------------------------------
def build_pools():
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    import registry as REG
    REG.GOLD_CHARS = 10 ** 9                                   # full gold turns
    pools = dict(zip(POLES2, REG._build_from_mixture(SRC)))
    DATA.mkdir(parents=True, exist_ok=True)
    for pole, rows in pools.items():
        seen, out = set(), []
        for r in rows:
            p = (r["prompt"] or "").strip()
            if p and P._keep(p) and P._norm(p) not in seen and r["gold"].strip():
                seen.add(P._norm(p)); out.append({"prompt": p, "gold": r["gold"], "category": r["category"]})
        with open(DATA / f"pool_{pole}.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(pole, len(out))


def pool(pole):
    return [json.loads(l) for l in open(DATA / f"pool_{pole}.jsonl")]


# ---- GPU: assistant-turn NLL -------------------------------------------------------------------------
def spans(model_key):
    """Per example: NLL summed over the gold assistant turn and over the prompt, in Tülu's rendering, cut at 512
    tokens (the span location of mia_rank.score_pass_spans)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out_f = OUT / f"spans_{model_key}.jsonl"
    done = {(r["pole"], r["i"]) for r in map(json.loads, open(out_f))} if out_f.exists() else set()
    ttok = AutoTokenizer.from_pretrained(TEMPLATE)
    ex = [(pole, i, r["prompt"], r["gold"]) for pole in POLES2 for i, r in enumerate(pool(pole)) if (pole, i) not in done]
    if not ex:
        print(f"[{model_key}] spans complete"); return
    texts, sp = [], []
    for pole, i, p, g in ex:
        t = ttok.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": g}], tokenize=False)
        enc = ttok(t, add_special_tokens=False, return_offsets_mapping=True)
        t = t[:enc["offset_mapping"][MAX_TOK - 1][1]] if len(enc["input_ids"]) > MAX_TOK else t
        ps = t.find(p.strip()); pe = ps + len(p.strip())
        gs = t.find(g.strip()[:200], max(pe, 0)) if pe <= len(t) else -1
        ge = min(gs + len(g.strip()), len(t)) if gs >= 0 else -1
        texts.append(t); sp.append(((ps, min(pe, len(t))), (gs, ge)))
    tok = AutoTokenizer.from_pretrained(MODELS[model_key])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODELS[model_key], dtype=torch.bfloat16, device_map="cuda").eval()
    order = sorted(range(len(ex)), key=lambda k: len(texts[k]))
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    with open(out_f, "a") as f:
        for b in range(0, len(order), BATCH):
            ks = order[b:b + BATCH]
            enc = tok([texts[k] for k in ks], add_special_tokens=False, return_tensors="pt", padding=True, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping"); enc = enc.to("cuda")
            with torch.no_grad():
                logits = model(**enc).logits[:, :-1].float()
            nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), enc["input_ids"][:, 1:], reduction="none").cpu()
            st = offs[:, 1:, 0]; am = enc["attention_mask"][:, 1:].cpu().bool()
            for j, k in enumerate(ks):
                (ps, pe), (gs, ge) = sp[k]
                inp = am[j] & (st[j] >= ps) & (st[j] < pe) if ps >= 0 else torch.zeros_like(am[j])
                ing = am[j] & (st[j] >= gs) & (st[j] < ge) if gs >= 0 else torch.zeros_like(am[j])
                f.write(json.dumps({"pole": ex[k][0], "i": ex[k][1], "nll_prompt": float(nll[j][inp].sum()), "n_tok_prompt": int(inp.sum()),
                                    "nll_resp": float(nll[j][ing].sum()), "n_tok_resp": int(ing.sum())}) + "\n")
            del logits, nll; torch.cuda.empty_cache()
            if (b // BATCH) % 200 == 0:
                print(f"  [{model_key}] {b + len(ks)}/{len(order)}  {(b + len(ks)) / (time.time() - t0):.1f} ex/s", flush=True)
    print(f"[{model_key}] {len(ex)} examples in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---- GPU: activations at the steering site -----------------------------------------------------------
def acts():
    """hidden_states[13] at the last prompt token of every pool prompt on Tülu-3-8B: install.py's recipe (the model's
    chat template, the tokenizer's special tokens, right padding, 512-token cut), layer 13 only."""
    import torch
    if ACTS.exists():
        print("acts exist"); return
    prompts = list(dict.fromkeys(r["prompt"] for pole in POLES2 for r in pool(pole)))
    model, tok = T.load_model(MODELS["final"])
    rendered = T.render(prompts, tok)
    order = np.argsort([len(tok(r, add_special_tokens=True)["input_ids"]) for r in rendered], kind="stable")
    tok.padding_side = "right"
    X = np.zeros((len(prompts), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, len(order), T.EXTRACT_BS):
            idx = order[s:s + T.EXTRACT_BS].tolist()
            enc = tok([rendered[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512,
                      add_special_tokens=True).to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states[T.LAYER]
            lens = enc["attention_mask"].sum(1)
            X[idx] = hs[torch.arange(hs.shape[0], device=hs.device), (lens - 1).clamp(min=0)].float().cpu().numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(ACTS, X=X.astype(np.float16), prompts=np.array(json.dumps(prompts)))
    print(f"acts: {len(prompts)} prompts")


def load_acts():
    z = np.load(ACTS)
    return z["X"].astype(np.float32), {p: i for i, p in enumerate(json.loads(str(z["prompts"])))}


# ---- CPU: scores and selections ----------------------------------------------------------------------
def scores():
    """DataFrame per (pole, example): raw and residualized scores, lengths, reference NLL, length decile."""
    import pandas as pd
    S = {m: {(r["pole"], r["i"]): r for r in map(json.loads, open(OUT / f"spans_{m}.jsonl"))} for m in MODELS}
    rows = []
    for pole in POLES2:
        for i, r0 in enumerate(pool(pole)):
            k = (pole, i)
            if not all(k in S[m] for m in S) or S["final"][k]["n_tok_resp"] < 5:
                continue
            row = {"pole": pole, "i": i, "prompt": r0["prompt"], "n_resp": S["final"][k]["n_tok_resp"],
                   "n_prompt": S["final"][k]["n_tok_prompt"], "n_char_resp": len(r0["gold"])}
            for kind, (tgt, ref) in SCORE_DEF.items():
                n = S[tgt][k]["n_tok_resp"]
                row[f"{kind}_raw"] = (S[ref][k]["nll_resp"] - S[tgt][k]["nll_resp"]) / n
                row[f"{kind}_refnll"] = S[ref][k]["nll_resp"] / n
            rows.append(row)
    D = pd.DataFrame(rows)
    parts = []
    for pole, d in D.groupby("pole"):
        d = d.copy()
        for kind in SCORE_DEF:
            X = np.c_[np.ones(len(d)), np.log(d["n_resp"]), np.log(d["n_prompt"]), d[f"{kind}_refnll"]]
            beta, *_ = np.linalg.lstsq(X, d[f"{kind}_raw"].values, rcond=None)
            d[kind] = d[f"{kind}_raw"].values - X @ beta
        d["decile"] = pd.qcut(d["n_resp"].rank(method="first"), N_DECILES, labels=False)
        parts.append(d)
    return pd.concat(parts)


def stratified_top(d, col, k):
    """Top k of one pole by `col`, the same share from each response-length decile (largest remainders fill up)."""
    sizes = d.groupby("decile").size(); quota = sizes / sizes.sum() * k
    q = np.floor(quota).astype(int)
    for dec in (quota - q).sort_values(ascending=False).index[:k - q.sum()]:
        q[dec] += 1
    return [p for dec, n in q.items() for p in d[d.decile == dec].sort_values(col, ascending=False).head(n)["prompt"]]


def arms():
    return [f"{a}_k{k}" for k in KS for a in ("mia_clean", "mia_released", "rand0", "rand1")]


def select():
    D = scores(); SETS.mkdir(parents=True, exist_ok=True)
    for k in KS:
        sel = {}
        for pole in POLES2:
            d = D[D.pole == pole]
            sel.setdefault(f"mia_clean_k{k}", {})[pole] = stratified_top(d, "clean", k)
            sel.setdefault(f"mia_released_k{k}", {})[pole] = stratified_top(d, "released", k)
            for s in RAND_SEEDS:
                sel.setdefault(f"rand{s}_k{k}", {})[pole] = P._subsample(d["prompt"].tolist(), k,
                                                                         np.random.default_rng(1000 * (s + 1) + k + (0 if pole == "refuse" else 7)))
        for a, s_ in sel.items():
            with open(SETS / f"{a}.jsonl", "w") as f:
                for pole in POLES2:
                    for p in s_[pole]:
                        f.write(json.dumps({"pole": pole, "prompt": p}, ensure_ascii=False) + "\n")
    D.to_csv(OUT / "scores.csv", index=False)
    print("scored per pole:", D.groupby("pole").size().to_dict(), "->", SETS)


def load_set(arm):
    rows = [json.loads(l) for l in open(SETS / f"{arm}.jsonl")]
    return {pole: [r["prompt"] for r in rows if r["pole"] == pole] for pole in POLES2}


def directions():
    X, ix = load_acts(); out = {}
    for a in arms():
        s = load_set(a)
        v = X[[ix[p] for p in s["refuse"]]].mean(0) - X[[ix[p] for p in s["comply"]]].mean(0)
        out[a] = (v / np.linalg.norm(v)).astype(np.float32)
    return out


# ---- GPU: steering -----------------------------------------------------------------------------------
def grid():
    """(eval, arm, alpha) cells: the unsteered baseline of each benchmark, then every arm at every dose with the
    benchmark's sign."""
    cells = [(e, "baseline", 0) for e in EVALS]
    return cells + [(e, a, T.EVAL_SIGN[e] * d) for a in arms() for e in EVALS for d in DOSES]


def steer():
    import behaviour as B          # its _done / _write keep part 2's row format
    dirs = directions()
    todo = [c for c in grid() if ("tulu_final", *c) not in B._done(STEER_COMP)]
    if not todo:
        print("grid complete"); return
    ev = T.load_evals()
    model, tok = T.load_model(T.MODELS["tulu_final"]); st = T.Steerer(model, tok)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(STEER_COMP, "a") as fh:
        for n, (e, a, al) in enumerate(todo, 1):
            t0 = time.time(); ps = ev[e][0][:N_PROMPTS]
            texts, early = st.generate(ps, None if a == "baseline" else al * dirs[a])
            B._write(fh, "tulu_final", e, a, al, ps, texts, early)
            print(f"[{n}/{len(todo)}] {e} {a} {al:+d} ({time.time() - t0:.0f}s)", flush=True)
    st.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pools", "spans", "acts", "select", "steer"])
    ap.add_argument("--model", choices=list(MODELS))
    a = ap.parse_args()
    {"pools": build_pools, "spans": lambda: spans(a.model), "acts": acts, "select": select, "steer": steer}[a.stage]()
