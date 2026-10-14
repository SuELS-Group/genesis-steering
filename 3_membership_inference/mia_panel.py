"""Token-level membership scores across all datasets on Tülu : the token gap and the other scores of
mia_tokens on every dataset of the study, with membership checked per row, rendered with Tülu's chat template exactly as in
its SFT training.

    data        per dataset and pole, every row of part 2's pole files (the pools the study draws from; part 2 caps them at
                1,000 per pole), N_PER = None (a number would draw that many, seed 0): with full gold turns (data/poles_full):
                WildGuardMix, WildJailbreak, CoCoNot (safety refusals; compliances from CoCoNot's release, prompt-only rows
                dropped), CoCoNot rows left out of the mixture (refusals only: its compliances are CoCoNot's, kept once), Do-Not-Answer, HH-RLHF, PKU-SafeRLHF; CAA's generate pairs
                twice: the two authored answers (the caa_mean texts) and Rimsky's A/B rendering (question with its Choices
                block, the gold letter "(A)"/"(B)" as the answer; the caa_ab texts); matched CoCoNot split (members / non-members, redacted)
    membership  per row: 1 if the (prompt, answer) pair is verbatim in Tülu-3's SFT mixture (its CoCoNot, WildGuardMix and
                WildJailbreak slices, whitespace-normalised), else 0; the matched split keeps labels
    rendering   tok.apply_chat_template([user, assistant], tokenize=True): the template exactly as Tülu's SFT tokenization
                (no BOS, role tags, <|end_of_text|> after the answer); the answer = every token after the tokens of
                apply_chat_template([user], add_generation_prompt=True), i.e. the answer and its closing <|end_of_text|>;
                a row is dropped if that prefix does not tokenize identically; no length cap below 4,096 tokens
    models      Tülu-3-8B-SFT and Tülu-3-8B final (targets), Llama-3.1-8B base and the no-safety twin (references)
    scores      mia_tokens' per-token quantities on the answer's tokens; per row loss, zlib, min-k, min-k++, gap, min-k gap,
                top-1 agreement, s (base), s (twin), and ReCaLL on the answer: NLL(answer | prefix, prompt) / NLL(answer | prompt),
                the prefix = fixed non-member prefix (its 10 HH-RLHF aux conversations, Tülu template, in the
                runner's seed-42 order, joined with " "), tokenized once and put in front of each row's tokens (its KV cache
                computed once and reused); higher = more member-like

    python mia_panel.py run        # GPU, results/mia_panel/rows_<model>.jsonl
    python mia_panel.py recall     # GPU, results/mia_panel/recall_<target>.jsonl: the answer's log p behind ReCaLL's prefix
    python mia_panel.py summary    # CPU, results/mia_panel/summary.json

    --template meta : the same rows (results/mia_panel/items.jsonl; `member`
                stays membership in Tülu-3's SFT mixture, a negative control on this model), rendered with
                Llama-3.1-8B-Instruct's chat template (its default system header; the answer = every token after the
                generation prompt, i.e. the answer and its <|eot_id|>); target Meta-Llama-3.1-8B-Instruct, reference the
                Llama-3.1-8B base scored in the same template (no twin exists: no s (twin)); ReCaLL's prefix = the Meta-template rendering of the same 10 HH-RLHF rows (data/datasets_split_meta/aux_chat.json);
                -> results/mia_panel_meta/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "2_genesis_steering")); sys.path.insert(0, str(HERE.parent / "1_models_and_datasets"))
import mia_caa as M          # noqa: E402
import mia_tokens as TK      # noqa: E402

OUT = HERE / "results" / "mia_panel"
ITEMS = OUT / "items.jsonl"                     # one row list for both templates
MODELS = TK.MODELS
TOK_ID, TARGETS, AUX = "allenai/Llama-3.1-Tulu-3-8B-SFT", ("sft", "final"), "datasets_split"
TEMPLATES = {"tulu": (OUT, MODELS, TOK_ID, TARGETS, AUX),
             "meta": (HERE / "results" / "mia_panel_meta",
                      {"meta": "meta-llama/Llama-3.1-8B-Instruct", "base": "meta-llama/Llama-3.1-8B"},
                      "meta-llama/Llama-3.1-8B-Instruct", ("meta",), "datasets_split_meta"),
             "qwen": (HERE / "results" / "mia_panel_qwen",                                 # the third model
                      {"qwen": "Qwen/Qwen3-8B", "base": "Qwen/Qwen3-8B-Base"},
                      "Qwen/Qwen3-8B", ("qwen",), "datasets_split_qwen")}


def use_template(name):
    global OUT, MODELS, TOK_ID, TARGETS, AUX
    OUT, MODELS, TOK_ID, TARGETS, AUX = TEMPLATES[name]
POLES_FULL = HERE / "data" / "poles_full"
ARMS = {"g_wildguardmix": "WildGuardMix", "g_wildjailbreak": "WildJailbreak", "g_coconot_safety": "CoCoNot",
        "x_coconot_leftover": "CoCoNot, left out", "x_donotanswer": "Do-Not-Answer", "x_hh_rlhf": "HH-RLHF", "x_pku_saferlhf": "PKU-SafeRLHF"}
N_PER, SEED, MAXLEN, K = None, 0, 4096, 0.2          # N_PER None: every row of the pool
norm = lambda s: " ".join((s or "").split())


def mixture_pairs():
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    import registry as REG
    pairs = set()
    for src in REG.MIX_SOURCE.values():
        for r in REG.mixture_rows(src):
            pairs.add((norm(r["user"]), norm(r["assistant"])))
    return pairs


def items():
    """[(dataset, pole, prompt, answer, member)]."""
    f = ITEMS
    if f.exists():
        return [json.loads(l) for l in open(f)]
    mix = mixture_pairs(); rng = np.random.default_rng(SEED); out = []; seen = set()
    for arm, name in ARMS.items():
        for pole in ("refuse", "comply"):
            rows = [r for r in map(json.loads, open(POLES_FULL / f"{arm}_{pole}.jsonl")) if (r.get("gold") or "").strip()]
            pick = list(range(len(rows))) if N_PER is None else sorted(rng.choice(len(rows), min(N_PER, len(rows)), replace=False).tolist())
            for j in pick:
                r = rows[j]
                if (pole, r["prompt"], r["gold"]) in seen:      # an arm reusing another arm's pool (left-out CoCoNot's compliances) counts once
                    continue
                seen.add((pole, r["prompt"], r["gold"]))
                out.append({"dataset": name, "pole": pole, "prompt": r["prompt"], "answer": r["gold"],
                            "member": int((norm(r["prompt"]), norm(r["gold"])) in mix)})
    ab = {norm(x["question"].split("\n\nChoices:\n", 1)[0]): x for x in json.load(open(M.RAW / M.SPLITS["generate"]))}
    for r in M.load_pairs():
        if r["split"] != "generate":
            continue
        for pole in ("refuse", "comply"):
            out.append({"dataset": "CAA, answers (mean)", "pole": pole, "prompt": r["prompt"], "answer": r[pole],
                        "member": int((norm(r["prompt"]), norm(r[pole])) in mix)})
        x = ab[norm(r["prompt"])]                         # Rimsky's A/B rendering: question + Choices block, the gold letter
        for pole, key in (("refuse", "answer_matching_behavior"), ("comply", "answer_not_matching_behavior")):
            out.append({"dataset": "CAA, A/B letter", "pole": pole, "prompt": x["question"], "answer": x[key].strip(),
                        "member": int((norm(x["question"]), norm(x[key])) in mix)})
    for r in TK.rows():                                   # matched split, labels as there
        t = r["text"]; j = t.find(TK.TAG)
        out.append({"dataset": "CoCoNot, matched split", "pole": "mixed", "prompt": t[len("<|user|>\n"):j].strip(), "answer": r["answer"], "member": r["member"]})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(f, "w") as fh:
        for x in out:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    return out


def encode(tok, it):
    """(ids, answer_start) with the full chat template, or None if the prompt prefix does not tokenize identically."""
    full = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": it["answer"]}], tokenize=True,
                                   enable_thinking=False)       # Qwen3: the answer directly after an empty think block; Llamas ignore it
    pre = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ids_of = lambda e: list(e["input_ids"] if hasattr(e, "keys") else e)          # a BatchEncoding in recent transformers, else a list
    full, pre = ids_of(full), ids_of(pre)
    if full[:len(pre)] != pre or len(full) <= len(pre) or len(full) > MAXLEN:
        return None
    return full, len(pre)


def run():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import mia_quartile as Q
    IT = items(); print(f"{len(IT)} items", flush=True)
    for mk, mid in MODELS.items():
        f = OUT / f"rows_{mk}.jsonl"
        if f.exists():
            print(f"[{mk}] exists"); continue
        Q.gpu_guard(); t0 = time.time()
        tok = AutoTokenizer.from_pretrained(TOK_ID)         # one template (Tülu's or Meta's) for every model
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
        enc = [encode(tok, it) for it in IT]
        order = sorted([k for k in range(len(IT)) if enc[k]], key=lambda k: len(enc[k][0]))
        OUT.mkdir(parents=True, exist_ok=True)
        pad = tok.eos_token_id          # not Tülu's <pad> (id 128256): outside Llama base's vocabulary; right padding is masked and never scored
        with open(f, "w") as fh, torch.inference_mode():
            for b in range(0, len(order), 4):
                ks = order[b:b + 4]; L = max(len(enc[k][0]) for k in ks)
                ids = torch.full((len(ks), L), pad, dtype=torch.long); am = torch.zeros((len(ks), L), dtype=torch.long)
                for j, k in enumerate(ks):
                    ids[j, :len(enc[k][0])] = torch.tensor(enc[k][0]); am[j, :len(enc[k][0])] = 1
                ids, am = ids.cuda(), am.cuda()
                h = model.model(input_ids=ids, attention_mask=am).last_hidden_state
                for j, k in enumerate(ks):
                    full, a = enc[k]; pos = torch.arange(a, len(full), device="cuda")
                    lp = torch.log_softmax(model.lm_head(h[j, pos - 1]).float(), -1)
                    g = lp.gather(-1, ids[j, pos].unsqueeze(-1)).squeeze(-1); p = lp.exp()
                    mu = (p * lp).sum(-1); sd = ((p * (lp - mu.unsqueeze(-1)) ** 2).sum(-1)).clamp_min(1e-12).sqrt()
                    fh.write(json.dumps({"k": k, "lp": g.cpu().numpy().round(4).tolist(), "lp_top": lp.max(-1).values.cpu().numpy().round(4).tolist(),
                                         "mu": mu.cpu().numpy().round(4).tolist(), "sd": sd.cpu().numpy().round(4).tolist()}) + "\n")
                del h; torch.cuda.empty_cache()
        print(f"[{mk}] {len(order)} rows in {(time.time() - t0) / 60:.1f} min", flush=True)
        del model; torch.cuda.empty_cache()


def qwen_aux():
    """data/datasets_split_qwen/aux_chat.json: 10 HH-RLHF aux conversations re-rendered in Qwen3-8B's chat template.
    The user and assistant turns are parsed back out of the T\"ulu rendering (data/datasets_split/aux_chat.json), so the text is
    identical; only the template changes."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); out = []
    for r in json.load(open(HERE / "data" / "datasets_split" / "aux_chat.json")):
        t = r["text"]; assert t.startswith("<|user|>\n") and "\n<|assistant|>\n" in t
        u, a = t[len("<|user|>\n"):].split("\n<|assistant|>\n", 1)
        a = a.removesuffix("<|end_of_text|>").rstrip("\n")
        out.append({**r, "text": tok.apply_chat_template([{"role": "user", "content": u}, {"role": "assistant", "content": a}],
                                                         tokenize=False, enable_thinking=False)})
    d = HERE / "data" / "datasets_split_qwen"; d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / "aux_chat.json", "w")); print(f"{len(out)} aux rows -> {d / 'aux_chat.json'}")


def prefix_ids(tok):
    """ReCaLL prefix (the runner's draw: random.seed(42), random.sample(range(10), 10), joined with " ")."""
    import random
    aux = json.load(open(HERE / "data" / AUX / "aux_chat.json"))
    random.seed(42)
    return tok(" ".join(aux[i]["text"] for i in random.sample(range(len(aux)), 10)) + " ", add_special_tokens=False).input_ids


def recall():
    """Per row and target (SFT, final): log p of each answer token behind the prefix. The prefix's KV cache is computed
    once and repeated over the batch; the first batch is checked against an uncached pass on the concatenated tokens."""
    import copy
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    import mia_quartile as Q
    IT = items(); tok = AutoTokenizer.from_pretrained(TOK_ID); P = prefix_ids(tok)
    print(f"{len(IT)} items, prefix {len(P)} tokens", flush=True)
    for mk in TARGETS:
        f = OUT / f"recall_{mk}.jsonl"
        if f.exists():
            print(f"[{mk}] exists"); continue
        Q.gpu_guard(); t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(MODELS[mk], dtype=torch.bfloat16, device_map="cuda").eval()
        enc = [encode(tok, it) for it in IT]
        order = sorted([k for k in range(len(IT)) if enc[k]], key=lambda k: len(enc[k][0]))
        pad, Pn = tok.eos_token_id, len(P)
        with torch.inference_mode():
            base_cache = model.model(input_ids=torch.tensor([P], device="cuda"), use_cache=True).past_key_values
            with open(f, "w") as fh:
                for b in range(0, len(order), 4):
                    ks = order[b:b + 4]; L = max(len(enc[k][0]) for k in ks)
                    ids = torch.full((len(ks), L), pad, dtype=torch.long); am = torch.zeros((len(ks), Pn + L), dtype=torch.long)
                    am[:, :Pn] = 1
                    for j, k in enumerate(ks):
                        ids[j, :len(enc[k][0])] = torch.tensor(enc[k][0]); am[j, Pn:Pn + len(enc[k][0])] = 1
                    ids, am = ids.cuda(), am.cuda()
                    cache = copy.deepcopy(base_cache); cache.batch_repeat_interleave(len(ks))
                    pos = torch.arange(Pn, Pn + L, device="cuda").unsqueeze(0).expand(len(ks), -1)
                    h = model.model(input_ids=ids, attention_mask=am, past_key_values=cache, position_ids=pos, use_cache=True).last_hidden_state
                    for j, k in enumerate(ks):
                        full, a = enc[k]; at = torch.arange(a, len(full), device="cuda")
                        lp = torch.log_softmax(model.lm_head(h[j, at - 1]).float(), -1).gather(-1, ids[j, at].unsqueeze(-1)).squeeze(-1)
                        if b == 0 and j == 0:            # the cache must equal one pass over prefix + row
                            hc = model.model(input_ids=torch.tensor([P + full], device="cuda")).last_hidden_state[0]
                            lpc = torch.log_softmax(model.lm_head(hc[Pn + at - 1]).float(), -1).gather(-1, ids[j, at].unsqueeze(-1)).squeeze(-1)
                            dev = float((lp - lpc).abs().max()); print(f"[{mk}] cached vs uncached, max |delta log p| {dev:.4f}", flush=True)
                            assert dev < 0.1, dev
                        fh.write(json.dumps({"k": k, "lp_pre": lp.cpu().numpy().round(4).tolist()}) + "\n")
                    del h, cache; torch.cuda.empty_cache()
        print(f"[{mk}] {len(order)} rows in {(time.time() - t0) / 60:.1f} min", flush=True)
        del model; torch.cuda.empty_cache()


def features():
    IT = items()
    S = {mk: {d["k"]: d for d in map(json.loads, open(OUT / f"rows_{mk}.jsonl"))} for mk in MODELS if (OUT / f"rows_{mk}.jsonl").exists()}
    ks = sorted(set.intersection(*[set(v) for v in S.values()]))
    R = {t: {d["k"]: d["lp_pre"] for d in map(json.loads, open(OUT / f"recall_{t}.jsonl"))} for t in TARGETS if (OUT / f"recall_{t}.jsonl").exists()}
    mink = lambda x: float(np.mean(np.sort(x)[:max(1, int(np.ceil(K * len(x))))]))
    F = []
    for k in ks:
        it = IT[k]; f = {"k": k, "dataset": it["dataset"], "pole": it["pole"], "member": it["member"], "n": len(S[TARGETS[0]][k]["lp"])}
        for tgt in TARGETS:
            lp, top, mu, sd = (np.array(S[tgt][k][q]) for q in ("lp", "lp_top", "mu", "sd")); gap = lp - top
            f.update({f"{tgt}: loss": float(lp.mean()), f"{tgt}: min-k": mink(lp), f"{tgt}: min-k++": mink((lp - mu) / sd),
                      f"{tgt}: gap (CH)": float(gap.mean()), f"{tgt}: min-k gap (CH)": mink(gap), f"{tgt}: top-1 agree (CH)": float((gap > -1e-6).mean()),
                      f"{tgt}: s (base)": float(lp.mean() - np.mean(S["base"][k]["lp"])),
                      f"{tgt}: zlib": float(lp.sum() / len(zlib.compress(it["answer"].encode())))})
            if k in R.get(tgt, {}):
                f[f"{tgt}: ReCaLL"] = float(np.mean(R[tgt][k]) / lp.mean())          # NLL(y | P, x) / NLL(y | x)
        if "twin" in S:
            f["sft: s (twin)"] = float(np.mean(S["sft"][k]["lp"]) - np.mean(S["twin"][k]["lp"]))
        F.append(f)
    return F


def summary():
    from sklearn.metrics import roc_auc_score
    F = features(); scores = [k for k in F[0] if ":" in k and all(k in f for f in F)]; out = {"by_dataset": {}, "auc": {}}
    groups = sorted({(f["dataset"], f["pole"]) for f in F})
    for g in groups:
        G = [f for f in F if (f["dataset"], f["pole"]) == g]
        out["by_dataset"][f"{g[0]} | {g[1]}"] = {"N": len(G), "members": int(sum(f["member"] for f in G)), "median_tokens": float(np.median([f["n"] for f in G])),
                                                 **{s: [float(np.mean([f[s] for f in G])), float(np.std([f[s] for f in G], ddof=1) / np.sqrt(len(G)))] for s in scores}}
    # member vs non-member across the panel, per pole (excluding the matched split, reported on its own)
    for pole in ("refuse", "comply", "mixed"):
        G = [f for f in F if f["pole"] == pole]
        y = np.array([f["member"] for f in G])
        if len(G) and 0 < y.sum() < len(y):
            out["auc"][pole] = {"members": int(y.sum()), "non_members": int(len(y) - y.sum()),
                                **{s: float(roc_auc_score(y, [f[s] for f in G])) for s in scores},
                                "length (longer)": float(roc_auc_score(y, [f["n"] for f in G]))}
    json.dump(out, open(OUT / "summary.json", "w"), indent=1)
    sel = [x for x in ["qwen: gap (CH)", "qwen: loss", "qwen: min-k++", "qwen: ReCaLL", "qwen: s (base)", "meta: gap (CH)", "meta: loss", "meta: zlib", "meta: min-k++", "meta: ReCaLL", "meta: s (base)", "sft: gap (CH)", "final: gap (CH)", "sft: loss", "sft: zlib", "sft: min-k++", "sft: ReCaLL", "final: ReCaLL", "sft: s (twin)", "final: s (base)"] if x in scores]
    print("per dataset and pole: mean (members of N) for " + ", ".join(sel))
    for g, d in out["by_dataset"].items():
        print(f"  {g:34s} N={d['N']:4d} mem={d['members']:4d} tok={d['median_tokens']:5.0f} " + " ".join(f"{d[s][0]:+.3f}" for s in sel))
    for pole, d in out["auc"].items():
        print(f"\nAUC members vs non-members, {pole} ({d['members']} / {d['non_members']}): " + ", ".join(f"{s} {d[s]:.3f}" for s in sel) + f", length {d['length (longer)']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["items", "run", "recall", "summary", "aux"])
    ap.add_argument("--template", choices=list(TEMPLATES), default="tulu")
    a = ap.parse_args(); use_template(a.template)
    {"items": lambda: print(len(items())), "run": run, "recall": recall, "summary": summary, "aux": qwen_aux}[a.stage]()
