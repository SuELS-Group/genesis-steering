"""CAA refusal A/B pairs through the Imperial benchmark: attacks on one prompt with two gold answers.

run the MIA on the CAA refusal dataset (Rimsky et al.) for Tülu-3-8B final and Meta-Llama-3.1-8B-
Instruct, the dataset's refusal answer as the positive pole and its compliance answer as the negative pole; the
benchmark's own runner (run_attacks.py), target-only scores reported next to the base model's; keep the subset labels.
Every CAA item is one question with two answer choices written by the dataset authors (neither model generated them),
so the two poles share the prompt exactly and only the assistant turn differs. Neither model is known to have trained
on CAA.

  pairs     generate/refusal (408) and test/refusal A/B (50) of nrimsky/CAA (1_models_and_datasets/data/_raw), the
            `Choices:` block stripped from the question; refuse = answer_matching_behavior, comply = the other answer,
            full text (the registry keeps 200 characters); a prompt that occurs twice is kept once (first occurrence)
  subset    persona / safety / hedge, from the refusal text by the regexes of the earlier experiment tree
            (experiments/refusal_subcluster/scripts/build_caa_pairs.py, re-implemented here unchanged; on the generate
            split it reproduces that tree's refusal_reason on all 393 pairs)
  views     caa_chat: prompt + gold answer in the chat template, rendered as build_datasets.py does (BOS string
            stripped, the runner adds it; 512-token cap); label 1 = refuse, 0 = comply, subcategory = subset; rows in
            pair order, refuse then comply (row 2k / 2k+1 = pair k). Two renderings: Tülu's template (data/caa_split)
            and Meta-Llama-3.1-Instruct's (data/caa_split_meta)
  prefix    ReCaLL's non-member shots = 10 hh-harmless rows (data/datasets_split{,_meta}/aux_chat.json,
            copied unchanged into the CAA folders)
  runs      run_attacks.py --views caa_chat --attacks loss,zlib,recall
              tulu template   final (configs/caa_final.yaml), base (configs/datasets_base.yaml)
              meta template   meta  (configs/datasets_meta.yaml), base (configs/datasets_base.yaml)
            -> results/caa_panel/<template>/<model>/scores.json; the runner's AUC is refuse vs comply
  summary   per pair refuse - comply of each attack score, on the target alone and target - base (the base-calibrated difference), overall, per split and per subset

    python mia_caa.py build                              # CPU
    (run_attacks.py passes: see run_all.sh)
    python mia_caa.py spans --model final|base_tulu|meta|base_meta   # GPU: our score, plain and behind the prefix
    python mia_caa.py pool_spans --model ...             # GPU: the SFT pools (1,000 per pole, seed 0), plain and behind the prefix
    python mia_caa.py summary                            # CPU
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2_genesis_steering"))
import poles as P            # noqa: E402

RAW = HERE.parent / "1_models_and_datasets" / "data" / "_raw"
SPLITS = {"generate": "caa_refusal_generate.json", "test_ab": "caa_refusal_test_ab.json"}
DATA = HERE / "data"
TEMPLATES = {"tulu": ("allenai/Llama-3.1-Tulu-3-8B-SFT", "caa_split", "datasets_split"),
             "meta": ("meta-llama/Llama-3.1-8B-Instruct", "caa_split_meta", "datasets_split_meta")}
RES = HERE / "results" / "caa_panel"
RUNS = {"tulu": ("final", "base"), "meta": ("meta", "base")}          # template -> (target, reference)
ATTACKS = ("loss", "zlib", "recall")
POLES2 = ("refuse", "comply")
MAX_TOK = 512

# ---- subset labels (build_caa_pairs.py of the earlier tree, unchanged) ----------------------------------
PERSONA = re.compile(
    r"\b(as an ai|an ai (?:system|assistant|model|language model)|as a (?:language|machine learning) model"
    r"|artificial intelligence|i(?:'m| am) (?:an ai|not conscious|not (?:a |)human|just an ai"
    r"|a (?:software|machine|computer|program|bot))"
    r"|i (?:do not|don'?t) (?:have|possess|experience|feel|eat|play|watch|sleep|attend"
    r"|take vacations|get (?:bored|angry|hungry|tired)|actually (?:have|eat|watch|read)"
    r"|make subjective|have personal|have the capability|read for pleasure)"
    r"|i lack (?:personal|the|consciousness)|i cannot (?:experience|feel|taste|physically)"
    r"|no personal (?:preferences|opinions|feelings|experiences|tastes)"
    r"|without (?:personal|training in)|i process|programmed|my (?:purpose|programming)"
    r"|don'?t have (?:feelings|emotions|preferences|a favorite|personal)"
    r"|helpful, harmless|machine learning model|do not (?:possess|experience|attend))\b",
    re.I,
)
SAFETY = re.compile(
    r"\b(illegal|unlawful|unethical|harmful|dangerous|unsafe|fraud|criminal|crime|violat\w*"
    r"|breach|unauthorized|copyright|plagiar|counterfeit|theft|steal|deceiv"
    r"|against (?:the law|legal|ethical|my)|not (?:appropriate|legal|ethical)"
    r"|(?:legal|ethical|academic) (?:standards|boundaries|concerns|issues|principles"
    r"|integrity|guidelines|consequences|dishonesty)"
    r"|i cannot (?:recommend|advise|provide|help|assist|support|condone|endorse|suggest"
    r"|fulfill|generate|guide)"
    r"|i can'?t (?:help|assist|provide|support|fulfill|explain)|cannot fulfill your request"
    r"|i (?:advise|urge|encourage) (?:against|you)|i'?d rather not"
    r"|i do not (?:recommend|suggest|support|condone)|goes against|puts safety|endangers"
    r"|harms (?:others|patrons|businesses|individuals)|i (?:must|have to) decline"
    r"|responsibly and legally|constitutes (?:fraud|theft|stealing)|sorry,? (?:but )?i can'?t"
    r"|make no judgments|do not make judgments|devalues|compromises your integrity"
    r"|undermines|responsible and ethical)\b",
    re.I,
)
HEDGE = re.compile(
    r"\b(varies|vary|depends?|subjective|each (?:individual|person)"
    r"|individual(?:'s)? (?:circumstances|needs|health|situation|factors)"
    r"|personal journey|specific to (?:each|someone|one'?s)"
    r"|based on (?:individual|someone'?s|each|one'?s|personal)"
    r"|differ(?:s|ent)? (?:among|based|for)|tailored to|highly personal|can differ"
    r"|not fully understood|complex|inherently uncertain)\b",
    re.I,
)


def subset(refusal):
    if PERSONA.search(refusal):
        return "persona"
    if SAFETY.search(refusal):
        return "safety"
    if HEDGE.search(refusal):
        return "hedge"
    return "safety" if re.search(r"\b(sorry|cannot|can'?t|can not|apologi)\b", refusal, re.I) else "persona"


def load_pairs():
    kept, seen = [], set()
    for split, fn in SPLITS.items():
        for x in json.load(open(RAW / fn)):
            stem, choices = x["question"].split("\n\nChoices:\n", 1)
            opts = {l.strip()[:3]: l.strip()[3:].strip() for l in choices.strip().split("\n") if l.strip().startswith("(")}
            m, n = x["answer_matching_behavior"].strip()[:3], x["answer_not_matching_behavior"].strip()[:3]
            if P._norm(stem) in seen:
                continue
            seen.add(P._norm(stem))
            kept.append({"pair": len(kept), "split": split, "subset": subset(opts[m]), "prompt": stem.strip(),
                         "refuse": opts[m], "comply": opts[n]})
    return kept


# ---- CPU: the runner's input files -----------------------------------------------------------------------
def build():
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from collections import Counter
    from transformers import AutoTokenizer
    ps = load_pairs()
    for tpl, (tid, out, aux_from) in TEMPLATES.items():
        tok = AutoTokenizer.from_pretrained(tid)

        def render(p, r):
            s = tok.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": r}], tokenize=False)
            s = s[len(tok.bos_token):] if tok.bos_token and s.startswith(tok.bos_token) else s
            ids = tok(s, add_special_tokens=False).input_ids
            return tok.decode(ids[:MAX_TOK]) if len(ids) > MAX_TOK else s
        rows = [{"text": render(r["prompt"], r[pole]), "label": int(pole == "refuse"), "subcategory": r["subset"]}
                for r in ps for pole in POLES2]
        (DATA / out).mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(DATA / out / "caa_chat.json", "w"))
        shutil.copy(DATA / aux_from / "aux_chat.json", DATA / out / "aux_chat.json")
        print(f"{tpl}: {len(rows)} rows -> data/{out}")
    print(f"pairs: {len(ps)};", dict(Counter((r["split"], r["subset"]) for r in ps)))


# ---- GPU: our score (NLL_base - NLL_target on the gold turn), plain and behind the prefix -----
OURS = HERE / "results" / "caa_ours"
OUR_MODELS = {"final": ("allenai/Llama-3.1-Tulu-3-8B", "tulu"), "base_tulu": ("meta-llama/Llama-3.1-8B", "tulu"),
              "meta": ("meta-llama/Llama-3.1-8B-Instruct", "meta"), "base_meta": ("meta-llama/Llama-3.1-8B", "meta")}
OUR_SCORE = {"tulu": ("final", "base_tulu"), "meta": ("meta", "base_meta")}      # template -> (target, reference)
BATCH = 8


def runner_prefix(tpl):
    """The runner's fixed ReCaLL prefix: recall.py draws random.sample(range(10), 10) after set_seed(42) (no other draw
    from Python's `random` precedes it in run_attacks.py) and joins the shots with " "."""
    import random
    aux = json.load(open(DATA / TEMPLATES[tpl][1] / "aux_chat.json"))
    random.seed(42)
    return " ".join(aux[i]["text"] for i in random.sample(range(len(aux)), 10))


def spans(model_key):
    """Per caa_chat row: NLL summed over the prompt, the gold turn (resp) and the whole conversation (text), plain and
    with the runner's prefix in front (the runner's string: prefix + " " + text, BOS added by the tokenizer; `pre` = the
    prefix tokens). Per-token logits only where needed. Writes results/caa_ours/spans_<model>.jsonl (rerun = overwrite)."""
    import os
    import time
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(HERE))
    import mia_quartile as Q
    mid, tpl = OUR_MODELS[model_key]
    rows = json.load(open(DATA / TEMPLATES[tpl][1] / "caa_chat.json")); ps = load_pairs()
    prefix = runner_prefix(tpl)
    items = []
    for k, row in enumerate(rows):
        pr = ps[k // 2]; g = pr[POLES2[k % 2]]; t = row["text"]
        s_p = t.find(pr["prompt"].strip()); s_g = t.find(g.strip()[:200], s_p + len(pr["prompt"].strip()))
        sp = {"prompt": (s_p, s_p + len(pr["prompt"].strip())), "resp": (s_g, min(s_g + len(g.strip()), len(t)) if s_g >= 0 else -1), "text": (0, len(t))}
        items.append(_with_prefix(t, sp, prefix))
    res = _passes(mid, items, model_key)
    OURS.mkdir(parents=True, exist_ok=True)
    with open(OURS / f"spans_{model_key}.jsonl", "w") as f:
        for k, r in enumerate(res):
            f.write(json.dumps({"row": k, **r}) + "\n")


def _with_prefix(t, sp, prefix):
    """(plain item, prefixed item): the runner's string prefix + " " + text, spans shifted, `pre` = the prefix."""
    sh = len(prefix) + 1
    return [(t, sp), (prefix + " " + t, {"pre": (0, len(prefix)), **{v: (a + sh, b + sh) if a >= 0 else (a, b) for v, (a, b) in sp.items()}})]


def _passes(mid, items, tag):
    """NLL summed over each item's character spans, plain (item[0]) and behind the prefix (item[1]); the tokenizer adds
    BOS, right padding, logits only at the needed positions. -> one dict per item."""
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(HERE))
    import mia_quartile as Q
    Q.gpu_guard()
    tok = AutoTokenizer.from_pretrained(mid); tok.pad_token = tok.eos_token; tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
    res = [{} for _ in items]; t0 = time.time()
    for pi, name in enumerate(("plain", "prefix")):
        texts = [it[pi] for it in items]
        order = sorted(range(len(texts)), key=lambda k: len(texts[k][0]))
        for bi in range(0, len(order), BATCH):
            ks = order[bi:bi + BATCH]
            enc = tok([texts[k][0] for k in ks], return_tensors="pt", padding=True, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping"); enc = enc.to("cuda")
            with torch.no_grad():
                h = model.model(**enc).last_hidden_state
                tgt = enc["input_ids"][:, 1:]; am = enc["attention_mask"][:, 1:].cpu().bool(); st = offs[:, 1:, 0]
                for j, k in enumerate(ks):
                    masks = {v: am[j] & (st[j] >= a) & (st[j] < b) if a >= 0 else torch.zeros_like(am[j]) for v, (a, b) in texts[k][1].items()}
                    pos = torch.stack(list(masks.values())).any(0).nonzero().squeeze(1).to("cuda")
                    nll = torch.nn.functional.cross_entropy(model.lm_head(h[j, pos]).float(), tgt[j, pos], reduction="none").cpu()
                    full = torch.zeros(am.shape[1]); full[pos.cpu()] = nll
                    for v, m in masks.items():
                        res[k][f"nll_{v}_{name}"] = float(full[m].sum()); res[k][f"n_{v}_{name}"] = int(m.sum())
            del h; torch.cuda.empty_cache()
        print(f"  [{tag}] {name}: {len(texts)} rows, {(time.time() - t0) / 60:.1f} min", flush=True)
    return res


# ---- GPU: the SFT pools behind part 2's vectors, plain and behind the prefix ------------------------
POOL_N, POOL_SEED = 1000, 0


def sft_pools(tpl):
    """{(dataset, pole): [(i, prompt, gold)]}: a seeded draw of up to POOL_N rows per pole. Tülu's template: WildGuardMix's
    in-SFT rows with the mixture turn (mia_topk pools) and CoCoNot's in-mixture pools (mia_coconot); Meta's template:
    WildGuardMix's in-SFT rows with their release response (mia_quartile rows, kind 'release') and the same CoCoNot pools."""
    import mia_coconot as CC
    rng = np.random.default_rng(POOL_SEED); out = {}
    if tpl == "tulu":
        for pole in POLES2:
            rows = [json.loads(l) for l in open(HERE / "data" / "mia_topk" / f"pool_{pole}.jsonl")]
            out[("WildGuardMix", pole)] = [(i, r["prompt"], r["gold"]) for i, r in enumerate(rows)]
    else:
        rows = [json.loads(l) for l in open(HERE / "data" / "mia_quartile" / "wildguardmix" / "rows.jsonl")]
        for pole in POLES2:
            out[("WildGuardMix", pole)] = [(r["i"], r["prompt"], r["gold"]) for r in rows if r["kind"] == "release" and r["pole"] == pole]
    for pole in POLES2:
        out[("CoCoNot, in-SFT", pole)] = [(r["i"], r["prompt"], r["gold"]) for r in CC.pool() if r["pole"] == pole]
    return {k: [v[j] for j in sorted(rng.choice(len(v), min(POOL_N, len(v)), replace=False))] for k, v in out.items()}


def pool_spans(model_key):
    """Our score's NLLs on the SFT pools' gold turns, plain and behind the runner's prefix, rendered as build() renders
    CAA (chat template, BOS string stripped, 512-token cap). -> results/caa_ours/pool_spans_<model>.jsonl"""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    mid, tpl = OUR_MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(TEMPLATES[tpl][0]); prefix = runner_prefix(tpl)
    keys, items = [], []
    for (ds, pole), rows in sft_pools(tpl).items():
        for i, p, g in rows:
            s = tok.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": g}], tokenize=False)
            s = s[len(tok.bos_token):] if tok.bos_token and s.startswith(tok.bos_token) else s
            ids = tok(s, add_special_tokens=False).input_ids
            t = tok.decode(ids[:MAX_TOK]) if len(ids) > MAX_TOK else s
            s_p = t.find(p.strip()); s_g = t.find(g.strip()[:200], s_p + len(p.strip()))
            sp = {"prompt": (s_p, s_p + len(p.strip())), "resp": (s_g, min(s_g + len(g.strip()), len(t)) if s_g >= 0 else -1), "text": (0, len(t))}
            keys.append({"ds": ds, "pole": pole, "i": i}); items.append(_with_prefix(t, sp, prefix))
    res = _passes(mid, items, model_key)
    OURS.mkdir(parents=True, exist_ok=True)
    with open(OURS / f"pool_spans_{model_key}.jsonl", "w") as f:
        for k, r in zip(keys, res):
            f.write(json.dumps({**k, **r}) + "\n")


def pool_scores():
    """Per (dataset, template, pole): our score per row (base - target per token on the gold turn), plain and behind the
    prefix. -> {(ds, tpl): {"plain"|"prefix": {pole: array}}}"""
    out = {}
    for tpl, (tgt, ref) in OUR_SCORE.items():
        f_t, f_r = OURS / f"pool_spans_{tgt}.jsonl", OURS / f"pool_spans_{ref}.jsonl"
        if not (f_t.exists() and f_r.exists()):
            continue
        Tg = {(r["ds"], r["pole"], r["i"]): r for r in map(json.loads, open(f_t))}
        Rf = {(r["ds"], r["pole"], r["i"]): r for r in map(json.loads, open(f_r))}
        for (ds, pole, i), t in Tg.items():
            r = Rf.get((ds, pole, i))
            if r is None or t["n_resp_plain"] == 0:
                continue
            for name in ("plain", "prefix"):
                v = r[f"nll_resp_{name}"] / r[f"n_resp_{name}"] - t[f"nll_resp_{name}"] / t[f"n_resp_{name}"]
                out.setdefault((ds, tpl), {}).setdefault(name, {}).setdefault(pole, []).append(v)
    return {k: {n: {p: np.array(v) for p, v in d.items()} for n, d in x.items()} for k, x in out.items()}


def our_scores():
    """One row per (pair, pole): per-token NLL of each model on resp / text, plain and behind the prefix; our score
    NLL_base - NLL_target (gold turn; text for the comparison with the runner); the runner-style ReCaLL
    rebuilt from the spans (mean NLL over prefix + text / mean NLL over text) as a check of the reconstruction."""
    import pandas as pd
    ps = load_pairs()
    D = pd.DataFrame([{"pair": r["pair"], "split": r["split"], "subset": r["subset"], "pole": pole} for r in ps for pole in POLES2])
    for m in OUR_MODELS:
        f = OURS / f"spans_{m}.jsonl"
        if not f.exists():
            continue
        S = pd.DataFrame([json.loads(l) for l in open(f)]).sort_values("row").reset_index(drop=True)
        for v in ("resp", "text"):
            for name in ("plain", "prefix"):
                D[f"{m}:nll_{v}_{name}"] = S[f"nll_{v}_{name}"] / S[f"n_{v}_{name}"]
        D[f"{m}:recall_rebuilt"] = ((S["nll_pre_prefix"] + S["nll_text_prefix"]) / (S["n_pre_prefix"] + S["n_text_prefix"])) / D[f"{m}:nll_text_plain"]
    for tpl, (tgt, ref) in OUR_SCORE.items():
        for v in ("resp", "text"):
            for name in ("plain", "prefix"):
                if f"{tgt}:nll_{v}_{name}" in D and f"{ref}:nll_{v}_{name}" in D:
                    D[f"{tpl}:ours_{v}_{name}"] = D[f"{ref}:nll_{v}_{name}"] - D[f"{tgt}:nll_{v}_{name}"]
    return D


# ---- CPU: summary ----------------------------------------------------------------------------------------
def scores():
    """One row per (pair, pole): every attack score per template and model (scores.json rows 2k / 2k+1 = pair k)."""
    import pandas as pd
    ps = load_pairs()
    D = pd.DataFrame([{"pair": r["pair"], "split": r["split"], "subset": r["subset"], "pole": pole} for r in ps for pole in POLES2])
    for tpl, models in RUNS.items():
        for m in models:
            f = RES / tpl / m / "scores.json"
            if not f.exists():
                continue
            s = json.load(open(f))["caa_chat"]
            assert s["label"] == [int(p == "refuse") for p in D.pole], f"{f}: row order differs from the pairs"
            for a in ("nlloss",) + ATTACKS:
                if a in s:
                    D[f"{tpl}:{m}:{a}"] = s[a]
        tgt, ref = models
        for a in ATTACKS:
            if f"{tpl}:{tgt}:{a}" in D and f"{tpl}:{ref}:{a}" in D:
                D[f"{tpl}:{tgt}-{ref}:{a}"] = D[f"{tpl}:{tgt}:{a}"] - D[f"{tpl}:{ref}:{a}"]
    return D


def summary():
    """Per pole (refuse, comply) and for the paired difference refuse - comply: mean, s.e. and N of every runner score
    (target, base, target - base) and of our score (per-token NLL per model, base - target, plain and behind the prefix),
    overall, per split and per subset. -> results/caa_panel/{pole_summary,diff_summary}.csv"""
    import pandas as pd
    D = scores(); O = our_scores()
    D = pd.concat([D, O[[c for c in O.columns if ":" in c]]], axis=1)
    cols = [c for c in D.columns if ":" in c and not c.endswith(":nlloss")]
    W = D.pivot(index=["pair", "split", "subset"], columns="pole", values=cols)
    PD = (W.xs("refuse", axis=1, level=1) - W.xs("comply", axis=1, level=1)).reset_index()
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    pole_rows, diff_rows = [], []
    groups = [("all", "all")] + [("split", s) for s in sorted(D.split.unique())] + [("subset", s) for s in sorted(D.subset.unique())]
    for by, sub in groups:
        sel = D if by == "all" else D[D[by] == sub]; pds = PD if by == "all" else PD[PD[by] == sub]
        for c in cols:
            for pole in POLES2:
                x = sel.loc[sel.pole == pole, c].dropna().values
                if len(x):
                    pole_rows.append({"by": by, "subset": sub, "quantity": c, "pole": pole, "N": len(x), "mean": x.mean(), "se": se(x)})
            x = pds[c].dropna().values
            if len(x):
                g = (x > 0).mean()
                diff_rows.append({"by": by, "subset": sub, "quantity": c, "N": len(x), "diff": x.mean(), "se": se(x),
                                  "share_gt0": g, "share_se": np.sqrt(g * (1 - g) / len(x))})
    PS, DS = pd.DataFrame(pole_rows), pd.DataFrame(diff_rows); RES.mkdir(parents=True, exist_ok=True)
    D.to_csv(RES / "scores.csv", index=False); PD.to_csv(RES / "paired.csv", index=False)
    PS.to_csv(RES / "pole_summary.csv", index=False); DS.to_csv(RES / "diff_summary.csv", index=False)
    wide = PS.pivot_table(index=["by", "subset", "quantity"], columns="pole", values=["mean", "se", "N"]).reset_index()
    wide.columns = [f"{a}_{b}" if b else a for a, b in wide.columns]
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_rows", 1000):
        print(wide[["by", "subset", "quantity", "N_refuse", "mean_refuse", "se_refuse", "N_comply", "mean_comply", "se_comply"]].round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "spans", "pool_spans", "summary"])
    ap.add_argument("--model", choices=list(OUR_MODELS))
    a = ap.parse_args()
    {"build": build, "spans": lambda: spans(a.model), "pool_spans": lambda: pool_spans(a.model), "summary": summary}[a.stage]()
