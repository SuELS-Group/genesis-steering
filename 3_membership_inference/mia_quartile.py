"""Quartiles of the membership score: direction, steering, the factors behind the score, the geometry of the selected
poles. One dataset at a time; WildGuardMix first, then every set of part 2.

WildGuardMix
  members      pools: every WildGuardMix row of Tülu's SFT mixture on both poles, gold = the mixture's
               assistant turn (Tülu regenerated WildGuardMix's responses), scored under Tülu-3-8B final / base / SFT /
               twin by mia_topk.py (reused here, not recomputed)
  non-members  membership.py's pools (WildGuardMix-train rows whose prompt is nowhere in the mixture, decontaminated
               against the evaluation prompts), gold = WildGuardMix's own response, kept only if it satisfies the pole
               rule (refusal / clean compliance)
  release      every member whose WildGuardMix release response satisfies the pole rule, scored on that response instead
               of the mixture turn (the generator of the non-members): members vs non-members with the generator matched,
               and the only member text Meta-Llama is scored on (Tülu's regenerated turns are Tülu-specific)
  scores       per token over the gold assistant turn, conditioned on the prompt (mia_topk's spans recipe)
                 clean      NLL_twin - NLL_sft            (Tülu; SFT vs its no-safety twin)
                 released   NLL_base - NLL_final          (Tülu; final vs the Llama-3.1-8B base)
                 meta       NLL_base - NLL_meta           (Meta-Llama-3.1-8B-Instruct vs the same base, Meta's template;
                                                           release-response rows only)
               residualized within each pole on log response tokens, log prompt tokens and the reference model's
               per-token NLL, the regression fitted on the score's population (FIT: members with the mixture turn for
               clean / released, members with the release response for meta) and applied to every row
  quartiles    per score and pole, the population split into quartiles Q1 (lowest) .. Q4 (highest) of the residual; from
               each quartile 128 drawn at random, stratified by response-length decile (the population's decile shares)
  arms         Tülu-3-8B: q_clean_Q1..Q4, q_released_Q1..Q4 (yardstick: mia_topk's rand0_k128 / rand1_k128, already
               steered on the same grid); Meta-Llama: q_meta_Q1..Q4 and q_meta_rand0 / q_meta_rand1 (random length-
               stratified draws from the same release population)
  steering     mia_topk's reduced grid: HarmBench (-alpha) and XSTest-safe (+alpha), first 50 prompts, |alpha| = 4/6/8,
               plus the unsteered cells; part 2's recipe (hidden_states[13], last prompt token, unit norm, 256 tokens)

Stages (GPU unless noted; resume-safe; one GPU job at a time)
    python mia_quartile.py pools                       # CPU: non-member and release rows -> data/mia_quartile/<set>/
    python mia_quartile.py spans --model final|base|sft|twin|meta|meta_base
    python mia_quartile.py acts --model tulu_final|meta_instruct   # hidden_states[13, 16, 20] of every prompt
    python mia_quartile.py select                      # CPU: scores, quartile sets
    python mia_quartile.py steer --model tulu_final|meta_instruct
    python ../judge.py --dump results/mia_quartile/<set>/steer_completions.jsonl --out .../steer_judge.jsonl
Not committed (answers to harmful requests, regenerated files): results/mia_quartile/, data/mia_quartile/*/rows.jsonl.
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
import mia_topk as K         # noqa: E402
import poles as P            # noqa: E402
import steering as T         # noqa: E402

SET = "wildguardmix"
DATA = HERE / "data" / "mia_quartile" / SET; SETS = DATA / "sets"
OUT = HERE / "results" / "mia_quartile" / SET
STEER_COMP, STEER_JUDGE = OUT / "steer_completions.jsonl", OUT / "steer_judge.jsonl"
LAYERS = [T.LAYER, 16, 20]                     # hidden_states indices (part 2 convention: input of block L)
MODELS = {**K.MODELS, "meta": "meta-llama/Llama-3.1-8B-Instruct", "meta_base": "meta-llama/Llama-3.1-8B"}
TEMPLATE = {**{m: K.TEMPLATE for m in K.MODELS}, "meta": MODELS["meta"], "meta_base": MODELS["meta"]}
SCORE_DEF = {"clean": ("sft", "twin"), "released": ("final", "base"), "meta": ("meta", "meta_base")}   # (target, reference)
STEER_MODEL = {"clean": "tulu_final", "released": "tulu_final", "meta": "meta_instruct"}
FIT = {"clean": "member", "released": "member", "meta": "release"}     # the population each score is ranked within
COUNTS = {"clean": "final", "released": "final", "meta": "meta"}        # whose token counts enter a score's regression
                                                                        # (Tülu: final's, as mia_topk.scores)
POP_KIND = {"member": "clean", "release": "meta"}                       # the score whose counts bin each population
POLES2, QS, N_DRAW, SEED = K.POLES2, (1, 2, 3, 4), P.N_SRC, 0


# ---- CPU: rows ---------------------------------------------------------------------------------------
def pools():
    """data/mia_quartile/<set>/rows.jsonl: the non-member rows and every member with a usable release response, with
    the hub's adversarial flag and subcategory."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset
    import membership as MB
    hub_all = {}                                      # a prompt can occur several times, with and without a response
    for x in load_dataset("allenai/wildguardmix", "wildguardtrain", split="train"):
        if x["prompt"]:
            hub_all.setdefault(P._norm(x["prompt"]), []).append(x)
    rule = {"refuse": P.is_refusal, "comply": P.is_clean_compliance}
    def usable(prompt, pole):                         # the first copy whose response satisfies the pole rule
        return next((x for x in hub_all.get(P._norm(prompt), []) if x["response"] and rule[pole](x["response"])), None)
    rows = []
    for pole in POLES2:
        for r in MB.pool("nonmember", pole):
            h = usable(r["prompt"], pole)
            if h:
                rows.append({"kind": "nonmember", "pole": pole, "prompt": r["prompt"], "gold": h["response"],
                             "adversarial": str(h["adversarial"]) == "True", "subcategory": h["subcategory"] or ""})
        mem = K.pool(pole)
        rel = {i: h for i, r in enumerate(mem) if (h := usable(r["prompt"], pole))}
        for i in sorted(rel):
            h = rel[i]
            rows.append({"kind": "release", "pole": pole, "prompt": mem[i]["prompt"], "gold": h["response"], "member_i": int(i),
                         "adversarial": str(h["adversarial"]) == "True", "subcategory": h["subcategory"] or ""})
        print(f"{pole}: {sum(r['kind'] == 'nonmember' and r['pole'] == pole for r in rows)} of {len(MB.pool('nonmember', pole))} non-members "
              f"with a usable response; members with a usable release response {len(rel)} of {len(mem)}")
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "rows.jsonl", "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"i": i, **r}, ensure_ascii=False) + "\n")
    with open(DATA / "member_meta.jsonl", "w") as f:            # hub fields of every member row
        for pole in POLES2:
            for i, r in enumerate(K.pool(pole)):
                hs = hub_all.get(P._norm(r["prompt"]), [{}]); h = usable(r["prompt"], pole) or hs[0]
                f.write(json.dumps({"pole": pole, "i": i, "adversarial": str(h.get("adversarial")) == "True",
                                    "subcategory": h.get("subcategory") or "", "release_response": h.get("response") or "",
                                    "release_matches_pole": usable(r["prompt"], pole) is not None},
                                   ensure_ascii=False) + "\n")


def rows():
    return [json.loads(l) for l in open(DATA / "rows.jsonl")]


# ---- GPU: assistant-turn NLL (mia_topk.spans, generalised to any rows and template) ---------------------
def spans(model_key):
    """The rows of rows.jsonl (non-members, members with their release response), in the model's template (Meta models:
    Meta-Llama-3.1-Instruct's). Tülu's members with the mixture turn are in mia_topk's spans files and are not scored
    on Meta. Output keyed by (src, pole, i), src 'row', i into rows.jsonl."""
    score_items(model_key, [("row", r["pole"], r["i"], r["prompt"], r["gold"]) for r in rows()], OUT / f"spans_{model_key}.jsonl")


def score_items(model_key, all_items, out_f):
    """Per item (src, pole, i, prompt, gold): NLL summed over the gold assistant turn and over the prompt, in the model's
    template (TEMPLATE), cut at 512 tokens (mia_topk.spans' recipe); appended to out_f, resume-safe on (src, pole, i)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    done = {(r["src"], r["pole"], r["i"]) for r in map(json.loads, open(out_f))} if Path(out_f).exists() else set()
    items = [x for x in all_items if x[:3] not in done]
    if not items:
        print(f"[{model_key}] spans complete"); return
    ttok = AutoTokenizer.from_pretrained(TEMPLATE[model_key])
    texts, sp = [], []
    for _, _, _, p, g in items:
        t = ttok.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": g}], tokenize=False)
        enc = ttok(t, add_special_tokens=False, return_offsets_mapping=True)
        t = t[:enc["offset_mapping"][K.MAX_TOK - 1][1]] if len(enc["input_ids"]) > K.MAX_TOK else t
        ps = t.find(p.strip()); pe = ps + len(p.strip())
        gs = t.find(g.strip()[:200], max(pe, 0)) if pe <= len(t) else -1
        ge = min(gs + len(g.strip()), len(t)) if gs >= 0 else -1
        texts.append(t); sp.append(((ps, min(pe, len(t))), (gs, ge)))
    gpu_guard()
    tok = AutoTokenizer.from_pretrained(MODELS[model_key])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODELS[model_key], dtype=torch.bfloat16, device_map="cuda").eval()
    order = sorted(range(len(items)), key=lambda k: len(texts[k]))
    Path(out_f).parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
    with open(out_f, "a") as f:
        for b in range(0, len(order), K.BATCH):
            ks = order[b:b + K.BATCH]
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
                src, pole, i = items[k][:3]
                f.write(json.dumps({"src": src, "pole": pole, "i": i, "nll_prompt": float(nll[j][inp].sum()), "n_tok_prompt": int(inp.sum()),
                                    "nll_resp": float(nll[j][ing].sum()), "n_tok_resp": int(ing.sum())}) + "\n")
            del logits, nll; torch.cuda.empty_cache()
            if (b // K.BATCH) % 200 == 0:
                print(f"  [{model_key}] {b + len(ks)}/{len(order)}  {(b + len(ks)) / (time.time() - t0):.1f} ex/s", flush=True)
    print(f"[{model_key}] {len(items)} examples in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---- GPU: activations at the steering site and two later layers ------------------------------------
def acts_file(model):
    return OUT / f"acts_{model}_L{'_'.join(map(str, LAYERS))}.npz"


GPU_CAP_GB = 60      # per-process ceiling for the CUDA allocator: on the Spark's unified memory the cgroup cap does not see GPU
                     # allocations, and an unbounded process starves the whole machine


def gpu_guard():
    """Cap this process's CUDA allocator at GPU_CAP_GB, so an overrun raises an out-of-memory error in this process."""
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_CAP_GB * 1e9 / total))


class _Stop(Exception):
    pass


def extract(m, tok, rendered, layers, bs=T.EXTRACT_BS, partial=None, every=500):
    """{L: float16 [n, d]} = hidden_states[L] (the input of decoder block L) at the last prompt token, for each layer in
    `layers`, by forward pre-hooks on the decoder blocks; the forward pass stops after the deepest requested block, so
    no later block and no logits are computed. Right padding, 512-token cut, the tokenizer's special tokens (install.py's
    recipe). `partial`: a .npz path checkpointed every `every` batches, from which an interrupted run resumes."""
    import torch
    order = np.argsort([len(tok(r, add_special_tokens=True)["input_ids"]) for r in rendered], kind="stable")
    tok.padding_side = "right"
    X = {L: np.zeros((len(rendered), m.config.hidden_size), dtype=np.float16) for L in layers}; start = 0
    if partial is not None and Path(partial).exists():
        z = np.load(partial); start = int(z["next"]); X = {L: z[f"L{L}"] for L in layers}
        print(f"  resuming at batch offset {start}/{len(order)}", flush=True)
    state, deepest = {}, max(layers)

    def hook(L):
        def f(mod, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            X[L][state["idx"]] = h[torch.arange(h.shape[0], device=h.device), state["last"]].float().cpu().numpy()
            if L == deepest:
                raise _Stop
        return f
    blocks = m.model.layers
    handles = [blocks[L].register_forward_pre_hook(hook(L), with_kwargs=True) for L in layers]
    try:
        with torch.inference_mode():
            for n, s in enumerate(range(start, len(order), bs), 1):
                idx = order[s:s + bs].tolist()
                enc = tok([rendered[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512,
                          add_special_tokens=True).to("cuda")
                state["idx"], state["last"] = idx, (enc["attention_mask"].sum(1) - 1).clamp(min=0)
                try:
                    m.model(**enc)
                except _Stop:
                    pass
                del enc; torch.cuda.empty_cache()
                if partial is not None and n % every == 0:
                    np.savez(partial, next=s + bs, **{f"L{L}": X[L] for L in layers})
                    print(f"  {s + bs}/{len(order)}  (allocator peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)", flush=True)
    finally:
        for h in handles:
            h.remove()
    return X


def acts(model):
    """hidden_states[L] for L in LAYERS (13 = part 2's steering site, the input of block 13; 16 and 20 for the layer
    ablation), last prompt token, of every prompt of the member pools and of rows.jsonl, on `model` (tulu_final or
    meta_instruct), with `extract` (resume-safe)."""
    f = acts_file(model)
    if f.exists():
        print(f"{f.name} exists"); return
    gpu_guard()
    prompts = list(dict.fromkeys([r["prompt"] for pole in POLES2 for r in K.pool(pole)] + [r["prompt"] for r in rows()]))
    m, tok = T.load_model(T.MODELS[model])
    OUT.mkdir(parents=True, exist_ok=True); part = OUT / f"acts_{model}.partial.npz"
    X = extract(m, tok, T.render(prompts, tok), LAYERS, partial=part)
    np.savez(f, **{f"L{L}": X[L] for L in LAYERS}, prompts=np.array(json.dumps(prompts)))
    part.unlink(missing_ok=True)
    print(f"{model} acts: {len(prompts)} prompts, layers {LAYERS}")


def load_acts(model, layer=T.LAYER):
    """(X, {prompt: row}) at `layer` for the member pools and rows.jsonl on `model`."""
    z = np.load(acts_file(model))
    return z[f"L{layer}"].astype(np.float32), {p: i for i, p in enumerate(json.loads(str(z["prompts"])))}


# ---- CPU: scores and quartile sets -------------------------------------------------------------------
def _spans_table(model_key):
    if model_key in K.MODELS:
        S = {("member", r["pole"], r["i"]): r for r in map(json.loads, open(K.OUT / f"spans_{model_key}.jsonl"))}
    else:
        S = {}
    f = OUT / f"spans_{model_key}.jsonl"
    if f.exists():
        S.update({(r["src"], r["pole"], r["i"]): r for r in map(json.loads, open(f))})
    return S


def scores():
    """DataFrame per row (members, non-members, release sample): kind, pole, prompt, lengths, per score the raw value,
    the reference per-token NLL and the residual (regression fitted within the pole on the score's FIT population);
    response-length decile within each FIT population (decile_member, decile_release), on the token counts of the
    score that ranks it (COUNTS), so a decile never changes when another model's spans arrive. n_resp / n_prompt:
    Tülu final's counts where present, else Meta's (same tokenizer family; for display). Missing spans give NaN."""
    import pandas as pd
    S = {m: _spans_table(m) for m in MODELS}
    base = [{"kind": "member", "src": "member", "pole": pole, "i": i, "prompt": r["prompt"], "gold": r["gold"]}
            for pole in POLES2 for i, r in enumerate(K.pool(pole))]
    base += [{**r, "src": "row"} for r in rows()]
    out = []
    for r in base:
        k = (r["src"], r["pole"], r["i"]); row = {x: r.get(x) for x in ("kind", "src", "pole", "i", "prompt", "member_i")}
        row["n_char_resp"] = len(r["gold"]); row["n_char_prompt"] = len(r["prompt"])
        for kind in SCORE_DEF:
            c = S[COUNTS[kind]].get(k)
            row[f"{kind}_n_resp"] = c["n_tok_resp"] if c else np.nan; row[f"{kind}_n_prompt"] = c["n_tok_prompt"] if c else np.nan
        sf = S["final"].get(k) or S["meta"].get(k)
        row["n_resp"] = sf["n_tok_resp"] if sf else np.nan; row["n_prompt"] = sf["n_tok_prompt"] if sf else np.nan
        for kind, (tgt, ref) in SCORE_DEF.items():
            a, b = S[tgt].get(k), S[ref].get(k)
            if a and b and a["n_tok_resp"] >= 5:
                row[f"{kind}_raw"] = (b["nll_resp"] - a["nll_resp"]) / a["n_tok_resp"]
                row[f"{kind}_refnll"] = b["nll_resp"] / b["n_tok_resp"]
            else:
                row[f"{kind}_raw"] = row[f"{kind}_refnll"] = np.nan
        out.append(row)
    D = pd.DataFrame(out)
    for pole in POLES2:
        sel = D.pole == pole
        for kind in SCORE_DEF:
            X = lambda d: np.c_[np.ones(len(d)), np.log(d[f"{kind}_n_resp"]), np.log(d[f"{kind}_n_prompt"]), d[f"{kind}_refnll"]]
            fit = D[sel & (D.kind == FIT[kind])].dropna(subset=[f"{kind}_raw", f"{kind}_n_resp"])
            if not len(fit):
                D.loc[sel, kind] = np.nan; continue
            beta, *_ = np.linalg.lstsq(X(fit), fit[f"{kind}_raw"].values, rcond=None)
            ok = sel & D[f"{kind}_raw"].notna()
            D.loc[ok, kind] = D.loc[ok, f"{kind}_raw"].values - X(D[ok]) @ beta
        for pop, kind in POP_KIND.items():                    # each population's deciles, from its ranking score's counts
            n = D[f"{kind}_n_resp"]
            m = sel & (D.kind == pop) & (n >= 5)              # mia_topk.scores bins the rows it scores (>= 5 response tokens)
            if m.any():
                D.loc[m, f"decile_{pop}"] = pd.qcut(n[m].rank(method="first"), K.N_DECILES, labels=False)
    return D


def with_prompts(D):
    """scores.csv without its text columns -> the same frame with each row's prompt (members: mia_topk.pool order;
    other rows: rows.jsonl order)."""
    mem = {pole: [r["prompt"] for r in K.pool(pole)] for pole in POLES2}; rw = {r["i"]: r["prompt"] for r in rows()}
    D = D.copy()
    D["prompt"] = [mem[p][int(i)] if s == "member" else rw[int(i)] for s, p, i in zip(D.src, D.pole, D.i)]
    return D


def stratified_draw(d, n, rng, col="decile_member"):
    """n prompts of d drawn at random, the same share from each response-length decile (column col) as d itself."""
    quota = d.groupby(col).size() / len(d) * n; q = np.floor(quota).astype(int)
    for dec in (quota - q).sort_values(ascending=False).index[:n - q.sum()]:
        q[dec] += 1
    out = []
    for dec, k in q.items():
        ps = d[d[col] == dec]["prompt"].tolist()
        out += [ps[j] for j in rng.choice(len(ps), min(k, len(ps)), replace=False)]
    return out


def arms(model):
    kinds = [k for k, m in STEER_MODEL.items() if m == model]
    if model == "tulu_final":        # the top 128 of clean / released = mia_topk's mia_*_k128, already steered on this grid
        return [f"q_{k}_Q{q}" for k in kinds for q in QS] + [f"x_{k}_bottom" for k in kinds]
    return [f"q_{k}_Q{q}" for k in kinds for q in QS] + ["q_meta_rand0", "q_meta_rand1"] + [f"x_{k}_{e}" for k in kinds for e in ("top", "bottom")]


def select():
    import pandas as pd
    D = scores(); SETS.mkdir(parents=True, exist_ok=True)
    for kind in SCORE_DEF:
        for pole_j, pole in enumerate(POLES2):
            col = f"decile_{FIT[kind]}"
            if not complete(kind):
                print(f"{kind}: spans incomplete, skipped"); break
            d = D[(D.kind == FIT[kind]) & (D.pole == pole)].dropna(subset=[kind, col]).copy()
            d["quartile"] = 1 + pd.qcut(d[kind].rank(method="first"), 4, labels=False).values
            draws = {f"q_{kind}_Q{q}": (d[d.quartile == q], 10_000 * (q + 1)) for q in QS}
            dd = d.rename(columns={col: "decile"})           # the extremes: mia_topk.stratified_top, and the same on the negated score
            ext = {f"x_{kind}_top": K.stratified_top(dd, kind, N_DRAW),
                   f"x_{kind}_bottom": K.stratified_top(dd.assign(neg=-dd[kind]), "neg", N_DRAW)}
            if kind in ("clean", "released"):
                assert set(ext[f"x_{kind}_top"]) == set(K.load_set(f"mia_{kind}_k128")[pole]), "top 128 differs from mia_topk"
            if kind == "meta":                       # Meta's yardstick: random draws from the same release population
                draws.update({f"q_meta_rand{s}": (d, 777 + 10 * s) for s in (0, 1)})
            for arm, (dd, seed) in draws.items():
                rng = np.random.default_rng(seed + 100 * list(SCORE_DEF).index(kind) + pole_j)
                pick = stratified_draw(dd, N_DRAW, rng, col)
                ext[arm] = pick
            for arm, pick in ext.items():
                with open(SETS / f"{arm}.jsonl", "w" if pole_j == 0 else "a") as f:
                    for p in pick:
                        f.write(json.dumps({"pole": pole, "prompt": p}, ensure_ascii=False) + "\n")
    D.drop(columns=["prompt"]).to_csv(OUT / "scores.csv", index=False)
    print("scored rows per kind and pole:", D.groupby(["kind", "pole"]).size().to_dict())


def complete(kind):
    """True when both models of a score have spans for every row they score (no selection from a partial pass)."""
    tgt, ref = SCORE_DEF[kind]
    need = {("member", pole, i) for pole in POLES2 for i in range(len(K.pool(pole)))} if FIT[kind] == "member" else set()
    need |= {("row", r["pole"], r["i"]) for r in rows()} if (kind == "meta" or FIT[kind] == "release") else set()
    return all(need <= set(_spans_table(m)) for m in (tgt, ref))


def load_set(arm):
    if arm.startswith("rand"):
        return K.load_set(arm)
    rows_ = [json.loads(l) for l in open(SETS / f"{arm}.jsonl")]
    return {pole: [r["prompt"] for r in rows_ if r["pole"] == pole] for pole in POLES2}


def directions(model):
    X, ix = load_acts(model); out = {}
    for a in arms(model):
        s = load_set(a)
        v = X[[ix[p] for p in s["refuse"]]].mean(0) - X[[ix[p] for p in s["comply"]]].mean(0)
        out[a] = (v / np.linalg.norm(v)).astype(np.float32)
    return out


# ---- CPU: the geometry of the top against the bottom 128 ------------
def plot_extremes(ax, model, kind, layer, D, n_bg=2000, seed=0):
    """Draw, in the plane spanned by the top-128 vector (horizontal) and the part of the bottom-128 vector orthogonal to
    it (vertical), through the midpoint of the population's pole means: the population (grey, n_bg per pole), the top
    (blue) and bottom (orange) refusals (filled) and compliances (open) with their centroids, and the three unit vectors
    (top, bottom, whole population; arrows). D: scores() with prompts. Returns the cosines drawn."""
    pop = FIT[kind]
    X, ix = load_acts(model, layer); R = lambda ps: X[[ix[p] for p in ps]]
    pool = {pole: D[(D.kind == pop) & (D.pole == pole)].dropna(subset=[kind]).prompt.tolist() for pole in POLES2}
    H = {pole: R(pool[pole]) for pole in POLES2}
    unit = lambda v: v / np.linalg.norm(v)
    vp = unit(H["refuse"].mean(0) - H["comply"].mean(0))
    S = {e: load_set(f"x_{kind}_{e}") for e in ("top", "bottom")}
    v = {e: unit(R(S[e]["refuse"]).mean(0) - R(S[e]["comply"]).mean(0)) for e in S}
    e1 = v["top"]; e2 = unit(v["bottom"] - (v["bottom"] @ e1) * e1)
    o = (H["refuse"].mean(0) + H["comply"].mean(0)) / 2
    proj = lambda A: np.c_[(A - o) @ e1, (A - o) @ e2]
    rng = np.random.default_rng(seed)
    for pole in POLES2:
        P_ = proj(H[pole][rng.choice(len(H[pole]), min(n_bg, len(H[pole])), replace=False)])
        ax.scatter(P_[:, 0], P_[:, 1], s=3, color="#d6d5d2", alpha=.5, rasterized=True, zorder=1)
    col = {"top": "#2a78d6", "bottom": "#eb6834"}
    for e in S:
        for pole in POLES2:
            P_ = proj(R(S[e][pole]))
            ax.scatter(P_[:, 0], P_[:, 1], s=11, facecolor=col[e] if pole == "refuse" else "none", edgecolor=col[e], lw=.7,
                       label=f"{e} 128, {'refusals' if pole == 'refuse' else 'compliances'}", zorder=3)
            ax.scatter(*P_.mean(0), s=150, marker="X", color=col[e], edgecolor="white", lw=1.2, zorder=5)
    L_ = 0.3 * np.ptp(proj(np.r_[H["refuse"], H["comply"]])[:, 0])
    for name, u, c in (("top", v["top"], col["top"]), ("bottom", v["bottom"], col["bottom"]), ("population", vp, "#0b0b0b")):
        ax.annotate("", xy=(L_ * u @ e1, L_ * u @ e2), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2), zorder=6)
    ax.set_xlabel("along the top-128 vector"); ax.set_ylabel("bottom-128 vector, part orthogonal to the top")
    ax.spines[["top", "right"]].set_visible(False)
    adv = adversarial_map(); mu = {(e, pole): R(S[e][pole]).mean(0) for e in S for pole in POLES2}
    return {"cos(top, bottom)": float(v["top"] @ v["bottom"]), "cos(top, pool)": float(v["top"] @ vp), "cos(bottom, pool)": float(v["bottom"] @ vp),
            **{f"{pole}: centroid distance top–bottom": float(np.linalg.norm(mu[("top", pole)] - mu[("bottom", pole)])) for pole in POLES2},
            **{f"{pole}: adversarial share {e}": float(np.mean([adv[p] for p in S[e][pole]])) for e in S for pole in POLES2}}


_BOUNDARY = {}


def boundary_fit(Hr, Hc):
    """Logistic regression (L2, C = 1, standardized inputs) of refusal (Hr) vs compliance (Hc) activations; returned in raw
    units as (w, b) with w.x + b = 0 the boundary, and its 5-fold held-out accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    Z = np.r_[Hr, Hc]; y = np.r_[np.ones(len(Hr)), np.zeros(len(Hc))]
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    clf = LogisticRegression(C=1.0, max_iter=3000).fit((Z - mu) / sd, y)
    acc = cross_val_score(LogisticRegression(C=1.0, max_iter=3000), (Z - mu) / sd, y, cv=5).mean()
    w = clf.coef_[0] / sd
    return w, clf.intercept_[0] - w @ mu, float(acc)


def boundary(model, layer, D, kind):
    """boundary_fit on the score's population of WildGuardMix at `layer`, cached per (model, layer, population)."""
    key = (model, layer, FIT[kind])
    if key not in _BOUNDARY:
        X, ix = load_acts(model, layer); pop = FIT[kind]
        ps = {pole: D[(D.kind == pop) & (D.pole == pole)].prompt.tolist() for pole in POLES2}
        _BOUNDARY[key] = boundary_fit(X[[ix[p] for p in ps["refuse"]]], X[[ix[p] for p in ps["comply"]]])
    return _BOUNDARY[key]


def plot_pca_core(ax, H, SH, vecs, wb, size, n_bg=2000, seed=0):
    """The first two principal components of a pool H = {pole: activations} (both poles, centred; PC1 oriented towards
    the refusal pole): the pool (grey), the top (blue) and bottom (orange) selections SH = {"top"|"bottom": {pole: acts}}
    (filled: refusals, open: compliances) with centroids; wb = (w, b, acc) a boundary of the full activations drawn where it
    cuts the plane (solid), the difference-of-means boundary (dashed); arrows: the unit vectors vecs = {name: (vector,
    colour)} plus the whole pool's, projected. `size` labels the selections ("128", "24"). Returns the explained variance,
    the boundary's accuracy and each vector's share inside the plane."""
    unit = lambda v: v / np.linalg.norm(v)
    Z = np.r_[H["refuse"], H["comply"]]; m0 = Z.mean(0); rng = np.random.default_rng(seed)
    _, sv, Vt = np.linalg.svd(Z[rng.choice(len(Z), min(8000, len(Z)), replace=False)] - m0, full_matrices=False)
    pc = Vt[:2].copy(); ev = (sv[:2] ** 2) / (sv ** 2).sum()
    if (H["refuse"].mean(0) - H["comply"].mean(0)) @ pc[0] < 0:
        pc[0] = -pc[0]
    proj = lambda A: (A - m0) @ pc.T
    for pole in POLES2:
        P2 = proj(H[pole][rng.choice(len(H[pole]), min(n_bg, len(H[pole])), replace=False)])
        ax.scatter(P2[:, 0], P2[:, 1], s=3, color="#d6d5d2", alpha=.5, rasterized=True, zorder=1)
    col = {"top": "#2a78d6", "bottom": "#eb6834"}
    for e in SH:
        for pole in POLES2:
            P2 = proj(SH[e][pole])
            ax.scatter(P2[:, 0], P2[:, 1], s=11, facecolor=col[e] if pole == "refuse" else "none", edgecolor=col[e], lw=.7, zorder=3,
                       label=f"{e} {size}, {'refusals' if pole == 'refuse' else 'compliances'}")
            ax.scatter(*P2.mean(0), s=150, marker="X", color=col[e], edgecolor="white", lw=1.2, zorder=5)
    lim = np.percentile(proj(Z), [0.1, 99.9], axis=0); pad = 0.05 * (lim[1] - lim[0]); lim = np.r_[[lim[0] - pad], [lim[1] + pad]]
    a = np.linspace(lim[0, 0], lim[1, 0], 200)
    def line(w, b, **kw):                                    # points m0 + a pc1 + c pc2 with w.x + b = 0
        wa, wc, w0 = w @ pc[0], w @ pc[1], w @ m0 + b
        if abs(wc) > 1e-12:
            c = -(w0 + wa * a) / wc; ok = (c > lim[0, 1]) & (c < lim[1, 1]); ax.plot(a[ok], c[ok], **kw)
        else:
            ax.axvline(-w0 / wa, **kw)
    w, b, acc = wb; w_in = float(np.linalg.norm(pc @ unit(w)))
    line(w, b, color="#0b0b0b", lw=1.6, zorder=4, label=f"logistic boundary, where it cuts the plane (held-out acc. {acc:.3f}; {w_in:.2f} of its normal in plane)")
    mr, mc = H["refuse"].mean(0), H["comply"].mean(0); vp = unit(mr - mc)
    line(vp, -vp @ (mr + mc) / 2, color="#0b0b0b", lw=1.2, ls="--", zorder=4, label="difference-of-means boundary")
    allv = {**{k: (unit(u), c) for k, (u, c) in vecs.items()}, "whole population": (vp, "#0b0b0b")}
    origin = proj(((mr + mc) / 2)[None])[0]; L_ = 0.35 * (lim[1, 0] - lim[0, 0]); share = {}
    for name, (u, c) in allv.items():
        q = pc @ u; share[name] = float(np.linalg.norm(q))
        ax.annotate("", xy=origin + L_ * q, xytext=origin, arrowprops=dict(arrowstyle="-|>", color=c, lw=2), zorder=6)
        ax.plot([], [], color=c, lw=2, label=f"{name} vector")
    ax.set_xlim(lim[0, 0], lim[1, 0]); ax.set_ylim(lim[0, 1], lim[1, 1])
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.0f} % of variance)"); ax.set_ylabel(f"PC2 ({100 * ev[1]:.0f} % of variance)")
    ax.spines[["top", "right"]].set_visible(False)
    return {"explained variance PC1": float(ev[0]), "explained variance PC2": float(ev[1]), "boundary held-out accuracy": acc,
            **{f"share in plane: {k}": v for k, v in share.items()},
            "cos(logistic normal, population vector)": float(unit(w) @ vp), "share in plane: logistic normal": w_in}


def plot_extremes_pca(ax, model, kind, layer, D, n_bg=2000, seed=0):
    """plot_pca_core for WildGuardMix: the score's population, its top / bottom 128, part 2's WildGuardMix vector."""
    import poles as P_
    X, ix = load_acts(model, layer); R = lambda ps: X[[ix[p] for p in ps]]; pop = FIT[kind]
    H = {pole: R(D[(D.kind == pop) & (D.pole == pole)].prompt.tolist()) for pole in POLES2}
    S = {e: load_set(f"x_{kind}_{e}") for e in ("top", "bottom")}
    SH = {e: {pole: R(S[e][pole]) for pole in POLES2} for e in S}
    vs = P_.load_vector_sets()["g_wildguardmix"]
    vecs = {"top 128": (SH["top"]["refuse"].mean(0) - SH["top"]["comply"].mean(0), "#2a78d6"),
            "bottom 128": (SH["bottom"]["refuse"].mean(0) - SH["bottom"]["comply"].mean(0), "#eb6834"),
            "part 2 steering": (R(vs["refuse"]).mean(0) - R(vs["comply"]).mean(0), "#2e9e6b")}
    return plot_pca_core(ax, H, SH, vecs, boundary(model, layer, D, kind), "128", n_bg, seed)


def atlas_vectors(layer, model="tulu_final"):
    """Every WildGuardMix vector at hidden_states[layer] on `model`: {name: (family, unit vector)} - part 2's WildGuardMix
    vector, the quartiles and extremes of both scores, top-k and random draws, member and
    non-member draws (all drawn from prompts in this file)."""
    import membership as MB
    X, ix = load_acts(model, layer); R = lambda ps: X[[ix[p] for p in ps]]
    vec = lambda s: (lambda v: v / np.linalg.norm(v))(R(s["refuse"]).mean(0) - R(s["comply"]).mean(0))
    out = {"part 2 g_wildguardmix": ("part 2 vector", vec(P.load_vector_sets()["g_wildguardmix"]))}
    for k in ("clean", "released"):
        for q in QS:
            if (SETS / f"q_{k}_Q{q}.jsonl").exists():
                out[f"{k} Q{q}"] = (f"{k} quartiles", vec(load_set(f"q_{k}_Q{q}")))
        for e in ("top", "bottom"):
            out[f"{k} {e} 128"] = (f"{e} 128", vec(load_set(f"x_{k}_{e}")))
    for a in K.arms():
        out[f"sweep {a}"] = ("sweep random draws" if a.startswith("rand") else "sweep top-k", vec(K.load_set(a)))
    for a in MB.arms():
        out[f"draw {a.replace('wgm_', '')}"] = ("member draws" if "_member_" in a else "non-member draws", vec(MB.load_draw(a)))
    return out


ATLAS_COLORS = {"part 2 vector": "#0b0b0b", "clean quartiles": "#2a78d6", "released quartiles": "#7fb0e8",
                "top 128": "#b04ad6", "bottom 128": "#eb6834", "sweep top-k": "#c9a0dc", "sweep random draws": "#b8b7b2",
                "member draws": "#2e9e6b", "non-member draws": "#9fd4b8",
                "top 24": "#b04ad6", "bottom 24": "#eb6834", "random 24": "#b8b7b2", "part 2 CoCoNot": "#0b0b0b", "MIA-selected": "#8a8984"}


def plot_atlas(ax, V, label=lambda name, fam: False):
    """PCA of the unit vectors V = {name: (family, vector)} centred on their mean; points coloured by family
    (ATLAS_COLORS), labels where label(name, family). Returns the variance shares and the 2-D coordinates."""
    names = list(V); M = np.stack([V[a][1] for a in names]); mu = M.mean(0)
    _, s, Vt = np.linalg.svd(M - mu, full_matrices=False); ev = s ** 2 / (s ** 2).sum(); P2 = (M - mu) @ Vt[:2].T
    for a, (x, y) in zip(names, P2):
        f = V[a][0]
        ax.scatter(x, y, s=46, color=ATLAS_COLORS.get(f, "#52514e"), edgecolor="white", lw=.7, zorder=3, label=f)
        if label(a, f):
            ax.annotate(a, (x, y), fontsize=6.5, xytext=(3, 2), textcoords="offset points", color="#52514e")
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.0f} %)"); ax.set_ylabel(f"PC2 ({100 * ev[1]:.0f} %)"); ax.spines[["top", "right"]].set_visible(False)
    return ev, dict(zip(names, map(tuple, P2)))


def adversarial_map():
    """{prompt: WildGuardMix's adversarial flag} over the members and rows.jsonl."""
    out = {r["prompt"]: r["adversarial"] for r in rows()}
    mm = {(r["pole"], r["i"]): r["adversarial"] for r in map(json.loads, open(DATA / "member_meta.jsonl"))}
    out.update({r["prompt"]: mm[(pole, i)] for pole in POLES2 for i, r in enumerate(K.pool(pole))})
    return out


# ---- CPU: text features of a row --------------------------------------------------------
import re  # noqa: E402
_F = {
    "opens with apology": r"^\W*(i'?m sorry|i am sorry|sorry|i apologi[sz]e|my apologies)",
    "opens with 'I can't / cannot'": r"^\W*(i can'?t|i cannot|i'?m (not able|unable)|i am (not able|unable)|i won'?t|i will not)",
    "AI self-reference": r"\b(as an ai|ai language model|language model|i'?m an ai|as a responsible ai|openai)\b",
    "'it's important to'": r"\b(it'?s|it is) (important|crucial|essential) to\b",
    "offers alternative": r"\b(instead|however, i can|i can help|i'?d be happy to|alternatively)\b",
    "markdown list": r"(^|\n)\s*([-*\u2022]|\d+[.)])\s+",
    "markdown bold/header": r"(\*\*[^*]+\*\*|(^|\n)#{1,4} )",
}
_P = {
    "prompt: role-play / fiction frame": r"\b(imagine|pretend|role-?play|you are a|as a (character|writer|historian|researcher)|in the capacity|story|fictional|novel|script)\b",
    "prompt: numbered tasks": r"(^|\s)(1[.)]|first,)\s",
    "prompt: research / academic claim": r"\b(research|academic|educational|study|purely hypothetical|for a (paper|class|project))\b",
}


def features(prompt, gold):
    """Model-free features of one (prompt, gold turn): lengths, refusal register, formatting, lexical diversity, prompt
    framing. Regexes are case-insensitive; booleans as 0/1."""
    g, p = gold.strip(), prompt.strip()
    gl, pl = g.lower(), p.lower()
    w = re.findall(r"[a-z']+", gl)
    out = {"response chars": len(g), "prompt chars": len(p),
           "response type-token ratio (first 100 words)": len(set(w[:100])) / max(len(w[:100]), 1),
           "response paragraphs": g.count("\n\n") + 1,
           "response first-person 'I' per 100 words": 100 * sum(x in ("i", "i'm", "i'd", "i'll", "me", "my") for x in w) / max(len(w), 1)}
    out.update({k: float(bool(re.search(r, gl, flags=re.M))) for k, r in _F.items()})
    out.update({k: float(bool(re.search(r, pl))) for k, r in _P.items()})
    return out


def similarity(a, b, n=400):
    """difflib ratio of the first n characters: how close Tülu's gold turn is to the release response."""
    import difflib
    return difflib.SequenceMatcher(None, a[:n], b[:n], autojunk=False).ratio()


# ---- GPU: steering -----------------------------------------------------------------------------------
def grid(model):
    cells = [(e, "baseline", 0) for e in K.EVALS]
    return cells + [(e, a, T.EVAL_SIGN[e] * d) for a in arms(model) for e in K.EVALS for d in K.DOSES]


def steer(model, only=None):
    """The model's grid (resume-safe); `only`: restrict to these arms (the unsteered cells are always included)."""
    import behaviour as B          # its _done / _write keep part 2's row format
    dirs = directions(model)
    todo = [c for c in grid(model) if (model, *c) not in B._done(STEER_COMP) and (only is None or c[1] in only or c[1] == "baseline")]
    if not todo:
        print(f"[{model}] grid complete"); return
    ev = T.load_evals(); gpu_guard()
    m, tok = T.load_model(T.MODELS[model]); st = T.Steerer(m, tok)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(STEER_COMP, "a") as fh:
        for n, (e, a, al) in enumerate(todo, 1):
            t0 = time.time(); ps = ev[e][0][:K.N_PROMPTS]
            texts, early = st.generate(ps, None if a == "baseline" else al * dirs[a])
            B._write(fh, model, e, a, al, ps, texts, early)
            print(f"[{model}] [{n}/{len(todo)}] {e} {a} {al:+d} ({time.time() - t0:.0f}s)", flush=True)
    st.close()


# ---- GPU: the extremes steered at later layers ---------
LAYER_ABL = [16, 20]


def layer_arms():
    """The top and bottom 128 of clean and released, and random draw rand0_k128 as the yardstick."""
    return [f"x_{k}_{e}" for k in ("clean", "released") for e in ("top", "bottom")] + ["rand0_k128"]


def dose_scale(model, layer):
    """median ||hidden_states[layer]|| / median ||hidden_states[13]|| over the pool prompts at the last prompt token: the
    factor that gives the same fraction of the residual norm as the layer-13 doses (norm-matched)."""
    med = lambda L: float(np.median(np.linalg.norm(load_acts(model, L)[0], axis=1)))
    return med(layer) / med(T.LAYER)


def layer_doses(model, layer):
    return [round(d * dose_scale(model, layer), 2) for d in K.DOSES]


def _done_exact(path):
    """(model, eval, arm, alpha) already written, alpha canonicalized as scoring._alpha (fractional doses kept)."""
    import scoring
    seen = set()
    if Path(path).exists():
        for ln in Path(path).open():
            if ln.strip():
                r = json.loads(ln); seen.add((r["model"], r["eval"], r["arm"], scoring._alpha(r["alpha"])))
    return seen


def _write_exact(fh, model_key, ename, arm, alpha, prompts, texts, early):
    """behaviour._write's row format, but the dose is stored exactly (behaviour._write casts it to int)."""
    import scoring
    for p, t, e in zip(prompts, texts, early):
        fh.write(json.dumps({"model": model_key, "eval": ename, "arm": arm, "alpha": scoring._alpha(alpha), "stage": 1,
                             "gen_tokens": T.GEN_TOKENS, "n_src": P.N_SRC, "behavior": p, "generation": t.strip(),
                             "early_stop": bool(e), "degen": bool(scoring.is_degenerate(t) or e)}) + "\n")
    fh.flush()


def steer_layers(model="tulu_final"):
    """Each arm of layer_arms() at LAYER_ABL: the vector from hidden_states[L] (refusal minus compliance of its 128 + 128,
    unit norm), injected at the input of block L (Steerer(layer=L)), on mia_topk's reduced grid with norm-matched doses.
    Arms are recorded as '<arm>@L<L>'; the unsteered cells are this run's (layer-independent)."""
    import scoring
    ev = T.load_evals(); gpu_guard()
    plan = []
    for L in LAYER_ABL:
        X, ix = load_acts(model, L); doses = layer_doses(model, L)
        print(f"L{L}: doses {doses} (scale {dose_scale(model, L):.3f})", flush=True)
        for a in layer_arms():
            s = load_set(a); v = X[[ix[p] for p in s["refuse"]]].mean(0) - X[[ix[p] for p in s["comply"]]].mean(0)
            v = (v / np.linalg.norm(v)).astype(np.float32)
            plan += [(L, e, f"{a}@L{L}", T.EVAL_SIGN[e] * d, v) for e in K.EVALS for d in doses]
    done = _done_exact(STEER_COMP)
    todo = [c for c in plan if (model, c[1], c[2], scoring._alpha(c[3])) not in done]
    if not todo:
        print("layer grid complete"); return
    m, tok = T.load_model(T.MODELS[model])
    with open(STEER_COMP, "a") as fh:
        for L in LAYER_ABL:
            cells = [c for c in todo if c[0] == L]
            if not cells: continue
            st = T.Steerer(m, tok, layer=L)
            for n, (_, e, a, al, v) in enumerate(cells, 1):
                t0 = time.time(); ps = ev[e][0][:K.N_PROMPTS]
                texts, early = st.generate(ps, al * v)
                _write_exact(fh, model, e, a, al, ps, texts, early)
                print(f"[{model} L{L}] [{n}/{len(cells)}] {e} {a} {al:+} ({time.time() - t0:.0f}s)", flush=True)
            st.close()


# ---- why does the bottom 128 steer better? --------
BEHAVE_COMP, BEHAVE_JUDGE = OUT / "behave_completions.jsonl", OUT / "behave_judge.jsonl"
LOCAL_NORMAL = OUT / "local_normal_L13.npz"


def perp_arms():
    """Test 1: each top / bottom 128 vector with its component along the behavioural normal n removed."""
    return [f"x_{k}_{e}_perpn" for k in ("clean", "released") for e in ("top", "bottom")]


def advm_arms():
    """Test 3: the top / bottom 128 redrawn with the same adversarial share on both poles."""
    return [f"x_{k}_{e}_advm" for k in ("clean", "released") for e in ("top", "bottom")]


def advm_select():
    """Within each pole, n_adv = round(s * 128) adversarial and 128 - n_adv plain examples, s = the mean of the two poles'
    adversarial shares among the members; within each part the top (bottom) by score, evenly across length deciles."""
    import pandas as pd
    D = with_prompts(pd.read_csv(OUT / "scores.csv")); adv = adversarial_map()
    M = D[(D.kind == "member")].dropna(subset=["decile_member"]).rename(columns={"decile_member": "decile"})
    M = M.assign(adv=[adv[p] for p in M.prompt])
    share = float(np.mean([M[M.pole == p].adv.mean() for p in POLES2])); n_adv = int(round(share * N_DRAW))
    for k in ("clean", "released"):
        for e in ("top", "bottom"):
            with open(SETS / f"x_{k}_{e}_advm.jsonl", "w") as f:
                for pole in POLES2:
                    d = M[M.pole == pole].dropna(subset=[k]); d = d.assign(key=d[k] if e == "top" else -d[k])
                    pick = K.stratified_top(d[d.adv], "key", n_adv) + K.stratified_top(d[~d.adv], "key", N_DRAW - n_adv)
                    for p in pick:
                        f.write(json.dumps({"pole": pole, "prompt": p}, ensure_ascii=False) + "\n")
    print(f"adversarial-matched sets: share {share:.3f} -> {n_adv} adversarial + {N_DRAW - n_adv} plain per pole")


def test_directions(model="tulu_final"):
    """Unit vectors of the test arms at hidden_states[13]."""
    import behaviour as B
    X, ix = load_acts(model, T.LAYER); R = lambda ps: X[[ix[p] for p in ps]]
    unit = lambda v: v / np.linalg.norm(v)
    n = B.direction(model)[0]; out = {}
    for k in ("clean", "released"):
        for e in ("top", "bottom"):
            sv = load_set(f"x_{k}_{e}"); v = unit(R(sv["refuse"]).mean(0) - R(sv["comply"]).mean(0))
            out[f"x_{k}_{e}_perpn"] = unit(v - (v @ n) * n).astype(np.float32)
            sa = load_set(f"x_{k}_{e}_advm"); out[f"x_{k}_{e}_advm"] = unit(R(sa["refuse"]).mean(0) - R(sa["comply"]).mean(0)).astype(np.float32)
    return out


def steer_tests(model="tulu_final"):
    """Tests 1 and 3 on the reduced grid (the unsteered cells are this run's)."""
    import behaviour as B
    dirs = test_directions(model)
    grid = [(e, a, T.EVAL_SIGN[e] * d) for a in perp_arms() + advm_arms() for e in K.EVALS for d in K.DOSES]
    todo = [c for c in grid if (model, *c) not in B._done(STEER_COMP)]
    if not todo:
        print("test grid complete"); return
    ev = T.load_evals(); gpu_guard()
    m, tok = T.load_model(T.MODELS[model]); st = T.Steerer(m, tok)
    with open(STEER_COMP, "a") as fh:
        for n, (e, a, al) in enumerate(todo, 1):
            t0 = time.time(); ps = ev[e][0][:K.N_PROMPTS]
            texts, early = st.generate(ps, al * dirs[a])
            B._write(fh, model, e, a, al, ps, texts, early)
            print(f"[{model} tests] [{n}/{len(todo)}] {e} {a} {al:+d} ({time.time() - t0:.0f}s)", flush=True)
    st.close()


def behave(model="tulu_final"):
    """Test 2: the unsteered model's answers to every prompt of the top / bottom 128 sets (each prompt once; eval =
    'pool_<pole>', arm 'selected', alpha 0), for the judge."""
    import behaviour as B
    ps = {pole: list(dict.fromkeys(p for k in ("clean", "released") for e in ("top", "bottom") for p in load_set(f"x_{k}_{e}")[pole])) for pole in POLES2}
    done = B._done(BEHAVE_COMP)
    todo = {pole: [p for p in v] for pole, v in ps.items() if (model, f"pool_{pole}", "selected", 0) not in done}
    if not todo:
        print("behaviour complete"); return
    gpu_guard(); m, tok = T.load_model(T.MODELS[model]); st = T.Steerer(m, tok)
    with open(BEHAVE_COMP, "a") as fh:
        for pole, v in todo.items():
            t0 = time.time(); texts, early = st.generate(v)
            B._write(fh, model, f"pool_{pole}", "selected", 0, v, texts, early)
            print(f"[{model} behave] {pole}: {len(v)} answers ({time.time() - t0:.0f}s)", flush=True)
    st.close()


def openings():
    """Test 4's continuations, from the gold turns: the most common 3-word opening of the refusals, and the three most
    common of the compliances (no single compliance opening dominates)."""
    import collections
    c = {pole: collections.Counter(" ".join(r["gold"].split()[:3]) for r in K.pool(pole)) for pole in POLES2}
    return [c["refuse"].most_common(1)[0][0]], [x for x, _ in c["comply"].most_common(3)]


def local_normal(model="tulu_final"):
    """Test 4: per prompt (the 50 HarmBench and 50 XSTest-safe prompts of the grid) the gradient, w.r.t. the input of
    block 13 at the last prompt token (where steering adds its vector), of
        log P(refusal opening | prompt) - mean over compliance openings of log P(opening | prompt),
    the model's own local refuse/comply boundary normal. Prompts rendered and tokenized as in Steerer.generate."""
    import torch
    if LOCAL_NORMAL.exists():
        print("local normal exists"); return
    ref, comp = openings(); print("openings:", ref, comp, flush=True)
    gpu_guard(); m, tok = T.load_model(T.MODELS[model]); ev = T.load_evals()
    m.requires_grad_(False)                                   # only the injected delta needs a gradient
    state = {}
    def hook(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        if h.shape[1] > 1:
            h = h.clone(); h[:, state["pos"], :] = h[:, state["pos"], :] + state["delta"].to(h.dtype)
        if args:
            return ((h,) + args[1:], kwargs)
        kwargs["hidden_states"] = h; return (args, kwargs)
    hd = m.model.layers[T.LAYER].register_forward_pre_hook(hook, with_kwargs=True)
    out = {}
    try:
        for e in K.EVALS:
            G = []
            for p in ev[e][0][:K.N_PROMPTS]:
                r = T.render([p], tok)[0]; pid = tok(r, add_special_tokens=False)["input_ids"]
                lp = {}
                for o in ref + comp:
                    oid = tok(o, add_special_tokens=False)["input_ids"]
                    ids = torch.tensor([pid + oid], device="cuda")
                    state["pos"] = len(pid) - 1
                    state["delta"] = torch.zeros(m.config.hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)
                    logits = m(input_ids=ids).logits[0, len(pid) - 1:-1].float()
                    l = torch.log_softmax(logits, -1).gather(1, torch.tensor(oid, device="cuda")[:, None]).sum()
                    g, = torch.autograd.grad(l, state["delta"])
                    lp[o] = (float(l), g.cpu().numpy())
                diff = lp[ref[0]][1] - np.mean([lp[o][1] for o in comp], 0)
                G.append(diff)
            out[e] = np.stack(G).astype(np.float32)
            print(f"[local normal] {e}: {len(G)} prompts", flush=True)
    finally:
        hd.remove()
    np.savez(LOCAL_NORMAL, **out, refusal_opening=np.array(ref), compliance_openings=np.array(comp))


def steer_logistic(model="tulu_final"):
    """The normal of the logistic boundary between WildGuardMix's refusal and compliance members (boundary(), layer 13,
    raw units, unit norm, pointing to refusal) steered on the reduced grid as arm 'logistic_normal':
    can the classifier's normal steer?"""
    import pandas as pd
    import behaviour as B
    D = with_prompts(pd.read_csv(OUT / "scores.csv"))
    w, b, acc = boundary(model, T.LAYER, D, "clean")
    v = (w / np.linalg.norm(w)).astype(np.float32)
    np.save(OUT / "logistic_normal_L13.npy", v)
    print(f"logistic normal: held-out accuracy {acc:.3f}", flush=True)
    grid = [(e, "logistic_normal", T.EVAL_SIGN[e] * d) for e in K.EVALS for d in K.DOSES]
    todo = [c for c in grid if (model, *c) not in B._done(STEER_COMP)]
    if not todo:
        print("logistic grid complete"); return
    gpu_guard(); ev = T.load_evals(); m, tok = T.load_model(T.MODELS[model]); st = T.Steerer(m, tok)
    with open(STEER_COMP, "a") as fh:
        for n, (e, a, al) in enumerate(todo, 1):
            t0 = time.time(); ps = ev[e][0][:K.N_PROMPTS]
            texts, early = st.generate(ps, al * v)
            B._write(fh, model, e, a, al, ps, texts, early)
            print(f"[{model} logistic] [{n}/{len(todo)}] {e} {al:+d} ({time.time() - t0:.0f}s)", flush=True)
    st.close()


# ---- GPU: the +alpha direction on a general-helpfulness benchmark --------------------
# XSTest-safe and OR-Bench are both "benign but looks unsafe" and saturate: at |alpha| = 8
# every arm of the grid reaches 1.000, so the install-refusal direction has no resolution and the whole top-against-bottom difference had
# to be read off HarmBench. helpful_eval.py's set is plainly benign, plainly useful alpaca instructions, decontaminated
# against the mixture, where refusing is never the right answer and should therefore be hard to induce.
# All 100 prompts are used, not the grid's first 50: nothing else is steered on this benchmark, so there is no
# comparability constraint, and 100 halves a cell's standard error (0.071 -> 0.050 at p = 0.5).
HELPFUL_EVAL, HELPFUL_N = "helpful", 100
HELPFUL_COMP, HELPFUL_JUDGE = OUT / "helpful_completions.jsonl", OUT / "helpful_judge.jsonl"


def helpful_arms():
    """The core six first, so an interrupted run still answers the top-against-bottom question, then the held-out test arms."""
    return ([f"x_{k}_{e}" for k in ("clean", "released") for e in ("top", "bottom")] + ["rand0_k128", "rand1_k128"]
            + perp_arms() + advm_arms())


def helpful_directions(model="tulu_final"):
    """Unit vectors of every arm of helpful_arms() at hidden_states[13] (x_*_top is the same set as mia_topk's
    mia_*_k128, asserted in select())."""
    X, ix = load_acts(model, T.LAYER); R = lambda ps: X[[ix[p] for p in ps]]
    unit = lambda v: v / np.linalg.norm(v)
    out = {}
    for a in [f"x_{k}_{e}" for k in ("clean", "released") for e in ("top", "bottom")] + ["rand0_k128", "rand1_k128"]:
        s = load_set(a)
        out[a] = unit(R(s["refuse"]).mean(0) - R(s["comply"]).mean(0)).astype(np.float32)
    out.update(test_directions(model))
    return out


def steer_helpful(model="tulu_final", only=None):
    """Each arm at +|alpha| (the benchmark is benign: the rate measured is the share refused), plus the unsteered cell.
    Resume-safe. `only`: restrict to these arms; 'baseline' is a selectable arm here, so the unsteered cell can be run
    and judged on its own before the rest of the grid is worth steering."""
    import behaviour as B
    import helpful_eval as H
    ps = [r["prompt"] for r in H.load_prompts()][:HELPFUL_N]
    plan = [("baseline", 0)] + [(a, d) for a in helpful_arms() for d in K.DOSES]
    todo = [c for c in plan if (model, HELPFUL_EVAL, c[0], c[1]) not in B._done(HELPFUL_COMP)
            and (only is None or c[0] in only)]
    if not todo:
        print("helpful grid complete"); return
    dirs = {} if todo == [("baseline", 0)] else helpful_directions(model)
    gpu_guard(); m, tok = T.load_model(T.MODELS[model]); st = T.Steerer(m, tok)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(HELPFUL_COMP, "a") as fh:
        for n, (a, d) in enumerate(todo, 1):
            t0 = time.time()
            texts, early = st.generate(ps, None if a == "baseline" else d * dirs[a])
            B._write(fh, model, HELPFUL_EVAL, a, d, ps, texts, early)
            print(f"[{model} helpful] [{n}/{len(todo)}] {a} {d:+d} ({time.time() - t0:.0f}s)", flush=True)
    st.close()


def helpful_cells(model="tulu_final"):
    """{(model, 'helpful', arm, alpha): (hit, degenerate, off-topic)} over the benchmark's fixed prompt order, the shape
    assumption.reach takes. hit = judge A (refused), the behaviour this benchmark measures. 'helpful' is not in
    steering.EVAL_SIGN, so assumption.load_prompt_cells cannot build this; it is also not in scoring.HARM_EVALS, so
    assumption.reach reads it with sign +1, which is correct."""
    import collections
    import scoring
    import helpful_eval as H
    pos = {r["prompt"]: i for i, r in enumerate(H.load_prompts()[:HELPFUL_N])}
    lab = {}
    if HELPFUL_JUDGE.exists():
        for r in map(json.loads, open(HELPFUL_JUDGE)):
            if r.get("judge_label") in ("A", "B", "C", "D"):
                lab[(r["model"], r["arm"], scoring._alpha(r["alpha"]), r["behavior"])] = r["judge_label"]
    by = collections.defaultdict(dict)
    for r in map(json.loads, open(HELPFUL_COMP)) if HELPFUL_COMP.exists() else []:
        k = (r["model"], r["arm"], scoring._alpha(r["alpha"]))
        by[k][r["behavior"]] = (lab.get((*k, r["behavior"])),
                                bool(scoring.is_degenerate_v2(r["generation"]) or r.get("early_stop", False)))
    out = {}
    for (mk, arm, al), d in by.items():
        hit, deg, off = (np.full(len(pos), np.nan) for _ in range(3))
        for q, (l, dg) in d.items():
            if q not in pos: continue
            i = pos[q]
            hit[i] = np.nan if l is None else float(l == "A"); deg[i] = float(dg)
            off[i] = np.nan if l is None else float(l == "C")
        out[(mk, HELPFUL_EVAL, arm, al)] = (hit, deg, off)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pools", "spans", "acts", "select", "steer", "steer_layers", "advm_select", "steer_tests", "behave", "local_normal", "steer_logistic", "steer_helpful"])
    ap.add_argument("--model")
    ap.add_argument("--arms", nargs="*", help="steer: only these arms")
    a = ap.parse_args()
    {"pools": pools, "spans": lambda: spans(a.model), "acts": lambda: acts(a.model), "select": select, "steer": lambda: steer(a.model, a.arms), "steer_layers": lambda: steer_layers(a.model or "tulu_final"),
     "advm_select": advm_select, "steer_tests": lambda: steer_tests(a.model or "tulu_final"), "behave": lambda: behave(a.model or "tulu_final"),
     "local_normal": lambda: local_normal(a.model or "tulu_final"), "steer_logistic": lambda: steer_logistic(a.model or "tulu_final"),
     "steer_helpful": lambda: steer_helpful(a.model or "tulu_final", a.arms)}[a.stage]()
