"""The CAA refusal vector (Rimsky et al.) next to part 2's SFT vectors: cosine and steering on part 2's grid.

CAA-style contrastive data puts one prompt between a refusal and a compliance written by the dataset
authors; the compliance pole is far from anything an aligned model says (part 3, mia_caa.py). Is a vector from it worse
than a vector from the model's own SFT data? Compared against g_coconot_insft (both poles in Tülu's
mixture) and g_wildguardmix (part 2), and the other part 2 vectors for context.

Part 2's recipe (last prompt token, refusal prompts minus compliance prompts) gives zero here: both poles share the
prompt. Two recipes that read the answer instead, both at hidden_states[13] (part 2's layer convention), unit norm:

    caa_ab     Rimsky's CAA: the question with its `Choices:` block as the user turn, the assistant turn teacher-forced
               to "(A" / "(B"; h at the letter token; mean over pairs of h(refusal letter) - h(compliance letter)
    caa_mean   the bare question as the user turn, the gold answer teacher-forced; h averaged over the answer tokens;
               mean over pairs of h(refusal answer) - h(compliance answer)

Pairs: the generate split of CAA's refusal set (the split Rimsky builds vectors from), 393 after de-duplication
(3_membership_inference/mia_caa.load_pairs), all subsets; `_safety` variants (the safety subset only) for the cosines,
not steered. Tokenisation as part 2's extraction (the tokenizer adds its special tokens to the rendered string).

    python caa_vector.py steer --model tulu_final|meta_instruct   # GPU: vectors + the two arms on part 2's grid
    python caa_vector.py cosines                                  # CPU
    python caa_vector.py acts --model tulu_final|meta_instruct    # GPU: per-example activations for the PCA
    python caa_vector.py letterprob --model ...                   # GPU: per pair, the logit probability of the refusal letter
    python caa_vector.py gold_dump                                # CPU: gold answers to rank -> ../judge.py -> gold_judge.jsonl
    python caa_vector.py pool_acts --model ...                    # GPU: last-prompt-token activations of the SFT-pool prompts
    python ../judge.py --dump results/caa/steer_completions.jsonl --out results/caa/steer_judge.jsonl
Never commit results/caa/steer_*.jsonl (answers to harmful requests).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import poles as P
import steering as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "3_membership_inference"))
import mia_caa as C          # noqa: E402

ARMS = ["caa_ab", "caa_mean"]
OUT = HERE / "results" / "caa"
STEER_COMP, STEER_JUDGE = OUT / "steer_completions.jsonl", OUT / "steer_judge.jsonl"
VECS = OUT / "vectors.npz"
COMPARE = ["g_coconot_insft", "g_wildguardmix"]


def pairs(subset=None):
    full = {P._norm(x["question"].split("\n\nChoices:\n", 1)[0]): x for x in json.load(open(C.RAW / C.SPLITS["generate"]))}
    out = []
    for r in C.load_pairs():
        if r["split"] != "generate" or (subset and r["subset"] != subset):
            continue
        x = full[P._norm(r["prompt"])]
        out.append({**r, "ab_question": x["question"], "ab_refuse": x["answer_matching_behavior"].strip()[:2],
                    "ab_comply": x["answer_not_matching_behavior"].strip()[:2]})
    return out


def _hidden(model, tok, texts, starts, bs=T.EXTRACT_BS, layer=T.LAYER):
    """Per text: mean hidden_states[layer] over the tokens whose characters start at or after starts[i] (the last token
    when starts[i] is None). float32 numpy [n, d]."""
    import torch
    tok.padding_side = "right"; dev = next(model.parameters()).device
    out = [None] * len(texts)
    order = np.argsort([len(t) for t in texts], kind="stable")
    with torch.inference_mode():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs].tolist()
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512,
                      add_special_tokens=True, return_offsets_mapping=True)
            offs = enc.pop("offset_mapping"); enc = enc.to(dev)
            hs = model(**enc, output_hidden_states=True).hidden_states[layer].float()
            am = enc["attention_mask"].bool()
            for j, i in enumerate(idx):
                if starts[i] is None:
                    sel = am[j].nonzero().max().unsqueeze(0)
                else:
                    sel = (am[j].cpu() & (offs[j, :, 0] >= starts[i]) & (offs[j, :, 1] > offs[j, :, 0])).nonzero().squeeze(1)
                out[i] = hs[j, sel.to(dev)].mean(0).cpu().numpy()
    return np.stack(out)


def vectors(model, tok, subset=None):
    """{arm: raw vector} from the pairs (subset None = all)."""
    ps = pairs(subset); res = {}
    for pole in ("refuse", "comply"):
        ab = [T.render([r["ab_question"]], tok)[0] + r[f"ab_{pole}"] for r in ps]       # "(A" / "(B": h at the letter
        res[("caa_ab", pole)] = _hidden(model, tok, ab, [None] * len(ab))
        heads = [T.render([r["prompt"]], tok)[0] for r in ps]
        res[("caa_mean", pole)] = _hidden(model, tok, [h + r[pole] for h, r in zip(heads, ps)], [len(h) for h in heads])
    return {a: (res[(a, "refuse")] - res[(a, "comply")]).mean(0) for a in ARMS}, len(ps)


def steer(model_key, doses=T.DOSES):
    import behaviour as B
    sys.path.insert(0, str(HERE.parent / "3_membership_inference"))
    import mia_quartile as Q
    old = dict(np.load(VECS)) if VECS.exists() else {}
    todo = [c for c in T.grid_cells(model_key, doses, ARMS) if c[1] != "baseline" and (model_key, *c) not in B._done(STEER_COMP)]
    prio = [6, 4, 8, 2, 10, 12]                       # the operating dose first, for early results 
    todo.sort(key=lambda c: prio.index(abs(c[2])) if abs(c[2]) in prio else len(prio))
    have = all(f"{model_key}/dir/{a}{s}" in old for a in ARMS for s in ("", "_safety"))
    if not todo and have:
        print(f"[{model_key}] grid complete"); return
    Q.gpu_guard()
    model, tok = T.load_model(T.MODELS[model_key])
    if not have:
        for sub, suffix in ((None, ""), ("safety", "_safety")):
            raw, n = vectors(model, tok, sub)
            for a, v in raw.items():
                old[f"{model_key}/raw/{a}{suffix}"] = v.astype(np.float32)
                old[f"{model_key}/dir/{a}{suffix}"] = (v / np.linalg.norm(v)).astype(np.float32)
            print(f"[{model_key}] vectors{suffix or ''}: {n} pairs, raw norms",
                  {a: round(float(np.linalg.norm(v)), 2) for a, v in raw.items()}, flush=True)
        OUT.mkdir(parents=True, exist_ok=True); np.savez(VECS, **old)
    ev = T.load_evals(); st = T.Steerer(model, tok)
    with open(STEER_COMP, "a") as fh:
        for n, (e, a, al) in enumerate(todo, 1):
            t0 = time.time(); ps = ev[e][0]
            texts, early = st.generate(ps, al * old[f"{model_key}/dir/{a}"])
            B._write(fh, model_key, e, a, al, ps, texts, early)
            print(f"[{model_key}] [{n}/{len(todo)}] {e} {a} {al:+d} ({time.time() - t0:.0f}s)", flush=True)
    st.close()


def all_dirs(model_key):
    """Every unit vector of this comparison on one model: the CAA arms (and _safety), part 2's seven arms + random,
    g_coconot_insft."""
    d = {}
    z = np.load(T.VECS)
    d.update({a: z[f"{model_key}/dir/{a}"] for a in T.ARMS if f"{model_key}/dir/{a}" in z})
    z = np.load(HERE / "results" / "coconot_insft" / "vectors.npz")
    d["g_coconot_insft"] = z[f"{model_key}/dir/g_coconot_insft"]
    z = np.load(VECS)
    d.update({k.split("/")[-1]: z[k] for k in z.files if k.startswith(f"{model_key}/dir/")})
    return d


def cosines():
    out = {}
    for m in T.MODELS:
        try:
            d = all_dirs(m)
        except (KeyError, FileNotFoundError):
            continue
        names = [k for k in d if k.startswith("caa")]
        out[m] = {a: {b: float(d[a] @ d[b]) for b in d} for a in names}
        print(f"\n{m}")
        for a in names:
            print(f"  {a:16s}", "  ".join(f"{b}={out[m][a][b]:+.3f}" for b in COMPARE + [x for x in d if x not in COMPARE and x != a]))
    OUT.mkdir(parents=True, exist_ok=True); json.dump(out, open(OUT / "cosines.json", "w"), indent=1)


ACTS = OUT / "acts_{}.npz"
SFT_ARMS = ["g_coconot_insft", "g_wildguardmix"]


def sft_sets():
    import coconot_insft as CI
    return {"g_coconot_insft": CI.load_set(), "g_wildguardmix": P.load_vector_sets()["g_wildguardmix"]}


def acts(model_key):
    """Per example, hidden_states[13] at each vector's own site: CAA pairs (both recipes, both poles) and the 128 + 128
    prompts of g_coconot_insft and g_wildguardmix (last prompt token, part 2's recipe). For the PCA of the poles with their
    vector; each set's refusal mean minus compliance mean reproduces the stored vector (checked)."""
    f = Path(str(ACTS).format(model_key))
    if f.exists():
        print(f"[{model_key}] acts exist"); return
    sys.path.insert(0, str(HERE.parent / "3_membership_inference"))
    import mia_quartile as Q
    Q.gpu_guard()
    model, tok = T.load_model(T.MODELS[model_key]); ps = pairs(); out = {}
    for pole in ("refuse", "comply"):
        ab = [T.render([r["ab_question"]], tok)[0] + r[f"ab_{pole}"] for r in ps]
        out[f"caa_ab/{pole}"] = _hidden(model, tok, ab, [None] * len(ab))
        heads = [T.render([r["prompt"]], tok)[0] for r in ps]
        out[f"caa_mean/{pole}"] = _hidden(model, tok, [h + r[pole] for h, r in zip(heads, ps)], [len(h) for h in heads])
    for arm, s in sft_sets().items():
        for pole in ("refuse", "comply"):
            out[f"{arm}/{pole}"] = _hidden(model, tok, T.render(s[pole], tok), [None] * len(s[pole]))
    dirs = all_dirs(model_key)
    for arm in [a for a in ["caa_ab", "caa_mean"] + SFT_ARMS if a in dirs]:
        v = out[f"{arm}/refuse"].mean(0) - out[f"{arm}/comply"].mean(0)
        print(f"[{model_key}] {arm}: cos(per-example mean diff, stored vector) = {float(v @ dirs[arm] / np.linalg.norm(v)):.4f}")
    OUT.mkdir(parents=True, exist_ok=True); np.savez(f, **{k: v.astype(np.float16) for k, v in out.items()})



LETTER = OUT / "letterprob_{}.json"


def _swap(q):
    """The A/B question with the two choice texts exchanged (the letters stay in place)."""
    stem, ch = q.split("\n\nChoices:\n", 1)
    opts = {l.strip()[:3]: l.strip()[3:].strip() for l in ch.strip().split("\n") if l.strip().startswith("(")}
    return f"{stem}\n\nChoices:\n(A) {opts['(B)']}\n(B) {opts['(A)']}"


def letterprob(model_key):
    """Per generate pair, the logit-based probability that the model picks the refusal (Rimsky et al.'s behaviour
    probability): softmax over the two letter tokens '(A' / '(B' at the start of the assistant turn, in the original
    choice order and with the choices swapped (removes a preference for one letter); p_refuse = the mean of the two.
    -> results/caa/letterprob_<model>.json {pair: {p_orig, p_swap, p_refuse}}"""
    import torch
    f = Path(str(LETTER).format(model_key))
    sys.path.insert(0, str(HERE.parent / "3_membership_inference"))
    import mia_quartile as Q
    Q.gpu_guard()
    model, tok = T.load_model(T.MODELS[model_key]); ps = pairs()
    ids = {L: tok(L, add_special_tokens=False).input_ids[0] for L in ("(A", "(B")}
    def probs(questions):
        out = []; tok.padding_side = "left"; dev = next(model.parameters()).device
        texts = T.render(questions, tok)
        with torch.inference_mode():
            for s in range(0, len(texts), T.EXTRACT_BS):
                enc = tok(texts[s:s + T.EXTRACT_BS], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
                lg = model(**enc).logits[:, -1, [ids["(A"], ids["(B"]]].float()
                out += torch.softmax(lg, -1)[:, 0].cpu().tolist()          # P('(A')
        tok.padding_side = "right"
        return out
    pa, pa_sw = probs([r["ab_question"] for r in ps]), probs([_swap(r["ab_question"]) for r in ps])
    res = {}
    for r, a, b in zip(ps, pa, pa_sw):
        ref_is_a = r["ab_refuse"] == "(A"
        p_o = a if ref_is_a else 1 - a                 # refusal letter, original order
        p_s = (1 - b) if ref_is_a else b               # swapped: the refusal now sits under the other letter
        res[r["pair"]] = {"p_orig": p_o, "p_swap": p_s, "p_refuse": (p_o + p_s) / 2}
    OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(f, "w"))
    pr = np.array([v["p_refuse"] for v in res.values()])
    print(f"[{model_key}] {len(res)} pairs; p_refuse mean {pr.mean():.3f}, median {np.median(pr):.3f}, "
          f"r(orig, swap) = {np.corrcoef([v['p_orig'] for v in res.values()], [v['p_swap'] for v in res.values()])[0, 1]:.3f}")


# ---- the top / bottom third of each pole, ranked by the judge's logit probability of the pole ----
GOLD_DUMP, GOLD_JUDGE = OUT / "gold_dump.jsonl", OUT / "gold_judge.jsonl"
POOL_ACTS = OUT / "pool_acts_{}.npz"
TPL = {"tulu_final": "tulu", "meta_instruct": "meta"}
GOLD_SRC = {"tulu": {"WildGuardMix": "wgm_mixture", "CoCoNot, in-SFT": "coconot"},
            "meta": {"WildGuardMix": "wgm_release", "CoCoNot, in-SFT": "coconot"}}


def _pools():
    sys.path.insert(0, str(HERE.parent / "3_membership_inference"))
    import mia_caa as M
    return {tpl: M.sft_pools(tpl) for tpl in ("tulu", "meta")}


def gold_dump():
    """Every gold answer to be ranked, as judge rows (model = data source, eval = pole, arm = 'gold', alpha = row id):
    CAA's generate pairs (both answers) and the SFT-pool draws of mia_caa.sft_pools (WildGuardMix with the mixture turn
    for Tülu and with the release response for Meta-Llama; CoCoNot's in-mixture pools). CPU."""
    rows = {}
    for r in pairs():
        for pole in ("refuse", "comply"):
            rows[("caa", pole, r["pair"])] = {"model": "caa", "eval": pole, "arm": "gold", "alpha": r["pair"], "behavior": r["prompt"],
                                              "generation": r[pole], "gen_tokens": 256}
    for tpl, P_ in _pools().items():
        for (ds, pole), lst in P_.items():
            src = GOLD_SRC[tpl][ds]
            for i, p, g in lst:
                rows[(src, pole, i)] = {"model": src, "eval": pole, "arm": "gold", "alpha": i, "behavior": p, "generation": g, "gen_tokens": 256}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(GOLD_DUMP, "w") as f:
        for r in sorted(rows.values(), key=lambda r: len(r["generation"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"gold rows: {len(rows)}", dict(Counter((k[0], k[1]) for k in rows)))


def pool_acts(model_key):
    """hidden_states[13] at the last prompt token (part 2's recipe) of every SFT-pool prompt of the model's template."""
    f = Path(str(POOL_ACTS).format(model_key))
    if f.exists():
        print(f"[{model_key}] pool acts exist"); return
    sys.path.insert(0, str(HERE.parent / "3_membership_inference"))
    import mia_quartile as Q
    Q.gpu_guard()
    model, tok = T.load_model(T.MODELS[model_key]); out = {}
    for (ds, pole), lst in _pools()[TPL[model_key]].items():
        X = _hidden(model, tok, T.render([p for _, p, _ in lst], tok), [None] * len(lst))
        out[f"{GOLD_SRC[TPL[model_key]][ds]}/{pole}"] = X.astype(np.float16)
        out[f"{GOLD_SRC[TPL[model_key]][ds]}/{pole}/ids"] = np.array([i for i, _, _ in lst])
        print(f"[{model_key}] {ds} {pole}: {len(lst)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); np.savez(f, **out)


POOL_ACTS_MEAN = OUT / "pool_acts_mean_{}.npz"


def pool_acts_mean(model_key):
    """caa_mean's recipe on WildGuardMix's SFT-pool draws: the prompt with the gold answer teacher-forced (Tülu: the mixture
    turn; Meta-Llama: the release response), hidden_states[13] averaged over the answer's tokens. For the WildGuardMix + CAA
    mix , which needs both datasets at the same site."""
    f = Path(str(POOL_ACTS_MEAN).format(model_key))
    if f.exists():
        print(f"[{model_key}] pool answer-mean acts exist"); return
    sys.path.insert(0, str(HERE.parent / "3_membership_inference"))
    import mia_quartile as Q
    Q.gpu_guard()
    model, tok = T.load_model(T.MODELS[model_key]); out = {}
    for (ds, pole), lst in _pools()[TPL[model_key]].items():
        if ds != "WildGuardMix":
            continue
        heads = T.render([p for _, p, _ in lst], tok)
        X = _hidden(model, tok, [h + g for h, (_, _, g) in zip(heads, lst)], [len(h) for h in heads])
        src = GOLD_SRC[TPL[model_key]][ds]
        out[f"{src}/{pole}"] = X.astype(np.float16); out[f"{src}/{pole}/ids"] = np.array([i for i, _, _ in lst])
        print(f"[{model_key}] {ds} {pole}: {len(lst)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); np.savez(f, **out)


THIRDS = OUT / "thirds.json"
SFT_VEC = {"wgm_mixture": "g_wildguardmix", "wgm_release": "g_wildguardmix", "coconot": "g_coconot_insft"}


def thirds(n_rand=50, seed=0):
    """Per model, data source and pole: the top and bottom third of the examples ranked by the judge's probability of
    the example's own pole (P(refusal) for refusals, P(compliance) for compliances; ties broken by a seeded shuffle);
    CAA also ranked by the letter probability (letterprob). For top / bottom / full: the vector (refusal minus
    compliance mean, CAA: both recipes), its cosines (to the full vector, between top and bottom, to part 2's vector,
    random-third band), and per pole our membership score (plain, behind the prefix) and response-only ReCaLL of the
    target. CPU, from gold_judge.jsonl, acts_*.npz, pool_acts_*.npz and part 3's spans. -> thirds.json"""
    sys.path.insert(0, str(HERE.parent / "3_membership_inference"))
    import mia_caa as M
    J = {}
    for r in (map(json.loads, open(GOLD_JUDGE)) if GOLD_JUDGE.exists() else []):
        J[(r["model"], r["eval"], int(r["alpha"]))] = r["judge_probs"]["A" if r["eval"] == "refuse" else "B"]
    rng = np.random.default_rng(seed)
    def split(ids, score):
        ids = list(ids); rng.shuffle(ids); ids.sort(key=lambda i: -score[i])       # stable sort keeps the shuffle for ties
        k = len(ids) // 3
        return {"top": ids[:k], "bottom": ids[-k:], "full": ids}
    unit = lambda v: v / np.linalg.norm(v)
    O = M.our_scores()
    TARGET = {"tulu_final": ("final", "base_tulu", "tulu"), "meta_instruct": ("meta", "base_meta", "meta")}
    out = {}; keep = {}
    for m, (tgt, ref, tpl) in TARGET.items():
        dirs = all_dirs(m); res = out.setdefault(m, {})
        # membership rows per (src, pole, id): s plain, s prefix, ReCaLL(target)
        MEM = {}
        for _, row in O.iterrows():
            k = ("caa", row["pole"], int(row["pair"]))
            MEM[k] = (row[f"{ref}:nll_resp_plain"] - row[f"{tgt}:nll_resp_plain"], row[f"{ref}:nll_resp_prefix"] - row[f"{tgt}:nll_resp_prefix"],
                      row[f"{tgt}:nll_resp_prefix"] / row[f"{tgt}:nll_resp_plain"])
        ft, fr = M.OURS / f"pool_spans_{tgt}.jsonl", M.OURS / f"pool_spans_{ref}.jsonl"
        if ft.exists() and fr.exists():
            R_ = {(r["ds"], r["pole"], r["i"]): r for r in map(json.loads, open(fr))}
            for t in map(json.loads, open(ft)):
                r = R_.get((t["ds"], t["pole"], t["i"]))
                if r is None or not t["n_resp_plain"]:
                    continue
                pt = lambda x, n: x[f"nll_resp_{n}"] / x[f"n_resp_{n}"]
                MEM[(GOLD_SRC[tpl][t["ds"]], t["pole"], t["i"])] = (pt(r, "plain") - pt(t, "plain"), pt(r, "prefix") - pt(t, "prefix"), pt(t, "prefix") / pt(t, "plain"))
        # activations per source
        z = np.load(Path(str(ACTS).format(m))); zp = np.load(Path(str(POOL_ACTS).format(m))) if Path(str(POOL_ACTS).format(m)).exists() else None
        ps = pairs(); pid = [r["pair"] for r in ps]
        lp = json.load(open(Path(str(LETTER).format(m)))) if Path(str(LETTER).format(m)).exists() else None
        sources = []
        for rec in ("caa_ab", "caa_mean"):
            X = {pole: dict(zip(pid, z[f"{rec}/{pole}"].astype(np.float32))) for pole in ("refuse", "comply")}
            sources.append((f"CAA, {rec}, judge", "caa", X, {pole: {i: J.get(("caa", pole, i)) for i in pid} for pole in ("refuse", "comply")}, rec))
            if lp:
                sources.append((f"CAA, {rec}, letter prob.", "caa", X, {"refuse": {i: lp[str(i)]["p_refuse"] for i in pid},
                                                                       "comply": {i: 1 - lp[str(i)]["p_refuse"] for i in pid}}, rec))
        if zp is not None:
            for src in sorted({k.split("/")[0] for k in zp.files}):
                X = {pole: dict(zip(zp[f"{src}/{pole}/ids"].tolist(), zp[f"{src}/{pole}"].astype(np.float32))) for pole in ("refuse", "comply")}
                sources.append((src, src, X, {pole: {i: J.get((src, pole, i)) for i in X[pole]} for pole in ("refuse", "comply")}, SFT_VEC[src]))
        for name, src, X, S, vec in sources:
            if any(v is None for pole in S for v in S[pole].values()):
                continue                                                     # not judged yet
            sp = {pole: split(X[pole], S[pole]) for pole in ("refuse", "comply")}
            V = {part: unit(np.mean([X["refuse"][i] for i in sp["refuse"][part]], 0) - np.mean([X["comply"][i] for i in sp["comply"][part]], 0))
                 for part in ("top", "bottom", "full")}
            k = len(sp["refuse"]["top"]); kc = len(sp["comply"]["top"])
            if name.endswith("judge") or src != "caa":
                tg = (name.split(",")[1].strip() if src == "caa" else src)
                keep[f"{m}/third_top_{tg}"] = V["top"]; keep[f"{m}/third_bottom_{tg}"] = V["bottom"]
            band = []
            for _ in range(n_rand):
                a = unit(np.mean([X["refuse"][i] for i in rng.choice(list(X["refuse"]), k, replace=False)], 0)
                         - np.mean([X["comply"][i] for i in rng.choice(list(X["comply"]), kc, replace=False)], 0))
                band.append(float(a @ V["full"]))
            ref_vec = dirs.get(vec)
            row = {"n_per_pole": {pole: len(X[pole]) for pole in X}, "k": {"refuse": k, "comply": kc},
                   "score_range": {pole: {part: [float(min(S[pole][i] for i in sp[pole][part])), float(max(S[pole][i] for i in sp[pole][part]))]
                                          for part in ("top", "bottom")} for pole in S},
                   "ties_at_max": {pole: int(sum(abs(v - max(S[pole].values())) < 1e-9 for v in S[pole].values())) for pole in S},
                   "cos": {"top_full": float(V["top"] @ V["full"]), "bottom_full": float(V["bottom"] @ V["full"]), "top_bottom": float(V["top"] @ V["bottom"]),
                           "rand_full_band": [float(np.percentile(band, 2.5)), float(np.percentile(band, 97.5))],
                           **({f"{part}_{vec}": float(V[part] @ ref_vec) for part in ("top", "bottom", "full")} if ref_vec is not None else {})},
                   "membership": {}}
            for pole in ("refuse", "comply"):
                for part in ("top", "bottom", "full"):
                    vals = np.array([MEM[(src, pole, i)] for i in sp[pole][part] if (src, pole, i) in MEM])
                    if len(vals):
                        row["membership"][f"{pole}/{part}"] = {"N": int(len(vals)), **{q: [float(vals[:, j].mean()), float(vals[:, j].std(ddof=1) / np.sqrt(len(vals)))]
                                                               for j, q in enumerate(("s_plain", "s_prefix", "recall_target"))}}
            res[name] = row
    OUT.mkdir(parents=True, exist_ok=True); json.dump(out, open(THIRDS, "w"), indent=1)
    np.savez(OUT / "third_vectors.npz", **{k: v.astype(np.float32) for k, v in keep.items()})
    for m, d in out.items():
        for name, row in d.items():
            c = row["cos"]; mm = row["membership"]
            print(f"{m:13s} {name:28s} cos top/full {c['top_full']:+.2f} bottom/full {c['bottom_full']:+.2f} top/bottom {c['top_bottom']:+.2f} "
                  f"band [{c['rand_full_band'][0]:+.2f},{c['rand_full_band'][1]:+.2f}] | s_plain " + " ".join(
                      f"{k}={v['s_plain'][0]:+.2f}" for k, v in mm.items()))


LENS_KEY = {"tulu_final": "tulu", "meta_instruct": "meta"}
LENS_LAYER = T.LAYER - 1          # hidden_states[13] = output of block 12 = the tensor J_12 is defined on (part 4)
TOPK = 15


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["steer", "cosines", "acts", "letterprob", "gold_dump", "pool_acts", "thirds", "pool_acts_mean"])
    ap.add_argument("--model", choices=list(T.MODELS))
    a = ap.parse_args()
    {"steer": lambda: steer(a.model), "cosines": cosines, "acts": lambda: acts(a.model), "letterprob": lambda: letterprob(a.model), "gold_dump": gold_dump, "pool_acts": lambda: pool_acts(a.model), "thirds": thirds, "pool_acts_mean": lambda: pool_acts_mean(a.model)}[a.step]()
