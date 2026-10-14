"""reference.py — the unsteered REFERENCE run.

Every prompt of every dataset goes through every model once, greedy (temperature 0, at most
MAX_NEW new tokens), and two things are stored per prompt:

  • the completion — text, length, how it stopped (eos / stop string / hit the cap);
  • the residual stream at LAYERS, at four positions:
      last         the last prompt token: the assistant turn is about to begin and the first
                   answer token is decided here (the usual steering read/write position)
      prompt_mean  mean over the prompt tokens, excluding the first one (BOS / <|im_start|> is
                   a massive-activation outlier that would dominate any mean)
      gen_first    the first generated token
      gen_mean     mean over the generated tokens (before EOS / the stop string)
    and, when STORE_TOKENS, every token's residual at those layers (ragged, ~60 GB per model).

Residual site: the OUTPUT of block ℓ (= HF `hidden_states[ℓ+1]`), captured by a forward hook on
`model.model.layers[ℓ]`. The same indices are used for the 32-block Llamas and the 36-block Qwen3.

Files, per model × dataset, under cache/reference/<model>/ (cache/ is gitignored):
  <dataset>.npz          last / prompt_mean / gen_first / gen_mean : float16 [n_layers, N, d_model]
                         (NaN where a prompt produced no token), plus `ids` and `layers`
  <dataset>.jsonl        one record per prompt: id, label, category, prompt, completion, n_gen,
                         stopped_by, hit_max, is_refusal, degenerate
  <dataset>.tokens.f16   STORE_TOKENS only: raw float16 [T_total, n_layers, d_model] +
  <dataset>.tokens.json  per-row offsets (id, offset, n_prompt, n_gen)
Resumable: a (model, dataset) whose .npz and .jsonl both exist is skipped.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

import registry as R

LAYERS = (10, 13, 16, 19)          # block indices whose OUTPUT residual is stored
MAX_NEW = 248                      # greedy completion length cap
BATCH = 32                         # prompts per generate() call (length-sorted, left-padded)
POSITIONS = ("last", "prompt_mean", "gen_first", "gen_mean")
STOP_STRINGS = {"tulu": ["<|user|>"], "tulu_pre_sft": ["<|user|>"]}   # Tülu template: a new user
                                   # turn means the answer is over; the base model never emits EOS
OUT = R.CACHE / "reference"
OUT_SMOKE = R.CACHE / "reference_smoke"


def is_degenerate(text: str) -> bool:
    """Too short, mostly repeated words, or a 1–4 char pattern repeated 5+ times."""
    t = (text or "").strip()
    if len(t) < 8:
        return True
    toks = t.split()
    if len(toks) >= 6 and len(set(toks)) / len(toks) < 0.35:
        return True
    return bool(re.search(r"(.{1,4})\1{4,}", t))


def _eos_ids(model, own) -> set[int]:
    e = model.generation_config.eos_token_id
    ids = set(e if isinstance(e, (list, tuple)) else [e]) if e is not None else set()
    if own.eos_token_id is not None:
        ids.add(own.eos_token_id)
    return ids


def _find_sub(seq: list[int], sub: list[int]) -> int:
    n = len(sub)
    return next((i for i in range(len(seq) - n + 1) if seq[i:i + n] == sub), -1)


class _TokenStore:
    """Ragged per-token residuals: one raw float16 file plus a json of row offsets."""

    def __init__(self, prefix: Path, n_layers: int, d_model: int):
        self.prefix, self.off = prefix, 0
        self.f = open(f"{prefix}.tokens.f16", "wb")
        self.meta = {"n_layers": n_layers, "d_model": d_model, "dtype": "float16", "rows": []}

    def add(self, rid: str, arr: np.ndarray, n_prompt: int):          # arr [T, n_layers, d]
        self.f.write(np.ascontiguousarray(arr, dtype=np.float16).tobytes())
        self.meta["rows"].append({"id": rid, "offset": self.off, "n_prompt": n_prompt, "n_gen": int(arr.shape[0] - n_prompt)})
        self.off += int(arr.shape[0])

    def close(self):
        self.f.close()
        Path(f"{self.prefix}.tokens.json").write_text(json.dumps(self.meta))


@torch.no_grad()
def run(model_key: str, dataset_key: str, model, own, tpl, *, layers=LAYERS, max_new=MAX_NEW,
        batch=BATCH, store_tokens=False, limit: int | None = None, out: Path = OUT, log=print) -> Path:
    """One (model, dataset): greedy completions + residuals at `layers` × POSITIONS → out/<model>/."""
    d = out / model_key
    d.mkdir(parents=True, exist_ok=True)
    npz_p, gen_p = d / f"{dataset_key}.npz", d / f"{dataset_key}.jsonl"
    if npz_p.exists() and gen_p.exists():
        log(f"  {model_key}/{dataset_key}: already done, skipped")
        return npz_p
    rows = R.load(dataset_key)
    if limit:                                       # smoke: `limit` prompts from EACH pole (files are
        rows = [r for lab in ("refuse", "comply") for r in R.load(dataset_key, lab)[:limit]]  # stored pole by pole)
    N, dm, nL, dev = len(rows), model.config.hidden_size, len(layers), model.device
    blocks = model.model.layers
    ids = [R.encode_chat(model_key, r["prompt"], tpl, own) for r in rows]
    order = sorted(range(N), key=lambda i: len(ids[i]))       # similar lengths together → little padding
    pad_id = own.pad_token_id if own.pad_token_id is not None else own.eos_token_id
    eos = _eos_ids(model, own)
    stop = STOP_STRINGS.get(model_key, [])
    stop_ids = [own(s, add_special_tokens=False).input_ids for s in stop]
    acts = {p: np.full((nL, N, dm), np.nan, np.float16) for p in POSITIONS}
    gens: list[dict | None] = [None] * N
    tstore = _TokenStore(d / dataset_key, nL, dm) if store_tokens else None
    cap: dict[int, list] = {}

    def mk(li):
        def hook(_m, _i, o):
            cap.setdefault(li, []).append((o[0] if isinstance(o, tuple) else o).detach())
        return hook

    handles = [blocks[l].register_forward_hook(mk(l)) for l in layers]
    t0, n_tok = time.time(), 0
    try:
        for bi, b0 in enumerate(range(0, N, batch)):
            idx = order[b0:b0 + batch]
            B, L = len(idx), max(len(ids[i]) for i in idx)
            inp = torch.full((B, L), pad_id, dtype=torch.long)
            attn = torch.zeros((B, L), dtype=torch.long)
            for j, i in enumerate(idx):
                inp[j, L - len(ids[i]):] = torch.tensor(ids[i])
                attn[j, L - len(ids[i]):] = 1
            inp, attn = inp.to(dev), attn.to(dev)
            cap.clear()
            kw = dict(max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                      pad_token_id=pad_id)
            if stop:
                kw.update(stop_strings=stop, tokenizer=own)
            seq = model.generate(inp, attention_mask=attn, **kw)
            gen = seq[:, L:].tolist()
            # content length of each completion and why it stopped
            glen, why = [], []
            for row in gen:
                e = next((k for k, t in enumerate(row) if t in eos), -1)
                s = min([x for x in (_find_sub(row, si) for si in stop_ids) if x >= 0], default=-1)
                if e < 0 and s < 0:
                    glen.append(len(row)); why.append("max_new")
                elif s >= 0 and (e < 0 or s <= e):
                    glen.append(s); why.append("stop_string")
                else:
                    glen.append(e); why.append("eos")
            n_tok += sum(glen)
            # residuals: cap[l][0] = prefill [B, L, d]; cap[l][1:] = one [B, 1, d] per decode step —
            # decode step k holds the residual AT generated token k-1, so the last generated token
            # of a capped row has none (fine: its mean covers the rest).
            pre = {l: cap[l][0] for l in layers}
            dec = {l: (torch.cat(cap[l][1:], 1) if len(cap[l]) > 1 else None) for l in layers}
            pm = attn.clone()
            for j, i in enumerate(idx):
                pm[j, L - len(ids[i])] = 0                         # drop BOS / <|im_start|> from the mean
            pmf = pm.unsqueeze(-1).to(pre[layers[0]].dtype)
            gl = torch.tensor(glen, device=dev)
            for a, l in enumerate(layers):
                lastv = pre[l][:, -1].float().cpu().numpy()
                pmv = ((pre[l] * pmf).sum(1) / pmf.sum(1).clamp(min=1)).float().cpu().numpy()
                if dec[l] is not None:
                    Gd = dec[l].shape[1]
                    gm = (torch.arange(Gd, device=dev)[None, :] < gl.clamp(max=Gd)[:, None]).to(dec[l].dtype).unsqueeze(-1)
                    gmv = ((dec[l] * gm).sum(1) / gm.sum(1).clamp(min=1)).float().cpu().numpy()
                    gfv = dec[l][:, 0].float().cpu().numpy()
                for j, i in enumerate(idx):
                    acts["last"][a, i] = lastv[j]
                    acts["prompt_mean"][a, i] = pmv[j]
                    if dec[l] is not None and glen[j] > 0:
                        acts["gen_first"][a, i] = gfv[j]
                        acts["gen_mean"][a, i] = gmv[j]
            if tstore is not None:
                for j, i in enumerate(idx):
                    n_p = len(ids[i])
                    parts = [torch.stack([pre[l][j, L - n_p:] for l in layers], 1)]          # [n_p, nL, d]
                    if dec[layers[0]] is not None and glen[j] > 0:
                        g = min(glen[j], dec[layers[0]].shape[1])
                        parts.append(torch.stack([dec[l][j, :g] for l in layers], 1))
                    tstore.add(rows[i]["id"], torch.cat(parts, 0).to(torch.float16).cpu().numpy(), n_p)
            for j, i in enumerate(idx):
                text = own.decode(gen[j][:glen[j]], skip_special_tokens=True)
                gens[i] = {"id": rows[i]["id"], "dataset": dataset_key, "label": rows[i]["label"],
                           "category": rows[i]["category"], "model": model_key, "prompt": rows[i]["prompt"],
                           "completion": text, "n_gen": glen[j], "stopped_by": why[j],
                           "hit_max": why[j] == "max_new", "is_refusal": R.is_refusal(text),
                           "degenerate": is_degenerate(text)}
            if bi % 5 == 0 or b0 + batch >= N:
                el = time.time() - t0
                log(f"  {model_key}/{dataset_key}: {min(b0 + batch, N)}/{N} prompts · {n_tok} new tokens · "
                    f"{n_tok / max(el, 1e-9):.0f} tok/s · {el / 60:.1f} min")
    finally:
        for h in handles:
            h.remove()
        if tstore is not None:
            tstore.close()
    np.savez(npz_p, layers=np.array(layers), ids=np.array([r["id"] for r in rows]), **acts)
    with gen_p.open("w") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    return npz_p


def run_all(models=None, datasets=None, *, smoke: int = 0, store_tokens=False, layers=LAYERS,
            max_new=MAX_NEW, batch=BATCH, log=print):
    """Every model once (loaded, all datasets, freed). smoke>0 → that many prompts per dataset,
    written to OUT_SMOKE instead of OUT."""
    import gc
    out = OUT_SMOKE if smoke else OUT
    models, datasets = models or list(R.MODELS), datasets or list(R.DATASETS)
    for mk in models:
        todo = [dk for dk in datasets if not ((out / mk / f"{dk}.npz").exists() and (out / mk / f"{dk}.jsonl").exists())]
        if not todo:
            log(f"{mk}: all {len(datasets)} datasets done, model not loaded")
            continue
        t0 = time.time()
        model, own = R.load_model(mk)
        tpl = R.load_tokenizer(mk)
        log(f"{mk}: loaded in {time.time() - t0:.0f}s → {len(todo)} dataset(s)")
        for dk in todo:
            run(mk, dk, model, own, tpl, layers=layers, max_new=max_new, batch=batch,
                store_tokens=store_tokens, limit=smoke or None, out=out, log=log)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return out


# --- readers ------------------------------------------------------------------------------
def load_gens(out: Path = OUT) -> list[dict]:
    return [json.loads(l) for p in sorted(out.glob("*/*.jsonl")) for l in p.open()]


def load_acts(model_key: str, dataset_key: str, position: str, layer: int, out: Path = OUT):
    """(ids, float32 [N, d_model]) for one model × dataset × position × layer."""
    z = np.load(out / model_key / f"{dataset_key}.npz")
    a = list(z["layers"]).index(layer)
    return list(z["ids"]), z[position][a].astype(np.float32)


def summarize(out: Path = OUT):
    """Reference behaviour per model × dataset × label: N, refusal rate ± SE, degenerate rate,
    mean completion length, share that hit the cap (pandas DataFrame)."""
    import pandas as pd
    g = pd.DataFrame(load_gens(out))
    if g.empty:
        return g
    def agg(x):
        n, p = len(x), x["is_refusal"].mean()
        return pd.Series({"N": n, "refusal": round(p, 3), "se": round(math.sqrt(p * (1 - p) / n), 3),
                          "degenerate": round(x["degenerate"].mean(), 3), "mean_n_gen": round(x["n_gen"].mean(), 1),
                          "hit_max": round(x["hit_max"].mean(), 3)})
    return g.groupby(["model", "dataset", "label"]).apply(agg, include_groups=False)
