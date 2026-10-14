"""Steering vectors and the steered-generation grid for "Do genesis vectors steer better?".

    vector(arm)  = mean h(refusal prompts of the dataset) - mean h(compliance prompts of the SAME dataset)
                   h = hidden_states[13] at the last prompt token = the tensor block 13 receives,
                   128 + 128 prompts (data/vector_sets/), each model's own chat template; unit-normalised.
    steering     h <- h + alpha * v at the last prompt token and at every generated token
                   (forward pre-hook on block 13); greedy, 256 new tokens.
    grid         2 models x 8 arms (7 datasets + random) x 4 benchmarks x doses |alpha| in {2, 4, 6, 8, 10, 12},
                 -alpha on the harmful benchmarks (jailbreak), +alpha on the benign ones (over-refusal),
                 plus the unsteered baseline. The same 100 prompts per benchmark for every arm and model.

Unit norm makes alpha an absolute dose in activation units, equal across arms and models;
|h_13| is about 8 on both models, so alpha = 2 / 4 / 6 / 8 / 10 / 12 is 0.25 / 0.50 / 0.75 / 1.00 / 1.25 / 1.50 |h_13|.

Two tokenisation details are kept exactly as in the run the paper's numbers come from:
  * vectors are extracted with the tokenizer adding its special tokens to the rendered chat string
    (Tülu: one BOS; Meta-Llama: two, because its template already writes one);
  * generation tokenises the rendered chat string as is (Tülu: no BOS; Meta-Llama: one).

Files (results/, written here):
    vectors.npz          <model>/mu/<group>, <model>/raw/<arm>, <model>/dir/<arm>   (not committed)
    vectors_meta.json    cosine matrix, raw norms and run settings per model
    completions.jsonl    one row per (model, eval, arm, alpha, prompt)                (never commit: harmful text)

Everything is resume-safe: a model whose means are in vectors.npz is not re-extracted and a cell
whose rows are in completions.jsonl is not regenerated.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import poles

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
VECS = RESULTS / "vectors.npz"
META = RESULTS / "vectors_meta.json"
DUMP = RESULTS / "completions.jsonl"

MODELS = {"tulu_final": "allenai/Llama-3.1-Tulu-3-8B",           # trained on the three genesis datasets
          "meta_instruct": "meta-llama/Llama-3.1-8B-Instruct"}   # same base model, trained on none of them
MODELS_ALL = {**MODELS, "qwen": "Qwen/Qwen3-8B"}                  # + a third model of another family ; kept out
                                                                 # of MODELS so the analyses that loop over MODELS are unchanged
DATA_ARMS = poles.ARMS                                           # 3 genesis (g_*) + 4 non-genesis (x_*)
ARMS = DATA_ARMS + ["random"]
GENESIS = [a for a in DATA_ARMS if a.startswith("g_")]
NON_GENESIS = [a for a in DATA_ARMS if a.startswith("x_")]
LABELS = {"g_coconot_safety": "CoCoNot, safety", "g_wildjailbreak": "WildJailbreak", "g_wildguardmix": "WildGuardMix",
          "x_coconot_leftover": "CoCoNot, left out", "x_donotanswer": "Do-Not-Answer", "x_hh_rlhf": "HH-RLHF",
          "x_pku_saferlhf": "PKU-SafeRLHF", "random": "Random direction"}

LAYER = 13                      # hidden_states[13] = input of block 13
SEED = 0
DOSES = [2, 4, 6, 8, 10, 12]
BENIGN_MAX_DOSE = 8          # over-refusal saturates by |alpha| = 8: benign benchmarks are not run at 10 and 12
GEN_TOKENS = 256
GEN_BS = 100                    # generation batch (one benchmark = one batch)
EXTRACT_BS = 8
EVAL_SIGN = {"harmbench": -1, "strongreject": -1, "xstest_safe": +1, "orbench": +1}
LLAMA_DATE_STRING = "26 Jul 2024"   # Llama-3.1-Instruct stamps a date into its system header; pinned


# ---- data ------------------------------------------------------------------------------------------
def load_evals():
    """{benchmark: (prompts, categories)} - the 100 prompts per benchmark, identical for every cell."""
    out = {}
    for name in EVAL_SIGN:
        rows = [json.loads(l) for l in open(DATA / "evals" / f"{name}.jsonl") if l.strip()]
        out[name] = ([r["prompt"] for r in rows], [r.get("category", "") for r in rows])
    return out


def prompt_groups():
    """{group: prompts}: <arm> = its 128 refusal prompts, neg:<arm> = its 128 compliance prompts."""
    sets = poles.load_vector_sets()
    g = {arm: sets[arm]["refuse"] for arm in DATA_ARMS}
    g.update({"neg:" + arm: sets[arm]["comply"] for arm in DATA_ARMS})
    return g


# ---- model I/O -------------------------------------------------------------------------------------
def load_model(hf_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    return model, tok


def render(prompts, tok):
    """Each prompt as a single user turn in the model's chat template, assistant turn about to begin."""
    return [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                    add_generation_prompt=True, date_string=LLAMA_DATE_STRING,
                                    enable_thinking=False)       # Qwen3 answers directly; the Llama templates ignore it
            for p in prompts]


def group_mean(model, tok, prompts, layer=LAYER, bs=EXTRACT_BS):
    """Mean hidden_states[layer] at the last prompt token over a prompt group (float32 numpy)."""
    import torch
    rendered = render(prompts, tok)
    order = np.argsort([len(tok(r, add_special_tokens=True)["input_ids"]) for r in rendered], kind="stable")
    tok.padding_side = "right"
    dev = next(model.parameters()).device
    acc, n = None, 0
    with torch.inference_mode():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs].tolist()
            enc = tok([rendered[i] for i in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=512, add_special_tokens=True).to(dev)
            hs = model(**enc, output_hidden_states=True).hidden_states[layer]
            lens = enc["attention_mask"].sum(1)
            last = hs[torch.arange(hs.shape[0], device=hs.device), (lens - 1).clamp(min=0)]
            v = last.float().sum(0).cpu().numpy()
            acc = v if acc is None else acc + v
            n += last.shape[0]
    return acc / max(n, 1)


# ---- vectors ---------------------------------------------------------------------------------------
def build_vectors(mu):
    """{group: mean activation} -> (unit vectors, raw vectors, {raw_norm, cos_matrix}).
    raw[arm] = mu[arm] - mu[neg:arm]; random = a seed-0 Gaussian direction; every vector unit-normalised."""
    raw = {a: mu[a] - mu["neg:" + a] for a in DATA_ARMS}
    raw["random"] = np.random.default_rng(SEED).standard_normal(raw[DATA_ARMS[0]].shape).astype(np.float32)
    dirs = {k: (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32) for k, v in raw.items()}
    return dirs, raw, {"raw_norm": {k: float(np.linalg.norm(v)) for k, v in raw.items()},
                       "cos_matrix": {a: {b: float(dirs[a] @ dirs[b]) for b in raw} for a in raw}}


def cached_means(model_key):
    """Group means of `model_key` stored in vectors.npz, or None if any is missing."""
    if not VECS.exists():
        return None
    z = np.load(VECS)
    keys = {g: f"{model_key}/mu/{g}" for g in prompt_groups()}
    return {g: z[k] for g, k in keys.items()} if all(k in z.files for k in keys.values()) else None


def save_vectors(model_key, mu, extra_meta=None):
    """Build the vectors of one model from its group means and store them with their metadata."""
    dirs, raw, dmeta = build_vectors(mu)
    RESULTS.mkdir(exist_ok=True)
    z = dict(np.load(VECS)) if VECS.exists() else {}
    z.update({f"{model_key}/mu/{k}": v for k, v in mu.items()})
    z.update({f"{model_key}/raw/{k}": v for k, v in raw.items()})
    z.update({f"{model_key}/dir/{k}": v for k, v in dirs.items()})
    np.savez(VECS, **z)
    meta = json.loads(META.read_text()) if META.exists() else {}
    meta[model_key] = {"hf": MODELS[model_key], "layer": LAYER, "n_src": poles.N_SRC, "seed": SEED, "arms": ARMS,
                       "unit_norm": True, "raw_norm": dmeta["raw_norm"], "cos_matrix": dmeta["cos_matrix"],
                       **(extra_meta or {})}
    META.write_text(json.dumps(meta, indent=1))
    return dirs, dmeta


def extract_vectors(model_key, model=None, tok=None):
    """Unit steering vectors of one model (GPU unless its means are cached). Returns (dirs, meta)."""
    mu = cached_means(model_key)
    if mu is None:
        own = model is None
        if own:
            model, tok = load_model(MODELS[model_key])
        t0 = time.time()
        mu = {g: group_mean(model, tok, pr) for g, pr in prompt_groups().items()}
        print(f"[{model_key}] extracted {len(mu)} group means in {time.time() - t0:.0f}s", flush=True)
        if own:
            import torch
            del model
            torch.cuda.empty_cache()
    return save_vectors(model_key, mu)


def load_meta():
    return json.loads(META.read_text()) if META.exists() else {}


# ---- generation ------------------------------------------------------------------------------------
class CycleStop:
    """Stop a sequence once its greedy decode has provably locked into a repeat: the last 3*P generated
    tokens are three identical blocks for some period P <= MAX_PERIOD. Under greedy decoding the rest is
    then determined, so nothing is lost; such rows are marked early_stop and count as degenerate.
    Per sequence (the other sequences of the batch keep generating). The pad mask matters: a finished
    sequence is pad-filled, and a run of pad tokens is itself a period-1 cycle."""
    MIN_NEW = 48          # never fire before this many generated tokens
    MAX_PERIOD = 32       # longest repeating unit considered
    EVERY = 16            # check every N steps

    def __init__(self, prompt_len, pad_id):
        self.p0, self.pad_id, self.n, self.hit = prompt_len, pad_id, 0, None

    def __call__(self, input_ids, scores, **kw):
        import torch
        self.n += 1
        b = input_ids.shape[0]
        gen = input_ids[:, self.p0:]
        m = gen.shape[1]
        if self.hit is None:
            self.hit = torch.zeros(b, dtype=torch.bool, device=input_ids.device)
        done = torch.zeros(b, dtype=torch.bool, device=input_ids.device)
        if m < self.MIN_NEW or self.n % self.EVERY:
            return done
        for P in range(1, min(self.MAX_PERIOD, m // 3) + 1):
            a = gen[:, m - 3 * P:m - 2 * P]
            c = gen[:, m - 2 * P:m - P]
            d = gen[:, m - P:]
            done |= ((a == c).all(1) & (c == d).all(1))
        done &= (gen[:, -1] != self.pad_id)
        self.hit |= done
        return done


class Steerer:
    """A loaded model with the block-`layer` pre-hook installed. `generate(prompts, vec)` returns the
    greedy answers with `vec` (already scaled by alpha; None = unsteered) added at the last prompt token
    and at every decode step. `vec` is one vector for all prompts, or one row per prompt (n_prompts x d),
    so that cells with different arms or doses can share a batch."""

    def __init__(self, model, tok, layer=LAYER, gen_tokens=GEN_TOKENS, batch=GEN_BS):
        self.model, self.tok, self.gen_tokens, self.batch = model, tok, gen_tokens, batch
        self._vec = None
        self._handle = model.model.layers[layer].register_forward_pre_hook(self._pre_hook, with_kwargs=True)

    def _pre_hook(self, module, args, kwargs):
        if self._vec is None:
            return None
        hs = kwargs.get("hidden_states")
        kw = hs is not None
        h = hs if kw else args[0]
        v = self._vec.to(h.dtype)
        if v.dim() == 2 and h.shape[1] == 1:      # one vector per row: broadcast over the position axis
            v = v[:, None, :]
        if h.shape[1] == 1:                       # a decode step (cached): the single new position
            h = h + v
        else:                                     # the prompt pass: the last prompt token only
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + v
        if kw:
            kwargs["hidden_states"] = h
            return (args, kwargs)
        return ((h,) + args[1:], kwargs)

    def generate(self, prompts, vec=None):
        """-> (texts, early_stop flags), in the order of `prompts`."""
        import torch
        from transformers import StoppingCriteriaList
        tok, model = self.tok, self.model
        dev = next(model.parameters()).device
        rendered = render(prompts, tok)
        order = np.argsort([len(tok(r)["input_ids"]) for r in rendered], kind="stable")
        tok.padding_side = "left"
        vec = None if vec is None else torch.as_tensor(np.asarray(vec), device=dev)
        per_row = vec is not None and vec.dim() == 2
        self._vec = None if per_row else vec
        outs = []
        try:
            with torch.inference_mode():
                for s in range(0, len(order), self.batch):
                    idx = order[s:s + self.batch].tolist()
                    if per_row:
                        self._vec = vec[idx]
                    enc = tok([rendered[i] for i in idx], return_tensors="pt", padding=True,
                              add_special_tokens=False).to(dev)
                    plen = enc["input_ids"].shape[1]
                    crit = CycleStop(plen, tok.pad_token_id)
                    gen = model.generate(**enc, max_new_tokens=self.gen_tokens, do_sample=False,
                                         pad_token_id=tok.pad_token_id,
                                         stopping_criteria=StoppingCriteriaList([crit]))
                    hit = crit.hit.cpu().numpy() if crit.hit is not None else np.zeros(len(idx), dtype=bool)
                    for j, i in enumerate(idx):
                        outs.append((i, tok.decode(gen[j, plen:], skip_special_tokens=True), bool(hit[j])))
        finally:
            self._vec = None
        outs.sort()
        return [t for _, t, _ in outs], [e for _, _, e in outs]

    def close(self):
        self._handle.remove()


# ---- the grid --------------------------------------------------------------------------------------
def done_cells(dump=DUMP):
    """{(model, eval, arm, alpha)} already in the completions file."""
    seen = set()
    if Path(dump).exists():
        for ln in Path(dump).open():
            if ln.strip():
                r = json.loads(ln)
                seen.add((r["model"], r["eval"], r["arm"], int(r["alpha"])))
    return seen


def grid_cells(model_key, doses=DOSES, arms=ARMS):
    """Every (eval, arm, alpha) of one model: the unsteered baseline, then each arm at each dose with the
    benchmark's sign."""
    cells = []
    for ename, sign in EVAL_SIGN.items():
        cells.append((ename, "baseline", 0))
        cells += [(ename, arm, sign * d) for arm in arms for d in doses if sign < 0 or d <= BENIGN_MAX_DOSE]
    return cells


def run_grid(model_key, doses=DOSES, arms=ARMS, dump=DUMP, max_cells=0):
    """Generate every missing cell of one model and append it to the completions file (GPU).
    Returns the number of cells generated."""
    import torch
    from scoring import is_degenerate, is_refusal
    done = done_cells(dump)
    todo = [c for c in grid_cells(model_key, doses, arms) if (model_key, *c) not in done]
    if max_cells:
        todo = todo[:max_cells]
    if not todo:
        print(f"[{model_key}] grid complete, nothing to generate", flush=True)
        return 0
    evals = load_evals()
    model, tok = load_model(MODELS[model_key])
    dirs, _ = extract_vectors(model_key, model, tok)
    st = Steerer(model, tok)
    Path(dump).parent.mkdir(exist_ok=True)
    print(f"[{model_key}] {len(todo)} cells to generate ({GEN_TOKENS} tokens, {len(evals['harmbench'][0])} prompts each)",
          flush=True)
    with Path(dump).open("a") as fh:
        for n, (ename, arm, alpha) in enumerate(todo, 1):
            t0 = time.time()
            prompts, cats = evals[ename]
            texts, early = st.generate(prompts, None if arm == "baseline" else alpha * dirs[arm])
            ref = [is_refusal(t) for t in texts]
            deg = [is_degenerate(t) or e for t, e in zip(texts, early)]    # a provable cycle is degenerate
            for q, p in enumerate(prompts):
                fh.write(json.dumps({"model": model_key, "eval": ename, "arm": arm, "alpha": int(alpha),
                                     "stage": 1, "gen_tokens": GEN_TOKENS, "n_src": poles.N_SRC,
                                     "behavior": p, "category": cats[q], "generation": texts[q].strip(),
                                     "early_stop": bool(early[q]), "degen": bool(deg[q]),
                                     "refused_regex": bool(ref[q])}) + "\n")
            fh.flush()
            print(f"  [{n}/{len(todo)}] {ename:12s} {arm:20s} a={alpha:+d}  regex-refusal={np.mean(ref):.2f} "
                  f"degen={np.mean(deg):.2f} cut={np.mean(early):.2f}  ({time.time() - t0:.0f}s)", flush=True)
    st.close()
    del model
    torch.cuda.empty_cache()
    return len(todo)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate the steering grid (resume-safe).")
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated model keys")
    ap.add_argument("--doses", default=",".join(map(str, DOSES)), help="comma-separated |alpha| values")
    ap.add_argument("--max-cells", type=int, default=0, help="stop after this many cells per model (smoke test)")
    a = ap.parse_args()
    for mk in a.models.split(","):
        run_grid(mk, doses=[int(x) for x in a.doses.split(",")], max_cells=a.max_cells)
