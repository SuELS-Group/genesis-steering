"""The direction the safety data installed, and which data aligns with it.

Tülu-3-8B-SFT and its twin Tülu-3-8B-SFT-no-safety-data share base model, recipe and data except the safety sets
(CoCoNot, WildJailbreak, WildGuardMix). Whatever differs between their activations was put there by those sets.

Activations (GPU, `python install.py --model <key>`, ~5 min per model, resume-safe per model): hidden_states[L] for
every L (13 = the steering layer, the headline; the sweep covers SWEEP, the full pass all of them), stored in float16, at the last prompt token (the steering recipe of this folder, steering.render / the same padding and truncation) for every
distinct prompt of data/poles and the 400 benchmark prompts of data/evals, on
    tulu_sft        allenai/Llama-3.1-Tulu-3-8B-SFT                   trained on the safety sets
    tulu_nosafety   allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data    identical except for them
    tulu_final      allenai/Llama-3.1-Tulu-3-8B                       the steered model (after DPO and RL)
    base            meta-llama/Llama-3.1-8B                           the shared pretrained model, no post-training
    meta_instruct   meta-llama/Llama-3.1-8B-Instruct                  Meta's post-training of the same base
                    (prompts rendered in Tülu's chat template, tokenized by the base tokenizer)
Out: results/install/acts_<model>.npz {L<layer>: float16 [n, 4096] per layer, prompts json}   (not committed: regenerated)

Quantities (CPU):
    install direction u   the change training made to the refusal contrast on prompts no vector is built from:
                          [mean h(harmful benchmarks) - mean h(benign benchmarks)] on SFT minus the same on the twin
    v_D(model)            a dataset's vector on a model: mean of its 128 refusal minus 128 compliance prompts (the
                          seed-0 draw of data/vector_sets), as in part 2
    a(x)                  per example: (h_SFT(x) - h_twin(x)) . u_hat, how far training moved it along u
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import poles
import steering

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "install"
MODELS = {"tulu_sft": "allenai/Llama-3.1-Tulu-3-8B-SFT",
          "tulu_nosafety": "allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data",
          "tulu_final": "allenai/Llama-3.1-Tulu-3-8B",
          "base": "meta-llama/Llama-3.1-8B",
          "meta_instruct": "meta-llama/Llama-3.1-8B-Instruct"}
TEMPLATE_FROM = {"base": "allenai/Llama-3.1-Tulu-3-8B"}   # the base model has no chat template: render in Tülu's
HARM = ("harmbench", "strongreject"); BENIGN = ("xstest_safe", "orbench")
LAYERS = list(range(33))         # every hidden_states index (0 = embeddings, 32 = last block); 13 = steering.LAYER
SWEEP = [8, 13, 16, 20, 24]      # the layers the sweep repeats its key numbers at


def all_prompts():
    """Every distinct pole prompt (all arms, both poles) followed by the benchmark prompts, in a fixed order."""
    seen, out = set(), []
    for arm in poles.ARMS:
        for pole in ("refuse", "comply"):
            for p in poles.load_pole(arm, pole):
                if p not in seen:
                    seen.add(p); out.append(p)
    for name, (ps, _) in steering.load_evals().items():
        for p in ps:
            if p not in seen:
                seen.add(p); out.append(p)
    return out


def extra_prompts(name):
    """Prompt sets outside poles.ARMS, extracted into their own file: 'wildchat' = two pools."""
    if name == "wildchat":
        return [p for pole in ("refuse", "comply") for p in poles.load_pole("w_wildchat", pole)]
    if name == "detect":                      # more harmful and benign look-alike prompts for the detection test
        D = HERE.parent / "1_models_and_datasets" / "data"
        rows = [json.loads(l) for k in ("advbench", "sorrybench", "xstest") for l in open(D / f"{k}.jsonl") if l.strip()]
        return [r["prompt"] for r in rows]
    if name == "behav":                       # the held-out benign prompts behind the behavioural normal
        import behaviour
        return [r["prompt"] for r in behaviour.load_prompts()]
    if name == "coconot_insft":               # the in-mixture CoCoNot compliance pool
        import coconot_insft
        return poles.load_pole(coconot_insft.ARM, "comply")
    if name == "membership":                  # WildGuardMix non-member pools and the member draws
        import membership as MB
        return [r["prompt"] for pole in MB.POLES2 for r in MB.pool("nonmember", pole)] + \
               [p for a in MB.arms() for pole in MB.POLES2 for p in MB.load_draw(a)[pole]]
    if name == "behav_harm":                  # the held-out adversarial harmful prompts behind n_harm
        import behaviour_harm
        return [r["prompt"] for r in behaviour_harm.load_prompts()]
    raise ValueError(name)


def extract(model_key, bs=steering.EXTRACT_BS, extra=None):
    import torch
    f = OUT / (f"acts_{model_key}.npz" if extra is None else f"acts_{model_key}_{extra}.npz")
    if f.exists():
        print(f"{model_key}: {f.name} exists, skip"); return
    prompts = all_prompts() if extra is None else list(dict.fromkeys(extra_prompts(extra)))
    model, tok = steering.load_model(MODELS[model_key])
    if model_key in TEMPLATE_FROM:
        from transformers import AutoTokenizer
        rendered = steering.render(prompts, AutoTokenizer.from_pretrained(TEMPLATE_FROM[model_key]))
    else:
        rendered = steering.render(prompts, tok)
    order = np.argsort([len(tok(r, add_special_tokens=True)["input_ids"]) for r in rendered], kind="stable")
    tok.padding_side = "right"
    acts = {L: np.zeros((len(prompts), model.config.hidden_size), dtype=np.float32) for L in LAYERS}
    with torch.inference_mode():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs].tolist()
            enc = tok([rendered[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512,
                      add_special_tokens=True).to("cuda")
            hss = model(**enc, output_hidden_states=True).hidden_states
            lens = enc["attention_mask"].sum(1)
            for L in LAYERS:
                hs = hss[L]
                acts[L][idx] = hs[torch.arange(hs.shape[0], device=hs.device), (lens - 1).clamp(min=0)].float().cpu().numpy()
            if s % 2000 == 0:
                print(f"  {model_key}: {s}/{len(order)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(f, **{f"L{L}": a.astype(np.float16) for L, a in acts.items()}, prompts=np.array(json.dumps(prompts)))
    print(f"{model_key}: wrote {len(prompts)} activations", flush=True)


def load(model_key, layer=None, extra=None):
    """(acts at `layer` (default the steering layer), {prompt: row}) or None if not extracted yet."""
    f = OUT / (f"acts_{model_key}.npz" if extra is None else f"acts_{model_key}_{extra}.npz")
    if not f.exists():
        return None
    z = np.load(f)
    return z[f"L{steering.LAYER if layer is None else layer}"].astype(np.float32), {p: i for i, p in enumerate(json.loads(str(z["prompts"])))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--extra", choices=["wildchat", "detect", "behav", "coconot_insft", "membership", "behav_harm"], default=None, help="extract an extra prompt set into its own file")
    a = ap.parse_args(); extract(a.model, extra=a.extra)
