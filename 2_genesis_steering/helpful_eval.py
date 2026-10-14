"""A general-helpfulness evaluation benchmark for the +alpha direction .

WHY. Every benign benchmark in this repository is of one genre: a prompt that is safe but *looks* unsafe
(`xstest_safe`, `orbench`, and `behaviour.py`'s held-out OR-Bench-hard set). Those sit close to the model's refusal
boundary, so almost any push along a refusal-ish direction flips them, and at |alpha| = 8
grid reaches 1.000 on XSTest-safe — the direction has no resolution. That run could therefore only read its
top-against-bottom difference off HarmBench, the -alpha (suppress refusal) direction.

This set is the opposite genre: plainly benign, plainly useful instructions with no safety hook at all, where refusing
is not a behaviour the model has any reason to produce. Making the model refuse *these* should be hard, which is what
gives the +alpha direction headroom. The judge's four labels separate the two ways a vector can fail here: A (it really
refused) against C / D (it merely derailed into off-topic or degenerate text).

SOURCE. tatsu-lab/alpaca, the 31,323 instructions that carry no separate `input` field, so each is a standalone prompt.
Filters, in order:
  - `poles._keep` (the house filter: 12..3000 chars, >= 95% ASCII), then 20..300 chars, which puts the prompt-length
    distribution alongside the existing benchmarks (median 55 against xstest_safe's 44, orbench's 109)
  - drop instructions that refer to an attachment they do not have ("summarize the following passage")
  - drop anything a safety-tuned model could *legitimately* decline, on a broad keyword screen: the point of the set is
    that a refusal is never the right answer, so any prompt with a defensible refusal is noise. (This removes only ~136
    of 28,902; alpaca is already benign.)
DECONTAMINATION. Against every prompt of `data/evals/*.jsonl` and of `data/evals_heldout/*.jsonl`, and against the
first user turn of every row of `allenai/tulu-3-sft-mixture` (normalised: lowercased, whitespace collapsed), so no
prompt of the benchmark is one Tulu was trained on. The mixture pass is required, not optional: a silently skipped
decontamination is worse than a missing file.
DRAW. 100 prompts, `poles.SEED`, matching the 100 of every other benchmark (the reduced grids steer the first 50).

    python helpful_eval.py prompts        # -> data/evals_heldout/helpful.jsonl

The file is {"source": "alpaca", "prompt": ..., "category": ""}: `source`/`prompt` as behaviour.py's held-out set,
`category` so steering.load_evals' reader can take it unchanged. The benchmark's sign is +1 (as xstest_safe): the rate
it measures is the share *refused*.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
import re
from pathlib import Path

import numpy as np

import poles

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "data" / "evals_heldout" / "helpful.jsonl"
SOURCE = "alpaca"
# Alpaca is read straight from a local arrow shard (datasets.Dataset.from_file takes no lock, which matters when the
# datasets cache is not writable). Set ALPACA_ARROW to the shard of your own tatsu-lab/alpaca download.
ARROW = os.environ.get("ALPACA_ARROW", "")
MIXTURE = "allenai/tulu-3-sft-mixture"
N_EVAL, MIN_CHARS, MAX_CHARS = 100, 20, 300

NEEDS_INPUT = re.compile(
    r"\b(the following|following (text|sentence|paragraph|passage|list|code|article|data)"
    r"|given (text|sentence|paragraph|passage|list|code|article|data)"
    r"|this (sentence|paragraph|passage|text|article|code|list)|below|above"
    r"|the (sentence|passage|paragraph|article) )", re.I)
DECLINABLE = re.compile(
    r"\b(weapon|gun|firearm|bomb|explosiv|poison|toxin|drug|narcotic|cocaine|heroin|meth|overdose"
    r"|suicid|self.harm|kill|murder|assault|abuse|harass|stalk|hack|malware|virus|phish|exploit"
    r"|steal|theft|fraud|launder|counterfeit|forge|smuggl|traffick|illegal|unlawful|crime|criminal"
    r"|porn|sexual|nude|explicit|racis|sexis|slur|hate|discriminat|extremis|terror|radicali"
    r"|vaccine|medical advice|diagnos|prescri|dosage|legal advice|lawsuit|invest|stock pick|gambl)\b", re.I)


def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def _key(s):
    return hashlib.blake2b(_norm(s).encode(), digest_size=16).digest()


def _used_prompts():
    """Every prompt already serving as an evaluation prompt anywhere in part 2."""
    out = set()
    for d in ("evals", "evals_heldout"):
        for f in sorted((HERE / "data" / d).glob("*.jsonl")):
            if f == PROMPTS:
                continue
            out |= {_key(json.loads(l)["prompt"]) for l in open(f) if l.strip()}
    return out


def _mixture_keys():
    """The first user turn of every row of Tulu's SFT mixture. Read from the cached parquet shards in batches:
    load_dataset cannot resolve the builder with HF_HUB_OFFLINE set, and the full frame does not need to be in memory."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(MIXTURE, repo_type="dataset", local_files_only=True))
    shards = sorted(root.glob("data/*.parquet"))
    assert shards, f"no parquet shards under {root}; the mixture must be cached for the decontamination pass"
    out, n = set(), 0
    for sh in shards:
        for batch in pq.ParquetFile(sh).iter_batches(batch_size=5000, columns=["messages"]):
            for msgs in batch.column("messages").to_pylist():
                n += 1
                for m in msgs:
                    if m.get("role") == "user":
                        out.add(_key(m["content"])); break
        print(f"  mixture: {sh.name} done, {n} rows, {len(out)} distinct first user turns", flush=True)
    return out


def build_prompts():
    from datasets import Dataset
    A = Dataset.from_file(ARROW)
    ins = [r["instruction"].strip() for r in A if not r["input"].strip()]
    pool = [p for p in ins
            if poles._keep(p) and MIN_CHARS <= len(p) <= MAX_CHARS
            and not NEEDS_INPUT.search(p) and not DECLINABLE.search(p)]
    pool = list(dict.fromkeys(pool))
    print(f"alpaca: {len(A)} rows, {len(ins)} standalone instructions, {len(pool)} after the filters", flush=True)

    used, mix = _used_prompts(), _mixture_keys()
    kept, n_eval, n_mix = [], 0, 0
    for p in pool:
        k = _key(p)
        if k in used:
            n_eval += 1; continue
        if k in mix:
            n_mix += 1; continue
        kept.append(p)
    print(f"decontamination: dropped {n_eval} already used as evaluation prompts, {n_mix} present in the mixture "
          f"-> {len(kept)} eligible", flush=True)

    rng = np.random.default_rng(poles.SEED)
    pick = [kept[i] for i in sorted(rng.choice(len(kept), N_EVAL, replace=False))]
    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROMPTS, "w") as f:
        for p in pick:
            f.write(json.dumps({"source": SOURCE, "prompt": p, "category": ""}, ensure_ascii=False) + "\n")
    L = [len(p) for p in pick]
    print(f"wrote {len(pick)} prompts -> {PROMPTS}\n"
          f"  prompt chars: p10 {int(np.percentile(L, 10))} median {int(np.median(L))} p90 {int(np.percentile(L, 90))}")


def load_prompts():
    return [json.loads(l) for l in open(PROMPTS) if l.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prompts"])
    ap.parse_args()
    build_prompts()
