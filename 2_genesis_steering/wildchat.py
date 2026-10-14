"""A vector from GENERAL instruction-tuning data: the WildChat slice of the Tülu-3 SFT mixture.

The no-safety twin was trained on the whole mixture except the three safety sets, and the direction that separates
refusal from compliance changes most in that step (section 2b), although the general data hold almost no
refusals: 0.7 % of first replies, most of them in WildChat (`tulu_v3.9_wildchat_100k`, GPT-4 answers to real user
chats). This builds that slice's refusal and compliance poles with exactly the rules of poles.py (single-turn rows,
pole from the gold reply with is_refusal / is_clean_compliance, 12-3000 characters, >= 95 % ASCII, de-duplicated,
shuffled with seed 0, capped at 1000 per pole) and its 128 + 128 draw (seed 0, its own generators, so the seven
existing arms' draws are untouched). The arm is called `w_wildchat` and is NOT added to poles.ARMS.

    python wildchat.py      # CPU, a minute; reads the mixture from the local HF cache
Out: data/poles/w_wildchat_{refuse,comply}.jsonl, data/vector_sets/w_wildchat.jsonl
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np

import poles as P

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "1_models_and_datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ARM, SOURCE = "w_wildchat", "allenai/tulu_v3.9_wildchat_100k"


def build():
    import registry as REG
    rows = REG.mixture_rows(_full_source(REG))
    refuse = [(r["user"], r["assistant"]) for r in rows if P.is_refusal(r["assistant"])]
    comply = [(r["user"], r["assistant"]) for r in rows if P.is_clean_compliance(r["assistant"])]
    n = {pole: P._write(ARM, pole, pairs, f"tulu-3-sft-mixture/{SOURCE.split('/')[-1]}") for pole, pairs in (("refuse", refuse), ("comply", comply))}
    print(f"{ARM}: {len(rows)} single-turn rows, {len(refuse)} refusals, {len(comply)} clean compliances -> written {n}")
    sets = {pole: P._subsample(P.load_pole(ARM, pole), P.N_SRC, np.random.default_rng(P.SEED + k)) for k, pole in enumerate(("refuse", "comply"))}
    with open(P.SETS / f"{ARM}.jsonl", "w") as f:
        for pole in ("refuse", "comply"):
            for p in sets[pole]:
                f.write(json.dumps({"pole": pole, "prompt": p}) + "\n")
    return n


def _full_source(REG):
    """The mixture's `source` tag of the WildChat slice (read from the parquet shards)."""
    import glob
    import pyarrow.dataset as pds
    from huggingface_hub import snapshot_download
    root = snapshot_download(REG.MIXTURE, repo_type="dataset", allow_patterns=["data/*.parquet"])
    ds = pds.dataset(sorted(glob.glob(f"{root}/data/*.parquet")), format="parquet")
    tags = set(ds.to_table(columns=["source"]).column("source").to_pylist())
    return next(t for t in tags if t.endswith("tulu_v3.9_wildchat_100k"))


def load_set():
    rows = [json.loads(l) for l in open(P.SETS / f"{ARM}.jsonl") if l.strip()]
    return {pole: [r["prompt"] for r in rows if r["pole"] == pole] for pole in ("refuse", "comply")}


if __name__ == "__main__":
    build()
