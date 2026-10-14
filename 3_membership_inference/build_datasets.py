"""Dataset-level panel for ReCaLL on Tülu-3-8B-SFT: in-training safety sets vs external safety sets.

members   (label 1, verbatim rows of allenai/tulu-3-sft-mixture): coconot_safety, wildjailbreak, wildguardmix
externals (label 0, never in the mixture):                         pku_saferlhf (safe response), hh_harmless (chosen),
                                                                   dna_gpt4, dna_claude (Do-Not-Answer), coconot_leftover_safety
aux pool: 10 hh_harmless rows outside the panel → ReCaLL's fixed non-member prefix, identical for every dataset and model.
Every response has the OpenAI self-reference redacted (same rule as build_sets.py), all texts are chat-rendered with the
Tülu template (plain-text tags, so all three models tokenize them identically) and capped at 512 tokens.
"""
import argparse, glob, gzip, json, os, random, re
from collections import Counter, defaultdict
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "datasets_split"))
ap.add_argument("--template", choices=["tulu", "meta"], default="tulu", help="chat template used to render the pairs (meta = Llama-3.1-Instruct's own)")
ap.add_argument("--n", type=int, default=300); ap.add_argument("--max-tokens", type=int, default=512); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args(); random.seed(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
HUB = os.path.expanduser("~/.cache/huggingface/hub")
tok = AutoTokenizer.from_pretrained({"tulu": "allenai/Llama-3.1-Tulu-3-8B-SFT", "meta": "meta-llama/Llama-3.1-8B-Instruct"}[a.template])
def render(p, r):
    s = tok.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": r}], tokenize=False)
    return s[len(tok.bos_token):] if tok.bos_token and s.startswith(tok.bos_token) else s   # the runner's tokenizer adds BOS itself
norm = lambda s: " ".join(s.split())
CLAUSE = re.compile(r",?\s*(?:which\s+(?:was|is)\s+|that\s+(?:was|is)\s+|who\s+(?:was|is)\s+)?(?:developed|created|made|trained|designed|built|programmed|powered|operated|provided)\s+by\s+OpenAI", re.I)
SENT = re.compile(r"(?<=[.!?])\s+")
def redact(t):
    t = CLAUSE.sub("", t)
    return t if "openai" not in t.lower() else " ".join(s for s in SENT.split(t) if "openai" not in s.lower())
def cap(t):
    ids = tok(t, add_special_tokens=False).input_ids
    return tok.decode(ids[:a.max_tokens]) if len(ids) > a.max_tokens else t
def pick(pairs, n):
    pairs = [(p, r) for p, r in pairs if len(redact(r).split()) >= 5]; random.shuffle(pairs); return pairs[:n]

panel = {}
# ── members: verbatim mixture rows ────────────────────────────────────────────
mix = load_dataset("allenai/tulu-3-sft-mixture", split="train")
orig = load_dataset("allenai/coconot", "original", split="train")
cat = {norm(p): c for p, c in zip(orig["prompt"], orig["category"])}
def rows(src_key):
    d = mix.filter(lambda x: src_key in x["source"].lower(), num_proc=8)
    return [(x["messages"][0]["content"], x["messages"][1]["content"]) for x in d if len(x["messages"]) == 2 and x["messages"][0]["role"] == "user"]
cc = rows("coconot")
panel["coconot_safety"] = pick([(p, r) for p, r in cc if cat.get(norm(p)) == "Requests with safety concerns"], a.n)
panel["wildjailbreak"] = pick(rows("wildjailbreak"), a.n)
panel["wildguardmix"] = pick(rows("wildguardmix"), a.n)
# ── externals ─────────────────────────────────────────────────────────────────
mix_cc_prompts = {norm(p) for p, _ in cc}
panel["coconot_leftover_safety"] = pick([(p, r) for p, r, c in zip(orig["prompt"], orig["response"], orig["category"])
                                         if norm(p) not in mix_cc_prompts and c == "Requests with safety concerns"], a.n)
pku = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train").shuffle(seed=a.seed).select(range(20000))
pk = []
for x in pku:
    if str(x["is_response_0_safe"]) == "True": pk.append((x["prompt"], x["response_0"]))
    elif str(x["is_response_1_safe"]) == "True": pk.append((x["prompt"], x["response_1"]))
panel["pku_saferlhf"] = pick(pk, a.n)
hh_path = glob.glob(f"{HUB}/datasets--Anthropic--hh-rlhf/snapshots/*/harmless-base/train.jsonl.gz")[0]
hh = []
with gzip.open(hh_path, "rt") as f:
    for line in f:
        c = json.loads(line)["chosen"]
        m = re.match(r"^\s*Human: (.*?)\n\nAssistant: (.*)$", c, re.S)
        if m and "\n\nHuman:" not in m.group(2) and len(m.group(2).split()) >= 5: hh.append((m.group(1).strip(), m.group(2).strip()))
random.shuffle(hh); panel["hh_harmless"] = hh[:a.n]; aux = hh[a.n:a.n + 10]
dna = load_dataset("LibrAI/do-not-answer", split="train").shuffle(seed=a.seed)
panel["dna_gpt4"] = pick([(x["question"], x["GPT4_response"]) for x in dna.select(range(0, 450))], a.n)
panel["dna_claude"] = pick([(x["question"], x["Claude_response"]) for x in dna.select(range(450, len(dna)))], a.n)
MEMBERS = {"coconot_safety", "wildjailbreak", "wildguardmix"}
rows_out = []
for name, pairs in panel.items():
    for p, r in pairs:
        rows_out.append({"text": cap(render(redact(p), redact(r))), "label": int(name in MEMBERS), "subcategory": name, "dataset": name})
random.shuffle(rows_out)
json.dump(rows_out, open(out / "datasets_chat.json", "w"))
json.dump([{"text": cap(render(redact(p), redact(r))), "label": 0, "dataset": "hh_harmless_aux"} for p, r in aux], open(out / "aux_chat.json", "w"))
print("panel:", {k: len(v) for k, v in panel.items()}, "| aux:", len(aux), "| total scored:", len(rows_out))
lens = defaultdict(list)
for r in rows_out: lens[r["dataset"]].append(len(tok(r["text"], add_special_tokens=False).input_ids))
for k, v in lens.items(): print(f"  {k:24s} n={len(v):3d}  tokens mean={sum(v)/len(v):6.1f}  max={max(v)}  openai-mentions={sum('openai' in r['text'].lower() for r in rows_out if r['dataset']==k)}")
