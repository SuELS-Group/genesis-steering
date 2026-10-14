"""Build the CoCoNot member / non-member sets for the Tülu-3-8B-SFT positive control.

members     = CoCoNot (prompt, response) pairs that are verbatim rows of allenai/tulu-3-sft-mixture
non-members = CoCoNot release rows that are NOT in the mixture (494 exist), subcategory-matched 1:1
aux pool    = the remaining leftovers, used only as ReCaLL / Con-ReCall non-member prefix shots
Two views are written: chat (Tülu chat template around prompt+response) and resp (response only).
Everything is read from the local HF cache (offline).
"""
import argparse, json, os, random, re
from collections import Counter, defaultdict
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from datasets import load_dataset
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
ap.add_argument("--n-eval", type=int, default=400, help="target number of eval non-members (rest -> aux pool)")
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--redact", action="store_true", help="remove the OpenAI self-reference from every text (see redact())")
ap.add_argument("--match-selfref", action="store_true", help="also match members to non-members on whether the (redacted) response self-references as an AI")
a = ap.parse_args(); random.seed(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

# ── Redaction ──────────────────────────────────────────────────────────────────
# AI2 dropped every CoCoNot row whose response mentions OpenAI (399/401 of the leftovers say
# "As an AI developed by OpenAI" or "against OpenAI's use case policy"; 0/401 mixture rows do).
# A model-free classifier separates the raw split at AUC 0.98, so the mention is removed from
# every text with the same rule: the "developed/created by OpenAI" clause is deleted, and any
# other sentence still containing OpenAI is dropped. Applied to both classes and both views.
CLAUSE = re.compile(r",?\s*(?:which\s+(?:was|is)\s+|that\s+(?:was|is)\s+|who\s+(?:was|is)\s+)?"
                    r"(?:developed|created|made|trained|designed|built|programmed|powered|operated|provided)\s+by\s+OpenAI", re.I)
SENT = re.compile(r"(?<=[.!?])\s+")
def redact(text):
    text = CLAUSE.sub("", text)
    if "OpenAI" not in text and "openai" not in text.lower(): return text
    kept = [s for s in SENT.split(text) if "openai" not in s.lower()]
    return " ".join(kept)

tok = AutoTokenizer.from_pretrained("allenai/Llama-3.1-Tulu-3-8B-SFT")
render = lambda p, r: tok.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": r}], tokenize=False)
norm = lambda s: " ".join(s.split())

mix = load_dataset("allenai/tulu-3-sft-mixture", split="train").filter(lambda x: "coconot" in x["source"].lower(), num_proc=8)
orig = load_dataset("allenai/coconot", "original", split="train")
meta = {norm(p): (c, s) for p, c, s in zip(orig["prompt"], orig["category"], orig["subcategory"])}
mix_prompts = {norm(x["messages"][0]["content"]) for x in mix}
print(f"mixture coconot rows: {len(mix)} | release train rows: {len(orig)} | leftover (never trained on): {len(orig) - len(mix_prompts & set(meta))}")

by_sub = defaultdict(list)
for x in mix:
    p, r = x["messages"][0]["content"], x["messages"][1]["content"]; by_sub[meta[norm(p)][1]].append((p, r))
left = [x for x in orig if norm(x["prompt"]) not in mix_prompts]; random.shuffle(left)
left_by_sub = defaultdict(list)
for x in left: left_by_sub[x["subcategory"]].append(x)
eval_nm, aux = [], []
for s, rows in left_by_sub.items():
    k = round(len(rows) * a.n_eval / len(left)); eval_nm += rows[:k]; aux += rows[k:]
SELFREF = re.compile(r"\b(?:as an? (?:ai|artificial intelligence|language model|ai language model|ai model|ai assistant|virtual assistant|digital assistant)|i am an? (?:ai|artificial intelligence|language model)|i'm an? (?:ai|artificial intelligence|language model))\b", re.I)
def key(sub, resp):
    r = redact(resp) if a.redact else resp
    return (sub, bool(SELFREF.search(r))) if a.match_selfref else (sub,)
pool_by_key = defaultdict(list)
for s, rows in by_sub.items():
    for p, r in rows: pool_by_key[key(s, r)].append((p, r, s))
need = Counter(key(x["subcategory"], x["response"]) for x in eval_nm)
members, short = [], Counter()
for k, n in need.items():
    pool = pool_by_key[k]; random.shuffle(pool); members += pool[:n]
    if len(pool) < n: short[k] = n - len(pool)
if short:  # not enough members for some keys: drop the surplus non-members of those keys so the classes stay matched
    print('keys short of members (dropping surplus non-members):', dict(short))
    for k, n in short.items():
        idx = [i for i, x in enumerate(eval_nm) if key(x['subcategory'], x['response']) == k][:n]
        eval_nm = [x for i, x in enumerate(eval_nm) if i not in set(idx)]
if a.match_selfref: print('self-reference rate  members:', round(sum(bool(SELFREF.search(redact(r) if a.redact else r)) for _, r, _ in members)/len(members), 3), ' non-members:', round(sum(bool(SELFREF.search(redact(x['response']) if a.redact else x['response'])) for x in eval_nm)/len(eval_nm), 3))
print(f"eval non-members: {len(eval_nm)} | members (subcategory-matched): {len(members)} | aux pool: {len(aux)}")

def cap(t):
    if a.redact: t = redact(t)
    ids = tok(t, add_special_tokens=False).input_ids
    return tok.decode(ids[:a.max_tokens]) if len(ids) > a.max_tokens else t
chat = [{"text": cap(render(p, r)), "label": 1, "subcategory": s} for p, r, s in members] + \
       [{"text": cap(render(x["prompt"], x["response"])), "label": 0, "subcategory": x["subcategory"]} for x in eval_nm]
resp = [{"text": cap(r), "label": 1, "subcategory": s} for p, r, s in members] + \
       [{"text": cap(x["response"]), "label": 0, "subcategory": x["subcategory"]} for x in eval_nm]
keep = [i for i, r in enumerate(resp) if len(r['text'].split()) >= 5]
if len(keep) < len(resp): print('dropped', len(resp) - len(keep), 'rows with (near-)empty response after redaction')
chat = [chat[i] for i in keep]; resp = [resp[i] for i in keep]
random.shuffle(chat); random.shuffle(resp)
json.dump(chat, open(out / "coconot_chat.json", "w")); json.dump(resp, open(out / "coconot_resp.json", "w"))
json.dump([{"text": cap(render(x["prompt"], x["response"])), "label": 0} for x in aux], open(out / "aux_chat.json", "w"))
json.dump([{"text": cap(x["response"]), "label": 0} for x in aux], open(out / "aux_resp.json", "w"))
print("wrote", sorted(p.name for p in out.glob("*.json")))
