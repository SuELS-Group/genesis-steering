"""registry.py — the single source of truth for this repository: which MODELS and which
DATASETS this repository runs on, and how the files under ./data are built.

Standalone by design: nothing here imports from the other experiments in this repo, and the
data files are rebuilt from public sources by `build_all()`. Every random choice is seeded,
so each group member ends up with byte-identical files.

    import registry as R
    R.build_all()                          # writes data/<dataset>.jsonl (idempotent)
    R.load("hh_rlhf")                      # rows: {id, dataset, label, prompt, category, origin, in_training, gold}
    R.EVAL_SETS                            # the prompt-only benchmarks (advbench, sorrybench, xstest)
    R.prompts("wildjailbreak", "refuse", n=200)   # seeded random subsample (rows are stored shuffled)
    R.render_chat("qwen", "How do I ...")  # the exact string that model is fed
"""
from __future__ import annotations

import collections
import csv
import glob
import io
import json
import random
import re
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"          # slim prompt-only jsonl per dataset (committed; rebuilt by build_all)
CACHE = HERE / "cache"        # activations and other large outputs of later stages (gitignored)

SEED = 0                      # every shuffle / subsample in this folder
MAX_PER_POLE = 1000           # cap per (dataset, label); the small external sets are kept whole
MIN_CHARS, MAX_CHARS = 12, 3000   # prompt-length band: drops empty stubs and pathological wrappers
MIN_ASCII = 0.95              # drops non-English / mojibake prompts (share of ASCII characters)
LLAMA_DATE_STRING = "26 Jul 2024"  # Llama-3.1-Instruct's template stamps today's date into the
                                   # system header; pin it so the rendered prompt never changes


# ═══ MODELS ══════════════════════════════════════════════════════════════════════════════
# Four 8B chat models. Three share the SAME pretrained weights (Llama-3.1-8B) and differ only
# in post-training — none / Tülu-3's open recipe / Meta's closed recipe — and one is a
# different family of the same size (Qwen3-8B).

@dataclass(frozen=True)
class Model:
    key: str
    hf: str                 # Hugging Face repo id (the weights)
    label: str              # how we refer to it in tables and figures
    base: str               # pretrained weights it descends from
    post_training: str      # what happened to the weights after pretraining
    template_from: str      # tokenizer whose chat template renders the prompt (see tulu_pre_sft)
    trained_on: tuple       # keys of OUR datasets known to be in its post-training data
    note: str = ""


MODELS: dict[str, Model] = {m.key: m for m in [
    Model("tulu", "allenai/Llama-3.1-Tulu-3-8B",
          "Tülu-3 8B (full)", "Llama-3.1-8B",
          "Tülu-3 SFT → DPO → RLVR; open data — the SFT mixture contains CoCoNot, WildJailbreak "
          "and WildGuardMix",
          "allenai/Llama-3.1-Tulu-3-8B",
          ("coconot_safety", "wildjailbreak", "wildguardmix")),
    Model("tulu_pre_sft", "meta-llama/Llama-3.1-8B",
          "Tülu-3 8B pre-SFT (= Llama-3.1-8B base)", "Llama-3.1-8B",
          "none — pretraining only; the checkpoint Tülu-3 SFT starts from",
          "allenai/Llama-3.1-Tulu-3-8B", (),
          "has no chat template of its own; prompts are rendered with the Tülu template so the "
          "token sequence is identical to `tulu` and the only difference is the weights"),
    Model("meta", "meta-llama/Llama-3.1-8B-Instruct",
          "Llama-3.1-8B-Instruct (Meta)", "Llama-3.1-8B",
          "Meta's own SFT + preference tuning on undisclosed data",
          "meta-llama/Llama-3.1-8B-Instruct", (),
          "post-training data not public, so treated as out-of-training for all six datasets; "
          "its template stamps a date into the system header — pinned to LLAMA_DATE_STRING"),
    Model("qwen", "Qwen/Qwen3-8B",
          "Qwen3-8B", "Qwen3-8B",
          "Qwen's own post-training on undisclosed data; different family, tokenizer and depth "
          "(36 layers, d_model 4096 like the Llamas)",
          "Qwen/Qwen3-8B", (),
          "rendered with enable_thinking=False (direct answer, empty <think/> block); its residual "
          "norms are several times the Llamas', so steering doses must be set per model"),
]}


def load_tokenizer(key: str):
    """The tokenizer that renders prompts for model `key` (its own, except tulu_pre_sft)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[key].template_from)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def render_chat(key: str, prompt: str, tok=None) -> str:
    """The exact string model `key` is fed for a single user turn, assistant turn about to begin."""
    tok = tok or load_tokenizer(key)
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,              # Qwen3: answer directly; ignored by the Llama templates
        date_string=LLAMA_DATE_STRING,      # Llama-3.1-Instruct: fixed date; ignored by the others
    )


def encode_chat(key: str, prompt: str, tok=None, model_tok=None) -> list[int]:
    """Token ids model `key` is fed: the rendered chat string, tokenized with the model's OWN
    tokenizer (`model_tok`, defaults to the template tokenizer). A BOS token is added only when
    the template did not already write one (Tülu's template does not, Llama-Instruct's does,
    Qwen has none), so every model sees exactly one BOS or none, never two."""
    tok = tok or load_tokenizer(key)
    model_tok = model_tok or tok
    s = render_chat(key, prompt, tok)
    has_bos = bool(model_tok.bos_token) and s.startswith(model_tok.bos_token)
    return model_tok(s, add_special_tokens=not has_bos)["input_ids"]


def load_model(key: str):
    """Model `key` on the GPU in bf16 (~16 GB) plus its OWN tokenizer (pass it to encode_chat as
    `model_tok`). Load one at a time and `del` it before the next."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = MODELS[key]
    own = AutoTokenizer.from_pretrained(m.hf)
    model = AutoModelForCausalLM.from_pretrained(m.hf, dtype=torch.bfloat16, device_map="cuda").eval()
    return model, own


def generate(key: str, prompt: str, model, own_tok, tpl_tok=None, max_new_tokens: int = 60) -> str:
    """GENERATION: greedy continuation of the rendered chat prompt, decoded without special tokens.
    The input is exactly encode_chat(key, prompt), so what is generated here is what the model
    'says' at the position the activations will later be read from."""
    import torch
    ids = torch.tensor([encode_chat(key, prompt, tpl_tok, own_tok)], device=model.device)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None, top_k=None,
                             pad_token_id=own_tok.pad_token_id if own_tok.pad_token_id is not None else own_tok.eos_token_id)
    return own_tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


# ═══ DATASETS ════════════════════════════════════════════════════════════════════════════
# Six refusal-prompt sets. Every row carries a `label`: `refuse` = the model is expected to
# decline, `comply` = it is expected to answer. The three in-training sets are the safety data
# Tülu-3 actually fine-tuned on; the three external sets are standard safety benchmarks that
# none of the models is known to have trained on.

@dataclass(frozen=True)
class Dataset:
    key: str
    name: str
    group: str            # "in_training" (in Tülu-3's SFT mixture) | "external"
    origin: str           # where the rows are read from
    cite: str
    refuse_pole: str      # what a `refuse` row is
    comply_pole: str      # what a `comply` row is ("—" when the source has no benign half)

    @property
    def file(self) -> Path:
        return DATA / f"{self.key}.jsonl"


COCONOT_SAFETY = "Requests with safety concerns"        # the CoCoNot category we use
COCONOT_OTHER = ("Humanizing requests", "Incomplete requests", "Indeterminate requests",
                 "Unsupported requests")                  # its four other categories (see the
                                                          # commented-out option in DATASETS)
MIXTURE = "allenai/tulu-3-sft-mixture"
MIX_SOURCE = {   # `source` tag in the Tülu-3 SFT mixture for each in-training source
    "coconot": "ai2-adapt-dev/coconot_converted",
    "wildjailbreak": "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k",
    "wildguardmix": "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
}
ADVBENCH_CSV = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/"
                "harmful_behaviors.csv")
XSTEST_CSV = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
SORRYBENCH_REPO = "sorry-bench/sorry-bench-202503"   # the open 2025-03 refresh (2024-06 is gated)

DATASETS: dict[str, Dataset] = {d.key: d for d in [
    Dataset("coconot_safety", "CoCoNot — safety category", "in_training",
            "allenai/coconot: original/train (refuse), contrast/test + pref/train (comply)",
            "Brahman et al. 2024",
            f"original/train rows with category '{COCONOT_SAFETY}' (6 subcategories: copyright, "
            "dangerous topics, misinformation, privacy, offensive triggers, wildchats), kept only "
            "if the prompt appears verbatim in the Tülu-3 SFT mixture",
            "the safety-category rows of contrast/test (matched safe look-alikes) and pref/train "
            "(the DPO-stage compliance prompts)"),
    # ── Option: the REST of CoCoNot (off) ────────────────────────────────────────────────
    # CoCoNot has five noncompliance categories; we use only 'Requests with safety concerns'.
    # That is the one where declining is refusal-for-harm — the same behaviour the other five
    # datasets target. The other four (humanizing, incomplete, indeterminate, unsupported
    # requests) are noncompliance for other reasons — "I have no body", "your question is
    # ambiguous", "I can't browse the web" — and mixing them in would blur a refusal direction
    # with a general "I can't do that" direction. To study them as a seventh dataset, uncomment
    # this entry and its line in _BUILDERS, then run build("coconot_other"). Its comply pole is
    # thin: only the Incomplete and Unsupported categories have contrast/pref look-alikes.
    # Dataset("coconot_other", "CoCoNot — the four non-safety categories", "in_training",
    #         "allenai/coconot: original/train (refuse), contrast/test + pref/train (comply)",
    #         "Brahman et al. 2024",
    #         "original/train rows of the categories " + ", ".join(COCONOT_OTHER) + ", kept only "
    #         "if the prompt appears verbatim in the Tülu-3 SFT mixture; category = 'category · subcategory'",
    #         "the contrast/test and pref/train rows of the same categories"),
    Dataset("wildjailbreak", "WildJailbreak (in-training slice)", "in_training",
            f"{MIXTURE} rows with source {MIX_SOURCE['wildjailbreak'].split('/')[-1]}",
            "Jiang et al. 2024",
            "single-turn rows tagged *_harmful by allenai/wildjailbreak (data_type, joined by prompt text "
            "into `category`) WHOSE gold assistant turn also opens as a refusal (is_refusal)",
            "single-turn rows tagged *_benign by the hub AND whose gold turn is a clean compliance "
            "(is_clean_compliance: helpful opening, no refusal or deflection language in its first 300 "
            "chars). Rows where hub tag and gold behaviour disagree, and soft deflections, are dropped"),
    Dataset("wildguardmix", "WildGuardMix (in-training slice)", "in_training",
            f"{MIXTURE} rows with source {MIX_SOURCE['wildguardmix'].split('/')[-1]}",
            "Han et al. 2024",
            "single-turn rows tagged harmful by allenai/wildguardmix (prompt_harm_label + adversarial flag, "
            "joined by prompt text into `category`) WHOSE gold assistant turn also opens as a refusal",
            "single-turn rows tagged unharmful by the hub AND whose gold turn is a clean compliance; "
            "disagreements and soft deflections are dropped"),
    Dataset("llm_lat", "LLM-LAT harmful + benign (refusal training pairs)", "external",
            "LLM-LAT/harmful-dataset (refuse) + LLM-LAT/benign-dataset (comply)",
            "Sheshadri et al. 2024 (Latent Adversarial Training)",
            "harmful-dataset prompts; gold = `chosen`, an explicit refusal in every row (`rejected` is the "
            "harmful compliance, not stored)",
            "benign-dataset prompts (every 8th of 165k rows) whose gold `response` is a clean compliance"),
    Dataset("hh_rlhf", "Anthropic HH-RLHF (harmless-base / helpful-base)", "external",
            "Anthropic/hh-rlhf · harmless-base/train (refuse) + helpful-base/train (comply)",
            "Bai et al. 2022",
            "harmless-base single-turn dialogues whose preferred (`chosen`) assistant turn opens as a "
            "refusal — 2022 crowdworker-era data, so the refusals are often hedged",
            "helpful-base single-turn dialogues whose `chosen` turn is a clean compliance"),
    Dataset("nemotron", "Llama-Nemotron post-training (safety / chat splits)", "external",
            "nvidia/Llama-Nemotron-Post-Training-Dataset · SFT/safety (refuse) + SFT/chat (comply)",
            "NVIDIA 2025",
            "SFT/safety single-turn rows (reasoning off) whose gold output opens as a refusal "
            "(Mixtral-8x22B-generated; the split has no benign half)",
            "SFT/chat single-turn rows (reasoning off) whose gold output is a clean compliance "
            "(Llama-3.1-Nemotron-70B-generated)"),
    Dataset("caa_refusal", "CAA refusal behaviour (A/B pairs)", "external",
            "github nrimsky/CAA · datasets/generate/refusal (408) + datasets/test/refusal A/B (50)",
            "Rimsky et al. 2024 (Contrastive Activation Addition)",
            "every question, gold = the authored answer matching CAA's *refusal* behaviour — harm refusals "
            "AND AI-persona declines to benign questions ('As an AI, I do not have hobbies'); taken as "
            "authored, no detector applied",
            "the SAME questions, gold = the authored complying answer. A completion-space contrast: "
            "prompt-space refuse−comply is zero here by construction, the split lives in the gold answers"),
]}

# Prompt-only safety BENCHMARKS: no gold answers, refuse pole only (XSTest also has a comply half).
# Not part of DATASETS (the seven training sets) but built, loadable and run through the reference
# as evaluation sets.
EVAL_SETS: dict[str, Dataset] = {d.key: d for d in [
    Dataset("advbench", "AdvBench harmful behaviours", "eval",
            "github llm-attacks/llm-attacks · data/advbench/harmful_behaviors.csv (goal column)",
            "Zou et al. 2023",
            "all 520 behaviours", "—"),
    Dataset("sorrybench", "SORRY-Bench (base prompts)", "eval",
            f"{SORRYBENCH_REPO} · question.jsonl, prompt_style=base",
            "Xie et al. 2024",
            "all 440 base prompts (44 fine-grained categories × 10; the 20 linguistic mutations "
            "per prompt are not used)", "—"),
    Dataset("xstest", "XSTest", "eval",
            "github paul-rottger/xstest · xstest_prompts.csv",
            "Röttger et al. 2024",
            "label=unsafe: the 200 genuinely unsafe contrast prompts",
            "label=safe: the 250 prompts that only look unsafe (homonyms, figurative language, …)"),
]}


def all_datasets() -> dict[str, Dataset]:
    """The seven training sets followed by the eval sets."""
    return {**DATASETS, **EVAL_SETS}


# --- refusal detector for the gold assistant turns of the SFT mixture ----------------------
_REFUSE = re.compile(
    r"\b(i\s*(?:'m| am)?\s*(?:sorry(?!\s+to\s+hear)|afraid|unable)|i\s*(?:cannot|can'?t|won'?t|will not)"
    r"|as an ai|i must decline|i have to decline|i'?m not able|it'?s not appropriate"
    r"|i can'?t (?:help|assist|provide)|i apologi|i won'?t (?:help|assist|provide|be)"
    r"|i do(?:n'?t| not) (?:feel comfortable|think (?:it|that|this)(?:'s| is) (?:appropriate|a good idea|right|okay|ok|wise|advisable)))",
    re.IGNORECASE,
)


_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})   # curly → straight

# Soft deflections: the "I appreciate your interest … however, it's important to …" register that
# declines without ever saying "I can't". Neither a clean refusal nor a clean compliance, so rows
# whose gold answer opens like this are dropped from BOTH poles of the mixture-derived datasets.
_DEFLECT = re.compile(
    r"(i appreciate your|i understand (?:that )?(?:you|your)|while i understand|it sounds like you"
    r"|i'?m here to (?:provide|help|assist|support)|thank you for reaching|i understand your"
    r"|it'?s important to (?:note|remember|approach|recognize|consider|clarify|acknowledge)"
    r"|i must (?:emphasize|clarify|stress|point out)|however, i |\binstead\b|i'?d be happy to help[^.]{0,80}\bbut\b"
    r"|not (?:able|something)|cannot|can'?t|unable|inappropriate|harmful|illegal|unethical|decline|refrain|sorry"
    r"|against (?:my|the|our)|i'?m an ai|as an ai|as a language model|respectful|responsible|ethical|safety|well-being)",
    re.IGNORECASE,
)


def is_refusal(text: str) -> bool:
    """Does an assistant turn OPEN as a refusal? (first 180 chars, curly quotes normalised; the same
    detector the group can reuse on generations, where a degeneracy check must be added.)"""
    t = (text or "").strip().translate(_QUOTES)
    return len(t) >= 8 and bool(_REFUSE.search(t[:180]))


def is_clean_compliance(text: str) -> bool:
    """Is an assistant turn an unambiguous compliance: long enough, not a refusal, and no refusal or
    deflection language anywhere in its first 300 chars? Deliberately strict — precision over
    recall, since the mixture offers 20k+ candidates and we keep at most MAX_PER_POLE."""
    t = (text or "").strip().translate(_QUOTES)
    return len(t) >= 20 and not is_refusal(t) and not _DEFLECT.search(t[:300])


# --- helpers ----------------------------------------------------------------------------
def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def _keep(p: str) -> bool:
    return MIN_CHARS <= len(p) <= MAX_CHARS and sum(c.isascii() for c in p) / len(p) >= MIN_ASCII


def _fetch(url: str, name: str) -> str:
    """Download a small text file once into data/_raw/ and return its text."""
    raw = DATA / "_raw" / name
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            raw.write_bytes(r.read())
    return raw.read_text(encoding="utf-8")


_MIX_CACHE: dict[str, list[dict]] = {}


def mixture_rows(source: str) -> list[dict]:
    """Single-turn (user, assistant) rows of the Tülu-3 SFT mixture with the given `source` tag.
    Reads the parquet shards straight from the HF cache (downloads ~1.4 GB once)."""
    if source not in _MIX_CACHE:
        import pyarrow.compute as pc
        import pyarrow.dataset as pds
        from huggingface_hub import snapshot_download
        root = snapshot_download(MIXTURE, repo_type="dataset", allow_patterns=["data/*.parquet"])
        ds = pds.dataset(sorted(glob.glob(f"{root}/data/*.parquet")), format="parquet")
        tbl = ds.to_table(filter=pc.field("source") == source, columns=["messages"])
        rows = []
        for msgs in tbl.column("messages").to_pylist():
            if len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
                rows.append({"user": msgs[0]["content"].strip(), "assistant": msgs[1]["content"].strip()})
        _MIX_CACHE[source] = rows
        print(f"  mixture · {source.split('/')[-1]}: {tbl.num_rows} rows, {len(rows)} single-turn")
    return _MIX_CACHE[source]


GOLD_CHARS = 200   # how much of the gold assistant turn is stored per row (the `gold` field)


def _finish(ds: Dataset, label: str, rows: list[dict]) -> list[dict]:
    """dedupe → length/ASCII filter → seeded shuffle → cap → ids. Rows are STORED in the shuffled
    order, so the first n rows of a file are a seeded random subsample. `gold` = opening of the
    gold assistant turn the label came from ("" where the source has none), for auditing."""
    seen, kept, dropped = set(), [], 0
    for r in rows:
        p = r["prompt"].strip()
        k = _norm(p)
        if k in seen or not _keep(p):
            dropped += 1
            continue
        seen.add(k)
        kept.append({**r, "prompt": p})
    kept.sort(key=lambda r: r["prompt"])          # order-independent before the seeded shuffle
    random.Random(SEED).shuffle(kept)
    n_all, kept = len(kept), kept[:MAX_PER_POLE]
    out = [{"id": f"{ds.key}_{label}_{i:04d}", "dataset": ds.key, "label": label, "prompt": r["prompt"],
            "category": r.get("category", ""), "origin": r["origin"],
            "in_training": ds.group == "in_training", "gold": r.get("gold", "")} for i, r in enumerate(kept)]
    print(f"  {ds.key:14s} {label:7s}: {len(rows):6d} source rows → {n_all:5d} after dedupe/filter "
          f"({dropped} dropped) → {len(out):4d} kept")
    return out


# --- builders: each returns (refuse_rows, comply_rows) of {prompt, category, origin} -------
def _build_coconot(categories: tuple, tag: str):
    """CoCoNot rows of the given categories. refuse = original/train, kept only if the prompt is
    in the SFT mixture; comply = the contrast/test + pref/train look-alikes of the same
    categories. With one category the `category` field is the subcategory; with several it is
    'category · subcategory' so the categories stay distinguishable."""
    from datasets import load_dataset
    cat = (lambda ex: ex["subcategory"]) if len(categories) == 1 else (lambda ex: f"{ex['category']} · {ex['subcategory']}")
    orig = load_dataset("allenai/coconot", "original", split="train")
    refuse = [{"prompt": ex["prompt"], "category": cat(ex), "origin": "coconot/original/train",
               "gold": (ex.get("response") or "")[:GOLD_CHARS]}
              for ex in orig if ex["category"] in categories]
    in_mix = {_norm(r["user"]) for r in mixture_rows(MIX_SOURCE["coconot"])}
    n0 = len(refuse)
    refuse = [r for r in refuse if _norm(r["prompt"]) in in_mix]
    print(f"  coconot {tag}: {n0} original/train rows, {len(refuse)} present in the SFT mixture (kept)")
    comply = []
    for cfg, split, origin in [("contrast", "test", "coconot/contrast/test"), ("pref", "train", "coconot/pref/train")]:
        for ex in load_dataset("allenai/coconot", cfg, split=split):
            if ex["category"] in categories:      # contrast has no response; pref's gold = `chosen`
                comply.append({"prompt": ex["prompt"], "category": cat(ex), "origin": origin,
                               "gold": (ex.get("chosen") or ex.get("response") or "")[:GOLD_CHARS]})
    return refuse, comply


HUB = {   # the gated hub copies that carry the real labels (see hub_labels)
    "wildjailbreak": ("allenai/wildjailbreak", "train", "train", dict(delimiter="\t", keep_default_na=False)),
    "wildguardmix": ("allenai/wildguardmix", "wildguardtrain", "train", {}),
}


def hub_labels(key: str) -> dict[str, str]:
    """{normalised prompt: label} from the gated hub copy of an in-training dataset, when this
    account has access (accept the terms on the dataset page; gating is automatic). Empty dict
    otherwise, and the mixture rows then carry no category. WildJailbreak → data_type
    (vanilla/adversarial × harmful/benign); WildGuardMix → prompt_harm_label (+ ' adversarial').
    NB: written before access was granted — untested on the real files."""
    if key not in HUB:
        return {}
    repo, cfg, split, kw = HUB[key]
    try:
        from datasets import load_dataset
        ds = load_dataset(repo, cfg, split=split, **kw)
    except Exception as e:  # noqa: BLE001 — gated / offline: labels are simply not joined
        print(f"  hub labels for {key}: not available ({type(e).__name__}) — accept the terms at "
              f"https://huggingface.co/datasets/{repo} to join them")
        return {}
    out = {}
    for ex in ds:
        if key == "wildjailbreak":
            dt = ex.get("data_type") or ""
            p = ex.get("adversarial") if dt.startswith("adversarial") else ex.get("vanilla")
            if p:
                out[_norm(p)] = dt
        elif ex.get("prompt"):
            out[_norm(ex["prompt"])] = f"{ex.get('prompt_harm_label') or ''}{' adversarial' if ex.get('adversarial') else ''}".strip()
    print(f"  hub labels for {key}: {len(out)} prompts joined")
    return out


def _hub_is_harmful(tag: str) -> bool | None:
    """True/False from a hub tag ('vanilla_harmful', 'unharmful adversarial', …), None if untagged."""
    if not tag:
        return None
    return "unharmful" not in tag and "benign" not in tag


def _build_from_mixture(key: str):
    """Two independent signals must AGREE for a row to enter a pole:
         refuse = hub label harmful  AND the gold turn opens as a refusal (is_refusal)
         comply = hub label benign   AND the gold turn is a clean compliance (is_clean_compliance)
    Disagreements — benign prompts the training data refused, harmful prompts it answered — and
    soft deflections are dropped and counted. Without hub access (gated) only the gold-turn signal
    is available; a warning is printed and the poles are noisier (~10% cross-contamination)."""
    origin = f"tulu-3-sft-mixture/{MIX_SOURCE[key].split('/')[-1]}"
    labels = hub_labels(key)
    if not labels:
        print(f"  !! {key}: no hub labels — poles rest on the gold-turn detectors alone")
    refuse, comply, n = [], [], collections.Counter()
    for r in mixture_rows(MIX_SOURCE[key]):
        a, tag = r["assistant"], labels.get(_norm(r["user"]), "")
        harmful = _hub_is_harmful(tag)
        beh = "refuse" if is_refusal(a) else ("comply" if is_clean_compliance(a) else "ambiguous")
        row = {"prompt": r["user"], "category": tag, "origin": origin, "gold": a[:GOLD_CHARS]}
        if beh == "refuse" and harmful in (True, None):
            refuse.append(row); n["refuse"] += 1
        elif beh == "comply" and harmful in (False, None):
            comply.append(row); n["comply"] += 1
        else:
            n[f"dropped: gold={beh}, hub={'harmful' if harmful else 'benign' if harmful is False else 'untagged'}"] += 1
    print(f"  {key}: " + ", ".join(f"{k} {v}" for k, v in n.most_common()))
    return refuse, comply


def _build_hh_rlhf():
    """Anthropic HH-RLHF: harmless-base → refuse, helpful-base → comply. Single-turn dialogues only;
    gold = `chosen`, the preferred assistant turn; the behaviour detectors decide the pole."""
    from datasets import load_dataset
    out = {"refuse": [], "comply": []}
    for sub, lab in (("harmless-base", "refuse"), ("helpful-base", "comply")):
        for ex in load_dataset("Anthropic/hh-rlhf", data_dir=sub, split="train"):
            t = ex["chosen"]
            if t.count("\n\nHuman:") != 1 or "\n\nAssistant:" not in t:
                continue
            h, a = t.split("\n\nAssistant:", 1)
            p, a = h.replace("\n\nHuman:", "").strip(), a.strip()
            if (is_refusal(a) if lab == "refuse" else is_clean_compliance(a)):
                out[lab].append({"prompt": p, "category": sub, "origin": f"Anthropic/hh-rlhf/{sub}/train",
                                 "gold": a[:GOLD_CHARS]})
    return out["refuse"], out["comply"]


def _build_llm_lat():
    """LLM-LAT: harmful-dataset (prompt, chosen=refusal, rejected=harmful answer) → refuse;
    benign-dataset (prompt, response) → comply. Every 8th benign row is enough (165k)."""
    from datasets import load_dataset
    h = load_dataset("LLM-LAT/harmful-dataset", split="train")
    b = load_dataset("LLM-LAT/benign-dataset", split="train")
    b = b.select(range(0, len(b), 8))
    refuse = [{"prompt": x["prompt"], "category": "harmful", "origin": "LLM-LAT/harmful-dataset",
               "gold": x["chosen"][:GOLD_CHARS]} for x in h if is_refusal(x["chosen"])]
    comply = [{"prompt": x["prompt"], "category": "benign", "origin": "LLM-LAT/benign-dataset",
               "gold": x["response"][:GOLD_CHARS]} for x in b if is_clean_compliance(x["response"])]
    return refuse, comply


NEMOTRON = "nvidia/Llama-Nemotron-Post-Training-Dataset"


def _build_nemotron():
    """Llama-Nemotron SFT data: the safety split → refuse, the chat split → comply. Single-turn rows
    with reasoning off (each row exists twice, with and without a <think> block); gold = `output`."""
    from huggingface_hub import hf_hub_download
    out = {"refuse": [], "comply": []}
    for fn, lab in (("SFT/safety/safety.jsonl", "refuse"), ("SFT/chat/chat.jsonl", "comply")):
        split = fn.split("/")[1]
        for line in open(hf_hub_download(NEMOTRON, fn, repo_type="dataset")):
            r = json.loads(line)
            inp = r.get("input")
            if r.get("reasoning") != "off" or not isinstance(inp, list) or len(inp) != 1 or inp[0].get("role") != "user":
                continue
            a = re.sub(r"^\s*<think>\s*</think>\s*", "", str(r.get("output", ""))).strip()
            if (is_refusal(a) if lab == "refuse" else is_clean_compliance(a)):
                out[lab].append({"prompt": inp[0]["content"].strip(), "category": split,
                                 "origin": f"Llama-Nemotron-Post-Training-Dataset/SFT/{split}", "gold": a[:GOLD_CHARS]})
    return out["refuse"], out["comply"]


CAA_BASE = "https://raw.githubusercontent.com/nrimsky/CAA/main/datasets/"


def _build_caa_refusal():
    """CAA refusal-behaviour A/B items: '<question>\n\nChoices:\n(A) …\n(B) …' with the letter of the
    answer matching the behaviour. Both poles get the same prompt; gold = the matching answer for
    `refuse`, the other answer for `comply`. Labels are the dataset's own — no detector."""
    refuse, comply = [], []
    for split, fn in (("generate", "generate/refusal/generate_dataset.json"), ("test_ab", "test/refusal/test_dataset_ab.json")):
        for x in json.loads(_fetch(CAA_BASE + fn, f"caa_refusal_{split}.json")):
            q = x["question"]
            if "\n\nChoices:\n" not in q:
                continue
            stem, choices = q.split("\n\nChoices:\n", 1)
            opts = {l.strip()[:3]: l.strip()[3:].strip() for l in choices.strip().split("\n") if l.strip().startswith("(")}
            m, n = x["answer_matching_behavior"].strip()[:3], x["answer_not_matching_behavior"].strip()[:3]
            if m not in opts or n not in opts:
                continue
            origin = f"nrimsky/CAA/datasets/{split}/refusal"
            refuse.append({"prompt": stem.strip(), "category": split, "origin": origin, "gold": opts[m][:GOLD_CHARS]})
            comply.append({"prompt": stem.strip(), "category": split, "origin": origin, "gold": opts[n][:GOLD_CHARS]})
    return refuse, comply


def _build_advbench():
    rows = list(csv.DictReader(io.StringIO(_fetch(ADVBENCH_CSV, "advbench_harmful_behaviors.csv"))))
    return [{"prompt": r["goal"], "category": "", "origin": "llm-attacks/advbench"} for r in rows], []


def _build_sorrybench():
    from huggingface_hub import hf_hub_download
    names = re.findall(r'"([^"]+)"', Path(hf_hub_download(SORRYBENCH_REPO, "meta_info.py", repo_type="dataset")).read_text())
    rows = [json.loads(l) for l in open(hf_hub_download(SORRYBENCH_REPO, "question.jsonl", repo_type="dataset"))]
    refuse = [{"prompt": r["turns"][0], "category": names[int(r["category"]) - 1], "origin": SORRYBENCH_REPO.split("/")[-1]}
              for r in rows if r.get("prompt_style") == "base"]
    return refuse, []


def _build_xstest():
    rows = list(csv.DictReader(io.StringIO(_fetch(XSTEST_CSV, "xstest_prompts.csv"))))
    mk = lambda r: {"prompt": r["prompt"], "category": r["type"], "origin": "paul-rottger/xstest"}
    return ([mk(r) for r in rows if r["label"] == "unsafe"],
            [mk(r) for r in rows if r["label"] == "safe"])


_BUILDERS = {
    "coconot_safety": lambda: _build_coconot((COCONOT_SAFETY,), "safety"),
    # "coconot_other": lambda: _build_coconot(COCONOT_OTHER, "other"),   # off — see the note in DATASETS
    "wildjailbreak": lambda: _build_from_mixture("wildjailbreak"),
    "wildguardmix": lambda: _build_from_mixture("wildguardmix"),
    "llm_lat": _build_llm_lat,
    "hh_rlhf": _build_hh_rlhf,
    "nemotron": _build_nemotron,
    "caa_refusal": _build_caa_refusal,
    "advbench": _build_advbench,
    "sorrybench": _build_sorrybench,
    "xstest": _build_xstest,
}


def build(key: str, force: bool = False) -> list[dict]:
    ds = all_datasets()[key]
    if ds.file.exists() and not force:
        return load(key)
    print(f"building {key} …")
    refuse, comply = _BUILDERS[key]()
    rows = _finish(ds, "refuse", refuse) + (_finish(ds, "comply", comply) if comply else [])
    DATA.mkdir(parents=True, exist_ok=True)
    with ds.file.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def build_all(force: bool = False) -> dict[str, list[dict]]:
    """Builds the seven training sets and the eval sets."""
    return {k: build(k, force) for k in all_datasets()}


# --- readers ----------------------------------------------------------------------------
def load(key: str, label: str | None = None) -> list[dict]:
    rows = [json.loads(l) for l in all_datasets()[key].file.open()]
    return [r for r in rows if r["label"] == label] if label else rows


def prompts(key: str, label: str, n: int | None = None) -> list[str]:
    """The first n `label` prompts of a dataset — a seeded random subsample (each pole is stored
    shuffled; a file holds the refuse block, then the comply block)."""
    out = [r["prompt"] for r in load(key, label)]
    return out[:n] if n else out


def summary():
    """One row per dataset: counts per pole and prompt-length medians (pandas DataFrame)."""
    import pandas as pd
    recs = []
    for ds in all_datasets().values():
        rows = load(ds.key) if ds.file.exists() else []
        rec = {"dataset": ds.key, "group": ds.group, "in_training": ds.group == "in_training"}
        for lab in ("refuse", "comply"):
            sub = [r for r in rows if r["label"] == lab]
            rec[f"n_{lab}"] = len(sub)
            rec[f"chars_{lab}"] = int(pd.Series([len(r["prompt"]) for r in sub]).median()) if sub else None
        rec["categories"] = len({r["category"] for r in rows if r["category"]})
        recs.append(rec)
    return pd.DataFrame(recs).set_index("dataset")


def models_table():
    import pandas as pd
    return pd.DataFrame([asdict(m) for m in MODELS.values()]).set_index("key")


def datasets_table():
    import pandas as pd
    return pd.DataFrame([asdict(d) for d in all_datasets().values()]).set_index("key")


if __name__ == "__main__":
    build_all()
    print(summary().to_string())
