"""Run the Imperial MIA benchmark (Meeus et al. SoK) on local member / non-member JSON files.

Thin wrapper: the attacks are the benchmark's own code (third_party/mia_llms_benchmark); this file
only (1) feeds local data instead of HF-hub datasets, (2) loads models in bf16 straight onto the GPU
with float32 logits, (3) shims two transformers>=5 API removals, (4) resumes from a saved scores.json,
(5) never holds two 8B models at once: the target is freed before `ratio` loads its reference (the
ratio attack only needs the reference pass plus the target losses already in scores.json).

  systemd-run --user --scope -p MemoryMax=100G -p MemoryHigh=85G $PY run_attacks.py \
      --config configs/tulu_coconot.yaml --data data/matched_split --out results/tulu_coconot_matched \
      --attacks bag_of_words,loss,zlib,lowercase,minkprob,minkplusplus,recall,conrecall --resume
  ... then the same with --attacks ratio --resume   (target model never loaded)
"""
import argparse, gc, json, logging, os, sys, time, types
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = Path(__file__).parent; BENCH = HERE / "third_party" / "mia_llms_benchmark"; sys.path.insert(0, str(BENCH))
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--data", default=str(HERE / "data")); ap.add_argument("--out", required=True)
ap.add_argument("--attacks", default="bag_of_words,loss,zlib,lowercase,minkprob,minkplusplus,recall,conrecall,ratio")
ap.add_argument("--views", default="coconot_chat,coconot_resp"); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--resume", action="store_true", help="load <out>/scores.json and skip attacks already scored")
args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

import numpy as np, torch
import datasets; datasets.disable_caching()
# Dataset.map() fingerprints an un-annotated transform by dill-pickling it; the benchmark's lambdas close over the
# attack object, i.e. the 8B model, so that pickle costs minutes and ~60 GB and stalls at the memory cap. Random
# fingerprints instead: all datasets here are in-memory, nothing is cached across runs.
import datasets.fingerprint as _F, datasets.arrow_dataset as _AD
_AD.update_fingerprint = _F.update_fingerprint = lambda *a, **k: _F.generate_random_fingerprint()
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from attacks.utils import batch_nlloss
from utils import load_attack, load_config, set_seed
from main import results_with_bootstrapping, get_header
from tabulate import tabulate
import attacks.recall as R, attacks.conrecall as C, attacks.ratio as RA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
set_seed(args.seed); config = load_config(args.config); g = config["global"]; device = g["device"]
META = {"label", "subcategory", "nlloss"}

def _bep(self, texts, **kw): return self(texts, **kw)          # transformers>=5 removed batch_encode_plus
def _float_logits(m):                                          # bf16 losses cannot go to numpy; float32 softmax for Min-K%++
    _orig = m.forward
    def _fwd(*a, **k):
        o = _orig(*a, **k); o.logits = o.logits.float(); return o
    m.forward = _fwd; return m
def load(model_id):
    logging.info(f"loading {model_id} -> {device}")
    m = _float_logits(AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device).eval())
    t = AutoTokenizer.from_pretrained(model_id); t.pad_token = t.eos_token; t.padding_side = "right"
    for cls in {type(t), PreTrainedTokenizerBase}:
        if not hasattr(cls, "batch_encode_plus"): cls.batch_encode_plus = _bep
    return m, t
def free(m):
    del m; gc.collect(); torch.cuda.empty_cache()

_target = {}
def target():
    if not _target: _target["m"], _target["t"] = load(g["target_model"])
    return _target["m"], _target["t"]
RA.RatioAttack._load_reference = lambda self, device: load(self.config["reference_model_path"])

# per-sample scores live in <out>/scores.json: {view: {label, subcategory, nlloss, <attack>: [...]}} (plain lists,
# small enough to commit; the analysis reads this file)
saved = json.load(open(out / "scores.json")) if args.resume and (out / "scores.json").exists() else {}
def save(): json.dump(saved, open(out / "scores.json", "w"))
attacks = args.attacks.split(",")
if "ratio" in attacks and len(attacks) > 1:
    sys.exit("run `ratio` in its own invocation (after the others, with --resume) so two models are never resident")

tables = {}
for view in args.views.split(","):
    rows = json.load(open(Path(args.data) / f"{view}.json"))
    aux = Dataset.from_list(json.load(open(Path(args.data) / f"aux_{view.split('_')[1]}.json")))
    R.load_dataset = lambda name, split, _a=aux: _a; C.load_dataset = lambda name, split, _a=aux: _a
    ds = Dataset.from_list(rows); prev = saved.get(view, {})
    if "nlloss" in prev and len(prev["nlloss"]) == len(ds):
        ds = ds.add_column("nlloss", list(prev["nlloss"])); logging.info(f"[{view}] base nlloss restored from scores.json")
    else:
        m, t = target(); t0 = time.time()
        ds = ds.map(lambda x: batch_nlloss(x, m, t, device), batched=True, batch_size=g["batch_size"])
        logging.info(f"[{view}] base nlloss done in {time.time()-t0:.0f}s; n={len(ds)}")
    y_true = ds["label"]; saved.setdefault(view, {}).update({"label": list(y_true), "subcategory": list(ds["subcategory"]), "nlloss": list(ds["nlloss"])})
    save()   # persist the base pass immediately
    for name in attacks:
        if name in prev and len(prev[name]) == len(ds):
            logging.info(f"[{view}] {name}: already scored, skipping"); continue
        t0 = time.time()
        if name == "ratio":
            if _target: free(_target.pop("m")); _target.clear()
            stand_in = types.SimpleNamespace(device=torch.device(device))          # RatioAttack.run never touches the target
            attack = load_attack(name, stand_in, None, config[name])
        else:
            m, t = target(); attack = load_attack(name, m, t, config[name])
        ds = attack.run(ds); y = list(ds[name]); saved[view][name] = y
        res = results_with_bootstrapping(y_true, y, fpr_thresholds=g["fpr_thresholds"], n_bootstraps=g["n_bootstrap_samples"])
        logging.info(f"[{view}] {name}: AUC {res[0]} | {time.time()-t0:.0f}s")
        if name == "ratio": free(attack.reference_model); attack.reference_model = None
        save()
    table = [[k] + results_with_bootstrapping(y_true, saved[view][k], fpr_thresholds=g["fpr_thresholds"], n_bootstraps=g["n_bootstrap_samples"])
             for k in saved[view] if k not in META]
    tables[view] = tabulate(table, headers=get_header(g), tablefmt="outline")
for k, v in tables.items(): print(f"\nDataset: {k}\n{v}")
open(out / "summary.txt", "w").write("\n\n".join(f"Dataset: {k}\n{v}" for k, v in tables.items()))
print("DONE")
