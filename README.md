# Genesis Steering

Code and stored results for *Genesis Steering: Better Activation Steering Through Training Data*.

The paper studies which contrastive data make a good refusal steering vector. It compares vectors
built from the safety datasets in a model's own fine-tuning mixture (its *genesis data*) against
vectors built from safety datasets the model was never trained on, using Tülu-3-8B and
Llama-3.1-8B-Instruct as a control, and tests whether membership inference can identify the genesis
data inside a larger candidate pool when the mixture is not public.

This repository contains the experiment code, the result files the paper's numbers are read from,
and the scripts that turn those results into the manuscript's macros, tables and figures.

## Repository layout

```
1_models_and_datasets/   models, safety datasets, unsteered reference run
2_genesis_steering/      poles, steering vectors, the dose grid, selection geometry
3_membership_inference/  membership scores, their agreement, and what they select
judge.py                 gemma-2-27b-it judge, shared by parts 2 and 3
paper/                   generators for the paper's numbers, tables and figures
```

Each part holds its scripts and a `results/` directory. The manuscript itself is on arXiv.

## Requirements

Regenerating the paper's tables and figures requires only:

```
numpy  scipy  scikit-learn  matplotlib
```

Rerunning the experiments additionally requires `torch`, `transformers`, `datasets`,
`huggingface_hub`, `pandas` and `pyarrow`, a GPU, and the model weights.

## Reproducing the tables and figures

```bash
cd paper
python build_numbers.py     # numbers.tex (658 macros) and tables/ (26 files)
python build_figures.py     # figures/*.pdf
```

This reads the stored result files and takes under a minute on a CPU. No model weights or datasets
are downloaded.

Every numerical value in the manuscript is a `\D{key}` macro written by `build_numbers.py`, and
every table body is generated into `tables/`. Both are committed, so the generated files can be
compared against the published PDF without running anything.

Three of the 29 committed table files are exceptions, because the data behind them is not released
(see [Data](#data)). `tab_flip_examples.tex` quotes model responses from the completion dumps; the
doses behind it are stored in `2_genesis_steering/results/selection_steer/flip_example_doses.json`.
`tab_datasets.tex` and `tab_examples.tex` are counts and one example pair per dataset, taken from an
item panel that repeats each dataset's prompts and gold answers. All three are committed as
generated, and `build_numbers.py` leaves them untouched when their sources are absent.

## Source of each result

| manuscript | result file | produced by |
|---|---|---|
| Table 1, Table 3, Tables 17–29, Figures 6–8 | `2_genesis_steering/results/selection_steer/*.json` | `selection_steer.py`, `layer_rerun.py`, `judge.py` |
| Figure 1, Tables 6–9, 12, 14 | `3_membership_inference/results/mia_agreement/summary.json`, `mia_accuracy.json` | `mia_agreement.py`, `mia_accuracy.py`, `mia_panel.py` |
| Table 2 | `3_membership_inference/results/mia_agreement/summary.json` | `mia_agreement.py` |
| Table 11 | `2_genesis_steering/results/selection_steer/top_overlap.json` | `selection_steer.py` |
| Table 10, Figure 3 | `3_membership_inference/results/mia_features/summary.json` | `mia_features.py` |
| Figure 4, Tables 15–16 | `2_genesis_steering/results/mia_geometry/geometry*.json` | `mia_geometry.py` |
| Table 29 | `2_genesis_steering/results/layer16/vectors_*.npz` | `layer_rerun.py` |
| Within-source control (Appendix E.2) | `3_membership_inference/results/mia_quartile/wildguardmix/behave_judge.jsonl` | `mia_quartile.py`, `judge.py` |
| WildGuardMix AUCs (§4.2) | `3_membership_inference/results/mia_accuracy.json` | `mia_accuracy.py` |
| Table 4, Table 5 | item panel, not released | `mia_panel.py` |

The manuscript has 30 tables and 8 figures; the rows group them by the experiment behind them.

## Rerunning the experiments

The commands below are the pipeline that produced the stored results. Two inputs they need are not
in this repository, so the pipeline is not self-contained:

- the benchmark prompt sets under `2_genesis_steering/data/evals/` (HarmBench, StrongREJECT,
  XSTest-safe, OR-Bench) and `data/evals_heldout/`, which come from their own releases;
- the vendored MIA benchmark and its configs that `run_attacks.py` wraps, which lives upstream at
  `mia_llms_benchmark`; `run_attacks.py` also requires `--config`;
- the per-example membership scores under `3_membership_inference/results/mia_rank/`, which a few
  analysis paths read and which `run_attacks.py` produces.

Each script is resume-safe and writes into its part's `results/`.

```bash
# datasets and the unsteered reference run
cd 1_models_and_datasets
python -c "import registry; registry.build_all()"

# poles, vectors, the dose grid, selection geometry
cd ../2_genesis_steering
python poles.py                 # refusal and compliance poles, per dataset, from the gold response
python selection_steer.py       # the dose grid over all arms
python layer_rerun.py           # the same grid at the input of block 16
python mia_geometry.py          # how a selection shifts each pole

# membership scores
cd ../3_membership_inference
python build_sets.py && python build_datasets.py
python run_attacks.py --config <config> --out results/<name>   # the 10 membership scores
python mia_panel.py && python mia_agreement.py && python mia_features.py
```

The judge is `gemma-2-27b-it` (`judge.py`) and requires roughly 55 GB of GPU memory. It should be
run on its own device.

## Conventions

Two conventions are load-bearing and easy to break.

**Steering site and dose scale.** Vectors are added at the input of block 16. Doses are reported on
a nominal scale anchored at block 13, which earlier runs used: the dose actually added is the
nominal value times the ratio of median activation norms at the two blocks, 1.360 on Tülu-3-8B and
1.140 on Llama-3.1-8B-Instruct. Appendix E.4 of the manuscript gives the details.

**Pole construction.** A dataset's vector is the mean activation of its own refusal examples minus
the mean of its own compliance examples. No compliance pool is shared across datasets, and an
example's pole is decided by its gold response rather than by the model's.

## Data

No safety dataset is redistributed here. `registry.py` rebuilds the files under
`1_models_and_datasets/data/` from their public sources: CoCoNot, WildJailbreak, WildGuardMix,
HH-RLHF, LLM-LAT, Nemotron, AdvBench, SORRY-Bench, XSTest and the CAA splits. PKU-SafeRLHF and
Do-Not-Answer are built by `2_genesis_steering/poles.py` and
`3_membership_inference/build_datasets.py`. All sampling is seeded, so the rebuilt files match the
ones the results were computed on.

Prompt text is not included either. The within-source control needs to know which WildGuardMix rows
fall in which selection, so `3_membership_inference/data/mia_quartile/wildguardmix/sets/` and the
judge file beside it identify each row by the SHA-256 of its prompt. The control's numbers recompute
from the hashes alone, and the prompts can be recovered by hashing a local copy of WildGuardMix.
Note that `mia_quartile.py` writes prompts rather than hashes when rerun.

Model responses to harmful prompts, and the judge files that repeat those prompts, are not released.
Appendix F of the manuscript shows one flipped example per benchmark, truncated after the first
sentence.

Each source dataset carries its own licence and terms, which govern anything rebuilt from them.

## Citation

The arXiv identifier will be added here once the preprint is announced.

```bibtex
@misc{bodin2026genesis,
  title  = {Genesis Steering: Better Activation Steering Through Training Data},
  author = {Bodin, Tora and Aubinais, Eric and Torre, Ilaria and Cheng, Chih-Hong},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## Funding

Funded by the European Union. Views and opinions expressed are however those of the author(s) only
and do not necessarily reflect those of the European Union or the European Health and Digital
Executive Agency (HADEA). Neither the European Union nor the granting authority can be held
responsible for them. RobustifAI project, ID 101212818.

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program
(WASP) funded by the Knut and Alice Wallenberg Foundation.
