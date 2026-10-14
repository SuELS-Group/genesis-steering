"""Figures of the experiment sections, read from the same result files as build_numbers.py. CPU only.

    fig_scale_auc.pdf  (a) where each dataset sits on the scores' scale; (b, c) separating SFT examples from the rest (part 3)
                       (fig_scale / fig_auc: the two earlier separate figures, no longer built by main())
    fig_geom.pdf    top / bottom thirds against the whole pool, and the shift of each pole (part 2)
    fig_dose_sel_<model>.pdf  dose response of the selected vectors against random draws and the whole pool (selection_steer)
    fig_dose_ref.pdf          dose response of the reference vectors, both models (selection_steer + layer-16 grid)

    python build_figures.py        # writes figures/fig_*.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import build_numbers as B  # noqa: E402  (sources, labels, dataset order)

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
TEXTWIDTH = 6.3                                  # inches, ACL two-column text width
# purple / yellow-orange: passes the dataviz validator's checks on white (CVD dE 31.1, normal dE 33.3, contrast 7.4 / 3.1:1);
# lighter oranges (#d98200 and up) fail the 3:1 contrast floor
MODEL = {"tulu": ("Tülu-3-8B", "#6b3fa0", "o"), "meta": ("Llama-3.1-8B-Instruct", "#d47d00", "s")}
INK, MUTED, GRID = "#1d2329", "#6b7580", "#e3e6e9"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0, "pdf.fonttype": 42})
G = B.G
PLAIN = {"Min-K\\%": "Min-K%", "Min-K\\%++": "Min-K%++", "$s_{\\text{base}}$": r"$s_{\mathrm{base}}$", "$s_{\\text{twin}}$": r"$s_{\mathrm{twin}}$"}
lab = lambda m: PLAIN.get(B.LABEL.get(m, m), B.LABEL.get(m, m))
DSNAME = {d: n.replace("$^\\dagger$", "†") for d, n in B.DS_ORDER}


def style(ax, xgrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if xgrid:
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)


def legend_models(fig, extra=(), y=1.0):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, marker=mk, ls="", ms=5, label=n) for n, c, mk in MODEL.values()] + list(extra)
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, y), handletextpad=0.3, columnspacing=1.4)


# ---- figure 1: dataset scale -----------------------------------------------------------------------------
def fig_scale():
    """Per dataset: the range over scores of its mean (s.d. of the pooled refusal pole), the median over scores, and
    s_base on its own (the score that orders CAA differently)."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.49, 2.6))
    ds = [d for d, _ in B.DS_ORDER][::-1]
    for j, model in enumerate(("tulu", "meta")):
        name, col, mk = MODEL[model]; z = G["all datasets | refuse"][model]["dataset_z"]
        ms = [m for m in AGR_METHODS(model) if m != "s (base)"]
        off = 0.17 if model == "tulu" else -0.17
        for i, d in enumerate(ds):
            v = [z[d][m][0] for m in ms]
            ax.plot([min(v), max(v)], [i + off] * 2, color=col, lw=1.6, alpha=0.45, solid_capstyle="round")
            ax.plot(np.median(v), i + off, mk, color=col, ms=4.5, mec="white", mew=0.6, zorder=3)
            ax.plot(z[d]["s (base)"][0], i + off, mk, mfc="white", mec=col, ms=4, mew=0.9, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.tick_params(axis="y", length=0); ax.set_yticks(range(len(ds))); ax.set_yticklabels([DSNAME[d] for d in ds])
    ax.set_xlabel("dataset mean, s.d. of the pooled refusal pole")
    style(ax)
    extra = [Line2D([], [], color=MUTED, lw=1.6, alpha=0.6, label="range over scores"),
             Line2D([], [], color=MUTED, marker="o", ls="", ms=4.5, label="median"),
             Line2D([], [], color=MUTED, marker="o", mfc="white", ls="", ms=4, label=r"$s_{\mathrm{base}}$")]
    legend_models(fig, y=1.10)
    fig.legend(handles=extra, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=6.5,
               handlelength=1.4, columnspacing=1.2)
    fig.tight_layout(); fig.savefig(FIG / "fig_scale.pdf", bbox_inches="tight"); plt.close(fig)


def AGR_METHODS(model):
    return B.AGR["methods"][model]


# ---- figure 2: AUC ------------------------------------------------------------------------------------------
def fig_auc():
    """AUC +- s.e. of each score for SFT examples against the rest; length (no model) as a reference line."""
    panels = [("all datasets | refuse", "refusal pole, pooled"), ("all datasets | comply", "compliance pole, pooled")]
    if B.MATCHED:                                       # CoCoNot matched split: out of the paper for now (build_numbers.MATCHED)
        panels.append(("CoCoNot, matched split | mixed", "CoCoNot, matched (one source)"))
    ms = [m for m in B.AUC_METHODS if m != "length (longer)"]
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXTWIDTH * len(panels) / 3, 2.6), sharey=True)
    for ax, (g, title) in zip(axes, panels):
        for model in ("tulu", "meta"):
            name, col, mk = MODEL[model]; a = G[g][model]["member"]["auc"]
            off = 0.15 if model == "tulu" else -0.15
            for i, m in enumerate(ms[::-1]):
                if m not in a:
                    continue
                ax.errorbar(a[m][0], i + off, xerr=a[m][1], fmt=mk, color=col, ms=4, mec="white", mew=0.5, elinewidth=0.8, capsize=0)
        L = G[g]["tulu"]["member"]["auc"]["length (longer)"][0]
        ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 2)))
        ax.axvline(L, color=INK, lw=0.8, alpha=0.6)
        ax.text(L + 0.006, -0.85, "length", fontsize=6.5, color=INK, ha="left", va="center")
        ax.tick_params(axis="y", length=0); ax.set_ylim(-1.2, len(ms) - 0.5)
        ax.set_title(title, color=INK); ax.set_xlim(0.45, 1.0); style(ax)
    axes[0].set_yticks(range(len(ms))); axes[0].set_yticklabels([lab(m) for m in ms[::-1]])
    fig.supxlabel("AUC, examples in Tülu's SFT mixture against the rest", fontsize=plt.rcParams["axes.labelsize"], y=0.02)
    legend_models(fig, y=1.07)
    fig.tight_layout(); fig.savefig(FIG / "fig_auc.pdf", bbox_inches="tight"); plt.close(fig)


DSABBR = {"WildGuardMix": "WGM†", "WildJailbreak": "WJB†", "CoCoNot": "CCN†", "CoCoNot, left out": "CCN-out", "Do-Not-Answer": "DNA",
          "HH-RLHF": "HH", "PKU-SafeRLHF": "PKU", "CAA, answers (mean)": "CAA"}     # panel (a) of fig_scale_auc; spelled out in its caption


# ---- figures 2 + 3 merged : dataset scale and AUC in one full-width row -------------------
def fig_scale_auc():
    """(a) fig_scale's panel, both poles side by side: per dataset, the range over scores of its mean (in s.d. of the pooled
    pole), the median, and s_base apart; (b, c) fig_auc's panels: AUC +- s.e. per score on the pooled refusal and compliance
    poles."""
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(TEXTWIDTH, 2.05))                      # 2.05 in (was 3.0): shorter for page space, 
    outer = fig.add_gridspec(1, 2, width_ratios=[1.75, 2], wspace=0.30)
    ina = outer[0].subgridspec(1, 2, wspace=0.06)
    a0 = fig.add_subplot(ina[0]); a0c = fig.add_subplot(ina[1], sharey=a0)
    inner = outer[1].subgridspec(1, 2, wspace=0.05)
    a1 = fig.add_subplot(inner[0]); a2 = fig.add_subplot(inner[1], sharey=a1)
    # (a) dataset scale, refusal and compliance pole (the left-out CoCoNot rows have no compliance pole)
    ds = [d for d, _ in B.DS_ORDER][::-1]
    for ax, pole, title in ((a0, "refuse", "(a) refusals"), (a0c, "comply", "compliances")):
        for model in ("tulu", "meta"):
            name, col, mk = MODEL[model]; z = G[f"all datasets | {pole}"][model]["dataset_z"]
            ms_ = [m for m in AGR_METHODS(model) if m != "s (base)"]
            off = 0.25 if model == "tulu" else -0.25                 # (a): more room between the two models' rows
            for i, d in enumerate(ds):
                if d not in z:
                    continue
                v = [z[d][m][0] for m in ms_]
                ax.plot([min(v), max(v)], [i + off] * 2, color=col, lw=1.5, alpha=0.5, solid_capstyle="round")
                ax.plot(np.median(v), i + off, mk, color=col, ms=4.5, mec="white", mew=0.6, zorder=3)
                ax.plot(z[d]["s (base)"][0], i + off, mk, mfc="white", mec=col, ms=4, mew=0.9, zorder=3)
        ax.axvline(0, color=MUTED, lw=0.6, ls=(0, (2, 2))); ax.tick_params(axis="y", length=0)
        ax.set_title(title, color=INK, pad=3); style(ax)
    a0.set_yticks(range(len(ds))); a0.set_yticklabels([DSABBR[d] for d in ds]); a0c.tick_params(axis="y", labelleft=False)
    lo = min(a0.dataLim.x0, a0c.dataLim.x0) - 0.05; hi = max(a0.dataLim.x1, a0c.dataLim.x1) + 0.05   # the data, not matplotlib's padded limits
    lo, hi = min(lo, -2.1), max(hi, 1.5)                             # -2 .. 1.5 holds every point (lowest: Llama's CAA compliances, -1.9)
    for ax in (a0, a0c):
        ax.set_xlim(lo, hi); ax.set_xticks([-2, -1, 0, 1])
    fig.text((a0.get_position().x0 + a0c.get_position().x1) / 2, 0.012, "dataset mean, s.d. of the pooled pole",
             ha="center", va="top", fontsize=plt.rcParams["axes.labelsize"], color=INK)

    # (b, c) AUC
    ms = [m for m in B.AUC_METHODS if m != "length (longer)"]
    for ax, (g, title) in zip((a1, a2), (("all datasets | refuse", "(b) AUC, refusals"), ("all datasets | comply", "(c) AUC, compliances"))):
        for model in ("tulu", "meta"):
            name, col, mk = MODEL[model]; a = G[g][model]["member"]["auc"]
            off = 0.22 if model == "tulu" else -0.22                 # (b, c): the two models' points further apart
            for i, m in enumerate(ms[::-1]):
                if m in a:
                    ax.errorbar(a[m][0], i + off, xerr=a[m][1], fmt=mk, color=col, ms=4, mec="white", mew=0.5, elinewidth=0.8, capsize=0)
        L = G[g]["tulu"]["member"]["auc"]["length (longer)"][0]
        ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 2))); ax.axvline(L, color=INK, lw=0.8, alpha=0.6)
        ax.text(L + 0.006, -0.85, "length", fontsize=6.5, color=INK, ha="left", va="center")
        ax.tick_params(axis="y", length=0); ax.set_ylim(-1.2, len(ms) - 0.5); ax.set_xlim(0.47, 1.005)
        ax.set_title(title, color=INK, pad=3); style(ax)
    a1.set_yticks(range(len(ms))); a1.set_yticklabels([lab(m) for m in ms[::-1]]); a2.tick_params(axis="y", labelleft=False)
    fig.text((a1.get_position().x0 + a2.get_position().x1) / 2, 0.012, "AUC, examples in Tülu's SFT mixture against the rest",
             ha="center", va="top", fontsize=plt.rcParams["axes.labelsize"], color=INK)
    extra = [Line2D([], [], color=MUTED, lw=1.6, alpha=0.6, label="(a) range over scores"), Line2D([], [], color=MUTED, marker="o", ls="", ms=4.5, label="median"),
             Line2D([], [], color=MUTED, marker="o", mfc="white", ls="", ms=4, label=r"$s_{\mathrm{base}}$")]
    legend_models(fig, extra=extra, y=1.035)                         # legend, titles and x labels closer to the panels 
    fig.savefig(FIG / "fig_scale_auc.pdf", bbox_inches="tight"); plt.close(fig)


# ---- figure 3: selection geometry ---------------------------------------------------------------------------
def fig_geom():
    """(a) cosine of the top / bottom third's vector to the whole pool, per score (mean over datasets), with the
    random-third band; (b) shift of the refusal pole against the shift of the compliance pole, per score and dataset."""
    from matplotlib.lines import Line2D
    key = f"prompt/L{B.LAYER}"
    scores = [s for s in B.GEO_SCORES]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7), gridspec_kw={"width_ratios": [1, 1.05]})
    los = []                                       # each model's random-third 2.5 % quantile (mean over datasets)
    for model in ("tulu", "meta"):
        name, col, mk = MODEL[model]; o = B.GEO[model]["sites"][key]
        off = 0.15 if model == "tulu" else -0.15
        for i, s in enumerate(scores[::-1]):
            if s not in o["per_score"]:
                continue
            P = o["per_score"][s]["per_dataset"]
            ct, cb = np.mean([v["cos_top_full"] for v in P.values()]), np.mean([v["cos_bottom_full"] for v in P.values()])
            a1.plot([ct, cb], [i + off] * 2, color=col, lw=0.8, alpha=0.5)
            a1.plot(ct, i + off, mk, color=col, ms=4.5, mec="white", mew=0.5, zorder=3)
            a1.plot(cb, i + off, mk, mfc="white", mec=col, ms=4.2, mew=0.9, zorder=3)
            mr = float(np.mean([v["shift_refuse_on_axis"] for v in P.values()]))          # mean over datasets, per pole
            mc = float(np.mean([v["shift_comply_on_axis"] for v in P.values()]))
            a2.plot([mr, mc], [i + off] * 2, color=col, lw=0.8, alpha=0.5)
            a2.plot(mr, i + off, mk, color=col, ms=4.5, mec="white", mew=0.5, zorder=3)
            a2.plot(mc, i + off, mk, mfc="white", mec=col, ms=4.2, mew=0.9, zorder=3)
        los.append(np.mean([v["cos_full"][0] for v in o["random"].values()]))
    a1.axvspan(min(los), 1.0, color=MUTED, alpha=0.15, lw=0)
    a1.tick_params(axis="y", length=0); a1.set_yticks(range(len(scores))); a1.set_yticklabels([PLAIN.get(B.GLABEL[s], B.GLABEL[s]) for s in scores[::-1]])
    a1.set_xlabel("cosine to the whole pool's vector"); a1.set_title("(a) selected thirds against the pool", color=INK)
    a1.legend(handles=[Line2D([], [], color=MUTED, marker="o", ls="", ms=4.5, label="top third"),
                       Line2D([], [], color=MUTED, marker="o", mfc="white", ls="", ms=4.2, label="bottom third"),
                       plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.25, label="random third (95%, both models)")],
              loc="lower left", frameon=False, fontsize=6.5)
    style(a1)
    a2.axvline(0, color=INK, lw=0.8, alpha=0.5)                      # a mean-preserving selection would put both poles here
    a2.set_ylim(a1.get_ylim()); a2.set_yticks(range(len(scores))); a2.tick_params(axis="y", length=0, labelleft=False)
    a2.set_xlabel("shift of the pole, top minus bottom third (s.d.)")
    a2.set_title("(b) how far each pole moves", color=INK)
    a2.legend(handles=[Line2D([], [], color=MUTED, marker="o", ls="", ms=4.5, label="refusal pole"),
                       Line2D([], [], color=MUTED, marker="o", mfc="white", ls="", ms=4.2, label="compliance pole")],
              loc="lower left", frameon=False, fontsize=6.5)
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)
    legend_models(fig, y=1.07)
    fig.tight_layout(); fig.savefig(FIG / "fig_geom.pdf", bbox_inches="tight"); plt.close(fig)


# ---- appendix figures: steering at every dose (values: the tab_dose tables) -------------------------------------------
STEER_BENCH = [("harmbench", "HarmBench", "complied"), ("strongreject", "StrongREJECT", "complied"), ("xstest_safe", "XSTest (safe)", "refused"),
               ("orbench", "OR-Bench", "refused"), ("alpaca", "Alpaca", "refused")]
KEYM = {"tulu": "tulu_final", "meta": "meta_instruct"}


def _steer():
    return json.load(open(B.P2 / "results" / "selection_steer" / "summary.json"))


def _curve(arm, e, base, tau):
    """(|alpha|, rate) from the unsteered point up to the last admissible dose, on part 2's even doses (the grid every arm has
    for every prompt; the finer doses are generated per prompt only up to its flip, so their per-dose rates are not comparable): a line stops at the first dose whose degenerate share exceeds tau (later doses are not plotted)."""
    xs, ys = [0], [base]
    for d, c in sorted(((float(d), c) for d, c in arm["per_dose"].get(e, {}).items() if float(d) % 2 == 0), key=lambda t: t[0]):
        if c["degen"] > tau:
            break
        xs.append(d); ys.append(c["rate"])
    return xs, ys


def _dose_axes(ax, e, first_col, ylab):
    import steering as T
    import scoring
    harm = e in scoring.HARM_EVALS
    top = max(T.DOSES) if harm else T.BENIGN_MAX_DOSE
    ax.set_xlim(-0.4, top + 0.4); ax.set_xticks([d for d in range(0, top + 1, 4)])   # every 4: the labels crowd at every dose
    ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, 0.5, 1])
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if first_col:
        ax.set_ylabel(ylab)
    else:
        ax.tick_params(axis="y", labelleft=False)


def fig_dose_sel():
    """Per model: rows = selection score, columns = benchmark. Top 128 (solid, filled) and bottom 128 (dashed, open) in the model's
    colour; grey band = range of the random draws; thin ink line = whole pool."""
    import scoring
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    S = _steer(); tau = scoring.TAU
    for model in ("tulu", "meta"):
        name, col, mk = MODEL[model]; A = S[KEYM[model]]["arms"]; base = S[KEYM[model]]["baseline"]
        sel = [(k, lab) for k, lab in (("gap", "Gap"), ("min-kpp", "Min-K%++"), ("consensus", "consensus"), ("stwin", r"$s_{\mathrm{twin}}$")) if f"{k}_top" in A]
        fig, axes = plt.subplots(len(sel), len(STEER_BENCH), figsize=(TEXTWIDTH, 1.65 * len(sel) + 0.6), sharey=True)
        for i, (k, lab_) in enumerate(sel):
            for j, (e, en, what) in enumerate(STEER_BENCH):
                ax = axes[i, j]
                rc = [_curve(A[a_], e, base[e], tau) for a_ in B.rand_arms(A)]
                xs = sorted({x for c in rc for x in c[0]})
                lo = [min(c[1][c[0].index(x)] for c in rc if x in c[0]) for x in xs]; hi = [max(c[1][c[0].index(x)] for c in rc if x in c[0]) for x in xs]
                ax.fill_between(xs, lo, hi, color=MUTED, alpha=0.18, lw=0, step=None)
                px, py = _curve(A["all"], e, base[e], tau); ax.plot(px, py, color=INK, lw=0.8)
                for side, ls, fc in (("top", "-", col), ("bottom", (0, (3, 1.6)), "white")):
                    x_, y_ = _curve(A[f"{k}_{side}"], e, base[e], tau)
                    ax.plot(x_, y_, ls=ls, color=col, lw=1.4, marker=mk, ms=3.2, mfc=fc, mec=col, mew=0.9)
                _dose_axes(ax, e, j == 0, lab_)
                if i == 0:
                    ax.set_title(f"{en}\n(share {what})", color=INK)
                if i == len(sel) - 1:
                    ax.set_xlabel(r"nominal $|\alpha|$")
        h = [Line2D([], [], color=col, lw=1.4, marker=mk, ms=3.5, label="top 128"),
             Line2D([], [], color=col, lw=1.4, ls=(0, (3, 1.6)), marker=mk, mfc="white", ms=3.5, label="bottom 128"),
             Patch(color=MUTED, alpha=0.3, label="random draws (range)"), Line2D([], [], color=INK, lw=0.8, label="whole pool")]
        fig.tight_layout(h_pad=0.6, w_pad=0.4, rect=(0, 0, 1, 1 - 0.42 / (1.15 * len(sel) + 0.5)))
        fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), title=name, title_fontsize=8)
        fig.savefig(FIG / f"fig_dose_sel_{model}.pdf", bbox_inches="tight"); plt.close(fig)


def fig_dose_ref():
    """Rows = model, columns = benchmark. Seven dataset vectors: thin lines in the model's colour (context); CoCoNot with both
    poles from Tülu's mixture: thick model colour; CAA letter token / answer mean and the random direction: ink and grey line styles."""
    import scoring
    from matplotlib.lines import Line2D
    S = _steer(); tau = scoring.TAU
    ds = ["g_wildguardmix", "g_wildjailbreak", "g_coconot_safety", "x_coconot_leftover", "x_donotanswer", "x_hh_rlhf", "x_pku_saferlhf"]
    special = [("caa_ab", INK, (0, (1, 1.2)), "CAA, letter token"), ("caa_mean", INK, (0, (4, 1.5, 1, 1.5)), "CAA, answer mean"),
               ("random", MUTED, (0, (3, 1.6)), "random direction")]
    fig, axes = plt.subplots(2, len(STEER_BENCH), figsize=(TEXTWIDTH, 4.6), sharey=True)
    for i, model in enumerate(("tulu", "meta")):
        name, col, mk = MODEL[model]; A = S[KEYM[model]]["arms"]; base = S[KEYM[model]]["baseline"]
        for j, (e, en, what) in enumerate(STEER_BENCH):
            ax = axes[i, j]
            for k in ds:
                x_, y_ = _curve(A[k], e, base[e], tau); ax.plot(x_, y_, color=col, lw=0.7, alpha=0.45)
            x_, y_ = _curve(A["g_coconot_insft"], e, base[e], tau); ax.plot(x_, y_, color=col, lw=1.8)
            for k, c, ls, _ in special:
                x_, y_ = _curve(A[k], e, base[e], tau); ax.plot(x_, y_, color=c, lw=1.2, ls=ls)
            _dose_axes(ax, e, j == 0, f"{name}\nrate")
            if i == 0:
                ax.set_title(f"{en}\n(share {what})", color=INK)
            if i == 1:
                ax.set_xlabel(r"nominal $|\alpha|$")
    h = [Line2D([], [], color=MUTED, lw=0.7, alpha=0.7, label="7 dataset vectors"),
         Line2D([], [], color=MUTED, lw=1.8, label="CoCoNot, both poles from Tülu's mixture")] + [Line2D([], [], color=c, lw=1.2, ls=ls, label=l) for _, c, ls, l in special]
    fig.tight_layout(h_pad=0.8, w_pad=0.4, rect=(0, 0, 1, 0.86))
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(FIG / "fig_dose_ref.pdf", bbox_inches="tight"); plt.close(fig)


# ---- appendix: surface features (part 3) -----------------------------------------------------
def fig_features():
    """(a, b) per score and model: the score's AUC (open) and the AUC of its residual after regressing it on the four surface
    features (filled), pooled refusal and compliance pole, with the features-only AUC (solid line); (c) the share of each
    score's variance the features explain (R^2), refusal pole filled, compliance pole open."""
    from matplotlib.lines import Line2D
    Gf = json.load(open(B.P3 / "results" / "mia_features" / "summary.json"))["groups"]
    ms = [m for m in B.AUC_METHODS if m in Gf["refusal, pooled"]["tulu"]["per_score"]]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(TEXTWIDTH, 3.0), sharey=True, gridspec_kw={"width_ratios": [1, 1, 0.75], "wspace": 0.08})
    for ax, (g, title) in zip((a1, a2), (("refusal, pooled", "(a) AUC, refusals"), ("compliance, pooled", "(b) AUC, compliances"))):
        for model in ("tulu", "meta"):
            name, col, mk = MODEL[model]; P = Gf[g][model]["per_score"]
            off = 0.18 if model == "tulu" else -0.18
            for i, m in enumerate(ms[::-1]):
                if m not in P:
                    continue
                raw, res = P[m]["auc"][0], P[m]["auc_residual"][0]
                ax.annotate("", xy=(res, i + off), xytext=(raw, i + off),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=0.8, alpha=0.55, mutation_scale=6, shrinkA=2.5, shrinkB=2.5))
                ax.plot(raw, i + off, mk, mfc="white", mec=col, ms=4, mew=0.9, zorder=3)
                ax.plot(res, i + off, mk, color=col, ms=4.2, mec="white", mew=0.5, zorder=3)
        F = Gf[g]["tulu"]["features_only_auc"][0]
        ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 2))); ax.axvline(F, color=INK, lw=0.8, alpha=0.6)
        ax.text(F - 0.006, -0.85, "features", fontsize=6.5, color=INK, ha="right", va="center")
        ax.set_xlim(0.35, 1.0); ax.set_xticks([0.4, 0.6, 0.8]); ax.set_title(title, color=INK); ax.tick_params(axis="y", length=0); style(ax)
    for model in ("tulu", "meta"):
        name, col, mk = MODEL[model]; off = 0.18 if model == "tulu" else -0.18
        for i, m in enumerate(ms[::-1]):
            for g, fill in (("refusal, pooled", True), ("compliance, pooled", False)):
                P = Gf[g][model]["per_score"]
                if m in P:
                    a3.plot(P[m]["r2"], i + off, mk, color=col, mfc=col if fill else "white", mec=col if not fill else "white",
                            ms=4.2 if fill else 4, mew=0.5 if fill else 0.9, zorder=3)
    a3.set_xlim(0, max(0.5, a3.get_xlim()[1])); a3.set_xticks([0.2, 0.4, 0.6]); a3.set_title("(c) $R^2$ of the features", color=INK); a3.tick_params(axis="y", length=0); style(a3)
    a1.set_yticks(range(len(ms))); a1.set_yticklabels([lab(m) for m in ms[::-1]]); a1.set_ylim(-1.2, len(ms) - 0.5)
    fig.text((a1.get_position().x0 + a2.get_position().x1) / 2, 0.0, "AUC, examples in Tülu's SFT mixture against the rest",
             ha="center", va="top", fontsize=plt.rcParams["axes.labelsize"], color=INK)
    a3.set_xlabel("share of variance")
    extra = [Line2D([], [], color=MUTED, marker="o", mfc="white", ls="", ms=4, label="(a, b) score"),
             Line2D([], [], color=MUTED, marker="o", ls="", ms=4.2, label="residual"),
             Line2D([], [], color=MUTED, marker="o", ls="", ms=4.2, label="(c) refusals"),
             Line2D([], [], color=MUTED, marker="o", mfc="white", ls="", ms=4, label="compliances")]
    legend_models(fig, extra=extra, y=1.06)
    fig.savefig(FIG / "fig_features.pdf", bbox_inches="tight"); plt.close(fig)


# ---- appendix: Table 3 as a figure ------------------------------------------------------------------------
def fig_steer_sel():
    """Table tab_steer as a dot plot: per selection, (a) S +- bootstrap s.e., (b) jailbreak range, (c) share lost to degeneration,
    both models; the random draws as one row (range over draws, mean marked), the whole pool apart. Whole-number grid when
    complete for every arm (as the table)."""
    from matplotlib.lines import Line2D
    R = json.load(open(B.P2 / "results" / "selection_steer" / "range_summary.json"))
    keym = {"tulu": "tulu_final", "meta": "meta_instruct"}
    grid, _ = B._range_arms(R, keym)
    rows = [(f"{PLAIN.get(lab, lab)}, {side}", f"{k}_{side}") for k, lab in B.STEER_SEL for side in ("top", "bottom")]
    rows += [("random draws", "random*"), ("whole pool", "all")]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 3.1), sharey=True, gridspec_kw={"width_ratios": [1.15, 1, 0.8], "wspace": 0.08})
    panels = [("S_range", "(a) steerability $S$", True), ("jailbreak", "(b) jailbreak range", False), ("lost", "(c) lost to degeneration", False)]
    for ax, (q, title, se) in zip(axes, panels):
        for model in ("tulu", "meta"):
            name, col, mk = MODEL[model]; A = R[keym[model]][grid]; off = 0.17 if model == "tulu" else -0.17
            for i, (lab_, k) in enumerate(rows[::-1]):
                y = i + off
                if k == "random*":
                    v = [A[a][q] for a in A if a.startswith("random") and a[6:].isdigit()]
                    ax.plot([min(v), max(v)], [y, y], color=col, lw=2.2, alpha=0.45, solid_capstyle="round")
                    ax.plot(np.mean(v), y, "|", color=col, ms=7, mew=1.4, zorder=3)
                    ax.axvspan(min(v), max(v), color=col, alpha=0.08, lw=0, zorder=0)   # the random draws' range, across all rows
                    continue
                if k not in A:
                    continue
                x = A[k][q]; filled = not k.endswith("_bottom")
                if se:
                    ax.errorbar(x, y, xerr=A[k]["S_se"], fmt="none", ecolor=col, elinewidth=0.8, capsize=0, alpha=0.8)
                ax.plot(x, y, mk, color=col, mfc=col if filled else "white", mec=col if not filled else "white", ms=4.2, mew=0.9 if not filled else 0.5, zorder=3)
        ax.set_title(title, color=INK); ax.tick_params(axis="y", length=0); style(ax)
    for ax in axes[:2]:
        ax.axhline(1.5, color=GRID, lw=0.8)
    axes[2].axhline(1.5, color=GRID, lw=0.8)
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels([r[0] for r in rows[::-1]]); axes[0].set_ylim(-0.6, len(rows) - 0.4)
    axes[0].set_xlabel("share of prompts"); axes[1].set_xlabel("share of prompts"); axes[2].set_xlabel("share of prompts")
    extra = [Line2D([], [], color=MUTED, marker="o", ls="", ms=4.2, label="top 128 / whole pool"),
             Line2D([], [], color=MUTED, marker="o", mfc="white", ls="", ms=4, label="bottom 128"),
             Line2D([], [], color=MUTED, lw=2.2, alpha=0.45, marker="|", ms=7, mew=1.4, label="random draws: range (shaded), mean")]
    legend_models(fig, extra=extra, y=1.06)
    fig.savefig(FIG / "fig_steer_sel.pdf", bbox_inches="tight"); plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True); fig_scale_auc(); fig_geom(); fig_dose_sel(); fig_dose_ref(); fig_features(); print("figures in", FIG)


if __name__ == "__main__":
    main()
