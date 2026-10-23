"""Step 8: build every README figure from the saved results.

Run from the repo root (after steps 5-7, Yahoo and CRSP):
    pip install matplotlib
    PYTHONPATH=src python scripts/make_figures.py

Reads results/ and data/processed*/ and writes PNGs to figures/ (commit these; they contain
only aggregated results, no licensed data). A figure whose inputs are missing is skipped.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alpha.signals import SIGNAL_NAMES

FIG = Path("figures")
R = Path("results")
YAHOO, CRSP = Path("data/processed"), Path("data/processed_crsp")

# Okabe-Ito: readable for colour-blind viewers
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7F7F7F", "ink": "#222222"}
STRAT = {"reversal_only": ("Reversal alone", C["blue"]), "equal": ("Equal weight", C["orange"]),
         "ic_weighted": ("IC-weighted", C["purple"]), "ridge": ("Ridge", C["green"]),
         "lgbm": ("LightGBM", C["red"])}
LABEL = {"momentum": "Momentum 12-1", "reversal": "Short-term reversal", "high_52w": "52-week high",
         "volatility": "Low volatility", "beta": "Low beta", "idio_vol": "Low idio. vol",
         "max_return": "Low MAX", "amihud": "Amihud illiquidity"}

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})
made, skipped = [], []


def figure(name):
    """Decorator: run the plot, save it, and skip cleanly if an input file is missing."""
    def wrap(fn):
        def run():
            try:
                fig = fn()
                fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
                plt.close(fig)
                made.append(name)
            except FileNotFoundError as e:
                skipped.append(f"{name} (missing {Path(e.filename).as_posix() if e.filename else e})")
        return run
    return wrap


def coverage(processed: Path) -> pd.Series:
    mask = pd.read_parquet(processed / "mask.parquet")
    rets = pd.read_parquet(processed / "returns.parquet")
    m = mask.reindex(index=rets.index, columns=mask.columns).fillna(False).astype(bool)
    have = m & rets.reindex(columns=mask.columns).notna()
    yr = m.index.year
    tot = m.sum(axis=1).groupby(yr).sum()
    return (100 * have.sum(axis=1).groupby(yr).sum() / tot)[tot > 0]


# ---------------------------------------------------------------------------------- figures
@figure("01_coverage")
def fig_coverage():
    y = pd.read_csv(YAHOO / "coverage_by_year.csv", index_col=0)["coverage_pct"]
    c = coverage(CRSP)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(c.index, c.values, color=C["blue"], lw=2.2, marker="o", ms=3, label="CRSP (point-in-time, incl. delisted)")
    ax.plot(y.index, y.values, color=C["red"], lw=2.2, marker="o", ms=3, label="Yahoo Finance (free)")
    ax.set_ylim(40, 102)
    ax.set_ylabel("% of index member-days with data")
    ax.set_title("Free data misses the companies that failed")
    ax.legend(loc="lower right")
    return fig


@figure("02_survivorship_shift")
def fig_shift():
    y = pd.read_csv(R / "step5_yahoo" / "summary.csv", index_col=0)["Q5_Q1_annual"] * 100
    c = pd.read_csv(R / "step5_crsp_2006" / "summary.csv", index_col=0)["Q5_Q1_annual"] * 100
    order = SIGNAL_NAMES
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.bar(x - 0.2, y[order], 0.4, color=C["red"], label="Yahoo (survivors only)")
    ax.bar(x + 0.2, c[order], 0.4, color=C["blue"], label="CRSP (everyone)")
    ax.axhline(0, color=C["ink"], lw=0.8)
    ax.set_xticks(x, [LABEL[k] for k in order], rotation=30, ha="right")
    ax.set_ylabel("Top minus bottom quintile, % / year")
    ax.set_title("Same signals, same years (2006-2025), different data")
    ax.legend()
    return fig


@figure("03_signal_tstats")
def fig_tstats():
    s = pd.read_csv(R / "step5_crsp" / "summary.csv", index_col=0)["ic_t_nw"].reindex(SIGNAL_NAMES)
    s = s.sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    colors = [C["blue"] if abs(v) > 3 else C["grey"] for v in s]
    ax.barh([LABEL[k] for k in s.index], s.values, color=colors)
    for v, ls, lab in [(3, "--", "t = 3 (pre-registered bar)"), (1.96, ":", "t = 1.96")]:
        ax.axvline(v, color=C["red"], ls=ls, lw=1.2, label=lab)
        ax.axvline(-v, color=C["red"], ls=ls, lw=1.2)
    ax.set_xlabel("21-day rank IC, Newey-West t-statistic")
    ax.set_title("Only reversal clears the multiple-testing bar (CRSP, 1997-2025)")
    ax.legend(loc="lower right")
    return fig


@figure("04_ic_decay")
def fig_decay():
    d = pd.read_csv(R / "step5_crsp" / "ic_decay.csv", index_col=0)
    h = [int(c.rstrip("d")) for c in d.columns]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    palette = [C["blue"], C["orange"], C["green"], C["red"], C["purple"], C["sky"], C["grey"], C["ink"]]
    for (k, row), col in zip(d.iterrows(), palette):
        lw = 2.6 if k == "reversal" else 1.3
        ax.plot(h, row.values, marker="o", ms=3, lw=lw, color=col, label=LABEL.get(k, k))
    ax.set_xscale("log")
    ax.set_xticks(h, [f"{x}d" for x in h])
    ax.axhline(0, color=C["ink"], lw=0.8)
    ax.set_xlabel("forecast horizon (trading days)")
    ax.set_ylabel("mean rank IC")
    ax.set_title("How long each signal's information lasts")
    ax.legend(ncol=2, fontsize=8)
    return fig


@figure("05_quintiles")
def fig_quintiles():
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.6), sharey=True)
    for ax, k in zip(axes.flat, SIGNAL_NAMES):
        b = pd.read_csv(R / "step5_crsp" / f"quintiles_{k}.csv", index_col=0)
        m = b.mean() * 1200
        ax.bar(range(1, 6), m.values, color=[C["sky"]] * 4 + [C["blue"]])
        ax.axhline(0, color=C["ink"], lw=0.8)
        ax.set_title(LABEL[k], fontsize=9)
        ax.set_xticks(range(1, 6), ["Q1", "", "Q3", "", "Q5"])
    axes[0, 0].set_ylabel("% / year")
    axes[1, 0].set_ylabel("% / year")
    fig.suptitle("Average return by signal quintile (CRSP, 1997-2025, monthly rebalance)", fontweight="bold")
    fig.tight_layout()
    return fig


@figure("06_signal_correlation")
def fig_corr():
    ranks = {k: pd.read_parquet(CRSP / "signals" / f"rank_{k}.parquet") for k in SIGNAL_NAMES}
    idx = ranks["momentum"].index
    month_ends = pd.Series(idx, index=idx).groupby(idx.to_period("M")).last().to_numpy()
    mats = []
    for d in month_ends:
        f = pd.DataFrame({k: ranks[k].loc[d] for k in SIGNAL_NAMES}).dropna()
        if len(f) > 50:
            mats.append(f.corr().to_numpy())
    M = np.nanmean(mats, axis=0)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [LABEL[k] for k in SIGNAL_NAMES]
    ax.set_xticks(range(8), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(8), labels, fontsize=8)
    ax.grid(False)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j]) > 0.5 else C["ink"])
    fig.colorbar(im, ax=ax, shrink=0.8, label="average cross-sectional rank correlation")
    ax.set_title("Eight signals, about four independent bets")
    return fig


@figure("07_combiners_oos")
def fig_combiners():
    s = pd.read_csv(R / "step6_crsp" / "summary.csv", index_col=0)
    se = (s["mean_ic"] / s["ic_t_nw"]).abs()
    order = [k for k in STRAT if k in s.index]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar([STRAT[k][0] for k in order], s.loc[order, "mean_ic"], yerr=2 * se[order], capsize=4,
           color=[STRAT[k][1] for k in order])
    ax.axhline(0, color=C["ink"], lw=0.8)
    ax.set_ylabel("out-of-sample mean monthly rank IC\n(error bars = 2 standard errors)")
    ax.set_title("Combining signals or adding ML does not beat reversal alone (2002-2025)")
    return fig


@figure("08_cumulative_net")
def fig_cum():
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for k, (lab, col) in STRAT.items():
        p = R / "step7_crsp" / f"daily_{k}.csv"
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        ax.plot(d.index, (1 + d["net"]).cumprod(), color=col, lw=1.6 if k == "reversal_only" else 1.1, label=lab)
        if k == "reversal_only":
            ax.plot(d.index, (1 + d["gross"]).cumprod(), color=col, lw=1.2, ls="--", label="Reversal, before costs")
    ax.axhline(1, color=C["ink"], lw=0.8)
    ax.set_ylabel("growth of $1 (net of spread + borrow)")
    ax.set_title("Long-short portfolios after trading costs")
    ax.legend(ncol=2, fontsize=8)
    return fig


@figure("09_cost_curve")
def fig_costs():
    s = pd.read_csv(R / "step7_crsp" / "cost_sensitivity.csv", index_col=0)
    x = [int(c.split()[0]) for c in s.columns]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, row in s.iterrows():
        lab, col = STRAT.get(k, (k, C["grey"]))
        ax.plot(x, row.values, marker="o", ms=4, color=col, lw=2.2 if k == "reversal_only" else 1.2, label=lab)
    ax.axhline(0, color=C["ink"], lw=0.8)
    ax.set_xlabel("one-way cost per $ traded (bps)")
    ax.set_ylabel("net Sharpe ratio")
    ax.set_title("Reversal's edge is gone by about 10 bps")
    ax.legend(fontsize=8)
    return fig


@figure("10_capacity")
def fig_capacity():
    s = pd.read_csv(R / "step7_crsp" / "capacity.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(s.columns))
    for k, row in s.iterrows():
        lab, col = STRAT.get(k, (k, C["grey"]))
        ax.plot(x, row.values, marker="o", color=col, lw=1.8, label=lab)
    ax.set_xticks(x, ["quoted spread only", "$100M fund", "$1B fund"])
    ax.axhline(0, color=C["ink"], lw=0.8)
    ax.set_ylabel("net Sharpe ratio")
    ax.set_title("Adding market impact: no capacity")
    ax.legend(fontsize=8)
    return fig


@figure("11_drawdowns")
def fig_dd():
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for k in ["reversal_only", "equal"]:
        d = pd.read_csv(R / "step7_crsp" / f"daily_{k}.csv", index_col=0, parse_dates=True)
        eq = (1 + d["net"]).cumprod()
        lab, col = STRAT[k]
        ax.fill_between(eq.index, 100 * (eq / eq.cummax() - 1), 0, color=col, alpha=0.35, label=lab)
    ax.set_ylabel("drawdown from peak, %")
    ax.set_title("Dollar-neutral is not market-neutral: the equal-weight mix is short beta")
    ax.legend(loc="lower left")
    return fig


def main():
    FIG.mkdir(exist_ok=True)
    for f in [fig_coverage, fig_shift, fig_tstats, fig_decay, fig_quintiles, fig_corr,
              fig_combiners, fig_cum, fig_costs, fig_capacity, fig_dd]:
        f()
    print(f"made {len(made)} figures in {FIG}/: {', '.join(made)}")
    for s in skipped:
        print(f"  skipped {s}")


if __name__ == "__main__":
    main()