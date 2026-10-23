"""Distribution plots of the network metrics. Reads results/daily_metrics.csv and
results/account_metrics.csv (written by `python -m src.run metrics`).

    python plot.py                      # both datasets pooled
    python plot.py --dataset pcpe_03    # one dataset
    python plot.py --keep-singletons    # keep graphs with a single node (d, <d> undefined)

    python plot.py --only values        # one family of figures (default: all that have inputs)

Figure families (inputs are the CSVs in results/, missing inputs are skipped):
  metrics     daily/account *_metrics_ecdf.png and *_metrics_hist.png   (daily_metrics.csv, account_metrics.csv)
  sequential  sequential-consistency bars, original vs processed         (sequential_consistency.csv)
  eulerian    Closed / Open / Not Eulerian percentage bars               (daily_metrics.csv)
  values      symlog ECDF of balances and values with KS vs Closed       (values.csv, ks_values.csv)
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import METRIC_COLUMNS

LABELS = {
    "N": "N (nodes)", "E": "E (edges)", "d": "d (diameter)", "avg_d": "<d> (average distance)",
    "density": "density", "components": "components", "avg_degree": "<k> (average degree)",
    "lcc_fraction": "largest-component fraction",
}
LOG_METRICS = {"N", "E", "d", "avg_d", "density"}  # heavy-tailed: log x-axis
STYLE = {True: ("Consistent", "#1f77b4"), False: ("Inconsistent", "#d62728")}


def _values(df: pd.DataFrame, metric: str, log: bool) -> np.ndarray:
    v = df[metric].to_numpy(float)
    v = v[np.isfinite(v)]
    return v[v > 0] if log else v


def _bins(arrays: list[np.ndarray], metric: str, log: bool) -> np.ndarray:
    pooled = np.concatenate(arrays)
    lo, hi = pooled.min(), pooled.max()
    if lo == hi:
        return np.array([lo - 0.5, hi + 0.5]) if not log else np.array([lo / 1.5, hi * 1.5])
    if metric == "components":
        return np.arange(lo, hi + 2) - 0.5
    return np.geomspace(lo, hi, 26) if log else np.linspace(lo, hi, 26)


def plot_level(df: pd.DataFrame, level: str, kind: str, title: str, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2))
    for ax, metric in zip(axes.flat, METRIC_COLUMNS):
        log = metric in LOG_METRICS
        groups = {flag: _values(df[df["consistent"]] if flag else df[~df["consistent"]], metric, log)
                  for flag in (True, False)}
        groups = {k: v for k, v in groups.items() if len(v)}
        if not groups:
            ax.set_visible(False)
            continue
        bins = _bins(list(groups.values()), metric, log) if kind == "hist" else None
        for flag, v in groups.items():
            name, color = STYLE[flag]
            label = f"{name} (n={len(v):,})"
            if kind == "ecdf":
                x = np.sort(v)
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color, lw=1.6, label=label)
            else:
                ax.hist(v, bins=bins, weights=np.full(len(v), 1 / len(v)), histtype="step",
                        color=color, lw=1.6, label=label)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(LABELS[metric])
        ax.set_ylabel("ECDF" if kind == "ecdf" else "Relative frequency")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=13, weight="bold")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------- paper figures (ported from the old scripts)
def _style() -> None:
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif",
                         "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8})


def _percentage_bars(percent: pd.Series, colors: list[str], title: str, xlabel: str, ylabel: str,
                     width: float, size: tuple[float, float], out: pathlib.Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=size)
    bars = ax.bar(percent.index, percent.to_numpy(), color=colors, width=width, edgecolor="none")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    for rect in bars:
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2.0, f"{rect.get_height():.1f}%",
                ha="center", va="bottom", fontsize=9.5, weight="bold", color="#333333")
    ax.set_title(title, fontsize=13, pad=18, weight="bold")
    ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=10)
    ax.yaxis.grid(True, ls="--", alpha=0.5, color="#cccccc")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved", out)


def plot_sequential_consistency(seq: pd.DataFrame, figures: pathlib.Path) -> None:
    """Percentage of consistent / inconsistent sequential pairs, per dataset and version."""
    for (dataset, version), df in seq.groupby(["dataset", "version"]):
        counts = pd.Series({"Consistent": df["consistent"].sum(), "Inconsistent": df["inconsistent"].sum()})
        total = counts.sum()
        percent = counts / total * 100 if total else counts * 0.0
        _percentage_bars(percent, ["#619dfd", "#ff6167"], f"{dataset} (N={int(total)})",
                         "Sequential State Transition Type", "Percentage of Sequences (%)", 0.4, (5.5, 3.6),
                         figures / dataset / f"{version}_{dataset}_sequential_consistency_histogram.png")


def plot_eulerian_histogram(daily: pd.DataFrame, figures: pathlib.Path) -> None:
    """Percentage of daily graphs that are Open trail, Closed trail or Not Eulerian."""
    from src.analysis import eulerian_group

    names = {"Open": "Open Trail", "Closed": "Closed Trail\n(Pseudosymmetric)", "Not_Eulerian": "Not Eulerian"}
    for dataset, df in daily.groupby("dataset"):
        counts = df["status"].map(eulerian_group).value_counts().reindex(list(names), fill_value=0)
        total = counts.sum()
        percent = (counts / total * 100).rename(index=names)
        _percentage_bars(percent, ["#4c72b0", "#dd8452", "#55a868"], f"{dataset} (N={int(total)})",
                         "Eulerian Characteristic", "Percentage of Subnetworks (%)", 0.5, (6.0, 3.6),
                         figures / dataset / "eulerian_trail_histogram.png")


def plot_value_cdfs(values: pd.DataFrame, ks: pd.DataFrame, figures: pathlib.Path) -> None:
    """Symmetric-log ECDF of node balances and edge values by Eulerian status; the legend
    carries the KS statistic against the Closed baseline (from ks_values.csv)."""
    from src.analysis import eulerian_group

    _style()
    config = {"Closed": ("Closed", "#4c72b0", "-"), "Open": ("Open", "#dd8452", "--"),
              "Not_Eulerian": ("Not Eulerian", "#55a868", "-.")}
    values = values.assign(group=values["status"].map(eulerian_group))
    for (dataset, kind), df in values.groupby(["dataset", "kind"]):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for group, (name, color, ls) in config.items():
            x = np.sort(df.loc[df["group"] == group, "value"].to_numpy(float))
            if not len(x):
                continue
            label = f"{name} (Baseline)" if group == "Closed" else name
            row = ks[(ks["dataset"] == dataset) & (ks["kind"] == kind) & (ks["group"] == group)]
            if group != "Closed" and len(row) and np.isfinite(row["D"].iloc[0]):
                p = row["p_value"].iloc[0]
                label += f" (KS D={row['D'].iloc[0]:.2f}, p={p:.3f})" if p >= 0.001 else f" (KS D={row['D'].iloc[0]:.2f}, p={p:.1e})"
            ax.plot(x, np.arange(1, len(x) + 1) / len(x) * 100, label=label, color=color, ls=ls, lw=2.0)
        ax.set_xscale("symlog", linthresh=10.0)
        title = kind.title()
        ax.set_xlabel(f"Transaction {title} SymLog Scale", labelpad=10)
        ax.set_ylabel(f"Percentage of {'Nodes' if kind == 'balance' else 'Edges'} (%)", labelpad=10)
        ax.set_title(f"Symmetric Log CDF of the Transactions {title} - {dataset}", fontsize=12, pad=15, weight="bold")
        ax.set_ylim(-2, 102)
        ax.grid(True, which="both", ls="--", alpha=0.3, color="#cccccc")
        ax.legend(loc="upper left", frameon=True, fontsize=9.0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        out = figures / dataset / f"triple_cdf_{kind}_symlog.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print("saved", out)


def _read(path: pathlib.Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"skipping: {path} not found")
        return None
    return pd.read_csv(path)


def plot_metrics(args) -> None:
    daily, accounts = _read(args.results / "daily_metrics.csv"), _read(args.results / "account_metrics.csv")
    if daily is None or accounts is None:
        return
    suffix = f"_{args.dataset}" if args.dataset else ""
    for level, df in {"daily": daily, "account": accounts}.items():
        df["consistent"] = df["consistent"].astype(bool)
        if args.dataset:
            df = df[df["dataset"] == args.dataset]
        if not args.keep_singletons:
            single = df["N"] < 2
            print(f"{level}: excluding {single.sum()} single-node graphs "
                  f"({(single & df['consistent']).sum()} consistent, {(single & ~df['consistent']).sum()} inconsistent)")
            df = df[~single]
        for kind in ("ecdf", "hist"):
            title = f"{level.capitalize()} networks{' - ' + args.dataset if args.dataset else ''}"
            plot_level(df, level, kind, title, args.figures / f"{level}{suffix}_metrics_{kind}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--results", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--figures", type=pathlib.Path, default=pathlib.Path("figures"))
    parser.add_argument("--dataset", help="restrict the metrics figures to one dataset")
    parser.add_argument("--keep-singletons", action="store_true")
    parser.add_argument("--only", choices=["metrics", "sequential", "eulerian", "values"])
    args = parser.parse_args()
    run = lambda name: args.only in (None, name)  # noqa: E731

    if run("metrics"):
        plot_metrics(args)
    if run("sequential") and (seq := _read(args.results / "sequential_consistency.csv")) is not None:
        plot_sequential_consistency(seq, args.figures)
    if run("eulerian") and (daily := _read(args.results / "daily_metrics.csv")) is not None:
        plot_eulerian_histogram(daily, args.figures)
    if run("values"):
        values, ks = _read(args.results / "values.csv"), _read(args.results / "ks_values.csv")
        if values is not None and ks is not None:
            plot_value_cdfs(values, ks, args.figures)


if __name__ == "__main__":
    main()