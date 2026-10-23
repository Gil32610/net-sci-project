"""Statistical analysis: pooled balance/value distributions by Eulerian status and KS tests.

  collect_values  one row per node (kind='balance') and per edge (kind='value') of every
                  daily graph, tagged with the graph's Eulerian status (amounts in currency
                  units, i.e. the integer cents divided by 100)
  ks_values       two-sample KS of Open and Not_Eulerian against the Closed baseline
  ks_metrics      two-sample KS of consistent vs inconsistent for every network metric
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.graphs import GRAPHS_DIR, iter_graph_files, read_network
from src.metrics import METRIC_COLUMNS
from src.quality import classify_eulerian

GROUPS = ("Closed", "Open", "Not_Eulerian")
VALUE_COLUMNS = ["dataset", "account", "date", "status", "kind", "value"]


def eulerian_group(status: str) -> str:
    """Open / Closed stay as they are; asymmetric and disconnected graphs are 'Not_Eulerian'."""
    return status if status in ("Open", "Closed") else "Not_Eulerian"


def collect_values(root: pathlib.Path = GRAPHS_DIR, datasets=None) -> pd.DataFrame:
    rows = []
    for gf in iter_graph_files(root, datasets):
        network = read_network(gf.path)
        base = (gf.dataset, gf.account, gf.date, classify_eulerian(network).value)
        rows.extend((*base, "balance", float(node) / 100.0) for node in network.nodes)
        rows.extend((*base, "value", float(v) / 100.0) for _, _, v in network.edges(data="tx_value"))
    return pd.DataFrame(rows, columns=VALUE_COLUMNS)


def ks_values(values: pd.DataFrame) -> pd.DataFrame:
    values = values.assign(group=values["status"].map(eulerian_group))
    rows = []
    for (dataset, kind), df in values.groupby(["dataset", "kind"]):
        baseline = df.loc[df["group"] == "Closed", "value"].to_numpy(float)
        for group in ("Open", "Not_Eulerian"):
            sample = df.loc[df["group"] == group, "value"].to_numpy(float)
            row = {"dataset": dataset, "kind": kind, "group": group,
                   "n_closed": len(baseline), "n_group": len(sample), "D": np.nan, "p_value": np.nan}
            if len(baseline) and len(sample):
                res = ks_2samp(baseline, sample)
                row["D"], row["p_value"] = res.statistic, res.pvalue
            rows.append(row)
    return pd.DataFrame(rows)


def ks_metrics(daily: pd.DataFrame, accounts: pd.DataFrame, keep_singletons: bool = False) -> pd.DataFrame:
    rows = []
    for level, df in (("daily", daily), ("account", accounts)):
        df = df.assign(consistent=df["consistent"].astype(bool))
        if not keep_singletons:
            df = df[df["N"] >= 2]
        for dataset, part in [("all", df), *df.groupby("dataset")]:
            for metric in METRIC_COLUMNS:
                a = part.loc[part["consistent"], metric].dropna().to_numpy(float)
                b = part.loc[~part["consistent"], metric].dropna().to_numpy(float)
                row = {"level": level, "dataset": dataset, "metric": metric,
                       "n_consistent": len(a), "n_inconsistent": len(b), "D": np.nan, "p_value": np.nan}
                if len(a) and len(b):
                    res = ks_2samp(a, b)
                    row["D"], row["p_value"] = res.statistic, res.pvalue
                rows.append(row)
    return pd.DataFrame(rows)