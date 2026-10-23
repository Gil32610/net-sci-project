"""Network science metrics on the undirected, unweighted view of the graphs.

Two cases (each split into consistent / inconsistent):
  * daily networks    - one graph per (account, date)
  * account networks  - union of all daily graphs of an account

Definitions
  N, E          nodes / edges of the SIMPLE undirected graph (parallel edges collapsed,
                self-loops dropped; the original multigraph counts are kept separately)
  density       2E / (N(N-1))
  components    number of connected components
  avg_degree    2E / N
  d, avg_d      diameter and average shortest path length of the LARGEST connected
                component; NaN when that component has fewer than 2 nodes
  lcc_fraction  fraction of the nodes that belong to the largest component
"""
from __future__ import annotations

import math
import pathlib
from itertools import groupby

import networkx as nx
import pandas as pd

from src.graphs import GRAPHS_DIR, iter_graph_files, merge_account_graph, read_network, to_simple_undirected
from src.quality import account_is_consistent, classify_eulerian, is_pseudosymmetric

METRIC_COLUMNS = ["N", "E", "d", "avg_d", "density", "components", "avg_degree", "lcc_fraction"]


def _distances(component: nx.Graph) -> tuple[float, float]:
    """Diameter and average shortest path of a connected graph in one BFS sweep."""
    n = component.number_of_nodes()
    if n < 2:
        return math.nan, math.nan
    diameter, total = 0, 0
    for source in component:
        lengths = nx.single_source_shortest_path_length(component, source)
        diameter = max(diameter, max(lengths.values()))
        total += sum(lengths.values())
    return diameter, total / (n * (n - 1))


def graph_metrics(graph: nx.Graph) -> dict:
    n, e = graph.number_of_nodes(), graph.number_of_edges()
    if n == 0:
        raise ValueError("empty graph")
    components = list(nx.connected_components(graph))
    largest = max(components, key=len)
    d, avg_d = _distances(graph.subgraph(largest))
    return {
        "N": n, "E": e, "d": d, "avg_d": avg_d,
        "density": nx.density(graph), "components": len(components),
        "avg_degree": 2 * e / n, "lcc_fraction": len(largest) / n,
    }


def build_metrics_tables(root: pathlib.Path = GRAPHS_DIR, datasets=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One pass over the GraphML files -> (daily_metrics, account_metrics)."""
    daily_rows, account_rows = [], []
    files = iter_graph_files(root, datasets)
    for (dataset, account), group in groupby(files, key=lambda g: (g.dataset, g.account)):
        merged = nx.Graph()
        n_days = n_multi = n_loops = 0
        all_consistent = []
        for gf in group:
            multi = read_network(gf.path)
            status = classify_eulerian(multi)
            simple = to_simple_undirected(multi)
            loops = nx.number_of_selfloops(multi)
            daily_rows.append({
                "dataset": dataset, "account": account, "date": gf.date,
                "status": status.value, "consistent": status.consistent,
                "pseudosymmetric": is_pseudosymmetric(multi),
                **graph_metrics(simple),
                "n_multi_edges": multi.number_of_edges(), "n_self_loops": loops,
            })
            merged.update(simple)
            all_consistent.append(status.consistent)
            n_days += 1
            n_multi += multi.number_of_edges()
            n_loops += loops
        account_rows.append({
            "dataset": dataset, "account": account, "n_days": n_days,
            "consistent": account_is_consistent(pd.Series(all_consistent)),
            **graph_metrics(merged),
            "n_multi_edges": n_multi, "n_self_loops": n_loops,
        })
    return pd.DataFrame(daily_rows), pd.DataFrame(account_rows)


def split_branches(daily: pd.DataFrame, accounts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The 4 branches of the analysis."""
    return {
        "daily/consistent": daily[daily["consistent"]],
        "daily/inconsistent": daily[~daily["consistent"]],
        "account/consistent": accounts[accounts["consistent"]],
        "account/inconsistent": accounts[~accounts["consistent"]],
    }


def summarize(branches: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count / mean / median / std / min / max of every metric per branch."""
    parts = {name: df[METRIC_COLUMNS].describe().T for name, df in branches.items() if len(df)}
    return pd.concat(parts, names=["branch", "metric"])