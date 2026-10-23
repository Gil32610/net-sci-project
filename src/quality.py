"""Consistency of the sequences/graphs: Eulerian classification and sequential consistency."""
from __future__ import annotations

import pathlib
from enum import Enum
from itertools import pairwise

import networkx as nx
import pandas as pd

from src.graphs import GRAPHS_DIR, iter_graph_files, read_network


class EulerianStatus(str, Enum):
    OPEN = "Open"                                   # Eulerian open trail
    CLOSED = "Closed"                               # Eulerian circuit (pseudosymmetric, k = 0)
    ASYMMETRIC = "Not_Eulerian_Asymmetric"          # connected but NOT pseudosymmetric
    DISCONNECTED = "Not_Eulerian_Disconnected"      # not connected (degrees may or may not balance)

    @property
    def consistent(self) -> bool:
        return self in (EulerianStatus.OPEN, EulerianStatus.CLOSED)


def _unbalanced(network: nx.MultiDiGraph) -> list[int]:
    """Sorted non-zero values of (out_degree - in_degree) over the nodes."""
    return sorted(d for d in (network.out_degree(n) - network.in_degree(n) for n in network) if d != 0)


def is_pseudosymmetric(network: nx.MultiDiGraph) -> bool:
    """Degree condition for an Eulerian trail (connectivity NOT checked): either every node
    has in == out (k = 0, closed trail) or exactly one node has out = in + 1 and exactly one
    has in = out + 1 (k = 1, open trail)."""
    return _unbalanced(network) in ([], [-1, 1])


def classify_eulerian(network: nx.MultiDiGraph) -> EulerianStatus:
    """Single source of truth for the Eulerian status of a daily graph:
    consistent  <=>  weakly connected AND pseudosymmetric."""
    if not nx.is_weakly_connected(network):
        return EulerianStatus.DISCONNECTED
    unbalanced = _unbalanced(network)
    if unbalanced == []:
        return EulerianStatus.CLOSED
    if unbalanced == [-1, 1]:
        return EulerianStatus.OPEN
    return EulerianStatus.ASYMMETRIC


def account_is_consistent(daily_consistent: pd.Series) -> bool:
    """An account is consistent only if every one of its dates is consistent."""
    return bool(daily_consistent.all())


# ---------------------------------------------------------------- graph labels
def label_daily_graphs(root: pathlib.Path = GRAPHS_DIR, datasets=None) -> pd.DataFrame:
    rows = []
    for gf in iter_graph_files(root, datasets):
        network = read_network(gf.path)
        status = classify_eulerian(network)
        rows.append(
            {"dataset": gf.dataset, "account": gf.account, "date": gf.date,
             "status": status.value, "consistent": status.consistent,
             "pseudosymmetric": is_pseudosymmetric(network)}
        )
    return pd.DataFrame(rows)


def label_accounts(daily: pd.DataFrame) -> pd.DataFrame:
    return (
        daily.groupby(["dataset", "account"])["consistent"]
        .agg(consistent=account_is_consistent, n_days="size")
        .reset_index()
    )


# ---------------------------------------------------------------- sequential consistency (before/after)
def _deep_len(item) -> int:
    if isinstance(item, (list, tuple, set, frozenset)):
        return sum(_deep_len(x) for x in item)
    return 1


def original_sequential_counts(sequence) -> tuple[int, int]:
    """(consistent, inconsistent) pairs in the ORIGINAL order. The first element of a
    sequence is counted as consistent, so the total equals len(sequence)."""
    consistent, inconsistent = 1, 0
    for cur, nxt in pairwise(sequence):
        if cur.balance_amount == nxt.balance_amount - nxt.tx_value:
            consistent += 1
        else:
            inconsistent += 1
    return consistent, inconsistent


def processed_sequential_counts(sequence) -> tuple[int, int]:
    """(consistent, inconsistent) AFTER the mlex Eulerian-trail reordering."""
    from mlex.analysis.consistency import IntraTimestamp

    S, W = IntraTimestamp.balance_consistency([sequence])
    return (len(S[0]) if S else 0), _deep_len(W)


def sequential_consistency_table(df, dataset: str) -> pd.DataFrame:
    """Long table: dataset, account, date, version, consistent, inconsistent."""
    from src.data import iter_timestamp_sequences

    rows = []
    for account, date, seq in iter_timestamp_sequences(df):
        for version, fn in (("original", original_sequential_counts),
                            ("processed", processed_sequential_counts)):
            c, i = fn(seq)
            rows.append({"dataset": dataset, "account": account, "date": date,
                         "version": version, "consistent": c, "inconsistent": i})
    return pd.DataFrame(rows)