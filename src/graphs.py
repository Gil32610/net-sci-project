from __future__ import annotations

import pathlib
from collections.abc import Iterable, Iterator
from typing import NamedTuple

import networkx as nx 

GRAPHS_DIR = pathlib.Path('transaction_networks')

class GraphFile(NamedTuple):
    dataset: str
    account: str
    date: str
    path: pathlib.Path

def read_network(path: pathlib.Path):
    return nx.MultiDiGraph(nx.read_graphml(path))

def write_network(network:nx.MultiDiGraph, directory:pathlib.Path, name:str) -> pathlib.Path:
    out = network.copy()

    for _,_, _, data in out.edges(keys=True, data=True):
        ts = data.get("timestamp")
        if hasattr(ts, "isoformat"):
            data["timestamp"] = ts.isoformat()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.graphml"
    nx.write_graphml(out, path)
    return path

def export_daily_networks(df, dataset: str, root: pathlib.Path = GRAPHS_DIR) -> int:
    """Build one multidigraph per (account, date) with mlex and save it as GraphML."""
    from mlex import TransactionMultiDigraph
 
    from src.data import iter_timestamp_sequences
 
    count = 0
    for account, date, sequence in iter_timestamp_sequences(df):
        network = TransactionMultiDigraph.create_multidigraph(sequence)
        write_network(network, root / dataset / account, date)
        count += 1
    return count

def iter_graph_files(root: pathlib.Path = GRAPHS_DIR, datasets: Iterable[str] | None = None) -> Iterator[GraphFile]:
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if datasets is not None and dataset_dir.name not in set(datasets):
            continue
        for account_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            for path in sorted(account_dir.glob("*.graphml")):
                yield GraphFile(dataset_dir.name, account_dir.name, path.stem, path)

def to_simple_undirected(network: nx.Graph) -> nx.Graph:
    """Undirected, unweighted view: parallel edges collapsed, self-loops dropped.
    Node ids (balance values) are kept as-is."""
    simple = nx.Graph()
    simple.add_nodes_from(network.nodes)
    simple.add_edges_from((u, v) for u, v in network.edges() if u != v)
    return simple
 
 
def merge_account_graph(paths: Iterable[pathlib.Path]) -> nx.Graph:
    """Union of the (simple, undirected) daily graphs of ONE account.
    Nodes with the same balance id are the same node, so consecutive days connect
    through shared balances."""
    merged = nx.Graph()
    for path in paths:
        merged.update(to_simple_undirected(read_network(path)))
    return merged