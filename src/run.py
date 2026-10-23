"""Pipeline entry point:  python -m src.run <stage> [--datasets pcpe_03 pcpe_04]

stages:
  graphs       CSV -> daily GraphML files (needs mlex + private data)
  sequential   sequential consistency before/after the Eulerian trail
  labels       classify every daily graph; derive account labels
  metrics      network metrics for daily and account networks (4 branches)
  values       node balances / edge values of every daily graph + Eulerian status
  ks           KS tests (values vs Closed baseline; consistent vs inconsistent metrics)
"""
from __future__ import annotations

import argparse
import pathlib

from src.data import DATASETS, load_dataframe
from src.graphs import GRAPHS_DIR, export_daily_networks
from src.analysis import collect_values, ks_metrics, ks_values
from src.metrics import build_metrics_tables, split_branches, summarize
from src.quality import label_accounts, label_daily_graphs, sequential_consistency_table

RESULTS = pathlib.Path("results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("stage", choices=["graphs", "sequential", "labels", "metrics", "values", "ks"])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--graphs-dir", type=pathlib.Path, default=GRAPHS_DIR)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    if args.stage == "graphs":
        for ds in args.datasets:
            n = export_daily_networks(load_dataframe(ds), ds, args.graphs_dir)
            print(f"{ds}: {n} daily graphs written")
    elif args.stage == "sequential":
        import pandas as pd

        table = pd.concat([sequential_consistency_table(load_dataframe(ds), ds) for ds in args.datasets])
        table.to_csv(RESULTS / "sequential_consistency.csv", index=False)
    elif args.stage == "labels":
        daily = label_daily_graphs(args.graphs_dir, args.datasets)
        if daily.empty:
            raise SystemExit(f"No GraphML files found in {args.graphs_dir} for {args.datasets}; run the 'graphs' stage first.")
        daily.to_csv(RESULTS / "daily_labels.csv", index=False)
        label_accounts(daily).to_csv(RESULTS / "account_labels.csv", index=False)
        print(daily.groupby(["dataset", "status"]).size())
    elif args.stage == "metrics":
        daily, accounts = build_metrics_tables(args.graphs_dir, args.datasets)
        if daily.empty:
            raise SystemExit(f"No GraphML files found in {args.graphs_dir} for {args.datasets}.")
        daily.to_csv(RESULTS / "daily_metrics.csv", index=False)
        accounts.to_csv(RESULTS / "account_metrics.csv", index=False)
        summary = summarize(split_branches(daily, accounts))
        summary.to_csv(RESULTS / "metrics_summary.csv")
        print(summary[["count", "mean", "50%", "max"]].round(3))
    elif args.stage == "values":
        values = collect_values(args.graphs_dir, args.datasets)
        if values.empty:
            raise SystemExit(f"No GraphML files found in {args.graphs_dir} for {args.datasets}.")
        values.to_csv(RESULTS / "values.csv", index=False)
        print(values.groupby(["dataset", "kind"]).size())
    elif args.stage == "ks":
        import pandas as pd

        needed = [RESULTS / f for f in ("values.csv", "daily_metrics.csv", "account_metrics.csv")]
        missing = [str(f) for f in needed if not f.exists()]
        if missing:
            raise SystemExit(f"Missing {missing}; run the 'values' and 'metrics' stages first.")
        ks_values(pd.read_csv(needed[0])).to_csv(RESULTS / "ks_values.csv", index=False)
        ks_metrics(pd.read_csv(needed[1]), pd.read_csv(needed[2])).to_csv(RESULTS / "ks_metrics.csv", index=False)
        print(pd.read_csv(RESULTS / "ks_values.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()