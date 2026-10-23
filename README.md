# Eulerian Graphs for Bank Transaction Data Quality

Network Science project, deliverable #2. Checks the consistency of bank transaction flows
using Eulerian graphs and computes network science metrics on the daily and account networks.

## Setup

Open the project in VS Code and choose "Reopen in Container" (uses `.devcontainer/`).

Place the datasets in `data/` (private, not included in the repository):

- `data/pcpe_03.csv`
- `data/pcpe_04.csv`

## Running the experiments

Run everything from the project root, in this order:

```bash
python -m src.run graphs       # build one GraphML file per account and day
python -m src.run metrics      # N, E, d, <d>, density, components, average degree
python -m src.run values       # node balances and edge values by Eulerian status
python -m src.run ks           # KS tests
python -m src.run sequential   # sequential consistency before/after the Eulerian trail
python plot.py                 # figures
```

To use a single dataset, add `--datasets pcpe_03` to any `src.run` command.

## Outputs

- `transaction_networks/`: daily GraphML files, organized as `<dataset>/<account>/<date>.graphml`
- `results/`: CSV tables (metrics, KS tests, sequential consistency)
- `figures/`: plots

## Definitions

- A daily network is consistent if it is connected and pseudosymmetric, meaning it has an open or closed Eulerian trail.
- An account is consistent if all of its days are consistent.
- Metrics are computed on the undirected, unweighted graph. Diameter and average distance use the largest connected component.