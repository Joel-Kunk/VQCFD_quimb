# VQCFD_quimb

Research code for classically emulating a variational quantum algorithm for
the one-dimensional viscous Burgers equation with
[Quimb](https://quimb.readthedocs.io/) tensor networks. The repository contains
the simulation runner, parameterized circuit families, initial-state presets,
gradient and expressibility experiments, tests, and the analysis notebooks used
for the master's thesis *Variational algorithms for partial differential
equations: Tensor network and quantum approaches*.

The nonlinear circuit construction follows the framework of Lubasch et al.,
[*Variational quantum algorithms for nonlinear problems*](https://doi.org/10.1103/PhysRevA.101.010301).
This repository is a research artifact and classical emulator; it is not a
production CFD solver or a hardware execution package.

## Quick start

The thesis environment used Python 3.13. Create an isolated environment and
install the pinned dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Run a tiny end-to-end calculation:

```bash
python smoke_test.py
```

The smoke test evolves the default (N=3, L=2) circuit on eight grid points for
ten timesteps. It normally takes only a few seconds and writes ignored test
output below `results/results_smoke/`. It checks that the preset can be loaded,
the noise-free optimizer completes, and the manifest and result files are written.
It is a runtime check, not a converged scientific experiment.

## Running simulations

`run.py` is the editable experiment runner. Its `main()` function has three
sweep layouts:

- `single`: run one `(N, L)` pair;
- `pairs`: run an explicit list of `(N, L)` pairs;
- `grid`: form all pairs from separate `ns` and `ls` lists.

The `common` dictionary controls the mode, circuit family, initial state, and
other settings shared by the selected runs. A circuit or initial state can be a
single name or a list, in which case the runner executes the Cartesian product
sequentially. Start the configured campaign with:

```bash
python run.py
```

The supported modes are:

| Mode | Purpose |
|---|---|
| `noise_free` | Exact expectation values with Quimb/Autograd and L-BFGS-B. |
| `adam_exact` | Exact expectation values with the custom Adam loop. |
| `adam_shots` | Shot-sampled expectation values with Adam. |
| `cobyla_shots` | Shot-sampled expectation values with COBYLA. |
| `gradient` | Sample parameter-shift gradients without time evolution. |
| `exp_only` | Compute expressibility and entangling-capability metrics only. |

Available circuit names are defined by `UNITARY_CIRCUITS` in
`functions/circuits_quimb.py`; available initial profiles are defined by
`INITIAL_STATES` in `functions/initial_states.py`. Passing `None` selects the
original `default` circuit and `sine` field.

Initial circuit parameters are loaded from `initial_params_presets.yaml` when a
matching `(N, L, circuit, initial state)` entry exists. Otherwise the runner
performs a fallback fit. `generate_initial_params_presets.py` contains the
checkpointed batch workflow used to create and validate additional presets.

## Output and analysis

Each run uses this layout:

```text
results/
└── results_<dir_label>/
    ├── manifest_<dir_label>_<label>.yaml
    ├── data_<label>/
    │   ├── values_<dir_label>_<label>.yaml
    │   ├── params_<dir_label>_<label>.npy
    │   ├── costs_<dir_label>_<label>.npy
    │   ├── times_<dir_label>_<label>.npy
    │   └── cost_iter_<dir_label>_<label>_<step>.npy
    └── figures_<label>/
```

Mode-specific files are added for gradient, expressibility, and shot-based
runs. Generated results and figures are intentionally excluded from Git.

- `analysis.py` contains reusable loading, comparison, table, and plotting
  functions.
- `analysis_results.ipynb` contains the time-evolution and cross-run analysis
  used for the thesis figures and tables.
- `analysis_variance.ipynb` contains the gradient-variance analysis.

For an archival release, publish the selected result files underlying the
thesis figures separately and keep their directory structure intact so the
notebooks can load them. Do not publish `.venv`, caches, or exploratory output.

## Contraction-path policy and known limitation

The release default is:

```python
optimize_contraction_paths = False
```

This uses Quimb's automatic `auto-hq` contraction selection for each current
network. The thesis results use this automatic behavior. The editable `run.py`
does not expose the experimental cached-path tuning options.

The repository retains an opt-in experiment that searches fixed Cotengra paths
for the five cost-function contractions and reuses them between optimizer
evaluations. It was not used for the reported thesis results because two bugs
were found during development:

1. In the initial implementation, embedded trainable gate matrices entered
   Quimb's global numeric-gate cache as Autograd objects. Complete
   differentiation graphs were then retained across evaluations, producing
   severe monotonic memory growth. The current code clears that cache after the
   five terms of one loss evaluation, which mitigates this first issue.
2. More importantly, a saved path is a sequence of positional tensor pairs.
   Reconstructing and simplifying an otherwise equivalent Quimb network did
   not always preserve the same tensor order. Applying the saved positions to
   the reordered network could therefore contract the wrong pairs. In observed
   full-staircase runs, a path rehearsed with a largest intermediate of
   (2^{14}) elements could be misapplied with an intermediate near (2^{44}),
   including failed allocation requests of 128 TiB. Disabling simplification
   reduced neither the ordering risk to zero nor the network cost sufficiently.

For that reason, cached positional paths remain experimental, disabled by
default, and emit a runtime warning when enabled. They should not be used for
scientific runs without canonical tensor ordering plus a topology and maximum-
intermediate guard. Existing files below `.contraction_paths/` are local
experimental caches and are not part of the release.

## Repository contents

```text
run.py                              editable simulation campaigns
smoke_test.py                       ten-step end-to-end runtime check
sim_config.py                       simulation configuration and validation
simulation_runner.py                execution and output workflow
sim_state.py                        evolving simulation state
functions/circuits_quimb.py         ansatz and Hadamard-test circuits
functions/functions_quimb.py        costs, expectations, and gradients
functions/initial_states.py         initial field profiles
functions/expr_entcap.py            circuit diagnostics
functions/plots.py                  simulation plotting
functions/contraction_paths.py      experimental cached-path implementation
analysis.py                         reusable analysis helpers
analysis_results.ipynb              thesis result analysis
analysis_variance.ipynb             variance analysis
initial_params_presets.yaml         fitted initial parameters
generate_initial_params_presets.py  preset-generation workflow
tests/                              unit and regression tests
```

## Reproducibility and citation

Before archiving the thesis version, commit the final code and notebooks, tag
that exact commit, create a release, and archive the release with a persistent
identifier such as a Zenodo DOI. Add the final repository URL, version, release
date, and DOI to `CITATION.cff`. Cite the version-specific archive rather than a
moving branch.

## License

The original source code and accompanying software documentation are licensed
under the [MIT License](LICENSE), copyright (c) 2026 Joel Kunkel. Third-party
dependencies retain their own licenses. Please cite the archived thesis release
using `CITATION.cff` when using this work in research.

Results distributed in a separate data archive should state their own data
license; this software license does not select a license for that deposit.

## Thesis software snapshot

The `thesis-v1.0` tag identifies the source and analysis snapshot accompanying
the thesis. Analysis notebooks retain all original cells and plot settings;
saved outputs are cleared. Run their setup cells before the required analysis
sections. Some cells are alternative plotting presets rather than a Run All
pipeline.

This software release includes source, runners, tests, dependency specifications,
analysis notebooks and helpers, and software documentation. Generated simulation
data, run manifests, figures, caches, environments, and local backups are excluded.
Restore selected result data separately using the relative paths in the notebooks.
The result-data archive and software DOI have not yet been published.
