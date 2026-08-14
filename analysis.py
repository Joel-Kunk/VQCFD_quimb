"""Shared loading, comparison, and plotting helpers for the analysis notebooks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import functions.circuits_quimb as fcq
import functions.initial_states as fist


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "results"

RUN_TABLE_COLUMNS = [
    "Run",
    "N",
    "L",
    "NumberOfParameters",
    "Var",
    "OverallGradientVariance",
    "OverlapGradientVariance",
    "AdvectionGradientVariance",
    "DiffusionGradientVariance",
    "Circuit1GradientVariance",
    "Circuit2GradientVariance",
    "Circuit3GradientVariance",
    "Circuit4GradientVariance",
    "Circuit5GradientVariance",
    "Expressibility",
    "EntanglingCapability",
    "Circuit",
    "InitialState",
    "R2",
    "Fidelity",
    "MSE",
    "RelativeL1Error",
    "InitialRelativeL2Error",
    "PropagatedInitialRelativeL2Error",
    "EvolutionRelativeL2Error",
    "RelativeL2Error",
    "RelativeLinfError",
    "RelativeDerivativeL2Error",
    "RelativeDerivativeLinfError",
    "RelativeMassError",
    "NormalizedMassDrift",
]


def configure_plots() -> None:
    """Apply the plotting defaults used by both notebooks."""
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.rcParams["axes.grid"] = True


def _safe_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _first_present(*mappings_and_keys: tuple[Mapping[str, Any], Sequence[str]], default=None):
    for mapping, keys in mappings_and_keys:
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                return value
    return default


@dataclass
class RunData:
    dir_label: str
    label: str
    full_label: str
    results_dir: Path
    data_dir: Path
    fig_dir: Path
    manifest: dict[str, Any]
    values: dict[str, Any]
    files: dict[str, Path]

    def load_array(self, key: str):
        path = self.files.get(key)
        if path is None or not path.exists():
            return None
        return np.load(path, allow_pickle=True)

    def metadata(self, key: str, default=None):
        return self.values.get(key, self.manifest.get(key, default))

    @property
    def n(self) -> int | None:
        value = self.metadata("n")
        return None if value is None else int(value)

    @property
    def l(self) -> int | None:
        value = self.metadata("l")
        return None if value is None else int(value)

    @property
    def mode(self) -> str | None:
        return self.metadata("mode")

    @property
    def circuit(self) -> str:
        keys = (
            "resolved_unitary_circuit",
            "unitary_circuit",
            "circuit",
            "circuit_name",
            "circuit_type",
            "circuit_variant",
            "circuit_used",
            "ansatz",
            "unitary_type",
            "initial_params_variant",
        )
        return str(
            _first_present(
                (self.values, keys),
                (self.manifest, keys),
                default="default",
            )
        )

    @property
    def initial_state(self) -> str:
        value = _first_present(
            (self.values, ("resolved_initial_state", "initial_state")),
            (self.manifest, ("resolved_initial_state", "initial_state")),
            default=fist.DEFAULT_INITIAL_STATE,
        )
        return fist.resolve_initial_state(value)

    @property
    def number_of_parameters(self) -> int | None:
        value = self.metadata("number_of_parameters")
        if value is not None:
            return int(value)
        if self.n is None or self.l is None:
            return None
        try:
            return fcq.num_unitary_parameters(self.n, self.l, self.circuit)
        except ValueError:
            return None

    @property
    def costs(self):
        return self.load_array("costs")

    @property
    def times(self):
        return self.load_array("times")

    @property
    def params(self):
        return self.load_array("params")

    @property
    def variance(self):
        arr = self.load_array("variance")
        if arr is not None:
            return float(np.asarray(arr).reshape(()))
        value = self.metadata("variance")
        return None if value is None else float(value)

    @property
    def gradient_circuit_derivatives(self):
        return self.load_array("gradient_circuit_derivatives")


def _run_paths(dir_label: str, label: str, results_root: Path = RESULTS_ROOT):
    results_dir = Path(results_root) / f"results_{dir_label}"
    data_dir = results_dir / f"data_{label}"
    fig_dir = results_dir / f"figures_{label}"
    full_label = f"{dir_label}_{label}"
    return results_dir, data_dir, fig_dir, full_label


def load_run(dir_label: str, label: str, results_root: Path = RESULTS_ROOT) -> RunData:
    results_dir, data_dir, fig_dir, full_label = _run_paths(dir_label, label, results_root)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    manifest_path = results_dir / f"manifest_{full_label}.yaml"
    values_path = data_dir / f"values_{full_label}.yaml"
    files = {
        "manifest": manifest_path,
        "values_yaml": values_path,
        "params": data_dir / f"params_{full_label}.npy",
        "costs": data_dir / f"costs_{full_label}.npy",
        "times": data_dir / f"times_{full_label}.npy",
        "variance": data_dir / f"variance_{full_label}.npy",
        "variance_grads": data_dir / f"variance_grads_{full_label}.npy",
        "gradient_circuit_derivatives": data_dir / f"gradient_circuit_derivatives_{full_label}.npy",
        "num_evals": data_dir / f"num_evals_{full_label}.npy",
    }
    return RunData(
        dir_label=dir_label,
        label=label,
        full_label=full_label,
        results_dir=results_dir,
        data_dir=data_dir,
        fig_dir=fig_dir,
        manifest=_safe_yaml(manifest_path),
        values=_safe_yaml(values_path),
        files=files,
    )


def load_runs(selection: Iterable[tuple[str, str]], results_root: Path = RESULTS_ROOT) -> list[RunData]:
    return [load_run(dir_label, label, results_root) for dir_label, label in selection]


def load_expr_run(dir_label: str, label: str, results_root: Path = RESULTS_ROOT) -> RunData:
    """Load an ``exp_only`` result stored alongside an ordinary run."""
    results_dir, data_dir, fig_dir, base_full_label = _run_paths(
        dir_label, label, results_root
    )
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    full_label = f"{base_full_label}_expr"
    manifest_path = results_dir / f"manifest_{full_label}.yaml"
    values_path = data_dir / f"values_{full_label}.yaml"
    if not manifest_path.exists() and not values_path.exists():
        raise FileNotFoundError(
            f"No expression-only result found for {base_full_label}: "
            f"expected {manifest_path} or {values_path}."
        )
    files = {
        "manifest": manifest_path,
        "values_yaml": values_path,
        "params": data_dir / f"params_{full_label}.npy",
        "costs": data_dir / f"costs_{full_label}.npy",
        "times": data_dir / f"times_{full_label}.npy",
        "variance": data_dir / f"variance_{full_label}.npy",
        "variance_grads": data_dir / f"variance_grads_{full_label}.npy",
        "gradient_circuit_derivatives": data_dir / f"gradient_circuit_derivatives_{full_label}.npy",
        "num_evals": data_dir / f"num_evals_{full_label}.npy",
    }
    return RunData(
        dir_label=dir_label,
        label=label,
        full_label=full_label,
        results_dir=results_dir,
        data_dir=data_dir,
        fig_dir=fig_dir,
        manifest=_safe_yaml(manifest_path),
        values=_safe_yaml(values_path),
        files=files,
    )


def load_expr_runs(
    selection: Iterable[tuple[str, str]],
    results_root: Path = RESULTS_ROOT,
) -> list[RunData]:
    return [load_expr_run(dir_label, label, results_root) for dir_label, label in selection]


def discover_expr_runs(
    results_root: Path = RESULTS_ROOT,
    include_full_runs: bool = True,
) -> list[RunData]:
    """Discover expression-only results and, optionally, metrics in full runs.

    When both variants exist for the same run folder, the independently saved
    ``*_expr`` result is preferred over the metrics embedded in a full run.
    """
    results_root = Path(results_root)
    runs = []
    expression_keys: set[tuple[str, str]] = set()
    for results_dir in sorted(results_root.glob("results_*")):
        dir_label = results_dir.name.removeprefix("results_")
        for data_dir in sorted(results_dir.glob("data_*")):
            label = data_dir.name.removeprefix("data_")
            base_full_label = f"{dir_label}_{label}"
            if (data_dir / f"values_{base_full_label}_expr.yaml").exists():
                runs.append(load_expr_run(dir_label, label, results_root))
                expression_keys.add((dir_label, label))

    if include_full_runs:
        for row in list_runs(results_root):
            key = (row["dir_label"], row["label"])
            if key in expression_keys:
                continue
            run = load_run(*key, results_root)
            if (
                run.metadata("expressibility") is not None
                or run.metadata("entangling_capability") is not None
            ):
                runs.append(run)

    runs.sort(
        key=lambda run: (
            run.n if run.n is not None else 10**9,
            run.l if run.l is not None else 10**9,
            run.full_label,
        )
    )
    return runs


def discover_simulation_runs(results_root: Path = RESULTS_ROOT) -> list[RunData]:
    """Load every time-evolution run that has a saved parameter trajectory."""
    return [
        load_run(row["dir_label"], row["label"], results_root)
        for row in list_runs(results_root)
        if row["mode"] not in {"variance", "gradient"} and row["has_params"]
    ]


def _filter_run_rows(
    rows: Sequence[Mapping[str, Any]],
    ns: Iterable[int] | int | None = None,
    ls: Iterable[int] | int | None = None,
    circuits: Iterable[str] | str | None = None,
    initial_states: Iterable[str] | str | None = None,
    modes: Iterable[str] | str | None = None,
) -> list[dict[str, Any]]:
    """Filter discovered run metadata by any saved configuration fields."""
    filtered = [dict(row) for row in rows]
    for key, selected in (
        ("n", ns),
        ("l", ls),
        ("circuit", circuits),
        ("initial_state", initial_states),
        ("mode", modes),
    ):
        if selected is None:
            continue
        allowed = {selected} if isinstance(selected, (str, int, np.integer)) else set(selected)
        filtered = [row for row in filtered if row.get(key) in allowed]
    return filtered


def list_runs(
    results_root: Path = RESULTS_ROOT,
    ns: Iterable[int] | int | None = None,
    ls: Iterable[int] | int | None = None,
    circuits: Iterable[str] | str | None = None,
    initial_states: Iterable[str] | str | None = None,
    modes: Iterable[str] | str | None = None,
) -> list[dict[str, Any]]:
    """Discover saved runs, optionally filtering by their configuration."""
    rows = []
    for results_dir in sorted(Path(results_root).glob("results_*")):
        dir_label = results_dir.name.removeprefix("results_")
        for data_dir in sorted(results_dir.glob("data_*")):
            label = data_dir.name.removeprefix("data_")
            run = load_run(dir_label, label, results_root)
            if not any(
                run.files[key].exists()
                for key in (
                    "manifest",
                    "values_yaml",
                    "params",
                    "variance",
                    "gradient_circuit_derivatives",
                )
            ):
                # The directory can contain only an independently named
                # expression result, which is handled by discover_expr_runs().
                continue
            rows.append(
                {
                    "dir_label": dir_label,
                    "label": label,
                    "full_label": run.full_label,
                    "mode": run.mode,
                    "n": run.n,
                    "l": run.l,
                    "circuit": run.circuit,
                    "initial_state": run.initial_state,
                    "number_of_parameters": run.number_of_parameters,
                    "shots": run.metadata("shots"),
                    "has_costs": run.files["costs"].exists(),
                    "has_params": run.files["params"].exists(),
                    "has_variance": run.files["variance"].exists() or run.variance is not None,
                    "has_gradients": run.files["gradient_circuit_derivatives"].exists(),
                    "path": str(data_dir),
                }
            )
    return _filter_run_rows(
        rows,
        ns=ns,
        ls=ls,
        circuits=circuits,
        initial_states=initial_states,
        modes=modes,
    )


def print_runs(runs: Sequence[Mapping[str, Any]] | None = None, limit: int = 200) -> None:
    runs = list_runs() if runs is None else runs
    table = pd.DataFrame.from_records(
        [
            {
                "N": row.get("n"),
                "L": row.get("l"),
                "Circuit": row.get("circuit"),
                "Initial state": row.get("initial_state"),
                "Mode": row.get("mode"),
                "Parameters": row.get("number_of_parameters"),
                "Saved": row.get("has_params"),
                "Variance": row.get("has_variance"),
            }
            for row in runs[:limit]
        ],
        columns=[
            "N",
            "L",
            "Circuit",
            "Initial state",
            "Mode",
            "Parameters",
            "Saved",
            "Variance",
        ],
    )
    if table.empty:
        print("No runs found.")
    else:
        print(table.to_string(index=False))
    print(f"count={len(runs)}")


def list_initial_param_presets(
    presets_path: Path | None = None,
    show_variants: bool = True,
    n: int | None = None,
    l: int | None = None,
    circuit: str | None = None,
    initial_state: str | None = None,
) -> list[dict[str, Any]]:
    """Print and return available entries from ``initial_params_presets.yaml``."""
    presets_path = presets_path or (REPO_ROOT / "initial_params_presets.yaml")
    if circuit is not None:
        circuit = fcq.resolve_unitary_circuit(circuit)
    if initial_state is not None:
        initial_state = fist.resolve_initial_state(initial_state)
    data = _safe_yaml(presets_path)
    if not data:
        print(f"No presets found at: {presets_path}")
        return []

    rows = []
    for n_key in sorted(data, key=lambda x: int(x)):
        n_int = int(n_key)
        if n is not None and n_int != int(n):
            continue
        for l_key in sorted(data[n_key] or {}, key=lambda x: int(x)):
            l_int = int(l_key)
            if l is not None and l_int != int(l):
                continue
            entry = (data[n_key] or {})[l_key] or {}
            default_params = entry.get("default") if isinstance(entry, dict) else entry
            variants = []
            state_presets = []
            if isinstance(entry, dict):
                for name, variant in (entry.get("variants") or {}).items():
                    params = variant.get("params") if isinstance(variant, dict) else variant
                    variants.append(
                        {
                            "name": name,
                            "len": len(params) if isinstance(params, list) else None,
                            "mse": variant.get("mse") if isinstance(variant, dict) else None,
                        }
                    )
                for state_name, circuits in (entry.get("initial_states") or {}).items():
                    if initial_state is not None and state_name != initial_state:
                        continue
                    for circuit_name, preset in (circuits or {}).items():
                        if circuit is not None and circuit_name != circuit:
                            continue
                        params = preset.get("params") if isinstance(preset, dict) else preset
                        state_presets.append(
                            {
                                "initial_state": state_name,
                                "circuit": circuit_name,
                                "len": len(params) if isinstance(params, list) else None,
                                "mse": preset.get("mse") if isinstance(preset, dict) else None,
                                "tries": preset.get("tries") if isinstance(preset, dict) else None,
                            }
                        )
            if (circuit is not None or initial_state is not None) and not state_presets:
                continue
            rows.append(
                {
                    "n": n_int,
                    "l": l_int,
                    "default_len": len(default_params) if isinstance(default_params, list) else None,
                    "default_mse": entry.get("default_mse") if isinstance(entry, dict) else None,
                    "variants": variants,
                    "state_presets": state_presets,
                }
            )

    if not rows:
        print(f"No presets found for filters n={n}, l={l}")
        return []

    for row in rows:
        line = f"N={row['n']}, L={row['l']} | default_len={row['default_len']}"
        if row["default_mse"] is not None:
            line += f" | default_mse={row['default_mse']}"
        print(line)
        if show_variants:
            for variant in row["variants"]:
                print(f"    variant={variant['name']} | len={variant['len']} | mse={variant['mse']}")
            for preset in row["state_presets"]:
                print(
                    f"    initial_state={preset['initial_state']} | circuit={preset['circuit']} "
                    f"| len={preset['len']} | mse={preset['mse']} | tries={preset['tries']}"
                )
    print(f"count={len(rows)} presets")
    return rows


def load_per_timestep_histories(run: RunData, prefix: str) -> list[tuple[int, np.ndarray]]:
    pattern = re.compile(rf"{re.escape(prefix)}_{re.escape(run.full_label)}_(\d+)\.npy$")
    entries = []
    for path in run.data_dir.glob(f"{prefix}_{run.full_label}_*.npy"):
        match = pattern.match(path.name)
        if match:
            entries.append((int(match.group(1)), path))
    entries.sort(key=lambda item: item[0])
    return [(index, np.load(path, allow_pickle=True)) for index, path in entries]


def normalize_cost_trace(trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=float).ravel()
    if trace.size == 0:
        return trace
    denominator = trace[0] - trace[-1]
    if abs(denominator) < 1e-15:
        return np.zeros_like(trace)
    return (trace - trace[-1]) / denominator


def mask_outliers(
    trace: np.ndarray,
    method: str = "iqr",
    factor: float = 3.0,
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    values = np.asarray(trace, dtype=float).copy()
    finite = np.isfinite(values)
    if finite.sum() < 4:
        return values
    finite_values = values[finite]
    if method == "percentile":
        low, high = np.percentile(finite_values, percentiles)
    elif method == "iqr":
        q1, q3 = np.percentile(finite_values, [25, 75])
        iqr = q3 - q1
        if iqr < 1e-15:
            return values
        low, high = q1 - factor * iqr, q3 + factor * iqr
    else:
        raise ValueError("method must be 'iqr' or 'percentile'")
    values[finite & ((values < low) | (values > high))] = np.nan
    return values


def plot_costs_and_times(run: RunData):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, values, title, ylabel in (
        (axes[0], run.costs, "Final cost per timestep", "cost"),
        (axes[1], run.times, "Runtime per timestep", "seconds"),
    ):
        if values is None:
            ax.text(0.5, 0.5, f"No {ylabel} file", ha="center", va="center")
        else:
            ax.plot(np.ravel(values), marker="o", ms=3)
        ax.set_title(f"{title} ({run.full_label})")
        ax.set_xlabel("timestep")
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig, axes


def plot_cost_histories(
    run: RunData,
    normalize: bool = True,
    remove_outliers: bool = True,
    outlier_method: str = "iqr",
    outlier_factor: float = 3.0,
    outlier_percentiles: tuple[float, float] = (1.0, 99.0),
    show_first_n_labels: int = 10,
):
    histories = load_per_timestep_histories(run, "cost_iter")
    if not histories:
        print("No cost_iter files found for this run.")
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    for timestep, values in histories:
        y = np.asarray(values, dtype=float).ravel()
        if normalize:
            y = normalize_cost_trace(y)
        if remove_outliers:
            y = mask_outliers(y, outlier_method, outlier_factor, outlier_percentiles)
        label = f"t={timestep}" if timestep < show_first_n_labels else None
        ax.plot(y, alpha=0.55, label=label)
    ax.set_title(f"cost_iter traces ({run.full_label})")
    ax.set_xlabel("optimizer iteration")
    ax.set_ylabel("relative cost (1=start, 0=end)" if normalize else "cost")
    if show_first_n_labels > 0:
        ax.legend(ncol=2, fontsize=8)
    return fig, ax


def plot_parameter_history(run: RunData, parameter_index: int = 0):
    params = run.params
    if params is None:
        print("No params file found for this run.")
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(params[:, parameter_index], marker="o", ms=3)
    ax.set_title(f"Parameter[{parameter_index}] across timesteps ({run.full_label})")
    ax.set_xlabel("timestep state index")
    ax.set_ylabel(f"params[:, {parameter_index}]")
    return fig, ax


def plot_shots(run: RunData):
    shots = run.values.get("shots_used_per_timestep")
    print("shots_used_total:", run.values.get("shots_used_total"))
    if shots is None:
        print("No shots_used_per_timestep metadata found for this run.")
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(shots, marker="o", ms=3)
    ax.set_title(f"Shots used per timestep ({run.full_label})")
    ax.set_xlabel("timestep")
    ax.set_ylabel("shots")
    return fig, ax


def plot_expr_entcap_vs_l(
    runs: Sequence[RunData],
    show_expressibility: bool = True,
    show_entangling_capability: bool = True,
    group_by_n: bool = True,
    use_log_y: bool = False,
    n_filter: int | Iterable[int] | None = None,
):
    if not show_expressibility and not show_entangling_capability:
        raise ValueError("At least one metric must be selected.")
    allowed_n = None if n_filter is None else ({int(n_filter)} if isinstance(n_filter, int) else {int(x) for x in n_filter})
    rows = []
    for run in runs:
        if run.n is None or run.l is None or (allowed_n is not None and run.n not in allowed_n):
            continue
        rows.append(
            {
                "run": run,
                "n": run.n,
                "l": run.l,
                "expressibility": run.metadata("expressibility"),
                "entangling_capability": run.metadata("entangling_capability"),
            }
        )
    if not rows:
        print(f"No runs found for n_filter={n_filter}.")
        return None

    metrics = []
    if show_expressibility:
        metrics.append(("expressibility", "Expressibility vs L"))
    if show_entangling_capability:
        metrics.append(("entangling_capability", "Entangling Capability vs L"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
    for ax, (metric, title) in zip(axes[0], metrics):
        available = [row for row in rows if row[metric] is not None]
        if not available:
            ax.text(0.5, 0.5, f"No {metric} data", ha="center", va="center")
        elif group_by_n:
            for n_value in sorted({row["n"] for row in available}):
                group = sorted((row for row in available if row["n"] == n_value), key=lambda row: row["l"])
                ax.plot([row["l"] for row in group], [float(row[metric]) for row in group], marker="o", label=f"N={n_value}")
            ax.legend()
        else:
            available.sort(key=lambda row: (row["l"], row["n"]))
            ax.plot([row["l"] for row in available], [float(row[metric]) for row in available], marker="o")
            for row in available:
                ax.annotate(row["run"].full_label, (row["l"], float(row[metric])), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("L")
        ax.set_ylabel(metric)
        if use_log_y:
            ax.set_yscale("log")
    fig.tight_layout()
    return fig, axes


def classical_evolution_from_initial_field(
    run: RunData,
    initial_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve one field with the classical update used by ``simulation_runner.py``."""
    if run.n is None:
        raise ValueError(f"Run {run.full_label} has no N metadata.")
    n_total = int(run.metadata("n_total", 2**run.n))
    t_total = float(run.metadata("t_total"))
    dt = float(run.metadata("dt"))
    mu = float(run.metadata("mu"))
    n_timesteps = int(run.metadata("n_timesteps", int(t_total / dt)))
    xs = np.linspace(0, 1, n_total)
    dx = xs[1] - xs[0]
    fields = np.zeros((n_total, n_timesteps + 1), dtype=float)
    initial = np.asarray(initial_field, dtype=float).reshape(-1)
    if initial.size != n_total:
        raise ValueError(
            f"Initial field for {run.full_label} has {initial.size} samples; "
            f"expected {n_total}."
        )
    fields[:, 0] = initial
    # Match the simulator's modular quantum shift operators.
    for timestep in range(n_timesteps):
        previous = fields[:, timestep]
        right = np.roll(previous, -1)
        left = np.roll(previous, 1)
        fields[:, timestep + 1] = (
            previous
            + (mu * dt / dx**2) * (right - 2 * previous + left)
            - (dt / (2 * dx)) * previous * (right - left)
        )
    return xs, fields


def classical_reference(run: RunData) -> tuple[np.ndarray, np.ndarray]:
    """Recreate the target classical trajectory used by ``simulation_runner.py``."""
    if run.n is None:
        raise ValueError(f"Run {run.full_label} has no N metadata.")
    n_total = int(run.metadata("n_total", 2**run.n))
    xs = np.linspace(0, 1, n_total)
    initial_field = fist.make_initial_field(xs, run.initial_state)
    return classical_evolution_from_initial_field(run, initial_field)


def encoded_initial_classical_reference(run: RunData) -> tuple[np.ndarray, np.ndarray]:
    """Classically evolve the field represented by the run's initial parameters."""
    _, initial_field = quantum_field(run, 0)
    return classical_evolution_from_initial_field(run, initial_field)


def _resolve_timestep(timestep: int, count: int) -> int:
    resolved = timestep if timestep >= 0 else count + timestep
    if not 0 <= resolved < count:
        raise IndexError(f"Timestep {timestep} is outside the available range 0..{count - 1}.")
    return resolved


def quantum_field(run: RunData, timestep: int = -1) -> tuple[int, np.ndarray]:
    params = run.params
    if params is None:
        raise FileNotFoundError(f"No parameter trajectory found for {run.full_label}.")
    if run.n is None or run.l is None:
        raise ValueError(f"Run {run.full_label} has incomplete N/L metadata.")
    index = _resolve_timestep(timestep, len(params))
    return index, _quantum_field_from_params(run, params[index])


def _quantum_field_from_params(run: RunData, params: np.ndarray) -> np.ndarray:
    """Reconstruct one real-space field from a saved parameter vector."""
    if run.n is None or run.l is None:
        raise ValueError(f"Run {run.full_label} has incomplete N/L metadata.")
    state_params = np.asarray(params, dtype=float)
    state = np.asarray(
        fcq.make_state_circuit(
            run.n,
            run.l,
            state_params[1:],
            parametrize=False,
            unitary_circuit=run.circuit,
        ).to_dense()
    ).reshape(-1)
    n_total = int(run.metadata("n_total", 2**run.n))
    field = state_params[0] * np.real_if_close(state[:n_total]).real
    return np.asarray(field, dtype=float)


def quantum_evolution(run: RunData) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct every saved quantum field and its physical time."""
    params = run.params
    if params is None:
        raise FileNotFoundError(f"No parameter trajectory found for {run.full_label}.")
    fields = np.column_stack(
        [_quantum_field_from_params(run, state_params) for state_params in params]
    )
    times = np.arange(fields.shape[1], dtype=float) * float(run.metadata("dt"))
    return times, fields


def _relative_norm_error(
    reference: np.ndarray,
    approximation: np.ndarray,
    order: int | float,
) -> float:
    denominator = float(np.linalg.norm(reference, ord=order))
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(np.linalg.norm(reference - approximation, ord=order) / denominator)


def _periodic_centered_derivative(field: np.ndarray, dx: float) -> np.ndarray:
    if dx <= 0:
        raise ValueError("dx must be positive.")
    values = np.asarray(field, dtype=float).reshape(-1)
    if values.size < 3:
        raise ValueError("At least three spatial samples are required for a derivative.")
    return (np.roll(values, -1) - np.roll(values, 1)) / (2 * dx)


def _discrete_mass(field: np.ndarray, dx: float) -> float:
    return float(dx * np.sum(np.asarray(field, dtype=float)))


def _relative_mass_difference(
    reference: np.ndarray,
    approximation: np.ndarray,
    dx: float,
) -> float:
    reference_mass = _discrete_mass(reference, dx)
    approximation_mass = _discrete_mass(approximation, dx)
    reference_scale = float(dx * np.sum(np.abs(reference)))
    zero_tolerance = 100 * np.finfo(float).eps * max(1.0, reference_scale)
    if abs(reference_mass) <= zero_tolerance:
        return float("nan")
    return float(abs(approximation_mass - reference_mass) / abs(reference_mass))


def comparison_metrics(
    classical: np.ndarray,
    quantum: np.ndarray,
    dx: float = 1.0,
) -> dict[str, float]:
    classical = np.asarray(classical, dtype=float).reshape(-1)
    quantum = np.asarray(quantum, dtype=float).reshape(-1)
    if classical.shape != quantum.shape:
        raise ValueError(f"Field shapes do not match: {classical.shape} != {quantum.shape}")
    residual = classical - quantum
    mse = float(np.mean(np.abs(residual) ** 2))
    total = float(np.sum(np.abs(classical - np.mean(classical)) ** 2))
    r2 = float(1.0 - np.sum(np.abs(residual) ** 2) / total) if total > 0 else float("nan")
    norm_product = float(np.vdot(classical, classical).real * np.vdot(quantum, quantum).real)
    fidelity = float(np.abs(np.vdot(classical, quantum)) ** 2 / norm_product) if norm_product > 0 else float("nan")
    classical_derivative = _periodic_centered_derivative(classical, dx)
    quantum_derivative = _periodic_centered_derivative(quantum, dx)
    return {
        "MSE": mse,
        "R2": r2,
        "Fidelity": fidelity,
        "RelativeL1Error": _relative_norm_error(classical, quantum, 1),
        "RelativeL2Error": _relative_norm_error(classical, quantum, 2),
        "RelativeLinfError": _relative_norm_error(classical, quantum, np.inf),
        "RelativeDerivativeL2Error": _relative_norm_error(
            classical_derivative, quantum_derivative, 2
        ),
        "RelativeDerivativeLinfError": _relative_norm_error(
            classical_derivative, quantum_derivative, np.inf
        ),
        "RelativeMassError": _relative_mass_difference(classical, quantum, dx),
    }


def compare_run_timestep(run: RunData, timestep: int = -1) -> dict[str, Any]:
    index, quantum = quantum_field(run, timestep)
    xs, classical_fields = classical_reference(run)
    if index >= classical_fields.shape[1]:
        raise IndexError(
            f"Quantum timestep {index} has no matching classical state; "
            f"classical range is 0..{classical_fields.shape[1] - 1}."
        )
    classical = classical_fields[:, index]
    dx = float(xs[1] - xs[0])
    metrics = comparison_metrics(classical, quantum, dx=dx)
    if index == 0:
        quantum_initial = quantum
    else:
        _, quantum_initial = quantum_field(run, 0)
    _, encoded_classical_fields = classical_evolution_from_initial_field(
        run, quantum_initial
    )
    encoded_classical = encoded_classical_fields[:, index]
    metrics["InitialRelativeL2Error"] = _relative_norm_error(
        classical_fields[:, 0], quantum_initial, 2
    )
    metrics["PropagatedInitialRelativeL2Error"] = _relative_norm_error(
        classical, encoded_classical, 2
    )
    target_norm = float(np.linalg.norm(classical, ord=2))
    metrics["EvolutionRelativeL2Error"] = (
        float(np.linalg.norm(quantum - encoded_classical, ord=2) / target_norm)
        if target_norm > np.finfo(float).eps
        else float("nan")
    )
    metrics["NormalizedMassDrift"] = _relative_mass_difference(
        quantum_initial, quantum, dx
    )
    return {
        "run": run,
        "timestep": index,
        "time": index * float(run.metadata("dt")),
        "x": xs,
        "classical": classical,
        "encoded_classical": encoded_classical,
        "quantum": quantum,
        **metrics,
    }


def plot_run_timestep_comparison(run: RunData, timestep: int = -1, ax=None):
    comparison = compare_run_timestep(run, timestep)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    ax.plot(comparison["x"], comparison["classical"], label="Desired", linewidth=2)
    ax.plot(
        comparison["x"],
        comparison["encoded_classical"],
        color="0.45",
        alpha=0.65,
        label="Classical",
        linewidth=2,
    )
    ax.plot(comparison["x"], comparison["quantum"], "--", label="Quantum", linewidth=2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Field value")
    ax.grid(False)
    ax.legend()
    metric_text = (
        f"Rel L2 err = {comparison['RelativeL2Error']:.2e}\n"
        f"Evo rel L2 err = {comparison['EvolutionRelativeL2Error']:.2e}"
    )
    ax.text(
        0.02,
        0.02,
        metric_text,
        transform=ax.transAxes,
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    return comparison, ax, fig


def plot_run_time_evolution(
    run: RunData,
    fields: str = "both",
    timestep_stride: int = 1,
    cmap: str = "viridis",
    ax=None,
):
    """Plot all classical fields, quantum fields, or both on one set of axes."""
    fields = str(fields).strip().lower()
    if fields not in {"classical", "quantum", "both"}:
        raise ValueError("fields must be 'classical', 'quantum', or 'both'.")
    if not isinstance(timestep_stride, (int, np.integer)) or timestep_stride < 1:
        raise ValueError("timestep_stride must be a positive integer.")

    xs, classical = classical_reference(run)
    dt = float(run.metadata("dt"))
    classical_times = np.arange(classical.shape[1], dtype=float) * dt
    quantum_times = quantum = None
    if fields in {"quantum", "both"}:
        quantum_times, quantum = quantum_evolution(run)

    visible_times = []
    if fields in {"classical", "both"}:
        visible_times.append(classical_times)
    if quantum_times is not None:
        visible_times.append(quantum_times)
    max_time = max(float(times[-1]) for times in visible_times if len(times))
    norm = plt.Normalize(vmin=0.0, vmax=max_time if max_time > 0 else 1.0)
    color_map = plt.get_cmap(cmap)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = ax.figure

    def plot_trajectory(values, times, linestyle, alpha):
        for index in range(0, values.shape[1], int(timestep_stride)):
            ax.plot(
                xs,
                values[:, index],
                color=color_map(norm(times[index])),
                linestyle=linestyle,
                linewidth=1.2,
                alpha=alpha,
            )

    if fields in {"classical", "both"}:
        plot_trajectory(classical, classical_times, "-", 0.65)
        ax.plot([], [], color="black", linestyle="-", label="Classical")
    if fields in {"quantum", "both"}:
        plot_trajectory(quantum, quantum_times, "--", 0.8)
        ax.plot([], [], color="black", linestyle="--", label="Quantum")

    ax.set_xlabel("Position")
    ax.set_ylabel("Field value")
    ax.grid(False)
    ax.legend()
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=color_map),
        ax=ax,
        label="Time",
    )
    evolution = {
        "x": xs,
        "classical_times": classical_times,
        "classical": classical,
        "quantum_times": quantum_times,
        "quantum": quantum,
    }
    return evolution, ax, fig


def _variance_gradient_samples(item: RunData | Mapping[str, Any]) -> np.ndarray | None:
    if isinstance(item, RunData):
        path = item.files.get("variance_grads")
    else:
        path = item.get("variance_grads_path")
        if path is None and item.get("data_dir") is not None and item.get("full_label") is not None:
            path = Path(item["data_dir"]) / f"variance_grads_{item['full_label']}.npy"
    if path is None or not Path(path).exists():
        return None
    return np.asarray(np.load(path, allow_pickle=True), dtype=float).ravel()


def _variance_lookup(variance_runs: Iterable[RunData | Mapping[str, Any]] | None):
    exact_items: defaultdict[tuple[int, int, str], list[tuple[float, int | None, np.ndarray | None]]] = defaultdict(list)
    by_n_l: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    for item in variance_runs or []:
        if isinstance(item, RunData):
            n, l, circuit, variance = item.n, item.l, item.circuit, item.variance
            num_samples = item.metadata("variance_num_samples")
        else:
            n, l = item.get("n"), item.get("l")
            circuit = str(item.get("circuit", "default"))
            variance = item.get("variance")
            num_samples = item.get("variance_num_samples")
        if n is None or l is None or variance is None:
            continue
        exact_items[(int(n), int(l), circuit)].append(
            (float(variance), None if num_samples is None else int(num_samples), _variance_gradient_samples(item))
        )
        by_n_l[(int(n), int(l))].append(float(variance))
    exact = {}
    for key, items in exact_items.items():
        gradients = [item[2] for item in items]
        if gradients and all(values is not None for values in gradients):
            exact[key] = float(np.var(np.concatenate(gradients)))
            continue
        weights = [item[1] for item in items]
        exact[key] = float(
            np.average(
                [item[0] for item in items],
                weights=weights if weights and all(weight is not None for weight in weights) else None,
            )
        )
    return exact, by_n_l


RUN_TABLE_GRADIENT_VARIANCE_COMPONENTS = {
    "OverallGradientVariance": "total",
    "OverlapGradientVariance": "contribution_1",
    "DiffusionGradientVariance": "contribution_2",
    "AdvectionGradientVariance": "contribution_3",
    "Circuit1GradientVariance": "circuit_1",
    "Circuit2GradientVariance": "circuit_2",
    "Circuit3GradientVariance": "circuit_3",
    "Circuit4GradientVariance": "circuit_4",
    "Circuit5GradientVariance": "circuit_5",
}


def _gradient_variance_lookup(
    gradient_runs: Iterable[RunData | Mapping[str, Any]] | None,
) -> dict[tuple[int, int, str, str], dict[str, float]]:
    """Index total, combined-term, and individual-circuit gradient variances.

    Samples from repeated gradient runs with the same configuration are
    concatenated before each variance is calculated.
    """
    grouped: defaultdict[
        tuple[int, int, str, str],
        dict[str, list[np.ndarray]],
    ] = defaultdict(lambda: defaultdict(list))
    for item in gradient_runs or []:
        row = (
            load_gradient_run(item.dir_label, item.label, item.results_dir.parent)
            if isinstance(item, RunData)
            else item
        )
        n, l = row.get("n"), row.get("l")
        if n is None or l is None:
            continue
        circuit = str(row.get("circuit", "default"))
        initial_state = fist.resolve_initial_state(
            row.get("initial_state", fist.DEFAULT_INITIAL_STATE)
        )
        components = load_gradient_components(row)
        key = (int(n), int(l), circuit, initial_state)
        for column, component in RUN_TABLE_GRADIENT_VARIANCE_COMPONENTS.items():
            grouped[key][column].append(
                np.asarray(components[component], dtype=float).ravel()
            )

    return {
        key: {
            column: float(np.var(np.concatenate(component_samples[column])))
            for column in RUN_TABLE_GRADIENT_VARIANCE_COMPONENTS
        }
        for key, component_samples in grouped.items()
    }


def _expression_metric_lookup(
    expression_runs: Iterable[RunData | Mapping[str, Any]] | None,
) -> dict[tuple[int, int, str], dict[str, float]]:
    """Index independently saved expression metrics by circuit configuration.

    Expressibility and entangling capability depend on the ansatz rather than
    the simulation's initial state, so the lookup key is ``(N, L, Circuit)``.
    If several independent metric runs share a key, their estimates are
    averaged, weighted by ``expr_entcap_samples`` when every run records it.
    """
    grouped: defaultdict[
        tuple[int, int, str],
        list[tuple[float | None, float | None, int | None]],
    ] = defaultdict(list)
    for item in expression_runs or []:
        if isinstance(item, RunData):
            n, layers, circuit = item.n, item.l, item.circuit
            expressibility = item.metadata("expressibility")
            entangling_capability = item.metadata("entangling_capability")
            samples = item.metadata("expr_entcap_samples")
        else:
            values = item.get("values") or {}
            manifest = item.get("manifest") or {}
            n = _first_present((item, ("n",)), (values, ("n",)), (manifest, ("n",)))
            layers = _first_present((item, ("l",)), (values, ("l",)), (manifest, ("l",)))
            circuit = _first_present(
                (item, ("circuit", "resolved_unitary_circuit", "unitary_circuit")),
                (values, ("resolved_unitary_circuit", "unitary_circuit", "circuit")),
                (manifest, ("resolved_unitary_circuit", "unitary_circuit", "circuit")),
                default=fcq.DEFAULT_UNITARY_CIRCUIT,
            )
            expressibility = _first_present(
                (item, ("expressibility",)),
                (values, ("expressibility",)),
                (manifest, ("expressibility",)),
            )
            entangling_capability = _first_present(
                (item, ("entangling_capability",)),
                (values, ("entangling_capability",)),
                (manifest, ("entangling_capability",)),
            )
            samples = _first_present(
                (item, ("expr_entcap_samples",)),
                (values, ("expr_entcap_samples",)),
                (manifest, ("expr_entcap_samples",)),
            )
        if n is None or layers is None or circuit is None:
            continue
        if expressibility is None and entangling_capability is None:
            continue
        grouped[(int(n), int(layers), str(circuit))].append(
            (
                None if expressibility is None else float(expressibility),
                None if entangling_capability is None else float(entangling_capability),
                None if samples is None else int(samples),
            )
        )

    lookup = {}
    for key, estimates in grouped.items():
        metrics = {}
        for column, metric_index in (
            ("Expressibility", 0),
            ("EntanglingCapability", 1),
        ):
            available = [estimate for estimate in estimates if estimate[metric_index] is not None]
            if not available:
                continue
            weights = [estimate[2] for estimate in available]
            metrics[column] = float(
                np.average(
                    [estimate[metric_index] for estimate in available],
                    weights=(
                        weights
                        if weights and all(weight is not None and weight > 0 for weight in weights)
                        else None
                    ),
                )
            )
        lookup[key] = metrics
    return lookup


def build_run_records(
    runs: Sequence[RunData],
    variance_runs: Iterable[RunData | Mapping[str, Any]] | None = None,
    circuit_overrides: Mapping[str, str] | None = None,
    gradient_runs: Iterable[RunData | Mapping[str, Any]] | None = None,
    expression_runs: Iterable[RunData | Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build per-run final-state metrics and metadata records.

    Separate variance-mode runs can be supplied through ``variance_runs``. They are
    matched by ``(N, L, Circuit)`` and then by ``(N, L)`` when unambiguous. Repeated
    variance runs for the same configuration are averaged. Raw gradient runs are
    matched exactly by ``(N, L, Circuit, InitialState)``; repeated matching runs
    are combined at the sample level before their variances are calculated.
    Independently saved expression-only runs are matched by ``(N, L, Circuit)``
    and take priority over metrics embedded in the simulation run.
    """
    exact_variance, n_l_variance = _variance_lookup(variance_runs)
    gradient_variances = _gradient_variance_lookup(gradient_runs)
    expression_metrics = _expression_metric_lookup(expression_runs)
    overrides = circuit_overrides or {}
    records = []
    for run in runs:
        circuit = str(overrides.get(run.full_label, run.circuit))
        variance = run.variance
        if variance is None and run.n is not None and run.l is not None:
            variance = exact_variance.get((run.n, run.l, circuit))
            candidates = n_l_variance.get((run.n, run.l), [])
            if variance is None and len(candidates) == 1:
                variance = candidates[0]
        matched_gradient_variances = gradient_variances.get(
            (run.n, run.l, circuit, run.initial_state), {}
        )
        matched_expression_metrics = expression_metrics.get(
            (run.n, run.l, circuit), {}
        )
        metrics = {
            key: np.nan
            for key in (
                "R2",
                "Fidelity",
                "MSE",
                "RelativeL1Error",
                "InitialRelativeL2Error",
                "PropagatedInitialRelativeL2Error",
                "EvolutionRelativeL2Error",
                "RelativeL2Error",
                "RelativeLinfError",
                "RelativeDerivativeL2Error",
                "RelativeDerivativeLinfError",
                "RelativeMassError",
                "NormalizedMassDrift",
            )
        }
        if run.params is not None:
            metrics.update({key: value for key, value in compare_run_timestep(run, -1).items() if key in metrics})
        records.append(
            {
                "Run": run.full_label,
                "N": run.n,
                "L": run.l,
                "NumberOfParameters": run.number_of_parameters,
                "Var": np.nan if variance is None else float(variance),
                **{
                    column: matched_gradient_variances.get(column, np.nan)
                    for column in RUN_TABLE_GRADIENT_VARIANCE_COMPONENTS
                },
                "Expressibility": matched_expression_metrics.get(
                    "Expressibility",
                    _float_or_nan(run.metadata("expressibility")),
                ),
                "EntanglingCapability": matched_expression_metrics.get(
                    "EntanglingCapability",
                    _float_or_nan(run.metadata("entangling_capability")),
                ),
                "Circuit": circuit,
                "InitialState": run.initial_state,
                **metrics,
            }
        )
    return records


def _float_or_nan(value) -> float:
    return float(value) if value is not None else float("nan")


def build_run_table(
    runs: Sequence[RunData],
    variance_runs: Iterable[RunData | Mapping[str, Any]] | None = None,
    circuit_overrides: Mapping[str, str] | None = None,
    gradient_runs: Iterable[RunData | Mapping[str, Any]] | None = None,
    expression_runs: Iterable[RunData | Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        build_run_records(
            runs,
            variance_runs=variance_runs,
            circuit_overrides=circuit_overrides,
            gradient_runs=gradient_runs,
            expression_runs=expression_runs,
        ),
        columns=RUN_TABLE_COLUMNS,
    )


def _as_table(table: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return table.copy() if isinstance(table, pd.DataFrame) else pd.DataFrame(table)


def filter_run_table(
    table: pd.DataFrame | Sequence[Mapping[str, Any]],
    ns: Iterable[int] | int | None = None,
    ls: Iterable[int] | int | None = None,
    circuits: Iterable[str] | str | None = None,
    initial_states: Iterable[str] | str | None = None,
) -> pd.DataFrame:
    filtered = _as_table(table)
    for column, selected in (
        ("N", ns),
        ("L", ls),
        ("Circuit", circuits),
        ("InitialState", initial_states),
    ):
        if selected is None:
            continue
        allowed = {selected} if isinstance(selected, (str, int, np.integer)) else set(selected)
        filtered = filtered[filtered[column].isin(allowed)]
    return filtered.reset_index(drop=True)


def plot_run_table(
    table: pd.DataFrame | Sequence[Mapping[str, Any]],
    x: str,
    y: str | Sequence[str],
    ns: Iterable[int] | int | None = None,
    ls: Iterable[int] | int | None = None,
    circuits: Iterable[str] | str | None = None,
    initial_states: Iterable[str] | str | None = None,
    group_by: str | Sequence[str] | None = None,
    kind: str = "line",
    log_y: bool = False,
    ax=None,
    x_label: str | None = None,
    y_label: str | None = None,
    x_limits: tuple[float | None, float | None] | None = None,
    y_limits: tuple[float | None, float | None] | None = None,
):
    """Plot table columns with optional filtering, grouping, and axis limits."""
    filtered = filter_run_table(
        table,
        ns=ns,
        ls=ls,
        circuits=circuits,
        initial_states=initial_states,
    )
    y_columns = [y] if isinstance(y, str) else list(y)
    if not y_columns:
        raise ValueError("y must contain at least one column.")
    if any(not isinstance(column, str) for column in y_columns):
        raise TypeError("Every y column must be a string.")
    y_columns = list(dict.fromkeys(y_columns))
    for column in (x, *y_columns):
        if column not in filtered.columns:
            raise KeyError(f"Unknown column {column!r}. Available columns: {list(filtered.columns)}")
    group_columns = (
        [] if group_by is None
        else [group_by] if isinstance(group_by, str)
        else list(group_by)
    )
    missing_groups = [column for column in group_columns if column not in filtered.columns]
    if missing_groups:
        raise KeyError(f"Unknown group_by columns: {missing_groups}.")
    filtered = (
        filtered.dropna(subset=[x])
        .dropna(subset=y_columns, how="all")
        .reset_index(drop=True)
    )
    if filtered.empty:
        raise ValueError("No rows remain after filtering and dropping missing x/y values.")
    if kind not in {"line", "scatter"}:
        raise ValueError("kind must be 'line' or 'scatter'")
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    multiple_y = len(y_columns) > 1
    for y_column in y_columns:
        metric_table = filtered.dropna(subset=[y_column])
        if group_columns:
            group_key = group_columns[0] if len(group_columns) == 1 else group_columns
            grouped = metric_table.groupby(group_key, dropna=False, sort=True)
        else:
            grouped = [(None, metric_table)]
        for group_value, group in grouped:
            group = group.sort_values(x, kind="stable")
            label_parts = [y_column] if multiple_y else []
            if group_columns:
                values = group_value if isinstance(group_value, tuple) else (group_value,)
                label_parts.extend(
                    f"{column}={value}"
                    for column, value in zip(group_columns, values)
                )
            label = ", ".join(label_parts) or None
            if kind == "scatter":
                ax.scatter(group[x], group[y_column], label=label)
            else:
                ax.plot(group[x], group[y_column], marker="o", label=label)
    ax.set_xlabel(x if x_label is None else x_label)
    default_y_label = y_columns[0] if not multiple_y else "Value"
    ax.set_ylabel(default_y_label if y_label is None else y_label)
    if log_y:
        ax.set_yscale("log")
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if group_columns or multiple_y:
        ax.legend()
    return filtered, ax, fig


def load_gradient_run(dir_label: str, label: str, results_root: Path = RESULTS_ROOT) -> dict[str, Any]:
    """Load the metadata and array paths for one gradient-mode run."""
    run = load_run(dir_label, label, results_root)
    n = run.n
    dx = run.metadata("gradient_dx")
    if dx is None and n is not None:
        dx = 1.0 / (2**n - 1)
    return {
        "dir_label": dir_label,
        "label": label,
        "full_label": run.full_label,
        "data_dir": run.data_dir,
        "values": run.values,
        "manifest": run.manifest,
        "n": run.n,
        "l": run.l,
        "mode": run.mode,
        "circuit": run.circuit,
        "initial_state": run.initial_state,
        "number_of_parameters": run.number_of_parameters,
        "gradient_tries": run.metadata("gradient_tries"),
        "gradient_sweeps": run.metadata("gradient_sweeps"),
        "gradient_num_samples": run.metadata("gradient_num_samples"),
        "gradient_num_values_saved": run.metadata("gradient_num_values_saved"),
        "gradient_runtime_s": run.metadata("gradient_runtime_s"),
        "dt": run.metadata("dt"),
        "mu": run.metadata("mu"),
        "dx": dx,
        "gradient_lbd0": run.metadata("gradient_lbd0", 1.0),
        "gradient_lbd0_t": run.metadata("gradient_lbd0_t", 1.0),
        "gradient_circuit_labels": run.metadata("gradient_circuit_labels"),
        "gradient_parameter_labels": run.metadata("gradient_parameter_labels"),
        "gradient_circuit_derivatives_path": run.files["gradient_circuit_derivatives"],
    }


def load_gradient_circuit_derivatives(
    item: RunData | Mapping[str, Any],
) -> np.ndarray:
    """Load the raw array with axes ``(circuit, parameter, sweep)``."""
    if isinstance(item, RunData):
        row = load_gradient_run(item.dir_label, item.label, item.results_dir.parent)
    else:
        row = item
    number_of_parameters = row.get("number_of_parameters")
    if number_of_parameters is None:
        raise ValueError(f"Run {row.get('full_label', '<unknown>')} has no parameter count.")
    path_value = row.get("gradient_circuit_derivatives_path")
    if path_value is None:
        raise FileNotFoundError(
            f"Run {row.get('full_label', '<unknown>')} has no circuit-derivative path."
        )
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Circuit-derivative array not found: {path}")
    derivatives = np.asarray(np.load(path), dtype=float)
    expected_parameters = int(number_of_parameters)
    if derivatives.ndim != 3 or derivatives.shape[:2] != (5, expected_parameters):
        raise ValueError(
            f"Circuit-derivative array {path} must have shape "
            f"(5, {expected_parameters}, sweeps), received {derivatives.shape}."
        )
    return derivatives


def load_gradient_components(
    item: RunData | Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Derive total, contribution, and individual-circuit gradient matrices."""
    row = (
        load_gradient_run(item.dir_label, item.label, item.results_dir.parent)
        if isinstance(item, RunData)
        else item
    )
    raw = load_gradient_circuit_derivatives(row)
    dc1, dc2, dc3, dc4, dc5 = raw
    contribution_1 = dc1
    contribution_2 = dc2 - 2 * dc1 + dc3
    contribution_3 = dc4 - dc5

    dt = float(row["dt"])
    mu = float(row["mu"])
    dx = float(row["dx"])
    lbd0 = float(row.get("gradient_lbd0", 1.0))
    lbd0_t = float(row.get("gradient_lbd0_t", 1.0))
    lbd0_t_conj = np.conjugate(lbd0_t)
    total = -2 * np.real(
        lbd0_t_conj
        * lbd0
        * (
            contribution_1
            + (dt * mu / dx**2) * contribution_2
            - (dt * lbd0_t_conj / (2 * dx)) * contribution_3
        )
    )
    return {
        "total": total,
        "contribution_1": contribution_1,
        "contribution_2": contribution_2,
        "contribution_3": contribution_3,
        "circuit_1": dc1,
        "circuit_2": dc2,
        "circuit_3": dc3,
        "circuit_4": dc4,
        "circuit_5": dc5,
    }


def discover_gradient_runs(results_root: Path = RESULTS_ROOT) -> list[dict[str, Any]]:
    """Discover runs produced by the new raw-gradient mode."""
    rows = []
    for row in list_runs(results_root):
        if row["has_gradients"]:
            rows.append(load_gradient_run(row["dir_label"], row["label"], results_root))
    rows.sort(
        key=lambda row: (
            row["n"] if row["n"] is not None else 10**9,
            row["l"] if row["l"] is not None else 10**9,
            row["full_label"],
        )
    )
    return rows


def print_gradient_runs(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        actual = row.get("gradient_num_samples")
        requested = row.get("gradient_tries")
        print(
            f"{row['full_label']:<32} N={str(row['n']):<3} L={str(row['l']):<3} "
            f"circuit={str(row.get('circuit')):<20} gradients={actual}/{requested}"
        )
    print("count=", len(rows))


GRADIENT_EXTRA_COMPONENTS = (
    "contribution_1",
    "contribution_2",
    "contribution_3",
    "circuit_1",
    "circuit_2",
    "circuit_3",
    "circuit_4",
    "circuit_5",
)
GRADIENT_STATISTICS = ("mean_gradient", "mean_gradient_norm", "variance")


def _gradient_statistics(gradients: np.ndarray, per_parameter: bool) -> dict[str, float]:
    norm = np.abs(gradients) if per_parameter else np.linalg.norm(gradients, axis=0)
    return {
        "mean_gradient": float(np.mean(gradients)),
        "mean_gradient_norm": float(np.mean(norm)),
        "variance": float(np.var(gradients)),
    }


def _add_component_statistics(
    record: dict[str, Any],
    components: Mapping[str, np.ndarray],
    per_parameter: bool,
) -> None:
    record.update(_gradient_statistics(components["total"], per_parameter))
    for component in GRADIENT_EXTRA_COMPONENTS:
        for statistic, value in _gradient_statistics(
            components[component], per_parameter
        ).items():
            record[f"{component}_{statistic}"] = value


def _gradient_statistic_columns() -> list[str]:
    return [
        f"{component}_{statistic}"
        for component in GRADIENT_EXTRA_COMPONENTS
        for statistic in GRADIENT_STATISTICS
    ]


def gradient_run_table(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Summarize total, contribution, and circuit gradients for each run."""
    records = []
    for row in rows:
        components = load_gradient_components(row)
        gradients = components["total"]
        dt = float(row["dt"])
        mu = float(row["mu"])
        dx = float(row["dx"])
        record = {
            "full_label": row["full_label"],
            "n": row.get("n"),
            "l": row.get("l"),
            "circuit": row.get("circuit"),
            "initial_state": row.get("initial_state"),
            "dt": dt,
            "mu": mu,
            "dx": dx,
            "diffusion_number": mu * dt / dx**2,
            "number_of_parameters": gradients.shape[0],
            "gradient_sweeps": gradients.shape[1],
            "gradient_num_samples": gradients.size,
            "gradient_num_values_saved": 5 * gradients.size,
            "gradient_tries_requested": row.get("gradient_tries"),
            "gradient_runtime_s": row.get("gradient_runtime_s"),
        }
        _add_component_statistics(record, components, per_parameter=False)
        records.append(record)
    columns = [
        "full_label",
        "n",
        "l",
        "circuit",
        "initial_state",
        "dt",
        "mu",
        "dx",
        "diffusion_number",
        "number_of_parameters",
        "gradient_sweeps",
        "gradient_num_samples",
        "gradient_num_values_saved",
        "gradient_tries_requested",
        "mean_gradient",
        "mean_gradient_norm",
        "variance",
        *_gradient_statistic_columns(),
        "gradient_runtime_s",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def gradient_parameter_table(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Summarize all reconstructed gradient components for each parameter."""
    records = []
    for row in rows:
        components = load_gradient_components(row)
        gradients = components["total"]
        labels = row.get("gradient_parameter_labels") or []
        dt = float(row["dt"])
        mu = float(row["mu"])
        dx = float(row["dx"])
        for parameter_index in range(gradients.shape[0]):
            parameter_components = {
                name: values[parameter_index]
                for name, values in components.items()
            }
            record = {
                "full_label": row["full_label"],
                "n": row.get("n"),
                "l": row.get("l"),
                "circuit": row.get("circuit"),
                "initial_state": row.get("initial_state"),
                "dt": dt,
                "mu": mu,
                "dx": dx,
                "diffusion_number": mu * dt / dx**2,
                "number_of_parameters": gradients.shape[0],
                "parameter_index": parameter_index,
                "parameter": (
                    labels[parameter_index]
                    if parameter_index < len(labels)
                    else f"theta_{parameter_index}"
                ),
                "gradient_sweeps": gradients.shape[1],
            }
            _add_component_statistics(
                record, parameter_components, per_parameter=True
            )
            records.append(record)
    columns = [
        "full_label",
        "n",
        "l",
        "circuit",
        "initial_state",
        "dt",
        "mu",
        "dx",
        "diffusion_number",
        "number_of_parameters",
        "parameter_index",
        "parameter",
        "gradient_sweeps",
        "mean_gradient",
        "mean_gradient_norm",
        "variance",
        *_gradient_statistic_columns(),
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def filter_gradient_table(
    table: pd.DataFrame | Sequence[Mapping[str, Any]],
    ns: Iterable[int] | int | None = None,
    ls: Iterable[int] | int | None = None,
    circuits: Iterable[str] | str | None = None,
    initial_states: Iterable[str] | str | None = None,
) -> pd.DataFrame:
    """Filter a run-level or parameter-level gradient statistics table."""
    filtered = _as_table(table)
    for column, selected in (
        ("n", ns),
        ("l", ls),
        ("circuit", circuits),
        ("initial_state", initial_states),
    ):
        if selected is None:
            continue
        if column not in filtered.columns:
            raise KeyError(f"Gradient table has no {column!r} column.")
        allowed = {selected} if isinstance(selected, (str, int, np.integer)) else set(selected)
        filtered = filtered[filtered[column].isin(allowed)]
    return filtered.reset_index(drop=True)


def _gradient_group_columns(group_by: str | Sequence[str] | None) -> list[str]:
    if group_by is None:
        return []
    return [group_by] if isinstance(group_by, str) else list(group_by)


def _gradient_y_columns(y: str | Sequence[str]) -> list[str]:
    columns = [y] if isinstance(y, str) else list(y)
    if not columns:
        raise ValueError("y must contain at least one column.")
    if any(not isinstance(column, str) for column in columns):
        raise TypeError("Every y column must be a string.")
    return list(dict.fromkeys(columns))


def _plot_gradient_table(
    table: pd.DataFrame,
    x: str,
    y: str | Sequence[str],
    group_by: str | Sequence[str] | None,
    kind: str,
    log_x: bool,
    log_y: bool,
    ax,
    x_label: str | None = None,
    y_label: str | None = None,
):
    y_columns = _gradient_y_columns(y)
    for column in (x, *y_columns):
        if column not in table.columns:
            raise KeyError(f"Unknown column {column!r}. Available columns: {list(table.columns)}")
    group_columns = _gradient_group_columns(group_by)
    missing_groups = [column for column in group_columns if column not in table.columns]
    if missing_groups:
        raise KeyError(f"Unknown group_by columns: {missing_groups}.")
    if kind not in {"line", "scatter"}:
        raise ValueError("kind must be 'line' or 'scatter'")

    filtered = (
        table.dropna(subset=[x])
        .dropna(subset=y_columns, how="all")
        .reset_index(drop=True)
    )
    if filtered.empty:
        raise ValueError("No rows remain after filtering and dropping missing x/y values.")
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    multiple_y = len(y_columns) > 1
    for y_column in y_columns:
        metric_table = filtered.dropna(subset=[y_column])
        if group_columns:
            group_key = group_columns[0] if len(group_columns) == 1 else group_columns
            grouped = metric_table.groupby(group_key, dropna=False, sort=True)
        else:
            grouped = [(None, metric_table)]
        for group_value, group in grouped:
            group = group.sort_values(x, kind="stable")
            label_parts = [y_column] if multiple_y else []
            if group_columns:
                values = group_value if isinstance(group_value, tuple) else (group_value,)
                label_parts.extend(
                    f"{column}={value}"
                    for column, value in zip(group_columns, values)
                )
            label = ", ".join(label_parts) or None
            if kind == "scatter":
                ax.scatter(group[x], group[y_column], label=label)
            else:
                ax.plot(group[x], group[y_column], marker="o", label=label)

    ax.set_xlabel(x if x_label is None else x_label)
    default_y_label = y_columns[0] if not multiple_y else "Value"
    ax.set_ylabel(default_y_label if y_label is None else y_label)
    grouping = "" if not group_columns else f" grouped by {', '.join(group_columns)}"
    title_y = y_columns[0] if not multiple_y else "Gradient statistics"
    # ax.set_title(f"{title_y} vs {x}{grouping}")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if group_columns or multiple_y:
        ax.legend()
    return filtered, ax, _


def plot_gradient_run_table(
    table: pd.DataFrame | Sequence[Mapping[str, Any]],
    x: str,
    y: str | Sequence[str],
    ns: Iterable[int] | int | None = None,
    ls: Iterable[int] | int | None = None,
    circuits: Iterable[str] | str | None = None,
    initial_states: Iterable[str] | str | None = None,
    group_by: str | Sequence[str] | None = None,
    kind: str = "line",
    log_x: bool = False,
    log_y: bool = False,
    ax=None,
    x_label: str | None = None,
    y_label: str | None = None,
):
    """Plot one or more overall-gradient columns with filters and grouping."""
    filtered = filter_gradient_table(
        table,
        ns=ns,
        ls=ls,
        circuits=circuits,
        initial_states=initial_states,
    )
    return _plot_gradient_table(
        filtered,
        x,
        y,
        group_by,
        kind,
        log_x,
        log_y,
        ax,
        x_label=x_label,
        y_label=y_label,
    )


def _middle_layer_parameter_index(row: Mapping[str, Any]) -> int:
    n_value = row.get("n")
    l_value = row.get("l")
    if pd.isna(n_value) or pd.isna(l_value):
        raise ValueError("The 'mid' selector requires N and L metadata for every run.")
    n = int(n_value)
    layers = int(l_value)
    circuit = fcq.resolve_unitary_circuit(row.get("circuit"))
    total = int(row.get("number_of_parameters") or fcq.num_unitary_parameters(n, layers, circuit))
    if layers < 1:
        return total // 2

    final_layer_parameters = fcq.num_unitary_parameters(n, 0, circuit)
    repeated_parameters = total - final_layer_parameters
    if repeated_parameters % layers:
        raise ValueError(
            f"Cannot identify equal parameter layers for {row.get('full_label', '<unknown>')}."
        )
    parameters_per_layer = repeated_parameters // layers
    return (layers // 2) * parameters_per_layer + parameters_per_layer // 2


def _select_gradient_parameters(
    table: pd.DataFrame,
    parameters: Iterable[int | str] | int | str | None,
) -> pd.DataFrame:
    if parameters is None:
        selected = table.copy()
        selected["parameter_selection"] = selected["parameter_index"].astype(str)
        return selected
    selectors = [parameters] if isinstance(parameters, (str, int, np.integer)) else list(parameters)
    if not selectors:
        raise ValueError("parameters must contain at least one index or 'mid'.")

    selected_groups = []
    for _full_label, group in table.groupby("full_label", dropna=False, sort=False):
        first = group.iloc[0]
        parameter_count = int(first["number_of_parameters"])
        for selector in selectors:
            if isinstance(selector, str):
                if selector.strip().lower() != "mid":
                    raise ValueError("String parameter selectors must be 'mid'.")
                resolved = _middle_layer_parameter_index(first)
                selection_label = "mid"
            elif isinstance(selector, (int, np.integer)):
                resolved = int(selector)
                if resolved < 0:
                    resolved += parameter_count
                selection_label = str(int(selector))
            else:
                raise TypeError("Parameter selectors must be integers or 'mid'.")
            if not 0 <= resolved < parameter_count:
                raise IndexError(
                    f"Parameter {selector!r} is out of range for {first['full_label']} "
                    f"with {parameter_count} parameters."
                )
            match = group[group["parameter_index"] == resolved].copy()
            if match.empty:
                raise ValueError(
                    f"Parameter table has no index {resolved} for {first['full_label']}."
                )
            match["parameter_selection"] = selection_label
            selected_groups.append(match)
    if not selected_groups:
        raise ValueError("No parameter rows remain after filtering.")
    return pd.concat(selected_groups, ignore_index=True)


def plot_gradient_parameter_table(
    table: pd.DataFrame | Sequence[Mapping[str, Any]],
    x: str,
    y: str | Sequence[str],
    parameters: Iterable[int | str] | int | str | None = None,
    ns: Iterable[int] | int | None = None,
    ls: Iterable[int] | int | None = None,
    circuits: Iterable[str] | str | None = None,
    initial_states: Iterable[str] | str | None = None,
    group_by: str | Sequence[str] | None = None,
    kind: str = "line",
    log_x: bool = False,
    log_y: bool = False,
    ax=None,
    x_label: str | None = None,
    y_label: str | None = None,
):
    """Plot one or more selected scalar-parameter gradient statistics.

    Parameter selectors can be non-negative indices, negative Python-style
    indices, or ``"mid"`` for the middle parameter of the middle repeated layer.
    """
    filtered = filter_gradient_table(
        table,
        ns=ns,
        ls=ls,
        circuits=circuits,
        initial_states=initial_states,
    )
    selected = _select_gradient_parameters(filtered, parameters)
    return _plot_gradient_table(
        selected,
        x,
        y,
        group_by,
        kind,
        log_x,
        log_y,
        ax,
        x_label=x_label,
        y_label=y_label,
    )


def load_variance_run(dir_label: str, label: str, results_root: Path = RESULTS_ROOT) -> dict[str, Any]:
    run = load_run(dir_label, label, results_root)
    return {
        "dir_label": dir_label,
        "label": label,
        "full_label": run.full_label,
        "data_dir": run.data_dir,
        "values": run.values,
        "manifest": run.manifest,
        "n": run.n,
        "l": run.l,
        "mode": run.mode,
        "circuit": run.circuit,
        "variance": run.variance,
        "variance_tries": run.metadata("variance_tries"),
        "variance_num_samples": run.metadata("variance_num_samples"),
        "variance_runtime_s": run.metadata("variance_runtime_s"),
        "variance_grads_path": run.files["variance_grads"],
    }


def combine_variance_runs(
    runs: Sequence[RunData | Mapping[str, Any]],
    full_label: str | None = None,
) -> dict[str, Any]:
    """Combine compatible variance runs by concatenating their gradient samples."""
    if not runs:
        raise ValueError("At least one variance run is required.")

    rows = [load_variance_run(run.dir_label, run.label) if isinstance(run, RunData) else dict(run) for run in runs]
    configurations = {
        (row.get("n"), row.get("l"), str(row.get("circuit", "default")))
        for row in rows
    }
    if len(configurations) != 1:
        raise ValueError(
            "Combined variance runs must have the same N, L, and circuit; "
            f"found {sorted(configurations, key=str)}."
        )

    gradient_arrays = []
    missing = []
    for row in rows:
        gradients = _variance_gradient_samples(row)
        if gradients is None:
            missing.append(row.get("full_label", "<unknown>"))
        else:
            gradient_arrays.append(gradients)
    if missing:
        raise FileNotFoundError(
            "Raw variance gradients are required for exact combination. Missing for: "
            + ", ".join(missing)
        )

    combined_gradients = np.concatenate(gradient_arrays)
    n, l, circuit = next(iter(configurations))
    source_labels = [str(row["full_label"]) for row in rows]
    label = full_label or f"combined_N{n}L{l}_{circuit}"
    tries_values = [row.get("variance_tries") for row in rows]
    runtime_values = [row.get("variance_runtime_s") for row in rows]
    total_tries = sum(int(value) for value in tries_values if value is not None)
    total_runtime = (
        float(sum(float(value) for value in runtime_values))
        if runtime_values and all(value is not None for value in runtime_values)
        else None
    )
    variance = float(np.var(combined_gradients))
    values = dict(rows[0].get("values") or {})
    values.update(
        {
            "variance": variance,
            "variance_tries": total_tries,
            "variance_num_samples": int(combined_gradients.size),
            "variance_runtime_s": total_runtime,
            "combined_from": source_labels,
        }
    )
    return {
        "dir_label": "combined",
        "label": label,
        "full_label": label,
        "data_dir": None,
        "values": values,
        "manifest": dict(rows[0].get("manifest") or {}),
        "n": int(n),
        "l": int(l),
        "mode": "variance",
        "circuit": circuit,
        "variance": variance,
        "variance_tries": total_tries,
        "variance_num_samples": int(combined_gradients.size),
        "variance_runtime_s": total_runtime,
        "variance_grads_path": None,
        "gradient_mean": float(np.mean(combined_gradients)),
        "combined_from": source_labels,
    }


def apply_variance_combinations(
    rows: Sequence[Mapping[str, Any]],
    combinations: Mapping[str, Sequence[tuple[str, str] | str]],
) -> list[dict[str, Any]]:
    """Replace selected source rows with exact combined-variance rows.

    Combination members can be given as ``(dir_label, label)`` pairs or as their
    ``full_label`` strings. A combination is skipped when none of its members are
    selected; if only some are selected, the incomplete combination is rejected.
    """
    selected = [dict(row) for row in rows]
    by_label = {str(row["full_label"]): row for row in selected}
    consumed: set[str] = set()
    combined_rows = []
    for combined_label, members in combinations.items():
        member_labels = [f"{member[0]}_{member[1]}" if isinstance(member, tuple) else str(member) for member in members]
        missing = [label for label in member_labels if label not in by_label]
        if len(missing) == len(member_labels):
            continue
        if missing:
            raise KeyError(
                f"Cannot build {combined_label!r}; these source runs are not selected: "
                + ", ".join(missing)
            )
        overlap = consumed.intersection(member_labels)
        if overlap:
            raise ValueError(f"Variance source runs cannot be reused across combinations: {sorted(overlap)}")
        consumed.update(member_labels)
        combined_rows.append(
            combine_variance_runs([by_label[label] for label in member_labels], full_label=combined_label)
        )

    result = [row for row in selected if str(row["full_label"]) not in consumed]
    result.extend(combined_rows)
    result.sort(
        key=lambda row: (
            row["n"] if row.get("n") is not None else 10**9,
            row["l"] if row.get("l") is not None else 10**9,
            row["full_label"],
        )
    )
    return result


def discover_variance_runs(results_root: Path = RESULTS_ROOT) -> list[dict[str, Any]]:
    rows = []
    for row in list_runs(results_root):
        if row["mode"] == "variance" or row["has_variance"]:
            rows.append(load_variance_run(row["dir_label"], row["label"], results_root))
    rows.sort(
        key=lambda row: (
            row["n"] if row["n"] is not None else 10**9,
            row["l"] if row["l"] is not None else 10**9,
            row["full_label"],
        )
    )
    return rows


def print_variance_runs(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        combined_count = len(row.get("combined_from", []))
        combined_text = f" combined={combined_count}" if combined_count else ""
        print(
            f"{row['full_label']:<24} N={str(row['n']):<3} L={str(row['l']):<3} "
            f"circuit={str(row.get('circuit')):<10} variance={str(row['variance']):<14} "
            f"tries={row['variance_tries']}{combined_text}"
        )
    print("count=", len(rows))


def variance_table(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    columns = [
        "full_label",
        "n",
        "l",
        "circuit",
        "variance",
        "variance_tries",
        "variance_num_samples",
        "variance_runtime_s",
        "gradient_mean",
        "combined_from",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


def _valid_variance_rows(rows: Sequence[Mapping[str, Any]]):
    return [row for row in rows if row.get("variance") is not None and row.get("n") is not None and row.get("l") is not None]


def plot_variance_vs_n(rows: Sequence[Mapping[str, Any]], log_y: bool = True, ax=None):
    valid = _valid_variance_rows(rows)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    by_l = defaultdict(list)
    for row in valid:
        by_l[int(row["l"])].append(row)
    for l_value, group in sorted(by_l.items()):
        group.sort(key=lambda row: int(row["n"]))
        ax.plot([int(row["n"]) for row in group], [float(row["variance"]) for row in group], marker="o", label=f"L={l_value}")
    ax.set_title("Variance of cost-function gradient vs N")
    ax.set_xlabel("N")
    ax.set_ylabel("variance")
    if log_y:
        ax.set_yscale("log")
    if by_l:
        ax.legend()
    return ax


def plot_variance_vs_l(rows: Sequence[Mapping[str, Any]], log_y: bool = True, ax=None):
    valid = _valid_variance_rows(rows)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    by_n = defaultdict(list)
    for row in valid:
        by_n[int(row["n"])].append(row)
    for n_value, group in sorted(by_n.items()):
        group.sort(key=lambda row: int(row["l"]))
        ax.plot([int(row["l"]) for row in group], [float(row["variance"]) for row in group], marker="o", label=f"N={n_value}")
    ax.set_title("Variance vs L (grouped by N)")
    ax.set_xlabel("L")
    ax.set_ylabel("variance")
    if log_y:
        ax.set_yscale("log")
    if by_n:
        ax.legend(ncol=2)
    return ax


def plot_variance_heatmap(rows: Sequence[Mapping[str, Any]], ax=None):
    valid = _valid_variance_rows(rows)
    if not valid:
        print("No variance rows available.")
        return None
    ns = sorted({int(row["n"]) for row in valid})
    ls = sorted({int(row["l"]) for row in valid})
    grid = np.full((len(ns), len(ls)), np.nan, dtype=float)
    n_index = {value: index for index, value in enumerate(ns)}
    l_index = {value: index for index, value in enumerate(ls)}
    for row in valid:
        variance = float(row["variance"])
        if variance > 0:
            grid[n_index[int(row["n"])], l_index[int(row["l"])]] = np.log10(variance)
    if ax is None:
        _, ax = plt.subplots(figsize=(1 + 1.2 * len(ls), 1 + 0.8 * len(ns)))
    image = ax.imshow(grid, aspect="auto")
    ax.set_xticks(range(len(ls)), labels=ls)
    ax.set_yticks(range(len(ns)), labels=ns)
    ax.set_xlabel("L")
    ax.set_ylabel("N")
    ax.set_title("log10(variance) heatmap")
    midpoint = np.nanmean(grid)
    for i in range(len(ns)):
        for j in range(len(ls)):
            if np.isfinite(grid[i, j]):
                color = "white" if grid[i, j] < midpoint else "black"
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color=color)
    colorbar = ax.figure.colorbar(image, ax=ax)
    colorbar.set_label("log10(variance)")
    return grid, ax
