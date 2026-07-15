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
    "Expressibility",
    "EntanglingCapability",
    "Circuit",
    "InitialState",
    "R2",
    "Fidelity",
    "MSE",
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


def discover_simulation_runs(results_root: Path = RESULTS_ROOT) -> list[RunData]:
    """Load every non-variance run that has a saved parameter trajectory."""
    return [
        load_run(row["dir_label"], row["label"], results_root)
        for row in list_runs(results_root)
        if row["mode"] != "variance" and row["has_params"]
    ]


def list_runs(results_root: Path = RESULTS_ROOT) -> list[dict[str, Any]]:
    rows = []
    for results_dir in sorted(Path(results_root).glob("results_*")):
        dir_label = results_dir.name.removeprefix("results_")
        for data_dir in sorted(results_dir.glob("data_*")):
            label = data_dir.name.removeprefix("data_")
            run = load_run(dir_label, label, results_root)
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
                    "path": str(data_dir),
                }
            )
    return rows


def print_runs(runs: Sequence[Mapping[str, Any]] | None = None, limit: int = 200) -> None:
    runs = list_runs() if runs is None else runs
    for row in runs[:limit]:
        print(
            f"{row['full_label']:<18} mode={str(row['mode']):<12} "
            f"N={str(row['n']):<3} L={str(row['l']):<3} "
            f"circuit={str(row.get('circuit')):<22} "
            f"initial_state={str(row.get('initial_state')):<18} "
            f"n_params={str(row.get('number_of_parameters')):<4} saved={row.get('has_params')} "
            f"variance={row.get('has_variance')}"
        )
    print(f"count={len(runs)}")


def list_initial_param_presets(
    presets_path: Path | None = None,
    show_variants: bool = True,
    n: int | None = None,
    l: int | None = None,
) -> list[dict[str, Any]]:
    """Print and return available entries from ``initial_params_presets.yaml``."""
    presets_path = presets_path or (REPO_ROOT / "initial_params_presets.yaml")
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
            rows.append(
                {
                    "n": n_int,
                    "l": l_int,
                    "default_len": len(default_params) if isinstance(default_params, list) else None,
                    "default_mse": entry.get("default_mse") if isinstance(entry, dict) else None,
                    "variants": variants,
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


def classical_reference(run: RunData) -> tuple[np.ndarray, np.ndarray]:
    """Recreate the exact classical trajectory used by ``simulation_runner.py``."""
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
    fields[:, 0] = fist.make_initial_field(xs, run.initial_state)
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
    state_params = np.asarray(params[index], dtype=float)
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
    return index, np.asarray(field, dtype=float)


def comparison_metrics(classical: np.ndarray, quantum: np.ndarray) -> dict[str, float]:
    classical = np.asarray(classical).reshape(-1)
    quantum = np.asarray(quantum).reshape(-1)
    if classical.shape != quantum.shape:
        raise ValueError(f"Field shapes do not match: {classical.shape} != {quantum.shape}")
    residual = classical - quantum
    mse = float(np.mean(np.abs(residual) ** 2))
    total = float(np.sum(np.abs(classical - np.mean(classical)) ** 2))
    r2 = float(1.0 - np.sum(np.abs(residual) ** 2) / total) if total > 0 else float("nan")
    norm_product = float(np.vdot(classical, classical).real * np.vdot(quantum, quantum).real)
    fidelity = float(np.abs(np.vdot(classical, quantum)) ** 2 / norm_product) if norm_product > 0 else float("nan")
    return {"MSE": mse, "R2": r2, "Fidelity": fidelity}


def compare_run_timestep(run: RunData, timestep: int = -1) -> dict[str, Any]:
    index, quantum = quantum_field(run, timestep)
    xs, classical_fields = classical_reference(run)
    if index >= classical_fields.shape[1]:
        raise IndexError(
            f"Quantum timestep {index} has no matching classical state; "
            f"classical range is 0..{classical_fields.shape[1] - 1}."
        )
    classical = classical_fields[:, index]
    metrics = comparison_metrics(classical, quantum)
    return {
        "run": run,
        "timestep": index,
        "time": index * float(run.metadata("dt")),
        "x": xs,
        "classical": classical,
        "quantum": quantum,
        **metrics,
    }


def plot_run_timestep_comparison(run: RunData, timestep: int = -1, ax=None):
    comparison = compare_run_timestep(run, timestep)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    ax.plot(comparison["x"], comparison["classical"], label="Classical", linewidth=2)
    ax.plot(comparison["x"], comparison["quantum"], "--", label="Quantum", linewidth=2)
    ax.set_title(
        f"{run.full_label}: timestep {comparison['timestep']} "
        f"(t={comparison['time']:.4g})"
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("Field value")
    ax.legend()
    metric_text = (
        f"MSE = {comparison['MSE']:.4e}\n"
        f"R² = {comparison['R2']:.6f}\n"
        f"Fidelity = {comparison['Fidelity']:.6f}"
    )
    ax.text(
        0.02,
        0.02,
        metric_text,
        transform=ax.transAxes,
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    return comparison, ax


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


def build_run_records(
    runs: Sequence[RunData],
    variance_runs: Iterable[RunData | Mapping[str, Any]] | None = None,
    circuit_overrides: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build per-run final-state metrics and metadata records.

    Separate variance-mode runs can be supplied through ``variance_runs``. They are
    matched by ``(N, L, Circuit)`` and then by ``(N, L)`` when unambiguous. Repeated
    variance runs for the same configuration are averaged.
    """
    exact_variance, n_l_variance = _variance_lookup(variance_runs)
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
        metrics = {"R2": np.nan, "Fidelity": np.nan, "MSE": np.nan}
        if run.params is not None:
            metrics.update({key: value for key, value in compare_run_timestep(run, -1).items() if key in metrics})
        records.append(
            {
                "Run": run.full_label,
                "N": run.n,
                "L": run.l,
                "NumberOfParameters": run.number_of_parameters,
                "Var": np.nan if variance is None else float(variance),
                "Expressibility": _float_or_nan(run.metadata("expressibility")),
                "EntanglingCapability": _float_or_nan(run.metadata("entangling_capability")),
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
) -> pd.DataFrame:
    return pd.DataFrame(
        build_run_records(runs, variance_runs, circuit_overrides),
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
    y: str,
    ns: Iterable[int] | int | None = None,
    ls: Iterable[int] | int | None = None,
    circuits: Iterable[str] | str | None = None,
    initial_states: Iterable[str] | str | None = None,
    group_by: str | None = None,
    kind: str = "line",
    log_y: bool = False,
    ax=None,
):
    """Plot any two table columns after filtering N, L, circuit, and initial state."""
    filtered = filter_run_table(
        table,
        ns=ns,
        ls=ls,
        circuits=circuits,
        initial_states=initial_states,
    )
    for column in (x, y):
        if column not in filtered.columns:
            raise KeyError(f"Unknown column {column!r}. Available columns: {list(filtered.columns)}")
    if group_by is not None and group_by not in filtered.columns:
        raise KeyError(f"Unknown group_by column {group_by!r}.")
    filtered = filtered.dropna(subset=[x, y])
    if filtered.empty:
        raise ValueError("No rows remain after filtering and dropping missing x/y values.")
    if kind not in {"line", "scatter"}:
        raise ValueError("kind must be 'line' or 'scatter'")
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    grouped = [(None, filtered)] if group_by is None else filtered.groupby(group_by, dropna=False, sort=True)
    for group_value, group in grouped:
        group = group.sort_values(x)
        label = None if group_by is None else f"{group_by}={group_value}"
        if kind == "scatter":
            ax.scatter(group[x], group[y], label=label)
        else:
            ax.plot(group[x], group[y], marker="o", label=label)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}" + ("" if group_by is None else f" grouped by {group_by}"))
    if log_y:
        ax.set_yscale("log")
    if group_by is not None:
        ax.legend()
    return filtered, ax


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
