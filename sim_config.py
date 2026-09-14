from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

import functions.circuits_quimb as fcq
import functions.initial_states as fist


def default_gradient_tries(n: int) -> int:
    """Return the requested number of scalar gradients for a gradient run."""
    if 2 <= n <= 5:
        return 20_000
    defaults = {
        6: 15_000,
        7: 10_000,
        8: 5_000,
        9: 4_000,
        10: 500,
    }
    try:
        return defaults[n]
    except KeyError as exc:
        raise ValueError("Gradient-run defaults are defined for N=2 through N=10.") from exc


@dataclass(slots=True)
class SimConfig:
    label: str = "L3"
    dir_label: str = "N3"
    mode: str = "noise_free"  # noise_free | adam_exact | adam_shots | cobyla_shots | gradient | exp_only
    unitary_circuit: str | None = None
    initial_state: str | None = None

    n: int = 3
    l: int = 3
    t_total: float = 0.29
    mu: float = 0.01
    dt: float = 0.01

    optimization_steps: int = 1000

    max_iter: int = 100
    grad_tol: float = 1e-3
    plat_tol: float = 0.00015
    patience: int = 15

    shots: int = 100
    cobyla_tol: float = 1e-5
    cobyla_maxiter: int = 100000

    verbose: bool = True

    compute_expr_cap: bool = False
    expr_entcap_samples: int = 100000
    expr_bins: int = 100

    # Experimental only: cached positional paths can be invalid after Quimb
    # reconstructs or simplifies a network. Automatic per-network path
    # selection is the safe release default and was used for the thesis runs.
    optimize_contraction_paths: bool = False
    contraction_path_repeats: int = 256
    contraction_path_max_time_s: float | None = 60.0
    contraction_path_seed: int = 0
    contraction_path_methods: tuple[str, ...] = ("greedy", "kahypar")
    contraction_path_objective: str = "combo"
    contraction_path_simplify_sequence: str = "RC"
    contraction_path_cache_dir: str = ".contraction_paths"
    contraction_path_reuse_saved: bool = True

    # Requested number of scalar parameter gradients. Gradient mode rounds this
    # down to the nearest complete random-parameter sweep.
    gradient_tries: int | None = None

    initial_params: tuple[float, ...] | None = None
    initial_params_presets_file: str = "initial_params_presets.yaml"
    initial_params_variant: str | None = None
    init_param_random_scale: float = 0.1
    init_param_random_center: float = np.pi - 0.25
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.gradient_tries is None and self.mode == "gradient":
            self.gradient_tries = default_gradient_tries(self.n)
        if self.gradient_tries is not None and self.gradient_tries < 1:
            raise ValueError("gradient_tries must be at least 1.")
        if self.contraction_path_repeats < 1:
            raise ValueError("contraction_path_repeats must be at least 1.")
        if (
            self.contraction_path_max_time_s is not None
            and self.contraction_path_max_time_s <= 0
        ):
            raise ValueError("contraction_path_max_time_s must be positive or None.")
        if not self.contraction_path_methods:
            raise ValueError("contraction_path_methods must contain at least one method.")

    @property
    def results_dir(self) -> Path:
        return Path(f"results/results_{self.dir_label}")

    @property
    def fig_dir(self) -> Path:
        return self.results_dir / f"figures_{self.label}"

    @property
    def data_dir(self) -> Path:
        return self.results_dir / f"data_{self.label}"

    @property
    def manifest_path(self) -> Path:
        return self.results_dir / f"manifest_{self.full_label}.yaml"

    @property
    def expr_manifest_path(self) -> Path:
        return self.results_dir / f"manifest_{self.full_label}_expr.yaml"

    @property
    def expr_values_path(self) -> Path:
        return self.data_dir / f"values_{self.full_label}_expr.yaml"

    @property
    def full_label(self) -> str:
        return f"{self.dir_label}_{self.label}"

    @property
    def n_total(self) -> int:
        return 2**self.n

    @property
    def n_timesteps(self) -> int:
        return int(self.t_total / self.dt)

    @property
    def resolved_unitary_circuit(self) -> str:
        return fcq.resolve_unitary_circuit(self.unitary_circuit)

    @property
    def resolved_initial_state(self) -> str:
        return fist.resolve_initial_state(self.initial_state)

    @property
    def number_of_parameters(self) -> int:
        """Number of unitary-circuit parameters, excluding the norm parameter."""
        return fcq.num_unitary_parameters(self.n, self.l, self.resolved_unitary_circuit)

    @property
    def number_of_optimization_parameters(self) -> int:
        return self.number_of_parameters + 1

    @property
    def gradient_sweeps(self) -> int:
        """Number of complete parameter sweeps in a gradient run."""
        if self.gradient_tries is None:
            raise ValueError("gradient_tries is only populated for gradient runs.")
        return int(self.gradient_tries) // self.number_of_parameters

    @property
    def gradient_num_samples(self) -> int:
        """Actual scalar-gradient count after rounding to complete sweeps."""
        return self.gradient_sweeps * self.number_of_parameters

    @property
    def initial_params_presets_path(self) -> Path:
        return Path(self.initial_params_presets_file)

    def resolve_initial_params(self) -> tuple[float, ...]:
        if self.initial_params is not None:
            return tuple(float(x) for x in self.initial_params)

        presets_path = self.initial_params_presets_path
        if not presets_path.exists():
            raise FileNotFoundError(
                f"Initial parameter preset file not found: {presets_path}. "
                "Set `initial_params` explicitly or create the presets YAML."
            )

        with presets_path.open("r", encoding="utf-8") as fh:
            presets = yaml.safe_load(fh) or {}

        n_key_int = self.n
        n_key_str = str(self.n)
        l_key_int = self.l
        l_key_str = str(self.l)

        by_n = presets.get(n_key_int, presets.get(n_key_str))
        if by_n is None:
            raise KeyError(
                f"No initial parameter presets found for n={self.n} in {presets_path}."
            )

        entry = by_n.get(l_key_int, by_n.get(l_key_str))
        if entry is None:
            raise KeyError(
                f"No initial parameter preset found for (n={self.n}, l={self.l}) in {presets_path}."
            )

        circuit_name = self.resolved_unitary_circuit
        initial_state_name = self.resolved_initial_state
        if isinstance(entry, dict):
            state_entry = (entry.get("initial_states") or {}).get(initial_state_name, {})
            circuit_entry = state_entry.get(circuit_name)
            if circuit_entry is not None:
                params = (
                    circuit_entry.get("params")
                    if isinstance(circuit_entry, dict)
                    else circuit_entry
                )
                if params is None:
                    raise KeyError(
                        f"Preset for (n={self.n}, l={self.l}, circuit={circuit_name!r}, "
                        f"initial_state={initial_state_name!r}) has no params in {presets_path}."
                    )
                return tuple(float(x) for x in params)

        # Backward-compatible lookup for the original sine/default and sine/UNI2 schema.
        if initial_state_name != fist.DEFAULT_INITIAL_STATE:
            raise KeyError(
                f"No initial parameter preset found for (n={self.n}, l={self.l}, "
                f"circuit={circuit_name!r}, initial_state={initial_state_name!r}) in {presets_path}."
            )

        variant_name = self.initial_params_variant
        if variant_name is None:
            if circuit_name == fcq.DEFAULT_UNITARY_CIRCUIT:
                variant_name = None
            elif circuit_name == "uni2":
                variant_name = "uni2"
            else:
                raise KeyError(
                    f"No stored initial-parameter family is configured for circuit {circuit_name!r}."
                )

        if isinstance(entry, dict):
            if variant_name is not None:
                variants = entry.get("variants", {})
                variant_entry = variants.get(variant_name)
                if variant_entry is None:
                    raise KeyError(
                        f"Variant '{variant_name}' not found for (n={self.n}, l={self.l}) in {presets_path}."
                    )
                if isinstance(variant_entry, dict):
                    params = variant_entry.get("params")
                else:
                    params = variant_entry
            else:
                params = entry.get("default")
                if params is None and "params" in entry:
                    params = entry["params"]
                if params is None:
                    raise KeyError(
                        f"Preset entry for (n={self.n}, l={self.l}) has no 'default' params in {presets_path}."
                    )
        elif variant_name is None:
            params = entry
        else:
            raise KeyError(
                f"Preset entry for (n={self.n}, l={self.l}) has no variants in {presets_path}."
            )

        return tuple(float(x) for x in params)
