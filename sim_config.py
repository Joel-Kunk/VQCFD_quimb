from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

import functions.circuits_quimb as fcq
import functions.initial_states as fist


@dataclass(slots=True)
class SimConfig:
    label: str = "L3"
    dir_label: str = "N3"
    mode: str = "noise_free"  # noise_free | adam_exact | adam_shots | cobyla_shots | variance
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

    variance_tries: int = 100
    
    initial_params: tuple[float, ...] | None = None
    initial_params_presets_file: str = "initial_params_presets.yaml"
    initial_params_variant: str | None = None
    init_param_random_scale: float = 0.1
    init_param_random_center: float = np.pi - 0.25
    random_seed: int | None = None

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
    def initial_params_presets_path(self) -> Path:
        return Path(self.initial_params_presets_file)

    def resolve_initial_params(self) -> tuple[float, ...]:
        if self.initial_params is not None:
            return tuple(float(x) for x in self.initial_params)

        if self.resolved_initial_state != fist.DEFAULT_INITIAL_STATE:
            raise KeyError(
                f"Stored presets target the default sine initial state, not {self.resolved_initial_state!r}."
            )

        variant_name = self.initial_params_variant
        if variant_name is None:
            if self.resolved_unitary_circuit == fcq.DEFAULT_UNITARY_CIRCUIT:
                variant_name = None
            elif self.resolved_unitary_circuit == "uni2":
                variant_name = "uni2"
            else:
                raise KeyError(
                    f"No stored initial-parameter family is configured for circuit "
                    f"{self.resolved_unitary_circuit!r}."
                )

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
