from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(slots=True)
class SimConfig:
    label: str = "L3"
    dir_label: str = "N3"
    mode: str = "noise_free"  # noise_free | adam_exact | adam_shots | cobyla_shots

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

        if isinstance(entry, dict):
            if self.initial_params_variant is not None:
                variants = entry.get("variants", {})
                variant_entry = variants.get(self.initial_params_variant)
                if variant_entry is None:
                    raise KeyError(
                        f"Variant '{self.initial_params_variant}' not found for (n={self.n}, l={self.l}) in {presets_path}."
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
        else:
            params = entry

        return tuple(float(x) for x in params)
