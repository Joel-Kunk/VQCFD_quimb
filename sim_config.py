from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_INITIAL_PARAMS = (
    1.87082869, 1.69994981, 2.60121564, 2.06343492, 2.16748982,
        2.16757722, 3.32352347, 3.28004099, 3.8192506 , 3.33380867,
        2.45316203, 2.3847443 , 2.76385659
)


@dataclass(slots=True)
class SimConfig:
    label: str = "L3"
    dir_label: str = "N3"
    mode: str = "noise_free"  # noise_free | adam_exact | adam_shots | cobyla_shots

    n: int = 3
    l: int = 3
    shots: int = 125000
    max_iter: int = 50
    grad_tol: float = 1e-3
    plat_tol: float = 0.0
    patience: int = 15

    t_total: float = 0.29
    mu: float = 0.01
    dt: float = 0.01

    optimization_steps: int = 1000
    cobyla_tol: float = 1e-5
    cobyla_maxiter: int = 100000

    init_param_random_scale: float = 0.1
    init_param_random_center: float = np.pi - 0.25
    random_seed: int | None = None
    verbose: bool = True

    compute_expr_cap: bool = False
    expr_samples: int = 100000
    entcap_samples: int = 100

    initial_params: tuple[float, ...] = DEFAULT_INITIAL_PARAMS

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
        return self.results_dir / f"manifest_{self.label}.yaml"

    @property
    def n_total(self) -> int:
        return 2**self.n

    @property
    def n_timesteps(self) -> int:
        return int(self.t_total / self.dt)
