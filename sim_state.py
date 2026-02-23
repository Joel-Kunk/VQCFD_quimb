from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class SimState:
    prev_params_quimb: np.ndarray
    params_list_quimb: list[np.ndarray] = field(default_factory=list)
    cost_list: list[float] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    num_evals: list[int] = field(default_factory=list)
