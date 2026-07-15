from __future__ import annotations

import numpy as np


DEFAULT_INITIAL_STATE = "sine"
INITIAL_STATES = (
    DEFAULT_INITIAL_STATE,
    "positive_hump",
    "positive_periodic_wave",
    "mixed_sine_modes",
    "tapered_gaussian",
    "tapered_tanh",
)


def resolve_initial_state(initial_state: str | None = None) -> str:
    if initial_state is None:
        return DEFAULT_INITIAL_STATE

    name = str(initial_state).strip().lower()
    aliases = {
        "": DEFAULT_INITIAL_STATE,
        "default": DEFAULT_INITIAL_STATE,
        "current": DEFAULT_INITIAL_STATE,
    }
    name = aliases.get(name, name)
    if name not in INITIAL_STATES:
        raise ValueError(
            f"Unknown initial state {initial_state!r}. "
            f"Expected one of: {', '.join(INITIAL_STATES)}."
        )
    return name


def make_initial_field(xs: np.ndarray, initial_state: str | None = None) -> np.ndarray:
    """Return one of the simple initial fields on ``xs``."""
    x = np.asarray(xs, dtype=float)
    name = resolve_initial_state(initial_state)

    if name == "sine":
        # Keep the legacy default bit-for-bit, including NumPy's tiny endpoint
        # roundoff, so ``initial_state=None`` does not alter existing runs.
        return np.sin(2 * np.pi * x)
    if name == "positive_periodic_wave":
        return 0.5 + 0.5 * np.sin(2 * np.pi * x)
    if name == "positive_hump":
        field = np.sin(np.pi * x)
    elif name == "mixed_sine_modes":
        field = np.sin(2 * np.pi * x) + 0.35 * np.sin(4 * np.pi * x)
    elif name == "tapered_gaussian":
        field = np.sin(np.pi * x) * np.exp(-((x - 0.5) / 0.18) ** 2)
    else:  # tapered_tanh
        field = np.sin(np.pi * x) * np.tanh((0.5 - x) / 0.08)

    field = np.asarray(field, dtype=float)
    if field.size:
        field = field.copy()
        field[0] = 0.0
        field[-1] = 0.0
    return field
