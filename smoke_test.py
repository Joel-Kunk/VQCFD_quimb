"""Run one tiny end-to-end VQCFD simulation.

This is a runtime smoke test, not a scientifically converged calculation.
It deliberately uses the release-default automatic contraction behavior.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

from simulation_runner import SimConfig, run_simulation


REPO_ROOT = Path(__file__).resolve().parent


def main() -> None:
    os.chdir(REPO_ROOT)
    cfg = SimConfig(
        n=3,
        l=2,
        mode="noise_free",
        unitary_circuit=None,
        initial_state=None,
        dir_label="smoke",
        label="quick",
        t_total=0.1,
        dt=0.01,
        optimization_steps=5,
        compute_expr_cap=False,
        optimize_contraction_paths=False,
        initial_params_presets_file=str(REPO_ROOT / "initial_params_presets.yaml"),
        random_seed=123,
        verbose=False,
    )

    state = run_simulation(cfg)
    values_path = cfg.data_dir / f"values_{cfg.full_label}.yaml"
    if len(state.params_list_quimb) != 2:
        raise RuntimeError("Smoke run did not complete its single timestep.")
    if not cfg.manifest_path.exists() or not values_path.exists():
        raise RuntimeError("Smoke run did not write the expected output files.")

    print("VQCFD smoke run passed.")
    print(f"Manifest: {cfg.manifest_path}")
    print(f"Values:   {values_path}")


if __name__ == "__main__":
    main()
