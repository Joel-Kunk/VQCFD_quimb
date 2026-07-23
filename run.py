import time

from simulation_runner import SimConfig, format_elapsed, run_simulation

import functions.circuits_quimb as fcq
import functions.initial_states as fist


def build_sweep_cfg(n: int, l: int, **overrides) -> SimConfig:
    circuit_name = fcq.resolve_unitary_circuit(overrides.get("unitary_circuit"))
    initial_state_name = fist.resolve_initial_state(overrides.get("initial_state"))
    default_circuit = circuit_name == fcq.DEFAULT_UNITARY_CIRCUIT
    default_initial_state = initial_state_name == fist.DEFAULT_INITIAL_STATE

    if overrides.get("mode", "noise_free") == "gradient":
        label_parts = [f"N{n}L{l}"]
        if not default_circuit:
            label_parts.append(circuit_name)
        if not default_initial_state:
            label_parts.append(initial_state_name)
        automatic_dir_label = "grad"
        automatic_label = "_".join(label_parts)
    else:
        automatic_dir_label = f"N{n}" if default_circuit else f"N{n}_{circuit_name}"
        automatic_label = f"L{l}" if default_initial_state else f"L{l}_{initial_state_name}"

    return SimConfig(
        n=n,
        l=l,
        dir_label=overrides.pop("dir_label", automatic_dir_label),
        label=overrides.pop("label", automatic_label),
        **overrides,
    )


def main() -> None:
    run_mode = "single"  # "single" | "pairs" | "grid"

    # Common options applied to every run.
    common = dict(
        mode="exp_only",   # e.g. "noise_free", "adam_exact", "adam_shots", "cobyla_shots", "gradient", "exp_only"
        # `exp_only` always computes both metrics; this flag is for full runs.
        compute_expr_cap=False,
        # None selects the default circuit and omits the circuit label suffix.
        unitary_circuit=None,
        # Options: "uni2", "brickwork_ring_ry", "ring_ry_rz",
        # "ring_trainable_crx", "multiscale_tree"
        initial_state=None,
        # Options: "positive_hump", "positive_periodic_wave",
        # "mixed_sine_modes", "tapered_gaussian", "tapered_tanh"
        expr_entcap_samples = 20000,
        # gradient_tries = 5000,
        dir_label = "exp2",
    )

    if run_mode == "single":
        cfg = build_sweep_cfg(
            n=8,
            l=1,
            # For single runs these are used verbatim. Remove either argument
            # to fall back to the automatic mode/circuit/initial-state-aware path.
            # dir_label="tests",
            # label="N4_multiscale_tree_L6_tapered_tanh",
            **common,
        )
        run_simulation(cfg)
        return

    if run_mode == "pairs":
        pairs = [
            (3, 2),
            (4, 2),
            (5, 2),
        ]
        todo = pairs
    elif run_mode == "grid":
        ns = [2,3,4,5]
        ls = [1,2]
        todo = [(n, l) for n in ns for l in ls]
    else:
        raise ValueError("run_mode must be one of: 'single', 'pairs', 'grid'")

    failures: list[tuple[int, int, str]] = []
    total_runs = len(todo)
    sweep_start = time.perf_counter()
    for run_number, (n, l) in enumerate(todo, start=1):
        run_start = time.perf_counter()
        print(f"\n=== Run {run_number}/{total_runs}: Starting N={n}, L={l} ===")
        try:
            cfg = build_sweep_cfg(n=n, l=l, **common)
            run_simulation(cfg)
        except Exception as exc:  # keep sweep running if one pair fails
            failures.append((n, l, str(exc)))
            elapsed = format_elapsed(time.perf_counter() - run_start)
            print(f"=== Run {run_number}/{total_runs} FAILED after {elapsed}: {exc} ===")
        else:
            elapsed = format_elapsed(time.perf_counter() - run_start)
            print(f"=== Run {run_number}/{total_runs} finished in {elapsed} ===")

    sweep_elapsed = format_elapsed(time.perf_counter() - sweep_start)
    print(
        f"\nSweep finished in {sweep_elapsed}. "
        f"total={total_runs}, failed={len(failures)}"
    )
    for n, l, msg in failures:
        print(f" - N={n}, L={l}: {msg}")


if __name__ == "__main__":
    main()
