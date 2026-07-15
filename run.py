from simulation_runner import SimConfig, run_simulation

import functions.circuits_quimb as fcq
import functions.initial_states as fist


def build_sweep_cfg(n: int, l: int, **overrides) -> SimConfig:
    circuit_name = fcq.resolve_unitary_circuit(overrides.get("unitary_circuit"))
    initial_state_name = fist.resolve_initial_state(overrides.get("initial_state"))
    default_circuit = circuit_name == fcq.DEFAULT_UNITARY_CIRCUIT
    default_initial_state = initial_state_name == fist.DEFAULT_INITIAL_STATE

    return SimConfig(
        n=n,
        l=l,
        dir_label=overrides.pop(
            "dir_label",
            f"N{n}" if default_circuit else f"N{n}_{circuit_name}",
        ),
        label=overrides.pop(
            "label",
            f"L{l}" if default_initial_state else f"L{l}_{initial_state_name}",
        ),
        **overrides,
    )


def main() -> None:
    run_mode = "single"  # "single" | "pairs" | "grid"

    # Common options applied to every run.
    common = dict(
        mode="noise_free",   # e.g. "noise_free", "adam_exact", "adam_shots", "cobyla_shots", "variance"
        compute_expr_cap=True,
        # None preserves the old circuit, paths, and saved labels.
        unitary_circuit=None,
        # Options: "uni2", "brickwork_ring_ry", "ring_ry_rz",
        # "ring_trainable_crx", "multiscale_tree"
        initial_state=None,
        # Options: "positive_hump", "mixed_sine_modes",
        # "tapered_gaussian", "tapered_tanh"
    )

    if run_mode == "single":
        cfg = build_sweep_cfg(
            n=6,
            l=5,
            variance_tries=100,
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
        ns = [3,4,5,6]
        ls = [5]
        todo = [(n, l) for n in ns for l in ls]
    else:
        raise ValueError("run_mode must be one of: 'single', 'pairs', 'grid'")

    failures: list[tuple[int, int, str]] = []
    for n, l in todo:
        print(f"\n=== Running N={n}, L={l} ===")
        try:
            cfg = build_sweep_cfg(n=n, l=l, **common)
            run_simulation(cfg)
        except Exception as exc:  # keep sweep running if one pair fails
            failures.append((n, l, str(exc)))
            print(f"FAILED N={n}, L={l}: {exc}")

    print(f"\nSweep finished. total={len(todo)}, failed={len(failures)}")
    for n, l, msg in failures:
        print(f" - N={n}, L={l}: {msg}")


if __name__ == "__main__":
    main()
