import time

from simulation_runner import SimConfig, format_elapsed, run_simulation

import functions.circuits_quimb as fcq
import functions.initial_states as fist


def _sweep_values(value, name: str, resolver) -> list[str]:
    """Resolve one scalar or an ordered collection of sweep-axis values."""
    if value is None or isinstance(value, str):
        raw_values = [value]
    else:
        try:
            raw_values = list(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be a name, None, or a collection of names.") from exc
    if not raw_values:
        raise ValueError(f"{name} sweep must contain at least one value.")
    resolved_values = []
    for item in raw_values:
        resolved = resolver(item)
        if resolved not in resolved_values:
            resolved_values.append(resolved)
    return resolved_values


def _shared_directory_label(
    n: int,
    l: int,
    circuit: str,
    initial_state: str,
) -> str:
    """Build a unique run label when all sweep cases share one directory."""
    parts = [f"N{n}L{l}"]
    if circuit != fcq.DEFAULT_UNITARY_CIRCUIT:
        parts.append(circuit)
    if initial_state != fist.DEFAULT_INITIAL_STATE:
        parts.append(initial_state)
    return "_".join(parts)


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
    run_mode = "pairs"  # "single" | "pairs" | "grid"

    # Common options applied to every run.
    common = dict(
        mode="noise_free",   # e.g. "noise_free", "adam_exact", "adam_shots", "cobyla_shots", "gradient", "exp_only"
        # `exp_only` always computes both metrics; this flag is for full runs.
        compute_expr_cap=False,
        # Search each fixed contraction topology thoroughly once, persist the
        # best exact path, and reuse it for every optimization/time step. (Better to not use, only worth it for large TNs, but prone to bugs there)
        optimize_contraction_paths=False,
        contraction_path_repeats=256,
        contraction_path_max_time_s=600.0,  # Per c_i (and pure-state path).
        contraction_path_seed=0,
        contraction_path_reuse_saved=True,
        # Use one name/None, or a list to sweep over multiple circuits.
        unitary_circuit= ["paper_viscid"],#["all_to_all_crx","block_crx","ring_trainable_crz","ghz_orbit","local_ry_rz"],
        # Options: "default_rzrxrz", "uni2", "paper_viscid" (the paper's
        # ancilla-ended G-block V, scalable to every N),
        # "brickwork_ring_ry", "ring_ry_rz", "ring_trainable_crx",
        # "multiscale_tree", "multiscale_tree_rzrxrz", "local_ry_rz",
        # "ghz_orbit", "ring_trainable_crz", "all_to_all_crx", "block_crx",
        # "mps_staircase", "mps_staircase_light"
        # Example sweep: [None, "uni2", "multiscale_tree"]
        # Use one name/None, or a list to sweep over multiple initial states.
        initial_state="tapered_gaussian",#[None,"positive_periodic_wave","mixed_sine_modes","tapered_gaussian"],
        # Options: "positive_hump", "positive_periodic_wave",
        # "mixed_sine_modes", "tapered_gaussian", "tapered_tanh"
        # Example sweep: ["positive_hump", "tapered_gaussian"]
        # expr_entcap_samples = 20000,
        # gradient_tries = 5000,
        # dir_label = "tests",
        # label = "N4_L3",
    )

    sweep_common = common.copy()
    circuits = _sweep_values(
        sweep_common.pop("unitary_circuit", None),
        "unitary_circuit",
        fcq.resolve_unitary_circuit,
    )
    initial_states = _sweep_values(
        sweep_common.pop("initial_state", None),
        "initial_state",
        fist.resolve_initial_state,
    )

    if run_mode == "single":
        base_todo = [(3, 4)]
    elif run_mode == "pairs":
        base_todo = [
            # (4, 3),
            # (4, 6),
            (5, 6),
        ]
    elif run_mode == "grid":
        ns = [4]
        ls = [6]
        base_todo = [(n, l) for n in ns for l in ls]
    else:
        raise ValueError("run_mode must be one of: 'single', 'pairs', 'grid'")

    todo = [
        (n, l, circuit, initial_state)
        for n, l in base_todo
        for circuit in circuits
        for initial_state in initial_states
    ]
    total_runs = len(todo)
    if total_runs > 1 and "label" in sweep_common:
        raise ValueError(
            "A fixed label cannot be used for a multi-run sweep because runs would "
            "overwrite each other. Remove label to use automatic unique labels."
        )

    if run_mode == "single" and total_runs == 1:
        n, l, circuit, initial_state = todo[0]
        cfg = build_sweep_cfg(
            n=n,
            l=l,
            unitary_circuit=circuit,
            initial_state=initial_state,
            **sweep_common,
        )
        run_simulation(cfg)
        return

    failures: list[tuple[int, int, str, str, str]] = []
    sweep_start = time.perf_counter()
    for run_number, (n, l, circuit, initial_state) in enumerate(todo, start=1):
        run_start = time.perf_counter()
        description = (
            f"N={n}, L={l}, circuit={circuit}, initial_state={initial_state}"
        )
        print(f"\n=== Run {run_number}/{total_runs}: Starting {description} ===")
        try:
            run_overrides = {
                **sweep_common,
                "unitary_circuit": circuit,
                "initial_state": initial_state,
            }
            if "dir_label" in sweep_common:
                run_overrides["label"] = _shared_directory_label(
                    n, l, circuit, initial_state
                )
            cfg = build_sweep_cfg(n=n, l=l, **run_overrides)
            run_simulation(cfg)
        except Exception as exc:  # keep sweep running if one pair fails
            failures.append((n, l, circuit, initial_state, str(exc)))
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
    for n, l, circuit, initial_state, msg in failures:
        print(
            f" - N={n}, L={l}, circuit={circuit}, "
            f"initial_state={initial_state}: {msg}"
        )


if __name__ == "__main__":
    main()
