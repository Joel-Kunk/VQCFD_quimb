from simulation_runner import SimConfig, run_simulation


def build_sweep_cfg(n: int, l: int, **overrides) -> SimConfig:
    return SimConfig(
        n=n,
        l=l,
        dir_label=f"N{n}",
        label=f"L{l}",
        **overrides,
    )


def main() -> None:
    run_mode = "single"  # "single" | "pairs" | "grid"

    # Common options applied to every run.
    common = dict(
        mode="variance",   # e.g. "noise_free", "adam_exact", "adam_shots", "cobyla_shots", "variance"
        compute_expr_cap=False,
    )

    if run_mode == "single":
        cfg = SimConfig(
            n=3,
            l=2,
            dir_label="test",
            label="var_N3_L2_2",
            variance_tries=500,
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
