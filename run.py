from simulation_runner import SimConfig, run_simulation


def main() -> None:
    cfg = SimConfig(
        label="L3",
        dir_label="N3",
        n=3,
        l=3,
        mode="noise_free",
    )
    run_simulation(cfg)


if __name__ == "__main__":
    main()
