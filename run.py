from simulation_runner import SimConfig, run_simulation


def main() -> None:
    cfg = SimConfig(
        label="N4L2",
        dir_label="var",
        n=4,
        l=2,
        mode="variance",
        variance_tries=200,
    )
    run_simulation(cfg)


if __name__ == "__main__":
    main()

# here for test runs
        # label="L2test",
        # dir_label="N2",
        # n=2,
        # l=2,
        # mode="noise_free",
        # t_total=0.2,
        # dt = 0.05,