from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable
import time

import numpy as np
import quimb.tensor as qtn
from scipy.optimize import minimize

import functions.circuits_quimb as fcq
import functions.expr_entcap as fex
import functions.functions_quimb as fqu
import functions.initial_states as fist
import functions.plots as fpl
from sim_config import SimConfig
from sim_state import SimState

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to write the simulation manifest.") from exc


@dataclass(slots=True)
class RuntimeContext:
    seed_params: list[float]
    wires: tuple[int, ...]
    n_params: int
    n: int
    l: int
    unitary_circuit: str


def ensure_dirs(cfg: SimConfig) -> None:
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)


def save_manifest(cfg: SimConfig) -> Path:
    manifest = asdict(cfg)
    manifest.update(
        {
            "results_dir": str(cfg.results_dir),
            "fig_dir": str(cfg.fig_dir),
            "data_dir": str(cfg.data_dir),
            "manifest_path": str(cfg.manifest_path),
            "n_total": cfg.n_total,
            "n_timesteps": cfg.n_timesteps,
            "resolved_unitary_circuit": cfg.resolved_unitary_circuit,
            "resolved_initial_state": cfg.resolved_initial_state,
            "number_of_parameters": cfg.number_of_parameters,
            "number_of_optimization_parameters": cfg.number_of_optimization_parameters,
        }
    )
    with cfg.manifest_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=True)
    return cfg.manifest_path


def setup_outputs(cfg: SimConfig) -> None:
    ensure_dirs(cfg)
    save_manifest(cfg)


def compute_classical_reference(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    xs = np.linspace(0, 1, cfg.n_total)
    dx = xs[1] - xs[0]
    initial_field = fist.make_initial_field(xs, cfg.resolved_initial_state)
    mod_init = np.sqrt(np.sum(initial_field**2))
    if mod_init == 0:
        raise ValueError(f"Initial state {cfg.resolved_initial_state!r} has zero norm.")
    psi_init = initial_field / mod_init

    x_plot = np.linspace(0, 1, cfg.n_total)
    t_plot = np.arange(0, cfg.t_total, cfg.dt)
    u = np.zeros((cfg.n_total, int(cfg.n_timesteps + 1)))
    u[:, 0] = fist.make_initial_field(x_plot, cfg.resolved_initial_state)

    for j in range(cfg.n_timesteps):
        for i in range(1, cfg.n_total - 1):
            u[i, j + 1] = (
                u[i, j]
                + (cfg.mu * cfg.dt / (dx**2)) * (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j])
                - (cfg.dt / (2 * dx)) * u[i, j] * (u[i + 1, j] - u[i - 1, j])
            )

    return xs, x_plot, t_plot, u, mod_init, psi_init


def build_runtime(cfg: SimConfig) -> RuntimeContext:
    if cfg.random_seed is not None:
        np.random.seed(cfg.random_seed)

    circuit_name = cfg.resolved_unitary_circuit
    fpl.plot_unitary(cfg.n, cfg.l, cfg.fig_dir, circuit_name)
    n_params = fcq.num_unitary_parameters(cfg.n, cfg.l, circuit_name)
    seed_params = (
        (np.random.random(n_params) * cfg.init_param_random_scale + cfg.init_param_random_center).tolist()
    )
    wires = tuple(range(cfg.n + 1))

    return RuntimeContext(
        seed_params=seed_params,
        wires=wires,
        n_params=n_params,
        n=cfg.n,
        l=cfg.l,
        unitary_circuit=circuit_name,
    )


def resolve_initial_params_with_fallback(
    cfg: SimConfig,
    rt: RuntimeContext,
    psi_init: np.ndarray,
    mod_init: float,
) -> tuple[np.ndarray, str, float | None]:
    try:
        resolved = np.asarray(cfg.resolve_initial_params(), dtype=float)
        expected = rt.n_params + 1
        if resolved.size != expected:
            raise ValueError(
                f"Initial parameters for circuit {rt.unitary_circuit!r} require {expected} "
                f"values including the norm parameter, received {resolved.size}."
            )
        return resolved, ("override" if cfg.initial_params is not None else "preset"), None
    except (FileNotFoundError, KeyError):
        if cfg.initial_params is not None:
            raise

    if cfg.verbose:
        print(
            f"No initial-parameter preset found for (n={cfg.n}, l={cfg.l}, "
            f"circuit={rt.unitary_circuit}, initial_state={cfg.resolved_initial_state}); "
            "running fallback initial fit optimization."
        )

    qc_t = fqu.make_state_circuit(
        cfg.n, cfg.l, rt.seed_params, rt.unitary_circuit
    )
    initial_opt = qtn.TNOptimizer(
        qc_t,
        fqu.initial_cost_quimb,
        loss_constants=dict(des=psi_init),
        autodiff_backend="autograd",
    )
    start = time.perf_counter()
    opt_initial = initial_opt.optimize(cfg.optimization_steps)
    elapsed = time.perf_counter() - start

    fitted_params = np.concatenate(
        ([mod_init], np.concatenate([np.atleast_1d(v) for v in opt_initial.get_params().values()]))
    )
    return np.asarray(fitted_params, dtype=float), "optimized_fallback", elapsed


def init_state(cfg: SimConfig, values: dict[str, float | int | str], initial_params: np.ndarray) -> SimState:
    prev = np.asarray(initial_params, dtype=float)
    return SimState(prev_params_quimb=prev, params_list_quimb=[prev.copy()], values=values)


def _build_inverse_circuits(cfg: SimConfig, rt: RuntimeContext, prev_params_quimb: np.ndarray) -> list:
    return fqu.make_inverse_reference_circuits(
        prev_params_quimb[1:], cfg.n, cfg.l, rt.unitary_circuit
    )


def _make_quimb_unitary(rt: RuntimeContext, prev_params_quimb: np.ndarray):
    return fqu.make_optimization_unitary(
        rt.n,
        rt.l,
        prev_params_quimb[1:],
        prev_params_quimb[0],
        rt.unitary_circuit,
    )


def step_noise_free(cfg: SimConfig, state: SimState, rt: RuntimeContext, dx: float, step_idx: int) -> None:
    quimb_unitary = _make_quimb_unitary(rt, state.prev_params_quimb)
    inv = _build_inverse_circuits(cfg, rt, state.prev_params_quimb)

    unitary_opt = qtn.TNOptimizer(
        quimb_unitary,
        fqu.cost_quimb,
        loss_constants=dict(
            qc1=inv[0],
            qc2=inv[1],
            qc3=inv[2],
            qc4=inv[3],
            qc5=inv[4],
            lbd0_t=state.prev_params_quimb[0],
            wires=rt.wires,
            dt=cfg.dt,
            dx=dx,
            mu=cfg.mu,
        ),
        autodiff_backend="autograd",
    )
    opt_unitary = unitary_opt.optimize(cfg.optimization_steps)

    t_params_quimb = np.concatenate([-np.atleast_1d(v) for v in opt_unitary.get_params().values()])
    next_params = t_params_quimb[::-1]
    next_params[0] = -t_params_quimb[-1]

    state.prev_params_quimb = next_params
    state.params_list_quimb.append(next_params.copy())
    state.cost_list.append(float(unitary_opt.loss_best))
    state.shots_per_timestep.append(0)
    np.save(cfg.data_dir / f"cost_iter_{cfg.full_label}_{step_idx}.npy", unitary_opt.losses)


def step_adam_exact(cfg: SimConfig, state: SimState, rt: RuntimeContext, dx: float, step_idx: int) -> None:
    curr_params = state.prev_params_quimb.copy()
    quimb_unitary = _make_quimb_unitary(rt, state.prev_params_quimb)
    inv = _build_inverse_circuits(cfg, rt, state.prev_params_quimb)
    unitary0 = fqu.make_optimization_unitary2(
        rt.n,
        rt.l,
        state.prev_params_quimb[1:],
        state.prev_params_quimb[0],
        rt.unitary_circuit,
    )
    trees = [fqu.build_local_exp_tree(unitary0, circuit, rt.wires) for circuit in inv]

    t = 0
    m = np.zeros(len(state.prev_params_quimb))
    v = np.zeros(len(state.prev_params_quimb))
    cost_iters = []
    params_iters = [curr_params.copy()]
    cost_iters.append(
        fqu.cost_quimb(
            quimb_unitary,
            inv[0],
            inv[1],
            inv[2],
            inv[3],
            inv[4],
            state.prev_params_quimb[0],
            rt.wires,
            cfg.dt,
            dx,
            cfg.mu,
        )
    )
    quimb_unitary = _make_quimb_unitary(rt, state.prev_params_quimb)

    dummy = np.zeros(cfg.patience)
    dummy[-1] = 1e5
    stop_crit = 100.0
    while t < cfg.max_iter and stop_crit > cfg.plat_tol:
        t += 1

        grad = fqu.whole_grad_param_shift(
            curr_params,
            inv[0],
            inv[1],
            inv[2],
            inv[3],
            inv[4],
            state.prev_params_quimb[0],
            rt.wires,
            cfg.dt,
            dx,
            cfg.mu,
            cfg.n,
            cfg.l,
            trees,
            rt.unitary_circuit,
        )
        grad[0] = fqu.grad_mod(
            quimb_unitary, inv[0], inv[1], inv[2], inv[3], inv[4], state.prev_params_quimb[0], rt.wires, cfg.dt, dx, cfg.mu
        )

        curr_params, m, v = fqu.adam_update(curr_params, grad, m, v, t)
        quimb_unitary = _make_quimb_unitary(rt, curr_params)
        cost_iters.append(
            fqu.cost_quimb(
                quimb_unitary,
                inv[0],
                inv[1],
                inv[2],
                inv[3],
                inv[4],
                state.prev_params_quimb[0],
                rt.wires,
                cfg.dt,
                dx,
                cfg.mu,
            )
        )
        params_iters.append(curr_params.copy())
        cost_plateau = np.concatenate([dummy, np.asarray(cost_iters)])[-cfg.patience :]
        stop_crit = float(np.max(cost_plateau) - np.min(cost_plateau))

        if cfg.verbose:
            print(f"Iteration {t}/{cfg.max_iter}: Cost = {cost_iters[-1]}, Plat diff = {stop_crit:.3e}/{cfg.plat_tol}")

    state.params_list_quimb.append(curr_params.copy())
    np.save(cfg.data_dir / f"cost_iter_{cfg.full_label}_{step_idx}.npy", np.asarray(cost_iters))
    np.save(cfg.data_dir / f"params_iter_{cfg.full_label}_{step_idx}.npy", np.asarray(params_iters))
    state.cost_list.append(float(cost_iters[-1]))
    state.prev_params_quimb = curr_params.copy()
    state.shots_per_timestep.append(0)


def step_adam_shots(cfg: SimConfig, state: SimState, rt: RuntimeContext, dx: float, step_idx: int) -> None:
    curr_params = state.prev_params_quimb.copy()
    quimb_unitary = _make_quimb_unitary(rt, state.prev_params_quimb)
    inv = _build_inverse_circuits(cfg, rt, state.prev_params_quimb)

    t = 0
    m = np.zeros(len(state.prev_params_quimb))
    v = np.zeros(len(state.prev_params_quimb))
    cost_iters = []
    params_iters = [curr_params.copy()]
    cost_iters.append(
        fqu.cost_quimb(
            quimb_unitary,
            inv[0],
            inv[1],
            inv[2],
            inv[3],
            inv[4],
            state.prev_params_quimb[0],
            rt.wires,
            cfg.dt,
            dx,
            cfg.mu,
        )
    )
    quimb_unitary = _make_quimb_unitary(rt, state.prev_params_quimb)

    dummy = np.zeros(cfg.patience)
    dummy[-1] = 1e5
    stop_crit = 100.0
    while t < cfg.max_iter and stop_crit > cfg.plat_tol:
        t += 1

        grad = fqu.grad_param_shift(
            curr_params,
            inv[0],
            inv[1],
            inv[2],
            inv[3],
            inv[4],
            state.prev_params_quimb[0],
            rt.wires,
            cfg.dt,
            dx,
            cfg.mu,
            cfg.shots,
            cfg.n,
            cfg.l,
            rt.unitary_circuit,
        )
        grad[0] = fqu.grad_mod_shots(
            quimb_unitary, inv[0], inv[1], inv[2], inv[3], inv[4], state.prev_params_quimb[0], rt.wires, cfg.dt, dx, cfg.mu, cfg.shots
        )

        curr_params, m, v = fqu.adam_update(curr_params, grad, m, v, t)
        quimb_unitary = _make_quimb_unitary(rt, curr_params)
        cost_iters.append(
            fqu.cost_quimb(
                quimb_unitary,
                inv[0],
                inv[1],
                inv[2],
                inv[3],
                inv[4],
                state.prev_params_quimb[0],
                rt.wires,
                cfg.dt,
                dx,
                cfg.mu,
            )
        )
        params_iters.append(curr_params.copy())
        cost_plateau = np.concatenate([dummy, np.asarray(cost_iters)])[-cfg.patience :]
        stop_crit = float(np.max(cost_plateau) - np.min(cost_plateau))

        if cfg.verbose:
            print(f"Iteration {t}/{cfg.max_iter}: Cost = {cost_iters[-1]}, Plat diff = {stop_crit:.3e}/{cfg.plat_tol}")

    state.params_list_quimb.append(curr_params.copy())
    np.save(cfg.data_dir / f"cost_iter_{cfg.full_label}_{step_idx}.npy", np.asarray(cost_iters))
    np.save(cfg.data_dir / f"params_iter_{cfg.full_label}_{step_idx}.npy", np.asarray(params_iters))
    state.cost_list.append(float(cost_iters[-1]))
    state.prev_params_quimb = curr_params.copy()
    shots_this_timestep = int(t * (10 * len(curr_params) - 5) * cfg.shots)
    state.shots_per_timestep.append(shots_this_timestep)


def step_cobyla_shots(cfg: SimConfig, state: SimState, rt: RuntimeContext, dx: float, step_idx: int) -> None:
    curr_params = state.prev_params_quimb.copy()
    inv = _build_inverse_circuits(cfg, rt, curr_params)
    bounds = [(-10000, 100000)] + [(-2 * np.pi, 2 * np.pi)] * rt.n_params
    cost_iters = []
    params_iters = [curr_params.copy()]

    result = minimize(
        fqu.cost_quimb_shots2,
        curr_params,
        args=(
            inv[0],
            inv[1],
            inv[2],
            inv[3],
            inv[4],
            state.prev_params_quimb[0],
            rt.wires,
            cfg.dt,
            dx,
            cfg.mu,
            cfg.shots,
            cost_iters,
            cfg.n,
            cfg.l,
            params_iters,
            rt.unitary_circuit,
        ),
        method="COBYLA",
        bounds=bounds,
        tol=cfg.cobyla_tol,
        options={"maxiter": cfg.cobyla_maxiter, "disp": cfg.verbose},
    )

    state.prev_params_quimb = np.asarray(result.x, dtype=float)
    state.num_evals.append(int(result.nfev))
    state.params_list_quimb.append(state.prev_params_quimb.copy())
    state.cost_list.append(float(result.fun))
    state.shots_per_timestep.append(int(5 * cfg.shots * result.nfev))
    np.save(cfg.data_dir / f"cost_iter_{cfg.full_label}_{step_idx}.npy", np.asarray(cost_iters))
    np.save(cfg.data_dir / f"params_iter_{cfg.full_label}_{step_idx}.npy", np.asarray(params_iters))


STEPPERS: dict[str, Callable[[SimConfig, SimState, RuntimeContext, float, int], None]] = {
    "noise_free": step_noise_free,
    "adam_exact": step_adam_exact,
    "adam_shots": step_adam_shots,
    "cobyla_shots": step_cobyla_shots,
}


def build_values(cfg: SimConfig) -> dict[str, float | int | str]:
    return {
        "n": cfg.n,
        "l": cfg.l,
        "shots": cfg.shots,
        "max_iter": cfg.max_iter,
        "t_total": cfg.t_total,
        "mu": cfg.mu,
        "dt": cfg.dt,
        "plat_tol": cfg.plat_tol,
        "patience": cfg.patience,
        "unitary_circuit": cfg.resolved_unitary_circuit,
        "initial_state": cfg.resolved_initial_state,
        "number_of_parameters": cfg.number_of_parameters,
        "number_of_optimization_parameters": cfg.number_of_optimization_parameters,
    }


def run_variance_analysis(cfg: SimConfig) -> SimState:
    setup_outputs(cfg)
    xs, _x_plot, _t_plot, _u, mod_init, psi_init = compute_classical_reference(cfg)
    dx = xs[1] - xs[0]

    rt = build_runtime(cfg)
    values = build_values(cfg)
    initial_params, initial_params_source, initial_params_fit_time = resolve_initial_params_with_fallback(
        cfg, rt, psi_init, mod_init
    )
    values["initial_params_source"] = initial_params_source
    if initial_params_fit_time is not None:
        values["initial_params_fit_time_s"] = float(initial_params_fit_time)

    state = init_state(cfg, values, initial_params)

    params_ref = initial_params.copy()
    params_ref[0] = 1.0
    wires = rt.wires
    n_params = rt.n_params
    qc1, qc2, qc3, qc4, qc5 = fqu.make_inverse_reference_circuits(
        params_ref[1:], cfg.n, cfg.l, rt.unitary_circuit
    )

    unitary0 = fcq.make_optimization_unitary2(
        cfg.n,
        cfg.l,
        params_ref[1:],
        params_ref[0],
        rt.unitary_circuit,
    )

    trees = [
    fqu.build_local_exp_tree(unitary0, qc1, wires),
    fqu.build_local_exp_tree(unitary0, qc2, wires),
    fqu.build_local_exp_tree(unitary0, qc3, wires),
    fqu.build_local_exp_tree(unitary0, qc4, wires),
    fqu.build_local_exp_tree(unitary0, qc5, wires),
]

    grads3: list[float] = []
    start = time.perf_counter()

    for i in range(cfg.variance_tries):
        params_rand = np.random.random(n_params + 1) * 2 * np.pi
        params_rand[0] = 1.0
        grad3 = fqu.whole_grad_param_shift(
            params_rand,
            qc1,
            qc2,
            qc3,
            qc4,
            qc5,
            params_ref[0],
            wires,
            cfg.dt,
            dx,
            cfg.mu,
            cfg.n,
            cfg.l,
            trees,
            rt.unitary_circuit,
        )
        grads3.extend(np.asarray(grad3[1:], dtype=float).tolist())
        if cfg.verbose:
            print(f"variance try = {i + 1}/{cfg.variance_tries}")
    elapsed = time.perf_counter() - start

    variance = float(np.var(grads3)) if grads3 else 0.0
    state.values["variance"] = variance
    state.values["variance_tries"] = int(cfg.variance_tries)
    state.values["variance_num_samples"] = int(len(grads3))
    state.values["variance_runtime_s"] = float(elapsed)
    state.times.append(elapsed)
    state.shots_per_timestep.append(0)
    state.values["shots_used_per_timestep"] = [0]
    state.values["shots_used_total"] = 0

    np.save(cfg.data_dir / f"variance_{cfg.full_label}.npy", np.asarray(variance))
    np.save(cfg.data_dir / f"variance_grads_{cfg.full_label}.npy", np.asarray(grads3, dtype=float))
    with (cfg.data_dir / f"values_{cfg.full_label}.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(state.values, fh, sort_keys=False)

    if cfg.verbose:
        print(cfg.dir_label)
        print(cfg.label)

    return state


def run_simulation(cfg: SimConfig) -> SimState:
    if cfg.verbose:
        print(cfg.dir_label)
        print(cfg.label)
        
    if cfg.mode == "variance":
        return run_variance_analysis(cfg)
    if cfg.mode not in STEPPERS:
        raise ValueError(f"Unknown mode '{cfg.mode}'. Expected one of: {', '.join([*STEPPERS, 'variance'])}")

    setup_outputs(cfg)
    xs, _x_plot, _t_plot, u, mod_init, psi_init = compute_classical_reference(cfg)
    dx = xs[1] - xs[0]
    fpl.plot_calssical_evolution(xs, u, cfg.full_label, cfg.fig_dir)

    rt = build_runtime(cfg)
    values = build_values(cfg)
    initial_params, initial_params_source, initial_params_fit_time = resolve_initial_params_with_fallback(
        cfg, rt, psi_init, mod_init
    )
    values["initial_params_source"] = initial_params_source
    if initial_params_fit_time is not None:
        values["initial_params_fit_time_s"] = float(initial_params_fit_time)

    state = init_state(cfg, values, initial_params)

    if cfg.compute_expr_cap:
        circ = fex.unitary_pure(cfg.n, cfg.l, rt.unitary_circuit)
        exp_and_cap = fex.expr_and_ent_cap(circ, cfg.expr_entcap_samples, cfg.expr_bins)
        state.values["expressibility"] = float(exp_and_cap[0])
        state.values["entangling_capability"] = float(exp_and_cap[1])

    initial_fit_mse = fpl.plot_initialfit(
        xs,
        mod_init,
        psi_init,
        state.prev_params_quimb,
        cfg.n,
        cfg.l,
        cfg.n_total,
        cfg.full_label,
        cfg.fig_dir,
        rt.unitary_circuit,
    )
    state.values["initial_fit_mse"] = float(initial_fit_mse)

    stepper = STEPPERS[cfg.mode]
    for i in range(cfg.n_timesteps):
        if cfg.verbose:
            print(f"Timestep: {i}")
        start = time.perf_counter()
        stepper(cfg, state, rt, dx, i)
        state.times.append(time.perf_counter() - start)

    state.values["shots_used_per_timestep"] = [int(x) for x in state.shots_per_timestep]
    state.values["shots_used_total"] = int(sum(state.shots_per_timestep))

    fpl.plot_final(
        xs,
        u[:, -1],
        state.params_list_quimb[-1],
        cfg.n,
        cfg.l,
        cfg.n_total,
        cfg.full_label,
        cfg.fig_dir,
        rt.unitary_circuit,
    )
    fpl.plot_evolution(
        xs,
        state.params_list_quimb,
        cfg.n,
        cfg.l,
        cfg.n_total,
        cfg.full_label,
        cfg.fig_dir,
        rt.unitary_circuit,
    )

    np.save(cfg.data_dir / f"params_{cfg.full_label}.npy", np.asarray(state.params_list_quimb))
    np.save(cfg.data_dir / f"costs_{cfg.full_label}.npy", np.asarray(state.cost_list))
    np.save(cfg.data_dir / f"times_{cfg.full_label}.npy", np.asarray(state.times))
    with (cfg.data_dir / f"values_{cfg.full_label}.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(state.values, fh, sort_keys=False)
    if state.num_evals:
        np.save(cfg.data_dir / f"num_evals_{cfg.full_label}.npy", np.asarray(state.num_evals))

    if cfg.verbose:
        print(cfg.dir_label)
        print(cfg.label)

    return state
