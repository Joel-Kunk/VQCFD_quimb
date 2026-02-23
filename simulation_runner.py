from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable
import time

import numpy as np
import quimb.tensor as qtn
from scipy.optimize import minimize

import functions.expr_entcap as fex
import functions.functions_qiskit as fqi
import functions.functions_quimb as fqu
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
    circuits: list
    wires: tuple[int, ...]
    n_params: int
    num_gates_in_u: int
    base_unitary: object


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
    mod_init = np.sqrt(np.sum(np.sin(2 * np.pi * xs) ** 2))
    psi_init = np.sin(2 * np.pi * xs) / mod_init

    x_plot = np.linspace(0, 1, cfg.n_total)
    t_plot = np.arange(0, cfg.t_total, cfg.dt)
    u = np.zeros((cfg.n_total, int(cfg.n_timesteps + 1)))
    u[:, 0] = np.sin(2 * np.pi * x_plot)

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

    fpl.plot_unitary(fqi.unitary(cfg.n, cfg.l), cfg.fig_dir)
    n_params = fqi.circuit(cfg.n, cfg.l).num_parameters
    seed_params = (
        (np.random.random(n_params) * cfg.init_param_random_scale + cfg.init_param_random_center).tolist()
    )
    circuits = fqi.get_circuits(seed_params, cfg.n, cfg.l)
    wires = tuple(range(cfg.n + 1))
    base_unitary = fqi.unitary(cfg.n, cfg.l)
    num_gates_in_u = len(fqi.circuit(cfg.n, cfg.l).data) - 1

    return RuntimeContext(
        seed_params=seed_params,
        circuits=circuits,
        wires=wires,
        n_params=n_params,
        num_gates_in_u=num_gates_in_u,
        base_unitary=base_unitary,
    )


def init_state(cfg: SimConfig, values: list[float]) -> SimState:
    prev = np.asarray(cfg.initial_params, dtype=float)
    return SimState(prev_params_quimb=prev, params_list_quimb=[prev.copy()], values=values)


def _build_inverse_circuits(cfg: SimConfig, rt: RuntimeContext, prev_params_quimb: np.ndarray) -> list:
    return [
        fqu.qisikit_to_quimb_inverse(
            rt.circuits[k][0].assign_parameters(rt.seed_params),
            rt.n_params,
            rt.num_gates_in_u,
            prev_params_quimb[1:],
        )
        for k in range(5)
    ]


def _make_quimb_unitary(rt: RuntimeContext, prev_params_quimb: np.ndarray):
    return fqu.qiskit_to_quimb_unitary(rt.base_unitary, prev_params_quimb[1:], prev_params_quimb[0])


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
    np.save(cfg.data_dir / f"cost_iter_{cfg.label}_{step_idx}.npy", unitary_opt.losses)


def step_adam_exact(cfg: SimConfig, state: SimState, rt: RuntimeContext, dx: float, step_idx: int) -> None:
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

        grad = fqu.whole_grad_param_shift(
            curr_params, inv[0], inv[1], inv[2], inv[3], inv[4], state.prev_params_quimb[0], rt.wires, cfg.dt, dx, cfg.mu, cfg.n, cfg.l
        )
        grad[0] = fqu.grad_mod(
            quimb_unitary, inv[0], inv[1], inv[2], inv[3], inv[4], state.prev_params_quimb[0], rt.wires, cfg.dt, dx, cfg.mu
        )

        curr_params, m, v = fqu.adam_update(curr_params, grad, m, v, t)
        quimb_unitary = fqu.qiskit_to_quimb_unitary(rt.base_unitary, curr_params[1:], curr_params[0])
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
    np.save(cfg.data_dir / f"cost_iters_{cfg.label}_{step_idx}.npy", np.asarray(cost_iters))
    np.save(cfg.data_dir / f"params_iters_{cfg.label}_{step_idx}.npy", np.asarray(params_iters))
    state.cost_list.append(float(cost_iters[-1]))
    state.prev_params_quimb = curr_params.copy()


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
        )
        grad[0] = fqu.grad_mod_shots(
            quimb_unitary, inv[0], inv[1], inv[2], inv[3], inv[4], state.prev_params_quimb[0], rt.wires, cfg.dt, dx, cfg.mu, cfg.shots
        )

        curr_params, m, v = fqu.adam_update(curr_params, grad, m, v, t)
        quimb_unitary = fqu.qiskit_to_quimb_unitary(rt.base_unitary, curr_params[1:], curr_params[0])
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
    np.save(cfg.data_dir / f"cost_iters_{cfg.label}_{step_idx}.npy", np.asarray(cost_iters))
    np.save(cfg.data_dir / f"params_iters_{cfg.label}_{step_idx}.npy", np.asarray(params_iters))
    state.cost_list.append(float(cost_iters[-1]))
    state.prev_params_quimb = curr_params.copy()


def step_cobyla_shots(cfg: SimConfig, state: SimState, rt: RuntimeContext, dx: float, step_idx: int) -> None:
    curr_params = state.prev_params_quimb.copy()
    inv = _build_inverse_circuits(cfg, rt, curr_params)
    n_params = fqi.unitary(cfg.n, cfg.l).num_parameters
    bounds = [(-10000, 100000)] + [(-2 * np.pi, 2 * np.pi)] * n_params
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
    np.save(cfg.data_dir / f"cost_iters_{cfg.label}_{step_idx}.npy", np.asarray(cost_iters))
    np.save(cfg.data_dir / f"params_iters_{cfg.label}_{step_idx}.npy", np.asarray(params_iters))


STEPPERS: dict[str, Callable[[SimConfig, SimState, RuntimeContext, float, int], None]] = {
    "noise_free": step_noise_free,
    "adam_exact": step_adam_exact,
    "adam_shots": step_adam_shots,
    "cobyla_shots": step_cobyla_shots,
}


def build_values(cfg: SimConfig) -> list[float]:
    return [
        cfg.n,
        cfg.l,
        cfg.shots,
        cfg.max_iter,
        cfg.grad_tol,
        cfg.t_total,
        cfg.mu,
        cfg.dt,
        cfg.plat_tol,
        cfg.patience,
    ]


def run_simulation(cfg: SimConfig) -> SimState:
    if cfg.mode not in STEPPERS:
        raise ValueError(f"Unknown mode '{cfg.mode}'. Expected one of: {', '.join(STEPPERS)}")

    setup_outputs(cfg)
    xs, _x_plot, _t_plot, u, mod_init, psi_init = compute_classical_reference(cfg)
    dx = xs[1] - xs[0]
    fpl.plot_calssical_evolution(xs, u, cfg.label, cfg.fig_dir)

    values = build_values(cfg)
    rt = build_runtime(cfg)
    state = init_state(cfg, values)

    if cfg.compute_expr_cap:
        circ = fex.unitary_pure(cfg.n, cfg.l)
        exp_and_cap = fex.expr_and_ent_cap(circ, cfg.expr_samples, cfg.entcap_samples)
        state.values.append(float(exp_and_cap[0]))
        state.values.append(float(exp_and_cap[1]))

    initial_fit_mse = fpl.plot_initialfit(
        xs, mod_init, psi_init, state.prev_params_quimb, cfg.n, cfg.l, cfg.n_total, cfg.label, cfg.fig_dir
    )
    state.values.append(float(initial_fit_mse))

    stepper = STEPPERS[cfg.mode]
    for i in range(cfg.n_timesteps):
        if cfg.verbose:
            print(f"Timestep: {i}")
        start = time.perf_counter()
        stepper(cfg, state, rt, dx, i)
        state.times.append(time.perf_counter() - start)

    fpl.plot_final(xs, u[:, -1], state.params_list_quimb[-1], cfg.n, cfg.l, cfg.n_total, cfg.label, cfg.fig_dir)
    fpl.plot_evolution(xs, state.params_list_quimb, cfg.n, cfg.l, cfg.n_total, cfg.label, cfg.fig_dir)

    np.save(cfg.data_dir / f"params_{cfg.label}.npy", np.asarray(state.params_list_quimb))
    np.save(cfg.data_dir / f"costs_{cfg.label}.npy", np.asarray(state.cost_list))
    np.save(cfg.data_dir / f"times_{cfg.label}.npy", np.asarray(state.times))
    np.save(cfg.data_dir / f"values_{cfg.label}.npy", np.asarray(state.values))
    if state.num_evals:
        np.save(cfg.data_dir / f"num_evals_{cfg.label}.npy", np.asarray(state.num_evals))

    if cfg.verbose:
        print(cfg.dir_label)
        print(cfg.label)

    return state
