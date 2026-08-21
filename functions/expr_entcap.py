import time

import numpy as np
from scipy.stats import entropy
import quimb as qu
import quimb.tensor as qtn

import functions.circuits_quimb as fcq
import functions.functions_quimb as fqu


def _circuit_parts(circ: tuple[int, int] | tuple[int, int, str]) -> tuple[int, int, str]:
    if len(circ) == 2:
        n, l = circ
        return n, l, fcq.DEFAULT_UNITARY_CIRCUIT
    n, l, unitary_circuit = circ
    return n, l, fcq.resolve_unitary_circuit(unitary_circuit)


def count_total_parameters(circ: tuple[int, int] | tuple[int, int, str]) -> int:
    n, l, unitary_circuit = _circuit_parts(circ)
    return fcq.num_unitary_parameters(n, l, unitary_circuit)


def select_numbers_np(N, position, value):
    """
    select positions in wich the bit at position takes value,from numbers in 0 to N (N should = 2**n)
    """
    xs = np.arange(N)
    bits = (xs >> position) & 1
    return xs[bits == value]


def special_distance(u, v, pos):
    val = 0
    dim = len(u)
    inds1 = select_numbers_np(dim, pos, 0)
    inds2 = select_numbers_np(dim, pos, 1)
    for i in range(int(dim / 2)):
        for j in range(i):
            val += np.abs(u[inds1[i]] * v[inds2[j]] - u[inds1[j]] * v[inds2[i]]) ** 2

    return val


def expressibility(circ, tries, n_bins, *, optimize="auto-hq"):
    n, l, unitary_circuit = _circuit_parts(circ)
    num_params = count_total_parameters(circ)
    fidelities = []
    for _ in range(tries):
        params_t1 = np.random.random(num_params) * 2 * np.pi
        params_t2 = np.random.random(num_params) * 2 * np.pi
        uni1 = fcq.make_pure_unitary_circuit(
            n, l, params_t1, parametrize=False, unitary_circuit=unitary_circuit
        )
        uni2 = fcq.make_pure_unitary_circuit(
            n, l, params_t2, parametrize=False, unitary_circuit=unitary_circuit
        )
        fidelities.append(
            np.abs(
                np.vdot(
                    uni1.to_dense(optimize=optimize),
                    uni2.to_dense(optimize=optimize),
                )
            )
            ** 2
        )

    p_pqc = np.histogram(fidelities, bins=n_bins)
    bins = p_pqc[1]
    hist_pqc = p_pqc[0] / p_pqc[0].sum()
    p_kl = np.zeros(n_bins)
    N_t = 2**n
    for i in range(n_bins):
        val_t = (bins[i] + bins[i + 1]) / 2
        p_kl[i] = (N_t - 1) * (1 - val_t) ** (N_t - 2)
    hist_kl = p_kl / p_kl.sum()

    kl = entropy(hist_pqc, hist_kl)
    return kl


def entangling_capability(circ, tries, *, optimize="auto-hq"):
    n, l, unitary_circuit = _circuit_parts(circ)
    num_params = count_total_parameters(circ)
    ent_cap = 0
    for _ in range(tries):
        params_t = np.random.random(num_params) * 2 * np.pi
        uni = fcq.make_pure_unitary_circuit(
            n, l, params_t, parametrize=False, unitary_circuit=unitary_circuit
        )
        vec_t = uni.to_dense(optimize=optimize)
        for j in range(n):
            ent_cap += special_distance(vec_t, vec_t, j)

    ent_cap = ent_cap * 4 / (n * tries)
    return ent_cap


def _format_elapsed(seconds: float) -> str:
    total_centiseconds = max(0, round(float(seconds) * 100))
    hours, remainder = divmod(total_centiseconds, 360_000)
    minutes, centiseconds = divmod(remainder, 6_000)
    whole_seconds, centiseconds = divmod(centiseconds, 100)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def expr_and_ent_cap(circ, tries, n_bins, *, verbose=True, optimize="auto-hq"):
    n, l, unitary_circuit = _circuit_parts(circ)
    num_params = count_total_parameters(circ)
    fidelities = []
    ent_cap = 0.0
    start = time.perf_counter()

    if verbose:
        print(f"expression samples = 0/{tries}, elapsed = {_format_elapsed(0)}")

    for i in range(tries):
        params_t1 = np.random.random(num_params) * 2 * np.pi
        params_t2 = np.random.random(num_params) * 2 * np.pi
        uni1 = fcq.make_pure_unitary_circuit(
            n, l, params_t1, parametrize=False, unitary_circuit=unitary_circuit
        )
        uni2 = fcq.make_pure_unitary_circuit(
            n, l, params_t2, parametrize=False, unitary_circuit=unitary_circuit
        )
        vec_t1 = uni1.to_dense(optimize=optimize)
        vec_t2 = uni2.to_dense(optimize=optimize)
        fidelities.append(np.abs(np.vdot(vec_t1, vec_t2)) ** 2)

        for j in range(n):
            ent_cap += special_distance(vec_t1, vec_t1, j)

        completed = i + 1
        if verbose and (completed % 1000 == 0 or completed == tries):
            elapsed = _format_elapsed(time.perf_counter() - start)
            print(f"expression samples = {completed}/{tries}, elapsed = {elapsed}")

    p_pqc = np.histogram(fidelities, bins=n_bins)
    bins = p_pqc[1]
    hist_pqc = p_pqc[0] / p_pqc[0].sum()
    p_kl = np.zeros(n_bins)
    N_t2 = 2**n
    for i in range(n_bins):
        val_t = (bins[i] + bins[i + 1]) / 2
        p_kl[i] = (N_t2 - 1) * (1 - val_t) ** (N_t2 - 2)
    hist_kl = p_kl / p_kl.sum()

    kl = entropy(hist_pqc, hist_kl)
    result_t = ent_cap * 4 / (n * tries)

    return kl, result_t[0] if hasattr(result_t, "__len__") else result_t


def initial_fit(circ, PSI_init, MOD_init):
    n, l, unitary_circuit = _circuit_parts(circ)
    n_params = count_total_parameters(circ)
    params = (np.random.random(n_params) + (np.pi - 0.5)).tolist()

    qc_t = fcq.make_pure_unitary_circuit(
        n, l, params, parametrize=True, unitary_circuit=unitary_circuit
    )
    initial_opt = qtn.TNOptimizer(
        qc_t,
        fqu.initial_cost_quimb,
        loss_constants=dict(des=PSI_init),
        autodiff_backend="autograd",
    )

    opt_initial = initial_opt.optimize(1000)
    initial_params = np.concatenate(([MOD_init], np.concatenate([np.atleast_1d(v) for v in opt_initial.get_params().values()])))
    best_value = initial_opt.loss_best
    iterations = len(initial_opt.losses)
    return best_value, iterations, initial_params


def test_functions(N_total):
    Xs = np.linspace(0, 1, N_total)

    funcs = []

    MOD_init = np.sqrt(np.sum(np.sin(2 * np.pi * Xs) ** 2))
    f1 = np.sin(2 * np.pi * Xs) / MOD_init
    f1 = np.append(f1, MOD_init)
    funcs.append(f1)

    MOD_init = np.sqrt(np.sum(np.cos(2 * np.pi * Xs) ** 2))
    f1 = np.cos(2 * np.pi * Xs) / MOD_init
    f1 = np.append(f1, MOD_init)
    funcs.append(f1)

    MOD_init = np.sqrt(np.sum(np.cos(4 * np.pi * Xs) ** 2))
    f1 = np.cos(4 * np.pi * Xs) / MOD_init
    f1 = np.append(f1, MOD_init)
    funcs.append(f1)

    MOD_init = np.sqrt(np.sum(np.sin(4 * np.pi * Xs) ** 2))
    f1 = np.sin(4 * np.pi * Xs) / MOD_init
    f1 = np.append(f1, MOD_init)
    funcs.append(f1)

    f1 = (Xs - 0.5) ** 2
    MOD_init = np.sqrt(np.sum(f1**2))
    f1 = f1 / MOD_init
    f1 = np.append(f1, MOD_init)
    funcs.append(f1)

    f1 = Xs * 2
    MOD_init = np.sqrt(np.sum(f1**2))
    f1 = f1 / MOD_init
    f1 = np.append(f1, MOD_init)
    funcs.append(f1)

    f1 = (Xs - 0.5) ** 3
    MOD_init = np.sqrt(np.sum(f1**2))
    f1 = f1 / MOD_init
    f1 = np.append(f1, MOD_init)
    funcs.append(f1)

    return funcs


def unitary_pure(qubits, layers, unitary_circuit=None):
    return (qubits, layers, fcq.resolve_unitary_circuit(unitary_circuit))


def gradient_of_cost(
    params1,
    params2,
    mod,
    idx,
    N,
    L,
    dt,
    dx,
    mu,
    N_params,
    _num_gates_in_U,
    unitary_circuit=None,
):
    params1_1 = params1.copy()
    params1_1[idx] += fqu.HADAMARD_TEST_PARAMETER_SHIFT
    params1_2 = params1.copy()
    params1_2[idx] -= fqu.HADAMARD_TEST_PARAMETER_SHIFT

    quimb_unitary1 = fqu.make_optimization_unitary(
        N, L, params1_1, mod, unitary_circuit
    )
    quimb_unitary2 = fqu.make_optimization_unitary(
        N, L, params1_2, mod, unitary_circuit
    )

    qc1, qc2, qc3, qc4, qc5 = fqu.make_inverse_reference_circuits(
        params2, N, L, unitary_circuit
    )
    wires = tuple(range(int(N) + 1))

    c1 = fqu.embed_circuit(quimb_unitary1, qc1, wires).local_expectation(qu.pauli("Z"), 0, simplify_sequence="RC")
    c2 = fqu.embed_circuit(quimb_unitary1, qc2, wires).local_expectation(qu.pauli("Z"), 0, simplify_sequence="RC")
    c3 = fqu.embed_circuit(quimb_unitary1, qc3, wires).local_expectation(qu.pauli("Z"), 0, simplify_sequence="RC")
    c4 = fqu.embed_circuit(quimb_unitary1, qc4, wires).local_expectation(qu.pauli("Z"), 0, simplify_sequence="RC")
    c5 = fqu.embed_circuit(quimb_unitary1, qc5, wires).local_expectation(qu.pauli("Z"), 0, simplify_sequence="RC")

    c12 = fqu.embed_circuit(quimb_unitary2, qc1, wires).local_expectation(qu.pauli("Z"), 0, simplify_sequence="RC")
    c22 = fqu.embed_circuit(quimb_unitary2, qc2, wires).local_expectation(qu.pauli("Z"), 0, simplify_sequence="RC")
    c32 = fqu.embed_circuit(quimb_unitary2, qc3, wires).local_expectation(qu.pauli("Z"), 0, simplify_sequence="RC")
    c42 = fqu.embed_circuit(quimb_unitary2, qc4, wires).local_expectation(qu.pauli("Z"), 0, simplify_sequence="RC")
    c52 = fqu.embed_circuit(quimb_unitary2, qc5, wires).local_expectation(qu.pauli("Z"), 0, simplify_sequence="RC")

    dc1 = fqu.hadamard_test_parameter_shift(c1, c12)
    dc2 = fqu.hadamard_test_parameter_shift(c2, c22)
    dc3 = fqu.hadamard_test_parameter_shift(c3, c32)
    dc4 = fqu.hadamard_test_parameter_shift(c4, c42)
    dc5 = fqu.hadamard_test_parameter_shift(c5, c52)

    cost = -2 * np.real(
        np.conjugate(mod)
        * mod
        * (
            dc1
            + ((dt * mu / (dx**2)) * (dc2 - 2 * dc1 + dc3))
            - ((dt * np.conjugate(mod) / (2 * dx)) * (dc4 - dc5))
        )
    )

    return np.real(cost).astype(np.float64).reshape(())
