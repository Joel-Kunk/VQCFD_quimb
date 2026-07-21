import quimb as qu
import quimb.tensor as qtn
import autograd.numpy as anp
import numpy as np
import functions.circuits_quimb as fcq


HADAMARD_TEST_PARAMETER_SHIFT = np.pi


def hadamard_test_parameter_shift(value_plus, value_minus):
    """Return the exact two-evaluation derivative for a unitary angle.

    In the Hadamard-test objective used here, every current circuit parameter
    occurs once in a rotation matrix and therefore enters the objective as

        f(theta) = a + b cos(theta / 2) + c sin(theta / 2).

    Thus ``f'(theta) = (f(theta + pi) - f(theta - pi)) / 4``.  Choosing the
    shift pi keeps the evaluation count at two and maximizes the difference
    signal, which is also preferable when the two values are shot estimates.
    """
    return (value_plus - value_minus) / 4


def make_state_circuit(qubits, layers, params, unitary_circuit=None):
    return fcq.make_state_circuit(
        qubits,
        layers,
        params,
        parametrize=True,
        unitary_circuit=unitary_circuit,
    )


def make_optimization_unitary(qubits, layers, params_insert, lbd0, unitary_circuit=None):
    return fcq.make_optimization_unitary(
        qubits,
        layers,
        params_insert,
        lbd0,
        unitary_circuit=unitary_circuit,
    )

def make_optimization_unitary2(qubits, layers, params_insert, lbd0, unitary_circuit=None):
    return fcq.make_optimization_unitary2(
        qubits,
        layers,
        params_insert,
        lbd0,
        unitary_circuit=unitary_circuit,
    )


def make_inverse_reference_circuits(prev_params, qubits, layers, unitary_circuit=None):
    return fcq.make_inverse_reference_circuits(
        prev_params,
        qubits,
        layers,
        unitary_circuit=unitary_circuit,
    )

def embed_circuit(qc_small,qc_big,qubits):
    qcn = qc_big.copy()
    for g in qc_small.gates:
        qcn.apply_gate(g.label,qubits = g.qubits ,params = g.params,contract = False)
    return qcn


def cost_quimb(unitary,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu):

    c1 = embed_circuit(unitary,qc1,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c2 = embed_circuit(unitary,qc2,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c3 = embed_circuit(unitary,qc3,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c4 = embed_circuit(unitary,qc4,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c5 = embed_circuit(unitary,qc5,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")

    lbd0 = unitary.gates[-1].params[0]

    cost = anp.abs(lbd0)**2 - 2*anp.real(anp.conjugate(lbd0_t)*lbd0*
                                       (
                                           c1 + ((dt*mu/(dx**2)) * (c2 - 2*c1 + c3))
                                           - ((dt * anp.conjugate(lbd0_t)/(2*dx))*(c4 - c5)) 
                                           )
                                       ) + np.abs(lbd0_t)**2
    
    return anp.real(cost).astype(anp.float64).reshape(())



def initial_cost_quimb(qc,des):
    u_temp = qc.psi.to_dense()
    u_temp.reshape((-1))
    n_t = len(des)
    des = anp.asarray(des, dtype=u_temp.dtype)
    temp = anp.sum(anp.array([anp.abs(u_temp[i] - des[i])**2 for i in range(n_t)]))
    return temp

def cost_quimb_shots(unitary,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu,shots):

    c1 = embed_circuit(unitary,qc1,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c2 = embed_circuit(unitary,qc2,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c3 = embed_circuit(unitary,qc3,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c4 = embed_circuit(unitary,qc4,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c5 = embed_circuit(unitary,qc5,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c1 = max(-1.0, min(1.0, np.real(c1)))
    c2 = max(-1.0, min(1.0, np.real(c2)))
    c3 = max(-1.0, min(1.0, np.real(c3)))
    c4 = max(-1.0, min(1.0, np.real(c4)))
    c5 = max(-1.0, min(1.0, np.real(c5)))

    lbd0 = unitary.gates[-1].params[0]
    
    p01 = (1 + c1) / 2
    p11 = (1 - c1) / 2
    outcomes = np.random.choice([+1, -1], size=shots, p=[p01, p11])
    c1s = np.mean(outcomes)

    p01 = (1 + c2) / 2
    p11 = (1 - c2) / 2
    outcomes = np.random.choice([+1, -1], size=shots, p=[p01, p11])
    c2s = np.mean(outcomes)

    p01 = (1 + c3) / 2
    p11 = (1 - c3) / 2
    outcomes = np.random.choice([+1, -1], size=shots, p=[p01, p11])
    c3s = np.mean(outcomes)

    p01 = (1 + c4) / 2
    p11 = (1 - c4) / 2
    outcomes = np.random.choice([+1, -1], size=shots, p=[p01, p11])
    c4s = np.mean(outcomes)

    p01 = (1 + c5) / 2
    p11 = (1 - c5) / 2
    outcomes = np.random.choice([+1, -1], size=shots, p=[p01, p11])
    c5s = np.mean(outcomes)

    cost = anp.abs(lbd0)**2 - 2*anp.real(anp.conjugate(lbd0_t)*lbd0*
                                       (
                                           c1s + ((dt*mu/(dx**2)) * (c2s - 2*c1s + c3s))
                                           - ((dt * anp.conjugate(lbd0_t)/(2*dx))*(c4s - c5s)) 
                                           )
                                       ) + anp.abs(lbd0_t)**2
    
    return anp.real(cost).astype(anp.float64).reshape(())


def cost_quimb_shots2(
    curr_params,
    qc1,
    qc2,
    qc3,
    qc4,
    qc5,
    lbd0_t,
    wires,
    dt,
    dx,
    mu,
    shots,
    cost_vals,
    N,
    L,
    params_iters,
    unitary_circuit=None,
):

    unitary_qu = make_optimization_unitary(
        N,
        L,
        curr_params[1:],
        lbd0=curr_params[0],
        unitary_circuit=unitary_circuit,
    )

    c1 = embed_circuit(unitary_qu,qc1,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c2 = embed_circuit(unitary_qu,qc2,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c3 = embed_circuit(unitary_qu,qc3,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c4 = embed_circuit(unitary_qu,qc4,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c5 = embed_circuit(unitary_qu,qc5,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c1 = max(-1.0, min(1.0, np.real(c1)))
    c2 = max(-1.0, min(1.0, np.real(c2)))
    c3 = max(-1.0, min(1.0, np.real(c3)))
    c4 = max(-1.0, min(1.0, np.real(c4)))
    c5 = max(-1.0, min(1.0, np.real(c5)))

    lbd0 = unitary_qu.gates[-1].params[0]
    
    p01 = (1 + c1) / 2
    p11 = (1 - c1) / 2
    outcomes = np.random.choice([+1, -1], size=shots, p=[p01, p11])
    c1s = np.mean(outcomes)

    p01 = (1 + c2) / 2
    p11 = (1 - c2) / 2
    outcomes = np.random.choice([+1, -1], size=shots, p=[p01, p11])
    c2s = np.mean(outcomes)

    p01 = (1 + c3) / 2
    p11 = (1 - c3) / 2
    outcomes = np.random.choice([+1, -1], size=shots, p=[p01, p11])
    c3s = np.mean(outcomes)

    p01 = (1 + c4) / 2
    p11 = (1 - c4) / 2
    outcomes = np.random.choice([+1, -1], size=shots, p=[p01, p11])
    c4s = np.mean(outcomes)

    p01 = (1 + c5) / 2
    p11 = (1 - c5) / 2
    outcomes = np.random.choice([+1, -1], size=shots, p=[p01, p11])
    c5s = np.mean(outcomes)

    cost = anp.abs(lbd0)**2 - 2*anp.real(anp.conjugate(lbd0_t)*lbd0*
                                       (
                                           c1s + ((dt*mu/(dx**2)) * (c2s - 2*c1s + c3s))
                                           - ((dt * anp.conjugate(lbd0_t)/(2*dx))*(c4s - c5s)) 
                                           )
                                       ) + anp.abs(lbd0_t)**2
    
    cost_vals.append(anp.real(cost).astype(anp.float64).reshape(()))
    params_iters.append(curr_params)
    
    return anp.real(cost).astype(anp.float64).reshape(())


def grad_param_shift(
    params,
    qc1,
    qc2,
    qc3,
    qc4,
    qc5,
    lbd0_t,
    wires,
    dt,
    dx,
    mu,
    shots,
    N,
    L,
    unitary_circuit=None,
):
    grad = np.zeros(len(params))
    for i in range(1,len(params)):
        params_p = params.copy()
        params_m = params.copy()
        params_p[i] += HADAMARD_TEST_PARAMETER_SHIFT
        params_m[i] -= HADAMARD_TEST_PARAMETER_SHIFT
        uni_p = make_optimization_unitary(
            N, L, params_p[1:], lbd0=params[0], unitary_circuit=unitary_circuit
        )
        uni_m = make_optimization_unitary(
            N, L, params_m[1:], lbd0=params[0], unitary_circuit=unitary_circuit
        )
        cost_p = cost_quimb_shots(uni_p,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu,shots)
        cost_m = cost_quimb_shots(uni_m,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu,shots)
        grad[i] = hadamard_test_parameter_shift(cost_p, cost_m)
    
    return grad

def single_grad_finite_diff(
    params,
    qc1,
    qc2,
    qc3,
    qc4,
    qc5,
    lbd0_t,
    wires,
    dt,
    dx,
    mu,
    h,
    N,
    L,
    unitary_circuit=None,
):
    uni1 = make_optimization_unitary(
        N, L, params[1:], params[0] + h / 2, unitary_circuit
    )
    uni2 = make_optimization_unitary(
        N, L, params[1:], params[0] - h / 2, unitary_circuit
    )
    diff = (cost_quimb(uni1,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu) - cost_quimb(uni2,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu))/h
    return diff

def adam_update(
    theta: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    t: int,
    lr: float = 0.005,
    beta1: float = 0.9,
    beta2: float = 0.9999,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a single Adam update step.

    Parameters
    ----------
    theta : np.ndarray
        Current parameter vector.
    grad : np.ndarray
        Current gradient estimate.
    m : np.ndarray
        First moment vector (running average of gradients).
    v : np.ndarray
        Second moment vector (running average of grad^2).
    t : int
        Time step (starting from 1).
    lr : float
        Learning rate.
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the second moment estimates.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    theta, m, v : tuple of np.ndarray
        Updated parameters and moment vectors.
    """
    # Update biased first and second moment estimates
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad ** 2)

    # Bias correction
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)

    # Parameter update
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)

    return theta, m, v


def grad_mod_shots(unitary,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu,shots):

    c1 = embed_circuit(unitary,qc1,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c2 = embed_circuit(unitary,qc2,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c3 = embed_circuit(unitary,qc3,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c4 = embed_circuit(unitary,qc4,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c5 = embed_circuit(unitary,qc5,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c1 = max(-1.0, min(1.0, np.real(c1)))
    c2 = max(-1.0, min(1.0, np.real(c2)))
    c3 = max(-1.0, min(1.0, np.real(c3)))
    c4 = max(-1.0, min(1.0, np.real(c4)))
    c5 = max(-1.0, min(1.0, np.real(c5)))

    lbd0 = unitary.gates[-1].params[0]
    
    p01 = (1 + c1) / 2
    p11 = (1 - c1) / 2
    outcomes = np.random.choice([+1, -1], size=shots, p=[p01, p11])
    c1s = np.mean(outcomes)

    p01 = (1 + c2) / 2
    p11 = (1 - c2) / 2
    outcomes = np.random.choice([+1, -1], size=shots, p=[p01, p11])
    c2s = np.mean(outcomes)

    p01 = (1 + c3) / 2
    p11 = (1 - c3) / 2
    outcomes = np.random.choice([+1, -1], size=shots, p=[p01, p11])
    c3s = np.mean(outcomes)

    p01 = (1 + c4) / 2
    p11 = (1 - c4) / 2
    outcomes = np.random.choice([+1, -1], size=shots, p=[p01, p11])
    c4s = np.mean(outcomes)

    p01 = (1 + c5) / 2
    p11 = (1 - c5) / 2
    outcomes = np.random.choice([+1, -1], size=shots, p=[p01, p11])
    c5s = np.mean(outcomes)

    cost = 2*lbd0 - 2*anp.real(anp.conjugate(lbd0_t)*
                                       (
                                           c1s + ((dt*mu/(dx**2)) * (c2s - 2*c1s + c3s))
                                           - ((dt * anp.conjugate(lbd0_t)/(2*dx))*(c4s - c5s)) 
                                           )
                                       )
    
    return anp.real(cost).astype(anp.float64).reshape(())

def build_local_exp_tree(unitary, qc, wires):
    circ = embed_circuit(unitary, qc, wires)

    info = circ.local_expectation(
        qu.pauli('Z'),
        0,
        simplify_sequence="",
        rehearse=True,
    )
    return info["tree"]

def local_expectations_cached(unitary, qc1, qc2, qc3, qc4, qc5, wires, trees):
    """Evaluate the five local expectations used by the VQCFD cost."""
    circuits = (qc1, qc2, qc3, qc4, qc5)
    return np.asarray(
        [
            embed_circuit(unitary, qc, wires).local_expectation(
                qu.pauli("Z"),
                0,
                simplify_sequence="",
                optimize=tree,
            )
            for qc, tree in zip(circuits, trees)
        ]
    )


def cost_from_local_expectations(expectations, lbd0_t, lbd0, dt, dx, mu):
    """Combine cached circuit expectations into the cost for one time step."""
    c1, c2, c3, c4, c5 = expectations
    lbd0_t_conj = np.conjugate(lbd0_t)
    cost = (
        abs(lbd0) ** 2
        - 2
        * np.real(
            lbd0_t_conj
            * lbd0
            * (
                c1
                + (dt * mu / dx**2) * (c2 - 2 * c1 + c3)
                - (dt * lbd0_t_conj / (2 * dx)) * (c4 - c5)
            )
        )
        + abs(lbd0_t) ** 2
    )
    return float(np.real(cost))


def cost_quimb_cached(unitary, qc1, qc2, qc3, qc4, qc5,
                      lbd0_t,lbd0, wires, dt, dx, mu, trees):
    expectations = local_expectations_cached(
        unitary, qc1, qc2, qc3, qc4, qc5, wires, trees
    )
    return cost_from_local_expectations(expectations, lbd0_t, lbd0, dt, dx, mu)


def whole_grads_param_shift(
    params,
    qc1,
    qc2,
    qc3,
    qc4,
    qc5,
    lbd0_t,
    wires,
    dts,
    dx,
    mu,
    N,
    L,
    trees,
    unitary_circuit=None,
):
    """Return gradients for several ``dt`` values using one pair of evaluations.

    The five expectation values are independent of ``dt``. Their shifted
    derivatives are therefore evaluated once per parameter, then combined with
    each requested set of PDE coefficients. The result has shape
    ``(len(dts), len(params))`` and includes the unused norm slot at index zero.
    """
    dts = np.atleast_1d(np.asarray(dts, dtype=float))
    grads = np.zeros((dts.size, len(params)), dtype=float)
    lbd0_t_conj = np.conjugate(lbd0_t)

    for i in range(1, len(params)):
        params_p = params.copy()
        params_m = params.copy()
        params_p[i] += HADAMARD_TEST_PARAMETER_SHIFT
        params_m[i] -= HADAMARD_TEST_PARAMETER_SHIFT
        uni_p = make_optimization_unitary2(
            N, L, params_p[1:], lbd0=params[0], unitary_circuit=unitary_circuit
        )
        uni_m = make_optimization_unitary2(
            N, L, params_m[1:], lbd0=params[0], unitary_circuit=unitary_circuit
        )
        exp_p = local_expectations_cached(
            uni_p, qc1, qc2, qc3, qc4, qc5, wires, trees
        )
        exp_m = local_expectations_cached(
            uni_m, qc1, qc2, qc3, qc4, qc5, wires, trees
        )
        dc1, dc2, dc3, dc4, dc5 = hadamard_test_parameter_shift(exp_p, exp_m)

        for dt_index, dt in enumerate(dts):
            bracket_derivative = (
                dc1
                + (dt * mu / dx**2) * (dc2 - 2 * dc1 + dc3)
                - (dt * lbd0_t_conj / (2 * dx)) * (dc4 - dc5)
            )
            grads[dt_index, i] = float(
                -2 * np.real(lbd0_t_conj * params[0] * bracket_derivative)
            )

    return grads

def whole_grad_param_shift(
    params,
    qc1,
    qc2,
    qc3,
    qc4,
    qc5,
    lbd0_t,
    wires,
    dt,
    dx,
    mu,
    N,
    L,
    trees,
    unitary_circuit=None,
):
    return whole_grads_param_shift(
        params,
        qc1,
        qc2,
        qc3,
        qc4,
        qc5,
        lbd0_t,
        wires,
        [dt],
        dx,
        mu,
        N,
        L,
        trees,
        unitary_circuit,
    )[0]


def grad_mod(unitary,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu):

    c1 = embed_circuit(unitary,qc1,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c2 = embed_circuit(unitary,qc2,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c3 = embed_circuit(unitary,qc3,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c4 = embed_circuit(unitary,qc4,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c5 = embed_circuit(unitary,qc5,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")

    lbd0 = unitary.gates[-1].params[0]


    cost = 2*lbd0 - 2*anp.real(anp.conjugate(lbd0_t)*
                                       (
                                           c1 + ((dt*mu/(dx**2)) * (c2 - 2*c1 + c3))
                                           - ((dt * anp.conjugate(lbd0_t)/(2*dx))*(c4 - c5)) 
                                           )
                                       )
    
    return anp.real(cost).astype(anp.float64).reshape(())
