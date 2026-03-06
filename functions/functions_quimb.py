import quimb as qu
import quimb.tensor as qtn
import autograd.numpy as anp
import numpy as np
import functions.functions_qiskit as fqi

def qisikit_to_quimb(qc,num_params,prev_params):
    qcn = qtn.Circuit(qc.num_qubits)
    par_num = 0
    par_counter = num_params
    #par_pos = []
    for inst in qc.data:
        gate = inst.name
        qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
        qubits_for_quimb = tuple(qc.num_qubits - 1 - qubits[i] for i in range(len(qubits)))
        params = tuple(1. for p in inst.params)
        
        #print(gate)
        if gate == 'barrier' or gate == 'measure':
            continue
        elif len(params) != 0:
            if par_num < num_params:
                qcn.apply_gate(gate.upper(),qubits=qubits_for_quimb,params=params,parametrize= True)
                par_num = par_num + 1
            else:
                par_counter = par_counter - 1
                qcn.apply_gate(gate.upper(),qubits=qubits_for_quimb,params=(prev_params[par_counter],))
            #par_pos.append(qubits_for_quimb)
        else:
            qcn.apply_gate(gate.upper(),qubits=qubits_for_quimb)
        
    return qcn#,par_pos

#convert qiskit circuits to quimb circuits, so we can use TNs

def qiskit_to_quimb_uni(qc,params):
    qcn = qtn.Circuit(qc.num_qubits)
    num_param = -1
    for inst in qc.data:
        gate = inst.name
        qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
        qubits_for_quimb = tuple(qc.num_qubits - 1 - qubits[i] for i in range(len(qubits)))
        paras = tuple(1. for p in inst.params)
        if gate == 'barrier' or gate == 'measure':
            continue
        elif len(paras) != 0:
            num_param = num_param + 1
            qcn.apply_gate(gate.upper(),qubits=qubits_for_quimb,params= (params[num_param],),parametrize= True)
            

        else:
            qcn.apply_gate(gate.upper(),qubits=qubits_for_quimb)
        
    return qcn

def qiskit_to_quimb_unitary(qc,params_insert,lbd0):
    qcn = qtn.Circuit(qc.num_qubits)
    num_param = len(params_insert)
    for inst in qc.inverse().data:
        gate = inst.name
        qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
        qubits_for_quimb = tuple(qc.num_qubits - 1 - qubits[i] for i in range(len(qubits)))
        params = tuple(1. for p in inst.params)
        
        #print(gate)
        if gate == 'barrier' or gate == 'measure':
            continue
        elif len(params) != 0:
            num_param = num_param - 1
            qcn.apply_gate(gate.upper(),qubits=qubits_for_quimb,params=(-params_insert[num_param],),parametrize= True)
        else:
            qcn.apply_gate(gate.upper(),qubits=qubits_for_quimb)

    qcn.apply_gate('H',0)
    qcn.apply_gate('RY',params = (lbd0,), qubits = (int(qc.num_qubits),),parametrize=True)
        
    return qcn

# create inverse circuits of the qiskit circuits, but without the end of the actual unitary encoding the VF, so we can just
# append that unitary to all the circuits and optimize as one.

def qisikit_to_quimb_inverse(qc,num_params,num_U_gates,prev_params):
    qcn = qtn.Circuit(qc.num_qubits)
    gate_num = 0
    gate_lim = len(qc.data) - num_U_gates
    par_counter = -num_params

    for inst in qc.inverse().data:
        gate = inst.name
        qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
        qubits_for_quimb = tuple(qc.num_qubits - 1 - qubits[i] for i in range(len(qubits)))
        params = tuple(1. for p in inst.params)
        
        if gate == 'barrier' or gate == 'measure' or gate_num >= gate_lim: 
            continue
        elif len(params) != 0:
            qcn.apply_gate(gate.upper(),qubits=qubits_for_quimb,params=(prev_params[par_counter],))
            gate_num = gate_num + 1
            par_counter = par_counter + 1
        elif gate == 'mcx':
            qcn.apply_gate('X', qubits = (qubits_for_quimb[-1],), controls = qubits_for_quimb[:-1])
            gate_num = gate_num + 1
        else:
            qcn.apply_gate(gate.upper(),qubits=qubits_for_quimb)
            gate_num = gate_num + 1        
    return qcn

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


def cost_quimb_shots2(curr_params,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu,shots,cost_vals,N,L,params_iters):

    unitary_qu = qiskit_to_quimb_unitary(fqi.unitary(N,L),curr_params[1:],lbd0=curr_params[0])

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


# using parameter shift rule to get gradients with shot noise. All derivatives are fine, but need to be devided by a factor of 1.414213562692549
# except for the last one, which is correct as it is.

def grad_param_shift(params,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu,shots,N,L):
    grad = np.zeros(len(params))
    for i in range(1,len(params)):
        params_p = params.copy()
        params_m = params.copy()
        params_p[i] += np.pi/2
        params_m[i] -= np.pi/2
        uni_p = qiskit_to_quimb_unitary(fqi.unitary(N,L),params_p[1:],lbd0=params[0])
        uni_m = qiskit_to_quimb_unitary(fqi.unitary(N,L),params_m[1:],lbd0=params[0])
        cost_p = cost_quimb_shots(uni_p,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu,shots)
        cost_m = cost_quimb_shots(uni_m,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu,shots)
        grad[i] = (cost_p - cost_m)/2

    grad[1:-1] /= np.sqrt(2)
    
    return grad

def single_grad_finite_diff(params,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu,h,N,L):
    uni1 = qiskit_to_quimb_unitary(fqi.unitary(N,L),params[1:],params[0]+h/2)
    uni2 = qiskit_to_quimb_unitary(fqi.unitary(N,L),params[1:],params[0]-h/2)
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

def whole_grad_param_shift(params,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu,N,L):
    grad = np.zeros(len(params))
    for i in range(1,len(params)):
        params_p = params.copy()
        params_m = params.copy()
        params_p[i] += np.pi/2
        params_m[i] -= np.pi/2
        uni_p = qiskit_to_quimb_unitary(fqi.unitary(N,L),params_p[1:],lbd0=params[0])
        uni_m = qiskit_to_quimb_unitary(fqi.unitary(N,L),params_m[1:],lbd0=params[0])
        cost_p = cost_quimb(uni_p,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu)
        cost_m = cost_quimb(uni_m,qc1,qc2,qc3,qc4,qc5,lbd0_t,wires,dt,dx,mu)
        grad[i] = (cost_p - cost_m)/2
    
    grad[1:-1] /= np.sqrt(2)
    
    return grad


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