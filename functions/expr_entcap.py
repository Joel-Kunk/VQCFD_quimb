import numpy as np
from qiskit import QuantumCircuit
from scipy.stats import entropy
import functions.functions_quimb as fqu
import functions.functions_qiskit as fqi
from qiskit.circuit import Parameter
import quimb.tensor as qtn
import quimb as qu





def count_total_parameters(qc: QuantumCircuit) -> int:
    """
    Counts the total number of parameters (including numeric constants)
    across all operations in the circuit.
    """
    total = 0
    for inst in qc.data:
        op = inst.operation      # the Gate / Instruction object
        total += len(op.params)  # list of parameters (float or Parameter)
    return total

def select_numbers_np(N, position, value):
    """ 
    select positions in wich the bit at position takes value,from numbers in 0 to N (N should = 2**n)
    """
    xs = np.arange(N)
    bits = (xs >> position) & 1
    return xs[bits == value]

def special_distance(u,v,pos):
    val = 0
    dim = len(u)
    inds1 = select_numbers_np(dim,pos,0)
    inds2 = select_numbers_np(dim,pos,1)
    for i in range(int(dim/2)):
        for j in range(i):
            val += np.abs(u[inds1[i]]*v[inds2[j]]-u[inds1[j]]*v[inds2[i]])**2

    return val

def expressibility(circ,tries,n_bins):
    num_params = count_total_parameters(circ)
    fidelities = []
    for i in range(tries):
        params_t1 = np.random.random(num_params) * 2 * np.pi
        params_t2 = np.random.random(num_params) * 2 * np.pi
        uni1 = fqu.qisikit_to_quimb(circ,0,params_t1)
        uni2 = fqu.qisikit_to_quimb(circ,0,params_t2)
        fidelities.append(np.abs(np.vdot(uni1.to_dense(),uni2.to_dense()))**2)

    p_pqc = np.histogram(fidelities, bins = n_bins)
    bins = p_pqc[1]
    hist_pqc = p_pqc[0]/p_pqc[0].sum()
    p_kl = np.zeros(n_bins)
    N_t = 2**circ.num_qubits
    for i in range(n_bins):
        val_t = (bins[i]+bins[i+1])/2
        p_kl[i] = (N_t-1)*(1-val_t)**(N_t-2)
    hist_kl = p_kl/p_kl.sum()

    KL = entropy(hist_pqc,hist_kl)

    return KL

def entangling_capability(circ,tries):
    num_params = count_total_parameters(circ)
    N_t = circ.num_qubits
    ent_cap = 0
    for i in range(tries):
        params_t = np.random.random(num_params)*2*np.pi
        uni = fqu.qisikit_to_quimb(circ,0,params_t)
        vec_t = uni.to_dense()
        for j in range(N_t):
            ent_cap += special_distance(vec_t,vec_t,j)
    
    ent_cap = ent_cap*4/(N_t*tries)
    return ent_cap

def expr_and_ent_cap(circ,tries,n_bins):
    num_params = count_total_parameters(circ)
    fidelities = []
    N_t = circ.num_qubits
    ent_cap = 0.0

    for i in range(tries):
        params_t1 = np.random.random(num_params) * 2 * np.pi
        params_t2 = np.random.random(num_params) * 2 * np.pi
        uni1 = fqu.qisikit_to_quimb(circ,0,params_t1)
        uni2 = fqu.qisikit_to_quimb(circ,0,params_t2)
        vec_t1 = uni1.to_dense()
        vec_t2 = uni2.to_dense()
        fidelities.append(np.abs(np.vdot(vec_t1,vec_t2))**2)

        for j in range(N_t):
            ent_cap += special_distance(vec_t1,vec_t1,j)

        if i % 1000 == 0:
            print(f"Progress: {i}/{tries} ")

    p_pqc = np.histogram(fidelities, bins = n_bins)
    bins = p_pqc[1]
    hist_pqc = p_pqc[0]/p_pqc[0].sum()
    p_kl = np.zeros(n_bins)
    N_t2 = 2**circ.num_qubits
    for i in range(n_bins):
        val_t = (bins[i]+bins[i+1])/2
        p_kl[i] = (N_t2-1)*(1-val_t)**(N_t2-2)
    hist_kl = p_kl/p_kl.sum()

    kl = entropy(hist_pqc,hist_kl)
    result_t = ent_cap*4/(N_t*tries)

    return kl,result_t[0]

def initial_fit(circ,PSI_init,MOD_init):
    N_params = count_total_parameters(circ)
    params = (np.random.random(N_params) + (np.pi-0.5)).tolist()

    qc_t = fqu.qiskit_to_quimb_uni(circ,params)
    initial_opt = qtn.TNOptimizer(qc_t,fqu.initial_cost_quimb,loss_constants=dict(des = PSI_init),autodiff_backend="autograd")

    opt_initial = initial_opt.optimize(1000)
    initial_params = np.concatenate(([MOD_init],np.concatenate([np.atleast_1d(v) for v in opt_initial.get_params().values()])))
    best_value = initial_opt.loss_best
    iterations = len(initial_opt.losses)
    return best_value , iterations, initial_params



def test_functions(N_total):
    Xs = np.linspace(0, 1, N_total)

    funcs = []

    MOD_init = np.sqrt(np.sum(np.sin(2*np.pi*Xs)**2))
    f1 = np.sin(2*np.pi*Xs)/MOD_init
    f1 = np.append(f1,MOD_init)
    funcs.append(f1)

    MOD_init = np.sqrt(np.sum(np.cos(2*np.pi*Xs)**2))
    f1 = np.cos(2*np.pi*Xs)/MOD_init
    f1 = np.append(f1,MOD_init)
    funcs.append(f1)

    MOD_init = np.sqrt(np.sum(np.cos(4*np.pi*Xs)**2))
    f1 = np.cos(4*np.pi*Xs)/MOD_init
    f1 = np.append(f1,MOD_init)
    funcs.append(f1)

    MOD_init = np.sqrt(np.sum(np.sin(4*np.pi*Xs)**2))
    f1 = np.sin(4*np.pi*Xs)/MOD_init
    f1 = np.append(f1,MOD_init)
    funcs.append(f1)

    f1 = (Xs-0.5)**2
    MOD_init = np.sqrt(np.sum(f1**2))
    f1 = f1/MOD_init
    f1 = np.append(f1,MOD_init)
    funcs.append(f1)

    f1 = Xs*2
    MOD_init = np.sqrt(np.sum(f1**2))
    f1 = f1/MOD_init
    f1 = np.append(f1,MOD_init)
    funcs.append(f1)

    f1 = (Xs-0.5)**3
    MOD_init = np.sqrt(np.sum(f1**2))
    f1 = f1/MOD_init
    f1 = np.append(f1,MOD_init)
    funcs.append(f1)

    return funcs

def unitary_pure(qubits, layers):
    qc = QuantumCircuit(qubits)
    #for i in range(qubits):
    #    qc.h(i)
    j = 0
    for k in range(layers):
        for i in range(qubits):
            theta = Parameter('theta'+str(j))
            j += 1
            qc.ry(theta, i)
        for i in range(qubits-1):
            qc.cx(i, i+1)
    for i in range(qubits):
        theta = Parameter('theta'+str(j))
        j += 1
        qc.ry(theta, i)
    return qc

def gradient_of_cost(params1,params2,mod,idx,N,L,dt,dx,mu,N_params,num_gates_in_U):
    params1_1 = params1.copy()
    params1_1[idx] += np.pi/2
    params1_2 = params1.copy()
    params1_2[idx] -= np.pi/2

    circuits = fqi.get_circuits(params1_1, N, L)
    quimb_unitary1 = fqu.qiskit_to_quimb_unitary(fqi.unitary(N,L),params1_1,mod)
    quimb_unitary2 = fqu.qiskit_to_quimb_unitary(fqi.unitary(N,L),params1_2,mod)

    qc1 = fqu.qisikit_to_quimb_inverse(circuits[0][0].assign_parameters(params1),N_params,num_gates_in_U,params2)
    qc2 = fqu.qisikit_to_quimb_inverse(circuits[1][0].assign_parameters(params1),N_params,num_gates_in_U,params2)
    qc3 = fqu.qisikit_to_quimb_inverse(circuits[2][0].assign_parameters(params1),N_params,num_gates_in_U,params2)
    qc4 = fqu.qisikit_to_quimb_inverse(circuits[3][0].assign_parameters(params1),N_params,num_gates_in_U,params2)
    qc5 = fqu.qisikit_to_quimb_inverse(circuits[4][0].assign_parameters(params1),N_params,num_gates_in_U,params2)
    wires = tuple(range(int(N) + 1))

    c1 = fqu.embed_circuit(quimb_unitary1,qc1,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c2 = fqu.embed_circuit(quimb_unitary1,qc2,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c3 = fqu.embed_circuit(quimb_unitary1,qc3,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c4 = fqu.embed_circuit(quimb_unitary1,qc4,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c5 = fqu.embed_circuit(quimb_unitary1,qc5,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")

    c12 = fqu.embed_circuit(quimb_unitary2,qc1,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c22 = fqu.embed_circuit(quimb_unitary2,qc2,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c32 = fqu.embed_circuit(quimb_unitary2,qc3,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c42 = fqu.embed_circuit(quimb_unitary2,qc4,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")
    c52 = fqu.embed_circuit(quimb_unitary2,qc5,wires).local_expectation(qu.pauli('Z'), 0,simplify_sequence="RC")

  

    cost = - 2*np.real(np.conjugate(mod)*mod*
                                       (
                                           (c1 - c12)/2 + ((dt*mu/(dx**2)) * ((c2-c22)/2 - 2*(c1 - c12)/2 + (c3-c32)/2))
                                           - ((dt * np.conjugate(mod)/(2*dx))*((c4-c42)/2 - (c5-c52)/2)) 
                                           )
                                       )
    
    return np.real(cost).astype(np.float64).reshape(())