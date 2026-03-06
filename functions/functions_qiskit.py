import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit import Parameter

# unitary action on the circuit, to initilize MPS in circuit (later this is the part that will probably get mroe complex)

def unitary(qubits, layers):
    qc = QuantumCircuit(qubits+1)
    #for i in range(qubits):
    #    qc.h(i)
    j = 0
    for k in range(layers):
        for i in range(qubits):
            theta = Parameter('theta'+str(j))
            j += 1
            qc.cry(theta, qubits, i)
        for i in range(qubits-1):
            qc.cx(i, i+1)
    for i in range(qubits):
        theta = Parameter('theta'+str(j))
        j += 1
        qc.cry(theta, qubits, i)
    return qc

# def unitary(qubits, layers):
#     qc = QuantumCircuit(qubits+1)
#     #for i in range(qubits):
#     #    qc.h(i)
#     j = 0
#     for k in range(layers):
#         for i in range(qubits):
#             theta = Parameter('theta'+str(j))
#             j += 1
#             qc.cry(theta, qubits, i)
#             theta = Parameter('theta'+str(j))
#             j += 1
#             qc.crz(theta, qubits, i)
#         for i in range(qubits-1):
#             qc.ccx(qubits,i, i+1)
#     for i in range(qubits):
#         theta = Parameter('theta'+str(j))
#         j += 1
#         qc.cry(theta, qubits, i)
#         theta = Parameter('theta'+str(j))
#         j += 1
#         qc.crz(theta, qubits, i)
#     return qc


# the unitary how we want it to be inserted in our circuit

def circuit(qubits, layers):
    qc = QuantumCircuit(qubits+1)
    qc.x(qubits)
    # qc.barrier()
    U = unitary(qubits, layers)
    qc = qc.compose(U)
    # qc.barrier()
    qc.x(qubits)
    return qc


# costfunction to initilize the parameters, so the circuit represents the desired start state (here the sin function)

def cost_initial(params, circuit, desired):
    circuit = circuit.assign_parameters(params)
    u_pred_initial = Statevector(circuit).data
    n = len(desired)
    temp = np.sum([np.abs(u_pred_initial[i] - desired[i])**2 for i in range(n)])
    return temp


# define circuits we need to evaluate cost function later. plus a function generating all the desired circuits
# define the cost functions (one within qiskit, one for the quimb converted circuits)

def U_Ut(prev_params, qubits, layers):
    qc = QuantumCircuit(qubits+1, 1)
    qc.h(-1)
    # qc.barrier()
    U_curr = unitary(qubits, layers)#.assign_parameters(curr_params)
    qc = qc.compose(U_curr)
    # qc.barrier()
    U_prev = unitary(qubits, layers).assign_parameters(prev_params).inverse()
    qc = qc.compose(U_prev)
    # qc.barrier()
    qc.h(-1)
    return qc

def adder(n):
    if n == 1:
        qc = QuantumCircuit(2)
        qc.cx(-1, 0)
    if n == 2:
        qc = QuantumCircuit(3)
        qc.cx(-1, 0)
        qc.ccx(-1, 0, 1)
    if n > 2:
        n_qubits = n-2+n+1
        qc = QuantumCircuit(n_qubits)
        qubits = np.arange(n_qubits)
        ar = qubits[:(n-2)]
        qr = qubits[(n-2):(n-2+n)]
        mr = qubits[-1:]
        qc.cx(mr[0], qr[0])
        qc.ccx(mr[0], qr[0], qr[1])
        #pillar left
        qc.ccx(mr[0], qr[0], ar[0])
        if len(ar)>1:
            qc.ccx(ar[0], qr[1], ar[1])
        if len(qr)>4:
            for i in range(len(ar)-2):
                qc.cx(ar[i+1], qr[i+2])
                qc.ccx(ar[i+1], qr[i+2], ar[i+2])
        if len(ar) == 1:
            qc.ccx(ar[0], qr[-2], qr[-1])
        if len(ar) >= 2:
            qc.cx(ar[-1], qr[-2])
            qc.ccx(ar[-1], qr[-2], qr[-1])
            for i in range(len(ar)-1, 0, -1):
                qc.ccx(ar[i-1], qr[i], ar[i])
        ## pillar right
        qc.ccx(mr[0], qr[0], ar[0])
    return qc

# def adder(n):
#     qc = QuantumCircuit(n+1)
#     qc.cx(-1, 0)
#     for i in range(n-1):
#         control = list(range(i + 1))
#         qc.mcx(control_qubits=control, target_qubit=[i+1], ctrl_state='1'*len(control))
#     return qc

def U_A_Ut(prev_params, qubits, layers, A_dagger=False):
    n = qubits
    t = n+1+n-2
    qc = QuantumCircuit(n+1+n-2)
    qc.h(-1)
    # qc.barrier()
    # U
    U_curr = unitary(qubits, layers)#.assign_parameters(curr_params)
    qubits_assign = np.arange(t)[-(n+1):]
    qc = qc.compose(U_curr, qubits_assign)
    # qc.barrier()
    # Adder
    if A_dagger:
        A = adder(qubits).inverse()
    else:
        A = adder(qubits)
    qc = qc.compose(A)
    # qc.barrier()
    # U_tilda
    U_prev = unitary(qubits, layers).assign_parameters(prev_params).inverse()
    qc = qc.compose(U_prev, qubits_assign)
    # qc.barrier()
    qc.h(-1)
    return qc

# def U_A_Ut(prev_params, qubits, layers, A_dagger=False):
#     n = qubits
#     t = n+1
#     qc = QuantumCircuit(t)
#     qc.h(-1)
#     # qc.barrier()
#     # U
#     U_curr = unitary(qubits, layers)#.assign_parameters(curr_params)
#     qc = qc.compose(U_curr)
#     # qc.barrier()
#     # Adder
#     if A_dagger:
#         A = adder(qubits).inverse()
#     else:
#         A = adder(qubits)
#     qc = qc.compose(A)
#     # qc.barrier()
#     # U_tilda
#     U_prev = unitary(qubits, layers).assign_parameters(prev_params).inverse()
#     qc = qc.compose(U_prev)
#     # qc.barrier()
#     qc.h(-1)
#     return qc

def U_Dt_A_Ut(prev_params, qubits, layers, A_dagger=False):
    n = qubits
    total_qubits = 1 + n + (n-2) + n
    t = np.arange(total_qubits)
    qc = QuantumCircuit(total_qubits)
    qc.h(-1)
    # qc.barrier()
    # U
    U_curr = unitary(qubits, layers)#.assign_parameters(curr_params)
    U_assign = t[-(n+1):]
    qc = qc.compose(U_curr, U_assign)
    # qc.barrier()
    # Diagonal
    for i in range(n):
        #qc.ccx(-1, i+n+1, i)
        qc.ccx(3*n-2,i+2*n-2, i)
    U_prev = unitary(qubits, layers).assign_parameters(prev_params).inverse()
    D_assign = np.concat((t[:n], t[-1:]))
    qc = qc.compose(U_prev, D_assign)
    # qc.barrier()
    # Adder
    if A_dagger:
        A = adder(qubits).inverse()
    else:
        A = adder(qubits)
    A_assign = t[-(n+1+n-2):]
    qc = qc.compose(A, A_assign)
    # qc.barrier()
    # Ut
    U_prev = unitary(qubits, layers).assign_parameters(prev_params).inverse()
    qc = qc.compose(U_prev, U_assign)
    # qc.barrier()
    qc.h(-1)
    return qc

# def U_Dt_A_Ut(prev_params, qubits, layers, A_dagger=False):
#     n = qubits
#     total_qubits = 1 + n + n
#     t = np.arange(total_qubits)
#     qc = QuantumCircuit(total_qubits)
#     qc.h(-1)
#     # qc.barrier()
#     # U
#     U_curr = unitary(qubits, layers)#.assign_parameters(curr_params)
#     U_assign = t[-(n+1):]
#     qc = qc.compose(U_curr, U_assign)
#     # qc.barrier()
#     # Diagonal
#     for i in range(n):
#         #qc.ccx(-1, i+n+1, i)
#         qc.ccx(2*n,i+n, i)
#     U_prev = unitary(qubits, layers).assign_parameters(prev_params).inverse()
#     D_assign = np.concat((t[:n], t[-1:]))
#     qc = qc.compose(U_prev, D_assign)
#     # qc.barrier()
#     # Adder
#     if A_dagger:
#         A = adder(qubits).inverse()
#     else:
#         A = adder(qubits)
#     A_assign = U_assign
#     qc = qc.compose(A, A_assign)
#     # qc.barrier()
#     # Ut
#     U_prev = unitary(qubits, layers).assign_parameters(prev_params).inverse()
#     qc = qc.compose(U_prev, U_assign)
#     # qc.barrier()
#     qc.h(-1)
#     return qc


# Create a function which returns the circuits, hamiltonians
def get_circuits(prev_c_params, qubits, layers):
    circuits = []
    c1 = U_Ut(prev_c_params, qubits, layers)
    o1 = 'Z' + 'I'*(c1.num_qubits-1)
    o1 = SparsePauliOp.from_list([(o1, 1)])
    circuits.append([c1, o1])
    c2 = U_A_Ut(prev_c_params, qubits, layers, A_dagger=False)
    o2 = 'Z' + 'I'*(c2.num_qubits-1)
    o2 = SparsePauliOp.from_list([(o2, 1)])
    circuits.append([c2, o2])
    c3 = U_A_Ut(prev_c_params, qubits, layers, A_dagger=True)
    o3 = 'Z' + 'I'*(c3.num_qubits-1)
    o3 = SparsePauliOp.from_list([(o3, 1)])
    circuits.append([c3, o3])
    c4 = U_Dt_A_Ut(prev_c_params, qubits, layers, A_dagger=True)
    o4 = 'Z' + 'I'*(c4.num_qubits-1)
    o4 = SparsePauliOp.from_list([(o4, 1)])
    circuits.append([c4, o4])
    c5 = U_Dt_A_Ut(prev_c_params, qubits, layers, A_dagger=False)
    o5 = 'Z' + 'I'*(c5.num_qubits-1)
    o5 = SparsePauliOp.from_list([(o5, 1)])
    circuits.append([c5, o5])
    return circuits

# cost function directly from Qiskit circuits

def cost(var_params, prev_params,n,l,dt,mu,dx,estimator):
    lbd0 = var_params[0]
    curr_params = var_params[1:]

    lbd0_t = prev_params[0]

    circuits = get_circuits(prev_params[1:], n, l)

    pubs = [(circuit, hamiltonian, [curr_params]) for circuit, hamiltonian in circuits]
    estimator_result = estimator.run(pubs).result()
    c = []
    for i in range(len(circuits)):
        t = estimator_result[i].data.evs[0]
        c.append(t)
    cost = np.abs(lbd0)**2 - 2*np.real(np.conjugate(lbd0_t)*lbd0*
                                       (
                                           c[0] + ((dt*mu/(dx**2)) * (c[1] - 2*c[0] + c[2]))
                                           - ((dt * np.conjugate(lbd0_t)/(2*dx))*(c[3] - c[4])) 
                                           )
                                       ) + np.abs(lbd0_t)**2
    #list_1.append([var_params, cost])
    return cost

