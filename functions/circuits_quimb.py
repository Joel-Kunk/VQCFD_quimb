import numpy as np
import quimb.tensor as qtn


def num_unitary_parameters(qubits: int, layers: int) -> int:
    return qubits * layers + qubits


def _unitary_gate_specs(control: int, targets_desc: list[int], layers: int, params: np.ndarray | list[float]):
    specs = []
    p = np.asarray(params, dtype=float)
    idx = 0
    for _ in range(layers):
        for t in targets_desc:
            specs.append(("CRY", (control, t), float(p[idx])))
            idx += 1
        for i in range(len(targets_desc) - 1):
            specs.append(("CX", (targets_desc[i], targets_desc[i + 1]), None))
    for t in targets_desc:
        specs.append(("CRY", (control, t), float(p[idx])))
        idx += 1
    return specs


def _apply_specs(circ: qtn.Circuit, specs, inverse: bool = False, parametrize: bool = False) -> None:
    seq = reversed(specs) if inverse else specs
    for gate, qubits, param in seq:
        if gate == "CRY":
            theta = -param if inverse else param
            circ.apply_gate("CRY", qubits=qubits, params=(theta,), parametrize=parametrize)
        else:
            circ.apply_gate(gate, qubits=qubits)


def apply_unitary_forward(circ: qtn.Circuit, qubits: int, layers: int, params: np.ndarray | list[float], parametrize: bool = False) -> None:
    targets_desc = list(range(qubits, 0, -1))
    specs = _unitary_gate_specs(control=0, targets_desc=targets_desc, layers=layers, params=params)
    _apply_specs(circ, specs, inverse=False, parametrize=parametrize)


def apply_unitary_inverse(circ: qtn.Circuit, qubits: int, layers: int, params: np.ndarray | list[float], parametrize: bool = False) -> None:
    targets_desc = list(range(qubits, 0, -1))
    specs = _unitary_gate_specs(control=0, targets_desc=targets_desc, layers=layers, params=params)
    _apply_specs(circ, specs, inverse=True, parametrize=parametrize)


def _apply_unitary_on_register(
    circ: qtn.Circuit,
    control: int,
    targets_desc: list[int],
    layers: int,
    params: np.ndarray | list[float],
    inverse: bool,
    parametrize: bool = False,
) -> None:
    specs = _unitary_gate_specs(control=control, targets_desc=targets_desc, layers=layers, params=params)
    _apply_specs(circ, specs, inverse=inverse, parametrize=parametrize)


def make_state_circuit(qubits: int, layers: int, params: np.ndarray | list[float], parametrize: bool = True) -> qtn.Circuit:
    qc = qtn.Circuit(qubits + 1)
    qc.apply_gate("X", qubits=(0,))
    apply_unitary_forward(qc, qubits, layers, params, parametrize=parametrize)
    qc.apply_gate("X", qubits=(0,))
    return qc


def make_optimization_unitary(qubits: int, layers: int, params: np.ndarray | list[float], lbd0: float) -> qtn.Circuit:
    qc = qtn.Circuit(qubits + 1)
    apply_unitary_inverse(qc, qubits, layers, params, parametrize=True)
    qc.apply_gate("H", qubits=(0,))
    qc.apply_gate("RY", qubits=(qubits + 1,), params=(float(lbd0),), parametrize=True)
    return qc

def make_optimization_unitary2(qubits: int, layers: int, params: np.ndarray | list[float], lbd0: float) -> qtn.Circuit:
    qc = qtn.Circuit(qubits + 1)
    apply_unitary_inverse(qc, qubits, layers, params)
    qc.apply_gate("H", qubits=(0,))
    return qc


def _adder_gate_specs(n: int):
    if n <= 1:
        raise ValueError("Adder construction requires n >= 2.")
    if n == 2:
        return [("CX", (0, 2)), ("CCX", (0, 2, 1))]

    qr = list(range(n, 0, -1))
    ar = list(range(2 * n - 2, n, -1))
    mr = 0

    gates = [("CX", (mr, qr[0]))]
    gates.append(("CCX", (mr, qr[0], qr[1])))
    gates.append(("CCX", (mr, qr[0], ar[0])))

    if len(ar) > 1:
        gates.append(("CCX", (ar[0], qr[1], ar[1])))

    if len(qr) > 4:
        for i in range(len(ar) - 2):
            gates.append(("CX", (ar[i + 1], qr[i + 2])))
            gates.append(("CCX", (ar[i + 1], qr[i + 2], ar[i + 2])))

    if len(ar) == 1:
        gates.append(("CCX", (ar[0], qr[-2], qr[-1])))

    if len(ar) >= 2:
        gates.append(("CX", (ar[-1], qr[-2])))
        gates.append(("CCX", (ar[-1], qr[-2], qr[-1])))
        for i in range(len(ar) - 1, 0, -1):
            gates.append(("CCX", (ar[i - 1], qr[i], ar[i])))

    gates.append(("CCX", (mr, qr[0], ar[0])))
    return gates


def _apply_adder(circ: qtn.Circuit, n: int, dagger: bool) -> None:
    gates = _adder_gate_specs(n)
    seq = reversed(gates) if dagger else gates
    for gate, qubits in seq:
        circ.apply_gate(gate, qubits=qubits)


def _apply_diagonal_inverse(circ: qtn.Circuit, n: int, layers: int, prev_params: np.ndarray | list[float]) -> None:
    anc = 0
    right_data = list(range(n, 0, -1))
    left_data = list(range(3 * n - 2, 2 * n - 2, -1))

    # Inverse of D = U_prev(left_data, ancilla) followed by diagonal CCX chain.
    _apply_unitary_on_register(
        circ,
        control=anc,
        targets_desc=left_data,
        layers=layers,
        params=prev_params,
        inverse=False,
        parametrize=False,
    )

    for i in range(n):
        circ.apply_gate("CCX", qubits=(anc, right_data[i], left_data[i]))


def make_inverse_reference_circuits(prev_params: np.ndarray | list[float], qubits: int, layers: int) -> list[qtn.Circuit]:
    prev = np.asarray(prev_params, dtype=float)

    # c1 = H + U_prev on the (ancilla + right-data) register.
    c1 = qtn.Circuit(qubits + 1)
    c1.apply_gate("H", qubits=(0,))
    apply_unitary_forward(c1, qubits, layers, prev, parametrize=False)

    # c2 / c3 use enlarged register containing adder ancillas.
    total_ua = 2 * qubits - 1
    c2 = qtn.Circuit(total_ua)
    c2.apply_gate("H", qubits=(0,))
    apply_unitary_forward(c2, qubits, layers, prev, parametrize=False)
    _apply_adder(c2, qubits, dagger=True)

    c3 = qtn.Circuit(total_ua)
    c3.apply_gate("H", qubits=(0,))
    apply_unitary_forward(c3, qubits, layers, prev, parametrize=False)
    _apply_adder(c3, qubits, dagger=False)

    # c4 / c5 additionally include the inverse-diagonal block.
    total_uda = 3 * qubits - 1
    c4 = qtn.Circuit(total_uda)
    c4.apply_gate("H", qubits=(0,))
    apply_unitary_forward(c4, qubits, layers, prev, parametrize=False)
    _apply_adder(c4, qubits, dagger=False)
    _apply_diagonal_inverse(c4, qubits, layers, prev)

    c5 = qtn.Circuit(total_uda)
    c5.apply_gate("H", qubits=(0,))
    apply_unitary_forward(c5, qubits, layers, prev, parametrize=False)
    _apply_adder(c5, qubits, dagger=True)
    _apply_diagonal_inverse(c5, qubits, layers, prev)

    return [c1, c2, c3, c4, c5]


def unitary_gate_count(qubits: int, layers: int) -> int:
    return layers * (2 * qubits - 1) + qubits


def make_pure_unitary_circuit(qubits: int, layers: int, params: np.ndarray | list[float], parametrize: bool = True) -> qtn.Circuit:
    qc = qtn.Circuit(qubits)
    p = np.asarray(params, dtype=float)
    idx = 0
    targets_desc = list(range(qubits - 1, -1, -1))
    for _ in range(layers):
        for t in targets_desc:
            qc.apply_gate("RY", qubits=(t,), params=(float(p[idx]),), parametrize=parametrize)
            idx += 1
        for i in range(len(targets_desc) - 1):
            qc.apply_gate("CX", qubits=(targets_desc[i], targets_desc[i + 1]))
    for t in targets_desc:
        qc.apply_gate("RY", qubits=(t,), params=(float(p[idx]),), parametrize=parametrize)
        idx += 1
    return qc
