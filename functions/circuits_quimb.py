import numpy as np
import quimb.tensor as qtn


DEFAULT_UNITARY_CIRCUIT = "default"
UNITARY_CIRCUITS = (
    DEFAULT_UNITARY_CIRCUIT,
    "uni2",
    "brickwork_ring_ry",
    "ring_ry_rz",
    "ring_trainable_crx",
    "multiscale_tree",
)


def resolve_unitary_circuit(unitary_circuit: str | None = None) -> str:
    if unitary_circuit is None:
        return DEFAULT_UNITARY_CIRCUIT

    name = str(unitary_circuit).strip().lower()
    aliases = {
        "": DEFAULT_UNITARY_CIRCUIT,
        "current": DEFAULT_UNITARY_CIRCUIT,
        "line_ry": DEFAULT_UNITARY_CIRCUIT,
    }
    name = aliases.get(name, name)
    if name not in UNITARY_CIRCUITS:
        raise ValueError(
            f"Unknown unitary circuit {unitary_circuit!r}. "
            f"Expected one of: {', '.join(UNITARY_CIRCUITS)}."
        )
    return name


def _ring_edges(targets: list[int]) -> list[tuple[int, int]]:
    edges = []
    edges.extend((targets[i], targets[i + 1]) for i in range(0, len(targets) - 1, 2))
    edges.extend((targets[i], targets[i + 1]) for i in range(1, len(targets) - 1, 2))
    if len(targets) > 2:
        edges.append((targets[-1], targets[0]))
    return edges


def _tree_edges(targets: list[int]) -> list[tuple[int, int]]:
    edges = []
    distance = 1
    while distance < len(targets):
        block_size = 2 * distance
        for block_start in range(0, len(targets), block_size):
            target = block_start + distance
            if target < len(targets):
                edges.append((targets[block_start], targets[target]))
        distance *= 2
    return edges


def num_unitary_parameters(
    qubits: int,
    layers: int,
    unitary_circuit: str | None = None,
) -> int:
    name = resolve_unitary_circuit(unitary_circuit)
    if name in {"uni2", "ring_ry_rz"}:
        return 2 * qubits * (layers + 1)
    if name == "ring_trainable_crx":
        return qubits * (layers + 1) + len(_ring_edges(list(range(qubits)))) * layers
    return qubits * (layers + 1)


def _unitary_gate_specs(
    control: int | None,
    targets_desc: list[int],
    layers: int,
    params: np.ndarray | list[float],
    unitary_circuit: str | None = None,
):
    specs = []
    name = resolve_unitary_circuit(unitary_circuit)
    p = np.asarray(params, dtype=float).reshape(-1)
    expected = num_unitary_parameters(len(targets_desc), layers, name)
    if p.size != expected:
        raise ValueError(
            f"Circuit {name!r} with qubits={len(targets_desc)} and layers={layers} "
            f"requires {expected} unitary parameters, received {p.size}."
        )

    idx = 0

    def add_rotation(axis: str, target: int) -> None:
        nonlocal idx
        gate = f"CR{axis}" if control is not None else f"R{axis}"
        qubits = (control, target) if control is not None else (target,)
        specs.append((gate, qubits, float(p[idx])))
        idx += 1

    def add_parameterized_gate(gate: str, qubits: tuple[int, ...]) -> None:
        nonlocal idx
        specs.append((gate, qubits, float(p[idx])))
        idx += 1

    for _ in range(layers):
        if name in {"uni2", "ring_ry_rz"}:
            for target in targets_desc:
                add_rotation("Y", target)
                add_rotation("Z", target)
        else:
            for target in targets_desc:
                add_rotation("Y", target)

        if name == "default":
            edges = list(zip(targets_desc[:-1], targets_desc[1:]))
            specs.extend(("CX", edge, None) for edge in edges)
        elif name == "uni2":
            edges = list(zip(targets_desc[:-1], targets_desc[1:]))
            if control is None:
                specs.extend(("CX", edge, None) for edge in edges)
            else:
                specs.extend(("CCX", (control, *edge), None) for edge in edges)
        elif name in {"brickwork_ring_ry", "ring_ry_rz"}:
            specs.extend(("CX", edge, None) for edge in _ring_edges(targets_desc))
        elif name == "ring_trainable_crx":
            for edge in _ring_edges(targets_desc):
                add_parameterized_gate("CRX", edge)
        else:  # multiscale_tree
            specs.extend(("CX", edge, None) for edge in _tree_edges(targets_desc))

    if name in {"uni2", "ring_ry_rz"}:
        for target in targets_desc:
            add_rotation("Y", target)
            add_rotation("Z", target)
    else:
        for target in targets_desc:
            add_rotation("Y", target)

    if idx != expected:
        raise RuntimeError(f"Circuit {name!r} consumed {idx} parameters; expected {expected}.")
    return specs


def _apply_specs(circ: qtn.Circuit, specs, inverse: bool = False, parametrize: bool = False) -> None:
    seq = reversed(specs) if inverse else specs
    for gate, qubits, param in seq:
        if param is not None:
            theta = -param if inverse else param
            circ.apply_gate(gate, qubits=qubits, params=(theta,), parametrize=parametrize)
        else:
            circ.apply_gate(gate, qubits=qubits)


def apply_unitary_forward(
    circ: qtn.Circuit,
    qubits: int,
    layers: int,
    params: np.ndarray | list[float],
    parametrize: bool = False,
    unitary_circuit: str | None = None,
) -> None:
    targets_desc = list(range(qubits, 0, -1))
    specs = _unitary_gate_specs(0, targets_desc, layers, params, unitary_circuit)
    _apply_specs(circ, specs, inverse=False, parametrize=parametrize)


def apply_unitary_inverse(
    circ: qtn.Circuit,
    qubits: int,
    layers: int,
    params: np.ndarray | list[float],
    parametrize: bool = False,
    unitary_circuit: str | None = None,
) -> None:
    targets_desc = list(range(qubits, 0, -1))
    specs = _unitary_gate_specs(0, targets_desc, layers, params, unitary_circuit)
    _apply_specs(circ, specs, inverse=True, parametrize=parametrize)


def _apply_unitary_on_register(
    circ: qtn.Circuit,
    control: int,
    targets_desc: list[int],
    layers: int,
    params: np.ndarray | list[float],
    inverse: bool,
    parametrize: bool = False,
    unitary_circuit: str | None = None,
) -> None:
    specs = _unitary_gate_specs(control, targets_desc, layers, params, unitary_circuit)
    _apply_specs(circ, specs, inverse=inverse, parametrize=parametrize)


def make_state_circuit(
    qubits: int,
    layers: int,
    params: np.ndarray | list[float],
    parametrize: bool = True,
    unitary_circuit: str | None = None,
) -> qtn.Circuit:
    qc = qtn.Circuit(qubits + 1)
    qc.apply_gate("X", qubits=(0,))
    apply_unitary_forward(qc, qubits, layers, params, parametrize, unitary_circuit)
    qc.apply_gate("X", qubits=(0,))
    return qc


def make_optimization_unitary(
    qubits: int,
    layers: int,
    params: np.ndarray | list[float],
    lbd0: float,
    unitary_circuit: str | None = None,
) -> qtn.Circuit:
    qc = qtn.Circuit(qubits + 1)
    apply_unitary_inverse(qc, qubits, layers, params, True, unitary_circuit)
    qc.apply_gate("H", qubits=(0,))
    qc.apply_gate("RY", qubits=(qubits + 1,), params=(float(lbd0),), parametrize=True)
    return qc

def make_optimization_unitary2(
    qubits: int,
    layers: int,
    params: np.ndarray | list[float],
    lbd0: float,
    unitary_circuit: str | None = None,
) -> qtn.Circuit:
    qc = qtn.Circuit(qubits + 1)
    apply_unitary_inverse(qc, qubits, layers, params, False, unitary_circuit)
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


def _apply_diagonal_inverse(
    circ: qtn.Circuit,
    n: int,
    layers: int,
    prev_params: np.ndarray | list[float],
    unitary_circuit: str | None = None,
) -> None:
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
        unitary_circuit=unitary_circuit,
    )

    for i in range(n):
        circ.apply_gate("CCX", qubits=(anc, right_data[i], left_data[i]))


def make_inverse_reference_circuits(
    prev_params: np.ndarray | list[float],
    qubits: int,
    layers: int,
    unitary_circuit: str | None = None,
) -> list[qtn.Circuit]:
    prev = np.asarray(prev_params, dtype=float)

    # c1 = H + U_prev on the (ancilla + right-data) register.
    c1 = qtn.Circuit(qubits + 1)
    c1.apply_gate("H", qubits=(0,))
    apply_unitary_forward(c1, qubits, layers, prev, False, unitary_circuit)

    # c2 / c3 use enlarged register containing adder ancillas.
    total_ua = 2 * qubits - 1
    c2 = qtn.Circuit(total_ua)
    c2.apply_gate("H", qubits=(0,))
    apply_unitary_forward(c2, qubits, layers, prev, False, unitary_circuit)
    _apply_adder(c2, qubits, dagger=True)

    c3 = qtn.Circuit(total_ua)
    c3.apply_gate("H", qubits=(0,))
    apply_unitary_forward(c3, qubits, layers, prev, False, unitary_circuit)
    _apply_adder(c3, qubits, dagger=False)

    # c4 / c5 additionally include the inverse-diagonal block.
    total_uda = 3 * qubits - 1
    c4 = qtn.Circuit(total_uda)
    c4.apply_gate("H", qubits=(0,))
    apply_unitary_forward(c4, qubits, layers, prev, False, unitary_circuit)
    _apply_adder(c4, qubits, dagger=False)
    _apply_diagonal_inverse(c4, qubits, layers, prev, unitary_circuit)

    c5 = qtn.Circuit(total_uda)
    c5.apply_gate("H", qubits=(0,))
    apply_unitary_forward(c5, qubits, layers, prev, False, unitary_circuit)
    _apply_adder(c5, qubits, dagger=True)
    _apply_diagonal_inverse(c5, qubits, layers, prev, unitary_circuit)

    return [c1, c2, c3, c4, c5]


def unitary_gate_count(
    qubits: int,
    layers: int,
    unitary_circuit: str | None = None,
) -> int:
    n_params = num_unitary_parameters(qubits, layers, unitary_circuit)
    specs = _unitary_gate_specs(
        control=0,
        targets_desc=list(range(qubits, 0, -1)),
        layers=layers,
        params=np.zeros(n_params),
        unitary_circuit=unitary_circuit,
    )
    return len(specs)


def make_pure_unitary_circuit(
    qubits: int,
    layers: int,
    params: np.ndarray | list[float],
    parametrize: bool = True,
    unitary_circuit: str | None = None,
) -> qtn.Circuit:
    qc = qtn.Circuit(qubits)
    targets_desc = list(range(qubits - 1, -1, -1))
    specs = _unitary_gate_specs(None, targets_desc, layers, params, unitary_circuit)
    _apply_specs(qc, specs, inverse=False, parametrize=parametrize)
    return qc
