from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
import yaml

import functions.circuits_quimb as fcq
import functions.initial_states as fist


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PRESETS_PATH = REPO_ROOT / "initial_params_presets.yaml"
DEFAULT_CHECKPOINT_DIR = Path("/tmp/vqcfd_initial_fit_presets")
N_VALUES = tuple(range(2, 7))
L_VALUES = tuple(range(1, 9))


def _ring_edges(targets: list[int]) -> list[tuple[int, int]]:
    edges = [(targets[i], targets[i + 1]) for i in range(0, len(targets) - 1, 2)]
    edges.extend((targets[i], targets[i + 1]) for i in range(1, len(targets) - 1, 2))
    if len(targets) > 2:
        edges.append((targets[-1], targets[0]))
    return edges


def _tree_edges(targets: list[int]) -> list[tuple[int, int]]:
    edges = []
    distance = 1
    while distance < len(targets):
        for start in range(0, len(targets), 2 * distance):
            target = start + distance
            if target < len(targets):
                edges.append((targets[start], targets[target]))
        distance *= 2
    return edges


def _gate_operations(n: int, layers: int, circuit: str):
    targets = list(range(n - 1, -1, -1))
    operations = []
    parameter_index = 0

    def rotation(gate: str, target: int) -> None:
        nonlocal parameter_index
        operations.append((gate, (target,), parameter_index))
        parameter_index += 1

    def parameterized_two_qubit(gate: str, edge: tuple[int, int]) -> None:
        nonlocal parameter_index
        operations.append((gate, edge, parameter_index))
        parameter_index += 1

    for _ in range(layers):
        for target in targets:
            rotation("RY", target)
            if circuit in {"uni2", "ring_ry_rz"}:
                rotation("RZ", target)

        if circuit in {"default", "uni2"}:
            operations.extend(("CX", edge, None) for edge in zip(targets[:-1], targets[1:]))
        elif circuit in {"brickwork_ring_ry", "ring_ry_rz"}:
            operations.extend(("CX", edge, None) for edge in _ring_edges(targets))
        elif circuit == "ring_trainable_crx":
            for edge in _ring_edges(targets):
                parameterized_two_qubit("CRX", edge)
        elif circuit == "multiscale_tree":
            operations.extend(("CX", edge, None) for edge in _tree_edges(targets))
        else:
            raise ValueError(f"Unsupported circuit: {circuit}")

    for target in targets:
        rotation("RY", target)
        if circuit in {"uni2", "ring_ry_rz"}:
            rotation("RZ", target)

    expected = fcq.num_unitary_parameters(n, layers, circuit)
    if parameter_index != expected:
        raise RuntimeError(f"Built {parameter_index} parameters for {circuit}; expected {expected}.")
    return operations


@lru_cache(maxsize=None)
def _gate_layout(qubits: tuple[int, ...], n: int):
    remaining = tuple(index for index in range(n) if index not in qubits)
    permutation = qubits + remaining
    indices = np.arange(2**n).reshape((2,) * n)
    return np.transpose(indices, permutation).reshape(2 ** len(qubits), -1)


def _apply_gate(state, matrix, qubits: tuple[int, ...], n: int):
    indices = _gate_layout(qubits, n)
    updated = np.dot(matrix, state[indices])
    result = np.empty_like(state)
    result[indices] = updated
    return result


def _gate_matrix(gate: str, theta: float | None = None):
    if gate == "CX":
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        )

    cosine = np.cos(theta / 2)
    sine = np.sin(theta / 2)
    if gate == "RY":
        return np.array([[cosine, -sine], [sine, cosine]], dtype=complex)
    if gate == "RZ":
        return np.array(
            [[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]],
            dtype=complex,
        )
    if gate == "CRX":
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, cosine, -1j * sine],
                [0, 0, -1j * sine, cosine],
            ],
            dtype=complex,
        )
    raise ValueError(f"Unsupported gate: {gate}")


def _gate_derivative(gate: str, theta: float):
    cosine = np.cos(theta / 2)
    sine = np.sin(theta / 2)
    if gate == "RY":
        return 0.5 * np.array([[-sine, -cosine], [cosine, -sine]], dtype=complex)
    if gate == "RZ":
        return np.array(
            [
                [-0.5j * np.exp(-0.5j * theta), 0],
                [0, 0.5j * np.exp(0.5j * theta)],
            ],
            dtype=complex,
        )
    if gate == "CRX":
        return np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, -0.5 * sine, -0.5j * cosine],
                [0, 0, -0.5j * cosine, -0.5 * sine],
            ],
            dtype=complex,
        )
    raise ValueError(f"Gate {gate} has no trainable derivative.")


def _statevector(params, n: int, operations):
    state = np.concatenate((np.array([1.0 + 0.0j]), np.zeros(2**n - 1, dtype=complex)))
    for gate, qubits, parameter_index in operations:
        theta = None if parameter_index is None else params[parameter_index]
        matrix = _gate_matrix(gate, theta)
        state = _apply_gate(state, matrix, qubits, n)
    return state


def _physical_mse(n: int, layers: int, circuit: str, initial_state: str, params) -> float:
    params = np.asarray(params, dtype=float)
    xs = np.linspace(0.0, 1.0, 2**n)
    field = fist.make_initial_field(xs, initial_state)
    state = _statevector(params[1:], n, _gate_operations(n, layers, circuit))
    return float(np.mean((params[0] * state.real - field) ** 2))


def _loss_and_gradient(params, n: int, operations, desired):
    states = [
        np.concatenate((np.array([1.0 + 0.0j]), np.zeros(2**n - 1, dtype=complex)))
    ]
    matrices = []
    for gate, qubits, parameter_index in operations:
        theta = None if parameter_index is None else params[parameter_index]
        matrix = _gate_matrix(gate, theta)
        matrices.append(matrix)
        states.append(_apply_gate(states[-1], matrix, qubits, n))

    difference = states[-1] - desired
    loss = float(np.real(np.vdot(difference, difference)))
    adjoint = difference
    gradient = np.zeros_like(params, dtype=float)

    for operation_index in range(len(operations) - 1, -1, -1):
        gate, qubits, parameter_index = operations[operation_index]
        if parameter_index is not None:
            derivative_state = _apply_gate(
                states[operation_index],
                _gate_derivative(gate, params[parameter_index]),
                qubits,
                n,
            )
            gradient[parameter_index] = 2.0 * float(np.real(np.vdot(adjoint, derivative_state)))
        adjoint = _apply_gate(adjoint, matrices[operation_index].conj().T, qubits, n)

    return loss, gradient


def _stable_seed(n: int, layers: int, circuit: str, initial_state: str, trial: int) -> int:
    key = f"{n}:{layers}:{circuit}:{initial_state}:{trial}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**32)


def _fit_once(
    n: int,
    layers: int,
    circuit: str,
    initial_state: str,
    trial: int,
    optimization_steps: int,
):
    xs = np.linspace(0.0, 1.0, 2**n)
    field = fist.make_initial_field(xs, initial_state)
    norm = float(np.linalg.norm(field))
    desired = np.asarray(field / norm, dtype=complex)
    operations = _gate_operations(n, layers, circuit)
    n_parameters = fcq.num_unitary_parameters(n, layers, circuit)
    rng = np.random.default_rng(_stable_seed(n, layers, circuit, initial_state, trial))
    initial = rng.random(n_parameters) * 0.1 + (np.pi - 0.25)

    result = minimize(
        _loss_and_gradient,
        initial,
        args=(n, operations, desired),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": optimization_steps},
    )
    fitted = np.asarray(result.x, dtype=float)
    fitted_state = np.asarray(_statevector(fitted, n, operations))
    mse = float(np.mean((norm * fitted_state.real - field) ** 2))
    normalized_loss = float(
        np.real(np.sum(np.conjugate(fitted_state - desired) * (fitted_state - desired)))
    )
    return {
        "trial": trial,
        "mse": mse,
        "normalized_loss": normalized_loss,
        "params": [norm, *fitted.tolist()],
        "success": bool(result.success),
        "iterations": int(result.nit),
        "evaluations": int(result.nfev),
    }


def _fit_combination(job):
    n, layers, circuit, initial_state, tries, optimization_steps = job
    start = time.perf_counter()
    results = [
        _fit_once(n, layers, circuit, initial_state, trial, optimization_steps)
        for trial in range(tries)
    ]
    best = min(results, key=lambda result: result["mse"])
    return {
        "n": n,
        "l": layers,
        "circuit": circuit,
        "initial_state": initial_state,
        "mse": best["mse"],
        "normalized_loss": best["normalized_loss"],
        "params": best["params"],
        "tries": tries,
        "best_trial": best["trial"],
        "optimizer": "L-BFGS-B",
        "optimization_steps": optimization_steps,
        "best_iterations": best["iterations"],
        "best_evaluations": best["evaluations"],
        "elapsed_seconds": time.perf_counter() - start,
    }


def _record_key(n: int, layers: int, circuit: str, initial_state: str) -> str:
    return f"n{n}_l{layers}_{initial_state}_{circuit}"


def _checkpoint_path(checkpoint_dir: Path, job) -> Path:
    n, layers, circuit, initial_state, *_ = job
    return checkpoint_dir / f"{_record_key(n, layers, circuit, initial_state)}.json"


def _load_presets(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _stored_preset_records(presets):
    records = {}
    for n in N_VALUES:
        by_l = presets.get(n, presets.get(str(n), {}))
        for layers in L_VALUES:
            entry = by_l.get(layers, by_l.get(str(layers)))
            if not isinstance(entry, dict):
                continue
            for initial_state, circuits in (entry.get("initial_states") or {}).items():
                for circuit, record in (circuits or {}).items():
                    records[(n, layers, circuit, initial_state)] = record
            if entry.get("default") is not None:
                mse = entry.get("default_mse")
                if mse is None:
                    mse = _physical_mse(n, layers, "default", "sine", entry["default"])
                records.setdefault(
                    (n, layers, "default", "sine"),
                    {
                        "circuit": "default",
                        "initial_state": "sine",
                        "mse": mse,
                        "params": entry["default"],
                        "source": "legacy_default",
                    },
                )
            variant = (entry.get("variants") or {}).get("uni2")
            if variant is not None:
                variant_params = (
                    variant.get("params") if isinstance(variant, dict) else variant
                )
                records.setdefault(
                    (n, layers, "uni2", "sine"),
                    {
                        "circuit": "uni2",
                        "initial_state": "sine",
                        "mse": variant.get("mse") if isinstance(variant, dict) else None,
                        "params": variant_params,
                        "source": "legacy_variant",
                    },
                )
    return records


def _load_checkpoint_records(checkpoint_dir: Path):
    records = {}
    if not checkpoint_dir.exists():
        return records
    for path in checkpoint_dir.glob("*.json"):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        key = (record["n"], record["l"], record["circuit"], record["initial_state"])
        records[key] = record
    return records


def _write_checkpoint(path: Path, record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    temporary.replace(path)


def _all_jobs(tries: int, optimization_steps: int):
    return [
        (n, layers, circuit, initial_state, tries, optimization_steps)
        for n in N_VALUES
        for layers in L_VALUES
        for initial_state in fist.INITIAL_STATES
        for circuit in fcq.UNITARY_CIRCUITS
    ]


def run_campaign(args) -> None:
    presets = _load_presets(args.presets)
    complete = set(_stored_preset_records(presets)) | set(
        _load_checkpoint_records(args.checkpoint_dir)
    )
    all_jobs = _all_jobs(args.tries, args.optimization_steps)
    sharded_jobs = all_jobs[args.shard_index :: args.shard_count]
    jobs = [
        job
        for job in sharded_jobs
        if (job[0], job[1], job[2], job[3]) not in complete
    ]
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    print(
        f"shard={args.shard_index + 1}/{args.shard_count} complete={len(complete)} "
        f"pending_this_run={len(jobs)} total_matrix=1440",
        flush=True,
    )
    if not jobs:
        return

    for done, job in enumerate(jobs, start=1):
        record = _fit_combination(job)
        _write_checkpoint(_checkpoint_path(args.checkpoint_dir, job), record)
        print(
            f"[{done}/{len(jobs)}] N={record['n']} L={record['l']} "
            f"state={record['initial_state']} circuit={record['circuit']} "
            f"mse={record['mse']:.6e} elapsed={record['elapsed_seconds']:.2f}s",
            flush=True,
        )


def merge_presets(args) -> None:
    presets = _load_presets(args.presets)
    records = _stored_preset_records(presets)
    records.update(_load_checkpoint_records(args.checkpoint_dir))
    expected = len(N_VALUES) * len(L_VALUES) * len(fist.INITIAL_STATES) * len(fcq.UNITARY_CIRCUITS)
    if len(records) != expected:
        raise RuntimeError(f"Cannot merge incomplete campaign: found {len(records)} of {expected} records.")

    for n in N_VALUES:
        by_l = presets.setdefault(n, {})
        for layers in L_VALUES:
            entry = by_l.setdefault(layers, {})
            if entry.get("default") is not None and entry.get("default_mse") is None:
                entry["default_mse"] = records[(n, layers, "default", "sine")]["mse"]
            initial_states = entry.setdefault("initial_states", {})
            for initial_state in fist.INITIAL_STATES:
                state_records = initial_states.setdefault(initial_state, {})
                for circuit in fcq.UNITARY_CIRCUITS:
                    record = records[(n, layers, circuit, initial_state)]
                    state_records[circuit] = {
                        key: value
                        for key, value in record.items()
                        if key not in {"n", "l", "elapsed_seconds"}
                    }
                    state_records[circuit]["params"] = list(record["params"])

    # Filling legacy gaps appends keys, so normalize the numeric ordering before
    # writing the expanded file.
    presets = {
        n: {layers: by_l[layers] for layers in sorted(by_l, key=int)}
        for n, by_l in sorted(presets.items(), key=lambda item: int(item[0]))
    }

    temporary = args.presets.with_suffix(".yaml.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(presets, handle, sort_keys=False)
    temporary.replace(args.presets)
    print(f"Merged {expected} records into {args.presets}.")


def validate_dense_simulator() -> None:
    for n in (2, 4, 6):
        for circuit in fcq.UNITARY_CIRCUITS:
            layers = 2
            count = fcq.num_unitary_parameters(n, layers, circuit)
            params = np.linspace(-0.7, 0.9, count)
            operations = _gate_operations(n, layers, circuit)
            dense = np.asarray(_statevector(params, n, operations))
            quimb = np.asarray(
                fcq.make_pure_unitary_circuit(
                    n,
                    layers,
                    params,
                    parametrize=False,
                    unitary_circuit=circuit,
                ).to_dense()
            ).reshape(-1)
            if not np.allclose(dense, quimb, atol=1e-12):
                raise AssertionError(f"Dense simulator mismatch for N={n}, circuit={circuit}.")
    print("Dense initial-fit simulator matches Quimb for every circuit family.")


def validate_presets(args) -> None:
    presets = _load_presets(args.presets)
    generated = 0
    legacy = 0
    checked = 0
    for n in N_VALUES:
        by_l = presets.get(n, presets.get(str(n), {}))
        for layers in L_VALUES:
            entry = by_l.get(layers, by_l.get(str(layers)))
            if not isinstance(entry, dict):
                raise AssertionError(f"Missing preset entry for N={n}, L={layers}.")
            initial_states = entry.get("initial_states") or {}
            for initial_state in fist.INITIAL_STATES:
                circuits = initial_states.get(initial_state) or {}
                for circuit in fcq.UNITARY_CIRCUITS:
                    record = circuits.get(circuit)
                    if not isinstance(record, dict):
                        raise AssertionError(
                            f"Missing preset for N={n}, L={layers}, circuit={circuit}, "
                            f"initial_state={initial_state}."
                        )
                    expected_params = fcq.num_unitary_parameters(n, layers, circuit) + 1
                    params = record.get("params")
                    if not isinstance(params, list) or len(params) != expected_params:
                        raise AssertionError(
                            f"Invalid parameter count for N={n}, L={layers}, circuit={circuit}, "
                            f"initial_state={initial_state}: expected {expected_params}."
                        )
                    mse = record.get("mse")
                    if mse is None or not np.isfinite(float(mse)):
                        raise AssertionError(
                            f"Missing finite MSE for N={n}, L={layers}, circuit={circuit}, "
                            f"initial_state={initial_state}."
                        )
                    if "tries" in record:
                        generated += 1
                        if record["tries"] != args.tries:
                            raise AssertionError(
                                f"Expected {args.tries} tries, got {record['tries']} for "
                                f"N={n}, L={layers}, circuit={circuit}, "
                                f"initial_state={initial_state}."
                            )
                        recomputed = _physical_mse(
                            n, layers, circuit, initial_state, record["params"]
                        )
                        if not np.isclose(recomputed, float(mse), rtol=1e-9, atol=1e-13):
                            raise AssertionError(
                                f"Stored MSE mismatch for N={n}, L={layers}, circuit={circuit}, "
                                f"initial_state={initial_state}: {mse} vs {recomputed}."
                            )
                    else:
                        legacy += 1
                    checked += 1

    expected = len(N_VALUES) * len(L_VALUES) * len(fist.INITIAL_STATES) * len(
        fcq.UNITARY_CIRCUITS
    )
    if checked != expected:
        raise AssertionError(f"Validated {checked} records; expected {expected}.")
    print(
        f"Validated {checked} presets ({generated} generated with {args.tries} tries, "
        f"{legacy} reused legacy records), including parameter counts and stored MSEs."
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "merge", "validate"))
    parser.add_argument("--presets", type=Path, default=DEFAULT_PRESETS_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--tries", type=int, default=20)
    parser.add_argument("--optimization-steps", type=int, default=1000)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    if args.tries < 1:
        parser.error("--tries must be at least 1")
    if args.optimization_steps < 1:
        parser.error("--optimization-steps must be at least 1")
    if args.shard_count < 1:
        parser.error("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must satisfy 0 <= index < shard count")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.action == "run":
        run_campaign(arguments)
    elif arguments.action == "merge":
        merge_presets(arguments)
    else:
        validate_dense_simulator()
        validate_presets(arguments)
