import unittest

import numpy as np
import quimb.tensor as qtn
import quimb.tensor.circuit as qtn_circuit

from functions import circuits_quimb as fcq
from functions import functions_quimb as fqu


class CircuitVariantTests(unittest.TestCase):
    def test_new_names_and_aliases_resolve(self):
        self.assertEqual(fcq.resolve_unitary_circuit("default_zxz"), "default_rzrxrz")
        self.assertEqual(fcq.resolve_unitary_circuit("paper_fig2"), "paper_viscid")
        self.assertEqual(
            fcq.resolve_unitary_circuit("tree_zxz"),
            "multiscale_tree_rzrxrz",
        )

    def test_rzrxrz_parameter_counts_triple_the_ry_counts(self):
        n = 4
        layers = 2
        self.assertEqual(
            fcq.num_unitary_parameters(n, layers, "default_rzrxrz"),
            3 * fcq.num_unitary_parameters(n, layers, "default"),
        )
        self.assertEqual(
            fcq.num_unitary_parameters(n, layers, "multiscale_tree_rzrxrz"),
            3 * fcq.num_unitary_parameters(n, layers, "multiscale_tree"),
        )

    def test_rzrxrz_variants_replace_each_ry_with_zxz(self):
        n = 4
        for circuit_name in ("default_rzrxrz", "multiscale_tree_rzrxrz"):
            params = np.zeros(fcq.num_unitary_parameters(n, 1, circuit_name))
            specs = fcq._unitary_gate_specs(
                None,
                list(range(n - 1, -1, -1)),
                1,
                params,
                circuit_name,
            )
            rotations = [gate for gate, _, _, trainable in specs if trainable]
            self.assertEqual(rotations, ["RZ", "RX", "RZ"] * (2 * n))

    def test_paper_viscid_matches_published_n3_plus_ancilla_topology(self):
        params = np.arange(5, dtype=float)
        specs = fcq._unitary_gate_specs(
            0,
            [3, 2, 1],
            1,
            params,
            "paper_viscid",
        )
        self.assertEqual(
            [(gate, qubits) for gate, qubits, _, _ in specs],
            [
                ("CX", (0, 3)),
                ("PAPER_G", (3, 2)),
                ("PAPER_G", (2, 1)),
                ("PAPER_G", (1, 2)),
                ("PAPER_G", (2, 3)),
                ("PAPER_G", (0, 3)),
            ],
        )
        self.assertEqual([parameter for _, _, parameter, _ in specs[1:]], list(params))
        self.assertEqual(fcq.unitary_gate_count(3, 1, "paper_viscid"), 16)

    def test_paper_g_is_ry_cx_ry_block(self):
        theta = 0.37
        identity = np.eye(2)
        ry_minus = qtn_circuit.PARAM_GATES["RY"]((-theta,))
        ry_plus = qtn_circuit.PARAM_GATES["RY"]((theta,))
        cx = qtn_circuit.CONSTANT_GATES["CX"]
        expected = np.kron(identity, ry_plus) @ cx @ np.kron(identity, ry_minus)
        np.testing.assert_allclose(fcq._paper_g_dense((theta,)), expected, atol=1e-12)

        x_gate = qtn_circuit.CONSTANT_GATES["X"]
        active_expected = ry_plus @ x_gate @ ry_minus
        np.testing.assert_allclose(
            fcq._paper_g_active_dense((theta,)),
            active_expected,
            atol=1e-12,
        )

    def test_paper_viscid_uses_one_uniform_parameter_formula(self):
        self.assertEqual(fcq.num_unitary_parameters(3, 1, "paper_viscid"), 5)
        self.assertEqual(fcq.num_unitary_parameters(4, 2, "paper_viscid"), 13)
        self.assertEqual(fcq.num_unitary_parameters(5, 2, "paper_viscid"), 17)

    def test_paper_viscid_has_only_two_ancilla_endpoint_connections(self):
        n = 5
        params = np.zeros(fcq.num_unitary_parameters(n, 2, "paper_viscid"))
        specs = fcq._unitary_gate_specs(
            0,
            list(range(n, 0, -1)),
            2,
            params,
            "paper_viscid",
        )
        ancilla_specs = [
            (gate, qubits)
            for gate, qubits, _, _ in specs
            if 0 in qubits
        ]
        self.assertEqual(
            ancilla_specs,
            [("CX", (0, 5)), ("PAPER_G", (0, 5))],
        )
        first_v_edges = [
            qubits for gate, qubits, _, _ in specs[1:9] if gate == "PAPER_G"
        ]
        self.assertEqual(
            first_v_edges,
            [
                (5, 4),
                (4, 3),
                (3, 2),
                (2, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
            ],
        )
        self.assertEqual(fcq.unitary_gate_count(n, 2, "paper_viscid"), 52)

    def test_paper_viscid_two_point_shift_rule_is_exact(self):
        theta = 0.37
        rule = fqu.parameter_shift_rule("paper_viscid", 7, qubits=5)
        self.assertEqual(
            rule,
            ((np.pi / 2, 0.5), (-np.pi / 2, -0.5)),
        )
        value = lambda angle: 0.3 + 0.7 * np.cos(angle) - 0.2 * np.sin(angle)
        derivative = sum(weight * value(theta + shift) for shift, weight in rule)
        expected = -0.7 * np.sin(theta) - 0.2 * np.cos(theta)
        self.assertAlmostEqual(derivative, expected, places=12)
        self.assertEqual(
            fqu.parameter_shift_evaluations_per_sweep(17, "paper_viscid", 5),
            34,
        )

    def test_controlled_paper_viscid_preparation_matches_pure_unitary(self):
        rng = np.random.default_rng(37)
        params = rng.normal(size=fcq.num_unitary_parameters(3, 1, "paper_viscid"))
        pure = fcq.make_pure_unitary_circuit(3, 1, params, False, "paper_viscid")
        controlled = fcq.make_state_circuit(3, 1, params, False, "paper_viscid")

        pure_state = np.asarray(pure.to_dense()).reshape(-1)
        controlled_state = np.asarray(controlled.to_dense()).reshape(2, -1)
        np.testing.assert_allclose(controlled_state[0], pure_state, atol=1e-12)
        np.testing.assert_allclose(controlled_state[1], 0.0, atol=1e-12)

    def test_general_paper_viscid_is_real_and_controlled_matches_pure(self):
        rng = np.random.default_rng(39)
        n = 5
        params = rng.normal(size=fcq.num_unitary_parameters(n, 2, "paper_viscid"))
        pure = fcq.make_pure_unitary_circuit(n, 2, params, False, "paper_viscid")
        controlled = fcq.make_state_circuit(n, 2, params, False, "paper_viscid")

        pure_state = np.asarray(pure.to_dense()).reshape(-1)
        controlled_state = np.asarray(controlled.to_dense()).reshape(2, -1)
        np.testing.assert_allclose(np.imag(pure_state), 0.0, atol=1e-12)
        np.testing.assert_allclose(controlled_state[0], pure_state, atol=1e-12)
        np.testing.assert_allclose(controlled_state[1], 0.0, atol=1e-12)

    def test_paper_g_blocks_have_exact_inverses(self):
        params = (0.31,)
        forward = fcq._paper_g_dense(params)
        inverse = fcq._paper_g_dense_dagger(params)
        np.testing.assert_allclose(inverse @ forward, np.eye(4), atol=1e-12)

        active = fcq._paper_g_active_dense(params)
        active_inverse = fcq._paper_g_active_dense_dagger(params)
        np.testing.assert_allclose(active_inverse @ active, np.eye(2), atol=1e-12)

    def test_new_variants_invert_exactly(self):
        rng = np.random.default_rng(41)
        cases = (
            (4, 2, "default_rzrxrz"),
            (4, 2, "multiscale_tree_rzrxrz"),
            (3, 1, "paper_viscid"),
            (5, 2, "paper_viscid"),
        )
        for n, layers, circuit_name in cases:
            params = rng.normal(
                size=fcq.num_unitary_parameters(n, layers, circuit_name)
            )
            specs = fcq._unitary_gate_specs(
                None,
                list(range(n - 1, -1, -1)),
                layers,
                params,
                circuit_name,
            )
            circuit = qtn.Circuit(n)
            fcq._apply_specs(circuit, specs)
            fcq._apply_specs(circuit, specs, inverse=True)

            actual = np.asarray(circuit.to_dense()).reshape(-1)
            expected = np.zeros(2**n, dtype=complex)
            expected[0] = 1.0
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_nonlinear_copy_network_does_not_touch_hadamard_ancilla(self):
        n = 4
        params = np.zeros(fcq.num_unitary_parameters(n, 1, "default"))
        circuit = qtn.Circuit(3 * n - 1)
        fcq._apply_diagonal_inverse(circuit, n, 1, params, "default")

        copy_gates = circuit.gates[-n:]
        self.assertEqual([gate.label for gate in copy_gates], ["CX"] * n)
        self.assertTrue(all(0 not in gate.qubits for gate in copy_gates))
        self.assertEqual(
            [gate.qubits for gate in copy_gates],
            [(4, 10), (3, 9), (2, 8), (1, 7)],
        )


if __name__ == "__main__":
    unittest.main()
