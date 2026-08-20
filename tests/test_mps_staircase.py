import unittest

import numpy as np
import quimb.tensor as qtn

from functions import circuits_quimb as fcq


class MPSStaircaseTests(unittest.TestCase):
    def test_parameter_count_matches_mpd_blocks(self):
        self.assertEqual(fcq.num_unitary_parameters(1, 2, "mps_staircase"), 6)
        self.assertEqual(fcq.num_unitary_parameters(4, 1, "mps_staircase"), 48)
        self.assertEqual(fcq.num_unitary_parameters(4, 3, "mps_staircase"), 144)
        self.assertEqual(
            fcq.num_unitary_parameters(1, 2, "mps_staircase_light"), 2
        )
        self.assertEqual(
            fcq.num_unitary_parameters(4, 1, "mps_staircase_light"), 19
        )
        self.assertEqual(
            fcq.num_unitary_parameters(4, 3, "mps_staircase_light"), 57
        )

    def test_fixed_entanglers_remain_data_only(self):
        for circuit_name in ("uni2", "mps_staircase", "mps_staircase_light"):
            n = 4
            layers = 1
            params = np.zeros(
                fcq.num_unitary_parameters(n, layers, circuit_name)
            )
            specs = fcq._unitary_gate_specs(
                0,
                list(range(n, 0, -1)),
                layers,
                params,
                circuit_name,
            )
            fixed_entanglers = [
                (gate, qubits)
                for gate, qubits, _, trainable in specs
                if not trainable and gate in {"CX", "CCX"}
            ]
            self.assertTrue(fixed_entanglers)
            self.assertTrue(all(gate == "CX" for gate, _ in fixed_entanglers))
            self.assertTrue(all(0 not in qubits for _, qubits in fixed_entanglers))

    def test_forward_then_inverse_is_identity(self):
        n = 4
        layers = 2
        rng = np.random.default_rng(4)
        params = rng.normal(
            size=fcq.num_unitary_parameters(n, layers, "mps_staircase")
        )
        specs = fcq._unitary_gate_specs(
            None,
            list(range(n - 1, -1, -1)),
            layers,
            params,
            "mps_staircase",
        )
        circuit = qtn.Circuit(n)
        fcq._apply_specs(circuit, specs)
        fcq._apply_specs(circuit, specs, inverse=True)

        actual = np.asarray(circuit.to_dense()).reshape(-1)
        expected = np.zeros(2**n, dtype=complex)
        expected[0] = 1.0
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_one_layer_has_mps_bond_dimension_two(self):
        n = 5
        rng = np.random.default_rng(8)
        params = rng.normal(
            size=fcq.num_unitary_parameters(n, 1, "mps_staircase")
        )
        circuit = fcq.make_pure_unitary_circuit(
            n,
            1,
            params,
            parametrize=False,
            unitary_circuit="mps_staircase",
        )
        state = np.asarray(circuit.to_dense()).reshape(-1)

        for cut in range(1, n):
            singular_values = np.linalg.svd(
                state.reshape(2**cut, 2 ** (n - cut)), compute_uv=False
            )
            rank = np.count_nonzero(singular_values > 1e-10)
            self.assertLessEqual(rank, 2)

    def test_controlled_state_preparation_matches_pure_circuit(self):
        n = 3
        layers = 1
        rng = np.random.default_rng(12)
        params = rng.normal(
            size=fcq.num_unitary_parameters(n, layers, "mps_staircase")
        )
        pure = fcq.make_pure_unitary_circuit(
            n, layers, params, False, "mps_staircase"
        )
        controlled = fcq.make_state_circuit(
            n, layers, params, False, "mps_staircase"
        )

        pure_state = np.asarray(pure.to_dense()).reshape(-1)
        controlled_state = np.asarray(controlled.to_dense()).reshape(2, -1)
        np.testing.assert_allclose(controlled_state[0], pure_state, atol=1e-12)
        np.testing.assert_allclose(controlled_state[1], 0.0, atol=1e-12)

    def test_data_only_cnots_preserve_inactive_hadamard_branch(self):
        n = 3
        rng = np.random.default_rng(15)
        for circuit_name in ("uni2", "mps_staircase", "mps_staircase_light"):
            params = rng.normal(
                size=fcq.num_unitary_parameters(n, 1, circuit_name)
            )
            pure = fcq.make_pure_unitary_circuit(
                n, 1, params, False, circuit_name
            )
            pure_state = np.asarray(pure.to_dense()).reshape(-1)

            controlled = qtn.Circuit(n + 1)
            controlled.apply_gate("H", qubits=(0,))
            fcq.apply_unitary_forward(
                controlled, n, 1, params, False, circuit_name
            )
            actual = np.asarray(controlled.to_dense()).reshape(2, -1)

            zero_state = np.zeros(2**n, dtype=complex)
            zero_state[0] = 1.0
            np.testing.assert_allclose(
                actual[0], zero_state / np.sqrt(2), atol=1e-12
            )
            np.testing.assert_allclose(
                actual[1], pure_state / np.sqrt(2), atol=1e-12
            )

    def test_light_staircase_is_real_and_has_bond_dimension_two(self):
        n = 5
        rng = np.random.default_rng(18)
        params = rng.normal(
            size=fcq.num_unitary_parameters(n, 1, "mps_staircase_light")
        )
        circuit = fcq.make_pure_unitary_circuit(
            n, 1, params, False, "mps_staircase_light"
        )
        state = np.asarray(circuit.to_dense()).reshape(-1)
        np.testing.assert_allclose(state.imag, 0.0, atol=1e-12)

        for cut in range(1, n):
            singular_values = np.linalg.svd(
                state.reshape(2**cut, 2 ** (n - cut)), compute_uv=False
            )
            rank = np.count_nonzero(singular_values > 1e-10)
            self.assertLessEqual(rank, 2)


if __name__ == "__main__":
    unittest.main()
