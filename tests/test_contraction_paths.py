import tempfile
import unittest
from pathlib import Path

import numpy as np
import quimb.tensor as qtn
import quimb.tensor.circuit as qtn_circuit

import functions.circuits_quimb as fcq
import functions.contraction_paths as fcp
import functions.functions_quimb as fqu
from sim_config import SimConfig
from simulation_runner import (
    _prepare_local_contraction_paths,
    _prepare_pure_state_path,
    build_runtime,
)


class ContractionPathTests(unittest.TestCase):
    def make_config(self, cache_dir: str) -> SimConfig:
        return SimConfig(
            n=2,
            l=1,
            unitary_circuit="mps_staircase_light",
            verbose=False,
            optimize_contraction_paths=True,
            contraction_path_repeats=2,
            contraction_path_max_time_s=2.0,
            contraction_path_cache_dir=cache_dir,
            contraction_path_seed=7,
        )

    def test_optimized_paths_are_disabled_by_default(self):
        cfg = SimConfig(n=2, l=1, verbose=False)
        rt = build_runtime(cfg, plot_circuit=False)
        self.assertFalse(cfg.optimize_contraction_paths)
        self.assertIsNone(_prepare_local_contraction_paths(cfg, rt))
        self.assertIsNone(_prepare_pure_state_path(cfg, rt))

    def test_local_paths_match_auto_and_work_with_autograd(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            cfg = self.make_config(cache_dir)
            rt = build_runtime(cfg, plot_circuit=False)
            path_set = _prepare_local_contraction_paths(cfg, rt)

            self.assertEqual(path_set.source, "searched")
            self.assertEqual(len(path_set.paths), 5)
            self.assertTrue(Path(path_set.cache_path).exists())

            rng = np.random.default_rng(19)
            current = rng.uniform(0.2, 5.8, rt.n_params)
            reference = rng.uniform(0.2, 5.8, rt.n_params)
            unitary = fqu.make_optimization_unitary(
                cfg.n,
                cfg.l,
                current,
                1.2,
                cfg.resolved_unitary_circuit,
            )
            refs = fqu.make_inverse_reference_circuits(
                reference,
                cfg.n,
                cfg.l,
                cfg.resolved_unitary_circuit,
            )

            exact = fqu.local_expectations(
                unitary,
                *refs,
                rt.wires,
                path_set.paths,
                path_set.simplify_sequence,
            )
            automatic = fqu.local_expectations(unitary, *refs, rt.wires)
            np.testing.assert_allclose(exact, automatic, atol=1e-12, rtol=1e-12)

            constants = dict(
                qc1=refs[0],
                qc2=refs[1],
                qc3=refs[2],
                qc4=refs[3],
                qc5=refs[4],
                lbd0_t=1.1,
                wires=rt.wires,
                dt=0.01,
                dx=1 / 3,
                mu=0.01,
            )
            optimizer = qtn.TNOptimizer(
                unitary,
                fqu.cost_quimb,
                loss_constants=constants,
                loss_kwargs=dict(
                    paths=path_set.paths,
                    simplify_sequence=path_set.simplify_sequence,
                ),
                autodiff_backend="autograd",
                progbar=False,
            )
            # A trainable gate embedded as a non-parametrized gate enters
            # Quimb's global numeric gate cache. With Autograd that cache then
            # retains the complete VJP graph from every evaluation.
            qtn_circuit._cached_param_gate_build.cache_clear()
            value, gradient = optimizer.vectorized_value_and_grad(
                optimizer.vectorizer.vector
            )
            self.assertTrue(np.isfinite(value))
            self.assertTrue(np.all(np.isfinite(gradient)))
            self.assertEqual(
                qtn_circuit._cached_param_gate_build.cache_info().currsize,
                0,
            )

            gradient_params = np.concatenate(([1.2], current))
            selected_gradient = fqu.whole_local_expectation_grads_param_shift(
                gradient_params,
                *refs,
                rt.wires,
                cfg.n,
                cfg.l,
                path_set.paths,
                cfg.resolved_unitary_circuit,
                path_set.simplify_sequence,
            )
            automatic_gradient = fqu.whole_local_expectation_grads_param_shift(
                gradient_params,
                *refs,
                rt.wires,
                cfg.n,
                cfg.l,
                None,
                cfg.resolved_unitary_circuit,
            )
            np.testing.assert_allclose(
                selected_gradient,
                automatic_gradient,
                atol=1e-11,
                rtol=1e-11,
            )

            fcp.clear_memory_cache()
            rt_reloaded = build_runtime(cfg, plot_circuit=False)
            reloaded = _prepare_local_contraction_paths(cfg, rt_reloaded)
            self.assertEqual(reloaded.source, "disk_cache")
            self.assertEqual(reloaded.paths, path_set.paths)

    def test_pure_state_path_matches_automatic_contraction(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            cfg = self.make_config(cache_dir)
            path_set = _prepare_pure_state_path(cfg)
            rng = np.random.default_rng(23)
            params = rng.uniform(0.2, 5.8, cfg.number_of_parameters)
            circuit = fcq.make_pure_unitary_circuit(
                cfg.n,
                cfg.l,
                params,
                parametrize=False,
                unitary_circuit=cfg.resolved_unitary_circuit,
            )

            exact = circuit.to_dense(optimize=path_set.paths[0])
            automatic = circuit.to_dense(optimize="greedy")
            np.testing.assert_allclose(exact, automatic, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
