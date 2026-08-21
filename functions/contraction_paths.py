from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Sequence
import json
import math
import os
import random
import time

import cotengra as ctg
import yaml


PATH_CACHE_FORMAT_VERSION = 1
ContractionPath = tuple[tuple[int, ...], ...]
RehearseFn = Callable[[object], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ContractionPathSet:
    """Exact positional contraction paths for one fixed family of networks."""

    kind: str
    paths: tuple[ContractionPath, ...]
    simplify_sequence: str
    metrics: tuple[dict[str, Any], ...]
    source: str
    search_time_s: float
    prepare_time_s: float
    cache_path: str | None
    metadata: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "simplify_sequence": self.simplify_sequence,
            "search_time_s": float(self.search_time_s),
            "prepare_time_s": float(self.prepare_time_s),
            "cache_path": self.cache_path,
            "networks": [dict(item) for item in self.metrics],
        }


_MEMORY_CACHE: dict[str, ContractionPathSet] = {}


def clear_memory_cache() -> None:
    """Clear process-local path sets, primarily for tests."""
    _MEMORY_CACHE.clear()


def _normalized_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _memory_key(cache_path: Path | None, metadata: dict[str, Any]) -> str:
    location = "<memory>" if cache_path is None else str(cache_path.resolve())
    return f"{location}:{_normalized_json(metadata)}"


def _coerce_path(raw_path: Sequence[Sequence[int]]) -> ContractionPath:
    return tuple(tuple(int(index) for index in step) for step in raw_path)


def _topology_fingerprint(tn) -> str:
    """Hash tensor order, index incidence, and dimensions, ignoring UUID names."""
    canonical_indices: dict[str, int] = {}
    tensors = []
    for tensor in tn.tensors:
        inds = []
        for index, dimension in zip(tensor.inds, tensor.shape):
            canonical = canonical_indices.setdefault(index, len(canonical_indices))
            inds.append((canonical, int(dimension)))
        tensors.append(inds)
    payload = _normalized_json(tensors).encode("utf-8")
    return sha256(payload).hexdigest()


def _tree_metrics(tree, *, label: str, trials: int, methods: Sequence[str]) -> dict[str, Any]:
    contraction_cost = float(tree.contraction_cost())
    return {
        "label": label,
        "trials": int(trials),
        "methods_sampled": dict(sorted(Counter(methods).items())),
        "num_tensors": int(tree.N),
        "path_length": int(len(tree.get_path())),
        "contraction_width": float(tree.contraction_width()),
        "log10_flops": (
            float(math.log10(contraction_cost)) if contraction_cost > 0 else 0.0
        ),
        "max_size": int(tree.max_size()),
        "peak_size": int(tree.peak_size()),
        "total_write": int(tree.total_write()),
        "combo_cost": float(tree.combo_cost()),
    }


@contextmanager
def _deterministic_python_random(seed: int):
    state = random.getstate()
    random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(state)


def _make_optimizer(
    *,
    methods: Sequence[str],
    objective: str,
    repeats: int,
    max_time_s: float | None,
    seed: int,
    progbar: bool,
):
    if "kahypar" in methods and find_spec("kahypar") is None:
        raise RuntimeError(
            "The requested Cotengra method 'kahypar' is unavailable. "
            "Install the pinned requirements or remove it from "
            "contraction_path_methods."
        )
    return ctg.HyperOptimizer(
        methods=tuple(methods),
        minimize=objective,
        max_repeats=int(repeats),
        max_time=max_time_s,
        parallel=False,
        reconf_opts={"subtree_size": 8, "maxiter": 500},
        optlib="optuna",
        sampler_opts={"seed": int(seed)},
        on_trial_error="warn",
        progbar=progbar,
    )


def _load_cached_paths(
    cache_path: Path,
    metadata: dict[str, Any],
    rehearse_fns: Sequence[RehearseFn],
) -> tuple[tuple[ContractionPath, ...], tuple[dict[str, Any], ...], float] | None:
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            record = yaml.safe_load(fh) or {}
        if record.get("format_version") != PATH_CACHE_FORMAT_VERSION:
            return None
        if record.get("metadata") != metadata:
            return None

        cached_networks = record.get("networks") or []
        if len(cached_networks) != len(rehearse_fns):
            return None

        paths = []
        metrics = []
        for rehearse, cached in zip(rehearse_fns, cached_networks):
            path = _coerce_path(cached["path"])
            info = rehearse(path)
            if _topology_fingerprint(info["tn"]) != cached["topology_fingerprint"]:
                return None
            sampled_methods = tuple(
                method
                for method, count in (cached.get("methods_sampled") or {}).items()
                for _ in range(int(count))
            )
            paths.append(path)
            metrics.append(
                _tree_metrics(
                    info["tree"],
                    label=str(cached["label"]),
                    trials=int(cached.get("trials", 0)),
                    methods=sampled_methods,
                )
            )

        return (
            tuple(paths),
            tuple(metrics),
            float(record.get("search_time_s", 0.0)),
        )
    except Exception:
        # A cache is only an optimization. Any stale or incompatible record
        # should trigger a fresh search rather than aborting the simulation.
        return None


def _write_cache(
    cache_path: Path,
    *,
    metadata: dict[str, Any],
    paths: Sequence[ContractionPath],
    fingerprints: Sequence[str],
    metrics: Sequence[dict[str, Any]],
    search_time_s: float,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "format_version": PATH_CACHE_FORMAT_VERSION,
        "metadata": metadata,
        "search_time_s": float(search_time_s),
        "networks": [
            {
                **dict(metric),
                "topology_fingerprint": fingerprint,
                "path": [list(step) for step in path],
            }
            for path, fingerprint, metric in zip(paths, fingerprints, metrics)
        ],
    }
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(record, fh, sort_keys=False)
        temporary.replace(cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_contraction_paths(
    *,
    kind: str,
    labels: Sequence[str],
    rehearse_fns: Sequence[RehearseFn],
    simplify_sequence: str,
    metadata: dict[str, Any],
    methods: Sequence[str],
    objective: str,
    repeats: int,
    max_time_s: float | None,
    seed: int,
    cache_path: Path | None,
    reuse_saved: bool,
    verbose: bool,
) -> ContractionPathSet:
    """Search, persist, and return one exact path per supplied network.

    Each ``rehearse_fn`` must accept a Quimb/Cotengra ``optimize`` argument
    and return Quimb's rehearsal dictionary containing ``tn`` and ``tree``.
    The selected positional paths are safe to reuse when tensor values change,
    provided the validated simplified topology remains unchanged.
    """
    if len(labels) != len(rehearse_fns):
        raise ValueError("labels and rehearse_fns must have the same length.")

    metadata = json.loads(_normalized_json(metadata))
    memory_key = _memory_key(cache_path, metadata)
    if reuse_saved and memory_key in _MEMORY_CACHE:
        cached = _MEMORY_CACHE[memory_key]
        return ContractionPathSet(
            kind=cached.kind,
            paths=cached.paths,
            simplify_sequence=cached.simplify_sequence,
            metrics=cached.metrics,
            source="memory_cache",
            search_time_s=cached.search_time_s,
            prepare_time_s=0.0,
            cache_path=cached.cache_path,
            metadata=cached.metadata,
        )

    prepare_start = time.perf_counter()
    if reuse_saved and cache_path is not None:
        loaded = _load_cached_paths(cache_path, metadata, rehearse_fns)
        if loaded is not None:
            paths, metrics, search_time_s = loaded
            result = ContractionPathSet(
                kind=kind,
                paths=paths,
                simplify_sequence=simplify_sequence,
                metrics=metrics,
                source="disk_cache",
                search_time_s=search_time_s,
                prepare_time_s=time.perf_counter() - prepare_start,
                cache_path=str(cache_path),
                metadata=metadata,
            )
            _MEMORY_CACHE[memory_key] = result
            if verbose:
                print(f"Loaded {kind} contraction paths from {cache_path}.")
            return result

    paths: list[ContractionPath] = []
    metrics: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    search_start = time.perf_counter()

    for index, (label, rehearse) in enumerate(zip(labels, rehearse_fns)):
        path_seed = int(seed) + index
        if verbose:
            limit = "unlimited" if max_time_s is None else f"{max_time_s:g}s"
            print(
                f"Searching contraction path {label} "
                f"(up to {repeats} trials, {limit}, seed={path_seed})"
            )
        optimizer = _make_optimizer(
            methods=methods,
            objective=objective,
            repeats=repeats,
            max_time_s=max_time_s,
            seed=path_seed,
            progbar=verbose,
        )
        with _deterministic_python_random(path_seed):
            info = rehearse(optimizer)

        tree = info["tree"]
        path = _coerce_path(tree.get_path())
        metric = _tree_metrics(
            tree,
            label=label,
            trials=len(optimizer.scores),
            methods=optimizer.method_choices,
        )
        paths.append(path)
        metrics.append(metric)
        fingerprints.append(_topology_fingerprint(info["tn"]))
        if verbose:
            print(
                f"Selected {label}: width={metric['contraction_width']:.1f}, "
                f"log10(FLOPs)={metric['log10_flops']:.2f}, "
                f"max_size={metric['max_size']}, trials={metric['trials']}"
            )

    search_time_s = time.perf_counter() - search_start
    if verbose:
        print(
            f"Finished {kind} contraction-path search in "
            f"{search_time_s:.2f}s."
        )
    if cache_path is not None:
        _write_cache(
            cache_path,
            metadata=metadata,
            paths=paths,
            fingerprints=fingerprints,
            metrics=metrics,
            search_time_s=search_time_s,
        )

    result = ContractionPathSet(
        kind=kind,
        paths=tuple(paths),
        simplify_sequence=simplify_sequence,
        metrics=tuple(metrics),
        source="searched",
        search_time_s=search_time_s,
        prepare_time_s=time.perf_counter() - prepare_start,
        cache_path=None if cache_path is None else str(cache_path),
        metadata=metadata,
    )
    _MEMORY_CACHE[memory_key] = result
    return result
