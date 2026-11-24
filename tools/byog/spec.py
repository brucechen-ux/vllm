# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Box 3: Parse `spec.yaml`, validate Phase 0 rules, and resolve executable jobs."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from tools.byog.catalog import Catalog, CatalogError
from tools.byog.schemas.execution import (
    BenchmarkSelection,
    EvalSelection,
    QualityThreshold,
    ResolvedJob,
    ResolvedWorkload,
    RunSpec,
    WorkloadDefinition,
    WorkloadKind,
)

DENIED_ENV_VARS = frozenset(
    {
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONHOME",
        "PATH",
        "HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HF_ENDPOINT",
    }
)
_SECRET_SUBSTRS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")
_SECRET_TOKENS = frozenset({"KEY", "PAT"})
_HEX_LOWER = frozenset("0123456789abcdef")


class SpecResolutionError(ValueError):
    """Raised when `spec.yaml` cannot be validated or resolved."""


def is_valid_git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(ch in _HEX_LOWER for ch in value)
    )


def validate_serve_env(env: dict[str, str]) -> dict[str, str]:
    blocked = [
        k
        for k in env
        if k.upper() in DENIED_ENV_VARS
        or any(s in k.upper() for s in _SECRET_SUBSTRS)
        or bool(set(k.upper().split("_")) & _SECRET_TOKENS)
    ]
    if blocked:
        raise SpecResolutionError(
            "Disallowed or secret env var(s) in serve.env: "
            f"{', '.join(sorted(blocked))}."
        )
    return dict(env)


def canonicalize_serve_config(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in sorted(config.items()):
        canon = k.replace("_", "-")
        if canon in out:
            raise SpecResolutionError(f"Duplicate serve.config option '{k}'.")
        out[canon] = v
    return out


def load_spec_file(spec_path: Path) -> RunSpec:
    try:
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SpecResolutionError(f"Failed to read spec '{spec_path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise SpecResolutionError(f"Spec '{spec_path}' must be a YAML mapping.")
    try:
        return RunSpec.model_validate(raw)
    except ValidationError as exc:
        raise SpecResolutionError(f"Invalid spec '{spec_path}': {exc}") from exc


def _resolve_workload(
    defn: WorkloadDefinition,
    overrides: dict[str, Any],
    *,
    min_accuracy: float | None = None,
) -> ResolvedWorkload:
    valid_keys = defn.valid_param_keys() | (
        {"repeats", "concurrency"} if defn.kind == WorkloadKind.EVAL else set()
    )
    unknown = sorted(k for k in overrides if k not in valid_keys)
    if unknown:
        raise SpecResolutionError(
            f"Unknown param(s) for '{defn.name}': {', '.join(unknown)}."
        )

    params = {**defn.defaults, **{k: v for k, v in overrides.items() if v is not None}}
    thresholds = list(defn.quality_thresholds)
    if min_accuracy is not None:
        for existing in thresholds:
            if (
                existing.metric == "accuracy"
                and existing.min_value is not None
                and min_accuracy < existing.min_value
            ):
                raise SpecResolutionError(
                    f"Spec min_accuracy={min_accuracy} for '{defn.name}' cannot "
                    f"lower catalog min_value={existing.min_value}."
                )
        thresholds.append(
            QualityThreshold(metric="accuracy", min_value=min_accuracy, source="spec")
        )

    return ResolvedWorkload(
        name=defn.name,
        kind=defn.kind.value,
        execution_mode=defn.execution_mode.value,
        lifecycle=defn.lifecycle.value,
        allow_lifecycle_fallback=defn.allow_lifecycle_fallback,
        evidence_source=defn.evidence_source,
        entrypoint=defn.entrypoint,
        protocol_version=defn.protocol_version,
        assets=dict(defn.assets),
        publication_policy=defn.publication_policy,
        quality_thresholds=thresholds,
        required_for_quality=defn.required_for_quality,
        params=params,
    )


def resolve_spec(
    spec: RunSpec,
    catalog: Catalog,
    *,
    allow_unpinned_revision: bool = False,
    attempt_id: str | None = None,
    claimed_source_tier: str = "community",
) -> list[ResolvedJob]:
    """Resolve `spec.yaml` into executable jobs (Phase 0 emits 1 job: `default`)."""
    if spec.recipe is not None:
        raise SpecResolutionError("`recipe:` is planned in Phase 1 (P1.3).")
    if spec.matrix or spec.matrix_include or spec.matrix_exclude:
        raise SpecResolutionError("Matrix expansion is planned in Phase 1 (P1.4).")
    if spec.compare_baseline:
        raise SpecResolutionError("`compare_baseline` is planned in Phase 1.")
    if spec.hardware.nodes > 1:
        raise SpecResolutionError("Multi-node jobs are not supported in Phase 0.")
    if spec.serve is None:
        raise SpecResolutionError("Spec must define `serve:`.")

    serve_config = dict(spec.serve.config)
    model = serve_config.get("model")
    if not isinstance(model, str) or not model.strip():
        raise SpecResolutionError("`serve.config.model` must be a non-empty string.")
    if not allow_unpinned_revision and not is_valid_git_sha(
        serve_config.get("revision")
    ):
        raise SpecResolutionError(
            "`serve.config.revision` must be a 40-char lowercase hex commit SHA "
            "(or pass --allow-unpinned-revision)."
        )

    canonical_cfg = canonicalize_serve_config(serve_config)
    serve_env = validate_serve_env(spec.serve.env)

    bench_entries: list[BenchmarkSelection] = []
    eval_entries: list[EvalSelection] = []
    try:
        if spec.benchmark_suite:
            bench_entries.extend(
                BenchmarkSelection.model_validate(i)
                for i in catalog.get_suite(spec.benchmark_suite).benchmarks
            )
        if spec.benchmarks:
            bench_entries.extend(spec.benchmarks)
        if spec.eval_suite:
            eval_entries.extend(
                EvalSelection.model_validate(i)
                for i in catalog.get_suite(spec.eval_suite).evals
            )
        if spec.evals:
            eval_entries.extend(spec.evals)
    except (CatalogError, ValidationError) as exc:
        raise SpecResolutionError(str(exc)) from exc

    if not bench_entries and not eval_entries:
        raise SpecResolutionError("Spec must select at least one benchmark or eval.")

    workloads: list[ResolvedWorkload] = []
    seen: set[str] = set()
    for b in bench_entries:
        try:
            defn = catalog.get_workload(b.type)
        except CatalogError as exc:
            raise SpecResolutionError(str(exc)) from exc
        if defn.kind != WorkloadKind.BENCHMARK or defn.name in seen:
            raise SpecResolutionError(f"Invalid or duplicate benchmark '{defn.name}'.")
        seen.add(defn.name)
        workloads.append(_resolve_workload(defn, b.model_dump(exclude={"type"})))

    for e in eval_entries:
        try:
            defn = catalog.get_workload(e.name)
        except CatalogError as exc:
            raise SpecResolutionError(str(exc)) from exc
        if defn.kind != WorkloadKind.EVAL or defn.name in seen:
            raise SpecResolutionError(f"Invalid or duplicate eval '{defn.name}'.")
        seen.add(defn.name)
        overrides = e.model_dump(exclude={"name", "min_accuracy"}, exclude_unset=True)
        workloads.append(
            _resolve_workload(defn, overrides, min_accuracy=e.min_accuracy)
        )

    cfg_hash = hashlib.sha256(
        json.dumps(
            {
                "serve_config": canonical_cfg,
                "serve_env": dict(sorted(serve_env.items())),
                "hardware": spec.hardware.model_dump(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return [
        ResolvedJob(
            job_id="default",
            attempt_id=attempt_id or f"att-{uuid.uuid4().hex[:16]}",
            spec_name=spec.name,
            serve_config=serve_config,
            serve_env=serve_env,
            hardware=spec.hardware,
            workloads=workloads,
            timeout_s=spec.timeout_s,
            claimed_source_tier=claimed_source_tier,  # type: ignore[arg-type]
            resolved_config_hash=cfg_hash,
        )
    ]
