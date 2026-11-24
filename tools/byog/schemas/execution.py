# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execution contracts: input `spec.yaml`, catalog definitions, and resolved jobs."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SAFE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"
_ID_FIRST = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
_ID_REST = _ID_FIRST | frozenset("._-")


def is_safe_id(value: str) -> bool:
    """Return True iff `value` matches `SAFE_ID_PATTERN`."""
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and value[0] in _ID_FIRST
        and all(ch in _ID_REST for ch in value[1:])
    )


def require_finite(value: float | None, name: str) -> None:
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")


class WorkloadKind(StrEnum):
    BENCHMARK = "benchmark"
    EVAL = "eval"


class ExecutionMode(StrEnum):
    SERVER = "server"
    DIRECT = "direct"


class LifecyclePolicy(StrEnum):
    NONE = "none"
    RESET_CACHE = "reset_cache"
    RESTART_SERVER = "restart_server"


class EvidenceSource(StrEnum):
    HARNESS = "harness"
    CANDIDATE = "candidate"


class ExecutionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class ProtocolVariant(StrEnum):
    NONE = "none"
    DIRECT = "direct"
    RESET_CACHE = "reset_cache"
    RESTART_SERVER = "restart_server"


class ErrorKind(StrEnum):
    NONE = "none"
    MODEL = "model"
    INFRA = "infra"


class PinnedAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(..., pattern=SAFE_ID_PATTERN)
    source: str = Field(..., min_length=1)
    revision: str = Field(..., min_length=1)
    sha256: str = Field(..., pattern=SHA256_PATTERN)
    mirror_url: str | None = None


class PublicationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reviewed: bool = False
    allow_question_text: bool = False
    allow_reference_answer: bool = False
    allow_model_response: bool = False

    def can_publish_question(self) -> bool:
        return self.reviewed and self.allow_question_text

    def can_publish_reference(self) -> bool:
        return self.reviewed and self.allow_reference_answer

    def can_publish_response(self) -> bool:
        return self.reviewed and self.allow_model_response


class QualityThreshold(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str = Field(..., min_length=1)
    min_value: float | None = None
    max_value: float | None = None
    source: Literal["catalog", "spec"] = "catalog"

    @model_validator(mode="after")
    def _validate_bounds(self) -> QualityThreshold:
        require_finite(self.min_value, "min_value")
        require_finite(self.max_value, "max_value")
        if self.min_value is None and self.max_value is None:
            raise ValueError("QualityThreshold requires min_value or max_value.")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value must be <= max_value.")
        return self


class WorkloadDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    name: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]*$")
    kind: WorkloadKind
    description: str = ""
    execution_mode: ExecutionMode
    lifecycle: LifecyclePolicy = LifecyclePolicy.NONE
    allow_lifecycle_fallback: bool = False
    evidence_source: EvidenceSource = EvidenceSource.HARNESS
    platforms: list[Literal["nvidia", "rocm"]] = Field(
        default_factory=lambda: ["nvidia"]
    )
    entrypoint: str = Field(..., pattern=r"^[a-zA-Z0-9_.]+:[a-zA-Z0-9_]+$")
    protocol_version: str = Field(default="1.0", min_length=1)
    assets: dict[str, PinnedAsset] = Field(default_factory=dict)
    publication_policy: PublicationPolicy = Field(default_factory=PublicationPolicy)
    defaults: dict[str, Any] = Field(default_factory=dict)
    allowed_params: list[str] = Field(default_factory=list)
    quality_thresholds: list[QualityThreshold] = Field(default_factory=list)
    required_for_quality: bool = False

    def valid_param_keys(self) -> frozenset[str]:
        return frozenset(self.defaults) | frozenset(self.allowed_params)

    @model_validator(mode="after")
    def _validate(self) -> WorkloadDefinition:
        if (
            self.execution_mode == ExecutionMode.DIRECT
            and self.lifecycle != LifecyclePolicy.NONE
        ):
            raise ValueError("Direct workloads must use lifecycle='none'.")
        if (
            self.kind == WorkloadKind.EVAL
            and self.execution_mode == ExecutionMode.DIRECT
        ):
            raise ValueError("Eval workloads must use execution_mode='server'.")
        if not self.platforms:
            raise ValueError("Workload must declare at least one platform.")
        if any(t.source != "catalog" for t in self.quality_thresholds):
            raise ValueError("Catalog thresholds must have source='catalog'.")
        if self.required_for_quality and not self.quality_thresholds:
            raise ValueError(
                "required_for_quality requires catalog quality_thresholds."
            )
        return self


class SuiteDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    name: str = Field(..., pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""
    benchmarks: list[dict[str, Any]] = Field(default_factory=list)
    evals: list[dict[str, Any]] = Field(default_factory=list)


class ServeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


class HardwareSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu: list[str] = Field(default_factory=list)
    count: int = Field(default=1, ge=1)
    nodes: int = Field(default=1, ge=1)


class BenchmarkSelection(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(..., min_length=1)


class EvalSelection(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(..., min_length=1)
    repeats: int | None = Field(default=None, ge=1)
    concurrency: list[int] | None = None
    min_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate(self) -> EvalSelection:
        require_finite(self.min_accuracy, "min_accuracy")
        if self.concurrency is not None and (
            not self.concurrency or any(c < 1 for c in self.concurrency)
        ):
            raise ValueError("concurrency must be a non-empty list of positive ints.")
        return self


class RunSpec(BaseModel):
    """Top-level user input `spec.yaml` (`schema_version: 1`)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    name: str | None = None
    recipe: dict[str, Any] | None = None
    serve: ServeSpec | None = None
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    matrix: dict[str, list[Any]] | None = None
    matrix_include: list[dict[str, Any]] | None = None
    matrix_exclude: list[dict[str, Any]] | None = None
    benchmark_suite: str | None = None
    benchmarks: list[BenchmarkSelection] | None = None
    eval_suite: str | None = None
    evals: list[EvalSelection] | None = None
    compare_baseline: bool = False
    timeout_s: int = Field(default=3600, ge=1)

    @model_validator(mode="after")
    def _check_source(self) -> RunSpec:
        if self.recipe is None and self.serve is None:
            raise ValueError("Spec must specify 'recipe' or 'serve'.")
        return self


class ResolvedWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["benchmark", "eval"]
    execution_mode: Literal["server", "direct"]
    lifecycle: Literal["none", "reset_cache", "restart_server"]
    allow_lifecycle_fallback: bool = False
    evidence_source: EvidenceSource
    entrypoint: str
    protocol_version: str
    assets: dict[str, PinnedAsset] = Field(default_factory=dict)
    publication_policy: PublicationPolicy = Field(default_factory=PublicationPolicy)
    quality_thresholds: list[QualityThreshold] = Field(default_factory=list)
    required_for_quality: bool = False
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> ResolvedWorkload:
        if self.execution_mode == "direct" and self.lifecycle != "none":
            raise ValueError("Direct workloads must use lifecycle='none'.")
        if self.kind == "eval" and self.execution_mode == "direct":
            raise ValueError("Eval workloads must use execution_mode='server'.")
        if self.required_for_quality and not any(
            t.source == "catalog" for t in self.quality_thresholds
        ):
            raise ValueError("required_for_quality requires a catalog threshold.")
        return self


class ResolvedJob(BaseModel):
    """Concrete job passed from Box 3 (Spec resolver) to Box 4 (Executor)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str = Field(default="default", pattern=SAFE_ID_PATTERN)
    attempt_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    spec_name: str | None = None
    serve_config: dict[str, Any] = Field(default_factory=dict)
    serve_env: dict[str, str] = Field(default_factory=dict)
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    workloads: list[ResolvedWorkload] = Field(default_factory=list)
    timeout_s: int = Field(default=3600, ge=1)
    claimed_source_tier: Literal["ci_nightly", "ci_pr", "community"] = "community"
    resolved_config_hash: str = ""
