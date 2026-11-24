# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Result contracts: `result.json`, eval JSONL, quality checks, and `manifest.json`."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tools.byog.schemas.execution import (
    SAFE_ID_PATTERN,
    SHA256_PATTERN,
    ErrorKind,
    EvidenceSource,
    ExecutionStatus,
    LifecyclePolicy,
    ProtocolVariant,
    QualityThreshold,
    require_finite,
)


class QualityState(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"
    INCONCLUSIVE = "inconclusive"


def compute_compatibility_key(
    components: dict[str, Any], rule_version: str = "1"
) -> str:
    """Explicit helper: canonical SHA256 over rule version and named components."""
    payload = {
        "compatibility_rule_version": rule_version,
        "compatibility_components": components,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _threshold_passes(
    observed: float | None, min_value: float | None, max_value: float | None
) -> bool:
    if observed is None or not math.isfinite(observed):
        return False
    if min_value is not None and observed < min_value:
        return False
    return not (max_value is not None and observed > max_value)


def evaluate_threshold(
    threshold: QualityThreshold, observed: float | None
) -> ThresholdCheckResult:
    """Explicit helper: evaluate one `QualityThreshold` against an observed metric."""
    return ThresholdCheckResult(
        metric=threshold.metric,
        observed=observed,
        min_value=threshold.min_value,
        max_value=threshold.max_value,
        source=threshold.source,
        passed=_threshold_passes(observed, threshold.min_value, threshold.max_value),
    )


class EvalSampleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workload: str = Field(..., min_length=1)
    dataset_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    dataset_revision: str = Field(..., min_length=1)
    sample_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    question: str | None = None
    reference_answer: str | None = None
    withheld_reason: str | None = None


class EvalRenderedInputRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workload: str = Field(..., min_length=1)
    dataset_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    dataset_revision: str = Field(..., min_length=1)
    sample_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    input_hash: str = Field(..., min_length=8)
    rendered_input: Any = None
    withheld_reason: str | None = None


class EvalResponseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    workload: str = Field(..., min_length=1)
    point_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    dataset_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    dataset_revision: str = Field(..., min_length=1)
    sample_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    input_hash: str = Field(..., min_length=8)
    repeat: int = Field(ge=0)
    concurrency: int = Field(ge=1)
    model_response: str | None = None
    extracted_answer: Any = None
    score: float | None = None
    correct: bool | None = None
    error_kind: ErrorKind = ErrorKind.NONE
    error_message: str | None = None
    latency_s: float | None = Field(default=None, ge=0.0)
    withheld_reason: str | None = None


class ThresholdCheckResult(BaseModel):
    """Serialized threshold check outcome (build via `evaluate_threshold`)."""

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(..., min_length=1)
    observed: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    source: Literal["catalog", "spec"] = "catalog"
    passed: bool

    @model_validator(mode="after")
    def _validate(self) -> ThresholdCheckResult:
        require_finite(self.observed, "observed")
        require_finite(self.min_value, "min_value")
        require_finite(self.max_value, "max_value")
        if self.min_value is None and self.max_value is None:
            raise ValueError("ThresholdCheckResult requires min_value or max_value.")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value must be <= max_value.")
        expected = _threshold_passes(self.observed, self.min_value, self.max_value)
        if self.passed != expected:
            raise ValueError(
                f"ThresholdCheckResult.passed={self.passed} != expected {expected}."
            )
        return self


class MeasurementRecord(BaseModel):
    """One benchmark or eval point inside `result.json`."""

    model_config = ConfigDict(extra="forbid")

    workload: str = Field(..., min_length=1)
    kind: Literal["benchmark", "eval"]
    point_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    execution_mode: Literal["server", "direct"]
    protocol_version: str = Field(..., min_length=1)
    requested_lifecycle: LifecyclePolicy
    protocol_variant: ProtocolVariant
    lifecycle_fallback_reason: str | None = None
    evidence_source: EvidenceSource
    required_for_quality: bool = False
    compatibility_rule_version: str = Field(default="1", min_length=1)
    compatibility_components: dict[str, Any] = Field(default_factory=dict)
    compatibility_key: str = Field(..., pattern=SHA256_PATTERN)
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    quality_state: QualityState = QualityState.NOT_EVALUATED
    threshold_checks: list[ThresholdCheckResult] = Field(default_factory=list)
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    attempted: int | None = Field(default=None, ge=0)
    completed: int | None = Field(default=None, ge=0)
    scoring_denominator: int | None = Field(default=None, ge=0)
    model_errors: int = Field(default=0, ge=0)
    infra_errors: int = Field(default=0, ge=0)
    error_message: str | None = None
    artifacts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> MeasurementRecord:
        if self.kind == "eval" and self.execution_mode == "direct":
            raise ValueError("Eval measurements cannot use execution_mode='direct'.")
        if not self.compatibility_components:
            raise ValueError("compatibility_components must be non-empty.")
        expected_key = compute_compatibility_key(
            self.compatibility_components, self.compatibility_rule_version
        )
        if self.compatibility_key != expected_key:
            raise ValueError("compatibility_key does not match components.")

        if self.attempted is not None:
            if self.completed is not None and self.completed > self.attempted:
                raise ValueError("completed cannot exceed attempted.")
            if (
                self.scoring_denominator is not None
                and self.scoring_denominator > self.attempted
            ):
                raise ValueError("scoring_denominator cannot exceed attempted.")
            if self.model_errors + self.infra_errors > self.attempted:
                raise ValueError("errors cannot exceed attempted.")

        if self.execution_mode == "direct":
            if self.protocol_variant != ProtocolVariant.DIRECT:
                raise ValueError(
                    "direct measurements require protocol_variant='direct'."
                )
        elif (
            self.requested_lifecycle.value != self.protocol_variant.value
            and not self.lifecycle_fallback_reason
        ):
            raise ValueError("Lifecycle fallback requires lifecycle_fallback_reason.")

        for check in self.threshold_checks:
            raw = self.metrics.get(check.metric)
            metric_val = (
                float(raw)
                if isinstance(raw, (int, float)) and not isinstance(raw, bool)
                else None
            )
            if check.observed is None:
                if metric_val is not None:
                    raise ValueError(
                        f"ThresholdCheckResult({check.metric}) observed=None "
                        f"contradicts metrics[{check.metric!r}]={metric_val}."
                    )
            elif metric_val is None or not math.isclose(
                check.observed, metric_val, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(
                    f"ThresholdCheckResult({check.metric}) observed={check.observed} "
                    f"contradicts metrics[{check.metric!r}]={raw!r}."
                )

        if self.quality_state == QualityState.PASSED and (
            self.status != ExecutionStatus.COMPLETED
            or self.infra_errors > 0
            or not self.threshold_checks
            or not all(c.passed for c in self.threshold_checks)
            or not any(c.source == "catalog" for c in self.threshold_checks)
        ):
            raise ValueError(
                "quality_state=passed requires completed status, zero infra_errors, "
                "and passing catalog threshold(s)."
            )
        return self


class RunResult(BaseModel):
    """Top-level `result.json` sealed inside `bundle.tar.gz`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    attempt_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    spec_name: str | None = None
    claimed_source_tier: Literal["ci_nightly", "ci_pr", "community"] = "community"
    execution_status: ExecutionStatus
    quality_state: QualityState = QualityState.NOT_EVALUATED
    started_at: datetime
    finished_at: datetime
    duration_s: float = Field(ge=0.0)
    harness_version: str
    vllm_commit: str | None = None
    image_digest: str | None = None
    requested_hardware: dict[str, Any] = Field(default_factory=dict)
    observed_hardware: dict[str, Any] = Field(default_factory=dict)
    resolved_config_hash: str = ""
    measurements: list[MeasurementRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> RunResult:
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("Timestamps must be timezone-aware.")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at.")
        if self.quality_state == QualityState.PASSED:
            req = [m for m in self.measurements if m.required_for_quality]
            if (
                self.execution_status != ExecutionStatus.COMPLETED
                or any(m.status != ExecutionStatus.COMPLETED for m in self.measurements)
                or not req
                or not all(m.quality_state == QualityState.PASSED for m in req)
            ):
                raise ValueError(
                    "RunResult quality_state=passed requires completed execution and "
                    "all required_for_quality measurements passing."
                )
        return self


class ArtifactEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(..., pattern=SHA256_PATTERN)


class BundleManifest(BaseModel):
    """Top-level `manifest.json` inside `bundle.tar.gz`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    job_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    attempt_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    files: list[ArtifactEntry] = Field(default_factory=list)
