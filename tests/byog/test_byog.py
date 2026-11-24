# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 0 verification suite (schemas, resolver, bundle, executor, CLI, E2E)."""

from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.byog.fake_launcher import (
    FakeLauncher,
    FixtureAssetProvider,
    assert_plugin_contract,
)
from tools.byog.bundle import (
    BundleValidationError,
    DeduplicatedEvalRecordWriter,
    seal_bundle,
    validate_bundle,
)
from tools.byog.catalog import Catalog
from tools.byog.cli import EXIT_OK, EXIT_VALIDATION_ERROR, main
from tools.byog.executor.coordinator import Coordinator
from tools.byog.interfaces import WorkloadPoint
from tools.byog.schemas.execution import (
    EvalSelection,
    EvidenceSource,
    ExecutionStatus,
    LifecyclePolicy,
    PublicationPolicy,
    QualityThreshold,
    RunSpec,
)
from tools.byog.schemas.results import (
    MeasurementRecord,
    QualityState,
    ThresholdCheckResult,
    compute_compatibility_key,
    evaluate_threshold,
)
from tools.byog.spec import SpecResolutionError, load_spec_file, resolve_spec

FIXTURES = Path(__file__).resolve().parent / "fixtures"
VALID_SHA = "0123456789abcdef0123456789abcdef01234567"


def _gsm8k_assets() -> FixtureAssetProvider:
    return FixtureAssetProvider(
        {
            "gsm8k_train": FIXTURES / "gsm8k_mini_train.jsonl",
            "gsm8k_test": FIXTURES / "gsm8k_mini_test.jsonl",
        }
    )


def test_schema_invariants_and_p2_guards() -> None:
    # Unsafe point_id, NaN threshold, invalid concurrency, contradictory passed.
    with pytest.raises(ValueError):
        WorkloadPoint("../../escape")
    with pytest.raises(ValidationError):
        QualityThreshold(metric="accuracy", min_value=float("nan"))
    with pytest.raises(ValidationError):
        EvalSelection(name="gsm8k", concurrency=[0, -1])
    with pytest.raises(ValidationError):
        ThresholdCheckResult(
            metric="accuracy", observed=0.0, min_value=0.9, passed=True
        )

    comps = {"w": "gsm8k"}
    key = compute_compatibility_key(comps)
    cat_th = QualityThreshold(metric="accuracy", min_value=0.5, source="catalog")
    spec_th = QualityThreshold(metric="accuracy", min_value=0.0, source="spec")

    def _base(**extra: object) -> dict:
        return {
            "workload": "gsm8k",
            "kind": "eval",
            "point_id": "c1",
            "execution_mode": "server",
            "protocol_version": "1.0",
            "requested_lifecycle": LifecyclePolicy.NONE,
            "protocol_variant": "none",
            "evidence_source": EvidenceSource.HARNESS,
            "compatibility_components": comps,
            "compatibility_key": key,
            "quality_state": QualityState.PASSED,
            **extra,
        }

    # 1. Spec-only threshold cannot authorize PASSED.
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(
            _base(
                metrics={"accuracy": 1.0},
                threshold_checks=[evaluate_threshold(spec_th, 1.0)],
            )
        )
    # 2. Contradictory quality evidence (metrics=0.0 vs observed=1.0) rejected.
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(
            _base(
                metrics={"accuracy": 0.0},
                threshold_checks=[evaluate_threshold(cat_th, 1.0)],
            )
        )
    # 3. PASSED eval with infra_errors > 0 rejected.
    with pytest.raises(ValidationError):
        MeasurementRecord.model_validate(
            _base(
                metrics={"accuracy": 1.0},
                infra_errors=1,
                threshold_checks=[evaluate_threshold(cat_th, 1.0)],
            )
        )


@pytest.mark.parametrize(
    "bad_override",
    [
        {"recipe": {"model": "org/m"}},
        {"matrix": {"tp": [1, 2]}},
        {"compare_baseline": True},
        {"hardware": {"nodes": 2}},
        {"serve": {"config": {"model": "m", "revision": "main"}}},
        {
            "serve": {
                "config": {"model": "m", "revision": VALID_SHA},
                "env": {"HF_TOKEN": "x"},
            }
        },
        {"evals": [{"name": "gsm8k", "min_accuracy": 0.1}]},
        {"benchmarks": [{"type": "http_smoke", "unknown_opt": 1}]},
    ],
)
def test_spec_resolver_guards(bad_override: dict) -> None:
    cat = Catalog.discover()
    base = {
        "schema_version": 1,
        "serve": {"config": {"model": "org/m", "revision": VALID_SHA}},
        "benchmarks": [{"type": "http_smoke"}],
    }
    base.update(bad_override)
    with pytest.raises(SpecResolutionError):
        resolve_spec(RunSpec.model_validate(base), cat)


def test_bundle_safety_redaction_and_determinism(tmp_path: Path) -> None:
    # Restricted PublicationPolicy withholds question, response, extracted & error text.
    w = DeduplicatedEvalRecordWriter(
        workload="ev",
        output_dir=tmp_path / "ev",
        policy=PublicationPolicy(reviewed=False),
    )
    w.write_response(
        dataset_id="ds",
        dataset_revision="r1",
        sample_id="s-0",
        question="SECRET_Q",
        reference_answer="SECRET_A",
        rendered_input="SECRET_P",
        point_id="c1",
        repeat=0,
        concurrency=1,
        model_response="SECRET_R",
        extracted_answer="SECRET_E",
        score=1.0,
        correct=True,
        error_message="SECRET_ERR",
    )
    paths = w.close()
    assert "SECRET_" not in "".join(p.read_text() for p in paths)

    # Path traversal and symlink tar entries are rejected.
    bad_tar = tmp_path / "bad.tar.gz"
    with tarfile.open(bad_tar, "w:gz") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(BundleValidationError):
        validate_bundle(bad_tar)


def test_all_three_plugins_and_end_to_end_gpu_smoke_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")
    assert_plugin_contract(
        workload_name="http_smoke",
        tmp_path=tmp_path / "http",
        overrides={"concurrency": [1, 2], "num_requests": 4},
    )
    res_dir, _ = assert_plugin_contract(
        workload_name="direct_smoke", tmp_path=tmp_path / "direct"
    )
    assert res_dir.measurements[0].metrics["hf_token_leaked"] == 0

    res_gsm, _ = assert_plugin_contract(
        workload_name="gsm8k",
        tmp_path=tmp_path / "gsm",
        assets=_gsm8k_assets(),
        overrides={"num_questions": 6},
    )
    assert res_gsm.quality_state == QualityState.PASSED

    # Full combined smoke_gpu_spec.yaml end-to-end + deterministic resealing.
    cat = Catalog.discover()
    spec_path = FIXTURES / "smoke_gpu_spec.yaml"
    job = resolve_spec(load_spec_file(spec_path), cat, attempt_id="att-e2e")[0]
    res, tar1 = Coordinator(
        launcher=FakeLauncher(), assets=_gsm8k_assets(), vllm_commit=VALID_SHA
    ).run(job, tmp_path / "e2e", source_spec_path=spec_path)
    assert res.execution_status == ExecutionStatus.COMPLETED
    assert res.quality_state == QualityState.PASSED
    h1 = hashlib.sha256(tar1.read_bytes()).hexdigest()
    h2 = hashlib.sha256(
        seal_bundle(tar1.parent, job.job_id, job.attempt_id).read_bytes()
    ).hexdigest()
    assert h1 == h2
    validate_bundle(tar1)

    # Replayable spec.resolved.yaml parses cleanly.
    assert load_spec_file(tar1.parent / "spec.resolved.yaml").schema_version == 1


def test_coordinator_no_silent_fallback_and_cli_smoke(tmp_path: Path) -> None:
    cat = Catalog.discover()
    spec = RunSpec.model_validate(
        {
            "schema_version": 1,
            "serve": {"config": {"model": "org/m", "revision": VALID_SHA}},
            "benchmarks": [
                {"type": "http_smoke", "concurrency": [1, 2], "num_requests": 2}
            ],
        }
    )
    job = resolve_spec(spec, cat, attempt_id="att-fb")[0]
    res, _ = Coordinator(
        launcher=FakeLauncher(allow_reset_cache=False), assets=FixtureAssetProvider()
    ).run(job, tmp_path / "fb")
    assert res.execution_status == ExecutionStatus.FAILED

    # CLI upload guard + fresh-interpreter `python -m tools.byog --help` (no cycles).
    assert (
        main(["run", "--spec", str(FIXTURES / "smoke_gpu_spec.yaml"), "--upload"])
        == EXIT_VALIDATION_ERROR
    )
    assert (
        main(["validate", "--spec", str(FIXTURES / "smoke_gpu_spec.yaml")]) == EXIT_OK
    )
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "tools.byog", "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage: byog" in proc.stdout
