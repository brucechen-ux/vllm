# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Box 4 (Engine): Coordinator owning server lifecycle, cleanup, and quality gates."""

from __future__ import annotations

import contextlib
import json
import signal
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.byog.bundle import (
    DeduplicatedEvalRecordWriter,
    redact_env,
    redact_serve_config,
    seal_bundle,
    write_bundle_metadata,
)
from tools.byog.catalog import instantiate_plugin
from tools.byog.interfaces import (
    AssetProvider,
    BenchmarkRunner,
    CacheResetUnsupportedError,
    DirectCommandResult,
    DirectRunContext,
    EvalRunner,
    PlatformAdapter,
    PointResult,
    ServerHandle,
    ServerRunContext,
    ServerTransportError,
    WorkloadPoint,
)
from tools.byog.schemas.execution import (
    ExecutionStatus,
    LifecyclePolicy,
    ProtocolVariant,
    QualityThreshold,
    ResolvedJob,
    ResolvedWorkload,
    WorkloadKind,
)
from tools.byog.schemas.results import (
    MeasurementRecord,
    QualityState,
    RunResult,
    ThresholdCheckResult,
    compute_compatibility_key,
    evaluate_threshold,
)
from tools.byog.spec import canonicalize_serve_config

HARNESS_VERSION = "0.1.0-phase0"
COMPATIBILITY_RULE_VERSION = "1"
_NON_FUNCTIONAL_ENV = frozenset(
    {"VLLM_LOGGING_LEVEL", "VLLM_LOG_STATS_INTERVAL", "VLLM_NO_USAGE_STATS"}
)


class _JobCancelled(BaseException):
    pass


def _compatibility(
    job: ResolvedJob,
    w: ResolvedWorkload,
    pt: WorkloadPoint,
    variant: ProtocolVariant,
    hw: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    env = {
        k: v
        for k, v in redact_env(job.serve_env, allowlist_only=False).items()
        if k.upper() not in _NON_FUNCTIONAL_ENV
    }
    comps: dict[str, Any] = {
        "workload": w.name,
        "kind": w.kind,
        "point_id": pt.point_id,
        "point_params": pt.params,
        "execution_mode": w.execution_mode,
        "protocol_version": w.protocol_version,
        "protocol_variant": variant.value,
        "evidence_source": w.evidence_source.value,
        "serve_config": canonicalize_serve_config(
            redact_serve_config(job.serve_config)
        ),
        "serve_env": env,
        "hardware": {
            k: hw.get(k) for k in ("platform", "gpu_id", "gpu_count", "driver_version")
        },
        "assets": {
            aid: {"revision": a.revision, "sha256": a.sha256}
            for aid, a in sorted(w.assets.items())
        },
    }
    if w.kind == "eval":
        comps["eval_settings"] = {
            k: pt.params[k]
            for k in sorted(pt.params)
            if k
            in (
                "temperature",
                "max_tokens",
                "num_shots",
                "num_questions",
                "repeats",
                "concurrency",
            )
        }
    return comps, compute_compatibility_key(comps, COMPATIBILITY_RULE_VERSION)


def _evaluate_thresholds(
    thresholds: list[QualityThreshold],
    metrics: dict[str, float | int | str | None],
) -> tuple[QualityState, list[ThresholdCheckResult]]:
    if not thresholds:
        return QualityState.INCONCLUSIVE, []
    checks: list[ThresholdCheckResult] = []
    any_missing = False
    for th in thresholds:
        raw = metrics.get(th.metric)
        obs = (
            float(raw)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool)
            else None
        )
        if obs is None:
            any_missing = True
        checks.append(evaluate_threshold(th, obs))
    if any_missing:
        return QualityState.INCONCLUSIVE, checks
    if not all(c.passed for c in checks):
        return QualityState.FAILED, checks
    if not any(c.source == "catalog" for c in checks):
        return QualityState.INCONCLUSIVE, checks
    return QualityState.PASSED, checks


def _aggregate_quality(
    exec_status: ExecutionStatus, measurements: list[MeasurementRecord]
) -> QualityState:
    req = [m for m in measurements if m.required_for_quality]
    if not req:
        evals = [m for m in measurements if m.kind == "eval"]
        if not evals:
            return QualityState.NOT_EVALUATED
        if any(m.quality_state == QualityState.FAILED for m in evals):
            return QualityState.FAILED
        return QualityState.INCONCLUSIVE
    if any(m.quality_state == QualityState.FAILED for m in req):
        return QualityState.FAILED
    if exec_status != ExecutionStatus.COMPLETED or any(
        m.status != ExecutionStatus.COMPLETED for m in measurements
    ):
        return QualityState.INCONCLUSIVE
    if any(m.quality_state != QualityState.PASSED for m in req):
        return QualityState.INCONCLUSIVE
    return QualityState.PASSED


class Coordinator:
    """Runs one `ResolvedJob`, owns server lifecycle, and seals `bundle.tar.gz`."""

    def __init__(
        self,
        *,
        launcher: PlatformAdapter,
        assets: AssetProvider,
        image_ref: str | None = None,
        vllm_commit: str | None = None,
        image_digest: str | None = None,
    ):
        self._launcher = launcher
        self._assets = assets
        self._image_ref = image_ref
        self._vllm_commit = vllm_commit
        self._image_digest = image_digest

    def run(
        self,
        job: ResolvedJob,
        output_root: Path,
        *,
        source_spec_path: Path | None = None,
    ) -> tuple[RunResult, Path]:
        bundle_dir = output_root / job.job_id / job.attempt_id
        logs_dir = bundle_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        started_at = datetime.now(UTC)
        t0 = time.monotonic()
        deadline = t0 + job.timeout_s
        hw = self._launcher.detect_hardware()
        base_mem = hw.get("device_memory_used_mib")
        base_mem_list = list(base_mem) if isinstance(base_mem, list) else None

        measurements: list[MeasurementRecord] = []
        errors: list[str] = []
        cleanups: list[dict[str, Any]] = []
        status = ExecutionStatus.COMPLETED
        active: ServerHandle | None = None
        active_dev = False

        def _rem_s() -> int:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError(f"Job exceeded timeout_s={job.timeout_s}s.")
            return max(1, int(rem))

        def _stop_server() -> bool:
            nonlocal active, active_dev
            if active is None:
                return True
            h, active, active_dev = active, None, False
            rep = self._launcher.stop(h, baseline_memory_mib=base_mem_list)
            cleanups.append(asdict(rep))
            ok = True
            if not rep.terminated_pgid:
                errors.append("Server process group failed to terminate.")
                ok = False
            if rep.vram_reclaimed is False:
                errors.append("GPU VRAM was not reclaimed to baseline.")
                ok = False
            return ok

        def _ensure_server(dev_mode: bool) -> ServerHandle:
            nonlocal active, active_dev
            if active is not None and active_dev == dev_mode:
                return active
            if active is not None and not _stop_server():
                raise RuntimeError("Aborting launch after failed server cleanup.")
            active = self._launcher.start(
                job.serve_config,
                job.serve_env,
                logs_dir,
                _rem_s(),
                enable_dev_mode=dev_mode,
            )
            active_dev = dev_mode
            return active

        def _record(
            w: ResolvedWorkload,
            pt: WorkloadPoint,
            variant: ProtocolVariant,
            pr: PointResult,
            *,
            q_state: QualityState = QualityState.NOT_EVALUATED,
            checks: list[ThresholdCheckResult] | None = None,
            fallback_reason: str | None = None,
        ) -> None:
            comps, key = _compatibility(job, w, pt, variant, hw)
            rels = []
            root = bundle_dir.resolve()
            for p in pr.artifacts:
                try:
                    rels.append(p.resolve().relative_to(root).as_posix())
                except ValueError:
                    rels.append(p.name)
            measurements.append(
                MeasurementRecord(
                    workload=w.name,
                    kind=w.kind,
                    point_id=pt.point_id,
                    execution_mode=w.execution_mode,
                    protocol_version=w.protocol_version,
                    requested_lifecycle=LifecyclePolicy(w.lifecycle),
                    protocol_variant=variant,
                    lifecycle_fallback_reason=fallback_reason,
                    evidence_source=w.evidence_source,
                    required_for_quality=w.required_for_quality,
                    compatibility_rule_version=COMPATIBILITY_RULE_VERSION,
                    compatibility_components=comps,
                    compatibility_key=key,
                    status=pr.status,
                    quality_state=q_state,
                    threshold_checks=checks or [],
                    metrics=pr.metrics,
                    attempted=pr.attempted,
                    completed=pr.completed,
                    scoring_denominator=pr.scoring_denominator,
                    model_errors=pr.model_errors,
                    infra_errors=pr.infra_errors,
                    error_message=pr.error_message,
                    artifacts=rels,
                )
            )

        prev_sig = None
        if threading.current_thread() is threading.main_thread():
            prev_sig = signal.getsignal(signal.SIGTERM)
            with contextlib.suppress(ValueError, OSError):
                signal.signal(
                    signal.SIGTERM,
                    lambda s, _f: (_ for _ in ()).throw(_JobCancelled(f"Signal {s}")),
                )

        try:
            obs_gpu = hw.get("gpu_id")
            if (
                job.hardware.gpu
                and isinstance(obs_gpu, str)
                and obs_gpu != "unknown"
                and obs_gpu not in job.hardware.gpu
            ):
                raise RuntimeError(
                    f"Observed GPU '{obs_gpu}' not in {job.hardware.gpu}."
                )

            served_model = str(
                job.serve_config.get("served-model-name")
                or job.serve_config.get("served_model_name")
                or job.serve_config.get("model", "")
            )

            for w in job.workloads:
                plugin = instantiate_plugin(w.entrypoint, WorkloadKind(w.kind))
                pts = plugin.points(w.params)
                w_out = (
                    bundle_dir
                    / ("evals" if w.kind == "eval" else "benchmarks")
                    / w.name
                )
                w_out.mkdir(parents=True, exist_ok=True)

                if w.execution_mode == "direct":
                    if not _stop_server():
                        break
                    assert isinstance(plugin, BenchmarkRunner)
                    for pt in pts:
                        rem = _rem_s()

                        def _run_cmd(
                            cmd: list[str], t: int | None = None
                        ) -> DirectCommandResult:
                            b = _rem_s()
                            return self._launcher.run_direct(
                                cmd, job.serve_env, min(b, t) if t else b
                            )

                        ctx = DirectRunContext(
                            serve_config=dict(job.serve_config),
                            serve_env=dict(job.serve_env),
                            assets=self._assets,
                            workload_assets=dict(w.assets),
                            output_dir=w_out,
                            timeout_s=rem,
                            run_command=_run_cmd,
                        )
                        try:
                            pr = plugin.run_point(pt, ctx)
                        except Exception as exc:  # noqa: BLE001
                            pr = PointResult(
                                status=ExecutionStatus.FAILED,
                                error_message=f"{type(exc).__name__}: {exc}",
                            )
                        _record(w, pt, ProtocolVariant.DIRECT, pr)
                    continue

                req_lc = LifecyclePolicy(w.lifecycle)
                dev_mode = req_lc == LifecyclePolicy.RESET_CACHE
                writer = (
                    DeduplicatedEvalRecordWriter(
                        workload=w.name,
                        output_dir=w_out,
                        policy=w.publication_policy,
                    )
                    if w.kind == "eval"
                    else None
                )
                try:
                    for idx, pt in enumerate(pts):
                        variant = ProtocolVariant(req_lc.value)
                        fb_reason: str | None = None
                        if idx == 0:
                            handle = _ensure_server(dev_mode)
                        elif req_lc == LifecyclePolicy.RESTART_SERVER:
                            if not _stop_server():
                                break
                            handle = _ensure_server(dev_mode)
                        elif req_lc == LifecyclePolicy.RESET_CACHE:
                            handle = _ensure_server(dev_mode)
                            try:
                                variant = self._launcher.reset_cache(handle)
                            except CacheResetUnsupportedError as exc:
                                if not w.allow_lifecycle_fallback:
                                    msg = f"reset_cache unsupported: {exc}"
                                    errors.append(msg)
                                    _record(
                                        w,
                                        pt,
                                        variant,
                                        PointResult(
                                            status=ExecutionStatus.FAILED,
                                            error_message=msg,
                                        ),
                                        q_state=(
                                            QualityState.INCONCLUSIVE
                                            if w.kind == "eval"
                                            else QualityState.NOT_EVALUATED
                                        ),
                                    )
                                    break
                                fb_reason = str(exc)
                                if not _stop_server():
                                    break
                                handle = _ensure_server(False)
                                variant = ProtocolVariant.RESTART_SERVER
                            except ServerTransportError as exc:
                                msg = f"reset_cache transport error: {exc}"
                                errors.append(msg)
                                _record(
                                    w,
                                    pt,
                                    variant,
                                    PointResult(
                                        status=ExecutionStatus.FAILED,
                                        error_message=msg,
                                    ),
                                    q_state=(
                                        QualityState.INCONCLUSIVE
                                        if w.kind == "eval"
                                        else QualityState.NOT_EVALUATED
                                    ),
                                )
                                break
                        else:
                            handle = _ensure_server(dev_mode)

                        s_ctx = ServerRunContext(
                            base_url=handle.base_url,
                            model=served_model,
                            assets=self._assets,
                            workload_assets=dict(w.assets),
                            output_dir=w_out,
                            timeout_s=_rem_s(),
                        )
                        try:
                            if w.kind == "eval":
                                assert isinstance(plugin, EvalRunner)
                                assert writer is not None
                                pr = plugin.run_point(pt, s_ctx, writer)
                            else:
                                assert isinstance(plugin, BenchmarkRunner)
                                pr = plugin.run_point(pt, s_ctx)
                        except Exception as exc:  # noqa: BLE001
                            pr = PointResult(
                                status=ExecutionStatus.FAILED,
                                error_message=f"{type(exc).__name__}: {exc}",
                            )

                        q_state = QualityState.NOT_EVALUATED
                        checks: list[ThresholdCheckResult] = []
                        if w.kind == "eval":
                            if (
                                pr.status != ExecutionStatus.COMPLETED
                                or pr.infra_errors > 0
                                or not pr.scoring_denominator
                            ):
                                q_state = QualityState.INCONCLUSIVE
                            else:
                                q_state, checks = _evaluate_thresholds(
                                    w.quality_thresholds, pr.metrics
                                )
                        _record(
                            w,
                            pt,
                            variant,
                            pr,
                            q_state=q_state,
                            checks=checks,
                            fallback_reason=fb_reason,
                        )
                finally:
                    if writer is not None:
                        rec_paths = writer.close()
                        root = bundle_dir.resolve()
                        rels = [
                            p.resolve().relative_to(root).as_posix() for p in rec_paths
                        ]
                        for m in measurements:
                            if m.workload == w.name:
                                m.artifacts.extend(
                                    r for r in rels if r not in m.artifacts
                                )

            if errors or any(
                m.status != ExecutionStatus.COMPLETED for m in measurements
            ):
                status = ExecutionStatus.FAILED
        except _JobCancelled as exc:
            errors.append(str(exc))
            status = ExecutionStatus.CANCELLED
        except TimeoutError as exc:
            errors.append(str(exc))
            status = ExecutionStatus.TIMED_OUT
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            status = ExecutionStatus.FAILED
        finally:
            if not _stop_server() and status == ExecutionStatus.COMPLETED:
                status = ExecutionStatus.FAILED
            if prev_sig is not None:
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(signal.SIGTERM, prev_sig)

        if cleanups:
            (logs_dir / "cleanup.json").write_text(
                json.dumps(cleanups, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        result = RunResult(
            job_id=job.job_id,
            attempt_id=job.attempt_id,
            spec_name=job.spec_name,
            claimed_source_tier=job.claimed_source_tier,
            execution_status=status,
            quality_state=_aggregate_quality(status, measurements),
            started_at=started_at,
            finished_at=datetime.now(UTC),
            duration_s=max(0.0, time.monotonic() - t0),
            harness_version=HARNESS_VERSION,
            vllm_commit=self._vllm_commit,
            image_digest=self._image_digest,
            requested_hardware=job.hardware.model_dump(),
            observed_hardware=hw,
            resolved_config_hash=job.resolved_config_hash,
            measurements=measurements,
            errors=errors,
        )
        write_bundle_metadata(
            bundle_dir,
            job=job,
            result=result,
            source_spec_path=source_spec_path,
            image_ref=self._image_ref,
        )
        return result, seal_bundle(bundle_dir, job.job_id, job.attempt_id)
