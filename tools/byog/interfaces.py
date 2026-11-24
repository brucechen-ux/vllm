# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plugin, asset-provider, eval-writer, and platform-launcher interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tools.byog.schemas.execution import (
    ErrorKind,
    ExecutionStatus,
    PinnedAsset,
    ProtocolVariant,
    is_safe_id,
)


class CacheResetUnsupportedError(RuntimeError):
    """Raised ONLY when `/reset_prefix_cache` returns HTTP 404/405/501."""


class ServerTransportError(RuntimeError):
    """Raised on connection/5xx/timeout errors talking to the vLLM server."""


@dataclass(frozen=True)
class ResolvedAsset:
    asset_id: str
    path: Path
    resolved_revision: str
    sha256: str
    is_override: bool = False


class AssetProvider(ABC):
    @abstractmethod
    def resolve(self, asset: PinnedAsset) -> ResolvedAsset:
        """Return a verified local path and hash for a pinned catalog asset."""


@dataclass(frozen=True)
class WorkloadPoint:
    point_id: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_safe_id(self.point_id):
            raise ValueError(f"Invalid point_id {self.point_id!r}.")


@dataclass(frozen=True)
class DirectCommandResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False


@dataclass(frozen=True)
class CleanupResult:
    """Typed outcome of stopping a server and verifying device release."""

    terminated_pgid: bool
    vram_reclaimed: bool | None
    baseline_memory_mib: list[int] | None = None
    after_memory_mib: list[int] | None = None


@dataclass
class ServerRunContext:
    """Passed to HTTP server-mode plugins (coordinator owns server lifecycle)."""

    base_url: str
    model: str
    assets: AssetProvider
    workload_assets: dict[str, PinnedAsset]
    output_dir: Path
    timeout_s: int


@dataclass
class DirectRunContext:
    """Passed to direct-mode plugins (runs via launcher with allowlisted env)."""

    serve_config: dict[str, Any]
    serve_env: dict[str, str]
    assets: AssetProvider
    workload_assets: dict[str, PinnedAsset]
    output_dir: Path
    timeout_s: int
    run_command: Callable[[list[str], int | None], DirectCommandResult]


@dataclass
class PointResult:
    status: ExecutionStatus = ExecutionStatus.COMPLETED
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)
    attempted: int | None = None
    completed: int | None = None
    scoring_denominator: int | None = None
    model_errors: int = 0
    infra_errors: int = 0
    error_message: str | None = None
    artifacts: list[Path] = field(default_factory=list)


class EvalRecordWriter(Protocol):
    @property
    def written_count(self) -> int: ...

    def write_response(
        self,
        *,
        dataset_id: str,
        dataset_revision: str,
        sample_id: str,
        question: str,
        reference_answer: str,
        rendered_input: Any,
        point_id: str,
        repeat: int,
        concurrency: int,
        model_response: str,
        extracted_answer: Any,
        score: float | None,
        correct: bool | None,
        error_kind: ErrorKind = ErrorKind.NONE,
        error_message: str | None = None,
        latency_s: float | None = None,
    ) -> None: ...

    def close(self) -> list[Path]: ...


class BenchmarkRunner(ABC):
    @abstractmethod
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]: ...

    @abstractmethod
    def run_point(
        self, point: WorkloadPoint, ctx: ServerRunContext | DirectRunContext
    ) -> PointResult: ...


class EvalRunner(ABC):
    @abstractmethod
    def points(self, params: dict[str, Any]) -> list[WorkloadPoint]: ...

    @abstractmethod
    def run_point(
        self,
        point: WorkloadPoint,
        ctx: ServerRunContext,
        record_writer: EvalRecordWriter,
    ) -> PointResult: ...


@dataclass
class ServerHandle:
    base_url: str
    command: list[str]
    log_path: Path
    pid: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PlatformAdapter(ABC):
    """Hardware-specific server launch, cache reset, cleanup, and direct run."""

    @abstractmethod
    def detect_hardware(self) -> dict[str, Any]: ...

    @abstractmethod
    def start(
        self,
        serve_config: dict[str, Any],
        serve_env: dict[str, str],
        log_dir: Path,
        timeout_s: int,
        *,
        enable_dev_mode: bool = False,
    ) -> ServerHandle: ...

    @abstractmethod
    def reset_cache(self, handle: ServerHandle) -> ProtocolVariant: ...

    @abstractmethod
    def stop(
        self,
        handle: ServerHandle,
        *,
        baseline_memory_mib: list[int] | None = None,
    ) -> CleanupResult: ...

    @abstractmethod
    def run_direct(
        self, cmd: list[str], serve_env: dict[str, str], timeout_s: int
    ) -> DirectCommandResult: ...
