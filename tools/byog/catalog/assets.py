# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Caching AssetProvider with timeout-bounded atomic downloads and SHA256 checks."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from tools.byog.interfaces import AssetProvider, ResolvedAsset
from tools.byog.schemas.execution import PinnedAsset

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "vllm-byog" / "assets"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CachingAssetProvider(AssetProvider):
    """Downloads and SHA256-verifies pinned catalog assets into a local cache."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        overrides: dict[str, Path] | None = None,
        timeout_s: int = 60,
    ):
        env_dir = os.environ.get("BYOG_ASSET_CACHE")
        self._cache = cache_dir or (Path(env_dir) if env_dir else DEFAULT_CACHE_DIR)
        self._overrides = dict(overrides or {})
        self._timeout_s = timeout_s

    def resolve(self, asset: PinnedAsset) -> ResolvedAsset:
        if asset.id in self._overrides:
            p = self._overrides[asset.id]
            return ResolvedAsset(
                asset.id, p, "local-override", _sha256(p), is_override=True
            )

        url = asset.mirror_url or asset.source
        if not url.startswith(("http://", "https://")):
            local = Path(url)
            digest = _sha256(local)
            if digest != asset.sha256:
                raise ValueError(f"SHA256 mismatch for local asset '{asset.id}'.")
            return ResolvedAsset(asset.id, local, asset.revision, digest)

        self._cache.mkdir(parents=True, exist_ok=True)
        dest = self._cache / f"{asset.id}-{asset.sha256[:16]}"
        if dest.is_file() and _sha256(dest) == asset.sha256:
            return ResolvedAsset(asset.id, dest, asset.revision, asset.sha256)

        fd, tmp_name = tempfile.mkstemp(prefix=f".{asset.id}-", dir=str(self._cache))
        tmp = Path(tmp_name)
        try:
            with (
                os.fdopen(fd, "wb") as out_fh,
                urllib.request.urlopen(url, timeout=self._timeout_s) as resp,
            ):
                while chunk := resp.read(1024 * 1024):
                    out_fh.write(chunk)
            digest = _sha256(tmp)
            if digest != asset.sha256:
                raise ValueError(
                    f"SHA256 mismatch for '{asset.id}': expected {asset.sha256}, "
                    f"got {digest}."
                )
            os.replace(tmp, dest)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return ResolvedAsset(asset.id, dest, asset.revision, digest)
