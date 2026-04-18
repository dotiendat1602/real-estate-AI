from __future__ import annotations

import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings


def _resolve_local_hf_snapshot(model_name: str) -> str:
    """Use cached HF snapshot path when available to avoid network metadata calls."""
    if not model_name:
        return model_name

    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)

    if "/" not in model_name:
        return model_name

    repo_suffix = f"models--{model_name.replace('/', '--')}"
    cache_roots: list[Path] = []
    for env_var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        env_path = os.getenv(env_var)
        if env_path:
            cache_roots.append(Path(env_path))
    cache_roots.append(Path.home() / ".cache" / "huggingface")

    snapshots_dir: Path | None = None
    for root in cache_roots:
        if root.name == "hub":
            repo_dir = root / repo_suffix
        else:
            repo_dir = root / "hub" / repo_suffix
        candidate = repo_dir / "snapshots"
        if candidate.is_dir():
            snapshots_dir = candidate
            break

    if snapshots_dir is None:
        return model_name

    repo_dir = snapshots_dir.parent

    refs_main = repo_dir / "refs" / "main"
    if refs_main.is_file():
        snapshot_id = refs_main.read_text(encoding="utf-8").strip()
        if snapshot_id:
            snapshot_path = snapshots_dir / snapshot_id
            if snapshot_path.is_dir():
                return str(snapshot_path)

    candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not candidates:
        return model_name

    latest_snapshot = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest_snapshot)


def build_embeddings() -> HuggingFaceEmbeddings:
    model_name = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
    device = os.getenv("EMBED_DEVICE", "cpu")
    resolved_model_name = _resolve_local_hf_snapshot(model_name)
    resolved_model_path = Path(resolved_model_name)

    model_kwargs = {"device": device}
    if resolved_model_path.exists():
        model_kwargs["local_files_only"] = True

    # normalize_embeddings=True để cosine similarity ổn
    return HuggingFaceEmbeddings(
        model_name=resolved_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )
