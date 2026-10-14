"""
Model weight resolution: every model lives in its own folder under one model directory.

    from rankify.utils.model_downloader import resolve_model_path

    resolve_model_path("kailinjiang/AdaTutoRank-8B")                       # <model_dir>/AdaTutoRank-8B
    resolve_model_path("BAAI/bge-reranker-large", local_path="/data/bge")   # absolute path, used as is
    resolve_model_path("lwhlwh/rubricranker_sft_rl", hub="modelscope")

If the folder already holds the weights it is loaded directly; otherwise the repo is
downloaded into exactly that folder (no hidden cache), so later runs find it there.
Entries in configs/weights.yaml map one-to-one onto these arguments.
"""

import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# Default model directory; configs/weights.yaml `model_dir` or env RANKIFY_MODEL_DIR override it.
DEFAULT_MODEL_DIR = str(REPO_ROOT / "models")

_WEIGHT_FILES = (
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def target_dir(repo_id: str, local_path: Optional[str] = None, model_dir: Optional[str] = None) -> str:
    """
    Folder for a model: an absolute ``local_path`` as is, a relative one under
    ``model_dir``, and the repo name (``org/name`` -> ``name``) when ``local_path`` is empty.
    """
    model_dir = os.path.expanduser(model_dir or os.environ.get("RANKIFY_MODEL_DIR") or DEFAULT_MODEL_DIR)
    if not os.path.isabs(model_dir):
        model_dir = str(REPO_ROOT / model_dir)
    if local_path:
        local_path = os.path.expanduser(str(local_path))
        return local_path if os.path.isabs(local_path) else os.path.join(model_dir, local_path)
    if not repo_id:
        raise ValueError("Either repo_id or local_path must be given")
    return os.path.join(model_dir, repo_id.rstrip("/").split("/")[-1])


def resolve_model_path(
    repo_id: str,
    local_path: Optional[str] = None,
    hub: str = "huggingface",
    model_dir: Optional[str] = None,
) -> str:
    """
    Return the local folder holding the weights of ``repo_id``, downloading into it if needed.

    Args:
        repo_id: HuggingFace repo name (ModelScope id when hub="modelscope").
        local_path: Folder for the weights; relative paths are taken under ``model_dir``.
            Empty -> ``<model_dir>/<repo name>``.
        hub: "huggingface" (default; retried on ModelScope if it fails) or "modelscope".
        model_dir: Root folder for all models (default: <repo>/models).
    """
    path = target_dir(repo_id, local_path, model_dir)
    if _has_weights(path):
        print(f"[weights] {repo_id or path}: {path}")
        return path
    if not repo_id:
        raise FileNotFoundError(f"No weights in {path} and no repo_id to download from")

    hub = (hub or "huggingface").lower()
    if hub not in ("hf", "huggingface", "ms", "modelscope"):
        raise ValueError(f"Unsupported hub '{hub}', use 'huggingface' or 'modelscope'")

    os.makedirs(path, exist_ok=True)
    if hub in ("ms", "modelscope"):
        return _download_modelscope(repo_id, path)
    try:
        return _download_huggingface(repo_id, path)
    except Exception as e:
        print(f"[weights] HuggingFace download of {repo_id} failed ({e}), trying ModelScope")
        try:
            return _download_modelscope(repo_id, path)
        except Exception as e2:
            raise RuntimeError(
                f"Could not download '{repo_id}' from HuggingFace ({e}) or ModelScope ({e2}). "
                f"Download it manually into {path}."
            ) from e2


def _has_weights(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and any((p / f).exists() for f in _WEIGHT_FILES)


def _download_huggingface(repo_id: str, path: str) -> str:
    from huggingface_hub import snapshot_download

    print(f"[weights] Downloading {repo_id} from HuggingFace -> {path}")
    snapshot_download(repo_id, local_dir=path, token=os.environ.get("HF_TOKEN"))
    return path


def _download_modelscope(repo_id: str, path: str) -> str:
    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise RuntimeError("`pip install modelscope` is required to download from ModelScope") from e

    print(f"[weights] Downloading {repo_id} from ModelScope -> {path}")
    try:
        snapshot_download(repo_id, local_dir=path)
        return path
    except TypeError:
        # modelscope < 1.14 has no local_dir; it stores the model at <cache_dir>/<org>/<name>
        return snapshot_download(repo_id, cache_dir=os.path.dirname(path))
