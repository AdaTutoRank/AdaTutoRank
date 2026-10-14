"""Model locations and reranker settings from ``agent/configs/``.

    configs/models.yaml        model name -> hub repo + local folder
    configs/rerankers.yaml     reranker name -> backend, vLLM launch flags, workflow overrides
    configs/prompts/*.yaml     prompt of each LLM reranker

Used by the agent code (tokenizer lookups, prompt files) and by serve.sh /
inference.sh, which call this file as a script so they do not import the whole
``dr_agent`` package::

    python dr_agent/run_config.py list
    python dr_agent/run_config.py download adatutorank_8b
    python dr_agent/run_config.py serve adatutorank_8b      # prints the vllm serve command
    python dr_agent/run_config.py overrides adatutorank_8b  # prints --config-overrides

``DR_TULU_CONFIG_DIR`` moves the config folder, ``DR_TULU_MODEL_DIR`` the weights folder.
"""

import argparse
import os
import shlex
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

AGENT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(os.environ.get("DR_TULU_CONFIG_DIR") or AGENT_DIR / "configs")
RERANKER_PROMPT_DIR = CONFIG_DIR / "prompts"

_WEIGHT_FILES = (
    "config.json",
    "adapter_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


@lru_cache(maxsize=None)
def _load(name: str) -> Dict[str, Any]:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def model_dir() -> Path:
    path = Path(os.path.expanduser(os.environ.get("DR_TULU_MODEL_DIR") or _load("models.yaml").get("model_dir", "models")))
    return path if path.is_absolute() else AGENT_DIR / path


def _model_entry(name: str) -> Dict[str, Any]:
    models = _load("models.yaml").get("models") or {}
    if name not in models:
        raise KeyError(f"Unknown model '{name}' in {CONFIG_DIR / 'models.yaml'}. Available: {list(models)}")
    return models[name]


def model_path(name: str, download: bool = False) -> str:
    """Local folder of a model in models.yaml; downloads it there if asked and missing."""
    entry = _model_entry(name)
    local = os.path.expanduser(str(entry.get("local_path") or entry["repo_id"].split("/")[-1]))
    path = Path(local) if os.path.isabs(local) else model_dir() / local
    if download and not any((path / f).exists() for f in _WEIGHT_FILES):
        _download(entry, path)
    return str(path)


def model_ref(name: str) -> str:
    """Local folder of a model if its weights are there, otherwise its hub repo id."""
    path = model_path(name)
    if any((Path(path) / f).exists() for f in _WEIGHT_FILES):
        return path
    return _model_entry(name)["repo_id"]


def _download(entry: Dict[str, Any], path: Path) -> None:
    repo_id, revision = entry["repo_id"], entry.get("revision")
    path.mkdir(parents=True, exist_ok=True)
    print(f"[run_config] downloading {repo_id}{'@' + revision if revision else ''} -> {path}", file=sys.stderr)
    if entry.get("hub", "huggingface") == "modelscope":
        from modelscope import snapshot_download

        snapshot_download(repo_id, revision=revision or "master", local_dir=str(path))
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id, revision=revision, local_dir=str(path), token=os.environ.get("HF_TOKEN"))


def truncation_tokenizer_path() -> Optional[str]:
    """Tokenizer that bounds document length for every reranker (``doc_max_words``)."""
    name = _load("models.yaml").get("truncation_tokenizer")
    return model_ref(name) if name else None


def reranker_prompt_file(filename: str) -> str:
    return str(RERANKER_PROMPT_DIR / filename)


def list_rerankers() -> List[str]:
    return list(_load("rerankers.yaml").get("rerankers") or {})


def reranker_spec(name: str) -> Dict[str, Any]:
    rerankers = _load("rerankers.yaml").get("rerankers") or {}
    if name not in rerankers:
        raise KeyError(f"Unknown reranker '{name}'. Available: {list(rerankers)}")
    return rerankers[name]


def _serve_entry(name: str) -> Dict[str, Any]:
    """vLLM launch settings for an agent (configs/rerankers.yaml `agents`) or a vLLM reranker."""
    agents = _load("rerankers.yaml").get("agents") or {}
    if name in agents:
        return agents[name]
    spec = reranker_spec(name)
    if spec.get("backend") != "vllm":
        raise ValueError(f"'{name}' has backend '{spec.get('backend')}' and needs no vLLM server")
    return spec


def serve_command(name: str) -> List[str]:
    entry = _serve_entry(name)
    serve = entry.get("serve") or {}
    cmd = ["vllm", "serve", model_path(entry["model"], download=True), "--port", str(serve["port"])]
    if serve.get("served_model_name"):
        cmd += ["--served-model-name", serve["served_model_name"]]
    if serve.get("max_model_len"):
        cmd += ["--max-model-len", str(serve["max_model_len"])]
    cmd += ["--dtype", serve.get("dtype", "auto")]
    extra = shlex.split(serve.get("extra_args", ""))
    if extra[:2] == ["--task", "score"] and _vllm_version() >= (0, 20):
        # vLLM 0.20 removed --task; cross-encoders are detected as rerankers on their own.
        extra = extra[2:]
    return cmd + extra


def _vllm_version() -> tuple:
    try:
        from importlib.metadata import version

        return tuple(int(p) for p in version("vllm").split(".")[:2])
    except Exception:
        return (0, 0)


def workflow_overrides(name: str) -> str:
    """``--config-overrides`` for workflows/auto_search_sft.py: agent models + this reranker."""
    agents = _load("rerankers.yaml").get("agents") or {}
    tokenizer = truncation_tokenizer_path()
    values: Dict[str, Any] = {
        "search_agent_model_name": agents["search_agent"]["serve"]["served_model_name"],
        "search_agent_tokenizer_name": tokenizer,
        "browse_agent_model_name": agents["browse_agent"]["serve"]["served_model_name"],
        "browse_agent_tokenizer_name": tokenizer,
    }

    spec = reranker_spec(name)
    backend = spec.get("backend")
    values["use_reranker"] = True
    if backend == "vllm":
        serve = spec["serve"]
        values["reranker_model_name"] = serve["served_model_name"]
        values["reranker_tokenizer_path"] = model_ref(spec["model"])
        values["reranker_api_url"] = f"http://localhost:{serve['port']}{serve.get('api_path', '/v1')}"
    elif backend == "hf":
        values["reranker_model_name"] = model_path(spec["model"], download=True)
        if spec.get("base_model"):
            values["reranker_hf_base_model"] = model_path(spec["base_model"], download=True)
    values.update(spec.get("overrides") or {})
    return ",".join(f"{k}={v}" for k, v in values.items() if v is not None)


def _cmd_list(_args) -> int:
    print(f"{'NAME':<20} {'BACKEND':<8} {'TYPE':<24} MODEL")
    for name in list_rerankers():
        spec = reranker_spec(name)
        rtype = (spec.get("overrides") or {}).get("reranker_type", "")
        print(f"{name:<20} {spec.get('backend', ''):<8} {rtype:<24} {spec.get('model') or '-'}")
    return 0


def _cmd_download(args) -> int:
    agents = _load("rerankers.yaml").get("agents") or {}
    names = [args.name] if args.name else [a["model"] for a in agents.values()]
    for name in names:
        if name in agents:
            name = agents[name]["model"]
        elif name in (_load("rerankers.yaml").get("rerankers") or {}):
            spec = reranker_spec(name)
            for key in ("base_model", "model"):
                if spec.get(key):
                    print(model_path(spec[key], download=True))
            continue
        print(model_path(name, download=True))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list rerankers")
    p = sub.add_parser("download", help="download a model / reranker / agent (default: both agents)")
    p.add_argument("name", nargs="?")
    p = sub.add_parser("serve", help="print the vllm serve command of an agent or reranker")
    p.add_argument("name")
    p = sub.add_parser("port", help="print the port of an agent or vLLM reranker")
    p.add_argument("name")
    p = sub.add_parser("overrides", help="print the workflow --config-overrides of a reranker")
    p.add_argument("name")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        return _cmd_list(args)
    if args.cmd == "download":
        return _cmd_download(args)
    if args.cmd == "serve":
        print(shlex.join(serve_command(args.name)))
    elif args.cmd == "port":
        print(_serve_entry(args.name)["serve"]["port"])
    elif args.cmd == "overrides":
        print(workflow_overrides(args.name))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, ValueError) as e:
        print(f"error: {e.args[0] if e.args else e}", file=sys.stderr)
        sys.exit(1)
