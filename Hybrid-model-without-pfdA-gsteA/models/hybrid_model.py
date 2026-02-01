# models/hybrid_model.py
from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Skip files that are obviously not the model definition
SKIP_PREFIXES = ("Xai", "xai", "predict", "train", "app", "__init__")
SKIP_DIRS = {"models", "scripts", "webapp", ".venv", "__pycache__"}


def _find_model_file() -> Path:
    # Look only in repo root for a python file that mentions the class name
    for py in sorted(REPO_ROOT.glob("*.py")):
        if py.name.startswith(SKIP_PREFIXES):
            continue
        try:
            txt = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "HybridResNet50V2_RViT" in txt:
            return py
    raise FileNotFoundError(
        f"Could not find a model .py in {REPO_ROOT} that defines HybridResNet50V2_RViT."
    )


def _load_module_from_path(py_path: Path):
    mod_name = f"_hybrid_model_{py_path.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_model_file = _find_model_file()
_mod = _load_module_from_path(_model_file)

if not hasattr(_mod, "HybridResNet50V2_RViT"):
    raise AttributeError(
        f"{_model_file.name} loaded, but it does not export HybridResNet50V2_RViT."
    )

HybridResNet50V2_RViT = getattr(_mod, "HybridResNet50V2_RViT")

__all__ = ["HybridResNet50V2_RViT"]
