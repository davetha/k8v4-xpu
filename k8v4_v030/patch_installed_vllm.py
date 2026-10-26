"""Register ``int8_k_int4_v`` in an installed vLLM tree.

The stock image is not edited. The experimental image runs this against
each tree that ``import vllm`` might pick up. Running it twice is a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

CACHE_DTYPE = "int8_k_int4_v"
BACKEND_PATH = "k8v4_v030.backend.Xe2K8V4AttentionBackend"

_CACHE_OLD = '    "nvfp4_4over6",\n]'
_CACHE_NEW = '    "nvfp4_4over6",\n    "int8_k_int4_v",\n]'
_TORCH_OLD = '    "turboquant_k8v4": torch.uint8,\n'
_TORCH_NEW = '    "turboquant_k8v4": torch.uint8,\n    "int8_k_int4_v": torch.uint8,\n'
_XPU_OLD = (
    '        kv_cache_dtype = attn_selector_config.kv_cache_dtype\n'
    '        if kv_cache_dtype is not None and kv_cache_dtype.startswith("turboquant_"):\n'
)
_XPU_NEW = (
    '        kv_cache_dtype = attn_selector_config.kv_cache_dtype\n'
    '        if kv_cache_dtype == "int8_k_int4_v":\n'
    '            logger.info_once("Using Xe2 K8/V4 attention backend.")\n'
    '            return "k8v4_v030.backend.Xe2K8V4AttentionBackend"\n'
    '        if kv_cache_dtype is not None and kv_cache_dtype.startswith("turboquant_"):\n'
)

GPTQ_UTILS_REL = "model_executor/layers/quantization/utils/gptq_utils.py"
IMPORT_ANCHOR = "from typing import TYPE_CHECKING, Any\n"
IMPORT_ANCHOR_NEW = "from typing import TYPE_CHECKING, Any, Final\n"
OLD_DYNAMIC_OVERRIDE = 'def get_dynamic_override(\n    config: AutoGPTQConfig,\n    layer_name: str,\n    key: str | None = None,\n    default_value: int | bool | None = None,\n) -> dict | int | bool | None:\n    for pattern, pattern_dict in config.dynamic.items():\n        # Negative match: matched modules are excluded from quantized init\n        if pattern.startswith("-:"):\n            if re.match(pattern.removeprefix("-:"), layer_name):\n                return False\n        # Positive match: matched modules have quant properties overrides\n        # base quant config\n        elif re.match(pattern.removeprefix("+:"), layer_name):\n            if key is None:\n                return pattern_dict\n            else:\n                return pattern_dict.get(key, default_value)\n    return default_value\n'
NEW_DYNAMIC_OVERRIDE = '_WRAPPER_SWAPS: Final = {\n    "model.language_model.": ("model.", "language_model.model."),\n    "model.": ("model.language_model.", "language_model.model."),\n    "language_model.model.": ("model.language_model.", "model."),\n}\n\ndef _dynamic_name_variants(layer_name: str) -> list[str]:\n    """Return the name forms one module can take across vLLM naming spaces.\n\n    Multimodal-wrapped checkpoints (e.g. Qwen3.5-VL-family text models) name a\n    tensor ``model.language_model.layers.0.self_attn.q_proj`` on disk, the\n    model\'s ``hf_to_vllm_mapper`` rewrites it to ``model.layers.0...`` at load\n    time, while the constructed module tree carries\n    ``language_model.model.layers.0...``. Dynamic quantization patterns are\n    authored against checkpoint names (the GPTQModel convention), but\n    ``get_dynamic_override`` is called with the module-tree name, so a pattern\n    written in any one form silently fails to match in the other two.\n    """\n    variants = [layer_name]\n    for prefix, alternates in _WRAPPER_SWAPS.items():\n        if layer_name.startswith(prefix):\n            rest = layer_name[len(prefix):]\n            variants.extend(alt + rest for alt in alternates)\n            break\n    seen: set[str] = set()\n    unique = [v for v in variants if not (v in seen or seen.add(v))]\n    return unique\n\n\ndef get_dynamic_override(\n    config: AutoGPTQConfig,\n    layer_name: str,\n    key: str | None = None,\n    default_value: int | bool | None = None,\n) -> dict | int | bool | None:\n    variants = _dynamic_name_variants(layer_name)\n    for pattern, pattern_dict in config.dynamic.items():\n        # Negative match: matched modules are excluded from quantized init\n        if pattern.startswith("-:"):\n            if any(re.match(pattern.removeprefix("-:"), v) for v in variants):\n                return False\n        # Positive match: matched modules have quant properties overrides\n        # base quant config\n        elif any(re.match(pattern.removeprefix("+:"), v) for v in variants):\n            if key is None:\n                return pattern_dict\n            else:\n                return pattern_dict.get(key, default_value)\n    return default_value\n'
_DEFAULT_ROOTS = (
    Path("/opt/venv/lib/python3.12/site-packages/vllm"),
    Path("/workspace/vllm/vllm"),
    Path("/workspace/vllm/build/lib.linux-x86_64-cpython-312/vllm"),
)


def _splice(path: Path, old: str, new: str, label: str) -> str:
    text = path.read_text(encoding="utf-8")
    if CACHE_DTYPE in text and (label != "xpu" or BACKEND_PATH in text):
        return "unchanged"
    if old not in text:
        raise RuntimeError("%s is missing the %s anchor" % (path, label))
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return "patched"


def _gptq_dynamic_fix(path: Path) -> str:
    if not path.is_file():
        return "absent"
    text = path.read_text(encoding="utf-8")
    if "_dynamic_name_variants" in text:
        return "unchanged"
    if OLD_DYNAMIC_OVERRIDE not in text:
        raise RuntimeError("%s is missing the gptq dynamic anchor" % path)
    if IMPORT_ANCHOR in text:
        text = text.replace(IMPORT_ANCHOR, IMPORT_ANCHOR_NEW, 1)
    text = text.replace(OLD_DYNAMIC_OVERRIDE, NEW_DYNAMIC_OVERRIDE, 1)
    path.write_text(text, encoding="utf-8")
    return "patched"


def patch_vllm_tree(root: Path) -> list[str]:
    """Patch one vLLM package root. Returns a status word per file."""
    root = Path(root)
    jobs = (
        (root / "config" / "cache.py", _CACHE_OLD, _CACHE_NEW, "cache"),
        (root / "utils" / "torch_utils.py", _TORCH_OLD, _TORCH_NEW, "torch_utils"),
        (root / "platforms" / "xpu.py", _XPU_OLD, _XPU_NEW, "xpu"),
    )
    notes: list[str] = []
    for path, old, new, label in jobs:
        if not path.is_file():
            raise RuntimeError("missing %s" % path)
        notes.append("%s:%s" % (label, _splice(path, old, new, label)))
    notes.append("gptq_dynamic:%s" % _gptq_dynamic_fix(root / GPTQ_UTILS_REL))
    return notes


def patch_installed(roots: tuple[Path, ...] = _DEFAULT_ROOTS) -> list[str]:
    notes: list[str] = []
    found = False
    for root in roots:
        if (root / "config" / "cache.py").is_file():
            found = True
            for item in patch_vllm_tree(root):
                notes.append("%s %s" % (root, item))
    if not found:
        raise RuntimeError("no vLLM tree found")
    return notes


def main(argv: list[str]) -> int:
    roots = tuple(Path(arg) for arg in argv) if argv else _DEFAULT_ROOTS
    for line in patch_installed(roots):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
