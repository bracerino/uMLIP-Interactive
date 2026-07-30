"""Shared code snippets for resolving custom (local) model files in generated scripts.

Every standalone script the app generates should behave the same way for a local
model: use the explicit path when the user typed one, otherwise look for the
model next to the generated script. That lets a user download the script, drop
it in a folder together with the model, and run it on another machine without
editing any path.
"""

import textwrap


def mace_model_resolution_code(custom_mace_path=None, indent="", var="_mace_model_path"):
    """Return code that resolves a custom MACE ``.model`` file into ``var``.

    Mirrors the SevenNet ``.pth`` behaviour: an explicit path wins, otherwise a
    single ``*.model`` file sitting next to the generated script is used.
    """
    path = (custom_mace_path or "").strip()
    # Aliased imports: this block is pasted inside functions that already use the
    # script-level `os` (e.g. os.environ) before it. A plain `import os` there would
    # make `os` local to the whole function and break those earlier uses with
    # "cannot access local variable 'os'".
    body = f'''import os as _os, glob as _glob
_explicit_path = r"{path}".strip()
if _explicit_path:
    {var} = _explicit_path
    print("📁 Using explicit MACE model path: " + {var})
else:
    _script_dir = _os.path.dirname(_os.path.abspath(__file__))
    _model_files = sorted(_glob.glob(_os.path.join(_script_dir, "*.model")))
    if not _model_files:
        raise FileNotFoundError(
            "No MACE '.model' file found next to this script (" + _script_dir + "). "
            "Place your custom .model file in the same folder as this script, "
            "or set an explicit path in the app before generating it."
        )
    if len(_model_files) > 1:
        print("⚠️ Multiple .model files found; using the first: " + _os.path.basename(_model_files[0]))
    {var} = _model_files[0]
    print("📁 Auto-discovered MACE model: " + {var})

if not _os.path.exists({var}):
    raise FileNotFoundError("MACE model file not found: " + {var})
'''
    return textwrap.indent(body, indent)


def mace_cueq_preamble(enable_cueq=False, device="cpu", indent=""):
    """Return code that decides whether cuEquivariance kernels can be used.

    Emitted only for MACE on CUDA. The generated script stops with an explicit
    message if ``cuequivariance`` is missing, rather than quietly running without
    the acceleration the user asked for.
    """
    if not (enable_cueq and device == "cuda"):
        return ""
    return textwrap.indent('''# cuEquivariance acceleration (same model/weights, faster CUDA kernels).
try:
    import cuequivariance  # noqa: F401
except ImportError:
    print("❌ cuEquivariance acceleration was requested, but the 'cuequivariance' package is not installed.")
    print("   Install it with:")
    print("     pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12")
    print("   ...or re-generate this script with the cuEquivariance option switched off.")
    raise SystemExit(1)
print("⚡ cuEquivariance acceleration enabled")
_enable_cueq = True
''', indent)


def mace_cueq_arg(enable_cueq=False, device="cpu"):
    """The ``mace_mp`` / ``mace_off`` keyword argument, or None when not applicable.

    Only valid next to :func:`mace_cueq_preamble` output, and never on a CPU
    fallback path — cuEquivariance is CUDA-only.
    """
    if enable_cueq and device == "cuda":
        return "enable_cueq=_enable_cueq"
    return None


# cuEquivariance acceleration for SevenNet. SevenNet also supports OpenEquivariance
# and FlashTP, but only cuEquivariance is offered here: it is the one that installs
# with a plain pip command (see requirements-sevennet.txt), while FlashTP has to be
# built from source. https://sevennet.readthedocs.io/en/latest/user_guide/accelerator.html
SEVENNET_CUEQ_MODULE = "cuequivariance"
SEVENNET_CUEQ_INSTALL = [
    "pip install 'sevenn[cueq12]'    # CUDA 12.x",
    "pip install 'sevenn[cueq13]'    # CUDA 13.x",
]


def sevennet_cueq_preamble(enable_cueq=False, device="cpu", indent=""):
    """Return code that checks cuEquivariance is usable before SevenNet starts.

    SevenNet itself only emits a ``UserWarning`` and quietly falls back to e3nn
    when the package is missing, which is easy to miss in a long log. The
    generated script stops instead, the same way the MACE option does.
    """
    if not (enable_cueq and device == "cuda"):
        return ""
    # repr() so command lines containing quotes survive into the generated script.
    hint_lines = "\n".join(f'    print("     " + {cmd!r})' for cmd in SEVENNET_CUEQ_INSTALL)
    body = f'''# cuEquivariance tensor product acceleration (same weights, faster CUDA kernels).
try:
    import {SEVENNET_CUEQ_MODULE}  # noqa: F401
except ImportError:
    print("❌ cuEquivariance acceleration was requested, but the '{SEVENNET_CUEQ_MODULE}' package is not installed.")
    print("   Install it with:")
{hint_lines}
    print("   ...or re-generate this script with the cuEquivariance option switched off.")
    raise SystemExit(1)
print("⚡ cuEquivariance acceleration enabled for SevenNet")
'''
    return textwrap.indent(body, indent)


def sevennet_cueq_arg(enable_cueq=False, device="cpu"):
    """The ``SevenNetCalculator`` keyword argument, or None when not applicable.

    Only valid next to :func:`sevennet_cueq_preamble` output, and never on a CPU
    fallback path — cuEquivariance is CUDA-only.
    """
    if enable_cueq and device == "cuda":
        return "enable_cueq=True"
    return None


def is_custom_mace_model(model_size=None, selected_model_key=None, custom_mace_path=None):
    """True when the user picked the 'Custom MACE Model' entry (path optional).

    The path is optional on purpose — an empty path means "find the .model file
    next to the generated script".
    """
    if custom_mace_path and str(custom_mace_path).strip():
        return True
    if model_size == "custom":
        return True
    return bool(selected_model_key) and "Custom MACE" in str(selected_model_key)
