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
    body = f'''import os, glob
_explicit_path = r"{path}".strip()
if _explicit_path:
    {var} = _explicit_path
    print("📁 Using explicit MACE model path: " + {var})
else:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _model_files = sorted(glob.glob(os.path.join(_script_dir, "*.model")))
    if not _model_files:
        raise FileNotFoundError(
            "No MACE '.model' file found next to this script (" + _script_dir + "). "
            "Place your custom .model file in the same folder as this script, "
            "or set an explicit path in the app before generating it."
        )
    if len(_model_files) > 1:
        print("⚠️ Multiple .model files found; using the first: " + os.path.basename(_model_files[0]))
    {var} = _model_files[0]
    print("📁 Auto-discovered MACE model: " + {var})

if not os.path.exists({var}):
    raise FileNotFoundError("MACE model file not found: " + {var})
'''
    return textwrap.indent(body, indent)


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
