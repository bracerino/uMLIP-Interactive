"""ALIGNN-FF (NIST / atomgptlab) support.

ALIGNN-FF ships an ASE calculator, so it plugs in like every other MLIP here:

    from alignn.ff.ff import AlignnAtomwiseCalculator, get_figshare_model_ff
    calculator = AlignnAtomwiseCalculator(path=get_figshare_model_ff(
        model_name="matpes_pbe"), device="cuda")

``get_figshare_model_ff`` downloads the checkpoint from Figshare on first use
and unpacks it into ~/.cache/atomgptlab/alignn_ff/<model_name>/, then returns that folder.
The calculator wants the *folder* (best_model.pt + config.json), not a file.

Only the two ALIGNN 2.0 MATPES force fields are listed. The paper also names
MPtrj and JV-DFT-DB1/DB2 force fields, but no key for them exists in the
package's all_models_alignn_atomwise.json, so there is nothing to download yet.
"""
import inspect

ALIGNN_FAMILY_NAME = "ALIGNN-FF"
ALIGNN_MODEL_PREFIX = "alignn:"

# The stored value is "alignn:<model_name>", where <model_name> is a key of
# alignn/ff/all_models_alignn_atomwise.json.
ALIGNN_MODELS = {
    # ALIGNN 2.0 force fields, 0.55M parameters each (2 ALIGNN + 2 GCN layers,
    # 128 hidden features, 5 A cutoff). matpes_r2scan is the package default.
    "ALIGNN-FF MATPES-PBE - Materials, 0.55M, 391k configs - recommended ⭐": f"{ALIGNN_MODEL_PREFIX}matpes_pbe",
    "ALIGNN-FF MATPES-r2SCAN - Materials, 0.55M [r2SCAN]": f"{ALIGNN_MODEL_PREFIX}matpes_r2scan",
}

ALIGNN_ENV_SETUP = {
    "pip": (
        "pip install alignn jarvis-tools torch==2.8.0 "
        "ase==3.27.0 pymatgen==2025.10.7 matscipy==1.2.0 "
        "phonopy==2.41.0 numpy pandas matplotlib"
    ),
    "note": (
        "alignn runs on plain PyTorch (DGL is no longer needed) and pulls in "
        "jarvis-tools, so it sits happily in its own environment — locally, "
        "`pip install -r requirements-alignn.txt` does the whole thing. "
        "Checkpoints download from Figshare on first use and are cached in "
        "~/.cache/atomgptlab/alignn_ff/."
    ),
}


# ---------------------------------------------------------------------------
# Model-string helpers
# ---------------------------------------------------------------------------
def is_alignn_model(selected_model_key=None, model_size=None):
    """True when the user picked an ALIGNN-FF checkpoint."""
    if isinstance(model_size, str) and model_size.startswith(ALIGNN_MODEL_PREFIX):
        return True
    if isinstance(selected_model_key, str) and selected_model_key in ALIGNN_MODELS:
        return True
    return False


def alignn_checkpoint_name(model_size):
    """Strip the "alignn:" family prefix off the stored value."""
    if isinstance(model_size, str) and model_size.startswith(ALIGNN_MODEL_PREFIX):
        return model_size[len(ALIGNN_MODEL_PREFIX):]
    return model_size


# ALIGNN hands back forces of shape (3,) for a one-atom cell instead of the
# (natoms, 3) every ASE consumer expects, so a single atom — an isolated-atom
# reference energy, say — raises "invalid index to scalar variable". Reshaping
# the stored result fixes it for optimizers, filters and our own code alike.
# Generated scripts get this same function, copied in by its source.
def _alignn_fix_forces(calculator):
    """ALIGNN returns (3,) forces for a one-atom cell; ASE expects (natoms, 3)."""
    import numpy as _np
    from ase.calculators.calculator import all_changes as _all_changes

    _original = calculator.calculate

    def _calculate(atoms=None, properties=None, system_changes=_all_changes):
        _original(atoms=atoms, properties=properties, system_changes=system_changes)
        _f = calculator.results.get("forces")
        if _f is not None:
            calculator.results["forces"] = _np.asarray(_f).reshape(-1, 3)

    calculator.calculate = _calculate
    return calculator


# ---------------------------------------------------------------------------
# In-app calculator
# ---------------------------------------------------------------------------
def build_alignn_calculator(model_size, device="cpu", log=None):
    """Build the ASE calculator for an ALIGNN-FF checkpoint, downloading if needed."""
    from alignn.ff.ff import AlignnAtomwiseCalculator, get_figshare_model_ff

    model_name = alignn_checkpoint_name(model_size)
    if log:
        log(f"  Checkpoint:     {model_name}")
        log("   (downloads from Figshare on first use, then cached "
            "in ~/.cache/atomgptlab/alignn_ff/)")
    path = get_figshare_model_ff(model_name=model_name)
    if log:
        log(f"  Model folder:   {path}")
    return _alignn_fix_forces(
        AlignnAtomwiseCalculator(path=path, device=device))


# ---------------------------------------------------------------------------
# Generated-script calculator block
# ---------------------------------------------------------------------------
def generate_alignn_calculator_code(model_size, device="cpu", indent=""):
    """Return the `calculator = ...` block for a generated standalone script."""
    model_name = alignn_checkpoint_name(model_size)

    lines = [
        'print("🔧 Initializing ALIGNN-FF calculator...")',
        f'print("🎯 Checkpoint:    {model_name}")',
        f'print("💻 Device:        {device}")',
        '',
        'try:',
        '    from alignn.ff.ff import AlignnAtomwiseCalculator, get_figshare_model_ff',
        'except ImportError as _alignn_ie:',
        '    print(f"❌ alignn is not installed: {_alignn_ie}")',
        '    print("   pip install alignn  (see requirements-alignn.txt)")',
        '    raise',
        '',
        '# Downloads from Figshare on first use and unpacks into',
        '# ~/.cache/atomgptlab/alignn_ff/<model>/; the calculator takes that folder.',
        f'_ALIGNN_PATH = get_figshare_model_ff(model_name="{model_name}")',
        f'_ALIGNN_DEVICE = "{device}"',
        '',
    ]
    lines += inspect.getsource(_alignn_fix_forces).rstrip().split("\n")
    lines += [
        '',
        'try:',
        '    calculator = _alignn_fix_forces(',
        '        AlignnAtomwiseCalculator(path=_ALIGNN_PATH, device=_ALIGNN_DEVICE))',
        f'    print(f"✅ ALIGNN-FF {model_name} initialized on {{_ALIGNN_DEVICE}}")',
        'except Exception as e:',
        f'    print(f"❌ ALIGNN-FF initialization failed on {{_ALIGNN_DEVICE}}: {{e}}")',
        '    if _ALIGNN_DEVICE == "cuda":',
        '        print("⚠️ GPU initialization failed, falling back to CPU...")',
        '        calculator = _alignn_fix_forces(',
        '            AlignnAtomwiseCalculator(path=_ALIGNN_PATH, device="cpu"))',
        '        print("✅ ALIGNN-FF initialized on CPU (fallback)")',
        '    else:',
        '        raise',
    ]
    # Trailing newline on purpose: some templates splice this block straight in
    # front of the next statement, so the last line is terminated here.
    return "\n".join(indent + line if line else "" for line in lines) + "\n"
