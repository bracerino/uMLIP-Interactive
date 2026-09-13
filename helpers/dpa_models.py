"""DPA (DeePMD-kit / DeepModeling) support.

The DPA family are the "large atomic models" from DeepModeling, run through
DeePMD-kit's ASE calculator (``deepmd.calculator.DP``). Three generations are
offered here:

  * **DPA-2** — the original attention-based descriptor (legacy, kept because
    it is still the reference in a lot of published work).
  * **DPA-3** — message-passing LAM trained multi-task on the OpenLAM
    datasets. One checkpoint carries a shared backbone plus one fitting net
    ("model branch") per training dataset, so a branch has to be picked; it
    also fixes the DFT reference, exactly like a UMA task does.
  * **DPA4 / SeZM** — the SO(3)-equivariant successor. Released under
    CC-BY-NC-4.0, i.e. **non-commercial use only**, unlike DPA-2/DPA-3 which
    are CC-BY-4.0.

Two things make this family different from the others here and shape the code
below:

  * **The branch is part of the model choice.** Rather than a second dropdown,
    each (checkpoint, branch) pair is its own entry, so the value stored in
    MODEL_FAMILIES carries everything needed to rebuild the calculator and
    every script generator stays stateless. See DPA_MODEL_PREFIX.
  * **DeePMD-kit needs its own environment.** The PyPI wheel is compiled
    against a specific torch (3.2.0 → torch 2.11) and refuses to load against
    another one, so it cannot share the app's torch 2.8 environment. See
    DPA_ENV_SETUP and requirements-dpa.txt.

Weights come from the ``deepmodelingcommunity`` Hugging Face org and are *not*
gated — no token, unlike UMA.
"""

import os
from pathlib import Path

DPA_FAMILY_NAME = "DPA (DeePMD-kit)"
DPA_MODEL_PREFIX = "dpa:"

# Where downloaded checkpoints are kept. Deliberately NOT the plain
# huggingface_hub cache: that stores every file as a symlink into a blob with
# no file extension, and DeePMD-kit picks its backend from the *suffix* after
# resolving symlinks ("Cannot detect the backend of the model file ..."). A
# local_dir download writes a real file that keeps its .pt / .pth name.
DPA_CACHE_DIRNAME = "dpa_models"

# ---------------------------------------------------------------------------
# Checkpoints
#
# The stored value is
#     dpa:<hf_repo_id>|<filename>|<model_branch>|<charge_spin_key>
# where the last two may be empty:
#   * model_branch is empty for single-task checkpoints (DPA4);
#   * charge_spin_key names the atoms.info entry that carries
#     [total_charge, spin_multiplicity] for the checkpoints that were trained
#     with that conditioning. The spelling genuinely differs between
#     checkpoints -- DPA-3.2 reads "fparam", DPA-3.3 and DPA4 read
#     "charge_spin" -- and passing the wrong one either errors or is silently
#     ignored, so it is recorded per model rather than guessed.
#
# Branch names are the exact keys inside each checkpoint (read back with
# `dp --pt show <model>.pt model-branch`). They are NOT consistent across
# generations -- DPA-3.2 spells it "OMat24" and "Alloy_APEX", DPA-3.3 spells
# the same two "Omat24" and "Alloy_tongqi", DPA-3.1/DPA-2 call MPtrj
# "MP_traj_v024_alldata_mixu" -- so they cannot be shared between entries.
# ---------------------------------------------------------------------------
_R33 = "deepmodelingcommunity/DPA-3.3-1M|DPA-3.3-1M.pt"
_R32 = "deepmodelingcommunity/DPA-3.2-5M|DPA-3.2-5M.pt"
_R31 = "deepmodelingcommunity/DPA-3.1-3M|DPA-3.1-3M.pt"
_R24 = "deepmodelingcommunity/DPA-2.4-7M|DPA-2.4-7M-patched-mt.pt"
_R4M = "deepmodelingcommunity/DPA4-OMat24|DPA4-%s-OMat24-v20260805.pt"
_R4O = "deepmodelingcommunity/DPA4-OMol|DPA4-%s-OMol25-100M-v20260820.pt"
# DPA4C: the compact, compressible member of the DPA4 family. Each local
# environment is read once (no message passing) and the radial functions are
# tabulated splines evaluated by fused CUDA kernels, so it trades a little
# accuracy for throughput in large MD. Same charge/spin conditioning and the
# same CC-BY-NC-4.0 licence as DPA4. Only the OMol25 line has weights: the
# DPA4C-OMat repo exists but is an empty placeholder as of 2026-09-13.
_R4CO = "deepmodelingcommunity/DPA4C-OMol|DPA4C-%s-OMol25-100M-v20260820.pt"

DPA_MODELS = {
    # Ordered by reported accuracy, most accurate first, newest generation
    # first. Figures are the numbers the model cards publish, so they are only
    # comparable *within* a block: DPA4 quotes OMat24 validation MAE
    # (meV/atom, meV/A), DPA-3.x quotes the LAMBench property error (lower is
    # better). There is no common benchmark across the two, so the blocks are
    # ordered by generation and the numbers are given for what they are.
    #
    # --- DPA4 / SeZM on OMat24 (v20260805, newest generation, CC-BY-NC-4.0).
    #     OMat24 validation MAE, most accurate first. The card recommends Mini
    #     and Neo for general use, Air or Plus when accuracy matters more.
    "DPA4-Plus-OMat24 - Materials, 8.8M, E 10.0 meV/at, F 47 - most accurate ⭐ [non-commercial]": f"{DPA_MODEL_PREFIX}{_R4M % 'Plus'}||",
    "DPA4-Air-OMat24 - Materials, 5.1M, E 10.7 meV/at, F 52 [non-commercial]": f"{DPA_MODEL_PREFIX}{_R4M % 'Air'}||",
    "DPA4-Neo-OMat24 - Materials, 1.1M, E 12.1 meV/at, F 59 - recommended ⭐ [non-commercial]": f"{DPA_MODEL_PREFIX}{_R4M % 'Neo'}||",
    "DPA4-Mini-OMat24 - Materials, 0.7M, E 14.0 meV/at, F 70 - recommended [non-commercial]": f"{DPA_MODEL_PREFIX}{_R4M % 'Mini'}||",
    "DPA4-Nano-OMat24 - Materials, 0.5M, E 18.7 meV/at, F 96 - fastest [non-commercial]": f"{DPA_MODEL_PREFIX}{_R4M % 'Nano'}||",
    # --- DPA-3.3-1M: 6 layers, OpenLAM v2. Best DPA-3 on LAMBench (0.238) and
    #     the most accurate CC-BY-4.0 model here. The card's default branch is
    #     Omat24, which it reports as strong across materials, catalysis and
    #     molecules. Note DPA-3.2 still edges it on the molecular torsion sets.
    "DPA-3.3-1M (Omat24) - Materials, LAMBench 0.238 - best CC-BY ⭐": f"{DPA_MODEL_PREFIX}{_R33}|Omat24|charge_spin",
    "DPA-3.3-1M (OMol25) - Molecules [ωB97M-V]": f"{DPA_MODEL_PREFIX}{_R33}|OMol25|charge_spin",
    "DPA-3.3-1M (OC20M) - Catalysis, adsorbate+slab [RPBE]": f"{DPA_MODEL_PREFIX}{_R33}|OC20M|charge_spin",
    "DPA-3.3-1M (Alex2D) - 2D materials": f"{DPA_MODEL_PREFIX}{_R33}|Alex2D|charge_spin",
    "DPA-3.3-1M (MPGen_OpenCSP) - Materials at high pressure": f"{DPA_MODEL_PREFIX}{_R33}|MPGen_OpenCSP|charge_spin",
    "DPA-3.3-1M (MPTrj) - MPtrj reference [PBE(+U)]": f"{DPA_MODEL_PREFIX}{_R33}|MPTrj|charge_spin",
    "DPA-3.3-1M (Alloy_tongqi) - Alloys": f"{DPA_MODEL_PREFIX}{_R33}|Alloy_tongqi|charge_spin",
    "DPA-3.3-1M (ODAC23) - MOFs / direct air capture [PBE-D3]": f"{DPA_MODEL_PREFIX}{_R33}|ODAC23|charge_spin",
    # --- DPA-3.2-5M: 24 layers, OpenLAM v2, LAMBench 0.260. Larger than 3.3 and
    #     beaten by it overall, but still the better one on Wiggle150 /
    #     TorsionNet-500, so it is kept for molecular work.
    "DPA-3.2-5M (OMat24) - Materials, LAMBench 0.260": f"{DPA_MODEL_PREFIX}{_R32}|OMat24|fparam",
    "DPA-3.2-5M (OMol25) - Molecules, best DPA-3 on torsions [ωB97M-V]": f"{DPA_MODEL_PREFIX}{_R32}|OMol25|fparam",
    "DPA-3.2-5M (OC20M) - Catalysis, adsorbate+slab [RPBE]": f"{DPA_MODEL_PREFIX}{_R32}|OC20M|fparam",
    "DPA-3.2-5M (Alex2D) - 2D materials": f"{DPA_MODEL_PREFIX}{_R32}|Alex2D|fparam",
    "DPA-3.2-5M (MPTrj) - MPtrj reference [PBE(+U)]": f"{DPA_MODEL_PREFIX}{_R32}|MPTrj|fparam",
    "DPA-3.2-5M (Alloy_APEX) - Alloys": f"{DPA_MODEL_PREFIX}{_R32}|Alloy_APEX|fparam",
    # --- DPA-3.1-3M: 16 layers, OpenLAM v1, LAMBench 0.293 (no charge/spin) ---
    "DPA-3.1-3M (Omat24) - Materials, LAMBench 0.293, OpenLAM v1": f"{DPA_MODEL_PREFIX}{_R31}|Omat24|",
    "DPA-3.1-3M (OC20M) - Catalysis, OpenLAM v1 [RPBE]": f"{DPA_MODEL_PREFIX}{_R31}|OC20M|",
    "DPA-3.1-3M (SPICE2) - Molecules, OpenLAM v1 [ωB97M-D3]": f"{DPA_MODEL_PREFIX}{_R31}|SPICE2|",
    # --- DPA-2.4-7M: legacy attention descriptor, patched multi-task ckpt ---
    "DPA-2.4-7M (Omat24) - Legacy DPA-2, OpenLAM v1": f"{DPA_MODEL_PREFIX}{_R24}|Omat24|",
    "DPA-2.4-7M (MPtrj) - Legacy DPA-2, MPtrj": f"{DPA_MODEL_PREFIX}{_R24}|MP_traj_v024_alldata_mixu|",
    # --- DPA4 / SeZM on OMol25. Molecular, charge/spin conditioned. Ordered by
    #     parameter count; the card publishes no per-variant MAE for this line.
    "DPA4-Plus-OMol25 - Molecules, 8.8M, most accurate [non-commercial]": f"{DPA_MODEL_PREFIX}{_R4O % 'Plus'}||charge_spin",
    "DPA4-Air-OMol25 - Molecules, 5.1M [non-commercial]": f"{DPA_MODEL_PREFIX}{_R4O % 'Air'}||charge_spin",
    "DPA4-Neo-OMol25 - Molecules, 1.1M [non-commercial]": f"{DPA_MODEL_PREFIX}{_R4O % 'Neo'}||charge_spin",
    "DPA4-Mini-OMol25 - Molecules, 0.7M [non-commercial]": f"{DPA_MODEL_PREFIX}{_R4O % 'Mini'}||charge_spin",
    "DPA4-Nano-OMol25 - Molecules, 0.5M, fastest [non-commercial]": f"{DPA_MODEL_PREFIX}{_R4O % 'Nano'}||charge_spin",
    # --- DPA4C on OMol25: compact/compressible, built for MD throughput rather
    #     than the last increment of accuracy. Ordered by parameter count.
    "DPA4C-Plus-OMol25 - Molecules, 2.19M, most accurate DPA4C [non-commercial]": f"{DPA_MODEL_PREFIX}{_R4CO % 'Plus'}||charge_spin",
    "DPA4C-Air-OMol25 - Molecules, 0.63M [non-commercial]": f"{DPA_MODEL_PREFIX}{_R4CO % 'Air'}||charge_spin",
    "DPA4C-Neo-OMol25 - Molecules, 0.54M [non-commercial] ⭐": f"{DPA_MODEL_PREFIX}{_R4CO % 'Neo'}||charge_spin",
    "DPA4C-Mini-OMol25 - Molecules, 0.20M [non-commercial]": f"{DPA_MODEL_PREFIX}{_R4CO % 'Mini'}||charge_spin",
    "DPA4C-Nano-OMol25 - Molecules, 0.035M, fastest [non-commercial]": f"{DPA_MODEL_PREFIX}{_R4CO % 'Nano'}||charge_spin",
}

# DPA4 / SeZM builds its neighbor list through vesin (or nvalchemiops) and
# refuses to run without one, so vesin is not optional here. Note that the
# vesin-torch wheel installs a `vesin_torch` module only -- plain `vesin` is
# a separate distribution and deepmd imports that one, so both are required.
DPA_ENV_SETUP = {
    "pip": (
        "pip install torch==2.11.0 deepmd-kit==3.2.0 mpich e3nn vesin vesin-torch "
        "huggingface_hub ase==3.28.0 pymatgen==2025.10.7 matscipy==1.2.0 "
        "phonopy==2.41.0 numpy pandas matplotlib"
    ),
    "note": (
        "the deepmd-kit wheel is compiled against one exact torch (3.2.0 → "
        "torch 2.11) and refuses to load against another, so DPA needs its own "
        "environment — locally, `pip install -r requirements-dpa.txt` does the "
        "whole thing. `mpich` and `e3nn` are missing from the wheel's declared "
        "dependencies but are imported by it, and `vesin` + `vesin-torch` build the "
        "neighbor list DPA4 requires. Weights download from Hugging Face on "
        "first use and are not gated (no token needed)."
    ),
}


# ---------------------------------------------------------------------------
# Active-settings cache (same contract as helpers.uma_models)
#
# Only the checkpoints trained with a frame-level total charge and spin
# multiplicity read these; for the rest they are ignored. Neutral singlet is
# the right default for an ordinary solid, and it is what the model cards use
# when the key is absent.
# ---------------------------------------------------------------------------
DPA_DEFAULTS = {"charge": 0, "spin": 1}

_ACTIVE_DPA_SETTINGS = None


def set_active_dpa_settings(settings):
    """Remember the sidebar's DPA settings so every script generator can reach
    them without threading two extra arguments through a dozen call sites."""
    global _ACTIVE_DPA_SETTINGS
    _ACTIVE_DPA_SETTINGS = dict(settings) if settings else None


def get_active_dpa_settings(settings=None):
    """Return the charge / spin picked in the sidebar, filled in from defaults."""
    merged = dict(DPA_DEFAULTS)
    if _ACTIVE_DPA_SETTINGS:
        merged.update(_ACTIVE_DPA_SETTINGS)
    if settings:
        merged.update(settings)
    return merged


# ---------------------------------------------------------------------------
# Model-string helpers
# ---------------------------------------------------------------------------
def is_dpa_model(selected_model_key=None, model_size=None):
    """True when the user picked a DPA / DeePMD-kit checkpoint."""
    if isinstance(model_size, str) and model_size.startswith(DPA_MODEL_PREFIX):
        return True
    if isinstance(selected_model_key, str) and selected_model_key in DPA_MODELS:
        return True
    return False


def parse_dpa_model(model_size):
    """Split the stored value into (repo_id, filename, head, charge_spin_key).

    ``head`` and ``charge_spin_key`` are None when the checkpoint has none.
    """
    raw = model_size
    if isinstance(raw, str) and raw.startswith(DPA_MODEL_PREFIX):
        raw = raw[len(DPA_MODEL_PREFIX):]
    parts = (raw or "").split("|")
    # Pad so an older/shorter value still unpacks instead of raising.
    parts += [""] * (4 - len(parts))
    repo_id, filename, head, cs_key = parts[:4]
    return repo_id, filename, (head or None), (cs_key or None)


def dpa_needs_charge_spin(model_size):
    """True for checkpoints that read a total charge and spin multiplicity."""
    return parse_dpa_model(model_size)[3] is not None


def dpa_cache_dir(repo_id):
    """Local folder a checkpoint of ``repo_id`` is downloaded into."""
    return Path.home() / ".cache" / DPA_CACHE_DIRNAME / repo_id.replace("/", "_")


def dpa_is_noncommercial(model_size):
    """DPA4 / SeZM weights are CC-BY-NC-4.0; DPA-2 / DPA-3 are CC-BY-4.0."""
    return "DPA4" in (parse_dpa_model(model_size)[0] or "")


# ---------------------------------------------------------------------------
# In-app calculator
# ---------------------------------------------------------------------------
def _set_deepmd_device(device):
    """Point DeePMD-kit's PyTorch backend at the requested device.

    DeePMD-kit reads a plain ``DEVICE`` environment variable *once*, when
    ``deepmd.pt.utils.env`` is first imported, and falls back to CUDA whenever
    it is anything other than "cpu". Setting it before the calculator is built
    is enough for the normal case, because ``deepmd.calculator`` itself pulls
    in only the backend-agnostic layer. If that module has already been
    imported this process (a second run after the device was switched), the
    constant it captured is patched too, otherwise the switch would silently
    not take.
    """
    import sys

    if device == "cpu":
        os.environ["DEVICE"] = "cpu"
    else:
        os.environ.pop("DEVICE", None)

    env_mod = sys.modules.get("deepmd.pt.utils.env")
    if env_mod is None:
        return
    try:
        import torch
        wanted = torch.device("cpu") if device == "cpu" else torch.device("cuda:0")
        if env_mod.DEVICE == wanted:
            return
        env_mod.DEVICE = wanted
        # deep_eval binds DEVICE by value at import time (`from ... import
        # DEVICE`), so the module attribute above does not reach it.
        eval_mod = sys.modules.get("deepmd.pt.infer.deep_eval")
        if eval_mod is not None and hasattr(eval_mod, "DEVICE"):
            eval_mod.DEVICE = wanted
    except Exception:
        pass


def download_dpa_model(model_size, log=None):
    """Fetch the checkpoint from Hugging Face and return a local file path.

    Downloads into ``dpa_cache_dir()`` rather than the shared hub cache: see
    DPA_CACHE_DIRNAME for why the file has to keep its real .pt / .pth name.
    """
    from huggingface_hub import hf_hub_download

    repo_id, filename, _, _ = parse_dpa_model(model_size)
    target = dpa_cache_dir(repo_id)
    existing = target / filename
    if existing.exists():
        if log:
            log(f"✅ Using cached checkpoint: {existing}")
        return str(existing)
    if log:
        log(f"⬇️  Downloading {filename} from huggingface.co/{repo_id} …")
        log("   (first use only — it is cached afterwards)")
    path = hf_hub_download(repo_id, filename, local_dir=str(target))
    if log:
        log(f"✅ Downloaded to {path}")
    return path


def attach_dpa_charge_spin(calculator, model_size, charge, spin, log=None):
    """Make every single point see the same total charge / spin multiplicity.

    The value lives in ``atoms.info``, which a structure file never carries and
    which optimizers and cell filters are free to drop when they copy the
    Atoms, so it is (re)written on every call rather than once up front.
    """
    import numpy as np
    from ase.calculators.calculator import all_changes

    key = parse_dpa_model(model_size)[3]
    if key is None:
        return calculator

    value = np.array([float(charge), float(spin)])
    original = calculator.calculate

    def _calculate(atoms=None, properties=None, system_changes=all_changes,
                   _o=original, _k=key, _v=value):
        if atoms is not None:
            atoms.info[_k] = _v.copy()
        return _o(atoms=atoms, properties=properties,
                  system_changes=system_changes)

    calculator.calculate = _calculate
    if log:
        log(f"  ⚡ DPA charge/spin injection enabled via atoms.info['{key}'] "
            f"(charge={int(charge)}, spin={int(spin)})")
    return calculator


def build_dpa_calculator(model_size, device="cpu", charge=None, spin=None, log=None):
    """Build the ASE calculator for a DPA checkpoint, downloading if needed."""
    repo_id, filename, head, _ = parse_dpa_model(model_size)

    _set_deepmd_device(device)
    model_path = download_dpa_model(model_size, log=log)

    from deepmd.calculator import DP

    if log:
        log(f"  Repository:     {repo_id}")
        log(f"  Checkpoint:     {filename}")
        log(f"  Model branch:   {head or '(single-task checkpoint)'}")
        log(f"  Device:         {device}")

    calculator = DP(model=model_path, head=head) if head else DP(model=model_path)
    _cfg = get_active_dpa_settings()
    charge = int(_cfg["charge"] if charge is None else charge)
    spin = int(_cfg["spin"] if spin is None else spin)
    return attach_dpa_charge_spin(calculator, model_size, charge, spin, log=log)


# ---------------------------------------------------------------------------
# Standalone-script calculator setup
# ---------------------------------------------------------------------------
def generate_dpa_calculator_code(model_size, device="cpu", indent="",
                                 charge=None, spin=None):
    """Return the `calculator = ...` block for a generated standalone script.

    charge / spin default to whatever the sidebar last set (neutral singlet if
    it set nothing), so every generator picks them up without passing them.
    """
    repo_id, filename, head, cs_key = parse_dpa_model(model_size)
    head_repr = repr(head) if head else "None"
    _cfg = get_active_dpa_settings()
    charge = int(_cfg["charge"] if charge is None else charge)
    spin = int(_cfg["spin"] if spin is None else spin)

    lines = [
        'print("🔧 Initializing DPA (DeePMD-kit) calculator...")',
        f'print("🤗 Repository:    {repo_id}")',
        f'print("🎯 Checkpoint:    {filename}")',
        f'print("🧭 Model branch:  {head or "(single-task checkpoint)"}")',
        '',
        '# DeePMD-kit reads a plain DEVICE environment variable when its',
        '# PyTorch backend is first imported, and falls back to CUDA for any',
        '# value other than "cpu". Set it before the calculator is built.',
        'import os as _dpa_os',
        f'_DPA_DEVICE = "{device}"',
        'if _DPA_DEVICE == "cpu":',
        '    _dpa_os.environ["DEVICE"] = "cpu"',
        'else:',
        '    _dpa_os.environ.pop("DEVICE", None)',
        '',
        '# The weights are downloaded into a folder of their own rather than',
        '# the shared Hugging Face cache: that cache stores every file as a',
        '# symlink to an extension-less blob, and DeePMD-kit picks its backend',
        '# from the file suffix *after* resolving symlinks, so a hub-cache path',
        '# fails with "Cannot detect the backend of the model file".',
        'from pathlib import Path as _DPAPath',
        f'_DPA_DIR = _DPAPath.home() / ".cache" / "{DPA_CACHE_DIRNAME}" / "{repo_id.replace("/", "_")}"',
        f'_DPA_MODEL = _DPA_DIR / "{filename}"',
        'if _DPA_MODEL.exists():',
        '    print(f"✅ Using cached checkpoint: {_DPA_MODEL}")',
        'else:',
        '    try:',
        '        from huggingface_hub import hf_hub_download',
        '    except ImportError:',
        '        print("❌ huggingface_hub is not installed: pip install huggingface_hub")',
        '        raise',
        f'    print("⬇️  Downloading {filename} from huggingface.co/{repo_id} …")',
        '    print("   (first use only — it is cached afterwards)")',
        f'    _DPA_MODEL = _DPAPath(hf_hub_download("{repo_id}", "{filename}",',
        '                                          local_dir=str(_DPA_DIR)))',
        '',
        'try:',
        '    from deepmd.calculator import DP as _DPCalculator',
        'except ImportError as _dpa_ie:',
        '    print(f"❌ DeePMD-kit is not installed: {_dpa_ie}")',
        '    print("   pip install deepmd-kit==3.2.0 mpich e3nn vesin vesin-torch")',
        '    print("   (it needs the exact torch it was compiled against — 2.11 for 3.2.0)")',
        '    raise',
        '',
        'try:',
        f'    calculator = _DPCalculator(model=str(_DPA_MODEL), head={head_repr})',
        f'    print(f"✅ DPA {filename} initialized on {{_DPA_DEVICE}}")',
        'except Exception as e:',
        '    print(f"❌ DPA initialization failed on {_DPA_DEVICE}: {e}")',
        '    if "mpich" in str(e) or "libmpi" in str(e):',
        '        print("   The deepmd-kit PyPI wheel imports MPI but does not declare it:")',
        '        print("     pip install mpich")',
        '    if "vesin" in str(e) or "nvalchemiops" in str(e):',
        '        print("   DPA4 / SeZM needs an O(N) neighbor list:")',
        '        print("     pip install vesin vesin-torch")',
            '        print("     (both: the vesin-torch wheel provides only vesin_torch,")',
            '        print("      while deepmd imports plain vesin)")',
        '    if "version of PyTorch" in str(e):',
        '        print("   Install the torch the deepmd-kit wheel was compiled against.")',
        '    raise',
    ]

    if cs_key:
        lines += [
            '',
            f'# This checkpoint was trained with the frame-level total charge and',
            f'# spin multiplicity as explicit inputs, read from',
            f'# atoms.info["{cs_key}"]. A structure file carries neither, and',
            '# optimizers / cell filters copy the Atoms freely, so they are',
            '# written on every call instead of once up front.',
            'import numpy as _dpa_np',
            'from ase.calculators.calculator import all_changes as _dpa_all_changes',
            f'_DPA_CHARGE_SPIN = _dpa_np.array([{float(charge)}, {float(spin)}])',
            '_dpa_orig_calculate = calculator.calculate',
            'def _dpa_calculate(atoms=None, properties=None,',
            '                   system_changes=_dpa_all_changes,',
            '                   _o=_dpa_orig_calculate, _v=_DPA_CHARGE_SPIN):',
            '    if atoms is not None:',
            f'        atoms.info["{cs_key}"] = _v.copy()',
            '    return _o(atoms=atoms, properties=properties,',
            '              system_changes=system_changes)',
            'calculator.calculate = _dpa_calculate',
            f'print("⚡ DPA charge/spin injection enabled via '
            f'atoms.info[\'{cs_key}\'] (charge={charge}, spin={spin})")',
        ]

    # Trailing newline on purpose: some templates splice this block straight in
    # front of the next statement, so the last line is terminated here.
    return "\n".join(indent + line if line else "" for line in lines) + "\n"
