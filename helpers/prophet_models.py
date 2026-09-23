"""Prophet (Kairos Materials) support.

Two research-preview checkpoints, both with an ASE calculator:

    from prophet import KairosCalculator          # energies / forces / stress
    from prophet_spin import SpinCalculator       # E(R, M): spin-dependent

Weights live on Hugging Face (kairosmaterial/prophet, CC-BY-4.0, not gated) and
the calculators take a local file, so the checkpoint is downloaded first and
cached in ~/.cache/prophet_models/.

Spins
-----
Prophet-Spin reads ASE's initial magnetic moments on every single point, which
a structure file does not carry, so the moments are (re)applied by wrapping the
calculator rather than by editing each calculation type. Three ways to set them:

  * per element  -- {"Fe": 2.5, "O": 0.0}, applied to every atom of that element
  * per element, non-collinear -- {"Fe": [0, 0, 2.5]} gives an (N, 3) array
  * per atom     -- one value (or one xyz triple) per atom, in file order
  * predicted    -- the values above become seeds that fix the magnetic ordering
                    and Prophet's moment head (prophet-v1-mag-head.pt) predicts
                    the moments the energy is then evaluated at

Kernels
-------
Both models can use CUDA tensor-product kernels from the optional
openequivariance extra, which also needs a CUDA toolchain at runtime. They are
used only when the run is on a GPU and that package imports; otherwise the pure
e3nn path is taken, which runs everywhere. For the spin model that choice lives
in the checkpoint's own config header (kernel=True), so it is overridden while
loading rather than passed as an argument.
"""
import inspect
from pathlib import Path

PROPHET_FAMILY_NAME = "Prophet (Kairos)"
PROPHET_MODEL_PREFIX = "prophet:"
PROPHET_HF_REPO = "kairosmaterial/prophet"
PROPHET_CACHE_DIRNAME = "prophet_models"

# The stored value is "prophet:<checkpoint filename>".
PROPHET_MODELS = {
    "Prophet-OAME-MBD - Materials, 62.3M, Matbench Discovery reference ⭐": f"{PROPHET_MODEL_PREFIX}prophet-oame-mbd.pt",
    "Prophet-V1-Mag - Spin-dependent E(R,M), set magnetic moments ⭐": f"{PROPHET_MODEL_PREFIX}prophet-v1-mag.pt",
}

# The spin-dependent checkpoint, by filename, and the moment-head bundle that
# belongs to it (used when the moments are predicted rather than given).
_SPIN_CHECKPOINTS = {"prophet-v1-mag.pt"}
PROPHET_HEAD_FILENAME = "prophet-v1-mag-head.pt"

PROPHET_ENV_SETUP = {
    "pip": (
        "pip install \"prophet-mlip @ git+https://github.com/kairosmaterial/prophet.git\" "
        "huggingface_hub torch>=2.7 ase>=3.29 pymatgen==2025.10.7 matscipy==1.2.0 "
        "phonopy==2.41.0 numpy pandas matplotlib"
    ),
    "note": (
        "prophet-mlip installs from GitHub (there is no PyPI release yet) and needs "
        "Python ≥ 3.11, torch ≥ 2.7 and ASE ≥ 3.29 — locally, `pip install -r "
        "requirements-prophet.txt` does the whole thing. Weights download from "
        "huggingface.co/kairosmaterial/prophet on first use (not gated, no token) "
        "and are cached in ~/.cache/prophet_models/. Research-preview models, "
        "CC-BY-4.0 weights / MIT code."
    ),
}


# ---------------------------------------------------------------------------
# Spin settings (same contract as helpers.dpa_models' charge/spin cache)
# ---------------------------------------------------------------------------
# mode: "element" (per element, collinear or a 3-vector), "atom" (per atom), or
# "predict" (the values are seeds and Prophet's moment head predicts the rest).
PROPHET_SPIN_DEFAULTS = {"mode": "element", "per_element": {}, "per_atom": [],
                         "seed_mode": "element"}

_ACTIVE_PROPHET_SPINS = None


def set_active_prophet_spins(settings):
    """Remember the sidebar's spin settings for every script generator."""
    global _ACTIVE_PROPHET_SPINS
    _ACTIVE_PROPHET_SPINS = dict(settings) if settings else None


def get_active_prophet_spins(settings=None):
    """Return the spin configuration picked in the sidebar, filled from defaults."""
    merged = dict(PROPHET_SPIN_DEFAULTS)
    if _ACTIVE_PROPHET_SPINS:
        merged.update(_ACTIVE_PROPHET_SPINS)
    if settings:
        merged.update(settings)
    return merged


# ---------------------------------------------------------------------------
# Model-string helpers
# ---------------------------------------------------------------------------
def is_prophet_model(selected_model_key=None, model_size=None):
    """True when the user picked a Prophet checkpoint."""
    if isinstance(model_size, str) and model_size.startswith(PROPHET_MODEL_PREFIX):
        return True
    if isinstance(selected_model_key, str) and selected_model_key in PROPHET_MODELS:
        return True
    return False


def prophet_checkpoint_name(model_size):
    """Strip the "prophet:" family prefix off the stored value."""
    if isinstance(model_size, str) and model_size.startswith(PROPHET_MODEL_PREFIX):
        return model_size[len(PROPHET_MODEL_PREFIX):]
    return model_size


def prophet_is_spin_model(model_size):
    """True for the checkpoint that takes magnetic moments as an input."""
    return prophet_checkpoint_name(model_size) in _SPIN_CHECKPOINTS


def prophet_cache_dir():
    """Local folder the checkpoints are downloaded into."""
    return Path.home() / ".cache" / PROPHET_CACHE_DIRNAME


# ---------------------------------------------------------------------------
# Spins -> per-atom moments
#
# Both the app and the generated scripts use these two functions; the scripts
# get them by their source, so there is one implementation to keep right.
# ---------------------------------------------------------------------------
def prophet_resolve_spins(symbols, spins):
    """Per-atom magnetic moments for ``symbols``, or None when nothing is set.

    Returns a list of floats (collinear) or of [x, y, z] (non-collinear).
    """
    mode = (spins or {}).get("mode", "element")
    if mode == "predict":
        # The values are seed moments; they are read the same way, per element
        # or per atom, and the moment head takes it from there.
        mode = (spins or {}).get("seed_mode", "element")

    if mode == "atom":
        per_atom = list((spins or {}).get("per_atom") or [])
        if not per_atom:
            return None
        if len(per_atom) != len(symbols):
            raise ValueError(
                f"{len(per_atom)} magnetic moments given for {len(symbols)} atoms")
        return [list(m) if isinstance(m, (list, tuple)) else float(m) for m in per_atom]

    per_element = dict((spins or {}).get("per_element") or {})
    if not per_element:
        return None
    missing = sorted({s for s in symbols if s not in per_element})
    if missing:
        raise ValueError(f"no magnetic moment set for {', '.join(missing)}")
    return [list(per_element[s]) if isinstance(per_element[s], (list, tuple))
            else float(per_element[s]) for s in symbols]


def prophet_apply_spins(atoms, spins):
    """Stamp the configured moments onto ``atoms``; returns True when it did."""
    moments = prophet_resolve_spins(atoms.get_chemical_symbols(), spins)
    if moments is None:
        return False
    atoms.set_initial_magnetic_moments(moments)
    return True


# ---------------------------------------------------------------------------
# In-app calculator
# ---------------------------------------------------------------------------
def download_prophet_model(model_size, log=None, filename=None):
    """Fetch a checkpoint from Hugging Face and return a local file path."""
    from huggingface_hub import hf_hub_download

    filename = filename or prophet_checkpoint_name(model_size)
    target = prophet_cache_dir()
    existing = target / filename
    if existing.exists():
        if log:
            log(f"✅ Using cached checkpoint: {existing}")
        return str(existing)
    if log:
        log(f"⬇️  Downloading {filename} from huggingface.co/{PROPHET_HF_REPO} …")
        log("   (first use only — it is cached afterwards)")
    path = hf_hub_download(PROPHET_HF_REPO, filename, local_dir=str(target))
    if log:
        log(f"✅ Downloaded to {path}")
    return path


def summarize_prophet_moments(symbols, moments):
    """Per-element summary of predicted moments, e.g. "Ni +1.02, O +0.04 (0.01..0.08)"."""
    import numpy as _np

    values = _np.asarray(moments, dtype=float)
    if values.ndim == 2:  # non-collinear: report the length of each vector
        values = _np.linalg.norm(values, axis=1)
    parts = []
    for element in dict.fromkeys(symbols):
        picked = values[[i for i, s in enumerate(symbols) if s == element]]
        if picked.max() - picked.min() > 0.01:
            parts.append(f"{element} {picked.mean():+.2f} "
                         f"({picked.min():+.2f}..{picked.max():+.2f})")
        else:
            parts.append(f"{element} {picked.mean():+.2f}")
    return ", ".join(parts)


def attach_prophet_spins(calculator, spins, log=None):
    """Re-apply the configured moments before every single point.

    Prophet-Spin reads them from the Atoms it is handed, and structures read
    from a file carry none, so they are written on every call rather than once
    up front — an optimizer or cell filter is free to hand over a fresh copy.

    When the moment head is doing the predicting, the moments it came up with
    are reported the first time each structure is seen; repeating that every
    step would bury an MD log.
    """
    from ase.calculators.calculator import all_changes

    original = calculator.calculate
    reported = set()

    def _calculate(atoms=None, properties=None, system_changes=all_changes,
                   _o=original, _s=spins):
        if atoms is not None:
            prophet_apply_spins(atoms, _s)
        result = _o(atoms=atoms, properties=properties, system_changes=system_changes)
        predicted = calculator.results.get("magmoms") if log else None
        if predicted is not None and atoms is not None:
            signature = tuple(atoms.get_chemical_symbols())
            if signature not in reported:
                reported.add(signature)
                log(f"  🧲 Predicted moments (μB): "
                    f"{summarize_prophet_moments(signature, predicted)}")
        return result

    calculator.calculate = _calculate
    if log:
        log(f"  ⚡ Prophet spin injection enabled ({describe_prophet_spins(spins)})")
    return calculator


def describe_prophet_spins(spins):
    """One-line summary of a spin configuration, for logs and script headers."""
    spins = spins or {}
    if spins.get("mode") == "predict":
        seeds = dict(spins, mode=spins.get("seed_mode", "element"))
        return f"predicted by the moment head, seeded with {describe_prophet_spins(seeds)}"
    if spins.get("mode") == "atom":
        n = len(spins.get("per_atom") or [])
        return f"{n} per-atom moments" if n else "no moments set"
    per_element = spins.get("per_element") or {}
    if not per_element:
        return "no moments set"
    return ", ".join(f"{el}={val}" for el, val in per_element.items())


def prophet_use_kernel(device):
    """True when the CUDA tensor-product kernels can actually be used."""
    if device == "cpu":
        return False
    try:
        import openequivariance  # noqa: F401
    except ImportError:
        return False
    return True


def prophet_force_e3nn(device="cpu"):
    """Take the pure e3nn path when the CUDA kernels are unavailable.

    The spin checkpoint and the moment-head bundle both ask for the kernels in
    their own config headers, and their loaders take no argument for that, so
    both loaders are swapped for ones that clear the flag. The tensor-product
    weights they drop are the ones the stock loaders drop too (they are not
    stored in the files).
    """
    import io
    import json

    import torch
    import prophet_spin.predictor as _pred
    import prophet_spin.runtime as _rt
    from prophet_spin.adapter import ProphetSpin

    if prophet_use_kernel(device):
        return

    if not getattr(_rt.load_spin_model, "_prophet_no_kernel", False):
        def _load_no_kernel(path):
            with open(path, "rb") as fh:
                config = json.loads(fh.readline().decode())
                state = torch.load(io.BytesIO(fh.read()), map_location="cpu")
            model = ProphetSpin(dict(config, kernel=False))
            model.load_state_dict(
                {k: v for k, v in state.items() if ".tp." not in k}, strict=False)
            model.eval()
            return model, config

        _load_no_kernel._prophet_no_kernel = True
        _rt.load_spin_model = _load_no_kernel

    if not getattr(_pred.load_head_bundle, "_prophet_no_kernel", False):
        _stock_head = _pred.load_head_bundle

        def _load_head_no_kernel(path):
            import prophet_spin.adapter as _adapter

            stock_cls = _adapter.ProphetSpin

            class _NoKernelProphetSpin(stock_cls):
                def __init__(self, config):
                    super().__init__(dict(config, kernel=False))

            _adapter.ProphetSpin = _NoKernelProphetSpin
            _pred.ProphetSpin = _NoKernelProphetSpin
            try:
                return _stock_head(path)
            finally:
                _adapter.ProphetSpin = stock_cls
                _pred.ProphetSpin = stock_cls

        _load_head_no_kernel._prophet_no_kernel = True
        _pred.load_head_bundle = _load_head_no_kernel


def prophet_spin_calculator(model_path, device="cpu", head_path=None):
    """The spin calculator: moments as given, or predicted when ``head_path`` is set.

    With a head bundle this is Prophet's MagmomPredictor: the configured moments
    become seeds encoding the magnetic ordering, the head predicts the moments,
    and energies and forces are evaluated at those predicted moments.
    """
    prophet_force_e3nn(device)

    if head_path:
        from prophet_spin import MagmomPredictor

        return MagmomPredictor(model_path=str(model_path), head_path=str(head_path),
                               device=device)

    from prophet_spin import SpinCalculator

    return SpinCalculator(model_path=str(model_path), device=device)


def build_prophet_calculator(model_size, device="cpu", spins=None, log=None):
    """Build the ASE calculator for a Prophet checkpoint, downloading if needed."""
    model_path = download_prophet_model(model_size, log=log)

    if prophet_is_spin_model(model_size):
        spins = get_active_prophet_spins(spins)
        head_path = None
        if spins.get("mode") == "predict":
            head_path = download_prophet_model(
                model_size, log=log, filename=PROPHET_HEAD_FILENAME)
        calculator = prophet_spin_calculator(
            model_path, device=device, head_path=head_path)
        return attach_prophet_spins(calculator, spins, log=log)

    from prophet import KairosCalculator

    return KairosCalculator(model_path=model_path,
                            use_kernel=prophet_use_kernel(device), device=device)


# ---------------------------------------------------------------------------
# Generated-script calculator block
# ---------------------------------------------------------------------------
def generate_prophet_calculator_code(model_size, device="cpu", indent="", spins=None):
    """Return the `calculator = ...` block for a generated standalone script."""
    filename = prophet_checkpoint_name(model_size)
    is_spin = prophet_is_spin_model(model_size)
    spins = get_active_prophet_spins(spins)

    lines = [
        'print("🔧 Initializing Prophet (Kairos) calculator...")',
        f'print("🤗 Repository:    {PROPHET_HF_REPO}")',
        f'print("🎯 Checkpoint:    {filename}")',
        f'print("💻 Device:        {device}")',
    ]
    if is_spin:
        lines.append(f'print("🧲 Magnetic moments: {describe_prophet_spins(spins)}")')
    lines += [
        '',
        '# The calculators take a local checkpoint, so it is fetched from',
        '# Hugging Face first (not gated) and cached in ~/.cache/prophet_models/.',
        'from pathlib import Path as _ProphetPath',
        f'_PROPHET_DIR = _ProphetPath.home() / ".cache" / "{PROPHET_CACHE_DIRNAME}"',
        f'_PROPHET_MODEL = _PROPHET_DIR / "{filename}"',
        f'_PROPHET_DEVICE = "{device}"',
        'if _PROPHET_MODEL.exists():',
        '    print(f"✅ Using cached checkpoint: {_PROPHET_MODEL}")',
        'else:',
        '    try:',
        '        from huggingface_hub import hf_hub_download',
        '    except ImportError:',
        '        print("❌ huggingface_hub is not installed: pip install huggingface_hub")',
        '        raise',
        f'    print("⬇️  Downloading {filename} from huggingface.co/{PROPHET_HF_REPO} …")',
        '    print("   (first use only — it is cached afterwards)")',
        f'    _PROPHET_MODEL = _ProphetPath(hf_hub_download("{PROPHET_HF_REPO}", "{filename}",',
        '                                                   local_dir=str(_PROPHET_DIR)))',
        '',
    ]

    if is_spin:
        if spins.get("mode") == "predict":
            lines += [
                f'_PROPHET_HEAD = _PROPHET_DIR / "{PROPHET_HEAD_FILENAME}"',
                'if not _PROPHET_HEAD.exists():',
                '    from huggingface_hub import hf_hub_download',
                f'    print("⬇️  Downloading {PROPHET_HEAD_FILENAME} (moment head) …")',
                f'    _PROPHET_HEAD = _ProphetPath(hf_hub_download("{PROPHET_HF_REPO}",',
                f'                                                 "{PROPHET_HEAD_FILENAME}",',
                '                                                 local_dir=str(_PROPHET_DIR)))',
                '',
            ]
        else:
            lines += ['_PROPHET_HEAD = None', '']
        lines += [f'_PROPHET_SPINS = {spins!r}', '']
        lines += inspect.getsource(prophet_resolve_spins).rstrip().split("\n")
        lines += ['']
        lines += inspect.getsource(prophet_apply_spins).rstrip().split("\n")
        lines += ['']
        lines += inspect.getsource(describe_prophet_spins).rstrip().split("\n")
        lines += ['']
        lines += inspect.getsource(summarize_prophet_moments).rstrip().split("\n")
        lines += ['']
        lines += inspect.getsource(attach_prophet_spins).rstrip().split("\n")
        lines += ['']
        lines += inspect.getsource(prophet_use_kernel).rstrip().split("\n")
        lines += ['']
        lines += inspect.getsource(prophet_force_e3nn).rstrip().split("\n")
        lines += ['']
        lines += inspect.getsource(prophet_spin_calculator).rstrip().split("\n")
        lines += [
            '',
            'try:',
            '    import prophet_spin  # noqa: F401',
            'except ImportError as _prophet_ie:',
            '    print(f"❌ prophet-mlip is not installed: {_prophet_ie}")',
            '    print(\'   pip install "prophet-mlip @ git+https://github.com/kairosmaterial/prophet.git"\')',
            '    raise',
            '',
            'try:',
            '    calculator = attach_prophet_spins(',
            '        prophet_spin_calculator(_PROPHET_MODEL, device=_PROPHET_DEVICE,',
            '                                head_path=_PROPHET_HEAD),',
            '        _PROPHET_SPINS, log=print)',
            f'    print(f"✅ Prophet {filename} initialized on {{_PROPHET_DEVICE}}")',
            'except Exception as e:',
            f'    print(f"❌ Prophet initialization failed on {{_PROPHET_DEVICE}}: {{e}}")',
            '    raise',
        ]
    else:
        lines += [
            'try:',
            '    from prophet import KairosCalculator',
            'except ImportError as _prophet_ie:',
            '    print(f"❌ prophet-mlip is not installed: {_prophet_ie}")',
            '    print(\'   pip install "prophet-mlip @ git+https://github.com/kairosmaterial/prophet.git"\')',
            '    raise',
            '',
            '# The CUDA tensor-product kernels need the openequivariance extra and',
            '# a CUDA toolchain; the pure e3nn path runs on both CPU and GPU.',
            '',
        ]
        lines += inspect.getsource(prophet_use_kernel).rstrip().split("\n")
        lines += [
            '',
            'try:',
            '    calculator = KairosCalculator(model_path=str(_PROPHET_MODEL),',
            '                                  use_kernel=prophet_use_kernel(_PROPHET_DEVICE),',
            '                                  device=_PROPHET_DEVICE)',
            f'    print(f"✅ Prophet {filename} initialized on {{_PROPHET_DEVICE}}")',
            'except Exception as e:',
            f'    print(f"❌ Prophet initialization failed on {{_PROPHET_DEVICE}}: {{e}}")',
            '    if _PROPHET_DEVICE == "cuda":',
            '        print("⚠️ GPU initialization failed, falling back to CPU...")',
            '        calculator = KairosCalculator(model_path=str(_PROPHET_MODEL),',
            '                                      use_kernel=False, device="cpu")',
            '        print("✅ Prophet initialized on CPU (fallback)")',
            '    else:',
            '        raise',
        ]

    # Trailing newline on purpose: some templates splice this block straight in
    # front of the next statement, so the last line is terminated here.
    return "\n".join(indent + line if line else "" for line in lines) + "\n"
