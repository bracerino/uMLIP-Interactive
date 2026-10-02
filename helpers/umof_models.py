"""uMOF support: MACE potentials fine-tuned for metal-organic frameworks.

Two checkpoints from Inizan, Kamath, Elena & Persson, arXiv:2608.28100 (2026),
both trained on the uMOF r2SCAN-D4 dataset (85,524 configurations, 19,950 MOFs):

    uMOF-MH      fine-tuned from MACE-MH-1, short-range; loaded with mace_mp
    uMOF-POLAR   fine-tuned from MACE-POLAR-1-M, explicit long-range
                 electrostatics; loaded with mace_polar (needs graph_longrange)

Both predict r2SCAN-D4 energies directly, so no D3 correction is added on top.
uMOF-MH keeps the foundation model's 'pt_head' next to the fine-tuned head, so
the head is always set explicitly; uMOF-POLAR has the fine-tuned head only.

The weights are distributed as one zip on Figshare (CC BY 4.0), not as single
files, so the zip is downloaded once and both .model files are unpacked into
~/.cache/mace_foundation_models/, next to the other URL-hosted MACE models.
"""
from pathlib import Path

UMOF_MODEL_PREFIX = "umof:"
UMOF_HEAD = "r2scan+d4"
UMOF_ZIP_URL = "https://ndownloader.figshare.com/files/67738656"
UMOF_ZIP_NAME = "umof_models.zip"
UMOF_DOI = "10.6084/m9.figshare.33311829"
UMOF_CACHE_DIRNAME = "mace_foundation_models"

# The stored value is "umof:<checkpoint name inside the zip, without .model>".
# The uMOF-MH label names MACE-MH-1, so app.is_multihead_model() excludes uMOF
# explicitly to keep the MACE-MH head and D3 options hidden; "POLAR" in the second label is what routes it through the MACE-POLAR settings
# (charge, spin, external field) and output extraction.
UMOF_MODELS = {
    "uMOF-MH (MACE) - MOFs, fine-tuned MACE-MH-1 [r2SCAN-D4] ⭐": f"{UMOF_MODEL_PREFIX}uMOF-MH",
    "uMOF-POLAR (MACE) - MOFs, fine-tuned MACE-POLAR-1-M [r2SCAN-D4]": f"{UMOF_MODEL_PREFIX}uMOF-POLAR",
}


def is_umof_model(selected_model_key=None, model_size=None):
    if isinstance(model_size, str) and model_size.startswith(UMOF_MODEL_PREFIX):
        return True
    return bool(selected_model_key) and selected_model_key in UMOF_MODELS


def umof_checkpoint_name(model_size):
    """'umof:uMOF-MH' -> 'uMOF-MH.model'."""
    name = str(model_size)
    if name.startswith(UMOF_MODEL_PREFIX):
        name = name[len(UMOF_MODEL_PREFIX):]
    return name if name.endswith(".model") else f"{name}.model"


def umof_is_polar(model_size):
    return "POLAR" in umof_checkpoint_name(model_size).upper()


def umof_cache_dir():
    return Path.home() / ".cache" / UMOF_CACHE_DIRNAME


def download_umof_model(model_size, log=None):
    """Return the local path of a uMOF checkpoint, fetching the zip on first use."""
    import urllib.request
    import zipfile

    log = log or (lambda _msg: None)
    filename = umof_checkpoint_name(model_size)
    cache_dir = umof_cache_dir()
    model_path = cache_dir / filename
    if model_path.exists():
        log(f"✅ Using cached model: {model_path}")
        return str(model_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / UMOF_ZIP_NAME
    if not zip_path.exists():
        log(f"📥 Downloading the uMOF models from Figshare (doi:{UMOF_DOI})")
        log("   ~106 MB zip with both checkpoints, first use only...")
        tmp_path = zip_path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(UMOF_ZIP_URL, str(tmp_path))
            tmp_path.replace(zip_path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to download the uMOF models: {e}") from e

    # Unpack every checkpoint so the other model is ready too. The zip also
    # carries macOS '__MACOSX/._*' entries, which are skipped.
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if not name.endswith(".model") or "__MACOSX" in member or name.startswith("._"):
                continue
            target = cache_dir / name
            if not target.exists():
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    if not model_path.exists():
        raise RuntimeError(f"{filename} was not found in {zip_path}")
    log(f"✅ Model unpacked: {model_path}")
    return str(model_path)


def build_umof_calculator(model_size, device="cpu", dtype="float64", enable_cueq=False,
                          log=None):
    """Build the ASE calculator for a live in-app run."""
    log = log or (lambda _msg: None)
    model_path = download_umof_model(model_size, log=log)
    if umof_is_polar(model_size):
        from mace.calculators import mace_polar
        return mace_polar(model=model_path, device=device, default_dtype=dtype)
    from mace.calculators import mace_mp
    kwargs = {"enable_cueq": True} if enable_cueq and device == "cuda" else {}
    return mace_mp(model=model_path, head=UMOF_HEAD, dispersion=False,
                   device=device, default_dtype=dtype, **kwargs)


def generate_umof_calculator_code(model_size, device="cpu", dtype="float64", indent="",
                                  enable_cueq=False):
    """Return the `calculator = ...` block for a generated standalone script."""
    filename = umof_checkpoint_name(model_size)
    is_polar = umof_is_polar(model_size)
    label = filename[:-len(".model")]
    if is_polar:
        ctor = "mace_polar"
        args = f'model=str(_UMOF_MODEL), default_dtype="{dtype}"'
        cueq = ""
    else:
        ctor = "mace_mp"
        args = (f'model=str(_UMOF_MODEL), head="{UMOF_HEAD}", dispersion=False, '
                f'default_dtype="{dtype}"')
        cueq = ", enable_cueq=True" if enable_cueq and device == "cuda" else ""

    lines = [
        f'print("🔧 Initializing {label} calculator (MACE fine-tuned for MOFs)...")',
        f'print("🎯 Checkpoint:    {filename}")',
        f'print("💻 Device:        {device}")',
        '',
        f'# The uMOF weights come as one zip on Figshare (doi:{UMOF_DOI}, CC BY 4.0);',
        f'# it is downloaded once and unpacked into ~/.cache/{UMOF_CACHE_DIRNAME}/.',
        'import zipfile as _umof_zip',
        'import urllib.request as _umof_url',
        'from pathlib import Path as _UmofPath',
        f'_UMOF_DIR = _UmofPath.home() / ".cache" / "{UMOF_CACHE_DIRNAME}"',
        f'_UMOF_MODEL = _UMOF_DIR / "{filename}"',
        'if _UMOF_MODEL.exists():',
        '    print(f"✅ Using cached checkpoint: {_UMOF_MODEL}")',
        'else:',
        '    _UMOF_DIR.mkdir(parents=True, exist_ok=True)',
        f'    _umof_zip_path = _UMOF_DIR / "{UMOF_ZIP_NAME}"',
        '    if not _umof_zip_path.exists():',
        '        print("⬇️  Downloading the uMOF models (~106 MB, first use only) ...")',
        f'        _umof_url.urlretrieve("{UMOF_ZIP_URL}", str(_umof_zip_path))',
        '    with _umof_zip.ZipFile(_umof_zip_path) as _zf:',
        '        for _member in _zf.namelist():',
        '            _name = _UmofPath(_member).name',
        '            if _name.endswith(".model") and "__MACOSX" not in _member and not _name.startswith("._"):',
        '                with _zf.open(_member) as _src, open(_UMOF_DIR / _name, "wb") as _dst:',
        '                    _dst.write(_src.read())',
        '    print(f"✅ Model unpacked: {_UMOF_MODEL}")',
        '',
        'try:',
        f'    from mace.calculators import {ctor}',
        'except ImportError:',
        f'    print("❌ {ctor} not available. Please run: pip install --upgrade mace-torch")',
        '    exit()',
    ]
    if is_polar:
        lines += [
            '# uMOF-POLAR also needs graph_longrange (pip install graph-longrange==0.4.0).',
        ]
    lines += [
        'try:',
        f'    calculator = {ctor}({args}, device="{device}"{cueq})',
        f'    print("✅ {label} initialized on {device}")',
        'except Exception as e:',
        f'    print(f"❌ {label} initialization failed on {device}: {{e}}")',
    ]
    if device == "cpu":
        lines += ['    exit()']
    else:
        lines += [
            '    print("Attempting fallback to CPU...")',
            '    try:',
            f'        calculator = {ctor}({args}, device="cpu")',
            f'        print("✅ {label} initialized on CPU (fallback)")',
            '    except Exception as cpu_e:',
            f'        print(f"❌ {label} CPU fallback failed: {{cpu_e}}")',
            '        exit()',
        ]
    return "\n".join(indent + line if line else line for line in lines) + "\n"
