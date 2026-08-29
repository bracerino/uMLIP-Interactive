"""
Model-dispatch code generator shared by the standalone-script builders.

`build_calculator_code()` returns the two pieces every generated script needs
to instantiate the calculator the user picked in the app:

    model_imports          extra ``import`` lines for that model family
    calculator_setup_str   the block that assigns ``calculator``

The block is written at module level (no indentation); callers that drop it
inside ``main()`` should run it through ``textwrap.indent``.
"""

from helpers.quantum_espresso import (
    is_qe_model, generate_qe_calculator_code, get_active_qe_settings,
)
from helpers.uma_models import (
    is_uma_model, generate_uma_calculator_code, get_active_uma_settings,
    uma_checkpoint_name,
)
from helpers.custom_model_paths import (
    mace_model_resolution_code, is_custom_mace_model,
    mace_cueq_preamble, mace_cueq_arg,
    sevennet_cueq_preamble, sevennet_cueq_arg,
)


def is_url_mace_model(model_size):
    """MACE foundation models that are distributed as a download URL."""
    return isinstance(model_size, str) and model_size.startswith(("http://", "https://"))


def is_multihead_mace(selected_model):
    """MACE-MH-* models, which require an explicit `head=` keyword."""
    return ("Multi-head" in selected_model
            or "MH-0" in selected_model or "MH-1" in selected_model)


def build_calculator_code(selected_model, model_size, device, dtype,
                          custom_sevennet_path=None, custom_grace_path=None,
                          custom_mace_path=None, mace_enable_cueq=False,
                          sevennet_enable_cueq=False, mace_head=None,
                          mace_dispersion=False, mace_dispersion_xc="pbe"):
    model_imports = ""
    # cuEquivariance (MACE + CUDA only).
    _cq_pre = mace_cueq_preamble(mace_enable_cueq, device)
    _cq_kw = mace_cueq_arg(mace_enable_cueq, device)

    # SevenNet tensor product accelerator (SevenNet + CUDA only).
    _7net_pre = sevennet_cueq_preamble(sevennet_enable_cueq, device)
    _7net_arg = sevennet_cueq_arg(sevennet_enable_cueq, device)
    _7net_kw = f", {_7net_arg}" if _7net_arg else ""
    _cq = f", {_cq_kw}" if _cq_kw else ""

    is_custom_mace = ("MACE" in selected_model and "OFF" not in selected_model
                      and is_custom_mace_model(model_size=model_size,
                                               selected_model_key=selected_model,
                                               custom_mace_path=custom_mace_path))
    is_custom_sevennet = ("SevenNet" in selected_model
                          and (model_size == "7net:custom" or bool(custom_sevennet_path)))
    is_grace = "GRACE" in selected_model
    is_custom_grace = is_grace and (model_size == "grace:custom" or bool(custom_grace_path))

    calculator_setup_str = ""
    if is_qe_model(selected_model, model_size):
        # Quantum ESPRESSO: external DFT binary, no MLIP setup applies.
        calculator_setup_str = generate_qe_calculator_code(
            get_active_qe_settings(), indent="")
    elif is_uma_model(selected_model, model_size):
        # UMA: a fairchem predict unit plus a task name, nothing below applies.
        _uma = get_active_uma_settings()
        _uma["model_id"] = uma_checkpoint_name(model_size)
        _uma["device"] = device
        calculator_setup_str = generate_uma_calculator_code(_uma, indent="")
    elif "CHGNet" in selected_model:
        model_imports += """
try:
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
except ImportError:
    print("Error: CHGNet not found. Please install with: pip install chgnet")
    exit()
"""
        calculator_setup_str = f"""
print("Setting up CHGNet calculator...")
original_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.float32)
try:
    chgnet = CHGNet.load(model_name="{model_size}", use_device="{device}", verbose=False)
    calculator = CHGNetCalculator(model=chgnet, use_device="{device}")
    torch.set_default_dtype(original_dtype)
    print(f"✅ CHGNet {model_size} initialized on {device}")
except Exception as e:
    print(f"❌ CHGNet initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        chgnet = CHGNet.load(model_name="{model_size}", use_device="cpu", verbose=False)
        calculator = CHGNetCalculator(model=chgnet, use_device="cpu")
        torch.set_default_dtype(original_dtype)
        print("✅ CHGNet initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ CHGNet CPU fallback failed: {{cpu_e}}")
        exit()
"""
    elif "MACE-OFF" in selected_model:
        model_imports += """
try:
    from mace.calculators import mace_off
except ImportError:
    print("Error: MACE-OFF not found. Please install with: pip install mace-torch")
    exit()
"""
        calculator_setup_str = f"""
print("Setting up MACE-OFF calculator...")
{_cq_pre}try:
    calculator = mace_off(
        model="{model_size}", default_dtype="{dtype}", device="{device}"{_cq}
    )
    print(f"✅ MACE-OFF {model_size} initialized on {device}")
except Exception as e:
    print(f"❌ MACE-OFF initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = mace_off(
            model="{model_size}", default_dtype="{dtype}", device="cpu"
        )
        print("✅ MACE-OFF initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ MACE-OFF CPU fallback failed: {{cpu_e}}")
        exit()
"""
    elif "MACE" in selected_model:
        model_imports += """
try:
    from mace.calculators import mace_mp
except ImportError:
    print("Error: MACE not found. Please install with: pip install mace-torch")
    exit()
"""
        if is_custom_mace:
            # Custom local MACE model: explicit path, or a *.model file
            # auto-discovered next to this generated script.
            calculator_setup_str = f"""
print("Setting up custom MACE calculator...")
{mace_model_resolution_code(custom_mace_path)}
{_cq_pre}try:
    calculator = mace_mp(
        model=_mace_model_path, dispersion=False, default_dtype="{dtype}", device="{device}"{_cq}
    )
    print(f"✅ Custom MACE model initialized on {device}")
except Exception as e:
    print(f"❌ Custom MACE initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = mace_mp(
            model=_mace_model_path, dispersion=False, default_dtype="{dtype}", device="cpu"
        )
        print("✅ Custom MACE initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ MACE CPU fallback failed: {{cpu_e}}")
        exit()
"""
        else:
            # Foundation models: plain names ("medium-omat-0") and the
            # URL-hosted ones (MACE-MH-*, MACE-MATPES-*). mace_mp downloads a
            # URL itself; the multi-head models additionally need `head=`.
            _mace_args = [f'model="{model_size}"', f'default_dtype="{dtype}"']
            if mace_head and (is_multihead_mace(selected_model)
                              or is_url_mace_model(model_size)):
                _mace_args.append(f'head="{mace_head}"')
            if mace_dispersion:
                _mace_args += ["dispersion=True",
                               f'dispersion_xc="{mace_dispersion_xc}"']
            else:
                _mace_args.append("dispersion=False")
            _mace_gpu = ", ".join(_mace_args + [f'device="{device}"'])
            _mace_cpu = ", ".join(_mace_args + ['device="cpu"'])
            # A URL is too long to echo back in the "initialized" line.
            _mace_label = (model_size.split("/")[-1]
                           if is_url_mace_model(model_size) else model_size)
            if mace_head:
                _mace_label += f" [head: {mace_head}]"

            if is_multihead_mace(selected_model) and not mace_head:
                calculator_setup_str = f"""
print("\u274c {selected_model} is a multi-head model and needs a prediction head.")
print("   Pick one in the app (MACE settings -> Prediction head, e.g. omat_pbe)")
print("   and generate this script again.")
exit()
"""
            else:
                calculator_setup_str = f"""
print("Setting up MACE calculator...")
{_cq_pre}try:
    calculator = mace_mp(
        {_mace_gpu}{_cq}
    )
    print(f"✅ MACE {_mace_label} initialized on {device}")
except Exception as e:
    print(f"❌ MACE initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = mace_mp(
            {_mace_cpu}
        )
        print("✅ MACE initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ MACE CPU fallback failed: {{cpu_e}}")
        exit()
"""
    elif "SevenNet" in selected_model:
        model_imports += """
try:
    from sevenn.calculator import SevenNetCalculator
except ImportError:
    print("Error: SevenNet not found. Please install with: pip install sevenn")
    exit()
"""
        if is_custom_sevennet:
            # Custom / fine-tuned SevenNet checkpoint (.pth): explicit path, or a
            # single *.pth auto-discovered next to this generated script.
            _csp = custom_sevennet_path or ""
            calculator_setup_str = f"""
print("Setting up custom SevenNet calculator...")
print("  Applying torch.load workaround for SevenNet (allowlisting 'slice')...")
try:
    torch.serialization.add_safe_globals([slice])
except AttributeError:
    print("  ... running on older torch version, add_safe_globals not needed.")
    pass
_explicit_path = r"{_csp}".strip()
if _explicit_path:
    _model_path = _explicit_path
    print(f"📁 Using explicit SevenNet checkpoint path: {{_model_path}}")
else:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _pth_files = sorted(glob.glob(os.path.join(_script_dir, "*.pth")))
    if not _pth_files:
        raise FileNotFoundError(
            "No SevenNet '.pth' checkpoint found next to this script (" + _script_dir + "). "
            "Place your fine-tuned checkpoint here or set an explicit path in the app."
        )
    if len(_pth_files) > 1:
        print("⚠️ Multiple .pth files found; using the first: " + os.path.basename(_pth_files[0]))
    _model_path = _pth_files[0]
    print(f"📁 Auto-discovered SevenNet checkpoint: {{_model_path}}")

if not os.path.exists(_model_path):
    raise FileNotFoundError("SevenNet checkpoint not found: " + _model_path)

{_7net_pre}
original_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.float32)
try:
    calculator = SevenNetCalculator(model=_model_path, device="{device}"{_7net_kw})
    torch.set_default_dtype(original_dtype)
    print(f"✅ Custom SevenNet initialized on {device}")
except Exception as e:
    print(f"❌ Custom SevenNet initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = SevenNetCalculator(model=_model_path, device="cpu")
        torch.set_default_dtype(original_dtype)
        print("✅ Custom SevenNet initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ SevenNet CPU fallback failed: {{cpu_e}}")
        exit()
"""
        else:
            calculator_setup_str = f"""
print("Setting up SevenNet calculator...")
print("  Applying torch.load workaround for SevenNet (allowlisting 'slice')...")
try:
    torch.serialization.add_safe_globals([slice])
except AttributeError:
    print("  ... running on older torch version, add_safe_globals not needed.")
    pass
{_7net_pre}
original_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.float32)
try:
    calculator = SevenNetCalculator(model="{model_size}", device="{device}"{_7net_kw})
    torch.set_default_dtype(original_dtype)
    print(f"✅ SevenNet {model_size} initialized on {device}")
except Exception as e:
    print(f"❌ SevenNet initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = SevenNetCalculator(model="{model_size}", device="cpu")
        torch.set_default_dtype(original_dtype)
        print("✅ SevenNet initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ SevenNet CPU fallback failed: {{cpu_e}}")
        exit()
"""
    elif "ORB" in selected_model:
        model_imports += """
try:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
except ImportError:
    print("Error: ORB models not found. Please install with: pip install orb-models")
    exit()
"""
        precision = "float32-high" if dtype == "float32" else "float32-highest"
        calculator_setup_str = f"""
print("Setting up ORB calculator...")
precision = "{precision}"
try:
    model_func = getattr(pretrained, "{model_size}")
    orbff = model_func(device="{device}", precision=precision)
    calculator = ORBCalculator(orbff, device="{device}")
    print(f"✅ ORB {model_size} initialized on {device}")
except Exception as e:
    print(f"❌ ORB initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        model_func = getattr(pretrained, "{model_size}")
        orbff = model_func(device="cpu", precision=precision)
        calculator = ORBCalculator(orbff, device="cpu")
        print("✅ ORB initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ ORB CPU fallback failed: {{cpu_e}}")
        exit()
"""
    elif selected_model.startswith(("Allegro", "NequIP")):
        calculator_setup_str = f"""
print("Setting up Allegro / NequIP calculator...")
print("First use downloads the model from nequip.net, then it is cached.")
try:
    _model = _nequip_load_saved_model("{model_size}")
    _model.eval()
    _md = _model.metadata
    _type_names = _md["type_names"]
    if isinstance(_type_names, str):
        _type_names = _type_names.split()
    calculator = NequIPCalculator(
        model=_model,
        device="{device}",
        transforms=basic_transforms(
            _md,
            float(_md["r_max"]),
            _type_names,
            handle_chemical_species_map(True, _type_names),
            neighborlist_backend="matscipy",
        ),
    )
    print(f"✅ {selected_model} initialized on {device}")
except NameError:
    print("❌ Allegro / NequIP initialization failed: nequip not found. Is nequip-allegro installed?")
    exit()
except Exception as e:
    print(f"❌ Allegro / NequIP initialization failed on {device}: {{e}}")
    exit()
"""
    elif "Nequix" in selected_model:
        # Nequix runs on JAX and picks its own device; it takes a model *name*
        # from its registry (not a path) and accepts no device argument.
        calculator_setup_str = f"""
print("Setting up Nequix calculator...")
try:
    try:
        calculator = NequixCalculator("{model_size}", use_kernel=True)
    except ImportError:
        # OpenEquivariance kernels are an optional extra; fall back to pure JAX.
        calculator = NequixCalculator("{model_size}", use_kernel=False)
        print("ℹ️ OpenEquivariance kernels unavailable, using pure-JAX path")
    print(f"✅ Nequix {model_size} initialized")
except NameError:
     print(f"❌ Nequix initialization failed: NequixCalculator class not found. Is nequix installed?")
     exit()
except Exception as e:
    print(f"❌ Nequix initialization failed: {{e}}")
    exit()
"""
    elif "DeePMD" in selected_model:
        calculator_setup_str = f"""
print("Setting up DeePMD calculator...")
try:
    calculator = DP(model="{model_size}")
    print(f"✅ DeePMD {model_size} initialized")
except NameError:
     print(f"❌ DeePMD initialization failed: DP class not found. Is deepmd-kit installed?")
     exit()
except Exception as e:
    print(f"❌ DeePMD initialization failed: {{e}}")
    exit()
"""
    elif "UPET" in selected_model:
        model_imports += """
try:
    from upet.calculator import UPETCalculator
except ImportError:
    print("Error: UPET not found. Will fail if UPET model is selected.")
"""
        if model_size.endswith(".ckpt"):
            calculator_setup_str = f"""
print("Setting up custom UPET calculator...")
try:
    calculator = UPETCalculator(
        checkpoint_path="{model_size}",
        device="{device}"
    )
    print(f"✅ Custom UPET initialized on {device}")
except Exception as e:
    print(f"❌ UPET initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = UPETCalculator(
            checkpoint_path="{model_size}",
            device="cpu"
        )
        print("✅ Custom UPET initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ UPET CPU fallback failed: {{cpu_e}}")
        exit()
"""
        else:
            upet_raw = model_size
            if upet_raw.startswith("upet:"):
                upet_raw = upet_raw[len("upet:"):]
            if "::" in upet_raw:
                upet_model_name, upet_version = upet_raw.split("::", 1)
            elif ":" in upet_raw:
                upet_model_name, upet_version = upet_raw.split(":", 1)
            else:
                upet_model_name = upet_raw
                upet_version = "latest"
            calculator_setup_str = f"""
print("Setting up UPET calculator...")
try:
    calculator = UPETCalculator(
        model="{upet_model_name}",
        version="{upet_version}",
        device="{device}"
    )
    print(f"✅ UPET {upet_model_name} v{upet_version} initialized on {device}")
except Exception as e:
    print(f"❌ UPET initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = UPETCalculator(
            model="{upet_model_name}",
            version="{upet_version}",
            device="cpu"
        )
        print("✅ UPET initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ UPET CPU fallback failed: {{cpu_e}}")
        exit()
"""
    elif is_grace:
        if is_custom_grace:
            # Local / fine-tuned GRACE saved-model FOLDER (saved_model.pb + variables/):
            # explicit path, or auto-discovered next to this generated script.
            _cgp = custom_grace_path or ""
            model_imports += """
try:
    from tensorpotential.calculator import TPCalculator
except ImportError:
    print("Error: GRACE (tensorpotential) not found. Please install with: pip install grace-tensorpotential")
    exit()
"""
            calculator_setup_str = f"""
print("Setting up custom GRACE calculator...")
def _complete_grace(d):
    return os.path.exists(os.path.join(d, "saved_model.pb")) and os.path.isdir(os.path.join(d, "variables"))
model_dir = r"{_cgp}".strip()
if not model_dir:
    _here = os.path.dirname(os.path.abspath(__file__))
    _dirs = sorted({{os.path.dirname(p) for p in glob.glob(os.path.join(_here, "**", "saved_model.pb"), recursive=True)}})
    _dirs = [d for d in _dirs if _complete_grace(d)]
    if not _dirs:
        raise FileNotFoundError("No complete GRACE saved-model found next to this script (" + _here + "). A GRACE model is a FOLDER with saved_model.pb AND variables/ - copy the whole model directory, not just saved_model.pb.")
    model_dir = _dirs[0]
    print("Auto-discovered GRACE model: " + model_dir)
else:
    print("Using GRACE model: " + model_dir)
if not _complete_grace(model_dir):
    raise FileNotFoundError("'" + model_dir + "' is not a complete GRACE saved-model (needs saved_model.pb + variables/). Copy the entire model folder, not just saved_model.pb.")
calculator = TPCalculator(model=model_dir)
print("✅ Custom GRACE model initialized successfully")
"""
        else:
            model_imports += """
try:
    from tensorpotential.calculator.foundation_models import grace_fm
except ImportError:
    print("Error: GRACE (tensorpotential) not found. Please install with: pip install grace-tensorpotential")
    exit()
"""
            calculator_setup_str = f"""
print("Setting up GRACE calculator...")
print(f"Model: {model_size}")
print("Note: GRACE uses TensorFlow and auto-detects GPU.")
try:
    calculator = grace_fm("{model_size}")
    print(f"✅ GRACE {model_size} initialized successfully")
except Exception as e:
    print(f"❌ GRACE initialization failed: {{e}}")
    print("   Make sure the model name is correct.")
    print("   You can list available models by running: grace_models list")
    exit()
"""
    else:
        calculator_setup_str = "print('Error: Could not determine calculator type.')\ncalculator = None\nexit()"


    return model_imports, calculator_setup_str
