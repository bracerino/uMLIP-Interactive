"""Grimme D3 dispersion for SevenNet.

SevenNet ships its own CUDA D3 kernel (``libpaird3.so``, built by nvcc when the
package is installed) and exposes it as ``sevenn.calculator.D3Calculator``.
``SevenNetD3Calculator`` is simply ``SumCalculator([SevenNetCalculator,
D3Calculator])``; this module adds the D3 term the same way, on top of whichever
SevenNet calculator a generator has already built. That matters because those
constructions differ a lot between script types (custom checkpoint discovery,
``modal=`` for the multi-fidelity models, cuEquivariance kwargs, CPU
fallbacks) — bolting the correction on afterwards keeps one code path instead
of rewriting seven.

The one hard constraint: **SevenNet's D3 is CUDA-only.** ``D3Calculator``
raises ``NotImplementedError("CPU + D3 is not implemented yet")``. Since most of
this app runs on CPU, the emitted code falls back to ``torch-dftd`` — the same
D3 physics, a different implementation — and says so out loud rather than
quietly changing what was computed. If neither backend works the run stops:
a calculation that was asked for a dispersion correction and silently did
without one is worse than no calculation.

Settings reach the script generators through the module-level cache below, the
same contract ``helpers.quantum_espresso`` and ``helpers.uma_models`` use, so
no generator signature has to grow three more arguments.
"""

import streamlit as st

# Functional names, spelled as the original DFT-D3 Fortran code spells them
# (both SevenNet's CUDA table and torch-dftd follow that convention — note
# "b-lyp", not "blyp"). Restricted to names present in SevenNet's BJ table.
# The flag says whether torch-dftd also knows the name, i.e. whether the CPU
# fallback can work.
SEVENNET_D3_FUNCTIONALS = {
    "pbe":    True,   # matches the PBE(+U) data most SevenNet models are trained on
    "pbesol": True,
    "revpbe": True,
    "rpbe":   True,
    "r2scan": True,
    "hse06":  True,
    "pbe0":   True,
    "tpss":   True,
    "b3-lyp": True,
    "b-lyp":  True,
    "wb97m":  False,  # in SevenNet's table, absent from torch-dftd → CUDA only
}

# D3Calculator validates against exactly these two, even though the CUDA kernel
# also carries damp_zerom / damp_bjm.
SEVENNET_D3_DAMPINGS = {
    "damp_bj":   "Becke-Johnson damping (D3-BJ) — the usual choice",
    "damp_zero": "Zero damping (D3-0)",
}

# torch-dftd spells the damping without the prefix.
_TORCH_DFTD_DAMPING = {"damp_bj": "bj", "damp_zero": "zero"}

SEVENNET_D3_DEFAULTS = {
    "enabled": False,
    "functional_name": "pbe",
    "damping_type": "damp_bj",
}


# ---------------------------------------------------------------------------
# Active-settings cache (same contract as helpers.quantum_espresso)
# ---------------------------------------------------------------------------
_ACTIVE_D3_SETTINGS = None


def set_active_sevennet_d3_settings(settings):
    """Remember the sidebar's SevenNet D3 settings for the script generators."""
    global _ACTIVE_D3_SETTINGS
    _ACTIVE_D3_SETTINGS = dict(settings) if settings else None


def get_active_sevennet_d3_settings(settings=None):
    merged = dict(SEVENNET_D3_DEFAULTS)
    if _ACTIVE_D3_SETTINGS:
        merged.update(_ACTIVE_D3_SETTINGS)
    if settings:
        merged.update(settings)
    return merged


def sevennet_d3_enabled(settings=None):
    return bool(get_active_sevennet_d3_settings(settings).get("enabled"))


def sevennet_d3_summary(settings=None):
    s = get_active_sevennet_d3_settings(settings)
    if not s.get("enabled"):
        return "off"
    return f"{s['functional_name']} / {s['damping_type']}"


# ---------------------------------------------------------------------------
# Sidebar panel
# ---------------------------------------------------------------------------
def setup_sevennet_d3_ui(device="cpu", default_settings=None):
    """Draw the SevenNet dispersion options and return the settings dict."""
    saved = dict(SEVENNET_D3_DEFAULTS)
    saved.update((default_settings or {}).get("sevennet_d3", {}) or {})

    st.markdown("---")
    st.subheader("🌊 D3 Dispersion Correction (SevenNet)")

    enabled = st.checkbox(
        "Enable Grimme D3 dispersion",
        value=bool(saved.get("enabled", False)),
        key="sevennet_d3_enable",
        help=(
            "Adds a D3 van der Waals term to SevenNet's energy, forces and "
            "stress. On a GPU this uses SevenNet's own CUDA D3 kernel "
            "(sevenn.calculator.D3Calculator, the same thing "
            "SevenNetD3Calculator wraps); on CPU it falls back to torch-dftd, "
            "because SevenNet's D3 is GPU-only."
        ),
    )

    functional = saved.get("functional_name", "pbe")
    damping = saved.get("damping_type", "damp_bj")

    if enabled:
        _names = list(SEVENNET_D3_FUNCTIONALS)
        col_a, col_b = st.columns(2)
        with col_a:
            functional = st.selectbox(
                "Functional (D3 parameter set)",
                _names,
                index=_names.index(functional) if functional in _names else 0,
                key="sevennet_d3_functional",
                help=(
                    "Picks the D3 parameters, so it should match the DFT level "
                    "the model was trained against — PBE for the MPtrj / "
                    "OMat24 checkpoints. Names follow the original DFT-D3 "
                    "spelling ('b-lyp', not 'blyp')."
                ),
            )
        with col_b:
            _damps = list(SEVENNET_D3_DAMPINGS)
            damping = st.selectbox(
                "Damping",
                _damps,
                index=_damps.index(damping) if damping in _damps else 0,
                format_func=lambda d: SEVENNET_D3_DAMPINGS[d],
                key="sevennet_d3_damping",
            )

        # Only the CPU case needs saying: there the backend actually changes.
        if device != "cuda":
            st.warning(
                "⚠️ CPU run — SevenNet's D3 is **GPU-only** "
                "(`CPU + D3 is not implemented yet`), so the calculation will "
                "use **torch-dftd** instead. Same D3 correction, different "
                "implementation; install it with `pip install torch-dftd`."
            )
            if not SEVENNET_D3_FUNCTIONALS.get(functional, True):
                st.error(
                    f"❌ `{functional}` exists only in SevenNet's own D3 table, "
                    "not in torch-dftd — it cannot be used on CPU. Pick "
                    "another functional or run on a GPU."
                )

    settings = {
        "enabled": bool(enabled),
        "functional_name": functional,
        "damping_type": damping,
    }
    if default_settings is not None:
        default_settings["sevennet_d3"] = dict(settings)
    set_active_sevennet_d3_settings(settings)
    return settings


# ---------------------------------------------------------------------------
# In-app calculator
# ---------------------------------------------------------------------------
def attach_sevennet_d3(calculator, device="cpu", settings=None, log=None):
    """Return ``calculator`` with a D3 term summed in, or raise.

    Raising rather than returning the bare calculator is deliberate: a run that
    was asked for dispersion and quietly produced uncorrected numbers is a
    silent wrong answer.
    """
    s = get_active_sevennet_d3_settings(settings)
    if not s.get("enabled"):
        return calculator

    func = s["functional_name"]
    damp = s["damping_type"]
    from ase.calculators.mixing import SumCalculator

    try:
        from sevenn.calculator import D3Calculator
        d3 = D3Calculator(damping_type=damp, functional_name=func)
        if log:
            log(f"  ✅ D3 dispersion: SevenNet native CUDA kernel ({func}, {damp})")
        return SumCalculator([calculator, d3])
    except Exception as native_err:
        if log:
            log(f"  ℹ️ SevenNet's CUDA D3 is unavailable "
                f"({str(native_err).splitlines()[0][:200]}) — "
                f"falling back to torch-dftd")
        try:
            from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
            d3 = TorchDFTD3Calculator(
                device=device, xc=func, damping=_TORCH_DFTD_DAMPING.get(damp, "bj"),
            )
            if log:
                log(f"  ✅ D3 dispersion: torch-dftd ({func}, "
                    f"{_TORCH_DFTD_DAMPING.get(damp, 'bj')}) on {device}")
            return SumCalculator([calculator, d3])
        except Exception as fallback_err:
            raise RuntimeError(
                f"D3 dispersion was requested but no backend could provide it. "
                f"SevenNet's CUDA kernel said: {native_err}. torch-dftd said: "
                f"{fallback_err}. Install torch-dftd (pip install torch-dftd), "
                f"pick a functional it supports, or switch the correction off."
            ) from fallback_err


# ---------------------------------------------------------------------------
# Standalone-script code
# ---------------------------------------------------------------------------
def sevennet_d3_code(device="cpu", indent="", settings=None):
    """Return the block that sums a D3 term onto an existing ``calculator``.

    Empty string when the correction is off, so callers can splice it in
    unconditionally.
    """
    s = get_active_sevennet_d3_settings(settings)
    if not s.get("enabled"):
        return ""

    func = s["functional_name"]
    damp = s["damping_type"]
    td_damp = _TORCH_DFTD_DAMPING.get(damp, "bj")

    lines = [
        "",
        "# --- Grimme D3 dispersion for SevenNet -------------------------------",
        "# SevenNet's own D3 is a CUDA kernel and is GPU-only (D3Calculator",
        '# raises "CPU + D3 is not implemented yet"), so try it first and fall',
        "# back to torch-dftd, which is the same D3 correction on the CPU.",
        "# Summing a D3Calculator onto the SevenNet calculator is exactly what",
        "# SevenNetD3Calculator does internally.",
        "from ase.calculators.mixing import SumCalculator as _D3SumCalculator",
        f'_D3_FUNCTIONAL = "{func}"',
        f'_D3_DAMPING = "{damp}"',
        "try:",
        "    from sevenn.calculator import D3Calculator as _SevenNetD3",
        "    calculator = _D3SumCalculator([",
        "        calculator,",
        "        _SevenNetD3(damping_type=_D3_DAMPING, functional_name=_D3_FUNCTIONAL),",
        "    ])",
        '    print(f"✅ D3 dispersion: SevenNet native CUDA kernel '
        '({_D3_FUNCTIONAL}, {_D3_DAMPING})")',
        "except Exception as _d3_native_err:",
        "    _d3_why = str(_d3_native_err).splitlines()[0][:200]",
        '    print(f"ℹ️  SevenNet CUDA D3 unavailable: {_d3_why}")',
        '    print("   Falling back to torch-dftd (same D3 correction, CPU-capable).")',
        "    try:",
        "        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator",
        "        calculator = _D3SumCalculator([",
        "            calculator,",
        f'            TorchDFTD3Calculator(device="{device}", xc=_D3_FUNCTIONAL,',
        f'                                 damping="{td_damp}"),',
        "        ])",
        f'        print(f"✅ D3 dispersion: torch-dftd ({{_D3_FUNCTIONAL}}, '
        f'{td_damp}) on {device}")',
        "    except Exception as _d3_fallback_err:",
        '        print(f"❌ D3 dispersion was requested but no backend could '
        'provide it.")',
        '        print(f"   SevenNet CUDA kernel: {_d3_why}")',
        '        print(f"   torch-dftd:           {_d3_fallback_err}")',
        '        print("   Install torch-dftd (pip install torch-dftd), pick a '
        'functional")',
        '        print("   it supports, or switch the correction off and '
        're-generate.")',
        '        print("   Stopping rather than reporting uncorrected energies '
        'as D3.")',
        "        raise SystemExit(1)",
        "",
    ]
    return "\n".join(indent + line if line else "" for line in lines) + "\n"


def sevennet_d3_preview(device="cpu", settings=None):
    """A short, readable version of the D3 block for the on-screen preview.

    The preview is meant to be skimmed, so it shows the two lines that matter
    rather than the generated script's full fallback machinery.
    """
    s = get_active_sevennet_d3_settings(settings)
    if not s.get("enabled"):
        return ""
    func, damp = s["functional_name"], s["damping_type"]
    td_damp = _TORCH_DFTD_DAMPING.get(damp, "bj")
    if device == "cuda":
        return (
            f"\n\n# D3 dispersion — SevenNet's own CUDA kernel\n"
            f"from sevenn.calculator import D3Calculator\n"
            f"from ase.calculators.mixing import SumCalculator\n"
            f"calculator = SumCalculator([calculator, D3Calculator(\n"
            f'    damping_type="{damp}", functional_name="{func}")])'
        )
    return (
        f"\n\n# D3 dispersion — SevenNet's D3 is GPU-only, so on CPU this\n"
        f"# falls back to torch-dftd (same correction, other implementation)\n"
        f"from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator\n"
        f"from ase.calculators.mixing import SumCalculator\n"
        f"calculator = SumCalculator([calculator, TorchDFTD3Calculator(\n"
        f'    device="{device}", xc="{func}", damping="{td_damp}")])'
    )
