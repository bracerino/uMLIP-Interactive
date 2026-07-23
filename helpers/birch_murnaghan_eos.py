"""Birch–Murnaghan Equation of State — UI, interactive runner and
standalone-script generator.

The user selects a symmetric range of volume changes (± a percentage of the
equilibrium volume), a step size, and whether the atomic positions should be
relaxed (at fixed cell shape/volume) at every volume point. For each scaled
volume the total energy is evaluated and the resulting E(V) curve is fitted
with the 3rd-order Birch–Murnaghan equation of state, yielding the equilibrium
volume V0, the equilibrium energy E0, the bulk modulus B0 and its pressure
derivative B0'.

This module mirrors the layout of the other calculation-mode helpers (e.g.
``energy_grid_scan.py`` / geometry optimization): it can run inside the
interactive Streamlit interface *and* emit a fully self-contained standalone
Python script that reproduces the calculation outside the GUI.
"""

import os
from datetime import datetime

import numpy as np

try:
    import streamlit as st
except Exception:  # allow importing the pure-python helpers without streamlit
    st = None

# Mirror the main app's online-demo flag (set by online_app.py).
ONLINE_MODE = os.environ.get("MLIP_ONLINE_MODE", "0") == "1"

# eV/Å³ → GPa
_EV_A3_TO_GPA = 160.21766208


DEFAULT_EOS_SETTINGS = {
    "optimize_positions": False,
    "volume_range_pct": 10.0,
    "volume_step_pct": 2.0,
    "optimizer": "LBFGS",
    "fmax": 0.01,
    "max_steps": 300,
    "save_trajectory": True,
}


# ---------------------------------------------------------------------------
# Birch–Murnaghan fit (self-contained, used by both the interactive runner and
# the generated standalone script).
# ---------------------------------------------------------------------------
def birch_murnaghan_energy(volume, e0, v0, b0, b0p):
    """3rd-order Birch–Murnaghan E(V).  ``b0`` is in eV/Å³."""
    volume = np.asarray(volume, dtype=float)
    eta = (v0 / volume) ** (2.0 / 3.0)
    return e0 + (9.0 * v0 * b0 / 16.0) * (
        (eta - 1.0) ** 3 * b0p + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )


def fit_birch_murnaghan(volumes, energies):
    """Fit E(V) with the 3rd-order Birch–Murnaghan EOS.

    Returns a dict with ``V0`` (Å³), ``E0`` (eV), ``B0`` (eV/Å³),
    ``B0_GPa`` (GPa), ``B0_prime`` (dimensionless) and ``rmse`` (eV), or
    ``None`` if the fit could not be performed (too few points / singular
    parabola).
    """
    volumes = np.asarray(volumes, dtype=float)
    energies = np.asarray(energies, dtype=float)

    if volumes.size < 4:
        return None

    # Initial guess from a quadratic fit  E ≈ a V² + b V + c.
    a, b, c = np.polyfit(volumes, energies, 2)
    if abs(a) < 1e-12:
        return None
    v0 = -b / (2.0 * a)
    e0 = np.polyval([a, b, c], v0)
    b0 = 2.0 * a * v0                # B0 = V d²E/dV²  at V0
    b0p = 4.0                        # typical starting value
    if v0 <= 0 or b0 <= 0:
        v0 = float(volumes[np.argmin(energies)])
        b0 = 1.0

    try:
        from scipy.optimize import curve_fit

        popt, _ = curve_fit(
            lambda V, E0, V0, B0, B0p: birch_murnaghan_energy(V, E0, V0, B0, B0p),
            volumes,
            energies,
            p0=[e0, v0, b0, b0p],
            maxfev=100000,
        )
        e0, v0, b0, b0p = (float(x) for x in popt)
    except Exception:
        # scipy missing or fit failed — keep the parabola-based estimate.
        pass

    residuals = energies - birch_murnaghan_energy(volumes, e0, v0, b0, b0p)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "V0": float(v0),
        "E0": float(e0),
        "B0": float(b0),
        "B0_GPa": float(b0 * _EV_A3_TO_GPA),
        "B0_prime": float(b0p),
        "rmse": rmse,
    }


def volume_percent_grid(volume_range_pct, volume_step_pct):
    """Symmetric list of volume-change percentages that always contains 0."""
    rng = abs(float(volume_range_pct))
    step = abs(float(volume_step_pct))
    if step <= 0:
        step = rng if rng > 0 else 1.0
    n = int(round(rng / step))
    pos = [round(i * step, 6) for i in range(1, n + 1)]
    grid = [-p for p in reversed(pos)] + [0.0] + pos
    return grid


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def setup_eos_ui(default_settings=None, save_settings_function=None):
    """Render the EOS parameter UI and return the collected ``eos_params``."""
    default_settings = default_settings or {}
    eos_defaults = default_settings.get("eos_calculation", DEFAULT_EOS_SETTINGS)

    def _g(key, fallback):
        val = eos_defaults.get(key, fallback)
        if isinstance(fallback, bool):
            return bool(val)
        if isinstance(fallback, int) and not isinstance(fallback, bool):
            try:
                return int(val)
            except (TypeError, ValueError):
                return fallback
        if isinstance(fallback, float):
            try:
                return float(val)
            except (TypeError, ValueError):
                return fallback
        return val

    st.subheader("📈 Birch–Murnaghan Equation of State Parameters")
    st.caption(
        "Scales the cell over a symmetric range of volume changes, evaluates "
        "the energy at each volume and fits the 3rd-order Birch–Murnaghan EOS "
        "to obtain the equilibrium volume V₀, bulk modulus B₀ and its pressure "
        "derivative B₀'."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        volume_range_pct = st.number_input(
            "Volume range ± (%)",
            min_value=0.5,
            max_value=50.0,
            value=_g("volume_range_pct", 10.0),
            step=0.5,
            format="%.1f",
            help="The cell volume is varied from −range% to +range% around the "
                 "input (equilibrium) volume.",
        )
    with col2:
        volume_step_pct = st.number_input(
            "Volume step (%)",
            min_value=0.1,
            max_value=25.0,
            value=_g("volume_step_pct", 2.0),
            step=0.1,
            format="%.1f",
            help="Spacing between successive volume points, in % of the input "
                 "volume.",
        )
    with col3:
        optimize_positions = st.checkbox(
            "Optimize atomic positions at each volume",
            value=_g("optimize_positions", False),
            help="If enabled, the atomic positions are relaxed at each volume "
                 "point while the cell shape and volume are held fixed "
                 "(constant-volume relaxation). If disabled, a single-point "
                 "energy is computed after uniformly scaling the coordinates.",
        )

    grid = volume_percent_grid(volume_range_pct, volume_step_pct)
    st.info(
        f"**{len(grid)} volume points** will be evaluated: "
        f"{', '.join(f'{p:+.1f}%' for p in grid)}"
    )
    if len(grid) < 4:
        st.warning(
            "⚠️ The Birch–Murnaghan fit needs at least 4 points. Increase the "
            "range or decrease the step."
        )

    optimizer = "LBFGS"
    fmax = _g("fmax", 0.01)
    max_steps = _g("max_steps", 300)
    save_trajectory = _g("save_trajectory", True)
    if optimize_positions:
        st.markdown("**Position relaxation settings (fixed cell)**")
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            optimizer_options = ["LBFGS", "BFGS", "FIRE", "LBFGSLineSearch", "MDMin"]
            _idx = optimizer_options.index(_g("optimizer", "LBFGS")) \
                if _g("optimizer", "LBFGS") in optimizer_options else 0
            optimizer = st.selectbox(
                "Optimizer", optimizer_options, index=_idx,
                help="ASE optimizer used to relax the atomic positions.",
            )
        with rcol2:
            fmax = st.number_input(
                "Force threshold (eV/Å)",
                min_value=0.001, max_value=1.0, value=float(fmax),
                step=0.005, format="%.3f",
                help="Convergence criterion for the maximum force.",
            )
        with rcol3:
            max_steps = st.number_input(
                "Max steps", min_value=10, max_value=5000,
                value=int(max_steps), step=10,
                help="Maximum optimizer steps per volume point.",
            )
        save_trajectory = st.checkbox(
            "Save relaxation trajectories", value=bool(save_trajectory),
            help="Store the per-volume relaxation trajectory (standalone script "
                 "only).",
        )

    eos_params = {
        "optimize_positions": bool(optimize_positions),
        "volume_range_pct": float(volume_range_pct),
        "volume_step_pct": float(volume_step_pct),
        "optimizer": optimizer,
        "fmax": float(fmax),
        "max_steps": int(max_steps),
        "save_trajectory": bool(save_trajectory),
    }

    if save_settings_function is not None:
        if st.button("💾 Save EOS settings as default", key="save_eos_defaults",
                     disabled=ONLINE_MODE):
            new_settings = dict(default_settings)
            new_settings["eos_calculation"] = eos_params
            if save_settings_function(new_settings):
                st.success("✅ EOS settings saved as default.")

    return eos_params


def display_eos_info(eos_params):
    if st is None:
        return
    grid = volume_percent_grid(
        eos_params.get("volume_range_pct", 10.0),
        eos_params.get("volume_step_pct", 2.0),
    )
    if eos_params.get("optimize_positions"):
        relax = (f"atomic positions relaxed at fixed cell "
                 f"(optimizer={eos_params.get('optimizer', 'LBFGS')}, "
                 f"fmax={eos_params.get('fmax', 0.01)} eV/Å)")
    else:
        relax = "single-point energies (no relaxation)"
    st.info(
        f"EOS will evaluate **{len(grid)} volumes** spanning "
        f"±{eos_params.get('volume_range_pct', 10.0):.1f}% "
        f"(step {eos_params.get('volume_step_pct', 2.0):.1f}%) — {relax}."
    )


# ---------------------------------------------------------------------------
# Interactive runner (called from run_mace_calculation in app.py)
# ---------------------------------------------------------------------------
def _make_optimizer(name, atoms):
    from ase.optimize import BFGS, FIRE, LBFGS, LBFGSLineSearch, MDMin
    n = (name or "LBFGS").upper()
    if n == "BFGS":
        return BFGS(atoms, logfile=None)
    if n == "FIRE":
        return FIRE(atoms, logfile=None)
    if n == "LBFGSLINESEARCH":
        return LBFGSLineSearch(atoms, logfile=None)
    if n == "MDMIN":
        return MDMin(atoms, logfile=None)
    return LBFGS(atoms, logfile=None)


def run_eos_calculation(atoms, calculator, eos_params, log_queue, name, stop_event=None):
    """Run the Birch–Murnaghan EOS scan on a single ASE ``atoms`` object.

    Returns a results dict consumed by the GUI (and auto-backup). ``atoms``
    must already carry the target calculator; it is not modified in place.
    """
    optimize_positions = bool(eos_params.get("optimize_positions", False))
    volume_range_pct = float(eos_params.get("volume_range_pct", 10.0))
    volume_step_pct = float(eos_params.get("volume_step_pct", 2.0))
    optimizer = eos_params.get("optimizer", "LBFGS")
    fmax = float(eos_params.get("fmax", 0.01))
    max_steps = int(eos_params.get("max_steps", 300))

    base = atoms.copy()
    base.calc = calculator
    v_ref = float(base.get_volume())
    natoms = len(base)
    percents = volume_percent_grid(volume_range_pct, volume_step_pct)

    log_queue.put(
        f"📈 Birch–Murnaghan EOS for {name}: {len(percents)} volume points "
        f"(±{volume_range_pct:.1f}%, step {volume_step_pct:.1f}%), "
        f"positions {'relaxed' if optimize_positions else 'fixed'}"
    )
    log_queue.put(f"  Reference volume: {v_ref:.4f} Å³ ({natoms} atoms)")

    volumes, energies, used_pct = [], [], []
    for pct in percents:
        if stop_event is not None and stop_event.is_set():
            log_queue.put("EOS calculation stopped by user")
            return {"success": False, "error": "stopped by user"}

        scale = (1.0 + pct / 100.0) ** (1.0 / 3.0)
        work = base.copy()
        work.calc = calculator
        work.set_cell(base.get_cell() * scale, scale_atoms=True)

        try:
            if optimize_positions:
                opt = _make_optimizer(optimizer, work)
                opt.run(fmax=fmax, steps=max_steps)
            energy = float(work.get_potential_energy())
        except Exception as exc:  # noqa: BLE001
            log_queue.put(f"  ⚠️ Energy evaluation failed at {pct:+.1f}%: {exc}")
            continue

        vol = float(work.get_volume())
        volumes.append(vol)
        energies.append(energy)
        used_pct.append(float(pct))
        log_queue.put(
            f"  {pct:+6.1f}%  V = {vol:9.4f} Å³   E = {energy:14.6f} eV"
        )

    log_queue.put({
        "type": "eos_progress",
        "structure": name,
        "completed": len(volumes),
        "total": len(percents),
    })

    if len(volumes) < 4:
        log_queue.put(f"❌ EOS fit for {name} needs ≥4 successful points "
                      f"(got {len(volumes)})")
        return {
            "success": False,
            "error": "not enough successful volume points for a fit",
            "volumes": volumes,
            "energies": energies,
            "volume_pct": used_pct,
            "natoms": natoms,
        }

    fit = fit_birch_murnaghan(volumes, energies)
    if fit is None:
        log_queue.put(f"❌ Birch–Murnaghan fit failed for {name}")
        return {
            "success": False,
            "error": "fit failed",
            "volumes": volumes,
            "energies": energies,
            "volume_pct": used_pct,
            "natoms": natoms,
        }

    # Dense curve for plotting.
    v_dense = np.linspace(min(volumes), max(volumes), 200)
    e_dense = birch_murnaghan_energy(
        v_dense, fit["E0"], fit["V0"], fit["B0"], fit["B0_prime"]
    )

    log_queue.put(
        f"✅ EOS fit for {name}: V0 = {fit['V0']:.4f} Å³, "
        f"B0 = {fit['B0_GPa']:.2f} GPa, B0' = {fit['B0_prime']:.3f}, "
        f"E0 = {fit['E0']:.6f} eV (RMSE {fit['rmse']*1e3:.3f} meV)"
    )

    return {
        "success": True,
        "optimize_positions": optimize_positions,
        "volume_range_pct": volume_range_pct,
        "volume_step_pct": volume_step_pct,
        "natoms": natoms,
        "volumes": volumes,
        "energies": energies,
        "volume_pct": used_pct,
        "fit_volumes": v_dense.tolist(),
        "fit_energies": e_dense.tolist(),
        "V0": fit["V0"],
        "E0": fit["E0"],
        "B0": fit["B0"],
        "B0_GPa": fit["B0_GPa"],
        "B0_prime": fit["B0_prime"],
        "rmse": fit["rmse"],
        "V0_per_atom": fit["V0"] / natoms if natoms else None,
        "E0_per_atom": fit["E0"] / natoms if natoms else None,
    }


# ---------------------------------------------------------------------------
# Results rendering (Streamlit)
# ---------------------------------------------------------------------------
def render_eos_results(results):
    """Render Birch–Murnaghan EOS results for every structure in ``results``.

    ``results`` is the list of GUI result dicts (each may carry an
    ``eos_results`` entry).
    """
    if st is None:
        return
    import pandas as pd
    import plotly.graph_objects as go

    eos_list = [r for r in results
                if r.get("eos_results") and r["eos_results"].get("success")]
    if not eos_list:
        st.info("No successful Birch–Murnaghan EOS results yet.")
        return

    st.subheader("📈 Birch–Murnaghan Equation of State")

    if len(eos_list) == 1:
        selected = eos_list[0]
        st.write(f"**Structure:** {selected['name']}")
    else:
        names = [r["name"] for r in eos_list]
        chosen = st.selectbox("Select structure:", names, key="eos_selector")
        selected = next(r for r in eos_list if r["name"] == chosen)

    eos = selected["eos_results"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("V₀ (Å³)", f"{eos['V0']:.4f}")
    c2.metric("B₀ (GPa)", f"{eos['B0_GPa']:.2f}")
    c3.metric("B₀'", f"{eos['B0_prime']:.3f}")
    c4.metric("E₀ (eV)", f"{eos['E0']:.6f}")

    if eos.get("V0_per_atom") is not None:
        st.caption(
            f"V₀/atom = {eos['V0_per_atom']:.4f} Å³ · "
            f"E₀/atom = {eos['E0_per_atom']:.6f} eV · "
            f"fit RMSE = {eos['rmse']*1e3:.3f} meV · "
            f"positions {'relaxed' if eos.get('optimize_positions') else 'fixed'}"
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eos["volumes"], y=eos["energies"], mode="markers",
        name="Calculated", marker=dict(size=9, color="#0099ff"),
    ))
    fig.add_trace(go.Scatter(
        x=eos["fit_volumes"], y=eos["fit_energies"], mode="lines",
        name="Birch–Murnaghan fit", line=dict(color="#dc3545", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[eos["V0"]], y=[eos["E0"]], mode="markers",
        name="Equilibrium (V₀, E₀)",
        marker=dict(size=13, color="#28a745", symbol="star"),
    ))
    fig.update_layout(
        xaxis_title="Volume (Å³)",
        yaxis_title="Energy (eV)",
        height=500,
        font=dict(size=15),
        legend=dict(x=0.5, y=1.05, xanchor="center", orientation="h"),
    )
    st.plotly_chart(fig, width="stretch", key=f"eos_plot_{selected['name']}")

    df = pd.DataFrame({
        "Volume change (%)": eos["volume_pct"],
        "Volume (Å³)": eos["volumes"],
        "Energy (eV)": eos["energies"],
    })
    st.dataframe(df, width="stretch", key=f"eos_table_{selected['name']}")

    csv = df.to_csv(index=False)
    st.download_button(
        "💾 Download E(V) data (CSV)", data=csv,
        file_name=f"eos_{selected['name'].replace('.', '_')}.csv",
        mime="text/csv", key=f"eos_dl_{selected['name']}", type="primary",
    )

    if len(eos_list) > 1:
        st.markdown("### Summary across structures")
        summary = pd.DataFrame([{
            "Structure": r["name"],
            "V₀ (Å³)": r["eos_results"]["V0"],
            "B₀ (GPa)": r["eos_results"]["B0_GPa"],
            "B₀'": r["eos_results"]["B0_prime"],
            "E₀ (eV)": r["eos_results"]["E0"],
        } for r in eos_list])
        st.dataframe(summary, width="stretch", key="eos_summary_table")


# ---------------------------------------------------------------------------
# Standalone-script generator
# ---------------------------------------------------------------------------
def generate_eos_script(
    eos_params,
    model_size,
    device,
    dtype,
    thread_count,
    selected_model_key=None,
    mace_head=None,
    mace_dispersion=False,
    mace_dispersion_xc="pbe",
    custom_mace_path=None,
    custom_upet_path=None,
    polar_settings=None,
):
    """Return a fully self-contained Birch–Murnaghan EOS script as a string.

    Calculator setup / MLIP imports are delegated to the project's shared
    generators so every model the GUI supports works unchanged.
    """
    from helpers.generate_python_code import (
        _generate_calculator_setup_code,
        _generate_mlip_imports,
    )

    optimize_positions = bool(eos_params.get("optimize_positions", False))
    volume_range_pct = float(eos_params.get("volume_range_pct", 10.0))
    volume_step_pct = float(eos_params.get("volume_step_pct", 2.0))
    optimizer = eos_params.get("optimizer", "LBFGS")
    fmax = float(eos_params.get("fmax", 0.01))
    max_steps = int(eos_params.get("max_steps", 300))
    save_trajectory = bool(eos_params.get("save_trajectory", True))

    is_mace_polar = (
        selected_model_key is not None and "POLAR" in selected_model_key.upper()
    )

    calculator_setup_code = _generate_calculator_setup_code(
        model_size, device, selected_model_key, dtype,
        mace_head=mace_head,
        mace_dispersion=mace_dispersion,
        mace_dispersion_xc=mace_dispersion_xc,
        custom_mace_path=custom_mace_path,
        custom_upet_path=custom_upet_path,
        polar_settings=polar_settings,
    )
    mlip_imports_code = _generate_mlip_imports()

    grid = volume_percent_grid(volume_range_pct, volume_step_pct)

    config_info = ""
    if mace_head:
        config_info += f"\nMACE Head: {mace_head}"
    if mace_dispersion:
        config_info += f"\nDispersion: D3-{mace_dispersion_xc}"

    relax_desc = (
        f"ON (optimizer={optimizer}, fmax={fmax} eV/Å, max_steps={max_steps})"
        if optimize_positions else "OFF (single-point energies)"
    )

    return f'''#!/usr/bin/env python3
"""Standalone Birch–Murnaghan Equation of State.

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {selected_model_key or model_size}
Device: {device}
Precision: {dtype}{config_info}

EOS settings
------------
  Volume range      : ±{volume_range_pct} %
  Volume step       : {volume_step_pct} %
  Volume points     : {len(grid)}
  Position relax    : {relax_desc}
  Threads (OMP)     : {thread_count}

For every structure file in the run folder the cell is uniformly scaled to a
symmetric set of volumes, the energy is evaluated at each volume (optionally
relaxing the atomic positions at fixed cell), and the E(V) curve is fitted with
the 3rd-order Birch–Murnaghan EOS.

Outputs (per structure, under ``eos_results/<basename>/``):
  * eos_data.csv    — volume-change %, volume (Å³), energy (eV)
  * eos_fit.png     — E(V) points with the Birch–Murnaghan fit
  * eos_summary.txt — V0, E0, B0 (GPa), B0'

Aggregate outputs (directly under ``eos_results/``, when >1 structure is found):
  * eos_summary_all.csv            — one row per structure (index, V0, B0, B0', E0)
  * eos_statistics.txt             — mean/std/min/max/median of B0, V0, B0'
  * bulk_modulus_by_structure.png  — bulk modulus B0 vs structure index
"""

import os
import sys
import glob
import time
from datetime import datetime
from pathlib import Path

# Threading must be set BEFORE torch is imported.
os.environ["OMP_NUM_THREADS"] = "{thread_count}"

import numpy as np

import torch
torch.set_num_threads({thread_count})

from ase.io import read, write
from ase.optimize import BFGS, FIRE, LBFGS, LBFGSLineSearch, MDMin

# ── MLIP imports (matches the GUI's available-models block) ─────────────
{mlip_imports_code}

# ── Parameters chosen in the GUI ────────────────────────────────────────
OPTIMIZE_POSITIONS = {optimize_positions}
VOLUME_RANGE_PCT   = {volume_range_pct}
VOLUME_STEP_PCT    = {volume_step_pct}
OPTIMIZER          = "{optimizer}"
FMAX               = {fmax}
MAX_STEPS          = {max_steps}
SAVE_TRAJECTORY    = {save_trajectory}
IS_MACE_POLAR      = {is_mace_polar}
OUTPUT_ROOT        = Path("eos_results")

EV_A3_TO_GPA = 160.21766208


def volume_percent_grid(rng, step):
    rng = abs(float(rng)); step = abs(float(step))
    if step <= 0:
        step = rng if rng > 0 else 1.0
    n = int(round(rng / step))
    pos = [round(i * step, 6) for i in range(1, n + 1)]
    return [-p for p in reversed(pos)] + [0.0] + pos


def fmt_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return "{{:.1f}} s".format(seconds)
    if seconds < 3600:
        return "{{:.1f}} min".format(seconds / 60.0)
    return "{{:.2f}} h".format(seconds / 3600.0)


def birch_murnaghan_energy(volume, e0, v0, b0, b0p):
    volume = np.asarray(volume, dtype=float)
    eta = (v0 / volume) ** (2.0 / 3.0)
    return e0 + (9.0 * v0 * b0 / 16.0) * (
        (eta - 1.0) ** 3 * b0p + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )


def fit_birch_murnaghan(volumes, energies):
    volumes = np.asarray(volumes, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if volumes.size < 4:
        return None
    a, b, c = np.polyfit(volumes, energies, 2)
    if abs(a) < 1e-12:
        return None
    v0 = -b / (2.0 * a)
    e0 = np.polyval([a, b, c], v0)
    b0 = 2.0 * a * v0
    b0p = 4.0
    if v0 <= 0 or b0 <= 0:
        v0 = float(volumes[np.argmin(energies)]); b0 = 1.0
    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(
            lambda V, E0, V0, B0, B0p: birch_murnaghan_energy(V, E0, V0, B0, B0p),
            volumes, energies, p0=[e0, v0, b0, b0p], maxfev=100000,
        )
        e0, v0, b0, b0p = (float(x) for x in popt)
    except Exception:
        pass
    resid = energies - birch_murnaghan_energy(volumes, e0, v0, b0, b0p)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return dict(V0=float(v0), E0=float(e0), B0=float(b0),
                B0_GPa=float(b0 * EV_A3_TO_GPA), B0_prime=float(b0p), rmse=rmse)


def make_optimizer(name, atoms):
    n = (name or "LBFGS").upper()
    if n == "BFGS":            return BFGS(atoms, logfile=None)
    if n == "FIRE":            return FIRE(atoms, logfile=None)
    if n == "LBFGSLINESEARCH": return LBFGSLineSearch(atoms, logfile=None)
    if n == "MDMIN":           return MDMin(atoms, logfile=None)
    return LBFGS(atoms, logfile=None)


def run_eos_for_structure(fname, calculator, out_dir):
    atoms = read(fname)
    atoms.calc = calculator
    v_ref = float(atoms.get_volume())
    natoms = len(atoms)
    percents = volume_percent_grid(VOLUME_RANGE_PCT, VOLUME_STEP_PCT)
    print(f"  Reference volume: {{v_ref:.4f}} A^3 ({{natoms}} atoms), "
          f"{{len(percents)}} volume points")

    volumes, energies, used_pct = [], [], []
    traj_frames = []
    for pct in percents:
        scale = (1.0 + pct / 100.0) ** (1.0 / 3.0)
        work = atoms.copy()
        work.calc = calculator
        work.set_cell(atoms.get_cell() * scale, scale_atoms=True)
        try:
            if OPTIMIZE_POSITIONS:
                opt = make_optimizer(OPTIMIZER, work)
                opt.run(fmax=FMAX, steps=MAX_STEPS)
            energy = float(work.get_potential_energy())
        except Exception as exc:
            print(f"    WARN: failed at {{pct:+.1f}}%: {{exc}}")
            continue
        vol = float(work.get_volume())
        volumes.append(vol); energies.append(energy); used_pct.append(float(pct))
        if SAVE_TRAJECTORY:
            traj_frames.append(work.copy())
        print(f"    {{pct:+6.1f}}%  V = {{vol:9.4f}} A^3   E = {{energy:14.6f}} eV")

    if len(volumes) < 4:
        print(f"  ERROR: need >=4 points for a fit (got {{len(volumes)}})")
        return None

    fit = fit_birch_murnaghan(volumes, energies)
    if fit is None:
        print("  ERROR: Birch-Murnaghan fit failed")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    with open(out_dir / "eos_data.csv", "w") as fh:
        fh.write("volume_change_pct,volume_A3,energy_eV\\n")
        for p, v, e in zip(used_pct, volumes, energies):
            fh.write(f"{{p}},{{v}},{{e}}\\n")

    # Summary
    with open(out_dir / "eos_summary.txt", "w") as fh:
        fh.write("Birch-Murnaghan Equation of State\\n")
        fh.write("=================================\\n")
        fh.write(f"Structure         : {{fname}}\\n")
        fh.write(f"Atoms             : {{natoms}}\\n")
        fh.write(f"Positions relaxed : {{OPTIMIZE_POSITIONS}}\\n\\n")
        fh.write(f"V0  (equilibrium volume) : {{fit['V0']:.6f}} A^3 "
                 f"({{fit['V0']/natoms:.6f}} A^3/atom)\\n")
        fh.write(f"E0  (equilibrium energy) : {{fit['E0']:.6f}} eV "
                 f"({{fit['E0']/natoms:.6f}} eV/atom)\\n")
        fh.write(f"B0  (bulk modulus)       : {{fit['B0_GPa']:.4f}} GPa\\n")
        fh.write(f"B0' (pressure deriv.)    : {{fit['B0_prime']:.4f}}\\n")
        fh.write(f"fit RMSE                 : {{fit['rmse']*1e3:.4f}} meV\\n")

    # Trajectory of all volume points
    if SAVE_TRAJECTORY and traj_frames:
        try:
            write(str(out_dir / "eos_volumes.xyz"), traj_frames)
        except Exception as exc:
            print(f"  WARN: could not write trajectory: {{exc}}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        v_dense = np.linspace(min(volumes), max(volumes), 200)
        e_dense = birch_murnaghan_energy(
            v_dense, fit["E0"], fit["V0"], fit["B0"], fit["B0_prime"])
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.plot(volumes, energies, "o", color="#0099ff", markersize=8,
                label="Calculated")
        ax.plot(v_dense, e_dense, "-", color="#dc3545", linewidth=2,
                label="Birch-Murnaghan fit")
        ax.set_xlabel("Volume (Å³)", fontsize=16)
        ax.set_ylabel("Energy (eV)", fontsize=16)
        ax.set_title("{{}}: B0 = {{:.1f}} GPa, B0' = {{:.2f}}".format(
            fname, fit["B0_GPa"], fit["B0_prime"]), fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.legend(fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / "eos_fit.png", dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"  WARN: could not create plot: {{exc}}")

    print(f"  V0 = {{fit['V0']:.4f}} A^3, B0 = {{fit['B0_GPa']:.2f}} GPa, "
          f"B0' = {{fit['B0_prime']:.3f}}, E0 = {{fit['E0']:.6f}} eV")
    return fit


def main():
    start = time.time()
    print("Birch-Murnaghan Equation of State")
    print(f"  {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
    print(f"  Model: {selected_model_key or model_size}  (device {device})")
    print(f"  Volume range: +/-{{VOLUME_RANGE_PCT}}%  step {{VOLUME_STEP_PCT}}%")
    print(f"  Position relaxation: {{'ON' if OPTIMIZE_POSITIONS else 'OFF'}}")
    print(f"  OMP threads: {{os.environ.get('OMP_NUM_THREADS')}}")

    print("\\nSetting up MLIP calculator...")
{calculator_setup_code}

    structure_files = sorted([
        f for f in os.listdir(".")
        if f.startswith("POSCAR") or f.endswith(".vasp")
        or f.endswith(".poscar") or f.endswith(".cif")
    ])
    if not structure_files:
        print("ERROR: No structure files (.cif / .vasp / POSCAR*) found")
        sys.exit(1)
    print(f"\\nFound {{len(structure_files)}} structure(s): {{structure_files}}")

    OUTPUT_ROOT.mkdir(exist_ok=True)
    summary = []
    n_total = len(structure_files)
    durations = []          # wall-clock seconds per structure
    ETA_WINDOW = 10         # average over the latest N structures for the ETA
    for i, fname in enumerate(structure_files):
        print("\\n" + "=" * 64)
        print("Processing [{{}}/{{}}] {{}}".format(i + 1, n_total, fname))

        t_struct = time.time()
        base = os.path.splitext(os.path.basename(fname))[0]
        try:
            fit = run_eos_for_structure(fname, calculator, OUTPUT_ROOT / base)
        except Exception as exc:
            print("  ERROR processing {{}}: {{}}".format(fname, exc))
            fit = None
        if fit is not None:
            summary.append((fname, fit))

        dt = time.time() - t_struct
        durations.append(dt)
        recent = durations[-ETA_WINDOW:]
        avg = sum(recent) / len(recent)
        n_left = n_total - (i + 1)
        eta = avg * n_left
        print("  Done in {{}}  |  avg/structure (last {{}}): {{}}  |  "
              "{{}} left  |  est. remaining: {{}}".format(
                  fmt_duration(dt), len(recent), fmt_duration(avg),
                  n_left, fmt_duration(eta)))

    if summary:
        print("\\n==== Summary ====")
        header = "{{:<4s}} {{:<28s}} {{:>12s}} {{:>12s}} {{:>10s}} {{:>14s}}".format(
            "idx", "Structure", "V0 (A^3)", "B0 (GPa)", "B0prime", "E0 (eV)")
        print(header)
        for idx, (fname, fit) in enumerate(summary):
            print("{{:<4d}} {{:<28s}} {{:>12.4f}} {{:>12.3f}} {{:>10.3f}} {{:>14.6f}}".format(
                idx, fname, fit["V0"], fit["B0_GPa"], fit["B0_prime"], fit["E0"]))

        # Aggregate CSV across all structures (one row per structure).
        with open(OUTPUT_ROOT / "eos_summary_all.csv", "w") as fh:
            fh.write("index,structure,V0_A3,B0_GPa,B0_prime,E0_eV\\n")
            for idx, (fname, fit) in enumerate(summary):
                fh.write("{{}},{{}},{{}},{{}},{{}},{{}}\\n".format(
                    idx, fname, fit["V0"], fit["B0_GPa"],
                    fit["B0_prime"], fit["E0"]))

        # Statistics across structures (most useful with >1 structure).
        b0_vals = np.array([fit["B0_GPa"] for _, fit in summary], dtype=float)
        v0_vals = np.array([fit["V0"] for _, fit in summary], dtype=float)
        bp_vals = np.array([fit["B0_prime"] for _, fit in summary], dtype=float)

        def _std(arr):
            return float(arr.std(ddof=1)) if arr.size > 1 else 0.0

        with open(OUTPUT_ROOT / "eos_statistics.txt", "w") as fh:
            fh.write("Birch-Murnaghan EOS - statistics across structures\\n")
            fh.write("==================================================\\n")
            fh.write("Structures fitted : {{}}\\n\\n".format(len(summary)))

            def _stat(name, arr, unit):
                fh.write("{{}}:\\n".format(name))
                fh.write("  mean   = {{:.4f}} {{}}\\n".format(arr.mean(), unit))
                fh.write("  std    = {{:.4f}} {{}}\\n".format(_std(arr), unit))
                fh.write("  min    = {{:.4f}} {{}}\\n".format(arr.min(), unit))
                fh.write("  max    = {{:.4f}} {{}}\\n".format(arr.max(), unit))
                fh.write("  median = {{:.4f}} {{}}\\n\\n".format(np.median(arr), unit))

            _stat("Bulk modulus B0", b0_vals, "GPa")
            _stat("Equilibrium volume V0", v0_vals, "A^3")
            _stat("Pressure derivative B0'", bp_vals, "")
            fh.write("Structure index legend:\\n")
            for idx, (fname, _f) in enumerate(summary):
                fh.write("  {{:>3d}}  {{}}\\n".format(idx, fname))

        # Chart: bulk modulus vs structure index (one index per structure).
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator
            idx_arr = np.arange(len(summary))
            fig, ax = plt.subplots(figsize=(8.0, 5.0))
            ax.scatter(idx_arr, b0_vals, color="#0099ff", s=60,
                       edgecolors="#005c99", zorder=3)
            mean_b0 = float(b0_vals.mean())
            ax.axhline(mean_b0, color="#dc3545", linestyle="--",
                       label="mean = {{:.1f}} GPa".format(mean_b0))
            if b0_vals.size > 1:
                ax.axhspan(mean_b0 - _std(b0_vals), mean_b0 + _std(b0_vals),
                           color="#dc3545", alpha=0.10, label="+/- 1 std")
            ax.set_xlabel("Structure index", fontsize=16)
            ax.set_ylabel("Bulk modulus B0 (GPa)", fontsize=16)
            ax.set_title("Bulk modulus per structure", fontsize=17)
            ax.tick_params(axis="both", which="major", labelsize=14)
            # Only a handful of integer ticks spread across the index range,
            # so the axis stays readable for any number of structures.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
            ax.margins(x=0.05)
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=13)
            plt.tight_layout()
            plt.savefig(OUTPUT_ROOT / "bulk_modulus_by_structure.png", dpi=200)
            plt.close()
            print("Saved chart: {{}}".format(
                OUTPUT_ROOT / "bulk_modulus_by_structure.png"))
        except Exception as exc:
            print("  WARN: could not create bulk-modulus chart: {{}}".format(exc))

        print("\\nStructure index legend:")
        for idx, (fname, _f) in enumerate(summary):
            print("  {{:>3d}}  {{}}".format(idx, fname))

    print("\\nDone in {{:.1f}} s. Results in '{{}}/'.".format(
        time.time() - start, OUTPUT_ROOT))

    # Final console line: average and std of the bulk modulus over all
    # successfully-fitted structures.
    if summary:
        b0_all = np.array([fit["B0_GPa"] for _, fit in summary], dtype=float)
        std_all = float(b0_all.std(ddof=1)) if b0_all.size > 1 else 0.0
        print("\\nAverage bulk modulus B0 over {{}} structure(s): "
              "{{:.3f}} +/- {{:.3f}} GPa".format(
                  b0_all.size, float(b0_all.mean()), std_all))


if __name__ == "__main__":
    main()
'''
