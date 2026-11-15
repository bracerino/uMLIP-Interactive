from __future__ import annotations

import json
import queue
import threading
import time
import traceback

import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    from upet.calculator import PETMADDOSCalculator
    PETMAD_DOS_AVAILABLE = True
except ImportError:
    PETMAD_DOS_AVAILABLE = False


def compute_petmad_dos(calc, atoms, per_atom=False):
    """DOS, Fermi level and band gap of one structure from a PETMADDOSCalculator.

    Returns (energies, dos, dos_total, fermi, bandgap) as numpy arrays / floats;
    ``dos`` is per atom (n_atoms x n_energies) when ``per_atom`` is set.
    upet 0.2.5 replaced calculate_dos / calculate_efermi / calculate_bandgap
    by a single calculate() that returns no energy grid; its DOS lies on a grid
    starting at 0 eV with a step of ENERGY_INTERVAL. Both APIs are supported.
    """
    import numpy as np

    def to_np(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        return np.asarray(value, dtype=float)

    if hasattr(calc, "calculate_dos"):  # upet <= 0.2.1
        energies, dos = calc.calculate_dos(atoms, per_atom=per_atom)
        dos_total = calc.calculate_dos(atoms, per_atom=False)[1] if per_atom else dos
        fermi = calc.calculate_efermi(atoms)
        bandgap = calc.calculate_bandgap(atoms, dos=dos_total)
        energies = to_np(energies)
    else:
        try:
            from upet.calculator import ENERGY_INTERVAL
        except ImportError:
            ENERGY_INTERVAL = 0.05
        properties = ["dos_raw", "fermi_level", "bandgap"]
        if per_atom:
            properties.append("dos_raw_per_atom")
        results = calc.calculate(atoms, properties=properties)
        dos_total = results["dos_raw"]
        dos = results["dos_raw_per_atom"] if per_atom else dos_total
        fermi = results["fermi_level"]
        bandgap = results["bandgap"]
        energies = np.arange(to_np(dos_total).shape[-1]) * ENERGY_INTERVAL
    return (energies, to_np(dos).squeeze(), to_np(dos_total).squeeze(),
            float(to_np(fermi)), float(to_np(bandgap)))


def render_petmad_dos_panel(structures: dict) -> None:
    st.markdown("## ⚛️ Electronic DOS — PET-MAD-DOS")

    st.info(
        "PET-MAD-DOS predicts the **electronic density of states**, "
        "**Fermi level**, and **band gap** directly from the crystal "
        "structure using an ML model trained on the MAD dataset."
    )

    with st.expander("⚙️ Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            version = st.selectbox("Model version", ["latest"], index=0, key="dos_version")
            device  = st.radio("Device", ["cpu"], index=0, horizontal=True, key="dos_device")
        with col2:
            per_atom = st.checkbox(
                "Per-atom DOS", value=False, key="dos_per_atom",
                help="Compute the DOS contribution from each individual atom. After calculation you can select specific atoms to plot.",
            )
            plot_mode = st.radio(
                "Plot mode",
                ["Raw (no broadening)", "Gaussian broadened"],
                index=1,
                key="dos_plot_mode",
                help="'Raw' plots the model output directly. 'Gaussian broadened' applies a smoothing kernel for a cleaner curve.",
            )
        with col3:
            n_dos_points = st.number_input(
                "DOS grid points", min_value=100, max_value=5000, value=1000, step=100, key="dos_n_points",
                help="Only used in broadened mode.",
            )
            dos_sigma = st.number_input(
                "Broadening σ (eV)", min_value=0.01, max_value=1.0, value=0.05, step=0.01,
                format="%.2f", key="dos_sigma",
                help="Gaussian broadening σ. Only used in broadened mode. Small negative values near zero are a known broadening artefact.",
            )
            e_window = st.number_input(
                "Energy window around E_F (eV)", min_value=1.0, max_value=50.0, value=10.0, step=0.5, key="dos_e_window",
            )

    dos_params = {
        "version":      version,
        "device":       device,
        "per_atom":     per_atom,
        "plot_mode":    plot_mode,
        "n_dos_points": int(n_dos_points),
        "dos_sigma":    float(dos_sigma),
        "e_window":     float(e_window),
    }
    # Also read by the "Generate Python Script" buttons in the Start Calculations tab.
    st.session_state["petmad_dos_params"] = dos_params

    st.markdown("#### 📝 Standalone Python script")
    st.caption("Runs the DOS calculation with the settings above outside the app, "
               "e.g. on a cluster or without the web interface.")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        script_embedded = st.button(
            "📝 Generate Python Script (embeds the uploaded structures)",
            type="tertiary", width="stretch", key="dos_script_embedded_btn",
            disabled=not structures,
        )
    with col_s2:
        script_local = st.button(
            "📂 Generate Python Script (uses structure files placed next to the script)",
            type="secondary", width="stretch", key="dos_script_local_btn",
        )
    if script_embedded or script_local:
        render_dos_script(
            generate_petmad_dos_script(
                dos_params,
                thread_count=st.session_state.get("thread_count", 4),
                structures=structures if script_embedded else None,
            ),
            embedded=script_embedded,
        )
    st.markdown("---")

    # The settings stay available without upet or structures, so the script
    # can still be generated.
    if not PETMAD_DOS_AVAILABLE:
        st.error("**upet is not installed** in this environment, so the calculation cannot "
                 "run here.  \nRun `pip install upet` (or use `requirements-upet.txt`) and "
                 "restart the app — or generate the standalone script.")
        return

    if not structures:
        st.warning("⚠️ Please upload and lock at least one structure first.")
        return

    _init_state()

    if st.session_state._dos_running:
        col_b1, col_b2 = st.columns([3, 1])
        with col_b1:
            start = st.button(
                "🚀  Start PET-MAD-DOS Calculation",
                type="primary", disabled=True,
                width='stretch', key="dos_start_btn",
            )
        with col_b2:
            if st.button("🛑 Stop", key="dos_stop_btn", width='stretch'):
                st.session_state._dos_stop_event.set()
    else:
        start = st.button(
            "🚀  Start PET-MAD-DOS Calculation",
            type="primary",
            width='stretch', key="dos_start_btn",
        )

    if start:
        st.session_state._dos_results    = []
        st.session_state._dos_log        = []
        st.session_state._dos_running    = True
        st.session_state._dos_done       = False
        st.session_state._dos_params     = dos_params
        st.session_state._dos_stop_event = threading.Event()
        q = queue.Queue()
        st.session_state._dos_queue = q

        threading.Thread(
            target=_run_calculation,
            args=(structures, dos_params, q, st.session_state._dos_stop_event),
            daemon=True,
        ).start()
        st.rerun()

    was_running = st.session_state._dos_running
    if st.session_state._dos_running or st.session_state._dos_done:
        _drain_queue()

    if was_running and not st.session_state._dos_running:
        st.rerun()

    if st.session_state._dos_log:
        with st.expander("📋 Live log", expanded=st.session_state._dos_running):
            st.text_area(
                "log", "\n".join(st.session_state._dos_log[-80:]),
                height=220, key="dos_log_area", label_visibility="collapsed",
            )

    if st.session_state._dos_running:
        done  = len(st.session_state._dos_results)
        total = len(structures)
        st.progress(done / total, text=f"Processing … {done}/{total} structures")
        time.sleep(0.8)
        st.rerun()
    elif st.session_state._dos_done and st.session_state._dos_results:
        st.success("✅ Calculation complete!")

    if st.session_state._dos_results:
        _display_results(st.session_state._dos_results, st.session_state._dos_params)


def _init_state() -> None:
    defaults = {
        "_dos_running":    False,
        "_dos_done":       False,
        "_dos_results":    [],
        "_dos_log":        [],
        "_dos_params":     {},
        "_dos_queue":      None,
        "_dos_stop_event": threading.Event(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _drain_queue() -> None:
    q = st.session_state.get("_dos_queue")
    if q is None:
        return
    while True:
        try:
            msg = q.get_nowait()
        except queue.Empty:
            break
        if isinstance(msg, str):
            if msg == "_DOS_DONE_":
                st.session_state._dos_running = False
                st.session_state._dos_done    = True
            else:
                st.session_state._dos_log.append(msg)
        elif isinstance(msg, dict):
            st.session_state._dos_results.append(msg)


def _run_calculation(structures, params, q, stop):
    def log(msg):
        q.put(msg)

    device   = params["device"]
    version  = params["version"]
    per_atom = params.get("per_atom", False)
    total    = len(structures)

    try:
        import torch
        log(f"⏳ Loading calculator  version={version}  device={device} …")
        calc = PETMADDOSCalculator(version=version, device=device)
        if device == "cuda" and torch.cuda.is_available():
            if hasattr(calc, "model") and hasattr(calc.model, "to"):
                calc.model.to(device)
            if hasattr(calc, "to"):
                calc.to(device)
        log("✅ Calculator ready.")
    except Exception as exc:
        log(f"❌ Failed to load calculator: {exc}")
        q.put("_DOS_DONE_")
        return

    for idx, (name, structure) in enumerate(structures.items()):
        if stop.is_set():
            log("⚠️ Stopped by user.")
            break

        log(f"\n[{idx+1}/{total}]  {name}")
        t0 = time.time()

        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            atoms = AseAtomsAdaptor.get_atoms(structure)

            log(f"  → DOS, Fermi level and band gap (per_atom={per_atom}) …")
            energies, dos, _, fermi, bandgap = compute_petmad_dos(calc, atoms, per_atom)
            if per_atom and dos.ndim == 1:  # a single atom squeezes to 1-D
                dos = dos[None, :]

            dt = time.time() - t0
            log(f"  ✅ {dt:.1f}s | E_F = {fermi:.3f} eV | gap = {bandgap:.3f} eV")

            atom_labels = []
            if per_atom:
                symbols = atoms.get_chemical_symbols()
                counts  = {}
                for sym in symbols:
                    counts[sym] = counts.get(sym, 0) + 1
                    atom_labels.append(f"{sym}{counts[sym]}  (atom {len(atom_labels)+1})")

            q.put({
                "type":        "petmad_dos_result",
                "name":        name,
                "success":     True,
                "energies":    np.asarray(energies).tolist(),
                "dos":         np.asarray(dos).tolist(),
                "per_atom":    per_atom,
                "atom_labels": atom_labels,
                "fermi_level": fermi,
                "band_gap":    bandgap,
                "duration":    dt,
                "n_atoms":     len(atoms),
                "formula":     structure.composition.reduced_formula,
            })

        except Exception as exc:
            dt = time.time() - t0
            log(f"  ❌ {exc}")
            log(traceback.format_exc())
            q.put({
                "type":     "petmad_dos_result",
                "name":     name,
                "success":  False,
                "error":    str(exc),
                "duration": dt,
            })

    q.put("_DOS_DONE_")


FONT        = dict(size=20)
AXIS_FONT   = dict(size=20)
TICK_FONT   = dict(size=18)
LEGEND_FONT = dict(size=18)
TITLE_FONT  = dict(size=24)


def _display_results(results, params):
    successful = [r for r in results if r.get("success")]
    failed     = [r for r in results if not r.get("success")]

    st.markdown("---")
    st.markdown("### 📊 Results")

    if failed:
        with st.expander(f"⚠️ {len(failed)} structure(s) failed"):
            for r in failed:
                st.error(f"**{r['name']}**: {r.get('error', 'unknown error')}")

    if not successful:
        return

    import pandas as pd
    rows = []
    for r in successful:
        rows.append({
            "Structure":        r["name"],
            "Formula":          r.get("formula", "—"),
            "Fermi level (eV)": f"{r['fermi_level']:.4f}",
            "Band gap (eV)":    f"{r['band_gap']:.4f}",
            "Metal?":           "Yes" if r["band_gap"] < 0.05 else "No",
            "Atoms":            r.get("n_atoms", "—"),
            "Time (s)":         f"{r['duration']:.1f}",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    if params.get("plot_mode", "").startswith("Gaussian"):
        st.caption(
            "ℹ️ **Note on negative DOS values:** Small negative values can appear in regions of "
            "very low spectral weight. This is a known artefact of Gaussian broadening — the model "
            "does not enforce positivity. Switch to **Raw** mode to see the unmodified model output."
        )

    if len(successful) > 1:
        st.markdown("#### 📈 Comparison")
        names    = [r["name"] for r in successful]
        bandgaps = [r["band_gap"]    for r in successful]
        fermis   = [r["fermi_level"] for r in successful]

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Bar(
                x=names, y=bandgaps,
                marker_color=["#e74c3c" if bg < 0.05 else "#2980b9" for bg in bandgaps],
                text=[f"{bg:.3f}" for bg in bandgaps], textposition="auto", textfont=dict(size=17),
            ))
            fig.update_layout(
                title=dict(text="Band Gap (eV)", font=TITLE_FONT),
                yaxis_title="Band gap (eV)",
                xaxis=dict(tickangle=45, tickfont=TICK_FONT, title_font=AXIS_FONT),
                yaxis=dict(tickfont=TICK_FONT, title_font=AXIS_FONT),
                height=400, font=FONT,
            )
            st.plotly_chart(fig, width='stretch')

        with col2:
            fig2 = go.Figure(go.Bar(
                x=names, y=fermis,
                marker_color="darkorange",
                text=[f"{ef:.3f}" for ef in fermis], textposition="auto", textfont=dict(size=17),
            ))
            fig2.update_layout(
                title=dict(text="Fermi Level (eV)", font=TITLE_FONT),
                yaxis_title="Fermi level (eV)",
                xaxis=dict(tickangle=45, tickfont=TICK_FONT, title_font=AXIS_FONT),
                yaxis=dict(tickfont=TICK_FONT, title_font=AXIS_FONT),
                height=400, font=FONT,
            )
            st.plotly_chart(fig2, width='stretch')

    st.markdown("#### 🔬 Density of States")

    overlay = False
    if len(successful) > 1:
        mode = st.radio("View", ["Individual", "Overlay (all structures)"], horizontal=True, key="dos_view_mode")
        overlay = mode.startswith("Overlay")

    if overlay:
        _plot_overlay(successful, params)
    else:
        if len(successful) > 1:
            chosen = st.selectbox("Select structure:", [r["name"] for r in successful], key="dos_select")
            result = next(r for r in successful if r["name"] == chosen)
        else:
            result = successful[0]
        _plot_single(result, params)

    st.markdown("#### 📥 Export JSON")
    cols = st.columns(min(len(successful), 4))
    for i, r in enumerate(successful):
        with cols[i % 4]:
            st.download_button(
                label=f"💾 {r.get('formula', r['name'])}",
                data=_build_json(r),
                file_name=f"dos_{r['name'].replace(' ', '_')}.json",
                mime="application/json",
                key=f"dl_dos_{i}",
                width='stretch',
            )


def _get_curve(result, params, atom_idx=None):
    energies = np.asarray(result["energies"], dtype=float)
    dos      = np.asarray(result["dos"],      dtype=float)
    fermi    = result["fermi_level"]
    e_window = params.get("e_window", 10.0)
    raw_mode = params.get("plot_mode", "Gaussian broadened").startswith("Raw")

    if dos.ndim == 2:
        if dos.shape[1] != len(energies):
            dos = dos.T
        dos = dos[atom_idx] if atom_idx is not None else dos.sum(axis=0)

    mask = (energies >= fermi - e_window) & (energies <= fermi + e_window)
    e_win = energies[mask]
    d_win = dos[mask]

    if raw_mode:
        return e_win - fermi, d_win

    n_points = params.get("n_dos_points", 1000)
    sigma    = params.get("dos_sigma", 0.05)
    grid     = np.linspace(fermi - e_window, fermi + e_window, n_points)
    smooth   = np.zeros_like(grid)
    for ei, di in zip(energies, dos):
        smooth += di * np.exp(-0.5 * ((grid - ei) / sigma) ** 2)
    smooth /= sigma * np.sqrt(2 * np.pi)
    return grid - fermi, smooth


def _layout_base(title):
    return dict(
        title=dict(text=title, font=TITLE_FONT),
        xaxis=dict(title="E − E<sub>F</sub> (eV)", title_font=AXIS_FONT, tickfont=TICK_FONT),
        yaxis=dict(title="DOS (states / eV)",       title_font=AXIS_FONT, tickfont=TICK_FONT),
        height=560, font=FONT, legend=dict(font=LEGEND_FONT),
        hoverlabel=dict(bgcolor="white", font_size=17),
    )


def _fermi_vline(fig):
    fig.add_vline(
        x=0, line_dash="dash", line_color="black", line_width=1.5,
        annotation_text="E<sub>F</sub> = VBM", annotation_font_size=18,
        annotation_position="top right",
    )


def _cbm_vline(fig, bandgap):
    """Draw the predicted conduction-band-minimum line. PET-MAD's
    `calculate_efermi` places E_F at the VBM, so CBM sits at x = +bandgap on
    the shifted axis. Drawn only when bandgap is non-trivial."""
    if bandgap is None or bandgap < 0.05:
        return
    fig.add_vline(
        x=float(bandgap), line_dash="dot", line_color="#c0392b", line_width=1.5,
        annotation_text="CBM", annotation_font_size=18,
        annotation_position="top left",
        annotation_font_color="#c0392b",
    )


def _shade_gap(fig, bandgap):
    """Lightly shade the [VBM, CBM] band so the user can see what PET-MAD's
    bandgap head predicts as forbidden — any DOS visible inside this band is
    a disagreement between the bandgap head and the DOS head."""
    if bandgap is None or bandgap < 0.05:
        return
    fig.add_vrect(
        x0=0.0, x1=float(bandgap),
        fillcolor="rgba(127, 127, 127, 0.08)",
        line_width=0,
        annotation_text="Predicted gap region",
        annotation_position="top",
        annotation_font_size=14,
        annotation_font_color="#555",
    )


def _plot_single(result, params):
    fermi    = result["fermi_level"]
    bandgap  = result["band_gap"]
    formula  = result.get("formula", result["name"])
    per_atom = result.get("per_atom", False)
    gap_txt  = f"Band gap: {bandgap:.3f} eV" if bandgap >= 0.05 else "Metal"

    if per_atom and result.get("atom_labels"):
        labels = result["atom_labels"]
        selected_labels = st.multiselect(
            "Select atoms to plot:",
            options=labels, default=[labels[0]],
            key=f"dos_atom_sel_{result['name']}",
            help="Each selected atom is drawn as a separate trace on the same plot.",
        )

        if not selected_labels:
            st.warning("Select at least one atom.")
            return

        selected_indices = [labels.index(lbl) for lbl in selected_labels]
        also_total = st.checkbox("Also show total DOS", value=True, key=f"dos_show_total_{result['name']}")

        palette = ["#2980b9","#e74c3c","#27ae60","#8e44ad","#f39c12","#16a085","#e67e22","#2c3e50"]
        fig = go.Figure()

        if also_total:
            gs, smooth = _get_curve(result, params, atom_idx=None)
            fig.add_trace(go.Scatter(
                x=gs, y=smooth, mode="lines",
                line=dict(color="lightgrey", width=2, dash="dot"),
                name="Total DOS",
            ))

        for i, (idx, lbl) in enumerate(zip(selected_indices, selected_labels)):
            gs, smooth = _get_curve(result, params, atom_idx=idx)
            col = palette[i % len(palette)]
            fig.add_trace(go.Scatter(
                x=gs, y=smooth,
                fill="tozeroy", fillcolor=f"rgba({_hex_to_rgb(col)},0.15)",
                mode="lines", line=dict(color=col, width=2.5), name=lbl,
            ))

        _shade_gap(fig, bandgap)
        _fermi_vline(fig)
        _cbm_vline(fig, bandgap)
        fig.update_layout(**_layout_base(f"Per-atom DOS — {formula}  ({gap_txt})"))
        st.plotly_chart(fig, width='stretch')

    else:
        gs, smooth = _get_curve(result, params)
        is_insulator = bandgap >= 0.05
        fig = go.Figure()
        # Occupied: everything below the VBM (== E_F at x=0).
        occ = gs <= 0.0
        fig.add_trace(go.Scatter(
            x=gs[occ], y=smooth[occ],
            fill="tozeroy", fillcolor="rgba(41,128,185,0.35)",
            mode="lines", line=dict(color="#2980b9", width=2.5), name="Occupied (≤ VBM)",
        ))
        if is_insulator:
            # Anything PET-MAD's DOS head predicted inside the gap-head's
            # forbidden window — this is the disagreement region.
            in_gap = (gs > 0.0) & (gs < bandgap)
            fig.add_trace(go.Scatter(
                x=gs[in_gap], y=smooth[in_gap],
                fill="tozeroy", fillcolor="rgba(127,127,127,0.30)",
                mode="lines", line=dict(color="#7f7f7f", width=2.0),
                name="DOS-head leakage into gap",
            ))
            unocc = gs >= bandgap
        else:
            unocc = gs > 0.0
        fig.add_trace(go.Scatter(
            x=gs[unocc], y=smooth[unocc],
            fill="tozeroy", fillcolor="rgba(231,76,60,0.25)",
            mode="lines", line=dict(color="#e74c3c", width=2.5),
            name="Unoccupied (≥ CBM)" if is_insulator else "Unoccupied",
        ))
        _shade_gap(fig, bandgap)
        _fermi_vline(fig)
        _cbm_vline(fig, bandgap)
        fig.update_layout(**_layout_base(f"Electronic DOS — {formula}  ({gap_txt})"))
        st.plotly_chart(fig, width='stretch')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fermi Level (VBM)", f"{fermi:.4f} eV")
    c2.metric("Band Gap",          f"{bandgap:.4f} eV")
    c3.metric("Formula",           formula)
    c4.metric("Atoms",             result.get("n_atoms", "—"))

    if bandgap >= 0.05:
        st.caption(
            "ℹ️ PET-MAD has **two independent prediction heads**: the Fermi "
            "level (E_F = VBM under the T = 0 K cumulative-DOS convention) and "
            "the band-gap scalar. The grey-shaded **[VBM, CBM]** region is "
            "where the band-gap head predicts no states. Any DOS still visible "
            "inside that band is the **DOS head disagreeing with the band-gap "
            "head**, plus a few × σ of Gaussian-broadening tail leakage — not "
            "a plotting error. Lower `dos_sigma` to reduce the tail; states "
            "still visible after that are PET-MAD's intrinsic DOS-vs-gap "
            "inconsistency."
        )


def _plot_overlay(results, params):
    palette = ["#2980b9","#e74c3c","#27ae60","#8e44ad","#f39c12","#16a085","#2c3e50","#e67e22"]
    fig = go.Figure()
    for i, r in enumerate(results):
        gs, smooth = _get_curve(r, params)
        label = r.get("formula", r["name"])
        fig.add_trace(go.Scatter(
            x=gs, y=smooth, mode="lines",
            line=dict(color=palette[i % len(palette)], width=2.5),
            name=label,
        ))
    _fermi_vline(fig)
    fig.update_layout(**_layout_base("DOS Comparison (E − E<sub>F</sub>)"))
    st.plotly_chart(fig, width='stretch')


def _hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def _build_json(result):
    return json.dumps({
        "structure_name": result["name"],
        "formula":        result.get("formula", ""),
        "fermi_level_eV": result["fermi_level"],
        "band_gap_eV":    result["band_gap"],
        "is_metal":       result["band_gap"] < 0.05,
        "n_atoms":        result.get("n_atoms"),
        "per_atom":       result.get("per_atom", False),
        "atom_labels":    result.get("atom_labels", []),
        "energies_eV":    result["energies"],
        "dos":            result["dos"],
    }, indent=2)


def render_dos_script(script_content, embedded):
    """Download button, code view and usage note for a generated DOS script."""
    st.download_button(
        label="💾 Download PET-MAD-DOS Script",
        data=script_content,
        file_name="run_petmad_dos.py",
        mime="text/x-python",
        key=f"download_petmad_dos_{'embedded' if embedded else 'local'}",
        type="primary",
    )
    with st.expander("📋 Generated PET-MAD-DOS Script", expanded=True):
        st.code(script_content, language="python")
        st.info(
            "**Usage:** install `upet` (e.g. from `requirements-upet.txt`), "
            + ("then run `python run_petmad_dos.py` — the uploaded structures are "
               "embedded in the script. "
               if embedded else
               "place your structure files (`.cif`, `.vasp`, `POSCAR*`, `.xyz`) in "
               "the same folder as the script and run `python run_petmad_dos.py`. ")
            + "Per-structure results (DOS data, Fermi level, band gap, plot) are "
            "written to `dos_results/<basename>/`, and a summary of all structures "
            "to `dos_results/dos_summary_all.csv`."
        )


def generate_petmad_dos_script(dos_params, thread_count=4, structures=None):
    """Return a self-contained PET-MAD-DOS script as a string.

    With ``structures`` (the uploaded ones) the script first writes them as
    .vasp files next to itself; without, it uses the structure files already
    in its folder. Either way it then runs every structure file it finds.
    """
    import inspect
    from datetime import datetime
    from helpers.generate_python_code import _generate_structure_creation_code

    # The script carries the same DOS helper as the app (old and new upet API).
    compute_src = inspect.getsource(compute_petmad_dos)

    version      = dos_params.get("version", "latest")
    device       = dos_params.get("device", "cpu")
    per_atom     = bool(dos_params.get("per_atom", False))
    broadened    = not dos_params.get("plot_mode", "Gaussian broadened").startswith("Raw")
    n_dos_points = int(dos_params.get("n_dos_points", 1000))
    dos_sigma    = float(dos_params.get("dos_sigma", 0.05))
    e_window     = float(dos_params.get("e_window", 10.0))

    if structures:
        structure_creation_code = (
            '    print("\\nWriting the embedded structures...")\n'
            + _generate_structure_creation_code(structures)
        )
        input_desc = (f"The {len(structures)} uploaded structure(s) are embedded and written as "
                      ".vasp files next to the script\nbefore the run; any other structure files "
                      "in the folder are processed as well.")
    else:
        structure_creation_code = ""
        input_desc = ("Place your structure files (.cif / .vasp / POSCAR* / .xyz / .extxyz) in "
                      "the same folder\nas this script.")

    plot_desc = (f"Gaussian broadened (sigma = {dos_sigma} eV, {n_dos_points} points)"
                 if broadened else "Raw model output (no broadening)")

    return f'''#!/usr/bin/env python3
"""Standalone PET-MAD-DOS: electronic density of states, Fermi level and band gap.

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: PET-MAD-DOS (version {version})
Device: {device}

DOS settings
------------
  Per-atom DOS       : {per_atom}
  Plot mode          : {plot_desc}
  Energy window      : E_F +/- {e_window} eV
  Threads (OMP)      : {thread_count}

{input_desc}

Requirements: pip install upet ase matplotlib  (or requirements-upet.txt)

Outputs (per structure, under ``dos_results/<basename>/``):
  * dos_data.csv     - raw model DOS: energy on the model's grid (eV), E - E_F (eV), total DOS{', one column per atom' if per_atom else ''}
  * dos_plot_data.csv - the plotted curve (window around E_F{', broadened' if broadened else ''})
  * dos_summary.txt  - Fermi level, band gap, metal / non-metal
  * dos.json         - all of the above in one file (same format as the GUI export)
  * dos_plot.png     - DOS plot (occupied / unoccupied, E_F and CBM marked)

Aggregate outputs (directly under ``dos_results/``):
  * dos_summary_all.csv    - one row per structure (Fermi level, band gap, ...)
  * band_gap_by_structure.png, dos_overlay.png  (when >1 structure)
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Threading must be set BEFORE torch is imported.
os.environ["OMP_NUM_THREADS"] = "{thread_count}"

import numpy as np

import torch
torch.set_num_threads({thread_count})

from ase.io import read
from upet.calculator import PETMADDOSCalculator

# ── Parameters chosen in the GUI ────────────────────────────────────────
MODEL_VERSION = "{version}"
DEVICE        = "{device}"      # "cuda" for a GPU (falls back to CPU if unavailable)
PER_ATOM      = {per_atom}
BROADENED     = {broadened}       # False = raw model output
N_DOS_POINTS  = {n_dos_points}
DOS_SIGMA     = {dos_sigma}       # eV, Gaussian broadening
E_WINDOW      = {e_window}       # eV around E_F (plots / dos_plot_data.csv)
METAL_GAP     = 0.05       # eV, band gaps below this are reported as metals
OUTPUT_ROOT   = Path("dos_results")


{compute_src}

def fmt_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return "{{:.1f}} s".format(seconds)
    if seconds < 3600:
        return "{{:.1f}} min".format(seconds / 60.0)
    return "{{:.2f}} h".format(seconds / 3600.0)


def total_dos(dos, n_energies):
    """Total DOS from a per-atom array (atoms x energies), or the array itself."""
    dos = np.asarray(dos, dtype=float)
    if dos.ndim == 2:
        if dos.shape[1] != n_energies:
            dos = dos.T
        return dos.sum(axis=0)
    return dos


def plot_curve(energies, dos, fermi):
    """The curve that is plotted: window around E_F, broadened or raw."""
    energies = np.asarray(energies, dtype=float)
    dos = np.asarray(dos, dtype=float)
    if not BROADENED:
        mask = (energies >= fermi - E_WINDOW) & (energies <= fermi + E_WINDOW)
        return energies[mask] - fermi, dos[mask]
    grid = np.linspace(fermi - E_WINDOW, fermi + E_WINDOW, N_DOS_POINTS)
    smooth = np.zeros_like(grid)
    for ei, di in zip(energies, dos):
        smooth += di * np.exp(-0.5 * ((grid - ei) / DOS_SIGMA) ** 2)
    smooth /= DOS_SIGMA * np.sqrt(2 * np.pi)
    return grid - fermi, smooth


def atom_labels(atoms):
    labels, counts = [], {{}}
    for i, sym in enumerate(atoms.get_chemical_symbols()):
        counts[sym] = counts.get(sym, 0) + 1
        labels.append("{{}}{{}}  (atom {{}})".format(sym, counts[sym], i + 1))
    return labels


def save_plot(x, y, fermi, gap, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    is_insulator = gap >= METAL_GAP
    occ = x <= 0.0
    ax.fill_between(x[occ], y[occ], color="#2980b9", alpha=0.35, label="Occupied (<= VBM)")
    ax.plot(x[occ], y[occ], color="#2980b9", linewidth=2)
    unocc = x >= gap if is_insulator else x > 0.0
    ax.fill_between(x[unocc], y[unocc], color="#e74c3c", alpha=0.25,
                    label="Unoccupied (>= CBM)" if is_insulator else "Unoccupied")
    ax.plot(x[unocc], y[unocc], color="#e74c3c", linewidth=2)
    if is_insulator:
        ax.axvspan(0.0, gap, color="grey", alpha=0.08, label="Predicted gap region")
        ax.axvline(gap, color="#c0392b", linestyle=":", linewidth=1.5, label="CBM")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5, label="E_F = VBM")
    ax.set_xlabel("E - E_F (eV)", fontsize=16)
    ax.set_ylabel("DOS (states / eV)", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_dos_for_structure(fname, calc, out_dir):
    atoms = read(fname)
    natoms = len(atoms)
    formula = str(atoms.symbols.formula.reduce()[0])  # Si8 -> Si, as in the app
    print("  {{}} ({{}} atoms)".format(formula, natoms))

    print("  -> DOS, Fermi level and band gap (per_atom={{}}) ...".format(PER_ATOM))
    energies, dos, dos_tot, fermi, gap = compute_petmad_dos(calc, atoms, PER_ATOM)
    if PER_ATOM and dos.ndim == 1:  # a single atom squeezes to 1-D
        dos = dos[None, :]
    is_metal = gap < METAL_GAP

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = atom_labels(atoms) if PER_ATOM else []

    # Raw model DOS on the model's own energy grid.
    tot = total_dos(dos_tot, len(energies))
    per_atom_cols = None
    if PER_ATOM and dos.ndim == 2:
        per_atom_cols = dos if dos.shape[1] == len(energies) else dos.T
    with open(out_dir / "dos_data.csv", "w") as fh:
        header = ["energy_eV", "E_minus_EF_eV", "total_dos"]
        if per_atom_cols is not None:
            header += ["atom_{{}}_{{}}".format(i + 1, s)
                       for i, s in enumerate(atoms.get_chemical_symbols())]
        fh.write(",".join(header) + "\\n")
        for j, e in enumerate(energies):
            row = [e, e - fermi, float(tot[j])]
            if per_atom_cols is not None:
                row += list(per_atom_cols[:, j])
            fh.write(",".join("{{:.8g}}".format(v) for v in row) + "\\n")

    # The plotted curve (window around E_F, broadened or raw).
    x, y = plot_curve(energies, tot, fermi)
    with open(out_dir / "dos_plot_data.csv", "w") as fh:
        fh.write("E_minus_EF_eV,dos\\n")
        for xi, yi in zip(x, y):
            fh.write("{{:.8g}},{{:.8g}}\\n".format(xi, yi))

    with open(out_dir / "dos_summary.txt", "w") as fh:
        fh.write("PET-MAD-DOS\\n")
        fh.write("===========\\n")
        fh.write("Structure      : {{}}\\n".format(fname))
        fh.write("Formula        : {{}}\\n".format(formula))
        fh.write("Atoms          : {{}}\\n".format(natoms))
        fh.write("Model version  : {{}}\\n\\n".format(MODEL_VERSION))
        fh.write("Fermi level (VBM) : {{:.6f}} eV\\n".format(fermi))
        fh.write("Band gap          : {{:.6f}} eV\\n".format(gap))
        fh.write("Metal             : {{}}\\n".format("yes" if is_metal else "no"))

    with open(out_dir / "dos.json", "w") as fh:
        json.dump({{
            "structure_name": fname,
            "formula": formula,
            "fermi_level_eV": fermi,
            "band_gap_eV": gap,
            "is_metal": is_metal,
            "n_atoms": natoms,
            "per_atom": PER_ATOM,
            "atom_labels": labels,
            "energies_eV": energies.tolist(),
            "dos": dos.tolist(),
        }}, fh, indent=2)

    try:
        gap_txt = "Metal" if is_metal else "band gap {{:.3f}} eV".format(gap)
        save_plot(x, y, fermi, gap, "{{}} ({{}})".format(formula, gap_txt), out_dir / "dos_plot.png")
    except Exception as exc:
        print("  WARN: could not create plot: {{}}".format(exc))

    print("  E_F = {{:.4f}} eV  |  band gap = {{:.4f}} eV  ({{}})".format(
        fermi, gap, "metal" if is_metal else "non-metal"))
    return dict(formula=formula, natoms=natoms, fermi=fermi, gap=gap,
                is_metal=is_metal, curve=(x, y))


def main():
    start = time.time()
    print("PET-MAD-DOS - electronic density of states")
    print("  {{}}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("  Model version: {{}}  (device {{}})".format(MODEL_VERSION, DEVICE))
    print("  Per-atom DOS: {{}}  |  broadened: {{}}".format(PER_ATOM, BROADENED))
    print("  OMP threads: {{}}".format(os.environ.get('OMP_NUM_THREADS')))
{structure_creation_code}
    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("  CUDA not available - using the CPU instead.")
        device = "cpu"
    print("\\nLoading the PET-MAD-DOS model (downloaded on first use)...")
    calc = PETMADDOSCalculator(version=MODEL_VERSION, device=device)

    structure_files = sorted([
        f for f in os.listdir(".")
        if f.startswith("POSCAR") or f.endswith(".vasp")
        or f.endswith(".poscar") or f.endswith(".cif")
        or f.endswith(".xyz") or f.endswith(".extxyz")
    ])
    if not structure_files:
        print("ERROR: No structure files (.cif / .vasp / POSCAR* / .xyz / .extxyz) found")
        sys.exit(1)
    print("\\nFound {{}} structure(s): {{}}".format(len(structure_files), structure_files))

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
            res = run_dos_for_structure(fname, calc, OUTPUT_ROOT / base)
        except Exception as exc:
            print("  ERROR processing {{}}: {{}}".format(fname, exc))
            res = None
        dt = time.time() - t_struct
        if res is not None:
            res["time"] = dt
            summary.append((fname, res))

        durations.append(dt)
        recent = durations[-ETA_WINDOW:]
        avg = sum(recent) / len(recent)
        n_left = n_total - (i + 1)
        print("  Done in {{}}  |  avg/structure (last {{}}): {{}}  |  "
              "{{}} left  |  est. remaining: {{}}".format(
                  fmt_duration(dt), len(recent), fmt_duration(avg),
                  n_left, fmt_duration(avg * n_left)))

    if summary:
        print("\\n==== Summary ====")
        print("{{:<4s}} {{:<28s}} {{:<12s}} {{:>14s}} {{:>14s}} {{:>6s}}".format(
            "idx", "Structure", "Formula", "E_F (eV)", "Gap (eV)", "Metal"))
        for idx, (fname, r) in enumerate(summary):
            print("{{:<4d}} {{:<28s}} {{:<12s}} {{:>14.4f}} {{:>14.4f}} {{:>6s}}".format(
                idx, fname, r["formula"], r["fermi"], r["gap"], "yes" if r["is_metal"] else "no"))

        with open(OUTPUT_ROOT / "dos_summary_all.csv", "w") as fh:
            fh.write("index,structure,formula,n_atoms,fermi_level_eV,band_gap_eV,is_metal,time_s\\n")
            for idx, (fname, r) in enumerate(summary):
                fh.write("{{}},{{}},{{}},{{}},{{}},{{}},{{}},{{:.2f}}\\n".format(
                    idx, fname, r["formula"], r["natoms"], r["fermi"], r["gap"],
                    r["is_metal"], r["time"]))

        if len(summary) > 1:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                names = [os.path.splitext(f)[0] for f, _ in summary]
                gaps = [r["gap"] for _, r in summary]
                fig, ax = plt.subplots(figsize=(max(7.0, 0.6 * len(names) + 3), 5.0))
                ax.bar(names, gaps, color=["#e74c3c" if g < METAL_GAP else "#2980b9" for g in gaps])
                ax.set_ylabel("Band gap (eV)", fontsize=16)
                ax.set_title("Band gap per structure", fontsize=17)
                ax.tick_params(axis="both", which="major", labelsize=12)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                fig.tight_layout()
                fig.savefig(OUTPUT_ROOT / "band_gap_by_structure.png", dpi=200)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(9.0, 5.5))
                for fname, r in summary:
                    x, y = r["curve"]
                    ax.plot(x, y, linewidth=2, label=r["formula"] + " (" + os.path.splitext(fname)[0] + ")")
                ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5)
                ax.set_xlabel("E - E_F (eV)", fontsize=16)
                ax.set_ylabel("DOS (states / eV)", fontsize=16)
                ax.set_title("DOS comparison", fontsize=17)
                ax.tick_params(axis="both", which="major", labelsize=14)
                ax.legend(fontsize=11)
                fig.tight_layout()
                fig.savefig(OUTPUT_ROOT / "dos_overlay.png", dpi=200)
                plt.close(fig)
            except Exception as exc:
                print("WARN: could not create comparison plots: {{}}".format(exc))

    print("\\nFinished {{}}/{{}} structure(s) in {{}}. Results in {{}}/".format(
        len(summary), n_total, fmt_duration(time.time() - start), OUTPUT_ROOT))


if __name__ == "__main__":
    main()
'''
