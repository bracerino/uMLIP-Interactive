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


def render_petmad_dos_panel(structures: dict) -> None:
    st.markdown("## ⚛️ Electronic DOS — PET-MAD-DOS")

    if not PETMAD_DOS_AVAILABLE:
        st.error("**upet is not installed.**  \nRun `pip install upet` and restart the app.")
        return

    if not structures:
        st.warning("⚠️ Please upload and lock at least one structure first.")
        return

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

    _init_state()

    if st.session_state._dos_running:
        col_b1, col_b2 = st.columns([3, 1])
        with col_b1:
            start = st.button(
                "🚀  Start PET-MAD-DOS Calculation",
                type="primary", disabled=True,
                use_container_width=True, key="dos_start_btn",
            )
        with col_b2:
            if st.button("🛑 Stop", key="dos_stop_btn", use_container_width=True):
                st.session_state._dos_stop_event.set()
    else:
        start = st.button(
            "🚀  Start PET-MAD-DOS Calculation",
            type="primary",
            use_container_width=True, key="dos_start_btn",
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

            log(f"  → DOS (per_atom={per_atom}) …")
            energies, dos = calc.calculate_dos(atoms, per_atom=per_atom)

            if per_atom:
                log("  → Total DOS for Fermi / gap …")
                _, dos_total = calc.calculate_dos(atoms, per_atom=False)
            else:
                dos_total = dos

            log("  → Fermi level …")
            fermi = float(calc.calculate_efermi(atoms))

            log("  → Band gap …")
            bandgap = float(calc.calculate_bandgap(atoms, dos=dos_total))

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
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
            st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig2, use_container_width=True)

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
                use_container_width=True,
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
        annotation_text="E<sub>F</sub>", annotation_font_size=18,
        annotation_position="top right",
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

        _fermi_vline(fig)
        fig.update_layout(**_layout_base(f"Per-atom DOS — {formula}  ({gap_txt})"))
        st.plotly_chart(fig, use_container_width=True)

    else:
        gs, smooth = _get_curve(result, params)
        occ = gs <= 0
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gs[occ], y=smooth[occ],
            fill="tozeroy", fillcolor="rgba(41,128,185,0.35)",
            mode="lines", line=dict(color="#2980b9", width=2.5), name="Occupied",
        ))
        fig.add_trace(go.Scatter(
            x=gs[~occ], y=smooth[~occ],
            fill="tozeroy", fillcolor="rgba(231,76,60,0.25)",
            mode="lines", line=dict(color="#e74c3c", width=2.5), name="Unoccupied",
        ))
        _fermi_vline(fig)
        fig.update_layout(**_layout_base(f"Electronic DOS — {formula}  ({gap_txt})"))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fermi Level", f"{fermi:.4f} eV")
    c2.metric("Band Gap",    f"{bandgap:.4f} eV")
    c3.metric("Formula",     formula)
    c4.metric("Atoms",       result.get("n_atoms", "—"))


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
    st.plotly_chart(fig, use_container_width=True)


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
