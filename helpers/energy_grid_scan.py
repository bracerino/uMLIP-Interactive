"""Energy Grid Scan — UI + standalone-script generator.

The user picks an insertion element, a real-space grid spacing, and a
minimum neighbour distance. The generated script loads each structure in
the run folder, builds an `na × nb × nc` grid in fractional coordinates
(with `na = ceil(|a|/spacing)` etc., so grid points are roughly
evenly-spaced in real space even for non-orthogonal cells), inserts a
probe atom at every grid point, computes the single-point energy and
saves the full 3-D map as both NPZ and CSV. Grid points that sit closer
than `min_distance` to an existing atom are not evaluated — their energy
slot is filled with a configurable large sentinel value so the spatial
shape of the output array is preserved.
"""

import io
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy.ndimage import gaussian_filter, zoom as ndi_zoom
    _SCIPY_AVAILABLE = True
except Exception:  # scipy is optional — degrade gracefully
    _SCIPY_AVAILABLE = False


def _preprocess_grid(energies, blocked, smoothing_sigma, zoom_factor):
    """Return (values, blocked_resampled, scale) ready for plotting.

    * Blocked points → NaN, then NaN-filled with the median of the valid
      energies so scipy can operate without producing more NaNs.
    * Optional Gaussian smoothing in grid-index space (sigma in voxels).
    * Optional cubic up-sampling via ``scipy.ndimage.zoom``.

    `blocked_resampled` follows the same shape as the returned `values` so
    we can mask the smoothed/zoomed field consistently. `scale` is the
    per-axis zoom factor (always 1.0 when no zoom or scipy unavailable).
    """
    energies = np.asarray(energies, dtype=float)
    blocked  = np.asarray(blocked, dtype=bool)

    work = np.where(blocked, np.nan, energies)
    valid = work[~np.isnan(work)]
    fill_val = float(np.nanmedian(valid)) if valid.size else 0.0
    filled = np.where(np.isnan(work), fill_val, work)

    blocked_out = blocked.copy()

    if _SCIPY_AVAILABLE and smoothing_sigma and smoothing_sigma > 0:
        # `mode="wrap"` is the right choice here: the grid covers one
        # unit cell, so periodic boundaries are physically correct.
        filled = gaussian_filter(filled, sigma=float(smoothing_sigma),
                                 mode="wrap")

    scale = 1.0
    if _SCIPY_AVAILABLE and zoom_factor and zoom_factor > 1.001:
        scale = float(zoom_factor)
        filled = ndi_zoom(filled, zoom=scale, order=3, mode="wrap")
        # Up-sample the blocked mask to the same shape using nearest-
        # neighbour interpolation so it remains boolean.
        blocked_out = ndi_zoom(blocked.astype(float), zoom=scale, order=0,
                               mode="wrap").astype(bool)

    return filled, blocked_out, scale

# Reuse the project-wide MLIP setup so the generated script supports the
# *same* model list (GRACE / MACE / MACE-OFF / MACE-POLAR / CHGNet /
# SevenNet / MatterSim / ORB / Nequix / PET-MAD / UPET / custom MACE).
from helpers.generate_python_code import (
    _generate_calculator_setup_code,
    _generate_mlip_imports,
)


# Plotly colourscales that read well for an energy landscape.
_VIEWER_COLORMAPS = [
    "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
    "Turbo", "Jet", "RdBu_r", "RdYlBu_r", "Hot", "Blues_r",
]


def _load_energy_grid_npz(uploaded_file):
    """Decode an uploaded `energy_grid.npz` into a plain dict of arrays /
    scalars. Returns None on failure (and shows an `st.error`)."""
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        npz = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        st.error(f"❌ Could not read NPZ: {exc}")
        return None

    needed = {"energies", "blocked", "cell"}
    missing = needed - set(npz.files)
    if missing:
        st.error(
            f"❌ NPZ is missing required arrays: {sorted(missing)}. "
            f"Expected an `energy_grid.npz` produced by the standalone script."
        )
        return None

    out = {k: npz[k] for k in npz.files}
    # Unwrap scalar metadata stored as 0-d arrays.
    for k in ("insert_element", "grid_spacing_A", "min_distance_A",
              "blocked_energy", "base_energy_eV",
              "region_enabled", "region_length", "relaxed"):
        if k in out and out[k].ndim == 0:
            try:
                out[k] = out[k].item()
            except Exception:
                pass
    return out


# Atomic numbers for the probe-atom line in CUBE files. Covers everything
# users normally insert into a host structure (H–Bi). Falls back to H.
_ELEMENT_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83,
}

_BOHR_PER_ANG = 1.0 / 0.52917721067


def _grid_box(data):
    """Return ``(cell, grid_cell, grid_origin)`` for a loaded NPZ.

    ``cell`` is the host's full unit cell. ``grid_cell`` is the spanning
    vectors of the (possibly sub-region) datagrid box and ``grid_origin`` its
    Cartesian origin. Old NPZ files produced before the sub-region feature
    lack these keys, so they fall back to the full cell with a zero origin —
    i.e. the grid spans the whole cell, exactly as it used to.
    """
    cell = np.asarray(data["cell"], dtype=float)
    gc = data.get("grid_cell")
    go = data.get("grid_origin")
    grid_cell = np.asarray(gc, dtype=float) if gc is not None else cell
    grid_origin = (
        np.asarray(go, dtype=float) if go is not None else np.zeros(3)
    )
    return cell, grid_cell, grid_origin


def _prepare_volumetric_field(data, smoothing_sigma, zoom_factor, shift_to_zero):
    """Shared preprocessing for XSF + CUBE export.

    Returns ``(arr, cell, shift_value)`` where ``arr`` has shape
    ``(na, nb, nc)`` and ``shift_value`` is the constant that was
    subtracted from every cell (0 when ``shift_to_zero=False``).
    """
    energies = np.asarray(data["energies"], dtype=float)
    blocked  = np.asarray(data["blocked"], dtype=bool)
    cell     = np.asarray(data["cell"], dtype=float)

    arr, blocked_out, _ = _preprocess_grid(
        energies, blocked, smoothing_sigma, zoom_factor
    )
    valid = arr[~blocked_out]
    fill = float(np.median(valid)) if valid.size else 0.0
    arr = np.where(blocked_out, fill, arr)

    shift_value = 0.0
    if shift_to_zero and valid.size:
        shift_value = float(arr.min())
        arr = arr - shift_value

    return arr, cell, shift_value


def _build_cube_from_loaded_npz(data, smoothing_sigma=0.0, zoom_factor=1.0,
                                shift_to_zero=False):
    """Build a Gaussian-cube volumetric file from the loaded NPZ.

    CUBE is the format VESTA / GaussView / Avogadro auto-detect a
    sensible default iso-level for. Coordinates and lattice vectors are
    in Bohr (CUBE convention); the energy field is in eV (with the
    optional ``shift_to_zero`` offset noted in the header).
    """
    arr, cell, shift = _prepare_volumetric_field(
        data, smoothing_sigma, zoom_factor, shift_to_zero
    )
    na, nb, nc = arr.shape
    _, grid_cell, grid_origin = _grid_box(data)
    cell_bohr   = grid_cell * _BOHR_PER_ANG
    origin_bohr = grid_origin * _BOHR_PER_ANG

    elem = str(data.get("insert_element", "H")) or "H"
    z_atom = _ELEMENT_Z.get(elem, 1)

    lines = []
    if shift > 0.0:
        lines.append(
            f"Energy grid scan ({elem} probe). Values are E_eV minus {shift:.6f}"
        )
    elif shift < 0.0:
        lines.append(
            f"Energy grid scan ({elem} probe). Values are E_eV plus {-shift:.6f}"
        )
    else:
        lines.append(f"Energy grid scan ({elem} probe). Values are absolute E_eV")
    lines.append("Generated by mace-md-gui (energy_grid_scan helper)")
    # natoms + origin (in Bohr); positive natoms → coords in Bohr.
    lines.append(
        f"   1   {origin_bohr[0]: .6f}   {origin_bohr[1]: .6f}   "
        f"{origin_bohr[2]: .6f}"
    )
    # Grid header: N + axis vector / N along that axis (Bohr per voxel).
    lines.append(
        f"  {na:5d}   {cell_bohr[0,0]/na: .8f}   "
        f"{cell_bohr[0,1]/na: .8f}   {cell_bohr[0,2]/na: .8f}"
    )
    lines.append(
        f"  {nb:5d}   {cell_bohr[1,0]/nb: .8f}   "
        f"{cell_bohr[1,1]/nb: .8f}   {cell_bohr[1,2]/nb: .8f}"
    )
    lines.append(
        f"  {nc:5d}   {cell_bohr[2,0]/nc: .8f}   "
        f"{cell_bohr[2,1]/nc: .8f}   {cell_bohr[2,2]/nc: .8f}"
    )
    # Probe atom anchored at the datagrid origin.
    lines.append(
        f"  {z_atom:3d}   0.000000   {origin_bohr[0]: .6f}   "
        f"{origin_bohr[1]: .6f}   {origin_bohr[2]: .6f}"
    )
    # CUBE iterates n3 (last axis) fastest, n2 middle, n1 slowest — which
    # is exactly what numpy's C-order .flatten() produces for shape
    # (na, nb, nc). 6 values per line, scientific.
    flat = arr.flatten()
    for chunk_start in range(0, len(flat), 6):
        row = flat[chunk_start : chunk_start + 6]
        lines.append("  " + "  ".join(f"{x: .5e}" for x in row))

    return "\n".join(lines) + "\n"


def _build_xsf_from_loaded_npz(data, smoothing_sigma=0.0, zoom_factor=1.0,
                                shift_to_zero=False):
    """Build a VESTA-friendly XSF string from a loaded NPZ.

    Mirrors the standalone script's `_write_xsf` fix (median-fill blocked
    points + periodic wrap) and, on top of that, lets the user export the
    *smoothed / interpolated* field that's currently on screen. The probe
    element is rendered as a single dummy atom at the cell origin so VESTA
    has something to anchor the unit-cell drawing to; the file otherwise
    stands alone (no need to load the host structure separately).
    """
    arr, cell, shift = _prepare_volumetric_field(
        data, smoothing_sigma, zoom_factor, shift_to_zero
    )
    _, grid_cell, grid_origin = _grid_box(data)
    # Periodic wrap (XSF spec convention — last slice = first slice).
    arr = np.concatenate([arr, arr[:1, :, :]], axis=0)
    arr = np.concatenate([arr, arr[:, :1, :]], axis=1)
    arr = np.concatenate([arr, arr[:, :, :1]], axis=2)
    na, nb, nc = arr.shape

    elem = str(data.get("insert_element", "X")) or "X"
    z_atom = _ELEMENT_Z.get(elem, 1)

    lines = []
    # Comment block at the top — VESTA / XCrysDen both tolerate this and
    # the user gets a record of what was shifted.
    lines.append("#")
    if shift > 0.0:
        lines.append(f"# Energies (eV) minus {shift:.6f}")
    elif shift < 0.0:
        lines.append(f"# Energies (eV) plus {-shift:.6f}")
    else:
        lines.append("# Absolute energies (eV)")
    lines.append("#")
    lines += ["CRYSTAL", "PRIMVEC"]
    for v in cell:
        lines.append(f"  {v[0]:.7f}   {v[1]:.7f}   {v[2]:.7f}")
    lines.append("PRIMCOORD")
    lines.append("1 1")
    # Single probe-atom anchor by atomic number (matches the format VESTA
    # has been shown to read reliably — element symbol followed by
    # coordinates ALSO works, but the Z-form is the safest fallback).
    # Anchored at the datagrid origin so the box lands in the right place
    # when a sub-region was scanned (origin = 0 for a full-cell scan).
    lines.append(
        f"{z_atom:>3d}   {grid_origin[0]:.7f}   "
        f"{grid_origin[1]:.7f}   {grid_origin[2]:.7f}"
    )
    lines.append("BEGIN_BLOCK_DATAGRID_3D")
    lines.append("energy_grid")
    lines.append("BEGIN_DATAGRID_3D_energy")
    # No leading indent on the grid dims / origin / vectors — matches the
    # exemplar VESTA accepts. The data is written ONE value per line with
    # PLAIN-decimal formatting (no scientific notation), because VESTA's
    # `DATAGRID_3D` parser silently drops the trace when it encounters
    # `1.23e+01`-style numbers in some versions.
    lines.append(f"{na} {nb} {nc}")
    lines.append(
        f"{grid_origin[0]:.7f} {grid_origin[1]:.7f} {grid_origin[2]:.7f}"
    )
    for v in grid_cell:
        lines.append(f"  {v[0]:.7f}   {v[1]:.7f}   {v[2]:.7f}")
    flat = arr.transpose(2, 1, 0).flatten()
    for v in flat:
        # %.7f mirrors the working exemplar (`8.4510592`); fall back to a
        # wider field for values whose magnitude blows past 8 chars.
        lines.append(f"  {float(v):.7f}")
    lines.append("END_DATAGRID_3D")
    lines.append("END_BLOCK_DATAGRID_3D")
    return "\n".join(lines) + "\n"


def _make_volume_figure(data, cmap, vmin, vmax, opacity, surface_count,
                        coord_mode="fractional", point_size=4,
                        smoothing_sigma=0.0, zoom_factor=1.0):
    """3-D rendering of the energy grid.

    * **fractional** mode → `go.Volume` on an axis-aligned cube. Plotly's
      volume tessellator silently drops the entire trace if any value is
      NaN, so blocked positions are hidden by setting their energy a hair
      above `isomax` (still inside the colorscale, but outside the
      iso-surface range — Plotly skips them).
    * **cartesian** mode → `go.Scatter3d` at each point's true
      `frac @ cell` position. Works for any cell shape. NaN points are
      simply omitted from the trace.
    """
    energies = np.asarray(data["energies"], dtype=float)
    blocked  = np.asarray(data["blocked"], dtype=bool)
    cell     = np.asarray(data["cell"], dtype=float)

    # Smoothing / interpolation (no-op when sigma=0 and zoom=1).
    smoothed, blocked_out, _ = _preprocess_grid(
        energies, blocked, smoothing_sigma, zoom_factor
    )
    na, nb, nc = smoothed.shape

    ii, jj, kk = np.meshgrid(
        np.arange(na) / na,
        np.arange(nb) / nb,
        np.arange(nc) / nc,
        indexing="ij",
    )
    frac = np.stack([ii, jj, kk], axis=-1)

    fig = go.Figure()

    if coord_mode == "cartesian":
        # Place points at their absolute Cartesian position. For a sub-region
        # scan this offsets by the box origin and scales by the box vectors;
        # for a full-cell scan grid_origin=0 and grid_cell=cell, so this
        # reduces to the original `frac @ cell`.
        _, grid_cell, grid_origin = _grid_box(data)
        cart = grid_origin + frac @ grid_cell
        masked = np.where(blocked_out, np.nan, smoothed)
        x_flat = cart[..., 0].flatten()
        y_flat = cart[..., 1].flatten()
        z_flat = cart[..., 2].flatten()
        e_flat = masked.flatten()
        keep = ~np.isnan(e_flat)
        e_disp = np.clip(e_flat[keep], vmin, vmax)
        fig.add_trace(go.Scatter3d(
            x=x_flat[keep], y=y_flat[keep], z=z_flat[keep],
            mode="markers",
            marker=dict(
                size=int(point_size),
                color=e_disp,
                cmin=float(vmin), cmax=float(vmax),
                colorscale=cmap,
                opacity=float(opacity),
                colorbar=dict(title="Energy (eV)"),
            ),
            hovertemplate=(
                "x: %{x:.3f} Å<br>y: %{y:.3f} Å<br>z: %{z:.3f} Å<br>"
                "E: %{marker.color:.4f} eV<extra></extra>"
            ),
        ))
        scene = dict(
            xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)",
            aspectmode="data",
        )
    else:  # fractional → go.Volume
        # Plotly's Volume bails on any NaN, so hide blocked points by
        # nudging them just past isomax. The colour-scale range is set by
        # cmin/cmax, so blocked positions stay invisible inside iso-shells.
        nudge = vmax + max(abs(vmax - vmin) * 1.0, 1e-3)
        values = np.where(blocked_out, nudge, smoothed)
        fig.add_trace(go.Volume(
            x=frac[..., 0].flatten(),
            y=frac[..., 1].flatten(),
            z=frac[..., 2].flatten(),
            value=values.flatten(),
            isomin=float(vmin),
            isomax=float(vmax),
            cmin=float(vmin), cmax=float(vmax),
            opacity=float(opacity),
            surface_count=int(surface_count),
            colorscale=cmap,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorbar=dict(title="Energy (eV)"),
            showscale=True,
        ))
        scene = dict(
            xaxis_title="frac a", yaxis_title="frac b", zaxis_title="frac c",
            aspectmode="cube",
        )

    fig.update_layout(
        scene=scene,
        height=560,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    return fig


def _make_slice_figure(data, axis_label, slice_index, cmap, vmin, vmax,
                       coord_mode="fractional",
                       smoothing_sigma=0.0, zoom_factor=1.0):
    """2-D heatmap of one slice through the grid.

    Fractional mode → axes are `frac_*` in [0, 1).
    Cartesian mode  → axes are `|lattice_vector| * frac` (Å) along the
    two in-plane lattice directions. For an orthogonal cell this is just
    true Cartesian XY; for a non-orthogonal cell it preserves real-space
    length scales along each lattice direction, which is what users want
    when toggling to "Cartesian" for a single 2-D slice.
    """
    energies_raw = np.asarray(data["energies"], dtype=float)
    blocked_raw  = np.asarray(data["blocked"], dtype=bool)
    cell         = np.asarray(data["cell"], dtype=float)
    na_orig, nb_orig, nc_orig = energies_raw.shape

    a_len = float(np.linalg.norm(cell[0]))
    b_len = float(np.linalg.norm(cell[1]))
    c_len = float(np.linalg.norm(cell[2]))

    smoothed, blocked_out, _ = _preprocess_grid(
        energies_raw, blocked_raw, smoothing_sigma, zoom_factor
    )
    na, nb, nc = smoothed.shape

    e = np.where(blocked_out, np.nan, smoothed)

    # Rescale the user-chosen `slice_index` (which was bounded by the
    # original grid) into the post-zoom grid so the slider continues to
    # mean "the same fractional position" after up-sampling.
    scale_a = na / na_orig
    scale_b = nb / nb_orig
    scale_c = nc / nc_orig

    if axis_label == "a (slice on a-axis, plot b vs c)":
        idx = int(round(slice_index * scale_a))
        idx = min(max(idx, 0), na - 1)
        plane = e[idx, :, :]                  # shape (nb, nc)
        if coord_mode == "cartesian":
            x_lab, y_lab = "c-axis (Å)", "b-axis (Å)"
            xs = (np.arange(nc) / nc) * c_len
            ys = (np.arange(nb) / nb) * b_len
        else:
            x_lab, y_lab = "frac c", "frac b"
            xs = np.arange(nc) / nc
            ys = np.arange(nb) / nb
    elif axis_label == "b (slice on b-axis, plot a vs c)":
        idx = int(round(slice_index * scale_b))
        idx = min(max(idx, 0), nb - 1)
        plane = e[:, idx, :]                  # shape (na, nc)
        if coord_mode == "cartesian":
            x_lab, y_lab = "c-axis (Å)", "a-axis (Å)"
            xs = (np.arange(nc) / nc) * c_len
            ys = (np.arange(na) / na) * a_len
        else:
            x_lab, y_lab = "frac c", "frac a"
            xs = np.arange(nc) / nc
            ys = np.arange(na) / na
    else:  # c slice
        idx = int(round(slice_index * scale_c))
        idx = min(max(idx, 0), nc - 1)
        plane = e[:, :, idx]                  # shape (na, nb)
        if coord_mode == "cartesian":
            x_lab, y_lab = "b-axis (Å)", "a-axis (Å)"
            xs = (np.arange(nb) / nb) * b_len
            ys = (np.arange(na) / na) * a_len
        else:
            x_lab, y_lab = "frac b", "frac a"
            xs = np.arange(nb) / nb
            ys = np.arange(na) / na

    fig = go.Figure(
        go.Heatmap(
            z=plane,
            x=xs,
            y=ys,
            zmin=float(vmin),
            zmax=float(vmax),
            colorscale=cmap,
            colorbar=dict(title="E (eV)"),
            hovertemplate=(
                f"{x_lab}: %{{x:.3f}}<br>"
                f"{y_lab}: %{{y:.3f}}<br>"
                "E: %{z:.4f} eV<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis_title=x_lab,
        yaxis_title=y_lab,
        height=520,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
    return fig


def render_energy_grid_viewer():
    """Streamlit panel: upload `energy_grid.npz`, pick colormap + range,
    render 3-D volume + 2-D slice views."""
    st.markdown("### 📊 Visualise an existing energy-grid result")
    st.caption(
        "Upload an `energy_grid.npz` produced by a previous run of the "
        "standalone script (the file from "
        "`energy_grid_results/<basename>/energy_grid.npz`) to explore the "
        "3-D energy map without re-computing."
    )

    upload = st.file_uploader(
        "Upload `energy_grid.npz`",
        type=["npz"],
        key="energy_grid_npz_upload",
        help="The standalone script saves this file in the per-structure "
             "results folder. The CSV next to it carries the same numbers "
             "in long form, but the NPZ keeps the 3-D shape needed here.",
    )

    data = _load_energy_grid_npz(upload)
    if data is None:
        return

    energies = np.asarray(data["energies"], dtype=float)
    blocked  = np.asarray(data["blocked"], dtype=bool)
    na, nb, nc = energies.shape
    n_total   = int(energies.size)
    n_blocked = int(blocked.sum())
    n_valid   = n_total - n_blocked

    valid_vals = energies[~blocked]
    if valid_vals.size == 0:
        st.error("All grid points are blocked — nothing to visualise.")
        return

    e_min   = float(valid_vals.min())
    e_max   = float(valid_vals.max())
    e_mean  = float(valid_vals.mean())
    # 1st / 99th percentile clip suppresses outlier blocked-edge points
    # while keeping the bulk of the energy landscape visible.
    e_p01   = float(np.percentile(valid_vals, 1))
    e_p99   = float(np.percentile(valid_vals, 99))

    info_cols = st.columns(4)
    info_cols[0].metric("Grid shape", f"{na} × {nb} × {nc}")
    info_cols[1].metric("Valid pts", f"{n_valid:,}")
    info_cols[2].metric("Blocked pts", f"{n_blocked:,}")
    info_cols[3].metric("Insert element",
                        str(data.get("insert_element", "—")))

    info_cols2 = st.columns(3)
    info_cols2[0].metric("E min (eV)", f"{e_min:.4f}")
    info_cols2[1].metric("E mean (eV)", f"{e_mean:.4f}")
    info_cols2[2].metric("E max (eV)", f"{e_max:.4f}")

    st.markdown("#### 🎨 Colormap, range & coordinates")
    cmap_cols = st.columns([2, 2, 1, 1])
    with cmap_cols[0]:
        cmap = st.selectbox(
            "Colormap", options=_VIEWER_COLORMAPS, index=0,
            key="energy_grid_cmap",
            help="Plotly colorscale used for both the 3-D volume and the "
                 "2-D slice views.",
        )
    with cmap_cols[1]:
        coord_mode_label = st.radio(
            "Coordinates",
            options=["Fractional", "Cartesian (Å)"],
            index=0,
            key="energy_grid_coord_mode",
            horizontal=True,
            help="Fractional → axes are `frac a/b/c ∈ [0, 1)`. The 3-D "
                 "volume render uses this — it needs an axis-aligned grid "
                 "to tessellate. Cartesian → 3-D switches to a coloured "
                 "scatter at each grid point's true `frac @ cell` position "
                 "(works for any cell shape), and the 2-D slice axes are "
                 "rescaled to Å along the in-plane lattice directions.",
        )
    coord_mode = "cartesian" if coord_mode_label.startswith("Cartesian") else "fractional"

    st.markdown("#### ✨ Smoothing & interpolation")
    if not _SCIPY_AVAILABLE:
        st.info("`scipy` is not installed — smoothing & interpolation controls "
                "are disabled. Run `pip install scipy` to enable them.")
        smoothing_sigma = 0.0
        zoom_factor     = 1.0
    else:
        smooth_cols = st.columns(2)
        with smooth_cols[0]:
            smoothing_sigma = st.slider(
                "Gaussian smoothing σ (voxels)",
                min_value=0.0, max_value=3.0, value=0.0, step=0.1,
                key="energy_grid_smoothing_sigma",
                help="Standard deviation (in grid-point units) of an "
                     "isotropic Gaussian applied to the raw energy grid "
                     "with periodic-boundary handling. 0 = off. ~0.5–1.0 "
                     "smooths MLIP noise without losing real basins.",
            )
        with smooth_cols[1]:
            zoom_factor = st.slider(
                "Interpolation factor",
                min_value=1.0, max_value=4.0, value=1.0, step=0.5,
                key="energy_grid_zoom_factor",
                help="Cubic-spline up-sample of the grid by this multiplier "
                     "along each axis (1 = original, 2 = 2× per axis = 8× "
                     "voxels). Renders the volume / heatmap much smoother "
                     "without re-running the MLIP.",
            )
        if zoom_factor > 1.0:
            new_shape = (
                int(round(na * zoom_factor)),
                int(round(nb * zoom_factor)),
                int(round(nc * zoom_factor)),
            )
            new_total = new_shape[0] * new_shape[1] * new_shape[2]
            st.caption(
                f"Up-sampled grid: {new_shape[0]} × {new_shape[1]} × "
                f"{new_shape[2]} = {new_total:,} voxels."
            )

    # Sensible default: percentile-clipped so outliers don't squash the
    # interesting low-energy region; user can override.
    default_lo = round(e_p01, 4)
    default_hi = round(e_p99, 4)
    with cmap_cols[2]:
        vmin = st.number_input(
            "Min (eV)",
            value=default_lo,
            step=0.01, format="%.4f",
            key="energy_grid_vmin",
            help=f"Lower colour-bar limit. Default = 1st-percentile "
                 f"({default_lo:.4f} eV) of valid points.",
        )
    with cmap_cols[3]:
        vmax = st.number_input(
            "Max (eV)",
            value=default_hi,
            step=0.01, format="%.4f",
            key="energy_grid_vmax",
            help=f"Upper colour-bar limit. Default = 99th-percentile "
                 f"({default_hi:.4f} eV) of valid points.",
        )
    if vmax <= vmin:
        st.warning("Max must be greater than Min — falling back to data range.")
        vmin, vmax = e_min, e_max

    tab_vol, tab_slice = st.tabs(["🌐 3-D volume", "🟦 2-D slice heatmap"])

    with tab_vol:
        if coord_mode == "cartesian":
            st.caption(
                "Cartesian mode shows each grid point as a coloured marker "
                "at its true `frac @ cell` position — works for any cell "
                "shape (orthogonal or not). Drop the opacity if the inner "
                "structure is hidden by the surface points."
            )
            vol_cols = st.columns(2)
            with vol_cols[0]:
                opacity = st.slider(
                    "Marker opacity",
                    min_value=0.05, max_value=1.0, value=0.7, step=0.05,
                    key="energy_grid_volume_opacity",
                )
            with vol_cols[1]:
                point_size = st.slider(
                    "Marker size",
                    min_value=1, max_value=15, value=4, step=1,
                    key="energy_grid_point_size",
                    help="Marker size in pixels for the Cartesian scatter "
                         "rendering.",
                )
            surface_count = 15  # unused in Cartesian mode
        else:
            st.caption(
                "Fractional mode uses an axis-aligned cube — Plotly's "
                "volume tessellation can interpolate across iso-surfaces, "
                "so this view is the most informative for spotting "
                "low-energy basins."
            )
            vol_cols = st.columns(2)
            with vol_cols[0]:
                opacity = st.slider(
                    "Volume opacity",
                    min_value=0.05, max_value=1.0, value=0.15, step=0.05,
                    key="energy_grid_volume_opacity",
                    help="Lower values make the inner structure visible "
                         "through outer iso-surfaces.",
                )
            with vol_cols[1]:
                surface_count = st.slider(
                    "Iso-surface count",
                    min_value=4, max_value=40, value=15, step=1,
                    key="energy_grid_surface_count",
                    help="Number of iso-surfaces plotly stacks to "
                         "approximate the volume — more is smoother but "
                         "slower.",
                )
            point_size = 4  # unused in fractional mode

        try:
            fig_vol = _make_volume_figure(
                data, cmap, vmin, vmax, opacity, surface_count,
                coord_mode=coord_mode, point_size=point_size,
                smoothing_sigma=smoothing_sigma, zoom_factor=zoom_factor,
            )
            st.plotly_chart(fig_vol, width="stretch",
                            key="energy_grid_volume_plot")
        except Exception as exc:
            st.error(f"❌ 3-D render failed: {exc}")

    with tab_slice:
        sl_cols = st.columns([2, 3])
        with sl_cols[0]:
            axis_choice = st.selectbox(
                "Slice axis",
                options=[
                    "a (slice on a-axis, plot b vs c)",
                    "b (slice on b-axis, plot a vs c)",
                    "c (slice on c-axis, plot a vs b)",
                ],
                index=2,
                key="energy_grid_slice_axis",
            )
            axis_idx_max = {
                "a (slice on a-axis, plot b vs c)": na - 1,
                "b (slice on b-axis, plot a vs c)": nb - 1,
                "c (slice on c-axis, plot a vs b)": nc - 1,
            }[axis_choice]
            slice_index = st.slider(
                "Slice index",
                min_value=0, max_value=int(axis_idx_max),
                value=int(axis_idx_max // 2),
                step=1,
                key="energy_grid_slice_idx",
            )
        with sl_cols[1]:
            try:
                fig_slice = _make_slice_figure(
                    data, axis_choice, slice_index, cmap, vmin, vmax,
                    coord_mode=coord_mode,
                    smoothing_sigma=smoothing_sigma,
                    zoom_factor=zoom_factor,
                )
                st.plotly_chart(fig_slice, width="stretch",
                                key="energy_grid_slice_plot")
            except Exception as exc:
                st.error(f"❌ Slice render failed: {exc}")

    base_e = data.get("base_energy_eV")
    if base_e is not None and not (isinstance(base_e, float)
                                   and np.isnan(base_e)):
        st.caption(
            f"Baseline host energy stored in NPZ: **{float(base_e):.6f} eV** "
            "(useful for computing binding-energy maps "
            "`E(grid) − E_host − E_atom`)."
        )

    st.markdown("#### 💾 Download volumetric file for VESTA / XCrysDen")
    st.caption(
        "VESTA does **not** auto-pick an iso-level from XSF data — it "
        "defaults to ~0.05 (a charge-density value), which for an energy "
        "field in the −60 eV range sits orders of magnitude outside the "
        "data and produces *0 polygons*. Two options below help: (1) "
        "exporting as Gaussian **CUBE** — VESTA picks a sensible default "
        "from cube files; (2) shifting the energies so the minimum is at "
        "0 — the default iso-level then falls inside the data range. "
        "Either way, the suggested iso-level is printed below the button."
    )
    dl_cols = st.columns([2, 2, 2])
    with dl_cols[0]:
        export_fmt = st.radio(
            "File format",
            options=["CUBE (.cube)", "XSF (.xsf)"],
            index=0,
            key="energy_grid_export_fmt",
            help="**CUBE** — Gaussian volumetric format. Almost always the "
                 "easier one to open in VESTA / Avogadro / GaussView.\n\n"
                 "**XSF** — XCrySDen's structure + volumetric format. "
                 "VESTA can read it, but iso-levels almost always need to "
                 "be set by hand.",
        )
    with dl_cols[1]:
        shift_to_zero = st.checkbox(
            "Shift energies so min = 0",
            value=True,
            key="energy_grid_shift_to_zero",
            help="Subtracts the minimum energy from every grid point "
                 "before writing the file. The data range becomes "
                 "[0, max−min], which is where VESTA's default iso-level "
                 "of ~0.05 actually intersects something. The subtracted "
                 "value is recorded in the file's comment line so you can "
                 "recover absolute energies later.",
        )
    with dl_cols[2]:
        st.markdown(
            "**After loading in VESTA:**\n\n"
            "1. *Edit ▸ Edit Data ▸ Volumetric Data*\n"
            "2. **Surfaces** tab → **New**\n"
            "3. Enter the iso-level shown below"
        )

    # Build + show the file
    try:
        if export_fmt.startswith("CUBE"):
            txt = _build_cube_from_loaded_npz(
                data,
                smoothing_sigma=smoothing_sigma,
                zoom_factor=zoom_factor,
                shift_to_zero=shift_to_zero,
            )
            fname = "energy_grid.cube"
            mime  = "chemical/x-gaussian-cube"
        else:
            txt = _build_xsf_from_loaded_npz(
                data,
                smoothing_sigma=smoothing_sigma,
                zoom_factor=zoom_factor,
                shift_to_zero=shift_to_zero,
            )
            fname = "energy_grid.xsf"
            mime  = "chemical/x-xcrysden-structured-file"

        # Compute the iso-level numbers VESTA should see.
        offset = float(e_min) if shift_to_zero else 0.0
        suggested_iso = float(e_mean) - offset
        iso_min_disp  = float(e_min)  - offset
        iso_max_disp  = float(e_max)  - offset

        st.download_button(
            label=f"💾 Download {fname}",
            data=txt.encode("utf-8"),
            file_name=fname,
            mime=mime,
            key=f"energy_grid_download_{fname}",
            type="primary",
        )
        if shift_to_zero:
            st.info(
                f"**File range (after shift): {iso_min_disp:.4f} → "
                f"{iso_max_disp:.4f}** (eV above the absolute minimum "
                f"{e_min:.4f} eV).\n\n"
                f"**Suggested iso-level to type into VESTA: "
                f"`{suggested_iso:.3f}`** (data mean). For a tighter "
                f"basin contour try `{iso_min_disp + 0.1*(iso_max_disp - iso_min_disp):.3f}`."
            )
        else:
            st.info(
                f"**File range (absolute eV): {iso_min_disp:.4f} → "
                f"{iso_max_disp:.4f}**.\n\n"
                f"**Suggested iso-level to type into VESTA: "
                f"`{suggested_iso:.3f}`** (data mean). Iso-levels outside "
                f"this range will draw nothing."
            )
    except Exception as exc:
        st.error(f"❌ File build failed: {exc}")


def setup_energy_grid_scan_ui(default_settings=None, save_settings_function=None):
    defaults = default_settings.get("energy_grid_scan", {}) if default_settings else {}

    st.subheader("Energy Grid Scan Parameters")
    st.caption(
        "Insert a probe atom at every point of a 3-D grid spanning the unit cell "
        "and record the single-point energy at each location. Useful for mapping "
        "interstitial sites, adsorption pockets, ion-migration channels, etc."
    )

    col1, col2 = st.columns(2)
    with col1:
        insert_element = st.text_input(
            "Element to insert",
            value=defaults.get("insert_element", "H"),
            help="Chemical symbol of the probe atom placed at each grid point "
                 "(e.g. H, Li, Na, He). Must be a valid ASE element symbol.",
        ).strip() or "H"
    with col2:
        grid_spacing = st.number_input(
            "Grid spacing (Å)",
            min_value=0.05, max_value=5.0,
            value=float(defaults.get("grid_spacing_A", 0.5)),
            step=0.05, format="%.2f",
            help="Target real-space spacing along each lattice direction. "
                 "The number of grid points per axis is set to "
                 "`ceil(|a_i| / spacing)`, so non-orthogonal cells get a grid "
                 "that's still roughly uniform in 3-D Euclidean distance.",
        )

    col3, col4 = st.columns(2)
    with col3:
        min_distance = st.number_input(
            "Min. distance to existing atoms (Å)",
            min_value=0.0, max_value=10.0,
            value=float(defaults.get("min_distance_A", 1.0)),
            step=0.05, format="%.3f",
            help="Grid points closer than this (under PBC minimum-image "
                 "convention) to any atom of the host structure are skipped — "
                 "no energy is computed, the blocked-energy sentinel is stored "
                 "in the output array at that location instead.",
        )
    with col4:
        blocked_energy = st.number_input(
            "Blocked-point energy sentinel (eV)",
            min_value=1.0, max_value=1e12,
            value=float(defaults.get("blocked_energy", 1e9)),
            step=1.0, format="%.6g",
            help="Value written to the output 3-D array (and CSV) at grid "
                 "points that were skipped because they would have overlapped "
                 "an existing atom. Choose something much larger than any "
                 "physically meaningful energy so blocked points are obvious "
                 "in visualisations / contour plots.",
        )

    col5, col6 = st.columns(2)
    with col5:
        compute_baseline = st.checkbox(
            "Compute baseline energy of unmodified host",
            value=bool(defaults.get("compute_baseline", True)),
            help="When on, the script also evaluates the pristine host energy "
                 "(no probe atom added) once per structure and stores it in "
                 "the NPZ metadata, so you can later plot a *binding energy* "
                 "map `E(grid) − (E_host + E_atom)` if you also know the "
                 "isolated-atom reference.",
        )
    with col6:
        save_xsf = st.checkbox(
            "Also write XSF volumetric file",
            value=bool(defaults.get("save_xsf", True)),
            help="Writes a `BLOCK_DATAGRID_3D` XSF file that VESTA / XCrysDen "
                 "can open directly as an iso-surface / slice plot. Blocked "
                 "points carry the sentinel value, so set a colour-bar cap "
                 "below it when visualising.",
        )

    col7, col8 = st.columns(2)
    with col7:
        save_min_structure = st.checkbox(
            "Save lowest-energy interstitial structure",
            value=bool(defaults.get("save_min_structure", True)),
            help="Writes `min_energy_structure.cif` and `.vasp` per structure "
                 "— the host with the probe atom placed at the grid point of "
                 "lowest computed energy. When several grid points are within "
                 "`degeneracy tolerance` of the minimum (symmetry-equivalent "
                 "sites or MLIP noise), one is picked at random.",
        )
    with col8:
        degen_tol_eV = st.number_input(
            "Degeneracy tolerance (eV)",
            min_value=0.0, max_value=1.0,
            value=float(defaults.get("degen_tol_eV", 1e-3)),
            step=1e-4, format="%.4f",
            disabled=not save_min_structure,
            help="Grid points whose energy is within this tolerance of the "
                 "global minimum are considered equivalent and one is chosen "
                 "uniformly at random. Default 1 meV — small enough that "
                 "MLIP-noise-driven ties get bundled, large enough that "
                 "physically-equivalent symmetry-related basins do too.",
        )

    col9, _col10 = st.columns(2)
    with col9:
        print_every = st.number_input(
            "Console print frequency (every N grid points)",
            min_value=1, max_value=1000000,
            value=int(defaults.get("print_every", 1)),
            step=1,
            help="Print a progress line to the console only once every N grid "
                 "points. 1 = print every point (most verbose); larger values "
                 "cut console spam on big grids. The first point, the final "
                 "point, and any FAILED point are always printed regardless.",
        )

    st.markdown("---")
    region_enabled = st.checkbox(
        "Restrict scan to a sub-region (cube)",
        value=bool(defaults.get("region_enabled", False)),
        help="When on, the grid only covers a cube instead of the whole unit "
             "cell. You set the cube's centre and its edge length; each can be "
             "given independently in fractional (relative) coordinates or in "
             "Cartesian Ångström (see the unit toggles below). The number of "
             "grid points per axis is still `ceil(span·|aᵢ| / spacing)`, "
             "keeping the real-space spacing roughly uniform. The cube is "
             "clamped to stay inside the cell.",
    )

    # Independent unit toggles for the cube centre and the edge length, so the
    # user can mix-and-match (e.g. centre in fractional, length in Å).
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        center_mode_label = st.radio(
            "Cube centre units",
            options=["Fractional", "Cartesian (Å)"],
            index=0 if defaults.get("region_center_mode", "fractional") ==
                  "fractional" else 1,
            key="energy_grid_region_center_mode",
            horizontal=True,
            disabled=not region_enabled,
            help="Fractional → centre given as (a, b, c) ∈ [0, 1]. "
                 "Cartesian → centre given in Å; converted to fractional per "
                 "structure via that structure's cell matrix.",
        )
    with mcol2:
        length_mode_label = st.radio(
            "Edge length units",
            options=["Fractional", "Cartesian (Å)"],
            index=0 if defaults.get("region_length_mode", "fractional") ==
                  "fractional" else 1,
            key="energy_grid_region_length_mode",
            horizontal=True,
            disabled=not region_enabled,
            help="Fractional → edge length as a fraction of each lattice "
                 "vector (a true cube only for a cubic cell). Cartesian → "
                 "edge length in Å, giving a real-space cube (the fractional "
                 "span on axis i becomes length / |aᵢ|).",
        )
    center_mode = ("cartesian" if center_mode_label.startswith("Cartesian")
                   else "fractional")
    length_mode = ("cartesian" if length_mode_label.startswith("Cartesian")
                   else "fractional")

    def _clamp(v, lo, hi):
        return float(min(hi, max(lo, v)))

    # Bounds / labels follow the selected units. Cartesian has no fixed upper
    # bound (cells vary), so we allow a generous range. The widgets carry no
    # explicit `key`, so changing the unit changes the label and thus the
    # widget identity — Streamlit re-seeds them from the (clamped) defaults
    # rather than reusing a now-out-of-range value from the other unit.
    if center_mode == "cartesian":
        c_min, c_max, c_step, c_unit = 0.0, 1.0e4, 0.1, "Å"
    else:
        c_min, c_max, c_step, c_unit = 0.0, 1.0, 0.05, "frac"

    ccol1, ccol2, ccol3, ccol4 = st.columns(4)
    _ctr = defaults.get("region_center", [0.5, 0.5, 0.5]) or [0.5, 0.5, 0.5]
    with ccol1:
        region_center_a = st.number_input(
            f"Centre a ({c_unit})", min_value=c_min, max_value=c_max,
            value=_clamp(float(_ctr[0]), c_min, c_max),
            step=c_step, format="%.3f",
            disabled=not region_enabled,
            help=f"{'Cartesian x' if center_mode == 'cartesian' else 'Fractional a'}"
                 "-coordinate of the cube centre.",
        )
    with ccol2:
        region_center_b = st.number_input(
            f"Centre b ({c_unit})", min_value=c_min, max_value=c_max,
            value=_clamp(float(_ctr[1]), c_min, c_max),
            step=c_step, format="%.3f",
            disabled=not region_enabled,
            help=f"{'Cartesian y' if center_mode == 'cartesian' else 'Fractional b'}"
                 "-coordinate of the cube centre.",
        )
    with ccol3:
        region_center_c = st.number_input(
            f"Centre c ({c_unit})", min_value=c_min, max_value=c_max,
            value=_clamp(float(_ctr[2]), c_min, c_max),
            step=c_step, format="%.3f",
            disabled=not region_enabled,
            help=f"{'Cartesian z' if center_mode == 'cartesian' else 'Fractional c'}"
                 "-coordinate of the cube centre.",
        )
    with ccol4:
        if length_mode == "cartesian":
            l_min, l_max, l_step, l_unit = 0.01, 1.0e4, 0.1, "Å"
        else:
            l_min, l_max, l_step, l_unit = 0.01, 1.0, 0.05, "frac"
        region_length = st.number_input(
            f"Edge length ({l_unit})", min_value=l_min, max_value=l_max,
            value=_clamp(float(defaults.get("region_length", 0.5)),
                         l_min, l_max),
            step=l_step, format="%.3f",
            disabled=not region_enabled,
            help="Cube edge length. The scan spans "
                 "[centre − length/2, centre + length/2] along each axis "
                 "(clamped to the cell). " +
                 ("Given in Å → a real-space cube."
                  if length_mode == "cartesian"
                  else "Given as a fraction of each lattice vector."),
        )

    st.markdown("---")
    relax_each_point = st.checkbox(
        "Relax structure at each grid point (geometry optimization)",
        value=bool(defaults.get("relax_each_point", False)),
        help="When on, a constrained geometry optimization is run at every "
             "non-blocked grid point and the *relaxed* energy is recorded "
             "instead of the single-point energy. The inserted probe atom is "
             "kept fixed at the grid point, and the host atom that is farthest "
             "(under the PBC minimum-image convention) from the probe is also "
             "kept fixed — this removes the rigid-body translation that would "
             "otherwise let the whole cell drift. All remaining atoms are free "
             "to move. This is far more expensive than a single-point scan.",
    )

    rcol1, rcol2, rcol3 = st.columns(3)
    with rcol1:
        relax_optimizer = st.selectbox(
            "Optimizer",
            ["BFGS", "LBFGS", "FIRE", "LBFGSLineSearch", "MDMin"],
            index=["BFGS", "LBFGS", "FIRE", "LBFGSLineSearch", "MDMin"].index(
                defaults.get("relax_optimizer", "LBFGS")
            ) if defaults.get("relax_optimizer", "LBFGS") in
                ["BFGS", "LBFGS", "FIRE", "LBFGSLineSearch", "MDMin"] else 1,
            disabled=not relax_each_point,
            help="ASE optimizer used for the per-grid-point relaxation. "
                 "LBFGS (default) / BFGS are good general choices; FIRE is "
                 "robust for difficult, highly non-linear cases.",
        )
    with rcol2:
        relax_fmax = st.number_input(
            "Force convergence fmax (eV/Å)",
            min_value=0.001, max_value=2.0,
            value=float(defaults.get("relax_fmax", 0.01)),
            step=0.01, format="%.3f",
            disabled=not relax_each_point,
            help="Optimization stops once the maximum force on any free atom "
                 "drops below this threshold.",
        )
    with rcol3:
        relax_max_steps = st.number_input(
            "Max optimization steps",
            min_value=1, max_value=10000,
            value=int(defaults.get("relax_max_steps", 500)),
            step=10,
            disabled=not relax_each_point,
            help="Hard cap on optimizer iterations per grid point. The last "
                 "energy is recorded even if convergence was not reached.",
        )

    params = {
        "insert_element": insert_element,
        "grid_spacing_A": float(grid_spacing),
        "min_distance_A": float(min_distance),
        "blocked_energy": float(blocked_energy),
        "print_every": int(print_every),
        "compute_baseline": bool(compute_baseline),
        "save_xsf": bool(save_xsf),
        "save_min_structure": bool(save_min_structure),
        "degen_tol_eV": float(degen_tol_eV),
        "relax_each_point": bool(relax_each_point),
        "relax_optimizer": str(relax_optimizer),
        "relax_fmax": float(relax_fmax),
        "relax_max_steps": int(relax_max_steps),
        "region_enabled": bool(region_enabled),
        "region_center": [
            float(region_center_a),
            float(region_center_b),
            float(region_center_c),
        ],
        "region_center_mode": center_mode,
        "region_length": float(region_length),
        "region_length_mode": length_mode,
    }

    if save_settings_function is not None and default_settings is not None:
        default_settings["energy_grid_scan"] = params
        try:
            save_settings_function(default_settings)
        except Exception:
            pass

    with st.expander("📋 Parameter summary", expanded=False):
        st.json(params)

    return params


def generate_energy_grid_scan_script(
    scan_params,
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
    """Build a fully self-contained Python script as a string.

    Calculator setup is delegated to ``_generate_calculator_setup_code`` (the
    same routine used by the project's main script generator), so this works
    with every model the GUI supports — including custom MACE / MACE-POLAR /
    UPET paths and dispersion settings — without re-implementing the model
    detection logic here.
    """
    elem        = scan_params.get("insert_element", "H")
    spacing     = float(scan_params.get("grid_spacing_A", 0.5))
    min_dist    = float(scan_params.get("min_distance_A", 1.0))
    blocked_e   = float(scan_params.get("blocked_energy", 1e9))
    do_base     = bool(scan_params.get("compute_baseline", True))
    save_xsf    = bool(scan_params.get("save_xsf", True))
    save_min    = bool(scan_params.get("save_min_structure", True))
    degen_tol_e = float(scan_params.get("degen_tol_eV", 1e-3))
    relax_each  = bool(scan_params.get("relax_each_point", False))
    relax_opt   = str(scan_params.get("relax_optimizer", "LBFGS"))
    relax_fmax  = float(scan_params.get("relax_fmax", 0.01))
    relax_steps = int(scan_params.get("relax_max_steps", 500))
    print_every = max(1, int(scan_params.get("print_every", 1)))
    region_on   = bool(scan_params.get("region_enabled", False))
    _rc         = scan_params.get("region_center", [0.5, 0.5, 0.5]) or [0.5, 0.5, 0.5]
    region_ctr  = (float(_rc[0]), float(_rc[1]), float(_rc[2]))
    region_ctr_mode = str(scan_params.get("region_center_mode", "fractional"))
    region_len  = float(scan_params.get("region_length", 0.5))
    region_len_mode = str(scan_params.get("region_length_mode", "fractional"))

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

    config_info = ""
    if mace_head:
        config_info += f"\nMACE Head: {mace_head}"
    if mace_dispersion:
        config_info += f"\nDispersion: D3-{mace_dispersion_xc}"
    if is_mace_polar:
        _ps = polar_settings or {}
        config_info += (
            f"\nMACE-POLAR-1: charge={_ps.get('charge', 0)}, "
            f"spin={_ps.get('spin', 1)}, "
            f"E_field={_ps.get('external_field', [0.0, 0.0, 0.0])} V/Å"
        )

    polar_call_args = ""
    if is_mace_polar:
        _ps = polar_settings or {}
        polar_call_args = (
            f", charges=[{_ps.get('charge', 0)}], "
            f"spins=[{_ps.get('spin', 1)}], "
            f"external_fields=[{_ps.get('external_field', [0.0, 0.0, 0.0])}]"
        )

    # Human-readable summary of every setting chosen in the GUI, written into
    # the script header so a saved script is self-documenting.
    if region_on:
        region_desc = (
            f"sub-region cube (centre={region_ctr} [{region_ctr_mode}], "
            f"edge length={region_len} [{region_len_mode}])"
        )
    else:
        region_desc = "full unit cell"
    if relax_each:
        relax_desc = (
            f"ON (optimizer={relax_opt}, fmax={relax_fmax} eV/Å, "
            f"max_steps={relax_steps})"
        )
    else:
        relax_desc = "OFF (single-point energies)"
    settings_info = f"""

Scan settings
-------------
  Probe element     : {elem}
  Grid spacing      : {spacing} Å
  Min. distance     : {min_dist} Å (blocking radius around host atoms)
  Blocked sentinel  : {blocked_e:g} eV
  Scan region       : {region_desc}
  Relaxation        : {relax_desc}
  Compute baseline  : {do_base}
  Degeneracy tol.   : {degen_tol_e} eV
  Save XSF          : {save_xsf}
  Save min. struct. : {save_min}
  Console output    : every {print_every} grid point(s)
  Threads (OMP)     : {thread_count}"""

    return f'''#!/usr/bin/env python3
"""Standalone Energy Grid Scan.

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {selected_model_key or model_size}
Device: {device}
Precision: {dtype}{config_info}
{settings_info}

Places an *{elem}* probe atom at every point of an `na × nb × nc`
fractional grid spanning each host unit cell, computes the single-point
energy at each location, and saves the full 3-D map.

Outputs (per structure, under `energy_grid_results/<basename>/`):
  • energy_grid.npz   — 3-D array of energies, blocked-mask, metadata
  • energy_grid.csv   — long-form table with one row per grid point
  • energy_grid.xsf   — optional VESTA / XCrysDen volumetric file
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from pathlib import Path


def _fmt_duration(seconds):
    """Format a duration in seconds as a compact human-readable string."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{{seconds:5.1f}} s"
    if seconds < 3600:
        return f"{{seconds/60:5.1f}} min"
    return f"{{seconds/3600:5.2f}} h"

# Threading must be set BEFORE torch is imported.
os.environ["OMP_NUM_THREADS"] = "{thread_count}"

import torch
torch.set_num_threads({thread_count})

# ── ASE / pymatgen ──────────────────────────────────────────────────────
from ase import Atom
from ase.io import read
from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE, LBFGS, LBFGSLineSearch, MDMin

# ── MLIP imports (matches the GUI's available-models block) ─────────────
{mlip_imports_code}

# ── Parameters chosen in the GUI ────────────────────────────────────────
INSERT_ELEMENT   = "{elem}"
GRID_SPACING_A   = {spacing}      # target real-space spacing (Å)
MIN_DISTANCE_A   = {min_dist}      # blocking radius around host atoms (Å)
BLOCKED_ENERGY   = {blocked_e}    # sentinel written at blocked points (eV)
PRINT_EVERY      = {print_every}        # print a progress line every N grid points
COMPUTE_BASELINE = {do_base}
SAVE_XSF         = {save_xsf}
SAVE_MIN_STRUCT  = {save_min}       # dump host + probe at the global E_min site
DEGEN_TOL_EV     = {degen_tol_e}    # grid points within this of E_min count as degenerate
RELAX_EACH_POINT = {relax_each}       # relax structure at each grid point, record relaxed E
RELAX_OPTIMIZER  = "{relax_opt}"      # ASE optimizer used for per-point relaxation
RELAX_FMAX       = {relax_fmax}       # force-convergence threshold (eV/Å)
RELAX_MAX_STEPS  = {relax_steps}      # max optimizer steps per grid point
REGION_ENABLED   = {region_on}        # restrict the scan to a sub-region cube
REGION_CENTER    = {region_ctr}    # cube centre (units set by REGION_CENTER_MODE)
REGION_CENTER_MODE = "{region_ctr_mode}"  # "fractional" or "cartesian" (Å)
REGION_LENGTH    = {region_len}       # cube edge length (units set by REGION_LENGTH_MODE)
REGION_LENGTH_MODE = "{region_len_mode}"  # "fractional" or "cartesian" (Å)
IS_MACE_POLAR    = {is_mace_polar}
OUTPUT_ROOT      = Path("energy_grid_results")


def _make_optimizer(name, obj):
    n = name.upper()
    if n == "LBFGS":           return LBFGS(obj, logfile=None)
    if n == "LBFGSLINESEARCH": return LBFGSLineSearch(obj, logfile=None)
    if n == "FIRE":            return FIRE(obj, logfile=None)
    if n == "MDMIN":           return MDMin(obj, logfile=None)
    return BFGS(obj, logfile=None)


def _write_xsf(path, atoms, energies_3d, blocked, origin, cell, grid_cell=None):
    """Write a BLOCK_DATAGRID_3D XSF file readable by VESTA / XCrysDen.

    ``cell`` is the host's full unit cell (used for PRIMVEC / atom drawing);
    ``grid_cell`` is the spanning vectors of the datagrid box (equal to
    ``cell`` for a full-cell scan, or the scaled sub-cube when a region is
    set) and ``origin`` its Cartesian origin.

    * Blocked points (the huge BLOCKED_ENERGY sentinel) are replaced with
      the median of the *valid* energies before writing — otherwise VESTA
      computes its default iso-level as `(min + max) / 2`, which lands
      somewhere near 5×10⁸ eV when the sentinel is 10⁹, and reports
      "0 polygons" on every iso-surface because no point is anywhere
      near that value.
    * The grid is emitted with periodic wrap-around (Na+1, Nb+1, Nc+1
      points, last slice = first slice), which is the XSF spec's
      convention — VESTA needs this to close iso-surfaces at the
      unit-cell boundaries.
    """
    arr  = np.asarray(energies_3d, dtype=float)
    mask = np.asarray(blocked, dtype=bool)
    valid = arr[~mask]
    fill = float(np.median(valid)) if valid.size else 0.0
    arr = np.where(mask, fill, arr)
    if grid_cell is None:
        grid_cell = cell

    # Periodic wrap along each axis.
    arr = np.concatenate([arr, arr[:1, :, :]], axis=0)
    arr = np.concatenate([arr, arr[:, :1, :]], axis=1)
    arr = np.concatenate([arr, arr[:, :, :1]], axis=2)

    na, nb, nc = arr.shape   # already (na+1, nb+1, nc+1)
    with open(path, "w") as fh:
        # Leading comment block — matches the format VESTA loads cleanly.
        fh.write("#\\n# Energy grid scan (eV)\\n#\\n")
        fh.write("CRYSTAL\\n")
        fh.write("PRIMVEC\\n")
        for v in cell:
            fh.write(f"  {{v[0]:.7f}}   {{v[1]:.7f}}   {{v[2]:.7f}}\\n")
        fh.write("PRIMCOORD\\n")
        fh.write(f"{{len(atoms)}} 1\\n")
        for atm in atoms:
            p = atm.position
            fh.write(
                f"{{atm.symbol:>3s}}   {{p[0]:.7f}}   {{p[1]:.7f}}   {{p[2]:.7f}}\\n"
            )
        fh.write("BEGIN_BLOCK_DATAGRID_3D\\n")
        fh.write("energy_grid\\n")
        fh.write("BEGIN_DATAGRID_3D_energy\\n")
        # No indent on grid dims / origin / vectors — and the data is
        # written in PLAIN DECIMAL (not scientific) one value per line,
        # because VESTA's XSF parser silently drops the volumetric trace
        # when it encounters `1.23e+01`-style numbers in some versions.
        fh.write(f"{{na}} {{nb}} {{nc}}\\n")
        fh.write(f"{{origin[0]:.7f}} {{origin[1]:.7f}} {{origin[2]:.7f}}\\n")
        for v in grid_cell:
            fh.write(f"  {{v[0]:.7f}}   {{v[1]:.7f}}   {{v[2]:.7f}}\\n")
        # XSF needs index 1 (along a) fastest, index 3 (along c) slowest.
        # We have (na, nb, nc) → transpose to (nc, nb, na), flatten C-order.
        flat = arr.transpose(2, 1, 0).flatten()
        for v in flat:
            fh.write(f"  {{float(v):.7f}}\\n")
        fh.write("END_DATAGRID_3D\\n")
        fh.write("END_BLOCK_DATAGRID_3D\\n")


def main():
    start_time = time.time()
    print("🚀 Energy Grid Scan")
    print(f"📅 {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
    print(f"🤖 Model: {selected_model_key or model_size}  (device {device})")
    print(f"🎯 Probe element: {{INSERT_ELEMENT}}")
    print(f"📐 Grid spacing:  {{GRID_SPACING_A}} Å")
    print(f"🛑 Min. distance: {{MIN_DISTANCE_A}} Å")
    if REGION_ENABLED:
        print(
            f"📦 Sub-region: centre={{REGION_CENTER}} ({{REGION_CENTER_MODE}}), "
            f"edge length={{REGION_LENGTH}} ({{REGION_LENGTH_MODE}})"
        )
    else:
        print("📦 Scan region: full unit cell")
    print(f"🚫 Blocked sentinel: {{BLOCKED_ENERGY:g}} eV")
    if RELAX_EACH_POINT:
        print(
            f"🧲 Relaxation: ON  (optimizer={{RELAX_OPTIMIZER}}, "
            f"fmax={{RELAX_FMAX}} eV/Å, max_steps={{RELAX_MAX_STEPS}}; "
            f"probe + farthest host atom fixed)"
        )
    else:
        print("🧲 Relaxation: OFF  (single-point energies)")
    if PRINT_EVERY > 1:
        print(f"🖨️  Console output: every {{PRINT_EVERY}} grid points "
              f"(plus first / last / failed)")
    else:
        print("🖨️  Console output: every grid point")
    print(f"🧵 OMP threads: {{os.environ.get('OMP_NUM_THREADS')}}")

    # ── Calculator setup (model-agnostic — generated from GUI selection) ──
    print("\\n🔧 Setting up MLIP calculator...")
{calculator_setup_code}

    # Holds convergence info from the most recent relaxation so the main loop
    # can report it without changing _evaluate_energy's return signature.
    # Keys: "converged" (bool/None) and "steps" (int). Empty when the last
    # call ran no relaxation (e.g. the baseline or a single-point scan).
    relax_status = {{}}

    # Nested closure so `calculator` (defined in this scope by the setup
    # block above) is captured directly — no global / sys.modules tricks.
    def _evaluate_energy(atoms, fixed_indices=None):
        atoms.calc = calculator
        relax_status.clear()
        # Optional constrained geometry optimization. The probe atom and the
        # host atom farthest from it (PBC minimum-image) are held fixed; all
        # other atoms relax. The relaxed energy is then returned.
        if RELAX_EACH_POINT and fixed_indices:
            atoms.set_constraint(FixAtoms(indices=sorted(set(fixed_indices))))
            opt = _make_optimizer(RELAX_OPTIMIZER, atoms)
            opt.run(fmax=RELAX_FMAX, steps=RELAX_MAX_STEPS)
            # `opt.converged()` re-checks the max force against fmax (reliable
            # across ASE versions); `opt.nsteps` is the iterations taken.
            try:
                relax_status["converged"] = bool(opt.converged())
            except Exception:
                relax_status["converged"] = None
            relax_status["steps"] = int(getattr(opt, "nsteps", 0))
        if IS_MACE_POLAR:
            try:
                return float(
                    calculator.get_potential_energy(atoms{polar_call_args})
                )
            except TypeError:
                return float(atoms.get_potential_energy())
        return float(atoms.get_potential_energy())

    # ── Structure discovery ────────────────────────────────────────────
    structure_files = sorted([
        f for f in os.listdir(".")
        if f.startswith("POSCAR")
        or f.endswith(".vasp") or f.endswith(".poscar")
        or f.endswith(".cif")
    ])
    if not structure_files:
        print("❌ No structure files (.cif / .vasp / POSCAR*) found in cwd")
        sys.exit(1)
    print(f"\\n📂 Found {{len(structure_files)}} structure(s): {{structure_files}}")

    OUTPUT_ROOT.mkdir(exist_ok=True)

    for fname in structure_files:
        print(f"\\n🔬 Processing {{fname}}")
        t0 = time.time()
        try:
            host = read(fname)
        except Exception as exc:
            print(f"  ⚠ Could not read {{fname}}: {{exc}}")
            continue

        cell = np.array(host.get_cell())
        a_len = float(np.linalg.norm(cell[0]))
        b_len = float(np.linalg.norm(cell[1]))
        c_len = float(np.linalg.norm(cell[2]))

        # Fractional lower bound (lo_*) and span per lattice direction. For a
        # full-cell scan these are 0 and 1, reproducing the original grid
        # (frac = i/n over [0, 1)). For a restricted cube the centre and the
        # edge length are each resolved to fractional coordinates according to
        # their own units toggle, then the box is clamped to stay inside [0, 1].
        if REGION_ENABLED:
            # Cube centre → fractional. A Cartesian (Å) centre is mapped through
            # the cell matrix (cart = frac @ cell  ⇒  frac solves cellᵀ·frac = cart).
            if REGION_CENTER_MODE == "cartesian":
                frac_ctr = np.linalg.solve(
                    cell.T, np.array(REGION_CENTER, dtype=float)
                )
            else:
                frac_ctr = np.array(REGION_CENTER, dtype=float)

            # Cube edge length → fractional span per lattice axis. A Cartesian
            # length in Å maps to span_i = length / |a_i| (a real-space cube);
            # a fractional length is the same span on every axis.
            if REGION_LENGTH_MODE == "cartesian":
                raw_span = [
                    REGION_LENGTH / a_len,
                    REGION_LENGTH / b_len,
                    REGION_LENGTH / c_len,
                ]
            else:
                raw_span = [REGION_LENGTH, REGION_LENGTH, REGION_LENGTH]

            lo_xyz, span_xyz = [], []
            for c_frac, span in zip(frac_ctr, raw_span):
                span = float(min(1.0, max(1e-6, span)))
                half = span / 2.0
                lo_i = float(c_frac) - half
                hi_i = float(c_frac) + half
                if lo_i < 0.0:
                    lo_i, hi_i = 0.0, span
                if hi_i > 1.0:
                    hi_i, lo_i = 1.0, 1.0 - span
                lo_xyz.append(max(0.0, lo_i))
                span_xyz.append(span)
            lo_a, lo_b, lo_c = lo_xyz
            sa, sb, sc = span_xyz
            print(
                f"  Sub-region cube: centre="
                f"{{tuple(round(float(x), 3) for x in frac_ctr)}} (frac), "
                f"span=({{sa:.3f}},{{sb:.3f}},{{sc:.3f}}) (frac)  →  "
                f"a∈[{{lo_a:.3f}},{{lo_a+sa:.3f}}] "
                f"b∈[{{lo_b:.3f}},{{lo_b+sb:.3f}}] "
                f"c∈[{{lo_c:.3f}},{{lo_c+sc:.3f}}]"
            )
        else:
            lo_a = lo_b = lo_c = 0.0
            sa = sb = sc = 1.0

        # ceil so the actual spacing is ≤ the user's target along every axis.
        na = max(2, int(np.ceil(sa * a_len / GRID_SPACING_A)))
        nb = max(2, int(np.ceil(sb * b_len / GRID_SPACING_A)))
        nc = max(2, int(np.ceil(sc * c_len / GRID_SPACING_A)))
        n_total = na * nb * nc

        # Cartesian origin + spanning vectors of the datagrid box (full cell
        # when no region is set). Stored in the NPZ so the volumetric exporters
        # and the GUI viewer can place the grid correctly.
        grid_origin = np.array([lo_a, lo_b, lo_c]) @ cell
        grid_cell   = np.array([sa * cell[0], sb * cell[1], sc * cell[2]])

        print(
            f"  Lattice |a|,|b|,|c| = {{a_len:.3f}}, {{b_len:.3f}}, {{c_len:.3f}} Å"
        )
        print(
            f"  Grid: {{na}} × {{nb}} × {{nc}} = {{n_total}} points  "
            f"(actual spacings: {{sa*a_len/na:.3f}}, {{sb*b_len/nb:.3f}}, "
            f"{{sc*c_len/nc:.3f}} Å)"
        )

        energies  = np.full((na, nb, nc), BLOCKED_ENERGY, dtype=float)
        blocked   = np.ones((na, nb, nc), dtype=bool)
        min_d_arr = np.zeros((na, nb, nc), dtype=float)

        base_e = None
        if COMPUTE_BASELINE:
            try:
                base_e = _evaluate_energy(host.copy())
                print(f"  Baseline E(host) = {{base_e:.6f}} eV")
            except Exception as exc:
                print(f"  ⚠ Baseline energy failed: {{exc}}")
                base_e = None

        rows = []
        n_done = 0
        n_skip = 0
        n_relax_conv = 0   # relaxations that reached fmax
        n_relax_unconv = 0  # relaxations that hit RELAX_MAX_STEPS first
        probe_index = len(host)
        # Rolling window of recent COMPUTED-step durations for ETA.
        # Blocked points take milliseconds, so estimating from energy-eval
        # times only gives a more honest estimate of the remaining work.
        compute_times = deque(maxlen=20)

        for i in range(na):
            for j in range(nb):
                for k in range(nc):
                    step_idx = i * nb * nc + j * nc + k + 1
                    remaining = n_total - step_idx
                    frac = np.array([
                        lo_a + (i / na) * sa,
                        lo_b + (j / nb) * sb,
                        lo_c + (k / nc) * sc,
                    ])
                    cart = frac @ cell

                    test = host.copy()
                    test.append(Atom(INSERT_ELEMENT, position=cart))

                    far_atom_idx = None
                    if probe_index > 0:
                        dists = test.get_distances(
                            probe_index,
                            list(range(probe_index)),
                            mic=True,
                        )
                        min_d = float(dists.min())
                        # Host atom farthest from the probe (PBC min-image) —
                        # pinned together with the probe during relaxation to
                        # suppress rigid-body drift of the whole cell.
                        far_atom_idx = int(dists.argmax())
                    else:
                        min_d = float("inf")
                    min_d_arr[i, j, k] = min_d

                    step_t0 = time.time()
                    status_str = ""
                    energy_str = ""
                    if min_d < MIN_DISTANCE_A:
                        n_skip += 1
                        rows.append((
                            i, j, k,
                            float(frac[0]), float(frac[1]), float(frac[2]),
                            float(cart[0]), float(cart[1]), float(cart[2]),
                            BLOCKED_ENERGY, True, min_d,
                        ))
                        status_str = "BLOCKED"
                        energy_str = f"d_min={{min_d:.3f}} Å"
                    else:
                        try:
                            _fixed = None
                            if RELAX_EACH_POINT:
                                _fixed = [probe_index]
                                if far_atom_idx is not None:
                                    _fixed.append(far_atom_idx)
                            e = _evaluate_energy(test, fixed_indices=_fixed)
                            energies[i, j, k] = e
                            blocked[i, j, k]  = False
                            n_done += 1
                            rows.append((
                                i, j, k,
                                float(frac[0]), float(frac[1]), float(frac[2]),
                                float(cart[0]), float(cart[1]), float(cart[2]),
                                e, False, min_d,
                            ))
                            status_str = "OK     "
                            energy_str = f"E={{e:.4f}} eV"
                            # Append relaxation outcome (converged + step count)
                            # when geometry optimization was performed here.
                            if RELAX_EACH_POINT and "steps" in relax_status:
                                _nstep = relax_status["steps"]
                                _conv  = relax_status.get("converged")
                                if _conv is True:
                                    n_relax_conv += 1
                                    energy_str += f"  [conv @ step {{_nstep}}]"
                                elif _conv is False:
                                    n_relax_unconv += 1
                                    energy_str += (
                                        f"  [NOT conv, {{_nstep}}/"
                                        f"{{RELAX_MAX_STEPS}} steps]"
                                    )
                                else:
                                    energy_str += f"  [relaxed {{_nstep}} steps]"
                            compute_times.append(time.time() - step_t0)
                        except Exception as exc:
                            n_skip += 1
                            rows.append((
                                i, j, k,
                                float(frac[0]), float(frac[1]), float(frac[2]),
                                float(cart[0]), float(cart[1]), float(cart[2]),
                                BLOCKED_ENERGY, True, min_d,
                            ))
                            status_str = "FAILED "
                            energy_str = f"{{exc}}"
                    step_dt = time.time() - step_t0

                    if compute_times:
                        avg_step = sum(compute_times) / len(compute_times)
                        eta_sec  = avg_step * remaining
                        eta_str  = f"ETA {{_fmt_duration(eta_sec)}}"
                    else:
                        eta_str = "ETA --"

                    # Throttle console output to every PRINT_EVERY-th grid
                    # point. The first and last points are always shown, and
                    # FAILED points are never suppressed so errors stay visible.
                    _do_print = (
                        PRINT_EVERY <= 1
                        or step_idx == 1
                        or step_idx == n_total
                        or step_idx % PRINT_EVERY == 0
                        or status_str.startswith("FAILED")
                    )
                    if _do_print:
                        print(
                            f"  {{step_idx:>6d}}/{{n_total}} "
                            f"({{i:>3d}},{{j:>3d}},{{k:>3d}})  "
                            f"frac=({{frac[0]:.3f}},{{frac[1]:.3f}},{{frac[2]:.3f}})  "
                            f"cart=({{cart[0]:6.2f}},{{cart[1]:6.2f}},{{cart[2]:6.2f}}) Å  "
                            f"{{status_str}}  {{energy_str:<28s}}  "
                            f"step {{_fmt_duration(step_dt)}}  "
                            f"{{eta_str}}  "
                            f"({{n_done}} ok / {{n_skip}} skip / {{remaining}} left)",
                            flush=True,
                        )

        elapsed = time.time() - t0
        base_name = Path(fname).stem
        out_dir = OUTPUT_ROOT / base_name
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            out_dir / "energy_grid.npz",
            energies=energies,
            blocked=blocked,
            min_neighbor_distance=min_d_arr,
            na=na, nb=nb, nc=nc,
            cell=cell,
            insert_element=INSERT_ELEMENT,
            grid_spacing_A=GRID_SPACING_A,
            min_distance_A=MIN_DISTANCE_A,
            blocked_energy=BLOCKED_ENERGY,
            base_energy_eV=(base_e if base_e is not None else np.nan),
            relaxed=RELAX_EACH_POINT,
            relax_optimizer=RELAX_OPTIMIZER,
            relax_fmax=RELAX_FMAX,
            relax_max_steps=RELAX_MAX_STEPS,
            region_enabled=REGION_ENABLED,
            region_center=np.array(REGION_CENTER, dtype=float),
            region_center_mode=REGION_CENTER_MODE,
            region_length=REGION_LENGTH,
            region_length_mode=REGION_LENGTH_MODE,
            grid_origin=grid_origin,
            grid_cell=grid_cell,
            frac_lo=np.array([lo_a, lo_b, lo_c], dtype=float),
            frac_span=np.array([sa, sb, sc], dtype=float),
        )

        df = pd.DataFrame(rows, columns=[
            "i", "j", "k",
            "frac_a", "frac_b", "frac_c",
            "cart_x_A", "cart_y_A", "cart_z_A",
            "energy_eV", "blocked", "min_neighbor_dist_A",
        ])
        df.to_csv(out_dir / "energy_grid.csv", index=False)

        if SAVE_XSF:
            try:
                _write_xsf(
                    out_dir / "energy_grid.xsf",
                    host,
                    energies,
                    blocked,
                    origin=grid_origin,
                    cell=cell,
                    grid_cell=grid_cell,
                )
            except Exception as exc:
                print(f"  ⚠ XSF write failed: {{exc}}")

        # ── Save host + probe at the lowest-energy site ──────────────
        if SAVE_MIN_STRUCT and (~blocked).any():
            try:
                from ase.io import write as _ase_write
                _valid_mask = ~blocked
                _e_min_val  = float(energies[_valid_mask].min())
                # All grid points within DEGEN_TOL_EV of the global minimum
                # are treated as physically/numerically equivalent.
                _degen_mask = _valid_mask & (
                    energies <= _e_min_val + DEGEN_TOL_EV
                )
                _candidates = np.argwhere(_degen_mask)
                _rng = np.random.default_rng()
                _pick_idx = _rng.integers(0, len(_candidates))
                _ci, _cj, _ck = (int(x) for x in _candidates[_pick_idx])
                _frac_min = np.array([
                    lo_a + (_ci / na) * sa,
                    lo_b + (_cj / nb) * sb,
                    lo_c + (_ck / nc) * sc,
                ])
                _cart_min = _frac_min @ cell

                _min_atoms = host.copy()
                _min_atoms.append(Atom(INSERT_ELEMENT, position=_cart_min))

                # If per-point relaxation was used, relax the saved structure
                # too (same constraints) so its geometry matches the recorded
                # relaxed energy rather than the raw grid placement.
                if RELAX_EACH_POINT and len(host) > 0:
                    _pidx = len(host)
                    _md = _min_atoms.get_distances(
                        _pidx, list(range(_pidx)), mic=True
                    )
                    _e_min_val = _evaluate_energy(
                        _min_atoms,
                        fixed_indices=[_pidx, int(_md.argmax())],
                    )
                    _min_atoms.set_constraint()  # drop constraint before write

                _cif_path  = out_dir / "min_energy_structure.cif"
                _vasp_path = out_dir / "min_energy_structure.vasp"
                _ase_write(_cif_path, _min_atoms)
                _ase_write(_vasp_path, _min_atoms,
                           format="vasp", direct=True, sort=True)

                _txt_path = out_dir / "min_energy_site.txt"
                with open(_txt_path, "w") as _fh:
                    _fh.write(
                        f"# Lowest-energy interstitial site for "
                        f"{{INSERT_ELEMENT}}\\n"
                    )
                    _fh.write(f"# Energy (eV):           {{_e_min_val:.6f}}\\n")
                    _fh.write(
                        f"# Degeneracy tolerance:   "
                        f"{{DEGEN_TOL_EV:g}} eV\\n"
                    )
                    _fh.write(
                        f"# Degenerate grid sites: {{len(_candidates)}}\\n"
                    )
                    _fh.write(
                        f"# Chosen grid index:     "
                        f"({{_ci}}, {{_cj}}, {{_ck}})\\n"
                    )
                    _fh.write(
                        f"# Fractional coords:     "
                        f"{{_frac_min[0]:.8f}} "
                        f"{{_frac_min[1]:.8f}} "
                        f"{{_frac_min[2]:.8f}}\\n"
                    )
                    _fh.write(
                        f"# Cartesian coords (Å):  "
                        f"{{_cart_min[0]:.8f}} "
                        f"{{_cart_min[1]:.8f}} "
                        f"{{_cart_min[2]:.8f}}\\n"
                    )

                _degen_str = (
                    f"{{len(_candidates)}} equivalent site"
                    + ("s" if len(_candidates) != 1 else "")
                )
                print(
                    f"  💾 min_energy_structure.cif + .vasp  "
                    f"({{INSERT_ELEMENT}} at "
                    f"frac=({{_frac_min[0]:.4f}}, "
                    f"{{_frac_min[1]:.4f}}, {{_frac_min[2]:.4f}}), "
                    f"E={{_e_min_val:.4f}} eV, {{_degen_str}})"
                )
            except Exception as _min_exc:
                print(f"  ⚠ Could not save min-energy structure: "
                      f"{{_min_exc}}")

        if RELAX_EACH_POINT and (n_relax_conv + n_relax_unconv) > 0:
            print(
                f"  🧲 Relaxation summary: {{n_relax_conv}} converged, "
                f"{{n_relax_unconv}} hit the {{RELAX_MAX_STEPS}}-step cap "
                f"(fmax={{RELAX_FMAX}} eV/Å)"
            )

        valid = energies[~blocked]
        if valid.size > 0:
            e_min = float(valid.min())
            e_max = float(valid.max())
            print(
                f"  ✅ {{n_done}}/{{n_total}} computed, {{n_skip}} blocked  "
                f"E ∈ [{{e_min:.4f}}, {{e_max:.4f}}] eV  "
                f"({{elapsed:.1f}} s)  → {{out_dir}}/"
            )
        else:
            print(
                f"  ⚠ All {{n_total}} grid points blocked — try a smaller "
                f"MIN_DISTANCE_A or a different insert element. "
                f"({{elapsed:.1f}} s)"
            )

    total_time = time.time() - start_time
    print(f"\\n🎉 Energy grid scan finished — total wall time {{total_time/60:.1f}} min")


if __name__ == "__main__":
    main()
'''
