"""
Builds the standalone Python script for the "Finite-T Elastic Properties"
calculation type.

The generated script implements the explicit stress-strain (direct) method for
isothermal elastic constants at finite temperature:

    0 K relaxation  ->  NPT pre-equilibration at (T, P)  ->  reference cell
    ->  affine strain +/- delta on every required Voigt component
    ->  NVT run per strain state, time-averaged thermodynamic stress
    ->  C_ij = d<sigma_i>/d eps_j  ->  VRH moduli, stability, Debye temperature

The production part can also run adaptively: all strain states of one
temperature are then advanced together in segments, C_ij is re-fitted after
every segment and the runs stop as soon as the constants have stopped moving
by more than the requested tolerance.

Everything below only assembles source text; nothing is executed here.
"""

import pprint
import textwrap

from helpers.calculator_setup_code import build_calculator_code


def _base_imports(thread_count):
    return f"""
import os
import sys
import glob
import json
import time
import hashlib
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ase import units
from ase.io import read, write
from ase.build import make_supercell
from ase.optimize import LBFGS, BFGS, FIRE
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen

try:
    from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
    INHOMO_NPT_AVAILABLE = True
except ImportError:
    INHOMO_NPT_AVAILABLE = False

try:
    from ase.md.nose_hoover_chain import NoseHooverChainNVT
    NVT_NOSE_HOOVER_AVAILABLE = True
except ImportError:
    NVT_NOSE_HOOVER_AVAILABLE = False

try:
    # ASE >= 3.27 moved (and renamed) the Melchionna/Parrinello-Rahman NPT.
    from ase.md.melchionna import MelchionnaNPT as FullCellNPT
    FULL_NPT_AVAILABLE = True
except ImportError:
    try:
        from ase.md.npt import NPT as FullCellNPT
        FULL_NPT_AVAILABLE = True
    except ImportError:
        FULL_NPT_AVAILABLE = False

try:
    from ase.constraints import FixCom
    FIXCOM_AVAILABLE = True
except ImportError:
    FIXCOM_AVAILABLE = False

try:
    from ase.filters import FrechetCellFilter as CellFilter
except ImportError:
    try:
        from ase.constraints import ExpCellFilter as CellFilter
    except ImportError:
        CellFilter = None

try:
    from nequix.calculator import NequixCalculator
except ImportError:
    pass
try:
    from nequip.model.saved_models.load_utils import load_saved_model as _nequip_load_saved_model
    from nequip.integrations.ase import NequIPCalculator
    from nequip.integrations.utils import basic_transforms, handle_chemical_species_map
except ImportError:
    pass
try:
    from deepmd.calculator import DP
except ImportError:
    pass

os.environ['OMP_NUM_THREADS'] = '{thread_count}'
torch.set_num_threads({thread_count})
"""


# ---------------------------------------------------------------------------
# Everything below is emitted verbatim into the generated script. It is a plain
# (non-f) string on purpose: the generated code is full of braces and needs no
# escaping this way. All settings are read at runtime from `finite_t_params`.
# ---------------------------------------------------------------------------
_SCRIPT_BODY = r'''

EV_TO_GPA = 160.21766208
VOIGT_LABELS = ["xx", "yy", "zz", "yz", "xz", "xy"]
RESULT_DIR = "elastic_md_results"
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoint")

# Voigt components that have to be strained for each symmetry. The remaining
# constants follow from the symmetry relations (see assemble_tensor).
SYMMETRY_STRAIN_COMPONENTS = {
    "triclinic": [0, 1, 2, 3, 4, 5],
    "cubic": [0, 3],
    "hexagonal": [0, 2, 3],
}


# ---------------------------------------------------------------------------
# Checkpointing. Every finished piece of work - the 0 K relaxation, the static
# tensor, each temperature's NPT reference cell and each individual strained NVT
# run - is written out as soon as it completes. Re-running the script picks up
# where it stopped instead of repeating any MD that is already done.
# ---------------------------------------------------------------------------

# Settings that change the numbers. Anything not listed here (log interval,
# trajectory writing, block count, ...) can be changed between runs without
# invalidating what has already been computed.
# Bumped whenever the dynamics themselves change, so that results produced by an
# older version of this script are never silently mixed with new ones.
METHOD_VERSION = 2

SIGNATURE_KEYS = [
    "timestep", "supercell", "seed", "pressure_GPa",
    "pre_optimize", "pre_opt_optimizer", "pre_opt_fmax", "pre_opt_steps",
    "pre_opt_relax_cell",
    "use_npt_equilibration", "npt_barostat", "npt_equilibration_steps",
    "npt_production_steps", "npt_ttime", "npt_ptime", "npt_bulk_modulus",
    "reference_cell",
    "nvt_thermostat", "friction", "thermostat_taut",
    "nvt_equilibration_steps", "nvt_production_steps", "sample_interval",
    "strain_magnitude", "use_multi_strain", "symmetry", "strain_components",
    "include_kinetic_stress",
]


def settings_signature(atoms):
    """Fingerprint of the settings + input structure that affect the results."""
    payload = {k: finite_t_params.get(k) for k in SIGNATURE_KEYS}
    payload["method_version"] = METHOD_VERSION
    payload["numbers"] = atoms.get_atomic_numbers().tolist()
    payload["cell"] = np.round(np.array(atoms.get_cell()[:], dtype=float), 6).tolist()
    payload["positions"] = np.round(atoms.get_positions(), 6).tolist()
    blob = json.dumps(payload, sort_keys=True, default=float)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def checkpoint_path(basename):
    return os.path.join(RESULT_DIR, "%s_checkpoint.json" % basename)


def save_atoms_npz(path, atoms):
    """Store a structure at full float64 precision.

    The human-readable .xyz files are written for the user, but reloading one
    truncates the positions to ~1e-8 A, which makes a resumed trajectory drift
    away from an uninterrupted one. The checkpoint uses this instead so that
    resuming reproduces the uninterrupted result bit for bit.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    payload = {
        "numbers": atoms.get_atomic_numbers(),
        "positions": atoms.get_positions(),
        "cell": np.array(atoms.get_cell()[:], dtype=float),
        "pbc": np.array(atoms.get_pbc(), dtype=bool),
    }
    # Velocities matter only for a snapshot taken in the middle of an MD run
    # (adaptive production length); everywhere else they are simply absent.
    if atoms.has("momenta"):
        payload["momenta"] = atoms.get_momenta()
    np.savez(path, **payload)


def load_atoms_npz(path):
    from ase import Atoms
    with np.load(path) as data:
        atoms = Atoms(numbers=data["numbers"], positions=data["positions"],
                      cell=data["cell"], pbc=data["pbc"])
        if "momenta" in data.files:
            atoms.set_momenta(data["momenta"])
        return atoms


def temperature_key(temperature_K):
    return "%.2f" % float(temperature_K)


def temperature_tag(temperature_K):
    """Filename-safe temperature label ("300", "350p5")."""
    return ("%g" % float(temperature_K)).replace(".", "p").replace("-", "m")


def temperature_dir(temperature_K, create=True):
    """Output folder for one temperature: elastic_md_results/T_300K/ ..."""
    path = os.path.join(RESULT_DIR, "T_%sK" % temperature_tag(temperature_K))
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def trajectory_dir(temperature_K, create=True):
    """Where the per-strain .xyz trajectories of one temperature go."""
    path = os.path.join(temperature_dir(temperature_K, create=create),
                        "trajectories")
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def state_key(j, delta):
    return "%d_%+.6f" % (int(j), float(delta))


def empty_checkpoint(signature):
    return {"signature": signature, "relaxed": None, "static_tensor_GPa": None,
            "temperatures": {}}


def load_checkpoint(basename, signature):
    """Previous progress for exactly these settings, or a blank slate."""
    path = checkpoint_path(basename)
    if not finite_t_params.get("resume_from_checkpoint", True):
        if os.path.exists(path):
            print("  Resume is switched off - the existing checkpoint will be "
                  "overwritten and everything recomputed.")
        return empty_checkpoint(signature)
    if not os.path.exists(path):
        return empty_checkpoint(signature)
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception as exc:
        print("  \u26a0\ufe0f  WARNING: could not read the checkpoint (%s) - starting fresh."
              % exc)
        return empty_checkpoint(signature)
    if data.get("signature") != signature:
        print("  The settings or the input structure changed since the previous "
              "run - the old checkpoint does not apply, starting fresh.")
        return empty_checkpoint(signature)
    n_prod = int(finite_t_params.get("nvt_production_steps", 0))
    n_done = 0
    n_partial = 0
    for entry in data.get("temperatures", {}).values():
        for record in entry.get("states", {}).values():
            if (bool(record.get("complete", True))
                    and int(record.get("production_steps", n_prod)) >= n_prod):
                n_done += 1
            elif record.get("snapshot_file"):
                n_partial += 1
    n_npt = sum(1 for t in data.get("temperatures", {}).values() if t.get("ref_cell"))
    print("  Found a checkpoint from a previous run: %d NPT reference cell(s) "
          "and %d finished strained run(s)%s."
          % (n_npt, n_done,
             ", plus %d that stopped part-way and will be continued" % n_partial
             if n_partial else ""))
    data.setdefault("temperatures", {})
    return data


def save_checkpoint(basename, checkpoint):
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = checkpoint_path(basename)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(checkpoint, fh, indent=1, default=float)
        os.replace(tmp, path)      # atomic: a crash mid-write cannot corrupt it
    except Exception as exc:
        print("  \u26a0\ufe0f  WARNING: could not write the checkpoint: %s" % exc)


def store_state(checkpoint, basename, temperature_K, state, flush=True):
    """Persist one strained run (scalars in JSON, samples in an .npz).

    A run is normally stored once, when it is finished. With the adaptive
    production length it is stored again after every segment, together with a
    snapshot of positions and velocities, so that an interrupted calculation
    can pick a half-finished production run up where it stopped.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tkey = temperature_key(temperature_K)
    skey = state_key(state["voigt_index"], state["delta"])
    stem = "%s_T%sK_state_%s" % (basename, tkey,
                                 skey.replace("+", "p").replace("-", "m"))
    samples_name = "%s.npz" % stem
    samples_path = os.path.join(CHECKPOINT_DIR, samples_name)
    try:
        np.savez_compressed(
            samples_path,
            times_ps=np.asarray(state["times_ps"], dtype=float),
            samples_GPa=np.asarray(state["samples_GPa"], dtype=float),
            temperatures_K=np.asarray(state.get("temperatures_K", []), dtype=float))
    except Exception as exc:
        print("  \u26a0\ufe0f  WARNING: could not save the stress samples: %s" % exc)
        samples_name = None

    snapshot_name = None
    snapshot = state.get("snapshot")
    if snapshot is not None:
        snapshot_name = "%s_snapshot.npz" % stem
        try:
            save_atoms_npz(os.path.join(CHECKPOINT_DIR, snapshot_name), snapshot)
        except Exception as exc:
            print("  \u26a0\ufe0f  WARNING: could not save the run snapshot: %s" % exc)
            snapshot_name = None

    entry = checkpoint["temperatures"].setdefault(tkey, {"states": {}})
    entry.setdefault("states", {})[skey] = {
        "voigt_index": int(state["voigt_index"]),
        "delta": float(state["delta"]),
        "stress_mean_GPa": np.asarray(state["stress_mean_GPa"], dtype=float).tolist(),
        "stress_sem_GPa": np.asarray(state["stress_sem_GPa"], dtype=float).tolist(),
        "mean_temperature_K": float(state["mean_temperature_K"]),
        "n_samples": int(state["n_samples"]),
        "samples_file": samples_name,
        "production_steps": int(state.get("production_steps",
                                          finite_t_params.get("nvt_production_steps", 0))),
        "complete": bool(state.get("complete", True)),
        "snapshot_file": snapshot_name,
    }
    if flush:
        save_checkpoint(basename, checkpoint)


def restore_state(checkpoint, temperature_K, j, delta, allow_partial=False):
    """A strained run from the checkpoint, or None.

    Unless `allow_partial` is set, only a run that finished the full production
    length counts as done - a half-finished one left behind by an adaptive run
    that was interrupted has to be continued, not used as a result.
    """
    entry = checkpoint.get("temperatures", {}).get(temperature_key(temperature_K))
    if not entry:
        return None
    record = entry.get("states", {}).get(state_key(j, delta))
    if not record:
        return None

    n_prod = int(finite_t_params.get("nvt_production_steps", 0))
    # Records written before partial storage existed have neither key and were
    # always complete runs of the full production length.
    steps_done = int(record.get("production_steps", n_prod))
    complete = bool(record.get("complete", True))
    if not allow_partial and (not complete or steps_done < n_prod):
        return None

    times = np.zeros(0)
    samples = np.zeros((0, 6))
    sampled_T = np.zeros(0)
    name = record.get("samples_file")
    if name:
        path = os.path.join(CHECKPOINT_DIR, name)
        if os.path.exists(path):
            try:
                with np.load(path) as data:
                    times = np.asarray(data["times_ps"], dtype=float)
                    samples = np.asarray(data["samples_GPa"], dtype=float)
                    if "temperatures_K" in data.files:
                        sampled_T = np.asarray(data["temperatures_K"], dtype=float)
            except Exception as exc:
                print("      \u26a0\ufe0f  WARNING: stored stress samples unreadable (%s) - "
                      "the plots will skip this state." % exc)

    # Re-derive the averages from the stored samples so that changing the block
    # count between runs still takes effect; fall back to the stored scalars.
    mean_GPa = np.asarray(record["stress_mean_GPa"], dtype=float)
    sem_GPa = np.asarray(record["stress_sem_GPa"], dtype=float)
    if len(samples) > 1:
        mean_GPa, sem_GPa = block_stats(
            samples, int(finite_t_params.get("n_blocks", 5)))

    return {
        "voigt_index": int(record["voigt_index"]),
        "delta": float(record["delta"]),
        "stress_mean_GPa": mean_GPa,
        "stress_sem_GPa": sem_GPa,
        "mean_temperature_K": float(record["mean_temperature_K"]),
        "n_samples": int(record["n_samples"]),
        "samples_GPa": samples,
        "times_ps": times,
        "temperatures_K": sampled_T,
        "production_steps": steps_done,
        "complete": complete,
        "snapshot_file": record.get("snapshot_file"),
        "restored": True,
    }


def restore_npt(checkpoint, temperature_K, basename, calculator):
    """A finished NPT pre-equilibration from the checkpoint, or None."""
    entry = checkpoint.get("temperatures", {}).get(temperature_key(temperature_K))
    if not entry or not entry.get("ref_cell"):
        return None
    path = entry.get("reference_npz")
    if not path or not os.path.exists(path):
        return None
    try:
        hot = load_atoms_npz(path)
    except Exception as exc:
        print("  \u26a0\ufe0f  WARNING: could not read the stored NPT reference (%s)." % exc)
        return None
    hot.set_pbc(True)
    hot.calc = calculator
    ref_cell = np.array(entry["ref_cell"], dtype=float)
    # The snapshot was stored *after* being squeezed into the reference cell.
    # Re-scaling it by the very same cell is a no-op mathematically but not in
    # floating point, and that tiny nudge would make a resumed trajectory differ
    # from an uninterrupted one - so only rescale if the cell really disagrees.
    if not np.allclose(np.array(hot.get_cell()[:], dtype=float), ref_cell,
                       rtol=0.0, atol=1e-10):
        hot.set_cell(ref_cell, scale_atoms=True)
    return ref_cell, hot, entry.get("npt_info", {})


def strain_values():
    """Strain magnitudes scanned for every strained Voigt component."""
    delta = float(finite_t_params["strain_magnitude"])
    if finite_t_params.get("use_multi_strain", False):
        return [-delta, -delta / 2.0, delta / 2.0, delta]
    return [-delta, delta]


def voigt_strain_matrix(j, d):
    """Strain tensor for engineering Voigt strain `d` on component `j`."""
    e = np.zeros((3, 3))
    if j == 0:
        e[0, 0] = d
    elif j == 1:
        e[1, 1] = d
    elif j == 2:
        e[2, 2] = d
    elif j == 3:
        e[1, 2] = e[2, 1] = d / 2.0
    elif j == 4:
        e[0, 2] = e[2, 0] = d / 2.0
    elif j == 5:
        e[0, 1] = e[1, 0] = d / 2.0
    return e


def strained_cell(ref_cell, j, d):
    """Affinely deform `ref_cell` (rows = lattice vectors) by Voigt strain j, d."""
    F = np.eye(3) + voigt_strain_matrix(j, d)
    return np.dot(F, np.asarray(ref_cell, dtype=float).T).T


def thermo_stress_voigt(atoms):
    """Thermodynamic stress in eV/A^3: virial plus the kinetic (ideal-gas) term.

    The kinetic term -(1/V) sum_k m_k v_k (x) v_k is part of the definition of
    the isothermal elastic constants, so it is included unless switched off.
    """
    if not finite_t_params.get("include_kinetic_stress", True):
        return np.asarray(atoms.get_stress(voigt=True), dtype=float)
    try:
        return np.asarray(
            atoms.get_stress(voigt=True, include_ideal_gas=True), dtype=float)
    except TypeError:
        # Older ASE without the include_ideal_gas switch: add the term by hand.
        s = np.asarray(atoms.get_stress(voigt=True), dtype=float).copy()
        if atoms.has("momenta"):
            p = atoms.get_momenta()
            masses = atoms.get_masses()
            invvol = 1.0 / atoms.get_volume()
            comp = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
            for a in range(3):
                for b in range(a, 3):
                    s[comp[a, b]] -= (p[:, a] * p[:, b] / masses).sum() * invvol
        return s


def block_stats(samples, n_blocks):
    """Mean and block-averaged standard error of a (n_samples, n_cols) series."""
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    mean = arr.mean(axis=0)
    n = len(arr)
    nb = int(max(2, min(int(n_blocks), n)))
    blocks = [b for b in np.array_split(np.arange(n), nb) if len(b) > 0]
    if len(blocks) < 2:
        return mean, np.zeros_like(mean)
    block_means = np.array([arr[idx].mean(axis=0) for idx in blocks])
    sem = block_means.std(axis=0, ddof=1) / np.sqrt(len(block_means))
    return mean, sem


def weighted_slope(x, y, yerr):
    """Slope of y(x) with 1/sigma^2 weights.

    Returns (slope, slope_error, max_residual). With only two points this is the
    central difference and the error is propagated analytically.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    if len(x) < 2:
        return 0.0, 0.0, 0.0
    if len(x) == 2:
        dx = x[1] - x[0]
        slope = (y[1] - y[0]) / dx
        err = float(np.sqrt(yerr[0] ** 2 + yerr[1] ** 2) / abs(dx))
        return float(slope), err, 0.0

    if np.all(yerr > 0):
        w = 1.0 / yerr ** 2
    else:
        w = np.ones_like(x)
    S = w.sum()
    Sx = (w * x).sum()
    Sxx = (w * x * x).sum()
    Sy = (w * y).sum()
    Sxy = (w * x * y).sum()
    denom = S * Sxx - Sx * Sx
    if abs(denom) < 1e-30:
        return 0.0, 0.0, 0.0
    slope = (S * Sxy - Sx * Sy) / denom
    intercept = (Sxx * Sy - Sx * Sxy) / denom
    slope_err = float(np.sqrt(S / denom))
    residual = float(np.max(np.abs(y - (intercept + slope * x))))
    return float(slope), slope_err, residual


def assemble_tensor(columns, symmetry):
    """Build the full 6x6 C (GPa) and its uncertainty from the measured columns.

    `columns` maps a strained Voigt index j to {"slope": (6,), "err": (6,)},
    i.e. the six stress derivatives d sigma_i / d eps_j.
    """
    C = np.zeros((6, 6))
    E = np.zeros((6, 6))

    if symmetry == "cubic":
        s0, e0 = columns[0]["slope"], columns[0]["err"]
        s3, e3 = columns[3]["slope"], columns[3]["err"]
        C11, e11 = s0[0], e0[0]
        C12 = 0.5 * (s0[1] + s0[2])
        e12 = 0.5 * np.sqrt(e0[1] ** 2 + e0[2] ** 2)
        C44, e44 = s3[3], e3[3]
        for i in range(3):
            C[i, i], E[i, i] = C11, e11
            C[i + 3, i + 3], E[i + 3, i + 3] = C44, e44
            for k in range(3):
                if i != k:
                    C[i, k], E[i, k] = C12, e12

    elif symmetry == "hexagonal":
        # Unique (c) axis assumed along z.
        s0, e0 = columns[0]["slope"], columns[0]["err"]
        s2, e2 = columns[2]["slope"], columns[2]["err"]
        s3, e3 = columns[3]["slope"], columns[3]["err"]
        C11, e11 = s0[0], e0[0]
        C12, e12 = s0[1], e0[1]
        C13 = 0.5 * (s0[2] + 0.5 * (s2[0] + s2[1]))
        e13 = 0.5 * np.sqrt(e0[2] ** 2 + 0.25 * (e2[0] ** 2 + e2[1] ** 2))
        C33, e33 = s2[2], e2[2]
        C44, e44 = s3[3], e3[3]
        C66 = 0.5 * (C11 - C12)
        e66 = 0.5 * np.sqrt(e11 ** 2 + e12 ** 2)
        C[0, 0] = C[1, 1] = C11
        E[0, 0] = E[1, 1] = e11
        C[0, 1] = C[1, 0] = C12
        E[0, 1] = E[1, 0] = e12
        C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13
        E[0, 2] = E[2, 0] = E[1, 2] = E[2, 1] = e13
        C[2, 2], E[2, 2] = C33, e33
        C[3, 3] = C[4, 4] = C44
        E[3, 3] = E[4, 4] = e44
        C[5, 5], E[5, 5] = C66, e66

    else:  # triclinic - measured in full, only symmetrised
        raw = np.zeros((6, 6))
        raw_e = np.zeros((6, 6))
        for j, col in columns.items():
            raw[:, j] = col["slope"]
            raw_e[:, j] = col["err"]
        C = 0.5 * (raw + raw.T)
        E = 0.5 * np.sqrt(raw_e ** 2 + raw_e.T ** 2)

    return C, E


def check_mechanical_stability(C):
    """Born stability criteria on the (symmetrised) elastic tensor."""
    criteria = {}
    try:
        eigenvals = np.linalg.eigvalsh(C)
        criteria["eigenvalues_GPa"] = [float(v) for v in eigenvals]
        criteria["positive_definite"] = bool(np.all(eigenvals > 0))
        for i in range(6):
            criteria["C%d%d_positive" % (i + 1, i + 1)] = bool(C[i, i] > 0)
        criteria["det_positive"] = bool(np.linalg.det(C) > 0)
        if abs(C[0, 0] - C[1, 1]) < 1.0 and abs(C[1, 1] - C[2, 2]) < 1.0:
            criteria["cubic_C11_gt_absC12"] = bool(C[0, 0] > abs(C[0, 1]))
            criteria["cubic_bulk_positive"] = bool((C[0, 0] + 2 * C[0, 1]) > 0)
        criteria["mechanically_stable"] = bool(
            criteria["positive_definite"] and criteria["det_positive"])
    except Exception as exc:
        criteria["error"] = str(exc)
        criteria["mechanically_stable"] = False
    return criteria


def vrh_moduli(C):
    """Voigt-Reuss-Hill averages plus derived quantities. C in GPa."""
    K_voigt = (C[0, 0] + C[1, 1] + C[2, 2]
               + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    G_voigt = (C[0, 0] + C[1, 1] + C[2, 2]
               - C[0, 1] - C[0, 2] - C[1, 2]
               + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.0

    K_reuss = G_reuss = K_hill = G_hill = None
    S = None
    try:
        S = np.linalg.inv(C)
        K_reuss = 1.0 / (S[0, 0] + S[1, 1] + S[2, 2]
                         + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
        G_reuss = 15.0 / (4 * (S[0, 0] + S[1, 1] + S[2, 2])
                          - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
                          + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
        K_hill = 0.5 * (K_voigt + K_reuss)
        G_hill = 0.5 * (G_voigt + G_reuss)
    except np.linalg.LinAlgError:
        pass

    K = K_hill if K_hill is not None else K_voigt
    G = G_hill if G_hill is not None else G_voigt
    if abs(3 * K + G) < 1e-12:
        E = float("nan")
        nu = float("nan")
    else:
        E = (9 * K * G) / (3 * K + G)
        nu = (3 * K - 2 * G) / (2 * (3 * K + G))
    return {
        "compliance": S,
        "bulk_modulus": {"voigt": K_voigt, "reuss": K_reuss, "hill": K_hill},
        "shear_modulus": {"voigt": G_voigt, "reuss": G_reuss, "hill": G_hill},
        "K": K, "G": G, "youngs_modulus": E, "poisson_ratio": nu,
    }


def sound_velocities_and_debye(K, G, density_kg_m3, n_atoms, total_mass_kg):
    """(v_longitudinal, v_transverse, v_average, Debye temperature) from K and G."""
    if not (K + 4 * G / 3 > 0 and G > 0):
        nan = float("nan")
        return nan, nan, nan, nan
    v_l = np.sqrt((K + 4 * G / 3) * 1e9 / density_kg_m3)
    v_t = np.sqrt(G * 1e9 / density_kg_m3)
    v_avg = ((1 / v_l ** 3 + 2 / v_t ** 3) / 3) ** (-1.0 / 3.0)
    h = 6.626e-34
    kB = 1.381e-23
    theta_D = (h / kB) * v_avg * (
        3 * n_atoms * density_kg_m3 / (4 * np.pi * total_mass_kg)) ** (1.0 / 3.0)
    return v_l, v_t, v_avg, theta_D


def analyze_tensor(C, C_err, atoms, n_mc=400, rng=None):
    """Moduli, sound velocities, Debye temperature and Monte-Carlo error bars."""
    res = vrh_moduli(C)
    K, G = res["K"], res["G"]

    volume = atoms.get_volume()
    n_atoms = len(atoms)
    total_mass_amu = float(np.sum(atoms.get_masses()))
    density = (total_mass_amu * 1.66053906660) / volume          # g/cm^3
    density_kg_m3 = density * 1000.0
    total_mass_kg = total_mass_amu * 1.66054e-27

    v_l, v_t, v_avg, theta_D = sound_velocities_and_debye(
        K, G, density_kg_m3, n_atoms, total_mass_kg)

    # Propagate the C_ij errors onto the moduli by resampling the tensor. The
    # Reuss average diverges for a near-singular sample, so unstable draws are
    # rejected and the spread is read off the 16/84 percentiles rather than as a
    # standard deviation, which a single outlier would dominate.
    errors = {}
    if C_err is not None and np.any(C_err > 0) and n_mc > 0:
        if rng is None:
            rng = np.random.RandomState(0)
        samples = {"K": [], "G": [], "E": [], "nu": [], "theta_D": [], "v_avg": []}
        for _ in range(int(n_mc)):
            noise = rng.normal(0.0, 1.0, size=(6, 6)) * C_err
            noise = 0.5 * (noise + noise.T)
            C_sample = C + noise
            try:
                if np.min(np.linalg.eigvalsh(C_sample)) <= 0.0:
                    continue
                m = vrh_moduli(C_sample)
            except Exception:
                continue
            samples["K"].append(m["K"])
            samples["G"].append(m["G"])
            samples["E"].append(m["youngs_modulus"])
            samples["nu"].append(m["poisson_ratio"])
            _, _, s_v_avg, s_theta = sound_velocities_and_debye(
                m["K"], m["G"], density_kg_m3, n_atoms, total_mass_kg)
            samples["theta_D"].append(s_theta)
            samples["v_avg"].append(s_v_avg)
        for key, vals in samples.items():
            vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
            if len(vals) > 8:
                lo, hi = np.percentile(vals, [16.0, 84.0])
                errors[key] = float(0.5 * (hi - lo))
            else:
                errors[key] = 0.0
        errors["n_valid_samples"] = int(len(samples["K"]))

    return {
        "elastic_tensor_GPa": C.tolist(),
        "elastic_tensor_error_GPa": (C_err.tolist() if C_err is not None else None),
        "compliance_matrix": (res["compliance"].tolist()
                              if res["compliance"] is not None else None),
        "bulk_modulus_GPa": res["bulk_modulus"],
        "shear_modulus_GPa": res["shear_modulus"],
        "youngs_modulus_GPa": res["youngs_modulus"],
        "poisson_ratio": res["poisson_ratio"],
        "modulus_errors_GPa": errors,
        "wave_velocities_ms": {"longitudinal": v_l, "transverse": v_t,
                               "average": v_avg},
        "debye_temperature_K": theta_D,
        "debye_temperature_error_K": errors.get("theta_D", 0.0),
        "density_g_cm3": density,
        "volume_A3": volume,
        "volume_per_atom_A3": volume / n_atoms,
        "n_atoms": n_atoms,
        "mechanical_stability": check_mechanical_stability(C),
    }


def print_tensor(C, C_err=None, title="Elastic tensor C_ij (GPa)"):
    print("  " + title)
    for i in range(6):
        if C_err is not None:
            cells = " ".join("%9.1f+-%-6.1f" % (C[i, k], C_err[i, k])
                             for k in range(6))
        else:
            cells = " ".join("%9.1f" % C[i, k] for k in range(6))
        print("    " + cells)


class JobClock:
    """Whole-job progress: every MD step planned across all strains and all T.

    The per-run ETA only covers the run in flight; this one answers "how long
    until the entire calculation is done", including the strain states and
    temperatures that have not been started yet.
    """

    def __init__(self, total_steps, total_runs):
        self.total_steps = max(1, int(total_steps))
        self.total_runs = max(1, int(total_runs))
        self.done_steps = 0
        self.done_runs = 0
        self.t0 = None

    def start(self):
        if self.t0 is None:
            self.t0 = time.perf_counter()

    def finish_run(self, steps):
        self.done_steps += int(steps)
        self.done_runs += 1

    def add_steps(self, steps):
        """Count steps of a run that is not finished yet (interleaved mode)."""
        self.done_steps += int(steps)

    def complete_run(self):
        self.done_runs += 1

    def drop_steps(self, steps):
        """Remove steps from the plan that will never run.

        The adaptive production length stops the runs as soon as C_ij is
        converged, so the steps that were budgeted for the rest of the
        production must leave the total or the ETA would never reach zero.
        """
        self.total_steps = max(self.done_steps + 1,
                               self.total_steps - int(max(0, steps)))

    def elapsed(self):
        return 0.0 if self.t0 is None else time.perf_counter() - self.t0

    def remaining_seconds(self, steps_in_flight=0):
        done = self.done_steps + int(steps_in_flight)
        elapsed = self.elapsed()
        if done <= 0 or elapsed <= 0:
            return float("nan")
        return elapsed / done * max(0, self.total_steps - done)

    def fraction(self, steps_in_flight=0):
        return min(1.0, (self.done_steps + int(steps_in_flight)) / self.total_steps)

    def summary(self, steps_in_flight=0):
        done = self.done_steps + int(steps_in_flight)
        return ("\u23f3 job %5.1f%% (%d/%d MD steps, run %d/%d), ~%s left in total"
                % (100.0 * self.fraction(steps_in_flight), done,
                   self.total_steps, min(self.done_runs + 1, self.total_runs),
                   self.total_runs, format_time(self.remaining_seconds(steps_in_flight))))


class MDProgress:
    """Console progress line for one MD run: run-local ETA plus the job ETA."""

    def __init__(self, atoms, dyn, total_steps, label, interval, job=None):
        self.atoms = atoms
        self.dyn = dyn
        self.total_steps = max(1, int(total_steps))
        self.label = label
        self.interval = max(1, int(interval))
        self.job = job
        self.start_step = int(dyn.get_number_of_steps())
        self.steps_before = 0      # steps of this run already counted elsewhere
        self.t0 = time.perf_counter()
        self.elapsed_before = 0.0  # wall time of this run's earlier segments

    def pause(self):
        """Stop this run's clock while the other strain states advance."""
        if self.t0 is not None:
            self.elapsed_before += time.perf_counter() - self.t0
            self.t0 = None

    def resume(self):
        self.t0 = time.perf_counter()

    def __call__(self):
        done = int(self.dyn.get_number_of_steps()) - self.start_step
        if done <= 0:
            return
        elapsed = self.elapsed_before + (
            0.0 if self.t0 is None else time.perf_counter() - self.t0)
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total_steps - done) / rate if rate > 0 else 0.0
        temperature = self.atoms.get_temperature()
        epot = self.atoms.get_potential_energy()
        stress = thermo_stress_voigt(self.atoms)
        pressure = -np.mean(stress[:3]) * EV_TO_GPA
        job_note = ""
        if self.job is not None:
            job_note = "  |  %s" % self.job.summary(self.steps_before + done)
        print("    %s step %6d/%-6d  T = %7.1f K  Epot = %14.4f eV  "
              "P = %8.3f GPa  (%.1f steps/s, ~%s)%s"
              % (self.label, done, self.total_steps, temperature, epot,
                 pressure, rate, format_time(remaining), job_note), flush=True)


class StressCollector:
    """Samples the thermodynamic stress (and T) during the production part."""

    def __init__(self, atoms, dyn, timestep_fs):
        self.atoms = atoms
        self.dyn = dyn
        self.timestep_fs = timestep_fs
        self.stresses = []
        self.temperatures = []
        self.times_ps = []

    def __call__(self):
        self.stresses.append(thermo_stress_voigt(self.atoms))
        self.temperatures.append(self.atoms.get_temperature())
        self.times_ps.append(
            int(self.dyn.get_number_of_steps()) * self.timestep_fs / 1000.0)


class CellCollector:
    """Samples the cell during the NPT production part."""

    def __init__(self, atoms):
        self.atoms = atoms
        self.cells = []
        self.volumes = []
        self.temperatures = []
        self.pressures = []

    def __call__(self):
        self.cells.append(np.array(self.atoms.get_cell()[:], dtype=float))
        self.volumes.append(self.atoms.get_volume())
        self.temperatures.append(self.atoms.get_temperature())
        stress = thermo_stress_voigt(self.atoms)
        self.pressures.append(-np.mean(stress[:3]) * EV_TO_GPA)


def format_time(seconds):
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    if seconds < 60:
        return "%.0fs" % seconds
    if seconds < 3600:
        return "%.1fm" % (seconds / 60.0)
    return "%.1fh" % (seconds / 3600.0)


def make_optimizer(name, target):
    if name == "FIRE":
        return FIRE(target, logfile=None)
    if name == "BFGS":
        return BFGS(target, logfile=None)
    return LBFGS(target, logfile=None)


def relax_structure(atoms):
    """0 K relaxation of the input structure (optionally variable cell)."""
    if not finite_t_params.get("pre_optimize", True):
        return atoms
    fmax = float(finite_t_params.get("pre_opt_fmax", 0.01))
    steps = int(finite_t_params.get("pre_opt_steps", 300))
    name = finite_t_params.get("pre_opt_optimizer", "LBFGS")
    relax_cell = bool(finite_t_params.get("pre_opt_relax_cell", True))

    print("\n--- 0 K relaxation (%s, fmax = %.4f eV/A, max %d steps, %s) ---"
          % (name, fmax, steps, "atoms + cell" if relax_cell else "atoms only"))
    target = atoms
    if relax_cell:
        if CellFilter is None:
            print("  \u26a0\ufe0f  WARNING: no cell filter available in this ASE build - "
                  "relaxing atomic positions only.")
        else:
            target = CellFilter(atoms)
    opt = make_optimizer(name, target)
    t0 = time.perf_counter()
    opt.run(fmax=fmax, steps=steps)
    final_fmax = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
    stress = np.asarray(atoms.get_stress(voigt=True), dtype=float) * EV_TO_GPA
    print("  Done in %d steps (%.1fs): fmax = %.5f eV/A, "
          "residual pressure = %.3f GPa, V = %.3f A^3"
          % (opt.nsteps, time.perf_counter() - t0, final_fmax,
             -np.mean(stress[:3]), atoms.get_volume()))
    if final_fmax > fmax:
        print("  \u26a0\ufe0f  WARNING: 0 K relaxation did not converge to the requested fmax.")
    return atoms


def build_nvt(atoms, temperature_K, timestep_ase, rng):
    thermostat = finite_t_params.get("nvt_thermostat", "Langevin")
    if thermostat == "Berendsen":
        taut = float(finite_t_params.get("thermostat_taut", 100.0)) * units.fs
        return NVTBerendsen(atoms, timestep=timestep_ase,
                            temperature_K=temperature_K, taut=taut)
    if thermostat == "Nose-Hoover" and NVT_NOSE_HOOVER_AVAILABLE:
        tdamp = float(finite_t_params.get("thermostat_taut", 100.0)) * units.fs
        return NoseHooverChainNVT(atoms, timestep=timestep_ase,
                                  temperature_K=temperature_K, tdamp=tdamp)
    if thermostat == "Nose-Hoover":
        print("  \u26a0\ufe0f  WARNING: NoseHooverChainNVT is not available in this ASE "
              "version - falling back to Langevin.")
    friction = float(finite_t_params.get("friction", 0.02)) / units.fs
    # ASE 3.28 deprecated Langevin's own `fixcm=True` (it does not sample the
    # canonical distribution exactly) in favour of `fixcm=False` plus a FixCom
    # constraint. FixCom removes the centre-of-mass motion exactly, and because
    # it reports 3 removed degrees of freedom the reported temperature and the
    # kinetic stress are both built from peculiar velocities only - which is
    # what the thermodynamic stress should use. FixCom defines no adjust_stress,
    # so the virial part of the stress is untouched.
    if FIXCOM_AVAILABLE:
        if not any(isinstance(c, FixCom) for c in atoms.constraints):
            atoms.set_constraint(list(atoms.constraints) + [FixCom()])
        try:
            return Langevin(atoms, timestep=timestep_ase,
                            temperature_K=temperature_K, friction=friction,
                            fixcm=False, rng=rng)
        except TypeError:
            pass   # ASE too old to accept the keyword - fall through
    return Langevin(atoms, timestep=timestep_ase,
                    temperature_K=temperature_K, friction=friction, rng=rng)


def to_triangular_cell(atoms):
    """Rotate into a triangular cell, which the full-cell NPT integrator needs."""
    try:
        rcell, _ = atoms.cell.standard_form()
        atoms.set_cell(rcell, scale_atoms=True)
    except Exception as exc:
        print("  \u26a0\ufe0f  WARNING: could not convert the cell to triangular form (%s)."
              % exc)
    return atoms


def build_npt(atoms, temperature_K, pressure_au, timestep_ase, rng):
    """Barostat for the pre-equilibration, per the user's choice."""
    barostat = finite_t_params.get("npt_barostat", "Berendsen (isotropic)")
    taut = float(finite_t_params.get("npt_ttime", 100.0)) * units.fs
    taup = float(finite_t_params.get("npt_ptime", 1000.0)) * units.fs
    bulk = float(finite_t_params.get("npt_bulk_modulus", 140.0))
    if bulk <= 0:
        bulk = 140.0

    if barostat.startswith("Nose-Hoover"):
        if FULL_NPT_AVAILABLE:
            to_triangular_cell(atoms)
            pfactor = (taup ** 2) * (bulk * units.GPa)
            return FullCellNPT(atoms, timestep=timestep_ase,
                               temperature_K=temperature_K,
                               externalstress=pressure_au,
                               ttime=taut, pfactor=pfactor)
        print("  \u26a0\ufe0f  WARNING: the full-cell NPT integrator is unavailable - "
              "falling back to the isotropic Berendsen barostat.")
        barostat = "Berendsen (isotropic)"

    compressibility = 1.0 / (bulk * units.GPa)
    if barostat.startswith("Berendsen (anisotropic)"):
        if INHOMO_NPT_AVAILABLE:
            return Inhomogeneous_NPTBerendsen(
                atoms, timestep=timestep_ase, temperature_K=temperature_K,
                pressure_au=pressure_au, taut=taut, taup=taup,
                compressibility_au=compressibility, mask=(1, 1, 1))
        print("  \u26a0\ufe0f  WARNING: Inhomogeneous_NPTBerendsen is unavailable - "
              "falling back to the isotropic Berendsen barostat.")

    return NPTBerendsen(atoms, timestep=timestep_ase,
                        temperature_K=temperature_K, pressure_au=pressure_au,
                        taut=taut, taup=taup,
                        compressibility_au=compressibility)


def npt_pre_equilibration(reference, calculator, temperature_K, basename, job=None):
    """Relax the cell at (T, P) and return the reference cell + a hot snapshot.

    The reference cell is the time average over the NPT production part (or the
    final cell if the user asked for that), which is what the strains are then
    applied to.
    """
    timestep_fs = float(finite_t_params["timestep"])
    timestep_ase = timestep_fs * units.fs
    pressure_GPa = float(finite_t_params.get("pressure_GPa", 0.0))
    pressure_au = pressure_GPa * units.GPa
    n_eq = int(finite_t_params.get("npt_equilibration_steps", 0))
    n_prod = int(finite_t_params.get("npt_production_steps", 0))
    log_interval = int(finite_t_params.get("log_interval", 200))
    seed = int(finite_t_params.get("seed", 42))

    atoms = reference.copy()
    atoms.calc = calculator
    rng = np.random.RandomState(seed + int(temperature_K))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    Stationary(atoms)

    print("\n--- NPT pre-equilibration at %.1f K / %.3f GPa "
          "(%s, %d + %d steps) ---"
          % (temperature_K, pressure_GPa,
             finite_t_params.get("npt_barostat", "Berendsen (isotropic)"),
             n_eq, n_prod))
    print("  Starting volume: %.4f A^3" % atoms.get_volume())

    dyn = build_npt(atoms, temperature_K, pressure_au, timestep_ase, rng)

    if n_eq > 0:
        progress = MDProgress(atoms, dyn, n_eq, "NPT equil", log_interval, job=job)
        dyn.attach(progress, interval=log_interval)
        dyn.run(n_eq)
        dyn.observers.clear()

    collector = CellCollector(atoms)
    sample_interval = max(1, int(finite_t_params.get("sample_interval", 5)))
    dyn.attach(collector, interval=sample_interval)
    progress = MDProgress(atoms, dyn, n_prod, "NPT prod ", log_interval, job=job)
    progress.steps_before = n_eq
    dyn.attach(progress, interval=log_interval)
    dyn.run(n_prod)
    dyn.observers.clear()
    if job is not None:
        job.finish_run(n_eq + n_prod)

    if not collector.cells:
        print("  \u26a0\ufe0f  WARNING: no NPT samples collected - using the final cell.")
        ref_cell = np.array(atoms.get_cell()[:], dtype=float)
        cell_info = {}
    else:
        cells = np.array(collector.cells)
        mean_cell = cells.mean(axis=0)
        n_blocks = int(finite_t_params.get("n_blocks", 5))
        vol_mean, vol_err = block_stats(np.array(collector.volumes), n_blocks)
        t_mean = float(np.mean(collector.temperatures))
        p_mean, p_err = block_stats(np.array(collector.pressures), n_blocks)
        if str(finite_t_params.get("reference_cell", "")).startswith("Final"):
            ref_cell = np.array(atoms.get_cell()[:], dtype=float)
            print("  Reference cell: final NPT cell")
        else:
            ref_cell = mean_cell
            print("  Reference cell: time average over %d samples"
                  % len(collector.cells))
        print("  <V> = %.4f +- %.4f A^3   <T> = %.1f K   <P> = %.3f +- %.3f GPa"
              % (vol_mean[0], vol_err[0], t_mean, p_mean[0], p_err[0]))
        cell_info = {
            "mean_volume_A3": float(vol_mean[0]),
            "volume_error_A3": float(vol_err[0]),
            "mean_temperature_K": t_mean,
            "mean_pressure_GPa": float(p_mean[0]),
            "pressure_error_GPa": float(p_err[0]),
            "mean_cell": mean_cell.tolist(),
            "n_samples": len(collector.cells),
        }

    # The hot snapshot is squeezed into the reference cell so every strained run
    # starts from an equilibrated configuration of exactly that cell.
    hot = atoms.copy()
    hot.set_cell(ref_cell, scale_atoms=True)
    reference_file = os.path.join(
        temperature_dir(temperature_K),
        "%s_T%sK_npt_reference.xyz" % (basename, temperature_tag(temperature_K)))
    write(reference_file, hot, format="extxyz")
    reference_npz = os.path.join(
        CHECKPOINT_DIR,
        "%s_T%sK_reference.npz" % (basename, temperature_tag(temperature_K)))
    save_atoms_npz(reference_npz, hot)
    return ref_cell, hot, cell_info, reference_npz


def run_strain_state(hot_reference, calculator, temperature_K, j, delta,
                     state_index, n_states, basename, job=None):
    """One strained NVT run; returns the time-averaged stress in GPa."""
    timestep_fs = float(finite_t_params["timestep"])
    timestep_ase = timestep_fs * units.fs
    n_eq = int(finite_t_params.get("nvt_equilibration_steps", 0))
    n_prod = int(finite_t_params.get("nvt_production_steps", 1000))
    sample_interval = max(1, int(finite_t_params.get("sample_interval", 5)))
    log_interval = int(finite_t_params.get("log_interval", 200))
    n_blocks = int(finite_t_params.get("n_blocks", 5))
    seed = int(finite_t_params.get("seed", 42))

    label = "eps_%s = %+.4f" % (VOIGT_LABELS[j], delta)
    print("\n  [%d/%d] %s at %.1f K"
          % (state_index, n_states, label, temperature_K))

    atoms = hot_reference.copy()
    atoms.set_cell(strained_cell(hot_reference.get_cell(), j, delta),
                   scale_atoms=True)
    atoms.calc = calculator

    rng = np.random.RandomState(seed + 977 * state_index + int(temperature_K))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    Stationary(atoms)

    dyn = build_nvt(atoms, temperature_K, timestep_ase, rng)

    if n_eq > 0:
        progress = MDProgress(atoms, dyn, n_eq, "NVT equil", log_interval, job=job)
        dyn.attach(progress, interval=log_interval)
        dyn.run(n_eq)
        dyn.observers.clear()

    collector = StressCollector(atoms, dyn, timestep_fs)
    dyn.attach(collector, interval=sample_interval)
    progress = MDProgress(atoms, dyn, n_prod, "NVT prod ", log_interval, job=job)
    progress.steps_before = n_eq
    dyn.attach(progress, interval=log_interval)

    if finite_t_params.get("save_trajectories", False):
        traj_interval = max(1, int(finite_t_params.get("traj_interval", 500)))
        traj_path = os.path.join(
            trajectory_dir(temperature_K), "%s_T%sK_eps%s_%+.4f.xyz"
            % (basename, temperature_tag(temperature_K), VOIGT_LABELS[j], delta))
        if os.path.exists(traj_path):
            os.remove(traj_path)

        def _write_frame(atoms=atoms, path=traj_path):
            write(path, atoms, format="extxyz", append=True)

        dyn.attach(_write_frame, interval=traj_interval)

    dyn.run(n_prod)
    dyn.observers.clear()
    if job is not None:
        job.finish_run(n_eq + n_prod)

    # Convert first, then average - the same order restore_state uses on the
    # stored samples, so a resumed run reproduces this bit for bit.
    samples_GPa = np.asarray(collector.stresses, dtype=float) * EV_TO_GPA
    mean_GPa, sem_GPa = block_stats(samples_GPa, n_blocks)
    mean_T = float(np.mean(collector.temperatures)) if collector.temperatures else 0.0

    print("      <T> = %.1f K over %d samples" % (mean_T, len(collector.stresses)))
    print("      <sigma> (GPa) = " + "  ".join(
        "%s %8.3f+-%.3f" % (VOIGT_LABELS[i], mean_GPa[i], sem_GPa[i])
        for i in range(6)))
    if job is not None:
        print("      %d of %d strain states left at this temperature  |  %s"
              % (n_states - state_index, n_states, job.summary()), flush=True)

    return {
        "voigt_index": j,
        "delta": float(delta),
        "stress_mean_GPa": mean_GPa,
        "stress_sem_GPa": sem_GPa,
        "mean_temperature_K": mean_T,
        "n_samples": len(collector.stresses),
        "samples_GPa": samples_GPa,
        "times_ps": np.asarray(collector.times_ps, dtype=float),
        "temperatures_K": np.asarray(collector.temperatures, dtype=float),
    }


def static_elastic_tensor(atoms_relaxed, calculator, symmetry, components):
    """0 K reference tensor from the same strains, without MD.

    The atoms are relaxed inside each strained cell (cell held fixed) unless
    that is switched off. This matters: for any crystal whose basis has
    internal degrees of freedom - hcp, diamond, most compounds - the
    clamped-ion constants are systematically too stiff, badly so for the shear
    constants. Only structures whose symmetry forbids internal relaxation
    (e.g. fcc/bcc with a one-atom basis) are unaffected.
    """
    relax_ions = bool(finite_t_params.get("static_ion_relax", True))
    fmax = float(finite_t_params.get("static_ion_fmax", 0.005))
    max_steps = int(finite_t_params.get("static_ion_steps", 200))
    opt_name = finite_t_params.get("pre_opt_optimizer", "LBFGS")

    print("\n--- 0 K (static) reference elastic tensor ---")
    if relax_ions:
        print("  Ions relaxed inside each strained cell "
              "(%s, fmax = %.4f eV/A, max %d steps)" % (opt_name, fmax, max_steps))
    else:
        print("  Clamped-ion: atoms are NOT relaxed inside the strained cell.")

    ref_cell = np.array(atoms_relaxed.get_cell()[:], dtype=float)
    columns = {}
    unconverged = 0
    for j in components:
        deltas, stresses = [], []
        for d in strain_values():
            work = atoms_relaxed.copy()
            work.set_cell(strained_cell(ref_cell, j, d), scale_atoms=True)
            work.calc = calculator
            if relax_ions and len(work) > 1:
                opt = make_optimizer(opt_name, work)
                opt.run(fmax=fmax, steps=max_steps)
                residual = float(np.max(np.linalg.norm(work.get_forces(), axis=1)))
                if residual > fmax:
                    unconverged += 1
            stresses.append(np.asarray(work.get_stress(voigt=True), dtype=float)
                            * EV_TO_GPA)
            deltas.append(d)
        stresses = np.array(stresses)
        slope = np.zeros(6)
        for i in range(6):
            slope[i], _, _ = weighted_slope(deltas, stresses[:, i],
                                            np.zeros(len(deltas)))
        columns[j] = {"slope": slope, "err": np.zeros(6)}
        print("  eps_%s done" % VOIGT_LABELS[j])

    if unconverged:
        print("  \u26a0\ufe0f  WARNING: the ionic relaxation hit the step limit in "
              "%d of the strained cells - the 0 K reference may be slightly too "
              "stiff." % unconverged)

    C0, _ = assemble_tensor(columns, symmetry)
    print_tensor(C0, title="Static (0 K) C_ij (GPa)%s"
                 % ("" if relax_ions else "  [clamped ion]"))
    if not relax_ions:
        print("  Note: clamped-ion constants are an upper bound for any crystal "
              "with\n        internal degrees of freedom.")
    return C0


def pressure_corrected_tensor(B, pressure_GPa):
    """Convert measured stress-strain coefficients into elastic constants.

    What this method measures is B_ij = d<sigma_i>/d eps_j, the Birch (or
    stress-strain) coefficients. Under a finite hydrostatic pressure P those
    differ from the elastic constants C_ij (the free-energy second derivatives
    that are normally tabulated) by

        C_ijkl = B_ijkl + P (delta_il delta_jk + delta_ik delta_jl - delta_ij delta_kl)

    which in Voigt notation is +P on C11/C22/C33 and on C44/C55/C66, and -P on
    C12/C13/C23, with every other component unchanged. At P = 0 the two
    coincide, which is why this only matters for runs at pressure.
    """
    P = float(pressure_GPa)
    C = np.array(B, dtype=float).copy()
    if abs(P) < 1e-12:
        return C
    for i in range(3):
        C[i, i] += P              # C11, C22, C33
        C[i + 3, i + 3] += P      # C44, C55, C66
        for k in range(3):
            if i != k:
                C[i, k] -= P      # C12, C13, C23
    return C


# ---------------------------------------------------------------------------
# Adaptive production length.
#
# Instead of running every strained cell for a fixed number of production
# steps, all strain states of one temperature can be advanced *together* in
# segments. After every segment the whole tensor is re-fitted from the
# trajectory accumulated so far and compared with the previous check; once the
# constants stop moving by more than the tolerance the production ends for
# every strain state at once. That is the usual "recompute C_ij from the first
# 10, 20 and 50 ps of the same trajectory" test, run automatically and used to
# decide when to stop.
#
# Running a trajectory in segments is not an approximation: the integrator and
# its random stream continue exactly where they left off, so the result is the
# same trajectory a single long run would have produced.
# ---------------------------------------------------------------------------

CONVERGENCE_DIRNAME = "convergence"
# A percentage change is meaningless for a constant that is essentially zero
# (the off-diagonal blocks of a triclinic fit), so below this magnitude only
# the absolute criterion applies.
CONVERGENCE_REL_FLOOR_GPA = 1.0


def convergence_enabled():
    return bool(finite_t_params.get("use_convergence_check", False))


def convergence_segment_steps():
    """MD steps every strain state advances between two convergence checks."""
    interval_ps = float(finite_t_params.get("convergence_interval_ps", 1.0))
    timestep_fs = float(finite_t_params["timestep"])
    return max(1, int(round(interval_ps * 1000.0 / timestep_fs)))


def convergence_criterion_key():
    key = finite_t_params.get("convergence_criterion_key")
    if key:
        return str(key)
    text = str(finite_t_params.get("convergence_criterion", "either")).lower()
    if text.startswith("both"):
        return "both"
    if text.startswith("absolute"):
        return "absolute"
    if text.startswith("relative"):
        return "relative"
    return "either"


def convergence_dir(temperature_K, create=True):
    """Where the convergence history of one temperature is written."""
    path = os.path.join(temperature_dir(temperature_K, create=create),
                        CONVERGENCE_DIRNAME)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def tracked_components(symmetry):
    """The C_ij whose drift decides whether the runs may stop.

    Only the constants the chosen symmetry determines independently - the ones
    filled in by a symmetry relation carry no extra information and would just
    count the same number twice.
    """
    if symmetry == "cubic":
        return [(0, 0), (0, 1), (3, 3)]
    if symmetry == "hexagonal":
        return [(0, 0), (0, 1), (0, 2), (2, 2), (3, 3)]
    return [(i, k) for i in range(6) for k in range(i, 6)]


def modulus_uncertainties(C, C_err, n_mc=200, rng=None):
    """Error bars on K, G and E from the error bars on C_ij.

    The moduli are non-linear functions of the tensor (the Reuss average needs
    its inverse), so the C_ij uncertainties are propagated by resampling the
    tensor rather than by a derivative. This is the same resampling
    analyze_tensor does for the final result, restricted to the three moduli so
    that it is cheap enough to run at every convergence check. Draws that come
    out mechanically unstable are rejected, and the spread is read off the
    16/84 percentiles so a single outlier cannot dominate it.
    """
    zero = {"K": 0.0, "G": 0.0, "E": 0.0}
    if C_err is None or not np.any(C_err > 0) or n_mc <= 0:
        return zero
    if rng is None:
        rng = np.random.RandomState(0)
    samples = {"K": [], "G": [], "E": []}
    for _ in range(int(n_mc)):
        noise = rng.normal(0.0, 1.0, size=(6, 6)) * C_err
        noise = 0.5 * (noise + noise.T)
        C_sample = C + noise
        try:
            if np.min(np.linalg.eigvalsh(C_sample)) <= 0.0:
                continue
            m = vrh_moduli(C_sample)
        except Exception:
            continue
        samples["K"].append(m["K"])
        samples["G"].append(m["G"])
        samples["E"].append(m["youngs_modulus"])
    errors = {}
    for key, values in samples.items():
        values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        if len(values) > 8:
            lo, hi = np.percentile(values, [16.0, 84.0])
            errors[key] = float(0.5 * (hi - lo))
        else:
            errors[key] = 0.0
    return errors


def tracked_quantities(C, C_err, symmetry, moduli=None, moduli_err=None):
    """[(label, value, error)] that one convergence check compares."""
    items = [("C%d%d" % (i + 1, k + 1), float(C[i, k]), float(C_err[i, k]))
             for (i, k) in tracked_components(symmetry)]
    if finite_t_params.get("convergence_include_moduli", True):
        if moduli is None:
            moduli = vrh_moduli(C)
        if moduli_err is None:
            moduli_err = modulus_uncertainties(C, C_err)
        items.append(("K", float(moduli["K"]), float(moduli_err.get("K", 0.0))))
        items.append(("G", float(moduli["G"]), float(moduli_err.get("G", 0.0))))
    return items


def convergence_verdict(previous, current):
    """Compare two consecutive checks.

    Returns (passed, max |change| in GPa, max |change| in %, per-quantity rows).
    """
    tol_abs = float(finite_t_params.get("convergence_tol_GPa", 5.0))
    tol_rel = float(finite_t_params.get("convergence_tol_percent", 2.0))
    key = convergence_criterion_key()

    before = dict((name, value) for name, value, _ in previous)
    rows = []
    max_abs = 0.0
    max_rel = 0.0
    for name, value, err in current:
        if name not in before:
            continue
        change = abs(value - before[name])
        scale = max(abs(value), abs(before[name]))
        rel = (100.0 * change / scale if scale > CONVERGENCE_REL_FLOOR_GPA
               else float("nan"))
        rows.append({"quantity": name, "value_GPa": value, "error_GPa": err,
                     "previous_GPa": before[name], "change_GPa": change,
                     "change_percent": rel})
        max_abs = max(max_abs, change)
        if np.isfinite(rel):
            max_rel = max(max_rel, rel)

    pass_abs = max_abs < tol_abs
    pass_rel = max_rel < tol_rel
    if key == "absolute":
        passed = pass_abs
    elif key == "relative":
        passed = pass_rel
    elif key == "both":
        passed = pass_abs and pass_rel
    else:
        passed = pass_abs or pass_rel
    return bool(passed), float(max_abs), float(max_rel), rows


class StrainRun:
    """One strained cell that can be advanced a segment at a time.

    Interleaving is what makes an early stop possible: C_ij can only be
    re-fitted once every strain state has reached the same amount of
    production time, so the states take turns instead of running one after
    the other.
    """

    def __init__(self, hot_reference, calculator, temperature_K, j, delta,
                 index, n_states, basename, job=None, restored=None):
        self.j = int(j)
        self.delta = float(delta)
        self.index = int(index)
        self.temperature_K = float(temperature_K)
        self.basename = basename
        self.job = job
        self.timestep_fs = float(finite_t_params["timestep"])
        self.n_eq = int(finite_t_params.get("nvt_equilibration_steps", 0))
        self.n_blocks = int(finite_t_params.get("n_blocks", 5))
        self.label = "eps_%s = %+.4f" % (VOIGT_LABELS[self.j], self.delta)
        self.prod_steps = 0
        self.samples_GPa = []
        self.times_ps = []
        self.temperatures = []
        self.restored_complete = False

        seed = int(finite_t_params.get("seed", 42))
        # Same stream as a plain sequential run of this state, so the two modes
        # produce the same trajectory.
        self.rng = np.random.RandomState(
            seed + 977 * self.index + int(temperature_K))

        if restored is not None:
            atoms = restored["atoms"]
            self.prod_steps = int(restored["production_steps"])
            # The samples are written before the checkpoint index that records
            # how far the run got, so a crash in between can leave the .npz one
            # segment ahead. The index decides; anything past it is dropped.
            horizon_ps = ((self.n_eq + self.prod_steps) * self.timestep_fs
                          / 1000.0 + 1e-9)
            keep = int(np.searchsorted(
                np.asarray(restored["times_ps"], dtype=float), horizon_ps,
                side="right"))
            self.samples_GPa = [np.asarray(row, dtype=float)
                                for row in restored["samples_GPa"][:keep]]
            self.times_ps = [float(t) for t in restored["times_ps"][:keep]]
            stored_T = np.asarray(restored.get("temperatures_K", []),
                                  dtype=float)[:keep]
            if len(stored_T) == len(self.samples_GPa):
                self.temperatures = [float(t) for t in stored_T]
            elif self.samples_GPa:
                # Older checkpoint without the temperature series: fall back to
                # its mean so the reported <T> stays representative.
                self.temperatures = [float(restored.get("mean_temperature_K", 0.0))
                                     ] * len(self.samples_GPa)
            self.restored_complete = bool(restored.get("complete", False))
            self.equilibrated = True
        else:
            atoms = hot_reference.copy()
            atoms.set_cell(strained_cell(hot_reference.get_cell(), self.j, self.delta),
                           scale_atoms=True)
            self.equilibrated = False
        atoms.calc = calculator
        self.atoms = atoms

        if not self.equilibrated:
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature_K,
                                         rng=self.rng)
            Stationary(self.atoms)

        self.dyn = build_nvt(self.atoms, temperature_K,
                             self.timestep_fs * units.fs, self.rng)
        if self.equilibrated:
            # Continue the step counter where the stored run stopped, so the
            # sampling interval keeps falling on the same steps.
            try:
                self.dyn.nsteps = self.n_eq + self.prod_steps
            except Exception:
                pass
        self.progress = None
        self.n_states = int(n_states)

    # -- setup -------------------------------------------------------------
    def equilibrate(self, log_interval):
        """The burn-in that is discarded after the strain is applied."""
        if self.equilibrated or self.n_eq <= 0:
            self.equilibrated = True
            return
        print("\n  [%d/%d] %s at %.1f K - NVT equilibration (%d steps)"
              % (self.index, self.n_states, self.label, self.temperature_K,
                 self.n_eq))
        progress = MDProgress(self.atoms, self.dyn, self.n_eq, "NVT equil",
                              log_interval, job=self.job)
        self.dyn.attach(progress, interval=log_interval)
        self.dyn.run(self.n_eq)
        self.dyn.observers.clear()
        self.equilibrated = True
        if self.job is not None:
            self.job.add_steps(self.n_eq)

    def start_production(self, n_prod_max, log_interval):
        """Attach the stress sampler and the progress line for the production."""
        sample_interval = max(1, int(finite_t_params.get("sample_interval", 5)))
        self.dyn.observers.clear()
        self.dyn.attach(self._sample, interval=sample_interval)
        self.progress = MDProgress(self.atoms, self.dyn, n_prod_max,
                                   "NVT prod %s" % self.label, log_interval,
                                   job=self.job)
        self.progress.start_step = self.n_eq
        self.dyn.attach(self.progress, interval=log_interval)
        self.progress.pause()

        if finite_t_params.get("save_trajectories", False):
            traj_interval = max(1, int(finite_t_params.get("traj_interval", 500)))
            self.traj_path = os.path.join(
                trajectory_dir(self.temperature_K), "%s_T%sK_eps%s_%+.4f.xyz"
                % (self.basename, temperature_tag(self.temperature_K),
                   VOIGT_LABELS[self.j], self.delta))
            if self.prod_steps == 0 and os.path.exists(self.traj_path):
                os.remove(self.traj_path)
            self.dyn.attach(self._write_frame, interval=traj_interval)

    # -- observers ---------------------------------------------------------
    def _sample(self):
        self.samples_GPa.append(
            np.asarray(thermo_stress_voigt(self.atoms), dtype=float) * EV_TO_GPA)
        self.temperatures.append(self.atoms.get_temperature())
        self.times_ps.append(
            int(self.dyn.get_number_of_steps()) * self.timestep_fs / 1000.0)

    def _write_frame(self):
        write(self.traj_path, self.atoms, format="extxyz", append=True)

    # -- running -----------------------------------------------------------
    def advance_to(self, target_steps):
        """Run production until this state has `target_steps` production steps."""
        steps = int(target_steps) - self.prod_steps
        if steps <= 0:
            return 0
        # The job note in the progress line should only show the segment that is
        # in flight; the finished ones are already counted in the job clock.
        self.progress.steps_before = -self.prod_steps
        self.progress.resume()
        self.dyn.run(steps)
        self.progress.pause()
        self.prod_steps += steps
        if self.job is not None:
            self.job.add_steps(steps)
        return steps

    def state(self, complete=False, with_snapshot=False):
        """The same record a sequential strained run produces."""
        samples = (np.asarray(self.samples_GPa, dtype=float)
                   if self.samples_GPa else np.zeros((0, 6)))
        if len(samples) > 0:
            mean_GPa, sem_GPa = block_stats(samples, self.n_blocks)
        else:
            mean_GPa, sem_GPa = np.zeros(6), np.zeros(6)
        record = {
            "voigt_index": self.j,
            "delta": self.delta,
            "stress_mean_GPa": mean_GPa,
            "stress_sem_GPa": sem_GPa,
            "mean_temperature_K": (float(np.mean(self.temperatures))
                                   if self.temperatures else 0.0),
            "n_samples": len(samples),
            "samples_GPa": samples,
            "times_ps": np.asarray(self.times_ps, dtype=float),
            "temperatures_K": np.asarray(self.temperatures, dtype=float),
            "production_steps": int(self.prod_steps),
            "complete": bool(complete),
        }
        if with_snapshot:
            record["snapshot"] = self.atoms
        return record


def restore_run(checkpoint, temperature_K, j, delta, calculator):
    """A half-finished (or finished) strained run ready to be continued."""
    if checkpoint is None:
        return None
    record = restore_state(checkpoint, temperature_K, j, delta, allow_partial=True)
    if record is None:
        return None
    name = record.get("snapshot_file")
    if not name:
        # Stored by an older run that only ever saved finished states: usable as
        # a result, but there is no snapshot to continue from.
        if not record.get("complete", True):
            return None
        name = None
    atoms = None
    if name:
        path = os.path.join(CHECKPOINT_DIR, name)
        if os.path.exists(path):
            try:
                atoms = load_atoms_npz(path)
                atoms.set_pbc(True)
            except Exception as exc:
                print("      ⚠️  WARNING: could not read the stored run "
                      "snapshot (%s) - this strain state restarts." % exc)
                return None
    if atoms is None:
        return None
    record["atoms"] = atoms
    return record


def tensor_from_states(states, components, symmetry):
    """(C, C_err, columns, warnings) from the strained runs done so far."""
    by_component = {}
    for state in states:
        by_component.setdefault(int(state["voigt_index"]), []).append(state)

    columns = {}
    warnings = []
    for j in components:
        per_delta = sorted(by_component.get(j, []), key=lambda s: s["delta"])
        if len(per_delta) < 2:
            columns[j] = {"slope": np.zeros(6), "err": np.zeros(6)}
            continue
        x = np.array([s["delta"] for s in per_delta])
        means = np.array([s["stress_mean_GPa"] for s in per_delta])
        sems = np.array([s["stress_sem_GPa"] for s in per_delta])
        slope = np.zeros(6)
        err = np.zeros(6)
        for i in range(6):
            slope[i], err[i], residual = weighted_slope(x, means[:, i], sems[:, i])
            if residual > 1.0:
                warnings.append("sigma_%s vs eps_%s: max fit residual %.2f GPa"
                                % (VOIGT_LABELS[i], VOIGT_LABELS[j], residual))
        columns[j] = {"slope": slope, "err": err}

    C, C_err = assemble_tensor(columns, symmetry)
    return C, C_err, columns, warnings


def print_convergence_check(entry, tol_abs, tol_rel, needed, streak):
    """One check, printed as a compact table."""
    print("\n  --- convergence check %d: %.2f ps of production per strain state "
          "(%d stress samples) ---"
          % (entry["check"], entry["production_ps"], entry["n_samples"]))
    if not entry["rows"]:
        print("      (first check - nothing to compare against yet: "
              + ", ".join("%s = %.2f GPa" % (name, value)
                          for name, value, _ in entry["quantities"]) + ")")
        return
    print("      %-6s %12s %11s %12s %10s"
          % ("", "value (GPa)", "stat. err", "change (GPa)", "change (%)"))
    for row in entry["rows"]:
        rel = ("%10.2f" % row["change_percent"]
               if np.isfinite(row["change_percent"]) else "%10s" % "-")
        print("      %-6s %12.2f %11.2f %12.3f %s"
              % (row["quantity"], row["value_GPa"], row["error_GPa"],
                 row["change_GPa"], rel))
    print("      max |dC| = %.3f GPa (tol %.2f)   max relative = %.3f %% (tol %.2f)"
          % (entry["max_abs_GPa"], tol_abs, entry["max_rel_percent"], tol_rel))
    # Drift that is already smaller than the statistical error is noise, not a
    # trend - worth seeing next to the tolerance.
    max_err = max([row["error_GPa"] for row in entry["rows"]] or [0.0])
    if max_err > 0:
        print("      largest statistical error among them: %.3f GPa - the drift "
              "is %s it" % (max_err,
                            "within" if entry["max_abs_GPa"] <= max_err else "above"))
    if entry["passed"]:
        print("      ✓ within tolerance (%d of %d consecutive checks needed)"
              % (streak, needed))
    else:
        print("      ✗ still drifting - continuing")


def write_convergence_outputs(basename, temperature_K, history, symmetry,
                              final=True):
    """The convergence history of one temperature: CSV + figures.

    Called after every check while the runs are still going (PNG only, so the
    files on disk always show the latest comparison) and once more when the
    temperature is finished, which adds the vector copies.
    """
    if not history:
        return
    outdir = convergence_dir(temperature_K)
    tag = temperature_tag(temperature_K)
    formats = ("png", "pdf") if final else ("png",)

    labels = [name for name, _, _ in history[-1]["quantities"]]
    final_values = dict((name, value) for name, value, _ in history[-1]["quantities"])
    c_labels = [name for name in labels if name.startswith("C")]
    # A triclinic fit tracks 21 constants; the CSV keeps all of them but the
    # overview figure would be unreadable, so it shows the largest ones.
    plotted = c_labels
    if len(c_labels) > 8:
        by_size = sorted(c_labels, key=lambda name: -abs(final_values[name]))
        keep = set(by_size[:8])
        plotted = [name for name in c_labels if name in keep]

    rows = []
    for entry in history:
        changes = dict((r["quantity"], r) for r in entry["rows"])
        row = {"check": entry["check"],
               "production_ps": entry["production_ps"],
               "production_steps": entry["production_steps"],
               "n_stress_samples": entry["n_samples"]}
        for name, value, err in entry["quantities"]:
            row["%s_GPa" % name] = value
            row["%s_err_GPa" % name] = err
            row["%s_change_GPa" % name] = changes.get(name, {}).get(
                "change_GPa", float("nan"))
            row["%s_change_percent" % name] = changes.get(name, {}).get(
                "change_percent", float("nan"))
        row["youngs_modulus_GPa"] = entry["moduli"]["E"]
        row["youngs_modulus_err_GPa"] = entry.get("moduli_err", {}).get("E", 0.0)
        row["max_change_GPa"] = entry["max_abs_GPa"] if entry["rows"] else float("nan")
        row["max_change_percent"] = (entry["max_rel_percent"] if entry["rows"]
                                     else float("nan"))
        row["within_tolerance"] = bool(entry["passed"])
        rows.append(row)
    csv_path = os.path.join(
        outdir, "%s_T%sK_convergence.csv" % (basename, tag))
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    times = np.array([e["production_ps"] for e in history], dtype=float)
    tol_abs = float(finite_t_params.get("convergence_tol_GPa", 5.0))
    tol_rel = float(finite_t_params.get("convergence_tol_percent", 2.0))
    written = []

    def series(name):
        values = np.array([dict((n, v) for n, v, _ in e["quantities"]).get(name, np.nan)
                           for e in history], dtype=float)
        errors = np.array([dict((n, s) for n, _, s in e["quantities"]).get(name, 0.0)
                           for e in history], dtype=float)
        return values, errors

    def single_quantity_figure(values, errors, ylabel, label, colour, stem,
                               title):
        """One quantity against production time, with its tolerance band."""
        fig, ax = plt.subplots(figsize=(7.6, 5.6))
        ax.errorbar(times, values, yerr=errors, marker="o", color=colour,
                    mfc="white", mew=2.2, label=label, zorder=3)
        last = values[-1]
        if np.isfinite(last):
            ax.axhline(last, ls=":", lw=1.6, color=colour, alpha=0.8, zorder=1)
            ax.axhspan(last - tol_abs, last + tol_abs, color=colour, alpha=0.10,
                       lw=0, zorder=0,
                       label=r"final value $\pm$ %.3g GPa" % tol_abs)
        # The tolerance band is a backdrop, not the subject: keep the axis on
        # the data so the curve stays readable however wide the tolerance is.
        lo = float(np.nanmin(values - errors))
        hi = float(np.nanmax(values + errors))
        if np.isfinite(lo) and np.isfinite(hi):
            pad = max(0.08 * (hi - lo), 0.005 * max(abs(hi), 1.0))
            ax.set_ylim(lo - pad, hi + pad)
        style_axis(ax, ylabel, xlabel="Production time per strain state (ps)")
        ax.set_title(title, fontsize=17)
        return save_publication_figure(fig, outdir, stem, formats=formats)

    with plt.rc_context(PUB_RC):
        # ---- 1) all tracked constants in one overview ----------------------
        if plotted:
            fig, ax = plt.subplots(figsize=(8.0, 5.8))
            for k, name in enumerate(plotted):
                values, errors = series(name)
                if np.all(np.abs(values) < 1e-9):
                    continue
                colour = SERIES_COLOURS[k % len(SERIES_COLOURS)]
                # More tracked constants than colours: the second pass is dashed.
                style = "-" if k < len(SERIES_COLOURS) else "--"
                ax.plot(times, values, marker="o", ls=style, color=colour,
                        mfc="white", mew=2.0,
                        label="$%s_{%s}$" % (name[0], name[1:]), zorder=3)
                ax.fill_between(times, values - errors, values + errors,
                                color=colour, alpha=0.15, lw=0, zorder=1)
                ax.axhline(values[-1], ls=":", lw=1.2, color=colour, alpha=0.6,
                           zorder=0)
            style_axis(ax, "Elastic constant (GPa)",
                       xlabel="Production time per strain state (ps)")
            ax.legend(ncol=2)
            ax.set_title("%s - C$_{ij}$ re-fitted from the trajectory so far "
                         "(%.0f K)" % (basename, temperature_K), fontsize=17)
            written += save_publication_figure(
                fig, outdir, "%s_T%sK_convergence_constants" % (basename, tag),
                formats=formats)

        # ---- 2) one figure per elastic constant ----------------------------
        # Constants that are zero by symmetry carry no information and would
        # only fill the folder with flat lines.
        for k, name in enumerate(c_labels):
            if abs(final_values.get(name, 0.0)) < CONVERGENCE_REL_FLOOR_GPA:
                continue
            values, errors = series(name)
            pretty = "$%s_{%s}$" % (name[0], name[1:])
            written += single_quantity_figure(
                values, errors, "Elastic constant (GPa)", pretty,
                SERIES_COLOURS[k % len(SERIES_COLOURS)],
                "%s_T%sK_convergence_%s" % (basename, tag, name),
                "%s - %s vs production time (%.0f K)"
                % (basename, pretty, temperature_K))

        # ---- 3) one figure per averaged modulus ----------------------------
        for key, ylabel, pretty, colour, stem in [
                ("K", "Bulk modulus $K$ (GPa)", "$K$ (Voigt-Reuss-Hill)",
                 SERIES_COLOURS[0], "bulk_modulus"),
                ("G", "Shear modulus $G$ (GPa)", "$G$ (Voigt-Reuss-Hill)",
                 SERIES_COLOURS[1], "shear_modulus"),
                ("E", "Young's modulus $E$ (GPa)", "$E$ (Voigt-Reuss-Hill)",
                 SERIES_COLOURS[2], "youngs_modulus")]:
            values = np.array([e["moduli"][key] for e in history], dtype=float)
            errors = np.array([e.get("moduli_err", {}).get(key, 0.0)
                               for e in history], dtype=float)
            if not np.any(np.isfinite(values)):
                continue
            written += single_quantity_figure(
                values, errors, ylabel, pretty, colour,
                "%s_T%sK_convergence_%s" % (basename, tag, stem),
                "%s - %s vs production time (%.0f K)"
                % (basename, ylabel.split(" (")[0].replace("$", ""),
                   temperature_K))

        # ---- 4) the three moduli together, for a quick look -----------------
        fig, ax = plt.subplots(figsize=(8.0, 5.8))
        for k, (key, label) in enumerate([("K", "Bulk $K$"), ("G", "Shear $G$"),
                                          ("E", "Young's $E$")]):
            values = np.array([e["moduli"][key] for e in history], dtype=float)
            errors = np.array([e.get("moduli_err", {}).get(key, 0.0)
                               for e in history], dtype=float)
            if not np.any(np.isfinite(values)):
                continue
            colour = SERIES_COLOURS[k % len(SERIES_COLOURS)]
            ax.errorbar(times, values, yerr=errors, marker="o", color=colour,
                        mfc="white", mew=2.0, label=label, zorder=3)
            ax.axhline(values[-1], ls=":", lw=1.2, color=colour, alpha=0.6)
        style_axis(ax, "Modulus (GPa)",
                   xlabel="Production time per strain state (ps)")
        ax.set_title("%s - Voigt-Reuss-Hill moduli vs production time (%.0f K)"
                     % (basename, temperature_K), fontsize=17)
        written += save_publication_figure(
            fig, outdir, "%s_T%sK_convergence_moduli" % (basename, tag),
            formats=formats)

        # ---- 5) how much each check moved, against the tolerances -----------
        if len(history) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.4))
            for k, name in enumerate(plotted + [n for n in labels
                                                if not n.startswith("C")]):
                colour = SERIES_COLOURS[k % len(SERIES_COLOURS)]
                style = "-" if k < len(SERIES_COLOURS) else "--"
                pts_t, pts_abs, pts_rel = [], [], []
                for entry in history[1:]:
                    row = dict((r["quantity"], r) for r in entry["rows"]).get(name)
                    if row is None:
                        continue
                    pts_t.append(entry["production_ps"])
                    # A change of exactly zero has no place on a log axis.
                    pts_abs.append(max(row["change_GPa"], 1e-3))
                    pts_rel.append(max(row["change_percent"], 1e-3)
                                   if np.isfinite(row["change_percent"])
                                   else np.nan)
                if not pts_t:
                    continue
                axes[0].plot(pts_t, pts_abs, marker="o", ms=6, lw=1.4, ls=style,
                             color=colour, mfc="white", alpha=0.85, label=name)
                axes[1].plot(pts_t, pts_rel, marker="o", ms=6, lw=1.4, ls=style,
                             color=colour, mfc="white", alpha=0.85, label=name)
            max_abs = [max(e["max_abs_GPa"], 1e-3) for e in history[1:]]
            max_rel = [max(e["max_rel_percent"], 1e-3) for e in history[1:]]
            t_rest = times[1:]
            axes[0].plot(t_rest, max_abs, color="black", lw=2.6, marker="s",
                         ms=7, mfc="white", label="max", zorder=4)
            axes[1].plot(t_rest, max_rel, color="black", lw=2.6, marker="s",
                         ms=7, mfc="white", label="max", zorder=4)
            axes[0].axhline(tol_abs, ls="--", lw=2.0, color=SERIES_COLOURS[1],
                            label="tolerance", zorder=2)
            axes[1].axhline(tol_rel, ls="--", lw=2.0, color=SERIES_COLOURS[1],
                            label="tolerance", zorder=2)
            style_axis(axes[0], "Change since the previous check (GPa)",
                       xlabel="Production time per strain state (ps)")
            style_axis(axes[1], "Change since the previous check (%)",
                       xlabel="Production time per strain state (ps)")
            # A converged run leaves the drift far below the tolerance, so a
            # linear axis would squash the interesting part into the baseline.
            for ax in axes:
                ax.set_yscale("log")
                ax.minorticks_off()
                # Both panels carry the same series, so they share one legend
                # placed beside the figure - inside, it would cover the very
                # curves it labels.
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
            handles, legend_labels = axes[0].get_legend_handles_labels()
            fig.suptitle("%s - drift between consecutive checks at %.0f K "
                         "(below the dashed line = converged)"
                         % (basename, temperature_K), fontsize=17)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            if handles:
                fig.legend(handles, legend_labels,
                           loc="center left", bbox_to_anchor=(1.005, 0.5),
                           ncol=1 if len(handles) <= 12 else 2,
                           fontsize=12, frameon=False)
            written += save_publication_figure(
                fig, outdir, "%s_T%sK_convergence_drift" % (basename, tag),
                formats=formats)

    if final:
        print("  \U0001f4c1 Wrote the convergence history (%s and %d figures) "
              "into ./%s/"
              % (os.path.basename(csv_path), len(written) // len(formats), outdir))


def collect_states_sequential(hot_reference, calculator, temperature_K,
                              components, basename, job=None, checkpoint=None):
    """Every strain state run to the full production length, one after another."""
    deltas = strain_values()
    n_states = len(components) * len(deltas)
    print("\n=== Strained NVT runs at %.1f K: %d component(s) x %d strain(s) "
          "= %d runs ===" % (temperature_K, len(components), len(deltas), n_states))

    states = []
    state_index = 0
    for j in components:
        for d in deltas:
            state_index += 1
            state = None
            if checkpoint is not None:
                state = restore_state(checkpoint, temperature_K, j, d)
            if state is not None:
                print("\n  ♻️  [%d/%d] eps_%s = %+.4f at %.1f K - already done, "
                      "loaded from the checkpoint (%d samples, <T> = %.1f K)"
                      % (state_index, n_states, VOIGT_LABELS[j], d, temperature_K,
                         state["n_samples"], state["mean_temperature_K"]))
            else:
                state = run_strain_state(hot_reference, calculator, temperature_K,
                                         j, d, state_index, n_states, basename,
                                         job=job)
                if checkpoint is not None:
                    store_state(checkpoint, basename, temperature_K, state)
            states.append(state)
    return states


def collect_states_converged(hot_reference, calculator, temperature_K, symmetry,
                             components, basename, job=None, checkpoint=None):
    """Adaptive production: advance every strain state together and stop as soon
    as C_ij has stopped changing.

    Returns the finished states plus the convergence history.
    """
    deltas = strain_values()
    n_states = len(components) * len(deltas)
    n_prod_max = int(finite_t_params.get("nvt_production_steps", 1000))
    seg = min(convergence_segment_steps(), n_prod_max)
    log_interval = int(finite_t_params.get("log_interval", 200))
    timestep_fs = float(finite_t_params["timestep"])
    min_ps = float(finite_t_params.get("convergence_min_ps", 0.0))
    needed = max(1, int(finite_t_params.get("convergence_consecutive", 2)))
    tol_abs = float(finite_t_params.get("convergence_tol_GPa", 5.0))
    tol_rel = float(finite_t_params.get("convergence_tol_percent", 2.0))

    print("\n=== Strained NVT runs at %.1f K: %d component(s) x %d strain(s) "
          "= %d runs ===" % (temperature_K, len(components), len(deltas), n_states))
    print("  Adaptive production length: all %d strain states advance together in "
          "segments of\n  %d steps (%.2f ps); C_ij is re-fitted after every segment "
          "and the runs stop after\n  %d consecutive check(s) within %.2f GPa / "
          "%.2f %% (criterion: %s), never before\n  %.2f ps and never beyond "
          "%.2f ps of production per strain state."
          % (n_states, seg, seg * timestep_fs / 1000.0, needed, tol_abs, tol_rel,
             convergence_criterion_key(), min_ps,
             n_prod_max * timestep_fs / 1000.0))
    print("  Watching: " + ", ".join(
        "C%d%d" % (i + 1, k + 1) for (i, k) in tracked_components(symmetry))
        + (", K, G" if finite_t_params.get("convergence_include_moduli", True)
           else ""))

    runs = []
    index = 0
    for j in components:
        for d in deltas:
            index += 1
            restored = restore_run(checkpoint, temperature_K, j, d, calculator)
            if restored is not None:
                print("  ♻️  [%d/%d] eps_%s = %+.4f - %.2f ps of production "
                      "restored from the checkpoint%s"
                      % (index, n_states, VOIGT_LABELS[j], d,
                         restored["production_steps"] * timestep_fs / 1000.0,
                         "" if restored.get("complete") else " (unfinished)"))
            runs.append(StrainRun(hot_reference, calculator, temperature_K, j, d,
                                  index, n_states, basename, job=job,
                                  restored=restored))

    if runs and all(r.restored_complete and r.prod_steps > 0 for r in runs):
        done_steps = min(r.prod_steps for r in runs)
        print("  ♻️  Every strain state at this temperature is already finished "
              "(%.2f ps of production each) - nothing left to run."
              % (done_steps * timestep_fs / 1000.0))
        if job is not None:
            for _ in runs:
                job.complete_run()
        return ([r.state(complete=True) for r in runs],
                restore_history(checkpoint, temperature_K, done_steps))

    for run in runs:
        run.equilibrate(log_interval)
        if checkpoint is not None and run.prod_steps == 0:
            store_state(checkpoint, basename, temperature_K,
                        run.state(complete=False, with_snapshot=True), flush=False)
    if checkpoint is not None:
        save_checkpoint(basename, checkpoint)

    for run in runs:
        run.start_production(n_prod_max, log_interval)

    history = restore_history(checkpoint, temperature_K,
                              min(r.prod_steps for r in runs))
    if history:
        print("  ♻️  %d earlier convergence check(s) restored, the history "
              "continues from %.2f ps." % (len(history), history[-1]["production_ps"]))
    streak = 0
    for past in reversed(history):
        if not past.get("passed"):
            break
        streak += 1
    converged = False

    while True:
        common = min(r.prod_steps for r in runs)
        # Check first, then advance: a run resumed from the checkpoint may
        # already be sitting on production time that has never been checked.
        due = (common > 0
               and len(set(r.prod_steps for r in runs)) == 1
               and (not history or common > history[-1]["production_steps"]))
        if due:
            C, C_err, _cols, _warn = tensor_from_states(
                [r.state() for r in runs], components, symmetry)
            moduli = vrh_moduli(C)
            moduli_err = modulus_uncertainties(
                C, C_err, rng=np.random.RandomState(
                    int(finite_t_params.get("seed", 42)) + int(common)))
            quantities = tracked_quantities(C, C_err, symmetry, moduli=moduli,
                                            moduli_err=moduli_err)
            if history:
                passed, max_abs, max_rel, rows = convergence_verdict(
                    history[-1]["quantities"], quantities)
            else:
                passed, max_abs, max_rel, rows = (False, float("nan"),
                                                  float("nan"), [])

            entry = {
                "check": (history[-1]["check"] + 1) if history else 1,
                "production_steps": int(common),
                "production_ps": common * timestep_fs / 1000.0,
                "n_samples": int(min(len(r.samples_GPa) for r in runs)),
                "quantities": quantities,
                "rows": rows,
                "max_abs_GPa": max_abs,
                "max_rel_percent": max_rel,
                "passed": bool(passed),
                "moduli": {"K": moduli["K"], "G": moduli["G"],
                           "E": moduli["youngs_modulus"]},
                "moduli_err": {"K": moduli_err.get("K", 0.0),
                               "G": moduli_err.get("G", 0.0),
                               "E": moduli_err.get("E", 0.0)},
                "elastic_tensor_GPa": C.tolist(),
            }
            history.append(entry)

            enough_time = entry["production_ps"] >= min_ps - 1e-9
            streak = streak + 1 if passed else 0
            entry["converged"] = bool(passed and enough_time and streak >= needed)
            print_convergence_check(entry, tol_abs, tol_rel, needed, streak)
            if passed and not enough_time:
                print("      ⏸  holding: the minimum production time of %.2f ps "
                      "has not been reached yet." % min_ps)
            if job is not None:
                print("      %s" % job.summary(), flush=True)

            if checkpoint is not None:
                store_history(checkpoint, basename, temperature_K, history)
                save_checkpoint(basename, checkpoint)

            # Refresh the convergence CSV and figures now, so they can be
            # watched while the calculation is still running.
            try:
                write_convergence_outputs(basename, temperature_K, history,
                                          symmetry, final=False)
            except Exception as exc:
                print("      \u26a0\ufe0f  WARNING: could not refresh the convergence "
                      "figures: %s" % exc)

            if entry["converged"]:
                converged = True
                break

        if common >= n_prod_max:
            break

        target = min(common + seg, n_prod_max)
        for run in runs:
            run.advance_to(target)
            if checkpoint is not None:
                store_state(checkpoint, basename, temperature_K,
                            run.state(complete=False, with_snapshot=True))

    final_steps = min(r.prod_steps for r in runs)
    if converged:
        print("\n  ✅ Converged after %.2f ps of production per strain state "
              "(cap was %.2f ps)."
              % (final_steps * timestep_fs / 1000.0,
                 n_prod_max * timestep_fs / 1000.0))
    else:
        print("\n  ⚠️  The production cap of %.2f ps was reached before the "
              "convergence criterion was met - C_ij may still be drifting. "
              "Raise the NVT production steps (or relax the tolerance) if the "
              "convergence figures are not flat."
              % (n_prod_max * timestep_fs / 1000.0))

    states = []
    for run in runs:
        state = run.state(complete=True, with_snapshot=True)
        if checkpoint is not None:
            store_state(checkpoint, basename, temperature_K, state, flush=False)
        state.pop("snapshot", None)
        states.append(state)
        print("      eps_%s = %+.4f: <T> = %.1f K, %d samples, "
              "<sigma_%s> = %.3f +- %.3f GPa"
              % (VOIGT_LABELS[state["voigt_index"]], state["delta"],
                 state["mean_temperature_K"], state["n_samples"],
                 VOIGT_LABELS[state["voigt_index"]],
                 state["stress_mean_GPa"][state["voigt_index"]],
                 state["stress_sem_GPa"][state["voigt_index"]]))
    if checkpoint is not None:
        save_checkpoint(basename, checkpoint)

    if job is not None:
        unrun = sum(max(0, n_prod_max - r.prod_steps) for r in runs)
        for _ in runs:
            job.complete_run()
        if unrun > 0:
            job.drop_steps(unrun)
            print("  ⏱️  %.1f ps of planned MD is no longer needed at this "
                  "temperature." % (unrun * timestep_fs / 1000.0))

    try:
        write_convergence_outputs(basename, temperature_K, history, symmetry)
    except Exception as exc:
        print("  ⚠️  WARNING: could not write the convergence outputs: %s" % exc)

    return states, history


def store_history(checkpoint, basename, temperature_K, history):
    """Keep the convergence history in the checkpoint.

    Without this a restarted run would begin a fresh history and the figures
    would only show what happened after the restart.
    """
    if checkpoint is None:
        return
    entry = checkpoint["temperatures"].setdefault(
        temperature_key(temperature_K), {"states": {}})
    entry["convergence_history"] = [
        {"check": e["check"],
         "production_steps": e["production_steps"],
         "production_ps": e["production_ps"],
         "n_samples": e["n_samples"],
         "quantities": [[name, value, err] for name, value, err in e["quantities"]],
         "rows": e["rows"],
         "max_abs_GPa": e["max_abs_GPa"],
         "max_rel_percent": e["max_rel_percent"],
         "passed": bool(e["passed"]),
         "moduli": e["moduli"],
         "moduli_err": e.get("moduli_err", {}),
         "elastic_tensor_GPa": e["elastic_tensor_GPa"],
         "converged": bool(e.get("converged", False))}
        for e in history
    ]


def restore_history(checkpoint, temperature_K, up_to_steps):
    """The convergence history of an interrupted run, up to the steps that
    actually survived in the stored strain states."""
    if checkpoint is None:
        return []
    entry = checkpoint.get("temperatures", {}).get(temperature_key(temperature_K), {})
    stored = entry.get("convergence_history") or []
    history = []
    for e in stored:
        if int(e.get("production_steps", 0)) > int(up_to_steps):
            continue
        item = dict(e)
        item["quantities"] = [(str(n), float(v), float(err))
                              for n, v, err in e.get("quantities", [])]
        history.append(item)
    return history


def convergence_summary(history):
    """The convergence history in a form that goes cleanly into JSON."""
    if not history:
        return None
    return {
        "checks": [
            {"check": e["check"],
             "production_ps": e["production_ps"],
             "production_steps": e["production_steps"],
             "n_stress_samples": e["n_samples"],
             "max_change_GPa": (e["max_abs_GPa"]
                                if np.isfinite(e["max_abs_GPa"]) else None),
             "max_change_percent": (e["max_rel_percent"]
                                    if np.isfinite(e["max_rel_percent"]) else None),
             "within_tolerance": bool(e["passed"]),
             "bulk_modulus_GPa": e["moduli"]["K"],
             "bulk_modulus_err_GPa": e.get("moduli_err", {}).get("K", 0.0),
             "shear_modulus_GPa": e["moduli"]["G"],
             "shear_modulus_err_GPa": e.get("moduli_err", {}).get("G", 0.0),
             "youngs_modulus_GPa": e["moduli"]["E"],
             "youngs_modulus_err_GPa": e.get("moduli_err", {}).get("E", 0.0),
             "quantities_GPa": dict((n, v) for n, v, _ in e["quantities"])}
            for e in history
        ],
        "converged": bool(history[-1].get("converged", False)),
        "production_ps_used": history[-1]["production_ps"],
        "tolerance_GPa": float(finite_t_params.get("convergence_tol_GPa", 5.0)),
        "tolerance_percent": float(finite_t_params.get("convergence_tol_percent", 2.0)),
        "criterion": convergence_criterion_key(),
        "consecutive_required": int(finite_t_params.get("convergence_consecutive", 2)),
    }


def run_temperature(hot_reference, calculator, temperature_K, symmetry,
                    components, basename, job=None, checkpoint=None):
    """Full set of strained runs at one temperature -> C_ij and its errors.

    Strain states already present in the checkpoint are loaded instead of rerun.
    With the convergence check switched on the production part runs adaptively
    and stops as soon as the constants have settled.
    """
    convergence = None
    if convergence_enabled():
        states, history = collect_states_converged(
            hot_reference, calculator, temperature_K, symmetry, components,
            basename, job=job, checkpoint=checkpoint)
        convergence = convergence_summary(history)
    else:
        states = collect_states_sequential(
            hot_reference, calculator, temperature_K, components, basename,
            job=job, checkpoint=checkpoint)

    C, C_err, columns, residual_warning = tensor_from_states(
        states, components, symmetry)

    if symmetry == "triclinic":
        raw = np.zeros((6, 6))
        for j, col in columns.items():
            raw[:, j] = col["slope"]
        asymmetry = float(np.max(np.abs(raw - raw.T)))
        print("\n  Max asymmetry |C_ij - C_ji| before symmetrisation: %.2f GPa"
              % asymmetry)
        if asymmetry > 5.0:
            print("  ⚠️  WARNING: large asymmetry - the stress averages are not yet "
                  "converged (longer production runs or a bigger cell needed).")

    for msg in residual_warning:
        print("  ⚠️  WARNING: non-linear stress response - %s "
              "(consider a smaller strain magnitude)" % msg)

    return C, C_err, columns, states, convergence


def write_temperature_outputs(basename, temperature_K, states, analysis, C, C_err):
    """Per-temperature CSV / text output."""
    rows = []
    for s in states:
        row = {
            "eps_component": VOIGT_LABELS[s["voigt_index"]],
            "delta": s["delta"],
            "mean_temperature_K": s["mean_temperature_K"],
            "n_samples": s["n_samples"],
        }
        for i in range(6):
            row["sigma_%s_GPa" % VOIGT_LABELS[i]] = s["stress_mean_GPa"][i]
            row["sigma_%s_err_GPa" % VOIGT_LABELS[i]] = s["stress_sem_GPa"][i]
        rows.append(row)
    states_csv = os.path.join(
        temperature_dir(temperature_K),
        "%s_T%sK_strain_states.csv" % (basename, temperature_tag(temperature_K)))
    pd.DataFrame(rows).to_csv(states_csv, index=False)

    sample_rows = []
    for s in states:
        for k in range(len(s["times_ps"])):
            row = {"eps_component": VOIGT_LABELS[s["voigt_index"]],
                   "delta": s["delta"],
                   "time_ps": s["times_ps"][k]}
            for i in range(6):
                row["sigma_%s_GPa" % VOIGT_LABELS[i]] = s["samples_GPa"][k, i]
            sample_rows.append(row)
    samples_csv = os.path.join(
        temperature_dir(temperature_K),
        "%s_T%sK_stress_samples.csv" % (basename, temperature_tag(temperature_K)))
    pd.DataFrame(sample_rows).to_csv(samples_csv, index=False)

    txt = os.path.join(
        temperature_dir(temperature_K),
        "%s_T%sK_elastic_summary.txt" % (basename, temperature_tag(temperature_K)))
    with open(txt, "w") as fh:
        fh.write("Finite-temperature elastic constants\n")
        fh.write("Structure   : %s\n" % basename)
        fh.write("Temperature : %.1f K\n" % temperature_K)
        fh.write("Pressure    : %.3f GPa\n" % finite_t_params.get("pressure_GPa", 0.0))
        fh.write("Symmetry    : %s\n" % finite_t_params.get("symmetry", "triclinic"))
        fh.write("Method      : explicit stress-strain MD "
                 "(strains %s)\n\n" % ", ".join("%+.4f" % d for d in strain_values()))
        fh.write("Elastic tensor C_ij (GPa), errors from block averaging:\n")
        for i in range(6):
            fh.write("  " + " ".join("%9.2f+-%-7.2f" % (C[i, k], C_err[i, k])
                                     for k in range(6)) + "\n")
        fh.write("\n")
        errs = analysis.get("modulus_errors_GPa", {})
        fh.write("Bulk modulus  (Hill) : %.2f +- %.2f GPa\n"
                 % (analysis["bulk_modulus_GPa"]["hill"]
                    if analysis["bulk_modulus_GPa"]["hill"] is not None
                    else analysis["bulk_modulus_GPa"]["voigt"], errs.get("K", 0.0)))
        fh.write("Shear modulus (Hill) : %.2f +- %.2f GPa\n"
                 % (analysis["shear_modulus_GPa"]["hill"]
                    if analysis["shear_modulus_GPa"]["hill"] is not None
                    else analysis["shear_modulus_GPa"]["voigt"], errs.get("G", 0.0)))
        fh.write("Young's modulus      : %.2f +- %.2f GPa\n"
                 % (analysis["youngs_modulus_GPa"], errs.get("E", 0.0)))
        fh.write("Poisson ratio        : %.4f +- %.4f\n"
                 % (analysis["poisson_ratio"], errs.get("nu", 0.0)))
        fh.write("Density              : %.4f g/cm^3\n" % analysis["density_g_cm3"])
        fh.write("Volume               : %.4f A^3 (%.4f A^3/atom)\n"
                 % (analysis["volume_A3"], analysis["volume_per_atom_A3"]))
        fh.write("Debye temperature    : %.1f K\n" % analysis["debye_temperature_K"])
        fh.write("Mechanically stable  : %s\n"
                 % analysis["mechanical_stability"].get("mechanically_stable"))
    print("  \U0001f4c1 Wrote %s, %s and %s into ./%s/"
          % (os.path.basename(states_csv), os.path.basename(samples_csv),
             os.path.basename(txt), temperature_dir(temperature_K)))


def panel_grid(n, panel_w=7.4, panel_h=5.4):
    """A readable grid of n panels: at most two columns, so six strain
    components come out as 3 rows x 2 columns rather than one squashed row."""
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(float(n) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows),
                             squeeze=False)
    flat = axes.ravel()
    for spare in flat[n:]:
        spare.axis("off")
    return fig, flat[:n]


def weighted_line(x, y, yerr):
    """Slope *and* intercept of the same weighted fit that produces C_ij,
    so the line drawn through the points is the fit that was actually used."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0
    w = 1.0 / yerr ** 2 if np.all(yerr > 0) else np.ones_like(x)
    S = w.sum()
    Sx = (w * x).sum()
    Sxx = (w * x * x).sum()
    Sy = (w * y).sum()
    Sxy = (w * x * y).sum()
    denom = S * Sxx - Sx * Sx
    if abs(denom) < 1e-30:
        return 0.0, float(np.mean(y))
    return (float((S * Sxy - Sx * Sy) / denom),
            float((Sxx * Sy - Sx * Sxy) / denom))


def plot_temperature(basename, temperature_K, columns_states, C):
    """Per-temperature diagnostics: the stress-strain fits that give C_ij, and
    the running stress averages that show whether they are converged."""
    components = sorted(columns_states.keys())
    n = len(components)
    tag = temperature_tag(temperature_K)
    outdir = temperature_dir(temperature_K)
    written = []

    with plt.rc_context(PUB_RC):
        # ---- stress vs strain, with the fit whose slope is C_ij ------------
        fig, axes = panel_grid(n)
        for ax, j in zip(axes, components):
            states = columns_states[j]
            deltas = np.array([s["delta"] for s in states])
            x = deltas * 100.0
            xline = np.linspace(x.min(), x.max(), 50)
            for i in range(6):
                y = np.array([s["stress_mean_GPa"][i] for s in states])
                e = np.array([s["stress_sem_GPa"][i] for s in states])
                conjugate = (i == j)
                colour = SERIES_COLOURS[i % len(SERIES_COLOURS)]
                slope, intercept = weighted_line(deltas, y, e)
                ax.plot(xline, intercept + slope * xline / 100.0, ls="-",
                        lw=2.4 if conjugate else 1.1, color=colour,
                        alpha=0.95 if conjugate else 0.45, zorder=2)
                ax.errorbar(x, y, yerr=e, marker="o", ls="none", color=colour,
                            mfc="white", mew=2.0 if conjugate else 1.4,
                            ms=10 if conjugate else 7,
                            alpha=1.0 if conjugate else 0.55,
                            label=r"$\sigma_{%s}$%s" % (
                                VOIGT_LABELS[i],
                                r"  (slope = $C_{%d%d}$)" % (i + 1, j + 1)
                                if conjugate else ""),
                            zorder=3)
            ax.set_xlabel(r"Applied strain $\varepsilon_{%s}$ (%%)" % VOIGT_LABELS[j])
            ax.set_ylabel(r"$\langle\sigma\rangle$ (GPa)")
            ax.set_title(r"$\varepsilon_{%s}$" % VOIGT_LABELS[j])
            ax.axhline(0.0, lw=0.9, color="0.6", ls=":", zorder=0)
            ax.axvline(0.0, lw=0.9, color="0.6", ls=":", zorder=0)
            ax.grid(True, alpha=0.2, ls=":", lw=0.9)
            thin_ticks(ax, nbins=5)
            ax.legend(ncol=2, fontsize=11, columnspacing=1.0, handletextpad=0.4)
        fig.suptitle("%s - time-averaged stress vs applied strain at %.0f K"
                     % (basename, temperature_K), fontsize=19)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        written += save_publication_figure(
            fig, outdir, "%s_T%sK_stress_strain" % (basename, tag))

        # ---- running stress average: has the production window converged? ---
        fig, axes = panel_grid(n, panel_h=5.0)
        for ax, j in zip(axes, components):
            for k, st in enumerate(columns_states[j]):
                series = st["samples_GPa"][:, j] if len(st["samples_GPa"]) else []
                if len(series) == 0:
                    continue
                running = np.cumsum(series) / np.arange(1, len(series) + 1)
                colour = SERIES_COLOURS[k % len(SERIES_COLOURS)]
                ax.plot(st["times_ps"], running, lw=2.2, color=colour,
                        label=r"$\varepsilon = %+.4f$" % st["delta"], zorder=3)
                ax.axhline(running[-1], ls=":", lw=1.2, color=colour,
                           alpha=0.7, zorder=1)
            ax.set_xlabel("Simulation time (ps)")
            ax.set_ylabel(r"Running $\langle\sigma_{%s}\rangle$ (GPa)"
                          % VOIGT_LABELS[j])
            ax.set_title(r"$\varepsilon_{%s}$" % VOIGT_LABELS[j])
            ax.grid(True, alpha=0.2, ls=":", lw=0.9)
            thin_ticks(ax, nbins=5)
            ax.legend(fontsize=12)
        fig.suptitle("%s - convergence of the stress average at %.0f K "
                     "(flat = production window long enough)"
                     % (basename, temperature_K), fontsize=19)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        written += save_publication_figure(
            fig, outdir, "%s_T%sK_stress_convergence" % (basename, tag))

    return written


# Publication-ready figure styling shared by every figure this script writes:
# large type, inward ticks on all four sides, colour-blind-safe series colours,
# no chart junk. Applied through a context manager rather than globally.
PUB_RC = {
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.major.size": 6.5,
    "ytick.major.size": 6.5,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    "lines.linewidth": 2.2,
    "lines.markersize": 9,
    "errorbar.capsize": 4,
    "legend.frameon": False,
    "savefig.bbox": "tight",
}

SERIES_COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
STATIC_COLOUR = "#444444"


def save_publication_figure(fig, outdir, stem, formats=("png", "pdf")):
    """Write one figure as a 300 dpi PNG and as a vector PDF.

    The convergence figures are refreshed after every check while the MD is
    still running, and ask for the PNG only; the vector copy is written once
    the temperature is finished.
    """
    os.makedirs(outdir, exist_ok=True)
    written = []
    for ext in formats:
        path = os.path.join(outdir, "%s.%s" % (stem, ext))
        fig.savefig(path, dpi=300)
        written.append(path)
    plt.close(fig)
    return written


def thin_ticks(ax, nbins=6):
    """A handful of labelled ticks per axis, and no minor ticks at all."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.minorticks_off()
    ax.tick_params(which="both", direction="in", top=True, right=True)


def style_axis(ax, ylabel, xlabel="Temperature (K)"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, ls=":", lw=0.9)
    thin_ticks(ax)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def plot_quantity_vs_T(temps, values, errors, ylabel, colour, label,
                       static_value=None, static_label="0 K static"):
    """One quantity against temperature, anchored by its 0 K static value."""
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.errorbar(temps, values, yerr=errors, marker="o", color=colour,
                mfc="white", mew=2.2, label=label, zorder=3)
    if static_value is not None and np.isfinite(static_value):
        ax.axhline(static_value, ls=":", lw=1.8, color=STATIC_COLOUR,
                   alpha=0.9, label=static_label, zorder=1)
    style_axis(ax, ylabel)
    return fig, ax


def plot_temperature_dependence(basename, sweep, C_static, static_analysis=None):
    """One publication-ready figure per quantity, written to summary_plots/."""
    if len(sweep) < 2:
        return []

    outdir = os.path.join(RESULT_DIR, "summary_plots")
    temps = np.array([entry["temperature_K"] for entry in sweep], dtype=float)
    ana = [entry["analysis"] for entry in sweep]
    C_all = np.array([a["elastic_tensor_GPa"] for a in ana])
    C_err_all = np.array([a["elastic_tensor_error_GPa"] for a in ana])
    written = []

    def modulus(a, key):
        if key == "K":
            return a["bulk_modulus_GPa"]["hill"] or a["bulk_modulus_GPa"]["voigt"]
        if key == "G":
            return a["shear_modulus_GPa"]["hill"] or a["shear_modulus_GPa"]["voigt"]
        if key == "E":
            return a["youngs_modulus_GPa"]
        return a["poisson_ratio"]

    def modulus_err(a, key):
        return a.get("modulus_errors_GPa", {}).get(key, 0.0)

    def static_of(key):
        if static_analysis is None:
            return None
        return modulus(static_analysis, key)

    with plt.rc_context(PUB_RC):
        # ---- 1) the elastic constants themselves --------------------------
        fig, ax = plt.subplots(figsize=(7.8, 5.8))
        plotted = []
        idx = 0
        for label, (i, j) in [("$C_{11}$", (0, 0)), ("$C_{12}$", (0, 1)),
                              ("$C_{13}$", (0, 2)), ("$C_{33}$", (2, 2)),
                              ("$C_{44}$", (3, 3)), ("$C_{66}$", (5, 5))]:
            values = C_all[:, i, j]
            # Skip constants that are zero or that symmetry made identical to
            # one already drawn (e.g. C13 == C12 for a cubic crystal).
            if np.allclose(values, 0.0):
                continue
            if any(np.allclose(values, done, atol=1e-6) for done in plotted):
                continue
            plotted.append(values)
            colour = SERIES_COLOURS[idx % len(SERIES_COLOURS)]
            idx += 1
            ax.errorbar(temps, values, yerr=C_err_all[:, i, j], marker="o",
                        color=colour, mfc="white", mew=2.0, label=label, zorder=3)
            if C_static is not None and abs(C_static[i, j]) > 1e-6:
                ax.axhline(C_static[i, j], ls=":", lw=1.6, color=colour,
                           alpha=0.8, zorder=1)
        if C_static is not None:
            ax.plot([], [], ls=":", lw=1.6, color=STATIC_COLOUR,
                    label="0 K static")
        style_axis(ax, "Elastic constant (GPa)")
        ax.legend(ncol=2)
        written += save_publication_figure(fig, outdir,
                                           "%s_elastic_constants_vs_T" % basename)

        # ---- 2-4) bulk / shear / Young, one figure each --------------------
        for key, ylabel, colour, stem in [
                ("K", "Bulk modulus $K$ (GPa)", SERIES_COLOURS[0], "bulk_modulus"),
                ("G", "Shear modulus $G$ (GPa)", SERIES_COLOURS[1], "shear_modulus"),
                ("E", "Young's modulus $E$ (GPa)", SERIES_COLOURS[2], "youngs_modulus")]:
            fig, _ = plot_quantity_vs_T(
                temps, [modulus(a, key) for a in ana],
                [modulus_err(a, key) for a in ana],
                ylabel, colour, "MD (Voigt-Reuss-Hill)",
                static_value=static_of(key))
            written += save_publication_figure(
                fig, outdir, "%s_%s_vs_T" % (basename, stem))

        # ---- 5) Poisson ratio ---------------------------------------------
        fig, _ = plot_quantity_vs_T(
            temps, [a["poisson_ratio"] for a in ana],
            [modulus_err(a, "nu") for a in ana],
            r"Poisson ratio $\nu$", SERIES_COLOURS[3], "MD",
            static_value=static_of("nu"))
        written += save_publication_figure(fig, outdir,
                                           "%s_poisson_ratio_vs_T" % basename)

        # ---- 6) Debye temperature ------------------------------------------
        fig, _ = plot_quantity_vs_T(
            temps, [a["debye_temperature_K"] for a in ana],
            [a.get("debye_temperature_error_K", 0.0) for a in ana],
            r"Debye temperature $\Theta_\mathrm{D}$ (K)", SERIES_COLOURS[4], "MD",
            static_value=(static_analysis["debye_temperature_K"]
                          if static_analysis else None))
        written += save_publication_figure(fig, outdir,
                                           "%s_debye_temperature_vs_T" % basename)

        # ---- 7) thermal expansion of the reference cell ---------------------
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        ax.plot(temps, [a["volume_per_atom_A3"] for a in ana], marker="o",
                color=SERIES_COLOURS[0], mfc="white", mew=2.2,
                label="MD (NPT reference cell)", zorder=3)
        if static_analysis is not None:
            v0 = static_analysis["volume_per_atom_A3"]
            ax.axhline(v0, ls=":", lw=1.8, color=STATIC_COLOUR, alpha=0.9,
                       label="0 K relaxed", zorder=1)
        style_axis(ax, r"Volume per atom ($\mathrm{\AA}^3$)")
        written += save_publication_figure(fig, outdir,
                                           "%s_thermal_expansion_vs_T" % basename)

        # ---- 8) softening relative to the 0 K static tensor ------------------
        if C_static is not None:
            fig, ax = plt.subplots(figsize=(7.2, 5.4))
            idx = 0
            for label, (i, j) in [("$C_{11}$", (0, 0)), ("$C_{12}$", (0, 1)),
                                  ("$C_{44}$", (3, 3))]:
                if abs(C_static[i, j]) < 1e-6 or np.allclose(C_all[:, i, j], 0.0):
                    continue
                rel = 100.0 * (C_all[:, i, j] - C_static[i, j]) / C_static[i, j]
                rel_err = 100.0 * C_err_all[:, i, j] / abs(C_static[i, j])
                ax.errorbar(temps, rel, yerr=rel_err, marker="o",
                            color=SERIES_COLOURS[idx % len(SERIES_COLOURS)],
                            mfc="white", mew=2.0, label=label, zorder=3)
                idx += 1
            ax.axhline(0.0, ls="-", lw=1.2, color=STATIC_COLOUR, alpha=0.6, zorder=1)
            style_axis(ax, "Change relative to 0 K (%)")
            written += save_publication_figure(
                fig, outdir, "%s_thermal_softening_vs_T" % basename)

    print("  \U0001f4c1 Wrote %d summary figures (PNG + PDF) into ./%s/"
          % (len(written) // 2, outdir))
    return written


def write_sweep_outputs(basename, sweep, C_static):
    rows = []
    for entry in sweep:
        a = entry["analysis"]
        C = np.array(a["elastic_tensor_GPa"])
        C_err = np.array(a["elastic_tensor_error_GPa"])
        errs = a.get("modulus_errors_GPa", {})
        row = {
            "temperature_K": entry["temperature_K"],
            "volume_A3": a["volume_A3"],
            "density_g_cm3": a["density_g_cm3"],
            "bulk_modulus_hill_GPa": (a["bulk_modulus_GPa"]["hill"]
                                      or a["bulk_modulus_GPa"]["voigt"]),
            "bulk_modulus_err_GPa": errs.get("K", 0.0),
            "shear_modulus_hill_GPa": (a["shear_modulus_GPa"]["hill"]
                                       or a["shear_modulus_GPa"]["voigt"]),
            "shear_modulus_err_GPa": errs.get("G", 0.0),
            "youngs_modulus_GPa": a["youngs_modulus_GPa"],
            "youngs_modulus_err_GPa": errs.get("E", 0.0),
            "poisson_ratio": a["poisson_ratio"],
            "poisson_ratio_err": errs.get("nu", 0.0),
            "debye_temperature_K": a["debye_temperature_K"],
            "debye_temperature_err_K": a.get("debye_temperature_error_K", 0.0),
            "mechanically_stable": a["mechanical_stability"].get("mechanically_stable"),
        }
        conv = entry.get("convergence")
        if conv:
            row["production_ps_used"] = conv["production_ps_used"]
            row["cij_converged"] = bool(conv["converged"])
        for i in range(6):
            for k in range(i, 6):
                row["C%d%d_GPa" % (i + 1, k + 1)] = C[i, k]
                row["C%d%d_err_GPa" % (i + 1, k + 1)] = C_err[i, k]
        rows.append(row)

    csv_path = os.path.join(RESULT_DIR, "%s_elastic_vs_temperature.csv" % basename)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    payload = {
        "structure": basename,
        "settings": finite_t_params,
        "static_0K_tensor_GPa": (C_static.tolist() if C_static is not None else None),
        "static_0K_ion_relaxed": bool(finite_t_params.get("static_ion_relax", True)),
        "measured_quantity": ("d<sigma_i>/d eps_j (Birch / stress-strain "
                              "coefficients; identical to C_ij at zero pressure)"),
        "pressure_GPa": float(finite_t_params.get("pressure_GPa", 0.0)),
        "temperatures": [
            {"temperature_K": entry["temperature_K"],
             "npt": entry["npt_info"],
             "convergence": entry.get("convergence"),
             "results": entry["analysis"]}
            for entry in sweep
        ],
    }
    json_path = os.path.join(RESULT_DIR, "%s_finite_T_elastic.json" % basename)
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print("\n\U0001f4c1 Wrote %s and %s" % (os.path.basename(csv_path),
                                 os.path.basename(json_path)))
'''


_MAIN_BODY = r'''
if 'calculator' not in locals() or calculator is None:
    print("Calculator could not be initialized. Exiting.")
    return

print("\nSearching for structure files (*.cif, *.vasp, *.poscar, POSCAR*)...")
structure_files = (glob.glob("*.cif") + glob.glob("*.vasp")
                   + glob.glob("*.poscar") + glob.glob("*.POSCAR")
                   + glob.glob("POSCAR*"))
structure_files = list(dict.fromkeys(structure_files))
if not structure_files:
    print("No structure files found. Place a .cif, .vasp/.poscar or POSCAR "
          "file next to this script.")
    return
if len(structure_files) > 1:
    print("Warning: multiple structure files found; using the first one: %s"
          % structure_files[0])

path = structure_files[0]
basename = os.path.splitext(os.path.basename(path))[0]
os.makedirs(RESULT_DIR, exist_ok=True)

print("\n--- Reading structure from %s ---" % path)
try:
    if path.lower().endswith(".poscar"):
        atoms = read(path, format="vasp")
    else:
        atoms = read(path)
except Exception as read_err:
    if (path.lower().endswith((".poscar", ".vasp"))
            or os.path.basename(path).upper().startswith("POSCAR")):
        print("  Could not read '%s' as a VASP/POSCAR file: %s" % (path, read_err))
        print("  A common cause is placeholder species names such as "
              "'Type_1 Type_2' instead of real element symbols.")
    raise

atoms.set_pbc(True)
rep = finite_t_params.get("supercell", [1, 1, 1])
if any(int(r) > 1 for r in rep):
    atoms = make_supercell(atoms, np.diag([int(r) for r in rep]))
    print("  Supercell %dx%dx%d applied" % (int(rep[0]), int(rep[1]), int(rep[2])))
print("  %s: %d atoms, V = %.3f A^3"
      % (atoms.get_chemical_formula(), len(atoms), atoms.get_volume()))
if len(atoms) < 32:
    print("  \u26a0\ufe0f  WARNING: only %d atoms - stress fluctuations scale as 1/sqrt(N), so "
          "the error bars will be large. Consider a bigger supercell." % len(atoms))

atoms.calc = calculator
print("  ... warming up the calculator ...")
t_warm = time.perf_counter()
_ = atoms.get_potential_energy()
_ = atoms.get_forces()
_ = atoms.get_stress()
print("  ... warmed up in %.2fs" % (time.perf_counter() - t_warm))

symmetry = finite_t_params.get("symmetry", "triclinic")
components = finite_t_params.get(
    "strain_components", SYMMETRY_STRAIN_COMPONENTS[symmetry])
components = [int(j) for j in components]

print("\n--- Checkpoint ---")
signature = settings_signature(atoms)
checkpoint = load_checkpoint(basename, signature)
if not checkpoint["temperatures"] and finite_t_params.get(
        "resume_from_checkpoint", True):
    print("  No previous results to reuse - starting from the beginning.")
print("  Progress is saved after every strained run, so an interrupted "
      "calculation\n  can be continued simply by starting this script again.")

relaxed_file = os.path.join(RESULT_DIR, "%s_relaxed_0K.xyz" % basename)
relaxed_npz = os.path.join(CHECKPOINT_DIR, "%s_relaxed_0K.npz" % basename)
if checkpoint.get("relaxed") and os.path.exists(relaxed_npz):
    relaxed = load_atoms_npz(relaxed_npz)
    relaxed.set_pbc(True)
    relaxed.calc = calculator
    print("\n--- 0 K relaxation: already done, loaded from the checkpoint ---")
else:
    atoms = relax_structure(atoms)
    relaxed = atoms.copy()
    write(relaxed_file, relaxed, format="extxyz")
    save_atoms_npz(relaxed_npz, relaxed)
    checkpoint["relaxed"] = os.path.basename(relaxed_npz)
    save_checkpoint(basename, checkpoint)

C_static = None
if finite_t_params.get("compute_static_reference", True):
    _want_relaxed = bool(finite_t_params.get("static_ion_relax", True))
    if (checkpoint.get("static_tensor_GPa") is not None
            and checkpoint.get("static_ion_relax") == _want_relaxed):
        C_static = np.array(checkpoint["static_tensor_GPa"], dtype=float)
        print("\n--- 0 K (static) reference tensor: loaded from the checkpoint ---")
        print_tensor(C_static, title="Static (0 K) C_ij (GPa)")
    else:
        if checkpoint.get("static_tensor_GPa") is not None:
            print("\n  The 0 K reference was stored with a different ionic-relaxation "
                  "setting - recomputing it.")
        try:
            C_static = static_elastic_tensor(relaxed, calculator, symmetry,
                                             components)
            checkpoint["static_tensor_GPa"] = C_static.tolist()
            checkpoint["static_ion_relax"] = _want_relaxed
            save_checkpoint(basename, checkpoint)
        except Exception as exc:
            print("  \u26a0\ufe0f  WARNING: static reference tensor failed: %s" % exc)
            C_static = None

static_analysis = None
if C_static is not None:
    try:
        # No MD errors here, so no Monte-Carlo pass: this is the 0 K anchor the
        # summary figures put at T = 0.
        static_analysis = analyze_tensor(C_static, None, relaxed, n_mc=0)
    except Exception as exc:
        print("  \u26a0\ufe0f  WARNING: could not analyse the 0 K tensor: %s" % exc)

temperatures = [float(t) for t in (finite_t_params.get("temperature_list") or [300.0])]
sweep = []
overall_start = time.perf_counter()

# Plan the whole workload up front so the remaining time can account for the
# strain states and temperatures that have not been started yet.
n_states_per_T = len(components) * len(strain_values())
nvt_steps_per_state = (int(finite_t_params.get("nvt_equilibration_steps", 0))
                       + int(finite_t_params.get("nvt_production_steps", 0)))
npt_steps_per_T = 0
if finite_t_params.get("use_npt_equilibration", True):
    npt_steps_per_T = (int(finite_t_params.get("npt_equilibration_steps", 0))
                       + int(finite_t_params.get("npt_production_steps", 0)))
runs_per_T = n_states_per_T + (1 if npt_steps_per_T else 0)
steps_per_T = n_states_per_T * nvt_steps_per_state + npt_steps_per_T
total_runs = runs_per_T * len(temperatures)
total_steps = steps_per_T * len(temperatures)

# Work already in the checkpoint is not going to be repeated, so leave it out of
# the plan - otherwise the remaining-time estimate would be far too pessimistic.
_nvt_eq_steps = int(finite_t_params.get("nvt_equilibration_steps", 0))
_nvt_prod_steps = int(finite_t_params.get("nvt_production_steps", 0))
skipped_runs = 0
skipped_steps = 0
for _T in temperatures:
    _entry = checkpoint.get("temperatures", {}).get(temperature_key(_T), {})
    if npt_steps_per_T and _entry.get("ref_cell"):
        skipped_runs += 1
        skipped_steps += npt_steps_per_T
    for _record in _entry.get("states", {}).values():
        _done = int(_record.get("production_steps", _nvt_prod_steps))
        _complete = bool(_record.get("complete", True))
        _finished = _complete and _done >= _nvt_prod_steps
        # A half-finished run only saves work if there is a snapshot to
        # continue it from; without one it has to start over.
        if not _finished and not _record.get("snapshot_file"):
            continue
        if _finished:
            skipped_runs += 1
        skipped_steps += min(nvt_steps_per_state, _nvt_eq_steps + _done)

remaining_runs = max(1, total_runs - skipped_runs)
remaining_steps = max(1, total_steps - skipped_steps)

print("\n--- Workload ---")
print("  %d temperature(s) x %d strain state(s) = %d MD runs, %d MD steps in total"
      % (len(temperatures), n_states_per_T, total_runs, total_steps))
if skipped_runs:
    print("  %d run(s) / %d step(s) already done and will be reused - "
          "%d run(s) / %d step(s) left to compute"
          % (skipped_runs, skipped_steps, total_runs - skipped_runs,
             total_steps - skipped_steps))
print("  %.1f ps of simulated time still to run"
      % (remaining_steps * float(finite_t_params["timestep"]) / 1000.0))
if convergence_enabled():
    print("  The production part is adaptive: this is the upper bound, the runs "
          "stop\n  as soon as C_ij is converged (every %.2f ps, tolerance "
          "%.2f GPa / %.2f %%)."
          % (float(finite_t_params.get("convergence_interval_ps", 1.0)),
             float(finite_t_params.get("convergence_tol_GPa", 5.0)),
             float(finite_t_params.get("convergence_tol_percent", 2.0))))
job = JobClock(remaining_steps, remaining_runs)
job.start()

for temperature_K in temperatures:
    print("\n" + "=" * 78)
    print("TEMPERATURE %.1f K   (%s)"
          % (temperature_K, job.summary()))
    print("=" * 78)

    if finite_t_params.get("use_npt_equilibration", True):
        restored = restore_npt(checkpoint, temperature_K, basename, calculator)
        if restored is not None:
            ref_cell, hot_reference, npt_info = restored
            print("\n--- NPT pre-equilibration at %.1f K: already done, "
                  "reference cell loaded from the checkpoint ---" % temperature_K)
        else:
            ref_cell, hot_reference, npt_info, reference_npz = npt_pre_equilibration(
                relaxed, calculator, temperature_K, basename, job=job)
            entry = checkpoint["temperatures"].setdefault(
                temperature_key(temperature_K), {"states": {}})
            entry["ref_cell"] = np.asarray(ref_cell, dtype=float).tolist()
            entry["npt_info"] = npt_info
            entry["reference_npz"] = reference_npz
            save_checkpoint(basename, checkpoint)
    else:
        print("\n--- NPT pre-equilibration skipped: straining the input cell ---")
        hot_reference = relaxed.copy()
        hot_reference.calc = calculator
        ref_cell = np.array(hot_reference.get_cell()[:], dtype=float)
        npt_info = {"skipped": True}

    C, C_err, _columns, states, convergence = run_temperature(
        hot_reference, calculator, temperature_K, symmetry, components, basename,
        job=job, checkpoint=checkpoint)

    print("")
    _P = float(finite_t_params.get("pressure_GPa", 0.0))
    print_tensor(C, C_err, title="C_ij(%.0f K) in GPa%s"
                 % (temperature_K,
                    "   [B_ij = d<sigma>/d eps, at %.3f GPa]" % _P
                    if abs(_P) > 1e-9 else ""))
    if abs(_P) > 1e-9:
        # At finite pressure the measured stress-strain coefficients are not the
        # tabulated elastic constants; show both rather than quietly conflating.
        print_tensor(pressure_corrected_tensor(C, _P), C_err,
                     title="C_ij(%.0f K) in GPa, corrected to the free-energy "
                           "convention (B + P terms)" % temperature_K)

    reference_atoms = hot_reference.copy()
    reference_atoms.set_cell(ref_cell, scale_atoms=True)
    analysis = analyze_tensor(
        C, C_err, reference_atoms,
        rng=np.random.RandomState(int(finite_t_params.get("seed", 42))))
    analysis["temperature_K"] = temperature_K
    errs = analysis.get("modulus_errors_GPa", {})
    K = (analysis["bulk_modulus_GPa"]["hill"]
         or analysis["bulk_modulus_GPa"]["voigt"])
    G = (analysis["shear_modulus_GPa"]["hill"]
         or analysis["shear_modulus_GPa"]["voigt"])
    print("\n  Bulk modulus  (Hill): %8.2f +- %.2f GPa" % (K, errs.get("K", 0.0)))
    print("  Shear modulus (Hill): %8.2f +- %.2f GPa" % (G, errs.get("G", 0.0)))
    print("  Young's modulus     : %8.2f +- %.2f GPa"
          % (analysis["youngs_modulus_GPa"], errs.get("E", 0.0)))
    print("  Poisson ratio       : %8.4f +- %.4f"
          % (analysis["poisson_ratio"], errs.get("nu", 0.0)))
    print("  Density             : %8.4f g/cm^3" % analysis["density_g_cm3"])
    print("  Debye temperature   : %8.1f K" % analysis["debye_temperature_K"])
    print("  Mechanically stable : %s"
          % analysis["mechanical_stability"].get("mechanically_stable"))
    if C_static is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            softening = np.where(np.abs(C_static) > 1e-6,
                                 100.0 * (C - C_static) / C_static, np.nan)
        print("  Change vs 0 K: C11 %+.1f%%, C12 %+.1f%%, C44 %+.1f%%"
              % (softening[0, 0], softening[0, 1], softening[3, 3]))

    write_temperature_outputs(basename, temperature_K, states, analysis, C, C_err)
    columns_states = {}
    for state in states:
        columns_states.setdefault(state["voigt_index"], []).append(state)
    try:
        plot_temperature(basename, temperature_K, columns_states, C)
    except Exception as exc:
        print("  \u26a0\ufe0f  WARNING: plotting failed for %.1f K: %s" % (temperature_K, exc))

    sweep.append({"temperature_K": temperature_K,
                  "analysis": analysis,
                  "npt_info": npt_info,
                  "convergence": convergence})

    remaining_T = len(temperatures) - len(sweep)
    if remaining_T > 0:
        print("\n  \u2705 Finished %.1f K. %d temperature(s) still to go  |  %s"
              % (temperature_K, remaining_T, job.summary()))

if sweep:
    write_sweep_outputs(basename, sweep, C_static)
    try:
        plot_temperature_dependence(basename, sweep, C_static, static_analysis)
    except Exception as exc:
        print("\u26a0\ufe0f  WARNING: temperature-dependence plot failed: %s" % exc)

print("\n\u2705 All done in %s - everything is under ./%s/"
      % (format_time(time.perf_counter() - overall_start), RESULT_DIR))
'''


def generate_finite_t_elastic_python_script(finite_t_params, selected_model,
                                            model_size, device, dtype,
                                            thread_count,
                                            custom_sevennet_path=None,
                                            custom_grace_path=None,
                                            custom_mace_path=None,
                                            mace_enable_cueq=False,
                                            sevennet_enable_cueq=False,
                                            mace_head=None,
                                            mace_dispersion=False,
                                            mace_dispersion_xc="pbe",
                                            nequip_accel=None):
    """Return the full source of the standalone finite-T elastic script."""
    imports_str = _base_imports(thread_count)

    model_imports, calculator_setup_str = build_calculator_code(
        selected_model, model_size, device, dtype,
        custom_sevennet_path=custom_sevennet_path,
        custom_grace_path=custom_grace_path,
        custom_mace_path=custom_mace_path,
        mace_enable_cueq=mace_enable_cueq,
        sevennet_enable_cueq=sevennet_enable_cueq,
        mace_head=mace_head,
        mace_dispersion=mace_dispersion,
        mace_dispersion_xc=mace_dispersion_xc,
        nequip_accel=nequip_accel,
    )
    imports_str += model_imports

    params = dict(finite_t_params)
    params.setdefault('symmetry', 'triclinic')
    params.setdefault('strain_components', [0, 1, 2, 3, 4, 5])
    params.setdefault('temperature_list', [300.0])
    params_str = pprint.pformat(params, indent=4, width=88, sort_dicts=True)

    header = (
        '"""\n'
        'Standalone finite-temperature elastic constants (explicit stress-strain MD).\n'
        'Generated by the uMLIP-Interactive Streamlit app.\n\n'
        'Method\n'
        '------\n'
        '  1. optional 0 K relaxation of the input structure,\n'
        '  2. optional NPT pre-equilibration at (T, P): the time-averaged cell of the\n'
        '     production part becomes the reference cell, so thermal expansion is\n'
        '     included and the reference stress is ~0,\n'
        '  3. every required Voigt component is strained affinely by +-delta and an\n'
        '     NVT run at T is performed for each strained cell,\n'
        '  4. the thermodynamic stress (virial + kinetic term) is time-averaged over\n'
        '     the production window, with block-averaged error bars,\n'
        '  5. C_ij(T) = d<sigma_i>/d eps_j from a (weighted) linear fit, followed by\n'
        '     Voigt-Reuss-Hill moduli, Born stability and the Debye temperature.\n\n'
        'These are the isothermal elastic constants at temperature T.\n\n'
        + ('Adaptive production length: all strain states of one temperature advance\n'
           'together in segments, C_ij is re-fitted from the trajectory so far after\n'
           'every segment and the runs stop once the change stays inside the tolerance.\n'
           'The history (CSV + figures) is written to\n'
           'elastic_md_results/T_<T>K/convergence/.\n\n'
           if finite_t_params.get('use_convergence_check') else '')
        + '--- SETTINGS ---\n'
        f'MLIP Model : {selected_model}\n'
        f'Model Key  : {model_size}\n'
        f'Device     : {device}\n'
        f'Precision  : {dtype}\n'
        f'CPU Threads: {thread_count}\n'
        'Parameters :\n'
        f'{textwrap.indent(params_str, "  ")}\n'
        '---\n'
        '"""\n'
    )

    main_str = ("def main():\n"
                + textwrap.indent(calculator_setup_str, "    ")
                + "\n"
                + textwrap.indent(_MAIN_BODY, "    ")
                + "\n\n"
                + 'if __name__ == "__main__":\n    main()\n')

    script = (header
              + imports_str
              + "\n\nfinite_t_params = "
              + params_str
              + "\n"
              + _SCRIPT_BODY
              + "\n\n"
              + main_str)
    return script.strip() + "\n"
