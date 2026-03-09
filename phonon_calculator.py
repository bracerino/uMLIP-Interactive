from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS, FIRE

logger = logging.getLogger(__name__)


try:
    import phonopy
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    import phonopy.units as phonopy_units
    PHONOPY_AVAILABLE = True
    PHONOPY_VERSION = tuple(int(x) for x in phonopy.__version__.split(".")[:2])
except ImportError:
    PHONOPY_AVAILABLE = False
    PHONOPY_VERSION = (0, 0)

try:
    from pymatgen.symmetry.bandstructure import HighSymmKpath
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.io.ase import AseAtomsAdaptor
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False


import contextlib
import io
import os
import sys


@contextlib.contextmanager
def _suppress_stderr_containing(substring: str):

    old = sys.stderr
    buf = io.StringIO()
    sys.stderr = buf
    try:
        yield
    finally:
        sys.stderr = old
        captured = buf.getvalue()

        for line in captured.splitlines(keepends=True):
            if substring not in line:
                old.write(line)


NEAR_ZERO_FREQ_THRESHOLD_MEV = 0.5
THZ_TO_MEV = phonopy_units.THzToEv * 1000 if PHONOPY_AVAILABLE else 0.24180


@dataclass
class PhononParams:
    auto_supercell: bool = True
    target_supercell_length: float = 15.0
    max_supercell_multiplier: int = 4
    max_supercell_atoms: int = 800

    supercell_size: tuple[int, int, int] = (2, 2, 2)


    displacement_distance: float = 0.01


    npoints_per_segment: int = 101
    use_auto_kpath: bool = True


    kpath_convention: str = "setyawan_curtarolo"


    manual_kpath: Optional[list] = None


    dos_mesh: tuple[int, int, int] = (30, 30, 30)


    temperature: float = 300.0
    t_min: float = 0.0
    t_max: float = 1500.0
    t_step: float = 10.0


    pre_relax: bool = True
    pre_relax_fmax: float = 0.01
    pre_relax_steps: int = 100
    pre_relax_optimizer: str = "LBFGS"


    imaginary_mode_tol_mev: float = -0.1


@dataclass
class PhononResult:
    success: bool
    error: Optional[str] = None


    frequencies: Optional[np.ndarray] = None
    kpoint_distances: Optional[np.ndarray] = None
    kpoint_labels: list[str] = field(default_factory=list)
    kpoint_label_positions: list[float] = field(default_factory=list)


    dos_energies: Optional[np.ndarray] = None
    dos_values: Optional[np.ndarray] = None


    supercell_matrix: Optional[list] = None
    n_supercell_atoms: int = 0
    n_unit_cell_atoms: int = 0
    n_displacements: int = 0


    pre_relax_converged: Optional[bool] = None
    pre_relax_final_fmax: Optional[float] = None


    imaginary_modes: int = 0
    min_frequency: float = 0.0
    is_stable: bool = True
    stability_notes: str = ""


    thermodynamics: Optional[dict] = None


    thermal_properties: Optional[dict] = None
    band_yaml_content: Optional[str] = None
    method: str = "Phonopy"
    kpath_convention: str = "setyawan_curtarolo"


def _atoms_to_phonopy(atoms: Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.get_cell()[:],
    )


def _estimate_supercell(atoms: Atoms, params: PhononParams, log: Callable) -> list:
    cell_lengths = np.linalg.norm(atoms.get_cell(), axis=1)
    log(f"  Unit cell: a={cell_lengths[0]:.3f} b={cell_lengths[1]:.3f} c={cell_lengths[2]:.3f} Å")

    mults = []
    for length in cell_lengths:
        m = max(1, int(np.ceil(params.target_supercell_length / length)))
        m = min(m, params.max_supercell_multiplier)
        mults.append(m)

    n = len(atoms)
    if n > 50:
        mults = [max(1, min(2, m)) for m in mults]

    total = n * int(np.prod(mults))
    if total > params.max_supercell_atoms:
        log(f"  ⚠ Supercell too large ({total} atoms), reducing to 1×1×1")
        mults = [1, 1, 1]
        total = n

    mat = [[mults[0], 0, 0], [0, mults[1], 0], [0, 0, mults[2]]]
    log(f"  Supercell {mults[0]}×{mults[1]}×{mults[2]} → {total} atoms")
    return mat


def _zero_near_acoustic(frequencies: np.ndarray, threshold: float = NEAR_ZERO_FREQ_THRESHOLD_MEV) -> np.ndarray:
    out = frequencies.copy()
    out[np.abs(out) < threshold] = 0.0
    return out


def _build_kpath_pymatgen(atoms: Atoms, npoints: int, log: Callable,
                          convention: str = "setyawan_curtarolo"):
    if not PYMATGEN_AVAILABLE:
        raise ImportError("pymatgen is required for automatic k-path detection")

    import warnings

    adaptor = AseAtomsAdaptor()
    pmg_raw = adaptor.get_structure(atoms)

    try:
        sga = SpacegroupAnalyzer(pmg_raw)
        pmg = sga.get_primitive_standard_structure()
        spg_symbol = sga.get_space_group_symbol()
        spg_number = sga.get_space_group_number()
        log(f"  Space group: {spg_symbol} (#{spg_number})  "
            f"— standardised from {len(pmg_raw)} → {len(pmg)} atoms")
    except Exception as exc:
        log(f"  ⚠ SpacegroupAnalyzer failed ({exc}); using raw structure for k-path")
        pmg = pmg_raw

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning,
                                message=".*standard primitive.*")
        try:
            kpath_obj = HighSymmKpath(pmg, path_type=convention)
        except TypeError:
            log(f"  ⚠ This pymatgen version does not support path_type='{convention}'; "
                f"falling back to default (setyawan_curtarolo)")
            kpath_obj = HighSymmKpath(pmg)

    path_segs    = kpath_obj.kpath["path"]
    kpoints_dict = kpath_obj.kpath["kpoints"]

    pmg_cell = np.array(pmg.lattice.matrix)
    rec_std  = np.linalg.inv(pmg_cell).T * 2 * np.pi

    all_kpts: list[np.ndarray] = []
    dist  = 0.0


    label_list: list[tuple[float, str]] = []

    def _add_label(pos: float, name: str):
        key = round(pos, 8)
        for idx, (p, l) in enumerate(label_list):
            if abs(p - pos) < 1e-8:

                parts = l.split("|")
                if name not in parts:
                    label_list[idx] = (p, l + "|" + name)
                return
        label_list.append((pos, name))

    for seg in path_segs:
        for i in range(len(seg) - 1):
            start_name = _greek(seg[i])
            end_name   = _greek(seg[i + 1])
            start = np.array(kpoints_dict[seg[i]],     dtype=float)
            end   = np.array(kpoints_dict[seg[i + 1]], dtype=float)

            seg_start_dist = dist

            for j in range(npoints):
                t   = j / (npoints - 1)
                kpt = start + t * (end - start)

                if j > 0:
                    dk    = (kpt - all_kpts[-1]) @ rec_std
                    dist += float(np.linalg.norm(dk))

                all_kpts.append(kpt)


            _add_label(seg_start_dist, start_name)
            _add_label(dist,           end_name)

    kpts_array = np.array(all_kpts)
    dist_array = np.zeros(len(all_kpts))

    d = 0.0
    dist_array[0] = 0.0
    for idx in range(1, len(kpts_array)):
        dk = (kpts_array[idx] - kpts_array[idx - 1]) @ rec_std
        d += float(np.linalg.norm(dk))
        dist_array[idx] = d


    label_list.sort(key=lambda x: x[0])
    clean_positions = [p for p, _ in label_list]
    clean_labels    = [l for _, l in label_list]

    path_str = " → ".join(clean_labels)
    convention_display = getattr(kpath_obj, "kpath_type",
                                 getattr(kpath_obj, "path_type", convention))
    log(f"  High-symmetry path ({convention_display}): {path_str}  ({len(all_kpts)} k-points)")
    return kpts_array, clean_labels, clean_positions, dist_array


def _build_kpath_fallback(npoints: int, log: Callable):
    log("  ⚠ Using fallback Γ–X–M–Γ path")
    segments = [
        (np.array([0, 0, 0]), np.array([0.5, 0, 0])),
        (np.array([0.5, 0, 0]), np.array([0.5, 0.5, 0])),
        (np.array([0.5, 0.5, 0]), np.array([0, 0, 0])),
    ]
    all_kpts, cumul, dist = [], [0.0], 0.0
    seg_lengths = [0.5, 0.5 / 2, np.sqrt(2) * 0.5]
    for (start, end), seg_len in zip(segments, seg_lengths):
        for j in range(1, npoints):
            t = j / (npoints - 1)
            all_kpts.append(start + t * (end - start))
            dist += seg_len / (npoints - 1)
            cumul.append(dist)
    kpts = np.array(all_kpts)
    d = np.array(cumul[:-1])
    labels = ["Γ", "X", "M", "Γ"]
    positions = [0.0, 0.5, 0.5 + 0.5 / 2, dist]
    return kpts, labels, positions, d


def _greek(name: str) -> str:
    return "Γ" if name.upper() in ("GAMMA", "G") else name


def _build_kpath_manual(segments: list, npoints: int, cell, log: Callable):
    if not segments:
        raise ValueError("manual_kpath is empty")

    rec = cell.reciprocal()

    all_kpts: list[np.ndarray] = []
    labels: list[str] = []
    label_positions: list[float] = []
    dist = 0.0

    for seg_idx, seg in enumerate(segments):
        start = np.array(seg["start_coords"], dtype=float)
        end   = np.array(seg["end_coords"],   dtype=float)
        s_lbl = _greek(str(seg.get("start_label", f"P{seg_idx}")))
        e_lbl = _greek(str(seg.get("end_label",   f"P{seg_idx+1}")))


        if label_positions and abs(label_positions[-1] - dist) < 1e-8:
            if labels[-1] != s_lbl:
                labels[-1] = labels[-1] + "|" + s_lbl
        else:
            labels.append(s_lbl)
            label_positions.append(dist)

        for j in range(npoints):
            t = j / (npoints - 1)
            kpt = start + t * (end - start)
            if len(all_kpts) > 0:
                dk_cart = (kpt - all_kpts[-1]) @ rec
                dist += float(np.linalg.norm(dk_cart))
            all_kpts.append(kpt)


        labels.append(e_lbl)
        label_positions.append(dist)

    kpts_array = np.array(all_kpts)
    dist_array = np.zeros(len(all_kpts))

    rec_mat = cell.reciprocal()
    dist_array[0] = 0.0
    for i in range(1, len(kpts_array)):
        dk = (kpts_array[i] - kpts_array[i - 1]) @ rec_mat
        dist_array[i] = dist_array[i - 1] + float(np.linalg.norm(dk))


    label_positions_clean: list[float] = []
    labels_clean: list[str] = []
    seg_boundary_idx = 0
    for seg_idx, seg in enumerate(segments):
        idx = seg_idx * npoints
        label_positions_clean.append(float(dist_array[min(idx, len(dist_array) - 1)]))
        labels_clean.append(_greek(str(seg.get("start_label", f"P{seg_idx}"))))

    label_positions_clean.append(float(dist_array[-1]))
    labels_clean.append(_greek(str(segments[-1].get("end_label", f"P{len(segments)}"))))

    path_str = " → ".join(labels_clean)
    log(f"  Manual k-path: {path_str}  ({len(all_kpts)} k-points)")
    return kpts_array, labels_clean, label_positions_clean, dist_array


def suggest_supercell_for_structure(
    atoms: Atoms,
    target_length: float = 15.0,
    max_multiplier: int = 4,
    max_atoms: int = 800,
) -> tuple[int, int, int]:
    cell_lengths = np.linalg.norm(atoms.get_cell(), axis=1)
    n = len(atoms)

    mults = []
    for length in cell_lengths:
        m = max(1, int(np.ceil(target_length / max(length, 1e-6))))
        m = min(m, max_multiplier)
        mults.append(m)

    if n > 50:
        mults = [max(1, min(2, m)) for m in mults]

    total = n * int(np.prod(mults))
    if total > max_atoms:
        mults = [1, 1, 1]

    return (int(mults[0]), int(mults[1]), int(mults[2]))


def _parse_phonopy_band_dict(band_dict: dict, log: Callable):
    raw_freq = band_dict["frequencies"]
    raw_kpts = band_dict["qpoints"]


    if isinstance(raw_freq, np.ndarray):
        freq = raw_freq
    else:

        freq = np.concatenate([np.atleast_2d(np.array(seg)) for seg in raw_freq], axis=0)


    if isinstance(raw_kpts, np.ndarray):
        kpts = raw_kpts
    else:
        kpts = np.concatenate([np.atleast_2d(np.array(seg)) for seg in raw_kpts], axis=0)

    if freq.ndim == 1:
        freq = freq[:, np.newaxis]


    n = min(len(freq), len(kpts))
    freq = freq[:n]
    kpts = kpts[:n]

    log(f"  Band structure: {freq.shape[0]} k-points, {freq.shape[1]} bands")
    return freq, kpts


class PhononCalculator:

    def __init__(
        self,
        atoms: Atoms,
        calculator,
        params: Optional[PhononParams] = None,
        log_fn: Optional[Callable] = None,
    ):
        if not PHONOPY_AVAILABLE:
            raise ImportError(
                "phonopy is required: pip install phonopy"
            )
        self.atoms = atoms.copy()
        self.calculator = calculator
        self.params = params or PhononParams()
        self._log = log_fn if log_fn is not None else lambda msg: logger.info(msg)


    def run(self) -> PhononResult:
        try:
            return self._run()
        except Exception as exc:
            self._log(f"❌ Phonon calculation failed: {exc}")
            self._log(traceback.format_exc())
            return PhononResult(success=False, error=str(exc))


    def _run(self) -> PhononResult:
        p = self.params


        pre_relax_converged: Optional[bool] = None
        pre_relax_final_fmax: Optional[float] = None
        if p.pre_relax:
            atoms, pre_relax_converged, pre_relax_final_fmax = self._pre_relax(self.atoms.copy())
        else:
            atoms = self.atoms.copy()

        n_unit_cell_atoms = len(atoms)


        sc_mat = (
            _estimate_supercell(atoms, p, self._log)
            if p.auto_supercell
            else [[p.supercell_size[i] if i == j else 0 for j in range(3)] for i in range(3)]
        )


        phonopy_atoms = _atoms_to_phonopy(atoms)
        phonon = Phonopy(
            phonopy_atoms,
            supercell_matrix=sc_mat,
            primitive_matrix="auto",
            log_level=0,
        )


        phonon.generate_displacements(distance=p.displacement_distance)
        supercells = phonon.supercells_with_displacements
        n_disp = len(supercells)
        n_sc = int(np.prod([sc_mat[i][i] for i in range(3)]))
        n_supercell_atoms = n_unit_cell_atoms * n_sc
        self._log(f"  {n_disp} displaced supercells to evaluate ({n_supercell_atoms} atoms each)")


        forces = self._compute_forces(supercells)


        phonon.forces = forces
        phonon.produce_force_constants()
        self._log("  Force constants produced")


        kpts, labels, label_positions, dist_array = self._build_kpath(atoms)
        freq_thz, _, band_yaml = self._run_band_structure(phonon, kpts)
        freq_mev = _zero_near_acoustic(freq_thz * THZ_TO_MEV)


        dos_energies, dos_values = self._run_dos(phonon, freq_mev)


        thermal_dict, thermo_at_T = self._run_thermodynamics(phonon, p.temperature)


        tol = p.imaginary_mode_tol_mev
        valid = freq_mev[~np.isnan(freq_mev)]
        imaginary_modes = int(np.sum(valid < tol))
        min_freq = float(np.min(valid)) if len(valid) else 0.0
        is_stable = imaginary_modes == 0

        if is_stable:
            if min_freq < 0:

                stability_notes = (
                    f"✅ Dynamically stable (most negative mode: {min_freq:.3f} meV, "
                    f"within tolerance of {tol:.2f} meV)"
                )
            else:
                stability_notes = "✅ Dynamically stable (no imaginary modes)"
        else:
            stability_notes = (
                f"⚠ {imaginary_modes} imaginary mode(s) below {tol:.2f} meV "
                f"(most negative: {min_freq:.3f} meV)"
            )

        self._log(f"  {stability_notes}")

        return PhononResult(
            success=True,
            frequencies=freq_mev,
            kpoint_distances=dist_array,
            kpoint_labels=labels,
            kpoint_label_positions=label_positions,
            dos_energies=dos_energies,
            dos_values=dos_values,
            supercell_matrix=sc_mat,
            n_supercell_atoms=n_supercell_atoms,
            n_unit_cell_atoms=n_unit_cell_atoms,
            n_displacements=n_disp,
            pre_relax_converged=pre_relax_converged,
            pre_relax_final_fmax=pre_relax_final_fmax,
            imaginary_modes=imaginary_modes,
            min_frequency=min_freq,
            is_stable=is_stable,
            stability_notes=stability_notes,
            thermodynamics=thermo_at_T,
            thermal_properties=thermal_dict,
            band_yaml_content=band_yaml,
            method="Phonopy (updated API)",
            kpath_convention=p.kpath_convention,
        )


    def _pre_relax(self, atoms: Atoms) -> tuple[Atoms, bool, float]:
        p = self.params
        optimizer_name = p.pre_relax_optimizer.upper()
        self._log(f"  Pre-relaxation ({optimizer_name}, fmax={p.pre_relax_fmax} eV/Å) …")
        atoms.calc = self.calculator
        converged = False
        final_fmax = float("nan")
        try:
            with _suppress_stderr_containing("__path__._path"):
                if optimizer_name == "FIRE":
                    opt = FIRE(atoms, logfile=None)
                else:
                    opt = LBFGS(atoms, logfile=None)
                opt.run(fmax=p.pre_relax_fmax, steps=p.pre_relax_steps)
            e = atoms.get_potential_energy()
            final_fmax = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
            converged = final_fmax <= p.pre_relax_fmax
            status = "✅ converged" if converged else f"⚠ did NOT converge (fmax={final_fmax:.4f} > {p.pre_relax_fmax})"
            self._log(f"  Pre-relax {status}: E={e:.6f} eV  F_max={final_fmax:.4f} eV/Å")
        except Exception as exc:
            self._log(f"  ⚠ Pre-relaxation failed ({exc}), continuing without")
        return atoms, converged, final_fmax

    def _compute_forces(self, supercells) -> list[np.ndarray]:
        forces = []
        n = len(supercells)
        for i, sc in enumerate(supercells):
            ase_sc = Atoms(
                symbols=sc.symbols,
                positions=sc.positions,
                cell=sc.cell,
                pbc=True,
            )


            with _suppress_stderr_containing("__path__._path"):
                ase_sc.calc = self.calculator
                f = ase_sc.get_forces()
            forces.append(f)
            if (i + 1) % max(1, n // 10) == 0 or i == n - 1:
                self._log(f"  Forces: {i + 1}/{n} ({(i + 1) / n * 100:.0f}%)")
        return forces

    def _build_kpath(self, atoms: Atoms):
        p = self.params


        if p.manual_kpath:
            try:
                return _build_kpath_manual(
                    p.manual_kpath, p.npoints_per_segment, atoms.get_cell(), self._log
                )
            except Exception as exc:
                self._log(f"  ⚠ Manual k-path failed ({exc}), falling back to auto")


        if p.use_auto_kpath and PYMATGEN_AVAILABLE:
            try:
                return _build_kpath_pymatgen(
                    atoms, p.npoints_per_segment, self._log,
                    convention=p.kpath_convention,
                )
            except Exception as exc:
                self._log(f"  ⚠ Pymatgen k-path failed ({exc}), using fallback")


        return _build_kpath_fallback(p.npoints_per_segment, self._log)

    def _run_band_structure(self, phonon: "Phonopy", kpts: np.ndarray):
        n_per = self.params.npoints_per_segment
        n_total = len(kpts)


        paths = []
        for start in range(0, n_total, n_per):
            seg = kpts[start: start + n_per]
            if len(seg) >= 2:
                paths.append(seg)

        if not paths:
            paths = [kpts]

        phonon.run_band_structure(
            paths,
            with_eigenvectors=True,
            is_band_connection=False,
        )
        band_yaml_content: Optional[str] = None
        try:
            import tempfile, os
            with tempfile.TemporaryDirectory() as tmpdir:
                yaml_path = os.path.join(tmpdir, "band.yaml")
                phonon.write_yaml_band_structure(filename=yaml_path)
                with open(yaml_path, "r") as fh:
                    band_yaml_content = fh.read()
            self._log("  ✅ band.yaml with eigenvectors written")
        except Exception as exc:
            self._log(f"  ⚠ Could not capture band.yaml: {exc}")
        band_dict = phonon.get_band_structure_dict()
        freq_thz, _ = _parse_phonopy_band_dict(band_dict, self._log)
        return freq_thz, band_dict, band_yaml_content

    def _run_dos(self, phonon: "Phonopy", freq_mev_ref: np.ndarray):
        p = self.params

        n_sc = phonon.supercell.get_number_of_atoms()
        mesh = p.dos_mesh if n_sc <= 150 else (20, 20, 20)

        try:
            phonon.run_mesh(mesh, with_eigenvectors=False)
            phonon.run_total_dos()
            dos_dict = phonon.get_total_dos_dict()
            energies = np.array(dos_dict["frequency_points"]) * THZ_TO_MEV
            values   = np.array(dos_dict["total_dos"])
            self._log(f"  DOS computed on {mesh} mesh, {len(energies)} points")
            return energies, values
        except Exception as exc:
            self._log(f"  ⚠ DOS failed ({exc}); using Gaussian smearing fallback")
            return self._gaussian_dos_fallback(freq_mev_ref)

    @staticmethod
    def _gaussian_dos_fallback(freq_mev: np.ndarray, sigma: float = 2.0):
        valid = freq_mev[~np.isnan(freq_mev)]
        positive = valid[valid > 0]
        if len(positive) == 0:
            x = np.linspace(0, 50, 500)
            return x, np.zeros(500)
        x = np.linspace(0, positive.max() * 1.2, 500)
        dos = np.zeros_like(x)
        for f in positive:
            dos += np.exp(-0.5 * ((x - f) / sigma) ** 2)
        dos /= len(positive) * sigma * np.sqrt(2 * np.pi)
        return x, dos

    def _run_thermodynamics(self, phonon: "Phonopy", temperature: float):
        p = self.params
        try:
            phonon.run_thermal_properties(
                t_min=p.t_min,
                t_max=p.t_max,
                t_step=p.t_step,
            )
            td = phonon.get_thermal_properties_dict()


            temps = np.array(td["temperatures"])

            def _get(key, alt_keys=()):
                if key in td:
                    return np.array(td[key])
                for k in alt_keys:
                    if k in td:
                        return np.array(td[k])
                return None

            free_energy = _get("free_energy", ("helmholtz_free_energy",))
            entropy      = _get("entropy",      ("entropies",))
            heat_cap     = _get("heat_capacity", ("cv", "heat_capacities"))


            if "zero_point_energy" in td:
                zpe = float(td["zero_point_energy"])
            elif free_energy is not None and len(temps) > 0:

                zpe = float(free_energy[0])
            else:
                zpe = 0.0

            thermal_dict = {
                "temperatures":       temps.tolist(),
                "free_energy":        free_energy.tolist() if free_energy is not None else [],
                "entropy":            entropy.tolist()     if entropy      is not None else [],
                "heat_capacity":      heat_cap.tolist()    if heat_cap     is not None else [],
                "zero_point_energy":  zpe,

                "_units": {
                    "free_energy":       "kJ/mol",
                    "entropy":           "J/K/mol",
                    "heat_capacity":     "J/K/mol",
                    "zero_point_energy": "kJ/mol",
                },
            }


            idx = int(np.argmin(np.abs(temps - temperature)))
            thermo_at_T = {
                "temperature":       float(temps[idx]),
                "zero_point_energy": zpe,
                "free_energy":       float(free_energy[idx]) if free_energy is not None else None,
                "entropy":           float(entropy[idx])     if entropy      is not None else None,
                "heat_capacity":     float(heat_cap[idx])    if heat_cap     is not None else None,
                "_units": thermal_dict["_units"],
            }

            def _fv(v):
                return f"{v:.4f}" if v is not None else "N/A"

            self._log(
                f"  Thermodynamics at {thermo_at_T['temperature']:.0f} K: "
                f"ZPE={_fv(zpe)} kJ/mol  "
                f"F={_fv(thermo_at_T.get('free_energy'))} kJ/mol  "
                f"S={_fv(thermo_at_T.get('entropy'))} J/K/mol  "
                f"Cv={_fv(thermo_at_T.get('heat_capacity'))} J/K/mol"
            )
            return thermal_dict, thermo_at_T

        except Exception as exc:
            self._log(f"  ⚠ Thermal properties failed ({exc}); using Einstein fallback")
            thermo_at_T = self._einstein_thermo_fallback(temperature)
            return None, thermo_at_T

    @staticmethod
    def _einstein_thermo_fallback(temperature: float) -> dict:
        return {
            "temperature": temperature,
            "zero_point_energy": None,
            "free_energy": None,
            "entropy": None,
            "heat_capacity": None,
        }


def calculate_phonons(
    atoms: Atoms,
    calculator,
    phonon_params: Optional[dict] = None,
    log_queue=None,
    structure_name: str = "structure",
) -> dict:

    params = PhononParams()
    if phonon_params:
        for k, v in phonon_params.items():
            if hasattr(params, k):
                setattr(params, k, v)

        if "delta" in phonon_params:
            params.displacement_distance = phonon_params["delta"]
        if "npoints" in phonon_params:
            params.npoints_per_segment = phonon_params["npoints"]
        if "auto_kpath" in phonon_params:
            params.use_auto_kpath = phonon_params["auto_kpath"]
        if "supercell_size" in phonon_params and not phonon_params.get("auto_supercell", True):
            params.supercell_size = tuple(phonon_params["supercell_size"])
            params.auto_supercell = False


    def _log(msg: str):
        if log_queue is not None:
            if hasattr(log_queue, "put"):
                log_queue.put(msg)
            elif isinstance(log_queue, list):
                log_queue.append(msg)
        logger.info(msg)

    _log(f"Starting phonon calculation for {structure_name}")

    calc = PhononCalculator(atoms, calculator, params, log_fn=_log)
    result = calc.run()

    if not result.success:
        return {"success": False, "error": result.error}

    sc = result.supercell_matrix
    sc_tuple = tuple(sc[i][i] for i in range(3)) if sc else (1, 1, 1)

    return {
        "success":                  True,
        "frequencies":              result.frequencies,
        "kpoints":                  np.zeros((len(result.frequencies), 3)) if result.frequencies is not None else None,
        "kpoint_distances":         result.kpoint_distances,
        "kpoint_labels":            result.kpoint_labels,
        "kpoint_label_positions":   result.kpoint_label_positions,
        "dos_energies":             result.dos_energies,
        "dos":                      result.dos_values,
        "thermodynamics":           result.thermodynamics,
        "thermal_properties_dict":  result.thermal_properties,
        "supercell_size":           sc_tuple,
        "n_supercell_atoms":        result.n_supercell_atoms,
        "n_unit_cell_atoms":        result.n_unit_cell_atoms,
        "imaginary_modes":          result.imaginary_modes,
        "min_frequency":            result.min_frequency,
        "is_stable":                result.is_stable,
        "stability_notes":          result.stability_notes,
        "pre_relax_converged":      result.pre_relax_converged,
        "pre_relax_final_fmax":     result.pre_relax_final_fmax,
        "method":                   "Phonopy (updated API)",
        "enhanced_kpoints":         True,
        "band_yaml_content": result.band_yaml_content,
        "kpath_convention":         result.kpath_convention,
    }


def extract_thermodynamics_at_temperatures(
    phonon_result: dict,
    target_temperatures: list[float],
) -> dict:
    if not phonon_result.get("success"):
        return {"error": "Phonon calculation was not successful"}

    td = phonon_result.get("thermal_properties_dict")
    if td is None:
        return {"error": "No thermal properties data available"}

    temps     = np.array(td["temperatures"])

    def _safe_array(key):
        val = td.get(key)
        if val is None:
            return None
        arr = np.array(val)
        return arr if arr.size > 0 else None

    free_e   = _safe_array("free_energy")
    entropy  = _safe_array("entropy")
    heat_cap = _safe_array("heat_capacity")
    zpe      = float(td.get("zero_point_energy") or 0.0)

    results = {}
    for T in target_temperatures:
        idx = int(np.argmin(np.abs(temps - T)))
        results[T] = {
            "temperature":       float(temps[idx]),
            "zero_point_energy": zpe,
            "free_energy":       float(free_e[idx])   if free_e   is not None else None,
            "entropy":           float(entropy[idx])  if entropy  is not None else None,
            "heat_capacity":     float(heat_cap[idx]) if heat_cap is not None else None,
        }
    return results


def create_phonon_data_export(phonon_result: dict, structure_name: str) -> Optional[dict]:
    if not phonon_result.get("success"):
        return None

    def _to_list(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    freqs = phonon_result.get("frequencies")
    dists = phonon_result.get("kpoint_distances")


    band_structure_table: Optional[dict] = None
    if freqs is not None and dists is not None:
        freqs_arr = np.array(freqs)
        dists_arr = np.array(dists)
        if freqs_arr.ndim == 2 and len(dists_arr) == len(freqs_arr):
            n_bands = freqs_arr.shape[1]
            columns = ["kpoint_distance_inv_ang"] + [f"band_{i+1}_meV" for i in range(n_bands)]
            data = np.column_stack([dists_arr[:, np.newaxis], freqs_arr]).tolist()
            band_structure_table = {"columns": columns, "data": data}

    out = {
        "structure_name":         structure_name,
        "method":                 phonon_result.get("method", "Phonopy"),
        "supercell_size":         list(phonon_result.get("supercell_size", [])),
        "imaginary_modes":        int(phonon_result["imaginary_modes"]),
        "min_frequency_meV":      float(phonon_result["min_frequency"]),

        "kpoint_distances_inv_ang": _to_list(dists),
        "kpoint_labels":          phonon_result.get("kpoint_labels", []),
        "kpoint_label_positions": phonon_result.get("kpoint_label_positions", []),
        "frequencies_meV":        _to_list(freqs),

        "band_structure":         band_structure_table,

        "dos_energies_meV":       _to_list(phonon_result.get("dos_energies")),
        "dos":                    _to_list(phonon_result.get("dos")),
    }

    thermo = phonon_result.get("thermodynamics")
    if thermo:
        out["thermodynamics"] = {
            k: (float(v) if (v is not None and not isinstance(v, dict)) else v)
            for k, v in thermo.items()
        }

    return out


def format_phonon_summary(phonon_result: dict, structure_name: str = "") -> dict:
    if not phonon_result.get("success"):
        return {"structure_name": structure_name, "error": phonon_result.get("error", "Unknown error")}


    sc = phonon_result.get("supercell_size", (1, 1, 1))
    sc_str = "×".join(str(s) for s in sc)


    n_unit = int(phonon_result.get("n_unit_cell_atoms", 0))
    n_sc   = int(phonon_result.get("n_supercell_atoms", 0))
    if n_unit == 0 or n_sc == 0:

        freqs = phonon_result.get("frequencies")
        if freqs is not None:
            try:
                n_bands = np.array(freqs).shape[1]
                n_sc_inferred = n_bands // 3
                n_sc_mult = int(np.prod(list(sc)))
                n_unit_inferred = n_sc_inferred // max(n_sc_mult, 1)
                if n_unit == 0:
                    n_unit = n_unit_inferred
                if n_sc == 0:
                    n_sc = n_sc_inferred
            except (IndexError, ZeroDivisionError):
                pass


    n_disp_raw = phonon_result.get("n_displacements", None)
    n_disp_str = str(int(n_disp_raw)) if n_disp_raw else "—"


    converged = phonon_result.get("pre_relax_converged")
    fmax_val  = phonon_result.get("pre_relax_final_fmax")
    if converged is None:
        pre_relax_status = "Not performed / unknown"
    elif converged:
        pre_relax_status = f"✅ Converged (F_max = {fmax_val:.4f} eV/Å)"
    else:
        if fmax_val is not None and not (fmax_val != fmax_val):
            pre_relax_status = f"⚠ Did not converge (F_max = {fmax_val:.4f} eV/Å)"
        else:
            pre_relax_status = "⚠ Did not converge (F_max unknown)"


    min_freq_mev = float(phonon_result.get("min_frequency", 0.0))
    tol_mev = float(phonon_result.get("imaginary_mode_tol_mev", -0.1))


    freqs_arr = phonon_result.get("frequencies")
    if freqs_arr is not None:
        valid = np.array(freqs_arr)[~np.isnan(np.array(freqs_arr))]
        n_imaginary = int(np.sum(valid < tol_mev))
    else:

        n_imaginary = int(phonon_result.get("imaginary_modes", 0))

    is_stable = n_imaginary == 0


    if is_stable:
        if min_freq_mev < 0:
            stability_notes = (
                f"✅ Dynamically stable (most negative mode: {min_freq_mev:.3f} meV, "
                f"within tolerance of {tol_mev:.2f} meV)"
            )
        else:
            stability_notes = "✅ Dynamically stable (no imaginary modes)"
    else:
        stability_notes = (
            f"⚠ {n_imaginary} imaginary mode(s) below {tol_mev:.2f} meV "
            f"(most negative: {min_freq_mev:.3f} meV)"
        )


    thermo = phonon_result.get("thermodynamics") or {}
    units  = thermo.get("_units", {
        "free_energy":       "kJ/mol",
        "entropy":           "J/K/mol",
        "heat_capacity":     "J/K/mol",
        "zero_point_energy": "kJ/mol",
    })
    thermo_clean = {k: v for k, v in thermo.items() if not k.startswith("_")}

    return {
        "structure_name":    structure_name or phonon_result.get("structure_name", ""),
        "method":            phonon_result.get("method", "Phonopy"),
        "stability":         "Stable ✅" if is_stable else "Unstable ⚠",
        "stability_detail":  stability_notes,
        "is_stable":         is_stable,
        "imaginary_modes":   n_imaginary,
        "min_frequency_meV": min_freq_mev,
        "unit_cell_atoms":   n_unit,
        "supercell_size":    sc_str,
        "supercell_atoms":   n_sc,
        "n_displacements":   n_disp_str,
        "pre_relax_status":  pre_relax_status,
        "thermodynamics":    thermo_clean,
        "thermo_units":      units,
    }


def render_phonon_results_tab(
    phonon_results: list,
    add_entropy_vs_temperature_plot=None,
    key_prefix: str = "phonon_tab",
) -> None:
    try:
        import streamlit as st
        import plotly.graph_objects as go
        import pandas as pd
        import json as _json
    except ImportError as e:
        raise ImportError(
            "render_phonon_results_tab requires streamlit, plotly and pandas"
        ) from e

    if not phonon_results:
        return

    st.subheader("🎵 Phonon Properties")


    if len(phonon_results) == 1:
        selected_phonon = phonon_results[0]
        st.write(f"**Structure:** {selected_phonon['name']}")
    else:
        selected_name = st.selectbox(
            "Select structure for phonon analysis:",
            [r["name"] for r in phonon_results],
            key=f"{key_prefix}_selector",
        )
        selected_phonon = next(r for r in phonon_results if r["name"] == selected_name)

    phonon_data = selected_phonon["phonon_results"]


    freq_unit = st.radio(
        "Frequency unit:",
        options=["meV", "THz"],
        index=0,
        horizontal=True,
        key=f"{key_prefix}_freq_unit",
        help="1 THz = 4.136 meV  (conversion via Phonopy THzToEv constant)",
    )
    MEV_TO_THZ = 1.0 / THZ_TO_MEV
    freq_scale = MEV_TO_THZ if freq_unit == "THz" else 1.0
    freq_label = f"Frequency ({freq_unit})"


    col_ph1, col_ph2 = st.columns(2)

    with col_ph1:
        st.write("**Phonon Dispersion**")

        frequencies_mev = np.array(phonon_data["frequencies"])
        frequencies     = frequencies_mev * freq_scale
        nkpts, nbands   = frequencies.shape

        if phonon_data.get("enhanced_kpoints") and "kpoint_distances" in phonon_data:
            x_axis     = phonon_data["kpoint_distances"]
            x_title    = "Distance along k-path"
            use_labels = True
        else:
            x_axis     = list(range(nkpts))
            x_title    = "k-point index"
            use_labels = False

        fig_disp = go.Figure()

        _BAND_COLORS = [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
        ]

        for band in range(nbands):
            color = _BAND_COLORS[band % len(_BAND_COLORS)]
            fig_disp.add_trace(go.Scatter(
                x=x_axis, y=frequencies[:, band],
                mode="lines", line=dict(width=1.5, color=color),
                showlegend=False,
                hovertemplate=f"Frequency: %{{y:.3f}} {freq_unit}<extra></extra>",
            ))

        tol_mev = float(phonon_data.get("imaginary_mode_tol_mev", -0.1))
        imag_threshold = min(tol_mev, 0.0) * freq_scale
        first_imag_legend = True
        for band in range(nbands):
            imag_idx = np.where(frequencies[:, band] < imag_threshold)[0]
            if len(imag_idx):
                fig_disp.add_trace(go.Scatter(
                    x=[x_axis[i] for i in imag_idx],
                    y=frequencies[imag_idx, band],
                    mode="markers", marker=dict(color="red", size=4),
                    name="Imaginary modes", showlegend=first_imag_legend,
                    hovertemplate=f"Imaginary: %{{y:.3f}} {freq_unit}<extra></extra>",
                ))
                first_imag_legend = False

        if use_labels and "kpoint_labels" in phonon_data and "kpoint_label_positions" in phonon_data:
            labels    = phonon_data["kpoint_labels"]
            positions = phonon_data["kpoint_label_positions"]
            display_labels = ["Γ" if l.upper() == "GAMMA" else l for l in labels]
            for pos in positions:
                fig_disp.add_vline(x=pos, line_dash="dash",
                                   line_color="gray", opacity=0.7, line_width=1)
            fig_disp.update_layout(xaxis=dict(
                tickmode="array", tickvals=positions, ticktext=display_labels,
                title=x_title, title_font=dict(size=20), tickfont=dict(size=18),
            ))
        else:
            fig_disp.update_layout(xaxis=dict(
                title=x_title, title_font=dict(size=20), tickfont=dict(size=18),
            ))

        fig_disp.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig_disp.update_layout(
            title=dict(text="Phonon Dispersion", font=dict(size=24)),
            yaxis=dict(title=freq_label,
                       title_font=dict(size=20), tickfont=dict(size=20)),
            height=750, font=dict(size=20), hovermode="closest",
            hoverlabel=dict(bgcolor="white", bordercolor="black",
                            font_size=20, font_family="Arial"),
        )
        st.plotly_chart(fig_disp, use_container_width=True)

        if use_labels and "kpoint_labels" in phonon_data:
            _CONVENTION_DISPLAY = {
                "setyawan_curtarolo": "Setyawan-Curtarolo 2010",
                "hinuma":             "SeeK-path / HPKOT (Hinuma 2017)",
                "latimer_munro":      "Latimer-Munro 2020",
            }
            _conv_key = phonon_data.get("kpath_convention", "setyawan_curtarolo")
            _conv_str = _CONVENTION_DISPLAY.get(_conv_key, _conv_key)
            st.info(
                f"🗺️ **k-path** ({_conv_str}): "
                f"{' → '.join(phonon_data['kpoint_labels'])}  "
                f"({len(x_axis)} k-points)"
            )

    with col_ph2:
        st.write("**Phonon Density of States**")

        dos_energies = np.array(phonon_data["dos_energies"]) * freq_scale
        dos_values   = np.array(phonon_data["dos"])
        dos_x_label  = f"DOS (states/{freq_unit})"

        fig_dos = go.Figure()
        fig_dos.add_trace(go.Scatter(
            x=dos_values, y=dos_energies,
            mode="lines", fill="tozerox",
            line=dict(color="blue", width=2), name="DOS",
        ))
        fig_dos.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig_dos.update_layout(
            title=dict(text="Phonon Density of States", font=dict(size=24)),
            xaxis=dict(title=dos_x_label,
                       title_font=dict(size=20), tickfont=dict(size=20)),
            yaxis=dict(title=freq_label,
                       title_font=dict(size=20), tickfont=dict(size=20)),
            height=750, font=dict(size=20), showlegend=False,
            hoverlabel=dict(bgcolor="white", bordercolor="black",
                            font_size=20, font_family="Arial"),
        )
        st.plotly_chart(fig_dos, use_container_width=True)


    st.write("**Phonon Analysis Summary**")

    summary      = format_phonon_summary(phonon_data, selected_phonon["name"])
    thermo       = summary.get("thermodynamics", {})
    thermo_units = summary.get("thermo_units", {})


    use_ev = st.toggle(
        "Show thermodynamic values in eV / eV·K⁻¹ per unit cell",
        value=False,
        key=f"{key_prefix}_ev_toggle",
        help=(
            "Off → Phonopy native: kJ/mol (energies), J/K/mol (entropy & Cv)\n"
            "On  → per unit cell:  eV (energies), eV/K (entropy & Cv)\n"
            "Conversion: kJ/mol ÷ 96.485 = eV,  J/K/mol ÷ 96485 = eV/K"
        ),
    )

    KJ_MOL_TO_EV  = 1.0 / 96.485
    J_KMOL_TO_EVK = 1.0 / 96485.0

    def _cv(val, factor):
        return val * factor if val is not None else None

    def _fmt(val, decimals=4):
        return f"{val:.{decimals}f}" if val is not None else "—"

    u_e  = "eV"   if use_ev else thermo_units.get("zero_point_energy", "kJ/mol")
    u_sk = "eV/K" if use_ev else thermo_units.get("entropy", "J/K/mol")

    prop_rows = [
        ("Structure",                    summary["structure_name"]),
        ("Method",                       summary["method"]),
        ("Unit cell atoms",              str(summary["unit_cell_atoms"])),
        ("Supercell size",               summary["supercell_size"]),
        ("Supercell atoms",              str(summary["supercell_atoms"])),
        ("Displaced supercells",         str(summary["n_displacements"])),
        ("Pre-optimisation",             summary["pre_relax_status"]),
        ("Dynamic stability",            summary["stability_detail"]),
        ("Imaginary modes (above tol.)", str(summary["imaginary_modes"])),
        ("Min frequency (meV)",          _fmt(summary["min_frequency_meV"], 3)),
        ("Max frequency (meV)",          _fmt(float(np.max(frequencies_mev)), 3)),
    ]

    if thermo:
        T     = thermo.get("temperature")
        zpe_d = _cv(thermo.get("zero_point_energy"), KJ_MOL_TO_EV)  if use_ev else thermo.get("zero_point_energy")
        fe_d  = _cv(thermo.get("free_energy"),       KJ_MOL_TO_EV)  if use_ev else thermo.get("free_energy")
        s_d   = _cv(thermo.get("entropy"),           J_KMOL_TO_EVK) if use_ev else thermo.get("entropy")
        cv_d  = _cv(thermo.get("heat_capacity"),     J_KMOL_TO_EVK) if use_ev else thermo.get("heat_capacity")

        prop_rows.append(("─── Thermodynamics ───────────────────", ""))
        prop_rows.append(("Temperature (K)", _fmt(T, 1) if T is not None else "—"))
        prop_rows.append((f"Zero-point energy ({u_e})",      _fmt(zpe_d)))
        prop_rows.append((f"Helmholtz free energy ({u_e})",  _fmt(fe_d)))
        prop_rows.append((f"Entropy ({u_sk})",               _fmt(s_d)))
        prop_rows.append((f"Heat capacity Cv ({u_sk})",      _fmt(cv_d)))

    st.dataframe(
        pd.DataFrame(prop_rows, columns=["Property", "Value"]),
        use_container_width=True, hide_index=True,
    )

    if summary["is_stable"]:
        st.success(summary["stability_detail"])
    else:
        st.warning(summary["stability_detail"])


    phonon_export = create_phonon_data_export(phonon_data, selected_phonon["name"])
    if phonon_export:
        st.download_button(
            label="📥 Download Phonon Data (JSON)",
            data=_json.dumps(phonon_export, indent=2),
            file_name=f"phonon_data_{selected_phonon['name'].replace('.', '_')}.json",
            mime="application/json",
            type="primary",
        )

    band_yaml = phonon_data.get("band_yaml_content")
    if band_yaml:
        st.download_button(
            label="📥 Download band.yaml",
            data=band_yaml,
            file_name=f"band_{selected_phonon['name'].replace('.', '_')}.yaml",
            mime="text/plain",
            type="primary",
        )
    if phonon_data.get("thermal_properties_dict"):
        st.write("**Temperature-Dependent Analysis**")

        col_t1, col_t2, col_t3 = st.columns(3)
        _key = f"{key_prefix}_{selected_phonon['name']}"
        with col_t1:
            min_temp = st.number_input("Min Temperature (K)", min_value=0,
                                       max_value=2000, value=0, step=10,
                                       key=f"min_temp_{_key}")
        with col_t2:
            max_temp = st.number_input("Max Temperature (K)", min_value=100,
                                       max_value=2000, value=1000, step=50,
                                       key=f"max_temp_{_key}")
        with col_t3:
            temp_step = st.number_input("Temperature Step (K)", min_value=1,
                                        max_value=100, value=10, step=1,
                                        key=f"temp_step_{_key}")

        if add_entropy_vs_temperature_plot is not None:
            if st.button("Generate Temperature Analysis",
                         key=f"temp_analysis_{_key}",
                         disabled=True):
                with st.spinner("Calculating thermodynamics over temperature range…"):
                    fig_temp, thermo_td = add_entropy_vs_temperature_plot(
                        phonon_data, temp_range=(min_temp, max_temp, temp_step)
                    )
                    if fig_temp is not None:
                        st.plotly_chart(fig_temp, use_container_width=True)
                        if isinstance(thermo_td, dict) and "error" not in thermo_td:
                            st.download_button(
                                label="📥 Download Temperature-Dependent Data (JSON)",
                                data=_json.dumps({
                                    "structure_name": selected_phonon["name"],
                                    "temperature_dependent_properties": thermo_td,
                                }, indent=2),
                                file_name=f"thermodynamics_vs_temp_{_key.replace('.', '_')}.json",
                                mime="application/json",
                                key=f"dl_temp_{_key}",
                                type="primary",
                            )
                    else:
                        st.error(f"Error generating analysis: {thermo_td}")

        st.write("**Quick Temperature Comparison**")
        target_temps = st.multiselect(
            "Select specific temperatures (K):",
            options=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            default=[300, 600, 1000],
            key=f"target_temps_{_key}",
        )

        if target_temps:
            specific = extract_thermodynamics_at_temperatures(phonon_data, target_temps)
            if "error" not in specific:
                rows = []
                for T in target_temps:
                    if T in specific:
                        d = specific[T]
                        fe  = _cv(d.get("free_energy"),       KJ_MOL_TO_EV)  if use_ev else d.get("free_energy")
                        s   = _cv(d.get("entropy"),           J_KMOL_TO_EVK) if use_ev else d.get("entropy")
                        cv  = _cv(d.get("heat_capacity"),     J_KMOL_TO_EVK) if use_ev else d.get("heat_capacity")
                        zpe = _cv(d.get("zero_point_energy"), KJ_MOL_TO_EV)  if use_ev else d.get("zero_point_energy")
                        rows.append({
                            "Temperature (K)":           f"{d['temperature']:.0f}",
                            f"Free energy ({u_e})":      _fmt(fe),
                            f"Entropy ({u_sk})":         _fmt(s),
                            f"Heat capacity ({u_sk})":   _fmt(cv),
                            f"ZPE ({u_e})":              _fmt(zpe),
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows),
                                 use_container_width=True, hide_index=True)


def extract_element_concentrations(structures_data: dict) -> dict:
    all_compositions = {}
    for name, structure in structures_data.items():
        composition = structure.composition.as_dict()
        total_atoms = sum(composition.values())
        all_compositions[name] = {
            el: (cnt / total_atoms) * 100 for el, cnt in composition.items()
        }
    return all_compositions


def get_common_elements(compositions_dict: dict) -> list:
    all_elements = set()
    for comp in compositions_dict.values():
        all_elements.update(comp.keys())
    common = []
    for el in all_elements:
        if all(el in comp for comp in compositions_dict.values()):
            if len({comp[el] for comp in compositions_dict.values()}) > 1:
                common.append(el)
    return sorted(common)


def calculate_phase_diagram_data(phonon_results: list, element_concentrations: dict,
                                  temp_range: list, use_ev: bool = False) -> "pd.DataFrame":
    import pandas as pd
    KJ_MOL_TO_EV  = 1.0 / 96.485
    J_KMOL_TO_EVK = 1.0 / 96485.0
    rows = []
    for result in phonon_results:
        name        = result["name"]
        phonon_data = result["phonon_results"]
        if name not in element_concentrations:
            continue
        temp_thermo = extract_thermodynamics_at_temperatures(phonon_data, temp_range)
        if "error" in temp_thermo:
            continue
        for T in temp_range:
            if T not in temp_thermo:
                continue
            th = temp_thermo[T]
            fe  = th.get("free_energy")
            s   = th.get("entropy")
            cv  = th.get("heat_capacity")
            zpe = th.get("zero_point_energy")
            if use_ev:
                fe  = fe  * KJ_MOL_TO_EV  if fe  is not None else None
                zpe = zpe * KJ_MOL_TO_EV  if zpe is not None else None
                s   = s   * J_KMOL_TO_EVK if s   is not None else None
                cv  = cv  * J_KMOL_TO_EVK if cv  is not None else None
            rows.append({
                "structure":    name,
                "concentration": list(element_concentrations[name].values())[0]
                                 if len(element_concentrations[name]) == 1
                                 else element_concentrations[name],
                "temperature":  T,
                "free_energy":  fe,
                "entropy":      s,
                "heat_capacity": cv,
                "zero_point_energy": zpe,
            })
    return pd.DataFrame(rows)


def find_stable_phases(phase_df: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd
    rows = []
    for T in phase_df["temperature"].unique():
        sub = phase_df[phase_df["temperature"] == T].dropna(subset=["free_energy"])
        if sub.empty:
            continue
        best = sub.loc[sub["free_energy"].idxmin()]
        rows.append({
            "temperature":        T,
            "stable_structure":   best["structure"],
            "stable_concentration": best["concentration"],
            "free_energy":        best["free_energy"],
        })
    return pd.DataFrame(rows)


def create_phase_diagram_plot(phase_df: "pd.DataFrame", stable_df: "pd.DataFrame",
                               selected_element: str,
                               use_ev: bool = False) -> tuple:
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly is required for create_phase_diagram_plot")

    e_unit  = "eV"        if use_ev else "kJ/mol"
    sk_unit = "eV/K"      if use_ev else "J/K/mol"

    structures = phase_df["structure"].unique()
    colors     = px.colors.qualitative.Set1[:len(structures)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Phase Stability Map ({selected_element} conc. vs Temperature)",
            "Free Energy vs Temperature",
            "Stable Phase Boundaries",
            "Free Energy Differences",
        ),
        specs=[[{}, {}], [{}, {}]],
    )

    for i, struct in enumerate(structures):
        sd = phase_df[phase_df["structure"] == struct]
        fig.add_trace(go.Scatter(
            x=sd["concentration"], y=sd["temperature"],
            mode="markers",
            marker=dict(
                size=8, color=sd["free_energy"],
                colorscale="RdYlBu_r", showscale=(i == 0),
                colorbar=dict(title=f"Free Energy ({e_unit})", x=1.02),
            ),
            name=struct,
            hovertemplate=(
                f"<b>{struct}</b><br>"
                f"{selected_element}: %{{x:.1f}}%<br>"
                f"T: %{{y}} K<br>"
                f"F: %{{marker.color:.4f}} {e_unit}<extra></extra>"
            ),
        ), row=1, col=1)

    for i, struct in enumerate(structures):
        sd    = phase_df[phase_df["structure"] == struct]
        temps = sorted(sd["temperature"].unique())
        avg_f = [sd[sd["temperature"] == t]["free_energy"].mean() for t in temps]
        fig.add_trace(go.Scatter(
            x=temps, y=avg_f,
            mode="lines+markers",
            name=struct, showlegend=False,
            line=dict(color=colors[i], width=2),
            marker=dict(size=6),
            hovertemplate=f"T: %{{x}} K<br>F: %{{y:.4f}} {e_unit}<extra></extra>",
        ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=stable_df["stable_concentration"], y=stable_df["temperature"],
        mode="lines+markers",
        line=dict(color="black", width=4),
        marker=dict(size=8, color="red"),
        name="Stable boundary", showlegend=False,
        hovertemplate=f"Conc: %{{x:.1f}}%<br>T: %{{y}} K<extra></extra>",
    ), row=2, col=1)

    phase_transitions = []
    for i in range(len(stable_df) - 1):
        if stable_df.iloc[i]["stable_structure"] != stable_df.iloc[i + 1]["stable_structure"]:
            phase_transitions.append({
                "temperature": stable_df.iloc[i + 1]["temperature"],
                "from_phase":  stable_df.iloc[i]["stable_structure"],
                "to_phase":    stable_df.iloc[i + 1]["stable_structure"],
            })

    if len(structures) >= 2:
        ref = structures[0]
        ref_data = phase_df[phase_df["structure"] == ref].groupby("temperature")["free_energy"].mean()
        for struct in structures[1:]:
            sd2   = phase_df[phase_df["structure"] == struct].groupby("temperature")["free_energy"].mean()
            temps = sorted(set(ref_data.index) & set(sd2.index))
            diffs = [sd2[t] - ref_data[t] for t in temps]
            fig.add_trace(go.Scatter(
                x=temps, y=diffs,
                mode="lines",
                name=f"{struct} − {ref}", showlegend=False,
                line=dict(width=2),
                hovertemplate=f"T: %{{x}} K<br>ΔF: %{{y:.4f}} {e_unit}<extra></extra>",
            ), row=2, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=2)

    fig.update_xaxes(title_text=f"{selected_element} concentration (%)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (K)",                        row=1, col=2)
    fig.update_xaxes(title_text=f"{selected_element} concentration (%)", row=2, col=1)
    fig.update_xaxes(title_text="Temperature (K)",                        row=2, col=2)

    fig.update_yaxes(title_text="Temperature (K)",          row=1, col=1)
    fig.update_yaxes(title_text=f"Free Energy ({e_unit})",  row=1, col=2)
    fig.update_yaxes(title_text="Temperature (K)",          row=2, col=1)
    fig.update_yaxes(title_text=f"ΔF ({e_unit})",           row=2, col=2)

    fig.update_layout(
        height=800,
        title_text="Computational Phase Diagram Analysis",
        showlegend=True,
        legend=dict(x=1.05, y=1),
    )
    return fig, phase_transitions


def create_concentration_heatmap(phase_df: "pd.DataFrame", selected_element: str,
                                  property_name: str = "free_energy",
                                  use_ev: bool = False) -> "go.Figure":
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required for create_concentration_heatmap")

    e_unit  = "eV"   if use_ev else "kJ/mol"
    sk_unit = "eV/K" if use_ev else "J/K/mol"
    unit    = sk_unit if property_name in ("entropy", "heat_capacity") else e_unit

    pivot = phase_df.pivot_table(
        values=property_name, index="temperature",
        columns="concentration", aggfunc="mean",
    )
    label = property_name.replace("_", " ").title()
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale="RdYlBu_r",
        colorbar=dict(title=f"{label} ({unit})"),
        hovertemplate=(
            f"{selected_element} conc: %{{x:.1f}}%<br>"
            f"T: %{{y}} K<br>"
            f"{label}: %{{z:.4f}} {unit}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=f"{label} vs {selected_element} Concentration and Temperature",
        xaxis_title=f"{selected_element} Concentration (%)",
        yaxis_title="Temperature (K)",
        height=500,
    )
    return fig


def export_phase_diagram_data(phase_df: "pd.DataFrame", stable_df: "pd.DataFrame",
                               phase_transitions: list, selected_element: str) -> str:
    import json as _json
    export = {
        "metadata": {
            "selected_element":  selected_element,
            "temperature_range": [int(phase_df["temperature"].min()),
                                   int(phase_df["temperature"].max())],
            "structures_analyzed": list(phase_df["structure"].unique()),
            "analysis_type": "computational_phase_diagram",
        },
        "phase_data":        phase_df.to_dict("records"),
        "stable_phases":     stable_df.to_dict("records"),
        "phase_transitions": phase_transitions,
    }
    return _json.dumps(export, indent=2, default=str)
