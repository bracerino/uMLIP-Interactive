import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import threading
import queue
import time
import io
import zipfile
from pathlib import Path
from plotly.subplots import make_subplots
import psutil
import GPUtil
import streamlit as st
import threading
import time

from helpers.phonons_help import *
from helpers.generate_python_code import *
from helpers.phase_diagram import *
from helpers.monitor_resources import *


import py3Dmol
import streamlit.components.v1 as components
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from ase.io import read, write
from ase import Atoms

try:
    from ase.optimize import BFGS, LBFGS
    from ase.constraints import FixAtoms, ExpCellFilter, UnitCellFilter
    from ase.stress import voigt_6_to_full_3x3_stress

    CELL_OPT_AVAILABLE = True
except ImportError:
    CELL_OPT_AVAILABLE = False

MACE_IMPORT_METHOD = None
MACE_AVAILABLE = False

import numpy as np
from ase.phonons import Phonons
from ase.dft.kpoints import bandpath
import json

try:
    from ase.eos import EquationOfState
    from ase.build import bulk

    ELASTIC_AVAILABLE = True
except ImportError:
    ELASTIC_AVAILABLE = False

if 'structures_locked' not in st.session_state:
    st.session_state.structures_locked = False
if 'uploaded_files_processed' not in st.session_state:
    st.session_state.uploaded_files_processed = False
if 'pending_structures' not in st.session_state:
    st.session_state.pending_structures = {}
if 'computation_times' not in st.session_state:
    st.session_state.computation_times = {}
if 'structure_start_times' not in st.session_state:
    st.session_state.structure_start_times = {}
if 'total_calculation_start_time' not in st.session_state:
    st.session_state.total_calculation_start_time = None

st.markdown("""
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #0099ff; color: white; font-size: 16px; font-weight: bold;
        padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
    }
    div.stButton > button[kind="primary"]:active, div.stButton > button[kind="primary"]:focus {
        background-color: #007acc !important; color: white !important; box-shadow: none !important;
    }

    div.stButton > button[kind="secondary"] {
        background-color: #dc3545; color: white; font-size: 16px; font-weight: bold;
        padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
    }
    div.stButton > button[kind="secondary"]:active, div.stButton > button[kind="secondary"]:focus {
        background-color: #c82333 !important; color: white !important; box-shadow: none !important;
    }

    div.stButton > button[kind="tertiary"] {
        background-color: #6f42c1; color: white; font-size: 16px; font-weight: bold;
        padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
    }
    div.stButton > button[kind="tertiary"]:active, div.stButton > button[kind="tertiary"]:focus {
        background-color: #5a2d91 !important; color: white !important; box-shadow: none !important;
    }

    div[data-testid="stDataFrameContainer"] table td { font-size: 16px !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)




# Monitor CPU, GPU, RAM
display_system_monitoring_detailed()



def estimate_phonon_supercell(atoms, target_min_length=15.0, max_supercell=4, log_queue=None):
    cell = atoms.get_cell()
    cell_lengths = np.linalg.norm(cell, axis=1)  # |a|, |b|, |c|

    if log_queue:
        log_queue.put(
            f"  Unit cell lengths: a={cell_lengths[0]:.3f} Å, b={cell_lengths[1]:.3f} Å, c={cell_lengths[2]:.3f} Å")
    supercell_multipliers = []
    for length in cell_lengths:
        multiplier = max(1, int(np.ceil(target_min_length / length)))
        multiplier = min(multiplier, max_supercell)
        supercell_multipliers.append(multiplier)
    num_atoms = len(atoms)

    if num_atoms < 5:
        supercell_multipliers = [min(max_supercell, max(2, m)) for m in supercell_multipliers]

    elif num_atoms > 50:
        supercell_multipliers = [max(1, min(2, m)) for m in supercell_multipliers]

    supercell_multipliers = [max(1, m) for m in supercell_multipliers]

    supercell_size = tuple(supercell_multipliers)
    total_atoms_in_supercell = num_atoms * np.prod(supercell_size)

    if log_queue:
        log_queue.put(f"  Estimated supercell: {supercell_size}")
        log_queue.put(
            f"  Supercell lengths: {cell_lengths[0] * supercell_size[0]:.1f} × {cell_lengths[1] * supercell_size[1]:.1f} × {cell_lengths[2] * supercell_size[2]:.1f} Å")
        log_queue.put(f"  Total atoms in supercell: {total_atoms_in_supercell}")

        if total_atoms_in_supercell > 500:
            log_queue.put(f"  ⚠️ Warning: Large supercell ({total_atoms_in_supercell} atoms) - calculation may be slow")

    return supercell_size


def wrap_positions_in_cell(atoms):
    wrapped_atoms = atoms.copy()
    fractional_coords = wrapped_atoms.get_scaled_positions()
    wrapped_fractional = fractional_coords % 1.0
    wrapped_atoms.set_scaled_positions(wrapped_fractional)

    return wrapped_atoms


def load_structure(file):
    try:
        file_content = file.read()
        file.seek(0)

        with open(file.name, "wb") as f:
            f.write(file_content)

        filename = file.name.lower()

        if filename.endswith(".cif"):
            mg_structure = Structure.from_file(file.name)
        elif filename.endswith(".data"):
            lmp_filename = file.name.replace(".data", ".lmp")
            os.rename(file.name, lmp_filename)
            lammps_data = LammpsData.from_file(lmp_filename, atom_style="atomic")
            mg_structure = lammps_data.structure
        elif filename.endswith(".lmp"):
            lammps_data = LammpsData.from_file(file.name, atom_style="atomic")
            mg_structure = lammps_data.structure
        else:
            atoms = read(file.name)
            mg_structure = AseAtomsAdaptor.get_structure(atoms)

        if os.path.exists(file.name):
            os.remove(file.name)

        return mg_structure

    except Exception as e:
        st.error(f"Failed to parse {file.name}: {e}")
        st.error(
            f"Failed to load structure file. Supported formats: CIF, POSCAR, LMP, XSF, PW, CFG, and other ASE-compatible formats. "
            f"Please check your file format and try again. 😊")
        if os.path.exists(file.name):
            os.remove(file.name)
        raise e

def ase_to_pymatgen_wrapped(atoms):
    wrapped_atoms = wrap_positions_in_cell(atoms)
    structure = Structure(
        lattice=wrapped_atoms.cell[:],
        species=[atom.symbol for atom in wrapped_atoms],
        coords=wrapped_atoms.positions,
        coords_are_cartesian=True
    )

    return structure


def create_wrapped_poscar_content(structure):
    wrapped_structure = structure.copy()
    frac_coords = wrapped_structure.frac_coords
    wrapped_frac_coords = frac_coords % 1.0
    wrapped_structure = Structure(
        lattice=wrapped_structure.lattice,
        species=wrapped_structure.species,
        coords=wrapped_frac_coords,
        coords_are_cartesian=False
    )

    return wrapped_structure.to(fmt="poscar")


def calculate_elastic_properties(atoms, calculator, elastic_params, log_queue, structure_name):
    try:
        log_queue.put(f"Starting elastic tensor calculation for {structure_name}")


        atoms.calc = calculator
        log_queue.put("  Calculating equilibrium energy and stress...")
        E0 = atoms.get_potential_energy()
        stress0 = atoms.get_stress(voigt=True)  # Voigt notation: [xx, yy, zz, yz, xz, xy]

        log_queue.put(f"  Equilibrium energy: {E0:.6f} eV")
        log_queue.put(f"  Equilibrium stress: {np.max(np.abs(stress0)):.6f} GPa")

        strain_magnitude = elastic_params.get('strain_magnitude', 0.01)  # 1% strain
        log_queue.put(f"  Using strain magnitude: {strain_magnitude * 100:.1f}%")

        # Initialize elastic tensor (6x6 in Voigt notation)
        C = np.zeros((6, 6))
        log_queue.put("  Applying strains and calculating stress response...")

        original_cell = atoms.get_cell().copy()
        volume = atoms.get_volume()

        strain_tensors = []

        for i in range(3):
            strain = np.zeros((3, 3))
            strain[i, i] = strain_magnitude
            strain_tensors.append(strain)

        shear_pairs = [(1, 2), (0, 2), (0, 1)]  # (yz, xz, xy)
        for i, j in shear_pairs:
            strain = np.zeros((3, 3))
            strain[i, j] = strain[j, i] = strain_magnitude / 2  # Engineering shear strain
            strain_tensors.append(strain)

        for strain_idx, strain_tensor in enumerate(strain_tensors):
            log_queue.put(f"    Strain {strain_idx + 1}/6...")
            deformed_cell = original_cell @ (np.eye(3) + strain_tensor)
            atoms.set_cell(deformed_cell, scale_atoms=True)
            stress_pos = atoms.get_stress(voigt=True)
            deformed_cell = original_cell @ (np.eye(3) - strain_tensor)
            atoms.set_cell(deformed_cell, scale_atoms=True)
            stress_neg = atoms.get_stress(voigt=True)


            # C_ij = d(stress_i)/d(strain_j)
            stress_derivative = (stress_pos - stress_neg) / (2 * strain_magnitude)

            C[strain_idx, :] = stress_derivative
            atoms.set_cell(original_cell, scale_atoms=True)

        # Convert from eV/Å³ to GPa
        eV_to_GPa = 160.2176  # Conversion factor
        C_GPa = C * eV_to_GPa

        log_queue.put("  Calculating elastic moduli...")

        K_voigt = (C_GPa[0, 0] + C_GPa[1, 1] + C_GPa[2, 2] + 2 * (C_GPa[0, 1] + C_GPa[0, 2] + C_GPa[1, 2])) / 9

        G_voigt = (C_GPa[0, 0] + C_GPa[1, 1] + C_GPa[2, 2] - C_GPa[0, 1] - C_GPa[0, 2] - C_GPa[1, 2] + 3 * (
                C_GPa[3, 3] + C_GPa[4, 4] + C_GPa[5, 5])) / 15

        try:
            S_GPa = np.linalg.inv(C_GPa)

            # Bulk modulus (Reuss)
            K_reuss = 1 / (S_GPa[0, 0] + S_GPa[1, 1] + S_GPa[2, 2] + 2 * (S_GPa[0, 1] + S_GPa[0, 2] + S_GPa[1, 2]))

            # Shear modulus (Reuss)
            G_reuss = 15 / (4 * (S_GPa[0, 0] + S_GPa[1, 1] + S_GPa[2, 2]) - 4 * (
                    S_GPa[0, 1] + S_GPa[0, 2] + S_GPa[1, 2]) + 3 * (S_GPa[3, 3] + S_GPa[4, 4] + S_GPa[5, 5]))

            K_hill = (K_voigt + K_reuss) / 2
            G_hill = (G_voigt + G_reuss) / 2

        except np.linalg.LinAlgError:
            log_queue.put("  ⚠️ Warning: Elastic tensor is singular - using Voigt averages only")
            K_reuss = G_reuss = K_hill = G_hill = None
            S_GPa = None

        K = K_hill if K_hill is not None else K_voigt
        G = G_hill if G_hill is not None else G_voigt

        # Young's modulus
        E = (9 * K * G) / (3 * K + G)

        # Poisson's ratio
        nu = (3 * K - 2 * G) / (2 * (3 * K + G))

        # Wave velocities
        density = elastic_params.get('density', None)
        if density is None:
            # Estimate density from structure if not provided by user
            total_mass_amu = np.sum(atoms.get_masses())  # amu
            # Convert atomic mass units to grams (1 amu = 1.66053906660 × 10^-24 g)
            # Convert Å^3 to cm^3 (1 Å^3 = 10^-24 cm^3)
            # Density (g/cm^3) = (total_mass_amu * 1.66053906660e-24 g/amu) / (volume_A3 * 1e-24 cm^3/A3)
            # Simplifies to: density = total_mass_amu * 1.66053906660 / volume_A3
            density = (total_mass_amu * 1.66053906660) / volume  # amu / A^3 to g/cm^3
            log_queue.put(f"  Estimated density from structure: {density:.3f} g/cm³")
        else:
            log_queue.put(f"  Using user-provided density: {density:.3f} g/cm³")

        density_kg_m3 = density * 1000

        v_l = np.sqrt((K + 4 * G / 3) * 1e9 / density_kg_m3)  # m/s
        v_t = np.sqrt(G * 1e9 / density_kg_m3)  # m/s

        v_avg = ((1 / v_l ** 3 + 2 / v_t ** 3) / 3) ** (-1 / 3)

        h = 6.626e-34  # J⋅s
        kB = 1.381e-23  # J/K
        N_atoms = len(atoms)
        total_mass_kg = np.sum(atoms.get_masses()) * 1.66054e-27
        theta_D = (h / kB) * v_avg * (3 * N_atoms * density_kg_m3 / (4 * np.pi * total_mass_kg)) ** (1 / 3)

        stability_criteria = check_mechanical_stability(C_GPa, log_queue)

        log_queue.put(f"✅ Elastic calculation completed for {structure_name}")
        log_queue.put(f"  Bulk modulus: {K:.1f} GPa")
        log_queue.put(f"  Shear modulus: {G:.1f} GPa")
        log_queue.put(f"  Young's modulus: {E:.1f} GPa")
        log_queue.put(f"  Poisson's ratio: {nu:.3f}")

        return {
            'success': True,
            'elastic_tensor': C_GPa.tolist(),  # 6x6 matrix in GPa
            'compliance_matrix': S_GPa.tolist() if S_GPa is not None else None,

            'bulk_modulus': {
                'voigt': K_voigt,
                'reuss': K_reuss,
                'hill': K_hill
            },
            'shear_modulus': {
                'voigt': G_voigt,
                'reuss': G_reuss,
                'hill': G_hill
            },
            'youngs_modulus': E,
            'poisson_ratio': nu,
            'wave_velocities': {
                'longitudinal': v_l,
                'transverse': v_t,
                'average': v_avg
            },
            'debye_temperature': theta_D,
            'density': density,
            'mechanical_stability': stability_criteria,
            'strain_magnitude': strain_magnitude
        }

    except Exception as e:
        log_queue.put(f"❌ Elastic calculation failed for {structure_name}: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


def check_mechanical_stability(C, log_queue):
    try:
        criteria = {}

        eigenvals = np.linalg.eigvals(C)
        all_positive = bool(np.all(eigenvals > 0))
        criteria['positive_definite'] = all_positive

        criteria['C11_positive'] = bool(C[0, 0] > 0)
        criteria['C22_positive'] = bool(C[1, 1] > 0)
        criteria['C33_positive'] = bool(C[2, 2] > 0)
        criteria['C44_positive'] = bool(C[3, 3] > 0)
        criteria['C55_positive'] = bool(C[4, 4] > 0)
        criteria['C66_positive'] = bool(C[5, 5] > 0)

        criteria['det_positive'] = bool(np.linalg.det(C) > 0)

        if abs(C[0, 0] - C[1, 1]) < 1 and abs(C[1, 1] - C[2, 2]) < 1:
            criteria['cubic_C11_C12'] = bool(C[0, 0] > abs(C[0, 1]))
            criteria['cubic_C44'] = bool(C[3, 3] > 0)
            criteria['cubic_bulk'] = bool((C[0, 0] + 2 * C[0, 1]) > 0)

        stable = bool(all_positive and criteria['det_positive'])
        criteria['mechanically_stable'] = stable

        if stable:
            log_queue.put("  ✅ Crystal is mechanically stable")
        else:
            log_queue.put("  ❌ Crystal may be mechanically unstable")
            log_queue.put(f"    Negative eigenvalues: {eigenvals[eigenvals <= 0]}")

        return criteria

    except Exception as e:
        log_queue.put(f"  ⚠️ Error checking stability: {str(e)}")
        return {'error': str(e)}


def create_phonon_data_export(phonon_results, structure_name):
    if not phonon_results['success']:
        return None

    export_data = {
        'structure_name': structure_name,
        'frequencies_meV': phonon_results['frequencies'].tolist(),
        'kpoints': phonon_results['kpoints'].tolist(),
        'dos_energies_meV': phonon_results['dos_energies'].tolist(),
        'dos': phonon_results['dos'].tolist(),
        'supercell_size': list(phonon_results['supercell_size']),  # Convert tuple to list
        'imaginary_modes': int(phonon_results['imaginary_modes']),  # Convert numpy int to Python int
        'min_frequency_meV': float(phonon_results['min_frequency'])  # Convert numpy float to Python float
    }

    if phonon_results['thermodynamics']:
        thermo = phonon_results['thermodynamics']
        export_data['thermodynamics'] = {
            'temperature': float(thermo['temperature']),
            'zero_point_energy': float(thermo['zero_point_energy']),
            'internal_energy': float(thermo['internal_energy']),
            'heat_capacity': float(thermo['heat_capacity']),
            'entropy': float(thermo['entropy']),
            'free_energy': float(thermo['free_energy'])
        }

    return export_data


def create_elastic_data_export(elastic_results, structure_name):
    if not elastic_results['success']:
        return None

    export_data = {
        'structure_name': structure_name,
        'elastic_tensor_GPa': elastic_results['elastic_tensor'],
        'bulk_modulus_GPa': {
            'voigt': float(elastic_results['bulk_modulus']['voigt']) if elastic_results['bulk_modulus'][
                                                                            'voigt'] is not None else None,
            'reuss': float(elastic_results['bulk_modulus']['reuss']) if elastic_results['bulk_modulus'][
                                                                            'reuss'] is not None else None,
            'hill': float(elastic_results['bulk_modulus']['hill']) if elastic_results['bulk_modulus'][
                                                                          'hill'] is not None else None
        },
        'shear_modulus_GPa': {
            'voigt': float(elastic_results['shear_modulus']['voigt']) if elastic_results['shear_modulus'][
                                                                             'voigt'] is not None else None,
            'reuss': float(elastic_results['shear_modulus']['reuss']) if elastic_results['shear_modulus'][
                                                                             'reuss'] is not None else None,
            'hill': float(elastic_results['shear_modulus']['hill']) if elastic_results['shear_modulus'][
                                                                           'hill'] is not None else None
        },
        'youngs_modulus_GPa': float(elastic_results['youngs_modulus']),
        'poisson_ratio': float(elastic_results['poisson_ratio']),
        'wave_velocities_ms': {
            'longitudinal': float(elastic_results['wave_velocities']['longitudinal']),
            'transverse': float(elastic_results['wave_velocities']['transverse']),
            'average': float(elastic_results['wave_velocities']['average'])
        },
        'debye_temperature_K': float(elastic_results['debye_temperature']),
        'density_g_cm3': float(elastic_results['density']),
        'mechanical_stability': convert_stability_to_json(elastic_results['mechanical_stability'])
    }

    return export_data


def convert_stability_to_json(stability_dict):
    json_stability = {}
    for key, value in stability_dict.items():
        if isinstance(value, (np.bool_, bool)):
            json_stability[key] = bool(value)
        elif isinstance(value, (np.integer, int)):
            json_stability[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            json_stability[key] = float(value)
        elif isinstance(value, np.ndarray):
            json_stability[key] = value.tolist()
        elif value is None:
            json_stability[key] = None
        else:
            json_stability[key] = value

    return json_stability


def create_elastic_data_export(elastic_results, structure_name):
    if not elastic_results['success']:
        return None

    return {
        'structure_name': structure_name,
        'elastic_tensor_GPa': elastic_results['elastic_tensor'],
        'bulk_modulus_GPa': elastic_results['bulk_modulus'],
        'shear_modulus_GPa': elastic_results['shear_modulus'],
        'youngs_modulus_GPa': elastic_results['youngs_modulus'],
        'poisson_ratio': elastic_results['poisson_ratio'],
        'wave_velocities_ms': elastic_results['wave_velocities'],
        'debye_temperature_K': elastic_results['debye_temperature'],
        'density_g_cm3': elastic_results['density'],
        'mechanical_stability': elastic_results['mechanical_stability']
    }


def calculate_phonons_pymatgen(atoms, calculator, phonon_params, log_queue, structure_name):
    try:
        log_queue.put(f"Starting Pymatgen+Phonopy phonon calculation for {structure_name}")

        try:
            from phonopy import Phonopy
            from phonopy.structure.atoms import PhonopyAtoms
            from pymatgen.io.phonopy import get_phonopy_structure
            from pymatgen.phonon.bandstructure import PhononBandStructure
            from pymatgen.phonon.dos import PhononDos
            import phonopy.units as units_phonopy
        except ImportError as e:
            log_queue.put(f"❌ Missing dependencies: {str(e)}")
            log_queue.put("Please install: pip install phonopy")
            return {'success': False, 'error': f'Missing phonopy: {str(e)}'}

        atoms.calc = calculator

        num_initial_atoms = len(atoms)
        log_queue.put(f"  Primitive cell has {num_initial_atoms} atoms")

        log_queue.put("  Running brief pre-phonon optimization...")
        try:
            from ase.optimize import LBFGS
            temp_atoms = atoms.copy()
            temp_atoms.calc = calculator
            temp_optimizer = LBFGS(temp_atoms, logfile=None)
            temp_optimizer.run(fmax=0.01, steps=50)
            atoms = temp_atoms
            energy = atoms.get_potential_energy()
            max_force = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
            log_queue.put(f"  Pre-optimization: E={energy:.6f} eV, F_max={max_force:.4f} eV/Å")
        except Exception as opt_error:
            log_queue.put(f"  ⚠️ Pre-optimization failed: {str(opt_error)}")

        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()
        pmg_structure = adaptor.get_structure(atoms)
        log_queue.put(f"  Converted to pymatgen structure: {pmg_structure.composition}")

        phonopy_atoms = PhonopyAtoms(
            symbols=[str(site.specie) for site in pmg_structure],
            scaled_positions=pmg_structure.frac_coords,
            cell=pmg_structure.lattice.matrix
        )

        if phonon_params.get('auto_supercell', True):
            log_queue.put("  Auto-determining supercell size...")

            a, b, c = pmg_structure.lattice.abc
            target_length = phonon_params.get('target_supercell_length', 15.0)
            max_multiplier = phonon_params.get('max_supercell_multiplier', 4)


            na = max(1, min(max_multiplier, int(np.ceil(target_length / a))))
            nb = max(1, min(max_multiplier, int(np.ceil(target_length / b))))
            nc = max(1, min(max_multiplier, int(np.ceil(target_length / c))))

            if num_initial_atoms > 50:
                na = max(1, na - 1)
                nb = max(1, nb - 1)
                nc = max(1, nc - 1)

            supercell_matrix = [[na, 0, 0], [0, nb, 0], [0, 0, nc]]
        else:

            sc = phonon_params.get('supercell_size', (2, 2, 2))
            supercell_matrix = [[sc[0], 0, 0], [0, sc[1], 0], [0, 0, sc[2]]]

        total_atoms = num_initial_atoms * np.prod([supercell_matrix[i][i] for i in range(3)])
        log_queue.put(f"  Supercell matrix: {supercell_matrix}")
        log_queue.put(f"  Total atoms in supercell: {total_atoms}")

        max_atoms = phonon_params.get('max_supercell_atoms', 800)
        if total_atoms > max_atoms:
            log_queue.put(f"  ⚠️ Supercell too large ({total_atoms} atoms), using smaller supercell")
            supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            total_atoms = num_initial_atoms

        log_queue.put("  Initializing Phonopy...")
        phonon = Phonopy(
            phonopy_atoms,
            supercell_matrix=supercell_matrix,
            primitive_matrix='auto'
        )

        displacement_distance = phonon_params.get('delta', 0.01)
        log_queue.put(f"  Generating displacements (distance={displacement_distance} Å)...")
        phonon.generate_displacements(distance=displacement_distance)

        supercells = phonon.get_supercells_with_displacements()
        log_queue.put(f"  Generated {len(supercells)} displaced supercells")

        log_queue.put("  Calculating forces for displaced supercells...")
        forces = []

        for i, supercell in enumerate(supercells):
            log_queue.put(f"    Calculating forces for supercell {i + 1}/{len(supercells)}")
            ase_supercell = Atoms(
                symbols=supercell.symbols,
                positions=supercell.positions,
                cell=supercell.cell,
                pbc=True
            )
            ase_supercell.calc = calculator

            try:
                supercell_forces = ase_supercell.get_forces()
                forces.append(supercell_forces)

                if (i + 1) % max(1, len(supercells) // 10) == 0:
                    progress = (i + 1) / len(supercells) * 100
                    log_queue.put(f"    Progress: {progress:.1f}% ({i + 1}/{len(supercells)})")

            except Exception as force_error:
                log_queue.put(f"    ❌ Force calculation failed for supercell {i + 1}: {str(force_error)}")
                return {'success': False, 'error': f'Force calculation failed: {str(force_error)}'}

        log_queue.put("  ✅ All force calculations completed")

        phonon.forces = forces
        log_queue.put("  Calculating force constants...")
        phonon.produce_force_constants()

        log_queue.put("  Calculating phonon band structure...")

        try:
            from pymatgen.symmetry.bandstructure import HighSymmKpath
            kpath = HighSymmKpath(pmg_structure)
            path = kpath.kpath["path"]
            kpoints = kpath.kpath["kpoints"]

            bands = []
            labels = []

            for segment in path:
                segment_points = []
                for point_name in segment:
                    segment_points.append(kpoints[point_name])
                bands.append(segment_points)
                labels.extend(segment)
            unique_labels = []
            unique_labels.append(labels[0])
            for i in range(1, len(labels)):
                if labels[i] != labels[i - 1]:
                    unique_labels.append(labels[i])

            log_queue.put(f"  Using high-symmetry path: {' → '.join(unique_labels)}")

        except Exception as path_error:
            log_queue.put(f"  ⚠️ High-symmetry path detection failed: {str(path_error)}")
            log_queue.put("  Using simple Γ-X-M-Γ path")
            bands = [[[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]]]
            unique_labels = ['Γ', 'X', 'M', 'Γ']

        npoints = phonon_params.get('npoints', 101)
        phonon.run_band_structure(
            bands,
            is_band_connection=False,
            with_eigenvectors=False,
            is_legacy_plot=False
        )
        log_queue.put("  Processing band structure data...")
        band_dict = phonon.get_band_structure_dict()

        raw_frequencies = band_dict['frequencies']
        log_queue.put(f"  Raw frequencies type: {type(raw_frequencies)}")
        log_queue.put(f"  Raw frequencies length: {len(raw_frequencies)}")

        if isinstance(raw_frequencies, list):
            freq_arrays = []
            max_bands = 0
            for i, freq_array in enumerate(raw_frequencies):
                freq_np = np.array(freq_array)
                freq_arrays.append(freq_np)
                if freq_np.ndim == 1:
                    max_bands = max(max_bands, len(freq_np))
                elif freq_np.ndim == 2:
                    max_bands = max(max_bands, freq_np.shape[1])

            log_queue.put(f"  Found {len(freq_arrays)} k-point groups with max {max_bands} bands")

            total_kpoints = sum(len(freq_array) if freq_array.ndim > 0 else 1 for freq_array in freq_arrays)
            frequencies = np.full((total_kpoints, max_bands), np.nan)

            kpoint_idx = 0
            for freq_array in freq_arrays:
                freq_np = np.array(freq_array)
                if freq_np.ndim == 1:
                    n_bands = len(freq_np)
                    frequencies[kpoint_idx, :n_bands] = freq_np
                    kpoint_idx += 1
                elif freq_np.ndim == 2:
                    n_kpts, n_bands = freq_np.shape
                    frequencies[kpoint_idx:kpoint_idx + n_kpts, :n_bands] = freq_np
                    kpoint_idx += n_kpts

        else:

            frequencies = np.array(raw_frequencies)

        # Convert units: THz to meV
        frequencies = frequencies * units_phonopy.THzToEv * 1000  # Convert to meV

        valid_frequencies = frequencies[~np.isnan(frequencies)]
        log_queue.put(f"  ✅ Band structure calculated: {frequencies.shape} (valid points: {len(valid_frequencies)})")


        raw_kpoints = band_dict['qpoints']
        log_queue.put(f"  Raw k-points type: {type(raw_kpoints)}")
        log_queue.put(f"  Raw k-points length: {len(raw_kpoints)}")

        if isinstance(raw_kpoints, list):
            kpoints_band = []
            for kpt_group in raw_kpoints:
                kpt_array = np.array(kpt_group)
                if kpt_array.ndim == 1:
                    kpoints_band.append(kpt_array)
                elif kpt_array.ndim == 2:
                    for kpt in kpt_array:
                        kpoints_band.append(kpt)
                else:
                    log_queue.put(f"  ⚠️ Unexpected k-point dimension: {kpt_array.ndim}")
            kpoints_band = np.array(kpoints_band)
        else:
            kpoints_band = np.array(raw_kpoints)

        log_queue.put(f"  Processed k-points shape: {kpoints_band.shape}")
        if len(kpoints_band) != frequencies.shape[0]:
            log_queue.put(f"  ⚠️ K-point count mismatch: {len(kpoints_band)} vs {frequencies.shape[0]}")
            min_len = min(len(kpoints_band), frequencies.shape[0])
            kpoints_band = kpoints_band[:min_len]
            frequencies = frequencies[:min_len]
            log_queue.put(f"  Adjusted to consistent length: {min_len}")

        log_queue.put("  Calculating phonon DOS...")
        try:
            if total_atoms > 100:
                mesh = [20, 20, 20]
            else:
                mesh = [30, 30, 30]

            log_queue.put(f"  Using DOS mesh: {mesh}")
            phonon.run_mesh(mesh)
            phonon.run_total_dos()

            dos_dict = phonon.get_total_dos_dict()
            dos_frequencies = dos_dict['frequency_points'] * units_phonopy.THzToEv * 1000  # Convert to meV
            dos_values = dos_dict['total_dos']

            log_queue.put(f"  ✅ DOS calculated with {len(dos_frequencies)} points")

        except Exception as dos_error:
            log_queue.put(f"  ⚠️ DOS calculation failed: {str(dos_error)}")
            freq_flat = valid_frequencies[valid_frequencies > 0]

            if len(freq_flat) > 0:
                dos_frequencies = np.linspace(0, np.max(freq_flat) * 1.2, 500)
                dos_values = np.zeros_like(dos_frequencies)
                sigma = 2.0  # meV
                for f in freq_flat:
                    dos_values += np.exp(-0.5 * ((dos_frequencies - f) / sigma) ** 2)
                dos_values /= len(freq_flat) * sigma * np.sqrt(2 * np.pi)
            else:
                dos_frequencies = np.linspace(0, 50, 500)
                dos_values = np.zeros_like(dos_frequencies)

        filtered_frequencies = filter_near_zero_frequencies(valid_frequencies)
        imaginary_count = np.sum(filtered_frequencies < 0)
        min_frequency = np.min(filtered_frequencies) if len(filtered_frequencies) > 0 else 0
        frequencies = filter_near_zero_frequencies(frequencies)  #

        if imaginary_count > 0:
            log_queue.put(f"  ⚠️ Found {imaginary_count} imaginary modes")
            log_queue.put(f"    Most negative frequency: {min_frequency:.3f} meV")
        else:
            log_queue.put("  ✅ No imaginary modes found")
        temp = phonon_params.get('temperature', 300)
        log_queue.put(f"  Calculating thermodynamics at {temp} K...")

        try:
            phonon.run_thermal_properties(
                t_step=10,
                t_max=1500,
                t_min=0
            )

            thermal_dict = phonon.get_thermal_properties_dict()
            temps = np.array(thermal_dict['temperatures'])
            temp_idx = np.argmin(np.abs(temps - temp))

            thermo_props = {
                'temperature': float(temps[temp_idx]),
                'zero_point_energy': float(thermal_dict['zero_point_energy']),  # eV
                'internal_energy': float(thermal_dict['internal_energy'][temp_idx]),  # eV
                'heat_capacity': float(thermal_dict['heat_capacity'][temp_idx]),  # eV/K
                'entropy': float(thermal_dict['entropy'][temp_idx]),  # eV/K
                'free_energy': float(thermal_dict['free_energy'][temp_idx])  # eV
            }

            log_queue.put(f"  Zero-point energy: {thermo_props['zero_point_energy']:.6f} eV")
            log_queue.put(f"  Heat capacity: {thermo_props['heat_capacity']:.6f} eV/K")

        except Exception as thermo_error:
            log_queue.put(f"  ⚠️ Thermodynamics calculation failed: {str(thermo_error)}")

            positive_freqs = valid_frequencies[valid_frequencies > 0] * 1e-3  # Convert to eV
            if len(positive_freqs) > 0:
                kB = 8.617e-5  # eV/K
                E_zp = 0.5 * np.sum(positive_freqs)

                x = positive_freqs / (kB * temp)
                x = np.minimum(x, 50)  # Prevent overflow
                exp_x = np.exp(x)

                U = np.sum(positive_freqs * exp_x / (exp_x - 1 + 1e-10))
                Cv = np.sum(kB * x ** 2 * exp_x / (exp_x - 1 + 1e-10) ** 2)

                thermo_props = {
                    'temperature': temp,
                    'zero_point_energy': E_zp,
                    'internal_energy': U,
                    'heat_capacity': Cv,
                    'entropy': 0.0,
                    'free_energy': U
                }
            else:
                thermo_props = None

        log_queue.put(f"✅ Pymatgen+Phonopy calculation completed for {structure_name}")

        return {
            'success': True,
            'frequencies': frequencies,  # meV, shape (nkpts, nbands)
            'kpoints': kpoints_band,
            'dos_energies': dos_frequencies,  # meV
            'dos': dos_values,
            'thermodynamics': thermo_props,
            'thermal_properties_dict': thermal_dict,  # ADD THIS LINE - full temperature data
            'supercell_size': tuple([supercell_matrix[i][i] for i in range(3)]),
            'imaginary_modes': int(imaginary_count),
            'min_frequency': float(min_frequency),
            'method': 'Pymatgen+Phonopy'
        }

    except Exception as e:
        log_queue.put(f"❌ Pymatgen+Phonopy phonon calculation failed for {structure_name}: {str(e)}")
        log_queue.put(f"Error type: {type(e).__name__}")
        import traceback
        log_queue.put(f"Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e)
        }


def check_and_install_phonopy():
    try:
        import phonopy
        return True, "Phonopy is available"
    except ImportError:
        return False, "Phonopy not found. Please install with: pip install phonopy"


try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.io.ase import AseAtomsAdaptor
    import phonopy.units as units_phonopy

    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False
    print("⚠️ Phonopy not available for phonon calculations")

try:
    from mace.calculators import mace_mp

    MACE_IMPORT_METHOD = "mace_mp"
    MACE_AVAILABLE = True
except ImportError:
    try:
        from mace.calculators import MACECalculator

        MACE_IMPORT_METHOD = "MACECalculator"
        MACE_AVAILABLE = True
    except ImportError:
        try:
            import mace
            from mace.calculators import MACECalculator

            MACE_IMPORT_METHOD = "MACECalculator"
            MACE_AVAILABLE = True
        except ImportError:
            MACE_AVAILABLE = False


class OptimizationLogger:
    def __init__(self, log_queue, structure_name):
        self.log_queue = log_queue
        self.structure_name = structure_name
        self.step_count = 0
        self.trajectory = []
        self.previous_energy = None

    def __call__(self, optimizer=None):
        if optimizer is not None and hasattr(optimizer, 'atoms'):
            atoms = optimizer.atoms
            self.step_count += 1
            forces = atoms.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))
            energy = atoms.get_potential_energy()

            energy_change = abs(energy - self.previous_energy) if self.previous_energy is not None else float('inf')
            self.previous_energy = energy

            trajectory_step = {
                'step': self.step_count,
                'energy': energy,
                'max_force': max_force,
                'energy_change': energy_change,
                'positions': atoms.positions.copy(),
                'cell': atoms.cell.array.copy(),
                'symbols': atoms.get_chemical_symbols(),
                'forces': forces.copy()
            }
            self.trajectory.append(trajectory_step)

            self.log_queue.put(
                f"  Step {self.step_count}: Energy = {energy:.6f} eV, Max Force = {max_force:.4f} eV/Å, ΔE = {energy_change:.2e} eV")
            self.log_queue.put({
                'type': 'opt_step',
                'structure': self.structure_name,
                'step': self.step_count,
                'energy': energy,
                'max_force': max_force,
                'energy_change': energy_change,
                'total_steps': None
            })

            self.log_queue.put({
                'type': 'trajectory_step',
                'structure': self.structure_name,
                'step': self.step_count,
                'trajectory_data': trajectory_step
            })


def create_xyz_content(trajectory_data, structure_name):
    xyz_content = ""

    for step_data in trajectory_data:
        step = step_data['step']
        energy = step_data['energy']
        positions = step_data['positions']
        cell = step_data['cell']
        symbols = step_data['symbols']
        forces = step_data['forces']
        max_force = step_data['max_force']

        xyz_content += f"{len(positions)}\n"

        a, b, c = np.linalg.norm(cell, axis=1)
        alpha = np.degrees(np.arccos(np.dot(cell[1], cell[2]) / (b * c)))
        beta = np.degrees(np.arccos(np.dot(cell[0], cell[2]) / (a * c)))
        gamma = np.degrees(np.arccos(np.dot(cell[0], cell[1]) / (a * b)))

        comment = (f"Step {step} | Energy={energy:.6f} eV | Max_Force={max_force:.4f} eV/A | "
                   f"Lattice=\"{a:.6f} {b:.6f} {c:.6f} {alpha:.2f} {beta:.2f} {gamma:.2f}\" | "
                   f"Properties=species:S:1:pos:R:3:forces:R:3")
        xyz_content += f"{comment}\n"

        for i, (symbol, pos, force) in enumerate(zip(symbols, positions, forces)):
            xyz_content += f"{symbol} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f} {force[0]:12.6f} {force[1]:12.6f} {force[2]:12.6f}\n"

    return xyz_content


MACE_MODELS = {
    "MACE-MP-0 (small)": "small",
    "MACE-MP-0 (medium)": "medium",
    "MACE-MP-0 (large)": "large"
}
PHONON_ZERO_THRESHOLD = 0.001  # meV


def filter_near_zero_frequencies(frequencies, threshold=PHONON_ZERO_THRESHOLD):

    filtered = frequencies.copy()
    mask = np.abs(filtered) < threshold
    filtered[mask] = 0.0
    return filtered


MACE_ELEMENTS = {
    "MACE-MP-0": ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
                  "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
                  "Kr",
                  "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
                  "Xe",
                  "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                  "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra",
                  "Ac", "Th", "Pa", "U", "Np", "Pu"]
}


def view_structure(structure, height=300, width=400):
    try:
        cif_writer = CifWriter(structure)
        cif_string = str(cif_writer)

        view = py3Dmol.view(width=width, height=height)
        view.addModel(cif_string, 'cif')
        view.setStyle({'sphere': {'scale': 0.6}})
        view.addUnitCell()
        view.zoomTo()

        return view._make_html()
    except:
        return f"<div style='height:{height}px;width:{width}px;background-color:#f0f0f0;display:flex;align-items:center;justify-content:center;'>Structure preview unavailable</div>"


def check_mace_compatibility(structure, model_name="MACE-MP-0"):
    elements = list(set([site.specie.symbol for site in structure]))
    supported_elements = MACE_ELEMENTS[model_name]
    unsupported = [elem for elem in elements if elem not in supported_elements]
    return len(unsupported) == 0, unsupported, elements


def pymatgen_to_ase(structure):
    atoms = Atoms(
        symbols=[str(site.specie) for site in structure],
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True
    )

    # Wrap positions to ensure they're within the cell
    return wrap_positions_in_cell(atoms)


def calculate_atomic_reference_energies(unique_elements, calculator, log_queue):
    reference_energies = {}

    log_queue.put("Calculating atomic reference energies...")

    for element in unique_elements:
        try:
            log_queue.put(f"  Calculating reference energy for {element}...")

            atom = Atoms(element, positions=[(0, 0, 0)], cell=[20, 20, 20], pbc=True)
            atom.calc = calculator

            energy = atom.get_potential_energy()
            reference_energies[element] = energy

            log_queue.put(f"  ✅ {element}: {energy:.6f} eV")

        except Exception as e:
            log_queue.put(f"  ❌ Failed to calculate reference energy for {element}: {str(e)}")
            reference_energies[element] = None

    return reference_energies


def get_atomic_concentrations(structure):
    element_counts = {}
    total_atoms = len(structure)

    for site in structure:
        element = site.specie.symbol
        element_counts[element] = element_counts.get(element, 0) + 1

    concentrations = {}
    for element, count in element_counts.items():
        concentrations[element] = (count / total_atoms) * 100

    return concentrations


def get_all_elements_from_results(results):
    all_elements = set()
    for result in results:
        if 'structure' in result and result['structure']:
            for site in result['structure']:
                all_elements.add(site.specie.symbol)
    return sorted(list(all_elements))


def calculate_formation_energy(structure_energy, structure, reference_energies):
    if structure_energy is None:
        return None

    element_counts = {}
    for site in structure:
        element = site.specie.symbol
        element_counts[element] = element_counts.get(element, 0) + 1

    total_reference_energy = 0
    for element, count in element_counts.items():
        if element not in reference_energies or reference_energies[element] is None:
            return None
        total_reference_energy += count * reference_energies[element]

    total_atoms = sum(element_counts.values())
    formation_energy_per_atom = (structure_energy - total_reference_energy) / total_atoms

    return formation_energy_per_atom


def create_cell_filter(atoms, optimization_params):
    pressure = optimization_params.get('pressure', 0.0)
    cell_constraint = optimization_params.get('cell_constraint', 'Lattice parameters only (fix angles)')
    optimize_lattice = optimization_params.get('optimize_lattice', {'a': True, 'b': True, 'c': True})
    hydrostatic = optimization_params.get('hydrostatic_strain', False)

    pressure_eV_A3 = pressure * 0.00624150913

    if cell_constraint == "Full cell (lattice + angles)":
        if hydrostatic:
            return ExpCellFilter(atoms, scalar_pressure=pressure_eV_A3, hydrostatic_strain=True)
        else:
            return ExpCellFilter(atoms, scalar_pressure=pressure_eV_A3)
    else:
        if hydrostatic:
            return UnitCellFilter(atoms, scalar_pressure=pressure_eV_A3, hydrostatic_strain=True)
        else:
            mask = [optimize_lattice['a'], optimize_lattice['b'], optimize_lattice['c'], False, False, False]
            return UnitCellFilter(atoms, mask=mask, scalar_pressure=pressure_eV_A3)


def setup_optimization_constraints(atoms, optimization_params):
    opt_type = optimization_params.get('optimization_type', 'Both atoms and cell')

    if opt_type == "Atoms only (fixed cell)":
        return atoms, None
    elif opt_type == "Cell only (fixed atoms)":
        atoms.set_constraint(FixAtoms(mask=[True] * len(atoms)))
        cell_filter = create_cell_filter(atoms, optimization_params)
        return cell_filter, "cell_only"
    else:
        cell_filter = create_cell_filter(atoms, optimization_params)
        return cell_filter, "both"


class CellOptimizationLogger:

    def __init__(self, log_queue, structure_name, opt_mode="both"):
        self.log_queue = log_queue
        self.structure_name = structure_name
        self.step_count = 0
        self.trajectory = []
        self.previous_energy = None
        self.opt_mode = opt_mode

    def __call__(self, optimizer=None):
        if optimizer is not None:
            if hasattr(optimizer.atoms, 'atoms'):
                atoms = optimizer.atoms.atoms
            else:
                atoms = optimizer.atoms

            self.step_count += 1

            try:
                forces = atoms.get_forces()
                max_force = np.max(np.linalg.norm(forces, axis=1))
                energy = atoms.get_potential_energy()
                stress = None
                max_stress = 0.0
                if self.opt_mode in ["cell_only", "both"]:
                    try:
                        stress_voigt = atoms.get_stress(voigt=True)
                        stress = stress_voigt
                        max_stress = np.max(np.abs(stress_voigt))
                    except Exception as e:
                        self.log_queue.put(f"  Warning: Could not get stress: {str(e)}")

                energy_change = abs(energy - self.previous_energy) if self.previous_energy is not None else float('inf')
                self.previous_energy = energy

                trajectory_step = {
                    'step': self.step_count,
                    'energy': energy,
                    'max_force': max_force,
                    'max_stress': max_stress,
                    'energy_change': energy_change,
                    'positions': atoms.positions.copy(),
                    'cell': atoms.cell.array.copy(),
                    'symbols': atoms.get_chemical_symbols(),
                    'forces': forces.copy(),
                    'stress': stress.copy() if stress is not None else None
                }
                self.trajectory.append(trajectory_step)

                if self.opt_mode == "cell_only":
                    self.log_queue.put(
                        f"  Step {self.step_count}: Energy = {energy:.6f} eV, Max Stress = {max_stress:.4f} GPa, ΔE = {energy_change:.2e} eV")
                elif self.opt_mode == "both":
                    self.log_queue.put(
                        f"  Step {self.step_count}: Energy = {energy:.6f} eV, Max Force = {max_force:.4f} eV/Å, Max Stress = {max_stress:.4f} GPa, ΔE = {energy_change:.2e} eV")
                else:
                    self.log_queue.put(
                        f"  Step {self.step_count}: Energy = {energy:.6f} eV, Max Force = {max_force:.4f} eV/Å, ΔE = {energy_change:.2e} eV")

                self.log_queue.put({
                    'type': 'opt_step',
                    'structure': self.structure_name,
                    'step': self.step_count,
                    'energy': energy,
                    'max_force': max_force,
                    'max_stress': max_stress,
                    'energy_change': energy_change,
                    'total_steps': None
                })

                self.log_queue.put({
                    'type': 'trajectory_step',
                    'structure': self.structure_name,
                    'step': self.step_count,
                    'trajectory_data': trajectory_step
                })

            except Exception as e:
                self.log_queue.put(f"  Error in optimization step {self.step_count}: {str(e)}")


def run_mace_calculation(structure_data, calc_type, model_size, device, optimization_params, phonon_params,
                         elastic_params,
                         calc_formation_energy, log_queue, stop_event):
    import time
    try:
        total_start_time = time.time()
        log_queue.put({
            'type': 'total_start_time',
            'start_time': total_start_time
        })

        log_queue.put("Setting up MACE calculator...")
        log_queue.put(f"Using import method: {MACE_IMPORT_METHOD}")
        log_queue.put(f"Model size: {model_size}")
        log_queue.put(f"Device: {device}")

        calculator = None

        if MACE_IMPORT_METHOD == "mace_mp":
            try:
                log_queue.put(f"Initializing mace_mp calculator on {device}...")
                calculator = mace_mp(model=model_size, dispersion=False, default_dtype="float64", device=device)
                log_queue.put(f"✅ mace_mp calculator initialized successfully on {device}")
            except Exception as e:
                log_queue.put(f"❌ mace_mp initialization failed on {device}: {str(e)}")
                if device == "cuda":
                    log_queue.put("⚠️ GPU initialization failed, falling back to CPU...")
                    try:
                        calculator = mace_mp(model=model_size, dispersion=False, default_dtype="float64", device="cpu")
                        log_queue.put("✅ mace_mp calculator initialized successfully on CPU (fallback)")
                    except Exception as cpu_error:
                        log_queue.put(f"❌ CPU fallback also failed: {str(cpu_error)}")
                        log_queue.put("This might be due to model download issues or device compatibility")
                        return
                else:
                    log_queue.put("This might be due to model download issues or device compatibility")
                    return

        elif MACE_IMPORT_METHOD == "MACECalculator":
            log_queue.put("Warning: Using MACECalculator - you may need to provide model paths manually")
            try:
                calculator = MACECalculator(device=device)
                log_queue.put(f"✅ MACECalculator initialized on {device}")
            except Exception as e:
                log_queue.put(f"❌ MACECalculator initialization failed on {device}: {str(e)}")
                if device == "cuda":
                    log_queue.put("⚠️ GPU initialization failed, falling back to CPU...")
                    try:
                        calculator = MACECalculator(device="cpu")
                        log_queue.put("✅ MACECalculator initialized on CPU (fallback)")
                    except Exception as cpu_error:
                        log_queue.put(f"❌ CPU fallback also failed: {str(cpu_error)}")
                        log_queue.put("Please ensure you have MACE models downloaded locally")
                        return
                else:
                    log_queue.put("Please ensure you have MACE models downloaded locally")
                    return
        else:
            log_queue.put("❌ MACE not available - please install with: pip install mace-torch")
            return

        if calculator is None:
            log_queue.put("❌ Failed to create calculator")
            return

        log_queue.put("Calculator setup complete, starting structure calculations...")

        reference_energies = {}
        if calc_formation_energy:
            all_elements = set()
            for structure in structure_data.values():
                for site in structure:
                    all_elements.add(site.specie.symbol)

            reference_energies = calculate_atomic_reference_energies(all_elements, calculator, log_queue)
            log_queue.put(f"✅ Reference energies calculated for: {', '.join(all_elements)}")

        for i, (name, structure) in enumerate(structure_data.items()):
            if stop_event.is_set():
                log_queue.put("Calculation stopped by user")
                break

            structure_start_time = time.time()
            log_queue.put({
                'type': 'structure_start_time',
                'structure': name,
                'start_time': structure_start_time
            })

            log_queue.put(f"Processing structure {i + 1}/{len(structure_data)}: {name}")
            log_queue.put({'type': 'progress', 'current': i, 'total': len(structure_data), 'name': name})

            try:
                atoms = pymatgen_to_ase(structure)
                atoms.calc = calculator

                log_queue.put(f"Testing calculator with {name}...")

                try:
                    test_energy = atoms.get_potential_energy()
                    test_forces = atoms.get_forces()
                    log_queue.put(f"✅ Calculator test successful for {name}")
                    log_queue.put(
                        f"Initial energy: {test_energy:.6f} eV, Initial max force: {np.max(np.abs(test_forces)):.6f} eV/Å")
                except Exception as calc_error:
                    log_queue.put(f"❌ Calculator test failed for {name}: {str(calc_error)}")
                    log_queue.put({
                        'type': 'result',
                        'name': name,
                        'energy': None,
                        'structure': structure,
                        'calc_type': calc_type,
                        'error': f"Calculator test failed: {str(calc_error)}"
                    })
                    continue

                energy = None
                final_structure = structure
                convergence_status = None
                phonon_results = None
                elastic_results = None

                if calc_type == "Energy Only":
                    try:
                        energy = atoms.get_potential_energy()
                        log_queue.put(f"✅ Energy for {name}: {energy:.6f} eV")
                    except Exception as energy_error:
                        log_queue.put(f"❌ Energy calculation failed for {name}: {str(energy_error)}")
                        raise energy_error


                elif calc_type == "Geometry Optimization":

                    log_queue.put(f"Starting geometry optimization for {name}")

                    opt_type = optimization_params['optimization_type']

                    log_queue.put(f"Optimization type: {opt_type}")

                    if optimization_params.get('cell_constraint'):
                        log_queue.put(f"Cell constraint: {optimization_params['cell_constraint']}")

                    if optimization_params.get('pressure', 0) > 0:
                        log_queue.put(f"External pressure: {optimization_params['pressure']} GPa")

                    if optimization_params.get('hydrostatic_strain'):
                        log_queue.put("Using hydrostatic strain constraint")

                    log_queue.put({
                        'type': 'opt_start',
                        'structure': name,
                        'max_steps': optimization_params['max_steps'],
                        'fmax': optimization_params['fmax'],
                        'ediff': optimization_params['ediff'],
                        'optimization_type': opt_type,
                        'stress_threshold': optimization_params.get('stress_threshold', 0.1)
                    })

                    try:
                        optimization_object, opt_mode = setup_optimization_constraints(atoms, optimization_params)
                        if opt_mode:
                            logger = CellOptimizationLogger(log_queue, name, opt_mode)
                        else:
                            logger = OptimizationLogger(log_queue, name)
                        if optimization_params['optimizer'] == "LBFGS":
                            optimizer = LBFGS(optimization_object, logfile=None)
                        else:
                            optimizer = BFGS(optimization_object, logfile=None)
                        optimizer.attach(lambda: logger(optimizer), interval=1)
                        if opt_type == "Cell only (fixed atoms)":
                            fmax_criterion = 0.1
                        else:
                            fmax_criterion = optimization_params['fmax']
                        optimizer.run(fmax=fmax_criterion, steps=optimization_params['max_steps'])

                        if hasattr(optimization_object, 'atoms'):
                            final_atoms = optimization_object.atoms
                        else:
                            final_atoms = optimization_object
                        energy = final_atoms.get_potential_energy()
                        final_forces = final_atoms.get_forces()
                        max_final_force = np.max(np.linalg.norm(final_forces, axis=1))

                        force_converged = max_final_force < optimization_params['fmax']
                        energy_converged = False
                        stress_converged = True  # Default for atom-only optimization

                        if len(logger.trajectory) > 1:
                            final_energy_change = logger.trajectory[-1]['energy_change']

                            energy_converged = final_energy_change < optimization_params['ediff']

                        if opt_mode in ["cell_only", "both"]:
                            try:
                                final_stress = final_atoms.get_stress(voigt=True)
                                max_final_stress = np.max(np.abs(final_stress))
                                stress_converged = max_final_stress < 0.1  # 0.1 GPa stress convergence
                                log_queue.put(f"  Final stress: {max_final_stress:.4f} GPa")
                            except:
                                stress_converged = True  # Assume converged if we can't get stress

                        if opt_type == "Atoms only (fixed cell)":
                            if force_converged and energy_converged:
                                convergence_status = "CONVERGED (Force & Energy)"
                            elif force_converged:
                                convergence_status = "CONVERGED (Force)"
                            else:
                                convergence_status = "MAX STEPS REACHED"
                        elif opt_type == "Cell only (fixed atoms)":
                            if stress_converged and energy_converged:
                                convergence_status = "CONVERGED (Stress & Energy)"
                            elif stress_converged:
                                convergence_status = "CONVERGED (Stress)"
                            else:
                                convergence_status = "MAX STEPS REACHED"
                        else:  # Both
                            if force_converged and stress_converged and energy_converged:
                                convergence_status = "CONVERGED (Force, Stress & Energy)"
                            elif force_converged and stress_converged:
                                convergence_status = "CONVERGED (Force & Stress)"
                            elif force_converged:
                                convergence_status = "CONVERGED (Force only)"
                            else:
                                convergence_status = "MAX STEPS REACHED"
                        optimized_structure = ase_to_pymatgen_wrapped(final_atoms)
                        final_structure = optimized_structure

                        if opt_mode == "cell_only":
                            log_queue.put(
                                f"✅ Optimization {convergence_status} for {name}: Final energy = {energy:.6f} eV, Final max stress = {max_final_stress:.4f} GPa ({optimizer.nsteps} steps)")
                        elif opt_mode == "both":
                            log_queue.put(
                                f"✅ Optimization {convergence_status} for {name}: Final energy = {energy:.6f} eV, Final max force = {max_final_force:.4f} eV/Å, Final max stress = {max_final_stress:.4f} GPa ({optimizer.nsteps} steps)")
                        else:
                            log_queue.put(
                                f"✅ Optimization {convergence_status} for {name}: Final energy = {energy:.6f} eV, Final max force = {max_final_force:.4f} eV/Å ({optimizer.nsteps} steps)")
                        log_queue.put({
                            'type': 'complete_trajectory',
                            'structure': name,
                            'trajectory': logger.trajectory
                        })
                        log_queue.put({
                            'type': 'opt_complete',
                            'structure': name,
                            'final_steps': optimizer.nsteps,
                            'converged': force_converged and stress_converged and energy_converged,
                            'force_converged': force_converged,
                            'energy_converged': energy_converged,
                            'stress_converged': stress_converged,
                            'convergence_status': convergence_status
                        })

                    except Exception as opt_error:
                        log_queue.put(f"❌ Optimization failed for {name}: {str(opt_error)}")
                        try:
                            energy = atoms.get_potential_energy()
                            final_structure = structure
                            log_queue.put(f"⚠️  Using initial energy for {name}: {energy:.6f} eV")
                        except:
                            raise opt_error


                elif calc_type == "Phonon Calculation":
                    if optimization_params['max_steps'] > 0:
                        log_queue.put(f"Running brief pre-phonon optimization for {name}")
                        temp_atoms = atoms.copy()
                        temp_atoms.calc = calculator
                        try:
                            temp_optimizer = LBFGS(temp_atoms, logfile=None)
                            temp_optimizer.run(fmax=0.02, steps=50)
                            atoms = temp_atoms
                            energy = atoms.get_potential_energy()
                            log_queue.put(f"Pre-phonon optimization completed. Energy: {energy:.6f} eV")

                        except Exception as pre_opt_error:
                            log_queue.put(f"⚠️ Pre-optimization failed: {str(pre_opt_error)}")
                            energy = atoms.get_potential_energy()

                    phonon_results = calculate_phonons_pymatgen(atoms, calculator, phonon_params, log_queue, name)
                    if phonon_results['success']:
                        energy = atoms.get_potential_energy()

                elif calc_type == "Elastic Properties":
                    if optimization_params['max_steps'] > 0:
                        log_queue.put(f"Running pre-elastic optimization for {name} to ensure stability.")
                        temp_atoms = atoms.copy()
                        temp_atoms.calc = calculator
                        temp_logger = OptimizationLogger(log_queue, f"{name}_pre_elastic_opt")
                        try:
                            temp_optimizer = LBFGS(temp_atoms, logfile=None)
                            temp_optimizer.attach(lambda: temp_logger(temp_optimizer), interval=1)
                            temp_optimizer.run(fmax=0.015, steps=100)
                            atoms = temp_atoms
                            energy = atoms.get_potential_energy()
                            log_queue.put(
                                f"Pre-elastic optimization finished for {name}. Final energy: {energy:.6f} eV")
                        except Exception as pre_opt_error:
                            log_queue.put(f"⚠️ Pre-elastic optimization failed for {name}: {str(pre_opt_error)}")
                            log_queue.put("Continuing with elastic calculation on potentially unoptimized structure.")
                            energy = atoms.get_potential_energy()

                    elastic_results = calculate_elastic_properties(atoms, calculator, elastic_params, log_queue, name)
                    if elastic_results['success']:
                        energy = atoms.get_potential_energy()

                formation_energy = None
                if calc_formation_energy and energy is not None:
                    formation_energy = calculate_formation_energy(energy, structure, reference_energies)
                    if formation_energy is not None:
                        log_queue.put(f"✅ Formation energy for {name}: {formation_energy:.6f} eV/atom")
                    else:
                        log_queue.put(f"⚠️ Could not calculate formation energy for {name}")

                log_queue.put({
                    'type': 'result',
                    'name': name,
                    'energy': energy,
                    'formation_energy': formation_energy,
                    'structure': final_structure,
                    'calc_type': calc_type,
                    'convergence_status': convergence_status,
                    'phonon_results': phonon_results,
                    'elastic_results': elastic_results
                })

                structure_end_time = time.time()
                structure_duration = structure_end_time - structure_start_time

                log_queue.put({
                    'type': 'structure_end_time',
                    'structure': name,
                    'end_time': structure_end_time,
                    'duration': structure_duration,
                    'calc_type': calc_type
                })
            except Exception as e:

                structure_end_time = time.time()
                structure_duration = structure_end_time - structure_start_time

                log_queue.put({
                    'type': 'structure_end_time',
                    'structure': name,
                    'end_time': structure_end_time,
                    'duration': structure_duration,
                    'calc_type': calc_type,
                    'failed': True
                })

                log_queue.put(f"❌ Error calculating {name}: {str(e)}")
                log_queue.put(f"Error type: {type(e).__name__}")
                log_queue.put({
                    'type': 'result',
                    'name': name,
                    'energy': None,
                    'structure': structure,
                    'calc_type': calc_type,
                    'error': str(e)
                })
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        log_queue.put({
            'type': 'total_end_time',
            'end_time': total_end_time,
            'total_duration': total_duration
        })
    except Exception as e:
        log_queue.put(f"❌ Fatal error in calculation thread: {str(e)}")
        log_queue.put(f"Error type: {type(e).__name__}")
        import traceback
        log_queue.put(f"Traceback: {traceback.format_exc()}")

    finally:
        log_queue.put("CALCULATION_FINISHED")


st.set_page_config(page_title="MACE Molecular Dynamics Batch Structure Calculator", layout="wide")
st.title("MACE Molecular Dynamic Batch Structure Calculator")

if 'structures' not in st.session_state:
    st.session_state.structures = {}
if 'results' not in st.session_state:
    st.session_state.results = []
if 'calculation_running' not in st.session_state:
    st.session_state.calculation_running = False
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'log_queue' not in st.session_state:
    st.session_state.log_queue = queue.Queue()
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()
if 'current_structure_progress' not in st.session_state:
    st.session_state.current_structure_progress = {}
if 'optimization_steps' not in st.session_state:
    st.session_state.optimization_steps = {}
if 'optimization_trajectories' not in st.session_state:
    st.session_state.optimization_trajectories = {}
if 'current_optimization_info' not in st.session_state:
    st.session_state.current_optimization_info = {}
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = 0


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


with st.sidebar:
    st.header("MACE Model Selection")

    if not MACE_AVAILABLE:
        st.error("⚠️ MACE not available!")
        st.error("Please install with: `pip install mace-torch`")
        st.stop()

    st.success(f"✅ MACE available via: {MACE_IMPORT_METHOD}")

    selected_model = st.selectbox("Choose MACE Model", list(MACE_MODELS.keys()))
    model_size = MACE_MODELS[selected_model]

    device_option = st.radio(
        "Compute Device",
        ["CPU", "GPU (CUDA)"],
        help="GPU will be much faster if available. Falls back to CPU if GPU unavailable."
    )
    device = "cuda" if device_option == "GPU (CUDA)" else "cpu"

css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem !important;
        color: #1e3a8a !important;
        font-weight: bold !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 25px !important;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

if st.session_state.calculation_running:
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.spinner("Calculation in progress..."):
                st.info("VASP calculations are running, please wait. 😊")
                if st.session_state.get('progress_text', ''):
                    st.write(f"📈 {st.session_state.progress_text}")
                st.write("👀 **Switch to 'Calculation Console' tab for detailed output**")
                if st.button(
                        "🛑 Stop Calculation (the current structure will still finish)",
                        key="stop_top"):
                    st.session_state.stop_event.set()

    if st.session_state.get('total_steps', 0) > 0:
        progress_value = st.session_state.progress / st.session_state.total_steps
        st.progress(progress_value, text=st.session_state.get('progress_text', ''))

tab1, tab2, tab3, tab4 = st.tabs(
    ["📁 Structure Upload & Setup", "🖥️ Calculation Console", "📊 Results & Analysis", "📈 Optimization Trajectories and Convergence"])

with tab1:
    st.sidebar.header("Upload Structure Files")
    if not st.session_state.structures_locked:
        uploaded_files = st.sidebar.file_uploader(
            "Upload structure files (CIF, POSCAR, LMP, XSF, PW, CFG, etc.)",
            accept_multiple_files=True,
            type=None,  # Accept any file type, let the parser handle format detection
            help="Upload multiple structure files for batch processing. Supports CIF, POSCAR, LMP, XSF, PW, CFG and other ASE-compatible formats",
            key="structure_uploader"
        )

        if uploaded_files:
            new_structures = {}
            upload_errors = []

            for uploaded_file in uploaded_files:
                try:
                    structure = load_structure(uploaded_file)
                    new_structures[uploaded_file.name] = structure
                except Exception as e:
                    upload_errors.append(f"Error reading {uploaded_file.name}: {str(e)}")

            if upload_errors:
                for error in upload_errors:
                    st.error(error)

            if new_structures:
                st.session_state.pending_structures.update(new_structures)
                st.info(f"📁 {len(new_structures)} new structures ready to be added")

                total_after_addition = len(st.session_state.structures) + len(st.session_state.pending_structures)
                st.write(f"Current structures: {len(st.session_state.structures)}")
                st.write(f"Pending structures: {len(st.session_state.pending_structures)}")

                if st.sidebar.button("✅ Accept & Add Structures", type="primary"):
                    st.session_state.structures.update(st.session_state.pending_structures)
                    added_count = len(st.session_state.pending_structures)
                    st.session_state.pending_structures = {}
                    st.success(f"✅ Added {added_count} structures. Total: {len(st.session_state.structures)}")
                    st.rerun()

        if st.session_state.structures:
            st.success(f"✅ {len(st.session_state.structures)} structures loaded and ready to be locked.")

            with st.expander("📋 View Current Structures", expanded=False):
                for i, (name, structure) in enumerate(st.session_state.structures.items(), 1):
                    st.write(f"{i}. **{name}** - {structure.composition.reduced_formula} ({structure.num_sites} atoms)")

        col1, col2 = st.columns(2)

        with col1:
            if st.sidebar.button("🔒 Lock Structures for Calculation",
                                 disabled=len(st.session_state.structures) == 0, type='primary'):
                st.session_state.structures_locked = True
                st.success("🔒 Structures locked! You can now start calculations.")
                st.rerun()

        with col2:
            if st.sidebar.button("🗑️ Clear All Structures", type='secondary'):
                st.session_state.structures = {}
                st.session_state.pending_structures = {}
                st.session_state.results = []
                st.session_state.optimization_trajectories = {}
                st.success("All structures cleared")
                st.rerun()

    else:
        st.success(f"🔒 Structures Locked ({len(st.session_state.structures)} structures)")
        st.info("📌 Structures are locked to avoid refreshing during the calculation run. Use 'Unlock' to modify.")

        with st.expander("📋 Locked Structures", expanded=True):
            for i, (name, structure) in enumerate(st.session_state.structures.items(), 1):
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.write(f"**{i}. {name}**")
                with col2:
                    st.write(f"{structure.composition.reduced_formula}")
                with col3:
                    st.write(f"{structure.num_sites} atoms")

        if st.sidebar.button("🔓 Unlock Structures", type='secondary',
                             disabled=st.session_state.calculation_running):
            st.session_state.structures_locked = False
            st.info("🔓 Structures unlocked. You can now modify the structure list.")
            st.rerun()

        if st.session_state.calculation_running:
            st.warning("⚠️ Cannot unlock structures while calculation is running")
    st.sidebar.info(f"❤️🫶 **[Donations always appreciated!](https://buymeacoffee.com/bracerino)**")
    st.sidebar.info(
        "Try also the main application **[XRDlicious](xrdlicious.com)**. 🌀 Developed by **[IMPLANT team](https://implant.fs.cvut.cz/)**. 📺 **[Quick tutorial here](https://youtu.be/GGo_9T5wqus?si=xJItv-j0shr8hte_)**. Spot a bug or have a feature requests? Let us know at **lebedmi2@cvut.cz**."
    )

    st.sidebar.link_button("GitHub page", "https://github.com/bracerino/atat-sqs-gui.git",
                           type="primary")
    if st.session_state.structures:
        show_preview = st.checkbox("Show Structure Preview & MACE Compatibility", value=False)

        if show_preview:
            st.header("2. Structure Preview & MACE Compatibility")

            structure_names = list(st.session_state.structures.keys())

            for i, (name, structure) in enumerate(st.session_state.structures.items()):
                with st.expander(f"Structure {i + 1}: {name}"):
                    col1, col2 = st.columns([1, 1.5])

                    with col1:
                        components.html(view_structure(structure, height=250, width=350), height=260)

                    with col2:
                        st.write(f"**Formula:** {structure.composition.reduced_formula}")
                        st.write(f"**Number of atoms:** {structure.num_sites}")
                        st.write(f"**Lattice parameters:**")
                        st.write(f"  a = {structure.lattice.a:.3f} Å")
                        st.write(f"  b = {structure.lattice.b:.3f} Å")
                        st.write(f"  c = {structure.lattice.c:.3f} Å")

                        is_compatible, unsupported, elements = check_mace_compatibility(structure)

                        if is_compatible:
                            st.success("✅ Compatible with MACE-MP-0")
                        else:
                            st.error(f"❌ Unsupported elements: {', '.join(unsupported)}")

                        st.write(f"**Elements:** {', '.join(elements)}")

        st.divider()

        st.header("Calculation Setup")

        all_compatible = all(check_mace_compatibility(struct)[0] for struct in st.session_state.structures.values())

        if not all_compatible:
            st.error(
                "⚠️ Some structures contain elements not supported by MACE-MP-0. Please remove incompatible structures.")

        col_calc_setup, col_calc_image = st.columns([2, 1])

        with col_calc_setup:
            calc_type = st.radio(
                "Calculation Type",
                ["Energy Only", "Geometry Optimization", "Phonon Calculation", "Elastic Properties"],
                help="Choose the type of calculation to perform"
            )

            calculate_formation_energy_flag = st.checkbox(
                "Calculate Formation Energy",
                value=True,
                help="Calculate formation energy per atom"
            )

        import streamlit.components.v1 as components

        with col_calc_image:
            svg_image = create_calculation_type_image(calc_type)

            components.html(
                f"""
                <div style="display: flex; justify-content: center; align-items: center; height: 220px; padding: 10px;">
                    {svg_image}
                </div>
                """,
                height=240
            )
        with col_calc_image:
            if calc_type == "Energy Only":
                st.markdown("""
                <div style="background: linear-gradient(135deg, #4f46e5, #7c3aed); padding: 20px; border-radius: 15px; margin-top: 15px; color: white; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <strong style="font-size: 22px; display: block; margin-bottom: 10px;">⚡ Fast & Efficient</strong>
                <div style="font-size: 18px; line-height: 1.4;">
                Single point energy calculation<br>
                Ideal for energy comparisons
                </div>
                </div>
                """, unsafe_allow_html=True)

            elif calc_type == "Geometry Optimization":
                st.markdown("""
                <div style="background: linear-gradient(135deg, #059669, #0d9488); padding: 20px; border-radius: 15px; margin-top: 15px; color: white; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <strong style="font-size: 22px; display: block; margin-bottom: 10px;">🔄 Structure Relaxation</strong>
                <div style="font-size: 18px; line-height: 1.4;">
                Optimizes atomic positions<br>
                & lattice parameters
                </div>
                </div>
                """, unsafe_allow_html=True)

            elif calc_type == "Phonon Calculation":
                st.markdown("""
                <div style="background: linear-gradient(135deg, #dc2626, #ea580c); padding: 20px; border-radius: 15px; margin-top: 15px; color: white; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <strong style="font-size: 22px; display: block; margin-bottom: 10px;">🎵 Vibrational Analysis</strong>
                <div style="font-size: 18px; line-height: 1.4;">
                Phonon dispersion & DOS<br>
                Thermodynamic properties
                </div>
                </div>
                """, unsafe_allow_html=True)

            elif calc_type == "Elastic Properties":
                st.markdown("""
                <div style="background: linear-gradient(135deg, #7c2d12, #a16207); padding: 20px; border-radius: 15px; margin-top: 15px; color: white; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <strong style="font-size: 22px; display: block; margin-bottom: 10px;">⚙️ Mechanical Properties</strong>
                <div style="font-size: 18px; line-height: 1.4;">
                Elastic tensor & moduli<br>
                Bulk, shear, Young's modulus
                </div>
                </div>
                """, unsafe_allow_html=True)


        optimization_params = {
            'optimizer': "BFGS",
            'fmax': 0.05,
            'ediff': 1e-4,
            'max_steps': 200
        }
        phonon_params = {
            'supercell_size': (2, 2, 2),
            'delta': 0.01,
            'auto_kpath': True,
            'npoints': 100,
            'dos_points': 1000,
            'dos_sigma': 1.0,
            'temperature': 300
        }
        elastic_params = {
            'strain_magnitude': 0.01,
            'density': None
        }

        if calc_type == "Geometry Optimization":
            st.subheader("Optimization Parameters")

            if not CELL_OPT_AVAILABLE:
                st.error("⚠️ Cell optimization features require ASE constraints. Some features may be limited.")

            col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)

            with col_opt1:
                optimization_params['optimizer'] = st.selectbox(
                    "Optimizer",
                    ["BFGS", "LBFGS"],
                    help="BFGS: More memory but faster convergence. LBFGS: Less memory usage."
                )

            with col_opt2:
                optimization_params['fmax'] = st.number_input(
                    "Force threshold (eV/Å)",
                    min_value=0.001,
                    max_value=1.0,
                    value=0.05,
                    step=0.005,
                    format="%.3f",
                    help="Convergence criterion for maximum force on any atom"
                )

            with col_opt3:
                optimization_params['ediff'] = st.number_input(
                    "Energy threshold (eV) (only for monitoring, not a convergence parameter)",
                    min_value=1e-6,
                    max_value=1e-2,
                    value=1e-3,
                    step=1e-5,
                    format="%.1e",
                    help="Convergence criterion for energy change between steps"
                )

            with col_opt4:
                optimization_params['max_steps'] = st.number_input(
                    "Max steps",
                    min_value=10,
                    max_value=1000,
                    value=200,
                    step=10,
                    help="Maximum number of optimization steps"
                )

            st.subheader("Cell Optimization Parameters")

            optimization_type = st.radio(
                "What to optimize:",
                ["Atoms only (fixed cell)", "Cell only (fixed atoms)", "Both atoms and cell"],
                index=2,
                help="Choose whether to optimize atomic positions, cell parameters, or both"
            )

            optimization_params['optimization_type'] = optimization_type

            if optimization_type in ["Cell only (fixed atoms)", "Both atoms and cell"]:
                st.write("**Cell Parameter Constraints:**")

                col_cell1, col_cell2 = st.columns(2)

                with col_cell1:
                    cell_constraint = st.radio(
                        "Cell optimization mode:",
                        ["Lattice parameters only (fix angles)", "Full cell (lattice + angles)"],
                        index=0,
                        help="Choose whether to optimize only lattice parameters or also angles"
                    )
                    optimization_params['cell_constraint'] = cell_constraint

                with col_cell2:
                    if cell_constraint == "Lattice parameters only (fix angles)":
                        st.write("**Lattice directions to optimize:**")
                        optimize_a = st.checkbox("Optimize a-direction", value=True)
                        optimize_b = st.checkbox("Optimize b-direction", value=True)
                        optimize_c = st.checkbox("Optimize c-direction", value=True)

                        optimization_params['optimize_lattice'] = {
                            'a': optimize_a,
                            'b': optimize_b,
                            'c': optimize_c
                        }

                        if not any([optimize_a, optimize_b, optimize_c]):
                            st.warning("⚠️ At least one lattice direction must be optimized!")
                    else:
                        optimization_params['optimize_lattice'] = {'a': True, 'b': True, 'c': True}

                col_press1, col_press2, col_press3 = st.columns(3)
                with col_press1:
                    pressure = st.number_input(
                        "External pressure (GPa)",
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=0.1,
                        format="%.1f",
                        help="External pressure for cell optimization (0 = atmospheric pressure)"
                    )
                with col_press2:
                    hydrostatic_strain = st.checkbox(
                        "Hydrostatic strain only",
                        value=False,
                        help="Constrain cell to change hydrostatically (preserve shape)"
                    )
                with col_press3:
                    stress_threshold = st.number_input(
                        "Stress threshold (GPa)",
                        min_value=0.001,
                        max_value=1.0,
                        value=0.1,
                        step=0.01,
                        format="%.3f",
                        help="Maximum stress for convergence"
                    )

                optimization_params['pressure'] = pressure
                optimization_params['hydrostatic_strain'] = hydrostatic_strain
                optimization_params['stress_threshold'] = stress_threshold
            else:
                optimization_params['cell_constraint'] = None
                optimization_params['optimize_lattice'] = None
                optimization_params['pressure'] = 0.0
                optimization_params['hydrostatic_strain'] = False

            if optimization_type == "Atoms only (fixed cell)":
                st.info(
                    f"Optimization will adjust atomic positions only with forces < {optimization_params['fmax']} eV/Å")
            elif optimization_type == "Cell only (fixed atoms)":
                constraint_text = optimization_params.get('cell_constraint', 'lattice parameters only')
                pressure_text = f" at {optimization_params['pressure']} GPa" if optimization_params[
                                                                                    'pressure'] > 0 else ""
                hydro_text = " (hydrostatic)" if optimization_params.get('hydrostatic_strain') else ""
                st.info(f"Optimization will adjust {constraint_text} only{pressure_text}{hydro_text}")
            else:
                constraint_text = optimization_params.get('cell_constraint', 'lattice parameters only')
                pressure_text = f" at {optimization_params['pressure']} GPa" if optimization_params[
                                                                                    'pressure'] > 0 else ""
                hydro_text = " (hydrostatic)" if optimization_params.get('hydrostatic_strain') else ""
                st.info(
                    f"Optimization will adjust both atoms (F < {optimization_params['fmax']} eV/Å) and {constraint_text}{pressure_text}{hydro_text}")

        elif calc_type == "Phonon Calculation":
            st.subheader("Phonon Calculation Parameters")
            st.info(
                "A brief pre-optimization (fmax=0.01 eV/Å, max 100 steps) will be performed for stability before phonon calculations.")

            st.write("**Supercell Configuration**")
            auto_supercell = st.checkbox("Automatic supercell size estimation", value=True,
                                         help="Automatically estimate appropriate supercell size based on structure")
            if auto_supercell:
                col_auto1, col_auto2, col_auto3 = st.columns(3)
                with col_auto1:
                    target_length = st.number_input("Target supercell length (Å)", min_value=8.0, max_value=150.0,
                                                    value=15.0, step=1.0,
                                                    help="Minimum length for each supercell dimension")
                with col_auto2:
                    max_multiplier = st.number_input("Max supercell multiplier", min_value=1, max_value=50,
                                                     value=4, step=1,
                                                     help="Maximum allowed multiplier for any dimension")
                with col_auto3:
                    max_atoms = st.number_input("Max supercell atoms", min_value=100, max_value=200000,
                                                value=800, step=100,
                                                help="Maximum total atoms in supercell")
                phonon_params = {
                    'auto_supercell': True,
                    'target_supercell_length': target_length,
                    'max_supercell_multiplier': max_multiplier,
                    'max_supercell_atoms': max_atoms,
                    'delta': 0.01,
                    'auto_kpath': True,
                    'npoints': 100,
                    'dos_points': 1000,
                    'dos_sigma': 1.0,
                    'temperature': 300
                }
                st.info(
                    f"Supercell will be automatically estimated to achieve ~{target_length} Å minimum length per dimension")

            else:
                st.write("**Manual supercell specification**")
                col_ph1, col_ph2, col_ph3 = st.columns(3)
                with col_ph1:
                    supercell_x = st.number_input("Supercell X", min_value=1, value=2)
                with col_ph2:
                    supercell_y = st.number_input("Supercell Y", min_value=1, value=2)
                with col_ph3:
                    supercell_z = st.number_input("Supercell Z", min_value=1, value=2)
                phonon_params = {
                    'auto_supercell': False,
                    'supercell_size': (supercell_x, supercell_y, supercell_z),
                    'delta': 0.01,
                    'auto_kpath': True,
                    'npoints': 100,
                    'dos_points': 1000,
                    'dos_sigma': 1.0,
                    'temperature': 300
                }

            col_ph4, col_ph5 = st.columns(2)
            with col_ph4:
                phonon_params['delta'] = st.number_input("Displacement Delta (Å)", min_value=0.001, max_value=0.1,
                                                         value=phonon_params['delta'], step=0.001, format="%.3f")
            with col_ph5:
                phonon_params['temperature'] = st.number_input("Temperature for Thermodynamics (K)", min_value=0,
                                                               value=phonon_params['temperature'], step=10)
            st.write("**k-point path for dispersion**")
            phonon_params['auto_kpath'] = st.checkbox("Use Automatic High-Symmetry k-path",
                                                      value=phonon_params['auto_kpath'])
            if phonon_params['auto_kpath']:
                phonon_params['npoints'] = st.number_input("Number of points per segment", min_value=10,
                                                           value=phonon_params['npoints'], step=10)
            else:
                st.warning("Manual k-point path not yet implemented in GUI. Automatic path will be used as fallback.")
            st.write("**DOS parameters**")
            col_dos1, col_dos2 = st.columns(2)
            with col_dos1:
                phonon_params['dos_points'] = st.number_input("DOS points", min_value=100,
                                                              value=phonon_params['dos_points'], step=100)
            with col_dos2:
                phonon_params['dos_sigma'] = st.number_input("DOS Broadening (meV)", min_value=0.1,
                                                             value=phonon_params['dos_sigma'], step=0.1, format="%.1f")
        elif calc_type == "Elastic Properties":
            if not ELASTIC_AVAILABLE:
                st.error(
                    "⚠️ `ase.utils.eos` or `ase.build` not found. Elastic calculations require a full ASE installation.")
                st.stop()
            st.subheader("Elastic Properties Parameters")
            st.info(
                "A brief pre-optimization (fmax=0.015 eV/Å, max 100 steps) will be performed for stability before elastic calculations.")

            elastic_params['strain_magnitude'] = st.number_input("Strain Magnitude (e.g., 0.01 for 1%)",
                                                                 min_value=0.001, max_value=0.1, value=0.01, step=0.001,
                                                                 format="%.3f")
            # elastic_params['density'] = st.number_input("Material Density (g/cm³)", min_value=0.1, value=None,
            #                                            help="Optional: Provide if known. Otherwise, it will be estimated from the structure.",
            #                                            format="%.3f")
            elastic_params['density'] = None
        col1, col2 = st.columns(2)  # Changed from 2 to 3 columns

        st.markdown("""
            <style>
            div.stButton > button[kind="primary"] {
                background-color: #0099ff; color: white; font-size: 16px; font-weight: bold;
                padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
            }
            div.stButton > button[kind="primary"]:active, div.stButton > button[kind="primary"]:focus {
                background-color: #007acc !important; color: white !important; box-shadow: none !important;
            }

            div.stButton > button[kind="secondary"] {
                background-color: #dc3545; color: white; font-size: 16px; font-weight: bold;
                padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
            }
            div.stButton > button[kind="secondary"]:active, div.stButton > button[kind="secondary"]:focus {
                background-color: #c82333 !important; color: white !important; box-shadow: none !important;
            }

            div.stButton > button[kind="tertiary"] {
                background-color: #6f42c1; color: white; font-size: 16px; font-weight: bold;
                padding: 0.5em 1em; border: none; border-radius: 5px; height: 3em; width: 100%;
            }
            div.stButton > button[kind="tertiary"]:active, div.stButton > button[kind="tertiary"]:focus {
                background-color: #5a2d91 !important; color: white !important; box-shadow: none !important;
            }

            div[data-testid="stDataFrameContainer"] table td { font-size: 16px !important; }
            #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
            </style>
        """, unsafe_allow_html=True)
        with col1:
            start_calc = st.button(
                "🚀 Start Batch Calculation",
                type="primary",
                disabled=not all_compatible or
                         st.session_state.calculation_running or
                         len(st.session_state.structures) == 0 or
                         not st.session_state.structures_locked,
            )

        if len(st.session_state.structures) > 0 and not st.session_state.structures_locked:
            st.warning("🔒 Please lock your structures before starting calculation to prevent accidental changes.")

        with col2:
            but_script = st.button(
                "📝 Generate Python Script (⚠️ In Testing Mode!)",
                type="tertiary",
                disabled=len(st.session_state.structures) == 0,
            )
        if but_script:
            script_content = generate_python_script(
                structures=st.session_state.structures,
                calc_type=calc_type,
                model_size=model_size,
                device=device,
                optimization_params=optimization_params,
                phonon_params=phonon_params,
                elastic_params=elastic_params,
                calc_formation_energy=calculate_formation_energy_flag
            )

            with st.expander("📋 Generated Python Script", expanded=True):
                st.code(script_content, language='python')
                st.download_button(
                    label="💾 Download Script",
                    data=script_content,
                    file_name="mace_calculation_script.py",
                    mime="text/x-python",
                    help="Download the Python script file"
                )

                st.info("""
                        **Usage Instructions:**
                        1. Save the script as `mace_calculation_script.py`
                        2. Place your POSCAR files in the same directory
                        3. Install required packages: `pip install mace-torch ase pymatgen numpy`
                        4. Run: `python mace_calculation_script.py`

                        **Output Files:**
                        - `results_summary.txt` - Main results and energies
                        - `trajectory_*.xyz` - Optimization trajectories (if applicable)
                        - `optimized_structures/` - Final optimized structures (if applicable)
                        - `phonon_data_*.json` - Phonon results (if applicable)
                        - `elastic_data_*.json` - Elastic properties (if applicable)
                        """)

        if start_calc:
            st.session_state.calculation_running = True
            st.session_state.log_messages = []
            st.session_state.results = []
            st.session_state.current_structure_progress = {}
            st.session_state.current_optimization_info = {}
            st.session_state.optimization_steps = {}
            st.session_state.optimization_trajectories = {}
            st.session_state.stop_event.clear()
            st.session_state.last_update_time = 0

            thread = threading.Thread(
                target=run_mace_calculation,
                args=(st.session_state.structures, calc_type, model_size, device, optimization_params,
                      phonon_params, elastic_params, calculate_formation_energy_flag, st.session_state.log_queue,
                      st.session_state.stop_event)
            )
            thread.start()
            st.rerun()

    else:
        st.info("Upload structure files to begin")
with st.sidebar:
    st.info(f"**Selected Model:** {selected_model}")
    st.info(f"**Device:** {device}")

    if MACE_IMPORT_METHOD == "mace_mp":
        st.info("Using mace_mp - models downloaded automatically")
    else:
        st.warning("Using MACECalculator - may require local model files")
with tab2:
    st.header("Calculation Console")

    if st.session_state.calculation_running:
        st.info("🔄 Calculation in progress...")

        if st.button("🛑 Stop Calculation (the current structure will still finish)", key="console_stop"):
            st.session_state.stop_event.set()

    has_new_messages = False
    message_count = 0
    max_messages_per_cycle = 10

    while not st.session_state.log_queue.empty() and message_count < max_messages_per_cycle:
        has_new_messages = True
        message_count += 1
        message = st.session_state.log_queue.get()

        if isinstance(message, dict):
            if message.get('type') == 'total_start_time':
                st.session_state.total_calculation_start_time = message['start_time']
            elif message.get('type') == 'structure_start_time':
                st.session_state.structure_start_times[message['structure']] = message['start_time']
            elif message.get('type') == 'structure_end_time':
                structure_name = message['structure']
                if structure_name in st.session_state.structure_start_times:
                    start_time = st.session_state.structure_start_times[structure_name]
                    duration = message['duration']

                    st.session_state.computation_times[structure_name] = {
                        'start_time': start_time,
                        'end_time': message['end_time'],
                        'duration': duration,
                        'calc_type': message['calc_type'],
                        'failed': message.get('failed', False),
                        'human_duration': format_duration(duration)
                    }
            if message.get('type') == 'progress':
                progress = message['current'] / message['total'] if message['total'] > 0 else 0
                st.session_state.current_structure_progress = {
                    'progress': progress,
                    'text': f"Processing: {message['name']} ({message['current'] + 1}/{message['total']})",
                    'current': message['current'],
                    'total': message['total'],
                    'name': message['name']
                }
            elif message.get('type') == 'opt_start':
                st.session_state.current_optimization_info = {
                    'structure': message['structure'],
                    'max_steps': message['max_steps'],
                    'current_step': 0,
                    'fmax': message['fmax'],
                    'ediff': message.get('ediff', 1e-4),
                    'optimization_type': message.get('optimization_type', 'Both atoms and cell'),
                    'stress_threshold': message.get('stress_threshold', 0.1),
                    'is_optimizing': True
                }
            elif message.get('type') == 'opt_step':
                structure_name = message['structure']
                if structure_name not in st.session_state.optimization_steps:
                    st.session_state.optimization_steps[structure_name] = []
                st.session_state.optimization_steps[structure_name].append({
                    'step': message['step'],
                    'energy': message['energy'],
                    'max_force': message['max_force'],
                    'max_stress': message.get('max_stress', 0.0),
                    'energy_change': message.get('energy_change', 0)
                })
                if st.session_state.current_optimization_info.get('structure') == structure_name:
                    st.session_state.current_optimization_info.update({
                        'current_step': message['step'],
                        'current_energy': message['energy'],
                        'current_max_force': message['max_force'],
                        'current_max_stress': message.get('max_stress', 0.0),
                        'current_energy_change': message.get('energy_change', 0)
                    })
            elif message.get('type') == 'opt_complete':
                st.session_state.current_optimization_info = {}
            elif message.get('type') == 'trajectory_step':
                structure_name = message['structure']
                if structure_name not in st.session_state.optimization_trajectories:
                    st.session_state.optimization_trajectories[structure_name] = []
                st.session_state.optimization_trajectories[structure_name].append(message['trajectory_data'])
            elif message.get('type') == 'complete_trajectory':
                st.session_state.optimization_trajectories[message['structure']] = message['trajectory']
            elif message.get('type') == 'result':
                st.session_state.results.append(message)
        elif message == "CALCULATION_FINISHED":
            st.session_state.calculation_running = False
            st.session_state.current_structure_progress = {}
            st.session_state.current_optimization_info = {}
            st.success("✅ All calculations completed!")

            st.rerun()
        else:
            st.session_state.log_messages.append(str(message))

    if st.session_state.calculation_running:
        if st.session_state.current_structure_progress:
            progress_data = st.session_state.current_structure_progress
            st.progress(progress_data['progress'], text=progress_data['text'])

        if st.session_state.current_optimization_info and st.session_state.current_optimization_info.get(
                'is_optimizing'):
            opt_info = st.session_state.current_optimization_info

            opt_progress = opt_info.get('current_step', 0) / opt_info['max_steps'] if opt_info['max_steps'] > 0 else 0
            opt_text = f"Optimizing {opt_info['structure']}: Step {opt_info.get('current_step', 0)}/{opt_info['max_steps']}"

            if 'current_energy' in opt_info:
                opt_text += f" | Energy: {opt_info['current_energy']:.6f} eV"

            if 'current_max_force' in opt_info:
                opt_text += f" | Max Force: {opt_info['current_max_force']:.4f} eV/Å"

            if 'current_max_stress' in opt_info:
                opt_text += f" | Max Stress: {opt_info['current_max_stress']:.4f} GPa"

            if 'current_energy_change' in opt_info:
                opt_text += f" | ΔE: {opt_info['current_energy_change']:.2e} eV"

            st.progress(opt_progress, text=opt_text)

            if 'current_step' in opt_info and opt_info['current_step'] > 0:
                opt_type = opt_info.get('optimization_type', 'both')

                if opt_type == "Atoms only (fixed cell)":
                    col1, col2, col3, col4 = st.columns(4)
                elif opt_type == "Cell only (fixed atoms)":
                    col1, col2, col3, col4 = st.columns(4)
                else:
                    col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Current Step", f"{opt_info['current_step']}/{opt_info['max_steps']}")

                with col2:
                    if 'current_energy' in opt_info:
                        st.metric("Energy (eV)", f"{opt_info['current_energy']:.6f}")

                if opt_type in ["Atoms only (fixed cell)", "Both atoms and cell"]:
                    with col3:
                        if 'current_max_force' in opt_info:
                            force_converged = opt_info['current_max_force'] < opt_info['fmax']
                            st.metric("Max Force (eV/Å)", f"{opt_info['current_max_force']:.4f}",
                                      delta="✅ Converged" if force_converged else "❌ Not converged")

                if opt_type in ["Cell only (fixed atoms)", "Both atoms and cell"]:
                    stress_col = col3 if opt_type == "Cell only (fixed atoms)" else col4
                    with stress_col:
                        if 'current_max_stress' in opt_info:
                            stress_threshold = opt_info.get('stress_threshold', 0.1)
                            stress_converged = opt_info['current_max_stress'] < stress_threshold
                            st.metric("Max Stress (GPa)", f"{opt_info['current_max_stress']:.4f}",
                                      delta="✅ Converged" if stress_converged else "❌ Not converged")

                energy_col = col4 if opt_type == "Atoms only (fixed cell)" else (
                    col4 if opt_type == "Cell only (fixed atoms)" else col5)
                with energy_col:
                    if 'current_energy_change' in opt_info:
                        energy_converged = opt_info['current_energy_change'] < opt_info['ediff']
                        st.metric("ΔE (eV)", f"{opt_info['current_energy_change']:.2e}",
                                  delta="✅ Converged" if energy_converged else "❌ Not converged")

    if st.session_state.log_messages:
        recent_messages = st.session_state.log_messages[-20:]
        st.text_area("Calculation Log", "\n".join(recent_messages), height=300)

    if has_new_messages and st.session_state.calculation_running:
        time.sleep(1)
        st.rerun()


def get_atomic_concentrations_from_structure(structure):
    element_counts = {}
    total_atoms = len(structure)

    for site in structure:
        element = site.specie.symbol
        element_counts[element] = element_counts.get(element, 0) + 1

    concentrations = {}
    for element, count in element_counts.items():
        concentrations[element] = (count / total_atoms) * 100

    return concentrations


with tab3:
    st.header("Results & Analysis")
    if st.session_state.results:
        results_tab1, results_tab2, results_tab3, results_tab4, results_tab5 = st.tabs(["📊 Energies",
                                                                                        "🔧 Geometry Optimization Details",
                                                                                        "Elastic properties", "Phonons",
                                                                                        "⏱️ Computation times"])
    else:
        st.info("Please start some calculation first.")

    if st.session_state.results:
        with results_tab5:
            st.subheader("⏱️ Computation Time Analysis")

            if st.session_state.computation_times:
                timing_data = []
                total_successful_time = 0
                total_failed_time = 0

                for structure_name, timing_info in st.session_state.computation_times.items():
                    status = "❌ Failed" if timing_info.get('failed', False) else "✅ Success"

                    timing_data.append({
                        'Structure': structure_name,
                        'Calculation Type': timing_info['calc_type'],
                        'Duration': timing_info['human_duration'],
                        'Duration (seconds)': f"{timing_info['duration']:.2f}",
                        'Status': status,
                        'Start Time': time.strftime('%H:%M:%S', time.localtime(timing_info['start_time'])),
                        'End Time': time.strftime('%H:%M:%S', time.localtime(timing_info['end_time']))
                    })

                    if timing_info.get('failed', False):
                        total_failed_time += timing_info['duration']
                    else:
                        total_successful_time += timing_info['duration']

                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

                with col_stats1:
                    st.metric("Total Structures", len(st.session_state.computation_times))

                with col_stats2:
                    successful_count = len(
                        [t for t in st.session_state.computation_times.values() if not t.get('failed', False)])
                    st.metric("Successful", successful_count)

                with col_stats3:
                    total_time = total_successful_time + total_failed_time
                    st.metric("Total Time", format_duration(total_time))

                with col_stats4:
                    if len(st.session_state.computation_times) > 0:
                        avg_time = total_time / len(st.session_state.computation_times)
                        st.metric("Average per Structure", format_duration(avg_time))

                st.subheader("Detailed Timing Information")
                df_timing = pd.DataFrame(timing_data)

                df_timing_sorted = df_timing.sort_values('Duration (seconds)', ascending=False)
                st.dataframe(df_timing_sorted, use_container_width=True, hide_index=True)

                if len(timing_data) > 1:
                    st.subheader("📊 Timing Visualizations")

                    col_viz1, col_viz2 = st.columns(2)

                    with col_viz1:
                        structures = [d['Structure'] for d in timing_data]
                        durations = [float(d['Duration (seconds)']) for d in timing_data]
                        statuses = [d['Status'] for d in timing_data]

                        colors = ['green' if '✅' in status else 'red' for status in statuses]

                        fig_timing = go.Figure(data=go.Bar(
                            x=structures,
                            y=durations,
                            marker_color=colors,
                            text=[format_duration(d) for d in durations],
                            textposition='auto',
                            hovertemplate='<b>%{x}</b><br>Duration: %{text}<br>Status: %{customdata}<extra></extra>',
                            customdata=statuses
                        ))

                        fig_timing.update_layout(
                            title=dict(text="Computation Time by Structure", font=dict(size=24)),
                            xaxis_title="Structure",
                            yaxis_title="Time (seconds)",
                            height=750,
                            font=dict(size=16),
                            xaxis=dict(
                                tickangle=45,
                                title_font=dict(size=18),
                                tickfont=dict(size=14)
                            ),
                            yaxis=dict(
                                title_font=dict(size=18),
                                tickfont=dict(size=14)
                            )
                        )

                        st.plotly_chart(fig_timing, use_container_width=True)

                    with col_viz2:
                        calc_types = {}
                        for timing_info in st.session_state.computation_times.values():
                            calc_type = timing_info['calc_type']
                            if calc_type not in calc_types:
                                calc_types[calc_type] = 0
                            calc_types[calc_type] += timing_info['duration']

                        if len(calc_types) > 1:
                            fig_pie = go.Figure(data=go.Pie(
                                labels=list(calc_types.keys()),
                                values=list(calc_types.values()),
                                textinfo='label+percent',
                                textposition='auto',
                                hovertemplate='<b>%{label}</b><br>Time: %{customdata}<br>Percentage: %{percent}<extra></extra>',
                                customdata=[format_duration(v) for v in calc_types.values()]
                            ))

                            fig_pie.update_layout(
                                title=dict(text="Time Distribution by Calculation Type", font=dict(size=20)),
                                height=400,
                                font=dict(size=16)
                            )

                            st.plotly_chart(fig_pie, use_container_width=True)
                        else:
                            calc_type = list(calc_types.keys())[0]
                            total_time = list(calc_types.values())[0]

                            st.markdown(f"""
                            <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 20px 0;">
                                <h2 style="color: white; margin: 0; font-size: 2em;">All Calculations: {calc_type}</h2>
                                <h1 style="color: white; margin: 10px 0; font-size: 3em;">{format_duration(total_time)}</h1>
                            </div>
                            """, unsafe_allow_html=True)

                if len(timing_data) > 2:
                    st.subheader("📅 Calculation Timeline")
                    timeline_data = []
                    for structure_name, timing_info in st.session_state.computation_times.items():
                        start_time = timing_info['start_time']
                        end_time = timing_info['end_time']

                        timeline_data.append({
                            'Structure': structure_name,
                            'Start': pd.to_datetime(start_time, unit='s'),
                            'Finish': pd.to_datetime(end_time, unit='s'),
                            'Duration': timing_info['duration'],
                            'Status': "Success" if not timing_info.get('failed', False) else "Failed"
                        })

                    fig_timeline = go.Figure()

                    for i, data in enumerate(timeline_data):
                        color = 'green' if data['Status'] == 'Success' else 'red'

                        fig_timeline.add_trace(go.Scatter(
                            x=[data['Start'], data['Finish']],
                            y=[i, i],
                            mode='lines+markers',
                            line=dict(color=color, width=8),
                            marker=dict(size=8, color=color),
                            name=data['Structure'],
                            hovertemplate=f"<b>{data['Structure']}</b><br>" +
                                          f"Start: %{{x}}<br>" +
                                          f"Duration: {format_duration(data['Duration'])}<br>" +
                                          f"Status: {data['Status']}<extra></extra>",
                            showlegend=False
                        ))

                    fig_timeline.update_layout(
                        title=dict(text="Calculation Timeline", font=dict(size=24)),
                        xaxis_title="Time",
                        yaxis=dict(
                            tickmode='array',
                            tickvals=list(range(len(timeline_data))),
                            ticktext=[d['Structure'] for d in timeline_data],
                            title="Structure",
                            title_font=dict(size=18),
                            tickfont=dict(size=14)
                        ),
                        height=max(650, len(timeline_data) * 40),
                        font=dict(size=16),
                        xaxis=dict(
                            title_font=dict(size=18),
                            tickfont=dict(size=14)
                        ),
                        hoverlabel=dict(
                            bgcolor="white",
                            bordercolor="black",
                            font_size=20,
                            font_family="Arial"
                        )
                    )

                    st.plotly_chart(fig_timeline, use_container_width=True)
                st.subheader("🎯 Performance Insights")

                if len(timing_data) > 1:
                    fastest = min(timing_data, key=lambda x: float(x['Duration (seconds)']))
                    slowest = max(timing_data, key=lambda x: float(x['Duration (seconds)']))

                    col_insight1, col_insight2 = st.columns(2)

                    with col_insight1:
                        st.markdown(f"""
                        **🏃 Fastest Calculation**
                        - Structure: {fastest['Structure']}
                        - Time: {fastest['Duration']}
                        - Type: {fastest['Calculation Type']}
                        """)

                    with col_insight2:
                        st.markdown(f"""
                        **🐌 Slowest Calculation**
                        - Structure: {slowest['Structure']}
                        - Time: {slowest['Duration']}
                        - Type: {slowest['Calculation Type']}
                        """)



                st.subheader("📥 Export Timing Data")

                col_export1, col_export2 = st.columns(2)

                with col_export1:
                    timing_csv = df_timing.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Timing Data (CSV)",
                        data=timing_csv,
                        file_name="computation_times.csv",
                        mime="text/csv",
                        help="Download detailed timing information as CSV"
                    )

                with col_export2:
                    timing_report = {
                        'summary': {
                            'total_structures': len(st.session_state.computation_times),
                            'successful_structures': successful_count,
                            'total_time_seconds': total_successful_time + total_failed_time,
                            'total_time_formatted': format_duration(total_successful_time + total_failed_time),
                            'average_time_per_structure': format_duration(
                                (total_successful_time + total_failed_time) / len(
                                    st.session_state.computation_times)) if len(
                                st.session_state.computation_times) > 0 else "0s"
                        },
                        'detailed_times': {
                            name: {
                                'duration_seconds': info['duration'],
                                'duration_formatted': info['human_duration'],
                                'calculation_type': info['calc_type'],
                                'status': 'failed' if info.get('failed', False) else 'success',
                                'start_timestamp': info['start_time'],
                                'end_timestamp': info['end_time']
                            }
                            for name, info in st.session_state.computation_times.items()
                        }
                    }

                    timing_json = json.dumps(timing_report, indent=2)
                    st.download_button(
                        label="📋 Download Timing Report (JSON)",
                        data=timing_json,
                        file_name="timing_report.json",
                        mime="application/json",
                        help="Download comprehensive timing report as JSON"
                    )

            else:
                st.info("⏱️ Computation timing data will appear here after calculations complete.")

                if st.session_state.calculation_running:
                    if st.session_state.total_calculation_start_time:
                        current_time = time.time()
                        elapsed = current_time - st.session_state.total_calculation_start_time
                        st.metric("Current Session Time", format_duration(elapsed))

                    if st.session_state.structure_start_times:
                        st.write("**Structures in Progress:**")
                        current_time = time.time()
                        for structure_name, start_time in st.session_state.structure_start_times.items():
                            if structure_name not in st.session_state.computation_times:
                                elapsed = current_time - start_time
                                st.write(f"• {structure_name}: {format_duration(elapsed)} (running)")
                else:
                    st.markdown("""
                    **What you'll see here:**

                    📊 **Timing Statistics**: Total time, average per structure, success rate

                    📈 **Visualizations**: Bar charts, pie charts, and timeline views

                    🏃 **Performance Insights**: Fastest/slowest calculations and efficiency metrics

                    📥 **Export Options**: Download timing data as CSV or detailed JSON reports

                    ⏱️ **Real-time Tracking**: Live timing updates during calculations
                    """)
        with results_tab1:

            all_elements = get_all_elements_from_results(st.session_state.results)

            results_data = []
            for r in st.session_state.results:
                if r['energy'] is not None:
                    if r['calc_type'] == 'Geometry Optimization' and 'convergence_status' in r:
                        status = r['convergence_status']
                    elif r['calc_type'] in ["Phonon Calculation", "Elastic Properties"]:
                        if (r.get('phonon_results') and not r['phonon_results']['success']) or \
                                (r.get('elastic_results') and not r['elastic_results']['success']):
                            status = 'Failed (Sub-calc)'
                        else:
                            status = 'Success'
                    else:
                        status = 'Success'
                else:
                    status = 'Failed'

                row_data = {
                    'Structure': r['name'],
                    'Energy (eV)': r['energy'] if r['energy'] is not None else 'Error',
                    'Formation Energy (eV/atom)': f"{r.get('formation_energy'):.6f}" if r.get(
                        'formation_energy') is not None else 'N/A',
                    'Calculation Type': r['calc_type'],
                    'Status': status,
                    'Error': r.get('error', '')
                }

                if 'structure' in r and r['structure'] and r['energy'] is not None:
                    concentrations = get_atomic_concentrations(r['structure'])
                    for element in all_elements:
                        concentration = concentrations.get(element, 0)
                        row_data[f'{element} (%)'] = f"{concentration:.1f}" if concentration > 0 else "0.0"
                else:
                    for element in all_elements:
                        row_data[f'{element} (%)'] = "N/A"

                results_data.append(row_data)

            df_results = pd.DataFrame(results_data)

            successful_results = [r for r in st.session_state.results if r['energy'] is not None]

            if successful_results:
                st.subheader("Energy Comparison")

                energies = [r['energy'] for r in successful_results]
                formation_energies = [r.get('formation_energy') for r in successful_results]
                names = [r['name'] for r in successful_results]

                has_formation_energies = any(fe is not None for fe in formation_energies)

                if has_formation_energies:
                    col_energy1, col_energy2 = st.columns(2)
                else:
                    col_energy1 = st.container()

                with col_energy1:
                    fig = go.Figure()

                    min_energy_idx = energies.index(min(energies))
                    colors = ['#28A745' if i == min_energy_idx else 'steelblue' for i in range(len(energies))]

                    fig.add_trace(go.Bar(
                        x=names,
                        y=energies,
                        name='Total Energy',
                        marker_color=colors,
                        hovertemplate='<b>%{x}</b><br>Energy: %{y:.6f} eV<extra></extra>'
                    ))

                    fig.update_layout(
                        title=dict(text="Total Energy Comparison", font=dict(size=24)),
                        xaxis_title="Structure",
                        yaxis_title="Energy (eV)",
                        height=750,
                        font=dict(size=20),
                        hoverlabel=dict(
                            bgcolor="white",
                            bordercolor="black",
                            font_size=20,
                            font_family="Arial"
                        ),
                        xaxis=dict(
                            tickangle=45,
                            title_font=dict(size=20),
                            tickfont=dict(size=20)
                        ),
                        yaxis=dict(
                            title_font=dict(size=20),
                            tickfont=dict(size=20)
                        ),
                        legend=dict(
                            font=dict(size=20)
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True, key=f"energy_plot_{len(successful_results)}")

                if has_formation_energies:
                    with col_energy2:
                        valid_formation_data = [(name, fe) for name, fe in zip(names, formation_energies) if
                                                fe is not None]
                        if valid_formation_data:
                            valid_names, valid_formation_energies = zip(*valid_formation_data)
                            min_formation_idx = valid_formation_energies.index(min(valid_formation_energies))
                            formation_colors = ['#28A745' if i == min_formation_idx else 'orange' for i in
                                                range(len(valid_formation_energies))]

                            fig_form = go.Figure()
                            fig_form.add_trace(go.Bar(
                                x=valid_names,
                                y=valid_formation_energies,
                                name='Formation Energy',
                                marker_color=formation_colors,
                                hovertemplate='<b>%{x}</b><br>Formation Energy: %{y:.6f} eV/atom<extra></extra>'
                            ))

                            fig_form.update_layout(
                                title=dict(text="Formation Energy per Atom", font=dict(size=24)),
                                xaxis_title="Structure",
                                yaxis_title="Formation Energy (eV/atom)",
                                height=750,
                                font=dict(size=20),
                                hoverlabel=dict(
                                    bgcolor="white",
                                    bordercolor="black",
                                    font_size=20,
                                    font_family="Arial"
                                ),
                                xaxis=dict(
                                    tickangle=45,
                                    title_font=dict(size=20),
                                    tickfont=dict(size=20)
                                ),
                                yaxis=dict(
                                    title_font=dict(size=20),
                                    tickfont=dict(size=20)
                                ),
                                legend=dict(
                                    font=dict(size=20)
                                )
                            )

                            st.plotly_chart(fig_form, use_container_width=True,
                                            key=f"formation_plot_{len(valid_formation_data)}")

                if len(successful_results) > 1:
                    st.subheader("Relative Energies")
                    min_energy = min(energies)
                    relative_energies = [(e - min_energy) * 1000 for e in energies]

                    min_relative_idx = relative_energies.index(min(relative_energies))

                    relative_colors = ['#28A745' if i == min_relative_idx else 'orange' for i in
                                       range(len(relative_energies))]

                    fig_rel = go.Figure()
                    fig_rel.add_trace(go.Bar(
                        x=names,
                        y=relative_energies,
                        name='Relative Energy',
                        marker_color=relative_colors,
                        hovertemplate='<b>%{x}</b><br>Relative Energy: %{y:.3f} meV<extra></extra>'
                    ))

                    fig_rel.update_layout(
                        title=dict(text="Relative Energies (meV)", font=dict(size=24)),
                        xaxis_title="Structure",
                        yaxis_title="Relative Energy (meV)",
                        height=750,
                        font=dict(size=20),
                        hoverlabel=dict(
                            bgcolor="white",
                            bordercolor="black",
                            font_size=20,
                            font_family="Arial"
                        ),
                        xaxis=dict(
                            tickangle=45,
                            title_font=dict(size=20),
                            tickfont=dict(size=20)
                        ),
                        yaxis=dict(
                            title_font=dict(size=20),
                            tickfont=dict(size=20)
                        ),
                        legend=dict(
                            font=dict(size=20)
                        )
                    )

                    st.plotly_chart(fig_rel, use_container_width=True, key=f"relative_plot_{len(successful_results)}")

            st.subheader("Detailed Results")
            st.dataframe(df_results, use_container_width=True, key=f"results_table_{len(st.session_state.results)}")

            if len(successful_results) <= 8 and successful_results:
                st.subheader("Structure Overview Cards")
                sorted_results = sorted(successful_results, key=lambda x: (
                    x.get('formation_energy') if x.get('formation_energy') is not None
                    else x['energy'] if x['energy'] is not None
                    else float('inf')
                ))

                cols = st.columns(min(len(sorted_results), 4))

                for i, result in enumerate(sorted_results):
                    with cols[i % 4]:
                        n_results = len(sorted_results)
                        if n_results == 1:
                            color = "#4ECDC4"
                        else:
                            ratio = i / (n_results - 1)

                            if ratio == 0:
                                color = "#0066CC"  # Pure blue for lowest energy
                            elif ratio == 1:
                                color = "#666666"  # Grey for highest energy
                            else:
                                # Interpolate between blue and grey
                                blue_r, blue_g, blue_b = 0, 102, 204  # #0066CC
                                grey_r, grey_g, grey_b = 102, 102, 102  # #666666

                                red_component = int(blue_r + (grey_r - blue_r) * ratio)
                                green_component = int(blue_g + (grey_g - blue_g) * ratio)
                                blue_component = int(blue_b + (grey_b - blue_b) * ratio)

                                color = f"#{red_component:02x}{green_component:02x}{blue_component:02x}"

                        if 'structure' in result and result['structure']:
                            composition = result['structure'].composition.reduced_formula
                            concentrations = get_atomic_concentrations_from_structure(result['structure'])
                            conc_text = ", ".join([f"{elem}: {conc:.1f}%" for elem, conc in concentrations.items()])
                        else:
                            composition = "Unknown"
                            conc_text = "N/A"

                        total_energy = f"{result['energy']:.3f}" if result['energy'] is not None else "Error"
                        formation_energy = f"{result.get('formation_energy'):.3f}" if result.get(
                            'formation_energy') is not None else "N/A"

                        if n_results > 1:
                            if i == 0:
                                rank_indicator = "🥇 Lowest Energy"
                            elif i == n_results - 1:
                                rank_indicator = "⚫ Highest Energy"
                            else:
                                rank_indicator = f"#{i + 1}"
                        else:
                            rank_indicator = ""

                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, {color}, {color}CC);
                            padding: 20px;
                            border-radius: 15px;
                            text-align: center;
                            margin: 10px 0;
                            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                            border: 2px solid rgba(255,255,255,0.2);
                            height: 280px;
                            display: flex;
                            flex-direction: column;
                            justify-content: center;
                            color: white;
                        ">
                            <div style="
                                font-size: 1.4em;
                                margin: 0 0 5px 0;
                                text-shadow: 1px 1px 2px rgba(0,0,0,0.4);
                                font-weight: bold;
                                line-height: 1.2;
                            ">{result['name']}</div>
                            <div style="
                                font-size: 1.3em;
                                margin: 0 0 12px 0;
                                text-shadow: 1px 1px 2px rgba(0,0,0,0.4);
                                opacity: 0.9;
                                font-weight: bold;
                            ">{conc_text}</div>
                            <div style="
                                font-size: 1.6em;
                                margin: 8px 0;
                                text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
                                font-weight: bold;
                            ">Total Energy<br><span style="font-size: 0.9em; opacity: 0.9;">{total_energy} eV</span></div>
                            <div style="
                                font-size: 1.4em;
                                margin: 8px 0;
                                text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
                                font-weight: bold;
                            ">Formation Energy<br><span style="font-size: 0.9em; opacity: 0.9;">{formation_energy} eV/atom</span></div>
                        </div>
                        """, unsafe_allow_html=True)

            if successful_results:
                csv_data = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_data,
                    file_name="mace_batch_results.csv",
                    mime="text/csv",
                    key=f"download_csv_{len(successful_results)}"
                )

                optimized_structures = [r for r in successful_results if r['calc_type'] == 'Geometry Optimization']
                if optimized_structures:
                    st.subheader("Download Optimized Structures")

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for result in optimized_structures:
                            poscar_content = create_wrapped_poscar_content(result['structure'])
                            zip_file.writestr(f"optimized_{result['name']}", poscar_content)

                    st.download_button(
                        label="📦 Download Optimized Structures (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name="optimized_structures.zip",
                        mime="application/zip",
                        key=f"download_zip_{len(optimized_structures)}"
                    )

        phonon_results = [r for r in st.session_state.results if
                          r.get('phonon_results') and r['phonon_results'].get('success')]
        elastic_results = [r for r in st.session_state.results if
                           r.get('elastic_results') and r['elastic_results'].get('success')]
        with results_tab4:
            if phonon_results:
                st.subheader("🎵 Phonon Properties")

                if len(phonon_results) == 1:
                    selected_phonon = phonon_results[0]
                    st.write(f"**Structure:** {selected_phonon['name']}")
                else:
                    phonon_names = [r['name'] for r in phonon_results]
                    selected_name = st.selectbox("Select structure for phonon analysis:", phonon_names,
                                                 key="phonon_selector")
                    selected_phonon = next(r for r in phonon_results if r['name'] == selected_name)

                phonon_data = selected_phonon['phonon_results']

                col_ph1, col_ph2 = st.columns(2)

                with col_ph1:
                    st.write("**Phonon Dispersion**")

                    frequencies = np.array(phonon_data['frequencies'])
                    nkpts, nbands = frequencies.shape

                    fig_disp = go.Figure()

                    for band in range(nbands):
                        fig_disp.add_trace(go.Scatter(
                            x=list(range(nkpts)),
                            y=frequencies[:, band],
                            mode='lines',
                            name=f'Branch {band + 1}',
                            line=dict(width=1),
                            showlegend=False
                        ))

                    if phonon_data['imaginary_modes'] > 0:
                        imaginary_mask = frequencies < 0
                        for band in range(nbands):
                            imaginary_points = np.where(imaginary_mask[:, band])[0]
                            if len(imaginary_points) > 0:
                                fig_disp.add_trace(go.Scatter(
                                    x=imaginary_points,
                                    y=frequencies[imaginary_points, band],
                                    mode='markers',
                                    marker=dict(color='red', size=4),
                                    name='Imaginary modes',
                                    showlegend=band == 0
                                ))

                    fig_disp.update_layout(
                        title=dict(text="Phonon Dispersion", font=dict(size=24)),
                        xaxis_title="k-point index",
                        yaxis_title="Frequency (meV)",
                        height=750,
                        font=dict(size=20),
                        hoverlabel=dict(
                            bgcolor="white",
                            bordercolor="black",
                            font_size=20,
                            font_family="Arial"
                        ),
                        xaxis=dict(
                            title_font=dict(size=20),
                            tickfont=dict(size=20)
                        ),
                        yaxis=dict(
                            title_font=dict(size=20),
                            tickfont=dict(size=20)
                        ),
                        hovermode='closest'
                    )

                    fig_disp.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

                    st.plotly_chart(fig_disp, use_container_width=True)

                with col_ph2:
                    st.write("**Phonon Density of States**")

                    dos_energies = phonon_data['dos_energies']
                    dos = phonon_data['dos']

                    fig_dos = go.Figure()
                    fig_dos.add_trace(go.Scatter(
                        x=dos,
                        y=dos_energies,
                        mode='lines',
                        fill='tozerox',
                        name='DOS',
                        line=dict(color='blue', width=2)
                    ))

                    fig_dos.update_layout(
                        title=dict(text="Phonon Density of States", font=dict(size=24)),
                        xaxis_title="DOS (states/meV)",
                        yaxis_title="Frequency (meV)",
                        height=750,
                        font=dict(size=20),
                        hoverlabel=dict(
                            bgcolor="white",
                            bordercolor="black",
                            font_size=20,
                            font_family="Arial"
                        ),
                        xaxis=dict(
                            title_font=dict(size=20),
                            tickfont=dict(size=20)
                        ),
                        yaxis=dict(
                            title_font=dict(size=20),
                            tickfont=dict(size=20)
                        ),
                        showlegend=False
                    )

                    fig_dos.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

                    st.plotly_chart(fig_dos, use_container_width=True)

                st.write("**Phonon Analysis Summary**")

                phonon_summary = {
                    'Property': [
                        'Supercell size',
                        'Number of k-points',
                        'Number of imaginary modes',
                        'Minimum frequency (meV)',
                        'Maximum frequency (meV)',
                    ],
                    'Value': [
                        f"{phonon_data['supercell_size']}",
                        f"{len(phonon_data['kpoints'])}",
                        f"{phonon_data['imaginary_modes']}",
                        f"{phonon_data['min_frequency']:.3f}",
                        f"{np.max(phonon_data['frequencies']):.3f}",
                    ]
                }

                if phonon_data.get('thermodynamics'):
                    thermo = phonon_data['thermodynamics']
                    phonon_summary['Property'].extend([
                        f"Temperature (K)",
                        "Zero-point energy (eV)",
                        "Heat capacity (eV/K)",
                        "Entropy (eV/K)",
                        "Free energy (eV)"
                    ])
                    phonon_summary['Value'].extend([
                        f"{thermo['temperature']}",
                        f"{thermo['zero_point_energy']:.6f}",
                        f"{thermo['heat_capacity']:.6f}",
                        f"{thermo['entropy']:.6f}",
                        f"{thermo['free_energy']:.6f}"
                    ])

                df_phonon_summary = pd.DataFrame(phonon_summary)
                st.dataframe(df_phonon_summary, use_container_width=True, hide_index=True)

                if phonon_data['imaginary_modes'] > 0:
                    st.warning(
                        f"⚠️ Structure has {phonon_data['imaginary_modes']} imaginary phonon modes, indicating potential instability.")
                else:
                    st.success("✅ No imaginary modes found - structure appears dynamically stable.")

                phonon_export_data = create_phonon_data_export(phonon_data, selected_phonon['name'])
                if phonon_export_data:
                    phonon_json = json.dumps(phonon_export_data, indent=2)
                    st.download_button(
                        label="📥 Download Phonon Data (JSON)",
                        data=phonon_json,
                        file_name=f"phonon_data_{selected_phonon['name'].replace('.', '_')}.json",
                        mime="application/json"
                    )
                if phonon_data.get('thermal_properties_dict'):
                    st.write("**Temperature-Dependent Analysis**")

                    col_temp1, col_temp2, col_temp3 = st.columns(3)
                    with col_temp1:
                        min_temp = st.number_input("Min Temperature (K)", min_value=0, max_value=2000, value=0, step=10,
                                                   key=f"min_temp_{selected_phonon['name']}")
                    with col_temp2:
                        max_temp = st.number_input("Max Temperature (K)", min_value=100, max_value=2000, value=1000,
                                                   step=50, key=f"max_temp_{selected_phonon['name']}")
                    with col_temp3:
                        temp_step = st.number_input("Temperature Step (K)", min_value=1, max_value=100, value=10,
                                                    step=1,
                                                    key=f"temp_step_{selected_phonon['name']}")

                    if st.button("Generate Temperature Analysis", key=f"temp_analysis_{selected_phonon['name']}"):
                        with st.spinner("Calculating thermodynamics over temperature range..."):
                            fig_temp, thermo_data = add_entropy_vs_temperature_plot(
                                phonon_data,
                                temp_range=(min_temp, max_temp, temp_step)
                            )

                            if fig_temp is not None:
                                st.plotly_chart(fig_temp, use_container_width=True)

                                if isinstance(thermo_data, dict) and 'error' not in thermo_data:
                                    import json

                                    thermo_json = json.dumps({
                                        'structure_name': selected_phonon['name'],
                                        'temperature_dependent_properties': thermo_data
                                    }, indent=2)

                                    st.download_button(
                                        label="📥 Download Temperature-Dependent Data (JSON)",
                                        data=thermo_json,
                                        file_name=f"thermodynamics_vs_temp_{selected_phonon['name'].replace('.', '_')}.json",
                                        mime="application/json",
                                        key=f"download_temp_{selected_phonon['name']}",
                                        type = 'primary'
                                    )
                            else:
                                st.error(f"Error generating analysis: {thermo_data}")

                    st.write("**Quick Temperature Comparison**")
                    target_temps = st.multiselect(
                        "Select specific temperatures (K):",
                        options=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        default=[300, 600, 1000],
                        key=f"target_temps_{selected_phonon['name']}"
                    )

                    if target_temps:
                        specific_data = extract_thermodynamics_at_temperatures(phonon_data, target_temps)

                        if 'error' not in specific_data:
                            comparison_data = []
                            for temp in target_temps:
                                if temp in specific_data:
                                    data = specific_data[temp]
                                    comparison_data.append({
                                        'Temperature (K)': data['temperature'],
                                        'Entropy (eV/K)': f"{data['entropy']:.6f}",
                                        'Heat Capacity (eV/K)': f"{data['heat_capacity']:.6f}",
                                        'Free Energy (eV)': f"{data['free_energy']:.6f}",
                                        'Internal Energy (eV)': f"{data['internal_energy']:.6f}"
                                    })

                            if comparison_data:
                                df_temp_compare = pd.DataFrame(comparison_data)
                                st.dataframe(df_temp_compare, use_container_width=True, hide_index=True)





                if phonon_results and len(phonon_results) > 1:
                    st.subheader("🗺️ Computational Phase Diagram Analysis")

                    structures_dict = {result['name']: result['structure'] for result in st.session_state.results
                                       if result.get('phonon_results') and result['phonon_results'].get('success')}

                    if len(structures_dict) > 1:
                        compositions = extract_element_concentrations(structures_dict)
                        common_elements = get_common_elements(compositions)

                        if common_elements:
                            st.write("**Phase Diagram Parameters**")

                            col_phase1, col_phase2, col_phase3, col_phase4 = st.columns(4)

                            with col_phase1:
                                selected_element = st.selectbox(
                                    "Select element for concentration axis:",
                                    common_elements,
                                    key="phase_element_selector"
                                )

                            with col_phase2:
                                min_temp_phase = st.number_input(
                                    "Min Temperature (K):",
                                    min_value=0, max_value=2000, value=0, step=50,
                                    key="phase_min_temp"
                                )

                            with col_phase3:
                                max_temp_phase = st.number_input(
                                    "Max Temperature (K):",
                                    min_value=100, max_value=3000, value=1000, step=50,
                                    key="phase_max_temp"
                                )

                            with col_phase4:
                                temp_step_phase = st.number_input(
                                    "Temperature Step (K):",
                                    min_value=10, max_value=100, value=25, step=5,
                                    key="phase_temp_step"
                                )

                            col_analysis1, col_analysis2 = st.columns([1, 3])

                            with col_analysis1:
                                if st.button("🔬 Generate Phase Diagram (", type="primary", key="generate_phase_diagram", disabled = True):
                                    with st.spinner("Calculating phase diagram..."):
                                        element_concentrations = {name: comp[selected_element]
                                                                  for name, comp in compositions.items()}

                                        temp_range = list(range(min_temp_phase, max_temp_phase + 1, temp_step_phase))

                                        phase_df = calculate_phase_diagram_data(phonon_results, element_concentrations,
                                                                                temp_range)

                                        if not phase_df.empty:
                                            stable_df = find_stable_phases(phase_df)

                                            st.session_state.phase_diagram_data = {
                                                'phase_df': phase_df,
                                                'stable_df': stable_df,
                                                'selected_element': selected_element,
                                                'element_concentrations': element_concentrations
                                            }

                                            st.success("✅ Phase diagram calculated successfully!")
                                        else:
                                            st.error("❌ No valid phase data calculated")

                            with col_analysis2:
                                if 'phase_diagram_data' in st.session_state:
                                    display_options = st.multiselect(
                                        "Select analysis to display:",
                                        ["Phase Stability Map", "Concentration Heatmaps", "Phase Transition Summary"],
                                        default=["Phase Stability Map"],
                                        key="phase_display_options"
                                    )

                            if 'phase_diagram_data' in st.session_state:
                                phase_data = st.session_state.phase_diagram_data
                                phase_df = phase_data['phase_df']
                                stable_df = phase_data['stable_df']
                                selected_element = phase_data['selected_element']
                                element_concentrations = phase_data['element_concentrations']

                                if "Phase Stability Map" in st.session_state.get('phase_display_options', []):
                                    st.write("**Phase Stability Analysis**")

                                    fig_phase, phase_transitions = create_phase_diagram_plot(
                                        phase_df, stable_df, selected_element
                                    )
                                    st.plotly_chart(fig_phase, use_container_width=True)

                                    col_summary1, col_summary2 = st.columns(2)

                                    with col_summary1:
                                        st.write("**Structure Concentrations**")
                                        conc_summary = []
                                        for name, conc in element_concentrations.items():
                                            conc_summary.append({
                                                'Structure': name,
                                                f'{selected_element} (%)': f"{conc:.1f}"
                                            })
                                        df_conc_summary = pd.DataFrame(conc_summary)
                                        st.dataframe(df_conc_summary, use_container_width=True, hide_index=True)

                                    with col_summary2:
                                        if phase_transitions:
                                            st.write("**Phase Transitions Detected**")
                                            transitions_df = pd.DataFrame(phase_transitions)
                                            st.dataframe(transitions_df, use_container_width=True, hide_index=True)
                                        else:
                                            st.info("No phase transitions detected in temperature range")

                                if "Concentration Heatmaps" in st.session_state.get('phase_display_options', []):
                                    st.write("**Property Heatmaps**")

                                    property_selector = st.selectbox(
                                        "Select property for heatmap:",
                                        ["free_energy", "entropy", "heat_capacity", "internal_energy"],
                                        key="heatmap_property"
                                    )

                                    fig_heatmap = create_concentration_heatmap(phase_df, selected_element,
                                                                               property_selector)
                                    st.plotly_chart(fig_heatmap, use_container_width=True)

                                if "Phase Transition Summary" in st.session_state.get('phase_display_options', []):
                                    st.write("**Thermodynamic Analysis Summary**")

                                    summary_stats = []
                                    for structure in phase_df['structure'].unique():
                                        struct_data = phase_df[phase_df['structure'] == structure]

                                        summary_stats.append({
                                            'Structure': structure,
                                            f'{selected_element} (%)': f"{element_concentrations[structure]:.1f}",
                                            'Min Free Energy (eV)': f"{struct_data['free_energy'].min():.6f}",
                                            'Max Free Energy (eV)': f"{struct_data['free_energy'].max():.6f}",
                                            #'Avg Entropy (eV/K)': f"{struct_data['entropy'].mean():.6f}",
                                            'Max Heat Capacity (eV/K)': f"{struct_data['heat_capacity'].max():.6f}",
                                            'Stable at T_min':
                                                stable_df[stable_df['temperature'] == stable_df['temperature'].min()][
                                                    'stable_structure'].iloc[0] == structure,
                                            'Stable at T_max':
                                                stable_df[stable_df['temperature'] == stable_df['temperature'].max()][
                                                    'stable_structure'].iloc[0] == structure
                                        })

                                    df_summary = pd.DataFrame(summary_stats)
                                    st.dataframe(df_summary, use_container_width=True, hide_index=True)


                        else:
                            st.warning("⚠️ No common elements with varying concentrations found across structures")

                    else:
                        st.info(
                            "Need at least 2 structures with successful phonon calculations for phase diagram analysis")


                def extract_phase_and_composition_from_filename(filename):
                    try:
                        base_name = filename.replace('.vasp', '').replace('POSCAR', '').replace('.', '')
                        parts = base_name.split('_')

                        if len(parts) < 5:
                            return None

                        phase = parts[0].lower()

                        element1_part = parts[1]
                        element1 = ''.join([c for c in element1_part if c.isalpha()])
                        n1 = int(''.join([c for c in element1_part if c.isdigit()]))

                        element2_part = parts[2]
                        element2 = ''.join([c for c in element2_part if c.isalpha()])
                        n2 = int(''.join([c for c in element2_part if c.isdigit()]))

                        conc1 = float(parts[3].replace('c', ''))
                        conc2 = float(parts[4])

                        return phase, element1, element2, n1, n2, conc1, conc2
                    except:
                        return None


                def identify_binary_system_from_results(phonon_results):
                    phase_data = []
                    elements_found = set()
                    phases_found = set()

                    for result in phonon_results:
                        filename = result['name']
                        phase_info = extract_phase_and_composition_from_filename(filename)

                        if phase_info:
                            phase, elem1, elem2, n1, n2, conc1, conc2 = phase_info

                            phase_data.append({
                                'structure_name': filename,
                                'phase': phase,
                                'element1': elem1,
                                'element2': elem2,
                                'n1': n1,
                                'n2': n2,
                                'concentration1': conc1,
                                'concentration2': conc2,
                                'total_atoms': n1 + n2,
                                'phonon_results': result['phonon_results']
                            })

                            elements_found.add(elem1)
                            elements_found.add(elem2)
                            phases_found.add(phase)

                    if len(elements_found) == 2 and len(phases_found) > 1:
                        return phase_data, list(elements_found), list(phases_found)
                    else:
                        return None, None, None


                def calculate_normal_phase_diagram(phase_data, temp_range):
                    diagram_data = []

                    for data in phase_data:
                        phonon_results = data['phonon_results']

                        if not phonon_results['success']:
                            continue
                        temp_thermo = extract_thermodynamics_at_temperatures(phonon_results, temp_range)

                        if 'error' in temp_thermo:
                            continue

                        for temp in temp_range:
                            if temp in temp_thermo:
                                thermo = temp_thermo[temp]

                                diagram_data.append({
                                    'structure_name': data['structure_name'],
                                    'phase': data['phase'],
                                    'element1': data['element1'],
                                    'element2': data['element2'],
                                    'concentration1': data['concentration1'],
                                    'concentration2': data['concentration2'],
                                    'temperature': temp,
                                    'free_energy': thermo['free_energy'],
                                    'entropy': thermo['entropy'],
                                    'heat_capacity': thermo['heat_capacity']
                                })

                    return pd.DataFrame(diagram_data)


                def find_stable_phase_at_each_point(diagram_df):
                    stable_points = []
                    for (conc1, temp), group in diagram_df.groupby(['concentration1', 'temperature']):
                        min_idx = group['free_energy'].idxmin()
                        stable_phase_data = group.loc[min_idx]

                        stable_points.append({
                            'concentration1': conc1,
                            'concentration2': 100 - conc1,
                            'temperature': temp,
                            'stable_phase': stable_phase_data['phase'],
                            'free_energy': stable_phase_data['free_energy'],
                            'structure_name': stable_phase_data['structure_name']
                        })

                    return pd.DataFrame(stable_points)


                def create_normal_phase_diagram_plot(stable_df, phase_data, element1, element2):
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            f'Phase Stability Diagram ({element1}-{element2})',
                            'Phase Boundaries (3D)',
                            'Free Energy Contours',
                            'Phase Fraction vs Temperature'
                        ),
                        specs=[[{"type": "xy"}, {"type": "scatter3d"}],
                               [{"type": "xy"}, {"type": "xy"}]]
                    )

                    phases = stable_df['stable_phase'].unique()
                    phase_colors = {
                        'fcc': '#FF6B6B',  # Red
                        'hcp': '#4ECDC4',  # Teal
                        'bcc': '#45B7D1',  # Blue
                        'liquid': '#FFA07A',  # Light salmon
                        'solid': '#98D8E8',  # Light blue
                        'gas': '#F7DC6F'  # Light yellow
                    }

                    colors = px.colors.qualitative.Set1
                    for i, phase in enumerate(phases):
                        if phase not in phase_colors:
                            phase_colors[phase] = colors[i % len(colors)]
                    for phase in phases:
                        phase_data_filtered = stable_df[stable_df['stable_phase'] == phase]

                        if not phase_data_filtered.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=phase_data_filtered['concentration1'],
                                    y=phase_data_filtered['temperature'],
                                    mode='markers',
                                    marker=dict(
                                        color=phase_colors[phase],
                                        size=8,
                                        opacity=0.8
                                    ),
                                    name=f'{phase.upper()}',
                                    hovertemplate=f'<b>{phase.upper()}</b><br>' +
                                                  f'{element1}: %{{x:.1f}}%<br>' +
                                                  'T: %{y}K<br>' +
                                                  'F: %{customdata:.4f} eV<extra></extra>',
                                    customdata=phase_data_filtered['free_energy']
                                ),
                                row=1, col=1
                            )

                    conc_range = np.linspace(stable_df['concentration1'].min(),
                                             stable_df['concentration1'].max(), 20)
                    temp_range = np.linspace(stable_df['temperature'].min(),
                                             stable_df['temperature'].max(), 20)

                    conc_mesh, temp_mesh = np.meshgrid(conc_range, temp_range)

                    from scipy.interpolate import griddata

                    points = stable_df[['concentration1', 'temperature']].values
                    values = stable_df['free_energy'].values

                    free_energy_mesh = griddata(points, values, (conc_mesh, temp_mesh), method='linear')

                    fig.add_trace(
                        go.Surface(
                            x=conc_mesh,
                            y=temp_mesh,
                            z=free_energy_mesh,
                            colorscale='RdYlBu_r',
                            showscale=False,
                            opacity=0.8,
                            name='Free Energy Surface'
                        ),
                        row=1, col=2
                    )
                    fig.add_trace(
                        go.Contour(
                            x=conc_range,
                            y=temp_range,
                            z=free_energy_mesh,
                            colorscale='RdYlBu_r',
                            showscale=True,
                            colorbar=dict(title="Free Energy (eV)", x=0.45),
                            contours=dict(
                                coloring='heatmap',
                                showlabels=True,
                                labelfont=dict(size=10)
                            ),
                            name='Free Energy Contours'
                        ),
                        row=2, col=1
                    )
                    selected_compositions = [0, 25, 50, 75, 100]

                    for conc in selected_compositions:
                        closest_conc = stable_df['concentration1'].iloc[
                            (stable_df['concentration1'] - conc).abs().argsort()[:1]
                        ].iloc[0]

                        conc_data = stable_df[stable_df['concentration1'] == closest_conc]

                        if not conc_data.empty:
                            temps = sorted(conc_data['temperature'].unique())
                            phase_fractions = []

                            for temp in temps:
                                temp_data = conc_data[conc_data['temperature'] == temp]
                                if not temp_data.empty:
                                    dominant_phase = temp_data.iloc[0]['stable_phase']
                                    phase_fractions.append(1 if dominant_phase == phases[0] else 0)
                                else:
                                    phase_fractions.append(0)

                            fig.add_trace(
                                go.Scatter(
                                    x=temps,
                                    y=phase_fractions,
                                    mode='lines+markers',
                                    name=f'{element1} {closest_conc:.0f}%',
                                    line=dict(width=2),
                                    showlegend=False
                                ),
                                row=2, col=2
                            )

                    fig.update_xaxes(title_text=f"{element1} Concentration (%)", row=1, col=1)
                    fig.update_xaxes(title_text=f"{element1} Concentration (%)", row=2, col=1)
                    fig.update_xaxes(title_text="Temperature (K)", row=2, col=2)

                    fig.update_yaxes(title_text="Temperature (K)", row=1, col=1)
                    fig.update_yaxes(title_text="Temperature (K)", row=2, col=1)
                    fig.update_yaxes(title_text="Phase Indicator", row=2, col=2)

                    fig.update_layout(
                        height=900,
                        title_text=f"Binary Phase Diagram: {element1}-{element2} System",
                        showlegend=True,
                        legend=dict(x=1.02, y=1),
                        scene=dict(
                            xaxis_title=f"{element1} Concentration (%)",
                            yaxis_title="Temperature (K)",
                            zaxis_title="Free Energy (eV)"
                        )
                    )

                    return fig


                def create_phase_region_plot(stable_df, element1, element2):
                    phase_pivot = stable_df.pivot_table(
                        values='stable_phase',
                        index='temperature',
                        columns='concentration1',
                        aggfunc='first'
                    )

                    phases = stable_df['stable_phase'].unique()
                    phase_to_num = {phase: i for i, phase in enumerate(phases)}
                    num_to_phase = {i: phase for phase, i in phase_to_num.items()}

                    Z = np.zeros(phase_pivot.shape)
                    for i, temp in enumerate(phase_pivot.index):
                        for j, conc in enumerate(phase_pivot.columns):
                            phase = phase_pivot.loc[temp, conc]
                            if pd.notna(phase):
                                Z[i, j] = phase_to_num[phase]
                            else:
                                Z[i, j] = -1  # No data

                    phase_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8E8', '#F7DC6F']
                    colorscale = []
                    n_phases = len(phases)

                    for i, phase in enumerate(phases):
                        color_val = i / (n_phases - 1) if n_phases > 1 else 0
                        colorscale.extend([
                            [color_val, phase_colors[i % len(phase_colors)]],
                            [color_val, phase_colors[i % len(phase_colors)]]
                        ])

                    fig = go.Figure(data=go.Heatmap(
                        z=Z,
                        x=phase_pivot.columns,
                        y=phase_pivot.index,
                        colorscale=colorscale,
                        showscale=False,
                        hovertemplate=f'{element1}: %{{x:.1f}}%<br>' +
                                      'T: %{y}K<br>' +
                                      'Phase: %{customdata}<extra></extra>',
                        customdata=[[num_to_phase.get(Z[i, j], 'Unknown') for j in range(Z.shape[1])]
                                    for i in range(Z.shape[0])]
                    ))
                    from scipy import ndimage

                    boundaries = ndimage.sobel(Z)
                    boundary_y, boundary_x = np.where(np.abs(boundaries) > 0.5)

                    if len(boundary_x) > 0:
                        fig.add_trace(go.Scatter(
                            x=[phase_pivot.columns[x] for x in boundary_x],
                            y=[phase_pivot.index[y] for y in boundary_y],
                            mode='markers',
                            marker=dict(size=1, color='black'),
                            name='Phase Boundaries',
                            showlegend=False
                        ))

                    for i, phase in enumerate(phases):
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(
                                size=15,
                                color=phase_colors[i % len(phase_colors)],
                                symbol='square'
                            ),
                            name=f'{phase.upper()}',
                            showlegend=True
                        ))

                    fig.update_layout(
                        title=f"Phase Regions: {element1}-{element2} Binary System",
                        xaxis_title=f"{element1} Concentration (%)",
                        yaxis_title="Temperature (K)",
                        height=600,
                        legend=dict(x=1.02, y=1)
                    )

                    return fig


                def export_phase_diagram_data(stable_df, phase_data, element1, element2):

                    export_data = {
                        'metadata': {
                            'system_type': 'binary_alloy',
                            'element1': element1,
                            'element2': element2,
                            'phases_analyzed': list(stable_df['stable_phase'].unique()),
                            'temperature_range': [float(stable_df['temperature'].min()),
                                                  float(stable_df['temperature'].max())],
                            'composition_range': [float(stable_df['concentration1'].min()),
                                                  float(stable_df['concentration1'].max())],
                            'total_data_points': len(stable_df)
                        },
                        'phase_boundaries': [],
                        'stable_phases': stable_df.to_dict('records'),
                        'phase_transitions': []
                    }

                    for conc in stable_df['concentration1'].unique():
                        conc_data = stable_df[stable_df['concentration1'] == conc].sort_values('temperature')

                        transitions = []
                        for i in range(len(conc_data) - 1):
                            if conc_data.iloc[i]['stable_phase'] != conc_data.iloc[i + 1]['stable_phase']:
                                transitions.append({
                                    'composition': float(conc),
                                    'temperature': float(conc_data.iloc[i + 1]['temperature']),
                                    'from_phase': conc_data.iloc[i]['stable_phase'],
                                    'to_phase': conc_data.iloc[i + 1]['stable_phase']
                                })

                        export_data['phase_transitions'].extend(transitions)

                    return json.dumps(export_data, indent=2)

                if phonon_results and len(phonon_results) > 1:

                    phase_data, elements, phases = identify_binary_system_from_results(phonon_results)

                    if phase_data and len(elements) == 2 and len(phases) > 1:
                        st.subheader("🔬 Binary Alloy Phase Diagram Analysis")

                        element1, element2 = elements
                        st.info(
                            f"Detected binary system: **{element1}-{element2}** with phases: **{', '.join(phases).upper()}**")

                        st.write("**Phase Diagram Parameters**")
                        col_normal1, col_normal2, col_normal3 = st.columns(3)

                        with col_normal1:
                            min_temp_normal = st.number_input(
                                "Min Temperature (K):",
                                min_value=0, max_value=2000, value=300, step=50,
                                key="normal_min_temp"
                            )

                        with col_normal2:
                            max_temp_normal = st.number_input(
                                "Max Temperature (K):",
                                min_value=100, max_value=3000, value=1200, step=50,
                                key="normal_max_temp"
                            )

                        with col_normal3:
                            temp_step_normal = st.number_input(
                                "Temperature Step (K):",
                                min_value=10, max_value=100, value=50, step=10,
                                key="normal_temp_step"
                            )

                        col_calc1, col_calc2 = st.columns([1, 3])

                        with col_calc1:
                            if st.button("🗺️ Calculate Phase Diagram", type="primary", key="calc_normal_phase"):
                                with st.spinner("Calculating binary phase diagram..."):
                                    temp_range = list(range(min_temp_normal, max_temp_normal + 1, temp_step_normal))

                                    diagram_df = calculate_normal_phase_diagram(phase_data, temp_range)

                                    if not diagram_df.empty:
                                        stable_df = find_stable_phase_at_each_point(diagram_df)

                                        st.session_state.normal_phase_data = {
                                            'diagram_df': diagram_df,
                                            'stable_df': stable_df,
                                            'element1': element1,
                                            'element2': element2,
                                            'phases': phases
                                        }

                                        st.success("✅ Phase diagram calculated successfully!")
                                    else:
                                        st.error("❌ No valid phase diagram data calculated")

                        with col_calc2:
                            if 'normal_phase_data' in st.session_state:
                                plot_options = st.multiselect(
                                    "Select visualizations:",
                                    ["Complete Phase Diagram", "Phase Regions", "Thermodynamic Analysis"],
                                    default=["Complete Phase Diagram"],
                                    key="normal_plot_options"
                                )
                        if 'normal_phase_data' in st.session_state:
                            normal_data = st.session_state.normal_phase_data
                            diagram_df = normal_data['diagram_df']
                            stable_df = normal_data['stable_df']
                            element1 = normal_data['element1']
                            element2 = normal_data['element2']
                            phases = normal_data['phases']

                            if "Complete Phase Diagram" in st.session_state.get('normal_plot_options', []):
                                st.write("**Complete Phase Diagram Analysis**")

                                fig_normal = create_normal_phase_diagram_plot(stable_df, phase_data, element1, element2)
                                st.plotly_chart(fig_normal, use_container_width=True)

                            if "Phase Regions" in st.session_state.get('normal_plot_options', []):
                                st.write("**Phase Stability Regions**")

                                fig_regions = create_phase_region_plot(stable_df, element1, element2)
                                st.plotly_chart(fig_regions, use_container_width=True)

                            if "Thermodynamic Analysis" in st.session_state.get('normal_plot_options', []):
                                st.write("**Phase Statistics and Transitions**")

                                col_stats1, col_stats2 = st.columns(2)

                                with col_stats1:
                                    st.write("**Phase Stability Statistics**")
                                    phase_stats = stable_df['stable_phase'].value_counts()
                                    stats_df = pd.DataFrame({
                                        'Phase': phase_stats.index,
                                        'Stable Points': phase_stats.values,
                                        'Percentage': (phase_stats.values / len(stable_df) * 100).round(1)
                                    })
                                    st.dataframe(stats_df, use_container_width=True, hide_index=True)

                                with col_stats2:
                                    st.write("**Temperature Ranges by Phase**")
                                    temp_ranges = []
                                    for phase in phases:
                                        phase_temps = stable_df[stable_df['stable_phase'] == phase]['temperature']
                                        if not phase_temps.empty:
                                            temp_ranges.append({
                                                'Phase': phase.upper(),
                                                'Min Temp (K)': int(phase_temps.min()),
                                                'Max Temp (K)': int(phase_temps.max()),
                                                'Range (K)': int(phase_temps.max() - phase_temps.min())
                                            })

                                    if temp_ranges:
                                        temp_df = pd.DataFrame(temp_ranges)
                                        st.dataframe(temp_df, use_container_width=True, hide_index=True)

                                transitions = []
                                for conc in sorted(stable_df['concentration1'].unique()):
                                    conc_data = stable_df[stable_df['concentration1'] == conc].sort_values(
                                        'temperature')

                                    for i in range(len(conc_data) - 1):
                                        if conc_data.iloc[i]['stable_phase'] != conc_data.iloc[i + 1]['stable_phase']:
                                            transitions.append({
                                                f'{element1} (%)': conc,
                                                f'{element2} (%)': 100 - conc,
                                                'Transition T (K)': conc_data.iloc[i + 1]['temperature'],
                                                'From Phase': conc_data.iloc[i]['stable_phase'].upper(),
                                                'To Phase': conc_data.iloc[i + 1]['stable_phase'].upper()
                                            })

                                if transitions:
                                    st.write("**Detected Phase Transitions**")
                                    transitions_df = pd.DataFrame(transitions)
                                    st.dataframe(transitions_df, use_container_width=True, hide_index=True)
                                else:
                                    st.info("No phase transitions detected in the analyzed temperature range")

                            st.write("**Export Phase Diagram Data**")
                            col_export1, col_export2, col_export3 = st.columns(3)

                            with col_export1:
                                phase_diagram_json = export_phase_diagram_data(stable_df, phase_data, element1,
                                                                               element2)
                                st.download_button(
                                    label="📥 Download Phase Diagram (JSON)",
                                    data=phase_diagram_json,
                                    file_name=f"phase_diagram_{element1}_{element2}.json",
                                    mime="application/json",
                                    key="download_normal_phase_json"
                                )

                            with col_export2:
                                stable_csv = stable_df.to_csv(index=False)
                                st.download_button(
                                    label="📊 Download Stable Phases (CSV)",
                                    data=stable_csv,
                                    file_name=f"stable_phases_{element1}_{element2}.csv",
                                    mime="text/csv",
                                    key="download_stable_csv"
                                )

                            with col_export3:
                                full_data_csv = diagram_df.to_csv(index=False)
                                st.download_button(
                                    label="📈 Download Full Data (CSV)",
                                    data=full_data_csv,
                                    file_name=f"full_phase_data_{element1}_{element2}.csv",
                                    mime="text/csv",
                                    key="download_full_csv"
                                )

                    else:
                        if phonon_results:
                            st.info("💡 **Binary Phase Diagram Analysis**")
                            st.write("""
                            To generate binary alloy phase diagrams, upload structures with naming convention:
                            - `{phase}_{Element1}{count1}_{Element2}{count2}_c{conc1}_{conc2}.vasp`
                            - Example: `fcc_Ti7_Ag1_c87_13.vasp`

                            **Supported phases**: FCC, HCP, BCC, Liquid

                            Use the separate structure generator script to create these automatically!
                            """)

                            with st.expander("📖 Understanding Binary Phase Diagrams"):
                                st.markdown("""
                                **What you'll see:**

                                🔴 **Phase Stability Map**: Shows which crystal structure (FCC/HCP/BCC/Liquid) is thermodynamically stable at each composition and temperature

                                🔵 **Phase Regions**: Colored areas showing where each phase dominates

                                🟢 **Phase Boundaries**: Lines separating different stable regions

                                🟡 **Phase Transitions**: Temperature points where one phase becomes more stable than another

                                **Real-world applications:**
                                - Alloy design and processing conditions
                                - Understanding why certain phases form during synthesis
                                - Predicting material properties at operating temperatures
                                - Optimizing heat treatment procedures
                                """)
                else:
                    st.info(
                        "Binary phase diagram analysis requires multiple structures with successful phonon calculations")

        with results_tab3:
            if elastic_results:
                st.subheader("🔧 Elastic Properties")

                if len(elastic_results) == 1:
                    selected_elastic = elastic_results[0]
                    st.write(f"**Structure:** {selected_elastic['name']}")
                else:
                    elastic_names = [r['name'] for r in elastic_results]
                    selected_name = st.selectbox("Select structure for elastic analysis:", elastic_names,
                                                 key="elastic_selector")
                    selected_elastic = next(r for r in elastic_results if r['name'] == selected_name)

                elastic_data = selected_elastic['elastic_results']
                col_el1, col_el2 = st.columns([1, 1])
                with col_el1:
                    st.write("**Elastic Tensor (GPa)**")

                    elastic_tensor = np.array(elastic_data['elastic_tensor'])

                    fig_tensor = go.Figure(data=go.Heatmap(
                        z=elastic_tensor,
                        x=['11', '22', '33', '23', '13', '12'],
                        y=['11', '22', '33', '23', '13', '12'],
                        colorscale='RdBu_r',
                        colorbar=dict(
                            title="C_ij (GPa)",
                            title_font=dict(size=18),
                            tickfont=dict(size=18)
                        ),
                        text=[[f"{val:.1f}" for val in row] for row in elastic_tensor],
                        texttemplate="%{text}",
                        textfont={"size": 20},
                        hovertemplate='C<sub>%{y}%{x}</sub> = %{z:.2f} GPa<extra></extra>'
                    ))

                    fig_tensor.update_layout(
                        title="Elastic Tensor C<sub>ij</sub>",
                        xaxis_title="j",
                        yaxis_title="i",
                        height=400,
                        xaxis=dict(
                            type='category',
                            tickvals=['11', '22', '33', '23', '13', '12'],
                            ticktext=['11', '22', '33', '23', '13', '12'],
                            tickfont=dict(size=16),
                            title_font=dict(size=18)
                        ),
                        yaxis=dict(
                            type='category',
                            tickvals=['11', '22', '33', '23', '13', '12'],
                            ticktext=['11', '22', '33', '23', '13', '12'],
                            autorange='reversed',
                            tickfont=dict(size=16),
                            title_font=dict(size=18)
                        ),
                        hoverlabel=dict(
                            bgcolor="white",
                            bordercolor="black",
                            font_size=16,
                            font_family="Arial"
                        ),
                    )

                    st.plotly_chart(fig_tensor, use_container_width=True)

                with col_el2:
                    st.write("**Elastic Moduli Comparison**")

                    bulk_data = elastic_data['bulk_modulus']
                    shear_data = elastic_data['shear_modulus']

                    moduli_comparison = {
                        'Method': ['Voigt', 'Reuss', 'Hill'],
                        'Bulk Modulus (GPa)': [
                            bulk_data['voigt'],
                            bulk_data['reuss'] if bulk_data['reuss'] else 'N/A',
                            bulk_data['hill'] if bulk_data['hill'] else 'N/A'
                        ],
                        'Shear Modulus (GPa)': [
                            shear_data['voigt'],
                            shear_data['reuss'] if shear_data['reuss'] else 'N/A',
                            shear_data['hill'] if shear_data['hill'] else 'N/A'
                        ]
                    }

                    df_moduli = pd.DataFrame(moduli_comparison)
                    st.dataframe(df_moduli, use_container_width=True, hide_index=True)

                    properties = ['Bulk Modulus', 'Shear Modulus', "Young's Modulus"]
                    values = [
                        bulk_data['hill'] if bulk_data['hill'] is not None else bulk_data['voigt'],
                        shear_data['hill'] if shear_data['hill'] is not None else shear_data['voigt'],
                        elastic_data['youngs_modulus']
                    ]

                    fig_moduli = go.Figure(data=go.Bar(
                        x=properties,
                        y=values,
                        marker_color=['steelblue', 'orange', 'green'],
                        text=[f"{v:.1f}" for v in values],
                        textposition='auto'
                    ))

                    fig_moduli.update_layout(
                        title="Key Elastic Moduli",
                        title_font_size=22,
                        yaxis_title="Modulus (GPa)",
                        font_size=22,
                        height=300,
                        showlegend=False
                    )
                    fig_moduli.update_layout(
                        title="Key Elastic Moduli",
                        title_font_size=24,
                        yaxis_title="Modulus (GPa)",
                        font_size=16,
                        xaxis=dict(
                            tickfont=dict(size=20),
                            title_font=dict(size=22)
                        ),
                        yaxis=dict(
                            tickfont=dict(size=20),
                            title_font=dict(size=22)
                        ),
                        height=300,
                        showlegend=False
                    )

                    st.plotly_chart(fig_moduli, use_container_width=True)

                st.write("**Detailed Elastic Properties**")

                display_format = st.radio("Display format:", ["Table", "Cards"], horizontal=True, index=1)

                elastic_properties = [
                    {
                        'name': 'Bulk Modulus',
                        'value': elastic_data['bulk_modulus']['hill'] if elastic_data['bulk_modulus'][
                                                                             'hill'] is not None else
                        elastic_data['bulk_modulus']['voigt'],
                        'unit': 'GPa',
                        'format': '.1f'
                    },
                    {
                        'name': 'Shear Modulus',
                        'value': elastic_data['shear_modulus']['hill'] if elastic_data['shear_modulus'][
                                                                              'hill'] is not None else
                        elastic_data['shear_modulus']['voigt'],
                        'unit': 'GPa',
                        'format': '.1f'
                    },
                    {
                        'name': "Young's Modulus",
                        'value': elastic_data['youngs_modulus'],
                        'unit': 'GPa',
                        'format': '.1f'
                    },
                    {
                        'name': "Poisson's Ratio",
                        'value': elastic_data['poisson_ratio'],
                        'unit': '',
                        'format': '.3f'
                    },
                    {
                        'name': 'Density',
                        'value': elastic_data['density'],
                        'unit': 'g/cm³',
                        'format': '.3f'
                    },
                    {
                        'name': 'Longitudinal Velocity',
                        'value': elastic_data['wave_velocities']['longitudinal'],
                        'unit': 'm/s',
                        'format': '.0f'
                    },
                    {
                        'name': 'Transverse Velocity',
                        'value': elastic_data['wave_velocities']['transverse'],
                        'unit': 'm/s',
                        'format': '.0f'
                    },
                    {
                        'name': 'Average Velocity',
                        'value': elastic_data['wave_velocities']['average'],
                        'unit': 'm/s',
                        'format': '.0f'
                    },
                    {
                        'name': 'Debye Temperature',
                        'value': elastic_data['debye_temperature'],
                        'unit': 'K',
                        'format': '.1f'
                    },
                    {
                        'name': 'Strain Magnitude',
                        'value': elastic_data['strain_magnitude'] * 100,
                        'unit': '%',
                        'format': '.1f'
                    }
                ]

                property_colors = [
                    "#2E4057",  # Dark Blue-Gray - Bulk Modulus
                    "#4A6741",  # Dark Forest Green - Shear Modulus
                    "#6B73FF",  # Purple-Blue - Young's Modulus
                    "#FF8C00",  # Dark Orange - Poisson's Ratio
                    "#4ECDC4",  # Teal - Density
                    "#45B7D1",  # Blue - Longitudinal Velocity
                    "#96CEB4",  # Green - Transverse Velocity
                    "#FECA57",  # Yellow - Average Velocity
                    "#DDA0DD",  # Plum - Debye Temperature
                    "#FF6B6B"  # Red - Strain Magnitude
                ]

                if display_format == "Cards":
                    cols = st.columns(4)
                    for i, prop in enumerate(elastic_properties):
                        with cols[i % 4]:
                            color = property_colors[i % len(property_colors)]
                            formatted_value = f"{prop['value']:{prop['format']}}"

                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, {color}, {color}CC);
                                padding: 20px;
                                border-radius: 15px;
                                text-align: center;
                                margin: 10px 0;
                                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                                border: 2px solid rgba(255,255,255,0.2);
                                height: 160px;
                                display: flex;
                                flex-direction: column;
                                justify-content: center;
                            ">
                                <h3 style="
                                    color: white;
                                    font-size: 1.2em;
                                    margin: 0 0 5px 0;
                                    text-shadow: 1px 1px 2px rgba(0,0,0,0.4);
                                    font-weight: bold;
                                    line-height: 1.2;
                                ">{prop['name']}</h3>
                                <h1 style="
                                    color: white;
                                    font-size: 2.5em;
                                    margin: 0;
                                    text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
                                    font-weight: bold;
                                ">{formatted_value} <span style="font-size: 0.6em; opacity: 0.9;">{prop['unit']}</span></h1>
                            </div>
                            """, unsafe_allow_html=True)

                else:
                    elastic_summary = {
                        'Property': [
                            'Bulk Modulus (GPa)',
                            'Shear Modulus (GPa)',
                            "Young's Modulus (GPa)",
                            "Poisson's Ratio",
                            'Density (g/cm³)',
                            'Longitudinal wave velocity (m/s)',
                            'Transverse wave velocity (m/s)',
                            'Average wave velocity (m/s)',
                            'Debye temperature (K)',
                            'Strain magnitude used (%)'
                        ],
                        'Value': [
                            f"{elastic_data['bulk_modulus']['hill'] if elastic_data['bulk_modulus']['hill'] is not None else elastic_data['bulk_modulus']['voigt']:.1f}",
                            f"{elastic_data['shear_modulus']['hill'] if elastic_data['shear_modulus']['hill'] is not None else elastic_data['shear_modulus']['voigt']:.1f}",
                            f"{elastic_data['youngs_modulus']:.1f}",
                            f"{elastic_data['poisson_ratio']:.3f}",
                            f"{elastic_data['density']:.3f}",
                            f"{elastic_data['wave_velocities']['longitudinal']:.0f}",
                            f"{elastic_data['wave_velocities']['transverse']:.0f}",
                            f"{elastic_data['wave_velocities']['average']:.0f}",
                            f"{elastic_data['debye_temperature']:.1f}",
                            f"{elastic_data['strain_magnitude'] * 100:.1f}"
                        ]
                    }

                    df_elastic_summary = pd.DataFrame(elastic_summary)
                    st.dataframe(df_elastic_summary, use_container_width=True, hide_index=True)

                st.write("**Mechanical Stability Analysis**")

                stability = elastic_data['mechanical_stability']
                if stability.get('mechanically_stable', False):
                    st.success("✅ Crystal is mechanically stable")

                    with st.expander("Detailed Stability Criteria"):
                        stability_details = []
                        for criterion, value in stability.items():
                            if criterion != 'mechanically_stable' and isinstance(value, bool):
                                status = "✅ Pass" if value else "❌ Fail"
                                stability_details.append({
                                    'Criterion': criterion.replace('_', ' ').title(),
                                    'Status': status
                                })

                        if stability_details:
                            df_stability = pd.DataFrame(stability_details)
                            st.dataframe(df_stability, use_container_width=True, hide_index=True)
                else:
                    st.error("❌ Crystal may be mechanically unstable")
                    st.warning("Check the elastic tensor eigenvalues and Born stability criteria")

                if bulk_data['reuss'] and shear_data['reuss'] and shear_data['reuss'] != 0 and bulk_data['reuss'] != 0:
                    A_U = 5 * (shear_data['voigt'] / shear_data['reuss']) + (
                                bulk_data['voigt'] / bulk_data['reuss']) - 6

                    elastic_tensor = np.array(elastic_data['elastic_tensor'])
                    if abs(elastic_tensor[0, 0] - elastic_tensor[1, 1]) < 10:
                        denominator = (elastic_tensor[0, 0] - elastic_tensor[0, 1])
                        if denominator != 0:
                            A_Z = 2 * elastic_tensor[3, 3] / denominator
                            st.write("**Elastic Anisotropy**")
                            anisotropy_data = {
                                'Index': ['Universal Anisotropy (A_U)', 'Zener Anisotropy (A_Z)'],
                                'Value': [f"{A_U:.3f}", f"{A_Z:.3f}"],
                                'Interpretation': [
                                    "Isotropic" if abs(A_U) < 0.1 else "Anisotropic",
                                    "Isotropic" if abs(A_Z - 1) < 0.1 else "Anisotropic"
                                ]
                            }
                            df_anisotropy = pd.DataFrame(anisotropy_data)
                            st.dataframe(df_anisotropy, use_container_width=True, hide_index=True)
                        else:
                            st.warning("Cannot calculate Zener anisotropy (C11 - C12 is zero).")

                elastic_export_data = create_elastic_data_export(elastic_data, selected_elastic['name'])
                if elastic_export_data:
                    elastic_json = json.dumps(elastic_export_data, indent=2)
                    st.download_button(
                        label="📥 Download Elastic Data (JSON)",
                        data=elastic_json,
                        file_name=f"elastic_data_{selected_elastic['name'].replace('.', '_')}.json",
                        mime="application/json"
                    )
            else:
                st.info(
                    "No completed calculations of elastic properties found. Results will appear here after this type of calculation is finish.")
        with results_tab2:
            with results_tab2:
                st.subheader("Geometry Optimization Details")
                geometry_results = [r for r in st.session_state.results if
                                    r['calc_type'] == 'Geometry Optimization' and r['energy'] is not None]

                if geometry_results:
                    st.info(f"Found {len(geometry_results)} completed geometry optimizations")
                    st.subheader("📐 Lattice Parameters Comparison")

                    lattice_data = []
                    for result in geometry_results:
                        try:
                            initial_structure = st.session_state.structures.get(result['name'])
                            final_structure = result['structure']

                            if initial_structure and final_structure:
                                initial_lattice = initial_structure.lattice
                                final_lattice = final_structure.lattice

                                a_change = ((final_lattice.a - initial_lattice.a) / initial_lattice.a) * 100
                                b_change = ((final_lattice.b - initial_lattice.b) / initial_lattice.b) * 100
                                c_change = ((final_lattice.c - initial_lattice.c) / initial_lattice.c) * 100

                                alpha_change = final_lattice.alpha - initial_lattice.alpha
                                beta_change = final_lattice.beta - initial_lattice.beta
                                gamma_change = final_lattice.gamma - initial_lattice.gamma

                                volume_change = ((
                                                         final_lattice.volume - initial_lattice.volume) / initial_lattice.volume) * 100

                                lattice_data.append({
                                    'Structure': result['name'],
                                    'Initial a (Å)': f"{initial_lattice.a:.4f}",
                                    'Final a (Å)': f"{final_lattice.a:.4f}",
                                    'Δa (%)': f"{a_change:+.2f}",
                                    'Initial b (Å)': f"{initial_lattice.b:.4f}",
                                    'Final b (Å)': f"{final_lattice.b:.4f}",
                                    'Δb (%)': f"{b_change:+.2f}",
                                    'Initial c (Å)': f"{initial_lattice.c:.4f}",
                                    'Final c (Å)': f"{final_lattice.c:.4f}",
                                    'Δc (%)': f"{c_change:+.2f}",
                                    'Initial α (°)': f"{initial_lattice.alpha:.2f}",
                                    'Final α (°)': f"{final_lattice.alpha:.2f}",
                                    'Δα (°)': f"{alpha_change:+.2f}",
                                    'Initial β (°)': f"{initial_lattice.beta:.2f}",
                                    'Final β (°)': f"{final_lattice.beta:.2f}",
                                    'Δβ (°)': f"{beta_change:+.2f}",
                                    'Initial γ (°)': f"{initial_lattice.gamma:.2f}",
                                    'Final γ (°)': f"{final_lattice.gamma:.2f}",
                                    'Δγ (°)': f"{gamma_change:+.2f}",
                                    'Initial Vol (Å³)': f"{initial_lattice.volume:.2f}",
                                    'Final Vol (Å³)': f"{final_lattice.volume:.2f}",
                                    'ΔVol (%)': f"{volume_change:+.2f}",
                                    'Convergence': result.get('convergence_status', 'Unknown')
                                })
                        except Exception as e:
                            st.warning(f"Could not process lattice data for {result['name']}: {str(e)}")

                    if lattice_data:
                        df_lattice = pd.DataFrame(lattice_data)
                        st.dataframe(df_lattice, use_container_width=True, hide_index=True)

                        lattice_csv = df_lattice.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Lattice Parameters (CSV)",
                            data=lattice_csv,
                            file_name="lattice_parameters_comparison.csv",
                            mime="text/csv"
                        )

                    if len(lattice_data) > 1:
                        st.subheader("📊 Lattice Parameter Changes")

                        col_vis1, col_vis2 = st.columns(2)

                        with col_vis1:
                            structures = [data['Structure'] for data in lattice_data]
                            a_changes = [float(data['Δa (%)'].replace('+', '')) for data in lattice_data]
                            b_changes = [float(data['Δb (%)'].replace('+', '')) for data in lattice_data]
                            c_changes = [float(data['Δc (%)'].replace('+', '')) for data in lattice_data]

                            fig_lattice = go.Figure()
                            fig_lattice.add_trace(go.Bar(name='Δa (%)', x=structures, y=a_changes, marker_color='red'))
                            fig_lattice.add_trace(
                                go.Bar(name='Δb (%)', x=structures, y=b_changes, marker_color='green'))
                            fig_lattice.add_trace(go.Bar(name='Δc (%)', x=structures, y=c_changes, marker_color='blue'))

                            fig_lattice.update_layout(
                                title=dict(text="Lattice Parameter Changes (%)", font=dict(size=24)),
                                xaxis_title="Structure",
                                yaxis_title="Change (%)",
                                barmode='group',
                                height=750,
                                font=dict(size=20),
                                hoverlabel=dict(
                                    bgcolor="white",
                                    bordercolor="black",
                                    font_size=20,
                                    font_family="Arial"
                                ),
                                xaxis=dict(
                                    tickangle=45,
                                    title_font=dict(size=20),
                                    tickfont=dict(size=20)
                                ),
                                yaxis=dict(
                                    title_font=dict(size=20),
                                    tickfont=dict(size=20)
                                ),
                                legend=dict(
                                    font=dict(size=20)
                                )
                            )

                            st.plotly_chart(fig_lattice, use_container_width=True)

                        with col_vis2:
                            volume_changes = [float(data['ΔVol (%)'].replace('+', '')) for data in lattice_data]

                            fig_volume = go.Figure()
                            fig_volume.add_trace(go.Bar(
                                x=structures,
                                y=volume_changes,
                                marker_color=['green' if v < 0 else 'red' for v in volume_changes],
                                text=[f"{v:+.2f}%" for v in volume_changes],
                                textposition='auto',
                                textfont=dict(size=16)
                            ))

                            fig_volume.update_layout(
                                title=dict(text="Volume Changes (%)", font=dict(size=24)),
                                xaxis_title="Structure",
                                yaxis_title="Volume Change (%)",
                                height=750,
                                font=dict(size=20),
                                hoverlabel=dict(
                                    bgcolor="white",
                                    bordercolor="black",
                                    font_size=20,
                                    font_family="Arial"
                                ),
                                xaxis=dict(
                                    tickangle=45,
                                    title_font=dict(size=20),
                                    tickfont=dict(size=20)
                                ),
                                yaxis=dict(
                                    title_font=dict(size=20),
                                    tickfont=dict(size=20)
                                )
                            )

                            st.plotly_chart(fig_volume, use_container_width=True)

                    st.subheader("📁 Download Optimized Structures")

                    col_download1, col_download2 = st.columns([2, 1])

                    with col_download1:
                        st.write("Download individual optimized POSCAR files:")

                        for result in geometry_results:
                            col_struct, col_btn = st.columns([3, 1])

                            with col_struct:
                                convergence_icon = "✅" if "CONVERGED" in result.get('convergence_status', '') else "⚠️"
                                st.write(
                                    f"{convergence_icon} **{result['name']}** - {result.get('convergence_status', 'Unknown')}")

                            with col_btn:
                                if 'structure' in result and result['structure']:
                                    poscar_content = create_wrapped_poscar_content(result['structure'])
                                    filename = f"optimized_{result['name']}"

                                    st.download_button(
                                        label="📥 POSCAR",
                                        data=poscar_content,
                                        file_name=filename,
                                        mime="text/plain",
                                        key=f"poscar_{result['name']}",
                                        type = 'primary'
                                    )

                    with col_download2:
                        if len(geometry_results) > 1:
                            st.write("**Bulk Download:**")

                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                                for result in geometry_results:
                                    if 'structure' in result and result['structure']:
                                        poscar_content = create_wrapped_poscar_content(result['structure'])
                                        filename = f"optimized_{result['name']}"
                                        zip_file.writestr(filename, poscar_content)

                            st.download_button(
                                label="📦 Download All (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name="optimized_structures.zip",
                                mime="application/zip", type = 'primary'
                            )

                    st.subheader("📈 Optimization Summary")

                    summary_data = []
                    for result in geometry_results:
                        initial_structure = st.session_state.structures.get(result['name'])
                        final_structure = result['structure']

                        if initial_structure and final_structure:
                            volume_change = ((
                                                     final_structure.lattice.volume - initial_structure.lattice.volume) / initial_structure.lattice.volume) * 100

                            summary_data.append({
                                'Structure': result['name'],
                                'Convergence Status': result.get('convergence_status', 'Unknown'),
                                'Final Energy (eV)': f"{result['energy']:.6f}",
                                'Volume Change (%)': f"{volume_change:+.2f}",
                                'Formula': final_structure.composition.reduced_formula
                            })

                    if summary_data:
                        df_summary = pd.DataFrame(summary_data)
                        st.dataframe(df_summary, use_container_width=True, hide_index=True)

                else:
                    st.info(
                        "No completed geometry optimizations found. Results will appear here after geometry optimization calculations finish.")

        with results_tab4:
            if len(phonon_results) > 1:
                st.subheader("🎵 Phonon Properties Comparison")

                phonon_comparison_data = []
                for result in phonon_results:
                    phonon_data = result['phonon_results']
                    row = {
                        'Structure': result['name'],
                        'Imaginary Modes': phonon_data['imaginary_modes'],
                        'Min Frequency (meV)': f"{phonon_data['min_frequency']:.3f}",
                        'Max Frequency (meV)': f"{np.max(phonon_data['frequencies']):.3f}",
                    }

                    if phonon_data.get('thermodynamics'):
                        thermo = phonon_data['thermodynamics']
                        row.update({
                            'Zero-point Energy (eV)': f"{thermo['zero_point_energy']:.6f}",
                            'Heat Capacity (eV/K)': f"{thermo['heat_capacity']:.6f}",
                           #'Entropy (eV/K)': f"{thermo['entropy']:.6f}"
                        })

                    phonon_comparison_data.append(row)

                df_phonon_compare = pd.DataFrame(phonon_comparison_data)
                st.dataframe(df_phonon_compare, use_container_width=True, hide_index=True)

                if len(phonon_comparison_data) > 1:
                    col_ph_comp1, col_ph_comp2 = st.columns(2)

                    with col_ph_comp1:
                        structures = [r['name'] for r in phonon_results]
                        imaginary_counts = [r['phonon_results']['imaginary_modes'] for r in phonon_results]

                        fig_img = go.Figure(data=go.Bar(
                            x=structures,
                            y=imaginary_counts,
                            marker_color='red',
                            text=imaginary_counts,
                            textposition='auto'
                        ))

                        fig_img.update_layout(
                            title=dict(text="Imaginary Modes Comparison", font=dict(size=24)),
                            xaxis_title="Structure",
                            yaxis_title="Number of Imaginary Modes",
                            height=750,
                            font=dict(size=20),
                            hoverlabel=dict(
                                bgcolor="white",
                                bordercolor="black",
                                font_size=20,
                                font_family="Arial"
                            ),
                            xaxis=dict(
                                tickangle=45,
                                title_font=dict(size=20),
                                tickfont=dict(size=20)
                            ),
                            yaxis=dict(
                                title_font=dict(size=20),
                                tickfont=dict(size=20)
                            )
                        )

                        st.plotly_chart(fig_img, use_container_width=True)

                    with col_ph_comp2:
                        min_freqs = [r['phonon_results']['min_frequency'] for r in phonon_results]

                        fig_min_freq = go.Figure(data=go.Bar(
                            x=structures,
                            y=min_freqs,
                            marker_color='blue',
                            text=[f"{f:.3f}" for f in min_freqs],
                            textposition='auto'
                        ))

                        fig_min_freq.update_layout(
                            title=dict(text="Minimum Frequency Comparison", font=dict(size=24)),
                            xaxis_title="Structure",
                            yaxis_title="Minimum Frequency (meV)",
                            height=750,
                            font=dict(size=20),
                            hoverlabel=dict(
                                bgcolor="white",
                                bordercolor="black",
                                font_size=20,
                                font_family="Arial"
                            ),
                            xaxis=dict(
                                tickangle=45,
                                title_font=dict(size=20),
                                tickfont=dict(size=20)
                            ),
                            yaxis=dict(
                                title_font=dict(size=20),
                                tickfont=dict(size=20)
                            )
                        )

                        fig_min_freq.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7)

                        st.plotly_chart(fig_min_freq, use_container_width=True)
            else:
                st.info(
                    "No completed phonons calculations found. Results will appear here after the phonons compputations are finish.")

        with results_tab3:
            if len(elastic_results) > 1:
                st.subheader("🔧 Elastic Properties Comparison")

                elastic_comparison_data = []
                for result in elastic_results:
                    elastic_data = result['elastic_results']
                    bulk = elastic_data['bulk_modulus']
                    shear = elastic_data['shear_modulus']

                    row = {
                        'Structure': result['name'],
                        'Bulk Modulus (GPa)': f"{bulk['hill'] if bulk['hill'] is not None else bulk['voigt']:.1f}",
                        'Shear Modulus (GPa)': f"{shear['hill'] if shear['hill'] is not None else shear['voigt']:.1f}",
                        "Young's Modulus (GPa)": f"{elastic_data['youngs_modulus']:.1f}",
                        "Poisson's Ratio": f"{elastic_data['poisson_ratio']:.3f}",
                        'Density (g/cm³)': f"{elastic_data['density']:.3f}",
                        'Debye Temperature (K)': f"{elastic_data['debye_temperature']:.1f}",
                        'Mechanically Stable': "✅" if elastic_data['mechanical_stability'].get('mechanically_stable',
                                                                                               False) else "❌"
                    }

                    elastic_comparison_data.append(row)

                df_elastic_compare = pd.DataFrame(elastic_comparison_data)
                st.dataframe(df_elastic_compare, use_container_width=True, hide_index=True)

                st.subheader("📊 Elastic Constants Comparison")
                all_elements_elastic = set()
                for result in elastic_results:
                    structure_result = next((r for r in st.session_state.results if r['name'] == result['name']), None)
                    if structure_result and 'structure' in structure_result and structure_result['structure']:
                        for site in structure_result['structure']:
                            all_elements_elastic.add(site.specie.symbol)
                all_elements_elastic = sorted(list(all_elements_elastic))
                elastic_constants_data = []
                for result in elastic_results:
                    elastic_data = result['elastic_results']
                    elastic_tensor = np.array(elastic_data['elastic_tensor'])
                    structure_result = next((r for r in st.session_state.results if r['name'] == result['name']), None)

                    row = {
                        'Structure': result['name'],
                        'C11 (GPa)': f"{elastic_tensor[0, 0]:.1f}",
                        'C22 (GPa)': f"{elastic_tensor[1, 1]:.1f}",
                        'C33 (GPa)': f"{elastic_tensor[2, 2]:.1f}",
                        'C44 (GPa)': f"{elastic_tensor[3, 3]:.1f}",
                        'C55 (GPa)': f"{elastic_tensor[4, 4]:.1f}",
                        'C66 (GPa)': f"{elastic_tensor[5, 5]:.1f}",
                    }

                    if structure_result and 'structure' in structure_result and structure_result['structure']:
                        concentrations = get_atomic_concentrations_from_structure(structure_result['structure'])
                        for element in all_elements_elastic:
                            concentration = concentrations.get(element, 0)
                            row[f'{element} (%)'] = f"{concentration:.1f}" if concentration > 0 else "0.0"
                    else:
                        for element in all_elements_elastic:
                            row[f'{element} (%)'] = "N/A"

                    elastic_constants_data.append(row)

                df_elastic_constants = pd.DataFrame(elastic_constants_data)
                st.dataframe(df_elastic_constants, use_container_width=True, hide_index=True)

                if len(elastic_comparison_data) > 1:
                    structures = [r['name'] for r in elastic_results]

                    col_el_comp1, col_el_comp2 = st.columns(2)

                    with col_el_comp1:
                        bulk_values = []
                        shear_values = []

                        for result in elastic_results:
                            elastic_data = result['elastic_results']
                            bulk = elastic_data['bulk_modulus']
                            shear = elastic_data['shear_modulus']
                            bulk_values.append(bulk['hill'] if bulk['hill'] is not None else bulk['voigt'])
                            shear_values.append(shear['hill'] if shear['hill'] is not None else shear['voigt'])

                        fig_moduli_comp = go.Figure()

                        fig_moduli_comp.add_trace(go.Bar(
                            name='Bulk Modulus',
                            x=structures,
                            y=bulk_values,
                            marker_color='steelblue'
                        ))

                        fig_moduli_comp.add_trace(go.Bar(
                            name='Shear Modulus',
                            x=structures,
                            y=shear_values,
                            marker_color='orange'
                        ))

                        fig_moduli_comp.update_layout(
                            title=dict(text="Bulk vs Shear Modulus", font=dict(size=24)),
                            xaxis_title="Structure",
                            yaxis_title="Modulus (GPa)",
                            height=650,
                            barmode='group',
                            font=dict(size=20),
                            hoverlabel=dict(
                                bgcolor="white",
                                bordercolor="black",
                                font_size=20,
                                font_family="Arial"
                            ),
                            xaxis=dict(
                                tickangle=45,
                                title_font=dict(size=20),
                                tickfont=dict(size=20)
                            ),
                            yaxis=dict(
                                title_font=dict(size=20),
                                tickfont=dict(size=20)
                            ),
                            legend=dict(
                                font=dict(size=20)
                            )
                        )

                        st.plotly_chart(fig_moduli_comp, use_container_width=True)

                    with col_el_comp2:
                        density_values = [r['elastic_results']['density'] for r in elastic_results]

                        fig_density = go.Figure(data=go.Bar(
                            x=structures,
                            y=density_values,
                            marker_color='purple',
                            text=[f"{d:.2f}" for d in density_values],
                            textposition='auto'
                        ))

                        fig_density.update_layout(
                            title=dict(text="Density Comparison", font=dict(size=24)),
                            xaxis_title="Structure",
                            yaxis_title="Density (g/cm³)",
                            height=650,
                            font=dict(size=20),
                            hoverlabel=dict(
                                bgcolor="white",
                                bordercolor="black",
                                font_size=20,
                                font_family="Arial"
                            ),
                            xaxis=dict(
                                tickangle=45,
                                title_font=dict(size=20),
                                tickfont=dict(size=20)
                            ),
                            yaxis=dict(
                                title_font=dict(size=20),
                                tickfont=dict(size=20)
                            )
                        )

                        st.plotly_chart(fig_density, use_container_width=True)
    elif st.session_state.calculation_running:
        st.info("🔄 Calculations in progress... Results will appear here as each structure completes.")

        if st.session_state.current_structure_progress:
            progress_data = st.session_state.current_structure_progress
            st.progress(progress_data['progress'], text=f"Overall Progress: {progress_data['text']}")

        if st.session_state.current_optimization_info and st.session_state.current_optimization_info.get(
                'is_optimizing'):
            st.info(f"🔧 Currently optimizing: {st.session_state.current_optimization_info['structure']}")

    else:
        st.info("Results will appear here after calculations complete")

with tab4:
    st.header("Optimization Trajectories")

    if st.session_state.optimization_trajectories:
        st.info(
            "📋 Download XYZ trajectory files for each optimized structure. Files include lattice parameters, energies, and forces for each optimization step.")

        trajectory_summary = []
        for structure_name, trajectory in st.session_state.optimization_trajectories.items():
            if trajectory:
                initial_energy = trajectory[0]['energy']
                final_energy = trajectory[-1]['energy']
                energy_change = final_energy - initial_energy
                max_force_final = trajectory[-1]['max_force']
                n_steps = len(trajectory)

                trajectory_summary.append({
                    'Structure': structure_name,
                    'Steps': n_steps,
                    'Initial Energy (eV)': f"{initial_energy:.6f}",
                    'Final Energy (eV)': f"{final_energy:.6f}",
                    'Energy Change (eV)': f"{energy_change:.6f}",
                    'Final Max Force (eV/Å)': f"{max_force_final:.4f}"
                })

        if trajectory_summary:
            df_trajectories = pd.DataFrame(trajectory_summary)
            st.subheader("Trajectory Summary")
            st.dataframe(df_trajectories, use_container_width=True)

            st.subheader("Download Individual Trajectories")

            for structure_name, trajectory in st.session_state.optimization_trajectories.items():
                if trajectory:
                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        st.write(f"**{structure_name}**")
                        st.write(f"  • {len(trajectory)} optimization steps")
                        st.write(f"  • Energy change: {trajectory[-1]['energy'] - trajectory[0]['energy']:.6f} eV")

                    with col2:
                        xyz_content = create_xyz_content(trajectory, structure_name)
                        filename = f"trajectory_{structure_name.replace('.', '_')}.xyz"

                        st.download_button(
                            label="📥 Download XYZ",
                            data=xyz_content,
                            file_name=filename,
                            mime="text/plain",
                            key=f"download_{structure_name}",
                            help="Extended XYZ format with lattice parameters, energies, and forces"
                        )

                    with col3:
                        if len(trajectory) > 1:
                            steps = [step['step'] for step in trajectory]
                            energies = [step['energy'] for step in trajectory]
                            max_forces = [step['max_force'] for step in trajectory]

                            from plotly.subplots import make_subplots

                            fig_traj = make_subplots(
                                rows=1, cols=2,
                                subplot_titles=('Energy vs Step', 'Max Force vs Step'),
                                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                            )

                            fig_traj.add_trace(
                                go.Scatter(
                                    x=steps,
                                    y=energies,
                                    mode='lines+markers',
                                    name='Energy',
                                    line=dict(width=2, color='blue'),
                                    marker=dict(size=4)
                                ),
                                row=1, col=1
                            )

                            fig_traj.add_trace(
                                go.Scatter(
                                    x=steps,
                                    y=max_forces,
                                    mode='lines+markers',
                                    name='Max Force',
                                    line=dict(width=2, color='red'),
                                    marker=dict(size=4)
                                ),
                                row=1, col=2
                            )

                            fig_traj.update_layout(
                                height=250,
                                margin=dict(l=40, r=20, t=40, b=40),
                                showlegend=False
                            )

                            fig_traj.update_xaxes(title_text="Optimization Step", row=1, col=1)
                            fig_traj.update_xaxes(title_text="Optimization Step", row=1, col=2)
                            fig_traj.update_yaxes(title_text="Energy (eV)", row=1, col=1)
                            fig_traj.update_yaxes(title_text="Max Force (eV/Å)", row=1, col=2)

                            st.plotly_chart(fig_traj, use_container_width=True, key=f"plot_{structure_name}")

            if len(st.session_state.optimization_trajectories) > 1:
                st.subheader("Download All Trajectories")

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for structure_name, trajectory in st.session_state.optimization_trajectories.items():
                        if trajectory:
                            xyz_content = create_xyz_content(trajectory, structure_name)
                            filename = f"trajectory_{structure_name.replace('.', '_')}.xyz"
                            zip_file.writestr(filename, xyz_content)

                st.download_button(
                    label="📦 Download All Trajectories (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="optimization_trajectories.zip",
                    mime="application/zip",
                    help="ZIP file containing all optimization trajectories in XYZ format"
                )

        with st.expander("📖 XYZ File Format Information"):
            st.markdown("""
            **Extended XYZ Format with Lattice Parameters:**

            Each XYZ file contains all optimization steps with:
            - **Line 1**: Number of atoms
            - **Line 2**: Comment line with:
              - Step number
              - Energy (eV)
              - Maximum force (eV/Å)
              - Lattice parameters (a, b, c, α, β, γ)
              - Properties specification
            - **Lines 3+**: Atomic coordinates and forces
              - Format: `Symbol X Y Z Fx Fy Fz`

            **Example:**
            ```
            8
            Step 1 | Energy=-123.456789 eV | Max_Force=0.1234 eV/A | Lattice="5.123456 5.123456 5.123456 90.00 90.00 90.00" | Properties=species:S:1:pos:R:3:forces:R:3
            Ti  0.000000  0.000000  0.000000  0.001234 -0.002345  0.000123
            O   1.234567  1.234567  1.234567 -0.001234  0.002345 -0.000123
            ...
            ```

            This format can be read by visualization software like OVITO, VMD, or ASE for trajectory analysis.
            """)

    else:
        st.info("🔄 Optimization trajectories will appear here after geometry optimization calculations complete.")
        st.markdown("""
        **Features:**
        - Download individual XYZ trajectory files for each structure
        - Files include lattice parameters, energies, and forces for each step
        - Energy vs. step plots for quick visualization
        - Bulk download option for multiple structures
        - Compatible with visualization software (OVITO, VMD, ASE)
        """)

if st.session_state.calculation_running:
    time.sleep(2)
    st.rerun()
