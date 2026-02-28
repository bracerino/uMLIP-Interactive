from ase.constraints import FixAtoms, FixCartesian
import streamlit as st

st.set_page_config(page_title="MLIP-Interactive: Compute properties with universal MLIPs", layout="wide")

import os
import pandas as pd
from datetime import datetime
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
from collections import deque
import streamlit as st
import threading
import time
from ase.constraints import FixAtoms
from ase.io import read, write

from helpers.phonons_help import *
from helpers.generate_python_code import *
from helpers.phase_diagram import *
from helpers.monitor_resources import *
from helpers.mace_cards import *
from helpers.generate_python_code import *

from helpers.nudge_elastic_band import (
    setup_neb_parameters_ui,
    run_neb_calculation,
    create_neb_trajectory_xyz,
    create_neb_plot,
    create_combined_neb_plot,
    export_neb_results,
    display_neb_results
)


from helpers.MD_settings import (
    setup_md_parameters_ui,
    run_md_simulation,
    create_md_trajectory_xyz,
    create_md_analysis_plots,
    export_md_results,
    create_npt_analysis_plots
)

from helpers.ga_optimization_module import (
    run_ga_optimization,
    setup_ga_parameters_ui,
    setup_substitution_ui,
    display_ga_results, display_ga_overview
)

from helpers.initial_settings import (
    setup_geometry_optimization_ui,
    display_optimization_info,
    DEFAULT_GEOMETRY_SETTINGS
)

from helpers.tensile_test import (
    setup_tensile_test_ui,
    run_tensile_test,
    create_stress_strain_plot,
    export_tensile_results
)

from helpers.generate_md_script import generate_md_python_script

import py3Dmol
import streamlit.components.v1 as components
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from ase.io import read, write
from ase import Atoms

try:
    from ase.optimize import (
        BFGS, LBFGS, FIRE,
        BFGSLineSearch, LBFGSLineSearch,
        GoodOldQuasiNewton, MDMin, GPMin,
        CellAwareBFGS
    )
    from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
    from ase.constraints import FixAtoms
    from ase.filters import ExpCellFilter, UnitCellFilter
    from ase.stress import voigt_6_to_full_3x3_stress

    CELL_OPT_AVAILABLE = True
except ImportError:
    CELL_OPT_AVAILABLE = False

MACE_IMPORT_METHOD = None
MACE_AVAILABLE = False
MACE_OFF_AVAILABLE = False

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

try:
    from upet.calculator import UPETCalculator
    UPET_AVAILABLE = True
except ImportError:
    UPET_AVAILABLE = False

try:
    from alignn.ff.ff import AlignnAtomwiseCalculator, default_path
    ALIGNN_AVAILABLE = True
except ImportError:
    ALIGNN_AVAILABLE = False

try:
    from pet_mad.calculator import PETMADCalculator
    PETMAD_AVAILABLE = True
except ImportError:
    PETMAD_AVAILABLE = False

try:
    from deepmd.calculator import DP
    DEEPMD_AVAILABLE = True
except ImportError:
    DEEPMD_AVAILABLE = False

import torch

# Add this after the existing THREAD_COUNT_FILE definition
SETTINGS_FILE = "default_settings.json"

DEFAULT_SETTINGS = {
    'thread_count': 1,
    'selected_model': "MACE-MP-0b3 (medium) - Latest",
    'device': "cpu",
    'dtype': "float64",
    'geometry_optimization': DEFAULT_GEOMETRY_SETTINGS,
'tensile_test': {  # Add this
        'strain_direction': 0,  # x-axis
        'strain_rate': 0.1,
        'max_strain': 10.0,
        'temperature': 300,
        'timestep': 1.0,
        'friction': 0.01,
        'equilibration_steps': 200,
        'sample_interval': 10,
        'relax_between_strain': False,
        'relax_steps': 100,
        'use_npt_transverse': False,
        'bulk_modulus': 110.0
    }
}


def extract_orb_confidence(atoms, calculator, log_queue, structure_name):
    """
    Extract ORB-v3 confidence that was already calculated.
    The confidence is automatically computed when forces are calculated.
    """
    try:
        log_queue.put(f"Extracting ORB-v3 confidence for {structure_name}...")

        # Check if confidence is available in calculator results
        if 'confidence' not in calculator.results:
            log_queue.put(f"  ⚠️ No confidence data - not an ORB-v3 model or forces not yet calculated")
            return {'success': False, 'error': 'Confidence not available'}

        # Get confidence from calculator results (already computed!)
        confidences = calculator.results["confidence"]  # Shape: (num_atoms, 50)

        # Find the predicted bin for each atom (highest probability bin)
        predicted_bin_per_atom = np.argmax(confidences, axis=-1)

        # Convert bins to predicted MAE values (0 to 0.4 Å in 50 bins)
        bin_width = 0.4 / 50  # 0.008 Å per bin
        per_atom_predicted_mae = predicted_bin_per_atom * bin_width

        mean_predicted_mae = float(np.mean(per_atom_predicted_mae))
        max_predicted_mae = float(np.max(per_atom_predicted_mae))


        confidence_score = 1.0 / (1.0 + mean_predicted_mae / 0.1)

        high_uncertainty_threshold = 0.1
        high_uncertainty_atoms = np.where(per_atom_predicted_mae > high_uncertainty_threshold)[0].tolist()

        log_queue.put(f"  Mean predicted MAE: {mean_predicted_mae:.4f} Å")
        log_queue.put(f"  Max predicted MAE: {max_predicted_mae:.4f} Å")
        log_queue.put(f"  Confidence score: {confidence_score:.4f}")
        log_queue.put(f"  High uncertainty atoms: {len(high_uncertainty_atoms)}/{len(atoms)}")

        return {
            'success': True,
            'per_atom_confidence_bins': predicted_bin_per_atom.tolist(),
            'per_atom_predicted_mae': per_atom_predicted_mae.tolist(),
            'confidence_distribution': confidences.tolist(),  # Full distribution
            'mean_predicted_mae': mean_predicted_mae,
            'max_predicted_mae': max_predicted_mae,
            'confidence_score': float(confidence_score),
            'high_uncertainty_atoms': high_uncertainty_atoms,
            'method': 'ORB-v3 built-in confidence head'
        }

    except Exception as e:
        log_queue.put(f"❌ Failed to extract ORB confidence for {structure_name}: {str(e)}")
        return {'success': False, 'error': str(e)}

def load_default_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()


def save_default_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f)
        return True
    except:
        return False


if 'default_settings' not in st.session_state:
    st.session_state.default_settings = load_default_settings()

if 'thread_count' not in st.session_state:
    st.session_state.thread_count = st.session_state.default_settings['thread_count']

os.environ['OMP_NUM_THREADS'] = str(st.session_state.thread_count)
torch.set_num_threads(st.session_state.thread_count)

# os.environ['OMP_NUM_THREADS'] = '8'
# torch.set_num_threads(8)
import json

# Settings files
THREAD_COUNT_FILE = "thread_count.txt"
SETTINGS_FILE = "default_settings.json"

if 'default_settings' not in st.session_state:
    st.session_state.default_settings = load_default_settings()

if 'thread_count' not in st.session_state:
    st.session_state.thread_count = st.session_state.default_settings['thread_count']

os.environ['OMP_NUM_THREADS'] = str(st.session_state.thread_count)
torch.set_num_threads(st.session_state.thread_count)

if 'md_trajectories' not in st.session_state:
    st.session_state.md_trajectories = {}
if 'current_md_info' not in st.session_state:
    st.session_state.current_md_info = {}

if 'thread_count' not in st.session_state:
    st.session_state.thread_count = default_thread_count

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
if 'results_backup_file' not in st.session_state:
    st.session_state.results_backup_file = None

if 'ga_progress_info' not in st.session_state:
    st.session_state.ga_progress_info = {}
if 'ga_structure_timings' not in st.session_state:
    st.session_state.ga_structure_timings = []

if 'last_ga_progress_update' not in st.session_state:
    st.session_state.last_ga_progress_update = 0

if 'current_tensile_info' not in st.session_state:
    st.session_state.current_tensile_info = {}

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

        if filename.endswith(".vasp") or filename.endswith("poscar") or filename.endswith("contcar"):
            atoms = read(file.name, format='vasp')

            selective_dynamics = None
            constraints = []

            with open(file.name, 'r') as f:
                lines = f.readlines()

            selective_line_idx = None
            for i, line in enumerate(lines):
                if line.strip().lower().startswith('selective'):
                    selective_line_idx = i
                    break

            if selective_line_idx is not None:
                coord_start = selective_line_idx + 2

                fixed_indices = []
                cartesian_constraints = []

                for i, line in enumerate(lines[coord_start:]):
                    if line.strip() == '' or line.startswith('#'):
                        break

                    parts = line.split()
                    if len(parts) >= 6:
                        fx, fy, fz = parts[3:6]

                        fix_x = fx.upper() == 'F'
                        fix_y = fy.upper() == 'F'
                        fix_z = fz.upper() == 'F'

                        if fix_x and fix_y and fix_z:
                            fixed_indices.append(i)
                        elif fix_x or fix_y or fix_z:
                            mask = [fix_x, fix_y, fix_z]
                            cartesian_constraints.append((i, mask))

                if fixed_indices:
                    constraints.append(FixAtoms(indices=fixed_indices))

                for atom_idx, mask in cartesian_constraints:
                    constraints.append(FixCartesian(atom_idx, mask))

                if constraints:
                    atoms.set_constraint(constraints)

            from pymatgen.io.ase import AseAtomsAdaptor
            mg_structure = AseAtomsAdaptor.get_structure(atoms)

            if constraints:
                mg_structure.constraints_info = constraints

        else:
            atoms = read(file.name)

            constraints_info = None
            if hasattr(atoms, 'constraints') and atoms.constraints:
                constraints_info = atoms.constraints

            from pymatgen.io.ase import AseAtomsAdaptor
            mg_structure = AseAtomsAdaptor.get_structure(atoms)

            if constraints_info:
                mg_structure.constraints_info = constraints_info

        if os.path.exists(file.name):
            os.remove(file.name)

        return mg_structure

    except Exception as e:
        st.error(f"Failed to parse {file.name}: {e}")
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


def append_to_backup_file(result, backup_file_path):
    try:
        backup_dir = os.path.dirname(backup_file_path)
        os.makedirs(backup_dir, exist_ok=True)

        name = result['name']
        energy = result.get('energy', 'N/A')
        formation_energy = result.get('formation_energy', 'N/A')

        if 'structure' in result and result['structure']:
            structure = result['structure']
            lattice = structure.lattice
            lattice_info = f"a={lattice.a:.4f}, b={lattice.b:.4f}, c={lattice.c:.4f}, α={lattice.alpha:.2f}°, β={lattice.beta:.2f}°, γ={lattice.gamma:.2f}°"
        else:
            lattice_info = "N/A"

        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"{timestamp}\t{name}\t{energy}\t{formation_energy}\t{lattice_info}\n"

        write_header = not os.path.exists(backup_file_path)

        with open(backup_file_path, 'a', encoding='utf-8') as f:
            if write_header:
                header = "Time\tStructure_Name\tTotal_Energy(eV)\tFormation_Energy(eV/atom)\tLattice_Parameters\n"
                f.write(header)
            f.write(line)

    except Exception as e:
        pass


def save_optimized_structure_backup(result, backup_dir):
    try:
        # Only save if this is a geometry optimization with a structure
        if (result.get('calc_type') == 'Geometry Optimization' and
                'structure' in result and result['structure'] and
                result.get('energy') is not None):

            structure = result['structure']
            name = result['name']

            # Create structures subdirectory
            structures_dir = os.path.join(backup_dir, "optimized_structures")
            os.makedirs(structures_dir, exist_ok=True)

            # Generate filename (remove extension and add _optimized)
            base_name = os.path.splitext(name)[0]
            poscar_filename = f"{base_name}_optimized_POSCAR.vasp"
            poscar_path = os.path.join(structures_dir, poscar_filename)

            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write

            new_struct = Structure(structure.lattice, [], [])
            for site in structure:
                new_struct.append(
                    species=site.species,
                    coords=site.frac_coords,
                    coords_are_cartesian=False,
                )

            ase_structure = AseAtomsAdaptor.get_atoms(new_struct)

            write(poscar_path, ase_structure, format="vasp", direct=True, sort=True)

            summary_path = os.path.join(structures_dir, f"{base_name}_optimization_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Optimization Summary for {name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Final Energy: {result['energy']:.6f} eV\n")
                if result.get('formation_energy'):
                    f.write(f"Formation Energy: {result['formation_energy']:.6f} eV/atom\n")
                f.write(f"Convergence Status: {result.get('convergence_status', 'Unknown')}\n")
                f.write(f"Calculation Type: {result['calc_type']}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                lattice = structure.lattice
                f.write("Final Lattice Parameters:\n")
                f.write(f"  a = {lattice.a:.6f} Å\n")
                f.write(f"  b = {lattice.b:.6f} Å\n")
                f.write(f"  c = {lattice.c:.6f} Å\n")
                f.write(f"  α = {lattice.alpha:.3f}°\n")
                f.write(f"  β = {lattice.beta:.3f}°\n")
                f.write(f"  γ = {lattice.gamma:.3f}°\n")
                f.write(f"  Volume = {lattice.volume:.6f} Å³\n\n")

                # Composition
                f.write(f"Composition: {structure.composition.reduced_formula}\n")
                f.write(f"Number of atoms: {len(structure)}\n\n")

                f.write(f"Optimized POSCAR saved as: {poscar_filename}\n")

            return poscar_path

    except Exception as e:
        pass

    return None


def save_ga_best_structure_backup(ga_results, structure_name, backup_dir, run_id):
    try:
        if not ga_results or not ga_results.get('best_structure'):
            return None

        best_structure = ga_results['best_structure']
        best_energy = ga_results['best_energy']

        ga_structures_dir = os.path.join(backup_dir, "ga_optimized_structures")
        os.makedirs(ga_structures_dir, exist_ok=True)

        base_name = os.path.splitext(structure_name)[0]
        poscar_filename = f"{base_name}_ga_run_{run_id + 1}_best_POSCAR.vasp"
        poscar_path = os.path.join(ga_structures_dir, poscar_filename)

        from pymatgen.io.ase import AseAtomsAdaptor
        from ase.io import write

        new_struct = Structure(best_structure.lattice, [], [])
        for site in best_structure:
            new_struct.append(
                species=site.species,
                coords=site.frac_coords,
                coords_are_cartesian=False,
            )

        ase_structure = AseAtomsAdaptor.get_atoms(new_struct)
        write(poscar_path, ase_structure, format="vasp", direct=True, sort=True)

        summary_filename = f"{base_name}_ga_run_{run_id + 1}_summary.txt"
        summary_path = os.path.join(ga_structures_dir, summary_filename)

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"GA Optimization Summary for {structure_name} - Run {run_id + 1}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Best Energy: {best_energy:.6f} eV\n")
            f.write(f"Number of GA Runs: {ga_results.get('num_runs', 'N/A')}\n")
            f.write(f"Population Size: {ga_results.get('ga_params', {}).get('population_size', 'N/A')}\n")
            f.write(f"Max Generations: {ga_results.get('ga_params', {}).get('max_generations', 'N/A')}\n")
            f.write(f"Final Generation: {len(ga_results.get('fitness_history', []))}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            lattice = best_structure.lattice
            f.write("Best Structure Lattice Parameters:\n")
            f.write(f"  a = {lattice.a:.6f} Å\n")
            f.write(f"  b = {lattice.b:.6f} Å\n")
            f.write(f"  c = {lattice.c:.6f} Å\n")
            f.write(f"  α = {lattice.alpha:.3f}°\n")
            f.write(f"  β = {lattice.beta:.3f}°\n")
            f.write(f"  γ = {lattice.gamma:.3f}°\n")
            f.write(f"  Volume = {lattice.volume:.6f} Å³\n\n")

            f.write(f"Composition: {best_structure.composition.reduced_formula}\n")
            f.write(f"Number of atoms: {len(best_structure)}\n\n")

            if 'substitutions' in ga_results:
                f.write("Substitutions Applied:\n")
                for orig_elem, sub_info in ga_results['substitutions'].items():
                    if sub_info['new_element'] == 'VACANCY':
                        f.write(f"  {orig_elem} → VACANCY: {sub_info['n_substitute']} sites\n")
                    else:
                        f.write(f"  {orig_elem} → {sub_info['new_element']}: {sub_info['n_substitute']} sites\n")
                f.write("\n")

            f.write(f"Optimized POSCAR saved as: {poscar_filename}\n")

        return poscar_path

    except Exception as e:
        print(f"Error saving GA structure backup: {str(e)}")
        return None


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

        supercells = phonon.supercells_with_displacements
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

        log_queue.put("  Calculating phonon band structure with enhanced k-point density...")

        try:
            from pymatgen.symmetry.bandstructure import HighSymmKpath
            from ase.dft.kpoints import bandpath

            kpath = HighSymmKpath(pmg_structure)
            path = kpath.kpath["path"]
            kpoints = kpath.kpath["kpoints"]

            ase_cell = atoms.get_cell()

            npoints_per_segment = phonon_params.get('npoints', 151)
            total_npoints = phonon_params.get('total_npoints', 501)

            log_queue.put(f"  Using {npoints_per_segment} points per segment")

            path_kpoints = []
            path_labels = []
            path_connections = []
            cumulative_distance = [0.0]
            current_distance = 0.0

            for segment in path:
                if len(segment) < 2:
                    continue

                segment_points = []
                segment_labels = []

                for i, point_name in enumerate(segment):
                    point_coords = kpoints[point_name]
                    segment_points.append(point_coords)
                    segment_labels.append(point_name)

                for i in range(len(segment_points) - 1):
                    start_point = np.array(segment_points[i])
                    end_point = np.array(segment_points[i + 1])

                    for j in range(npoints_per_segment):
                        t = j / (npoints_per_segment - 1)
                        interpolated_point = start_point + t * (end_point - start_point)
                        path_kpoints.append(interpolated_point.tolist())

                        if len(path_kpoints) > 1:
                            prev_point = np.array(path_kpoints[-2])
                            curr_point = np.array(path_kpoints[-1])
                            prev_cart = prev_point @ ase_cell.reciprocal()
                            curr_cart = curr_point @ ase_cell.reciprocal()
                            current_distance += np.linalg.norm(curr_cart - prev_cart)

                        cumulative_distance.append(current_distance)

                        if j == 0:
                            path_labels.append(segment_labels[i])
                        elif j == npoints_per_segment - 1:
                            path_labels.append(segment_labels[i + 1])
                        else:
                            path_labels.append('')

            bands = []
            current_band = []

            for kpt in path_kpoints:
                current_band.append(kpt)

            if current_band:
                bands.append(current_band)

            log_queue.put(f"  Generated enhanced k-point path with {len(path_kpoints)} points")

            unique_label_positions = []
            unique_labels = []
            seen_labels = set()

            for i, label in enumerate(path_labels):
                if label and label not in seen_labels:
                    unique_label_positions.append(cumulative_distance[i])
                    unique_labels.append(label)
                    seen_labels.add(label)
                elif label and i == len(path_labels) - 1:
                    unique_label_positions.append(cumulative_distance[i])
                    if label not in unique_labels:
                        unique_labels.append(label)

            log_queue.put(f"  High-symmetry path: {' → '.join(unique_labels)}")

        except Exception as path_error:
            log_queue.put(f"  ⚠️ High-symmetry path detection failed: {str(path_error)}")
            log_queue.put("  Using simple Γ-X-M-Γ path with enhanced density")

            npoints_fallback = phonon_params.get('npoints', 151)
            gamma = [0, 0, 0]
            x_point = [0.5, 0, 0]
            m_point = [0.5, 0.5, 0]

            path_kpoints = []
            cumulative_distance = [0.0]
            current_distance = 0.0

            # Γ to X
            for i in range(npoints_fallback):
                t = i / (npoints_fallback - 1)
                kpt = [t * 0.5, 0, 0]
                path_kpoints.append(kpt)
                if i > 0:
                    current_distance += 0.5 / (npoints_fallback - 1)
                cumulative_distance.append(current_distance)

            # X to M
            for i in range(1, npoints_fallback):
                t = i / (npoints_fallback - 1)
                kpt = [0.5, t * 0.5, 0]
                path_kpoints.append(kpt)
                current_distance += 0.5 / (npoints_fallback - 1)
                cumulative_distance.append(current_distance)

            # M to Γ
            for i in range(1, npoints_fallback):
                t = i / (npoints_fallback - 1)
                kpt = [0.5 * (1 - t), 0.5 * (1 - t), 0]
                path_kpoints.append(kpt)
                current_distance += np.sqrt(2) * 0.5 / (npoints_fallback - 1)
                cumulative_distance.append(current_distance)

            bands = [path_kpoints]
            unique_labels = ['Γ', 'X', 'M', 'Γ']
            unique_label_positions = [0, cumulative_distance[npoints_fallback - 1],
                                      cumulative_distance[2 * npoints_fallback - 2],
                                      cumulative_distance[-1]]

        # Run phonon calculation
        phonon.run_band_structure(
            bands,
            is_band_connection=False,
            with_eigenvectors=False,
            is_legacy_plot=False
        )

        log_queue.put("  Processing enhanced band structure data...")
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
        log_queue.put(
            f"  ✅ Enhanced band structure calculated: {frequencies.shape} (valid points: {len(valid_frequencies)})")

        # Process k-points
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
            cumulative_distance = cumulative_distance[:min_len + 1]
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
        frequencies = filter_near_zero_frequencies(frequencies)

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
            'frequencies': frequencies,
            'kpoints': kpoints_band,
            'kpoint_distances': np.array(cumulative_distance[:-1]) if len(cumulative_distance) > len(
                kpoints_band) else np.array(cumulative_distance),
            'kpoint_labels': unique_labels,
            'kpoint_label_positions': unique_label_positions,
            'dos_energies': dos_frequencies,  # meV
            'dos': dos_values,
            'thermodynamics': thermo_props,
            'thermal_properties_dict': thermal_dict,  # Full temperature data
            'supercell_size': tuple([supercell_matrix[i][i] for i in range(3)]),
            'imaginary_modes': int(imaginary_count),
            'min_frequency': float(min_frequency),
            'method': 'Pymatgen+Phonopy',
            'enhanced_kpoints': True
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

# Nequix models
try:
    from nequix.calculator import NequixCalculator

    NEQUIX_AVAILABLE = True
except ImportError:
    NEQUIX_AVAILABLE = False

try:
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator

    CHGNET_AVAILABLE = True
    CHGNET_IMPORT_METHOD = "CHGNet"
except ImportError:
    CHGNET_AVAILABLE = False

# MatterSim
try:
    from mattersim.forcefield import MatterSimCalculator

    MATTERSIM_AVAILABLE = True
except ImportError:
    MATTERSIM_AVAILABLE = False

# Orb-Models
try:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    ORB_AVAILABLE = True
except ImportError:
    ORB_AVAILABLE = False

# try:
#    from orb_models.forcefield import pretrained
#    from orb_models.forcefield.calculator import ORBCalculator
#    print("✅ ORB imports successful")
#    print("Available models:", [attr for attr in dir(pretrained) if not attr.startswith('_')])
# except ImportError as e:
#    print(f"❌ Import failed: {e}")

# Added for torch 2.6 and SevenNet
torch.serialization.add_safe_globals([slice])
try:
    from sevenn.calculator import SevenNetCalculator

    SEVENNET_AVAILABLE = True
except ImportError:
    SEVENNET_AVAILABLE = False

try:
    from mace.calculators import mace_mp, mace_off

    MACE_IMPORT_METHOD = "mace_mp_and_off"
    MACE_AVAILABLE = True
    MACE_OFF_AVAILABLE = True
except ImportError:
    try:
        from mace.calculators import mace_mp

        MACE_IMPORT_METHOD = "mace_mp"
        MACE_AVAILABLE = True
        MACE_OFF_AVAILABLE = False
    except ImportError:
        try:
            from mace.calculators import MACECalculator

            MACE_IMPORT_METHOD = "MACECalculator"
            MACE_AVAILABLE = True
            MACE_OFF_AVAILABLE = False
        except ImportError:
            try:
                import mace
                from mace.calculators import MACECalculator

                MACE_IMPORT_METHOD = "MACECalculator"
                MACE_AVAILABLE = True
                MACE_OFF_AVAILABLE = False
            except ImportError:
                MACE_AVAILABLE = False


class OptimizationLogger:
    def __init__(self, log_queue, structure_name,save_trajectory=True,tetragonal_callback=None):
        self.log_queue = log_queue
        self.structure_name = structure_name
        self.step_count = 0
        self.trajectory = [] if save_trajectory else None
        self.save_trajectory = save_trajectory
        self.previous_energy = None

        self.step_start_time = None
        self.step_times = deque(maxlen=10)  # Keep last 10 step times for averaging
        self.optimization_start_time = time.time()

    def __call__(self, optimizer=None):
        current_time = time.time()

        if self.step_start_time is not None:
            step_duration = current_time - self.step_start_time
            self.step_times.append(step_duration)

        self.step_start_time = current_time

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
                'forces': forces.copy(),
                'timestamp': current_time
            }
            if self.save_trajectory:
                self.trajectory.append(trajectory_step)

            avg_step_time, estimated_remaining_time, total_estimated_time = self._calculate_time_estimates(optimizer)

            elapsed_time = current_time - self.optimization_start_time
            log_message = (f"  Step {self.step_count}: Energy = {energy:.6f} eV, "
                           f"Max Force = {max_force:.4f} eV/Å, ΔE = {energy_change:.2e} eV")

            # if avg_step_time > 0:
            #    log_message += f" | Avg/step: {avg_step_time:.1f}s"
            #    if estimated_remaining_time:
            #        log_message += f" | Est. remaining: {self._format_time(estimated_remaining_time)}"
            #        log_message += f" | Total est.: {self._format_time(total_estimated_time)}"

            # log_message += f" | Elapsed: {self._format_time(elapsed_time)}"

            self.log_queue.put(log_message)

            # Send enhanced progress data
            self.log_queue.put({
                'type': 'opt_step',
                'structure': self.structure_name,
                'step': self.step_count,
                'energy': energy,
                'max_force': max_force,
                'energy_change': energy_change,
                'total_steps': getattr(optimizer, 'max_steps', None),
                'avg_step_time': avg_step_time,
                'estimated_remaining_time': estimated_remaining_time,
                'total_estimated_time': total_estimated_time,
                'elapsed_time': elapsed_time
            })

            if self.save_trajectory:
                self.log_queue.put({
                    'type': 'trajectory_step',
                    'structure': self.structure_name,
                    'step': self.step_count,
                    'trajectory_data': trajectory_step
                })

    def _calculate_time_estimates(self, optimizer):
        if len(self.step_times) < 2:
            return 0, None, None

        avg_step_time = np.mean(list(self.step_times)[1:]) if len(self.step_times) > 1 else self.step_times[0]

        max_steps = getattr(optimizer, 'max_steps', None)
        if max_steps is None:
            return avg_step_time, None, None

        remaining_steps = max_steps - self.step_count
        estimated_remaining_time = remaining_steps * avg_step_time if remaining_steps > 0 else 0

        elapsed_time = time.time() - self.optimization_start_time
        total_estimated_time = elapsed_time + estimated_remaining_time

        return avg_step_time, estimated_remaining_time, total_estimated_time

    def _format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


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

        def safe_angle(v1, v2):
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.degrees(np.arccos(cos_angle))

        alpha = safe_angle(cell[1], cell[2])
        beta = safe_angle(cell[0], cell[2])
        gamma = safe_angle(cell[0], cell[1])

        cell_flat = cell.flatten()
        lattice_str = " ".join([f"{x:.6f}" for x in cell_flat])

        comment = (f'Step={step} Energy={energy:.6f} Max_Force={max_force:.6f} '
                   f'Lattice="{lattice_str}" '
                   f'Properties=species:S:1:pos:R:3:forces:R:3')

        xyz_content += f"{comment}\n"

        for i, (symbol, pos, force) in enumerate(zip(symbols, positions, forces)):
            xyz_content += f"{symbol} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f} {force[0]:12.6f} {force[1]:12.6f} {force[2]:12.6f}\n"

    return xyz_content


MACE_MODELS = {
    "UPET PET-MAD (S) - Materials & Molecules [PBEsol]": "upet:pet-mad-s:1.0.2",
    "UPET PET-OMAD (XS) - Materials & Molecules [PBEsol]": "upet:pet-omad-xs:1.0.0",
    "UPET PET-OMAD (S) - Materials & Molecules [PBEsol]": "upet:pet-omad-s:1.0.0",
    "UPET PET-OMAD (L) - Materials & Molecules [PBEsol]": "upet:pet-omad-l:0.1.0",
    "UPET PET-OAM (L) - Materials Discovery [PBE]": "upet:pet-oam-l:1.0.0",
    "UPET PET-OAM (XL) ⭐ - Materials Discovery [PBE]": "upet:pet-oam-xl:1.0.0",
    "UPET PET-OMat (XS) - Materials [PBE]": "upet:pet-omat-xs:1.0.0",
    "UPET PET-OMat (S) - Materials [PBE]": "upet:pet-omat-s:1.0.0",
    "UPET PET-OMat (M) - Materials [PBE]": "upet:pet-omat-m:1.0.0",
    "UPET PET-OMat (L) - Materials [PBE]": "upet:pet-omat-l:0.1.0",
    "UPET PET-OMat (XL) - Materials [PBE]": "upet:pet-omat-xl:1.0.0",
    "UPET PET-OMATPES (L) - Materials [r2SCAN]": "upet:pet-omatpes-l:0.1.0",
    "UPET PET-SPICE (S) - Molecules [wB97M-D3]": "upet:pet-spice-s:1.0.0",
    "UPET PET-SPICE (L) - Molecules [wB97M-D3]": "upet:pet-spice-l:0.1.0",
    "Custom MACE Model 🔧": "custom",
    "MACE-MP-0b3 (medium) - Latest": "medium-0b3",
    "MACE-MP-0 (small) - Original": "small",
    "MACE-MP-0 (medium) - Original": "medium",
    "MACE-MP-0 (large) - Original": "large",

    "MACE-MP-0b (small) - Improved": "small-0b",
    "MACE-MP-0b (medium) - Improved": "medium-0b",

    "MACE-MP-0b2 (small) - Enhanced": "small-0b2",
    "MACE-MP-0b2 (medium) - Enhanced": "medium-0b2",
    "MACE-MP-0b2 (large) - Enhanced": "large-0b2",

    "MACE-MPA-0 (medium) - Latest": "medium-mpa-0",

    "MACE-OMAT-0 (medium)": "medium-omat-0",

    # ========== MATPES MODELS (need full URLs) ==========
    "MACE-MATPES-PBE-0 (medium) - No +U": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
    "MACE-MATPES-r2SCAN-0 (medium) - r2SCAN": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model",

    # ========== MACE-OFF MODELS (Organic Force Fields) ==========
    "MACE-OFF23 (small) - Organic": "small",
    "MACE-OFF23 (medium) - Organic": "medium",
    "MACE-OFF23 (large) - Organic": "large",

    "MACE-MH-0 (Multi-head Foundation) - Linear": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model",
    "MACE-MH-1 (Multi-head Foundation) - Non-linear ⭐": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",



    # ========== MACE-OFF MODELS (Organic Force Fields) ==========
    "CHGNet-0.3.0 (Latest Universal)": "chgnet-0.3.0",
    "CHGNet-0.2.0 (Legacy Universal)": "chgnet-0.2.0",

    # ========== SEVENNET MODELS ==========
    "SevenNet-0": "7net-0",
    "SevenNet-MF-OMPA (MPA Modal)": "7net-mf-ompa-mpa",
    "SevenNet-MF-OMPA (OMat24 Modal)": "7net-mf-ompa-omat24",
    "SevenNet-OMAT24": "7net-omat",
    "SevenNet-L3I5": "7net-l3i5",

    # ========== MATTERSIM MODELS ==========
    "MatterSim-v1.0.0-1M (Fast Universal)": "mattersim-1m",
    "MatterSim-v1.0.0-5M (Accurate Universal)": "mattersim-5m",
    # ========== ORB MODELS ==========
    "ORB-v3 Conservative OMAT (Recommended)": "orb_v3_conservative_inf_omat",
    "ORB-v3 Conservative OMol (Molecular)": "orb_v3_conservative_omol",
    "ORB-v3 Direct OMAT (Fast)": "orb_v3_direct_inf_omat",
    "ORB-v3 Direct OMol (Molecular Fast)": "orb_v3_direct_omol",
    "ORB-v3 Conservative 20-neighbors OMAT": "orb_v3_conservative_20_omat",
    "ORB-v3 Direct 20-neighbors OMAT": "orb_v3_direct_20_omat",
    "ORB-v2 (Legacy)": "orb_v2",

    # ========== NEQUIX MODELS ==========
    "Nequix-MP-1 (Universal Materials)": "nequix-mp-1",

    # ========== DeepMD-kit MODELS ==========
    #"DeePMD DPA-2 (Small)": "dpa2-small",
    #"DeePMD DPA-2 (Medium)": "dpa2-medium",
    #"DeePMD DPA-2 (Large)": "dpa2-large",
    #"DeePMD DPA-3 (Universal)": "dpa3-universal",

    #"AlignN-FF (JARVIS-DFT)": "alignn-ff-jarvis",
    #"AlignN-FF (Custom)": "alignn-ff-custom",

}


def is_url_model(model_size):
    """Check if model_size is a URL that needs downloading."""
    return isinstance(model_size, str) and (model_size.startswith("http://") or model_size.startswith("https://"))


def is_multihead_model(selected_model):
    """Check if the selected model supports multiple heads."""
    return "Multi-head" in selected_model or "MH-0" in selected_model or "MH-1" in selected_model


def download_mace_foundation_model(model_url, log_queue=None):
    """
    Download MACE foundation model from GitHub releases and cache it locally.

    Args:
        model_url: URL to download the model from
        log_queue: Optional queue for logging messages

    Returns:
        Path to the downloaded model file
    """
    from pathlib import Path
    import urllib.request

    # Extract model filename from URL
    model_filename = model_url.split("/")[-1]

    # Create cache directory in user's home
    cache_dir = Path.home() / ".cache" / "mace_foundation_models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Full path to cached model
    model_path = cache_dir / model_filename

    # Check if already downloaded
    if model_path.exists():
        if log_queue:
            log_queue.put(f"✅ Using cached model: {model_filename}")
            log_queue.put(f"   Location: {model_path}")
        return str(model_path)

    # Download the model
    if log_queue:
        log_queue.put(f"📥 Downloading foundation model: {model_filename}")
        log_queue.put(f"   This may take a few minutes (38-57 MB)...")

    try:
        urllib.request.urlretrieve(model_url, str(model_path))

        if log_queue:
            log_queue.put(f"✅ Model downloaded successfully!")
            log_queue.put(f"   Cached at: {model_path}")

        return str(model_path)

    except Exception as e:
        if log_queue:
            log_queue.put(f"❌ Failed to download model: {str(e)}")
        if model_path.exists():
            model_path.unlink()
        raise Exception(f"Failed to download MACE foundation model: {str(e)}")

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
                  "Ac", "Th", "Pa", "U", "Np", "Pu"],
    "MACE-OFF": ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

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


def check_mace_compatibility(structure, selected_model_key="MACE-MP-0 (medium) - Original"):
    elements = list(set([site.specie.symbol for site in structure]))

    if "OFF" in selected_model_key:
        model_type = "MACE-OFF"
    else:
        model_type = "MACE-MP-0"

    supported_elements = MACE_ELEMENTS[model_type]
    unsupported = [elem for elem in elements if elem not in supported_elements]

    return len(unsupported) == 0, unsupported, elements, model_type


def get_model_type_from_selection(selected_model_name):
    if "OFF" in selected_model_name:
        return "MACE-OFF", "Organic molecules (H, C, N, O, F, P, S, Cl, Br, I)"
    elif "SevenNet" in selected_model_name:
        return "SevenNet", "Universal potential (89 elements)"
    elif "MatterSim" in selected_model_name:
        return "MatterSim", "Universal potential (89 elements, high T/P, specifically made for bulk materials)"
    elif "ORB" in selected_model_name:
        return "ORB", "Universal potential with confidence estimation"
    elif "Nequix" in selected_model_name:
        return "Nequix", "Universal materials potential (foundation model)"
    else:
        return "MACE-MP", "General materials (89 elements)"


def pymatgen_to_ase(structure):
    atoms = Atoms(
        symbols=[str(site.specie) for site in structure],
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True
    )

    wrapped_atoms = wrap_positions_in_cell(atoms)

    if hasattr(structure, 'constraints_info') and structure.constraints_info:
        if isinstance(structure.constraints_info, list):
            wrapped_atoms.set_constraint(structure.constraints_info)
        else:
            wrapped_atoms.set_constraint([structure.constraints_info])

    return wrapped_atoms


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

    elif cell_constraint == "Tetragonal (a=b, optimize a and c)":

        return ExpCellFilter(atoms, scalar_pressure=pressure_eV_A3)

    else:  # "Lattice parameters only (fix angles)"
        if hydrostatic:
            return UnitCellFilter(atoms, scalar_pressure=pressure_eV_A3, hydrostatic_strain=True)
        else:
            mask = [optimize_lattice['a'], optimize_lattice['b'], optimize_lattice['c'], False, False, False]
            return UnitCellFilter(atoms, mask=mask, scalar_pressure=pressure_eV_A3)


def check_selective_dynamics(atoms):
    if not hasattr(atoms, 'constraints') or not atoms.constraints:
        return False, 0, len(atoms)

    total_atoms = len(atoms)
    constrained_atoms = set()

    for constraint in atoms.constraints:
        try:
            if isinstance(constraint, FixAtoms):
                fixed_indices = constraint.get_indices()
                constrained_atoms.update(fixed_indices)
            else:
                constrained_atoms.add(0)
        except Exception:
            continue

    has_constraints = len(constrained_atoms) > 0
    total_constrained = len(constrained_atoms)


    return has_constraints, total_constrained, total_atoms


def setup_optimization_constraints(atoms, optimization_params):
    opt_type = optimization_params.get('optimization_type', 'Both atoms and cell')

    has_constraints, total_constrained, total_atoms = check_selective_dynamics(atoms)

    # Preserve existing constraints
    existing_constraints = atoms.constraints if hasattr(atoms, 'constraints') and atoms.constraints else []

    if opt_type == "Atoms only (fixed cell)":
        # Keep existing constraints (selective dynamics)
        return atoms, None, None

    elif opt_type == "Cell only (fixed atoms)":
        # For cell-only optimization, we want to keep atoms fixed
        # If there are already selective dynamics, they're redundant (all atoms will be fixed anyway)
        # But we should preserve them for consistency
        atoms.set_constraint(FixAtoms(indices=list(range(len(atoms)))))

        cell_constraint = optimization_params.get('cell_constraint', 'Lattice parameters only (fix angles)')

        # Handle tetragonal for cell-only optimization
        if cell_constraint == "Tetragonal (a=b, optimize a and c)":
            mask = [True, True, True, False, False, False]  # a, b, c can change; angles fixed
            pressure = optimization_params.get('pressure', 0.0)
            pressure_eV_A3 = pressure * 0.00624150913

            cell_filter = UnitCellFilter(atoms, mask=mask, scalar_pressure=pressure_eV_A3)

            # Create callback to enforce a=b after each step
            def enforce_tetragonal():
                cell = atoms.get_cell()
                cellpar = cell.cellpar()

                # Average a and b
                avg_ab = (cellpar[0] + cellpar[1]) / 2.0
                old_a = cellpar[0]
                old_b = cellpar[1]

                cellpar[0] = avg_ab
                cellpar[1] = avg_ab
                # cellpar[3:] = 90.0  # Ensure angles stay at 90

                atoms.set_cell(cellpar, scale_atoms=True)

                return old_a, old_b, avg_ab

            return cell_filter, "cell_only", enforce_tetragonal
        else:
            cell_filter = create_cell_filter(atoms, optimization_params)
            return cell_filter, "cell_only", None

    else:  # Both atoms and cell
        # Keep existing selective dynamics constraints
        # They will restrict which atoms can move during optimization

        cell_constraint = optimization_params.get('cell_constraint', 'Lattice parameters only (fix angles)')

        if cell_constraint == "Tetragonal (a=b, optimize a and c)":
            mask = [True, True, True, False, False, False]  # a, b, c can change; angles fixed

            pressure = optimization_params.get('pressure', 0.0)
            pressure_eV_A3 = pressure * 0.00624150913

            cell_filter = UnitCellFilter(atoms, mask=mask, scalar_pressure=pressure_eV_A3)

            # Create callback to enforce a=b after each step
            def enforce_tetragonal():
                cell = atoms.get_cell()
                cellpar = cell.cellpar()

                # Average a and b
                avg_ab = (cellpar[0] + cellpar[1]) / 2.0
                old_a = cellpar[0]
                old_b = cellpar[1]

                cellpar[0] = avg_ab
                cellpar[1] = avg_ab
                # cellpar[3:] = 90.0  # Ensure angles stay at 90

                atoms.set_cell(cellpar, scale_atoms=True)

                return old_a, old_b, avg_ab

            return cell_filter, "both", enforce_tetragonal

        cell_filter = create_cell_filter(atoms, optimization_params)
        return cell_filter, "both", None


class CellOptimizationLogger:
    def __init__(self, log_queue, structure_name, opt_mode="both",save_trajectory=True, tetragonal_callback=None):
        self.log_queue = log_queue
        self.structure_name = structure_name
        self.step_count = 0
        self.trajectory = [] if save_trajectory else None
        self.save_trajectory = save_trajectory
        self.previous_energy = None
        self.opt_mode = opt_mode
        self.tetragonal_callback = tetragonal_callback

        self.step_start_time = None
        self.step_times = deque(maxlen=10)
        self.optimization_start_time = time.time()

    def __call__(self, optimizer=None):
        current_time = time.time()

        if self.step_start_time is not None:
            step_duration = current_time - self.step_start_time
            self.step_times.append(step_duration)

        self.step_start_time = current_time

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
                    'stress': stress.copy() if stress is not None else None,
                    'timestamp': current_time
                }
                if self.save_trajectory:
                    self.trajectory.append(trajectory_step)

                avg_step_time, estimated_remaining_time, total_estimated_time = self._calculate_time_estimates(
                    optimizer)

                elapsed_time = current_time - self.optimization_start_time

                if self.opt_mode == "cell_only":
                    log_message = (f"  Step {self.step_count}: Energy = {energy:.6f} eV, "
                                   f"Max Stress = {max_stress:.4f} GPa, ΔE = {energy_change:.2e} eV")
                elif self.opt_mode == "both":
                    log_message = (f"  Step {self.step_count}: Energy = {energy:.6f} eV, "
                                   f"Max Force = {max_force:.4f} eV/Å, Max Stress = {max_stress:.4f} GPa, "
                                   f"ΔE = {energy_change:.2e} eV")
                else:
                    log_message = (f"  Step {self.step_count}: Energy = {energy:.6f} eV, "
                                   f"Max Force = {max_force:.4f} eV/Å, ΔE = {energy_change:.2e} eV")

                # if avg_step_time > 0:
                #    log_message += f" | Avg/step: {avg_step_time:.1f}s"
                #    if estimated_remaining_time:
                #        log_message += f" | Est. remaining: {self._format_time(estimated_remaining_time)}"
                #        log_message += f" | Total est.: {self._format_time(total_estimated_time)}"

                # log_message += f" | Elapsed: {self._format_time(elapsed_time)}"

                self.log_queue.put(log_message)

                self.log_queue.put({
                    'type': 'opt_step',
                    'structure': self.structure_name,
                    'step': self.step_count,
                    'energy': energy,
                    'max_force': max_force,
                    'max_stress': max_stress,
                    'energy_change': energy_change,
                    'total_steps': getattr(optimizer, 'max_steps', None),
                    'avg_step_time': avg_step_time,
                    'estimated_remaining_time': estimated_remaining_time,
                    'total_estimated_time': total_estimated_time,
                    'elapsed_time': elapsed_time
                })

                if self.save_trajectory:
                    self.log_queue.put({
                        'type': 'trajectory_step',
                        'structure': self.structure_name,
                        'step': self.step_count,
                        'trajectory_data': trajectory_step
                    })
                if self.tetragonal_callback is not None:
                    old_a, old_b, new_ab = self.tetragonal_callback()
                    if abs(old_a - old_b) > 1e-6:  # Only log if there was a change
                        self.log_queue.put(
                            f"  🔷 Tetragonal constraint enforced: a={old_a:.6f} Å, b={old_b:.6f} Å → a=b={new_ab:.6f} Å")
            except Exception as e:
                self.log_queue.put(f"  Error in optimization step {self.step_count}: {str(e)}")

    def _calculate_time_estimates(self, optimizer):
        """Calculate time estimates based on recent step times"""
        if len(self.step_times) < 2:
            return 0, None, None

        # Average time per step (excluding first step which is usually slower)
        avg_step_time = np.mean(list(self.step_times)[1:]) if len(self.step_times) > 1 else self.step_times[0]

        # Get maximum steps from optimizer
        max_steps = getattr(optimizer, 'max_steps', None)
        if max_steps is None:
            return avg_step_time, None, None

        remaining_steps = max_steps - self.step_count
        estimated_remaining_time = remaining_steps * avg_step_time if remaining_steps > 0 else 0

        elapsed_time = time.time() - self.optimization_start_time
        total_estimated_time = elapsed_time + estimated_remaining_time

        return avg_step_time, estimated_remaining_time, total_estimated_time

    def _format_time(self, seconds):
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


def run_mace_calculation(structure_data, calc_type, model_size, device, optimization_params, phonon_params,
                         elastic_params, calc_formation_energy, log_queue, stop_event, substitutions=None,
                         ga_params=None,  neb_initial=None, neb_finals=None,mace_head=None, mace_dispersion=False, mace_dispersion_xc="pbe"):
    import time
    try:
        total_start_time = time.time()
        log_queue.put({
            'type': 'total_start_time',
            'start_time': total_start_time
        })
        if calc_type == "NEB Calculation":
            log_queue.put("NEB calculation mode - using separate structure handling")
        elif not structure_data:
            log_queue.put("❌ No structures provided")
            log_queue.put("CALCULATION_FINISHED")
            return

        is_chgnet = model_size.startswith("chgnet")
        is_sevennet = selected_model.startswith("SevenNet")
        is_mattersim = selected_model.startswith("MatterSim")
        is_orb = selected_model.startswith("ORB")
        is_nequix = selected_model.startswith("Nequix")
        is_deepmd = selected_model.startswith("DeePMD")
        is_alignn = selected_model.startswith("AlignN")

        #PET-MAD
        is_upet = model_size.startswith("upet:")

        if is_upet:
            log_queue.put("Setting up UPET calculator...")
            log_queue.put(f"Selected model: {selected_model}")
            log_queue.put(f"Device: {device}")

            if not UPET_AVAILABLE:
                log_queue.put("❌ UPET not available! Please install: pip install upet")
                log_queue.put("CALCULATION_FINISHED")
                return

            try:
                _, upet_model_name, upet_version = model_size.split(":")

                log_queue.put(f"  Model: {upet_model_name}, Version: {upet_version}")

                calculator = UPETCalculator(
                    model=upet_model_name,
                    version=upet_version,
                    device=device,
                )
                log_queue.put(f"✅ UPET {upet_model_name} v{upet_version} initialized successfully on {device}")

            except Exception as e:
                log_queue.put(f"❌ UPET initialization failed on {device}: {str(e)}")
                if device == "cuda":
                    log_queue.put("⚠️ GPU initialization failed, falling back to CPU...")
                    try:
                        calculator = UPETCalculator(
                            model=upet_model_name,
                            version=upet_version,
                            device="cpu",
                        )
                        log_queue.put("✅ UPET initialized successfully on CPU (fallback)")
                    except Exception as cpu_error:
                        log_queue.put(f"❌ CPU fallback also failed: {str(cpu_error)}")
                        log_queue.put("CALCULATION_FINISHED")
                        return
                else:
                    log_queue.put("CALCULATION_FINISHED")
                    return
        elif is_nequix:
            # Nequix setup
            log_queue.put("Setting up Nequix calculator...")
            log_queue.put(f"Selected model: {selected_model}")
            log_queue.put(f"Device: {device}")

            try:
                calculator = NequixCalculator(model_size)
                log_queue.put(f"✅ Nequix {model_size} initialized successfully")

            except Exception as e:
                log_queue.put(f"❌ Nequix initialization failed: {str(e)}")
                return
        elif is_deepmd:
            log_queue.put("Setting up DeePMD calculator...")
            try:
                model_path = model_size  #
                calculator = DP(model=model_path)
                log_queue.put(f"✅ DeePMD {model_size} initialized successfully")
            except Exception as e:
                log_queue.put(f"❌ DeePMD initialization failed: {str(e)}")
                return
        elif is_alignn:
            log_queue.put("Setting up AlignN calculator...")
            try:
                if model_size == "alignn-ff-jarvis":
                    # Use pretrained JARVIS-DFT model
                    calculator = AlignnAtomwiseCalculator(path=default_path())
                else:
                    # Custom model path
                    calculator = AlignnAtomwiseCalculator(path=model_size)

                log_queue.put(f"✅ AlignN {model_size} initialized successfully")
            except Exception as e:
                log_queue.put(f"❌ AlignN initialization failed: {str(e)}")
                return
        elif is_orb:
            # ORB setup
            log_queue.put("Setting up ORB calculator...")
            log_queue.put(f"Selected model: {selected_model}")
            log_queue.put(f"Model function: {model_size}")
            log_queue.put(f"Device: {device}")

            try:
                # Convert dtype to ORB precision format
                if dtype == "float32":
                    precision = "float32-high"
                else:
                    precision = "float32-highest"  # Higher precision option

                log_queue.put(f"Precision: {precision}")

                model_function = getattr(pretrained, model_size)
                orbff = model_function(
                    device=device,
                    precision=precision
                )
                calculator = ORBCalculator(orbff, device=device)
                log_queue.put(f"✅ ORB {model_size} initialized successfully on {device}")

            except Exception as e:
                log_queue.put(f"❌ ORB initialization failed on {device}: {str(e)}")
                if device == "cuda":
                    log_queue.put("⚠️ GPU initialization failed, falling back to CPU...")
                    try:
                        model_function = getattr(pretrained, model_size)
                        orbff = model_function(
                            device="cpu",
                            precision=precision
                        )
                        calculator = ORBCalculator(orbff, device="cpu")
                        log_queue.put("✅ ORB initialized successfully on CPU (fallback)")
                    except Exception as cpu_error:
                        log_queue.put(f"❌ CPU fallback also failed: {str(cpu_error)}")
                        return
                else:
                    return
        elif is_mattersim:
            # MatterSim setup
            log_queue.put("Setting up MatterSim calculator...")
            log_queue.put(f"Selected model: {selected_model}")
            log_queue.put(f"Device: {device}")

            try:
                if model_size == "mattersim-1m":
                    model_path = "MatterSim-v1.0.0-1M.pth"
                elif model_size == "mattersim-5m":
                    model_path = "MatterSim-v1.0.0-5M.pth"
                else:
                    model_path = model_size

                log_queue.put(f"Model path: {model_path}")

                calculator = MatterSimCalculator(
                    model_path=model_path,
                    device=device
                )
                log_queue.put(f"✅ MatterSim {model_path} initialized successfully on {device}")

            except Exception as e:
                log_queue.put(f"❌ MatterSim initialization failed on {device}: {str(e)}")
                if device == "cuda":
                    log_queue.put("⚠️ GPU initialization failed, falling back to CPU...")
                    try:
                        calculator = MatterSimCalculator(
                            model_path=model_path,
                            device="cpu"
                        )
                        log_queue.put("✅ MatterSim initialized successfully on CPU (fallback)")
                    except Exception as cpu_error:
                        log_queue.put(f"❌ CPU fallback also failed: {str(cpu_error)}")
                        return
                else:
                    return
        elif is_chgnet:
            # CHGNet setup
            log_queue.put("Setting up CHGNet calculator...")
            chgnet_version = model_size.split("-")[1]
            log_queue.put(f"CHGNet version: {chgnet_version}")
            log_queue.put(f"Device: {device}")
            log_queue.put("Note: CHGNet requires float32 precision")

            try:
                original_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)

                chgnet = CHGNet.load(model_name=chgnet_version, use_device=device, verbose=False)
                calculator = CHGNetCalculator(model=chgnet, use_device=device)

                torch.set_default_dtype(original_dtype)

                log_queue.put(f"✅ CHGNet {chgnet_version} initialized successfully on {device}")
            except Exception as e:
                log_queue.put(f"❌ CHGNet initialization failed on {device}: {str(e)}")
                if device == "cuda":
                    log_queue.put("⚠️ GPU initialization failed, falling back to CPU...")
                    try:
                        chgnet = CHGNet.load(model_name=chgnet_version, use_device="cpu", verbose=False)
                        calculator = CHGNetCalculator(model=chgnet, use_device="cpu")
                        log_queue.put("✅ CHGNet initialized successfully on CPU (fallback)")
                    except Exception as cpu_error:
                        log_queue.put(f"❌ CPU fallback also failed: {str(cpu_error)}")
                        return
                else:
                    return

        elif is_sevennet:
            log_queue.put("Setting up SevenNet calculator...")
            log_queue.put(f"Selected model: {selected_model}")
            log_queue.put(f"Device: {device}")
            log_queue.put("Note: SevenNet requires float32 precision")

            try:
                original_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)
                print(model_size)
                if model_size == "7net-mf-ompa-mpa":
                    calculator = SevenNetCalculator(model='7net-mf-ompa', modal='mpa', device=device)
                    log_queue.put("✅ SevenNet 7net-mf-ompa (MPA modal) initialized successfully")
                elif model_size == "7net-mf-ompa-omat24":
                    print('here')
                    calculator = SevenNetCalculator(model='SevenNet-mf-ompa', modal='omat24', device=device)
                    log_queue.put("✅ SevenNet 7net-mf-ompa (OMat24 modal) initialized successfully")
                else:
                    calculator = SevenNetCalculator(model=model_size, device=device)
                    log_queue.put(f"✅ SevenNet {model_size} initialized successfully on {device}")

                torch.set_default_dtype(original_dtype)

            except Exception as e:
                log_queue.put(f"❌ SevenNet initialization failed on {device}: {str(e)}")
                if device == "cuda":
                    log_queue.put("⚠️ GPU initialization failed, falling back to CPU...")
                    try:
                        if model_size == "7net-mf-ompa-mpa":
                            calculator = SevenNetCalculator(model='7net-mf-ompa', modal='mpa', device="cpu")
                        elif model_size == "7net-mf-ompa-omat24":
                            calculator = SevenNetCalculator(model='7net-mf-ompa', modal='omat24', device="cpu")
                        else:
                            calculator = SevenNetCalculator(model=model_size, device="cpu")
                        log_queue.put("✅ SevenNet initialized successfully on CPU (fallback)")
                    except Exception as cpu_error:
                        log_queue.put(f"❌ CPU fallback also failed: {str(cpu_error)}")
                        return
                else:
                    return

        else:
            log_queue.put("Setting up MACE calculator...")
            log_queue.put(f"Using import method: {MACE_IMPORT_METHOD}")
            log_queue.put(f"Model identifier: {model_size}")
            log_queue.put(f"Device: {device}")

            # Get MACE configuration from session state
            #mace_config = st.session_state.get('mace_config', {})
            ##mace_head = mace_config.get('head')
            #mace_dispersion = mace_config.get('dispersion', False)
            #mace_dispersion_xc = mace_config.get('dispersion_xc', 'pbe')

            # Log configuration
            if mace_head:
                log_queue.put(f"📊 Prediction head: {mace_head}")
            if mace_dispersion:
                log_queue.put(f"🔬 D3 dispersion: Enabled ({mace_dispersion_xc})")

            calculator = None

            is_mace_off = "OFF" in selected_model

            # Check if model needs to be downloaded from URL
            # Check if model needs to be downloaded from URL
            if is_url_model(model_size):
                log_queue.put(f"Model requires download from URL")

                try:
                    # Download and cache the model
                    local_model_path = download_mace_foundation_model(model_size, log_queue)
                    log_queue.put(f"Initializing calculator from: {local_model_path}")

                    # Check if this is a multi-head model
                    is_mh = is_multihead_model(selected_model)

                    # For multi-head models, head is required
                    if is_mh and not mace_head:
                        log_queue.put("❌ Multi-head model requires head selection!")
                        log_queue.put(
                            "Available heads: omat_pbe, matpes_r2scan, omol, mp_pbe_refit_add, spice_wB97M, oc20_usemppbe")
                        return

                    # Try to load with mace_mp first (supports head and dispersion)
                    if MACE_IMPORT_METHOD == "mace_mp_and_off" or MACE_IMPORT_METHOD == "mace_mp":
                        try:
                            # Build calculator arguments
                            calc_kwargs = {
                                'model': local_model_path,
                                'device': device,
                                'default_dtype': dtype,
                            }

                            # Add head for multi-head models
                            if mace_head:
                                calc_kwargs['head'] = mace_head
                                log_queue.put(f"Using head: {mace_head}")

                            # Add dispersion if enabled
                            if mace_dispersion:
                                calc_kwargs['dispersion'] = True
                                calc_kwargs['dispersion_xc'] = mace_dispersion_xc
                                log_queue.put(f"Using D3 dispersion: {mace_dispersion_xc}")

                            calculator = mace_mp(**calc_kwargs)
                            log_queue.put(f"✅ MACE calculator initialized from downloaded model on {device}")

                        except Exception as e:
                            log_queue.put(f"❌ mace_mp failed: {str(e)}")
                            if device == "cuda":
                                log_queue.put("⚠️ Trying CPU fallback...")
                                calc_kwargs['device'] = "cpu"
                                calculator = mace_mp(**calc_kwargs)
                                log_queue.put("✅ Calculator initialized on CPU (fallback)")
                            else:
                                raise
                    else:
                        log_queue.put("❌ URL-based models require mace_mp import method")
                        return

                except Exception as e:
                    log_queue.put(f"❌ Failed to initialize calculator from URL: {str(e)}")
                    return

            elif is_mace_off and not MACE_OFF_AVAILABLE:
                log_queue.put(
                    "❌ MACE-OFF models requested but not available. Please update your MACE installation.")
                return

            elif MACE_IMPORT_METHOD == "mace_mp_and_off":
                try:
                    if is_mace_off:
                        log_queue.put(
                            f"Initializing MACE-OFF calculator on {device}...")
                        calculator = mace_off(
                            model=model_size, default_dtype=dtype, device=device)
                        log_queue.put(
                            f"✅ MACE-OFF calculator initialized successfully on {device}")
                    else:
                        log_queue.put(
                            f"Initializing MACE-MP calculator on {device}...")
                        calculator = mace_mp(
                            model=model_size,
                            dispersion=mace_dispersion,
                            dispersion_xc=mace_dispersion_xc if mace_dispersion else None,
                            default_dtype=dtype,
                            device=device
                        )
                        log_queue.put(
                            f"✅ MACE-MP calculator initialized successfully on {device}")
                except Exception as e:
                    log_queue.put(
                        f"❌ Calculator initialization failed on {device}: {str(e)}")
                    if device == "cuda":
                        log_queue.put(
                            "⚠️ GPU initialization failed, falling back to CPU...")
                        try:
                            if is_mace_off:
                                calculator = mace_off(
                                    model=model_size, default_dtype=dtype, device="cpu")
                            else:
                                calculator = mace_mp(
                                    model=model_size,
                                    dispersion=mace_dispersion,
                                    dispersion_xc=mace_dispersion_xc if mace_dispersion else None,
                                    default_dtype=dtype,
                                    device="cpu"
                                )
                            log_queue.put(
                                "✅ Calculator initialized successfully on CPU (fallback)")
                        except Exception as cpu_error:
                            log_queue.put(
                                f"❌ CPU fallback also failed: {str(cpu_error)}")
                            return
                    else:
                        return

            elif MACE_IMPORT_METHOD == "mace_mp":
                if is_mace_off:
                    log_queue.put(
                        "❌ MACE-OFF models requested but only MACE-MP available. Please update your MACE installation.")
                    return
                try:
                    log_queue.put(
                        f"Initializing mace_mp calculator on {device}...")
                    calculator = mace_mp(
                        model=model_size,
                        dispersion=mace_dispersion,
                        dispersion_xc=mace_dispersion_xc if mace_dispersion else None,
                        default_dtype=dtype,
                        device=device
                    )
                    log_queue.put(
                        f"✅ mace_mp calculator initialized successfully on {device}")
                except Exception as e:
                    log_queue.put(
                        f"❌ mace_mp initialization failed on {device}: {str(e)}")
                    if device == "cuda":
                        log_queue.put(
                            "⚠️ GPU initialization failed, falling back to CPU...")
                        try:
                            calculator = mace_mp(
                                model=model_size,
                                dispersion=mace_dispersion,
                                dispersion_xc=mace_dispersion_xc if mace_dispersion else None,
                                default_dtype=dtype,
                                device="cpu"
                            )
                            log_queue.put(
                                "✅ mace_mp calculator initialized successfully on CPU (fallback)")
                        except Exception as cpu_error:
                            log_queue.put(
                                f"❌ CPU fallback also failed: {str(cpu_error)}")
                            return
                    else:
                        return

            elif MACE_IMPORT_METHOD == "MACECalculator":
                if is_mace_off:
                    log_queue.put(
                        "❌ MACE-OFF models not supported with MACECalculator import method.")
                    return
                log_queue.put(
                    "Warning: Using MACECalculator - you may need to provide model paths manually")
                try:
                    calculator = MACECalculator(device=device)
                    log_queue.put(f"✅ MACECalculator initialized on {device}")
                except Exception as e:
                    log_queue.put(
                        f"❌ MACECalculator initialization failed on {device}: {str(e)}")
                    if device == "cuda":
                        log_queue.put(
                            "⚠️ GPU initialization failed, falling back to CPU...")
                        try:
                            calculator = MACECalculator(device="cpu")
                            log_queue.put(
                                "✅ MACECalculator initialized on CPU (fallback)")
                        except Exception as cpu_error:
                            log_queue.put(
                                f"❌ CPU fallback also failed: {str(cpu_error)}")
                            return
                    else:
                        return
            else:
                log_queue.put(
                    "❌ MACE not available - please install with: pip install mace-torch")
                return

        if calculator is None:
            log_queue.put("❌ Failed to create calculator")
            return

        log_queue.put("Calculator setup complete, starting structure calculations...")

        reference_energies = {}
        if calc_formation_energy and calc_type != "NEB Calculation":
            all_elements = set()
            for structure in structure_data.values():
                for site in structure:
                    all_elements.add(site.specie.symbol)

            reference_energies = calculate_atomic_reference_energies(all_elements, calculator, log_queue)
            log_queue.put(f"✅ Reference energies calculated for: {', '.join(all_elements)}")

        if calc_type == "NEB Calculation":
            log_queue.put(f"DEBUG: Using passed parameters - initial: {neb_initial is not None}, finals: {len(neb_finals) if neb_finals else 0}")

            if not neb_initial or not neb_finals:
                log_queue.put("❌ No NEB structures configured")
                log_queue.put("CALCULATION_FINISHED")
                return
            else:
                log_queue.put(f"Starting NEB calculations: 1 initial → {len(neb_finals)} final states")

                for i, (final_name, final_structure) in enumerate(neb_finals.items()):
                    if stop_event.is_set():
                        log_queue.put("Calculation stopped by user")
                        break

                    structure_start_time = time.time()
                    neb_structure_name = f"Initial_to_{final_name}"

                    log_queue.put({
                        'type': 'structure_start_time',
                        'structure': neb_structure_name,
                        'start_time': structure_start_time
                    })

                    log_queue.put(f"Processing NEB {i + 1}/{len(neb_finals)}: Initial → {final_name}")
                    log_queue.put(
                        {'type': 'progress', 'current': i, 'total': len(neb_finals), 'name': neb_structure_name})

                    try:
                        neb_results = run_neb_calculation(
                            neb_initial,
                            final_structure,
                            calculator,
                            neb_params,
                            log_queue,
                            stop_event,
                            neb_structure_name
                        )

                        if neb_results['success']:
                            energy = neb_results['energies'][0]
                            log_queue.put(f"✅ NEB completed: {neb_structure_name}")
                        else:
                            log_queue.put(f"❌ NEB failed: {neb_structure_name}")
                            energy = None

                        log_queue.put({
                            'type': 'result',
                            'name': neb_structure_name,
                            'energy': energy,
                            'formation_energy': None,
                            'structure': neb_initial,
                            'calc_type': calc_type,
                            'convergence_status': None,
                            'phonon_results': None,
                            'elastic_results': None,
                            'ga_results': None,
                            'md_results': None,
                            'tensile_results': None,
                            'neb_results': neb_results,
                            'orb_confidence': None
                        })

                        structure_end_time = time.time()
                        structure_duration = structure_end_time - structure_start_time

                        log_queue.put({
                            'type': 'structure_end_time',
                            'structure': neb_structure_name,
                            'end_time': structure_end_time,
                            'duration': structure_duration,
                            'calc_type': calc_type
                        })

                    except Exception as e:
                        log_queue.put(f"❌ Error in NEB calculation {neb_structure_name}: {str(e)}")
                        import traceback
                        log_queue.put(f"Traceback: {traceback.format_exc()}")

                        structure_end_time = time.time()
                        structure_duration = structure_end_time - structure_start_time

                        log_queue.put({
                            'type': 'structure_end_time',
                            'structure': neb_structure_name,
                            'end_time': structure_end_time,
                            'duration': structure_duration,
                            'calc_type': calc_type,
                            'failed': True
                        })

                        log_queue.put({
                            'type': 'result',
                            'name': neb_structure_name,
                            'energy': None,
                            'structure': neb_initial,
                            'calc_type': calc_type,
                            'error': str(e)
                        })
        else:
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
                    orb_confidence = None
                    if is_orb:
                        orb_confidence = extract_orb_confidence(atoms, calculator, log_queue, name)
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

                        atoms = pymatgen_to_ase(structure)
                        atoms.calc = calculator

                        has_constraints, fixed_atoms, total_atoms = check_selective_dynamics(atoms)

                        if has_constraints:
                            log_queue.put(
                                f"📌 Selective dynamics detected: {fixed_atoms}/{total_atoms} atoms are constrained")
                            log_queue.put(
                                f"  Constrained atoms will respect their original selective dynamics during optimization")
                        else:
                            log_queue.put(
                                f"🔄 No selective dynamics found - all {total_atoms} atoms will be optimized")

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
                            save_traj = optimization_params.get('save_trajectory', True)
                            optimization_object, opt_mode, tetragonal_callback = setup_optimization_constraints(atoms,
                                                                                                                optimization_params)


                            # Check if this is tetragonal mode
                            is_tetragonal = (tetragonal_callback is not None)

                            if opt_mode:
                                logger = CellOptimizationLogger(log_queue, name, opt_mode, save_trajectory=save_traj,
                                                                tetragonal_callback=tetragonal_callback)
                            else:
                                logger = OptimizationLogger(log_queue, name, save_trajectory=save_traj)

                            opt_name = optimization_params['optimizer']

                            if opt_name == "LBFGS":
                                optimizer = LBFGS(optimization_object, logfile=None)
                            elif opt_name == "FIRE":
                                optimizer = FIRE(optimization_object, logfile=None)
                            elif opt_name == "BFGSLineSearch (QuasiNewton)":
                                optimizer = BFGSLineSearch(optimization_object, logfile=None)
                            elif opt_name == "LBFGSLineSearch":
                                optimizer = LBFGSLineSearch(optimization_object, logfile=None)
                            elif opt_name == "GoodOldQuasiNewton":
                                optimizer = GoodOldQuasiNewton(optimization_object, logfile=None)
                            elif opt_name == "MDMin":
                                optimizer = MDMin(optimization_object, logfile=None)
                            elif opt_name == "GPMin":
                                if len(atoms) > 100:
                                    log_queue.put("⚠️ GPMin not recommended for >100 atoms. Using BFGS instead.")
                                    optimizer = BFGS(optimization_object, logfile=None)
                                else:
                                    optimizer = GPMin(optimization_object, logfile=None, update_hyperparams=True)
                            elif opt_name == "SciPyFminBFGS":
                                optimizer = SciPyFminBFGS(optimization_object, logfile=None)
                            elif opt_name == "SciPyFminCG":
                                optimizer = SciPyFminCG(optimization_object, logfile=None)
                            else:  # Default fallback
                                optimizer = BFGS(optimization_object, logfile=None)

                            optimizer.max_steps = optimization_params['max_steps']
                            optimizer.attach(lambda: logger(optimizer), interval=1)

                            # Set convergence criteria based on optimization type
                            if opt_type == "Cell only (fixed atoms)":
                                fmax_criterion = 0.1
                            if is_tetragonal or (opt_type == "Cell only (fixed atoms)" and tetragonal_callback):
                                # Use manual convergence loop for tetragonal mode
                                log_queue.put(f"  🔷 Tetragonal mode: will check convergence manually")

                                fmax_criterion = optimization_params['fmax']

                                for step in range(optimization_params['max_steps']):
                                    optimizer.run(fmax=fmax_criterion, steps=1)

                                    if hasattr(optimization_object, 'atoms'):
                                        current_atoms = optimization_object.atoms
                                    else:
                                        current_atoms = optimization_object

                                    # Check convergence
                                    forces = current_atoms.get_forces()
                                    max_force = np.max(np.linalg.norm(forces, axis=1))
                                    energy = current_atoms.get_potential_energy()

                                    # Check stress for cell optimization
                                    try:
                                        stress_voigt = current_atoms.get_stress(voigt=True)
                                        max_stress = np.max(np.abs(stress_voigt))
                                    except:
                                        max_stress = 0.0

                                    # Check energy convergence
                                    energy_converged = False
                                    if logger.trajectory and len(logger.trajectory) > 1:
                                        energy_change = abs(
                                            logger.trajectory[-1]['energy'] - logger.trajectory[-2]['energy'])
                                        energy_converged = energy_change < optimization_params['ediff']


                                    if opt_type == "Atoms only (fixed cell)":
                                        force_converged = max_force < optimization_params['fmax']
                                        stress_converged = True
                                        converged = force_converged and energy_converged
                                    elif opt_type == "Cell only (fixed atoms)":
                                        force_converged = True
                                        stress_converged = max_stress < optimization_params.get('stress_threshold', 0.1)
                                        converged = stress_converged and energy_converged
                                    else:  # Both
                                        force_converged = max_force < optimization_params['fmax']
                                        stress_converged = max_stress < optimization_params.get('stress_threshold', 0.1)
                                        converged = force_converged and stress_converged and energy_converged

                                    if converged:
                                        log_queue.put(f"  ✅ Tetragonal optimization converged at step {step + 1}!")
                                        if opt_type == "Cell only (fixed atoms)":
                                            log_queue.put(
                                                f"     Stress: {max_stress:.4f} < {optimization_params.get('stress_threshold', 0.1)} GPa ✓")
                                        else:
                                            log_queue.put(
                                                f"     Force: {max_force:.4f} < {optimization_params['fmax']} eV/Å ✓")
                                            log_queue.put(
                                                f"     Stress: {max_stress:.4f} < {optimization_params.get('stress_threshold', 0.1)} GPa ✓")
                                        log_queue.put(
                                            f"     Energy change: {energy_change:.2e} < {optimization_params['ediff']} eV ✓")
                                        break

                                    if step >= optimization_params['max_steps'] - 1:
                                        log_queue.put(
                                            f"  ⚠️ Reached maximum steps ({optimization_params['max_steps']})")
                                        break
                            else:
                                # For non-tetragonal modes, use standard ASE convergence
                                fmax_criterion = optimization_params['fmax']
                                optimizer.run(fmax=fmax_criterion, steps=optimization_params['max_steps'])

                            # Get final structure
                            if hasattr(optimization_object, 'atoms'):
                                final_atoms = optimization_object.atoms
                            else:
                                final_atoms = optimization_object

                            energy = final_atoms.get_potential_energy()
                            final_forces = final_atoms.get_forces()
                            max_final_force = np.max(np.linalg.norm(final_forces, axis=1))

                            force_converged = max_final_force < optimization_params['fmax']
                            energy_converged = False
                            stress_converged = True

                            if len(logger.trajectory) > 1:
                                final_energy_change = logger.trajectory[-1]['energy_change']
                                energy_converged = final_energy_change < optimization_params['ediff']

                            if opt_mode in ["cell_only", "both"]:
                                try:
                                    final_stress = final_atoms.get_stress(voigt=True)
                                    max_final_stress = np.max(np.abs(final_stress))
                                    stress_converged = max_final_stress < 0.1
                                    log_queue.put(f"  Final stress: {max_final_stress:.4f} GPa")
                                except:
                                    stress_converged = True

                            # Determine convergence status
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

                            # Log final results
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
                                'final_steps': optimizer.nsteps if not is_tetragonal else len(logger.trajectory),
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







                    elif calc_type == "GA Structure Optimization":
                        if not substitutions:
                            log_queue.put("❌ No substitutions configured for GA optimization")
                            continue

                        log_queue.put(f"Starting GA structure optimization for {name}")
                        log_queue.put(f"Substitutions: {substitutions}")
                        log_queue.put(f"GA parameters: {ga_params}")

                        # Use the structure (which should be the supercell if set)
                        base_structure = structure
                        log_queue.put(f"Using structure with {len(base_structure)} atoms")

                        # Run GA optimization
                        ga_results = None
                        try:
                            # Start GA optimization in a separate thread
                            ga_thread = threading.Thread(
                                target=run_ga_optimization,
                                args=(base_structure, calculator, substitutions, ga_params, log_queue, stop_event)
                            )
                            ga_thread.start()

                            # Wait for GA thread to complete
                            ga_thread.join()

                            # Process messages - look for GA results
                            temp_messages = []
                            ga_finished = False

                            # Keep processing until we get the finish signal
                            while not ga_finished:
                                try:
                                    msg = log_queue.get(timeout=1.0)

                                    if isinstance(msg, dict) and msg.get('type') == 'ga_result':
                                        ga_results = msg
                                        energy = msg['best_energy']
                                        final_structure = msg['best_structure']
                                        log_queue.put(
                                            f"✅ GA optimization completed for {name}: Best energy = {energy:.6f} eV")

                                    elif msg == "GA_OPTIMIZATION_FINISHED":
                                        ga_finished = True
                                        log_queue.put(f"🏁 GA optimization finished for {name}")

                                    else:
                                        temp_messages.append(msg)

                                except queue.Empty:
                                    # Check if thread is still alive
                                    if not ga_thread.is_alive():
                                        # Thread finished but no finish message received
                                        log_queue.put("⚠️ GA thread finished unexpectedly")
                                        break
                                    continue

                            # Put back non-GA messages
                            for msg in temp_messages:
                                log_queue.put(msg)

                            if ga_results is None:
                                log_queue.put(f"⚠️ No GA results received for {name}")
                                energy = None

                        except Exception as ga_error:
                            log_queue.put(f"❌ GA optimization failed for {name}: {str(ga_error)}")
                            import traceback
                            log_queue.put(f"Traceback: {traceback.format_exc()}")
                            energy = None
                            ga_results = None


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
                                temp_optimizer.run(fmax=0.01, steps=400)
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
                    elif calc_type == "Molecular Dynamics":
                        log_queue.put(f"Starting molecular dynamics simulation for {name}")

                        # Run MD simulation
                        md_results = run_md_simulation(atoms, calculator, md_params, log_queue, name)

                        if md_results['success']:
                            energy = md_results['final_energy']

                            # --- THIS IS THE FIX ---
                            # 1. Get the final ASE 'atoms' object from the results
                            final_atoms_object = md_results['final_atoms']
                            # 2. Convert it back to a Pymatgen structure
                            final_structure = ase_to_pymatgen_wrapped(final_atoms_object)
                            # --- END FIX ---

                            log_queue.put(f"✅ MD simulation completed for {name}")
                        else:
                            log_queue.put(f"❌ MD simulation failed for {name}")
                            energy = None
                            final_structure = structure  # Keep original structure
                            md_results = None
                    elif calc_type == "Virtual Tensile Test":
                        log_queue.put(f"Starting virtual tensile test for {name}")

                        tensile_results = run_tensile_test(atoms, calculator, tensile_params, log_queue, name, stop_event)

                        if tensile_results['success']:
                            energy = atoms.get_potential_energy()
                            final_structure = structure
                            log_queue.put(f"✅ Tensile test completed for {name}")
                        else:
                            log_queue.put(f"❌ Tensile test failed for {name}")
                            energy = None
                            tensile_results = None

                    elif calc_type == "NEB Calculation":
                        log_queue.put(
                            f"DEBUG: Received NEB structures - initial: {neb_initial is not None}, finals: {len(neb_finals) if neb_finals else 0}")

                        if not neb_initial or not neb_finals:
                            log_queue.put("❌ No NEB structures configured")
                            log_queue.put("CALCULATION_FINISHED")
                            return

                        log_queue.put(f"Starting NEB calculations: 1 initial → {len(neb_finals)} final states")

                        log_queue.put(f"Starting NEB calculations: 1 initial → {len(neb_finals)} final states")

                        initial_structure = neb_initial

                        for final_name, final_structure in neb_finals.items():
                            if stop_event.is_set():
                                log_queue.put("Calculation stopped by user")
                                break

                            structure_start_time = time.time()
                            log_queue.put({
                                'type': 'structure_start_time',
                                'structure': f"Initial_to_{final_name}",
                                'start_time': structure_start_time
                            })

                            log_queue.put(f"Processing NEB: Initial → {final_name}")

                            neb_structure_name = f"Initial_to_{final_name}"

                            try:
                                neb_results = run_neb_calculation(
                                    initial_structure,
                                    final_structure,
                                    calculator,
                                    neb_params,
                                    log_queue,
                                    stop_event,
                                    neb_structure_name
                                )

                                if neb_results['success']:
                                    energy = neb_results['energies'][0]
                                    log_queue.put(f"✅ NEB completed: {neb_structure_name}")
                                else:
                                    log_queue.put(f"❌ NEB failed: {neb_structure_name}")
                                    energy = None

                                log_queue.put({
                                    'type': 'result',
                                    'name': neb_structure_name,
                                    'energy': energy,
                                    'structure': initial_structure,
                                    'calc_type': calc_type,
                                    'neb_results': neb_results
                                })

                                structure_end_time = time.time()
                                structure_duration = structure_end_time - structure_start_time

                                log_queue.put({
                                    'type': 'structure_end_time',
                                    'structure': neb_structure_name,
                                    'end_time': structure_end_time,
                                    'duration': structure_duration,
                                    'calc_type': calc_type
                                })

                            except Exception as e:
                                log_queue.put(f"❌ Error in NEB calculation {neb_structure_name}: {str(e)}")

                                structure_end_time = time.time()
                                structure_duration = structure_end_time - structure_start_time

                                log_queue.put({
                                    'type': 'structure_end_time',
                                    'structure': neb_structure_name,
                                    'end_time': structure_end_time,
                                    'duration': structure_duration,
                                    'calc_type': calc_type,
                                    'failed': True
                                })

                                log_queue.put({
                                    'type': 'result',
                                    'name': neb_structure_name,
                                    'energy': None,
                                    'structure': initial_structure,
                                    'calc_type': calc_type,
                                    'error': str(e)
                                })
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
                        'elastic_results': elastic_results,
                        'ga_results': ga_results if calc_type == "GA Structure Optimization" else None,
                        'md_results': md_results if calc_type == "Molecular Dynamics" else None,
                        'tensile_results': tensile_results if calc_type == "Virtual Tensile Test" else None,
                        'orb_confidence': orb_confidence
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


#st.title("uMLIP-Interactive: Compute properties with universal MLIPs")
colx1, colx2 = st.columns([2,1])
with colx1:
    st.markdown("## uMLIP-Interactive: Compute properties with universal MLIPs")
with colx2:
    # to cite the universal MLIPs
    show_citations = st.checkbox("📚 Show Model **Citations & GitHub** Repositories", value=False)
if show_citations:
    from helpers.cite_models import  create_citation_info
    create_citation_info()


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
    st.header("Model Selection")

    #if not MACE_AVAILABLE and not CHGNET_AVAILABLE and not MATTERSIM_AVAILABLE and not ORB_AVAILABLE:
    #    st.error("⚠️ No calculators available!")
    #    st.error("Please install MACE: `pip install mace-torch`")
    #    st.error("Or install CHGNet: `pip install chgnet`")
    #    st.error("Or install MatterSim: `pip install mattersim`")
    #    st.error("Or install ORB: `pip install orb-models`")
    #    st.error("Or install SevenNet: `pip install sevenn`")
    # st.stop()

    if MACE_OFF_AVAILABLE:
        #  st.success("✅ MACE-OFF (organic molecules) available"
        pass
    else:
        st.warning("⚠️ MACE-OFF not available (only MACE-MP)")

    defaults = st.session_state.default_settings
    model_keys = list(MACE_MODELS.keys())

    default_model_index = 0
    if defaults['selected_model'] in model_keys:
        default_model_index = model_keys.index(defaults['selected_model'])

    selected_model = st.selectbox(
        "Choose MLIP Model (MACE, CHGNet, SevenNet, Nequix, Orb-v3, MatterSim, PET-MAD)",
        model_keys,
        index=default_model_index
    )
    model_size = MACE_MODELS[selected_model]

    is_custom_mace = (selected_model == "Custom MACE Model 🔧")
    custom_mace_path = None

    if is_custom_mace:
        st.markdown("---")
        st.markdown("### 🔧 Custom MACE Model Configuration")

        custom_mace_path = st.text_input(
            "Path to .model file *",
            value="",
            placeholder="/path/to/your/model.model",
            help="Provide the full path to your custom MACE .model file"
        )

        # Validate the path
        if custom_mace_path:
            import os

            if os.path.exists(custom_mace_path):
                if os.path.isfile(custom_mace_path):
                    if custom_mace_path.endswith('.model'):
                        st.success(f"✅ Model file found")
                        model_name = os.path.basename(custom_mace_path)
                        model_dir = os.path.dirname(custom_mace_path)
                        st.info(f"**Name:** `{model_name}`\n\n**Path:** `{model_dir}`")
                    else:
                        st.warning("⚠️ File should have .model extension")
                elif os.path.isdir(custom_mace_path):
                    st.error("❌ This is a directory. Please provide the full path to the .model file.")
                    try:
                        model_files = [f for f in os.listdir(custom_mace_path) if f.endswith('.model')]
                        if model_files:
                            st.info(f"Found {len(model_files)} .model file(s) in this directory:")
                            for mf in model_files[:5]:
                                st.code(os.path.join(custom_mace_path, mf))
                    except:
                        pass
                else:
                    st.error("❌ Path exists but is not a regular file")
            else:
                st.error("❌ File not found at this path")
        else:
            st.warning("⚠️ Please provide a path to your custom .model file")

    is_petmad = selected_model.startswith("PET-MAD")
    is_chgnet = selected_model.startswith("CHGNet")

    is_sevennet = selected_model.startswith("SevenNet")

    is_mattersim = selected_model.startswith("MatterSim")
    is_nequix = selected_model.startswith("Nequix")
    if is_petmad:
        if not PETMAD_AVAILABLE:
            st.error("❌ PET-MAD not available! Please install: `pip install pet-mad`")
            st.stop()

        st.info("🧠 **PET-MAD**: Equivariant message-passing neural network potential")

        is_universal_petmad = model_size == "petmad-v1.0.2-universal"
    elif is_mattersim:
        if not MATTERSIM_AVAILABLE:
            st.error("❌ MatterSim not available! Please install: `pip install mattersim`")
            st.stop()
        st.info("🧠 **MatterSim**: Universal ML potential for bulk materials")
    elif is_nequix:
        if not NEQUIX_AVAILABLE:
            st.error("❌ Nequix not available! Please install: `pip install nequix`")
            st.stop()
        st.info("🧠 **Nequix**: Foundation model for materials trained on MPtrj data")
    elif is_sevennet:
        if not SEVENNET_AVAILABLE:
            st.error("❌ SevenNet not available! Please install: `pip install sevenn`")
            st.stop()
        st.info("⚡ **SevenNet**: Fast universal ML potential")
    elif is_chgnet:
        if not CHGNET_AVAILABLE:
            st.error("❌ CHGNet not available! Please install: `pip install chgnet`")
            st.stop()
        st.info("🔬 **CHGNet**: Universal potential with magnetic moments")
        st.info("📊 **Coverage**: All elements, 146k compounds from Materials Project")
    else:
        model_type, description = get_model_type_from_selection(selected_model)
        if model_type == "MACE-OFF":
            st.info(f"🧪 **{model_type}**: {description}")
            if not MACE_OFF_AVAILABLE:
                st.error("❌ MACE-OFF models require updated MACE installation!")
        else:
            if not MACE_AVAILABLE:
                st.error("❌ MACE models not available! Please install: 'pip install mace-torch'")
            st.info(f"🔬 **{model_type}**: {description}")

    cols1, cols2 = st.columns([1, 1])
    with cols1:
        default_device_index = 0 if defaults['device'] == "cpu" else 1
        device_option = st.radio(
            "Compute Device",
            ["CPU", "GPU (CUDA)"],
            index=default_device_index,
            help="GPU will be much faster if available. Falls back to CPU if GPU unavailable."
        )
        device = "cuda" if device_option == "GPU (CUDA)" else "cpu"

    with cols2:
        if not selected_model.startswith("CHGNet"):
            default_precision_index = 0 if defaults['dtype'] == "float32" else 1
            precision_option = st.radio(
                "Precision",
                ["Float32", "Float64"],
                index=default_precision_index,
                help="Float32 uses less memory but lower precision. Float64 is more accurate but uses more memory."
            )
            dtype = "float32" if precision_option == "Float32" else "float64"
        else:
            st.info("CHGNet uses fixed precision.")
            dtype = "float32"
    mace_head = None
    mace_dispersion = False
    mace_dispersion_xc = "pbe"

    col_mult1, col_mult2 = st.columns([1, 1])
    # Only show for MACE models (not other calculators)

    if not any(x in selected_model for x in ["CHGNet", "SevenNet", "MatterSim", "ORB", "Nequix"]):
       # st.markdown("---")

        is_mh_model = is_multihead_model(selected_model)

        if is_mh_model:
            #st.subheader("🎯 Multi-Head Configuration")
            #st.success("Multi-head model - head selection required")
            with col_mult1:
                mace_head = st.selectbox(
                    "Select Prediction Head *",
                    ["omat_pbe", "matpes_r2scan", "omol", "mp_pbe_refit_add", "spice_wB97M", "oc20_usemppbe"],
                    index=0,  # Default to omat_pbe (recommended)
                    help="REQUIRED: Select which head to use for predictions"
                )

            with st.expander("ℹ️ About Prediction Heads"):
                st.markdown("""
                **Available Heads** (from model):

                - **omat_pbe** ⭐ (Recommended): State-of-the-art across inorganic, organic, surfaces. Trained on OMAT dataset with PBE.

                - **matpes_r2scan**: MATPES dataset with r2SCAN functional. Good for strongly correlated materials.

                - **omol**: Organic molecules (OMOL dataset). Best for molecular systems and conformations.

                - **mp_pbe_refit_add**: Materials Project PBE refit. Good for general materials.

                - **spice_wB97M**: SPICE dataset with ωB97M-D3. Excellent for small molecules and reactions.

                - **oc20_usemppbe**: Open Catalyst 2020. Optimized for catalysis and surface reactions.

                **Note**: The `omat_pbe` head demonstrates state-of-the-art performance and is recommended for most applications.

                📄 Paper: https://arxiv.org/abs/2510.25380
                """)



        # Dispersion correction
            with col_mult2:
                st.subheader("🔬 Dispersion Correction")
                mace_dispersion = st.checkbox(
                    "Enable D3 Dispersion",
                    value=False,
                    help="Add D3 dispersion correction for van der Waals interactions"
                )

                if mace_dispersion:
                    mace_dispersion_xc = st.selectbox(
                        "Functional",
                        ["pbe", "pbesol", "rpbe", "blyp", "revpbe"],
                        index=0
                    )
                    st.caption(f"D3-{mace_dispersion_xc} will be applied")

    # Store in session state
    if 'mace_config' not in st.session_state:
        st.session_state.mace_config = {}

    st.session_state.mace_config = {
        'head': mace_head,
        'dispersion': mace_dispersion,
        'dispersion_xc': mace_dispersion_xc
    }
    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        st.session_state.thread_count = st.number_input(
            "CPU Threads",
            min_value=1,
            max_value=32,
            value=st.session_state.thread_count,
            step=1,
            help="Number of CPU threads for calculations"
        )

    with col_c2:
        if st.button("💾 Save as Default"):
            new_settings = {
                'thread_count': st.session_state.thread_count,
                'selected_model': selected_model,
                'device': device,
                'dtype': dtype
            }

            if save_default_settings(new_settings):
                st.session_state.default_settings = new_settings
                os.environ['OMP_NUM_THREADS'] = str(st.session_state.thread_count)
                torch.set_num_threads(st.session_state.thread_count)
                st.toast("✅ All settings saved as default!")
            else:
                st.toast("❌ Failed to save settings")
css = '''
<style>
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.15rem !important;
    color: #1e3a8a !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 20px !important;
}

.stTabs [data-baseweb="tab-list"] button {
    background-color: #f0f4ff !important;
    border-radius: 12px !important;
    padding: 8px 16px !important;
    transition: all 0.3s ease !important;
    border: none !important;
    color: #1e3a8a !important;
}

.stTabs [data-baseweb="tab-list"] button:hover {
    background-color: #dbe5ff !important;
    cursor: pointer;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: #e0e7ff !important;
    color: #1e3a8a !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(30, 58, 138, 0.3) !important;
}

.stTabs [data-baseweb="tab-list"] button:focus {
    outline: none !important;
}
</style>
'''

st.markdown(css, unsafe_allow_html=True)

if st.session_state.calculation_running:
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.spinner("Calculation in progress..."):
                st.info("The calculations are running, please wait. 😊")
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


tab1, tab_st, tab2, tab3, tab4, tab4_1,tab_vir, tab5,  = st.tabs(
    ["📁 Structure Upload & Setup", "✅ Start Calculations", "🖥️ Calculation Console", "📊 Results & Analysis",
     "📈 Optimization Trajectories and Convergence", "🧬 MD Trajectories and Analysis", "🔧 Virtual Tensile Tests",  "🔬 MACE Models Info"])

with tab_vir:
    st.header("Virtual Tensile Test Results")

    tensile_results_list = [r for r in st.session_state.results if
                            r['calc_type'] == 'Virtual Tensile Test' and r.get('tensile_results')]

    if tensile_results_list:
        st.subheader("🔧 Mechanical Properties from Tensile Testing")

        if len(tensile_results_list) == 1:
            selected_tensile = tensile_results_list[0]
        else:
            tensile_names = [r['name'] for r in tensile_results_list]
            selected_name = st.selectbox("Select structure:", tensile_names, key="tensile_selector")
            selected_tensile = next(r for r in tensile_results_list if r['name'] == selected_name)

        tensile_data = selected_tensile['tensile_results']

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.metric("Ultimate Stress", f"{tensile_data['ultimate_stress']:.2f} GPa")

        with col_m2:
            if tensile_data['youngs_modulus']:
                st.metric("Young's Modulus", f"{tensile_data['youngs_modulus']:.2f} GPa")
            else:
                st.metric("Young's Modulus", "N/A")

        with col_m3:
            if tensile_data['yield_strain']:
                st.metric("Yield Strain", f"{tensile_data['yield_strain']:.2f}%")
            else:
                st.metric("Yield Strain", "N/A")

        with col_m4:
            st.metric("Max Strain", f"{tensile_data['max_strain_reached']:.2f}%")

        st.subheader("📊 Stress-Strain Analysis")
        fig_tensile = create_stress_strain_plot(tensile_data)
        st.plotly_chart(fig_tensile, use_container_width=True)

        with st.expander("Test Parameters"):
            params_data = {
                'Parameter': [
                    'Strain Direction',
                    'Temperature',
                    'Strain Rate',
                    'Max Strain',
                    'Timestep',
                    'Equilibration Steps'
                ],
                'Value': [
                    tensile_data['strain_direction'],
                    f"{tensile_data['tensile_params']['temperature']} K",
                    f"{tensile_data['tensile_params']['strain_rate']}%/ps",
                    f"{tensile_data['tensile_params']['max_strain']}%",
                    f"{tensile_data['tensile_params']['timestep']} fs",
                    f"{tensile_data['tensile_params']['equilibration_steps']}"
                ]
            }
            st.dataframe(params_data, use_container_width=True, hide_index=True)

        st.subheader("📥 Download Data")
        col_dl1, col_dl2, col_dl3 = st.columns(3)

        with col_dl1:
            tensile_json = export_tensile_results(tensile_data, selected_tensile['name'])
            st.download_button(
                label="📊 Download Results (JSON)",
                data=tensile_json,
                file_name=f"tensile_test_{selected_tensile['name'].replace('.', '_')}.json",
                mime="application/json",
                type='primary'
            )

        with col_dl2:
            if tensile_data.get('trajectory_data'):
                from helpers.tensile_test import create_tensile_trajectory_xyz

                xyz_content = create_tensile_trajectory_xyz(
                    tensile_data['trajectory_data'],
                    selected_tensile['name'],
                    tensile_data['tensile_params']
                )

                if xyz_content:
                    st.download_button(
                        label="📥 Download Trajectory (XYZ)",
                        data=xyz_content,
                        file_name=f"tensile_trajectory_{selected_tensile['name'].replace('.', '_')}.xyz",
                        mime="text/plain",
                        type='primary'
                    )
            else:
                st.info("No trajectory data available")
with tab4_1:
    st.header("MD Trajectories and Analysis")

    st.write("DEBUG INFO:")
    st.write(f"'md_trajectories' in session_state: {'md_trajectories' in st.session_state}")
    if 'md_trajectories' in st.session_state:
        st.write(f"md_trajectories keys: {list(st.session_state.md_trajectories.keys())}")
        for key, value in st.session_state.md_trajectories.items():
            st.write(
                f"  {key}: success={value.get('success', 'N/A')}, has_trajectory={len(value.get('trajectory_data', [])) > 0}")
    st.write("---")

    if 'md_trajectories' in st.session_state and st.session_state.md_trajectories:
        for structure_name, result in st.session_state.md_trajectories.items():
            if result['success']:
                st.subheader(f"Results for: {structure_name}")

                trajectory_data = result.get('trajectory_data', [])
                md_params = result.get('md_params', {})

                if trajectory_data:
                    fig_main, fig_pressure, fig_conservation = create_md_analysis_plots(
                        trajectory_data,
                        md_params
                    )

                    if fig_main:
                        st.plotly_chart(fig_main, use_container_width=True)

                    if fig_pressure:
                        st.subheader("Pressure Evolution")
                        st.plotly_chart(fig_pressure, use_container_width=True)

                    if fig_conservation:
                        st.subheader("Energy Conservation")
                        st.plotly_chart(fig_conservation, use_container_width=True)

                    if md_params.get('ensemble') == 'NPT':
                        st.subheader("NPT Cell Evolution Analysis")
                        st.info(
                            "📊 Tracking lattice parameters, volume, angles, and density changes during NPT simulation")

                        npt_fig = create_npt_analysis_plots(trajectory_data, md_params)
                        if npt_fig:
                            st.plotly_chart(npt_fig, use_container_width=True)

                        # NPT metrics
                        final_data = trajectory_data[-1]
                        initial_data = trajectory_data[0]

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Volume Change",
                                f"{final_data['volume']:.2f} Å³",
                                f"{((final_data['volume'] - initial_data['volume']) / initial_data['volume'] * 100):.2f}%"
                            )
                        with col2:
                            if 'mass' in final_data and final_data['volume'] > 0:
                                final_density = (final_data['mass'] / final_data['volume']) * 1.66054
                                st.metric("Final Density", f"{final_density:.3f} g/cm³")
                        with col3:
                            if final_data.get('pressure') is not None:
                                st.metric("Final Pressure", f"{final_data['pressure']:.2f} GPa")

                    st.markdown("---")

                    st.subheader("Download Trajectory")

                    element_symbols_list = None
                    if structure_name in st.session_state.structures:
                        try:
                            element_symbols_list = [site.specie.symbol for site in
                                                    st.session_state.structures[structure_name]]
                        except AttributeError:
                            try:
                                element_symbols_list = st.session_state.structures[
                                    structure_name].get_chemical_symbols()
                            except:
                                pass

                    xyz_content = create_md_trajectory_xyz(
                        trajectory_data,
                        structure_name,
                        md_params,
                        element_symbols=element_symbols_list
                    )

                    st.download_button(
                        label="📥 Download MD Trajectory (XYZ)",
                        data=xyz_content,
                        file_name=f"md_trajectory_{structure_name.replace('.', '_')}.xyz",
                        mime="text/plain",
                        help="Extended XYZ format with cell parameters, velocities, and energies",
                        type='primary'
                    )

                    json_export = export_md_results(result, structure_name)
                    if json_export:
                        st.download_button(
                            label="📊 Download MD Summary (JSON)",
                            data=json_export,
                            file_name=f"md_summary_{structure_name.replace('.', '_')}.json",
                            mime="application/json",
                            help="JSON file with MD parameters and statistics"
                        )

                else:
                    st.warning(f"No trajectory data available for {structure_name}")

    elif st.session_state.calculation_running:
        st.info("🔄 MD simulation in progress... Results will appear here when complete.")

    else:
        st.info("Run an MD simulation to see trajectory analysis and results here.")
        st.markdown("""
        **Available MD Features:**
        - Energy, temperature, and pressure evolution plots
        - **NPT-specific:** Cell parameter evolution, volume changes, density tracking
        - Trajectory download in extended XYZ format
        - Summary statistics export
        """)

with tab5:
    display_mace_models_info()
    st.markdown("---")

    create_citation_info()

    st.markdown("---")

    st.info("""
    💡 **Tips for Model Selection:**

    • **For general use**: MACE-MP-0b3 (medium) - Latest and most stable
    • **For highest accuracy**: MACE-MPA-0 (medium) - State-of-the-art performance  
    • **For phonon calculations**: MACE-OMAT-0 (medium) - Excellent vibrational properties
    • **For fast screening**: Any small model - Lower computational cost
    • **For complex systems**: Large models - Higher accuracy for difficult cases
    """)

with tab1:
    st.sidebar.header("Upload Structure Files")
    if not st.session_state.structures_locked:
        uploaded_files = st.sidebar.file_uploader(
            "Upload structure files (CIF, POSCAR, LMP, XSF, PW, CFG, etc.)",
            accept_multiple_files=True,
            type=None,
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
        st.success(f"🔒 Structures Locked ({len(st.session_state.structures)} structures). "
                   f"📌 Structures are locked to avoid refreshing during the calculation run. Use 'Unlock' to modify.")

        with st.expander("📋 Locked Structures", expanded=False):
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
        "Try also the main application **[XRDlicious](xrdlicious.com)**. 🌀 Developed by **[IMPLANT team](https://implant.fs.cvut.cz/)**. "
        "📺 **[Quick tutorial here](https://youtu.be/xh98fQqKXaI?si=JaOUFhYoMQvPmNbB)**. See our corresponding **[article (arXiv)](https://arxiv.org/abs/2512.05568)**. Spot a bug or have a feature requests? Let us know at **lebedmi2@cvut.cz**."
    )

    st.sidebar.link_button("GitHub page", "https://github.com/bracerino/mace-md-gui",
                           type="primary")
    if st.session_state.structures:
        pass
    else:
        st.markdown(
            """
        <div style="
          background-color: #e8f4fd;
          border-left: 6px solid #2196f3;
          padding: 15px;
          border-radius: 8px;
          font-family: Arial, sans-serif;
          color: #0d47a1;
          max-width: 800px;
          margin: 10px 0;
        ">
          <strong>ℹ️ Info:</strong> Please upload at least one crystal structure file 
          (<code>.cif</code>, <code>.poscar / .vasp / POSCAR</code>, <code>extended .xyz</code>, <code>.lmp</code>).<br><br>

          If you like the uMLIP-Interactive, please
          <strong><a style="color:#0b63c4;" href="https://arxiv.org/abs/2512.05568" target="_blank">📖 cite this work</a></strong>
          Please also cite the:
          <strong><a style="color:#0b63c4;" href="https://doi.org/10.1088/1361-648X/aa680e" target="_blank">📖 Atomic Simulation Environment (ASE)</a></strong>
          and the publications corresponding to the employed uMLIPs:
          <strong><span style="color:#0b63c4;">(see 'Show Model Citations' in the right corner)</span></strong>
        </div>
            """,
            unsafe_allow_html=True,
        )

    if True:
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

                        is_compatible, unsupported, elements, detected_model_type = check_mace_compatibility(structure,
                                                                                                             selected_model)

                        if is_compatible:
                            st.success(f"✅ Compatible with {detected_model_type}")
                        else:
                            st.error(f"❌ Unsupported elements: {', '.join(unsupported)}")

                        st.write(f"**Elements:** {', '.join(elements)}")
                        if hasattr(structure, 'constraints_info') and structure.constraints_info:
                            st.write("📌 **Selective Dynamics:** Present (some atoms fixed)")
                        else:
                            st.write("🔄 **Selective Dynamics:** None (all atoms free to move)")

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
                ["Energy Only", "Geometry Optimization", "Phonon Calculation", "Elastic Properties",
                 "GA Structure Optimization", "Molecular Dynamics", "Virtual Tensile Test",  "NEB Calculation"],
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
            elif calc_type == "GA Structure Optimization":
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 20px; border-radius: 15px; margin-top: 15px; color: white; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <strong style="font-size: 22px; display: block; margin-bottom: 10px;">🧬 Evolutionary Optimization</strong>
                <div style="font-size: 18px; line-height: 1.4;">
                Optimal substitution patterns<br>
                & defect configurations
                </div>
                </div>
                """, unsafe_allow_html=True)
        if calc_type == "Molecular Dynamics":
            #st.warning("**!!! UNDER CONSTRUCTION !!! NOT EVERTHING WORKING YET PROPERLY FOR THIS OPTION**")
            md_params = setup_md_parameters_ui()
            st.divider()
            st.subheader("Generate Standalone MD Script")

            if 'generated_md_script' not in st.session_state:
                st.session_state.generated_md_script = None

            if st.button("📝 Generate MD Python Script (using current settings)", key="generate_md_script_button",
                         type="secondary"):
                try:
                    current_selected_model = selected_model
                    current_model_size = model_size
                    current_device = device
                    current_dtype = dtype
                    current_thread_count = st.session_state.thread_count
                    mace_config = st.session_state.get('mace_config', {})
                    mace_head_for_script = mace_config.get('head')
                    mace_dispersion_for_script = mace_config.get('dispersion', False)
                    mace_dispersion_xc_for_script = mace_config.get('dispersion_xc', 'pbe')
                    generated_script = generate_md_python_script(
                        md_params,
                        current_selected_model,
                        current_model_size,
                        current_device,
                        current_dtype,
                        current_thread_count,
                        mace_head=mace_head_for_script,
                        mace_dispersion=mace_dispersion_for_script,
                        mace_dispersion_xc=mace_dispersion_xc_for_script,
                        custom_mace_path=custom_mace_path
                    )
                    st.session_state.generated_md_script = generated_script
                    st.success("✅ MD script generated successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to generate script: {str(e)}")
                    st.session_state.generated_md_script = None

            if st.session_state.generated_md_script:
                with st.expander("🐍 View Generated MD Script", expanded=True):
                    st.code(st.session_state.generated_md_script, language='python')

                    st.download_button(
                        label="💾 Download MD Script (.py)",
                        data=st.session_state.generated_md_script,
                        file_name="run_md_simulation.py",
                        mime="text/x-python",
                        key="download_generated_md_script",
                        type='primary'
                    )
                st.info("""
                        **Instructions:**
                        1. Save the script (e.g., `run_md_simulation.py`).
                        2. Place your structure files (`.cif`, `.vasp`, `POSCAR`) in the same directory.
                        3. Create a subdirectory named `md_results`.
                        4. Ensure necessary libraries are installed (`pip install ase mace-torch ...`).
                        5. Run the script from your terminal: `python run_md_simulation.py`
                        """)

        if calc_type == "Virtual Tensile Test":
            st.warning("**!!! UNDER CONSTRUCTION !!! NOT EVERTHING WORKING YET PROPERLY FOR THIS OPTION**")
            tensile_params = setup_tensile_test_ui(
                default_settings=st.session_state.default_settings,
                save_settings_function=save_default_settings
            )
            st.subheader("Generate Standalone MD Script")

            st.markdown("---")
            st.subheader("Generate Standalone Tensile Test Script")

            if 'generated_tensile_script' not in st.session_state:
                st.session_state.generated_tensile_script = None


            from helpers.generate_tensile_test_python_script import generate_tensile_test_python_script
            if st.button("📝 Generate Tensile Test Python Script (using current settings)",
                         key="generate_tensile_script_button",
                         type="secondary"):
                try:
                    current_selected_model = selected_model
                    current_model_size = model_size
                    current_device = device
                    current_dtype = dtype
                    current_thread_count = st.session_state.thread_count

                    # Call the tensile script generation function, passing tensile_params
                    generated_script = generate_tensile_test_python_script(
                        tensile_params,
                        current_selected_model,
                        current_model_size,
                        current_device,
                        current_dtype,
                        current_thread_count
                    )
                    st.session_state.generated_tensile_script = generated_script
                    st.success("✅ Tensile test script generated successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to generate tensile script: {str(e)}")
                    st.session_state.generated_tensile_script = None

            if st.session_state.generated_tensile_script:
                with st.expander("🐍 View Generated Tensile Test Script", expanded=True):
                    st.code(st.session_state.generated_tensile_script, language='python')


                    st.download_button(
                        label="💾 Download Tensile Script (.py)",
                        data=st.session_state.generated_tensile_script,
                        file_name="run_tensile_test.py",
                        mime="text/x-python",
                        key="download_generated_tensile_script",
                        type='primary'
                    )
                st.info("""
                        **Instructions (Tensile Test):**
                        1. Save the script (e.g., `run_tensile_test.py`).
                        2. Place your structure file (`.cif`, `.vasp`, `POSCAR`) in the same directory. **(Usually only one structure per test)**.
                        3. Ensure necessary libraries are installed (`pip install ase pandas matplotlib ...`).
                        4. Run the script from your terminal: `python run_tensile_test.py`
                        5. Results (`_tensile_data.csv`, plots `.png`) will be saved in the `md_results` subdirectory.
                        """)
        if calc_type == "NEB Calculation":
            st.subheader("NEB: Initial and Final States")

            st.info(
                "Upload ONE initial structure and ONE or MORE final structures. NEB will compute the minimum energy path for each initial→final pair.")

            if 'neb_initial_structure' not in st.session_state:
                st.session_state.neb_initial_structure = None
            if 'neb_final_structures' not in st.session_state:
                st.session_state.neb_final_structures = {}

            col_init, col_final = st.columns(2)

            with col_init:
                st.write("**Initial Structure**")
                initial_file = st.file_uploader(
                    "Upload initial structure",
                    type=['vasp', 'cif', 'poscar', 'xyz'],
                    key="neb_initial_uploader"
                )

                if initial_file:
                    try:
                        initial_structure = load_structure(initial_file)
                        st.session_state.neb_initial_structure = initial_structure
                        st.success(
                            f"Initial: {initial_structure.composition.reduced_formula} ({len(initial_structure)} atoms)")
                    except Exception as e:
                        st.error(f"Error loading initial structure: {e}")

            with col_final:
                st.write("**Final Structure(s)**")
                final_files = st.file_uploader(
                    "Upload final structure(s)",
                    type=['vasp', 'cif', 'poscar', 'xyz'],
                    accept_multiple_files=True,
                    key="neb_final_uploader"
                )

                if final_files:
                    new_finals = {}
                    for final_file in final_files:
                        try:
                            final_structure = load_structure(final_file)
                            new_finals[final_file.name] = final_structure
                        except Exception as e:
                            st.error(f"Error loading {final_file.name}: {e}")

                    if new_finals:
                        st.session_state.neb_final_structures = new_finals
                        st.success(f"{len(new_finals)} final structure(s) loaded")
                        for name, struct in new_finals.items():
                            st.write(f"  • {name}: {struct.composition.reduced_formula} ({len(struct)} atoms)")

            if st.session_state.neb_initial_structure and st.session_state.neb_final_structures:
                st.success(f"Ready to compute {len(st.session_state.neb_final_structures)} NEB path(s)")

                neb_params = setup_neb_parameters_ui()
        optimization_params = {
            'optimizer': "BFGS",
            'fmax': 0.05,
            'ediff': 1e-4,
            'max_steps': 200,

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

        if calc_type == "GA Structure Optimization":
            st.divider()
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.subheader("🔢 Supercell Generation")
            if not st.session_state.structures:
                st.warning("⚠️ Please upload at least one structure first")
            else:
                first_structure_name = list(st.session_state.structures.keys())[0]
                first_structure = st.session_state.structures[first_structure_name]

                st.info(f"🧬 **GA Optimization will use base structure:** {first_structure_name}")
                st.info(
                    f"📊 **Base composition:** {first_structure.composition.reduced_formula} ({len(first_structure)} atoms)")

                if 'supercell_confirmed' not in st.session_state:
                    st.session_state.supercell_confirmed = False
                if 'confirmed_supercell_structure' not in st.session_state:
                    st.session_state.confirmed_supercell_structure = None

                if not st.session_state.supercell_confirmed:

                    col_super1, col_super2, col_super3, col_super4 = st.columns(4)

                    with col_super1:
                        enable_supercell = st.checkbox("Generate Supercell", value=False,
                                                       help="Create a larger supercell before applying substitutions")

                    supercell_structure = first_structure

                    if enable_supercell:
                        with col_super2:
                            supercell_a = st.number_input("a-direction", min_value=1, max_value=10, value=2, step=1)
                        with col_super3:
                            supercell_b = st.number_input("b-direction", min_value=1, max_value=10, value=2, step=1)
                        with col_super4:
                            supercell_c = st.number_input("c-direction", min_value=1, max_value=10, value=2, step=1)

                        st.session_state.supercell_multipliers = [supercell_a, supercell_b, supercell_c]

                        supercell_structure = first_structure.copy()
                        supercell_structure.make_supercell([supercell_a, supercell_b, supercell_c])

                        # Display supercell info
                        st.success(
                            f"✅ Supercell preview: {supercell_structure.composition.reduced_formula} ({len(supercell_structure)} atoms)")

                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.write("**Original Structure:**")
                            st.write(f"• Atoms: {len(first_structure)}")
                            st.write(f"• Formula: {first_structure.composition.reduced_formula}")
                            st.write(
                                f"• Lattice: {first_structure.lattice.a:.3f} × {first_structure.lattice.b:.3f} × {first_structure.lattice.c:.3f} Å")

                        with col_info2:
                            st.write("**Supercell Preview:**")
                            st.write(f"• Atoms: {len(supercell_structure)}")
                            st.write(f"• Formula: {supercell_structure.composition.reduced_formula}")
                            st.write(
                                f"• Lattice: {supercell_structure.lattice.a:.3f} × {supercell_structure.lattice.b:.3f} × {supercell_structure.lattice.c:.3f} Å")
                            st.write(f"• Multiplier: {supercell_a}×{supercell_b}×{supercell_c}")

                        st.subheader("📊 Concentration Resolution Analysis")

                        unique_elements = list(set([site.specie.symbol for site in supercell_structure]))

                        for element in unique_elements:
                            original_count = sum(1 for site in first_structure if site.specie.symbol == element)
                            supercell_count = sum(1 for site in supercell_structure if site.specie.symbol == element)

                            col_elem1, col_elem2 = st.columns(2)

                            with col_elem1:
                                st.write(f"**{element} atoms:**")
                                st.write(f"• Original: {original_count} atoms")
                                st.write(f"• Supercell: {supercell_count} atoms")

                            with col_elem2:
                                st.write(f"**Concentration steps:**")
                                if original_count == 1:
                                    st.write("• Original: 0%, 100% (2 options)")
                                else:
                                    original_step = 100 / original_count
                                    st.write(f"• Original: {original_step:.1f}% steps ({original_count + 1} options)")

                                supercell_step = 100 / supercell_count
                                st.write(f"• Supercell: {supercell_step:.1f}% steps ({supercell_count + 1} options)")

                    col_confirm1, col_confirm2 = st.columns([1, 3])

                    with col_confirm1:
                        if st.button("✅ Confirm Structure", type="primary", ):
                            st.session_state.supercell_confirmed = True
                            st.session_state.confirmed_supercell_structure = supercell_structure
                            st.success("✅ Structure confirmed!")
                            st.rerun()

                    with col_confirm2:
                        if enable_supercell:
                            st.info("⚠️ Confirm the supercell configuration before proceeding to substitution setup")

                else:
                    confirmed_structure = st.session_state.confirmed_supercell_structure
                    is_supercell = len(confirmed_structure) > len(first_structure)

                    if is_supercell:
                        st.success(
                            f"✅ **Confirmed Supercell:** {confirmed_structure.composition.reduced_formula} ({len(confirmed_structure)} atoms)")
                    else:
                        st.success(
                            f"✅ **Confirmed Original Structure:** {confirmed_structure.composition.reduced_formula} ({len(confirmed_structure)} atoms)")

                    col_confirmed1, col_confirmed2, col_confirmed3 = st.columns(3)

                    with col_confirmed1:
                        st.write("**Structure Details:**")
                        st.write(f"• Atoms: {len(confirmed_structure)}")
                        st.write(f"• Formula: {confirmed_structure.composition.reduced_formula}")
                        st.write(
                            f"• Lattice: {confirmed_structure.lattice.a:.3f} × {confirmed_structure.lattice.b:.3f} × {confirmed_structure.lattice.c:.3f} Å")

                    with col_confirmed2:
                        if is_supercell:
                            volume_ratio = confirmed_structure.lattice.volume / first_structure.lattice.volume
                            multiplier = round(volume_ratio ** (1 / 3))
                            st.write("**Supercell Info:**")
                            st.write(f"• Multiplier: ~{multiplier}×{multiplier}×{multiplier}")
                            st.write(f"• Volume ratio: {volume_ratio:.1f}×")
                            st.write(f"• Atom ratio: {len(confirmed_structure) / len(first_structure):.0f}×")
                        if st.button("🔄 Reset Structure", type="secondary",
                                     ):
                            st.session_state.supercell_confirmed = False
                            st.session_state.confirmed_supercell_structure = None
                            st.session_state.substitutions = {}
                            st.session_state.ga_base_structure = None
                            st.info("🔄 Structure reset. You can now reconfigure the supercell.")
                            st.rerun()

                if st.session_state.supercell_confirmed and st.session_state.confirmed_supercell_structure:
                    working_structure = st.session_state.confirmed_supercell_structure

                    substitutions = setup_substitution_ui(working_structure)
                    st.session_state.substitutions = substitutions
                    st.session_state.ga_base_structure = working_structure

                    ga_params = setup_ga_parameters_ui(
                        working_structure=working_structure,
                        substitutions=substitutions,
                        load_structure_func=load_structure
                    )
                    st.session_state.ga_params = ga_params

                    if not substitutions:
                        st.warning("⚠️ Please configure at least one element substitution to enable GA optimization")
                    else:
                        st.subheader("✅ GA Configuration Summary")

                        total_atoms = len(working_structure)
                        # Calculate totals based on the new structure
                        total_substitutions = 0
                        total_vacancies = 0

                        for element, sub_info in substitutions.items():
                            if 'concentration_list' in sub_info:
                                # New structure: use first concentration for display
                                concentration = sub_info['concentration_list'][0]
                                element_count = sub_info['element_count']
                                n_substitute = int(element_count * concentration)

                                if sub_info['new_element'] == 'VACANCY':
                                    total_vacancies += n_substitute
                                else:
                                    total_substitutions += n_substitute
                            else:
                                # Fallback for old structure (shouldn't happen with new code)
                                if 'n_substitute' in sub_info:
                                    if sub_info['new_element'] == 'VACANCY':
                                        total_vacancies += sub_info['n_substitute']
                                    else:
                                        total_substitutions += sub_info['n_substitute']

                        col_val1, col_val2, col_val3, col_val4 = st.columns(4)

                        with col_val1:
                            st.metric("Total Atoms", total_atoms)
                        with col_val2:
                            st.metric("Substitutions", total_substitutions)
                        with col_val3:
                            st.metric("Substitution %", f"{(total_substitutions / total_atoms) * 100:.1f}%")
                        with col_val4:
                            st.metric("GA Runs", ga_params.get('num_runs', 1))
                    st.subheader(
                        "✅ Start the GA run by changing the Tab to 'Start Calculations' at the top of the site.")
                else:
                    st.info("👆 Please confirm your structure configuration above before setting up substitutions")
            display_ga_overview()
        if calc_type == "Geometry Optimization":
            optimization_params = setup_geometry_optimization_ui(
                default_settings=st.session_state.default_settings,
                cell_opt_available=CELL_OPT_AVAILABLE,
                save_settings_function=save_default_settings
            )
            display_optimization_info(optimization_params)

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
                "A brief pre-optimization (fmax=0.01 eV/Å, max 400 steps) will be performed for stability before elastic calculations.")

            elastic_params['strain_magnitude'] = st.number_input("Strain Magnitude (e.g., 0.01 for 1%)",
                                                                 min_value=0.001, max_value=0.1, value=0.01, step=0.001,
                                                                 format="%.3f")
            # elastic_params['density'] = st.number_input("Material Density (g/cm³)", min_value=0.1, value=None,
            #                                            help="Optional: Provide if known. Otherwise, it will be estimated from the structure.",
            #                                            format="%.3f")
            elastic_params['density'] = None
            save_trajectory= True



    else:
        st.info("Upload structure files to begin")
with tab_st:
    if st.session_state.structures_locked:
        pass
    else:
        st.markdown(
                    """
                    <div style="
                      background-color: #e8f4fd;
                      border-left: 6px solid #2196f3;
                      padding: 15px;
                      border-radius: 8px;
                      font-family: Arial, sans-serif;
                      color: #0d47a1;
                      max-width: 800px;
                      margin: 10px 0;
                    ">
                      <strong>ℹ️ Info:</strong> Please upload at least one crystal structure file 
                      (<code>.cif</code>, <code>.poscar / .vasp / POSCAR</code>, <code>extended .xyz</code>, <code>.lmp</code>)..
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    if True:
        current_script_folder = os.getcwd()
        backup_folder = os.path.join(current_script_folder, "results_backup")
        st.info(
            f"💾 **Auto-backup**: Results (energies, lattice parameters, optimized structures) will be automatically saved to: `{backup_folder}`.\n"
            "Calculations of phonons currently works only within the GUI.")
        col1, col2, col3 = st.columns(3)

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
                         (calc_type != "NEB Calculation" and (
                                 len(st.session_state.structures) == 0 or not st.session_state.structures_locked)) or
                         (calc_type == "NEB Calculation" and (
                                 not st.session_state.get('neb_initial_structure') or
                                 not st.session_state.get('neb_final_structures'))),
            )

            # Add debug info right after the button:
            if calc_type == "NEB Calculation":
                st.write("**NEB Debug Info:**")
                st.write(
                    f"Initial structure: {'✅ Loaded' if st.session_state.get('neb_initial_structure') else '❌ Missing'}")
                st.write(f"Final structures: {len(st.session_state.get('neb_final_structures', {}))} loaded")

        if len(st.session_state.structures) > 0 and not st.session_state.structures_locked:
            st.warning("🔒 Please lock your structures before starting calculation to prevent accidental changes.")

        with col2:
            but_script = st.button(
                "📝 Generate Python Script (Will automatically create structures from the uploaded ones)",
                type="tertiary",
                disabled=len(st.session_state.structures) == 0,
            )

        with col3:
            but_local_script = st.button(
                "📂 Generate Python Script (Will use POSCAR files in the same folder where the script will be placed)",
                type="secondary",
                disabled=False,
            )

        if but_local_script:
            substitutions_for_script = None
            ga_params_for_script = None

            if calc_type == "GA Structure Optimization":
                substitutions_for_script = st.session_state.get('substitutions', {})
                ga_params_for_script = st.session_state.get('ga_params', {})

                if not substitutions_for_script:
                    st.error("❌ No substitutions configured for GA optimization. Please configure substitutions first.")
                    st.stop()

                if not ga_params_for_script:
                    st.error("❌ No GA parameters configured. Please configure GA parameters first.")
                    st.stop()

            supercell_info = None
            if (hasattr(st.session_state, 'confirmed_supercell_structure') and
                    st.session_state.confirmed_supercell_structure and
                    hasattr(st.session_state, 'supercell_multipliers')):
                supercell_info = {
                    'enabled': True,
                    'multipliers': getattr(st.session_state, 'supercell_multipliers', [1, 1, 1])
                }

            thread_count = st.session_state.get('thread_count', 4)
            mace_config = st.session_state.get('mace_config', {})
            mace_head_for_script = mace_config.get('head')
            mace_dispersion_for_script = mace_config.get('dispersion', False)
            mace_dispersion_xc_for_script = mace_config.get('dispersion_xc', 'pbe')

            custom_mace_path_for_script = custom_mace_path if is_custom_mace else None
            if is_custom_mace and not custom_mace_path:
                st.error("❌ Please provide a path to your custom MACE model")
                st.stop()

            local_script_content = generate_python_script_local_files(
                calc_type=calc_type,
                model_size=model_size,
                device=device,
                dtype=dtype,
                optimization_params=optimization_params,
                phonon_params=phonon_params,
                elastic_params=elastic_params,
                calc_formation_energy=calculate_formation_energy_flag,
                selected_model_key=selected_model,
                substitutions=substitutions_for_script,
                ga_params=ga_params_for_script,
                supercell_info=supercell_info,
                thread_count=thread_count,
                mace_head=mace_head_for_script,
                mace_dispersion=mace_dispersion_for_script,
                mace_dispersion_xc=mace_dispersion_xc_for_script,
                custom_mace_path=custom_mace_path_for_script
            )

            local_script_key = f"local_script_{hash(local_script_content) % 10000}"

            if f"copied_{local_script_key}" not in st.session_state:
                st.session_state[f"copied_{local_script_key}"] = False

            st.download_button(
                label="💾 Download Local POSCAR Script",
                data=local_script_content,
                file_name="mace_local_calculation_script.py",
                mime="text/x-python",
                help="Download the Python script that reads local POSCAR files",
                type='primary'
            )

            with st.expander("📋 Generated Local POSCAR Script", expanded=True):
                st.code(local_script_content, language='python')

                st.info("""
                        **Usage Instructions:**
                        1. Save the script as `mace_local_calculation_script.py`
                        2. Place your POSCAR files in the same directory as the script
                        3. Install required packages: `pip install mace-torch ase pymatgen numpy pandas matplotlib`
                        4. Run: `python mace_local_calculation_script.py`

                        **Supported File Names:**
                        - Files starting with "POSCAR" or ending with ".vasp"
                        - Examples: `POSCAR`, `POSCAR_1`, `structure.vasp`, etc.

                        **Output Files:**
                        - `results_summary.txt` - Main results and energies
                        - `trajectory_*.xyz` - Optimization trajectories (if applicable)
                        - `optimized_structures/` - Final optimized structures (if applicable)
                        - `phonon_data_*.json` - Phonon results (if applicable)
                        - `elastic_data_*.json` - Elastic properties (if applicable)
                        - Various plots and visualizations
                        """)

        if but_script:
            substitutions_for_script = None
            ga_params_for_script = None

            if calc_type == "GA Structure Optimization":
                substitutions_for_script = st.session_state.get('substitutions', {})
                ga_params_for_script = st.session_state.get('ga_params', {})

                if not substitutions_for_script:
                    st.error("❌ No substitutions configured for GA optimization. Please configure substitutions first.")
                    st.stop()

                if not ga_params_for_script:
                    st.error("❌ No GA parameters configured. Please configure GA parameters first.")
                    st.stop()

            supercell_info = None
            if (hasattr(st.session_state, 'confirmed_supercell_structure') and
                    st.session_state.confirmed_supercell_structure and
                    hasattr(st.session_state, 'supercell_multipliers')):
                supercell_info = {
                    'enabled': True,
                    'multipliers': getattr(st.session_state, 'supercell_multipliers', [1, 1, 1])
                }

            thread_count = st.session_state.get('thread_count', 4)

            mace_config = st.session_state.get('mace_config', {})
            mace_head_for_script = mace_config.get('head')
            mace_dispersion_for_script = mace_config.get('dispersion', False)
            mace_dispersion_xc_for_script = mace_config.get('dispersion_xc', 'pbe')

            script_content = generate_python_script(
                structures=st.session_state.structures,
                calc_type=calc_type,
                model_size=model_size,
                device=device,
                dtype=dtype,
                optimization_params=optimization_params,
                phonon_params=phonon_params,
                elastic_params=elastic_params,
                calc_formation_energy=calculate_formation_energy_flag,
                selected_model_key=selected_model,
                substitutions=substitutions_for_script,
                ga_params=ga_params_for_script,
                supercell_info=supercell_info,
                thread_count=thread_count,
                mace_head=mace_head_for_script,
                mace_dispersion=mace_dispersion_for_script,
                mace_dispersion_xc=mace_dispersion_xc_for_script
            )

            script_key = f"script_{hash(script_content) % 10000}"

            if f"copied_{script_key}" not in st.session_state:
                st.session_state[f"copied_{script_key}"] = False

            st.download_button(
                label="💾 Download Script",
                data=script_content,
                file_name="mace_calculation_script.py",
                mime="text/x-python",
                help="Download the Python script file", type='primary'
            )

            with st.expander("📋 Generated Python Script", expanded=True):
                st.code(script_content, language='python')

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

            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"results_{current_time}.txt"
            st.session_state.results_backup_file = os.path.join("results_backup", backup_filename)

            if calc_type == "GA Structure Optimization":
                if hasattr(st.session_state, 'ga_base_structure') and st.session_state.ga_base_structure:
                    first_structure_name = list(st.session_state.structures.keys())[0]
                    st.session_state.structures[first_structure_name] = st.session_state.ga_base_structure

            substitutions = st.session_state.get('substitutions', {})
            ga_params = st.session_state.get('ga_params', {})

            if calc_type == "NEB Calculation":
                structures_to_pass = {}
                neb_initial_to_pass = st.session_state.get('neb_initial_structure')
                neb_finals_to_pass = st.session_state.get('neb_final_structures', {})

                st.write(
                    f"DEBUG: Passing to thread - initial={neb_initial_to_pass is not None}, finals={len(neb_finals_to_pass)}")
            else:
                structures_to_pass = st.session_state.structures
                neb_initial_to_pass = None
                neb_finals_to_pass = None

            thread = threading.Thread(
                target=run_mace_calculation,
                args=(structures_to_pass, calc_type, model_size, device, optimization_params,
                      phonon_params, elastic_params, calculate_formation_energy_flag, st.session_state.log_queue,
                      st.session_state.stop_event, substitutions, ga_params, neb_initial_to_pass, neb_finals_to_pass, mace_head, mace_dispersion, mace_dispersion_xc)
            )
            thread.start()
            st.rerun()

    else:
        st.info("**Upload** or **Lock** structure files to begin")
with st.sidebar:
    st.info(f"**Selected Model:** {selected_model}")
    st.info(f"**Device:** {device}")


    if MACE_IMPORT_METHOD == "mace_mp":
        st.info("Using mace_mp - models downloaded automatically")
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
            elif message.get('type') == 'ga_combination_start':
                st.session_state.current_ga_combination = {
                    'combination_idx': message['combination_idx'],
                    'total_combinations': message['total_combinations'],
                    'combination_name': message['combination_name'],
                    'combination_substitutions': message['combination_substitutions']
                }
            elif message.get('type') == 'md_step':
                structure_name = message['structure']
                st.session_state.current_md_info = {
                    'structure': structure_name,
                    'step': message['step'],
                    'total_steps': message['total_steps'],
                    'progress': message['progress'],
                    'potential_energy': message['potential_energy'],
                    'kinetic_energy': message['kinetic_energy'],
                    'total_energy': message['total_energy'],
                    'temperature': message['temperature'],
                    'pressure': message.get('pressure'),
                    'avg_time_per_step': message.get('avg_time_per_step', 0),
                    'estimated_remaining_time': message.get('estimated_remaining_time'),
                    'elapsed_time': message.get('elapsed_time', 0)
                }
            elif message.get('type') == 'tensile_step':
                structure_name = message['structure']
                st.session_state.current_tensile_info = {
                    'structure': structure_name,
                    'step': message['step'],
                    'total_steps': message.get('total_steps', 0),
                    'strain_percent': message['strain_percent'],
                    'stress_GPa': message['stress_GPa'],
                    'temperature': message['temperature'],
                    'energy': message['energy'],
                    'avg_step_time': message.get('avg_step_time', 0),
                    'estimated_remaining_time': message.get('estimated_remaining_time'),
                    'elapsed_time': message.get('elapsed_time', 0)
                }
            elif message.get('type') == 'ga_progress':
                current_time = time.time()
                if current_time - st.session_state.last_ga_progress_update > 0.1:
                    st.session_state.ga_progress_info = {
                        'run_id': message['run_id'],
                        'generation': message['generation'],
                        'current_structure': message['current_structure'],
                        'total_structures': message['total_structures'],
                        'phase': message['phase']
                    }
                    st.session_state.last_ga_progress_update = current_time
            elif message.get('type') == 'reset_ga_progress':
                st.session_state.ga_progress_info = {}
                st.session_state.ga_structure_timings = []
                st.session_state.last_ga_progress_update = 0
            elif message.get('type') == 'ga_structure_timing':
                st.session_state.ga_structure_timings.append({
                    'run_id': message['run_id'],
                    'duration': message['duration'],
                    'energy': message['energy'],
                    'timestamp': time.time()
                })

                # Keep only recent timings (last 20 for averaging)
                if len(st.session_state.ga_structure_timings) > 20:
                    st.session_state.ga_structure_timings = st.session_state.ga_structure_timings[-20:]

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
                        'current_energy_change': message.get('energy_change', 0),
                        'avg_step_time': message.get('avg_step_time', 0),
                        'estimated_remaining_time': message.get('estimated_remaining_time'),
                        'total_estimated_time': message.get('total_estimated_time'),
                        'elapsed_time': message.get('elapsed_time', 0)
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
                if message.get('md_results'):
                    md_data = message['md_results']
                    if md_data['success'] and md_data.get('trajectory_data'):
                        st.session_state.md_trajectories[message['name']] = md_data
                st.session_state.results.append(message)
                if st.session_state.results_backup_file:
                    append_to_backup_file(message, st.session_state.results_backup_file)
                    backup_dir = os.path.dirname(st.session_state.results_backup_file)
                    save_optimized_structure_backup(message, backup_dir)

        elif message == "CALCULATION_FINISHED":
            st.session_state.calculation_running = False
            st.session_state.current_structure_progress = {}
            st.session_state.current_optimization_info = {}
            st.success("✅ All calculations completed!")

            st.rerun()
        else:
            st.session_state.log_messages.append(str(message))

    if st.session_state.calculation_running and calc_type == "GA Structure Optimization":
        if st.session_state.ga_progress_info:
            ga_info = st.session_state.ga_progress_info
            ga_params = st.session_state.get('ga_params', {})

            current_run = ga_info['run_id'] + 1
            total_runs = ga_params.get('num_runs', 1)
            current_generation = ga_info['generation']
            max_generations = ga_params.get('max_generations', 100)
            current_structure = ga_info['current_structure']
            total_structures = ga_info['total_structures']

            # Show current combination info if available
            combination_info = ""
            if 'current_ga_combination' in st.session_state:
                combo_info = st.session_state.current_ga_combination
                combo_idx = combo_info['combination_idx']
                total_combos = combo_info['total_combinations']
                combo_name = combo_info['combination_name']
                combination_info = f" | Combination: {combo_name} ({combo_idx + 1}/{total_combos})"

            if len(st.session_state.ga_structure_timings) >= 5:
                recent_timings = st.session_state.ga_structure_timings[-5:]
                avg_time_per_structure = np.mean([t['duration'] for t in recent_timings])

                remaining_structures_this_gen = max(0, total_structures - current_structure)
                remaining_generations = max(0, max_generations - current_generation)
                remaining_runs = max(0, total_runs - current_run)

                remaining_time_this_gen = remaining_structures_this_gen * avg_time_per_structure
                remaining_time_this_run = remaining_generations * total_structures * avg_time_per_structure
                remaining_time_other_runs = remaining_runs * max_generations * total_structures * avg_time_per_structure

                total_remaining_time = remaining_time_this_gen + remaining_time_this_run + remaining_time_other_runs


                def format_time(seconds):
                    if seconds < 60:
                        return f"{seconds:.0f}s"
                    elif seconds < 3600:
                        return f"{seconds / 60:.1f}m"
                    else:
                        return f"{seconds / 3600:.1f}h"
            else:
                avg_time_per_structure = 0
                total_remaining_time = 0

            st.markdown("### 🧬 Genetic Algorithm Progress")

            phase_text = "Initialization" if ga_info['phase'] == 'initialization' else "Evolution"
            progress_text = f"Run {current_run}/{total_runs} | Gen {current_generation}/{max_generations} | Structure {current_structure}/{total_structures}{combination_info}"

            if total_runs > 0 and max_generations > 0 and total_structures > 0:
                total_structures_overall = total_runs * max_generations * total_structures
                completed_structures = ((current_run - 1) * max_generations * total_structures +
                                        current_generation * total_structures + current_structure)
                overall_progress = min(1.0, completed_structures / total_structures_overall)
            else:
                overall_progress = 0.0

            st.progress(overall_progress, text=progress_text)

            col_ga1, col_ga2, col_ga3 = st.columns(3)

            with col_ga1:
                st.metric("Run", f"{current_run}/{total_runs}")

            with col_ga2:
                st.metric("Generation", f"{current_generation}/{max_generations}")

            with col_ga3:
                if total_remaining_time > 0:
                    st.metric("Est. Remaining", format_time(total_remaining_time))
                else:
                    st.metric("Est. Remaining", "Calculating...")

            # Show current combination details if in sweep mode
            if 'current_ga_combination' in st.session_state:
                combo_info = st.session_state.current_ga_combination
                st.info(f"🎯 Current concentration: {combo_info['combination_name']}")

                # Show substitution details
                substitution_details = []
                for elem, sub_info in combo_info['combination_substitutions'].items():
                    conc_pct = sub_info['concentration'] * 100
                    new_elem = sub_info['new_element']
                    if new_elem == 'VACANCY':
                        substitution_details.append(f"{elem}: {100 - conc_pct:.1f}% → {conc_pct:.1f}% vacant")
                    else:
                        substitution_details.append(f"{elem}: {100 - conc_pct:.1f}% → {new_elem}: {conc_pct:.1f}%")

                if substitution_details:
                    st.caption(" | ".join(substitution_details))

    elif st.session_state.calculation_running:
        if st.session_state.current_structure_progress:
            progress_data = st.session_state.current_structure_progress
            st.progress(progress_data['progress'], text=progress_data['text'])

        if st.session_state.current_optimization_info and st.session_state.current_optimization_info.get(
                'is_optimizing'):
            opt_info = st.session_state.current_optimization_info

            opt_progress = min(1.0, opt_info.get('current_step', 0) / opt_info['max_steps']) if opt_info[
                                                                                                    'max_steps'] > 0 else 0
            opt_text = f"Optimizing {opt_info['structure']}: Step {opt_info.get('current_step', 0)}/{opt_info['max_steps']}"

            if 'current_energy' in opt_info:
                opt_text += f" | Energy: {opt_info['current_energy']:.6f} eV"

            if 'current_max_force' in opt_info:
                opt_text += f" | Max Force: {opt_info['current_max_force']:.4f} eV/Å"

            if 'current_max_stress' in opt_info:
                opt_text += f" | Max Stress: {opt_info['current_max_stress']:.4f} GPa"

            if 'estimated_remaining_time' in opt_info and opt_info['estimated_remaining_time']:
                remaining_time = opt_info['estimated_remaining_time']
                if remaining_time < 60:
                    time_str = f"{remaining_time:.0f}s"
                elif remaining_time < 3600:
                    time_str = f"{remaining_time / 60:.1f}m"
                else:
                    time_str = f"{remaining_time / 3600:.1f}h"
                opt_text += f" | Est. remaining: {time_str}"

            if 'current_energy_change' in opt_info:
                opt_text += f" | ΔE: {opt_info['current_energy_change']:.2e} eV"

            st.progress(opt_progress, text=opt_text)

            if 'current_step' in opt_info and opt_info['current_step'] > 0:
                opt_type = opt_info.get('optimization_type', 'both')

                if opt_type == "Atoms only (fixed cell)":
                    col1, col2, col3, col4, col5 = st.columns(5)
                elif opt_type == "Cell only (fixed atoms)":
                    col1, col2, col3, col4, col5 = st.columns(5)
                else:
                    col1, col2, col3, col4, col5, col6 = st.columns(6)

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

                time_col = col4 if opt_type == "Atoms only (fixed cell)" else (
                    col4 if opt_type == "Cell only (fixed atoms)" else col5)
                with time_col:
                    if 'avg_step_time' in opt_info and opt_info['avg_step_time'] > 0:
                        st.metric("Avg Time/Step", f"{opt_info['avg_step_time']:.1f}s")

                remaining_time_col = col5 if opt_type in ["Atoms only (fixed cell)",
                                                          "Cell only (fixed atoms)"] else col6
                with remaining_time_col:
                    if 'estimated_remaining_time' in opt_info and opt_info['estimated_remaining_time']:
                        remaining_time = opt_info['estimated_remaining_time']
                        if remaining_time < 60:
                            time_display = f"{remaining_time:.0f}s"
                        elif remaining_time < 3600:
                            time_display = f"{remaining_time / 60:.1f}m"
                        else:
                            time_display = f"{remaining_time / 3600:.1f}h"

                        if remaining_time < 300:  # < 5 minutes
                            delta_color = "< less than, ✅ Soon"
                        elif remaining_time < 1800:  # < 30 minutes
                            delta_color = "less than, 🟡 Medium"
                        else:
                            delta_color = "less than, 🔴 Long"

                        st.metric("Est. Remaining", time_display, delta=delta_color)

                energy_col = col5 if opt_type == "Cell only (fixed atoms)" else (
                    col5 if opt_type == "Atoms only (fixed cell)" else col6)

                if opt_type == "Both atoms and cell" and 'estimated_remaining_time' in opt_info:
                    if 'current_energy_change' in opt_info:
                        st.markdown("---")
                        col_energy = st.columns(1)[0]
                        with col_energy:
                            energy_converged = opt_info['current_energy_change'] < opt_info['ediff']
                            st.metric("ΔE (eV)", f"{opt_info['current_energy_change']:.2e}",
                                      delta="✅ Converged" if energy_converged else "❌ Not converged")
                            st.info('ΔE is not convergence criterion, it is shown only for control.')
                else:
                    with energy_col:
                        if 'current_energy_change' in opt_info:
                            energy_converged = opt_info['current_energy_change'] < opt_info['ediff']
                            st.metric("ΔE (eV)", f"{opt_info['current_energy_change']:.2e}",
                                      delta="✅ Converged" if energy_converged else "❌ Not converged")

        elif calc_type == "Molecular Dynamics" and st.session_state.get('current_md_info'):
            md_info = st.session_state.current_md_info

            progress = md_info.get('progress', 0)
            progress_text = f"MD Simulation {md_info['structure']}: Step {md_info['step']}/{md_info['total_steps']}"

            st.progress(progress, text=progress_text)

            # Display MD metrics
            col_md1, col_md2, col_md3, col_md4 = st.columns(4)

            with col_md1:
                st.metric("Current Step", f"{md_info['step']}/{md_info['total_steps']}")

            with col_md2:
                st.metric("Temperature (K)", f"{md_info['temperature']:.1f}")

            with col_md3:
                st.metric("Total Energy (eV)", f"{md_info['total_energy']:.6f}")

            with col_md4:
                if md_info.get('pressure') is not None:
                    st.metric("Pressure (GPa)", f"{md_info['pressure']:.2f}")
                else:
                    st.metric("Kinetic Energy (eV)", f"{md_info['kinetic_energy']:.6f}")

            # Time estimates
            if md_info.get('estimated_remaining_time'):
                remaining_time = md_info['estimated_remaining_time']
                if remaining_time < 60:
                    time_str = f"{remaining_time:.0f}s"
                elif remaining_time < 3600:
                    time_str = f"{remaining_time / 60:.1f}m"
                else:
                    time_str = f"{remaining_time / 3600:.1f}h"
                st.info(f"Estimated remaining time: {time_str}")
        elif calc_type == "Virtual Tensile Test" and st.session_state.get('current_tensile_info'):
            tensile_info = st.session_state.current_tensile_info

            strain_percent = tensile_info['strain_percent']
            stress = tensile_info['stress_GPa']
            current_step = tensile_info['step']
            total_steps = tensile_info.get('total_steps', 0)

            if total_steps > 0:
                progress_value = min(1.0, current_step / total_steps)
                progress_text = (f"Tensile Test {tensile_info['structure']}: "
                                 f"Step {current_step:,}/{total_steps:,} | "
                                 f"Strain {strain_percent:.2f}%, Stress {stress:.2f} GPa")
            else:
                progress_value = min(1.0, strain_percent / 10.0)
                progress_text = f"Tensile Test {tensile_info['structure']}: Strain {strain_percent:.2f}%, Stress {stress:.2f} GPa"

            st.progress(progress_value, text=progress_text)

            col_t1, col_t2, col_t3, col_t4, col_t5, col_t6 = st.columns(6)

            with col_t1:
                if total_steps > 0:
                    st.metric("Step", f"{current_step:,}/{total_steps:,}")
                else:
                    st.metric("Strain", f"{strain_percent:.2f}%")

            with col_t2:
                st.metric("Strain", f"{strain_percent:.2f}%")

            with col_t3:
                st.metric("Stress", f"{stress:.2f} GPa")

            with col_t4:
                st.metric("Temperature", f"{tensile_info['temperature']:.1f} K")

            with col_t5:
                if tensile_info.get('avg_step_time', 0) > 0:
                    st.metric("Avg Time/Step", f"{tensile_info['avg_step_time']:.2f}s")

            with col_t6:
                if tensile_info.get('estimated_remaining_time'):
                    remaining = tensile_info['estimated_remaining_time']
                    if remaining < 60:
                        time_display = f"{remaining:.0f}s"
                    elif remaining < 3600:
                        time_display = f"{remaining / 60:.1f}m"
                    else:
                        time_display = f"{remaining / 3600:.1f}h"
                    st.metric("Est. Remaining", time_display)
    if st.session_state.log_messages:
        recent_messages = st.session_state.log_messages[-40:]
        st.markdown("""
            <style>
            /* Make text area scrollbar more visible */
            textarea {
                scrollbar-width: auto !important;  /* Firefox */
                scrollbar-color: #888 #f1f1f1 !important;  /* Firefox */
            }

            /* Webkit browsers (Chrome, Safari, Edge) */
            textarea::-webkit-scrollbar {
                width: 12px !important;
                height: 12px !important;
            }

            textarea::-webkit-scrollbar-track {
                background: #f1f1f1 !important;
                border-radius: 10px !important;
            }

            textarea::-webkit-scrollbar-thumb {
                background: #888 !important;
                border-radius: 10px !important;
            }

            textarea::-webkit-scrollbar-thumb:hover {
                background: #555 !important;
            }
            </style>
        """, unsafe_allow_html=True)
        st.text_area("Calculation Log", "\n".join(recent_messages), height=300)

    if has_new_messages and st.session_state.calculation_running:
        time.sleep(0.5)
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
        results_tab1, results_tab2, results_tab3, results_tab4, results_tab6, results_NEB, results_tab5, = st.tabs(["📊 Energies",
                                                                                                       "🔧 Geometry Optimization Details",
                                                                                                       "Elastic properties",
                                                                                                       "Phonons",

                                                                                                       "🧬 GA Optimization",
                                                                                                       "NEB Calculations",
                                                                                                       "⏱️ Computation times"])
    else:
        st.info("Please start some calculation first.")

    if st.session_state.results:

        with results_NEB:
            st.header("NEB Calculation Results")

            neb_results_list = [r for r in st.session_state.results if
                                r['calc_type'] == 'NEB Calculation' and r.get('neb_results')]

            if neb_results_list:
                st.subheader("Diffusion Barrier Analysis")

                all_neb_results = {r['name']: r['neb_results'] for r in neb_results_list
                                   if r['neb_results']['success']}

                coord_type = st.radio(
                    "Reaction coordinate:",
                    ["Image Index", "Structural Distance"],
                    horizontal=True,
                    help="Display as image numbers or cumulative distance along the path"
                )

                use_distance = (coord_type == "Structural Distance")

                if len(all_neb_results) > 1:
                    plot_option = st.radio(
                        "Display mode:",
                        ["Combined Plot", "Individual Plots"],
                        horizontal=True,
                        key="neb_plot_option"
                    )

                    if plot_option == "Combined Plot":
                        fig_combined = create_combined_neb_plot(all_neb_results, use_distance=use_distance)
                        st.plotly_chart(fig_combined, use_container_width=True)
                    else:
                        for name, neb_result in all_neb_results.items():
                            with st.expander(f"{name}", expanded=True):
                                display_neb_results(neb_result, name, use_distance=use_distance)
                else:
                    for result in neb_results_list:
                        if result['neb_results']['success']:
                            display_neb_results(result['neb_results'], result['name'], use_distance=use_distance)

                st.subheader("Download Options")

                col_download_all = st.columns(1)[0]

                for result in neb_results_list:
                    if result['neb_results']['success']:
                        with st.expander(f"{result['name']}"):
                            col_d1, col_d2 = st.columns(2)

                            with col_d1:
                                xyz_content = create_neb_trajectory_xyz(
                                    result['neb_results']['trajectory_data'],
                                    result['name']
                                )
                                st.download_button(
                                    label="Download NEB Trajectory (XYZ)",
                                    data=xyz_content,
                                    file_name=f"neb_{result['name'].replace('.', '_')}.xyz",
                                    mime="text/plain",
                                    key=f"neb_xyz_{result['name']}",
                                    type='primary'
                                )

                            with col_d2:
                                json_content = export_neb_results(result['neb_results'], result['name'])
                                st.download_button(
                                    label="Download NEB Data (JSON)",
                                    data=json_content,
                                    file_name=f"neb_data_{result['name'].replace('.', '_')}.json",
                                    mime="application/json",
                                    key=f"neb_json_{result['name']}",
                                    type='primary'
                                )
            else:
                st.info("No NEB results available. Results will appear after NEB calculations complete.")

                st.markdown("""
                **What you'll see here after NEB calculations:**

                - Energy profiles showing the minimum energy path
                - Forward and reverse diffusion barriers (in eV and kJ/mol)
                - Transition state location
                - Downloadable trajectory files in XYZ format
                - Detailed energy data for each image
                - Combined plots for multiple NEB calculations
                """)

        with results_tab6:
            ga_results_list = [r for r in st.session_state.results if
                               r['calc_type'] == 'GA Structure Optimization' and r.get('ga_results')]

            if ga_results_list:
                st.subheader("🧬 Genetic Algorithm Optimization Results")

                if len(ga_results_list) == 1:
                    selected_ga = ga_results_list[0]
                else:
                    ga_names = [r['name'] for r in ga_results_list]
                    selected_name = st.selectbox("Select GA run:", ga_names, key="ga_selector")
                    selected_ga = next(r for r in ga_results_list if r['name'] == selected_name)

                display_ga_results(selected_ga['ga_results'])

                if selected_ga['name'] in st.session_state.structures:
                    original_structure = st.session_state.structures[selected_ga['name']]
                    optimized_structure = selected_ga['ga_results']['best_structure']

                    st.subheader("📊 Structure Comparison")

                    col_comp1, col_comp2 = st.columns(2)

                    with col_comp1:
                        st.write("**Original Structure:**")
                        st.write(f"Composition: {original_structure.composition.reduced_formula}")
                        st.write(f"Total atoms: {len(original_structure)}")

                        orig_composition = {}
                        for site in original_structure:
                            element = site.specie.symbol
                            orig_composition[element] = orig_composition.get(element, 0) + 1

                        for element, count in orig_composition.items():
                            percentage = (count / len(original_structure)) * 100
                            st.write(f"• {element}: {count} atoms ({percentage:.1f}%)")

                    with col_comp2:
                        st.write("**Optimized Structure:**")
                        st.write(f"Composition: {optimized_structure.composition.reduced_formula}")
                        st.write(f"Total atoms: {len(optimized_structure)}")
                        st.write(f"**Energy: {selected_ga['ga_results']['best_energy']:.6f} eV**")

                        opt_composition = {}
                        for site in optimized_structure:
                            element = site.specie.symbol
                            opt_composition[element] = opt_composition.get(element, 0) + 1

                        for element, count in opt_composition.items():
                            percentage = (count / len(optimized_structure)) * 100
                            st.write(f"• {element}: {count} atoms ({percentage:.1f}%)")
            else:
                st.info("No GA optimization results found. Results will appear here after GA calculations complete.")

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
                    key=f"download_csv_{len(successful_results)}", type='primary'
                )

                optimized_structures = [r for r in successful_results if r['calc_type'] == 'Geometry Optimization']
            orb_confidence_results = [r for r in st.session_state.results if
                                      r.get('orb_confidence') and r['orb_confidence'].get('success')]

            if orb_confidence_results:
                st.subheader("🎯 ORB-v3 Confidence Analysis")

                if len(orb_confidence_results) == 1:
                    selected_conf = orb_confidence_results[0]
                else:
                    conf_names = [r['name'] for r in orb_confidence_results]
                    selected_name = st.selectbox("Select structure for confidence analysis:",
                                                 conf_names, key="conf_selector")
                    selected_conf = next(r for r in orb_confidence_results if r['name'] == selected_name)

                conf_data = selected_conf['orb_confidence']

                col_conf1, col_conf2, col_conf3, col_conf4 = st.columns(4)

                with col_conf1:
                    st.metric("Confidence Score", f"{conf_data['confidence_score']:.4f}")
                with col_conf2:
                    st.metric("Mean Predicted MAE", f"{conf_data['mean_predicted_mae']:.4f} Å")
                with col_conf3:
                    st.metric("Max Predicted MAE", f"{conf_data['max_predicted_mae']:.4f} Å")
                with col_conf4:
                    n_uncertain = len(conf_data['high_uncertainty_atoms'])
                    total_atoms = len(conf_data['per_atom_predicted_mae'])
                    st.metric("High Uncertainty Atoms", f"{n_uncertain}/{total_atoms}")


                fig_uncertainty = go.Figure()
                fig_uncertainty.add_trace(go.Bar(
                    y=conf_data['per_atom_predicted_mae'],
                    marker_color=['red' if mae > 0.1 else 'orange' if mae > 0.05 else 'green'
                                  for mae in conf_data['per_atom_predicted_mae']],
                    marker_line_color='black',
                    marker_line_width=1.5,
                    name='Predicted Force MAE',
                    hovertemplate='<b>Atom %{x}</b><br>Predicted MAE: %{y:.4f} Å<extra></extra>'
                ))

                fig_uncertainty.add_hline(
                    y=0.1,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text="High Uncertainty Threshold (0.1 Å)",
                    annotation_font_size=16
                )

                fig_uncertainty.update_layout(
                    title=dict(text="Per-Atom Predicted Force Error (MAE)", font=dict(size=26)),  # INCREASED from 24
                    xaxis_title="Atom Index",
                    yaxis_title="Predicted MAE (Å)",
                    height=600,
                    font=dict(size=20),
                    xaxis=dict(
                        tickfont=dict(size=18),
                        title_font=dict(size=22)
                    ),
                    yaxis=dict(
                        tickfont=dict(size=18),
                        title_font=dict(size=22)
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=20,
                        font_family="Arial",
                        bordercolor="black"
                    )
                )
                st.plotly_chart(fig_uncertainty, use_container_width=True)

                confidence = conf_data['confidence_score']
                if confidence > 0.9:
                    st.success("✅ High confidence - predictions are very reliable")
                elif confidence > 0.7:
                    st.info("ℹ️ Good confidence - predictions generally reliable")
                elif confidence > 0.5:
                    st.warning("⚠️ Moderate confidence - use with caution")
                else:
                    st.error("❌ Low confidence - predictions may be unreliable")

                if conf_data['high_uncertainty_atoms']:
                    with st.expander(f"⚠️ High Uncertainty Atoms ({len(conf_data['high_uncertainty_atoms'])} atoms)"):
                        uncertain_data = []
                        for idx in conf_data['high_uncertainty_atoms']:
                            uncertain_data.append({
                                'Atom Index': idx,
                                'Predicted MAE (Å)': f"{conf_data['per_atom_predicted_mae'][idx]:.4f}"
                            })
                        df_uncertain = pd.DataFrame(uncertain_data)
                        st.dataframe(df_uncertain, use_container_width=True, hide_index=True)
                st.markdown("---")

                with st.expander("📚 Understanding ORB-v3 Confidence Predictions", expanded=False):
                    st.markdown("""
                            ### What is the Confidence Head?

                            ORB-v3 models include a **trained confidence classifier** that predicts how accurate the force 
                            predictions are likely to be. This classifier was trained to predict the **Mean Absolute Error (MAE)** 
                            between predicted and true forces on the training data.

                            ### How It Works

                            - The confidence head outputs **50 bins** representing force error magnitudes from **0 to 0.4 Å**
                            - Each atom gets a predicted MAE bin based on how uncertain the model is about that atom's forces
                            - The classifier learned these patterns during training by comparing its force predictions to DFT reference data

                            ### What is MAE?

                            **Mean Absolute Error (MAE)** measures the average magnitude of errors in predictions:
                            - It calculates the absolute difference between predicted and actual force values
                            - Lower MAE = more accurate predictions
                            - MAE is expressed in the same units as forces (eV/Å)

                            **Example:** If the predicted force on an atom is 0.15 eV/Å but the true force is 0.20 eV/Å, 
                            the absolute error is 0.05 eV/Å.

                            ### Interpretation Guidelines

                            The **Predicted MAE per atom** tells you how much error to expect in the force predictions:

                            | Predicted MAE | Confidence Level | Interpretation |
                            |---------------|------------------|----------------|
                            | **< 0.05 Å** | Very High (Green) | Forces are highly reliable, suitable for all applications |
                            | **0.05 - 0.1 Å** | Good (Yellow) | Forces are generally reliable for most calculations |
                            | **0.1 - 0.2 Å** | Moderate (Orange) | Use with caution, verify important results |
                            | **> 0.2 Å** | Low (Red) | High uncertainty, predictions may be unreliable |

                            ### Confidence Score

                            The **overall confidence score (0-1)** is derived from the mean predicted MAE:
                            - **> 0.9**: Very high confidence - predictions are trustworthy
                            - **0.7 - 0.9**: Good confidence - predictions generally reliable
                            - **0.5 - 0.7**: Moderate confidence - consider validation
                            - **< 0.5**: Low confidence - results should be verified

                            ### When to Pay Attention

                            High uncertainty (high predicted MAE) often occurs when:
                            - Atoms are in unusual chemical environments
                            - The structure is outside the model's training distribution
                            - There are strong electronic effects or complex bonding
                            - The system has defects, surfaces, or interfaces

                            ### Practical Use

                            Use confidence predictions to:
                            - **Identify problematic atoms** that may need special attention
                            - **Validate results** by checking high-uncertainty regions with higher-level calculations
                            - **Guide active learning** by selecting uncertain structures for additional DFT calculations
                            - **Filter predictions** by rejecting low-confidence results in high-stakes applications

                            ### Reference

                            For more details, see the [ORB models documentation](https://github.com/orbital-materials/orb-models) 
                            and the paper: *"Orb-v3: atomistic simulation at scale"* (arXiv:2504.06231)
                            """)
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

                    if phonon_data.get('enhanced_kpoints') and 'kpoint_distances' in phonon_data:
                        x_axis = phonon_data['kpoint_distances']
                        x_title = "Distance along k-path"
                        use_labels = True
                    else:
                        x_axis = list(range(nkpts))
                        x_title = "k-point index"
                        use_labels = False

                    fig_disp = go.Figure()

                    for band in range(nbands):
                        fig_disp.add_trace(go.Scatter(
                            x=x_axis,
                            y=frequencies[:, band],
                            mode='lines',
                            name=f'Branch {band + 1}',
                            line=dict(width=1.5),
                            showlegend=False,
                            hovertemplate='Frequency: %{y:.3f} meV<extra></extra>'
                        ))

                    if phonon_data['imaginary_modes'] > 0:
                        imaginary_mask = frequencies < 0
                        for band in range(nbands):
                            imaginary_points = np.where(imaginary_mask[:, band])[0]
                            if len(imaginary_points) > 0:
                                fig_disp.add_trace(go.Scatter(
                                    x=[x_axis[i] for i in imaginary_points],
                                    y=frequencies[imaginary_points, band],
                                    mode='markers',
                                    marker=dict(color='red', size=4),
                                    name='Imaginary modes',
                                    showlegend=band == 0,
                                    hovertemplate='Imaginary mode: %{y:.3f} meV<extra></extra>'
                                ))

                    if use_labels and 'kpoint_labels' in phonon_data and 'kpoint_label_positions' in phonon_data:
                        labels = phonon_data['kpoint_labels']
                        positions = phonon_data['kpoint_label_positions']

                        display_labels = []
                        for label in labels:
                            if label.upper() == 'GAMMA':
                                display_labels.append('Γ')
                            else:
                                display_labels.append(label)

                        for pos in positions:
                            fig_disp.add_vline(
                                x=pos,
                                line_dash="dash",
                                line_color="gray",
                                opacity=0.7,
                                line_width=1
                            )

                        fig_disp.update_layout(
                            xaxis=dict(
                                tickmode='array',
                                tickvals=positions,
                                ticktext=display_labels,
                                title=x_title,
                                title_font=dict(size=20),
                                tickfont=dict(size=18)
                            )
                        )
                    else:
                        fig_disp.update_layout(
                            xaxis=dict(
                                title=x_title,
                                title_font=dict(size=20),
                                tickfont=dict(size=18)
                            )
                        )

                    fig_disp.update_layout(
                        title=dict(text="Phonon Dispersion", font=dict(size=24)),
                        yaxis_title="Frequency (meV)",
                        height=750,
                        font=dict(size=20),
                        hoverlabel=dict(
                            bgcolor="white",
                            bordercolor="black",
                            font_size=20,
                            font_family="Arial"
                        ),
                        yaxis=dict(
                            title_font=dict(size=20),
                            tickfont=dict(size=20)
                        ),
                        hovermode='closest'
                    )

                    fig_disp.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

                    st.plotly_chart(fig_disp, use_container_width=True)

                    if use_labels and 'kpoint_labels' in phonon_data:
                        st.info(f"🎯 **Enhanced k-point sampling**: {len(x_axis)} points along path")
                        st.info(f"🗺️ **High-symmetry path**: {' → '.join(phonon_data['kpoint_labels'])}")

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
                        mime="application/json", type='primary'
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
                                        type='primary'
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
                                if st.button("🔬 Generate Phase Diagram (", type="primary", key="generate_phase_diagram",
                                             disabled=True):
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
                                            # 'Avg Entropy (eV/K)': f"{struct_data['entropy'].mean():.6f}",
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
                else:
                    st.error("❌ Crystal may be mechanically unstable")
                    st.warning("Check the elastic tensor eigenvalues and Born stability criteria")
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
                        mime="application/json", type='primary'
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

                    # Get all unique elements across all structures
                    all_elements = set()
                    for result in geometry_results:
                        initial_structure = st.session_state.structures.get(result['name'])
                        final_structure = result['structure']

                        if initial_structure:
                            for site in initial_structure:
                                all_elements.add(site.specie.symbol)
                        if final_structure:
                            for site in final_structure:
                                all_elements.add(site.specie.symbol)

                    all_elements = sorted(list(all_elements))  # Sort alphabetically

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

                                # Calculate element concentrations for final structure
                                final_element_counts = {}
                                total_atoms = len(final_structure)

                                for site in final_structure:
                                    element = site.specie.symbol
                                    final_element_counts[element] = final_element_counts.get(element, 0) + 1

                                # Create the basic lattice data row
                                row_data = {
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
                                }

                                # Add element concentration columns
                                for element in all_elements:
                                    count = final_element_counts.get(element, 0)
                                    concentration = (count / total_atoms) * 100 if total_atoms > 0 else 0
                                    row_data[f'{element} (%)'] = f"{concentration:.1f}"

                                lattice_data.append(row_data)

                        except Exception as e:
                            st.warning(f"Could not process lattice data for {result['name']}: {str(e)}")

                    if lattice_data:
                        df_lattice = pd.DataFrame(lattice_data)

                        st.dataframe(df_lattice, use_container_width=True, hide_index=True)


                        lattice_csv = df_lattice.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Lattice Parameters with Concentrations (CSV)",
                            data=lattice_csv,
                            file_name="lattice_parameters_with_concentrations.csv",
                            mime="text/csv",
                            type='primary'
                        )

                    if len(lattice_data) >= 1:
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
                            with st.expander(f"📁 {result['name']} - {result.get('convergence_status', 'Unknown')}",
                                             expanded=False):
                                convergence_status = result.get('convergence_status', '')
                                convergence_icon = "✅" if convergence_status and "CONVERGED" in convergence_status else "⚠️"

                                col_info, col_options = st.columns([1, 2])

                                with col_info:
                                    st.write(
                                        f"{convergence_icon} **Status:** {result.get('convergence_status', 'Unknown')}")
                                    if 'structure' in result and result['structure']:
                                        st.write(
                                            f"**Formula:** {result['structure'].composition.reduced_formula}")
                                        st.write(f"**Atoms:** {len(result['structure'])}")

                                with col_options:
                                    if 'structure' in result and result['structure']:
                                        # Format selector
                                        output_format = st.selectbox(
                                            "Output format:",
                                            ["POSCAR", "CIF", "LAMMPS", "XYZ"],
                                            key=f"format_{result['name']}",
                                            index=0
                                        )

                                        # Show format-specific options
                                        if output_format == "POSCAR":
                                            use_fractional = st.checkbox(
                                                "Fractional coordinates",
                                                value=True,
                                                key=f"poscar_frac_{result['name']}"
                                            )
                                            use_selective = st.checkbox(
                                                "Selective dynamics (all free)",
                                                value=False,
                                                key=f"poscar_sel_{result['name']}"
                                            )

                                        elif output_format == "LAMMPS":
                                            lmp_style = st.selectbox(
                                                "Atom style:",
                                                ["atomic", "charge", "full"],
                                                index=0,
                                                key=f"lmp_style_{result['name']}"
                                            )
                                            lmp_units = st.selectbox(
                                                "Units:",
                                                ["metal", "real", "si"],
                                                index=0,
                                                key=f"lmp_units_{result['name']}"
                                            )
                                            lmp_masses = st.checkbox(
                                                "Include masses",
                                                value=True,
                                                key=f"lmp_masses_{result['name']}"
                                            )
                                            lmp_skew = st.checkbox(
                                                "Force triclinic",
                                                value=False,
                                                key=f"lmp_skew_{result['name']}"
                                            )

                                        elif output_format == "CIF":
                                            cif_symprec = st.number_input(
                                                "Symmetry precision:",
                                                value=0.1,
                                                min_value=0.001,
                                                max_value=1.0,
                                                step=0.001,
                                                format="%.3f",
                                                key=f"cif_symprec_{result['name']}"
                                            )

                                        # Generate button
                                        if st.button(f"📥 Download {output_format}",
                                                     key=f"download_btn_{result['name']}_{output_format}",
                                                     type="primary"):
                                            try:
                                                base_name = result['name'].split('.')[0]
                                                structure = result['structure']

                                                if output_format == "POSCAR":
                                                    from pymatgen.io.ase import AseAtomsAdaptor
                                                    from ase.io import write
                                                    from ase.constraints import FixAtoms
                                                    from io import StringIO

                                                    # Convert to ASE
                                                    new_struct = Structure(structure.lattice, [], [])
                                                    for site in structure:
                                                        new_struct.append(
                                                            species=site.species,
                                                            coords=site.frac_coords,
                                                            coords_are_cartesian=False,
                                                        )

                                                    ase_structure = AseAtomsAdaptor.get_atoms(
                                                        new_struct)

                                                    if use_selective:
                                                        constraint = FixAtoms(
                                                            indices=[])  # All atoms free
                                                        ase_structure.set_constraint(constraint)

                                                    out = StringIO()
                                                    write(out, ase_structure, format="vasp",
                                                          direct=use_fractional, sort=True)
                                                    file_content = out.getvalue()
                                                    file_extension = ".vasp"
                                                    mime_type = "text/plain"

                                                elif output_format == "CIF":
                                                    from pymatgen.io.cif import CifWriter

                                                    new_struct = Structure(structure.lattice, [], [])
                                                    for site in structure:
                                                        species_dict = {}
                                                        for element, occupancy in site.species.items():
                                                            species_dict[element] = float(occupancy)
                                                        new_struct.append(
                                                            species=species_dict,
                                                            coords=site.frac_coords,
                                                            coords_are_cartesian=False,
                                                        )
                                                    file_content = CifWriter(
                                                        new_struct, symprec=cif_symprec,
                                                        write_site_properties=True).__str__()
                                                    file_extension = ".cif"
                                                    mime_type = "chemical/x-cif"

                                                elif output_format == "LAMMPS":
                                                    from pymatgen.io.ase import AseAtomsAdaptor
                                                    from ase.io import write
                                                    from io import StringIO

                                                    new_struct = Structure(structure.lattice, [], [])
                                                    for site in structure:
                                                        new_struct.append(
                                                            species=site.species,
                                                            coords=site.frac_coords,
                                                            coords_are_cartesian=False,
                                                        )

                                                    ase_structure = AseAtomsAdaptor.get_atoms(
                                                        new_struct)
                                                    out = StringIO()
                                                    write(
                                                        out, ase_structure, format="lammps-data",
                                                        atom_style=lmp_style, units=lmp_units,
                                                        masses=lmp_masses, force_skew=lmp_skew
                                                    )
                                                    file_content = out.getvalue()
                                                    file_extension = ".lmp"
                                                    mime_type = "text/plain"

                                                elif output_format == "XYZ":
                                                    xyz_lines = []
                                                    xyz_lines.append(str(len(structure)))

                                                    lattice_matrix = structure.lattice.matrix
                                                    lattice_string = " ".join(
                                                        [f"{x:.6f}" for row in lattice_matrix for x in row])

                                                    comment_line = f'Lattice="{lattice_string}" Properties=species:S:1:pos:R:3'
                                                    xyz_lines.append(comment_line)

                                                    for site in structure:
                                                        if site.is_ordered:
                                                            element = site.specie.symbol
                                                        else:
                                                            element = max(site.species.items(), key=lambda x: x[1])[
                                                                0].symbol

                                                        cart_coords = structure.lattice.get_cartesian_coords(
                                                            site.frac_coords)
                                                        xyz_lines.append(
                                                            f"{element} {cart_coords[0]:.6f} {cart_coords[1]:.6f} {cart_coords[2]:.6f}")

                                                    file_content = "\n".join(xyz_lines)
                                                    file_extension = ".xyz"
                                                    mime_type = "chemical/x-xyz"
                                                filename = f"optimized_{base_name}{file_extension}"
                                                st.download_button(
                                                    label=f"📥 Download {output_format} File",
                                                    data=file_content,
                                                    file_name=filename,
                                                    mime=mime_type,
                                                    key=f"final_download_{result['name']}_{output_format}",
                                                    type="secondary"
                                                )

                                            except Exception as e:
                                                st.error(f"Error generating {output_format}: {str(e)}")

                    with col_download2:
                        if len(geometry_results) > 1:
                            st.write("**Bulk Download Options:**")

                            bulk_formats = st.multiselect(
                                "Select formats:",
                                ["POSCAR", "CIF", "LAMMPS", "XYZ"],
                                default=["POSCAR"],
                                key="bulk_format_selector"
                            )

                            if "POSCAR" in bulk_formats:
                                st.write("**VASP POSCAR Options:**")
                                bulk_vasp_fractional = st.checkbox("Fractional coordinates", value=True,
                                                                   key="bulk_vasp_frac")
                                bulk_vasp_selective = st.checkbox("Selective dynamics (all free)", value=False,
                                                                  key="bulk_vasp_sel")

                            if "LAMMPS" in bulk_formats:
                                st.write("**LAMMPS Options:**")
                                bulk_lmp_style = st.selectbox("Atom style:", ["atomic", "charge", "full"], index=0,
                                                              key="bulk_lmp_style")
                                bulk_lmp_units = st.selectbox("Units:", ["metal", "real", "si"], index=0,
                                                              key="bulk_lmp_units")
                                bulk_lmp_masses = st.checkbox("Include masses", value=True, key="bulk_lmp_masses")
                                bulk_lmp_skew = st.checkbox("Force triclinic", value=False, key="bulk_lmp_skew")

                            if bulk_formats and st.button("📦 Generate ZIP", type="primary", key="generate_bulk_zip"):
                                try:
                                    zip_buffer = io.BytesIO()
                                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                                        for result in geometry_results:
                                            if 'structure' in result and result['structure']:
                                                base_name = result['name'].split('.')[0]
                                                structure = result['structure']

                                                for fmt in bulk_formats:
                                                    try:
                                                        if fmt == "POSCAR":
                                                            from pymatgen.io.ase import AseAtomsAdaptor
                                                            from ase.io import write
                                                            from ase.constraints import FixAtoms
                                                            from io import StringIO

                                                            # Convert to ASE
                                                            new_struct = Structure(structure.lattice, [], [])
                                                            for site in structure:
                                                                new_struct.append(
                                                                    species=site.species,
                                                                    coords=site.frac_coords,
                                                                    coords_are_cartesian=False,
                                                                )

                                                            ase_structure = AseAtomsAdaptor.get_atoms(new_struct)

                                                            if bulk_vasp_selective:
                                                                constraint = FixAtoms(indices=[])  # All atoms free
                                                                ase_structure.set_constraint(constraint)

                                                            out = StringIO()
                                                            write(out, ase_structure, format="vasp",
                                                                  direct=bulk_vasp_fractional, sort=True)
                                                            file_content = out.getvalue()
                                                            filename = f"POSCAR/{base_name}_POSCAR.vasp"

                                                        elif fmt == "CIF":
                                                            from pymatgen.io.cif import CifWriter

                                                            new_struct = Structure(structure.lattice, [], [])
                                                            for site in structure:
                                                                species_dict = {}
                                                                for element, occupancy in site.species.items():
                                                                    species_dict[element] = float(occupancy)
                                                                new_struct.append(
                                                                    species=species_dict,
                                                                    coords=site.frac_coords,
                                                                    coords_are_cartesian=False,
                                                                )
                                                            file_content = CifWriter(new_struct, symprec=0.1,
                                                                                     write_site_properties=True).__str__()
                                                            filename = f"CIF/{base_name}.cif"

                                                        elif fmt == "LAMMPS":
                                                            from pymatgen.io.ase import AseAtomsAdaptor
                                                            from ase.io import write
                                                            from io import StringIO

                                                            new_struct = Structure(structure.lattice, [], [])
                                                            for site in structure:
                                                                new_struct.append(
                                                                    species=site.species,
                                                                    coords=site.frac_coords,
                                                                    coords_are_cartesian=False,
                                                                )

                                                            ase_structure = AseAtomsAdaptor.get_atoms(new_struct)
                                                            out = StringIO()
                                                            write(
                                                                out, ase_structure, format="lammps-data",
                                                                atom_style=bulk_lmp_style, units=bulk_lmp_units,
                                                                masses=bulk_lmp_masses, force_skew=bulk_lmp_skew
                                                            )
                                                            file_content = out.getvalue()
                                                            filename = f"LAMMPS/{base_name}.lmp"

                                                        elif fmt == "XYZ":
                                                            xyz_lines = []
                                                            xyz_lines.append(str(len(structure)))

                                                            lattice_matrix = structure.lattice.matrix
                                                            lattice_string = " ".join(
                                                                [f"{x:.6f}" for row in lattice_matrix for x in row])

                                                            comment_line = f'Lattice="{lattice_string}" Properties=species:S:1:pos:R:3'
                                                            xyz_lines.append(comment_line)

                                                            for site in structure:
                                                                if site.is_ordered:
                                                                    element = site.specie.symbol
                                                                else:
                                                                    element = \
                                                                    max(site.species.items(), key=lambda x: x[1])[
                                                                        0].symbol

                                                                cart_coords = structure.lattice.get_cartesian_coords(
                                                                    site.frac_coords)
                                                                xyz_lines.append(
                                                                    f"{element} {cart_coords[0]:.6f} {cart_coords[1]:.6f} {cart_coords[2]:.6f}")

                                                            file_content = "\n".join(xyz_lines)
                                                            filename = f"XYZ/{base_name}.xyz"

                                                        zip_file.writestr(filename, file_content)

                                                    except Exception as e:
                                                        st.warning(f"Failed to convert {base_name} to {fmt}: {str(e)}")
                                                        continue

                                        # Add README
                                        readme_content = f"""Optimized Structures Package
                    =============================

                    This package contains {len(geometry_results)} optimized structures in the following formats:
                    {', '.join(bulk_formats)}

                    Structure Information:
                    """
                                        for result in geometry_results:
                                            readme_content += f"- {result['name']}: {result.get('convergence_status', 'Unknown status')}\n"

                                        readme_content += f"""
                    Generation Settings:
                    - VASP POSCAR: {'Fractional' if bulk_vasp_fractional else 'Cartesian'} coordinates"""
                                        if "POSCAR" in bulk_formats and bulk_vasp_selective:
                                            readme_content += ", Selective dynamics (all free)"
                                        if "LAMMPS" in bulk_formats:
                                            readme_content += f"""
                    - LAMMPS: {bulk_lmp_style} style, {bulk_lmp_units} units"""
                                            if bulk_lmp_masses:
                                                readme_content += ", with masses"
                                            if bulk_lmp_skew:
                                                readme_content += ", triclinic forced"

                                        readme_content += f"""

                    Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """

                                        zip_file.writestr("README.txt", readme_content)

                                    st.download_button(
                                        label=f"📦 Download All ({', '.join(bulk_formats)})",
                                        data=zip_buffer.getvalue(),
                                        file_name="optimized_structures.zip",
                                        mime="application/zip",
                                        type='primary',
                                        key="download_bulk_zip_final"
                                    )

                                    st.success(f"✅ ZIP package created with {len(bulk_formats)} format(s)")

                                except Exception as e:
                                    st.error(f"Error creating ZIP package: {str(e)}")

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
                            # 'Entropy (eV/K)': f"{thermo['entropy']:.6f}"
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
                            help="Extended XYZ format with lattice parameters, energies, and forces", type='primary'
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
                    help="ZIP file containing all optimization trajectories in XYZ format", type='primary'
                )

        # with st.expander("📖 XYZ File Format Information"):
        #    st.markdown("""
        #    **Extended XYZ Format with Lattice Parameters:**#
        #
        #    Each XYZ file contains all optimization steps with:
        #    - **Line 1**: Number of atoms
        #    - **Line 2**: Comment line with:
        #      - Step number
        #      - Energy (eV)
        #      - Maximum force (eV/Å)
        #      - Lattice parameters (a, b, c, α, β, γ)
        #      - Properties specification
        #    - **Lines 3+**: Atomic coordinates and forces
        #      - Format: `Symbol X Y Z Fx Fy Fz`
        #
        #    **Example:**
        #    ```
        #    8
        #    Step 1 | Energy=-123.456789 eV | Max_Force=0.1234 eV/A | Lattice="5.123456 5.123456 5.123456 90.00 90.00 90.00" | Properties=species:S:1:pos:R:3:forces:R:3
        #    Ti  0.000000  0.000000  0.000000  0.001234 -0.002345  0.000123
        #    O   1.234567  1.234567  1.234567 -0.001234  0.002345 -0.000123
        #    ...
        #    ```
        #
        #    This format can be read by visualization software like OVITO, VMD, or ASE for trajectory analysis.
        #    """)

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
