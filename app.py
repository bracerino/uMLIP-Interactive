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

import py3Dmol
import streamlit.components.v1 as components
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from ase.io import read, write
from ase import Atoms
from ase.optimize import BFGS, LBFGS

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
#st.markdown(css, unsafe_allow_html=True)


def estimate_phonon_supercell(atoms, target_min_length=15.0, max_supercell=4, log_queue=None):
    """
    Estimate appropriate supercell size for phonon calculations

    Parameters:
    atoms: ASE Atoms object
    target_min_length: minimum supercell length in Angstroms (default: 15.0)
    max_supercell: maximum supercell multiplier (default: 4)
    log_queue: logging queue (optional)

    Returns:
    tuple: (nx, ny, nz) supercell multipliers
    """
    cell = atoms.get_cell()
    cell_lengths = np.linalg.norm(cell, axis=1)  # |a|, |b|, |c|

    if log_queue:
        log_queue.put(
            f"  Unit cell lengths: a={cell_lengths[0]:.3f} Å, b={cell_lengths[1]:.3f} Å, c={cell_lengths[2]:.3f} Å")

    # Calculate supercell multipliers to achieve target minimum length
    supercell_multipliers = []
    for length in cell_lengths:
        multiplier = max(1, int(np.ceil(target_min_length / length)))
        multiplier = min(multiplier, max_supercell)  # Cap at max_supercell
        supercell_multipliers.append(multiplier)

    # For very small cells or very anisotropic cells, use conservative approach
    num_atoms = len(atoms)

    # If unit cell is very small (< 5 atoms), use larger supercells
    if num_atoms < 5:
        supercell_multipliers = [min(max_supercell, max(2, m)) for m in supercell_multipliers]

    # If unit cell is very large (> 50 atoms), use smaller supercells
    elif num_atoms > 50:
        supercell_multipliers = [max(1, min(2, m)) for m in supercell_multipliers]

    # Ensure all multipliers are at least 1
    supercell_multipliers = [max(1, m) for m in supercell_multipliers]

    supercell_size = tuple(supercell_multipliers)
    total_atoms_in_supercell = num_atoms * np.prod(supercell_size)

    if log_queue:
        log_queue.put(f"  Estimated supercell: {supercell_size}")
        log_queue.put(
            f"  Supercell lengths: {cell_lengths[0] * supercell_size[0]:.1f} × {cell_lengths[1] * supercell_size[1]:.1f} × {cell_lengths[2] * supercell_size[2]:.1f} Å")
        log_queue.put(f"  Total atoms in supercell: {total_atoms_in_supercell}")

        # Warn if supercell is very large
        if total_atoms_in_supercell > 500:
            log_queue.put(f"  ⚠️ Warning: Large supercell ({total_atoms_in_supercell} atoms) - calculation may be slow")

    return supercell_size





def calculate_elastic_properties(atoms, calculator, elastic_params, log_queue, structure_name):
    """
    Calculate elastic tensor and derived elastic moduli

    Parameters:
    atoms: ASE Atoms object
    calculator: MACE calculator
    elastic_params: dict with elastic calculation parameters
    log_queue: queue for logging messages
    structure_name: name of the structure

    Returns:
    dict: elastic tensor, moduli, and mechanical properties
    """
    try:
        log_queue.put(f"Starting elastic tensor calculation for {structure_name}")

        # Set calculator
        atoms.calc = calculator

        # Get equilibrium energy and stress
        log_queue.put("  Calculating equilibrium energy and stress...")
        E0 = atoms.get_potential_energy()
        stress0 = atoms.get_stress(voigt=True)  # Voigt notation: [xx, yy, zz, yz, xz, xy]

        log_queue.put(f"  Equilibrium energy: {E0:.6f} eV")
        log_queue.put(f"  Equilibrium stress: {np.max(np.abs(stress0)):.6f} GPa")

        # Parameters for strain calculations
        strain_magnitude = elastic_params.get('strain_magnitude', 0.01)  # 1% strain
        log_queue.put(f"  Using strain magnitude: {strain_magnitude * 100:.1f}%")

        # Initialize elastic tensor (6x6 in Voigt notation)
        C = np.zeros((6, 6))

        # Apply strains and calculate stress response
        log_queue.put("  Applying strains and calculating stress response...")

        original_cell = atoms.get_cell().copy()
        volume = atoms.get_volume()

        # Define strain tensors (symmetric)
        strain_tensors = []

        # Diagonal strains (normal strains)
        for i in range(3):
            strain = np.zeros((3, 3))
            strain[i, i] = strain_magnitude
            strain_tensors.append(strain)

        # Shear strains
        shear_pairs = [(1, 2), (0, 2), (0, 1)]  # (yz, xz, xy)
        for i, j in shear_pairs:
            strain = np.zeros((3, 3))
            strain[i, j] = strain[j, i] = strain_magnitude / 2  # Engineering shear strain
            strain_tensors.append(strain)

        for strain_idx, strain_tensor in enumerate(strain_tensors):
            log_queue.put(f"    Strain {strain_idx + 1}/6...")

            # Apply positive strain
            deformed_cell = original_cell @ (np.eye(3) + strain_tensor)
            atoms.set_cell(deformed_cell, scale_atoms=True)
            stress_pos = atoms.get_stress(voigt=True)

            # Apply negative strain
            deformed_cell = original_cell @ (np.eye(3) - strain_tensor)
            atoms.set_cell(deformed_cell, scale_atoms=True)
            stress_neg = atoms.get_stress(voigt=True)

            # Calculate elastic constants using finite differences
            # C_ij = d(stress_i)/d(strain_j)
            stress_derivative = (stress_pos - stress_neg) / (2 * strain_magnitude)

            # Store in elastic tensor
            C[strain_idx, :] = stress_derivative

            # Restore original cell
            atoms.set_cell(original_cell, scale_atoms=True)

        # Convert from eV/Å³ to GPa
        eV_to_GPa = 160.2176  # Conversion factor
        C_GPa = C * eV_to_GPa

        log_queue.put("  Calculating elastic moduli...")

        # Calculate bulk modulus (Voigt average)
        K_voigt = (C_GPa[0, 0] + C_GPa[1, 1] + C_GPa[2, 2] + 2 * (C_GPa[0, 1] + C_GPa[0, 2] + C_GPa[1, 2])) / 9

        # Calculate shear modulus (Voigt average)
        G_voigt = (C_GPa[0, 0] + C_GPa[1, 1] + C_GPa[2, 2] - C_GPa[0, 1] - C_GPa[0, 2] - C_GPa[1, 2] + 3 * (
                    C_GPa[3, 3] + C_GPa[4, 4] + C_GPa[5, 5])) / 15

        # Calculate Reuss averages (requires matrix inversion)
        try:
            S_GPa = np.linalg.inv(C_GPa)  # Compliance matrix

            # Bulk modulus (Reuss)
            K_reuss = 1 / (S_GPa[0, 0] + S_GPa[1, 1] + S_GPa[2, 2] + 2 * (S_GPa[0, 1] + S_GPa[0, 2] + S_GPa[1, 2]))

            # Shear modulus (Reuss)
            G_reuss = 15 / (4 * (S_GPa[0, 0] + S_GPa[1, 1] + S_GPa[2, 2]) - 4 * (
                        S_GPa[0, 1] + S_GPa[0, 2] + S_GPa[1, 2]) + 3 * (S_GPa[3, 3] + S_GPa[4, 4] + S_GPa[5, 5]))

            # Hill averages (arithmetic mean of Voigt and Reuss)
            K_hill = (K_voigt + K_reuss) / 2
            G_hill = (G_voigt + G_reuss) / 2

        except np.linalg.LinAlgError:
            log_queue.put("  ⚠️ Warning: Elastic tensor is singular - using Voigt averages only")
            K_reuss = G_reuss = K_hill = G_hill = None
            S_GPa = None # Ensure S_GPa is None if inversion fails

        # Calculate derived properties using Hill averages (or Voigt if Hill unavailable)
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
            density = (total_mass_amu * 1.66053906660) / volume # amu / A^3 to g/cm^3
            log_queue.put(f"  Estimated density from structure: {density:.3f} g/cm³")
        else:
            log_queue.put(f"  Using user-provided density: {density:.3f} g/cm³")


        # Convert density to kg/m³
        density_kg_m3 = density * 1000

        # Longitudinal and transverse wave velocities
        v_l = np.sqrt((K + 4 * G / 3) * 1e9 / density_kg_m3)  # m/s
        v_t = np.sqrt(G * 1e9 / density_kg_m3)  # m/s

        # Average wave velocity
        v_avg = ((1 / v_l ** 3 + 2 / v_t ** 3) / 3) ** (-1 / 3)

        # Debye temperature (approximate)
        h = 6.626e-34  # J⋅s
        kB = 1.381e-23  # J/K
        N_atoms = len(atoms)
        # sum(atoms.get_masses()) is total mass in amu. Convert to kg for consistency.
        total_mass_kg = np.sum(atoms.get_masses()) * 1.66054e-27 # amu to kg
        theta_D = (h / kB) * v_avg * (3 * N_atoms * density_kg_m3 / (4 * np.pi * total_mass_kg)) ** (1 / 3)


        # Mechanical stability criteria
        stability_criteria = check_mechanical_stability(C_GPa, log_queue)

        log_queue.put(f"✅ Elastic calculation completed for {structure_name}")
        log_queue.put(f"  Bulk modulus: {K:.1f} GPa")
        log_queue.put(f"  Shear modulus: {G:.1f} GPa")
        log_queue.put(f"  Young's modulus: {E:.1f} GPa")
        log_queue.put(f"  Poisson's ratio: {nu:.3f}")

        return {
            'success': True,
            'elastic_tensor': C_GPa.tolist(),  # 6x6 matrix in GPa
            'compliance_matrix': S_GPa.tolist() if S_GPa is not None else None, # Only convert to list if not None
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
    """
    Check mechanical stability criteria for different crystal systems
    """
    try:
        criteria = {}

        eigenvals = np.linalg.eigvals(C)
        # Convert numpy.bool_ to standard bool
        all_positive = bool(np.all(eigenvals > 0))
        criteria['positive_definite'] = all_positive

        # Convert numpy.bool_ to standard bool for all criteria
        criteria['C11_positive'] = bool(C[0, 0] > 0)
        criteria['C22_positive'] = bool(C[1, 1] > 0)
        criteria['C33_positive'] = bool(C[2, 2] > 0)
        criteria['C44_positive'] = bool(C[3, 3] > 0)
        criteria['C55_positive'] = bool(C[4, 4] > 0)
        criteria['C66_positive'] = bool(C[5, 5] > 0)

        # Convert numpy.bool_ to standard bool
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
    """Create phonon data export with proper JSON serialization"""
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
        # Convert all thermodynamic values to Python native types
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
    """Create elastic data export with proper JSON serialization"""
    if not elastic_results['success']:
        return None

    export_data = {
        'structure_name': structure_name,
        'elastic_tensor_GPa': elastic_results['elastic_tensor'],  # Already converted to list in the calculation
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
    """Convert mechanical stability dictionary to JSON-serializable format"""
    json_stability = {}
    for key, value in stability_dict.items():
        if isinstance(value, (np.bool_, bool)):
            json_stability[key] = bool(value)  # Convert numpy bool to Python bool
        elif isinstance(value, (np.integer, int)):
            json_stability[key] = int(value)  # Convert numpy int to Python int
        elif isinstance(value, (np.floating, float)):
            json_stability[key] = float(value)  # Convert numpy float to Python float
        elif isinstance(value, np.ndarray):
            json_stability[key] = value.tolist()  # Convert numpy array to list
        elif value is None:
            json_stability[key] = None
        else:
            json_stability[key] = value  # Keep as is for strings and other types

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
    """
    Calculate phonons using Pymatgen + Phonopy with MACE calculator
    This is much more robust than direct ASE phonon implementation
    """
    try:
        log_queue.put(f"Starting Pymatgen+Phonopy phonon calculation for {structure_name}")

        # Check if phonopy is available
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

        # Set calculator
        atoms.calc = calculator

        # Get number of atoms in primitive cell
        num_initial_atoms = len(atoms)
        log_queue.put(f"  Primitive cell has {num_initial_atoms} atoms")

        # Brief optimization for stability
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

        # Convert ASE atoms to pymatgen Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()
        pmg_structure = adaptor.get_structure(atoms)
        log_queue.put(f"  Converted to pymatgen structure: {pmg_structure.composition}")

        # Convert to PhonopyAtoms
        phonopy_atoms = PhonopyAtoms(
            symbols=[str(site.specie) for site in pmg_structure],
            scaled_positions=pmg_structure.frac_coords,
            cell=pmg_structure.lattice.matrix
        )

        # Estimate supercell size
        if phonon_params.get('auto_supercell', True):
            log_queue.put("  Auto-determining supercell size...")

            # Get lattice parameters
            a, b, c = pmg_structure.lattice.abc
            target_length = phonon_params.get('target_supercell_length', 15.0)
            max_multiplier = phonon_params.get('max_supercell_multiplier', 4)

            # Calculate multipliers
            na = max(1, min(max_multiplier, int(np.ceil(target_length / a))))
            nb = max(1, min(max_multiplier, int(np.ceil(target_length / b))))
            nc = max(1, min(max_multiplier, int(np.ceil(target_length / c))))

            # Adjust for large structures
            if num_initial_atoms > 50:
                na = max(1, na - 1)
                nb = max(1, nb - 1)
                nc = max(1, nc - 1)

            supercell_matrix = [[na, 0, 0], [0, nb, 0], [0, 0, nc]]
        else:
            # Use user-specified supercell
            sc = phonon_params.get('supercell_size', (2, 2, 2))
            supercell_matrix = [[sc[0], 0, 0], [0, sc[1], 0], [0, 0, sc[2]]]

        total_atoms = num_initial_atoms * np.prod([supercell_matrix[i][i] for i in range(3)])
        log_queue.put(f"  Supercell matrix: {supercell_matrix}")
        log_queue.put(f"  Total atoms in supercell: {total_atoms}")

        # Limit supercell size for very large structures
        max_atoms = phonon_params.get('max_supercell_atoms', 800)
        if total_atoms > max_atoms:
            log_queue.put(f"  ⚠️ Supercell too large ({total_atoms} atoms), using smaller supercell")
            supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            total_atoms = num_initial_atoms

        # Initialize Phonopy
        log_queue.put("  Initializing Phonopy...")
        phonon = Phonopy(
            phonopy_atoms,
            supercell_matrix=supercell_matrix,
            primitive_matrix='auto'  # Let phonopy find the primitive cell
        )

        # Generate displacements
        displacement_distance = phonon_params.get('delta', 0.01)
        log_queue.put(f"  Generating displacements (distance={displacement_distance} Å)...")
        phonon.generate_displacements(distance=displacement_distance)

        # Get supercells with displacements
        supercells = phonon.get_supercells_with_displacements()
        log_queue.put(f"  Generated {len(supercells)} displaced supercells")

        # Calculate forces for each displaced supercell
        log_queue.put("  Calculating forces for displaced supercells...")
        forces = []

        for i, supercell in enumerate(supercells):
            log_queue.put(f"    Calculating forces for supercell {i + 1}/{len(supercells)}")

            # Convert PhonopyAtoms to ASE Atoms
            ase_supercell = Atoms(
                symbols=supercell.symbols,
                positions=supercell.positions,
                cell=supercell.cell,
                pbc=True
            )
            ase_supercell.calc = calculator

            try:
                # Get forces
                supercell_forces = ase_supercell.get_forces()
                forces.append(supercell_forces)

                # Progress indicator
                if (i + 1) % max(1, len(supercells) // 10) == 0:
                    progress = (i + 1) / len(supercells) * 100
                    log_queue.put(f"    Progress: {progress:.1f}% ({i + 1}/{len(supercells)})")

            except Exception as force_error:
                log_queue.put(f"    ❌ Force calculation failed for supercell {i + 1}: {str(force_error)}")
                return {'success': False, 'error': f'Force calculation failed: {str(force_error)}'}

        log_queue.put("  ✅ All force calculations completed")

        # Set forces in Phonopy
        phonon.forces = forces

        # Produce force constants
        log_queue.put("  Calculating force constants...")
        phonon.produce_force_constants()

        # Calculate phonon band structure
        log_queue.put("  Calculating phonon band structure...")

        # Get high-symmetry path
        try:
            # Use pymatgen to get the high-symmetry path
            from pymatgen.symmetry.bandstructure import HighSymmKpath
            kpath = HighSymmKpath(pmg_structure)
            path = kpath.kpath["path"]
            kpoints = kpath.kpath["kpoints"]

            # Convert to the format expected by phonopy
            bands = []
            labels = []

            for segment in path:
                segment_points = []
                for point_name in segment:
                    segment_points.append(kpoints[point_name])
                bands.append(segment_points)
                labels.extend(segment)

            # Remove duplicate labels at segment boundaries
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

        # Set up band structure calculation
        npoints = phonon_params.get('npoints', 101)
        phonon.run_band_structure(
            bands,
            is_band_connection=False,
            with_eigenvectors=False,
            is_legacy_plot=False
        )

        # Get band structure data - FIXED SECTION
        log_queue.put("  Processing band structure data...")
        band_dict = phonon.get_band_structure_dict()
        
        # Handle the irregular frequency array structure
        raw_frequencies = band_dict['frequencies']
        log_queue.put(f"  Raw frequencies type: {type(raw_frequencies)}")
        log_queue.put(f"  Raw frequencies length: {len(raw_frequencies)}")
        
        # Check the structure of the frequency data
        if isinstance(raw_frequencies, list):
            # Convert each sub-array and find the maximum number of bands
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
            
            # Create a regular array by padding with NaN if necessary
            total_kpoints = sum(len(freq_array) if freq_array.ndim > 0 else 1 for freq_array in freq_arrays)
            frequencies = np.full((total_kpoints, max_bands), np.nan)
            
            kpoint_idx = 0
            for freq_array in freq_arrays:
                freq_np = np.array(freq_array)
                if freq_np.ndim == 1:
                    # Single k-point
                    n_bands = len(freq_np)
                    frequencies[kpoint_idx, :n_bands] = freq_np
                    kpoint_idx += 1
                elif freq_np.ndim == 2:
                    # Multiple k-points
                    n_kpts, n_bands = freq_np.shape
                    frequencies[kpoint_idx:kpoint_idx+n_kpts, :n_bands] = freq_np
                    kpoint_idx += n_kpts
                    
        else:
            # Try direct conversion
            frequencies = np.array(raw_frequencies)
            
        # Convert units: THz to meV
        frequencies = frequencies * units_phonopy.THzToEv * 1000  # Convert to meV
        
        # Remove NaN values for statistics
        valid_frequencies = frequencies[~np.isnan(frequencies)]
        log_queue.put(f"  ✅ Band structure calculated: {frequencies.shape} (valid points: {len(valid_frequencies)})")

        # Get k-points - handle irregular structure
        raw_kpoints = band_dict['qpoints']
        log_queue.put(f"  Raw k-points type: {type(raw_kpoints)}")
        log_queue.put(f"  Raw k-points length: {len(raw_kpoints)}")
        
        # Handle irregular k-point array structure
        if isinstance(raw_kpoints, list):
            # Flatten the k-points list
            kpoints_band = []
            for kpt_group in raw_kpoints:
                kpt_array = np.array(kpt_group)
                if kpt_array.ndim == 1:
                    # Single k-point
                    kpoints_band.append(kpt_array)
                elif kpt_array.ndim == 2:
                    # Multiple k-points
                    for kpt in kpt_array:
                        kpoints_band.append(kpt)
                else:
                    log_queue.put(f"  ⚠️ Unexpected k-point dimension: {kpt_array.ndim}")
                    
            # Convert to numpy array
            kpoints_band = np.array(kpoints_band)
        else:
            # Try direct conversion
            kpoints_band = np.array(raw_kpoints)
            
        log_queue.put(f"  Processed k-points shape: {kpoints_band.shape}")
        
        # Ensure k-points and frequencies have consistent dimensions
        if len(kpoints_band) != frequencies.shape[0]:
            log_queue.put(f"  ⚠️ K-point count mismatch: {len(kpoints_band)} vs {frequencies.shape[0]}")
            # Truncate or pad as needed
            min_len = min(len(kpoints_band), frequencies.shape[0])
            kpoints_band = kpoints_band[:min_len]
            frequencies = frequencies[:min_len]
            log_queue.put(f"  Adjusted to consistent length: {min_len}")

        # Calculate DOS
        log_queue.put("  Calculating phonon DOS...")
        try:
            # Use smaller mesh for large structures
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
            # Create simple DOS from band structure
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

        # Check for imaginary modes
        imaginary_count = np.sum(valid_frequencies < 0)
        min_frequency = np.min(valid_frequencies) if len(valid_frequencies) > 0 else 0

        if imaginary_count > 0:
            log_queue.put(f"  ⚠️ Found {imaginary_count} imaginary modes")
            log_queue.put(f"    Most negative frequency: {min_frequency:.3f} meV")
        else:
            log_queue.put("  ✅ No imaginary modes found")

        # Calculate thermodynamic properties
        temp = phonon_params.get('temperature', 300)
        log_queue.put(f"  Calculating thermodynamics at {temp} K...")

        try:
            phonon.run_thermal_properties(
                t_step=50,
                t_max=temp + 50,
                t_min=temp - 50 if temp > 50 else 1
            )

            thermal_dict = phonon.get_thermal_properties_dict()

            # Find closest temperature
            temps = np.array(thermal_dict['temperatures'])
            temp_idx = np.argmin(np.abs(temps - temp))

            thermo_props = {
                'temperature': float(temps[temp_idx]),
                'zero_point_energy': float(thermal_dict['zero_point_energy']),  # eV
                'internal_energy': float(thermal_dict['heat_capacity'][temp_idx]),  # eV
                'heat_capacity': float(thermal_dict['heat_capacity'][temp_idx]),  # eV/K
                'entropy': float(thermal_dict['entropy'][temp_idx]),  # eV/K
                'free_energy': float(thermal_dict['free_energy'][temp_idx])  # eV
            }

            log_queue.put(f"  Zero-point energy: {thermo_props['zero_point_energy']:.6f} eV")
            log_queue.put(f"  Heat capacity: {thermo_props['heat_capacity']:.6f} eV/K")

        except Exception as thermo_error:
            log_queue.put(f"  ⚠️ Thermodynamics calculation failed: {str(thermo_error)}")

            # Fallback calculation
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


# Also add this function to install phonopy if not available
def check_and_install_phonopy():
    """Check if phonopy is available and provide installation instructions"""
    try:
        import phonopy
        return True, "Phonopy is available"
    except ImportError:
        return False, "Phonopy not found. Please install with: pip install phonopy"


# Add these imports to your existing imports
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
        view.setStyle({'sphere': {'scale': 0.3}, 'stick': {'radius': 0.2}})
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
    return atoms


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


def run_mace_calculation(structure_data, calc_type, model_size, device, optimization_params, phonon_params,
                         elastic_params,
                         calc_formation_energy, log_queue, stop_event):
    try:
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
                    log_queue.put(
                        f"Optimizer: {optimization_params['optimizer']}, fmax: {optimization_params['fmax']:.3f} eV/Å, max steps: {optimization_params['max_steps']}")

                    log_queue.put({
                        'type': 'opt_start',
                        'structure': name,
                        'max_steps': optimization_params['max_steps'],
                        'fmax': optimization_params['fmax'],
                        'ediff': optimization_params['ediff']
                    })

                    logger = OptimizationLogger(log_queue, name)

                    try:
                        if optimization_params['optimizer'] == "LBFGS":
                            optimizer = LBFGS(atoms, logfile=None)
                        else:
                            optimizer = BFGS(atoms, logfile=None)

                        optimizer.attach(lambda: logger(optimizer), interval=1)
                        optimizer.run(fmax=optimization_params['fmax'], steps=optimization_params['max_steps'])

                        energy = atoms.get_potential_energy()
                        final_forces = atoms.get_forces()
                        max_final_force = np.max(np.linalg.norm(final_forces, axis=1))

                        force_converged = max_final_force < optimization_params['fmax']
                        energy_converged = False
                        if len(logger.trajectory) > 1:
                            final_energy_change = logger.trajectory[-1]['energy_change']
                            energy_converged = final_energy_change < optimization_params['ediff']

                        if force_converged and energy_converged:
                            convergence_status = "CONVERGED (Force & Energy)"
                        elif force_converged:
                            convergence_status = "CONVERGED (Force only)"
                        elif energy_converged:
                            convergence_status = "CONVERGED (Energy only)"
                        else:
                            convergence_status = "MAX STEPS REACHED"

                        optimized_structure = Structure(
                            lattice=atoms.cell[:],
                            species=[atom.symbol for atom in atoms],
                            coords=atoms.positions,
                            coords_are_cartesian=True
                        )
                        final_structure = optimized_structure

                        final_energy_change = logger.trajectory[-1]['energy_change'] if len(
                            logger.trajectory) > 1 else 0
                        log_queue.put(
                            f"✅ Optimization {convergence_status} for {name}: Final energy = {energy:.6f} eV, Final max force = {max_final_force:.4f} eV/Å, Final ΔE = {final_energy_change:.2e} eV ({optimizer.nsteps} steps)")

                        log_queue.put({
                            'type': 'complete_trajectory',
                            'structure': name,
                            'trajectory': logger.trajectory
                        })

                        log_queue.put({
                            'type': 'opt_complete',
                            'structure': name,
                            'final_steps': optimizer.nsteps,
                            'converged': force_converged and energy_converged,
                            'force_converged': force_converged,
                            'energy_converged': energy_converged,
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


                # Replace the phonon parameters section in your GUI with this:

                elif calc_type == "Phonon Calculation":

                    # Brief pre-optimization for stability

                    if optimization_params['max_steps'] > 0:

                        log_queue.put(f"Running brief pre-phonon optimization for {name}")

                        temp_atoms = atoms.copy()

                        temp_atoms.calc = calculator

                        try:

                            temp_optimizer = LBFGS(temp_atoms, logfile=None)

                            temp_optimizer.run(fmax=0.02, steps=50)  # Quick optimization

                            atoms = temp_atoms

                            energy = atoms.get_potential_energy()

                            log_queue.put(f"Pre-phonon optimization completed. Energy: {energy:.6f} eV")

                        except Exception as pre_opt_error:

                            log_queue.put(f"⚠️ Pre-optimization failed: {str(pre_opt_error)}")

                            energy = atoms.get_potential_energy()

                    # Use simple phonon calculation

                    phonon_results = calculate_phonons_pymatgen(atoms, calculator, phonon_params, log_queue, name)

                    if phonon_results['success']:
                        energy = atoms.get_potential_energy()

                elif calc_type == "Elastic Properties":
                    # Ensure structure is optimized or nearly optimized before elastic
                    if optimization_params['max_steps'] > 0:  # If optimization was implicitly run or parameters set
                        log_queue.put(f"Running pre-elastic optimization for {name} to ensure stability.")
                        temp_atoms = atoms.copy()
                        temp_atoms.calc = calculator
                        temp_logger = OptimizationLogger(log_queue, f"{name}_pre_elastic_opt")
                        try:
                            temp_optimizer = LBFGS(temp_atoms, logfile=None)
                            temp_optimizer.attach(lambda: temp_logger(temp_optimizer), interval=1)
                            temp_optimizer.run(fmax=0.015, steps=100)  # Stricter fmax for elastic pre-opt
                            atoms = temp_atoms  # Use optimized atoms for elastic calculation
                            energy = atoms.get_potential_energy()
                            log_queue.put(
                                f"Pre-elastic optimization finished for {name}. Final energy: {energy:.6f} eV")
                        except Exception as pre_opt_error:
                            log_queue.put(f"⚠️ Pre-elastic optimization failed for {name}: {str(pre_opt_error)}")
                            log_queue.put("Continuing with elastic calculation on potentially unoptimized structure.")
                            energy = atoms.get_potential_energy()  # Still get current energy

                    elastic_results = calculate_elastic_properties(atoms, calculator, elastic_params, log_queue, name)
                    if elastic_results['success']:
                        energy = atoms.get_potential_energy()  # Get energy after elastic calc if successful

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

            except Exception as e:
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

    except Exception as e:
        log_queue.put(f"❌ Fatal error in calculation thread: {str(e)}")
        log_queue.put(f"Error type: {type(e).__name__}")
        import traceback
        log_queue.put(f"Traceback: {traceback.format_exc()}")

    finally:
        log_queue.put("CALCULATION_FINISHED")


st.set_page_config(page_title="MACE Batch Structure Calculator", layout="wide")
st.title("MACE Batch Structure Calculator")

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

    st.info(f"**Selected Model:** {selected_model}")
    st.info(f"**Device:** {device}")

    if MACE_IMPORT_METHOD == "mace_mp":
        st.info("Using mace_mp - models downloaded automatically")
    else:
        st.warning("Using MACECalculator - may require local model files")

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
                st.success("✅ VASP calculations are running")
                if st.session_state.get('progress_text', ''):
                    st.write(f"📈 {st.session_state.progress_text}")
                st.write("👀 **Switch to 'Live Console or Live Results' tab for detailed output**")
                if st.button("🛑 Stop Calculation", key="stop_top"):
                    st.session_state.stop_event.set()

    # Optional: Add progress bar at the top level too
    if st.session_state.get('total_steps', 0) > 0:
        progress_value = st.session_state.progress / st.session_state.total_steps
        st.progress(progress_value, text=st.session_state.get('progress_text', ''))


tab1, tab2, tab3, tab4 = st.tabs(
    ["📁 Structure Upload & Setup", "🖥️ Calculation Console", "📊 Results & Analysis", "📈 Optimization Trajectories"])


def generate_python_script(structures, calc_type, model_size, device, optimization_params,
                           phonon_params, elastic_params, calc_formation_energy):
    """Generate a standalone Python script with current settings"""

    # Get structure filenames
    structure_files = list(structures.keys())

    script = f'''#!/usr/bin/env python3
"""
MACE Batch Structure Calculator - Standalone Script
Generated from Streamlit GUI settings

Usage: python mace_calculation_script.py
Make sure POSCAR files are in the same directory as this script.
"""

import os
import json
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

# Import MACE
try:
    from mace.calculators import mace_mp
    MACE_AVAILABLE = True
    print("✅ MACE imported successfully")
except ImportError:
    try:
        from mace.calculators import MACECalculator
        MACE_AVAILABLE = True
        print("✅ MACE imported successfully (MACECalculator)")
    except ImportError:
        print("❌ MACE not available. Install with: pip install mace-torch")
        exit(1)

# Import additional modules for advanced calculations
try:
    from ase.phonons import Phonons
    PHONON_AVAILABLE = True
except ImportError:
    PHONON_AVAILABLE = False
    print("⚠️ Phonon calculations not available")

try:
    from ase.utils.eos import EquationOfState
    ELASTIC_AVAILABLE = True
except ImportError:
    ELASTIC_AVAILABLE = False
    print("⚠️ Elastic calculations not available")


class CalculationLogger:
    """Simple logger for console output and file writing"""

    def __init__(self, log_file="calculation.log"):
        self.log_file = log_file

    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\\n")


def pymatgen_to_ase(structure):
    """Convert pymatgen Structure to ASE Atoms"""
    atoms = Atoms(
        symbols=[str(site.specie) for site in structure],
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True
    )
    return atoms


def create_directories():
    """Create output directories"""
    dirs = ['optimized_structures', 'trajectories', 'results']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)


def save_trajectory_xyz(trajectory, filename):
    """Save optimization trajectory to XYZ format"""
    with open(filename, 'w') as f:
        for step_data in trajectory:
            step = step_data['step']
            energy = step_data['energy']
            positions = step_data['positions']
            cell = step_data['cell']
            symbols = step_data['symbols']
            forces = step_data['forces']
            max_force = step_data['max_force']

            f.write(f"{{len(positions)}}\\n")

            # Calculate lattice parameters
            a, b, c = np.linalg.norm(cell, axis=1)
            alpha = np.degrees(np.arccos(np.dot(cell[1], cell[2]) / (b * c)))
            beta = np.degrees(np.arccos(np.dot(cell[0], cell[2]) / (a * c)))
            gamma = np.degrees(np.arccos(np.dot(cell[0], cell[1]) / (a * b)))

            comment = (f"Step {{step}} | Energy={{energy:.6f}} eV | Max_Force={{max_force:.4f}} eV/A | "
                      f"Lattice=\\"{{a:.6f}} {{b:.6f}} {{c:.6f}} {{alpha:.2f}} {{beta:.2f}} {{gamma:.2f}}\\" | "
                      f"Properties=species:S:1:pos:R:3:forces:R:3")
            f.write(f"{{comment}}\\n")

            for symbol, pos, force in zip(symbols, positions, forces):
                f.write(f"{{symbol}} {{pos[0]:12.6f}} {{pos[1]:12.6f}} {{pos[2]:12.6f}} {{force[0]:12.6f}} {{force[1]:12.6f}} {{force[2]:12.6f}}\\n")


def calculate_atomic_reference_energies(elements, calculator, logger):
    """Calculate atomic reference energies for formation energy calculation"""
    reference_energies = {{}}

    logger.log("Calculating atomic reference energies...")

    for element in elements:
        try:
            logger.log(f"  Calculating reference energy for {{element}}...")
            atom = Atoms(element, positions=[(0, 0, 0)], cell=[20, 20, 20], pbc=True)
            atom.calc = calculator
            energy = atom.get_potential_energy()
            reference_energies[element] = energy
            logger.log(f"  ✅ {{element}}: {{energy:.6f}} eV")
        except Exception as e:
            logger.log(f"  ❌ Failed to calculate reference energy for {{element}}: {{str(e)}}")
            reference_energies[element] = None

    return reference_energies


def calculate_formation_energy(structure_energy, structure, reference_energies):
    """Calculate formation energy per atom"""
    if structure_energy is None:
        return None

    element_counts = {{}}
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


def main():
    """Main calculation function"""

    # Configuration from GUI settings
    MODEL_SIZE = "{model_size}"
    DEVICE = "{device}"
    CALC_TYPE = "{calc_type}"
    CALC_FORMATION_ENERGY = {calc_formation_energy}

    # Structure files to process
    STRUCTURE_FILES = {structure_files}

    # Optimization parameters
    OPTIMIZATION_PARAMS = {optimization_params}

    # Phonon parameters  
    PHONON_PARAMS = {phonon_params}

    # Elastic parameters
    ELASTIC_PARAMS = {elastic_params}

    # Initialize logger
    logger = CalculationLogger("results/calculation.log")
    logger.log("="*60)
    logger.log("MACE Batch Structure Calculator")
    logger.log("="*60)
    logger.log(f"Model: {{MODEL_SIZE}}")
    logger.log(f"Device: {{DEVICE}}")
    logger.log(f"Calculation Type: {{CALC_TYPE}}")
    logger.log(f"Formation Energy: {{CALC_FORMATION_ENERGY}}")
    logger.log("")

    # Create output directories
    create_directories()

    # Initialize MACE calculator
    logger.log("Setting up MACE calculator...")
    try:
        calculator = mace_mp(model=MODEL_SIZE, dispersion=False, default_dtype="float64", device=DEVICE)
        logger.log("✅ MACE calculator initialized successfully")
    except Exception as e:
        logger.log(f"❌ MACE initialization failed: {{str(e)}}")
        if DEVICE == "cuda":
            logger.log("⚠️ Trying CPU fallback...")
            try:
                calculator = mace_mp(model=MODEL_SIZE, dispersion=False, default_dtype="float64", device="cpu")
                logger.log("✅ MACE calculator initialized on CPU")
            except Exception as cpu_error:
                logger.log(f"❌ CPU fallback failed: {{str(cpu_error)}}")
                return
        else:
            return

    # Load structures
    structures = {{}}
    logger.log("Loading structures...")

    for filename in STRUCTURE_FILES:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    content = f.read()
                structure = Structure.from_str(content, fmt="poscar")
                structures[filename] = structure
                logger.log(f"✅ Loaded {{filename}} - {{structure.composition.reduced_formula}}")
            except Exception as e:
                logger.log(f"❌ Error loading {{filename}}: {{str(e)}}")
        else:
            logger.log(f"❌ File not found: {{filename}}")

    if not structures:
        logger.log("❌ No structures loaded. Make sure POSCAR files are in the current directory.")
        return

    # Calculate reference energies if needed
    reference_energies = {{}}
    if CALC_FORMATION_ENERGY:
        all_elements = set()
        for structure in structures.values():
            for site in structure:
                all_elements.add(site.specie.symbol)
        reference_energies = calculate_atomic_reference_energies(all_elements, calculator, logger)

    # Main calculation loop
    results = []

    for i, (name, structure) in enumerate(structures.items()):
        logger.log(f"\\nProcessing structure {{i+1}}/{{len(structures)}}: {{name}}")
        logger.log("-" * 50)

        try:
            atoms = pymatgen_to_ase(structure)
            atoms.calc = calculator

            # Test calculator
            try:
                test_energy = atoms.get_potential_energy()
                test_forces = atoms.get_forces()
                logger.log(f"✅ Calculator test successful")
                logger.log(f"Initial energy: {{test_energy:.6f}} eV")
                logger.log(f"Initial max force: {{np.max(np.abs(test_forces)):.6f}} eV/Å")
            except Exception as calc_error:
                logger.log(f"❌ Calculator test failed: {{str(calc_error)}}")
                continue

            energy = None
            final_structure = structure
            trajectory = []

            if CALC_TYPE == "Energy Only":
                energy = atoms.get_potential_energy()
                logger.log(f"✅ Energy: {{energy:.6f}} eV")

            elif CALC_TYPE == "Geometry Optimization":
                logger.log("Starting geometry optimization...")

                class OptTracker:
                    def __init__(self):
                        self.trajectory = []
                        self.step_count = 0

                    def __call__(self, optimizer):
                        self.step_count += 1
                        forces = optimizer.atoms.get_forces()
                        max_force = np.max(np.linalg.norm(forces, axis=1))
                        energy = optimizer.atoms.get_potential_energy()

                        step_data = {{
                            'step': self.step_count,
                            'energy': energy,
                            'max_force': max_force,
                            'positions': optimizer.atoms.positions.copy(),
                            'cell': optimizer.atoms.cell.array.copy(),
                            'symbols': optimizer.atoms.get_chemical_symbols(),
                            'forces': forces.copy()
                        }}
                        self.trajectory.append(step_data)

                        logger.log(f"  Step {{self.step_count}}: E={{energy:.6f}} eV, F_max={{max_force:.4f}} eV/Å")

                tracker = OptTracker()

                if OPTIMIZATION_PARAMS['optimizer'] == "LBFGS":
                    optimizer = LBFGS(atoms, logfile=None)
                else:
                    optimizer = BFGS(atoms, logfile=None)

                optimizer.attach(lambda: tracker(optimizer), interval=1)
                optimizer.run(fmax=OPTIMIZATION_PARAMS['fmax'], steps=OPTIMIZATION_PARAMS['max_steps'])

                energy = atoms.get_potential_energy()
                final_forces = atoms.get_forces()
                max_final_force = np.max(np.linalg.norm(final_forces, axis=1))

                force_converged = max_final_force < OPTIMIZATION_PARAMS['fmax']
                convergence_status = "CONVERGED" if force_converged else "MAX STEPS REACHED"

                logger.log(f"✅ Optimization {{convergence_status}}: Final energy = {{energy:.6f}} eV")
                logger.log(f"   Final max force = {{max_final_force:.4f}} eV/Å ({{optimizer.nsteps}} steps)")

                # Save trajectory
                if tracker.trajectory:
                    trajectory_file = f"trajectories/trajectory_{{name.replace('.', '_')}}.xyz"
                    save_trajectory_xyz(tracker.trajectory, trajectory_file)
                    logger.log(f"✅ Trajectory saved to {{trajectory_file}}")

                # Save optimized structure
                optimized_structure = Structure(
                    lattice=atoms.cell[:],
                    species=[atom.symbol for atom in atoms],
                    coords=atoms.positions,
                    coords_are_cartesian=True
                )
                final_structure = optimized_structure

                # Save optimized POSCAR
                optimized_file = f"optimized_structures/optimized_{{name}}"
                with open(optimized_file, 'w') as f:
                    f.write(optimized_structure.to(fmt="poscar"))
                logger.log(f"✅ Optimized structure saved to {{optimized_file}}")

            # Calculate formation energy if requested
            formation_energy = None
            if CALC_FORMATION_ENERGY and energy is not None:
                formation_energy = calculate_formation_energy(energy, structure, reference_energies)
                if formation_energy is not None:
                    logger.log(f"✅ Formation energy: {{formation_energy:.6f}} eV/atom")

            # Store result
            result = {{
                'name': name,
                'energy': energy,
                'formation_energy': formation_energy,
                'structure': final_structure,
                'calc_type': CALC_TYPE
            }}
            results.append(result)

        except Exception as e:
            logger.log(f"❌ Error calculating {{name}}: {{str(e)}}")

    # Save results summary
    logger.log("\\n" + "="*60)
    logger.log("CALCULATION SUMMARY")
    logger.log("="*60)

    with open("results/results_summary.txt", 'w') as f:
        f.write("MACE Batch Calculation Results\\n")
        f.write("="*50 + "\\n\\n")
        f.write(f"Model: {{MODEL_SIZE}}\\n")
        f.write(f"Device: {{DEVICE}}\\n")
        f.write(f"Calculation Type: {{CALC_TYPE}}\\n")
        f.write(f"Formation Energy: {{CALC_FORMATION_ENERGY}}\\n\\n")

        successful_results = [r for r in results if r['energy'] is not None]

        f.write(f"Successful calculations: {{len(successful_results)}}/{{len(results)}}\\n\\n")

        if successful_results:
            f.write("Structure\\tEnergy (eV)\\tFormation Energy (eV/atom)\\n")
            f.write("-" * 70 + "\\n")

            for result in successful_results:
                name = result['name']
                energy = result['energy']
                form_energy = result.get('formation_energy')

                f.write(f"{{name:<20}}\\t{{energy:12.6f}}\\t")
                if form_energy is not None:
                    f.write(f"{{form_energy:12.6f}}")
                else:
                    f.write("N/A")
                f.write("\\n")

                logger.log(f"{{name:<20}} | Energy: {{energy:12.6f}} eV | Formation: {{form_energy:12.6f if form_energy else 'N/A'}} eV/atom")

        if len(successful_results) > 1:
            energies = [r['energy'] for r in successful_results]
            min_energy = min(energies)
            f.write("\\n\\nRelative Energies (meV):\\n")
            f.write("-" * 40 + "\\n")
            for result in successful_results:
                relative_energy = (result['energy'] - min_energy) * 1000
                f.write(f"{{result['name']:<20}}\\t{{relative_energy:8.3f}}\\n")

    logger.log("\\n✅ Results summary saved to results/results_summary.txt")
    logger.log("✅ Calculation completed successfully!")


if __name__ == "__main__":
    main()
'''

    return script



with tab1:
    st.header("1. Upload Structure Files")

    # Show different interface based on whether structures are locked
    if not st.session_state.structures_locked:
        uploaded_files = st.file_uploader(
            "Upload POSCAR files",
            accept_multiple_files=True,
            type=['POSCAR', 'vasp', 'poscar', 'contcar'],
            help="Upload multiple POSCAR format files for batch processing",
            key="structure_uploader"
        )

        if uploaded_files:
            new_structures = {}
            upload_errors = []

            for uploaded_file in uploaded_files:
                try:
                    content = uploaded_file.getvalue().decode("utf-8")
                    structure = Structure.from_str(content, fmt="poscar")
                    new_structures[uploaded_file.name] = structure
                except Exception as e:
                    upload_errors.append(f"Error reading {uploaded_file.name}: {str(e)}")

            if upload_errors:
                for error in upload_errors:
                    st.error(error)

            if new_structures:
                # Store in pending structures
                st.session_state.pending_structures.update(new_structures)
                
                # Show preview of what will be added
                st.info(f"📁 {len(new_structures)} new structures ready to be added")
                
                # Show current + pending count
                total_after_addition = len(st.session_state.structures) + len(st.session_state.pending_structures)
                st.write(f"Current structures: {len(st.session_state.structures)}")
                st.write(f"Pending structures: {len(st.session_state.pending_structures)}")
                st.write(f"Total after addition: {total_after_addition}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Accept & Add Structures", type="primary"):
                        st.session_state.structures.update(st.session_state.pending_structures)
                        added_count = len(st.session_state.pending_structures)
                        st.session_state.pending_structures = {}
                        st.success(f"✅ Added {added_count} structures. Total: {len(st.session_state.structures)}")
                        st.rerun()
                
                with col2:
                    if st.button("❌ Cancel Upload"):
                        st.session_state.pending_structures = {}
                        st.info("Upload cancelled")
                        st.rerun()

        # Show current accepted structures
        if st.session_state.structures:
            st.success(f"✅ {len(st.session_state.structures)} structures loaded and ready")
            
            # Show list of current structures
            with st.expander("📋 View Current Structures", expanded=False):
                for i, (name, structure) in enumerate(st.session_state.structures.items(), 1):
                    st.write(f"{i}. **{name}** - {structure.composition.reduced_formula} ({structure.num_sites} atoms)")

        # Lock/Clear buttons
        col1, col2 = st.columns(2)
        

        with col1:  
            if st.button("🔒 Lock Structures for Calculation", 
                        disabled=len(st.session_state.structures) == 0, type = 'primary'):
                st.session_state.structures_locked = True
                st.success("🔒 Structures locked! You can now start calculations safely.")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All Structures", type ='secondary'):
                st.session_state.structures = {}
                st.session_state.pending_structures = {}
                st.session_state.results = []
                st.session_state.optimization_trajectories = {}
                st.success("All structures cleared")
                st.rerun()

    else:
        # Structures are locked - show read-only view
        st.success(f"🔒 Structures Locked ({len(st.session_state.structures)} structures)")
        st.info("📌 Structures are locked to avoid refreshing during the calculation run. Use 'Unlock' to modify.")
        
        # Show locked structures list
        with st.expander("📋 Locked Structures", expanded=True):
            for i, (name, structure) in enumerate(st.session_state.structures.items(), 1):
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.write(f"**{i}. {name}**")
                with col2:
                    st.write(f"{structure.composition.reduced_formula}")
                with col3:
                    st.write(f"{structure.num_sites} atoms")
        
        # Unlock button (disabled during calculation)
        if st.button("🔓 Unlock Structures", type = 'secondary',
                    disabled=st.session_state.calculation_running):
            st.session_state.structures_locked = False
            st.info("🔓 Structures unlocked. You can now modify the structure list.")
            st.rerun()
        
        if st.session_state.calculation_running:
            st.warning("⚠️ Cannot unlock structures while calculation is running")

    st.divider()

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

        st.header("3. Calculation Setup")

        all_compatible = all(check_mace_compatibility(struct)[0] for struct in st.session_state.structures.values())

        if not all_compatible:
            st.error(
                "⚠️ Some structures contain elements not supported by MACE-MP-0. Please remove incompatible structures.")

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
            'density': None  # Will be estimated if not provided
        }

        if calc_type == "Geometry Optimization":
            st.subheader("Optimization Parameters")

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
                    "Energy threshold (eV)",
                    min_value=1e-6,
                    max_value=1e-2,
                    value=1e-4,
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

            st.info(
                f"Optimization will stop when forces < {optimization_params['fmax']} eV/Å AND energy change < {optimization_params['ediff']:.1e} eV, or after {optimization_params['max_steps']} steps")


        elif calc_type == "Phonon Calculation":

            st.subheader("Phonon Calculation Parameters")

            st.info(
                "A brief pre-optimization (fmax=0.01 eV/Å, max 100 steps) will be performed for stability before phonon calculations.")

            # Supercell size options

            st.write("**Supercell Configuration**")

            auto_supercell = st.checkbox("Automatic supercell size estimation", value=True,

                                         help="Automatically estimate appropriate supercell size based on structure")

            if auto_supercell:

                col_auto1, col_auto2, col_auto3 = st.columns(3)

                with col_auto1:

                    target_length = st.number_input("Target supercell length (Å)", min_value=8.0, max_value=30.0,

                                                    value=15.0, step=1.0,

                                                    help="Minimum length for each supercell dimension")

                with col_auto2:

                    max_multiplier = st.number_input("Max supercell multiplier", min_value=1, max_value=6,

                                                     value=4, step=1,

                                                     help="Maximum allowed multiplier for any dimension")

                with col_auto3:

                    max_atoms = st.number_input("Max supercell atoms", min_value=100, max_value=2000,

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

            # Other phonon parameters

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
            #elastic_params['density'] = st.number_input("Material Density (g/cm³)", min_value=0.1, value=None,
            #                                            help="Optional: Provide if known. Otherwise, it will be estimated from the structure.",
            #                                            format="%.3f")
            elastic_params['density'] = None
        col1, col2= st.columns(2)  # Changed from 2 to 3 columns
        
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
                         not st.session_state.structures_locked,  # Add this condition
            )

        # Add helpful message if structures aren't locked
        if len(st.session_state.structures) > 0 and not st.session_state.structures_locked:
            st.warning("🔒 Please lock your structures before starting calculation to prevent accidental changes.")

        with col2:
            but_script = st.button(
                    "📝 Generate Python Script",
                    type="tertiary",
                    disabled=len(st.session_state.structures) == 0,
            )
        if but_script:
            # Generate the script content
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

                # Display the script in an expandable code block
            with st.expander("📋 Generated Python Script", expanded=True):
                st.code(script_content, language='python')

                # Download button for the script
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
        st.info("Upload POSCAR files to begin")

with tab2:
    st.header("Calculation Console")

    if st.session_state.calculation_running:
        st.info("🔄 Calculation in progress...")

        if st.button("🛑 Stop Calculation", key="console_stop"):
            st.session_state.stop_event.set()

    has_new_messages = False
    message_count = 0
    max_messages_per_cycle = 10

    while not st.session_state.log_queue.empty() and message_count < max_messages_per_cycle:
        has_new_messages = True
        message_count += 1
        message = st.session_state.log_queue.get()

        if isinstance(message, dict):
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
                    'energy_change': message.get('energy_change', 0)
                })
                if st.session_state.current_optimization_info.get('structure') == structure_name:
                    st.session_state.current_optimization_info.update({
                        'current_step': message['step'],
                        'current_energy': message['energy'],
                        'current_max_force': message['max_force'],
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
                opt_text += f" | Energy: {opt_info['current_energy']:.6f} eV | Max Force: {opt_info['current_max_force']:.4f} eV/Å"
                if 'current_energy_change' in opt_info:
                    opt_text += f" | ΔE: {opt_info['current_energy_change']:.2e} eV"

            st.progress(opt_progress, text=opt_text)

            if 'current_step' in opt_info and opt_info['current_step'] > 0:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Step", f"{opt_info['current_step']}/{opt_info['max_steps']}")
                with col2:
                    if 'current_energy' in opt_info:
                        st.metric("Energy (eV)", f"{opt_info['current_energy']:.6f}")
                with col3:
                    if 'current_max_force' in opt_info:
                        force_converged = opt_info['current_max_force'] < opt_info['fmax']
                        st.metric("Max Force (eV/Å)", f"{opt_info['current_max_force']:.4f}",
                                  delta="Converged" if force_converged else "Not converged")
                with col4:
                    if 'current_energy_change' in opt_info:
                        energy_converged = opt_info['current_energy_change'] < opt_info['ediff']
                        st.metric("ΔE (eV)", f"{opt_info['current_energy_change']:.2e}",
                                  delta="Converged" if energy_converged else "Not converged")

    if st.session_state.log_messages:
        recent_messages = st.session_state.log_messages[-20:]
        st.text_area("Calculation Log", "\n".join(recent_messages), height=300)

    if has_new_messages and st.session_state.calculation_running:
        time.sleep(1)
        st.rerun()

with tab3:
    st.header("Results & Analysis")

    if st.session_state.results:
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
                fig.add_trace(go.Bar(
                    x=names,
                    y=energies,
                    name='Total Energy',
                    marker_color='steelblue',
                    hovertemplate='<b>%{x}</b><br>Energy: %{y:.6f} eV<extra></extra>'
                ))

                fig.update_layout(
                    title=dict(text="Total Energy Comparison", font=dict(size=20)),
                    xaxis_title="Structure",
                    yaxis_title="Energy (eV)",
                    height=500,
                    xaxis=dict(tickangle=45)
                )

                st.plotly_chart(fig, use_container_width=True, key=f"energy_plot_{len(successful_results)}")

            if has_formation_energies:
                with col_energy2:
                    valid_formation_data = [(name, fe) for name, fe in zip(names, formation_energies) if fe is not None]
                    if valid_formation_data:
                        valid_names, valid_formation_energies = zip(*valid_formation_data)

                        fig_form = go.Figure()
                        fig_form.add_trace(go.Bar(
                            x=valid_names,
                            y=valid_formation_energies,
                            name='Formation Energy',
                            marker_color='orange',
                            hovertemplate='<b>%{x}</b><br>Formation Energy: %{y:.6f} eV/atom<extra></extra>'
                        ))

                        fig_form.update_layout(
                            title=dict(text="Formation Energy per Atom", font=dict(size=20)),
                            xaxis_title="Structure",
                            yaxis_title="Formation Energy (eV/atom)",
                            height=500,
                            xaxis=dict(tickangle=45)
                        )

                        st.plotly_chart(fig_form, use_container_width=True,
                                        key=f"formation_plot_{len(valid_formation_data)}")

            if len(successful_results) > 1:
                st.subheader("Relative Energies")
                min_energy = min(energies)
                relative_energies = [(e - min_energy) * 1000 for e in energies]

                fig_rel = go.Figure()
                fig_rel.add_trace(go.Bar(
                    x=names,
                    y=relative_energies,
                    name='Relative Energy',
                    marker_color='orange',
                    hovertemplate='<b>%{x}</b><br>Relative Energy: %{y:.3f} meV<extra></extra>'
                ))

                fig_rel.update_layout(
                    title=dict(text="Relative Energies (meV)", font=dict(size=20)),
                    xaxis_title="Structure",
                    yaxis_title="Relative Energy (meV)",
                    height=500,
                    xaxis=dict(tickangle=45)
                )

                st.plotly_chart(fig_rel, use_container_width=True, key=f"relative_plot_{len(successful_results)}")

        st.subheader("Detailed Results")
        st.dataframe(df_results, use_container_width=True, key=f"results_table_{len(st.session_state.results)}")

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
                        poscar_content = result['structure'].to(fmt="poscar")
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
                    title="Phonon Dispersion",
                    xaxis_title="k-point index",
                    yaxis_title="Frequency (meV)",
                    height=400,
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
                    title="Phonon Density of States",
                    xaxis_title="DOS (states/meV)",
                    yaxis_title="Frequency (meV)",
                    height=400,
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
            col_el1, col_el2 = st.columns([1,1])
            with col_el1:
                st.write("**Elastic Tensor (GPa)**")

                elastic_tensor = np.array(elastic_data['elastic_tensor'])

                # Your debug lines (can keep or remove, they confirm data is good)
                st.write(f"Shape of elastic_tensor: {elastic_tensor.shape}")
                st.write(f"Content of elastic_tensor (first 2x2): {elastic_tensor[:2, :2]}")
                st.write(f"Type of elastic_tensor elements: {elastic_tensor.dtype}")

                fig_tensor = go.Figure(data=go.Heatmap(
                    z=elastic_tensor,
                    x=['11', '22', '33', '23', '13', '12'],
                    y=['11', '22', '33', '23', '13', '12'],
                    colorscale='RdBu_r',
                    colorbar=dict(
                        title="C_ij (GPa)",
                        title_font=dict(size=18),  # Colorbar title font size
                        tickfont=dict(size=18)  # Colorbar tick font size
                    ),
                    text=[[f"{val:.1f}" for val in row] for row in elastic_tensor],
                    texttemplate="%{text}",
                    textfont={"size": 20},  # Increased heatmap text size
                    hovertemplate='C<sub>%{y}%{x}</sub> = %{z:.2f} GPa<extra></extra>'
                ))

                fig_tensor.update_layout(
                    title="Elastic Tensor C<sub>ij</sub>",
                    xaxis_title="j",
                    yaxis_title="i",
                    height=400,
                    # --- ADD OR MODIFY THESE LINES ---
                    xaxis=dict(
                        type='category',  # Force categorical axis
                        tickvals=['11', '22', '33', '23', '13', '12'],  # Ensure these specific tick values are used
                        ticktext=['11', '22', '33', '23', '13', '12'],
                        tickfont = dict(size=16),
                        title_font = dict(size=18)
                    ),
                    yaxis=dict(
                        type='category',  # Force categorical axis
                        tickvals=['11', '22', '33', '23', '13', '12'],  # Ensure these specific tick values are used
                        ticktext=['11', '22', '33', '23', '13', '12'],
                        autorange='reversed',  # Keep this for matrix-like display),
                        tickfont = dict(size=16),
                        title_font = dict(size=18)
                    ),
                    hoverlabel=dict(
                        bgcolor="white",
                        bordercolor="black",
                        font_size=16,  # Hover text font size
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

            # Add toggle for display format
            display_format = st.radio("Display format:", ["Table", "Cards"], horizontal=True, index=1)

            # Define the elastic properties with their values and units
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

            # Define colors for different property types
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

            # Create columns for the cards (4 per row)
            if display_format == "Cards":
                cols = st.columns(4)
                for i, prop in enumerate(elastic_properties):
                    with cols[i % 4]:
                        color = property_colors[i % len(property_colors)]

                        # Format the value
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

            else:  # Table format
                # Create the original table format
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
                A_U = 5 * (shear_data['voigt'] / shear_data['reuss']) + (bulk_data['voigt'] / bulk_data['reuss']) - 6

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
                        'Entropy (eV/K)': f"{thermo['entropy']:.6f}"
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
                        title="Imaginary Modes Comparison",
                        xaxis_title="Structure",
                        yaxis_title="Number of Imaginary Modes",
                        height=300
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
                        title="Minimum Frequency Comparison",
                        xaxis_title="Structure",
                        yaxis_title="Minimum Frequency (meV)",
                        height=300
                    )

                    fig_min_freq.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7)

                    st.plotly_chart(fig_min_freq, use_container_width=True)

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
                        title="Bulk vs Shear Modulus",
                        xaxis_title="Structure",
                        yaxis_title="Modulus (GPa)",
                        height=400,
                        barmode='group'
                    )

                    st.plotly_chart(fig_moduli_comp, use_container_width=True)

                with col_el_comp2:
                    poisson_values = [r['elastic_results']['poisson_ratio'] for r in elastic_results]

                    fig_poisson = go.Figure(data=go.Scatter(
                        x=structures,
                        y=poisson_values,
                        mode='markers+lines',
                        marker=dict(size=10, color='green'),
                        line=dict(width=2, color='green')
                    ))

                    fig_poisson.update_layout(
                        title="Poisson's Ratio Comparison",
                        xaxis_title="Structure",
                        yaxis_title="Poisson's Ratio",
                        height=300
                    )

                    st.plotly_chart(fig_poisson, use_container_width=True)

                    density_values = [r['elastic_results']['density'] for r in elastic_results]

                    fig_density = go.Figure(data=go.Bar(
                        x=structures,
                        y=density_values,
                        marker_color='purple',
                        text=[f"{d:.2f}" for d in density_values],
                        textposition='auto'
                    ))

                    fig_density.update_layout(
                        title="Density Comparison",
                        xaxis_title="Structure",
                        yaxis_title="Density (g/cm³)",
                        height=300
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
