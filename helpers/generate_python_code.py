# Function for generating Python script that can be run separately with the settings in GUI

def generate_python_script(structures, calc_type, model_size, device, optimization_params,
                           phonon_params, elastic_params, calc_formation_energy):
    """Generate a standalone Python script with current settings"""

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
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.io.ase import AseAtomsAdaptor
    try:
        # Try new import first
        from phonopy import physical_units
        units_phonopy = physical_units
    except ImportError:
        try:
            # Alternative new import
            import phonopy.physical_units as units_phonopy
        except ImportError:
            # Fallback to deprecated import
            import phonopy.units as units_phonopy
    PHONOPY_AVAILABLE = True
except ImportError:
    PHONOPY_AVAILABLE = False
    print("⚠️ Phonopy not available for phonon calculations")

try:
    from ase.eos import EquationOfState
    from ase.build import bulk
    ELASTIC_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        from ase.utils.eos import EquationOfState
        from ase.build import bulk
        ELASTIC_AVAILABLE = True
    except ImportError:
        ELASTIC_AVAILABLE = False
        print("⚠️ Elastic calculations not available - install latest ASE version")


class CalculationLogger:
    def __init__(self, log_file="calculation.log"):
        self.log_file = log_file
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.clear_log()

    def clear_log(self):
        with open(self.log_file, 'w') as f:
            f.write("")

    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\\n")


def pymatgen_to_ase(structure):
    atoms = Atoms(
        symbols=[str(site.specie) for site in structure],
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True
    )
    return atoms


def create_directories():
    dirs = ['optimized_structures', 'trajectories', 'results', 'phonon_data', 'elastic_data']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)


def calculate_atomic_reference_energies(elements, calculator, logger):
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


def calculate_phonons_pymatgen(atoms, calculator, phonon_params, logger, structure_name):
    if not PHONOPY_AVAILABLE:
        logger.log("❌ Phonopy not available for phonon calculations")
        return {{'success': False, 'error': 'Phonopy not available'}}

    try:
        logger.log(f"Starting phonon calculation for {{structure_name}}")
        atoms.calc = calculator

        # Brief optimization
        logger.log("  Running brief pre-phonon optimization...")
        temp_atoms = atoms.copy()
        temp_atoms.calc = calculator
        temp_optimizer = LBFGS(temp_atoms, logfile=None)
        temp_optimizer.run(fmax=0.01, steps=50)
        atoms = temp_atoms

        # Convert to pymatgen and phonopy
        adaptor = AseAtomsAdaptor()
        pmg_structure = adaptor.get_structure(atoms)
        phonopy_atoms = PhonopyAtoms(
            symbols=[str(site.specie) for site in pmg_structure],
            scaled_positions=pmg_structure.frac_coords,
            cell=pmg_structure.lattice.matrix
        )

        # Determine supercell
        if phonon_params.get('auto_supercell', True):
            target_length = phonon_params.get('target_supercell_length', 15.0)
            max_multiplier = phonon_params.get('max_supercell_multiplier', 4)
            
            a, b, c = pmg_structure.lattice.abc
            na = max(1, min(max_multiplier, int(np.ceil(target_length / a))))
            nb = max(1, min(max_multiplier, int(np.ceil(target_length / b))))
            nc = max(1, min(max_multiplier, int(np.ceil(target_length / c))))
            
            if len(atoms) > 50:
                na = max(1, na - 1)
                nb = max(1, nb - 1)
                nc = max(1, nc - 1)
                
            supercell_matrix = [[na, 0, 0], [0, nb, 0], [0, 0, nc]]
        else:
            sc = phonon_params.get('supercell_size', (2, 2, 2))
            supercell_matrix = [[sc[0], 0, 0], [0, sc[1], 0], [0, 0, sc[2]]]

        logger.log(f"  Supercell matrix: {{supercell_matrix}}")

        # Initialize Phonopy
        phonon = Phonopy(phonopy_atoms, supercell_matrix=supercell_matrix, primitive_matrix='auto')
        
        # Generate displacements and calculate forces
        displacement_distance = phonon_params.get('delta', 0.01)
        phonon.generate_displacements(distance=displacement_distance)
        supercells = phonon.get_supercells_with_displacements()
        
        logger.log(f"  Calculating forces for {{len(supercells)}} displaced supercells...")
        forces = []
        
        for i, supercell in enumerate(supercells):
            ase_supercell = Atoms(
                symbols=supercell.symbols,
                positions=supercell.positions,
                cell=supercell.cell,
                pbc=True
            )
            ase_supercell.calc = calculator
            supercell_forces = ase_supercell.get_forces()
            forces.append(supercell_forces)

        phonon.forces = forces
        phonon.produce_force_constants()

        # Calculate band structure
        logger.log("  Calculating phonon band structure...")
        try:
            from pymatgen.symmetry.bandstructure import HighSymmKpath
            kpath = HighSymmKpath(pmg_structure)
            path = kpath.kpath["path"]
            kpoints = kpath.kpath["kpoints"]
            bands = []
            for segment in path:
                segment_points = [kpoints[point_name] for point_name in segment]
                bands.append(segment_points)
        except:
            bands = [[[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]]]

        phonon.run_band_structure(bands, is_band_connection=False, with_eigenvectors=False)
        band_dict = phonon.get_band_structure_dict()
        
        # Process frequencies
        raw_frequencies = band_dict['frequencies']
        if isinstance(raw_frequencies, list):
            all_freqs = []
            for freq_array in raw_frequencies:
                freq_np = np.array(freq_array)
                if freq_np.ndim == 1:
                    all_freqs.append(freq_np)
                elif freq_np.ndim == 2:
                    all_freqs.extend(freq_np)
            frequencies = np.array(all_freqs)
        else:
            frequencies = np.array(raw_frequencies)

        # Convert THz to meV
        frequencies = frequencies * units_phonopy.THzToEv * 1000

        # Process k-points
        raw_kpoints = band_dict['qpoints']
        if isinstance(raw_kpoints, list):
            kpoints_flat = []
            for kpt_group in raw_kpoints:
                kpt_array = np.array(kpt_group)
                if kpt_array.ndim == 1:
                    kpoints_flat.append(kpt_array)
                elif kpt_array.ndim == 2:
                    kpoints_flat.extend(kpt_array)
            kpoints_band = np.array(kpoints_flat)
        else:
            kpoints_band = np.array(raw_kpoints)

        # Calculate DOS
        logger.log("  Calculating phonon DOS...")
        mesh = [20, 20, 20] if len(atoms) > 100 else [30, 30, 30]
        phonon.run_mesh(mesh)
        phonon.run_total_dos()
        dos_dict = phonon.get_total_dos_dict()
        dos_frequencies = dos_dict['frequency_points'] * units_phonopy.THzToEv * 1000
        dos_values = dos_dict['total_dos']

        # Check for imaginary modes
        valid_frequencies = frequencies[~np.isnan(frequencies)]
        imaginary_count = np.sum(valid_frequencies < 0)
        min_frequency = np.min(valid_frequencies) if len(valid_frequencies) > 0 else 0

        # Calculate thermodynamics
        temp = phonon_params.get('temperature', 300)
        logger.log(f"  Calculating thermodynamics at {{temp}} K...")
        
        phonon.run_thermal_properties(t_step=10, t_max=1500, t_min=0)
        thermal_dict = phonon.get_thermal_properties_dict()
        temps = np.array(thermal_dict['temperatures'])
        temp_idx = np.argmin(np.abs(temps - temp))

        thermo_props = {{
            'temperature': float(temps[temp_idx]),
            'zero_point_energy': float(thermal_dict['zero_point_energy']),
            'internal_energy': float(thermal_dict['internal_energy'][temp_idx]),
            'heat_capacity': float(thermal_dict['heat_capacity'][temp_idx]),
            'entropy': float(thermal_dict['entropy'][temp_idx]),
            'free_energy': float(thermal_dict['free_energy'][temp_idx])
        }}

        logger.log(f"✅ Phonon calculation completed for {{structure_name}}")

        return {{
            'success': True,
            'frequencies': frequencies,
            'kpoints': kpoints_band,
            'dos_energies': dos_frequencies,
            'dos': dos_values,
            'thermodynamics': thermo_props,
            'supercell_size': tuple([supercell_matrix[i][i] for i in range(3)]),
            'imaginary_modes': int(imaginary_count),
            'min_frequency': float(min_frequency),
            'method': 'Pymatgen+Phonopy'
        }}

    except Exception as e:
        logger.log(f"❌ Phonon calculation failed for {{structure_name}}: {{str(e)}}")
        return {{'success': False, 'error': str(e)}}


def calculate_elastic_properties(atoms, calculator, elastic_params, logger, structure_name):
    if not ELASTIC_AVAILABLE:
        logger.log("❌ Elastic calculations not available")
        return {{'success': False, 'error': 'Elastic calculations not available'}}

    try:
        logger.log(f"Starting elastic calculation for {{structure_name}}")
        atoms.calc = calculator

        # Brief optimization
        logger.log("  Running brief pre-elastic optimization...")
        temp_atoms = atoms.copy()
        temp_atoms.calc = calculator
        temp_optimizer = LBFGS(temp_atoms, logfile=None)
        temp_optimizer.run(fmax=0.015, steps=100)
        atoms = temp_atoms

        strain_magnitude = elastic_params.get('strain_magnitude', 0.01)
        logger.log(f"  Using strain magnitude: {{strain_magnitude * 100:.1f}}%")

        # Initialize elastic tensor
        C = np.zeros((6, 6))
        original_cell = atoms.get_cell().copy()

        # Define strain tensors
        strain_tensors = []
        for i in range(3):
            strain = np.zeros((3, 3))
            strain[i, i] = strain_magnitude
            strain_tensors.append(strain)

        for i, j in [(1, 2), (0, 2), (0, 1)]:
            strain = np.zeros((3, 3))
            strain[i, j] = strain[j, i] = strain_magnitude / 2
            strain_tensors.append(strain)

        # Apply strains and calculate elastic constants
        for strain_idx, strain_tensor in enumerate(strain_tensors):
            # Positive strain
            deformed_cell = original_cell @ (np.eye(3) + strain_tensor)
            atoms.set_cell(deformed_cell, scale_atoms=True)
            stress_pos = atoms.get_stress(voigt=True)

            # Negative strain
            deformed_cell = original_cell @ (np.eye(3) - strain_tensor)
            atoms.set_cell(deformed_cell, scale_atoms=True)
            stress_neg = atoms.get_stress(voigt=True)

            # Calculate elastic constants
            stress_derivative = (stress_pos - stress_neg) / (2 * strain_magnitude)
            C[strain_idx, :] = stress_derivative

            # Restore original cell
            atoms.set_cell(original_cell, scale_atoms=True)

        # Convert to GPa
        eV_to_GPa = 160.2176
        C_GPa = C * eV_to_GPa

        # Calculate moduli
        K_voigt = (C_GPa[0, 0] + C_GPa[1, 1] + C_GPa[2, 2] + 2 * (C_GPa[0, 1] + C_GPa[0, 2] + C_GPa[1, 2])) / 9
        G_voigt = (C_GPa[0, 0] + C_GPa[1, 1] + C_GPa[2, 2] - C_GPa[0, 1] - C_GPa[0, 2] - C_GPa[1, 2] + 3 * (
                C_GPa[3, 3] + C_GPa[4, 4] + C_GPa[5, 5])) / 15

        try:
            S_GPa = np.linalg.inv(C_GPa)
            K_reuss = 1 / (S_GPa[0, 0] + S_GPa[1, 1] + S_GPa[2, 2] + 2 * (S_GPa[0, 1] + S_GPa[0, 2] + S_GPa[1, 2]))
            G_reuss = 15 / (4 * (S_GPa[0, 0] + S_GPa[1, 1] + S_GPa[2, 2]) - 4 * (
                    S_GPa[0, 1] + S_GPa[0, 2] + S_GPa[1, 2]) + 3 * (S_GPa[3, 3] + S_GPa[4, 4] + S_GPa[5, 5]))
            K_hill = (K_voigt + K_reuss) / 2
            G_hill = (G_voigt + G_reuss) / 2
        except:
            K_reuss = G_reuss = K_hill = G_hill = None

        K = K_hill if K_hill else K_voigt
        G = G_hill if G_hill else G_voigt

        E = (9 * K * G) / (3 * K + G)
        nu = (3 * K - 2 * G) / (2 * (3 * K + G))

        # Estimate density
        total_mass_amu = np.sum(atoms.get_masses())
        volume = atoms.get_volume()
        density = (total_mass_amu * 1.66053906660) / volume

        # Wave velocities
        density_kg_m3 = density * 1000
        v_l = np.sqrt((K + 4 * G / 3) * 1e9 / density_kg_m3)
        v_t = np.sqrt(G * 1e9 / density_kg_m3)
        v_avg = ((1 / v_l ** 3 + 2 / v_t ** 3) / 3) ** (-1 / 3)

        # Debye temperature
        h = 6.626e-34
        kB = 1.381e-23
        N_atoms = len(atoms)
        total_mass_kg = np.sum(atoms.get_masses()) * 1.66054e-27
        theta_D = (h / kB) * v_avg * (3 * N_atoms * density_kg_m3 / (4 * np.pi * total_mass_kg)) ** (1 / 3)

        # Stability check
        eigenvals = np.linalg.eigvals(C_GPa)
        stable = bool(np.all(eigenvals > 0) and np.linalg.det(C_GPa) > 0)

        logger.log(f"✅ Elastic calculation completed for {{structure_name}}")

        return {{
            'success': True,
            'elastic_tensor': C_GPa.tolist(),
            'bulk_modulus': {{'voigt': K_voigt, 'reuss': K_reuss, 'hill': K_hill}},
            'shear_modulus': {{'voigt': G_voigt, 'reuss': G_reuss, 'hill': G_hill}},
            'youngs_modulus': E,
            'poisson_ratio': nu,
            'wave_velocities': {{'longitudinal': v_l, 'transverse': v_t, 'average': v_avg}},
            'debye_temperature': theta_D,
            'density': density,
            'mechanical_stability': {{'mechanically_stable': stable}},
            'strain_magnitude': strain_magnitude
        }}

    except Exception as e:
        logger.log(f"❌ Elastic calculation failed for {{structure_name}}: {{str(e)}}")
        return {{'success': False, 'error': str(e)}}


def main():
    # Configuration from GUI settings
    MODEL_SIZE = "{model_size}"
    DEVICE = "{device}"
    CALC_TYPE = "{calc_type}"
    CALC_FORMATION_ENERGY = {calc_formation_energy}
    STRUCTURE_FILES = {structure_files}
    OPTIMIZATION_PARAMS = {optimization_params}
    PHONON_PARAMS = {phonon_params}
    ELASTIC_PARAMS = {elastic_params}

    # Initialize logger and directories
    logger = CalculationLogger("results/calculation.log")
    logger.log("="*60)
    logger.log("MACE Batch Structure Calculator")
    logger.log("="*60)
    logger.log(f"Model: {{MODEL_SIZE}}")
    logger.log(f"Device: {{DEVICE}}")
    logger.log(f"Calculation Type: {{CALC_TYPE}}")
    logger.log("")

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
            test_energy = atoms.get_potential_energy()
            logger.log(f"✅ Calculator test successful - Initial energy: {{test_energy:.6f}} eV")

            energy = None
            final_structure = structure

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
                        
                        self.trajectory.append({{
                            'step': self.step_count,
                            'energy': energy,
                            'max_force': max_force,
                            'positions': optimizer.atoms.positions.copy(),
                            'cell': optimizer.atoms.cell.array.copy(),
                            'symbols': optimizer.atoms.get_chemical_symbols(),
                            'forces': forces.copy()
                        }})
                        
                        logger.log(f"  Step {{self.step_count}}: E={{energy:.6f}} eV, F_max={{max_force:.4f}} eV/Å")

                tracker = OptTracker()
                optimizer = LBFGS(atoms, logfile=None) if OPTIMIZATION_PARAMS['optimizer'] == "LBFGS" else BFGS(atoms, logfile=None)
                optimizer.attach(lambda: tracker(optimizer), interval=1)
                optimizer.run(fmax=OPTIMIZATION_PARAMS['fmax'], steps=OPTIMIZATION_PARAMS['max_steps'])

                energy = atoms.get_potential_energy()
                final_forces = atoms.get_forces()
                max_final_force = np.max(np.linalg.norm(final_forces, axis=1))
                force_converged = max_final_force < OPTIMIZATION_PARAMS['fmax']
                
                logger.log(f"✅ Optimization {{'CONVERGED' if force_converged else 'MAX STEPS'}}: Final energy = {{energy:.6f}} eV")

                # Save optimized structure
                optimized_structure = Structure(
                    lattice=atoms.cell[:],
                    species=[atom.symbol for atom in atoms],
                    coords=atoms.positions,
                    coords_are_cartesian=True
                )
                final_structure = optimized_structure
                
                with open(f"optimized_structures/optimized_{{name}}", 'w') as f:
                    f.write(optimized_structure.to(fmt="poscar"))

            elif CALC_TYPE == "Phonon Calculation":
                phonon_results = calculate_phonons_pymatgen(atoms, calculator, PHONON_PARAMS, logger, name)
                if phonon_results['success']:
                    energy = atoms.get_potential_energy()
                    # Save phonon data
                    with open(f"phonon_data/phonon_{{name.replace('.', '_')}}.json", 'w') as f:
                        json.dump(phonon_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x) if isinstance(x, np.number) else x)

            elif CALC_TYPE == "Elastic Properties":
                elastic_results = calculate_elastic_properties(atoms, calculator, ELASTIC_PARAMS, logger, name)
                if elastic_results['success']:
                    energy = atoms.get_potential_energy()
                    # Save elastic data
                    with open(f"elastic_data/elastic_{{name.replace('.', '_')}}.json", 'w') as f:
                        json.dump(elastic_results, f, indent=2)

            # Calculate formation energy
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
        f.write(f"Calculation Type: {{CALC_TYPE}}\\n\\n")

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

    logger.log("\\n✅ Results summary saved to results/results_summary.txt")
    logger.log("✅ Calculation completed successfully!")


if __name__ == "__main__":
    main()
'''

    return script