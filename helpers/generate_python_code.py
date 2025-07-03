"""
Python Script Generator for MACE Calculations
This module generates standalone Python scripts for MACE molecular dynamics calculations.
"""

import json
from datetime import datetime


def generate_python_script(structures, calc_type, model_size, device, optimization_params,
                           phonon_params, elastic_params, calc_formation_energy, selected_model_key=None):
    """
    Generate a complete Python script for MACE calculations with all parameters properly configured.
    """

    structure_creation_code = _generate_structure_creation_code(structures)
    calculator_setup_code = _generate_calculator_setup_code(
        model_size, device, selected_model_key)

    if calc_type == "Energy Only":
        calculation_code = _generate_energy_only_code(calc_formation_energy)
    elif calc_type == "Geometry Optimization":
        calculation_code = _generate_optimization_code(
            optimization_params, calc_formation_energy)
    elif calc_type == "Phonon Calculation":
        calculation_code = _generate_phonon_code(
            phonon_params, optimization_params, calc_formation_energy)
    elif calc_type == "Elastic Properties":
        calculation_code = _generate_elastic_code(
            elastic_params, optimization_params, calc_formation_energy)
    else:
        calculation_code = _generate_energy_only_code(calc_formation_energy)

    script = f"""#!/usr/bin/env python3
\"\"\"
MACE Calculation Script
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Calculation Type: {calc_type}
Model: {model_size}
Device: {device}
\"\"\"

import os
import time
import numpy as np
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# Set threading before other imports
os.environ['OMP_NUM_THREADS'] = '4'

import torch
torch.set_num_threads(4)

# ASE imports
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, LBFGS
from ase.constraints import FixAtoms, ExpCellFilter, UnitCellFilter

# PyMatGen imports
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# MACE imports
try:
    from mace.calculators import mace_mp, mace_off
    MACE_AVAILABLE = True
except ImportError:
    try:
        from mace.calculators import MACECalculator
        MACE_AVAILABLE = True
    except ImportError:
        MACE_AVAILABLE = False
        print("❌ MACE not available. Please install with: pip install mace-torch")
        exit(1)

{_generate_utility_functions()}

def main():
    start_time = time.time()
    print("🚀 Starting MACE calculation script...")
    print(f"📅 Timestamp: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
    print(f"🔬 Calculation type: {calc_type}")
    print(f"🤖 Model: {model_size}")
    print(f"💻 Device: {device}")
    print(f"🧵 CPU threads: {{os.environ.get('OMP_NUM_THREADS', 'default')}}")

    # Create output directories
    Path("optimized_structures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # Create structure files
    print("\\n📁 Creating structure files...")
{structure_creation_code}

    # Setup calculator
    print("\\n🔧 Setting up MACE calculator...")
{calculator_setup_code}

    # Run calculations
    print("\\n⚡ Starting calculations...")
    calc_start_time = time.time()
{calculation_code}

    total_time = time.time() - start_time
    calc_time = time.time() - calc_start_time
    print(f"\\n✅ All calculations completed!")
    print(f"⏱️ Total time: {{total_time/60:.1f}} minutes")
    print(f"⏱️ Calculation time: {{calc_time/60:.1f}} minutes")
    print("📊 Check the results/ directory for output files")

if __name__ == "__main__":
    main()
"""

    return script


def _generate_structure_creation_code(structures):
    """Generate code to create POSCAR files from structures."""
    code_lines = []

    for name, structure_data in structures.items():
        # Always use .vasp extension for consistency (without using os.path)
        if '.' in name:
            # Remove extension by splitting on last dot
            base_name = name.rsplit('.', 1)[0]
        else:
            base_name = name
        vasp_filename = f"{base_name}.vasp"

        # Check if structure_data is already a pymatgen Structure
        if hasattr(structure_data, 'to'):
            # Already a pymatgen Structure object
            poscar_content = structure_data.to(fmt="poscar")
            poscar_content_escaped = repr(poscar_content)

            code_lines.append(f"""    with open("{vasp_filename}", "w") as f:
        f.write({poscar_content_escaped})
    print(f"✅ Created {{'{vasp_filename}'}}")""")
        else:
            # Assume it's a file path or needs to be converted
            code_lines.append(f"""    try:
        # Try to read the structure file
        from pathlib import Path
        if Path("{structure_data}").exists():
            # Read from file using ASE, then convert to pymatgen
            from ase.io import read
            from pymatgen.io.ase import AseAtomsAdaptor
            
            ase_atoms = read("{structure_data}")
            adaptor = AseAtomsAdaptor()
            pmg_structure = adaptor.get_structure(ase_atoms)
            poscar_content = pmg_structure.to(fmt="poscar")
        else:
            # Assume it's already POSCAR content
            poscar_content = "{structure_data}"
        
        with open("{vasp_filename}", "w") as f:
            f.write(poscar_content)
        print(f"✅ Created {{'{vasp_filename}'}}")
    except Exception as e:
        print(f"❌ Failed to create {{'{vasp_filename}'}}: {{e}}")""")

    return "\n".join(code_lines)


def _generate_calculator_setup_code(model_size, device, selected_model_key=None):
    """Generate calculator setup code."""
    # Check if this is a MACE-OFF model by looking at the selected model key
    is_mace_off = selected_model_key is not None and "OFF" in selected_model_key

    if is_mace_off:
        calc_code = f'''    device = "{device}"
    print(f"🔧 Initializing MACE-OFF calculator on {{device}}...")
    try:
        calculator = mace_off(
            model="{model_size}", default_dtype="float64", device=device)
        print(f"✅ MACE-OFF calculator initialized successfully on {{device}}")
    except Exception as e:
        print(f"❌ MACE-OFF initialization failed on {{device}}: {{e}}")
        if device == "cuda":
            print("⚠️ GPU initialization failed, falling back to CPU...")
            try:
                calculator = mace_off(
                    model="{model_size}", default_dtype="float64", device="cpu")
                print("✅ MACE-OFF calculator initialized successfully on CPU (fallback)")
            except Exception as cpu_error:
                print(f"❌ CPU fallback also failed: {{cpu_error}}")
                raise cpu_error
        else:
            raise e'''
    else:
        calc_code = f'''    device = "{device}"
    print(f"🔧 Initializing MACE-MP calculator on {{device}}...")
    try:
        calculator = mace_mp(
            model="{model_size}", dispersion=False, default_dtype="float64", device=device)
        print(f"✅ MACE-MP calculator initialized successfully on {{device}}")
    except Exception as e:
        print(f"❌ MACE-MP initialization failed on {{device}}: {{e}}")
        if device == "cuda":
            print("⚠️ GPU initialization failed, falling back to CPU...")
            try:
                calculator = mace_mp(
                    model="{model_size}", dispersion=False, default_dtype="float64", device="cpu")
                print("✅ MACE-MP calculator initialized successfully on CPU (fallback)")
            except Exception as cpu_error:
                print(f"❌ CPU fallback also failed: {{cpu_error}}")
                raise cpu_error
        else:
            raise e'''

    return calc_code


def _generate_energy_only_code(calc_formation_energy):
    """Generate code for energy-only calculations."""
    code = '''    structure_files = [f for f in os.listdir(".") if f.startswith("POSCAR") or f.endswith(".vasp")]
    results = []
    print(f"📊 Found {len(structure_files)} structure files")

    reference_energies = {}'''

    if calc_formation_energy:
        code += '''
    print("🔬 Calculating atomic reference energies...")
    all_elements = set()
    for filename in structure_files:
        atoms = read(filename)
        for symbol in atoms.get_chemical_symbols():
            all_elements.add(symbol)

    print(f"🧪 Found elements: {', '.join(sorted(all_elements))}")
    
    for i, element in enumerate(sorted(all_elements)):
        print(f"  📍 Calculating reference for {element} ({i+1}/{len(all_elements)})...")
        atom = Atoms(element, positions=[(0, 0, 0)], cell=[20, 20, 20], pbc=True)
        atom.calc = calculator
        reference_energies[element] = atom.get_potential_energy()
        print(f"  ✅ {element}: {reference_energies[element]:.6f} eV")'''

    code += '''

    for i, filename in enumerate(structure_files):
        print(f"\\n📊 Processing structure {i+1}/{len(structure_files)}: {filename}")
        structure_start_time = time.time()
        try:
            atoms = read(filename)
            atoms.calc = calculator
            
            print(f"  🔬 Calculating energy for {len(atoms)} atoms...")
            energy = atoms.get_potential_energy()

            result = {
                "structure": filename,
                "energy_eV": energy,
                "calculation_type": "energy_only",
                "num_atoms": len(atoms)
            }'''

    if calc_formation_energy:
        code += '''

            formation_energy = calculate_formation_energy(energy, atoms, reference_energies)
            result["formation_energy_eV_per_atom"] = formation_energy

            print(f"  ✅ Energy: {energy:.6f} eV")
            if formation_energy is not None:
                print(f"  ✅ Formation energy: {formation_energy:.6f} eV/atom")
            else:
                print(f"  ⚠️ Could not calculate formation energy")'''
    else:
        code += '''
            print(f"  ✅ Energy: {energy:.6f} eV")'''

    code += '''
            structure_time = time.time() - structure_start_time
            print(f"  ⏱️ Structure time: {structure_time:.1f}s")


            results.append(result)
            
            # Save results after each structure completes
            df_results = pd.DataFrame(results)
            df_results.to_csv("results/energy_results.csv", index=False)
            print(f"  💾 Results updated and saved after structure {i+1}/{len(structure_files)}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({"structure": filename, "error": str(e)})
            
            # Save results even for failed structures
            df_results = pd.DataFrame(results)
            df_results.to_csv("results/energy_results.csv", index=False)
            print(f"  💾 Results updated and saved after structure {i+1}/{len(structure_files)} (with error)")

    # Final summary save
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/energy_results.csv", index=False)

    print(f"\\n💾 Saved results to results/energy_results.csv")

    with open("results/energy_summary.txt", "w") as f:
        f.write("MACE Energy Calculation Results\\n")
        f.write("=" * 40 + "\\n\\n")
        for result in results:
            if "error" not in result:
                f.write(f"Structure: {result['structure']}\\n")
                f.write(f"Energy: {result['energy_eV']:.6f} eV\\n")
                f.write(f"Atoms: {result['num_atoms']}\\n")'''

    if calc_formation_energy:
        code += '''
                if "formation_energy_eV_per_atom" in result and result["formation_energy_eV_per_atom"] is not None:
                    f.write(f"Formation Energy: {result['formation_energy_eV_per_atom']:.6f} eV/atom\\n")'''

    code += '''
                f.write("\\n")
            else:
                f.write(f"Structure: {result['structure']} - ERROR: {result['error']}\\n\\n")
    
    print(f"💾 Saved summary to results/energy_summary.txt")
'''
    return code


def _generate_elastic_code(elastic_params, optimization_params, calc_formation_energy):
    """Generate code for elastic property calculations."""
    strain_magnitude = elastic_params.get('strain_magnitude', 0.01)

    code = f'''    structure_files = [f for f in os.listdir(".") if f.startswith("POSCAR") or f.endswith(".vasp")]
    results = []
    print(f"🔧 Found {{len(structure_files)}} structure files for elastic calculations")

    strain_magnitude = {strain_magnitude}
    pre_opt_steps = {optimization_params.get('max_steps', 100)}
    print(f"⚙️ Using strain magnitude: {{strain_magnitude*100:.1f}}%")

    reference_energies = {{}}'''

    if calc_formation_energy:
        code += '''
    print("🔬 Calculating atomic reference energies...")
    all_elements = set()
    for filename in structure_files:
        atoms = read(filename)
        for symbol in atoms.get_chemical_symbols():
            all_elements.add(symbol)

    print(f"🧪 Found elements: {', '.join(sorted(all_elements))}")
    
    for i, element in enumerate(sorted(all_elements)):
        print(f"  📍 Calculating reference for {element} ({i+1}/{len(all_elements)})...")
        atom = Atoms(element, positions=[(0, 0, 0)], cell=[20, 20, 20], pbc=True)
        atom.calc = calculator
        reference_energies[element] = atom.get_potential_energy()
        print(f"  ✅ {element}: {reference_energies[element]:.6f} eV")'''

    code += '''

    for i, filename in enumerate(structure_files):
        print(f"\\n🔧 Processing structure {i+1}/{len(structure_files)}: {filename}")
        structure_start_time = time.time()
        try:
            atoms = read(filename)
            atoms.calc = calculator
            print(f"  📊 Structure has {len(atoms)} atoms")

            print("  🔧 Running pre-optimization for stability...")
            temp_atoms = atoms.copy()
            temp_atoms.calc = calculator
            temp_optimizer = LBFGS(temp_atoms, logfile=None)
            temp_optimizer.run(fmax=0.015, steps=pre_opt_steps)
            atoms = temp_atoms
            print(f"  ✅ Pre-optimization completed in {temp_optimizer.nsteps} steps")

            print("  ⚖️ Calculating equilibrium energy and stress...")
            E0 = atoms.get_potential_energy()
            stress0 = atoms.get_stress(voigt=True)

            print(f"    Equilibrium energy: {E0:.6f} eV")
            print(f"    Equilibrium stress: {np.max(np.abs(stress0)):.6f} GPa")

            C = np.zeros((6, 6))
            original_cell = atoms.get_cell().copy()
            volume = atoms.get_volume()

            print(f"  📏 Applying strains and calculating elastic constants...")

            strain_tensors = []

            for idx in range(3):
                strain = np.zeros((3, 3))
                strain[idx, idx] = strain_magnitude
                strain_tensors.append(strain)

            shear_pairs = [(1, 2), (0, 2), (0, 1)]
            for i, j in shear_pairs:
                strain = np.zeros((3, 3))
                strain[i, j] = strain[j, i] = strain_magnitude / 2
                strain_tensors.append(strain)

            for strain_idx, strain_tensor in enumerate(strain_tensors):
                print(f"    📐 Strain {strain_idx + 1}/6...")

                deformed_cell = original_cell @ (np.eye(3) + strain_tensor)
                atoms.set_cell(deformed_cell, scale_atoms=True)
                stress_pos = atoms.get_stress(voigt=True)

                deformed_cell = original_cell @ (np.eye(3) - strain_tensor)
                atoms.set_cell(deformed_cell, scale_atoms=True)
                stress_neg = atoms.get_stress(voigt=True)

                stress_derivative = (stress_pos - stress_neg) / (2 * strain_magnitude)
                C[strain_idx, :] = stress_derivative

                atoms.set_cell(original_cell, scale_atoms=True)

            eV_to_GPa = 160.2176
            C_GPa = C * eV_to_GPa

            print("  📊 Calculating elastic moduli...")

            K_voigt = (C_GPa[0, 0] + C_GPa[1, 1] + C_GPa[2, 2] +
                       2*(C_GPa[0, 1] + C_GPa[0, 2] + C_GPa[1, 2])) / 9
            G_voigt = (C_GPa[0, 0] + C_GPa[1, 1] + C_GPa[2, 2] - C_GPa[0, 1] - C_GPa[0, 2] - C_GPa[1, 2] +
                       3*(C_GPa[3, 3] + C_GPa[4, 4] + C_GPa[5, 5])) / 15

            try:
                S_GPa = np.linalg.inv(C_GPa)
                K_reuss = 1 / (S_GPa[0, 0] + S_GPa[1, 1] + S_GPa[2, 2] +
                               2*(S_GPa[0, 1] + S_GPa[0, 2] + S_GPa[1, 2]))
                G_reuss = 15 / (4*(S_GPa[0, 0] + S_GPa[1, 1] + S_GPa[2, 2]) - 4*(S_GPa[0, 1] + S_GPa[0, 2] + S_GPa[1, 2]) +
                               3*(S_GPa[3, 3] + S_GPa[4, 4] + S_GPa[5, 5]))
                K_hill = (K_voigt + K_reuss) / 2
                G_hill = (G_voigt + G_reuss) / 2
                reuss_available = True
            except np.linalg.LinAlgError:
                print("    ⚠️ Elastic tensor is singular - using Voigt averages only")
                K_reuss = G_reuss = K_hill = G_hill = None
                reuss_available = False

            K = K_hill if K_hill is not None else K_voigt
            G = G_hill if G_hill is not None else G_voigt

            E = (9 * K * G) / (3 * K + G)
            nu = (3 * K - 2 * G) / (2 * (3 * K + G))

            total_mass_amu = np.sum(atoms.get_masses())
            density = (total_mass_amu * 1.66053906660) / volume
            density_kg_m3 = density * 1000

            v_l = np.sqrt((K + 4*G/3) * 1e9 / density_kg_m3)
            v_t = np.sqrt(G * 1e9 / density_kg_m3)
            v_avg = ((1/v_l**3 + 2/v_t**3) / 3)**(-1/3)

            h = 6.626e-34
            kB = 1.381e-23
            N_atoms = len(atoms)
            total_mass_kg = total_mass_amu * 1.66054e-27
            theta_D = (h / kB) * v_avg * (3 * N_atoms * density_kg_m3 / (4 * np.pi * total_mass_kg))**(1/3)

            eigenvals = np.linalg.eigvals(C_GPa)
            mechanically_stable = bool(np.all(eigenvals > 0) and np.linalg.det(C_GPa) > 0)

            # Create result dictionary with flat structure to avoid unhashable type error
            result = {{
                "structure": filename,
                "energy_eV": float(E0),
                "calculation_type": "elastic_properties",
                "elastic_tensor_GPa": C_GPa.tolist(),
                "bulk_modulus_voigt_GPa": float(K_voigt),
                "bulk_modulus_reuss_GPa": float(K_reuss) if reuss_available else None,
                "bulk_modulus_hill_GPa": float(K_hill) if reuss_available else None,
                "shear_modulus_voigt_GPa": float(G_voigt),
                "shear_modulus_reuss_GPa": float(G_reuss) if reuss_available else None,
                "shear_modulus_hill_GPa": float(G_hill) if reuss_available else None,
                "youngs_modulus_GPa": float(E),
                "poisson_ratio": float(nu),
                "density_g_cm3": float(density),
                "longitudinal_velocity_ms": float(v_l),
                "transverse_velocity_ms": float(v_t),
                "average_velocity_ms": float(v_avg),
                "debye_temperature_K": float(theta_D),
                "mechanically_stable": bool(mechanically_stable),
                "strain_magnitude": float(strain_magnitude),
                "num_atoms": int(len(atoms))
            }}'''

    if calc_formation_energy:
        code += '''

            formation_energy = calculate_formation_energy(E0, atoms, reference_energies)
            result["formation_energy_eV_per_atom"] = formation_energy'''

    code += '''

            # Create separate CSV data for detailed output
            elastic_data_dict = {{
                "structure_name": [filename],
                "energy_eV": [float(E0)],
                "C11_GPa": [float(C_GPa[0, 0])],
                "C22_GPa": [float(C_GPa[1, 1])],
                "C33_GPa": [float(C_GPa[2, 2])],
                "C44_GPa": [float(C_GPa[3, 3])],
                "C55_GPa": [float(C_GPa[4, 4])],
                "C66_GPa": [float(C_GPa[5, 5])],
                "C12_GPa": [float(C_GPa[0, 1])],
                "C13_GPa": [float(C_GPa[0, 2])],
                "C23_GPa": [float(C_GPa[1, 2])],
                "bulk_modulus_voigt_GPa": [float(K_voigt)],
                "bulk_modulus_reuss_GPa": [float(K_reuss) if reuss_available else None],
                "bulk_modulus_hill_GPa": [float(K_hill) if reuss_available else None],
                "shear_modulus_voigt_GPa": [float(G_voigt)],
                "shear_modulus_reuss_GPa": [float(G_reuss) if reuss_available else None],
                "shear_modulus_hill_GPa": [float(G_hill) if reuss_available else None],
                "youngs_modulus_GPa": [float(E)],
                "poisson_ratio": [float(nu)],
                "density_g_cm3": [float(density)],
                "longitudinal_velocity_ms": [float(v_l)],
                "transverse_velocity_ms": [float(v_t)],
                "average_velocity_ms": [float(v_avg)],
                "debye_temperature_K": [float(theta_D)],
                "mechanically_stable": [bool(mechanically_stable)],
                "strain_magnitude": [float(strain_magnitude)],
                "num_atoms": [int(len(atoms))]
            }}'''

    if calc_formation_energy:
        code += '''
            if formation_energy is not None:
                elastic_data_dict["formation_energy_eV_per_atom"] = [formation_energy]'''

    code += '''

            df_elastic = pd.DataFrame(elastic_data_dict)
            df_elastic.to_csv(f"results/elastic_data_{filename.replace('.', '_')}.csv", index=False)

            elastic_tensor_df = pd.DataFrame(C_GPa,
                                           columns=['C11', 'C22', 'C33', 'C23', 'C13', 'C12'],
                                           index=['C11', 'C22', 'C33', 'C23', 'C13', 'C12'])
            elastic_tensor_df.to_csv(f"results/elastic_tensor_{filename.replace('.', '_')}.csv")

            structure_time = time.time() - structure_start_time
            print(f"  ✅ Elastic calculation completed in {structure_time:.1f}s")
            print(f"  ✅ Energy: {E0:.6f} eV")
            print(f"  ✅ Bulk modulus: {K:.1f} GPa")
            print(f"  ✅ Shear modulus: {G:.1f} GPa")
            print(f"  ✅ Young's modulus: {E:.1f} GPa")
            print(f"  ✅ Poisson's ratio: {nu:.3f}")
            print(f"  ✅ Mechanically stable: {mechanically_stable}")'''

    if calc_formation_energy:
        code += '''
            if formation_energy is not None:
                print(f"  ✅ Formation energy: {formation_energy:.6f} eV/atom")'''

    code += '''

            results.append(result)
            
            # Save results after each structure completes
            df_results = pd.DataFrame(results)
            df_results.to_csv("results/elastic_results.csv", index=False)
            print(f"  💾 Results updated and saved after structure {i+1}/{len(structure_files)}")

        except Exception as e:
            print(f"  ❌ Elastic calculation failed: {e}")
            results.append({{"structure": filename, "error": str(e)}})

            # Save results even for failed structures
            df_results = pd.DataFrame(results)
            df_results.to_csv("results/elastic_results.csv", index=False)
            print(f"  💾 Results updated and saved after structure {i+1}/{len(structure_files)} (with error)")

    # Final summary save
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/elastic_results.csv", index=False)

    print(f"\\n💾 Saved results to results/elastic_results.csv")

    with open("results/elastic_summary.txt", "w") as f:
        f.write("MACE Elastic Properties Results\\n")
        f.write("=" * 40 + "\\n\\n")
        for result in results:
            if "error" not in result:
                f.write(f"Structure: {result['structure']}\\n")
                f.write(f"Energy: {result['energy_eV']:.6f} eV\\n")
                bulk_mod = result.get('bulk_modulus_hill_GPa') or result.get('bulk_modulus_voigt_GPa')
                shear_mod = result.get('shear_modulus_hill_GPa') or result.get('shear_modulus_voigt_GPa')
                f.write(f"Bulk Modulus: {bulk_mod:.1f} GPa\\n")
                f.write(f"Shear Modulus: {shear_mod:.1f} GPa\\n")
                f.write(f"Young's Modulus: {result['youngs_modulus_GPa']:.1f} GPa\\n")
                f.write(f"Poisson's Ratio: {result['poisson_ratio']:.3f}\\n")
                f.write(f"Density: {result['density_g_cm3']:.3f} g/cm³\\n")
                f.write(f"Debye Temperature: {result['debye_temperature_K']:.1f} K\\n")
                f.write(f"Mechanically Stable: {result['mechanically_stable']}\\n")
                f.write(f"Atoms: {result['num_atoms']}\\n")
                if "formation_energy_eV_per_atom" in result and result["formation_energy_eV_per_atom"] is not None:
                    f.write(f"Formation Energy: {result['formation_energy_eV_per_atom']:.6f} eV/atom\\n")
                f.write("\\n")
            else:
                f.write(f"Structure: {result['structure']} - ERROR: {result['error']}\\n\\n")
    
    print(f"💾 Saved summary to results/elastic_summary.txt")'''

    return code


def _generate_utility_functions():
    """Generate utility functions needed by the script."""
    return '''

def wrap_positions_in_cell(atoms):
    wrapped_atoms = atoms.copy()
    fractional_coords = wrapped_atoms.get_scaled_positions()
    wrapped_fractional = fractional_coords % 1.0
    wrapped_atoms.set_scaled_positions(wrapped_fractional)
    return wrapped_atoms


def get_lattice_parameters(atoms):
    cell = atoms.get_cell()
    a, b, c = np.linalg.norm(cell, axis=1)
    
    def angle_between_vectors(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    alpha = angle_between_vectors(cell[1], cell[2])
    beta = angle_between_vectors(cell[0], cell[2])
    gamma = angle_between_vectors(cell[0], cell[1])
    
    volume = np.abs(np.linalg.det(cell))
    
    return {
        'a': a, 'b': b, 'c': c,
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'volume': volume
    }


def get_atomic_composition(atoms):
    symbols = atoms.get_chemical_symbols()
    total_atoms = len(symbols)
    
    composition = {}
    for symbol in symbols:
        composition[symbol] = composition.get(symbol, 0) + 1
    
    concentrations = {}
    for element, count in composition.items():
        concentrations[element] = (count / total_atoms) * 100
    
    return composition, concentrations

def append_optimization_summary(filename, structure_name, initial_atoms, final_atoms, 
                               initial_energy, final_energy, convergence_status, steps, selective_dynamics=None):
    
    initial_lattice = get_lattice_parameters(initial_atoms)
    final_lattice = get_lattice_parameters(final_atoms)
    composition, concentrations = get_atomic_composition(final_atoms)
    
    energy_change = final_energy - initial_energy
    volume_change = ((final_lattice['volume'] - initial_lattice['volume']) / initial_lattice['volume']) * 100
    
    a_change = ((final_lattice['a'] - initial_lattice['a']) / initial_lattice['a']) * 100
    b_change = ((final_lattice['b'] - initial_lattice['b']) / initial_lattice['b']) * 100
    c_change = ((final_lattice['c'] - initial_lattice['c']) / initial_lattice['c']) * 100
    
    alpha_change = final_lattice['alpha'] - initial_lattice['alpha']
    beta_change = final_lattice['beta'] - initial_lattice['beta']
    gamma_change = final_lattice['gamma'] - initial_lattice['gamma']
    
    comp_formula = "".join([f"{element}{composition[element]}" for element in sorted(composition.keys())])
    
    elements = sorted(composition.keys())
    conc_values = [concentrations[element] for element in elements]
    conc_string = " ".join([f"{element}:{conc:.1f}" for element, conc in zip(elements, conc_values)])
    
    constraint_info = "None"
    if selective_dynamics is not None:
        total_atoms = len(selective_dynamics)
        completely_fixed = sum(1 for flags in selective_dynamics if not any(flags))
        partially_fixed = sum(1 for flags in selective_dynamics if not all(flags) and any(flags))
        free_atoms = sum(1 for flags in selective_dynamics if all(flags))
        
        constraint_parts = []
        if completely_fixed > 0:
            constraint_parts.append(f"{completely_fixed}complete")
        if partially_fixed > 0:
            constraint_parts.append(f"{partially_fixed}partial")
        if free_atoms > 0:
            constraint_parts.append(f"{free_atoms}free")
        
        constraint_info = ",".join(constraint_parts)
    
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a') as f:
        if not file_exists:
            header = "Structure,Formula,Atoms,Composition,Steps,Convergence,E_initial_eV,E_final_eV,E_change_eV,E_per_atom_eV,a_init_A,b_init_A,c_init_A,alpha_init_deg,beta_init_deg,gamma_init_deg,V_init_A3,a_final_A,b_final_A,c_final_A,alpha_final_deg,beta_final_deg,gamma_final_deg,V_final_A3,a_change_percent,b_change_percent,c_change_percent,alpha_change_deg,beta_change_deg,gamma_change_deg,V_change_percent"
            f.write(header + "\\n")
        
        line = f"{structure_name},{comp_formula},{len(final_atoms)},{conc_string},{steps},{convergence_status},{initial_energy:.6f},{final_energy:.6f},{energy_change:.6f},{final_energy/len(final_atoms):.6f},{initial_lattice['a']:.6f},{initial_lattice['b']:.6f},{initial_lattice['c']:.6f},{initial_lattice['alpha']:.3f},{initial_lattice['beta']:.3f},{initial_lattice['gamma']:.3f},{initial_lattice['volume']:.6f},{final_lattice['a']:.6f},{final_lattice['b']:.6f},{final_lattice['c']:.6f},{final_lattice['alpha']:.3f},{final_lattice['beta']:.3f},{final_lattice['gamma']:.3f},{final_lattice['volume']:.6f},{a_change:.3f},{b_change:.3f},{c_change:.3f},{alpha_change:.3f},{beta_change:.3f},{gamma_change:.3f},{volume_change:.3f}"
        f.write(line + "\\n")


def read_poscar_with_selective_dynamics(filename):
    atoms = read(filename)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    selective_dynamics = None
    if len(lines) > 7:
        line_7 = lines[7].strip().upper()
        if line_7.startswith('S'):
            selective_dynamics = []
            coord_start = 9
            
            for i in range(coord_start, len(lines)):
                line = lines[i].strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        flags = [parts[j].upper() == 'T' for j in [3, 4, 5]]
                        selective_dynamics.append(flags)
                    except (IndexError, ValueError):
                        break
                elif len(parts) == 3:
                    break
    
    return atoms, selective_dynamics


def write_poscar_with_selective_dynamics(atoms, filename, selective_dynamics=None, comment="Optimized structure"):
    if selective_dynamics is not None and len(selective_dynamics) == len(atoms):
        with open(filename, 'w') as f:
            f.write(f"{comment}\\n")
            f.write("1.0\\n")
            
            cell = atoms.get_cell()
            for i in range(3):
                f.write(f"  {cell[i][0]:16.12f}  {cell[i][1]:16.12f}  {cell[i][2]:16.12f}\\n")
            
            symbols = atoms.get_chemical_symbols()
            unique_symbols = []
            symbol_counts = []
            for symbol in symbols:
                if symbol not in unique_symbols:
                    unique_symbols.append(symbol)
                    symbol_counts.append(symbols.count(symbol))
            
            f.write("  " + "  ".join(unique_symbols) + "\\n")
            f.write("  " + "  ".join(map(str, symbol_counts)) + "\\n")
            
            f.write("Selective dynamics\\n")
            f.write("Direct\\n")
            
            scaled_positions = atoms.get_scaled_positions()
            for symbol in unique_symbols:
                for i, atom_symbol in enumerate(symbols):
                    if atom_symbol == symbol:
                        pos = scaled_positions[i]
                        flags = selective_dynamics[i]
                        flag_str = "  ".join(["T" if flag else "F" for flag in flags])
                        f.write(f"  {pos[0]:16.12f}  {pos[1]:16.12f}  {pos[2]:16.12f}   {flag_str}\\n")
    else:
        write(filename, atoms, format='vasp', direct=True, sort=True)
        with open(filename, 'r') as f:
            lines = f.readlines()
        with open(filename, 'w') as f:
            f.write(f"{comment}\\n")
            for line in lines[1:]:
                f.write(line)

                
def apply_selective_dynamics_constraints(atoms, selective_dynamics):
    """Apply selective dynamics as ASE constraints with support for partial fixing."""
    if selective_dynamics is None or len(selective_dynamics) != len(atoms):
        return atoms
    
    # Check if we have any constraints to apply
    has_constraints = False
    for flags in selective_dynamics:
        if not all(flags):  # If any direction is False (fixed)
            has_constraints = True
            break
    
    if not has_constraints:
        print(f"  🔄 Selective dynamics found but all atoms are completely free")
        return atoms
    
    # Apply constraints
    try:
        from ase.constraints import FixCartesian, FixAtoms
        
        constraints = []
        constraint_summary = []
        
        # Group atoms by constraint type
        completely_fixed_indices = []
        partial_constraints = []
        
        for i, flags in enumerate(selective_dynamics):
            if not any(flags):  # All directions fixed (F F F)
                completely_fixed_indices.append(i)
            elif not all(flags):  # Some directions fixed (partial)
                # ASE FixCartesian uses True for FIXED directions (opposite of VASP)
                # VASP: T=free, F=fixed
                # ASE:  T=fixed, F=free
                mask = [not flag for flag in flags]  # Invert the flags
                partial_constraints.append((i, mask))
        
        # Apply complete fixing
        if completely_fixed_indices:
            constraints.append(FixAtoms(indices=completely_fixed_indices))
            constraint_summary.append(f"{len(completely_fixed_indices)} atoms completely fixed")
        
        # Apply partial constraints - create individual FixCartesian for each atom
        if partial_constraints:
            partial_groups = {}
            for atom_idx, mask in partial_constraints:
                mask_key = tuple(mask)
                if mask_key not in partial_groups:
                    partial_groups[mask_key] = []
                partial_groups[mask_key].append(atom_idx)
            
            for mask, atom_indices in partial_groups.items():
                # Create individual FixCartesian constraints for each atom
                for atom_idx in atom_indices:
                    constraints.append(FixCartesian(atom_idx, mask))
                
                fixed_dirs = [dir_name for dir_name, is_fixed in zip(['x', 'y', 'z'], mask) if is_fixed]
                constraint_summary.append(f"{len(atom_indices)} atoms fixed in {','.join(fixed_dirs)} directions")
        
        # Apply all constraints
        if constraints:
            atoms.set_constraint(constraints)
            
            total_constrained = len(completely_fixed_indices) + len(partial_constraints)
            print(f"  📌 Applied selective dynamics to {total_constrained}/{len(atoms)} atoms:")
            for summary in constraint_summary:
                print(f"    - {summary}")
        
    except ImportError:
        # Fallback: only handle completely fixed atoms
        print(f"  ⚠️ FixCartesian not available, only applying complete atom fixing")
        fixed_indices = []
        for i, flags in enumerate(selective_dynamics):
            if not any(flags):  # All directions False (completely fixed)
                fixed_indices.append(i)
        
        if fixed_indices:
            from ase.constraints import FixAtoms
            constraint = FixAtoms(indices=fixed_indices)
            atoms.set_constraint(constraint)
            print(f"  📌 Applied complete fixing to {len(fixed_indices)}/{len(atoms)} atoms")
        else:
            print(f"  ⚠️ No completely fixed atoms found, partial constraints not supported")
    
    except Exception as e:
        # If FixCartesian fails for any reason, fall back to complete fixing only
        print(f"  ⚠️ FixCartesian failed ({str(e)}), falling back to complete atom fixing only")
        fixed_indices = []
        for i, flags in enumerate(selective_dynamics):
            if not any(flags):  # All directions False (completely fixed)
                fixed_indices.append(i)
        
        if fixed_indices:
            from ase.constraints import FixAtoms
            constraint = FixAtoms(indices=fixed_indices)
            atoms.set_constraint(constraint)
            print(f"  📌 Applied complete fixing to {len(fixed_indices)}/{len(atoms)} atoms (fallback)")
        else:
            print(f"  ⚠️ No completely fixed atoms found")
    
    return atoms


def calculate_formation_energy(structure_energy, atoms, reference_energies):
    if structure_energy is None:
        return None

    element_counts = {}
    for symbol in atoms.get_chemical_symbols():
        element_counts[symbol] = element_counts.get(symbol, 0) + 1

    total_reference_energy = 0
    for element, count in element_counts.items():
        if element not in reference_energies or reference_energies[element] is None:
            return None
        total_reference_energy += count * reference_energies[element]

    total_atoms = sum(element_counts.values())
    formation_energy_per_atom = (structure_energy - total_reference_energy) / total_atoms
    return formation_energy_per_atom


def create_cell_filter(atoms, pressure, cell_constraint, optimize_lattice, hydrostatic_strain):
    pressure_eV_A3 = pressure * 0.00624150913

    if cell_constraint == "Full cell (lattice + angles)":
        if hydrostatic_strain:
            return ExpCellFilter(atoms, scalar_pressure=pressure_eV_A3, hydrostatic_strain=True)
        else:
            return ExpCellFilter(atoms, scalar_pressure=pressure_eV_A3)
    else:
        if hydrostatic_strain:
            return UnitCellFilter(atoms, scalar_pressure=pressure_eV_A3, hydrostatic_strain=True)
        else:
            mask = [optimize_lattice['a'], optimize_lattice['b'], optimize_lattice['c'], False, False, False]
            return UnitCellFilter(atoms, mask=mask, scalar_pressure=pressure_eV_A3)


class OptimizationLogger:
    def __init__(self, filename, max_steps, output_dir="optimized_structures"):
        self.filename = filename
        self.step_count = 0
        self.max_steps = max_steps
        self.step_times = []
        self.step_start_time = time.time()
        self.output_dir = output_dir
        self.trajectory = []
        
    def __call__(self, optimizer=None):
        current_time = time.time()
        
        if self.step_count > 0:
            step_time = current_time - self.step_start_time
            self.step_times.append(step_time)
        
        self.step_count += 1
        self.step_start_time = current_time
        
        if optimizer is not None and hasattr(optimizer, 'atoms'):
            if hasattr(optimizer.atoms, 'atoms'):
                atoms = optimizer.atoms.atoms
            else:
                atoms = optimizer.atoms
                
            forces = atoms.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))
            energy = atoms.get_potential_energy()
            
            lattice = get_lattice_parameters(atoms)
            
            self.trajectory.append({
                'step': self.step_count,
                'energy': energy,
                'max_force': max_force,
                'positions': atoms.positions.copy(),
                'cell': atoms.cell.array.copy(),
                'lattice': lattice.copy()
            })
            
            if len(self.step_times) > 0:
                avg_time = np.mean(self.step_times)
                remaining_steps = max(0, self.max_steps - self.step_count)
                estimated_remaining_time = avg_time * remaining_steps
                
                if avg_time < 60:
                    avg_time_str = f"{avg_time:.1f}s"
                else:
                    avg_time_str = f"{avg_time/60:.1f}m"
                
                if estimated_remaining_time < 60:
                    remaining_time_str = f"{estimated_remaining_time:.1f}s"
                elif estimated_remaining_time < 3600:
                    remaining_time_str = f"{estimated_remaining_time/60:.1f}m"
                else:
                    remaining_time_str = f"{estimated_remaining_time/3600:.1f}h"
                
                print(f"    Step {self.step_count}: E={energy:.6f} eV, F_max={max_force:.4f} eV/Å | "
                      f"Avg: {avg_time_str}/step, Est. remaining: {remaining_time_str} ({remaining_steps} steps)")
            else:
                print(f"    Step {self.step_count}: E={energy:.6f} eV, F_max={max_force:.4f} eV/Å")
'''

def _generate_optimization_code(optimization_params, calc_formation_energy):
    """Generate code for geometry optimization with selective dynamics support."""
    optimizer = optimization_params.get('optimizer', 'BFGS')
    fmax = optimization_params.get('fmax', 0.05)
    max_steps = optimization_params.get('max_steps', 200)
    opt_type = optimization_params.get(
        'optimization_type', 'Both atoms and cell')
    cell_constraint = optimization_params.get(
        'cell_constraint', 'Lattice parameters only (fix angles)')
    pressure = optimization_params.get('pressure', 0.0)
    hydrostatic_strain = optimization_params.get('hydrostatic_strain', False)
    optimize_lattice = optimization_params.get(
        'optimize_lattice', {'a': True, 'b': True, 'c': True})

    code = f'''    structure_files = [f for f in os.listdir(".") if f.startswith("POSCAR") or f.endswith(".vasp")]
    results = []
    print(f"🔧 Found {{len(structure_files)}} structure files for optimization")

    optimizer_type = "{optimizer}"
    fmax = {fmax}
    max_steps = {max_steps}
    optimization_type = "{opt_type}"
    cell_constraint = "{cell_constraint}"
    pressure = {pressure}
    hydrostatic_strain = {hydrostatic_strain}
    optimize_lattice = {optimize_lattice}
    
    print(f"⚙️ Optimization settings:")
    print(f"  - Optimizer: {{optimizer_type}}")
    print(f"  - Force threshold: {{fmax}} eV/Å")
    print(f"  - Max steps: {{max_steps}}")
    print(f"  - Type: {{optimization_type}}")
    if pressure > 0:
        print(f"  - Pressure: {{pressure}} GPa")

    reference_energies = {{}}'''

    if calc_formation_energy:
        code += '''
    print("🔬 Calculating atomic reference energies...")
    all_elements = set()
    for filename in structure_files:
        atoms, _ = read_poscar_with_selective_dynamics(filename)
        for symbol in atoms.get_chemical_symbols():
            all_elements.add(symbol)

    print(f"🧪 Found elements: {', '.join(sorted(all_elements))}")
    
    for i, element in enumerate(sorted(all_elements)):
        print(f"  📍 Calculating reference for {element} ({i+1}/{len(all_elements)})...")
        atom = Atoms(element, positions=[(0, 0, 0)], cell=[20, 20, 20], pbc=True)
        atom.calc = calculator
        reference_energies[element] = atom.get_potential_energy()
        print(f"  ✅ {element}: {reference_energies[element]:.6f} eV")'''

    code += '''

    for i, filename in enumerate(structure_files):
        print(f"\\n🔧 Processing structure {i+1}/{len(structure_files)}: {filename}")
        structure_start_time = time.time()
        try:
            # Read structure with selective dynamics information
            atoms, selective_dynamics = read_poscar_with_selective_dynamics(filename)
            atoms.calc = calculator
            print(f"  📊 Structure has {len(atoms)} atoms")
            initial_atoms_copy = atoms.copy()
            # Apply selective dynamics constraints if present
            if selective_dynamics is not None:
                atoms = apply_selective_dynamics_constraints(atoms, selective_dynamics)
            else:
                print(f"  🔄 No selective dynamics found - all atoms free to move")

            initial_energy = atoms.get_potential_energy()
            initial_forces = atoms.get_forces()
            initial_max_force = np.max(np.linalg.norm(initial_forces, axis=1))
            print(f"  📊 Initial energy: {initial_energy:.6f} eV")
            print(f"  📊 Initial max force: {initial_max_force:.4f} eV/Å")

            if optimization_type == "Atoms only (fixed cell)":
                optimization_object = atoms
                opt_mode = "atoms_only"
                print(f"  🔒 Optimizing atoms only (fixed cell)")
            elif optimization_type == "Cell only (fixed atoms)":
                # Add constraint to fix all atoms for cell-only optimization
                existing_constraints = atoms.constraints if hasattr(atoms, 'constraints') and atoms.constraints else []
                all_fixed_constraint = FixAtoms(mask=[True] * len(atoms))
                atoms.set_constraint([all_fixed_constraint] + existing_constraints)
                optimization_object = create_cell_filter(atoms, pressure, cell_constraint, optimize_lattice, hydrostatic_strain)
                opt_mode = "cell_only"
                print(f"  🔒 Optimizing cell only (fixed atoms)")
            else:
                optimization_object = create_cell_filter(atoms, pressure, cell_constraint, optimize_lattice, hydrostatic_strain)
                opt_mode = "both"
                print(f"  🔄 Optimizing both atoms and cell")

            logger = OptimizationLogger(filename, max_steps, "optimized_structures")
            
            if optimizer_type == "LBFGS":
                optimizer = LBFGS(optimization_object, logfile=f"results/{filename}_opt.log")
            else:
                optimizer = BFGS(optimization_object, logfile=f"results/{filename}_opt.log")

            optimizer.attach(lambda: logger(optimizer), interval=1)

            print(f"  🏃 Running {optimizer_type} optimization...")
            if opt_mode == "cell_only":
                optimizer.run(fmax=0.1, steps=max_steps)
            else:
                optimizer.run(fmax=fmax, steps=max_steps)

            if hasattr(optimization_object, 'atoms'):
                final_atoms = optimization_object.atoms
            else:
                final_atoms = optimization_object

            final_energy = final_atoms.get_potential_energy()
            final_forces = final_atoms.get_forces()
            max_final_force = np.max(np.linalg.norm(final_forces, axis=1))

            force_converged = max_final_force < fmax
            if opt_mode in ["cell_only", "both"]:
                try:
                    final_stress = final_atoms.get_stress(voigt=True)
                    max_final_stress = np.max(np.abs(final_stress))
                    stress_converged = max_final_stress < 0.1
                except:
                    stress_converged = True
                    max_final_stress = 0.0
            else:
                stress_converged = True
                max_final_stress = 0.0

            if opt_mode == "atoms_only":
                convergence_status = "CONVERGED" if force_converged else "MAX_STEPS_REACHED"
            elif opt_mode == "cell_only":
                convergence_status = "CONVERGED" if stress_converged else "MAX_STEPS_REACHED"
            else:
                convergence_status = "CONVERGED" if (force_converged and stress_converged) else "MAX_STEPS_REACHED"

            # Save optimized structure with selective dynamics preserved
            base_name = filename.replace('.vasp', '').replace('POSCAR', '')
            
            output_filename = f"optimized_structures/optimized_{base_name}.vasp"
            
            print(f"  💾 Saving optimized structure to {output_filename}")
            write_poscar_with_selective_dynamics(
                final_atoms, 
                output_filename, 
                selective_dynamics, 
                f"Optimized - {convergence_status}"
            )
            
            detailed_summary_file = "results/optimization_detailed_summary.csv"
            print(f"  📊 Appending detailed summary to {detailed_summary_file}")
            append_optimization_summary(
                detailed_summary_file, 
                filename, 
                initial_atoms_copy, 
                final_atoms,      
                initial_energy, 
                final_energy, 
                convergence_status, 
                optimizer.nsteps,
                selective_dynamics
            )
            
            trajectory_filename = f"optimized_structures/trajectory_{base_name}.xyz"
            print(f"  📈 Saving optimization trajectory to {trajectory_filename}")
            
            with open(trajectory_filename, 'w') as traj_file:
                symbols = final_atoms.get_chemical_symbols()
                for step_data in logger.trajectory:
                    num_atoms = len(step_data['positions'])
                    energy = step_data['energy']
                    max_force = step_data['max_force']
                    lattice = step_data['lattice']
                    step = step_data['step']
                    cell_matrix = step_data['cell']
                    
                    lattice_string = " ".join([f"{x:.6f}" for row in cell_matrix for x in row])
                    
                    traj_file.write(f"{num_atoms}\\n")
                    
                    comment = (f'Step={step} Energy={energy:.6f} Max_Force={max_force:.6f} '
                              f'a={lattice["a"]:.6f} b={lattice["b"]:.6f} c={lattice["c"]:.6f} '
                              f'alpha={lattice["alpha"]:.3f} beta={lattice["beta"]:.3f} gamma={lattice["gamma"]:.3f} '
                              f'Volume={lattice["volume"]:.6f} '
                              f'Lattice="{lattice_string}" '
                              f'Properties=species:S:1:pos:R:3')
                    traj_file.write(f"{comment}\\n")
                    
                    for j, pos in enumerate(step_data['positions']):
                        symbol = symbols[j] if j < len(symbols) else 'X'
                        traj_file.write(f"{symbol} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\\n")

            result = {
                "structure": filename,
                "initial_energy_eV": initial_energy,
                "final_energy_eV": final_energy,
                "energy_change_eV": final_energy - initial_energy,
                "initial_max_force_eV_per_A": initial_max_force,
                "final_max_force_eV_per_A": max_final_force,
                "max_stress_GPa": max_final_stress,
                "convergence_status": convergence_status,
                "optimization_steps": optimizer.nsteps,
                "calculation_type": "geometry_optimization",
                "num_atoms": len(atoms),
                "opt_mode": opt_mode,
                "optimizer_type": optimizer_type,
                "fmax": fmax,
                "max_steps": max_steps,
                "optimization_type": optimization_type,
                "cell_constraint": cell_constraint,
                "pressure": pressure,
                "hydrostatic_strain": hydrostatic_strain,
                "has_selective_dynamics": selective_dynamics is not None,
                "num_fixed_atoms": len([flags for flags in (selective_dynamics or []) if not any(flags)]),
                "output_structure": output_filename,
                "trajectory_file": trajectory_filename,
                # Convert dict to individual fields for CSV compatibility
                "optimize_lattice_a": optimize_lattice.get('a', True) if isinstance(optimize_lattice, dict) else True,
                "optimize_lattice_b": optimize_lattice.get('b', True) if isinstance(optimize_lattice, dict) else True,
                "optimize_lattice_c": optimize_lattice.get('c', True) if isinstance(optimize_lattice, dict) else True
            }'''

    if calc_formation_energy:
        code += '''

            formation_energy = calculate_formation_energy(final_energy, final_atoms, reference_energies)
            result["formation_energy_eV_per_atom"] = formation_energy'''

    code += '''

            structure_time = time.time() - structure_start_time
            print(f"  ✅ Optimization completed: {convergence_status}")
            print(f"  ✅ Final energy: {final_energy:.6f} eV (Δ={final_energy - initial_energy:.6f} eV)")
            print(f"  ✅ Final max force: {max_final_force:.4f} eV/Å")
            if opt_mode in ["cell_only", "both"]:
                print(f"  ✅ Final max stress: {max_final_stress:.4f} GPa")
            print(f"  ✅ Steps: {optimizer.nsteps}")
            print(f"  ⏱️ Structure time: {structure_time:.1f}s")
            print(f"  💾 Saved to: {output_filename}")'''

    if calc_formation_energy:
        code += '''
            if formation_energy is not None:
                print(f"  ✅ Formation energy: {formation_energy:.6f} eV/atom")'''

    code += '''
            results.append(result)
            
            # Save results after each structure
            df_results = pd.DataFrame(results)
            df_results.to_csv("results/optimization_results.csv", index=False)
            print(f"  💾 Results updated and saved after structure {i+1}/{len(structure_files)}")

        except Exception as e:
            print(f"  ❌ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({"structure": filename, "error": str(e)})
            
            df_results = pd.DataFrame(results)
            df_results.to_csv("results/optimization_results.csv", index=False)
            print(f"  💾 Results updated and saved after structure {i+1}/{len(structure_files)} (with error)")

    # Final summary save
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/optimization_results.csv", index=False)

    print(f"\\n💾 Saved all results to results/optimization_results.csv")
    print(f"📁 Optimized structures saved in optimized_structures/ directory")

    with open("results/optimization_summary.txt", "w") as f:
        f.write("MACE Geometry Optimization Results\\n")
        f.write("=" * 50 + "\\n\\n")
        for result in results:
            if "error" not in result:
                f.write(f"Structure: {result['structure']}\\n")
                f.write(f"Initial Energy: {result['initial_energy_eV']:.6f} eV\\n")
                f.write(f"Final Energy: {result['final_energy_eV']:.6f} eV\\n")
                f.write(f"Energy Change: {result['energy_change_eV']:.6f} eV\\n")
                f.write(f"Final Max Force: {result['final_max_force_eV_per_A']:.4f} eV/Å\\n")
                f.write(f"Max Stress: {result['max_stress_GPa']:.4f} GPa\\n")
                f.write(f"Convergence: {result['convergence_status']}\\n")
                f.write(f"Steps: {result['optimization_steps']}\\n")
                f.write(f"Atoms: {result['num_atoms']}\\n")
                f.write(f"Selective Dynamics: {result['has_selective_dynamics']}\\n")
                if result['has_selective_dynamics']:
                    f.write(f"Fixed Atoms: {result['num_fixed_atoms']}/{result['num_atoms']}\\n")
                f.write(f"Output File: {result['output_structure']}\\n")'''

    if calc_formation_energy:
        code += '''
                if "formation_energy_eV_per_atom" in result and result["formation_energy_eV_per_atom"] is not None:
                    f.write(f"Formation Energy: {result['formation_energy_eV_per_atom']:.6f} eV/atom\\n")'''

    code += '''
                f.write("\\n")
            else:
                f.write(f"Structure: {result['structure']} - ERROR: {result['error']}\\n\\n")
    
    print(f"💾 Saved summary to results/optimization_summary.txt")'''

    return code


def _generate_phonon_code(phonon_params, optimization_params, calc_formation_energy):
    """Generate code for phonon calculations."""
    auto_supercell = phonon_params.get('auto_supercell', True)
    if auto_supercell:
        target_length = phonon_params.get('target_supercell_length', 15.0)
        max_multiplier = phonon_params.get('max_supercell_multiplier', 4)
        max_atoms = phonon_params.get('max_supercell_atoms', 800)
    else:
        supercell_size = phonon_params.get('supercell_size', (2, 2, 2))

    delta = phonon_params.get('delta', 0.01)
    temperature = phonon_params.get('temperature', 300)
    npoints = phonon_params.get('npoints', 100)

    code = f'''    structure_files = [f for f in os.listdir(".") if f.startswith("POSCAR") or f.endswith(".vasp")]
    results = []
    print(f"🎵 Found {{len(structure_files)}} structure files for phonon calculations")

    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
        import phonopy.units as units_phonopy
        PHONOPY_AVAILABLE = True
        print("✅ Phonopy available")
    except ImportError:
        print("❌ Phonopy not available. Please install with: pip install phonopy")
        return

    auto_supercell = {auto_supercell}'''

    if auto_supercell:
        code += f'''
    target_supercell_length = {target_length}
    max_supercell_multiplier = {max_multiplier}
    max_supercell_atoms = {max_atoms}
    print(f"⚙️ Auto supercell: target length {target_supercell_length} Å, max multiplier {max_supercell_multiplier}")'''
    else:
        code += f'''
    supercell_size = {supercell_size}
    print(f"⚙️ Manual supercell: {supercell_size}")'''

    code += f'''
    displacement_distance = {delta}
    temperature = {temperature}
    npoints_per_segment = {npoints}
    pre_opt_steps = {optimization_params.get('max_steps', 50)}
    
    print(f"⚙️ Phonon settings:")
    print(f"  - Displacement: {{displacement_distance}} Å")
    print(f"  - Temperature: {{temperature}} K")
    print(f"  - Pre-opt steps: {{pre_opt_steps}}")

    reference_energies = {{}}'''

    if calc_formation_energy:
        code += '''
    print("🔬 Calculating atomic reference energies...")
    all_elements = set()
    for filename in structure_files:
        atoms = read(filename)
        for symbol in atoms.get_chemical_symbols():
            all_elements.add(symbol)

    print(f"🧪 Found elements: {', '.join(sorted(all_elements))}")
    
    for i, element in enumerate(sorted(all_elements)):
        print(f"  📍 Calculating reference for {element} ({i+1}/{len(all_elements)})...")
        atom = Atoms(element, positions=[(0, 0, 0)], cell=[20, 20, 20], pbc=True)
        atom.calc = calculator
        reference_energies[element] = atom.get_potential_energy()
        print(f"  ✅ {element}: {reference_energies[element]:.6f} eV")'''

    code += '''

    for i, filename in enumerate(structure_files):
        print(f"\\n🎵 Processing structure {i+1}/{len(structure_files)}: {filename}")
        structure_start_time = time.time()
        try:
            atoms = read(filename)
            atoms.calc = calculator
            print(f"  📊 Structure has {len(atoms)} atoms")

            print("  🔧 Running pre-optimization...")
            temp_atoms = atoms.copy()
            temp_atoms.calc = calculator
            temp_optimizer = LBFGS(temp_atoms, logfile=None)
            temp_optimizer.run(fmax=0.02, steps=pre_opt_steps)
            atoms = temp_atoms
            print(f"  ✅ Pre-optimization completed in {temp_optimizer.nsteps} steps")

            from pymatgen.io.ase import AseAtomsAdaptor
            adaptor = AseAtomsAdaptor()
            pmg_structure = adaptor.get_structure(atoms)

            phonopy_atoms = PhonopyAtoms(
                symbols=[str(site.specie) for site in pmg_structure],
                scaled_positions=pmg_structure.frac_coords,
                cell=pmg_structure.lattice.matrix
            )'''

    if auto_supercell:
        code += '''
            a, b, c = pmg_structure.lattice.abc
            na = max(1, min(max_supercell_multiplier, int(np.ceil(target_supercell_length / a))))
            nb = max(1, min(max_supercell_multiplier, int(np.ceil(target_supercell_length / b))))
            nc = max(1, min(max_supercell_multiplier, int(np.ceil(target_supercell_length / c))))

            if len(atoms) > 50:
                na = max(1, na - 1)
                nb = max(1, nb - 1)
                nc = max(1, nc - 1)

            supercell_matrix = [[na, 0, 0], [0, nb, 0], [0, 0, nc]]
            total_atoms = len(atoms) * na * nb * nc

            if total_atoms > max_supercell_atoms:
                print(f"  ⚠️ Supercell too large ({total_atoms} atoms), using 1x1x1")
                supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                total_atoms = len(atoms)'''
    else:
        code += f'''
            supercell_matrix = [[{supercell_size[0]}, 0, 0], [0, {supercell_size[1]}, 0], [0, 0, {supercell_size[2]}]]
            total_atoms = len(atoms) * {supercell_size[0]} * {supercell_size[1]} * {supercell_size[2]}'''

    code += '''
            
            print(f"  📏 Using supercell: {supercell_matrix} ({total_atoms} total atoms)")
            
            phonon = Phonopy(phonopy_atoms, supercell_matrix=supercell_matrix, primitive_matrix='auto')
            
            print(f"  📍 Generating displacements...")
            phonon.generate_displacements(distance=displacement_distance)
            supercells = phonon.get_supercells_with_displacements()
            print(f"  📊 Generated {len(supercells)} displaced supercells")
            
            print("  ⚡ Calculating forces...")
            forces = []
            for j, supercell in enumerate(supercells):
                if j % max(1, len(supercells) // 5) == 0:
                    print(f"    📊 Progress: {j+1}/{len(supercells)} ({100*(j+1)/len(supercells):.1f}%)")
                    
                ase_supercell = Atoms(
                    symbols=supercell.symbols,
                    positions=supercell.positions,
                    cell=supercell.cell,
                    pbc=True
                )
                ase_supercell.calc = calculator
                supercell_forces = ase_supercell.get_forces()
                forces.append(supercell_forces)
            
            print("  ✅ All force calculations completed")
            phonon.forces = forces
            print("  🔧 Calculating force constants...")
            phonon.produce_force_constants()
            
            print("  📈 Calculating phonon band structure...")
            try:
                from pymatgen.symmetry.bandstructure import HighSymmKpath
                kpath = HighSymmKpath(pmg_structure)
                path = kpath.kpath["path"]
                kpoints_dict = kpath.kpath["kpoints"]
                
                path_kpoints = []
                for segment in path:
                    if len(segment) >= 2:
                        start_point = np.array(kpoints_dict[segment[0]])
                        end_point = np.array(kpoints_dict[segment[-1]])
                        for j in range(npoints_per_segment):
                            t = j / (npoints_per_segment - 1)
                            kpt = start_point + t * (end_point - start_point)
                            path_kpoints.append(kpt.tolist())
                
                bands = [path_kpoints]
                print(f"  📍 Generated {len(path_kpoints)} k-points along high-symmetry path")
                
            except Exception:
                print("  ⚠️ Using fallback k-point path")
                bands = [[[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]]]
            
            phonon.run_band_structure(bands, is_band_connection=False, with_eigenvectors=False)
            band_dict = phonon.get_band_structure_dict()
            
            print("  📊 Calculating phonon DOS...")
            mesh_density = [30, 30, 30] if len(atoms) <= 100 else [20, 20, 20]
            phonon.run_mesh(mesh_density)
            phonon.run_total_dos()
            dos_dict = phonon.get_total_dos_dict()
            
            print(f"  🌡️ Calculating thermodynamics at {temperature} K...")
            phonon.run_thermal_properties(t_step=10, t_max=1500, t_min=0)
            thermal_dict = phonon.get_thermal_properties_dict()
            
            frequencies = np.array(band_dict['frequencies']) * units_phonopy.THzToEv * 1000
            dos_frequencies = dos_dict['frequency_points'] * units_phonopy.THzToEv * 1000
            dos_values = dos_dict['total_dos']
            
            valid_frequencies = frequencies[~np.isnan(frequencies)]
            imaginary_modes = np.sum(valid_frequencies < -0.001)
            min_frequency = np.min(valid_frequencies) if len(valid_frequencies) > 0 else 0
            max_frequency = np.max(valid_frequencies) if len(valid_frequencies) > 0 else 0
            
            temps = np.array(thermal_dict['temperatures'])
            temp_idx = np.argmin(np.abs(temps - temperature))
            
            thermo_props = {{
                'temperature': float(temps[temp_idx]),
                'zero_point_energy': float(thermal_dict['zero_point_energy']),
                'internal_energy': float(thermal_dict['internal_energy'][temp_idx]),
                'heat_capacity': float(thermal_dict['heat_capacity'][temp_idx]),
                'entropy': float(thermal_dict['entropy'][temp_idx]),
                'free_energy': float(thermal_dict['free_energy'][temp_idx])
            }}
            
            result = {{
                "structure": filename,
                "calculation_type": "phonon_calculation",
                "supercell_matrix": supercell_matrix,
                "imaginary_modes": int(imaginary_modes),
                "min_frequency_meV": float(min_frequency),
                "max_frequency_meV": float(max_frequency),
                "thermodynamics": thermo_props,
                "num_atoms": len(atoms),
                "total_supercell_atoms": total_atoms
            }}
            
            phonon_data_dict = {{
                "structure_name": [filename],
                "supercell_matrix_00": [supercell_matrix[0][0]],
                "supercell_matrix_11": [supercell_matrix[1][1]], 
                "supercell_matrix_22": [supercell_matrix[2][2]],
                "imaginary_modes": [int(imaginary_modes)],
                "min_frequency_meV": [float(min_frequency)],
                "max_frequency_meV": [float(max_frequency)],
                "zero_point_energy_eV": [thermo_props['zero_point_energy']],
                "internal_energy_eV": [thermo_props['internal_energy']],
                "heat_capacity_eV_K": [thermo_props['heat_capacity']],
                "entropy_eV_K": [thermo_props['entropy']],
                "free_energy_eV": [thermo_props['free_energy']],
                "temperature_K": [thermo_props['temperature']],
                "num_atoms": [len(atoms)],
                "total_supercell_atoms": [total_atoms]
            }}
            
            df_phonon = pd.DataFrame(phonon_data_dict)
            df_phonon.to_csv(f"results/phonon_data_{filename.replace('.', '_')}.csv", index=False)
            
            if len(valid_frequencies) > 0:
                freq_data = {{
                    "frequency_meV": valid_frequencies[~np.isnan(valid_frequencies)].flatten(),
                    "structure": [filename] * len(valid_frequencies[~np.isnan(valid_frequencies)].flatten())
                }}
                df_freq = pd.DataFrame(freq_data)
                df_freq.to_csv(f"results/phonon_frequencies_{filename.replace('.', '_')}.csv", index=False)
            
            dos_data = {{
                "energy_meV": dos_frequencies,
                "dos_states_per_meV": dos_values,
                "structure": [filename] * len(dos_frequencies)
            }}
            df_dos = pd.DataFrame(dos_data)
            df_dos.to_csv(f"results/phonon_dos_{filename.replace('.', '_')}.csv", index=False)
            
            final_energy = atoms.get_potential_energy()
            result["energy_eV"] = final_energy
            
            if calc_formation_energy:
                formation_energy = calculate_formation_energy(final_energy, atoms, reference_energies)
                result["formation_energy_eV_per_atom"] = formation_energy
            
            structure_time = time.time() - structure_start_time
            print(f"  ✅ Phonon calculation completed in {structure_time:.1f}s")
            print(f"  ✅ Energy: {final_energy:.6f} eV")
            print(f"  ✅ Imaginary modes: {imaginary_modes}")
            print(f"  ✅ Frequency range: {min_frequency:.3f} to {max_frequency:.3f} meV")
            if imaginary_modes > 0:
                print(f"  ⚠️ Structure may be dynamically unstable")
            else:
                print(f"  ✅ Structure appears dynamically stable")
            
                
            results.append(result)
            
            df_results = pd.DataFrame(results)
            df_results.to_csv("results/phonon_results.csv", index=False)
            print(f"  💾 Results updated and saved after structure {i+1}/{len(structure_files)}")
            
        except Exception as e:
            print(f"  ❌ Phonon calculation failed: {e}")
            results.append({"structure": filename, "error": str(e)}")
            
            df_results = pd.DataFrame(results)
            df_results.to_csv("results/phonon_results.csv", index=False)
            print(f"  💾 Results updated and saved after structure {i+1}/{len(structure_files)} (with error)")
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/phonon_results.csv", index=False)

    print(f"\\n💾 Saved results to results/phonon_results.csv")
    
    with open("results/phonon_summary.txt", "w") as f:
        f.write("MACE Phonon Calculation Results\\n")
        f.write("=" * 40 + "\\n\\n")
        for result in results:
            if "error" not in result:
                f.write(f"Structure: {result['structure']}\\n")
                f.write(f"Energy: {result['energy_eV']:.6f} eV\\n")
                f.write(f"Imaginary modes: {result['imaginary_modes']}\\n")
                f.write(f"Min frequency: {result['min_frequency_meV']:.3f} meV\\n")
                f.write(f"Max frequency: {result['max_frequency_meV']:.3f} meV\\n")
                f.write(f"Atoms: {result['num_atoms']}\\n")
                f.write(f"Supercell atoms: {result['total_supercell_atoms']}\\n")
                if "formation_energy_eV_per_atom" in result and result["formation_energy_eV_per_atom"] is not None:
                    f.write(f"Formation Energy: {result['formation_energy_eV_per_atom']:.6f} eV/atom\\n")
                f.write("\\n")
            else:
                f.write(f"Structure: {result['structure']} - ERROR: {result['error']}\\n\\n")
    
    print(f"💾 Saved summary to results/phonon_summary.txt")
'''
    return code
