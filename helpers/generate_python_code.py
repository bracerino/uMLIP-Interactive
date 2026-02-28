import json
from datetime import datetime


def _generate_mlip_imports():
    return """# MACE imports
try:
    from mace.calculators import mace_mp, mace_off
    MACE_AVAILABLE = True
except ImportError:
    try:
        from mace.calculators import MACECalculator
        MACE_AVAILABLE = True
    except ImportError:
        MACE_AVAILABLE = False
        
# UPET imports (successor of PET-MAD)
try:
    from upet.calculator import UPETCalculator
    UPET_AVAILABLE = True
except ImportError:
    UPET_AVAILABLE = False
# CHGNet imports
try:
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    CHGNET_AVAILABLE = True
except ImportError:
    CHGNET_AVAILABLE = False

# SevenNet imports (requires torch 2.6 compatibility)
try:
    torch.serialization.add_safe_globals([slice])  # Required for torch 2.6
    from sevenn.calculator import SevenNetCalculator
    SEVENNET_AVAILABLE = True
except ImportError:
    SEVENNET_AVAILABLE = False

# MatterSim imports
try:
    from mattersim.forcefield import MatterSimCalculator
    MATTERSIM_AVAILABLE = True
except ImportError:
    MATTERSIM_AVAILABLE = False

# ORB imports
try:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
    ORB_AVAILABLE = True
except ImportError:
    ORB_AVAILABLE = False

# Nequix imports
try:
    from nequix.calculator import NequixCalculator
    NEQUIX_AVAILABLE = True
except ImportError:
    NEQUIX_AVAILABLE = False

#MAD-PET
try:
    from pet_mad.calculator import PETMADCalculator
    PETMAD_AVAILABLE = True
except ImportError:
    PETMAD_AVAILABLE = False


# Check if any calculator is available
if not (MACE_AVAILABLE or CHGNET_AVAILABLE or UPET_AVAILABLE or SEVENNET_AVAILABLE or MATTERSIM_AVAILABLE or ORB_AVAILABLE or NEQUIX_AVAILABLE or PETMAD_AVAILABLE):
    print("❌ No MLIP calculators available!")
    print("Please install at least one:")
    print("  - MACE: pip install mace-torch")
    print("  - CHGNet: pip install chgnet") 
    print("  - SevenNet: pip install sevenn")
    print("  - MatterSim: pip install mattersim")
    print("  - ORB: pip install orb-models")
    print("  - Nequix: pip install nequix")
    print("  - PET-MAD: pip install pet-mad")
    print("  - UPET: pip install upet")
    exit(1)
else:
    available_models = []
    if MACE_AVAILABLE:
        available_models.append("MACE")
    if CHGNET_AVAILABLE:
        available_models.append("CHGNet")
    if SEVENNET_AVAILABLE:
        available_models.append("SevenNet")
    if MATTERSIM_AVAILABLE:
        available_models.append("MatterSim")
    if ORB_AVAILABLE:
        available_models.append("ORB")
    if NEQUIX_AVAILABLE:
        available_models.append("Nequix")
    if PETMAD_AVAILABLE:
        available_models.append("PET-MAD")
    if UPET_AVAILABLE:
        available_models.append("UPET")
    print(f"✅ Available MLIP models: {', '.join(available_models)}")"""


def generate_python_script(structures, calc_type, model_size, device, dtype, optimization_params,
                           phonon_params, elastic_params, calc_formation_energy, selected_model_key=None,
                           substitutions=None, ga_params=None, supercell_info=None, thread_count=4,
                           mace_head=None, mace_dispersion=False, mace_dispersion_xc="pbe"
                           ):
    structure_creation_code = _generate_structure_creation_code(structures)
    calculator_setup_code = _generate_calculator_setup_code(
        model_size, device, selected_model_key, dtype, mace_head=mace_head,
        mace_dispersion=mace_dispersion,
        mace_dispersion_xc=mace_dispersion_xc)

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
    elif calc_type == "GA Structure Optimization":
        calculation_code = _generate_ga_code(
            substitutions, ga_params, calc_formation_energy, supercell_info)
    else:
        calculation_code = _generate_energy_only_code(calc_formation_energy)
    config_info = ""
    if mace_head:
        config_info += f"\nMACE Head: {mace_head}"
    optimizer_info = ""
    if calc_type == "Geometry Optimization" and optimization_params:
        optimizer_info = f"\nOptimizer: {optimization_params.get('optimizer', 'BFGS')}"
    if mace_dispersion:
        config_info += f"\nDispersion: D3-{mace_dispersion_xc}"
    script = f"""#!/usr/bin/env python3
\"\"\"
MACE Calculation Script
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Calculation Type: {calc_type}{optimizer_info}
Model: {selected_model_key or model_size}
Device: {device}
Precision: {dtype}{config_info}
\"\"\"

import os
import time
import numpy as np
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import random
from copy import deepcopy
import threading
import queue
import zipfile
import io

# Set threading before other imports
os.environ['OMP_NUM_THREADS'] = '{thread_count}'

import torch
torch.set_num_threads({thread_count})

# ASE imports
from ase import Atoms
from ase.io import read, write
from ase.optimize import (
    BFGS, LBFGS, FIRE,
    BFGSLineSearch, LBFGSLineSearch,
    GoodOldQuasiNewton, MDMin, GPMin
)
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import FixAtoms
from ase.filters import ExpCellFilter, UnitCellFilter

# PyMatGen imports
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

{_generate_mlip_imports()}

{_generate_utility_functions()}

{_generate_ga_classes() if calc_type == "GA Structure Optimization" else ""}

def main():
    start_time = time.time()
    print("🚀 Starting MACE calculation script...")
    print(f"📅 Timestamp: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
    print(f"🔬 Calculation type: {calc_type}")
    print(f"🤖 Model: {selected_model_key or model_size}")
    print(f"💻 Device: {device}")
    print(f"🧵 CPU threads: {{os.environ.get('OMP_NUM_THREADS', 'default')}}")

    # Create output directories
    Path("optimized_structures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    {'Path("ga_results").mkdir(exist_ok=True)' if calc_type == "GA Structure Optimization" else ''}

    # Create structure files
    print("\\n📁 Creating structure files...")
{structure_creation_code}

    # Setup calculator
    print("\\n🔧 Setting up MLIP calculator...")
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

if __name__ == "__main__":
    main()
"""

    return script


def generate_python_script_local_files(calc_type, model_size, device, dtype, optimization_params,
                                       phonon_params, elastic_params, calc_formation_energy, selected_model_key=None,
                                       substitutions=None, ga_params=None, supercell_info=None, thread_count=4,
                                       mace_head=None, mace_dispersion=False, mace_dispersion_xc="pbe",
                                       custom_mace_path=None):
    """
    Generate a complete Python script for MACE calculations that reads POSCAR files from the local directory.
    """

    calculator_setup_code = _generate_calculator_setup_code(
        model_size, device, selected_model_key, dtype, mace_head=mace_head,
        mace_dispersion=mace_dispersion,
        mace_dispersion_xc=mace_dispersion_xc,
        custom_mace_path=custom_mace_path
    )

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
    elif calc_type == "GA Structure Optimization":
        calculation_code = _generate_ga_code(
            substitutions, ga_params, calc_formation_energy, supercell_info)
    else:
        calculation_code = _generate_energy_only_code(calc_formation_energy)
    config_info = ""
    if mace_head:
        config_info += f"\nMACE Head: {mace_head}"
    optimizer_info = ""
    if calc_type == "Geometry Optimization" and optimization_params:
        optimizer_info = f"\nOptimizer: {optimization_params.get('optimizer', 'BFGS')}"
    if mace_dispersion:
        config_info += f"\nDispersion: D3-{mace_dispersion_xc}"
    script = f"""#!/usr/bin/env python3
\"\"\"
MACE Calculation Script (Local POSCAR Files)
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Calculation Type: {calc_type}{optimizer_info}
Model: {selected_model_key or model_size}
Device: {device}
Precision: {dtype}{config_info}

This script reads POSCAR files from the current directory.
Place your POSCAR files in the same directory as this script before running.
\"\"\"

import os
import time
import numpy as np
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import random
from copy import deepcopy
import threading
import queue
import zipfile
import io

# Set threading before other imports
os.environ['OMP_NUM_THREADS'] = '{thread_count}'

import torch
torch.set_num_threads({thread_count})

# ASE imports
from ase import Atoms
from ase.io import read, write
from ase.optimize import (
    BFGS, LBFGS, FIRE,
    BFGSLineSearch, LBFGSLineSearch,
    GoodOldQuasiNewton, MDMin, GPMin
)
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import FixAtoms
from ase.filters import ExpCellFilter, UnitCellFilter

# PyMatGen imports
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

{_generate_mlip_imports()}

{_generate_utility_functions()}

{_generate_ga_classes() if calc_type == "GA Structure Optimization" else ""}

def main():
    start_time = time.time()
    print("🚀 Starting MACE calculation script...")
    print(f"📅 Timestamp: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
    print(f"🔬 Calculation type: {calc_type}")
    print(f"🤖 Model: {selected_model_key or model_size}")
    print(f"💻 Device: {device}")
    print(f"🧵 CPU threads: {{os.environ.get('OMP_NUM_THREADS', 'default')}}")

    # Create output directories
    Path("optimized_structures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    {'Path("ga_results").mkdir(exist_ok=True)' if calc_type == "GA Structure Optimization" else ''}

    # Find and validate POSCAR files in current directory
    print("\\n📁 Looking for POSCAR files in current directory...")
    structure_files = sorted([f for f in os.listdir(".") if f.startswith("POSCAR") or f.endswith(".vasp") or f.endswith(".poscar")])

    if not structure_files:
        print("❌ No POSCAR files found in current directory!")
        print("Please place files starting with 'POSCAR' or ending with '.vasp' in the same directory as this script.")
        return

    print(f"✅ Found {{len(structure_files)}} structure files:")
    for i, filename in enumerate(structure_files, 1):
        try:
            atoms = read(filename)
            composition = "".join([f"{{symbol}}{{list(atoms.get_chemical_symbols()).count(symbol)}}" 
                                 for symbol in sorted(set(atoms.get_chemical_symbols()))])
            print(f"  {{i}}. {{filename}} - {{composition}} ({{len(atoms)}} atoms)")
        except Exception as e:
            print(f"  {{i}}. {{filename}} - ❌ Error: {{str(e)}}")

    # Setup calculator
    print("\\n🔧 Setting up MLIP calculator...")
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


def _generate_ga_classes():
    return '''
class GeneticAlgorithmOptimizer:
    def __init__(self, base_structure, calculator, substitutions, ga_params, run_id=0):
        self.base_structure = base_structure
        self.calculator = calculator
        self.substitutions = substitutions
        self.ga_params = ga_params
        self.run_id = run_id

        self.population = []
        self.fitness_history = []
        self.detailed_history = []
        self.best_individual = None
        self.best_energy = float('inf')
        self.final_population = []
        self.final_fitness = []

        # Set random seeds
        random.seed(run_id * 12345)
        np.random.seed(run_id * 12345)

        self.create_site_id_mapping()
        self.setup_substitution_sites()
        self.setup_fixed_substitution_counts()

    def create_site_id_mapping(self):
        self.site_ids = {}
        self.id_to_site = {}

        for i, site in enumerate(self.base_structure):
            site_id = f"{site.specie.symbol}_{i}"
            self.site_ids[i] = site_id
            self.id_to_site[site_id] = {
                'original_index': i,
                'element': site.specie.symbol,
                'coords': site.coords.copy(),
                'frac_coords': site.frac_coords.copy()
            }

        print(f"  🗺️ Created site ID mapping for {len(self.site_ids)} sites")

    def setup_substitution_sites(self):
        self.substitutable_sites = {}

        for i, site in enumerate(self.base_structure):
            element = site.specie.symbol
            if element in self.substitutions:
                if element not in self.substitutable_sites:
                    self.substitutable_sites[element] = []
                self.substitutable_sites[element].append(i)

        print(f"  🎯 Found substitutable sites: {self.substitutable_sites}")

    def setup_fixed_substitution_counts(self):
        self.fixed_substitution_counts = {}

        for original_element, sub_info in self.substitutions.items():
            if original_element in self.substitutable_sites:
                total_sites = len(self.substitutable_sites[original_element])
                n_substitute = int(total_sites * sub_info['concentration'])

                self.fixed_substitution_counts[original_element] = {
                    'n_substitute': n_substitute,
                    'new_element': sub_info['new_element'],
                    'total_sites': total_sites
                }

                if sub_info['new_element'] == 'VACANCY':
                    print(f"  🕳️ Fixed vacancy creation: {n_substitute}/{total_sites} {original_element} → VACANCY")
                else:
                    print(f"  🔄 Fixed substitution: {n_substitute}/{total_sites} {original_element} → {sub_info['new_element']}")

    def create_individual_from_pattern(self, substitution_pattern):
        structure = Structure(
            lattice=self.base_structure.lattice,
            species=[],
            coords=[],
            coords_are_cartesian=True
        )

        for original_idx in range(len(self.base_structure)):
            site_id = self.site_ids[original_idx]
            original_site_info = self.id_to_site[site_id]
            original_element = original_site_info['element']

            is_vacancy = False
            new_element = original_element

            for element, pattern_info in substitution_pattern.items():
                if element == original_element and original_idx in pattern_info['substituted_sites']:
                    if pattern_info['new_element'] == 'VACANCY':
                        is_vacancy = True
                        break
                    else:
                        new_element = pattern_info['new_element']
                        break

            if not is_vacancy:
                structure.append(
                    species=new_element,
                    coords=original_site_info['coords'],
                    coords_are_cartesian=True
                )

        return structure

    def create_random_individual(self):
        substitution_pattern = {}

        for original_element, sub_info in self.fixed_substitution_counts.items():
            sites = self.substitutable_sites[original_element]
            n_substitute = sub_info['n_substitute']
            new_element = sub_info['new_element']

            substitute_indices = random.sample(sites, n_substitute)

            substitution_pattern[original_element] = {
                'substituted_sites': substitute_indices,
                'new_element': new_element,
                'n_substitute': n_substitute
            }

        structure = self.create_individual_from_pattern(substitution_pattern)

        if self.ga_params.get('perturb_positions', True):
            structure = self.apply_position_perturbations(structure)

        return structure

    def apply_position_perturbations(self, structure):
        if not self.ga_params.get('perturb_positions', True):
            return structure

        max_displacement = self.ga_params.get('max_displacement', 0.1)
        perturbed_structure = structure.copy()

        for i in range(len(perturbed_structure)):
            displacement = np.random.uniform(-max_displacement, max_displacement, 3)
            cart_displacement = structure.lattice.get_cartesian_coords(
                displacement / np.linalg.norm(structure.lattice.matrix, axis=1))

            old_coords = perturbed_structure[i].coords
            new_coords = old_coords + cart_displacement
            perturbed_structure.replace(i, perturbed_structure[i].specie, new_coords, coords_are_cartesian=True)

        return perturbed_structure

    def get_substitution_pattern(self, structure):
        pattern = {}
        current_site_mapping = self.map_current_to_original_sites(structure)

        for original_element in self.substitutions:
            if original_element not in self.substitutable_sites:
                continue

            sites = self.substitutable_sites[original_element]
            new_element = self.fixed_substitution_counts[original_element]['new_element']
            substituted_sites = []

            for original_site_idx in sites:
                if original_site_idx in current_site_mapping:
                    current_idx = current_site_mapping[original_site_idx]
                    current_element = structure[current_idx].specie.symbol

                    if current_element != original_element:
                        substituted_sites.append(original_site_idx)
                else:
                    if new_element == 'VACANCY':
                        substituted_sites.append(original_site_idx)

            pattern[original_element] = {
                'substituted_sites': substituted_sites,
                'new_element': new_element,
                'n_substitute': len(substituted_sites)
            }

        return pattern

    def map_current_to_original_sites(self, structure):
        mapping = {}
        used_original_sites = set()

        for current_idx, site in enumerate(structure):
            min_distance = float('inf')
            best_original_idx = None

            for original_idx in range(len(self.base_structure)):
                if original_idx in used_original_sites:
                    continue

                original_site_info = self.id_to_site[self.site_ids[original_idx]]

                if site.specie.symbol != original_site_info['element']:
                    original_element = original_site_info['element']
                    if (original_element in self.substitutions and
                            self.substitutions[original_element]['new_element'] == site.specie.symbol):
                        pass
                    else:
                        continue

                distance = np.linalg.norm(site.coords - original_site_info['coords'])

                if distance < min_distance:
                    min_distance = distance
                    best_original_idx = original_idx

            if best_original_idx is not None:
                mapping[best_original_idx] = current_idx
                used_original_sites.add(best_original_idx)

        return mapping

    def crossover(self, parent1, parent2):
        pattern1 = self.get_substitution_pattern(parent1)
        pattern2 = self.get_substitution_pattern(parent2)

        child_pattern = {}

        for original_element in self.substitutions:
            if (original_element not in pattern1 or
                    original_element not in pattern2 or
                    original_element not in self.substitutable_sites):
                continue

            sites1 = set(pattern1[original_element]['substituted_sites'])
            sites2 = set(pattern2[original_element]['substituted_sites'])
            n_substitute = self.fixed_substitution_counts[original_element]['n_substitute']
            new_element = self.fixed_substitution_counts[original_element]['new_element']
            available_sites = self.substitutable_sites[original_element]

            all_substituted_sites = sites1.union(sites2)

            if len(all_substituted_sites) < n_substitute:
                remaining_sites = set(available_sites) - all_substituted_sites
                if remaining_sites:
                    additional_needed = n_substitute - len(all_substituted_sites)
                    additional_sites = random.sample(list(remaining_sites),
                                                     min(additional_needed, len(remaining_sites)))
                    all_substituted_sites.update(additional_sites)

            if len(all_substituted_sites) >= n_substitute:
                child_substituted_sites = random.sample(list(all_substituted_sites), n_substitute)
            else:
                child_substituted_sites = list(all_substituted_sites)

            child_pattern[original_element] = {
                'substituted_sites': child_substituted_sites,
                'new_element': new_element,
                'n_substitute': len(child_substituted_sites)
            }

        child = self.create_individual_from_pattern(child_pattern)

        if self.ga_params.get('perturb_positions', True):
            child = self.mix_positions_from_parents(child, parent1, parent2)

        return child

    def mix_positions_from_parents(self, child, parent1, parent2):
        child_mapping = self.map_current_to_original_sites(child)
        parent1_mapping = self.map_current_to_original_sites(parent1)
        parent2_mapping = self.map_current_to_original_sites(parent2)

        mixed_child = child.copy()

        for original_idx, child_idx in child_mapping.items():
            source_parent = parent2 if random.random() < 0.5 else parent1
            source_mapping = parent2_mapping if source_parent == parent2 else parent1_mapping

            if original_idx in source_mapping:
                source_idx = source_mapping[original_idx]
                source_coords = source_parent[source_idx].coords
                mixed_child.replace(child_idx, mixed_child[child_idx].specie, source_coords, coords_are_cartesian=True)

        return mixed_child

    def mutate(self, structure):
        pattern = self.get_substitution_pattern(structure)
        mutation_rate = self.ga_params.get('mutation_rate', 0.1)

        for original_element in self.substitutions:
            if (original_element not in pattern or
                    original_element not in self.substitutable_sites):
                continue

            sites = self.substitutable_sites[original_element]
            substituted_sites = set(pattern[original_element]['substituted_sites'])
            non_substituted_sites = set(sites) - substituted_sites

            sites_to_mutate = []
            for substituted_site in substituted_sites:
                if random.random() < mutation_rate:
                    sites_to_mutate.append(substituted_site)

            for site_to_swap in sites_to_mutate:
                if non_substituted_sites:
                    new_site = random.choice(list(non_substituted_sites))

                    substituted_sites.remove(site_to_swap)
                    substituted_sites.add(new_site)
                    non_substituted_sites.remove(new_site)
                    non_substituted_sites.add(site_to_swap)

            pattern[original_element]['substituted_sites'] = list(substituted_sites)

        mutated = self.create_individual_from_pattern(pattern)

        if self.ga_params.get('perturb_positions', True):
            mutated = self.mutate_positions(mutated)

        return mutated

    def mutate_positions(self, structure):
        mutation_rate = self.ga_params.get('mutation_rate', 0.1)
        max_displacement = self.ga_params.get('max_displacement', 0.1)

        mutated = structure.copy()

        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                displacement = np.random.uniform(-max_displacement / 2, max_displacement / 2, 3)
                cart_displacement = structure.lattice.get_cartesian_coords(
                    displacement / np.linalg.norm(structure.lattice.matrix, axis=1))

                old_coords = mutated[i].coords
                new_coords = old_coords + cart_displacement
                mutated.replace(i, mutated[i].specie, new_coords, coords_are_cartesian=True)

        return mutated

    def validate_individual(self, structure):
        pattern = self.get_substitution_pattern(structure)

        for original_element, expected_info in self.fixed_substitution_counts.items():
            expected_count = expected_info['n_substitute']

            if original_element not in pattern:
                if expected_count > 0:
                    print(f"    ❌ Validation failed: Expected {expected_count} substitutions for {original_element}, got 0")
                    return False
                continue

            actual_count = pattern[original_element]['n_substitute']

            if actual_count != expected_count:
                print(f"    ❌ Validation failed: Expected {expected_count} substitutions for {original_element}, got {actual_count}")
                return False

        return True

    def calculate_fitness(self, structure):
        try:
            if not self.validate_individual(structure):
                return float('inf')

            adaptor = AseAtomsAdaptor()
            atoms = adaptor.get_atoms(structure)
            atoms.calc = self.calculator

            if self.ga_params.get('perturb_positions', True):
                optimizer_name = self.ga_params.get('optimizer', 'BFGS')
                fmax = self.ga_params.get('fmax', 0.05)
                max_steps = self.ga_params.get('max_steps', 100)
                maxstep = self.ga_params.get('maxstep', 0.2)

                if optimizer_name == 'LBFGS':
                    optimizer = LBFGS(atoms, maxstep=maxstep, logfile=None)
                else:
                    optimizer = BFGS(atoms, maxstep=maxstep, logfile=None)

                optimizer.run(fmax=fmax, steps=max_steps)

            energy = atoms.get_potential_energy()
            return energy
        except Exception as e:
            print(f"    ❌ Error calculating energy: {str(e)}")
            return float('inf')

    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        selected = []

        for _ in range(2):
            tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx])

        return selected

    def optimize(self):
        import time

        population_size = self.ga_params.get('population_size', 50)
        max_generations = self.ga_params.get('max_generations', 100)
        elitism_ratio = self.ga_params.get('elitism_ratio', 0.1)

        print(f"  🧬 Starting GA run {self.run_id + 1}: {population_size} individuals, {max_generations} generations")

        # Track timing for estimates
        total_start_time = time.time()
        generation_times = []

        print(f"  🔄 Creating initial population...")
        self.population = []
        fitness_scores = []

        # Time the initial population creation
        init_start_time = time.time()

        for i in range(population_size):
            individual = self.create_random_individual()
            fitness = self.calculate_fitness(individual)

            self.population.append(individual)
            fitness_scores.append(fitness)

            if fitness < self.best_energy:
                self.best_energy = fitness
                self.best_individual = individual.copy()

            if (i + 1) % 10 == 0:
                print(f"    Generated {i + 1}/{population_size} individuals")

        init_time = time.time() - init_start_time
        print(f"  ✅ Initial population created in {init_time:.1f}s. Best energy: {self.best_energy:.6f} eV")

        # Print header for generation table
        print("")
        print("=" * 120)
        print(f"{'Gen':>4} | {'Best Energy (eV)':>15} | {'Avg Energy (eV)':>14} | {'Worst Energy (eV)':>15} | {'Range (meV)':>12} | {'Improvement (meV)':>12} | {'Est. Remaining':>15}")
        print("=" * 120)

        # Evolution loop
        for generation in range(max_generations):
            generation_start_time = time.time()

            new_population = []
            new_fitness = []

            # Elitism: keep best individuals
            n_elite = int(population_size * elitism_ratio)
            if n_elite > 0:
                elite_indices = np.argsort(fitness_scores)[:n_elite]
                for idx in elite_indices:
                    new_population.append(self.population[idx].copy())
                    new_fitness.append(fitness_scores[idx])

            # Generate offspring
            while len(new_population) < population_size:
                parents = self.tournament_selection(self.population, fitness_scores)

                if random.random() < self.ga_params.get('crossover_rate', 0.8):
                    child = self.crossover(parents[0], parents[1])
                else:
                    child = random.choice(parents).copy()

                child = self.mutate(child)
                child_fitness = self.calculate_fitness(child)

                new_population.append(child)
                new_fitness.append(child_fitness)

                if child_fitness < self.best_energy:
                    self.best_energy = child_fitness
                    self.best_individual = child.copy()

            self.population = new_population
            fitness_scores = new_fitness

            # Calculate generation statistics
            best_fitness = np.min(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            worst_fitness = np.max(fitness_scores)
            energy_range = (worst_fitness - best_fitness) * 1000  # Convert to meV

            # Calculate improvement from first generation
            if hasattr(self, 'first_generation_best'):
                improvement = (self.first_generation_best - best_fitness) * 1000  # meV
            else:
                self.first_generation_best = best_fitness
                improvement = 0.0

            # Time tracking and estimation
            generation_end_time = time.time()
            generation_duration = generation_end_time - generation_start_time
            generation_times.append(generation_duration)

            # Keep only recent generation times for better estimation (last 10 generations)
            if len(generation_times) > 10:
                generation_times = generation_times[-10:]

            # Calculate time estimates
            if len(generation_times) >= 2:
                avg_generation_time = np.mean(generation_times)
                remaining_generations = max_generations - generation - 1
                estimated_remaining_time = remaining_generations * avg_generation_time

                # Format remaining time
                if estimated_remaining_time < 60:
                    time_str = f"{estimated_remaining_time:.0f}s"
                elif estimated_remaining_time < 3600:
                    time_str = f"{estimated_remaining_time/60:.1f}m"
                else:
                    time_str = f"{estimated_remaining_time/3600:.1f}h"
            else:
                time_str = "Calculating..."

            # Store detailed history
            self.detailed_history.append({
                'generation': generation,
                'best': best_fitness,
                'average': avg_fitness,
                'worst': worst_fitness,
                'run_id': self.run_id,
                'generation_time': generation_duration,
                'improvement_meV': improvement
            })

            self.fitness_history.append({'generation': generation, 'best': best_fitness, 'average': avg_fitness})

            # Print generation information in table format
            print(f"{generation:>4} | {best_fitness:>15.6f} | {avg_fitness:>14.6f} | {worst_fitness:>15.6f} | {energy_range:>10.1f} | {improvement:>10.1f} | {time_str:>15}")

            # Check for convergence
            if generation > 20:
                recent_best = [f['best'] for f in self.fitness_history[-10:]]
                convergence_threshold = self.ga_params.get('convergence_threshold', 1e-6)
                if max(recent_best) - min(recent_best) < convergence_threshold:
                    print("=" * 120)
                    print(f"  ✅ Converged at generation {generation}")
                    break

        self.final_population = self.population.copy()
        self.final_fitness = fitness_scores.copy()

        # Final summary
        total_time = time.time() - total_start_time
        print("=" * 120)
        print(f"  ✅ GA run {self.run_id + 1} completed in {total_time:.1f}s")
        print(f"  Final best energy: {self.best_energy:.6f} eV")

        if hasattr(self, 'first_generation_best'):
            total_improvement = (self.first_generation_best - self.best_energy) * 1000
            print(f"  Total improvement: {total_improvement:.1f} meV")

        if generation_times:
            avg_gen_time = np.mean(generation_times)
            print(f"  Average generation time: {avg_gen_time:.1f}s")

        print("")

        return {
            'run_id': self.run_id,
            'best_structure': self.best_individual,
            'best_energy': self.best_energy,
            'fitness_history': self.fitness_history,
            'detailed_history': self.detailed_history,
            'final_population': self.final_population,
            'final_fitness': self.final_fitness,
            'substitutions': self.substitutions,
            'ga_params': self.ga_params,
            'total_time': total_time,
            'generation_times': generation_times
        }
'''


def _generate_ga_code(substitutions, ga_params, calc_formation_energy, supercell_info=None):
    """Generate code for GA structure optimization with supercell support, final generation saving, and concentration sweep support."""

    # Convert parameters to proper Python format instead of JSON
    def format_python_dict(d, indent=4):
        """Convert dictionary to properly formatted Python code."""
        if not isinstance(d, dict):
            if isinstance(d, bool):
                return str(d)  # Python True/False
            elif isinstance(d, str):
                return f'"{d}"'
            elif isinstance(d, (int, float)):
                return str(d)
            elif isinstance(d, list):
                items = [format_python_dict(item, 0) for item in d]
                return f"[{', '.join(items)}]"
            else:
                return str(d)

        lines = ["{"]
        for key, value in d.items():
            formatted_value = format_python_dict(value, 0)
            lines.append(f"{'    ' * (indent // 4 + 1)}'{key}': {formatted_value},")
        lines.append(f"{'    ' * (indent // 4)}}}")
        return '\n'.join(lines)

    substitutions_str = format_python_dict(substitutions, 8)
    ga_params_str = format_python_dict(ga_params, 8)
    calc_formation_energy_str = str(calc_formation_energy)

    # Generate supercell creation code
    supercell_code = ""
    if supercell_info and supercell_info.get('enabled', False):
        supercell_multipliers = supercell_info.get('multipliers', [1, 1, 1])
        supercell_code = f'''
    # Create supercell as specified
    supercell_multipliers = {supercell_multipliers}
    print(f"🔬 Creating supercell: {{supercell_multipliers[0]}}x{{supercell_multipliers[1]}}x{{supercell_multipliers[2]}}")
    print(f"📊 Original structure: {{base_structure.composition.reduced_formula}} ({{len(base_structure)}} atoms)")

    # Apply supercell transformation
    base_structure.make_supercell(supercell_multipliers)
    print(f"📊 Supercell structure: {{base_structure.composition.reduced_formula}} ({{len(base_structure)}} atoms)")

    # Update lattice information
    lattice = base_structure.lattice
    print(f"📏 Supercell lattice: a={{lattice.a:.3f}} Å, b={{lattice.b:.3f}} Å, c={{lattice.c:.3f}} Å")
    print(f"📦 Supercell volume: {{lattice.volume:.2f}} Å³")
    print(f"📈 Volume ratio: {{lattice.volume / (lattice.volume / np.prod(supercell_multipliers)):.1f}}x")
'''
    else:
        supercell_code = '''
    # Using original structure (no supercell)
    print(f"📊 Using original structure: {base_structure.composition.reduced_formula} ({len(base_structure)} atoms)")
'''

    # Main GA code with concentration sweep support
    code = f'''    structure_files = [f for f in os.listdir(".") if f.startswith("POSCAR") or f.endswith(".vasp") or f.endswith(".poscar")]

    if len(structure_files) == 0:
        print("❌ No structure files found!")
        return

    # Use the first structure file as base structure
    base_structure_file = structure_files[0]
    print(f"🧬 Using base structure file: {{base_structure_file}}")

    # Load base structure
    base_atoms = read(base_structure_file)

    # Convert to pymatgen structure
    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()
    base_structure = adaptor.get_structure(base_atoms)

    print(f"📊 Loaded original structure: {{base_structure.composition.reduced_formula}} ({{len(base_structure)}} atoms)")
{supercell_code}
    # GA configuration
    substitutions = {substitutions_str}

    ga_params = {ga_params_str}

    # Formation energy calculation flag
    calc_formation_energy = {calc_formation_energy_str}

    print(f"🎯 Substitutions configured: {{len(substitutions)}} element types")
    print(f"🧬 GA parameters: {{ga_params['population_size']}} individuals, {{ga_params['max_generations']}} generations, {{ga_params['num_runs']}} runs")

    # Generate concentration combinations for analysis
    concentration_combinations = generate_concentration_combinations(substitutions)
    concentration_combinations = sort_concentration_combinations(concentration_combinations)

    runs_per_concentration = ga_params.get('num_runs', 1)

    if len(concentration_combinations) > 1:
        print(f"🧬 Starting GA concentration sweep: {{len(concentration_combinations)}} combinations × {{runs_per_concentration}} runs each")
    else:
        print(f"🧬 Starting {{runs_per_concentration}} GA runs")

    # Validate substitutions against the final structure (after supercell creation)
    for element, sub_info in substitutions.items():
        element_sites = [i for i, site in enumerate(base_structure) if site.specie.symbol == element]
        if len(element_sites) == 0:
            print(f"❌ Element {{element}} not found in structure!")
            continue

        if 'concentration_list' in sub_info:
            conc_range = f"{{min(sub_info['concentration_list']) * 100:.1f}}-{{max(sub_info['concentration_list']) * 100:.1f}}%"
            print(f"  🔄 {{element}}: {{len(element_sites)}} sites → {{sub_info['new_element']}} ({{conc_range}})")
        else:
            expected_substitutions = int(len(element_sites) * sub_info['concentration'])
            print(f"  🔄 {{element}}: {{len(element_sites)}} sites → {{sub_info['new_element']}} ({{sub_info['concentration']*100:.1f}}% = {{expected_substitutions}} atoms)")

    # Calculate reference energies if needed
    reference_energies = {{}}'''

    if calc_formation_energy:
        code += '''
    print("🔬 Calculating atomic reference energies...")
    all_elements = set()
    for site in base_structure:
        all_elements.add(site.specie.symbol)

    # Add substitution elements
    for sub_info in substitutions.values():
        if sub_info['new_element'] != 'VACANCY':
            all_elements.add(sub_info['new_element'])

    print(f"🧪 Found elements: {', '.join(sorted(all_elements))}")

    for i, element in enumerate(sorted(all_elements)):
        print(f"  📍 Calculating reference for {element} ({i+1}/{len(all_elements)})...")
        atom = Atoms(element, positions=[(0, 0, 0)], cell=[20, 20, 20], pbc=True)
        atom.calc = calculator
        reference_energies[element] = atom.get_potential_energy()
        print(f"  ✅ {element}: {reference_energies[element]:.6f} eV")'''

    code += '''

    # Run GA optimization with concentration sweep support
    all_results = []
    combination_results = {}

    for combo_idx, combo_substitutions in enumerate(concentration_combinations):
        if len(concentration_combinations) > 1:
            combo_name = create_combination_name(combo_substitutions)
            print(f"\\n🔄 Starting combination {combo_idx + 1}/{len(concentration_combinations)}: {combo_name}")
            print("="*80)
        else:
            combo_name = "single_combination"

        combo_results = []

        for run_id in range(runs_per_concentration):
            if len(concentration_combinations) > 1:
                print(f"\\n🔄 Starting GA run {run_id + 1}/{runs_per_concentration} for {combo_name}")
                print(f"   └─ Combination {combo_idx + 1}/{len(concentration_combinations)} | Overall progress: {((combo_idx * runs_per_concentration + run_id) / (len(concentration_combinations) * runs_per_concentration)) * 100:.1f}%")
            else:
                print(f"\\n" + "="*80)
                print(f"🔄 Starting GA run {run_id + 1}/{runs_per_concentration}")
                print("="*80)

            try:
                optimizer = GeneticAlgorithmOptimizer(
                    base_structure, calculator, combo_substitutions, ga_params, run_id
                )

                results = optimizer.optimize()

                if results:
                    # Add combination info if multiple combinations
                    if len(concentration_combinations) > 1:
                        results['concentration_combination'] = combo_substitutions
                        results['combination_name'] = combo_name
                        results['combination_idx'] = combo_idx
                        results['run_within_combination'] = run_id + 1

                    combo_results.append(results)
                    all_results.append(results)

                    if len(concentration_combinations) > 1:
                        print(f"\\n✅ GA run {run_id + 1}/{runs_per_concentration} for {combo_name} completed: Best energy = {results['best_energy']:.6f} eV")
                    else:
                        print(f"\\n✅ GA run {run_id + 1}/{runs_per_concentration} completed: Best energy = {results['best_energy']:.6f} eV")

                    # Save individual run results
                    run_result = {
                        "run_id": run_id,
                        "combination_idx": combo_idx if len(concentration_combinations) > 1 else 0,
                        "combination_name": combo_name,
                        "best_energy_eV": results['best_energy'],
                        "final_generation": len(results['fitness_history']),
                        "converged": len(results['fitness_history']) < ga_params['max_generations'],
                        "final_population_size": len(results['final_population']),
                        "total_time_seconds": results.get('total_time', 0),
                        "calculation_type": "ga_structure_optimization"
                    }

                    # Save best structure for this run
                    best_structure = results['best_structure']
                    if best_structure:
                        # Convert to ASE and save
                        best_atoms = AseAtomsAdaptor().get_atoms(best_structure)

                        # Create combination-specific directory
                        if len(concentration_combinations) > 1:
                            combo_dir = f"ga_results/{combo_name}"
                            os.makedirs(combo_dir, exist_ok=True)
                            output_filename = f"{combo_dir}/best_structure_run_{run_id + 1:02d}.vasp"
                        else:
                            output_filename = f"ga_results/best_structure_run_{run_id + 1:02d}.vasp"

                        # Save as POSCAR
                        write(output_filename, best_atoms, format='vasp', direct=True, sort=True)
                        print(f"  💾 Saved best structure: {output_filename}")

                        run_result["output_structure"] = output_filename
                        run_result["best_formula"] = best_structure.composition.reduced_formula
                        run_result["best_num_atoms"] = len(best_structure)

                        # Calculate formation energy if needed
                        if calc_formation_energy:
                            formation_energy = calculate_formation_energy(
                                results['best_energy'], best_atoms, reference_energies
                            )
                            if formation_energy is not None:
                                run_result["formation_energy_eV_per_atom"] = formation_energy
                                print(f"  ✅ Formation energy: {formation_energy:.6f} eV/atom")

                    # Save fitness history
                    if len(concentration_combinations) > 1:
                        fitness_file = f"ga_results/{combo_name}/fitness_history_run_{run_id + 1:02d}.csv"
                    else:
                        fitness_file = f"ga_results/fitness_history_run_{run_id + 1:02d}.csv"

                    fitness_df = pd.DataFrame(results['fitness_history'])
                    fitness_df.to_csv(fitness_file, index=False)

                    # Save detailed history if available
                    if 'detailed_history' in results:
                        if len(concentration_combinations) > 1:
                            detailed_file = f"ga_results/{combo_name}/detailed_history_run_{run_id + 1:02d}.csv"
                        else:
                            detailed_file = f"ga_results/detailed_history_run_{run_id + 1:02d}.csv"

                        detailed_df = pd.DataFrame(results['detailed_history'])
                        detailed_df.to_csv(detailed_file, index=False)

                    # Save top 20% of final generation structures
                    if results['final_population'] and results['final_fitness']:
                        final_population = results['final_population']
                        final_fitness = results['final_fitness']

                        # Sort by fitness (energy) and get top 20%
                        sorted_indices = np.argsort(final_fitness)
                        top_20_percent = max(1, int(len(sorted_indices) * 0.2))
                        best_indices = sorted_indices[:top_20_percent]

                        print(f"  💾 Saving top {top_20_percent} structures ({len(best_indices)}) from final generation")

                        # Create directory for this run's final generation
                        if len(concentration_combinations) > 1:
                            final_gen_dir = f"ga_results/{combo_name}/run_{run_id + 1:02d}_final_generation_top20"
                        else:
                            final_gen_dir = f"ga_results/run_{run_id + 1:02d}_final_generation_top20"

                        os.makedirs(final_gen_dir, exist_ok=True)

                        for rank, idx in enumerate(best_indices):
                            structure = final_population[idx]
                            energy = final_fitness[idx]

                            # Convert to ASE and save as POSCAR
                            best_atoms = AseAtomsAdaptor().get_atoms(structure)

                            # Generate filename with rank and energy
                            poscar_filename = f"final_gen_rank_{rank+1:02d}_energy_{energy:.6f}eV.vasp"
                            poscar_path = os.path.join(final_gen_dir, poscar_filename)

                            write(poscar_path, best_atoms, format='vasp', direct=True, sort=True)

                        print(f"  ✅ Saved {len(best_indices)} final generation structures to {final_gen_dir}")

                else:
                    if len(concentration_combinations) > 1:
                        print(f"❌ GA run {run_id + 1}/{runs_per_concentration} for {combo_name} failed")
                    else:
                        print(f"❌ GA run {run_id + 1}/{runs_per_concentration} failed")

            except Exception as run_error:
                if len(concentration_combinations) > 1:
                    print(f"❌ GA run {run_id + 1}/{runs_per_concentration} for {combo_name} failed with error: {str(run_error)}")
                else:
                    print(f"❌ GA run {run_id + 1}/{runs_per_concentration} failed with error: {str(run_error)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                continue

        if combo_results:
            combination_results[combo_name] = combo_results
            if len(concentration_combinations) > 1:
                best_for_combination = min(combo_results, key=lambda x: x['best_energy'])
                print(f"\\n✅ Combination {combo_name} completed. Best energy: {best_for_combination['best_energy']:.6f} eV from {len(combo_results)} runs")
                print(f"   └─ Combination {combo_idx + 1}/{len(concentration_combinations)} finished")

    # Process overall results
    if all_results:
        best_overall = min(all_results, key=lambda x: x['best_energy'])

        if len(concentration_combinations) > 1:
            print(f"\\n🏆 Overall best energy from {len(all_results)} total runs: {best_overall['best_energy']:.6f} eV")
            print(f"🏆 Best combination: {best_overall.get('combination_name', 'Unknown')}")
        else:
            print(f"\\n" + "="*80)
            print(f"★ FINAL RESULTS - Best energy from {len(all_results)} runs: {best_overall['best_energy']:.6f} eV")
            print("="*80)

        # Save overall best structure
        overall_best_structure = best_overall['best_structure']
        if overall_best_structure:
            best_atoms = AseAtomsAdaptor().get_atoms(overall_best_structure)

            # Save as multiple formats
            base_name = "overall_best_structure"
            if len(concentration_combinations) > 1:
                base_name += f"_{best_overall.get('combination_name', 'unknown')}"

            # POSCAR format
            write(f"ga_results/{base_name}.vasp", best_atoms, format='vasp', direct=True, sort=True)
            print(f"💾 Saved overall best structure: ga_results/{base_name}.vasp")

            # CIF format
            try:
                from pymatgen.io.cif import CifWriter
                cif_writer = CifWriter(overall_best_structure)
                with open(f"ga_results/{base_name}.cif", 'w') as f:
                    f.write(str(cif_writer))
                print(f"💾 Saved overall best structure: ga_results/{base_name}.cif")
            except Exception as cif_error:
                print(f"⚠️ Could not save CIF: {cif_error}")

        # Create comparison dataframe
        comparison_data = []
        for result in all_results:
            row = {
                'Run_ID': result['run_id'] + 1,
                'Best_Energy_eV': result['best_energy'],
                'Generations': len(result['fitness_history']),
                'Converged': len(result['fitness_history']) < ga_params['max_generations'],
                'Final_Population_Size': len(result['final_population']),
                'Total_Time_s': result.get('total_time', 0)
            }

            # Add combination info if available
            if len(concentration_combinations) > 1:
                row['Combination_Name'] = result.get('combination_name', 'Unknown')
                row['Combination_Idx'] = result.get('combination_idx', 0)
                row['Run_Within_Combination'] = result.get('run_within_combination', 1)

            if result['best_structure']:
                row['Best_Formula'] = result['best_structure'].composition.reduced_formula
                row['Best_Num_Atoms'] = len(result['best_structure'])

                # Calculate formation energy if available
                if calc_formation_energy and reference_energies:
                    best_atoms = AseAtomsAdaptor().get_atoms(result['best_structure'])
                    formation_energy = calculate_formation_energy(
                        result['best_energy'], best_atoms, reference_energies
                    )
                    if formation_energy is not None:
                        row['Formation_Energy_eV_per_atom'] = formation_energy

            comparison_data.append(row)

        # Save comparison results
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv("ga_results/ga_runs_comparison.csv", index=False)
        print(f"💾 Saved comparison: ga_results/ga_runs_comparison.csv")

        # Save concentration sweep summary if multiple combinations
        if len(concentration_combinations) > 1:
            sweep_data = []
            for combo_name, combo_runs in combination_results.items():
                best_run = min(combo_runs, key=lambda x: x['best_energy'])
                sweep_data.append({
                    'Combination_Name': combo_name,
                    'Best_Energy_eV': best_run['best_energy'],
                    'Runs_Completed': len(combo_runs),
                    'Avg_Generations': np.mean([len(run['fitness_history']) for run in combo_runs]),
                    'Energy_Range_meV': (max(run['best_energy'] for run in combo_runs) - min(run['best_energy'] for run in combo_runs)) * 1000,
                    'Best_Formula': best_run['best_structure'].composition.reduced_formula if best_run['best_structure'] else 'Unknown'
                })

            df_sweep = pd.DataFrame(sweep_data)
            df_sweep.to_csv("ga_results/concentration_sweep_summary.csv", index=False)
            print(f"💾 Saved concentration sweep summary: ga_results/concentration_sweep_summary.csv")

        # Generate plots if matplotlib is available
        try:
            import matplotlib.pyplot as plt

            # Set global font sizes
            plt.rcParams.update({
                'font.size': 18,
                'axes.titlesize': 24,
                'axes.labelsize': 20,
                'xtick.labelsize': 18,
                'ytick.labelsize': 18,
                'legend.fontsize': 18,
                'figure.titlesize': 26
            })

            # 1. Best energy comparison across runs
            plt.figure(figsize=(16, 10))
            run_ids = [r['run_id'] + 1 for r in all_results]
            best_energies = [r['best_energy'] for r in all_results]

            # Color by combination if multiple combinations
            if len(concentration_combinations) > 1:
                colors = []
                color_map = plt.cm.tab10
                for r in all_results:
                    combo_idx = r.get('combination_idx', 0)
                    colors.append(color_map(combo_idx % 10))
            else:
                colors = ['green' if e == min(best_energies) else 'steelblue' for e in best_energies]

            bars = plt.bar(run_ids, best_energies, color=colors, alpha=0.7)

            plt.xlabel('GA Run', fontsize=22, fontweight='bold')
            plt.ylabel('Best Energy (eV)', fontsize=22, fontweight='bold')

            if len(concentration_combinations) > 1:
                plt.title('Best Energy Comparison Across GA Runs (Concentration Sweep)', fontsize=26, fontweight='bold', pad=20)
            else:
                plt.title('Best Energy Comparison Across GA Runs', fontsize=26, fontweight='bold', pad=20)

            plt.xticks(run_ids, fontsize=18, fontweight='bold')
            plt.yticks(fontsize=18, fontweight='bold')

            # Add value labels
            for bar, energy in zip(bars, best_energies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(best_energies)-min(best_energies))*0.01,
                        f'{energy:.4f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('ga_results/ga_best_energies_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Saved energy comparison plot: ga_results/ga_best_energies_comparison.png")

            # 2. Concentration sweep plot if multiple combinations
            if len(concentration_combinations) > 1 and combination_results:
                plt.figure(figsize=(16, 10))

                combinations = list(combination_results.keys())
                combo_best_energies = []
                for combo_name in combinations:
                    combo_runs = combination_results[combo_name]
                    best_energy = min(run['best_energy'] for run in combo_runs)
                    combo_best_energies.append(best_energy)

                # Find the best energy combination
                best_idx = combo_best_energies.index(min(combo_best_energies))
                colors = ['#28A745' if i == best_idx else '#667eea' for i in range(len(combo_best_energies))]

                bars = plt.bar(range(len(combinations)), combo_best_energies, color=colors, alpha=0.7)

                plt.xlabel('Concentration Combination', fontsize=22, fontweight='bold')
                plt.ylabel('Best Energy (eV)', fontsize=22, fontweight='bold')
                plt.title('Energy vs Concentration Combination', fontsize=26, fontweight='bold', pad=20)

                # Clean up combination names for display
                clean_names = [name.replace('_', ' ').replace('pct', '%') for name in combinations]
                plt.xticks(range(len(combinations)), clean_names, rotation=45, ha='right', fontsize=16, fontweight='bold')
                plt.yticks(fontsize=18, fontweight='bold')

                # Add value labels
                for bar, energy in zip(bars, combo_best_energies):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(combo_best_energies)-min(combo_best_energies))*0.01,
                            f'{energy:.4f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

                plt.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                plt.savefig('ga_results/concentration_sweep_energies.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ Saved concentration sweep plot: ga_results/concentration_sweep_energies.png")'''

    if calc_formation_energy:
        code += '''

            # 3. Formation energy comparison if available
            formation_energies = []
            valid_runs = []
            for i, result in enumerate(all_results):
                if result['best_structure'] and reference_energies:
                    best_atoms = AseAtomsAdaptor().get_atoms(result['best_structure'])
                    formation_energy = calculate_formation_energy(
                        result['best_energy'], best_atoms, reference_energies
                    )
                    if formation_energy is not None:
                        formation_energies.append(formation_energy)
                        valid_runs.append(i + 1)

            if formation_energies:
                plt.figure(figsize=(16, 10))
                colors = ['green' if fe == min(formation_energies) else 'orange' for fe in formation_energies]
                bars = plt.bar(valid_runs, formation_energies, color=colors, alpha=0.7)

                plt.xlabel('GA Run', fontsize=22, fontweight='bold')
                plt.ylabel('Formation Energy (eV/atom)', fontsize=22, fontweight='bold')
                plt.title('Formation Energy Comparison Across GA Runs', fontsize=26, fontweight='bold', pad=20)
                plt.xticks(valid_runs, fontsize=18, fontweight='bold')
                plt.yticks(fontsize=18, fontweight='bold')

                # Add value labels
                for bar, fe in zip(bars, formation_energies):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(formation_energies)-min(formation_energies))*0.02,
                            f'{fe:.4f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

                plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Stability Line')
                plt.legend(fontsize=18)
                plt.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                plt.savefig('ga_results/ga_formation_energies_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ Saved formation energy plot: ga_results/ga_formation_energies_comparison.png")'''

    code += '''

            # 4. Multi-Run Convergence Plot
            if len(all_results) > 1:
                plt.figure(figsize=(16, 10))

                # Color palette for different runs/combinations
                if len(concentration_combinations) > 1:
                    # Use different colors for different combinations
                    color_map = plt.cm.tab10
                    for i, result in enumerate(all_results):
                        if result['fitness_history']:
                            generations = [f['generation'] for f in result['fitness_history']]
                            best_energies = [f['best'] for f in result['fitness_history']]

                            combo_idx = result.get('combination_idx', 0)
                            combo_name = result.get('combination_name', 'Unknown')
                            color = color_map(combo_idx % 10)

                            plt.plot(generations, best_energies, 
                                    marker='o', markersize=4, linewidth=2.5, 
                                    color=color, alpha=0.8,
                                    label=f'{combo_name.replace("_", " ").replace("pct", "%")} - Run {result["run_id"] + 1}')
                else:
                    # Original single-combination coloring
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

                    for i, result in enumerate(all_results):
                        if result['fitness_history']:
                            generations = [f['generation'] for f in result['fitness_history']]
                            best_energies = [f['best'] for f in result['fitness_history']]

                            color = colors[i % len(colors)]

                            plt.plot(generations, best_energies, 
                                    marker='o', markersize=4, linewidth=2.5, 
                                    color=color, alpha=0.8,
                                    label=f'Run {result["run_id"] + 1} (Final: {result["best_energy"]:.6f} eV)')

                # Find and highlight the overall best
                best_run = min(all_results, key=lambda x: x['best_energy'])
                if best_run['fitness_history']:
                    generations = [f['generation'] for f in best_run['fitness_history']]
                    best_energies = [f['best'] for f in best_run['fitness_history']]

                    plt.plot(generations, best_energies, 
                            marker='s', markersize=6, linewidth=4, 
                            color='red', alpha=0.9,
                            label=f'★ BEST Run {best_run["run_id"] + 1}', zorder=10)

                plt.xlabel('Generation', fontsize=22, fontweight='bold')
                plt.ylabel('Best Energy (eV)', fontsize=22, fontweight='bold')

                if len(concentration_combinations) > 1:
                    plt.title('GA Convergence Comparison - Concentration Sweep', fontsize=26, fontweight='bold', pad=20)
                else:
                    plt.title('GA Convergence Comparison - All Runs', fontsize=26, fontweight='bold', pad=20)

                plt.legend(fontsize=16, loc='upper right')
                plt.grid(True, alpha=0.3)
                plt.xticks(fontsize=18, fontweight='bold')
                plt.yticks(fontsize=18, fontweight='bold')

                # Add statistics box
                best_energy = min(r['best_energy'] for r in all_results)
                worst_energy = max(r['best_energy'] for r in all_results)
                energy_range = (worst_energy - best_energy) * 1000  # Convert to meV
                avg_generations = np.mean([len(r['fitness_history']) for r in all_results])

                stats_text = f'Runs: {len(all_results)}\\n'
                stats_text += f'Best: {best_energy:.6f} eV\\n'
                stats_text += f'Range: {energy_range:.1f} meV\\n'
                stats_text += f'Avg Gen: {avg_generations:.0f}'

                if len(concentration_combinations) > 1:
                    stats_text += f'\\nCombinations: {len(concentration_combinations)}'

                plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        fontsize=14, fontweight='bold', verticalalignment='bottom')

                plt.tight_layout()
                plt.savefig('ga_results/multi_run_convergence_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ Saved multi-run convergence plot: ga_results/multi_run_convergence_comparison.png")

                # Reset matplotlib settings
                plt.rcParams.update(plt.rcParamsDefault)

        except ImportError:
            print("⚠️ Matplotlib not available. Install with: pip install matplotlib")
        except Exception as plot_error:
            print(f"⚠️ Error generating plots: {plot_error}")

        # Create comprehensive summary
        with open("ga_results/ga_optimization_summary.txt", "w") as f:
            f.write("MACE Genetic Algorithm Optimization Results\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(f"Total GA runs: {len(all_results)}\\n")
            f.write(f"Overall best energy: {best_overall['best_energy']:.6f} eV\\n")
            f.write(f"Best run ID: {best_overall['run_id'] + 1}\\n")

            if len(concentration_combinations) > 1:
                f.write(f"Total concentration combinations: {len(concentration_combinations)}\\n")
                f.write(f"Best combination: {best_overall.get('combination_name', 'Unknown')}\\n")

            if 'total_time' in best_overall:
                f.write(f"Best run time: {best_overall['total_time']:.1f} seconds\\n")

            total_time_all_runs = sum(r.get('total_time', 0) for r in all_results)
            f.write(f"Total computation time: {total_time_all_runs:.1f} seconds ({total_time_all_runs/60:.1f} minutes)\\n\\n")

            if best_overall['best_structure']:
                f.write(f"Best structure formula: {best_overall['best_structure'].composition.reduced_formula}\\n")
                f.write(f"Best structure atoms: {len(best_overall['best_structure'])}\\n\\n")

            f.write("Substitution Configuration:\\n")
            if len(concentration_combinations) > 1:
                best_combo = best_overall.get('concentration_combination', {})
                for element, sub_info in best_combo.items():
                    f.write(f"  {element} → {sub_info['new_element']} ({sub_info['concentration']*100:.1f}%)\\n")
            else:
                for element, sub_info in substitutions.items():
                    if 'concentration_list' in sub_info:
                        conc_str = f"{sub_info['concentration_list'][0]*100:.1f}%"
                    else:
                        conc_str = f"{sub_info['concentration']*100:.1f}%"
                    f.write(f"  {element} → {sub_info['new_element']} ({conc_str})\\n")
            f.write("\\n")

            f.write("GA Parameters:\\n")
            for param, value in ga_params.items():
                f.write(f"  {param}: {value}\\n")
            f.write("\\n")

            if len(concentration_combinations) > 1:
                f.write("Concentration Sweep Results:\\n")
                for combo_name, combo_runs in combination_results.items():
                    best_combo_run = min(combo_runs, key=lambda x: x['best_energy'])
                    f.write(f"  {combo_name}: {best_combo_run['best_energy']:.6f} eV ({len(combo_runs)} runs)\\n")
                f.write("\\n")

            f.write("Run-by-run Results:\\n")
            for result in all_results:
                runtime = result.get('total_time', 0)
                combo_info = ""
                if len(concentration_combinations) > 1:
                    combo_info = f" ({result.get('combination_name', 'Unknown')})"
                f.write(f"Run {result['run_id'] + 1}{combo_info}: {result['best_energy']:.6f} eV ({len(result['fitness_history'])} generations, {runtime:.1f}s)\\n")

                # Calculate formation energy if possible
                if calc_formation_energy and result['best_structure'] and reference_energies:
                    best_atoms = AseAtomsAdaptor().get_atoms(result['best_structure'])
                    formation_energy = calculate_formation_energy(
                        result['best_energy'], best_atoms, reference_energies
                    )
                    if formation_energy is not None:
                        f.write(f"  Formation Energy: {formation_energy:.6f} eV/atom\\n")

        print(f"💾 Saved comprehensive summary: ga_results/ga_optimization_summary.txt")

        # Save best structures from all runs in a ZIP file if multiple runs
        if len(all_results) > 1:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for i, result in enumerate(all_results):
                    if result['best_structure']:
                        best_atoms = AseAtomsAdaptor().get_atoms(result['best_structure'])

                        # Save as POSCAR in memory
                        from io import StringIO
                        poscar_buffer = StringIO()
                        write(poscar_buffer, best_atoms, format='vasp', direct=True, sort=True)
                        poscar_content = poscar_buffer.getvalue()

                        combo_suffix = ""
                        if len(concentration_combinations) > 1:
                            combo_name = result.get('combination_name', 'unknown')
                            combo_suffix = f"_{combo_name}"

                        filename = f"best_structure_run_{i+1:02d}{combo_suffix}_energy_{result['best_energy']:.6f}eV.vasp"
                        zip_file.writestr(filename, poscar_content)

                # Add summary to ZIP
                with open("ga_results/ga_optimization_summary.txt", 'r') as summary_file:
                    zip_file.writestr("GA_OPTIMIZATION_SUMMARY.txt", summary_file.read())

            # Save ZIP file
            with open("ga_results/all_best_structures.zip", 'wb') as zip_out:
                zip_out.write(zip_buffer.getvalue())

            print(f"💾 Saved all best structures: ga_results/all_best_structures.zip")

        # Save top 20% final generation structures from all runs in a separate ZIP file
        final_gen_zip_buffer = io.BytesIO()
        with zipfile.ZipFile(final_gen_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            total_final_gen_structures = 0

            for result in all_results:
                run_id = result['run_id']
                combo_name = result.get('combination_name', 'single')

                if result['final_population'] and result['final_fitness']:
                    final_population = result['final_population']
                    final_fitness = result['final_fitness']

                    # Sort by fitness and get top 20%
                    sorted_indices = np.argsort(final_fitness)
                    top_20_percent = max(1, int(len(sorted_indices) * 0.2))
                    best_indices = sorted_indices[:top_20_percent]
                    total_final_gen_structures += len(best_indices)

                    for rank, idx in enumerate(best_indices):
                        structure = final_population[idx]
                        energy = final_fitness[idx]

                        # Convert to ASE and save as POSCAR in memory
                        best_atoms = AseAtomsAdaptor().get_atoms(structure)

                        from io import StringIO
                        poscar_buffer = StringIO()
                        write(poscar_buffer, best_atoms, format='vasp', direct=True, sort=True)
                        poscar_content = poscar_buffer.getvalue()

                        if len(concentration_combinations) > 1:
                            filename = f"{combo_name}/run_{run_id+1:02d}/final_generation_rank_{rank+1:02d}_energy_{energy:.6f}eV.vasp"
                        else:
                            filename = f"run_{run_id+1:02d}/final_generation_rank_{rank+1:02d}_energy_{energy:.6f}eV.vasp"

                        zip_file.writestr(filename, poscar_content)

            # Add summary for final generation structures
            final_gen_summary = f"Top 20% Final Generation Structures from {len(all_results)} GA Runs\\n"
            final_gen_summary += f"Total structures included: {total_final_gen_structures}\\n"
            final_gen_summary += f"Selection criteria: Top 20% by energy from final generation\\n"
            final_gen_summary += f"Format: VASP POSCAR files\\n\\n"

            if len(concentration_combinations) > 1:
                final_gen_summary += f"Concentration combinations tested: {len(concentration_combinations)}\\n"
                final_gen_summary += "\\nCombination breakdown:\\n"
                for combo_name, combo_runs in combination_results.items():
                    combo_structures = 0
                    for result in combo_runs:
                        if result['final_population']:
                            final_population_size = len(result['final_population'])
                            top_20_count = max(1, int(final_population_size * 0.2))
                            combo_structures += top_20_count
                    final_gen_summary += f"  {combo_name}: {combo_structures} structures from {len(combo_runs)} runs\\n"
            else:
                final_gen_summary += "\\nRun-by-run breakdown:\\n"
                for result in all_results:
                    if result['final_population'] and result['final_fitness']:
                        final_population_size = len(result['final_population'])
                        top_20_count = max(1, int(final_population_size * 0.2))
                        final_gen_summary += f"Run {result['run_id'] + 1}: {top_20_count} structures from {final_population_size} final population\\n"

            zip_file.writestr("FINAL_GENERATION_README.txt", final_gen_summary)

        # Save final generation ZIP file
        with open("ga_results/final_generation_top20_all_runs.zip", 'wb') as zip_out:
            zip_out.write(final_gen_zip_buffer.getvalue())

        print(f"💾 Saved final generation top 20% structures: ga_results/final_generation_top20_all_runs.zip")
        print(f"📊 Total final generation structures saved: {total_final_gen_structures}")

    else:
        print("❌ No successful GA runs completed")

    print(f"\\n🏁 GA optimization completed!")
    print(f"📁 Results saved in ga_results/ directory")
    '''
    return code


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


def _generate_calculator_setup_code(model_size, device, selected_model_key=None, dtype="float64",
                                    mace_head=None, mace_dispersion=False, mace_dispersion_xc="pbe",
                                    custom_mace_path=None
                                    ):
    """Generate calculator setup code with support for all MLIP models."""
    if custom_mace_path:
        # Custom MACE model setup
        mace_args = []
        mace_args.append(f'model="{custom_mace_path}"')

        if mace_head:
            mace_args.append(f'head="{mace_head}"')

        if mace_dispersion:
            mace_args.append(f'dispersion=True')
            mace_args.append(f'dispersion_xc="{mace_dispersion_xc}"')

        mace_args.append(f'default_dtype="{dtype}"')
        mace_args.append(f'device=device')

        mace_args_str = ',\n        '.join(mace_args)

        calc_code = f'''    device = "{device}"
    print(f"🔧 Initializing MACE calculator with custom model...")
    print(f"📁 Custom model path: {custom_mace_path}")


    if not os.path.exists("{custom_mace_path}"):
        print(f"❌ Custom model file not found: {custom_mace_path}")
        print(f"Please ensure the model file exists at the specified path.")
        raise FileNotFoundError(f"Model file not found: {custom_mace_path}")

    try:
        from mace.calculators import mace_mp

        print(f"⚙️  Device: {{device}}")
        print(f"⚙️  Dtype: {dtype}")'''

        if mace_head:
            calc_code += f'''
        print(f"🎯 Head: {mace_head}")'''

        if mace_dispersion:
            calc_code += f'''
        print(f"🔬 Dispersion: D3-{mace_dispersion_xc}")'''

        calc_code += f'''

        calculator = mace_mp(
            {mace_args_str}
        )
        print(f"✅ Custom MACE model initialized successfully on {{device}}")

    except Exception as e:
        print(f"❌ Custom MACE initialization failed on {{device}}: {{e}}")
        if device == "cuda":
            print("⚠️ GPU initialization failed, falling back to CPU...")
            try:
                calculator = mace_mp(
                    {mace_args_str.replace('device=device', 'device="cpu"')}
                )
                print("✅ Custom MACE initialized successfully on CPU (fallback)")
            except Exception as cpu_error:
                print(f"❌ CPU fallback also failed: {{cpu_error}}")
                raise cpu_error
        else:
            raise e'''

        return calc_code
    # Determine model type from selected model key
    is_petmad = selected_model_key is not None and selected_model_key.startswith("PET-MAD")
    is_chgnet = selected_model_key is not None and selected_model_key.startswith("CHGNet")
    is_sevennet = selected_model_key is not None and selected_model_key.startswith("SevenNet")
    is_nequix = selected_model_key is not None and selected_model_key.startswith("Nequix")
    is_mattersim = selected_model_key is not None and selected_model_key.startswith("MatterSim")
    is_orb = selected_model_key is not None and selected_model_key.startswith("ORB")
    is_mace_off = selected_model_key is not None and "OFF" in selected_model_key
    is_url_model = isinstance(model_size, str) and (
            model_size.startswith("http://") or model_size.startswith("https://"))
    is_upet = selected_model_key is not None and selected_model_key.startswith("UPET")
    if is_nequix:
        calc_code = f'''    device = "{device}"
    print(f"🔧 Initializing Nequix calculator...")
    try:
        from nequix.calculator import NequixCalculator

        print(f"🎯 Using Nequix model: {model_size}")

        calculator = NequixCalculator("{model_size}")
        print(f"✅ Nequix {model_size} initialized successfully")

    except Exception as e:
        print(f"❌ Nequix initialization failed: {{e}}")
        raise e'''
    elif is_upet:
        upet_raw = model_size

        if upet_raw.startswith("upet:"):
            upet_raw = upet_raw[len("upet:"):]

        if "::" in upet_raw:
            upet_model_name, upet_version = upet_raw.split("::", 1)
        elif ":" in upet_raw:
            upet_model_name, upet_version = upet_raw.split(":", 1)
        else:
            upet_model_name = upet_raw
            upet_version = "latest"

        calc_code = f'''    device = "{device}"
    print(f"🔧 Initializing UPET calculator on {{device}}...")
    try:
        from upet.calculator import UPETCalculator

        upet_model_name = "{upet_model_name}"
        upet_version = "{upet_version}"

        print(f"🎯 Model: {{upet_model_name}}, Version: {{upet_version}}")
        print(f"⚙️  Device: {{device}}")
        print(f"⚙️  Dtype: {dtype}")

        calculator = UPETCalculator(
            model=upet_model_name,
            version=upet_version,
            device=device,
        )
        print(f"✅ UPET {{upet_model_name}} v{{upet_version}} initialized successfully on {{device}}")

    except Exception as e:
        print(f"❌ UPET initialization failed on {{device}}: {{e}}")
        if device == "cuda":
            print("⚠️ GPU initialization failed, falling back to CPU...")
            try:
                calculator = UPETCalculator(
                    model=upet_model_name,
                    version=upet_version,
                    device="cpu",
                )
                print("✅ UPET initialized successfully on CPU (fallback)")
            except Exception as cpu_error:
                print(f"❌ CPU fallback also failed: {{cpu_error}}")
                raise cpu_error
        else:
            raise e'''
    elif is_petmad:
        calc_code = f'''    device = "{device}"
    print(f"🔧 Initializing PET-MAD calculator...")
    try:
        from pet_mad.calculator import PETMADCalculator

        print(f"🎯 Using PET-MAD v1.0.2 (universal)")

        calculator = PETMADCalculator(
            version="v1.0.2",
            device=device
        )
        print(f"✅ PET-MAD v1.0.2 initialized successfully on {{device}}")
        print("   Trained on MAD dataset (95,595 structures)")

    except Exception as e:
        print(f"❌ PET-MAD initialization failed on {{device}}: {{e}}")
        if device == "cuda":
            print("⚠️ GPU initialization failed, falling back to CPU...")
            try:
                calculator = PETMADCalculator(
                    version="v1.0.2",
                    device="cpu"
                )
                print("✅ PET-MAD initialized successfully on CPU (fallback)")
            except Exception as cpu_error:
                print(f"❌ CPU fallback also failed: {{cpu_error}}")
                raise cpu_error
        else:
            raise e'''
    elif is_orb:
        # ORB setup
        calc_code = f'''    device = "{device}"
    print(f"🔧 Initializing ORB calculator on {{device}}...")
    try:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        # Convert dtype to ORB precision format
        if "{dtype}" == "float32":
            precision = "float32-high"  # Recommended for GPU acceleration
        else:
            precision = "float32-highest"  # Higher precision option

        print(f"🎯 Using precision: {{precision}}")

        # Get the pretrained model function by name
        model_function = getattr(pretrained, "{model_size}")
        orbff = model_function(
            device=device,
            precision=precision
        )
        calculator = ORBCalculator(orbff, device=device)
        print(f"✅ ORB {model_size} initialized successfully on {{device}}")

    except Exception as e:
        print(f"❌ ORB initialization failed on {{device}}: {{e}}")
        if device == "cuda":
            print("⚠️ GPU initialization failed, falling back to CPU...")
            try:
                model_function = getattr(pretrained, "{model_size}")
                orbff = model_function(
                    device="cpu",
                    precision=precision
                )
                calculator = ORBCalculator(orbff, device="cpu")
                print("✅ ORB initialized successfully on CPU (fallback)")
            except Exception as cpu_error:
                print(f"❌ CPU fallback also failed: {{cpu_error}}")
                raise cpu_error
        else:
            raise e'''

    elif is_mattersim:
        # MatterSim setup
        calc_code = f'''    device = "{device}"
    print(f"🔧 Initializing MatterSim calculator on {{device}}...")
    try:
        from mattersim.forcefield import MatterSimCalculator

        # Determine model path based on model_size
        if "{model_size}" == "mattersim-1m":
            model_path = "MatterSim-v1.0.0-1M.pth"
        elif "{model_size}" == "mattersim-5m":
            model_path = "MatterSim-v1.0.0-5M.pth"
        else:
            model_path = "{model_size}"

        print(f"📁 Model path: {{model_path}}")

        calculator = MatterSimCalculator(
            model_path=model_path,
            device=device
        )
        print(f"✅ MatterSim {{model_path}} initialized successfully on {{device}}")

    except Exception as e:
        print(f"❌ MatterSim initialization failed on {{device}}: {{e}}")
        if device == "cuda":
            print("⚠️ GPU initialization failed, falling back to CPU...")
            try:
                calculator = MatterSimCalculator(
                    model_path=model_path,
                    device="cpu"
                )
                print("✅ MatterSim initialized successfully on CPU (fallback)")
            except Exception as cpu_error:
                print(f"❌ CPU fallback also failed: {{cpu_error}}")
                raise cpu_error
        else:
            raise e'''

    elif is_sevennet:
        # SevenNet setup
        calc_code = f'''    device = "{device}"
    print(f"🔧 Initializing SevenNet calculator on {{device}}...")
    try:
        from sevenn.calculator import SevenNetCalculator

        print(f"🎯 Selected model: {selected_model_key}")
        print(f"🎯 Model function: {model_size}")

        # Parse model and modal from the model_size
        if "{model_size}" == "7net-mf-ompa-mpa":
            calculator = SevenNetCalculator(model='7net-mf-ompa', modal='mpa', device=device)
            print("✅ SevenNet 7net-mf-ompa (MPA modal) initialized successfully")
        elif "{model_size}" == "7net-mf-ompa-omat24":
            calculator = SevenNetCalculator(model='7net-mf-ompa', modal='omat24', device=device)
            print("✅ SevenNet 7net-mf-ompa (OMat24 modal) initialized successfully")
        else:
            # Standard models without modal parameter
            calculator = SevenNetCalculator(model="{model_size}", device=device)
            print(f"✅ SevenNet {model_size} initialized successfully on {{device}}")

    except Exception as e:
        print(f"❌ SevenNet initialization failed on {{device}}: {{e}}")
        if device == "cuda":
            print("⚠️ GPU initialization failed, falling back to CPU...")
            try:
                if "{model_size}" == "7net-mf-ompa-mpa":
                    calculator = SevenNetCalculator(model='7net-mf-ompa', modal='mpa', device="cpu")
                elif "{model_size}" == "7net-mf-ompa-omat24":
                    calculator = SevenNetCalculator(model='7net-mf-ompa', modal='omat24', device="cpu")
                else:
                    calculator = SevenNetCalculator(model="{model_size}", device="cpu")
                print("✅ SevenNet initialized successfully on CPU (fallback)")
            except Exception as cpu_error:
                print(f"❌ CPU fallback also failed: {{cpu_error}}")
                raise cpu_error
        else:
            raise e'''

    elif is_chgnet:
        # CHGNet setup
        chgnet_version = model_size.split("-")[1] if "-" in model_size else "0.3.0"

        calc_code = f'''    device = "{device}"
    print(f"🔧 Initializing CHGNet calculator on {{device}}...")
    try:
        from chgnet.model.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator

        chgnet = CHGNet.load(model_name="{chgnet_version}", use_device=device, verbose=False)
        calculator = CHGNetCalculator(model=chgnet, use_device=device)
        print(f"✅ CHGNet {chgnet_version} initialized successfully on {{device}}")

    except Exception as e:
        print(f"❌ CHGNet initialization failed on {{device}}: {{e}}")
        if device == "cuda":
            print("⚠️ GPU initialization failed, falling back to CPU...")
            try:
                chgnet = CHGNet.load(model_name="{chgnet_version}", use_device="cpu", verbose=False)
                calculator = CHGNetCalculator(model=chgnet, use_device="cpu")
                print("✅ CHGNet initialized successfully on CPU (fallback)")
            except Exception as cpu_error:
                print(f"❌ CPU fallback also failed: {{cpu_error}}")
                raise cpu_error
        else:
            raise e'''

    elif is_mace_off:
        # MACE-OFF setup
        calc_code = f'''    device = "{device}"
    print(f"🔧 Initializing MACE-OFF calculator on {{device}}...")
    try:
        from mace.calculators import mace_off

        calculator = mace_off(
            model="{model_size}", default_dtype="{dtype}", device=device)
        print(f"✅ MACE-OFF calculator initialized successfully on {{device}}")

    except Exception as e:
        print(f"❌ MACE-OFF initialization failed on {{device}}: {{e}}")
        if device == "cuda":
            print("⚠️ GPU initialization failed, falling back to CPU...")
            try:
                calculator = mace_off(
                    model="{model_size}", default_dtype="{dtype}", device="cpu")
                print("✅ MACE-OFF calculator initialized successfully on CPU (fallback)")
            except Exception as cpu_error:
                print(f"❌ CPU fallback also failed: {{cpu_error}}")
                raise cpu_error
        else:
            raise e'''

    elif is_url_model:
        # NEW: URL-based foundation models (e.g., MACE-MH-0, MACE-MH-1, MACE-MATPES)
        model_filename = model_size.split("/")[-1]

        # Build the mace_mp arguments
        mace_args = []
        mace_args.append(f'model=local_model_path')
        mace_args.append(f'device=device')
        mace_args.append(f'default_dtype="{dtype}"')

        if mace_head:
            mace_args.append(f'head="{mace_head}"')

        if mace_dispersion:
            mace_args.append(f'dispersion=True')
            mace_args.append(f'dispersion_xc="{mace_dispersion_xc}"')

        mace_args_str = ',\n        '.join(mace_args)

        calc_code = f'''    device = "{device}"
    print(f"🔧 Initializing MACE foundation model from URL...")

    def download_mace_model(model_url):
        """Download MACE model from URL and cache it."""
        from pathlib import Path
        import urllib.request

        model_filename = model_url.split("/")[-1]
        cache_dir = Path.home() / ".cache" / "mace_foundation_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / model_filename

        if model_path.exists():
            print(f"✅ Using cached model: {{model_filename}}")
            return str(model_path)

        print(f"📥 Downloading {{model_filename}}... (this may take a few minutes)")
        try:
            urllib.request.urlretrieve(model_url, str(model_path))
            print(f"✅ Model downloaded and cached")
            return str(model_path)
        except Exception as e:
            print(f"❌ Download failed: {{e}}")
            if model_path.exists():
                model_path.unlink()
            raise

    try:
        from mace.calculators import mace_mp

        model_url = "{model_size}"
        local_model_path = download_mace_model(model_url)
        print(f"📁 Model path: {{local_model_path}}")

        print(f"⚙️  Device: {{device}}")
        print(f"⚙️  Dtype: {dtype}")'''

        if mace_head:
            calc_code += f'''
        print(f"🎯 Head: {mace_head}")'''

        if mace_dispersion:
            calc_code += f'''
        print(f"🔬 Dispersion: D3-{mace_dispersion_xc}")'''

        calc_code += f'''

        calculator = mace_mp(
            {mace_args_str}
        )
        print(f"✅ MACE foundation model initialized successfully on {{device}}")

    except Exception as e:
        print(f"❌ MACE initialization failed on {{device}}: {{e}}")
        if device == "cuda":
            print("⚠️ GPU initialization failed, falling back to CPU...")
            try:
                calculator = mace_mp(
                    {mace_args_str.replace('device=device', 'device="cpu"')}
                )
                print("✅ MACE initialized successfully on CPU (fallback)")
            except Exception as cpu_error:
                print(f"❌ CPU fallback also failed: {{cpu_error}}")
                raise cpu_error
        else:
            raise e'''

    else:
        # MACE-MP setup (default) - now with optional dispersion
        mace_args = []
        mace_args.append(f'model="{model_size}"')

        if mace_dispersion:
            mace_args.append(f'dispersion=True')
            mace_args.append(f'dispersion_xc="{mace_dispersion_xc}"')
        else:
            mace_args.append(f'dispersion=False')

        mace_args.append(f'default_dtype="{dtype}"')
        mace_args.append(f'device=device')

        mace_args_str = ', '.join(mace_args)

        calc_code = f'''    device = "{device}"
    print(f"🔧 Initializing MACE-MP calculator on {{device}}...")'''

        if mace_dispersion:
            calc_code += f'''
        print(f"🔬 Dispersion correction: D3-{mace_dispersion_xc}")'''

        calc_code += f'''
    try:
        from mace.calculators import mace_mp

        calculator = mace_mp({mace_args_str})
        print(f"✅ MACE-MP calculator initialized successfully on {{device}}")

    except Exception as e:
        print(f"❌ MACE-MP initialization failed on {{device}}: {{e}}")
        if device == "cuda":
            print("⚠️ GPU initialization failed, falling back to CPU...")
            try:
                calculator = mace_mp({mace_args_str.replace('device=device', 'device="cpu"')})
                print("✅ MACE-MP calculator initialized successfully on CPU (fallback)")
            except Exception as cpu_error:
                print(f"❌ CPU fallback also failed: {{cpu_error}}")
                raise cpu_error
        else:
            raise e'''

    return calc_code


def _generate_energy_only_code(calc_formation_energy):
    """Generate code for energy-only calculations."""
    code = '''    structure_files = sorted([f for f in os.listdir(".") if f.startswith("POSCAR") or f.endswith(".vasp") or f.endswith(".poscar")])
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
        print(f"  ✅ {element}: {reference_energies[element]:.6f} eV")
        iso_xyz_filename = f"optimized_structures/IsolatedAtom_{element}.xyz"
        forces = atom.get_forces()
        with open(iso_xyz_filename, "w") as f:
            f.write("1\\n")
            lattice_string = "20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0"
            f.write(
                f'Lattice="{lattice_string}" '
                'Properties=species:S:1:pos:R:3:forces:R:3 '
                f'config_type=IsolatedAtom Energy={reference_energies[element]:.6f} pbc="F F F"\\n'
            )
            f.write(
                f"{element}  {0.0:12.6f} {0.0:12.6f} {0.0:12.6f} "
                f"{forces[0][0]:12.6f} {forces[0][1]:12.6f} {forces[0][2]:12.6f}\\n"
            )
        print(f"  💾 Saved isolated atom XYZ: {iso_xyz_filename}")'''

    code += '''

    for i, filename in enumerate(structure_files):
        print(f"\\n📊 Processing structure {i+1}/{len(structure_files)}: {filename}")
        structure_start_time = time.time()
        try:
            atoms = read(filename)
            atoms.calc = calculator

            print(f"  🔬 Calculating energy for {len(atoms)} atoms...")
            energy = atoms.get_potential_energy()

            # Calculate forces
            print(f"  🔬 Calculating forces...")
            forces = atoms.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))

            # Get lattice parameters
            lattice = get_lattice_parameters(atoms)

            # Save XYZ file with lattice and force information
            base_name = filename.replace('.vasp', '').replace('.poscar', '').replace('POSCAR', '').replace('POSCAR_', '')
            if not base_name:
                base_name = f"structure_{i+1}"

            #xyz_filename = f"results/{base_name}_energy.xyz"
            xyz_filename = f"optimized_structures/{i+1}_{base_name}.xyz"
            print(f"  💾 Saving XYZ file: {xyz_filename}")

            with open(xyz_filename, 'w') as xyz_file:
                num_atoms = len(atoms)
                cell_matrix = atoms.get_cell()
                lattice_string = " ".join([f"{x:.6f}" for row in cell_matrix for x in row])

                # Write number of atoms
                xyz_file.write(f"{num_atoms}\\n")

                # Write comment line with all information
                comment = (f'Energy={energy:.6f} Max_Force={max_force:.6f} '
                          f'a={lattice["a"]:.6f} b={lattice["b"]:.6f} c={lattice["c"]:.6f} '
                          f'alpha={lattice["alpha"]:.3f} beta={lattice["beta"]:.3f} gamma={lattice["gamma"]:.3f} '
                          f'Volume={lattice["volume"]:.6f} '
                          f'Lattice="{lattice_string}" '
                          f'Properties=species:S:1:pos:R:3:forces:R:3')
                xyz_file.write(f"{comment}\\n")

                # Write atomic positions and forces
                symbols = atoms.get_chemical_symbols()
                positions = atoms.get_positions()
                for j, (symbol, pos, force) in enumerate(zip(symbols, positions, forces)):
                    xyz_file.write(f"{symbol} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f} "
                                 f"{force[0]:12.6f} {force[1]:12.6f} {force[2]:12.6f}\\n")

            result = {
                "structure": filename,
                "energy_eV": energy,
                "max_force_eV_per_A": max_force,
                "calculation_type": "energy_only",
                "num_atoms": len(atoms),
                "xyz_file": xyz_filename
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
            print(f"  💾 Results updated and saved")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({"structure": filename, "error": str(e)})

            # Save results even for failed structures
            df_results = pd.DataFrame(results)
            df_results.to_csv("results/energy_results.csv", index=False)
            print(f"  💾 Results updated and saved (with error)")

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

    # Add plotting functionality
    code += '''
    # Generate energy plots
    print("\\n📊 Generating energy plots...")
    successful_results = [r for r in results if "error" not in r]

    if len(successful_results) > 0:
        try:
            import matplotlib.pyplot as plt

            # Set global font sizes
            plt.rcParams.update({
                'font.size': 18,
                'axes.titlesize': 24,
                'axes.labelsize': 20,
                'xtick.labelsize': 18,
                'ytick.labelsize': 18,
                'legend.fontsize': 18,
                'figure.titlesize': 26
            })

            # Prepare data
            structure_names = [r["structure"] for r in successful_results]
            energies = [r["energy_eV"] for r in successful_results]

            # 1. Total Energy Plot
            plt.figure(figsize=(16, 12))
            bars = plt.bar(range(len(structure_names)), energies, color='steelblue', alpha=0.7)
            plt.xlabel('Structure', fontsize=22, fontweight='bold')
            plt.ylabel('Total Energy (eV)', fontsize=22, fontweight='bold')
            plt.title('Total Energy Comparison', fontsize=26, fontweight='bold', pad=20)
            plt.xticks(range(len(structure_names)), [name.replace('.vasp', '').replace('POSCAR_', '') for name in structure_names], 
                      rotation=45, ha='right', fontsize=18, fontweight='bold')
            plt.yticks(fontsize=18, fontweight='bold')

            # Extend y-axis to accommodate labels above bars
            y_min, y_max = plt.ylim()
            y_range = y_max - y_min
            plt.ylim(y_min, y_max + y_range * 0.15)

            # Add vertical value labels above bars
            for i, (bar, energy) in enumerate(zip(bars, energies)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_range * 0.02, 
                        f'{energy:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold', 
                        rotation=90, color='black')

            plt.tight_layout()
            plt.savefig('results/total_energy_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✅ Saved total energy plot: results/total_energy_comparison.png")'''

    if calc_formation_energy:
        code += '''

            # 2. Formation Energy Plot
            formation_energies = [r.get("formation_energy_eV_per_atom") for r in successful_results]
            valid_formation = [(name, fe) for name, fe in zip(structure_names, formation_energies) if fe is not None]

            if valid_formation:
                valid_names, valid_fe = zip(*valid_formation)

                plt.figure(figsize=(16, 12))
                colors = ['green' if fe == min(valid_fe) else 'orange' for fe in valid_fe]
                bars = plt.bar(range(len(valid_names)), valid_fe, color=colors, alpha=0.7)
                plt.xlabel('Structure', fontsize=22, fontweight='bold')
                plt.ylabel('Formation Energy (eV/atom)', fontsize=22, fontweight='bold')
                plt.title('Formation Energy per Atom Comparison', fontsize=26, fontweight='bold', pad=20)
                plt.xticks(range(len(valid_names)), [name.replace('.vasp', '').replace('POSCAR_', '') for name in valid_names], 
                          rotation=45, ha='right', fontsize=18, fontweight='bold')
                plt.yticks(fontsize=18, fontweight='bold')

                # Extend y-axis to accommodate labels (handle positive and negative values)
                y_min, y_max = plt.ylim()
                y_range = y_max - y_min

                # Check if we have negative values
                has_negative = any(fe < 0 for fe in valid_fe)
                has_positive = any(fe > 0 for fe in valid_fe)

                if has_negative and has_positive:
                    plt.ylim(y_min - y_range * 0.15, y_max + y_range * 0.15)
                elif has_negative and not has_positive:
                    plt.ylim(y_min - y_range * 0.15, y_max + y_range * 0.05)
                else:
                    plt.ylim(y_min - y_range * 0.05, y_max + y_range * 0.15)

                # Add vertical value labels outside bars
                for i, (bar, fe) in enumerate(zip(bars, valid_fe)):
                    if fe >= 0:
                        y_pos = bar.get_height() + y_range * 0.02
                        va_align = 'bottom'
                    else:
                        y_pos = bar.get_height() - y_range * 0.02
                        va_align = 'top'
                    plt.text(bar.get_x() + bar.get_width()/2, y_pos, 
                            f'{fe:.4f}', ha='center', va=va_align, fontsize=16, fontweight='bold', 
                            rotation=90, color='black')

                plt.tight_layout()
                plt.savefig('results/formation_energy_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("  ✅ Saved formation energy plot: results/formation_energy_comparison.png")'''

    code += '''

            # 3. Relative Energy Plot
            if len(energies) > 1:
                min_energy = min(energies)
                relative_energies = [(e - min_energy) * 1000 for e in energies]  # Convert to meV

                plt.figure(figsize=(16, 12))
                colors = ['green' if re == 0 else 'orange' for re in relative_energies]
                bars = plt.bar(range(len(structure_names)), relative_energies, color=colors, alpha=0.7)
                plt.xlabel('Structure', fontsize=22, fontweight='bold')
                plt.ylabel('Relative Energy (meV)', fontsize=22, fontweight='bold')
                plt.title('Relative Energy Comparison (vs. Lowest Energy)', fontsize=26, fontweight='bold', pad=20)
                plt.xticks(range(len(structure_names)), [name.replace('.vasp', '').replace('POSCAR_', '') for name in structure_names], 
                          rotation=45, ha='right', fontsize=18, fontweight='bold')
                plt.yticks(fontsize=18, fontweight='bold')

                # Extend y-axis to accommodate labels above bars
                y_min, y_max = plt.ylim()
                y_range = max(relative_energies) if max(relative_energies) > 0 else 1
                plt.ylim(-y_range * 0.1, max(relative_energies) + y_range * 0.15)

                # Add vertical value labels above bars
                for i, (bar, re) in enumerate(zip(bars, relative_energies)):
                    if re > 0:
                        y_pos = bar.get_height() + y_range * 0.02
                        va_align = 'bottom'
                    else:
                        y_pos = y_range * 0.05  # Position above zero line for zero values
                        va_align = 'bottom'
                    plt.text(bar.get_x() + bar.get_width()/2, y_pos, 
                            f'{re:.1f}', ha='center', va=va_align, fontsize=16, fontweight='bold', 
                            rotation=90, color='black')

                plt.tight_layout()
                plt.savefig('results/relative_energy_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("  ✅ Saved relative energy plot: results/relative_energy_comparison.png")

            # Reset matplotlib settings
            plt.rcParams.update(plt.rcParamsDefault)

        except ImportError:
            print("  ⚠️ Matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"  ⚠️ Error generating plots: {e}")

    else:
        print("  ℹ️ No successful calculations to plot")
'''
    return code


def _generate_elastic_code(elastic_params, optimization_params, calc_formation_energy):
    """Generate code for elastic property calculations."""
    strain_magnitude = elastic_params.get('strain_magnitude', 0.01)

    code = f'''    structure_files = sorted([f for f in os.listdir(".") if f.startswith("POSCAR") or f.endswith(".vasp") or f.endswith(".poscar")])
    results = []
    print(f"🔧 Found {{len(structure_files)}} structure files for elastic calculations")

    strain_magnitude = {strain_magnitude}
    pre_opt_steps = {optimization_params.get('max_steps', 400)}
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

            # Create a simple logger for pre-optimization
            class PreOptLogger:
                def __init__(self, max_steps=50):
                    self.step_count = 0
                    self.max_steps = max_steps
                    self.previous_energy = None

                def __call__(self, optimizer=None):
                    if optimizer is not None and hasattr(optimizer, 'atoms'):
                        atoms_obj = optimizer.atoms
                        forces = atoms_obj.get_forces()
                        max_force = np.max(np.linalg.norm(forces, axis=1))
                        energy = atoms_obj.get_potential_energy()
                        energy_per_atom = energy / len(atoms_obj)

                        if self.previous_energy is not None:
                            energy_change = abs(energy - self.previous_energy)
                            energy_change_per_atom = energy_change / len(atoms_obj)
                        else:
                            energy_change = float('inf')
                            energy_change_per_atom = float('inf')
                        self.previous_energy = energy

                        try:
                            stress = atoms_obj.get_stress(voigt=True)
                            max_stress = np.max(np.abs(stress))
                        except:
                            max_stress = 0.0

                        self.step_count += 1
                        print(f"    Pre-opt step {self.step_count}: E={energy:.6f} eV ({energy_per_atom:.6f} eV/atom), "
                              f"F_max={max_force:.4f} eV/Å, Max_Stress={max_stress:.4f} GPa, "
                              f"ΔE={energy_change_per_atom:.2e} eV/atom")

            pre_opt_logger = PreOptLogger(pre_opt_steps)
            temp_optimizer = LBFGS(temp_atoms, logfile=None)
            temp_optimizer.attach(lambda: pre_opt_logger(temp_optimizer), interval=1)
            temp_optimizer.run(fmax=0.01, steps=pre_opt_steps)
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


            result = {
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
            }'''

    if calc_formation_energy:
        code += '''

            formation_energy = calculate_formation_energy(E0, atoms, reference_energies)
            result["formation_energy_eV_per_atom"] = formation_energy'''

    code += '''

            save_elastic_constants_to_csv(filename, C_GPa)
            elastic_data_dict = {
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
            }'''

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
            print(f"  💾 Results updated and saved")

        except Exception as e:
            print(f"  ❌ Elastic calculation failed: {e}")
            results.append({{"structure": filename, "error": str(e)}})

            # Save results even for failed structures
            df_results = pd.DataFrame(results)
            df_results.to_csv("results/elastic_results.csv", index=False)
            print(f"  💾 Results updated and saved(with error)")

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

    # Add plotting functionality for elastic properties
    code += '''
    # Generate elastic properties plots
    print("\\n📊 Generating elastic properties plots...")
    successful_results = [r for r in results if "error" not in r]

    if len(successful_results) > 0:
        try:
            import matplotlib.pyplot as plt

            # Set global font sizes
            plt.rcParams.update({
                'font.size': 18,
                'axes.titlesize': 24,
                'axes.labelsize': 20,
                'xtick.labelsize': 18,
                'ytick.labelsize': 18,
                'legend.fontsize': 18,
                'figure.titlesize': 26
            })

            # Prepare data
            structure_names = [r["structure"] for r in successful_results]
            energies = [r["energy_eV"] for r in successful_results]

            # 1. Total Energy Plot
            plt.figure(figsize=(16, 10))
            bars = plt.bar(range(len(structure_names)), energies, color='steelblue', alpha=0.7)
            plt.xlabel('Structure', fontsize=22, fontweight='bold')
            plt.ylabel('Total Energy (eV)', fontsize=22, fontweight='bold')
            plt.title('Total Energy (Elastic Properties Calculation)', fontsize=26, fontweight='bold', pad=20)
            plt.xticks(range(len(structure_names)), [name.replace('.vasp', '').replace('POSCAR_', '') for name in structure_names], 
                      rotation=45, ha='right', fontsize=18, fontweight='bold')
            plt.yticks(fontsize=18, fontweight='bold')

            # Add value labels and stability indicators
            for i, (bar, energy, result) in enumerate(zip(bars, energies, successful_results)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(energies)*0.01, 
                        f'{energy:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
                # Add mechanical stability indicator
                stable = result.get("mechanically_stable", False)
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - abs(max(energies)-min(energies))*0.05, 
                        '✓' if stable else '✗', ha='center', va='center', 
                        fontsize=20, color='green' if stable else 'red', fontweight='bold')

            plt.tight_layout()
            plt.savefig('results/elastic_total_energy_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✅ Saved total energy plot: results/elastic_total_energy_comparison.png")'''

    if calc_formation_energy:
        code += '''

            # 2. Formation Energy Plot
            formation_energies = [r.get("formation_energy_eV_per_atom") for r in successful_results]
            valid_formation = [(name, fe, result) for name, fe, result in zip(structure_names, formation_energies, successful_results) if fe is not None]

            if valid_formation:
                valid_names, valid_fe, valid_results = zip(*valid_formation)

                plt.figure(figsize=(16, 10))
                colors = ['green' if fe == min(valid_fe) else 'orange' for fe in valid_fe]
                bars = plt.bar(range(len(valid_names)), valid_fe, color=colors, alpha=0.7)
                plt.xlabel('Structure', fontsize=22, fontweight='bold')
                plt.ylabel('Formation Energy (eV/atom)', fontsize=22, fontweight='bold')
                plt.title('Formation Energy per Atom (Elastic Properties)', fontsize=26, fontweight='bold', pad=20)
                plt.xticks(range(len(valid_names)), [name.replace('.vasp', '').replace('POSCAR_', '') for name in valid_names], 
                          rotation=45, ha='right', fontsize=18, fontweight='bold')
                plt.yticks(fontsize=18, fontweight='bold')

                # Add value labels on bars
                for i, (bar, fe, result) in enumerate(zip(bars, valid_fe, valid_results)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(valid_fe)-min(valid_fe))*0.02, 
                            f'{fe:.4f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
                    # Add mechanical stability indicator
                    stable = result.get("mechanically_stable", False)
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - abs(max(valid_fe)-min(valid_fe))*0.08, 
                            '✓' if stable else '✗', ha='center', va='center', 
                            fontsize=20, color='green' if stable else 'red', fontweight='bold')

                plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
                plt.legend(['Stability line'], fontsize=18)
                plt.tight_layout()
                plt.savefig('results/elastic_formation_energy_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("  ✅ Saved formation energy plot: results/elastic_formation_energy_comparison.png")'''

    code += '''

            # 3. Relative Energy Plot
            if len(energies) > 1:
                min_energy = min(energies)
                relative_energies = [(e - min_energy) * 1000 for e in energies]  # Convert to meV

                plt.figure(figsize=(16, 10))
                colors = ['green' if re == 0 else 'orange' for re in relative_energies]
                bars = plt.bar(range(len(structure_names)), relative_energies, color=colors, alpha=0.7)
                plt.xlabel('Structure', fontsize=22, fontweight='bold')
                plt.ylabel('Relative Energy (meV)', fontsize=22, fontweight='bold')
                plt.title('Relative Energy Comparison (Elastic Properties, vs. Lowest Energy)', fontsize=26, fontweight='bold', pad=20)
                plt.xticks(range(len(structure_names)), [name.replace('.vasp', '').replace('POSCAR_', '') for name in structure_names], 
                          rotation=45, ha='right', fontsize=18, fontweight='bold')
                plt.yticks(fontsize=18, fontweight='bold')

                # Add value labels on bars
                for i, (bar, re, result) in enumerate(zip(bars, relative_energies, successful_results)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(relative_energies)*0.02, 
                            f'{re:.1f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
                    # Add mechanical stability indicator
                    stable = result.get("mechanically_stable", False)
                    plt.text(bar.get_x() + bar.get_width()/2, max(relative_energies)*0.1, 
                            '✓' if stable else '✗', ha='center', va='center', 
                            fontsize=20, color='green' if stable else 'red', fontweight='bold')

                plt.tight_layout()
                plt.savefig('results/elastic_relative_energy_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("  ✅ Saved relative energy plot: results/elastic_relative_energy_comparison.png")

            # 4. Bulk Modulus Comparison Plot
            bulk_moduli = []
            for result in successful_results:
                # Use Hill average if available, otherwise Voigt
                bulk_hill = result.get("bulk_modulus_hill_GPa")
                bulk_voigt = result.get("bulk_modulus_voigt_GPa")
                bulk_value = bulk_hill if bulk_hill is not None else bulk_voigt
                bulk_moduli.append(bulk_value)

            plt.figure(figsize=(16, 10))
            # Color based on mechanical stability
            colors = ['green' if result.get("mechanically_stable", False) else 'red' for result in successful_results]
            bars = plt.bar(range(len(structure_names)), bulk_moduli, color=colors, alpha=0.7)
            plt.xlabel('Structure', fontsize=22, fontweight='bold')
            plt.ylabel('Bulk Modulus (GPa)', fontsize=22, fontweight='bold')
            plt.title('Bulk Modulus Comparison', fontsize=26, fontweight='bold', pad=20)
            plt.xticks(range(len(structure_names)), [name.replace('.vasp', '').replace('POSCAR_', '') for name in structure_names], 
                      rotation=45, ha='right', fontsize=18, fontweight='bold')
            plt.yticks(fontsize=18, fontweight='bold')

            # Add value labels on bars
            for i, (bar, bulk, result) in enumerate(zip(bars, bulk_moduli, successful_results)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bulk_moduli)*0.01, 
                        f'{bulk:.1f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

                # Add shear and Young's modulus as smaller text
                shear_hill = result.get("shear_modulus_hill_GPa")
                shear_voigt = result.get("shear_modulus_voigt_GPa")
                shear_value = shear_hill if shear_hill is not None else shear_voigt
                youngs_value = result.get("youngs_modulus_GPa")

                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*0.5, 
                        f'G: {shear_value:.0f}\\nE: {youngs_value:.0f}', 
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

            # Add legend for colors
            import matplotlib.patches as mpatches
            stable_patch = mpatches.Patch(color='green', alpha=0.7, label='Mechanically Stable')
            unstable_patch = mpatches.Patch(color='red', alpha=0.7, label='Mechanically Unstable')
            plt.legend(handles=[stable_patch, unstable_patch], loc='upper right', fontsize=18)

            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('results/bulk_modulus_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✅ Saved bulk modulus plot: results/bulk_modulus_comparison.png")

            # Reset matplotlib settings
            plt.rcParams.update(plt.rcParamsDefault)

        except ImportError:
            print("  ⚠️ Matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"  ⚠️ Error generating plots: {e}")

    else:
        print("  ℹ️ No successful calculations to plot")
'''

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



def save_elastic_constants_to_csv(structure_name, elastic_tensor, csv_filename="results/elastic_constants_cij.csv"):


    elastic_data = {
        'structure_name': structure_name,
        'C11_GPa': float(elastic_tensor[0, 0]),
        'C12_GPa': float(elastic_tensor[0, 1]),
        'C13_GPa': float(elastic_tensor[0, 2]),
        'C14_GPa': float(elastic_tensor[0, 3]),
        'C15_GPa': float(elastic_tensor[0, 4]),
        'C16_GPa': float(elastic_tensor[0, 5]),
        'C21_GPa': float(elastic_tensor[1, 0]),
        'C22_GPa': float(elastic_tensor[1, 1]),
        'C23_GPa': float(elastic_tensor[1, 2]),
        'C24_GPa': float(elastic_tensor[1, 3]),
        'C25_GPa': float(elastic_tensor[1, 4]),
        'C26_GPa': float(elastic_tensor[1, 5]),
        'C31_GPa': float(elastic_tensor[2, 0]),
        'C32_GPa': float(elastic_tensor[2, 1]),
        'C33_GPa': float(elastic_tensor[2, 2]),
        'C34_GPa': float(elastic_tensor[2, 3]),
        'C35_GPa': float(elastic_tensor[2, 4]),
        'C36_GPa': float(elastic_tensor[2, 5]),
        'C41_GPa': float(elastic_tensor[3, 0]),
        'C42_GPa': float(elastic_tensor[3, 1]),
        'C43_GPa': float(elastic_tensor[3, 2]),
        'C44_GPa': float(elastic_tensor[3, 3]),
        'C45_GPa': float(elastic_tensor[3, 4]),
        'C46_GPa': float(elastic_tensor[3, 5]),
        'C51_GPa': float(elastic_tensor[4, 0]),
        'C52_GPa': float(elastic_tensor[4, 1]),
        'C53_GPa': float(elastic_tensor[4, 2]),
        'C54_GPa': float(elastic_tensor[4, 3]),
        'C55_GPa': float(elastic_tensor[4, 4]),
        'C56_GPa': float(elastic_tensor[4, 5]),
        'C61_GPa': float(elastic_tensor[5, 0]),
        'C62_GPa': float(elastic_tensor[5, 1]),
        'C63_GPa': float(elastic_tensor[5, 2]),
        'C64_GPa': float(elastic_tensor[5, 3]),
        'C65_GPa': float(elastic_tensor[5, 4]),
        'C66_GPa': float(elastic_tensor[5, 5])
    }

    if os.path.exists(csv_filename):
        df_existing = pd.read_csv(csv_filename)

        if structure_name in df_existing['structure_name'].values:
            df_existing.loc[df_existing['structure_name'] == structure_name, list(elastic_data.keys())] = list(elastic_data.values())
        else:
            df_new_row = pd.DataFrame([elastic_data])
            df_existing = pd.concat([df_existing, df_new_row], ignore_index=True)

        df_existing.to_csv(csv_filename, index=False)
    else:
        df_new = pd.DataFrame([elastic_data])
        df_new.to_csv(csv_filename, index=False)

    print(f"  💾 Elastic constants saved to {csv_filename}")


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



def generate_concentration_combinations(substitutions):
    """Generate all possible combinations of concentrations."""
    import itertools

    # Check if any element has multiple concentrations
    has_multiple = any('concentration_list' in sub_info and len(sub_info['concentration_list']) > 1
                       for sub_info in substitutions.values())

    if not has_multiple:
        # Convert single concentrations to the original format
        single_combo = {}
        for element, sub_info in substitutions.items():
            if 'concentration_list' in sub_info:
                concentration = sub_info['concentration_list'][0]
            else:
                concentration = sub_info.get('concentration', 0.5)

            element_count = sub_info.get('element_count', 0)
            n_substitute = int(element_count * concentration)

            single_combo[element] = {
                'new_element': sub_info['new_element'],
                'concentration': concentration,
                'n_substitute': n_substitute,
                'n_remaining': element_count - n_substitute
            }
        return [single_combo]

    # Generate all combinations for multiple concentrations
    elements = []
    concentration_lists = []

    for element, sub_info in substitutions.items():
        elements.append(element)
        if 'concentration_list' in sub_info:
            concentration_lists.append(sub_info['concentration_list'])
        else:
            concentration_lists.append([sub_info.get('concentration', 0.5)])

    combinations = []
    for conc_combo in itertools.product(*concentration_lists):
        combo_substitutions = {}
        for i, element in enumerate(elements):
            concentration = conc_combo[i]
            element_count = substitutions[element].get('element_count', 0)
            n_substitute = int(element_count * concentration)

            combo_substitutions[element] = {
                'new_element': substitutions[element]['new_element'],
                'concentration': concentration,
                'n_substitute': n_substitute,
                'n_remaining': element_count - n_substitute
            }

        combinations.append(combo_substitutions)

    return combinations

def create_combination_name(combo_substitutions):
    """Create a descriptive name for a concentration combination."""
    name_parts = []

    for original_element, sub_info in combo_substitutions.items():
        new_element = sub_info['new_element']
        concentration = sub_info['concentration']
        remaining_concentration = 1 - concentration

        if concentration == 0:
            # No substitution, pure original element
            name_parts.append(f"{original_element}100pct")
        elif concentration == 1:
            # Complete substitution
            if new_element == 'VACANCY':
                name_parts.append(f"{original_element}0pct_100pct_vacant")
            else:
                name_parts.append(f"{new_element}100pct")
        else:
            # Partial substitution
            remaining_pct = int(remaining_concentration * 100)
            substitute_pct = int(concentration * 100)

            if new_element == 'VACANCY':
                name_parts.append(f"{original_element}{remaining_pct}pct_{substitute_pct}pct_vacant")
            else:
                name_parts.append(f"{original_element}{remaining_pct}pct_{new_element}{substitute_pct}pct")

    return "_".join(name_parts)

def sort_concentration_combinations(concentration_combinations):
    """Sort concentration combinations for consistent ordering."""
    def get_sort_key(combo_substitutions):
        sort_values = []
        for element in sorted(combo_substitutions.keys()):
            concentration = combo_substitutions[element]['concentration']
            sort_values.append(concentration)
        return tuple(sort_values)

    return sorted(concentration_combinations, key=get_sort_key)

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
    elif cell_constraint == "Tetragonal (a=b, optimize a and c)":
        from ase.constraints import FixSymmetry
        existing_constraints = atoms.constraints if hasattr(atoms, 'constraints') and atoms.constraints else []
        symmetry_constraint = FixSymmetry(atoms)
        atoms.set_constraint(existing_constraints + [symmetry_constraint])
        return ExpCellFilter(atoms, scalar_pressure=pressure_eV_A3)
    else:  # "Lattice parameters only (fix angles)"
        if hydrostatic_strain:
            return UnitCellFilter(atoms, scalar_pressure=pressure_eV_A3, hydrostatic_strain=True)
        else:
            mask = [optimize_lattice['a'], optimize_lattice['b'], optimize_lattice['c'], False, False, False]
            return UnitCellFilter(atoms, mask=mask, scalar_pressure=pressure_eV_A3)


class OptimizationLogger:
    def __init__(self, filename, max_steps, output_dir="optimized_structures", save_trajectory=True):
        self.filename = filename
        self.step_count = 0
        self.max_steps = max_steps
        self.step_times = []
        self.step_start_time = time.time()
        self.output_dir = output_dir
        self.save_trajectory = save_trajectory  
        self.trajectory = [] if save_trajectory else None

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

            # Calculate energy per atom
            energy_per_atom = energy / len(atoms)

            # Calculate energy change
            if hasattr(self, 'previous_energy') and self.previous_energy is not None:
                energy_change = abs(energy - self.previous_energy)
                energy_change_per_atom = energy_change / len(atoms)
            else:
                energy_change = float('inf')
                energy_change_per_atom = float('inf')
            self.previous_energy = energy

            try:
                stress = atoms.get_stress(voigt=True)
                max_stress = np.max(np.abs(stress))
            except:
                max_stress = 0.0

            lattice = get_lattice_parameters(atoms)

            if self.save_trajectory:
                self.trajectory.append({
                    'step': self.step_count,
                    'energy': energy,
                    'max_force': max_force,
                    'positions': atoms.positions.copy(),
                    'cell': atoms.cell.array.copy(),
                    'lattice': lattice.copy(),
                    'forces': forces.copy()
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

                print(f"    Step {self.step_count}: E={energy:.6f} eV ({energy_per_atom:.6f} eV/atom), "
                      f"F_max={max_force:.4f} eV/Å, Max_Stress={max_stress:.4f} GPa, "
                      f"ΔE={energy_change:.2e} eV ({energy_change_per_atom:.2e} eV/atom) | "
                      f"Max. time: {remaining_time_str} ({remaining_steps} steps)")
            else:
                print(f"    Step {self.step_count}: E={energy:.6f} eV ({energy_per_atom:.6f} eV/atom), "
                      f"F_max={max_force:.4f} eV/Å, Max_Stress={max_stress:.4f} GPa, "
                      f"ΔE={energy_change:.2e} eV ({energy_change_per_atom:.2e} eV/atom)")
'''


def _generate_optimization_code(optimization_params, calc_formation_energy):
    """Generate code for geometry optimization with selective dynamics support."""
    optimizer = optimization_params.get('optimizer', 'BFGS')
    fmax = optimization_params.get('fmax', 0.05)
    max_steps = optimization_params.get('max_steps', 200)
    save_trajectory = optimization_params.get('save_trajectory', True)
    opt_type = optimization_params.get(
        'optimization_type', 'Both atoms and cell')
    cell_constraint = optimization_params.get(
        'cell_constraint', 'Lattice parameters only (fix angles)')
    pressure = optimization_params.get('pressure', 0.0)
    hydrostatic_strain = optimization_params.get('hydrostatic_strain', False)
    optimize_lattice = optimization_params.get(
        'optimize_lattice', {'a': True, 'b': True, 'c': True})

    # Check if tetragonal mode is enabled
    is_tetragonal = (cell_constraint == "Tetragonal (a=b, optimize a and c)")

    code = f'''    structure_files = sorted([f for f in os.listdir(".") if f.startswith("POSCAR") or f.endswith(".vasp") or f.endswith(".poscar")])
    results = []
    print(f"🔧 Found {{len(structure_files)}} structure files for optimization")

    optimizer_type = "{optimizer}"
    fmax = {fmax}
    max_steps = {max_steps}
    save_trajectory = {save_trajectory}
    optimization_type = "{opt_type}"
    cell_constraint = "{cell_constraint}"
    pressure = {pressure}
    hydrostatic_strain = {hydrostatic_strain}
    optimize_lattice = {optimize_lattice}
    is_tetragonal = {is_tetragonal}

    print(f"⚙️ Optimization settings:")
    print(f"  - Optimizer: {{optimizer_type}}")
    print(f"  - Force threshold: {{fmax}} eV/Å")
    print(f"  - Max steps: {{max_steps}}")
    print(f"  - Type: {{optimization_type}}")
    if pressure > 0:
        print(f"  - Pressure: {{pressure}} GPa")
    if is_tetragonal:
        print(f"  - 🔷 Tetragonal constraint: a=b will be enforced")

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

            # Setup optimization object based on type
            if optimization_type == "Atoms only (fixed cell)":
                optimization_object = atoms
                opt_mode = "atoms_only"
                print(f"  🔒 Optimizing atoms only (fixed cell)")
                tetragonal_callback = None
            elif optimization_type == "Cell only (fixed atoms)":
                existing_constraints = atoms.constraints if hasattr(atoms, 'constraints') and atoms.constraints else []
                all_fixed_constraint = FixAtoms(mask=[True] * len(atoms))
                atoms.set_constraint([all_fixed_constraint] + existing_constraints)

                if is_tetragonal:
                    # For tetragonal cell-only optimization
                    mask = [True, True, True, False, False, False]  # a, b, c can change; angles fixed
                    pressure_eV_A3 = pressure * 0.00624150913
                    optimization_object = UnitCellFilter(atoms, mask=mask, scalar_pressure=pressure_eV_A3)

                    # Define tetragonal enforcement callback
                    def enforce_tetragonal():
                        cell = atoms.get_cell()
                        cellpar = cell.cellpar()
                        avg_ab = (cellpar[0] + cellpar[1]) / 2.0
                        old_a = cellpar[0]
                        old_b = cellpar[1]
                        cellpar[0] = avg_ab
                        cellpar[1] = avg_ab
                        # cellpar[3:] = 90.0  # Ensure angles stay at 90

                        atoms.set_cell(cellpar, scale_atoms=True)
                        return old_a, old_b, avg_ab

                    tetragonal_callback = enforce_tetragonal
                    print(f"  🔒 Optimizing cell only (fixed atoms)")
                    print(f"  📐 Using tetragonal constraint (a=b)")
                else:
                    optimization_object = create_cell_filter(atoms, pressure, cell_constraint, optimize_lattice, hydrostatic_strain)
                    tetragonal_callback = None
                    print(f"  🔒 Optimizing cell only (fixed atoms)")

                opt_mode = "cell_only"
            else:  # Both atoms and cell
                if is_tetragonal:
                    # For tetragonal both optimization
                    mask = [True, True, True, False, False, False]  # a, b, c can change; angles fixed
                    pressure_eV_A3 = pressure * 0.00624150913
                    optimization_object = UnitCellFilter(atoms, mask=mask, scalar_pressure=pressure_eV_A3)

                    # Define tetragonal enforcement callback
                    def enforce_tetragonal():
                        cell = atoms.get_cell()
                        cellpar = cell.cellpar()
                        avg_ab = (cellpar[0] + cellpar[1]) / 2.0
                        old_a = cellpar[0]
                        old_b = cellpar[1]
                        cellpar[0] = avg_ab
                        cellpar[1] = avg_ab
                        # cellpar[3:] = 90.0  # Ensure angles stay at 90
                        atoms.set_cell(cellpar, scale_atoms=True)
                        return old_a, old_b, avg_ab

                    tetragonal_callback = enforce_tetragonal
                    print(f"  🔄 Optimizing both atoms and cell")
                    print(f"  📐 Using tetragonal constraint (a=b)")
                else:
                    optimization_object = create_cell_filter(atoms, pressure, cell_constraint, optimize_lattice, hydrostatic_strain)
                    tetragonal_callback = None
                    print(f"  🔄 Optimizing both atoms and cell")

                opt_mode = "both"

            logger = OptimizationLogger(filename, max_steps, "optimized_structures", save_trajectory)

            if optimizer_type == "LBFGS":
                optimizer = LBFGS(optimization_object, logfile=f"results/{filename}_opt.log")
            elif optimizer_type == "FIRE":
                optimizer = FIRE(optimization_object, logfile=f"results/{filename}_opt.log")
            elif optimizer_type == "BFGSLineSearch (QuasiNewton)":
                optimizer = BFGSLineSearch(optimization_object, logfile=f"results/{filename}_opt.log")
            elif optimizer_type == "LBFGSLineSearch":
                optimizer = LBFGSLineSearch(optimization_object, logfile=f"results/{filename}_opt.log")
            elif optimizer_type == "GoodOldQuasiNewton":
                optimizer = GoodOldQuasiNewton(optimization_object, logfile=f"results/{filename}_opt.log")
            elif optimizer_type == "MDMin":
                optimizer = MDMin(optimization_object, logfile=f"results/{filename}_opt.log")
            elif optimizer_type == "GPMin":
                optimizer = GPMin(optimization_object, logfile=f"results/{filename}_opt.log", update_hyperparams=True)
            elif optimizer_type == "SciPyFminBFGS":
                optimizer = SciPyFminBFGS(optimization_object, logfile=f"results/{filename}_opt.log")
            elif optimizer_type == "SciPyFminCG":
                optimizer = SciPyFminCG(optimization_object, logfile=f"results/{filename}_opt.log")
            else:
                optimizer = BFGS(optimization_object, logfile=f"results/{filename}_opt.log")

            optimizer.attach(lambda: logger(optimizer), interval=1)

            print(f"  🏃 Running {optimizer_type} optimization...")

            # Tetragonal mode uses manual convergence loop
            if is_tetragonal and tetragonal_callback is not None:
                print(f"  🔷 Using manual convergence loop for tetragonal constraint")

                fmax_criterion = fmax if opt_mode != "cell_only" else 0.1
                stress_threshold = 0.1
                ediff = 1e-4

                for step in range(max_steps):
                    # Single optimization step
                    optimizer.run(fmax=fmax_criterion, steps=1)

                    # Apply tetragonal constraint
                    old_a, old_b, new_ab = tetragonal_callback()
                    if abs(old_a - old_b) > 1e-6:
                        print(f"    🔷 Enforced a=b: {old_a:.6f}, {old_b:.6f} → {new_ab:.6f} Å")

                    # Get current atoms
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
                        energy_change = abs(logger.trajectory[-1]['energy'] - logger.trajectory[-2]['energy'])
                        energy_converged = energy_change < ediff

                    # Determine convergence
                    if opt_mode == "atoms_only":
                        force_converged = max_force < fmax
                        stress_converged = True
                        converged = force_converged and energy_converged
                    elif opt_mode == "cell_only":
                        force_converged = True
                        stress_converged = max_stress < stress_threshold
                        converged = stress_converged and energy_converged
                    else:  # Both
                        force_converged = max_force < fmax
                        stress_converged = max_stress < stress_threshold
                        converged = force_converged and stress_converged and energy_converged

                    if converged:
                        print(f"  ✅ Tetragonal optimization converged at step {step + 1}!")
                        if opt_mode != "atoms_only":
                            print(f"     Stress: {max_stress:.4f} < {stress_threshold} GPa ✓")
                        if opt_mode != "cell_only":
                            print(f"     Force: {max_force:.4f} < {fmax} eV/Å ✓")
                        print(f"     Energy change: {energy_change:.2e} < {ediff} eV ✓")
                        break

                    if step >= max_steps - 1:
                        print(f"  ⚠️ Reached maximum steps ({max_steps})")
                        break
            else:
                # Standard optimization (non-tetragonal)
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

            output_filename = f"optimized_structures/optimized-{base_name}.vasp"

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



            result = {
                "structure": filename,
                "optimized_structure_filename": f"optimized-{base_name}.vasp",
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
                # Convert dict to individual fields for CSV compatibility
                "optimize_lattice_a": optimize_lattice.get('a', True) if isinstance(optimize_lattice, dict) else True,
                "optimize_lattice_b": optimize_lattice.get('b', True) if isinstance(optimize_lattice, dict) else True,
                "optimize_lattice_c": optimize_lattice.get('c', True) if isinstance(optimize_lattice, dict) else True
            }
            if save_trajectory and logger.trajectory is not None:
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
                        forces = step_data.get('forces', np.zeros_like(step_data['positions']))
                        cell_matrix = step_data['cell'] 
            
                        lattice_string = " ".join([f"{x:.6f}" for row in cell_matrix for x in row]) 
            
                        traj_file.write(f"{num_atoms}\\n") 
            
                        comment = (f'Step={step} Energy={energy:.6f} Max_Force={max_force:.6f} '
                                  f'a={lattice["a"]:.6f} b={lattice["b"]:.6f} c={lattice["c"]:.6f} '
                                  f'alpha={lattice["alpha"]:.3f} beta={lattice["beta"]:.3f} gamma={lattice["gamma"]:.3f} '
                                  f'Volume={lattice["volume"]:.6f} '
                                  f'Lattice="{lattice_string}" '
                                  f'Properties=species:S:1:pos:R:3:forces:R:3:total_force:R:1')
                        traj_file.write(f"{comment}\\n") 
            
                        for j, (pos, force) in enumerate(zip(step_data['positions'], forces)):
                            symbol = symbols[j] if j < len(symbols) else 'X'
                            total_force = np.linalg.norm(force)
                            traj_file.write(f"{symbol} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f} "
                                          f"{force[0]:12.6f} {force[1]:12.6f} {force[2]:12.6f} "
                                          f"{total_force:12.6f}\\n")

                result["trajectory_file"] = trajectory_filename
                print(f"  💾 Trajectory saved: {trajectory_filename}")
            else:
                if not save_trajectory:
                    print(f"  ⏭️ Trajectory saving disabled by user")
                else:
                    print(f"  ⏭️ No trajectory data available")
                result["trajectory_file"] = None
            '''

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
            print(f"  💾 Results updated and saved")

        except Exception as e:
            print(f"  ❌ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({"structure": filename, "error": str(e)})

            df_results = pd.DataFrame(results)
            df_results.to_csv("results/optimization_results.csv", index=False)
            print(f"  💾 Results updated and saved (with error)")

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

    code += '''
    # Calculate and save statistics
    successful_results = [r for r in results if "error" not in r]
    if len(successful_results) > 0:
        final_energies = [r["final_energy_eV"] for r in successful_results]
        optimization_steps = [r["optimization_steps"] for r in successful_results]
        stats = {
            "total_energy_mean_eV": [np.mean(final_energies)],
            "total_energy_std_eV": [np.std(final_energies, ddof=1) if len(final_energies) > 1 else 0.0],
            "optimization_steps_mean": [np.mean(optimization_steps)],
            "optimization_steps_std": [np.std(optimization_steps, ddof=1) if len(optimization_steps) > 1 else 0.0]
        }
    '''

    if calc_formation_energy:
        code += '''
    formation_energies = [r["formation_energy_eV_per_atom"] for r in successful_results if "formation_energy_eV_per_atom" in r and r["formation_energy_eV_per_atom"] is not None]
    if formation_energies:
        stats["formation_energy_mean_eV_per_atom"] = [np.mean(formation_energies)]
        stats["formation_energy_std_eV_per_atom"] = [np.std(formation_energies, ddof=1) if len(formation_energies) > 1 else 0.0]
    '''

    code += '''
    pd.DataFrame(stats).to_csv("results/optimization_statistics.csv", index=False)
    print(f"💾 Saved statistics to results/optimization_statistics.csv")
    '''

    code += '''
    # Generate optimization plots
    print("\\n📊 Generating optimization plots...")
    successful_results = [r for r in results if "error" not in r]

    if len(successful_results) > 0:
        try:
            import matplotlib.pyplot as plt

            # Set global font sizes
            plt.rcParams.update({
                'font.size': 18,
                'axes.titlesize': 24,
                'axes.labelsize': 20,
                'xtick.labelsize': 18,
                'ytick.labelsize': 18,
                'legend.fontsize': 18,
                'figure.titlesize': 26
            })

            # Prepare data
            structure_names = [r["structure"] for r in successful_results]
            final_energies = [r["final_energy_eV"] for r in successful_results]

            # 1. Total Energy Plot
            plt.figure(figsize=(16, 12))
            bars = plt.bar(range(len(structure_names)), final_energies, color='steelblue', alpha=0.7)
            plt.xlabel('Structure', fontsize=22, fontweight='bold')
            plt.ylabel('Final Energy (eV)', fontsize=22, fontweight='bold')
            plt.title('Final Energy After Optimization', fontsize=26, fontweight='bold', pad=20)
            plt.xticks(range(len(structure_names)), [name.replace('.vasp', '').replace('POSCAR_', '') for name in structure_names], 
                      rotation=45, ha='right', fontsize=18, fontweight='bold')
            plt.yticks(fontsize=18, fontweight='bold')

            # Extend y-axis to accommodate labels above bars
            y_min, y_max = plt.ylim()
            y_range = y_max - y_min
            plt.ylim(y_min, y_max + y_range * 0.15)

            # Add vertical value labels above bars
            for i, (bar, energy) in enumerate(zip(bars, final_energies)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_range * 0.02, 
                        f'{energy:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold', 
                        rotation=90, color='black')

            plt.tight_layout()
            plt.savefig('results/optimization_final_energy_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  ✅ Saved final energy plot: results/optimization_final_energy_comparison.png")'''

    if calc_formation_energy:
        code += '''

            # 2. Formation Energy Plot
            formation_energies = [r.get("formation_energy_eV_per_atom") for r in successful_results]
            valid_formation = [(name, fe, result) for name, fe, result in zip(structure_names, formation_energies, successful_results) if fe is not None]

            if valid_formation:
                valid_names, valid_fe, valid_results = zip(*valid_formation)

                plt.figure(figsize=(16, 12))
                colors = ['green' if fe == min(valid_fe) else 'orange' for fe in valid_fe]
                bars = plt.bar(range(len(valid_names)), valid_fe, color=colors, alpha=0.7)
                plt.xlabel('Structure', fontsize=22, fontweight='bold')
                plt.ylabel('Formation Energy (eV/atom)', fontsize=22, fontweight='bold')
                plt.title('Formation Energy per Atom After Optimization', fontsize=26, fontweight='bold', pad=20)
                plt.xticks(range(len(valid_names)), [name.replace('.vasp', '').replace('POSCAR_', '') for name in valid_names], 
                          rotation=45, ha='right', fontsize=18, fontweight='bold')
                plt.yticks(fontsize=18, fontweight='bold')

                # Extend y-axis to accommodate labels (handle positive and negative values)
                y_min, y_max = plt.ylim()
                y_range = y_max - y_min

                # Check if we have negative values
                has_negative = any(fe < 0 for fe in valid_fe)
                has_positive = any(fe > 0 for fe in valid_fe)

                if has_negative and has_positive:
                    plt.ylim(y_min - y_range * 0.15, y_max + y_range * 0.15)
                elif has_negative and not has_positive:
                    plt.ylim(y_min - y_range * 0.15, y_max + y_range * 0.05)
                else:
                    plt.ylim(y_min - y_range * 0.05, y_max + y_range * 0.15)

                # Add vertical value labels outside bars
                for i, (bar, fe) in enumerate(zip(bars, valid_fe)):
                    if fe >= 0:
                        y_pos = bar.get_height() + y_range * 0.02
                        va_align = 'bottom'
                    else:
                        y_pos = bar.get_height() - y_range * 0.02
                        va_align = 'top'
                    plt.text(bar.get_x() + bar.get_width()/2, y_pos, 
                            f'{fe:.4f}', ha='center', va=va_align, fontsize=16, fontweight='bold', 
                            rotation=90, color='black')

                plt.tight_layout()
                plt.savefig('results/optimization_formation_energy_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("  ✅ Saved formation energy plot: results/optimization_formation_energy_comparison.png")'''

    code += '''

            # 3. Relative Energy Plot
            if len(final_energies) > 1:
                min_energy = min(final_energies)
                relative_energies = [(e - min_energy) * 1000 for e in final_energies]  # Convert to meV

                plt.figure(figsize=(16, 12))
                colors = ['green' if re == 0 else 'orange' for re in relative_energies]
                bars = plt.bar(range(len(structure_names)), relative_energies, color=colors, alpha=0.7)
                plt.xlabel('Structure', fontsize=22, fontweight='bold')
                plt.ylabel('Relative Energy (meV)', fontsize=22, fontweight='bold')
                plt.title('Relative Energy Comparison After Optimization (vs. Lowest Energy)', fontsize=26, fontweight='bold', pad=20)
                plt.xticks(range(len(structure_names)), [name.replace('.vasp', '').replace('POSCAR_', '') for name in structure_names], 
                          rotation=45, ha='right', fontsize=18, fontweight='bold')
                plt.yticks(fontsize=18, fontweight='bold')

                # Extend y-axis to accommodate labels above bars
                y_min, y_max = plt.ylim()
                y_range = max(relative_energies) if max(relative_energies) > 0 else 1
                plt.ylim(-y_range * 0.1, max(relative_energies) + y_range * 0.15)

                # Add vertical value labels above bars
                for i, (bar, re) in enumerate(zip(bars, relative_energies)):
                    if re > 0:
                        y_pos = bar.get_height() + y_range * 0.02
                        va_align = 'bottom'
                    else:
                        y_pos = y_range * 0.05  # Position above zero line for zero values
                        va_align = 'bottom'
                    plt.text(bar.get_x() + bar.get_width()/2, y_pos, 
                            f'{re:.1f}', ha='center', va=va_align, fontsize=16, fontweight='bold', 
                            rotation=90, color='black')

                plt.tight_layout()
                plt.savefig('results/optimization_relative_energy_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("  ✅ Saved relative energy plot: results/optimization_relative_energy_comparison.png")

            # 4. Lattice Parameter Changes Plot
            print("  📏 Reading detailed optimization summary for lattice changes...")
            try:
                # Read the detailed summary file that contains lattice information
                detailed_summary_file = "results/optimization_detailed_summary.csv"
                if os.path.exists(detailed_summary_file):
                    df_lattice = pd.read_csv(detailed_summary_file)

                    if len(df_lattice) > 0:
                        plt.figure(figsize=(18, 10))

                        # Extract lattice changes
                        structures = df_lattice['Structure'].tolist()
                        a_changes = df_lattice['a_change_percent'].tolist()
                        b_changes = df_lattice['b_change_percent'].tolist()
                        c_changes = df_lattice['c_change_percent'].tolist()

                        # Create grouped bar chart
                        x = np.arange(len(structures))
                        width = 0.25

                        bars1 = plt.bar(x - width, a_changes, width, label='a parameter', color='red', alpha=0.7)
                        bars2 = plt.bar(x, b_changes, width, label='b parameter', color='green', alpha=0.7)
                        bars3 = plt.bar(x + width, c_changes, width, label='c parameter', color='blue', alpha=0.7)

                        plt.xlabel('Structure', fontsize=22, fontweight='bold')
                        plt.ylabel('Lattice Parameter Change (%)', fontsize=22, fontweight='bold')
                        plt.title('Lattice Parameter Changes After Optimization', fontsize=26, fontweight='bold', pad=20)
                        plt.xticks(x, [name.replace('.vasp', '').replace('POSCAR_', '') for name in structures], 
                                  rotation=45, ha='right', fontsize=18, fontweight='bold')
                        plt.yticks(fontsize=18, fontweight='bold')

                        # Add horizontal line at zero
                        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
                        plt.grid(True, alpha=0.3, axis='y')

                        # Add legend below x-axis
                        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                                  ncol=3, fontsize=18, frameon=False)

                        # Adjust layout to accommodate legend
                        plt.subplots_adjust(bottom=0.2)
                        plt.tight_layout()
                        plt.savefig('results/lattice_parameter_changes.png', dpi=300, bbox_inches='tight')
                        plt.close()
                        print("  ✅ Saved lattice changes plot: results/lattice_parameter_changes.png")
                    else:
                        print("  ⚠️ No lattice data found in detailed summary")
                else:
                    print("  ⚠️ Detailed optimization summary file not found")

            except Exception as lattice_error:
                print(f"  ⚠️ Error creating lattice changes plot: {lattice_error}")

            # Reset matplotlib settings
            plt.rcParams.update(plt.rcParamsDefault)

        except ImportError:
            print("  ⚠️ Matplotlib not available. Install with: pip install matplotlib")
        except Exception as e:
            print(f"  ⚠️ Error generating plots: {e}")

    else:
        print("  ℹ️ No successful calculations to plot")
'''

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

    code = f'''    structure_files = sorted([f for f in os.listdir(".") if f.startswith("POSCAR") or f.endswith(".vasp") or f.endswith(".poscar")])
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
            print(f"  💾 Results updated and saved")

        except Exception as e:
            print(f"  ❌ Phonon calculation failed: {e}")
            results.append({"structure": filename, "error": str(e)}")

            df_results = pd.DataFrame(results)
            df_results.to_csv("results/phonon_results.csv", index=False)
            print(f"  💾 Results updated and saved")

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
