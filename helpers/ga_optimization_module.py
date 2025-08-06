import numpy as np
import random
from copy import deepcopy
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import threading
import time



class GeneticAlgorithmOptimizer:
    def __init__(self, base_structure, calculator, substitutions, ga_params, log_queue, stop_event, run_id=0):
        self.base_structure = base_structure
        self.calculator = calculator
        self.substitutions = substitutions
        self.ga_params = ga_params
        self.log_queue = log_queue
        self.stop_event = stop_event
        self.run_id = run_id

        self.population = []
        self.fitness_history = []
        self.detailed_history = []
        self.best_individual = None
        self.best_energy = float('inf')
        self.final_population = []
        self.final_fitness = []

        # ADD THESE NEW LINES - Handle uploaded initial structures
        self.use_uploaded_structures = ga_params.get('use_uploaded_structures', False)
        self.uploaded_initial_structures = ga_params.get('uploaded_initial_structures', [])
        self.uploaded_structure_names = ga_params.get('uploaded_structure_names', [])

        random.seed(run_id * 12345)
        np.random.seed(run_id * 12345)

        self.create_site_id_mapping()
        self.setup_substitution_sites()
        self.setup_fixed_substitution_counts()

    def validate_uploaded_structure(self, structure):
        """
        Improved validation that focuses on composition rather than exact site mapping
        """
        try:
            # Check basic structure properties
            if len(structure) == 0:
                return False, "Empty structure"

            # Check if structure has reasonable size compared to base structure
            size_difference = abs(len(structure) - len(self.base_structure))
            max_allowed_difference = max(5, len(self.base_structure) * 0.4)  # Allow 40% difference

            if size_difference > max_allowed_difference:
                return False, f"Structure size mismatch: {len(structure)} vs expected ~{len(self.base_structure)} atoms (difference: {size_difference})"

            # Get composition of uploaded structure
            uploaded_composition = {}
            for site in structure:
                element = site.specie.symbol
                uploaded_composition[element] = uploaded_composition.get(element, 0) + 1

            # Calculate expected composition based on base structure and substitutions
            expected_composition = {}
            for site in self.base_structure:
                element = site.specie.symbol
                expected_composition[element] = expected_composition.get(element, 0) + 1

            # Apply substitutions to get expected final composition
            for original_element, sub_info in self.fixed_substitution_counts.items():
                n_substitute = sub_info['n_substitute']
                new_element = sub_info['new_element']

                if original_element in expected_composition:
                    # Remove substituted atoms
                    expected_composition[original_element] -= n_substitute

                    # Add new atoms (unless it's a vacancy)
                    if new_element != 'VACANCY':
                        expected_composition[new_element] = expected_composition.get(new_element, 0) + n_substitute

                    # Remove element if count reaches zero
                    if expected_composition[original_element] <= 0:
                        del expected_composition[original_element]

            # Compare compositions with reasonable tolerance
            for element, expected_count in expected_composition.items():
                uploaded_count = uploaded_composition.get(element, 0)
                tolerance = max(2, int(expected_count * 0.15))  # 15% tolerance or at least ±2 atoms

                if abs(uploaded_count - expected_count) > tolerance:
                    return False, f"Element {element}: expected ~{expected_count} atoms (±{tolerance}), found {uploaded_count}"


            unexpected_elements = set(uploaded_composition.keys()) - set(expected_composition.keys())
            if unexpected_elements:
                for element in unexpected_elements:
                    count = uploaded_composition[element]
                    max_allowed_unexpected = max(1, int(len(structure) * 0.05))
                    if count > max_allowed_unexpected:
                        return False, f"Unexpected element {element}: found {count} atoms, max allowed: {max_allowed_unexpected}"

            return True, f"Structure validated successfully - composition matches expected pattern"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def process_uploaded_structure(self, structure):

        try:
            is_valid, message = self.validate_uploaded_structure(structure)
            if not is_valid:
                self.log_queue.put(f"Run {self.run_id + 1} - Uploaded structure validation failed: {message}")
                return None

            processed_structure = structure.copy()

            if self.ga_params.get('perturb_positions', True):
                processed_structure = self.apply_position_perturbations(processed_structure)

            return processed_structure

        except Exception as e:
            self.log_queue.put(f"Run {self.run_id + 1} - Error processing uploaded structure: {str(e)}")
            return None

    def create_initial_population(self, population_size):
        population = []
        fitness_scores = []

        if self.use_uploaded_structures and self.uploaded_initial_structures:
            self.log_queue.put(
                f"Run {self.run_id + 1} - Processing {len(self.uploaded_initial_structures)} uploaded structures...")

            successful_uploads = 0
            failed_uploads = 0

            for i, structure in enumerate(self.uploaded_initial_structures):
                if len(population) >= population_size:
                    self.log_queue.put(
                        f"Run {self.run_id + 1} - Population limit reached, skipping remaining uploaded structures")
                    break

                if self.stop_event.is_set():
                    self.log_queue.put(f"🛑 GA run {self.run_id + 1} stopped during uploaded structure processing")
                    return None, None

                structure_name = self.uploaded_structure_names[i] if i < len(
                    self.uploaded_structure_names) else f"Structure_{i + 1}"

                processed_structure = self.process_uploaded_structure(structure)
                if processed_structure is not None:
                    population.append(processed_structure)

                    fitness = self.calculate_fitness(processed_structure)
                    fitness_scores.append(fitness)

                    if fitness < self.best_energy:
                        self.best_energy = fitness
                        self.best_individual = processed_structure.copy()

                    successful_uploads += 1
                    self.log_queue.put(f"  ✅ Uploaded structure {i + 1} ({structure_name}): {fitness:.6f} eV")
                else:
                    failed_uploads += 1
                    self.log_queue.put(f"  ❌ Failed to process uploaded structure {i + 1} ({structure_name})")

            self.log_queue.put(
                f"Run {self.run_id + 1} - Upload summary: {successful_uploads} successful, {failed_uploads} failed out of {len(self.uploaded_initial_structures)} total")

        remaining_needed = population_size - len(population)
        if remaining_needed > 0:
            self.log_queue.put(
                f"Run {self.run_id + 1} - Generating {remaining_needed} random structures to complete population...")

            for i in range(remaining_needed):
                if self.stop_event.is_set():
                    self.log_queue.put(f"🛑 GA run {self.run_id + 1} stopped during random structure generation")
                    return None, None

                current_total = len(population) + 1
                if current_total % max(1, population_size // 10) == 0 or i == 0 or current_total == population_size:
                    self.log_queue.put({
                        'type': 'ga_progress',
                        'run_id': self.run_id,
                        'generation': 0,
                        'current_structure': current_total,
                        'total_structures': population_size,
                        'phase': 'initialization'
                    })

                individual = self.create_random_individual()
                fitness = self.calculate_fitness(individual)

                population.append(individual)
                fitness_scores.append(fitness)

                if fitness < self.best_energy:
                    self.best_energy = fitness
                    self.best_individual = individual.copy()

                if (i + 1) % 10 == 0:
                    self.log_queue.put(
                        f"Run {self.run_id + 1} - Generated {len(population)}/{population_size} total individuals")

        return population, fitness_scores


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

        self.log_queue.put(f"Run {self.run_id + 1} - Created site ID mapping for {len(self.site_ids)} sites")

    def setup_substitution_sites(self):
        self.substitutable_sites = {}

        for i, site in enumerate(self.base_structure):
            element = site.specie.symbol
            if element in self.substitutions:
                if element not in self.substitutable_sites:
                    self.substitutable_sites[element] = []
                self.substitutable_sites[element].append(i)

        self.log_queue.put(f"Run {self.run_id + 1} - Found substitutable sites: {self.substitutable_sites}")

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
                    self.log_queue.put(
                        f"Run {self.run_id + 1} - Fixed vacancy creation: {n_substitute}/{total_sites} {original_element} → VACANCY")
                else:
                    self.log_queue.put(
                        f"Run {self.run_id + 1} - Fixed substitution: {n_substitute}/{total_sites} {original_element} → {sub_info['new_element']}")

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

                    # Perform the swap
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
                    self.log_queue.put(
                        f"❌ Run {self.run_id + 1} - Validation failed: Expected {expected_count} substitutions for {original_element}, got 0")
                    return False
                continue

            actual_count = pattern[original_element]['n_substitute']

            if actual_count != expected_count:
                self.log_queue.put(
                    f"❌ Run {self.run_id + 1} - Validation failed: Expected {expected_count} substitutions for {original_element}, got {actual_count}")
                return False

        return True

    def calculate_fitness(self, structure):
        try:
            if not self.validate_individual(structure):
                return float('inf')

            structure_start_time = time.time()

            adaptor = AseAtomsAdaptor()
            atoms = adaptor.get_atoms(structure)
            atoms.calc = self.calculator

            if self.ga_params.get('perturb_positions', True):
                from ase.optimize import BFGS, LBFGS

                optimizer_name = self.ga_params.get('optimizer', 'BFGS')
                fmax = self.ga_params.get('fmax', 0.05)
                max_steps = self.ga_params.get('max_steps', 100)
                maxstep = self.ga_params.get('maxstep', 0.2)

                if optimizer_name == 'LBFGS':
                    optimizer = LBFGS(atoms, maxstep=maxstep)
                else:
                    optimizer = BFGS(atoms, maxstep=maxstep)

                optimizer.run(fmax=fmax, steps=max_steps)

            energy = atoms.get_potential_energy()

            structure_end_time = time.time()
            structure_duration = structure_end_time - structure_start_time

            if not hasattr(self, '_structure_count'):
                self._structure_count = 0

            self._structure_count += 1

            if self._structure_count % 10 == 0:
                self.log_queue.put({
                    'type': 'ga_structure_timing',
                    'run_id': self.run_id,
                    'duration': structure_duration,
                    'energy': energy
                })

            return energy
        except Exception as e:
            self.log_queue.put(f"Run {self.run_id + 1} - Error calculating energy: {str(e)}")
            return float('inf')

    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        selected = []

        for _ in range(2):  # Select 2 parents
            tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx])

        return selected

    def optimize(self):
        population_size = self.ga_params.get('population_size', 50)
        max_generations = self.ga_params.get('max_generations', 100)
        elitism_ratio = self.ga_params.get('elitism_ratio', 0.1)

        self.log_queue.put(
            f"Starting GA run {self.run_id + 1}: {population_size} individuals, {max_generations} generations")

        if self.stop_event.is_set():
            self.log_queue.put(f"🛑 GA run {self.run_id + 1} stopped before initialization")
            return None


        if self.use_uploaded_structures and self.uploaded_initial_structures:
            self.log_queue.put(
                f"Run {self.run_id + 1} - Using {len(self.uploaded_initial_structures)} uploaded structures + random generation for initial population")
        else:
            self.log_queue.put(f"Run {self.run_id + 1} - Creating fully random initial population...")

        self.log_queue.put({
            'type': 'ga_progress',
            'run_id': self.run_id,
            'generation': 0,
            'current_structure': 0,
            'total_structures': population_size,
            'phase': 'initialization'
        })


        result = self.create_initial_population(population_size)
        if result is None:
            return None

        self.population, fitness_scores = result

        self.log_queue.put(
            f"Run {self.run_id + 1} - Initial population created. Best energy: {self.best_energy:.6f} eV")


        for generation in range(max_generations):
            if self.stop_event.is_set():
                self.log_queue.put(f"🛑 GA run {self.run_id + 1} stopped at generation {generation}")
                break

            self.log_queue.put({
                'type': 'ga_progress',
                'run_id': self.run_id,
                'generation': generation,
                'current_structure': 0,
                'total_structures': population_size,
                'phase': 'evolution'
            })

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
            structure_count = len(new_population)
            update_frequency = max(1, population_size // 10)

            while len(new_population) < population_size:
                if structure_count % 5 == 0 and self.stop_event.is_set():
                    self.log_queue.put(f"🛑 GA run {self.run_id + 1} stopped during generation {generation}")
                    break

                if structure_count % update_frequency == 0 or structure_count == population_size - 1:
                    self.log_queue.put({
                        'type': 'ga_progress',
                        'run_id': self.run_id,
                        'generation': generation,
                        'current_structure': structure_count + 1,
                        'total_structures': population_size,
                        'phase': 'evolution'
                    })

                parents = self.tournament_selection(self.population, fitness_scores)

                if random.random() < self.ga_params.get('crossover_rate', 0.8):
                    child = self.crossover(parents[0], parents[1])
                else:
                    child = random.choice(parents).copy()

                child = self.mutate(child)
                child_fitness = self.calculate_fitness(child)

                new_population.append(child)
                new_fitness.append(child_fitness)
                structure_count += 1

                if child_fitness < self.best_energy:
                    self.best_energy = child_fitness
                    self.best_individual = child.copy()
                    self.log_queue.put(
                        f"  Run {self.run_id + 1} - New best energy: {self.best_energy:.6f} eV (gen {generation})")

            if self.stop_event.is_set():
                break

            self.population = new_population
            fitness_scores = new_fitness

            best_fitness = np.min(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            worst_fitness = np.max(fitness_scores)

            self.detailed_history.append({
                'generation': generation,
                'best': best_fitness,
                'average': avg_fitness,
                'worst': worst_fitness,
                'run_id': self.run_id
            })

            self.fitness_history.append({'generation': generation, 'best': best_fitness, 'average': avg_fitness})

            if generation % 10 == 0 or generation == max_generations - 1:
                self.log_queue.put(
                    f"Run {self.run_id + 1} - Generation {generation}: Best={best_fitness:.6f} eV, Avg={avg_fitness:.6f} eV, Worst={worst_fitness:.6f} eV")

            if generation > 20:
                recent_best = [f['best'] for f in self.fitness_history[-10:]]
                if max(recent_best) - min(recent_best) < self.ga_params.get('convergence_threshold', 1e-6):
                    self.log_queue.put(f"Run {self.run_id + 1} - Converged at generation {generation}")
                    break

        self.final_population = self.population.copy()
        self.final_fitness = fitness_scores.copy()

        if self.stop_event.is_set():
            self.log_queue.put(f"🛑 GA run {self.run_id + 1} stopped by user")
        else:
            self.log_queue.put(
                f"Run {self.run_id + 1} - GA optimization completed. Final best energy: {self.best_energy:.6f} eV")

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
            'used_uploaded_structures': self.use_uploaded_structures,
            'num_uploaded_structures': len(self.uploaded_initial_structures) if self.uploaded_initial_structures else 0
        }


def run_ga_optimization(base_structure, calculator, substitutions, ga_params, log_queue, stop_event):
    try:
        log_queue.put("🔄 Starting GA optimization module")

        num_runs = ga_params.get('num_runs', 1)
        all_results = []

        log_queue.put(f"🧬 Starting {num_runs} GA runs in serial execution")

        for run_id in range(num_runs):
            if stop_event.is_set():
                log_queue.put(f"🛑 GA optimization stopped at run {run_id + 1}")
                break

            log_queue.put(f"🔄 Starting GA run {run_id + 1}/{num_runs}")

            try:
                optimizer = GeneticAlgorithmOptimizer(
                    base_structure, calculator, substitutions, ga_params, log_queue, stop_event, run_id
                )

                results = optimizer.optimize()

                if results:
                    all_results.append(results)
                    log_queue.put(f"✅ GA run {run_id + 1} completed: Best energy = {results['best_energy']:.6f} eV")
                else:
                    log_queue.put(f"❌ GA run {run_id + 1} failed or was stopped")

            except Exception as run_error:
                log_queue.put(f"❌ GA run {run_id + 1} failed with error: {str(run_error)}")
                import traceback
                log_queue.put(f"Run {run_id + 1} traceback: {traceback.format_exc()}")
                continue

        if all_results:
            best_overall = min(all_results, key=lambda x: x['best_energy'])

            log_queue.put(f"🏆 Overall best energy from {len(all_results)} runs: {best_overall['best_energy']:.6f} eV")

            log_queue.put({
                'type': 'ga_result',
                'best_structure': best_overall['best_structure'],
                'best_energy': best_overall['best_energy'],
                'all_runs': all_results,
                'substitutions': substitutions,
                'ga_params': ga_params
            })
        else:
            log_queue.put("❌ No successful GA runs completed")

        log_queue.put("GA_OPTIMIZATION_FINISHED")

    except Exception as e:
        import traceback
        log_queue.put(f"❌ GA optimization failed: {str(e)}")
        log_queue.put(f"Traceback: {traceback.format_exc()}")
        log_queue.put("GA_OPTIMIZATION_FINISHED")


def validate_structure_for_substitutions(structure, substitutions, base_structure):
    try:
        uploaded_composition = {}
        for site in structure:
            element = site.specie.symbol
            uploaded_composition[element] = uploaded_composition.get(element, 0) + 1


        expected_composition = {}
        for site in base_structure:
            element = site.specie.symbol
            expected_composition[element] = expected_composition.get(element, 0) + 1

        substitution_info = {}
        for original_element, sub_info in substitutions.items():
            if original_element in expected_composition:
                total_sites = expected_composition[original_element]
                n_substitute = int(total_sites * sub_info['concentration'])
                new_element = sub_info['new_element']

                substitution_info[original_element] = {
                    'total_sites': total_sites,
                    'n_substitute': n_substitute,
                    'new_element': new_element,
                    'concentration': sub_info['concentration']
                }

                expected_composition[original_element] -= n_substitute

                if new_element != 'VACANCY':
                    expected_composition[new_element] = expected_composition.get(new_element, 0) + n_substitute

                if expected_composition[original_element] <= 0:
                    del expected_composition[original_element]

        size_difference = abs(len(structure) - len(base_structure))
        max_allowed_difference = max(5, len(base_structure) * 0.4)

        if size_difference > max_allowed_difference:
            return False, f"Size mismatch: {len(structure)} vs expected ~{len(base_structure)} atoms", None

        validation_details = []
        composition_match = True

        for element, expected_count in expected_composition.items():
            uploaded_count = uploaded_composition.get(element, 0)
            tolerance = max(2, int(expected_count * 0.15))  # 15% tolerance or at least ±2 atoms

            if abs(uploaded_count - expected_count) <= tolerance:
                validation_details.append(f"✅ {element}: {uploaded_count} atoms (expected ~{expected_count})")
            else:
                validation_details.append(
                    f"❌ {element}: {uploaded_count} atoms (expected {expected_count} ± {tolerance})")
                composition_match = False

        unexpected_elements = set(uploaded_composition.keys()) - set(expected_composition.keys())
        if unexpected_elements:
            for element in unexpected_elements:
                count = uploaded_composition[element]
                max_allowed_unexpected = max(1, int(len(structure) * 0.05))
                if count <= max_allowed_unexpected:
                    validation_details.append(f"⚠️ {element}: {count} atoms (unexpected but allowed)")
                else:
                    validation_details.append(
                        f"❌ {element}: {count} atoms (unexpected, max allowed: {max_allowed_unexpected})")
                    composition_match = False

        composition_info = {
            'uploaded_composition': uploaded_composition,
            'expected_composition': expected_composition,
            'substitution_info': substitution_info,
            'validation_details': validation_details,
            'size_info': f"{len(structure)} atoms (base: {len(base_structure)})"
        }

        if composition_match:
            return True, "Structure composition matches expected substitution pattern", composition_info
        else:
            return False, "Structure composition doesn't match expected pattern", composition_info

    except Exception as e:
        return False, f"Validation error: {str(e)}", None



def setup_ga_parameters_ui(working_structure, substitutions, load_structure_func=None):
    import streamlit as st
    st.divider()
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.subheader("Genetic Algorithm Parameters")

    col_ga1, col_ga2, col_ga3 = st.columns(3)

    with col_ga1:
        population_size = st.number_input("Population Size", min_value=3, max_value=1000, value=50, step=10)
        max_generations = st.number_input("Max Generations", min_value=1, max_value=1000, value=20, step=10)
        num_runs = st.number_input("Number of GA Runs", min_value=1, max_value=1000, value=3, step=1,
                                   help="Run GA multiple times with different random seeds")

    with col_ga2:
        crossover_rate = st.number_input("Crossover Rate", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
        mutation_rate = st.number_input("Mutation Rate", min_value=0.00, max_value=0.5, value=0.1, step=0.01)

    with col_ga3:
        elitism_ratio = st.number_input("Elitism Ratio", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        convergence_threshold = st.number_input("Convergence Threshold", min_value=1e-12, max_value=1e-1, value=1e-6,
                                                format="%.2e")

    st.divider()
    st.subheader("Initial Population Configuration")

    use_uploaded_structures = st.checkbox(
        "Use uploaded structures for initial generation",
        value=False,
        help="Upload your own structures to seed the initial population instead of using purely random generation"
    )

    uploaded_initial_structures = []
    uploaded_structure_names = []
    validated_structures = []

    valid_count = 0
    invalid_count = 0

    if use_uploaded_structures:
        if load_structure_func is None and 'load_structure_for_ga' not in globals():
            st.error("❌ Structure loading function not available. Please contact the developer.")
            return {}

        st.info(
            "📁 **Upload Structure Files**: These structures will be validated immediately against your substitution requirements.")

        uploaded_files = st.file_uploader(
            "Upload initial population structures (POSCAR, CIF, etc.)",
            accept_multiple_files=True,
            type=None,
            help="Upload structure files that will be used as initial population. Validation happens immediately after upload.",
            key="initial_population_uploader"
        )

        if uploaded_files:
            st.write(f"📋 **Processing {len(uploaded_files)} uploaded files...**")

            if 'ga_validation_cache' not in st.session_state:
                st.session_state.ga_validation_cache = {}

            valid_count = 0
            invalid_count = 0


            valid_results = []
            invalid_results = []

            for i, uploaded_file in enumerate(uploaded_files):
                file_key = f"{uploaded_file.name}_{uploaded_file.size}"

                if file_key not in st.session_state.ga_validation_cache:
                    try:
                        if load_structure_func:
                            structure = load_structure_func(uploaded_file)
                        else:
                            structure = load_structure_for_ga(uploaded_file)

                        is_valid, message, composition_info = validate_structure_for_substitutions(
                            structure, substitutions, working_structure
                        )


                        st.session_state.ga_validation_cache[file_key] = {
                            'structure': structure,
                            'name': uploaded_file.name,
                            'is_valid': is_valid,
                            'message': message,
                            'composition_info': composition_info
                        }

                    except Exception as e:
                        st.session_state.ga_validation_cache[file_key] = {
                            'structure': None,
                            'name': uploaded_file.name,
                            'is_valid': False,
                            'message': f"Loading error: {str(e)}",
                            'composition_info': None
                        }

                cached_result = st.session_state.ga_validation_cache[file_key]

                if cached_result['is_valid'] and cached_result['structure']:
                    valid_results.append(cached_result)
                    uploaded_initial_structures.append(cached_result['structure'])
                    uploaded_structure_names.append(cached_result['name'])
                    validated_structures.append(cached_result['structure'])
                    valid_count += 1
                else:
                    invalid_results.append(cached_result)
                    invalid_count += 1

            if valid_count > 0 or invalid_count > 0:
                col_summary1, col_summary2, col_summary3 = st.columns(3)
                with col_summary1:
                    st.metric("✅ Valid Structures", valid_count)
                with col_summary2:
                    st.metric("❌ Invalid Structures", invalid_count)
                with col_summary3:
                    success_rate = (valid_count / (valid_count + invalid_count)) * 100 if (
                                                                                                      valid_count + invalid_count) > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.0f}%")

            if valid_results:
                with st.expander(f"✅ Valid Structures ({len(valid_results)})", expanded=False):
                    st.success("These structures passed validation and will be used in the GA:")

                    for i, result in enumerate(valid_results):
                        st.markdown(f"**{i + 1}. {result['name']}**")

                        col_v1, col_v2, col_v3 = st.columns(3)
                        with col_v1:
                            if result['structure']:
                                st.write(f"📋 Formula: `{result['structure'].composition.reduced_formula}`")
                                st.write(f"🔢 Atoms: {len(result['structure'])}")

                        with col_v2:
                            if result['composition_info']:
                                st.write(f"📏 {result['composition_info']['size_info']}")

                        with col_v3:
                            st.write(f"💚 {result['message']}")


                        if result['composition_info'] and result['composition_info']['validation_details']:
                            valid_details = [d for d in result['composition_info']['validation_details'] if
                                             d.startswith('✅')]
                            if valid_details:
                                details_text = " | ".join([d.replace('✅ ', '') for d in valid_details])
                                st.write(f"🧪 Composition: {details_text}")

                        if i < len(valid_results) - 1:
                            st.markdown("---")


            if invalid_results:
                with st.expander(f"❌ Invalid Structures ({len(invalid_results)})", expanded=False):
                    st.error("These structures failed validation and need to be fixed:")

                    for i, result in enumerate(invalid_results):
                        st.markdown(f"**{i + 1}. {result['name']}**")

                        col_i1, col_i2 = st.columns(2)
                        with col_i1:
                            if result['structure']:
                                st.write(f"📋 Formula: `{result['structure'].composition.reduced_formula}`")
                                st.write(f"🔢 Atoms: {len(result['structure'])}")
                            else:
                                st.write("❌ Failed to load structure")

                        with col_i2:
                            st.error(f"❌ {result['message']}")


                        if result['composition_info'] and result['composition_info']['validation_details']:
                            st.write("**Issues found:**")
                            for detail in result['composition_info']['validation_details']:
                                if detail.startswith('❌'):
                                    st.write(f"  • {detail.replace('❌ ', '')}")
                                elif detail.startswith('⚠️'):
                                    st.write(f"  • {detail.replace('⚠️ ', '(Warning) ')}")

                        if i < len(invalid_results) - 1:
                            st.markdown("---")

                    st.info("💡 **Tips to fix invalid structures:**\n"
                            "- Check element composition matches your substitution settings\n"
                            "- Verify structure has reasonable number of atoms\n"
                            "- Ensure file format is supported (POSCAR, CIF, etc.)")


            if len(validated_structures) > 0:
                st.markdown("---")
                col_pop1, col_pop2 = st.columns(2)
                with col_pop1:
                    st.metric("Valid Uploaded Structures", len(validated_structures))
                with col_pop2:
                    if len(validated_structures) > population_size:
                        st.warning(
                            f"⚠️ You have {len(validated_structures)} valid structures but population size is {population_size}. Only the first {population_size} will be used.")
                    elif len(validated_structures) < population_size:
                        remaining = population_size - len(validated_structures)
                        st.info(
                            f"📈 {remaining} additional random structures will be generated to reach population size of {population_size}")

                if len(validated_structures) >= 2:
                    adjust_population = st.checkbox(
                        f"Adjust population size to match valid uploaded structures ({len(validated_structures)})",
                        value=False,
                        help="Use only the valid uploaded structures without generating additional random ones"
                    )
                    if adjust_population:
                        population_size = len(validated_structures)
                        st.success(f"✅ Population size adjusted to {population_size}")


            elif invalid_count > 0:
                st.error(
                    f"❌ All {invalid_count} uploaded structures are invalid. Please fix the issues or upload different structures.")

    st.subheader("Position Optimization")

    col_pos1, col_pos2 = st.columns(2)

    with col_pos1:
        perturb_positions = st.checkbox("Optimize Atomic Positions", value=False)

    if perturb_positions:
        with col_pos2:
            max_displacement = st.number_input("Max Random Position Displacement When Creating Initial Generation (Å)",
                                               min_value=0.00, max_value=1.0, value=0.1, step=0.01)

        col_geom1, col_geom2 = st.columns(2)

        with col_geom1:
            fmax = st.number_input("Force Convergence (eV/Å)", min_value=0.001, max_value=1.0, value=0.05, step=0.005,
                                   format="%.3f")
            max_steps = st.number_input("Max Optimization Steps", min_value=10, max_value=500, value=100, step=10)

        with col_geom2:
            optimizer = st.selectbox("Optimizer", ["BFGS", "LBFGS"], index=0)
            maxstep = st.number_input("Max Step Size During Optimization (Å)", min_value=0.01, max_value=0.5, value=0.2,
                                      step=0.01,
                                      format="%.2f")
    else:
        max_displacement = 0.1
        fmax = 0.05
        max_steps = 100
        optimizer = "BFGS"
        maxstep = 0.2

    return {
        'population_size': population_size,
        'max_generations': max_generations,
        'num_runs': num_runs,
        'crossover_rate': crossover_rate,
        'mutation_rate': mutation_rate,
        'elitism_ratio': elitism_ratio,
        'convergence_threshold': convergence_threshold,
        'perturb_positions': perturb_positions,
        'max_displacement': max_displacement,
        'fmax': fmax,
        'max_steps': max_steps,
        'optimizer': optimizer,
        'maxstep': maxstep,
        'use_uploaded_structures': use_uploaded_structures,
        'uploaded_initial_structures': validated_structures,
        'uploaded_structure_names': uploaded_structure_names,
        'valid_upload_count': valid_count,
        'invalid_upload_count': invalid_count,
    }

def setup_substitution_ui(structure):
    import streamlit as st
    st.divider()
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("Element Substitution Setup")

    unique_elements = list(set([site.specie.symbol for site in structure]))

    if not unique_elements:
        st.error("No elements found in structure")
        return {}

    st.info(f"📋 **Structure Info:** {len(structure)} atoms, Formula: {structure.composition.reduced_formula}")

    substitutions = {}

    st.write("**Configure which elements to substitute:**")

    for element in unique_elements:
        element_count = sum(1 for site in structure if site.specie.symbol == element)
        element_percentage = (element_count / len(structure)) * 100

        with st.expander(f"🔄 Substitute {element} atoms ({element_count} sites, {element_percentage:.1f}%)",
                         expanded=False):
            col_sub1, col_sub2, col_sub3, col_sub4 = st.columns(4)

            with col_sub1:
                enable_substitution = st.checkbox(f"Enable substitution", key=f"enable_{element}")

            if enable_substitution:
                with col_sub2:
                    common_elements = ['VACANCY', 'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P',
                                       'S', 'Cl',
                                       'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
                                       'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
                                       'Ag',
                                       'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os',
                                       'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']

                    available_elements = [e for e in common_elements if e != element]

                    new_element = st.selectbox(f"Replace with:", available_elements, key=f"new_element_{element}")

                with col_sub3:
                    if element_count == 1:
                        available_concentrations = [0.0, 1.0]
                        if new_element == "VACANCY":
                            concentration_labels = ["0% (0 vacancies)", "100% (1 vacancy)"]
                        else:
                            concentration_labels = ["0% (0 atoms)", "100% (1 atom)"]
                    else:
                        available_concentrations = []
                        concentration_labels = []

                        for n_atoms in range(element_count + 1):
                            concentration = n_atoms / element_count
                            available_concentrations.append(concentration)
                            if new_element == "VACANCY":
                                concentration_labels.append(f"{concentration * 100:.1f}% ({n_atoms} vacancies)")
                            else:
                                concentration_labels.append(f"{concentration * 100:.1f}% ({n_atoms} atoms)")

                    selected_concentration_idx = st.selectbox(
                        f"Substitution level:",
                        options=range(len(available_concentrations)),
                        format_func=lambda x: concentration_labels[x],
                        index=len(available_concentrations) // 2,
                        key=f"conc_{element}"
                    )

                    concentration = available_concentrations[selected_concentration_idx]

                with col_sub4:
                    n_substitute = int(element_count * concentration)
                    remaining_atoms = element_count - n_substitute

                    if new_element == "VACANCY":
                        st.metric("Will remove", f"{n_substitute}")
                        st.metric("Will remain", f"{remaining_atoms}")
                    else:
                        st.metric("Will substitute", f"{n_substitute}")
                        st.metric("Will remain", f"{remaining_atoms}")

                if new_element == "VACANCY":
                    st.write(f"**Result:** {remaining_atoms} {element} + {n_substitute} 🕳️ vacancies")
                else:
                    st.write(
                        f"**Result:** {remaining_atoms} {element} + {n_substitute} {new_element} = {element_count} total sites")

                substitutions[element] = {
                    'new_element': new_element,
                    'concentration': concentration,
                    'n_substitute': n_substitute,
                    'n_remaining': remaining_atoms
                }

    if substitutions:
        st.success(f"✅ Configured substitutions for {len(substitutions)} element(s)")

        with st.expander("📋 Detailed Substitution Summary", expanded=False):
            total_substitutions = 0
            total_vacancies = 0

            summary_data = []

            for orig_element, sub_info in substitutions.items():
                element_count = sum(1 for site in structure if site.specie.symbol == orig_element)
                n_substitute = sub_info['n_substitute']
                n_remaining = sub_info['n_remaining']

                if sub_info['new_element'] == 'VACANCY':
                    total_vacancies += n_substitute
                    target_display = "🕳️ Vacancy"
                else:
                    total_substitutions += n_substitute
                    target_display = sub_info['new_element']

                summary_data.append({
                    'Original Element': orig_element,
                    'Substitution Target': target_display,
                    'Total Sites': element_count,
                    'Remaining Original': n_remaining,
                    'Substituted/Removed': n_substitute,
                    'Concentration': f"{sub_info['concentration'] * 100:.1f}%"
                })


            import pandas as pd
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)


            st.subheader("🔮 Expected Final Composition")

            final_composition = {}
            original_total_atoms = len(structure)


            for site in structure:
                element = site.specie.symbol
                final_composition[element] = final_composition.get(element, 0) + 1


            for orig_element, sub_info in substitutions.items():
                n_substitute = sub_info['n_substitute']
                new_element = sub_info['new_element']

                if new_element == 'VACANCY':
                    final_composition[orig_element] -= n_substitute
                else:
                    final_composition[orig_element] -= n_substitute
                    final_composition[new_element] = final_composition.get(new_element, 0) + n_substitute

                if final_composition[orig_element] == 0:
                    del final_composition[orig_element]

            final_total_atoms = sum(final_composition.values())

            if final_composition:
                comp_cols = st.columns(len(final_composition))
                for i, (element, count) in enumerate(final_composition.items()):
                    with comp_cols[i]:
                        percentage = (count / original_total_atoms) * 100
                        st.metric(f"{element}", f"{count} atoms", f"{percentage:.1f}%")

            formula_parts = []
            for element, count in sorted(final_composition.items()):
                if count == 1:
                    formula_parts.append(element)
                else:
                    formula_parts.append(f"{element}{count}")

            predicted_formula = "".join(formula_parts) if formula_parts else "Empty"

            st.write(f"**Original Formula:** {structure.composition.reduced_formula} ({original_total_atoms} atoms)")
            st.write(f"**Predicted Formula:** {predicted_formula} ({final_total_atoms} atoms)")

            if total_vacancies > 0:
                vacancy_percentage = (total_vacancies / original_total_atoms) * 100
                st.write(f"**Vacancies Created:** {total_vacancies} ({vacancy_percentage:.1f}% of original sites)")

            if total_substitutions > 0:
                substitution_percentage = (total_substitutions / original_total_atoms) * 100
                st.write(
                    f"**Element Substitutions:** {total_substitutions} ({substitution_percentage:.1f}% of original sites)")

    else:
        st.info("💡 Select elements to substitute to enable GA optimization")

    return substitutions


def display_ga_results(ga_results):
    import streamlit as st
    import plotly.graph_objects as go
    import pandas as pd
    import zipfile
    import io
    from pymatgen.io.cif import CifWriter
    import numpy as np

    if not ga_results:
        return


    st.markdown("""
    <style>
    .ga-results {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .ga-results h1, .ga-results h2, .ga-results h3 {
        font-weight: 600;
        line-height: 1.2;
    }
    .ga-results .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .ga-results .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .ga-results .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 400;
    }
    .ga-results table {
        font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
        font-size: 0.9rem;
    }
    .ga-results .download-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ga-results">', unsafe_allow_html=True)

    st.markdown("## 🧬 Genetic Algorithm Results")

    all_runs = ga_results.get('all_runs', [])
    num_runs = len(all_runs)

    col_res1, col_res2, col_res3, col_res4 = st.columns(4)

    with col_res1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">BEST ENERGY FOUND</div>
            <div class="metric-value">{ga_results['best_energy']:.6f} eV</div>
        </div>
        """, unsafe_allow_html=True)

    with col_res2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">NUMBER OF RUNS</div>
            <div class="metric-value">{num_runs}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_res3:
        if all_runs:
            avg_generations = np.mean([len(run['fitness_history']) for run in all_runs])
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">AVG GENERATIONS</div>
                <div class="metric-value">{avg_generations:.0f}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_res4:
        if all_runs and len(all_runs[0]['fitness_history']) > 1:
            best_run = min(all_runs, key=lambda x: x['best_energy'])
            improvement = best_run['fitness_history'][0]['best'] - best_run['best_energy']
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">BEST IMPROVEMENT</div>
                <div class="metric-value">{improvement:.6f} eV</div>
            </div>
            """, unsafe_allow_html=True)

    if num_runs > 1:
        st.markdown("### 📊 Multi-Run Comparison")

        run_summary = []
        for run in all_runs:
            run_summary.append({
                'Run ID': run['run_id'] + 1,
                'Best Energy (eV)': f"{run['best_energy']:.6f}",
                'Generations': len(run['fitness_history']),
                'Final Avg Energy (eV)': f"{run['fitness_history'][-1]['average']:.6f}" if run[
                    'fitness_history'] else "N/A",
                'Final Population Size': len(run.get('final_population', []))
            })

        df_runs = pd.DataFrame(run_summary)
        st.dataframe(df_runs, use_container_width=True, hide_index=True)

    st.markdown("### 📋 Detailed Generation Analysis")

    if all_runs:
        if num_runs > 1:
            selected_run_id = st.selectbox("**Select run to analyze:**",
                                           options=range(num_runs),
                                           format_func=lambda
                                               x: f"Run {x + 1} (Best: {all_runs[x]['best_energy']:.6f} eV)")
            selected_run = all_runs[selected_run_id]
        else:
            selected_run = all_runs[0]

        if 'detailed_history' in selected_run:
            detailed_data = []
            for entry in selected_run['detailed_history']:
                detailed_data.append({
                    'Generation': entry['generation'],
                    'Best Energy (eV)': f"{entry['best']:.6f}",
                    'Average Energy (eV)': f"{entry['average']:.6f}",
                    'Worst Energy (eV)': f"{entry['worst']:.6f}",
                    'Energy Range (eV)': f"{entry['worst'] - entry['best']:.6f}",
                    'Improvement from Gen 0': f"{selected_run['detailed_history'][0]['best'] - entry['best']:.6f}"
                })

            df_detailed = pd.DataFrame(detailed_data)

            show_all_generations = st.checkbox("Show all generations", value=False)
            if not show_all_generations:
                max_rows = st.slider("Rows to display", 10, len(df_detailed), 20)
                df_display = df_detailed.tail(max_rows)
            else:
                df_display = df_detailed

            st.dataframe(df_display, use_container_width=True, hide_index=True)


            csv_detailed = df_detailed.to_csv(index=False)
            st.download_button(
                label="📥 Download Generation History (CSV)",
                data=csv_detailed,
                file_name=f"ga_generation_history_run_{selected_run['run_id'] + 1}.csv",
                mime="text/csv",
                type="secondary"
            )


    if all_runs:
        st.markdown("### 📈 Convergence Analysis")

        fig_conv = go.Figure()

        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7', '#ffecd2',
                  '#fcb69f']

        for i, run in enumerate(all_runs):
            generations = [f['generation'] for f in run['fitness_history']]
            best_energies = [f['best'] for f in run['fitness_history']]

            color = colors[i % len(colors)]
            fig_conv.add_trace(go.Scatter(
                x=generations,
                y=best_energies,
                mode='lines+markers',
                name=f'Run {run["run_id"] + 1}',
                line=dict(color=color, width=3),
                marker=dict(size=6)
            ))

        fig_conv.update_layout(
            title=dict(text="GA Convergence - All Runs", font=dict(size=20, family="Arial")),
            xaxis_title="Generation",
            yaxis_title="Best Energy (eV)",
            height=500,
            font=dict(size=14, family="Arial"),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        st.plotly_chart(fig_conv, use_container_width=True)


    st.markdown('<div class="download-section">', unsafe_allow_html=True)
    st.markdown("### 💾 Download Optimized Structures")


    def generate_structure_content(structure, output_format, **kwargs):
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase.io import write
            from ase.constraints import FixAtoms
            from pymatgen.io.cif import CifWriter
            from io import StringIO

            if output_format == "POSCAR":
                use_fractional = kwargs.get('use_fractional', True)
                use_selective = kwargs.get('use_selective', False)

                new_struct = Structure(structure.lattice, [], [])
                for site in structure:
                    new_struct.append(
                        species=site.species,
                        coords=site.frac_coords,
                        coords_are_cartesian=False,
                    )

                ase_structure = AseAtomsAdaptor.get_atoms(new_struct)

                if use_selective:
                    constraint = FixAtoms(indices=[])
                    ase_structure.set_constraint(constraint)

                out = StringIO()
                write(out, ase_structure, format="vasp", direct=use_fractional, sort=True)
                return out.getvalue(), ".vasp", "text/plain"

            elif output_format == "CIF":
                symprec = kwargs.get('cif_symprec', 0.1)
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
                file_content = CifWriter(new_struct, symprec=symprec, write_site_properties=True).__str__()
                return file_content, ".cif", "chemical/x-cif"

            elif output_format == "LAMMPS":
                lmp_style = kwargs.get('lmp_style', 'atomic')
                lmp_units = kwargs.get('lmp_units', 'metal')
                lmp_masses = kwargs.get('lmp_masses', True)
                lmp_skew = kwargs.get('lmp_skew', False)

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
                    atom_style=lmp_style, units=lmp_units,
                    masses=lmp_masses, force_skew=lmp_skew
                )
                return out.getvalue(), ".lmp", "text/plain"

            elif output_format == "XYZ":
                xyz_lines = []
                xyz_lines.append(str(len(structure)))

                lattice_matrix = structure.lattice.matrix
                lattice_string = " ".join([f"{x:.6f}" for row in lattice_matrix for x in row])

                comment_line = f'Lattice="{lattice_string}" Properties=species:S:1:pos:R:3'
                xyz_lines.append(comment_line)

                for site in structure:
                    if site.is_ordered:
                        element = site.specie.symbol
                    else:
                        element = max(site.species.items(), key=lambda x: x[1])[0].symbol

                    cart_coords = structure.lattice.get_cartesian_coords(site.frac_coords)
                    xyz_lines.append(f"{element} {cart_coords[0]:.6f} {cart_coords[1]:.6f} {cart_coords[2]:.6f}")

                return "\n".join(xyz_lines), ".xyz", "chemical/x-xyz"

        except Exception as e:
            return None, None, None

    col_down1, col_down2, col_down3 = st.columns(3)

    with col_down1:
        st.markdown("#### 🥇 Best Structure Download")
        best_structure = ga_results['best_structure']

        with st.expander("📁 Download Best Structure", expanded=False):
            output_format = st.selectbox(
                "Output format:",
                ["POSCAR", "CIF", "LAMMPS", "XYZ"],
                key="best_format",
                index=0
            )

            format_kwargs = {}

            if output_format == "POSCAR":
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    use_fractional = st.checkbox("Fractional coordinates", value=True, key="best_poscar_frac")
                with col_opt2:
                    use_selective = st.checkbox("Selective dynamics (all free)", value=False, key="best_poscar_sel")
                format_kwargs = {'use_fractional': use_fractional, 'use_selective': use_selective}

            elif output_format == "LAMMPS":
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    lmp_style = st.selectbox("Atom style:", ["atomic", "charge", "full"], index=0, key="best_lmp_style")
                    lmp_units = st.selectbox("Units:", ["metal", "real", "si"], index=0, key="best_lmp_units")
                with col_opt2:
                    lmp_masses = st.checkbox("Include masses", value=True, key="best_lmp_masses")
                    lmp_skew = st.checkbox("Force triclinic", value=False, key="best_lmp_skew")
                format_kwargs = {'lmp_style': lmp_style, 'lmp_units': lmp_units, 'lmp_masses': lmp_masses,
                                 'lmp_skew': lmp_skew}

            elif output_format == "CIF":
                cif_symprec = st.number_input("Symmetry precision:", value=0.1, min_value=0.001, max_value=1.0,
                                              step=0.001, format="%.3f", key="best_cif_symprec")
                format_kwargs = {'cif_symprec': cif_symprec}

            if st.button("📥 Generate Best Structure", key="gen_best_structure", type="primary"):
                file_content, file_ext, mime_type = generate_structure_content(best_structure, output_format,
                                                                               **format_kwargs)

                if file_content:
                    filename = f"ga_best_structure{file_ext}"
                    st.download_button(
                        label=f"📥 Download {output_format} File",
                        data=file_content,
                        file_name=filename,
                        mime=mime_type,
                        key="download_best_final",
                        type="secondary"
                    )
                else:
                    st.error(f"Error generating {output_format} file")

    with col_down2:
        st.markdown("#### 📦 Bulk Structure Download")
        if all_runs:
            if num_runs > 1:
                bulk_run_id = st.selectbox("**Select run for bulk download:**",
                                           options=range(num_runs),
                                           format_func=lambda x: f"Run {x + 1}",
                                           key="bulk_download_run")
                bulk_run = all_runs[bulk_run_id]
            else:
                bulk_run = all_runs[0]

            if bulk_run['final_population'] and bulk_run['final_fitness']:
                with st.expander("📦 Bulk Download Configuration", expanded=False):

                    percentage = st.slider("**Percentage of best structures to download**",
                                           min_value=10, max_value=100, value=50, step=10, key="bulk_percentage")

                    bulk_formats = st.multiselect(
                        "Select formats:",
                        ["POSCAR", "CIF", "LAMMPS", "XYZ"],
                        default=["POSCAR"],
                        key="bulk_format_selector"
                    )

                    bulk_format_kwargs = {}

                    if "POSCAR" in bulk_formats:
                        st.write("**VASP POSCAR Options:**")
                        col_bulk1, col_bulk2 = st.columns(2)
                        with col_bulk1:
                            bulk_vasp_fractional = st.checkbox("Fractional coordinates", value=True,
                                                               key="bulk_vasp_frac")
                        with col_bulk2:
                            bulk_vasp_selective = st.checkbox("Selective dynamics (all free)", value=False,
                                                              key="bulk_vasp_sel")
                        bulk_format_kwargs['POSCAR'] = {'use_fractional': bulk_vasp_fractional,
                                                        'use_selective': bulk_vasp_selective}

                    if "LAMMPS" in bulk_formats:
                        st.write("**LAMMPS Options:**")
                        col_bulk3, col_bulk4 = st.columns(2)
                        with col_bulk3:
                            bulk_lmp_style = st.selectbox("Atom style:", ["atomic", "charge", "full"], index=0,
                                                          key="bulk_lmp_style")
                            bulk_lmp_units = st.selectbox("Units:", ["metal", "real", "si"], index=0,
                                                          key="bulk_lmp_units")
                        with col_bulk4:
                            bulk_lmp_masses = st.checkbox("Include masses", value=True, key="bulk_lmp_masses")
                            bulk_lmp_skew = st.checkbox("Force triclinic", value=False, key="bulk_lmp_skew")
                        bulk_format_kwargs['LAMMPS'] = {'lmp_style': bulk_lmp_style, 'lmp_units': bulk_lmp_units,
                                                        'lmp_masses': bulk_lmp_masses, 'lmp_skew': bulk_lmp_skew}

                    if "CIF" in bulk_formats:
                        bulk_cif_symprec = st.number_input("CIF symmetry precision:", value=0.1, min_value=0.001,
                                                           max_value=1.0, step=0.001, format="%.3f",
                                                           key="bulk_cif_symprec")
                        bulk_format_kwargs['CIF'] = {'cif_symprec': bulk_cif_symprec}

                    population = bulk_run['final_population']
                    fitness = bulk_run['final_fitness']

                    sorted_indices = np.argsort(fitness)
                    top_percent_count = max(1, int(len(sorted_indices) * percentage / 100))
                    best_indices = sorted_indices[:top_percent_count]

                    st.write(f"**Will download {top_percent_count} structures (top {percentage}%)**")

                    def create_top_structures_zip():
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                            for i, idx in enumerate(best_indices):
                                structure = population[idx]
                                energy = fitness[idx]
                                rank = i + 1

                                for fmt in bulk_formats:
                                    try:
                                        kwargs = bulk_format_kwargs.get(fmt, {})
                                        file_content, file_ext, _ = generate_structure_content(structure, fmt, **kwargs)

                                        if file_content:
                                            filename = f"{fmt}/rank_{rank:03d}_energy_{energy:.6f}{file_ext}"
                                            zip_file.writestr(filename, file_content)

                                    except Exception as e:
                                        st.warning(f"Failed to convert structure {rank} to {fmt}: {str(e)}")
                                        continue

                            summary = f"Top {percentage}% structures from GA Run {bulk_run['run_id'] + 1}\n"
                            summary += f"Total structures: {len(best_indices)}\n"
                            summary += f"Best energy: {min(fitness[i] for i in best_indices):.6f} eV\n"
                            summary += f"Worst energy in selection: {max(fitness[i] for i in best_indices):.6f} eV\n"
                            summary += f"Energy range: {(max(fitness[i] for i in best_indices) - min(fitness[i] for i in best_indices)) * 1000:.3f} meV\n"
                            summary += f"Generation: Final (Generation {len(bulk_run['fitness_history']) - 1})\n"
                            summary += f"Population size: {len(population)}\n"
                            summary += f"Formats included: {', '.join(bulk_formats)}\n"

                            zip_file.writestr("README.txt", summary)

                        return zip_buffer.getvalue()

                    if bulk_formats and st.button("📦 Generate Top Structures ZIP", key="gen_bulk_zip"):
                        with st.spinner("Generating ZIP file..."):
                            zip_data = create_top_structures_zip()
                            st.download_button(
                                label=f"📥 Download Top {percentage}% ({top_percent_count} structures)",
                                data=zip_data,
                                file_name=f"ga_top_{percentage}percent_structures.zip",
                                mime="application/zip",
                                type="primary"
                            )

    with col_down3:
        st.markdown("#### 🏃 Multi-Run Download")
        if num_runs > 1:
            with st.expander("🔄 Best from Each Run", expanded=False):
                multirun_formats = st.multiselect(
                    "Select formats:",
                    ["POSCAR", "CIF", "LAMMPS", "XYZ"],
                    default=["POSCAR"],
                    key="multirun_format_selector"
                )

                multirun_kwargs = {}
                if "POSCAR" in multirun_formats:
                    multirun_vasp_frac = st.checkbox("POSCAR: Fractional coords", value=True, key="multirun_vasp_frac")
                    multirun_kwargs['POSCAR'] = {'use_fractional': multirun_vasp_frac, 'use_selective': False}

                if "LAMMPS" in multirun_formats:
                    multirun_lmp_style = st.selectbox("LAMMPS atom style:", ["atomic", "charge", "full"], index=0,
                                                      key="multirun_lmp_style")
                    multirun_kwargs['LAMMPS'] = {'lmp_style': multirun_lmp_style, 'lmp_units': 'metal',
                                                 'lmp_masses': True, 'lmp_skew': False}

                if "CIF" in multirun_formats:
                    multirun_kwargs['CIF'] = {'cif_symprec': 0.1}

                def create_best_from_each_run_zip():
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for run in all_runs:
                            structure = run['best_structure']
                            energy = run['best_energy']
                            run_id = run['run_id']

                            for fmt in multirun_formats:
                                try:
                                    kwargs = multirun_kwargs.get(fmt, {})
                                    file_content, file_ext, _ = generate_structure_content(structure, fmt, **kwargs)

                                    if file_content:
                                        filename = f"{fmt}/best_run_{run_id + 1:02d}_energy_{energy:.6f}{file_ext}"
                                        zip_file.writestr(filename, file_content)

                                except Exception as e:
                                    st.warning(f"Failed to convert run {run_id + 1} structure to {fmt}: {str(e)}")
                                    continue

                        summary = f"Best structures from {num_runs} GA runs\n"
                        summary += f"Overall best energy: {ga_results['best_energy']:.6f} eV\n\n"
                        summary += "Run-by-run results:\n"
                        for run in all_runs:
                            summary += f"Run {run['run_id'] + 1}: {run['best_energy']:.6f} eV ({len(run['fitness_history'])} generations)\n"
                        summary += f"\nFormats included: {', '.join(multirun_formats)}\n"
                        zip_file.writestr("README.txt", summary)

                    return zip_buffer.getvalue()

                if multirun_formats and st.button("📦 Generate Multi-Run ZIP", key="gen_multirun_zip"):
                    with st.spinner("Generating multi-run ZIP..."):
                        zip_data = create_best_from_each_run_zip()
                        st.download_button(
                            label=f"📥 Download Best from {num_runs} Runs",
                            data=zip_data,
                            file_name="ga_best_from_each_run.zip",
                            mime="application/zip",
                            type="primary"
                        )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🔬 Final Structure Analysis")

    best_structure = ga_results['best_structure']

    composition_data = []
    for element in set([site.specie.symbol for site in best_structure]):
        count = sum(1 for site in best_structure if site.specie.symbol == element)
        percentage = (count / len(best_structure)) * 100
        composition_data.append({
            'Element': element,
            'Count': count,
            'Percentage': f"{percentage:.1f}%"
        })

    df_composition = pd.DataFrame(composition_data)
    st.dataframe(df_composition, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)


def display_ga_overview():
    import streamlit as st
    st.divider()
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    ## 🧬 How the Genetic Algorithm Works in This Code

    ### **Core Concept:**
    The GA tries to find the **optimal arrangement of substituted atoms** (e.g., where to place 20% Ag atoms in a Ti structure) that gives the **lowest energy**, while keeping the exact substitution concentration fixed.

    ### **Step-by-Step Process:**

    #### **1. Initialization**
    - Creates **random substitution patterns** (e.g., 50 different ways to place 10 Ag atoms in 50 Ti sites)
    - Each "individual" = one specific arrangement of substituted atoms
    - Optionally adds random position perturbations to atoms (Max Random Position Displacement parameter when Optimize Atomic Positions is checked)
    

    #### **2. Fitness Evaluation** 
    - Calculates **energy** of each arrangement using MACE calculator
    - Lower energy = better fitness
    - If position optimization enabled: performs geometry optimization before energy calculation
    - **🎯 GUI Parameters**: 
      - **Optimize Atomic Positions** = enables geometry optimization

    #### **3. Selection (Tournament)**
    - Picks **best parents** for breeding using tournament selection
    - Randomly selects 3 individuals, chooses the one with lowest energy
    - Repeats to get 2 parents

    #### **4. Crossover (Breeding)**
    - **Combines substitution patterns** from 2 parents **OR** directly copies one parent
    - **🎯 GUI Parameter**: **Crossover Rate** controls this choice:
      - **80% Crossover Rate** = 80% of children created by **mixing two parents**, 20% by **copying one parent**
      - **When mixing**: Takes union of all substituted sites from both parents, then randomly selects exactly the required number to maintain concentration
      - **When copying**: Child is exact clone of one randomly chosen parent
    - **Example mixing**: Parent1 has Ag at sites [1,5,9], Parent2 has Ag at sites [3,5,7] → Child gets union [1,3,5,7,9], then randomly picks 3 of these 5 sites

    #### **5. Mutation**
    - **Each substituted atom individually** has a chance to be swapped to a different site (unlimited distance allowed)
    - **🎯 GUI Parameter**: **Mutation Rate** = probability that **each individual substituted atom** gets moved to a new site
    - **Example with Mutation Rate = 0.1 (10%)**:
      - 10 Ag atoms in structure → each Ag has 10% chance of being moved
      - **On average**: ~1 Ag atom will be swapped per mutation
      - **Could be**: 0, 1, 2, 3+ swaps depending on random chance
    - Maintains exact same concentration (always same number of substituted atoms)
    - **🎯 GUI Parameter**: **Max Position Displacement** = maximum distance for small random coordinate adjustments (separate from substitution swapping)

    #### **6. Evolution Loop**
    - Repeats steps 2-5 for multiple generations
    - **Elitism**: Always keeps best individuals from previous generation
    - Stops when converged or max generations reached
    - **🎯 GUI Parameters**:
      - **Max Generations** = maximum number of evolution cycles
      - **Elitism Ratio** = percentage of best arrangements automatically kept each generation (prevents losing good solutions)
      - **Convergence Threshold** = stops early when energy improvements become smaller than this value (the convergence is considered only above 20 generations and when 10 generations fulfill this treshold)

    #### **7. Multiple Independent Runs**
    - Runs the entire GA process multiple times with different random starting points
    - Reports the overall best result found across all runs
    - **🎯 GUI Parameter**: **Number of GA Runs** = how many independent GA optimizations to perform

    ---

    ### **Key Features:**
    - **Fixed concentration**: Always maintains exact substitution percentages (e.g., always exactly 20% Ag)
    - **Only optimizes arrangement**: Doesn't change which elements or how many, just WHERE they go
    - **Energy-driven**: Evolves toward arrangements that minimize total system energy
    - **Supports vacancies**: Can create holes in the structure by removing atoms

    """)




