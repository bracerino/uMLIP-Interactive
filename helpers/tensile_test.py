import numpy as np
import streamlit as st
from ase import Atoms
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import time
from collections import deque

def setup_tensile_test_ui(default_settings=None, save_settings_function=None):
    st.subheader("Virtual Tensile Test Parameters")


    defaults = {
            'strain_direction_index': 0,
            'strain_rate': 0.1,
            'max_strain': 10.0,
            'temperature': 300,
            'timestep': 1.0,
            'friction': 0.01,
            'equilibration_steps': 1000,
            'md_steps_per_increment': 100,
            'log_frequency': 10,
            'traj_frequency': 100,
            'relax_between_strain': False,
            'relax_steps': 100,
            'use_npt_transverse': False,
            'bulk_modulus': 110.0,
            'use_grip_mode': False,
            'grip_fraction': 0.1,
            'symmetric_pull': True,
        }
    if default_settings and 'tensile_test' in default_settings:
        defaults.update({k: v for k, v in default_settings['tensile_test'].items() if k in defaults})
        if 'strain_direction' in default_settings['tensile_test']:
             defaults['strain_direction_index'] = default_settings['tensile_test'].get('strain_direction', 0)


    col1, col2, col3 = st.columns(3)

    with col1:
        strain_direction_options = ["x (a-axis)", "y (b-axis)", "z (c-axis)"]
        strain_direction = st.selectbox(
            "Strain Direction",
            strain_direction_options,
            index=defaults['strain_direction_index'],
            help="Direction along which to apply strain"
        )
        strain_direction_index = strain_direction_options.index(strain_direction)


        strain_rate = st.number_input(
            "Strain Rate (%/ps)",
            min_value=0.001,
            max_value=50.0,
            value=defaults['strain_rate'],
            step=0.01,
            format="%.3f",
            help="Rate of strain application. Determines strain per MD step."
        )
        strain_rate_si = strain_rate * 1e10 # Convert %/ps to s^-1
        st.caption(f"📊 Equivalent to {strain_rate_si:.2e} s⁻¹")

        max_strain = st.number_input(
            "Maximum Strain (%)",
            min_value=0.05,
            max_value=50.00,
            value=defaults['max_strain'],
            step=1.00,
            help="Maximum total engineering strain before test stops"
        )

    with col2:
        temperature = st.number_input(
            "Temperature (K)",
            min_value=0,
            max_value=3000,
            value=defaults['temperature'],
            step=10,
            help="Temperature for MD thermostat"
        )

        timestep = st.number_input(
            "Timestep (fs)",
            min_value=0.1,
            max_value=5.0,
            value=defaults['timestep'],
            step=0.1,
            format="%.1f",
            help="MD integration timestep"
        )

        friction = st.number_input(
            "Friction (1/ps)",
            min_value=0.001,
            max_value=10.0,
            value=defaults['friction'],
            step=0.001,
            format="%.3f",
            help="Langevin thermostat friction coefficient (used for NVT parts)"
        )

    with col3:
        equilibration_steps = st.number_input(
            "Initial Equilibration Steps",
            min_value=0,
            max_value=10000,
            value=defaults['equilibration_steps'],
            step=100,
            help="Optional NPT steps at 0 P before straining"
        )

        md_steps_per_increment = st.number_input(
            "MD Steps per Increment",
            min_value=1,
            max_value=5000,
            value=defaults['md_steps_per_increment'],
            step=10,
            help="Number of MD steps run *between* each strain application. Strain is applied, then MD runs."
        )

        log_frequency = st.number_input( # New parameter
            "Log Frequency (steps)",
            min_value=1,
            max_value=md_steps_per_increment, # Cannot log more often than steps run
            value=max(1, min(defaults['log_frequency'], md_steps_per_increment)), # Ensure valid default
            step=1,
            help="Record CSV/Console data every N steps *during* the MD between increments."
        )

        # --- NEW: Trajectory Frequency ---
        traj_frequency = st.number_input(
            "Trajectory Frequency (steps)",
            min_value=1,
            max_value=md_steps_per_increment,
             # Ensure valid default, often less frequent than logging
            value=max(1, min(defaults['traj_frequency'], md_steps_per_increment)),
            step=10,
            help="Save XYZ frame every N steps *during* the MD between increments."
        )
        # --- END NEW ---


    relax_between_strain = st.checkbox(
        "Relax Between Strain Steps",
        value=defaults['relax_between_strain'],
        help="Run additional MD steps *after* applying strain but *before* logging stress (uses same dynamics as main MD)."
    )

    if relax_between_strain:
        relax_steps = st.number_input(
            "Relaxation Steps",
            min_value=1, # Allow at least 1 step
            max_value=5000,
            value=defaults.get('relax_steps', 100),
            step=10,
            help="Number of extra MD steps for relaxation if enabled."
        )
    else:
        relax_steps = 0


    st.write("**Transverse Pressure Control (Optional)**")
    use_npt_transverse = st.checkbox(
        "Use NPT in Transverse Directions",
        value=defaults['use_npt_transverse'],
        help="Apply pressure control (target P=0) in directions perpendicular to tensile strain (attempts true uniaxial tension)."
    )

    if use_npt_transverse:
        bulk_modulus = st.number_input(
            "Bulk Modulus (GPa)",
            min_value=1.0,
            max_value=1000.0,
            value=defaults['bulk_modulus'],
            step=10.0,
            help="Approximate bulk modulus for pressure coupling (NPT Berendsen)"
        )
    else:
        bulk_modulus = defaults['bulk_modulus'] # Keep value even if unused

    # =========================================================================
    # Grip-Based Pulling Mode (for finite structures in vacuum)
    # =========================================================================
    st.write("---")
    st.write("**Finite Structure Mode (Nanotubes / Molecules in Vacuum)**")

    use_grip_mode = st.checkbox(
        "Use Grip-Based Pulling (for finite structures in vacuum)",
        value=defaults['use_grip_mode'],
        help=(
            "Instead of scaling the periodic cell, this mode identifies atoms at "
            "the two ends of the structure, fixes one end (left grip), and "
            "incrementally displaces the other end (right grip). "
            "The middle atoms move freely during MD/relaxation. "
            "Use this for nanotubes, nanowires, or molecules in a vacuum box."
        )
    )

    grip_fraction = defaults['grip_fraction']
    symmetric_pull = defaults['symmetric_pull']

    if use_grip_mode:
        st.info(
            "**Grip mode active.** The code automatically identifies atoms at both ends "
            "of your structure along the chosen strain direction. Both grips are fixed "
            "during MD; each increment displaces the grips and lets the middle atoms respond. "
            "Output: axial force (eV/Å) vs engineering strain."
        )

        if use_npt_transverse:
            st.warning(
                "⚠️ Transverse NPT is not meaningful in grip mode "
                "(the cell is not being strained). It will be ignored."
            )

        grip_fraction = st.number_input(
            "Grip Fraction (per end)",
            min_value=0.01,
            max_value=0.4,
            value=defaults.get('grip_fraction', 0.1),
            step=0.01,
            format="%.2f",
            help=(
                "Fraction of the structure length at each end to treat as "
                "grip atoms. E.g., 0.1 means the bottom 10% and top 10% "
                "of atoms (by position along strain axis) are gripped."
            )
        )

        symmetric_pull = st.checkbox(
            "Symmetric Pull (both grips move)",
            value=defaults.get('symmetric_pull', True),
            help=(
                "If checked, both grips move in opposite directions each increment "
                "(each by half the displacement). This keeps the center of mass "
                "stationary and reduces artifacts. "
                "If unchecked, only the right grip moves while the left grip stays fixed."
            )
        )

    # --- Save Button Logic ---
    if st.button("💾 Save as Default Tensile Parameters", key="save_tensile_defaults"):
        new_tensile_settings = {
            'strain_direction_index': strain_direction_index,
            'strain_rate': strain_rate,
            'max_strain': max_strain,
            'temperature': temperature,
            'timestep': timestep,
            'friction': friction,
            'equilibration_steps': equilibration_steps,
            'md_steps_per_increment': md_steps_per_increment,
            'log_frequency': log_frequency,
            'traj_frequency': traj_frequency, # Save new param
            'relax_between_strain': relax_between_strain,
            'relax_steps': relax_steps,
            'use_npt_transverse': use_npt_transverse,
            'bulk_modulus': bulk_modulus,
            'use_grip_mode': use_grip_mode,
            'grip_fraction': grip_fraction,
            'symmetric_pull': symmetric_pull,
        }
        if 'default_settings' not in st.session_state:
            st.session_state.default_settings = {}
        st.session_state.default_settings['tensile_test'] = new_tensile_settings
        if save_settings_function and save_settings_function(st.session_state.default_settings):
            st.success("✅ Tensile test parameters saved as default!")
        else:
            st.error("❌ Failed to save tensile parameters")
    # --- End Save Button Logic ---

    strain_increment = (strain_rate / 100.0) * (timestep * md_steps_per_increment / 1000.0)
    num_increments = int(np.ceil((max_strain / 100.0) / strain_increment)) if strain_increment > 0 else 0
    total_md_steps = equilibration_steps + num_increments * (md_steps_per_increment + relax_steps)
    total_sim_time_ps = total_md_steps * timestep / 1000.0

    mode_str = "GRIP-BASED (finite structure)" if use_grip_mode else "CELL-SCALING (periodic)"
    st.info(f"Mode: {mode_str}\n"
            f"Strain increment: {strain_increment*100:.4f}% per {md_steps_per_increment} MD steps.\n"
            f"~{num_increments} increments needed for max strain.\n"
            f"Estimated total MD steps: ~{total_md_steps:,} ({total_sim_time_ps:.1f} ps)")

    return {
        'strain_direction': strain_direction_index,
        'strain_rate': strain_rate,
        'max_strain': max_strain,
        'temperature': temperature,
        'timestep': timestep,
        'friction': friction,
        'equilibration_steps': equilibration_steps,
        'md_steps_per_increment': md_steps_per_increment,
        'log_frequency': log_frequency,
        'traj_frequency': traj_frequency,
        'relax_between_strain': relax_between_strain,
        'relax_steps': relax_steps,
        'use_npt_transverse': use_npt_transverse,
        'bulk_modulus': bulk_modulus,
        'use_grip_mode': use_grip_mode,
        'grip_fraction': grip_fraction,
        'symmetric_pull': symmetric_pull,
    }

def apply_strain_increment(atoms, direction, strain_increment):
    cell = atoms.get_cell().copy()
    positions = atoms.get_positions().copy()

    cell[direction] *= (1.0 + strain_increment)

    fractional = np.linalg.solve(atoms.get_cell().T, positions.T).T

    atoms.set_cell(cell, scale_atoms=False)

    new_positions = fractional @ cell
    atoms.set_positions(new_positions)

    return atoms


def create_tensile_trajectory_xyz(trajectory_data, structure_name, tensile_params):
    if not trajectory_data:
        return None

    xyz_content = ""

    for step_data in trajectory_data:
        n_atoms = len(step_data['positions'])
        xyz_content += f"{n_atoms}\n"

        strain_percent = step_data['strain_percent']
        energy = step_data['energy']
        temp = step_data['temperature']

        # Build comment line — handle both cell-scaling (stress_GPa) and grip (axial_force) modes
        comment = f"Step={step_data['step']} Strain={strain_percent:.4f}%"
        if 'stress_GPa' in step_data:
            comment += f" Stress={step_data['stress_GPa']:.4f}GPa"
        if 'axial_force_eV_per_A' in step_data:
            comment += f" AxialForce={step_data['axial_force_eV_per_A']:.4f}eV/A"
        comment += f" E={energy:.6f}eV T={temp:.1f}K"

        cell = step_data['cell']
        cell_flat = cell.flatten()
        lattice_str = " ".join([f"{x:.6f}" for x in cell_flat])

        has_forces = 'forces' in step_data
        has_atom_energies = 'atom_energies' in step_data and step_data['atom_energies'] is not None

        if has_forces and has_atom_energies:
            comment += f' Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:forces:R:3:atom_energy:R:1'
        elif has_forces:
            comment += f' Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:forces:R:3'
        else:
            comment += f' Lattice="{lattice_str}" Properties=species:S:1:pos:R:3'

        xyz_content += f"{comment}\n"

        positions = step_data['positions']
        symbols = step_data['symbols']

        if has_forces:
            forces = step_data['forces']
            atom_energies = step_data.get('atom_energies', None)
            for i in range(n_atoms):
                line = (f"{symbols[i]} "
                        f"{positions[i][0]:12.6f} {positions[i][1]:12.6f} {positions[i][2]:12.6f} "
                        f"{forces[i][0]:12.6f} {forces[i][1]:12.6f} {forces[i][2]:12.6f}")
                if has_atom_energies:
                    line += f" {atom_energies[i]:12.6f}"
                xyz_content += line + "\n"
        else:
            for i in range(n_atoms):
                xyz_content += f"{symbols[i]} {positions[i][0]:12.6f} {positions[i][1]:12.6f} {positions[i][2]:12.6f}\n"

    return xyz_content


# =============================================================================
# Grip-mode helper functions
# =============================================================================

def identify_grip_atoms(atoms, direction, grip_fraction):
    """
    Identify left-grip and right-grip atom indices based on position
    along the strain axis.
    """
    positions = atoms.get_positions()
    coords = positions[:, direction]

    x_min = coords.min()
    x_max = coords.max()
    span = x_max - x_min

    if span < 1e-6:
        raise ValueError(
            f"Structure has zero extent along axis {direction}. "
            f"Check your structure or strain direction."
        )

    left_cutoff = x_min + grip_fraction * span
    right_cutoff = x_max - grip_fraction * span

    left_grip = np.where(coords <= left_cutoff)[0]
    right_grip = np.where(coords >= right_cutoff)[0]
    free_atoms = np.where((coords > left_cutoff) & (coords < right_cutoff))[0]

    initial_length = x_max - x_min

    return left_grip, right_grip, free_atoms, initial_length


def compute_grip_axial_force(atoms, left_grip_indices, right_grip_indices, direction):
    """
    Compute the net axial force from both grips (like LAMMPS f_mysf1 - f_mysf2).
    Reads forces directly from calculator results dict, completely bypassing
    ASE's constraint system which zeroes forces on FixAtoms atoms.

    Returns: (axial_force, f_left, f_right)
    - axial_force: average tension from both grips = (f_left - f_right) / 2
    - f_left: total raw force on left grip along direction (>0 = pulled rightward)
    - f_right: total raw force on right grip along direction (<0 = pulled leftward)
    """
    raw_forces = np.array(atoms.calc.results['forces'])
    f_left = float(np.sum(raw_forces[left_grip_indices, direction]))
    f_right = float(np.sum(raw_forces[right_grip_indices, direction]))
    axial_force = (f_left - f_right) / 2.0
    return axial_force, f_left, f_right



# =============================================================================
# Tensile test runners
# =============================================================================

def run_tensile_test(atoms, calculator, tensile_params, log_queue, structure_name, stop_event):
    """
    Dispatcher: choose cell-scaling or grip-based tensile test.
    """
    use_grip_mode = tensile_params.get('use_grip_mode', False)

    if use_grip_mode:
        return _run_tensile_test_grip(atoms, calculator, tensile_params,
                                      log_queue, structure_name, stop_event)
    else:
        return _run_tensile_test_cell_scaling(atoms, calculator, tensile_params,
                                              log_queue, structure_name, stop_event)


def _run_tensile_test_cell_scaling(atoms, calculator, tensile_params, log_queue, structure_name, stop_event):
    """Original cell-scaling tensile test (unchanged)."""
    try:
        log_queue.put(f"Starting virtual tensile test for {structure_name}")

        atoms.calc = calculator

        direction = tensile_params['strain_direction']
        direction_names = ['x', 'y', 'z']

        strain_increment_per_step = (tensile_params['strain_rate'] / 100.0) * (tensile_params['timestep'] / 1000.0)
        total_strain = tensile_params['max_strain'] / 100.0
        estimated_total_steps = int(total_strain / strain_increment_per_step)

        log_queue.put(f"  Strain direction: {direction_names[direction]}-axis")
        log_queue.put(f"  Max strain: {tensile_params['max_strain']}%")
        log_queue.put(f"  Strain rate: {tensile_params['strain_rate']}%/ps")
        log_queue.put(f"  Temperature: {tensile_params['temperature']} K")
        log_queue.put(f"  Strain increment per step: {strain_increment_per_step * 100:.6f}%")
        log_queue.put(f"  Estimated total steps: {estimated_total_steps}")

        if tensile_params['use_npt_transverse']:
            log_queue.put(f"  Using NPT in transverse directions (true uniaxial tension)")
        else:
            log_queue.put(f"  Using fixed cell in transverse directions")

        original_cell = atoms.get_cell().copy()
        original_length = np.linalg.norm(original_cell[direction])

        log_queue.put(f"  Original cell length ({direction_names[direction]}): {original_length:.4f} Å")

        MaxwellBoltzmannDistribution(atoms, temperature_K=tensile_params['temperature'])

        timestep_ase_units = tensile_params['timestep'] * units.fs
        friction_ase_units = tensile_params['friction'] / units.fs

        log_queue.put(f"  Timestep: {tensile_params['timestep']} fs = {timestep_ase_units:.2e} ASE units")
        log_queue.put(f"  Friction: {tensile_params['friction']} 1/ps = {friction_ase_units:.2e} ASE units")

        if tensile_params['equilibration_steps'] > 0:
            log_queue.put(
                f"  Equilibrating at {tensile_params['temperature']} K for {tensile_params['equilibration_steps']} steps...")

            from ase.md.npt import NPT

            dyn_eq = NPT(
                atoms,
                timestep_ase_units,
                temperature_K=tensile_params['temperature'],
                externalstress=0.0,  # Relax to zero stress
                ttime=50.0 * units.fs,
                pfactor=(75.0 * units.fs) ** 2 * units.GPa
            )
            log_queue.put(f"  Using NPT equilibration to relax to zero stress")

            dyn_eq.run(tensile_params['equilibration_steps'])

            eq_temp = atoms.get_temperature()
            eq_energy = atoms.get_potential_energy()
            log_queue.put(f"  Equilibration complete: T={eq_temp:.1f} K, E={eq_energy:.6f} eV")

        if tensile_params['use_npt_transverse']:
            try:
                from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen

                transverse_dirs = [i for i in range(3) if i != direction]

                mask = [0, 0, 0]
                for i in transverse_dirs:
                    mask[i] = 1
                mask = tuple(mask)

                bulk_modulus_ase = tensile_params['bulk_modulus'] * units.GPa

                dyn = Inhomogeneous_NPTBerendsen(
                    atoms,
                    timestep_ase_units,
                    temperature_K=tensile_params['temperature'],
                    pressure_au=0.0,
                    taut=100.0 * units.fs,
                    taup=1000.0 * units.fs,
                    compressibility_au=1.0 / bulk_modulus_ase,
                    mask=mask
                )

                log_queue.put(f"  Inhomogeneous NPT initialized with mask {mask}")
                log_queue.put(f"  Transverse directions will maintain zero pressure")

            except ImportError:
                log_queue.put(f"  Inhomogeneous_NPTBerendsen not available, falling back to NVT")
                dyn = Langevin(
                    atoms,
                    timestep_ase_units,
                    temperature_K=tensile_params['temperature'],
                    friction=friction_ase_units
                )
        else:
            dyn = Langevin(
                atoms,
                timestep_ase_units,
                temperature_K=tensile_params['temperature'],
                friction=friction_ase_units
            )

        strain_data = []
        stress_data = []
        energy_data = []
        temperature_data = []
        trajectory_data = []

        current_strain = 0.0

        step = 0
        ultimate_stress = 0.0
        yield_strain = None
        failure_detected = False

        step_start_time = None
        step_times = deque(maxlen=10)
        tensile_start_time = time.time()

        log_queue.put(f"  Starting strain application...")

        while current_strain < tensile_params['max_strain'] / 100.0 and not failure_detected:

            if stop_event.is_set():
                log_queue.put(f"  Tensile test stopped by user at {current_strain * 100:.2f}% strain")
                break

            current_time = time.time()

            if step_start_time is not None:
                step_duration = current_time - step_start_time
                step_times.append(step_duration)

            step_start_time = current_time

            if step % tensile_params['sample_interval'] == 0:
                apply_strain_increment(atoms, direction, strain_increment_per_step * tensile_params['sample_interval'])
                current_strain += strain_increment_per_step * tensile_params['sample_interval']

                if tensile_params['relax_between_strain']:
                    dyn.run(tensile_params['relax_steps'])

                try:
                    stress_tensor = atoms.get_stress(voigt=True)
                    stress_value = stress_tensor[direction] * 160.21766208

                    energy = atoms.get_potential_energy()
                    temp = atoms.get_temperature()
                    forces = atoms.get_forces()
                    max_force = np.max(np.linalg.norm(forces, axis=1))

                    strain_percent = current_strain * 100.0

                    # Try to get per-atom energies
                    try:
                        atom_energies = atoms.get_potential_energies().copy()
                    except Exception:
                        atom_energies = None

                    strain_data.append(strain_percent)
                    stress_data.append(stress_value)
                    energy_data.append(energy)
                    temperature_data.append(temp)

                    trajectory_data.append({
                        'step': step,
                        'strain_percent': strain_percent,
                        'stress_GPa': stress_value,
                        'energy': energy,
                        'temperature': temp,
                        'positions': atoms.positions.copy(),
                        'cell': atoms.cell.array.copy(),
                        'symbols': atoms.get_chemical_symbols(),
                        'forces': forces.copy(),
                        'atom_energies': atom_energies
                    })

                    if stress_value > ultimate_stress:
                        ultimate_stress = stress_value

                    if yield_strain is None and len(stress_data) > 5:
                        stress_diff = stress_data[-1] - stress_data[-2]
                        if stress_diff < 0 and stress_data[-1] > 0.1 * max(stress_data):
                            yield_strain = strain_percent

                    if len(step_times) >= 2:
                        avg_step_time = np.mean(list(step_times)[1:])
                        max_strain_steps = int((tensile_params['max_strain'] / 100.0 - current_strain) / (
                                    strain_increment_per_step * tensile_params['sample_interval']))
                        estimated_remaining_time = max_strain_steps * avg_step_time * tensile_params['sample_interval']
                    else:
                        avg_step_time = 0
                        estimated_remaining_time = None

                    elapsed_time = current_time - tensile_start_time

                    if step % (tensile_params['sample_interval'] * 5) == 0:
                        log_message = (f"  Step {step}: Strain={strain_percent:.2f}%, "
                                       f"Stress={stress_value:.2f} GPa, "
                                       f"T={temp:.1f}K, "
                                       f"E={energy:.6f} eV, "
                                       f"F_max={max_force:.4f} eV/Å")

                        if estimated_remaining_time is not None and estimated_remaining_time > 0:
                            if estimated_remaining_time < 60:
                                time_str = f"{estimated_remaining_time:.0f}s"
                            elif estimated_remaining_time < 3600:
                                time_str = f"{estimated_remaining_time / 60:.1f}m"
                            else:
                                time_str = f"{estimated_remaining_time / 3600:.1f}h"
                            log_message += f" | Est: {time_str}"

                        log_queue.put(log_message)

                    log_queue.put({
                        'type': 'tensile_step',
                        'structure': structure_name,
                        'step': step,
                        'total_steps': estimated_total_steps,
                        'strain_percent': strain_percent,
                        'stress_GPa': stress_value,
                        'temperature': temp,
                        'energy': energy,
                        'avg_step_time': avg_step_time,
                        'estimated_remaining_time': estimated_remaining_time,
                        'elapsed_time': elapsed_time
                    })



                except Exception as e:
                    log_queue.put(f"    Warning at step {step}: {str(e)}")
            else:
                dyn.run(1)

            step += 1

        if len(strain_data) > 5:
            elastic_region = min(10, len(strain_data) // 4)
            youngs_modulus = np.polyfit(
                strain_data[:elastic_region],
                stress_data[:elastic_region],
                1
            )[0] * 100
        else:
            youngs_modulus = None

        log_queue.put(f"  Tensile test complete")
        log_queue.put(f"  Final strain: {current_strain * 100:.2f}%")
        log_queue.put(f"  Ultimate stress: {ultimate_stress:.2f} GPa")
        if youngs_modulus:
            log_queue.put(f"  Young's modulus: {youngs_modulus:.2f} GPa")
        if yield_strain:
            log_queue.put(f"  Yield strain: {yield_strain:.2f}%")

        return {
            'success': True,
            'strain_data': strain_data,
            'stress_data': stress_data,
            'energy_data': energy_data,
            'temperature_data': temperature_data,
            'trajectory_data': trajectory_data,
            'ultimate_stress': ultimate_stress,
            'youngs_modulus': youngs_modulus,
            'yield_strain': yield_strain,
            'strain_direction': direction_names[direction],
            'max_strain_reached': current_strain * 100.0,
            'tensile_params': tensile_params,
            'failure_detected': failure_detected
        }

    except Exception as e:
        log_queue.put(f"Tensile test failed for {structure_name}: {str(e)}")
        import traceback
        log_queue.put(f"Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e)
        }


def _run_tensile_test_grip(atoms, calculator, tensile_params, log_queue, structure_name, stop_event):
    """
    Grip-based tensile test for finite structures in vacuum.

    Instead of scaling the periodic cell:
    - Identifies atoms at the two ends as "grips"
    - Fixes the left grip
    - Incrementally displaces the right grip
    - Fixes both grips during MD/relaxation
    - Free (middle) atoms respond to forces
    - Strain = imposed displacement / initial grip-to-grip length
    - Reports axial force (eV/Å) vs strain
    """
    try:
        log_queue.put(f"Starting GRIP-BASED tensile test for {structure_name}")

        atoms.calc = calculator

        direction = tensile_params['strain_direction']
        direction_names = ['x', 'y', 'z']
        grip_fraction = tensile_params.get('grip_fraction', 0.1)

        # --- Identify grip atoms ---
        left_grip, right_grip, free_atoms, initial_length = \
            identify_grip_atoms(atoms, direction, grip_fraction)

        all_grip = np.concatenate([left_grip, right_grip])

        log_queue.put(f"  Mode: Grip-based (finite structure in vacuum)")
        log_queue.put(f"  Strain direction: {direction_names[direction]}-axis")
        log_queue.put(f"  Left grip: {len(left_grip)} atoms (fixed)")
        log_queue.put(f"  Right grip: {len(right_grip)} atoms (displaced each increment)")
        log_queue.put(f"  Free atoms: {len(free_atoms)} atoms")
        log_queue.put(f"  Initial grip-to-grip length: {initial_length:.4f} Å")
        log_queue.put(f"  Max strain: {tensile_params['max_strain']}%")
        log_queue.put(f"  Strain rate: {tensile_params['strain_rate']}%/ps")
        log_queue.put(f"  Temperature: {tensile_params['temperature']} K")
        log_queue.put(f"  Output: Axial force (eV/Å) vs. strain")
        symmetric_pull = tensile_params.get('symmetric_pull', True)
        if symmetric_pull:
            log_queue.put(f"  Pull mode: Symmetric (both grips move in opposite directions)")
        else:
            log_queue.put(f"  Pull mode: One-sided (only right grip moves)")

        # Set initial velocities for free atoms only
        MaxwellBoltzmannDistribution(atoms, temperature_K=tensile_params['temperature'])
        velocities = atoms.get_velocities()
        velocities[all_grip] = 0.0
        atoms.set_velocities(velocities)

        # Fix both grips initially
        atoms.set_constraint(FixAtoms(indices=all_grip))

        timestep_ase = tensile_params['timestep'] * units.fs
        friction_ase = tensile_params['friction'] / units.fs

        # --- Equilibration (NVT only — NPT doesn't make sense for vacuum) ---
        if tensile_params['equilibration_steps'] > 0:
            log_queue.put(
                f"  Equilibrating free atoms at {tensile_params['temperature']} K "
                f"for {tensile_params['equilibration_steps']} steps (NVT)..."
            )
            dyn_eq = Langevin(
                atoms, timestep_ase,
                temperature_K=tensile_params['temperature'],
                friction=friction_ase
            )
            dyn_eq.run(tensile_params['equilibration_steps'])
            eq_temp = atoms.get_temperature()
            log_queue.put(f"  Equilibration done: T={eq_temp:.1f} K")

        # --- Dynamics for straining (always NVT in grip mode) ---
        dyn = Langevin(
            atoms, timestep_ase,
            temperature_K=tensile_params['temperature'],
            friction=friction_ase
        )

        # --- Strain loop ---
        strain_data = []
        force_data = []
        energy_data = []
        temperature_data = []
        trajectory_data = []

        # Compute displacement per increment
        md_steps = tensile_params['md_steps_per_increment']
        time_per_increment_ps = md_steps * tensile_params['timestep'] / 1000.0
        strain_per_increment = (tensile_params['strain_rate'] / 100.0) * time_per_increment_ps
        displacement_per_increment = strain_per_increment * initial_length

        max_strain_frac = tensile_params['max_strain'] / 100.0
        n_increments = int(np.ceil(max_strain_frac / strain_per_increment)) \
            if strain_per_increment > 0 else 0

        log_queue.put(f"  Displacement per increment: {displacement_per_increment:.6f} Å")
        log_queue.put(f"  Strain per increment: {strain_per_increment * 100:.4f}%")
        log_queue.put(f"  Number of increments: ~{n_increments}")

        total_displacement = 0.0
        step = 0
        ultimate_force = 0.0
        yield_strain = None

        step_times = deque(maxlen=10)
        tensile_start_time = time.time()

        log_queue.put(f"  Starting grip-based strain application...")

        for i in range(n_increments):
            if stop_event.is_set():
                log_queue.put(f"  Stopped by user at {total_displacement / initial_length * 100:.2f}% strain")
                break

            increment_start = time.time()

            # 1. Remove constraints temporarily
            atoms.set_constraint()

            # 2. Displace grip atoms
            positions = atoms.get_positions()
            symmetric_pull = tensile_params.get('symmetric_pull', True)
            if symmetric_pull:
                half_disp = displacement_per_increment / 2.0
                positions[right_grip, direction] += half_disp
                positions[left_grip, direction] -= half_disp
            else:
                positions[right_grip, direction] += displacement_per_increment
            atoms.set_positions(positions)
            total_displacement += displacement_per_increment

            # 3. Fix both grips
            atoms.set_constraint(FixAtoms(indices=all_grip))

            # 4. Zero grip velocities
            velocities = atoms.get_velocities()
            velocities[all_grip] = 0.0
            atoms.set_velocities(velocities)

            # 5. Relaxation between strain steps (optional)
            if tensile_params.get('relax_between_strain', False):
                relax_steps = tensile_params.get('relax_steps', 100)
                dyn.run(relax_steps)

            # 6. Run MD steps
            if md_steps > 0:
                dyn.run(md_steps)

            # 7. Compute observables
            try:
                current_strain = total_displacement / initial_length
                strain_percent = current_strain * 100.0

                axial_force, f_left, f_right = compute_grip_axial_force(
                    atoms, left_grip, right_grip, direction
                )

                energy = atoms.get_potential_energy()
                temp = atoms.get_temperature()
                forces = np.array(atoms.calc.results['forces'])  # Raw calculator forces
                max_force = np.max(np.linalg.norm(forces[free_atoms], axis=1)) \
                    if len(free_atoms) > 0 else 0.0

                # Try to get per-atom energies
                try:
                    atom_energies = atoms.get_potential_energies().copy()
                except Exception:
                    atom_energies = None

                strain_data.append(strain_percent)
                force_data.append(axial_force)
                energy_data.append(energy)
                temperature_data.append(temp)

                trajectory_data.append({
                    'step': step,
                    'strain_percent': strain_percent,
                    'axial_force_eV_per_A': axial_force,
                    'energy': energy,
                    'temperature': temp,
                    'positions': atoms.positions.copy(),
                    'cell': atoms.cell.array.copy(),
                    'symbols': atoms.get_chemical_symbols(),
                    'forces': forces.copy(),
                    'atom_energies': atom_energies
                })

                if axial_force > ultimate_force:
                    ultimate_force = axial_force

                # Yield detection
                if yield_strain is None and len(force_data) > 5:
                    if force_data[-1] < force_data[-2] and \
                       force_data[-1] > 0.1 * max(force_data):
                        yield_strain = strain_percent

                # Timing
                increment_time = time.time() - increment_start
                step_times.append(increment_time)

                if len(step_times) >= 2:
                    avg_step_time = np.mean(list(step_times)[1:])
                    estimated_remaining_time = (n_increments - i - 1) * avg_step_time
                else:
                    avg_step_time = 0
                    estimated_remaining_time = None

                elapsed_time = time.time() - tensile_start_time

                # Log every 5 increments
                if i % 5 == 0 or i == n_increments - 1:
                    log_message = (
                        f"  Inc {i + 1}/{n_increments}: "
                        f"ε={strain_percent:.2f}%, "
                        f"F={axial_force:.4f} eV/Å, "
                        f"T={temp:.1f}K, "
                        f"F_max={max_force:.4f} eV/Å"
                    )

                    if estimated_remaining_time is not None and estimated_remaining_time > 0:
                        if estimated_remaining_time < 60:
                            time_str = f"{estimated_remaining_time:.0f}s"
                        elif estimated_remaining_time < 3600:
                            time_str = f"{estimated_remaining_time / 60:.1f}m"
                        else:
                            time_str = f"{estimated_remaining_time / 3600:.1f}h"
                        log_message += f" | Est: {time_str}"

                    log_queue.put(log_message)

                # Send progress update
                log_queue.put({
                    'type': 'tensile_step',
                    'structure': structure_name,
                    'step': i,
                    'total_steps': n_increments,
                    'strain_percent': strain_percent,
                    'axial_force_eV_per_A': axial_force,
                    'temperature': temp,
                    'energy': energy,
                    'avg_step_time': avg_step_time,
                    'estimated_remaining_time': estimated_remaining_time,
                    'elapsed_time': elapsed_time
                })

            except Exception as e:
                log_queue.put(f"    Warning at increment {i}: {str(e)}")

            step += md_steps

            # Check max strain
            if total_displacement / initial_length >= max_strain_frac:
                log_queue.put(f"  Max strain reached.")
                break

        # --- Post-processing ---
        if len(strain_data) > 5:
            elastic_region = min(10, len(strain_data) // 4)
            stiffness = np.polyfit(
                strain_data[:elastic_region],
                force_data[:elastic_region],
                1
            )[0] * 100  # eV/Å per unit strain
        else:
            stiffness = None

        log_queue.put(f"  Grip-based tensile test complete")
        log_queue.put(f"  Final strain: {total_displacement / initial_length * 100:.2f}%")
        log_queue.put(f"  Ultimate force: {ultimate_force:.4f} eV/Å")
        if stiffness:
            log_queue.put(f"  Initial stiffness: {stiffness:.2f} eV/Å")
        if yield_strain:
            log_queue.put(f"  Yield strain: {yield_strain:.2f}%")

        return {
            'success': True,
            'strain_data': strain_data,
            'stress_data': force_data,  # force_data used as stress_data for compatibility
            'force_data': force_data,
            'energy_data': energy_data,
            'temperature_data': temperature_data,
            'trajectory_data': trajectory_data,
            'ultimate_stress': ultimate_force,  # keep key name for compatibility
            'youngs_modulus': stiffness,
            'yield_strain': yield_strain,
            'strain_direction': direction_names[direction],
            'max_strain_reached': total_displacement / initial_length * 100.0,
            'tensile_params': tensile_params,
            'failure_detected': False,
            'grip_mode': True,
            'initial_length': initial_length,
            'left_grip_count': len(left_grip),
            'right_grip_count': len(right_grip),
            'free_atom_count': len(free_atoms),
        }

    except Exception as e:
        log_queue.put(f"Grip tensile test failed for {structure_name}: {str(e)}")
        import traceback
        log_queue.put(f"Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e)
        }


def create_stress_strain_plot(tensile_results):
    strain = tensile_results['strain_data']
    stress = tensile_results['stress_data']

    grip_mode = tensile_results.get('grip_mode', False)

    if grip_mode:
        stress_label = "Axial Force (eV/Å)"
        stress_rate_label = "Force Rate (eV/Å/%)"
    else:
        stress_label = "Stress (GPa)"
        stress_rate_label = "Stress Rate (GPa/%)"

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'{"Force" if grip_mode else "Stress"}-Strain Curve',
            'Energy vs Strain',
            'Temperature vs Strain',
            f'{"Force" if grip_mode else "Stress"} Rate vs Strain'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    fig.add_trace(
        go.Scatter(
            x=strain,
            y=stress,
            mode='lines',
            name=stress_label.split('(')[0].strip(),
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    if tensile_results['youngs_modulus']:
        elastic_strain = [0, min(strain) if strain else 0]
        elastic_stress = [0, tensile_results['youngs_modulus'] * elastic_strain[1] / 100]

        fig.add_trace(
            go.Scatter(
                x=elastic_strain,
                y=elastic_stress,
                mode='lines',
                name='Linear Fit',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )

    if tensile_results['yield_strain']:
        fig.add_vline(
            x=tensile_results['yield_strain'],
            line_dash="dot",
            line_color="orange",
            annotation_text="Yield",
            row=1, col=1
        )

    max_stress_idx = stress.index(max(stress))
    fig.add_trace(
        go.Scatter(
            x=[strain[max_stress_idx]],
            y=[stress[max_stress_idx]],
            mode='markers',
            name='Ultimate',
            marker=dict(color='red', size=10, symbol='star')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=strain,
            y=tensile_results['energy_data'],
            mode='lines',
            name='Energy',
            line=dict(color='green', width=2),
            showlegend=False
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=strain,
            y=tensile_results['temperature_data'],
            mode='lines',
            name='Temperature',
            line=dict(color='purple', width=2),
            showlegend=False
        ),
        row=2, col=1
    )

    if len(stress) > 1:
        stress_rate = np.gradient(stress, strain)
        fig.add_trace(
            go.Scatter(
                x=strain,
                y=stress_rate,
                mode='lines',
                name='dσ/dε',
                line=dict(color='orange', width=2),
                showlegend=False
            ),
            row=2, col=2
        )

    fig.update_xaxes(title_text="Strain (%)", title_font=dict(size=20), tickfont=dict(size=16), row=1, col=1)
    fig.update_xaxes(title_text="Strain (%)", title_font=dict(size=20), tickfont=dict(size=16), row=1, col=2)
    fig.update_xaxes(title_text="Strain (%)", title_font=dict(size=20), tickfont=dict(size=16), row=2, col=1)
    fig.update_xaxes(title_text="Strain (%)", title_font=dict(size=20), tickfont=dict(size=16), row=2, col=2)

    fig.update_yaxes(title_text=stress_label, title_font=dict(size=20), tickfont=dict(size=16), row=1, col=1)
    fig.update_yaxes(title_text="Energy (eV)", title_font=dict(size=20), tickfont=dict(size=16), row=1, col=2)
    fig.update_yaxes(title_text="Temperature (K)", title_font=dict(size=20), tickfont=dict(size=16), row=2, col=1)
    fig.update_yaxes(title_text=stress_rate_label, title_font=dict(size=20), tickfont=dict(size=16), row=2, col=2)

    fig.update_annotations(font_size=22)

    mode_label = " (Grip Mode)" if grip_mode else ""
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Virtual Tensile Test Results ({tensile_results['strain_direction']}-direction){mode_label}",
        title_font_size=24,
        font=dict(size=18),
        margin=dict(l=80, r=80, t=100, b=80)
    )

    return fig


def export_tensile_results(tensile_results, structure_name):
    grip_mode = tensile_results.get('grip_mode', False)

    export_data = {
        'structure_name': structure_name,
        'test_parameters': {
            'strain_direction': tensile_results['strain_direction'],
            'max_strain': tensile_results['tensile_params']['max_strain'],
            'strain_rate': tensile_results['tensile_params']['strain_rate'],
            'temperature': tensile_results['tensile_params']['temperature'],
            'timestep_fs': tensile_results['tensile_params']['timestep'],
            'friction': tensile_results['tensile_params']['friction'],
            'grip_mode': grip_mode,
        },
        'mechanical_properties': {
            'ultimate_stress_GPa': float(tensile_results['ultimate_stress']),
            'youngs_modulus_GPa': float(tensile_results['youngs_modulus']) if tensile_results[
                'youngs_modulus'] else None,
            'yield_strain_percent': float(tensile_results['yield_strain']) if tensile_results['yield_strain'] else None,
            'max_strain_reached_percent': float(tensile_results['max_strain_reached']),
            'failure_detected': tensile_results.get('failure_detected', False)
        },
        'raw_data': {
            'strain_percent': [float(s) for s in tensile_results['strain_data']],
            'energy_eV': [float(e) for e in tensile_results['energy_data']],
            'temperature_K': [float(t) for t in tensile_results['temperature_data']]
        },
        'timestamp': datetime.now().isoformat()
    }

    if grip_mode:
        export_data['mechanical_properties']['ultimate_force_eV_per_A'] = export_data['mechanical_properties'].pop('ultimate_stress_GPa')
        export_data['mechanical_properties']['stiffness_eV_per_A'] = export_data['mechanical_properties'].pop('youngs_modulus_GPa')
        export_data['test_parameters']['grip_fraction'] = tensile_results['tensile_params'].get('grip_fraction', 0.1)
        export_data['mechanical_properties']['initial_length_A'] = tensile_results.get('initial_length')
        export_data['mechanical_properties']['left_grip_atoms'] = tensile_results.get('left_grip_count')
        export_data['mechanical_properties']['right_grip_atoms'] = tensile_results.get('right_grip_count')
        export_data['mechanical_properties']['free_atoms'] = tensile_results.get('free_atom_count')
        if 'force_data' in tensile_results:
            export_data['raw_data']['axial_force_eV_per_A'] = [float(f) for f in tensile_results['force_data']]
    else:
        export_data['raw_data']['stress_GPa'] = [float(s) for s in tensile_results['stress_data']]

    return json.dumps(export_data, indent=2)
