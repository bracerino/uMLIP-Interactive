import numpy as np
import streamlit as st
from ase import Atoms
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
            'bulk_modulus': 110.0
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
            'bulk_modulus': bulk_modulus
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

    st.info(f"Strain increment: {strain_increment*100:.4f}% per {md_steps_per_increment} MD steps.\n"
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
        'bulk_modulus': bulk_modulus
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
        stress = step_data['stress_GPa']
        energy = step_data['energy']
        temp = step_data['temperature']

        comment = (f"Step={step_data['step']} Strain={strain_percent:.4f}% "
                   f"Stress={stress:.4f}GPa E={energy:.6f}eV T={temp:.1f}K")

        cell = step_data['cell']
        cell_flat = cell.flatten()
        lattice_str = " ".join([f"{x:.6f}" for x in cell_flat])

        has_forces = 'forces' in step_data
        if has_forces:
            comment += f' Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:forces:R:3'
        else:
            comment += f' Lattice="{lattice_str}" Properties=species:S:1:pos:R:3'

        xyz_content += f"{comment}\n"

        positions = step_data['positions']
        symbols = step_data['symbols']

        if has_forces:
            forces = step_data['forces']
            for i in range(n_atoms):
                xyz_content += (f"{symbols[i]} "
                                f"{positions[i][0]:12.6f} {positions[i][1]:12.6f} {positions[i][2]:12.6f} "
                                f"{forces[i][0]:12.6f} {forces[i][1]:12.6f} {forces[i][2]:12.6f}\n")
        else:
            for i in range(n_atoms):
                xyz_content += f"{symbols[i]} {positions[i][0]:12.6f} {positions[i][1]:12.6f} {positions[i][2]:12.6f}\n"

    return xyz_content


def run_tensile_test(atoms, calculator, tensile_params, log_queue, structure_name, stop_event):
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
                        'forces': forces.copy()
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


def create_stress_strain_plot(tensile_results):
    strain = tensile_results['strain_data']
    stress = tensile_results['stress_data']

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Stress-Strain Curve',
            'Energy vs Strain',
            'Temperature vs Strain',
            'Stress Rate vs Strain'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    fig.add_trace(
        go.Scatter(
            x=strain,
            y=stress,
            mode='lines',
            name='Stress',
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
            name='Ultimate Stress',
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

    fig.update_yaxes(title_text="Stress (GPa)", title_font=dict(size=20), tickfont=dict(size=16), row=1, col=1)
    fig.update_yaxes(title_text="Energy (eV)", title_font=dict(size=20), tickfont=dict(size=16), row=1, col=2)
    fig.update_yaxes(title_text="Temperature (K)", title_font=dict(size=20), tickfont=dict(size=16), row=2, col=1)
    fig.update_yaxes(title_text="Stress Rate (GPa/%)", title_font=dict(size=20), tickfont=dict(size=16), row=2, col=2)

    fig.update_annotations(font_size=22)

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Virtual Tensile Test Results ({tensile_results['strain_direction']}-direction)",
        title_font_size=24,
        font=dict(size=18),
        margin=dict(l=80, r=80, t=100, b=80)
    )

    return fig


def export_tensile_results(tensile_results, structure_name):
    export_data = {
        'structure_name': structure_name,
        'test_parameters': {
            'strain_direction': tensile_results['strain_direction'],
            'max_strain': tensile_results['tensile_params']['max_strain'],
            'strain_rate': tensile_results['tensile_params']['strain_rate'],
            'temperature': tensile_results['tensile_params']['temperature'],
            'timestep_fs': tensile_results['tensile_params']['timestep'],
            'friction': tensile_results['tensile_params']['friction']
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
            'stress_GPa': [float(s) for s in tensile_results['stress_data']],
            'energy_eV': [float(e) for e in tensile_results['energy_data']],
            'temperature_K': [float(t) for t in tensile_results['temperature_data']]
        },
        'timestamp': datetime.now().isoformat()
    }

    return json.dumps(export_data, indent=2)
