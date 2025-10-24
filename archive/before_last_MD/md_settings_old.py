

import numpy as np
from ase.md import VelocityVerlet, Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen as NPT_Berendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
import json
import time
import streamlit as st
import os


NPT_AVAILABLE = False
NPT_TYPE = None

try:
    from ase.md.nptberendsen import NPTBerendsen

    NPT_AVAILABLE = True
    NPT_TYPE = "Berendsen"
    NPT = NPTBerendsen
except ImportError:
    try:
        from ase.md.npt import NPT

        NPT_AVAILABLE = True
        NPT_TYPE = "Nose-Hoover"
    except ImportError:
        NPT_AVAILABLE = False
        NPT_TYPE = None

print(f"MD Status: NPT Available = {NPT_AVAILABLE}, Type = {NPT_TYPE}")


class MDTrajectoryLogger:

    def __init__(self, log_queue, structure_name, md_params):
        self.log_queue = log_queue
        self.structure_name = structure_name
        self.md_params = md_params
        self.trajectory_data = []
        self.step_count = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.log_interval = md_params.get('log_interval', 10)

    def __call__(self):
        """Called at each MD step"""
        current_time = time.time()
        self.step_count += 1


        if self.step_count % self.log_interval == 0:
            atoms = self.md_object.atoms

            try:
                energy = atoms.get_potential_energy()
                kinetic_energy = atoms.get_kinetic_energy()
                total_energy = energy + kinetic_energy
                temperature = atoms.get_temperature()

                try:
                    stress = atoms.get_stress(voigt=True)
                    pressure = -np.mean(stress[:3])
                except:
                    pressure = None

                forces = atoms.get_forces()
                max_force = np.max(np.linalg.norm(forces, axis=1))

                step_data = {
                    'step': self.step_count,
                    'time_ps': self.step_count * self.md_params['timestep'] / 1000.0,
                    'potential_energy': energy,
                    'kinetic_energy': kinetic_energy,
                    'total_energy': total_energy,
                    'temperature': temperature,
                    'pressure': pressure,
                    'max_force': max_force,
                    'positions': atoms.positions.copy(),
                    'velocities': atoms.get_velocities().copy(),
                    'cell': atoms.cell.array.copy(),
                    'volume': atoms.get_volume(),
                    'timestamp': current_time
                }

                self.trajectory_data.append(step_data)

                if len(self.trajectory_data) >= 2:
                    prev_pos = self.trajectory_data[-2]['positions']
                    curr_pos = self.trajectory_data[-1]['positions']
                    max_displacement = np.max(np.linalg.norm(curr_pos - prev_pos, axis=1))

                    if self.step_count % (self.log_interval * 10) == 0:
                        self.log_queue.put(f"    Step-to-step displacement: {max_displacement:.6f} Å")

                elapsed_time = current_time - self.start_time
                if self.step_count > 0:
                    avg_time_per_step = elapsed_time / self.step_count
                    remaining_steps = self.md_params['n_steps'] - self.step_count
                    estimated_remaining = remaining_steps * avg_time_per_step
                else:
                    avg_time_per_step = 0
                    estimated_remaining = 0

                if current_time - self.last_log_time > 2.0:
                    progress = self.step_count / self.md_params['n_steps']

                    self.log_queue.put({
                        'type': 'md_step',
                        'structure': self.structure_name,
                        'step': self.step_count,
                        'total_steps': self.md_params['n_steps'],
                        'progress': progress,
                        'potential_energy': energy,
                        'kinetic_energy': kinetic_energy,
                        'total_energy': total_energy,
                        'temperature': temperature,
                        'pressure': pressure,
                        'avg_time_per_step': avg_time_per_step,
                        'estimated_remaining_time': estimated_remaining,
                        'elapsed_time': elapsed_time
                    })

                    self.last_log_time = current_time


                    log_message = (f"  MD Step {self.step_count}/{self.md_params['n_steps']}: "
                                   f"T = {temperature:.1f}K, "
                                   f"E_pot = {energy:.6f} eV, "
                                   f"E_total = {total_energy:.6f} eV, "
                                   f"F_max = {max_force:.4f} eV/Å")
                    if pressure is not None:
                        log_message += f", P = {pressure:.2f} GPa"

                    self.log_queue.put(log_message)

            except Exception as e:
                self.log_queue.put(f"  Warning: MD logging error at step {self.step_count}: {str(e)}")

    def set_md_object(self, md_object):
        """Set reference to MD object after creation"""
        self.md_object = md_object


def setup_md_parameters_ui():
    st.subheader("Molecular Dynamics Parameters")

    md_params = {}

    try:
        from ase.md.nptberendsen import NPTBerendsen
        npt_available = True
        npt_type = "Berendsen"
    except ImportError:
        try:
            from ase.md.npt import NPT
            npt_available = True
            npt_type = "Nose-Hoover"
        except ImportError:
            npt_available = False
            npt_type = None

    col_md1, col_md2, col_md3, col_md4 = st.columns(4)

    with col_md1:
        ensemble_options = ["NVE", "NVT-Langevin", "NVT-Berendsen"]
        if npt_available:
            ensemble_options.append("NPT")
        else:
            st.info("NPT ensemble not available in this ASE version")

        md_params['ensemble'] = st.selectbox(
            "Ensemble",
            ensemble_options,
            index=1,
            help="NVE: Constant energy, NVT: Constant temperature, NPT: Constant pressure & temperature"
        )

    with col_md2:
        md_params['n_steps'] = st.number_input(
            "Number of steps",
            min_value=100,
            max_value=1000000,
            value=10000,
            step=1000,
            help="Total number of MD steps to run"
        )

    with col_md3:
        md_params['timestep'] = st.number_input(
            "Timestep (fs)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.1f",
            help="MD timestep in femtoseconds"
        )

    with col_md4:
        md_params['temperature'] = st.number_input(
            "Temperature (K)",
            min_value=1,
            max_value=5000,
            value=300,
            step=50,
            help="Target temperature for the simulation"
        )


    if md_params['ensemble'] in ['NVT-Langevin', 'NPT']:
        st.write("**Thermostat Parameters**")
        col_thermo1, col_thermo2 = st.columns(2)

        with col_thermo1:
            md_params['friction'] = st.number_input(
                "Friction coefficient (1/ps)",
                min_value=0.001,
                max_value=1.0,
                value=0.02,
                step=0.001,
                format="%.3f",
                help="Langevin friction coefficient (higher = stronger coupling)"
            )

    elif md_params['ensemble'] == 'NVT-Berendsen':
        st.write("**Berendsen Thermostat Parameters**")
        col_berendsen1, col_berendsen2 = st.columns(2)

        with col_berendsen1:
            md_params['taut'] = st.number_input(
                "Temperature coupling (fs)",
                min_value=10.0,
                max_value=1000.0,
                value=100.0,
                step=10.0,
                help="Berendsen temperature coupling time constant"
            )

    if md_params['ensemble'] == 'NPT':
        if npt_available:
            st.write(f"**Barostat Parameters ({npt_type})**")
            col_baro1, col_baro2, col_baro3 = st.columns(3)

            with col_baro1:
                md_params['external_pressure'] = st.number_input(
                    "External pressure (GPa)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.1,
                    format="%.1f",
                    help="External pressure for NPT simulation"
                )

            with col_baro2:
                md_params['bulk_modulus'] = st.number_input(
                    "Bulk modulus (GPa)",
                    min_value=1.0,
                    max_value=1000.0,
                    value=140.0,
                    step=10.0,
                    help="Bulk modulus for compressibility calculation"
                )

            with col_baro3:
                md_params['taup'] = st.number_input(
                    "Pressure time constant (fs)",
                    min_value=10.0,
                    max_value=10000.0,
                    value=1000.0,
                    step=100.0,
                    help="Berendsen pressure coupling time constant"
                )

                if npt_type == "Berendsen":
                    md_params['taut'] = st.number_input(
                        "Temperature time constant (fs)",
                        min_value=10.0,
                        max_value=1000.0,
                        value=100.0,
                        step=10.0,
                        help="Berendsen temperature coupling time constant"
                    )
        else:
            st.error("NPT ensemble not available in your ASE version. Please use NVT instead.")

    st.write("**Initialization & Output**")
    col_init1, col_init2, col_init3, col_init4 = st.columns(4)

    with col_init1:
        md_params['remove_com_motion'] = st.checkbox(
            "Remove COM motion",
            value=True,
            help="Remove center-of-mass motion every step"
        )

    with col_init2:
        md_params['log_interval'] = st.number_input(
            "Log interval (steps)",
            min_value=1,
            max_value=1000,
            value=10,
            step=10,
            help="Frequency of logging MD properties"
        )

    with col_init3:
        md_params['traj_interval'] = st.number_input(
            "Trajectory interval (steps)",
            min_value=1,
            max_value=1000,
            value=100,
            step=10,
            help="Frequency of saving trajectory frames"
        )

    with col_init4:
        md_params['equilibration_steps'] = st.number_input(
            "Equilibration steps",
            min_value=0,
            max_value=50000,
            value=1000,
            step=500,
            help="Number of initial equilibration steps"
        )

    with st.expander("Advanced MD Options"):
        col_adv1, col_adv2, col_adv3 = st.columns(3)

        with col_adv1:
            try:
                from ase.md.rattle import Rattle
                md_params['use_rattle'] = st.checkbox(
                    "Use RATTLE constraints",
                    value=False,
                    help="Apply RATTLE algorithm for bond constraints"
                )

                if md_params['use_rattle']:
                    md_params['rattle_tolerance'] = st.number_input(
                        "RATTLE tolerance",
                        min_value=1e-8,
                        max_value=1e-4,
                        value=1e-6,
                        format="%.2e"
                    )
            except ImportError:
                st.info("RATTLE constraints not available")
                md_params['use_rattle'] = False

        with col_adv2:
            md_params['random_seed'] = st.number_input(
                "Random seed (0=random)",
                min_value=0,
                max_value=999999,
                value=42,
                step=1,
                help="Seed for reproducible simulations"
            )

        with col_adv3:
            md_params['fix_com'] = st.checkbox(
                "Fix center of mass",
                value=False,
                help="Keep center of mass fixed throughout simulation"
            )

    total_time_fs = md_params['n_steps'] * md_params['timestep']
    total_time_ps = total_time_fs / 1000.0

    st.info(f"**Simulation time:** {total_time_fs:.0f} fs = {total_time_ps:.2f} ps")

    if md_params['equilibration_steps'] > 0:
        eq_time_fs = md_params['equilibration_steps'] * md_params['timestep']
        production_time_fs = (md_params['n_steps'] - md_params['equilibration_steps']) * md_params['timestep']
        st.info(f"**Equilibration:** {eq_time_fs:.0f} fs, **Production:** {production_time_fs:.0f} fs")

    return md_params


def run_md_simulation(atoms, calculator, md_params, log_queue, structure_name):

    try:
        log_queue.put(f"Starting MD simulation for {structure_name}")
        log_queue.put(
            f"Ensemble: {md_params['ensemble']}, Steps: {md_params['n_steps']}, T: {md_params['temperature']}K")

        atoms.calc = calculator

        # Initialize velocities according to Maxwell-Boltzmann distribution
        log_queue.put("  Initializing velocities from Maxwell-Boltzmann distribution...")

        if md_params.get('random_seed', 0) != 0:
            np.random.seed(md_params['random_seed'])

        MaxwellBoltzmannDistribution(atoms, temperature_K=md_params['temperature'])

        initial_kinetic = atoms.get_kinetic_energy()
        initial_temp = atoms.get_temperature()
        log_queue.put(f"  Initial kinetic energy: {initial_kinetic:.6f} eV")
        log_queue.put(f"  Initial temperature: {initial_temp:.1f} K")

        velocities = atoms.get_velocities()
        max_velocity = np.max(np.linalg.norm(velocities, axis=1))
        log_queue.put(f"  Maximum velocity: {max_velocity:.6f} Å/fs")
        if max_velocity < 1e-6:
            log_queue.put("  ⚠️ Warning: Velocities are very small - atoms may not move!")

        if md_params['remove_com_motion']:
            momenta = atoms.get_momenta()
            com_momentum = momenta.sum(axis=0)
            momenta -= com_momentum / len(atoms)
            atoms.set_momenta(momenta)
            log_queue.put(f"  Center of mass motion removed")

        md_logger = MDTrajectoryLogger(log_queue, structure_name, md_params)

        timestep_ase_units = md_params['timestep'] * units.fs

        log_queue.put(f"  Using timestep: {md_params['timestep']} fs = {timestep_ase_units:.2e} ASE units")
        initial_positions = atoms.positions.copy()
        log_queue.put(f"  Initial position of first atom: {initial_positions[0]}")

        if md_params['ensemble'] == 'NVE':
            log_queue.put("  Setting up NVE (microcanonical) ensemble...")
            md_object = VelocityVerlet(atoms, timestep_ase_units)

        elif md_params['ensemble'] == 'NVT-Langevin':
            log_queue.put("  Setting up NVT ensemble with Langevin thermostat...")

            friction_ase_units = md_params['friction'] / units.fs
            md_object = Langevin(
                atoms,
                timestep_ase_units,
                temperature_K=md_params['temperature'],
                friction=friction_ase_units
            )

        elif md_params['ensemble'] == 'NVT-Berendsen':
            log_queue.put("  Setting up NVT ensemble with Berendsen thermostat...")
            taut_ase_units = md_params['taut'] * units.fs  # Convert fs to ASE units
            md_object = NVTBerendsen(
                atoms,
                timestep_ase_units,
                temperature_K=md_params['temperature'],
                taut=taut_ase_units
            )

        elif md_params['ensemble'] == 'NPT':
            if not NPT_AVAILABLE:
                raise ImportError("NPT ensemble not available in this ASE version")

            log_queue.put(f"  Setting up NPT ensemble ({NPT_TYPE})...")

            if NPT_TYPE == "Berendsen":
                md_object = NPT(
                    atoms,
                    timestep_ase_units,
                    temperature_K=md_params['temperature'],
                    pressure_au=md_params['external_pressure'] * units.GPa,
                    taut=md_params['taut'] * units.fs,
                    taup=md_params['taup'] * units.fs,
                    compressibility_au=1.0 / (md_params['bulk_modulus'] * units.GPa)
                )
            else:
                friction_ase_units = md_params['friction'] / units.fs
                pressure_ase = md_params['external_pressure'] * units.GPa
                bulk_modulus_ase = md_params['bulk_modulus'] * units.GPa
                taup_ase_units = md_params['taup'] * units.fs

                md_object = NPT(
                    atoms,
                    timestep_ase_units,
                    temperature_K=md_params['temperature'],
                    externalstress=pressure_ase,
                    ttime=1 / friction_ase_units,
                    pfactor=(taup_ase_units ** 2) * bulk_modulus_ase
                )

        if md_params.get('use_rattle', False):
            try:
                from ase.md.rattle import Rattle
                log_queue.put("  Applying RATTLE constraints...")
                rattle = Rattle(tolerance=md_params.get('rattle_tolerance', 1e-6))
                atoms.set_constraint(rattle)
            except ImportError:
                log_queue.put("  ⚠️ RATTLE not available, skipping constraints")

        md_logger.set_md_object(md_object)

        md_object.attach(md_logger, interval=1)

        log_queue.put("  Running diagnostic single-step test...")
        test_initial_pos = atoms.positions.copy()
        md_object.run(1)
        test_final_pos = atoms.positions.copy()
        test_displacement = np.max(np.linalg.norm(test_final_pos - test_initial_pos, axis=1))
        log_queue.put(f"  Single-step test displacement: {test_displacement:.8f} Å")

        if test_displacement < 1e-8:
            log_queue.put("  ❌ WARNING: Atoms barely moved in test - MD may have issues!")
        else:
            log_queue.put(f"  ✅ SUCCESS: Atoms moved properly - MD should work correctly")

        if md_params['equilibration_steps'] > 0:
            log_queue.put(f"  Running equilibration for {md_params['equilibration_steps']} steps...")
            md_object.run(md_params['equilibration_steps'] - 1)
            log_queue.put("  Equilibration completed, starting production run...")

            md_logger.trajectory_data = []
            md_logger.step_count = 0
            md_logger.start_time = time.time()

        production_steps = md_params['n_steps'] - md_params['equilibration_steps'] - 1
        log_queue.put(f"  Running production MD for {production_steps} steps...")

        md_object.run(production_steps)

        final_positions = atoms.positions.copy()
        position_change = np.max(np.linalg.norm(final_positions - initial_positions, axis=1))
        log_queue.put(f"  Total maximum atomic displacement: {position_change:.6f} Å")

        if position_change < 1e-6:
            log_queue.put("  ⚠️ Warning: Atoms barely moved! Check timestep, temperature, or constraints")
        else:
            log_queue.put(f"  ✅ Atoms moved successfully (max displacement: {position_change:.3f} Å)")

        final_energy = atoms.get_potential_energy()
        final_kinetic = atoms.get_kinetic_energy()
        final_temperature = atoms.get_temperature()

        try:
            final_stress = atoms.get_stress(voigt=True)
            final_pressure = -np.mean(final_stress[:3])
        except:
            final_stress = None
            final_pressure = None

        log_queue.put(f"✅ MD simulation completed for {structure_name}")
        log_queue.put(f"  Final temperature: {final_temperature:.1f} K")
        log_queue.put(f"  Final potential energy: {final_energy:.6f} eV")
        log_queue.put(f"  Final total energy: {final_energy + final_kinetic:.6f} eV")
        if final_pressure is not None:
            log_queue.put(f"  Final pressure: {final_pressure:.2f} GPa")

        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()
        final_structure = adaptor.get_structure(atoms)

        return {
            'success': True,
            'final_structure': final_structure,
            'final_energy': final_energy,
            'final_temperature': final_temperature,
            'final_pressure': final_pressure,
            'trajectory_data': md_logger.trajectory_data,
            'md_params': md_params,
            'total_steps_run': len(md_logger.trajectory_data),
            'simulation_time_ps': len(md_logger.trajectory_data) * md_params['timestep'] / 1000.0,
            'element_symbols': atoms.get_chemical_symbols()
        }

    except Exception as e:
        log_queue.put(f"❌ MD simulation failed for {structure_name}: {str(e)}")
        import traceback
        log_queue.put(f"Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e)
        }


def create_md_trajectory_xyz(trajectory_data, structure_name, md_params, element_symbols=None):

    if not trajectory_data:
        return None

    xyz_content = ""

    for step_data in trajectory_data:
        n_atoms = len(step_data['positions'])
        xyz_content += f"{n_atoms}\n"

        time_ps = step_data.get('time_ps', step_data['step'] * md_params['timestep'] / 1000.0)
        comment = (f"Step={step_data['step']} Time={time_ps:.3f}ps "
                   f"E_pot={step_data['potential_energy']:.6f}eV "
                   f"E_kin={step_data['kinetic_energy']:.6f}eV "
                   f"T={step_data['temperature']:.1f}K")

        if step_data.get('pressure') is not None:
            comment += f" P={step_data['pressure']:.2f}GPa"

        cell = step_data['cell']
        cell_flat = cell.flatten()
        lattice_str = " ".join([f"{x:.6f}" for x in cell_flat])
        comment += f' Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:vel:R:3'

        xyz_content += f"{comment}\n"

        positions = step_data['positions']
        velocities = step_data['velocities']

        for i in range(n_atoms):
            if element_symbols and i < len(element_symbols):
                symbol = element_symbols[i]
            else:
                symbol = "X"

            xyz_content += f"{symbol} {positions[i][0]:12.6f} {positions[i][1]:12.6f} {positions[i][2]:12.6f} "
            xyz_content += f"{velocities[i][0]:12.6f} {velocities[i][1]:12.6f} {velocities[i][2]:12.6f}\n"

    return xyz_content


def create_md_analysis_plots(trajectory_data, md_params):
    """Create analysis plots for MD trajectory with improved layout and debugging"""

    if not trajectory_data or len(trajectory_data) < 2:
        return None, None, None

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots


    steps = [data['step'] for data in trajectory_data]
    times_ps = [data.get('time_ps', data['step'] * md_params['timestep'] / 1000.0) for data in trajectory_data]
    pot_energies = [data['potential_energy'] for data in trajectory_data]
    kin_energies = [data['kinetic_energy'] for data in trajectory_data]
    total_energies = [data['potential_energy'] + data['kinetic_energy'] for data in trajectory_data]
    temperatures = [data['temperature'] for data in trajectory_data]
    volumes = [data.get('volume', 0) for data in trajectory_data]
    max_forces = [data.get('max_force', 0) for data in trajectory_data]
    pressures = [data.get('pressure') for data in trajectory_data if data.get('pressure') is not None]

    fig_main = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Potential Energy vs Time', 'Kinetic Energy vs Time',
            'Total Energy vs Time', 'Temperature vs Time',
            'Volume vs Time', 'Maximum Force vs Time'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    fig_main.add_trace(
        go.Scatter(x=times_ps, y=pot_energies, name='Potential Energy',
                   line=dict(color='blue', width=3), showlegend=False),
        row=1, col=1
    )

    fig_main.add_trace(
        go.Scatter(x=times_ps, y=kin_energies, name='Kinetic Energy',
                   line=dict(color='red', width=3), showlegend=False),
        row=1, col=2
    )

    fig_main.add_trace(
        go.Scatter(x=times_ps, y=total_energies, name='Total Energy',
                   line=dict(color='black', width=3), showlegend=False),
        row=2, col=1
    )

    # Temperature
    fig_main.add_trace(
        go.Scatter(x=times_ps, y=temperatures, name='Temperature',
                   line=dict(color='orange', width=3), showlegend=False),
        row=2, col=2
    )

    # Volume
    fig_main.add_trace(
        go.Scatter(x=times_ps, y=volumes, name='Volume',
                   line=dict(color='green', width=3), showlegend=False),
        row=3, col=1
    )

    # Maximum Force (debugging)
    fig_main.add_trace(
        go.Scatter(x=times_ps, y=max_forces, name='Max Force',
                   line=dict(color='purple', width=3), showlegend=False),
        row=3, col=2
    )

    fig_main.update_xaxes(title_text="Time (ps)", title_font=dict(size=20), tickfont=dict(size=16), row=3, col=1)
    fig_main.update_xaxes(title_text="Time (ps)", title_font=dict(size=20), tickfont=dict(size=16), row=3, col=2)
    fig_main.update_xaxes(title_text="Time (ps)", title_font=dict(size=20), tickfont=dict(size=16), row=2, col=1)
    fig_main.update_xaxes(title_text="Time (ps)", title_font=dict(size=20), tickfont=dict(size=16), row=2, col=2)
    fig_main.update_xaxes(title_text="Time (ps)", title_font=dict(size=20), tickfont=dict(size=16), row=1, col=1)
    fig_main.update_xaxes(title_text="Time (ps)", title_font=dict(size=20), tickfont=dict(size=16), row=1, col=2)

    fig_main.update_yaxes(title_text="Potential Energy (eV)", title_font=dict(size=20), tickfont=dict(size=16), row=1,
                          col=1)
    fig_main.update_yaxes(title_text="Kinetic Energy (eV)", title_font=dict(size=20), tickfont=dict(size=16), row=1,
                          col=2)
    fig_main.update_yaxes(title_text="Total Energy (eV)", title_font=dict(size=20), tickfont=dict(size=16), row=2,
                          col=1)
    fig_main.update_yaxes(title_text="Temperature (K)", title_font=dict(size=20), tickfont=dict(size=16), row=2, col=2)
    fig_main.update_yaxes(title_text="Volume (Å³)", title_font=dict(size=20), tickfont=dict(size=16), row=3, col=1)
    fig_main.update_yaxes(title_text="Max Force (eV/Å)", title_font=dict(size=20), tickfont=dict(size=16), row=3, col=2)

    fig_main.update_annotations(font_size=22)

    fig_main.update_layout(
        height=900,
        title=dict(text="MD Trajectory Analysis", font=dict(size=26)),
        showlegend=False,
        font=dict(size=18),
        margin=dict(l=80, r=80, t=100, b=80)
    )

    fig_pressure = None
    if pressures and len(pressures) == len(times_ps):
        fig_pressure = go.Figure()
        fig_pressure.add_trace(
            go.Scatter(x=times_ps, y=pressures, name='Pressure',
                       line=dict(color='darkblue', width=3))
        )

        fig_pressure.update_layout(
            title=dict(text="Pressure vs Time", font=dict(size=26)),
            xaxis=dict(title="Time (ps)", title_font=dict(size=20), tickfont=dict(size=16)),
            yaxis=dict(title="Pressure (GPa)", title_font=dict(size=20), tickfont=dict(size=16)),
            height=400,
            font=dict(size=18),
            showlegend=False,
            margin=dict(l=80, r=80, t=60, b=60)
        )

    energy_drift = [e - total_energies[0] for e in total_energies]
    fig_conservation = go.Figure()
    fig_conservation.add_trace(
        go.Scatter(x=times_ps, y=energy_drift, name='Energy Drift',
                   line=dict(color='red', width=3))
    )

    fig_conservation.update_layout(
        title=dict(text="Energy Conservation (Total Energy Drift)", font=dict(size=26)),
        xaxis=dict(title="Time (ps)", title_font=dict(size=20), tickfont=dict(size=16)),
        yaxis=dict(title="Energy Drift (eV)", title_font=dict(size=20), tickfont=dict(size=16)),
        height=400,
        font=dict(size=18),
        showlegend=False,
        margin=dict(l=80, r=80, t=60, b=60)
    )

    return fig_main, fig_pressure, fig_conservation


def export_md_results(md_results, structure_name):


    if not md_results['success']:
        return None

    export_data = {
        'structure_name': structure_name,
        'md_parameters': md_results['md_params'],
        'simulation_summary': {
            'total_steps': md_results['total_steps_run'],
            'simulation_time_ps': md_results['simulation_time_ps'],
            'final_energy_eV': md_results['final_energy'],
            'final_temperature_K': md_results['final_temperature'],
            'final_pressure_GPa': md_results.get('final_pressure'),
        },
        'trajectory_statistics': {}
    }

    if md_results['trajectory_data']:
        trajectory = md_results['trajectory_data']

        pot_energies = [data['potential_energy'] for data in trajectory]
        temperatures = [data['temperature'] for data in trajectory]

        export_data['trajectory_statistics'] = {
            'average_potential_energy_eV': float(np.mean(pot_energies)),
            'std_potential_energy_eV': float(np.std(pot_energies)),
            'average_temperature_K': float(np.mean(temperatures)),
            'std_temperature_K': float(np.std(temperatures)),
            'energy_drift_eV': float(pot_energies[-1] - pot_energies[0]) if len(pot_energies) > 1 else 0.0
        }

        if any(data.get('pressure') for data in trajectory):
            pressures = [data['pressure'] for data in trajectory if data.get('pressure') is not None]
            export_data['trajectory_statistics']['average_pressure_GPa'] = float(np.mean(pressures))
            export_data['trajectory_statistics']['std_pressure_GPa'] = float(np.std(pressures))

    return json.dumps(export_data, indent=2)
