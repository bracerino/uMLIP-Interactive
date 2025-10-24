import pprint
import textwrap
from collections import deque


def generate_md_python_script(md_params, selected_model, model_size, device, dtype, thread_count):
    """
    Generates a standalone Python script for running an MD simulation
    based on the provided parameters.

    This script is designed to load structures from the local directory
    (e.g., .cif, .vasp) and run the MD simulation. It now includes
    console logging with time estimation and saves trajectories in XYZ format.

    Args:
        md_params (dict): Dictionary of MD parameters from the Streamlit UI.
        selected_model (str): The full name of the model (e.g., "MACE-MP-0b3...").
        model_size (str): The model key/path (e.g., "medium-0b3").
        device (str): "cpu" or "cuda".
        dtype (str): "float32" or "float64".
        thread_count (int): The number of CPU threads to use.

    Returns:
        str: A string containing the complete, runnable Python script.
    """

    # 1. Generate the Calculator Setup Code Block
    calculator_setup_str = ""
    imports_str = f"""
import os
import glob
import time
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from ase import units
from ase.io import read 
from ase.build import make_supercell
from ase.md import VelocityVerlet, Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.logger import MDLogger
from collections import deque 
import io 

try:
    from nequip.ase.nequip_calculator import NequIPCalculator
except ImportError:
    print("Warning: NequIP not found. Will fail if Nequix model is selected.")
try:
    from deepmd.calculator import DP
except ImportError:
    print("Warning: DeePMD-kit not found. Will fail if DeePMD model is selected.")

os.environ['OMP_NUM_THREADS'] = '{thread_count}'
torch.set_num_threads({thread_count})
"""

    if "CHGNet" in selected_model:
        imports_str += """
try:
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
except ImportError:
    print("Error: CHGNet not found. Please install with: pip install chgnet")
    exit()
"""
        calculator_setup_str = f"""
print("Setting up CHGNet calculator...")
original_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.float32)
try:
    chgnet = CHGNet.load(model_name="{model_size}", use_device="{device}", verbose=False)
    calculator = CHGNetCalculator(model=chgnet, use_device="{device}")
    torch.set_default_dtype(original_dtype)
    print(f"✅ CHGNet {model_size} initialized on {device}")
except Exception as e:
    print(f"❌ CHGNet initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        chgnet = CHGNet.load(model_name="{model_size}", use_device="cpu", verbose=False)
        calculator = CHGNetCalculator(model=chgnet, use_device="cpu")
        torch.set_default_dtype(original_dtype)
        print("✅ CHGNet initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ CHGNet CPU fallback failed: {{cpu_e}}")
        exit()
"""
    elif "MACE-OFF" in selected_model:
        imports_str += """
try:
    from mace.calculators import mace_off
except ImportError:
    print("Error: MACE-OFF not found. Please install with: pip install mace-torch")
    exit()
"""
        calculator_setup_str = f"""
print("Setting up MACE-OFF calculator...")
try:
    calculator = mace_off(
        model="{model_size}", default_dtype="{dtype}", device="{device}"
    )
    print(f"✅ MACE-OFF {model_size} initialized on {device}")
except Exception as e:
    print(f"❌ MACE-OFF initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = mace_off(
            model="{model_size}", default_dtype="{dtype}", device="cpu"
        )
        print("✅ MACE-OFF initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ MACE-OFF CPU fallback failed: {{cpu_e}}")
        exit()
"""
    elif "MACE" in selected_model:
        imports_str += """
try:
    from mace.calculators import mace_mp
except ImportError:
    print("Error: MACE not found. Please install with: pip install mace-torch")
    exit()
"""
        calculator_setup_str = f"""
print("Setting up MACE calculator...")
try:
    calculator = mace_mp(
        model="{model_size}", dispersion=False, default_dtype="{dtype}", device="{device}"
    )
    print(f"✅ MACE {model_size} initialized on {device}")
except Exception as e:
    print(f"❌ MACE initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = mace_mp(
            model="{model_size}", dispersion=False, default_dtype="{dtype}", device="cpu"
        )
        print("✅ MACE initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ MACE-OFF CPU fallback failed: {{cpu_e}}")
        exit()
"""
    elif "SevenNet" in selected_model:
        imports_str += """
try:
    from sevenn.calculator import SevenNetCalculator
except ImportError:
    print("Error: SevenNet not found. Please install with: pip install sevenn")
    exit()
"""
        calculator_setup_str = f"""
print("Setting up SevenNet calculator...")
print("  Applying torch.load workaround for SevenNet (allowlisting 'slice')...")
try:
    torch.serialization.add_safe_globals([slice])
except AttributeError:
    print("  ... running on older torch version, add_safe_globals not needed.")
    pass
original_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.float32)
try:
    calculator = SevenNetCalculator(model="{model_size}", device="{device}")
    torch.set_default_dtype(original_dtype)
    print(f"✅ SevenNet {model_size} initialized on {device}")
except Exception as e:
    print(f"❌ SevenNet initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = SevenNetCalculator(model="{model_size}", device="cpu")
        torch.set_default_dtype(original_dtype)
        print("✅ SevenNet initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ SevenNet CPU fallback failed: {{cpu_e}}")
        exit()
"""
    elif "ORB" in selected_model:
        imports_str += """
try:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
except ImportError:
    print("Error: ORB models not found. Please install with: pip install orb-models")
    exit()
"""
        precision = "float32-high" if dtype == "float32" else "float32-highest"
        calculator_setup_str = f"""
print("Setting up ORB calculator...")
precision = "{precision}"
try:
    model_func = getattr(pretrained, "{model_size}")
    orbff = model_func(device="{device}", precision=precision)
    calculator = ORBCalculator(orbff, device="{device}")
    print(f"✅ ORB {model_size} initialized on {device}")
except Exception as e:
    print(f"❌ ORB initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        model_func = getattr(pretrained, "{model_size}")
        orbff = model_func(device="cpu", precision=precision)
        calculator = ORBCalculator(orbff, device="cpu")
        print("✅ ORB initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ ORB CPU fallback failed: {{cpu_e}}")
        exit()
"""
    # --- NEW: Nequix and DeePMD implementation ---
    elif "Nequix" in selected_model:
        calculator_setup_str = f"""
print("Setting up Nequix calculator...")
try:
    calculator = NequIPCalculator.from_deployed_model(
        model_path="{model_size}",
        device="{device}"
    )
    print(f"✅ Nequix {model_size} initialized on {device}")
except Exception as e:
    print(f"❌ Nequix initialization failed: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = NequIPCalculator.from_deployed_model(
            model_path="{model_size}",
            device="cpu"
        )
        print("✅ Nequix initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ Nequix CPU fallback failed: {{cpu_e}}")
        exit()
"""
    elif "DeePMD" in selected_model:
        calculator_setup_str = f"""
print("Setting up DeePMD calculator...")
try:
    calculator = DP(model="{model_size}")
    print(f"✅ DeePMD {model_size} initialized")
except Exception as e:
    print(f"❌ DeePMD initialization failed: {{e}}")
    exit()
"""
    # --- END NEW ---
    else:
        calculator_setup_str = "print('Error: Could not determine calculator type.')\ncalculator = None\nexit()"

    # 2. Clean up md_params and pretty-print it
    # Fix for NPT explosion: Standardize on 'taup' and remove old/conflicting keys

    # 1. Get the value from *either* key, prioritizing the UI's 'pressure_damping_time'
    taup_val = md_params.get('taup', 1000.0)  # Default if neither exists
    if 'pressure_damping_time' in md_params:
        taup_val = md_params.pop('pressure_damping_time')  # Get value from this key
    elif 'taup' in md_params:
        taup_val = md_params.pop('taup')  # Otherwise, get it from this key

    # 2. Delete both keys to be safe
    if 'taup' in md_params: md_params.pop('taup')
    if 'pressure_damping_time' in md_params: md_params.pop('pressure_damping_time')

    # 3. Set the *correct* key and value
    md_params['taup'] = taup_val
    # --- END NPT FIX ---

    md_params_str = pprint.pformat(md_params, indent=4, width=80)

    indented_calculator_setup = textwrap.indent(calculator_setup_str, "    ")

    # 3. Add Custom Logger and XYZ Writer Classes to the script
    custom_classes_str = """
class ConsoleMDLogger:
    def __init__(self, atoms, total_steps, log_interval=10, steps_for_avg=10):
        self.atoms = atoms
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.step_times = deque(maxlen=steps_for_avg)
        self.step_count = 0
        self.averaging_started = False
        self.start_time = time.perf_counter()
        self.last_log_time = time.perf_counter()
        self.step_start_time = time.perf_counter()

    def __call__(self):
        self.step_count += 1
        current_time = time.perf_counter()
        step_duration = current_time - self.step_start_time

        if self.step_count > self.log_interval:
            self.step_times.append(step_duration)
            self.averaging_started = True

        self.step_start_time = current_time

        if self.step_count % self.log_interval == 0:
            elapsed_time = current_time - self.start_time

            avg_step_time = 0
            estimated_remaining_time = None

            if self.averaging_started and len(self.step_times) > 0:
                avg_step_time = np.mean(list(self.step_times))
                remaining_steps = self.total_steps - self.step_count
                if remaining_steps > 0:
                    estimated_remaining_time = remaining_steps * avg_step_time

            try:
                epot = self.atoms.get_potential_energy()
                ekin = self.atoms.get_kinetic_energy()
                temp = self.atoms.get_temperature()
                try:
                    stress = self.atoms.get_stress(voigt=True)
                    pressure = -np.mean(stress[:3]) / units.GPa
                except Exception:
                    pressure = None
            except Exception as e:
                print(f"  Warning: Error getting properties at step {self.step_count}: {e}")
                return

            log_str = f"  MD Step {self.step_count}/{self.total_steps}: "
            log_str += f"T = {temp:.1f} K, "
            log_str += f"E_pot = {epot:.4f} eV, "
            log_str += f"E_kin = {ekin:.4f} eV"
            if pressure is not None:
                log_str += f", P = {pressure:.2f} GPa"

            if estimated_remaining_time is not None:
                log_str += f" | Avg/step: {avg_step_time:.2f}s"
                log_str += f" | Est. time: {self._format_time(estimated_remaining_time)}"
                log_str += f" | Elapsed: {self._format_time(elapsed_time)}"
            elif self.step_count <= self.log_interval:
                log_str += " | (Calculating time estimates...)"

            print(log_str)
            self.last_log_time = current_time

    def _format_time(self, seconds):
        if seconds < 0: return "N/A"
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

class XYZTrajectoryWriter:
    def __init__(self, atoms, filename="md_trajectory.xyz"):
        self.atoms = atoms
        self.filename = filename
        self.file = open(self.filename, 'w', encoding='utf-8') 
        self.step_count = 0
        print(f"  Writing XYZ trajectory to: {self.filename} (will overwrite)")

    def __call__(self):
        try:
            current_atoms = self.atoms
            positions = current_atoms.get_positions()
            symbols = current_atoms.get_chemical_symbols()
            num_atoms = len(current_atoms)

            energy = current_atoms.get_potential_energy()
            temp = current_atoms.get_temperature()
            cell = current_atoms.get_cell()
            lattice_str = " ".join(f"{x:.8f}" for x in cell.array.flatten())

            comment = (f'Step={self.step_count} Time={self.step_count * md_params["timestep"]:.3f}fs '
                       f'Energy={energy:.6f}eV Temp={temp:.2f}K '
                       f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3')

            self.file.write(f"{num_atoms}\\n")
            self.file.write(f"{comment}\\n")

            for i in range(num_atoms):
                self.file.write(f"{symbols[i]} {positions[i, 0]:15.8f} {positions[i, 1]:15.8f} {positions[i, 2]:15.8f}\\n")

            self.file.flush()

            self.step_count += md_params['traj_interval']

        except Exception as e:
            print(f"  Error writing XYZ frame at step {self.step_count}: {e}")

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            print(f"  Closed XYZ trajectory file: {self.filename}")

class LatticeLogger:
    def __init__(self, atoms, dyn, filename="md_lattice.log"):
        self.atoms = atoms
        self.dyn = dyn
        self.filename = filename
        self.file = open(self.filename, 'w', encoding='utf-8')

        header = f"#{'Time[ps]':>11} {'a[A]':>10} {'b[A]':>10} {'c[A]':>10} {'alpha[deg]':>10} {'beta[deg]':>10} {'gamma[deg]':>10}\\n"

        self.file.write(header)
        self.file.flush()
        print(f"  Logging lattice parameters to: {self.filename}")

    def __call__(self):
        try:
            time_ps = self.dyn.get_time() / (1000.0 * units.fs)
            cell_params = self.atoms.get_cell().cellpar()
            a, b, c, alpha, beta, gamma = cell_params

            line = f" {time_ps:11.4f} {a:10.5f} {b:10.5f} {c:10.5f} {alpha:10.5f} {beta:10.5f} {gamma:10.5f}\\n"
            self.file.write(line)
            self.file.flush()

        except Exception as e:
            print(f"  Error writing lattice frame: {e}")

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            print(f"  Closed lattice log file: {self.filename}")

class CSVLogger:
    def __init__(self, atoms, dyn, filename="md_data.csv"):
        self.atoms = atoms
        self.dyn = dyn
        self.filename = filename
        self.file = open(self.filename, 'w', encoding='utf-8')

        header = "Time[ps],a[A],b[A],c[A],alpha[deg],beta[deg],gamma[deg],Etot[eV],Epot[eV],Ekin[eV],T[K],P[GPa]\\n"
        self.file.write(header)
        self.file.flush()
        print(f"  Logging data to CSV: {self.filename}")

    def __call__(self):
        try:
            time_ps = self.dyn.get_time() / (1000.0 * units.fs)
            cell_params = self.atoms.get_cell().cellpar()
            a, b, c, alpha, beta, gamma = cell_params

            epot = self.atoms.get_potential_energy()
            ekin = self.atoms.get_kinetic_energy()
            etot = epot + ekin

            temp = self.atoms.get_temperature()
            try:
                stress = self.atoms.get_stress(voigt=True)
                pressure = -np.mean(stress[:3]) / units.GPa
            except Exception:
                pressure = np.nan

            line = (
                f"{time_ps:.4f},"
                f"{a:.5f},{b:.5f},{c:.5f},"
                f"{alpha:.5f},{beta:.5f},{gamma:.5f},"
                f"{etot:.6f},{epot:.6f},{ekin:.6f},"
                f"{temp:.2f},{pressure:.4f}\\n"
            )

            self.file.write(line)
            self.file.flush()

        except Exception as e:
            print(f"  Error writing CSV frame: {e}")

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            print(f"  Closed CSV log file: {self.filename}")

"""

    # 4. Assemble the final script string
    # --- MODIFICATION: Updated plotting function and font sizes ---
    script_content = f"""
\"\"\"
Standalone Python script for Molecular Dynamics simulation.
Generated by the uMLIP-Interactive Streamlit app.

--- SETTINGS ---
MLIP Model: {selected_model}
Model Key: {model_size}
Device: {device}
Precision: {dtype}
CPU Threads: {thread_count}

MD Parameters:
{textwrap.indent(md_params_str, '  ')}
---
\"\"\"
{imports_str}

{custom_classes_str}

md_params = {md_params_str}

def run_md_simulation(atoms, basename, calculator):
    print(f"--- Starting MD for {{basename}} ---")
    print(f"  Ensemble: {{md_params['ensemble']}}")
    print(f"  Temperature: {{md_params['temperature']}} K")
    print(f"  Steps: {{md_params['n_steps']}}")
    print(f"  Timestep: {{md_params['timestep']}} fs")
    print(f"  Console Log Interval: {{md_params['log_interval']}} steps")
    print(f"  Trajectory Save Interval: {{md_params['traj_interval']}} steps")

    atoms.calc = calculator

    try:
        MaxwellBoltzmannDistribution(atoms, temperature_K=md_params['temperature'])
        Stationary(atoms)
        print("  Initialized velocities (Maxwell-Boltzmann) and removed CoM motion.")
    except Exception as e:
        print(f"  Warning: Could not set initial velocities. {{e}}")

    dt = md_params['timestep'] * units.fs
    ensemble = md_params['ensemble']
    temperature_K = md_params['temperature']

    md = None

    if ensemble == "NVE":
        md = VelocityVerlet(atoms, timestep=dt)
        print("  Initialized NVE (VelocityVerlet) ensemble.")
    elif ensemble == "NVT-Langevin":
        friction = md_params.get('friction', 0.02) / units.fs
        md = Langevin(atoms, timestep=dt, temperature_K=temperature_K, friction=friction)
        print(f"  Initialized NVT-Langevin ensemble (friction={{friction * units.fs:.3f}} 1/ps).")
    elif ensemble == "NVT-Berendsen":
        taut = md_params.get('taut', 100.0) * units.fs
        md = NVTBerendsen(atoms, timestep=dt, temperature_K=temperature_K, taut=taut)
        print(f"  Initialized NVT-Berendsen ensemble (taut={{taut / units.fs:.1f}} fs).")
    elif ensemble == "NPT":
        pressure_gpa = md_params.get('target_pressure_gpa', 0.0)
        pressure_au = pressure_gpa * 1e9 * units.Pascal
        taut_fs = md_params.get('taut', 100.0) * units.fs
        taup_fs = md_params.get('taup', 1000.0) * units.fs

        compressibility_au = 1 / (100 * units.GPa) 

        md = NPTBerendsen(atoms, timestep=dt, temperature_K=temperature_K, pressure_au=pressure_au, taut=taut_fs, taup=taup_fs, compressibility_au=compressibility_au)
        print(f"  Initialized NPT-Berendsen (isotropic) ensemble.")
        print(f"    Target P: {{pressure_gpa}} GPa")
        print(f"    T coupling (taut): {{taut_fs / units.fs:.1f}} fs")
        print(f"    P coupling (taup): {{taup_fs / units.fs:.1f}} fs")

    if md is None:
        print(f"Error: Unknown ensemble '{{ensemble}}'")
        return

    logfile = f"md_results/md_{{basename}}.log"
    xyz_trajectory_file = f"md_results/md_{{basename}}.xyz"
    lattice_logfile = f"md_results/md_{{basename}}.lattice"
    csv_logfile = f"md_results/md_{{basename}}_data.csv"

    print(f"  Logging detailed steps to: {{logfile}}")
    md.attach(
        MDLogger(
            dyn=md, 
            atoms=atoms, 
            logfile=logfile, 
            header=True, 
            stress=True, 
            peratom=False, 
            mode='w'
        ),
        interval=md_params['log_interval']
    )

    console_logger = ConsoleMDLogger(atoms, md_params['n_steps'], md_params['log_interval'])
    md.attach(console_logger, interval=1)

    xyz_writer = XYZTrajectoryWriter(atoms, xyz_trajectory_file) 
    md.attach(xyz_writer, interval=md_params['traj_interval'])

    lattice_logger = LatticeLogger(atoms, md, lattice_logfile)
    md.attach(lattice_logger, interval=md_params['log_interval'])

    csv_logger = CSVLogger(atoms, md, csv_logfile)
    md.attach(csv_logger, interval=md_params['log_interval'])

    print(f"\\n🚀 Running simulation for {{md_params['n_steps']}} steps...")
    start_time = time.perf_counter()
    try:
        md.run(md_params['n_steps'])
    except Exception as e:
        print(f"❌ MD RUN FAILED: {{e}}")
        import traceback
        traceback.print_exc()
    finally:
        xyz_writer.close() 
        lattice_logger.close()
        csv_logger.close()

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"\\n--- MD for {{basename}} finished in {{elapsed:.2f}} seconds ---")

def generate_plots(basename):
    print(f"  Generating plots for: {{basename}}")

    plt.rcParams.update({{
        'font.size': 20,
        'axes.titlesize': 24,
        'axes.labelsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20,
        'figure.titlesize': 26
    }})

    csv_file = f"md_results/md_{{basename}}_data.csv"

    try:
        data = pd.read_csv(csv_file)
        if data.empty:
            print(f"    ... skipping, {{csv_file}} is empty.")
            return
        data = data.dropna()
        if data.empty:
            print(f"    ... skipping, no valid data to plot in {{csv_file}}.")
            return

    except FileNotFoundError:
        print(f"    ... skipping, {{csv_file}} not found.")
        return
    except Exception as e:
        print(f"    ... skipping, error reading {{csv_file}}: {{e}}")
        return

    time_ps = data['Time[ps]']
    plot_prefix = f"md_results/md_{{basename}}"

    try:
        plt.figure(figsize=(12, 8))
        plt.plot(time_ps, data['T[K]'], label='Temperature')
        plt.xlabel("Time (ps)")
        plt.ylabel("Temperature (K)")
        plt.title(f"{{basename}} - Temperature vs. Time")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{{plot_prefix}}_temperature.png")
        plt.close()
    except Exception as e:
        print(f"    ... failed to plot temperature: {{e}}")

    try:
        plt.figure(figsize=(12, 8))
        plt.plot(time_ps, data['P[GPa]'], label='Pressure')
        plt.xlabel("Time (ps)")
        plt.ylabel("Pressure (GPa)")
        plt.title(f"{{basename}} - Pressure vs. Time")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{{plot_prefix}}_pressure.png")
        plt.close()
    except Exception as e:
        print(f"    ... failed to plot pressure: {{e}}")

    try:
        plt.figure(figsize=(12, 8))
        plt.plot(time_ps, data['Etot[eV]'], label='Total Energy', linestyle='-', linewidth=2)
        plt.xlabel("Time (ps)")
        plt.ylabel("Energy (eV)")
        plt.title(f"{{basename}} - Total Energy vs. Time")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{{plot_prefix}}_total_energy.png")
        plt.close()
    except Exception as e:
        print(f"    ... failed to plot total energy: {{e}}")

    try:
        plt.figure(figsize=(12, 8))
        plt.plot(time_ps, data['Epot[eV]'], label='Potential Energy', linestyle='--')
        plt.xlabel("Time (ps)")
        plt.ylabel("Energy (eV)")
        plt.title(f"{{basename}} - Potential Energy vs. Time")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{{plot_prefix}}_potential_energy.png")
        plt.close()
    except Exception as e:
        print(f"    ... failed to plot potential energy: {{e}}")

    try:
        plt.figure(figsize=(12, 8))
        plt.plot(time_ps, data['Ekin[eV]'], label='Kinetic Energy', linestyle=':')
        plt.xlabel("Time (ps)")
        plt.ylabel("Energy (eV)")
        plt.title(f"{{basename}} - Kinetic Energy vs. Time")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{{plot_prefix}}_kinetic_energy.png")
        plt.close()
    except Exception as e:
        print(f"    ... failed to plot kinetic energy: {{e}}")

    try:
        plt.figure(figsize=(12, 8))
        plt.plot(time_ps, data['a[A]'], label='a (Å)')
        plt.plot(time_ps, data['b[A]'], label='b (Å)')
        plt.plot(time_ps, data['c[A]'], label='c (Å)')
        plt.ylabel("Lattice Length (Å)")
        plt.title(f"{{basename}} - Lattice Lengths vs. Time")
        plt.xlabel("Time (ps)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{{plot_prefix}}_lattice_lengths.png")
        plt.close()
    except Exception as e:
        print(f"    ... failed to plot lattice lengths: {{e}}")

    try:
        plt.figure(figsize=(12, 8))
        plt.plot(time_ps, data['alpha[deg]'], label='α (°)')
        plt.plot(time_ps, data['beta[deg]'], label='β (°)')
        plt.plot(time_ps, data['gamma[deg]'], label='γ (°)')
        plt.xlabel("Time (ps)")
        plt.ylabel("Lattice Angle (°)")
        plt.title(f"{{basename}} - Lattice Angles vs. Time")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{{plot_prefix}}_lattice_angles.png")
        plt.close()
    except Exception as e:
        print(f"    ... failed to plot lattice angles: {{e}}")

    print(f"  ... plots saved to md_results/md_{{basename}}_*.png")

def main():
{indented_calculator_setup} 

    if 'calculator' not in locals() or calculator is None:
        print("Calculator could not be initialized. Exiting.")
        exit()

    print("\\nSearching for structure files (*.cif, *.vasp, POSCAR*)...")
    structure_files = glob.glob("*.cif") + glob.glob("*.vasp") + glob.glob("POSCAR*")

    if not structure_files:
        print("No structure files found. Please place .cif or .vasp/POSCAR files in this directory.")
        exit()

    print(f"Found {{len(structure_files)}} structure(s): {{', '.join(structure_files)}}")

    os.makedirs("md_results", exist_ok=True)

    basenames_to_plot = []
    try:
        for f in structure_files:
            basename = os.path.splitext(os.path.basename(f))[0]
            basenames_to_plot.append(basename)

            try:
                print(f"\\n--- Reading structure from {{f}} ---")
                atoms = read(f)
                print(f"Read {{basename}}: {{atoms.get_chemical_formula()}} ({{len(atoms)}} atoms)")

                atoms.set_pbc(True)
                print(f"  ... Using provided cell with vectors (A): {{atoms.get_cell().lengths()}}")

                print("  ... Warming up calculator (JIT compilation)...")
                warmup_start = time.perf_counter()
                atoms.calc = calculator

                _ = atoms.get_potential_energy()
                print("      ... energy graph compiled.")
                _ = atoms.get_forces()
                print("      ... forces graph compiled.")

                if md_params['ensemble'] == 'NPT':
                    try:
                        _ = atoms.get_stress()
                        print("      ... stress graph compiled (for NPT).")
                    except Exception as e:
                        print(f"      ... warning: could not compile stress: {{e}}")

                warmup_end = time.perf_counter()
                print(f"  ... Calculator warmed up in {{warmup_end - warmup_start:.2f}}s")

                run_md_simulation(atoms, basename, calculator)

            except Exception as e:
                print(f"❌ FAILED: Error processing {{f}}: {{e}}")
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print(f"\\n\\n*** KeyboardInterrupt detected. Stopping simulation run. ***")
        print("Proceeding to plot generation...")

    except Exception as e:
        print(f"\\n\\n*** An unexpected error occurred in main loop: {{e}} ***")

    finally:
        print("\\n--- Generating plots for all processed simulations ---")
        if not basenames_to_plot:
            print("No simulations were started, nothing to plot.")

        for basename in sorted(list(set(basenames_to_plot))):
            generate_plots(basename)

        print("\\n--- All MD simulations and plotting complete ---")

if __name__ == "__main__":
    main()
"""
    return script_content.strip()
