import pprint
import textwrap
from collections import deque
from ase import units
import numpy as np


def generate_tensile_test_python_script(tensile_params, selected_model, model_size, device, dtype, thread_count):

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
try:
    from ase.md.npt import NPT as NPTNoseHoover 
    NPT_NH_AVAILABLE = True
except ImportError:
    NPT_NH_AVAILABLE = False
try:
    from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
    INHOMO_NPT_AVAILABLE = True
except ImportError:
    INHOMO_NPT_AVAILABLE = False
    print("Warning: Inhomogeneous_NPTBerendsen not found. Transverse NPT option will fallback to NVT.")

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.logger import MDLogger 
from ase.constraints import StrainFilter 
from ase.optimize import FIRE 
from collections import deque
import io

try:
    from nequix.ase_calculator import NequixCalculator
except ImportError:
    print("Warning: Nequix (from atomicarchitects) not found. Will fail if Nequix model is selected.")
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
    elif "Nequix" in selected_model:
        calculator_setup_str = f"""
print("Setting up Nequix calculator...")
try:
    calculator = NequixCalculator(
        model_path="{model_size}",
        device="{device}"
    )
    print(f"✅ Nequix {model_size} initialized on {device}")
except NameError:
     print(f"❌ Nequix initialization failed: NequixCalculator class not found. Is nequix installed?")
     exit()
except Exception as e:
    print(f"❌ Nequix initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = NequixCalculator(
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
except NameError:
     print(f"❌ DeePMD initialization failed: DP class not found. Is deepmd-kit installed?")
     exit()
except Exception as e:
    print(f"❌ DeePMD initialization failed: {{e}}")
    exit()
"""
    else:
        calculator_setup_str = "print('Error: Could not determine calculator type.')\ncalculator = None\nexit()"


    tensile_params_str = pprint.pformat(tensile_params, indent=4, width=80)
    indented_calculator_setup = textwrap.indent(calculator_setup_str, "    ")

    tensile_helpers = """

class ConsoleMDLogger:
    def __init__(self, atoms, total_steps_in_run, log_interval=10, steps_for_avg=10, prefix=""):
        self.atoms = atoms
        self.total_steps_in_run = total_steps_in_run
        self.log_interval = log_interval
        self.steps_for_avg = max(1, steps_for_avg)
        self.prefix = prefix
        self.reset()

    def reset(self):
        self.step_times = deque(maxlen=self.steps_for_avg)
        self.step_count_this_run = 0
        self.averaging_started = False
        self.start_time = time.perf_counter()
        self.last_log_time = time.perf_counter()
        self.step_start_time = time.perf_counter()

    def __call__(self):
        self.step_count_this_run += 1
        current_time = time.perf_counter()
        step_duration = current_time - self.step_start_time

        if self.step_count_this_run > self.log_interval:
            self.step_times.append(step_duration)
            self.averaging_started = True

        self.step_start_time = current_time

        if self.step_count_this_run % self.log_interval == 0:
            elapsed_time_run = current_time - self.start_time
            avg_step_time = 0
            estimated_remaining_time = None
            if self.averaging_started and len(self.step_times) > 0:
                avg_step_time = np.mean(list(self.step_times))
                remaining_steps_run = self.total_steps_in_run - self.step_count_this_run
                if remaining_steps_run > 0:
                    estimated_remaining_time = remaining_steps_run * avg_step_time
            try:
                epot = self.atoms.get_potential_energy()
                ekin = self.atoms.get_kinetic_energy()
                temp = self.atoms.get_temperature()
                try:
                    stress = self.atoms.get_stress(voigt=True)
                    pressure = -np.mean(stress[:3]) / units.GPa
                except Exception:
                    pressure = None
                try:
                    cell_params = self.atoms.get_cell().cellpar()
                    a, b, c, alpha, beta, gamma = cell_params
                    cell_str = f", Cell: a={a:.3f} b={b:.3f} c={c:.3f} α={alpha:.2f} β={beta:.2f} γ={gamma:.2f}"
                except Exception:
                    cell_str = ""
            except Exception as e:
                print(f"  {self.prefix}Warning: Error getting properties at step {self.step_count_this_run}: {e}")
                return
            log_str = f"  {self.prefix}Step {self.step_count_this_run}/{self.total_steps_in_run}: "
            log_str += f"T = {temp:.1f} K, "
            log_str += f"E_pot = {epot:.4f} eV, "
            log_str += f"E_kin = {ekin:.4f} eV"
            if pressure is not None:
                log_str += f", P = {pressure:.2f} GPa"
            log_str += cell_str
            if estimated_remaining_time is not None:
                log_str += f" | Avg/step: {avg_step_time:.3f}s"
                log_str += f" | Est. time (this run): {self._format_time(estimated_remaining_time)}"
            elif self.step_count_this_run <= self.log_interval:
                log_str += " | (Calculating time estimates...)"
            print(log_str)
            self.last_log_time = current_time

    def _format_time(self, seconds):
        if seconds is None or seconds < 0: return "N/A"
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"


class TensileXYZWriter:
    def __init__(self, atoms, filename="tensile_traj.xyz", traj_frequency=100, strain_direction=0, initial_length=1.0):
        self.atoms = atoms
        self.filename = filename
        self.traj_frequency = max(1, traj_frequency)
        self.strain_direction = strain_direction
        self.initial_length = initial_length if initial_length > 0 else 1.0
        self.file = open(self.filename, 'w', encoding='utf-8')
        self.step_count_in_increment = 0
        self.global_step_offset = 0
        print(f"  Writing tensile trajectory to: {self.filename} (will overwrite)")

    def __call__(self):
        self.step_count_in_increment += 1
        global_step = self.global_step_offset + self.step_count_in_increment
        if self.step_count_in_increment % self.traj_frequency != 0:
             return
        try:
            time_ps = global_step * tensile_params['timestep'] / 1000.0
            current_atoms = self.atoms
            positions = current_atoms.get_positions()
            symbols = current_atoms.get_chemical_symbols()
            num_atoms = len(current_atoms)
            energy = current_atoms.get_potential_energy()
            temp = current_atoms.get_temperature()
            cell = current_atoms.get_cell()
            lattice_str = " ".join(f"{x:.8f}" for x in cell.array.flatten())
            current_length = np.linalg.norm(cell[self.strain_direction])
            current_strain_percent = ((current_length - self.initial_length) / self.initial_length) * 100.0
            stress_voigt = self.atoms.get_stress(voigt=True)
            stress_gpa = stress_voigt[self.strain_direction] / units.GPa
            comment = (f'Step={global_step} Time={time_ps:.4f}ps Strain={current_strain_percent:.6f}% '
                       f'Stress={stress_gpa:.6f}GPa Energy={energy:.6f}eV Temp={temp:.2f}K '
                       f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3')
            self.file.write(f"{num_atoms}\\n")
            self.file.write(f"{comment}\\n")
            for i in range(num_atoms):
                self.file.write(f"{symbols[i]} {positions[i, 0]:15.8f} {positions[i, 1]:15.8f} {positions[i, 2]:15.8f}\\n")
            self.file.flush()
        except Exception as e:
            print(f"  Error writing tensile XYZ frame around step {global_step}: {e}")

    def reset_increment_step_count(self, current_global_step=0):
        self.global_step_offset = current_global_step
        self.step_count_in_increment = 0

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            print(f"  Closed tensile trajectory file: {self.filename}")


class TensileLogger:
    def __init__(self, atoms, filename="tensile_data.csv", log_frequency=10, strain_direction=0, initial_length=1.0):
        self.atoms = atoms
        self.filename = filename
        self.log_frequency = max(1, log_frequency)
        self.strain_direction = strain_direction
        self.initial_length = initial_length if initial_length > 0 else 1.0
        self.file = open(self.filename, 'w', encoding='utf-8')
        header = "Step,Time[ps],Strain[%],Stress[GPa],Etot[eV],Epot[eV],Ekin[eV],T[K],a[A],b[A],c[A],alpha[deg],beta[deg],gamma[deg]\\n"
        self.file.write(header)
        self.file.flush()
        print(f"  Logging tensile data to CSV: {self.filename}")
        self.step_count_in_increment = 0
        self.global_step_offset = 0

    def __call__(self):
        self.step_count_in_increment += 1
        global_step = self.global_step_offset + self.step_count_in_increment
        if self.step_count_in_increment % self.log_frequency != 0:
             return
        try:
            time_ps = global_step * tensile_params['timestep'] / 1000.0
            current_cell = self.atoms.get_cell()
            current_length = np.linalg.norm(current_cell[self.strain_direction])
            current_strain_percent = ((current_length - self.initial_length) / self.initial_length) * 100.0
            stress_voigt = self.atoms.get_stress(voigt=True)
            stress_gpa = stress_voigt[self.strain_direction] / units.GPa
            epot = self.atoms.get_potential_energy()
            ekin = self.atoms.get_kinetic_energy()
            etot = epot + ekin
            temp = self.atoms.get_temperature()
            cell_params = self.atoms.get_cell().cellpar()
            a, b, c, alpha, beta, gamma = cell_params
            line = (
                f"{global_step},"
                f"{time_ps:.4f},"
                f"{current_strain_percent:.6f},"
                f"{stress_gpa:.6f},"
                f"{etot:.6f},{epot:.6f},{ekin:.6f},"
                f"{temp:.2f},"
                f"{a:.5f},{b:.5f},{c:.5f},"
                f"{alpha:.5f},{beta:.5f},{gamma:.5f}\\n"
            )
            self.file.write(line)
            self.file.flush()
        except Exception as e:
             print(f"  Error writing tensile CSV frame around step {global_step}: {e}")

    def reset_increment_step_count(self, current_global_step=0):
        self.global_step_offset = current_global_step
        self.step_count_in_increment = 0

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            print(f"  Closed tensile CSV log file: {self.filename}")

def generate_tensile_plots(basename):
    print(f"  Generating tensile plots for: {basename}")

    plt.rcParams.update({
        'font.size': 22,
        'axes.titlesize': 26,
        'axes.labelsize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 22,
        'figure.titlesize': 28,
        'figure.figsize': (15, 10)
    })

    csv_file = f"md_results/{basename}_tensile_data.csv"
    plot_prefix = f"md_results/{basename}"

    try:
        data = pd.read_csv(csv_file)
        if data.empty or len(data) < 2:
            print(f"    ... skipping, {csv_file} is empty or has too few data points.")
            return

        data = data.sort_values(by='Step').drop_duplicates(subset=['Step'], keep='last')
        if 'Strain[%]' not in data.columns or data['Strain[%]'].nunique() < 2:
             print(f"    ... skipping, not enough unique strain points after sorting/dropping duplicates.")
             return

        data = data.dropna()
        if data.empty or len(data) < 2:
            print(f"    ... skipping, no valid data rows left after dropna in {csv_file}.")
            return

    except FileNotFoundError:
        print(f"    ... skipping, {csv_file} not found.")
        return
    except Exception as e:
        print(f"    ... skipping, error reading or processing {csv_file}: {e}")
        import traceback
        traceback.print_exc()
        return

    strain = data['Strain[%]']
    stress = data['Stress[GPa]']
    time_ps = data['Time[ps]']

    try:
        data['strain_group'] = data['Strain[%]'].round(decimals=4)

        grouped = data.groupby('strain_group')['Stress[GPa]']
        mean_stress = grouped.mean()
        std_stress = grouped.std()

        strain_points = mean_stress.index.values

        plt.figure()
        plt.plot(strain_points, mean_stress, marker='o', linestyle='-', markersize=6, label='Mean Stress')

        plt.fill_between(strain_points, 
                         mean_stress - std_stress, 
                         mean_stress + std_stress, 
                         color='blue', alpha=0.2, label='± 1 Std Dev')

        plt.xlabel("Engineering Strain (%)")
        plt.ylabel("Engineering Stress (GPa)")
        plt.title(f"{basename} - Stress vs. Strain (Averaged)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_stress_strain_avg.png")
        plt.close()
    except Exception as e:
        print(f"    ... failed to plot averaged stress-strain curve: {e}")

    try:
        plt.figure()
        plt.plot(strain, stress, marker='.', linestyle='-', markersize=2, alpha=0.5)
        plt.xlabel("Engineering Strain (%)")
        plt.ylabel("Engineering Stress (GPa)")
        plt.title(f"{basename} - Stress vs. Strain (Raw Data)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_stress_strain_raw.png")
        plt.close()
    except Exception as e:
        print(f"    ... failed to plot raw stress-strain curve: {e}")


    try:
        plt.figure()
        plt.plot(time_ps, data['T[K]'], marker='.', linestyle='-', markersize=4)
        plt.xlabel("Time (ps)")
        plt.ylabel("Temperature (K)")
        plt.title(f"{basename} - Temperature vs. Time")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_temp_vs_time.png")
        plt.close()
    except Exception as e:
        print(f"    ... failed to plot temperature vs time: {e}")

    try:
        plt.figure()
        plt.plot(time_ps, data['Epot[eV]'], marker='.', linestyle='-', markersize=4, label='Potential Energy')
        plt.xlabel("Time (ps)")
        plt.ylabel("Potential Energy (eV)")
        plt.title(f"{basename} - Potential Energy vs. Time")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_epot_vs_time.png")
        plt.close()
    except Exception as e:
        print(f"    ... failed to plot energy vs time: {e}")

    try:
        plt.figure()
        plt.plot(time_ps, data['a[A]'], label='a (Å)')
        plt.plot(time_ps, data['b[A]'], label='b (Å)')
        plt.plot(time_ps, data['c[A]'], label='c (Å)')
        plt.ylabel("Lattice Length (Å)")
        plt.title(f"{basename} - Lattice Lengths vs. Time")
        plt.xlabel("Time (ps)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_lengths_vs_time.png")
        plt.close()
    except Exception as e:
        print(f"    ... failed to plot lattice lengths vs time: {e}")

    try:
        angle_cols = ['alpha[deg]', 'beta[deg]', 'gamma[deg]']
        angle_variation = data[angle_cols].std().sum()
        if angle_variation > 0.1: 
            plt.figure()
            plt.plot(time_ps, data['alpha[deg]'], label='α (°)')
            plt.plot(time_ps, data['beta[deg]'], label='β (°)')
            plt.plot(time_ps, data['gamma[deg]'], label='γ (°)')
            plt.xlabel("Time (ps)")
            plt.ylabel("Lattice Angle (°)")
            plt.title(f"{basename} - Lattice Angles vs. Time")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"{plot_prefix}_angles_vs_time.png")
            plt.close()
        else:
             print("    ... skipping angle plot, angles remained constant.")
    except Exception as e:
        print(f"    ... failed to plot lattice angles vs time: {e}")

    print(f"  ... tensile plots saved to md_results/{basename}_*.png")

"""

    script_content = f"""
\"\"\"
Standalone Python script for Virtual Tensile Test simulation.
Generated by the uMLIP-Interactive Streamlit app.
--- SETTINGS ---
MLIP Model: {selected_model}
Model Key: {model_size}
Device: {device}
Precision: {dtype}
CPU Threads: {thread_count}
Tensile Parameters:
{textwrap.indent(tensile_params_str, '  ')}
---
\"\"\"
{imports_str}

{tensile_helpers} # Includes Loggers, Plotting, ConsoleMDLogger, TensileXYZWriter

tensile_params = {tensile_params_str}

def apply_strain(atoms, direction_index, strain_value):
    cell = atoms.get_cell()
    new_cell = cell.copy()
    current_vec = cell[direction_index, :]
    new_vec = current_vec * (1.0 + strain_value)
    new_cell[direction_index, :] = new_vec
    atoms.set_cell(new_cell, scale_atoms=True)

def run_tensile_test_simulation(atoms, basename, calculator):
    print(f"--- Starting Tensile Test for {{basename}} ---")
    direction_index = tensile_params['strain_direction']
    direction_vector = ['x', 'y', 'z'][direction_index]
    print(f"  Strain Direction: {{direction_vector}}-axis")
    print(f"  Max Strain: {{tensile_params['max_strain']}} %")
    print(f"  Strain Rate: {{tensile_params['strain_rate']}} %/ps")
    print(f"  Temperature: {{tensile_params['temperature']}} K")
    print(f"  Timestep: {{tensile_params['timestep']}} fs")
    print(f"  MD Steps per Increment: {{tensile_params['md_steps_per_increment']}}")
    print(f"  Log Frequency: {{tensile_params['log_frequency']}} steps")
    print(f"  Trajectory Frequency: {{tensile_params['traj_frequency']}} steps")
    if tensile_params['relax_between_strain']:
        print(f"  Relaxation between increments: Yes ({{tensile_params['relax_steps']}} steps)")
    else:
        print(f"  Relaxation between increments: No")
    if tensile_params['use_npt_transverse']:
        print(f"  Transverse Pressure Control: NPT (Berendsen)")
        if not INHOMO_NPT_AVAILABLE:
            print("  WARNING: Inhomogeneous_NPTBerendsen not found, will use NVT instead!")
    else:
        print(f"  Transverse Pressure Control: NVT (Langevin)")

    atoms.calc = calculator
    global_step_counter = 0

    equilibration_steps = tensile_params['equilibration_steps']
    if equilibration_steps > 0:
        print(f"\\n--- Running Initial Equilibration ({{equilibration_steps}} steps) ---")
        print(f"  Target T = {{tensile_params['temperature']}} K, Target P = 0 GPa (Isotropic)")

        eq_timestep = tensile_params['timestep'] * units.fs
        eq_taut = 100.0 * units.fs
        eq_taup = 1000.0 * units.fs
        eq_bulk_mod = tensile_params['bulk_modulus']
        if eq_bulk_mod <= 0: eq_bulk_mod = 140.0
        eq_compressibility = 1.0 / (eq_bulk_mod * units.GPa)

        try:
             MaxwellBoltzmannDistribution(atoms, temperature_K=tensile_params['temperature'])
             Stationary(atoms)
             print(f"  Set initial velocities for equilibration.")
        except Exception as e:
             print(f"  Warning: Could not set initial velocities for equilibration: {{e}}")

        dyn_eq = NPTBerendsen(atoms,
                              timestep=eq_timestep,
                              temperature_K=tensile_params['temperature'],
                              pressure_au=0.0 * units.Pascal,
                              taut=eq_taut,
                              taup=eq_taup,
                              compressibility_au=eq_compressibility)

        console_logger_eq = ConsoleMDLogger(atoms, equilibration_steps, 
                                            log_interval=tensile_params['log_frequency'],
                                            prefix="Equil: ")
        dyn_eq.attach(console_logger_eq, interval=1)

        eq_logfile = f"md_results/{{basename}}_equilibration.log"
        dyn_eq.attach(MDLogger(dyn=dyn_eq, atoms=atoms, logfile=eq_logfile, mode='w', header=True, stress=True), 
                      interval=tensile_params['log_frequency'])
        print(f"  Equilibration log being written to {{eq_logfile}}")

        try:
            console_logger_eq.reset()
            dyn_eq.run(equilibration_steps)
            global_step_counter += equilibration_steps
            print(f"  Equilibration finished. Final T = {{atoms.get_temperature():.1f}} K")
        except Exception as e:
            print(f"❌ Equilibration FAILED: {{e}}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                dyn_eq._loggers = []
            except AttributeError:
                pass 
    else:
        try:
            MaxwellBoltzmannDistribution(atoms, temperature_K=tensile_params['temperature'])
            Stationary(atoms)
            print(f"  Set initial velocities (Maxwell-Boltzmann, T={{tensile_params['temperature']}} K).")
        except Exception as e:
            print(f"  Warning: Could not set initial velocities: {{e}}")

    print(f"\\n--- Preparing Main Dynamics for Straining ---")
    timestep_ase = tensile_params['timestep'] * units.fs
    friction_ase = tensile_params['friction'] / units.fs
    temperature_K = tensile_params['temperature']


    csv_logfile = f"md_results/{{basename}}_tensile_data.csv"
    xyz_logfile = f"md_results/{{basename}}_tensile_traj.xyz"
    initial_cell = atoms.get_cell().copy()
    initial_length = np.linalg.norm(initial_cell[direction_index])

    tensile_logger = TensileLogger(atoms, csv_logfile,
                                   log_frequency=tensile_params['log_frequency'],
                                   strain_direction=direction_index,
                                   initial_length=initial_length)

    traj_writer = TensileXYZWriter(atoms, xyz_logfile,
                                   traj_frequency=tensile_params['traj_frequency'],
                                   strain_direction=direction_index,
                                   initial_length=initial_length)

    steps_per_increment = tensile_params['md_steps_per_increment']
    console_logger_strain = ConsoleMDLogger(atoms, 
                                            steps_per_increment, 
                                            log_interval=tensile_params['log_frequency'],
                                            prefix="Strain MD: ") 

    console_logger_relax = None
    relax_steps = tensile_params['relax_steps'] if tensile_params['relax_between_strain'] else 0
    if relax_steps > 0:
        console_logger_relax = ConsoleMDLogger(atoms, 
                                               relax_steps, 
                                               log_interval=tensile_params['log_frequency'],
                                               prefix="Relax: ")

    current_strain_value = 0.0 
    max_strain_target = tensile_params['max_strain'] / 100.0
    strain_rate_ps = tensile_params['strain_rate'] / 100.0 
    md_time_per_increment_ps = steps_per_increment * tensile_params['timestep'] / 1000.0
    strain_increment = strain_rate_ps * md_time_per_increment_ps 
    n_increments = int(np.ceil(max_strain_target / strain_increment)) if strain_increment > 0 else 0

    total_md_steps_strain = n_increments * (steps_per_increment + relax_steps)
    total_md_steps_all = equilibration_steps + total_md_steps_strain
    total_sim_time_ps = total_md_steps_all * tensile_params['timestep'] / 1000.0
    print(f"  Strain increment per block: {{strain_increment * 100:.6f}} %")
    print(f"  Approximate number of increments: {{n_increments}}")
    print(f"  Estimated total MD steps (equilibration + strain loop): {{total_md_steps_all:,}} (approx. {{total_sim_time_ps:.1f}} ps)")

    strain_start_time = time.perf_counter()

    print(f"\\n--- Starting Strain Application Loop ---")
    traj_file_closed = False 
    tensile_log_file_closed = False
    try:
        tensile_logger.reset_increment_step_count(current_global_step=global_step_counter)
        tensile_logger.step_count_in_increment = tensile_logger.log_frequency
        tensile_logger() 
        traj_writer.reset_increment_step_count(current_global_step=global_step_counter)
        traj_writer.step_count_in_increment = traj_writer.traj_frequency
        traj_writer()

        for i in range(n_increments): 

            target_strain_value = min((i + 1) * strain_increment, max_strain_target) 

            current_cell = atoms.get_cell()
            current_length = np.linalg.norm(current_cell[direction_index])
            current_strain_value = (current_length - initial_length) / initial_length if initial_length > 0 else 0.0

            target_length = initial_length * (1.0 + target_strain_value)
            length_increment_needed = target_length - current_length
            if current_length > 1e-6:
                 strain_step = length_increment_needed / current_length 
            else:
                 strain_step = 0.0

            if strain_step > 1e-9: 
                apply_strain(atoms, direction_index, strain_step ) 
                applied_strain_percent = target_strain_value * 100.0 
                print(f"\\nIncrement {{i+1}}/{{n_increments}}: Applied strain up to ~{{applied_strain_percent:.4f}}%")
            else:
                 print(f"\\nIncrement {{i+1}}/{{n_increments}}: Target strain reached or step too small, running MD steps.")

            current_strain_value = (np.linalg.norm(atoms.get_cell()[direction_index]) - initial_length) / initial_length if initial_length > 0 else 0.0

            if relax_steps > 0:
                print(f"  Running {{relax_steps}} relaxation steps...")

                dyn_strain_relax = None
                if tensile_params['use_npt_transverse'] and INHOMO_NPT_AVAILABLE:
                    transverse_mask = [1, 1, 1]; transverse_mask[direction_index] = 0
                    bulk_mod = tensile_params['bulk_modulus']; 
                    if bulk_mod <= 0: bulk_mod = 140.0
                    compressibility = 1.0 / (bulk_mod * units.GPa)
                    dyn_strain_relax = Inhomogeneous_NPTBerendsen(atoms, timestep=timestep_ase, temperature_K=temperature_K,
                                                            pressure_au=0.0, taut=100.0 * units.fs, taup=1000.0 * units.fs,
                                                            compressibility_au=compressibility, mask=tuple(transverse_mask))
                else:
                    dyn_strain_relax = Langevin(atoms, timestep=timestep_ase, temperature_K=temperature_K, friction=friction_ase)

                if console_logger_relax: 
                     console_logger_relax.reset()
                     dyn_strain_relax.attach(console_logger_relax, interval=1)
                tensile_logger.reset_increment_step_count(current_global_step=global_step_counter) 
                traj_writer.reset_increment_step_count(current_global_step=global_step_counter) 
                dyn_strain_relax.attach(tensile_logger, interval=1)
                dyn_strain_relax.attach(traj_writer, interval=1)    

                dyn_strain_relax.run(relax_steps)
                global_step_counter += relax_steps 
                print(f"  Relaxation finished.")


            if steps_per_increment > 0:
                print(f"  Running {{steps_per_increment}} MD steps...")

                dyn_strain_main = None
                if tensile_params['use_npt_transverse'] and INHOMO_NPT_AVAILABLE:
                    transverse_mask = [1, 1, 1]; transverse_mask[direction_index] = 0
                    bulk_mod = tensile_params['bulk_modulus']; 
                    if bulk_mod <= 0: bulk_mod = 140.0
                    compressibility = 1.0 / (bulk_mod * units.GPa)
                    dyn_strain_main = Inhomogeneous_NPTBerendsen(atoms, timestep=timestep_ase, temperature_K=temperature_K,
                                                            pressure_au=0.0, taut=100.0 * units.fs, taup=1000.0 * units.fs,
                                                            compressibility_au=compressibility, mask=tuple(transverse_mask))
                else:
                    dyn_strain_main = Langevin(atoms, timestep=timestep_ase, temperature_K=temperature_K, friction=friction_ase)

                console_logger_strain.reset()
                dyn_strain_main.attach(console_logger_strain, interval=1)
                tensile_logger.reset_increment_step_count(current_global_step=global_step_counter) 
                traj_writer.reset_increment_step_count(current_global_step=global_step_counter) 
                dyn_strain_main.attach(tensile_logger, interval=1)
                dyn_strain_main.attach(traj_writer, interval=1)

                dyn_strain_main.run(steps_per_increment) 
                global_step_counter += steps_per_increment 
                print(f"  MD steps finished for increment {{i+1}}.")

            final_length_increment = np.linalg.norm(atoms.get_cell()[direction_index])
            final_strain_increment = (final_length_increment - initial_length) / initial_length if initial_length > 0 else 0.0
            if final_strain_increment >= max_strain_target:
                 print(f"\\nMaximum strain target reached or exceeded.")
                 tensile_logger.reset_increment_step_count(current_global_step=global_step_counter)
                 tensile_logger.step_count_in_increment = tensile_logger.log_frequency
                 tensile_logger()
                 traj_writer.reset_increment_step_count(current_global_step=global_step_counter)
                 traj_writer.step_count_in_increment = traj_writer.traj_frequency
                 traj_writer()
                 break

    except KeyboardInterrupt:
        print(f"\\n\\n*** KeyboardInterrupt detected during strain loop. Stopping. ***")
        try:
            tensile_logger.reset_increment_step_count(current_global_step=global_step_counter)
            tensile_logger.step_count_in_increment = tensile_logger.log_frequency
            tensile_logger()
            traj_writer.reset_increment_step_count(current_global_step=global_step_counter)
            traj_writer.step_count_in_increment = traj_writer.traj_frequency
            traj_writer()
        except: pass
    except Exception as e:
        print(f"❌ Tensile Test FAILED during strain loop: {{e}}")
        import traceback
        traceback.print_exc()
    finally:
        if 'tensile_logger' in locals() and tensile_logger.file and not tensile_log_file_closed:
            tensile_logger.close()
            tensile_log_file_closed = True
        if 'traj_writer' in locals() and traj_writer.file and not traj_file_closed:
            traj_writer.close()
            traj_file_closed = True 

    strain_end_time = time.perf_counter()
    elapsed = strain_end_time - strain_start_time
    print(f"\\n--- Tensile Test Loop finished in {{elapsed:.2f}} seconds ---")


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

    if len(structure_files) > 1:
        print(f"Warning: Found multiple structure files. Tensile test will run only on the first one found: {{structure_files[0]}}")

    f = structure_files[0] 

    basename = os.path.splitext(os.path.basename(f))[0]
    os.makedirs("md_results", exist_ok=True)

    basenames_to_plot = [basename]
    try:
        print(f"\\n--- Reading structure from {{f}} ---")
        atoms_initial = read(f)
        print(f"Read {{basename}}: {{atoms_initial.get_chemical_formula()}} ({{len(atoms_initial)}} atoms)")

        atoms_initial.set_pbc(True) 

        print("  ... Warming up calculator ...")
        warmup_start = time.perf_counter()
        atoms_initial.calc = calculator 

        _ = atoms_initial.get_potential_energy()
        _ = atoms_initial.get_forces()
        _ = atoms_initial.get_stress()

        warmup_end = time.perf_counter()
        print(f"  ... Calculator warmed up in {{warmup_end - warmup_start:.2f}}s")

        run_tensile_test_simulation(atoms_initial.copy(), basename, calculator)

    except KeyboardInterrupt:
        print(f"\\n\\n*** KeyboardInterrupt detected during setup/warmup. Stopping. ***")
        print("Proceeding to plot generation...")

    except Exception as e:
        print(f"\\n\\n*** An unexpected error occurred: {{e}} ***")
        import traceback
        traceback.print_exc()

    finally:
        print("\\n--- Generating tensile plots ---")
        if not basenames_to_plot:
            print("No simulations were started, nothing to plot.")

        if 'basename' not in locals() and basenames_to_plot:
             basename = basenames_to_plot[0]

        if 'basename' in locals():
            generate_tensile_plots(basename) 

        print("\\n--- Tensile Test and plotting complete ---")

if __name__ == "__main__":
    main()
"""
    return script_content.strip()
