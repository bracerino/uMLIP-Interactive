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
from ase.io import read, write
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
try:
    from ase.md.nose_hoover_chain import NoseHooverChainNVT
    NVT_NOSE_HOOVER_AVAILABLE = True
except ImportError:
    NVT_NOSE_HOOVER_AVAILABLE = False
try:
    from ase.md.nose_hoover_chain import MaskedMTKNPT
    MASKED_MTK_AVAILABLE = True
except ImportError:
    MASKED_MTK_AVAILABLE = False

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.logger import MDLogger 
try:
    from ase.constraints import StrainFilter
except ImportError:
    try:
        from ase.filters import StrainFilter
    except ImportError:
        StrainFilter = None
        print("Warning: StrainFilter not found in ase.constraints or ase.filters.")
from ase.constraints import FixAtoms
from ase.optimize import FIRE 
from collections import deque
import io

try:
    from nequix.calculator import NequixCalculator
except ImportError:
    pass
try:
    from nequip.model.saved_models.load_utils import load_saved_model as _nequip_load_saved_model
    from nequip.integrations.ase import NequIPCalculator
    from nequip.integrations.utils import basic_transforms, handle_chemical_species_map
except ImportError:
    pass
try:
    from deepmd.calculator import DP
except ImportError:
    pass

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
    elif selected_model.startswith(("Allegro", "NequIP")):
        calculator_setup_str = f"""
print("Setting up Allegro / NequIP calculator...")
print("First use downloads the model from nequip.net, then it is cached.")
try:
    _model = _nequip_load_saved_model("{model_size}")
    _model.eval()
    _md = _model.metadata
    _type_names = _md["type_names"]
    if isinstance(_type_names, str):
        _type_names = _type_names.split()
    calculator = NequIPCalculator(
        model=_model,
        device="{device}",
        transforms=basic_transforms(
            _md,
            float(_md["r_max"]),
            _type_names,
            handle_chemical_species_map(True, _type_names),
            neighborlist_backend="matscipy",
        ),
    )
    print(f"✅ {selected_model} initialized on {device}")
except NameError:
    print("❌ Allegro / NequIP initialization failed: nequip not found. Is nequip-allegro installed?")
    exit()
except Exception as e:
    print(f"❌ Allegro / NequIP initialization failed on {device}: {{e}}")
    exit()
"""
    elif "Nequix" in selected_model:
        # Nequix runs on JAX and picks its own device; it takes a model *name*
        # from its registry (not a path) and accepts no device argument.
        calculator_setup_str = f"""
print("Setting up Nequix calculator...")
try:
    try:
        calculator = NequixCalculator("{model_size}", use_kernel=True)
    except ImportError:
        # OpenEquivariance kernels are an optional extra; fall back to pure JAX.
        calculator = NequixCalculator("{model_size}", use_kernel=False)
        print("ℹ️ OpenEquivariance kernels unavailable, using pure-JAX path")
    print(f"✅ Nequix {model_size} initialized")
except NameError:
     print(f"❌ Nequix initialization failed: NequixCalculator class not found. Is nequix installed?")
     exit()
except Exception as e:
    print(f"❌ Nequix initialization failed: {{e}}")
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
    elif "UPET" in selected_model:
        imports_str += """
try:
    from upet.calculator import UPETCalculator
except ImportError:
    print("Error: UPET not found. Will fail if UPET model is selected.")
"""
        if model_size.endswith(".ckpt"):
            calculator_setup_str = f"""
print("Setting up custom UPET calculator...")
try:
    calculator = UPETCalculator(
        checkpoint_path="{model_size}",
        device="{device}"
    )
    print(f"✅ Custom UPET initialized on {device}")
except Exception as e:
    print(f"❌ UPET initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = UPETCalculator(
            checkpoint_path="{model_size}",
            device="cpu"
        )
        print("✅ Custom UPET initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ UPET CPU fallback failed: {{cpu_e}}")
        exit()
"""
        else:
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
            calculator_setup_str = f"""
print("Setting up UPET calculator...")
try:
    calculator = UPETCalculator(
        model="{upet_model_name}",
        version="{upet_version}",
        device="{device}"
    )
    print(f"✅ UPET {upet_model_name} v{upet_version} initialized on {device}")
except Exception as e:
    print(f"❌ UPET initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = UPETCalculator(
            model="{upet_model_name}",
            version="{upet_version}",
            device="cpu"
        )
        print("✅ UPET initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ UPET CPU fallback failed: {{cpu_e}}")
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
        # Whole-run progress (for total remaining-time estimate). When total_run_steps
        # is None the logger falls back to a per-run estimate.
        self.global_offset = 0
        self.total_run_steps = None

    def set_global_progress(self, global_offset, total_run_steps):
        \"\"\"Tell the logger how many steps are already done (global_offset) and how
        many steps the whole tensile loop will run (total_run_steps), so it can show
        the total estimated time remaining instead of just this increment's.\"\"\"
        self.global_offset = global_offset
        self.total_run_steps = total_run_steps

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
            est_label = "Est. time (this run)"
            if self.averaging_started and len(self.step_times) > 0:
                avg_step_time = np.mean(list(self.step_times))
                if self.total_run_steps is not None:
                    # Total remaining over ALL remaining increments and their MD steps.
                    remaining_steps_run = self.total_run_steps - (self.global_offset + self.step_count_this_run)
                    est_label = "Est. total remaining"
                else:
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
                log_str += f" | {est_label}: {self._format_time(estimated_remaining_time)}"
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
            forces = current_atoms.get_forces()
            cell = current_atoms.get_cell()
            lattice_str = " ".join(f"{x:.8f}" for x in cell.array.flatten())
            current_length = np.linalg.norm(cell[self.strain_direction])
            current_strain_percent = ((current_length - self.initial_length) / self.initial_length) * 100.0
            stress_voigt = self.atoms.get_stress(voigt=True)
            stress_gpa = stress_voigt[self.strain_direction] / units.GPa
            # Try to get per-atom energies (not all calculators support this)
            atom_energies = None
            try:
                atom_energies = current_atoms.get_potential_energies()
            except Exception:
                pass
            if atom_energies is not None:
                props = 'Properties=species:S:1:pos:R:3:forces:R:3:atom_energy:R:1'
            else:
                props = 'Properties=species:S:1:pos:R:3:forces:R:3'
            comment = (f'Step={global_step} Time={time_ps:.4f}ps Strain={current_strain_percent:.6f}% '
                       f'Stress={stress_gpa:.6f}GPa Energy={energy:.6f}eV Temp={temp:.2f}K '
                       f'Lattice="{lattice_str}" {props}')
            self.file.write(f"{num_atoms}\\n")
            self.file.write(f"{comment}\\n")
            for i in range(num_atoms):
                line = f"{symbols[i]} {positions[i, 0]:15.8f} {positions[i, 1]:15.8f} {positions[i, 2]:15.8f}"
                line += f" {forces[i, 0]:15.8f} {forces[i, 1]:15.8f} {forces[i, 2]:15.8f}"
                if atom_energies is not None:
                    line += f" {atom_energies[i]:15.8f}"
                self.file.write(line + "\\n")
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
        'font.size': 24,
        'font.family': 'sans-serif',
        'axes.titlesize': 28,
        'axes.labelsize': 26,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,
        'figure.titlesize': 30,
        'figure.figsize': (15, 10),
        'lines.linewidth': 3.0,
        'lines.markersize': 8,
        'axes.linewidth': 2.0,
        'xtick.major.width': 2.0,
        'ytick.major.width': 2.0,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.titlepad': 18.0,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.3,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
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

    # =========================================================================
    # Grip-mode helpers — appended to tensile_helpers
    # =========================================================================
    grip_helpers = """

# ---------------------------------------------------------------------------
# Grip-mode helpers (for finite structures in vacuum)
# ---------------------------------------------------------------------------

def identify_grip_atoms(atoms, direction, grip_fraction):
    positions = atoms.get_positions()
    coords = positions[:, direction]
    x_min = coords.min()
    x_max = coords.max()
    span = x_max - x_min
    if span < 1e-6:
        raise ValueError(f"Structure has zero extent along axis {direction}.")
    left_cutoff = x_min + grip_fraction * span
    right_cutoff = x_max - grip_fraction * span
    left_grip = np.where(coords <= left_cutoff)[0]
    right_grip = np.where(coords >= right_cutoff)[0]
    free_atoms = np.where((coords > left_cutoff) & (coords < right_cutoff))[0]
    initial_length = x_max - x_min
    return left_grip, right_grip, free_atoms, initial_length


def compute_grip_axial_force(atoms, left_grip_indices, right_grip_indices, direction):
    # Read forces directly from calculator results dict,
    # completely bypassing ASE's constraint system.
    # atoms.get_forces() can zero constrained atoms even with apply_constraint=False
    # in some ASE versions, so we go directly to the calculator's stored results.
    raw_forces = np.array(atoms.calc.results['forces'])

    # Left grip: structure pulls it in +direction  → f_left > 0 under tension
    # Right grip: structure pulls it in -direction → f_right < 0 under tension
    # Axial force = (f_left - f_right) / 2  (average from both grips, like LAMMPS tutorial)
    f_left = float(np.sum(raw_forces[left_grip_indices, direction]))
    f_right = float(np.sum(raw_forces[right_grip_indices, direction]))
    axial_force = (f_left - f_right) / 2.0

    return axial_force, f_left, f_right


class GripTensileLogger:
    def __init__(self, atoms, filename, log_frequency, direction,
                 initial_length, left_grip_indices, right_grip_indices):
        self.atoms = atoms
        self.filename = filename
        self.log_frequency = max(1, log_frequency)
        self.direction = direction
        self.initial_length = initial_length
        self.left_grip_indices = left_grip_indices
        self.right_grip_indices = right_grip_indices
        self.file = open(self.filename, 'w', encoding='utf-8')
        header = "Step,Time[ps],Strain[%],Displacement[A],Length[A],AxialForce[eV/A],F_left[eV/A],F_right[eV/A],Etot[eV],Epot[eV],Ekin[eV],T[K]\\n"
        self.file.write(header)
        self.file.flush()
        print(f"  Logging grip tensile data to CSV: {self.filename}")
        self.step_count_in_increment = 0
        self.global_step_offset = 0
        self.total_displacement = 0.0

    def set_displacement(self, displacement):
        self.total_displacement = displacement

    def __call__(self):
        self.step_count_in_increment += 1
        global_step = self.global_step_offset + self.step_count_in_increment
        if self.step_count_in_increment % self.log_frequency != 0:
            return
        try:
            time_ps = global_step * tensile_params['timestep'] / 1000.0
            strain_pct = (self.total_displacement / self.initial_length) * 100.0 if self.initial_length > 0 else 0.0
            current_length = self.initial_length + self.total_displacement
            axial_force, f_left, f_right = compute_grip_axial_force(
                self.atoms, self.left_grip_indices, self.right_grip_indices, self.direction
            )
            epot = self.atoms.get_potential_energy()
            ekin = self.atoms.get_kinetic_energy()
            etot = epot + ekin
            temp = self.atoms.get_temperature()
            line = (f"{global_step},{time_ps:.4f},{strain_pct:.6f},"
                    f"{self.total_displacement:.6f},{current_length:.6f},"
                    f"{axial_force:.6f},{f_left:.6f},{f_right:.6f},"
                    f"{etot:.6f},{epot:.6f},{ekin:.6f},{temp:.2f}\\n")
            self.file.write(line)
            self.file.flush()
        except Exception as e:
            print(f"  Error writing grip CSV at step {global_step}: {e}")

    def reset_increment_step_count(self, current_global_step=0):
        self.global_step_offset = current_global_step
        self.step_count_in_increment = 0

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            print(f"  Closed grip tensile CSV log: {self.filename}")


class GripTensileXYZWriter:
    def __init__(self, atoms, filename, traj_frequency, direction,
                 initial_length, left_grip_indices, right_grip_indices):
        self.atoms = atoms
        self.filename = filename
        self.traj_frequency = max(1, traj_frequency)
        self.direction = direction
        self.initial_length = initial_length
        self.left_grip_indices = left_grip_indices
        self.right_grip_indices = right_grip_indices
        self.file = open(self.filename, 'w', encoding='utf-8')
        self.step_count_in_increment = 0
        self.global_step_offset = 0
        self.total_displacement = 0.0
        print(f"  Writing grip trajectory to: {self.filename}")

    def set_displacement(self, displacement):
        self.total_displacement = displacement

    def __call__(self):
        self.step_count_in_increment += 1
        global_step = self.global_step_offset + self.step_count_in_increment
        if self.step_count_in_increment % self.traj_frequency != 0:
            return
        try:
            time_ps = global_step * tensile_params['timestep'] / 1000.0
            atoms = self.atoms
            positions = atoms.get_positions()
            symbols = atoms.get_chemical_symbols()
            num_atoms = len(atoms)
            energy = atoms.get_potential_energy()
            temp = atoms.get_temperature()
            forces = np.array(atoms.calc.results['forces'])  # Raw calculator forces
            cell = atoms.get_cell()
            lattice_str = " ".join(f"{x:.8f}" for x in cell.array.flatten())
            strain_pct = (self.total_displacement / self.initial_length) * 100.0 if self.initial_length > 0 else 0.0
            axial_force, _, _ = compute_grip_axial_force(atoms, self.left_grip_indices, self.right_grip_indices, self.direction)
            # Try to get per-atom energies
            atom_energies = None
            try:
                atom_energies = atoms.get_potential_energies()
            except Exception:
                pass
            if atom_energies is not None:
                props = 'Properties=species:S:1:pos:R:3:forces:R:3:atom_energy:R:1'
            else:
                props = 'Properties=species:S:1:pos:R:3:forces:R:3'
            comment = (f'Step={global_step} Time={time_ps:.4f}ps '
                       f'Strain={strain_pct:.6f}% AxialForce={axial_force:.6f}eV/A '
                       f'Energy={energy:.6f}eV Temp={temp:.2f}K '
                       f'Lattice="{lattice_str}" {props}')
            self.file.write(f"{num_atoms}\\n")
            self.file.write(f"{comment}\\n")
            for i in range(num_atoms):
                line = f"{symbols[i]} {positions[i, 0]:15.8f} {positions[i, 1]:15.8f} {positions[i, 2]:15.8f}"
                line += f" {forces[i, 0]:15.8f} {forces[i, 1]:15.8f} {forces[i, 2]:15.8f}"
                if atom_energies is not None:
                    line += f" {atom_energies[i]:15.8f}"
                self.file.write(line + "\\n")
            self.file.flush()
        except Exception as e:
            print(f"  Error writing grip XYZ at step {global_step}: {e}")

    def reset_increment_step_count(self, current_global_step=0):
        self.global_step_offset = current_global_step
        self.step_count_in_increment = 0

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            print(f"  Closed grip trajectory file: {self.filename}")


def generate_grip_tensile_plots(basename):
    print(f"  Generating grip tensile plots for: {basename}")
    plt.rcParams.update({
        'font.size': 24, 'font.family': 'sans-serif',
        'axes.titlesize': 28, 'axes.labelsize': 26,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 22, 'ytick.labelsize': 22, 'legend.fontsize': 22,
        'figure.titlesize': 30, 'figure.figsize': (15, 10),
        'lines.linewidth': 3.0, 'lines.markersize': 8,
        'axes.linewidth': 2.0,
        'xtick.major.width': 2.0, 'ytick.major.width': 2.0,
        'xtick.major.size': 8, 'ytick.major.size': 8,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'axes.titlepad': 18.0,
        'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.3,
        'legend.frameon': True, 'legend.edgecolor': 'black',
    })
    csv_file = f"md_results/{basename}_tensile_data.csv"
    plot_prefix = f"md_results/{basename}"
    try:
        data = pd.read_csv(csv_file)
        print(f"    Read {len(data)} rows from {csv_file}")
        print(f"    Columns: {list(data.columns)}")
        if data.empty or len(data) < 2:
            print(f"    ... skipping, too few data points ({len(data)} rows).")
            return
        data = data.sort_values(by='Step').drop_duplicates(subset=['Step'], keep='last')
        data = data.dropna()
        if data.empty or len(data) < 2:
            print(f"    ... skipping, no valid data after cleaning.")
            return
        print(f"    {len(data)} valid data points after cleaning")
    except FileNotFoundError:
        print(f"    ... skipping, {csv_file} not found.")
        return
    except Exception as e:
        print(f"    ... skipping: {e}")
        import traceback
        traceback.print_exc()
        return

    strain = data['Strain[%]'].values
    time_ps = data['Time[ps]'].values

    # Helper: rolling average for smoothing (like the blue curves in the LAMMPS tutorial)
    def smooth(y, window=None):
        if window is None:
            window = max(5, len(y) // 20)
        if len(y) < window:
            return y
        # Use centered rolling mean — min_periods=1 avoids edge artifacts
        # (at boundaries, averages over available points instead of padding with zeros)
        return pd.Series(y).rolling(window, center=True, min_periods=1).mean().values

    # =========================================================================
    # Plot 1: Force-Strain curve (raw + smoothed) — main result
    # =========================================================================
    if 'AxialForce[eV/A]' in data.columns:
        force = data['AxialForce[eV/A]'].values
        try:
            plt.figure(figsize=(15, 10))
            plt.plot(strain, force, '-', color='orange', alpha=0.5, linewidth=1, label='Raw data')
            plt.plot(strain, smooth(force), '-', color='blue', linewidth=2.5, label='Smoothed')
            plt.xlabel("Engineering Strain (%)")
            plt.ylabel("Axial Force (eV/Å)")
            plt.title(f"{basename} — Force vs. Strain")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"{plot_prefix}_force_strain.png", dpi=200)
            plt.close()
            print(f"    ✅ {plot_prefix}_force_strain.png")
        except Exception as e:
            print(f"    ... failed force-strain plot: {e}")

    # =========================================================================
    # Plot 2: Energy vs Time (raw + smoothed) — like LAMMPS tutorial Fig. b
    # =========================================================================
    if 'Etot[eV]' in data.columns:
        etot = data['Etot[eV]'].values
        try:
            plt.figure(figsize=(15, 10))
            plt.plot(time_ps, etot, '-', color='orange', alpha=0.5, linewidth=1, label='Raw data')
            plt.plot(time_ps, smooth(etot), '-', color='blue', linewidth=2.5, label='Smoothed')
            plt.xlabel("Time (ps)")
            plt.ylabel("Total Energy (eV)")
            plt.title(f"{basename} — Total Energy vs. Time")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"{plot_prefix}_energy_vs_time.png", dpi=200)
            plt.close()
            print(f"    ✅ {plot_prefix}_energy_vs_time.png")
        except Exception as e:
            print(f"    ... failed energy vs time: {e}")

    # =========================================================================
    # Plot 3: Length / Displacement vs Time — like LAMMPS tutorial Fig. a
    # =========================================================================
    if 'Length[A]' in data.columns:
        length = data['Length[A]'].values
        L0 = length[0] if len(length) > 0 else 1.0
        try:
            fig, ax1 = plt.subplots(figsize=(15, 10))
            color1 = 'blue'
            ax1.plot(time_ps, length, '-', color=color1, linewidth=2)
            ax1.set_xlabel("Time (ps)")
            ax1.set_ylabel("Grip-to-Grip Length (Å)", color=color1)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, linestyle='--', alpha=0.6)

            # Secondary axis: ΔL/L₀
            ax2 = ax1.twinx()
            color2 = 'red'
            delta_L_ratio = (length - L0) / L0 * 100.0
            ax2.plot(time_ps, delta_L_ratio, '--', color=color2, linewidth=1.5, alpha=0.7)
            ax2.set_ylabel("(L − L₀) / L₀  (%)", color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)

            plt.title(f"{basename} — Length vs. Time  (L₀ = {L0:.2f} Å)")
            fig.tight_layout()
            plt.savefig(f"{plot_prefix}_length_vs_time.png", dpi=200)
            plt.close()
            print(f"    ✅ {plot_prefix}_length_vs_time.png")
        except Exception as e:
            print(f"    ... failed length vs time: {e}")
    elif 'Displacement[A]' in data.columns:
        disp = data['Displacement[A]'].values
        try:
            plt.figure(figsize=(15, 10))
            plt.plot(time_ps, disp, '-', color='blue', linewidth=2)
            plt.xlabel("Time (ps)")
            plt.ylabel("Grip Displacement (Å)")
            plt.title(f"{basename} — Displacement vs. Time")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"{plot_prefix}_displacement_vs_time.png", dpi=200)
            plt.close()
            print(f"    ✅ {plot_prefix}_displacement_vs_time.png")
        except Exception as e:
            print(f"    ... failed displacement vs time: {e}")

    # =========================================================================
    # Plot 4: Temperature vs Time
    # =========================================================================
    if 'T[K]' in data.columns:
        temp = data['T[K]'].values
        try:
            plt.figure(figsize=(15, 10))
            plt.plot(time_ps, temp, '-', color='orange', alpha=0.5, linewidth=1, label='Raw data')
            plt.plot(time_ps, smooth(temp), '-', color='purple', linewidth=2.5, label='Smoothed')
            plt.xlabel("Time (ps)")
            plt.ylabel("Temperature (K)")
            plt.title(f"{basename} — Temperature vs. Time")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"{plot_prefix}_temp_vs_time.png", dpi=200)
            plt.close()
            print(f"    ✅ {plot_prefix}_temp_vs_time.png")
        except Exception as e:
            print(f"    ... failed temp vs time: {e}")

    # =========================================================================
    # Plot 5: Potential Energy vs Strain
    # =========================================================================
    if 'Epot[eV]' in data.columns:
        epot = data['Epot[eV]'].values
        try:
            plt.figure(figsize=(15, 10))
            plt.plot(strain, epot, '-', color='orange', alpha=0.5, linewidth=1, label='Raw data')
            plt.plot(strain, smooth(epot), '-', color='green', linewidth=2.5, label='Smoothed')
            plt.xlabel("Engineering Strain (%)")
            plt.ylabel("Potential Energy (eV)")
            plt.title(f"{basename} — Potential Energy vs. Strain")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"{plot_prefix}_epot_vs_strain.png", dpi=200)
            plt.close()
            print(f"    ✅ {plot_prefix}_epot_vs_strain.png")
        except Exception as e:
            print(f"    ... failed energy vs strain: {e}")

    # =========================================================================
    # Plot 6: Combined 2×2 overview
    # =========================================================================
    try:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f"{basename} — Grip Tensile Test Overview", fontsize=28, fontweight='bold')

        # (a) Length vs Time
        ax = axes[0, 0]
        if 'Length[A]' in data.columns:
            ax.plot(time_ps, data['Length[A]'].values, '-', color='blue', linewidth=2)
            ax.set_ylabel("Length (Å)")
        elif 'Displacement[A]' in data.columns:
            ax.plot(time_ps, data['Displacement[A]'].values, '-', color='blue', linewidth=2)
            ax.set_ylabel("Displacement (Å)")
        ax.set_xlabel("Time (ps)")
        ax.set_title("(a) Length vs. Time")
        ax.grid(True, linestyle='--', alpha=0.4)

        # (b) Energy vs Time
        ax = axes[0, 1]
        if 'Etot[eV]' in data.columns:
            etot = data['Etot[eV]'].values
            ax.plot(time_ps, etot, '-', color='orange', alpha=0.4, linewidth=1)
            ax.plot(time_ps, smooth(etot), '-', color='blue', linewidth=2)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Total Energy (eV)")
        ax.set_title("(b) Energy vs. Time")
        ax.grid(True, linestyle='--', alpha=0.4)

        # (c) Force vs Strain
        ax = axes[1, 0]
        if 'AxialForce[eV/A]' in data.columns:
            force = data['AxialForce[eV/A]'].values
            ax.plot(strain, force, '-', color='orange', alpha=0.4, linewidth=1)
            ax.plot(strain, smooth(force), '-', color='blue', linewidth=2)
        ax.set_xlabel("Strain (%)")
        ax.set_ylabel("Axial Force (eV/Å)")
        ax.set_title("(c) Force vs. Strain")
        ax.grid(True, linestyle='--', alpha=0.4)

        # (d) Temperature vs Time
        ax = axes[1, 1]
        if 'T[K]' in data.columns:
            temp = data['T[K]'].values
            ax.plot(time_ps, temp, '-', color='orange', alpha=0.4, linewidth=1)
            ax.plot(time_ps, smooth(temp), '-', color='purple', linewidth=2)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Temperature (K)")
        ax.set_title("(d) Temperature vs. Time")
        ax.grid(True, linestyle='--', alpha=0.4)

        for ax in axes.flat:
            ax.tick_params(labelsize=16)

        plt.tight_layout()
        plt.savefig(f"{plot_prefix}_overview.png", dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    ✅ {plot_prefix}_overview.png")
    except Exception as e:
        print(f"    ... failed overview plot: {e}")

    print(f"  ... grip tensile plots saved to md_results/{basename}_*.png")

"""

    # Combine all helpers
    tensile_helpers = tensile_helpers + grip_helpers

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

{tensile_helpers} # Includes Loggers, Plotting, ConsoleMDLogger, TensileXYZWriter, Grip helpers

tensile_params = {tensile_params_str}

def apply_strain(atoms, direction_index, strain_value):
    cell = atoms.get_cell()
    new_cell = cell.copy()
    current_vec = cell[direction_index, :]
    new_vec = current_vec * (1.0 + strain_value)
    new_cell[direction_index, :] = new_vec
    atoms.set_cell(new_cell, scale_atoms=True)


def build_nvt_dynamics(atoms, timestep_ase, grip_indices=None):
    \"\"\"Dynamics for straining: NVE (no thermostat) / Langevin / Berendsen / Nose-Hoover.
    Fixed grip atoms (grip_indices) are excluded from the thermostat: the Langevin
    thermostat uses a per-atom friction of 0 on those atoms so they are not heated.\"\"\"
    thermostat = tensile_params.get('nvt_thermostat', 'Langevin')
    temperature_K = tensile_params['temperature']
    if str(thermostat).startswith('NVE'):
        return VelocityVerlet(atoms, timestep=timestep_ase)
    if thermostat == 'Berendsen':
        taut = tensile_params.get('thermostat_taut', 100.0) * units.fs
        return NVTBerendsen(atoms, timestep=timestep_ase, temperature_K=temperature_K, taut=taut)
    if thermostat == 'Nose-Hoover' and NVT_NOSE_HOOVER_AVAILABLE:
        tdamp = tensile_params.get('thermostat_taut', 100.0) * units.fs
        return NoseHooverChainNVT(atoms, timestep=timestep_ase, temperature_K=temperature_K, tdamp=tdamp)
    friction = tensile_params['friction'] / units.fs
    if grip_indices is not None and len(grip_indices) > 0:
        friction_arr = np.full((len(atoms), 1), friction, dtype=float)
        friction_arr[np.asarray(grip_indices)] = 0.0
        friction = friction_arr
    return Langevin(atoms, timestep=timestep_ase, temperature_K=temperature_K, friction=friction)


def build_strain_dynamics(atoms, timestep_ase, direction_index, grip_indices=None):
    \"\"\"Dynamics for a strain increment: transverse-masked NPT if enabled, else NVT/NVE.\"\"\"
    # Boundary grips fix end atoms, which conflicts with barostat cell scaling -> NVT/NVE.
    if not tensile_params['use_npt_transverse'] or tensile_params.get('use_boundary_grips', False):
        return build_nvt_dynamics(atoms, timestep_ase, grip_indices=grip_indices)

    temperature_K = tensile_params['temperature']
    transverse_mask = [1, 1, 1]; transverse_mask[direction_index] = 0
    barostat = tensile_params.get('transverse_barostat', 'Inhomogeneous Berendsen')

    if barostat == 'Masked MTK' and MASKED_MTK_AVAILABLE:
        tdamp = tensile_params.get('thermostat_taut', 100.0) * units.fs
        pdamp = tensile_params.get('barostat_taup', 1000.0) * units.fs
        return MaskedMTKNPT(atoms, timestep=timestep_ase, temperature_K=temperature_K,
                            pressure_au=0.0, tdamp=tdamp, pdamp=pdamp,
                            mask=tuple(bool(m) for m in transverse_mask))

    if INHOMO_NPT_AVAILABLE:
        bulk_mod = tensile_params['bulk_modulus']
        if bulk_mod <= 0: bulk_mod = 140.0
        compressibility = 1.0 / (bulk_mod * units.GPa)
        taut = tensile_params.get('thermostat_taut', 100.0) * units.fs
        taup = tensile_params.get('barostat_taup', 1000.0) * units.fs
        return Inhomogeneous_NPTBerendsen(atoms, timestep=timestep_ase, temperature_K=temperature_K,
                                          pressure_au=0.0, taut=taut, taup=taup,
                                          compressibility_au=compressibility, mask=tuple(transverse_mask))

    # No barostat available -> NVT fallback
    return build_nvt_dynamics(atoms, timestep_ase, grip_indices=grip_indices)


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
        print(f"  Transverse Pressure Control: NPT ({{tensile_params.get('transverse_barostat', 'Inhomogeneous Berendsen')}})")
        if tensile_params.get('transverse_barostat') == 'Masked MTK' and not MASKED_MTK_AVAILABLE:
            print("  WARNING: MaskedMTKNPT not found, will fall back to Inhomogeneous Berendsen / NVT!")
        elif not INHOMO_NPT_AVAILABLE:
            print("  WARNING: Inhomogeneous_NPTBerendsen not found, will use NVT instead!")
    else:
        print(f"  Thermostat: NVT ({{tensile_params.get('nvt_thermostat', 'Langevin')}})")

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
            try:
                eq_poscar = f"md_results/{{basename}}_equilibrated.poscar"
                write(eq_poscar, atoms, format="vasp", sort=True, vasp5=True)
                print(f"  💾 Saved equilibrated structure to {{eq_poscar}} (use it as a restart point)")
            except Exception as we:
                print(f"  Warning: could not save equilibrated POSCAR: {{we}}")
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

    # --- Optional boundary grips: hold two end slabs rigid, free interior under MD ---
    use_boundary_grips = tensile_params.get('use_boundary_grips', False)
    bg_all_grip = None
    bg_left = bg_right = None
    bg_center0 = None
    bg_symmetric = tensile_params.get('boundary_grip_symmetric', True)
    if use_boundary_grips:
        grip_frac = tensile_params.get('boundary_grip_fraction', 0.08)
        bg_left, bg_right, bg_free, _ = identify_grip_atoms(atoms, direction_index, grip_frac)
        bg_all_grip = np.concatenate([bg_left, bg_right])
        _p0 = atoms.get_positions()
        bg_center0 = 0.5 * (_p0[bg_left, direction_index].mean() + _p0[bg_right, direction_index].mean())
        atoms.set_constraint(FixAtoms(indices=bg_all_grip))
        _vel = atoms.get_velocities(); _vel[bg_all_grip] = 0.0; atoms.set_velocities(_vel)
        print(f"  Boundary grips ON: {{len(bg_left)}} + {{len(bg_right)}} grip atoms fixed, {{len(bg_free)}} interior free")
        print(f"    Grip thickness per end: {{grip_frac * 100:.1f}}%  |  Pull: {{'symmetric' if bg_symmetric else 'one-sided'}}")

    strain_start_time = time.perf_counter()

    # Periodically refresh the saved plots so they can be monitored while the run is
    # ongoing. Throttled by both an increment count and a minimum wall-time interval so
    # the plotting overhead stays negligible relative to the MD cost.
    plot_refresh_every = 20
    plot_min_interval_s = 10.0
    last_plot_time = 0.0

    print(f"\\n--- Starting Strain Application Loop ---")
    traj_file_closed = False
    tensile_log_file_closed = False
    try:
        tensile_logger.reset_increment_step_count(current_global_step=global_step_counter)
        tensile_logger.step_count_in_increment = tensile_logger.log_frequency - 1
        tensile_logger() 
        traj_writer.reset_increment_step_count(current_global_step=global_step_counter)
        traj_writer.step_count_in_increment = traj_writer.traj_frequency - 1
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
                if use_boundary_grips:
                    atoms.set_constraint()  # release so the strain/shift also moves the grips
                apply_strain(atoms, direction_index, strain_step )
                if use_boundary_grips:
                    # Symmetric pull: translate so the midpoint between the two grips stays
                    # fixed, so both grip slabs move apart equally about a fixed centre.
                    if bg_symmetric:
                        _pos = atoms.get_positions()
                        _c = 0.5 * (_pos[bg_left, direction_index].mean() + _pos[bg_right, direction_index].mean())
                        _pos[:, direction_index] += (bg_center0 - _c); atoms.set_positions(_pos)
                    # Re-fix grips and zero their velocities for the MD that follows.
                    atoms.set_constraint(FixAtoms(indices=bg_all_grip))
                    _vel = atoms.get_velocities(); _vel[bg_all_grip] = 0.0; atoms.set_velocities(_vel)
                applied_strain_percent = target_strain_value * 100.0
                print(f"\\nIncrement {{i+1}}/{{n_increments}}: Applied strain up to ~{{applied_strain_percent:.4f}}%")
            else:
                 print(f"\\nIncrement {{i+1}}/{{n_increments}}: Target strain reached or step too small, running MD steps.")

            current_strain_value = (np.linalg.norm(atoms.get_cell()[direction_index]) - initial_length) / initial_length if initial_length > 0 else 0.0

            if relax_steps > 0:
                print(f"  Running {{relax_steps}} relaxation steps...")

                dyn_strain_relax = build_strain_dynamics(atoms, timestep_ase, direction_index, grip_indices=bg_all_grip)

                if console_logger_relax:
                     console_logger_relax.reset()
                     console_logger_relax.set_global_progress(global_step_counter - equilibration_steps, total_md_steps_strain)
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

                dyn_strain_main = build_strain_dynamics(atoms, timestep_ase, direction_index, grip_indices=bg_all_grip)

                console_logger_strain.reset()
                console_logger_strain.set_global_progress(global_step_counter - equilibration_steps, total_md_steps_strain)
                dyn_strain_main.attach(console_logger_strain, interval=1)
                tensile_logger.reset_increment_step_count(current_global_step=global_step_counter)
                traj_writer.reset_increment_step_count(current_global_step=global_step_counter)
                dyn_strain_main.attach(tensile_logger, interval=1)
                dyn_strain_main.attach(traj_writer, interval=1)

                dyn_strain_main.run(steps_per_increment)
                global_step_counter += steps_per_increment 
                print(f"  MD steps finished for increment {{i+1}}.")

            if ((i + 1) % plot_refresh_every == 0
                    and (time.perf_counter() - last_plot_time) > plot_min_interval_s):
                try:
                    generate_tensile_plots(basename)
                except Exception:
                    pass
                last_plot_time = time.perf_counter()

            final_length_increment = np.linalg.norm(atoms.get_cell()[direction_index])
            final_strain_increment = (final_length_increment - initial_length) / initial_length if initial_length > 0 else 0.0
            if final_strain_increment >= max_strain_target:
                 print(f"\\nMaximum strain target reached or exceeded.")
                 tensile_logger.reset_increment_step_count(current_global_step=global_step_counter)
                 tensile_logger.step_count_in_increment = tensile_logger.log_frequency - 1
                 tensile_logger()
                 traj_writer.reset_increment_step_count(current_global_step=global_step_counter)
                 traj_writer.step_count_in_increment = traj_writer.traj_frequency - 1
                 traj_writer()
                 break

    except KeyboardInterrupt:
        print(f"\\n\\n*** KeyboardInterrupt detected during strain loop. Stopping. ***")
        try:
            tensile_logger.reset_increment_step_count(current_global_step=global_step_counter)
            tensile_logger.step_count_in_increment = tensile_logger.log_frequency - 1
            tensile_logger()
            traj_writer.reset_increment_step_count(current_global_step=global_step_counter)
            traj_writer.step_count_in_increment = traj_writer.traj_frequency - 1
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


def run_grip_tensile_test_simulation(atoms, basename, calculator):
    \"\"\"Grip-based tensile test for finite structures in vacuum.\"\"\"
    print(f"--- Starting GRIP-BASED Tensile Test for {{basename}} ---")

    direction_index = tensile_params['strain_direction']
    direction_name = ['x', 'y', 'z'][direction_index]
    grip_fraction = tensile_params.get('grip_fraction', 0.1)

    atoms.calc = calculator

    left_grip, right_grip, free_atoms, initial_length = \\
        identify_grip_atoms(atoms, direction_index, grip_fraction)

    all_grip = np.concatenate([left_grip, right_grip])

    print(f"  Strain Direction: {{direction_name}}-axis")
    print(f"  Grip fraction: {{grip_fraction*100:.1f}}% per end")
    print(f"  Left grip: {{len(left_grip)}} atoms (fixed)")
    print(f"  Right grip: {{len(right_grip)}} atoms (displaced each increment)")
    print(f"  Free atoms: {{len(free_atoms)}} atoms")
    print(f"  Initial grip-to-grip length: {{initial_length:.4f}} Å")
    print(f"  Max Strain: {{tensile_params['max_strain']}} %")
    print(f"  Strain Rate: {{tensile_params['strain_rate']}} %/ps")
    print(f"  Temperature: {{tensile_params['temperature']}} K")
    print(f"  Output: Axial force (eV/Å) vs. strain")
    if tensile_params.get('symmetric_pull', True):
        print(f"  Pull mode: Symmetric (both grips move in opposite directions)")
    else:
        print(f"  Pull mode: One-sided (only right grip moves)")

    global_step_counter = 0

    try:
        MaxwellBoltzmannDistribution(atoms, temperature_K=tensile_params['temperature'])
        Stationary(atoms)
        print(f"  Set initial velocities (all atoms; grips NOT fixed during equilibration).")
    except Exception as e:
        print(f"  Warning: Could not set initial velocities: {{e}}")

    equilibration_steps = tensile_params['equilibration_steps']
    if equilibration_steps > 0:
        print(f"\\n--- Running Initial Equilibration ({{equilibration_steps}} steps, {{tensile_params.get('nvt_thermostat', 'Langevin')}}); grips NOT fixed yet ---")
        timestep_ase = tensile_params['timestep'] * units.fs
        friction_ase = tensile_params['friction'] / units.fs
        dyn_eq = build_nvt_dynamics(atoms, timestep_ase)  # whole structure free, all atoms thermostatted
        console_logger_eq = ConsoleMDLogger(atoms, equilibration_steps,
                                            log_interval=tensile_params['log_frequency'],
                                            prefix="Equil: ")
        dyn_eq.attach(console_logger_eq, interval=1)
        try:
            console_logger_eq.reset()
            dyn_eq.run(equilibration_steps)
            global_step_counter += equilibration_steps
            print(f"  Equilibration finished. T = {{atoms.get_temperature():.1f}} K")
            # Re-measure the grip-to-grip reference length on the equilibrated structure.
            _coords = atoms.get_positions()[:, direction_index]
            initial_length = float(_coords.max() - _coords.min())
            print(f"  Reference length after equilibration: {{initial_length:.4f}} Å")
            try:
                eq_poscar = f"md_results/{{basename}}_equilibrated.poscar"
                write(eq_poscar, atoms, format="vasp", sort=True, vasp5=True)
                print(f"  💾 Saved equilibrated structure to {{eq_poscar}} (use it as a restart point)")
            except Exception as we:
                print(f"  Warning: could not save equilibrated POSCAR: {{we}}")
        except Exception as e:
            print(f"❌ Equilibration FAILED: {{e}}")
            import traceback
            traceback.print_exc()

    # Now fix the grips for the straining phase (after equilibration; zero their velocities first).
    _vel = atoms.get_velocities(); _vel[all_grip] = 0.0; atoms.set_velocities(_vel)
    atoms.set_constraint(FixAtoms(indices=all_grip))

    print(f"\\n--- Preparing Grip-Based Straining ---")
    timestep_ase = tensile_params['timestep'] * units.fs
    friction_ase = tensile_params['friction'] / units.fs

    csv_logfile = f"md_results/{{basename}}_tensile_data.csv"
    xyz_logfile = f"md_results/{{basename}}_tensile_traj.xyz"

    tensile_logger = GripTensileLogger(
        atoms, csv_logfile,
        log_frequency=tensile_params['log_frequency'],
        direction=direction_index,
        initial_length=initial_length,
        left_grip_indices=left_grip,
        right_grip_indices=right_grip
    )

    traj_writer = GripTensileXYZWriter(
        atoms, xyz_logfile,
        traj_frequency=tensile_params['traj_frequency'],
        direction=direction_index,
        initial_length=initial_length,
        left_grip_indices=left_grip,
        right_grip_indices=right_grip
    )

    steps_per_increment = tensile_params['md_steps_per_increment']
    relax_steps = tensile_params['relax_steps'] if tensile_params['relax_between_strain'] else 0

    max_strain_frac = tensile_params['max_strain'] / 100.0
    strain_rate_ps = tensile_params['strain_rate'] / 100.0
    md_time_per_increment_ps = steps_per_increment * tensile_params['timestep'] / 1000.0
    strain_per_increment = strain_rate_ps * md_time_per_increment_ps
    displacement_per_increment = strain_per_increment * initial_length
    n_increments = int(np.ceil(max_strain_frac / strain_per_increment)) if strain_per_increment > 0 else 0
    total_md_steps_strain = n_increments * (steps_per_increment + relax_steps)

    print(f"  Displacement per increment: {{displacement_per_increment:.6f}} Å")
    print(f"  Strain per increment: {{strain_per_increment*100:.6f}} %")
    print(f"  Approximate number of increments: {{n_increments}}")

    total_displacement = 0.0
    strain_start_time = time.perf_counter()

    console_logger_strain = ConsoleMDLogger(
        atoms, steps_per_increment,
        log_interval=tensile_params['log_frequency'],
        prefix="Strain MD: "
    )

    print(f"\\n--- Starting Grip Strain Application Loop ---")
    tensile_log_closed = False
    traj_closed = False

    # Throttled periodic plot refresh (see cell-scaling loop for rationale).
    plot_refresh_every = 20
    plot_min_interval_s = 10.0
    last_plot_time = 0.0

    try:
        tensile_logger.set_displacement(0.0)
        tensile_logger.reset_increment_step_count(current_global_step=global_step_counter)
        tensile_logger.step_count_in_increment = tensile_logger.log_frequency - 1
        tensile_logger()

        traj_writer.set_displacement(0.0)
        traj_writer.reset_increment_step_count(current_global_step=global_step_counter)
        traj_writer.step_count_in_increment = traj_writer.traj_frequency - 1
        traj_writer()

        for i in range(n_increments):
            # 1. Remove constraints
            atoms.set_constraint()

            # 2. Displace grips
            positions = atoms.get_positions()
            symmetric_pull = tensile_params.get('symmetric_pull', True)
            if symmetric_pull:
                # Both grips move in opposite directions (like LAMMPS cnt_top/cnt_bot)
                half_disp = displacement_per_increment / 2.0
                positions[right_grip, direction_index] += half_disp
                positions[left_grip, direction_index] -= half_disp
            else:
                # Only right grip moves, left grip stays fixed
                positions[right_grip, direction_index] += displacement_per_increment
            atoms.set_positions(positions)
            total_displacement += displacement_per_increment

            current_strain_pct = (total_displacement / initial_length) * 100.0
            pull_mode = "symmetric" if symmetric_pull else "one-sided"
            print(f"\\nIncrement {{i+1}}/{{n_increments}}: Δ={{displacement_per_increment:.4f}} Å ({{pull_mode}}), strain ~{{current_strain_pct:.4f}}%")

            # 3. Fix both grips
            atoms.set_constraint(FixAtoms(indices=all_grip))

            # 4. Zero grip velocities
            velocities = atoms.get_velocities()
            velocities[all_grip] = 0.0
            atoms.set_velocities(velocities)

            tensile_logger.set_displacement(total_displacement)
            traj_writer.set_displacement(total_displacement)

            # 5. Relaxation (optional)
            if relax_steps > 0:
                print(f"  Running {{relax_steps}} relaxation steps...")
                dyn_relax = build_nvt_dynamics(atoms, timestep_ase, grip_indices=all_grip)
                tensile_logger.reset_increment_step_count(current_global_step=global_step_counter)
                traj_writer.reset_increment_step_count(current_global_step=global_step_counter)
                dyn_relax.attach(tensile_logger, interval=1)
                dyn_relax.attach(traj_writer, interval=1)
                dyn_relax.run(relax_steps)
                global_step_counter += relax_steps

            # 6. Main MD steps
            if steps_per_increment > 0:
                print(f"  Running {{steps_per_increment}} MD steps...")
                dyn_main = build_nvt_dynamics(atoms, timestep_ase, grip_indices=all_grip)
                console_logger_strain.reset()
                console_logger_strain.set_global_progress(global_step_counter - equilibration_steps, total_md_steps_strain)
                dyn_main.attach(console_logger_strain, interval=1)
                tensile_logger.reset_increment_step_count(current_global_step=global_step_counter)
                traj_writer.reset_increment_step_count(current_global_step=global_step_counter)
                dyn_main.attach(tensile_logger, interval=1)
                dyn_main.attach(traj_writer, interval=1)
                dyn_main.run(steps_per_increment)
                global_step_counter += steps_per_increment

            if ((i + 1) % plot_refresh_every == 0
                    and (time.perf_counter() - last_plot_time) > plot_min_interval_s):
                try:
                    generate_grip_tensile_plots(basename)
                except Exception:
                    pass
                last_plot_time = time.perf_counter()

            if total_displacement / initial_length >= max_strain_frac:
                print(f"\\nMaximum strain target reached.")
                tensile_logger.reset_increment_step_count(current_global_step=global_step_counter)
                tensile_logger.step_count_in_increment = tensile_logger.log_frequency - 1
                tensile_logger()
                traj_writer.reset_increment_step_count(current_global_step=global_step_counter)
                traj_writer.step_count_in_increment = traj_writer.traj_frequency - 1
                traj_writer()
                break

    except KeyboardInterrupt:
        print(f"\\n*** KeyboardInterrupt — stopping grip tensile test. ***")
    except Exception as e:
        print(f"❌ Grip Tensile Test FAILED: {{e}}")
        import traceback
        traceback.print_exc()
    finally:
        if not tensile_log_closed:
            tensile_logger.close()
            tensile_log_closed = True
        if not traj_closed:
            traj_writer.close()
            traj_closed = True

    elapsed = time.perf_counter() - strain_start_time
    print(f"\\n--- Grip Tensile Test Loop finished in {{elapsed:.2f}} seconds ---")


def main():
{indented_calculator_setup} 

    if 'calculator' not in locals() or calculator is None:
        print("Calculator could not be initialized. Exiting.")
        exit()

    print("\\nSearching for structure files (*.cif, *.vasp, *.poscar, POSCAR*)...")
    structure_files = (glob.glob("*.cif") + glob.glob("*.vasp")
                       + glob.glob("*.poscar") + glob.glob("*.POSCAR")
                       + glob.glob("POSCAR*"))
    # Drop duplicates (e.g. a file matched by both *.POSCAR and POSCAR*) while keeping order
    structure_files = list(dict.fromkeys(structure_files))

    if not structure_files:
        print("No structure files found. Please place .cif, .vasp/.poscar or POSCAR files in this directory.")
        exit()

    if len(structure_files) > 1:
        print(f"Warning: Found multiple structure files. Tensile test will run only on the first one found: {{structure_files[0]}}")

    f = structure_files[0] 

    basename = os.path.splitext(os.path.basename(f))[0]
    os.makedirs("md_results", exist_ok=True)

    basenames_to_plot = [basename]
    try:
        print(f"\\n--- Reading structure from {{f}} ---")
        # ASE cannot guess the VASP format from a ".poscar" extension, so set it explicitly.
        try:
            if f.lower().endswith(".poscar"):
                atoms_initial = read(f, format="vasp")
            else:
                atoms_initial = read(f)
        except Exception as read_err:
            if (f.lower().endswith((".poscar", ".vasp"))
                    or os.path.basename(f).upper().startswith("POSCAR")):
                print(f"  ⚠️ Could not read '{{f}}' as a VASP/POSCAR file: {{read_err}}")
                print(f"  ⚠️ The file may have an unsupported VASP format. A common cause is")
                print(f"     placeholder species names like 'Type_1 Type_2' (written by tools")
                print(f"     such as OVITO) instead of real element symbols. Edit the POSCAR")
                print(f"     species line to use real elements (e.g. 'Fe Ni') and try again.")
            raise
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

        # Choose simulation mode
        if tensile_params.get('use_grip_mode', False):
            print("  Mode: GRIP-BASED (finite structure in vacuum)")
            run_grip_tensile_test_simulation(atoms_initial.copy(), basename, calculator)
        else:
            print("  Mode: CELL-SCALING (periodic structure)")
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
            if tensile_params.get('use_grip_mode', False):
                generate_grip_tensile_plots(basename)
            else:
                generate_tensile_plots(basename)

        print("\\n--- Tensile Test and plotting complete ---")

if __name__ == "__main__":
    main()
"""
    return script_content.strip()
