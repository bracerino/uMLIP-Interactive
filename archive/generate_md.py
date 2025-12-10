import pprint
import textwrap
from collections import deque
import copy  # Import copy


def generate_md_python_script(md_params, selected_model, model_size, device, dtype, thread_count,
                              mace_head=None, mace_dispersion=False, mace_dispersion_xc="pbe"):
    if md_params.get('use_fairchem'):
        actual_selected_model = "Fairchem (UMA Override)"
        actual_model_size = md_params.get('fairchem_model_name', 'Unknown Fairchem Model')
        add_fairchem_imports = True
    else:
        actual_selected_model = selected_model
        actual_model_size = model_size
        add_fairchem_imports = "Fairchem" in selected_model

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
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.logger import MDLogger
from collections import deque
import io

try:
    from nequix.calculator import NequixCalculator
except ImportError:
    print("Warning: Nequix (from atomicarchitects) not found. Will fail if Nequix model is selected.")
try:
    from deepmd.calculator import DP
except ImportError:
    print("Warning: DeePMD-kit not found. Will fail if DeePMD model is selected.")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    print("Warning: GPUtil not found. Cannot report GPU memory usage.")
    GPUTIL_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("Warning: psutil not found. Cannot report RAM usage. (pip install psutil)")
    PSUTIL_AVAILABLE = False

os.environ['OMP_NUM_THREADS'] = '{thread_count}'
torch.set_num_threads({thread_count})
"""

    if add_fairchem_imports:
        imports_str += """
try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
except ImportError:
    print("Error: fairchem-core not found. Please install with: pip install fairchem-core")
    exit()
"""

    if "Fairchem" in actual_selected_model:
        fairchem_model_name = md_params.get('fairchem_model_name', 'MISSING_FAIRCHEM_MODEL_NAME')
        calculator_setup_str = f"""
print("Setting up Fairchem/UMA calculator...")

fairchem_key = md_params.get('fairchem_key', None)
if fairchem_key:
    print("  Setting HF_TOKEN (Hugging Face) environment variable for UMA models...")
    os.environ['HF_TOKEN'] = str(fairchem_key)
else:
    print("  Warning: 'fairchem_key' not found in md_params. This may be required for gated UMA models.")
    print("  Proceeding without it (OK if token is already set globally).")

fairchem_task = md_params.get('fairchem_task', None)
if not fairchem_task:
    print("❌ ERROR: 'fairchem_task' not found in md_params.")
    print("  To use Fairchem UMA models, please add `fairchem_task` (e.g., 'oc20', 'omat', 'omol') to md_params.")
    exit()

calc_device = "{device}" # Use the top-level device setting passed to the function

try:
    print(f"  Loading predictor for model: {fairchem_model_name} on device: {{calc_device}}")
    # Use the specific fairchem_model_name from md_params here
    predictor = pretrained_mlip.get_predict_unit("{fairchem_model_name}", device=calc_device)

    print(f"  Initializing FAIRChemCalculator with task: {{fairchem_task}}")
    calculator = FAIRChemCalculator(predictor, task_name=fairchem_task)

    print(f"✅ Fairchem/UMA {fairchem_model_name} (task: {{fairchem_task}}) initialized on {{calc_device}}")

except Exception as e:
    print(f"❌ Fairchem/UMA initialization failed on {{calc_device}}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        predictor_cpu = pretrained_mlip.get_predict_unit("{fairchem_model_name}", device="cpu")
        calculator = FAIRChemCalculator(predictor_cpu, task_name=fairchem_task)
        print("✅ Fairchem/UMA initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ Fairchem/UMA CPU fallback failed: {{cpu_e}}")
        exit()
"""

    elif "CHGNet" in actual_selected_model:
        imports_str += """
try:
    actual_model_size_str = "{actual_model_size}"
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    chgnet_version = actual_model_size_str.split("-")[1]
except ImportError:
    print("Error: CHGNet not found. Please install with: pip install chgnet")
    chgnet_version = actual_model_size_str
    exit()
"""
        calculator_setup_str = f"""
print("Setting up CHGNet calculator...")

# Parse version from model size string (e.g., "CHGNet-0.3.0" -> "0.3.0")
actual_model_size_str = "{actual_model_size}"
try:
    chgnet_version = actual_model_size_str.split("-")[1]
except IndexError:
    print(f"  Warning: Could not parse CHGNet version from '{{actual_model_size_str}}'. Using full string.")
    chgnet_version = actual_model_size_str

print(f"  CHGNet version: {{chgnet_version}}")
print(f"  Device: {device}")
print("  Note: CHGNet requires float32 precision")

original_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.float32)
try:
    # Use the parsed 'chgnet_version' variable here
    chgnet = CHGNet.load(model_name=chgnet_version, use_device="{device}", verbose=False) 
    calculator = CHGNetCalculator(model=chgnet, use_device="{device}")
    torch.set_default_dtype(original_dtype)
    # Use the 'chgnet_version' variable in the success message
    print(f"✅ CHGNet {{chgnet_version}} initialized on {device}")
except Exception as e:
    print(f"❌ CHGNet initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        # Also use 'chgnet_version' in the fallback
        chgnet = CHGNet.load(model_name=chgnet_version, use_device="cpu", verbose=False)
        calculator = CHGNetCalculator(model=chgnet, use_device="cpu")
        torch.set_default_dtype(original_dtype)
        print(f"✅ CHGNet {{chgnet_version}} initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ CHGNet CPU fallback failed: {{cpu_e}}")
        exit()
"""


    elif "PET-MAD" in actual_selected_model:
        imports_str += """
try:
    from pet_mad.calculator import PETMADCalculator
except ImportError:
    print("Error: PET-MAD not found. Please install with: pip install pet-mad")
    exit()
"""
        calculator_setup_str = f"""
print("Setting up PET-MAD calculator...")
try:
    calculator = PETMADCalculator(
        version="v1.0.2",
        device="{device}"
    )
    print(f"✅ PET-MAD v1.0.2 initialized on {device}")
    print("   Trained on MAD dataset (95,595 structures)")
except Exception as e:
    print(f"❌ PET-MAD initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = PETMADCalculator(
            version="v1.0.2",
            device="cpu"
        )
        print("✅ PET-MAD initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ PET-MAD CPU fallback failed: {{cpu_e}}")
        exit()
"""
    elif "MACE-OFF" in actual_selected_model:
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
        model="{actual_model_size}", default_dtype="{dtype}", device="{device}"
    )
    print(f"✅ MACE-OFF {actual_model_size} initialized on {device}")
except Exception as e:
    print(f"❌ MACE-OFF initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = mace_off(
            model="{actual_model_size}", default_dtype="{dtype}", device="cpu"
        )
        print("✅ MACE-OFF initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ MACE-OFF CPU fallback failed: {{cpu_e}}")
        exit()
"""

    elif "MACE" in actual_selected_model:  # Catch MACE after MACE-OFF
        imports_str += """
try:
    from mace.calculators import mace_mp
except ImportError:
    print("Error: MACE not found. Please install with: pip install mace-torch")
    exit()
"""
        # Check if this is a URL-based foundation model
        is_url_model = actual_model_size.startswith("http://") or actual_model_size.startswith("https://")

        if is_url_model:
            # URL-based foundation model with download support
            calculator_setup_str = f"""
print("Setting up MACE foundation model from URL...")

def download_mace_model(model_url):
    \"\"\"Download MACE model from URL and cache it.\"\"\"
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
    model_url = "{actual_model_size}"
    local_model_path = download_mace_model(model_url)
    print(f"📁 Model path: {{local_model_path}}")

    print(f"⚙️  Device: {device}")
    print(f"⚙️  Dtype: {dtype}")"""

            if mace_head:
                calculator_setup_str += f"""
    print(f"🎯 Head: {mace_head}")"""

            if mace_dispersion:
                calculator_setup_str += f"""
    print(f"🔬 Dispersion: D3-{mace_dispersion_xc}")"""

            # Build calculator arguments
            calc_args = [
                f'model=local_model_path',
                f'device="{device}"',
                f'default_dtype="{dtype}"'
            ]

            if mace_head:
                calc_args.append(f'head="{mace_head}"')

            if mace_dispersion:
                calc_args.append(f'dispersion=True')
                calc_args.append(f'dispersion_xc="{mace_dispersion_xc}"')

            calc_args_str = ',\n        '.join(calc_args)

            calculator_setup_str += f"""

    calculator = mace_mp(
        {calc_args_str}
    )
    print(f"✅ MACE foundation model initialized on {device}")

except Exception as e:
    print(f"❌ MACE initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = mace_mp(
            {calc_args_str.replace('device="' + device + '"', 'device="cpu"')}
        )
        print("✅ MACE initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ MACE CPU fallback failed: {{cpu_e}}")
        exit()
"""

        else:
            # Standard MACE-MP model
            calc_args = [
                f'model="{actual_model_size}"',
                f'device="{device}"',
                f'default_dtype="{dtype}"'
            ]

            if mace_dispersion:
                calc_args.append(f'dispersion=True')
                calc_args.append(f'dispersion_xc="{mace_dispersion_xc}"')
            else:
                calc_args.append(f'dispersion=False')

            calc_args_str = ',\n        '.join(calc_args)

            calculator_setup_str = f"""
print("Setting up MACE calculator...")
try:
    calculator = mace_mp(
        {calc_args_str}
    )
    print(f"✅ MACE {actual_model_size} initialized on {device}")
except Exception as e:
    print(f"❌ MACE initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = mace_mp(
            {calc_args_str.replace('device="' + device + '"', 'device="cpu"')}
        )
        print("✅ MACE initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ MACE CPU fallback failed: {{cpu_e}}")
        exit()
"""

    elif "SevenNet" in actual_selected_model:
        imports_str += """
try:
    torch.serialization.add_safe_globals([slice])
except AttributeError:
    print("  ... running on older torch version, add_safe_globals not needed.")
    pass

try:
    from sevenn.calculator import SevenNetCalculator
except ImportError:
    print("Error: SevenNet not found. Please install with: pip install sevenn")
    exit()
"""
        calculator_setup_str = f"""
print("Setting up SevenNet calculator...")
original_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.float32)
try:
    calculator = SevenNetCalculator(model="{actual_model_size}", device="{device}")
    torch.set_default_dtype(original_dtype)
    print(f"✅ SevenNet {actual_model_size} initialized on {device}")
except Exception as e:
    print(f"❌ SevenNet initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = SevenNetCalculator(model="{actual_model_size}", device="cpu")
        torch.set_default_dtype(original_dtype)
        print("✅ SevenNet initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ SevenNet CPU fallback failed: {{cpu_e}}")
        exit()
"""
    elif "ORB" in actual_selected_model:
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
    model_func = getattr(pretrained, "{actual_model_size}")
    orbff = model_func(device="{device}", precision=precision)
    calculator = ORBCalculator(orbff, device="{device}")
    print(f"✅ ORB {actual_model_size} initialized on {device}")
except Exception as e:
    print(f"❌ ORB initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        model_func = getattr(pretrained, "{actual_model_size}")
        orbff = model_func(device="cpu", precision=precision)
        calculator = ORBCalculator(orbff, device="cpu")
        print("✅ ORB initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ ORB CPU fallback failed: {{cpu_e}}")
        exit()
"""
    elif "Nequix" in actual_selected_model:
        calculator_setup_str = f"""
print("Setting up Nequix calculator...")
try:
    calculator = NequixCalculator(
       "{actual_model_size}",
        #device="{device}"
    )
    print(f"✅ Nequix {actual_model_size} initialized on {device}")
except NameError:
   print(f"❌ Nequix initialization failed: NequixCalculator class not found. Is nequix installed?")
   exit()
except Exception as e:
    print(f"❌ Nequix initialization failed on {device}: {{e}}")
    print("Attempting fallback to CPU...")
    try:
        calculator = NequixCalculator(
            "{actual_model_size}",
            device="cpu"
        )
        print("✅ Nequix initialized on CPU (fallback)")
    except Exception as cpu_e:
        print(f"❌ Nequix CPU fallback failed: {{cpu_e}}")
        exit()
"""
    elif "DeePMD" in actual_selected_model:
        calculator_setup_str = f"""
print("Setting up DeePMD calculator...")
try:
    calculator = DP(model="{actual_model_size}")
    print(f"✅ DeePMD {actual_model_size} initialized")
except NameError:
   print(f"❌ DeePMD initialization failed: DP class not found. Is deepmd-kit installed?")
   exit()
except Exception as e:
    print(f"❌ DeePMD initialization failed: {{e}}")
    exit()
"""

    else:
        calculator_setup_str = f"print('Error: Could not determine calculator type for {actual_selected_model}.')\ncalculator = None\nexit()"

    md_params_for_header = copy.deepcopy(md_params)
    md_params_for_header.pop('use_fairchem', None)
    md_params_for_header.pop('fairchem_key', None)
    md_params_for_header.pop('fairchem_task', None)
    md_params_for_header.pop('fairchem_model_name', None)
    taup_val_header = md_params_for_header.get('taup', 1000.0)
    if 'pressure_damping_time' in md_params_for_header:
        taup_val_header = md_params_for_header.pop('pressure_damping_time')
    elif 'taup' in md_params_for_header:
        taup_val_header = md_params_for_header.pop('taup')
    md_params_for_header['taup'] = taup_val_header
    md_params_header_str = pprint.pformat(md_params_for_header, indent=4, width=80)

    taup_val_body = md_params.get('taup', 1000.0)
    if 'pressure_damping_time' in md_params:
        taup_val_body = md_params.get('pressure_damping_time')
    elif 'taup' in md_params:
        taup_val_body = md_params.get('taup')
    md_params['taup'] = taup_val_body
    md_params.pop('pressure_damping_time', None)

    md_params_script_str = pprint.pformat(md_params, indent=4, width=80)

    indented_calculator_setup = textwrap.indent(calculator_setup_str, "    ")

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

        self.device_str = "cpu"
        self.is_cuda = False
        self.device_id = 0
        try:
            if hasattr(atoms, 'calc') and atoms.calc is not None:
                calc_device = None
                if hasattr(atoms.calc, 'device'): # MACE, Nequix, ORB
                    calc_device = str(atoms.calc.device)
                elif hasattr(atoms.calc, 'use_device'): # CHGNet
                    calc_device = str(atoms.calc.use_device)
                elif hasattr(atoms.calc, 'predictor'): # FAIRChemCalculator
                    calc_device = str(atoms.calc.predictor.device)

                if calc_device:
                    self.device_str = calc_device
                    self.is_cuda = "cuda" in self.device_str
                    if self.is_cuda and ":" in self.device_str:
                        try:
                            self.device_id = int(self.device_str.split(':')[-1])
                        except (ValueError, TypeError):
                            self.device_id = 0 # default
        except Exception:
            pass # Best effort, default to no GPU logging

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

            if GPUTIL_AVAILABLE and self.is_cuda:
                try:
                    gpus = GPUtil.getGPUs()
                    if self.device_id < len(gpus):
                        gpu = gpus[self.device_id]
                        log_str += f" | GPU Mem: {gpu.memoryUsed:.0f} MB"
                except Exception:
                    pass # Fail silently if GPU read fails mid-run
            elif PSUTIL_AVAILABLE and (not self.is_cuda or self.device_str == "cpu"):
                try:
                    mem = psutil.virtual_memory()
                    log_str += f" | RAM Used: {mem.percent:.1f}%"
                except Exception:
                    pass # Fail silently if RAM read fails mid-run

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
            forces = current_atoms.get_forces()  # Add this line to get forces
            symbols = current_atoms.get_chemical_symbols()
            num_atoms = len(current_atoms)
    
            energy = current_atoms.get_potential_energy()
            temp = current_atoms.get_temperature()
            cell = current_atoms.get_cell()
            lattice_str = " ".join(f"{x:.8f}" for x in cell.array.flatten())
    
            comment = (f'Step={self.step_count} Time={self.step_count * md_params["timestep"]:.3f}fs '
                       f'Energy={energy:.6f}eV Temp={temp:.2f}K '
                       f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3:forces:R:3')
    
            self.file.write(f"{num_atoms}\\n")
            self.file.write(f"{comment}\\n")
    
            for i in range(num_atoms):
                self.file.write(f"{symbols[i]} {positions[i, 0]:15.8f} {positions[i, 1]:15.8f} {positions[i, 2]:15.8f} "
                              f"{forces[i, 0]:15.8f} {forces[i, 1]:15.8f} {forces[i, 2]:15.8f}\\n")
    
            self.file.flush()
    
            self.step_count += md_params.get('traj_interval', 1)
    
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
    mace_header_lines = ""
    if "MACE" in actual_selected_model and mace_head:
        mace_header_lines += f"\nMACE Head: {mace_head}"
    if "MACE" in actual_selected_model and mace_dispersion:
        mace_header_lines += f"\nMACE Dispersion: D3-{mace_dispersion_xc}"
    script_content = f"""
\"\"\"
Standalone Python script for Molecular Dynamics simulation.
Generated by the uMLIP-Interactive Streamlit app.

--- SETTINGS ---
MLIP Model: {actual_selected_model}{mace_header_lines}
Model Key: {actual_model_size}
Device: {device}
Precision: {dtype}
CPU Threads: {thread_count}

MD Parameters:
{textwrap.indent(md_params_header_str, '  ')}
---
\"\"\"
{imports_str}

{custom_classes_str}

md_params = {md_params_script_str}

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
        if md_params.get('remove_com_motion', True):
            Stationary(atoms)
            print("  Initialized velocities (Maxwell-Boltzmann) and removed CoM motion.")
        else:
            print("  Initialized velocities (Maxwell-Boltzmann).")
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
        taut_fs = md_params.get('taut', 100.0) * units.fs
        md = NVTBerendsen(atoms, timestep=dt, temperature_K=temperature_K, taut=taut_fs)
        print(f"  Initialized NVT-Berendsen ensemble (taut={{taut_fs / units.fs:.1f}} fs).")

    elif ensemble == "NPT (Berendsen)":
        taut_fs = md_params.get('taut', 100.0) * units.fs
        taup_fs = md_params.get('taup', 1000.0) * units.fs 

        bulk_modulus_gpa = md_params.get('bulk_modulus', 140.0)
        if bulk_modulus_gpa <= 0:
            print("  Warning: Bulk modulus must be positive. Using default 140 GPa.")
            bulk_modulus_gpa = 140.0
        compressibility_au = 1.0 / (bulk_modulus_gpa * units.GPa) 
        print(f"    Using Bulk Modulus: {{bulk_modulus_gpa:.1f}} GPa -> Compressibility: {{compressibility_au * units.GPa:.4e}} 1/GPa")

        coupling_type = md_params.get('pressure_coupling_type', 'isotropic').lower()
        pressure_gpa = md_params.get('target_pressure_gpa', 0.0)

        npt_kwargs = {{}} 

        if coupling_type == 'anisotropic':
            px_gpa = md_params.get('pressure_x', pressure_gpa)
            py_gpa = md_params.get('pressure_y', pressure_gpa)
            pz_gpa = md_params.get('pressure_z', pressure_gpa)

            pressure_factor = 1e9 * units.Pascal
            px_au = px_gpa * pressure_factor
            py_au = py_gpa * pressure_factor
            pz_au = pz_gpa * pressure_factor

            pressure_au = np.diag([px_au, py_au, pz_au])

            print(f"  Initialized NPT-Berendsen (ANISOTROPIC) ensemble.")
            print(f"    Target P (xx,yy,zz): {{px_gpa:.2f}}, {{py_gpa:.2f}}, {{pz_gpa:.2f}} GPa")
            npt_kwargs['pressure_au'] = pressure_au

        else: # Isotropic case
            pressure_au = pressure_gpa * 1e9 * units.Pascal

            print(f"  Initialized NPT-Berendsen (ISOTROPIC) ensemble.")
            print(f"    Target P: {{pressure_gpa}} GPa")
            npt_kwargs['pressure_au'] = pressure_au

        npt_kwargs['timestep'] = dt
        npt_kwargs['temperature_K'] = temperature_K
        npt_kwargs['taut'] = taut_fs
        npt_kwargs['taup'] = taup_fs
        npt_kwargs['compressibility_au'] = compressibility_au

        md = NPTBerendsen(atoms, **npt_kwargs) 

        print(f"    T coupling (taut): {{taut_fs / units.fs:.1f}} fs")
        print(f"    P coupling (taup): {{taup_fs / units.fs:.1f}} fs")

    elif ensemble == "NPT (Nose-Hoover)":
        if not NPT_NH_AVAILABLE:
             print("\\n*** ERROR: ase.md.npt.NPT (Nose-Hoover) not found. Please check your ASE installation. ***")
             sys.exit(1)

        ttime_fs = md_params.get('taut', 100.0) * units.fs
        taup_fs_val = md_params.get('taup', 1000.0) * units.fs 

        bulk_modulus_gpa = md_params.get('bulk_modulus', 140.0)
        if bulk_modulus_gpa <= 0:
            print("  Warning: Bulk modulus must be positive. Using default 140 GPa.")
            bulk_modulus_gpa = 140.0
        pfactor = (taup_fs_val**2) * bulk_modulus_gpa * units.GPa
        print(f"    Using taup={{md_params.get('taup', 1000.0)}} fs, Bulk Modulus={{bulk_modulus_gpa:.1f}} GPa -> pfactor={{pfactor / ((units.fs**2)*units.GPa):.2e}} GPa*fs^2")

        coupling_type = md_params.get('pressure_coupling_type', 'isotropic').lower()
        pressure_gpa = md_params.get('target_pressure_gpa', 0.0)
        fix_angles = md_params.get('fix_angles', True) 
        mask = None

        npt_nh_kwargs = {{}}

        if coupling_type == 'anisotropic':
            px_gpa = md_params.get('pressure_x', pressure_gpa)
            py_gpa = md_params.get('pressure_y', pressure_gpa)
            pz_gpa = md_params.get('pressure_z', pressure_gpa)

            stress_factor = units.GPa 
            px_stress = px_gpa * stress_factor
            py_stress = py_gpa * stress_factor
            pz_stress = pz_gpa * stress_factor

            externalstress = np.array([px_stress, py_stress, pz_stress, 0.0, 0.0, 0.0])

            print(f"  Initialized NPT-Nose-Hoover (ANISOTROPIC) ensemble.")
            print(f"    Target P (xx,yy,zz): {{px_gpa:.2f}}, {{py_gpa:.2f}}, {{pz_gpa:.2f}} GPa")
            npt_nh_kwargs['externalstress'] = externalstress
            if fix_angles:
                 mask = np.diag([1, 1, 1])
                 print("    Angles fixed (only diagonal strain allowed).")
            else:
                 mask = None 
                 print("    Angles variable (all strain components allowed).")
            npt_nh_kwargs['mask'] = mask


        elif coupling_type == 'directional':
            couple_x = md_params.get('couple_x', True)
            couple_y = md_params.get('couple_y', True)
            couple_z = md_params.get('couple_z', True)

            # Use 3x3 diagonal matrix mask to fix angles
            if fix_angles:
                mask = np.diag([int(couple_x), int(couple_y), int(couple_z)])
                print(f"  Initialized NPT-Nose-Hoover (DIRECTIONAL with fixed angles) ensemble.")
            else:
                mask = [int(couple_x), int(couple_y), int(couple_z)] 
                print(f"  Initialized NPT-Nose-Hoover (DIRECTIONAL with variable angles) ensemble.")

            externalstress = pressure_gpa * units.GPa 

            coupled_dirs = []
            if couple_x: coupled_dirs.append('X')
            if couple_y: coupled_dirs.append('Y')
            if couple_z: coupled_dirs.append('Z')

            print(f"    Coupling directions: {{', '.join(coupled_dirs)}}")
            print(f"    Target P: {{pressure_gpa}} GPa (applied hydrostatically to coupled directions)")

            npt_nh_kwargs['externalstress'] = externalstress
            npt_nh_kwargs['mask'] = mask

        else: # Isotropic case (default)
            externalstress = pressure_gpa * units.GPa

            print(f"  Initialized NPT-Nose-Hoover (ISOTROPIC) ensemble.")
            print(f"    Target P: {{pressure_gpa}} GPa")
            npt_nh_kwargs['externalstress'] = externalstress
             # Determine mask based on fix_angles for isotropic
            if fix_angles:
                 mask = np.diag([1, 1, 1]) # Fix shear components
                 print("    Angles fixed (only diagonal strain allowed).")
            else:
                 mask = None # Standard isotropic scaling
                 print("    Angles variable (standard isotropic scaling).")
            npt_nh_kwargs['mask'] = mask

        npt_nh_kwargs['timestep'] = dt
        npt_nh_kwargs['temperature_K'] = temperature_K
        npt_nh_kwargs['ttime'] = ttime_fs
        npt_nh_kwargs['pfactor'] = pfactor

        md = NPTNoseHoover(atoms, **npt_nh_kwargs)

        print(f"    T coupling (ttime): {{ttime_fs / units.fs:.1f}} fs")
        print(f"    P coupling used pfactor (calculated from taup & bulk modulus)")

    if md is None:
        print(f"Error: Unknown ensemble specified: '{{ensemble}}'")
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
    # -- CLeaning the GPU RAM during the run
    def clear_torch_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    md.attach(clear_torch_cache, interval=10) 
    # -- End of GPU cleaning
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

        final_vasp_file = f"md_results/md_{{basename}}_final.vasp"
        try:
            print(f"  Saving final structure to: {{final_vasp_file}}")
            write(final_vasp_file, atoms, format='vasp')
        except Exception as e:
            print(f"  Warning: Could not save final VASP file: {{e}}")

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"\\n--- MD for {{basename}} finished in {{elapsed:.2f}} seconds ---")

def generate_plots(basename):
    print(f"  Generating plots for: {{basename}}")

    plt.rcParams.update({{
        'font.size': 22,
        'axes.titlesize': 26,
        'axes.labelsize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 22,
        'figure.titlesize': 28,
        'figure.figsize': (15, 10)
    }})

    csv_file = f"md_results/md_{{basename}}_data.csv"

    try:
        data = pd.read_csv(csv_file)
        if data.empty or len(data) < 1:
            print(f"    ... skipping, {{csv_file}} is empty or has no data.")
            return

        initial_vals = data.iloc[0]
        a_0 = initial_vals['a[A]']
        b_0 = initial_vals['b[A]']
        c_0 = initial_vals['c[A]']
        alpha_0 = initial_vals['alpha[deg]']
        beta_0 = initial_vals['beta[deg]']
        gamma_0 = initial_vals['gamma[deg]']

        data['a_perc'] = ((data['a[A]'] - a_0) / a_0) * 100 if a_0 != 0 else 0
        data['b_perc'] = ((data['b[A]'] - b_0) / b_0) * 100 if b_0 != 0 else 0
        data['c_perc'] = ((data['c[A]'] - c_0) / c_0) * 100 if c_0 != 0 else 0
        data['alpha_perc'] = ((data['alpha[deg]'] - alpha_0) / alpha_0) * 100 if alpha_0 != 0 else 0
        data['beta_perc'] = ((data['beta[deg]'] - beta_0) / beta_0) * 100 if beta_0 != 0 else 0
        data['gamma_perc'] = ((data['gamma[deg]'] - gamma_0) / gamma_0) * 100 if gamma_0 != 0 else 0

        data = data.dropna()
        if data.empty:
            print(f"    ... skipping, no valid data rows left after dropna in {{csv_file}}.")
            return

    except FileNotFoundError:
        print(f"    ... skipping, {{csv_file}} not found.")
        return
    except Exception as e:
        print(f"    ... skipping, error reading or processing {{csv_file}}: {{e}}")
        import traceback
        traceback.print_exc()
        return

    time_ps = data['Time[ps]']
    plot_prefix = f"md_results/md_{{basename}}"

    try:
        plt.figure()
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
        plt.figure()
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
        plt.figure()
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
        plt.figure()
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
        plt.figure()
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
        plt.figure()
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
        plt.figure()
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

    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 12))

        ax1.plot(time_ps, data['a_perc'], label='a (%)')
        ax1.plot(time_ps, data['b_perc'], label='b (%)')
        ax1.plot(time_ps, data['c_perc'], label='c (%)')
        ax1.set_ylabel("Lattice Length (% Change)")
        ax1.set_title(f"{{basename}} - Lattice Parameter % Change vs. Time")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2.plot(time_ps, data['alpha_perc'], label='α (%)')
        ax2.plot(time_ps, data['beta_perc'], label='β (%)')
        ax2.plot(time_ps, data['gamma_perc'], label='γ (%)')
        ax2.set_xlabel("Time (ps)")
        ax2.set_ylabel("Lattice Angle (% Change)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(f"{{plot_prefix}}_lattice_perc_change.png")
        plt.close()
    except Exception as e:
        print(f"    ... failed to plot lattice percentage change: {{e}}")

    print(f"  ... plots saved to md_results/md_{{basename}}_*.png")

def main():
    global md_params


{indented_calculator_setup} 

    if 'calculator' not in locals() or calculator is None:
        print("Calculator could not be initialized. Exiting.")
        exit()

    print("\\nSearching for structure files (*.cif, *.vasp, POSCAR*)...")
    structure_files = glob.glob("*.cif") + glob.glob("*.vasp") + glob.glob("POSCAR*") + glob.glob("*.poscar") 

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

                if 'NPT' in md_params['ensemble']:
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
