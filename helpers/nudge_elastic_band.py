import streamlit as st
import numpy as np
from ase.mep import NEB, interpolate
from ase.optimize import BFGS, FIRE, LBFGS, LBFGSLineSearch, MDMin
from ase.io import read, write
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
from datetime import datetime


def setup_neb_parameters_ui():
    st.subheader("NEB Calculation Parameters")
    neb_params = {}

    st.write("**Band geometry**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        neb_params['n_images'] = st.number_input(
            "Number of Images", min_value=3, max_value=20, value=7, step=1,
            help="Number of intermediate images (endpoints are NOT counted)")
    with col2:
        neb_params['spring_constant'] = st.number_input(
            "Spring Constant (eV/Å)", min_value=0.01, max_value=10.0,
            value=0.1, step=0.01, format="%.2f",
            help="Elastic spring constant connecting adjacent images")
    with col3:
        neb_params['neb_method'] = st.selectbox(
            "NEB Method",
            ["improvedtangent", "aseneb", "eb", "spline", "string"],
            index=0,
            help=(
                "**improvedtangent** — recommended (Henkelman 2000)\n\n"
                "**aseneb** — original ASE NEB\n\n"
                "**eb** — elastic band without tangent\n\n"
                "**spline** / **string** — alternative formulations"))
    with col4:
        neb_params['interpolation'] = st.selectbox(
            "Interpolation Method", ["idpp", "linear"], index=0,
            help=(
                "**idpp** — Image Dependent Pair Potential (recommended)\n\n"
                "**linear** — simple linear interpolation"))

    st.write("**Climbing Image**")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        neb_params['climb'] = st.checkbox(
            "Climbing Image NEB", value=True,
            help="CI-NEB for accurate saddle point location")
    with col6:
        neb_params['climb_from_start'] = st.checkbox(
            "Climb from first step", value=False,
            help="If OFF, climbing activates only after fmax drops below the threshold",
            disabled=not neb_params['climb'])
    with col7:
        neb_params['climb_switch_fmax'] = st.number_input(
            "Activate CI at fmax <= (eV/Å)",
            min_value=0.01, max_value=5.0, value=0.5, step=0.05, format="%.2f",
            help="Only used when 'Climb from first step' is OFF",
            disabled=(not neb_params['climb']) or neb_params['climb_from_start'])
    with col8:
        neb_params['remove_rotation'] = st.checkbox(
            "Remove Rotation & Translation", value=True,
            help="Projects out global rotation/translation from image forces")

    st.write("**Convergence & Optimiser**")
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        neb_params['fmax'] = st.number_input(
            "Force Convergence (eV/Å)",
            min_value=0.001, max_value=0.5, value=0.05, step=0.005, format="%.3f")
    with col10:
        neb_params['max_steps'] = st.number_input(
            "Maximum Steps", min_value=10, max_value=10000, value=500, step=50)
    with col11:
        neb_params['optimizer'] = st.selectbox(
            "Optimizer",
            ["LBFGS", "BFGS", "FIRE", "LBFGSLineSearch", "MDMin"],
            index=0,
            help=(
                "**LBFGS** — fast quasi-Newton, good default for NEB\n\n"
                "**BFGS** — classic quasi-Newton\n\n"
                "**FIRE** — robust for highly non-linear bands\n\n"
                "**LBFGSLineSearch** — LBFGS with line-search\n\n"
                "**MDMin** — molecular dynamics minimiser"))
    with col12:
        neb_params['log_interval'] = st.number_input(
            "Log every N steps", min_value=1, max_value=200, value=5, step=1,
            help="Print per-image energy/force table every N steps")

    st.write("**Endpoint pre-optimisation**")
    col13, col14, col15, _ = st.columns(4)
    with col13:
        neb_params['pre_optimize'] = st.checkbox(
            "Pre-optimise endpoints", value=True,
            help="Recommended: ensures endpoints sit at true local minima")
    with col14:
        neb_params['pre_opt_fmax'] = st.number_input(
            "Pre-opt fmax (eV/A)",
            min_value=0.001, max_value=1.0, value=0.05, step=0.005, format="%.3f",
            disabled=not neb_params['pre_optimize'])
    with col15:
        neb_params['pre_opt_steps'] = st.number_input(
            "Pre-opt max steps", min_value=10, max_value=2000, value=200, step=10,
            disabled=not neb_params['pre_optimize'])

    st.info(
        "Recommended: **improvedtangent + idpp + LBFGS + delayed CI-NEB**. "
        "Use FIRE if the band is highly non-linear.")

    return neb_params


def _make_optimizer(name, neb_obj):
    name = name.upper()
    if name == "LBFGS":           return LBFGS(neb_obj, logfile=None)
    if name == "LBFGSLINESEARCH": return LBFGSLineSearch(neb_obj, logfile=None)
    if name == "FIRE":            return FIRE(neb_obj, logfile=None)
    if name == "MDMIN":           return MDMin(neb_obj, logfile=None)
    return BFGS(neb_obj, logfile=None)


def _mic_distance(pos1, pos2, cell):

    diff = pos2 - pos1
    frac = np.linalg.solve(cell.T, diff.T).T
    frac -= np.round(frac)
    return float(np.linalg.norm(frac @ cell))


def _calc_block(selected_model, model_size, device, dtype,
                mace_head=None, mace_dispersion=False, mace_dispersion_xc='pbe',
                custom_mace_path=None, indent="    "):
    """Return Python source-code string that creates `calculator`."""
    i = indent
    is_chgnet    = selected_model.startswith("CHGNet")
    is_sevennet  = selected_model.startswith("SevenNet")
    is_mattersim = selected_model.startswith("MatterSim")
    is_orb       = selected_model.startswith("ORB")
    is_nequix    = selected_model.startswith("Nequix")
    is_mace_off  = "OFF" in selected_model
    is_upet      = model_size.startswith("upet:")
    is_petmad    = selected_model.startswith("PET-MAD")
    is_url       = isinstance(model_size, str) and model_size.startswith("http")

    if is_chgnet:
        ver = model_size.split("-")[1] if "-" in model_size else "0.3.0"
        return (f"{i}from chgnet.model.model import CHGNet\n"
                f"{i}from chgnet.model.dynamics import CHGNetCalculator\n"
                f"{i}chgnet = CHGNet.load(model_name='{ver}', use_device='{device}', verbose=False)\n"
                f"{i}calculator = CHGNetCalculator(model=chgnet, use_device='{device}')\n"
                f"{i}print('CHGNet {ver} ready')\n")
    if is_sevennet:
        return (f"{i}import torch; torch.serialization.add_safe_globals([slice])\n"
                f"{i}from sevenn.calculator import SevenNetCalculator\n"
                f"{i}calculator = SevenNetCalculator(model='{model_size}', device='{device}')\n"
                f"{i}print('SevenNet ready')\n")
    if is_mattersim:
        mp = "MatterSim-v1.0.0-5M.pth" if "5m" in model_size else "MatterSim-v1.0.0-1M.pth"
        return (f"{i}from mattersim.forcefield import MatterSimCalculator\n"
                f"{i}calculator = MatterSimCalculator(model_path='{mp}', device='{device}')\n"
                f"{i}print('MatterSim ready')\n")
    if is_orb:
        return (f"{i}from orb_models.forcefield import pretrained\n"
                f"{i}from orb_models.forcefield.calculator import ORBCalculator\n"
                f"{i}orbff = pretrained.{model_size}(device='{device}', precision='float32-high')\n"
                f"{i}calculator = ORBCalculator(orbff, device='{device}')\n"
                f"{i}print('ORB ready')\n")
    if is_nequix:
        return (f"{i}from nequix.calculator import NequixCalculator\n"
                f"{i}calculator = NequixCalculator('{model_size}')\n"
                f"{i}print('Nequix ready')\n")
    if is_upet or is_petmad:
        if is_upet:
            parts = model_size.replace("upet:", "").split(":")
            mn, mv = parts[0], (parts[1] if len(parts) > 1 else "latest")
        else:
            mn, mv = "pet-mad-s", "1.0.2"
        return (f"{i}from upet.calculator import UPETCalculator\n"
                f"{i}calculator = UPETCalculator(model='{mn}', version='{mv}', device='{device}')\n"
                f"{i}print('UPET ready')\n")
    if custom_mace_path:
        hl = f", head='{mace_head}'" if mace_head else ""
        return (f"{i}from mace.calculators import mace_mp\n"
                f"{i}calculator = mace_mp(model='{custom_mace_path}', device='{device}', "
                f"default_dtype='{dtype}'{hl})\n"
                f"{i}print('Custom MACE ready')\n")
    if is_url:
        hl   = f", head='{mace_head}'" if mace_head else ""
        disp = (f", dispersion=True, dispersion_xc='{mace_dispersion_xc}'"
                if mace_dispersion else "")
        return (f"{i}import urllib.request; from pathlib import Path\n"
                f"{i}from mace.calculators import mace_mp\n"
                f"{i}_url = '{model_size}'\n"
                f"{i}_name = _url.split('/')[-1]\n"
                f"{i}_cache = Path.home() / '.cache' / 'mace_foundation_models' / _name\n"
                f"{i}_cache.parent.mkdir(parents=True, exist_ok=True)\n"
                f"{i}if not _cache.exists():\n"
                f"{i}    print(f'Downloading {{_name}} ...')\n"
                f"{i}    urllib.request.urlretrieve(_url, str(_cache))\n"
                f"{i}calculator = mace_mp(model=str(_cache), device='{device}', "
                f"default_dtype='{dtype}'{hl}{disp})\n"
                f"{i}print('MACE foundation model ready')\n")
    if is_mace_off:
        return (f"{i}from mace.calculators import mace_off\n"
                f"{i}calculator = mace_off(model='{model_size}', "
                f"default_dtype='{dtype}', device='{device}')\n"
                f"{i}print('MACE-OFF ready')\n")

    disp = (f", dispersion=True, dispersion_xc='{mace_dispersion_xc}'"
            if mace_dispersion else ", dispersion=False")
    return (f"{i}from mace.calculators import mace_mp\n"
            f"{i}calculator = mace_mp(model='{model_size}', "
            f"default_dtype='{dtype}', device='{device}'{disp})\n"
            f"{i}print('MACE-MP ready')\n")


def run_neb_calculation(initial_structure, final_structure, calculator,
                        neb_params, log_queue, stop_event, structure_name):
    try:
        import tempfile, os
        from ase.io import Trajectory

        log_queue.put(f"Starting NEB calculation: {structure_name}")

        adaptor = AseAtomsAdaptor()
        initial_atoms = adaptor.get_atoms(initial_structure)
        final_atoms   = adaptor.get_atoms(final_structure)


        if neb_params.get('pre_optimize', True):
            pre_fmax  = float(neb_params.get('pre_opt_fmax',  0.05))
            pre_steps = int(neb_params.get('pre_opt_steps', 200))
            log_queue.put(f"  Pre-optimising endpoints (fmax={pre_fmax} eV/A, max {pre_steps} steps)...")
            initial_atoms.calc = calculator
            BFGS(initial_atoms, logfile=None).run(fmax=pre_fmax, steps=pre_steps)
            E0    = float(initial_atoms.get_potential_energy())
            fmax0 = float(np.max(np.linalg.norm(initial_atoms.get_forces(), axis=1)))
            log_queue.put(f"  Initial: E = {E0:.6f} eV | fmax = {fmax0:.4f} eV/A")
            final_atoms.calc = calculator
            BFGS(final_atoms, logfile=None).run(fmax=pre_fmax, steps=pre_steps)
            Ef    = float(final_atoms.get_potential_energy())
            fmaxf = float(np.max(np.linalg.norm(final_atoms.get_forces(), axis=1)))
            log_queue.put(f"  Final  : E = {Ef:.6f} eV | fmax = {fmaxf:.4f} eV/A")
        else:
            initial_atoms.calc = calculator
            final_atoms.calc   = calculator
            E0 = float(initial_atoms.get_potential_energy())
            Ef = float(final_atoms.get_potential_energy())
            log_queue.put("  Endpoint pre-optimisation skipped")
            log_queue.put(f"  Initial E = {E0:.6f} eV  |  Final E = {Ef:.6f} eV")


        n_images = int(neb_params['n_images'])
        log_queue.put(f"  Building {n_images} intermediate images ...")
        images = ([initial_atoms.copy()]
                  + [initial_atoms.copy() for _ in range(n_images - 2)]
                  + [final_atoms.copy()])

        interp = neb_params.get('interpolation', 'idpp')
        log_queue.put(f"  Interpolating with '{interp}' ...")
        if interp == 'idpp':
            for img in images:
                img.calc = calculator
            try:
                NEB(images, allow_shared_calculator=True).interpolate(method='idpp')
                log_queue.put("  IDPP interpolation done")
            except Exception as e:
                log_queue.put(f"  IDPP failed ({e}), falling back to linear")
                images = ([initial_atoms.copy()]
                          + [initial_atoms.copy() for _ in range(n_images - 2)]
                          + [final_atoms.copy()])
                interpolate(images)
                for img in images:
                    img.calc = calculator
        else:
            interpolate(images)
            for img in images:
                img.calc = calculator


        neb_method  = neb_params.get('neb_method', 'improvedtangent')
        k           = float(neb_params['spring_constant'])
        use_climb   = bool(neb_params.get('climb', True))
        from_start  = bool(neb_params.get('climb_from_start', False))
        start_climb = use_climb and from_start
        switch_fmax = float(neb_params.get('climb_switch_fmax', 0.5))

        log_queue.put(
            f"  NEB: method={neb_method}, k={k} eV/A, "
            f"climb={'immediate' if start_climb else ('delayed @' + str(switch_fmax) if use_climb else 'off')}, "
            f"remove_rot={neb_params.get('remove_rotation', True)}")

        neb = NEB(
            images, k=k, climb=start_climb, method=neb_method,
            remove_rotation_and_translation=bool(neb_params.get('remove_rotation', True)),
            allow_shared_calculator=True)

        optimizer = _make_optimizer(neb_params.get('optimizer', 'LBFGS'), neb)
        optimizer.max_steps = int(neb_params['max_steps'])

        traj_path = os.path.join(
            tempfile.gettempdir(),
            f'neb_{structure_name.replace(".", "_").replace(" ", "_")}.traj')
        traj = Trajectory(traj_path, 'w', neb)
        optimizer.attach(traj.write, interval=1)


        trajectory_data  = []
        energies_history = []
        step_count       = [0]
        climb_activated  = [start_climb]
        log_interval     = int(neb_params.get('log_interval', 5))

        def neb_callback():
            step_count[0] += 1
            if stop_event.is_set():
                raise RuntimeError("Calculation stopped by user")

            cur_e, cur_f = [], []
            for img in images:
                try:
                    cur_e.append(float(img.get_potential_energy()))
                    cur_f.append(float(np.max(np.linalg.norm(img.get_forces(), axis=1))))
                except Exception:
                    cur_e.append(float('nan'))
                    cur_f.append(float('nan'))
            energies_history.append(cur_e)

            valid_f = [v for v in cur_f if not np.isnan(v)]
            mf = max(valid_f) if valid_f else float('nan')


            if use_climb and not climb_activated[0] and mf < switch_fmax:
                neb.climb = True
                climb_activated[0] = True
                log_queue.put(
                    f"    Step {step_count[0]}: fmax={mf:.4f} eV/A — CI-NEB activated")

            if step_count[0] % log_interval == 0 or step_count[0] == 1:
                ci_tag = " [CI]" if climb_activated[0] else ""
                log_queue.put(
                    f"  -- Step {step_count[0]:4d}{ci_tag}  |  fmax = {mf:.4f} eV/A --")
                log_queue.put(
                    f"    {'Img':>4}  {'E (eV)':>14}  {'dE (meV)':>10}  {'fmax (eV/A)':>12}")
                e0_row = cur_e[0] if not np.isnan(cur_e[0]) else 0.0
                valid_inner = [v for ii, v in enumerate(cur_e)
                               if not np.isnan(v) and ii not in (0, len(cur_e)-1)]
                max_inner = max(valid_inner) if valid_inner else None
                for ii, (ee, ff) in enumerate(zip(cur_e, cur_f)):
                    de = (ee - e0_row) * 1000 if not np.isnan(ee) else float('nan')
                    tag = " <- TS?" if (max_inner is not None and not np.isnan(ee)
                                        and ee == max_inner
                                        and ii not in (0, len(cur_e)-1)) else ""
                    log_queue.put(
                        f"    {ii:>4}  {ee:>14.6f}  {de:>10.2f}  {ff:>12.4f}{tag}")

            log_queue.put({
                'type': 'neb_step', 'structure': structure_name,
                'step': step_count[0], 'max_steps': neb_params['max_steps'],
                'energies': cur_e, 'forces': cur_f, 'max_force': mf,
            })

        optimizer.attach(neb_callback, interval=1)


        log_queue.put(
            f"  Running {neb_params.get('optimizer','LBFGS')} "
            f"(max={neb_params['max_steps']} steps, fmax={neb_params['fmax']} eV/A) ...")
        optimizer.run(fmax=float(neb_params['fmax']), steps=int(neb_params['max_steps']))
        traj.close()
        log_queue.put(f"  Optimisation finished after {step_count[0]} steps")


        final_energies  = []
        final_forces    = []
        final_positions = []

        for i, img in enumerate(images):
            e  = float(img.get_potential_energy())
            f  = float(np.max(np.linalg.norm(img.get_forces(), axis=1)))
            final_energies.append(e)
            final_forces.append(f)
            final_positions.append(img.positions.copy())
            trajectory_data.append({
                'image_index': i,
                'energy':      e,
                'max_force':   f,
                'positions':   img.positions.tolist(),
                'cell':        img.cell.array.tolist(),
                'symbols':     img.get_chemical_symbols(),
                'forces':      img.get_forces().tolist(),
            })

        final_energies = np.array(final_energies)
        reaction_coord = list(range(len(final_energies)))


        reaction_distances = [0.0]
        for i in range(1, len(images)):
            reaction_distances.append(
                reaction_distances[-1] +
                _mic_distance(final_positions[i-1], final_positions[i],
                              images[i].cell.array))

        max_idx         = int(np.argmax(final_energies))
        forward_barrier = float(final_energies[max_idx] - final_energies[0])
        reverse_barrier = float(final_energies[max_idx] - final_energies[-1])
        reaction_energy = float(final_energies[-1]       - final_energies[0])

        log_queue.put(f"  Forward barrier : {forward_barrier:.4f} eV  ({forward_barrier*96.4853:.2f} kJ/mol)")
        log_queue.put(f"  Reverse barrier : {reverse_barrier:.4f} eV  ({reverse_barrier*96.4853:.2f} kJ/mol)")
        log_queue.put(f"  Reaction energy : {reaction_energy:.4f} eV  ({reaction_energy*96.4853:.2f} kJ/mol)")
        log_queue.put(f"  Saddle point    : image {max_idx}")
        log_queue.put(f"  Path length     : {reaction_distances[-1]:.3f} A")

        try:
            os.remove(traj_path)
        except Exception:
            pass

        return {
            'success':             True,
            'energies':            [float(e) for e in final_energies],
            'forces':              [float(f) for f in final_forces],
            'reaction_coordinate': reaction_coord,
            'reaction_distances':  reaction_distances,
            'forward_barrier_eV':  forward_barrier,
            'reverse_barrier_eV':  reverse_barrier,
            'reaction_energy_eV':  reaction_energy,
            'forward_barrier_kJ':  forward_barrier * 96.4853,
            'reverse_barrier_kJ':  reverse_barrier * 96.4853,
            'reaction_energy_kJ':  reaction_energy * 96.4853,
            'barrier_index':       max_idx,
            'trajectory_data':     trajectory_data,
            'energies_history':    energies_history,
            'n_images':            n_images,
            'converged_steps':     step_count[0],
            'neb_method':          neb_method,
            'climb_activated':     climb_activated[0],
            'neb_params':          neb_params,
        }

    except Exception as e:
        log_queue.put(f"NEB calculation failed: {str(e)}")
        import traceback
        log_queue.put(f"Traceback: {traceback.format_exc()}")
        return {'success': False, 'error': str(e)}


def create_neb_trajectory_xyz(trajectory_data, structure_name):

    xyz_content = ""
    for step_data in trajectory_data:
        positions = np.array(step_data['positions'])
        cell      = np.array(step_data['cell'])
        symbols   = step_data['symbols']
        forces    = np.array(step_data['forces']) if step_data.get('forces') else None
        energy    = step_data['energy']
        max_force = step_data['max_force']
        img_idx   = step_data['image_index']

        lattice_str = " ".join(f"{x:.8f}" for x in cell.flatten())
        props = "species:S:1:pos:R:3:forces:R:3" if forces is not None else "species:S:1:pos:R:3"
        comment = (
            f'Image={img_idx} Energy={energy:.8f} Max_Force={max_force:.6f} '
            f'Lattice="{lattice_str}" Properties={props} pbc="T T T"')
        xyz_content += f"{len(positions)}\n{comment}\n"
        for i in range(len(positions)):
            if forces is not None:
                xyz_content += (
                    f"{symbols[i]:2s} "
                    f"{positions[i,0]:16.8f} {positions[i,1]:16.8f} {positions[i,2]:16.8f} "
                    f"{forces[i,0]:14.8f} {forces[i,1]:14.8f} {forces[i,2]:14.8f}\n")
            else:
                xyz_content += (
                    f"{symbols[i]:2s} "
                    f"{positions[i,0]:16.8f} {positions[i,1]:16.8f} {positions[i,2]:16.8f}\n")
    return xyz_content


def export_neb_results(neb_result, structure_name):
    export_data = {
        'structure_name':         structure_name,
        'calculation_type':       'Nudged Elastic Band',
        'timestamp':              datetime.now().isoformat(),
        'neb_method':             neb_result.get('neb_method', 'improvedtangent'),
        'climb_activated':        neb_result.get('climb_activated', False),
        'forward_barrier_eV':     float(neb_result['forward_barrier_eV']),
        'forward_barrier_kJ_mol': float(neb_result['forward_barrier_kJ']),
        'reverse_barrier_eV':     float(neb_result['reverse_barrier_eV']),
        'reverse_barrier_kJ_mol': float(neb_result['reverse_barrier_kJ']),
        'reaction_energy_eV':     float(neb_result['reaction_energy_eV']),
        'reaction_energy_kJ_mol': float(neb_result['reaction_energy_kJ']),
        'transition_state_image': int(neb_result['barrier_index']),
        'number_of_images':       int(neb_result['n_images']),
        'converged_steps':        int(neb_result['converged_steps']),
        'energies_eV':            [float(e) for e in neb_result['energies']],
        'forces_eV_Ang':          [float(f) for f in neb_result['forces']],
        'reaction_distances_Ang': [float(d) for d in neb_result.get('reaction_distances', [])],
        'neb_parameters': {
            'spring_constant':  float(neb_result['neb_params']['spring_constant']),
            'climbing_image':   bool(neb_result['neb_params']['climb']),
            'force_convergence':float(neb_result['neb_params']['fmax']),
            'optimizer':        str(neb_result['neb_params']['optimizer']),
            'interpolation':    str(neb_result['neb_params']['interpolation']),
            'neb_method':       str(neb_result['neb_params'].get('neb_method', 'improvedtangent')),
        }
    }
    return json.dumps(export_data, indent=2)


def create_neb_plot(neb_results, structure_name, use_distance=False):
    energies     = np.array(neb_results['energies'])
    energies_meV = (energies - energies[0]) * 1000
    bi           = neb_results['barrier_index']
    rc           = neb_results['reaction_coordinate']

    if use_distance and 'reaction_distances' in neb_results:
        x      = neb_results['reaction_distances']
        xtitle = "Distance along path (Å)"
    else:
        x      = rc
        xtitle = "Image Number"

    fwd_eV = neb_results['forward_barrier_eV']
    fwd_kJ = neb_results['forward_barrier_kJ']
    title  = (f"NEB Energy Profile — {structure_name} | "
              f"Eb = {fwd_eV:.4f} eV ({fwd_kJ:.2f} kJ/mol)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=energies_meV, mode='lines+markers', name='NEB Path',
        line=dict(width=3, color='#1f77b4'), marker=dict(size=10),
        hovertemplate='<b>Image %{customdata}</b><br>E: %{y:.3f} meV<extra></extra>',
        customdata=rc))
    fig.add_trace(go.Scatter(
        x=[x[0]], y=[energies_meV[0]], mode='markers', name='Initial',
        marker=dict(size=14, color='green', symbol='star')))
    fig.add_trace(go.Scatter(
        x=[x[-1]], y=[energies_meV[-1]], mode='markers', name='Final',
        marker=dict(size=14, color='red', symbol='star')))
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=xtitle, yaxis_title="Relative Energy (meV)",
        height=550, font=dict(size=16), hovermode='closest',
        legend=dict(font=dict(size=14), x=1.02, y=1),
        xaxis=dict(tickfont=dict(size=14), title_font=dict(size=18)),
        yaxis=dict(tickfont=dict(size=14), title_font=dict(size=18)))
    return fig


def create_combined_neb_plot(all_neb_results, use_distance=False):
    fig    = go.Figure()
    colors = ['#1f77b4','#d62728','#2ca02c','#9467bd',
              '#e377c2','#8c564b','#bcbd22','#17becf']
    xtitle = "Distance along path (A)" if use_distance else "Image Number"
    for idx, (name, r) in enumerate(all_neb_results.items()):
        if not r.get('success'):
            continue
        e    = np.array(r['energies'])
        emeV = (e - e[0]) * 1000
        rc   = r['reaction_coordinate']
        x    = r['reaction_distances'] if (use_distance and 'reaction_distances' in r) else rc
        c    = colors[idx % len(colors)]
        bi   = r['barrier_index']
        fig.add_trace(go.Scatter(
            x=x, y=emeV, mode='lines+markers', name=name,
            line=dict(width=2, color=c), marker=dict(size=8),
            hovertemplate=f'<b>{name}</b><br>Image %{{customdata}}<br>E: %{{y:.2f}} meV<extra></extra>',
            customdata=rc))
        fig.add_trace(go.Scatter(
            x=[x[bi]], y=[emeV[bi]], mode='markers', showlegend=False,
            marker=dict(size=14, color=c, symbol='diamond'),
            hovertemplate=f'<b>{name} TS</b><br>Eb={r["forward_barrier_eV"]:.4f} eV<extra></extra>'))
    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.4)
    fig.update_layout(
        title=dict(text="Combined NEB Energy Profiles", font=dict(size=22)),
        xaxis_title=xtitle, yaxis_title="Relative Energy (meV)",
        height=700, font=dict(size=16), hovermode='closest',
        legend=dict(font=dict(size=14), x=1.02, y=1),
        xaxis=dict(tickfont=dict(size=14), title_font=dict(size=18)),
        yaxis=dict(tickfont=dict(size=14), title_font=dict(size=18)))
    return fig


def display_neb_results(neb_results, structure_name, use_distance=False):
    st.subheader(f"NEB Results: {structure_name}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Forward Barrier",
                  f"{neb_results['forward_barrier_eV']:.4f} eV",
                  delta=f"{neb_results['forward_barrier_kJ']:.2f} kJ/mol")
    with col2:
        st.metric("Reverse Barrier",
                  f"{neb_results['reverse_barrier_eV']:.4f} eV",
                  delta=f"{neb_results['reverse_barrier_kJ']:.2f} kJ/mol")
    with col3:
        st.metric("Reaction Energy",
                  f"{neb_results['reaction_energy_eV']:.4f} eV",
                  delta=f"{neb_results['reaction_energy_kJ']:.2f} kJ/mol")
    with col4:
        st.metric("Transition State",
                  f"Image {neb_results['barrier_index']}",
                  delta=f"{neb_results['converged_steps']} steps")
    col5, col6 = st.columns(2)
    with col5:
        if 'reaction_distances' in neb_results:
            st.info(f"Path length: {neb_results['reaction_distances'][-1]:.3f} A")
    with col6:
        st.info(f"Method: {neb_results.get('neb_method','?')} | "
                f"CI: {'active' if neb_results.get('climb_activated') else 'off'}")
    st.plotly_chart(create_neb_plot(neb_results, structure_name, use_distance),
                    use_container_width=True)


def generate_neb_script(neb_params, selected_model, model_size, device, dtype,
                        thread_count=4, mace_head=None,
                        mace_dispersion=False, mace_dispersion_xc='pbe',
                        custom_mace_path=None):

    calc_src     = _calc_block(selected_model, model_size, device, dtype,
                               mace_head, mace_dispersion, mace_dispersion_xc,
                               custom_mace_path)
    climb        = neb_params.get('climb', True)
    ci_from_start= neb_params.get('climb_from_start', False)

    return f'''#!/usr/bin/env python3
"""
Standalone NEB calculation script  (full version: plots + CSV + per-image POSCARs)
Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model     : {selected_model}
Device    : {device}
Method    : {neb_params.get("neb_method", "improvedtangent")}

Place POSCAR_initial and POSCAR_final (or any file with "initial"/"final"
in the name) in the same directory, then run:
    python neb_calculation.py

All output goes to neb_results/
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "{thread_count}"
import torch
torch.set_num_threads({thread_count})

from ase.io import read, write
from ase.mep import NEB, interpolate
from ase.optimize import BFGS, FIRE, LBFGS, LBFGSLineSearch, MDMin

Path("neb_results").mkdir(exist_ok=True)

# Parameters
N_IMAGES          = {int(neb_params["n_images"])}
SPRING_K          = {float(neb_params["spring_constant"])}
NEB_METHOD        = "{neb_params.get("neb_method", "improvedtangent")}"
INTERPOLATION     = "{neb_params.get("interpolation", "idpp")}"
CLIMB             = {climb}
CLIMB_FROM_START  = {ci_from_start}
CLIMB_SWITCH_FMAX = {float(neb_params.get("climb_switch_fmax", 0.5))}
FMAX              = {float(neb_params["fmax"])}
MAX_STEPS         = {int(neb_params["max_steps"])}
OPTIMIZER         = "{neb_params.get("optimizer", "LBFGS")}"
REMOVE_ROT_TRANS  = {neb_params.get("remove_rotation", True)}
LOG_INTERVAL      = {int(neb_params.get("log_interval", 5))}
PRE_OPTIMIZE      = {neb_params.get("pre_optimize", True)}
PRE_OPT_FMAX      = {float(neb_params.get("pre_opt_fmax", 0.05))}
PRE_OPT_STEPS     = {int(neb_params.get("pre_opt_steps", 200))}


def make_optimizer(name, obj):
    name = name.upper()
    if name == "LBFGS":           return LBFGS(obj, logfile=None)
    if name == "LBFGSLINESEARCH": return LBFGSLineSearch(obj, logfile=None)
    if name == "FIRE":            return FIRE(obj, logfile=None)
    if name == "MDMIN":           return MDMin(obj, logfile=None)
    return BFGS(obj, logfile=None)


def setup_calculator():
{calc_src}    return calculator


def pre_opt(atoms, calc, fmax, steps, label):
    atoms.calc = calc
    BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    e  = atoms.get_potential_energy()
    mf = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
    print(f"  {{label}}: E={{e:.6f}} eV | fmax={{mf:.4f}} eV/A")
    return atoms


def mic_dist(p1, p2, cell):
    d = p2 - p1
    f = np.linalg.solve(cell.T, d.T).T
    f -= np.round(f)
    return float(np.linalg.norm(f @ cell))


def save_trajectory_xyz(images, path):

    with open(path, "w") as fh:
        for i, img in enumerate(images):
            pos  = img.positions          # raw Cartesian — no frac%1
            cell = img.cell.array
            sym  = img.get_chemical_symbols()
            try:
                e  = img.get_potential_energy()
                f  = img.get_forces()
                mf = float(np.max(np.linalg.norm(f, axis=1)))
                has_f = True
            except Exception:
                e, mf, has_f = 0.0, 0.0, False
            lat = " ".join(f"{{x:.8f}}" for x in cell.flatten())
            props = "species:S:1:pos:R:3:forces:R:3" if has_f else "species:S:1:pos:R:3"
            fh.write(f"{{len(pos)}}\\n")
            fh.write(f'Image={{i}} Energy={{e:.8f}} Max_Force={{mf:.6f}} '
                     f'Lattice="{{lat}}" Properties={{props}} pbc="T T T"\\n')
            for j in range(len(pos)):
                if has_f:
                    fh.write(f"{{sym[j]:2s}} {{pos[j,0]:16.8f}} {{pos[j,1]:16.8f}} {{pos[j,2]:16.8f}}"
                             f" {{f[j,0]:14.8f}} {{f[j,1]:14.8f}} {{f[j,2]:14.8f}}\\n")
                else:
                    fh.write(f"{{sym[j]:2s}} {{pos[j,0]:16.8f}} {{pos[j,1]:16.8f}} {{pos[j,2]:16.8f}}\\n")
    print(f"  Trajectory -> {{path}}")


def save_images_poscar(images, out_dir):
    for i, img in enumerate(images):
        label = "initial" if i == 0 else ("final" if i == len(images)-1 else f"image_{{i:02d}}")
        write(os.path.join(out_dir, f"neb_{{label}}.vasp"), img,
              format="vasp", direct=True, sort=False)
    print(f"  {{len(images)}} image POSCARs -> {{out_dir}}/")


def plot_profile(energies, distances, bi, result, out_path):
    try:
        import matplotlib.pyplot as plt
        e_arr = np.array(energies)
        eV_rel = e_arr - e_arr[0]
        x = np.array(distances)

        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax.plot(x, eV_rel, "o-", lw=2.5, color="#1f77b4", ms=9, label="NEB path")
        ax.plot(x[0],  eV_rel[0],  "*", ms=18, color="green",  label="Initial")
        ax.plot(x[-1], eV_rel[-1], "*", ms=18, color="red",    label="Final")
        ax.plot(x[bi], eV_rel[bi], "D", ms=14, color="orange", label="TS")
        ax.axhline(0, ls="--", color="gray", lw=1, alpha=0.5)
        ax.annotate("", xy=(x[bi], eV_rel[bi]), xytext=(x[0], 0),
                    arrowprops=dict(arrowstyle="<->", color="orange", lw=1.5))
        ax.text((x[0]+x[bi])/2, eV_rel[bi]*0.55,
                f"Eb = {{result['forward_barrier_eV']:.4f}} eV\\n"
                f"   = {{result['forward_barrier_kJ']:.2f}} kJ/mol",
                color="darkorange", fontsize=10, ha="center")
        ax.set_xlabel("Distance along path (A)", fontsize=12)
        ax.set_ylabel("Relative Energy (eV)",    fontsize=12)
        ax.set_title("NEB Energy Profile", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

        ax2.axis("off")
        rows = [
            ["Forward barrier",  f"{{result['forward_barrier_eV']:.4f}} eV",
             f"{{result['forward_barrier_kJ']:.2f}} kJ/mol"],
            ["Reverse barrier",  f"{{result['reverse_barrier_eV']:.4f}} eV",
             f"{{result['reverse_barrier_kJ']:.2f}} kJ/mol"],
            ["Reaction energy",  f"{{result['reaction_energy_eV']:.4f}} eV",
             f"{{result['reaction_energy_kJ']:.2f}} kJ/mol"],
            ["TS image",         str(bi), ""],
            ["Path length",      f"{{x[-1]:.3f}} A", ""],
            ["NEB method",       NEB_METHOD, ""],
            ["Optimizer",        OPTIMIZER, ""],
            ["N images",         str(N_IMAGES), ""],
            ["Spring k",         f"{{SPRING_K}} eV/A", ""],
        ]
        tbl = ax2.table(cellText=rows, colLabels=["Property","Value","Alt."],
                        loc="center", cellLoc="left")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Plot -> {{out_path}}")
    except ImportError:
        print("  matplotlib not available - skipping plot")
    except Exception as err:
        print(f"  Plot failed: {{err}}")


def find_structure(keyword):
    for f in sorted(os.listdir(".")):
        low = f.lower()
        if keyword in low and (
            f.startswith("POSCAR") or
            f.endswith((".vasp", ".cif", ".xyz", ".poscar"))):
            return f
    return None


def main():
    t0 = time.time()
    print("NEB calculation |", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"  Model     : {selected_model}")
    print(f"  Method    : {{NEB_METHOD}} | Interp: {{INTERPOLATION}}")
    print(f"  Images    : {{N_IMAGES}} intermediate | k = {{SPRING_K}} eV/A")
    print(f"  Climb     : {{CLIMB}} | from_start={{CLIMB_FROM_START}} | switch@{{CLIMB_SWITCH_FMAX}}")
    print(f"  Optimizer : {{OPTIMIZER}} | fmax={{FMAX}} | max_steps={{MAX_STEPS}}")

    init_file  = find_structure("initial")
    final_file = find_structure("final")
    if not init_file or not final_file:
        print("ERROR: need files with 'initial' and 'final' in the name")
        sys.exit(1)
    print(f"\\n  Initial : {{init_file}}")
    print(f"  Final   : {{final_file}}")

    initial_atoms = read(init_file)
    final_atoms   = read(final_file)
    n = len(initial_atoms)
    print(f"  {{n}} atoms loaded")
    if n != len(final_atoms):
        print("ERROR: atom count mismatch!"); sys.exit(1)

    print("\\nSetting up calculator ...")
    calculator = setup_calculator()

    if PRE_OPTIMIZE:
        print(f"\\nPre-optimising (fmax={{PRE_OPT_FMAX}}, steps={{PRE_OPT_STEPS}}) ...")
        initial_atoms = pre_opt(initial_atoms, calculator, PRE_OPT_FMAX, PRE_OPT_STEPS, "Initial")
        final_atoms   = pre_opt(final_atoms,   calculator, PRE_OPT_FMAX, PRE_OPT_STEPS, "Final")
    else:
        initial_atoms.calc = calculator
        final_atoms.calc   = calculator
        print(f"  Initial E = {{initial_atoms.get_potential_energy():.6f}} eV")
        print(f"  Final   E = {{final_atoms.get_potential_energy():.6f}} eV")

    print(f"\\nBuilding {{N_IMAGES}} intermediate images ...")
    images = ([initial_atoms.copy()]
              + [initial_atoms.copy() for _ in range(N_IMAGES - 2)]
              + [final_atoms.copy()])

    print(f"Interpolating with {{INTERPOLATION}} ...")
    if INTERPOLATION == "idpp":
        for img in images: img.calc = calculator
        try:
            NEB(images, allow_shared_calculator=True).interpolate(method="idpp")
            print("  IDPP done")
        except Exception as err:
            print(f"  IDPP failed ({{err}}), using linear fallback")
            images = ([initial_atoms.copy()]
                      + [initial_atoms.copy() for _ in range(N_IMAGES - 2)]
                      + [final_atoms.copy()])
            interpolate(images)
            for img in images: img.calc = calculator
    else:
        interpolate(images)
        for img in images: img.calc = calculator

    start_climb = CLIMB and CLIMB_FROM_START
    neb = NEB(images, k=SPRING_K, climb=start_climb, method=NEB_METHOD,
              remove_rotation_and_translation=REMOVE_ROT_TRANS,
              allow_shared_calculator=True)
    optimizer = make_optimizer(OPTIMIZER, neb)

    step_count      = [0]
    climb_activated = [start_climb]

    def callback():
        step_count[0] += 1
        cur_e, cur_f = [], []
        for img in images:
            try:
                cur_e.append(float(img.get_potential_energy()))
                cur_f.append(float(np.max(np.linalg.norm(img.get_forces(), axis=1))))
            except Exception:
                cur_e.append(float("nan")); cur_f.append(float("nan"))
        valid = [v for v in cur_f if not np.isnan(v)]
        mf = max(valid) if valid else float("nan")

        if CLIMB and not CLIMB_FROM_START and not climb_activated[0] and mf < CLIMB_SWITCH_FMAX:
            neb.climb = True; climb_activated[0] = True
            print(f"  Step {{step_count[0]}}: CI-NEB activated (fmax={{mf:.4f}})")

        if step_count[0] % LOG_INTERVAL == 0 or step_count[0] == 1:
            ci = " [CI]" if climb_activated[0] else ""
            print(f"  -- Step {{step_count[0]:4d}}{{ci}}  fmax = {{mf:.4f}} eV/A --")
            print(f"    {{'Img':>4}}  {{'E (eV)':>14}}  {{'dE (meV)':>10}}  {{'fmax':>10}}")
            e0 = cur_e[0] if not np.isnan(cur_e[0]) else 0.0
            n_last = len(cur_e) - 1
            valid_inner = [v for jj,v in enumerate(cur_e)
                           if not np.isnan(v) and jj not in (0, n_last)]
            max_inner = max(valid_inner) if valid_inner else None
            for ii, (ee, ff) in enumerate(zip(cur_e, cur_f)):
                de  = (ee - e0) * 1000 if not np.isnan(ee) else float("nan")
                tag = " <- TS?" if (max_inner is not None and not np.isnan(ee)
                                    and ee == max_inner
                                    and ii not in (0, n_last)) else ""
                print(f"    {{ii:>4}}  {{ee:>14.6f}}  {{de:>10.2f}}  {{ff:>10.4f}}{{tag}}")

    optimizer.attach(callback, interval=1)
    print(f"\\nRunning {{OPTIMIZER}} (max {{MAX_STEPS}} steps, fmax={{FMAX}}) ...")
    optimizer.run(fmax=FMAX, steps=MAX_STEPS)
    print(f"Finished after {{step_count[0]}} steps")

    # Extract
    final_e, final_f, pos_list = [], [], []
    for img in images:
        final_e.append(float(img.get_potential_energy()))
        final_f.append(float(np.max(np.linalg.norm(img.get_forces(), axis=1))))
        pos_list.append(img.positions.copy())

    e_arr = np.array(final_e)
    dists = [0.0]
    for i in range(1, len(images)):
        dists.append(dists[-1] + mic_dist(pos_list[i-1], pos_list[i], images[i].cell.array))

    bi  = int(np.argmax(e_arr))
    fwd = float(e_arr[bi] - e_arr[0])
    rev = float(e_arr[bi] - e_arr[-1])
    rxn = float(e_arr[-1] - e_arr[0])

    result = dict(
        forward_barrier_eV=fwd, forward_barrier_kJ=fwd*96.4853,
        reverse_barrier_eV=rev, reverse_barrier_kJ=rev*96.4853,
        reaction_energy_eV=rxn, reaction_energy_kJ=rxn*96.4853,
        barrier_index=bi, n_images=N_IMAGES, steps=step_count[0],
        neb_method=NEB_METHOD, climb_activated=climb_activated[0],
        energies_eV=final_e, forces_eV_Ang=final_f,
        reaction_distances_Ang=dists,
        timestamp=datetime.now().isoformat(),
        parameters=dict(n_images=N_IMAGES, spring_k=SPRING_K,
                        neb_method=NEB_METHOD, interpolation=INTERPOLATION,
                        climb=CLIMB, fmax=FMAX, optimizer=OPTIMIZER))

    print(f"\\n  Forward barrier : {{fwd:.4f}} eV  ({{fwd*96.4853:.2f}} kJ/mol)")
    print(f"  Reverse barrier : {{rev:.4f}} eV  ({{rev*96.4853:.2f}} kJ/mol)")
    print(f"  Reaction energy : {{rxn:.4f}} eV  ({{rxn*96.4853:.2f}} kJ/mol)")
    print(f"  Saddle point    : image {{bi}}")
    print(f"  Path length     : {{dists[-1]:.3f}} A")

    with open("neb_results/neb_results.json","w") as fh:
        json.dump(result, fh, indent=2)
    print("  JSON  -> neb_results/neb_results.json")

    rows = [dict(image=i, energy_eV=round(float(e),8),
                 relative_energy_eV=round(float(e-e_arr[0]),8),
                 relative_energy_meV=round(float((e-e_arr[0])*1000),4),
                 max_force_eV_Ang=round(float(f),6), distance_Ang=round(float(d),6))
            for i,(e,f,d) in enumerate(zip(final_e,final_f,dists))]
    pd.DataFrame(rows).to_csv("neb_results/neb_energies.csv", index=False)
    print("  CSV   -> neb_results/neb_energies.csv")

    save_trajectory_xyz(images, "neb_results/neb_trajectory.xyz")
    save_images_poscar(images, "neb_results")
    write("neb_results/initial_optimised.vasp", images[0],  format="vasp", direct=True, sort=False)
    write("neb_results/final_optimised.vasp",   images[-1], format="vasp", direct=True, sort=False)
    plot_profile(e_arr, dists, bi, result, "neb_results/neb_profile.png")

    print(f"\\nTotal time: {{(time.time()-t0)/60:.1f}} min  |  Done! -> neb_results/")


if __name__ == "__main__":
    main()
'''


def generate_neb_script_minimal(neb_params, selected_model, model_size, device, dtype,
                                 thread_count=4, mace_head=None,
                                 mace_dispersion=False, mace_dispersion_xc='pbe',
                                 custom_mace_path=None):

    calc_src     = _calc_block(selected_model, model_size, device, dtype,
                               mace_head, mace_dispersion, mace_dispersion_xc,
                               custom_mace_path)
    climb        = neb_params.get('climb', True)
    ci_from_start= neb_params.get('climb_from_start', False)

    return f'''#!/usr/bin/env python3
"""
Minimal NEB script  (no matplotlib, no pandas — bare essentials only)
Generated : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model     : {selected_model}  |  Device: {device}

Place POSCAR_initial and POSCAR_final in the same directory and run:
    python neb_minimal.py

Output: neb_results/neb_results.json  +  neb_trajectory.xyz  +  per-image POSCARs
"""

import os, sys, json, time
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "{thread_count}"
import torch
torch.set_num_threads({thread_count})

from ase.io import read, write
from ase.mep import NEB, interpolate
from ase.optimize import BFGS, FIRE, LBFGS, LBFGSLineSearch, MDMin

Path("neb_results").mkdir(exist_ok=True)

N_IMAGES          = {int(neb_params["n_images"])}
SPRING_K          = {float(neb_params["spring_constant"])}
NEB_METHOD        = "{neb_params.get("neb_method", "improvedtangent")}"
INTERPOLATION     = "{neb_params.get("interpolation", "idpp")}"
CLIMB             = {climb}
CLIMB_FROM_START  = {ci_from_start}
CLIMB_SWITCH_FMAX = {float(neb_params.get("climb_switch_fmax", 0.5))}
FMAX              = {float(neb_params["fmax"])}
MAX_STEPS         = {int(neb_params["max_steps"])}
OPTIMIZER         = "{neb_params.get("optimizer", "LBFGS")}"
LOG_INTERVAL      = {int(neb_params.get("log_interval", 5))}
PRE_OPTIMIZE      = {neb_params.get("pre_optimize", True)}
PRE_OPT_FMAX      = {float(neb_params.get("pre_opt_fmax", 0.05))}
PRE_OPT_STEPS     = {int(neb_params.get("pre_opt_steps", 200))}


def make_optimizer(name, obj):
    n = name.upper()
    if n == "LBFGS":           return LBFGS(obj, logfile=None)
    if n == "LBFGSLINESEARCH": return LBFGSLineSearch(obj, logfile=None)
    if n == "FIRE":            return FIRE(obj, logfile=None)
    if n == "MDMIN":           return MDMin(obj, logfile=None)
    return BFGS(obj, logfile=None)


def setup_calculator():
{calc_src}    return calculator


def mic_dist(p1, p2, cell):
    d = p2 - p1
    f = np.linalg.solve(cell.T, d.T).T
    f -= np.round(f)
    return float(np.linalg.norm(f @ cell))


def find_structure(keyword):
    for f in sorted(os.listdir(".")):
        if keyword in f.lower() and (
            f.startswith("POSCAR") or
            f.endswith((".vasp", ".cif", ".xyz", ".poscar"))):
            return f
    return None


def main():
    t0 = time.time()
    print("NEB (minimal) |", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    init_file  = find_structure("initial")
    final_file = find_structure("final")
    if not init_file or not final_file:
        print("ERROR: need files with 'initial' and 'final' in the name")
        sys.exit(1)

    initial_atoms = read(init_file)
    final_atoms   = read(final_file)
    print(f"{{init_file}} + {{final_file}}  ({{len(initial_atoms)}} atoms)")

    calculator = setup_calculator()

    if PRE_OPTIMIZE:
        for atoms, lbl in [(initial_atoms,"Initial"), (final_atoms,"Final")]:
            atoms.calc = calculator
            BFGS(atoms, logfile=None).run(fmax=PRE_OPT_FMAX, steps=PRE_OPT_STEPS)
            e  = atoms.get_potential_energy()
            mf = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
            print(f"  {{lbl}}: E={{e:.6f}} eV  fmax={{mf:.4f}}")
    else:
        for a in (initial_atoms, final_atoms):
            a.calc = calculator

    images = ([initial_atoms.copy()]
              + [initial_atoms.copy() for _ in range(N_IMAGES - 2)]
              + [final_atoms.copy()])

    if INTERPOLATION == "idpp":
        for img in images: img.calc = calculator
        try:
            NEB(images, allow_shared_calculator=True).interpolate(method="idpp")
        except Exception:
            images = ([initial_atoms.copy()]
                      + [initial_atoms.copy() for _ in range(N_IMAGES - 2)]
                      + [final_atoms.copy()])
            interpolate(images)
            for img in images: img.calc = calculator
    else:
        interpolate(images)
        for img in images: img.calc = calculator

    start_climb = CLIMB and CLIMB_FROM_START
    neb = NEB(images, k=SPRING_K, climb=start_climb, method=NEB_METHOD,
              allow_shared_calculator=True)
    optimizer = make_optimizer(OPTIMIZER, neb)

    step_count      = [0]
    climb_activated = [start_climb]

    def callback():
        step_count[0] += 1
        cur_e, cur_f = [], []
        for img in images:
            try:
                cur_e.append(float(img.get_potential_energy()))
                cur_f.append(float(np.max(np.linalg.norm(img.get_forces(), axis=1))))
            except Exception:
                cur_e.append(float("nan")); cur_f.append(float("nan"))
        valid = [v for v in cur_f if not np.isnan(v)]
        mf = max(valid) if valid else float("nan")
        if CLIMB and not CLIMB_FROM_START and not climb_activated[0] and mf < CLIMB_SWITCH_FMAX:
            neb.climb = True; climb_activated[0] = True
            print(f"  Step {{step_count[0]}}: CI-NEB ON (fmax={{mf:.4f}})")
        if step_count[0] % LOG_INTERVAL == 0 or step_count[0] == 1:
            ci = " [CI]" if climb_activated[0] else ""
            print(f"  Step {{step_count[0]:4d}}{{ci}}  fmax={{mf:.4f}} eV/A")
            e0 = cur_e[0] if not np.isnan(cur_e[0]) else 0.0
            n_last = len(cur_e) - 1
            valid_inner = [v for jj,v in enumerate(cur_e)
                           if not np.isnan(v) and jj not in (0, n_last)]
            max_inner = max(valid_inner) if valid_inner else None
            for ii, (ee, ff) in enumerate(zip(cur_e, cur_f)):
                de  = (ee - e0) * 1000 if not np.isnan(ee) else float("nan")
                tag = " <- TS?" if (max_inner and not np.isnan(ee) and ee == max_inner
                                    and ii not in (0, n_last)) else ""
                print(f"    img{{ii:02d}}  E={{ee:.6f}}  dE={{de:+8.2f}} meV  f={{ff:.4f}}{{tag}}")

    optimizer.attach(callback, interval=1)
    print(f"Running {{OPTIMIZER}} ...")
    optimizer.run(fmax=FMAX, steps=MAX_STEPS)
    print(f"Finished after {{step_count[0]}} steps")

    final_e, final_f, pos_list = [], [], []
    for img in images:
        final_e.append(float(img.get_potential_energy()))
        final_f.append(float(np.max(np.linalg.norm(img.get_forces(), axis=1))))
        pos_list.append(img.positions.copy())

    e_arr = np.array(final_e)
    dists = [0.0]
    for i in range(1, len(images)):
        dists.append(dists[-1] + mic_dist(pos_list[i-1], pos_list[i], images[i].cell.array))

    bi  = int(np.argmax(e_arr))
    fwd = float(e_arr[bi] - e_arr[0])
    rev = float(e_arr[bi] - e_arr[-1])
    rxn = float(e_arr[-1] - e_arr[0])

    print(f"\\nForward barrier : {{fwd:.4f}} eV  ({{fwd*96.4853:.2f}} kJ/mol)")
    print(f"Reverse barrier : {{rev:.4f}} eV  ({{rev*96.4853:.2f}} kJ/mol)")
    print(f"Reaction energy : {{rxn:.4f}} eV  ({{rxn*96.4853:.2f}} kJ/mol)")
    print(f"Saddle point    : image {{bi}}")
    print(f"Path length     : {{dists[-1]:.3f}} A")

    result = dict(
        forward_barrier_eV=fwd, forward_barrier_kJ=fwd*96.4853,
        reverse_barrier_eV=rev, reverse_barrier_kJ=rev*96.4853,
        reaction_energy_eV=rxn, reaction_energy_kJ=rxn*96.4853,
        barrier_index=bi, n_images=N_IMAGES, steps=step_count[0],
        energies_eV=final_e, forces_eV_Ang=final_f,
        reaction_distances_Ang=dists, timestamp=datetime.now().isoformat())

    with open("neb_results/neb_results.json","w") as fh:
        json.dump(result, fh, indent=2)
    print("  JSON -> neb_results/neb_results.json")

    # XYZ trajectory — raw Cartesian positions, no wrapping
    with open("neb_results/neb_trajectory.xyz","w") as fh:
        for i, img in enumerate(images):
            pos  = img.positions
            cell = img.cell.array
            sym  = img.get_chemical_symbols()
            try:
                e  = img.get_potential_energy()
                f  = img.get_forces()
                mf = float(np.max(np.linalg.norm(f, axis=1)))
                has_f = True
            except Exception:
                e, mf, has_f = 0.0, 0.0, False
            lat = " ".join(f"{{x:.8f}}" for x in cell.flatten())
            props = "species:S:1:pos:R:3:forces:R:3" if has_f else "species:S:1:pos:R:3"
            fh.write(f"{{len(pos)}}\\n")
            fh.write(f'Image={{i}} Energy={{e:.8f}} Lattice="{{lat}}" '
                     f'Properties={{props}} pbc="T T T"\\n')
            for j in range(len(pos)):
                if has_f:
                    fh.write(f"{{sym[j]:2s}} {{pos[j,0]:16.8f}} {{pos[j,1]:16.8f}} {{pos[j,2]:16.8f}}"
                             f" {{f[j,0]:14.8f}} {{f[j,1]:14.8f}} {{f[j,2]:14.8f}}\\n")
                else:
                    fh.write(f"{{sym[j]:2s}} {{pos[j,0]:16.8f}} {{pos[j,1]:16.8f}} {{pos[j,2]:16.8f}}\\n")
    print("  XYZ  -> neb_results/neb_trajectory.xyz")

    # Per-image POSCARs
    for i, img in enumerate(images):
        label = "initial" if i==0 else ("final" if i==len(images)-1 else f"image_{{i:02d}}")
        write(f"neb_results/neb_{{label}}.vasp", img, format="vasp", direct=True, sort=False)
    print(f"  {{len(images)}} POSCARs -> neb_results/")

    print(f"\\nTime: {{(time.time()-t0)/60:.1f}} min  Done!")


if __name__ == "__main__":
    main()
'''


def render_neb_script_button(neb_params, selected_model, model_size, device, dtype,
                              thread_count=4, mace_head=None,
                              mace_dispersion=False, mace_dispersion_xc='pbe',
                              custom_mace_path=None):
    st.markdown("---")
    st.subheader("📝 Generate Standalone NEB Scripts")
    st.info(
        "Scripts look for files with **'initial'** and **'final'** in the name "
        "(e.g. `POSCAR_initial`, `initial.vasp`) and write everything to `neb_results/`.")

    kw = dict(neb_params=neb_params, selected_model=selected_model,
              model_size=model_size, device=device, dtype=dtype,
              thread_count=thread_count, mace_head=mace_head,
              mace_dispersion=mace_dispersion, mace_dispersion_xc=mace_dispersion_xc,
              custom_mace_path=custom_mace_path)

    col_a, col_b = st.columns(2)

    with col_a:
        st.write("**Full script** — CSV, matplotlib plot, per-image POSCARs")
        if st.button("🐍 Generate full NEB script", key="gen_neb_full_btn", type="secondary"):
            try:
                st.session_state['gen_neb_full'] = generate_neb_script(**kw)
                st.success("✅ Generated!")
            except Exception as exc:
                st.error(f"❌ {exc}")
        if st.session_state.get('gen_neb_full'):
            st.download_button(
                "💾 Download neb_calculation.py",
                data=st.session_state['gen_neb_full'],
                file_name="neb_calculation.py",
                mime="text/x-python", type="primary", key="dl_neb_full")
            with st.expander("👁 Preview full script", expanded=False):
                st.code(st.session_state['gen_neb_full'], language="python")

    with col_b:
        st.write("**Minimal script** — no matplotlib, no pandas")
        if st.button("🐍 Generate minimal NEB script", key="gen_neb_min_btn", type="secondary"):
            try:
                st.session_state['gen_neb_min'] = generate_neb_script_minimal(**kw)
                st.success("✅ Generated!")
            except Exception as exc:
                st.error(f"❌ {exc}")
        if st.session_state.get('gen_neb_min'):
            st.download_button(
                "💾 Download neb_minimal.py",
                data=st.session_state['gen_neb_min'],
                file_name="neb_minimal.py",
                mime="text/x-python", type="primary", key="dl_neb_min")
            with st.expander("👁 Preview minimal script", expanded=False):
                st.code(st.session_state['gen_neb_min'], language="python")

    st.markdown("""
**Output files in `neb_results/`:**

| File | Full | Minimal |
|------|:----:|:-------:|
| `neb_results.json` | ✅ | ✅ |
| `neb_trajectory.xyz` (raw Cartesian, pbc=TTT) | ✅ | ✅ |
| `neb_initial.vasp` … `neb_image_NN.vasp` … `neb_final.vasp` | ✅ | ✅ |
| `initial_optimised.vasp` / `final_optimised.vasp` | ✅ | ✅ |
| `neb_energies.csv` | ✅ | ❌ |
| `neb_profile.png` | ✅ | ❌ |
""")
