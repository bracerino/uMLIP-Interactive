def generate_core_preview(calc_type, selected_model, model_size, device, dtype,
                          thread_count=4,
                          optimization_params=None, phonon_params=None,
                          elastic_params=None, md_params=None,
                          neb_params=None, tensile_params=None,
                          mace_head=None,
                          mace_dispersion=False, mace_dispersion_xc="pbe",
                          custom_mace_path=None, custom_upet_path=None):

    use_fairchem = False
    if md_params and md_params.get('use_fairchem', False):
        use_fairchem = True

    if use_fairchem:
        calc_snippet = _fairchem_calculator_snippet(md_params, device)
        effective_model_label = f"Fairchem/UMA: {md_params.get('fairchem_model_name', 'uma-s-1p1')}"
    else:
        calc_snippet = _calculator_snippet(
            selected_model, model_size, device, dtype,
            mace_head, mace_dispersion, mace_dispersion_xc,
            custom_mace_path, custom_upet_path,
        )
        effective_model_label = selected_model or model_size

    if calc_type == "Energy Only":
        body = _energy_only()
    elif calc_type == "Geometry Optimization":
        body = _geometry_optimization(optimization_params or {})
    elif calc_type == "Phonon Calculation":
        body = _phonon(phonon_params or {})
    elif calc_type == "Elastic Properties":
        body = _elastic(elastic_params or {})
    elif calc_type == "Molecular Dynamics":
        body = _molecular_dynamics(md_params or {})
    elif calc_type == "Virtual Tensile Test":
        body = _tensile(tensile_params or {})
    elif calc_type == "NEB Calculation":
        body = _neb(neb_params or {})
    elif calc_type == "GA Structure Optimization":
        body = _ga()
    else:
        body = _energy_only()

    return (
        f"# ── Core code: {calc_type} ──\n"
        f"#\n"
        f"# Model : {effective_model_label}\n"
        f"# Device: {device}  |  Precision: {dtype}\n"
        f"# Threads: {thread_count}\n"
        f"# ─────────────────────────────────────────────\n\n"
        f"import os\n"
        f"import numpy as np\n"
        f"from ase.io import read, write\n"
        f"from ase import Atoms, units\n\n"
        f"# Thread configuration\n"
        f"os.environ['OMP_NUM_THREADS'] = '{thread_count}'\n"
        f"import torch\n"
        f"torch.set_num_threads({thread_count})\n\n"
        f"# 1) Load your structure (any ASE-readable format)\n"
        f"atoms = read(\"your_structure.cif\")   # .vasp / POSCAR / .xyz / …\n\n"
        f"# 2) Set up calculator\n"
        f"{calc_snippet}\n"
        f"atoms.calc = calculator\n\n"
        f"# 3) Run calculation\n"
        f"{body}"
    )


def _fairchem_calculator_snippet(md_params, device):
    model_name = md_params.get('fairchem_model_name', 'uma-s-1p1')
    task = md_params.get('fairchem_task', 'omat')
    key = md_params.get('fairchem_key', '')
    lines = [
        "from fairchem.core import pretrained_mlip, FAIRChemCalculator",
    ]
    if key:
        lines += [
            "",
            f"os.environ['HF_TOKEN'] = \"{key}\"",
        ]
    lines += [
        "",
        f"predictor = pretrained_mlip.get_predict_unit(\"{model_name}\", device=\"{device}\")",
        f"calculator = FAIRChemCalculator(predictor, task_name=\"{task}\")",
    ]
    return "\n".join(lines)


def _calculator_snippet(selected_model, model_size, device, dtype,
                        mace_head, mace_dispersion, mace_dispersion_xc,
                        custom_mace_path, custom_upet_path):

    if custom_mace_path:
        args = [f'model="{custom_mace_path}"']
        if mace_head:
            args.append(f'head="{mace_head}"')
        if mace_dispersion:
            args += [f'dispersion=True', f'dispersion_xc="{mace_dispersion_xc}"']
        args += [f'default_dtype="{dtype}"', f'device="{device}"']
        return "from mace.calculators import mace_mp\ncalculator = mace_mp(\n    " + ",\n    ".join(args) + "\n)"

    if model_size.startswith("upet:"):
        if model_size == "upet:custom":
            path = custom_upet_path or "/path/to/model.ckpt"
            base = (
                f'from upet.calculator import UPETCalculator\n'
                f'calculator = UPETCalculator(checkpoint_path="{path}", device="{device}")'
            )
        else:
            parts = model_size.split(":")
            name = parts[1]
            version = parts[2] if len(parts) > 2 else "latest"
            base = (
                f'from upet.calculator import UPETCalculator\n'
                f'calculator = UPETCalculator(model="{name}", version="{version}", device="{device}")'
            )
        if mace_dispersion:
            base += (
                f'\n\nfrom torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator\n'
                f'from ase.calculators.mixing import SumCalculator\n'
                f'dft_d3 = TorchDFTD3Calculator(device="{device}", xc="{mace_dispersion_xc}", damping="bj")\n'
                f'calculator = SumCalculator([calculator, dft_d3])'
            )
        return base

    if "GRACE" in (selected_model or ""):
        return (
            f'from tensorpotential.calculator.foundation_models import grace_fm\n'
            f'calculator = grace_fm("{model_size}")'
        )

    if "Nequix" in (selected_model or ""):
        return (
            f'from nequix.calculator import NequixCalculator\n'
            f'calculator = NequixCalculator("{model_size}")'
        )

    if "CHGNet" in (selected_model or ""):
        ver = model_size.split("-")[1] if "-" in model_size else "0.3.0"
        return (
            f'torch.set_default_dtype(torch.float32)\n'
            f'from chgnet.model.model import CHGNet\n'
            f'from chgnet.model.dynamics import CHGNetCalculator\n'
            f'chgnet = CHGNet.load(model_name="{ver}", use_device="{device}")\n'
            f'calculator = CHGNetCalculator(model=chgnet, use_device="{device}")'
        )

    if "SevenNet" in (selected_model or ""):
        if model_size.startswith("7net-omni-"):
            modal = model_size.split("7net-omni-")[1]
            return (
                f'torch.serialization.add_safe_globals([slice])\n'
                f'from sevenn.calculator import SevenNetCalculator\n'
                f'calculator = SevenNetCalculator(model="7net-omni", modal="{modal}", device="{device}")'
            )
        if model_size == "7net-mf-ompa-mpa":
            return (
                f'from sevenn.calculator import SevenNetCalculator\n'
                f'calculator = SevenNetCalculator(model="7net-mf-ompa", modal="mpa", device="{device}")'
            )
        if model_size == "7net-mf-ompa-omat24":
            return (
                f'from sevenn.calculator import SevenNetCalculator\n'
                f'calculator = SevenNetCalculator(model="7net-mf-ompa", modal="omat24", device="{device}")'
            )
        return (
            f'from sevenn.calculator import SevenNetCalculator\n'
            f'calculator = SevenNetCalculator(model="{model_size}", device="{device}")'
        )

    if "MatterSim" in (selected_model or ""):
        path = "MatterSim-v1.0.0-5M.pth" if "5m" in model_size else "MatterSim-v1.0.0-1M.pth"
        return (
            f'from mattersim.forcefield import MatterSimCalculator\n'
            f'calculator = MatterSimCalculator(model_path="{path}", device="{device}")'
        )

    if "ORB" in (selected_model or ""):
        prec = "float32-high" if dtype == "float32" else "float32-highest"
        return (
            f'from orb_models.forcefield import pretrained\n'
            f'from orb_models.forcefield.calculator import ORBCalculator\n'
            f'orbff = pretrained.{model_size}(device="{device}", precision="{prec}")\n'
            f'calculator = ORBCalculator(orbff, device="{device}")'
        )

    if "OFF" in (selected_model or ""):
        return (
            f'from mace.calculators import mace_off\n'
            f'calculator = mace_off(model="{model_size}", default_dtype="{dtype}", device="{device}")'
        )

    if isinstance(model_size, str) and model_size.startswith("http"):
        args = [f'model="{model_size}"']
        if mace_head:
            args.append(f'head="{mace_head}"')
        if mace_dispersion:
            args += ['dispersion=True', f'dispersion_xc="{mace_dispersion_xc}"']
        args += [f'default_dtype="{dtype}"', f'device="{device}"']
        return "from mace.calculators import mace_mp\ncalculator = mace_mp(\n    " + ",\n    ".join(args) + "\n)"

    args = [f'model="{model_size}"']
    if mace_dispersion:
        args += ['dispersion=True', f'dispersion_xc="{mace_dispersion_xc}"']
    args += [f'default_dtype="{dtype}"', f'device="{device}"']
    return "from mace.calculators import mace_mp\ncalculator = mace_mp(\n    " + ",\n    ".join(args) + "\n)"


def _energy_only():
    return (
        "energy = atoms.get_potential_energy()\n"
        "forces = atoms.get_forces()\n"
        "stress = atoms.get_stress(voigt=True)\n\n"
        "print(f\"Energy:    {energy:.6f} eV\")\n"
        "print(f\"E/atom:    {energy / len(atoms):.6f} eV\")\n"
        "print(f\"Max force: {np.max(np.linalg.norm(forces, axis=1)):.4f} eV/Å\")\n"
    )


def _geometry_optimization(p):
    optimizer_name  = p.get('optimizer', 'BFGS')
    fmax            = p.get('fmax', 0.05)
    ediff           = p.get('ediff', 1e-4)
    max_steps       = p.get('max_steps', 200)
    opt_type        = p.get('optimization_type', 'Both atoms and cell')
    cell_constraint = p.get('cell_constraint', 'Lattice parameters only (fix angles)')
    pressure        = p.get('pressure', 0.0)
    hydrostatic     = p.get('hydrostatic_strain', False)
    optimize_lat    = p.get('optimize_lattice', {'a': True, 'b': True, 'c': True})
    fix_symmetry    = p.get('fix_symmetry', False)
    force_div       = p.get('force_divergence_threshold', 500)
    is_tetragonal   = (cell_constraint == "Tetragonal (a=b, optimize a and c)")

    opt_class = {
        'BFGS': 'BFGS', 'LBFGS': 'LBFGS', 'FIRE': 'FIRE',
        'BFGSLineSearch (QuasiNewton)': 'BFGSLineSearch',
        'LBFGSLineSearch': 'LBFGSLineSearch',
        'GoodOldQuasiNewton': 'GoodOldQuasiNewton',
        'MDMin': 'MDMin', 'GPMin': 'GPMin',
        'SciPyFminBFGS': 'SciPyFminBFGS',
        'SciPyFminCG': 'SciPyFminCG',
    }.get(optimizer_name, 'BFGS')

    if opt_class in ('SciPyFminBFGS', 'SciPyFminCG'):
        opt_import = f"from ase.optimize.sciopt import {opt_class}"
    else:
        opt_import = f"from ase.optimize import {opt_class}"

    lines = [opt_import]

    if fix_symmetry:
        lines += [
            "from ase.constraints import FixSymmetry",
            "from ase.spacegroup.symmetrize import check_symmetry",
            "",
            "spg = check_symmetry(atoms, symprec=1e-2, verbose=False)",
            "print(f\"Space group: {spg.get('international','?')} (#{spg.get('number','?')})\")",
            "existing = list(atoms.constraints) if atoms.constraints else []",
            "atoms.set_constraint(existing + [FixSymmetry(atoms)])",
        ]

    pressure_eV = pressure * 0.00624150913

    if opt_type == "Atoms only (fixed cell)":
        lines += ["", "opt_target = atoms"]
    elif opt_type == "Cell only (fixed atoms)":
        lines += [
            "",
            "from ase.constraints import FixAtoms",
            "existing = list(atoms.constraints) if atoms.constraints else []",
            "atoms.set_constraint(existing + [FixAtoms(indices=list(range(len(atoms))))])",
        ]
        lines += _cell_filter_lines(cell_constraint, pressure_eV, hydrostatic,
                                     optimize_lat, is_tetragonal)
    else:
        lines += [""]
        lines += _cell_filter_lines(cell_constraint, pressure_eV, hydrostatic,
                                     optimize_lat, is_tetragonal)

    if is_tetragonal and opt_type != "Atoms only (fixed cell)":
        lines += [
            "",
            "def enforce_tetragonal():",
            "    cell = atoms.get_cell().array.copy()",
            "    a_len = np.linalg.norm(cell[0])",
            "    b_len = np.linalg.norm(cell[1])",
            "    avg = (a_len + b_len) / 2.0",
            "    cell[0] = cell[0] / a_len * avg",
            "    cell[1] = cell[1] / b_len * avg",
            "    atoms.set_cell(cell, scale_atoms=True)",
            "    return a_len, b_len, avg",
        ]

    gpmin_extra = ", update_hyperparams=True" if opt_class == "GPMin" else ""
    lines += ["", f"opt = {opt_class}(opt_target, logfile=\"opt.log\"{gpmin_extra})"]

    if is_tetragonal and opt_type != "Atoms only (fixed cell)":
        lines += [
            "",
            f"fmax_crit    = {fmax}",
            f"ediff_crit   = {ediff}",
            f"stress_crit  = 0.1",
            f"max_steps    = {max_steps}",
            f"force_div_th = {force_div}",
            f"prev_energy  = None",
            "",
            "for step in range(max_steps):",
            "    opt.run(fmax=fmax_crit, steps=1)",
            "    old_a, old_b, new_ab = enforce_tetragonal()",
            "",
            "    cur = opt_target.atoms if hasattr(opt_target, 'atoms') else opt_target",
            "    forces   = cur.get_forces()",
            "    max_f    = float(np.max(np.linalg.norm(forces, axis=1)))",
            "    energy   = cur.get_potential_energy()",
            "",
            "    if max_f > force_div_th:",
            "        print(f\"Force divergence: {max_f:.2f} eV/Å\")",
            "        break",
            "",
            "    try:",
            "        max_s = float(np.max(np.abs(cur.get_stress(voigt=True))))",
            "    except Exception:",
            "        max_s = 0.0",
            "",
            "    dE = abs(energy - prev_energy) if prev_energy is not None else float('inf')",
            "    prev_energy = energy",
            "",
        ]
        if opt_type == "Cell only (fixed atoms)":
            lines.append("    converged = (max_s < stress_crit) and (dE < ediff_crit)")
        else:
            lines.append("    converged = (max_f < fmax_crit) and (max_s < stress_crit) and (dE < ediff_crit)")
        lines += [
            "",
            "    print(f\"Step {step+1}: E={energy:.6f} eV, F_max={max_f:.4f}, \"",
            "          f\"stress_max={max_s:.4f} GPa, dE={dE:.2e}, a=b={new_ab:.4f} Å\")",
            "",
            "    if converged:",
            "        print(\"Converged!\")",
            "        break",
        ]
    else:
        fmax_run = "0.1" if opt_type == "Cell only (fixed atoms)" else str(fmax)
        lines += [
            "",
            f"force_div_threshold = {force_div}",
            f"for _ in opt.irun(fmax={fmax_run}, steps={max_steps}):",
            f"    f_max = np.max(np.linalg.norm(atoms.get_forces(), axis=1))",
            f"    if f_max > force_div_threshold:",
            f"        print(f\"Force divergence: {{f_max:.2f}} eV/Å — stopping\")",
            f"        break",
        ]

    lines += [
        "",
        "final_atoms = opt_target.atoms if hasattr(opt_target, 'atoms') else opt_target",
        "energy    = final_atoms.get_potential_energy()",
        "max_force = np.max(np.linalg.norm(final_atoms.get_forces(), axis=1))",
        "print(f\"Final energy: {energy:.6f} eV\")",
        "print(f\"Final F_max:  {max_force:.4f} eV/Å\")",
        "write(\"optimized_POSCAR.vasp\", final_atoms, format=\"vasp\", direct=True)",
    ]

    return "\n".join(lines) + "\n"


def _cell_filter_lines(cell_constraint, pressure_eV, hydrostatic, optimize_lat, is_tetragonal):
    lines = []
    if is_tetragonal:
        lines += [
            "from ase.filters import UnitCellFilter",
            "opt_target = UnitCellFilter(atoms, mask=[True,True,True,False,False,False]"
            + (f", scalar_pressure={pressure_eV}" if pressure_eV else "")
            + ")",
        ]
    elif cell_constraint == "Full cell (lattice + angles)":
        lines += ["from ase.filters import ExpCellFilter"]
        args = []
        if pressure_eV:
            args.append(f"scalar_pressure={pressure_eV}")
        if hydrostatic:
            args.append("hydrostatic_strain=True")
        arg_str = ", ".join(args)
        lines.append(f"opt_target = ExpCellFilter(atoms, {arg_str})" if arg_str
                     else "opt_target = ExpCellFilter(atoms)")
    else:
        lines += ["from ase.filters import UnitCellFilter"]
        mask = [optimize_lat.get('a', True), optimize_lat.get('b', True),
                optimize_lat.get('c', True), False, False, False]
        args = [f"mask={mask}"]
        if pressure_eV:
            args.append(f"scalar_pressure={pressure_eV}")
        if hydrostatic:
            args.append("hydrostatic_strain=True")
        lines.append(f"opt_target = UnitCellFilter(atoms, {', '.join(args)})")
    return lines


def _phonon(p):
    auto_sc         = p.get('auto_supercell', True)
    target_len      = p.get('target_supercell_length', 15.0)
    max_mult        = p.get('max_supercell_multiplier', 4)
    max_atoms       = p.get('max_supercell_atoms', 800)
    sc              = p.get('supercell_size', (2, 2, 2))
    delta           = p.get('displacement_distance', p.get('delta', 0.01))
    npts            = p.get('npoints_per_segment', p.get('npoints', 101))
    use_auto_kpath  = p.get('use_auto_kpath', True)
    kpath_conv      = p.get('kpath_convention', 'setyawan_curtarolo')
    dos_mesh        = list(p.get('dos_mesh', (30, 30, 30)))
    pre_relax       = p.get('pre_relax', True)
    pre_relax_opt   = p.get('pre_relax_optimizer', 'LBFGS')
    pre_relax_fmax  = p.get('pre_relax_fmax', 0.01)
    pre_relax_steps = p.get('pre_relax_steps', 100)
    imag_tol        = p.get('imaginary_mode_tol_mev', -0.1)

    lines = [
        "from phonopy import Phonopy",
        "from phonopy.structure.atoms import PhonopyAtoms",
        "from pymatgen.io.ase import AseAtomsAdaptor",
    ]

    if pre_relax:
        lines += [
            "",
            f"from ase.optimize import {pre_relax_opt}",
            f"pre_opt = {pre_relax_opt}(atoms, logfile=None)",
            f"pre_opt.run(fmax={pre_relax_fmax}, steps={pre_relax_steps})",
        ]

    lines += ["", "pmg = AseAtomsAdaptor().get_structure(atoms)"]

    if auto_sc:
        lines += [
            "",
            "a, b, c = pmg.lattice.abc",
            f"na = max(1, min({max_mult}, int(np.ceil({target_len} / a))))",
            f"nb = max(1, min({max_mult}, int(np.ceil({target_len} / b))))",
            f"nc = max(1, min({max_mult}, int(np.ceil({target_len} / c))))",
            "if len(atoms) > 50:",
            "    na, nb, nc = max(1, na-1), max(1, nb-1), max(1, nc-1)",
            f"if len(atoms) * na * nb * nc > {max_atoms}:",
            "    na = nb = nc = 1",
            "sc_matrix = [[na,0,0],[0,nb,0],[0,0,nc]]",
        ]
    else:
        lines += ["", f"sc_matrix = [[{sc[0]},0,0],[0,{sc[1]},0],[0,0,{sc[2]}]]"]

    lines += [
        "",
        "ph_atoms = PhonopyAtoms(",
        "    symbols=[str(s.specie) for s in pmg],",
        "    scaled_positions=pmg.frac_coords,",
        "    cell=pmg.lattice.matrix,",
        ")",
        "phonon = Phonopy(ph_atoms, supercell_matrix=sc_matrix, primitive_matrix=\"auto\")",
        f"phonon.generate_displacements(distance={delta})",
        "",
        "forces = []",
        "for i, sc in enumerate(phonon.supercells_with_displacements):",
        "    sc_atoms = Atoms(symbols=sc.symbols, positions=sc.positions,",
        "                     cell=sc.cell, pbc=True)",
        "    sc_atoms.calc = calculator",
        "    forces.append(sc_atoms.get_forces())",
        "",
        "phonon.forces = forces",
        "phonon.produce_force_constants()",
    ]

    if use_auto_kpath:
        _CONV_MAP = {
            "setyawan_curtarolo": "Setyawan-Curtarolo",
            "hinuma": "SeeK-path (Hinuma)",
            "latimer_munro": "Latimer-Munro",
        }
        conv_label = _CONV_MAP.get(kpath_conv, kpath_conv)
        lines += [
            "",
            "from pymatgen.symmetry.bandstructure import HighSymmKpath",
            "from pymatgen.symmetry.analyzer import SpacegroupAnalyzer",
            "import warnings",
            "",
            "prim = phonon.primitive",
            "pmg_prim = AseAtomsAdaptor().get_structure(",
            "    Atoms(symbols=prim.symbols, scaled_positions=prim.scaled_positions,",
            "          cell=prim.cell, pbc=True))",
            "sga = SpacegroupAnalyzer(pmg_prim)",
            "with warnings.catch_warnings():",
            "    warnings.simplefilter('ignore')",
            f"    kpath_obj = HighSymmKpath(pmg_prim, path_type=\"{kpath_conv}\")",
            "",
            "path_segs    = kpath_obj.kpath['path']",
            "kpoints_dict = kpath_obj.kpath['kpoints']",
            f"npts = {npts}",
            "bands = []",
            "for seg in path_segs:",
            "    for i in range(len(seg) - 1):",
            "        start = np.array(kpoints_dict[seg[i]])",
            "        end   = np.array(kpoints_dict[seg[i+1]])",
            "        band  = [start + t/(npts-1) * (end - start) for t in range(npts)]",
            "        bands.append(band)",
        ]
    else:
        lines += [
            "",
            f"npts = {npts}",
            "segments = [([0,0,0],[0.5,0,0]), ([0.5,0,0],[0.5,0.5,0]), ([0.5,0.5,0],[0,0,0])]",
            "bands = []",
            "for start, end in segments:",
            "    s, e = np.array(start), np.array(end)",
            "    bands.append([s + t/(npts-1) * (e - s) for t in range(npts)])",
        ]

    lines += [
        "",
        "phonon.run_band_structure(bands, is_band_connection=True, with_eigenvectors=True)",
        "phonon.write_yaml_band_structure(filename=\"band.yaml\")",
        "",
        f"phonon.run_mesh({dos_mesh}, with_eigenvectors=False)",
        "phonon.run_total_dos()",
        "",
        "phonon.run_thermal_properties(t_step=10, t_max=1500, t_min=0)",
        "",
        "import phonopy.units as pu",
        "bd = phonon.get_band_structure_dict()",
        "freqs_meV = np.concatenate([np.array(s) for s in bd['frequencies']]) * pu.THzToEv * 1000",
        f"n_imag = int(np.sum(freqs_meV < {imag_tol}))",
        "print(f\"Imaginary modes: {n_imag}\")",
        "print(\"Dynamically stable\" if n_imag == 0 else \"Imaginary modes detected!\")",
    ]

    return "\n".join(lines) + "\n"


def _elastic(p):
    strain_mag    = p.get('strain_magnitude', 0.01)
    multi         = p.get('use_multi_strain', True)
    ionic         = p.get('ionic_relax_after_strain', True)
    ionic_fmax    = p.get('ionic_relax_fmax', 0.01)
    ionic_steps   = p.get('ionic_relax_steps', 50)
    ionic_opt     = p.get('ionic_relax_optimizer', 'LBFGS')
    pre_opt       = p.get('pre_optimize', True)
    pre_opt_steps = p.get('pre_opt_steps', 400)
    pre_opt_fmax  = p.get('pre_opt_fmax', 0.01)
    pre_opt_optim = p.get('pre_opt_optimizer', 'LBFGS')

    if multi:
        strains_repr = f"[-{strain_mag}, -{strain_mag}/2, 0.0, {strain_mag}/2, {strain_mag}]"
    else:
        strains_repr = f"[-{strain_mag}, {strain_mag}]"

    lines = []
    if pre_opt and pre_opt_steps > 0:
        lines += [
            f"from ase.optimize import {pre_opt_optim}",
            f"pre = {pre_opt_optim}(atoms, logfile=None)",
            f"pre.run(fmax={pre_opt_fmax}, steps={pre_opt_steps})",
            "",
        ]

    lines += [
        "eV_to_GPa = 160.21766208",
        f"strains = {strains_repr}",
        "original_cell = atoms.get_cell().copy()",
        "original_pos  = atoms.get_positions().copy()",
        "",
        "def voigt_strain(idx, d):",
        "    e = np.zeros((3, 3))",
        "    pairs = [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]",
        "    i, j = pairs[idx]",
        "    e[i, j] = d / (2 if idx >= 3 else 1)",
        "    if idx >= 3: e[j, i] = d / 2",
        "    return e",
        "",
        "C = np.zeros((6, 6))",
        "for j in range(6):",
        "    sigma_list = []",
        "    for d in strains:",
        "        T = np.eye(3) + voigt_strain(j, d)",
        "        atoms.set_cell(T @ original_cell.T, scale_atoms=False)",
        "        atoms.set_positions((T @ original_pos.T).T)",
        "        atoms.calc = calculator",
    ]
    if ionic:
        lines += [
            "",
            f"        from ase.optimize import {ionic_opt}",
            f"        ion_opt = {ionic_opt}(atoms, logfile=None)",
            f"        ion_opt.run(fmax={ionic_fmax}, steps={ionic_steps})",
            "        atoms.calc = calculator",
        ]
    lines += [
        "",
        "        sigma_list.append(atoms.get_stress(voigt=True))",
        "        atoms.set_cell(original_cell, scale_atoms=False)",
        "        atoms.set_positions(original_pos)",
        "        atoms.calc = calculator",
        "",
    ]
    if multi:
        lines += [
            "    for i in range(6):",
            "        slope, _ = np.polyfit(strains, [s[i] for s in sigma_list], 1)",
            "        C[i, j] = slope * eV_to_GPa",
        ]
    else:
        lines += [
            "    for i in range(6):",
            "        C[i, j] = (sigma_list[1][i] - sigma_list[0][i]) / (strains[1] - strains[0]) * eV_to_GPa",
        ]
    lines += [
        "",
        "C = (C + C.T) / 2",
        "",
        "K = (C[0,0]+C[1,1]+C[2,2] + 2*(C[0,1]+C[0,2]+C[1,2])) / 9",
        "G = (C[0,0]+C[1,1]+C[2,2] - C[0,1]-C[0,2]-C[1,2]",
        "     + 3*(C[3,3]+C[4,4]+C[5,5])) / 15",
        "E_mod = 9*K*G / (3*K + G)",
        "nu = (3*K - 2*G) / (2*(3*K + G))",
        "",
        "print(f\"Bulk  modulus (Voigt): {K:.1f} GPa\")",
        "print(f\"Shear modulus (Voigt): {G:.1f} GPa\")",
        "print(f\"Young's modulus:       {E_mod:.1f} GPa\")",
        "print(f\"Poisson's ratio:       {nu:.3f}\")",
        "print(f\"Mechanically stable:   {bool(np.all(np.linalg.eigvals(C) > 0))}\")",
    ]
    return "\n".join(lines) + "\n"


def _molecular_dynamics(p):
    ensemble        = p.get('ensemble', 'NVT-Langevin')
    temperature     = p.get('temperature', 300)
    timestep        = p.get('timestep', 1.0)
    n_steps         = p.get('n_steps', 10000)
    friction        = p.get('friction', 0.02)
    taut            = p.get('taut', 100.0)
    taup            = p.get('taup', p.get('pressure_damping_time', 1000.0))
    target_p        = p.get('target_pressure_gpa', 0.0)
    bulk_mod        = p.get('bulk_modulus', 140.0)
    log_interval    = p.get('log_interval', 10)
    traj_interval   = p.get('traj_interval', 100)
    remove_com      = p.get('remove_com_motion', True)
    coupling_type   = p.get('pressure_coupling_type', 'isotropic')
    fix_angles      = p.get('fix_angles', True)
    px              = p.get('pressure_x', target_p)
    py              = p.get('pressure_y', target_p)
    pz              = p.get('pressure_z', target_p)
    couple_x        = p.get('couple_x', True)
    couple_y        = p.get('couple_y', True)
    couple_z        = p.get('couple_z', True)

    is_npt = "NPT" in ensemble

    lines = [
        "from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary",
        "",
        f"MaxwellBoltzmannDistribution(atoms, temperature_K={temperature})",
    ]
    if remove_com:
        lines.append("Stationary(atoms)")

    lines += ["", f"dt = {timestep} * units.fs"]

    if ensemble == "NVE":
        lines += [
            "from ase.md import VelocityVerlet",
            "",
            "dyn = VelocityVerlet(atoms, timestep=dt)",
        ]

    elif ensemble == "NVT-Langevin":
        lines += [
            "from ase.md import Langevin",
            "",
            f"dyn = Langevin(atoms, timestep=dt,",
            f"               temperature_K={temperature},",
            f"               friction={friction} / units.fs)",
        ]

    elif ensemble == "NVT-Berendsen":
        lines += [
            "from ase.md.nvtberendsen import NVTBerendsen",
            "",
            f"dyn = NVTBerendsen(atoms, timestep=dt,",
            f"                   temperature_K={temperature},",
            f"                   taut={taut} * units.fs)",
        ]

    elif ensemble == "NPT (Berendsen)":
        lines += [
            "from ase.md.nptberendsen import NPTBerendsen",
            "",
            f"compressibility_au = 1.0 / ({bulk_mod} * units.GPa)",
        ]
        if coupling_type == 'anisotropic':
            lines += [
                f"pressure_factor = 1e9 * units.Pascal",
                f"pressure_au = np.diag([{px} * pressure_factor,",
                f"                       {py} * pressure_factor,",
                f"                       {pz} * pressure_factor])",
            ]
        else:
            lines.append(f"pressure_au = {target_p} * 1e9 * units.Pascal")
        lines += [
            "",
            f"dyn = NPTBerendsen(atoms, timestep=dt,",
            f"                   temperature_K={temperature},",
            f"                   pressure_au=pressure_au,",
            f"                   taut={taut} * units.fs,",
            f"                   taup={taup} * units.fs,",
            f"                   compressibility_au=compressibility_au)",
        ]

    elif ensemble == "NPT (Nose-Hoover)":
        lines += [
            "from ase.md.npt import NPT as NPTNoseHoover",
            "",
            f"ttime_fs = {taut} * units.fs",
            f"taup_fs  = {taup} * units.fs",
            f"pfactor  = (taup_fs**2) * {bulk_mod} * units.GPa",
        ]
        if coupling_type == 'anisotropic':
            lines += [
                f"externalstress = np.array([{px}, {py}, {pz}, 0.0, 0.0, 0.0]) * units.GPa",
            ]
            if fix_angles:
                lines.append("mask = np.diag([1, 1, 1])")
            else:
                lines.append("mask = None")
        elif coupling_type == 'directional':
            lines += [
                f"externalstress = {target_p} * units.GPa",
            ]
            if fix_angles:
                lines.append(f"mask = np.diag([{int(couple_x)}, {int(couple_y)}, {int(couple_z)}])")
            else:
                lines.append(f"mask = [{int(couple_x)}, {int(couple_y)}, {int(couple_z)}]")
        else:
            lines.append(f"externalstress = {target_p} * units.GPa")
            if fix_angles:
                lines.append("mask = np.diag([1, 1, 1])")
            else:
                lines.append("mask = None")
        lines += [
            "",
            f"dyn = NPTNoseHoover(atoms, timestep=dt,",
            f"                    temperature_K={temperature},",
            f"                    externalstress=externalstress,",
            f"                    ttime=ttime_fs,",
            f"                    pfactor=pfactor,",
            f"                    mask=mask)",
        ]

    elif ensemble == "NPT (MTK Isotropic)":
        lines += [
            "from ase.md.nose_hoover_chain import IsotropicMTKNPT",
            "",
            f"dyn = IsotropicMTKNPT(atoms, timestep=dt,",
            f"                      temperature_K={temperature},",
            f"                      pressure_au={target_p} * units.GPa,",
            f"                      tdamp={taut} * units.fs,",
            f"                      pdamp={taup} * units.fs)",
        ]

    elif ensemble == "NPT (MTK Full)":
        lines += [
            "from ase.md.nose_hoover_chain import MTKNPT",
            "",
        ]
        if coupling_type == "semi_isotropic":
            lines += [
                f"dyn = MTKNPT(atoms, timestep=dt,",
                f"             temperature_K={temperature},",
                f"             pressure_au={target_p} * units.GPa,",
                f"             tdamp={taut} * units.fs,",
                f"             pdamp={taup} * units.fs,",
                f"             mask=np.array([1, 1, 1, 0, 0, 0]))",
            ]
        else:
            lines += [
                f"dyn = MTKNPT(atoms, timestep=dt,",
                f"             temperature_K={temperature},",
                f"             pressure_au={target_p} * units.GPa,",
                f"             tdamp={taut} * units.fs,",
                f"             pdamp={taup} * units.fs)",
            ]

    elif ensemble == "NPT (BAOAB Langevin)":
        lines += [
            "from ase.md.langevinbaoab import LangevinBAOAB",
            "",
            f"externalstress = -{target_p} * units.GPa",
            f"dyn = LangevinBAOAB(atoms, timestep=dt,",
            f"                    temperature_K={temperature},",
            f"                    externalstress=externalstress,",
            f"                    T_tau={taut} * units.fs,",
            f"                    P_tau={taup} * units.fs)",
        ]

    elif "Melchionna" in ensemble:
        lines += [
            "from ase.md.melchionna import MelchionnaNPT",
            "",
            f"ttime_fs = {taut} * units.fs",
            f"taup_fs  = {taup} * units.fs",
            f"pfactor  = (taup_fs**2) * {bulk_mod} * units.GPa",
            f"externalstress = -{target_p} * units.GPa",
            "",
            f"dyn = MelchionnaNPT(atoms, timestep=dt,",
            f"                    temperature_K={temperature},",
            f"                    externalstress=externalstress,",
            f"                    ttime=ttime_fs,",
            f"                    pfactor=pfactor)",
        ]

    elif ensemble == "NVT":
        lines += [
            "from ase.md import Langevin",
            f"dyn = Langevin(atoms, timestep=dt, temperature_K={temperature},",
            f"               friction={friction} / units.fs)",
        ]
    elif ensemble == "NPT":
        lines += [
            "from ase.md.nptberendsen import NPTBerendsen",
            f"compressibility_au = 1.0 / ({bulk_mod} * units.GPa)",
            f"dyn = NPTBerendsen(atoms, timestep=dt, temperature_K={temperature},",
            f"                   pressure_au={target_p} * 1e9 * units.Pascal,",
            f"                   taut={taut} * units.fs, taup={taup} * units.fs,",
            f"                   compressibility_au=compressibility_au)",
        ]
    else:
        lines += [f"raise ValueError(\"Unknown ensemble: {ensemble}\")"]

    total_time_ps = n_steps * timestep / 1000.0
    lines += [
        "",
        f"traj_file = open(\"md_trajectory.xyz\", \"w\")",
        f"csv_file  = open(\"md_data.csv\", \"w\")",
    ]

    if is_npt:
        lines.append(
            "csv_file.write(\"step,time_ps,E_pot_eV,E_kin_eV,E_tot_eV,T_K,P_GPa,"
            "a_A,b_A,c_A,alpha_deg,beta_deg,gamma_deg,volume_A3\\n\")"
        )
    else:
        lines.append(
            "csv_file.write(\"step,time_ps,E_pot_eV,E_kin_eV,E_tot_eV,T_K\\n\")"
        )

    lines += [
        "",
        f"for step in range({n_steps}):",
        f"    dyn.run(1)",
        "",
        f"    if step % {log_interval} == 0:",
        f"        T  = atoms.get_temperature()",
        f"        E  = atoms.get_potential_energy()",
        f"        Ek = atoms.get_kinetic_energy()",
        f"        time_ps = step * {timestep} / 1000.0",
    ]

    if is_npt:
        lines += [
            f"        stress = atoms.get_stress(voigt=True)",
            f"        P = -np.mean(stress[:3]) / units.GPa",
            f"        cell_params = atoms.get_cell().cellpar()",
            f"        vol = atoms.get_volume()",
            f"        print(f\"Step {{step}}: E_pot={{E:.4f}} eV, E_kin={{Ek:.4f}} eV, "
            f"T={{T:.1f}} K, P={{P:.2f}} GPa\")",
            f"        csv_file.write(",
            f"            f\"{{step}},{{time_ps:.4f}},{{E:.6f}},{{Ek:.6f}},{{E+Ek:.6f}},{{T:.2f}},{{P:.4f}},\"",
            f"            f\"{{cell_params[0]:.5f}},{{cell_params[1]:.5f}},{{cell_params[2]:.5f}},\"",
            f"            f\"{{cell_params[3]:.3f}},{{cell_params[4]:.3f}},{{cell_params[5]:.3f}},\"",
            f"            f\"{{vol:.4f}}\\n\"",
            f"        )",
        ]
    else:
        lines += [
            f"        print(f\"Step {{step}}: E_pot={{E:.4f}} eV, E_kin={{Ek:.4f}} eV, T={{T:.1f}} K\")",
            f"        csv_file.write(f\"{{step}},{{time_ps:.4f}},{{E:.6f}},{{Ek:.6f}},{{E+Ek:.6f}},{{T:.2f}}\\n\")",
        ]

    lines += [
        "",
        f"    if step % {traj_interval} == 0:",
        f"        positions = atoms.get_positions()",
        f"        forces    = atoms.get_forces()",
        f"        symbols   = atoms.get_chemical_symbols()",
        f"        n         = len(atoms)",
        f"        energy    = atoms.get_potential_energy()",
        f"        temp      = atoms.get_temperature()",
        f"        cell      = atoms.get_cell()",
        f"        lat_str   = \" \".join(f\"{{x:.8f}}\" for x in cell.array.flatten())",
        f"        time_fs   = step * {timestep}",
        f"",
        f"        traj_file.write(f\"{{n}}\\n\")",
        f"        traj_file.write(",
        f"            f'Step={{step}} Time={{time_fs:.3f}}fs Energy={{energy:.6f}}eV '",
        f"            f'Temp={{temp:.2f}}K Lattice=\"{{lat_str}}\" '",
        f"            f'Properties=species:S:1:pos:R:3:forces:R:3\\n'",
        f"        )",
        f"        for i in range(n):",
        f"            traj_file.write(",
        f"                f\"{{symbols[i]}} {{positions[i,0]:15.8f}} {{positions[i,1]:15.8f}} {{positions[i,2]:15.8f}} \"",
        f"                f\"{{forces[i,0]:15.8f}} {{forces[i,1]:15.8f}} {{forces[i,2]:15.8f}}\\n\"",
        f"            )",
        f"        traj_file.flush()",
    ]

    lines += [
        "",
        "traj_file.close()",
        "csv_file.close()",
        "write(\"final_md_structure.vasp\", atoms, format=\"vasp\")",
        "print(f\"Trajectory saved: md_trajectory.xyz\")",
        "print(f\"Data log saved:   md_data.csv\")",
        "print(f\"Final structure:  final_md_structure.vasp\")",
    ]

    return "\n".join(lines) + "\n"


def _tensile(p):
    direction     = p.get('strain_direction', 0)
    strain_rate   = p.get('strain_rate', 0.1)
    max_strain    = p.get('max_strain', 10.0)
    temperature   = p.get('temperature', 300)
    timestep      = p.get('timestep', 1.0)
    friction      = p.get('friction', 0.01)
    eq_steps      = p.get('equilibration_steps', 200)
    sample_int    = p.get('sample_interval', 10)
    relax_between = p.get('relax_between_strain', False)
    relax_steps   = p.get('relax_steps', 100)

    dir_label = ['x', 'y', 'z'][direction]

    lines = [
        "from ase.md.langevin import Langevin",
        "from ase.md.velocitydistribution import MaxwellBoltzmannDistribution",
        "",
        f"MaxwellBoltzmannDistribution(atoms, temperature_K={temperature})",
        f"dyn = Langevin(atoms, timestep={timestep} * units.fs,",
        f"               temperature_K={temperature}, friction={friction} / units.fs)",
        "",
        f"dyn.run({eq_steps})",
        "",
        f"cell0 = atoms.get_cell().copy()",
        f"strain_increment = {strain_rate / 100} * {timestep} * 1e-3",
        f"total_md_steps = int({max_strain / 100} / strain_increment)",
        "",
        "stresses, strains = [], []",
        "for step in range(total_md_steps):",
        "    cell = atoms.get_cell().copy()",
        f"    cell[{direction}] *= (1 + strain_increment)",
        "    atoms.set_cell(cell, scale_atoms=True)",
        "    atoms.calc = calculator",
        f"    dyn.run({sample_int})",
    ]
    if relax_between:
        lines += [
            "",
            f"    from ase.optimize import LBFGS",
            f"    opt = LBFGS(atoms, logfile=None)",
            f"    opt.run(fmax=0.05, steps={relax_steps})",
            f"    atoms.calc = calculator",
        ]
    lines += [
        "",
        f"    strain_pct = (np.linalg.norm(atoms.get_cell()[{direction}]) /",
        f"                  np.linalg.norm(cell0[{direction}]) - 1) * 100",
        f"    stress_GPa = atoms.get_stress(voigt=True)[{direction}] * (-160.21766208)",
        "    stresses.append(stress_GPa)",
        "    strains.append(strain_pct)",
        f"    if step % 50 == 0:",
        f"        print(f\"Strain {{strain_pct:.2f}}%: stress = {{stress_GPa:.2f}} GPa\")",
    ]
    return "\n".join(lines) + "\n"


def _neb(p):
    n_images  = p.get('n_images', 7)
    fmax      = p.get('fmax', 0.05)
    max_steps = p.get('max_steps', 200)
    climb     = p.get('climb', True)
    optimizer = p.get('optimizer', 'FIRE')
    spring_k  = p.get('spring_constant', 0.1)
    method    = p.get('method', 'aseneb')
    pre_relax = p.get('pre_relax_endpoints', True)
    pre_fmax  = p.get('pre_relax_fmax', 0.02)
    pre_steps = p.get('pre_relax_steps', 100)

    lines = [
        "from ase.neb import NEB",
        f"from ase.optimize import {optimizer}",
        "",
        "initial = read(\"POSCAR_initial\")",
        "final   = read(\"POSCAR_final\")",
        "initial.calc = calculator",
        "final.calc   = calculator",
    ]
    if pre_relax:
        lines += [
            "",
            "from ase.optimize import LBFGS",
            f"LBFGS(initial, logfile=None).run(fmax={pre_fmax}, steps={pre_steps})",
            f"LBFGS(final,   logfile=None).run(fmax={pre_fmax}, steps={pre_steps})",
        ]
    lines += [
        "",
        f"images = [initial.copy() for _ in range({n_images} + 2)]",
        "images[-1] = final.copy()",
        "",
        f"neb = NEB(images, climb={climb}, k={spring_k}, method=\"{method}\")",
        "neb.interpolate()",
        "",
        "for img in images[1:-1]:",
        "    img.calc = calculator",
        "",
        f"opt = {optimizer}(neb, logfile=\"neb.log\")",
        f"opt.run(fmax={fmax}, steps={max_steps})",
        "",
        "energies = [img.get_potential_energy() for img in images]",
        "barrier_fwd = max(energies) - energies[0]",
        "barrier_rev = max(energies) - energies[-1]",
        "print(f\"Forward barrier: {barrier_fwd:.4f} eV\")",
        "print(f\"Reverse barrier: {barrier_rev:.4f} eV\")",
    ]
    return "\n".join(lines) + "\n"


def _ga():
    return (
        "# GA Structure Optimisation — core loop outline\n"
        "#\n"
        "# The full GA implementation involves ~500 lines.\n"
        "# Use 'Generate Python Script' for the complete, runnable version.\n"
        "#\n"
        "# Core algorithm:\n"
        "#   population = [random_substitution_pattern(base_structure)\n"
        "#                 for _ in range(pop_size)]\n"
        "#   for generation in range(max_generations):\n"
        "#       fitness = [calc_energy(ind, calculator) for ind in population]\n"
        "#       elite   = select_best(population, fitness, ratio=0.1)\n"
        "#       offspring = []\n"
        "#       while len(offspring) < pop_size - len(elite):\n"
        "#           p1, p2 = tournament_select(population, fitness)\n"
        "#           child  = crossover(p1, p2)\n"
        "#           child  = mutate(child)\n"
        "#           offspring.append(child)\n"
        "#       population = elite + offspring\n"
        "#   best = min(population, key=lambda ind: calc_energy(ind, calculator))\n"
    )
