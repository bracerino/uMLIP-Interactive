"""
Finite-temperature elastic constants from molecular dynamics.

The calculation is the explicit stress-strain (direct) method:

  1. optional 0 K relaxation of the input cell,
  2. optional NPT pre-equilibration at (T, P) so the reference cell is the
     thermally expanded one — the time-averaged cell of the NPT production
     part is what the strains are applied to,
  3. for every Voigt component j a set of affine strains ±delta is applied to
     that reference cell and an NVT run at T is performed for each,
  4. the thermodynamic stress (virial + kinetic) is time-averaged over the
     production part of each run,
  5. C_ij = d<sigma_i> / d eps_j from a linear fit (or central difference),
     with block-averaged uncertainties propagated into every C_ij.

This gives the isothermal elastic constants at temperature T. The panel below
only collects the settings; the run itself happens in the standalone script
built by helpers/generate_finite_t_elastic_script.py.
"""

import math
import os

import numpy as np
import streamlit as st

# Mirror the main app's online-demo flag (set by online_app.py).
ONLINE_MODE = os.environ.get("MLIP_ONLINE_MODE", "0") == "1"

VOIGT_LABELS = ["xx", "yy", "zz", "yz", "xz", "xy"]

NVT_THERMOSTATS = ["Langevin", "Berendsen", "Nose-Hoover"]
NPT_BAROSTATS = [
    "Berendsen (isotropic)",
    "Berendsen (anisotropic)",
    "Nose-Hoover / Parrinello-Rahman",
]
SYMMETRY_MODES = [
    "Triclinic — all 6 strain components",
    "Cubic — 2 strain components (C11, C12, C44)",
    "Hexagonal — 3 strain components (C11, C12, C13, C33, C44)",
]

# Voigt components that actually have to be strained for each symmetry, and how
# many independent constants that yields. Everything else is filled in by the
# symmetry relations in the generated script.
SYMMETRY_STRAIN_COMPONENTS = {
    "triclinic": [0, 1, 2, 3, 4, 5],
    "cubic": [0, 3],
    "hexagonal": [0, 2, 3],
}


def _defaults():
    return {
        'temperature_start': 300.0,
        'temperature_end': 900.0,
        'temperature_step': 0.0,     # 0 = single temperature
        'pressure_GPa': 0.0,
        'timestep': 1.0,
        'supercell': [1, 1, 1],
        'seed': 42,

        # 0 K relaxation before anything thermal
        'pre_optimize': True,
        'pre_opt_optimizer': 'LBFGS',
        'pre_opt_fmax': 0.01,
        'pre_opt_steps': 300,
        'pre_opt_relax_cell': True,

        # NPT pre-equilibration
        'use_npt_equilibration': True,
        'npt_barostat': 'Berendsen (isotropic)',
        'npt_equilibration_steps': 5000,
        'npt_production_steps': 5000,
        'npt_ttime': 100.0,
        'npt_ptime': 1000.0,
        'npt_bulk_modulus': 140.0,
        'reference_cell': 'Time-averaged NPT cell',

        # strained NVT runs
        'nvt_thermostat': 'Langevin',
        'friction': 0.02,
        'thermostat_taut': 100.0,
        'nvt_equilibration_steps': 2000,
        'nvt_production_steps': 8000,
        'sample_interval': 10,
        'n_blocks': 5,

        # strain sampling
        'strain_magnitude': 0.01,
        'use_multi_strain': False,
        'symmetry_mode': SYMMETRY_MODES[0],
        'include_kinetic_stress': True,

        # extras
        'resume_from_checkpoint': True,
        'compute_static_reference': True,
        'static_ion_relax': True,
        'static_ion_fmax': 0.005,
        'static_ion_steps': 200,
        'log_interval': 200,
        'save_trajectories': False,
        'traj_interval': 500,
    }


def build_temperature_list(t_start, t_end, t_step):
    """Temperatures to run, from an initial / final / step specification.

    A step of 0 (or a final temperature at or below the initial one) means a
    single temperature. When the step does not divide the range evenly the final
    temperature is still appended, so the requested end point is always covered.
    """
    t_start = float(t_start)
    t_end = float(t_end)
    t_step = float(t_step)
    if t_step <= 0 or t_end <= t_start:
        return [t_start]
    n = int(math.floor((t_end - t_start) / t_step + 1e-9))
    temps = [t_start + i * t_step for i in range(n + 1)]
    if temps[-1] < t_end - 1e-9:
        temps.append(t_end)
    return [round(t, 6) for t in temps]


def symmetry_key(symmetry_mode):
    """Map the verbose radio label onto the short key used by the script."""
    label = str(symmetry_mode).lower()
    if label.startswith('cubic'):
        return 'cubic'
    if label.startswith('hexagonal'):
        return 'hexagonal'
    return 'triclinic'


def strain_magnitudes(params):
    """The strain values scanned for every strained Voigt component."""
    delta = float(params.get('strain_magnitude', 0.01))
    if params.get('use_multi_strain', False):
        return [-delta, -delta / 2.0, delta / 2.0, delta]
    return [-delta, delta]


def estimate_md_steps(params):
    """(number of MD runs, total MD steps) over the whole temperature sweep."""
    temps = params.get('temperature_list') or build_temperature_list(
        params.get('temperature_start', 300.0),
        params.get('temperature_end', 300.0),
        params.get('temperature_step', 0.0))
    components = SYMMETRY_STRAIN_COMPONENTS[symmetry_key(params.get('symmetry_mode'))]
    n_strained = len(components) * len(strain_magnitudes(params))

    nvt_steps = (int(params.get('nvt_equilibration_steps', 0))
                 + int(params.get('nvt_production_steps', 0)))
    npt_steps = 0
    if params.get('use_npt_equilibration', True):
        npt_steps = (int(params.get('npt_equilibration_steps', 0))
                     + int(params.get('npt_production_steps', 0)))

    runs_per_T = n_strained + (1 if npt_steps else 0)
    steps_per_T = n_strained * nvt_steps + npt_steps
    return len(temps) * runs_per_T, len(temps) * steps_per_T


def setup_finite_t_elastic_ui(default_settings=None, save_settings_function=None):
    st.subheader("Finite-Temperature Elastic Properties Parameters")
    st.caption(
        "Explicit stress–strain MD: the reference cell is equilibrated at (T, P) "
        "with NPT, each Voigt strain is then run in NVT and the time-averaged "
        "thermodynamic stress gives C_ij(T)."
    )

    defaults = _defaults()
    if default_settings and 'finite_t_elastic' in default_settings:
        stored = default_settings['finite_t_elastic']
        defaults.update({k: v for k, v in stored.items() if k in defaults})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🌡️ State point**")
        temperature_start = st.number_input(
            "Initial Temperature (K)",
            min_value=1.0,
            max_value=10000.0,
            value=float(defaults['temperature_start']),
            step=50.0,
            format="%.1f",
            help="The temperature C_ij is evaluated at (the first one of a sweep)."
        )

        temperature_step = st.number_input(
            "Temperature Step (K)",
            min_value=0.0,
            max_value=5000.0,
            value=float(defaults['temperature_step']),
            step=50.0,
            format="%.1f",
            help=(
                "Set to 0 for a single temperature — the final temperature is "
                "then ignored and locked. Any value above 0 turns this into a "
                "sweep: every temperature gets its own NPT pre-equilibration and "
                "its own full set of strained NVT runs, giving C_ij(T)."
            )
        )

        single_temperature = temperature_step <= 0.0
        temperature_end = st.number_input(
            "Final Temperature (K)",
            min_value=1.0,
            max_value=10000.0,
            value=(float(temperature_start) if single_temperature
                   else float(defaults['temperature_end'])),
            step=50.0,
            format="%.1f",
            disabled=single_temperature,
            help=("Locked while the step is 0 — a zero step means a single "
                  "temperature." if single_temperature else
                  "Last temperature of the sweep. It is always included, even "
                  "if the step does not divide the range evenly.")
        )

        temperature_list = build_temperature_list(
            temperature_start, temperature_end, temperature_step)
        if single_temperature:
            st.caption(f"🌡️ Single temperature: {temperature_start:g} K "
                       "(step = 0)")
        else:
            st.caption(f"📊 Sweep over {len(temperature_list)} temperatures: "
                       + ", ".join(f"{t:g}" for t in temperature_list) + " K")

        pressure_GPa = st.number_input(
            "External Pressure (GPa)",
            min_value=-50.0,
            max_value=500.0,
            value=float(defaults['pressure_GPa']),
            step=0.1,
            format="%.3f",
            help="Target pressure of the NPT pre-equilibration. 0 = ambient."
        )

        timestep = st.number_input(
            "Timestep (fs)",
            min_value=0.1,
            max_value=5.0,
            value=float(defaults['timestep']),
            step=0.1,
            format="%.1f",
            help="MD integration timestep, used for both the NPT and NVT parts."
        )

        sc = list(defaults['supercell']) + [1, 1, 1]
        st.markdown("**Supercell replication**")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            nx = st.number_input("nx", min_value=1, max_value=10,
                                 value=int(sc[0]), step=1, key="fte_nx")
        with sc2:
            ny = st.number_input("ny", min_value=1, max_value=10,
                                 value=int(sc[1]), step=1, key="fte_ny")
        with sc3:
            nz = st.number_input("nz", min_value=1, max_value=10,
                                 value=int(sc[2]), step=1, key="fte_nz")
        st.caption(
            "⚠️ Stress fluctuations scale as 1/√N — a cell of at least a few "
            "hundred atoms is needed before the averages settle."
        )

        seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=10_000_000,
            value=int(defaults['seed']),
            step=1,
            help="Seed for the Maxwell-Boltzmann velocities and the Langevin noise."
        )

    with col2:
        st.markdown("**⚖️ NPT pre-equilibration**")
        use_npt_equilibration = st.checkbox(
            "Pre-equilibrate the cell with NPT",
            value=bool(defaults['use_npt_equilibration']),
            help=(
                "Strongly recommended. Relaxing the cell at (T, P) captures thermal "
                "expansion, so the strains are measured around the true equilibrium "
                "cell at that temperature and the reference stress is ~0. Without it "
                "the input cell is used as-is and the constants are biased by the "
                "residual thermal pressure."
            )
        )

        if use_npt_equilibration:
            npt_barostat = st.selectbox(
                "Barostat",
                NPT_BAROSTATS,
                index=(NPT_BAROSTATS.index(defaults['npt_barostat'])
                       if defaults['npt_barostat'] in NPT_BAROSTATS else 0),
                help=(
                    "• Berendsen (isotropic): robust, scales the cell uniformly — "
                    "keeps the cell shape, best for cubic systems\n"
                    "• Berendsen (anisotropic): each cell axis relaxes separately — "
                    "needed for non-cubic cells (c/a relaxes too)\n"
                    "• Nose-Hoover / Parrinello-Rahman: full cell dynamics with "
                    "correct NPT sampling, also relaxes cell angles"
                )
            )

            npt_equilibration_steps = st.number_input(
                "NPT Equilibration Steps (discarded)",
                min_value=0,
                max_value=1_000_000,
                value=int(defaults['npt_equilibration_steps']),
                step=500,
                help="Burn-in of the NPT run; not used for the cell average."
            )

            npt_production_steps = st.number_input(
                "NPT Production Steps (cell average)",
                min_value=10,
                max_value=1_000_000,
                value=int(defaults['npt_production_steps']),
                step=500,
                help="The cell is averaged over these steps to define the reference cell."
            )

            reference_cell = st.radio(
                "Reference cell taken as",
                ["Time-averaged NPT cell", "Final NPT cell"],
                index=0 if defaults['reference_cell'] == 'Time-averaged NPT cell' else 1,
                help=(
                    "The time-averaged cell is far less noisy than any single "
                    "snapshot and is the standard choice."
                )
            )

            npt_ptime = st.number_input(
                "Barostat Time Constant (fs)",
                min_value=50.0,
                max_value=20000.0,
                value=float(defaults['npt_ptime']),
                step=100.0,
                help="taup (Berendsen) / ptime (Nose-Hoover). Should be well above the thermostat constant."
            )
            st.caption(
                "🌡️ The NPT thermostat reuses the *Thermostat Time Constant* "
                "from the NVT column (right)."
            )

            npt_bulk_modulus = st.number_input(
                "Estimated Bulk Modulus (GPa)",
                min_value=1.0,
                max_value=1000.0,
                value=float(defaults['npt_bulk_modulus']),
                step=10.0,
                help=(
                    "Only used by the Berendsen barostats, which need a "
                    "compressibility (1/B) to set the cell-scaling rate. A rough "
                    "value is fine."
                )
            )
        else:
            npt_barostat = defaults['npt_barostat']
            npt_equilibration_steps = 0
            npt_production_steps = 0
            reference_cell = 'Final NPT cell'
            npt_ptime = defaults['npt_ptime']
            npt_bulk_modulus = defaults['npt_bulk_modulus']
            st.warning(
                "⚠️ NPT pre-equilibration off — the input cell is strained directly. "
                "Only correct if the cell is already equilibrated at this temperature."
            )

    with col3:
        st.markdown("**🔥 Strained NVT runs**")
        nvt_thermostat = st.selectbox(
            "Thermostat",
            NVT_THERMOSTATS,
            index=(NVT_THERMOSTATS.index(defaults['nvt_thermostat'])
                   if defaults['nvt_thermostat'] in NVT_THERMOSTATS else 0),
            help=(
                "• Langevin: stochastic, robust, decorrelates quickly — best "
                "default for stress averaging\n"
                "• Berendsen: weak coupling, fast to equilibrate but not canonical\n"
                "• Nose-Hoover chain: deterministic canonical sampling"
            )
        )

        if nvt_thermostat == 'Langevin':
            friction = st.number_input(
                "Friction (1/fs)",
                min_value=0.001,
                max_value=1.0,
                value=float(defaults['friction']),
                step=0.001,
                format="%.3f",
                help="Langevin friction. Typical 0.01–0.05 1/fs."
            )
            thermostat_taut = float(defaults['thermostat_taut'])
        else:
            friction = float(defaults['friction'])
            thermostat_taut = st.number_input(
                "Thermostat Time Constant (fs)",
                min_value=10.0,
                max_value=5000.0,
                value=float(defaults['thermostat_taut']),
                step=10.0,
                help="taut (Berendsen) / tdamp (Nose-Hoover chain)."
            )

        nvt_equilibration_steps = st.number_input(
            "NVT Equilibration Steps (discarded)",
            min_value=0,
            max_value=1_000_000,
            value=int(defaults['nvt_equilibration_steps']),
            step=500,
            help=(
                "Run after each strain is applied and before stress sampling starts. "
                "Must be long enough for the stress to relax to its new plateau."
            )
        )

        nvt_production_steps = st.number_input(
            "NVT Production Steps (stress average)",
            min_value=10,
            max_value=1_000_000,
            value=int(defaults['nvt_production_steps']),
            step=500,
            help=(
                "Stress-averaging window. This is what sets the error bar on C_ij — "
                "the stress autocorrelation time is typically a few hundred fs. "
                "The minimum of 10 is there so you can do a quick end-to-end test "
                "run; anything that short is a plumbing check, not a result."
            )
        )

        sample_interval = st.number_input(
            "Stress Sampling Interval (steps)",
            min_value=1,
            max_value=1000,
            value=int(defaults['sample_interval']),
            step=1,
            help=(
                "Stress is recorded every N production steps. The stress "
                "autocorrelation time is short, so there is nothing to gain "
                "from sampling every step — and each sample costs one stress "
                "evaluation."
            )
        )

        n_blocks = st.number_input(
            "Averaging Blocks (error bars)",
            min_value=2,
            max_value=50,
            value=int(defaults['n_blocks']),
            step=1,
            help=(
                "The production stress series is split into this many blocks; the "
                "scatter between block means gives the standard error, which is "
                "propagated into every C_ij."
            )
        )

    st.markdown("---")
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("**📐 Strain sampling**")
        symmetry_mode = st.radio(
            "Crystal symmetry",
            SYMMETRY_MODES,
            index=(SYMMETRY_MODES.index(defaults['symmetry_mode'])
                   if defaults['symmetry_mode'] in SYMMETRY_MODES else 0),
            help=(
                "Choosing the symmetry of the crystal reduces how many Voigt "
                "components have to be strained; the remaining C_ij are filled in "
                "from the symmetry relations. Use Triclinic when unsure — it makes "
                "no assumptions and also reports how far the result is from the "
                "assumed symmetry."
            )
        )
        sym_key = symmetry_key(symmetry_mode)
        components = SYMMETRY_STRAIN_COMPONENTS[sym_key]
        st.caption(
            "Strained components: "
            + ", ".join(f"ε_{VOIGT_LABELS[j]}" for j in components)
        )

        strain_magnitude = st.number_input(
            "Strain Magnitude δ",
            min_value=0.001,
            max_value=0.05,
            value=float(defaults['strain_magnitude']),
            step=0.001,
            format="%.4f",
            help=(
                "Applied as ±δ. Too small and the stress difference drowns in the "
                "MD noise; too large and anharmonicity bends the response. "
                "0.005–0.02 is the usual window at finite temperature."
            )
        )

        use_multi_strain = st.checkbox(
            "Use 4 strain magnitudes (±δ/2, ±δ) instead of ±δ",
            value=bool(defaults['use_multi_strain']),
            help=(
                "Doubles the cost but fits a line through 4 points, which both "
                "averages down the MD noise and exposes non-linearity."
            )
        )
        st.caption(
            "Strain values per component: "
            + ", ".join(f"{d:+.4f}" for d in strain_magnitudes(
                {'strain_magnitude': strain_magnitude,
                 'use_multi_strain': use_multi_strain}))
        )

        include_kinetic_stress = st.checkbox(
            "Include the kinetic (ideal-gas) stress term",
            value=bool(defaults['include_kinetic_stress']),
            help=(
                "The thermodynamic stress is the virial plus the kinetic term "
                "−(1/V)Σ m v⊗v. Keep this on: it is part of the definition of the "
                "isothermal elastic constants and matters most for light elements "
                "and high temperatures."
            )
        )

    with col5:
        st.markdown("**🔧 Structure preparation & output**")
        pre_optimize = st.checkbox(
            "Relax the structure at 0 K first",
            value=bool(defaults['pre_optimize']),
            help="Removes residual forces/stress from the input before heating."
        )
        if pre_optimize:
            pre_opt_optimizer = st.selectbox(
                "Optimizer",
                ["LBFGS", "BFGS", "FIRE"],
                index=["LBFGS", "BFGS", "FIRE"].index(
                    defaults['pre_opt_optimizer'])
                if defaults['pre_opt_optimizer'] in ["LBFGS", "BFGS", "FIRE"] else 0,
            )
            pre_opt_fmax = st.number_input(
                "Force Convergence (eV/Å)",
                min_value=0.0001,
                max_value=1.0,
                value=float(defaults['pre_opt_fmax']),
                step=0.001,
                format="%.4f",
            )
            pre_opt_steps = st.number_input(
                "Max Optimization Steps",
                min_value=1,
                max_value=5000,
                value=int(defaults['pre_opt_steps']),
                step=50,
            )
            pre_opt_relax_cell = st.checkbox(
                "Relax the cell as well (variable-cell)",
                value=bool(defaults['pre_opt_relax_cell']),
                help="Uses a FrechetCellFilter so the 0 K cell is at its own equilibrium."
            )
        else:
            pre_opt_optimizer = defaults['pre_opt_optimizer']
            pre_opt_fmax = float(defaults['pre_opt_fmax'])
            pre_opt_steps = int(defaults['pre_opt_steps'])
            pre_opt_relax_cell = bool(defaults['pre_opt_relax_cell'])

        resume_from_checkpoint = st.checkbox(
            "Resume an interrupted run instead of recomputing",
            value=bool(defaults['resume_from_checkpoint']),
            help=(
                "Every finished piece of work — the 0 K relaxation, each "
                "temperature's NPT reference cell and each individual strained "
                "NVT run — is written to `elastic_md_results/*_checkpoint.json` "
                "as soon as it completes. If the run is killed, times out or "
                "crashes, just start the script again: it reloads what is "
                "already done and only computes what is missing.\n\n"
                "The checkpoint is tied to a fingerprint of the settings and the "
                "input structure, so changing anything that affects the numbers "
                "automatically starts a fresh calculation. Untick this to force "
                "everything to be recomputed."
            )
        )

        compute_static_reference = st.checkbox(
            "Also compute the 0 K (static) elastic tensor",
            value=bool(defaults['compute_static_reference']),
            help=(
                "Cheap extra reference: the same strains evaluated statically on "
                "the relaxed structure, so you can read off how much of C_ij is "
                "softened by temperature. Clamped-ion (the atoms are not relaxed "
                "inside the strained cell), so for crystals with internal degrees "
                "of freedom it is an upper bound — use the main *Elastic "
                "Properties* calculation type for a fully relaxed 0 K tensor."
            )
        )

        if compute_static_reference:
            static_ion_relax = st.checkbox(
                "Relax ions inside each strained cell (0 K reference)",
                value=bool(defaults['static_ion_relax']),
                help=(
                    "Keep this on. For any crystal whose basis has internal "
                    "degrees of freedom — hcp, diamond, most compounds — the "
                    "clamped-ion constants are systematically too stiff, badly "
                    "so for the shear constants, and the 0 K anchor drawn on "
                    "every summary plot would be wrong. It costs one short "
                    "optimisation per strained cell, which is negligible next "
                    "to the MD. Structures whose symmetry forbids internal "
                    "relaxation (fcc/bcc with a one-atom basis) are unaffected "
                    "either way."
                )
            )
            static_ion_fmax = st.number_input(
                "0 K Reference Force Convergence (eV/Å)",
                min_value=0.0001,
                max_value=0.1,
                value=float(defaults['static_ion_fmax']),
                step=0.001,
                format="%.4f",
                disabled=not static_ion_relax,
            )
        else:
            static_ion_relax = bool(defaults['static_ion_relax'])
            static_ion_fmax = float(defaults['static_ion_fmax'])

        log_interval = st.number_input(
            "Console Log Interval (steps)",
            min_value=1,
            max_value=10000,
            value=int(defaults['log_interval']),
            step=50,
            help="How often each MD run prints its progress."
        )

        save_trajectories = st.checkbox(
            "Save strained-run trajectories (.xyz)",
            value=bool(defaults['save_trajectories']),
            help="One extended-XYZ file per strain state. Useful for debugging, large on disk."
        )
        if save_trajectories:
            traj_interval = st.number_input(
                "Trajectory Interval (steps)",
                min_value=1,
                max_value=100000,
                value=int(defaults['traj_interval']),
                step=100,
            )
        else:
            traj_interval = int(defaults['traj_interval'])

    params = {
        'temperature_start': float(temperature_start),
        'temperature_end': float(temperature_end),
        'temperature_step': float(temperature_step),
        'temperature_list': temperature_list,
        'pressure_GPa': float(pressure_GPa),
        'timestep': float(timestep),
        'supercell': [int(nx), int(ny), int(nz)],
        'seed': int(seed),

        'pre_optimize': bool(pre_optimize),
        'pre_opt_optimizer': pre_opt_optimizer,
        'pre_opt_fmax': float(pre_opt_fmax),
        'pre_opt_steps': int(pre_opt_steps),
        'pre_opt_relax_cell': bool(pre_opt_relax_cell),

        'use_npt_equilibration': bool(use_npt_equilibration),
        'npt_barostat': npt_barostat,
        'npt_equilibration_steps': int(npt_equilibration_steps),
        'npt_production_steps': int(npt_production_steps),
        'npt_ttime': float(thermostat_taut),
        'npt_ptime': float(npt_ptime),
        'npt_bulk_modulus': float(npt_bulk_modulus),
        'reference_cell': reference_cell,

        'nvt_thermostat': nvt_thermostat,
        'friction': float(friction),
        'thermostat_taut': float(thermostat_taut),
        'nvt_equilibration_steps': int(nvt_equilibration_steps),
        'nvt_production_steps': int(nvt_production_steps),
        'sample_interval': int(sample_interval),
        'n_blocks': int(n_blocks),

        'strain_magnitude': float(strain_magnitude),
        'use_multi_strain': bool(use_multi_strain),
        'symmetry_mode': symmetry_mode,
        'symmetry': sym_key,
        'strain_components': list(components),
        'include_kinetic_stress': bool(include_kinetic_stress),

        'resume_from_checkpoint': bool(resume_from_checkpoint),
        'compute_static_reference': bool(compute_static_reference),
        'static_ion_relax': bool(static_ion_relax),
        'static_ion_fmax': float(static_ion_fmax),
        'static_ion_steps': int(defaults['static_ion_steps']),
        'log_interval': int(log_interval),
        'save_trajectories': bool(save_trajectories),
        'traj_interval': int(traj_interval),
    }

    if st.button("💾 Save as Default Finite-T Elastic Parameters",
                 key="save_finite_t_elastic_defaults", disabled=ONLINE_MODE):
        stored = {k: v for k, v in params.items() if k in _defaults()}
        if 'default_settings' not in st.session_state:
            st.session_state.default_settings = {}
        st.session_state.default_settings['finite_t_elastic'] = stored
        if save_settings_function and save_settings_function(st.session_state.default_settings):
            st.success("✅ Finite-T elastic parameters saved as default!")
        else:
            st.error("❌ Failed to save finite-T elastic parameters")

    n_runs, n_steps = estimate_md_steps(params)
    total_ps = n_steps * timestep / 1000.0
    st.info(
        f"Method: explicit stress–strain MD, "
        f"{len(components)} strained Voigt component(s) × "
        f"{len(strain_magnitudes(params))} strain magnitude(s)"
        f"{' × ' + str(len(temperature_list)) + ' temperatures' if len(temperature_list) > 1 else ''}\n\n"
        f"MD runs: {n_runs:,} — total {n_steps:,} MD steps ({total_ps:,.1f} ps of simulated time)\n\n"
        f"Reference cell: "
        + ("NPT-equilibrated at "
           f"{', '.join(f'{t:g}' for t in temperature_list)} K and {pressure_GPa:g} GPa"
           if use_npt_equilibration else "the input cell, unchanged")
    )

    return params
