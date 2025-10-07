import streamlit as st

# Default geometry optimization settings
DEFAULT_GEOMETRY_SETTINGS = {
    'optimizer': "BFGS",
    'fmax': 0.05,
    'ediff': 1e-4,
    'max_steps': 200,
    'optimization_type': "Both atoms and cell",
    'cell_constraint': "Lattice parameters only (fix angles)",
    'optimize_lattice': {'a': True, 'b': True, 'c': True},
    'pressure': 0.0,
    'hydrostatic_strain': False,
    'stress_threshold': 0.1
}


def save_geometry_settings(current_settings, new_geometry_params, save_function):
    updated_settings = current_settings.copy()
    updated_settings['geometry_optimization'] = new_geometry_params

    return save_function(updated_settings)


def setup_geometry_optimization_ui(default_settings, cell_opt_available, save_settings_function):

    # Load saved geometry optimization settings
    geom_defaults = default_settings.get('geometry_optimization', DEFAULT_GEOMETRY_SETTINGS)

    st.subheader("Optimization Parameters")

    if not cell_opt_available:
        st.error("⚠️ Cell optimization features require ASE constraints. Some features may be limited.")

    col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)

    with col_opt1:
        optimizer_options = ["BFGS", "LBFGS", "FIRE"]
        optimizer_index = 0  # Default to BFGS
        if geom_defaults['optimizer'] in optimizer_options:
            optimizer_index = optimizer_options.index(geom_defaults['optimizer'])

        optimizer = st.selectbox(
            "Optimizer",
            optimizer_options,
            index=optimizer_index,
            help="BFGS: More memory but faster convergence. LBFGS: Less memory usage. FIRE: Good for difficult surfaces."
        )

    with col_opt2:
        fmax = st.number_input(
            "Force threshold (eV/Å)",
            min_value=0.001,
            max_value=1.0,
            value=geom_defaults['fmax'],
            step=0.005,
            format="%.3f",
            help="Convergence criterion for maximum force on any atom"
        )

    with col_opt3:
        ediff = st.number_input(
            "Energy threshold (eV) (only for monitoring, not a convergence parameter)",
            min_value=1e-6,
            max_value=1e-2,
            value=geom_defaults['ediff'],
            step=1e-5,
            format="%.1e",
            help="Convergence criterion for energy change between steps"
        )

    with col_opt4:
        max_steps = st.number_input(
            "Max steps",
            min_value=10,
            max_value=1000,
            value=geom_defaults['max_steps'],
            step=10,
            help="Maximum number of optimization steps"
        )

    st.subheader("Cell Optimization Parameters")

    optimization_types = ["Atoms only (fixed cell)", "Cell only (fixed atoms)", "Both atoms and cell"]
    opt_type_index = optimization_types.index(geom_defaults['optimization_type']) if geom_defaults[
                                                                                         'optimization_type'] in optimization_types else 2

    optimization_type = st.radio(
        "What to optimize:",
        optimization_types,
        index=opt_type_index,
        help="Choose whether to optimize atomic positions, cell parameters, or both"
    )

    cell_constraint = None
    optimize_lattice = {'a': True, 'b': True, 'c': True}
    pressure = 0.0
    hydrostatic_strain = False
    stress_threshold = 0.1

    if optimization_type in ["Cell only (fixed atoms)", "Both atoms and cell"]:
        st.write("**Cell Parameter Constraints:**")

        col_cell1, col_cell2 = st.columns(2)

        with col_cell1:
            constraint_options = [
                "Lattice parameters only (fix angles)",
                "Full cell (lattice + angles)",
                "Fix a=b, optimize a and c"
            ]
            if geom_defaults['cell_constraint'] == constraint_options[0]:
                constraint_index = 0
            elif geom_defaults['cell_constraint'] == constraint_options[1]:
                constraint_index = 1
            elif geom_defaults['cell_constraint'] == "Fix a=b, optimize a and c":
                constraint_index = 2
            else:
                constraint_index = 0

            cell_constraint = st.radio(
                "Cell optimization mode:",
                constraint_options,
                index=constraint_index,
                help="Choose whether to optimize only lattice parameters or also angles"
            )

        with col_cell2:
            if cell_constraint == "Lattice parameters only (fix angles)":
                st.write("**Lattice directions to optimize:**")
                optimize_a = st.checkbox("Optimize a-direction", value=geom_defaults['optimize_lattice']['a'])
                optimize_b = st.checkbox("Optimize b-direction", value=geom_defaults['optimize_lattice']['b'])
                optimize_c = st.checkbox("Optimize c-direction", value=geom_defaults['optimize_lattice']['c'])

                optimize_lattice = {
                    'a': optimize_a,
                    'b': optimize_b,
                    'c': optimize_c
                }

                if not any([optimize_a, optimize_b, optimize_c]):
                    st.warning("⚠️ At least one lattice direction must be optimized!")
            elif cell_constraint == "Fix a=b, optimize a and c":
                st.info(
                    "ℹ️ Constraint: a=b will be maintained during optimization. Both a(=b) and c will be optimized. Angles remain fixed.")
                optimize_lattice = {'a': True, 'b': True, 'c': True}
            else:
                optimize_lattice = {'a': True, 'b': True, 'c': True}

        col_press1, col_press2, col_press3 = st.columns(3)
        with col_press1:
            pressure = st.number_input(
                "External pressure (GPa)",
                min_value=0.0,
                max_value=100.0,
                value=geom_defaults['pressure'],
                step=0.1,
                format="%.1f",
                help="External pressure for cell optimization (0 = atmospheric pressure)"
            )
        with col_press2:
            hydrostatic_strain = st.checkbox(
                "Hydrostatic strain only (preserve cell shape)",
                value=geom_defaults['hydrostatic_strain'],
                help="Constrain cell to change hydrostatically (preserve shape)"
            )
        with col_press3:
            stress_threshold = st.number_input(
                "Stress threshold (GPa)",
                min_value=0.001,
                max_value=1.0,
                value=geom_defaults['stress_threshold'],
                step=0.01,
                format="%.3f",
                help="Maximum stress for convergence"
            )
    save_trajectory = st.checkbox(
        "Save optimization trajectory (.xyz) (⚠️ Currently, do not turn it off for calculations within GUI)",
        value=True,
        help="Save step-by-step trajectory in XYZ format. Disable for large batches to save disk space."
    )


    optimization_params = {
        'optimizer': optimizer,
        'fmax': fmax,
        'ediff': ediff,
        'max_steps': max_steps,
        'optimization_type': optimization_type,
        'cell_constraint': cell_constraint,
        'optimize_lattice': optimize_lattice,
        'pressure': pressure,
        'hydrostatic_strain': hydrostatic_strain,
        'stress_threshold': stress_threshold,
        'save_trajectory': save_trajectory
    }

    if st.button("💾 Save Geometry Optimization Defaults", key="save_geom_defaults"):
        success = save_geometry_settings(
            st.session_state.default_settings,
            optimization_params,
            save_settings_function
        )

        if success:
            st.session_state.default_settings['geometry_optimization'] = optimization_params
            st.toast("✅ Geometry optimization defaults saved!")
        else:
            st.toast("❌ Failed to save geometry optimization defaults")

    return optimization_params


def display_optimization_info(optimization_params):
    if optimization_params['optimization_type'] == "Atoms only (fixed cell)":
        st.info(f"Optimization will adjust atomic positions only with forces < {optimization_params['fmax']} eV/Å")
    elif optimization_params['optimization_type'] == "Cell only (fixed atoms)":
        constraint_text = optimization_params.get('cell_constraint', 'lattice parameters only')
        pressure_text = f" at {optimization_params['pressure']} GPa" if optimization_params['pressure'] > 0 else ""
        hydro_text = " (hydrostatic)" if optimization_params.get('hydrostatic_strain') else ""
        st.info(f"Optimization will adjust {constraint_text} only{pressure_text}{hydro_text}")
    else:
        constraint_text = optimization_params.get('cell_constraint', 'lattice parameters only')
        pressure_text = f" at {optimization_params['pressure']} GPa" if optimization_params['pressure'] > 0 else ""
        hydro_text = " (hydrostatic)" if optimization_params.get('hydrostatic_strain') else ""
        st.info(
            f"Optimization will adjust both atoms (F < {optimization_params['fmax']} eV/Å) and {constraint_text}{pressure_text}{hydro_text}")
