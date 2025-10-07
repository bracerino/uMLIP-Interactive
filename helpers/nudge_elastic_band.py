import streamlit as st
import numpy as np
from ase.mep import NEB, interpolate
from ase.mep.neb import idpp_interpolate
from ase.optimize import BFGS, FIRE
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

    col_neb1, col_neb2, col_neb3, col_neb4 = st.columns(4)

    with col_neb1:
        neb_params['n_images'] = st.number_input(
            "Number of Images",
            min_value=3,
            max_value=20,
            value=7,
            step=1,
            help="Number of intermediate images between initial and final states"
        )

    with col_neb2:
        neb_params['spring_constant'] = st.number_input(
            "Spring Constant (eV/Å)",
            min_value=0.01,
            max_value=10.0,
            value=0.1,
            step=0.01,
            format="%.2f",
            help="Spring constant for the elastic band"
        )

    with col_neb3:
        neb_params['climb'] = st.checkbox(
            "Climbing Image",
            value=True,
            help="Use climbing image method for accurate saddle point location"
        )

    with col_neb4:
        neb_params['fmax'] = st.number_input(
            "Force Convergence (eV/Å)",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.01,
            format="%.3f",
            help="Maximum force for convergence"
        )

    col_neb5, col_neb6, col_neb7, col_neb8 = st.columns(4)

    with col_neb5:
        neb_params['max_steps'] = st.number_input(
            "Maximum Steps",
            min_value=100,
            max_value=5000,
            value=500,
            step=50,
            help="Maximum optimization steps"
        )

    with col_neb6:
        neb_params['optimizer'] = st.selectbox(
            "Optimizer",
            ["BFGS", "FIRE"],
            index=0,
            help="Optimization algorithm"
        )

    with col_neb7:
        neb_params['interpolation'] = st.selectbox(
            "Interpolation Method",
            ["linear", "idpp"],
            index=1,
            help="Method for generating intermediate images"
        )

    with col_neb8:
        neb_params['remove_rotation'] = st.checkbox(
            "Remove Rotation",
            value=True,
            help="Remove rotation and translation from images"
        )

    st.info("NEB will compute the minimum energy path between initial and final states")

    return neb_params


def interpolate_images(initial, final, n_images, method='linear'):
    from ase.mep import interpolate

    images = [initial.copy()]

    for i in range(n_images - 2):
        image = initial.copy()
        images.append(image)

    images.append(final.copy())

    if method == 'idpp':
        try:
            from ase.mep.idpp import IDPP
            idpp_images = [img.copy() for img in images]
            idpp = IDPP(idpp_images)
            idpp.interpolate()
            images = idpp_images
        except:
            interpolate(images)
    else:
        interpolate(images)

    return images


def run_neb_calculation(initial_structure, final_structure, calculator, neb_params, log_queue, stop_event,
                        structure_name):
    try:
        from ase.io import Trajectory
        from ase.mep import interpolate
        #from ase.mep.neb import SingleCalculatorNEB

        log_queue.put(f"Starting NEB calculation: {structure_name}")

        adaptor = AseAtomsAdaptor()
        initial_atoms = adaptor.get_atoms(initial_structure)
        final_atoms = adaptor.get_atoms(final_structure)

        # Pre-optimize endpoints to ensure they're at minima
        log_queue.put("  Pre-optimizing initial structure...")
        initial_atoms.calc = calculator
        opt_initial = BFGS(initial_atoms, logfile=None)
        opt_initial.run(fmax=0.05, steps=100)
        initial_energy = initial_atoms.get_potential_energy()
        log_queue.put(f"  Initial structure optimized: E = {initial_energy:.6f} eV")

        log_queue.put("  Pre-optimizing final structure...")
        final_atoms.calc = calculator
        opt_final = BFGS(final_atoms, logfile=None)
        opt_final.run(fmax=0.05, steps=100)
        final_energy = final_atoms.get_potential_energy()
        log_queue.put(f"  Final structure optimized: E = {final_energy:.6f} eV")

        n_images = neb_params['n_images']
        log_queue.put(f"  Creating {n_images} images...")

        # Create image list
        images = [initial_atoms]
        for i in range(n_images - 2):
            images.append(initial_atoms.copy())
        images.append(final_atoms)

        # Interpolate with error handling
        log_queue.put(f"  Interpolating {n_images} images using {neb_params['interpolation']} method...")

        if neb_params['interpolation'] == 'idpp':
            try:
                # Attach calculators to all images
                for img in images:
                    img.calc = calculator

                # Method 1: Use NEB's interpolate method (recommended in ASE 3.23+)
                neb_temp = NEB(images)
                neb_temp.interpolate(method='idpp')

                log_queue.put("  ✓ IDPP interpolation successful")
            except Exception as idpp_error:
                log_queue.put(f"  ⚠ IDPP interpolation failed: {str(idpp_error)}")
                log_queue.put("  Falling back to linear interpolation...")
                # Use linear interpolation as fallback
                interpolate(images)
                # Re-attach calculators
                for img in images:
                    img.calc = calculator
        else:
            # Linear interpolation
            interpolate(images)
            for img in images:
                img.calc = calculator

        # Create separate calculator instances for each image
        log_queue.put(f"  Using SingleCalculatorNEB (shared calculator mode for ML potentials)...")

        # Ensure all images have the calculator attached
        for img in images:
            img.calc = calculator

        use_single_calculator = True


        # Create NEB object
        log_queue.put(f"  Creating NEB object with k={neb_params['spring_constant']} eV/Å...")

        for img in images:
            img.calc = calculator

        neb = NEB(
            images,
            k=neb_params['spring_constant'],
            climb=neb_params['climb'],
            remove_rotation_and_translation=neb_params['remove_rotation'],
            allow_shared_calculator=True
        )

        # Setup optimizer
        if neb_params['optimizer'] == 'FIRE':
            optimizer = FIRE(neb, logfile=None)
        else:
            optimizer = BFGS(neb, logfile=None)

        log_queue.put(f"  Starting NEB optimization with {neb_params['optimizer']}...")

        # Use proper trajectory file
        import tempfile
        import os
        traj_path = os.path.join(tempfile.gettempdir(),
                                 f'neb_{structure_name.replace(".", "_").replace(" ", "_")}.traj')
        traj = Trajectory(traj_path, 'w', neb)
        optimizer.attach(traj.write, interval=1)

        # Tracking variables
        trajectory_data = []
        energies_history = []
        step_count = [0]

        def neb_callback():
            step_count[0] += 1

            if stop_event.is_set():
                raise Exception("Calculation stopped by user")

            try:
                current_energies = []
                current_forces = []

                for i, image in enumerate(images):
                    try:
                        energy = image.get_potential_energy()
                        forces = image.get_forces()
                        max_force = np.max(np.linalg.norm(forces, axis=1))

                        current_energies.append(energy)
                        current_forces.append(max_force)
                    except Exception as e:
                        current_energies.append(np.nan)
                        current_forces.append(np.nan)

                energies_history.append(current_energies)

                if step_count[0] % 10 == 0 or step_count[0] == 1:
                    valid_forces = [f for f in current_forces if not np.isnan(f)]
                    if valid_forces:
                        max_force = np.max(valid_forces)
                        log_queue.put(f"    Step {step_count[0]}: Max force = {max_force:.4f} eV/Å")

                        log_queue.put({
                            'type': 'neb_step',
                            'structure': structure_name,
                            'step': step_count[0],
                            'max_steps': neb_params['max_steps'],
                            'energies': current_energies,
                            'forces': current_forces,
                            'max_force': max_force
                        })
            except Exception as callback_error:
                pass

        optimizer.attach(neb_callback, interval=1)

        # Run optimization
        optimizer.run(fmax=neb_params['fmax'], steps=neb_params['max_steps'])

        # Close trajectory file
        traj.close()

        log_queue.put(f"  Optimization completed in {step_count[0]} steps")

        # Extract final results
        final_energies = []
        final_forces = []
        final_positions = []
        final_cells = []

        for i, image in enumerate(images):
            energy = image.get_potential_energy()
            forces = image.get_forces()
            max_force = np.max(np.linalg.norm(forces, axis=1))

            final_energies.append(energy)
            final_forces.append(max_force)
            final_positions.append(image.positions.copy())
            final_cells.append(image.cell.array.copy())

            trajectory_data.append({
                'image_index': i,
                'energy': float(energy),
                'max_force': float(max_force),
                'positions': image.positions.copy().tolist(),
                'cell': image.cell.array.copy().tolist(),
                'symbols': image.get_chemical_symbols(),
                'forces': forces.copy().tolist()
            })

        final_energies = np.array(final_energies)
        reaction_coordinate = np.arange(len(final_energies))

        reaction_distances = [0.0]
        cumulative_distance = 0.0

        for i in range(1, len(images)):
            pos_prev = np.array(final_positions[i - 1])
            pos_curr = np.array(final_positions[i])

            displacement = pos_curr - pos_prev
            distance = np.sqrt(np.sum(displacement ** 2))

            cumulative_distance += distance
            reaction_distances.append(cumulative_distance)

        reaction_distances = np.array(reaction_distances)

        # Calculate barriers
        max_energy = np.max(final_energies)
        barrier_index = np.argmax(final_energies)

        forward_barrier = max_energy - final_energies[0]
        reverse_barrier = max_energy - final_energies[-1]
        reaction_energy = final_energies[-1] - final_energies[0]

        forward_barrier_kJ = forward_barrier * 96.4853
        reverse_barrier_kJ = reverse_barrier * 96.4853
        reaction_energy_kJ = reaction_energy * 96.4853

        log_queue.put(
            f"  ✅ NEB completed: Forward barrier = {forward_barrier:.3f} eV ({forward_barrier_kJ:.1f} kJ/mol)")
        log_queue.put(f"     Reverse barrier = {reverse_barrier:.3f} eV ({reverse_barrier_kJ:.1f} kJ/mol)")
        log_queue.put(f"     Reaction energy = {reaction_energy:.3f} eV ({reaction_energy_kJ:.1f} kJ/mol)")
        log_queue.put(f"     Saddle point at image {barrier_index}")

        # Clean up temporary trajectory file
        try:
            if os.path.exists(traj_path):
                os.remove(traj_path)
        except:
            pass

        return {
            'success': True,
            'energies': [float(e) for e in final_energies],
            'forces': [float(f) for f in final_forces],
            'reaction_coordinate': [int(x) for x in reaction_coordinate],
            'reaction_distances': [float(d) for d in reaction_distances],
            'forward_barrier_eV': float(forward_barrier),
            'reverse_barrier_eV': float(reverse_barrier),
            'reaction_energy_eV': float(reaction_energy),
            'forward_barrier_kJ': float(forward_barrier_kJ),
            'reverse_barrier_kJ': float(reverse_barrier_kJ),
            'reaction_energy_kJ': float(reaction_energy_kJ),
            'barrier_index': int(barrier_index),
            'trajectory_data': trajectory_data,
            'energies_history': energies_history,
            'n_images': int(n_images),
            'converged_steps': int(step_count[0]),
            'neb_params': neb_params
        }

    except Exception as e:
        log_queue.put(f"❌ NEB calculation failed: {str(e)}")
        import traceback
        log_queue.put(f"Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e)
        }


def create_neb_trajectory_xyz(trajectory_data, structure_name):
    xyz_content = ""

    for step_data in trajectory_data:
        image_idx = step_data['image_index']
        energy = step_data['energy']
        positions = np.array(step_data['positions'])
        cell = np.array(step_data['cell'])
        symbols = step_data['symbols']
        max_force = step_data['max_force']

        forces = step_data.get('forces', None)
        if forces is not None:
            forces = np.array(forces)

        cell_inv = np.linalg.inv(cell)
        frac_coords = positions @ cell_inv
        wrapped_frac_coords = frac_coords % 1.0
        wrapped_positions = wrapped_frac_coords @ cell

        n_atoms = len(wrapped_positions)
        xyz_content += f"{n_atoms}\n"

        cell_flat = cell.flatten()
        lattice_str = " ".join([f"{x:.6f}" for x in cell_flat])

        if forces is not None:
            comment = (f'Image={image_idx} Energy={energy:.6f} Max_Force={max_force:.6f} '
                       f'Lattice="{lattice_str}" '
                       f'Properties=species:S:1:pos:R:3:forces:R:3')
        else:
            comment = (f'Image={image_idx} Energy={energy:.6f} Max_Force={max_force:.6f} '
                       f'Lattice="{lattice_str}" '
                       f'Properties=species:S:1:pos:R:3')

        xyz_content += f"{comment}\n"

        for i in range(n_atoms):
            symbol = symbols[i]
            pos = wrapped_positions[i]

            if forces is not None:
                force = forces[i]
                xyz_content += f"{symbol} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f} {force[0]:12.6f} {force[1]:12.6f} {force[2]:12.6f}\n"
            else:
                xyz_content += f"{symbol} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n"

    return xyz_content


def create_neb_plot(neb_results, structure_name, use_distance=False):
    energies = np.array(neb_results['energies'])
    reaction_coord = np.array(neb_results['reaction_coordinate'])

    energies_rel = (energies - energies[0]) * 1000

    barrier_idx = neb_results['barrier_index']

    if use_distance and 'reaction_distances' in neb_results:
        x_values = neb_results['reaction_distances']
        x_title = "Distance along path (Å)"
    else:
        x_values = reaction_coord
        x_title = "Reaction Coordinate (Image Number)"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_values,
        y=energies_rel,
        mode='lines+markers',
        name='NEB Path',
        line=dict(width=3, color='blue'),
        marker=dict(size=10, color='blue'),
        hovertemplate='<b>Image %{customdata}</b><br>Distance: %{x:.2f} Å<br>Energy: %{y:.2f} meV<extra></extra>' if use_distance else '<b>Image %{x}</b><br>Energy: %{y:.2f} meV<extra></extra>',
        customdata=reaction_coord if use_distance else None
    ))

    fig.add_trace(go.Scatter(
        x=[x_values[0]],
        y=[energies_rel[0]],
        mode='markers',
        name='Initial State',
        marker=dict(size=15, color='green', symbol='star'),
        hovertemplate='<b>Initial State</b><br>Energy: %{y:.2f} meV<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=[x_values[-1]],
        y=[energies_rel[-1]],
        mode='markers',
        name='Final State',
        marker=dict(size=15, color='red', symbol='star'),
        hovertemplate='<b>Final State</b><br>Energy: %{y:.2f} meV<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=[x_values[barrier_idx]],
        y=[energies_rel[barrier_idx]],
        mode='markers',
        name='Transition State',
        marker=dict(size=18, color='orange', symbol='diamond'),
        hovertemplate='<b>Transition State</b><br>Image: %{customdata}<br>Barrier: %{y:.2f} meV<extra></extra>' if use_distance else '<b>Transition State</b><br>Image: %{x}<br>Barrier: %{y:.2f} meV<extra></extra>',
        customdata=[barrier_idx] if use_distance else None
    ))

    fig.update_layout(
        title=dict(text=f"NEB Energy Profile: {structure_name}", font=dict(size=24)),
        xaxis_title=x_title,
        yaxis_title="Relative Energy (meV)",
        height=600,
        font=dict(size=18),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            font=dict(size=16),
            x=1.02,
            y=1
        ),
        xaxis=dict(
            tickfont=dict(size=16),
            title_font=dict(size=20)
        ),
        yaxis=dict(
            tickfont=dict(size=16),
            title_font=dict(size=20)
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Initial State",
        annotation_position="right"
    )

    return fig


def create_combined_neb_plot(all_neb_results, use_distance=False):
    fig = go.Figure()

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']

    for idx, (name, neb_result) in enumerate(all_neb_results.items()):
        if neb_result['success']:
            energies = np.array(neb_result['energies'])
            reaction_coord = np.array(neb_result['reaction_coordinate'])

            energies_rel = (energies - energies[0]) * 1000

            color = colors[idx % len(colors)]

            if use_distance and 'reaction_distances' in neb_result:
                x_values = neb_result['reaction_distances']
                x_title = "Distance along path (Å)"
            else:
                x_values = reaction_coord
                x_title = "Reaction Coordinate (Image Number)"

            fig.add_trace(go.Scatter(
                x=x_values,
                y=energies_rel,
                mode='lines+markers',
                name=name,
                line=dict(width=2, color=color),
                marker=dict(size=8),
                hovertemplate=f'<b>{name}</b><br>Image: %{{customdata}}<br>Distance: %{{x:.2f}} Å<br>Energy: %{{y:.2f}} meV<extra></extra>' if use_distance else f'<b>{name}</b><br>Image: %{{x}}<br>Energy: %{{y:.2f}} meV<extra></extra>',
                customdata=reaction_coord if use_distance else None
            ))

            barrier_idx = neb_result['barrier_index']
            fig.add_trace(go.Scatter(
                x=[x_values[barrier_idx]],
                y=[energies_rel[barrier_idx]],
                mode='markers',
                name=f'{name} TS',
                marker=dict(size=12, color=color, symbol='diamond'),
                showlegend=False,
                hovertemplate=f'<b>{name} TS</b><br>Barrier: {neb_result["forward_barrier_eV"]:.3f} eV<extra></extra>'
            ))

    fig.update_layout(
        title=dict(text="Combined NEB Energy Profiles", font=dict(size=24)),
        xaxis_title=x_title,
        yaxis_title="Relative Energy (meV)",
        height=700,
        font=dict(size=18),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            font=dict(size=14),
            x=1.02,
            y=1
        ),
        xaxis=dict(
            tickfont=dict(size=16),
            title_font=dict(size=20)
        ),
        yaxis=dict(
            tickfont=dict(size=16),
            title_font=dict(size=20)
        )
    )

    return fig


def export_neb_results(neb_result, structure_name):
    export_data = {
        'structure_name': structure_name,
        'calculation_type': 'Nudged Elastic Band',
        'timestamp': datetime.now().isoformat(),
        'forward_barrier_eV': float(neb_result['forward_barrier_eV']),
        'forward_barrier_kJ_mol': float(neb_result['forward_barrier_kJ']),
        'reverse_barrier_eV': float(neb_result['reverse_barrier_eV']),
        'reverse_barrier_kJ_mol': float(neb_result['reverse_barrier_kJ']),
        'reaction_energy_eV': float(neb_result['reaction_energy_eV']),
        'reaction_energy_kJ_mol': float(neb_result['reaction_energy_kJ']),
        'transition_state_image': int(neb_result['barrier_index']),
        'number_of_images': int(neb_result['n_images']),
        'converged_steps': int(neb_result['converged_steps']),
        'energies_eV': [float(e) for e in neb_result['energies']],
        'forces_eV_Ang': [float(f) for f in neb_result['forces']],
        'neb_parameters': {
            'spring_constant': float(neb_result['neb_params']['spring_constant']),
            'climbing_image': bool(neb_result['neb_params']['climb']),
            'force_convergence': float(neb_result['neb_params']['fmax']),
            'optimizer': str(neb_result['neb_params']['optimizer']),
            'interpolation': str(neb_result['neb_params']['interpolation'])
        }
    }

    return json.dumps(export_data, indent=2)


def display_neb_results(neb_results, structure_name, use_distance=False):
    st.subheader(f"NEB Results: {structure_name}")

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

    with col_r1:
        st.metric(
            "Forward Barrier",
            f"{neb_results['forward_barrier_eV']:.3f} eV",
            delta=f"{neb_results['forward_barrier_kJ']:.1f} kJ/mol"
        )

    with col_r2:
        st.metric(
            "Reverse Barrier",
            f"{neb_results['reverse_barrier_eV']:.3f} eV",
            delta=f"{neb_results['reverse_barrier_kJ']:.1f} kJ/mol"
        )

    with col_r3:
        st.metric(
            "Reaction Energy",
            f"{neb_results['reaction_energy_eV']:.3f} eV",
            delta=f"{neb_results['reaction_energy_kJ']:.1f} kJ/mol"
        )

    with col_r4:
        st.metric(
            "Transition State",
            f"Image {neb_results['barrier_index']}",
            delta=f"{neb_results['converged_steps']} steps"
        )

    if 'reaction_distances' in neb_results:
        total_distance = neb_results['reaction_distances'][-1]
        st.info(f"📏 Total path length: {total_distance:.2f} Å")

    fig = create_neb_plot(neb_results, structure_name, use_distance=use_distance)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detailed Energy Data"):
        energy_data = []
        for i, (energy, force) in enumerate(zip(neb_results['energies'], neb_results['forces'])):
            rel_energy = (energy - neb_results['energies'][0]) * 1000
            row = {
                'Image': i,
                'Energy (eV)': f"{energy:.6f}",
                'Relative Energy (meV)': f"{rel_energy:.2f}",
                'Max Force (eV/Å)': f"{force:.4f}"
            }

            if 'reaction_distances' in neb_results:
                row['Distance (Å)'] = f"{neb_results['reaction_distances'][i]:.2f}"

            energy_data.append(row)

        import pandas as pd
        df_energy = pd.DataFrame(energy_data)
        st.dataframe(df_energy, use_container_width=True, hide_index=True)
