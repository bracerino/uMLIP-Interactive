import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import threading
import queue
import time
import io
import zipfile
from pathlib import Path
from plotly.subplots import make_subplots

from helpers.phonons_help import *
from helpers.generate_python_code import *

import py3Dmol
import streamlit.components.v1 as components
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from ase.io import read, write
from ase import Atoms

def extract_element_concentrations(structures_data):
    all_compositions = {}
    for name, structure in structures_data.items():
        composition = structure.composition.as_dict()
        total_atoms = sum(composition.values())
        concentrations = {element: (count / total_atoms) * 100 for element, count in
                            composition.items()}
        all_compositions[name] = concentrations
    return all_compositions


def get_common_elements(compositions_dict):
    all_elements = set()
    for comp in compositions_dict.values():
        all_elements.update(comp.keys())

    common_elements = []
    for element in all_elements:
        if all(element in comp for comp in compositions_dict.values()):
            concentrations = [comp[element] for comp in compositions_dict.values()]
            if len(set(concentrations)) > 1:
                common_elements.append(element)
    return sorted(common_elements)


def calculate_phase_diagram_data(phonon_results, element_concentrations, temp_range):
    phase_data = []

    for result in phonon_results:
        structure_name = result['name']
        phonon_data = result['phonon_results']

        if structure_name not in element_concentrations:
            continue

        element_conc = element_concentrations[structure_name]

        temp_thermo = extract_thermodynamics_at_temperatures(phonon_data, temp_range)
        if 'error' in temp_thermo:
            continue

        for temp in temp_range:
            if temp in temp_thermo:
                thermo = temp_thermo[temp]
                phase_data.append({
                    'structure': structure_name,
                    'concentration': element_conc,
                    'temperature': temp,
                    'free_energy': thermo['free_energy'],
                    'entropy': thermo['entropy'],
                    'heat_capacity': thermo['heat_capacity'],
                    'internal_energy': thermo['internal_energy']
                })

    return pd.DataFrame(phase_data)


def find_stable_phases(phase_df):
    stable_phases = []

    for temp in phase_df['temperature'].unique():
        temp_data = phase_df[phase_df['temperature'] == temp]
        min_free_energy_idx = temp_data['free_energy'].idxmin()
        stable_phase = temp_data.loc[min_free_energy_idx]
        stable_phases.append({
            'temperature': temp,
            'stable_structure': stable_phase['structure'],
            'stable_concentration': stable_phase['concentration'],
            'free_energy': stable_phase['free_energy']
        })

    return pd.DataFrame(stable_phases)


def create_phase_diagram_plot(phase_df, stable_df, selected_element):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Phase Stability Map ({selected_element} concentration vs Temperature)',
            'Free Energy vs Temperature',
            'Stable Phase Boundaries',
            'Free Energy Differences'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]]
    )

    structures = phase_df['structure'].unique()
    colors = px.colors.qualitative.Set1[:len(structures)]

    for i, structure in enumerate(structures):
        struct_data = phase_df[phase_df['structure'] == structure]

        fig.add_trace(
            go.Scatter(
                x=struct_data['concentration'],
                y=struct_data['temperature'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=struct_data['free_energy'],
                    colorscale='RdYlBu_r',
                    showscale=i == 0,
                    colorbar=dict(title="Free Energy (eV)", x=1.02)
                ),
                name=structure,
                hovertemplate=f'<b>{structure}</b><br>' +
                                f'{selected_element}: %{{x:.1f}}%<br>' +
                                'T: %{y}K<br>' +
                                'F: %{marker.color:.4f} eV<extra></extra>'
            ),
            row=1, col=1
        )

    for i, structure in enumerate(structures):
        struct_data = phase_df[phase_df['structure'] == structure]
        unique_temps = sorted(struct_data['temperature'].unique())
        avg_free_energies = [struct_data[struct_data['temperature'] == t]['free_energy'].mean()
                                for t in unique_temps]

        fig.add_trace(
            go.Scatter(
                x=unique_temps,
                y=avg_free_energies,
                mode='lines+markers',
                name=structure,
                line=dict(color=colors[i], width=2),
                marker=dict(size=6),
                showlegend=False
            ),
            row=1, col=2
        )

    phase_transitions = []
    for i in range(len(stable_df) - 1):
        if stable_df.iloc[i]['stable_structure'] != stable_df.iloc[i + 1]['stable_structure']:
            phase_transitions.append({
                'temperature': stable_df.iloc[i + 1]['temperature'],
                'from_phase': stable_df.iloc[i]['stable_structure'],
                'to_phase': stable_df.iloc[i + 1]['stable_structure']
            })

    fig.add_trace(
        go.Scatter(
            x=stable_df['stable_concentration'],
            y=stable_df['temperature'],
            mode='lines+markers',
            line=dict(color='black', width=4),
            marker=dict(size=8, color='red'),
            name='Stable boundary',
            showlegend=False
        ),
        row=2, col=1
    )

    if len(structures) >= 2:
        ref_structure = structures[0]
        for structure in structures[1:]:
            struct1_data = phase_df[phase_df['structure'] == ref_structure].groupby('temperature')[
                'free_energy'].mean()
            struct2_data = phase_df[phase_df['structure'] == structure].groupby('temperature')[
                'free_energy'].mean()

            common_temps = sorted(set(struct1_data.index) & set(struct2_data.index))
            free_energy_diff = [struct2_data[t] - struct1_data[t] for t in common_temps]

            fig.add_trace(
                go.Scatter(
                    x=common_temps,
                    y=free_energy_diff,
                    mode='lines',
                    name=f'{structure} - {ref_structure}',
                    line=dict(width=2),
                    showlegend=False
                ),
                row=2, col=2
            )

    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=2)

    fig.update_xaxes(title_text=f"{selected_element} concentration (%)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (K)", row=1, col=2)
    fig.update_xaxes(title_text=f"{selected_element} concentration (%)", row=2, col=1)
    fig.update_xaxes(title_text="Temperature (K)", row=2, col=2)

    fig.update_yaxes(title_text="Temperature (K)", row=1, col=1)
    fig.update_yaxes(title_text="Free Energy (eV)", row=1, col=2)
    fig.update_yaxes(title_text="Temperature (K)", row=2, col=1)
    fig.update_yaxes(title_text="ΔF (eV)", row=2, col=2)

    fig.update_layout(
        height=800,
        title_text="Computational Phase Diagram Analysis",
        showlegend=True,
        legend=dict(x=1.05, y=1)
    )

    return fig, phase_transitions


def create_concentration_heatmap(phase_df, selected_element, property_name='free_energy'):
    pivot_data = phase_df.pivot_table(
        values=property_name,
        index='temperature',
        columns='concentration',
        aggfunc='mean'
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlBu_r',
        colorbar=dict(title=f"{property_name.replace('_', ' ').title()} (eV)")
    ))

    fig.update_layout(
        title=f"{property_name.replace('_', ' ').title()} vs {selected_element} Concentration and Temperature",
        xaxis_title=f"{selected_element} Concentration (%)",
        yaxis_title="Temperature (K)",
        height=500
    )

    return fig


def export_phase_diagram_data(phase_df, stable_df, phase_transitions, selected_element):
    export_data = {
        'metadata': {
            'selected_element': selected_element,
            'temperature_range': [int(phase_df['temperature'].min()),
                                    int(phase_df['temperature'].max())],
            'concentration_range': [float(phase_df['concentration'].min()),
                                    float(phase_df['concentration'].max())],
            'structures_analyzed': list(phase_df['structure'].unique()),
            'analysis_type': 'computational_phase_diagram'
        },
        'phase_data': phase_df.to_dict('records'),
        'stable_phases': stable_df.to_dict('records'),
        'phase_transitions': phase_transitions
    }

    return json.dumps(export_data, indent=2)