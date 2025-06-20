import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_and_process_data(input_file='lagrangian_results.csv'):

    try:

        df = pd.read_csv(input_file)

        df = df.dropna(subset=['converted_step'])
        df = df.sort_values('converted_step')

        print(df['converted_step'])
        shake_min = df['shake_average'].min()
        rattle_min = df['rattle_average'].min()

        df['shifted_shake'] = df['shake_average'] #- shake_min
        df['shifted_rattle'] = df['rattle_average'] #- rattle_min


        df['modified_shake_avg'] = df['shifted_shake'] * df['converted_step']
        df['modified_rattle_avg'] = df['shifted_rattle'] * df['converted_step']

        from scipy.integrate import cumulative_trapezoid
        df['modified_shake_avg'] = -cumulative_trapezoid(df['shifted_shake'], df['converted_step'], initial=0)#*27.2
        df['modified_rattle_avg'] = -cumulative_trapezoid(df['shifted_rattle'], df['converted_step'], initial=0)#*27.2
        print(df['modified_rattle_avg'])
        print(df['shifted_rattle'])
        return df

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        print("Please make sure you have run the main processing script first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_plots(df):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))


    ax1.plot(df['converted_step'], df['shifted_shake'], 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Distance [bohr]')
    ax1.set_ylabel('Shifted Shake [hartree]')
    ax1.set_title('Shifted Shake Lagrangian Multipliers vs Distance')
    ax1.grid(True, alpha=0.3)


    ax2.plot(df['converted_step'], df['shifted_rattle'], 'r-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Distance [bohr]')
    ax2.set_ylabel('Shifted Rattle [hartree]')
    ax2.set_title('Shifted Rattle Lagrangian Multipliers vs Distance')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Modified Shake (shifted × distance) vs Distance
    ax3.plot(df['converted_step'], df['modified_shake_avg'], 'g-o', linewidth=2, markersize=4)
    ax3.set_xlabel('Distance [bohr]')
    ax3.set_ylabel('Energy [hartree]')
    ax3.set_title('Modified Shake (Shifted Shake × Distance) vs Distance')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Modified Rattle (shifted × distance) vs Distance
    ax4.plot(df['converted_step'], df['modified_rattle_avg'], 'm-o', linewidth=2, markersize=4)
    ax4.set_xlabel('Distance [bohr]')
    ax4.set_ylabel('Energy [hartree]')
    ax4.set_title('Modified Rattle (Shifted Rattle × Distance) vs Distance')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lagrangian_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create separate plots for modified forces with better scaling
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))

    ax5.plot(df['converted_step'], df['modified_shake_avg'], 'g-o', linewidth=2, markersize=5)
    ax5.set_xlabel('Distance [bohr]')
    ax5.set_ylabel('Energy [hartree]')
    ax5.set_title('Modified Shake Energy vs Distance')
    ax5.grid(True, alpha=0.3)
    # Add some statistics to the plot
    ax5.text(0.02, 0.98, f'Range: 0 to {df["modified_shake_avg"].max():.3e}',
             transform=ax5.transAxes, verticalalignment='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax6.plot(df['converted_step'], df['modified_rattle_avg'], 'm-o', linewidth=2, markersize=5)
    ax6.set_xlabel('Distance [bohr]')
    ax6.set_ylabel('Energy [hartree]')
    ax6.set_title('Modified Rattle Energy vs Distance')
    ax6.grid(True, alpha=0.3)
    # Add some statistics to the plot
    ax6.text(0.02, 0.98, f'Range: 0 to {df["modified_rattle_avg"].max():.3e}',
             transform=ax6.transAxes, verticalalignment='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('modified_forces_plots.png', dpi=300, bbox_inches='tight')
    #plt.show()

    fig3, (ax7, ax8) = plt.subplots(1, 2, figsize=(16, 6))


    ax7.plot(df['converted_step'], df['shifted_shake'], 'b-o', linewidth=2, markersize=4, label='Shifted Shake')
    ax7_twin = ax7.twinx()
    ax7_twin.plot(df['converted_step'], df['modified_shake_avg'], 'g-s', linewidth=2, markersize=4,
                  label='Modified Shake (×Distance)')
    ax7.set_xlabel('Distance [bohr]')
    ax7.set_ylabel('Shifted Shake [hartree]', color='b')
    ax7_twin.set_ylabel('Modified Shake Energy [hartree]', color='g')
    ax7.set_title('Shake: Shifted vs Modified')
    ax7.grid(True, alpha=0.3)
    ax7.legend(loc='upper left')
    ax7_twin.legend(loc='upper right')

    ax8.plot(df['converted_step'], df['shifted_rattle'], 'r-o', linewidth=2, markersize=4, label='Shifted Rattle')
    ax8_twin = ax8.twinx()
    ax8_twin.plot(df['converted_step'], df['modified_rattle_avg'], 'm-s', linewidth=2, markersize=4,
                  label='Modified Rattle (×Distance)')
    ax8.set_xlabel('Distance [bohr]')
    ax8.set_ylabel('Shifted Rattle [hartree]', color='r')
    ax8_twin.set_ylabel('Modified Rattle Energy [hartree]', color='m')
    ax8.set_title('Rattle: Shifted vs Modified')
    ax8.grid(True, alpha=0.3)
    ax8.legend(loc='upper left')
    ax8_twin.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('original_vs_modified_comparison.png', dpi=300, bbox_inches='tight')
    #plt.show()

def create_rattle_plot(df):
        import matplotlib.pyplot as plt

        fontsize_title = 28
        fontsize_labels = 28
        fontsize_ticks = 28

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(df['converted_step']* 0.529177, df['modified_rattle_avg']*27.2, 'm-o', linewidth=2, markersize=6)
        ax.set_xlabel('Distance [Å]', fontsize=fontsize_labels)
        ax.set_ylabel('Energy [eV]', fontsize=fontsize_labels)
        ax.grid(True, alpha=0.3)

        output_df = pd.DataFrame({
            'Distance (Å)': df['converted_step'] * 0.529177,
            'Energy (eV)': df['modified_rattle_avg'] * 27.2
        })

        # Save to a new CSV file
        output_df.to_csv('Distance_Energy.csv', index=False)

        ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

        plt.tight_layout()
        plt.savefig('modified_rattle_presentation.png', dpi=300, bbox_inches='tight')
        plt.show()


def save_modified_data(df):
    complete_filename = 'lagrangian_results_with_modified_forces.csv'
    df.to_csv(complete_filename, index=False)

    shake_energy_distance_filename = 'shake_energy_vs_distance.csv'
    shake_energy_df = df[['converted_step', 'modified_shake_avg']].copy()
    shake_energy_df.columns = ['distance_bohr', 'energy_hartree']
    shake_energy_df.to_csv(shake_energy_distance_filename, index=False)

    rattle_energy_distance_filename = 'rattle_energy_vs_distance.csv'
    rattle_energy_df = df[['converted_step', 'modified_rattle_avg']].copy()
    rattle_energy_df.columns = ['distance_bohr', 'energy_hartree']
    rattle_energy_df.to_csv(rattle_energy_distance_filename, index=False)

    shake_modified_filename = 'shake_modified_data.csv'
    shake_df = df[['converted_step', 'modified_shake_avg']].copy()
    shake_df.columns = ['distance_bohr', 'modified_shake_energy_hartree']
    shake_df.to_csv(shake_modified_filename, index=False)


    rattle_modified_filename = 'rattle_modified_data.csv'
    rattle_df = df[['converted_step', 'modified_rattle_avg']].copy()
    rattle_df.columns = ['distance_bohr', 'modified_rattle_energy_hartree']
    rattle_df.to_csv(rattle_modified_filename, index=False)

    print(f"\nData files created:")
    print(f"  - {complete_filename} (Complete data with modified forces)")
    print(f"  - {shake_modified_filename} (Distance [bohr] vs Modified Shake Energy [hartree])")
    print(f"  - {rattle_modified_filename} (Distance [bohr] vs Modified Rattle Energy [hartree])")
    print(f"  - lagrangian_comparison_plots.png (4-panel comparison)")
    print(f"  - modified_forces_plots.png (Modified forces with ranges)")
    print(f"  - original_vs_modified_comparison.png (Direct comparison plots)")


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

from helpers import *

import py3Dmol
import streamlit.components.v1 as components
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from ase.io import read, write
from ase import Atoms

def extract_thermodynamics_at_temperatures(phonon_results, target_temperatures):
    if not phonon_results['success']:
        return {'error': 'Phonon calculation was not successful'}
    
    if 'thermal_properties_dict' in phonon_results:
        thermal_dict = phonon_results['thermal_properties_dict']
        temps = np.array(thermal_dict['temperatures'])
        
        print(f"Available keys in thermal_dict: {list(thermal_dict.keys())}")
        
        results = {}
        for target_temp in target_temperatures:
            temp_idx = np.argmin(np.abs(temps - target_temp))
            actual_temp = temps[temp_idx]
            
            zero_point_energy = 0.0
            if 'zero_point_energy' in thermal_dict:
                zero_point_energy = float(thermal_dict['zero_point_energy'])
            elif 'zpe' in thermal_dict:
                zero_point_energy = float(thermal_dict['zpe'])
            elif 'zero_point' in thermal_dict:
                zero_point_energy = float(thermal_dict['zero_point'])
            else:
                frequencies = np.array(phonon_results['frequencies'])  # in meV
                freq_eV = frequencies[frequencies > 0] * 1e-3  # meV to eV
                if len(freq_eV) > 0:
                    zero_point_energy = float(0.5 * np.sum(freq_eV))

            result = {
                'actual_temperature': float(actual_temp),
                'temperature': float(target_temp),
                'zero_point_energy': zero_point_energy
            }
            
            if 'internal_energy' in thermal_dict:
                result['internal_energy'] = float(thermal_dict['internal_energy'][temp_idx])
            elif 'internal_energies' in thermal_dict:
                result['internal_energy'] = float(thermal_dict['internal_energies'][temp_idx])
            else:
                result['internal_energy'] = zero_point_energy  
            
            if 'heat_capacity' in thermal_dict:
                result['heat_capacity'] = float(thermal_dict['heat_capacity'][temp_idx])
            elif 'heat_capacities' in thermal_dict:
                result['heat_capacity'] = float(thermal_dict['heat_capacities'][temp_idx])
            elif 'cv' in thermal_dict:
                result['heat_capacity'] = float(thermal_dict['cv'][temp_idx])
            else:
                result['heat_capacity'] = 0.0  
            
            if 'entropy' in thermal_dict:
                result['entropy'] = float(thermal_dict['entropy'][temp_idx])
            elif 'entropies' in thermal_dict:
                result['entropy'] = float(thermal_dict['entropies'][temp_idx])
            else:
                result['entropy'] = 0.0 
            
            if 'free_energy' in thermal_dict:
                result['free_energy'] = float(thermal_dict['free_energy'][temp_idx])
            elif 'free_energies' in thermal_dict:
                result['free_energy'] = float(thermal_dict['free_energies'][temp_idx])
            elif 'helmholtz_free_energy' in thermal_dict:
                result['free_energy'] = float(thermal_dict['helmholtz_free_energy'][temp_idx])
            else:
                if 'internal_energy' in result and 'entropy' in result:
                    result['free_energy'] = result['internal_energy'] - actual_temp * result['entropy']
                else:
                    result['free_energy'] = zero_point_energy 
            
            results[target_temp] = result
        
        return results
    
    frequencies = np.array(phonon_results['frequencies'])  # in meV
    
    freq_eV = frequencies[frequencies > 0] * 1e-3  # meV to eV
    
    if len(freq_eV) == 0:
        return {'error': 'No positive frequencies found'}
    
    kB = 8.617e-5  # eV/K (Boltzmann constant)
    
    results = {}
    for temp in target_temperatures:
        if temp <= 0:
            E_zp = 0.5 * np.sum(freq_eV)  
            results[temp] = {
                'temperature': 0.0,
                'zero_point_energy': float(E_zp),
                'internal_energy': float(E_zp),
                'heat_capacity': 0.0,
                'entropy': 0.0,
                'free_energy': float(E_zp)
            }
            continue
        
        x = freq_eV / (kB * temp)
        x = np.minimum(x, 50)  

        exp_x = np.exp(x)
        n_BE = 1.0 / (exp_x - 1 + 1e-12)  
        
        E_zp = 0.5 * np.sum(freq_eV)
        
        U = E_zp + np.sum(freq_eV * n_BE)
        
        Cv = np.sum(kB * x**2 * exp_x / (exp_x - 1 + 1e-12)**2)
        
        # Entropy
        S = np.sum(kB * (x * n_BE - np.log(1 - np.exp(-x) + 1e-12)))
        
        # Free energy (Helmholtz)
        F = U - temp * S
        
        results[temp] = {
            'temperature': float(temp),
            'zero_point_energy': float(E_zp),
            'internal_energy': float(U),
            'heat_capacity': float(Cv),
            'entropy': float(S),
            'free_energy': float(F)
        }
    
    return results


def recalculate_with_broader_temperature_range(atoms, calculator, phonon_params, log_queue, structure_name, 
                                               min_temp=0, max_temp=1000, temp_step=10):

    try:
        
        log_queue.put(f"Recalculating thermodynamics for {structure_name} from {min_temp}K to {max_temp}K")

        phonon.run_thermal_properties(
            t_step=temp_step,
            t_max=max_temp,
            t_min=min_temp
        )
        
        thermal_dict = phonon.get_thermal_properties_dict()
        
        log_queue.put(f"✅ Thermodynamics calculated for temperature range {min_temp}-{max_temp}K")
        
        return {
            'success': True,
            'thermal_properties_dict': thermal_dict,
            'temperature_range': (min_temp, max_temp),
            'temperature_step': temp_step
        }
        
    except Exception as e:
        log_queue.put(f"❌ Failed to recalculate thermodynamics: {str(e)}")
        return {'success': False, 'error': str(e)}


def add_entropy_vs_temperature_plot(phonon_results, temp_range=(0, 1000, 10)):
    """
    Create entropy vs temperature plot from phonon results
    """
    import plotly.graph_objects as go
    
    min_temp, max_temp, step = temp_range
    temperatures = list(range(min_temp, max_temp + 1, step))
    
    # Calculate thermodynamics at all temperatures
    thermo_data = extract_thermodynamics_at_temperatures(phonon_results, temperatures)
    
    if 'error' in thermo_data:
        return None, thermo_data['error']
    
    # Extract data for plotting
    temps = []
    entropies = []
    heat_capacities = []
    free_energies = []
    
    for temp in temperatures:
        if temp in thermo_data:
            data = thermo_data[temp]
            temps.append(data['temperature'])
            entropies.append(data['entropy'])
            heat_capacities.append(data['heat_capacity'])
            free_energies.append(data['free_energy'])
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Entropy vs Temperature', 'Heat Capacity vs Temperature',
                       'Free Energy vs Temperature', 'Internal Energy vs Temperature'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Entropy
    fig.add_trace(
        go.Scatter(x=temps, y=entropies, mode='lines', name='Entropy',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Heat capacity
    fig.add_trace(
        go.Scatter(x=temps, y=heat_capacities, mode='lines', name='Heat Capacity',
                  line=dict(color='red', width=2)),
        row=1, col=2
    )
    
    # Free energy
    fig.add_trace(
        go.Scatter(x=temps, y=free_energies, mode='lines', name='Free Energy',
                  line=dict(color='green', width=2)),
        row=2, col=1
    )
    
    
    # Update axes labels
    fig.update_xaxes(title_text="Temperature (K)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (K)", row=1, col=2)
    fig.update_xaxes(title_text="Temperature (K)", row=2, col=1)
    fig.update_xaxes(title_text="Temperature (K)", row=2, col=2)
    
    fig.update_yaxes(title_text="Entropy (eV/K)", row=1, col=1)
    fig.update_yaxes(title_text="Heat Capacity (eV/K)", row=1, col=2)
    fig.update_yaxes(title_text="Free Energy (eV)", row=2, col=1)
    fig.update_yaxes(title_text="Internal Energy (eV)", row=2, col=2)
    
    fig.update_layout(
        height=600,
        title_text="Thermodynamic Properties vs Temperature",
        showlegend=False
    )
    
    return fig, thermo_data




def create_calculation_type_image(calc_type):
    """Create improved SVG visualization with bigger fonts and larger size"""
    
    if calc_type == "Energy Only":
        svg_content = """
        <svg width="400" height="280" viewBox="0 0 400 280" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="energyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#4f46e5;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#7c3aed;stop-opacity:1" />
                </linearGradient>
            </defs>
            
            <!-- Background -->
            <rect width="400" height="280" fill="url(#energyGrad)" rx="15"/>
            
            <!-- Energy symbol -->
            <circle cx="200" cy="140" r="70" fill="none" stroke="white" stroke-width="5"/>
            <text x="200" y="160" text-anchor="middle" fill="white" font-size="54" font-weight="bold">E</text>
            
            <!-- Energy lines -->
            <line x1="130" y1="140" x2="270" y2="140" stroke="white" stroke-width="4"/>
            <line x1="200" y1="70" x2="200" y2="210" stroke="white" stroke-width="4"/>
            
            <!-- Title -->
            <text x="200" y="40" text-anchor="middle" fill="white" font-size="26" font-weight="bold">Energy Calculation</text>
            <text x="200" y="250" text-anchor="middle" fill="white" font-size="20">Single Point Energy</text>
        </svg>
        """
    
    elif calc_type == "Geometry Optimization":
        svg_content = """
        <svg width="400" height="280" viewBox="0 0 400 280" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="optGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#059669;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#0d9488;stop-opacity:1" />
                </linearGradient>
            </defs>
            
            <!-- Background -->
            <rect width="400" height="280" fill="url(#optGrad)" rx="15"/>
            
            <!-- Initial structure (dotted) -->
            <rect x="70" y="100" width="60" height="60" fill="none" stroke="white" stroke-width="4" stroke-dasharray="8,8" opacity="0.7"/>
            <circle cx="100" r="14" cy="130" fill="white" opacity="0.7"/>
            
            <!-- Arrow -->
            <path d="M145 130 L220 130 M205 115 L220 130 L205 145" stroke="white" stroke-width="5" fill="none"/>
            
            <!-- Optimized structure (solid) -->
            <rect x="270" y="100" width="60" height="60" fill="none" stroke="white" stroke-width="4"/>
            <circle cx="300" r="14" cy="130" fill="white"/>
            
            <!-- Force vectors -->
            <path d="M250 85 L270 105 M255 85 L270 105 L250 100" stroke="#fbbf24" stroke-width="4" fill="none"/>
            <path d="M340 175 L320 155 M340 170 L320 155 L335 175" stroke="#fbbf24" stroke-width="4" fill="none"/>
            
            <!-- Title -->
            <text x="200" y="40" text-anchor="middle" fill="white" font-size="26" font-weight="bold">Geometry Optimization</text>
            <text x="200" y="250" text-anchor="middle" fill="white" font-size="20">Structure Relaxation</text>
        </svg>
        """
    
    elif calc_type == "Phonon Calculation":
        svg_content = """
        <svg width="400" height="280" viewBox="0 0 400 280" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="phononGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#dc2626;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#ea580c;stop-opacity:1" />
                </linearGradient>
            </defs>
            
            <!-- Background -->
            <rect width="400" height="280" fill="url(#phononGrad)" rx="15"/>
            
            <!-- Atoms in a chain -->
            <circle cx="80" cy="140" r="18" fill="white"/>
            <circle cx="160" cy="140" r="18" fill="white"/>
            <circle cx="240" cy="140" r="18" fill="white"/>
            <circle cx="320" cy="140" r="18" fill="white"/>
            
            <!-- Vibrational waves -->
            <path d="M80 140 Q120 105 160 140 T240 140 T320 140" stroke="#fbbf24" stroke-width="5" fill="none" opacity="0.9"/>
            <path d="M80 140 Q120 175 160 140 T240 140 T320 140" stroke="#fbbf24" stroke-width="5" fill="none" opacity="0.9"/>
            
            <!-- Springs -->
            <path d="M98 140 Q113 132 128 140 T142 140" stroke="white" stroke-width="4" fill="none"/>
            <path d="M178 140 Q193 132 208 140 T222 140" stroke="white" stroke-width="4" fill="none"/>
            <path d="M258 140 Q273 132 288 140 T302 140" stroke="white" stroke-width="4" fill="none"/>
            
            <!-- Frequency spectrum -->
            <rect x="350" y="90" width="15" height="35" fill="white"/>
            <rect x="368" y="75" width="15" height="50" fill="white"/>
            <rect x="386" y="60" width="15" height="65" fill="white"/>
            
            <!-- Title -->
            <text x="200" y="40" text-anchor="middle" fill="white" font-size="26" font-weight="bold">Phonon Calculation</text>
            <text x="200" y="250" text-anchor="middle" fill="white" font-size="20">Vibrational Properties</text>
        </svg>
        """
    
    elif calc_type == "Elastic Properties":
        svg_content = """
        <svg width="400" height="280" viewBox="0 0 400 280" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="elasticGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#7c2d12;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#a16207;stop-opacity:1" />
                </linearGradient>
            </defs>
            
            <!-- Background -->
            <rect width="400" height="280" fill="url(#elasticGrad)" rx="15"/>
            
            <!-- Original cube -->
            <rect x="100" y="100" width="70" height="70" fill="none" stroke="white" stroke-width="4" opacity="0.6" stroke-dasharray="5,5"/>
            
            <!-- Strained cube -->
            <path d="M220 85 L310 85 L325 105 L325 175 L310 195 L220 195 L205 175 L205 105 Z" 
                  fill="none" stroke="white" stroke-width="4"/>
            
            <!-- Left side stress arrows (pointing inward) -->
            <path d="M180 110 L205 110 M195 105 L205 110 L195 115" stroke="#fbbf24" stroke-width="5" fill="none"/>
            <path d="M180 170 L205 170 M195 165 L205 170 L195 175" stroke="#fbbf24" stroke-width="5" fill="none"/>
            
            <!-- Right side stress arrows (pointing outward) -->
            <path d="M325 110 L350 110 M340 105 L350 110 L340 115" stroke="#fbbf24" stroke-width="5" fill="none"/>
            <path d="M325 170 L350 170 M340 165 L350 170 L340 175" stroke="#fbbf24" stroke-width="5" fill="none"/>
            
            <!-- Strain indicator -->
            <text x="265" y="145" text-anchor="middle" fill="#fbbf24" font-size="24" font-weight="bold">ε</text>
            
            <!-- Elastic tensor representation -->
            <rect x="60" y="220" width="15" height="15" fill="white"/>
            <rect x="77" y="220" width="15" height="15" fill="white" opacity="0.8"/>
            <rect x="94" y="220" width="15" height="15" fill="white" opacity="0.6"/>
            <text x="115" y="232" fill="white" font-size="16">C_ij</text>
            
            <!-- Title -->
            <text x="200" y="40" text-anchor="middle" fill="white" font-size="26" font-weight="bold">Elastic Properties</text>
            <text x="200" y="250" text-anchor="middle" fill="white" font-size="20">Mechanical Properties</text>
        </svg>
        """
    
    else:
        # Default image
        svg_content = """
        <svg width="400" height="280" viewBox="0 0 400 280" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="defaultGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#6b7280;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#4b5563;stop-opacity:1" />
                </linearGradient>
            </defs>
            
            <!-- Background -->
            <rect width="400" height="280" fill="url(#defaultGrad)" rx="15"/>
            
            <!-- MACE logo/symbol -->
            <circle cx="200" cy="140" r="70" fill="none" stroke="white" stroke-width="5"/>
            <text x="200" y="155" text-anchor="middle" fill="white" font-size="32" font-weight="bold">MACE</text>
            
            <!-- Title -->
            <text x="200" y="40" text-anchor="middle" fill="white" font-size="26" font-weight="bold">Select Calculation Type</text>
            <text x="200" y="250" text-anchor="middle" fill="white" font-size="20">Choose calculation method</text>
        </svg>
        """
    
    return svg_content
