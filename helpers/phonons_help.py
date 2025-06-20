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


def main():
    # Load and process the data
    df = load_and_process_data('lagrangian_results.csv')

    if df is None:
        return


    print("\nCreating plots...")
    create_plots(df)
    create_rattle_plot(df)
    print("\nSaving modified data...")
    save_modified_data(df)

    print("\nPlotting complete!")


if __name__ == "__main__":
    main()