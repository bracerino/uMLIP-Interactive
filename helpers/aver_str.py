import re
import numpy as np
from collections import Counter
import sys



# Input file name
file_name = 'data.txt'
# Output file name
output_file_name = 'average_pos.poscar'


#  captures Step, Time, Energy, Temp, and the Lattice matrix
info_pattern = r"Step=(\d+) Time=([\d\.]+)fs Energy=([-\d\.]+)eV Temp=([\d\.]+)K Lattice=\"([\d\s\.]+)\" Properties=(.*)"

#  expression pattern for extracting atom data (Species, x, y, z)
atom_pattern = r"([A-Za-z]+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)"



def read_dat(file_name):
    """
    Reads an extended XYZ trajectory file.
    
    Extracts atom positions and lattice parameters for every frame.
    Atom types and counts are read from the first frame.
    """
    all_atoms_pos = []
    all_lattices = []
    first_frame_atom_types = []

    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {file_name}")
        return None, None, None, None

    if not lines:
        print("Error: File is empty.")
        return None, None, None, None

    try:
        num_at = int(lines[0].strip())
    except (ValueError, IndexError):
        print(f"Error: Could not read number of atoms from first line: {lines[0]}")
        return None, None, None, None

    frame_size = num_at + 2
    num_frames = len(lines) // frame_size

    if len(lines) % frame_size != 0:
        print(f"Warning: File size ({len(lines)}) is not an even multiple of frame size ({frame_size}).")
        print("The file might be truncated. Proceeding with complete frames only.")

    if num_frames == 0:
        print("Error: No complete frames found in the file.")
        return None, None, None, None

    for i in range(num_at):
        line_index = 2 + i
        atom_match = re.match(atom_pattern, lines[line_index])
        if atom_match:
            first_frame_atom_types.append(atom_match.group(1))
        else:
            print(f"Error: Could not parse atom line {line_index} in first frame:")
            print(f"-> {lines[line_index].strip()}")
            return None, None, None, None
            
    atom_counts_dict = Counter(first_frame_atom_types)
    at_types_ordered = []
    for at_type in first_frame_atom_types:
        if at_type not in at_types_ordered:
            at_types_ordered.append(at_type)
    
    at_counts_ordered = [atom_counts_dict[at_type] for at_type in at_types_ordered]

    for i in range(num_frames):
        frame_start_line = i * frame_size
        
        # Read lattice from the comment line (line 2 of the frame)
        comment_line = lines[frame_start_line + 1]
        info_match = re.search(info_pattern, comment_line)
        
        if info_match:
            try:
                lattice_values = list(map(float, info_match.group(5).split()))
                lattice = np.array(lattice_values).reshape(3, 3)
                all_lattices.append(lattice)
            except (ValueError, IndexError):
                print(f"Error: Could not parse lattice values from comment line {frame_start_line + 1}:")
                print(f"-> {comment_line.strip()}")
                return None, None, None, None
        else:
            print(f"Error: Comment line format did not match expected pattern on line {frame_start_line + 1}:")
            print(f"-> {comment_line.strip()}")
            return None, None, None, None

        # Read atom positions for this frame
        frame_atoms = []
        for j in range(num_at):
            line_index = frame_start_line + 2 + j
            atom_match = re.match(atom_pattern, lines[line_index])
            if atom_match:
                x = float(atom_match.group(2))
                y = float(atom_match.group(3))
                z = float(atom_match.group(4))
                frame_atoms.append([x, y, z])
            else:
                print(f"Error: Could not parse atom line {line_index}:")
                print(f"-> {lines[line_index].strip()}")
                return None, None, None, None
        all_atoms_pos.append(frame_atoms)

    print(f"Successfully read {num_frames} frames.")
    return np.array(all_atoms_pos), at_types_ordered, at_counts_ordered, np.array(all_lattices)

def calc_average(data_array, start_step, stop_step):
    """
    Calculates the average of a numpy array over a specified range of the first axis.
    
    data_array: The (num_steps, ...) numpy array to average.
    start_step: The starting index (inclusive).
    stop_step: The stopping index (exclusive). Use 'end' for all steps until the end.
    """
    tot_steps = data_array.shape[0]
    
    if stop_step == 'end':
        stop_step_index = tot_steps
    else:
        stop_step_index = int(stop_step)

    start_step_index = int(start_step)

    if start_step_index < 0 or start_step_index >= tot_steps:
        print(f"Error: start_step ({start_step}) is out of bounds (0 to {tot_steps - 1}).")
        return None

    if stop_step_index <= start_step_index or stop_step_index > tot_steps:
        print(f"Error: stop_step ({stop_step}) is invalid or out of bounds.")
        return None

    # Slice the array and calculate the mean along the first axis (axis=0)
    avr_data = np.mean(data_array[start_step_index:stop_step_index], axis=0)
    
    print(f"Averaged data from frame {start_step_index} up to (but not including) frame {stop_step_index}.")
    return avr_data

def write_poscar(frac_positions, at_types, at_count, lattice_par):
    """
    Writes the averaged structure data to a POSCAR file.
    
    frac_positions: (num_at, 3) array of fractional coordinates.
    at_types: List of unique atom type strings (e.g., ['Si', 'O']).
    at_count: List of atom counts corresponding to at_types (e.g., [4, 8]).
    lattice_par: (3, 3) numpy array of the average lattice vectors.
    """
    try:
        with open(output_file_name, "w") as file:
            file.write("Average structure: " + " ".join(at_types) + "\n")
            
            # Lattice scale (universal scaling factor)
            file.write(f"{1.0: .16f}\n")
            
            # Lattice vectors
            for i in range(3):
                file.write(f"    {lattice_par[i][0]: .16f} {lattice_par[i][1]: .16f} {lattice_par[i][2]: .16f}\n")
            
            # Atom types
            file.write("   " + " ".join(at_types) + "\n")
            
            # Atom counts
            file.write("   " + " ".join(map(str, at_count)) + "\n")
            
            # Coordinate type
            file.write("Direct\n")
            
            # Atom positions (fractional)
            for i in range(len(frac_positions)):
                file.write(f"  {frac_positions[i][0]: .16f} {frac_positions[i][1]: .16f} {frac_positions[i][2]: .16f}\n")
                
    except IOError as e:
        print(f"Error writing to file {output_file_name}: {e}")
        return False
        
    return True

# --- Main Script Execution ---

def main():
    all_atom_pos, at_types, at_count, all_lattices = read_dat(file_name)

    if all_atom_pos is None:
        print("Failed to read data. Exiting.")
        sys.exit(1) 

    start_step = 0
    stop_step = 'end'  # or specify an integer, e.g., 31

    average_pos_cart = calc_average(all_atom_pos, start_step=start_step, stop_step=stop_step)

    average_lattice = calc_average(all_lattices, start_step=start_step, stop_step=stop_step)

    if average_pos_cart is None or average_lattice is None:
        print("Failed to calculate averages. Exiting.")
        sys.exit(1)

    try:
        inv_average_lattice = np.linalg.inv(average_lattice)
        average_pos_frac = average_pos_cart @ inv_average_lattice
        
        average_pos_frac = average_pos_frac - np.floor(average_pos_frac)

    except np.linalg.LinAlgError:
        print("Error: The average lattice matrix is singular and cannot be inverted.")
        print("This may be due to averaging to a 2D or 1D structure, or bad data.")
        sys.exit(1)
        
    if write_poscar(average_pos_frac, at_types, at_count, average_lattice):
        print(f"\nSuccessfully wrote average structure to {output_file_name}")
        print(f"Atom Types: {at_types}")
        print(f"Atom Counts: {at_count}")
        print("Average Lattice Vectors:\n", average_lattice)
    else:
        print(f"Failed to write output file.")
        sys.exit(1)

if __name__ == "__main__":
    main()
