import streamlit as st

st.set_page_config(
    page_title="SimplySQS: Create Input Files for Generation of SQS using ATAT mcsqs and Analyse Its Outputs",
    page_icon="🎲",
    layout="wide"
)

from pymatgen.core import Structure
import io
import streamlit.components.v1 as components
import os
import re
from pymatgen.io.cif import CifWriter
import numpy as np
import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer



#import pkg_resources
#installed_packages = sorted([(d.project_name, d.version) for d in pkg_resources.working_set])
#st.subheader("Installed Python Modules")
#for package, version in installed_packages:
#    st.write(f"{package}=={version}")

if 'full_structures' not in st.session_state:
    st.session_state.full_structures = {}

if 'uploaded_files' not in st.session_state or st.session_state['uploaded_files'] is None:
    st.session_state['uploaded_files'] = []

if 'previous_uploaded_files' not in st.session_state:
    st.session_state['previous_uploaded_files'] = []

def load_structure(file):
    try:
        file_content = file.read()
        file.seek(0)


        with open(file.name, "wb") as f:
            f.write(file_content)

        structure = Structure.from_file(file.name)

        if os.path.exists(file.name):
            os.remove(file.name)

        return structure
    except Exception as e:
        st.error(f"Failed to parse {file.name}: {e}")
        raise e

def remove_fractional_occupancies_safely(structure):
    species = []
    coords = []

    for site in structure:
        if site.is_ordered:
            species.append(site.specie)
        else:
            dominant_sp = max(site.species.items(), key=lambda x: x[1])[0]
            species.append(dominant_sp)
        coords.append(site.frac_coords)

    ordered_structure = Structure(
        lattice=structure.lattice,
        species=species,
        coords=coords,
        coords_are_cartesian=False
    )

    return ordered_structure

st.markdown(
    """
    <style>
        /* Sidebar background with soft transparent red-green gradient */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                rgba(231, 76, 60, 0.15),   /* very soft red */
                rgba(46, 204, 113, 0.15)   /* very soft green */
            );
            backdrop-filter: blur(6px);  /* frosted glass effect */
        }

        /* Custom caption style */
        .sidebar-caption {
            font-size: 1.15rem;
            font-weight: 600;
            color: inherit;
            margin: 1rem 0 0.5rem 0;
            position: relative;
            display: inline-block;
        }

        .sidebar-caption::after {
            content: "";
            display: block;
            width: 100%;
            height: 3px;
            margin-top: 4px;
            border-radius: 2px;
            background: linear-gradient(to right, #e74c3c, #2ecc71);  /* vivid red → green underline */
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown('<div class="sidebar-caption">SQS ATAT</div>', unsafe_allow_html=True)

# File uploader in the sidebar


st.sidebar.subheader("📁 Upload Your Structure Files")
uploaded_files_user_sidebar = st.sidebar.file_uploader(
    "Upload Structure Files (CIF, POSCAR, LMP, extended XYZ):",
    type=None,
    accept_multiple_files=True,
    key="sidebar_uploader"
)


current_file_names = [file.name for file in uploaded_files_user_sidebar] if uploaded_files_user_sidebar else []
previous_file_names = [file.name for file in st.session_state['previous_uploaded_files']]

# Detect removed files
removed_files = set(previous_file_names) - set(current_file_names)


for removed_file in removed_files:
    if removed_file in st.session_state.full_structures:
        del st.session_state.full_structures[removed_file]
        st.success(f"Removed structure: {removed_file}")


st.session_state['uploaded_files'] = [
    file for file in st.session_state['uploaded_files']
    if file.name in current_file_names
]

import os
from ase.io import read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData
import streamlit as st


def load_structure(file):
    try:
        file_content = file.read()
        file.seek(0)  # Reset file pointer

        with open(file.name, "wb") as f:
            f.write(file_content)
        filename = file.name.lower()

        if filename.endswith(".cif"):
            mg_structure = Structure.from_file(file.name)
        elif filename.endswith(".data"):
            lmp_filename = file.name.replace(".data", ".lmp")
            os.rename(file.name, lmp_filename)
            lammps_data = LammpsData.from_file(lmp_filename, atom_style="atomic")
            mg_structure = lammps_data.structure
        elif filename.endswith(".lmp"):
            lammps_data = LammpsData.from_file(file.name, atom_style="atomic")
            mg_structure = lammps_data.structure
        else:
            atoms = read(file.name)
            mg_structure = AseAtomsAdaptor.get_structure(atoms)

        if os.path.exists(file.name):
            os.remove(file.name)

        return mg_structure

    except Exception as e:
        st.error(f"Failed to parse {file.name}: {e}")
        st.error(
            f"This does not work. Are you sure you tried to upload here the structure files (CIF, POSCAR, LMP, XSF, PW)? For the **experimental XY data**, put them to the other uploader\n"
            f"and please remove this wrongly placed file. 😊")
        raise e


def handle_uploaded_files(uploaded_files_user_sidebar):

    if uploaded_files_user_sidebar:
        for file in uploaded_files_user_sidebar:
            if file.name not in st.session_state.full_structures:
                try:
                    structure = load_structure(file)
                    st.session_state.full_structures[file.name] = structure
                    st.success(f"Successfully loaded structure: {file.name}")

                    # Add to uploaded_files list for tracking
                    if 'uploaded_files' not in st.session_state:
                        st.session_state['uploaded_files'] = []
                    if all(f.name != file.name for f in st.session_state['uploaded_files']):
                        st.session_state['uploaded_files'].append(file)

                except Exception as e:
                    pass

def update_file_upload_section():
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []
    if 'previous_uploaded_files' not in st.session_state:
        st.session_state['previous_uploaded_files'] = []

    current_file_names = [file.name for file in uploaded_files_user_sidebar] if uploaded_files_user_sidebar else []
    previous_file_names = [file.name for file in st.session_state['previous_uploaded_files']]

    removed_files = set(previous_file_names) - set(current_file_names)
    for removed_file in removed_files:
        if removed_file in st.session_state.full_structures:
            del st.session_state.full_structures[removed_file]
            st.success(f"Removed structure: {removed_file}")
    st.session_state['uploaded_files'] = [
        file for file in st.session_state['uploaded_files']
        if file.name in current_file_names
    ]

    handle_uploaded_files(uploaded_files_user_sidebar)


    st.session_state['previous_uploaded_files'] = uploaded_files_user_sidebar if uploaded_files_user_sidebar else []

    if st.session_state.full_structures:
        st.sidebar.subheader("📋 Currently Loaded Structures")
        for filename in st.session_state.full_structures.keys():
            st.sidebar.text(f"• {filename}")

st.sidebar.info(f"❤️🫶 **[Donations always appreciated!](https://buymeacoffee.com/bracerino)**")
st.sidebar.info(
    "Try also our XRD application **[XRDlicious](xrdlicious.com)** and **[SQS generation using ICET](https://icet-sqs.streamlit.app/)**. 🌀 Developed by **[IMPLANT team](https://implant.fs.cvut.cz/)**. 📺 **[Tutorial here](https://youtu.be/GGo_9T5wqus?si=xJItv-j0shr8hte_)**. Spot a bug or have a feature requests? Let us know at **lebedmi2@cvut.cz**."
    " You can consider to compile the app **locally** on your computer from **[GitHub](https://github.com/bracerino/atat-sqs-gui.git)** for better performance."
)

st.sidebar.link_button("GitHub page (for local compilation)", "https://github.com/bracerino/atat-sqs-gui.git", type="primary" )
update_file_upload_section()
st.session_state['previous_uploaded_files'] = uploaded_files_user_sidebar if uploaded_files_user_sidebar else []




# Render the SQS transformation module
from st_trans import render_sqs_module, check_sqs_mode

# Call the SQS module
render_sqs_module()


st.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
#def get_memory_usage():
#    process = psutil.Process(os.getpid())
#    mem_info = process.memory_info()
#    return mem_info.rss / (1024 ** 2)  # in MB

components.html(
    """
    <head>
        <meta name="description" content="ATAT SQS GUI: Create Input Files for Generation of SQS using ATAT mcsqs and Analyse Its Outputs">
    </head>
    """,
    height=0,
)

#memory_usage = get_memory_usage()
#st.write(
#    f"🔍 Current memory usage: **{memory_usage:.2f} MB**. We are now using free hosting by Streamlit Community Cloud servis, which has a limit for RAM memory of 2.6 GBs. For more extensive computations, please compile the application locally from the [GitHub](https://github.com/bracerino/atat-sqs-gui.git).")
st.markdown("""
**The GUI SQS application is open-source and released under the [MIT License](https://github.com/bracerino/atat-sqs-gui/blob/main/LICENSE).**
""")

st.markdown("""

### Acknowledgments

This project uses several open-source tools and datasets. We gratefully acknowledge their authors: **[ATAT](https://axelvandewalle.github.io/www-avdw/atat)** Licensed under the Creative Commons Attribution‑NoDerivatives 4.0 International License. **[Matminer](https://github.com/hackingmaterials/matminer)** Licensed under the [Modified BSD License](https://github.com/hackingmaterials/matminer/blob/main/LICENSE). **[Pymatgen](https://github.com/materialsproject/pymatgen)** Licensed under the [MIT License](https://github.com/materialsproject/pymatgen/blob/master/LICENSE).
 **[ASE (Atomic Simulation Environment)](https://gitlab.com/ase/ase)** Licensed under the [GNU Lesser General Public License (LGPL)](https://gitlab.com/ase/ase/-/blob/master/COPYING.LESSER). **[Py3DMol](https://github.com/avirshup/py3dmol/tree/master)** Licensed under the [BSD-style License](https://github.com/avirshup/py3dmol/blob/master/LICENSE.txt). **[Materials Project](https://next-gen.materialsproject.org/)** Data from the Materials Project is made available under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). **[AFLOW](http://aflow.org)** Licensed under the [GNU General Public License (GPL)](https://www.gnu.org/licenses/gpl-3.0.html)
 **[Crystallographic Open Database (COD)](https://www.crystallography.net/cod/)** under the CC0 license.
""")
