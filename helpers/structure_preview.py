import streamlit as st
import py3Dmol
from pymatgen.io.cif import CifWriter


def view_structure(structure, height=300, width=400):
    try:
        cif_writer = CifWriter(structure)
        cif_string = str(cif_writer)

        view = py3Dmol.view(width=width, height=height)
        view.addModel(cif_string, 'cif')
        view.setStyle({'sphere': {'scale': 0.6}})
        view.addUnitCell()
        view.zoomTo()

        return view._make_html()
    except:
        return f"<div style='height:{height}px;width:{width}px;background-color:#f0f0f0;display:flex;align-items:center;justify-content:center;'>Structure preview unavailable</div>"


def render_structure_preview(structures):
    """Render a preview of the provided structures.

    Works independently of the selected model family — it only shows
    structural information (geometry, formula, lattice, elements) without
    any model compatibility checks.
    """
    st.header("2. Structure Preview")

    for i, (name, structure) in enumerate(structures.items()):
        with st.expander(f"Structure {i + 1}: {name}"):
            col1, col2 = st.columns([1, 1.5])

            with col1:
                st.iframe(view_structure(structure, height=250, width=350), height=260)

            with col2:
                st.write(f"**Formula:** {structure.composition.reduced_formula}")
                st.write(f"**Number of atoms:** {structure.num_sites}")
                st.write(f"**Lattice parameters:**")
                st.write(f"  a = {structure.lattice.a:.3f} Å")
                st.write(f"  b = {structure.lattice.b:.3f} Å")
                st.write(f"  c = {structure.lattice.c:.3f} Å")

                elements = list(set([site.specie.symbol for site in structure]))
                st.write(f"**Elements:** {', '.join(elements)}")
                if hasattr(structure, 'constraints_info') and structure.constraints_info:
                    st.write("📌 **Selective Dynamics:** Present (some atoms fixed)")
                else:
                    st.write("🔄 **Selective Dynamics:** None (all atoms free to move)")
