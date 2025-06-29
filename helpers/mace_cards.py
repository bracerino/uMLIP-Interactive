import streamlit as st
import pandas as pd

# MACE Models Information Database
MACE_MODELS_INFO = {
    "MACE-MP-0b3 (medium) - Latest": {
        "official_name": "MACE-MP-0b3",
        "size": "medium",
        "elements": 89,
        "training_dataset": "MPTrj",
        "level_of_theory": "DFT (PBE+U)",
        "target_system": "Materials",
        "version_required": ">=v0.3.9",
        "description": "Latest recommended model with fixed phonon issues compared to b2",
        "features": ["High stability", "Improved phonons", "Materials chemistry"],
        "license": "MIT",
        "color": "#4ECDC4",  # Teal - Latest/Recommended
        "recommended": True
    },
    "MACE-MP-0 (small) - Original": {
        "official_name": "MACE-MP-0",
        "size": "small",
        "elements": 89,
        "training_dataset": "MPTrj",
        "level_of_theory": "DFT (PBE+U)",
        "target_system": "Materials",
        "version_required": ">=v0.3.6",
        "description": "Initial foundation model release, lightweight version",
        "features": ["Fast inference", "Lower memory", "Baseline accuracy"],
        "license": "MIT",
        "color": "#45B7D1",  # Blue
        "recommended": False
    },
    "MACE-MP-0 (medium) - Original": {
        "official_name": "MACE-MP-0",
        "size": "medium",
        "elements": 89,
        "training_dataset": "MPTrj",
        "level_of_theory": "DFT (PBE+U)",
        "target_system": "Materials",
        "version_required": ">=v0.3.6",
        "description": "Initial foundation model release, balanced performance",
        "features": ["Balanced accuracy", "Moderate speed", "General purpose"],
        "license": "MIT",
        "color": "#96CEB4",  # Green
        "recommended": False
    },
    "MACE-MP-0 (large) - Original": {
        "official_name": "MACE-MP-0",
        "size": "large",
        "elements": 89,
        "training_dataset": "MPTrj",
        "level_of_theory": "DFT (PBE+U)",
        "target_system": "Materials",
        "version_required": ">=v0.3.6",
        "description": "Initial foundation model release, highest accuracy",
        "features": ["High accuracy", "Slower inference", "Complex systems"],
        "license": "MIT",
        "color": "#A8E6CF",  # Light Green
        "recommended": False
    },
    "MACE-MP-0b (small) - Improved": {
        "official_name": "MACE-MP-0b",
        "size": "small",
        "elements": 89,
        "training_dataset": "MPTrj",
        "level_of_theory": "DFT (PBE+U)",
        "target_system": "Materials",
        "version_required": ">=v0.3.10",
        "description": "Improved pair repulsion and corrected isolated atoms",
        "features": ["Better repulsion", "Corrected atoms", "Fast inference"],
        "license": "MIT",
        "color": "#87CEEB",  # Sky Blue
        "recommended": False
    },
    "MACE-MP-0b (medium) - Improved": {
        "official_name": "MACE-MP-0b",
        "size": "medium",
        "elements": 89,
        "training_dataset": "MPTrj",
        "level_of_theory": "DFT (PBE+U)",
        "target_system": "Materials",
        "version_required": ">=v0.3.10",
        "description": "Improved pair repulsion and corrected isolated atoms",
        "features": ["Better repulsion", "Corrected atoms", "Balanced performance"],
        "license": "MIT",
        "color": "#98D8E8",  # Light Blue
        "recommended": False
    },
    "MACE-MP-0b2 (small) - Enhanced": {
        "official_name": "MACE-MP-0b2",
        "size": "small",
        "elements": 89,
        "training_dataset": "MPTrj",
        "level_of_theory": "DFT (PBE+U)",
        "target_system": "Materials",
        "version_required": ">=v0.3.9",
        "description": "Enhanced stability at high pressure conditions",
        "features": ["High pressure stable", "MD simulations", "Fast inference"],
        "license": "MIT",
        "color": "#B6D7FF",  # Powder Blue
        "recommended": False
    },
    "MACE-MP-0b2 (medium) - Enhanced": {
        "official_name": "MACE-MP-0b2",
        "size": "medium",
        "elements": 89,
        "training_dataset": "MPTrj",
        "level_of_theory": "DFT (PBE+U)",
        "target_system": "Materials",
        "version_required": ">=v0.3.9",
        "description": "Enhanced stability at high pressure conditions",
        "features": ["High pressure stable", "MD simulations", "Balanced accuracy"],
        "license": "MIT",
        "color": "#C8E6C9",  # Light Green
        "recommended": False
    },
    "MACE-MP-0b2 (large) - Enhanced": {
        "official_name": "MACE-MP-0b2",
        "size": "large",
        "elements": 89,
        "training_dataset": "MPTrj",
        "level_of_theory": "DFT (PBE+U)",
        "target_system": "Materials",
        "version_required": ">=v0.3.9",
        "description": "Enhanced stability at high pressure conditions",
        "features": ["High pressure stable", "High accuracy", "Complex systems"],
        "license": "MIT",
        "color": "#DCEDC8",  # Very Light Green
        "recommended": False
    },
    "MACE-MPA-0 (medium) - Latest": {
        "official_name": "MACE-MPA-0",
        "size": "medium",
        "elements": 89,
        "training_dataset": "MPTrj + sAlex",
        "level_of_theory": "DFT (PBE+U)",
        "target_system": "Materials",
        "version_required": ">=v0.3.10",
        "description": "State-of-the-art accuracy with enlarged 3.5M crystal dataset",
        "features": ["Matbench SOTA", "3.5M crystals", "Best accuracy"],
        "license": "MIT",
        "color": "#FF6B6B",  # Red - High Performance
        "recommended": True
    },
    "MACE-OMAT-0 (medium)": {
        "official_name": "MACE-OMAT-0",
        "size": "medium",
        "elements": 89,
        "training_dataset": "OMAT",
        "level_of_theory": "DFT (PBE+U) VASP 54",
        "target_system": "Materials",
        "version_required": ">=v0.3.10",
        "description": "Excellent phonon properties with OMAT dataset",
        "features": ["Excellent phonons", "VASP 54", "Vibrational analysis"],
        "license": "ASL",
        "color": "#DDA0DD",  # Plum - Specialized
        "recommended": False
    },
    "MACE-MATPES-PBE-0 (medium) - No +U": {
        "official_name": "MACE-MATPES-PBE-0",
        "size": "medium",
        "elements": 89,
        "training_dataset": "MATPES-PBE",
        "level_of_theory": "DFT (PBE)",
        "target_system": "Materials",
        "version_required": ">=v0.3.10",
        "description": "Pure PBE functional without +U correction",
        "features": ["No +U correction", "Pure PBE", "Clean DFT"],
        "license": "ASL",
        "color": "#F0E68C",  # Khaki - Alternative
        "recommended": False
    },
    "MACE-MATPES-r2SCAN-0 (medium) - r2SCAN": {
        "official_name": "MACE-MATPES-r2SCAN-0",
        "size": "medium",
        "elements": 89,
        "training_dataset": "MATPES-r2SCAN",
        "level_of_theory": "DFT (r2SCAN)",
        "target_system": "Materials",
        "version_required": ">=v0.3.10",
        "description": "Advanced r2SCAN functional for better materials description",
        "features": ["r2SCAN functional", "Better materials", "Advanced DFT"],
        "license": "ASL",
        "color": "#FFA07A",  # Light Salmon - Advanced
        "recommended": False
    },
    "MACE-OFF23 (small) - Organic": {
        "official_name": "MACE-OFF23",
        "size": "small",
        "elements": 10,
        "element_list": "H, C, N, O, F, P, S, Cl, Br, I",
        "training_dataset": "SPICE",
        "level_of_theory": "DFT (ωB97M-D3)",
        "target_system": "Organic molecules",
        "version_required": ">=v0.3.6",
        "description": "Fast organic force field for drug-like molecules",
        "features": ["Fast inference", "Organic chemistry", "Drug-like molecules"],
        "license": "ASL",
        "recommended": False,
        "model_type": "MACE-OFF"
    },
    "MACE-OFF23 (medium) - Organic": {
        "official_name": "MACE-OFF23",
        "size": "medium",
        "elements": 10,
        "element_list": "H, C, N, O, F, P, S, Cl, Br, I",
        "training_dataset": "SPICE",
        "level_of_theory": "DFT (ωB97M-D3)",
        "target_system": "Organic molecules",
        "version_required": ">=v0.3.6",
        "description": "Balanced organic force field for biomolecular simulations",
        "features": ["Balanced accuracy", "Biomolecules", "Molecular dynamics"],
        "license": "ASL",
        "recommended": True,
        "model_type": "MACE-OFF"
    },
    "MACE-OFF23 (large) - Organic": {
        "official_name": "MACE-OFF23",
        "size": "large",
        "elements": 10,
        "element_list": "H, C, N, O, F, P, S, Cl, Br, I",
        "training_dataset": "SPICE",
        "level_of_theory": "DFT (ωB97M-D3)",
        "target_system": "Organic molecules",
        "version_required": ">=v0.3.6",
        "description": "High-accuracy organic force field for demanding applications",
        "features": ["High accuracy", "Complex organics", "Torsion barriers"],
        "license": "ASL",
        "recommended": False,
        "model_type": "MACE-OFF"
    }
}


def display_mace_models_info():
    """Display comprehensive information about MACE foundation models"""
    
    st.markdown("""
    <style>
    .model-card {
        width: 100%;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("🔬 MACE Foundation Models")
    st.markdown("""
    **MACE** (Machine Learning Assisted Chemical Engineering) foundation models are pre-trained universal 
    neural network potentials for atomistic materials chemistry, covering **89 chemical elements**.
    """)
    
    gen_tabs = st.tabs(["📖 Model Comparison"])
    
    
    with gen_tabs[0]:
        st.subheader("📊 Model Comparison Table")
        
        comparison_data = []
        for model_name, info in MACE_MODELS_INFO.items():
            comparison_data.append({
                "Model": model_name,
                "Official Name": info["official_name"],
                "Size": info["size"].capitalize(),
                "Training Dataset": info["training_dataset"],
                "Level of Theory": info["level_of_theory"],
                "Version Required": info["version_required"],
                "License": info["license"],
                "Personal Recommendation": "⭐" if info.get("recommended", False) else ""
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        
def create_citation_info():
    """Display citation information for MACE models"""
    
    st.subheader("📚 Citation Information")
    
    citation_text = """@article{batatia2023foundation,
    title={A foundation model for atomistic materials chemistry},
    author={Ilyes Batatia and Philipp Benner and Yuan Chiang and Alin M. Elena and others},
    year={2023},
    eprint={2401.00096},
    archivePrefix={arXiv},
    primaryClass={physics.chem-ph}
}"""
    
    col_cite1, col_cite2 = st.columns([2, 1])
    
    with col_cite1:
        st.code(citation_text, language="bibtex")
    
    with col_cite2:
        st.markdown("""
        **Additional Resources:**
        
        🔗 [GitHub Repository](https://github.com/ACEsuit/mace-foundations)
        
        🤗 [Hugging Face Models](https://huggingface.co/mace-foundations)
        
        📖 [MACE Documentation](https://mace-docs.readthedocs.io)
        
        📊 [Training Data (MPTrj)](https://figshare.com/articles/dataset/MPtrj/23713842)
        """)

def display_models_info_tab():
    display_mace_models_info()
    
    st.markdown("---")
    
    create_citation_info()
    
    st.markdown("---")
    
