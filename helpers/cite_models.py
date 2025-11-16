import streamlit as st
def create_citation_info():
        st.markdown("""
        <style>
        .model-section {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-left: 5px solid #667eea;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .model-title {
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .model-icon {
            font-size: 24px;
        }
        .link-container {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 10px;
        }
        .citation-link {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .citation-link:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .github-link {
            background: linear-gradient(135deg, #24292e 0%, #000000 100%);
            color: white !important;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .github-link:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }
        .paper-icon::before {
            content: "📄";
            font-size: 16px;
        }
        .github-icon::before {
            content: "⚙️";
            font-size: 16px;
        }
        </style>

        <div class="model-section" style="border-left-color: #4CAF50;">
            <div class="model-title">
                <span class="model-icon">1️⃣</span>
                <span>MACE-MP</span>
            </div>
            <div class="link-container">
                <a href="https://arxiv.org/abs/2401.00096" target="_blank" class="citation-link">
                    <span class="paper-icon"></span> arXiv paper, 2025
                </a>
                <a href="https://github.com/ACEsuit/mace-foundations" target="_blank" class="github-link">
                    <span class="github-icon"></span> GitHub
                </a>
            </div>
        </div>

        <div class="model-section" style="border-left-color: #FF9800;">
            <div class="model-title">
                <span class="model-icon">2️⃣</span>
                <span>MACE-OFF (Organic)</span>
            </div>
            <div class="link-container">
                <a href="https://pubs.acs.org/doi/abs/10.1021/jacs.4c07099" target="_blank" class="citation-link">
                    <span class="paper-icon"></span> JACS MACE-OFF Paper, 2025
                </a>
                <a href="https://github.com/ACEsuit/mace-off" target="_blank" class="github-link">
                    <span class="github-icon"></span> GitHub
                </a>
            </div>
        </div>

        <div class="model-section" style="border-left-color: #9C27B0;">
            <div class="model-title">
                <span class="model-icon">3️⃣</span>
                <span>MACE-MH (Multi-Head Foundation)</span>
            </div>
            <div class="link-container">
                <a href="https://arxiv.org/abs/2510.25380" target="_blank" class="citation-link">
                    <span class="paper-icon"></span> arXiv Paper, 2025
                </a>
                <a href="https://github.com/ACEsuit/mace-foundations" target="_blank" class="github-link">
                    <span class="github-icon"></span> GitHub
                </a>
            </div>
        </div>

        <div class="model-section" style="border-left-color: #2196F3;">
            <div class="model-title">
                <span class="model-icon">4️⃣</span>
                <span>CHGNet</span>
            </div>
            <div class="link-container">
                <a href="https://www.nature.com/articles/s42256-023-00716-3" target="_blank" class="citation-link">
                    <span class="paper-icon"></span> Nature Machine Intelligence, 2023
                </a>
                <a href="https://github.com/CederGroupHub/chgnet" target="_blank" class="github-link">
                    <span class="github-icon"></span> GitHub
                </a>
            </div>
        </div>

        <div class="model-section" style="border-left-color: #FF5722;">
            <div class="model-title">
                <span class="model-icon">5️⃣</span>
                <span>SevenNet</span>
            </div>
            <div class="link-container">
                <a href="https://pubs.acs.org/doi/full/10.1021/jacs.4c14455?casa_token=4BHW_eAM-5gAAAAA%3AgvZXUbXDbgWtlsARu2CMr6cOwu_2dGrN2gLhf3jxFEWSDMXuyu4FZwCxoKaJHJYxEhz_clsSQ-jPfwVRig8" target="_blank" class="citation-link">
                    <span class="paper-icon"></span> JACS Paper, 2024
                </a>
                <a href="https://github.com/MDIL-SNU/SevenNet" target="_blank" class="github-link">
                    <span class="github-icon"></span> GitHub
                </a>
            </div>
        </div>

        <div class="model-section" style="border-left-color: #00BCD4;">
            <div class="model-title">
                <span class="model-icon">6️⃣</span>
                <span>MatterSim</span>
            </div>
            <div class="link-container">
                <a href="https://arxiv.org/abs/2405.04967" target="_blank" class="citation-link">
                    <span class="paper-icon"></span> arXiv Paper, 2024
                </a>
                <a href="https://github.com/microsoft/mattersim" target="_blank" class="github-link">
                    <span class="github-icon"></span> GitHub
                </a>
            </div>
        </div>

        <div class="model-section" style="border-left-color: #E91E63;">
            <div class="model-title">
                <span class="model-icon">7️⃣</span>
                <span>ORB-v3 Models</span>
            </div>
            <div class="link-container">
                <a href="https://arxiv.org/abs/2504.06231" target="_blank" class="citation-link">
                    <span class="paper-icon"></span> arXiv Paper, 2025
                </a>
                <a href="https://github.com/orbital-materials/orb-models" target="_blank" class="github-link">
                    <span class="github-icon"></span> GitHub
                </a>
            </div>
        </div>

        <div class="model-section" style="border-left-color: #795548;">
            <div class="model-title">
                <span class="model-icon">8️⃣</span>
                <span>Nequix</span>
            </div>
            <div class="link-container">
                <a href="https://arxiv.org/abs/2508.16067" target="_blank" class="citation-link">
                    <span class="paper-icon"></span> arXiv Paper, 2025
                </a>
                <a href="https://github.com/atomicarchitects/nequix" target="_blank" class="github-link">
                    <span class="github-icon"></span> GitHub
                </a>
            </div>
        </div>
        
        <div class="model-section" style="border-left-color: #E91E63;">
            <div class="model-title">
                <span class="model-icon">9️⃣</span>
                <span>PET-MAD</span>
            </div>
            <div class="link-container">
                <a href="https://arxiv.org/abs/2503.14118" target="_blank" class="citation-link">
                    <span class="paper-icon"></span> arXiv Paper, 2025
                </a>
                <a href="https://github.com/lab-cosmo/pet-mad" target="_blank" class="github-link">
                    <span class="github-icon"></span> GitHub
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
