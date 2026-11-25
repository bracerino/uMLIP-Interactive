"""Search-engine metadata for the public online app.

Streamlit has no API for <head> tags: set_page_config only sets the <title>.
Streamlit Community Cloud serves crawlers a prerendered snapshot of the page
once the app has rendered, so tags that a script adds to document.head end up
in what Google indexes. The script runs through st.html(...,
unsafe_allow_javascript=True) (Streamlit >= 1.49; requirements pin 1.64),
which renders in the page itself rather than in an iframe, so `document.head`
is the app's own head.

Added tags (same approach as XRDlicious):
    <meta name="description">        the snippet Google shows under the title
    <meta name="keywords">           ignored by Google, still read by Bing and others
    <meta name="robots">             index, follow
    application/ld+json              schema.org SoftwareApplication

Streamlit's <noscript> text ("You need to enable JavaScript to run this app"
and its note for AI agents) is removed, so search engines do not use it as the
page text or snippet.

Canonical, og: and twitter: tags are left to Streamlit Community Cloud, which
adds its own after the app renders; adding them here too gave duplicate,
conflicting tags (two different canonical URLs).

Each tag is updated in place if it already exists, so a rerun does not
duplicate it.
"""
import json

import streamlit as st

SEO_URL = "https://umlip-interactive.streamlit.app"

SEO_TITLE = (
    "uMLIP-Interactive: Online Generator of Python Scripts for Phonons, Elastic Constants, MD, "
    "NEB and Geometry Optimization with Universal Machine-Learning Interatomic Potentials "
    "(MACE, UMA, Orb, SevenNet, ...)"
)

SEO_DESCRIPTION = (
    "uMLIP-Interactive, Online Generator of Python Scripts for Calculations with Universal "
    "Machine-Learning Interatomic Potentials (uMLIPs: MACE, UMA, Orb-v3, SevenNet, MatterSim, "
    "CHGNet, GRACE, PET-MAD, ...): Single-Point Energy, Geometry Optimization, Phonons, "
    "Elastic Constants, Equation of State, Molecular Dynamics (MD) and NEB from Crystal "
    "Structures (CIF, POSCAR, LMP, XYZ ...)"
)

SEO_KEYWORDS = [
    "MLIP", "uMLIP", "uMLIP calculation", "uMLIP script generator", "MLIP Python script generator",
    "machine learning interatomic potential",
    "universal machine learning interatomic potentials", "machine learning force field",
    "MLIP simulation", "molecular dynamics", "MD simulation online",
    "atomistic simulation GUI", "ASE", "MACE", "MACE-MP", "UMA", "fairchem",
    "Orb-v3", "SevenNet", "MatterSim", "CHGNet", "GRACE", "PET-MAD", "NequIP",
    "Allegro", "Nequix", "UPET", "Prophet", "DPA", "ALIGNN-FF", "phonons", "phonon calculation",
    "elastic constants", "Birch-Murnaghan equation of state", "geometry optimization",
    "nudged elastic band", "NEB", "metal-organic frameworks", "MOF",
    "materials science", "computational materials science", "DFT alternative",
    "POSCAR", "CIF",
]

SEO_JSON_LD = {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    "name": "uMLIP-Interactive",
    "url": SEO_URL,
    "description": SEO_DESCRIPTION,
    "applicationCategory": "ScientificApplication",
    "applicationSubCategory": "Atomistic simulation",
    "operatingSystem": "Web browser, Linux, Windows (WSL), macOS",
    "keywords": ", ".join(SEO_KEYWORDS),
    "codeRepository": "https://github.com/bracerino/uMLIP-Interactive",
    "author": {"@type": "Person", "name": "Miroslav Lebeda"},
    "citation": "https://doi.org/10.1016/j.jmrt.2026.02.204",
}

_CONTAINER_KEY = "mlip_seo_meta"


def _head_script():
    meta = {
        "description": SEO_DESCRIPTION,
        "keywords": ", ".join(SEO_KEYWORDS),
        "robots": "index, follow",
    }
    # json.dumps gives valid JS literals; "</" is escaped so the payload cannot
    # close the surrounding <script> tag.
    meta_js = json.dumps(meta).replace("</", "<\\/")
    ld_js = json.dumps(json.dumps(SEO_JSON_LD)).replace("</", "<\\/")
    return f"""<script>
(function () {{
  var head = document.head;
  var meta = {meta_js};
  Object.keys(meta).forEach(function (key) {{
    var el = head.querySelector('meta[name="' + key + '"]');
    if (!el) {{
      el = document.createElement('meta');
      el.setAttribute('name', key);
      head.appendChild(el);
    }}
    el.setAttribute('content', meta[key]);
  }});
  var ld = head.querySelector('script[data-mlip-seo]');
  if (!ld) {{
    ld = document.createElement('script');
    ld.type = 'application/ld+json';
    ld.setAttribute('data-mlip-seo', '1');
    head.appendChild(ld);
  }}
  ld.textContent = {ld_js};
  document.querySelectorAll('noscript').forEach(function (el) {{ el.remove(); }});
}})();
</script>"""


def inject_seo_metadata():
    """Add the search metadata to the page <head>. Call once per script run."""
    # A style-only st.html goes to the event container and takes no space. The
    # script block cannot, so it sits in a keyed container that this rule
    # hides; scripts inside a hidden element still run. The container's
    # stLayoutWrapper parent is hidden too, or it still takes a flex gap.
    st.html(
        f"<style>.st-key-{_CONTAINER_KEY}, "
        f'[data-testid="stLayoutWrapper"]:has(> .st-key-{_CONTAINER_KEY}) '
        "{ display: none; }</style>"
    )
    with st.container(key=_CONTAINER_KEY):
        st.html(_head_script(), unsafe_allow_javascript=True)
