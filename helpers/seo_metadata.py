"""Search-engine metadata for the public online app.

Streamlit has no API for <head> tags: set_page_config only sets the <title>.
Streamlit Community Cloud serves crawlers a prerendered snapshot of the page
once the app has rendered, so tags that a script adds to document.head end up
in what Google indexes. The script runs through st.html(...,
unsafe_allow_javascript=True) (Streamlit >= 1.49; requirements pin 1.64),
which renders in the page itself rather than in an iframe, so `document.head`
is the app's own head.

Added tags:
    <meta name="description">        the snippet Google shows under the title
    <meta name="keywords">           ignored by Google, still read by Bing and others
    <meta name="robots">             index, follow
    <link rel="canonical">           the streamlit.app URL, without query strings
    og:/twitter: title, description  link previews (Cloud's own tags are overwritten)
    application/ld+json              schema.org SoftwareApplication

Each tag is updated in place if it already exists, so a rerun does not
duplicate it.
"""
import json

import streamlit as st

SEO_URL = "https://umlip-interactive.streamlit.app/"

SEO_TITLE = "MLIP-Interactive: Compute properties with universal MLIPs"

SEO_DESCRIPTION = (
    "Web GUI for atomistic simulations with universal machine learning "
    "interatomic potentials (uMLIPs): MACE, UMA, Orb-v3, SevenNet, MatterSim, "
    "CHGNet, GRACE, PET-MAD and more. Set up single-point energy, geometry "
    "optimization, molecular dynamics (NVE, NVT, NPT), phonons, elastic "
    "constants, equation of state and NEB calculations, and generate a "
    "ready-to-run Python (ASE) script."
)

SEO_KEYWORDS = [
    "MLIP", "uMLIP", "machine learning interatomic potential",
    "universal machine learning interatomic potentials", "machine learning force field",
    "MLIP simulation", "molecular dynamics", "MD simulation online",
    "atomistic simulation GUI", "ASE", "MACE", "MACE-MP", "UMA", "fairchem",
    "Orb-v3", "SevenNet", "MatterSim", "CHGNet", "GRACE", "PET-MAD", "NequIP",
    "Allegro", "DPA", "ALIGNN-FF", "phonons", "phonon calculation",
    "elastic constants", "Birch-Murnaghan equation of state", "geometry optimization",
    "nudged elastic band", "NEB", "metal-organic frameworks", "MOF",
    "materials science", "computational materials science", "DFT alternative",
    "POSCAR", "CIF",
]

SEO_JSON_LD = {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    "name": "MLIP-Interactive (uMLIP-Interactive)",
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
        "name": {
            "description": SEO_DESCRIPTION,
            "keywords": ", ".join(SEO_KEYWORDS),
            "robots": "index, follow",
            "twitter:title": SEO_TITLE,
            "twitter:description": SEO_DESCRIPTION,
        },
        "property": {
            "og:title": SEO_TITLE,
            "og:description": SEO_DESCRIPTION,
            "og:url": SEO_URL,
            "og:type": "website",
        },
    }
    # json.dumps gives valid JS literals; "</" is escaped so the payload cannot
    # close the surrounding <script> tag.
    meta_js = json.dumps(meta).replace("</", "<\\/")
    ld_js = json.dumps(json.dumps(SEO_JSON_LD)).replace("</", "<\\/")
    url_js = json.dumps(SEO_URL)
    return f"""<script>
(function () {{
  var head = document.head;
  var meta = {meta_js};
  Object.keys(meta).forEach(function (attr) {{
    Object.keys(meta[attr]).forEach(function (key) {{
      var el = head.querySelector('meta[' + attr + '="' + key + '"]');
      if (!el) {{
        el = document.createElement('meta');
        el.setAttribute(attr, key);
        head.appendChild(el);
      }}
      el.setAttribute('content', meta[attr][key]);
    }});
  }});
  var link = head.querySelector('link[rel="canonical"]');
  if (!link) {{
    link = document.createElement('link');
    link.setAttribute('rel', 'canonical');
    head.appendChild(link);
  }}
  link.setAttribute('href', {url_js});
  var ld = head.querySelector('script[data-mlip-seo]');
  if (!ld) {{
    ld = document.createElement('script');
    ld.type = 'application/ld+json';
    ld.setAttribute('data-mlip-seo', '1');
    head.appendChild(ld);
  }}
  ld.textContent = {ld_js};
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
