"""
Online / public-demo entry point for the MLIP-Interactive app.

Run it exactly like the normal app, but pointing at this file:

    streamlit run app_online.py

It is a thin launcher around ``app.py``: it sets the ``MLIP_ONLINE_MODE``
environment variable and then executes ``app.py`` unchanged. Because there is a
single source of truth (``app.py``), every setting and feature stays identical
to the local app — the online flag only:

  * disables uploading structures in the interface,
  * disables running simulations in the interface,
  * keeps generating the standalone Python script fully available, and
  * shows guidance to compile locally (to run in-interface) and how to run the
    generated standalone script (create a venv, install the packages for the
    selected MLIP, drop the structures next to the script, and run it).

To run simulations and upload structures in the interface, users should compile
the app locally: https://github.com/bracerino/uMLIP-Interactive
"""

import os
import sys
import runpy

# Enable online/demo behaviour. app.py reads this and gates upload + running.
os.environ["MLIP_ONLINE_MODE"] = "1"

_HERE = os.path.dirname(os.path.abspath(__file__))

# Make sure the project directory (helpers/, phonon_calculator.py, ...) is
# importable when app.py is executed below.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Execute app.py as if it were the main script, so Streamlit renders it.
runpy.run_path(os.path.join(_HERE, "app.py"), run_name="__main__")
