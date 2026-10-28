"""MACE-MDP support: molecular dipole moments and polarizabilities.

MACE-MDP (mace-torch >= 0.3.17, ``mace.calculators.mace_mdp``) is a
DipolePolarizabilityMACE model for organic molecules. It predicts the dipole
moment, the polarizability tensor and per-atom charges, and nothing else: no
energy, no forces, no stress. It can therefore only run its own calculation
type, which the interface switches to automatically when the model is picked.

Units, as checked against experiment (H2O: 1.88 D / 1.47 A^3, benzene 10.2 A^3):
    dipole          e*A           (x 4.80320 -> Debye)
    polarizability  e*A^2/V       (x 14.3996 -> A^3, i.e. alpha / 4 pi eps0)
    charges         e   - auxiliary model charges, NOT chemical partial charges:
                          the dipole is mostly a sum of per-atom dipole vectors,
                          and sum(q_i r_i) adds only a few percent to it

The property extraction is kept as source (MDP_PROPERTIES_SRC) so the interface
and the generated standalone scripts run exactly the same code: the app execs
it, the script embeds it.
"""

# The label must not contain "POLAR": the app and the script generators route a
# model through the MACE-POLAR-1 settings (charge, spin, field) by that substring,
# and "polarizability" would match it.
MDP_MODEL_KEY = "MACE-MDP - Dipoles & α tensors only, organic molecules [10 elem]"
MDP_MODEL_VALUE = "mace:mdp"
MDP_CALC_TYPE = "Dipole & Polarizability"
MDP_ELEMENTS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
MDP_MIN_MACE_VERSION = "0.3.17"


def is_mdp_model(selected_model_key=None, model_size=None):
    if model_size == MDP_MODEL_VALUE:
        return True
    return selected_model_key == MDP_MODEL_KEY


def mdp_unsupported_elements(symbols):
    """Elements in ``symbols`` the model was not trained on (sorted, unique)."""
    return sorted(set(symbols) - set(MDP_ELEMENTS))


MDP_PROPERTIES_SRC = r'''
import numpy as _mdp_np

MDP_E_ANG_TO_DEBYE = 4.80320471   # 1 e*A in Debye
MDP_POL_TO_ANG3 = 14.3996454      # 1 e*A^2/V -> A^3 (alpha / 4 pi eps0)
MDP_ELEMENTS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]


def mdp_properties(atoms, calculator):
    """Dipole, polarizability and charges of one structure from MACE-MDP.

    Returns a dict of plain floats/lists (JSON-ready). Raises ValueError for an
    element the model does not know - the model itself would only report
    "<Z> is not in list".
    """
    symbols = atoms.get_chemical_symbols()
    unknown = sorted(set(symbols) - set(MDP_ELEMENTS))
    if unknown:
        raise ValueError("MACE-MDP supports only %s; this structure also has %s"
                         % (", ".join(MDP_ELEMENTS), ", ".join(unknown)))
    atoms.calc = calculator
    calculator.get_property("dipole", atoms)   # one pass fills every property
    results = calculator.results

    dipole = _mdp_np.asarray(results["dipole"], dtype=float).reshape(3)
    alpha = _mdp_np.asarray(results["polarizability"], dtype=float).reshape(3, 3)
    alpha_sym = 0.5 * (alpha + alpha.T)
    eigenvalues = _mdp_np.linalg.eigvalsh(alpha_sym)
    a1, a2, a3 = eigenvalues
    anisotropy = float(_mdp_np.sqrt(0.5 * ((a1 - a2) ** 2 + (a2 - a3) ** 2 + (a3 - a1) ** 2)))
    charges = _mdp_np.asarray(results.get("charges", []), dtype=float).reshape(-1)

    return {
        "success": True,
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "symbols": symbols,
        "dipole_eA": dipole.tolist(),
        "dipole_D": (dipole * MDP_E_ANG_TO_DEBYE).tolist(),
        "dipole_norm_eA": float(_mdp_np.linalg.norm(dipole)),
        "dipole_norm_D": float(_mdp_np.linalg.norm(dipole) * MDP_E_ANG_TO_DEBYE),
        "polarizability_eA2_per_V": alpha.tolist(),
        "polarizability_A3": (alpha * MDP_POL_TO_ANG3).tolist(),
        "polarizability_iso_A3": float(_mdp_np.trace(alpha) / 3.0 * MDP_POL_TO_ANG3),
        "polarizability_eigenvalues_A3": (eigenvalues * MDP_POL_TO_ANG3).tolist(),
        "polarizability_anisotropy_A3": anisotropy * MDP_POL_TO_ANG3,
        "charges_e": charges.tolist(),
        "charge_sum_e": float(charges.sum()) if charges.size else 0.0,
    }


def mdp_summary_lines(props):
    """Human-readable lines for a log or console."""
    mx, my, mz = props["dipole_D"]
    axx, ayy, azz = (props["polarizability_A3"][k][k] for k in range(3))
    return [
        "  Dipole moment: |mu| = %.4f D  (%.4f e*A)" % (props["dipole_norm_D"],
                                                     props["dipole_norm_eA"]),
        "     mu = (%.4f, %.4f, %.4f) D" % (mx, my, mz),
        "  Polarizability: alpha_iso = %.4f A^3, anisotropy = %.4f A^3"
        % (props["polarizability_iso_A3"], props["polarizability_anisotropy_A3"]),
        "     alpha_xx, alpha_yy, alpha_zz = %.4f, %.4f, %.4f A^3" % (axx, ayy, azz),
        "  Model charges (not partial charges): sum = %.4f e over %d atoms"
        % (props["charge_sum_e"], props["n_atoms"]),
    ]
'''

_MDP_NS = {}


def mdp_runtime():
    """mdp_properties / mdp_summary_lines for in-app use (same source as the scripts)."""
    if not _MDP_NS:
        exec(MDP_PROPERTIES_SRC, _MDP_NS)
    return _MDP_NS


def build_mdp_calculator(device="cpu", dtype="float64", enable_cueq=False):
    """The ASE calculator for a live in-app run."""
    try:
        from mace.calculators import mace_mdp
    except ImportError as exc:
        raise RuntimeError(
            f"MACE-MDP needs mace-torch >= {MDP_MIN_MACE_VERSION} "
            f"(pip install mace-torch=={MDP_MIN_MACE_VERSION})") from exc
    import warnings
    with warnings.catch_warnings():
        # "not suitable for energies or forces" - the interface already says so.
        warnings.filterwarnings("ignore", message="The MACE-MDP model is designed")
        kwargs = {"enable_cueq": True} if enable_cueq and device == "cuda" else {}
        return mace_mdp(device=device, default_dtype=dtype, **kwargs)


def generate_mdp_calculator_code(device="cpu", dtype="float64", enable_cueq=False,
                                 indent="    "):
    """Calculator-setup block for a standalone script (defines `calculator`)."""
    cueq = ", enable_cueq=True" if enable_cueq and device == "cuda" else ""
    body = f'''import warnings as _mdp_warnings
try:
    from mace.calculators import mace_mdp
except ImportError:
    raise SystemExit("MACE-MDP needs mace-torch >= {MDP_MIN_MACE_VERSION}: "
                     "pip install mace-torch=={MDP_MIN_MACE_VERSION}")
print("Setting up MACE-MDP (dipole & polarizability) calculator...")
with _mdp_warnings.catch_warnings():
    _mdp_warnings.filterwarnings("ignore", message="The MACE-MDP model is designed")
    try:
        calculator = mace_mdp(device="{device}", default_dtype="{dtype}"{cueq})
        print("✅ MACE-MDP initialized on {device}")
    except Exception as e:
        print(f"❌ MACE-MDP initialization failed on {device}: {{e}}")
        print("Attempting fallback to CPU...")
        calculator = mace_mdp(device="cpu", default_dtype="{dtype}")
        print("✅ MACE-MDP initialized on CPU (fallback)")
'''
    return "".join(indent + line if line.strip() else line
                   for line in body.splitlines(keepends=True))


def generate_mdp_calculation_code():
    """The calculation block of a standalone script: every structure file in the
    folder -> dipole, polarizability and charges, written to mdp_results/."""
    return '''    structure_files = sorted([f for f in os.listdir(".") if f.startswith("POSCAR") or f.endswith(".vasp") or f.endswith(".poscar") or f.endswith(".cif") or f.endswith(".xyz") or f.endswith(".extxyz")])
    print(f"📊 Found {len(structure_files)} structure files")
    Path("mdp_results").mkdir(exist_ok=True)
''' + "\n".join("    " + line if line.strip() else line
               for line in MDP_PROPERTIES_SRC.strip("\n").splitlines()) + '''

    summary_rows, all_results = [], {}
    for i, filename in enumerate(structure_files):
        print(f"\\n📊 Processing structure {i+1}/{len(structure_files)}: {filename}")
        try:
            atoms = read(filename)
            props = mdp_properties(atoms, calculator)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            summary_rows.append({"structure": filename, "error": str(e)})
            continue
        for line in mdp_summary_lines(props):
            print(line)
        all_results[filename] = props
        base = os.path.splitext(os.path.basename(filename))[0]
        pd.DataFrame({
            "atom_index": list(range(props["n_atoms"])),
            "element": props["symbols"],
            "charge_e": props["charges_e"],
        }).to_csv(f"mdp_results/{base}_charges.csv", index=False)
        alpha = props["polarizability_A3"]
        summary_rows.append({
            "structure": filename,
            "formula": props["formula"],
            "n_atoms": props["n_atoms"],
            "dipole_D": props["dipole_norm_D"],
            "mu_x_D": props["dipole_D"][0],
            "mu_y_D": props["dipole_D"][1],
            "mu_z_D": props["dipole_D"][2],
            "dipole_eA": props["dipole_norm_eA"],
            "alpha_iso_A3": props["polarizability_iso_A3"],
            "alpha_aniso_A3": props["polarizability_anisotropy_A3"],
            "alpha_xx_A3": alpha[0][0], "alpha_yy_A3": alpha[1][1], "alpha_zz_A3": alpha[2][2],
            "alpha_xy_A3": alpha[0][1], "alpha_xz_A3": alpha[0][2], "alpha_yz_A3": alpha[1][2],
            "charge_sum_e": props["charge_sum_e"],
            "error": "",
        })

    pd.DataFrame(summary_rows).to_csv("mdp_results/mdp_summary.csv", index=False)
    with open("mdp_results/mdp_results.json", "w") as handle:
        json.dump(all_results, handle, indent=2)
    print(f"\\n💾 Saved mdp_results/mdp_summary.csv, mdp_results/mdp_results.json "
          f"and model charges ({len(all_results)} of {len(structure_files)} structures)")'''
