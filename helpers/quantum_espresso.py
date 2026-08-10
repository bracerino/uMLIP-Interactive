"""Quantum ESPRESSO (pw.x) support.

This module is the single place that knows how to turn the GUI's Quantum
ESPRESSO settings into

* a live ``ase.calculators.espresso.Espresso`` object (used by the interface), and
* a block of Python source code that builds the very same calculator (used by
  the standalone scripts the app generates).

Both paths go through :func:`build_qe_input_data` / :func:`build_qe_command`
so the interface and the generated scripts can never drift apart.

Because ``run_mace_calculation`` and the many script generators are spread over
several modules, the active settings are also cached module-side
(:func:`set_active_qe_settings` / :func:`get_active_qe_settings`) exactly the
way the sidebar caches the selected model.
"""

import hashlib
import json
import math
import os

# Name of the pseudo-model family/entry used in MODEL_FAMILIES.
QE_FAMILY_NAME = "Quantum ESPRESSO (DFT) (in testing)"
QE_MODEL_KEY = "Quantum ESPRESSO — pw.x (ab initio DFT) 🧪"
QE_MODEL_VALUE = "qe:pw"


QE_DEFAULTS = {
    # --- executable / environment -----------------------------------------
    'pw_binary': 'pw.x',
    'pseudo_dir': '',
    'use_gpu': False,
    'use_mpi': True,
    'mpi_command': 'mpirun',
    'mpi_cores': 4,
    'omp_threads': 1,
    'npool': 1,
    'ndiag': 0,          # 0 -> flag omitted (let QE decide)
    'extra_pw_flags': '',
    'work_dir': 'qe_calc',
    # --- plane-wave basis --------------------------------------------------
    'ecutwfc': 60.0,
    'ecutrho': 480.0,
    # --- k-points ----------------------------------------------------------
    'kpoint_mode': 'kspacing',   # 'kspacing' | 'grid' | 'gamma'
    'kspacing': 0.25,            # 1/Angstrom
    'kgrid': [4, 4, 4],
    'koffset': [0, 0, 0],
    # --- occupations -------------------------------------------------------
    'occupations': 'smearing',   # 'smearing' | 'fixed' | 'tetrahedra_opt'
    'smearing': 'mv',
    'degauss': 0.01,             # Ry
    # --- SCF ---------------------------------------------------------------
    'conv_thr': 1e-6,            # Ry
    'mixing_beta': 0.4,
    'mixing_mode': 'plain',
    'electron_maxstep': 200,
    'diagonalization': 'david',
    'scf_must_converge': True,   # False -> pw.x carries on after NELM
    'reuse_density': True,       # start each step from the last SCF density
    'reuse_wavefunctions': False,
    'compute_stress': 'auto',    # 'auto' | 'always' | 'never' (tstress)
    # --- system ------------------------------------------------------------
    'nspin': 1,
    'starting_magnetization': 0.0,
    # --- spin / +U overrides (MAGMOM, LDAU) --------------------------------
    'per_structure_overrides': False,
    'overrides_source': 'inline',   # 'inline' (typed/imported) | 'file'
    'overrides_text': '',           # the INCAR-syntax block, when inline
    'overrides_dir': '',         # '' -> the folder the run is started from
    'overrides_required': False,
    'magmom_units': 'bohr',      # 'bohr' (VASP MAGMOM) | 'fraction' (QE)
    'carry_magmoms': False,      # start from the moments the last run reached
    'hubbard_style': 'card',     # 'card' (QE >= 7.1) | 'namelist' (QE <= 7.0)
    'hubbard_projectors': 'ortho-atomic',
    'tot_charge': 0.0,
    'nbnd': 0,                   # 0 -> let QE decide
    'input_dft': '',             # '' -> take functional from the pseudopotentials
    'vdw_corr': 'none',
    'assume_isolated': 'none',
    # --- pseudopotentials --------------------------------------------------
    'pseudo_overrides': {},      # {'Fe': 'Fe.pbe-spn-kjpaw_psl.1.0.0.UPF'}
    # --- escape hatch ------------------------------------------------------
    'extra_input_data': '',      # raw "section.key = value" lines, one per line
}


# pw.x takes every energy in Rydberg, but the sidebar shows eV (as VASP does).
# The settings dict, the saved presets and the generated scripts all stay in Ry
# — the conversion happens only at the widget boundary.
RY_TO_EV = 13.605693122994   # CODATA 2018, same value as ase.units.Rydberg


def ry_to_ev(value):
    """Rydberg -> eV, for display."""
    return float(value) * RY_TO_EV


def ev_to_ry(value):
    """eV -> Rydberg, for the pw.x namelist.

    Rounded to 10 significant digits so a round trip gives back a clean
    ``60`` instead of ``59.99999999999999``, while conv_thr values as small
    as 1e-12 Ry survive intact (a fixed number of decimals would flush them
    to zero).
    """
    return float(f"{float(value) / RY_TO_EV:.10g}")


def _clamp(value, low, high):
    """Keep a converted value inside a widget's range.

    A preset saved before the eV switch (or written by hand) can hold a cutoff
    outside the range offered here; Streamlit raises if `value` is out of
    bounds, so it is pulled back in instead.
    """
    return max(low, min(high, float(value)))


SMEARING_CHOICES = ['mv', 'gaussian', 'mp', 'fd']
DIAGONALIZATION_CHOICES = ['david', 'cg', 'ppcg', 'paro', 'rmm-davidson']
MIXING_MODE_CHOICES = ['plain', 'TF', 'local-TF']
VDW_CORR_CHOICES = ['none', 'grimme-d2', 'grimme-d3', 'ts-vdw', 'xdm', 'mbd']
ASSUME_ISOLATED_CHOICES = ['none', 'makov-payne', 'martyna-tuckerman', 'esm']
HUBBARD_PROJECTOR_CHOICES = ['ortho-atomic', 'atomic', 'norm-atomic',
                             'wf', 'pseudo']


# ---------------------------------------------------------------------------
# Active-settings cache
# ---------------------------------------------------------------------------
_ACTIVE_QE_SETTINGS = None


def set_active_qe_settings(settings):
    """Remember the settings picked in the sidebar so that the calculation
    thread and every script generator can reach them without extra plumbing."""
    global _ACTIVE_QE_SETTINGS
    _ACTIVE_QE_SETTINGS = dict(settings) if settings else None


def get_active_qe_settings():
    """Return the settings from the sidebar, falling back to the defaults."""
    if _ACTIVE_QE_SETTINGS:
        merged = dict(QE_DEFAULTS)
        merged.update(_ACTIVE_QE_SETTINGS)
        return merged
    return dict(QE_DEFAULTS)


def is_qe_model(selected_model_key=None, model_size=None):
    """True when the user picked Quantum ESPRESSO instead of an MLIP."""
    if isinstance(model_size, str) and model_size.startswith("qe:"):
        return True
    if isinstance(selected_model_key, str) and "Quantum ESPRESSO" in selected_model_key:
        return True
    return False


# ---------------------------------------------------------------------------
# Pseudopotential discovery
# ---------------------------------------------------------------------------
def find_pseudopotentials(pseudo_dir):
    """Scan ``pseudo_dir`` and map every chemical symbol to the candidate UPF
    files found for it.

    Handles the naming conventions of the common libraries, e.g.
    ``Fe.pbe-spn-kjpaw_psl.1.0.0.UPF`` (PSLibrary),
    ``Fe_ONCV_PBE-1.0.upf`` (SG15), ``fe_pbe_v1.5.uspp.F.UPF`` (GBRV).

    Returns ``{'Fe': ['Fe.pbe-...UPF', ...], ...}`` with the candidates sorted
    so that the shortest (usually the plainest) name comes first.
    """
    from ase.data import chemical_symbols

    symbol_lookup = {s.lower(): s for s in chemical_symbols if s}
    found = {}

    if not pseudo_dir or not os.path.isdir(pseudo_dir):
        return found

    for fname in sorted(os.listdir(pseudo_dir)):
        if not fname.lower().endswith('.upf'):
            continue
        # Element token = leading run of letters before the first separator.
        token = ''
        for ch in fname:
            if ch.isalpha():
                token += ch
            else:
                break
        symbol = symbol_lookup.get(token.lower())
        if symbol is None and len(token) > 2:
            # Names such as "Feb..." never occur, but a 3+ letter run means the
            # separator was missing; retry with the first two/one characters.
            symbol = symbol_lookup.get(token[:2].lower()) or symbol_lookup.get(token[:1].lower())
        if symbol is None:
            continue
        found.setdefault(symbol, []).append(fname)

    for symbol in found:
        found[symbol].sort(key=lambda n: (len(n), n))
    return found


def suggest_cutoffs(pseudo_dir, symbols=None):
    """Read the recommended cutoffs from an SSSP ``*.json`` manifest, if one sits
    in ``pseudo_dir``.

    SSSP ships a per-element ``cutoff_wfc``/``cutoff_rho``; the correct choice
    for a structure is the maximum over its elements. Returns
    ``(ecutwfc, ecutrho, source_filename)`` or ``None``.
    """
    import glob
    import json

    if not pseudo_dir or not os.path.isdir(pseudo_dir):
        return None

    for path in sorted(glob.glob(os.path.join(pseudo_dir, '*.json'))):
        try:
            with open(path) as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue

        entries = {
            sym: info for sym, info in data.items()
            if isinstance(info, dict) and 'cutoff_wfc' in info
        }
        if not entries:
            continue

        wanted = [s for s in (symbols or entries) if s in entries]
        if not wanted:
            continue

        wfc = max(float(entries[s]['cutoff_wfc']) for s in wanted)
        rho = max(float(entries[s].get('cutoff_rho', 8 * entries[s]['cutoff_wfc']))
                  for s in wanted)
        return wfc, rho, os.path.basename(path)

    return None


def resolve_pseudopotentials(pseudo_dir, overrides=None):
    """Return the ``{symbol: filename}`` dict handed to the ASE calculator.

    Every element present in ``pseudo_dir`` is included — ASE only looks up the
    species that actually occur in a structure, so one dict serves all
    structures in a run.
    """
    overrides = overrides or {}
    mapping = {sym: files[0] for sym, files in find_pseudopotentials(pseudo_dir).items()}
    for sym, fname in overrides.items():
        if fname:
            mapping[sym] = fname
    return mapping


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------
def normalize_pw_binary(path):
    """Accept either the pw.x executable itself or the directory holding it.

    Pointing at ``<qe>/bin`` is the natural thing to type, so treat a directory
    that contains pw.x as if the user had typed the full path.
    """
    if not path:
        return path
    expanded = os.path.expanduser(path)
    if os.path.isdir(expanded):
        candidate = os.path.join(expanded, 'pw.x')
        if os.path.isfile(candidate):
            return candidate
    return expanded


def _merged(settings):
    merged = dict(QE_DEFAULTS)
    merged.update(settings or {})
    merged['pw_binary'] = normalize_pw_binary(merged.get('pw_binary'))
    if merged.get('pseudo_dir'):
        merged['pseudo_dir'] = os.path.expanduser(merged['pseudo_dir'])
    return merged


def parse_extra_input_data(text):
    """Parse the free-form ``section.key = value`` escape hatch into a nested
    dict. Unknown/blank lines are ignored, values are coerced to bool/int/float
    where possible."""
    extra = {}
    for raw in (text or '').splitlines():
        line = raw.split('!')[0].split('#')[0].strip()
        if not line or '=' not in line:
            continue
        lhs, rhs = line.split('=', 1)
        lhs, rhs = lhs.strip(), rhs.strip().strip(',')
        if '.' in lhs:
            section, key = lhs.split('.', 1)
        else:
            section, key = 'system', lhs
        section, key = section.strip().lower(), key.strip()

        value = rhs.strip("'\"")
        low = value.lower()
        if low in ('.true.', 'true'):
            value = True
        elif low in ('.false.', 'false'):
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value.replace('d', 'e').replace('D', 'e'))
                except ValueError:
                    pass
        extra.setdefault(section, {})[key] = value
    return extra


def build_qe_input_data(settings, calculation='scf'):
    """Build the nested ``input_data`` dict for ``ase.io.espresso``."""
    s = _merged(settings)

    control = {
        'calculation': calculation,
        'restart_mode': 'from_scratch',
        'tprnfor': True,      # forces are mandatory for relaxation/MD/phonons
        # The stress costs as much as several SCF iterations on a PAW/DFT+U
        # cell and is pure waste for a phonon run or a fixed-cell relaxation,
        # so by default it is switched on only once something asks for it
        # (see qe_prepare_structure).
        'tstress': str(s.get('compute_stress', 'auto')) == 'always',
        'disk_io': 'low',
        'outdir': 'qe_tmp',
        'prefix': 'pwscf',
    }

    system = {
        'ecutwfc': float(s['ecutwfc']),
        'ecutrho': float(s['ecutrho']),
        'occupations': s['occupations'],
    }
    if s['occupations'] == 'smearing':
        system['smearing'] = s['smearing']
        system['degauss'] = float(s['degauss'])
    if int(s['nspin']) == 2:
        # Only the switch goes here. starting_magnetization is per species and
        # its index has to match the ATOMIC_SPECIES order, which ASE builds from
        # the atoms' magnetic moments — so the moments are set on the structure
        # instead (see qe_prepare_structure), and ASE writes the indexed keys.
        system['nspin'] = 2
    if float(s['tot_charge']):
        system['tot_charge'] = float(s['tot_charge'])
    if int(s['nbnd'] or 0) > 0:
        system['nbnd'] = int(s['nbnd'])
    if s['input_dft']:
        system['input_dft'] = s['input_dft']
    if s['vdw_corr'] and s['vdw_corr'] != 'none':
        system['vdw_corr'] = s['vdw_corr']
    if s['assume_isolated'] and s['assume_isolated'] != 'none':
        system['assume_isolated'] = s['assume_isolated']

    electrons = {
        'conv_thr': float(s['conv_thr']),
        'mixing_beta': float(s['mixing_beta']),
        'mixing_mode': s['mixing_mode'],
        'electron_maxstep': int(s['electron_maxstep']),
        'diagonalization': s['diagonalization'],
    }
    # Default pw.x behaviour is to stop with exit status 2 as soon as one SCF
    # cycle runs out of iterations, which kills an ASE-driven relaxation at that
    # ionic step. scf_must_converge = .false. makes pw.x carry on with the last
    # density instead.
    if not bool(s.get('scf_must_converge', True)):
        electrons['scf_must_converge'] = False

    input_data = {'control': control, 'system': system, 'electrons': electrons}

    for section, values in parse_extra_input_data(s['extra_input_data']).items():
        input_data.setdefault(section, {}).update(values)

    return input_data


def build_qe_kpoint_kwargs(settings):
    """Return the ``kpts``/``kspacing``/``koffset`` kwargs for ``Espresso``."""
    s = _merged(settings)
    mode = s['kpoint_mode']
    if mode == 'gamma':
        return {'kpts': None}
    if mode == 'grid':
        return {
            'kpts': tuple(int(k) for k in s['kgrid']),
            'koffset': tuple(int(k) for k in s['koffset']),
        }
    return {'kspacing': float(s['kspacing'])}


def build_qe_command(settings):
    """Assemble the shell command that launches pw.x.

    ASE appends ``-in <inputfile>``, so parallelisation flags belong here.
    On GPU builds each MPI rank drives one GPU, which is why the sidebar
    labels the rank count "GPUs to use" in that mode.
    """
    s = _merged(settings)
    parts = []

    # Always go through the launcher when MPI is enabled, even for a single
    # rank: an MPI-linked pw.x started bare falls back to OpenMPI's singleton
    # init, which needs orted at the path baked in at build time and typically
    # aborts in MPI_Init_thread.
    if s['use_mpi']:
        parts += [s['mpi_command'], '-np', str(max(1, int(s['mpi_cores'])))]
    parts.append(s['pw_binary'])

    if int(s['npool'] or 1) > 1:
        parts += ['-nk', str(int(s['npool']))]
    if int(s['ndiag'] or 0) > 0:
        parts += ['-nd', str(int(s['ndiag']))]
    if s['extra_pw_flags']:
        parts += s['extra_pw_flags'].split()

    return ' '.join(parts)


def apply_qe_environment(settings):
    """Set the threading environment pw.x inherits.

    ``BaseProfile.run`` passes ``os.environ`` to the subprocess, so exporting
    here is enough. GPU builds want one rank per GPU with the remaining cores
    used as OpenMP threads.
    """
    s = _merged(settings)
    threads = str(max(1, int(s['omp_threads'])))
    os.environ['OMP_NUM_THREADS'] = threads
    os.environ['MKL_NUM_THREADS'] = threads
    os.environ['OPENBLAS_NUM_THREADS'] = threads


def validate_qe_settings(settings):
    """Return a list of human-readable problems; empty means good to go."""
    s = _merged(settings)
    problems = []

    binary = s['pw_binary']
    if not binary:
        problems.append("No pw.x binary given.")
    elif os.sep in binary or binary.startswith('.'):
        if os.path.isdir(binary):
            problems.append(
                f"'{binary}' is a directory and contains no pw.x — "
                "point at the pw.x executable or the bin/ folder that holds it."
            )
        elif not os.path.isfile(binary):
            problems.append(f"pw.x binary not found: {binary}")
        elif not os.access(binary, os.X_OK):
            problems.append(f"pw.x binary is not executable: {binary}")
    else:
        import shutil
        if shutil.which(binary) is None:
            problems.append(f"'{binary}' is not on PATH — give the full path to pw.x.")

    if not s['pseudo_dir']:
        problems.append("No pseudopotential directory given.")
    elif not os.path.isdir(s['pseudo_dir']):
        problems.append(f"Pseudopotential directory not found: {s['pseudo_dir']}")
    elif not find_pseudopotentials(s['pseudo_dir']):
        problems.append(f"No .UPF files found in {s['pseudo_dir']}")

    if s['use_mpi'] and s['mpi_command']:
        import shutil
        if os.sep not in s['mpi_command'] and shutil.which(s['mpi_command']) is None:
            problems.append(f"MPI launcher '{s['mpi_command']}' is not on PATH.")

    if float(s['ecutrho']) < float(s['ecutwfc']):
        problems.append("ecutrho must be >= ecutwfc (4x for NC, 8-12x for US/PAW).")

    return problems


def missing_pseudopotentials(settings, symbols):
    """Which of ``symbols`` have no pseudopotential in the chosen directory."""
    s = _merged(settings)
    mapping = resolve_pseudopotentials(s['pseudo_dir'], s['pseudo_overrides'])
    return sorted({sym for sym in symbols if sym not in mapping})


# ---------------------------------------------------------------------------
# Per-structure spin / Hubbard-U overrides
#
# Everything a structure needs beyond the sidebar (which moment sits on which
# atom, which shell carries a U) is per structure, not per run, so it is read
# from a small VASP-style file that lives next to the structures. Like the
# diagnostics below this is kept as source, so the interface and the generated
# standalone scripts run exactly the same code.
# ---------------------------------------------------------------------------
QE_OVERRIDES_SRC = '''
import copy as _qecopy
import os as _qeos
import re as _qere

# Filled in by qe_configure_overrides() from the sidebar settings.
QE_OVERRIDES_CONFIG = {
    "enabled": False,
    "source": "inline",              # "inline" (text below) | "file" (per structure)
    "text": "",                      # INCAR-syntax block used when inline
    "dir": "",                       # "" -> the folder the run starts in
    "magmom_units": "bohr",          # "bohr" (VASP MAGMOM) | "fraction" (QE)
    "hubbard_style": "card",         # "card" (QE >= 7.1) | "namelist" (<= 7.0)
    "hubbard_projectors": "ortho-atomic",
    "require_file": False,
    "compute_stress": "auto",        # "auto" | "always" | "never" (tstress)
    "structure_name": None,          # set per structure by the caller
    "nspin": 1,
    "starting_magnetization": 0.0,
    "reuse_density": True,           # start each step from the last SCF density
    "reuse_wavefunctions": False,
    "carry_magmoms": False,          # reuse the moments the last run converged to
}

QE_L_LETTERS = {0: "s", 1: "p", 2: "d", 3: "f"}

# (file, mtime) pairs already reported in full, so a long relaxation does not
# repeat the whole translation at every ionic step.
QE_REPORTED_FILES = set()

# VASP keywords that have no pw.x counterpart; recognised so the user gets a
# reason instead of silence.
QE_IGNORED_KEYS = {
    "LORBIT": "pw.x always writes the atomic moments to its output",
    "LDAUPRINT": "use control.verbosity = 'high' for the same detail",
    "LMAXMIX": "pw.x does not mix the density l-channel by l-channel",
    "ICHARG": "no equivalent (see electrons.startingpot / startingwfc)",
    "PREC": "pw.x accuracy is set by the cutoffs alone",
    "LREAL": "pw.x evaluates the projectors in reciprocal space anyway",
    "ADDGRID": "pw.x has no additional support grid; raise ecutrho instead",
    "LASPH": "pw.x PAW is aspherical already",
    "LWAVE": "pw.x keeps its wavefunctions in the outdir (see control.disk_io)",
    "LCHARG": "pw.x writes the density to the outdir (see control.disk_io)",
    "SYSTEM": "a title only",
    "NPAR": "use the -nk / -nd / -nt parallelisation flags in the sidebar",
    "NCORE": "use the -nk / -nd / -nt parallelisation flags in the sidebar",
}
# The same, for keywords the sidebar (or the INCAR import) already covers - the
# per-structure file is only asked about spin and Hubbard U.
QE_SIDEBAR_KEYS = (
    "ENCUT", "ENAUG", "EDIFF", "NELM", "NELMIN", "ALGO", "ISMEAR", "SIGMA",
    "IBRION", "ISIF", "NSW", "EDIFFG", "POTIM", "ISYM", "SYMPREC", "KSPACING",
    "KPAR", "NBANDS", "IVDW", "GGA", "METAGGA", "AMIX", "BMIX", "IMIX",
)


def qe_configure_overrides(**kwargs):
    """Point the runtime at the sidebar settings (called once per run)."""
    QE_OVERRIDES_CONFIG.update(kwargs)


def qe_set_structure_name(name):
    """Name of the structure about to be calculated, used to find its file."""
    QE_OVERRIDES_CONFIG["structure_name"] = name or None


def qe_emit(message):
    print(message, flush=True)


# --- file discovery and parsing -------------------------------------------
def qe_override_dir():
    directory = QE_OVERRIDES_CONFIG.get("dir") or "."
    return _qeos.path.abspath(_qeos.path.expanduser(directory))


def qe_override_candidates(structure_name=None, formula=None):
    """Filenames searched for, most specific first."""
    names = []
    for base in (structure_name, formula):
        if not base:
            continue
        stem = _qeos.path.splitext(str(base))[0]
        for candidate in (str(base) + ".incar", stem + ".incar",
                          "INCAR_" + stem, stem + ".qeset"):
            if candidate not in names:
                names.append(candidate)
    names.append("INCAR")
    return names


def qe_find_override_file(structure_name=None, formula=None, directory=None):
    directory = directory or qe_override_dir()
    for name in qe_override_candidates(structure_name, formula):
        path = _qeos.path.join(directory, name)
        if _qeos.path.isfile(path):
            return path
    return None


def qe_coerce(text):
    """'.TRUE.' -> True, '2' -> 2, '4.2d0' -> 4.2, anything else -> str."""
    value = str(text).strip().strip("'\\"")
    low = value.lower()
    if low in (".true.", "true", ".t.", "t"):
        return True
    if low in (".false.", "false", ".f.", "f"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value.replace("d", "e").replace("D", "e"))
    except ValueError:
        return value


def qe_is_true(text):
    return qe_coerce(text) is True


def qe_expand_vasp_values(text):
    """Expand VASP's repeat syntax: '48*0.0 16*1.0' -> [0.0]*48 + [1.0]*16."""
    values = []
    for token in str(text or "").replace(",", " ").split():
        if "*" in token:
            count, _, value = token.partition("*")
            values.extend([float(value)] * int(float(count)))
        else:
            values.append(float(token))
    return values


def qe_magmom_values(text, symbols):
    """The per-atom moments a MAGMOM line describes, as (values, per_element).

    Two forms are accepted:

      MAGMOM = 48*0.0 16*1.0 32*0.0    one value per atom, as VASP writes it
      MAGMOM = K:0.0 Ni:1.0 O:0.0      one value per element

    The per-element form does not depend on how many atoms there are, so the
    same block serves a unit cell, a phonon supercell and any other cell built
    from it - which a per-atom list cannot do.
    """
    raw = str(text or "")
    if ":" not in raw:
        return qe_expand_vasp_values(raw), False

    table = {}
    for token in raw.replace(",", " ").split():
        if ":" not in token:
            raise RuntimeError(
                "MAGMOM mixes the per-element form with plain numbers: %r" % raw)
        element, _, value = token.partition(":")
        try:
            table[element.strip()] = float(value)
        except ValueError:
            raise RuntimeError("MAGMOM entry %r is not a number." % token)

    missing = []
    for symbol in symbols:
        if symbol not in table and symbol not in missing:
            missing.append(symbol)
    if missing:
        raise RuntimeError(
            "MAGMOM (per element) has no entry for %s - the structure contains "
            "%s." % (", ".join(missing), ", ".join(sorted(set(symbols))))
        )
    return [table[symbol] for symbol in symbols], True


def qe_parse_override_text(text):
    """Parse VASP-style (INCAR syntax) settings.

    Returns (keys, passthrough, species_order):
      keys           {'ISPIN': '2', 'MAGMOM': '48*0.0 ...'} - upper-cased
      passthrough    {'system': {'nosym': True}} - raw 'section.key = value'
      species_order  ['K', 'Ni', 'O'] from 'SPECIES =' or '# Species order:'
    """
    keys = {}
    passthrough = {}
    species_order = None

    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line[0] in "#!":
            match = _qere.search(r"species\\s*order\\s*[:=]\\s*(.+)", line,
                                 _qere.IGNORECASE)
            if match:
                species_order = match.group(1).replace(",", " ").split()
            continue
        line = line.split("#")[0].split("!")[0].strip()
        if not line or "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        lhs, rhs = lhs.strip(), rhs.strip().rstrip(";").strip()
        if "." in lhs:      # raw pw.x entry, e.g. "system.nosym = .true."
            section, key = lhs.split(".", 1)
            passthrough.setdefault(section.strip().lower(),
                                   {})[key.strip()] = qe_coerce(rhs)
        else:
            keys[lhs.upper()] = rhs

    if "SPECIES" in keys:
        species_order = keys["SPECIES"].replace(",", " ").split()
    return keys, passthrough, species_order


def qe_parse_override_file(path):
    """qe_parse_override_text() applied to a file."""
    with open(path, errors="replace") as handle:
        return qe_parse_override_text(handle.read())


# --- pseudopotential lookups ----------------------------------------------
# A UPF is read whole (the wavefunction block sits behind megabytes of
# projector data) but only once: the two things wanted from it are cached per
# file, since this runs at every ionic step.
QE_UPF_CACHE = {}


def qe_upf_info(path):
    """(z_valence, channels) of a UPF file, parsed once per file."""
    try:
        key = (str(path), _qeos.path.getmtime(path))
    except OSError:
        return None, []
    if key not in QE_UPF_CACHE:
        try:
            with open(path, errors="replace") as handle:
                text = handle.read()
        except OSError:
            text = ""
        QE_UPF_CACHE[key] = (qe_parse_upf_valence(text),
                             qe_parse_upf_channels(text))
    return QE_UPF_CACHE[key]


def qe_parse_upf_valence(text):
    """Number of valence electrons the pseudopotential describes, or None."""
    match = _qere.search(r'z_valence\\s*=\\s*"?\\s*([-+0-9.eEdD]+)', text or "")
    if match is None:
        # UPF v1 writes it as a plain "   8.00000   Z valence" line.
        match = _qere.search(r'([-+0-9.eEdD]+)\\s+Z\\s+valence', text or "",
                             _qere.IGNORECASE)
    if match is None:
        return None
    try:
        value = float(match.group(1).replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    return value if value > 0 else None


def qe_upf_valence(path):
    return qe_upf_info(path)[0]


def qe_upf_channels(path):
    return qe_upf_info(path)[1]


def qe_parse_upf_channels(text):
    """The atomic wavefunctions a UPF carries, as [(label, l, occupation), ...].

    The HUBBARD card names a manifold (Ni-3d), and the principal quantum number
    has to be the one the pseudopotential actually has, so it is read from the
    file rather than guessed.
    """
    if not text:
        return []

    channels = []
    # UPF v2: <PP_CHI.1 ... label="3D" l="2" occupation="6.0" ...>
    for chunk in _qere.findall(r"<PP_CHI[^>]*>", text):
        label = _qere.search(r'label\\s*=\\s*"\\s*([1-7][spdfSPDF])\\s*"', chunk)
        lval = _qere.search(r'\\bl\\s*=\\s*"\\s*(\\d)\\s*"', chunk)
        if not label or not lval:
            continue
        occ = _qere.search(r'occupation\\s*=\\s*"\\s*([-+0-9.eEdD]+)', chunk)
        occupation = float(occ.group(1).replace("D", "E")) if occ else 0.0
        channels.append((label.group(1).lower(), int(lval.group(1)), occupation))
    if channels:
        return channels

    # UPF v1: the PP_PSWFC block lists "3D  2  6.00  Wavefunction".
    block = _qere.search(r"<PP_PSWFC>(.*?)</PP_PSWFC>", text, _qere.DOTALL)
    if block:
        for line in block.group(1).splitlines():
            fields = line.split()
            if len(fields) >= 3 and _qere.match(r"^[1-7][spdfSPDF]$", fields[0]):
                try:
                    channels.append((fields[0].lower(), int(fields[1]),
                                     float(fields[2])))
                except ValueError:
                    continue
    return channels


def qe_upf_manifold(path, l):
    """Label of the wavefunction with angular momentum ``l``, e.g. '3d'."""
    matching = [c for c in qe_upf_channels(path) if c[1] == int(l)]
    if not matching:
        return None
    return max(matching, key=lambda c: c[2])[0]


def qe_default_manifold(symbol, l):
    """Fallback manifold from the periodic table (3d for Fe, 4f for Ce, ...)."""
    from ase.data import atomic_numbers

    letter = QE_L_LETTERS.get(int(l))
    if letter is None:
        return None
    number = atomic_numbers.get(symbol, 0)
    period = 1
    for limit in (2, 10, 18, 36, 54, 86):
        if number > limit:
            period += 1
    if l == 2:
        period -= 1          # 3d in period 4, 4d in period 5, ...
    elif l == 3:
        period -= 2          # 4f for the lanthanides, 5f for the actinides
    return "%d%s" % (max(1, period), letter)


def qe_manifold(symbol, l, pseudo_map, pseudo_dir):
    fname = (pseudo_map or {}).get(symbol)
    if fname and pseudo_dir:
        channels = qe_upf_channels(_qeos.path.join(str(pseudo_dir), fname))
        matching = [c for c in channels if c[1] == int(l)]
        if matching:
            return max(matching, key=lambda c: c[2])[0]
        if channels:
            # The file lists its wavefunctions and has none of this kind, so a
            # guess here would only produce a label pw.x cannot resolve.
            return None
    return qe_default_manifold(symbol, l)


# --- species bookkeeping ---------------------------------------------------
def qe_species_labels(symbols, magmoms=None):
    """The ATOMIC_SPECIES labels ase.io.espresso will write, in its order.

    ASE gives every distinct (element, magnetic moment) pair its own species:
    the first Ni stays "Ni", the next one becomes "Ni1", and so on. The HUBBARD
    card and the indexed Hubbard_U() keys have to use exactly those names.
    """
    labels = []
    seen = []
    for index, symbol in enumerate(symbols):
        moment = 0.0 if magmoms is None else float(magmoms[index])
        key = (symbol, moment)
        if key in seen:
            continue
        tidx = sum(1 for sym, _m in seen if sym == symbol)
        seen.append(key)
        labels.append((symbol if tidx == 0 else "%s%d" % (symbol, tidx),
                       symbol, moment))
    return labels


def qe_magmoms_to_fractions(symbols, magmoms, pseudo_map, pseudo_dir, units,
                            messages):
    """VASP MAGMOM (Bohr magnetons) -> QE starting_magnetization (fraction).

    starting_magnetization is (n_up - n_down) / n_valence for the species, so
    the moment is divided by the valence charge taken from the UPF file.
    """
    if str(units).startswith("frac"):
        return [max(-1.0, min(1.0, round(float(m), 6))) for m in magmoms]

    valence_cache = {}
    fractions = []
    for symbol, moment in zip(symbols, magmoms):
        if symbol not in valence_cache:
            fname = (pseudo_map or {}).get(symbol)
            valence = None
            if fname and pseudo_dir:
                valence = qe_upf_valence(_qeos.path.join(str(pseudo_dir), fname))
            if valence is None:
                messages.append(
                    "   \\u26a0\\ufe0f  no z_valence found for %s - its MAGMOM is "
                    "used as a fraction instead of Bohr magnetons" % symbol
                )
            valence_cache[symbol] = valence
        valence = valence_cache[symbol]
        fraction = float(moment) / valence if valence else float(moment)
        fractions.append(max(-1.0, min(1.0, round(fraction, 6))))
    return fractions


# --- Hubbard U -------------------------------------------------------------
def qe_hubbard_from_keys(keys, species_order, labels, pseudo_map, pseudo_dir,
                         style, projectors, messages):
    """Translate LDAU* into a HUBBARD card (QE >= 7.1) or Hubbard_U() keys.

    LDAUU/LDAUJ are in eV and so are the QE parameters, so no conversion is
    needed. LDAUTYPE = 2 (Dudarev) is QE's simplified scheme with U_eff = U - J;
    LDAUTYPE = 1 (Liechtenstein) keeps U and J apart.
    """
    if not qe_is_true(keys.get("LDAU", "")):
        return [], {}

    try:
        ldaul = [int(round(v)) for v in qe_expand_vasp_values(keys.get("LDAUL", ""))]
    except ValueError:
        raise RuntimeError("LDAUL could not be parsed: %r" % keys.get("LDAUL"))
    ldauu = qe_expand_vasp_values(keys.get("LDAUU", ""))
    ldauj = qe_expand_vasp_values(keys.get("LDAUJ", ""))
    ldau_type = int(qe_coerce(keys.get("LDAUTYPE", 2)) or 2)

    if not ldaul:
        raise RuntimeError("LDAU = .TRUE. but no LDAUL given.")
    if not ldauu:
        ldauu = [0.0] * len(ldaul)
    if not ldauj:
        ldauj = [0.0] * len(ldaul)
    if not (len(ldauu) == len(ldauj) == len(ldaul)):
        raise RuntimeError(
            "LDAUL/LDAUU/LDAUJ have different lengths (%d/%d/%d)."
            % (len(ldaul), len(ldauu), len(ldauj))
        )

    # Which entry belongs to which element. VASP takes the POTCAR order; here it
    # is either stated in the file or taken from the order the elements first
    # appear in the structure.
    structure_order = []
    for _label, symbol, _moment in labels:
        if symbol not in structure_order:
            structure_order.append(symbol)
    if species_order:
        order = list(species_order)
        unknown = [s for s in order if s not in structure_order]
        if unknown:
            raise RuntimeError(
                "Species order %s does not match the structure (%s): %s not present."
                % (order, structure_order, ", ".join(unknown))
            )
    else:
        order = structure_order
        messages.append(
            "   \\u2139\\ufe0f  no species order given - using the order they appear "
            "in the structure: %s" % " ".join(order)
        )
    if len(order) != len(ldaul):
        raise RuntimeError(
            "LDAUL has %d entries but the species order has %d (%s). Add a "
            "'# Species order: ...' line (or SPECIES = ...) to the file."
            % (len(ldaul), len(order), " ".join(order))
        )

    if ldau_type not in (1, 2):
        messages.append(
            "   \\u26a0\\ufe0f  LDAUTYPE = %d is not supported; treated as 2 "
            "(Dudarev)." % ldau_type
        )
        ldau_type = 2

    per_symbol = {}
    for symbol, l, u_value, j_value in zip(order, ldaul, ldauu, ldauj):
        if l < 0 or (abs(u_value) < 1e-12 and abs(j_value) < 1e-12):
            continue
        manifold = qe_manifold(symbol, l, pseudo_map, pseudo_dir)
        if manifold is None:
            raise RuntimeError(
                "%s has no %s channel in %s, so LDAUL = %d cannot be turned into "
                "a Hubbard manifold. Check the species order of LDAUL/LDAUU, or "
                "pick a pseudopotential that carries that shell."
                % (symbol, QE_L_LETTERS.get(int(l), "?"),
                   (pseudo_map or {}).get(symbol, "its pseudopotential"), l)
            )
        per_symbol[symbol] = (manifold, float(u_value), float(j_value))

    if not per_symbol:
        messages.append("   \\u2139\\ufe0f  LDAU = .TRUE. but every U is zero - "
                        "no Hubbard term applied.")
        return [], {}

    if str(style).startswith("name"):
        # QE <= 7.0: indexed keys in &SYSTEM, index = ATOMIC_SPECIES position.
        system = {"lda_plus_u": True, "lda_plus_u_kind": 0}
        if ldau_type == 1:
            messages.append(
                "   \\u26a0\\ufe0f  LDAUTYPE = 1 needs the HUBBARD card; the "
                "namelist form applies U_eff = U - J instead."
            )
        for index, (label, symbol, _moment) in enumerate(labels, start=1):
            if symbol not in per_symbol:
                continue
            _manifold, u_value, j_value = per_symbol[symbol]
            system["Hubbard_U(%d)" % index] = round(u_value - j_value, 8)
            messages.append("   \\U0001f9f2 Hubbard_U(%d) = %.4f eV  (%s)"
                            % (index, u_value - j_value, label))
        return [], system

    # QE >= 7.1: the HUBBARD card, named per species label.
    card = ["HUBBARD (%s)" % projectors]
    for label, symbol, _moment in labels:
        if symbol not in per_symbol:
            continue
        manifold, u_value, j_value = per_symbol[symbol]
        if ldau_type == 1:
            card.append("U %s-%s %.6g" % (label, manifold, u_value))
            if abs(j_value) > 1e-12:
                card.append("J %s-%s %.6g" % (label, manifold, j_value))
        else:
            card.append("U %s-%s %.6g" % (label, manifold, u_value - j_value))
    for line in card[1:]:
        messages.append("   \\U0001f9f2 %s eV" % line)
    return card, {}


# --- carrying the converged moments to the next structure ------------------
def qe_magmom_groups(symbols, raw_magmoms):
    """(element, requested moment) for every atom - the groups MAGMOM defines.

    The same key describes the same sublattice in a unit cell and in a supercell
    built from it, which is what makes the converged moments transferable.
    """
    return [(symbol, round(float(moment), 6))
            for symbol, moment in zip(symbols, raw_magmoms)]


def qe_record_converged_magmoms(calc):
    """Remember what the moments converged to, per MAGMOM group."""
    groups = getattr(calc, "_qe_magmom_groups", None)
    magmoms = (getattr(calc, "results", None) or {}).get("magmoms")
    if not groups or magmoms is None or len(magmoms) != len(groups):
        return
    totals = {}
    for key, moment in zip(groups, magmoms):
        total, count = totals.get(key, (0.0, 0))
        totals[key] = (total + float(moment), count + 1)
    calc._qe_converged_magmoms = {key: total / count
                                  for key, (total, count) in totals.items()}


def qe_carry_converged_magmoms(calc, cfg, symbols, raw_magmoms):
    """The moments the last run converged to, or None to use the block's.

    MAGMOM is only ever a starting guess, and a rough one: a Ni asked to start at
    1 muB may settle at 1.7. When the density cannot be reused - which is exactly
    the case for the first phonon supercell, since it has a different cell - that
    converged value is a much better guess than the number in the block, and it
    keeps the supercell in the magnetic state the relaxed cell ended in.
    """
    calc._qe_magmom_groups = qe_magmom_groups(symbols, raw_magmoms)
    if not cfg.get("carry_magmoms"):
        return None

    converged = getattr(calc, "_qe_converged_magmoms", None)
    if not converged:
        return None
    keys = calc._qe_magmom_groups
    if any(key not in converged for key in keys):
        return None      # a structure the previous run says nothing about
    return [converged[key] for key in keys]


# --- reusing the previous step's SCF result --------------------------------
def qe_structure_fingerprint(atoms, nspin):
    """What has to stay the same for a saved charge density to be reusable.

    The density is stored on the G-vectors of one cell and with one species
    list, so both have to match; the atomic positions are exactly what may
    differ - moving the atoms is the whole point.
    """
    cell = tuple(round(float(value), 8)
                 for row in atoms.get_cell() for value in row)
    return (tuple(atoms.get_chemical_symbols()), cell, int(nspin))


def qe_reuse_previous_scf(calc, atoms, input_data, cfg, system):
    """Start the SCF from the density (and wavefunctions) of the last step.

    pw.x is launched once per ionic step, and every launch otherwise begins from
    a superposition of free atoms - throwing away a converged density that is
    already sitting in the outdir. QE's own relax reuses it; this does the same
    across the separate runs by pointing startingpot/startingwfc at the file.
    """
    fingerprint = qe_structure_fingerprint(atoms, system.get("nspin", 1))
    previous = getattr(calc, "_qe_last_fingerprint", None)
    calc._qe_last_fingerprint = fingerprint

    if not cfg.get("reuse_density", True):
        return None
    control = input_data.setdefault("control", {})
    if str(control.get("restart_mode", "")) == "restart":
        return None     # pw.x is restarting a run of its own; leave it alone

    outdir = str(control.get("outdir", "qe_tmp"))
    prefix = str(control.get("prefix", "pwscf"))
    directory = str(getattr(calc, "directory", "."))
    if not _qeos.path.isabs(outdir):
        outdir = _qeos.path.join(directory, outdir)
    save = _qeos.path.join(outdir, prefix + ".save")

    if previous != fingerprint:
        # First run, a different structure, or a changed cell: the saved density
        # does not belong to this system.
        return None
    if not any(_qeos.path.isfile(_qeos.path.join(save, name))
               for name in ("charge-density.dat", "charge-density.hdf5")):
        return None

    electrons = input_data.setdefault("electrons", {})
    electrons["startingpot"] = "file"
    reused = "density"
    if cfg.get("reuse_wavefunctions"):
        electrons["startingwfc"] = "file"
        # 'low' keeps the wavefunctions in memory; they have to be on disk for
        # the next pw.x process to find them.
        if str(control.get("disk_io", "low")) == "low":
            control["disk_io"] = "medium"
        reused = "density and wavefunctions"
    return reused


# --- the entry point the calculator calls ----------------------------------
def qe_prepare_structure(calc, atoms):
    """Apply the sidebar spin settings and the per-structure file to ``atoms``.

    Called by the calculator right before pw.x is launched, so it works the same
    way in the interface and in the generated scripts. It rebuilds input_data
    from a pristine copy every time, so nothing leaks from one structure to the
    next.
    """
    cfg = QE_OVERRIDES_CONFIG
    base = getattr(calc, "_qe_base_parameters", None)
    if base is None:
        return

    calc.parameters["input_data"] = _qecopy.deepcopy(base.get("input_data") or {})
    calc.parameters.pop("additional_cards", None)
    input_data = calc.parameters["input_data"]
    system = input_data.setdefault("system", {})

    symbols = list(atoms.get_chemical_symbols())
    pseudo_map = calc.parameters.get("pseudopotentials") or {}
    pseudo_dir = str(getattr(getattr(calc, "profile", None), "pseudo_dir", "") or "")

    messages = []
    keys, passthrough, species_order = {}, {}, None
    path = None          # the file the settings came from, if any
    stamp = None         # identifies the source, so it is reported once
    source_name = None

    if cfg.get("enabled"):
        inline = str(cfg.get("text") or "")
        # An empty block falls through to the files, so switching the source back
        # and forth never silently drops the settings.
        if str(cfg.get("source", "inline")) == "inline" and inline.strip():
            keys, passthrough, species_order = qe_parse_override_text(inline)
            stamp = ("<inline>", hash(inline))
            source_name = "the settings panel"
            messages.append("   \\U0001f9f2 spin/+U settings from the settings panel")
        else:
            try:
                formula = atoms.get_chemical_formula(mode="metal", empirical=True)
            except TypeError:
                formula = atoms.get_chemical_formula()
            path = qe_find_override_file(cfg.get("structure_name"), formula)
            if path is None:
                searched = ", ".join(
                    qe_override_candidates(cfg.get("structure_name"), formula))
                if cfg.get("require_file"):
                    raise RuntimeError(
                        "No spin/+U file found in %s (looked for %s)."
                        % (qe_override_dir(), searched)
                    )
                messages.append(
                    "   \\u2139\\ufe0f  no spin/+U file in %s (looked for %s) - using "
                    "the plain spin setting" % (qe_override_dir(), searched)
                )
            else:
                keys, passthrough, species_order = qe_parse_override_file(path)
                source_name = _qeos.path.basename(path)
                messages.append("   \\U0001f9f2 spin/+U settings from %s" % path)
                try:
                    stamp = (path, _qeos.path.getmtime(path))
                except OSError:
                    stamp = (path, None)

    # --- magnetic moments --------------------------------------------------
    # The file has the last word: ISPIN = 1 in it turns off a spin-polarised
    # sidebar setting for this structure, exactly as ISPIN = 2 turns it on.
    ispin = qe_coerce(keys["ISPIN"]) if "ISPIN" in keys else None
    if ispin in (1, 2):
        magnetic = ispin == 2
    else:
        magnetic = int(cfg.get("nspin", 1) or 1) == 2

    raw_magmoms = None
    if "MAGMOM" in keys and ispin == 1:
        messages.append("   \\u26a0\\ufe0f  MAGMOM ignored - the file sets ISPIN = 1")
    elif "MAGMOM" in keys:
        raw_magmoms, per_element = qe_magmom_values(keys["MAGMOM"], symbols)
        if not per_element:
            if len(symbols) and len(raw_magmoms) == 3 * len(symbols):
                raise RuntimeError(
                    "MAGMOM has 3 values per atom (non-collinear). pw.x needs "
                    "noncolin = .true. for that, which this interface does not "
                    "set up."
                )
            if len(raw_magmoms) != len(symbols):
                raise RuntimeError(
                    "MAGMOM has %d values but this structure has %d atoms. A "
                    "per-atom list only ever fits one cell - a phonon supercell "
                    "or a different structure will not match. Write it per "
                    "element instead (MAGMOM = %s), which fits any cell built "
                    "from the same elements."
                    % (len(raw_magmoms), len(symbols),
                       " ".join("%s:0.0" % s for s in sorted(set(symbols))))
                )
        magnetic = True

    if magnetic:
        system["nspin"] = 2
        system.pop("starting_magnetization", None)
        if raw_magmoms is None:
            uniform = float(cfg.get("starting_magnetization", 0.0) or 0.0)
            fractions = [uniform] * len(symbols)
            if uniform:
                messages.append(
                    "   \\U0001f9f2 spin polarised, starting_magnetization = %.3f "
                    "on every species" % uniform
                )
            else:
                messages.append("   \\U0001f9f2 spin polarised (nspin = 2), no "
                                "starting moments")
        else:
            # What the block asks for, and then what is actually used - they
            # differ when the moments of the previous run are carried over.
            moments = list(raw_magmoms)
            carried = qe_carry_converged_magmoms(calc, cfg, symbols, raw_magmoms)
            if carried is not None:
                moments = carried
            fractions = qe_magmoms_to_fractions(
                symbols, moments, pseudo_map, pseudo_dir,
                cfg.get("magmom_units", "bohr"), messages)
            summary = {}
            for symbol, requested, used, fraction in zip(
                    symbols, raw_magmoms, moments, fractions):
                key = (symbol, requested, used, fraction)
                summary[key] = summary.get(key, 0) + 1
            for (symbol, requested, used, fraction), count in summary.items():
                origin = ""
                if carried is not None:
                    origin = (" [carried over from the last run, the block asks "
                              "for %+.3f]" % requested)
                messages.append(
                    "   \\U0001f9f2 %d x %-2s  MAGMOM %+.3f \\u03bcB  \\u2192  "
                    "starting_magnetization %+.4f%s"
                    % (count, symbol, used, fraction, origin)
                )
        atoms.set_initial_magnetic_moments(fractions)
        labels = qe_species_labels(symbols, fractions)
        if len(labels) > 10:
            messages.append(
                "   \\u26a0\\ufe0f  %d distinct species - pw.x is built with "
                "ntypx = 10 by default and will refuse the input." % len(labels)
            )
    else:
        # The sidebar may have asked for nspin = 2; ISPIN = 1 in the file undoes
        # it for this structure, and a leftover moment array would switch spin
        # back on inside ase.io.espresso.
        if system.pop("nspin", None) == 2:
            messages.append("   \\U0001f9f2 spin polarisation off for this "
                            "structure (ISPIN = 1)")
        system.pop("starting_magnetization", None)
        if atoms.has("initial_magmoms"):
            atoms.set_initial_magnetic_moments([0.0] * len(symbols))
        labels = qe_species_labels(symbols)

    if "NUPDOWN" in keys:
        nupdown = qe_coerce(keys["NUPDOWN"])
        if isinstance(nupdown, (int, float)) and nupdown >= 0:
            system["tot_magnetization"] = float(nupdown)
            messages.append("   \\U0001f9f2 tot_magnetization = %.2f (NUPDOWN)"
                            % float(nupdown))

    # --- Hubbard U ---------------------------------------------------------
    cards, hubbard_system = qe_hubbard_from_keys(
        keys, species_order, labels, pseudo_map, pseudo_dir,
        cfg.get("hubbard_style", "card"), cfg.get("hubbard_projectors", "atomic"),
        messages)
    system.update(hubbard_system)
    if cards:
        calc.parameters["additional_cards"] = cards

    # --- the stress tensor, only when something wants it -------------------
    # pw.x computes the stress after the SCF, in silence, and it is expensive.
    # Nothing but a cell relaxation, an elastic run or an EOS asks for it, so in
    # "auto" mode it stays off until ASE requests the stress property, and the
    # answer is remembered for as long as the same structure is being worked on
    # (a cell relaxation keeps it; the supercells of a phonon run start over).
    stress_mode = str(cfg.get("compute_stress", "auto"))
    if stress_mode == "always":
        input_data.setdefault("control", {})["tstress"] = True
    elif stress_mode == "never":
        input_data.setdefault("control", {})["tstress"] = False
    else:
        species = tuple(symbols)
        latched_for, latched = getattr(calc, "_qe_stress_latch", (None, False))
        if latched_for != species:
            latched = False
        latched = latched or bool(getattr(calc, "_qe_stress_requested", False))
        calc._qe_stress_latch = (species, latched)
        input_data.setdefault("control", {})["tstress"] = latched

    # --- start from the previous step instead of free atoms ----------------
    reused = qe_reuse_previous_scf(calc, atoms, input_data, cfg, system)
    if reused:
        messages.append("   \\u267b\\ufe0f  starting from the previous step's %s "
                        "(startingpot = 'file')" % reused)

    # --- raw pw.x lines from the same file (highest precedence) ------------
    for section, values in (passthrough or {}).items():
        input_data.setdefault(section, {}).update(values)
        for key, value in values.items():
            messages.append("   \\U0001f9f2 %s.%s = %r" % (section, key, value))

    for key in keys:
        if key in QE_IGNORED_KEYS:
            messages.append("   \\u2139\\ufe0f  %s ignored - %s"
                            % (key, QE_IGNORED_KEYS[key]))
        elif key in QE_SIDEBAR_KEYS:
            messages.append("   \\u2139\\ufe0f  %s ignored here - it comes from "
                            "the sidebar settings" % key)
        elif key not in ("ISPIN", "MAGMOM", "NUPDOWN", "SPECIES", "LDAU",
                         "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ"):
            messages.append("   \\u26a0\\ufe0f  %s ignored - not translated to a "
                            "pw.x keyword" % key)

    # A relaxation calls this once per ionic step; the same twenty lines every
    # time would bury the SCF progress. The details go out once per file (and
    # again if the file is edited mid-run), then a one-line reminder.
    headline = []
    if magnetic:
        if raw_magmoms:
            headline.append("nspin = 2, total moment %g \\u03bcB, %d species"
                            % (sum(raw_magmoms), len(labels)))
        else:
            headline.append("nspin = 2")
    if cards:
        headline.append(", ".join(cards[1:]) + " eV")
    elif hubbard_system:
        headline.append("Hubbard_U on %d species"
                        % len([k for k in hubbard_system if k.startswith("Hubbard_U")]))
    if reused:
        headline.append("\\u267b\\ufe0f  reusing the previous %s" % reused)

    if stamp is not None and stamp in QE_REPORTED_FILES:
        qe_emit("   \\U0001f9f2 %s: %s" % (source_name or "spin/+U",
                                          "; ".join(headline) or "applied"))
    else:
        if stamp is not None:
            QE_REPORTED_FILES.add(stamp)
        for message in messages:
            qe_emit(message)

    # pw.x refuses fixed occupations for a magnetic system unless the total
    # moment is pinned ("fixed occupations and lsda need tot_magnetization").
    # Catching it here costs nothing; letting pw.x catch it costs an mpirun and
    # gives a message that does not say what to do about it.
    if (magnetic and str(system.get("occupations", "")).startswith("fixed")
            and "tot_magnetization" not in system):
        total = sum(raw_magmoms) if raw_magmoms else None
        suggestion = ("NUPDOWN = %g" % total) if total is not None else "NUPDOWN = <moment>"
        where = ("to %s" % path) if path else "to the spin/+U block"
        raise RuntimeError(
            "pw.x cannot use fixed occupations for a spin-polarised system "
            "unless the total magnetization is given (it stops with 'fixed "
            "occupations and lsda need tot_magnetization').\\n"
            "Either set ISMEAR/occupations to 'smearing' in the settings - the "
            "usual choice for metals and transition-metal oxides - or pin the "
            "moment by adding '%s' %s, which becomes tot_magnetization "
            "(the number of unpaired electrons per cell, kept fixed for the "
            "whole run)." % (suggestion, where)
        )

    # get_property() copies the atoms *before* calling calculate(), so the copy
    # it kept has no moments yet; without this the very next property request
    # would look like a changed system and run pw.x a second time.
    if getattr(calc, "atoms", None) is not None:
        calc.atoms = atoms.copy()
'''


# ---------------------------------------------------------------------------
# Failure diagnostics and live SCF progress
#
# A failed pw.x run only reaches Python as "returned non-zero exit status 1",
# which says nothing. The real message ("Error in routine ...", a missing
# pseudopotential, an MPI abort) is in espresso.pwo/espresso.err. This snippet
# wraps the calculator so those lines travel with the exception.
#
# ASE hands pw.x's stdout straight to espresso.pwo, so nothing at all reaches
# the console while a run is going - for DFT that can mean hours of silence.
# The same wrapper therefore tails that file in a background thread and echoes
# each SCF iteration with its estimated accuracy against conv_thr, so the user
# can see how far the electronic loop still has to go.
#
# It is kept as source so the interface and the standalone scripts run byte-for
# byte the same code: the generated script embeds it, the app execs it.
# ---------------------------------------------------------------------------
QE_DIAGNOSTICS_SRC = '''
import math as _qemath
import os as _qeos
import threading as _qethreading
import time as _qetime

# Live progress is on by default; QE_SCF_PROGRESS=0 silences it.
QE_PROGRESS_ENABLED = _qeos.environ.get("QE_SCF_PROGRESS", "1").lower() not in (
    "0", "false", "no", "off",
)
QE_PROGRESS_POLL = float(_qeos.environ.get("QE_SCF_PROGRESS_POLL", "2"))
QE_PROGRESS_HEARTBEAT = float(_qeos.environ.get("QE_SCF_PROGRESS_HEARTBEAT", "120"))

# pw.x reports in Ry; the interface asks for eV (as VASP does), so the progress
# lines are converted to match. QE_SCF_UNITS=Ry keeps the raw pwo numbers.
QE_RY_TO_EV = 13.605693122994
QE_BOHR_TO_ANG = 0.529177210903
QE_PROGRESS_IN_RY = _qeos.environ.get("QE_SCF_UNITS", "eV").lower().startswith("ry")
QE_ENERGY_UNIT = "Ry" if QE_PROGRESS_IN_RY else "eV"
QE_ENERGY_SCALE = 1.0 if QE_PROGRESS_IN_RY else QE_RY_TO_EV
QE_FORCE_UNIT = "Ry/au" if QE_PROGRESS_IN_RY else "eV/A"
QE_FORCE_SCALE = 1.0 if QE_PROGRESS_IN_RY else QE_RY_TO_EV / QE_BOHR_TO_ANG

# Six decimals hide the tail end of a tight SCF: pw.x prints the total energy
# with eight decimals in Ry, and one unit of that last digit is 1.4e-7 eV, so a
# run heading for conv_thr = 1e-8 Ry looks frozen unless the eV value is shown
# to ten decimals. The accuracy is printed with matching resolution.
QE_ENERGY_DECIMALS = max(2, int(_qeos.environ.get("QE_SCF_DECIMALS", "10")))
QE_ACCURACY_DECIMALS = 3


def qe_format_energy(value, decimals=None):
    """Total energy in the reporting unit, wide enough to see the last digit."""
    if decimals is None:
        decimals = QE_ENERGY_DECIMALS
    return f"{value * QE_ENERGY_SCALE:.{decimals}f}"


def qe_format_accuracy(value, decimals=None):
    """Scf accuracy / threshold on the log scale it actually converges on."""
    if decimals is None:
        decimals = QE_ACCURACY_DECIMALS
    return f"{value * QE_ENERGY_SCALE:.{decimals}e}"


def qe_format_elapsed(seconds):
    """Human-readable duration for the progress lines."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds / 3600.0:.2f} h"


class QEScfProgress:
    """Tail a pw.x output file while it runs and report SCF convergence.

    pw.x prints one block per electronic iteration:

        iteration #  3     ecut=    50.00 Ry     beta= 0.70
        total energy              =    -155.12345678 Ry
        estimated scf accuracy    <       0.00021 Ry

    The estimated accuracy falls roughly geometrically, so the distance to
    conv_thr is reported on a log scale - that is the honest answer to "how
    much is left", far more so than the raw iteration count.
    """

    def __init__(self, path, poll=None, heartbeat=None):
        self.path = str(path)
        self.poll = QE_PROGRESS_POLL if poll is None else poll
        self.heartbeat = QE_PROGRESS_HEARTBEAT if heartbeat is None else heartbeat
        self.conv_thr = None
        self.cycle = 0
        self._buffer = ""
        self._stop = _qethreading.Event()
        self._thread = None
        self._started = _qetime.time()
        self._last_event = self._started
        self._reset_cycle()

    def _reset_cycle(self):
        self.iteration = 0
        self.first_accuracy = None
        self.last_accuracy = None
        self.energy = None
        # What pw.x is busy with. The SCF is only the first part of a run: the
        # forces and (with tstress) the stress follow it in silence, and on a
        # big PAW/DFT+U cell they take as long as several SCF iterations.
        self.phase = "scf"

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        # ASE opens the output file in "wb", so a stale file from an earlier
        # run would be replayed before the truncation is noticed. Drop it;
        # pw.x is about to overwrite it anyway.
        try:
            _qeos.remove(self.path)
        except OSError:
            pass
        self._thread = _qethreading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.poll * 3))
            self._thread = None

    def _run(self):
        handle = None
        try:
            while True:
                # Read the stop flag first, so the loop always drains the file
                # once more after pw.x has exited.
                stopping = self._stop.is_set()
                if handle is None:
                    try:
                        handle = open(self.path, "r", errors="replace")
                    except OSError:
                        handle = None
                if handle is not None:
                    self._drain(handle)
                if stopping:
                    break
                self._heartbeat()
                self._stop.wait(self.poll)
        except Exception:
            # Progress reporting must never break the calculation itself.
            pass
        finally:
            if handle is not None:
                handle.close()

    def _drain(self, handle):
        chunk = handle.read()
        if not chunk:
            return
        lines = (self._buffer + chunk).split("\\n")
        self._buffer = lines.pop()
        for line in lines:
            self._feed(line)

    # -- parsing -----------------------------------------------------------
    @staticmethod
    def _number(text, separator):
        try:
            token = text.split(separator, 1)[1].split()[0]
            return float(token.replace("D", "E").replace("d", "e"))
        except (IndexError, ValueError):
            return None

    def _feed(self, line):
        text = line.strip()
        if not text:
            return

        # pw.x 7.x writes "scf convergence threshold =", older versions leave
        # the "scf" off, so match on the part they have in common.
        if "convergence threshold" in text and "=" in text:
            self.conv_thr = self._number(text, "=")
        elif text.startswith("Self-consistent Calculation"):
            self.cycle += 1
            self._reset_cycle()
            if self.cycle > 1:
                self._emit(f"   \\u2500\\u2500 ionic step {self.cycle}: new SCF cycle")
        elif text.startswith("iteration #"):
            number = self._number(text.replace("#", "# "), "#")
            if number is not None:
                self.iteration = int(number)
        elif text.startswith("!") and "total energy" in text:
            self.energy = self._number(text, "=")
        elif text.startswith("total energy") and "=" in text:
            self.energy = self._number(text, "=")
        elif text.startswith("estimated scf accuracy"):
            self._report(self._number(text, "<"))
        elif text.startswith("convergence has been achieved"):
            energy = ""
            if self.energy is not None:
                energy = (f" | E = {qe_format_energy(self.energy)} "
                          f"{QE_ENERGY_UNIT}")
            elapsed = qe_format_elapsed(_qetime.time() - self._started)
            # With scf_must_converge = .false. pw.x prints this line even when
            # it simply ran out of iterations, so the accuracy has the last word.
            if (self.conv_thr and self.last_accuracy
                    and self.last_accuracy > self.conv_thr * 1.01):
                self._emit(
                    f"   \\u26a0\\ufe0f  SCF stopped at iteration {self.iteration} "
                    f"WITHOUT reaching the threshold "
                    f"({qe_format_accuracy(self.last_accuracy)} vs "
                    f"{qe_format_accuracy(self.conv_thr)} {QE_ENERGY_UNIT}) - "
                    f"scf_must_converge is off, so this step is used as it is"
                    f"{energy} | {elapsed}"
                )
            else:
                self._emit(
                    f"   \\u2705 SCF converged after {self.iteration} iterations"
                    f"{energy} | {elapsed}"
                )
            self._enter_phase("forces")
        elif text.startswith("convergence NOT achieved"):
            self._emit(f"   \\u26a0\\ufe0f  {text}")
            self._enter_phase("forces")
        elif text.startswith("Forces acting on atoms"):
            self._enter_phase("forces")
        elif text.startswith("Total force ="):
            force = self._number(text, "=")
            if force is not None:
                self._emit(f"   \\U0001f4d0 Total force = "
                           f"{force * QE_FORCE_SCALE:.6f} {QE_FORCE_UNIT} "
                           f"| {qe_format_elapsed(_qetime.time() - self._started)}")
        elif text.startswith("Computing stress"):
            self._enter_phase("stress")
        elif text.startswith("total   stress") or text.startswith("total stress"):
            self._enter_phase("writing")
        elif text.startswith("Writing") or text.startswith("JOB DONE"):
            self._enter_phase("writing")
        elif text.startswith("End of BFGS Geometry Optimization"):
            self._emit("   \\U0001f3c1 Geometry optimization finished")

    # What each phase costs is invisible in the output file - pw.x prints
    # nothing between "convergence has been achieved" and the force table, and
    # nothing at all while the stress is computed - so the phase is announced
    # once and then repeated by the heartbeat.
    QE_PHASE_TEXT = {
        "forces": "computing the forces",
        "stress": "computing the stress tensor",
        "writing": "writing the output files",
    }

    def _enter_phase(self, phase):
        if phase == self.phase:
            return
        self.phase = phase
        message = self.QE_PHASE_TEXT.get(phase)
        if message:
            self._emit(f"   \\u2699\\ufe0f  {message} (pw.x prints nothing until "
                       f"it is done)")

    def _report(self, accuracy):
        if accuracy is None:
            return
        if self.first_accuracy is None and accuracy > 0:
            self.first_accuracy = accuracy
        self.last_accuracy = accuracy

        parts = [f"   \\u269b\\ufe0f  SCF iter {self.iteration:>3d}"]
        if self.energy is not None:
            parts.append(f"E = {qe_format_energy(self.energy)} {QE_ENERGY_UNIT}")
        if self.conv_thr:
            parts.append(
                f"accuracy {qe_format_accuracy(accuracy)} {QE_ENERGY_UNIT} "
                f"(target {qe_format_accuracy(self.conv_thr)})"
            )
        else:
            parts.append(f"accuracy {qe_format_accuracy(accuracy)} {QE_ENERGY_UNIT}")

        remaining = self._remaining(accuracy)
        if remaining is not None:
            done, decades = remaining
            parts.append(f"~{done:.0f}% ({decades:.1f} decades to go)")
        parts.append(qe_format_elapsed(_qetime.time() - self._started))
        self._emit(" | ".join(parts))

    def _remaining(self, accuracy):
        """(percent done, decades left) on the log scale the SCF converges on."""
        if not self.conv_thr or accuracy is None or accuracy <= 0:
            return None
        if not self.first_accuracy or self.first_accuracy <= 0:
            return None
        decades = max(0.0, _qemath.log10(accuracy) - _qemath.log10(self.conv_thr))
        span = _qemath.log10(self.first_accuracy) - _qemath.log10(self.conv_thr)
        if span <= 0:
            return 100.0, decades
        walked = _qemath.log10(self.first_accuracy) - _qemath.log10(accuracy)
        return max(0.0, min(100.0, 100.0 * walked / span)), decades

    # -- output ------------------------------------------------------------
    def _heartbeat(self):
        """Reassure the user during a long silence, saying what it is waiting on."""
        if self.heartbeat <= 0 or self.iteration == 0:
            return
        now = _qetime.time()
        if now - self._last_event >= self.heartbeat:
            elapsed = qe_format_elapsed(now - self._started)
            what = self.QE_PHASE_TEXT.get(
                self.phase, f"in SCF iteration {self.iteration}")
            if self.phase == "scf":
                what = f"in SCF iteration {self.iteration}"
            self._emit(f"   \\u23f3 still {what} ({elapsed} into this pw.x run)")

    def _emit(self, message):
        self._last_event = _qetime.time()
        print(message, flush=True)


QE_FAILURE_MARKERS = (
    "Error in routine",             # the %%%%% banner QE prints for real errors
    "convergence NOT achieved",     # ran out of SCF iterations
    "Maximum CPU time exceeded",
    "SCF correction compared to forces is large",
)

# pw.x says what went wrong but rarely what to do about it. These are the ones
# this interface can actually give an answer to.
QE_FAILURE_HINTS = (
    (
        "convergence NOT achieved",
        "The SCF cycle ran out of iterations, so pw.x stopped with exit status "
        "2 and the run above it (relaxation, MD, phonons) could not go on. "
        "Options: raise NELM (electron_maxstep), lower AMIX (mixing_beta) to "
        "~0.2, use mixing_mode = 'local-TF', add smearing - or tick 'Carry on "
        "when the SCF hits NELM' in the sidebar "
        "(electrons.scf_must_converge = .false.), which lets pw.x return the "
        "unconverged step instead of stopping.",
    ),
    (
        "fixed occupations and lsda need tot_magnetization",
        "A spin-polarised calculation with fixed occupations has to have its "
        "total moment pinned. Set ISMEAR/occupations to 'smearing' in the "
        "sidebar (the usual choice for metals and transition-metal oxides), or "
        "add NUPDOWN = <unpaired electrons per cell> to the structure's spin/+U "
        "file, which becomes tot_magnetization.",
    ),
    (
        "Gamma-only calculation for this case not implemented",
        "pw.x has no ortho-atomic/norm-atomic Hubbard projectors for a "
        "Gamma-only run. Switch the projectors to 'atomic' in the sidebar, or "
        "use a 1x1x1 Monkhorst-Pack grid instead of Gamma-only.",
    ),
    (
        "not orthogonal operation",
        "The cell and the symmetry pw.x detected disagree - this usually means "
        "the structure is slightly distorted. Relax with FixSymmetry off, or "
        "add system.nosym = .true. to the extra pw.x parameters.",
    ),
    (
        "charge is wrong",
        "The charge density integrates to the wrong number of electrons, which "
        "normally means the cutoffs are too low for these pseudopotentials. "
        "Raise ENCUT/ENAUG (a JSON manifest in the pseudopotential folder gives "
        "the recommended pair).",
    ),
)


def qe_failure_details(directory, max_lines=25):
    """Pull the useful part of a failed pw.x run out of its output files."""
    import os as _os

    chunks = []
    seen = []
    for name in ("espresso.pwo", "espresso.err"):
        path = _os.path.join(str(directory), name)
        try:
            with open(path, errors="replace") as handle:
                lines = handle.read().splitlines()
        except OSError:
            continue
        if not lines:
            continue

        # Match the hints against the whole file: the cause is often far above
        # the excerpt that gets shown.
        for marker, hint in QE_FAILURE_HINTS:
            if hint not in seen and any(marker in line for line in lines):
                seen.append(hint)

        # The cause is rarely in the last lines - pw.x keeps printing its timing
        # summary after it has decided to stop - so look for it by name first.
        marked = [n for n, line in enumerate(lines)
                  if any(marker in line for marker in QE_FAILURE_MARKERS)]
        if marked:
            start = max(0, marked[0] - 2)
            excerpt = lines[start:marked[0] + max_lines]
        else:
            excerpt = lines[-max_lines:]

        excerpt = [line for line in excerpt if line.strip()]
        if excerpt:
            chunks.append(f"--- {name} ---\\n" + "\\n".join(excerpt))

    if not chunks:
        return f"(no output found in {directory}; check that pw.x can start at all)"

    report = "\\n".join(chunks)
    for hint in seen:
        report += "\\n\\n" + hint
    return report


# A DFT+U run prints its occupation matrices exactly where ase.io.espresso
# expects the Kohn-Sham eigenvalues, and the reader gives up with an
# AssertionError instead of an energy. The block is filtered out of a copy of
# the output before it is parsed; espresso.pwo itself is left alone.
QE_HUBBARD_BLOCK_END = ("occupied Hubbard levels", "occupied +U levels")
QE_HUBBARD_BLOCK_ABORT = ("SPIN UP", "bands (ev)", "the Fermi energy is",
                          "highest occupied", "End of self-consistent",
                          "Forces acting on atoms", "total energy")


def qe_strip_hubbard_occupations(path):
    """Write a copy of ``path`` without the HUBBARD OCCUPATIONS blocks.

    Returns the new path, or None when there is nothing to strip.
    """
    try:
        with open(path, errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return None
    if not any("HUBBARD OCCUPATIONS" in line for line in lines):
        return None

    kept = []
    skipping = False
    for line in lines:
        if not skipping:
            if "HUBBARD OCCUPATIONS" in line:
                skipping = True
            else:
                kept.append(line)
            continue
        if any(marker in line for marker in QE_HUBBARD_BLOCK_END):
            skipping = False
        elif any(marker in line for marker in QE_HUBBARD_BLOCK_ABORT):
            skipping = False
            kept.append(line)

    base, ext = _qeos.path.splitext(path)
    cleaned = base + "_ase" + (ext or ".pwo")
    with open(cleaned, "w") as handle:
        handle.writelines(kept)
    return cleaned


def qe_patch_output_reader(template):
    """Make the ASE output reader survive a DFT+U output file."""
    original = template.read_results

    def read_results(directory):
        from ase.io import read as _qe_read

        path = _qeos.path.join(str(directory), template.outputname)
        cleaned = qe_strip_hubbard_occupations(path)
        if cleaned is None:
            return original(directory)
        atoms = _qe_read(cleaned, format="espresso-out")
        return dict(atoms.calc.properties())

    template.read_results = read_results


class EspressoWithDiagnostics(Espresso):
    """Espresso calculator that streams SCF progress and reports pw.x errors."""

    def check_state(self, atoms, tol=1e-15):
        """Ignore the magnetic moments this calculator sets itself.

        The moments come from the settings, not from the structure, so a fresh
        Atoms object built from the same structure carries none - and ASE would
        read that as a changed system and run a whole extra SCF on a geometry it
        has just finished. Everything that really defines the system (positions,
        cell, species) is still compared.
        """
        from ase.calculators.calculator import compare_atoms

        return compare_atoms(self.atoms, atoms, tol=tol,
                             excluded_properties={"initial_magmoms"})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Kept pristine: ase.io.espresso adds nspin/starting_magnetization(i)
        # to input_data while writing, which would otherwise carry over to the
        # next structure.
        self._qe_base_parameters = _qecopy.deepcopy(dict(self.parameters))
        qe_patch_output_reader(self.template)

    def calculate(self, *args, **kwargs):
        atoms = args[0] if args else kwargs.get("atoms")
        properties = args[1] if len(args) > 1 else kwargs.get("properties")
        self._qe_stress_requested = "stress" in (properties or [])
        if atoms is not None:
            qe_prepare_structure(self, atoms)

        outputname = getattr(self.template, "outputname", "espresso.pwo")
        monitor = None
        if QE_PROGRESS_ENABLED:
            monitor = QEScfProgress(
                _qeos.path.join(str(self.directory), outputname)
            ).start()
        try:
            try:
                super().calculate(*args, **kwargs)
                qe_record_converged_magmoms(self)
            finally:
                # Stop before the diagnostics read the file, so the last
                # iterations are echoed ahead of any error banner.
                if monitor is not None:
                    monitor.stop()
        except Exception as exc:
            raise RuntimeError(
                "Quantum ESPRESSO (pw.x) failed.\\n"
                + qe_failure_details(self.directory)
            ) from exc
'''


_DIAGNOSTICS_NS = {}
_OVERRIDES_NS = {}
_ACTIVE_STRUCTURE_NAME = None


def _espresso_with_diagnostics():
    """Build (once) the diagnostics-wrapped Espresso class for in-app use."""
    if 'EspressoWithDiagnostics' not in _DIAGNOSTICS_NS:
        from ase.calculators.espresso import Espresso
        _DIAGNOSTICS_NS['Espresso'] = Espresso
        exec(QE_OVERRIDES_SRC, _DIAGNOSTICS_NS)
        exec(QE_DIAGNOSTICS_SRC, _DIAGNOSTICS_NS)
    return _DIAGNOSTICS_NS['EspressoWithDiagnostics']


def overrides_runtime():
    """The override helpers (parsing, file lookup) for use outside a run.

    Same source the calculator runs, exec'd on its own so the sidebar can use it
    without pulling in ASE.
    """
    if not _OVERRIDES_NS:
        exec(QE_OVERRIDES_SRC, _OVERRIDES_NS)
    return _OVERRIDES_NS


def overrides_runtime_config(settings):
    """The kwargs qe_configure_overrides() takes, from the sidebar settings."""
    s = _merged(settings)
    return {
        'enabled': bool(s['per_structure_overrides']),
        'source': s['overrides_source'],
        'text': s['overrides_text'] or '',
        'dir': s['overrides_dir'] or '',
        'magmom_units': s['magmom_units'],
        'hubbard_style': s['hubbard_style'],
        'hubbard_projectors': s['hubbard_projectors'],
        'require_file': bool(s['overrides_required']),
        'nspin': int(s['nspin']),
        'starting_magnetization': float(s['starting_magnetization']),
        'carry_magmoms': bool(s['carry_magmoms']),
        'reuse_density': bool(s['reuse_density']),
        'reuse_wavefunctions': bool(s['reuse_wavefunctions']),
        'compute_stress': s['compute_stress'],
    }


# ---------------------------------------------------------------------------
# INCAR import
#
# Pasting a working INCAR is the fastest way to reproduce a VASP calculation
# here: the electronic settings become pw.x keywords, the relaxation settings
# become the app's geometry-optimisation settings, and the magnetic/Hubbard part
# stays where it belongs - in the per-structure file, because it is per atom.
# ---------------------------------------------------------------------------
ALGO_TO_DIAGONALIZATION = {
    'normal': 'david', 'fast': 'david', 'veryfast': 'rmm-davidson',
    'all': 'cg', 'damped': 'cg', 'conjugate': 'cg', 'a': 'cg', 'n': 'david',
    'f': 'david', 'v': 'rmm-davidson',
}
IBRION_TO_OPTIMIZER = {
    -1: None, 1: 'BFGS', 2: 'BFGS', 3: 'FIRE', 5: None, 6: None, 7: None, 8: None,
}
IVDW_TO_VDW_CORR = {
    1: 'grimme-d2', 10: 'grimme-d2', 11: 'grimme-d3', 12: 'grimme-d3',
    2: 'ts-vdw', 20: 'ts-vdw', 21: 'mbd', 202: 'mbd', 4: 'xdm',
}
GGA_TO_INPUT_DFT = {
    'pe': 'PBE', 'ps': 'PBESOL', 'rp': 'RPBE', 'am': 'PBE', '91': 'PW91',
    'b3': 'B3LYP', 're': 'REVPBE',
}
METAGGA_TO_INPUT_DFT = {
    'scan': 'SCAN', 'r2scan': 'R2SCAN', 'rscan': 'RSCAN', 'tpss': 'TPSS',
}

# Per-atom keys: they cannot become plain sidebar values, so they are kept as a
# block of INCAR text that the calculator translates against each structure.
INCAR_MAGNETIC_KEYS = ('MAGMOM', 'LDAU', 'LDAUTYPE', 'LDAUL', 'LDAUU', 'LDAUJ',
                       'NUPDOWN')
INCAR_BLOCK_KEYS = INCAR_MAGNETIC_KEYS + ('ISPIN', 'SPECIES')


def extract_magnetic_block(text):
    """The MAGMOM/LDAU part of an INCAR, verbatim.

    Comments are dropped except the ``# Species order:`` line, which says which
    element each LDAUL/LDAUU entry belongs to.
    """
    import re

    kept = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line[0] in '#!':
            if re.search(r"species\s*order", line, re.IGNORECASE):
                kept.append(raw.rstrip())
            continue
        if '=' not in line:
            continue
        key = line.split('=', 1)[0].strip().upper()
        if key in INCAR_BLOCK_KEYS:
            kept.append(raw.rstrip())
    return "\n".join(kept)


def _incar_number(value, default=None):
    """First numeric token of an INCAR value ('1E-6', '-1E-3 ...')."""
    runtime = overrides_runtime()
    token = str(value).replace(',', ' ').split()
    if not token:
        return default
    coerced = runtime['qe_coerce'](token[0])
    return coerced if isinstance(coerced, (int, float)) and not isinstance(coerced, bool) else default


def incar_to_settings(text, current=None, structure_sizes=None):
    """Translate a pasted INCAR.

    Returns ``(qe_updates, geometry_updates, report)`` where ``report`` is a list
    of ``(kind, message)`` with kind in ``'ok' | 'warn' | 'skip'``. Nothing is
    applied here — the caller decides.
    """
    runtime = overrides_runtime()
    s = _merged(current)
    keys, passthrough, _species = runtime['qe_parse_override_text'](text)

    qe_updates = {}
    geometry = {}
    report = []
    handled = set()

    def ok(message):
        report.append(('ok', message))

    def warn(message):
        report.append(('warn', message))

    def skip(key, reason):
        report.append(('skip', f"`{key}` — {reason}"))

    def take(key):
        handled.add(key)
        return keys.get(key)

    # --- plane-wave basis --------------------------------------------------
    if 'ENCUT' in keys:
        encut = _incar_number(take('ENCUT'))
        if encut:
            ratio = float(s['ecutrho']) / float(s['ecutwfc']) if s['ecutwfc'] else 8.0
            ratio = min(max(ratio, 4.0), 12.0)
            qe_updates['ecutwfc'] = ev_to_ry(encut)
            qe_updates['ecutrho'] = ev_to_ry(encut * ratio)
            ok(f"`ENCUT = {encut:g}` → `ecutwfc = {ev_to_ry(encut):.1f}` Ry "
               f"(ENAUG = {encut * ratio:.0f} eV, {ratio:g}× ENCUT)")
            warn("A VASP PAW cutoff does not transfer one-to-one: check ENCUT/ENAUG "
                 "against what your pseudopotentials recommend (the 📖 box above "
                 "reads SSSP's table when it finds one).")

    # --- SCF ---------------------------------------------------------------
    if 'EDIFF' in keys:
        ediff = _incar_number(take('EDIFF'))
        if ediff:
            qe_updates['conv_thr'] = ev_to_ry(abs(ediff))
            ok(f"`EDIFF = {abs(ediff):g}` → `conv_thr = {ev_to_ry(abs(ediff)):.2e}` Ry")
    if 'NELM' in keys:
        nelm = _incar_number(take('NELM'))
        if nelm:
            qe_updates['electron_maxstep'] = int(nelm)
            ok(f"`NELM = {int(nelm)}` → `electron_maxstep`")
    if 'ALGO' in keys:
        algo = str(take('ALGO')).strip().lower()
        mapped = ALGO_TO_DIAGONALIZATION.get(algo)
        if mapped:
            qe_updates['diagonalization'] = mapped
            ok(f"`ALGO = {algo}` → `diagonalization = '{mapped}'`")
        else:
            skip('ALGO', f"'{algo}' has no pw.x counterpart; leaving "
                         f"`diagonalization = '{s['diagonalization']}'`")
    if 'AMIX' in keys:
        amix = _incar_number(take('AMIX'))
        if amix:
            qe_updates['mixing_beta'] = min(max(float(amix), 0.01), 1.0)
            ok(f"`AMIX = {amix:g}` → `mixing_beta`")

    # --- occupations -------------------------------------------------------
    if 'ISMEAR' in keys:
        ismear = _incar_number(take('ISMEAR'))
        if ismear is not None:
            ismear = int(ismear)
            if ismear <= -4:
                qe_updates['occupations'] = 'tetrahedra_opt'
                ok(f"`ISMEAR = {ismear}` → `occupations = 'tetrahedra_opt'`")
            elif ismear == -2:
                qe_updates['occupations'] = 'fixed'
                ok("`ISMEAR = -2` → `occupations = 'fixed'`")
            elif ismear == -1:
                qe_updates.update({'occupations': 'smearing', 'smearing': 'fd'})
                ok("`ISMEAR = -1` → `occupations = 'smearing'`, `smearing = 'fd'`")
            elif ismear == 0:
                qe_updates.update({'occupations': 'smearing', 'smearing': 'gaussian'})
                ok("`ISMEAR = 0` → `occupations = 'smearing'`, `smearing = 'gaussian'`")
            else:
                qe_updates.update({'occupations': 'smearing', 'smearing': 'mp'})
                ok(f"`ISMEAR = {ismear}` → `occupations = 'smearing'`, "
                   f"`smearing = 'mp'` (Methfessel-Paxton)")
    if 'SIGMA' in keys:
        sigma = _incar_number(take('SIGMA'))
        if sigma:
            qe_updates['degauss'] = ev_to_ry(sigma)
            ok(f"`SIGMA = {sigma:g}` → `degauss = {ev_to_ry(sigma):.4f}` Ry")

    # --- k-points and parallelisation --------------------------------------
    if 'KSPACING' in keys:
        kspacing = _incar_number(take('KSPACING'))
        if kspacing:
            qe_updates['kpoint_mode'] = 'kspacing'
            qe_updates['kspacing'] = kspacing / (2 * math.pi)
            ok(f"`KSPACING = {kspacing:g}` → `kspacing = "
               f"{kspacing / (2 * math.pi):.4f}` Å⁻¹ (VASP counts the 2π, ASE does not)")
    if 'KPAR' in keys:
        kpar = _incar_number(take('KPAR'))
        if kpar:
            qe_updates['npool'] = int(kpar)
            ok(f"`KPAR = {int(kpar)}` → `-nk {int(kpar)}` (k-point pools)")

    # --- system ------------------------------------------------------------
    if 'ISPIN' in keys:
        ispin = _incar_number(take('ISPIN'))
        if ispin:
            qe_updates['nspin'] = 2 if int(ispin) == 2 else 1
            ok(f"`ISPIN = {int(ispin)}` → `nspin = {qe_updates['nspin']}`")
    if 'NBANDS' in keys:
        nbands = _incar_number(take('NBANDS'))
        if nbands:
            qe_updates['nbnd'] = int(nbands)
            ok(f"`NBANDS = {int(nbands)}` → `nbnd`")
            warn("VASP counts bands per spin channel and pads them; `nbnd` is often "
                 "better left at 0 (automatic).")
    if 'IVDW' in keys:
        ivdw = _incar_number(take('IVDW'))
        mapped = IVDW_TO_VDW_CORR.get(int(ivdw)) if ivdw else None
        if mapped:
            qe_updates['vdw_corr'] = mapped
            ok(f"`IVDW = {int(ivdw)}` → `vdw_corr = '{mapped}'`")
        elif ivdw:
            skip('IVDW', f"{int(ivdw)} has no direct pw.x counterpart")
    if 'METAGGA' in keys:
        metagga = str(take('METAGGA')).strip().lower()
        mapped = METAGGA_TO_INPUT_DFT.get(metagga)
        if mapped:
            qe_updates['input_dft'] = mapped
            ok(f"`METAGGA = {metagga}` → `input_dft = '{mapped}'`")
        else:
            skip('METAGGA', f"'{metagga}' is not available in pw.x")
    elif 'GGA' in keys:
        gga = str(take('GGA')).strip().lower()
        mapped = GGA_TO_INPUT_DFT.get(gga)
        if mapped:
            qe_updates['input_dft'] = mapped
            ok(f"`GGA = {gga}` → `input_dft = '{mapped}'`")
            warn("`input_dft` overrides the functional the pseudopotentials were "
                 "built for — leave it empty unless you mean it.")
        else:
            skip('GGA', f"'{gga}' has no obvious pw.x name")
    if 'ISYM' in keys:
        isym = _incar_number(take('ISYM'))
        if isym is not None and int(isym) <= 0:
            extra = [line for line in str(s['extra_input_data']).splitlines()
                     if 'nosym' not in line.lower()]
            extra.append('system.nosym = .true.')
            if int(isym) < 0:
                extra.append('system.noinv = .true.')
            qe_updates['extra_input_data'] = "\n".join(x for x in extra if x.strip())
            ok(f"`ISYM = {int(isym)}` → `system.nosym = .true.` in the extra "
               f"pw.x parameters")

    # --- ionic relaxation → the app's geometry settings --------------------
    if 'NSW' in keys:
        nsw = _incar_number(take('NSW'))
        if nsw is not None:
            if int(nsw) <= 0:
                warn("`NSW = 0` is a single-point run — pick **Energy Only** as the "
                     "calculation type.")
            else:
                geometry['max_steps'] = min(max(int(nsw), 10), 5000)
                ok(f"`NSW = {int(nsw)}` → geometry optimisation **Max steps**")
    if 'EDIFFG' in keys:
        ediffg = _incar_number(take('EDIFFG'))
        if ediffg is not None and ediffg < 0:
            geometry['fmax'] = min(max(abs(ediffg), 0.001), 1.0)
            ok(f"`EDIFFG = {ediffg:g}` → **Force threshold** "
               f"{geometry['fmax']:g} eV/Å")
            if abs(ediffg) < 0.001:
                warn(f"`EDIFFG = {ediffg:g}` is below the smallest force threshold "
                     "this interface offers (0.001 eV/Å) and has been clamped.")
        elif ediffg:
            warn(f"`EDIFFG = {ediffg:g}` is an energy criterion; the optimiser here "
                 "converges on forces, so set the **Force threshold** yourself.")
    if 'ISIF' in keys:
        isif = _incar_number(take('ISIF'))
        if isif is not None:
            isif = int(isif)
            if isif <= 2:
                geometry['optimization_type'] = "Atoms only (fixed cell)"
            elif isif == 5:
                geometry['optimization_type'] = "Cell only (fixed atoms)"
            else:
                geometry['optimization_type'] = "Both atoms and cell"
                geometry['cell_constraint'] = "Full cell (lattice + angles)"
            ok(f"`ISIF = {isif}` → **{geometry['optimization_type']}**")
            if isif in (4, 6, 7):
                warn(f"`ISIF = {isif}` keeps the volume (or only the shape) fixed; "
                     "that constraint is not reproduced — the cell is relaxed freely.")
    if 'IBRION' in keys:
        ibrion = _incar_number(take('IBRION'))
        mapped = IBRION_TO_OPTIMIZER.get(int(ibrion)) if ibrion is not None else None
        if mapped:
            geometry['optimizer'] = mapped
            ok(f"`IBRION = {int(ibrion)}` → optimiser **{mapped}** "
               f"(VASP's algorithm has no exact ASE twin)")
        elif ibrion is not None:
            skip('IBRION', f"{int(ibrion)} is not an ionic relaxation this "
                           f"interface performs")

    # --- magnetism and +U --------------------------------------------------
    # These are per atom, so they cannot become single sidebar values; the block
    # is kept as INCAR text and translated against each structure at run time.
    magnetic_present = [k for k in INCAR_MAGNETIC_KEYS if k in keys]
    if magnetic_present:
        handled.update(magnetic_present)
        block = extract_magnetic_block(text)
        qe_updates['per_structure_overrides'] = True
        qe_updates['overrides_source'] = 'inline'
        qe_updates['overrides_text'] = block
        report.append(('ok', "`" + "`, `".join(magnetic_present) + "` → loaded "
                       "into **🧲 Spin & Hubbard U**; they are per atom, so they "
                       "are kept as a block and applied to each structure at run "
                       "time (no separate file needed)."))

    if 'MAGMOM' in keys:
        moments = runtime['qe_expand_vasp_values'](keys['MAGMOM'])
        sizes = sorted(set((structure_sizes or {}).values()))
        if sizes and len(moments) not in sizes:
            warn(f"`MAGMOM` expands to **{len(moments)} values**, but the loaded "
                 f"structure(s) have {', '.join(str(n) for n in sizes)} atoms — "
                 "pw.x will not be started until that matches.")
        else:
            report.append(('ok', f"`MAGMOM` expands to {len(moments)} values "
                                 f"(total moment {sum(moments):g} μB)"))

    # --- everything else ---------------------------------------------------
    for key in keys:
        if key in handled:
            continue
        reason = runtime['QE_IGNORED_KEYS'].get(key)
        skip(key, reason or "no pw.x counterpart")

    for section, values in (passthrough or {}).items():
        for key, value in values.items():
            report.append(('ok', f"`{section}.{key} = {value!r}` → extra pw.x "
                                 "parameter"))
    if passthrough:
        extra = [line for line in str(s['extra_input_data']).splitlines() if line.strip()]
        for section, values in passthrough.items():
            for key, value in values.items():
                extra.append(f"{section}.{key} = {value}")
        qe_updates['extra_input_data'] = "\n".join(extra)

    return qe_updates, geometry, report


def set_active_structure_name(name):
    """Tell the calculator which structure is next, so it can find its file.

    The calculation runs one structure at a time in a single thread, so a module
    global is enough — the same arrangement the settings themselves use.
    """
    global _ACTIVE_STRUCTURE_NAME
    _ACTIVE_STRUCTURE_NAME = name or None
    if 'qe_set_structure_name' in _DIAGNOSTICS_NS:
        _DIAGNOSTICS_NS['qe_set_structure_name'](_ACTIVE_STRUCTURE_NAME)


def find_override_file(settings, structure_name=None, formula=None):
    """Locate the spin/+U file for a structure (used by the sidebar preview)."""
    runtime = overrides_runtime()
    s = _merged(settings)
    directory = os.path.abspath(os.path.expanduser(s['overrides_dir'] or '.'))
    return runtime['qe_find_override_file'](structure_name, formula, directory)


# ---------------------------------------------------------------------------
# Phonon force cache (resume an interrupted run)
# ---------------------------------------------------------------------------
PHONON_CACHE_SUBDIR = 'phonon_cache'


def _overrides_fingerprint(s, structure_name=None):
    """What the per-structure MAGMOM/+U settings contribute to the cache key."""
    if not s['per_structure_overrides']:
        return None
    if s['overrides_source'] != 'file':
        return s['overrides_text']
    directory = os.path.abspath(os.path.expanduser(s['overrides_dir'] or '.'))
    name = structure_name if structure_name is not None else _ACTIVE_STRUCTURE_NAME
    try:
        path = overrides_runtime()['qe_find_override_file'](name, None, directory)
        if not path:
            return None
        with open(path, 'r', errors='replace') as handle:
            return handle.read()
    except Exception:
        # An unreadable file must not make the key unstable: fall back to the
        # path alone, and let the per-displacement structure check do the rest.
        return str(name)


def qe_settings_fingerprint(settings=None, structure_name=None):
    """Short hash of everything in the settings that changes a computed force.

    Cached forces may only be reused when this is identical, so it covers the
    pw.x namelists, the k-points and the pseudopotentials: a raised cutoff or a
    denser mesh then lands in its own cache folder instead of silently handing
    back the numbers from the old settings.
    """
    s = _merged(settings if settings is not None else get_active_qe_settings())
    pseudo_dir = os.path.abspath(s['pseudo_dir']) if s['pseudo_dir'] else ''
    try:
        pseudopotentials = resolve_pseudopotentials(pseudo_dir, s['pseudo_overrides'])
    except Exception:
        pseudopotentials = dict(s['pseudo_overrides'] or {})

    payload = {
        'input_data': build_qe_input_data(s, calculation='scf'),
        'kpoints': build_qe_kpoint_kwargs(s),
        'pseudo_dir': pseudo_dir,
        'pseudopotentials': pseudopotentials,
        'nspin': int(s['nspin']),
        'starting_magnetization': float(s['starting_magnetization']),
        'magmom_units': s['magmom_units'],
        'hubbard_style': s['hubbard_style'],
        'hubbard_projectors': s['hubbard_projectors'],
        'overrides': _overrides_fingerprint(s, structure_name),
    }
    blob = json.dumps(payload, sort_keys=True, default=repr)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:10]


def _cache_slug(text):
    keep = '-_.'
    cleaned = ''.join(c if (c.isalnum() or c in keep) else '_' for c in str(text))
    return cleaned.strip('_')[:60] or 'structure'


# Phonon settings that decide which structures get displaced, and therefore
# which cached forces still apply. The supercell and the displacement distance
# are spelled out in the folder name instead.
PHONON_CACHE_PARAM_KEYS = (
    'pre_relax', 'pre_relax_fmax', 'pre_relax_steps', 'pre_relax_optimizer',
    'pre_relax_lattice_mode', 'pre_relax_fix_symmetry', 'pre_relax_symprec',
    'max_supercell_atoms', 'max_supercell_multiplier',
)


def qe_structure_cache_fingerprint(atoms):
    """Hash of the structure the displacements are generated from.

    Two structures may carry the same name — an edited lattice parameter, a
    swapped species — and their forces must never be mixed up, so the geometry
    itself goes into the cache key as well.
    """
    if atoms is None:
        return None
    payload = {
        'numbers': [int(z) for z in atoms.get_atomic_numbers()],
        'cell': [round(float(v), 6) for row in atoms.get_cell() for v in row],
        'positions': [round(float(v), 6) for row in atoms.get_positions() for v in row],
        'pbc': [bool(v) for v in atoms.get_pbc()],
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:8]


def qe_phonon_cache_dir(structure_name, phonon_params=None, settings=None, atoms=None):
    """Folder holding the per-displacement forces of one phonon run.

    Everything that would change a force sits in the folder name, so a rerun
    with different settings starts a fresh cache and an identical rerun picks up
    where the interrupted one stopped.
    """
    s = _merged(settings if settings is not None else get_active_qe_settings())
    p = phonon_params or {}

    delta = float(p.get('displacement_distance', p.get('delta', 0.01)))
    if p.get('auto_supercell', True):
        supercell = 'auto{:g}'.format(float(p.get('target_supercell_length', 15.0)))
    else:
        size = tuple(int(x) for x in p.get('supercell_size', (2, 2, 2)))
        supercell = '{}x{}x{}'.format(*size)

    params_blob = json.dumps({k: p.get(k) for k in PHONON_CACHE_PARAM_KEYS},
                             sort_keys=True, default=repr)
    params_key = hashlib.sha1(params_blob.encode('utf-8')).hexdigest()[:6]

    structure_key = qe_structure_cache_fingerprint(atoms) or ''
    tag = '{}_{}_d{:g}_{}{}{}'.format(
        _cache_slug(structure_name), supercell, delta,
        qe_settings_fingerprint(s, structure_name), params_key, structure_key,
    )
    root = os.path.abspath(os.path.expanduser(s['work_dir'] or 'qe_calc'))
    return os.path.join(root, PHONON_CACHE_SUBDIR, tag)


# ---------------------------------------------------------------------------
# Live calculator (interface)
# ---------------------------------------------------------------------------
def build_qe_calculator(settings=None, directory=None, calculation='scf', log=None):
    """Instantiate the ASE Quantum ESPRESSO calculator from ``settings``."""
    from ase.calculators.espresso import EspressoProfile

    espresso_cls = _espresso_with_diagnostics()
    s = _merged(settings if settings is not None else get_active_qe_settings())

    def _log(msg):
        if log is not None:
            log(msg)

    apply_qe_environment(s)

    command = build_qe_command(s)
    directory = directory or s['work_dir'] or 'qe_calc'
    os.makedirs(directory, exist_ok=True)

    # pw.x is launched with cwd=directory, so a relative pseudo_dir would be
    # resolved against the wrong folder.
    pseudo_dir = os.path.abspath(s['pseudo_dir']) if s['pseudo_dir'] else ''

    pseudopotentials = resolve_pseudopotentials(pseudo_dir, s['pseudo_overrides'])
    if not pseudopotentials:
        raise RuntimeError(
            f"No pseudopotentials (.UPF) found in '{s['pseudo_dir']}'. "
            "Set the correct pseudopotential directory in the sidebar."
        )

    input_data = build_qe_input_data(s, calculation=calculation)
    kpoint_kwargs = build_qe_kpoint_kwargs(s)

    _log(f"  Command:        {command}")
    _log(f"  Pseudo dir:     {pseudo_dir} ({len(pseudopotentials)} elements)")
    _log(f"  Work dir:       {os.path.abspath(directory)}")
    _log(f"  ENCUT/ENAUG:    {ry_to_ev(s['ecutwfc']):.0f} / {ry_to_ev(s['ecutrho']):.0f} eV"
         f"  (ecutwfc/ecutrho {s['ecutwfc']:g} / {s['ecutrho']:g} Ry)")
    _log(f"  k-points:       {kpoint_kwargs}")
    _log(f"  OMP threads:    {os.environ.get('OMP_NUM_THREADS')}")
    if s['use_gpu']:
        _log("  Mode:           GPU build (one MPI rank per GPU)")

    profile = EspressoProfile(command=command, pseudo_dir=pseudo_dir)

    _DIAGNOSTICS_NS['qe_configure_overrides'](**overrides_runtime_config(s))
    _DIAGNOSTICS_NS['qe_set_structure_name'](_ACTIVE_STRUCTURE_NAME)
    if s['per_structure_overrides']:
        _log("  Spin/+U files:  "
             f"{os.path.abspath(os.path.expanduser(s['overrides_dir'] or '.'))}")
    elif int(s['nspin']) == 2:
        _log(f"  Spin:           nspin = 2, starting_magnetization = "
             f"{float(s['starting_magnetization']):.3f} on every species")

    return espresso_cls(
        profile=profile,
        directory=directory,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        **kpoint_kwargs,
    )


# ---------------------------------------------------------------------------
# Generated-script code
# ---------------------------------------------------------------------------
def generate_qe_calculator_code(settings=None, indent="    ", calculation='scf'):
    """Emit the calculator-setup block for a standalone script.

    The generated code re-scans the pseudopotential directory at runtime so the
    script keeps working if the library is moved or extended, and it embeds the
    resolved settings literally so it needs nothing from this app.
    """
    s = _merged(settings if settings is not None else get_active_qe_settings())

    command = build_qe_command(s)
    input_data = build_qe_input_data(s, calculation=calculation)
    kpoint_kwargs = build_qe_kpoint_kwargs(s)
    threads = max(1, int(s['omp_threads']))
    diagnostics = (QE_OVERRIDES_SRC.strip() + "\n\n\n"
                   + QE_DIAGNOSTICS_SRC.strip())
    overrides_config = overrides_runtime_config(s)

    # NOTE: this block is pasted both at module level and inside functions that
    # already use `os`. A plain `import os` here would rebind `os` as a local of
    # the enclosing function and shadow the module-level import, so every use of
    # `os` *before* this block would raise UnboundLocalError. Hence the alias.
    body = f'''import os as _qe_os
_qe_os.environ["OMP_NUM_THREADS"] = "{threads}"
_qe_os.environ["MKL_NUM_THREADS"] = "{threads}"
_qe_os.environ["OPENBLAS_NUM_THREADS"] = "{threads}"

from ase.calculators.espresso import Espresso, EspressoProfile
from ase.data import chemical_symbols

QE_COMMAND = {command!r}
# pw.x is launched with cwd=QE_WORK_DIR, so the pseudo dir must be absolute.
QE_PSEUDO_DIR = _qe_os.path.abspath({s['pseudo_dir']!r})
QE_WORK_DIR = {(s['work_dir'] or 'qe_calc')!r}
QE_PSEUDO_OVERRIDES = {dict(s['pseudo_overrides'])!r}
QE_INPUT_DATA = {input_data!r}
QE_KPOINT_KWARGS = {kpoint_kwargs!r}


def resolve_qe_pseudopotentials(pseudo_dir, overrides=None):
    """Map every element with a .UPF file in pseudo_dir to its filename.

    ASE only looks up the species actually present in a structure, so this one
    dict serves every structure the script processes.
    """
    symbol_lookup = {{sym.lower(): sym for sym in chemical_symbols if sym}}
    candidates = {{}}
    if not _qe_os.path.isdir(pseudo_dir):
        raise FileNotFoundError(f"Pseudopotential directory not found: {{pseudo_dir}}")
    for fname in sorted(_qe_os.listdir(pseudo_dir)):
        if not fname.lower().endswith(".upf"):
            continue
        token = ""
        for ch in fname:
            if ch.isalpha():
                token += ch
            else:
                break
        symbol = symbol_lookup.get(token.lower())
        if symbol is None and len(token) > 2:
            symbol = symbol_lookup.get(token[:2].lower()) or symbol_lookup.get(token[:1].lower())
        if symbol is None:
            continue
        candidates.setdefault(symbol, []).append(fname)
    mapping = {{sym: sorted(files, key=lambda n: (len(n), n))[0]
               for sym, files in candidates.items()}}
    for sym, fname in (overrides or {{}}).items():
        if fname:
            mapping[sym] = fname
    return mapping


print("🔧 Initializing Quantum ESPRESSO (pw.x) calculator...")
print(f"   Command:    {{QE_COMMAND}}")
print(f"   Pseudo dir: {{QE_PSEUDO_DIR}}")

qe_pseudopotentials = resolve_qe_pseudopotentials(QE_PSEUDO_DIR, QE_PSEUDO_OVERRIDES)
if not qe_pseudopotentials:
    raise RuntimeError(f"No .UPF pseudopotentials found in {{QE_PSEUDO_DIR}}")
print(f"   Found pseudopotentials for {{len(qe_pseudopotentials)}} elements")

_qe_os.makedirs(QE_WORK_DIR, exist_ok=True)

{diagnostics}

# Spin polarisation and Hubbard U. With per-structure files enabled, each
# structure may carry a VASP-style file (INCAR syntax) in the folder below —
# named after the structure ("Fe2O3.cif.incar", "Fe2O3.incar", "INCAR_Fe2O3")
# or simply "INCAR" for all of them. Call qe_set_structure_name("Fe2O3.cif")
# before a structure to have its own file picked up.
qe_configure_overrides(**{overrides_config!r})

calculator = EspressoWithDiagnostics(
    profile=EspressoProfile(command=QE_COMMAND, pseudo_dir=QE_PSEUDO_DIR),
    directory=QE_WORK_DIR,
    input_data=QE_INPUT_DATA,
    pseudopotentials=qe_pseudopotentials,
    **QE_KPOINT_KWARGS,
)
print("✅ Quantum ESPRESSO calculator ready")
if QE_PROGRESS_ENABLED:
    print(f"   Live SCF progress is printed per iteration in {{QE_ENERGY_UNIT}} "
          "(QE_SCF_PROGRESS=0 silences it, QE_SCF_UNITS=Ry keeps pw.x's units)")
'''

    if not indent:
        return body
    return ''.join(indent + line if line.strip() else line
                   for line in body.splitlines(keepends=True))


# ---------------------------------------------------------------------------
# Settings panel
# ---------------------------------------------------------------------------
INCAR_IMPORT_PLACEHOLDER = """ENCUT  = 520
EDIFF  = 1E-6
NELM   = 200
ALGO   = Normal
ISMEAR = 0
SIGMA  = 0.05
ISPIN  = 2
MAGMOM = 12*0.0 4*1.0 8*0.0
IBRION = 2
ISIF   = 3
NSW    = 200
EDIFFG = -1E-2"""


def _render_incar_import(s, structure_sizes=None):
    """The 'paste your INCAR' box. Returns the settings, updated on import."""
    import streamlit as st

    st.caption(
        "Paste a working INCAR and it is translated into the tabs: the "
        "electronic part into **Basis**, **Electronic structure** and **SCF**, "
        "`MAGMOM`/`LDAU*` into **🧲 Spin & Hubbard U**, and "
        "`IBRION`/`ISIF`/`NSW`/`EDIFFG` into the geometry-optimisation "
        "settings below the calculation type."
    )
    text = st.text_area(
        "INCAR", height=240, key="qe_incar_text",
        placeholder=INCAR_IMPORT_PLACEHOLDER,
        label_visibility="collapsed",
    )

    col_a, col_b, _ = st.columns([1, 1, 2])
    with col_a:
        apply_it = st.button("🔍 Analyse & apply", key="qe_incar_apply",
                             type="primary", disabled=not (text or "").strip())
    with col_b:
        if st.button("🧹 Clear report", key="qe_incar_clear"):
            st.session_state.pop('qe_incar_report', None)

    if apply_it:
        try:
            qe_updates, geometry, report = incar_to_settings(
                text, current=s, structure_sizes=structure_sizes)
        except Exception as exc:
            st.error(f"❌ Could not read that INCAR: {exc}")
            return s

        s.update(qe_updates)
        # The cutoff widgets keep their value in session_state, and Streamlit
        # refuses a write to a key whose widget already exists in this run. So
        # the new cutoffs are parked here and applied at the top of the next
        # run, before any widget is built.
        if 'ecutwfc' in qe_updates or 'ecutrho' in qe_updates:
            st.session_state['qe_pending_cutoffs'] = {
                'qe_ecutwfc_ev': _clamp(ry_to_ev(qe_updates.get(
                    'ecutwfc', s['ecutwfc'])), 100.0, 5500.0),
                'qe_ecutrho_ev': _clamp(ry_to_ev(qe_updates.get(
                    'ecutrho', s['ecutrho'])), 200.0, 44000.0),
            }

        if geometry:
            defaults = st.session_state.get('default_settings')
            if isinstance(defaults, dict):
                try:
                    from helpers.initial_settings import DEFAULT_GEOMETRY_SETTINGS
                    base = dict(DEFAULT_GEOMETRY_SETTINGS)
                except Exception:
                    base = {}
                base.update(defaults.get('geometry_optimization') or {})
                base.update(geometry)
                defaults['geometry_optimization'] = base
            else:
                report.append(('warn', "The geometry-optimisation settings "
                                       "could not be reached — set them by hand."))

        st.session_state['qe_settings'] = dict(s)
        st.session_state['qe_incar_report'] = report
        st.rerun()

    report = st.session_state.get('qe_incar_report')
    if report:
        applied = [m for kind, m in report if kind == 'ok']
        warnings = [m for kind, m in report if kind == 'warn']
        skipped = [m for kind, m in report if kind == 'skip']
        st.success(f"✅ Applied {len(applied)} settings")
        if applied:
            st.markdown("\n".join(f"- {m}" for m in applied))
        for message in warnings:
            st.warning(f"⚠️ {message}")
        if skipped:
            with st.expander(f"Not translated ({len(skipped)})"):
                st.markdown("\n".join(f"- {m}" for m in skipped))
    return s


def render_qe_settings(saved=None, symbols=None, structure_names=None,
                       structure_sizes=None, on_save_defaults=None):
    """Draw the Quantum ESPRESSO settings panel and return the settings.

    Laid out as tabs because pw.x has far more knobs than an MLIP: one tab per
    part of the input file, so a setting can be found where its keyword lives.

    ``symbols`` are the elements of the currently loaded structures; when given,
    the pseudopotential picker is restricted to them and missing ones are
    flagged before the user starts a run. ``structure_names`` are the loaded
    file names, used to show which per-structure spin/+U file each one picks up,
    and ``structure_sizes`` their atom counts, used to sanity-check a MAGMOM.
    ``on_save_defaults`` is called with the settings when the user asks for them
    to be kept as the defaults; it returns True when they were written.
    """
    import streamlit as st

    s = dict(QE_DEFAULTS)
    s.update(saved or {})

    # Cutoffs parked by the INCAR import on the previous run. They have to be
    # written before the widgets that own these keys exist, hence up here.
    for key, value in (st.session_state.pop('qe_pending_cutoffs', None) or {}).items():
        st.session_state[key] = value

    st.markdown("### 🧪 QE Settings")
    st.caption(
        "Ab initio DFT via an external `pw.x` — orders of magnitude slower than "
        "an MLIP, so start with a small cell. **All energies here are in eV**; "
        "pw.x is fed the equivalent in Ry. Where a VASP keyword means the same "
        "thing it is shown in front of the label, with the QE name in the tooltip."
    )

    available = find_pseudopotentials(s['pseudo_dir'])

    tab_program, tab_basis, tab_electronic, tab_scf, tab_spin, tab_pseudo = \
        st.tabs([
            "🖥️ Program & hardware",
            "🧱 Basis & k-points",
            "🔬 Electronic structure",
            "🔁 SCF convergence",
            "🧲 Spin & Hubbard U",
            "📦 Pseudopotentials",
        ])

    # ======================================================================
    # Program & hardware (and the INCAR import, which fills the other tabs)
    # ======================================================================
    with tab_program:
        with st.expander(
            "📥 Import a VASP INCAR — fills in every tab",
            expanded=bool(st.session_state.get('qe_incar_report')),
        ):
            s = _render_incar_import(s, structure_sizes)

        col_bin, col_pseudo_dir = st.columns(2)
        with col_bin:
            s['pw_binary'] = st.text_input(
                "Path to `pw.x` binary *",
                value=s['pw_binary'],
                placeholder="/opt/qe-7.4/bin/pw.x",
                help="The pw.x executable, the bin/ folder holding it, or just "
                     "'pw.x' if it is on your PATH.",
            )
            _resolved_pw = normalize_pw_binary(s['pw_binary'])
            if _resolved_pw != s['pw_binary']:
                st.caption(f"→ resolved to `{_resolved_pw}`")
            s['pw_binary'] = _resolved_pw
        with col_pseudo_dir:
            s['pseudo_dir'] = st.text_input(
                "Pseudopotential directory * (VASP: POTCAR library)",
                value=s['pseudo_dir'],
                placeholder="/opt/qe-7.4/pseudo",
                help="Folder holding the .UPF files (e.g. SSSP, PSLibrary, SG15, "
                     "GBRV) — the QE counterpart of the VASP POTCAR library.",
            )
            available = find_pseudopotentials(s['pseudo_dir'])
            if s['pseudo_dir']:
                if available:
                    st.caption(f"✅ pseudopotentials for {len(available)} elements")
                else:
                    st.caption("❌ no `.UPF` files in this directory")

        st.markdown("**Hardware**")
        qe_device = st.radio(
            "pw.x build",
            ["CPU", "GPU (CUDA build)"],
            index=1 if s['use_gpu'] else 0,
            horizontal=True,
            help="Select GPU only if your pw.x was compiled with GPU support.",
        )
        s['use_gpu'] = qe_device.startswith("GPU")

        s['use_mpi'] = st.checkbox(
            "Run under MPI", value=s['use_mpi'],
            help="Leave this on unless pw.x was built without MPI — an MPI build "
                 "started bare usually aborts in MPI_Init.",
        )
        if not s['use_mpi']:
            st.warning(
                "⚠️ Only uncheck this for a serial (non-MPI) pw.x build. An MPI-linked "
                "pw.x launched without `mpirun` falls back to singleton startup and "
                "typically aborts in `MPI_Init_thread`."
            )

        if s['use_mpi']:
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                s['mpi_command'] = st.text_input(
                    "MPI launcher", value=s['mpi_command'],
                    help="mpirun, mpiexec or srun.",
                )
            with col_b:
                s['mpi_cores'] = st.number_input(
                    "GPUs to use" if s['use_gpu'] else "MPI cores",
                    min_value=1, max_value=1024, step=1, value=int(s['mpi_cores']),
                    help=(
                        "One MPI rank per GPU is the recommended layout for GPU builds."
                        if s['use_gpu'] else
                        "Number of MPI ranks (usually the number of physical cores)."
                    ),
                )
        else:
            s['mpi_cores'] = 1
            col_c, col_d = st.columns(2)

        with col_c:
            s['omp_threads'] = st.number_input(
                "OpenMP threads / rank",
                min_value=1, max_value=256, step=1, value=int(s['omp_threads']),
                help=(
                    "On GPU builds put the leftover CPU cores here (cores / GPUs). "
                    "On CPU builds 1 is usually fastest when all cores are MPI ranks."
                ),
            )
        with col_d:
            s['npool'] = st.number_input(
                "KPAR — k-point pools (`-nk`)",
                min_value=1, max_value=256, step=1, value=int(s['npool']),
                help="QE `-nk`, the counterpart of VASP's KPAR: splits k-points across "
                     "pools. Must divide the number of MPI ranks.",
            )

        if s['use_gpu']:
            st.info(
                "GPU build: use **one MPI rank per GPU** and set `-nk` to the number of "
                "GPUs when you have several k-points."
            )
        if s['use_mpi'] and int(s['mpi_cores']) % int(s['npool']) != 0:
            st.warning(
                f"⚠️ `-nk {int(s['npool'])}` does not divide {int(s['mpi_cores'])} MPI ranks — "
                "pw.x will refuse to start."
            )

        with st.expander("⚙️ Advanced parallelisation and paths"):
            col_e, col_f, col_g = st.columns(3)
            with col_e:
                s['ndiag'] = st.number_input(
                    "Linear-algebra procs (-nd, 0 = auto)",
                    min_value=0, max_value=1024, step=1, value=int(s['ndiag']),
                )
            with col_f:
                s['extra_pw_flags'] = st.text_input(
                    "Extra pw.x flags", value=s['extra_pw_flags'], placeholder="-ntg 2",
                )
            with col_g:
                s['work_dir'] = st.text_input(
                    "Working directory", value=s['work_dir'],
                    help="Where the .pwi/.pwo files and the QE scratch folder are written.",
                )

        st.code(build_qe_command(s), language="bash")

    # ======================================================================
    # Basis & k-points
    # ======================================================================
    with tab_basis:
        st.markdown("**Plane-wave basis**")

        # SSSP ships recommended cutoffs; using them is the single easiest way to
        # avoid an under-converged calculation. They are tabulated in Ry, so they
        # are converted here like everything else the user sees.
        suggestion = suggest_cutoffs(s['pseudo_dir'], symbols)
        if suggestion:
            sug_wfc, sug_rho, sug_src = suggestion
            scope = "your structures" if symbols else "the whole library"
            st.info(
                f"📖 `{sug_src}` recommends **ENCUT {ry_to_ev(sug_wfc):.0f} eV / "
                f"ENAUG {ry_to_ev(sug_rho):.0f} eV** for {scope} "
                f"({sug_wfc:g} / {sug_rho:g} Ry)."
            )
            if st.button("Use recommended cutoffs", key="qe_apply_cutoffs"):
                # Write straight into the widgets' state, then rerun so the number
                # inputs below pick the new values up.
                st.session_state['qe_ecutwfc_ev'] = ry_to_ev(sug_wfc)
                st.session_state['qe_ecutrho_ev'] = ry_to_ev(sug_rho)
                st.rerun()

        # Seed the widget state once; afterwards session_state is the source of
        # truth (passing both `value` and a stored key makes Streamlit complain).
        # The keys carry an _ev suffix so a session or preset holding the old
        # Rydberg-valued keys can never be read back as eV.
        # The eV ranges span the old Ry ones (10-400 / 40-3200 Ry) so no preset
        # saved before the switch is silently clamped.
        st.session_state.setdefault(
            'qe_ecutwfc_ev', _clamp(ry_to_ev(s['ecutwfc']), 100.0, 5500.0))
        st.session_state.setdefault(
            'qe_ecutrho_ev', _clamp(ry_to_ev(s['ecutrho']), 200.0, 44000.0))

        col_h, col_i = st.columns(2)
        with col_h:
            s['ecutwfc'] = ev_to_ry(st.number_input(
                "ENCUT — wavefunction cutoff (eV)",
                min_value=100.0, max_value=5500.0, step=25.0, format="%.1f",
                key="qe_ecutwfc_ev",
                help="QE `ecutwfc`, the direct equivalent of VASP's ENCUT. Use the "
                     "value recommended for your pseudopotentials.",
            ))
        with col_i:
            s['ecutrho'] = ev_to_ry(st.number_input(
                "ENAUG — charge-density cutoff (eV)",
                min_value=200.0, max_value=44000.0, step=100.0, format="%.1f",
                key="qe_ecutrho_ev",
                help="QE `ecutrho`, closest to VASP's ENAUG: ~4x ENCUT for "
                     "norm-conserving, 8-12x for US/PAW pseudopotentials.",
            ))
        st.caption(
            f"→ written to the pw.x input as `ecutwfc = {s['ecutwfc']:g}` / "
            f"`ecutrho = {s['ecutrho']:g}` Ry"
        )
        if float(s['ecutrho']) < 4 * float(s['ecutwfc']):
            st.warning("⚠️ ENAUG below 4x ENCUT (`ecutrho` < 4x `ecutwfc`) — fine only "
                       "for norm-conserving pseudopotentials.")

        st.markdown("**k-points**")
        kmode_labels = {
            'kspacing': "Automatic (k-spacing / KSPACING)",
            'grid': "Explicit Monkhorst-Pack grid (KPOINTS)",
            'gamma': "Gamma point only",
        }
        kmode_keys = list(kmode_labels)
        s['kpoint_mode'] = st.selectbox(
            "k-point mode", kmode_keys,
            index=kmode_keys.index(s['kpoint_mode']) if s['kpoint_mode'] in kmode_keys else 0,
            format_func=lambda k: kmode_labels[k],
            help="k-spacing adapts the grid to each structure's cell — best when running several structures.",
        )
        if s['kpoint_mode'] == 'kspacing':
            s['kspacing'] = st.number_input(
                "KSPACING — k-spacing (Å⁻¹)", min_value=0.01, max_value=1.0, step=0.01,
                value=float(s['kspacing']), format="%.3f",
                help="Smaller = denser grid. 0.25 is a reasonable default, 0.15 for metals. "
                     "Same idea as VASP's KSPACING but NOT the same number: ASE/QE measure "
                     "the reciprocal cell without the 2π that VASP includes.",
            )
            # The 2π convention difference bites everyone who transfers a value
            # straight from an INCAR, so spell out the equivalent.
            st.caption(
                f"≈ VASP `KSPACING = {2 * math.pi * float(s['kspacing']):.3f}` "
                "(VASP counts the 2π in **b**, ASE does not)"
            )
        elif s['kpoint_mode'] == 'grid':
            cols_k = st.columns(3)
            grid = list(s['kgrid'])
            for i, axis in enumerate("abc"):
                with cols_k[i]:
                    grid[i] = st.number_input(
                        f"k{axis}", min_value=1, max_value=64, step=1, value=int(grid[i]),
                    )
            s['kgrid'] = grid
            offset = list(s['koffset'])
            cols_o = st.columns(3)
            for i, axis in enumerate("abc"):
                with cols_o[i]:
                    offset[i] = 1 if st.checkbox(f"shift {axis}", value=bool(offset[i])) else 0
            s['koffset'] = offset
        else:
            st.caption("Γ-only: valid for molecules and very large cells.")

    # ======================================================================
    # Electronic structure
    # ======================================================================
    with tab_electronic:
        occ_labels = {
            'smearing': "Smearing (metals)",
            'fixed': "Fixed (insulators)",
            'tetrahedra_opt': "Tetrahedra (optimised)",
        }
        occ_keys = list(occ_labels)
        col_occ, col_smear, col_sigma = st.columns(3)
        with col_occ:
            s['occupations'] = st.selectbox(
                "ISMEAR — occupations", occ_keys,
                index=occ_keys.index(s['occupations']) if s['occupations'] in occ_keys else 0,
                format_func=lambda k: occ_labels[k],
                help="QE `occupations`. Plays the role VASP's ISMEAR sign does: "
                     "smearing for metals, fixed for insulators, tetrahedra for DOS.",
            )
        if s['occupations'] == 'smearing':
            with col_smear:
                s['smearing'] = st.selectbox(
                    "Smearing type (ISMEAR flavour)", SMEARING_CHOICES,
                    index=SMEARING_CHOICES.index(s['smearing'])
                    if s['smearing'] in SMEARING_CHOICES else 0,
                    help="QE `smearing`. mv = Marzari-Vanderbilt (VASP ISMEAR=1 "
                         "territory), the usual choice for metals.",
                )
            with col_sigma:
                s['degauss'] = ev_to_ry(st.number_input(
                    "SIGMA — smearing width (eV)",
                    min_value=0.001, max_value=7.0, step=0.01,
                    value=_clamp(ry_to_ev(s['degauss']), 0.001, 7.0), format="%.4f",
                    help="QE `degauss`, the direct equivalent of VASP's SIGMA.",
                ))
                st.caption(f"→ `degauss = {s['degauss']:g}` Ry")

        if (s['occupations'] == 'fixed'
                and (int(s['nspin']) == 2 or s['per_structure_overrides'])):
            st.warning(
                "⚠️ Fixed occupations + spin polarisation needs a pinned total "
                "moment — pw.x stops with *fixed occupations and lsda need "
                "`tot_magnetization`*. Either use smearing, or add `NUPDOWN` to "
                "the block in **🧲 Spin & Hubbard U**."
            )

        col_charge, col_bands = st.columns(2)
        with col_charge:
            s['tot_charge'] = st.number_input(
                "Total charge (VASP: NELECT)", min_value=-20.0, max_value=20.0,
                step=1.0, value=float(s['tot_charge']),
                help="QE `tot_charge`: added/removed electrons. VASP asks for the "
                     "absolute electron count (NELECT) instead — same knob, "
                     "opposite sign convention.",
            )
        with col_bands:
            s['nbnd'] = st.number_input(
                "NBANDS — number of bands (0 = auto)", min_value=0, max_value=100000,
                step=1, value=int(s['nbnd']),
                help="QE `nbnd`, the equivalent of VASP's NBANDS.",
            )

        s['input_dft'] = st.text_input(
            "GGA / METAGGA — override functional (input_dft)", value=s['input_dft'],
            placeholder="leave empty to use the pseudopotential's functional",
            help="QE `input_dft`, e.g. PBE, PBESOL, SCAN. Like setting GGA/METAGGA "
                 "in an INCAR, it overrides what the pseudopotential was built for.",
        )
        col_vdw, col_iso = st.columns(2)
        with col_vdw:
            s['vdw_corr'] = st.selectbox(
                "IVDW — dispersion correction", VDW_CORR_CHOICES,
                index=VDW_CORR_CHOICES.index(s['vdw_corr'])
                if s['vdw_corr'] in VDW_CORR_CHOICES else 0,
                help="QE `vdw_corr`, the counterpart of VASP's IVDW.",
            )
        with col_iso:
            s['assume_isolated'] = st.selectbox(
                "assume_isolated (VASP: monopole/dipole corrections)",
                ASSUME_ISOLATED_CHOICES,
                index=ASSUME_ISOLATED_CHOICES.index(s['assume_isolated'])
                if s['assume_isolated'] in ASSUME_ISOLATED_CHOICES else 0,
                help="Use for charged or molecular systems in a periodic box — what "
                     "VASP does with LMONO/LDIPOL/IDIPOL.",
            )

        st.caption(
            "🧲 Spin polarisation (ISPIN, MAGMOM) and Hubbard U live in their own "
            "tab; everything pw.x understands but this panel does not offer goes "
            "into **📝 Extra pw.x parameters** under 📦 Pseudopotentials."
        )

    # ======================================================================
    # SCF convergence
    # ======================================================================
    with tab_scf:
        col_m, col_n = st.columns(2)
        with col_m:
            s['conv_thr'] = ev_to_ry(st.number_input(
                "EDIFF — SCF convergence (eV)",
                min_value=1e-11, max_value=2e-1,
                value=_clamp(ry_to_ev(s['conv_thr']), 1e-11, 2e-1), format="%.2e",
                help="QE `conv_thr`, the equivalent of VASP's EDIFF. ~1.4e-5 eV "
                     "(1e-6 Ry) for energies; tighten by 2-4 orders for phonons.",
            ))
            st.caption(f"→ `conv_thr = {s['conv_thr']:.2e}` Ry")
            s['mixing_beta'] = st.number_input(
                "AMIX — mixing beta", min_value=0.01, max_value=1.0, step=0.05,
                value=float(s['mixing_beta']),
                help="QE `mixing_beta`, VASP's AMIX. Lower (0.1-0.3) helps "
                     "hard-to-converge metals and magnets.",
            )
        with col_n:
            s['electron_maxstep'] = st.number_input(
                "NELM — max SCF steps", min_value=10, max_value=5000, step=10,
                value=int(s['electron_maxstep']),
                help="QE `electron_maxstep`, VASP's NELM.",
            )
            s['mixing_mode'] = st.selectbox(
                "IMIX — mixing mode", MIXING_MODE_CHOICES,
                index=MIXING_MODE_CHOICES.index(s['mixing_mode'])
                if s['mixing_mode'] in MIXING_MODE_CHOICES else 0,
                help="QE `mixing_mode`, the counterpart of VASP's IMIX. "
                     "local-TF often helps slabs and inhomogeneous systems.",
            )
        s['diagonalization'] = st.selectbox(
            "ALGO — diagonalisation", DIAGONALIZATION_CHOICES,
            index=DIAGONALIZATION_CHOICES.index(s['diagonalization'])
            if s['diagonalization'] in DIAGONALIZATION_CHOICES else 0,
            help="QE `diagonalization`: david is the Davidson solver VASP calls "
                 "ALGO=Normal, cg the conjugate-gradient one (ALGO=All).",
        )

        s['scf_must_converge'] = not st.checkbox(
            "Carry on when the SCF hits NELM (`scf_must_converge = .false.`)",
            value=not bool(s.get('scf_must_converge', True)),
            help="By default pw.x stops with an error the moment one SCF cycle "
                 "runs out of iterations, which ends the geometry optimisation "
                 "or MD at that step. With this ticked pw.x keeps the "
                 "unconverged density, returns its energy and forces, and the "
                 "run continues — the QE equivalent of ignoring an unconverged "
                 "electronic step in VASP.",
        )
        if not s['scf_must_converge']:
            st.warning(
                "⚠️ Energies and forces from an unconverged SCF cycle are not "
                "trustworthy: a relaxation can wander or stall. Use it to get "
                "past the odd bad step, and check the log for how often "
                "`convergence NOT achieved` appears."
            )

        st.markdown("---")
        st.markdown("**Between ionic steps**")
        st.caption(
            "pw.x is launched once per ionic step, and by default each launch "
            "starts from a superposition of free atoms — throwing away the "
            "density it converged one step earlier. Reusing it is what QE's own "
            "`relax` does and typically saves a third of the SCF iterations."
        )
        col_rho, col_wfc = st.columns(2)
        with col_rho:
            s['reuse_density'] = st.checkbox(
                "Start from the previous SCF density (`startingpot = 'file'`)",
                value=bool(s.get('reuse_density', True)),
                help="Reused only when the cell, the species and nspin are "
                     "unchanged — the density is stored on the G-vectors of one "
                     "cell. A cell relaxation therefore starts each step fresh.",
            )
        with col_wfc:
            s['reuse_wavefunctions'] = st.checkbox(
                "…and the wavefunctions (`startingwfc = 'file'`)",
                value=bool(s.get('reuse_wavefunctions', False)),
                disabled=not s['reuse_density'],
                help="Saves another iteration or two, but the wavefunctions have "
                     "to be written to disk (`disk_io = 'medium'`), which is a lot "
                     "of I/O for a big cell.",
            )
        if s['reuse_density'] and s['kpoint_mode'] != 'gamma':
            st.caption(
                "ℹ️ Fixed-cell relaxations, MD and phonon displacements benefit; "
                "**Both atoms and cell** changes the cell every step, so each step "
                "starts from atoms again."
            )

        stress_labels = {
            'auto': "Auto — only when something asks for it",
            'always': "Always",
            'never': "Never",
        }
        stress_keys = list(stress_labels)
        s['compute_stress'] = st.selectbox(
            "Stress tensor (`tstress`)", stress_keys,
            index=stress_keys.index(s['compute_stress'])
            if s['compute_stress'] in stress_keys else 0,
            format_func=lambda k: stress_labels[k],
            help="pw.x computes the stress after the SCF, in silence, and on a "
                 "PAW/DFT+U cell it can cost as much as several SCF iterations. "
                 "Only a cell relaxation, an elastic run or an EOS needs it — a "
                 "phonon run, a fixed-cell relaxation or MD does not. Auto leaves "
                 "it off until the first request and then keeps it on for that "
                 "structure.",
        )
        if s['compute_stress'] == 'never':
            st.warning(
                "⚠️ With `Never`, a cell relaxation, elastic or EOS run will fail "
                "when it asks for the stress."
            )

    # ======================================================================
    # Spin & Hubbard U
    # ======================================================================
    with tab_spin:
        col_ispin, col_mag = st.columns(2)
        with col_ispin:
            s['nspin'] = 2 if st.checkbox(
                "ISPIN — spin polarised", value=int(s['nspin']) == 2,
                help="QE `nspin=2`, i.e. VASP ISPIN=2. Also switched on "
                     "automatically by a MAGMOM in the block below.",
            ) else 1
        with col_mag:
            if int(s['nspin']) == 2:
                s['starting_magnetization'] = st.number_input(
                    "MAGMOM — starting magnetisation", min_value=-1.0, max_value=1.0,
                    step=0.1, value=float(s['starting_magnetization']),
                    help="QE `starting_magnetization`: the initial moment as a "
                         "fraction of the valence charge, not the Bohr magnetons "
                         "VASP's MAGMOM takes. Applied to every species — for a "
                         "moment per atom (antiferromagnets, one magnetic "
                         "sublattice) use the block below.",
                )
        if int(s['nspin']) == 2 and not s['per_structure_overrides']:
            st.caption(
                "Same starting moment on every species, and no Hubbard U. For a "
                "moment per atom or a U, switch on the block below."
            )

        st.markdown("---")
        s['per_structure_overrides'] = st.checkbox(
            "Per-atom moments and Hubbard U (MAGMOM, LDAU — INCAR syntax)",
            value=bool(s['per_structure_overrides']),
            help="ISPIN, MAGMOM, NUPDOWN and the LDAU* keywords are read as VASP "
                 "writes them and translated into pw.x keywords (nspin, "
                 "starting_magnetization per species, HUBBARD card).",
        )

        if s['per_structure_overrides']:
            source_labels = {
                'inline': "Here — one block for every structure",
                'file': "A file next to each structure",
            }
            source_keys = list(source_labels)
            s['overrides_source'] = st.radio(
                "Where the values come from", source_keys,
                index=source_keys.index(s['overrides_source'])
                if s['overrides_source'] in source_keys else 0,
                format_func=lambda k: source_labels[k],
                horizontal=True,
                help="A file is worth it when each structure needs its own "
                     "moments; otherwise type the block here (📥 Import INCAR "
                     "fills it in for you).",
            )

            if s['overrides_source'] == 'inline':
                s['overrides_text'] = st.text_area(
                    "MAGMOM / LDAU block",
                    value=s['overrides_text'], height=220,
                    placeholder=(
                        "ISPIN  = 2\n"
                        "MAGMOM = K:0.0 Ni:1.0 O:0.0    # or 12*0.0 4*1.0 8*0.0\n"
                        "\n"
                        "# Species order: K Ni O\n"
                        "LDAU     = .TRUE.\n"
                        "LDAUTYPE = 2\n"
                        "LDAUL    = -1 2 -1\n"
                        "LDAUU    = 0.0 4.2 0.0\n"
                        "LDAUJ    = 0.0 0.0 0.0"
                    ),
                    help="INCAR syntax. Applied to every structure, so MAGMOM has "
                         "to match their atom count.",
                )
                block = str(s['overrides_text'] or "")
                if block.strip():
                    try:
                        runtime = overrides_runtime()
                        keys, extra, order = runtime['qe_parse_override_text'](block)
                        bits = []
                        if 'MAGMOM' in keys:
                            if ':' in str(keys['MAGMOM']):
                                pairs = str(keys['MAGMOM']).replace(',', ' ').split()
                                bits.append("MAGMOM per element: " + " ".join(pairs))
                            else:
                                moments = runtime['qe_expand_vasp_values'](keys['MAGMOM'])
                                bits.append(f"MAGMOM: {len(moments)} atoms, total "
                                            f"{sum(moments):g} μB")
                                sizes = sorted(set((structure_sizes or {}).values()))
                                if sizes and len(moments) not in sizes:
                                    st.error(
                                        f"❌ MAGMOM has {len(moments)} values but the "
                                        f"loaded structure(s) have "
                                        f"{', '.join(str(n) for n in sizes)} atoms — "
                                        "the run will stop on this. A per-element "
                                        "line (`Ni:1.0 O:0.0`) fits any cell."
                                    )
                                elif sizes:
                                    st.caption(
                                        "ℹ️ A per-atom MAGMOM fits only this cell. For "
                                        "phonons (which build a supercell) write it "
                                        "per element: `Ni:1.0 O:0.0`."
                                    )
                        if runtime['qe_is_true'](keys.get('LDAU', '')):
                            bits.append("Hubbard U on "
                                        + " ".join(order or ["(structure order)"]))
                        if order:
                            bits.append("species order: " + " ".join(order))
                        elif 'LDAU' in keys:
                            st.warning(
                                "⚠️ No `# Species order:` line — LDAUL/LDAUU are "
                                "matched to the elements in the order they appear "
                                "in the structure."
                            )
                        if bits:
                            st.success("✅ " + " · ".join(bits))
                    except Exception as exc:
                        st.warning(f"⚠️ Could not read the block: {exc}")
                else:
                    st.info(
                        "Empty — paste your INCAR under **📥 Import INCAR** and the "
                        "magnetic part lands here automatically."
                    )
            else:
                s['overrides_dir'] = st.text_input(
                    "Folder holding the files",
                    value=s['overrides_dir'],
                    placeholder="leave empty for the folder the app was started in",
                    help="The generated scripts look in their own working directory "
                         "when this is empty, i.e. next to the structure files.",
                )
                resolved_dir = os.path.abspath(
                    os.path.expanduser(s['overrides_dir'] or '.'))
                st.caption(f"→ `{resolved_dir}`")
                st.markdown(
                    "**File name** — for a structure `Fe2O3.cif` the first of these "
                    "that exists is used: `Fe2O3.cif.incar` → `Fe2O3.incar` → "
                    "`INCAR_Fe2O3` → `Fe2O3.qeset`, then the same four for the "
                    "chemical formula, and finally a plain **`INCAR`** that applies "
                    "to every structure."
                )
                s['overrides_required'] = st.checkbox(
                    "Fail if a structure has no file",
                    value=bool(s['overrides_required']),
                    help="Off: structures without a file simply use the plain ISPIN "
                         "setting above. On: the run stops instead, so a typo in a "
                         "file name cannot silently give you a non-magnetic "
                         "calculation.",
                )
                try:
                    runtime = overrides_runtime()
                    names = list(structure_names or [])
                    if names:
                        for name in names:
                            found = runtime['qe_find_override_file'](
                                name, None, resolved_dir)
                            if found:
                                st.success(f"✅ `{name}` → `{os.path.basename(found)}`")
                            else:
                                st.warning(f"⚠️ `{name}` → no file, plain ISPIN used")
                    elif os.path.isfile(os.path.join(resolved_dir, "INCAR")):
                        st.success("✅ `INCAR` found — applies to every structure "
                                   "without one of its own")
                    else:
                        st.info("Load structures to see which file each one picks up.")
                except Exception as exc:      # never break the panel over a preview
                    st.info(f"Could not scan the folder: {exc}")

            s['carry_magmoms'] = st.checkbox(
                "Start each structure from the moments the last one converged to",
                value=bool(s.get('carry_magmoms', False)),
                help="MAGMOM is only a starting guess: a Ni asked to start at 1 μB "
                     "may settle at 1.7. With this on, the next structure starts "
                     "from the converged value of the same MAGMOM group instead of "
                     "the number in the block. It matters where the density cannot "
                     "be reused — above all the first phonon supercell, which has a "
                     "different cell — and keeps it in the magnetic state the "
                     "relaxed cell ended in. Leave it off if you want every "
                     "structure to start from exactly what you wrote.",
            )

            col_units, col_style, col_proj = st.columns(3)
            with col_units:
                magmom_labels = {
                    'bohr': "Bohr magnetons (VASP MAGMOM)",
                    'fraction': "Fraction of valence (QE)",
                }
                magmom_keys = list(magmom_labels)
                s['magmom_units'] = st.selectbox(
                    "MAGMOM units", magmom_keys,
                    index=magmom_keys.index(s['magmom_units'])
                    if s['magmom_units'] in magmom_keys else 0,
                    format_func=lambda k: magmom_labels[k],
                    help="VASP's MAGMOM is in μB; QE's starting_magnetization is "
                         "the moment divided by the valence charge. The valence "
                         "is read from each element's UPF file (`z_valence`) to "
                         "do the conversion.",
                )
            with col_style:
                hub_labels = {
                    'card': "HUBBARD card (QE ≥ 7.1)",
                    'namelist': "Hubbard_U() in &SYSTEM (QE ≤ 7.0)",
                }
                hub_keys = list(hub_labels)
                s['hubbard_style'] = st.selectbox(
                    "Hubbard input style", hub_keys,
                    index=hub_keys.index(s['hubbard_style'])
                    if s['hubbard_style'] in hub_keys else 0,
                    format_func=lambda k: hub_labels[k],
                    help="pw.x 7.1 moved the Hubbard parameters out of &SYSTEM "
                         "into their own HUBBARD card and rejects the old keys. "
                         "Pick the namelist form only for QE 7.0 and older.",
                )
            with col_proj:
                s['hubbard_projectors'] = st.selectbox(
                    "Hubbard projectors", HUBBARD_PROJECTOR_CHOICES,
                    index=HUBBARD_PROJECTOR_CHOICES.index(s['hubbard_projectors'])
                    if s['hubbard_projectors'] in HUBBARD_PROJECTOR_CHOICES else 0,
                    help="The `HUBBARD (…)` header. ortho-atomic (Löwdin-orthogonalised "
                         "atomic orbitals) is the usual recommendation and is closest "
                         "to what VASP's PAW projectors do.",
                    disabled=s['hubbard_style'] != 'card',
                )
            if (s['hubbard_style'] == 'card'
                    and s['hubbard_projectors'] in ('ortho-atomic', 'norm-atomic')
                    and s['kpoint_mode'] == 'gamma'):
                st.warning(
                    f"⚠️ `{s['hubbard_projectors']}` projectors are not "
                    "implemented for Γ-only calculations — pw.x stops in "
                    "`orthoUwfc`. Use `atomic` projectors, or a 1×1×1 "
                    "Monkhorst-Pack grid instead of Γ-only."
                )

            with st.expander("What is translated"):
                st.markdown(
                    "- `ISPIN = 2` → `nspin = 2`\n"
                    "- `MAGMOM` → `starting_magnetization(i)` per species. Either "
                    "one value per atom as VASP writes it (`48*0.0 16*1.0 32*0.0`) "
                    "or **per element** (`K:0.0 Ni:1.0 O:0.0`), which fits any "
                    "cell — use that one for phonons, where a supercell is built. "
                    "Atoms of one element with different moments become separate "
                    "pw.x species (`Ni`, `Ni1`, …), which is how you get an "
                    "antiferromagnet\n"
                    "- `NUPDOWN` → `tot_magnetization`\n"
                    "- `LDAU`, `LDAUTYPE`, `LDAUL`, `LDAUU`, `LDAUJ` → the Hubbard "
                    "parameters, in eV as in VASP. `LDAUTYPE = 2` (Dudarev) becomes "
                    "U − J on the manifold read from the UPF file (`Ni-3d`)\n"
                    "- any `section.key = value` line goes straight into the pw.x "
                    "namelists, e.g. `electrons.mixing_ndim = 12`\n"
                    "- `LORBIT`, `LMAXMIX`, `LDAUPRINT` have no pw.x counterpart and "
                    "are reported as ignored in the log"
                )
                st.caption(
                    "The species order for LDAUL/LDAUU/LDAUJ comes from the "
                    "`# Species order:` comment (or a `SPECIES =` line). Without it "
                    "the order the elements first appear in the structure is used."
                )
        else:
            st.caption(
                "Off: every structure uses the ISPIN setting above, with the "
                "same starting magnetisation on every species and no U."
            )

    # ======================================================================
    # Pseudopotentials & the escape hatch
    # ======================================================================
    with tab_pseudo:
        if not s['pseudo_dir']:
            st.info("Set the pseudopotential directory under **🖥️ Program & hardware**.")
        elif not available:
            st.error("❌ No `.UPF` files found in this directory.")
        else:
            st.caption(
                "The shortest matching `.UPF` is picked automatically. Override any "
                "element here."
            )
            overrides = dict(s['pseudo_overrides'])
            listed = sorted(symbols) if symbols else sorted(available)
            cols = st.columns(2)
            for n, sym in enumerate(listed):
                choices = available.get(sym, [])
                with cols[n % 2]:
                    if not choices:
                        st.error(f"❌ **{sym}** — no pseudopotential found here")
                        continue
                    current = overrides.get(sym, choices[0])
                    picked = st.selectbox(
                        sym, choices,
                        index=choices.index(current) if current in choices else 0,
                        key=f"qe_pseudo_{sym}",
                    )
                    if picked != choices[0]:
                        overrides[sym] = picked
                    else:
                        overrides.pop(sym, None)
            s['pseudo_overrides'] = overrides

        st.markdown("---")
        st.markdown("**📝 Extra pw.x parameters** (the INCAR escape hatch)")
        s['extra_input_data'] = st.text_area(
            "One `section.key = value` per line",
            value=s['extra_input_data'],
            placeholder="control.verbosity = 'high'\nsystem.nosym = .true.\nelectrons.mixing_ndim = 12",
            help="Raw pw.x namelist entries — anything the fields above do not "
                 "cover. Merged into input_data, overriding anything set above. "
                 "These are QE keywords in QE units (Ry), not VASP ones.",
        )

    # ======================================================================
    # INCAR import
    # ======================================================================
    # --- status and defaults, always visible ------------------------------
    st.markdown("---")
    col_status, col_save = st.columns([3, 1])

    with col_status:
        problems = validate_qe_settings(s)
        if problems:
            for problem in problems:
                st.error(f"❌ {problem}")
        else:
            st.success("✅ Quantum ESPRESSO setup looks valid")

        if symbols:
            missing = missing_pseudopotentials(s, symbols)
            if missing:
                st.error(
                    f"❌ No pseudopotential for: {', '.join(missing)} — "
                    "the run will fail on these elements."
                )

    with col_save:
        if on_save_defaults is not None:
            if st.button("💾 Save as defaults", key="qe_save_defaults",
                         type="primary", use_container_width=True,
                         help="Keep everything on this page — pw.x path, "
                              "pseudopotential library, cutoffs, k-points, spin "
                              "and +U — as the values the panel starts from next "
                              "time."):
                if on_save_defaults(s):
                    st.session_state['qe_defaults_saved'] = True
                    st.rerun()
                else:
                    st.error("❌ Could not write the defaults file")
            if st.session_state.pop('qe_defaults_saved', False):
                st.success("✅ Saved as defaults")

    return s


# The panel used to live in the sidebar; keep the old name working.
render_qe_sidebar = render_qe_settings


QE_ENV_SETUP = {
    "pip": "pip install ase==3.28.0 pymatgen==2025.10.7 matscipy==1.2.0 phonopy==2.40.0 numpy pandas matplotlib",
    "note": (
        "Quantum ESPRESSO runs as an external program — no torch/MLIP packages are "
        "needed, but pw.x itself must be installed and the pseudopotential directory "
        "must exist on the machine that runs the script."
    ),
}
