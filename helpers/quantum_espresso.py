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
    # --- system ------------------------------------------------------------
    'nspin': 1,
    'starting_magnetization': 0.0,
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


SMEARING_CHOICES = ['mv', 'gaussian', 'mp', 'fd']
DIAGONALIZATION_CHOICES = ['david', 'cg', 'ppcg', 'paro', 'rmm-davidson']
MIXING_MODE_CHOICES = ['plain', 'TF', 'local-TF']
VDW_CORR_CHOICES = ['none', 'grimme-d2', 'grimme-d3', 'ts-vdw', 'xdm', 'mbd']
ASSUME_ISOLATED_CHOICES = ['none', 'makov-payne', 'martyna-tuckerman', 'esm']


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
        'tstress': True,      # stress is mandatory for cell relaxation/elastic
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
        system['nspin'] = 2
        system['starting_magnetization'] = float(s['starting_magnetization'])
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
# Failure diagnostics
#
# A failed pw.x run only reaches Python as "returned non-zero exit status 1",
# which says nothing. The real message ("Error in routine ...", a missing
# pseudopotential, an MPI abort) is in espresso.pwo/espresso.err. This snippet
# wraps the calculator so those lines travel with the exception.
#
# It is kept as source so the interface and the standalone scripts run byte-for
# byte the same code: the generated script embeds it, the app execs it.
# ---------------------------------------------------------------------------
QE_DIAGNOSTICS_SRC = '''
def qe_failure_details(directory, max_lines=25):
    """Pull the useful part of a failed pw.x run out of its output files."""
    import os as _os

    chunks = []
    for name in ("espresso.pwo", "espresso.err"):
        path = _os.path.join(str(directory), name)
        try:
            with open(path, errors="replace") as handle:
                lines = handle.read().splitlines()
        except OSError:
            continue
        if not lines:
            continue

        # QE prints the cause inside a %%%%% banner; prefer that when present.
        marked = [n for n, line in enumerate(lines) if "Error in routine" in line]
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
    return "\\n".join(chunks)


class EspressoWithDiagnostics(Espresso):
    """Espresso calculator that reports what pw.x actually said when it fails."""

    def calculate(self, *args, **kwargs):
        try:
            super().calculate(*args, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Quantum ESPRESSO (pw.x) failed.\\n"
                + qe_failure_details(self.directory)
            ) from exc
'''


_DIAGNOSTICS_NS = {}


def _espresso_with_diagnostics():
    """Build (once) the diagnostics-wrapped Espresso class for in-app use."""
    if 'EspressoWithDiagnostics' not in _DIAGNOSTICS_NS:
        from ase.calculators.espresso import Espresso
        _DIAGNOSTICS_NS['Espresso'] = Espresso
        exec(QE_DIAGNOSTICS_SRC, _DIAGNOSTICS_NS)
    return _DIAGNOSTICS_NS['EspressoWithDiagnostics']


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
    _log(f"  ecutwfc/ecutrho: {s['ecutwfc']} / {s['ecutrho']} Ry")
    _log(f"  k-points:       {kpoint_kwargs}")
    _log(f"  OMP threads:    {os.environ.get('OMP_NUM_THREADS')}")
    if s['use_gpu']:
        _log("  Mode:           GPU build (one MPI rank per GPU)")

    profile = EspressoProfile(command=command, pseudo_dir=pseudo_dir)

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
    diagnostics = QE_DIAGNOSTICS_SRC.strip()

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

calculator = EspressoWithDiagnostics(
    profile=EspressoProfile(command=QE_COMMAND, pseudo_dir=QE_PSEUDO_DIR),
    directory=QE_WORK_DIR,
    input_data=QE_INPUT_DATA,
    pseudopotentials=qe_pseudopotentials,
    **QE_KPOINT_KWARGS,
)
print("✅ Quantum ESPRESSO calculator ready")
'''

    if not indent:
        return body
    return ''.join(indent + line if line.strip() else line
                   for line in body.splitlines(keepends=True))


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------
def render_qe_sidebar(saved=None, symbols=None):
    """Draw the Quantum ESPRESSO settings in the sidebar and return them.

    ``symbols`` are the elements of the currently loaded structures; when given,
    the pseudopotential picker is restricted to them and missing ones are
    flagged before the user starts a run.
    """
    import streamlit as st

    s = dict(QE_DEFAULTS)
    s.update(saved or {})

    st.markdown("---")
    st.markdown("### 🧪 Quantum ESPRESSO Settings")
    st.caption(
        "Ab initio DFT via an external `pw.x`. Orders of magnitude slower than an "
        "MLIP — start with a small cell."
    )

    # --- executable & pseudopotentials ------------------------------------
    s['pw_binary'] = st.text_input(
        "Path to `pw.x` binary *",
        value=s['pw_binary'],
        placeholder="/opt/qe-7.4/bin/pw.x",
        help="The pw.x executable, the bin/ folder holding it, or just 'pw.x' if it is on your PATH.",
    )
    _resolved_pw = normalize_pw_binary(s['pw_binary'])
    if _resolved_pw != s['pw_binary']:
        st.caption(f"→ resolved to `{_resolved_pw}`")
    s['pw_binary'] = _resolved_pw
    s['pseudo_dir'] = st.text_input(
        "Pseudopotential directory *",
        value=s['pseudo_dir'],
        placeholder="/opt/qe-7.4/pseudo",
        help="Folder holding the .UPF files (e.g. SSSP, PSLibrary, SG15, GBRV).",
    )

    available = find_pseudopotentials(s['pseudo_dir'])
    if s['pseudo_dir']:
        if available:
            st.success(f"✅ Found pseudopotentials for {len(available)} elements")
        else:
            st.error("❌ No `.UPF` files found in this directory")

    # --- hardware ----------------------------------------------------------
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

    col_a, col_b = st.columns(2)
    if s['use_mpi']:
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
            "k-point pools (-nk)",
            min_value=1, max_value=256, step=1, value=int(s['npool']),
            help="Splits k-points across pools. Must divide the number of MPI ranks.",
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

    with st.expander("⚙️ Advanced parallelisation"):
        s['ndiag'] = st.number_input(
            "Linear-algebra procs (-nd, 0 = auto)",
            min_value=0, max_value=1024, step=1, value=int(s['ndiag']),
        )
        s['extra_pw_flags'] = st.text_input(
            "Extra pw.x flags", value=s['extra_pw_flags'], placeholder="-ntg 2",
        )
        s['work_dir'] = st.text_input(
            "Working directory", value=s['work_dir'],
            help="Where the .pwi/.pwo files and the QE scratch folder are written.",
        )

    st.code(build_qe_command(s), language="bash")

    # --- basis & k-points --------------------------------------------------
    st.markdown("**Plane-wave basis**")

    # SSSP ships recommended cutoffs; using them is the single easiest way to
    # avoid an under-converged calculation.
    suggestion = suggest_cutoffs(s['pseudo_dir'], symbols)
    if suggestion:
        sug_wfc, sug_rho, sug_src = suggestion
        scope = "your structures" if symbols else "the whole library"
        st.info(
            f"📖 `{sug_src}` recommends **ecutwfc {sug_wfc:g} Ry / ecutrho {sug_rho:g} Ry** "
            f"for {scope}."
        )
        if st.button("Use recommended cutoffs", key="qe_apply_cutoffs"):
            # Write straight into the widgets' state, then rerun so the number
            # inputs below pick the new values up.
            st.session_state['qe_ecutwfc'] = float(sug_wfc)
            st.session_state['qe_ecutrho'] = float(sug_rho)
            st.rerun()

    # Seed the widget state once; afterwards session_state is the source of
    # truth (passing both `value` and a stored key makes Streamlit complain).
    st.session_state.setdefault('qe_ecutwfc', float(s['ecutwfc']))
    st.session_state.setdefault('qe_ecutrho', float(s['ecutrho']))

    col_e, col_f = st.columns(2)
    with col_e:
        s['ecutwfc'] = st.number_input(
            "ecutwfc (Ry)", min_value=10.0, max_value=400.0, step=5.0,
            key="qe_ecutwfc",
            help="Wavefunction cutoff. Use the value recommended for your pseudopotentials.",
        )
    with col_f:
        s['ecutrho'] = st.number_input(
            "ecutrho (Ry)", min_value=40.0, max_value=3200.0, step=20.0,
            key="qe_ecutrho",
            help="Charge-density cutoff: ~4x ecutwfc for norm-conserving, 8-12x for US/PAW.",
        )
    if float(s['ecutrho']) < 4 * float(s['ecutwfc']):
        st.warning("⚠️ ecutrho below 4x ecutwfc — fine only for norm-conserving pseudopotentials.")

    st.markdown("**k-points**")
    kmode_labels = {
        'kspacing': "Automatic (k-spacing)",
        'grid': "Explicit Monkhorst-Pack grid",
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
            "k-spacing (1/Å)", min_value=0.01, max_value=1.0, step=0.01,
            value=float(s['kspacing']), format="%.3f",
            help="Smaller = denser grid. 0.25 is a reasonable default, 0.15 for metals.",
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

    # --- electronic structure ---------------------------------------------
    with st.expander("🔬 Electronic structure"):
        occ_labels = {
            'smearing': "Smearing (metals)",
            'fixed': "Fixed (insulators)",
            'tetrahedra_opt': "Tetrahedra (optimised)",
        }
        occ_keys = list(occ_labels)
        s['occupations'] = st.selectbox(
            "Occupations", occ_keys,
            index=occ_keys.index(s['occupations']) if s['occupations'] in occ_keys else 0,
            format_func=lambda k: occ_labels[k],
        )
        if s['occupations'] == 'smearing':
            col_g, col_h = st.columns(2)
            with col_g:
                s['smearing'] = st.selectbox(
                    "Smearing type", SMEARING_CHOICES,
                    index=SMEARING_CHOICES.index(s['smearing'])
                    if s['smearing'] in SMEARING_CHOICES else 0,
                    help="mv = Marzari-Vanderbilt, the usual choice for metals.",
                )
            with col_h:
                s['degauss'] = st.number_input(
                    "degauss (Ry)", min_value=0.0001, max_value=0.5, step=0.005,
                    value=float(s['degauss']), format="%.4f",
                )

        col_i, col_j = st.columns(2)
        with col_i:
            s['nspin'] = 2 if st.checkbox(
                "Spin polarised", value=int(s['nspin']) == 2,
                help="Also switched on automatically if the structure carries magnetic moments.",
            ) else 1
        with col_j:
            if int(s['nspin']) == 2:
                s['starting_magnetization'] = st.number_input(
                    "Starting magnetisation", min_value=-1.0, max_value=1.0, step=0.1,
                    value=float(s['starting_magnetization']),
                )

        col_k, col_l = st.columns(2)
        with col_k:
            s['tot_charge'] = st.number_input(
                "Total charge", min_value=-20.0, max_value=20.0, step=1.0,
                value=float(s['tot_charge']),
            )
        with col_l:
            s['nbnd'] = st.number_input(
                "nbnd (0 = auto)", min_value=0, max_value=100000, step=1,
                value=int(s['nbnd']),
            )

        s['input_dft'] = st.text_input(
            "Override functional (input_dft)", value=s['input_dft'],
            placeholder="leave empty to use the pseudopotential's functional",
        )
        s['vdw_corr'] = st.selectbox(
            "Dispersion correction", VDW_CORR_CHOICES,
            index=VDW_CORR_CHOICES.index(s['vdw_corr'])
            if s['vdw_corr'] in VDW_CORR_CHOICES else 0,
        )
        s['assume_isolated'] = st.selectbox(
            "assume_isolated", ASSUME_ISOLATED_CHOICES,
            index=ASSUME_ISOLATED_CHOICES.index(s['assume_isolated'])
            if s['assume_isolated'] in ASSUME_ISOLATED_CHOICES else 0,
            help="Use for charged or molecular systems in a periodic box.",
        )

    with st.expander("🔁 SCF convergence"):
        col_m, col_n = st.columns(2)
        with col_m:
            s['conv_thr'] = st.number_input(
                "conv_thr (Ry)", min_value=1e-12, max_value=1e-2,
                value=float(s['conv_thr']), format="%.2e",
                help="1e-6 for energies; tighten to 1e-8/1e-10 for phonons.",
            )
            s['mixing_beta'] = st.number_input(
                "mixing_beta", min_value=0.01, max_value=1.0, step=0.05,
                value=float(s['mixing_beta']),
                help="Lower (0.1-0.3) helps hard-to-converge metals and magnets.",
            )
        with col_n:
            s['electron_maxstep'] = st.number_input(
                "electron_maxstep", min_value=10, max_value=5000, step=10,
                value=int(s['electron_maxstep']),
            )
            s['mixing_mode'] = st.selectbox(
                "mixing_mode", MIXING_MODE_CHOICES,
                index=MIXING_MODE_CHOICES.index(s['mixing_mode'])
                if s['mixing_mode'] in MIXING_MODE_CHOICES else 0,
                help="local-TF often helps slabs and inhomogeneous systems.",
            )
        s['diagonalization'] = st.selectbox(
            "diagonalization", DIAGONALIZATION_CHOICES,
            index=DIAGONALIZATION_CHOICES.index(s['diagonalization'])
            if s['diagonalization'] in DIAGONALIZATION_CHOICES else 0,
        )

    # --- pseudopotential picker -------------------------------------------
    if available:
        with st.expander("📦 Pseudopotential assignment"):
            st.caption(
                "The shortest matching `.UPF` is picked automatically. Override any "
                "element here."
            )
            overrides = dict(s['pseudo_overrides'])
            listed = sorted(symbols) if symbols else sorted(available)
            for sym in listed:
                choices = available.get(sym, [])
                if not choices:
                    st.error(f"❌ **{sym}** — no pseudopotential found in this directory")
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

    with st.expander("📝 Extra pw.x parameters"):
        s['extra_input_data'] = st.text_area(
            "One `section.key = value` per line",
            value=s['extra_input_data'],
            placeholder="system.lda_plus_u = .true.\nsystem.Hubbard_U(1) = 4.0\nelectrons.mixing_ndim = 12",
            help="Merged into input_data, overriding anything set above.",
        )

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

    return s


QE_ENV_SETUP = {
    "pip": "pip install ase==3.28.0 pymatgen==2025.10.7 matscipy==1.2.0 phonopy==2.40.0 numpy pandas matplotlib",
    "note": (
        "Quantum ESPRESSO runs as an external program — no torch/MLIP packages are "
        "needed, but pw.x itself must be installed and the pseudopotential directory "
        "must exist on the machine that runs the script."
    ),
}
