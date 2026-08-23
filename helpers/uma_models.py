"""UMA (Meta FAIR Chemistry) support.

UMA is the one model family here that cannot be run from inside the web
interface. Two reasons, both hard:

  * the weights live in **gated** Hugging Face repos (facebook/UMA for the
    UMA checkpoints, facebook/OMol25 / OC25 / ODAC25 for the specialists), so
    they only download once a personal access token that has been granted
    access to that particular repo is visible to `huggingface_hub`. Asking every visitor of a shared app to paste
    a personal token into a server they do not control is not something this
    app should encourage, and the token would have to be held in server-side
    session state for the whole run.
  * `fairchem-core` pins `torch~=2.13` — a different torch from every other
    model family here, which are all on 2.8 — plus a narrow numpy window, so
    it wants its own virtual environment rather than the one the GUI runs in.

So UMA is offered as a *script-only* family: the sidebar collects the token,
the task and the inference preset, and those end up in the standalone Python
script the user downloads and runs in their own environment. Everything else in
this module mirrors `helpers.quantum_espresso`, which solves the same problem
for pw.x.
"""

import os

import streamlit as st

# Set to "1" by the hosted deployment. The token is only ever offered for
# saving when this is false: on a shared server "remember my token" would mean
# writing a visitor's credential to a disk they do not control.
ONLINE_MODE = os.environ.get("MLIP_ONLINE_MODE", "0") == "1"


UMA_FAMILY_NAME = "UMA (Meta FAIR) — script only"
UMA_MODEL_PREFIX = "uma:"

# ---------------------------------------------------------------------------
# Checkpoints
#
# Names and ids come from fairchem's own registry
# (fairchem/core/calculate/pretrained_models.json). The four uma-* checkpoints
# are the multi-task universal models — they need a task name. Everything below
# them is a single-task specialist: fairchem picks the task itself, so no task
# name is sent. Those live in a different gated repo — see CHECKPOINT_REPOS.
#
# uma-s-1 is deliberately absent: it has a known extensivity bug and upstream
# archived it in favour of 1.1 / 1.2.1.
# ---------------------------------------------------------------------------
UMA_MODELS = {
    "UMA-S-1.2.1 (small) - Universal, 6 tasks, latest ⭐": "uma:uma-s-1p2p1",
    "UMA-S-1.2 (small) - Universal, 6 tasks": "uma:uma-s-1p2",
    "UMA-S-1.1 (small) - Universal, 6 tasks, stable": "uma:uma-s-1p1",
    "UMA-M-1.1 (medium) - Universal, 6 tasks, most accurate": "uma:uma-m-1p1",
    # --- single-task specialists (separate gated repos, see CHECKPOINT_REPOS) ---
    "eSEN-S-Conserving - Molecules only [OMol25, ωB97M-V]": "uma:esen-sm-conserving-all-omol",
    "eSEN-S-Direct - Molecules only, fast [OMol25]": "uma:esen-sm-direct-all-omol",
    "eSEN-M-Direct - Molecules only, fast [OMol25]": "uma:esen-md-direct-all-omol",
    "AllScAIP-M-Conserving - Molecules only [OMol25]": "uma:allscaip-md-conserving-all-omol",
    "AllScAIP-M-Direct - Molecules only [OMol25]": "uma:allscaip-md-direct-all-omol",
    "eSEN-S-Conserving - Electrolyte interfaces only [OC25]": "uma:esen-sm-conserving-all-oc25",
    "eSEN-M-Direct - Electrolyte interfaces only [OC25]": "uma:esen-md-direct-all-oc25",
    "eSEN-S (filtered) - MOFs / direct air capture only [ODAC25]": "uma:esen-sm-filtered-odac25",
    "eSEN-S (full) - MOFs / direct air capture only [ODAC25]": "uma:esen-sm-full-odac25",
}

# Checkpoints that expose all six UMA tasks and therefore need one to be picked.
MULTI_TASK_CHECKPOINTS = (
    "uma-s-1p2p1", "uma-s-1p2", "uma-s-1p1", "uma-m-1p1",
)

# Which gated Hugging Face repo each checkpoint actually lives in. Only the
# four UMA models are in facebook/UMA; the specialists sit in the dataset repo
# they were trained on, and each is gated separately — accepting the UMA
# licence does NOT grant access to the others. Getting this wrong sends the
# user to the wrong page after a 403, so it is kept per checkpoint rather than
# hard-coded. Mirrors repo_id in fairchem/core/calculate/pretrained_models.json.
CHECKPOINT_REPOS = {
    "uma-s-1p2p1": "facebook/UMA",
    "uma-s-1p2": "facebook/UMA",
    "uma-s-1p1": "facebook/UMA",
    "uma-m-1p1": "facebook/UMA",
    "esen-md-direct-all-omol": "facebook/OMol25",
    "esen-sm-conserving-all-omol": "facebook/OMol25",
    "esen-sm-direct-all-omol": "facebook/OMol25",
    "allscaip-md-conserving-all-omol": "facebook/OMol25",
    "allscaip-md-direct-all-omol": "facebook/OMol25",
    "esen-sm-conserving-all-oc25": "facebook/OC25",
    "esen-md-direct-all-oc25": "facebook/OC25",
    "esen-sm-filtered-odac25": "facebook/ODAC25",
    "esen-sm-full-odac25": "facebook/ODAC25",
}


def uma_repo_id(model_size):
    """The gated HF repo holding this checkpoint."""
    return CHECKPOINT_REPOS.get(uma_checkpoint_name(model_size), "facebook/UMA")

# fairchem's UMATask enum, in the order that makes sense for this app's users
# (crystals first — most of the calculation types here are periodic).
UMA_TASKS = ["omat", "omc", "odac", "omol", "oc20", "oc25"]

UMA_TASK_LABELS = {
    "omat": "omat — inorganic materials / bulk crystals (OMat24, PBE)",
    "omc":  "omc — molecular crystals (OMC25, PBE-D3)",
    "odac": "odac — MOFs & direct air capture (ODAC23, PBE-D3)",
    "omol": "omol — molecules, polymers, clusters (OMol25, ωB97M-V)",
    "oc20": "oc20 — heterogeneous catalysis, adsorbate/slab (OC20, RPBE)",
    "oc25": "oc25 — electrolyte / solid-liquid interfaces (OC25)",
}

# "default" and "turbo" both take the merge_mole + compile fast path; "turbo"
# also turns on TF32. "batch" keeps MOLE unmerged, which is what heterogeneous
# input needs — a pose scan or an EOS sweep that changes composition or charge.
UMA_INFERENCE_PRESETS = ["default", "turbo", "batch"]

UMA_INFERENCE_LABELS = {
    "default": "default — merged MOLE + compile (recommended)",
    "turbo":   "turbo — same, plus TF32 (fastest, slightly less accurate)",
    "batch":   "batch — no compile, lowest RAM & fastest start-up",
}

UMA_DEFAULTS = {
    "model_id": "uma-s-1p2p1",
    "repo_id": "facebook/UMA",
    "task": "omat",
    "inference_settings": "default",
    "seed": 41,
    "charge": 0,
    "spin": 1,
    "hf_token": "",
    "embed_token": True,
    "device": "cpu",
}

# Always safe to write to default_settings.json.
PERSISTED_KEYS = ("task", "inference_settings", "seed", "charge", "spin")

# The token is a credential, so it is persisted only when the user asks for it
# *and* this is a local install (see ONLINE_MODE). "remember_token" itself is
# saved so the checkbox comes back ticked; clearing it drops the stored token.
TOKEN_KEYS = ("hf_token", "remember_token")

UMA_ENV_SETUP = {
    "pip": (
        "pip install fairchem-core==2.22.0 \"huggingface_hub[cli]>=0.34.0\" "
        "ase==3.28.0 pymatgen==2025.10.7 matscipy==1.2.0 phonopy==2.41.0 "
        "numpy pandas matplotlib"
    ),
    "note": (
        "fairchem-core needs Python 3.11–3.14 and pins torch~=2.13 (every "
        "other family here is on torch 2.8), so it wants "
        "its own environment — locally, `pip install -r requirements-uma.txt` "
        "does the whole thing. The weights are gated: request access to the "
        "Hugging Face repo named in the sidebar for the checkpoint you picked, "
        "then either run `hf auth login` or export HF_TOKEN=<your token>."
    ),
}


# ---------------------------------------------------------------------------
# Active-settings cache (same contract as helpers.quantum_espresso)
# ---------------------------------------------------------------------------
_ACTIVE_UMA_SETTINGS = None


def set_active_uma_settings(settings):
    """Remember the sidebar's UMA settings so every script generator can reach
    them without threading an extra argument through a dozen call sites."""
    global _ACTIVE_UMA_SETTINGS
    _ACTIVE_UMA_SETTINGS = dict(settings) if settings else None


def get_active_uma_settings(settings=None):
    """Return the settings picked in the sidebar, filled in from the defaults."""
    merged = dict(UMA_DEFAULTS)
    if _ACTIVE_UMA_SETTINGS:
        merged.update(_ACTIVE_UMA_SETTINGS)
    if settings:
        merged.update(settings)
    return merged


def is_uma_model(selected_model_key=None, model_size=None):
    """True when the user picked a UMA / fairchem checkpoint."""
    if isinstance(model_size, str) and model_size.startswith(UMA_MODEL_PREFIX):
        return True
    if isinstance(selected_model_key, str) and selected_model_key in UMA_MODELS:
        return True
    return False


def uma_checkpoint_name(model_size):
    """Strip the "uma:" family prefix off the stored value."""
    if isinstance(model_size, str) and model_size.startswith(UMA_MODEL_PREFIX):
        return model_size[len(UMA_MODEL_PREFIX):]
    return model_size


def uma_is_multi_task(model_size):
    return uma_checkpoint_name(model_size) in MULTI_TASK_CHECKPOINTS


# ---------------------------------------------------------------------------
# Sidebar panel
# ---------------------------------------------------------------------------
def setup_uma_ui(model_size, device="cpu", default_settings=None,
                 save_settings_function=None):
    """Draw the UMA options in the sidebar and return the settings dict.

    `model_size` is the value from MODEL_FAMILIES (with the "uma:" prefix),
    `device` the one already chosen by the generic Compute Device radio.
    """
    saved = dict(UMA_DEFAULTS)
    saved.update((default_settings or {}).get("uma_settings", {}) or {})

    checkpoint = uma_checkpoint_name(model_size)
    multi_task = uma_is_multi_task(model_size)
    repo_id = uma_repo_id(model_size)

    st.markdown("---")
    st.markdown("### 🧬 UMA (Meta FAIR) Configuration")

    st.caption("Script only — use the *Generate Python Script* buttons.")

    # --- access token ------------------------------------------------------
    # Each repo is gated separately: accepting the UMA licence does not grant
    # access to OMol25 / OC25 / ODAC25, so the user is pointed at the one that
    # actually holds the checkpoint they picked.
    st.caption(
        f"Gated repo: **[{repo_id}](https://huggingface.co/{repo_id})** — "
        f"accept the licence there, or the download returns 403."
    )

    # A token saved on a previous run pre-fills the box, so a local install
    # only has to be told once. Never on the hosted app: a stored credential
    # there would belong to a visitor but live on someone else's disk.
    saved_token = "" if ONLINE_MODE else (saved.get("hf_token") or "")

    hf_token = st.text_input(
        "Hugging Face access token",
        value=saved_token,
        type="password",
        key="uma_hf_token",
        placeholder="hf_...",
        help=(
            f"{repo_id} is a gated repository. Accept its licence at "
            f"huggingface.co/{repo_id}, then create a token at "
            "huggingface.co/settings/tokens with 'Read access to contents of "
            "all public gated repos you can access'. One such token covers "
            "every repo you have been granted."
        ),
    )

    remember_token = False
    if not ONLINE_MODE:
        remember_token = st.checkbox(
            "Remember this token",
            value=bool(saved.get("remember_token", False)),
            key="uma_remember_token",
            help=(
                "Saves the token to default_settings.json next to the app, so "
                "you do not have to paste it every session. It is stored in "
                "plain text, so only do this on a machine you control — and "
                "note default_settings.json is git-ignored so it is not "
                "committed by accident. Untick to forget it again."
            ),
        )
        if remember_token and hf_token:
            st.caption("💾 Saved to `default_settings.json` in plain text.")

    embed_token = st.checkbox(
        "Write the token into the generated script",
        value=bool(saved.get("embed_token", True)),
        key="uma_embed_token",
        help=(
            "On: the script is self-contained and runs anywhere. Off: the "
            "script reads HF_TOKEN from the environment instead, which keeps "
            "the credential out of a file you might share."
        ),
    )

    if hf_token and embed_token:
        # Kept deliberately: the downloaded file really does carry a live
        # credential, and that is not obvious from the download button.
        st.caption("🔐 Script will contain your token in plain text — don't share it.")
    elif not hf_token:
        st.caption("Falls back to `HF_TOKEN` in your shell, or `hf auth login`.")

    # --- task --------------------------------------------------------------
    if multi_task:
        task_default = saved.get("task", "omat")
        task_index = UMA_TASKS.index(task_default) if task_default in UMA_TASKS else 0
        task = st.selectbox(
            "Task (training domain)",
            UMA_TASKS,
            index=task_index,
            format_func=lambda t: UMA_TASK_LABELS.get(t, t),
            key="uma_task",
            help=(
                "One UMA checkpoint holds six experts. The task selects which "
                "one answers, and it also fixes the reference DFT level — "
                "energies from two different tasks are not comparable."
            ),
        )
    else:
        task = None
        st.info(
            f"ℹ️ `{checkpoint}` is a single-task checkpoint — fairchem picks "
            "its task automatically, so there is nothing to choose here."
        )

    # --- inference preset --------------------------------------------------
    preset_default = saved.get("inference_settings", "default")
    preset_index = (UMA_INFERENCE_PRESETS.index(preset_default)
                    if preset_default in UMA_INFERENCE_PRESETS else 0)
    inference_settings = st.selectbox(
        "Inference preset",
        UMA_INFERENCE_PRESETS,
        index=preset_index,
        format_func=lambda p: UMA_INFERENCE_LABELS.get(p, p),
        key="uma_inference",
        help=(
            "'default' and 'turbo' merge the MOLE experts and compile the "
            "model, which is fast but assumes composition, charge and spin stay "
            "put; they fall back automatically when that breaks. Pick 'batch' "
            "for a scan that deliberately varies them.\n\n"
            "Start-up cost differs a lot. Measured on a 64-atom cell, RTX 3060: "
            "'default' spends ~35 s compiling then runs at ~75 ms/call; 'batch' "
            "starts in ~5 s but runs at ~167 ms/call. So 'batch' wins for a "
            "handful of single-points, 'default' for anything long (MD, NEB, "
            "phonons). 'batch' also needs the least RAM, which matters on a "
            "machine with under ~12 GB."
        ),
    )
    if inference_settings == "turbo" and device != "cuda":
        st.caption(
            "💡 'turbo' only differs from 'default' by enabling TF32, which is "
            "a GPU feature — on CPU the two are the same."
        )

    # --- molecular electronic state ---------------------------------------
    charge = int(saved.get("charge", 0))
    spin = int(saved.get("spin", 1))
    if task == "omol":
        st.markdown("**Electronic state** (the `omol` task is molecular)")
        col_c, col_s = st.columns(2)
        with col_c:
            charge = st.number_input(
                "Total charge",
                min_value=-100, max_value=100, value=charge, step=1,
                key="uma_charge",
                help="Net charge of the whole system. Written to `atoms.info['charge']`.",
            )
        with col_s:
            spin = st.number_input(
                "Spin multiplicity (2S+1)",
                min_value=1, max_value=100, value=max(1, spin), step=1,
                key="uma_spin",
                help="Unpaired electrons + 1. Written to `atoms.info['spin']`.",
            )
        st.caption(
            "Structure files carry neither value, so the script injects them "
            "into every single-point call."
        )

    seed = int(st.number_input(
        "Random seed",
        min_value=0, max_value=999999, value=int(saved.get("seed", 41)), step=1,
        key="uma_seed",
        help="Passed to get_predict_unit() so a rerun reproduces the same numbers.",
    ))

    settings = {
        "model_id": checkpoint,
        "repo_id": repo_id,
        "task": task,
        "inference_settings": inference_settings,
        "seed": seed,
        "charge": int(charge),
        "spin": int(spin),
        "hf_token": hf_token or "",
        "embed_token": bool(embed_token),
        "remember_token": bool(remember_token),
        "device": device,
    }

    st.caption(
        "📄 [UMA paper](https://arxiv.org/abs/2506.23971) · "
        "[fairchem docs](https://fair-chem.github.io/) · "
        f"[{repo_id} weights](https://huggingface.co/{repo_id})"
    )

    if default_settings is not None:
        # This only updates the in-memory defaults; without a
        # save_settings_function nothing reaches disk until the user presses
        # "Save as Default".
        to_persist = {
            k: settings[k] for k in PERSISTED_KEYS if settings.get(k) is not None
        }
        # The token rides along only on a local install and only on request.
        # Unticking "Remember this token" must actively drop a previously
        # stored one, so this writes remember_token either way.
        if not ONLINE_MODE:
            to_persist["remember_token"] = bool(remember_token)
            if remember_token and settings["hf_token"]:
                to_persist["hf_token"] = settings["hf_token"]
        default_settings["uma_settings"] = to_persist
        if save_settings_function is not None:
            try:
                save_settings_function(default_settings)
            except Exception:
                pass

    set_active_uma_settings(settings)
    return settings


# ---------------------------------------------------------------------------
# Standalone-script calculator setup
# ---------------------------------------------------------------------------
def generate_uma_calculator_code(settings=None, indent="    "):
    """Return the `calculator = ...` block for a generated standalone script."""
    s = get_active_uma_settings(settings)

    model_id = s.get("model_id") or UMA_DEFAULTS["model_id"]
    # Resolved from the checkpoint rather than trusted from the settings, so a
    # stale cached repo can never point the user at the wrong licence page.
    repo_id = CHECKPOINT_REPOS.get(model_id, s.get("repo_id") or "facebook/UMA")
    task = s.get("task")
    device = s.get("device", "cpu") or "cpu"
    preset = s.get("inference_settings", "default")
    seed = int(s.get("seed", 41))
    charge = int(s.get("charge", 0))
    spin = int(s.get("spin", 1))
    token = (s.get("hf_token") or "") if s.get("embed_token", True) else ""

    task_repr = repr(task) if task else "None"
    task_disp = task if task else "auto (single-task checkpoint)"
    inject_charge_spin = (task == "omol")

    lines = [
        f'device = "{device}"',
        'print("🔧 Initializing UMA (Meta FAIR / fairchem) calculator...")',
        f'print("🎯 Checkpoint:       {model_id}")',
        f'print("🧭 Task:             {task_disp}")',
        f'print("⚙️  Inference preset: {preset}")',
        f'print("🤗 Hugging Face repo: {repo_id} (gated)")',
        '',
    ]

    # The "default" and "turbo" presets torch.compile the model. Inductor forks
    # one compile worker per core, each of which imports torch again, and that
    # transient spike -- not the GPU, and not the size of the structure -- is
    # what gets these runs OOM-killed on small-RAM machines. Measured on a
    # 64-atom cell: 10.1 GB peak RSS with the default worker count, 5.2 GB with
    # one worker, for the same steady-state speed (75 ms/call) and a *shorter*
    # compile (34 s vs 86 s). So capping is free here, not a trade.
    if preset in ("default", "turbo"):
        lines += [
            '# torch.compile forks one inductor worker per core, each re-importing',
            '# torch. On a small-RAM box that spike, not the GPU, is what gets the',
            '# run killed -- measured 10.1 GB peak vs 5.2 GB with a single worker,',
            '# at the same per-call speed and half the compile time. Set before',
            '# torch is imported, because inductor reads it at import time.',
            'import os as _uma_os',
            'if not _uma_os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"):',
            '    try:',
            '        _uma_ram_gb = (_uma_os.sysconf("SC_PAGE_SIZE")',
            '                       * _uma_os.sysconf("SC_PHYS_PAGES")) / 1024 ** 3',
            '    except (ValueError, AttributeError, OSError):',
            '        _uma_ram_gb = None',
            '    if _uma_ram_gb is not None and _uma_ram_gb < 12:',
            '        _uma_os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"',
            '        print(f"🧠 RAM: {_uma_ram_gb:.1f} GB — capping torch.compile to one")',
            '        print("   worker (halves peak RAM, no cost to speed). Under WSL2,")',
            '        print("   raise this in %UserProfile%\\\\.wslconfig: memory=12GB")',
            f'        print("   Or pick the \'batch\' preset to skip compiling entirely.")',
            '',
        ]

    lines += [
        '# CUDA is checked *before* the 2.2 GB checkpoint downloads, because the',
        '# most common failure here is not the model at all: torch 2.13 ships a',
        '# CUDA 13 wheel by default, and CUDA 13 needs an r580+ driver. On an',
        '# older driver torch silently reports no GPU and fairchem then raises',
        '# "cannot set cpu=false and no cuda available", which points at the',
        '# wrong thing. Say plainly what is wrong and how to fix it.',
        'if device == "cuda":',
        '    import torch as _uma_torch',
        '    if not _uma_torch.cuda.is_available():',
        '        _bt = _uma_torch.version.cuda',
        '        print("⚠️  CUDA was requested but torch cannot see a GPU.")',
        '        print(f"   torch {_uma_torch.__version__} (built for CUDA {_bt})")',
        '        try:',
        '            _uma_torch.zeros(1).cuda()',
        '        except Exception as _ce:',
        '            _msg = str(_ce)',
        '            print(f"   reason: {_msg.splitlines()[0][:160]}")',
        '            if "driver" in _msg and "old" in _msg:',
        '                print("   → Your NVIDIA driver predates the CUDA build of torch.")',
        '                print("     Either update the driver (on WSL2 that means the")',
        '                print("     Windows driver), or install a torch built for your")',
        '                print("     CUDA, staying inside fairchem\'s window:")',
        '                print("       pip install \\"torch>=2.13,<2.14\\" \\\\")',
        '                print("         --index-url https://download.pytorch.org/whl/cu126")',
        '            elif _bt is None:',
        '                print("   → This is a CPU-only torch build. Reinstall from a")',
        '                print("     CUDA index, e.g. .../whl/cu126 (see requirements-uma.txt).")',
        '        print("   Continuing on CPU — correct, just slower.")',
        '        device = "cpu"',
        '',
        f'# {repo_id} is a gated Hugging Face repo: nothing downloads until a',
        '# token that has been granted access to *that* repo is visible to',
        '# huggingface_hub. Access is granted per repo, not once for all of them.',
        'import os as _uma_os',
        f'_UMA_HF_TOKEN = {token!r}',
        'if not _UMA_HF_TOKEN:',
        '    _UMA_HF_TOKEN = (_uma_os.environ.get("HF_TOKEN")',
        '                     or _uma_os.environ.get("HUGGING_FACE_HUB_TOKEN") or "")',
        'if _UMA_HF_TOKEN:',
        '    _uma_os.environ["HF_TOKEN"] = _UMA_HF_TOKEN',
        '    _uma_os.environ["HUGGING_FACE_HUB_TOKEN"] = _UMA_HF_TOKEN',
        '    print("🔑 Hugging Face token: set")',
        'else:',
        '    print("🔑 Hugging Face token: not set in this script")',
        '    print("   Falling back to a previous `hf auth login`. If the download")',
        '    print("   fails with 401/403: accept the licence at")',
        f'    print("   https://huggingface.co/{repo_id}, create a read token at")',
        '    print("   https://huggingface.co/settings/tokens, then export HF_TOKEN=...")',
        '',
        'try:',
        '    from fairchem.core import pretrained_mlip, FAIRChemCalculator',
        'except ImportError as _uma_ie:',
        '    print(f"❌ fairchem-core is not installed: {_uma_ie}")',
        '    print("   pip install fairchem-core  (needs Python 3.11-3.14, torch 2.13)")',
        '    raise',
        '',
        'def _uma_build(_device):',
        '    _predictor = pretrained_mlip.get_predict_unit(',
        f'        "{model_id}",',
        '        device=_device,',
        f'        inference_settings="{preset}",',
        f'        seed={seed},',
        '    )',
        f'    return FAIRChemCalculator(_predictor, task_name={task_repr})',
        '',
        'try:',
        '    calculator = _uma_build(device)',
        f'    print(f"✅ UMA {model_id} initialized successfully on {{device}}")',
        'except Exception as e:',
        '    print(f"❌ UMA initialization failed on {device}: {e}")',
        '    if any(_c in str(e) for _c in ("401", "403", "gated", "Unauthorized")):',
        f'        print("   This looks like a Hugging Face access problem, not a")',
        f'        print("   model problem. Accept the licence for {repo_id} at")',
        f'        print("   https://huggingface.co/{repo_id} and check your token")',
        '        print("   grants \'Read access to public gated repos\'.")',
        '    if device == "cuda":',
        '        print("⚠️ GPU initialization failed, falling back to CPU...")',
        '        try:',
        '            calculator = _uma_build("cpu")',
        '            device = "cpu"',
        '            print("✅ UMA initialized successfully on CPU (fallback)")',
        '        except Exception as cpu_error:',
        '            print(f"❌ CPU fallback also failed: {cpu_error}")',
        '            raise cpu_error',
        '    else:',
        '        raise e',
    ]

    if inject_charge_spin:
        lines += [
            '',
            '# The omol task is molecular: fairchem reads the total charge and the',
            '# spin multiplicity from atoms.info, and a structure file carries',
            '# neither. They are injected on every call so the whole run sees one',
            '# consistent electronic state instead of silently defaulting to 0 / 1.',
            'from ase.calculators.calculator import all_changes as _uma_all_changes',
            f'_UMA_CHARGE = {charge}',
            f'_UMA_SPIN = {spin}',
            '_uma_orig_calculate = calculator.calculate',
            'def _uma_calculate(atoms=None, properties=None,',
            '                   system_changes=_uma_all_changes,',
            '                   _o=_uma_orig_calculate,',
            '                   _c=_UMA_CHARGE, _s=_UMA_SPIN):',
            '    if atoms is not None:',
            '        atoms.info["charge"] = _c',
            '        atoms.info["spin"] = _s',
            '    return _o(atoms=atoms, properties=properties,',
            '              system_changes=system_changes)',
            'calculator.calculate = _uma_calculate',
            'print(f"⚡ UMA omol charge/spin injection enabled '
            '(charge={_UMA_CHARGE}, spin={_UMA_SPIN})")',
        ]

    # Trailing newline on purpose: some templates splice this block straight in
    # front of the next statement (the NEB script does), so the last line has to
    # be terminated here rather than by the caller.
    return "\n".join(indent + line if line else "" for line in lines) + "\n"
