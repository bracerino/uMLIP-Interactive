"""NPT pre-equilibration ahead of an NVE / NVT molecular-dynamics run.

The MD panel stores the stage settings in ``md_params['pre_npt']``:

    enabled            bool
    ensemble           one of the NPT ensembles of the MD panel
    n_steps            length of the NPT stage (same timestep / temperature)
    pressure_coupling_type, target_pressure_gpa, taup, bulk_modulus, taut,
    friction, rng_seed, mtk_mask_*, berendsen_mask_*
                       barostat / thermostat settings, same meaning as for an
                       NPT production run
    start_from         "last"    - continue from the last NPT frame, velocities kept
                       "average" - start from the time-averaged NPT structure
    average_fraction   tail fraction of the NPT run used for the average

The helpers are kept as source (PRE_NPT_SRC) and embedded in the generated MD
script, like the other shared-source helpers.
"""

PRE_NPT_SRC = r'''
import numpy as _pre_np

PRE_NPT_DIR = "md_results/npt_preequilibration"
PRE_NPT_STAGE_KEYS = (
    "ensemble", "n_steps", "pressure_coupling_type", "target_pressure_gpa",
    "taup", "bulk_modulus", "taut", "friction", "rng_seed",
    "mtk_mask_a", "mtk_mask_b", "mtk_mask_c",
    "berendsen_mask_a", "berendsen_mask_b", "berendsen_mask_c",
)


def pre_npt_enabled(md_params):
    pre = md_params.get("pre_npt") or {}
    return bool(pre.get("enabled")) and "NPT" not in md_params.get("ensemble", "")


def pre_npt_stage_params(md_params):
    """md_params of the NPT stage: the production settings (temperature,
    timestep, intervals, ...) with the NPT-stage settings on top."""
    pre = md_params["pre_npt"]
    stage = {k: v for k, v in md_params.items() if k != "pre_npt"}
    for key in PRE_NPT_STAGE_KEYS:
        if key in pre:
            stage[key] = pre[key]
    # The panel marks Melchionna as deprecated in its label.
    stage["ensemble"] = stage["ensemble"].replace("⚠️", "").strip()
    return stage


class PreNPTAverager:
    """Mean cell and mean fractional coordinates over the end of an NPT run.

    Fractional coordinates are unwrapped frame to frame, so an atom crossing a
    cell face is not averaged with its periodic image. The average is
    meaningful for solids (atoms vibrating about lattice sites), not for
    liquids or diffusing atoms.
    """

    def __init__(self, atoms, n_steps, fraction, interval):
        self.atoms = atoms
        self.interval = max(1, int(interval))
        fraction = min(max(float(fraction), 0.0), 1.0)
        self.first_step = int(n_steps) - int(round(int(n_steps) * fraction))
        self.step = 0
        self.cells, self.frac = [], []

    def __call__(self):
        self.step += 1
        if self.step <= self.first_step or self.step % self.interval:
            return
        frac = self.atoms.get_scaled_positions(wrap=False)
        if self.frac:
            frac = frac - _pre_np.round(frac - self.frac[-1])
        self.cells.append(self.atoms.cell.array.copy())
        self.frac.append(frac)

    def averaged_atoms(self):
        """The averaged structure (no velocities), or None without frames."""
        if not self.frac:
            return None
        avg = self.atoms.copy()
        avg.set_cell(_pre_np.mean(self.cells, axis=0), scale_atoms=False)
        avg.set_scaled_positions(_pre_np.mean(self.frac, axis=0))
        avg.wrap()
        avg.set_momenta(_pre_np.zeros((len(avg), 3)))
        return avg
'''
