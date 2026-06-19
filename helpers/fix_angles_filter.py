"""Cell filter that relaxes lattice-vector *lengths* while keeping all cell
angles fixed.

The usual way to "fix angles" during a variable-cell relaxation is to wrap the
atoms in ``UnitCellFilter`` with a Voigt strain mask ``[1,1,1,0,0,0]`` that
zeroes the shear strain components. That is only correct when the lattice
vectors are aligned with the Cartesian axes (cubic / tetragonal / orthorhombic):
the mask zeroes shear strain *in the Cartesian frame*, which is not the same as
holding the lattice angles fixed. For hexagonal, trigonal, monoclinic and
triclinic cells, independent diagonal strains rotate the relative orientation of
the (non-orthogonal) lattice vectors and the angles drift.

``FixAnglesCellFilter`` instead parametrises the cell by a single scale factor
per lattice vector, keeping each vector's *direction* frozen::

    cell[i] = scale[i] * orig_cell[i]

Because the directions never change, the angles between the vectors are preserved
exactly for any cell, while a, b and c lengths still relax independently. This is
the rigorous version of "Lattice parameters only (fix angles)".
"""

import numpy as np
from ase.filters import UnitCellFilter
from ase.stress import voigt_6_to_full_3x3_stress
from ase.calculators.calculator import PropertyNotImplementedError


class FixAnglesCellFilter(UnitCellFilter):
    """Relax atoms + lattice-vector lengths with all cell angles held fixed.

    Parameters
    ----------
    atoms : Atoms
        The atoms to relax.
    axis_mask : sequence of 3 bool
        Which of the a, b, c lattice vectors may change length. A ``False``
        entry holds that vector's length (and direction) constant.
    scalar_pressure : float
        External (hydrostatic) pressure in eV/Å³, added as a ``P V`` enthalpy
        term, matching ``UnitCellFilter``'s convention.
    cell_factor : float, optional
        Scaling applied to the cell degrees of freedom (defaults to the number
        of atoms, as in ``UnitCellFilter``) so the cell and atomic coordinates
        are optimised on comparable scales.
    """

    def __init__(self, atoms, axis_mask=(True, True, True),
                 scalar_pressure=0.0, cell_factor=None):
        UnitCellFilter.__init__(self, atoms, scalar_pressure=scalar_pressure,
                                cell_factor=cell_factor)
        self.axis_mask = np.array([bool(x) for x in axis_mask])
        self.orig_lens = np.linalg.norm(self.orig_cell, axis=1)

    def _scales(self):
        """Current per-vector scale factors relative to the original cell."""
        cur_lens = np.linalg.norm(self.atoms.cell, axis=1)
        return cur_lens / self.orig_lens

    def _deform(self, s):
        """Cartesian deformation mapping orig -> scaled cell: cell = diag(s) C0.

        Atoms transform as ``r = r0 @ G`` with ``G = C0^{-1} diag(s) C0``.
        """
        C0 = np.array(self.orig_cell)
        return np.linalg.solve(C0, np.diag(s) @ C0)

    def get_positions(self):
        """(natoms+3, 3) array: atomic positions (undeformed frame) then the
        three scale factors on the diagonal of the trailing block."""
        s = self._scales()
        G = self._deform(s)
        natoms = len(self.atoms)
        pos = np.zeros((natoms + 3, 3))
        pos[:natoms] = np.linalg.solve(G.T, self.atoms.positions.T).T
        np.fill_diagonal(pos[natoms:], self.cell_factor * s)
        return pos

    def set_positions(self, new, **kwargs):
        natoms = len(self.atoms)
        s = np.diag(new[natoms:]) / self.cell_factor
        # Hold frozen axes at their current scale instead of whatever the
        # optimiser proposed for them.
        s = np.where(self.axis_mask, s, self._scales())
        C0 = np.array(self.orig_cell)
        newcell = np.diag(s) @ C0
        G = self._deform(s)
        self.atoms.set_cell(newcell, scale_atoms=False)
        self.atoms.set_positions(new[:natoms] @ G, **kwargs)

    def get_forces(self, **kwargs):
        s = self._scales()
        G = self._deform(s)
        atoms_forces = self.atoms.get_forces(**kwargs) @ G.T

        stress = voigt_6_to_full_3x3_stress(self.atoms.get_stress(**kwargs))
        C = np.array(self.atoms.cell)
        volume = self.atoms.get_volume()
        # dE/ds_i = (V / s_i) * [ (C @ sigma @ C^{-1})_ii + P ];  force = -dE/ds_i
        CsCinv = C @ stress @ np.linalg.inv(C)
        g = -(volume / s) * (np.diag(CsCinv) + self.scalar_pressure)
        g *= self.axis_mask

        natoms = len(self.atoms)
        forces = np.zeros((natoms + 3, 3))
        forces[:natoms] = atoms_forces
        np.fill_diagonal(forces[natoms:], g / self.cell_factor)
        return forces

    def get_stress(self):
        raise PropertyNotImplementedError
