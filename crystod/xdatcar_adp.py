"""
XDATCAR -> ADP (Anisotropic Displacement Parameters) workflow for crystod.

Computes time-averaged atomic positions and symmetry-constrained ADPs (U_ij)
from a molecular-dynamics XDATCAR trajectory and writes them as a CIF file.
Based on script/xdatcar_to_adp.py by Sato (Mochizuki group).
"""

from __future__ import annotations

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawDescriptionHelpFormatter,
    RawTextHelpFormatter,
)
from collections import Counter, defaultdict

import numpy as np
import spglib
from numpy.typing import NDArray

from .runtime_compat import SymmetryDatasetAdapter


class MyHelpFormatter(
    RawTextHelpFormatter,
    RawDescriptionHelpFormatter,
    ArgumentDefaultsHelpFormatter,
):
    pass


desc = """
Compute anisotropic displacement parameters (ADPs) from an MD trajectory
(XDATCAR) and write the time-averaged structure with symmetry-constrained
U_ij tensors as a CIF file.

# Command Example:
crystod-md --adp --dim 4 4 4 --start-step 1000 --xdatcar XDATCAR --output ADP.cif
"""

# U-component order used throughout: [U11, U22, U33, U12, U13, U23], the CIF
# components U^ij = <dx_i dx_j> / (|a*_i| |a*_j|) (dx fractional; Trueblood et al. 1996)
U_INDICES = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
U_NAMES = ["U11", "U22", "U33", "U12", "U13", "U23"]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=desc, formatter_class=MyHelpFormatter)
    parser.add_argument(
        "--dim",
        required=True,
        type=str,
        help='MD supercell dimension relative to the unit cell, e.g. "4 4 4" (diagonal only).',
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="First MD step used in the analysis (earlier steps are discarded as equilibration).",
    )
    parser.add_argument(
        "--xdatcar",
        type=str,
        default="XDATCAR",
        help="Input XDATCAR path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ADP.cif",
        help="Output CIF path.",
    )
    parser.add_argument(
        "--symprec",
        "--tolerance",
        dest="symprec",
        type=float,
        default=0.1,
        help="Symmetry tolerance (Angstrom) for spglib on the time-averaged structure;\n"
        "the same Cartesian distance selects the site-symmetry operations and maps\n"
        "every site onto its asymmetric-unit representative.",
    )
    parser.add_argument(
        "--grouping-tolerance",
        type=float,
        default=0.1,
        help="Tolerance for grouping supercell atoms into unit-cell sites.",
    )
    return parser


# =============================================================================
# XDATCAR reader (handles both single-header and repeated-header/NpT formats)
# =============================================================================

def _read_header(lines: list[str], start: int) -> tuple[NDArray[np.float64], list[str], int, int]:
    scale = float(lines[start + 1].split()[0])
    lattice = scale * np.array(
        [lines[start + 2 + k].split()[:3] for k in range(3)], dtype=float
    )
    symbols = lines[start + 5].split()
    counts = [int(value) for value in lines[start + 6].split()]
    chem_formula = [symbol for symbol, count in zip(symbols, counts) for _ in range(count)]
    return lattice, chem_formula, sum(counts), start + 7


def read_xdatcar(path: str) -> tuple[list[str], list[NDArray[np.float64]], NDArray[np.float64]]:
    """Read a VASP ``XDATCAR`` trajectory into arrays.

    This is the input step of ``crystod-md --adp``. Both layouts written by
    VASP are handled: the fixed-cell one, a single header followed by
    ``Direct configuration=`` blocks, and the variable-cell (NpT) one, in
    which every configuration repeats the header. The scaling factor of the
    header is folded into the lattice. A truncated trailing frame (a run
    that is still writing) is dropped silently.

    Args:
        path: Path of the ``XDATCAR`` file.

    Returns:
        A tuple ``(symbols, lattices, frames)``:

        - ``symbols``: the element symbol of every atom, in file order (one
          entry per atom, so ``len(symbols) == n_atoms``);
        - ``lattices``: one ``(3, 3)`` lattice matrix per frame, rows being
          the lattice vectors in Angstrom; all identical for a fixed-cell run;
        - ``frames``: the fractional coordinates as written by VASP (wrapped
          into the cell, not unwrapped), shape ``(n_frames, n_atoms, 3)``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: The file holds no complete configuration.

    Example:
        >>> from crystod import md
        >>> symbols, lattices, frames = md.read_xdatcar("XDATCAR")
        >>> frames.shape          # example/39_xdatcar2adp: ScF3, 4x4x4 cell, NpT
        (4001, 256, 3)
        >>> symbols[0], symbols[-1], lattices[0].shape
        ('Sc', 'F', (3, 3))
    """
    with open(path) as fp:
        lines = fp.read().splitlines()

    lattice, chem_formula, n_atoms, index = _read_header(lines, 0)
    lattices: list[NDArray[np.float64]] = []
    frames: list[NDArray[np.float64]] = []
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.lower().startswith("direct"):
            block = lines[index + 1 : index + 1 + n_atoms]
            try:
                coords = np.array(" ".join(block).split(), dtype=float).reshape(n_atoms, -1)[:, :3]
            except ValueError:
                # truncated or corrupt trailing frame
                break
            frames.append(coords)
            lattices.append(lattice)
            index += 1 + n_atoms
        else:
            # repeated header (variable-cell / NpT trajectory)
            if index + 7 > len(lines):
                break
            lattice, chem_formula, n_atoms, index = _read_header(lines, index)

    if not frames:
        raise ValueError(f"No configurations found in '{path}'.")
    return chem_formula, lattices, np.array(frames)


# =============================================================================
# ADP symmetry-constraint helpers
# =============================================================================

# the 27 lattice translations {-1, 0, 1}^3, for minimum-image distances
_IMAGE_SHIFTS = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float
)


def _min_image_distance(diff: NDArray[np.float64], lattice: NDArray[np.float64]) -> float:
    """Cartesian length (Angstrom) of the shortest lattice image of a
    fractional difference vector; ``lattice`` rows are the lattice vectors."""
    diff = np.asarray(diff, dtype=float)
    diff = diff - np.round(diff)
    return float(np.min(np.linalg.norm((diff + _IMAGE_SHIFTS) @ lattice, axis=1)))


def _averaged_metric(lattice, rotations=None) -> NDArray[np.float64]:
    """The metric ``G = L L^T`` of ``lattice`` (rows = lattice vectors),
    averaged over ``rotations`` (fractional, spglib convention) when given,
    ``G -> mean(R^T G R)``: a cell that is only approximately symmetric (an
    instantaneous MD cell) becomes exactly symmetric, an exactly symmetric
    cell is left unchanged."""
    lattice = np.asarray(lattice, dtype=float)
    metric = lattice @ lattice.T
    if rotations is not None and len(rotations):
        metric = sum(
            np.asarray(rotation, dtype=float).T @ metric @ np.asarray(rotation, dtype=float)
            for rotation in rotations
        ) / len(rotations)
    return metric


def _symmetrized_lattice(lattice, rotations) -> NDArray[np.float64]:
    """A lattice matrix (rows = lattice vectors, arbitrary orientation) whose
    metric is that of ``lattice`` averaged over ``rotations``."""
    return np.linalg.cholesky(_averaged_metric(lattice, rotations))


def _reciprocal_lengths(lattice, rotations=None) -> NDArray[np.float64]:
    """The reciprocal-axis lengths ``|a*_i|`` (1/Angstrom, no 2 pi) of
    ``lattice`` (rows = lattice vectors), from the metric averaged over
    ``rotations`` when given (see ``_averaged_metric``)."""
    return np.sqrt(np.diag(np.linalg.inv(_averaged_metric(lattice, rotations))))


def _equivalent_isotropic_u(u_cryst: NDArray[np.float64], lattice) -> float:
    """``U_eq`` of a CIF ``U^{ij}`` tensor (Fischer and Tillmanns 1988).

    ``U_eq = (1/3) sum_ij U^{ij} a*_i a*_j (a_i . a_j)``, i.e. one third of
    the trace of the Cartesian tensor ``U_cart = L^T N* U N* L`` with
    ``N* = diag(|a*_i|)``; ``lattice`` rows are the lattice vectors (Angstrom)
    of the cell the ``U^{ij}`` refer to. This is the value ``crystod-md --adp``
    prints next to every site, computed from the tensor it writes.
    """
    lattice = np.asarray(lattice, dtype=float)
    n_star = np.diag(_reciprocal_lengths(lattice))
    u_cart = lattice.T @ n_star @ np.asarray(u_cryst, dtype=float) @ n_star @ lattice
    return float(np.trace(u_cart) / 3.0)


def build_symmetry_projector(rotations, lattice=None) -> NDArray[np.float64]:
    """Build the 6x6 projector onto ``U`` tensors with a given site symmetry.

    ``crystod-md --adp`` calls this once per Wyckoff position with the
    rotations found by ``get_site_symmetry_operations``. A symmetric tensor
    is handled as the component vector ``[U11, U22, U33, U12, U13, U23]`` of
    the CIF ``U^{ij}`` (``U^{ij} = <dx_i dx_j> / (|a*_i| |a*_j|)`` with ``dx``
    the fractional displacement; Trueblood et al. 1996). A fractional rotation
    ``R`` maps ``dx -> R dx``, hence ``U -> M @ U @ M.T`` with
    ``M = N*^-1 @ R @ N*`` and ``N* = diag(|a*_1|, |a*_2|, |a*_3|)``; the
    projector is the average of those 6x6 maps over the group. Multiplying a
    component vector by it leaves the closest tensor that has the full site
    symmetry, which is what ``apply_symmetry_constraints`` does.

    ``M`` equals ``R`` when every rotation only mixes axes of equal
    reciprocal length, which holds for the conventional cells of the
    standard settings and for the F-, C-centred, rhombohedral and
    body-centred cubic primitive cells, but not for body-centred tetragonal
    or orthorhombic primitive cells (or a general basis). ``lattice`` must be
    given for those.

    Args:
        rotations (Sequence[numpy.ndarray]): The rotation matrices of the
            site-symmetry group, each of shape ``(3, 3)`` and acting on
            fractional coordinates (the spglib convention). The sequence
            must be a complete group; ``[identity]`` gives the identity
            projector.
        lattice (array-like, optional): The ``(3, 3)`` lattice matrix of the
            cell (rows = lattice vectors, Angstrom), from which ``N*`` is
            taken; its metric is averaged over ``rotations`` first, so an
            approximately symmetric (instantaneous MD) cell gives exact
            constraints. ``None`` (default) uses ``R`` itself, i.e. assumes
            ``M == R`` as described above.

    Returns:
        The ``(6, 6)`` projector ``P`` (idempotent, ``P @ P == P``).

    Example:
        A two-fold axis along ``c`` forbids ``U13`` and ``U23``:

        >>> import numpy as np
        >>> from crystod import md
        >>> projector = md.build_symmetry_projector(
        ...     [np.eye(3, dtype=int), np.diag([-1, -1, 1])])
        >>> md.get_constraint_description(projector)
        'U13=0, U23=0'
    """
    rotations = [np.asarray(rotation, dtype=float) for rotation in rotations]
    if lattice is not None:
        n_star = _reciprocal_lengths(lattice, rotations)
        # M = N*^-1 R N*, element-wise M_ij = R_ij |a*_j| / |a*_i|
        rotations = [rotation * n_star[np.newaxis, :] / n_star[:, np.newaxis] for rotation in rotations]
    averaged = np.zeros((6, 6))
    for rotation in rotations:
        transform = np.zeros((6, 6))
        for src_index, (i, j) in enumerate(U_INDICES):
            u_basis = np.zeros((3, 3))
            u_basis[i, j] = 1.0
            u_basis[j, i] = 1.0
            u_transformed = rotation @ u_basis @ rotation.T
            for dst_index, (di, dj) in enumerate(U_INDICES):
                transform[dst_index, src_index] = u_transformed[di, dj]
        averaged += transform
    return averaged / len(rotations)


def apply_symmetry_constraints(u_cryst: NDArray[np.float64], projector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetrize a ``U`` tensor with a site-symmetry projector.

    This is the step of ``crystod-md --adp`` that turns the raw displacement
    covariance of a Wyckoff position into the ``U_ij`` written to the CIF
    file: the six independent components of ``u_cryst`` (upper triangle,
    order ``[U11, U22, U33, U12, U13, U23]``) are multiplied by ``projector``
    and reassembled into a symmetric matrix.

    Args:
        u_cryst: Symmetric ``(3, 3)`` tensor of CIF components ``U^{ij}``
            (the basis ``build_symmetry_projector`` works in). Only the upper
            triangle is read.
        projector: The ``(6, 6)`` matrix from ``build_symmetry_projector``.

    Returns:
        The constrained ``(3, 3)`` tensor, symmetric by construction.

    Example:
        >>> import numpy as np
        >>> from crystod import md
        >>> projector = md.build_symmetry_projector(          # two-fold along c
        ...     [np.eye(3, dtype=int), np.diag([-1, -1, 1])])
        >>> u = np.array([[0.010, 0.001, 0.002],
        ...               [0.001, 0.012, 0.003],
        ...               [0.002, 0.003, 0.015]])
        >>> md.apply_symmetry_constraints(u, projector)
        array([[0.01 , 0.001, 0.   ],
               [0.001, 0.012, 0.   ],
               [0.   , 0.   , 0.015]])
    """
    u_vector = np.array([u_cryst[i, j] for i, j in U_INDICES])
    p = projector @ u_vector
    return np.array(
        [
            [p[0], p[3], p[4]],
            [p[3], p[1], p[5]],
            [p[4], p[5], p[2]],
        ]
    )


def get_constraint_description(projector: NDArray[np.float64], tol: float = 1e-6) -> str:
    """Spell out the relations a site-symmetry projector imposes on ``U``.

    ``crystod-md --adp`` prints the result next to every Wyckoff position
    and in its ``Site / Ueq / Constraint`` table, for example
    ``U11=U33, U12=0, U13=0, U23=0`` for the F site of cubic ScF3. Every
    symmetric tensor is ``P @ v`` for some ``v``, so its component ``i`` is
    row ``i`` of the projector ``P`` applied to ``v``: a component whose row
    vanishes is reported as ``Uij=0``, and two components with equal
    (opposite) rows as ``Uij=Ukl`` (``Uij=-Ukl``). This reading holds for
    oblique projectors too, such as those of body-centred tetragonal
    primitive cells. Other linear relations, such as ``U12 = U11/2`` on a
    three-fold axis of a hexagonal cell, are enforced by the projector but
    not spelled out.

    Args:
        projector: The ``(6, 6)`` matrix from ``build_symmetry_projector``.
        tol: Absolute tolerance below which a coefficient counts as zero
            and within which two coefficients count as equal.

    Returns:
        The relations joined by ``", "``; ``"no constraint"`` when the
        projector is the identity (site symmetry 1 or -1), and ``"general
        linear constraints only"`` when it constrains ``U`` only through
        relations of the other kind (a two-fold axis along ``a`` of a
        hexagonal cell: ``U12 = U22/2``, ``U13 = U23/2``).

    Example:
        >>> import numpy as np
        >>> from crystod import md
        >>> three_fold = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]])
        >>> group = [np.eye(3, dtype=int), three_fold, three_fold @ three_fold]
        >>> md.get_constraint_description(md.build_symmetry_projector(group))
        'U11=U22, U13=0, U23=0'
    """
    projector = np.asarray(projector, dtype=float)
    constraints = []
    for i in range(6):
        row = projector[i]
        if np.max(np.abs(row)) < tol:
            constraints.append(f"{U_NAMES[i]}=0")
        else:
            for j in range(i + 1, 6):
                if np.max(np.abs(projector[j] - row)) < tol:
                    constraints.append(f"{U_NAMES[i]}={U_NAMES[j]}")
                elif np.max(np.abs(projector[j] + row)) < tol:
                    constraints.append(f"{U_NAMES[i]}=-{U_NAMES[j]}")
    if constraints:
        return ", ".join(constraints)
    if np.max(np.abs(projector - np.eye(6))) < tol:
        return "no constraint"
    return "general linear constraints only"


def get_site_symmetry_operations(coords, rotations, translations, symprec: float = 0.1, lattice=None):
    """Select the space-group operations that leave a site fixed.

    This is where the ADP constraints of ``crystod-md --adp`` start: the
    site-symmetry group of a Wyckoff position is the subset of the
    space-group operations ``(R, t)`` for which ``R @ x + t`` equals ``x``
    up to a lattice translation. Only the rotation parts are returned,
    which is all ``build_symmetry_projector`` needs.

    Args:
        coords (array-like): Fractional coordinates ``(x, y, z)`` of the site.
        rotations (array-like): Rotation parts of the space-group operations,
            shape ``(n_ops, 3, 3)``, as in ``spglib.get_symmetry(cell)``.
        translations (array-like): The matching translation parts, shape
            ``(n_ops, 3)``.
        symprec: Tolerance on the distance between ``R @ x + t`` and ``x``.
            With ``lattice`` it is the Cartesian minimum-image distance in
            Angstrom, the unit of spglib's ``symprec`` (``crystod-md --adp``
            passes its ``--tolerance`` here and to spglib). Without
            ``lattice`` it is the Euclidean norm of the wrapped fractional
            difference (unitless; kept for backward compatibility).
        lattice (array-like, optional): The ``(3, 3)`` lattice matrix (rows =
            lattice vectors, Angstrom) that makes ``symprec`` a Cartesian
            distance. ``None`` (default) compares fractional coordinates.

    Returns:
        The rotation matrices of the site-symmetry group, a list of ``(3, 3)``
        arrays in the order of ``rotations``.

    Example:
        The F site of cubic ScF3 (Pm-3m) has site symmetry 4/mmm, order 16:

        >>> import numpy as np, spglib
        >>> from crystod import md
        >>> cell = (4.0 * np.eye(3),
        ...         [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
        ...         [21, 9, 9, 9])
        >>> symmetry = spglib.get_symmetry(cell)
        >>> site_ops = md.get_site_symmetry_operations(
        ...     [0.5, 0, 0], symmetry["rotations"], symmetry["translations"])
        >>> len(site_ops)
        16
        >>> md.get_constraint_description(md.build_symmetry_projector(site_ops))
        'U22=U33, U12=0, U13=0, U23=0'
    """
    if lattice is not None:
        lattice = np.asarray(lattice, dtype=float)
    site_operations = []
    for rotation, translation in zip(rotations, translations):
        transformed = (rotation @ coords + translation) % 1.0
        diff = transformed - coords
        diff = diff - np.round(diff)
        distance = np.linalg.norm(diff) if lattice is None else _min_image_distance(diff, lattice)
        if distance < symprec:
            site_operations.append(rotation)
    return site_operations


def _representative_operations(site_coords, equivalent_atoms, rotations, translations, lattice, symprec):
    """For every unit-cell site, the index of the space-group operation that
    maps it onto its asymmetric-unit representative ``equivalent_atoms[site]``.

    The operation whose image ``R @ x + t`` lies closest to the representative
    (Cartesian minimum-image distance; the first one among equally close
    operations) is taken when that distance is below ``symprec`` (Angstrom,
    the tolerance spglib used to call the sites equivalent); otherwise the
    entry is ``None`` and the caller reports the site's atoms.
    """
    lattice = np.asarray(lattice, dtype=float)
    symop_ids = []
    for site_index, coord in enumerate(site_coords):
        representative = site_coords[equivalent_atoms[site_index]]
        best_index, best_distance = None, np.inf
        for symop_index, (rotation, translation) in enumerate(zip(rotations, translations)):
            distance = _min_image_distance(rotation @ coord + translation - representative, lattice)
            if distance < best_distance - 1e-8:
                best_index, best_distance = symop_index, distance
        symop_ids.append(best_index if best_distance < symprec else None)
    return symop_ids


def rotation_to_xyz(rotation, translation) -> str:
    """Convert a rotation matrix and translation vector to a CIF xyz string."""
    axes = ["x", "y", "z"]
    result = []
    for i in range(3):
        terms = []
        for j in range(3):
            if rotation[i, j] == 1:
                terms.append(axes[j])
            elif rotation[i, j] == -1:
                terms.append(f"-{axes[j]}")
            elif rotation[i, j] != 0:
                terms.append(f"{rotation[i, j]}*{axes[j]}")
        if abs(translation[i]) > 1e-6:
            frac = translation[i]
            for value, text in ((0.5, "1/2"), (0.25, "1/4"), (0.75, "3/4"), (1 / 3, "1/3"), (2 / 3, "2/3")):
                if abs(frac - value) < 1e-6:
                    terms.append(text)
                    break
            else:
                terms.append(f"{frac:.4f}")
        if not terms:
            terms = ["0"]
        result.append("+".join(terms).replace("+-", "-"))
    return ", ".join(result)


def _circular_mean(values: NDArray[np.float64], period: float = 1.0) -> float:
    """Mean of periodic values (period-aware, robust against boundary wrapping)."""
    angles = values * 2 * np.pi / period
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    return (mean_angle / (2 * np.pi) * period) % period


# =============================================================================
# Main workflow
# =============================================================================

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    supercell_size = np.array([int(value) for value in args.dim.split()], dtype=int)
    if supercell_size.shape != (3,):
        raise SystemExit('ERROR: --dim requires three integers, e.g. --dim "4 4 4".')

    print("\n* Input *")
    print(f"  Supercell size : {supercell_size.tolist()}")
    print(f"  Start step     : {args.start_step}")
    print(f"  Input file     : {args.xdatcar}")
    print(f"  Output file    : {args.output}")

    # Step 1: read XDATCAR
    print("  Reading XDATCAR... (this may take a while)")
    chem_formula, lattices, all_coordinates = read_xdatcar(args.xdatcar)

    if args.start_step >= len(all_coordinates):
        raise SystemExit(
            f"ERROR: --start-step {args.start_step} exceeds the number of MD steps "
            f"({len(all_coordinates)})."
        )
    coordinates = all_coordinates[args.start_step :]
    n_steps, n_atoms_super, _ = coordinates.shape
    lattice_vec_super = lattices[args.start_step]
    lattice_vec_unit = lattice_vec_super / supercell_size[:, np.newaxis]

    composition = Counter(chem_formula)
    print("\n* Supercell info *")
    print(f"  atoms          : {n_atoms_super}")
    print(f"  composition    : {dict(composition)}")
    print(f"  analyzed steps : {n_steps}")

    # Step 2: time-averaged coordinates (circular mean per axis)
    avg_coords = np.zeros((n_atoms_super, 3))
    for i in range(n_atoms_super):
        for axis in range(3):
            avg_coords[i, axis] = _circular_mean(coordinates[:, i, axis])

    # Step 3: fold into one unit cell of the supercell
    fold_thresholds = 1.0 / supercell_size
    folded_coords = avg_coords % fold_thresholds

    # Step 4: group supercell atoms into unit-cell sites
    grouped_indices = []
    remaining_indices = set(range(n_atoms_super))
    adjusted_tolerance = args.grouping_tolerance / np.mean(supercell_size)
    while remaining_indices:
        ref_index = min(remaining_indices)
        ref_coord = folded_coords[ref_index]
        ref_element = chem_formula[ref_index]
        group = []
        for index in list(remaining_indices):
            if chem_formula[index] != ref_element:
                continue
            diff = ref_coord - folded_coords[index]
            for axis in range(3):
                threshold = fold_thresholds[axis]
                diff[axis] = diff[axis] - np.round(diff[axis] / threshold) * threshold
            if np.linalg.norm(diff) < adjusted_tolerance:
                group.append(index)
                remaining_indices.remove(index)
        grouped_indices.append(group)

    print(f"  Number of unit-cell sites (groups): {len(grouped_indices)}")
    print(f"  Expected replicas per group       : {int(np.prod(supercell_size))}")

    # Step 5: averaged unit-cell coordinate of each group
    group_unitcell_coords = []
    group_elements = []
    for atom_indices in grouped_indices:
        group_elements.append(chem_formula[atom_indices[0]])
        avg_folded = np.array(
            [
                _circular_mean(folded_coords[atom_indices][:, axis], period=fold_thresholds[axis])
                for axis in range(3)
            ]
        )
        group_unitcell_coords.append((avg_folded * supercell_size) % 1.0)
    group_unitcell_coords = np.array(group_unitcell_coords)

    # Step 6: symmetry analysis of the averaged unit cell
    unique_elements = sorted(set(group_elements))
    element_to_number = {element: i + 1 for i, element in enumerate(unique_elements)}
    atom_numbers = [element_to_number[element] for element in group_elements]
    cell = (lattice_vec_unit, group_unitcell_coords, atom_numbers)
    raw_dataset = spglib.get_symmetry_dataset(cell, symprec=args.symprec)
    if raw_dataset is None:
        raise SystemExit("ERROR: spglib could not find any symmetry for the averaged structure.")
    dataset = SymmetryDatasetAdapter(raw_dataset)

    spacegroup = dataset["international"]
    spacegroup_number = dataset["number"]
    rotations = np.array(dataset["rotations"])
    translations = np.array(dataset["translations"])
    equivalent_atoms = np.array(dataset["equivalent_atoms"])

    print("\n* Time-averaged unit cell *")
    print(f"  Space group: {spacegroup} (No. {spacegroup_number})")
    print(f"  Atoms in unit cell   : {len(group_unitcell_coords)}")
    print(f"  Symmetry operations  : {len(rotations)}")

    wyckoff_ids_sorted = sorted(set(equivalent_atoms))
    wyckoff_to_cif_index = {wid: i for i, wid in enumerate(wyckoff_ids_sorted)}
    print(f"  Asymmetric-unit sites: {len(wyckoff_ids_sorted)}")

    asym_sites = []
    for cif_index, wyckoff_id in enumerate(wyckoff_ids_sorted):
        element = group_elements[wyckoff_id]
        coords = group_unitcell_coords[wyckoff_id]
        multiplicity = int(np.sum(equivalent_atoms == wyckoff_id))
        asym_sites.append(
            {
                "label": f"{element}{cif_index}",
                "element": element,
                "frac_coords": coords,
                "multiplicity": multiplicity,
            }
        )
        print(
            f"    {asym_sites[-1]['label']}: mult={multiplicity}, "
            f"coords=({coords[0]:.5f}, {coords[1]:.5f}, {coords[2]:.5f})"
        )

    # ADP constraints per Wyckoff position. The U^{ij} transformation
    # M = N*^-1 R N* takes N* from the cell metric made exactly invariant under
    # the whole space group (the first-frame NpT cell is only approximately
    # so); M == R for every conventional cell.
    symmetric_lattice = _symmetrized_lattice(lattice_vec_unit, rotations)
    wyckoff_constraints = {}
    wyckoff_projectors = {}
    print("\n* ADP constraints per Wyckoff position *")
    for wyckoff_id in wyckoff_ids_sorted:
        coords = group_unitcell_coords[wyckoff_id]
        site_operations = get_site_symmetry_operations(
            coords, rotations, translations, args.symprec, lattice=lattice_vec_unit
        )
        projector = build_symmetry_projector(site_operations, lattice=symmetric_lattice)
        wyckoff_projectors[wyckoff_id] = projector
        constraint = get_constraint_description(projector)
        wyckoff_constraints[wyckoff_id] = constraint
        element = group_elements[wyckoff_id]
        print(
            f"  Wyckoff {wyckoff_id} ({element}): site-symmetry order={len(site_operations)}, {constraint}"
        )

    # Step 9: unwrap trajectories into unit-cell Cartesian coordinates
    lattice_inv = np.linalg.inv(lattice_vec_unit)
    all_atom_cart_coords = []
    all_atom_group_ids = []
    all_atom_indices = []
    for group_index, atom_indices in enumerate(grouped_indices):
        for atom_index in atom_indices:
            all_atom_indices.append(atom_index)
            supercell_frac = coordinates[:, atom_index, :] % 1.0
            unitcell_frac = supercell_frac @ lattice_vec_super @ lattice_inv
            steps_diff = np.diff(unitcell_frac, axis=0)
            steps_diff -= np.round(steps_diff)
            unwrapped = np.empty_like(unitcell_frac)
            unwrapped[0] = unitcell_frac[0] - np.round(unitcell_frac[0])
            if n_steps > 1:
                unwrapped[1:] = unwrapped[0] + np.cumsum(steps_diff, axis=0)
            all_atom_cart_coords.append(unwrapped @ lattice_vec_unit)
            all_atom_group_ids.append(group_index)
    all_atom_cart_coords = np.array(all_atom_cart_coords)
    all_atom_group_ids = np.array(all_atom_group_ids)
    print("\n* Trajectory unwrapping *")
    print(f"  Coordinate unwrapping done: {len(all_atom_cart_coords)} atoms x {n_steps} steps")

    # Step 10: map every atom onto its Wyckoff representative
    group_symop_ids = _representative_operations(
        group_unitcell_coords, equivalent_atoms, rotations, translations, lattice_vec_unit, args.symprec
    )
    all_atom_wyckoff_ids = [equivalent_atoms[group_index] for group_index in all_atom_group_ids]
    all_atom_symop_ids = [group_symop_ids[group_index] for group_index in all_atom_group_ids]
    unmapped = sorted(
        all_atom_indices[atom_index]
        for atom_index, symop_index in enumerate(all_atom_symop_ids)
        if symop_index is None
    )
    if unmapped:
        print(
            f"  WARNING: {len(unmapped)} atom(s) could not be mapped onto their asymmetric-unit "
            f"representative within --tolerance {args.symprec:g} A and are excluded from the "
            f"ADP statistics (XDATCAR atom numbers, 1-based): "
            + ", ".join(str(index + 1) for index in unmapped)
        )

    # Step 11: collect displacements rotated into the representative frame
    volume = np.linalg.det(lattice_vec_unit)
    a_star = np.cross(lattice_vec_unit[1], lattice_vec_unit[2]) / volume
    b_star = np.cross(lattice_vec_unit[2], lattice_vec_unit[0]) / volume
    c_star = np.cross(lattice_vec_unit[0], lattice_vec_unit[1]) / volume
    reciprocal_unit = np.array(
        [
            a_star / np.linalg.norm(a_star),
            b_star / np.linalg.norm(b_star),
            c_star / np.linalg.norm(c_star),
        ]
    )

    wyckoff_displacements = defaultdict(list)
    for atom_index in range(len(all_atom_cart_coords)):
        symop_index = all_atom_symop_ids[atom_index]
        if symop_index is None:
            continue
        atom_cart = all_atom_cart_coords[atom_index]
        displacements = atom_cart - np.mean(atom_cart, axis=0)
        rotation_cart = lattice_vec_unit.T @ rotations[symop_index] @ np.linalg.inv(lattice_vec_unit.T)
        wyckoff_displacements[all_atom_wyckoff_ids[atom_index]].append(displacements @ rotation_cart.T)
    for wyckoff_id in list(wyckoff_displacements.keys()):
        wyckoff_displacements[wyckoff_id] = np.concatenate(wyckoff_displacements[wyckoff_id], axis=0)

    # Step 12: ADP tensors
    wyckoff_averaged_u_cryst = {}
    print("\n* Atomic displacement parameters *")
    print(f"{'Site':<10} {'Ueq (A^2)':<12} {'Constraint':<35}")
    print("-" * 60)
    for wyckoff_id in wyckoff_ids_sorted:
        displacements_all = wyckoff_displacements[wyckoff_id]
        u_cart = np.cov(displacements_all.T, bias=True)
        u_cryst_before = reciprocal_unit @ u_cart @ reciprocal_unit.T
        u_cryst = apply_symmetry_constraints(u_cryst_before, wyckoff_projectors[wyckoff_id])
        wyckoff_averaged_u_cryst[wyckoff_id] = u_cryst

        label = asym_sites[wyckoff_to_cif_index[wyckoff_id]]["label"]
        # U_eq of the tensor written to the CIF (in the CIF cell)
        u_eq = _equivalent_isotropic_u(u_cryst, lattice_vec_unit)
        print(f"{label:<10} {u_eq:<12.6f} {wyckoff_constraints[wyckoff_id]:<35}")

    # Step 13: CIF output
    a, b, c = np.linalg.norm(lattice_vec_unit, axis=1)
    alpha = np.degrees(np.arccos(np.dot(lattice_vec_unit[1], lattice_vec_unit[2]) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(lattice_vec_unit[0], lattice_vec_unit[2]) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(lattice_vec_unit[0], lattice_vec_unit[1]) / (a * b)))

    element_counts = Counter(group_elements)
    formula = "".join(
        f"{element}{count}" if count > 1 else element for element, count in sorted(element_counts.items())
    )
    formula_sum = " ".join(f"{element}{count}" for element, count in sorted(element_counts.items()))

    cif = []
    cif.append("# generated using crystod-md --adp")
    cif.append(f"data_{formula}")
    cif.append(f"_symmetry_space_group_name_H-M   '{spacegroup}'")
    cif.append(f"_cell_length_a   {a:.8f}")
    cif.append(f"_cell_length_b   {b:.8f}")
    cif.append(f"_cell_length_c   {c:.8f}")
    cif.append(f"_cell_angle_alpha   {alpha:.8f}")
    cif.append(f"_cell_angle_beta   {beta:.8f}")
    cif.append(f"_cell_angle_gamma   {gamma:.8f}")
    cif.append(f"_symmetry_Int_Tables_number   {spacegroup_number}")
    cif.append(f"_chemical_formula_structural   {formula}")
    cif.append(f"_chemical_formula_sum   '{formula_sum}'")
    cif.append(f"_cell_volume   {volume:.8f}")
    cif.append("_cell_formula_units_Z   1")
    cif.append("")
    cif.append("loop_")
    cif.append(" _symmetry_equiv_pos_site_id")
    cif.append(" _symmetry_equiv_pos_as_xyz")
    for i, (rotation, translation) in enumerate(zip(rotations, translations)):
        cif.append(f"  {i + 1}  '{rotation_to_xyz(rotation, translation)}'")
    cif.append("")
    cif.append("loop_")
    cif.append(" _atom_site_type_symbol")
    cif.append(" _atom_site_label")
    cif.append(" _atom_site_symmetry_multiplicity")
    cif.append(" _atom_site_fract_x")
    cif.append(" _atom_site_fract_y")
    cif.append(" _atom_site_fract_z")
    cif.append(" _atom_site_occupancy")
    for site in asym_sites:
        coords = site["frac_coords"]
        cif.append(
            f"  {site['element']}  {site['label']}  {site['multiplicity']}  "
            f"{coords[0]:.8f}  {coords[1]:.8f}  {coords[2]:.8f}  1"
        )
    cif.append("")
    cif.append("loop_")
    cif.append(" _atom_site_aniso_label")
    cif.append(" _atom_site_aniso_U_11")
    cif.append(" _atom_site_aniso_U_22")
    cif.append(" _atom_site_aniso_U_33")
    cif.append(" _atom_site_aniso_U_23")
    cif.append(" _atom_site_aniso_U_13")
    cif.append(" _atom_site_aniso_U_12")
    for wyckoff_id in wyckoff_ids_sorted:
        u = wyckoff_averaged_u_cryst[wyckoff_id]
        label = asym_sites[wyckoff_to_cif_index[wyckoff_id]]["label"]
        cif.append(
            f" {label:<8}{u[0, 0]:<10.5f}{u[1, 1]:<10.5f}{u[2, 2]:<10.5f}"
            f"{u[1, 2]:<10.5f}{u[0, 2]:<10.5f}{u[0, 1]:<10.5f}"
        )

    with open(args.output, "w") as fp:
        fp.write("\n".join(cif))

    print("\n* Output files *")
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
