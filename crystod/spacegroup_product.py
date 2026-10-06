"""Direct products of space-group irreps (crystod-group --product --space-group).

Decomposes the direct product of full space-group irreps (induced from the
little-group irreps of high-symmetry k points) into full space-group irreps:

    chi_(k1,mu) x chi_(k2,nu) = sum over stars k3 in star(k1)+star(k2) of
                                n_(k3,lam) chi_(k3,lam)

The characters come from the ISO-IR tables bundled with crystod (the same
source as every other crystod irrep label), induced from the little group
to the full group over the star arms. The reduction coefficients are
the standard character inner products over the finite factor group G/T_N;
the translation sum is carried out analytically, leaving the momentum
conservation condition k1_a + k2_b = k3_c (mod reciprocal lattice) over the
star arms:

    n = (1/|P|) sum_i sum_{a,b,c: q_a+q_b=q_c mod Z}
        C1[i,a] C2[i,b] conj(C3[i,c])

where C[i,a] = chi_small(s_a^{-1} g_i s_a) is the arm-transported small
character (zero when g_i does not fix arm a) and q_a the arm wave vector.

The implementation is validated line by line against the DIRPRO program of
the Bilbao Crystallographic Server (https://cryst.ehu.es/rep/dirpro.html):
M. I. Aroyo, A. Kirov, C. Capillas, J. M. Perez-Mato and H. Wondratschek,
"Bilbao Crystallographic Server II: Representations of crystallographic
point groups and space groups", Acta Cryst. A62, 115-128 (2006).

Product terms at k points absent from the tables (symmetry lines reached
by sums of star arms) are computed with spgrep and named from the ISO-IR
tables (``crystod.isoir``); they are marked ``[ISO-IR labels]`` in the
report.  All labels follow the ISO-IR (ISOTROPY, Miller-Love) convention
throughout.

The symmetric and antisymmetric squares of one irrep (``--symmetric``,
``--antisymmetric``) are reduced from the characters
``(chi(g)^2 +- chi(g^2)) / 2`` of the explicit order-parameter matrices
(``crystod.isotropy_subgroup.InducedRepresentation``, the physically
irreducible real form) over the finite factor group ``G / T_N``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from phonopy.structure.cells import get_primitive_matrix_by_centring

from .basis_function import _resolve_space_group_type, _synthetic_conventional_cell
from .irreptables_compat import load_irreptables

IrrepTable, _Irrep = load_irreptables()

# global denominator for exact fractional arithmetic on k vectors and
# translations: every special-point coordinate tabulated in the ISO-IR
# tables (all 230 space groups) and every space-group fractional
# translation is a multiple of 1/24, and sums of 1/24-grid vectors stay
# on the grid
DEN = 24

# sign convention of the translation phase in the crystod-internal small
# representations: D(W, v + t) = exp(SIGMA * 2j*pi * k.t) * D(W, v)
SIGMA = -1.0


class _ComputedIrrep:
    """Line-point small irrep computed on the fly (spgrep); mimics the
    tabulated-irrep interface used by the report."""

    def __init__(
        self, name: str, dim: int, kpname: str, k_int, star_size: int,
        label_source: str | None = None,
    ):
        self.name = name
        self.dim = dim
        self.kpname = kpname
        self.k_int = np.asarray(k_int, dtype=np.int64)
        self.star_size = star_size
        # naming convention of .name: "isoir" (ISO-IR / ISOTROPY Miller-Love
        # tables) or None (positional)
        self.label_source = label_source


class _SyntheticIrrep:
    """Tabulated-like irrep synthesized from another one (e.g. the conjugate
    irreps of the -k star for polar space groups, 'A'-suffixed points)."""

    def __init__(self, name: str, dim: int, kpname: str, characters: dict):
        self.name = name
        self.dim = dim
        self.kpname = kpname
        self.characters = characters


@dataclass
class SquareDecomposition:
    """Symmetric or antisymmetric square of a full space-group irrep.

    Returned by ``SpaceGroupIrrepAlgebra.decompose_square``.

    Attributes:
        label: Label of the squared representation: the ISO-IR irrep label,
            or the ISOTROPY pair label (``P1P2``) of the physically
            irreducible real form of a complex-type irrep.
        irrep: The tabulated irrep record of the requested label.
        kind: ``"symmetric"`` or ``"antisymmetric"``.
        dimension: Dimension of the square, ``n(n+1)/2`` or ``n(n-1)/2``.
        rep_dimension: Dimension ``n`` of the squared (real) representation.
        doubled: ``True`` when the square is that of the real form
            ``D + D*`` of a complex- or pseudoreal-type irrep.
        fs_type: ``"complex"`` or ``"pseudoreal"`` when ``doubled``, else
            ``None``.
        terms: ``(kname, irrep, multiplicity)`` as in
            ``SpaceGroupIrrepAlgebra.decompose_product``.
        leftovers: k vectors (units of ``1/DEN``) that could not be
            decomposed.
    """

    label: str
    irrep: object
    kind: str
    dimension: int
    rep_dimension: int
    doubled: bool
    fs_type: str | None
    terms: list = field(default_factory=list)
    leftovers: list = field(default_factory=list)


def _character_fingerprint(chi: dict) -> tuple:
    return tuple(
        (int(op), round(complex(value).real, 4), round(complex(value).imag, 4))
        for op, value in sorted(chi.items())
    )


def _resolve_space_group(space_group: str):
    """Resolve a space-group symbol or number to the spglib type info."""
    text = str(space_group).strip()
    if text.isdigit():
        import spglib

        from .runtime_compat import get_spacegroup_type

        number = int(text)
        for hall_number in range(1, 531):
            info = get_spacegroup_type(spglib.get_spacegroup_type(hall_number))
            if info.number == number:
                return info
        raise SystemExit(f"ERROR: unknown space-group number {number}.")
    return _resolve_space_group_type(text)


def _snap(values, denominator: int = DEN) -> np.ndarray:
    """Snap floats to exact multiples of 1/denominator, returned as integers."""
    array = np.asarray(values, dtype=float) * denominator
    snapped = np.rint(array)
    if not np.allclose(array, snapped, atol=1e-4 * denominator):
        raise SystemExit(
            f"ERROR: coordinate {np.asarray(values)} is not commensurate with 1/{denominator}."
        )
    return snapped.astype(np.int64)


class SpaceGroupIrrepAlgebra:
    """Space-group operations and ISO-IR irrep tables in the primitive basis.

    The algebra behind ``crystod-group --product IRREP... --sg SG`` and the
    isotropy-subgroup and symmetry-mode machinery: it holds the coset
    representatives of the space group (rotations and translations in the
    primitive basis, translations as integers in units of ``1/DEN``), the
    tabulated ISO-IR small irreps grouped by k-point name, and the induced
    (full) characters over the star of every k point.  Every convention is
    verified at run time (group closure, little-group match against the
    tables), and the tables are extended on the fly with spgrep for k points
    that are not tabulated (symmetry lines and planes reached by sums of
    star arms).

    Args:
        space_group_symbol: International short symbol (``"Pm-3m"``,
            ``"P6_3/mmc"``) or space-group number (``"221"``).

    Attributes:
        sg_type: spglib space-group type record (``number``,
            ``international_short``, ``hall_number``, ...).
        table: The ISO-IR irrep table of the space group (``irreps``,
            ``symmetries``).
        primitive_matrix: Conventional-to-primitive transformation matrix
            (phonopy convention for the centring).
        rotations: Integer rotation parts of the coset representatives in
            the primitive basis, shape ``(n_ops, 3, 3)``.
        translations: Translation parts, shape ``(n_ops, 3)``, integers in
            units of ``1/DEN`` (``DEN = 24``), reduced modulo lattice
            translations.
        n_ops: Number of coset representatives (order of the point group).
        irreps_by_kname: Tabulated irreps grouped by k-point name, in table
            order (``{"GM": [...], "R": [...], ...}``); each irrep record has
            ``name``, ``dim``, ``kpname`` and ``characters``.
        k_by_kname: k vector of every tabulated k point in the primitive
            basis, integers in units of ``1/DEN``.

    Raises:
        SystemExit: Unknown space-group symbol or number, or a table whose
            conventions cannot be reconciled with the primitive setting.

    Example:
        >>> from crystod import group
        >>> algebra = group.SpaceGroupIrrepAlgebra("Pm-3m")
        >>> algebra.n_ops, list(algebra.k_by_kname)
        (48, ['GM', 'R', 'X', 'M'])
        >>> [irrep.name for irrep in algebra.irreps_by_kname["R"]]
        ['R1+', 'R2+', 'R3+', 'R4+', 'R5+', 'R1-', 'R2-', 'R3-', 'R4-', 'R5-']
    """

    def __init__(self, space_group_symbol: str):
        sg_type = _resolve_space_group(space_group_symbol)
        self.sg_type = sg_type
        self.table = IrrepTable(sg_type.number, spinor=False)
        primitive_matrix = np.array(
            get_primitive_matrix_by_centring(sg_type.international_short[0]), dtype=float
        )
        self.primitive_matrix = primitive_matrix

        conventional_rotations = np.array(
            [sym.R for sym in self.table.symmetries], dtype=float
        )
        conventional_translations = np.array(
            [sym.t for sym in self.table.symmetries], dtype=float
        )
        inverse = np.linalg.inv(primitive_matrix)
        rotations = np.rint(
            np.array([inverse @ rotation @ primitive_matrix for rotation in conventional_rotations])
        ).astype(np.int64)

        # translation transform convention: the pure column form
        # t_prim = M^-1 t_conv closes for all 230 space groups (the row form
        # t_conv M^-1, kept as a safety net, coincides with it whenever it
        # closes at all); the column form is required so that the exact
        # inverse t_conv = M t_prim is available for the ISO-IR labeler
        for candidate in (
            conventional_translations @ inverse.T,
            conventional_translations @ inverse,
        ):
            exact_translations = _snap(candidate)
            translations = np.mod(exact_translations, DEN)
            if self._is_closed(rotations, translations):
                break
        else:
            raise SystemExit("ERROR: could not build a closed primitive-setting space group.")

        # lattice translation wrapped away by the mod above (integer
        # vectors, primitive basis): needed to relate the tabulated small
        # characters (defined at the exact translations) to spgrep
        # characters computed at the wrapped translations
        self.wrap_delta = ((exact_translations - translations) // DEN).astype(np.int64)

        self.rotations = rotations                      # (n, 3, 3) int
        self.inverse_rotations = np.rint(
            np.array([np.linalg.inv(rotation) for rotation in rotations])
        ).astype(np.int64)
        self.translations = translations                # (n, 3) int, units of 1/DEN
        self.n_ops = len(rotations)
        self._rotation_index = {
            self._key(rotation): index for index, rotation in enumerate(rotations)
        }

        # irreps grouped by k-point name, in table order
        self.irreps_by_kname: dict[str, list] = {}
        self.k_by_kname: dict[str, np.ndarray] = {}
        for irrep in self.table.irreps:
            self.irreps_by_kname.setdefault(irrep.kpname, []).append(irrep)
            if irrep.kpname not in self.k_by_kname:
                k_primitive = np.mod(_snap(np.array(irrep.k, dtype=float) @ primitive_matrix), DEN)
                self.k_by_kname[irrep.kpname] = k_primitive

        self._star_cache: dict[str, tuple[np.ndarray, list[int]]] = {}
        self._induced_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._computed_cache: dict[tuple, list] = {}
        self._isoir_cell = None  # lazy synthetic cell for the ISO-IR labeler
        self._isoir_line_cache: dict[tuple, tuple | None] = {}

        self._add_minus_k_stars()

    def _add_minus_k_stars(self) -> None:
        """Synthesize the -k ('A') stars of the special points of acentric
        space groups.

        For space groups without inversion, -k of a tabulated special point
        may belong to a star that the ISO-IR tables do not list (e.g. PA of
        I-43m). The allowed small irreps at -k are the complex conjugates of
        those at k; they are named as the ISOTROPY software names them, with
        an 'A' suffix on the k-point letter (P1 -> PA1), as the labeller of
        ``crystod.isoir`` does.  The -k partners of lines and planes are
        named in ``_isoir_line_labels``.
        """
        from .isoir import minus_k_label
        for kname in list(self.k_by_kname):
            k = self.k_by_kname[kname]
            arms, _ = self.star(kname)
            minus_k = np.mod(-k, DEN)
            if any(np.all((arm - minus_k) % DEN == 0) for arm in arms):
                continue  # -k inside the same star (e.g. via inversion)
            covered = False
            for other in self.k_by_kname:
                if other == kname:
                    continue
                other_arms, _ = self.star(other)
                if any(np.all((arm - minus_k) % DEN == 0) for arm in other_arms):
                    covered = True
                    break
            if covered:
                continue
            new_kname = minus_k_label(kname, self.sg_type.number)
            self.k_by_kname[new_kname] = minus_k
            self.irreps_by_kname[new_kname] = [
                _SyntheticIrrep(
                    name=new_kname + irrep.name[len(kname):],
                    dim=int(irrep.dim),
                    kpname=new_kname,
                    characters={
                        key: np.conj(complex(value))
                        for key, value in irrep.characters.items()
                    },
                )
                for irrep in self.irreps_by_kname[kname]
            ]

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _key(rotation: np.ndarray) -> bytes:
        return np.asarray(rotation, dtype=np.int64).tobytes()

    @staticmethod
    def _is_closed(rotations: np.ndarray, translations: np.ndarray) -> bool:
        index = {SpaceGroupIrrepAlgebra._key(r): i for i, r in enumerate(rotations)}
        for i in range(len(rotations)):
            for j in range(len(rotations)):
                rotation = rotations[i] @ rotations[j]
                m = index.get(SpaceGroupIrrepAlgebra._key(rotation))
                if m is None:
                    return False
                translation = rotations[i] @ translations[j] + translations[i]
                if np.any((translation - translations[m]) % DEN != 0):
                    return False
        return True

    def find_irrep(self, label: str):
        """Tabulated irrep record with the given ISO-IR label.

        Args:
            label: ISO-IR irrep label, e.g. ``"R4+"``.

        Returns:
            The irrep record (attributes ``name``, ``dim``, ``kpname``,
            ``characters``) of the ISO-IR table.

        Raises:
            SystemExit: The label is not tabulated for this space group; the
                message lists the available labels.
        """
        for irreps in self.irreps_by_kname.values():
            for irrep in irreps:
                if irrep.name == label:
                    return irrep
        available = ", ".join(
            irrep.name for irreps in self.irreps_by_kname.values() for irrep in irreps
        )
        raise SystemExit(
            f'ERROR: irrep "{label}" is not tabulated for space group '
            f"{self.sg_type.international_short} (No. {self.sg_type.number}).\n"
            f"Available irreps: {available}"
        )

    def little_group(self, k: np.ndarray) -> list[int]:
        """Indices of the coset representatives in the little group of k.

        Args:
            k: k vector in the primitive basis, integers in units of
                ``1/DEN``.

        Returns:
            The operation indices ``i`` whose q-action ``k . W_i^-1`` leaves
            k invariant modulo the reciprocal lattice.
        """
        return [
            i
            for i in range(self.n_ops)
            if np.all((k @ self.inverse_rotations[i] - k) % DEN == 0)
        ]

    def star(self, kname: str) -> tuple[np.ndarray, list[int]]:
        """Arms of the star of a tabulated k point.

        Args:
            kname: k-point name of the ISO-IR table, e.g. ``"X"``.

        Returns:
            ``(arms, representatives)``: the arms as an integer array of
            shape ``(n_arms, 3)`` in the q-convention
            ``q_a = k . W_{s_a}^-1`` (units of ``1/DEN``), and the index of
            the coset representative ``s_a`` generating every arm.
        """
        if kname in self._star_cache:
            return self._star_cache[kname]
        k = self.k_by_kname[kname]
        arms = [tuple(k % DEN)]
        representatives = [self._rotation_index[self._key(np.eye(3))]]
        for i in range(self.n_ops):
            q = tuple((k @ self.inverse_rotations[i]) % DEN)
            if q not in arms:
                arms.append(q)
                representatives.append(i)
        result = (np.array(arms, dtype=np.int64), representatives)
        self._star_cache[kname] = result
        return result

    def induced_characters(self, irrep) -> tuple[np.ndarray, np.ndarray]:
        """Characters of the full (induced) irrep, arm by arm.

        Args:
            irrep: A tabulated irrep record (from ``find_irrep`` or
                ``irreps_by_kname``).

        Returns:
            ``(arms, C)`` with ``C[i, a] = chi_small(s_a^-1 g_i s_a)`` when
            operation ``g_i`` fixes arm ``a`` and ``0`` otherwise; the full
            character of ``(g_i, t)`` is
            ``sum_a C[i, a] exp(SIGMA * 2j * pi * q_a . t)``.

        Raises:
            SystemExit: The tabulated characters do not match any small
                representation allowed at that k point (convention
                mismatch).
        """
        cache_key = (irrep.kpname, irrep.name)
        if cache_key in self._induced_cache:
            return self._induced_cache[cache_key]

        k = self.k_by_kname[irrep.kpname]
        arms, representatives = self.star(irrep.kpname)
        little = self.little_group(k)
        table_keys = sorted(int(key) - 1 for key in irrep.characters.keys())
        if table_keys != sorted(little):
            raise SystemExit(
                f"ERROR: little group of {irrep.kpname} does not match the "
                f"tabulated operations of {irrep.name} (convention mismatch)."
            )
        small = {
            int(key) - 1: complex(value) for key, value in irrep.characters.items()
        }
        refined = self._refine_small_characters(k, small)
        if refined is None:
            raise SystemExit(
                f"ERROR: the tabulated characters of {irrep.name} "
                f"(space group {self.sg_type.number}) do not correspond to "
                "any allowed small representation. Please report this case."
            )
        small = refined

        C = np.zeros((self.n_ops, len(arms)), dtype=np.complex128)
        for a, s in enumerate(representatives):
            W_s, v_s = self.rotations[s], self.translations[s]
            W_s_inv = self.inverse_rotations[s]
            # s^{-1} = (W_s^{-1}, -W_s^{-1} v_s)
            v_s_inv = -W_s_inv @ v_s
            for i in range(self.n_ops):
                W_h = W_s_inv @ self.rotations[i] @ W_s
                m = self._rotation_index[self._key(W_h)]
                if m not in small:
                    continue
                # tau_h = W_s^{-1} (W_i v_s + v_i) + v_s^{-1}
                tau_h = W_s_inv @ (self.rotations[i] @ v_s + self.translations[i]) + v_s_inv
                t_extra = tau_h - self.translations[m]
                if np.any(t_extra % DEN != 0):
                    raise SystemExit("ERROR: broken coset bookkeeping (non-lattice residue).")
                phase = np.exp(SIGMA * 2j * np.pi * float(k @ (t_extra // DEN)) / DEN)
                C[i, a] = phase * small[m]

        self._induced_cache[cache_key] = (arms, C)
        return arms, C

    def _refine_small_characters(self, k: np.ndarray, small: dict) -> dict | None:
        """Match the tabulated characters onto the exact spgrep values.

        The tabulated ISO-IR characters (phase convention exp(+2*pi*i k.t),
        exact translations) and the spgrep candidates (exp(-2*pi*i k.t),
        translations reduced mod 1) describe the same small irrep when
        chi_spgrep(op) = conj(chi_table(op)) * exp(+2*pi*i k.delta_op),
        delta_op being the lattice translation wrapped away when the
        primitive operators were reduced mod 1.  The candidate selected this
        way is the irrep of the same name in the ISO-IR data (checked for
        every special-point irrep of all 230 space groups: always exactly
        one candidate).  A direct comparison chi_spgrep = chi_table ignores
        both factors and attaches the label to a physically different irrep
        at some points (P of I4/mcm, H/K of the trigonal and hexagonal
        groups, Y and T of Ccce, ...).  Returns None when no unique
        candidate exists; the tabulated characters are returned unchanged
        when spgrep cannot compute the small irreps at k.
        """
        try:
            computed = self.computed_irreps_at(k)
        except SystemExit:
            return small
        conjugated = []
        for candidate in computed:
            chi = candidate["chi"]
            if set(chi.keys()) != set(small.keys()):
                continue
            if all(
                abs(chi[op] - np.conj(small[op]) * self._wrap_phase(k, op)) < 5e-3
                for op in small
            ):
                conjugated.append(chi)
        if len(conjugated) == 1:
            return {op: complex(value) for op, value in conjugated[0].items()}
        return None

    def _wrap_phase(self, k: np.ndarray, op: int) -> complex:
        """exp(+2*pi*i k.delta) for the wrapped-away lattice translation."""
        return complex(
            np.exp(2j * np.pi * float(k @ self.wrap_delta[op]) / DEN)
        )

    # -------------------------------------------- non-tabulated (line) k points

    def _star_of_vector(self, k_int: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """Arms and coset-representative indices of the star of an arbitrary
        k vector (units of 1/DEN), in the same q-convention as ``star``."""
        arms = [tuple(np.mod(k_int, DEN))]
        representatives = [self._rotation_index[self._key(np.eye(3))]]
        for i in range(self.n_ops):
            q = tuple((np.array(arms[0]) @ self.inverse_rotations[i]) % DEN)
            if q not in arms:
                arms.append(q)
                representatives.append(i)
        return np.array(arms, dtype=np.int64), representatives

    def computed_irreps_at(self, k_int: np.ndarray) -> list:
        """Small irreps at an arbitrary k point, computed with spgrep.

        Args:
            k_int: k vector in the primitive basis, integers in units of
                ``1/DEN``.

        Returns:
            A list with one dict per small irrep.  Each dict holds ``"chi"``
            (``{op_index: character}``), ``"dim"`` (the dimension) and
            ``"small"`` (``{op_index: matrix}``), keyed by this algebra's
            operation indices (the little group of k).

        Raises:
            SystemExit: spgrep could not compute the irreps at this k point,
                or its little group disagrees with the q-convention one.
        """
        key = tuple(np.mod(k_int, DEN))
        if key in self._computed_cache:
            return self._computed_cache[key]
        from .runtime_compat import get_spacegroup_irreps_from_primitive_symmetry

        kpoint = np.array(key, dtype=float) / DEN
        try:
            irreps, mapping = get_spacegroup_irreps_from_primitive_symmetry(
                rotations=self.rotations,
                translations=np.array(self.translations, dtype=float) / DEN,
                kpoint=kpoint,
            )
        except Exception as exc:  # spgrep raises plain ValueError on rare k points
            raise SystemExit(
                f"ERROR: spgrep could not compute the small irreps at k={kpoint} "
                f"for space group {self.sg_type.number}: {exc}"
            ) from exc
        mapping = [int(m) for m in np.asarray(mapping).ravel()]
        expected = set(self.little_group(np.array(key, dtype=np.int64)))
        if set(mapping) != expected:
            raise SystemExit(
                f"ERROR: spgrep little group at k={kpoint} does not match "
                "the q-convention little group."
            )
        result = []
        for matrices in irreps:
            matrices = np.asarray(matrices)
            chi = {
                op_index: complex(np.trace(matrices[j]))
                for j, op_index in enumerate(mapping)
            }
            result.append({
                "chi": chi,
                "dim": int(matrices.shape[1]),
                # the matrices themselves, for building full induced irreps
                # at non-tabulated k points (symmetry-mode analysis)
                "small": {
                    op_index: np.asarray(matrices[j])
                    for j, op_index in enumerate(mapping)
                },
            })
        self._computed_cache[key] = result
        return result

    def induced_characters_at(self, k_int: np.ndarray, small: dict) -> tuple[np.ndarray, np.ndarray]:
        """Induced characters of a computed small irrep at an arbitrary k.

        Args:
            k_int: k vector in the primitive basis, integers in units of
                ``1/DEN``.
            small: One entry of ``computed_irreps_at(k_int)`` (only its
                ``"chi"`` is used).

        Returns:
            ``(arms, C)`` with the same structure as ``induced_characters``.
        """
        k = np.mod(np.asarray(k_int, dtype=np.int64), DEN)
        arms, representatives = self._star_of_vector(k)
        chi = small["chi"]
        C = np.zeros((self.n_ops, len(arms)), dtype=np.complex128)
        for a, s in enumerate(representatives):
            W_s, v_s = self.rotations[s], self.translations[s]
            W_s_inv = self.inverse_rotations[s]
            v_s_inv = -W_s_inv @ v_s
            for i in range(self.n_ops):
                W_h = W_s_inv @ self.rotations[i] @ W_s
                m = self._rotation_index[self._key(W_h)]
                if m not in chi:
                    continue
                tau_h = W_s_inv @ (self.rotations[i] @ v_s + self.translations[i]) + v_s_inv
                t_extra = tau_h - self.translations[m]
                if np.any(t_extra % DEN != 0):
                    raise SystemExit("ERROR: broken coset bookkeeping (non-lattice residue).")
                phase = np.exp(SIGMA * 2j * np.pi * float(k @ (t_extra // DEN)) / DEN)
                C[i, a] = phase * chi[m]
        return arms, C

    # ------------------------------------------------------- main computation

    def decompose_product(self, labels: list[str]):
        """Decompose the direct product of the full irreps named by labels.

        The computation behind ``crystod-group --product IRREP... --sg SG``:
        the reduction coefficients are character inner products over the
        finite factor group, with the momentum-conservation condition
        ``k1_a + k2_b = k3_c`` (modulo the reciprocal lattice) over the star
        arms.  Product terms at non-tabulated k points are computed with
        spgrep and named from the ISO-IR tables.

        Args:
            labels: ISO-IR labels of the factors, e.g. ``["R4-", "R5+"]``.

        Returns:
            ``(factors, terms, leftovers)``: ``factors`` are the resolved
            irrep records; ``terms`` is a list of ``(kname, irrep,
            multiplicity)`` where ``irrep`` has ``name``, ``dim`` and
            ``kpname`` (a tabulated ISO-IR irrep or a computed line irrep);
            ``leftovers`` lists the k vectors (units of ``1/DEN``) that could
            not be decomposed at all.

        Raises:
            SystemExit: A label is not tabulated for this space group.

        Example:
            >>> from crystod import group
            >>> algebra = group.SpaceGroupIrrepAlgebra("Pm-3m")
            >>> factors, terms, left = algebra.decompose_product(["R4-", "R5+"])
            >>> [(irrep.name, n) for _, irrep, n in terms]
            [('GM2-', 1), ('GM3-', 1), ('GM4-', 1), ('GM5-', 1)]
        """
        factors = [self.find_irrep(label) for label in labels]
        factor_data = [self.induced_characters(irrep) for irrep in factors]

        # candidate k3 vectors: sums of one arm from each factor (mod 1)
        sums = np.zeros((1, 3), dtype=np.int64)
        for arms, _ in factor_data:
            sums = (sums[:, None, :] + arms[None, :, :]).reshape(-1, 3) % DEN
        candidate_vectors = {tuple(vector) for vector in sums}

        terms = []
        covered: set[tuple] = set()

        # 1. tabulated special points (ISO-IR labels)
        for kname in self.k_by_kname:
            arms, _ = self.star(kname)
            arm_set = {tuple(arm) for arm in arms}
            if not (arm_set & candidate_vectors):
                continue
            star_terms = []
            integral = True
            for irrep in self.irreps_by_kname[kname]:
                try:
                    arms3, C3 = self.induced_characters(irrep)
                except SystemExit:
                    # unusable table entry at a candidate result star: fall
                    # through to the computed (spgrep) route for this star
                    integral = False
                    break
                multiplicity = self._multiplicity_core(factor_data, arms3, C3)
                if multiplicity is None:
                    integral = False
                    break
                if multiplicity:
                    star_terms.append((kname, irrep, multiplicity))
            if integral:
                terms.extend(star_terms)
                covered |= arm_set
            # non-integral tabulated decomposition (e.g. paired "physical"
            # irreps): fall through to the computed small irreps below

        # 2. remaining stars: compute small irreps on the fly (spgrep)
        remaining = sorted(candidate_vectors - covered)
        while remaining:
            representative = np.array(remaining[0], dtype=np.int64)
            arms, _ = self._star_of_vector(representative)
            canonical = np.array(min(tuple(arm) for arm in arms), dtype=np.int64)
            arms, _ = self._star_of_vector(canonical)
            arm_set = {tuple(arm) for arm in arms}
            point_name, names, label_source = self._line_names(canonical)
            for index, small in enumerate(self.computed_irreps_at(canonical)):
                arms3, C3 = self.induced_characters_at(canonical, small)
                multiplicity = self._multiplicity_core(factor_data, arms3, C3)
                if multiplicity is None:
                    raise SystemExit(
                        "ERROR: non-integer multiplicity in the computed "
                        f"decomposition at k={tuple(canonical)} (bug)."
                    )
                if multiplicity:
                    name = names[index] if names else f"{point_name}({index + 1})"
                    terms.append(
                        (
                            point_name,
                            _ComputedIrrep(
                                name, small["dim"], point_name, canonical,
                                len(arms), label_source,
                            ),
                            multiplicity,
                        )
                    )
            covered |= arm_set
            remaining = sorted(set(remaining) - arm_set)

        leftovers = sorted(candidate_vectors - covered)
        return factors, terms, leftovers

    def decompose_square(self, label: str, kind: str = "symmetric") -> SquareDecomposition:
        """Decompose the symmetric or antisymmetric square of a full irrep.

        The computation behind ``crystod-group --product IR IR --sg SG
        --symmetric|--antisymmetric``.  The order-parameter matrices of
        ``InducedRepresentation`` (the physically irreducible real form:
        ``D + D*`` for complex- and pseudoreal-type irreps) give the
        characters ``(chi(g)^2 +- chi(g^2)) / 2`` with
        ``chi(g^2) = tr M(g)^2`` on every element of the finite factor group
        ``G / T_N``; these are reduced by character inner products with the
        induced characters of the irreps at the stars of ``k_a + k_b``
        (tabulated special points first, other stars computed with spgrep
        and named from the ISO-IR tables, as in ``decompose_product``).

        Args:
            label: ISO-IR label of the irrep, e.g. ``"R4+"``.
            kind: ``"symmetric"`` or ``"antisymmetric"``.

        Returns:
            A ``SquareDecomposition`` record.

        Raises:
            SystemExit: The label is not tabulated, or its order-parameter
                matrices are not available (``InducedRepresentation``).

        Example:
            >>> from crystod import group
            >>> algebra = group.SpaceGroupIrrepAlgebra("Pm-3m")
            >>> square = algebra.decompose_square("R4+", "symmetric")
            >>> [(irrep.name, n) for _, irrep, n in square.terms]
            [('GM1+', 1), ('GM3+', 1), ('GM5+', 1)]
        """
        if kind not in ("symmetric", "antisymmetric"):
            raise SystemExit(f"ERROR: unknown square kind {kind!r}.")
        from .isotropy_subgroup import InducedRepresentation

        rep = InducedRepresentation(self, label)
        ops = np.array([i for i, _, _ in rep.elements], dtype=np.int64)
        shifts = np.array([t for _, t, _ in rep.elements], dtype=np.int64)
        chi = np.array([np.trace(matrix) for _, _, matrix in rep.elements], dtype=float)
        chi2 = np.array(
            [np.trace(matrix @ matrix) for _, _, matrix in rep.elements], dtype=float
        )
        sign = 1.0 if kind == "symmetric" else -1.0
        character = (chi**2 + sign * chi2) / 2.0
        order = len(rep.elements)

        def multiplicity(arms3: np.ndarray, C3: np.ndarray) -> int | None:
            phases = np.exp(SIGMA * 2j * np.pi * (shifts @ arms3.T) / DEN)
            chi3 = np.sum(C3[ops] * phases, axis=1)
            value = np.sum(character * np.conj(chi3)) / order
            if abs(value.imag) > 1e-6 or abs(value.real - round(value.real)) > 1e-6:
                return None
            result = int(round(value.real))
            return result if result >= 0 else None

        # candidate k vectors: q_a + q_b over the arms of the real form
        # (the arms and, for D + D*, their negatives)
        arms = np.asarray(rep.arms, dtype=np.int64) % DEN
        if rep.doubled:
            arms = np.unique(np.concatenate([arms, (-arms) % DEN]), axis=0)
        sums = (arms[:, None, :] + arms[None, :, :]).reshape(-1, 3) % DEN
        candidate_vectors = {tuple(vector) for vector in sums}

        terms = []
        covered: set[tuple] = set()
        for kname in self.k_by_kname:
            star_arms, _ = self.star(kname)
            arm_set = {tuple(arm) for arm in star_arms}
            if not (arm_set & candidate_vectors):
                continue
            star_terms = []
            integral = True
            for irrep in self.irreps_by_kname[kname]:
                try:
                    arms3, C3 = self.induced_characters(irrep)
                except SystemExit:
                    integral = False
                    break
                count = multiplicity(arms3, C3)
                if count is None:
                    integral = False
                    break
                if count:
                    star_terms.append((kname, irrep, count))
            if integral:
                terms.extend(star_terms)
                covered |= arm_set

        remaining = sorted(candidate_vectors - covered)
        while remaining:
            representative = np.array(remaining[0], dtype=np.int64)
            star_arms, _ = self._star_of_vector(representative)
            canonical = np.array(min(tuple(arm) for arm in star_arms), dtype=np.int64)
            star_arms, _ = self._star_of_vector(canonical)
            arm_set = {tuple(arm) for arm in star_arms}
            point_name, names, label_source = self._line_names(canonical)
            for index, small in enumerate(self.computed_irreps_at(canonical)):
                arms3, C3 = self.induced_characters_at(canonical, small)
                count = multiplicity(arms3, C3)
                if count is None:
                    raise SystemExit(
                        "ERROR: non-integer multiplicity in the computed "
                        f"square decomposition at k={tuple(canonical)} (bug)."
                    )
                if count:
                    name = names[index] if names else f"{point_name}({index + 1})"
                    terms.append((
                        point_name,
                        _ComputedIrrep(name, small["dim"], point_name, canonical,
                                       len(star_arms), label_source),
                        count,
                    ))
            covered |= arm_set
            remaining = sorted(set(remaining) - arm_set)

        n = rep.dimension
        return SquareDecomposition(
            label=rep.label,
            irrep=rep.irrep,
            kind=kind,
            dimension=n * (n + 1) // 2 if kind == "symmetric" else n * (n - 1) // 2,
            rep_dimension=n,
            doubled=rep.doubled,
            fs_type=getattr(rep, "fs_type", None) if rep.doubled else None,
            terms=terms,
            leftovers=sorted(candidate_vectors - covered),
        )

    def _isoir_labeler_inputs(self):
        """(cell, conventional rotations, conventional translations) for the
        ISO-IR labeler, or None when construction failed.

        The synthetic conventional cell carries exactly this space group's
        symmetry, so spglib can determine the transformation into the
        ISOTROPY standard setting (same route as the structure-based
        commands).  The conventional translations are recovered from the
        primitive ones through the exact column-convention inverse
        t_conv = M t_prim, so each (R, t) pair denotes precisely the
        operation whose characters spgrep computed (residual differences
        are centring-lattice translations, which the labeler
        phase-corrects)."""
        if self._isoir_cell is None:
            try:
                conventional_rotations = np.array(
                    [sym.R for sym in self.table.symmetries], dtype=float
                )
                cell = _synthetic_conventional_cell(
                    self.sg_type,
                    conventional_rotations,
                    np.array([sym.t for sym in self.table.symmetries], dtype=float),
                )
                conventional_translations = (
                    self.primitive_matrix
                    @ (np.array(self.translations, dtype=float) / DEN).T
                ).T
                self._isoir_cell = (
                    cell,
                    np.rint(conventional_rotations).astype(int),
                    conventional_translations,
                )
            except Exception:
                self._isoir_cell = False
        return self._isoir_cell or None

    def _isoir_line_labels(self, canonical: np.ndarray) -> tuple[str, list[str] | None] | None:
        """ISO-IR (ISOTROPY, Miller-Love) names of the computed small irreps
        at a non-tabulated k point: (k-type label, names) on a full match,
        (k-type label, None) when only the k-vector type is identified, or
        None.

        The names are those of the ISO-IR labeller of every other command
        (``crystod.isoir.IsoIRLabeler.label_characters``).  In a group
        without inversion the stars of k and -k can be distinct; the one
        ISO-IR tabulates (on a line or plane: the one at the canonical,
        positive parameter) keeps the tabulated name, and the other is named
        as the ISOTROPY software names it, with the conjugate irreps: LE1
        for the conjugate of LD1, DU for DT, SN for SM, GQ for GP, the
        suffix A (C in a few hexagonal groups) otherwise, so the two stars
        of one product keep distinct names."""
        key = tuple(np.mod(canonical, DEN))
        if key in self._isoir_line_cache:
            return self._isoir_line_cache[key]
        result = self._isoir_line_labels_uncached(np.asarray(key, dtype=np.int64))
        self._isoir_line_cache[key] = result
        return result

    def _isoir_line_labels_uncached(self, canonical: np.ndarray):
        from .isoir import get_isoir_kpoint_name, get_isoir_label_map

        inputs = self._isoir_labeler_inputs()
        if inputs is None:
            return None
        cell, conventional_rotations, conventional_translations = inputs
        try:
            smalls = self.computed_irreps_at(canonical)
        except SystemExit:
            return None
        k_conv = (np.asarray(canonical, dtype=float) / DEN) @ np.linalg.inv(
            self.primitive_matrix
        )
        little = sorted(self.little_group(canonical))
        little_rotations = [conventional_rotations[i] for i in little]
        little_translations = [conventional_translations[i] for i in little]
        characters = [
            np.array([small["chi"][i] for i in little], dtype=np.complex128)
            for small in smalls
        ]
        result = get_isoir_label_map(
            self.sg_type.number, cell, 1e-5, k_conv,
            little_rotations, little_translations, characters,
        )
        if result is not None and len(result[0]) == len(smalls):
            label_map, ktype = result
            return ktype, [label_map[index] for index in range(len(smalls))]
        # no irrep-level match: identify at least the k-vector type letter
        name = get_isoir_kpoint_name(self.sg_type.number, cell, 1e-5, k_conv)
        if name is not None:
            return name, None
        return None

    def isoir_display_arm(self, canonical: np.ndarray) -> np.ndarray:
        """Star arm in the tabulated ISO-IR parametrization, for display.

        Args:
            canonical: Representative arm of a non-tabulated star (units of
                ``1/DEN``).

        Returns:
            The arm matching arm 0 of the ISO-IR k-vector type (e.g. ``SM``
            as ``(a,a,0)`` rather than ``(0,a,a)``); ``canonical`` itself
            when no ISO-IR entry matches.
        """
        inputs = self._isoir_labeler_inputs()
        if inputs is None:
            return np.asarray(canonical, dtype=np.int64)
        from .isoir import get_cached_labeler

        labeler = get_cached_labeler(self.sg_type.number, inputs[0], 1e-5)
        if labeler is None:
            return np.asarray(canonical, dtype=np.int64)
        arms, _ = self._star_of_vector(np.asarray(canonical, dtype=np.int64))
        M_inv = np.linalg.inv(self.primitive_matrix)
        entries = [
            ir for ir in labeler.irreps if ir.num_free_params > 0
        ]
        best = None  # (num_free_params, sum |params|, arm tuple, arm)
        for arm in arms:
            k_conv = labeler.conventional_k((np.asarray(arm) / DEN) @ M_inv)
            for entry in entries:
                match = entry.match_k(k_conv)
                if match is None or match[0] != 0:
                    continue
                params = match[1]
                # prefer positive parameters inside the first zone
                if np.any(params < -1e-9) or np.any(params > 0.5 + 1e-9):
                    penalty = 1.0
                else:
                    penalty = 0.0
                key = (
                    entry.num_free_params,
                    penalty,
                    float(np.sum(np.abs(params))),
                    tuple(int(v) for v in arm),
                )
                if best is None or key < best[0]:
                    best = (key, np.asarray(arm, dtype=np.int64))
        return best[1] if best is not None else np.asarray(canonical, dtype=np.int64)

    def _line_names(self, canonical: np.ndarray) -> tuple[str, list[str] | None, str | None]:
        """Display name of a non-tabulated star, the names of its small
        irreps (or None) and the naming source ("isoir" or None).

        Priority: the name of a tabulated star reached through the computed
        route (paired "physical" irreps); then the ISO-IR (ISOTROPY,
        Miller-Love) tables; positional names as the last resort.
        """
        # check whether this star is a tabulated star (paired-irrep fallback):
        for kname in self.k_by_kname:
            arms, _ = self.star(kname)
            if any(np.all((arm - canonical) % DEN == 0) for arm in arms):
                return kname, None, None
        isoir = self._isoir_line_labels(canonical)
        if isoir is not None:
            point_name, names = isoir
            return point_name, names, "isoir" if names is not None else None
        coordinates = ",".join(_format_fraction(v) for v in canonical)
        return f"({coordinates})", None, None

    def _multiplicity_core(self, factor_data, arms3, C3) -> int | None:
        """Reduction coefficient; None when it is not a non-negative integer."""
        arm_lists = [arms for arms, _ in factor_data]
        combos: list[tuple[int, ...]] = [()]
        for arms in arm_lists:
            combos = [indices + (a,) for indices in combos for a in range(len(arms))]
        valid: list[tuple[tuple[int, ...], int]] = []
        for indices in combos:
            total = np.zeros(3, dtype=np.int64)
            for arms, a in zip(arm_lists, indices):
                total = total + arms[a]
            for c in range(len(arms3)):
                if np.all((total - arms3[c]) % DEN == 0):
                    valid.append((indices, c))
                    break

        if not valid:
            return 0

        C_factors = [C for _, C in factor_data]
        total = 0.0 + 0.0j
        for i in range(self.n_ops):
            for indices, c in valid:
                value = np.conj(C3[i, c])
                if value == 0:
                    continue
                for C, a in zip(C_factors, indices):
                    value *= C[i, a]
                    if value == 0:
                        break
                total += value
        multiplicity = total / self.n_ops
        if abs(multiplicity.imag) > 1e-6 or abs(multiplicity.real - round(multiplicity.real)) > 1e-6:
            return None
        result = int(round(multiplicity.real))
        return result if result >= 0 else None

    def full_dimension(self, irrep) -> int:
        """Dimension of a full space-group irrep (star size times small dim).

        Args:
            irrep: A tabulated irrep record or a computed line irrep (as
                returned in the ``terms`` of ``decompose_product``).

        Returns:
            The number of star arms times the small-irrep dimension.
        """
        if isinstance(irrep, _ComputedIrrep):
            return irrep.star_size * int(irrep.dim)
        arms, _ = self.star(irrep.kpname)
        return len(arms) * int(irrep.dim)


# ------------------------------------------------------------------ reporting


def _format_fraction(value: int) -> str:
    fraction = Fraction(int(value), DEN)
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def format_square_block(algebra: SpaceGroupIrrepAlgebra, square: SquareDecomposition) -> list[str]:
    """Lines of the ``* Symmetric square *`` / ``* Antisymmetric square *``
    block of ``crystod-group --product IR IR --sg SG --symmetric``.

    Args:
        algebra: The ``SpaceGroupIrrepAlgebra`` of the space group.
        square: The record from ``SpaceGroupIrrepAlgebra.decompose_square``.

    Returns:
        The block as a list of lines: the title, a note for the doubled
        real form of a complex- or pseudoreal-type irrep, the decomposition
        (``[R4+ x R4+] = ...`` symmetric, ``{R4+ x R4+} = ...``
        antisymmetric) and the dimension check.
    """
    symmetric = square.kind == "symmetric"
    title = "Symmetric square" if symmetric else "Antisymmetric square"
    lines = [f"* {title} (full space-group irreps) *"]
    if square.doubled:
        lines.append(
            f"note: {square.irrep.name} is of {square.fs_type} type; the square "
            "is that of the physically irreducible real representation "
            f"{square.label} (D + D*, dimension {square.rep_dimension})"
        )
    pair = f"{square.label} x {square.label}"
    left = f"[{pair}]" if symmetric else f"{{{pair}}}"
    right = " + ".join(
        (f"{multiplicity}" if multiplicity > 1 else "") + irrep.name
        for _, irrep, multiplicity in square.terms
    )
    lines.append(f"{left} = {right if right else '(none)'}")
    dims = " + ".join(
        (f"{multiplicity}x" if multiplicity > 1 else "") + str(algebra.full_dimension(irrep))
        for _, irrep, multiplicity in square.terms
    )
    lines.append(f"dimension: {square.dimension}" + (f" = {dims}" if dims else ""))
    total = sum(
        multiplicity * algebra.full_dimension(irrep) for _, irrep, multiplicity in square.terms
    )
    if not square.leftovers and (
        total != square.dimension or any(m < 0 for _, _, m in square.terms)
    ):
        lines.append("WARNING: dimension mismatch - please report this case.")
    if square.leftovers:
        lines.append(
            "NOTE: part of the square lives at non-tabulated k point(s): "
            + "; ".join(
                "(" + ", ".join(_format_fraction(v) for v in vector) + ")"
                for vector in square.leftovers
            )
        )
    return lines


def format_product_report(
    algebra: SpaceGroupIrrepAlgebra, labels: list[str], squares: tuple[str, ...] = ()
) -> str:
    """Text report of a space-group direct product.

    Exactly what ``crystod-group --product IRREP... --sg SG`` prints: the
    space group, the k points and star sizes involved, the decomposition
    line, a dimension check (star size times small dimension on both sides)
    and the DIRPRO cross-validation reference.  With ``squares`` (two
    identical labels, ``--symmetric`` / ``--antisymmetric``), the blocks of
    ``format_square_block`` follow the dimension check and the reference
    comes last.

    Args:
        algebra: The ``SpaceGroupIrrepAlgebra`` of the space group.
        labels: ISO-IR labels of the factors, e.g. ``["R4-", "R5+"]``.
        squares: ``"symmetric"`` and/or ``"antisymmetric"``; requires two
            identical labels.

    Returns:
        The report as one string (no trailing newline).

    Raises:
        SystemExit: A label is not tabulated for this space group
            (``ValueError`` when called through ``crystod.group``).

    Example:
        >>> from crystod import group
        >>> algebra = group.SpaceGroupIrrepAlgebra("Pm-3m")
        >>> print(group.format_product_report(algebra, ["R4-", "R5+"]))
        * Space group *
        Pm-3m (No. 221)
        <BLANKLINE>
        * K points (primitive basis) *
        R: (1/2, 1/2, 1/2)   star of 1 arm(s)
        GM: (0, 0, 0)   star of 1 arm(s)
        <BLANKLINE>
        * Direct product (full space-group irreps) *
        R4- x R5+ = GM2- + GM3- + GM4- + GM5-
        ...
    """
    factors, terms, leftovers = algebra.decompose_product(labels)
    if squares and (len(labels) != 2 or labels[0] != labels[1]):
        raise SystemExit("ERROR: the symmetric and antisymmetric squares need "
                         "two identical irreps, e.g. R4+ R4+.")
    square_records = [algebra.decompose_square(labels[0], kind) for kind in squares]

    lines = []
    lines.append("* Space group *")
    lines.append(
        f"{algebra.sg_type.international_short} (No. {algebra.sg_type.number})"
    )
    lines.append("")
    lines.append("* K points (primitive basis) *")
    shown = []
    for irrep in factors:
        if irrep.kpname not in shown:
            shown.append(irrep.kpname)
    for _, irrep, _ in terms + [term for square in square_records for term in square.terms]:
        if irrep.kpname not in shown:
            shown.append(irrep.kpname)
    for kname in shown:
        if kname not in algebra.k_by_kname:
            continue
        k = algebra.k_by_kname[kname]
        arms, _ = algebra.star(kname)
        coordinates = ", ".join(_format_fraction(v) for v in k)
        lines.append(f"{kname}: ({coordinates})   star of {len(arms)} arm(s)")
    for _, irrep, _ in terms + [term for square in square_records for term in square.terms]:
        if isinstance(irrep, _ComputedIrrep) and irrep.kpname not in algebra.k_by_kname:
            coordinates = ", ".join(_format_fraction(v) for v in irrep.k_int)
            entry = (f"{irrep.kpname}: ({coordinates})   star of "
                     f"{irrep.star_size} arm(s)  [non-tabulated]")
            if entry not in lines:
                lines.append(entry)
    lines.append("")

    lines.append("* Direct product (full space-group irreps) *")
    left = " x ".join(irrep.name for irrep in factors)
    right_parts = []
    for _, irrep, multiplicity in terms:
        prefix = f"{multiplicity}" if multiplicity > 1 else ""
        right_parts.append(f"{prefix}{irrep.name}")
    lines.append(f"{left} = {' + '.join(right_parts) if right_parts else '(none)'}")
    lines.append("")

    product_dimension = 1
    for irrep in factors:
        product_dimension *= algebra.full_dimension(irrep)
    resolved = sum(
        multiplicity * algebra.full_dimension(irrep) for _, irrep, multiplicity in terms
    )
    dims_left = " x ".join(str(algebra.full_dimension(irrep)) for irrep in factors)
    dims_right = " + ".join(
        (f"{multiplicity}x" if multiplicity > 1 else "")
        + str(algebra.full_dimension(irrep))
        for _, irrep, multiplicity in terms
    )
    lines.append(f"* Dimension check (star size x small dim) *")
    lines.append(f"{dims_left} = {product_dimension} -> {dims_right} = {resolved}")
    if leftovers:
        lines.append("")
        lines.append(
            "NOTE: part of the product lives at non-tabulated k point(s): "
            + "; ".join(
                "(" + ", ".join(_format_fraction(v) for v in vector) + ")"
                for vector in leftovers
            )
        )
        lines.append(
            "The decomposition above covers only the tabulated special points."
        )
    elif resolved != product_dimension:
        lines.append("WARNING: dimension mismatch - please report this case.")
    for square in square_records:
        lines.append("")
        lines.extend(format_square_block(algebra, square))
    lines.append("")
    lines.append(
        "Cross-validated against the Bilbao Crystallographic Server DIRPRO:"
    )
    lines.append(
        "M. I. Aroyo et al., Acta Cryst. A62, 115-128 (2006); "
        "https://cryst.ehu.es/rep/dirpro.html"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Direct products of space-group irreps (ISO-IR labels)."
    )
    parser.add_argument("--space-group", required=True, help='e.g. "Pm-3m" or "P6_3/mmc".')
    parser.add_argument("--irreps", nargs="+", required=True, help="e.g. R4- R5+")
    parser.add_argument("--symmetric", action="store_true",
                        help="Also decompose the symmetric square (two identical irreps).")
    parser.add_argument("--antisymmetric", action="store_true",
                        help="Also decompose the antisymmetric square (two identical irreps).")
    args = parser.parse_args(argv)

    squares = tuple(kind for kind, wanted in (("symmetric", args.symmetric),
                                              ("antisymmetric", args.antisymmetric))
                    if wanted)
    algebra = SpaceGroupIrrepAlgebra(args.space_group)
    print()
    print(format_product_report(algebra, args.irreps, squares))
    print()


if __name__ == "__main__":
    main()
