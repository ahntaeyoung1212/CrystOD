"""Direct products of space-group irreps (crystod-group --product --space-group).

Decomposes the direct product of full space-group irreps (induced from the
little-group irreps of high-symmetry k points) into full space-group irreps:

    chi_(k1,mu) x chi_(k2,nu) = sum over stars k3 in star(k1)+star(k2) of
                                n_(k3,lam) chi_(k3,lam)

The characters come from the CDML tables shipped with ``irreptables`` (the
same source as every other crystod irrep label), induced from the little
group to the full group over the star arms. The reduction coefficients are
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
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from phonopy.structure.cells import get_primitive_matrix_by_centring

from .basis_function import _resolve_space_group_type
from .dirpro_line_names import LINE_IRREP_NAMES
from .irreptables_compat import load_irreptables

IrrepTable, _Irrep = load_irreptables()

# global denominator for exact fractional arithmetic on k vectors and
# translations: every special-point coordinate tabulated in irreptables
# (all 230 space groups) and every space-group fractional translation is a
# multiple of 1/24, and sums of 1/24-grid vectors stay on the grid
DEN = 24

# sign convention of the translation phase in the CDML/irreptables small
# representations: D(W, v + t) = exp(SIGMA * 2j*pi * k.t) * D(W, v)
SIGMA = -1.0


class _ComputedIrrep:
    """Line-point small irrep computed on the fly (spgrep); mimics the
    irreptables irrep interface used by the report."""

    def __init__(self, name: str, dim: int, kpname: str, k_int, star_size: int):
        self.name = name
        self.dim = dim
        self.kpname = kpname
        self.k_int = np.asarray(k_int, dtype=np.int64)
        self.star_size = star_size


class _SyntheticIrrep:
    """Tabulated-like irrep synthesized from another one (e.g. the conjugate
    irreps of the -k star for polar space groups, CDML 'A' points)."""

    def __init__(self, name: str, dim: int, kpname: str, characters: dict):
        self.name = name
        self.dim = dim
        self.kpname = kpname
        self.characters = characters


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
    """Space-group symmetry + CDML irrep tables in the primitive basis,
    with runtime-verified conventions (group closure, little-group match)."""

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

        # translation transform convention (row vs column) is fixed by
        # requiring group closure of (W, v) modulo integer translations
        for candidate in (
            conventional_translations @ inverse,
            conventional_translations @ inverse.T,
        ):
            translations = np.mod(_snap(candidate), DEN)
            if self._is_closed(rotations, translations):
                break
        else:
            raise SystemExit("ERROR: could not build a closed primitive-setting space group.")

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

        self._add_minus_k_stars()

    def _add_minus_k_stars(self) -> None:
        """Synthesize the -k ('A') stars of polar space groups.

        For space groups without inversion, -k may belong to a star that is
        not tabulated in irreptables (e.g. PA of I-43m). The allowed small
        irreps at -k are the complex conjugates of those at k; CDML names
        them with an 'A' suffix on the k-point letter (P1 -> PA1).
        """
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
            new_kname = kname + "A"
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
        """Operation indices whose q-action leaves k invariant (mod 1)."""
        return [
            i
            for i in range(self.n_ops)
            if np.all((k @ self.inverse_rotations[i] - k) % DEN == 0)
        ]

    def star(self, kname: str) -> tuple[np.ndarray, list[int]]:
        """Arms of the star of the tabulated k point, in the q-convention
        q_a = k . W_{s_a}^{-1}, with the coset-representative indices s_a."""
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
        """(arms, C) of the full (induced) irrep.

        C[i, a] = chi_small(s_a^{-1} g_i s_a) when g_i fixes arm a, else 0.
        The full character of (g_i, t) is sum_a C[i, a] exp(SIGMA*2j*pi*q_a.t).
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
            # tabulated characters are not those of any allowed small irrep
            # (broken table entry, e.g. P of SG 230): use the fitted
            # computed small irrep carrying this CDML name instead
            substitute = self._substitute_broken_small(irrep, k)
            result = self.induced_characters_at(
                np.array(substitute["__at__"], dtype=np.int64),
                {"chi": substitute["chi"]},
            )
            self._induced_cache[cache_key] = result
            return result
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
        """Replace rounded table characters with the exact spgrep values.

        irreptables stores characters with a few decimals (e.g. 0.8660 for
        sqrt(3)/2), which is too coarse for exact reduction coefficients.
        When exactly one spgrep-computed small irrep at the same k matches
        the table characters within the rounding error, its exact characters
        are used instead. Paired ("physical") table irreps that correspond
        to a sum of two computed irreps are kept as tabulated. Returns None
        when the tabulated characters match NO allowed small irrep at all
        (broken table entry).
        """
        try:
            computed = self.computed_irreps_at(k)
        except SystemExit:
            return small
        matches = []
        for candidate in computed:
            chi = candidate["chi"]
            if set(chi.keys()) != set(small.keys()):
                continue
            if all(abs(chi[op] - small[op]) < 5e-3 for op in small):
                matches.append(chi)
        if len(matches) == 1:
            return {op: complex(value) for op, value in matches[0].items()}
        if matches:
            return small
        # sum-of-two pairing ("physical" combined irrep)?
        for i, c1 in enumerate(computed):
            for c2 in computed[i:]:
                if set(c1["chi"]) == set(small) and all(
                    abs(c1["chi"][op] + c2["chi"][op] - small[op]) < 5e-3 for op in small
                ):
                    return small
        # opposite translation-phase gauge: a few irreptables entries (P/N/W
        # quarter-k points of nonsymmorphic centred groups, e.g. W of Fd-3m)
        # store chi_table = chi_spgrep * exp(+4j*pi*k.v); identify the irrep
        # through that gauge and use the consistent-gauge exact characters
        flipped = []
        for candidate in computed:
            chi = candidate["chi"]
            if set(chi.keys()) != set(small.keys()):
                continue
            if all(
                abs(chi[op] * self._gauge_factor(k, op) - small[op]) < 5e-3
                for op in small
            ):
                flipped.append(chi)
        if len(flipped) == 1:
            return {op: complex(value) for op, value in flipped[0].items()}
        return None

    def _gauge_factor(self, k: np.ndarray, op: int) -> complex:
        return complex(
            np.exp(4j * np.pi * float(k @ self.translations[op]) / (DEN * DEN))
        )

    def _substitute_broken_small(self, irrep, k: np.ndarray) -> dict:
        arms, _ = self._star_of_vector(k)
        canonical = np.array(min(tuple(arm) for arm in arms), dtype=np.int64)
        entry = LINE_IRREP_NAMES.get((self.sg_type.number, tuple(canonical)))
        if entry is not None:
            _, name_map = entry
            for candidate in self.computed_irreps_at(canonical):
                fingerprint = _character_fingerprint(candidate["chi"])
                if name_map.get(fingerprint) == irrep.name:
                    # transport the small irrep from the canonical arm back to
                    # the tabulated representative arm is not needed: the
                    # canonical arm IS in the star of k, and the induced full
                    # irrep is arm-independent, so induce from canonical
                    return {"__at__": tuple(canonical), "chi": candidate["chi"]}
        raise SystemExit(
            f"ERROR: the irreptables characters of {irrep.name} "
            f"(space group {self.sg_type.number}) do not correspond to any "
            "allowed small representation, and no fitted substitute is "
            "available. Please report this case."
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
        """Small irreps at a non-tabulated k point, computed with spgrep.

        Returns a list of dicts {"chi": {op_index: character}, "dim": d}
        keyed by this algebra's operation indices.
        """
        key = tuple(np.mod(k_int, DEN))
        if key in self._computed_cache:
            return self._computed_cache[key]
        from spgrep.core import get_spacegroup_irreps_from_primitive_symmetry

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
            result.append({"chi": chi, "dim": int(matrices.shape[1])})
        self._computed_cache[key] = result
        return result

    def induced_characters_at(self, k_int: np.ndarray, small: dict) -> tuple[np.ndarray, np.ndarray]:
        """(arms, C) of the irrep induced from a computed small irrep at an
        arbitrary k point (same structure as ``induced_characters``)."""
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

        Returns (factors, terms, leftovers): factors = resolved irreps;
        terms = list of (kname, irrep-like, multiplicity) where irrep-like has
        .name/.dim/.kpname (a tabulated irreptables irrep or a computed line
        irrep); leftovers = k vectors (fractions of DEN) that could not be
        decomposed at all.
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

        # 1. tabulated special points (CDML labels from irreptables)
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
            point_name, names = self._line_names(canonical)
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
                            _ComputedIrrep(name, small["dim"], point_name, canonical, len(arms)),
                            multiplicity,
                        )
                    )
            covered |= arm_set
            remaining = sorted(set(remaining) - arm_set)

        leftovers = sorted(candidate_vectors - covered)
        return factors, terms, leftovers

    def _line_names(self, canonical: np.ndarray) -> tuple[str, list[str] | None]:
        """Display name of a non-tabulated star and (optionally) the CDML
        names of its small irreps, keyed by character fingerprints."""
        entry = LINE_IRREP_NAMES.get((self.sg_type.number, tuple(canonical)))
        smalls = self.computed_irreps_at(canonical)
        if entry is not None:
            point_name, name_map = entry
            names = []
            for small in smalls:
                fingerprint = _character_fingerprint(small["chi"])
                names.append(name_map.get(fingerprint))
            if all(name is not None for name in names):
                return point_name, names
            return point_name, None
        # check whether this star is a tabulated star (paired-irrep fallback):
        for kname in self.k_by_kname:
            arms, _ = self.star(kname)
            if any(np.all((arm - canonical) % DEN == 0) for arm in arms):
                return kname, None
        coordinates = ",".join(_format_fraction(v) for v in canonical)
        return f"({coordinates})", None

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


def format_product_report(algebra: SpaceGroupIrrepAlgebra, labels: list[str]) -> str:
    factors, terms, leftovers = algebra.decompose_product(labels)

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
    for _, irrep, _ in terms:
        if irrep.kpname not in shown:
            shown.append(irrep.kpname)
    for kname in shown:
        if kname not in algebra.k_by_kname:
            continue
        k = algebra.k_by_kname[kname]
        arms, _ = algebra.star(kname)
        coordinates = ", ".join(_format_fraction(v) for v in k)
        lines.append(f"{kname}: ({coordinates})   star of {len(arms)} arm(s)")
    for _, irrep, _ in terms:
        if isinstance(irrep, _ComputedIrrep) and irrep.kpname not in algebra.k_by_kname:
            coordinates = ", ".join(_format_fraction(v) for v in irrep.k_int)
            entry = f"{irrep.kpname}: ({coordinates})   star of {irrep.star_size} arm(s)  [non-tabulated]"
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
        description="Direct products of space-group irreps (CDML labels)."
    )
    parser.add_argument("--space-group", required=True, help='e.g. "Pm-3m" or "P6_3/mmc".')
    parser.add_argument("--irreps", nargs="+", required=True, help="e.g. R4- R5+")
    args = parser.parse_args(argv)

    algebra = SpaceGroupIrrepAlgebra(args.space_group)
    print()
    print(format_product_report(algebra, args.irreps))
    print()


if __name__ == "__main__":
    main()
