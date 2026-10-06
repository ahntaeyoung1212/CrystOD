"""Powder X-ray diffraction patterns of a crystal structure (``crystod-xrd``).

The Bragg peaks of the structure in a POSCAR are computed with pymatgen's
:class:`~pymatgen.analysis.diffraction.xrd.XRDCalculator` (atomic scattering
factors, Lorentz-polarization factor, multiplicities); the peak list --
``h k l``, multiplicity, ``d`` spacing, ``2theta`` and relative intensity --
is printed, written as a text table, and broadened with a Gaussian or
Lorentzian profile into the pattern that is drawn as a PDF.

The radiation is chosen by name.  A K-alpha *line* (``CuKa1``, ``MoKa2``,
...) gives a single-wavelength pattern; a K-alpha *doublet* (``CuKa``,
``MoKa``, ...) superposes the Ka1 and Ka2 patterns with the 2:1 intensity
ratio, which is what a laboratory diffractometer without a monochromator
records.  The wavelengths are those tabulated in the RIETAN-FP manual.

Originally ``script/xrd_pattern_poscar.py`` (Y. Mochizuki, 2026); the
computation is unchanged, the broadening profiles are now normalized to
unit area so that the Gaussian and Lorentzian patterns share one scale.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

#: X-ray wavelengths in Angstrom, from the RIETAN-FP manual (F. Izumi and
#: K. Momma).  Keys are the names ``--xraytype`` accepts for a single line;
#: the doublets of :data:`KALPHA_DOUBLETS` are built from these.
WAVELENGTHS: dict[str, float] = {
    "CuKa1": 1.5405929,
    "CuKa2": 1.5444274,
    "CuKb": 1.392234,
    "AgKa1": 0.55942178,
    "AgKa2": 0.5638131,
    "MoKa1": 0.70931715,
    "MoKa2": 0.713607,
    "CoKa1": 1.788996,
    "CoKa2": 1.792835,
    "FeKa1": 1.936041,
    "FeKa2": 1.939973,
    "CrKa1": 2.289726,
    "CrKa2": 2.293651,
}

#: K-alpha doublets: ``name -> ((line, weight), (line, weight))``.  The Ka2
#: pattern enters with half the weight of Ka1, the 2:1 ratio of the two lines.
KALPHA_DOUBLETS: dict[str, tuple[tuple[str, float], ...]] = {
    anode + "Ka": ((anode + "Ka1", 1.0), (anode + "Ka2", 0.5))
    for anode in ("Cu", "Ag", "Mo", "Co", "Fe", "Cr")
}

#: Every name ``--xraytype`` accepts, doublets first.
XRAY_TYPES: tuple[str, ...] = (*KALPHA_DOUBLETS, *WAVELENGTHS)

#: The broadening profiles of :func:`smear_pattern`.
PEAK_PROFILES: tuple[str, ...] = ("lorentzian", "gaussian")

DEFAULT_XRAY_TYPE = "CuKa"
DEFAULT_PEAK_PROFILE = "lorentzian"
DEFAULT_TWO_THETA_RANGE = (10.0, 120.0)
DEFAULT_WIDTH = 0.1


def _three_index(hkl) -> tuple[int, int, int]:
    """The Miller indices ``(h, k, l)`` of a reflection.

    pymatgen's ``XRDCalculator`` writes the reflections of a hexagonal
    lattice (hexagonal and trigonal crystals, rhombohedral ones in the
    hexagonal setting) as Miller-Bravais indices ``(h, k, i, l)``.  The third
    index is redundant, ``i = -(h + k)``, and is dropped here, so that every
    lattice is indexed with ``(h, k, l)`` on the basis of its own cell.

    Raises:
        ValueError: Neither three indices nor four with ``i = -(h + k)``.
    """
    indices = tuple(int(index) for index in hkl)
    if len(indices) == 4 and indices[2] == -(indices[0] + indices[1]):
        return indices[0], indices[1], indices[3]
    if len(indices) != 3:
        raise ValueError(f"not Miller (h k l) or Miller-Bravais (h k i l) indices: {indices}")
    return indices


@dataclass(frozen=True)
class Peak:
    """One Bragg reflection of a computed pattern.

    Attributes:
        hkl: Miller indices ``(h, k, l)`` of the first family of planes
            contributing to the peak.  Miller-Bravais indices ``(h, k, i, l)``,
            which pymatgen gives for a hexagonal lattice, are stored without
            the redundant ``i = -(h + k)``.
        multiplicity: Multiplicity of that family.
        families: Every ``((h, k, l), multiplicity)`` that pymatgen merged
            into this peak because the ``d`` spacings coincide (cubic
            ``(3 0 0)`` and ``(2 2 1)``, for instance); the first entry is
            ``hkl``.
        d: Interplanar spacing in Angstrom.
        two_theta: Scattering angle ``2theta`` in degrees.
        intensity: Relative intensity, 100 for the strongest reflection of
            the line it belongs to (50 for the strongest Ka2 reflection of a
            doublet).
        line: The radiation line that produced the peak (``"CuKa1"``).
    """

    hkl: tuple[int, int, int]
    multiplicity: int
    families: tuple[tuple[tuple[int, int, int], int], ...]
    d: float
    two_theta: float
    intensity: float
    line: str

    def __post_init__(self) -> None:
        # (h k i l) of a hexagonal lattice -> (h k l), here and in every family
        object.__setattr__(self, "hkl", _three_index(self.hkl))
        object.__setattr__(self, "families", tuple(
            (_three_index(hkl), int(mult)) for hkl, mult in self.families))

    @property
    def families_label(self) -> str:
        """``"(3 0 0) x6 + (2 2 1) x24"``: every merged family with its multiplicity."""
        return " + ".join(f"({h} {k} {l}) x{mult}" for (h, k, l), mult in self.families)


@dataclass
class XRDPattern:
    """A computed powder pattern: the peak list plus what it was computed for.

    Attributes:
        peaks: The Bragg peaks in order of increasing ``2theta``.
        xray_type: The radiation name given (``"CuKa"``, ``"MoKa1"``, ...).
        wavelengths: ``line -> wavelength`` in Angstrom for every line that
            contributed (two entries for a doublet).
        formula: Reduced chemical formula of the structure.
        space_group: International symbol found by spglib through pymatgen.
        two_theta_range: The ``(min, max)`` window in degrees the peaks were
            collected in.
        structure_name: The file name the structure was read from, used in
            output names and plot legends (empty when built from a Structure).
    """

    peaks: list[Peak]
    xray_type: str
    wavelengths: dict[str, float]
    formula: str
    space_group: str
    two_theta_range: tuple[float, float]
    structure_name: str = ""
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def two_theta(self) -> NDArray[np.float64]:
        """The ``2theta`` positions of the peaks, in degrees."""
        return np.array([peak.two_theta for peak in self.peaks], dtype=float)

    @property
    def intensity(self) -> NDArray[np.float64]:
        """The relative intensities of the peaks."""
        return np.array([peak.intensity for peak in self.peaks], dtype=float)

    def table_lines(self) -> list[str]:
        """The peak table as printed by ``crystod-xrd``, one string per line."""
        header = (f"{'h':>4} {'k':>3} {'l':>3} {'mult':>5} {'d (A)':>11} "
                  f"{'2theta (deg)':>13} {'intensity':>10}  line")
        lines = [header]
        for peak in self.peaks:
            h, k, l = peak.hkl
            extra = ""
            if len(peak.families) > 1:
                extra = "   + " + " + ".join(
                    f"({a} {b} {c}) x{mult}" for (a, b, c), mult in peak.families[1:])
            lines.append(f"{h:>4} {k:>3} {l:>3} {peak.multiplicity:>5} {peak.d:>11.6f} "
                         f"{peak.two_theta:>13.4f} {peak.intensity:>10.3f}  {peak.line}{extra}")
        return lines


def _resolve_radiation(xray_type: str) -> tuple[tuple[str, float], ...]:
    """``((line, weight), ...)`` for a radiation name, case-insensitively.

    Raises:
        SystemExit: The name is neither a tabulated line nor a doublet.
    """
    lookup = {name.lower(): name for name in XRAY_TYPES}
    name = lookup.get(xray_type.strip().lower())
    if name is None:
        raise SystemExit(
            f"ERROR: unknown X-ray type '{xray_type}'. Doublets (Ka1 + Ka2, 2:1): "
            f"{', '.join(KALPHA_DOUBLETS)}; single lines: {', '.join(WAVELENGTHS)}."
        )
    if name in KALPHA_DOUBLETS:
        return KALPHA_DOUBLETS[name]
    return ((name, 1.0),)


@contextlib.contextmanager
def _intensity_floor(minimum: float):
    """Temporarily set pymatgen's peak-intensity cut-off (a class attribute).

    ``XRDCalculator.get_pattern`` drops reflections whose scaled intensity
    falls below ``AbstractDiffractionPatternCalculator.SCALED_INTENSITY_TOL``
    (0.1 % by default) and reads that value from the class, so it has to be
    patched there; the original value is restored afterwards.
    """
    from pymatgen.analysis.diffraction.core import AbstractDiffractionPatternCalculator

    original = AbstractDiffractionPatternCalculator.SCALED_INTENSITY_TOL
    AbstractDiffractionPatternCalculator.SCALED_INTENSITY_TOL = minimum
    try:
        yield
    finally:
        AbstractDiffractionPatternCalculator.SCALED_INTENSITY_TOL = original


def load_structure(path: str):
    """Read a POSCAR into a pymatgen ``Structure``.

    Args:
        path: The structure file (VASP POSCAR/CONTCAR format).

    Returns:
        The ``pymatgen.core.Structure``.

    Raises:
        SystemExit: The file does not exist or cannot be parsed (``ValueError``
            when called through ``crystod.xrd``).
    """
    from pymatgen.io.vasp.inputs import Poscar

    from .vasp_io import poscar_structure

    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: No POSCAR named {path}!")
    try:
        # the cell VASP reads from the file (scale line, Cartesian coordinates)
        return poscar_structure(
            Poscar.from_file(path, read_velocities=False).structure, path)
    except Exception as exc:  # pymatgen raises several types for a bad file
        raise SystemExit(f"ERROR: failed to read POSCAR file '{path}': {exc}") from None


def compute_xrd_pattern(
    structure,
    xray_type: str = DEFAULT_XRAY_TYPE,
    two_theta_range: tuple[float, float] = DEFAULT_TWO_THETA_RANGE,
    *,
    min_intensity: float = 0.0,
    symprec: float = 0.01,
    structure_name: str = "",
) -> XRDPattern:
    """Compute the powder pattern of a structure for one radiation.

    This is ``crystod-xrd -c POSCAR --xraytype TYPE`` without the files.
    Each line of the radiation is run through pymatgen's
    :class:`~pymatgen.analysis.diffraction.xrd.XRDCalculator` (intensities
    scaled to 100 for the strongest peak); for a doublet the Ka2 peaks are
    weighted by 1/2 and the two lists are merged in order of ``2theta``.

    Args:
        structure: A ``pymatgen.core.Structure`` (see :func:`load_structure`).
        xray_type: A K-alpha doublet (``"CuKa"``, the default, ``"MoKa"``,
            ...) or a single line (``"CuKa1"``, ``"CuKb"``, ...);
            :data:`XRAY_TYPES` lists them, case-insensitively.
        two_theta_range: ``(min, max)`` in degrees; peaks outside are dropped.
        min_intensity: Peaks whose scaled intensity is below this percentage
            are dropped. ``0`` keeps every reflection (pymatgen's own default
            would hide those below 0.1 %).
        symprec: Tolerance in Angstrom of the space-group determination used
            for the printed symbol only; the intensities are computed from
            the structure as given, without symmetrization.
        structure_name: Name recorded in the result for output names and
            legends (the CLI passes the file name).

    Returns:
        The :class:`XRDPattern`.

    Raises:
        SystemExit: Unknown radiation name (``ValueError`` when called
            through ``crystod.xrd``).

    Example:
        >>> from crystod import xrd
        >>> from crystod.examples import example_path
        >>> s = xrd.load_structure(str(example_path("221_PPOSCAR_ScF3")))
        >>> pattern = xrd.compute_xrd_pattern(s, "CuKa1", (10, 60))
        >>> pattern.space_group, len(pattern.peaks)
        ('Pm-3m', 6)
        >>> peak = pattern.peaks[0]
        >>> peak.hkl, round(peak.two_theta, 2), round(peak.intensity, 1)
        ((1, 0, 0), 21.82, 100.0)
    """
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.core.composition import Composition
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    radiation = _resolve_radiation(xray_type)
    low, high = float(two_theta_range[0]), float(two_theta_range[1])
    if not 0.0 <= low < high <= 180.0:
        raise SystemExit(
            f"ERROR: the 2theta range must satisfy 0 <= min < max <= 180 degrees "
            f"(got {two_theta_range[0]} {two_theta_range[1]})."
        )

    peaks: list[Peak] = []
    wavelengths: dict[str, float] = {}
    with _intensity_floor(min_intensity):
        for line, weight in radiation:
            wavelength = WAVELENGTHS[line]
            wavelengths[line] = wavelength
            pattern = XRDCalculator(wavelength=wavelength).get_pattern(
                structure, two_theta_range=(low, high))
            for two_theta, intensity, families, d in zip(
                    pattern.x, pattern.y, pattern.hkls, pattern.d_hkls):
                merged = tuple((tuple(int(i) for i in fam["hkl"]), int(fam["multiplicity"]))
                               for fam in families)
                peaks.append(Peak(
                    hkl=merged[0][0], multiplicity=merged[0][1], families=merged,
                    d=float(d), two_theta=float(two_theta),
                    intensity=float(intensity) * weight, line=line,
                ))
    peaks.sort(key=lambda peak: (peak.two_theta, peak.line))

    try:
        space_group = SpacegroupAnalyzer(structure, symprec=symprec).get_space_group_symbol()
    except Exception:  # spglib found no symmetry at this tolerance
        space_group = "P1"
    formula = Composition(structure.formula).reduced_formula
    name = _resolve_name(xray_type)
    return XRDPattern(peaks=peaks, xray_type=name, wavelengths=wavelengths,
                      formula=formula, space_group=space_group,
                      two_theta_range=(low, high), structure_name=structure_name)


def _resolve_name(xray_type: str) -> str:
    """The canonical spelling of a radiation name (``"cuka"`` -> ``"CuKa"``)."""
    lookup = {name.lower(): name for name in XRAY_TYPES}
    return lookup[xray_type.strip().lower()]


def smear_pattern(
    pattern: XRDPattern,
    profile: str = DEFAULT_PEAK_PROFILE,
    width: float = DEFAULT_WIDTH,
    npoints: int = 5000,
    two_theta_range: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Broaden the peaks into a continuous pattern.

    Every peak becomes a unit-area profile of the same width scaled by its
    intensity: a Lorentzian ``(w/pi) / ((x - x0)^2 + w^2)`` or a Gaussian
    ``exp(-(x - x0)^2 / (2 w^2)) / (w sqrt(2 pi))``.  The two profiles share
    the same integrated intensity per peak, so they can be compared directly.

    Args:
        pattern: The peaks, from :func:`compute_xrd_pattern`.
        profile: ``"lorentzian"`` (default) or ``"gaussian"``.
        width: The profile parameter ``w`` in degrees (Lorentzian half width
            at half maximum, Gaussian standard deviation); 0.1 by default.
        npoints: Number of ``2theta`` samples.
        two_theta_range: The window to sample; the pattern's own by default.

    Returns:
        ``(two_theta, intensity)`` arrays of length ``npoints``.

    Raises:
        SystemExit: Unknown profile name or non-positive width (``ValueError``
            when called through ``crystod.xrd``).
    """
    profile = profile.strip().lower()
    if profile not in PEAK_PROFILES:
        raise SystemExit(
            f"ERROR: unknown peak profile '{profile}'; choose one of {', '.join(PEAK_PROFILES)}.")
    if width <= 0:
        raise SystemExit("ERROR: the peak width must be positive.")
    low, high = two_theta_range if two_theta_range is not None else pattern.two_theta_range
    x = np.linspace(float(low), float(high), int(npoints))
    y = np.zeros_like(x)
    if not pattern.peaks:
        return x, y
    delta = x[:, None] - pattern.two_theta[None, :]
    if profile == "gaussian":
        shapes = np.exp(-delta**2 / (2.0 * width**2)) / (width * np.sqrt(2.0 * np.pi))
    else:
        shapes = (width / np.pi) / (delta**2 + width**2)
    y = shapes @ pattern.intensity
    return x, y


def write_peak_table(pattern: XRDPattern, path: str) -> str:
    """Write the peak list as a comma-separated text table.

    Columns: ``h, k, l, multiplicity, d, two_theta, intensity, line,
    families`` -- the last holds every merged family (``(3 0 0) x6 + (2 2 1)
    x24``) and is the only non-numeric column besides ``line``.  The header
    lines record the structure, the radiation and the wavelength(s).

    Args:
        pattern: The peaks to write.
        path: Output file name.

    Returns:
        ``path``.
    """
    lines = [
        f"# powder XRD pattern of {pattern.structure_name or pattern.formula}: "
        f"{pattern.formula}, {pattern.space_group}",
        "# radiation: " + pattern.xray_type + " = " + ", ".join(
            f"{line} {wavelength:.7f} A" for line, wavelength in pattern.wavelengths.items())
        + (" (Ka1 : Ka2 = 2 : 1)" if len(pattern.wavelengths) > 1 else ""),
        f"# 2theta range: {pattern.two_theta_range[0]:g} - {pattern.two_theta_range[1]:g} deg; "
        "intensities relative to the strongest Ka1 peak = 100",
        "# h, k, l, multiplicity, d (A), two_theta (deg), intensity, line, families",
    ]
    for peak in pattern.peaks:
        h, k, l = peak.hkl
        lines.append(f"{h},{k},{l},{peak.multiplicity},{peak.d:.6f},{peak.two_theta:.4f},"
                     f"{peak.intensity:.4f},{peak.line},{peak.families_label}")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def plot_xrd_pattern(
    pattern: XRDPattern,
    path: str,
    profile: str = DEFAULT_PEAK_PROFILE,
    width: float = DEFAULT_WIDTH,
    two_theta_range: tuple[float, float] | None = None,
    show: bool = False,
) -> str:
    """Draw the broadened pattern with tick marks at the Bragg positions.

    Args:
        pattern: The peaks to draw.
        path: Output file; the extension selects the format (``.pdf``,
            ``.png``, ``.svg``, ... as matplotlib supports).
        profile: Broadening profile, see :func:`smear_pattern`.
        width: Profile width in degrees, see :func:`smear_pattern`.
        two_theta_range: The window to draw; the pattern's own by default.
        show: Open an interactive matplotlib window as well.

    Returns:
        ``path``.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = smear_pattern(pattern, profile, width, two_theta_range=two_theta_range)
    top = float(y.max()) if y.size and y.max() > 0 else 1.0
    label = pattern.structure_name or pattern.formula
    lines = " + ".join(pattern.wavelengths)

    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.plot(x, y, color="tab:red", linewidth=0.8, label=f"{label} ({lines})")
    axis.vlines(pattern.two_theta, -top / 5, -top / 10, color="tab:red", linewidth=0.6)
    axis.set_xlim(x[0], x[-1])
    axis.set_ylim(-top / 4, top * 1.08)
    axis.set_xlabel(r"2$\theta$ (deg.)", fontsize=12)
    axis.set_ylabel("Intensity", fontsize=12)
    axis.tick_params(labelleft=False, labelsize=12)
    axis.set_title(f"{pattern.formula} ({pattern.space_group}), {pattern.xray_type}, "
                   f"{profile} profile", fontsize=11)
    axis.legend(fontsize=10, loc="upper right")
    figure.tight_layout()
    figure.savefig(path)
    if show:
        plt.show()
    plt.close(figure)
    return path


def build_parser():
    """The argument parser of the implementation entry point.

    ``crystod-xrd`` (``crystod.cli.xrd``) is the documented front end; this
    parser is what it dispatches to.
    """
    from argparse import ArgumentParser, RawTextHelpFormatter

    parser = ArgumentParser(
        prog="python -m crystod.xrd_pattern",
        description="Powder X-ray diffraction pattern of a POSCAR (crystod-xrd).",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--poscar", default="POSCAR", help="structure file (VASP POSCAR)")
    parser.add_argument("--xraytype", default=DEFAULT_XRAY_TYPE,
                        help="radiation: " + ", ".join(XRAY_TYPES))
    parser.add_argument("--peak-profile", default=DEFAULT_PEAK_PROFILE,
                        help="broadening profile: " + ", ".join(PEAK_PROFILES))
    parser.add_argument("--two-theta", nargs=2, type=float, default=list(DEFAULT_TWO_THETA_RANGE),
                        metavar=("MIN", "MAX"), help="2theta window in degrees")
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH,
                        help="profile width in degrees")
    parser.add_argument("--min-intensity", type=float, default=0.0,
                        help="drop peaks below this relative intensity (%%)")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="symmetry tolerance of the printed space group (Angstrom)")
    parser.add_argument("--output", default=None, metavar="PREFIX",
                        help="output prefix: PREFIX.txt and PREFIX.pdf")
    parser.add_argument("--show", action="store_true", help="open the matplotlib window")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point behind ``crystod-xrd``."""
    args = build_parser().parse_args(argv)

    structure = load_structure(args.poscar)
    name = os.path.basename(args.poscar)
    pattern = compute_xrd_pattern(
        structure, args.xraytype, tuple(args.two_theta),
        min_intensity=args.min_intensity, symprec=args.tolerance, structure_name=name)
    profile = args.peak_profile.strip().lower()
    if profile not in PEAK_PROFILES:
        raise SystemExit(
            f"ERROR: unknown peak profile '{args.peak_profile}'; choose one of "
            f"{', '.join(PEAK_PROFILES)}.")

    print(f"\n * Structure *\n {name}: {pattern.formula}, {pattern.space_group}\n")
    radiation = ", ".join(f"{line} = {wl:.7f} A" for line, wl in pattern.wavelengths.items())
    if len(pattern.wavelengths) > 1:
        radiation += "  (Ka1 : Ka2 = 2 : 1)"
    print(f" * Radiation *\n {pattern.xray_type}: {radiation}\n")
    print(f" * Bragg peaks ({len(pattern.peaks)}) in {pattern.two_theta_range[0]:g} - "
          f"{pattern.two_theta_range[1]:g} deg *")
    if pattern.peaks:
        for line in pattern.table_lines():
            print(" " + line)
    else:
        print(" (no reflection in this window)")

    prefix = args.output or f"XRD_{name}_{pattern.xray_type}"
    table_path = write_peak_table(pattern, prefix + ".txt")
    figure_path = plot_xrd_pattern(pattern, prefix + ".pdf", profile, args.width, show=args.show)
    print("\n * Output files *")
    print(f" Peak table written to: {table_path}")
    print(f" Pattern ({profile} profile, width {args.width:g} deg) written to: {figure_path}")
    print("\nIntensities: pymatgen XRDCalculator (S. P. Ong et al., Comput. Mater. Sci. 68, "
          "314 (2013));\nwavelengths: RIETAN-FP manual (F. Izumi and K. Momma).")


if __name__ == "__main__":
    main()
