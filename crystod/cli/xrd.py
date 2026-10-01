"""crystod-xrd: powder X-ray diffraction patterns.

The structure of a POSCAR is turned into its Bragg peak list (``h k l``,
multiplicity, ``d``, ``2theta``, intensity) and a broadened pattern drawn as
a PDF (backed by :mod:`crystod.xrd_pattern`).  ``--xraytype`` selects the
radiation, Cu K-alpha by default as the Ka1 + Ka2 doublet; ``--peak-profile``
selects the broadening, Lorentzian by default.
"""

from __future__ import annotations

from argparse import ArgumentParser, RawTextHelpFormatter

from ..xrd_pattern import (
    DEFAULT_PEAK_PROFILE,
    DEFAULT_TWO_THETA_RANGE,
    DEFAULT_WIDTH,
    DEFAULT_XRAY_TYPE,
    KALPHA_DOUBLETS,
    PEAK_PROFILES,
    WAVELENGTHS,
)
from .common import (
    CRYSTOD_CITATION,
    ExampleRequested,
    add_cell_argument,
    add_example_argument,
    add_output_argument,
    banner,
    run_example,
)

desc = """\
Compute the powder X-ray diffraction pattern of a crystal structure: the
Bragg peaks (h k l, multiplicity, d, 2theta, relative intensity) are printed
and written as a text table, and the pattern broadened with a Lorentzian or
Gaussian profile is drawn as a PDF with tick marks at the peak positions.

--xraytype names the radiation. A K-alpha doublet (CuKa, MoKa, ...) is the
Ka1 + Ka2 superposition with the 2:1 intensity ratio, as a laboratory
diffractometer records it; a single line (CuKa1, CuKb, ...) gives the
monochromatic pattern. The wavelengths are those of the RIETAN-FP manual.

# Command Examples:
crystod-xrd -c 221_PPOSCAR_ScF3
crystod-xrd -c 221_PPOSCAR_ScF3 --xraytype CuKa1 --peak-profile gaussian
crystod-xrd -c 221_PPOSCAR_ScF3 --xraytype MoKa --two-theta 5 60 --width 0.05
crystod-xrd -c 221_PPOSCAR_ScF3 --output ScF3_xrd --show
crystod-xrd --example ScF3   (bundled input; --example alone lists the names)
"""


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="crystod-xrd",
        description=f"{banner()}\n\n{desc}",
        epilog=CRYSTOD_CITATION,
        formatter_class=RawTextHelpFormatter,
    )
    add_cell_argument(parser)
    add_example_argument(parser, "crystod-xrd")
    parser.add_argument(
        "--xraytype",
        default=DEFAULT_XRAY_TYPE,
        metavar="TYPE",
        help="X-ray radiation (default: CuKa). Doublets, Ka1 + Ka2 in the\n"
        f"2:1 ratio: {', '.join(KALPHA_DOUBLETS)}.\n"
        "Single lines (wavelength in Angstrom):\n"
        + "\n".join(
            "  " + ", ".join(f"{line} ({WAVELENGTHS[line]:.7g})" for line in group)
            for group in (
                ("CuKa1", "CuKa2", "CuKb"),
                ("AgKa1", "AgKa2", "MoKa1", "MoKa2"),
                ("CoKa1", "CoKa2", "FeKa1", "FeKa2", "CrKa1", "CrKa2"),
            )
        ),
    )
    parser.add_argument(
        "--peak-profile",
        default=DEFAULT_PEAK_PROFILE,
        choices=PEAK_PROFILES,
        help="Broadening profile of the drawn pattern (default: lorentzian).",
    )
    parser.add_argument(
        "--two-theta",
        nargs=2,
        type=float,
        default=list(DEFAULT_TWO_THETA_RANGE),
        metavar=("MIN", "MAX"),
        help="2theta window in degrees for the peak list and the plot\n"
        "(default: 10 120).",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=DEFAULT_WIDTH,
        metavar="W",
        help="Peak width in degrees: the Lorentzian half width at half\n"
        "maximum, or the Gaussian standard deviation (default: 0.1).",
    )
    parser.add_argument(
        "--min-intensity",
        type=float,
        default=0.0,
        metavar="PERCENT",
        help="Drop reflections weaker than this relative intensity\n"
        "(default: 0, every reflection is kept).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the pattern in a matplotlib window as well as writing the PDF.",
    )
    add_output_argument(
        parser,
        "Output prefix: PREFIX.txt (peak table) and PREFIX.pdf (pattern).\n"
        "Default: XRD_{cell file}_{xraytype}.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Symmetry tolerance in Angstrom of the printed space-group symbol\n"
        "(default: 0.01); the intensities use the structure as given.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    import sys

    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except ExampleRequested as request:
        # --example: put the bundled input in place, then run the ordinary
        # command line it stands for (plus whatever else was given)
        main(run_example("crystod-xrd", request.name, argv))
        return

    dispatch_argv = [
        "--poscar", args.cell,
        "--xraytype", args.xraytype,
        "--peak-profile", args.peak_profile,
        "--two-theta", *(str(value) for value in args.two_theta),
        "--width", str(args.width),
        "--min-intensity", str(args.min_intensity),
    ]
    if args.output:
        dispatch_argv.extend(["--output", args.output])
    if args.tolerance is not None:
        dispatch_argv.extend(["--tolerance", str(args.tolerance)])
    if args.show:
        dispatch_argv.append("--show")

    from ..xrd_pattern import main as xrd_pattern_main

    xrd_pattern_main(dispatch_argv)


if __name__ == "__main__":
    main()
