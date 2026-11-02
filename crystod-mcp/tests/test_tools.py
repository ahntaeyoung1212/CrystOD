"""crystod-mcp tests: every tool against known CrystOD results, plus the protocol.

The inputs are the files bundled with CrystOD (``crystod.examples``), read
as text the way an MCP client would pass them. Expected values are the ones
CrystOD's own testsuite checks (Pm-3m R4+ subgroups, SrTiO3 R-point phonons,
ScF3 Sc d irreps, CH4 point group, ...).

Run: ``python -m pytest -q tests`` (the phonon tools rebuild the 4x4x4
SrTiO3 force constants, a few seconds each).
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from crystod.examples import example_path
from crystod_mcp import tools, utils
from crystod_mcp.server import create_server
from crystod_mcp.tools import TOOL_NAMES

EXPECTED_TOOLS = (
    "crystod_isotropy_subgroups",
    "crystod_irrep_product",
    "crystod_decompose_representation",
    "crystod_ligand_field",
    "crystod_character_table",
    "crystod_invariants",
    "crystod_find_isotropy_irreps",
    "crystod_subgroup_graph",
    "crystod_correlate",
    "crystod_tensor_form",
    "crystod_phonon_irreps",
    "crystod_imaginary_mode_subgroups",
    "crystod_phonon_activity",
    "crystod_mag_classify",
    "crystod_crystal_orbital_irreps",
    "crystod_special_kpoints",
    "crystod_molecular_symmetry",
    "crystod_slab",
)

SCF3 = example_path("221_PPOSCAR_ScF3").read_text()
SRTIO3 = example_path("221_PPOSCAR_SrTiO3").read_text()
FORCE_SETS_SRTIO3 = example_path("FORCE_SETS_SrTiO3").read_text()
CH4 = example_path("XYZ_CH4.xyz").read_text()
NH3 = example_path("XYZ_NH3.xyz").read_text()

R4P_SUBGROUPS = ("I4/mcm", "R-3c", "Imma", "C2/m", "C2/c", "P-1")

# diamond Si (Fd-3m), primitive cell: one Raman-active optical set (GM5+, T2g)
SI_DIAMOND = """Si (Fd-3m), primitive cell
   1.0
     0.0000000000000000    2.7343639976914762    2.7343639976914762
     2.7343639976914762    0.0000000000000000    2.7343639976914762
     2.7343639976914762    2.7343639976914762    0.0000000000000000
Si
   2
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.2500000000000000  0.2500000000000000  0.2500000000000000
"""

# AlPO4 (I-4) with its origin moved to a general point, outside the ISO-IR
# setting: the table's P (1/4, 1/4, 1/4) is there the PA of the labels, and
# the point the labels call P is (-1/4, -1/4, -1/4)
ALPO4_SHIFTED = """AlPO4 (I-4), origin shifted by (0.0731, 0.1593, 0.2417)
   1.0
    -2.5416199990173731    2.5416199990173731    3.7261219986515584
     2.5416199990173731   -2.5416199990173745    3.7261219986515584
     2.5416199990173731    2.5416199990173745   -3.7261219986515584
Al P O
   1    1    4
Direct
  0.0731000000  0.1593000000  0.2417000000
  0.8231000000  0.4093000000  0.7417000000
  0.9364380000  0.3841090000  0.0649420000
  0.8482910000  0.7577330000  0.8802290000
  0.4746670000  0.1993960000  0.4184580000
  0.0330040000  0.2959620000  0.6031710000
"""


@pytest.fixture(scope="session", autouse=True)
def isotropy_cache(tmp_path_factory):
    """Keep the isotropy tables the reverse lookup builds out of the user's
    ``~/.cache`` unless ``CRYSTOD_CACHE_DIR`` names a cache already."""
    if os.environ.get("CRYSTOD_CACHE_DIR"):
        yield os.environ["CRYSTOD_CACHE_DIR"]
        return
    directory = str(tmp_path_factory.mktemp("crystod-cache"))
    os.environ["CRYSTOD_CACHE_DIR"] = directory
    try:
        yield directory
    finally:
        os.environ.pop("CRYSTOD_CACHE_DIR", None)


def run(coroutine):
    return asyncio.run(coroutine)


def assert_result(text: str) -> str:
    """Every result is Markdown text ending with the citation footer."""
    assert isinstance(text, str) and text.strip()
    assert text.rstrip().endswith(utils.CITATION)
    assert "iso.byu.edu" in text and "Phys. Rev. B 110, 064104" in text
    return text


def assert_error(coroutine, *fragments: str) -> str:
    with pytest.raises(ToolError) as excinfo:
        run(coroutine)
    message = str(excinfo.value)
    assert "\n" not in message.strip(), message
    for fragment in fragments:
        assert fragment in message, message
    return message


def tool_error(coroutine) -> str:
    """The one-line message of the ToolError a call ends in.

    A ``SystemExit`` or ``KeyboardInterrupt`` that escapes the tool wrapper
    (and with it ``asyncio.run``) fails the test instead of ending the run.
    """
    try:
        run(coroutine)
    except ToolError as exc:
        message = str(exc)
        assert "\n" not in message.strip(), message
        return message
    except (SystemExit, KeyboardInterrupt) as exc:
        pytest.fail(f"{type(exc).__name__} escaped the tool wrapper: {exc}")
    pytest.fail("the call returned a result instead of an error")


# --------------------------------------------------------------------------
# group
# --------------------------------------------------------------------------

def test_isotropy_subgroups_pm3m_r4p():
    out = assert_result(run(tools.crystod_isotropy_subgroups("Pm-3m", "R4+")))
    for symbol in R4P_SUBGROUPS:
        assert f"| {symbol} |" in out
    assert "| (0,0,a) | I4/mcm | 140 | 2 | 6 | 1 |" in out
    assert "| (a,b,c) | P-1 | 2 | 2 | 48 | 3 |" in out
    assert "Pm-3m (No. 221)" in out


def test_isotropy_subgroups_by_number_and_lowercase_label():
    out = assert_result(run(tools.crystod_isotropy_subgroups("221", "r4+")))
    assert "I4/mcm" in out and "R4+" in out


def test_isotropy_subgroups_single_direction_reports_setting():
    out = assert_result(run(tools.crystod_isotropy_subgroups("Pm-3m", "R4+", "0 0 a")))
    assert "| (0,0,a) | I4/mcm | 140 | 2 | 6 | 1 |" in out
    assert "R-3c" not in out
    assert "Conventional cell of I4/mcm" in out and "basis rows" in out and "origin" in out


def test_isotropy_subgroups_every_irrep_of_a_kpoint():
    out = assert_result(run(tools.crystod_isotropy_subgroups("Pm-3m", "GM")))
    assert "## Isotropy subgroups of every irrep at GM in Pm-3m (No. 221)" in out
    assert "| GM1+ | (a) | Pm-3m | 221 | 1 | 1 | 1 |" in out
    assert "| GM4- | (0,0,a) | P4mm | 99 | 1 | 6 | 1 |" in out
    assert "| GM5- | (a,b,c) | P1 | 1 | 1 | 48 | 3 |" in out
    assert "29 isotropy subgroup(s) of 10 irrep(s) at GM = (0, 0, 0)" in out
    assert run(tools.crystod_isotropy_subgroups("221", "gamma")) == out
    message = assert_error(tools.crystod_isotropy_subgroups("Pm-3m", "Q"), "not tabulated")
    assert "R (1/2, 1/2, 1/2)" in message  # the remedy lists the tabulated k points
    assert_error(tools.crystod_isotropy_subgroups("Pm-3m", "GM", "0 0 a"), "k-point name")


def test_isotropy_subgroups_bad_irrep_and_space_group():
    message = assert_error(tools.crystod_isotropy_subgroups("Pm-3m", "R9+"), "not tabulated")
    assert "R4+" in message  # the remedy lists the tabulated labels
    assert_error(tools.crystod_isotropy_subgroups("Xyz", "R4+"), "not recognized")
    assert_error(tools.crystod_isotropy_subgroups("Pm-3m", "octahedral"), "ISO-IR")
    assert_error(tools.crystod_isotropy_subgroups("Pm-3m", "R4+", "0 0 ?"), "order-parameter")


def test_irrep_product_space_group():
    out = assert_result(run(tools.crystod_irrep_product("Pm-3m", ["R4-", "R5+"])))
    assert "R4- x R5+ = GM2- + GM3- + GM4- + GM5-" in out
    assert "3 x 3 = 9" in out
    assert_error(tools.crystod_irrep_product("Pm-3m", ["R4-"]), "at least two")
    assert_error(tools.crystod_irrep_product("Pm-3m", ["R4-", "Q7"]), "not tabulated")


def test_irrep_product_bad_input_is_an_error_result():
    # SpaceGroupIrrepAlgebra("Pm3m") raises SystemExit in the worker thread; on
    # Python 3.10/3.11 asyncio.wait_for re-raised it out of asyncio.run
    message = tool_error(tools.crystod_irrep_product("Pm3m", ["R4-", "R5+"]))
    assert message.startswith('"Pm3m" is not recognized as a standard space-group symbol'), message
    assert "not tabulated" in tool_error(tools.crystod_irrep_product("Pm-3m", ["R4-", "R9+"]))


def test_irrep_product_point_group():
    out = assert_result(run(tools.crystod_irrep_product("m-3m", ["T2g", "T2g"])))
    assert "T2g x T2g = A1g + Eg + T1g + T2g" in out


def test_decompose_representation():
    out = assert_result(run(tools.crystod_decompose_representation("3m", [3, 0, 1])))
    assert "Gamma = 1(A1) + 1(E)" in out
    assert "| E | 1 | 3 |" in out and "| sgv | 3 | 1 |" in out
    message = assert_error(tools.crystod_decompose_representation("3m", [3, 0]), "3 classes")
    assert "1E, 2C3, 3sgv" in message
    assert_error(tools.crystod_decompose_representation("Oh", [1]), "Hermann-Mauguin")


def test_decompose_representation_flags_non_representation():
    out = assert_result(run(tools.crystod_decompose_representation("3m", [3, 1, 1])))
    assert "WARNING" in out and "non-integer" in out


def test_ligand_field():
    out = assert_result(run(tools.crystod_ligand_field("m-3m", "d")))
    assert "d in m-3m = 1(Eg) + 1(T2g)" in out
    out = assert_result(run(tools.crystod_ligand_field("-43m", "d")))
    assert "1(E) + 1(T2)" in out
    assert_error(tools.crystod_ligand_field("m-3m", "x"), "orbital shell")


def test_ligand_field_complex_pair_counts_once():
    # E of 4 is a time-reversal pair of complex-conjugate irreps listed as one
    # real E: the d shell gives one doubly degenerate E level (5 orbitals)
    out = assert_result(run(tools.crystod_ligand_field("4", "d")))
    assert "d in 4 = 1(A) + 2(B) + 1(E)" in out
    assert "1(E) is one doubly degenerate level" in out and "2(E)" not in out
    assert "WARNING" not in out
    # the reduction check divides by the norm of each irrep (2|G| for E of 4)
    reduced = assert_result(run(tools.crystod_decompose_representation("4", [2, 0, -2])))
    assert "Gamma = 1(E)" in reduced and "WARNING" not in reduced
    half = assert_result(run(tools.crystod_decompose_representation("4", [1, 0, -1])))
    assert "WARNING" in half and "E: 0.500" in half


def test_character_table_point_group():
    out = assert_result(run(tools.crystod_character_table("m-3m")))
    assert "Character table of point group m-3m" in out
    assert "| T2g | 3 | 0 | 1 | -1 | -1 | 3 | -1 | 0 | -1 | 1 |" in out
    assert "| Irrep | 1E | 8C3 |" in out
    assert "Order 48" in out


def test_character_table_space_group_at_r():
    out = assert_result(run(tools.crystod_character_table("Pm-3m", "R")))
    assert "little group of k = R (1/2, 1/2, 1/2) in Pm-3m (No. 221)" in out
    for label in ("R1+", "R4+", "R5-"):
        assert label in out
    same = assert_result(run(tools.crystod_character_table("221", [0.5, 0.5, 0.5])))
    assert "R5-" in same
    gamma = assert_result(run(tools.crystod_character_table("Pm-3m")))
    assert "Gamma point" in gamma and "GM4-" in gamma
    assert_error(tools.crystod_character_table("Pm-3m", "Q"), "not a special k point", "GM, X, M, R")


def test_invariants_pm3m_r4p():
    out = assert_result(run(tools.crystod_invariants("Pm-3m", "R4+")))
    assert "## Invariant polynomials of R4+ in Pm-3m (No. 221)" in out
    assert "R4+: order-parameter dimension 3" in out
    assert "Landau condition: satisfied" in out and "Lifshitz condition: satisfied" in out
    assert "I2_1 = Q1^2 + Q2^2 + Q3^2" in out
    assert "I4_1 = Q1^2*Q2^2 + Q1^2*Q3^2 + Q2^2*Q3^2" in out
    assert "degree 3: none" in out and "Molien series" in out
    assert "J. Appl. Cryst. 36, 951-952 (2003)" in out  # the INVARIANTS citation
    # M1+ has a cubic invariant: the Landau condition fails
    m1 = assert_result(run(tools.crystod_invariants("221", "m1+", 3)))
    assert "I3_1 = Q1*Q2*Q3" in m1 and "Landau condition: violated (1 cubic invariant)" in m1


def test_invariants_direct_sum_coupling_term():
    out = assert_result(run(tools.crystod_invariants("I4/mmm", ["X2+", "X3-", "GM5-"], 3)))
    assert "### Lowest-order coupling term" in out
    assert "degree (1, 1, 1) (trilinear): 1 invariant" in out
    assert "Coupled order parameter dimension 6 (2 + 2 + 2)" in out
    biquadratic = assert_result(run(tools.crystod_invariants("Pm-3m", "R4+ M3+")))
    assert "degree (2, 2): " in biquadratic and "trilinear" not in biquadratic


def test_invariants_direction_and_secondary_order_parameters():
    out = assert_result(run(tools.crystod_invariants("Pm-3m", "R4+", 4, "a 0 0")))
    assert "### Restricted to R4+(a,0,0)" in out
    assert "Isotropy subgroup I4/mcm (No. 140), size 2, index 6." in out
    assert "- degree 2: `a^2`" in out and "- degree 4: `a^4`" in out
    assert "### Secondary order parameters of I4/mcm (No. 140)" in out
    assert "| GM1+ | (0,0,0) | 1 | 1 | (a) | Q^2 eta | strain |" in out
    assert "| GM3+ | (0,0,0) | 2 | 1 |" in out
    assert "| R4+ | (1/2,1/2,1/2) | 3 | 1 | (a,0,0) | - | primary |" in out
    # coupled order parameters: the hybrid improper GM5- appears as polar
    hybrid = assert_result(run(tools.crystod_invariants("I4/mmm", "X2+ X3-", 4, "0 a 0 b")))
    assert "| GM5- |" in hybrid and "Q1^1 Q2^1 eta | polar |" in hybrid


def test_invariants_errors():
    message = assert_error(tools.crystod_invariants("Pm-3m", "R9+"), "not tabulated")
    assert "Traceback" not in message and "R4+" in message
    assert_error(tools.crystod_invariants("Pm-3m", "R4+", 0), "degree", "1 to 12")
    assert_error(tools.crystod_invariants("Pm-3m", "R4+", "four"), "degree")
    assert_error(tools.crystod_invariants("Pm-3m", "R4+", 4, "a 0"), "3 components")
    assert_error(tools.crystod_invariants("Pm-3m", "R4+", 1, "a 0 0"), "degree 2")
    assert_error(tools.crystod_invariants("Pm-3m", "R4+", 4, "0 0 ?"), "order-parameter")
    assert_error(tools.crystod_invariants("Fm-3m", "W5", 7), "monomials", "lower the degree")


def test_find_isotropy_irreps_pm3m_i4mcm():
    out = assert_result(run(tools.crystod_find_isotropy_irreps("Pm-3m", "I4/mcm")))
    assert "## Irreps of Pm-3m (No. 221) with isotropy subgroups of type I4/mcm (No. 140)" in out
    assert "40 irreps at the special k points GM, R, X, M; 243 strata" in out
    assert ("| R4+ | R | (0,0,a) | I4/mcm (140) | 2 | 6 | (-1,0,1),(1,0,1),(0,2,0) "
            "| (0,0,0) |") in out
    assert "10 strata" in out and "J. Appl. Cryst. 49, 1849-1853 (2016)" in out
    # the second call reads the cached table; filters by index and k point
    filtered = assert_result(run(tools.crystod_find_isotropy_irreps("221", "140", index=12, kpoints="M")))
    assert "read from the cache" in filtered and "Selected: index 12, k point M." in filtered
    assert "| M5+ | M |" in filtered and "| R4+ |" not in filtered
    # no single irrep gives the Pnma tilt system with cell size 4
    pnma = assert_result(run(tools.crystod_find_isotropy_irreps("Pm-3m", "Pnma", size=4)))
    assert "No single-irrep stratum of type Pnma (No. 62) with size 4" in pnma
    assert "coupled=true" in pnma


def test_find_isotropy_irreps_coupled_pnma():
    out = assert_result(run(tools.crystod_find_isotropy_irreps("Pm-3m", "Pnma", size=4, coupled=True)))
    assert "### Coupled pairs of irreps" in out
    assert "| R4+ | (0,a,a) | M3+ | (d;0;0) | Pnma (62) | 4 | 24 |" in out
    assert "32 coupled strata" in out and "neither irrep alone" in out


def test_find_isotropy_irreps_enantiomorphic_partner_and_errors():
    out = assert_result(run(tools.crystod_find_isotropy_irreps("P4_2/mcm", "P4_322")))
    assert "| Z3 | Z | (0,a) | P4_122 (91) | 2 | 4 |" in out
    assert "enantiomorphic partner types" in out
    assert_error(tools.crystod_find_isotropy_irreps("Pm-3m", "Xyz"), "not recognized")
    assert_error(tools.crystod_find_isotropy_irreps("Pm-3m", "I4/mcm", size=0), "positive integer")
    assert_error(tools.crystod_find_isotropy_irreps("Pm-3m", "I4/mcm", kpoints="Q"),
                 "not a special k point", "GM, R, X, M")
    assert_error(tools.crystod_find_isotropy_irreps("Pm-3m", " "), "subgroup type")


def test_subgroup_graph_pm3m_r4p():
    out = assert_result(run(tools.crystod_subgroup_graph("Pm-3m", "R4+")))
    assert "## Group-subgroup graph of R4+ in Pm-3m (No. 221)" in out
    assert "7 node(s) (the parent and 6 strata), 10 edge(s) (0 dashed)." in out
    assert "- index 24: C2/m (12), C2/c (15)" in out
    assert "| 2 | I4/mcm (140) | stratum | 2 | 6 | R4+(0,0,a) | 1 | 1 | solid |" in out
    assert "| 6 | C2/c (15) | stratum | 2 | 24 | R4+(a,a,b) | 2 | 2,3,4 | - |" in out
    assert "| 1 Pm-3m (221) | 2 I4/mcm (140) | solid | 6 |" in out
    assert "Howard and H. T. Stokes" in out and "<svg" not in out


def test_subgroup_graph_dashed_and_errors():
    out = assert_result(run(tools.crystod_subgroup_graph("221", "m1+")))
    assert "Landau condition: violated" in out
    assert "| 1 Pm-3m (221) | 2 Im-3m (229) | dashed | 4 |" in out
    tilts = assert_result(run(tools.crystod_subgroup_graph("Pm-3m", ["R4+", "M3+"])))
    assert "25 node(s) (the parent and 24 strata), 56 edge(s) (0 dashed)." in tilts
    assert "R4+(0,a,a) M3+(d;0;0)" in tilts
    message = assert_error(tools.crystod_subgroup_graph("Pm-3m", "R9+"), "not tabulated")
    assert "R4+" in message
    assert_error(tools.crystod_subgroup_graph("Pm-3m", ""), "ISO-IR")


def test_correlate_point_group_table():
    out = assert_result(run(tools.crystod_correlate(point_group="m-3m", subgroup="4/mmm")))
    assert "## Correlation table m-3m -> 4/mmm" in out
    assert "| Irrep of m-3m | 4/mmm (4 \\|\\| z) |" in out
    assert "| T2g | B2g + Eg |" in out and "| Eg | A1g + B1g |" in out
    mm2 = assert_result(run(tools.crystod_correlate(point_group="m-3m", subgroup="mm2")))
    assert "3 inequivalent orientations" in mm2 and "| Eg | 2 A1 | A1 + A2 | A1 + B2 |" in mm2


def test_correlate_compatibility_relations():
    out = assert_result(run(tools.crystod_correlate(space_group="Pm-3m", kpoints="GM X")))
    assert "### Line DT between GM (0, 0, 0) and X (0, 1/2, 0)" in out
    assert "kdelta = (0, 1/24, 0)" in out
    assert "| GM4- | GM | 3 | DT1 + DT5 |" in out and "| X5+ | X | 2 | DT5 |" in out
    every = assert_result(run(tools.crystod_correlate(space_group="221", kpoints=["X"])))
    assert "### Line S between X" in every and "### Line Z between X" in every


def test_correlate_subduction():
    out = assert_result(run(tools.crystod_correlate(
        parent="Pm-3m", irreps="R4+", direction="a 0 0", irrep_list=["R4+", "R5+", "M3+", "GM4-"])))
    assert "Subduction to the Gamma point of I4/mcm (No. 140)" in out
    assert "| R4+ | R | 3 | GM1+ [A1g] + GM5+ [Eg] | 0 | yes |" in out
    assert "| M3+ | M | 3 | - | 3 | no |" in out
    pnma = assert_result(run(tools.crystod_correlate(
        parent="Pm-3m", irreps=["R4+", "M3+"], direction="0 a a d 0 0")))
    assert "Pnma (No. 62)" in pnma and "| M3+ | M | 3 | GM1+ [Ag] | 2 | yes |" in pnma


def test_correlate_needs_exactly_one_form():
    for call in (tools.crystod_correlate(),
                 tools.crystod_correlate(point_group="m-3m", space_group="Pm-3m")):
        assert_error(call, "exactly one of point_group")
    assert_error(tools.crystod_correlate(point_group="m-3m", subgroup="4/mmm", kpoints="GM"),
                 "kpoints belongs to another form")
    assert_error(tools.crystod_correlate(point_group="m-3m"), "needs subgroup")
    assert_error(tools.crystod_correlate(point_group="m-3m", subgroup="6/mmm"),
                 "not a subgroup of m-3m")
    assert_error(tools.crystod_correlate(space_group="Pm-3m"), "need kpoints")
    assert_error(tools.crystod_correlate(space_group="Pm-3m", kpoints="GM X R"), "one or two")
    assert_error(tools.crystod_correlate(space_group="Pm-3m", kpoints="GM X", line="Q"),
                 "no line named Q")
    assert_error(tools.crystod_correlate(parent="Pm-3m", irreps="R4+"), "needs direction")
    assert_error(tools.crystod_correlate(parent="Pm-3m", irreps="R4+", direction="a 0"),
                 "direction needs 3 components")


def test_tensor_form_piezoelectric_4mm():
    out = assert_result(run(tools.crystod_tensor_form("piezoelectric", point_group="4mm")))
    assert "## Piezoelectric tensor in point group 4mm (C4v)" in out
    assert "**3 independent components: d15, d31, d33**" in out
    assert "| 3 | d31 | d31 | d33 | 0 | 0 | 0 |" in out
    assert "- d24 = d15" in out and "- every other component is zero" in out
    hexagonal = assert_result(run(tools.crystod_tensor_form("elastic", space_group="P6_3mc")))
    assert "point group of P6_3mc (No. 186)" in hexagonal and "- c66 = (c11-c12)/2" in hexagonal
    odd = assert_result(run(tools.crystod_tensor_form("a[V2]", point_group="4mm")))
    assert "Time-reversal odd: forbidden" in odd


def test_tensor_form_raman_and_errors():
    out = assert_result(run(tools.crystod_tensor_form("raman", point_group="m-3m")))
    assert "T2g (3 tensors):" in out and "Eg (2 tensors):" in out
    si = assert_result(run(tools.crystod_tensor_form("raman", poscar=SI_DIAMOND)))
    assert "point group of Fd-3m (No. 227)" in si and "GM5+ [T2g] (3 tensors):" in si
    dielectric = assert_result(run(tools.crystod_tensor_form("dielectric", poscar=SRTIO3)))
    assert "**1 independent component: eps11**" in dielectric
    assert "source: the POSCAR structure" in dielectric
    assert_error(tools.crystod_tensor_form("foo", point_group="4mm"), "not a tensor kind")
    assert_error(tools.crystod_tensor_form("elastic"), "exactly one of point_group")
    assert_error(tools.crystod_tensor_form("elastic", point_group="4mm", space_group="P4mm"),
                 "exactly one of point_group")
    assert_error(tools.crystod_tensor_form("elastic", point_group="C4v"), "Hermann-Mauguin")
    assert_error(tools.crystod_tensor_form("elastic", poscar="not a poscar"), "POSCAR")


# --------------------------------------------------------------------------
# phonon
# --------------------------------------------------------------------------

def test_phonon_irreps_srtio3_at_r():
    out = assert_result(run(tools.crystod_phonon_irreps(SRTIO3, [4, 4, 4], FORCE_SETS_SRTIO3, "R")))
    assert "### R (1/2, 1/2, 1/2)" in out
    assert "| 1,2,3 | -1.0867 (imaginary) | R5- | 3 |" in out
    assert "| 4,5,6 | 3.9891 | R4- | 3 |" in out
    assert "Pm-3m (No. 221)" in out


def test_phonon_irreps_all_special_points_and_errors():
    out = assert_result(run(tools.crystod_phonon_irreps(SRTIO3, [4, 4, 4], FORCE_SETS_SRTIO3)))
    for name in ("### GM", "### X", "### M", "### R"):
        assert name in out
    assert "R5-" in out and "GM4-" in out
    assert_error(tools.crystod_phonon_irreps(SRTIO3, [4, 4], FORCE_SETS_SRTIO3), "three")
    assert_error(tools.crystod_phonon_irreps("not a poscar", [4, 4, 4], FORCE_SETS_SRTIO3), "POSCAR")
    assert_error(tools.crystod_phonon_irreps(SRTIO3, [4, 4, 4], "garbage"), "FORCE_SETS")
    assert_error(tools.crystod_phonon_irreps(SRTIO3, [2, 2, 2], FORCE_SETS_SRTIO3), "phonopy")


def test_phonon_special_qpoints_in_the_frame_of_the_labels():
    """The survey and a qpoint label name the points as the labels do."""
    import numpy as np
    from phonopy import Phonopy
    from phonopy.interface.calculator import read_crystal_structure

    from crystod_mcp.tools import phonon as phonon_tools

    def special_qpoints(poscar, number):
        with utils.temporary_files({"POSCAR": poscar}) as directory:
            unit, _ = read_crystal_structure(str(directory / "POSCAR"), interface_mode="vasp")
        phonon = Phonopy(unit, supercell_matrix=np.eye(3, dtype=int), primitive_matrix="auto")
        return phonon_tools._special_qpoints(phonon, number)

    # outside the ISO-IR setting P moves to -k (it was listed at the table's
    # (1/4, 1/4, 1/4), above PA labels) ...
    names, points = special_qpoints(ALPO4_SHIFTED, 82)
    assert names == ["GM", "N", "X", "P", "M"]
    assert points[names.index("P")] == [-0.25, -0.25, -0.25]
    # ... and a structure in the ISO-IR setting keeps the table
    _, table_names, table_points, _ = utils.special_kpoints("221")
    names, points = special_qpoints(SRTIO3, 221)
    assert names == list(table_names)
    assert points == [[float(v) for v in k] for k in table_points]


def test_imaginary_mode_subgroups_srtio3():
    out = assert_result(run(tools.crystod_imaginary_mode_subgroups(SRTIO3, [4, 4, 4], FORCE_SETS_SRTIO3)))
    assert "### R (1/2, 1/2, 1/2): bands 1,2,3, -1.0867 THz, irrep R5-" in out
    for symbol in R4P_SUBGROUPS:
        assert f"| {symbol} |" in out
    assert "| R5-(0,0,a) | I4/mcm | 140 | 2 | 6 | 1 |" in out
    assert "64 q points" in out
    stable = assert_result(run(tools.crystod_imaginary_mode_subgroups(SRTIO3, [4, 4, 4], FORCE_SETS_SRTIO3, -5.0)))
    assert "No phonon level below -5 THz" in stable


def test_phonon_activity_srtio3():
    out = assert_result(run(tools.crystod_phonon_activity(SRTIO3)))
    assert "## Gamma-point phonon activity of Pm-3m (No. 221)" in out
    assert "| GM4- | 3 | 4 | 1 | 3 | IR |" in out
    assert "| GM5- | 3 | 1 | 0 | 1 | silent |" in out
    assert "Gamma-point activity: GM4- x4 (3 IR, 1 acoustic), GM5- x1 (silent)" in out
    assert "- Wyckoff orbit O (3c): 2 GM4- + GM5-" in out
    assert "(equals the mode-space list)" in out
    assert "no Raman-active irrep at Gamma" in out
    plain = assert_result(run(tools.crystod_phonon_activity(SRTIO3, raman_tensors=False)))
    assert "Raman tensors" not in plain


def test_phonon_activity_raman_tensors_and_errors():
    out = assert_result(run(tools.crystod_phonon_activity(SI_DIAMOND)))
    assert "Fd-3m (No. 227)" in out
    assert "| GM4- | 3 | 1 | 1 | 0 | acoustic |" in out
    assert "| GM5+ | 3 | 1 | 0 | 1 | Raman |" in out
    assert "GM5+ (3 tensors):  [[0, 0, 0], [0, 0, a], [0, a, 0]]" in out
    message = assert_error(tools.crystod_phonon_activity("not a poscar"), "POSCAR")
    assert "Traceback" not in message
    assert_error(tools.crystod_phonon_activity(SRTIO3, True, -1), "symprec")


def test_phonon_activity_writes_nothing_to_stdout(capsys):
    # stdout is the protocol stream of the stdio server: the primitive-cell
    # note of SymmetryOnlyVibrations used to be printed there
    assert_result(run(tools.crystod_phonon_activity(SRTIO3, raman_tensors=False)))
    assert capsys.readouterr().out == ""


def test_phonon_activity_keeps_stdout_clean(capsys):
    # the vibration analysis prints a primitive-cell banner; on the stdio
    # transport stdout is the message stream, so it must land on stderr
    out = assert_result(run(tools.crystod_phonon_activity(SRTIO3, raman_tensors=False)))
    assert "Gamma-point activity: GM4- x4" in out
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# magnetism
# --------------------------------------------------------------------------

# AlNi3 (Pm-3m), Ni on 3c; rutile TiO2 (P4_2/mnm), Ti on 2a
ALNI3 = """AlNi3 (Pm-3m)
   1.0
     3.5610840000000000    0.0000000000000000    0.0000000000000000
     0.0000000000000000    3.5610840000000000    0.0000000000000000
     0.0000000000000000    0.0000000000000000    3.5610840000000000
Al Ni
   1    3
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.5000000000000000  0.5000000000000000  0.0000000000000000
  0.5000000000000000  0.0000000000000000  0.5000000000000000
  0.0000000000000000  0.5000000000000000  0.5000000000000000
"""

TIO2 = """TiO2 (P4_2/mnm)
   1.0
     4.6532720000000000    0.0000000000000000    0.0000000000000000
     0.0000000000000000    4.6532720000000000    0.0000000000000000
     0.0000000000000000    0.0000000000000000    2.9692030000000000
Ti O
   2    4
Direct
  0.5000000000000000  0.5000000000000000  0.5000000000000000
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.8045800000000000  0.1954200000000000  0.5000000000000000
  0.1954200000000000  0.8045800000000000  0.5000000000000000
  0.3045800000000000  0.3045800000000000  0.0000000000000000
  0.6954200000000000  0.6954200000000000  0.0000000000000000
"""


def test_mag_classify_alni3_ferromagnet():
    out = assert_result(run(tools.crystod_mag_classify(ALNI3, "Ni")))
    assert "Pm-3m (No. 221)" in out and "2 x GM4+(3) + GM5+(3)" in out
    assert ("| GM4+(3) [FM, dipole] | z | P4/mm'm' (123.345) | III | 4/mm'm' | along z "
            "| ferromagnet |") in out
    assert "altermagnet (g-wave)" not in out and "### " not in out


def test_mag_classify_rutile_altermagnet_details_and_errors():
    out = assert_result(run(tools.crystod_mag_classify(TIO2, "ti", "GM", classify_only=False)))
    assert ("| GM2+(1) [AFM, octupole] | z | P4_2'/mnm' (136.499) | III | 4'/mmm' | none "
            "| altermagnet (d-wave) |") in out
    assert "spin splitting allowed by symmetry ~ k_x k_y" in out
    assert "piezomagnetism 2 components" in out and "standard BNS setting" in out
    assert_error(tools.crystod_mag_classify(TIO2, "Cu"), "not in this POSCAR")
    assert_error(tools.crystod_mag_classify("not a poscar", "Ti"), "POSCAR")


FE_BCC = """Fe (Im-3m, primitive)
   1.0
    -1.4200258395987300    1.4200258395987300    1.4200258395987300
     1.4200258395987300   -1.4200258395987300    1.4200258395987300
     1.4200258395987300    1.4200258395987300   -1.4200258395987300
Fe
   1
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
"""


def test_mag_classify_keeps_every_row():
    # spglib cannot name the x component in the 2x2x2 cell given; an
    # equivalent basis of the same lattice is identified, no row is dropped
    out = assert_result(run(tools.crystod_mag_classify(FE_BCC, "Fe", "H")))
    for tag in ("z", "x", "y"):
        assert (f"| H4+(3) [AFM (q != 0)] | {tag} | P_I4/mnc (128.410) | IV | 4/mmm1' | none "
                "| antiferromagnet (T tau) |") in out


# --------------------------------------------------------------------------
# SALC, BZ, molecules
# --------------------------------------------------------------------------

def test_crystal_orbital_irreps_scf3_sc_d():
    out = assert_result(run(tools.crystod_crystal_orbital_irreps(SCF3, "Sc", "d", "GM")))
    assert "GM3+(2)" in out and "GM5+(3)" in out
    assert "Pm-3m (221)" in out
    everywhere = assert_result(run(tools.crystod_crystal_orbital_irreps(SCF3, "sc", "d")))
    assert "GM3+(2)" in everywhere and "R3+(2)" in everywhere and "X5+(2)" in everywhere
    explicit = assert_result(run(tools.crystod_crystal_orbital_irreps(SCF3, "Sc", "d", "1/2 1/2 1/2")))
    assert "R3+(2)" in explicit and "GM3+" not in explicit
    assert_error(tools.crystod_crystal_orbital_irreps(SCF3, "Ti", "d"), "Ti")
    assert_error(tools.crystod_crystal_orbital_irreps(SCF3, "Sc", "d", "Q"), "GM, X, M, R")


def test_crystal_orbital_irreps_label_in_the_frame_of_the_labels():
    """kpoint="P" is the point the labels call P, as for crystod --kpoint P."""
    named = assert_result(run(tools.crystod_crystal_orbital_irreps(ALPO4_SHIFTED, "Al", "p", "P")))
    assert "## Crystal-orbital irreps of Al p at P" in named
    assert " P [-0.25, -0.25, -0.25]" in named
    assert "1.0 [P1(1)] + 1.0 [P2(1)] + 1.0 [P3(1)]" in named
    given = assert_result(run(tools.crystod_crystal_orbital_irreps(ALPO4_SHIFTED, "Al", "p", "1/4 1/4 1/4")))
    assert " PA [0.25, 0.25, 0.25]" in given
    assert "1.0 [PA1(1)] + 1.0 [PA2(1)] + 1.0 [PA3(1)]" in given


def test_special_kpoints():
    out = assert_result(run(tools.crystod_special_kpoints("Pm-3m")))
    for row in ("| GM | (0, 0, 0) |", "| X | (0, 1/2, 0) |", "| M | (1/2, 1/2, 0) |", "| R | (1/2, 1/2, 1/2) |"):
        assert row in out
    assert "Conventional" not in out
    fcc = assert_result(run(tools.crystod_special_kpoints("Fm-3m")))
    assert "Conventional reciprocal basis" in fcc and "| L | (1/2, 1/2, 1/2) | (1/2, 1/2, 1/2) |" in fcc
    from_poscar = assert_result(run(tools.crystod_special_kpoints(poscar=SCF3)))
    assert "detected from the POSCAR" in from_poscar and "Pm-3m (No. 221)" in from_poscar
    assert "| R | (1/2, 1/2, 1/2) |" in from_poscar
    assert_error(tools.crystod_special_kpoints(), "exactly one")
    assert_error(tools.crystod_special_kpoints("Pm-3m", SCF3), "exactly one")
    assert_error(tools.crystod_special_kpoints("999"), "1 to 230")


def test_molecular_symmetry_ch4():
    out = assert_result(run(tools.crystod_molecular_symmetry(CH4)))
    assert "Td (Hermann-Mauguin: -43m)" in out and "8C3" in out
    salc = assert_result(run(tools.crystod_molecular_symmetry(CH4, "H", "s")))
    assert "Gamma = 1(A1) + 1(T2)" in salc
    assert "A1: [s(H1) + s(H2) + s(H3) + s(H4)]" in salc
    nh3 = assert_result(run(tools.crystod_molecular_symmetry(NH3, "N", "p")))
    assert "C3v" in nh3 and "A1: [pz(N1)]" in nh3
    assert_error(tools.crystod_molecular_symmetry(CH4, "H"), "both element and orbital")
    assert_error(tools.crystod_molecular_symmetry("hello"), "XYZ")
    assert_error(tools.crystod_molecular_symmetry(CH4, "Xe", "s"), "Xe")


# --------------------------------------------------------------------------
# surface slabs
# --------------------------------------------------------------------------

def _slab_rows(out: str) -> list[tuple[str, str, str]]:
    """``(h k l, atoms, polar)`` of every row of the slab table in ``out``."""
    table = out.split("```")[1].strip("\n").splitlines()
    assert table[0].split()[:5] == ["#", "h", "k", "l", "shift"], table[0]
    rows = [line.split() for line in table[1:]]
    return [(" ".join(row[1:4]), row[5], row[10]) for row in rows]


def test_slab_srtio3_max_index_1(capsys):
    from crystod.slab import SLAB_CITATION_LINES

    out = assert_result(run(tools.crystod_slab(SRTIO3, max_index=1)))
    assert "## Surface slabs of SrTiO3 (Pm-3m, No. 221)" in out
    assert "distinct under the Laue group m-3m: (1 1 1) (1 1 0) (1 0 0)" in out
    assert _slab_rows(out) == [("1 1 1", "25", "yes"), ("1 1 0", "20", "yes"),
                               ("1 0 0", "15", "no")]
    assert "Sr+2 Ti+4 O-2 (guessed)" in out and "not written (--no-files)" in out
    assert "None: no output_dir was given" in out
    for line in SLAB_CITATION_LINES:
        assert line in out
    assert out.index("10.1021/acsami.5c17978") < out.index(utils.CITATION)
    assert max(len(line) for line in out.split(utils.CITATION)[0].splitlines()) <= 100
    assert capsys.readouterr().out == ""


def test_slab_files_and_options(tmp_path):
    out = assert_result(run(tools.crystod_slab(SRTIO3, "1 0 0", output_dir=str(tmp_path))))
    assert "Miller index: (1 0 0)" in out and f"Written into {tmp_path}:" in out
    assert "- #1 POSCAR_slab_SrTiO3_100_t1" in out
    lines = (tmp_path / "POSCAR_slab_SrTiO3_100_t1").read_text().splitlines()
    assert lines[5].split() == ["Sr", "Ti", "O"] and lines[6].split() == ["3", "3", "9"]
    again = assert_result(run(tools.crystod_slab(SRTIO3, [1, 0, 0], output_dir=str(tmp_path))))
    assert "(kept: identical file already there)" in again
    (tmp_path / "POSCAR_slab_SrTiO3_100_t1").write_text("changed\n")
    assert_error(tools.crystod_slab(SRTIO3, "1 0 0", output_dir=str(tmp_path)),
                 "already exists with different content", "never replaces")

    deeper = tmp_path / "new" / "dir"
    limited = assert_result(run(tools.crystod_slab(
        SRTIO3, max_index=1, max_atoms=16, oxidation_states={"Sr": 2, "Ti": 4, "O": -2},
        output_dir=str(deeper))))
    assert "skipped: 25 > 16 atoms" in limited and "skipped: 20 > 16 atoms" in limited
    assert "Sr+2 Ti+4 O-2 (given)" in limited
    assert sorted(p.name for p in deeper.iterdir()) == ["POSCAR_slab_SrTiO3_100_t1"]
    nonpolar = assert_result(run(tools.crystod_slab(SRTIO3, max_index=1, no_polar=True)))
    assert nonpolar.count("skipped: polar") == 2 and "polar slabs: skipped" in nonpolar
    hexagonal = assert_result(run(tools.crystod_slab(SRTIO3, "1 0 -1 0")))
    assert "Miller index: (1 0 0)" in hexagonal and "assumes hexagonal axes" in hexagonal


def test_slab_conventional_cell_and_unknown_polarity():
    primitive = assert_result(run(tools.crystod_slab(SI_DIAMOND, "1 0 0")))
    assert "Fd-3m (No. 227), 2 atoms" in primitive
    assert "not the conventional standard cell (8 atoms)" in primitive
    assert "polarity unknown (no oxidation states)" in primitive
    standard = assert_result(run(tools.crystod_slab(SI_DIAMOND, "1 0 0", conventional=True)))
    assert "standardized conventional cell: 8 atoms" in standard
    assert "not the conventional standard cell" not in standard


def test_slab_bad_input_is_an_error_result(tmp_path):
    assert_error(tools.crystod_slab(SRTIO3), "exactly one")
    assert_error(tools.crystod_slab(SRTIO3, "1 0 0", max_index=1), "exactly one")
    assert_error(tools.crystod_slab(SRTIO3, "1 0"), "h k l")
    assert_error(tools.crystod_slab(SRTIO3, "1 1 1 0"), "not a Miller-Bravais index")
    assert_error(tools.crystod_slab(SRTIO3, max_index=0), "positive integer")
    assert_error(tools.crystod_slab(SRTIO3, "1 0 0", slab_size=-1.0), "must be positive")
    assert_error(tools.crystod_slab(SRTIO3, "1 0 0", max_atoms=0), "max_atoms")
    assert_error(tools.crystod_slab(SRTIO3, "1 0 0", oxidation_states={"Sr": 2}),
                 "no oxidation state for Ti, O", "oxidation_states")
    assert_error(tools.crystod_slab(
        SRTIO3, "1 0 0", oxidation_states={"Sr": 2, "Ti": 4, "O": -2, "Xe": 1}), "Xe")
    assert_error(tools.crystod_slab(SRTIO3, "1 0 0", output_dir="slabs"), "absolute")
    (tmp_path / "file").write_text("x\n")
    assert_error(tools.crystod_slab(SRTIO3, "1 0 0", output_dir=str(tmp_path / "file")),
                 "not a directory")
    assert_error(tools.crystod_slab("not a poscar", "1 0 0"), "POSCAR")


# --------------------------------------------------------------------------
# timeouts and the citation footer
# --------------------------------------------------------------------------

def test_timeout_is_a_clear_error(monkeypatch):
    monkeypatch.setenv(utils.TIMEOUT_ENV, "0.01")
    assert utils.timeout_seconds() == 0.01
    message = assert_error(tools.crystod_crystal_orbital_irreps(SCF3, "Sc", "d"), "did not finish within 0.01 s", utils.TIMEOUT_ENV)
    assert "smaller" in message
    monkeypatch.setenv(utils.TIMEOUT_ENV, "not a number")
    assert utils.timeout_seconds() == utils.DEFAULT_TIMEOUT
    monkeypatch.delenv(utils.TIMEOUT_ENV)
    assert utils.timeout_seconds() == 60.0


def test_with_citation_footer():
    assert utils.with_citation("x").endswith(utils.CITATION + "\n")
    assert "Acta Cryst. A69, 388 (2013)" in utils.CITATION


def test_exit_and_interrupt_in_a_tool_body_become_errors():
    # converted inside the worker thread: from a thread, asyncio.wait_for
    # (Python 3.10/3.11) re-raised them out of the event loop
    @utils.blocking_tool
    def exits(code):
        raise SystemExit(code)

    @utils.blocking_tool
    def interrupted():
        raise KeyboardInterrupt

    assert tool_error(exits("ERROR: bad input; give a valid one.")) == "bad input; give a valid one."
    assert tool_error(exits(2)) == "invalid input"
    assert "interrupted" in tool_error(interrupted())


def test_tool_output_on_stdout_goes_to_stderr(capsys):
    # what a tool body prints must miss stdout (the stdio message stream), also
    # when the call that redirected stdout first ends while another still prints
    stdout = sys.stdout
    fast_inside, slow_inside, release = threading.Event(), threading.Event(), threading.Event()

    @utils.blocking_tool
    def fast():
        fast_inside.set()
        slow_inside.wait(10)
        print("early progress line")
        return "fast result"

    @utils.blocking_tool
    def slow():
        slow_inside.set()
        release.wait(10)
        print("late progress line")
        return "slow result"

    async def overlapping():
        first = asyncio.ensure_future(fast())
        await asyncio.to_thread(fast_inside.wait, 10)
        second = asyncio.ensure_future(slow())
        fast_text = await first
        release.set()
        return fast_text, await second

    fast_text, slow_text = run(overlapping())
    assert assert_result(fast_text).startswith("fast result")
    assert assert_result(slow_text).startswith("slow result")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "early progress line" in captured.err and "late progress line" in captured.err
    assert sys.stdout is stdout


# --------------------------------------------------------------------------
# protocol level: the server lists the eighteen tools and answers a call
# --------------------------------------------------------------------------

def test_tool_registry_matches_spec():
    assert TOOL_NAMES == EXPECTED_TOOLS


def test_protocol_list_and_call_in_process():
    from mcp.client import Client

    async def go():
        async with Client(create_server()) as client:
            listing = await client.list_tools()
            by_name = {tool.name: tool for tool in listing.tools}
            assert tuple(by_name) == EXPECTED_TOOLS
            for name, tool in by_name.items():
                assert tool.input_schema["type"] == "object", name
                assert "properties" in tool.input_schema, name
                assert tool.description and not tool.description.startswith(" "), name
                assert tool.description.splitlines()[0].strip(), name
            schema = by_name["crystod_phonon_irreps"].input_schema
            assert set(schema["required"]) == {"poscar", "dim", "force_sets"}
            assert schema["properties"]["dim"]["items"]["type"] == "integer"
            assert set(by_name["crystod_isotropy_subgroups"].input_schema["required"]) == {"parent", "irrep"}
            assert set(by_name["crystod_invariants"].input_schema["required"]) == {"space_group", "irreps"}
            assert set(by_name["crystod_phonon_activity"].input_schema["required"]) == {"poscar"}
            assert set(by_name["crystod_find_isotropy_irreps"].input_schema["required"]) == {"parent", "child"}
            assert set(by_name["crystod_subgroup_graph"].input_schema["required"]) == {"parent", "irreps"}
            assert not by_name["crystod_correlate"].input_schema.get("required")
            assert set(by_name["crystod_tensor_form"].input_schema["required"]) == {"kind"}
            assert set(by_name["crystod_mag_classify"].input_schema["required"]) == {"poscar", "element"}
            assert set(by_name["crystod_slab"].input_schema["required"]) == {"poscar"}

            table = await client.call_tool(
                "crystod_correlate", {"point_group": "m-3m", "subgroup": "4/mmm"})
            assert not table.is_error and "| T2g | B2g + Eg |" in table.content[0].text

            result = await client.call_tool("crystod_special_kpoints", {"space_group": "Pm-3m"})
            assert not result.is_error
            text = result.content[0].text
            assert "| R | (1/2, 1/2, 1/2) |" in text and utils.CITATION in text

            failed = await client.call_tool("crystod_isotropy_subgroups", {"parent": "Pm-3m", "irrep": "R9+"})
            assert failed.is_error
            assert "not tabulated" in failed.content[0].text

            slabs = await client.call_tool("crystod_slab", {"poscar": SRTIO3, "miller_index": [1, 0, 0]})
            assert not slabs.is_error and "Sr3Ti3O9" in slabs.content[0].text
            no_index = await client.call_tool("crystod_slab", {"poscar": SRTIO3})
            assert no_index.is_error and "exactly one" in no_index.content[0].text
            assert "Traceback" not in no_index.content[0].text

    run(go())


def test_protocol_over_stdio_subprocess():
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "crystod_mcp"])
        async with Client(params) as client:
            names = [tool.name for tool in (await client.list_tools()).tools]
            assert tuple(names) == EXPECTED_TOOLS
            result = await client.call_tool("crystod_ligand_field", {"point_group": "m-3m", "orbital": "d"})
            assert not result.is_error
            assert "1(Eg) + 1(T2g)" in result.content[0].text

    run(go())


def test_protocol_over_stdio_survives_systemexit_of_bad_input():
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "crystod_mcp"])
        async with Client(params) as client:
            # the SystemExit of an unknown space-group symbol used to leave the
            # server without a reply (the call waited forever): hence the deadline
            failed = await asyncio.wait_for(client.call_tool(
                "crystod_irrep_product", {"space_group": "Pm3m", "irreps": ["R4-", "R5+"]}), 60)
            assert failed.is_error and "not recognized" in failed.content[0].text
            activity = await client.call_tool(
                "crystod_phonon_activity", {"poscar": SRTIO3, "raman_tensors": False})
            assert not activity.is_error
            assert "Gamma-point activity: GM4- x4" in activity.content[0].text

    run(go())


# --------------------------------------------------------------------------
# input robustness
# --------------------------------------------------------------------------

def test_poscar_with_empty_comment_line_is_accepted():
    # the first line of a POSCAR is a free comment and may be empty ...
    blank_comment = "\n" + "\n".join(SCF3.splitlines()[1:]) + "\n"
    out = assert_result(run(tools.crystod_special_kpoints(poscar=blank_comment)))
    assert "Pm-3m (No. 221)" in out
    # ... and a pasted block often carries a stray blank line before the comment
    out = assert_result(run(tools.crystod_special_kpoints(poscar="\n\n" + SCF3)))
    assert "Pm-3m (No. 221)" in out


def test_force_sets_of_another_cell_are_refused():
    # newer phonopy refuses the file itself ("... has forces on 320 atoms, but the
    # supercell has 256 atoms."); with phonopy 4.3 phonopy.load accepts it and the
    # server's own atom-count check refuses it ("... a 320-atom supercell ... 256 atoms")
    message = assert_error(tools.crystod_phonon_irreps(SCF3, [4, 4, 4], FORCE_SETS_SRTIO3, "R"),
                           "256 atoms")
    assert "320 atoms" in message or "320-atom" in message, message


def _cartesian(poscar: str, scale: float) -> str:
    """A POSCAR (direct coordinates, scale 1) again, in Cartesian coordinates
    under a scale factor: the same crystal for VASP."""
    import numpy as np

    lines = poscar.splitlines()
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    total = sum(int(count) for count in lines[6].split())
    fractional = np.array([[float(x) for x in lines[8 + i].split()[:3]] for i in range(total)])

    def numbers(rows):
        return ["  ".join(repr(float(value)) for value in row) for row in rows]

    return "\n".join([lines[0], repr(scale), *numbers(lattice / scale), lines[5], lines[6],
                      "Cartesian", *numbers(fractional @ lattice / scale)]) + "\n"


def test_cartesian_poscar_is_read_as_vasp_reads_it():
    # VASP multiplies the lattice vectors AND Cartesian coordinates by the
    # scale line; phonopy's own reader leaves the coordinates unscaled, and
    # ScF3 and SrTiO3 written with scale 2 were taken for R3m crystals
    out = assert_result(run(tools.crystod_special_kpoints(poscar=_cartesian(SCF3, 2.0))))
    assert "Pm-3m (No. 221)" in out
    out = assert_result(run(tools.crystod_phonon_irreps(
        _cartesian(SRTIO3, 2.0), [4, 4, 4], FORCE_SETS_SRTIO3, "R")))
    assert "Pm-3m (No. 221)" in out
    assert "| 1,2,3 | -1.0867 (imaginary) | R5- | 3 |" in out


def test_non_finite_timeout_falls_back(monkeypatch):
    monkeypatch.setenv(utils.TIMEOUT_ENV, "inf")
    assert utils.timeout_seconds() == utils.DEFAULT_TIMEOUT
