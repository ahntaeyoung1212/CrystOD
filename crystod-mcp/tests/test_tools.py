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
import sys

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
    "crystod_phonon_irreps",
    "crystod_imaginary_mode_subgroups",
    "crystod_crystal_orbital_irreps",
    "crystod_special_kpoints",
    "crystod_molecular_symmetry",
)

SCF3 = example_path("221_PPOSCAR_ScF3").read_text()
SRTIO3 = example_path("221_PPOSCAR_SrTiO3").read_text()
FORCE_SETS_SRTIO3 = example_path("FORCE_SETS_SrTiO3").read_text()
CH4 = example_path("XYZ_CH4.xyz").read_text()
NH3 = example_path("XYZ_NH3.xyz").read_text()

R4P_SUBGROUPS = ("I4/mcm", "R-3c", "Imma", "C2/m", "C2/c", "P-1")


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


def test_imaginary_mode_subgroups_srtio3():
    out = assert_result(run(tools.crystod_imaginary_mode_subgroups(SRTIO3, [4, 4, 4], FORCE_SETS_SRTIO3)))
    assert "### R (1/2, 1/2, 1/2): bands 1,2,3, -1.0867 THz, irrep R5-" in out
    for symbol in R4P_SUBGROUPS:
        assert f"| {symbol} |" in out
    assert "| R5-(0,0,a) | I4/mcm | 140 | 2 | 6 | 1 |" in out
    assert "64 q points" in out
    stable = assert_result(run(tools.crystod_imaginary_mode_subgroups(SRTIO3, [4, 4, 4], FORCE_SETS_SRTIO3, -5.0)))
    assert "No phonon level below -5 THz" in stable


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


# --------------------------------------------------------------------------
# protocol level: the server lists the ten tools and answers a call
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

            result = await client.call_tool("crystod_special_kpoints", {"space_group": "Pm-3m"})
            assert not result.is_error
            text = result.content[0].text
            assert "| R | (1/2, 1/2, 1/2) |" in text and utils.CITATION in text

            failed = await client.call_tool("crystod_isotropy_subgroups", {"parent": "Pm-3m", "irrep": "R9+"})
            assert failed.is_error
            assert "not tabulated" in failed.content[0].text

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
    assert_error(tools.crystod_phonon_irreps(SCF3, [4, 4, 4], FORCE_SETS_SRTIO3, "R"),
                 "320-atom", "256 atoms")


def test_non_finite_timeout_falls_back(monkeypatch):
    monkeypatch.setenv(utils.TIMEOUT_ENV, "inf")
    assert utils.timeout_seconds() == utils.DEFAULT_TIMEOUT
