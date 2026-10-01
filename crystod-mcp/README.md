# crystod-mcp

A [Model Context Protocol](https://modelcontextprotocol.io/) server that
exposes the symmetry analyses of [CrystOD](https://github.com/ahntaeyoung1212/CrystOD)
as tools an AI assistant (Claude Desktop, Claude Code, or any MCP client) can
call: isotropy subgroups of a space-group irrep, direct products of irreps,
reducible-representation and ligand-field decompositions, character tables,
phonon irrep labels and the subgroups reachable from imaginary modes,
crystal-orbital (SALC) irreps, special k points, and molecular point groups
with molecular SALCs.

Every tool takes text and returns Markdown: space groups as symbols or
numbers, irreps in ISO-IR notation (`R4+`, `GM5-`), structures as the full
text of a POSCAR or XYZ file, phonon forces as the text of a phonopy
`FORCE_SETS`. Nothing is read from or written to the user's disk; structure
text is written to a temporary directory for the duration of one call.

Ask an assistant with the server attached:

- "Which space groups can the R-point instability of cubic SrTiO3 condense
  into?" -- `crystod_isotropy_subgroups("Pm-3m", "R4+")`, or the whole
  phonon workflow with `crystod_imaginary_mode_subgroups` on the POSCAR and
  FORCE_SETS.
- "Decompose the direct product of the R4+ and R5+ irreps of Pm-3m" --
  `crystod_irrep_product("Pm-3m", ["R4+", "R5+"])`.
- "Which irreps do the F p orbitals of ScF3 span at Gamma?" --
  `crystod_crystal_orbital_irreps(poscar, "F", "p", "GM")`.
- "How does a d shell split in a tetrahedral field?" --
  `crystod_ligand_field("-43m", "d")`.
- "What is the point group of this molecule, and what SALCs do its H 1s
  orbitals form?" -- `crystod_molecular_symmetry(xyz, "H", "s")`.

## Installation

crystod-mcp is a separate package so that CrystOD itself carries no MCP
dependency. From a checkout of the CrystOD repository:

```bash
pip install ./crystod-mcp
```

This installs CrystOD (>= 0.4.0) and the MCP Python SDK (2.x) as
dependencies; Python 3.10 or later. Publication on PyPI as `crystod-mcp` is
planned, after which `pip install crystod-mcp` and the `uvx` form below work
without a checkout.

Check the installation:

```bash
crystod-mcp --version
```

The server speaks MCP over standard input/output; it is started by the
client, not by hand.

## Client configuration

### Claude Desktop

Add the server to `claude_desktop_config.json` (macOS:
`~/Library/Application Support/Claude/claude_desktop_config.json`; Windows:
`%APPDATA%\Claude\claude_desktop_config.json`). `python` must be the
interpreter in which crystod-mcp is installed -- give its full path if it is
inside a conda environment or a venv.

```json
{
  "mcpServers": {
    "crystod": {
      "command": "python",
      "args": ["-m", "crystod_mcp"],
      "env": {
        "CRYSTOD_MCP_TIMEOUT": "60"
      }
    }
  }
}
```

### Claude Code

```bash
claude mcp add crystod -- crystod-mcp
```

or, naming the interpreter explicitly,

```bash
claude mcp add crystod -- /path/to/env/bin/python -m crystod_mcp
```

### uvx (once crystod-mcp is on PyPI)

```json
{
  "mcpServers": {
    "crystod": {
      "command": "uvx",
      "args": ["crystod-mcp"]
    }
  }
}
```

`mcp_config.json` in this directory holds both forms.

## Tools

| Tool | Question it answers | Inputs |
|---|---|---|
| `crystod_isotropy_subgroups` | which space groups a distortion of one irrep leads to (`crystod-group --parent`) | `parent`, `irrep` (ISO-IR label, e.g. `R4+`; two labels for a coupled order parameter; a bare k-point name such as `GM` for every irrep of that k point), `order_parameter` (optional, e.g. `"0 0 a"`) |
| `crystod_irrep_product` | direct product of space-group irreps (or point-group irreps) decomposed into irreps (`crystod-group --product`) | `space_group` (or a point-group symbol), `irreps` (list) |
| `crystod_decompose_representation` | reduction of a reducible representation from its characters (`crystod-group --decompose`) | `point_group`, `characters` (one per class) |
| `crystod_ligand_field` | ligand-field splitting of an s/p/d/f shell (`crystod-group --ligand-field`) | `point_group`, `orbital` |
| `crystod_character_table` | character table of a point group, or of the little group of a k point with ISO-IR labels (`crystod-group --table`) | `group`, `kpoint` (optional label or coordinates) |
| `crystod_phonon_irreps` | ISO-IR labels of the phonon modes at a q point (`crystod-phonon --irreps`) | `poscar`, `dim`, `force_sets`, `qpoint` (optional; all special points otherwise) |
| `crystod_imaginary_mode_subgroups` | irreps of the imaginary modes and the subgroups they condense into (`crystod-phonon --subgroup`) | `poscar`, `dim`, `force_sets`, `threshold` (THz, default -0.1) |
| `crystod_crystal_orbital_irreps` | irreps of the crystal orbitals of one element's shell (`crystod`) | `poscar`, `element`, `orbital`, `kpoint` (optional) |
| `crystod_special_kpoints` | special k points of a space group, primitive and conventional coordinates (`crystod-bz --show-kpoint`) | `space_group` or `poscar` |
| `crystod_molecular_symmetry` | point group of a molecule and its molecular SALCs (`crystod-mol`) | `xyz`, `element` and `orbital` (optional, together) |

Input conventions, shared by all tools:

- space groups: international symbol (`Pm-3m`, `P6_3/mmc`) or number
  (`221`); point groups: Hermann-Mauguin (`m-3m`, `4/mmm`, `3m`, `-43m`);
- irreps: ISO-IR labels of the space group (`R4+`, `GM5-`, `X3-`) --
  `crystod_character_table` lists the labels a k point has; point-group
  irreps in Mulliken notation (`T2g`, `Eg`, `A1`);
- k points: a label (`GM`, `X`, `M`, `R`) or three fractional coordinates
  in the primitive reciprocal basis (`"1/2 1/2 1/2"`); `crystod_special_kpoints`
  lists them;
- structures: the full text of a VASP POSCAR (VASP 5 format, with the
  element line); molecules: the full text of an XYZ file; forces: the full
  text of a phonopy `FORCE_SETS` together with the supercell `dim` it was
  computed for.

A bad input does not produce a traceback: the tool returns one sentence
naming the problem and the remedy (the tabulated irreps of the space group,
the class order of the character table, the special k points, ...), which
the assistant can act on.

## Citation

The irrep labels come from the ISO-IR tables of the ISOTROPY Software Suite
that CrystOD bundles, and the analyses are CrystOD's. Every tool result
therefore ends with

> Irrep labels follow ISO-IR (H. T. Stokes, B. J. Campbell and R. Cordes,
> Acta Cryst. A69, 388 (2013), https://iso.byu.edu). CrystOD: H. Koiso and
> Y. Mochizuki et al., Phys. Rev. B 110, 064104 (2024).

Please keep both citations when results obtained through the server are
published. The server returns computed results (labels, decompositions,
subgroups), never the tables themselves.

## Time limits and resources

Every call runs under a time limit of 60 s by default; the
`CRYSTOD_MCP_TIMEOUT` environment variable of the server process (in
seconds, see the configuration above) raises or lowers it. A call that
exceeds the limit returns an error naming the limit and the variable instead
of hanging the client. The limit bounds the wait, not the computation: the tools that return a CrystOD report run it in a subprocess that is killed at the limit, while the group-theory, k-point, molecular and phonon tools run in a worker thread that finishes in the background after the error has been returned (retrying a long scan with a larger limit while the first one is still running costs a second scan). The group-theory, k-point and molecular tools answer
in well under a second; the phonon tools rebuild the force constants from
`FORCE_SETS` with phonopy (a few seconds for the 4x4x4 SrTiO3 example, more
for large supercells), and `crystod_imaginary_mode_subgroups` scans every q
point the supercell resolves. The server runs one calculation per call in a
worker thread (or a subprocess for the tools that return a CrystOD report),
needs no network access, and keeps no state between calls.

## Development

```bash
pip install -e "./crystod-mcp[test]"
python -m pytest -q crystod-mcp/tests
```

The tests call every tool on the inputs bundled with CrystOD
(`crystod.examples`: ScF3, SrTiO3 with its 4x4x4 `FORCE_SETS`, CH4, NH3) and
check the results CrystOD's own testsuite expects, then connect to the server
in-process and over stdio to list the ten tools and make a call.

## License

MIT, as CrystOD. The MCP Python SDK that the server is built on is MIT
licensed (Copyright (c) 2024 Anthropic, PBC); its notice is reproduced in
`LICENSE`.
