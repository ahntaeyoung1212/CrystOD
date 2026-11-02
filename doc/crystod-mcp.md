# MCP server (crystod-mcp)

The CrystOD analyses as tools for an AI assistant. `crystod-mcp` is a
[Model Context Protocol](https://modelcontextprotocol.io/) server, shipped as
a separate package in the `crystod-mcp/` directory of the repository, that
lets Claude Desktop, Claude Code, or any MCP client call CrystOD directly:
ask "which space groups can the R-point instability of SrTiO3 condense into?"
and the assistant runs `crystod_isotropy_subgroups` (or the full phonon
workflow) and answers with the table.

| I want to ... | tool |
|---|---|
| list the isotropy subgroups of an irrep | `crystod_isotropy_subgroups` |
| decompose a product of irreps | `crystod_irrep_product` |
| reduce a representation from its characters | `crystod_decompose_representation` |
| see how an orbital shell splits in a ligand field | `crystod_ligand_field` |
| read a character table (point group, or little group of k) | `crystod_character_table` |
| list the Landau invariants, coupling terms and secondary order parameters | `crystod_invariants` |
| find which irreps (or coupled pairs) give a subgroup type | `crystod_find_isotropy_irreps` |
| see the group-subgroup graph of the isotropy subgroups | `crystod_subgroup_graph` |
| correlate irreps to a subgroup (correlation table, compatibility relations, subduction) | `crystod_correlate` |
| get the allowed form of a property tensor (dielectric, piezoelectric, elastic, Raman, ...) | `crystod_tensor_form` |
| label the phonon modes with irreps | `crystod_phonon_irreps` |
| find the subgroups the imaginary modes lead to | `crystod_imaginary_mode_subgroups` |
| see which Gamma phonons are IR or Raman active (structure only) | `crystod_phonon_activity` |
| get the magnetic space group and the collinear class (altermagnet or not) of spin arrangements | `crystod_mag_classify` |
| get the irreps of the crystal orbitals of an element | `crystod_crystal_orbital_irreps` |
| list the special k points of a space group | `crystod_special_kpoints` |
| get the point group and SALCs of a molecule | `crystod_molecular_symmetry` |
| cut the surface slabs of a crystal and see which are polar | `crystod_slab` |

Every tool takes text and returns Markdown, so nothing needs to be on the
assistant's disk: space groups as symbols or numbers, irreps in ISO-IR
notation (`R4+`, `GM5-`), crystal structures as the full text of a POSCAR,
molecules as the full text of an XYZ file, phonon forces as the full text of
a phonopy `FORCE_SETS`. The text of a structure is written to a temporary
directory for the duration of one call and removed afterwards; the only files
the server keeps are the isotropy-table cache of `crystod_find_isotropy_irreps`
(see [Time limits and resources](#time-limits-and-resources)) and the slab
POSCAR files that `crystod_slab` writes into an `output_dir` the call names
(none without it).

## Installation

crystod-mcp is a separate package so that CrystOD itself carries no MCP
dependency. From a checkout of the repository:

```bash
pip install ./crystod-mcp
```

This installs CrystOD (>= 0.4.4) and the MCP Python SDK (2.x) as
dependencies; Python 3.10 or later. `crystod_slab` needs CrystOD 0.4.6 or
later (`crystod.slab`); with an older CrystOD it returns an error saying so,
and the other tools work. Publication on PyPI as `crystod-mcp` is
planned, after which `pip install crystod-mcp` and the `uvx` form below work
without a checkout. `crystod-mcp --version` checks the installation; the
server itself is started by the MCP client over standard input/output, not
by hand.

## Client configuration

Claude Desktop reads `claude_desktop_config.json` (macOS:
`~/Library/Application Support/Claude/claude_desktop_config.json`; Windows:
`%APPDATA%\Claude\claude_desktop_config.json`). `python` must be the
interpreter in which crystod-mcp is installed -- give its full path when it
lives in a conda environment or a venv:

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

Claude Code registers the server from the command line:

```bash
claude mcp add crystod -- crystod-mcp
```

(or `claude mcp add crystod -- /path/to/env/bin/python -m crystod_mcp` to
name the interpreter). With `uv` installed, once the package is on PyPI:

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

`crystod-mcp/mcp_config.json` holds both forms.

## The tools

| Tool | Command it wraps | Inputs |
|---|---|---|
| `crystod_isotropy_subgroups` | [`crystod-group --parent`](crystod-group.md) | `parent` (`Pm-3m` or `221`), `irrep` (`R4+`; two labels for a coupled order parameter; a bare k-point name such as `GM` lists every irrep of that k point, as `--parent SG --kpoint GM` does), `order_parameter` (optional, `"0 0 a"`) |
| `crystod_irrep_product` | [`crystod-group --product`](crystod-group.md) | `space_group` (a point-group symbol such as `m-3m` selects the point-group product), `irreps` (list) |
| `crystod_decompose_representation` | [`crystod-group --decompose`](crystod-group.md) | `point_group`, `characters` (one per class, E first) |
| `crystod_ligand_field` | [`crystod-group --ligand-field`](crystod-group.md) | `point_group`, `orbital` (`s`/`p`/`d`/`f`) |
| `crystod_character_table` | [`crystod-group --table`](crystod-group.md) | `group` (point group, or space group), `kpoint` (optional label or coordinates; Gamma by default for a space group) |
| `crystod_invariants` | [`crystod-group --invariants`](crystod-group.md) | `space_group`, `irreps` (`R4+`, or several labels for a direct sum such as `["X2+", "X3-", "GM5-"]`), `degree` (default 4), `direction` (optional, `"a 0 0"`; adds the restricted free energy and the `--secondary` table) |
| `crystod_find_isotropy_irreps` | [`crystod-group --parent G --child H`](crystod-group.md) | `parent`, `child` (`I4/mcm` or `140`), `size`, `index`, `kpoints` (optional filters, `"R M"`), `coupled` (default false; adds the coupled pairs of `--coupled`) |
| `crystod_subgroup_graph` | [`crystod-group --graph`](crystod-group.md) | `parent`, `irreps` (`R4+`, or several labels for a direct sum such as `["R4+", "M3+"]`); the nodes by index layer and the edge list, as text |
| `crystod_correlate` | [`crystod-group --correlate`](crystod-group.md) | exactly one form: `point_group` and `subgroup` (`m-3m`, `4/mmm`); `space_group` and `kpoints` (`"GM X"`), `line` (optional); `parent`, `irreps` and `direction` (`"a 0 0"`), `irrep_list` (optional) |
| `crystod_tensor_form` | [`crystod-group --tensor`](crystod-group.md) | `kind` (`dielectric`, `pyroelectric`, `piezoelectric`, `elastic`, `compliance`, `gyration`, `raman`, or a Jahn symbol such as `V[V2]`), and one of `point_group`, `space_group`, `poscar` |
| `crystod_phonon_irreps` | [`crystod-phonon --irreps`](crystod-phonon.md) | `poscar`, `dim` (`[4, 4, 4]`), `force_sets`, `qpoint` (optional; every special point otherwise) |
| `crystod_imaginary_mode_subgroups` | [`crystod-phonon --subgroup`](crystod-phonon.md) | `poscar`, `dim`, `force_sets`, `threshold` (THz, default -0.1) |
| `crystod_phonon_activity` | [`crystod-phonon --vibration --qpoint GM`](crystod-phonon.md) | `poscar`, `raman_tensors` (default true; `--raman-tensor`), `symprec` (default 1e-5) |
| `crystod_mag_classify` | [`crystod-mag --qpoint --classify-only`](crystod-mag.md) | `poscar`, `element`, `qpoint` (default `GM`; label or `"1/2 1/2 1/2"`), `classify_only` (default true; false adds the spins, the responses allowed by symmetry and the full collinear description of every component) |
| `crystod_crystal_orbital_irreps` | [`crystod`](crystod.md) | `poscar`, `element`, `orbital`, `kpoint` (optional) |
| `crystod_special_kpoints` | [`crystod-bz --show-kpoint`](crystod-bz.md) | `space_group` or `poscar` |
| `crystod_molecular_symmetry` | [`crystod-mol`](crystod-mol.md) | `xyz`, `element` and `orbital` (optional, together) |
| `crystod_slab` | [`crystod-slab`](crystod-slab.md) | `poscar`, `miller_index` (three or four integers) or `max_index`, `slab_size` (default 10 A), `vacuum_size` (default 15 A), `symmetrize`, `no_polar`, `max_atoms`, `oxidation_states` (e.g. `{"Sr": 2, "Ti": 4, "O": -2}`; pymatgen's guess by default), `conventional`, `output_dir` (optional absolute directory; no files without it) |

The tools are built on the [Python API](python-api.md)
(`crystod.group.isotropy_subgroups`, `crystod.phonon.label_phonon_modes`,
`crystod.phonon.scan_imaginary_modes`, `crystod.group.invariant_polynomials`,
`crystod.group.isotropy_table`, `crystod.group.subgroup_graph`,
`crystod.group.correlation_table`, `crystod.group.tensor_form`,
`crystod.phonon.gamma_mode_activities`, `crystod.bz.get_special_kpoints`, ...);
the three whose printed report is the natural answer (`crystod`, the
little-group character table and `crystod-mol`) return the report of the
command itself. `crystod_isotropy_subgroups` always returns the whole table:
the API has no maximum-index cut-off (there is no `max_index` argument).
The two tools that take a structure name their k points as the commands do,
in the frame of the labels of that structure (see
[Irrep labels and the frame of a structure](theory-representations.md#irrep-labels-and-the-frame-of-a-structure)):
for a POSCAR outside the ISO-IR setting, `qpoint="P"` and `kpoint="P"` can
be another arm than the coordinates `crystod_special_kpoints` lists (P of a
shifted I-4 cell is (-1/4, -1/4, -1/4), with `P1`, `P2`, ... labels; the
table's (1/4, 1/4, 1/4) carries `PA` labels there).

A tool call with a bad input returns one sentence naming the problem and the
remedy -- the tabulated irreps of the space group, the class order of the
character table, the special k points of the space group -- instead of a
traceback, so the assistant can correct itself and call again. For example
`crystod_isotropy_subgroups("Pm-3m", "R9+")` answers

```
irrep "R9+" is not tabulated for space group Pm-3m (No. 221). Available irreps:
GM1+, GM2+, GM3+, GM4+, GM5+, GM1-, ..., M5-
```

The same holds where the CrystOD code underneath stops with `SystemExit`, as
the command-line implementations do on bad input (or with
`KeyboardInterrupt`): the tool body runs in a worker thread, and the
exception becomes an error result inside that thread, so the call returns
its message and the server keeps answering later calls.
`crystod_irrep_product("Pm3m", ["R4-", "R5+"])` answers

```
"Pm3m" is not recognized as a standard space-group symbol or number (1-230).
```

### Example

`crystod_isotropy_subgroups("Pm-3m", "R4+")` returns

```
## Isotropy subgroups of R4+ in Pm-3m (No. 221)

| Direction | Subgroup | No. | Size | Index | Free parameters |
|---|---|---|---|---|---|
| (0,0,a) | I4/mcm | 140 | 2 | 6 | 1 |
| (a,a,a) | R-3c | 167 | 2 | 8 | 1 |
| (0,a,a) | Imma | 74 | 2 | 12 | 1 |
| (0,a,b) | C2/m | 12 | 2 | 24 | 2 |
| (a,a,b) | C2/c | 15 | 2 | 24 | 2 |
| (a,b,c) | P-1 | 2 | 2 | 48 | 3 |

6 isotropy subgroup(s). Size: primitive-cell multiplication relative to the
parent; index: index of the subgroup in the parent; free parameters:
independent order-parameter components (1 = a single amplitude fixes the
structure).

Irrep labels follow ISO-IR (H. T. Stokes, B. J. Campbell and R. Cordes,
Acta Cryst. A69, 388 (2013), https://iso.byu.edu). CrystOD: H. Koiso and
Y. Mochizuki et al., Phys. Rev. B 110, 064104 (2024).
```

-- the same six subgroups as [`crystod-group --parent Pm-3m --irrep R4+`](crystod-group.md)
and the [Python API](python-api.md), with the citation footer every result
carries.

## Citation

The irrep labels come from the ISO-IR tables of the ISOTROPY Software Suite
that CrystOD bundles, and the analyses are CrystOD's, so every tool result
ends with the two citations (ISO-IR: H. T. Stokes, B. J. Campbell and
R. Cordes, Acta Cryst. A69, 388 (2013); CrystOD: H. Koiso and Y. Mochizuki
et al., Phys. Rev. B 110, 064104 (2024)). Please keep both when results
obtained through the server are published; see [Citation](citation.md).
`crystod_slab` results carry, before them, the papers to cite for the slabs:
R. Kiribayashi et al., ACS Appl. Mater. Interfaces 17, 61872-61887 (2025)
(https://doi.org/10.1021/acsami.5c17978), the paper of `crystod-slab`, and
pymatgen's slab construction (W. Sun and G. Ceder, Surf. Sci. 617, 53 (2013);
R. Tran et al., Sci. Data 3, 160080 (2016)).
The server returns computed results (labels, decompositions, subgroups),
never the tables themselves.

## Time limits and resources

Every call runs under a time limit of 60 s by default; the
`CRYSTOD_MCP_TIMEOUT` environment variable of the server process (seconds,
set in the client configuration as above) raises or lowers it, and a call
that exceeds it returns an error naming the limit and the variable instead of
hanging the client. The limit bounds the wait, not the computation: the tools that return a CrystOD report run it in a subprocess that is killed at the limit, while the group-theory, k-point, molecular, phonon and slab tools run in a worker thread that finishes in the background after the error has been returned (retrying a long scan with a larger limit while the first one is still running costs a second scan, and a `crystod_slab` call with an `output_dir` still writes its files when it finishes). The group-theory, k-point and molecular tools answer in
well under a second, except the reverse lookup: the first
`crystod_find_isotropy_irreps` call on a parent builds the isotropy table of
that parent (a few seconds; about a minute for the face-centred cubic groups
with their large W stars) and caches it under `$CRYSTOD_CACHE_DIR/isotropy`
(default `~/.cache/crystod`) of the server process, as
[`crystod-group --child`](crystod-group.md) does; later calls read it, and
`coupled=true` searches every irrep pair (about 10 s for Pm-3m). The phonon tools rebuild the force constants from
`FORCE_SETS` with phonopy -- a few seconds for the 4x4x4 SrTiO3 example, more
for large supercells -- and `crystod_imaginary_mode_subgroups` scans every q
point the supercell resolves. `crystod_slab` builds every termination with
pymatgen (about a second for SrTiO3 with `max_index=1`; a larger `max_index`
or a large cell takes longer). The server runs one calculation per call in a
worker thread (a subprocess for the tools that return a CrystOD report),
needs no network access and keeps no state between calls apart from that
cache. Standard output is the message stream of the stdio transport, so
whatever a tool body prints there (progress lines of the CrystOD modules,
such as the primitive-cell note of the vibration analysis behind
`crystod_phonon_activity`) is sent to the server's standard error instead.

## Tests

```bash
pip install -e "./crystod-mcp[test]"
python -m pytest -q crystod-mcp/tests
```

The tests call every tool on the inputs bundled with CrystOD
(`crystod.examples`: ScF3, SrTiO3 with its 4x4x4 `FORCE_SETS`, CH4, NH3),
check the results the CrystOD testsuite expects (Pm-3m R4+ -> I4/mcm, R-3c,
Imma, C2/m, C2/c, P-1; the imaginary R5- mode of SrTiO3 at -1.09 THz; Sc 3d
of ScF3 -> GM3+ + GM5+ at Gamma; CH4 -> Td with A1 + T2 for the H 1s SALCs;
the trilinear X2+ X3- GM5- coupling term of I4/mmm; 3 IR-active GM4- sets
and a silent GM5- set for SrTiO3; R4+ among the Pm-3m irreps giving I4/mcm
and the coupled R4+ + M3+ pair giving Pnma; the m-3m -> 4/mmm correlation
T2g -> B2g + Eg, GM4- -> DT1 + DT5 along Pm-3m GM-X and R4+ -> A1g + Eg of
I4/mcm; the three piezoelectric components d15, d31, d33 of 4mm),
confirm the citation footer on every result and the one-sentence errors
(also for a `SystemExit` or `KeyboardInterrupt` raised inside a tool, and
over stdio, where the server answers the next call after the `SystemExit`
of a misspelled space group), check that nothing a tool prints reaches
standard output, and connect to the server in-process and over stdio to list
the eighteen tools and make a call; 60 tests in all.

## License

MIT, as CrystOD. The MCP Python SDK the server is built on is MIT licensed
(Copyright (c) 2024 Anthropic, PBC); its notice is reproduced in
`crystod-mcp/LICENSE`.
