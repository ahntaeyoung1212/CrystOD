# crystod-search

Search the Materials Project and download crystal structures as POSCAR files
for the other CrystOD commands.

| I want to ... | command |
|---|---|
| list every polymorph of a composition | `crystod-search SrTiO3` |
| list the compounds of exactly these elements | `crystod-search Sr-Ti-O` |
| keep only the experimentally observed ones | `crystod-search Sr-Ti-O --experimental` |
| download one structure (conventional cell) | `crystod-search --get mp-5532` |
| download the primitive cell instead | `crystod-search --get mp-5532 --cell primitive` |
| download every listed structure | `crystod-search Sr-Ti-O --experimental --get` |

```{note}
`crystod-search` is the only CrystOD command that goes online. It needs a
network connection and a free Materials Project API key (see
[The API key](#the-api-key)).
```

## 48. Searching the Materials Project

*Example directory: `example/48_mp_search` (testsuite section 48); after pip install: `crystod-example 48`*

### Searching

```bash
crystod-search SrTiO3
```

```text
* Materials Project: formula SrTiO3 *
 5 materials, sorted by energy above hull

* Materials *
 Formula  Space group  Material ID  Band Gap (eV)  Energy Above Hull (eV/atom)  Sites
 SrTiO3   I4/mcm       mp-4651              1.856                        0.000     10
 SrTiO3   I4/mcm       mp-551830            1.787                        0.000     10
 SrTiO3   Pm-3m        mp-5229 *            1.766                        0.000      5
 SrTiO3   P6_3/mmc     mp-776018            1.736                        0.039     30
 SrTiO3   R-3          mp-aaaieiuj          4.075                        0.130     10

 * experimentally observed (the structure matches an ICSD or other experimental entry)
```

The table is the one on the website, in the same order (by energy above the
hull). A star after the ID marks a material that has been observed
experimentally. The band gaps are DFT values, typically below the measured
ones.

The query is read the way the website's search box reads it:

| query | lists |
|---|---|
| `SrTiO3` | every polymorph of this formula (`'*TiO3'`: `*` is any element) |
| `Sr-Ti-O` | the compounds of exactly these elements (`'Sr-*'`: Sr plus any one element) |
| `Sr,Ti,O` | every material that contains at least these elements |
| `ABO3` | anonymous formula: every letter is any element |
| `mp-5229` | that material (several: `mp-5229,mp-5532`) |

Filters narrow the list and can be combined:

| option | keeps |
|---|---|
| `--experimental` | experimentally observed materials only |
| `--stable`, `--ehull MAX` | materials on the convex hull, or at most `MAX` eV/atom above it |
| `--band-gap MIN MAX`, `--sites MIN MAX` | band gaps and cell sizes in a range |
| `--spg SPACEGROUP` | one space group (`139`, `I4/mmm`) |
| `--exclude EL ...` | materials without these elements |
| `--subsystems` | with `Sr-Ti-O`: also Sr, Ti, O, Sr-Ti, Sr-O and Ti-O |

`--sort` changes the order (`ehull`, `gap`, `sites`, `id`, `formula`, `spg`)
and `--max N` caps the list (default 1000).

### Downloading POSCAR files

```bash
crystod-search --get mp-5532
```

```text
* POSCAR files (standardized conventional cell, tolerance 0.1 A) *
 Wrote POSCAR_Sr2TiO4_I4mmm_mp-5532: Sr2TiO4, I4/mmm, 14 atoms
```

The file lands in the current directory. Its name tells which cell it holds:

| `--cell` | file | Sr2TiO4 |
|---|---|---|
| `conventional` (default) | `POSCAR_Sr2TiO4_I4mmm_mp-5532` | 14 atoms |
| `primitive` | `PPOSCAR_Sr2TiO4_I4mmm_mp-5532` | 7 atoms |

Both files describe the same crystal. The SALC and crystal-orbital analyses
of `crystod` convert their input to the primitive cell themselves, so they
give the same result for either file. The two names also let both cells sit
in one directory. Other options:

- `--get` without IDs, after a query, downloads every listed material.
- `--directory` writes `Sr2TiO4_I4mmm_mp-5532/POSCAR` (or `/PPOSCAR`) instead,
  one directory per material.
- `-o NAME` names the file for a single material.
- A file that is already there is kept if it is identical, and replaced only
  with `--force` if it differs.

### How it works

1. **Search.** The query goes to the Materials Project REST API (the
   `materials/summary` endpoint), sent with `requests` and your API key.
   The star marks the entries the Materials Project has matched to an
   experimental structure (ICSD and others), that is, the entries not
   flagged as theoretical.
2. **Material IDs.** The API returns every ID in an 8-letter form: the number
   written in base 26 with a = 0. For example, `mp-aaaaahtd` is `mp-5229`.
   `crystod-search` converts the IDs back to the numbers the website shows,
   up to `mp-3347529`. Newer IDs stay alphabetical (`mp-aaaieiuj`). Both
   forms are accepted as input.
3. **Structure.** The stored (relaxed) structure is standardized with spglib
   (pymatgen reads the structure and writes the POSCAR). The tolerances are
   0.1 Å and 5°, the values the Materials Project itself uses to assign
   space groups. The atoms are moved onto their ideal positions, so the file
   has the listed space group exactly, whatever tolerance a later CrystOD
   command uses. The stored cell of Sr3Ti2O7 (mp-3349) is an example:
   `crystod` reads it as C2/m, but the downloaded file is I4/mmm.
   `--tolerance` changes the 0.1 Å, and a warning appears if the written
   cell ends up with another space group.

### The API key

The key is free: log in at <https://next-gen.materialsproject.org/api> and
copy it. Then either export it or store it once in the pymatgen settings
file:

```bash
export MP_API_KEY=<your key>
```

```bash
pmg config --add PMG_MAPI_KEY <your key>
```

The key is sent only in the request header and is never printed or saved.
A missing or rejected key, or an unreachable server, stops the command with
a one-line `ERROR:` that says what to do.

### From Python

```python
from crystod import search

result = search.search_materials("Sr-Ti-O", experimental=True)
for material in result.materials:
    print(material.label, material.formula, material.space_group)

(sr2tio4,) = search.fetch_materials("mp-5532")  # or cell="primitive"
search.write_poscar(sr2tio4)                    # POSCAR_Sr2TiO4_I4mmm_mp-5532
```

Bad input raises `ValueError`. Key, network and server problems raise
`search.MaterialsProjectError`. See [crystod.search](api/search.md) for the
full API.

```{admonition} Citation
The data come from the Materials Project: A. Jain *et al.*, *APL Mater.*
**1**, 011002 (2013), <https://doi.org/10.1063/1.4812323> (licence CC BY
4.0). Every search prints this reference.
```
