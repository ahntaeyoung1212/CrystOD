"""Re-record mp_summary_Sr-Ti-O.json.gz, the offline fixture of testsuite section 48.

Needs network access and a Materials Project API key (MP_API_KEY, or
PMG_MAPI_KEY in the pymatgen settings file); the key is sent to the API
only and is never printed or written.  Run in this directory:

    python record_fixture.py

The testsuite itself never contacts the Materials Project: it serves this
file from a local stub server.
"""

from __future__ import annotations

import datetime
import gzip
import json
import os

import requests

URL = "https://api.materialsproject.org/materials/summary/"
OUTPUT = "mp_summary_Sr-Ti-O.json.gz"
SUBSYSTEMS = "O,Sr,Ti,O-Sr,O-Ti,Sr-Ti,O-Sr-Ti"
FIELDS = ("material_id,formula_pretty,symmetry,band_gap,energy_above_hull,nsites,"
          "theoretical,deprecated,elements,nelements,chemsys,formula_anonymous,"
          "is_stable,composition_reduced")


def api_key() -> str:
    key = os.environ.get("MP_API_KEY")
    if not key:
        from pymatgen.core import SETTINGS

        key = SETTINGS.get("PMG_MAPI_KEY")
    if not key or len(str(key).strip()) != 32:
        raise SystemExit("ERROR: set MP_API_KEY (32 characters, https://next-gen.materialsproject.org/api)")
    return str(key).strip()


def summary(session: requests.Session, params: dict) -> dict:
    response = session.get(URL, params=params, timeout=120)
    if response.status_code != 200:
        raise SystemExit(f"ERROR: HTTP {response.status_code}: {response.text[:200]}")
    return response.json()


def main() -> None:
    session = requests.Session()
    session.headers["x-api-key"] = api_key()

    # every summary document of Sr-Ti-O and its subsystems, in the order the
    # API returns them unsorted (the stub keeps that order); deprecated
    # entries too, which the API hides unless asked for with deprecated=true
    documents, meta = [], {}
    for deprecated in ("false", "true"):
        body = summary(session, {"chemsys": SUBSYSTEMS, "_fields": FIELDS,
                                 "_limit": "1000", "deprecated": deprecated})
        assert body["meta"]["total_doc"] == len(body["data"])
        documents.extend(body["data"])
        meta = body["meta"]

    # structures of the entries the tests download: every SrTiO3 polymorph and
    # every experimentally observed Sr-Ti-O ternary
    wanted = [d["material_id"] for d in documents if not d["deprecated"] and (
        d["formula_pretty"] == "SrTiO3"
        or (d["chemsys"] == "O-Sr-Ti" and d["theoretical"] is False))]
    body = summary(session, {"material_ids": ",".join(wanted),
                             "_fields": "material_id,structure", "_limit": "1000"})
    structures = {d["material_id"]: d["structure"] for d in body["data"]}
    assert len(structures) == len(wanted)
    for doc in documents:
        if doc["material_id"] in structures:
            doc["structure"] = structures[doc["material_id"]]

    fixture = {
        "description": ("Materials Project summary documents of the chemical system Sr-Ti-O "
                        "and all its subsystems (O, Sr, Ti, O-Sr, O-Ti, Sr-Ti, O-Sr-Ti), "
                        "deprecated entries included, in the order the API returned them "
                        "without a sort; 'structure' only for the entries the tests download."),
        "source": URL,
        "recorded": datetime.date.today().isoformat(),
        "api_version": meta.get("api_version"),
        "license": "CC BY 4.0 (Materials Project data)",
        "citation": ("A. Jain et al., APL Mater. 1, 011002 (2013). "
                     "https://doi.org/10.1063/1.4812323"),
        "query": {"chemsys": SUBSYSTEMS, "_fields": FIELDS + " (+ structure)",
                  "deprecated": "false, then true"},
        "documents": documents,
    }
    raw = json.dumps(fixture, separators=(",", ":")).encode()
    with open(OUTPUT, "wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as archive:
            archive.write(raw)
    print(f"{OUTPUT}: {len(documents)} documents ({sum(d['deprecated'] for d in documents)} "
          f"deprecated), {len(structures)} structures, {os.path.getsize(OUTPUT)} bytes")


if __name__ == "__main__":
    main()
