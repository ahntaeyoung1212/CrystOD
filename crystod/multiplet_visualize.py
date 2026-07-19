"""Term-state visualization for crystod-group --multiplet --visualize.

For every term symbol of an irrep-shell configuration (e.g. (T2g)^2 (Eg)^1
in m-3m), the exact eigenstates are computed by projecting the Slater
determinant space onto the (S, irrep) block — the same machinery as the
multiplet energies — and written as an interactive HTML page: each state is
shown as its expansion in Slater determinants, every determinant drawn as an
orbital box diagram (dxy/dyz/dxz | dz2/dx2-y2 with up/down arrows) with the
exact expansion coefficient.

A term eigenstate is in general a superposition of determinants, not a
single box configuration; the page shows the full expansion. The states are
shown at the highest spin projection Ms = S, and the degenerate spatial
partners are canonicalized (RREF + Gram-Schmidt), so single-determinant
states (e.g. the ^4A2g of t2g^3) appear as one box diagram with
coefficient 1.
"""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np

from .ligand_field import ORBITAL_AZIMUTHAL_NUMBER  # noqa: F401 (re-export)
from .molecular_salc import ORBITAL_LABELS, _rref_orthogonal, format_salc
from .multiplet_energy import (
    _PARAM_NAMES,
    _apply_one_body,
    _coulomb_hamiltonians,
    _DeterminantSpace,
    _group_action,
    _s2_matrix,
    _shell_bases,
    real_two_electron_integrals,
)


def _coefficient_string(value: float) -> str:
    """Exact-looking display of an expansion coefficient: 1, -1/2, 1/√2,
    -√(2/3), ...; falls back to a decimal."""
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    as_fraction = Fraction(magnitude).limit_denominator(720)
    if abs(float(as_fraction) - magnitude) < 1e-9:
        if as_fraction == 1:
            return f"{sign}1"
        return f"{sign}{as_fraction.numerator}/{as_fraction.denominator}"
    squared = Fraction(magnitude * magnitude).limit_denominator(720)
    if abs(float(squared) - magnitude * magnitude) < 1e-9 and squared > 0:
        p, q = squared.numerator, squared.denominator
        root_q = int(q ** 0.5 + 0.5)
        if root_q * root_q == q:
            return f"{sign}√{p}/{root_q}" if root_q > 1 else f"{sign}√{p}"
        if p == 1:
            return f"{sign}1/√{q}"
        return f"{sign}√({p}/{q})"
    return f"{value:+.4f}"


def _canonical_shell_bases(bases):
    """Snap each shell's irrep basis onto canonical real-orbital combinations
    (RREF + Gram-Schmidt over the real-orbital components), so that e.g. the
    t2g basis of m-3m comes out as the pure dxy, dyz, dxz orbitals."""
    canonical = []
    for basis in bases:
        rows = [basis[:, k] for k in range(basis.shape[1])]
        snapped = _rref_orthogonal(rows)
        if len(snapped) != len(rows):
            canonical.append(basis)
        else:
            canonical.append(np.column_stack(snapped))
    return canonical


def _orbital_label(vector, l: int) -> str:
    return format_salc(vector, ORBITAL_LABELS[l])


def _det_boxes(det, shell_dims):
    """Determinant -> per-shell, per-orbital occupation code
    (0 empty, 1 up, 2 down, 3 up+down)."""
    offsets = np.cumsum([0] + list(shell_dims))
    codes = [[0] * dim for dim in shell_dims]
    for spin_orbital in det:
        orbital, spin = spin_orbital // 2, spin_orbital % 2
        shell = int(np.searchsorted(offsets, orbital, side="right")) - 1
        codes[shell][orbital - offsets[shell]] |= 1 if spin == 0 else 2
    return codes


def _one_rdm(space, sector, vector):
    """Spin-resolved one-particle reduced density matrices
    gamma^sigma_pq = <Psi| a+_{p sigma} a_{q sigma} |Psi> over the shell
    orbitals, for a state given on the determinant sector."""
    position = {space.dets[i]: row for row, i in enumerate(sector)}
    n_orb = space.n_orbitals
    gammas = [np.zeros((n_orb, n_orb)), np.zeros((n_orb, n_orb))]
    for row, det_index in enumerate(sector):
        det = space.dets[det_index]
        c_ket = vector[row]
        if abs(c_ket) < 1e-14:
            continue
        for q_so in det:
            spin = q_so % 2
            for p_orb in range(n_orb):
                result = _apply_one_body(det, 2 * p_orb + spin, q_so)
                if result is None:
                    continue
                sign, det_bra = result
                bra_row = position.get(det_bra)
                if bra_row is not None:
                    gammas[spin][p_orb, q_so // 2] += (
                        sign * vector[bra_row] * c_ket
                    )
    return gammas


def compute_term_states(character_table, l, shells, ordered_terms,
                        reference):
    """Explicit projected eigenstates of every term at Ms = S.

    Returns (shell_info, term_states) where shell_info carries the canonical
    orbital labels per shell and term_states is a list (parallel to
    ordered_terms) of branch lists; each branch is
    (reference_energy, [partner determinant expansions]), an expansion being
    a list of (per-shell box codes, coefficient)."""
    shell_irreps = [name for name, _ in shells]
    occupations = [count for _, count in shells]
    bases, operations, class_labels, d_matrices, class_index = _shell_bases(
        character_table, l, shell_irreps
    )
    bases = _canonical_shell_bases(bases)
    basis = np.hstack(bases)
    shell_dims = [b.shape[1] for b in bases]

    shell_info = []
    for (irrep, count), b in zip(shells, bases):
        full = [_orbital_label(b[:, k], l) for k in range(b.shape[1])]
        short = []
        for k, label in enumerate(full):
            # pure real orbital -> its own name; mixed combination -> a short
            # symbol, expanded in the legend of the HTML page
            pure = int(np.sum(np.abs(b[:, k]) > 1e-6)) == 1
            short.append(label if pure else f"{irrep.lower()}({k + 1})")
        shell_info.append({
            "irrep": irrep,
            "n": count,
            "labels": short,
            "full": full,
        })

    params = _PARAM_NAMES[l]
    n_params = len(params)
    exact = real_two_electron_integrals(l)
    size = 2 * l + 1
    integrals_by_param = []
    for p in range(n_params):
        tensor = np.zeros((size, size, size, size))
        for (a, b, c, d), coeffs in exact.items():
            tensor[a, b, c, d] = float(coeffs[p])
        transformed = np.einsum(
            "abcd,ap,bq,cr,ds->pqrs", tensor, basis, basis, basis, basis
        )
        entries = {}
        m = basis.shape[1]
        for a in range(m):
            for b in range(m):
                for c in range(m):
                    for d in range(m):
                        value = transformed[a, b, c, d]
                        if abs(value) > 1e-12:
                            entries[(a, b, c, d)] = value
        integrals_by_param.append(entries)

    orbital_matrices = []
    for dmat in d_matrices:
        m = basis.shape[1]
        u = np.zeros((m, m))
        offset = 0
        for b in bases:
            k = b.shape[1]
            u[offset:offset + k, offset:offset + k] = b.T @ dmat @ b
            offset += k
        orbital_matrices.append(u)

    space = _DeterminantSpace(shell_dims, occupations)
    n_electrons = sum(occupations)
    order = len(operations)
    class_names = list(character_table["rotation_list"])

    sector_cache: dict[int, dict] = {}
    term_states = []
    for spin, irrep, multiplicity in ordered_terms:
        sz2 = int(2 * spin)
        if sz2 not in sector_cache:
            sector = space.sector(sz2)
            sector_cache[sz2] = {
                "sector": sector,
                "h": _coulomb_hamiltonians(space, sector, integrals_by_param),
                "s2": _s2_matrix(space, sector),
                "group": _group_action(space, sector, orbital_matrices),
            }
        cache = sector_cache[sz2]
        sector, h_matrices, s2 = cache["sector"], cache["h"], cache["s2"]

        target = float(spin * (spin + 1))
        projector = np.eye(len(sector))
        s_iter = Fraction(sz2, 2)
        while s_iter <= Fraction(n_electrons, 2):
            other = float(s_iter * (s_iter + 1))
            if abs(other - target) > 1e-9:
                projector = projector @ (s2 - other * np.eye(len(sector))) / (
                    target - other
                )
            s_iter += 1

        characters = np.asarray(
            character_table["character_table"][irrep], dtype=float
        )
        dim = int(round(characters[class_index["E"]]))
        pg_projector = np.zeros((len(sector), len(sector)))
        for label, gmat in zip(class_labels, cache["group"]):
            pg_projector += characters[class_index[label]] * gmat
        pg_projector *= dim / order

        combined = pg_projector @ projector
        combined = (combined + combined.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(combined)
        kept = eigenvectors[:, eigenvalues > 0.5]
        if kept.shape[1] != multiplicity * dim:
            raise SystemExit(
                f"ERROR: projector rank {kept.shape[1]} != "
                f"{multiplicity * dim} for term ({spin}, {irrep})."
            )

        # split CI branches by the reference-point Coulomb energies
        reference_block = sum(
            r * (kept.T @ h @ kept) for r, h in zip(reference, h_matrices)
        )
        block_values, block_vectors = np.linalg.eigh(reference_block)
        groups: list[list[int]] = []
        for i in np.argsort(block_values):
            if groups and abs(block_values[groups[-1][0]] - block_values[i]) < 1e-6:
                groups[-1].append(int(i))
            else:
                groups.append([int(i)])

        branches = []
        for group in groups:
            vectors = [kept @ block_vectors[:, k] for k in group]
            snapped = _rref_orthogonal(vectors)
            if len(snapped) == len(vectors):
                vectors = snapped
            partners = []
            for vector in vectors:
                peak = np.max(np.abs(vector))
                entries = []
                for row, value in enumerate(vector):
                    if abs(value) < 1e-6 * max(peak, 1.0) or abs(value) < 1e-8:
                        continue
                    det = space.dets[sector[row]]
                    entries.append((abs(value), _det_boxes(det, shell_dims),
                                    float(value)))
                entries.sort(key=lambda e: -e[0])
                gamma_up, gamma_dn = _one_rdm(space, sector, vector)
                charge = basis @ (gamma_up + gamma_dn) @ basis.T
                spin_density = basis @ (gamma_up - gamma_dn) @ basis.T
                if abs(np.trace(charge) - n_electrons) > 1e-8 or abs(
                    np.trace(spin_density) - sz2
                ) > 1e-8:
                    raise SystemExit(
                        "ERROR: density-matrix trace check failed for term "
                        f"({spin}, {irrep}); please report this case."
                    )
                partners.append({
                    "dets": [
                        {"c": _coefficient_string(value),
                         "cf": round(value, 6), "boxes": boxes}
                        for _, boxes, value in entries
                    ],
                    "gc": [[round(float(v), 6) for v in row_]
                           for row_ in charge],
                    "gs": [[round(float(v), 6) for v in row_]
                           for row_ in spin_density],
                })
            branches.append({
                "eref": round(float(block_values[group[0]]), 6),
                "states": len(group) // max(dim, 1),
                "partners": partners,
            })
        term_states.append(branches)
    return shell_info, term_states


# ----------------------------------------------------------------- HTML page


_PAGE_SCRIPT = r"""
const DATA = __DATA__;
let curTerm = 0, curBranch = 0, curPartner = 0;

function subSup(target, symbol) {
  const sup = document.createElement('sup');
  sup.textContent = symbol.mult;
  target.appendChild(sup);
  const m = symbol.irrep.match(/^([A-Z])(.*)$/);
  target.appendChild(document.createTextNode(m ? m[1] : symbol.irrep));
  if (m && m[2]) {
    const sub = document.createElement('sub');
    sub.textContent = m[2];
    target.appendChild(sub);
  }
}

function boxSvg(boxes) {
  const shells = DATA.shells;
  const maxLabel = Math.max(...shells.flatMap(s => s.labels.map(l => l.length)));
  const BW = Math.max(34, maxLabel * 5.6 + 6), BH = 26, GAP = 8, SGAP = 22,
        PAD = 6, TAGW = 30;
  let width = PAD * 2 + TAGW;
  shells.forEach(s => {
    width = Math.max(width, PAD * 2 + TAGW + s.labels.length * (BW + GAP) - GAP);
  });
  const rowH = BH + 16;
  const height = PAD * 2 + shells.length * rowH + (shells.length - 1) * SGAP - 2;
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.setAttribute('class', 'boxes');
  // first shell drawn at the bottom (as in a ligand-field diagram)
  shells.forEach((shell, si) => {
    const y = PAD + (shells.length - 1 - si) * (rowH + SGAP);
    const rowW = shell.labels.length * (BW + GAP) - GAP;
    const x0 = TAGW + (width - TAGW - rowW) / 2;
    shell.labels.forEach((label, oi) => {
      const x = x0 + oi * (BW + GAP);
      const rect = document.createElementNS(NS, 'rect');
      rect.setAttribute('x', x); rect.setAttribute('y', y);
      rect.setAttribute('width', BW); rect.setAttribute('height', BH);
      rect.setAttribute('class', 'obox');
      svg.appendChild(rect);
      const code = boxes[si][oi];
      const arrows = [];
      if (code & 1) arrows.push('↑');
      if (code & 2) arrows.push('↓');
      arrows.forEach((a, k) => {
        const t = document.createElementNS(NS, 'text');
        t.setAttribute('x', x + BW / 2 + (arrows.length > 1 ? (k ? 8 : -8) : 0));
        t.setAttribute('y', y + BH - 6);
        t.setAttribute('text-anchor', 'middle');
        t.setAttribute('class', 'arrow' + (a === '↑' ? '' : ' dn'));
        t.textContent = a;
        svg.appendChild(t);
      });
      const lab = document.createElementNS(NS, 'text');
      lab.setAttribute('x', x + BW / 2); lab.setAttribute('y', y + BH + 12);
      lab.setAttribute('text-anchor', 'middle');
      lab.setAttribute('class', 'olab');
      lab.textContent = label;
      svg.appendChild(lab);
    });
    const tag = document.createElementNS(NS, 'text');
    tag.setAttribute('x', 2); tag.setAttribute('y', y + BH / 2 + 4);
    tag.setAttribute('class', 'stag');
    tag.textContent = shell.irrep.toLowerCase();
    svg.appendChild(tag);
  });
  return svg;
}

function render() {
  const term = DATA.terms[curTerm];
  document.querySelectorAll('.titem').forEach((n, i) =>
    n.classList.toggle('sel', i === curTerm));
  const head = document.getElementById('thead');
  head.textContent = '';
  subSup(head, term.symbol);
  const meta = document.getElementById('tmeta');
  meta.textContent = 'spatial dimension ' + term.dim
    + (term.mult > 1 ? ', ' + term.mult + ' independent occurrences '
       + '(configuration mixing)' : '')
    + ' — shown at Ms = S = ' + term.S;
  const en = document.getElementById('tenergy');
  en.textContent = term.energies.length
    ? 'E = ' + term.energies.join('   |   ') : '';

  const bnav = document.getElementById('bnav');
  bnav.textContent = '';
  if (term.branches.length > 1) {
    term.branches.forEach((b, k) => {
      const btn = document.createElement('button');
      btn.className = 'nbtn' + (k === curBranch ? ' sel' : '');
      btn.textContent = 'state ' + (k + 1) + ' (Eₑ = ' + b.eref.toFixed(3) + ')';
      btn.addEventListener('click', () => { curBranch = k; curPartner = 0; render(); });
      bnav.appendChild(btn);
    });
    const note = document.createElement('span');
    note.className = 'hint';
    note.textContent = ' Coulomb energy at the reference parameters';
    bnav.appendChild(note);
  }
  const branch = term.branches[curBranch];

  const pnav = document.getElementById('pnav');
  pnav.textContent = '';
  if (branch.partners.length > 1) {
    branch.partners.forEach((p, k) => {
      const btn = document.createElement('button');
      btn.className = 'nbtn' + (k === curPartner ? ' sel' : '');
      btn.textContent = k + 1;
      btn.addEventListener('click', () => { curPartner = k; render(); });
      pnav.appendChild(btn);
    });
    const note = document.createElement('span');
    note.className = 'hint';
    note.textContent = ' degenerate spatial partner';
    pnav.appendChild(note);
  }

  const list = document.getElementById('dets');
  list.textContent = '';
  const partner = branch.partners[Math.min(curPartner, branch.partners.length - 1)];
  partner.dets.forEach((entry, k) => {
    const row = document.createElement('div');
    row.className = 'detrow';
    const coeff = document.createElement('div');
    coeff.className = 'coeff';
    coeff.textContent = (k ? (entry.cf >= 0 ? '+ ' : '') : '') + entry.c;
    row.appendChild(coeff);
    row.appendChild(boxSvg(entry.boxes));
    list.appendChild(row);
  });
  drawDensity();
}

const side = document.getElementById('terms');
DATA.terms.forEach((term, i) => {
  const item = document.createElement('div');
  item.className = 'titem';
  subSup(item, term.symbol);
  if (term.ground) {
    const g = document.createElement('span');
    g.className = 'gnd';
    g.textContent = ' ground';
    item.appendChild(g);
  }
  item.addEventListener('click', () => { curTerm = i; curBranch = 0; curPartner = 0; render(); });
  side.appendChild(item);
});

// ---- charge / spin density (angular distribution) surface

let denYaw = -0.6, denPitch = 0.4;

function harmonics(x, y, z) {
  if (DATA.l === 0) return [0.2820948];
  if (DATA.l === 1) return [0.4886025*x, 0.4886025*y, 0.4886025*z];
  if (DATA.l === 2) return [
    1.0925484*x*y, 1.0925484*y*z, 0.3153916*(3*z*z - 1),
    1.0925484*x*z, 0.5462742*(x*x - y*y)];
  return [
    0.5900436*x*(x*x - 3*y*y), 0.5900436*y*(3*x*x - y*y),
    1.4453058*z*(x*x - y*y), 2.8906114*x*y*z,
    0.4570458*x*(5*z*z - 1), 0.4570458*y*(5*z*z - 1),
    0.3731763*z*(5*z*z - 3)];
}

function quadForm(G, v) {
  let total = 0;
  for (let a = 0; a < v.length; a++)
    for (let b = 0; b < v.length; b++) total += G[a][b] * v[a] * v[b];
  return total;
}

function drawDensity() {
  const view = document.getElementById('dview');
  if (!view) return;
  const term = DATA.terms[curTerm];
  const branch = term.branches[curBranch];
  const partner = branch.partners[Math.min(curPartner, branch.partners.length - 1)];
  const NT = 26, NP = 52;
  const cy2 = Math.cos(denYaw), sy2 = Math.sin(denYaw);
  const cp = Math.cos(denPitch), sp = Math.sin(denPitch);
  function rot(p) {
    const x1 = cy2 * p[0] + sy2 * p[2];
    const z1 = -sy2 * p[0] + cy2 * p[2];
    return [x1, cp * p[1] - sp * z1, sp * p[1] + cp * z1];
  }
  // grid of directions with density values
  const verts = [], dens = [], spins = [];
  let nmax = 1e-12;
  for (let it = 0; it <= NT; it++) {
    const th = Math.PI * it / NT, st = Math.sin(th), ct = Math.cos(th);
    for (let ip = 0; ip <= NP; ip++) {
      const ph = 2 * Math.PI * ip / NP;
      const u = [st * Math.cos(ph), st * Math.sin(ph), ct];
      const yv = harmonics(u[0], u[1], u[2]);
      const n = Math.max(0, quadForm(partner.gc, yv));
      const s = quadForm(partner.gs, yv);
      verts.push(u); dens.push(n); spins.push(s);
      if (n > nmax) nmax = n;
    }
  }
  const cx = 130, cyc = 115, scale = 100;
  const proj = verts.map((u, i) => {
    const r = dens[i] / nmax;
    return rot([u[0] * r, u[1] * r, u[2] * r]);
  });
  const light = [0.35, 0.3, 0.89];
  const quads = [];
  for (let it = 0; it < NT; it++) {
    for (let ip = 0; ip < NP; ip++) {
      const i00 = it * (NP + 1) + ip, i01 = i00 + 1;
      const i10 = i00 + NP + 1, i11 = i10 + 1;
      const P = [proj[i00], proj[i01], proj[i11], proj[i10]];
      const ax = P[1][0] - P[0][0], ay = P[1][1] - P[0][1], az = P[1][2] - P[0][2];
      const bx = P[3][0] - P[0][0], by = P[3][1] - P[0][1], bz = P[3][2] - P[0][2];
      let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
      const nn = Math.hypot(nx, ny, nz) || 1e-9;
      nx /= nn; ny /= nn; nz /= nn;
      if (nz < 0) { nx = -nx; ny = -ny; nz = -nz; }
      const bright = 0.52 + 0.48 * Math.max(0, nx * light[0] + ny * light[1] + nz * light[2]);
      const nAvg = (dens[i00] + dens[i01] + dens[i11] + dens[i10]) / 4;
      const sAvg = (spins[i00] + spins[i01] + spins[i11] + spins[i10]) / 4;
      const f = nAvg > 1e-9 * nmax ? Math.max(-1, Math.min(1, sAvg / nAvg)) : 0;
      // gray -> blue for up-spin excess, gray -> red for down
      const base = [176, 190, 197], up = [21, 101, 192], dn = [198, 40, 40];
      const tgt = f >= 0 ? up : dn, w = Math.abs(f);
      const rgb = base.map((c0, k) => Math.round(
        bright * ((1 - w) * c0 + w * tgt[k])));
      const depth = (P[0][2] + P[1][2] + P[2][2] + P[3][2]) / 4;
      const pts = P.map(p =>
        (cx + scale * p[0]).toFixed(1) + ',' + (cyc - scale * p[1]).toFixed(1)
      ).join(' ');
      quads.push([depth,
        '<polygon points="' + pts + '" fill="rgb(' + rgb.join(',') + ')"/>']);
    }
  }
  quads.sort((a, b) => a[0] - b[0]);
  view.innerHTML = quads.map(q => q[1]).join('');
}

const dview = document.getElementById('dview');
let denDrag = null;
dview.addEventListener('pointerdown', e => {
  denDrag = [e.clientX, e.clientY];
  dview.setPointerCapture(e.pointerId);
});
dview.addEventListener('pointermove', e => {
  if (!denDrag) return;
  denYaw += (e.clientX - denDrag[0]) * 0.012;
  denPitch += (e.clientY - denDrag[1]) * 0.012;
  denDrag = [e.clientX, e.clientY];
  drawDensity();
});
dview.addEventListener('pointerup', () => denDrag = null);

render();
"""


def write_term_state_html(
    output_path: str,
    point_group: str,
    config_label: str,
    shell_info: list[dict],
    ordered_terms,
    term_states,
    energies=None,
    ground_symbols=None,
    reference_note: str = "",
    l: int = 2,
) -> None:
    """Write the standalone term-state viewer page."""
    terms_json = []
    for index, (spin, irrep, multiplicity) in enumerate(ordered_terms):
        entry = {
            "symbol": {"mult": str(int(2 * spin + 1)), "irrep": irrep},
            "S": str(Fraction(spin)),
            "dim": (len(term_states[index][0]["partners"])
                    if term_states[index] else 0),
            "mult": multiplicity,
            "branches": term_states[index],
            "energies": list(energies[index].describe()) if energies else [],
            "ground": bool(ground_symbols and (spin, irrep) in ground_symbols),
        }
        terms_json.append(entry)

    data = {
        "pointGroup": point_group,
        "config": config_label,
        "shells": shell_info,
        "terms": terms_json,
        "l": l,
    }
    script = _PAGE_SCRIPT.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    shells_text = " ".join(
        f"({info['irrep']})^{info['n']}" for info in shell_info
    )
    splitting = " | ".join(
        f"{info['irrep'].lower()}: " + ", ".join(info["labels"])
        for info in shell_info
    )
    legend_lines = []
    for info in shell_info:
        for short, full in zip(info["labels"], info.get("full", info["labels"])):
            if short != full:
                legend_lines.append(f"{short} = {full}")
    legend_html = ""
    if legend_lines:
        rows = "<br>".join(legend_lines)
        legend_html = (
            '<div id="legend"><b>Orbital basis functions</b><br>'
            f"{rows}</div>"
        )
    note = (f"CI branches evaluated at {reference_note}. " if reference_note else "")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Multiplet states: {config_label} ({point_group})</title>
<style>
 body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; background: #fafafa; color: #222; }}
 #page {{ max-width: 1100px; margin: 0 auto; padding: 14px 18px; }}
 h1 {{ font-size: 19px; margin: 4px 0 6px; font-weight: 600; }}
 .chip {{ display: inline-block; background: #eceff1; border-radius: 4px; padding: 2px 9px;
          margin: 0 6px 6px 0; font-size: 12.5px; color: #37474f; }}
 #flex {{ display: flex; gap: 14px; align-items: flex-start; }}
 #terms {{ width: 130px; background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
           padding: 8px; font-size: 14px; }}
 .titem {{ padding: 5px 8px; border-radius: 4px; cursor: pointer; }}
 .titem:hover {{ background: #eceff1; }}
 .titem.sel {{ background: #1565c0; color: #fff; }}
 .gnd {{ font-size: 10px; color: #e65100; font-weight: 600; vertical-align: middle; }}
 .titem.sel .gnd {{ color: #ffe0b2; }}
 #main {{ flex: 1; background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
          padding: 14px 18px; min-height: 300px; }}
 #thead {{ font-size: 20px; font-weight: 600; }}
 #tmeta {{ color: #666; font-size: 12.5px; margin: 3px 0 6px; }}
 #tenergy {{ font-size: 13px; color: #37474f; margin-bottom: 8px; font-family: monospace; }}
 .nbtn {{ font-size: 12px; padding: 2px 10px; margin: 2px 6px 8px 0;
          border: 1px solid #b0bec5; border-radius: 3px; background: #eceff1; cursor: pointer; }}
 .nbtn.sel {{ background: #1565c0; color: #fff; border-color: #1565c0; }}
 .hint {{ color: #90a4ae; font-size: 11.5px; }}
 .detrow {{ display: flex; align-items: center; gap: 12px; padding: 6px 0;
            border-top: 1px solid #f0f0f0; }}
 .coeff {{ min-width: 84px; text-align: right; font-family: monospace; font-size: 14px; }}
 .obox {{ fill: #fff; stroke: #546e7a; stroke-width: 1.2; }}
 .arrow {{ font-size: 15px; fill: #1565c0; font-weight: 600; }}
 .arrow.dn {{ fill: #c62828; }}
 .olab {{ font-size: 10px; fill: #555; }}
 .stag {{ font-size: 10.5px; fill: #90a4ae; }}
 #legend {{ font-size: 12px; color: #546e7a; background: #f7f9fa;
            border: 1px solid #e0e6e9; border-radius: 5px;
            padding: 7px 10px; margin: 4px 0 8px; font-family: monospace; }}
 #dflex {{ display: flex; gap: 18px; align-items: flex-start; flex-wrap: wrap; }}
 #dets {{ flex: 1; min-width: 320px; }}
 #dpanel {{ width: 270px; }}
 #dhead {{ font-size: 12.5px; font-weight: 600; color: #37474f; margin: 6px 0 4px; }}
 #dview {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 5px;
           cursor: grab; touch-action: none; display: block; }}
 #foot {{ color: #777; font-size: 11.5px; margin-top: 10px; }}
</style>
</head>
<body>
<div id="page">
<h1>Multiplet term states: {config_label}</h1>
<div>
<span class="chip">{point_group}</span>
<span class="chip">{shells_text}</span>
<span class="chip">{splitting}</span>
</div>
<div id="flex">
<div id="terms"></div>
<div id="main">
<div id="thead"></div>
<div id="tmeta"></div>
<div id="tenergy"></div>
<div id="bnav"></div>
<div id="pnav"></div>
{legend_html}
<div id="dflex">
<div id="dets"></div>
<div id="dpanel">
<div id="dhead">Charge / spin density (angular part)</div>
<svg id="dview" viewBox="0 0 260 230" width="260" height="230"></svg>
<div class="hint">r(θ,φ) ∝ n(θ,φ) from the exact 1-RDM; color = spin
polarization (blue: ↑ excess) — drag to rotate</div>
</div>
</div>
</div>
</div>
<div id="foot">Exact projected eigenstates at the highest spin projection
Ms = S; each state is its full Slater-determinant expansion (the coefficients
are exact up to the printed precision). Degenerate spatial partners are
canonicalized (RREF + Gram-Schmidt); any orthogonal mixture is equivalent.
{note}Generated by CrystOD (crystod-group --multiplet --visualize).</div>
</div>
<script>
{script}
</script>
</body>
</html>
"""
    with open(output_path, "w") as handle:
        handle.write(html)
