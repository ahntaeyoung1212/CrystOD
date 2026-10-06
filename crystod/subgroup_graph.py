"""Group-subgroup graph of the isotropy subgroups of one or several irreps.

``crystod-group --parent SG --irrep IR [IR2 ...] --graph`` draws the
isotropy subgroups of the order parameter as a lattice: the parent at the
top, every stratum of the representation (for several irreps the strata of
the direct sum, including those in which only some of the irreps condense,
i.e. the single-irrep strata of every part), and the kernel (the stratum of
the generic order parameter) at the bottom, layered by index.  It is the
generalization of the group-subgroup figure of the perovskite tilt systems
of Howard and Stokes (Acta Cryst. B54, 782-789 (1998), Fig. 1): the graph
of ``--parent Pm-3m --irrep R4+ M3+`` contains their 15 tilt systems with
the group-subgroup relations among them (checked in the testsuite against
the list derived from their Glazer patterns).

How it works:

- The nodes are the strata of ``IsotropyAnalyzer.enumerate_directions`` on
  the (direct sum) representation, with the isotropy subgroup of each
  stratum identified as in ``--parent`` (type, cell size, index,
  conventional basis and origin).  Distinct strata have non-conjugate
  isotropy subgroups, so two nodes of the same type are distinct embeddings
  (their directions and settings tell them apart).
- An edge joins A to B when H_B is a subgroup of a conjugate of H_A: for
  isotropy subgroups this holds exactly when some image ``g Fix(H_A)`` of
  the fixed space of A lies in the fixed space of B, i.e. when
  ``P_B Q = Q`` for one projector ``Q = D(g) P_A D(g)^T`` of the orbit of
  ``P_A``.  The relation is transitively reduced (only the maximal
  subgroups among the nodes are joined).
- An edge from the parent is solid when the irrep that alone gives the
  stratum (its primary irrep) satisfies the Landau condition (no cubic
  invariant) and the Lifshitz condition (no Lifshitz invariant), so that a
  continuous transition is allowed by symmetry; dashed otherwise (also when
  the stratum needs several irreps at once).  Edges between strata are
  solid: whether the transition between two low-symmetry phases can be
  continuous (the cubic terms of the free energy restricted to the smaller
  fixed space) is not tested.

The HTML page ``SUBGROUP_<SG>_<irreps>.html`` holds the graph as an inline
SVG with a small script: hovering a node shows its direction, condensing
irreps, cell size, index, conventional basis and origin.  ``--graph-dot``
also writes the graph in the Graphviz DOT language.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

HOWARD_STOKES_CITATION = (
    'C. J. Howard and H. T. Stokes, "Group-Theoretical Analysis of Octahedral '
    'Tilting in Perovskites", Acta Cryst. B54, 782-789 (1998).'
)


# ---------------------------------------------------------------- data model


@dataclass(frozen=True)
class GraphNode:
    """One node of the group-subgroup graph.

    Attributes:
        id: Position of the node in ``SubgroupGraph.nodes`` (0-based; the
            command line prints it 1-based).
        kind: ``"parent"`` (the root), ``"stratum"``, or ``"kernel"`` (the
            stratum of the generic order parameter, the bottom of the graph).
        number: Space-group number of the (isotropy) subgroup.
        symbol: Its international short symbol.
        size: Primitive-cell multiplication relative to the parent.
        index: Index of the subgroup in the parent.
        direction: Order-parameter direction with the labels of the
            condensing irreps, e.g. ``"R4+(0,a,a) M3+(d;0;0)"``; ``""`` for
            the parent.
        n_free: Number of free order-parameter parameters.
        irreps: Labels of the irreps that condense (nonzero amplitude).
        primary: Labels of the irreps whose amplitude alone already has this
            isotropy subgroup (the others are secondary); empty when the
            stratum needs several irreps at once.
        basis: Conventional basis of the subgroup (rows, parent conventional
            units), or ``None``.
        origin: Origin of the subgroup cell (parent conventional
            coordinates), or ``None``.
        projector: Orthogonal projector onto the fixed space of the subgroup
            in the representation (zero for the parent).
    """

    id: int
    kind: str
    number: int
    symbol: str
    size: int
    index: int
    direction: str = ""
    n_free: int = 0
    irreps: tuple = ()
    primary: tuple = ()
    basis: np.ndarray | None = field(default=None, compare=False)
    origin: np.ndarray | None = field(default=None, compare=False)
    projector: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def label(self) -> str:
        """``"I4/mcm (140)"``."""
        return f"{self.symbol} ({self.number})"


@dataclass(frozen=True)
class GraphEdge:
    """One edge of the graph: ``child`` is a maximal subgroup of ``parent``
    among the nodes (up to conjugacy in the parent space group).

    Attributes:
        parent: ``id`` of the larger group.
        child: ``id`` of the subgroup.
        dashed: ``True`` for an edge from the parent whose primary irrep
            violates the Landau or the Lifshitz condition (or that needs
            several irreps), so that the transition cannot be continuous.
        index_ratio: ``index(child) / index(parent)``.
    """

    parent: int
    child: int
    dashed: bool = False
    index_ratio: int = 1


@dataclass
class SubgroupGraph:
    """The group-subgroup graph of the isotropy subgroups of some irreps.

    Returned by :func:`subgroup_graph`.

    Attributes:
        space_group: International short symbol of the parent.
        number: Its number.
        irreps: The irrep labels, in input order.
        dimensions: Order-parameter dimension of every irrep.
        nodes: The :class:`GraphNode` records, the parent first, then by
            index, number of free parameters and type.
        edges: The :class:`GraphEdge` records of the transitive reduction.
        inclusions: Every pair ``(a, b)`` of node ids with ``b`` a proper
            subgroup of a conjugate of ``a`` (the full relation, before the
            reduction).
        landau: Maps every irrep label to its ``LandauLifshitz`` record.
    """

    space_group: str
    number: int
    irreps: list
    dimensions: list
    nodes: list
    edges: list
    inclusions: set = field(default_factory=set)
    landau: dict = field(default_factory=dict)

    def contains(self, a: int, b: int) -> bool:
        """``True`` when node ``b`` is a proper subgroup of a conjugate of
        node ``a`` (both given by ``id``)."""
        return (int(a), int(b)) in self.inclusions

    def layers(self) -> list:
        """``(index, [nodes])`` for every index, from the parent down."""
        by_index: dict = {}
        for node in self.nodes:
            by_index.setdefault(node.index, []).append(node)
        return sorted(by_index.items())

    def above(self, node_id: int) -> list:
        """The edges ending at a node (its maximal supergroups)."""
        return [edge for edge in self.edges if edge.child == node_id]


# ---------------------------------------------------------------- building


def _transitive_reduction(n: int, relation: set) -> list:
    """Pairs of ``relation`` (a strict order on ``range(n)``) with nothing
    in between."""
    above = {b: {a for a, c in relation if c == b} for b in range(n)}
    edges = []
    for a, b in sorted(relation):
        if not any((a, c) in relation for c in above[b] if c != a):
            edges.append((a, b))
    return edges


def subgroup_graph(parent, irreps) -> SubgroupGraph:
    """The group-subgroup graph of the isotropy subgroups of some irreps.

    The data behind ``crystod-group --parent SG --irrep IR [IR2 ...]
    --graph``: the parent, every stratum of the (direct sum)
    representation, including those in which only some irreps condense,
    and the kernel, joined by the group-subgroup inclusions up to
    conjugacy, transitively reduced.

    Args:
        parent: Symbol (``"Pm-3m"``) or number of the parent space group.
        irreps: One ISO-IR label (``"R4+"``) or a list of labels
            (``["R4+", "M3+"]``).

    Returns:
        A :class:`SubgroupGraph`.

    Raises:
        SystemExit: Unknown space group or irrep (``ValueError`` through
            ``crystod.group``).

    Example:
        >>> from crystod import group
        >>> graph = group.subgroup_graph("Pm-3m", "R4+")
        >>> [node.label for node in graph.nodes]
        ['Pm-3m (221)', 'I4/mcm (140)', 'R-3c (167)', 'Imma (74)', 'C2/m (12)', 'C2/c (15)', 'P-1 (2)']
        >>> len(graph.edges)
        10
    """
    from .invariants import landau_lifshitz_of_elements
    from .isotropy_subgroup import (
        CoupledRepresentation,
        IsotropyAnalyzer,
        _orth_basis,
        _rep_cache,
    )

    labels = [irreps] if isinstance(irreps, str) else [str(x) for x in irreps]
    if not labels:
        raise SystemExit("ERROR: --graph needs at least one irrep label.")
    analyzer = IsotropyAnalyzer(str(parent), labels)
    algebra = analyzer.algebra
    representation = analyzer.representation
    coupled = isinstance(representation, CoupledRepresentation)
    parts = representation.parts if coupled else [representation]
    dims = [part.dimension for part in parts]
    bounds = np.cumsum([0] + dims)
    n = representation.dimension
    cache = _rep_cache(representation, analyzer.elements)
    elements = analyzer.elements
    landau = {part.label: landau_lifshitz_of_elements(part.elements, algebra.rotations)
              for part in parts}

    records = []
    for projector, _members in analyzer.enumerate_directions():
        label, generic = analyzer.direction_label(projector)
        nonzero = [
            j for j in range(len(parts))
            if np.linalg.norm(generic[bounds[j]:bounds[j + 1]]) > 1e-8
        ]
        if coupled:
            chunks = label.split(" ")
            direction = " ".join(chunks[j] for j in nonzero)
        else:
            direction = representation.label + label
        stabilizer = cache.vector_stabilizer_mask(generic)
        members = [(elements[g][0], elements[g][1]) for g in np.nonzero(stabilizer)[0]]
        info, size, index, B, rotations, translations, lattice = analyzer.subgroup_of(members)
        try:
            setting = analyzer.conventional_setting(B, rotations, translations, lattice, info)
        except Exception:  # noqa: BLE001 - the setting is optional
            setting = None
        count = int(stabilizer.sum())
        primary = []
        for j in nonzero:
            a, b = int(bounds[j]), int(bounds[j + 1])
            block = cache.E[:, a:b, a:b]
            piece = generic[a:b]
            alone = np.all(np.abs(block @ piece - piece[None]) <= 1e-6 + 1e-5 * np.abs(piece),
                           axis=1)
            if int(alone.sum()) == count:
                primary.append(parts[j].label)
        n_free = _orth_basis(projector).shape[1]
        records.append({
            "number": int(info.number), "symbol": str(info.international_short),
            "size": int(size), "index": int(index), "direction": direction,
            "n_free": int(n_free), "irreps": tuple(parts[j].label for j in nonzero),
            "primary": tuple(primary),
            "basis": None if setting is None else setting[0] + 0.0,
            "origin": None if setting is None else setting[1] + 0.0,
            "projector": np.real(np.asarray(projector, dtype=float)),
        })
    records.sort(key=lambda r: (r["index"], r["n_free"], r["number"], r["direction"]))

    nodes = []
    root_record = next((r for r in records if r["index"] == 1), None)
    if root_record is None:
        nodes.append(GraphNode(
            id=0, kind="parent", number=int(algebra.sg_type.number),
            symbol=str(algebra.sg_type.international_short), size=1, index=1,
            projector=np.zeros((n, n)),
        ))
    for record in records:
        if record is root_record:
            kind = "parent"
        elif record["n_free"] == n:
            kind = "kernel"
        else:
            kind = "stratum"
        node = GraphNode(id=len(nodes), kind=kind, **record)
        if kind == "parent":
            nodes.insert(0, node)
        else:
            nodes.append(node)
    nodes = [GraphNode(**{**_fields(node), "id": i}) for i, node in enumerate(nodes)]

    # inclusion: H_b <= g H_a g^-1  <=>  D(g) Fix(H_a) inside Fix(H_b)
    relation = set()
    for a in nodes:
        if a.kind == "parent":
            relation.update((a.id, b.id) for b in nodes if b.id != a.id)
            continue
        images = np.einsum("gab,bc,gdc->gad", cache.Ed, a.projector, cache.Ed)
        for b in nodes:
            if b.id == a.id or b.kind == "parent" or b.index <= a.index:
                continue
            if b.index % a.index:
                continue
            residual = np.einsum("ab,gbc->gac", b.projector, images) - images
            if np.any(np.max(np.abs(residual), axis=(1, 2)) < 1e-6):
                relation.add((a.id, b.id))

    edges = []
    for a, b in _transitive_reduction(len(nodes), relation):
        dashed = False
        if nodes[a].kind == "parent":
            primary = nodes[b].primary
            dashed = not any(landau[label].continuous_allowed for label in primary)
        edges.append(GraphEdge(parent=a, child=b, dashed=dashed,
                               index_ratio=nodes[b].index // nodes[a].index))
    return SubgroupGraph(
        space_group=str(algebra.sg_type.international_short),
        number=int(algebra.sg_type.number), irreps=[part.label for part in parts],
        dimensions=dims, nodes=nodes, edges=edges, inclusions=relation, landau=landau,
    )


def _fields(node: GraphNode) -> dict:
    return {name: getattr(node, name) for name in GraphNode.__dataclass_fields__}


# ---------------------------------------------------------------- text output


def _setting_text(values) -> str:
    from .isotropy_table import _setting_text as text

    return text(values)


def _origin_text(values) -> str:
    from .isotropy_table import _origin_text as text

    return text(values)


def graph_file_stem(graph: SubgroupGraph) -> str:
    """``SUBGROUP_<SG>_<irreps>`` (``/`` and spaces removed from the symbol)."""
    symbol = graph.space_group.replace("/", "").replace(" ", "")
    return "SUBGROUP_" + "_".join([symbol] + list(graph.irreps))


def format_graph_report(graph: SubgroupGraph, files=()) -> str:
    """The terminal report of ``--graph`` as text.

    Args:
        graph: The :class:`SubgroupGraph`.
        files: Paths of the files written (listed in ``* Output files *``).

    Returns:
        The report, ``* ... *`` blocks separated by blank lines.
    """
    import textwrap

    from .invariants import landau_lifshitz_lines

    lines = ["", "* Supergroup *", f"{graph.space_group} (No. {graph.number})", ""]
    lines.append("* Irrep *" if len(graph.irreps) == 1 else "* Coupled irreps *")
    for label, dim in zip(graph.irreps, graph.dimensions):
        lines.append(f"{label}: order parameter dimension {dim}")
        lines.extend(f"  {line}" for line in landau_lifshitz_lines(graph.landau[label]))
    lines.append("")
    n_dashed = sum(edge.dashed for edge in graph.edges)
    n_strata = sum(node.kind != "parent" for node in graph.nodes)
    lines.append("* Group-subgroup graph *")
    n_nodes = len(graph.nodes)
    lines.append(f"{n_nodes} {'node' if n_nodes == 1 else 'nodes'} "
                 f"(the parent and {n_strata} {'stratum' if n_strata == 1 else 'strata'}), "
                 f"{len(graph.edges)} edges ({n_dashed} dashed)")
    for index, layer in graph.layers():
        # labels are kept whole: their inner space is protected while wrapping
        text = f"index {index}: " + ", ".join(node.label.replace(" ", "\0")
                                              for node in layer)
        lines.extend(line.replace("\0", " ") for line in textwrap.wrap(
            text, width=78, initial_indent="  ", subsequent_indent="    ",
            break_on_hyphens=False))
    lines.append("")
    lines.append("* Nodes *")
    header = ["node", "subgroup", "size", "index", "direction", "maximal in", "from parent"]
    cells = []
    for node in graph.nodes:
        above = graph.above(node.id)
        edge_text = "-"
        if any(graph.nodes[edge.parent].kind == "parent" for edge in above):
            dashed = any(edge.dashed for edge in above
                         if graph.nodes[edge.parent].kind == "parent")
            edge_text = "dashed" if dashed else "solid"
        cells.append([
            str(node.id + 1), node.label, str(node.size), str(node.index),
            node.direction or "-",
            ",".join(str(edge.parent + 1) for edge in above) or "-", edge_text,
        ])
    widths = [max(len(header[c]), *(len(row[c]) for row in cells))
              for c in range(len(header))]
    for row in [header] + cells:
        lines.append("  ".join(t.ljust(w) for t, w in zip(row, widths)).rstrip())
    lines.append("maximal in: the nodes in which this subgroup is maximal (among the nodes)")
    lines.append("from parent: the line from the parent; dashed when the Landau or the")
    lines.append("Lifshitz condition of its irrep fails (no continuous transition)")
    if files:
        lines.append("")
        lines.append("* Output files *")
        lines.extend(f"  {path}" for path in files)
    lines.append("")
    lines.append("Conventions and validation: ISOSUBGROUP (https://iso.byu.edu):")
    lines.append('H. T. Stokes, S. van Orden and B. J. Campbell, "Tool for Generating')
    lines.append('Isotropy Subgroups of Crystallographic Space Groups",')
    lines.append("J. Appl. Cryst. 49, 1849-1853 (2016).")
    if graph.number == 221 and set(graph.irreps) <= {"R4+", "M3+"}:
        lines.append("Octahedral tilt systems of the perovskites: C. J. Howard and")
        lines.append('H. T. Stokes, "Group-Theoretical Analysis of Octahedral Tilting in')
        lines.append('Perovskites", Acta Cryst. B54, 782-789 (1998).')
    return "\n".join(lines)


def format_dot(graph: SubgroupGraph) -> str:
    """The graph in the Graphviz DOT language (layers as ``rank=same``)."""

    def quote(text: str) -> str:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = [f"digraph {quote(graph_file_stem(graph))} {{",
             "  rankdir=TB;",
             '  node [shape=box, fontname="Helvetica", fontsize=11];']
    for node in graph.nodes:
        text = node.label + (f"\\n{node.direction}" if node.direction else "")
        text = text.replace('"', '\\"')
        lines.append(f'  n{node.id + 1} [label="{text}"];')
    for _, layer in graph.layers():
        lines.append("  { rank=same; " + " ".join(f"n{node.id + 1};" for node in layer) + " }")
    for edge in graph.edges:
        style = " [style=dashed]" if edge.dashed else ""
        lines.append(f"  n{edge.parent + 1} -> n{edge.child + 1}{style};")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- HTML output

_BOX_HEIGHT = 40
_LAYER_GAP = 84
_NODE_GAP = 18
_CHAR_WIDTH = 7.2


def _layout(graph: SubgroupGraph) -> dict:
    """Node centres ``id -> (x, y)``, box widths and the canvas size: layers
    by index, the order within a layer by barycenters of the neighbours."""
    layers = [list(layer) for _, layer in graph.layers()]
    width_of = {
        node.id: max(len(node.label), len(node.direction)) * _CHAR_WIDTH + 22
        for node in graph.nodes
    }
    neighbours: dict = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        neighbours[edge.parent].append(edge.child)
        neighbours[edge.child].append(edge.parent)
    position = {}
    for layer in layers:
        for i, node in enumerate(layer):
            position[node.id] = (i + 0.5) / len(layer)
    for sweep in range(8):
        order = layers[1:] if sweep % 2 == 0 else layers[::-1][1:]
        for layer in order:
            def barycenter(node, layer=layer):
                linked = [position[j] for j in neighbours[node.id]]
                return (sum(linked) / len(linked)) if linked else position[node.id]
            layer.sort(key=lambda node: (barycenter(node), node.id))
            for i, node in enumerate(layer):
                position[node.id] = (i + 0.5) / len(layer)
    row_widths = [sum(width_of[node.id] for node in layer) + _NODE_GAP * (len(layer) - 1)
                  for layer in layers]
    margin_left = 92
    canvas_width = margin_left + max(row_widths) + 24
    centres = {}
    for row, layer in enumerate(layers):
        x = margin_left + (max(row_widths) - row_widths[row]) / 2
        y = 30 + row * _LAYER_GAP + _BOX_HEIGHT / 2
        for node in layer:
            centres[node.id] = (x + width_of[node.id] / 2, y)
            x += width_of[node.id] + _NODE_GAP
    canvas_height = 30 + (len(layers) - 1) * _LAYER_GAP + _BOX_HEIGHT + 30
    return {"centres": centres, "widths": width_of, "width": canvas_width,
            "height": canvas_height, "layers": layers}


def format_html(graph: SubgroupGraph) -> str:
    """The graph as a self-contained HTML page (inline SVG, hover details)."""
    from html import escape

    layout = _layout(graph)
    centres, widths = layout["centres"], layout["widths"]
    svg = [f'<svg id="graph" xmlns="http://www.w3.org/2000/svg" '
           f'width="{layout["width"]:.0f}" height="{layout["height"]:.0f}" '
           f'viewBox="0 0 {layout["width"]:.0f} {layout["height"]:.0f}">']
    for row, layer in enumerate(layout["layers"]):
        y = 30 + row * _LAYER_GAP + _BOX_HEIGHT / 2 + 4
        svg.append(f'<text class="layer" x="8" y="{y:.1f}">index {layer[0].index}</text>')
    for edge in graph.edges:
        xa, ya = centres[edge.parent]
        xb, yb = centres[edge.child]
        cls = "edge dashed" if edge.dashed else "edge"
        svg.append(
            f'<line class="{cls}" data-a="{edge.parent + 1}" data-b="{edge.child + 1}" '
            f'x1="{xa:.1f}" y1="{ya + _BOX_HEIGHT / 2:.1f}" '
            f'x2="{xb:.1f}" y2="{yb - _BOX_HEIGHT / 2:.1f}"/>'
        )
    for node in graph.nodes:
        x, y = centres[node.id]
        w = widths[node.id]
        data = {
            "id": str(node.id + 1), "label": node.label, "kind": node.kind,
            "direction": node.direction or "-", "irreps": " + ".join(node.irreps) or "-",
            "primary": " + ".join(node.primary) or "-",
            "size": str(node.size), "index": str(node.index), "nfree": str(node.n_free),
            "basis": _setting_text(node.basis) if node.kind != "parent" else "-",
            "origin": _origin_text(node.origin) if node.kind != "parent" else "-",
        }
        attributes = " ".join(f'data-{key}="{escape(value)}"' for key, value in data.items())
        svg.append(f'<g class="node {node.kind}" {attributes}>')
        svg.append(f'<rect x="{x - w / 2:.1f}" y="{y - _BOX_HEIGHT / 2:.1f}" '
                   f'width="{w:.1f}" height="{_BOX_HEIGHT}" rx="5"/>')
        if node.direction:
            svg.append(f'<text x="{x:.1f}" y="{y - 3:.1f}">{escape(node.label)}</text>')
            svg.append(f'<text class="dir" x="{x:.1f}" y="{y + 12:.1f}">'
                       f'{escape(node.direction)}</text>')
        else:
            svg.append(f'<text x="{x:.1f}" y="{y + 4:.1f}">{escape(node.label)}</text>')
        svg.append("</g>")
    svg.append("</svg>")
    title = f"Isotropy subgroups of {graph.space_group}: {' + '.join(graph.irreps)}"
    landau_text = "; ".join(
        f"{label}: {'continuous transition allowed' if record.continuous_allowed else 'Landau or Lifshitz condition fails'}"
        for label, record in graph.landau.items()
    )
    n_dashed = sum(edge.dashed for edge in graph.edges)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
 :root {{ --bg: #fafafa; --fg: #222; --box: #ffffff; --line: #546e7a; --muted: #607d8b;
         --accent: #1565c0; --parent: #e3f2fd; --kernel: #f1f8e9; --tip: #ffffff; }}
 @media (prefers-color-scheme: dark) {{
  :root {{ --bg: #1e2124; --fg: #e8eaed; --box: #2a2e33; --line: #9aa7b0; --muted: #9aa7b0;
          --accent: #64b5f6; --parent: #1c3346; --kernel: #26331c; --tip: #2a2e33; }}
 }}
 body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 0; background: var(--bg);
        color: var(--fg); }}
 #page {{ padding: 14px 18px; }}
 h1 {{ font-size: 19px; margin: 4px 0 6px; font-weight: 600; }}
 .meta {{ color: var(--muted); font-size: 12.5px; margin: 2px 0; }}
 #wrap {{ overflow-x: auto; margin-top: 10px; }}
 svg text {{ font-size: 12px; fill: var(--fg); text-anchor: middle; }}
 svg text.dir {{ font-size: 11px; fill: var(--muted); font-family: monospace; }}
 svg text.layer {{ text-anchor: start; fill: var(--muted); font-size: 11.5px; }}
 .node rect {{ fill: var(--box); stroke: var(--line); stroke-width: 1.2; }}
 .node.parent rect {{ fill: var(--parent); }}
 .node.kernel rect {{ fill: var(--kernel); }}
 .node:hover rect, .node.hi rect {{ stroke: var(--accent); stroke-width: 2.2; }}
 .edge {{ stroke: var(--line); stroke-width: 1.2; }}
 .edge.dashed {{ stroke-dasharray: 5 4; }}
 .edge.hi {{ stroke: var(--accent); stroke-width: 2.4; }}
 #tip {{ position: fixed; display: none; background: var(--tip); color: var(--fg);
        border: 1px solid var(--line); border-radius: 5px; padding: 7px 10px;
        font-size: 12.5px; pointer-events: none; max-width: 420px; }}
 #tip b {{ font-size: 13.5px; }}
 #tip td {{ padding: 1px 6px 1px 0; vertical-align: top; }}
 #tip td.v {{ font-family: monospace; }}
 #foot {{ color: var(--muted); font-size: 11.5px; margin-top: 10px; max-width: 900px; }}
</style>
</head>
<body>
<div id="page">
<h1>{escape(title)}</h1>
<div class="meta">{len(graph.nodes)} nodes, {len(graph.edges)} edges ({n_dashed} dashed), layered by index; hover a node for its setting.</div>
<div class="meta">{escape(landau_text)}</div>
<div id="wrap">
{chr(10).join(svg)}
</div>
<div id="foot">Nodes: the parent, every stratum (order-parameter direction type) of the
irreps and of their direct sum, and the kernel; several nodes of one type are distinct
embeddings. Lines join a group to its maximal subgroups among the nodes (up to
conjugacy). A dashed line from the parent: the Landau or Lifshitz condition of the
irrep fails, so the transition cannot be continuous; lines between strata are drawn
solid (their continuity is not tested). Directions and settings as in
crystod-group --parent {escape(graph.space_group)} --irrep {escape(' '.join(graph.irreps))}.
Reference: {escape(HOWARD_STOKES_CITATION)} Generated by CrystOD (crystod-group --graph).</div>
</div>
<div id="tip"></div>
<script>
(function () {{
  var tip = document.getElementById("tip");
  var rows = [["direction", "direction"], ["irreps", "irreps"], ["primary", "primary"],
              ["size", "size"], ["index", "index"], ["free parameters", "nfree"],
              ["conventional basis", "basis"], ["origin", "origin"]];
  function mark(id, on) {{
    document.querySelectorAll(".edge").forEach(function (line) {{
      if (line.dataset.a === id || line.dataset.b === id) line.classList.toggle("hi", on);
    }});
  }}
  document.querySelectorAll(".node").forEach(function (node) {{
    node.addEventListener("mouseenter", function () {{
      var d = node.dataset, html = "<b>" + d.id + ". " + d.label + "</b> (" + d.kind + ")<table>";
      rows.forEach(function (r) {{
        html += "<tr><td>" + r[0] + "</td><td class=\\"v\\">" + d[r[1]] + "</td></tr>";
      }});
      tip.innerHTML = html + "</table>";
      tip.style.display = "block";
      mark(d.id, true);
    }});
    node.addEventListener("mousemove", function (event) {{
      tip.style.left = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + "px";
      tip.style.top = (event.clientY + 14) + "px";
    }});
    node.addEventListener("mouseleave", function () {{
      tip.style.display = "none";
      mark(node.dataset.id, false);
    }});
  }});
}})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> None:
    """``crystod-group --parent SG --irrep IR [IR2 ...] --graph``
    (dispatched by cli/group.py)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Group-subgroup graph of the isotropy subgroups of some irreps."
    )
    parser.add_argument("--parent", required=True)
    parser.add_argument("--irrep", nargs="+", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--graph-dot", dest="graph_dot", action="store_true")
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(list(argv))
    graph = subgroup_graph(args.parent, args.irrep)
    html_path = args.output or graph_file_stem(graph) + ".html"
    directory = os.path.dirname(html_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(format_html(graph))
    files = [html_path]
    if args.graph_dot:
        dot_path = os.path.splitext(html_path)[0] + ".dot"
        with open(dot_path, "w", encoding="utf-8") as handle:
            handle.write(format_dot(graph))
        files.append(dot_path)
    print(format_graph_report(graph, files))
