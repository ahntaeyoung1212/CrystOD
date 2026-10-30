"""Figures of ``crystod-phonon --unfold`` (matplotlib, Agg).

Page 1: the parent's harmonic bands (or, without parent force data, the
child modes at K = M k without weights), the child unfolded onto the path
(one domain) and averaged over the orientation domains, with the weight as
marker area and color; below, every arm of the stars of the path vertices.
Page 2 (unless ``--no-irreps``): the parent bands colored by their parent
irrep, the unfolded child colored by the label of each degenerate group,
level panels at the special points of the path and a legend.

Every text is measured in the font in use (Arial, or DejaVu Sans where Arial
is not installed, whose glyphs are about 12 % wider) and fitted: titles and
notes are set smaller or wrapped to their room, labels that would run into
each other go into staggered rows or upright (the k labels of many star
arms), and crowded panels grow.

Conventions: those of :mod:`crystod.phonon_unfold` (THz, imaginary negative;
distances in the reciprocal metric of the pseudo-parent cell).  Called by
:func:`crystod.phonon_unfold.plot`; computes nothing but the drawing.
"""

from __future__ import annotations

import re

import numpy as np

# sequential ramp (light -> dark) for the weights, categorical slots for the irreps
BLUE_RAMP = [
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MIXED = "#898781"
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#c3c2b7", "#ecebe6"
MARKER = 7.0  # marker area (pt^2) per unit weight, page 1
GREEK = {"GM": r"\Gamma", "GAMMA": r"\Gamma", "DT": r"\Delta", "SM": r"\Sigma", "LD": r"\Lambda"}
_WORDS = {"GM": "Gamma", "DT": "Delta", "SM": "Sigma", "LD": "Lambda"}
PT_CM = 2.54 / 72.0  # one point in cm
LEVEL_FONT = 5.5  # pt, labels of the level panels (page 2)
LEVEL_FONT_TALL = 4.5  # pt, smallest font before the level panels grow taller
LEVEL_FONT_MIN = 3.0  # pt, smallest font; beyond it labels of the smallest P_g are left out
LEVEL_HEIGHT = 7.3  # cm, height of the band and level panels of page 2
LEVEL_HEIGHT_MAX = 16.0  # cm, the panels grow up to this height for crowded level panels
LEVEL_WIDTH = 1.6  # cm, width of a level panel when its labels fit
LEVEL_GAP = 0.1  # em, gap kept between the ink of two stacked labels of a level panel
PDF_LIFT = 1.0  # pt, matplotlib's PDF output draws mathtext up to this much higher than Agg
_RENDERER = None
_INK = {}  # (text, font family) -> ink of the text above and below its baseline, per em


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Arial where it is installed; elsewhere (most Linux systems) DejaVu Sans directly, because a
    # missing Arial costs one findfont warning per text (about 700 lines on stderr per run)
    arial = any(entry.name == "Arial" for entry in font_manager.fontManager.ttflist)
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"] if arial else ["DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "ytick.minor.width": 0.4,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "mathtext.default": "regular",
        }
    )
    return plt


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------
def kpoint_mathtext(label: str) -> str:
    """``'GM'`` -> Gamma, ``'X|R'`` -> ``'X|R'``, DT/SM/LD -> Greek (mathtext)."""
    return "|".join((f"${GREEK[p]}$" if p in GREEK else p) for p in str(label).split("|"))


def _irrep_part(name: str) -> str:
    m = re.match(r"^([A-Z]+)(\d+)([+-]?)$", name)
    if m is None:
        return name
    kt, index, sign = m.groups()
    base = GREEK.get(kt, r"\mathrm{%s}" % kt)
    return r"$%s_{%s}%s$" % (base, index, ("^{%s}" % sign) if sign else "")


def _split_unit(unit: str) -> list:
    name = re.compile(r"^[A-Z]+\d+[+-]?$")
    if name.match(unit):
        return [unit]
    for i, character in enumerate(unit):
        if character == "+" and name.match(unit[:i]) and name.match(unit[i + 1 :]):
            return [unit[:i], unit[i + 1 :]]
    return [unit]


def irrep_mathtext(label: str, mulliken: dict | None = None) -> str:
    """``'GM4-'`` -> Gamma_4^-; sets ``'A/B'`` and time-reversal units ``'S1+S2'`` part by part;
    Mulliken symbols (GM) appended in brackets."""
    units = str(label).split("/")
    text = "/".join("+".join(_irrep_part(p) for p in _split_unit(u)) for u in units)
    if mulliken:
        symbols = []
        for unit in units:
            parts = [mulliken.get(p) for p in _split_unit(unit)]
            if any(s is None for s in parts):
                return text
            symbols.append("+".join(dict.fromkeys(parts)))
        shown = []
        for s in symbols:
            mm = re.match(r"^([A-Z])(.*)$", s)
            shown.append(
                r"$\mathrm{%s}_{\mathrm{%s}}$" % (mm.group(1), mm.group(2))
                if mm and mm.group(2)
                else s
            )
        text += " [" + "/".join(shown) + "]"
    return text


def _kname(label: str) -> str | None:
    m = re.match(r"^([A-Z]+)\d", label)
    return m.group(1) if m else None


def _formula(symbols) -> str:
    from math import gcd

    counts = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    divisor = 0
    for n in counts.values():
        divisor = gcd(divisor, n)
    return "".join(f"{s}{n // divisor}" if n // divisor != 1 else s for s, n in counts.items())


def _compact(k) -> str:
    from .phonon_unfold import _frac_text

    return "(" + ",".join(_frac_text(v) for v in k) + ")"


def _matrix_compact(M) -> str:
    return (
        "[" + ",".join("[" + ",".join("%d" % v for v in row) + "]" for row in np.asarray(M)) + "]"
    )


def _arrow(text: str) -> str:
    return text.replace("->", r"$\rightarrow$").replace("Gamma", r"$\Gamma$")


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def _text_extent_pt(text: str, size: float, style: str = "normal") -> tuple:
    """Width and height (with the descent) in points of the layout box of a one-line text
    (mathtext between $ signs) at a font size (pt), in the font in use (rcParams).

    Measured at 720 dpi, where it agrees with the glyph outlines to about 1 %: at 72 dpi the
    advances of a text of a few points are hinted to whole pixels (Arial: 9 % too short at
    6.3 pt, 5 % too long at 5.5 pt).
    """
    import matplotlib
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties

    global _RENDERER
    if _RENDERER is None:
        _RENDERER = RendererAgg(10, 10, 720.0)  # 10 pixels per point
    prop = FontProperties(family=matplotlib.rcParams["font.family"], size=size, style=style)
    ismath = text.count("$") >= 2 and text.count("$") % 2 == 0
    width, height, _descent = _RENDERER.get_text_width_height_descent(text, prop, ismath=ismath)
    return float(width) / 10.0, float(height) / 10.0


def _text_width_pt(text: str, size: float, style: str = "normal") -> float:
    """Width in points of a text (mathtext between $ signs) at a font size (pt)."""
    return _text_extent_pt(text, size, style)[0]


def _fit_font(texts, size: float, width_pt: float, minimum: float) -> float:
    """The largest font size from ``minimum`` to ``size`` (pt) at which the widest of the texts is
    at most ``width_pt`` wide, measured in the font in use (Arial, or DejaVu Sans where Arial is
    missing, whose glyphs are about 12 % wider)."""
    texts = [t for t in texts if t]
    if not texts:
        return size
    widest = max(_text_width_pt(t, size) for t in texts)
    if widest <= width_pt:
        return size
    font = size * width_pt / widest  # the width is close to linear in the size
    for _ in range(8):  # hinting: shrink until it fits
        if font <= minimum or max(_text_width_pt(t, font) for t in texts) <= width_pt:
            break
        font *= 0.97
    return max(minimum, font)


def _wrap(text: str, size: float, width_pt: float, drawn=None) -> list:
    """Lines of ``text``, broken at spaces, each at most ``width_pt`` wide at ``size`` in the font
    in use; measured on the text as drawn (``drawn`` maps a line to it, e.g. :func:`_arrow`).
    A single word wider than the line keeps a line of its own."""
    drawn = drawn or (lambda line: line)
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and _text_width_pt(drawn(candidate), size) > width_pt:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _label_ink(texts) -> tuple:
    """Ink of texts (mathtext between $ signs) relative to their baseline, per em of the font size:
    ``(above, below)``, the largest extents over the texts.

    From the glyph outlines of the font in use (``TextPath``), because the placement of sub- and
    superscripts differs between fonts and matplotlib versions (matplotlib 3.10 sets them about
    0.4 em further apart with Arial than 3.11 does).
    """
    import matplotlib
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    family = tuple(np.atleast_1d(matplotlib.rcParams["font.family"]).tolist())
    above = below = 0.0
    for text in texts:
        key = (text, family)
        if key not in _INK:
            prop = FontProperties(family=list(family), size=100.0)
            box = TextPath((0, 0), text, size=100.0, prop=prop).get_extents()
            ok = np.all(np.isfinite([box.y0, box.y1]))
            _INK[key] = (max(0.0, box.y1 / 100.0), max(0.0, -box.y0 / 100.0)) if ok else (0.0, 0.0)
        above, below = max(above, _INK[key][0]), max(below, _INK[key][1])
    return above, below


def _rows(centres, widths, gap: float = 1.5) -> np.ndarray:
    """Row of every label (0, 1, ...) so that labels of one row clear each other by ``gap``:
    labels centred at ``centres`` (pt) with ``widths`` (pt), each in the first row where it
    clears the previous label of that row."""
    rows = np.zeros(len(centres), dtype=int)
    ends: list = []
    for i in np.argsort(centres, kind="stable"):
        left, right = centres[i] - widths[i] / 2, centres[i] + widths[i] / 2
        for r, end in enumerate(ends):
            if left >= end + gap:
                rows[i], ends[r] = r, right
                break
        else:
            rows[i] = len(ends)
            ends.append(right)
    return rows


def _label_rows(
    ticks, texts, width_cm: float, size: float, gap: float = 1.5, style: str = "normal"
) -> np.ndarray:
    """Row of every tick label (0, 1, ...) so that labels of one row do not overlap.

    The labels are centred on their ticks along an axis ``width_cm`` long;
    each goes to the first row where it clears the previous label of that row
    by ``gap`` points.  All zeros when the labels fit side by side.
    """
    ticks = np.asarray(ticks, dtype=float)
    span = ticks[-1] - ticks[0] if len(ticks) > 1 and ticks[-1] > ticks[0] else 1.0
    centres = (ticks - ticks[0]) / span * width_cm / PT_CM
    widths = np.array([_text_width_pt(t, size, style) if t else 0.0 for t in texts])
    return _rows(centres, widths, gap)


def _staggered_labels(ax, ticks, texts, rows, size, pad, side, color) -> None:
    """Tick labels drawn as texts in staggered rows below (``side='bottom'``) or above the axes."""
    from matplotlib.transforms import offset_copy

    line = 1.15 * size
    trans = ax.get_xaxis_transform()
    for t, text, row in zip(ticks, texts, rows):
        # below: tops aligned; above: baselines aligned (a '|' would lift a bottom-aligned text)
        shift = pad + row * line if side == "bottom" else pad + 0.25 * size + row * line
        ax.text(
            t,
            0.0 if side == "bottom" else 1.0,
            text,
            transform=offset_copy(
                trans, fig=ax.figure, y=-shift if side == "bottom" else shift, units="points"
            ),
            fontsize=size,
            color=color,
            ha="center",
            va="top" if side == "bottom" else "baseline",
            clip_on=False,
        )


def _fitted_rows(ticks, texts, width_cm, size: float):
    """``(font size, rows)`` of tick labels: the size itself when they fit side by side, else a
    font down to 85 % of it when that is enough, else the size with the labels in staggered rows."""
    if width_cm is None:
        return size, np.zeros(len(texts), dtype=int)
    rows = _label_rows(ticks, texts, width_cm, size)
    if rows.max(initial=0) == 0:
        return size, rows
    smaller = 0.85 * size
    if _label_rows(ticks, texts, width_cm, smaller).max(initial=0) == 0:
        return smaller, np.zeros(len(texts), dtype=int)
    return size, rows


def _decorate(ax, ticks, labels, ylim, width_cm=None, pad: float = 4.0):
    """Grid, limits and vertex labels of a band panel; labels that would overlap are made a
    little smaller or, if that is not enough, staggered."""
    import matplotlib
    from matplotlib.ticker import AutoMinorLocator

    for t in ticks[1:-1]:
        ax.axvline(t, color=GRID, lw=0.5, zorder=0)
    ax.axhline(0.0, color=INK, lw=0.5, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylim(*ylim)
    ax.set_xticks(ticks)
    texts = [kpoint_mathtext(t) for t in labels]
    default = float(matplotlib.rcParams["xtick.labelsize"])
    size, rows = _fitted_rows(ticks, texts, width_cm, default)
    if rows.max(initial=0) > 0:
        ax.set_xticklabels([""] * len(ticks))
        _staggered_labels(ax, ticks, texts, rows, size, pad, "bottom", INK)
    elif size != default:
        ax.set_xticklabels(texts, fontsize=size)
    else:
        ax.set_xticklabels(texts)
    ax.tick_params(axis="x", length=0, pad=pad)
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def _child_axis(ax, ticks, child_labels, fontsize=6.5, width_cm=None) -> float:
    """Child point K = M k of every tick above the panel (``->U``: folds onto the child U point).

    Returns the extra height (points) the labels take when they had to be
    staggered (0 when they fit in one row), for the title pad.
    """
    texts = [r"$\rightarrow$" + kpoint_mathtext(c) for c in child_labels]
    size, rows = _fitted_rows(ticks, texts, width_cm, fontsize)
    if rows.max(initial=0) > 0:
        _staggered_labels(ax, ticks, texts, rows, size, 1.5, "top", INK2)
        return float(rows.max()) * 1.15 * size
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(ticks)
    sec.set_xticklabels(texts, fontsize=size, color=INK2)
    sec.tick_params(axis="x", length=0, pad=1.5)
    for side in sec.spines.values():
        side.set_visible(False)
    return 0.0


def _plot_bands(ax, dist, freqs, segment_index, color, lw):
    for s in np.unique(segment_index):
        part = segment_index == s
        ax.plot(
            dist[part],
            freqs[part],
            color=color,
            lw=lw,
            solid_joinstyle="round",
            solid_capstyle="round",
        )


def _scatter(ax, x, f, w, cmap, norm, weight_min):
    keep = w >= weight_min
    order = np.argsort(w[keep], kind="stable")
    return ax.scatter(
        x[keep][order],
        f[keep][order],
        s=MARKER * w[keep][order],
        c=w[keep][order],
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        rasterized=False,
        zorder=3,
    )


def _ylim(*arrays):
    lo = min(float(np.min(a)) for a in arrays if a is not None and np.size(a))
    hi = max(float(np.max(a)) for a in arrays if a is not None and np.size(a))
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


def _notes(run, weight_min: float) -> list:
    """Figure notes of page 1, generated from the run."""
    from .phonon_unfold import _parent_point_name

    arms, davg, mapping, path = run.arms, run.domain_average, run.mapping, run.path

    def arm_text(i):
        return "%s %s" % (_WORDS.get(arms.star[i], arms.star[i]), _compact(arms.k_parent[i]))

    def point(name):
        return _WORDS.get(name, name)

    st = mapping.strain
    notes = [
        "Pseudo-parent cell inv(M) A_child: a = %.4f, %.4f, %.4f A, angles %.2f, %.2f, %.2f deg "
        "(strain %+.2f, %+.2f, %+.2f %%, volume %+.2f %% against the parent cell)."
        % (
            *st["lengths_pp"],
            *st["angles_pp"],
            *st["length_strain_percent"],
            st["volume_strain_percent"],
        )
    ]
    domains = "%d orientation domain%s" % (davg.n_domains, "s" if davg.n_domains != 1 else "")
    notes.append(
        (
            "Left panel: the harmonic bands of the parent, imaginary frequencies as negative "
            "numbers. "
            if run.parent_frequencies is not None
            else "Left panel: the child modes at K = M k, folded, without weights (no parent "
            "force data). "
        )
        + "Middle panel: one domain (child point K = M k above it). Right panel: average over "
        "all arms of the star of each k, i.e. over the %s of %s (a twinned crystal); m "
        "equivalent arms out of n carry m/n of the weight." % (domains, mapping.child_symbol)
    )
    on = [i for i in range(len(arms.star)) if arms.on_path[i] and arms.star[i] != "GM"]
    if on:
        notes.append(
            "Arms on the path: "
            + "; ".join("%s -> child %s" % (arm_text(i), point(arms.child_point[i])) for i in on)
            + "."
        )
    condensed = [arm_text(i) for i in range(len(arms.star)) if arms.condensed[i]]
    off = [arm_text(i) for i in range(len(arms.star)) if arms.condensed[i] and not arms.on_path[i]]
    text = "Shaded columns: arms that fold onto the child Gamma point (condensed): %s." % ", ".join(
        condensed
    )
    if off:
        text += " %s %s not on the path; only the lower panel shows %s." % (
            " and ".join(off),
            "is" if len(off) == 1 else "are",
            "it" if len(off) == 1 else "them",
        )
    notes.append(text)
    equivalent = []
    for star in dict.fromkeys(arms.star):
        idx = [i for i in range(len(arms.star)) if arms.star[i] == star]
        if len(idx) < 2:
            continue
        classes = sorted(set(int(arms.orbit[i]) for i in idx))
        if len(classes) == 1:
            equivalent.append("all %s arms equivalent" % point(star))
        else:
            equivalent.append(
                "%s: %s"
                % (
                    point(star),
                    " | ".join(
                        " = ".join(_compact(arms.k_parent[i]) for i in idx if arms.orbit[i] == c)
                        for c in classes
                    ),
                )
            )
    n_classes = np.bincount(davg.path_index, minlength=len(path.kpoints))
    lines = []
    for s in np.unique(path.segment_index):
        part = np.flatnonzero(path.segment_index == s)
        interior = part[1:-1] if len(part) > 2 else part
        if np.any(n_classes[interior] > 1):
            middle = path.kpoints[part[len(part) // 2]]
            name = _parent_point_name(mapping, middle) or "?"
            lines.append(point(name))
    lines = list(dict.fromkeys(lines))
    notes.append(
        "Arms give identical spectra only when a child operation or time reversal relates them. "
        "%s: %s; inequivalent arms on the lines: %s."
        % (
            mapping.child_symbol,
            "; ".join(equivalent) or "-",
            ", ".join(lines) if lines else "none",
        )
    )
    notes.append(
        "Weights: canonical per child mode; marker area and color proportional to the "
        "weight; modes with weight < %g not drawn." % weight_min
    )
    return notes


# ---------------------------------------------------------------------------
# page 1
# ---------------------------------------------------------------------------
ARM_FONT = 5.8  # pt, k labels of the star-arm panel
ARM_FONT_MIN = 4.0  # pt, smallest; with more arms than that fits, the panel widens
ARM_PANEL = 9.25  # cm, width of the star-arm panel when its labels fit
LABEL_GAP = 1.5  # pt, least room between two labels of one row
NOTE_FONT = 6.3  # pt, notes of page 1 (smaller, down to NOTE_FONT_MIN, when they do not fit)
NOTE_FONT_MIN = 4.5


def _arm_positions(arms):
    """x of every arm (one unit apart, 0.45 more between stars) and the stars as
    ``(star, first x, last x, number of arms)``."""
    xpos = np.zeros(len(arms.star))
    groups, x = [], 0.0
    for star in dict.fromkeys(arms.star):
        idx = [i for i in range(len(arms.star)) if arms.star[i] == star]
        start = x
        for i in idx:
            xpos[i] = x
            x += 1.0
        groups.append((star, start, x - 1.0, len(idx)))
        x += 0.45
    return xpos, groups


def _arm_label_layout(xpos, texts):
    """``(font, rotation, panel width in cm)`` of the k labels of the star-arm panel.

    Side by side at ARM_FONT (or 85 % of it) when they clear each other by
    LABEL_GAP; else upright (rotated by 90 deg) at the largest font, up to
    ARM_FONT, whose line height fits the spacing of the arms; when that font
    would be below ARM_FONT_MIN the panel is widened instead (more arms than
    the 26 of an fcc path).  Measured in the font in use.
    """
    x = np.sort(np.asarray(xpos, dtype=float))
    if len(x) < 2:
        return ARM_FONT, 0, ARM_PANEL
    units = x[-1] - x[0] + 1.2  # x range of the panel: 0.6 beyond the first and the last arm
    d = np.diff(x)
    per_unit = ARM_PANEL / PT_CM / units  # pt per x unit
    order = np.argsort(xpos, kind="stable")
    for size in (ARM_FONT, 0.85 * ARM_FONT):
        w = np.array([_text_width_pt(texts[i], size) for i in order])
        if np.all(0.5 * (w[1:] + w[:-1]) + LABEL_GAP <= d * per_unit + 1e-9):
            return size, 0, ARM_PANEL
    height = max(_text_extent_pt(t, 10.0)[1] for t in texts) / 10.0  # per pt of font
    font = (d.min() * per_unit - LABEL_GAP) / height
    if font >= ARM_FONT_MIN:
        return min(ARM_FONT, font), 90, ARM_PANEL
    per_unit = (ARM_FONT_MIN * height + LABEL_GAP) / d.min()
    return ARM_FONT_MIN, 90, per_unit * units * PT_CM


def _notes_layout(paragraphs, width_pt: float, height_pt: float):
    """``(font, line pitch in pt, wrapped paragraphs)`` of the notes of page 1: wrapped to
    ``width_pt`` as drawn, at the largest font from NOTE_FONT_MIN to NOTE_FONT whose lines (pitch
    1.26 em, 0.36 of a pitch between paragraphs) fit ``height_pt``."""

    def layout(size):
        wrapped = [_wrap(p, size, width_pt, _arrow) for p in paragraphs]
        n = sum(len(w) for w in wrapped)
        return wrapped, 1.26 * size * (n + 0.36 * (len(wrapped) - 1))

    size = NOTE_FONT
    wrapped, height = layout(size)
    while height > height_pt and size > NOTE_FONT_MIN:
        size = max(NOTE_FONT_MIN, size - 0.2)
        wrapped, height = layout(size)
    return size, 1.26 * size, wrapped


def _page_bands(plt, run, weight_min: float):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import AutoMinorLocator
    from matplotlib.transforms import offset_copy

    from .phonon_unfold import _tick_child_points

    mapping, path, bands, davg, arms = (
        run.mapping,
        run.path,
        run.bands,
        run.domain_average,
        run.arms,
    )
    cmap = LinearSegmentedColormap.from_list("unfold_blue", BLUE_RAMP)
    norm = Normalize(vmin=0.0, vmax=1.0)
    yb, hb, x0, wb, gap = 8.15, 6.0, 1.25, 4.5, 0.25
    # the star-arm panel and its rows of labels decide the width of the page
    xpos, groups = _arm_positions(arms)
    n_arms = len(xpos)
    k_texts = [_compact(k) for k in arms.k_parent]
    k_font, k_rotation, arm_w = _arm_label_layout(xpos, k_texts)
    x_notes = x0 + arm_w + 0.35
    W_cm, H_cm = max(17.0, x_notes + 6.15), 16.3
    fig = plt.figure(figsize=(W_cm / 2.54, H_cm / 2.54))

    def ax_cm(x, y, w, h):
        return fig.add_axes([x / W_cm, y / H_cm, w / W_cm, h / H_cm])

    ax0 = ax_cm(x0, yb, wb, hb)
    ax1 = ax_cm(x0 + wb + gap, yb, wb, hb)
    ax2 = ax_cm(x0 + 2 * (wb + gap), yb, wb, hb)
    cax = ax_cm(x0 + 3 * wb + 2 * gap + 0.22, yb, 0.2, hb)
    dist, ticks, seg = path.distances, path.tick_positions, path.segment_index
    ylim = _ylim(bands.frequencies, run.parent_frequencies)
    child, n_dom = mapping.child_symbol, davg.n_domains
    titles = [
        "%s parent" % mapping.parent_symbol
        if run.parent_frequencies is not None
        else "child at K = M k, folded",
        "%s child, one domain" % child,
        "%s child, %d-domain average" % (child, n_dom)
        if n_dom > 1
        else "%s child, average (1 domain)" % child,
    ]
    if run.parent_frequencies is not None:
        _plot_bands(ax0, dist, run.parent_frequencies, seg, INK, 0.6)
    else:
        _plot_bands(ax0, dist, bands.frequencies, seg, INK2, 0.4)
    ax0.set_ylabel("Frequency (THz)", fontsize=8, labelpad=4)
    nb = bands.frequencies.shape[1]
    sc = _scatter(
        ax1,
        np.repeat(dist, nb),
        bands.frequencies.ravel(),
        bands.weights_canonical.ravel(),
        cmap,
        norm,
        weight_min,
    )
    _scatter(
        ax2,
        np.repeat(dist[davg.path_index], nb),
        davg.frequencies.ravel(),
        davg.weights.ravel(),
        cmap,
        norm,
        weight_min,
    )
    for ax in (ax0, ax1, ax2):
        _decorate(ax, ticks, path.tick_labels, ylim, width_cm=wb)
    for ax in (ax1, ax2):
        plt.setp(ax.get_yticklabels(), visible=False)
    extra = _child_axis(ax1, ticks, _tick_child_points(path), width_cm=wb)
    # panel titles centred on their panels: each at most one panel pitch wide
    title_font = _fit_font(titles, 7.0, (wb + gap) / PT_CM - 2 * LABEL_GAP, 5.0)
    for ax, title in zip((ax0, ax1, ax2), titles):
        ax.set_title(title, fontsize=title_font, pad=13 + extra)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("unfolding weight (marker area and color)", fontsize=7)
    cb.ax.tick_params(labelsize=7, width=0.5, direction="in")
    cb.outline.set_linewidth(0.5)
    title = "%s %s -> %s   M = %s (N = %d)" % (
        _formula(mapping.parent_cell.symbols),
        child,
        mapping.parent_symbol,
        _matrix_compact(mapping.M),
        mapping.N,
    )
    size = _fit_font([title], 8.0, (W_cm - 0.8) / PT_CM, 5.0)
    fig.text(0.5, (H_cm - 0.25) / H_cm, title, fontsize=size, ha="center", va="top")
    # every arm of the stars of the path vertices; the rows of labels below the panel are laid
    # out in points: k (side by side or upright), child point, 'path' marks, stars
    per_unit = arm_w / PT_CM / (xpos.max() - xpos.min() + 1.2)  # pt per x unit
    k_extent = [_text_extent_pt(t, k_font) for t in k_texts]
    k_height = max(e[1] if k_rotation == 0 else e[0] for e in k_extent)
    child_texts = [r"$\rightarrow$" + kpoint_mathtext(c) for c in arms.child_point]
    child_font = 6.5
    child_rows = _rows(xpos * per_unit, [_text_width_pt(t, child_font) for t in child_texts])
    on = np.flatnonzero(arms.on_path)
    path_font = 6.0
    path_width = _text_width_pt("path", path_font, "italic")
    path_rows = _rows(xpos[on] * per_unit, [path_width] * len(on))
    star_texts = ["%s star" % kpoint_mathtext(star) for star, _a, _b, _n in groups]
    star_font = 7.0
    star_rows = _rows(
        np.array([0.5 * (a + b) for _s, a, b, _n in groups]) * per_unit,
        [_text_width_pt(t, star_font) for t in star_texts],
    )
    pad = 3.0
    y_child = pad + k_height + 2.0
    y_path = y_child + (child_rows.max(initial=0) + 1) * 1.15 * child_font + 0.5
    y_bracket = y_path + (path_rows.max(initial=-1) + 1) * 1.15 * path_font + 1.0
    y_star = y_bracket + 1.5
    stack = y_star + (star_rows.max(initial=0) + 1) * 1.15 * star_font
    ya = max(2.75, 0.3 + stack * PT_CM)
    ax3 = ax_cm(x0, ya, arm_w, 6.75 - ya)
    for i in range(n_arms):
        if arms.condensed[i]:
            ax3.axvspan(xpos[i] - 0.45, xpos[i] + 0.45, color=SHADE, lw=0, zorder=0)
        _scatter(
            ax3,
            np.full(nb, xpos[i]),
            arms.frequencies[i],
            arms.weights_canonical[i],
            cmap,
            norm,
            weight_min,
        )
    ax3.axhline(0.0, color=INK, lw=0.5, ls=(0, (4, 3)), zorder=1)
    ax3.set_xlim(xpos.min() - 0.6, xpos.max() + 0.6)
    ax3.set_ylim(*ylim)
    ax3.set_xticks(xpos)
    ax3.set_xticklabels(k_texts, fontsize=k_font, rotation=k_rotation)
    ax3.tick_params(axis="x", length=0, pad=pad)
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.set_ylabel("Frequency (THz)", fontsize=8, labelpad=4)
    arm_title = "every arm of the stars of the path vertices (parent fractional k)"
    ax3.set_title(arm_title, fontsize=_fit_font([arm_title], 7.0, arm_w / PT_CM, 5.0), pad=4)
    trans = ax3.get_xaxis_transform()

    def below(points):
        return offset_copy(trans, fig=fig, y=-points, units="points")

    for i in range(n_arms):
        ax3.text(
            xpos[i],
            0.0,
            child_texts[i],
            transform=below(y_child + child_rows[i] * 1.15 * child_font),
            fontsize=child_font,
            ha="center",
            va="top",
            color=INK2,
        )
    for i, row in zip(on, path_rows):
        ax3.text(
            xpos[i],
            0.0,
            "path",
            transform=below(y_path + row * 1.15 * path_font),
            fontsize=path_font,
            ha="center",
            va="top",
            color=INK2,
            style="italic",
        )
    for (star, a, b, n_star), text, row in zip(groups, star_texts, star_rows):
        ax3.text(
            0.5 * (a + b),
            0.0,
            text,
            transform=below(y_star + row * 1.15 * star_font),
            fontsize=star_font,
            ha="center",
            va="top",
            color=INK,
        )
        if n_star > 1:  # a bracket under the arms of one star (not under a single arm)
            ax3.plot(
                [a - 0.3, b + 0.3],
                [0.0, 0.0],
                transform=below(y_bracket),
                color=GRID,
                lw=0.5,
                clip_on=False,
            )
    # notes, right of the star-arm panel, wrapped to the column and fitted to its height
    top, bottom = 7.1, 0.3
    size, pitch, wrapped = _notes_layout(
        _notes(run, weight_min), (W_cm - x_notes - 0.25) / PT_CM, (top - bottom) / PT_CM
    )
    y = top / PT_CM
    for paragraph in wrapped:
        for line in paragraph:
            fig.text(
                x_notes / W_cm,
                y * PT_CM / H_cm,
                _arrow(line),
                fontsize=size,
                ha="left",
                va="top",
                color=INK2,
            )
            y -= pitch
        y -= 0.36 * pitch
    return fig


# ---------------------------------------------------------------------------
# page 2
# ---------------------------------------------------------------------------
def _repel(targets, h, lo, hi):
    """1-D label placement: positions in the order of their targets, at least ``h`` apart, inside
    ``[lo, hi]`` and as close to the targets as possible (least squares).

    With the labels in target order and ``q_i = p_i - i h`` the conditions read ``q`` non-decreasing
    and ``lo <= q_i <= hi - (n - 1) h``: an isotonic regression of ``t_i - i h`` (pool adjacent
    violators: blocks of labels stacked ``h`` apart, centred on the mean of their targets), cut to
    those bounds, which is the optimum with the bounds.  Every block is cut with the same bounds, so
    blocks pushed against an end of the range stack onto their neighbours instead of landing on
    them.  When ``n`` labels do not fit (``(n - 1) h > hi - lo``) they are spread evenly over
    ``[lo, hi]``.
    """
    order = np.argsort(targets)
    t = np.asarray(targets, dtype=float)[order]
    n = len(t)
    if n > 1 and (n - 1) * h > hi - lo:
        h = (hi - lo) / (n - 1)
    blocks = []  # [first label, number of labels, mean of t_i - i h]
    for i, v in enumerate(t - h * np.arange(n)):
        blocks.append([i, 1, v])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2] + 1e-12:
            first, m1, v1 = blocks[-2]
            _i, m2, v2 = blocks.pop()
            blocks[-1] = [first, m1 + m2, (m1 * v1 + m2 * v2) / (m1 + m2)]
    pos = t.copy()
    for first, m, _v in blocks:
        start = np.mean(t[first : first + m]) - 0.5 * h * (m - 1)
        start = min(max(start, lo + h * first), hi - h * (n - 1 - first))
        pos[first : first + m] = start + h * np.arange(m)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def _colors(units) -> dict:
    """Color of every unit: slot = rank of the unit among the units of its k type."""
    from .phonon_unfold_irreps import irrep_sort_key

    by_type = {}
    for unit in units:
        by_type.setdefault(_kname(unit), []).append(unit)
    colors = {}
    for kt, lst in by_type.items():
        for i, unit in enumerate(sorted(lst, key=irrep_sort_key)):
            colors[unit] = PALETTE[i % len(PALETTE)]
    return colors, by_type


def _group_color(label, ik, g, irreps, colors, index):
    if label == "mixed" or label == "-":
        return MIXED
    if "/" not in label:
        return colors.get(label, MIXED)
    parts = label.split("/")
    w = [float(irreps.weights[ik, g, index[p]].sum()) if p in index else -1.0 for p in parts]
    return colors.get(parts[int(np.argmax(np.round(w, 9)))], MIXED)  # ties: the first member


def _special_vertices(run, limit: int = 8):
    """Distinct path vertices that are ISO-IR special points (no free parameter), at most limit."""
    from .isoir import load_isoir_irreps, minus_k_type

    mapping, path = run.mapping, run.path
    try:
        irreps = load_isoir_irreps(mapping.parent_number)
        special = {ir.ktype for ir in irreps if ir.num_free_params == 0}
        special |= {minus_k_type(t, mapping.parent_number) for t in list(special)}
    except Exception:  # without the tables every vertex counts
        special = None
    out, seen = [], []
    for label, k, _K, point in path.vertices:
        if any(np.allclose(k, s) for s in seen):
            continue
        seen.append(k)
        ik = int(np.flatnonzero(np.all(np.abs(path.kpoints - k) < 1e-12, axis=1))[0])
        kname = run.irreps.kname[ik]
        if special is not None and kname not in special:
            continue
        out.append((label, k, point, ik, kname))
    return out[:limit]


def _level_items(run, ik: int, weight_min: float):
    """Bars and labels of the level panel at path index ``ik``.

    Returns ``(bars, items)``: bars ``(mean frequency, P_g, unit weights)`` of
    every degenerate group with ``P_g >= weight_min``; items ``([frequencies],
    label, [P_g])`` of the labels (``P_g >= 0.5``, same labels within 1 THz
    merged), in frequency order.
    """
    bands, ir = run.bands, run.irreps
    gid = bands.degenerate_group[ik]
    bars, entries = [], []
    for ig in np.unique(gid):
        g = np.flatnonzero(gid == ig)
        pg = float(bands.weights_canonical[ik, g].sum())
        if pg < weight_min:
            continue
        fm = float(bands.frequencies[ik, g].mean())
        bars.append((fm, pg, ir.weights[ik, g].sum(axis=0)))
        entries.append((fm, pg, ir.group_label[ik, g[0]]))
    items = []
    for fm, wm, name in sorted(entries):
        if wm < 0.5 or name in ("mixed", "-"):
            continue
        if items and items[-1][1] == name and fm - items[-1][0][-1] < 1.0:
            items[-1][0].append(fm)
            items[-1][2].append(wm)
        else:
            items.append(([fm], name, [wm]))
    return bars, items


def _level_pitch(font: float, ink=(0.0, 0.0)) -> float:
    """Distance (cm) between two stacked labels of a level panel at a font size (pt): 1.2 em, or
    more when the ink of the labels needs it.

    ``ink`` is ``(above, below)`` of :func:`_label_ink`.  The pitch keeps a gap of ``LEVEL_GAP`` em
    between the ink of neighbouring labels both where Agg draws them and where matplotlib's PDF
    output does: its vector layout of mathtext rounds the box height up to whole points and so
    lifts a label by up to ``PDF_LIFT`` (by a different amount for each label).
    """
    above, below = ink
    stacked = ((above + below + LEVEL_GAP) * font + PDF_LIFT) * PT_CM
    return max(0.032 * LEVEL_HEIGHT * font / LEVEL_FONT, stacked)


def _level_layout(panels, LX: float):
    """Height of the panels, font, shown labels and label pitch of each level panel, and the panel
    width.

    The labels of a panel are stacked at least one pitch apart inside the
    panel; the pitch (:func:`_level_pitch`) follows the ink of the panel's
    labels in the font in use.  When they do not fit at 5.5 pt the panels
    grow taller (labels at 4.5 pt, up to 16 cm), then the font shrinks (down
    to 3 pt), and beyond that only the labels of the largest P_g are shown.
    The panels widen when the widest label would run into the next panel.

    Returns ``(height_cm, fonts, shown, pitches_cm, width_cm, bars_cm)``; ``shown``
    holds the items drawn in each panel.
    """
    inks = [_label_ink([irrep_mathtext(name) for _f, name, _w in items]) for _bars, items in panels]
    hb = LEVEL_HEIGHT
    need = max(
        [len(p[1]) * _level_pitch(LEVEL_FONT_TALL, ink) for p, ink in zip(panels, inks)] + [0]
    )
    if need > hb:
        hb = min(LEVEL_HEIGHT_MAX, need)
    fonts, shown, pitches = [], [], []
    for (_bars, items), ink in zip(panels, inks):
        n = len(items)
        font = LEVEL_FONT
        if n * _level_pitch(font, ink) > hb + 1e-9:
            # the largest font with n pitches in the panel (the pitch is linear in the font)
            stacked = (hb / n / PT_CM - PDF_LIFT) / (ink[0] + ink[1] + LEVEL_GAP)
            font = max(
                LEVEL_FONT_MIN, min(LEVEL_FONT * hb / (n * _level_pitch(LEVEL_FONT)), stacked)
            )
        pitch = _level_pitch(font, ink)
        keep = min(n, int(np.floor(hb / pitch + 1e-9)))
        if keep < n:  # the labels of the largest P_g, in frequency order
            chosen = sorted(range(n), key=lambda i: (-max(items[i][2]), i))[:keep]
            items = [items[i] for i in sorted(chosen)]
        fonts.append(font)
        shown.append(items)
        pitches.append(pitch)
    bars_cm = LEVEL_WIDTH * LX / (LX + 1.13)
    widest = max(
        [
            _text_width_pt(irrep_mathtext(name), font) * PT_CM
            for font, items in zip(fonts, shown)
            for _f, name, _w in items
        ]
        + [0.0]
    )
    room = LEVEL_WIDTH - bars_cm + 0.05  # up to 0.03 cm before the next panel (0.08 cm apart)
    width = LEVEL_WIDTH if widest <= room else bars_cm + widest + 0.05
    return hb, fonts, shown, pitches, width, bars_cm


def _page_irreps(plt, run, weight_min: float):
    from matplotlib.ticker import AutoMinorLocator

    from .phonon_unfold import _tick_child_points
    from .phonon_unfold_irreps import irrep_sort_key

    ir, path, bands, mapping = run.irreps, run.path, run.bands, run.mapping
    units = list(ir.units)
    index = {u: i for i, u in enumerate(units)}
    colors, by_type = _colors(units)
    if ir.parent_labels is not None:
        for u in np.unique(ir.parent_labels):
            if u not in colors:
                colors[u] = PALETTE[len(colors) % len(PALETTE)]
    mulliken = ir.mulliken
    vertices = _special_vertices(run)
    n_lev = max(1, len(vertices))
    parent = ir.parent_labels is not None and run.parent_frequencies is not None
    rows_pts = [kt for kt in dict.fromkeys(ir.kname[v[3]] for v in vertices) if kt in by_type]
    rows_lin = [kt for kt in by_type if kt not in rows_pts]
    dy = 0.47
    # level panels at the special points: their labels fix the height and width of the panels
    panels = [_level_items(run, ik, weight_min) for _label, _k, _point, ik, _kn in vertices]
    pg_max = max([1.0] + [pg for bars, _items in panels for _fm, pg, _gw in bars])
    LX = max(2.12, pg_max + 0.15)
    hb, fonts, shown, pitches, level_w, bars_cm = _level_layout(panels, LX)
    # titles of the level panels: each within its slot (one panel pitch), at 6 pt or smaller;
    # below 4.5 pt the panels widen instead
    level_titles = [
        "%s  %s\n" % (kpoint_mathtext(label), _compact(k).replace(",", ", "))
        + r"$\rightarrow$ child "
        + kpoint_mathtext(point)
        for label, k, point, _ik, _kn in vertices
    ]
    lines = [line for t in level_titles for line in t.split("\n")]
    room = (level_w + 0.08) / PT_CM - LABEL_GAP
    level_title_font = _fit_font(lines, 6.0, room, 4.5)
    widest = max([_text_width_pt(t, level_title_font) for t in lines] + [0.0])
    if widest > room:
        level_w += (widest - room) * PT_CM
    step = level_w + 0.08

    def entry_width(unit, kt):  # marker, label and room to the next entry (cm)
        text = irrep_mathtext(unit, mulliken if kt == "GM" else None)
        fixed = (2.2 if kt == "GM" else 1.05) * (1.6 if len(_split_unit(unit)) > 1 else 1.0)
        return max(fixed, 0.28 + _text_width_pt(text, 6.3) * PT_CM + 0.2)

    def row_width(kt):
        return 0.75 + sum(entry_width(u, kt) for u in by_type[kt])

    mixed_text = "mixed (grey): no irrep and no inseparable set holds 50 % of P$_g$(k)"
    left_w = max(
        [row_width(kt) for kt in rows_pts] + [7.4, 0.38 + _text_width_pt(mixed_text, 6.3) * PT_CM]
    )
    column = max(9.6, left_w + 0.3)
    right_w = max([row_width(kt) for kt in rows_lin] + [0.0])
    W_cm = max(
        18.2 + max(0, n_lev - 4) * step + 4 * (step - 1.68) - (0.0 if parent else 4.95),
        0.35 + column + right_w + 0.3,
    )
    text = (
        "band panel of the child: one marker per degenerate group of child modes (|df| <= 1e-4 "
        "THz), area = P$_g$(k), P$_g$ < %g not drawn; a set A/B has the color of its larger "
        "member; colors repeat between k types (read them with the row of the k type). Level "
        "panels: one bar per degenerate group at K = M k, length P$_g$(k), split into its irrep "
        "weights; labels for P$_g$ >= 0.5 (same labels within 1 THz merged)." % weight_min
    )
    for (label, _k, _point, _ik, _kn), (_bars, items), drawn in zip(vertices, panels, shown):
        if len(drawn) < len(items):
            text += (
                " At %s %d of the %d labels are shown (those of the largest P$_g$; the terminal "
                "block lists every level)." % (kpoint_mathtext(label), len(drawn), len(items))
            )
    notes = _wrap(text, 6.0, (W_cm - 1.0) / PT_CM)
    # the two header lines, fitted (and if need be wrapped) to the page width
    header = (W_cm - 1.0) / PT_CM
    title = (
        "%s %s -> %s: unfolded phonons resolved by the irreps of the parent (CrystOD / ISO-IR "
        "labels, origin of the parent file)"
        % (_formula(mapping.parent_cell.symbols), mapping.child_symbol, mapping.parent_symbol)
    )
    title_font = _fit_font([title], 7.5, header, 6.0)
    title_lines = _wrap(title, title_font, header)
    second = (
        "M = %s (N = %d); color = label of each degenerate group (>= 50 %% of P$_g$ in one irrep "
        "or in an inseparable set), grey = mixed" % (_matrix_compact(mapping.M), mapping.N)
    )
    second_font = _fit_font([second], 6.3, header, 5.0)
    second_lines = _wrap(second, second_font, header)
    extra_h = (
        (len(title_lines) - 1) * 1.3 * title_font + (len(second_lines) - 1) * 1.3 * second_font
    ) * PT_CM
    n_rows = max(len(rows_pts) + 1, len(rows_lin))
    legend_h = n_rows * dy + 0.3 * len(notes) + 0.6
    H_cm = 10.75 + legend_h + (hb - LEVEL_HEIGHT) + extra_h
    fig = plt.figure(figsize=(W_cm / 2.54, H_cm / 2.54))

    def ax_cm(x, y, w, h):
        return fig.add_axes([x / W_cm, y / H_cm, w / W_cm, h / H_cm])

    yb = legend_h + 1.0
    x = 1.25
    axes_bands = []
    if parent:
        ax0 = ax_cm(x, yb, 4.6, hb)
        axes_bands.append(ax0)
        x += 4.95
    ax1 = ax_cm(x, yb, 4.6, hb)
    axes_bands.append(ax1)
    x += 4.95
    lev = [ax_cm(x + i * step, yb, level_w, hb) for i in range(len(vertices))]
    dist, ticks, seg = path.distances, path.tick_positions, path.segment_index
    ylim = _ylim(bands.frequencies, run.parent_frequencies if parent else None)
    band_titles = (["%s parent: irreps" % mapping.parent_symbol] if parent else []) + [
        "%s child: parent irreps" % mapping.child_symbol
    ]
    band_font = _fit_font(band_titles, 7.0, 4.95 / PT_CM - 2 * LABEL_GAP, 5.0)
    if parent:
        pf = run.parent_frequencies
        _plot_bands(ax0, dist, pf, seg, GRID, 0.45)
        pc = [colors.get(u, MIXED) for u in ir.parent_labels.ravel()]
        ax0.scatter(
            np.repeat(dist, pf.shape[1]), pf.ravel(), s=1.6, c=pc, edgecolors="none", zorder=3
        )
        ax0.set_ylabel("Frequency (THz)", fontsize=7, labelpad=3)
    xs, ys, ss, cs = [], [], [], []
    n_k = bands.frequencies.shape[0]
    for ik in range(n_k):
        gid = bands.degenerate_group[ik]
        for ig in np.unique(gid):
            g = np.flatnonzero(gid == ig)
            pg = float(bands.weights_canonical[ik, g].sum())
            if pg < weight_min:
                continue
            xs.append(dist[ik])
            ys.append(float(bands.frequencies[ik, g].mean()))
            ss.append(pg)
            cs.append(_group_color(ir.group_label[ik, g[0]], ik, g, ir, colors, index))
    xs, ys, ss, cs = np.array(xs), np.array(ys), np.array(ss), np.array(cs)
    order = np.argsort(ss, kind="stable")
    ax1.scatter(xs[order], ys[order], s=2.6 * ss[order], c=cs[order], edgecolors="none", zorder=3)
    for ax in axes_bands:
        _decorate(ax, ticks, path.tick_labels, ylim, width_cm=4.6, pad=3.0)
    if parent:
        plt.setp(ax1.get_yticklabels(), visible=False)
    else:
        ax1.set_ylabel("Frequency (THz)", fontsize=7, labelpad=3)
    extra = _child_axis(ax1, ticks, _tick_child_points(path), fontsize=6, width_cm=4.6)
    for ax, band_title in zip(axes_bands, band_titles):  # lifted over staggered child points
        ax.set_title(band_title, fontsize=band_font, pad=13 + extra)
    # level panels: one bar per degenerate group, split by its unit weights
    for ax, level_title, (bars, _items), font, items, pitch in zip(
        lev, level_titles, panels, fonts, shown, pitches
    ):
        for fm, _pg, gw in bars:
            x0 = 0.0
            for u in np.argsort(-gw, kind="stable"):
                if gw[u] < 1e-3:
                    continue
                ax.plot(
                    [x0, x0 + gw[u]],
                    [fm] * 2,
                    color=colors.get(units[u], MIXED),
                    lw=1.3,
                    solid_capstyle="butt",
                    zorder=3,
                )
                x0 += gw[u]
        if items:
            texts = [irrep_mathtext(name) for _f, name, _w in items]
            targets = np.array([np.mean(t[0]) for t in items])
            per_cm = (ylim[1] - ylim[0]) / hb  # data units per cm of the panel
            h = pitch * per_cm
            ypos = _repel(targets, h, ylim[0] + 0.5 * h, ylim[1] - 0.5 * h)
            # baselines placed so that the ink of the labels (with the PDF lift) is centred on ypos
            above, below = _label_ink(texts)
            centre = 0.5 * ((above - below) * font + PDF_LIFT) * PT_CM * per_cm
            for (_f, _name, wl), text, y, t in zip(items, texts, ypos, targets):
                ax.text(
                    LX,
                    y - centre,
                    text,
                    fontsize=font,
                    color=INK,
                    va="baseline",
                    ha="left",
                    zorder=4,
                )
                ax.plot([max(wl) + 0.05, LX - 0.05], [t, y], color=INK2, lw=0.3, zorder=2)
        ax.set_xlim(0.0, LX * level_w / bars_cm)
        ax.set_ylim(*ylim)
        xt = np.arange(0, int(np.floor(LX - 0.1)) + 1)
        ax.set_xticks(xt)
        ax.set_xticklabels(["%d" % v for v in xt], fontsize=5.5)
        ax.tick_params(axis="x", length=1.5, pad=1.5)
        ax.axhline(0.0, color=INK, lw=0.5, ls=(0, (4, 3)), zorder=1)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.setp(ax.get_yticklabels(), visible=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(level_title, fontsize=level_title_font, pad=3, linespacing=1.3)
    if lev:
        # under the level panels, at most as wide as the page and kept inside it
        caption = "unfolding weight P$_g$(k) of each degenerate group at the special point"
        span = lev[-1].get_position().x1 * W_cm - lev[0].get_position().x0 * W_cm
        size = _fit_font([caption], 5.5, max(span, 6.0) / PT_CM, 4.0)
        half = 0.5 * _text_width_pt(caption, size) * PT_CM
        x_mid = 0.5 * (lev[0].get_position().x0 + lev[-1].get_position().x1) * W_cm
        x_mid = min(max(x_mid, 0.25 + half), W_cm - 0.25 - half)
        fig.text(
            x_mid / W_cm,
            (yb - 0.75) / H_cm,
            caption,
            fontsize=size,
            color=INK2,
            ha="center",
            va="top",
        )
    # legend: one row per k type
    axl = ax_cm(0.35, 0.05, W_cm - 0.5, legend_h - 0.05)
    axl.set_xlim(0, W_cm - 0.5)
    axl.set_ylim(0.0, legend_h - 0.05)
    axl.axis("off")
    top = legend_h - 0.35
    for colx, rows in ((0.0, rows_pts), (column, rows_lin)):
        for r, kt in enumerate(rows):
            y = top - r * dy
            axl.text(colx, y, kpoint_mathtext(kt), fontsize=6.5, va="center", ha="left", color=INK)
            xx = colx + 0.75
            for unit in sorted(by_type[kt], key=irrep_sort_key):
                axl.scatter([xx + 0.1], [y], s=14, color=colors[unit], edgecolors="none")
                axl.text(
                    xx + 0.28,
                    y,
                    irrep_mathtext(unit, mulliken if kt == "GM" else None),
                    fontsize=6.3,
                    va="center",
                    ha="left",
                    color=INK,
                )
                xx += entry_width(unit, kt)
    y = top - len(rows_pts) * dy
    axl.scatter([0.1], [y], s=14, color=MIXED, edgecolors="none")
    axl.text(0.38, y, mixed_text, fontsize=6.3, va="center", ha="left", color=INK)
    y_note = top - n_rows * dy
    for i, line in enumerate(notes):
        axl.text(0.0, y_note - 0.3 * i, line, fontsize=6.0, va="center", ha="left", color=INK2)
    y = H_cm - 0.35
    for line in title_lines:
        fig.text(0.5 / W_cm, y / H_cm, line, fontsize=title_font, ha="left", va="top", color=INK)
        y -= 1.3 * title_font * PT_CM
    y = H_cm - 0.80 - (len(title_lines) - 1) * 1.3 * title_font * PT_CM
    for line in second_lines:
        fig.text(0.5 / W_cm, y / H_cm, line, fontsize=second_font, ha="left", va="top", color=INK2)
        y -= 1.3 * second_font * PT_CM
    return fig


def plot_unfold(pdf_path, run, *, weight_min: float = 0.02) -> int:
    """Write the PDF of a run; returns the number of pages (1 with ``--no-irreps``, else 2)."""
    plt = _style()
    from matplotlib.backends.backend_pdf import PdfPages

    pages = 0
    with PdfPages(pdf_path) as pdf:
        fig = _page_bands(plt, run, weight_min)
        pdf.savefig(fig)
        plt.close(fig)
        pages += 1
        if run.irreps is not None:
            fig = _page_irreps(plt, run, weight_min)
            pdf.savefig(fig)
            plt.close(fig)
            pages += 1
    return pages
