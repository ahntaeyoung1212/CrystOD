"""Sphinx configuration for the CrystOD documentation (phonopy-style setup)."""

project = "CrystOD"
copyright = "2024-2026, Yasuhide Mochizuki and Hiroki Koiso"
author = "Yasuhide Mochizuki and Hiroki Koiso"

# single source of truth: the version in pyproject.toml. It is read straight out
# of the file next to this checkout whenever there is one -- the docs workflow
# installs the package editably (autodoc has to import it) and so does a
# developer checkout, and an editable install records whatever the version
# was at "pip install -e ." time, i.e. stale right after a bump.
# The installed metadata is the fallback, for a build outside a source tree.
def _read_version() -> str:
    import pathlib
    import re

    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        text = ""
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match:
        return match.group(1)
    try:
        from importlib.metadata import version as _package_version

        return _package_version("CrystOD")
    except Exception:
        return "unknown"


version = _read_version()
release = version

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    # doc/api/: the nine domain namespaces (crystod.salc, .group, ...)
    # documented from their docstrings
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# -- autodoc (doc/api/) -------------------------------------------------------
# The domain modules are lazy views (crystod/_api.py): ``__all__`` names the
# members and ``__getattr__`` imports the implementation module on first
# access, wrapping functions with functools.wraps.  autodoc documents exactly
# the names in ``__all__`` (an explicit ``__all__`` disables its
# "imported member" filter), and recovers the real signature through
# ``__wrapped__``; viewcode follows ``__module__`` to the implementation
# source, so the [source] links point at crystod/<implementation>.py.
#
# PySCF is an optional dependency (pip install "CrystOD[quantum]") and the
# docs workflow does not install it; nothing imports it at module level, so
# the mock is a safety net for the PySCF engines rather than a requirement.
autodoc_mock_imports = ["pyscf"]
# type hints go into the Args/Returns description, not into the signature
# line, which stays readable for the long keyword-only signatures; only the
# parameters an Args: section describes get one (the default would give every
# dataclass a bare "Parameters" list of types next to its Attributes: section)
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_member_order = "groupwise"
# Google-style docstrings only (the NumPy style is off, so a "Parameters"
# heading in prose is not mistaken for a section)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# the Attributes: section of a class docstring becomes a "Variables" field
# list (:ivar:) rather than one ".. attribute::" directive per entry: the
# dataclass fields are listed by :undoc-members: as well, and two directives
# for the same attribute are a "duplicate object description" error under -W
napoleon_use_ivar = True


def _reexported_data_comment(app, what, name, obj, options, lines):
    """Give a re-exported module constant the ``#:`` comment of its source.

    autodoc reads the comment of a data member from the source of the module
    being documented; ``crystod.mag.MULTIPOLE_NAMES`` and its kind are
    assigned in the implementation module (``crystod.spin_basis``) and only
    resolved through ``crystod.mag``, so their ``.. autodata::`` entries would
    come out bare.  The lazy namespaces record the implementation in
    ``_EXPORTS``; this follows it and copies the comment lines over.
    """
    if what != "data":
        return
    import importlib
    import inspect

    from sphinx.pycode import ModuleAnalyzer

    module_name, _, attribute = name.rpartition(".")
    exports = getattr(importlib.import_module(module_name), "_EXPORTS", {})
    if attribute not in exports:
        return
    implementation, implementation_attribute = exports[attribute]
    analyzer = ModuleAnalyzer.for_module(f"crystod.{implementation}")
    comment = analyzer.find_attr_docs().get(("", implementation_attribute))
    if comment:
        lines[:] = list(comment)
    elif "\n".join(lines).strip() == (inspect.getdoc(type(obj)) or "").strip():
        # a bare dict or list comes with the docstring of its type, which is
        # not documentation of the constant
        lines[:] = []


def setup(app):
    app.connect("autodoc-process-docstring", _reexported_data_comment)

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]
myst_heading_anchors = 3
# the README Changelog is {include}d with its own ###-level headings
suppress_warnings = ["myst.header"]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_title = f"CrystOD v{version}"
# the mark shown above the title at the top of the left sidebar; Sphinx copies
# it into _static/ on build (_static/logo.css sizes it and gives it a white
# plate on the dark theme, where the dark line-art would otherwise disappear)
html_logo = "_static/CrystOD_logo.png"
# browser-tab icon, matching how the laboratory site sets its own
# (<link rel="icon"> to a square PNG); the mark alone, without the wordmark,
# since it is drawn at 16-32 px
html_favicon = "_static/CrystOD_logo_only.png"
# the built HTML is published under the laboratory site, in the CrystOD/
# subdirectory; this fills in the canonical <link> of every page
html_baseurl = "https://mochizuki-tus.github.io/CrystOD/"
html_theme_options = {
    "repository_url": "https://github.com/ahntaeyoung1212/CrystOD",
    "use_repository_button": True,
    # expand the sidebar to the section level, so the feature list of every
    # command (the numbered sections) is visible at a glance; the sections
    # themselves are pulled in by the local _templates/sbt-sidebar-nav.html
    # override (the stock theme template lists page titles only)
    "show_navbar_depth": 2,
    "max_navbar_depth": 2,
    "show_toc_level": 2,
    "home_page_in_toc": True,
    # setting html_logo alone makes pydata drop the sidebar title; naming the
    # text here keeps the version visible underneath the mark
    "logo": {"text": html_title},
}
templates_path = ["_templates"]
# _static/embed/ holds interactive HTML outputs of the commands (BZ plots,
# SALC viewers, MO diagrams) embedded into the pages as iframes
html_static_path = ["_static"]
# sphinx-book-theme gives the left navigation sidebar a desktop show/hide
# button but leaves the right-hand "Contents" sidebar without one; these add
# the missing half (see the comments in the files themselves)
html_css_files = ["sidebar-toggle.css", "logo.css"]
html_js_files = ["sidebar-toggle.js", "embed-autosize.js"]
