"""Sphinx configuration for the CrystOD documentation (phonopy-style setup)."""

project = "CrystOD"
copyright = "2024-2026, Yasuhide Mochizuki and Hiroki Koiso"
author = "Yasuhide Mochizuki and Hiroki Koiso"

# single source of truth: the version from pyproject.toml (installed package),
# with a fallback for building the docs without an installed CrystOD
try:
    from importlib.metadata import version as _package_version

    version = _package_version("CrystOD")
except Exception:
    version = "0.3.3"
release = version

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

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
html_theme_options = {
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "home_page_in_toc": True,
}
html_static_path = []
