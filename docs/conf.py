"""Sphinx configuration for SigFeatX."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import SigFeatX

project = "SigFeatX"
copyright = "2026, Diptiman Mohanta"
author = "Diptiman Mohanta"
release = SigFeatX.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
# Render numpydoc "Attributes" sections as a field list rather than
# individual object descriptions -- avoids "duplicate object description"
# warnings for dataclasses, whose fields autodoc already documents.
napoleon_use_ivar = True

# -- intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- HTML output --------------------------------------------------------------
# No custom static assets or templates yet -- html_static_path /
# templates_path are omitted rather than pointed at empty directories,
# since git doesn't track empty dirs and a fresh checkout (e.g. CI) would
# then be missing them, failing the -W (warnings-as-errors) docs build.
html_theme = "sphinx_rtd_theme"
html_title = f"SigFeatX {release}"
