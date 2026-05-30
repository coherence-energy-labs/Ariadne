"""Sphinx configuration for the Ariadne documentation site."""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable for autodoc without a full install
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

# -- Project information -----------------------------------------------------
project = "Ariadne"
author = "Ariadne project contributors"
copyright = "2026, Ariadne project contributors"

try:
    from ariadne import __version__ as version
except Exception:
    version = "1.0.0rc2"
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "myst_parser",                                 # Markdown support
]
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
myst_enable_extensions = ["colon_fence", "deflist", "tasklist"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"Ariadne {version}"
html_short_title = "Ariadne"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# Default to no fail-on-warning unless `-W` is passed at the command line
suppress_warnings = []
