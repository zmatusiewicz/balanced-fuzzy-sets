"""Sphinx configuration for the Balanced Fuzzy Sets API documentation."""

project = "Balanced Fuzzy Sets"
author = "Zofia Matusiewicz"
copyright = "2026, Zofia Matusiewicz"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
exclude_patterns = ["_build"]
html_theme = "alabaster"
