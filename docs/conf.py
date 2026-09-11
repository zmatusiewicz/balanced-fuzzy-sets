"""Sphinx and Read the Docs configuration for the API documentation."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

project = "Balanced Fuzzy Sets"
author = "Zofia Matusiewicz"
copyright = "2026, Zofia Matusiewicz"

try:
    release = package_version("balanced-fuzzy-sets")
except PackageNotFoundError:
    release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"
html_title = f"Balanced Fuzzy Sets {release}"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
html_context = {
    "display_github": True,
    "github_user": "zmatusiewicz",
    "github_repo": "balanced-fuzzy-sets",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
