"""Sphinx configuration for the frgfp documentation."""

import subprocess
from pathlib import Path

# -- Paths -------------------------------------------------------------------
DOC_SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOC_SOURCE_DIR.parents[1]

def _detect_version() -> str:
    """Resolve the version exactly as ``pyproject.toml`` does, with safe fallbacks."""
    # 1) The same mechanism pyproject.toml declares (setuptools_scm -> last git tag).
    try:
        from setuptools_scm import get_version
        return get_version(root=str(REPO_ROOT), relative_to=__file__)
    except Exception:
        pass
    # 2) The most recent git tag.
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL,
        ).strip().lstrip("v")
    except Exception:
        pass
    # 3) The installed distribution metadata.
    try:
        from importlib.metadata import version
        return version("frgfp")
    except Exception:
        return "0.0.0"

# -- Project information -----------------------------------------------------
project = "frgfp"
author = "Barnabas Molnar"
copyright = "2026, Barnabas Molnar"

release = _detect_version()
version = release.split("+")[0]

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "numpydoc",
    "pydata_sphinx_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = ".rst"
root_doc = "index"
language = "en"

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "../../frgfp_logo.png"

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/molbarnabas/frgfp",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "collapse_navigation": True,
    "navigation_depth": 4,
    "show_nav_level": 2,
    "show_prev_next": True,
    "search_bar_text": "Search the docs...",
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "footer_end": ["theme-version"],
}

html_context = {
    "github_user": "molbarnabas",
    "github_repo": "frgfp",
    "github_version": "main",
}

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autoclass_content = "both"

# -- Options for numpydoc ----------------------------------------------------
# autosummary renders the member tables, so numpydoc must not duplicate them.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_validate = False

# -- Options for napoleon (NumPy sections are handled by numpydoc) -----------
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- Options for autosummary -------------------------------------------------
autosummary_generate = True
autosummary_generate_overwrite = True
