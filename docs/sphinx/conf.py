# Configuration file for the Sphinx documentation builder.
# CUDA Graph Routing Project

import os
import sys

# -- Project information -----------------------------------------------------
project = 'CUDA Graph Routing'
copyright = '2025, Gaurank Maheshwari - RIT CUDA Project'
author = 'Gaurank Maheshwari'
release = 'v1.0'
version = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz',
    'sphinx.ext.todo',
    'breathe',  # For Doxygen integration
    'myst_parser',  # For Markdown support
]

# Breathe Configuration (Doxygen integration)
breathe_projects = {"CUDA_Graph_Routing": "../api/xml"}
breathe_default_project = "CUDA_Graph_Routing"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix of source filenames.
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_static_path = ['_static']
html_title = 'CUDA Graph Routing Documentation'
html_short_title = 'CUDA Graph Routing'

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'CUDAGraphRouting.tex', 'CUDA Graph Routing Documentation',
     'Gaurank Maheshwari', 'manual'),
]

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Todo extension settings -------------------------------------------------
todo_include_todos = True

# -- Math settings -----------------------------------------------------------
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
