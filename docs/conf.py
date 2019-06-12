# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from recommonmark.transform import AutoStructify
from recommonmark.parser import CommonMarkParser

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'Kashgari'
copyright = '2019, BrikerMan'
author = 'BrikerMan'

# -- General configuration ---------------------------------------------------
source_suffix = ['.rst', '.md']

# source_parsers = {
#     '.md': CommonMarkParser,
# }

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    # 'recommonmark',
    'sphinx_markdown_tables',
    'm2r'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Autodoc settings
autodoc_default_flags = ['members', 'undoc-members', 'inherited-members']

# Autosummary settings
autosummary_generate = True

autoclass_content = "both"
autodoc_member_order = 'bysource'
master_doc = 'index'
pygments_style = 'sphinx'

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# https://help.farbox.com/pygments.html
pygments_style = 'tango'

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    # 'canonical_url': '',
    # 'analytics_id': '',
    # 'logo_only': False,
    'display_version': False,
    # 'prev_next_buttons_location': 'bottom',
    'prev_next_buttons_location': None,
    'style_external_links': False,
    # 'vcs_pageview_mode': '',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}


import kashgari
# The short X.Y version.
version = kashgari.__version__
# The full version, including alpha/beta/rc tags.
release = version


def setup(app):
    app.add_config_value('recommonmark_config', {
        # 'url_resolver': lambda url: github_doc_root + url,
        'auto_toc_tree_section': 'Contents',
        'enable_eval_rst': True,
    }, True)
    app.add_transform(AutoStructify)
    app.add_stylesheet('theme.css')


intersphinx_mapping = {
    'python': ('http://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
}
