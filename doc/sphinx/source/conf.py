# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
three_up = os.path.abspath(os.path.join('..', '..', '..'))
sys.path.insert(0, three_up)

from recommonmark.parser import CommonMarkParser


# -- Project information -----------------------------------------------------

project = 'torch-fidelity'
copyright = '2020-2023, Anton Obukhov'
author = 'Anton Obukhov'

with open(os.path.join(three_up, 'torch_fidelity', 'version.py')) as f:
    version_pycode = f.read()
exec(version_pycode)

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.intersphinx',
  'sphinx.ext.viewcode',
  'sphinx.ext.napoleon',
  'sphinx.ext.autosectionlabel',
  'sphinx_paramlinks',
  # 'sphinx_markdown_tables',
  'recommonmark',
]

source_parsers = {
    '.md': CommonMarkParser,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = ['.rst', '.md']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom_rtd.css',  # customizations of the read-the-docs CSS
]

# -- Other options -----------------------------------------------------------

# Set up external references to python and torch types and classes
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

autodoc_member_order = 'groupwise'

autoclass_content = 'both'

autodoc_inherit_docstrings = False

autodoc_default_options = {
    'members': True,
    'methods': True,
    'special-members': '__call__',
    'exclude-members': '_abc_impl',
    'show-inheritance': True,
}