# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import re
import json

sys.path.append(os.path.abspath('../surmise'))
sys.path.append(os.path.abspath('../surmise/emulationmethods'))
sys.path.append(os.path.abspath('../surmise/calibrationmethods'))
sys.path.append(os.path.abspath('../surmise/utilitiesmethods'))

latex_packages = [
    'xspace',
    'mathtools',
    'amsfonts', 'amsmath', 'amssymb'
]
latex_macro_files = ['base', 'notation']

# -- Project information -----------------------------------------------------

project = 'surmise'
copyright = '2023, Matthew Plumlee, Özge Sürer, Stefan M. Wild, Moses Y-H. Chan'
author = 'Matthew Plumlee, Özge Sürer, Stefan M. Wild, Moses Y-H. Chan'

# The full version, including alpha/beta/rc tags.
release = re.sub('^v', '', os.popen('git describe --tags').read().strip())
# The short X.Y version.
version = release
# release = '0.1'

needs_sphinx = '3.0'
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex'
]
autoclass_content = 'both'
autosummary_generate = True

todo_include_todos = True

# https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#substitutions
rst_prolog = ""
with open("sphinx_macros.json", "r") as fptr:
    macro_configs = json.load(fptr)
for key, value in macro_configs.items():
    rst_prolog += f".. |{key}| replace:: {value}\n"

mathjax3_config = {
    'loader': {},
    'tex': {
        'macros': {}
    }
}
for each in latex_macro_files:
    with open(f"latex_macros_{each}.json", "r") as fptr:
        macro_configs = json.load(fptr)
    for cmd_type, macros_all in macro_configs.items():
        for command, value in macros_all.items():
            assert command not in mathjax3_config['tex']['macros']
            mathjax3_config['tex']['macros'][command] = value

# bibliography file
bibtex_bibfiles = ['surmise.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
html_static_path = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# -- LaTeX configuration -----------------------------------------------------
# Some of this configuration is from
# https://stackoverflow.com/questions/9728292/creating-latex-math-macros-within-sphinx

latex_engine = "pdflatex"
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": ""
}
for package in latex_packages:
    latex_elements['preamble'] += f'\\usepackage{{{package}}}\n'

# Configure LaTeX with macros
for each in latex_macro_files:
    with open(f"latex_macros_{each}.json", "r") as fptr:
        macro_configs = json.load(fptr)
    for cmd_type, macros_all in macro_configs.items():
        for command, value in macros_all.items():
            if isinstance(value, str):
                macro = rf"\{cmd_type}{{\{command}}} {{{value}}}"
            elif len(value) == 2:
                value, n_args = value
                macro = rf"\{cmd_type}{{\{command}}}[{n_args}] {{{value}}}"
            else:
                raise NotImplementedError("No use case yet")
            latex_elements['preamble'] += (macro + "\n")
