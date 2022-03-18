# -*- coding: utf-8 -*-
import os
import sys

from datetime import datetime

sys.path.insert(0, os.path.abspath("../../"))

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_external_toc",
]

external_toc_exclude_missing = False
external_toc_path = "_toc.yml"


# This is used to suppress warnings about explicit "toctree" directives.
suppress_warnings = ["etoc.toctree"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_book_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "repository_url": "https://github.com/maxpumperla/rllib-trainer",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "doc/source",
    "home_page_in_toc": False,
    "show_navbar_depth": 0,
    "launch_buttons": {
        "notebook_interface": "jupyterlab",
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
    },
}

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Python methods should be presented in source code order
autodoc_member_order = "bysource"


def setup(app):
    app.add_css_file("css/custom.css", priority=800)
