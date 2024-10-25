import importlib
import inspect
import os

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "madnis"
copyright = "2024 Machine Learning for MadGraph5_aMC@NLO"
author = "Ramon Winterhalder, Theo Heimel"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    print(info["fullname"])
    filename = info["module"].replace(".", "/")
    return f"https://github.com/madgraph-ml/madnis/blob/main/{filename}.py"


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None

    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])

    try:
        file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except TypeError:
        # e.g. object is a typing.Union
        return None
    file = os.path.relpath(file, os.path.abspath(".."))
    start, end = lines[1], lines[1] + len(lines[0]) - 1

    return f"https://github.com/madgraph-ml/madnis/blob/main/{file}#L{start}-L{end}"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
}


autoclass_content = "both"
