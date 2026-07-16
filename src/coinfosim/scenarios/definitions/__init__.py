"""Built-in dataset-anchored scenario execution specs.

Each module here defines one ``DatasetAnchoredExecutionSpec`` for a built-in
scenario. These modules are intentionally not imported eagerly from this
package's ``__init__``: they pull in dataset loaders, model builders, and
report generators, and eager import here would slow down simple CLI
commands (``--help``, ``scenario list``) that never need them. Import the
specific submodule (or go through :mod:`coinfosim.scenarios.catalog`, which
imports lazily) instead of importing from this package directly.
"""
