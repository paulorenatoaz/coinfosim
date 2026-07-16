"""
CoInfoSim - A Simulator for Cooperative Classification from Multiple Information Channels

A scientific Python package for evaluating when cooperation among information
channels improves supervised classification. CoInfoSim compares isolated
channels, channel pairs, and larger channel subsets through Monte Carlo
simulation of the average classification loss. It is an academic evolution of
the SLACGS and CoSenSim lines of work.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Public API (backward compatible). Every name here is resolved lazily via
# module-level __getattr__ (PEP 562): `coinfosim.core` and `coinfosim.reporting`
# transitively import matplotlib and scikit-learn, and this package is
# imported by every `coinfosim.*` submodule (including `coinfosim.cli`), so
# eager imports here would slow down `coinfosim --help` and similar
# lightweight commands even when they never touch simulation or reporting.
__all__ = [
    # Core simulation classes
    'Model',
    'Simulator',
    'DictionaryType',
    'LossType',
    # Reporting
    'Report',
    'create_scenario_report',
    # Configuration
    'load_config',
    'validate_config',
    'get_output_dir',
    'get_reports_dir',
    'get_data_dir',
    'get_log_file',
    'init_project_config',
    'init_user_config',
    'ConfigError',
    'DEFAULT_CONFIG',
    # Logging
    'setup_logging',
    'setup_logging_from_config',
    'get_logger',
    'reset_logging',
    'is_logging_configured',
    # Utils
    'init_report_service_conf',
]

_LAZY_ATTRIBUTES: dict[str, tuple[str, str]] = {
    'Model': ('coinfosim.core', 'Model'),
    'Simulator': ('coinfosim.core', 'Simulator'),
    'DictionaryType': ('coinfosim.core.enumtypes', 'DictionaryType'),
    'LossType': ('coinfosim.core.enumtypes', 'LossType'),
    'Report': ('coinfosim.reporting', 'Report'),
    'create_scenario_report': ('coinfosim.reporting', 'create_scenario_report'),
    'load_config': ('coinfosim.config', 'load_config'),
    'validate_config': ('coinfosim.config', 'validate_config'),
    'get_output_dir': ('coinfosim.config', 'get_output_dir'),
    'get_reports_dir': ('coinfosim.config', 'get_reports_dir'),
    'get_data_dir': ('coinfosim.config', 'get_data_dir'),
    'get_log_file': ('coinfosim.config', 'get_log_file'),
    'init_project_config': ('coinfosim.config', 'init_project_config'),
    'init_user_config': ('coinfosim.config', 'init_user_config'),
    'ConfigError': ('coinfosim.config', 'ConfigError'),
    'DEFAULT_CONFIG': ('coinfosim.config', 'DEFAULT_CONFIG'),
    'setup_logging': ('coinfosim.logging_config', 'setup_logging'),
    'setup_logging_from_config': ('coinfosim.logging_config', 'setup_logging_from_config'),
    'get_logger': ('coinfosim.logging_config', 'get_logger'),
    'reset_logging': ('coinfosim.logging_config', 'reset_logging'),
    'is_logging_configured': ('coinfosim.logging_config', 'is_logging_configured'),
    'init_report_service_conf': ('coinfosim.utils', 'init_report_service_conf'),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_ATTRIBUTES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRIBUTES))


try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("coinfosim")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"
del PackageNotFoundError, _pkg_version
