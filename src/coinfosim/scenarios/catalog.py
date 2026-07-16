"""Catalog of built-in dataset-anchored scenarios.

Listing scenarios must stay cheap: importing a scenario's
``DatasetAnchoredExecutionSpec`` pulls in its dataset loader and report
modules, which in turn import matplotlib and scikit-learn (empirically
~2 seconds and 2000+ modules for a single scenario). ``coinfosim --help``
and ``coinfosim scenario list`` must not pay that cost, so ``spec`` is
resolved lazily: the underlying definitions module is only imported the
first time an attribute of ``spec`` is actually accessed (i.e. when a
scenario is actually run, regenerated, or shown in detail).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only imported for static type checking: at runtime this module pulls in
    # matplotlib and scikit-learn (~1.5s), which would defeat the point of
    # resolving `spec` lazily. Never import this at module level.
    from coinfosim.scenarios.dataset_anchored_runner import DatasetAnchoredExecutionSpec

_ALIASES: dict[str, str] = {
    "occupancy": "occupancy",
    "occupancy-detection": "occupancy",
    "occupancy_detection": "occupancy",
    "air-quality": "air-quality",
    "air_quality": "air-quality",
    "airquality": "air-quality",
    "support2": "support2",
    "support-2": "support2",
    "support_2": "support2",
}


class UnknownScenarioError(KeyError):
    """Raised when a scenario slug or alias does not resolve to a definition."""

    def __init__(self, slug_or_alias: str) -> None:
        self.slug_or_alias = slug_or_alias
        super().__init__(
            f"unknown scenario {slug_or_alias!r}; known scenarios: "
            f"{', '.join(sorted(set(_ALIASES.values())))}"
        )


_UNRESOLVED = object()


class _LazyAttribute:
    """Attribute-delegating proxy that imports the real value on first use.

    Dunder-based protocols (``dict(x)``, ``x[k]``, iteration, ...) are looked
    up on the type and bypass instance-level ``__getattr__``, so code that
    needs the real underlying object for those protocols (e.g. before passing
    a classifier configuration mapping into code that calls ``dict(...)`` on
    it) must call :meth:`resolve` explicitly instead of relying on proxying.
    """

    __slots__ = ("_module_path", "_attr_name", "_resolved")

    def __init__(self, module_path: str, attr_name: str) -> None:
        object.__setattr__(self, "_module_path", module_path)
        object.__setattr__(self, "_attr_name", attr_name)
        object.__setattr__(self, "_resolved", _UNRESOLVED)

    def resolve(self) -> Any:
        resolved = object.__getattribute__(self, "_resolved")
        if resolved is _UNRESOLVED:
            module = import_module(object.__getattribute__(self, "_module_path"))
            resolved = getattr(module, object.__getattribute__(self, "_attr_name"))
            object.__setattr__(self, "_resolved", resolved)
        return resolved

    def __getattr__(self, name: str) -> Any:
        return getattr(self.resolve(), name)

    def __repr__(self) -> str:
        return repr(self.resolve())


@dataclass(frozen=True)
class ScenarioDefinition:
    """Catalog entry for one built-in dataset-anchored scenario."""

    slug: str
    aliases: tuple[str, ...]
    display_name: str
    description: str
    dataset_slug: str
    spec: DatasetAnchoredExecutionSpec
    supports_dataset_report_only: bool
    # Mapping[str, Any] | None at runtime; may be a lazily-imported proxy that
    # only pays its (sklearn-importing) cost when actually accessed.
    default_classifier_configuration: Any = None


_REGISTRY: dict[str, ScenarioDefinition] = {
    "occupancy": ScenarioDefinition(
        slug="occupancy",
        aliases=("occupancy-detection", "occupancy_detection"),
        display_name="Occupancy Detection",
        description=(
            "Real vs. single-Gaussian vs. GMM synthetic training data evaluated "
            "against the fixed real UCI Occupancy Detection test set."
        ),
        dataset_slug="occupancy",
        spec=_LazyAttribute(
            "coinfosim.scenarios.definitions.occupancy", "OCCUPANCY_SPEC"
        ),
        supports_dataset_report_only=True,
    ),
    "air-quality": ScenarioDefinition(
        slug="air-quality",
        aliases=("air_quality", "airquality"),
        display_name="UCI Air Quality",
        description=(
            "Real vs. single-Gaussian vs. GMM synthetic training data evaluated "
            "against the fixed chronological future UCI Air Quality test set."
        ),
        dataset_slug="air-quality",
        spec=_LazyAttribute(
            "coinfosim.scenarios.definitions.air_quality", "AIR_QUALITY_SPEC"
        ),
        supports_dataset_report_only=False,
    ),
    "support2": ScenarioDefinition(
        slug="support2",
        aliases=("support-2", "support_2"),
        display_name="SUPPORT2 180-Day Mortality",
        description=(
            "Real vs. single-Gaussian vs. GMM synthetic training data evaluated "
            "against the fixed real SUPPORT2 180-day mortality test set."
        ),
        dataset_slug="support2",
        spec=_LazyAttribute(
            "coinfosim.scenarios.definitions.support2", "SUPPORT2_SPEC"
        ),
        supports_dataset_report_only=False,
        default_classifier_configuration=_LazyAttribute(
            "coinfosim.scenarios.definitions.support2",
            "SUPPORT2_CLASSIFIER_CONFIGURATION",
        ),
    ),
}


def list_scenarios() -> tuple[ScenarioDefinition, ...]:
    """Return all built-in scenario definitions, in catalog order."""

    return tuple(_REGISTRY.values())


def _normalize_slug(slug_or_alias: str) -> str | None:
    candidate = slug_or_alias.strip().lower()
    if candidate in _ALIASES:
        return _ALIASES[candidate]
    dashed = candidate.replace("_", "-")
    if dashed in _ALIASES:
        return _ALIASES[dashed]
    underscored = candidate.replace("-", "_")
    if underscored in _ALIASES:
        return _ALIASES[underscored]
    return None


def get_scenario(slug_or_alias: str) -> ScenarioDefinition:
    """Resolve ``slug_or_alias`` (including known aliases) to a definition.

    Raises
    ------
    UnknownScenarioError
        If ``slug_or_alias`` does not resolve to any built-in scenario.
    """

    canonical = _normalize_slug(slug_or_alias)
    if canonical is None or canonical not in _REGISTRY:
        raise UnknownScenarioError(slug_or_alias)
    return _REGISTRY[canonical]
