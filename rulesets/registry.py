"""Single validated lookup point for RulesetConfig instances.

Used by both video_analysis/run.py (local dev script) and
service/session.py (Azure worker) so there's exactly one place that knows
the set of valid ruleset names.
"""

from __future__ import annotations

from rulesets.base import RulesetConfig
from rulesets.classic import CLASSIC
from rulesets.futsal import FUTSAL

_RULESETS: dict[str, RulesetConfig] = {
    FUTSAL.name: FUTSAL,
    CLASSIC.name: CLASSIC,
}

DEFAULT_RULESET = FUTSAL.name


def get_ruleset(name: str) -> RulesetConfig:
    """Look up a RulesetConfig by name. Raises ValueError on an unknown name."""
    try:
        return _RULESETS[name]
    except KeyError:
        valid = ", ".join(sorted(_RULESETS))
        raise ValueError(f"Unknown ruleset {name!r} — valid rulesets: {valid}") from None


def available_rulesets() -> list[str]:
    return sorted(_RULESETS)
