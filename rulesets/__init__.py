# rulesets package — per-sport parameter profiles for the modules/ CV pipeline.
from rulesets.base import RulesetConfig
from rulesets.classic import CLASSIC
from rulesets.futsal import FUTSAL
from rulesets.registry import DEFAULT_RULESET, available_rulesets, get_ruleset

__all__ = [
    "RulesetConfig",
    "FUTSAL",
    "CLASSIC",
    "DEFAULT_RULESET",
    "available_rulesets",
    "get_ruleset",
]
