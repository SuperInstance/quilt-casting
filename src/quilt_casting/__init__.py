"""quilt_casting — The casting-call plugin: Wilson + LinUCB model router.

The casting-call picks the right AI model for a given (opener, role,
primitive) tuple. It uses:

- **Wilson lower bound** for n < 10: optimistic prior
- **LinUCB** for n >= 10: per-(user, app) contextual bandit
- **Gale-aware resources**: battery, network, compute
- **4 failure modes**: budget, network, hardware, time

This is one of the small, focused repos that compose the Quilt.
It was extracted from `quilt-substrate` as part of the v4.0 split.
"""
from .casting import (
    QuiltCastingCallPlugin, WilsonProfiles, Probes, Situation,
    ResourceBudget, CastingDecision, PRIOR_ATLAS, ROLE_TO_OPENER,
    wilson_lower,
)
from .linucb import (
    LinUCBCaster, LinUCBCastingPlugin, LinUCBModel, FeatureExtractor,
    alpha_for, TIME_OF_DAY, WEATHER, NETWORK, CREW_STATE,
)

__version__ = "1.0.0"
__all__ = [
    "QuiltCastingCallPlugin", "WilsonProfiles", "Probes", "Situation",
    "ResourceBudget", "CastingDecision", "PRIOR_ATLAS", "ROLE_TO_OPENER",
    "wilson_lower",
    "LinUCBCaster", "LinUCBCastingPlugin", "LinUCBModel", "FeatureExtractor",
    "alpha_for", "TIME_OF_DAY", "WEATHER", "NETWORK", "CREW_STATE",
]
