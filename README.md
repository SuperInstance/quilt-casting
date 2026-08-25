# quilt-casting

The casting-call plugin: Wilson + LinUCB model router for the Quilt.

The casting-call picks the right AI model for a given (opener, role,
primitive) tuple. It uses:

- **Wilson lower bound** for n < 10: optimistic prior
- **LinUCB** for n >= 10: per-(user, app) contextual bandit
- **Gale-aware resources**: battery, network, compute
- **4 failure modes**: budget, network, hardware, time

This is one of the small, focused repos that compose the Quilt.
It was extracted from `quilt-substrate` as part of the v4.0 split.

## What this provides

- `QuiltCastingCallPlugin` — the main plugin
- `WilsonProfiles` — per-(primitive, opener, model) success tracking
- `Probes`, `Situation`, `ResourceBudget` — the resource context
- `CastingDecision` — the result of a cast
- `PRIOR_ATLAS` — the 16+ AI models with their priors
- `ROLE_TO_OPENER` — the role-to-opener mapping
- `wilson_lower` — the Wilson lower bound function
- `LinUCBCaster`, `LinUCBCastingPlugin` — the LinUCB layer
- `LinUCBModel`, `FeatureExtractor` — the bandit internals
- `alpha_for` — the exploration budget function

## 4-axis characterization

The plugin characterizes each model on 4 axes:
1. **Voice** — what it sounds/feels/looks like
2. **Tempo** — BPM, how fast it runs
3. **Cost** — what it consumes (per-token USD)
4. **Failure mode** — how it breaks (budget, network, hardware, time)

## 16+ models in the atlas

GLM-5.3, GLM-5, GLM-4.6, PHI-4, HERMES_405B, QWEN_0_5B, QWEN_4B,
CLAUDE_OPUS, DEEPSEEK_V4_FLASH, DEEPSEEK_V3, SEED_MINI, NEMOTRON_ULTRA,
and more. The plugin picks the best one for each (opener, role,
primitive) given the current (weather, time, crew_state, battery,
network, compute) context.

## Gale-aware resource budget

- **Calm + full battery**: alpha = 1.0 (max exploration)
- **Gale + low battery + tired**: alpha = 0.3 (conservative)
- The plugin tracks battery, network, compute as consumables
- The plugin knows when to be patient vs. when to be greedy

## Why split out as a separate repo?

The casting-call is a self-contained concern. It does not depend
on the substrate (it takes a Substrate as input). It does not
depend on the cowboy. It does not depend on the bus.

The casting-call can be reused by any substrate-like system that
wants learned model selection. The Wilson + LinUCB blend is the
state-of-the-art for warm-up + contextual bandits.

## Test count

48 tests (28 casting + 20 linucb), all passing.

## Version

1.0.0 — extracted from quilt-substrate v4.0-cowboy-loop.
