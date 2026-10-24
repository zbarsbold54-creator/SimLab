# Extending SimLab

Drop a Python module into `simlab/plugins/`; the GUI imports plugin modules and their `@register` decorators automatically. Names become selectable in the Scenario dock where supported. Implement the interface from `simlab/interfaces/` and keep plugin imports lightweight.

## Add a drive

A drive core implements `reset(n_agents, rng)` and `step(obs_batch, info)`, returning an `(agents, drives)` NumPy array in `[0, 1]`. Decorate and save this module under `simlab/plugins/`:

```python
import numpy as np
from simlab.core.registry import register


@register("drive", "curiosity")
class CuriosityDrive:
    names = ("novelty",)

    def reset(self, n_agents, rng):
        self.rng = rng

    def step(self, obs_batch, info):
        return np.asarray(
            [[min(len(obs.rays) / 32, 1)] for obs in obs_batch], dtype=np.float32
        )
```

Select `curiosity` in the Scenario dock. Replace the example formula with one value per agent for every name.

## Add a policy

A policy implements `act(observation, drive_state)` and returns an `Action`:

```python
from simlab.core.registry import register
from simlab.core.types import Action


@register("policy", "coast")
class CoastPolicy:
    def act(self, obs, drive_state=None):
        return Action(turn=0.0, thrust=0.2, eat=False)
```

The name `coast` appears in the Scenario dock. Policies may instead implement `select_action(agent_state, world_state, available_actions, step)` for a world-level controller.

## Add a learned policy (template)

For a policy with recurrent internal state that must be stepped every step, implement `needs_observations = True`, `reset(n_agents, rng)`, `get_state()` and `set_state()`. `reset` resizes/zeroes hidden state; `get_state`/`set_state` let snapshots capture and restore it so scrub-back replay reproduces identical actions. See `simlab/plugins/example_learned_policy.py` for the full template (a numpy-only GRU-like cell with fixed random weights; it must not import torch). Key points:

- Declare `needs_observations = True` so fast paths never skip sensing/drives for your policy.
- Build input features only from `obs` (rays + proprioception) and the `drive_state` tuple passed into `act`.
- Keep hidden state in a NumPy array; `get_state()` returns a copy, `set_state()` restores it.
- `act` is called once per live agent per step without an agent id — index hidden state by a stable per-agent identity if your policy depends on which agent is acting.
- Register as usual with `@register("policy", "my_policy")`; the Scenario dock and the eval CLI then discover it.

## Add a retention rule

A retention rule implements `step(previous, drives, context)` and returns a drive array of the same shape:

```python
from simlab.core.registry import register


@register("retention", "fast_decay")
class FastDecay:
    def step(self, previous, drives, context):
        return 0.5 * previous + 0.5 * drives
```

This registers the rule for discovery by name. Retention selection is not currently wired into GUI drive construction; the configured retention option is groundwork for that integration.
