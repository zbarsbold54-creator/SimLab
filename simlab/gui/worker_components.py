"""Policy and drive execution for one simulation worker."""

from __future__ import annotations

from types import MappingProxyType

from simlab.core.config import SimLabConfig
from simlab.core.registry import get, load_plugins
from simlab.core.types import Action, Observation
from simlab.gui.frame import AgentFrame


class WorkerComponents:
    def __init__(self, config: SimLabConfig, seed: int, agent_count: int, rng) -> None:
        self.config = config
        self.seed = seed
        self.agent_count = agent_count
        self.reset(rng)

    def reset(self, rng) -> None:
        load_plugins()
        policy_type = get("policy", self.config.policy)
        if policy_type is None:
            self.policy = None
        else:
            try:
                self.policy = policy_type(seed=self.seed)
            except TypeError:
                self.policy = policy_type()
        drive_type = get("drive", self.config.drive)
        self.drive_core = drive_type() if drive_type else None
        if self.drive_core is not None:
            self.drive_core.reset(self.agent_count, rng)
        self.drive_names = tuple(getattr(self.drive_core, "names", ()))

    def update_config(
        self, config: SimLabConfig, seed: int, agent_count: int, rng
    ) -> None:
        self.config = config
        self.seed = seed
        self.agent_count = agent_count
        self.reset(rng)

    def actions_for(self, world, step: int) -> tuple[dict[str, Action], object]:
        observations: dict[str, Observation] = {
            agent_id: world._observation(agent_id) for agent_id in world._agents
        }
        live_ids = [
            agent_id for agent_id, state in world._agents.items() if state["is_alive"]
        ]
        live_observations = [observations[agent_id] for agent_id in live_ids]
        values = (
            self.drive_core.step(live_observations, {"rng": world._rng})
            if self.drive_core is not None
            else []
        )
        drives = {
            agent_id: tuple(float(value) for value in values[index])
            for index, agent_id in enumerate(live_ids)
        }
        actions: dict[str, Action] = {}
        for agent_id in live_ids:
            drive = drives.get(agent_id, ())
            if hasattr(self.policy, "select_action"):
                state = world._agents[agent_id]
                action = self.policy.select_action(
                    state,
                    {"food_locations": [p["position"] for p in world._food_patches]},
                    [],
                    step,
                )
            elif self.policy is not None:
                action = self.policy.act(observations[agent_id], drive)
            else:
                action = Action()
            actions[agent_id] = action
        details = MappingProxyType(
            {
                agent_id: AgentFrame(
                    actions.get(agent_id),
                    observations[agent_id],
                    drives.get(agent_id, ()),
                )
                for agent_id in observations
            }
        )
        return actions, details
