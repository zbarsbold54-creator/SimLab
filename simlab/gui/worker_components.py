"""Policy and drive execution for one simulation worker."""

from __future__ import annotations

from types import MappingProxyType

from simlab.core.config import SimLabConfig
from simlab.core.pipeline import compute_actions, needs_observations
from simlab.core.registry import get, load_plugins
from simlab.core.types import Action
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
                self.policy = policy_type(seed=self.seed, config=self.config)
            except TypeError:
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

    def needs_observations(self) -> bool:
        return needs_observations(self.policy, self.drive_core)

    def actions_for(self, world, step: int) -> tuple[dict[str, Action], object]:
        actions, drives, observations = compute_actions(
            world, self.policy, self.drive_core, step
        )
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
