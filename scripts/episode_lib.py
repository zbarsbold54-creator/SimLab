"""Shared episode runner with principled death-cause classification.

A death is classified by what actually drove health to zero:

- ``hazard_loss`` accumulates health removed by hazard contact (recomputed
  each step from the agent's position against each hazard's contact radius).
- ``starve_loss`` accumulates the 0.5 health decrement applied each step an
  agent's energy sits at 0 (the starvation drain).
- ``truncation`` is a survivor at the step cap.

The cause is hazard when hazard damage was the dominant health loss, otherwise
starvation (energy reached zero and its sustained drain drove health down).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simlab.core.config import load_config
from simlab.core.runner import Runner


def run_episode(
    policy: str,
    seed: int,
    agents: int = 5,
    cap: int = 10000,
    config_path: str = "configs/default.toml",
    radius: float | None = None,
    regrow_rate: float | None = None,
) -> tuple[dict[str, dict[str, Any]], float]:
    base = load_config(config_path)
    cfg = replace(base, policy=policy, agent=replace(base.agent, count=agents))
    if radius is not None:
        cfg = replace(cfg, hazard=replace(cfg.hazard, detection_radius=float(radius)))
    if regrow_rate is not None:
        cfg = replace(cfg, food=replace(cfg.food, regrow_rate=regrow_rate))
    runner = Runner(cfg, seed=seed, agents=agents)
    runner.setup()
    world = runner._world
    dt = world.dt
    track: dict[str, dict[str, Any]] = {}
    for aid in world._agents:
        track[aid] = {
            "hazard_loss": 0.0,
            "starve_loss": 0.0,
            "contact_steps": [],
            "zero_energy_step": None,
            "death_step": None,
            "energy_at_death": None,
            "health_at_death": None,
            "food_eaten": 0,
        }
    agents_alive = set(world._agents)
    for step in range(1, cap + 1):
        result = runner.run_step(
            include_observations=False, include_agent_states=False
        )
        for aid in list(agents_alive):
            state = world._agents[aid]
            t = track[aid]
            pos = state["position"]
            hazard_damage = 0.0
            for hazard in world._hazard_patches:
                dx = pos[0] - hazard["position"][0]
                dy = pos[1] - hazard["position"][1]
                if dx * dx + dy * dy <= hazard["radius"] ** 2:
                    hazard_damage += hazard["damage"] * dt
            if hazard_damage > 0.0:
                t["hazard_loss"] += hazard_damage
                t["contact_steps"].append(step)
            if state["energy"] <= 0.0:
                t["starve_loss"] += 0.5
                if t["zero_energy_step"] is None:
                    t["zero_energy_step"] = step
            if not state["is_alive"]:
                t["death_step"] = step
                t["energy_at_death"] = state["energy"]
                t["health_at_death"] = state["health"]
                t["food_eaten"] = state["food_eaten"]
                agents_alive.discard(aid)
        if result.terminated:
            break
    food_left = sum(patch["amount"] for patch in world._food_patches)
    for aid, t in track.items():
        if t["death_step"] is None:
            t["death_step"] = cap
            t["class"] = "truncation"
        else:
            if t["hazard_loss"] > t["starve_loss"]:
                t["class"] = "hazard"
            else:
                t["class"] = "starvation"
            t["contacts_final_50"] = sum(
                1 for s in t["contact_steps"] if s >= t["death_step"] - 50
            )
    return track, food_left


def split(outcomes: list[str]) -> tuple[int, int, int]:
    counts = {"starvation": 0, "hazard": 0, "truncation": 0}
    for cause in outcomes:
        counts[cause] += 1
    return counts["starvation"], counts["hazard"], counts["truncation"]
