"""Regenerate BASELINES.md rows with 95% bootstrap CIs over seeds.

Grid: scenarios (Balanced 1 / 5 / 20, Solo) x policies
(random / heuristic_noflee / heuristic / heuristic_v2), 20 seeds,
cap 10000. Multiprocessing, one process per episode.
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from episode_lib import run_episode

REPO = Path(__file__).resolve().parents[1]

SEEDS = 20
CAP = 10000
BOOTSTRAP = 1000
RNG_SEED = 1234

SCENARIOS = [
    ("Balanced 1", str(REPO / "configs/balanced.toml"), 1),
    ("Balanced 5", str(REPO / "configs/balanced.toml"), 5),
    ("Balanced 20", str(REPO / "configs/balanced.toml"), 20),
    ("Solo", str(REPO / "configs/solo.toml"), 1),
]
POLICIES = ["random", "heuristic_noflee", "heuristic", "heuristic_v2"]


def worker(task: tuple[str, str, int, int]) -> dict[str, Any]:
    config_path, policy, agents, seed = task
    track, food_left = run_episode(
        policy, seed, agents=agents, cap=CAP, config_path=config_path
    )
    classes = [t["class"] for t in track.values()]
    deaths = [t["death_step"] for t in track.values()]
    return {"seed": seed, "classes": classes, "deaths": deaths, "food_left": food_left}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = 0
    capn = 0
    starve = 0
    hazard = 0
    trunc = 0
    deaths = []
    food_left = []
    for row in rows:
        food_left.append(row["food_left"])
        for cls, death in zip(row["classes"], row["deaths"]):
            n += 1
            deaths.append(death)
            if cls == "truncation":
                capn += 1
                trunc += 1
            elif cls == "hazard":
                hazard += 1
            else:
                starve += 1
    deaths.sort()
    median = deaths[len(deaths) // 2]

    # Bootstrap cap% over seeds (resample seeds with replacement).
    rng = np.random.default_rng(RNG_SEED)
    classes_by_seed = [row["classes"] for row in rows]
    n_agents = max(len(c) for c in classes_by_seed)
    caps = []
    for _ in range(BOOTSTRAP):
        sample = rng.integers(0, len(rows), size=len(rows))
        cnt = sum(len(classes_by_seed[i]) for i in sample)
        cap_cnt = sum(
            sum(1 for cls in classes_by_seed[i] if cls == "truncation")
            for i in sample
        )
        caps.append(100.0 * cap_cnt / cnt)
    caps.sort()
    lo = caps[int(0.025 * len(caps))]
    hi = caps[int(0.975 * len(caps))]

    return {
        "n": n,
        "n_agents": n_agents,
        "cap_pct": 100.0 * capn / n,
        "ci": (lo, hi),
        "median": median,
        "starve": starve,
        "hazard": hazard,
        "trunc": trunc,
        "food_left_mean": float(np.mean(food_left)),
    }


def main() -> None:
    tasks = []
    for scenario_label, config_path, agents in SCENARIOS:
        for policy in POLICIES:
            for seed in range(SEEDS):
                tasks.append((config_path, policy, agents, seed))

    with Pool() as pool:
        results = pool.map(worker, tasks)

    header = (
        f"{'scenario':<14}{'policy':<18}{'cap%':>7}{'95% CI':>14}"
        f"{'median':>8}{'starve':>8}{'hazard':>8}{'trunc':>8}{'food_left':>10}"
    )
    print(header)
    print("-" * len(header))

    idx = 0
    for scenario_label, config_path, agents in SCENARIOS:
        for policy in POLICIES:
            rows = results[idx: idx + SEEDS]
            idx += SEEDS
            s = summarize(rows)
            print(
                f"{scenario_label:<14}{policy:<18}"
                f"{s['cap_pct']:6.1f}%  "
                f"[{s['ci'][0]:5.1f},{s['ci'][1]:5.1f}]  "
                f"{s['median']:>8}  "
                f"{s['starve']:>8}{s['hazard']:>8}{s['trunc']:>8}  "
                f"{s['food_left_mean']:8.1f}"
            )


if __name__ == "__main__":
    main()
