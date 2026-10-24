"""Command line entry point for SimLab."""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from simlab import __version__
from simlab.core.config import load_config
from simlab.core.runner import Runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="simlab-cli")
    parser.add_argument("--version", action="version", version=__version__)
    subs = parser.add_subparsers(dest="command")
    run = subs.add_parser("run")
    run.add_argument("--steps", type=int, default=100)
    run.add_argument("--seed", type=int, default=None)
    bench = subs.add_parser("bench")
    bench.add_argument("--steps", type=int, default=1000)
    bench.add_argument("--agents", type=int, default=1)
    subs.add_parser("gui")
    evalp = subs.add_parser("eval")
    evalp.add_argument("--policy", required=True)
    evalp.add_argument("--drive", default="simple")
    evalp.add_argument("--scenario", choices=["balanced", "solo", "hard"], required=True)
    evalp.add_argument("--agents", type=int, required=True)
    evalp.add_argument("--seeds", type=int, default=20)
    evalp.add_argument("--cap", type=int, default=10000)
    evalp.add_argument("--workers", type=int, default=4)
    evalp.add_argument("--out", default=None)
    evalp.add_argument("--compare", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_simulation(args.steps, args.seed)
    if args.command == "bench":
        return run_benchmark(args.steps, args.agents)
    if args.command == "eval":
        return run_eval(args)
    if args.command == "gui":
        from simlab.gui.app import main as gui_main

        return gui_main()
    parser.print_help()
    return 1


def _eval_worker(task: tuple[str, str, int, int, int, str]) -> dict[str, Any]:
    policy, drive, seed, agents, cap, config_path = task
    from simlab.eval_lib import run_episode_metrics

    return run_episode_metrics(policy, seed, agents, cap, config_path, drive=drive)


def run_eval(args: argparse.Namespace) -> int:
    from simlab.eval_lib import (
        REPO,
        aggregate,
        compare_to_baselines,
        scenario_baseline_key,
    )

    config_path = str(REPO / {
        "balanced": "configs/balanced.toml",
        "solo": "configs/solo.toml",
        "hard": "configs/hard.toml",
    }[args.scenario])
    tasks = [
        (args.policy, args.drive, seed, args.agents, args.cap, config_path)
        for seed in range(args.seeds)
    ]
    start = time.monotonic()
    with Pool(processes=max(1, args.workers)) as pool:
        results = pool.map(_eval_worker, tasks)
    wall = time.monotonic() - start
    agg = aggregate(results, wall, args.cap)

    s = agg
    print(f"policy={args.policy} drive={args.drive} scenario={args.scenario} agents={args.agents}")
    print(f"seeds={args.seeds} cap={args.cap} workers={args.workers}")
    print(
        f"cap_pct={s['cap_pct']:.1f}% 95%CI=[{s['ci'][0]:.1f},{s['ci'][1]:.1f}] "
        f"median_survival={s['median_survival']}"
    )
    print(
        "death_cause starve={} hazard={} trunc={}".format(
            s["death_cause"]["starvation"],
            s["death_cause"]["hazard"],
            s["death_cause"]["truncation"],
        )
    )
    print(
        f"food_eaten_per_agent={s['food_eaten_per_agent']} "
        f"hazard_contacts_per_1000={s['hazard_contacts_per_1000']} "
        f"mean_energy={s['mean_energy']} food_left_mean={s['food_left_mean']}"
    )
    print(f"wall_time_s={s['wall_time_s']} hash={s['hash']}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {out}")

    if args.compare:
        baseline = REPO / "docs" / "baselines.json"
        key = scenario_baseline_key(args.scenario, args.agents)
        for entry in compare_to_baselines(agg, key, baseline):
            print(f"vs {entry['against']:<18}: {entry['verdict']}")
    return 0


def run_simulation(steps: int, seed: int | None = None) -> int:
    runner = Runner(load_config(), seed)
    runner.setup()
    initial = {aid: tuple(s["position"]) for aid, s in runner._world._agents.items()}
    for _ in range(steps):
        runner.run_step()
    print(f"steps={runner._step_count} seed={runner._seed}")
    for aid, state in runner._world._agents.items():
        distance = (
            (state["position"][0] - initial[aid][0]) ** 2
            + (state["position"][1] - initial[aid][1]) ** 2
        ) ** 0.5
        print(
            f"{aid} position=({state['position'][0]:.3f},{state['position'][1]:.3f}) distance_traveled={distance:.3f} food_eaten={state['food_eaten']} energy={state['energy']:.3f}"
        )
    print(f"state_hash={runner._world.state_hash()}")
    return 0


def run_benchmark(steps: int, agents: int = 1) -> int:
    runner = Runner(load_config(), 42, agents=agents)
    runner.setup()
    runner.run_bench(steps)
    print(f"agents={agents} steps={steps}")
    print(f"world_only_steps_per_second={runner.last_bench_world_steps_per_second:.1f}")
    print(f"full_pipeline_steps_per_second={runner.last_bench_steps_per_second:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
