"""Command line entry point for SimLab."""

from __future__ import annotations

import argparse
import sys

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
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_simulation(args.steps, args.seed)
    if args.command == "bench":
        return run_benchmark(args.steps, args.agents)
    if args.command == "gui":
        from simlab.gui.app import main as gui_main

        return gui_main()
    parser.print_help()
    return 1


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
    print(
        f"agents={agents} steps={steps} "
        f"steps_per_second={runner.last_bench_steps_per_second:.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
