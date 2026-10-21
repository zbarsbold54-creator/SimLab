"""SimLab CLI entry point."""

from __future__ import annotations

import argparse
import sys

from simlab.core.config import load_config
from simlab.core.runner import Runner


def main(argv: list[str] | None = None) -> int:
    """SimLab command-line interface.

    Supported commands:
      run     Run a simulation for a given number of steps with optional seed.
      gui     Launch the GUI (not yet implemented).
      bench   Run a benchmark for a fixed number of steps.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(prog="simlab-cli", description="SimLab simulation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run a simulation")
    run_parser.add_argument("--steps", type=int, default=100, help="Number of steps to run")
    run_parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # gui command
    subparsers.add_parser("gui", help="Launch GUI (not yet implemented)")

    # bench command
    bench_parser = subparsers.add_parser("bench", help="Run a benchmark")
    bench_parser.add_argument("--steps", type=int, required=True, help="Number of steps to benchmark")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "run":
        return run_simulation(args.steps, args.seed)
    elif args.command == "bench":
        return run_benchmark(args.steps)


def run_simulation(steps: int, seed: int | None = None) -> int:
    """Run a simulation for a given number of steps.

    Args:
        steps: Number of steps to run.
        seed: Random seed. If None, uses config seed.

    Returns:
        Exit code.
    """
    config = load_config()
    runner = Runner(config=config, seed=seed)
    runner.setup()

    for _ in range(steps):
        result = runner.run_step({})
        if result.terminated:
            break

    # Print summary
    print(f"SimLab Simulation Summary")
    print(f"========================")
    print(f"Steps run: {runner._step_count}")
    print(f"Terminated: {result.terminated}")
    print(f"Seed: {runner._seed}")
    print(f"Config seed: {config.world.seed}")

    # Print agent states if any (read from world)
    world_agents = runner._world._agents
    for agent_id, agent_data in world_agents.items():
        if agent_data.get("is_alive", True):
            print(f"  {agent_id}: position=({int(agent_data['position'][0])},{int(agent_data['position'][1])}), energy={agent_data['energy']}")

    return 0


def run_benchmark(steps: int) -> int:
    """Run a benchmark for a fixed number of steps.

    Args:
        steps: Number of steps to run.

    Returns:
        Exit code.
    """
    config = load_config()
    runner = Runner(config=config, seed=42)
    runner.setup()

    results = runner.run_bench(num_steps=steps)

    print(f"SimLab Benchmark Summary")
    print(f"========================")
    print(f"Steps run: {len(results)}")
    print(f"Final step count: {runner._step_count}")

    terminated_count = sum(1 for r in results if r.terminated)
    print(f"Episodes terminated: {terminated_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())