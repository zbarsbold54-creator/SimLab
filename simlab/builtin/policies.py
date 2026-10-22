"""Registered policies."""

from simlab.core.registry import register
from simlab.interfaces.policy import HeuristicPolicy, RandomPolicy

register("policy", "heuristic")(HeuristicPolicy)
register("policy", "random")(RandomPolicy)
