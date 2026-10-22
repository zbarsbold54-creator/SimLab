"""Minimal task protocol implementation."""

from simlab.core.registry import register


@register("task", "survive")
class SurviveTask:
    def reward(self, result: object) -> float:
        return 0.0 if getattr(result, "terminated", False) else 1.0
