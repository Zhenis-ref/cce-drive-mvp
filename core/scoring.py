from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import math


@dataclass
class RolloutMetrics:
    collided: bool
    lane_departure: bool
    min_collision_distance: float
    min_lane_margin: float
    min_control_margin: float
    progress: float
    comfort_cost: float
    feasible: bool


@dataclass
class ScoredAction:
    total_cost: float
    collision_risk: float
    stability_margin: float
    feasible: bool


def clamp_nonnegative(value: float) -> float:
    return max(0.0, float(value))


def compute_stability_margin(metrics: RolloutMetrics) -> float:
    return min(
        clamp_nonnegative(metrics.min_collision_distance),
        clamp_nonnegative(metrics.min_lane_margin),
        clamp_nonnegative(metrics.min_control_margin),
    )


def score_rollout(metrics: RolloutMetrics, weights: Dict[str, float]) -> ScoredAction:
    margin = compute_stability_margin(metrics)
    collision_risk = 1.0 if metrics.collided else 0.0
    lane_penalty = 1.0 if metrics.lane_departure else 0.0
    feasibility_penalty = 0.0 if metrics.feasible else 1.0
    progress_penalty = max(0.0, 15.0 - metrics.progress) / 15.0

    total_cost = (
        weights['collision'] * collision_risk
        + weights['margin_inverse'] * (1.0 / (margin + 1e-3))
        + weights['lane'] * lane_penalty
        + weights['feasibility'] * feasibility_penalty
        + weights['progress'] * progress_penalty
        + weights['comfort'] * metrics.comfort_cost
    )
    return ScoredAction(
        total_cost=float(total_cost),
        collision_risk=float(collision_risk),
        stability_margin=float(margin),
        feasible=metrics.feasible,
    )
