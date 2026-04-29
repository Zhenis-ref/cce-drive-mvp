from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from core.dn_metrics import ActionPreview, action_cluster
from core.scoring import RolloutMetrics, ScoredAction, score_rollout
from env.vehicle_model import PhysicalContext, VehicleParams, VehicleState, control_margin, propagate
from env.world_model import World, collided, collision_distance, lane_margin


@dataclass
class EvaluatedAction:
    action: np.ndarray
    metrics: RolloutMetrics
    score: ScoredAction


@dataclass
class PlannerResult:
    name: str
    best_action: np.ndarray
    best_evaluation: EvaluatedAction
    evaluated_count: int
    planning_time_ms: float
    delta_n: float
    delta_d: float
    risk_level: str
    severity: float = 0.0
    selected_k: int = 0
    selected_total: int = 0


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def generate_action_lattice(config: Dict[str, object]) -> List[np.ndarray]:
    accels = config['action_lattice']['accel_values']
    steers = config['action_lattice']['steer_values']
    return [np.array([a, s], dtype=float) for a in accels for s in steers]


def rollout_action(
    world: World,
    action: np.ndarray,
    dt: float,
    horizon_steps: int,
    weights: Dict[str, float],
    ctx: PhysicalContext,
    params: VehicleParams,
) -> EvaluatedAction:
    sim_world = world.clone()
    ego = VehicleState(sim_world.ego_start.x, sim_world.ego_start.y, sim_world.ego_start.v, sim_world.ego_start.theta)

    min_collision = 999.0
    min_lane = 999.0
    min_control = control_margin(action, params, ctx)
    collided_flag = False
    lane_departure = False

    for step in range(horizon_steps):
        t = step * dt
        ego = propagate(ego, action, dt, params, ctx)
        for obj in sim_world.objects:
            obj.step(dt, t)

        min_collision = min(min_collision, collision_distance(ego, sim_world))
        lm = lane_margin(ego, sim_world.lane_width)
        min_lane = min(min_lane, lm)
        if lm <= 0.0:
            lane_departure = True
        if collided(ego, sim_world):
            collided_flag = True
            break

    progress = ego.x - sim_world.ego_start.x
    comfort_cost = abs(float(action[0])) * 0.1 + abs(float(action[1])) * 2.0
    feasible = (not collided_flag) and min_control > 0.0

    metrics = RolloutMetrics(
        collided=collided_flag,
        lane_departure=lane_departure,
        min_collision_distance=float(min_collision),
        min_lane_margin=float(min_lane),
        min_control_margin=float(min_control),
        progress=float(progress),
        comfort_cost=float(comfort_cost),
        feasible=bool(feasible),
    )
    score = score_rollout(metrics, weights)
    return EvaluatedAction(action=action, metrics=metrics, score=score)


def gross_preview(
    world: World,
    action: np.ndarray,
    gross_horizon_seconds: float,
    dt: float,
    ctx: PhysicalContext,
    params: VehicleParams,
) -> ActionPreview:
    sim_world = world.clone()
    ego = VehicleState(sim_world.ego_start.x, sim_world.ego_start.y, sim_world.ego_start.v, sim_world.ego_start.theta)
    steps = max(1, int(round(gross_horizon_seconds / dt)))

    accel, steer = float(action[0]), float(action[1])

    min_collision = 999.0
    first_collision = None
    last_collision = None
    min_lane = 999.0

    for step in range(steps):
        t = step * dt
        ego = propagate(ego, action, dt, params, ctx)
        for obj in sim_world.objects:
            obj.step(dt, t)

        cdist = collision_distance(ego, sim_world)
        if first_collision is None:
            first_collision = cdist
        last_collision = cdist
        min_collision = min(min_collision, cdist)

        lm = lane_margin(ego, sim_world.lane_width)
        min_lane = min(min_lane, lm)

    if first_collision is None:
        first_collision = min_collision
    if last_collision is None:
        last_collision = min_collision

    control_m = control_margin(action, params, ctx)
    feasible = min_lane > 0.0 and control_m > 0.0

    # Основные прокси риска
    ttc_like = 1.0 / max(min_collision + 0.5, 1.0)
    ttc_like = _clip01(ttc_like)

    corridor_narrowing = 1.0 - min(1.0, max(0.0, min_lane / (sim_world.lane_width / 2.0)))
    corridor_narrowing = _clip01(corridor_narrowing)

    closing_gain = max(0.0, float(first_collision - last_collision))
    closing_term = _clip01(closing_gain / 6.0)

    risk = min(1.0, 0.60 * ttc_like + 0.20 * corridor_narrowing + 0.20 * closing_term)

    # Контекст экстренности
    emergency_context = (min_collision < 7.0) or (ttc_like > 0.17) or (closing_term > 0.20)

    brake_bonus = 0.0
    accel_penalty = 0.0
    steer_penalty = 0.0

    if emergency_context:
        # В опасной ситуации прямолинейное торможение должно подниматься выше.
        if accel <= -3.5 and abs(steer) <= 0.09:
            brake_bonus += 0.65
        elif accel <= -2.0 and abs(steer) <= 0.12:
            brake_bonus += 0.40
        elif accel <= -1.0:
            brake_bonus += 0.20

        # Газ или слабое торможение в опасной ситуации наказываем сильнее.
        if accel > -0.5:
            accel_penalty += 0.55
        elif accel > -1.5:
            accel_penalty += 0.25

        # Сильный руль в hard_brake часто вреден.
        if abs(steer) > 0.12:
            steer_penalty += 0.20
        elif abs(steer) > 0.08:
            steer_penalty += 0.10
    else:
        # В спокойной сцене слишком агрессивные манёвры не любим.
        steer_penalty += 0.04 * abs(steer) / 0.14
        accel_penalty += 0.03 * max(0.0, accel) / 3.0

    control_term = _clip01((control_m + 1.0) / 2.0)

    score_hint = (
        2.8 * (1.0 - risk)
        + 1.0 * (1.0 if feasible else 0.0)
        + 0.35 * control_term
        + brake_bonus
        - accel_penalty
        - steer_penalty
        - 0.02 * abs(accel)
    )

    return ActionPreview(
        action=action,
        cluster=action_cluster(action),
        gross_risk=float(risk),
        ttc_like=float(ttc_like),
        corridor_narrowing=float(corridor_narrowing),
        feasible=bool(feasible),
        future_position=np.array([ego.x, ego.y], dtype=float),
        score_hint=float(score_hint),
    )