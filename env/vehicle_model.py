from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class VehicleState:
    x: float
    y: float
    v: float
    theta: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.v, self.theta], dtype=float)


@dataclass
class VehicleParams:
    wheelbase: float = 2.8
    max_speed: float = 45.0
    min_speed: float = 0.0
    max_abs_steer: float = 0.35
    max_accel: float = 5.0
    max_brake: float = -6.0


@dataclass
class PhysicalContext:
    friction: float
    steering_lag: float
    brake_scale: float


def propagate(state: VehicleState, action: np.ndarray, dt: float, params: VehicleParams, ctx: PhysicalContext) -> VehicleState:
    accel_cmd = float(action[0])
    steer_cmd = float(action[1])

    accel_cmd = max(params.max_brake, min(params.max_accel, accel_cmd))
    steer_cmd = max(-params.max_abs_steer, min(params.max_abs_steer, steer_cmd))

    # Friction and actuation degrade effective braking/steering.
    effective_accel = accel_cmd
    if effective_accel < 0.0:
        effective_accel *= ctx.brake_scale * max(0.25, ctx.friction)
    effective_steer = steer_cmd * max(0.35, 1.0 - ctx.steering_lag)

    # Simple bicycle model.
    v_next = max(params.min_speed, min(params.max_speed, state.v + effective_accel * dt))
    beta = math.atan(0.5 * math.tan(effective_steer))
    theta_next = state.theta + (v_next / params.wheelbase) * math.sin(beta) * dt
    x_next = state.x + v_next * math.cos(state.theta + beta) * dt
    y_next = state.y + v_next * math.sin(state.theta + beta) * dt

    return VehicleState(x=x_next, y=y_next, v=v_next, theta=theta_next)


def control_margin(action: np.ndarray, params: VehicleParams, ctx: PhysicalContext) -> float:
    accel, steer = float(action[0]), float(action[1])
    steer_room = max(0.0, params.max_abs_steer - abs(steer)) / max(params.max_abs_steer, 1e-6)
    if accel < 0.0:
        brake_limit = abs(params.max_brake) * max(0.25, ctx.friction) * ctx.brake_scale
        brake_room = max(0.0, brake_limit - abs(accel)) / max(brake_limit, 1e-6)
    else:
        brake_room = 1.0
    return min(steer_room, brake_room) * 10.0
