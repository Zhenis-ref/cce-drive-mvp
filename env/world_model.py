from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np

from env.vehicle_model import VehicleState


@dataclass
class DynamicObject:
    x: float
    y: float
    v: float
    lane_behavior: str = 'straight'
    brake_accel: float = 0.0
    trigger_time: float = 0.0

    def step(self, dt: float, t: float) -> None:
        accel = self.brake_accel if t >= self.trigger_time else 0.0
        self.v = max(0.0, self.v + accel * dt)
        self.x += self.v * dt
        if self.lane_behavior == 'cutin' and t >= self.trigger_time:
            self.y *= max(0.0, 1.0 - 1.8 * dt)


@dataclass
class World:
    lane_width: float
    road_length: float
    ego_radius: float
    object_radius: float
    ego_start: VehicleState
    objects: List[DynamicObject]
    obstacle: Tuple[float, float] | None = None

    def clone(self) -> 'World':
        objects = [DynamicObject(o.x, o.y, o.v, o.lane_behavior, o.brake_accel, o.trigger_time) for o in self.objects]
        return World(
            lane_width=self.lane_width,
            road_length=self.road_length,
            ego_radius=self.ego_radius,
            object_radius=self.object_radius,
            ego_start=VehicleState(self.ego_start.x, self.ego_start.y, self.ego_start.v, self.ego_start.theta),
            objects=objects,
            obstacle=None if self.obstacle is None else (self.obstacle[0], self.obstacle[1]),
        )


def lane_margin(state: VehicleState, lane_width: float) -> float:
    half = lane_width / 2.0
    return max(0.0, half - abs(state.y))


def collision_distance(ego: VehicleState, world: World) -> float:
    distances = []
    for obj in world.objects:
        d = math.hypot(ego.x - obj.x, ego.y - obj.y) - (world.ego_radius + world.object_radius)
        distances.append(d)
    if world.obstacle is not None:
        ox, oy = world.obstacle
        d = math.hypot(ego.x - ox, ego.y - oy) - (world.ego_radius + world.object_radius)
        distances.append(d)
    if not distances:
        return 999.0
    return min(distances)


def collided(ego: VehicleState, world: World) -> bool:
    return collision_distance(ego, world) <= 0.0
