from __future__ import annotations

from typing import Dict
import random

from env.vehicle_model import PhysicalContext, VehicleState
from env.world_model import DynamicObject, World


def make_scenario(name: str, seed: int, lane_width: float = 3.7, road_length: float = 200.0) -> Dict[str, object]:
    rng = random.Random(seed)

    ego = None
    obj = None
    obstacle = None
    ctx = PhysicalContext(friction=1.0, steering_lag=0.05, brake_scale=1.0)

    if name == 'hard_brake':
        # Делаем часть прогонов реально жёсткими, чтобы severity гулял.
        aggressive = rng.random() < 0.45

        ego_speed = 24.0 + rng.uniform(-1.0, 3.5)
        if aggressive:
            gap = rng.uniform(16.0, 24.0)
            obj_speed = rng.uniform(8.0, 16.0)
            brake_accel = rng.uniform(-10.0, -7.0)
            trigger = rng.uniform(0.15, 0.45)
        else:
            gap = rng.uniform(24.0, 36.0)
            obj_speed = rng.uniform(15.0, 22.0)
            brake_accel = rng.uniform(-7.5, -5.5)
            trigger = rng.uniform(0.35, 0.80)

        ego = VehicleState(x=0.0, y=0.0, v=ego_speed, theta=0.0)
        obj = DynamicObject(
            x=gap,
            y=rng.uniform(-0.18, 0.18),
            v=obj_speed,
            brake_accel=brake_accel,
            trigger_time=trigger,
        )

        # Небольшая вариация физики даже на обычной дороге.
        ctx = PhysicalContext(
            friction=rng.uniform(0.85, 1.00),
            steering_lag=rng.uniform(0.04, 0.08),
            brake_scale=rng.uniform(0.90, 1.05),
        )

    elif name == 'obstacle':
        ego_speed = 24.0 + rng.uniform(-1.5, 2.0)
        gap = rng.uniform(22.0, 34.0)

        ego = VehicleState(x=0.0, y=0.0, v=ego_speed, theta=0.0)
        obj = DynamicObject(
            x=gap + rng.uniform(5.0, 11.0),
            y=rng.uniform(-0.15, 0.15),
            v=rng.uniform(15.0, 20.0),
        )
        obstacle = (rng.uniform(19.0, 30.0), rng.uniform(-0.45, 0.45))

    elif name == 'ice_trap':
        ego_speed = 23.0 + rng.uniform(-1.0, 2.0)
        ego = VehicleState(x=0.0, y=0.0, v=ego_speed, theta=0.0)

        obj = DynamicObject(
            x=rng.uniform(22.0, 31.0),
            y=rng.uniform(-0.22, 0.22),
            v=rng.uniform(14.0, 20.0),
            brake_accel=rng.uniform(-5.5, -3.5),
            trigger_time=rng.uniform(0.20, 0.60),
        )
        obstacle = (rng.uniform(18.0, 27.0), rng.uniform(-0.50, 0.50))
        ctx = PhysicalContext(
            friction=rng.uniform(0.22, 0.42),
            steering_lag=rng.uniform(0.10, 0.20),
            brake_scale=rng.uniform(0.50, 0.72),
        )

    elif name == 'cutin':
        ego_speed = 24.0 + rng.uniform(-1.0, 2.0)
        ego = VehicleState(x=0.0, y=0.0, v=ego_speed, theta=0.0)

        obj = DynamicObject(
            x=rng.uniform(20.0, 30.0),
            y=lane_width + rng.uniform(-0.35, 0.35),
            v=rng.uniform(18.0, 25.0),
            lane_behavior='cutin',
            trigger_time=rng.uniform(0.25, 0.85),
        )

    else:
        raise ValueError(f'Unknown scenario: {name}')

    world = World(
        lane_width=lane_width,
        road_length=road_length,
        ego_radius=1.2,
        object_radius=1.2,
        ego_start=ego,
        objects=[obj] if obj is not None else [],
        obstacle=obstacle,
    )

    return {'world': world, 'physical_context': ctx}