from __future__ import annotations

import time
from typing import Dict, List
import numpy as np

from planners.shared_rollout import PlannerResult, generate_action_lattice, rollout_action
from env.vehicle_model import PhysicalContext, VehicleParams
from env.world_model import World


class BruteForcePlanner:
    def __init__(self, config: Dict[str, object]) -> None:
        self.config = config
        self.params = VehicleParams()

    def plan(self, world: World, ctx: PhysicalContext) -> PlannerResult:
        start = time.perf_counter()
        dt = float(self.config['dt'])
        horizon_steps = int(self.config['horizon_steps'])
        weights = self.config['weights']

        actions = generate_action_lattice(self.config)
        evaluations = [
            rollout_action(world, action, dt, horizon_steps, weights, ctx, self.params)
            for action in actions
        ]
        best = min(evaluations, key=lambda item: item.score.total_cost)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return PlannerResult(
            name='brute_force',
            best_action=best.action,
            best_evaluation=best,
            evaluated_count=len(evaluations),
            planning_time_ms=float(elapsed_ms),
            delta_n=0.0,
            delta_d=0.0,
            risk_level='baseline',
        )
