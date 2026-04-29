from __future__ import annotations

import time
from typing import Dict

from core.cce_gate import select_candidates
from core.dn_metrics import evaluate_dn_state
from planners.shared_rollout import PlannerResult, generate_action_lattice, gross_preview, rollout_action
from env.vehicle_model import PhysicalContext, VehicleParams
from env.world_model import World


class CCEPlanner:
    def __init__(self, config: Dict[str, object]) -> None:
        self.config = config
        self.params = VehicleParams()

    def plan(self, world: World, ctx: PhysicalContext) -> PlannerResult:
        start = time.perf_counter()
        dt = float(self.config['dt'])
        horizon_steps = int(self.config['horizon_steps'])
        weights = self.config['weights']
        planner_cfg = self.config['planner']

        actions = generate_action_lattice(self.config)
        previews = [
            gross_preview(world, action, float(planner_cfg['gross_horizon_seconds']), dt, ctx, self.params)
            for action in actions
        ]

        dn_state = evaluate_dn_state(previews)
        selected_actions, selected_k = select_candidates(previews, dn_state, planner_cfg)

        evaluations = [
            rollout_action(world, action, dt, horizon_steps, weights, ctx, self.params)
            for action in selected_actions
        ]
        best = min(evaluations, key=lambda item: item.score.total_cost)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return PlannerResult(
            name='cce',
            best_action=best.action,
            best_evaluation=best,
            evaluated_count=len(evaluations),
            planning_time_ms=float(elapsed_ms),
            delta_n=float(dn_state.delta_n),
            delta_d=float(dn_state.delta_d),
            risk_level=dn_state.risk_level,
            severity=float(dn_state.severity),
            selected_k=int(selected_k),
            selected_total=len(selected_actions),
        )