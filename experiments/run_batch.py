from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.scenarios import make_scenario
from planners.brute_force import BruteForcePlanner
from planners.cce_planner import CCEPlanner


def load_config(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', required=True, choices=['hard_brake', 'obstacle', 'ice_trap', 'cutin'])
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    bf = BruteForcePlanner(config)
    cce = CCEPlanner(config)

    out_dir = ROOT / 'outputs' / 'csv'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'results_{args.scenario}.csv'

    rows = []
    for seed in range(1, args.seeds + 1):
        scenario = make_scenario(args.scenario, seed)
        world = scenario['world']
        ctx = scenario['physical_context']

        bf_result = bf.plan(world, ctx)
        cce_result = cce.plan(world, ctx)

        bf_safe = int((not bf_result.best_evaluation.metrics.collided) and (not bf_result.best_evaluation.metrics.lane_departure))
        cce_safe = int((not cce_result.best_evaluation.metrics.collided) and (not cce_result.best_evaluation.metrics.lane_departure))
        cr = bf_result.evaluated_count / max(cce_result.evaluated_count, 1)
        matched_safety = int(cce_safe >= bf_safe)
        false_pruning_proxy = int(cce_result.best_evaluation.score.total_cost > bf_result.best_evaluation.score.total_cost * 1.15)

        rows.append({
            'seed': seed,
            'scenario': args.scenario,
            'bf_safe': bf_safe,
            'cce_safe': cce_safe,
            'matched_safety': matched_safety,
            'bf_margin': bf_result.best_evaluation.score.stability_margin,
            'cce_margin': cce_result.best_evaluation.score.stability_margin,
            'bf_cost': bf_result.best_evaluation.score.total_cost,
            'cce_cost': cce_result.best_evaluation.score.total_cost,
            'bf_eval_count': bf_result.evaluated_count,
            'cce_eval_count': cce_result.evaluated_count,
            'compression_ratio': cr,
            'bf_latency_ms': bf_result.planning_time_ms,
            'cce_latency_ms': cce_result.planning_time_ms,
            'delta_n': cce_result.delta_n,
            'delta_d': cce_result.delta_d,
            'severity': cce_result.severity,
            'risk_level': cce_result.risk_level,
            'selected_k': cce_result.selected_k,
            'selected_total': cce_result.selected_total,
            'false_pruning_proxy': false_pruning_proxy,
        })

    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_cr = sum(r['compression_ratio'] for r in rows) / len(rows)
    mean_bf_safe = sum(r['bf_safe'] for r in rows) / len(rows)
    mean_cce_safe = sum(r['cce_safe'] for r in rows) / len(rows)
    mean_matched = sum(r['matched_safety'] for r in rows) / len(rows)

    print(f'Saved: {out_csv}')
    print(f'Mean BF safety:  {mean_bf_safe:.3f}')
    print(f'Mean CCE safety: {mean_cce_safe:.3f}')
    print(f'Mean matched safety: {mean_matched:.3f}')
    print(f'Mean compression ratio: {mean_cr:.3f}x')
if __name__ == '__main__':
    main()