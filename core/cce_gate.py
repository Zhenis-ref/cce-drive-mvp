from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from core.dn_metrics import ActionPreview, DNState


ANCHORS = [
    np.array([-6.0, 0.0]),
    np.array([-4.5, 0.0]),
    np.array([-3.0, 0.0]),
    np.array([-3.0, -0.14]),
    np.array([-3.0, 0.14]),
    np.array([-1.5, 0.0]),
    np.array([-1.5, -0.07]),
    np.array([-1.5, 0.07]),
    np.array([0.0, 0.0]),
    np.array([0.0, -0.07]),
    np.array([0.0, 0.07]),
]


def action_key(action: np.ndarray) -> tuple[float, float]:
    return (round(float(action[0]), 6), round(float(action[1]), 6))


def _interp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def choose_k(dn_state: DNState, planner_cfg: Dict[str, float]) -> int:
    low_k = int(planner_cfg['low_risk_k'])
    medium_k = int(planner_cfg['medium_risk_k'])
    high_k = int(planner_cfg['high_risk_k'])
    critical_k = int(planner_cfg['critical_risk_k'])

    s = float(dn_state.severity)

    if s < 0.33:
        t = s / 0.33
        k = round(_interp(low_k, medium_k, t))
    elif s < 0.58:
        t = (s - 0.33) / (0.58 - 0.33)
        k = round(_interp(medium_k, high_k, t))
    elif s < 0.78:
        t = (s - 0.58) / (0.78 - 0.58)
        k = round(_interp(high_k, critical_k, t))
    else:
        k = critical_k

    if dn_state.delta_d >= 0.45:
        k += 3
    elif dn_state.delta_d >= 0.35:
        k += 2

    return int(max(critical_k, min(low_k, k)))


def select_candidates(
    previews: List[ActionPreview],
    dn_state: DNState,
    planner_cfg: Dict[str, float],
) -> Tuple[List[np.ndarray], int]:
    if not previews:
        return [], 0

    by_key = {action_key(p.action): p for p in previews}
    selected: List[np.ndarray] = []
    seen = set()

    def add(action: np.ndarray) -> None:
        key = action_key(action)
        if key in seen:
            return
        preview = by_key.get(key)
        if preview is None:
            return
        selected.append(preview.action)
        seen.add(key)

    ordered = sorted(previews, key=lambda p: p.score_hint, reverse=True)
    feasible_ordered = [p for p in ordered if p.feasible]
    k = choose_k(dn_state, planner_cfg)

    # 1. Якоря
    anchor_count = int(planner_cfg['emergency_anchor_count'])
    for anchor in ANCHORS[:anchor_count]:
        add(anchor)

    # 2. Лучшие feasible по score_hint
    best_take = max(6, min(10, k // 2))
    for preview in feasible_ordered[:best_take]:
        add(preview.action)

    # 3. Кластерное покрытие
    clusters: Dict[str, List[ActionPreview]] = {}
    for preview in feasible_ordered:
        clusters.setdefault(preview.cluster, []).append(preview)

    cluster_priority = [
        'hard_brake',
        'brake_left',
        'brake_right',
        'center',
        'left',
        'right',
    ]

    diversity_per_cluster = int(planner_cfg.get('diversity_per_cluster', 1))
    extra_cluster_take = 1 if dn_state.risk_level == 'medium' else 0

    for cluster_name in cluster_priority:
        take_n = diversity_per_cluster + extra_cluster_take
        for preview in clusters.get(cluster_name, [])[:take_n]:
            add(preview.action)

    # 4. Rescue-pass: отдельно добираем самые безопасные feasible кандидаты
    # по низкому gross_risk, даже если score_hint у них не топовый
    rescue_sorted = sorted(
        feasible_ordered,
        key=lambda p: (p.gross_risk, -p.score_hint)
    )
    rescue_take = 4 if dn_state.risk_level == 'medium' else 2
    for preview in rescue_sorted[:rescue_take]:
        add(preview.action)

    # 5. Основной добор до целевого размера
    target_total = max(k + 2, len(selected)) if dn_state.risk_level == 'medium' else max(k, len(selected))
    for preview in feasible_ordered:
        if len(selected) >= target_total:
            break
        add(preview.action)

    # 6. Если feasible не хватило, добираем из общего списка
    for preview in ordered:
        if len(selected) >= target_total:
            break
        add(preview.action)

    return selected, k