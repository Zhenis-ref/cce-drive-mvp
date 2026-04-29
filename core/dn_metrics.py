from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List
import math
import numpy as np


@dataclass
class ActionPreview:
    action: np.ndarray
    cluster: str
    gross_risk: float
    ttc_like: float
    corridor_narrowing: float
    feasible: bool
    future_position: np.ndarray
    score_hint: float


@dataclass
class DNState:
    delta_n: float
    delta_d: float
    severity: float
    risk_level: str


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def softmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    denom = np.sum(exp)
    if denom <= 0.0:
        return np.full_like(values, 1.0 / len(values))
    return exp / denom


def action_cluster(action: np.ndarray) -> str:
    accel, steer = float(action[0]), float(action[1])
    if accel <= -3.5 and abs(steer) < 0.08:
        return 'hard_brake'
    if accel <= -1.0 and steer < -0.08:
        return 'brake_left'
    if accel <= -1.0 and steer > 0.08:
        return 'brake_right'
    if abs(steer) <= 0.08:
        return 'center'
    if steer < 0.0:
        return 'left'
    return 'right'


def compute_delta_n(previews: Iterable[ActionPreview]) -> float:
    items = list(previews)
    if not items:
        return 0.0

    risks = np.array([p.gross_risk for p in items], dtype=float)
    ttc_vals = np.array([p.ttc_like for p in items], dtype=float)
    corridor_vals = np.array([p.corridor_narrowing for p in items], dtype=float)
    infeasible_frac = 1.0 - float(np.mean([1.0 if p.feasible else 0.0 for p in items]))

    # Усиливаем чувствительность к опасному хвосту, а не только к среднему.
    risk_tail = float(np.percentile(risks, 80))
    ttc_tail = float(np.percentile(ttc_vals, 80))
    corridor_tail = float(np.percentile(corridor_vals, 80))

    raw = (
        0.35 * risk_tail
        + 0.25 * ttc_tail
        + 0.20 * corridor_tail
        + 0.10 * float(np.mean(risks))
        + 0.10 * infeasible_frac
    )
    return clamp01(raw)


def compute_delta_d(previews: Iterable[ActionPreview], top_m: int = 20) -> float:
    items = sorted(list(previews), key=lambda p: p.score_hint, reverse=True)[:top_m]
    if len(items) < 2:
        return 0.0

    positions = np.array([p.future_position for p in items], dtype=float)
    pos_var = np.var(positions[:, 0]) + np.var(positions[:, 1])
    pos_var_norm = clamp01(float(pos_var / 18.0))

    scores = np.array([p.score_hint for p in items], dtype=float)
    probs = softmax(scores)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    entropy_norm = clamp01(entropy / math.log(len(items)))

    clusters: Dict[str, float] = {}
    for p, prob in zip(items, probs):
        clusters[p.cluster] = clusters.get(p.cluster, 0.0) + float(prob)
    cluster_mass = np.array(sorted(clusters.values(), reverse=True), dtype=float)
    if cluster_mass.size <= 1:
        conflict = 0.0
    else:
        conflict = clamp01(float(np.sum(cluster_mass[1:])) / 0.7)

    raw = 0.45 * pos_var_norm + 0.30 * entropy_norm + 0.25 * conflict
    return clamp01(raw)


def compute_severity(delta_n: float, delta_d: float) -> float:
    # Немного усиливаем роль внешнего натиска, но сохраняем вклад внутренней дивергенции.
    return clamp01(0.72 * delta_n + 0.28 * delta_d)


def risk_level_from_severity(severity: float) -> str:
    if severity >= 0.78:
        return 'critical'
    if severity >= 0.58:
        return 'high'
    if severity >= 0.33:
        return 'medium'
    return 'low'


def evaluate_dn_state(previews: List[ActionPreview]) -> DNState:
    dn = compute_delta_n(previews)
    dd = compute_delta_d(previews)
    severity = compute_severity(dn, dd)
    return DNState(
        delta_n=dn,
        delta_d=dd,
        severity=severity,
        risk_level=risk_level_from_severity(severity),
    )