from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
from collections import Counter
import matplotlib.pyplot as plt


def read_rows(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        raise ValueError('CSV is empty')

    scenario = rows[0]['scenario']
    cr_values = [float(r['compression_ratio']) for r in rows]
    bf_margin = [float(r['bf_margin']) for r in rows]
    cce_margin = [float(r['cce_margin']) for r in rows]
    bf_safe = [int(r['bf_safe']) for r in rows]
    cce_safe = [int(r['cce_safe']) for r in rows]
    bf_latency = [float(r['bf_latency_ms']) for r in rows]
    cce_latency = [float(r['cce_latency_ms']) for r in rows]
    selected_k = [int(r['selected_k']) for r in rows]
    selected_total = [int(r['selected_total']) for r in rows]
    severity = [float(r['severity']) for r in rows]
    risk_levels = [r['risk_level'] for r in rows]

    out_dir = args.csv.parents[1] / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scatter margin
    plt.figure(figsize=(7, 5))
    plt.scatter(bf_margin, cce_margin, alpha=0.75)
    mn = min(min(bf_margin), min(cce_margin))
    mx = max(max(bf_margin), max(cce_margin))
    plt.plot([mn, mx], [mn, mx], linestyle='--')
    plt.xlabel('BF minimum safety margin')
    plt.ylabel('CCE minimum safety margin')
    plt.title(f'Margin comparison: {scenario}')
    scatter_path = out_dir / f'margin_scatter_{scenario}.png'
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()

    # CR histogram
    plt.figure(figsize=(7, 5))
    bins = min(12, max(5, len(set(round(v, 3) for v in cr_values))))
    plt.hist(cr_values, bins=bins)
    plt.xlabel('Compression ratio')
    plt.ylabel('Count')
    plt.title(f'Compression ratio histogram: {scenario}')
    hist_path = out_dir / f'cr_hist_{scenario}.png'
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()

    level_counter = Counter(risk_levels)

    print(f'Scenario: {scenario}')
    print(f'Rows: {len(rows)}')
    print(f'BF safety mean:   {statistics.mean(bf_safe):.3f}')
    print(f'CCE safety mean:  {statistics.mean(cce_safe):.3f}')
    print(f'Mean CR:          {statistics.mean(cr_values):.3f}x')
    print(f'Std CR:           {statistics.pstdev(cr_values):.3f}')
    print(f'Min CR:           {min(cr_values):.3f}x')
    print(f'Max CR:           {max(cr_values):.3f}x')
    print(f'Mean BF latency:  {statistics.mean(bf_latency):.3f} ms')
    print(f'Mean CCE latency: {statistics.mean(cce_latency):.3f} ms')
    print(f'Mean severity:    {statistics.mean(severity):.3f}')
    print(f'Min/Max severity: {min(severity):.3f} / {max(severity):.3f}')
    print(f'Mean selected_k:  {statistics.mean(selected_k):.3f}')
    print(f'Min/Max selected_k: {min(selected_k)} / {max(selected_k)}')
    print(f'Mean selected_total: {statistics.mean(selected_total):.3f}')
    print(f'Risk levels:      {dict(level_counter)}')
    print(f'Saved: {scatter_path}')
    print(f'Saved: {hist_path}')


if __name__ == '__main__':
    main()