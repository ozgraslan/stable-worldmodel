#!/usr/bin/env python3
"""Plot planning-evaluation success rates, averaging repeated configurations."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml
except ImportError as exc:  # pragma: no cover - gives CLI users a useful error
    raise SystemExit(
        f'Missing dependency {exc.name!r}. Install matplotlib, numpy, and pyyaml.'
    ) from exc


DEFAULT_RESULT_NAME = 'ogb_cube_results.txt'
DEFAULT_EVAL_PREFIXES = ('solver.', 'plan_config.', 'eval.')
DEFAULT_EVAL_KEYS = ('world.env_name', 'world.env_type', 'world.ob_type')
IGNORED_KEYS = {'subdir', 'cache_dir', 'output.filename'}


def flatten(value: Any, prefix: str = '') -> dict[str, Any]:
    """Flatten a nested YAML mapping into dot-separated keys."""
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, child in value.items():
        child_key = f'{prefix}.{key}' if prefix else str(key)
        result.update(flatten(child, child_key))
    return result


def parse_result(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding='utf-8')
    try:
        config_text = text.split('==== CONFIG ====', 1)[1].split(
            '==== RESULTS ====', 1
        )[0]
    except IndexError as exc:
        raise ValueError('missing CONFIG or RESULTS marker') from exc

    config = yaml.safe_load(config_text)
    if not isinstance(config, dict):
        raise ValueError('CONFIG section is not a YAML mapping')
    match = re.search(r"['\"]success_rate['\"]\s*:\s*([-+\d.eE]+)", text)
    if not match:
        raise ValueError('success_rate was not found')

    flat = flatten(config)
    return {
        'file': str(path),
        'eval_dir': path.parent.name,
        'success_rate': float(match.group(1)),
        'config': flat,
    }


def short_value(key: str, value: Any) -> str:
    """Make common configuration values compact enough for plot labels."""
    if key == 'policy':
        name = Path(str(value)).name
        match = re.search(r'(?:weights?_)?step[_-]?(\d+)', name)
        return f'step {int(match.group(1)):,}' if match else name
    if key.endswith('._target_'):
        return str(value).rsplit('.', 1)[-1]
    if isinstance(value, list):
        return ','.join(map(str, value))
    return str(value)


def signature(row: dict[str, Any], keys: list[str], fallback: str) -> str:
    config = row['config']
    if keys == ['policy'] and 'policy' in config:
        return short_value('policy', config['policy'])
    parts = [
        f'{key.rsplit(".", 1)[-1]}={short_value(key, config[key])}'
        for key in keys
        if key in config
    ]
    return '\n'.join(parts) or fallback


def varying_keys(rows: list[dict[str, Any]]) -> set[str]:
    # Older result files may omit options that newer configs write explicitly
    # (for example, a false-valued flag). Such schema differences should not
    # turn a shared/default setting into an apparent evaluation configuration.
    common_keys = set(rows[0]['config']).intersection(
        *(row['config'] for row in rows[1:])
    )
    return {
        key
        for key in common_keys
        if key not in IGNORED_KEYS
        and len({repr(row['config'][key]) for row in rows}) > 1
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys = sorted(set().union(*(row['config'] for row in rows)))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'eval_dir',
                'success_rate',
                'wm_label',
                'eval_label',
                'file',
                *keys,
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                'eval_dir': row['eval_dir'],
                'success_rate': row['success_rate'],
                'wm_label': row['wm_label'],
                'eval_label': row['eval_label'],
                'file': row['file'],
                **row['config'],
            })


def find_result_files(checkpoint: Path, result_name: str) -> list[Path]:
    """Find results in either a run directory or its nested ``evals`` tree."""
    files: list[Path] = []
    direct_result = checkpoint / result_name
    if direct_result.is_file():
        files.append(direct_result)

    evals_dir = checkpoint / 'evals'
    if evals_dir.is_dir():
        files.extend(evals_dir.rglob(result_name))
    return sorted(set(files))


def plot(rows: list[dict[str, Any]], output: Path, title: str) -> None:
    wm_labels = list(dict.fromkeys(row['wm_label'] for row in rows))
    eval_labels = list(dict.fromkeys(row['eval_label'] for row in rows))
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[row['wm_label'], row['eval_label']].append(row['success_rate'])

    x = np.arange(len(wm_labels))
    group_width = 0.82
    width = group_width / len(eval_labels)
    fig_width = max(8.0, 1.15 * len(wm_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5), constrained_layout=True)
    for index, eval_label in enumerate(eval_labels):
        positions = x - group_width / 2 + width * (index + 0.5)
        values = [
            np.mean(grouped[label, eval_label])
            if grouped[label, eval_label] else np.nan
            for label in wm_labels
        ]
        errors = [
            np.std(grouped[label, eval_label])
            if grouped[label, eval_label] else np.nan
            for label in wm_labels
        ]
        bars = ax.bar(
            positions,
            values,
            width=width * 0.92,
            yerr=errors,
            capsize=3,
            label=eval_label,
        )
        ax.bar_label(bars, fmt='%.1f', padding=2, fontsize=8)

    ax.set_title(title)
    ax.set_xlabel('World model')
    ax.set_ylabel('Success rate (%)')
    ax.set_ylim(0, 105)
    ax.set_xticks(x, wm_labels, rotation=35, ha='right')
    ax.grid(axis='y', alpha=0.25)
    if len(eval_labels) > 1 or eval_labels[0] != 'eval':
        ax.legend(title='Planning / eval configuration', fontsize=8)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'checkpoints',
        type=Path,
        nargs='+',
        help='Run directories (each directory and its evals/ tree are searched)',
    )
    parser.add_argument('-o', '--output', type=Path,
                        help='Output PNG (default: CHECKPOINT/planning_success_rates.png)')
    parser.add_argument(
        '--result-name',
        default=DEFAULT_RESULT_NAME,
        help=f'Result filename to find (default: {DEFAULT_RESULT_NAME})',
    )
    parser.add_argument('--wm-param', action='append', default=[],
                        help='Dot-separated config key for WM grouping (repeatable)')
    parser.add_argument('--eval-param', action='append', default=[],
                        help='Dot-separated config key for eval grouping (repeatable)')
    parser.add_argument('--title', help='Plot title')
    args = parser.parse_args()

    files = sorted({
        path
        for checkpoint in args.checkpoints
        for path in find_result_files(checkpoint, args.result_name)
    })
    if not files:
        searched = ', '.join(map(str, args.checkpoints))
        raise SystemExit(
            f'No {args.result_name} in these directories or their evals/ trees: '
            f'{searched}'
        )

    rows = []
    for path in files:
        try:
            rows.append(parse_result(path))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            print(f'warning: skipping {path}: {exc}')
    if not rows:
        raise SystemExit('No valid result files were found')

    varying = varying_keys(rows)
    wm_keys = args.wm_param or (['policy'] if any(
        'policy' in row['config'] for row in rows
    ) else [])
    eval_keys = args.eval_param or sorted(
        key for key in varying
        if key not in wm_keys
        and (key.startswith(DEFAULT_EVAL_PREFIXES) or key in DEFAULT_EVAL_KEYS)
    )
    for row in rows:
        row['wm_label'] = signature(row, wm_keys, row['eval_dir'])
        row['eval_label'] = signature(row, eval_keys, 'eval')

    # Numeric checkpoint steps should appear in training order.
    def sort_key(row: dict[str, Any]) -> tuple[float, str, str]:
        if row['wm_label'].lower() == 'random':
            return -1, row['wm_label'], row['eval_label']
        match = re.search(r'[\d,]+', row['wm_label'])
        step = int(match.group().replace(',', '')) if match else float('inf')
        return step, row['wm_label'], row['eval_label']

    rows.sort(key=sort_key)
    output = args.output or args.checkpoints[0] / 'planning_success_rates.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    default_title = (
        f'{args.checkpoints[0].name}: planning success rates'
        if len(args.checkpoints) == 1
        else 'Combined planning success rates'
    )
    plot(rows, output, args.title or default_title)
    csv_path = output.with_suffix('.csv')
    write_csv(rows, csv_path)
    print(f'Plotted {len(rows)} evaluations from {len(files)} files')
    print(f'WM keys: {", ".join(wm_keys) or "eval directory"}')
    print(f'Eval keys: {", ".join(eval_keys) or "none vary"}')
    print(f'Wrote {output}')
    print(f'Wrote {csv_path}')


if __name__ == '__main__':
    main()
