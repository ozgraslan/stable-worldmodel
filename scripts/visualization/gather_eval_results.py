#!/usr/bin/env python3
"""Collect prefixed evaluation directories into a Markdown comparison report.

Example (run from the repository root)::

    python scripts/visualization/gather_eval_results.py /path/to/checkpoints \
        --prefix cube_ --output results.md

Searches recursively, uses the latest record in each *_results.txt file, and
copies up to two success and two failure videos per evaluation beside the
report. Shared settings go into a linked *_shared_parameters.md file; the
main report compares differing model, planning, and evaluation parameters.
Add --landscape-dir and --expert-dir to include saved diagnostic experiments.
Videos use HTML players with fallback links. Requires only PyYAML;
no model, GPU, or video decoder is needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
OUTCOME = re.compile(r'(?:success_|suc_)(true|false)\.mp4$', re.IGNORECASE)
INDEX = re.compile(r'^(?:episode|env)_(\d+)\.mp4$', re.IGNORECASE)
RECORD = re.compile(
    r'^==== CONFIG ====\s*\n(.*?)^==== RESULTS ====\s*\n(.*?)(?='
    r'^==== CONFIG ====|\Z)',
    re.MULTILINE | re.DOTALL,
)


@dataclass
class Evaluation:
    source: Path
    config: dict[str, Any]
    config_text: str
    results_text: str
    success_rate: float
    episode_successes: list[bool]
    record_count: int


def parse_result(path: Path) -> Evaluation:
    """Read the latest appended record without executing metric expressions."""
    records = list(RECORD.finditer(path.read_text(encoding='utf-8')))
    if not records:
        raise ValueError('missing CONFIG / RESULTS sections')
    config_text, results_text = records[-1].groups()
    config = yaml.safe_load(config_text)
    if not isinstance(config, dict):
        raise TypeError('CONFIG must be a YAML mapping')
    match = re.search(
        rf"['\"]success_rate['\"]\s*:\s*"
        rf'(?:np\.float(?:32|64)\()?({NUMBER})',
        results_text,
    )
    if not match:
        raise ValueError('missing numeric success_rate')
    rate = float(match.group(1))
    if not math.isfinite(rate) or not 0 <= rate <= 100:
        raise ValueError('success_rate must be between 0 and 100 percent')

    # NumPy prints arrays as Python-like text, not JSON. Parse only complete
    # boolean / 0-or-1 arrays; ellipsized arrays cannot map video indices safely.
    match = re.search(
        r"['\"]episode_successes['\"]\s*:\s*"
        r'(?:array\(\s*)?\[([^\]]*)\]',
        results_text,
        re.DOTALL,
    )
    outcomes: list[bool] = []
    if match:
        tokens = re.split(r'[,\s]+', match.group(1).strip())
        if tokens and all(
            token.lower() in {'true', 'false'}
            or re.fullmatch(NUMBER, token)
            and float(token) in {0, 1}
            for token in tokens
        ):
            outcomes = [
                token.lower() == 'true'
                or (token.lower() != 'false' and float(token) == 1)
                for token in tokens
            ]
    return Evaluation(
        path,
        config,
        config_text.strip(),
        results_text.strip(),
        rate,
        outcomes,
        len(records),
    )


def discover(root: Path, prefix: str, result_pattern: str) -> list[Path]:
    """Match literal directory-name prefixes, including the root itself."""
    directories = [root] if root.name.startswith(prefix) else []
    directories.extend(
        path
        for path in root.rglob('*')
        if path.is_dir() and path.name.startswith(prefix)
    )
    return sorted(
        {
            path
            for directory in directories
            for path in directory.glob(result_pattern)
            if path.is_file()
        }
    )


def select_videos(evaluation: Evaluation) -> dict[bool, list[Path]]:
    """Select two videos per outcome, in filename order, from the same run."""
    selected: dict[bool, list[Path]] = {True: [], False: []}
    for path in sorted(evaluation.source.parent.glob('*.mp4')):
        match = OUTCOME.search(path.name)
        if match:
            success = match.group(1).lower() == 'true'
        else:
            match = INDEX.match(path.name)
            if not match or int(match.group(1)) >= len(
                evaluation.episode_successes
            ):
                continue
            success = evaluation.episode_successes[int(match.group(1))]
        if len(selected[success]) < 2:
            selected[success].append(path)
    return selected


def cell(value: Any) -> str:
    return (
        str(value)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('|', '&#124;')
        .replace('\n', '<br>')
    )


def flatten_config(config: dict[str, Any], prefix: str = '') -> dict[str, Any]:
    """Flatten nested settings, retaining lists and empty mappings as values."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        name = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(value, dict) and value:
            flat.update(flatten_config(value, name))
        else:
            flat[name] = value
    return flat


def important_parameter(key: str) -> bool:
    """Include experiment settings, omitting output and infrastructure paths."""
    return key in {
        'policy',
        'seed',
        'bf16',
        'compile',
        'state_columns',
        'action_block',
        'horizon',
        'history_len',
        'controller_scale',
        'grid_size',
        'samples',
        'normalize_reps',
        'goal_source',
        'goal_action_index',
        'dataset_name',
        'episode',
        'start_step',
        'goal_offset_steps',
        'play_in_reverse',
    } or (
        key.startswith(
            (
                'world.',
                'solver.',
                'objective.',
                'plan_config.',
                'eval.',
                'dataset.',
                'model.',
                'wm.',
            )
        )
        and key not in {'world.expert_output_dir'}
    )


def relative_url(path: Path, output: Path) -> str:
    relative = os.path.relpath(path, output.parent)
    return quote(Path(relative).as_posix(), safe='/.')


def link(path: Path, output: Path, label: str) -> str:
    return f'[{label}]({relative_url(path, output)})'


def experiment_files(directories: list[Path], name: str) -> list[Path]:
    """Accept individual experiment directories or trees of experiments."""
    return sorted(
        {
            path
            for directory in directories
            for path in directory.expanduser().resolve().rglob(name)
            if path.is_file()
        }
    )


def experiment_section(
    title: str,
    files: list[Path],
    output: Path,
    copy_assets: bool,
    warnings: list[str],
    landscape: bool = False,
) -> tuple[list[str], list[str]]:
    """Summarize saved diagnostic JSON and link or copy visualization assets."""
    rows = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or not isinstance(
                data.get('config', {}), dict
            ):
                raise TypeError('experiment metadata must contain mappings')
            required_key = (
                'best_grid_energy' if landscape else 'expert_selection_rate'
            )
            if required_key not in data:
                raise ValueError(
                    f'missing {required_key} in experiment results'
                )
            rows.append((path, data))
        except (OSError, TypeError, ValueError) as exc:
            warning = f'{path}: {exc}'
            warnings.append(warning)
            print(f'warning: skipping {warning}', file=sys.stderr)
    if not rows:
        return [], []
    configs = [flatten_config(data.get('config', {})) for _, data in rows]
    keys = sorted(set().union(*(config.keys() for config in configs)))
    shared = {
        key: configs[0][key]
        for key in keys
        if all(key in config for config in configs)
        and all(config[key] == configs[0][key] for config in configs[1:])
    }
    differing = [
        key for key in keys if key not in shared and important_parameter(key)
    ]
    common = ['', f'## {title}', '', '| Parameter | Value |', '| --- | --- |']
    common.extend(f'| {cell(k)} | {cell(v)} |' for k, v in shared.items())
    lines = ['', f'## {title}', '']
    if not landscape:
        lines.extend(
            [
                (
                    'Expert selection rate (0–1) measures how often the injected '
                    'expert sequence was selected. It does not by itself measure '
                    'task success.'
                ),
                '',
            ]
        )
    metric_keys = (
        [
            'best_grid_energy',
            'cem_energy',
            'ground_truth_action_energy',
            'grid_actions_with_lower_energy',
        ]
        if landscape
        else ['expert_selection_rate', 'num_replans', 'mode', 'source']
    )
    headers = ['Experiment', *metric_keys, *differing]
    lines.extend(
        [
            '| ' + ' | '.join(map(cell, headers)) + ' |',
            '| ' + ' | '.join('---' for _ in headers) + ' |',
        ]
    )
    for (path, data), config in zip(rows, configs):
        values = [link(path, output, cell(path.parent.name))]
        values.extend(cell(data.get(key, '—')) for key in metric_keys)
        values.extend(cell(config.get(key, '—')) for key in differing)
        lines.append('| ' + ' | '.join(values) + ' |')
    for index, (path, diagnostics) in enumerate(rows, 1):
        lines.extend(
            [
                '',
                f'### {cell(path.parent.name)}',
                '',
                link(path, output, 'Full diagnostic results'),
                '',
            ]
        )
        if landscape:
            video = diagnostics.get('comparison_video')
            if isinstance(video, str):
                source = (path.parent / video).resolve()
                if (
                    source.is_relative_to(path.parent.resolve())
                    and source.is_file()
                ):
                    target = source
                    if copy_assets:
                        target = (
                            output.parent
                            / f'{output.stem}_videos'
                            / f'landscape_{index:03d}'
                            / source.name
                        )
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if source != target.resolve():
                            shutil.copy2(source, target)
                    lines.extend(
                        [
                            '**Minimum-energy grid action | Goal**',
                            '',
                            '<video controls width="640" preload="none">',
                            f'  <source src="{relative_url(target, output)}" type="video/mp4">',
                            '</video>',
                            '',
                            link(
                                target, output, 'Open action/goal comparison'
                            ),
                            '',
                        ]
                    )
                else:
                    warnings.append(
                        f'{path}: comparison video missing or outside experiment: {source}'
                    )
            images = [
                path.parent / name
                for name in (
                    'landscape.png',
                    'landscape_3d.png',
                    'trajectory.png',
                    'start.png',
                    'goal.png',
                )
                if (path.parent / name).is_file()
            ]
            if not images:
                lines.extend(['No saved PNG visualizations found.', ''])
            for source in images:
                target = source
                if copy_assets:
                    target = (
                        output.parent
                        / f'{output.stem}_figures'
                        / f'landscape_{index:03d}'
                        / source.name
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if source.resolve() != target.resolve():
                        shutil.copy2(source, target)
                lines.extend(
                    [
                        f'![{cell(source.stem)}]({relative_url(target, output)})',
                        '',
                    ]
                )
        else:
            comparisons = diagnostics.get('comparison_videos', [])
            for video_index, comparison in enumerate(comparisons, 1):
                if not isinstance(comparison, dict) or not isinstance(
                    comparison.get('video'), str
                ):
                    continue
                source = (path.parent / comparison['video']).resolve()
                if (
                    not source.is_relative_to(path.parent.resolve())
                    or not source.is_file()
                ):
                    warnings.append(
                        f'{path}: comparison video missing or outside experiment: {source}'
                    )
                    continue
                target = source
                if copy_assets:
                    target = (
                        output.parent
                        / f'{output.stem}_videos'
                        / f'expert_{index:03d}'
                        / f'comparison_{video_index:03d}'
                        / source.name
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if source != target.resolve():
                        shutil.copy2(source, target)
                label = (
                    f'Expert / selected / goal — environment '
                    f'{comparison.get("env_index", "—")}'
                )
                lines.extend(
                    [
                        f'**{cell(label)}**',
                        '',
                        '<video controls width="960" preload="none">',
                        f'  <source src="{relative_url(target, output)}" type="video/mp4">',
                        '</video>',
                        '',
                        link(target, output, 'Open comparison'),
                        '',
                    ]
                )
                outcomes = comparison.get('outcomes', {})
                if isinstance(outcomes, dict):
                    lines.extend(
                        ['| Sequence | Final goal success |', '| --- | --- |']
                    )
                    for name, outcome in outcomes.items():
                        if isinstance(outcome, dict):
                            lines.append(
                                f'| {cell(name)} | {cell(outcome.get("final_goal_success", "—"))} |'
                            )
                    lines.append('')
            selections = diagnostics.get('selections', [])
            if isinstance(selections, list) and selections:
                first_iterations = {}
                last_iterations = {}
                for row in diagnostics.get('iterations', []):
                    if not isinstance(row, dict) or not isinstance(
                        row.get('iteration'), int
                    ):
                        continue
                    key = (row.get('replan_index'), row.get('env_index'))
                    if row['iteration'] == 0:
                        first_iterations[key] = row
                    if row['iteration'] > last_iterations.get(key, {}).get(
                        'iteration', -1
                    ):
                        last_iterations[key] = row
                fields = [
                    'env_index',
                    'env_step',
                    'expert_selected',
                    'initial_expert_rank',
                    'last_iteration_expert_rank',
                    'expert_tied_for_best',
                    'native_cem_mean_expert_rmse',
                ]
                if len(selections) > 20:
                    lines.extend(
                        [
                            (
                                f'Showing the first 20 of {len(selections)} '
                                'selections; see the full diagnostic results '
                                'for all selections.'
                            ),
                            '',
                        ]
                    )
                lines.extend(
                    [
                        (
                            'Expert ranks use the sampled candidates at '
                            'iterations 0 and the last optimization iteration; '
                            'rank 1 is best (ties share rank).'
                        ),
                        '',
                        '| ' + ' | '.join(fields) + ' |',
                        '| ' + ' | '.join('---' for _ in fields) + ' |',
                    ]
                )
                for selection in selections[:20]:
                    if isinstance(selection, dict):
                        selection = dict(selection)
                        key = (
                            selection.get('replan_index'),
                            selection.get('env_index'),
                        )
                        selection.setdefault(
                            'initial_expert_rank',
                            first_iterations.get(key, {}).get(
                                'expert_rank', '—'
                            ),
                        )
                        selection.setdefault(
                            'last_iteration_expert_rank',
                            last_iterations.get(key, {}).get(
                                'expert_rank', '—'
                            ),
                        )
                        lines.append(
                            '| '
                            + ' | '.join(
                                cell(selection.get(key, '—')) for key in fields
                            )
                            + ' |'
                        )
    return lines, common


def write_report(
    evaluations: list[Evaluation],
    output: Path,
    root: Path,
    prefix: str,
    copy_videos: bool = True,
    warnings: list[str] | None = None,
    landscape_dirs: list[Path] | None = None,
    expert_dirs: list[Path] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    warnings = list(warnings or [])
    configs = [flatten_config(evaluation.config) for evaluation in evaluations]
    keys = sorted(set().union(*(config.keys() for config in configs)))
    shared = {
        key: configs[0][key]
        for key in keys
        if all(key in config for config in configs)
        and all(config[key] == configs[0][key] for config in configs[1:])
    }
    differing = [
        key for key in keys if key not in shared and important_parameter(key)
    ]
    shared_path = output.with_name(f'{output.stem}_shared_parameters.md')
    shared_lines = [
        '# Shared evaluation parameters',
        '',
        (
            f'Settings present with the same value in all {len(evaluations)} '
            'evaluations.'
        ),
        '',
        link(output, shared_path, 'Back to evaluation results'),
        '',
    ]
    if shared:
        shared_lines.extend(['| Parameter | Value |', '| --- | --- |'])
        shared_lines.extend(
            f'| {cell(key)} | {cell(value)} |' for key, value in shared.items()
        )
    else:
        shared_lines.append('No parameters are shared across all evaluations.')
    headers = ['Evaluation', 'Success rate (%)', *map(cell, differing)]
    lines = [
        '# Evaluation results',
        '',
        f'Searched `{root}` for directory names starting with `{prefix}`.',
        '',
        (
            'Success rates are percentages. Each result file uses its latest '
            'appended evaluation. Videos are selected in filename order '
            '(up to two per outcome).'
        ),
        '',
        link(shared_path, output, 'Shared parameters'),
        '',
        (
            'Only differing model, planning, and evaluation settings are shown '
            'below. A dash means the parameter was absent from that run.'
        ),
        '',
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---', '---:', *['---'] * len(differing)]) + ' |',
    ]
    for index, (evaluation, config) in enumerate(zip(evaluations, configs), 1):
        name = evaluation.source.parent.relative_to(root).as_posix()
        values = [
            f'[Run {index}](#run-{index}) — {cell(name)}',
            f'{evaluation.success_rate:.2f}',
            *(cell(config.get(key, '—')) for key in differing),
        ]
        lines.append('| ' + ' | '.join(values) + ' |')

    for index, evaluation in enumerate(evaluations, 1):
        lines.extend(
            [
                '',
                f'## Run {index}',
                '',
                f'Source: {link(evaluation.source, output, "result file")}',
                '',
            ]
        )
        if evaluation.record_count > 1:
            lines.extend(
                [
                    (
                        f'Using record {evaluation.record_count} of '
                        f'{evaluation.record_count}. Videos reflect the files '
                        'currently in the evaluation directory.'
                    ),
                    '',
                ]
            )
        videos = select_videos(evaluation)
        for success, label in ((False, 'Failure'), (True, 'Success')):
            lines.extend([f'### {label} videos', ''])
            for video_index, source in enumerate(videos[success], 1):
                target = source
                if copy_videos:
                    target = (
                        output.parent
                        / f'{output.stem}_videos'
                        / f'run_{index:03d}'
                        / source.name
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if source.resolve() != target.resolve():
                        shutil.copy2(source, target)
                lines.extend(
                    [
                        f'**{label} {video_index}**',
                        '',
                        '<video controls width="640" preload="none">',
                        (
                            f'  <source src="{relative_url(target, output)}" '
                            'type="video/mp4">'
                        ),
                        '  Your viewer does not support embedded video.',
                        '</video>',
                        '',
                        link(target, output, f'Open {label} {video_index}'),
                        '',
                    ]
                )
            if len(videos[success]) < 2:
                lines.append(
                    f'Only {len(videos[success])} of 2 requested '
                    f'{label.lower()} videos available. Unclassified videos '
                    'are omitted.'
                )
            lines.append('')
        lines.extend(
            [
                '### Metrics',
                '',
                '```text',
                evaluation.results_text,
                '```',
            ]
        )
    expert_files = sorted(
        {
            *experiment_files(expert_dirs or [], 'expert_candidates.json'),
            *(
                evaluation.source.parent / 'expert_candidates.json'
                for evaluation in evaluations
                if (
                    evaluation.source.parent / 'expert_candidates.json'
                ).is_file()
            ),
        }
    )
    for title, files, landscape in (
        (
            'Action-space energy landscapes',
            experiment_files(landscape_dirs or [], 'results.json'),
            True,
        ),
        ('Expert-action selection', expert_files, False),
    ):
        section, common = experiment_section(
            title, files, output, copy_videos, warnings, landscape
        )
        lines.extend(section)
        shared_lines.extend(common)
    shared_path.write_text('\n'.join(shared_lines) + '\n', encoding='utf-8')
    if warnings:
        lines.extend(['', '## Skipped result files', ''])
        lines.extend(f'- {cell(warning)}' for warning in warnings)
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'root', type=Path, help='Directory searched recursively'
    )
    parser.add_argument(
        '--prefix', required=True, help='Literal evaluation directory prefix'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=Path('results.md'),
        help='Markdown output (default: results.md)',
    )
    parser.add_argument(
        '--result-pattern',
        default='*_results.txt',
        help='Result filename glob within each matched directory',
    )
    parser.add_argument(
        '--link-videos',
        action='store_true',
        help='Link original videos and figures instead of copying them',
    )
    parser.add_argument(
        '--landscape-dir',
        type=Path,
        action='append',
        default=[],
        help='Energy-landscape output directory or tree (repeatable)',
    )
    parser.add_argument(
        '--expert-dir',
        type=Path,
        action='append',
        default=[],
        help='Expert-candidate output directory or tree (repeatable)',
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    for directory in [*args.landscape_dir, *args.expert_dir]:
        if not directory.expanduser().is_dir():
            parser.error(f'experiment directory does not exist: {directory}')
    if not root.is_dir():
        parser.error(f'root is not a directory: {root}')
    if not args.prefix:
        parser.error('--prefix must not be empty')
    evaluations: list[Evaluation] = []
    warnings: list[str] = []
    for path in discover(root, args.prefix, args.result_pattern):
        try:
            evaluations.append(parse_result(path))
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            warning = f'{path}: {exc}'
            warnings.append(warning)
            print(f'warning: skipping {warning}', file=sys.stderr)
    if not evaluations:
        raise SystemExit('No valid results in matching evaluation directories')
    write_report(
        evaluations,
        args.output.expanduser().resolve(),
        root,
        args.prefix,
        copy_videos=not args.link_videos,
        warnings=warnings,
        landscape_dirs=args.landscape_dir,
        expert_dirs=args.expert_dir,
    )
    print(f'Wrote {len(evaluations)} evaluations to {args.output}')


if __name__ == '__main__':
    main()
