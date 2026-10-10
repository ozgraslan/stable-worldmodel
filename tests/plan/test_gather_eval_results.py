"""Evaluation reports preserve run settings and select classified videos."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.visualization.gather_eval_results import (
    discover,
    parse_result,
    select_videos,
    write_report,
)

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / 'scripts/visualization/gather_eval_results.py'
)


def result(directory, rate=50, outcomes='True, False, True, False'):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'cube_results.txt'
    path.write_text(
        '==== CONFIG ====\n'
        'policy: models/checkpoint.pt\n'
        'plan_config:\n  horizon: 5\n  receding_horizon: 2\n'
        'solver:\n  num_samples: 128\n'
        '==== RESULTS ====\n'
        f"metrics: {{'success_rate': {rate}, "
        f"'episode_successes': array([{outcomes}])}}\n"
        'evaluation_time: 1.5 seconds\n',
        encoding='utf-8',
    )
    return path


def test_discovery_uses_literal_prefix_and_includes_root(tmp_path):
    first = result(tmp_path / 'cube_[one]')
    second = result(tmp_path / 'model' / 'evals' / 'cube_[two]')
    result(tmp_path / 'other_cube_[three]')
    assert discover(tmp_path, 'cube_[', '*_results.txt') == [first, second]
    assert discover(first.parent, 'cube_[', '*_results.txt') == [first]


def test_latest_record_and_complete_numpy_arrays(tmp_path):
    path = result(tmp_path)
    previous = path.read_text()
    result(tmp_path, rate='np.float64(75.0)', outcomes='1., 0.,\n 1., 1.')
    path.write_text(previous + path.read_text())
    evaluation = parse_result(path)
    assert evaluation.record_count == 2
    assert evaluation.success_rate == 75
    assert evaluation.episode_successes == [True, False, True, True]
    assert evaluation.config['plan_config']['receding_horizon'] == 2


@pytest.mark.parametrize('outcomes', ['True, ..., False', '', '1, 2, 0'])
def test_incomplete_outcomes_do_not_guess_video_class(tmp_path, outcomes):
    evaluation = parse_result(result(tmp_path, outcomes=outcomes))
    (tmp_path / 'env_0.mp4').touch()
    (tmp_path / 'episode_remaining_0.mp4').touch()
    assert evaluation.episode_successes == []
    assert select_videos(evaluation) == {True: [], False: []}


def test_video_selection_supports_both_naming_styles(tmp_path):
    evaluation = parse_result(result(tmp_path))
    names = [
        'episode_0.mp4',
        'episode_1.mp4',
        'env_2.mp4',
        'env_3.mp4',
        'env_99.mp4',
        'zzz_suc_True.mp4',
        'zzz_success_False.mp4',
        'unknown.mp4',
    ]
    for name in names:
        (tmp_path / name).touch()
    videos = select_videos(evaluation)
    assert [path.name for path in videos[True]] == [
        'env_2.mp4',
        'episode_0.mp4',
    ]
    assert [path.name for path in videos[False]] == [
        'env_3.mp4',
        'episode_1.mp4',
    ]


@pytest.mark.parametrize('copy_videos', [True, False])
def test_report_embeds_videos_and_keeps_metrics_and_config(
    tmp_path, copy_videos
):
    root = tmp_path / 'inputs'
    evaluation = parse_result(result(root / 'cube_one'))
    for success in ('True', 'False'):
        for index in range(3):
            (
                evaluation.source.parent / f'case {index}_suc_{success}.mp4'
            ).write_bytes(b'video')
    output = tmp_path / 'report' / 'results.md'
    write_report([evaluation], output, root, 'cube_', copy_videos=copy_videos)
    report = output.read_text()
    assert '| 50.00 |' in report
    assert '[Shared parameters](results_shared_parameters.md)' in report
    assert '### Configuration' not in report
    assert 'num_samples: 128' not in report
    shared = output.with_name('results_shared_parameters.md').read_text()
    assert '| solver.num_samples | 128 |' in shared
    assert '| policy | models/checkpoint.pt |' in shared
    assert '[Back to evaluation results](results.md)' in shared
    assert 'evaluation_time: 1.5 seconds' in report
    video_root = (
        'results_videos/run_001' if copy_videos else '../inputs/cube_one'
    )
    video_url = f'{video_root}/case%200_suc_True.mp4'
    assert report.count('<video controls width="640" preload="none">') == 4
    assert report.count('</video>') == 4
    assert f'<source src="{video_url}" type="video/mp4">' in report
    assert f'[Open Success 1]({video_url})' in report
    copied = list(output.parent.rglob('*.mp4'))
    assert len(copied) == (4 if copy_videos else 0)
    assert all(path.read_bytes() == b'video' for path in copied)


def test_report_compares_differing_important_settings(tmp_path):
    first = parse_result(result(tmp_path / 'cube_a'))
    second = parse_result(result(tmp_path / 'cube_b', rate=75))
    second.config['policy'] = 'models/other.pt'
    second.config['plan_config']['horizon'] = 10
    second.config['solver']['num_samples'] = 256
    first.config['eval'] = {'source': 'env', 'start_from_beginning': False}
    second.config['eval'] = {'source': 'env'}
    first.config['subdir'] = 'cube_a'
    second.config['subdir'] = 'cube_b'
    first.config['world'] = {'options': {}, 'image_keys': ['pixels', 'goal']}
    second.config['world'] = {'options': {}, 'image_keys': ['pixels', 'goal']}
    output = tmp_path / 'comparison.md'
    write_report([first, second], output, tmp_path, 'cube_')
    report = output.read_text()
    shared = (tmp_path / 'comparison_shared_parameters.md').read_text()
    assert 'models/checkpoint.pt' in report and 'models/other.pt' in report
    assert 'plan_config.horizon' in report
    assert 'solver.num_samples' in report
    assert 'eval.start_from_beginning' in report
    assert 'False' in report and '—' in report
    assert 'subdir' not in report and 'subdir' not in shared
    assert 'plan_config.receding_horizon' not in report
    assert '| plan_config.receding_horizon | 2 |' in shared
    assert '| eval.source | env |' in shared
    assert '| world.options | {} |' in shared
    assert 'world.image_keys' in shared
    assert 'eval.start_from_beginning' not in shared
    assert 'solver.num_samples' not in shared


def test_shared_report_handles_no_common_settings(tmp_path):
    first = parse_result(result(tmp_path / 'cube_a'))
    second = parse_result(result(tmp_path / 'cube_b'))
    first.config = {'policy': 'one.pt'}
    second.config = {'policy': 'two.pt'}
    output = tmp_path / 'comparison.md'
    write_report([first, second], output, tmp_path, 'cube_')
    shared = (tmp_path / 'comparison_shared_parameters.md').read_text()
    assert 'No parameters are shared' in shared
    assert 'one.pt' in output.read_text()


def test_cli_skips_bad_results_and_reports_missing_videos(tmp_path):
    result(tmp_path / 'cube_good')
    bad = result(tmp_path / 'cube_bad')
    bad.write_text('incomplete evaluation')
    output = tmp_path / 'results.md'
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path),
            '--prefix',
            'cube_',
            '-o',
            str(output),
            '--link-videos',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert 'warning: skipping' in completed.stderr
    report = output.read_text()
    assert 'Only 0 of 2 requested failure videos available.' in report
    assert 'Only 0 of 2 requested success videos available.' in report
    assert 'Skipped result files' in report
    assert not (tmp_path / 'results_videos').exists()


def test_cli_no_matches_does_not_write_report(tmp_path):
    result(tmp_path / 'other_run')
    output = tmp_path / 'results.md'
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path),
            '--prefix',
            'cube_',
            '-o',
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert 'No valid results' in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize('copy_assets', [True, False])
def test_report_includes_landscapes_and_expert_selection(
    tmp_path, copy_assets
):
    evaluation = parse_result(result(tmp_path / 'cube_eval'))
    diagnostics = {
        'config': evaluation.config,
        'expert_selection_rate': 0.5,
        'mode': 'mpc',
        'source': 'env',
        'num_replans': 2,
        'iterations': [
            {'env_index': 0, 'iteration': 0, 'expert_rank': 17},
            {'env_index': 0, 'iteration': 29, 'expert_rank': 3},
        ],
        'selections': [
            {
                'env_index': 0,
                'expert_selected': True,
                'expert_rank': 1,
                'expert_cost': 0.1,
                'selected_cost': 0.1,
            }
        ],
    }
    comparison = evaluation.source.parent / 'comparisons' / 'env_0.mp4'
    comparison.parent.mkdir()
    comparison.write_bytes(b'comparison video')
    diagnostics['comparison_videos'] = [
        {
            'video': 'comparisons/env_0.mp4',
            'env_index': 0,
            'replan_index': 1,
            'outcomes': {
                'expert': {'final_goal_success': True},
                'selected': {'final_goal_success': False},
            },
        }
    ]
    expert_path = evaluation.source.parent / 'expert_candidates.json'
    expert_path.write_text(json.dumps(diagnostics))
    landscapes = tmp_path / 'landscapes'
    for index in range(2):
        directory = landscapes / f'landscape_{index}'
        directory.mkdir(parents=True)
        (directory / 'results.json').write_text(
            json.dumps(
                {
                    'config': {
                        'policy': 'model.pt',
                        'controller_scale': 0.1 + index,
                    },
                    'best_grid_energy': 0.2,
                    'comparison_video': 'best_action/env_0.mp4',
                    'cem_energy': 0.3,
                    'ground_truth_action_energy': 0.1,
                    'grid_actions_with_lower_energy': 0,
                }
            )
        )
        (directory / 'landscape.png').write_bytes(b'plot')
        (directory / 'best_action').mkdir()
        (directory / 'best_action' / 'env_0.mp4').write_bytes(b'video')
        (directory / 'landscape_3d.png').write_bytes(b'3d plot')
    output = tmp_path / 'report' / 'results.md'
    write_report(
        [evaluation],
        output,
        tmp_path,
        'cube_',
        copy_videos=copy_assets,
        landscape_dirs=[landscapes],
        expert_dirs=[evaluation.source.parent],
    )
    report = output.read_text()
    assert '## Action-space energy landscapes' in report
    assert '## Expert-action selection' in report
    assert 'controller_scale' in report and '1.1' in report
    assert 'expert_selection_rate' in report and '0.5' in report
    assert 'expert_rank' in report
    assert 'initial_expert_rank' in report
    assert 'last_iteration_expert_rank' in report
    assert '| 17 | 3 |' in report
    assert 'initial_expert_cost' not in report
    assert 'initial_cem_mean_cost' not in report
    assert '<video controls width="960"' in report
    assert 'Final goal success' in report
    assert '| expert | True |' in report
    assert '| selected | False |' in report
    comparison_copies = list((output.parent / 'results_videos').rglob('*.mp4'))
    assert len(comparison_copies) == (3 if copy_assets else 0)
    assert report.count('### cube_eval') == 1
    assert '![landscape]' in report and '![landscape_3d]' in report
    assert 'Minimum-energy grid action | Goal' in report
    assert (
        output.parent / 'results_videos/landscape_001/env_0.mp4'
    ).is_file() == copy_assets
    figures = list((output.parent / 'results_figures').rglob('*.png'))
    assert len(figures) == (4 if copy_assets else 0)
    shared = (output.parent / 'results_shared_parameters.md').read_text()
    assert '## Action-space energy landscapes' in shared
    assert '| policy | model.pt |' in shared
    assert 'controller_scale' not in shared
    assert '## Expert-action selection' in shared


def test_cli_includes_external_expert_diagnostics_and_skips_bad_json(tmp_path):
    result(tmp_path / 'cube_eval')
    expert = tmp_path / 'external_experiment'
    expert.mkdir()
    (expert / 'expert_candidates.json').write_text(
        json.dumps(
            {
                'expert_selection_rate': 1.0,
                'config': {'policy': 'model.pt'},
                'selections': [
                    {'env_index': index, 'expert_selected': True}
                    for index in range(21)
                ],
            }
        )
    )
    landscape = tmp_path / 'bad_landscape'
    landscape.mkdir()
    (landscape / 'results.json').write_text('not JSON')
    output = tmp_path / 'results.md'
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path),
            '--prefix',
            'cube_',
            '-o',
            str(output),
            '--expert-dir',
            str(expert),
            '--landscape-dir',
            str(landscape),
            '--link-videos',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = output.read_text()
    assert '## Expert-action selection' in report
    assert 'first 20 of 21 selections' in report
    assert 'No saved PNG' not in report
    assert 'bad_landscape' in report
    assert 'warning: skipping' in completed.stderr
