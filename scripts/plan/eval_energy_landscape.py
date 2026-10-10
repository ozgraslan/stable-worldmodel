"""Plot endpoint prediction energy for expert OverheadPushT trajectories.

Run from the repository root, for example::

    python scripts/plan/eval_energy_landscape.py \
        policy=/path/to/weights.pt expert_checkpoint=/path/to/expert.pt \
        dataset_name=/path/to/training.lance use_expert_action_mean=true

The dataset supplies normalization statistics only. Start and goal observations
are sampled from a successful expert rollout using the eval_wm environment.
See ``docs/tutorial/energy_landscape.md`` for settings and output descriptions.
"""

import json
from contextlib import closing
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import v2

import stable_worldmodel as swm
from stable_worldmodel.data.normalization import get_scaler
from stable_worldmodel.planning.energy_landscape import (
    EndpointEnergy,
    NotebookDynamics,
    evaluate_action_sequences,
    xy_action_grid,
)
from stable_worldmodel.wm.utils import load_pretrained

if not OmegaConf.has_resolver('energy_rollout_steps'):
    OmegaConf.register_new_resolver(
        'energy_rollout_steps',
        lambda horizon, block: int(horizon) * int(block),
    )


def case_from_observations(start, goal, stats, model, cfg):
    """Apply training preprocessing to expert start and goal observations."""
    columns = list(getattr(model, 'state_columns', ()) or ['pixels'])
    scalers = {
        col: get_scaler('zscore').fit(stats.get_col_data(col))
        for col in ['action', *[c for c in columns if c != 'pixels']]
    }
    info = {}
    for col in columns:
        values = np.stack([start[col], goal[col]])
        if col == 'pixels':
            images = torch.as_tensor(values).permute(0, 3, 1, 2)
            transform = v2.Compose(
                [
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    v2.Resize(cfg.img_size),
                ]
            )
            values = transform(images)
            goal_key = 'goal'
        else:
            values = torch.as_tensor(scalers[col].transform(values)).float()
            goal_key = f'goal_{col}'
        info[col] = values[:1].unsqueeze(0)
        info[goal_key] = values[-1:].unsqueeze(0)
    info['action_history'] = torch.empty(1, 0, cfg.action_block * 2)
    return info, scalers['action']


def make_goal_environment(cfg):
    """Reuse eval_wm's expert rollout, pair sampling, and prefix replay."""
    from stable_worldmodel.envs.maniskill.pusht import PushTSWMEnv

    return PushTSWMEnv(
        env_id='OverheadPushT-v1',
        control_mode='pd_ee_delta_xy',
        camera_name='overhead_camera',
        image_size=cfg.img_size,
        sim_backend=cfg.sim_backend,
        expert_checkpoint=cfg.expert_checkpoint,
        expert_policy_type=cfg.expert_policy_type,
        expert_encoder=cfg.expert_encoder,
        expert_max_steps=cfg.expert_max_steps,
        expert_max_attempts=cfg.expert_max_attempts,
        expert_min_object_displacement=cfg.expert_min_object_displacement,
        expert_min_eef_displacement=cfg.expert_min_eef_displacement,
        goal_step_distance=cfg.goal_offset_steps,
        goal_eef_position_tolerance=0.02,
        state_goal=True,
        simulation_max_episode_steps=cfg.expert_max_steps,
    )


def prepare_expert_case(env, stats, model, cfg):
    """Read the sampled expert endpoints and exact intervening commands."""
    steps = cfg.horizon * cfg.action_block
    if min(cfg.horizon, cfg.action_block) < 1:
        raise ValueError('horizon and action_block must be positive')
    if cfg.goal_offset_steps != steps:
        raise ValueError('Expert goals require horizon * action_block steps')
    start, reset_info = env.reset(
        seed=cfg.seed,
        options={'start_from_beginning': cfg.start_from_beginning},
    )
    if not reset_info.get('expert_task_success', False):
        raise RuntimeError('Expert did not solve the native task')
    if env.expert_goal_index != steps:
        raise ValueError(
            f'Expert start/goal pair spans {env.expert_goal_index} steps; '
            f'need {steps}. Use a shorter horizon or a longer expert rollout.'
        )
    raw = torch.as_tensor(env.expert_actions[:steps]).float()
    if raw.shape != (steps, 2) or not torch.isfinite(raw).all():
        raise ValueError('Expected finite 2D expert actions for the horizon')
    if (raw.abs() > 1 + 1e-6).any():
        raise ValueError('Expert actions exceed controller bounds')
    raw = raw.reshape(cfg.horizon, cfg.action_block, 2)
    goal = {**env.goal_observation, 'pixels': env.expert_frames[steps]}
    start = {**start, 'pixels': env.expert_frames[0]}
    info, scaler = case_from_observations(start, goal, stats, model, cfg)
    preview = env.expert_frames[: steps + 1 : cfg.action_block].copy()
    pose_delta = (
        np.asarray(goal[cfg.pose_column])[:2]
        - np.asarray(start[cfg.pose_column])[:2]
    )
    metadata = {
        key: value
        for key, value in reset_info.items()
        if key.startswith('expert_') or key == 'goal_step_distance'
    }
    return info, scaler, preview, pose_delta, raw, metadata


def build_action_grid(cfg, scaler, expert_raw):
    """Sweep XY offsets around zero or the exact expert controller sequence.

    Each offset is applied evenly across a block. Clipping uses the controller
    bounds; axes are calculated from the commands actually evaluated.
    """
    if not np.isfinite(cfg.controller_scale) or cfg.controller_scale <= 0:
        raise ValueError('controller_scale must be positive and finite')
    _, offsets = xy_action_grid(cfg.samples, -cfg.grid_size, cfg.grid_size)
    mean = (
        expert_raw
        if cfg.use_expert_action_mean
        else torch.zeros_like(expert_raw)
    )
    commands = mean[None] + offsets[:, None, None] / (
        cfg.action_block * cfg.controller_scale
    )
    raw = commands.clamp(-1, 1)
    clipped = int((raw != commands).flatten(1).any(1).sum())
    candidates = scaler.transform(raw).flatten(2)
    total = raw.sum((1, 2)) * cfg.controller_scale
    coordinates = total.reshape(cfg.samples, cfg.samples, 2)
    axes = torch.stack([coordinates[0, :, 0], coordinates[:, 0, 1]])
    return axes, raw, candidates, clipped


def evaluate(cfg):
    """Load model and normalization stats, then evaluate one expert pair."""
    model = load_pretrained(cfg.policy, cache_dir=cfg.cache_dir)
    model = model.to(cfg.device).eval().requires_grad_(False)
    columns = list(getattr(model, 'state_columns', ()) or ['pixels'])
    keys = ['action', *[col for col in columns if col != 'pixels']]
    stats = swm.data.load_dataset(
        cfg.dataset_name,
        cache_dir=cfg.cache_dir,
        keys_to_load=keys,
        keys_to_cache=keys,
    )
    with closing(make_goal_environment(cfg)) as env:
        return evaluate_case(model, stats, env, cfg)


def evaluate_case(model, stats, env, cfg):
    """Score the grid and expert reference from the restored start."""
    info, scaler, preview, pose_delta, expert_raw, expert_metadata = (
        prepare_expert_case(env, stats, model, cfg)
    )
    axes, raw_candidates, candidates, clipped = build_action_grid(
        cfg, scaler, expert_raw
    )
    model = NotebookDynamics(model, normalize=cfg.normalize_reps).eval()
    objective = EndpointEnergy('l1', normalize=False)
    energy = evaluate_action_sequences(
        model, info, candidates, objective, cfg.chunk_size
    )
    expert_actions = scaler.transform(expert_raw).flatten(1)
    expert_energy = evaluate_action_sequences(
        model, info, expert_actions.unsqueeze(0), objective, cfg.chunk_size
    )[0].item()
    total_delta = (raw_candidates.sum((1, 2)) * cfg.controller_scale).numpy()
    expert_delta = (expert_raw.sum((0, 1)) * cfg.controller_scale).numpy()
    mean_raw = (
        expert_raw
        if cfg.use_expert_action_mean
        else torch.zeros_like(expert_raw)
    )
    best_idx = int(energy.argmin())
    best_raw = raw_candidates[best_idx].numpy()
    best_deltas = best_raw.sum(1) * cfg.controller_scale
    best_energy = energy[best_idx].item()
    histogram, xedges, yedges = np.histogram2d(
        total_delta[:, 0],
        total_delta[:, 1],
        weights=energy.numpy(),
        bins=cfg.samples,
    )
    heatmap = energy.reshape(cfg.samples, cfg.samples).numpy()
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    comparison = env.planning_action_comparison(
        {
            'Grid minimum': best_raw.reshape(-1, 2),
            'Expert actions': expert_raw.numpy().reshape(-1, 2),
        },
        output / 'best_action',
    )
    Image.fromarray(preview[0]).save(output / 'start.png')
    Image.fromarray(preview[-1]).save(output / 'goal.png')
    start_step = expert_metadata['expert_start_step']
    metadata = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'goal_source': 'expert',
        'expert': expert_metadata,
        'comparison_video': 'best_action/env_0.mp4',
        'comparison_results': comparison,
        'reference_action_label': 'Expert actions',
        'grid_center_total_delta': (
            mean_raw.sum((0, 1)) * cfg.controller_scale
        ).tolist(),
        'grid_center_source': 'expert_sequence'
        if cfg.use_expert_action_mean
        else 'zero',
        'grid_clipped_candidates': clipped,
        'best_grid_action': best_deltas[0].tolist(),
        'best_grid_delta_sequence': best_deltas.tolist(),
        'best_grid_energy': best_energy,
        'best_grid_total_delta': total_delta[best_idx].tolist(),
        'ground_truth_delta': pose_delta.tolist(),
        'ground_truth_action_total_delta': expert_delta.tolist(),
        'ground_truth_action_energy': expert_energy,
        'ground_truth_action_minus_grid_min': expert_energy - best_energy,
        'grid_actions_with_lower_energy': int(
            (energy < expert_energy - 1e-6).sum()
        ),
        'context_step': start_step,
        'goal_step': start_step + cfg.goal_offset_steps,
        'action_units': (
            'XY delta in meters per model prediction; '
            'plot axes sum applied controller deltas'
        ),
    }
    np.savez_compressed(
        output / 'landscape.npz',
        goal_source=np.array('expert'),
        reference_action_label=np.array('Expert actions'),
        axis=axes.numpy(),
        histogram=histogram.T,
        xedges=xedges,
        yedges=yedges,
        total_deltas=total_delta,
        ground_truth_delta=pose_delta,
        ground_truth_model_action_sequence=expert_actions.numpy(),
        ground_truth_controller_action_sequence=expert_raw.numpy(),
        ground_truth_action_total_delta=expert_delta,
        ground_truth_action_energy=expert_energy,
        grid_center_controller_action_sequence=mean_raw.numpy(),
        energy=heatmap,
        raw_actions=raw_candidates[:, 0, 0].numpy(),
        controller_action_sequences=raw_candidates.numpy(),
        model_action_sequences=candidates.numpy(),
        clip_frames=preview,
    )
    metadata_json = json.dumps(metadata, indent=2)
    (output / 'results.json').write_text(metadata_json + '\n')
    plot_landscape(
        axes.numpy(),
        heatmap,
        histogram.T,
        xedges,
        yedges,
        pose_delta,
        metadata,
        output,
        preview,
    )
    print(metadata_json)
    print(f'Saved energy landscape to {output.resolve()}')
    return metadata


def plot_landscape(
    axis,
    energy,
    histogram,
    xedges,
    yedges,
    ground_truth,
    metadata,
    output,
    preview=None,
):
    """Render measured samples with Figure 9's mesh and color presentation.

    No interpolation, smoothing, or energy rescaling is applied. The histogram
    retains the notebook's binning; the surface uses exact action coordinates.
    """
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list(
        'figure9',
        ['#e5f5ff', '#9bd2fc', '#1689fc', '#3559ee', '#8b30f2', '#ed00ed'],
    )
    # Legacy saved landscapes used one shared, zero-centered XY axis.
    axis = np.asarray(axis)
    xaxis, yaxis = (axis, axis) if axis.ndim == 1 else axis
    lower = np.array([xaxis.min(), yaxis.min()])
    upper = np.array([xaxis.max(), yaxis.max()])
    index = np.unravel_index(energy.argmin(), energy.shape)
    minimum = (xaxis[index[1]], yaxis[index[0]])
    inside = bool(np.all((ground_truth >= lower) & (ground_truth <= upper)))
    boundary = any(i in (0, len(xaxis) - 1) for i in index)
    notes = [f'{len(xaxis)} × {len(yaxis)} evaluated actions']
    if boundary:
        notes.append('minimum on sweep boundary')
    if not inside:
        notes.append('pose reference outside sweep')
    reference_action = metadata.get('ground_truth_action_total_delta')
    reference_energy = metadata.get('ground_truth_action_energy')
    recorded = reference_action is not None and reference_energy is not None
    reference_label = metadata.get('reference_action_label', 'Recorded GT')
    points = [lower, upper]
    if recorded:
        reference_action = np.asarray(reference_action)
        points.append(reference_action)
        if np.any((reference_action < lower) | (reference_action > upper)):
            notes.append('reference action outside sweep')
    bounds = np.stack(points)
    padding = (upper - lower) * 0.04
    limits = (bounds.min(0) - padding, bounds.max(0) + padding)
    note = ' · '.join(notes)
    if recorded:
        gap = reference_energy - float(energy[index])
        note += (
            f'\n{reference_label} energy {reference_energy:.4g} · '
            f'grid minimum {energy[index]:.4g} · difference {gap:+.4g}'
        )
    cfg = metadata.get('config', {})
    context_step = metadata.get('context_step', cfg.get('start_step', '?'))
    title = f'Episode {cfg.get("episode", "?")} · step {context_step}'
    if metadata.get('goal_source') == 'expert':
        expert = metadata.get('expert', {})
        title = (
            f'Expert seed {expert.get("expert_seed", cfg.get("seed", "?"))}'
            f' · steps {context_step}–{metadata.get("goal_step", "?")}'
        )
    elif metadata.get('goal_source') == 'simulator':
        title = (
            f'Simulator seed {cfg.get("seed", "?")} · '
            f'grid action {metadata.get("goal_action_index", "?")}'
        )
    with plt.rc_context(
        {
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'savefig.dpi': 250,
        }
    ):
        fig, ax = plt.subplots(figsize=(6.2, 5.6), layout='constrained')
        image = ax.imshow(
            histogram,
            origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=cmap,
            interpolation='nearest',
            aspect='equal',
        )
        ax.scatter(
            *minimum,
            marker='*',
            c='#c82333',
            s=110,
            edgecolors='white',
            linewidths=0.6,
            label=f'Grid minimum (E={energy[index]:.4g})',
        )
        if inside:
            ax.scatter(
                *ground_truth,
                marker='x',
                c='white',
                s=75,
                linewidths=2,
                label='Pose reference',
            )
        if recorded:
            ax.scatter(
                *reference_action,
                marker='D',
                c='#efb52b',
                s=65,
                edgecolors='#242424',
                linewidths=0.8,
                label=f'{reference_label} (E={reference_energy:.4g})',
                zorder=10,
            )
        ax.set(xlabel=r'$\Delta x$ (m)', ylabel=r'$\Delta y$ (m)', title=title)
        ax.set_xlim(limits[0][0], limits[1][0])
        ax.set_ylim(limits[0][1], limits[1][1])
        ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
        fig.colorbar(
            image,
            ax=ax,
            orientation='horizontal',
            pad=0.08,
            shrink=0.85,
            label='Endpoint L1 energy (weighted histogram)',
        )
        fig.supxlabel(note, fontsize=8, color='#555555')
        for suffix in ('png', 'pdf'):
            fig.savefig(output / f'landscape.{suffix}')
        plt.close(fig)

        x, y = np.meshgrid(xaxis, yaxis, indexing='xy')
        fig = plt.figure(figsize=(6.4, 6.0))
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        surface = ax.plot_surface(
            x,
            y,
            energy,
            cmap=cmap,
            rcount=len(yaxis),
            ccount=len(xaxis),
            edgecolor=(0.12, 0.24, 0.4, 0.25),
            linewidth=0.3,
            antialiased=True,
            alpha=1.0,
            norm=Normalize(vmin=float(energy.min()), vmax=float(energy.max())),
        )
        ax.scatter(
            *minimum,
            float(energy[index]),
            c='#c82333',
            marker='*',
            s=65,
            edgecolors='white',
            linewidths=0.4,
            depthshade=False,
            zorder=10,
            label=f'Grid minimum (E={energy[index]:.4g})',
        )
        if recorded:
            ax.scatter(
                *reference_action,
                reference_energy,
                marker='D',
                c='#efb52b',
                s=60,
                edgecolors='#242424',
                linewidths=0.8,
                depthshade=False,
                zorder=11,
                label=f'{reference_label} (E={reference_energy:.4g})',
            )
        ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
        ax.set(
            xlabel=r'$\Delta x$ (m)',
            ylabel=r'$\Delta y$ (m)',
            zlabel='',
            title=title,
            xlim=(limits[0][0], limits[1][0]),
            ylim=(limits[0][1], limits[1][1]),
        )
        ax.view_init(elev=25, azim=45)
        ax.set_proj_type('ortho')
        ax.set_box_aspect((1, 1, 0.65))
        for coord in (ax.xaxis, ax.yaxis, ax.zaxis):
            coord.set_major_locator(MaxNLocator(4))
            coord.set_pane_color((0.98, 0.985, 0.99, 1))
            coord._axinfo['grid'].update(
                color=(0.6, 0.65, 0.7, 0.35), linewidth=0.5
            )
        for coord in (ax.xaxis, ax.yaxis):
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, -2))
            coord.set_major_formatter(formatter)
        fig.subplots_adjust(left=0.13, right=0.87, bottom=0.20, top=0.91)
        fig.text(0.055, 0.55, 'Energy', rotation=90, va='center', fontsize=11)
        cax = fig.add_axes((0.19, 0.11, 0.62, 0.025))
        fig.colorbar(surface, cax=cax, orientation='horizontal')
        fig.text(0.5, 0.025, note, ha='center', fontsize=8, color='#555555')
        for suffix in ('png', 'pdf'):
            fig.savefig(output / f'landscape_3d.{suffix}', bbox_inches='tight')
        plt.close(fig)

        if preview is not None and len(preview):
            fig, axes = plt.subplots(
                1,
                len(preview),
                figsize=(3 * len(preview), 3),
                squeeze=False,
                layout='constrained',
            )
            for i, (ax, frame) in enumerate(zip(axes[0], preview)):
                ax.imshow(frame)
                ax.set_title(
                    'Start'
                    if i == 0
                    else 'Goal'
                    if i == len(preview) - 1
                    else f'Frame {i}'
                )
                ax.axis('off')
            fig.savefig(output / 'trajectory.png')
            plt.close(fig)
    print(note)


def replot_landscape(path, output_dir):
    """Render an existing landscape without loading any model or dataset."""
    path = Path(path)
    metadata_path = path.with_name('results.json')
    metadata = (
        json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    )
    with np.load(path) as data:
        axis = data['axis']
        for key in (
            'goal_source',
            'reference_action_label',
            'goal_action_index',
        ):
            if key in data:
                metadata[key] = data[key].item()
        energy = data['energy']
        histogram = data.get('histogram', energy)
        if 'xedges' in data:
            xedges, yedges = data['xedges'], data['yedges']
        else:
            xaxis, yaxis = (axis, axis) if axis.ndim == 1 else axis
            xedges = np.linspace(xaxis.min(), xaxis.max(), len(xaxis) + 1)
            yedges = np.linspace(yaxis.min(), yaxis.max(), len(yaxis) + 1)
        ground_truth = (
            data['ground_truth_delta']
            if 'ground_truth_delta' in data
            else np.array(metadata.get('ground_truth_delta', [np.nan, np.nan]))
        )
        if (
            'ground_truth_action_energy' in data
            and np.isfinite(data['ground_truth_action_energy']).all()
        ):
            metadata['ground_truth_action_energy'] = float(
                data['ground_truth_action_energy']
            )
            metadata['ground_truth_action_total_delta'] = data[
                'ground_truth_action_total_delta'
            ].tolist()
        preview = data.get('clip_frames')
        plot_landscape(
            axis,
            energy,
            histogram,
            xedges,
            yedges,
            ground_truth,
            metadata,
            output_dir,
            preview,
        )
    print(f'Rendered saved energy samples to {Path(output_dir).resolve()}')


@hydra.main(
    version_base=None,
    config_path='./config',
    config_name='energy_landscape_overhead',
)
def run(cfg):
    if cfg.get('landscape_file') is not None:
        replot_landscape(cfg.landscape_file, cfg.output_dir)
    else:
        evaluate(cfg)


if __name__ == '__main__':
    run()
