"""Plot endpoint prediction energy for OverheadPushT state/RGB checkpoints.

Run from the repository root, for example::

    python scripts/plan/eval_energy_landscape.py \
        --config-name=energy_landscape_state_overhead \
        policy=/path/to/weights.pt dataset_name=/path/to/overhead.lance

For RGB, select ``--config-name=energy_landscape_overhead``. Set
``STABLEWM_HOME`` for the default output directory. See
``docs/tutorial/energy_landscape.md`` for settings and output descriptions.

By default, a sampled grid action is executed in ManiSkill to generate the
goal. Dataset mode remains available; no PPO expert is required.
"""

import io
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from gymnasium.spaces import Box
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import v2

import stable_worldmodel as swm
from stable_worldmodel.data.normalization import get_scaler
from stable_worldmodel.planning import ShootingCostEvaluator
from stable_worldmodel.planning.energy_landscape import (
    EndpointEnergy,
    NotebookDynamics,
    evaluate_action_sequences,
    pack_delta_sequences,
    xy_action_grid,
)
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.wm.utils import load_pretrained

if not OmegaConf.has_resolver('energy_rollout_steps'):
    OmegaConf.register_new_resolver(
        'energy_rollout_steps',
        lambda horizon, block: int(horizon) * int(block),
    )


def trajectory_rows(dataset, episode, start, history, block, horizon):
    """Select an exact, contiguous window without crossing episode boundaries.

    ``start`` is the current observation's step index. Context observations
    precede it by ``block`` steps; the goal is ``horizon * block`` later.
    """
    if min(history, block, horizon) < 1:
        raise ValueError('history, action_block and horizon must be positive')
    names = set(dataset.column_names)
    names.update(getattr(dataset, '_schema_names', ()))
    ep_key = 'episode_idx' if 'episode_idx' in names else 'ep_idx'
    episodes = dataset.get_col_data(ep_key).reshape(-1)
    steps = dataset.get_col_data('step_idx').reshape(-1)
    rows = np.flatnonzero(episodes == episode)
    mapping = {}
    for row in rows:
        step = int(steps[row])
        if step in mapping:
            raise ValueError(f'Duplicate step {step} in episode {episode}')
        mapping[step] = int(row)
    first = start - (history - 1) * block
    last = start + horizon * block
    if first < 0 or any(s not in mapping for s in range(first, last + 1)):
        raise ValueError(
            f'Episode {episode} needs contiguous steps {first}..{last}. '
            'Choose a later start or a shorter history/horizon.'
        )
    return [mapping[s] for s in range(first, last + 1)]


def rgb_frames(values):
    """Convert row-reader images (JPEG bytes or arrays) to uint8 NHWC RGB.

    Lance row reads preserve compressed JPEG blobs in an object array;
    unlike training windows, they do not decode the images automatically.
    """
    frames = []
    for value in values:
        if isinstance(value, (bytes, bytearray, memoryview)):
            with Image.open(io.BytesIO(bytes(value))) as image:
                frame = np.array(image.convert('RGB'))
        else:
            frame = np.asarray(value)
            if frame.dtype == object:
                frame = np.asarray(frame.tolist(), dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                'Expected decoded RGB frames with shape (H, W, 3)'
            )
        frames.append(frame)
    return np.stack(frames)


def prepare_case(dataset, stats, model, cfg):
    """Load a clip, starting from its first frame, as in the notebook."""
    offset = cfg.goal_offset_steps
    rows = trajectory_rows(dataset, cfg.episode, cfg.start_step, 1, 1, offset)
    sorted_rows = sorted(rows)
    inverse = np.argsort(np.argsort(rows))
    data = dataset.get_row_data(sorted_rows)
    if cfg.play_in_reverse:
        inverse = inverse[::-1].copy()
    if dataset.get_dim('action') != 2:
        raise ValueError('Overhead pd_ee_delta_xy requires 2D actions')
    data = {key: np.asarray(value)[inverse] for key, value in data.items()}
    frame_indices = list(range(0, offset + 1, cfg.action_block))
    if frame_indices[-1] != offset:
        frame_indices.append(offset)
    return case_from_observations(data, stats, model, cfg, frame_indices)


def case_from_observations(data, stats, model, cfg, frame_indices):
    """Apply the same checkpoint input processing to dataset and sim frames."""
    columns = list(getattr(model, 'state_columns', ()) or ['pixels'])
    scalers = {
        col: get_scaler('zscore').fit(stats.get_col_data(col))
        for col in ['action', *[c for c in columns if c != 'pixels']]
    }
    info = {}
    preview = None
    for col in columns:
        values = np.asarray(data[col])
        if col == 'pixels':
            values = rgb_frames(values)
            preview = values[frame_indices].copy()
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
    poses = np.asarray(data[cfg.pose_column])
    # Reference notebook poses_to_diff compares the first two clip frames.
    ground_truth = poses[frame_indices[1], :2] - poses[0, :2]
    info['action_history'] = torch.empty(1, 0, cfg.action_block * 2)
    return info, scalers['action'], preview, ground_truth


def make_goal_environment(cfg):
    """Create the existing overhead bridge without an expert policy."""
    from stable_worldmodel.envs.maniskill.pusht import PushTSWMEnv

    return PushTSWMEnv(
        env_id='OverheadPushT-v1',
        control_mode='pd_ee_delta_xy',
        camera_name='overhead_camera',
        image_size=cfg.img_size,
        sim_backend=cfg.sim_backend,
        simulation_max_episode_steps=cfg.horizon * cfg.action_block + 1,
    )


def prepare_simulator_case(stats, model, cfg, delta_grid):
    """Execute one grid candidate and use its true endpoint as the goal."""
    if cfg.play_in_reverse:
        raise ValueError('Simulator-generated goals require forward playback')
    steps = cfg.horizon * cfg.action_block
    if cfg.goal_offset_steps != steps:
        raise ValueError(
            'Simulator goals require horizon * action_block steps'
        )
    index = cfg.get('goal_action_index')
    if index is None:
        index = int(np.random.default_rng(cfg.seed).integers(len(delta_grid)))
    if not 0 <= index < len(delta_grid):
        raise ValueError('goal_action_index must index the sampled XY grid')
    raw = (
        (delta_grid[index] / (cfg.action_block * cfg.controller_scale))
        .expand(cfg.horizon, cfg.action_block, 2)
        .clone()
    )
    if (raw.abs() > 1 + 1e-6).any():
        raise ValueError('Selected grid action exceeds controller bounds')
    env = make_goal_environment(cfg)
    try:
        obs, _ = env.reset(seed=cfg.seed)
        frames = [{**obs, 'pixels': env.render()}]
        for step, action in enumerate(raw.reshape(-1, 2).numpy()):
            obs, _, _, truncated, _ = env.step(action)
            frames.append({**obs, 'pixels': env.render()})
            if truncated and step + 1 < steps:
                raise RuntimeError('Simulator ended before the goal horizon')
        data = {
            key: np.stack([frame[key] for frame in frames])
            for key in frames[0]
        }
    finally:
        env.close()
    frame_indices = list(range(0, steps + 1, cfg.action_block))
    info, scaler, preview, pose_delta = case_from_observations(
        data, stats, model, cfg, frame_indices
    )
    # Save start/goal RGB even for state checkpoints.
    if preview is None:
        preview = data['pixels'][frame_indices].copy()
    actions = scaler.transform(raw).flatten(1)
    return info, scaler, preview, pose_delta, actions, raw, index


def recorded_action_sequence(dataset, scaler, cfg):
    """Read the executed actions for the same horizon as the action grid.

    Reverse playback has no recorded inverse controls; negating the commands
    would invent an unexecuted trajectory, so no recorded score is provided.
    """
    if cfg.play_in_reverse:
        return None, None
    steps = cfg.horizon * cfg.action_block
    rows = trajectory_rows(dataset, cfg.episode, cfg.start_step, 1, 1, steps)
    sorted_rows = sorted(rows)
    inverse = np.argsort(np.argsort(rows))
    values = np.asarray(dataset.get_row_data(sorted_rows)['action'])[inverse][
        :-1
    ]
    if not np.isfinite(values).all():
        raise ValueError(
            'Recorded ground-truth sequence contains invalid actions'
        )
    raw = (
        torch.as_tensor(values)
        .float()
        .reshape(cfg.horizon, cfg.action_block, 2)
    )
    normalized = scaler.transform(raw).flatten(1)
    return normalized, raw


@torch.inference_mode()
def plan_with_cem(model, info, scaler, cfg, objective):
    """Use the shared planning solver on full trained action chunks."""
    plan = PlanConfig(**OmegaConf.to_container(cfg.plan_config, resolve=True))
    if plan.action_block != cfg.action_block:
        raise ValueError('plan_config.action_block must match action_block')
    cost = ShootingCostEvaluator(model, objective)
    solver = hydra.utils.instantiate(cfg.solver, cost=cost)
    # CEM expects the batched World action-space shape (environments, XY).
    solver.configure(
        action_space=Box(-1.0, 1.0, shape=(1, 2)), n_envs=1, config=plan
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = {k: v.to(device=device, dtype=dtype) for k, v in info.items()}
    actions = solver.solve(inputs)['actions'][0]
    raw = scaler.inverse_transform(
        actions.reshape(plan.horizon, plan.action_block, 2)
    )
    delta = raw.sum(1) * cfg.controller_scale
    energy = evaluate_action_sequences(
        model, info, actions.unsqueeze(0), objective, cfg.chunk_size
    )[0].item()
    return actions, raw, delta, energy


def evaluate(cfg):
    """Run one landscape and save 2D/3D plots, raw arrays, and metadata."""
    model = load_pretrained(cfg.policy, cache_dir=cfg.cache_dir)
    model = model.to(cfg.device).eval().requires_grad_(False)
    columns = list(getattr(model, 'state_columns', ()) or ['pixels'])
    dataset = swm.data.load_dataset(
        cfg.dataset_name,
        cache_dir=cfg.cache_dir,
        keys_to_load=list(
            dict.fromkeys(['action', *columns, cfg.pose_column])
        ),
        keys_to_cache=['action'],
    )
    stats = (
        dataset
        if cfg.stats_dataset is None
        else swm.data.load_dataset(
            cfg.stats_dataset,
            cache_dir=cfg.cache_dir,
            keys_to_load=['action', *[c for c in columns if c != 'pixels']],
        )
    )
    source = cfg.get('goal_source', 'simulator')
    axis, delta_grid = xy_action_grid(
        cfg.samples, -cfg.grid_size, cfg.grid_size
    )
    center = torch.zeros(2)
    goal_index = None
    if source == 'dataset':
        info, scaler, preview, ground_truth = prepare_case(
            dataset, stats, model, cfg
        )
        recorded_actions, recorded_raw = recorded_action_sequence(
            dataset, scaler, cfg
        )
        if recorded_raw is not None:
            center = (
                recorded_raw.sum((0, 1)) * cfg.controller_scale / cfg.horizon
            )
        delta_grid = delta_grid + center
    elif source == 'simulator':
        (
            info,
            scaler,
            preview,
            ground_truth,
            recorded_actions,
            recorded_raw,
            goal_index,
        ) = prepare_simulator_case(stats, model, cfg, delta_grid)
    else:
        raise ValueError(f'Unknown goal_source: {source!r}')
    model = NotebookDynamics(model, normalize=cfg.normalize_reps).eval()
    plot_axes = (axis[None, :] + center[:, None]) * cfg.horizon
    delta_sequences = delta_grid[:, None].expand(-1, cfg.horizon, -1)
    candidates = pack_delta_sequences(
        delta_sequences, scaler, cfg.action_block, cfg.controller_scale
    )
    # Representations are already normalized at encode and every prediction.
    objective = EndpointEnergy('l1', normalize=False)

    energy = evaluate_action_sequences(
        model, info, candidates, objective, cfg.chunk_size
    )
    recorded_energy = None
    recorded_delta = None
    if recorded_actions is not None:
        recorded_energy = evaluate_action_sequences(
            model,
            info,
            recorded_actions.unsqueeze(0),
            objective,
            cfg.chunk_size,
        )[0].item()
        recorded_delta = (
            recorded_raw.sum((0, 1)) * cfg.controller_scale
        ).numpy()
    cem_actions, cem_raw, cem_delta, cem_energy = plan_with_cem(
        model, info, scaler, cfg, objective
    )
    total_delta = delta_sequences.sum(1).numpy()
    # Match the notebook's weighted histogram, with XY replacing XZ.
    histogram, xedges, yedges = np.histogram2d(
        total_delta[:, 0],
        total_delta[:, 1],
        weights=energy.numpy(),
        bins=cfg.samples,
    )
    best_idx = int(energy.argmin())
    best_action = delta_grid[best_idx].numpy()
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    heatmap = energy.reshape(cfg.samples, cfg.samples).numpy()
    if preview is not None and len(preview):
        Image.fromarray(preview[0]).save(output / 'start.png')
        Image.fromarray(preview[-1]).save(output / 'goal.png')
    metadata = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'goal_source': source,
        'goal_action_index': goal_index,
        'reference_action_label': 'Executed grid action'
        if source == 'simulator'
        else 'Recorded GT',
        'grid_center_total_delta': (center * cfg.horizon).tolist(),
        'grid_center_source': 'zero_simulator_grid'
        if source == 'simulator'
        else (
            'recorded_action'
            if recorded_raw is not None
            else 'zero_no_recorded_reverse_action'
        ),
        'best_grid_action': best_action.tolist(),
        'best_grid_energy': energy[best_idx].item(),
        'best_grid_total_delta': total_delta[best_idx].tolist(),
        'ground_truth_delta': ground_truth.tolist(),
        'ground_truth_action_total_delta': recorded_delta.tolist()
        if recorded_delta is not None
        else None,
        'ground_truth_action_energy': recorded_energy,
        'ground_truth_action_minus_grid_min': recorded_energy
        - energy[best_idx].item()
        if recorded_energy is not None
        else None,
        'grid_actions_with_lower_energy': int(
            (energy < recorded_energy - 1e-6).sum()
        )
        if recorded_energy is not None
        else None,
        'cem_delta_sequence': cem_delta.tolist(),
        'cem_first_delta': cem_delta[0].tolist(),
        'cem_first_controller_action': cem_raw[0, 0].tolist(),
        'cem_energy': cem_energy,
        'context_step': (0 if source == 'simulator' else cfg.start_step)
        + (cfg.goal_offset_steps if cfg.play_in_reverse else 0),
        'goal_step': (0 if source == 'simulator' else cfg.start_step)
        + (0 if cfg.play_in_reverse else cfg.goal_offset_steps),
        'action_units': 'XY delta in meters per model prediction; plot axes sum applied deltas',
    }
    np.savez_compressed(
        output / 'landscape.npz',
        goal_source=np.array(source),
        goal_action_index=np.array(
            goal_index if goal_index is not None else -1
        ),
        reference_action_label=np.array(metadata['reference_action_label']),
        axis=plot_axes.numpy(),
        histogram=histogram.T,
        xedges=xedges,
        yedges=yedges,
        total_deltas=total_delta,
        ground_truth_delta=ground_truth,
        ground_truth_model_action_sequence=recorded_actions.numpy()
        if recorded_actions is not None
        else np.empty((0,)),
        ground_truth_controller_action_sequence=recorded_raw.numpy()
        if recorded_raw is not None
        else np.empty((0,)),
        ground_truth_action_total_delta=recorded_delta
        if recorded_delta is not None
        else np.empty((0,)),
        ground_truth_action_energy=recorded_energy
        if recorded_energy is not None
        else np.nan,
        cem_delta_sequence=cem_delta.numpy(),
        cem_model_action_sequence=cem_actions.numpy(),
        cem_controller_action_sequence=cem_raw.numpy(),
        energy=heatmap,
        raw_actions=(
            delta_grid / (cfg.action_block * cfg.controller_scale)
        ).numpy(),
        model_action_sequences=candidates.numpy(),
        clip_frames=preview if preview is not None else np.empty((0,)),
    )
    (output / 'results.json').write_text(json.dumps(metadata, indent=2) + '\n')
    plot_landscape(
        plot_axes.numpy(),
        heatmap,
        histogram.T,
        xedges,
        yedges,
        ground_truth,
        metadata,
        output,
        preview,
    )
    print(json.dumps(metadata, indent=2))
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
            notes.append('recorded action outside sweep')
    bounds = np.stack(points)
    padding = (upper - lower) * 0.04
    limits = (bounds.min(0) - padding, bounds.max(0) + padding)
    note = ' · '.join(notes)
    if recorded:
        gap = reference_energy - float(energy[index])
        note += f'\n{reference_label} energy {reference_energy:.4g} · grid minimum {energy[index]:.4g} · difference {gap:+.4g}'
    cfg = metadata.get('config', {})
    title = f'Episode {cfg.get("episode", "?")} · step {metadata.get("context_step", cfg.get("start_step", "?"))}'
    if metadata.get('goal_source') == 'simulator':
        title = f'Simulator seed {cfg.get("seed", "?")} · grid action {metadata.get("goal_action_index", "?")}'
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
