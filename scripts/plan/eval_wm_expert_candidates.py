"""Evaluate CEM with an expert sequence included among its candidates.

Uses the same configs, datasets, normalization, and resets as eval_wm.py.
MPC is the default: preserve receding_horizon and eval_budget and query the
expert from the current state at each replan. This requires eval.source=env.
Use +experiment.mode=open_loop to execute one horizon, including for dataset
starts with recorded expert actions. The dataset supplies normalization
statistics. The final CEM mean is scored alongside the sampled candidates;
MPC executes the native mean unless return_best is enabled. Selection is
measured,
never forced. Environment evaluations render up to four expert/selected/goal
comparison videos from matched states; use +experiment.render_comparisons=false
to disable or +experiment.max_comparisons=N to change the limit.
MPC returns the native CEM mean by default; set
+experiment.return_best=true to select the cheapest final candidate instead.
"""

import os

os.environ['MUJOCO_GL'] = 'egl'

import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing

import stable_worldmodel as swm
from stable_worldmodel.planning.solver.cem import CEMSolver

# Support both direct execution and importing through scripts.plan.
if __package__ in (None, ''):
    from eval_wm import evaluation_kwargs, get_dataset, img_transform
else:
    from .eval_wm import evaluation_kwargs, get_dataset, img_transform


class ExpertCandidateCost:
    """Inject a normalized expert sequence without changing CEM's budget."""

    def __init__(self, cost, expert, solver):
        self.cost = cost
        self.expert = expert
        self.solver = solver
        self.calls = 0
        self.records = []
        self.finals = []

    def __getattr__(self, name):
        return getattr(self.cost, name)

    def get_cost(self, info_dict, candidates):
        batch = self.calls // self.solver.n_steps
        iteration = self.calls % self.solver.n_steps
        start = batch * self.solver.batch_size
        end = start + candidates.shape[0]
        expert = self.expert[start:end].to(candidates)
        # Preserve the nominal candidate at slot zero.
        candidates[:, -1] = expert
        costs = self.cost.get_cost(dict(info_dict), candidates)
        if not torch.isfinite(costs).all():
            raise ValueError('Non-finite candidate costs in expert experiment')
        expert_cost = costs[:, -1]
        rank = 1 + (costs < expert_cost[:, None]).sum(dim=1)
        gap = expert_cost - costs[:, :-1].min(dim=1).values
        for row in range(len(expert)):
            self.records.append(
                {
                    'env_index': start + row,
                    'iteration': iteration,
                    'expert_rank': int(rank[row]),
                    'num_candidates': int(candidates.shape[1]),
                    'expert_cost': float(expert_cost[row]),
                    'native_cem_mean_cost': float(costs[row, 0]),
                    'expert_minus_best_other_cost': float(gap[row]),
                }
            )
        if iteration == self.solver.n_steps - 1:
            best_idx = costs.argmin(dim=1)
            rows = torch.arange(len(expert), device=costs.device)
            self.finals.append(
                (
                    start,
                    end,
                    dict(info_dict),
                    candidates[rows, best_idx].clone(),
                    costs[rows, best_idx].clone(),
                    expert.clone(),
                    expert_cost.clone(),
                )
            )
        self.calls += 1
        return costs


class ExpertCandidateCEM(CEMSolver):
    """Run CEM with injected experts and an optional final candidate choice."""

    def set_expert(self, expert):
        self.expert = expert
        self.diagnostics = {}

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        if self.num_samples < 2 or self.n_steps < 1:
            raise ValueError('Need num_samples >= 2 and n_steps >= 1')
        metadata = None
        provider = getattr(self, 'expert_provider', None)
        if provider is not None:
            info_dict = dict(info_dict)
            env_indices = (
                info_dict.pop('_expert_env_index').reshape(-1).tolist()
            )
            expert, metadata = provider(env_indices)
            self.set_expert(expert)
        if len(next(iter(info_dict.values()))) != len(self.expert):
            raise ValueError('Expected all evaluation environments together')
        original_cost = self.cost
        probe = ExpertCandidateCost(original_cost, self.expert, self)
        self.cost = probe
        try:
            outputs = super().solve(info_dict, init_action=init_action)
        finally:
            self.cost = original_cost
        first_records = {
            record['env_index']: record
            for record in probe.records
            if record['iteration'] == 0
        }
        final_records = {
            record['env_index']: record
            for record in probe.records
            if record['iteration'] == self.n_steps - 1
        }
        selections = []
        for (
            start,
            end,
            infos,
            best,
            best_cost,
            expert,
            expert_cost,
        ) in probe.finals:
            mean = outputs['actions'][start:end].to(expert)
            # CEM expands observations over its candidate axis; the mean
            # evaluation has only one candidate and needs matching infos.
            mean_infos = {
                key: value[:, :1]
                if isinstance(value, (torch.Tensor, np.ndarray))
                else value
                for key, value in infos.items()
            }
            mean_cost = original_cost.get_cost(mean_infos, mean[:, None])[:, 0]
            if not torch.isfinite(mean_cost).all():
                raise ValueError('Non-finite final CEM mean cost')
            # Prefer the mean on ties, keeping selection conservative.
            use_sample = best_cost < mean_cost
            best_available_cost = torch.minimum(best_cost, mean_cost)
            if getattr(self, 'return_best', True):
                chosen = torch.where(use_sample[:, None, None], best, mean)
                selected_cost = best_available_cost
            else:
                chosen = mean
                selected_cost = mean_cost
            outputs['actions'][start:end] = chosen.cpu()
            outputs['costs'][start:end] = selected_cost.cpu().tolist()
            for row in range(len(expert)):
                selections.append(
                    {
                        'env_index': start + row,
                        'initial_expert_rank': first_records[start + row][
                            'expert_rank'
                        ],
                        'last_iteration_expert_rank': final_records[
                            start + row
                        ]['expert_rank'],
                        'expert_rank': (
                            final_records[start + row]['expert_rank']
                            + int(mean_cost[row] < expert_cost[row])
                        ),
                        'num_candidates': self.num_samples + 1,
                        'expert_selected': bool(
                            torch.equal(chosen[row], expert[row])
                        ),
                        'expert_tied_for_best': bool(
                            expert_cost[row] == best_available_cost[row]
                        ),
                        'expert_cost': float(expert_cost[row]),
                        'initial_expert_cost': first_records[start + row][
                            'expert_cost'
                        ],
                        'initial_cem_mean_cost': first_records[start + row][
                            'native_cem_mean_cost'
                        ],
                        'native_cem_mean_cost': float(mean_cost[row]),
                        'selected_cost': float(selected_cost[row]),
                        'native_cem_mean_expert_rmse': float(
                            (mean[row] - expert[row]).square().mean().sqrt()
                        ),
                    }
                )
        self.diagnostics = {
            'planning_parameters': {
                'solver': {
                    'name': 'CEM',
                    'num_samples': self.num_samples,
                    'n_steps': self.n_steps,
                    'topk': self.topk,
                    'var_scale': self.var_scale,
                    'batch_size': self.batch_size,
                    'device': str(self.device),
                    'seed': self.torch_gen.initial_seed(),
                },
                'horizon': self.horizon,
                'action_block': self._config.action_block,
                'horizon_env_steps': (
                    self.horizon * self._config.action_block
                ),
                'rank_definition': '1 + number of strictly lower costs',
                'final_selection': (
                    'best final candidate or CEM mean'
                    if getattr(self, 'return_best', True)
                    else 'native CEM mean'
                ),
                'tie_break': 'prefer mean; first candidate for sample ties',
            },
            'iterations': probe.records,
            'selections': selections,
            'expert_selection_rate': float(
                np.mean([row['expert_selected'] for row in selections])
            ),
        }
        if metadata is not None:
            replan = len(self.diagnostic_history)
            self.diagnostics['replan_index'] = replan
            self.diagnostics['environment_experts'] = metadata
            for record in (*probe.records, *selections):
                local_index = record['env_index']
                record['env_index'] = env_indices[local_index]
                record['env_step'] = metadata[local_index]['env_step']
                record['replan_index'] = replan
            self.diagnostic_history.append(self.diagnostics)
        renderer = getattr(self, 'comparison_renderer', None)
        if renderer is not None:
            self.diagnostics['comparison_videos'] = renderer(
                self.expert.detach().cpu(), outputs['actions'], selections
            )
        return outputs


class ExpertComparisonRenderer:
    """Replay expert and selected controller actions in isolated scenes."""

    def __init__(self, envs, config, processor, output, max_comparisons=4):
        if (
            isinstance(max_comparisons, bool)
            or not isinstance(max_comparisons, int)
            or max_comparisons < 1
        ):
            raise ValueError(
                'experiment.max_comparisons must be a positive integer'
            )
        self.envs = envs
        self.config = config
        self.processor = processor
        self.output = Path(output)
        self.max_comparisons = max_comparisons
        self.count = 0
        for wrapped in envs.envs:
            if not callable(
                getattr(wrapped.unwrapped, 'planning_action_comparison', None)
            ):
                raise TypeError(
                    'Comparison rendering requires planning_action_comparison()'
                )

    def controller_actions(self, blocked):
        actions = (
            blocked.detach()
            .cpu()
            .float()
            .numpy()
            .reshape(self.config.plan_len, -1)
        )
        return (
            self.processor.inverse_transform(actions)
            if self.processor
            else actions
        )

    def __call__(self, expert, selected, selections):
        videos = []
        for row, selection in enumerate(selections):
            if self.count >= self.max_comparisons:
                break
            index = selection['env_index']
            env = self.envs.envs[index].unwrapped
            relative = Path('comparisons') / f'comparison_{self.count:03d}'
            results = env.planning_action_comparison(
                {
                    'expert': self.controller_actions(expert[row]),
                    'selected': self.controller_actions(selected[row]),
                },
                self.output / relative,
            )
            videos.append(
                {
                    'env_index': index,
                    'env_step': selection.get(
                        'env_step', len(getattr(env, '_executed_actions', []))
                    ),
                    'replan_index': selection.get('replan_index', 0),
                    'video': (relative / 'env_0.mp4').as_posix(),
                    'actions_file': (relative / 'actions.npz').as_posix(),
                    'outcomes': results,
                }
            )
            self.count += 1
        return videos


def expert_sequences(dataset, eval_kwargs, config, processor):
    """Load actions at the exact selected start steps and block them."""
    starts = np.asarray(eval_kwargs['start_steps'])
    chunks = dataset.load_chunk(
        np.asarray(eval_kwargs['episodes_idx']),
        starts,
        starts + config.plan_len,
    )
    return block_expert_actions(
        [chunk['action'] for chunk in chunks], config, processor
    )


def block_expert_actions(sequences, config, processor):
    """Normalize environment actions before grouping into solver blocks."""
    blocked = []
    for sequence in sequences:
        actions = np.asarray(sequence)[: config.plan_len]
        if len(actions) != config.plan_len:
            raise ValueError(
                f'Expert has {len(actions)} actions but the planning horizon '
                f'requires {config.plan_len}; reduce plan_config.horizon'
            )
        actions = actions.reshape(config.plan_len, -1)
        if not np.isfinite(actions).all():
            raise ValueError('Expert trajectory has non-finite actions')
        if processor is not None:
            actions = processor.transform(actions)
        blocked.append(actions.reshape(config.horizon, -1))
    return torch.as_tensor(np.stack(blocked), dtype=torch.float32)


def environment_expert_sequences(envs, config, processor):
    """Read expert witnesses after the same reset used for planning."""
    sequences = []
    metadata = []
    for index, wrapped in enumerate(envs.envs):
        env = wrapped.unwrapped
        actions = getattr(env, 'expert_actions', None)
        if actions is None:
            raise ValueError(
                'Environment must expose expert_actions after reset; '
                'for ManiSkill, set world.expert_checkpoint'
            )
        goal_index = getattr(env, 'expert_goal_index', None)
        if goal_index != config.plan_len:
            raise ValueError(
                f'Environment {index} expert goal is at step {goal_index}, '
                f'but the planning horizon is {config.plan_len}; the expert '
                'may have terminated early. Reduce plan_config.horizon'
            )
        report = getattr(env, 'replay_report', {})
        if not report.get('passed', False):
            raise ValueError(f'Environment {index} expert replay failed')
        sequences.append(actions)
        metadata.append(
            {
                'env_index': index,
                'seed': int(envs.seeds[index]),
                'expert_length': len(actions),
                'expert_goal_index': int(goal_index),
                'replay_report': dict(report),
            }
        )
    return block_expert_actions(sequences, config, processor), metadata


class EnvironmentExpertPolicy(swm.policy.WorldModelPolicy):
    """Bind reset-generated expert actions before the first planning call."""

    def reset(self):
        self._expert_pending = True
        self.solver.diagnostic_history = []
        for buffer in self._action_buffer or []:
            buffer.clear()
        self._next_init = None
        if self._history_buffer is not None:
            self._history_buffer.reset(list(range(self.env.num_envs)))

    def current_expert(self, env_indices):
        sequences = []
        metadata = []
        for index in env_indices:
            env = self.env.envs[index].unwrapped
            query = getattr(env, 'planning_expert_actions', None)
            if query is None:
                raise ValueError('MPC requires planning_expert_actions()')
            actions, context = query(self.cfg.plan_len)
            sequences.append(actions)
            metadata.append({'env_index': index, **context})
        return block_expert_actions(
            sequences, self.cfg, self.process.get('action')
        ), metadata

    def get_action(self, info_dict, **kwargs):
        if getattr(self, 'mpc', False):
            self.solver.expert_provider = self.current_expert
            info_dict = dict(info_dict)
            info_dict['_expert_env_index'] = np.arange(
                self.env.num_envs
            ).reshape(-1, 1)
            # Wait-mode truncations must not trigger another planning call.
            if 'truncated' in info_dict:
                info_dict['terminated'] = np.asarray(
                    info_dict.get('terminated', False), dtype=bool
                ) | np.asarray(info_dict['truncated'], dtype=bool)
        elif getattr(self, '_expert_pending', True):
            expert, metadata = environment_expert_sequences(
                self.env, self.cfg, self.process.get('action')
            )
            self.solver.set_expert(expert)
            self.expert_metadata = metadata
            self._expert_pending = False
        return super().get_action(info_dict, **kwargs)


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Evaluate expert selection from dataset starts or environment resets."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    experiment = cfg.get('experiment', {})
    mode = experiment.get('mode', 'mpc')
    if mode not in ('open_loop', 'mpc'):
        raise ValueError('experiment.mode must be open_loop or mpc')
    source = cfg.eval.get('source', 'dataset')
    render_comparisons = experiment.get('render_comparisons', source == 'env')
    if render_comparisons and source != 'env':
        raise ValueError('Comparison replay requires eval.source=env')
    if mode == 'mpc' and source != 'env':
        raise ValueError('Current-state expert MPC requires eval.source=env')
    if source not in ('dataset', 'env'):
        raise ValueError(f'Unknown evaluation source: {source!r}')
    if cfg.policy == 'random':
        raise ValueError('Set policy to a trained world-model checkpoint')
    if source == 'dataset' and cfg.eval.goal_offset_steps < (
        cfg.plan_config.horizon * cfg.plan_config.action_block
    ):
        raise ValueError('Goal offset must cover the full expert horizon')
    if mode == 'open_loop':
        cfg.plan_config.receding_horizon = cfg.plan_config.horizon
        cfg.eval.eval_budget = (
            cfg.plan_config.horizon * cfg.plan_config.action_block
        )
    if not 1 <= cfg.plan_config.receding_horizon <= cfg.plan_config.horizon:
        raise ValueError('Require 1 <= receding_horizon <= horizon')
    if source == 'env':
        # Freeze completed environments and respect the experiment budget.
        cfg.world.max_episode_steps = cfg.eval.eval_budget
        if 'goal_step_distance' not in cfg.world:
            raise ValueError('Environment must support goal_step_distance')
        if mode == 'open_loop':
            cfg.world.goal_step_distance = cfg.eval.eval_budget
    # create world environment
    if cfg.world.get('max_episode_steps') is None:
        cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    image_size = cfg.eval.get('img_size', 224)
    world = swm.World(**cfg.world, image_shape=(image_size, image_size))

    # create the transform
    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {
        key: img_transform(cfg, img_dtype)
        for key in cfg.eval.get('image_keys', ['pixels', 'goal'])
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != 'action':
            process[f'goal_{col}'] = process[col]

    # -- run evaluation
    policy = cfg.get('policy', 'random')

    if policy != 'random':
        model = swm.wm.utils.load_pretrained(cfg.policy)
        if cfg.get('bf16', False):
            model = model.to(torch.bfloat16)
        model = model.to(cfg.solver.get('device', 'cuda'))
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        if cfg.get('compile', False):
            encoder_attr = (
                'backbone' if hasattr(model, 'backbone') else 'encoder'
            )
            setattr(
                model,
                encoder_attr,
                torch.compile(getattr(model, encoder_attr)),
            )
            model.predictor = torch.compile(model.predictor)
        config = swm.PlanConfig(**cfg.plan_config)
        objective = hydra.utils.instantiate(cfg.objective)
        cost = swm.planning.ShootingCostEvaluator(model, objective)
        if hydra.utils.get_class(cfg.solver._target_) is not CEMSolver:
            raise ValueError('This experiment currently requires solver=cem')
        solver_cfg = OmegaConf.to_container(cfg.solver, resolve=True)
        solver_cfg['_target_'] = f'{__name__}.ExpertCandidateCEM'
        solver = hydra.utils.instantiate(solver_cfg, cost=cost)
        solver.return_best = experiment.get('return_best', mode != 'mpc')
        solver.diagnostic_history = []
        policy_class = (
            EnvironmentExpertPolicy
            if source == 'env'
            else swm.policy.WorldModelPolicy
        )
        policy = policy_class(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy
        ).parent
        if cfg.policy != 'random'
        else Path(__file__).parent
    )

    results_path = results_path / 'evals' / 'expert_candidates'

    if cfg.get('subdir'):
        results_path = results_path / 'evals' / cfg.subdir

    eval_kwargs = evaluation_kwargs(cfg, dataset)

    if world.num_envs != cfg.eval.num_eval:
        raise ValueError('world.num_envs must equal eval.num_eval')
    if source == 'dataset':
        expert = expert_sequences(
            dataset, eval_kwargs, config, process.get('action')
        )
        solver.set_expert(expert)
    else:
        eval_kwargs['reset_mode'] = 'wait'

    if source == 'env':
        policy.mpc = mode == 'mpc'
    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)
    print(
        f'[eval] saving videos to {results_path.resolve()} '
        '(one env_{i}.mp4 per env)'
    )

    autocast_ctx = torch.autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=cfg.get('bf16', False),
    )

    if cfg.get('compile', False):
        print('Warming up compiled model...')
        warmup_autocast_ctx = torch.autocast(
            device_type='cuda',
            dtype=torch.bfloat16,
            enabled=cfg.get('bf16', False),
        )
        with warmup_autocast_ctx:
            warmup_kwargs = dict(eval_kwargs)
            if 'episodes' in warmup_kwargs:
                warmup_kwargs['episodes'] = world.num_envs
            world.evaluate(**warmup_kwargs, video=results_path)
        print('Warmup done.')

    if render_comparisons:
        solver.comparison_renderer = ExpertComparisonRenderer(
            world.envs,
            config,
            process.get('action'),
            results_path,
            max_comparisons=experiment.get('max_comparisons', 4),
        )
    start_time = time.time()
    with autocast_ctx:
        metrics = world.evaluate(**eval_kwargs, video=results_path)
    end_time = time.time()

    world.close()
    print(metrics)
    diagnostics = dict(solver.diagnostics)
    if mode == 'mpc':
        replans = solver.diagnostic_history
        diagnostics['replans'] = replans
        diagnostics['iterations'] = [
            row for replan in replans for row in replan['iterations']
        ]
        diagnostics['selections'] = [
            row for replan in replans for row in replan['selections']
        ]
        diagnostics['expert_selection_rate'] = float(
            np.mean(
                [row['expert_selected'] for row in diagnostics['selections']]
            )
        )
        diagnostics['num_replans'] = len(replans)
        diagnostics['comparison_videos'] = [
            video
            for replan in replans
            for video in replan.get('comparison_videos', [])
        ]
    diagnostics['mode'] = mode
    diagnostics['planning_parameters']['plan_config'] = OmegaConf.to_container(
        cfg.plan_config, resolve=True
    )
    diagnostics['source'] = source
    if source == 'dataset':
        diagnostics['episodes_idx'] = eval_kwargs['episodes_idx']
        diagnostics['start_steps'] = eval_kwargs['start_steps']
    elif mode != 'mpc':
        diagnostics['environment_experts'] = policy.expert_metadata
    diagnostics['config'] = OmegaConf.to_container(cfg, resolve=True)
    (results_path / 'expert_candidates.json').write_text(
        json.dumps(diagnostics, indent=2) + '\n'
    )
    print(f'Expert selection rate: {diagnostics["expert_selection_rate"]:.3f}')
    print(f'[eval] videos saved to {results_path.resolve()}')

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open('a') as f:
        f.write('\n')  # separate from previous runs

        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n')

        f.write('==== RESULTS ====\n')
        f.write(f'metrics: {metrics}\n')
        f.write(f'evaluation_time: {end_time - start_time} seconds\n')


if __name__ == '__main__':
    run()
