# OverheadPushT energy landscape

`scripts/plan/eval_energy_landscape.py` evaluates native overhead state and RGB
LeWM checkpoints from a start/goal pair sampled from a successful expert
trajectory. It uses the same `PushTSWMEnv` expert rollout, pair selection,
retry rules, and prefix replay as `scripts/plan/eval_wm.py`.

The dataset supplies training normalization statistics only. The script never
reads dataset trajectory rows or samples dataset goals.

Run from the repository root with SWM's data/training dependencies, ManiSkill,
and `matplotlib` installed:

```bash
export STABLEWM_HOME=/absolute/path/to/experiment_storage
python scripts/plan/eval_energy_landscape.py \
  --config-name=energy_landscape_state_overhead \
  policy=/absolute/path/to/weights.pt \
  expert_checkpoint=/absolute/path/to/expert.pt \
  dataset_name=/absolute/path/to/training.lance \
  use_expert_action_mean=true
```

For RGB, use `--config-name=energy_landscape_overhead`. State columns and their
order come from the world-model checkpoint. `dataset_name` must provide the
training action and numeric-state statistics. RGB inputs use ImageNet
normalization and `img_size=128`; these settings must match training.
`device=cuda` is the default; use `device=cpu` for model inference on CPU.
Rendering still needs a supported SAPIEN render device.

## Expert start and goal

The environment runs its deterministic expert until completion or
`expert_max_steps=100`. It accepts a trajectory only when the expert solved the
native task and the sampled pair has enough object or end-effector movement.
Failed or static cases retry using the existing reproducible seed handling,
up to `expert_max_attempts=20`.

`start_from_beginning=false` samples a start along the accepted trajectory,
leaving room for the goal offset. The goal is `horizon * action_block`
environment steps later. The environment replays the trajectory prefix to
restore the sampled start, including the controller targets, before evaluation.
Use `start_from_beginning=true` to select the trajectory's first observation.

`expert_policy_type=rgb` and `expert_encoder=spatial_softmax` match the overhead
expert defaults in `eval_wm.py`. Set `expert_policy_type=state` for a state
expert checkpoint. This setting describes the expert independently of whether
the world model consumes state or RGB observations.

The script requires the full requested start/goal separation. If a successful
expert trajectory is shorter, it raises an error rather than padding controls
or scoring against a different endpoint. Reduce `horizon` for such cases.
`goal_offset_steps` must equal `horizon * action_block`.

The context contains one starting observation. Intermediate expert frames are
saved for visualization, and the exact intervening expert actions are scored
as a separate reference sequence.

## Action-grid mean

`use_expert_action_mean=false` uses zero controller commands as the grid center.
Each grid point represents a constant XY offset repeated across all physical
steps of the evaluated sequence.

`use_expert_action_mean=true` uses the full expert action sequence between the
sampled start and goal as the grid center. Commands
can differ within a block and between blocks; the expert sequence is never
replaced by its average command.

For a sampled metric offset `delta`, each expert controller command receives:

```text
candidate[t, b] = expert[t, b] + delta / (action_block * controller_scale)
```

Offsets are sampled on an XY Cartesian grid. Defaults are `samples=15`
(225 evaluated sequences) and `grid_size=0.1` meters per model step. Use
`samples=41` for a denser surface. With odd `samples`, zero offset is an evaluated
point, and enabling the expert mean makes that point the exact expert sequence.

Grid controller commands are clipped to `[-1, 1]`, matching the environment's
controller bounds. This keeps the sweep usable when the expert already uses
saturated commands. `grid_clipped_candidates` counts affected candidates.
Plot coordinates sum the actual clipped commanded deltas; clipping can make
coordinate spacing uneven. The 2D weighted histogram can combine several
samples in one bin, whereas the 3D surface uses each evaluated sample directly.

With `action_block=5` and `controller_scale=0.1`, a 0.1-meter block offset
adds 0.2 to each physical controller command. The commands are z-score
normalized using the dataset statistics, then packed into the model's
10-dimensional action input. Commanded displacement can differ from actual
simulator motion.

## Endpoint energy

The script follows the endpoint L1 diagnostic: it encodes the starting and
goal observations, rolls out candidate sequences, and compares only the final
prediction with the goal embedding using mean absolute error.

`normalize_reps=true` layer-normalizes context, goal, and every predicted
embedding before it is fed back into subsequent predictions. Set it to
`false` to disable this additional normalization. Input z-score and RGB
preprocessing remain active. These rules apply equally to the grid and expert
reference. `horizon` controls their number of model predictions and the expert
start/goal separation. For example:

```bash
python scripts/plan/eval_energy_landscape.py \
  policy=/path/to/weights.pt expert_checkpoint=/path/to/expert.pt \
  dataset_name=/path/to/training.lance \
  horizon=5 use_expert_action_mean=true
```

This samples an expert pair 25 physical steps apart and centers the grid on
all 25 expert commands.

## Outputs and interpretation

- `landscape.png` and `landscape.pdf`: energy-weighted XY histogram.
- `landscape_3d.png` and `landscape_3d.pdf`: measured endpoint-energy surface.
- `trajectory.png`: expert frames sampled every action block, including start
  and goal. `start.png` and `goal.png` save the endpoints separately.
- `best_action/env_0.mp4`: three panels showing the minimum-energy grid action,
  replayed expert actions, and the fixed goal. Both action sequences are executed
  from the same sampled expert start using the existing environment oracle.
- `best_action/actions.npz`: controller commands for both replayed sequences.
- `landscape.npz`: energies, histogram edges, actual summed deltas, complete
  candidate controller/model sequences, expert reference, grid center,
  and sampled RGB frames.
- `results.json`: resolved configuration, accepted expert seed and trajectory
  indices, grid minimum, clipping count, and expert energy,
  and comparison replay results for both sequences.

The red star marks the grid minimum. The gold diamond marks the exact expert
sequence and its independently evaluated energy. The white cross on the 2D
plot marks the measured TCP XY displacement from the sampled start to goal.
Both action coordinates and this pose reference now cover the full horizon.
No smoothing or energy rescaling is applied.

The historical `ground_truth_action_*` fields now contain the expert sequence.
`ground_truth_action_minus_grid_min` compares its energy with the grid minimum;
`grid_actions_with_lower_energy` counts candidates the model prefers over the
expert reference. With the expert mean enabled and odd grid resolution, the
reference energy equals the center candidate's energy. It need not be the
minimum because prediction error can make the model prefer another sequence.

`controller_action_sequences` contains all physical commands for every grid
candidate. The legacy `raw_actions` array retains the first command of each
candidate. `best_grid_delta_sequence` contains each block's summed displacement;
`best_grid_action` retains the first block's displacement.

Re-render an existing result without expert execution, model inference, or
normalization data:

```bash
python scripts/plan/eval_energy_landscape.py \
  landscape_file=/path/to/landscape.npz output_dir=/path/to/new_plots
```

Legacy saved landscapes remain readable. The removed `goal_source`,
`goal_action_index`, dataset episode/start, reverse-playback, and separate
`stats_dataset` options no longer apply to new evaluations.
