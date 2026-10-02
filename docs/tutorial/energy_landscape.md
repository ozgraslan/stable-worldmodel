# OverheadPushT energy landscape

`scripts/plan/eval_energy_landscape.py` follows the
[V-JEPA2 notebook](https://github.com/facebookresearch/vjepa2/blob/main/notebooks/energy_landscape_example.ipynb)
evaluation procedure for native overhead StateLeWM and RGB LeWM checkpoints.
It uses one starting observation, a Cartesian action grid, repeated
predictions, optional layer-normalized representations, mean endpoint L1,
weighted histogram plotting, and native CEM planning. By default, the goal is
the simulator observation reached by executing one of the sampled actions.

Run from the repository root with SWM's data/training dependencies, ManiSkill,
and `matplotlib` installed:

```bash
export STABLEWM_HOME=/absolute/path/to/experiment_storage
python scripts/plan/eval_energy_landscape.py \
  --config-name=energy_landscape_state_overhead \
  policy=/absolute/path/to/weights_epoch_10.pt \
  dataset_name=/absolute/path/to/overhead_state.lance
```

For RGB, use `--config-name=energy_landscape_overhead` with an RGB checkpoint
and RGB dataset. `img_size=128` and input normalization must match training.
`device=cuda` is the default; use `device=cpu` without a GPU. State columns
and their order come from the checkpoint. Use the training normalization
dataset, or set `stats_dataset` explicitly to another normalization dataset.

## Simulator-generated goals (default)

`goal_source=simulator` resets the native `OverheadPushT-v1` simulator with
`seed`, using the overhead camera and `pd_ee_delta_xy` controller. It selects
one of the visualization grid candidates and executes its commands for
exactly `horizon * action_block` environment steps. The resulting RGB image
or named state observations become the goal for every model rollout.
The dataset supplies normalization statistics; expert trajectory rows are
not used for either the starting observation or the goal.

`goal_action_index=null` selects a grid candidate deterministically from
`seed`. Choose a specific flat index with `goal_action_index`, using
`row * samples + column` (rows vary Y, columns vary X). For example:

```bash
python scripts/plan/eval_energy_landscape.py \
  policy=/path/to/weights.pt dataset_name=/path/to/training.lance \
  goal_source=simulator samples=41 goal_action_index=1200 horizon=3
```

This executes 15 commands when `action_block=5`. The simulator grid is
centered on zero, and the selected action is already an evaluated grid point.
Both plots show a gold **Executed grid action** diamond at its predicted
energy, which matches the surface's energy at that point. This need not be
the model's minimum: the evaluation tests whether its predictions prefer
the action that actually generated the goal. The energy remains a model
prediction error, not an energy computed by the simulator.

`start.png`, `goal.png`, and `trajectory.png` show the actual simulator
observations. `results.json` records the selected index and goal source;
`landscape.npz` retains the executed commands and model inputs. The existing
`ground_truth_action_*` fields store this executed reference in simulator
mode. State checkpoints also save the corresponding RGB images.

`sim_backend=physx_cpu` is the default. RGB rendering still needs a supported
SAPIEN render device. `episode` and `start_step` apply only to dataset mode.
Simulator mode requires forward playback and a goal offset equal to
`horizon * action_block`. To retain the previous recorded-trajectory case,
set `goal_source=dataset`; the dataset then supplies the start, goal, and
recorded reference actions, and the sweep centers on their total displacement.

## Reference behavior

The energy defaults follow the example: `grid_size=0.075`, `horizon=1`
(the notebook's `action_repeat`), and `normalize_reps=true`. Encoded context
and goal are layer-normalized along the feature dimension. Each prediction
is normalized **before** it is appended to the context for the next prediction.
Only the final prediction is scored, using mean absolute latent error.
The grid defaults to `samples=41` for detailed plots; `samples=5` reproduces
the notebook's coarse example.

Toggle this extra latent normalization from the command line for either state
or RGB evaluation:

```bash
# On: match the notebook's representation normalization (default).
python scripts/plan/eval_energy_landscape.py \
  policy=/path/to/weights.pt dataset_name=/path/to/overhead.lance \
  normalize_reps=true

# Off: retain the checkpoint's native embedding scale throughout rollout.
python scripts/plan/eval_energy_landscape.py \
  policy=/path/to/weights.pt dataset_name=/path/to/overhead.lance \
  normalize_reps=false
```

This option controls context, goal, and every predicted embedding for both
the grid and CEM. Training input preprocessing and normalization layers inside
the checkpoint remain active with either setting.

In `goal_source=dataset` mode, `start_step=0` selects the first clip frame. By default, `goal_offset_steps`
is `horizon * action_block`, matching the endpoint of the evaluated action
sequence: `horizon=1 action_block=5` selects a goal five environment steps
later; `horizon=3 action_block=5` selects a goal fifteen steps later.
`history_len=1` is required to
match the notebook's single starting frame. `play_in_reverse=true` reverses
the clip observations and pose sequence, matching the example's reverse flag.
The ground-truth delta is the pose difference between the first two sampled
clip frames, from `pose_column=obs_extra_tcp_pose`, rather than a recorded
controller command. RGB output shows the sampled clip frames.

The heatmap uses `numpy.histogram2d` with energy as weights, then transposes
and displays it with the histogram edges, exactly as the notebook does.
Coordinates sum **all applied metric deltas** over the prediction sequence;
no unused future action is included. The notebook samples XYZ and plots XZ,
summing energy over Y. Overhead's controller only accepts XY, so SWM samples
and plots XY without introducing an uncontrolled third action dimension.
A red star shows the grid minimum; a white cross shows the first-frame pose
delta. The additional 3D surface requested here uses the same XY coordinates
and endpoint energy.

CEM uses the repository's `solver/cem.yaml`, shared with `eval_wm.py`:
300 samples, 30 iterations, 30 elites, and variance scale 1.0. The shared
`CEMSolver` update and sampling rules apply. The planning horizon defaults to
five model steps with five physical actions in each chunk, matching the
Overhead planning configuration. Override native options with, for example:

```bash
solver.num_samples=300 solver.n_steps=30 solver.topk=30 plan_config.horizon=5
```

`horizon` controls the repeated constant-action grid rollout independently of
`plan_config.horizon`, which controls the optimized CEM trajectory. CEM uses
full normalized action chunks; it can choose different commands within each
chunk. Output includes model-space actions, inverse-normalized controller
commands, and summed physical XY deltas for each block. CEM optimizes the same
L1 energy and representation-normalization option as the grid. Its native
candidate distribution is not restricted to the grid's `grid_size` sweep.
The first controller command is reported along with the first block delta.

## ManiSkill and checkpoint adaptations

Grid coordinates are **meters per model prediction**, not normalized controller
commands. With `action_block=5`, a sampled delta is split equally into five
physical commands. `controller_scale=0.1` maps native PandaStick controller
commands in [-1, 1] to deltas in [-0.1, 0.1] meters. Each command is then
normalized using the training action statistics and packed into the model's
10-dimensional input. Summing the five physical deltas recovers the sampled
delta exactly. Match the controller scale and frameskip to your training run.
These are commanded deltas; simulator motion may differ.

The native LeWM rollout is reused with normalization at encoding and each
prediction. SWM checkpoints use their trained encoders and image preprocessing;
they do not use V-JEPA2's duplicated video frames or its separate pose input.
The state/RGB latent layouts and action-chunk interface necessarily differ
from the reference model. This script evaluates recorded trajectories without
launching ManiSkill and does not measure simulator task success.

## Outputs

`landscape.png` contains the histogram heatmap; `trajectory.png` contains RGB
clip frames when present. `landscape_3d.png` contains the energy surface. `landscape.npz` stores endpoint
energies, histogram/edges, summed deltas, packed model actions, the pose
reference, and CEM deltas. `results.json` records configuration, grid minimum,
pose delta, native CEM plan, and CEM energy. Earlier per-command axes and recorded
sequence scoring have been replaced by the notebook conventions.

## Figure 9 rendering and grid resolution

The [paper's Figure 9](https://arxiv.org/html/2506.09985v1#S4.F9) shows a
fine mesh of evaluated XY actions. The example notebook's `samples=5` gives
only 25 evaluations and 16 surface patches, so it is a coarse diagnostic.
The default is now a finer 41 by 41 grid. Use `samples=5` for notebook
reproduction, or select the explicit Figure 9 preset:

```bash
python scripts/plan/eval_energy_landscape.py \
  --config-name=energy_landscape_figure9 \
  policy=/path/to/weights.pt dataset_name=/path/to/overhead.lance
```

This preset uses `samples=41` (1,681 actual evaluations). All rollout, goal,
action conversion, and normalization settings are inherited. Override
`normalize_reps`, `start_step`, and other experiment settings as usual.
Increasing resolution evaluates the model at more points; it does not smooth
or interpolate existing energies. The model and selected clip determine the
landscape's shape, so the paper's convex basin is not guaranteed.

Both plots use a blue-to-magenta color map, with a mesh on the 3D surface and
horizontal colorbars. PNG and PDF exports are saved. The RGB frame strip is
saved separately as `trajectory.png`, keeping plots compact. Captions flag
boundary minima and pose references outside the sweep. Plot limits expand to
keep the recorded-action marker visible outside the sweep; the evaluated
heatmap and surface remain restricted to their original range. Pose references
outside the sweep are reported in the caption.
The scoring function and measured energy values are unchanged by rendering.

Re-render a saved grid without loading a checkpoint or dataset:

```bash
python scripts/plan/eval_energy_landscape.py \
  landscape_file=/path/to/landscape.npz output_dir=/path/to/new_plots
```

Re-rendering retains the saved grid resolution. If a minimum lies on a sweep
boundary, a wider `grid_size` can test whether the basin lies outside that
range; respect the controller chunk bounds and training distribution.

## Dataset ground-truth action comparison

The following describes `goal_source=dataset`.

The XY sweep is centered on the recorded action sequence’s total commanded
displacement. `grid_size` is its half-width in meters per model step; plot
limits span the center ± `horizon * grid_size` on each axis. Constant grid
chunks use the sequence’s mean displacement per model step, so the center
shares the recorded sequence’s summed XY coordinates. With odd `samples`,
the center is an evaluated grid point. Commands must still fit the controller
bounds; reduce `grid_size` if the shifted sweep exceeds them. Reverse playback
has no recorded inverse commands and retains a zero-centered sweep.

Both plots mark the **recorded ground-truth action sequence** with a gold
diamond, alongside the red grid-minimum star. The 3D diamond's height is the
model's endpoint energy after rolling out the exact recorded action chunks.
The heatmap legend displays the same score. It is evaluated separately,
without snapping to a grid point or interpolating the surface.

The sequence spans `horizon * action_block` physical steps starting at
`start_step`, using the same context, goal, preprocessing, normalization,
and rollout length as the grid. Plot XY coordinates sum the physical deltas
of all commands. Recorded chunks can contain different commands within each
block, so their energy can differ from the constant-action surface at the
same summed XY location. The white cross remains the measured pose-delta
reference when it lies inside the sweep, which is distinct from commanded
displacement.

`results.json` reports `ground_truth_action_energy`, its difference from the
grid minimum (`ground_truth_action_minus_grid_min`), and
`grid_actions_with_lower_energy`. A negative difference means the recorded
sequence scores below every grid candidate. Arrays retain the full recorded
controller/model sequences, total delta, and exact energy.

Run the evaluation again to obtain this score. Replotting new saved arrays
preserves the marker and score, but old arrays lacking the recorded energy
cannot produce it without model inference. Reverse playback has no executed
inverse action sequence, so it retains the pose reference and omits the
recorded-action score rather than fabricating reverse controls.
