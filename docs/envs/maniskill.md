# ManiSkill PushT and state LeWM

SWM includes a ManiSkill bridge for `RandomGoalPushT-v1` and
`OverheadPushT-v1`, registered as `swm/ManiSkillPushT-v1`. Registration is lazy:
importing SWM does not import ManiSkill or initialize a simulator.

Install the optional simulator and training dependencies in your environment:

```bash
uv sync --extra train --extra maniskill
export STABLEWM_HOME=/absolute/path/to/experiment_storage
```

ManiSkill assets and a working rendering backend must be configured separately.
Run the commands below from the SWM repository root. No `mani-exps` checkout or
cross-repository `PYTHONPATH` is required.

## State-only training

`StateLeWM` uses an MLP over normalized numeric columns. The predictor and
rollout are inherited from `LeWM`; planning uses `ShootingCostEvaluator` and
`GoalMSE`. Images are not required in state-training datasets.

```bash
python scripts/train/lewm.py --config-name=state_lewm data=maniskill_state \
  data.dataset.name=/absolute/path/random_goal.lance \
  output_model_name=my_state_run subdir=my_state_run
```

Use `data=maniskill_state_overhead` for Overhead training data. The configs
select ordered `obs_*` columns matching the collector schema: RandomGoal has 33 features,
Overhead has 31. The `state` dataset column may instead contain simulator
restoration data; these configs deliberately do not use it.

For another numeric schema, use the generic `state_lewm` config and specify
`state_columns` and `state_dim`. The loaded columns default to `action` plus
`state_columns`. Configured columns take precedence over any unrelated `state` field in model inputs.
With no configured columns, the model reads `state` and its goal is
`goal_state`.

For a small CPU training smoke check use `--config-name=state_lewm_debug`
with `data=maniskill_state` and a small Lance dataset. It runs two epochs
with limited batches. The native trainer exports `weights_epoch_N.pt`; use the same `output_model_name` and `subdir`
to keep the exported model and training configuration together.

Training configs inherit the shared model and optimizer from `lewm.yaml` and
logging/output settings from `launcher/local.yaml`. ManiSkill dataset schemas
live in `scripts/train/config/data/` and are selected with `data=...`.

## Planning from environment resets

```bash
python scripts/plan/eval_wm.py --config-name=maniskill_state_pusht \
  policy=/absolute/path/my_state_run/weights_epoch_10.pt \
  eval.dataset_name=/absolute/path/random_goal.lance \
  world.expert_checkpoint=/absolute/path/ppo/final_ckpt.pt
```

Use `maniskill_state_overhead` for an Overhead state model. Vision models use
`maniskill_pusht` or `maniskill_overhead`. Train 128px Overhead vision models
with `--config-name=lewm data=maniskill_vision`; the RandomGoal
vision evaluation default is 224px and must match the model's training setup.

The bridge generates a goal by executing the PPO expert, captures its numeric
observation and image, verifies replay, then resets to the same initial state.
PPO weights and their `training_config.json` describe the expert architecture,
controller, and camera settings. Expert inference classes are included; PPO
training and collection workflows are not part of this port.

`eval.source=env` selects SWM's episodic `World.evaluate` mode. Existing
configs retain dataset-driven evaluation by default. The evaluation dataset
supplies normalization statistics; it does not supply starts or goals in
reset mode. Use the training dataset and match `plan_config.action_block` to
training `data.dataset.frameskip`.

State configs set `eval.image_keys=[]`; numeric observation/goal columns use
SWM's existing scalers. Rendering remains enabled for SWM's video output.
Vision configs transform `pixels` and `goal` as before. Set
`world.expert_output_dir` to retain task-specific expert/replay diagnostics.

For a small simulator check use `maniskill_state_debug` in the evaluation
config directory with the checkpoint and dataset overrides above. Its planner
runs on CPU, but ManiSkill still requires a working rendering backend.

## Checkpoints and compatibility

New model configs target `stable_worldmodel.wm.lewm.StateLeWM`. The encoder,
projector, action encoder and predictor parameter names are preserved.
For an older exported model with the root target `state_lewm.StateLeWM`,
SWM's existing loader can override the target explicitly:

```python
from stable_worldmodel.wm.utils import load_pretrained

model = load_pretrained(
    '/path/to/weights.pt',
    extra_args={'_target_': 'stable_worldmodel.wm.lewm.StateLeWM'},
)
```

For CLI evaluation, update the copied export's `config.json` target to the
native target. Matching weights alone does not establish normalization
compatibility: exports trained with older local preprocessing conventions
must be evaluated with their corresponding statistics/protocol.

Planning uses SWM's current future-action contract. Contexts longer than one
frame require `action_history` containing the executed blocks between frames;
the policy constructs it. The state model introduces no action padding.

## Validation

```bash
pytest tests/wm/test_state_lewm.py tests/planning tests/envs/test_maniskill*.py
RUN_MANISKILL_GPU_TESTS=1 pytest tests/envs/test_maniskill*.py
```

The state integration test trains on a numeric-only dataset and exercises
native checkpoint loading, scaling, policy, CEM, metrics and video output with
a toy environment. Simulator tests are opt-in and require assets/rendering.
