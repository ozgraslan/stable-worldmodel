# Training with detached autoregressive predictions

The separate `scripts/train/detached_lewm.py` entry point trains LeWM or
StateLeWM on its own predicted embeddings. Existing training and planning
functions are unchanged, and exported models use the existing planner.

For a single-frame context and a rollout horizon of three:

```text
z1 = stop_gradient(predict(stop_gradient(z0), a0))
z2 = stop_gradient(predict(z1, a1))
z3 = predict(z2, a2)
rollout_loss = MSE(z3, stop_gradient(encode(observation3)))
```

The prefix runs under `torch.no_grad()`. Only the final predictor call and
its action encoder build a graph. For longer contexts, every embedding in
the final sliding window is detached, including any original observation
embeddings still in that window. Parameters are shared across steps, so an
update changes subsequent rollouts, but there is no backpropagation through
the earlier predictions.

## Run

Run from the repository root with the training dependencies installed and
`STABLEWM_HOME` set to your experiment directory:

```bash
python -m scripts.train.detached_lewm \
  --config-name detached_state_lewm \
  data.dataset.name=/path/to/trajectories.lance \
  rollout.horizon=3 rollout.sample_horizon=false
```

The state preset uses the columns and dimensions in
`scripts/train/config/data/maniskill_state.yaml`. Override `data` or the
state columns/dimension to match other datasets. Image training uses the
`detached_lewm` config:

```bash
python -m scripts.train.detached_lewm \
  data.dataset.name=/path/to/pusht.lance \
  rollout.horizon=5
```

The default context has three frames. Use `wm.history_size=1` for exactly
the single-frame chain illustrated above. Actions are normalized recorded
action blocks, with `data.dataset.frameskip` actions per block. Match this
to the planner's `action_block`.

## Objectives and sampling

```text
loss = rollout.teacher_weight * teacher_loss
     + rollout.weight * rollout_loss
     + loss.sigreg.weight * sigreg_loss
```

Defaults retain the existing one-step teacher-forced loss and SIGReg to
train the observation encoder. The new rollout loss cannot train that
encoder because its context and targets are detached. To use only the
endpoint loss, set `rollout.teacher_weight=0 loss.sigreg.weight=0`; the
encoder then receives no learning signal. Training a fresh random encoder
this way is usually not a useful representation-learning experiment.

`rollout.sample_horizon=true` samples one depth uniformly from 1 through
`rollout.horizon` per training batch. Set it to false to supervise the same
endpoint depth every batch. There is no automatic horizon curriculum.
Validation always uses the maximum horizon and logs each step's error,
mean error, and embedding standard deviation. Only the final step's error
contributes to the rollout training objective.

Samples contain `wm.history_size + rollout.horizon` observations and stay
within an episode. Training and validation are split by episode, requiring
at least two episodes long enough to provide complete samples. Column
normalization retains the existing dataset-wide statistics behavior.

Prefix calls preserve the model's training mode: dropout stays active and
BatchNorm running statistics still update even under `no_grad()`. Validation
uses evaluation mode. The forward rollout algorithm and action alignment
match planning; model train/eval behavior can still differ.

This entry point uses AdamW with the configured constant learning rate,
ten epochs by default, and disables W&B by default. `resume_from` accepts a
Lightning checkpoint for a full training resume. Model-only exports are
saved with the existing `save_pretrained` callback and can be evaluated with
`scripts/plan/eval_wm.py`.
