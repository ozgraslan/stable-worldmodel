Refactor LeWM models and expand expert planning diagnostics

Standardize the LeWM prediction interface and extend training, expert
evaluation, energy-landscape analysis, and experiment reporting.

Model and training changes:
- Change LeWM.predict() to consume and return an info dictionary containing
  embeddings, action embeddings, and predictions; update rollout, training,
  and energy-landscape adapters to use this interface.
- Replace the rollout normalization boolean with a configurable module,
  defaulting to Identity.
- Rename StateLeWM to LeWMState and move its implementation and Hydra config
  to lewm_state.
- Add residual prediction through LeWMRes, retaining ResLeWM as a checkpoint
  target alias.
- Remove detached-rollout and augmentation training entry points, their
  configs, and legacy state-debug and vision-specific configs.
- Add optional compilation of encode/predict, explicit full-checkpoint
  resume support, and scheduler sizing for max_steps-based training.
- Log prediction and unweighted SIGReg encoder gradient norms, weighted
  relative magnitude, and gradient cosine similarity every 100 steps.
  Preserve unused-parameter alignment, handle zero norms, reduce in FP32,
  and retain the graph without modifying parameter gradient buffers.

Expert goals and evaluation:
- Retry expert trajectories that fail the native task or produce start/goal
  pairs with insufficient object and end-effector displacement.
- Add configurable retry limits and movement thresholds, reproducible retry
  seeds, and accepted-seed, attempt-count, and displacement metadata.
- Reuse isolated state replay for current-state expert queries and action
  comparisons, saving controller sequences, final goal errors, and videos.
- Avoid generating unused expert goals when each evaluation environment
  runs exactly once.
- Default expert-candidate evaluation to MPC; retain explicit open-loop mode.
- Record initial/final expert ranks and CEM mean costs, and render a bounded
  set of expert/selected/goal comparisons from matched starting states.

Energy-landscape analysis:
- Sample start/goal pairs from successful expert trajectories using the
  shared environment adapter; use datasets only for normalization statistics.
- Sweep XY offsets around zero or the exact intervening expert sequence,
  clip controller commands, and derive coordinates from applied commands.
- Require goal separation to match horizon times action_block.
- Score the expert reference independently and replay it alongside the
  minimum-energy grid sequence.
- Save complete controller/model sequences, clipping counts, comparison
  outcomes, and expert metadata; retain saved-landscape replot support.
- Update configuration defaults and documentation for the expert workflow.

Reporting and visualization:
- Add an MPPI solver configuration.
- Add a Markdown evaluation-report utility that reads the latest result
  records, compares differing settings, separates shared parameters, and
  embeds or copies success/failure videos and diagnostic visualizations.
- Include energy-landscape and expert-selection results in reports.
- Add a MoviePy utility for overlaying patch grids on evaluation video.

Tests:
- Extend coverage for expert retries, replay isolation, comparison rendering,
  expert-centered grids, normalization, result parsing, and report generation.
- Validate gradient calculations, logging cadence, mixed-precision
  reductions, and optimizer parity with diagnostics enabled and disabled.
- Update LeWM rollout tests for dictionary predictions and module-based
  normalization.