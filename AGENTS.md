# Agent guide

stable-worldmodel is a Python/PyTorch library for world-model research:
collecting trajectories, training models, and evaluating policies with planning.

## Repository map

- `stable_worldmodel/`: library. `world/` manages environments and rollouts;
  `envs/` and `wrapper/` provide environments and wrappers; `data/` provides
  datasets, replay buffers, normalization, and storage adapters; `wm/` contains
  baseline models; `planning/` contains objectives, evaluators, and solvers.
  `policy.py`, `protocols.py`, and `cli.py` define policies, shared interfaces,
  and the `swm` CLI.
- `scripts/train/`, `scripts/plan/`, `scripts/data/`: training, evaluation,
  and collection entry points, with adjacent Hydra `config/` directories.
  `scripts/expert/`, `scripts/benchmark/`, and `scripts/visualization/` contain
  expert training, data benchmarks, and visualization utilities.
- `tests/`: pytest tests, including `data/`, `envs/`, `planning/`, and `wm/`.
- `docs/` and `mkdocs.yaml`: tutorials, API references, and MkDocs configuration.
- `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/tests.yaml`:
  dependencies/build settings, formatting hooks, and CI checks.

## Setup and checks

Run from the repository root. Python >=3.10 is required; CI tests 3.10–3.12.
The README development setup is:

```bash
uv venv --python=3.10
source .venv/bin/activate
uv sync --extra all --group dev
```

The `all` extra includes training, environments, and data formats; LeRobot is
separate and requires Python >=3.12. CI installs with `uv sync --all-extras`.

```bash
uv run --group dev pytest                           # full CI test command
uv run --group dev pytest tests/test_protocols.py    # small import-contract smoke check
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev pre-commit run --all-files        # may modify files
uv build                                           # packaging changes
uv run --group dev mkdocs build --strict            # documentation changes
uv run swm --version                                # CLI smoke check
```

For focused changes, run pytest on the affected test file/directory first.
Full tests include environment and data integration checks; optional decoder
backends can cause skips. Linux CI installs Mesa/OpenGL system libraries and
sets `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa`; see the workflow for
exact packages. Some evaluation scripts explicitly select EGL.

## Experiments and conventions

- Follow existing repository dataset, training, checkpoint, and evaluation
  implementations for new baselines. Reuse their infrastructure and matching
  experiment settings for fair comparisons; keep algorithm-specific changes
  explicit instead of introducing parallel pipelines.

- Ruff sets 79-column lines, four-space indentation, and single quotes.
  Keep code compatible with Python 3.10. Follow nearby type annotations and
  docstrings; consult `protocols.py` for shared model/planning contracts.
- Change experiment settings through the adjacent Hydra YAML configs and
  command-line overrides. Check dataset, device, launcher, and W&B settings
  before launching a run; defaults can require a GPU and local datasets.
- `scripts/train/lewm.py` and `scripts/train/prejepa.py` are reference training
  entry points. `scripts/plan/eval_wm.py` evaluates model-based planning;
  `eval_ff.py` evaluates feed-forward policies. Evaluation needs matching
  datasets and, for learned policies, checkpoints.
- Set `STABLEWM_HOME` explicitly for experiments; the library default is
  `~/.stable_worldmodel/`, but Hydra configs reference the environment variable.
- After code changes, run focused tests and lint/format checks on changed
  files, then the full suite when dependencies are available. Report failures,
  skips, and checks not run. Run build/docs checks when relevant.

## Further reading

Start with `README.md` and `docs/quick_start.md`. Dataset collection and
training smoke-run examples are in `docs/tutorial/collect_data.md` and
`docs/tutorial/training_wm.md`; verify overrides against current YAML configs.
See `docs/guides/checkpoints.md`, `docs/tutorial/new_env.md`, `docs/cli.md`,
and `docs/api/` for specialized work. Prefer current source signatures when
examples disagree with the code.
