#!/usr/bin/env python3
"""Combine successful and failed planning evaluations into one video grid."""

from __future__ import annotations

import argparse
import random
import re
from functools import lru_cache
from pathlib import Path

try:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - helpful CLI error
    raise SystemExit(
        f'Missing dependency {exc.name!r}. Install imageio[ffmpeg], numpy, and pillow.'
    ) from exc


OUTCOME_PATTERN = re.compile(
    r'(?:success_|suc_)(True|False)(?=\.mp4$)', re.IGNORECASE
)
CHECKPOINT_PATTERN = re.compile(
    r'^policy:\s*.*(?:weights?_)?step[_-]?(\d+)', re.MULTILINE
)
EVAL_BUDGET_PATTERN = re.compile(r'^\s+eval_budget:\s*(\d+)', re.MULTILINE)
RECEDING_HORIZON_PATTERN = re.compile(
    r'^\s+receding_horizon:\s*(\d+)', re.MULTILINE
)
SEED_PATTERN = re.compile(r'^seed:\s*(\d+)', re.MULTILINE)


def outcome(path: Path) -> bool | None:
    """Return the outcome encoded in an evaluation-video filename."""
    match = OUTCOME_PATTERN.search(path.name)
    if match is None:
        return None
    return match.group(1).lower() == 'true'


@lru_cache(maxsize=None)
def eval_metadata(directory: Path) -> dict[str, int | None]:
    """Read checkpoint and planning settings from an eval result file."""
    metadata = {
        'checkpoint': None,
        'eval_budget': None,
        'receding_horizon': None,
        'seed': None,
    }
    for result_file in sorted(directory.glob('*_results.txt')):
        text = result_file.read_text(encoding='utf-8', errors='replace')
        for key, pattern in (
            ('checkpoint', CHECKPOINT_PATTERN),
            ('eval_budget', EVAL_BUDGET_PATTERN),
            ('receding_horizon', RECEDING_HORIZON_PATTERN),
            ('seed', SEED_PATTERN),
        ):
            match = pattern.search(text)
            if match:
                metadata[key] = int(match.group(1))
        if metadata['checkpoint'] is not None:
            break
    return metadata


def discover(
    root: Path,
    include: str | None,
    checkpoint_steps: set[int],
    eval_budgets: set[int],
    receding_horizons: set[int],
    eval_seeds: set[int],
) -> tuple[list[Path], list[Path]]:
    pattern = re.compile(include) if include else None
    successes: list[Path] = []
    failures: list[Path] = []
    for path in sorted(root.rglob('*.mp4')):
        relative = str(path.relative_to(root))
        if pattern and not pattern.search(relative):
            continue
        metadata = eval_metadata(path.parent)
        if checkpoint_steps and metadata['checkpoint'] not in checkpoint_steps:
            continue
        if eval_budgets and metadata['eval_budget'] not in eval_budgets:
            continue
        if (
            receding_horizons
            and metadata['receding_horizon'] not in receding_horizons
        ):
            continue
        if eval_seeds and metadata['seed'] not in eval_seeds:
            continue
        result = outcome(path)
        if result is True:
            successes.append(path)
        elif result is False:
            failures.append(path)
    return successes, failures


def choose(paths: list[Path], count: int, rng: random.Random) -> list[Path]:
    if len(paths) <= count:
        return paths
    return rng.sample(paths, count)


def get_font(size: int):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def fit_frame(frame: np.ndarray, width: int, height: int) -> Image.Image:
    image = Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8))
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (width, height), 'black')
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def combine(
    selected: list[tuple[Path, bool]],
    root: Path,
    output: Path,
    fps: float | None,
) -> None:
    readers = [imageio.get_reader(str(path)) for path, _ in selected]
    try:
        first_frames = [reader.get_data(0) for reader in readers]
        cell_width = max(frame.shape[1] for frame in first_frames)
        cell_height = max(frame.shape[0] for frame in first_frames)
        header_height = max(34, cell_height // 8)
        rows = max(sum(success for _, success in selected),
                   sum(not success for _, success in selected))
        canvas_width = 2 * cell_width
        canvas_height = rows * (cell_height + header_height)
        # libx264 requires even dimensions.
        canvas_width += canvas_width % 2
        canvas_height += canvas_height % 2

        metadata = readers[0].get_meta_data()
        output_fps = fps or float(metadata.get('fps', 15))
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(output), fps=output_fps, codec='libx264'
        )
        font = get_font(max(13, header_height // 3))

        iterators = [iter(reader) for reader in readers]
        current: list[np.ndarray | None] = list(first_frames)
        active = [True] * len(readers)
        # The first iterator frame duplicates get_data(0), so consume it.
        for iterator in iterators:
            next(iterator, None)

        while any(active):
            canvas = Image.new('RGB', (canvas_width, canvas_height), 'black')
            draw = ImageDraw.Draw(canvas)
            success_row = failure_row = 0

            for index, ((path, success), frame) in enumerate(
                zip(selected, current)
            ):
                row = success_row if success else failure_row
                if success:
                    success_row += 1
                else:
                    failure_row += 1
                x = 0 if success else cell_width
                y = row * (cell_height + header_height)
                color = (45, 190, 90) if success else (220, 70, 70)
                metadata = eval_metadata(path.parent)
                title = 'SUCCESS' if success else 'FAILURE'
                details = (
                    f'checkpoint={metadata["checkpoint"]}  '
                    f'budget={metadata["eval_budget"]}  '
                    f'RH={metadata["receding_horizon"]}  '
                    f'eval_seed={metadata["seed"]}'
                )
                source = str(path.relative_to(root))
                if len(source) > 72:
                    source = '...' + source[-69:]
                draw.rectangle(
                    (x, y, x + cell_width, y + header_height), fill=(25, 25, 25)
                )
                draw.text(
                    (x + 8, y + 3),
                    f'{title}  |  {details}',
                    fill=color,
                    font=font,
                )
                draw.text(
                    (x + 8, y + header_height // 2),
                    source,
                    fill=(210, 210, 210),
                    font=font,
                )
                if frame is not None:
                    fitted = fit_frame(frame, cell_width, cell_height)
                    canvas.paste(fitted, (x, y + header_height))

            writer.append_data(np.asarray(canvas))

            for index, iterator in enumerate(iterators):
                if not active[index]:
                    continue
                try:
                    current[index] = next(iterator)
                except StopIteration:
                    active[index] = False
        writer.close()
    finally:
        for reader in readers:
            reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'root', type=Path, help='Checkpoint/run directory searched recursively'
    )
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument(
        '-n', '--num-per-class', type=int, default=2,
        help='Maximum success and failure videos to include (default: 2)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Seed used to sample cases (default: 42)',
    )
    parser.add_argument(
        '--include',
        help='Only include videos whose relative path matches this regex',
    )
    parser.add_argument(
        '--checkpoint-step',
        action='append',
        type=int,
        default=[],
        help='Only include this WM checkpoint step (repeatable)',
    )
    parser.add_argument(
        '--eval-budget',
        action='append',
        type=int,
        default=[],
        help='Only include this evaluation budget (repeatable)',
    )
    parser.add_argument(
        '--receding-horizon',
        action='append',
        type=int,
        default=[],
        help='Only include this receding horizon (repeatable)',
    )
    parser.add_argument(
        '--eval-seed',
        action='append',
        type=int,
        default=[],
        help='Only include this evaluation seed (repeatable)',
    )
    parser.add_argument('--fps', type=float, help='Override output frame rate')
    args = parser.parse_args()

    if args.num_per_class < 1:
        parser.error('--num-per-class must be at least 1')
    if not args.root.is_dir():
        parser.error(f'not a directory: {args.root}')

    successes, failures = discover(
        args.root,
        args.include,
        set(args.checkpoint_step),
        set(args.eval_budget),
        set(args.receding_horizon),
        set(args.eval_seed),
    )
    if not successes or not failures:
        raise SystemExit(
            f'Need both outcomes; found {len(successes)} successes and '
            f'{len(failures)} failures under {args.root}'
        )

    rng = random.Random(args.seed)
    chosen_successes = choose(successes, args.num_per_class, rng)
    chosen_failures = choose(failures, args.num_per_class, rng)
    selected = [
        *((path, True) for path in chosen_successes),
        *((path, False) for path in chosen_failures),
    ]

    print(
        f'Found {len(successes)} successes and {len(failures)} failures; '
        f'combining {len(chosen_successes)} and {len(chosen_failures)}'
    )
    combine(selected, args.root, args.output, args.fps)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
