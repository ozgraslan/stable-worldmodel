from moviepy import VideoFileClip
from PIL import Image, ImageDraw
import numpy as np

PATCH_SIZE = 64
COLOR = (255, 0, 0)  # RGB red
WIDTH, HEIGHT = 128, 128
SCALE = 4  # Scale factor for the output video size
def add_grid(frame):
    image = Image.fromarray(frame[:HEIGHT, :WIDTH].copy())
    draw = ImageDraw.Draw(image)

    for x in range(0, WIDTH, PATCH_SIZE):
        for y in range(0, HEIGHT, PATCH_SIZE):
            draw.rectangle(
                (x, y, x + PATCH_SIZE, y + PATCH_SIZE),
                outline=COLOR,
                width=1,
            )

    return np.array(image)


with VideoFileClip(
    'vision_lewm_videos/run_001/episode_0.mp4',
    audio=False,
) as clip:
    grid = clip.image_transform(add_grid)
    scaled = grid.resized(new_size=(WIDTH * SCALE, HEIGHT * SCALE))  # 4× larger
    scaled.write_videofile(
        f'output_{PATCH_SIZE}.mp4',
        fps=clip.fps,
        codec='libx264',
        audio=False,
        ffmpeg_params=[
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
        ],
    )