"""Shared helpers for the simulation runners.

`VideoRecorder` encapsulates the frame-saving + MP4-assembly logic that was
previously inlined in the runner: it manages a frames directory, writes numbered
PNG frames, and stitches them into a video with ffmpeg when the run finishes.
"""

import os
import shutil
import subprocess


class VideoRecorder:
    """Save rendered frames as numbered PNGs and assemble them into an MP4.

    The recorder is agnostic about *how* a frame is drawn: :meth:`save_frame`
    takes a callback ``save_fn(path, dpi)`` (e.g. ``simulator.save_frame``) so it
    stays decoupled from the plotting code.
    """

    def __init__(self, save_dir, video_path=None, fps=4, dpi=300):
        self.save_dir = save_dir
        self.frames_dir = os.path.join(save_dir, "frames")
        self.video_path = video_path or os.path.join(save_dir, "simulation.mp4")
        self.fps = fps
        self.dpi = dpi
        self._frame_count = 0
        self._video_done = False

    def initialize(self):
        """Prepare a clean frames directory for this run."""
        os.makedirs(self.frames_dir, exist_ok=True)
        # Remove stale frames from a previous run so the video is clean.
        for name in os.listdir(self.frames_dir):
            if name.startswith("frame_") and name.endswith(".png"):
                os.remove(os.path.join(self.frames_dir, name))
        self._frame_count = 0
        self._video_done = False
        print(f"Saving frames to: {os.path.abspath(self.frames_dir)}")

    def save_frame(self, save_fn):
        """Write the next numbered PNG frame via ``save_fn(path, dpi)``."""
        path = os.path.join(self.frames_dir, f"frame_{self._frame_count:04d}.png")
        save_fn(path, self.dpi)
        self._frame_count += 1

    def finalize(self):
        """Assemble the saved PNG frames into an MP4 with ffmpeg (runs once)."""
        if self._video_done:
            return
        self._video_done = True

        if self._frame_count == 0:
            print("No frames were saved; skipping video creation.")
            return

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print(
                "ffmpeg not found on PATH; frames are saved in "
                f"{os.path.abspath(self.frames_dir)} but no video was created."
            )
            return

        os.makedirs(os.path.dirname(os.path.abspath(self.video_path)), exist_ok=True)
        pattern = os.path.join(self.frames_dir, "frame_%04d.png")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(self.fps),
            "-i",
            pattern,
            # libx264 + yuv420p needs even dimensions; pad up if the figure is odd.
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            self.video_path,
        ]
        print(
            f"Encoding {self._frame_count} frames at {self.fps} fps -> "
            f"{os.path.abspath(self.video_path)}"
        )
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Saved video: {os.path.abspath(self.video_path)}")
        except subprocess.CalledProcessError as exc:
            print("ffmpeg failed to create the video:")
            print(exc.stderr)
