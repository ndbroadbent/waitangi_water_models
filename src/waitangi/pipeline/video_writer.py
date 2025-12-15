"""Video writer using ffmpeg.

Consumes rendered frames in order and writes them to video.
Handles out-of-order frame arrival by buffering.
"""

import subprocess
import threading
from datetime import datetime
from queue import Queue

from waitangi.pipeline.data import END_OF_STREAM, RenderConfig, RenderedFrame


class VideoWriter:
    """Ordered video writer using ffmpeg.

    Receives rendered frames (possibly out of order) and writes them
    in correct sequence to ffmpeg stdin.
    """

    def __init__(
        self,
        input_queue: Queue,
        config: RenderConfig,
        output_path: str | None = None,
        log_fn=print,
    ):
        self.input_queue = input_queue
        self.config = config
        self.log = log_fn

        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"waitangi_tidal_{timestamp}.mp4"
        self.output_path = output_path

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ffmpeg_proc: subprocess.Popen | None = None

        # Buffer for out-of-order frames
        self._frame_buffer: dict[int, RenderedFrame] = {}
        self._next_frame_to_write = 0
        self._frames_written = 0

    def _start_ffmpeg(self):
        """Start the ffmpeg process."""
        self.log(f"Starting ffmpeg (output: {self.output_path})...")
        try:
            self._ffmpeg_proc = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "image2pipe",
                    "-framerate",
                    str(self.config.framerate),
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-crf",
                    str(self.config.crf),
                    self.output_path,
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return True
        except FileNotFoundError:
            self.log("Error: ffmpeg not found in PATH")
            return False

    def _write_frame(self, frame: RenderedFrame):
        """Write a single frame to ffmpeg."""
        if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
            try:
                self._ffmpeg_proc.stdin.write(frame.png_data)
                self._frames_written += 1
            except BrokenPipeError:
                self.log("Warning: ffmpeg pipe broken")

    def _flush_buffer(self):
        """Write any buffered frames that are now in sequence."""
        while self._next_frame_to_write in self._frame_buffer:
            frame = self._frame_buffer.pop(self._next_frame_to_write)
            self._write_frame(frame)
            self._next_frame_to_write += 1

    def _writer_loop(self):
        """Main writer loop - receives frames and writes in order."""
        if not self._start_ffmpeg():
            return

        while not self._stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.1)
            except Exception:
                continue

            if item is END_OF_STREAM:
                # Flush any remaining buffered frames
                self._flush_buffer()
                break

            if isinstance(item, RenderedFrame):
                if item.frame_number == self._next_frame_to_write:
                    # Frame is in order, write immediately
                    self._write_frame(item)
                    self._next_frame_to_write += 1
                    # Try to flush any buffered frames that are now in sequence
                    self._flush_buffer()
                else:
                    # Frame is out of order, buffer it
                    self._frame_buffer[item.frame_number] = item

        # Finalize ffmpeg
        if self._ffmpeg_proc:
            if self._ffmpeg_proc.stdin:
                self._ffmpeg_proc.stdin.close()
            stderr = self._ffmpeg_proc.stderr.read() if self._ffmpeg_proc.stderr else b""
            self._ffmpeg_proc.wait()

            if self._ffmpeg_proc.returncode != 0:
                self.log(f"ffmpeg error: {stderr.decode()}")
            else:
                self.log(f"Video saved to: {self.output_path}")
                self.log(f"Total frames written: {self._frames_written}")

    def start(self):
        """Start the video writer thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the video writer gracefully."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        if self._ffmpeg_proc:
            self._ffmpeg_proc.terminate()

    def join(self):
        """Wait for video writer to complete."""
        if self._thread:
            self._thread.join()

    @property
    def frames_written(self) -> int:
        """Return number of frames written so far."""
        return self._frames_written
