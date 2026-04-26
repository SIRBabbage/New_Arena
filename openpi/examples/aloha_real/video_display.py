from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override

from openpi.shared import opencv_display


class VideoDisplay(_subscriber.Subscriber):
    """Displays video frames."""

    def __init__(self, *, window_name: str = "Aloha Camera Views", target_height: int = 224) -> None:
        self._window_name = window_name
        self._target_height = target_height
        self._active = True

    @override
    def on_episode_start(self) -> None:
        self._active = True

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        del action
        if not self._active:
            return

        self._active = opencv_display.show_camera_views(
            self._window_name,
            observation["images"],
            target_height=self._target_height,
        )

    @override
    def on_episode_end(self) -> None:
        opencv_display.close_camera_window(self._window_name)
