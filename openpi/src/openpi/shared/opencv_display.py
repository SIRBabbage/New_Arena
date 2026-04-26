from collections.abc import Mapping

import cv2
import numpy as np


def _to_hwc_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Converts a camera frame to RGB uint8 in HWC layout."""
    frame = np.asarray(image)
    if frame.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {frame.shape}.")

    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] == 4:
        frame = frame[..., :3]
    elif frame.shape[-1] != 3:
        raise ValueError(f"Expected 1, 3, or 4 channels, got shape {frame.shape}.")

    if np.issubdtype(frame.dtype, np.floating):
        finite_frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
        max_value = float(finite_frame.max()) if finite_frame.size else 0.0
        if max_value <= 1.0:
            frame = np.clip(finite_frame, 0.0, 1.0) * 255.0
        else:
            frame = np.clip(finite_frame, 0.0, 255.0)
        frame = frame.astype(np.uint8)
    else:
        frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)

    return frame


def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if target_height <= 0:
        raise ValueError(f"target_height must be positive, got {target_height}.")

    height, width = image.shape[:2]
    if height == target_height:
        return image

    scale = target_height / height
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def _draw_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (min(labeled.shape[1], 220), 28), (0, 0, 0), thickness=-1)
    cv2.putText(
        labeled,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def compose_camera_view(
    images: Mapping[str, np.ndarray],
    *,
    target_height: int = 224,
    draw_labels: bool = True,
) -> np.ndarray:
    """Creates a single RGB preview image from one or more named camera frames."""
    if not images:
        raise ValueError("images must not be empty.")

    frames: list[np.ndarray] = []
    for name, image in images.items():
        frame = _to_hwc_rgb_uint8(image)
        frame = _resize_to_height(frame, target_height)
        if draw_labels:
            frame = _draw_label(frame, name)
        frames.append(frame)

    return np.concatenate(frames, axis=1)


def show_camera_views(
    window_name: str,
    images: Mapping[str, np.ndarray],
    *,
    target_height: int = 224,
    draw_labels: bool = True,
    delay_ms: int = 1,
    quit_keys: tuple[int, ...] = (27, ord("q")),
) -> bool:
    """Displays the current camera views in a single OpenCV window.

    Returns False when the user presses one of the quit keys.
    """
    frame = compose_camera_view(images, target_height=target_height, draw_labels=draw_labels)
    cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(delay_ms) & 0xFF
    return key not in quit_keys


def close_camera_window(window_name: str) -> None:
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass
