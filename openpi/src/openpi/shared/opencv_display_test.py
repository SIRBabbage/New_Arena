import numpy as np

from openpi.shared import opencv_display


def test_to_hwc_rgb_uint8_converts_chw_float_image():
    image = np.array(
        [
            [[0.0, 0.5], [1.0, 0.25]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.9, 0.8], [0.7, 0.6]],
        ],
        dtype=np.float32,
    )

    converted = opencv_display._to_hwc_rgb_uint8(image)

    assert converted.shape == (2, 2, 3)
    assert converted.dtype == np.uint8
    np.testing.assert_array_equal(converted[0, 0], np.array([0, 25, 229], dtype=np.uint8))
    np.testing.assert_array_equal(converted[1, 0], np.array([255, 76, 178], dtype=np.uint8))


def test_compose_camera_view_keeps_camera_order():
    left = np.zeros((8, 10, 3), dtype=np.uint8)
    right = np.full((8, 6, 3), 127, dtype=np.uint8)

    preview = opencv_display.compose_camera_view(
        {
            "left": left,
            "right": right,
        },
        target_height=8,
        draw_labels=False,
    )

    assert preview.shape == (8, 16, 3)
    np.testing.assert_array_equal(preview[:, :10], left)
    np.testing.assert_array_equal(preview[:, 10:], right)
