"""
Tests for lighting perturbation helpers.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest


arena_module = pytest.importorskip(
    'vla_arena.vla_arena.envs.arenas.arena', exc_type=ImportError
)
Arena = arena_module.Arena


def test_light_color_multiplier_changes_rgb_channels():
    light = ET.fromstring(
        '<light diffuse=".8 .8 .8" specular="0.3 0.3 0.3" '
        'ambient="0.1 0.1 0.1"/>'
    )

    Arena._apply_light_color_multipliers_to_element(
        light, np.array([1.25, 0.85, 0.55])
    )

    diffuse = np.fromstring(light.get('diffuse'), sep=' ')
    specular = np.fromstring(light.get('specular'), sep=' ')
    ambient = np.fromstring(light.get('ambient'), sep=' ')

    assert np.allclose(diffuse, [1.0, 0.68, 0.44])
    assert np.allclose(specular, [0.375, 0.255, 0.165])
    assert np.allclose(ambient, [0.125, 0.085, 0.055])
    assert not np.allclose(diffuse[0], diffuse[1])
    assert not np.allclose(diffuse[1], diffuse[2])


def test_light_color_tint_sets_saturated_light_color():
    light = ET.fromstring(
        '<light diffuse=".8 .8 .8" specular="0.3 0.3 0.3"/>'
    )

    Arena._apply_light_color_tint_to_element(light, np.array([1.0, 0.04, 0.03]))

    diffuse = np.fromstring(light.get('diffuse'), sep=' ')
    specular = np.fromstring(light.get('specular'), sep=' ')
    ambient = np.fromstring(light.get('ambient'), sep=' ')

    assert np.allclose(diffuse, [1.0, 0.04, 0.03])
    assert np.allclose(specular, [0.6, 0.024, 0.018])
    assert np.allclose(ambient, [0.25, 0.01, 0.0075])
    assert diffuse[0] > 20 * diffuse[1]
