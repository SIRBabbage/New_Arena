"""
Tests for rotation-based predicates used by custom tasks.
"""

import math

import pytest

predicates_module = pytest.importorskip(
    'vla_arena.vla_arena.envs.predicates', exc_type=ImportError
)
get_predicate_fn = predicates_module.get_predicate_fn

bddl_base_domain_module = pytest.importorskip(
    'vla_arena.vla_arena.envs.bddl_base_domain', exc_type=ImportError
)
BDDLBaseDomain = bddl_base_domain_module.BDDLBaseDomain

try:
    from vla_arena.vla_arena.envs.object_states.base_object_states import (
        ObjectState,
    )

    OBJECT_STATE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    OBJECT_STATE_AVAILABLE = False


class DummyObjectState:
    def __init__(self):
        self.last_call = None

    def check_yaw_rotated(self, target_degrees, tolerance_degrees=15.0):
        self.last_call = (target_degrees, tolerance_degrees)
        return True


def test_yaw_rotated_predicate_is_registered():
    predicate = get_predicate_fn('yawrotated')
    dummy = DummyObjectState()

    assert predicate(dummy, 90, 25) is True
    assert dummy.last_call == (90, 25)


def test_eval_predicate_routes_yaw_rotated_numeric_literals():
    env = object.__new__(BDDLBaseDomain)
    dummy = DummyObjectState()
    env.object_states_dict = {'pen_1': dummy}

    assert env._eval_predicate(['yawrotated', 'pen_1', '90', '25']) is True
    assert dummy.last_call == (90.0, 25.0)


@pytest.mark.skipif(
    not OBJECT_STATE_AVAILABLE, reason='object state utilities not available'
)
def test_relative_yaw_from_quats_matches_90_degree_rotation():
    original_quat = [1.0, 0.0, 0.0, 0.0]
    current_quat = [
        math.cos(math.pi / 4),
        0.0,
        0.0,
        math.sin(math.pi / 4),
    ]

    yaw_delta = ObjectState.get_relative_yaw_from_quats(
        original_quat,
        current_quat,
    )

    assert yaw_delta == pytest.approx(math.pi / 2, abs=1e-6)
