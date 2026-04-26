from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vla_arena.models.gr00t import evaluator


def _make_gr00t_root(tmp_path: Path) -> Path:
    root = tmp_path / 'Isaac-GR00T'
    script = root / 'gr00t' / 'eval' / 'run_gr00t_server.py'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('# stub\n')
    return root


def test_build_gr00t_server_command_defaults_to_uv(tmp_path: Path):
    gr00t_root = _make_gr00t_root(tmp_path)
    cfg = evaluator.GenerateConfig(
        policy_mode='server',
        model_path='../Isaac-GR00T/checkpoints/demo',
        embodiment_tag='LIBERO_PANDA',
        gr00t_root=str(gr00t_root),
    )

    cmd, cwd = evaluator._build_gr00t_server_command(cfg)

    assert cwd == gr00t_root.resolve()
    assert cmd[:4] == ['uv', 'run', 'python', 'gr00t/eval/run_gr00t_server.py']
    assert '--model-path' in cmd
    assert '--embodiment-tag' in cmd
    assert '--use-sim-policy-wrapper' in cmd


def test_build_gr00t_server_command_uses_explicit_python(tmp_path: Path):
    gr00t_root = _make_gr00t_root(tmp_path)
    python_bin = tmp_path / 'python'
    python_bin.write_text('#!/bin/sh\n')
    cfg = evaluator.GenerateConfig(
        policy_mode='server',
        model_path='checkpoint',
        embodiment_tag='LIBERO_PANDA',
        gr00t_root=str(gr00t_root),
        gr00t_python=str(python_bin),
    )

    cmd, _ = evaluator._build_gr00t_server_command(cfg)

    assert cmd[0] == str(python_bin.resolve())
    assert cmd[1].endswith('gr00t/eval/run_gr00t_server.py')


def test_initialize_policy_server_autostarts_and_normalizes_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    gr00t_root = _make_gr00t_root(tmp_path)
    cfg = evaluator.GenerateConfig(
        policy_mode='server',
        model_path='checkpoint',
        embodiment_tag='LIBERO_PANDA',
        host='0.0.0.0',
        port=5566,
        gr00t_root=str(gr00t_root),
    )

    class DummyClient:
        def __init__(self, host, port, api_token, strict):
            self.host = host
            self.port = port
            self.api_token = api_token
            self.strict = strict

    class DummyProcess:
        pid = 1234

        def poll(self):
            return None

    started: dict[str, object] = {}

    monkeypatch.setattr(
        evaluator, '_load_gr00t_policy_client_cls', lambda cfg: DummyClient
    )
    monkeypatch.setattr(
        evaluator, '_is_port_open', lambda host, port, timeout_sec=1.0: False
    )
    monkeypatch.setattr(
        evaluator,
        '_start_policy_server_process',
        lambda cmd, cwd: started.update({'cmd': cmd, 'cwd': cwd}) or DummyProcess(),
    )
    monkeypatch.setattr(
        evaluator,
        '_wait_for_policy_server_ready',
        lambda host, port, timeout_sec, poll_interval_sec, process: None,
    )

    policy, managed_process = evaluator._initialize_policy(cfg)

    assert isinstance(policy, DummyClient)
    assert policy.host == '127.0.0.1'
    assert policy.port == 5566
    assert managed_process is not None
    assert started['cwd'] == gr00t_root.resolve()


def test_initialize_policy_server_without_autostart_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    gr00t_root = _make_gr00t_root(tmp_path)
    cfg = evaluator.GenerateConfig(
        policy_mode='server',
        model_path='checkpoint',
        embodiment_tag='LIBERO_PANDA',
        auto_start_policy_server=False,
        gr00t_root=str(gr00t_root),
    )

    monkeypatch.setattr(
        evaluator, '_load_gr00t_policy_client_cls', lambda cfg: object
    )
    monkeypatch.setattr(
        evaluator, '_is_port_open', lambda host, port, timeout_sec=1.0: False
    )

    with pytest.raises(RuntimeError, match='auto_start_policy_server'):
        evaluator._initialize_policy(cfg)


def test_prepare_observation_libero_panda_keeps_two_dof_gripper():
    obs = {
        'agentview_image': np.zeros((256, 256, 3), dtype=np.uint8),
        'robot0_eye_in_hand_image': np.zeros((256, 256, 3), dtype=np.uint8),
        'robot0_eef_pos': np.array([0.1, 0.2, 0.3], dtype=np.float32),
        'robot0_eef_quat': np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        'robot0_gripper_qpos': np.array([0.01, -0.02], dtype=np.float32),
    }
    cfg = evaluator.GenerateConfig(embodiment_tag='LIBERO_PANDA')

    flat_obs, _ = evaluator._prepare_observation(obs, 'task', cfg)

    assert flat_obs['state.gripper'].shape == (1, 1, 2)


def test_generate_config_defaults_save_first_success_failure():
    cfg = evaluator.GenerateConfig()

    assert cfg.save_video_mode == 'first_success_failure'
    assert cfg.save_video_every_n_episodes == 2


def test_should_save_video_first_success_failure_triggers_on_first_episode():
    assert evaluator._should_save_video(
        mode='first_success_failure',
        episode_idx=1,
        success=False,
        first_success_saved=False,
        first_failure_saved=False,
        every_n=2,
    )
    assert evaluator._should_save_video(
        mode='first_success_failure',
        episode_idx=1,
        success=True,
        first_success_saved=False,
        first_failure_saved=False,
        every_n=2,
    )
