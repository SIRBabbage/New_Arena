from __future__ import annotations

import json
import logging
import math
import os
import pathlib
import random
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import imageio
import numpy as np
import tqdm
import yaml

from vla_arena.vla_arena import benchmark, get_vla_arena_path
from vla_arena.vla_arena.envs import OffScreenRenderEnv
from vla_arena.vla_arena.utils.eval_init_state import select_init_state_index
from vla_arena.vla_arena.utils.utils import (
    apply_instruction_replacement,
    load_replacements_dict,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

DATE_TIME = time.strftime('%Y_%m_%d-%H_%M_%S')
DATE = time.strftime('%Y_%m_%d')

_SUITE_SPEC_SEPARATOR = '+'
_PERTURBATION_ALIAS_MAP: dict[str, dict[str, Any]] = {
    'light': {'adjust_light': True},
    'light_color': {'adjust_light': True},
    'adjust_light': {'adjust_light': True},
    'lighting': {'adjust_light': True},
    'noise': {'add_noise': True},
    'gaussian_noise': {'add_noise': True},
    'blur': {'blur': True},
    'gaussian_blur': {'blur': True},
    'color': {'randomize_color': True},
    'randomize_color': {'randomize_color': True},
    'color_randomize': {'randomize_color': True},
    'camera': {'camera_offset': True},
    'camera_offset': {'camera_offset': True},
    'layout': {
        'layout_perturbation': True,
        'init_state_selection_mode': 'episode_idx',
        'init_state_offset': 0,
        'init_state_offset_random': False,
    },
    'layout_random': {
        'layout_random': True,
    },
    'random_layout': {
        'layout_random': True,
    },
    'lang1': {'use_replacements': True, 'replacement_level': 1},
    'lang2': {'use_replacements': True, 'replacement_level': 2},
    'lang3': {'use_replacements': True, 'replacement_level': 3},
    'lang4': {'use_replacements': True, 'replacement_level': 4},
    'language1': {'use_replacements': True, 'replacement_level': 1},
    'language2': {'use_replacements': True, 'replacement_level': 2},
    'language3': {'use_replacements': True, 'replacement_level': 3},
    'language4': {'use_replacements': True, 'replacement_level': 4},
    'language_l1': {'use_replacements': True, 'replacement_level': 1},
    'language_l2': {'use_replacements': True, 'replacement_level': 2},
    'language_l3': {'use_replacements': True, 'replacement_level': 3},
    'language_l4': {'use_replacements': True, 'replacement_level': 4},
    'replacement1': {'use_replacements': True, 'replacement_level': 1},
    'replacement2': {'use_replacements': True, 'replacement_level': 2},
    'replacement3': {'use_replacements': True, 'replacement_level': 3},
    'replacement4': {'use_replacements': True, 'replacement_level': 4},
}

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DEFAULT_GR00T_ROOT = _REPO_ROOT / 'Isaac-GR00T'


@dataclass
class GenerateConfig:
    #################################################################################################################
    # GR00T policy parameters
    #################################################################################################################
    policy_mode: str = 'local'  # "local" | "server"
    model_path: str = 'nvidia/GR00T-N1.6-DROID'
    embodiment_tag: str = 'OXE_DROID'  # "OXE_DROID" | "LIBERO_PANDA"
    use_sim_policy_wrapper: bool = True
    strict: bool = True
    device: str = 'cuda'

    # Server mode options
    host: str = '127.0.0.1'
    port: int = 5555
    api_token: str | None = None
    auto_start_policy_server: bool = True
    policy_server_start_timeout_sec: int = 180
    policy_server_poll_interval_sec: float = 1.0
    gr00t_root: str | None = None
    gr00t_python: str | None = None

    # Inference behavior
    open_loop_horizon: int = 8
    image_resize_height: int | None = 180
    image_resize_width: int | None = 320
    controller_mode: str = 'auto'  # "auto" | "osc_pose" | "joint_position"
    joint_delta_clip: float | None = 0.05
    gripper_threshold: float = 0.5
    normalize_gripper_to_env: bool = True
    invert_gripper_for_env: bool = True
    binarize_gripper: bool = True

    #################################################################################################################
    # VLA-Arena environment-specific parameters
    #################################################################################################################
    task_suite_name: str | list[str] = 'safety_static_obstacles'
    task_level: int = 0
    task_name: str | None = None
    task_names: list[str] | None = None
    num_steps_wait: int = 10
    num_trials_per_task: int = 10
    add_noise: bool = False
    adjust_light: bool = False
    randomize_color: bool = False
    camera_offset: bool = False
    blur: bool = False
    safety: bool = False
    layout_perturbation: bool = False
    layout_random: bool = False
    unseen_object_perturbation: bool = False
    unseen_object_category: str | None = None
    init_state_selection_mode: str = 'first'
    init_state_offset: int = 0
    init_state_offset_random: bool = False

    #################################################################################################################
    # Utils
    #################################################################################################################
    use_local_log: bool = True
    local_log_dir: str = './experiments/logs'
    run_id_note: str | None = None
    use_wandb: bool = False
    wandb_entity: str = 'your-wandb-entity'
    wandb_project: str = 'your-wandb-project'
    save_video_mode: str = (
        'first_success_failure'
    )  # "all" | "first_success_failure" | "interval" | "none"
    save_video_every_n_episodes: int = 2
    seed: int = 7
    result_json_path: str | None = None

    #################################################################################################################
    # Language perturbation parameters
    #################################################################################################################
    replacements_file: str = 'VLA-Arena/language_replacements'
    use_replacements: bool = False
    replacement_probability: float = 1.0
    replacement_level: int = 1


@dataclass(frozen=True)
class _ResolvedSuiteSpec:
    requested_name: str
    benchmark_name: str
    display_name: str
    cfg_overrides: dict[str, Any]


def _normalize_host(host: str) -> str:
    normalized = str(host).strip()
    if normalized in {'', '0.0.0.0', '::'}:
        return '127.0.0.1'
    return normalized


def _is_local_host(host: str) -> bool:
    normalized = str(host).strip().lower()
    return normalized in {'', 'localhost', '127.0.0.1', '0.0.0.0', '::1', '::'}


def _is_port_open(host: str, port: int, timeout_sec: float = 1.0) -> bool:
    try:
        with socket.create_connection(
            (_normalize_host(host), int(port)), timeout=max(0.05, float(timeout_sec))
        ):
            return True
    except OSError:
        return False


def _resolve_gr00t_root(cfg: GenerateConfig) -> Path:
    candidate = Path(cfg.gr00t_root).expanduser() if cfg.gr00t_root else _DEFAULT_GR00T_ROOT
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            'Unable to locate GR00T repo root. '
            f'Checked: {candidate}. Set gr00t_root in gr00t.yaml.'
        )
    server_script = candidate / 'gr00t' / 'eval' / 'run_gr00t_server.py'
    if not server_script.exists():
        raise FileNotFoundError(
            f'Unable to find GR00T server entrypoint at {server_script}.'
        )
    return candidate


def _ensure_gr00t_import_root(gr00t_root: Path) -> None:
    gr00t_root_str = str(gr00t_root)
    if gr00t_root_str not in sys.path:
        sys.path.insert(0, gr00t_root_str)


def _build_gr00t_server_command(
    cfg: GenerateConfig,
) -> tuple[list[str], Path]:
    gr00t_root = _resolve_gr00t_root(cfg)
    server_script = gr00t_root / 'gr00t' / 'eval' / 'run_gr00t_server.py'

    if cfg.gr00t_python:
        cmd = [str(Path(cfg.gr00t_python).expanduser().resolve()), str(server_script)]
    else:
        cmd = ['uv', 'run', 'python', 'gr00t/eval/run_gr00t_server.py']

    cmd.extend(
        [
            '--model-path',
            str(cfg.model_path),
            '--embodiment-tag',
            str(cfg.embodiment_tag),
            '--device',
            str(cfg.device),
            '--host',
            str(cfg.host),
            '--port',
            str(cfg.port),
        ]
    )
    if cfg.strict:
        cmd.append('--strict')
    if cfg.use_sim_policy_wrapper:
        cmd.append('--use-sim-policy-wrapper')
    return cmd, gr00t_root


def _start_policy_server_process(
    cmd: list[str],
    *,
    cwd: Path,
) -> subprocess.Popen[bytes]:
    logger.info('Auto-starting GR00T policy server: %s', shlex.join(cmd))
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        start_new_session=True,
        env=os.environ.copy(),
    )


def _wait_for_policy_server_ready(
    host: str,
    port: int,
    timeout_sec: float,
    poll_interval_sec: float,
    process: subprocess.Popen[bytes],
) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                'Auto-started GR00T policy server exited early with code '
                f'{process.returncode}.'
            )
        if _is_port_open(host, port, timeout_sec=max(0.05, poll_interval_sec)):
            return
        time.sleep(max(0.05, poll_interval_sec))

    raise TimeoutError(
        'Timed out waiting for GR00T policy server to become ready at '
        f'{host}:{port} after {timeout_sec}s.'
    )


def _stop_managed_policy_server(
    process: subprocess.Popen[bytes] | None,
    timeout_sec: float = 10.0,
) -> None:
    if process is None or process.poll() is not None:
        return

    logger.info(
        'Stopping auto-started GR00T policy server (pid=%s)...', process.pid
    )
    process.terminate()
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        logger.warning(
            'GR00T policy server did not stop within %.1fs; killing process.',
            timeout_sec,
        )
        process.kill()
        process.wait(timeout=5)


def _load_gr00t_local_policy_classes(cfg: GenerateConfig):
    _ensure_gr00t_import_root(_resolve_gr00t_root(cfg))
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    return EmbodimentTag, Gr00tPolicy, Gr00tSimPolicyWrapper


def _load_gr00t_policy_client_cls(cfg: GenerateConfig):
    _ensure_gr00t_import_root(_resolve_gr00t_root(cfg))
    from gr00t.policy.server_client import PolicyClient

    return PolicyClient


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1e-12, 1.0 - quat[3] * quat[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image

    if image.shape[:2] == (height, width):
        return image

    pil_image = Image.fromarray(image)
    cur_width, cur_height = pil_image.size
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = pil_image.resize((resized_width, resized_height), resample=Image.BILINEAR)
    canvas = Image.new(resized.mode, (width, height), 0)
    pad_height = max(0, (height - resized_height) // 2)
    pad_width = max(0, (width - resized_width) // 2)
    canvas.paste(resized, (pad_width, pad_height))
    return np.asarray(canvas, dtype=np.uint8)


def _suite_category(suite_name: str) -> tuple[str, bool]:
    if suite_name.startswith('safety_'):
        return 'Safety', True
    if suite_name.startswith('distractor_'):
        return 'Distractor', False
    if suite_name.startswith('extrapolation_'):
        return 'Extrapolation', False
    if suite_name == 'long_horizon':
        return 'Long Horizon', False
    return 'Other', False


def _normalize_task_selector(text: str) -> str:
    return ' '.join(str(text).strip().lower().replace('_', ' ').split())


def _resolve_suite_spec(raw_suite_name: str) -> _ResolvedSuiteSpec:
    raw_name = str(raw_suite_name).strip()
    if raw_name.lower() == 'all' or _SUITE_SPEC_SEPARATOR not in raw_name:
        return _ResolvedSuiteSpec(
            requested_name=raw_name,
            benchmark_name=raw_name,
            display_name=raw_name,
            cfg_overrides={},
        )

    parts = [part.strip() for part in raw_name.split(_SUITE_SPEC_SEPARATOR)]
    parts = [part for part in parts if part]
    if len(parts) < 2:
        return _ResolvedSuiteSpec(
            requested_name=raw_name,
            benchmark_name=raw_name,
            display_name=raw_name,
            cfg_overrides={},
        )

    benchmark_name, modifiers = parts[0], parts[1:]
    cfg_overrides: dict[str, Any] = {}
    normalized_modifiers: list[str] = []

    for modifier in modifiers:
        modifier_key = modifier.lower()
        if modifier_key not in _PERTURBATION_ALIAS_MAP:
            supported = ', '.join(sorted(_PERTURBATION_ALIAS_MAP))
            raise ValueError(
                f'Unknown task_suite_name perturbation alias: {modifier!r}. '
                f'Use "<suite>+<alias>", where <alias> is one of: {supported}'
            )
        normalized_modifiers.append(modifier_key)
        cfg_overrides.update(_PERTURBATION_ALIAS_MAP[modifier_key])

    return _ResolvedSuiteSpec(
        requested_name=raw_name,
        benchmark_name=benchmark_name,
        display_name=f'{benchmark_name}+{"+".join(normalized_modifiers)}',
        cfg_overrides=cfg_overrides,
    )


def _prepare_layout_perturbation_states(
    initial_states,
) -> tuple[list[Any], list[tuple[int, float]]]:
    states = list(initial_states or [])
    if len(states) <= 1:
        return states, []

    baseline = np.asarray(states[0])
    scored_states: list[tuple[float, int, Any]] = []
    for idx, state in enumerate(states[1:], start=1):
        score = float(np.linalg.norm(np.asarray(state) - baseline))
        scored_states.append((score, idx, state))

    scored_states.sort(key=lambda item: (item[0], item[1]), reverse=True)
    reordered_states = [state for _, _, state in scored_states]
    layout_scores = [(idx, score) for score, idx, _ in scored_states]
    return reordered_states, layout_scores


def _load_initial_states(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    task_level: int,
    log_file=None,
):
    if cfg.unseen_object_perturbation:
        raise NotImplementedError(
            'GR00T evaluator does not yet support +unseen task rewriting. '
            'Standard suites and common perturbation aliases such as +blur, '
            '+noise, +camera, +layout, and +layout_random are supported.'
        )

    if cfg.layout_random:
        _log_message(
            'Using layout_random | skipping fixed init states and relying on env.reset() '
            'randomized placements',
            log_file,
        )
        return [None]

    initial_states = task_suite.get_task_init_states(task_level, task_id)
    if cfg.layout_perturbation:
        initial_states, layout_scores = _prepare_layout_perturbation_states(
            initial_states
        )
        if layout_scores:
            preview = ', '.join(
                f'#{idx}:{score:.4f}'
                for idx, score in layout_scores[: min(5, len(layout_scores))]
            )
            _log_message(
                'Using layout perturbation | ranked non-default init states by '
                f'distance from baseline: {preview}',
                log_file,
            )
    return initial_states


def _resolve_selected_tasks(
    task_suite,
    task_level: int,
    task_name: str | None = None,
    task_names: list[str] | None = None,
) -> list[tuple[int, Any]]:
    selectors: list[str] = []
    if task_name:
        selectors.append(task_name)
    if task_names:
        selectors.extend(task_names)

    level_tasks = task_suite.get_all_tasks_by_level(task_level)
    indexed_tasks = list(enumerate(level_tasks))
    if not selectors:
        return indexed_tasks

    resolved: list[tuple[int, Any]] = []
    seen_ids: set[int] = set()
    for selector in selectors:
        normalized_selector = _normalize_task_selector(selector)
        matches = [
            (task_id, task)
            for task_id, task in indexed_tasks
            if _normalize_task_selector(task.name) == normalized_selector
            or _normalize_task_selector(task.language) == normalized_selector
        ]
        if not matches:
            available = ', '.join(task.name for _, task in indexed_tasks)
            raise ValueError(
                f'Unable to resolve task selector {selector!r}. Available tasks: {available}'
            )
        for task_id, task in matches:
            if task_id not in seen_ids:
                resolved.append((task_id, task))
                seen_ids.add(task_id)
    return resolved


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _build_controller_configs(cfg: GenerateConfig) -> dict[str, Any] | None:
    controller_mode = cfg.controller_mode.lower()
    if controller_mode == 'auto':
        controller_mode = (
            'joint_position'
            if cfg.embodiment_tag.upper() == 'OXE_DROID'
            else 'osc_pose'
        )

    if controller_mode == 'osc_pose':
        return None

    if controller_mode != 'joint_position':
        raise ValueError(
            f'Unsupported controller_mode={cfg.controller_mode!r}. '
            'Expected "auto", "osc_pose", or "joint_position".'
        )

    import robosuite as suite
    from robosuite.controllers import load_part_controller_config

    controller_configs = suite.load_composite_controller_config(robot='Panda')
    controller_configs['body_parts']['right'] = load_part_controller_config(
        default_controller='JOINT_POSITION'
    )
    controller_configs['body_parts']['right']['gripper'] = {'type': 'GRIP'}
    return controller_configs


def _make_env(task, cfg: GenerateConfig):
    task_bddl_file = os.path.join(
        get_vla_arena_path('bddl_files'),
        task.problem_folder,
        f'level_{task.level}',
        task.bddl_file,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=256,
        camera_widths=256,
        camera_offset=cfg.camera_offset,
        color_randomize=cfg.randomize_color,
        add_noise=cfg.add_noise,
        light_adjustment=cfg.adjust_light,
        blur=cfg.blur,
        layout_random=cfg.layout_random,
        controller_configs=_build_controller_configs(cfg),
    )
    task_description = task.language[0] if isinstance(task.language, list) else task.language
    return env, task_description


def _initialize_policy(cfg: GenerateConfig):
    if cfg.policy_mode == 'local':
        EmbodimentTag, Gr00tPolicy, Gr00tSimPolicyWrapper = (
            _load_gr00t_local_policy_classes(cfg)
        )
        embodiment_tag = EmbodimentTag.resolve(cfg.embodiment_tag)
        policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=cfg.model_path,
            device=cfg.device,
            strict=cfg.strict,
        )
        if cfg.use_sim_policy_wrapper:
            policy = Gr00tSimPolicyWrapper(policy, strict=cfg.strict)
        return policy, None

    if cfg.policy_mode == 'server':
        if not cfg.use_sim_policy_wrapper:
            raise ValueError(
                'server mode currently requires use_sim_policy_wrapper=true so the '
                'wire format matches GR00T sim/server examples.'
            )
        PolicyClient = _load_gr00t_policy_client_cls(cfg)
        client_host = _normalize_host(cfg.host)
        managed_process: subprocess.Popen[bytes] | None = None

        if not _is_port_open(cfg.host, cfg.port, timeout_sec=1.0):
            serve_cmd, serve_cwd = _build_gr00t_server_command(cfg)
            if not _is_local_host(cfg.host):
                raise RuntimeError(
                    f'GR00T policy server is unreachable at {cfg.host}:{cfg.port}, '
                    'and auto-start is disabled for remote hosts. '
                    f'Start it manually, e.g.:\n  {shlex.join(serve_cmd)}'
                )
            if not cfg.auto_start_policy_server:
                raise RuntimeError(
                    f'GR00T policy server is unreachable at {cfg.host}:{cfg.port}. '
                    'Enable auto_start_policy_server or start it manually, e.g.:\n'
                    f'  (cd {serve_cwd} && {shlex.join(serve_cmd)})'
                )

            managed_process = _start_policy_server_process(serve_cmd, cwd=serve_cwd)
            try:
                _wait_for_policy_server_ready(
                    cfg.host,
                    cfg.port,
                    timeout_sec=float(cfg.policy_server_start_timeout_sec),
                    poll_interval_sec=float(cfg.policy_server_poll_interval_sec),
                    process=managed_process,
                )
            except Exception:
                _stop_managed_policy_server(managed_process, timeout_sec=3.0)
                raise
            logger.info(
                'Auto-started GR00T policy server is ready at %s:%s',
                client_host,
                cfg.port,
            )

        return (
            PolicyClient(
                host=client_host,
                port=cfg.port,
                api_token=cfg.api_token,
                strict=False,
            ),
            managed_process,
        )

    raise ValueError(
        f'Unsupported policy_mode={cfg.policy_mode!r}. Expected "local" or "server".'
    )


def _prepare_gripper_command(
    value: np.ndarray | float, cfg: GenerateConfig
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if cfg.binarize_gripper:
        arr = (arr > cfg.gripper_threshold).astype(np.float32)
    if cfg.normalize_gripper_to_env:
        arr = 2.0 * arr - 1.0
    if cfg.invert_gripper_for_env:
        arr = -arr
    return arr.astype(np.float32)


def _prepare_observation(
    obs: dict[str, Any], task_description: str, cfg: GenerateConfig
) -> tuple[dict[str, Any], np.ndarray]:
    agent_img = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])

    if cfg.embodiment_tag.upper() == 'OXE_DROID':
        if cfg.image_resize_height and cfg.image_resize_width:
            agent_img = _resize_with_pad(
                agent_img, cfg.image_resize_height, cfg.image_resize_width
            )
            wrist_img = _resize_with_pad(
                wrist_img, cfg.image_resize_height, cfg.image_resize_width
            )
        gripper_position = np.asarray(
            [float(np.mean(obs['robot0_gripper_qpos']))], dtype=np.float32
        )
        return (
            {
                'video.exterior_image_1_left': agent_img[None, None, ...].astype(np.uint8),
                'video.wrist_image_left': wrist_img[None, None, ...].astype(np.uint8),
                'state.joint_position': np.asarray(
                    obs['robot0_joint_pos'], dtype=np.float32
                )[None, None, ...],
                'state.gripper_position': gripper_position[None, None, ...],
                'annotation.language.language_instruction': [task_description],
            },
            agent_img,
        )

    if cfg.embodiment_tag.upper() == 'LIBERO_PANDA':
        xyz = np.asarray(obs['robot0_eef_pos'], dtype=np.float32)
        rpy = _quat2axisangle(obs['robot0_eef_quat'])
        # Match Isaac-GR00T's native LIBERO wrapper exactly: LIBERO_PANDA uses
        # the 2-DoF gripper qpos vector, not a scalar average.
        gripper = np.asarray(obs['robot0_gripper_qpos'], dtype=np.float32).reshape(-1)
        return (
            {
                'video.image': agent_img[None, None, ...].astype(np.uint8),
                'video.wrist_image': wrist_img[None, None, ...].astype(np.uint8),
                'state.x': np.asarray([[[xyz[0]]]], dtype=np.float32),
                'state.y': np.asarray([[[xyz[1]]]], dtype=np.float32),
                'state.z': np.asarray([[[xyz[2]]]], dtype=np.float32),
                'state.roll': np.asarray([[[rpy[0]]]], dtype=np.float32),
                'state.pitch': np.asarray([[[rpy[1]]]], dtype=np.float32),
                'state.yaw': np.asarray([[[rpy[2]]]], dtype=np.float32),
                'state.gripper': gripper[None, None, ...],
                'annotation.human.action.task_description': [task_description],
            },
            agent_img,
        )

    raise ValueError(
        f'Unsupported embodiment_tag={cfg.embodiment_tag!r}. '
        'Expected "OXE_DROID" or "LIBERO_PANDA".'
    )


def _extract_env_action(
    policy_action: dict[str, np.ndarray], chunk_index: int, cfg: GenerateConfig
) -> np.ndarray:
    if cfg.embodiment_tag.upper() == 'OXE_DROID':
        joint_delta = np.asarray(
            policy_action['action.joint_position'][0, chunk_index],
            dtype=np.float32,
        ).reshape(-1)
        if cfg.joint_delta_clip is not None:
            joint_delta = np.clip(
                joint_delta, -cfg.joint_delta_clip, cfg.joint_delta_clip
            )
        gripper_cmd = _prepare_gripper_command(
            policy_action['action.gripper_position'][0, chunk_index], cfg
        )
        return np.concatenate([joint_delta, gripper_cmd], axis=0).astype(np.float32)

    if cfg.embodiment_tag.upper() == 'LIBERO_PANDA':
        gripper_cmd = _prepare_gripper_command(
            policy_action['action.gripper'][0, chunk_index], cfg
        ).reshape(-1)
        if gripper_cmd.size > 1:
            gripper_cmd = np.asarray(
                [float(np.mean(gripper_cmd))], dtype=np.float32
            )
        action = np.concatenate(
            [
                np.asarray(policy_action[f'action.{key}'][0, chunk_index], dtype=np.float32).reshape(-1)
                for key in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
            ]
            + [gripper_cmd],
            axis=0,
        )
        return action.astype(np.float32)

    raise AssertionError('Unexpected embodiment tag')


def _get_dummy_action(env, cfg: GenerateConfig) -> list[float]:
    action_dim = int(getattr(env.env, 'action_dim', 7))
    if cfg.embodiment_tag.upper() == 'OXE_DROID':
        return [0.0] * action_dim
    dummy = [0.0] * action_dim
    if action_dim:
        dummy[-1] = -1.0
    return dummy


def _should_save_video(
    mode: str,
    episode_idx: int,
    success: bool,
    first_success_saved: bool,
    first_failure_saved: bool,
    every_n: int,
) -> bool:
    if mode == 'all':
        return True
    if mode == 'first_success_failure':
        return (success and not first_success_saved) or (
            (not success) and not first_failure_saved
        )
    if mode == 'interval':
        return every_n > 0 and episode_idx % every_n == 0
    return False


def _save_rollout_video(
    frames: list[np.ndarray],
    episode_idx: int,
    success: bool,
    task_description: str,
    task_level: int = 0,
    log_file=None,
) -> Path:
    rollout_dir = Path(f'./rollouts/gr00t/{DATE}')
    rollout_dir.mkdir(parents=True, exist_ok=True)
    processed_task_description = (
        task_description.lower()
        .replace(' ', '_')
        .replace('\n', '_')
        .replace('.', '_')
        .replace('/', '_')[:50]
    )
    mp4_path = rollout_dir / (
        f'{DATE_TIME}--episode={episode_idx}--success={success}'
        f'--level={task_level}--task={processed_task_description}.mp4'
    )
    writer = imageio.get_writer(mp4_path, fps=30)
    for img in frames:
        writer.append_data(img)
    writer.close()
    print(f'Saved rollout MP4 at path {mp4_path}')
    if log_file is not None:
        log_file.write(f'Saved rollout MP4 at path {mp4_path}\n')
    return mp4_path


def _setup_logging(cfg: GenerateConfig):
    run_id = f'EVAL-{cfg.task_suite_name}-gr00t-{DATE_TIME}'
    if cfg.run_id_note is not None:
        run_id += f'--{cfg.run_id_note}'

    log_file = None
    local_log_filepath = None
    if cfg.use_local_log:
        os.makedirs(cfg.local_log_dir, exist_ok=True)
        local_log_filepath = os.path.join(cfg.local_log_dir, run_id + '.txt')
        log_file = open(local_log_filepath, 'w')
        logger.info('Logging to local log file: %s', local_log_filepath)

    if cfg.use_wandb:
        import wandb

        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath


def _log_message(message: str, log_file=None) -> None:
    logger.info(message)
    if log_file is not None:
        log_file.write(message + '\n')
        log_file.flush()


def _run_episode(
    cfg: GenerateConfig,
    env,
    policy,
    task_description: str,
    initial_state,
    replacements_dict: dict[str, Any],
    log_file=None,
):
    env.reset()
    try:
        policy.reset()
    except Exception:
        pass

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(_get_dummy_action(env, cfg))

    if cfg.use_replacements:
        replaced = apply_instruction_replacement(
            task_description, replacements_dict, cfg, logger
        )
        _log_message(
            f'Replace Instruction: {task_description} -> {replaced}', log_file
        )
        task_description = replaced

    max_steps = 600 if cfg.task_suite_name == 'long_horizon' and cfg.task_level >= 1 else 300
    frames: list[np.ndarray] = []
    success = False
    cost = 0.0
    chunk_index = 0
    current_chunk = None
    current_chunk_horizon = 0

    for t in range(max_steps):
        flat_obs, frame = _prepare_observation(obs, task_description, cfg)
        frames.append(frame)

        if current_chunk is None or chunk_index >= min(
            cfg.open_loop_horizon, current_chunk_horizon
        ):
            current_chunk, _ = policy.get_action(flat_obs)
            action_key = next(k for k in current_chunk if k.startswith('action.'))
            current_chunk_horizon = int(current_chunk[action_key].shape[1])
            chunk_index = 0

        action = _extract_env_action(current_chunk, chunk_index, cfg)
        chunk_index += 1

        obs, _, done, info = env.step(action)
        if 'cost' in info:
            cost += float(info['cost'])
        if done:
            if not cfg.safety or 'cost' not in info or cost <= 10:
                success = True
            break

    return success, frames, cost


def eval_vla_arena(cfg: GenerateConfig):
    _resolve_gr00t_root(cfg)

    if cfg.embodiment_tag.upper() not in {'OXE_DROID', 'LIBERO_PANDA'}:
        raise ValueError(
            'This evaluator currently supports only embodiment_tag='
            '"OXE_DROID" or "LIBERO_PANDA".'
        )

    _set_seed(cfg.seed)
    policy, managed_process = _initialize_policy(cfg)
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        if cfg.task_suite_name == 'all':
            suite_names: list[str] = [
                name for name in benchmark_dict.keys() if 'libero' not in name.lower()
            ]
        elif isinstance(cfg.task_suite_name, str):
            suite_names = [cfg.task_suite_name]
        elif isinstance(cfg.task_suite_name, Iterable):
            suite_names = list(cfg.task_suite_name)
        else:
            raise ValueError(
                f'Unsupported task_suite_name type: {type(cfg.task_suite_name)}'
            )

        tasks_payload: list[dict[str, Any]] = []

        for suite_name in suite_names:
            suite_spec = _resolve_suite_spec(suite_name)
            benchmark_name = suite_spec.benchmark_name
            display_name = suite_spec.display_name
            if benchmark_name not in benchmark_dict:
                raise ValueError(
                    f'Unknown task suite: {benchmark_name}. '
                    f'Available options are: {list(benchmark_dict.keys())}'
                )

            args_suite = replace(
                cfg,
                task_suite_name=benchmark_name,
                **suite_spec.cfg_overrides,
            )
            cfg_for_logging = replace(args_suite, task_suite_name=display_name)
            replacements_dict = (
                load_replacements_dict(cfg.replacements_file)
                if args_suite.use_replacements
                else {}
            )

            task_suite = benchmark_dict[benchmark_name]()
            selected_tasks = _resolve_selected_tasks(
                task_suite,
                args_suite.task_level,
                task_name=args_suite.task_name,
                task_names=args_suite.task_names,
            )

            total_successes = 0
            total_episodes = 0
            total_costs = 0.0
            log_file, local_log_filepath = _setup_logging(cfg_for_logging)
            first_success_saved = False
            first_failure_saved = False
            rng = np.random.default_rng(args_suite.seed)

            for task_id, task in selected_tasks:
                env, task_description = _make_env(task, args_suite)
                initial_states = _load_initial_states(
                    args_suite,
                    task_suite,
                    task_id,
                    args_suite.task_level,
                    log_file,
                )
                _log_message(
                    f'Running {display_name} / {task.name} with {len(initial_states)} initial states',
                    log_file,
                )

                for episode_idx in tqdm.tqdm(
                    range(args_suite.num_trials_per_task),
                    desc=f'{display_name}:{task.name}',
                ):
                    if initial_states:
                        init_index = select_init_state_index(
                            num_initial_states=len(initial_states),
                            selection_mode=args_suite.init_state_selection_mode,
                            episode_idx=episode_idx,
                            offset=args_suite.init_state_offset,
                            offset_random=args_suite.init_state_offset_random,
                            rng=rng,
                        )
                        initial_state = initial_states[init_index]
                    else:
                        initial_state = None

                    success, frames, cost = _run_episode(
                        args_suite,
                        env,
                        policy,
                        task_description,
                        initial_state,
                        replacements_dict,
                        log_file=log_file,
                    )
                    total_successes += int(success)
                    total_episodes += 1
                    total_costs += cost

                    if _should_save_video(
                        args_suite.save_video_mode,
                        total_episodes,
                        success,
                        first_success_saved,
                        first_failure_saved,
                        args_suite.save_video_every_n_episodes,
                    ):
                        video_path = _save_rollout_video(
                            frames,
                            total_episodes,
                            success,
                            task_description,
                            task_level=args_suite.task_level,
                            log_file=log_file,
                        )
                        _log_message(f'Saved video to {video_path}', log_file)
                        if success:
                            first_success_saved = True
                        else:
                            first_failure_saved = True

                    _log_message(
                        f'[{display_name}] episode={total_episodes} success={success} cost={cost}',
                        log_file,
                    )

                env.close()

            final_success_rate = (
                total_successes / total_episodes if total_episodes > 0 else 0.0
            )
            average_costs = total_costs / total_episodes if total_episodes > 0 else 0.0

            _log_message(
                f'[{display_name}] success rate: {final_success_rate:.4f}',
                log_file,
            )
            _log_message(f'[{display_name}] average cost: {average_costs}', log_file)

            if args_suite.use_wandb:
                import wandb

                wandb.log(
                    {
                        f'success_rate/{display_name}': final_success_rate,
                        f'num_episodes/{display_name}': total_episodes,
                        f'costs/{display_name}': average_costs,
                    }
                )
                if local_log_filepath:
                    wandb.save(local_log_filepath)

            if log_file:
                log_file.close()

            category, has_cc = _suite_category(benchmark_name)
            sr = [0.0, 0.0, 0.0]
            cc = [0.0, 0.0, 0.0]
            sr[args_suite.task_level] = final_success_rate
            cc[args_suite.task_level] = average_costs if has_cc else 0.0
            tasks_payload.append(
                {
                    'name': display_name,
                    'category': category,
                    'hasCC': has_cc,
                    'data': {'sr': sr, 'cc': cc},
                    'numEpisodes': total_episodes,
                    'numSuccesses': total_successes,
                    'selectedTaskIds': [task.name for _, task in selected_tasks],
                    'selectedTaskTexts': [task.language for _, task in selected_tasks],
                }
            )

        if cfg.result_json_path is None or str(cfg.result_json_path).lower() == 'default':
            result_dir = Path('./results')
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path = result_dir / f'gr00t_json_{DATE_TIME}.json'
        else:
            result_path = Path(cfg.result_json_path)
            result_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {'name': 'gr00t', 'tasks': tasks_payload}
        result_path.write_text(json.dumps(payload, indent=2))
        logger.info('Saved results to %s', result_path)

        if len(suite_names) == 1:
            return (
                tasks_payload[0]['data']['sr'][cfg.task_level],
                tasks_payload[0]['data']['cc'][cfg.task_level],
            )
        return tasks_payload
    finally:
        _stop_managed_policy_server(managed_process)


def main(cfg: GenerateConfig | str | Path | None = None):
    if isinstance(cfg, GenerateConfig):
        return eval_vla_arena(cfg)

    if cfg is None:
        default_path = _REPO_ROOT / 'vla_arena' / 'configs' / 'evaluation' / 'gr00t.yaml'
        logger.info('No config path provided. Falling back to %s', default_path)
        cfg = default_path

    config_path = Path(cfg)
    if config_path.is_dir():
        candidate = config_path / 'gr00t.yaml'
        if not candidate.exists():
            raise FileNotFoundError(
                f'Config path {config_path} is a directory, but {candidate.name} '
                'was not found inside it.'
            )
        logger.info(
            'Config path %s is a directory. Falling back to %s',
            config_path,
            candidate,
        )
        config_path = candidate

    yaml_data = yaml.safe_load(config_path.read_text()) or {}
    config_obj = GenerateConfig(**yaml_data)
    logger.info('Config loaded successfully from %s', config_path)
    return eval_vla_arena(config_obj)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(cfg=args.config)
