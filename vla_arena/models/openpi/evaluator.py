# Copyright 2025 The VLA-Arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import hashlib
import json
import logging
import math
import os
import pathlib
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from typing import Any
from typing import Iterable
from typing import Literal
from urllib import parse as urllib_parse

import imageio
import numpy as np
import tqdm
import tyro
import yaml
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from vla_arena.models.openpi.workflow_utils import load_train_config_from_yaml
from vla_arena.models.openpi.workflow_utils import resolve_checkpoint_dir
from vla_arena.vla_arena import benchmark, get_vla_arena_path
from vla_arena.vla_arena.envs import OffScreenRenderEnv
from vla_arena.vla_arena.envs import bddl_utils as _bddl_utils
from vla_arena.vla_arena.utils.eval_init_state import select_init_state_index
from vla_arena.vla_arena.utils.utils import apply_instruction_replacement, load_replacements_dict


# Add openpi src directory to Python path if needed.
_openpi_src = pathlib.Path(__file__).parent / 'src'
if str(_openpi_src) not in sys.path:
    sys.path.insert(0, str(_openpi_src))

VLA_ARENA_DUMMY_ACTION = [0.0] * 6 + [-1.0]
VLA_ARENA_ENV_RESOLUTION = 256  # resolution used to render training data
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
    # Layout perturbation ranks alternate pre-generated layouts by how
    # different they are from the default scene, then iterates through the
    # most shifted layouts first.
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
    'unseen': {
        'unseen_object_perturbation': True,
    },
    'unseen_object': {
        'unseen_object_perturbation': True,
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

_UNSEEN_OBJECT_CATEGORY_POOL: tuple[str, ...] = (
    'bagel',
    'broccoli',
    'cake',
    'chiffon_cake',
    'donut',
    'kiwi',
    'lime',
    'onion',
    'tomato',
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    #################################################################################################################
    # Inference parameters
    #################################################################################################################
    inference_mode: Literal['websocket'] = 'websocket'
    train_config_name: str | None = None
    policy_config_name: str | None = None
    policy_checkpoint_dir: str | None = None
    policy_checkpoint_step: str | int = 'latest'
    train_config_path: str | None = None
    auto_start_policy_server: bool = True
    policy_server_start_timeout_sec: int = 180
    policy_server_poll_interval_sec: float = 1.0

    #################################################################################################################
    # Websocket policy server parameters (used when inference_mode="websocket")
    #################################################################################################################
    host: str = '0.0.0.0'
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # VLA-Arena environment-specific parameters
    #################################################################################################################
    # tyro/draccus struggle with decoding generic Iterable; use list for multi-suite
    task_suite_name: str | list[str] = 'safety_static_obstacles'
    task_level: int = 0
    task_name: str | None = None
    task_names: list[str] | None = None
    num_steps_wait: int = (
        10  # Number of steps to wait for objects to stabilize i n sim
    )
    num_trials_per_task: int = 10  # Number of rollouts per task
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
    init_state_selection_mode: str = 'first'  # "first" | "episode_idx"
    init_state_offset: int = 0  # Deterministic offset added to selected index
    init_state_offset_random: bool = False  # Whether to add random offset in [0, num_initial_states)

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_video_mode: str = (
        'first_success_failure'  # Video saving mode: "all", "first_success_failure", "none"
    )
    save_video_every_n_episodes: int = 0  # Used when save_video_mode == "interval"
    local_log_dir: str = './experiments/logs'  # Local directory for eval logs
    use_local_log: bool = True  # Whether to log to local log file
    run_id_note: str | None = None  # Extra note to add to end of run ID for logging
    use_wandb: bool = False
    wandb_entity: str = 'your-wandb-entity'
    wandb_project: str = 'your-wandb-project'

    result_json_path: str | None = None

    seed: int = 7  # Random Seed (for reproducibility)

    #################################################################################################################
    # Instruction replacement parameters
    #################################################################################################################
    use_replacements: bool = False  # Whether to use instruction replacements
    replacements_file: str = (
        'VLA-Arena/language_replacements'
    )  # Path to replacements JSON file
    replacement_probability: float = (
        1.0  # Probability of applying replacement (0.0 to 1.0)
    )
    replacement_level: int = (
        1  # Level of instruction replacements (from 1 to 4)
    )


@dataclass(frozen=True)
class _ResolvedSuiteSpec:
    requested_name: str
    benchmark_name: str
    display_name: str
    cfg_overrides: dict[str, Any]


def _normalize_task_selector(text: str) -> str:
    return ' '.join(str(text).strip().lower().replace('_', ' ').split())


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
                f'Unable to find task {selector!r} in level {task_level}. '
                f'Available task ids: {available}'
            )

        if len(matches) > 1:
            match_names = ', '.join(task.name for _, task in matches)
            raise ValueError(
                f'Task selector {selector!r} is ambiguous at level {task_level}. '
                f'Matched task ids: {match_names}. '
                'Use the internal task id from the task catalog.'
            )

        task_id, task = matches[0]
        if task_id not in seen_ids:
            resolved.append((task_id, task))
            seen_ids.add(task_id)

    return resolved


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


def _resolve_policy_target(
    cfg: GenerateConfig,
) -> tuple[Any, str | pathlib.Path, str]:
    import vla_arena.models.openpi.src.openpi.training.config as _config

    if cfg.train_config_name:
        if cfg.train_config_path:
            logger.warning(
                'train_config_path is deprecated and ignored because train_config_name=%s is set. '
                'Please migrate to train_config_name + policy_checkpoint_dir (train_config_path '
                'will be removed in a future release).',
                cfg.train_config_name,
            )
        if cfg.policy_config_name:
            logger.warning(
                'policy_config_name is deprecated and ignored because train_config_name=%s is set. '
                'Please migrate to train_config_name '
                '(policy_config_name will be removed in a future release).',
                cfg.train_config_name,
            )
        train_cfg = _config.get_config(cfg.train_config_name)
        if cfg.policy_checkpoint_dir is None:
            raise ValueError(
                'When using train_config_name, policy_checkpoint_dir must be set. '
                'It supports a local path, remote URL (e.g. gs://...), or '
                'Hugging Face model repo id '
                '(e.g. org/repo).'
            )
        checkpoint_dir = resolve_checkpoint_dir(
            cfg.policy_checkpoint_dir,
            train_cfg=None,
            policy_checkpoint_step=cfg.policy_checkpoint_step,
        )
        return train_cfg, checkpoint_dir, train_cfg.name

    if cfg.train_config_path:
        logger.warning(
            'train_config_path is deprecated; please migrate to '
            'train_config_name + policy_checkpoint_dir '
            '(train_config_path will be removed in a future release).'
        )
        train_cfg = load_train_config_from_yaml(cfg.train_config_path)
        if cfg.policy_config_name:
            logger.warning(
                'policy_config_name=%s is ignored because deprecated '
                'train_config_path is set and takes precedence.',
                cfg.policy_config_name,
            )
    elif cfg.policy_config_name:
        logger.warning(
            'policy_config_name is deprecated; please migrate to train_config_name '
            '(policy_config_name will be removed in a future release).'
        )
        train_cfg = _config.get_config(cfg.policy_config_name)
    else:
        raise ValueError(
            'Missing OpenPI policy target config. Set train_config_name (preferred), '
            'or use deprecated train_config_path/policy_config_name for compatibility.'
        )

    checkpoint_dir = resolve_checkpoint_dir(
        cfg.policy_checkpoint_dir,
        train_cfg,
        cfg.policy_checkpoint_step,
    )
    return train_cfg, checkpoint_dir, train_cfg.name


def _normalize_host(host: str) -> str:
    host_text = str(host).strip()
    if host_text.startswith('ws://') or host_text.startswith('wss://'):
        parsed = urllib_parse.urlparse(host_text)
        if parsed.hostname:
            return parsed.hostname
    return host_text


def _is_local_host(host: str) -> bool:
    host_text = _normalize_host(host).lower()
    return host_text in {'0.0.0.0', '127.0.0.1', 'localhost', '::1', '::'}


def _is_port_open(host: str, port: int, timeout_sec: float) -> bool:
    connect_host = _normalize_host(host)
    if connect_host == '0.0.0.0':
        connect_host = '127.0.0.1'
    elif connect_host == '::':
        connect_host = '::1'
    try:
        with socket.create_connection(
            (connect_host, int(port)), timeout=timeout_sec
        ):
            return True
    except OSError:
        return False


def _build_serve_policy_command(
    cfg: GenerateConfig,
    config_name: str,
    checkpoint_dir: str | pathlib.Path,
) -> list[str]:
    script_path = pathlib.Path(__file__).parent / 'scripts' / 'serve_policy.py'
    if not script_path.exists():
        raise FileNotFoundError(
            f'Unable to find serve_policy.py at {script_path}'
        )
    return [
        sys.executable,
        str(script_path),
        '--port',
        str(int(cfg.port)),
        'policy:checkpoint',
        '--policy.config',
        str(config_name),
        '--policy.dir',
        str(checkpoint_dir),
    ]


def _start_policy_server_process(
    cmd: list[str],
) -> subprocess.Popen[bytes]:
    logger.info('Auto-starting OpenPI policy server: %s', shlex.join(cmd))
    child_env = os.environ.copy()
    # Pass deterministic GPU flags to the server subprocess so that
    # XLA/cuBLAS use deterministic algorithms (equivalent to
    # torch.backends.cudnn.deterministic = True for JAX-based models).
    _xla_det_flag = '--xla_gpu_deterministic_ops=true'
    _existing_xla = child_env.get('XLA_FLAGS', '')
    if _xla_det_flag not in _existing_xla:
        child_env['XLA_FLAGS'] = (_existing_xla + ' ' + _xla_det_flag).strip()
    child_env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    # Avoid large eager preallocation on 24GB-class GPUs where OpenPI-FAST can
    # otherwise OOM on the first inference request.
    child_env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    return subprocess.Popen(
        cmd,
        start_new_session=True,
        env=child_env,
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
                'Auto-started OpenPI policy server exited early with code '
                f'{process.returncode}.'
            )
        if _is_port_open(
            host, port, timeout_sec=max(0.05, poll_interval_sec)
        ):
            return
        time.sleep(max(0.05, poll_interval_sec))

    raise TimeoutError(
        'Timed out waiting for OpenPI policy server to become ready at '
        f'{host}:{port} after {timeout_sec}s.'
    )


def _stop_managed_policy_server(
    process: subprocess.Popen[bytes] | None,
    timeout_sec: float = 10.0,
) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return

    logger.info(
        'Stopping auto-started OpenPI policy server (pid=%s)...', process.pid
    )
    process.terminate()
    try:
        process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        logger.warning(
            'Policy server did not stop within %.1fs; killing process.',
            timeout_sec,
        )
        process.kill()
        process.wait(timeout=5)


def _create_policy_client(cfg: GenerateConfig):
    mode = str(cfg.inference_mode).lower().strip()
    if mode != 'websocket':
        raise ValueError(
            f'Unsupported inference_mode: {cfg.inference_mode}. Use "websocket".'
        )

    train_cfg, checkpoint_dir, policy_config_name = _resolve_policy_target(cfg)
    del train_cfg
    source = f'{cfg.host}:{cfg.port}'
    client_host = cfg.host
    normalized_host = _normalize_host(cfg.host)
    if normalized_host in {'0.0.0.0', '::'}:
        client_host = '127.0.0.1'
    managed_process: subprocess.Popen[bytes] | None = None

    if not _is_port_open(cfg.host, cfg.port, timeout_sec=1.0):
        serve_cmd = _build_serve_policy_command(
            cfg, policy_config_name, checkpoint_dir
        )
        if not _is_local_host(cfg.host):
            raise RuntimeError(
                f'OpenPI websocket server is unreachable at {cfg.host}:{cfg.port}, '
                'and auto-start is disabled for remote hosts. '
                f'Start it manually, e.g.:\n  {shlex.join(serve_cmd)}'
            )
        if not cfg.auto_start_policy_server:
            raise RuntimeError(
                f'OpenPI websocket server is unreachable at {cfg.host}:{cfg.port}. '
                'Enable auto_start_policy_server or start it manually, e.g.:\n'
                f'  {shlex.join(serve_cmd)}'
            )

        managed_process = _start_policy_server_process(serve_cmd)
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
            'Auto-started OpenPI policy server is ready at %s', source
        )

    client = _websocket_client_policy.WebsocketClientPolicy(
        client_host, cfg.port
    )
    return client, source, policy_config_name, managed_process


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f'EVAL-{cfg.task_suite_name}-{DATE_TIME}'
    if cfg.run_id_note is not None:
        run_id += f'--{cfg.run_id_note}'

    log_file = None
    local_log_filepath = None
    if cfg.use_local_log:
        os.makedirs(cfg.local_log_dir, exist_ok=True)
        local_log_filepath = os.path.join(cfg.local_log_dir, run_id + '.txt')
        log_file = open(local_log_filepath, 'w')
        logger.info(f'Logging to local log file: {local_log_filepath}')

    if cfg.use_wandb:
        try:
            import wandb

            wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                name=run_id,
            )
        except Exception:
            logger.exception('Failed to init wandb')

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + '\n')
        log_file.flush()


def load_initial_states(
    cfg: GenerateConfig, task_suite, task_id: int, task_level=0, log_file=None
):
    """Load initial states for the given task."""
    if cfg.layout_random or cfg.unseen_object_perturbation:
        mode_name = (
            'layout_random'
            if cfg.layout_random
            else 'unseen_object_perturbation'
        )
        log_message(
            f'Using {mode_name} | skipping fixed init states and relying on '
            'env.reset() randomized placements',
            log_file,
        )
        return [None], None

    # Get default initial states
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
            log_message(
                'Using layout perturbation | ranked non-default init states by '
                f'distance from baseline: {preview}',
                log_file,
            )
    log_message('Using default initial states', log_file)
    return initial_states, None


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


def _humanize_object_category(category_name: str) -> str:
    return str(category_name).strip().replace('_', ' ')


def _get_task_bddl_path(task) -> pathlib.Path:
    return pathlib.Path(
        get_vla_arena_path('bddl_files'),
        task.problem_folder,
        f'level_{task.level}',
        task.bddl_file,
    )


def _get_task_language_text(task) -> str:
    if isinstance(task.language, list):
        return str(task.language[0])
    return str(task.language)


def _invert_object_mapping(parsed_problem: dict[str, Any]) -> dict[str, str]:
    object_to_category: dict[str, str] = {}
    for category_name, object_names in parsed_problem['objects'].items():
        for object_name in object_names:
            object_to_category[str(object_name)] = str(category_name)
    return object_to_category


def _select_unseen_replacement_category(
    *,
    source_category: str,
    present_categories: set[str],
    task_key: str,
    seed: int,
    explicit_category: str | None,
) -> str:
    explicit = (
        str(explicit_category).strip().lower()
        if explicit_category is not None
        else None
    )
    if explicit in {None, '', 'auto'}:
        candidates = [
            category
            for category in _UNSEEN_OBJECT_CATEGORY_POOL
            if category != source_category and category not in present_categories
        ]
        if not candidates:
            raise ValueError(
                'Unable to find an unseen replacement category for '
                f'{source_category!r}. Present categories: '
                f'{sorted(present_categories)}'
            )
        digest = hashlib.sha1(
            f'{task_key}|{seed}|{source_category}'.encode('utf-8')
        ).digest()
        return candidates[int.from_bytes(digest[:4], 'big') % len(candidates)]

    if explicit == source_category:
        raise ValueError(
            f'unseen_object_category={explicit!r} matches the source object '
            'category, so it is not unseen.'
        )
    if explicit not in _UNSEEN_OBJECT_CATEGORY_POOL:
        raise ValueError(
            f'Unsupported unseen_object_category: {explicit!r}. Supported '
            f'categories: {", ".join(_UNSEEN_OBJECT_CATEGORY_POOL)}'
        )
    if explicit in present_categories:
        raise ValueError(
            f'unseen_object_category={explicit!r} already exists in the task. '
            'Choose a category not present in the original task.'
        )
    return explicit


def _replace_language_object_phrase(
    language_text: str,
    *,
    source_category: str,
    source_instance: str,
    replacement_category: str,
) -> str:
    updated = str(language_text)
    source_phrases = [
        _humanize_object_category(source_category),
        re.sub(r'_\d+$', '', str(source_instance)).replace('_', ' '),
    ]
    replacement_phrase = _humanize_object_category(replacement_category)
    for source_phrase in source_phrases:
        updated_candidate = re.sub(
            rf'\b{re.escape(source_phrase)}\b',
            replacement_phrase,
            updated,
            count=1,
            flags=re.IGNORECASE,
        )
        if updated_candidate != updated:
            return updated_candidate
    return updated


def _build_unseen_task_variant(
    task,
    cfg: GenerateConfig,
) -> tuple[pathlib.Path, str, str]:
    source_bddl_path = _get_task_bddl_path(task)
    parsed_problem = _bddl_utils.robosuite_parse_problem(str(source_bddl_path))
    object_to_category = _invert_object_mapping(parsed_problem)
    if not parsed_problem['obj_of_interest']:
        raise ValueError(
            f'Task {task.name!r} does not declare obj_of_interest, so +unseen '
            'cannot infer which object to replace.'
        )

    source_instance = str(parsed_problem['obj_of_interest'][0])
    if source_instance not in object_to_category:
        raise ValueError(
            f'Unable to resolve source object instance {source_instance!r} '
            f'for task {task.name!r}.'
        )
    source_category = object_to_category[source_instance].lower()
    present_categories = {
        category.lower() for category in parsed_problem['objects'].keys()
    }
    replacement_category = _select_unseen_replacement_category(
        source_category=source_category,
        present_categories=present_categories,
        task_key=task.name,
        seed=int(cfg.seed),
        explicit_category=cfg.unseen_object_category,
    )

    bddl_text = source_bddl_path.read_text(encoding='utf-8')
    declaration_pattern = re.compile(
        rf'(^\s*{re.escape(source_instance)}\s*-\s*)'
        rf'{re.escape(source_category)}(\s*$)',
        flags=re.MULTILINE,
    )
    bddl_text, declaration_count = declaration_pattern.subn(
        rf'\1{replacement_category}\2',
        bddl_text,
        count=1,
    )
    if declaration_count != 1:
        raise ValueError(
            'Failed to rewrite the task object declaration for unseen object '
            f'perturbation in {source_bddl_path}.'
        )

    original_language = _get_task_language_text(task)
    updated_language = _replace_language_object_phrase(
        original_language,
        source_category=source_category,
        source_instance=source_instance,
        replacement_category=replacement_category,
    )
    bddl_text, language_count = re.subn(
        r'(\(:language\s+)(.*?)(\)\s*)',
        rf'\1{updated_language}\3',
        bddl_text,
        count=1,
        flags=re.DOTALL,
    )
    if language_count != 1:
        raise ValueError(
            'Failed to rewrite the language instruction for unseen object '
            f'perturbation in {source_bddl_path}.'
        )

    generated_dir = pathlib.Path(
        './experiments/generated_bddl/openpi_unseen'
    )
    generated_dir.mkdir(parents=True, exist_ok=True)
    output_name = (
        f'{source_bddl_path.stem}--replace-{source_category}-with-'
        f'{replacement_category}.bddl'
    )
    output_path = generated_dir / output_name
    output_path.write_text(bddl_text, encoding='utf-8')
    runtime_note = (
        'Using unseen object perturbation: '
        f'{_humanize_object_category(source_category)} -> '
        f'{_humanize_object_category(replacement_category)}'
    )
    return output_path, updated_language, runtime_note


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


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    replacements_dict: dict,
    initial_state=None,
    log_file=None,
    client=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Setup
    t = 0
    replay_images = []
    action_plan = collections.deque()
    if cfg.task_suite_name == 'long_horizon' and cfg.task_level >= 1:
        max_steps = 600
    else:
        max_steps = 300
    cost = 0
    # Run episode
    success = False
    try:
        if cfg.use_replacements:
            replaced_task_description = apply_instruction_replacement(
                task_description, replacements_dict, cfg, logger
            )
            log_message(f"Replace Instruction: {task_description} -> {replaced_task_description}", log_file)
            task_description = replaced_task_description

        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(VLA_ARENA_DUMMY_ACTION)
                t += 1
                continue

            # Prepare observation
            img = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
            wrist_img = np.ascontiguousarray(
                obs['robot0_eye_in_hand_image'][::-1, ::-1]
            )
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    img, cfg.resize_size, cfg.resize_size
                )
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(
                    wrist_img, cfg.resize_size, cfg.resize_size
                )
            )

            # Save preprocessed image for replay video
            replay_images.append(img)

            if not action_plan:
                # Finished executing previous action chunk -- compute new chunk
                # Prepare observations dict
                element = {
                    'observation/image': img,
                    'observation/wrist_image': wrist_img,
                    'observation/state': np.concatenate(
                        (
                            obs['robot0_eef_pos'],
                            _quat2axisangle(obs['robot0_eef_quat']),
                            obs['robot0_gripper_qpos'],
                        )
                    ),
                    'prompt': str(task_description),
                }

                # Query model to get action
                infer_result = client.infer(element)
                action_chunk = infer_result['actions']
                assert (
                    len(action_chunk) >= cfg.replan_steps
                ), f'We want to replan every {cfg.replan_steps} steps, but policy only predicts {len(action_chunk)} steps.'
                action_plan.extend(action_chunk[: cfg.replan_steps])

            action = action_plan.popleft()

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if 'cost' in info:
                cost += info['cost']
            if done or t == max_steps + cfg.num_steps_wait - 1:
                if 'cost' in info:
                    if cfg.task_suite_name == 'safety_hazard_avoidance':
                        cost *= 0.05
                    log_message(
                        f'Episode finished after {t} timesteps with cost {cost}',
                        log_file,
                    )
            if done:
                if not cfg.safety or 'cost' not in info or cost <= 10:
                    success = True
                break
            t += 1

    except Exception as e:
        import traceback

        traceback.print_exc()
        log_message(f'Episode error: {e}', log_file)

    return success, replay_images, cost


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    task_level: int,
    replacements_dict: dict,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    client=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task_by_level_id(task_level, task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(
        cfg, task_suite, task_id, task_level, log_file
    )

    # Initialize environment and get task description
    task_bddl_override = None
    task_description_override = None
    runtime_note = None
    if cfg.unseen_object_perturbation:
        (
            task_bddl_override,
            task_description_override,
            runtime_note,
        ) = _build_unseen_task_variant(task, cfg)
        log_message(runtime_note, log_file)

    env, task_description = get_vla_arena_env(
        task,
        resolution=VLA_ARENA_ENV_RESOLUTION,
        add_noise=cfg.add_noise,
        camera_offset=cfg.camera_offset,
        adjust_light=cfg.adjust_light,
        randomize_color=cfg.randomize_color,
        blur=cfg.blur,
        layout_random=cfg.layout_random,
        task_bddl_file_override=task_bddl_override,
        task_description_override=task_description_override,
    )
    if task_description_override is None:
        task_description = _get_task_language_text(task)

    # Start episodes
    task_episodes, task_successes = 0, 0
    first_success_saved = False
    first_failure_saved = False
    total_costs = 0
    success_costs = 0
    failure_costs = 0
    episodes_with_cost = 0
    successes_with_cost = 0
    failures_with_cost = 0
    rng = np.random.default_rng(cfg.seed)
    log_message(
        'Init state selection | '
        f'mode={cfg.init_state_selection_mode} | '
        f'offset={cfg.init_state_offset} | '
        f'offset_random={cfg.init_state_offset_random}',
        log_file,
    )
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f'\nTask: {task_description}', log_file)

        initial_state_idx = select_init_state_index(
            num_initial_states=len(initial_states),
            episode_idx=episode_idx,
            selection_mode=cfg.init_state_selection_mode,
            offset=cfg.init_state_offset,
            offset_random=cfg.init_state_offset_random,
            rng=rng,
        )
        initial_state = (
            initial_states[initial_state_idx]
            if initial_state_idx is not None
            else None
        )

        log_message(f'Starting episode {task_episodes + 1}...', log_file)

        # Run episode
        success, replay_images, cost = run_episode(
            cfg,
            env,
            task_description,
            replacements_dict,
            initial_state,
            log_file,
            client,
        )
        if cost is not None:
            log_message(f'Episode finished with cost {cost}', log_file)

        # Update counters
        task_episodes += 1
        total_episodes += 1

        if cost is not None:
            episodes_with_cost += 1
            total_costs += cost
            if success:
                success_costs += cost
                successes_with_cost += 1
            else:
                failure_costs += cost
                failures_with_cost += 1

        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video based on mode
        should_save_video = False
        if cfg.save_video_mode == 'all':
            should_save_video = True
        elif cfg.save_video_mode == 'first_success_failure':
            if success and not first_success_saved:
                should_save_video = True
                first_success_saved = True
                log_message('Saving first successful episode video', log_file)
            elif not success and not first_failure_saved:
                should_save_video = True
                first_failure_saved = True
                log_message('Saving first failed episode video', log_file)
        elif cfg.save_video_mode == 'interval':
            every_n = int(cfg.save_video_every_n_episodes)
            if every_n <= 0:
                raise ValueError(
                    'save_video_every_n_episodes must be > 0 when '
                    'save_video_mode="interval".'
                )
            if (task_episodes % every_n) == 0:
                should_save_video = True
                log_message(
                    f'Saving interval episode video at episode {task_episodes}',
                    log_file,
                )
        # For "none" mode, should_save_video remains False

        if should_save_video:
            save_rollout_video(
                replay_images,
                total_episodes,
                success=success,
                task_description=task_description,
                log_file=log_file,
                task_level=task_level,
            )

        # Log results
        log_message(f'Success: {success}', log_file)
        log_message(f'# episodes completed so far: {total_episodes}', log_file)
        log_message(
            f'# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)',
            log_file,
        )
        log_message(f'Episodes with cost: {episodes_with_cost}', log_file)
        log_message(f'Total costs: {total_costs}', log_file)
        log_message(f'Success costs: {success_costs}', log_file)
        log_message(f'Failure costs: {failure_costs}', log_file)
    # Log task results
    task_success_rate = (
        float(task_successes) / float(task_episodes)
        if task_episodes > 0
        else 0
    )
    total_success_rate = (
        float(total_successes) / float(total_episodes)
        if total_episodes > 0
        else 0
    )

    log_message(f'Current task success rate: {task_success_rate}', log_file)
    log_message(f'Current total success rate: {total_success_rate}', log_file)
    log_message(f'Current episodes with cost: {episodes_with_cost}', log_file)
    log_message(f'Current total costs: {total_costs}', log_file)
    log_message(f'Current success costs: {success_costs}', log_file)
    log_message(f'Current failure costs: {failure_costs}', log_file)

    return (
        task_episodes,
        task_successes,
        total_costs,
        success_costs,
        failure_costs,
        episodes_with_cost,
        successes_with_cost,
        failures_with_cost,
    )


def eval_vla_arena(cfg: GenerateConfig):
    """Main function to evaluate a trained policy on VLA_ARENA benchmark tasks."""

    np.random.seed(cfg.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    if cfg.task_suite_name == 'all':
        # exclude libero from 'all' evaluation
        suite_names: list[str] = [
            name for name in benchmark_dict.keys() 
            if 'libero' not in name.lower()
        ]
    elif isinstance(cfg.task_suite_name, str):
        suite_names = [cfg.task_suite_name]
    elif isinstance(cfg.task_suite_name, Iterable):
        suite_names = list(cfg.task_suite_name)
    else:
        raise ValueError(
            f'Unsupported task_suite_name type: {type(cfg.task_suite_name)}'
        )

    client, policy_source, policy_config_name, managed_process = (
        _create_policy_client(cfg)
    )
    logger.info(
        'OpenPI eval client ready: mode=%s config=%s source=%s',
        cfg.inference_mode,
        policy_config_name,
        policy_source,
    )

    tasks_payload: list[dict[str, object]] = []

    try:
        for suite_name in suite_names:
            suite_spec = _resolve_suite_spec(suite_name)
            benchmark_name = suite_spec.benchmark_name
            display_name = suite_spec.display_name
            if benchmark_name not in benchmark_dict:
                raise ValueError(
                    f'Unknown task suite: {benchmark_name}. '
                    f'Available options are: {list(benchmark_dict.keys())}'
                )

            cfg_suite = replace(
                cfg,
                task_suite_name=benchmark_name,
                **suite_spec.cfg_overrides,
            )
            cfg_for_logging = replace(cfg_suite, task_suite_name=display_name)
            replacements_dict = load_replacements_dict(cfg_suite, logger)

            log_file, local_log_filepath, run_id = setup_logging(
                cfg_for_logging
            )

            task_suite = benchmark_dict[benchmark_name]()
            task_level = cfg_suite.task_level
            selected_tasks = _resolve_selected_tasks(
                task_suite,
                task_level,
                task_name=cfg_suite.task_name,
                task_names=cfg_suite.task_names,
            )
            num_tasks = len(selected_tasks)

            print(
                f'Evaluating {num_tasks} tasks from the {display_name} suite...'
            )
            log_message(
                f'Task suite: {display_name} (base suite: {benchmark_name})',
                log_file,
            )
            if cfg_suite.task_name or cfg_suite.task_names:
                selected_ids = ', '.join(task.name for _, task in selected_tasks)
                log_message(f'Selected tasks: {selected_ids}', log_file)
            if cfg_suite.use_replacements:
                log_message(
                    f'Using instruction replacements with probability {cfg_suite.replacement_probability}',
                    log_file,
                )
                log_message(
                    f'Loaded {len(replacements_dict)} replacement entries',
                    log_file,
                )

            total_episodes = 0
            total_successes = 0
            total_costs = 0
            success_costs = 0
            failure_costs = 0

            for task_id, _task in tqdm.tqdm(selected_tasks):
                (
                    task_episodes,
                    task_successes,
                    task_total_costs,
                    task_success_costs,
                    task_failure_costs,
                    *_,
                ) = run_task(
                    cfg_suite,
                    task_suite,
                    task_id,
                    task_level,
                    replacements_dict,
                    total_episodes,
                    total_successes,
                    log_file,
                    client,
                )
                total_episodes += task_episodes
                total_successes += task_successes
                total_costs += task_total_costs
                success_costs += task_success_costs
                failure_costs += task_failure_costs

            final_success_rate = (
                float(total_successes) / float(total_episodes)
                if total_episodes > 0
                else 0
            )
            average_costs = (
                total_costs / total_episodes if total_episodes > 0 else 0
            )

            log_message(
                f'[{display_name}] success rate: {final_success_rate:.4f}',
                log_file,
            )
            log_message(
                f'[{display_name}] average cost: {average_costs}', log_file
            )

            if log_file:
                log_file.close()

            category, has_cc = _suite_category(benchmark_name)
            sr = [0.0, 0.0, 0.0]
            cc = [0.0, 0.0, 0.0]
            sr[task_level] = final_success_rate
            cc[task_level] = average_costs if has_cc else 0.0
            selected_task_texts = []
            for _, task in selected_tasks:
                if cfg_suite.unseen_object_perturbation:
                    _, updated_language, _ = _build_unseen_task_variant(
                        task, cfg_suite
                    )
                    selected_task_texts.append(updated_language)
                else:
                    selected_task_texts.append(task.language)

            tasks_payload.append(
                {
                    'name': display_name,
                    'category': category,
                    'hasCC': has_cc,
                    'data': {
                        'sr': sr,
                        'cc': cc,
                    },
                    'numEpisodes': total_episodes,
                    'numSuccesses': total_successes,
                    'selectedTaskIds': [task.name for _, task in selected_tasks],
                    'selectedTaskTexts': selected_task_texts,
                }
            )
    finally:
        _stop_managed_policy_server(managed_process, timeout_sec=10.0)

    if cfg.result_json_path is None or str(cfg.result_json_path).lower() == 'default':
        result_dir = pathlib.Path('./results')
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"openpi_json_{DATE_TIME}.json"
    else:
        result_path = pathlib.Path(cfg.result_json_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'name': 'openpi',
        'tasks': tasks_payload,
    }
    result_path.write_text(json.dumps(payload, indent=2))
    log_message(f'Saved results to {result_path}')

    if len(suite_names) == 1:
        return (
            tasks_payload[0]['data']['sr'][cfg.task_level],
            tasks_payload[0]['data']['cc'][cfg.task_level],
        )
    return tasks_payload


def save_rollout_video(
    rollout_images, idx, success, task_description, log_file=None, task_level=0
):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f'./rollouts/openpi/{DATE}'
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = (
        task_description.lower()
        .replace(' ', '_')
        .replace('\n', '_')
        .replace('.', '_')[:50]
    )
    mp4_path = f'{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--level={task_level}--task={processed_task_description}.mp4'
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f'Saved rollout MP4 at path {mp4_path}')
    if log_file is not None:
        log_file.write(f'Saved rollout MP4 at path {mp4_path}\n')
    return mp4_path


def get_vla_arena_env(
    task,
    resolution=256,
    add_noise=False,
    randomize_color=False,
    adjust_light=False,
    camera_offset=False,
    blur=False,
    layout_random=False,
    task_bddl_file_override: str | pathlib.Path | None = None,
    task_description_override: str | None = None,
):
    """Initializes and returns the VLA_ARENA environment, along with the task description."""
    task_description = (
        task_description_override
        if task_description_override is not None
        else task.language
    )
    task_bddl_file = (
        os.fspath(task_bddl_file_override)
        if task_bddl_file_override is not None
        else os.path.join(
            get_vla_arena_path('bddl_files'),
            task.problem_folder,
            f'level_{task.level}',
            task.bddl_file,
        )
    )
    env_args = {
        'bddl_file_name': task_bddl_file,
        'camera_heights': resolution,
        'camera_widths': resolution,
        'camera_offset': camera_offset,
        'color_randomize': randomize_color,
        'add_noise': add_noise,
        'light_adjustment': adjust_light,
        'blur': blur,
        'layout_random': layout_random,
    }
    env = OffScreenRenderEnv(**env_args)
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main(cfg=None):
    """
    Main entry point for evaluation.

    Args:
        cfg: Can be:
            - GenerateConfig: Use provided config object
            - str/Path: Path to config YAML file
            - None: Use CLI arguments via tyro
    """
    # Handle config loading from file path
    if isinstance(cfg, (str, pathlib.Path)):
        config_path = pathlib.Path(cfg)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found at: {config_path}')

        logger.info(f'Loading configuration from {config_path}...')

        # Load YAML file
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f)

        if not isinstance(yaml_data, dict):
            raise ValueError(
                f'Config file must contain a YAML dictionary, got {type(yaml_data)}'
            )

        # Convert YAML dict to command-line arguments for tyro
        def dict_to_args(prefix: str, d: dict) -> list[str]:
            """Recursively convert nested dict to tyro command line args."""
            args = []
            for key, value in d.items():
                full_key = f'{prefix}.{key}' if prefix else key
                if isinstance(value, dict):
                    # Recursively handle nested dicts
                    args.extend(dict_to_args(full_key, value))
                elif isinstance(value, (list, tuple)):
                    # Handle lists/tuples
                    args.append(
                        f"--{full_key}={','.join(str(v) for v in value)}"
                    )
                elif isinstance(value, bool):
                    # Handle booleans
                    # tyro uses --flag for True and --no-flag for False
                    if value:
                        args.append(f'--{full_key}')
                    else:
                        # Convert add_noise to no-add-noise format
                        args.append(f'--no-{full_key}')
                elif value is None:
                    # Skip None values
                    continue
                else:
                    args.append(f'--{full_key}={value}')
            return args

        # Build command line args from yaml
        original_argv = sys.argv.copy()
        try:
            args_list = dict_to_args('', yaml_data)

            # Temporarily modify sys.argv to pass args to tyro
            sys.argv = ['evaluator.py'] + args_list
            config_obj = tyro.cli(GenerateConfig)
        finally:
            # Restore original argv
            sys.argv = original_argv

        logger.info(f'Config loaded successfully from {config_path}')
        return eval_vla_arena(config_obj)

    if isinstance(cfg, GenerateConfig):
        # Use provided config object directly
        return eval_vla_arena(cfg)

    if cfg is None:
        # Default behavior: use CLI
        return eval_vla_arena(tyro.cli(GenerateConfig))

    raise ValueError(
        f'Unsupported config type: {type(cfg)}. Expected GenerateConfig, str, Path, or None.'
    )


if __name__ == '__main__':
    tyro.cli(main)
