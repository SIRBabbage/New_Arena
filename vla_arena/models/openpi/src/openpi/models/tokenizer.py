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

import logging
import os
import pathlib

import jax
import numpy as np
import vla_arena.models.openpi.src.openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import vla_arena.models.openpi.src.openpi.shared.download as download
import orbax.checkpoint as ocp
import sentencepiece
from scipy.fft import idct
from transformers import AutoProcessor

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download(
            'gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'}
        )
        with path.open('rb') as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=f.read()
            )

    def tokenize(
        self, prompt: str, state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace('_', ' ').replace('\n', ' ')
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = (
                np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            )
            state_str = ' '.join(map(str, discretized_state))
            full_prompt = (
                f'Task: {cleaned_text}, State: {state_str};\nAction: '
            )
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(
                cleaned_text, add_bos=True
            ) + self._tokenizer.encode('\n')
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f'Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. '
                    'Consider increasing the `max_token_len` in your model config if this happens frequently.'
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class FASTTokenizer:
    def __init__(
        self,
        max_len: int = 256,
        fast_tokenizer_path: str = 'physical-intelligence/fast',
        relaxed_decoding: bool = True,
    ):
        self._max_len = max_len
        self._relaxed_decoding = relaxed_decoding

        # Download base PaliGemma tokenizer
        path = download.maybe_download(
            'gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'}
        )
        with path.open('rb') as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=f.read()
            )

        # Resolve the FAST tokenizer repo to a local path first so downloads go
        # through huggingface_hub (which respects HF_ENDPOINT / local cache)
        # instead of relying on Transformers to resolve remote code directly.
        resolved_fast_tokenizer_path = self._resolve_fast_tokenizer_path(
            fast_tokenizer_path
        )

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(
            resolved_fast_tokenizer_path, trust_remote_code=True
        )
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens
        self._action_prefix_tokens = tuple(
            self._paligemma_tokenizer.encode('Action: ')
        )
        pipe_tokens = self._paligemma_tokenizer.encode('|', add_eos=True)
        self._action_stop_tokens = frozenset(pipe_tokens)

    @staticmethod
    def _resolve_fast_tokenizer_path(fast_tokenizer_path: str) -> str:
        path = pathlib.Path(os.path.expanduser(fast_tokenizer_path))
        if path.exists():
            return str(path.resolve())

        if snapshot_download is None:
            return fast_tokenizer_path

        return snapshot_download(
            repo_id=fast_tokenizer_path,
            repo_type='model',
        )

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace('_', ' ')

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = (
            np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        )

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = ' '.join(map(str, discretized_state))
        prefix = f'Task: {cleaned_text}, State: {state_str};\n'
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(
                action_tokens
            )

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode('Action: ')
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode('|', add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(
            postfix_tokens
        )  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f'Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. '
                    'Consider increasing the `max_token_len` in your model config if this happens frequently.'
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return (
            np.asarray(tokens),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
        )

    def extract_actions(
        self, tokens: np.ndarray, action_horizon: int, action_dim: int
    ) -> np.ndarray:
        action_tokens = self._extract_fast_action_token_ids(tokens)
        if action_tokens.size == 0:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        return self._decode_action_tokens(
            action_tokens,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )

    def _extract_fast_action_token_ids(self, tokens: np.ndarray) -> np.ndarray:
        """Extract FAST tokenizer ids directly from generated PaliGemma tokens.

        The model is trained on the raw token sequence:

          "Action: " + <FAST-action-tokens-mapped-into-PaliGemma-vocab> + "|" + eos

        Parsing this structure directly is more faithful than decoding to text and
        re-encoding with SentencePiece, and it lets us distinguish generation
        truncation from text round-trip issues.
        """
        token_list = np.asarray(tokens, dtype=np.int32).tolist()
        prefix = list(self._action_prefix_tokens)
        prefix_len = len(prefix)
        start_idx = None
        for idx in range(0, max(0, len(token_list) - prefix_len + 1)):
            if token_list[idx : idx + prefix_len] == prefix:
                start_idx = idx + prefix_len
                break
        if start_idx is None:
            return np.array([], dtype=np.int32)

        payload: list[int] = []
        for token in token_list[start_idx:]:
            token_int = int(token)
            if token_int == 0 or token_int in self._action_stop_tokens:
                break
            payload.append(token_int)

        if not payload:
            return np.array([], dtype=np.int32)
        return self._act_tokens_to_paligemma_tokens(
            np.asarray(payload, dtype=np.int32)
        ).astype(np.int32)

    def _act_tokens_to_paligemma_tokens(
        self, tokens: np.ndarray | list[int]
    ) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return (
            self._paligemma_tokenizer.vocab_size()
            - 1
            - self._fast_skip_tokens
            - tokens
        )

    def _decode_action_tokens(
        self,
        action_tokens: np.ndarray,
        *,
        action_horizon: int,
        action_dim: int,
    ) -> np.ndarray:
        if not self._relaxed_decoding:
            return self._fast_tokenizer.decode(
                [action_tokens.tolist()],
                time_horizon=action_horizon,
                action_dim=action_dim,
            )[0]

        decoded_tokens = self._fast_tokenizer.bpe_tokenizer.decode(
            action_tokens.tolist()
        )
        decoded_dct_coeff = (
            np.array(list(map(ord, decoded_tokens)), dtype=np.float32)
            + self._fast_tokenizer.min_token
        )
        expected_seq_len = action_horizon * action_dim
        diff = expected_seq_len - decoded_dct_coeff.shape[0]
        if diff < 0:
            decoded_dct_coeff = decoded_dct_coeff[:expected_seq_len]
        elif diff > 0:
            decoded_dct_coeff = np.pad(
                decoded_dct_coeff,
                (0, diff),
                mode='constant',
                constant_values=0,
            )

        decoded_dct_coeff = decoded_dct_coeff.reshape(
            action_horizon, action_dim
        )
        return idct(
            decoded_dct_coeff / self._fast_tokenizer.scale,
            axis=0,
            norm='ortho',
        ).astype(np.float32)


###########################################################################
## The tokenizers below are used for RoboArena baseline implementations. ##
## They are *not* used for pi0-style models.                             ##
###########################################################################


class BinningTokenizer:
    """
    Standard RT-2 / OpenVLA style binning tokenizer.
    """

    def __init__(self, max_len: int = 256, n_bins: int = 256):
        self._max_len = max_len
        self._n_bins = n_bins

        # Download base PaliGemma tokenizer
        path = download.maybe_download(
            'gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'}
        )
        with path.open('rb') as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=f.read()
            )

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a prompt and state into a sequence of tokens.

        Args:
            prompt: The text prompt to tokenize.
            state: The state array to discretize and tokenize.
            actions: Must be None. Action encoding is not currently supported.

        Returns:
            A tuple of (tokens, token_mask, ar_mask, targets).

        Raises:
            NotImplementedError: If actions is not None.
        """
        cleaned_text = prompt.lower().strip().replace('_', ' ')

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = (
            np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        )

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = ' '.join(map(str, discretized_state))
        prefix = f'Task: {cleaned_text}, State: {state_str};\n'
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError(
                'BinningTokenizer does not support encoding actions atm (only for inference use)'
            )
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(
            postfix_tokens
        )  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f'Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. '
                    'Consider increasing the `max_token_len` in your model config if this happens frequently.'
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return (
            np.asarray(tokens),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
        )

    def extract_actions(
        self, tokens: np.ndarray, action_horizon: int, action_dim: int
    ) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if 'Action: ' not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(
                decoded_tokens.split('Action: ')[1].split('|')[0].strip()
            )
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        if len(action_tokens) < action_horizon * action_dim:
            return np.zeros([action_horizon, action_dim], dtype=np.float32)
        action_tokens = action_tokens[: (action_horizon * action_dim)].reshape(
            [action_horizon, action_dim]
        )
        return action_tokens / self._n_bins * 2 - 1

    def _act_tokens_to_paligemma_tokens(
        self, tokens: np.ndarray | list[int]
    ) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return (
            self._paligemma_tokenizer.vocab_size()
            - 1
            - self._fast_skip_tokens
            - tokens
        )


class FSQTokenizer:
    """
    FSQ tokenizer from the FAST paper baselines.
    """

    def __init__(
        self, max_len: int = 256, fsq_tokenizer_path: str | None = None
    ):
        self._max_len = max_len

        assert (
            fsq_tokenizer_path is not None
        ), 'fsq_tokenizer_path must be provided'
        # Download tokenizer
        path = download.maybe_download(fsq_tokenizer_path)
        tok_path = os.path.join(path, os.listdir(path)[0])

        # Split step from path
        step = int(tok_path.split('/')[-1])
        base_path = tok_path.rsplit('/', 1)[0]

        mgr = ocp.CheckpointManager(
            base_path,
            item_handlers={
                'params': ocp.StandardCheckpointHandler(),
                'opt_state': ocp.StandardCheckpointHandler(),
                'config': ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )

        try:
            restored = mgr.restore(
                step,
                args=ocp.args.Composite(
                    config=ocp.args.JsonRestore(),
                    params=ocp.args.StandardRestore(),
                ),
            )
            config = restored['config']
            self._params = restored['params']
            self._fsq_tokenizer = fsq_tokenizer.FsqAttentionTokenizer(**config)
        except Exception as e:
            raise RuntimeError(
                f'Failed to load FSQ tokenizer checkpoint from {fsq_tokenizer_path}. Error: {e!s}'
            ) from e

        # Compile tokenize and detokenize functions
        self._tokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply(
                {'params': params}, x, method=self._fsq_tokenizer.tokenize
            )
        )
        self._detokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply(
                {'params': params}, x, method=self._fsq_tokenizer.detokenize
            )
        )

        # Download base PaliGemma tokenizer
        path = download.maybe_download(
            'gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'}
        )
        with path.open('rb') as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=f.read()
            )

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace('_', ' ')

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = (
            np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        )

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = ' '.join(map(str, discretized_state))
        prefix = f'Task: {cleaned_text}, State: {state_str};\n'
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError(
                'FSQTokenizer does not support encoding actions atm (only for inference use)'
            )
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(
            postfix_tokens
        )  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f'Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. '
                    'Consider increasing the `max_token_len` in your model config if this happens frequently.'
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return (
            np.asarray(tokens),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
        )

    def extract_actions(
        self, tokens: np.ndarray, action_horizon: int, action_dim: int
    ) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if 'Action: ' not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(
                decoded_tokens.split('Action: ')[1].split('|')[0].strip()
            )
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        try:
            # Move computation to CPU and compile on-demand
            device = jax.devices('cpu')[0]
            with jax.default_device(device):
                detok_act = self._detokenize_fn(
                    self._params, action_tokens[None, ...]
                )[0]
            return detok_act[: action_horizon * action_dim].reshape(
                [action_horizon, action_dim]
            )
        except Exception as e:
            logging.warning(f'Error decoding FSQ: {e}')
            return np.zeros((action_horizon, action_dim))

    def _act_tokens_to_paligemma_tokens(
        self, tokens: np.ndarray | list[int]
    ) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return (
            self._paligemma_tokenizer.vocab_size()
            - 1
            - self._fast_skip_tokens
            - tokens
        )
