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

import numpy as np
from vla_arena.models.openpi.src.openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize('Hello, world!')

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_fast_tokenizer():
    prompt = 'Hello, world!'
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(
        prompt, state, action
    )

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)


class _FakeBPETokenizer:
    def __init__(self, decoded_text: str):
        self.decoded_text = decoded_text

    def decode(self, _tokens):
        return self.decoded_text


class _FakeFastProcessor:
    def __init__(self, decoded_text: str, *, scale: float = 10, min_token: int = 0):
        self.bpe_tokenizer = _FakeBPETokenizer(decoded_text)
        self.scale = scale
        self.min_token = min_token
        self.decode_calls = []

    def decode(self, token_batches, *, time_horizon: int, action_dim: int):
        self.decode_calls.append(
            (token_batches, time_horizon, action_dim)
        )
        return [
            np.arange(time_horizon * action_dim, dtype=np.float32).reshape(
                time_horizon, action_dim
            )
        ]


def test_fast_tokenizer_extract_actions_uses_fast_processor_decode_when_strict():
    tokenizer = _tokenizer.FASTTokenizer.__new__(_tokenizer.FASTTokenizer)
    tokenizer._fast_tokenizer = _FakeFastProcessor('ignored')
    tokenizer._paligemma_tokenizer = _FakePaligemmaTokenizer()
    tokenizer._fast_skip_tokens = 128
    tokenizer._action_prefix_tokens = (11, 12, 13)
    tokenizer._action_stop_tokens = frozenset({99, 1})
    tokenizer._relaxed_decoding = False

    raw_tokens = np.array([11, 12, 13, 861, 851, 99, 1, 0], dtype=np.int32)
    actions = tokenizer.extract_actions(
        raw_tokens,
        action_horizon=3,
        action_dim=2,
    )

    assert actions.shape == (3, 2)
    assert tokenizer._fast_tokenizer.decode_calls == [
        ([[10, 20]], 3, 2)
    ]


def test_fast_tokenizer_relaxed_decode_pads_short_sequences():
    tokenizer = _tokenizer.FASTTokenizer.__new__(_tokenizer.FASTTokenizer)
    tokenizer._fast_tokenizer = _FakeFastProcessor('A' * 68)
    tokenizer._relaxed_decoding = True

    actions = tokenizer._decode_action_tokens(
        np.array([1, 2, 3], dtype=np.int32),
        action_horizon=10,
        action_dim=7,
    )

    assert actions.shape == (10, 7)
    assert np.isfinite(actions).all()


class _FakePaligemmaTokenizer:
    def encode(self, text, add_eos=False):
        if text == 'Action: ':
            return [11, 12, 13]
        if text == '|' and add_eos:
            return [99, 1]
        raise AssertionError((text, add_eos))

    def vocab_size(self):
        return 1000


def test_fast_tokenizer_extracts_action_ids_from_raw_tokens():
    tokenizer = _tokenizer.FASTTokenizer.__new__(_tokenizer.FASTTokenizer)
    tokenizer._paligemma_tokenizer = _FakePaligemmaTokenizer()
    tokenizer._fast_skip_tokens = 128
    tokenizer._action_prefix_tokens = (11, 12, 13)
    tokenizer._action_stop_tokens = frozenset({99, 1})

    # Inverse of _act_tokens_to_paligemma_tokens with vocab_size=1000 and skip=128.
    raw_tokens = np.array([11, 12, 13, 861, 851, 99, 1, 0], dtype=np.int32)

    fast_ids = tokenizer._extract_fast_action_token_ids(raw_tokens)

    assert fast_ids.tolist() == [10, 20]
