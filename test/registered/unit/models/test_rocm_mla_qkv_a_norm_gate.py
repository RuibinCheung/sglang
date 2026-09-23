"""The gate that picks the fused MLA qkv_a GEMM + RMSNorm kernel.

The kernel replaces the qkv_a projection and both latent norms for small
decode and verify batches on gfx950. Every term below guards an input the
kernel does not implement, so dropping one would route such an input through
it and return a plausible but wrong q_lora / k_nope / k_rope.

The forward-invariant terms are evaluated once and cached on the module; the
per-batch terms are evaluated on every call.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ELIGIBLE = forward_mla_rocm._mla_qkv_a_norm_eligible
_MAX_M = 16


def _proj(weight=None, **extra):
    if weight is None:
        weight = torch.empty(8, 4, dtype=torch.bfloat16)
    return SimpleNamespace(weight=weight, **extra)


def _attn(**overrides):
    """An attention module every forward-invariant term accepts."""
    attn = SimpleNamespace(
        has_fused_proj=True,
        fused_qkv_a_proj_with_mqa=_proj(),
        q_a_layernorm=SimpleNamespace(variance_epsilon=1e-6),
        kv_a_layernorm=SimpleNamespace(variance_epsilon=1e-6),
        _mla_qkv_a_norm_static_ok=None,
    )
    for k, v in overrides.items():
        setattr(attn, k, v)
    return attn


def _batch(mode=ForwardMode.DECODE):
    return SimpleNamespace(forward_mode=mode)


def _hidden(m=4, **kwargs):
    return torch.empty(m, 4, dtype=kwargs.pop("dtype", torch.bfloat16), **kwargs)


def _fail(*_args, **_kwargs):
    raise AssertionError("gate consulted a term after the platform ruled it out")


class TestMlaQkvANormGate(CustomTestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"SGLANG_ROCM_MLA_QKV_A_NORM": "1"})
        env.start()
        self.addCleanup(env.stop)

        self.deterministic = SimpleNamespace(enable_deterministic_inference=False)
        self.tp_context = SimpleNamespace(input_scattered=False)
        self.piecewise = False
        # The platform terms short-circuit everything else; patch them so the
        # remaining terms are reachable off gfx950. The runtime getters need an
        # initialized server, so stand in for them too.
        for name, value in (
            ("_is_hip", True),
            ("_is_gfx95_supported", True),
            ("_MLA_QKV_A_NORM_MAX_M", _MAX_M),
            ("get_exec", lambda: SimpleNamespace(deterministic=self.deterministic)),
            ("get_attn_tp_context", lambda: self.tp_context),
            ("is_in_tc_piecewise_cuda_graph", lambda: self.piecewise),
        ):
            patcher = patch.object(forward_mla_rocm, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _eligible(self, attn=None, hidden=None, batch=None, q_replicate=False):
        return _ELIGIBLE(
            attn if attn is not None else _attn(),
            hidden if hidden is not None else _hidden(),
            batch if batch is not None else _batch(),
            q_replicate,
        )

    def test_takes_the_fused_path_when_every_term_holds(self):
        for mode in (ForwardMode.DECODE, ForwardMode.IDLE, ForwardMode.TARGET_VERIFY):
            for m in (1, 3, _MAX_M):
                with self.subTest(mode=mode.name, m=m):
                    self.assertTrue(
                        self._eligible(hidden=_hidden(m), batch=_batch(mode))
                    )

    def test_off_hip_nothing_else_is_consulted(self):
        forward_mla_rocm._is_hip = False
        forward_mla_rocm.get_exec = _fail
        forward_mla_rocm.get_attn_tp_context = _fail
        forward_mla_rocm.is_in_tc_piecewise_cuda_graph = _fail
        # Only the cache slot exists; reading any other attribute would raise.
        attn = SimpleNamespace(_mla_qkv_a_norm_static_ok=None)
        self.assertFalse(_ELIGIBLE(attn, None, object(), False))
        self.assertIs(attn._mla_qkv_a_norm_static_ok, False)

    def test_hip_but_not_gfx950_turns_it_off(self):
        # The kernel uses gfx950-only MFMA; an explicit opt-in must not reach it.
        forward_mla_rocm._is_gfx95_supported = False
        self.assertFalse(self._eligible())

    def test_each_forward_invariant_term_alone_turns_it_off(self):
        # The kernel reads a contiguous BF16 weight, applies one epsilon to both
        # norms, and has no LoRA or deterministic variant.
        for name, attn in (
            ("has_fused_proj", _attn(has_fused_proj=False)),
            (
                "weight dtype",
                _attn(
                    fused_qkv_a_proj_with_mqa=_proj(
                        torch.empty(8, 4, dtype=torch.float8_e4m3fn)
                    )
                ),
            ),
            (
                "weight contiguity",
                _attn(
                    fused_qkv_a_proj_with_mqa=_proj(
                        torch.empty(4, 8, dtype=torch.bfloat16).t()
                    )
                ),
            ),
            (
                "norm epsilon",
                _attn(kv_a_layernorm=SimpleNamespace(variance_epsilon=1e-5)),
            ),
            ("lora", _attn(fused_qkv_a_proj_with_mqa=_proj(set_lora=True))),
        ):
            with self.subTest(term=name):
                self.assertFalse(self._eligible(attn=attn))

        with self.subTest(term="flag"):
            with patch.dict(os.environ, {"SGLANG_ROCM_MLA_QKV_A_NORM": "0"}):
                self.assertFalse(self._eligible())

        with self.subTest(term="deterministic inference"):
            self.deterministic.enable_deterministic_inference = True
            self.assertFalse(self._eligible())

    def test_each_per_batch_term_alone_turns_it_off(self):
        hidden = _hidden()
        for name, kwargs in (
            ("empty batch", dict(hidden=_hidden(0))),
            ("too many rows", dict(hidden=_hidden(_MAX_M + 1))),
            ("prefill", dict(batch=_batch(ForwardMode.EXTEND))),
            ("mixed", dict(batch=_batch(ForwardMode.MIXED))),
            # Draft extend runs at bs * steps rows but is not validated yet.
            ("draft extend", dict(batch=_batch(ForwardMode.DRAFT_EXTEND_V2))),
            # An upstream fused quant hands over (data, scale).
            ("quantized input", dict(hidden=(hidden, hidden))),
            ("3D input", dict(hidden=hidden.unsqueeze(0))),
            ("input dtype", dict(hidden=_hidden(dtype=torch.float32))),
            ("input contiguity", dict(hidden=torch.empty(4, 4).bfloat16().t())),
            ("replicated q_b_proj", dict(q_replicate=True)),
        ):
            with self.subTest(term=name):
                self.assertFalse(self._eligible(**kwargs))

        with self.subTest(term="piecewise cuda graph"):
            self.piecewise = True
            self.assertFalse(self._eligible())
            self.piecewise = False

        with self.subTest(term="scattered attention input"):
            self.tp_context.input_scattered = True
            self.assertFalse(self._eligible())

    def test_forward_invariant_terms_are_cached(self):
        attn = _attn()
        self.assertTrue(self._eligible(attn=attn))
        attn.has_fused_proj = False
        self.assertTrue(self._eligible(attn=attn))
        # The per-batch terms are still evaluated on every call.
        self.assertFalse(self._eligible(attn=attn, hidden=_hidden(_MAX_M + 1)))


if __name__ == "__main__":
    unittest.main()
