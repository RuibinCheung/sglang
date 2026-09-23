"""The fused MLA qkv_a GEMM + RMSNorm must match an FP32 reference for every decode row count."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

# GLM-5.2's MLA latent shapes, the only ones the kernels are tuned and validated on.
_HIDDEN, _Q_LORA, _KV_LORA, _ROPE = 6144, 2048, 512, 64
_EPS = 1e-5
# Max error relative to max |reference|; GLM-5.2's real weights reach 3.3e-3.
_TOL = 5e-3


def _reference(x, weight, q_gamma, kv_gamma):
    # The kernel rounds the projection to BF16 once and normalizes in FP32.
    y = (x.float() @ weight.float().t()).bfloat16().float()
    q, kv, rope = y.split([_Q_LORA, _KV_LORA, _ROPE], dim=-1)

    def rms(v, gamma):
        return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + _EPS) * gamma.float()

    return rms(q, q_gamma), rms(kv, kv_gamma), rope


def _rel_err(out, ref):
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "the kernels use gfx950-only MFMA",
)
class TestMlaQkvANormGfx950(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention.mla_qkv_a_norm_gfx950 import (
            MAX_M,
            mla_qkv_a_norm,
        )

        cls.max_m = MAX_M
        cls.op = staticmethod(mla_qkv_a_norm)
        g = torch.Generator(device="cuda").manual_seed(0)
        n = _Q_LORA + _KV_LORA + _ROPE
        cls.weight = (
            torch.randn(n, _HIDDEN, device="cuda", generator=g) * 0.02
        ).bfloat16()
        cls.q_gamma = (torch.rand(_Q_LORA, device="cuda", generator=g) + 0.5).bfloat16()
        cls.kv_gamma = (
            torch.rand(_KV_LORA, device="cuda", generator=g) + 0.5
        ).bfloat16()

    def _hidden(self, m, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(m, _HIDDEN, device="cuda", generator=g).bfloat16()

    def _run(self, x):
        return self.op(
            x, self.weight, self.q_gamma, self.kv_gamma, rope_dim=_ROPE, eps=_EPS
        )

    def test_matches_fp32_reference_for_every_m(self):
        """Every row count selects a variant; non-powers of two run a masked tail."""
        for m in range(1, self.max_m + 1):
            with self.subTest(m=m):
                x = self._hidden(m, seed=m)
                outs = self._run(x)
                refs = _reference(x, self.weight, self.q_gamma, self.kv_gamma)
                for name, out, ref, width in zip(
                    ("q_lora", "k_nope", "k_rope"),
                    outs,
                    refs,
                    (_Q_LORA, _KV_LORA, _ROPE),
                ):
                    self.assertEqual(out.shape, (m, width), name)
                    self.assertEqual(out.dtype, torch.bfloat16, name)
                    self.assertTrue(out.is_contiguous(), name)
                    self.assertLessEqual(_rel_err(out, ref), _TOL, name)

    def test_rejects_row_counts_outside_the_variants(self):
        for m in (0, self.max_m + 1):
            with self.subTest(m=m), self.assertRaises(ValueError):
                self._run(self._hidden(m, seed=0))

    def test_cuda_graph_replay_matches_eager(self):
        """Decode runs the kernel inside captured graphs, reading new rows each replay."""
        for m in range(1, self.max_m + 1):
            with self.subTest(m=m):
                x = self._hidden(m, seed=m)
                self._run(x)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = self._run(x)
                x.copy_(self._hidden(m, seed=m + 100))
                graph.replay()
                eager = self._run(x)
                torch.cuda.synchronize()
                for out, ref in zip(captured, eager):
                    self.assertTrue(torch.equal(out, ref))


if __name__ == "__main__":
    unittest.main()
