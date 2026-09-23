"""model_hook turns the fused MLA qkv_a kernels on for GLM-5.2 on gfx950 only.

The kernels are validated on GLM-5.2's MLA shapes and use gfx950 MFMA, so every
other model or device keeps the flag off, and an explicit setting always wins.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import model_hook
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_FLAG = envs.SGLANG_ROCM_MLA_QKV_A_NORM


def _config(arch):
    return SimpleNamespace(architectures=[arch])


class TestRocmMlaQkvANormDefault(CustomTestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        _FLAG.clear()

    def _apply(self, arch, *, gfx95=True):
        with patch.object(model_hook, "is_gfx95_supported", return_value=gfx95):
            model_hook._default_rocm_mla_qkv_a_norm(_config(arch))

    def test_glm_on_gfx950_turns_flag_on(self):
        for arch in ("GlmMoeDsaForCausalLM", "GlmMoeDsaForCausalLMNextN"):
            with self.subTest(arch=arch):
                _FLAG.clear()
                self._apply(arch)
                self.assertTrue(_FLAG.is_set())
                self.assertTrue(_FLAG.get())

    def test_explicit_off_is_kept(self):
        _FLAG.set(False)
        self._apply("GlmMoeDsaForCausalLM")
        self.assertFalse(_FLAG.get())

    def test_other_dsa_model_stays_off(self):
        self._apply("DeepseekV32ForCausalLM")
        self.assertFalse(_FLAG.is_set())
        self.assertFalse(_FLAG.get())

    def test_non_gfx950_stays_off(self):
        # is_gfx95_supported() is also False on every non-HIP build.
        self._apply("GlmMoeDsaForCausalLM", gfx95=False)
        self.assertFalse(_FLAG.is_set())
        self.assertFalse(_FLAG.get())


if __name__ == "__main__":
    unittest.main()
