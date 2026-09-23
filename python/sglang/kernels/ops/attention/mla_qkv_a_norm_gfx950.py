# SPDX-License-Identifier: Apache-2.0
"""Fused MLA q/kv-a projection + both latent RMSNorms for decode shapes on gfx950.

    q_lora, k_nope, k_rope = rmsnorm_q(x @ Wq_a^T), rmsnorm_kv(x @ Wkv_a^T), x @ Wk_rope^T

where ``weight`` is the packed ``fused_qkv_a_proj_with_mqa`` matrix. Each variant is a
split-K Gluon GEMM whose FP32 partials are reduced together with the two RMSNorms, so
the projection is rounded to BF16 once. Variants are tuned per row count (M <= 16) on
GLM-5.2's shapes (N=2624, K=6144) and selected by ``mla_qkv_a_norm``.

Vendored from OpenAI-Partners/artemis-kernel-integrations@14eb5a6; the top-level
functions of each source file carry the prefix of the variant they came from:
    _m1_  qkv_a_norm_tp4_m1.py  sha256=e31cf4490550d0c17409a1335a0e7e719e1c88cf254b7874cc72e08b7bafcf50
    _m2_  qkv_a_norm_tp4_m2.py  sha256=c13ae7b5ff288266c0fa340e3dcb0f4c782f363a7fb6bed661ccc87ca600a5e2
    _m4_  qkv_a_norm_tp4_m4.py  sha256=0f08b44f81c7cf11eebf02b6333d2af1c6e20e3308d3a22094fa76457ca98c0d
    _m8_  qkv_a_norm_tp4_m8.py  sha256=544963fda4285958444b594508ecf640c923b3bc0031f9f8c683819e5d1fb1f6
    _m16_  qkv_a_norm_tp8_m16.py  sha256=2eab983578fc9116cb3114a4a91f4640475c939151f057a42fa62ba17573e57d
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@triton.constexpr_function
def _artifact_next_power_of_2(n):
    return 1 << (n - 1).bit_length()


# ---- variant M <= 1: qkv_a_norm_tp4_m1.py ----


@gluon.jit
def _m1_tile_and_split(
    pid,
    N: gl.constexpr,
    BN: gl.constexpr,
    SPLITS: gl.constexpr,
    GROUP: gl.constexpr,
    STRIPES: gl.constexpr,
):
    TOTAL: gl.constexpr = gl.cdiv(N, BN * GROUP) * GROUP * SPLITS
    if STRIPES > 1:
        stripe = pid % STRIPES
        pid = (
            stripe * (TOTAL // STRIPES)
            + gl.minimum(stripe, TOTAL % STRIPES)
            + pid // STRIPES
        )
    tile = pid // (GROUP * SPLITS) * GROUP + pid % GROUP
    split = pid // GROUP % SPLITS
    return (tile, split)


@gluon.jit
def _m1_store_partial(
    P, acc, tile, split, M: gl.constexpr, N: gl.constexpr, PITCH: gl.constexpr
):
    BM: gl.constexpr = acc.type.shape[0]
    BN: gl.constexpr = acc.type.shape[1]
    mma: gl.constexpr = acc.type.layout
    om = gl.arange(0, BM, gl.SliceLayout(1, mma))
    on = gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.amd.cdna4.buffer_store(
        ptr=P + split * M * PITCH + tile * BN,
        offsets=om[:, None] * PITCH + on[None, :],
        stored_value=acc,
        mask=(om[:, None] < M) & (tile * BN + on[None, :] < N),
        cache=".wt",
    )


@gluon.jit
def _m1_copy_panel(a_slot, b_slot, X, W, a_offsets, b_offsets, k_start):
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a_slot, X, a_offsets + k_start, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b_slot, W, b_offsets + k_start, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _m1_project_async(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    PITCH: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    SPLITS: gl.constexpr,
    NW: gl.constexpr,
    STAGES: gl.constexpr,
    STRIPES: gl.constexpr,
):
    gl.static_assert(N % BN == 0 and K % (BK * SPLITS) == 0)
    pid = gl.program_id(0).to(gl.uint32)
    tile, split = _m1_tile_and_split(pid, N, BN, SPLITS, 1, STRIPES)
    CHUNKS: gl.constexpr = gl.cdiv(K, BK * SPLITS)
    DEPTH: gl.constexpr = min(STAGES, CHUNKS)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=M != 16, warps_per_cta=[1, NW]
    )
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    linear: gl.constexpr = gl.BlockedLayout([8], [64], [NW], [0])
    a_linear: gl.constexpr = gl.BlockedLayout(
        [2 if M * BK < NW * 512 else 8], [64], [NW], [0]
    )
    flat_shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    a_shared: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
    b_shared: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [0, 1])
    a_storage = gl.allocate_shared_memory(gl.bfloat16, [DEPTH, M * BK], flat_shared)
    b_storage = gl.allocate_shared_memory(gl.bfloat16, [DEPTH, BN * BK], flat_shared)
    ai = gl.arange(0, M * BK, a_linear)
    bi = gl.arange(0, BN * BK, linear)
    am = ai // BK
    bn = tile * BN + bi // BK
    ak = ai % BK ^ am % 16 * 8
    bk = bi % BK ^ bi // BK % 16 * 8
    ak = gl.max_contiguous(gl.multiple_of(ak, 8), 8)
    bk = gl.max_contiguous(gl.multiple_of(bk, 8), 8)
    a_offsets = am * SX + ak
    b_offsets = bn * SW + bk
    start = split * CHUNKS * BK
    for stage in gl.static_range(DEPTH):
        _m1_copy_panel(
            a_storage.index(stage),
            b_storage.index(stage),
            X,
            W,
            a_offsets,
            b_offsets,
            start + stage * BK,
        )
    acc = gl.zeros((M, BN), gl.float32, mma)
    for i in gl.static_range(CHUNKS):
        gl.amd.cdna4.async_copy.wait_group(min(DEPTH - 1, CHUNKS - 1 - i))
        if NW > 1:
            gl.barrier()
        a_slot = a_storage.index(i % DEPTH)
        b_slot = b_storage.index(i % DEPTH)
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_slot._reinterpret(shape=[M, BK], layout=a_shared), ad
        )
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_slot._reinterpret(shape=[BK, BN], layout=b_shared), bd
        )
        if i + DEPTH < CHUNKS:
            gl.inline_asm_elementwise(
                "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
                constraints="=v,~{memory}",
                args=[],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
            if NW > 1:
                gl.barrier()
            _m1_copy_panel(
                a_slot, b_slot, X, W, a_offsets, b_offsets, start + (i + DEPTH) * BK
            )
        acc = gl.amd.cdna4.mfma(a, b, acc)
    _m1_store_partial(P, acc, tile, split, M, N, PITCH)


@gluon.jit
def _m1_project_grouped(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    PITCH: gl.constexpr,
    SPLITS: gl.constexpr,
    STRIPES: gl.constexpr,
):
    BN: gl.constexpr = 64
    BK: gl.constexpr = 128
    CHUNKS: gl.constexpr = K // (BK * SPLITS)
    gl.static_assert(M == 4 or M == 8 or M == 16)
    gl.static_assert(CHUNKS == 8 and N % BN == 0)
    pid = gl.program_id(0).to(gl.uint32)
    tile, split = _m1_tile_and_split(pid, N, BN, SPLITS, 1, STRIPES)
    start = split * CHUNKS * BK
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=M != 16, warps_per_cta=[1, 4]
    )
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    ash: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
    bsh: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [0, 1])
    al: gl.constexpr = gl.BlockedLayout([8], [64], [4], [0])
    a_storage = gl.allocate_shared_memory(gl.bfloat16, [CHUNKS * M * BK], flat)
    ai = gl.arange(0, CHUNKS * M * BK, al)
    am = ai // BK % M
    ak = ai % BK ^ am % 16 * 8
    ak = gl.max_contiguous(gl.multiple_of(ak, 8), 8)
    a_offsets = am * SX + ak + ai // (M * BK) * BK
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a_storage, X, a_offsets + start, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.commit_group()
    COUNT: gl.constexpr = BN * BK
    bl: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1], [2], [4], [512], [1024]],
        lane_bases=[[8], [16], [32], [64], [128], [256]],
        warp_bases=[[2048], [4096]],
        block_bases=[],
        shape=[COUNT],
    )
    b0 = gl.allocate_shared_memory(gl.bfloat16, [COUNT], flat)
    b1 = gl.allocate_shared_memory(gl.bfloat16, [COUNT], flat)
    b2 = gl.allocate_shared_memory(gl.bfloat16, [COUNT], flat)
    b3 = gl.allocate_shared_memory(gl.bfloat16, [COUNT], flat)
    slots = (b0, b1, b2, b3)
    bi = gl.arange(0, COUNT, bl)
    bn = tile * BN + bi // BK
    bk = bi % BK ^ bi // BK % 16 * 8
    bk = gl.max_contiguous(gl.multiple_of(bk, 8), 8)
    b_offsets = bn * SW + bk + start
    for i in gl.static_range(4):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            slots[i], W, b_offsets + i * BK, cache_modifier=".ca"
        )
        gl.amd.cdna4.async_copy.commit_group()
    acc = gl.zeros((M, BN), gl.float32, mma)
    for i in gl.static_range(CHUNKS):
        if i == 0:
            gl.amd.cdna4.async_copy.wait_group(3)
            gl.barrier()
        else:
            gl.inline_asm_elementwise(
                "s_waitcnt vmcnt($1)\n v_mov_b32 $0, 0",
                constraints="=v,n,~{memory}",
                args=[4 * min(3, CHUNKS - 1 - i)],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_storage._reinterpret(shape=[CHUNKS, M * BK])
            .index(i)
            ._reinterpret(shape=[M, BK], layout=ash),
            ad,
        )
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            slots[i % 4]._reinterpret(dtype=gl.bfloat16, shape=[BK, BN], layout=bsh), bd
        )
        if i + 4 < CHUNKS:
            gl.inline_asm_elementwise(
                "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
                constraints="=v,~{memory}",
                args=[],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                slots[i % 4], W, b_offsets + (i + 4) * BK, cache_modifier=".ca"
            )
            gl.amd.cdna4.async_copy.commit_group()
        acc = gl.amd.cdna4.mfma(a, b, acc)
    _m1_store_partial(P, acc, tile, split, M, N, PITCH)


@gluon.jit
def _m1_project_prefetched(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    SPLITS: gl.constexpr,
    GROUP: gl.constexpr,
    STRIPES: gl.constexpr,
    NW: gl.constexpr,
):
    gl.static_assert(NW == 1 or NW == 2)
    gl.static_assert(N % (BN * GROUP) == 0 and K % (BK * SPLITS) == 0)
    pid = gl.program_id(0).to(gl.uint32)
    tile, split = _m1_tile_and_split(pid, N, BN, SPLITS, GROUP, STRIPES)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, NW]
    )
    al: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [NW, 1], [1, 0])
    if NW == 1:
        bl: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 1], [0, 1])
    else:
        bl: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
            lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]],
            warp_bases=[[0, 16]],
            block_bases=[],
            shape=[BK, BN],
        )
    am = gl.arange(0, BM, gl.SliceLayout(1, al))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    bn = gl.arange(0, BN, gl.SliceLayout(0, bl))
    CHUNKS: gl.constexpr = gl.cdiv(K, BK * SPLITS)
    x_base = X + split * CHUNKS * BK
    w_base = W + tile * BN * SW + split * CHUNKS * BK
    ma = gl.full((BM, BK), True, gl.int1, al)
    if BM > M:
        ma = ma & (am[:, None] < M)
    a = gl.amd.cdna4.buffer_load(x_base, am[:, None] * SX + ak[None, :], ma, 0)
    b = gl.amd.cdna4.buffer_load(w_base, bn[None, :] * SW + bk[:, None])
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for i in gl.static_range(CHUNKS):
        if i + 1 < CHUNKS:
            start = (i + 1) * BK
            next_a = gl.amd.cdna4.buffer_load(
                x_base, am[:, None] * SX + start + ak[None, :], ma, 0
            )
            next_b = gl.amd.cdna4.buffer_load(
                w_base, bn[None, :] * SW + start + bk[:, None]
            )
        a_dot = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
        b_dot = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))
        acc = gl.amd.cdna4.mfma(a_dot, b_dot, acc)
        if i + 1 < CHUNKS:
            a = next_a
            b = next_b
    _m1_store_partial(P, acc, tile, split, M, N, N)


@gluon.jit
def _m1_finish_part(
    P,
    G,
    O,
    eps,
    row,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    WIDTH: gl.constexpr,
    OFFSET: gl.constexpr,
    SPLITS: gl.constexpr,
    B: gl.constexpr,
    NORM: gl.constexpr,
    VEC: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [4], [0])
    d = gl.arange(0, B, layout)
    value = gl.full((B,), 0, gl.float32, layout)
    base = P + row * PITCH + OFFSET
    for s in gl.static_range(SPLITS):
        if VEC == 8:
            value += gl.amd.cdna4.buffer_load(base + s * M * PITCH, d, d < WIDTH, 0)
        else:
            value += gl.load(P + s * M * PITCH + row * PITCH + OFFSET + d, d < WIDTH, 0)
    if NORM:
        value = value.to(gl.bfloat16).to(gl.float32)
        gamma = gl.load(G + d, d < WIDTH, 0).to(gl.float32)
        inv_rms = gl.rsqrt(gl.sum(value * value, 0) / WIDTH + eps)
        value = value * inv_rms * gamma
    gl.store(O + row * WIDTH + d, value, d < WIDTH)


@gluon.jit
def _m1_finish(
    P,
    QG,
    KG,
    QO,
    KO,
    RO,
    eps,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    Q: gl.constexpr,
    KV: gl.constexpr,
    R: gl.constexpr,
    SPLITS: gl.constexpr,
    VEC: gl.constexpr,
):
    BQ: gl.constexpr = _artifact_next_power_of_2(Q)
    BKV: gl.constexpr = _artifact_next_power_of_2(KV)
    BR: gl.constexpr = _artifact_next_power_of_2(R)
    row = gl.program_id(0)
    if M == 16:
        row = row.to(gl.uint32)
    part = gl.program_id(1)
    if M <= 2:
        part = 2 - part
    if part == 0:
        _m1_finish_part(P, QG, QO, eps, row, M, PITCH, Q, 0, SPLITS, BQ, True, VEC)
    elif part == 1:
        _m1_finish_part(P, KG, KO, eps, row, M, PITCH, KV, Q, SPLITS, BKV, True, VEC)
    else:
        _m1_finish_part(
            P, QG, RO, eps, row, M, PITCH, R, Q + KV, SPLITS, BR, False, VEC
        )


def _m1_projection_config(m, k, n, sx, sw):
    use_async = (
        m in (1, 2, 4, 8, 16)
        and k % 768 == 0
        and (n % (16 if m <= 4 else 64) == 0)
        and (sx % 8 == 0)
        and (sw % 8 == 0)
    )
    pitch = n
    if use_async:
        bn = 16 if m <= 4 else 4 * m
        nw = 1 if m <= 4 else m // 4
        stages = 5 if m == 1 else 4 if m == 2 else 3
        group, stripes = (1, 8 if m <= 4 else 4)
        if k == 6144 and n == 2624:
            pitch = 4096 if m <= 4 else 3072 if m == 8 else 2816
    else:
        wide = m in (15, 16)
        bn, nw = (32, 2) if wide else (16, 1)
        stages = 1
        group = 4 if m <= 8 else 1
        stripes = 4 if wide else 1 if 5 <= m <= 8 else 8
    return (use_async, bn, nw, stages, group, stripes, pitch)


def _m1_mla_qkv_a_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    q_gamma: torch.Tensor,
    kv_gamma: torch.Tensor,
    *,
    rope_dim: int = 64,
    eps: float = 1e-05,
):
    m, k = hidden.shape
    q, kv = (q_gamma.numel(), kv_gamma.numel())
    n = q + kv + rope_dim
    sx, sw = (hidden.stride(0), weight.stride(0))
    use_async, bn, nw, stages, group, stripes, pitch = _m1_projection_config(
        m, k, n, sx, sw
    )
    splits = 6
    grouped = use_async and m in (4, 8, 16) and (k == 6144) and (n % 64 == 0)
    if grouped:
        bn, nw = (64, 4)
    partial = torch.empty((splits, m, pitch), device=hidden.device, dtype=torch.float32)
    qo = torch.empty((m, q), device=hidden.device, dtype=torch.bfloat16)
    ko = torch.empty((m, kv), device=hidden.device, dtype=torch.bfloat16)
    ro = torch.empty((m, rope_dim), device=hidden.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(n, bn * group) * group * splits,)
    if grouped:
        _m1_project_grouped[grid](
            hidden,
            weight,
            partial,
            M=m,
            N=n,
            K=k,
            SX=sx,
            SW=sw,
            PITCH=pitch,
            SPLITS=splits,
            STRIPES=stripes,
            num_warps=4,
        )
    elif use_async:
        _m1_project_async[grid](
            hidden,
            weight,
            partial,
            M=m,
            N=n,
            K=k,
            SX=sx,
            SW=sw,
            PITCH=pitch,
            BN=bn,
            BK=128,
            SPLITS=splits,
            NW=nw,
            STAGES=stages,
            STRIPES=stripes,
            num_warps=nw,
        )
    else:
        _m1_project_prefetched[grid](
            hidden,
            weight,
            partial,
            M=m,
            N=n,
            K=k,
            SX=sx,
            SW=sw,
            BM=_artifact_next_power_of_2(m),
            BN=bn,
            BK=128,
            SPLITS=splits,
            GROUP=group,
            STRIPES=stripes,
            NW=nw,
            num_warps=nw,
        )
    _m1_finish[m, 3](
        partial,
        q_gamma,
        kv_gamma,
        qo,
        ko,
        ro,
        eps,
        m,
        pitch,
        q,
        kv,
        rope_dim,
        splits,
        4 if m == 4 else 8,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return (qo, ko, ro)


# ---- variant M <= 2: qkv_a_norm_tp4_m2.py ----


@gluon.jit
def _m2_project_masked(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    SPLITS: gl.constexpr,
    GROUP: gl.constexpr,
    UNROLL: gl.constexpr,
    CHAINS: gl.constexpr,
):
    pid = gl.program_id(0)
    tile = pid // (GROUP * SPLITS) * GROUP + pid % GROUP
    split = pid // GROUP % SPLITS
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1, 1]
    )
    al: gl.constexpr = gl.BlockedLayout([1, 1, 8], [1, 4, 16], [1, 1, 1], [2, 1, 0])
    bl: gl.constexpr = gl.BlockedLayout([1, 8, 1], [1, 16, 4], [1, 1, 1], [1, 2, 0])
    ap = gl.arange(0, CHAINS, gl.SliceLayout(1, gl.SliceLayout(2, al)))
    am = gl.arange(0, BM, gl.SliceLayout(0, gl.SliceLayout(2, al)))
    ak = gl.arange(0, BK, gl.SliceLayout(0, gl.SliceLayout(1, al)))
    bp = gl.arange(0, CHAINS, gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    bk = gl.arange(0, BK, gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    bn = gl.arange(0, BN, gl.SliceLayout(0, gl.SliceLayout(1, bl)))
    CHUNKS: gl.constexpr = K // (BK * SPLITS * CHAINS)
    x_base = X + split * CHAINS * CHUNKS * BK
    w_base = W + tile * BN * SW + split * CHAINS * CHUNKS * BK
    acc = gl.zeros((CHAINS, BM, BN), gl.float32, mma)
    for block in range(gl.cdiv(CHUNKS, UNROLL)):
        for u in gl.static_range(UNROLL):
            i = block * UNROLL + u
            ka = ap[:, None, None] * CHUNKS * BK + i * BK + ak[None, None, :]
            kb = bp[:, None, None] * CHUNKS * BK + i * BK + bk[None, :, None]
            a = gl.amd.cdna4.buffer_load(
                x_base, am[None, :, None] * SX + ka, am[None, :, None] < M, 0
            )
            b = gl.amd.cdna4.buffer_load(w_base, bn[None, None, :] * SW + kb)
            a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
            b = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))
            acc = gl.amd.cdna4.mfma(a, b, acc)
    reduced = gl.sum(acc, 0)
    out: gl.constexpr = gl.SliceLayout(0, mma)
    om = gl.arange(0, BM, gl.SliceLayout(1, out))
    on = gl.arange(0, BN, gl.SliceLayout(0, out))
    gl.amd.cdna4.buffer_store(
        ptr=P + split * M * N + tile * BN,
        offsets=om[:, None] * N + on[None, :],
        stored_value=reduced,
        mask=om[:, None] < M,
    )


@gluon.jit
def _m2_copy_async_panel(AS, BS, X, W, ax, bw, start):
    gl.amd.cdna4.async_copy.buffer_load_to_shared(AS, X, ax + start)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(BS, W, bw + start)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _m2_project_async(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    PITCH: gl.constexpr,
    BK: gl.constexpr,
    SPLITS: gl.constexpr,
    STAGES: gl.constexpr,
    TRANSPOSED: gl.constexpr,
    STRIPES: gl.constexpr,
    BN: gl.constexpr,
    NW: gl.constexpr,
):
    pid = gl.program_id(0).to(gl.uint32)
    total: gl.constexpr = N // BN * SPLITS
    stripe = pid % STRIPES
    pid = (
        stripe * (total // STRIPES)
        + gl.minimum(stripe, total % STRIPES)
        + pid // STRIPES
    )
    tile = pid // SPLITS
    split = pid % SPLITS
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=TRANSPOSED,
        warps_per_cta=[1, NW],
    )
    CHUNKS: gl.constexpr = K // (BK * SPLITS)
    xb = X + split * CHUNKS * BK
    wb = W + tile * BN * SW + split * CHUNKS * BK
    asl: gl.constexpr = gl.SwizzledSharedLayout(8, 1, min(16, BK // 8), [1, 0])
    bsl: gl.constexpr = gl.SwizzledSharedLayout(8, 1, min(16, BK // 8), [0, 1])
    fl: gl.constexpr = gl.BlockedLayout([8], [64], [NW], [0])
    af: gl.constexpr = gl.BlockedLayout([2 if M * BK < 512 else 8], [64], [NW], [0])
    ai = gl.arange(0, M * BK, af)
    bi = gl.arange(0, BN * BK, fl)
    ax = ai // BK * SX + (ai % BK ^ ai // BK % min(16, BK // 8) * 8)
    bw = bi // BK * SW + (bi % BK ^ bi // BK % min(16, BK // 8) * 8)
    ax = gl.max_contiguous(gl.multiple_of(ax, 8), 8)
    bw = gl.max_contiguous(gl.multiple_of(bw, 8), 8)
    flat_shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    as0 = gl.allocate_shared_memory(X.dtype.element_ty, [M * BK], flat_shared)
    bs0 = gl.allocate_shared_memory(W.dtype.element_ty, [BN * BK], flat_shared)
    as1 = gl.allocate_shared_memory(X.dtype.element_ty, [M * BK], flat_shared)
    bs1 = gl.allocate_shared_memory(W.dtype.element_ty, [BN * BK], flat_shared)
    as2 = gl.allocate_shared_memory(X.dtype.element_ty, [M * BK], flat_shared)
    bs2 = gl.allocate_shared_memory(W.dtype.element_ty, [BN * BK], flat_shared)
    a_stages = (as0, as1, as2)
    b_stages = (bs0, bs1, bs2)
    if STAGES >= 4:
        as3 = gl.allocate_shared_memory(X.dtype.element_ty, [M * BK], flat_shared)
        bs3 = gl.allocate_shared_memory(W.dtype.element_ty, [BN * BK], flat_shared)
        a_stages += (as3,)
        b_stages += (bs3,)
    for i in gl.static_range(STAGES):
        _m2_copy_async_panel(a_stages[i], b_stages[i], xb, wb, ax, bw, i * BK)
    acc = gl.zeros((M, BN), gl.float32, mma)
    for i in gl.static_range(CHUNKS):
        gl.amd.cdna4.async_copy.wait_group(min(STAGES - 1, CHUNKS - i - 1))
        ac = a_stages[i % STAGES]
        bc = b_stages[i % STAGES]
        da = gl.amd.cdna4.async_copy.load_shared_relaxed(
            ac._reinterpret(shape=[M, BK], layout=asl), gl.DotOperandLayout(0, mma, 8)
        )
        db = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bc._reinterpret(shape=[BK, BN], layout=bsl), gl.DotOperandLayout(1, mma, 8)
        )
        if i + STAGES < CHUNKS:
            gl.barrier()
            _m2_copy_async_panel(ac, bc, xb, wb, ax, bw, (i + STAGES) * BK)
        acc = gl.amd.cdna4.mfma(da, db, acc)
    om = gl.arange(0, M, gl.SliceLayout(1, mma))
    on = gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.amd.cdna4.buffer_store(
        ptr=P + split * M * PITCH + tile * BN,
        offsets=om[:, None] * PITCH + on[None, :],
        stored_value=acc,
        cache=".wt",
    )


@gluon.jit
def _m2_wait_private_panel(INDEX: gl.constexpr, CHUNKS: gl.constexpr):
    outstanding: gl.constexpr = 4 * min(3, CHUNKS - INDEX - 1)
    gl.inline_asm_elementwise(
        f"s_waitcnt vmcnt({outstanding})\n v_mov_b32 $0, 0",
        constraints="=v,~{memory}",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _m2_project_private(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    PITCH: gl.constexpr,
    BM: gl.constexpr,
    EARLY_A: gl.constexpr,
    STRIPES: gl.constexpr,
):
    BK: gl.constexpr = 128
    BN: gl.constexpr = 64
    SPLITS: gl.constexpr = 6
    STAGES: gl.constexpr = 4
    CHUNKS: gl.constexpr = K // (BK * SPLITS)
    pid = gl.program_id(0).to(gl.uint32)
    total: gl.constexpr = N // BN * SPLITS
    stripe = pid % STRIPES
    pid = (
        stripe * (total // STRIPES)
        + gl.minimum(stripe, total % STRIPES)
        + pid // STRIPES
    )
    tile, split = (pid // SPLITS, pid % SPLITS)
    xb = X + split * CHUNKS * BK
    wb = W + tile * BN * SW + split * CHUNKS * BK
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    asl: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
    bsl: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [0, 1])
    af: gl.constexpr = gl.BlockedLayout([2 if BM == 2 else 8], [64], [4], [0])
    bf: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [4, 1], [1, 0])
    ai = gl.arange(0, BM * BK, af)
    wave = gl.arange(0, 4, gl.SliceLayout(1, bf))
    lane_data = gl.arange(0, 16 * BK, gl.SliceLayout(0, bf))
    bi = wave[:, None] * (16 * BK) + lane_data[None, :]
    ax = ai // BK * SX + (ai % BK ^ ai // BK % 16 * 8)
    bw = bi // BK * SW + (bi % BK ^ bi // BK % 16 * 8)
    ax = gl.max_contiguous(gl.multiple_of(ax, 8), 8)
    bw = gl.max_contiguous(gl.multiple_of(bw, [1, 8]), [1, 8])
    flat_a: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    flat_b: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    a_panels = ()
    for i in gl.static_range(CHUNKS):
        panel = gl.allocate_shared_memory(X.dtype.element_ty, [BM * BK], flat_a)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            panel, xb, ax + i * BK, ai // BK < M, 0
        )
        a_panels += (panel,)
    gl.amd.cdna4.async_copy.commit_group()
    b_panels = ()
    for i in gl.static_range(STAGES):
        panel = gl.allocate_shared_memory(W.dtype.element_ty, [4, 16 * BK], flat_b)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(panel, wb, bw + i * BK)
        gl.amd.cdna4.async_copy.commit_group()
        b_panels += (panel,)
    gl.amd.cdna4.async_copy.wait_group(STAGES)
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for i in gl.static_range(CHUNKS):
        ac = a_panels[i]._reinterpret(shape=[BM, BK], layout=asl)
        bc = b_panels[i % STAGES]._reinterpret(shape=[BK, BN], layout=bsl)
        if EARLY_A:
            da = gl.amd.cdna4.async_copy.load_shared_relaxed(
                ac, gl.DotOperandLayout(0, mma, 8)
            )
        _m2_wait_private_panel(i, CHUNKS)
        if not EARLY_A:
            da = gl.amd.cdna4.async_copy.load_shared_relaxed(
                ac, gl.DotOperandLayout(0, mma, 8)
            )
        db = gl.amd.cdna4.async_copy.load_shared_relaxed(
            bc, gl.DotOperandLayout(1, mma, 8)
        )
        if i + STAGES < CHUNKS:
            gl.inline_asm_elementwise(
                "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
                constraints="=v,~{memory}",
                args=[],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                b_panels[i % STAGES], wb, bw + (i + STAGES) * BK
            )
        acc = gl.amd.cdna4.mfma(da, db, acc)
    for panel in gl.static_range(CHUNKS):
        a_panels[panel]._keep_alive()
    for panel in gl.static_range(STAGES):
        b_panels[panel]._keep_alive()
    om = gl.arange(0, BM, gl.SliceLayout(1, mma))
    on = gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.amd.cdna4.buffer_store(
        ptr=P + split * M * PITCH + tile * BN,
        offsets=om[:, None] * PITCH + on[None, :],
        stored_value=acc,
        mask=om[:, None] < M,
        cache=".wt",
    )


@gluon.jit
def _m2_finish_part(
    P,
    G,
    O,
    eps,
    row,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    WIDTH: gl.constexpr,
    OFFSET: gl.constexpr,
    SPLITS: gl.constexpr,
    B: gl.constexpr,
    NORM: gl.constexpr,
    VEC: gl.constexpr,
    LOAD_MODE: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [4], [0])
    d = gl.arange(0, B, layout)
    value = gl.full((B,), 0, gl.float32, layout)
    base = P + row * PITCH + OFFSET
    if M == 1:
        value = gl.amd.cdna4.buffer_load(base, d, d < WIDTH, 0)
    for s in gl.static_range(1 if M == 1 else 0, SPLITS):
        if LOAD_MODE == 2:
            value += gl.amd.cdna4.buffer_load(
                P, row * PITCH + OFFSET + d + s * M * PITCH, d < WIDTH, 0
            )
        elif LOAD_MODE == 1:
            value += gl.amd.cdna4.buffer_load(base + s * M * PITCH, d, d < WIDTH, 0)
        else:
            value += gl.load(P + s * M * PITCH + row * PITCH + OFFSET + d, d < WIDTH, 0)
    if NORM:
        value = value.to(gl.bfloat16).to(gl.float32)
        gamma = gl.load(G + d, d < WIDTH, 0).to(gl.float32)
        inv_rms = gl.rsqrt(gl.sum(value * value, 0) / WIDTH + eps)
        value = value * inv_rms * gamma
    gl.store(O + row * WIDTH + d, value, d < WIDTH)


@gluon.jit
def _m2_finish(
    P,
    QG,
    KG,
    QO,
    KO,
    RO,
    eps,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    Q: gl.constexpr,
    KV: gl.constexpr,
    R: gl.constexpr,
    SPLITS: gl.constexpr,
    BQ: gl.constexpr,
    BKV: gl.constexpr,
    BR: gl.constexpr,
    VEC: gl.constexpr,
    LOAD_MODE: gl.constexpr,
    REVERSE: gl.constexpr,
):
    row = gl.program_id(0)
    if M == 16:
        row = row.to(gl.uint32)
    part = gl.program_id(1)
    if REVERSE:
        part = 2 - part
    if part == 0:
        _m2_finish_part(
            P, QG, QO, eps, row, M, PITCH, Q, 0, SPLITS, BQ, True, VEC, LOAD_MODE
        )
    elif part == 1:
        _m2_finish_part(
            P, KG, KO, eps, row, M, PITCH, KV, Q, SPLITS, BKV, True, VEC, LOAD_MODE
        )
    else:
        _m2_finish_part(
            P, QG, RO, eps, row, M, PITCH, R, Q + KV, SPLITS, BR, False, VEC, LOAD_MODE
        )


def _m2_allocate_outputs(device, m, q, kv, rope_dim, packed):
    n = q + kv + rope_dim
    if packed:
        output = torch.empty(m * n, device=device, dtype=torch.bfloat16)
        qo = output[: m * q].view(m, q)
        ko = output[m * q : m * (q + kv)].view(m, kv)
        ro = output[m * (q + kv) :].view(m, rope_dim)
    else:
        qo = torch.empty((m, q), device=device, dtype=torch.bfloat16)
        ko = torch.empty((m, kv), device=device, dtype=torch.bfloat16)
        ro = torch.empty((m, rope_dim), device=device, dtype=torch.bfloat16)
    return (qo, ko, ro)


def _m2_mla_qkv_a_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    q_gamma: torch.Tensor,
    kv_gamma: torch.Tensor,
    *,
    rope_dim: int = 64,
    eps: float = 1e-05,
):
    m, k = hidden.shape
    q, kv = (q_gamma.numel(), kv_gamma.numel())
    n = q + kv + rope_dim
    splits = 6
    pitch = n
    if m in (2, 4):
        pitch = 4096
    elif m == 8:
        pitch = 3072
    elif m == 16:
        pitch = 2816
    partial = torch.empty((splits, m, pitch), device=hidden.device, dtype=torch.float32)
    qo, ko, ro = _m2_allocate_outputs(hidden.device, m, q, kv, rope_dim, m in (2, 16))
    if m in (2, 4, 8):
        _m2_project_private[n // 64 * splits,](
            hidden,
            weight,
            partial,
            m,
            n,
            k,
            hidden.stride(0),
            weight.stride(0),
            pitch,
            m,
            True,
            4,
            num_warps=4,
        )
    elif m in (1, 16):
        bn = 16 if m <= 4 else 64
        nw = bn // 16
        _m2_project_async[n // bn * splits,](
            hidden,
            weight,
            partial,
            m,
            n,
            k,
            hidden.stride(0),
            weight.stride(0),
            pitch,
            128,
            splits,
            4 if m == 8 else 3,
            m != 16,
            4 if m >= 8 else 8,
            bn,
            nw,
            num_warps=nw,
        )
    else:
        bm = triton.next_power_of_2(m)
        bk, group, unroll, chains = (
            (128, 4, 2, 2) if m == 7 else (256, 4 if m == 3 else 1, 4, 1)
        )
        _m2_project_masked[n // 16 * splits,](
            hidden,
            weight,
            partial,
            m,
            n,
            k,
            hidden.stride(0),
            weight.stride(0),
            bm,
            16,
            bk,
            splits,
            group,
            unroll,
            chains,
            num_warps=1,
        )
    vec = 4 if m <= 2 else 8
    load_mode = 2 if m == 2 else int(m == 1 or vec == 8)
    _m2_finish[m, 3](
        partial,
        q_gamma,
        kv_gamma,
        qo,
        ko,
        ro,
        eps,
        m,
        pitch,
        q,
        kv,
        rope_dim,
        splits,
        triton.next_power_of_2(q),
        triton.next_power_of_2(kv),
        triton.next_power_of_2(rope_dim),
        vec,
        load_mode,
        m <= 2,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return (qo, ko, ro)


# ---- variant M <= 4: qkv_a_norm_tp4_m4.py ----


@gluon.jit
def _m4_striped_program_id(
    pid,
    N: gl.constexpr,
    BN: gl.constexpr,
    SPLITS: gl.constexpr,
    GROUP: gl.constexpr,
    STRIPES: gl.constexpr,
):
    if STRIPES > 1:
        TOTAL: gl.constexpr = gl.cdiv(N, BN * GROUP) * GROUP * SPLITS
        stripe = pid % STRIPES
        pid = (
            stripe * (TOTAL // STRIPES)
            + gl.minimum(stripe, TOTAL % STRIPES)
            + pid // STRIPES
        )
    return pid


@gluon.jit
def _m4_copy_panel(a_slot, b_slot, X, W, a_offsets, b_offsets, k_start):
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a_slot, X, a_offsets + k_start, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b_slot, W, b_offsets + k_start, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _m4_wait_weight_wave(DEPTH: gl.constexpr, CHUNKS: gl.constexpr, STEP: gl.constexpr):
    OUTSTANDING: gl.constexpr = 4 * min(DEPTH - 1, CHUNKS - 1 - STEP)
    gl.inline_asm_elementwise(
        f"s_waitcnt vmcnt({OUTSTANDING})\n v_mov_b32 $0, 0",
        constraints="=v,~{memory}",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _m4_project_async(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    PITCH: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    SPLITS: gl.constexpr,
    NW: gl.constexpr,
    STAGES: gl.constexpr,
    GROUP: gl.constexpr,
    STRIPES: gl.constexpr,
    PHASE: gl.constexpr = 1,
    MAX_PHASE: gl.constexpr = 16,
    PRELOAD_A: gl.constexpr = False,
    WAVE_B: gl.constexpr = False,
):
    gl.static_assert(BM == M and N % (BN * GROUP) == 0 and (K % (BK * SPLITS) == 0))
    pid = gl.program_id(0).to(gl.uint32)
    pid = _m4_striped_program_id(pid, N, BN, SPLITS, GROUP, STRIPES)
    tile = pid // (GROUP * SPLITS) * GROUP + pid % GROUP
    split = pid // GROUP % SPLITS
    CHUNKS: gl.constexpr = gl.cdiv(K, BK * SPLITS)
    DEPTH: gl.constexpr = min(STAGES, CHUNKS)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=M != 16, warps_per_cta=[1, NW]
    )
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    WORDS: gl.constexpr = M == 8
    COPY_K: gl.constexpr = BK // 2 if WORDS else BK
    COPY_VEC: gl.constexpr = 4 if WORDS else 8
    if WORDS:
        X = X.to(gl.pointer_type(gl.uint32))
        W = W.to(gl.pointer_type(gl.uint32))
    if WAVE_B:
        gl.static_assert(
            (M == 4 or M == 8) and BK == 128 and PRELOAD_A and (BN == 16 * NW)
        )
        gl.static_assert(NW == 2 or NW == 4)
        if WORDS:
            linear: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[1], [2], [256], [512]],
                lane_bases=[[4], [8], [16], [32], [64], [128]],
                warp_bases=[[1024]] if NW == 2 else [[1024], [2048]],
                block_bases=[],
                shape=[BN * COPY_K],
            )
        else:
            linear: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[1], [2], [4], [512], [1024]],
                lane_bases=[[8], [16], [32], [64], [128], [256]],
                warp_bases=[[2048]] if NW == 2 else [[2048], [4096]],
                block_bases=[],
                shape=[BN * COPY_K],
            )
    else:
        linear: gl.constexpr = gl.BlockedLayout([COPY_VEC], [64], [NW], [0])
    a_linear: gl.constexpr = gl.BlockedLayout(
        [
            COPY_VEC
            if PRELOAD_A
            else (1 if WORDS else 2)
            if BM * BK < NW * 512
            else COPY_VEC
        ],
        [64],
        [NW],
        [0],
    )
    flat_shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    a_shared: gl.constexpr = gl.SwizzledSharedLayout(8, PHASE, MAX_PHASE, [1, 0])
    b_shared: gl.constexpr = gl.SwizzledSharedLayout(8, PHASE, MAX_PHASE, [0, 1])
    A_DEPTH: gl.constexpr = CHUNKS if PRELOAD_A else DEPTH
    A_ITEMS: gl.constexpr = (CHUNKS if PRELOAD_A else 1) * BM * COPY_K
    a_storage = gl.allocate_shared_memory(
        gl.uint32 if WORDS else gl.bfloat16, [A_DEPTH, BM * COPY_K], flat_shared
    )
    if WAVE_B:
        b_slots = ()
        for stage in gl.static_range(DEPTH):
            b_slots += (
                gl.allocate_shared_memory(
                    gl.uint32 if WORDS else gl.bfloat16, [BN * COPY_K], flat_shared
                ),
            )
    else:
        b_storage = gl.allocate_shared_memory(
            gl.uint32 if WORDS else gl.bfloat16, [DEPTH, BN * COPY_K], flat_shared
        )
    ai = gl.arange(0, A_ITEMS, a_linear)
    bi = gl.arange(0, BN * COPY_K, linear)
    am = ai // COPY_K % BM if PRELOAD_A else ai // COPY_K
    bn = tile * BN + bi // COPY_K
    ak = ai % COPY_K ^ am // PHASE % MAX_PHASE * COPY_VEC
    if PRELOAD_A:
        ak += ai // (BM * COPY_K) * COPY_K
    bk = bi % COPY_K ^ bi // COPY_K // PHASE % MAX_PHASE * COPY_VEC
    ak = gl.max_contiguous(gl.multiple_of(ak, COPY_VEC), COPY_VEC)
    bk = gl.max_contiguous(gl.multiple_of(bk, COPY_VEC), COPY_VEC)
    a_offsets = am * (SX // 2 if WORDS else SX) + ak
    b_offsets = bn * (SW // 2 if WORDS else SW) + bk
    start = split * CHUNKS * COPY_K
    if PRELOAD_A:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            a_storage._reinterpret(shape=[CHUNKS * BM * COPY_K], layout=flat_shared),
            X,
            a_offsets + start,
            cache_modifier=".ca",
            mask=am < M,
            other=0,
        )
    for stage in gl.static_range(DEPTH):
        b_dest = b_slots[stage] if WAVE_B else b_storage.index(stage)
        if PRELOAD_A:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                b_dest, W, b_offsets + start + stage * COPY_K, cache_modifier=".ca"
            )
            gl.amd.cdna4.async_copy.commit_group()
        else:
            _m4_copy_panel(
                a_storage.index(stage),
                b_dest,
                X,
                W,
                a_offsets,
                b_offsets,
                start + stage * COPY_K,
            )
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for i in gl.static_range(CHUNKS):
        a_slot = a_storage.index(i if PRELOAD_A else i % DEPTH)
        if WAVE_B and i > 0:
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_slot._reinterpret(dtype=gl.bfloat16, shape=[BM, BK], layout=a_shared),
                ad,
            )
        if WAVE_B and i > 0:
            _m4_wait_weight_wave(DEPTH, CHUNKS, i)
        else:
            gl.amd.cdna4.async_copy.wait_group(min(DEPTH - 1, CHUNKS - 1 - i))
        if NW > 1 and (not WAVE_B or i == 0):
            gl.barrier()
        b_slot = b_slots[i % DEPTH] if WAVE_B else b_storage.index(i % DEPTH)
        if not WAVE_B or i == 0:
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_slot._reinterpret(dtype=gl.bfloat16, shape=[BM, BK], layout=a_shared),
                ad,
            )
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_slot._reinterpret(dtype=gl.bfloat16, shape=[BK, BN], layout=b_shared), bd
        )
        if i + DEPTH < CHUNKS:
            gl.inline_asm_elementwise(
                "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
                constraints="=v,~{memory}",
                args=[],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
            if NW > 1 and (not WAVE_B):
                gl.barrier()
            if PRELOAD_A:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    b_slot,
                    W,
                    b_offsets + start + (i + DEPTH) * COPY_K,
                    cache_modifier=".ca",
                )
                gl.amd.cdna4.async_copy.commit_group()
            else:
                _m4_copy_panel(
                    a_slot,
                    b_slot,
                    X,
                    W,
                    a_offsets,
                    b_offsets,
                    start + (i + DEPTH) * COPY_K,
                )
        acc = gl.amd.cdna4.mfma(a, b, acc)
    om = gl.arange(0, BM, gl.SliceLayout(1, mma))
    on = gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.amd.cdna4.buffer_store(
        ptr=P + split * M * PITCH + tile * BN,
        offsets=om[:, None] * PITCH + on[None, :],
        stored_value=acc,
        mask=(om[:, None] < M) & (tile * BN + on[None, :] < N),
        cache=".wt",
    )


@gluon.jit
def _m4_project_packed(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    PITCH: gl.constexpr,
    TILE: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    PACK: gl.constexpr,
    NW: gl.constexpr,
    SPLITS: gl.constexpr,
    GROUP: gl.constexpr,
    UNROLL: gl.constexpr,
    STRIPES: gl.constexpr,
):
    pid = gl.program_id(0)
    pid = _m4_striped_program_id(pid, N, BN, SPLITS, GROUP, STRIPES)
    col = (pid // (GROUP * SPLITS) * GROUP + pid % GROUP) * BN
    split = pid // GROUP % SPLITS
    al: gl.constexpr = gl.BlockedLayout([BM, PACK], [1, 64], [NW, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([1, PACK], [1, 64], [NW, 1], [1, 0])
    m = gl.arange(0, BM, gl.SliceLayout(1, al))
    n = col + gl.arange(0, BN, gl.SliceLayout(1, bl))
    ak = gl.arange(0, TILE, gl.SliceLayout(0, al))
    bk = gl.arange(0, TILE, gl.SliceLayout(0, bl))
    PARTS: gl.constexpr = TILE // PACK
    dots: gl.constexpr = gl.BlockedLayout([1, 1, 1], [64, 1, 1], [1, 1, NW], [0, 1, 2])
    acc = gl.zeros((PARTS, BM, BN), gl.float32, dots)
    CHUNKS: gl.constexpr = gl.cdiv(K, TILE * SPLITS)
    for block in range(gl.cdiv(CHUNKS, UNROLL)):
        for u in gl.static_range(UNROLL):
            chunk = block * UNROLL + u
            if chunk < CHUNKS:
                ka = (split * CHUNKS + chunk) * TILE + ak
                kb = (split * CHUNKS + chunk) * TILE + bk
                if M == BM and N % (BN * GROUP) == 0 and (K % (TILE * SPLITS) == 0):
                    x = gl.amd.cdna4.buffer_load(X, m[:, None] * SX + ka[None, :])
                    w = gl.amd.cdna4.buffer_load(W, n[:, None] * SW + kb[None, :])
                else:
                    x = gl.amd.cdna4.buffer_load(
                        X,
                        m[:, None] * SX + ka[None, :],
                        (m[:, None] < M) & (ka[None, :] < K),
                        0,
                    )
                    w = gl.amd.cdna4.buffer_load(
                        W,
                        n[:, None] * SW + kb[None, :],
                        (n[:, None] < N) & (kb[None, :] < K),
                        0,
                    )
                a = gl.convert_layout(
                    x.reshape((BM, PARTS, PACK)).permute((1, 0, 2)),
                    gl.DotOperandLayout(0, dots, 0),
                )
                b = gl.convert_layout(
                    w.reshape((BN, PARTS, PACK)).permute((1, 2, 0)),
                    gl.DotOperandLayout(1, dots, 0),
                )
                acc = gl.dot_fma(a, b, acc)
    value = gl.sum(acc, 0)
    out: gl.constexpr = gl.SliceLayout(0, dots)
    om = gl.arange(0, BM, gl.SliceLayout(1, out))
    on = col + gl.arange(0, BN, gl.SliceLayout(0, out))
    gl.store(
        P + split * M * PITCH + om[:, None] * PITCH + on[None, :],
        value,
        (om[:, None] < M) & (on[None, :] < N),
    )


@gluon.jit
def _m4_project_masked(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    PITCH: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    SPLITS: gl.constexpr,
    GROUP: gl.constexpr,
    AK_LANES: gl.constexpr,
    STRIPES: gl.constexpr,
    NW: gl.constexpr,
):
    gl.static_assert(NW == 1 or NW == 2)
    pid = gl.program_id(0).to(gl.uint32)
    pid = _m4_striped_program_id(pid, N, BN, SPLITS, GROUP, STRIPES)
    tile = pid // (GROUP * SPLITS) * GROUP + pid % GROUP
    split = pid // GROUP % SPLITS
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, NW]
    )
    al: gl.constexpr = gl.BlockedLayout(
        [1, 8], [64 // AK_LANES, AK_LANES], [NW, 1], [1, 0]
    )
    if NW == 1:
        bl: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 1], [0, 1])
    else:
        bl: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
            lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2]],
            warp_bases=[[0, 16]],
            block_bases=[],
            shape=[BK, BN],
        )
    am = gl.arange(0, BM, gl.SliceLayout(1, al))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    bn = gl.arange(0, BN, gl.SliceLayout(0, bl))
    CHUNKS: gl.constexpr = gl.cdiv(K, BK * SPLITS)
    x_base = X + split * CHUNKS * BK
    w_base = W + tile * BN * SW + split * CHUNKS * BK
    ma = gl.full((BM, BK), True, gl.int1, al)
    mb = gl.full((BK, BN), True, gl.int1, bl)
    if BM > M:
        ma = ma & (am[:, None] < M)
    if N % (BN * GROUP) != 0:
        mb = mb & (tile * BN + bn[None, :] < N)
    first_ma = ma
    first_mb = mb
    if K % (BK * SPLITS) != 0:
        first_ma = first_ma & (split * CHUNKS * BK + ak[None, :] < K)
        first_mb = first_mb & (split * CHUNKS * BK + bk[:, None] < K)
    a = gl.amd.cdna4.buffer_load(x_base, am[:, None] * SX + ak[None, :], first_ma, 0)
    b = gl.amd.cdna4.buffer_load(w_base, bn[None, :] * SW + bk[:, None], first_mb, 0)
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for i in gl.static_range(CHUNKS):
        if i + 1 < CHUNKS:
            start = (i + 1) * BK
            next_ma = ma
            next_mb = mb
            if K % (BK * SPLITS) != 0:
                next_ma = next_ma & (split * CHUNKS * BK + start + ak[None, :] < K)
                next_mb = next_mb & (split * CHUNKS * BK + start + bk[:, None] < K)
            next_a = gl.amd.cdna4.buffer_load(
                x_base, am[:, None] * SX + start + ak[None, :], next_ma, 0
            )
            next_b = gl.amd.cdna4.buffer_load(
                w_base, bn[None, :] * SW + start + bk[:, None], next_mb, 0
            )
        a_dot = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
        b_dot = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))
        acc = gl.amd.cdna4.mfma(a_dot, b_dot, acc)
        if i + 1 < CHUNKS:
            a = next_a
            b = next_b
    om = gl.arange(0, BM, gl.SliceLayout(1, mma))
    on = gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.amd.cdna4.buffer_store(
        ptr=P + split * M * PITCH + tile * BN,
        offsets=om[:, None] * PITCH + on[None, :],
        stored_value=acc,
        mask=(om[:, None] < M) & (tile * BN + on[None, :] < N),
        cache=".wt",
    )


@gluon.jit
def _m4_finish_part(
    P,
    G,
    O,
    eps,
    row,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    WIDTH: gl.constexpr,
    OFFSET: gl.constexpr,
    SPLITS: gl.constexpr,
    B: gl.constexpr,
    NORM: gl.constexpr,
    VEC: gl.constexpr,
    BUFFER: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [4], [0])
    d = gl.arange(0, B, layout)
    value = gl.full((B,), 0, gl.float32, layout)
    base = P + row * PITCH + OFFSET
    for s in gl.static_range(SPLITS):
        if BUFFER:
            value += gl.amd.cdna4.buffer_load(base + s * M * PITCH, d, d < WIDTH, 0)
        else:
            value += gl.load(P + s * M * PITCH + row * PITCH + OFFSET + d, d < WIDTH, 0)
    if NORM:
        value = value.to(gl.bfloat16).to(gl.float32)
        gamma = gl.load(G + d, d < WIDTH, 0).to(gl.float32)
        inv_rms = gl.rsqrt(gl.sum(value * value, 0) / WIDTH + eps)
        value = value * inv_rms * gamma
    gl.store(O + row * WIDTH + d, value, d < WIDTH)


@gluon.jit
def _m4_finish(
    P,
    QG,
    KG,
    QO,
    KO,
    RO,
    eps,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    Q: gl.constexpr,
    KV: gl.constexpr,
    R: gl.constexpr,
    SPLITS: gl.constexpr,
    BQ: gl.constexpr,
    BKV: gl.constexpr,
    BR: gl.constexpr,
    VEC: gl.constexpr,
    BUFFER: gl.constexpr,
):
    row = gl.program_id(0)
    if M == 16:
        row = row.to(gl.uint32)
    part = gl.program_id(1)
    if M <= 2:
        part = 2 - part
    if part == 0:
        _m4_finish_part(
            P, QG, QO, eps, row, M, PITCH, Q, 0, SPLITS, BQ, True, VEC, BUFFER
        )
    elif part == 1:
        _m4_finish_part(
            P, KG, KO, eps, row, M, PITCH, KV, Q, SPLITS, BKV, True, VEC, BUFFER
        )
    else:
        _m4_finish_part(
            P, QG, RO, eps, row, M, PITCH, R, Q + KV, SPLITS, BR, False, VEC, BUFFER
        )


def _m4_mla_qkv_a_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    q_gamma: torch.Tensor,
    kv_gamma: torch.Tensor,
    *,
    rope_dim: int = 64,
    eps: float = 1e-05,
):
    m, k = hidden.shape
    q, kv = (q_gamma.numel(), kv_gamma.numel())
    n = q + kv + rope_dim
    use_async = (
        m in (1, 2, 4, 8, 16)
        and k % 768 == 0
        and (n % (16 if m <= 4 else 64) == 0)
        and (hidden.stride(0) % 8 == 0)
        and (weight.stride(0) % 8 == 0)
    )
    splits = 3 if m == 1 and (not use_async) else 6
    pitch = n
    if k == 6144 and n == 2624:
        if m == 1 or (m in (2, 4) and use_async):
            pitch = 4096
        elif m == 8 and use_async:
            pitch = 3072
        elif m == 16 and use_async:
            pitch = 2816
    partial = torch.empty((splits, m, pitch), device=hidden.device, dtype=torch.float32)
    qo = torch.empty((m, q), device=hidden.device, dtype=torch.bfloat16)
    ko = torch.empty((m, kv), device=hidden.device, dtype=torch.bfloat16)
    ro = torch.empty((m, rope_dim), device=hidden.device, dtype=torch.bfloat16)
    if m == 1 and (not use_async):
        _m4_project_packed[triton.cdiv(n, 8) * splits,](
            hidden,
            weight,
            partial,
            m,
            n,
            k,
            hidden.stride(0),
            weight.stride(0),
            pitch,
            TILE=512,
            BM=1,
            BN=8,
            PACK=8,
            NW=4,
            SPLITS=splits,
            GROUP=1,
            UNROLL=2,
            STRIPES=8,
            num_warps=4,
        )
    elif use_async:
        bn = 16 if m <= 4 else 64
        nw = 1 if m <= 4 else 4
        stripes = 8 if m <= 4 else 4
        preload_a = m in (1, 4, 8) and k == 6144
        stages = 5 if m == 1 or (m == 4 and preload_a) else 4 if m in (2, 8) else 3
        wave_b = m in (4, 8) and preload_a and (n % 64 == 0)
        if wave_b:
            bn, nw, stripes, stages = (64, 4, 4, 4)
        _m4_project_async[triton.cdiv(n, bn) * splits,](
            hidden,
            weight,
            partial,
            m,
            n,
            k,
            hidden.stride(0),
            weight.stride(0),
            pitch,
            BM=m,
            BN=bn,
            BK=128,
            SPLITS=splits,
            NW=nw,
            STAGES=stages,
            GROUP=1,
            STRIPES=stripes,
            PRELOAD_A=preload_a,
            WAVE_B=wave_b,
            num_warps=nw,
        )
    else:
        nw = 2 if m in (15, 16) else 1
        bn = 16 * nw
        group = 4 if m <= 8 else 1
        stripes = 4 if nw == 2 else 1 if 5 <= m <= 8 else 8
        _m4_project_masked[triton.cdiv(n, bn * group) * group * splits,](
            hidden,
            weight,
            partial,
            m,
            n,
            k,
            hidden.stride(0),
            weight.stride(0),
            pitch,
            BM=triton.next_power_of_2(m),
            BN=bn,
            BK=128,
            SPLITS=splits,
            GROUP=group,
            AK_LANES=16,
            STRIPES=stripes,
            NW=nw,
            num_warps=nw,
        )
    vec = 8 if 2 <= m <= 16 and m != 4 or (m == 1 and use_async) else 4
    _m4_finish[m, 3](
        partial,
        q_gamma,
        kv_gamma,
        qo,
        ko,
        ro,
        eps,
        m,
        pitch,
        q,
        kv,
        rope_dim,
        splits,
        triton.next_power_of_2(q),
        triton.next_power_of_2(kv),
        triton.next_power_of_2(rope_dim),
        vec,
        vec == 8,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return (qo, ko, ro)


# ---- variant M <= 8: qkv_a_norm_tp4_m8.py ----


@gluon.jit
def _m8_stripe_program(pid, TOTAL: gl.constexpr, STRIPES: gl.constexpr):
    if STRIPES > 1:
        stripe = pid % STRIPES
        pid = (
            stripe * (TOTAL // STRIPES)
            + gl.minimum(stripe, TOTAL % STRIPES)
            + pid // STRIPES
        )
    return pid


@gluon.jit
def _m8_async_issue(a_stage, b_stage, X, W, ax, bw, start):
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a_stage, X, ax + start)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(b_stage, W, bw + start)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _m8_async_read(
    a_stage,
    b_stage,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    MMA: gl.constexpr,
    NW: gl.constexpr,
    OUTSTANDING: gl.constexpr,
):
    gl.amd.cdna4.async_copy.wait_group(OUTSTANDING)
    if NW > 1:
        gl.barrier()
    a_view = a_stage._reinterpret(
        shape=[BM, BK], layout=gl.SwizzledSharedLayout(8, 1, BK // 8, [1, 0])
    )
    b_view = b_stage._reinterpret(
        shape=[BK, BN], layout=gl.SwizzledSharedLayout(8, 1, BK // 8, [0, 1])
    )
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(
        a_view, gl.DotOperandLayout(0, MMA, 8)
    )
    b = gl.amd.cdna4.async_copy.load_shared_relaxed(
        b_view, gl.DotOperandLayout(1, MMA, 8)
    )
    gl.inline_asm_elementwise(
        "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
        constraints="=v,~{memory}",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    if NW > 1:
        gl.barrier()
    return (a, b)


@gluon.jit
def _m8_project_async(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    PITCH: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    SPLITS: gl.constexpr,
    NW: gl.constexpr,
    STRIPES: gl.constexpr,
    STAGES: gl.constexpr,
    TRANSPOSED: gl.constexpr,
):
    total: gl.constexpr = N // BN * SPLITS
    pid = gl.program_id(0).to(gl.uint32)
    stripe = pid % STRIPES
    pid = (
        stripe * (total // STRIPES)
        + gl.minimum(stripe, total % STRIPES)
        + pid // STRIPES
    )
    tile = pid // SPLITS
    split = pid % SPLITS
    CHUNKS: gl.constexpr = K // (BK * SPLITS)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=TRANSPOSED,
        warps_per_cta=[1, NW],
    )
    acopy: gl.constexpr = gl.BlockedLayout(
        [8 if BM * BK >= 512 * NW else 2], [64], [NW], [0]
    )
    bcopy: gl.constexpr = gl.BlockedLayout([8], [64], [NW], [0])
    ai = gl.arange(0, BM * BK, acopy)
    bi = gl.arange(0, BN * BK, bcopy)
    ar = ai // BK
    br = bi // BK
    ak = (ai % BK // 8 ^ ar % (BK // 8)) * 8
    bk = (bi % BK // 8 ^ br % (BK // 8)) * 8
    ax = gl.max_contiguous(gl.multiple_of(ar % M * SX + ak, 8) + ai % 8, 8)
    bw = gl.max_contiguous(gl.multiple_of(br * SW + bk, 8) + bi % 8, 8)
    ax = gl.max_contiguous(gl.multiple_of(ax, 8), 8)
    bw = gl.max_contiguous(gl.multiple_of(bw, 8), 8)
    if M == 2 or M == 8:
        ax = gl.max_contiguous(gl.multiple_of(ax + split * CHUNKS * BK, 8), 8)
        bw = gl.max_contiguous(
            gl.multiple_of(bw + tile * BN * SW + split * CHUNKS * BK, 8), 8
        )
        xb = X
        wb = W
    else:
        xb = X + split * CHUNKS * BK
        wb = W + tile * BN * SW + split * CHUNKS * BK
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    a_stages = ()
    b_stages = ()
    for stage in gl.static_range(STAGES):
        a_stages += (gl.allocate_shared_memory(X.dtype.element_ty, [BM * BK], flat),)
        b_stages += (gl.allocate_shared_memory(W.dtype.element_ty, [BN * BK], flat),)
    for stage in gl.static_range(STAGES):
        _m8_async_issue(a_stages[stage], b_stages[stage], xb, wb, ax, bw, stage * BK)
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for i in gl.static_range(CHUNKS):
        if i + STAGES < CHUNKS:
            a, b = _m8_async_read(
                a_stages[i % STAGES],
                b_stages[i % STAGES],
                BM,
                BN,
                BK,
                mma,
                NW,
                STAGES - 1,
            )
            _m8_async_issue(
                a_stages[i % STAGES],
                b_stages[i % STAGES],
                xb,
                wb,
                ax,
                bw,
                (i + STAGES) * BK,
            )
        else:
            a, b = _m8_async_read(
                a_stages[i % STAGES],
                b_stages[i % STAGES],
                BM,
                BN,
                BK,
                mma,
                NW,
                CHUNKS - i - 1,
            )
        acc = gl.amd.cdna4.mfma(a, b, acc)
    for stage in gl.static_range(STAGES):
        a_stages[stage]._keep_alive()
        b_stages[stage]._keep_alive()
    om = gl.arange(0, BM, gl.SliceLayout(1, mma))
    on = gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.amd.cdna4.buffer_store(
        ptr=P + split * M * PITCH + tile * BN,
        offsets=om[:, None] * PITCH + on[None, :],
        stored_value=acc,
        mask=om[:, None] < M,
        cache=".wt",
    )


@gluon.jit
def _m8_project_private_weights(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    PITCH: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    SPLITS: gl.constexpr,
    STRIPES: gl.constexpr,
    STAGES: gl.constexpr,
    EARLY_A: gl.constexpr,
):
    gl.static_assert(BN == 64 and BK == 128)
    gl.static_assert(K % (BK * SPLITS) == 0)
    total: gl.constexpr = N // BN * SPLITS
    pid = _m8_stripe_program(gl.program_id(0).to(gl.uint32), total, STRIPES)
    tile = pid // SPLITS
    split = pid % SPLITS
    chunks: gl.constexpr = K // (BK * SPLITS)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    acopy: gl.constexpr = gl.BlockedLayout(
        [8 if BM * BK >= 2048 else 2], [64], [4], [0]
    )
    bcopy: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1], [2], [4], [512], [1024]],
        lane_bases=[[8], [16], [32], [64], [128], [256]],
        warp_bases=[[2048], [4096]],
        block_bases=[],
        shape=[BN * BK],
    )
    ai = gl.arange(0, BM * BK, acopy)
    bi = gl.arange(0, BN * BK, bcopy)
    ar = ai // BK
    br = bi // BK
    ak = (ai % BK // 8 ^ ar % (BK // 8)) * 8
    bk = (bi % BK // 8 ^ br % (BK // 8)) * 8
    ax = gl.max_contiguous(gl.multiple_of(ar * SX + ak, 8) + ai % 8, 8)
    bw = gl.max_contiguous(gl.multiple_of(br * SW + bk, 8) + bi % 8, 8)
    ax = gl.max_contiguous(gl.multiple_of(ax, 8), 8)
    bw = gl.max_contiguous(gl.multiple_of(bw, 8), 8)
    xb = X + split * chunks * BK
    wb = W + tile * BN * SW + split * chunks * BK
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    a_panels = ()
    b_stages = ()
    for i in gl.static_range(chunks):
        a_panels += (gl.allocate_shared_memory(X.dtype.element_ty, [BM * BK], flat),)
    for i in gl.static_range(STAGES):
        b_stages += (gl.allocate_shared_memory(W.dtype.element_ty, [BN * BK], flat),)
    for i in gl.static_range(chunks):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            a_panels[i],
            xb,
            ax + i * BK,
            ar < M,
            gl.full((BM * BK,), 0, X.dtype.element_ty, acopy),
            cache_modifier=".ca",
        )
    gl.amd.cdna4.async_copy.commit_group()
    for i in gl.static_range(STAGES):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            b_stages[i], wb, bw + i * BK, cache_modifier=".ca"
        )
        gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.wait_group(STAGES)
    gl.barrier()
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for i in gl.static_range(chunks):
        a_view = a_panels[i]._reinterpret(
            shape=[BM, BK], layout=gl.SwizzledSharedLayout(8, 1, BK // 8, [1, 0])
        )
        b_view = b_stages[i % STAGES]._reinterpret(
            shape=[BK, BN], layout=gl.SwizzledSharedLayout(8, 1, BK // 8, [0, 1])
        )
        if EARLY_A:
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_view, gl.DotOperandLayout(0, mma, 8)
            )
        gl.inline_asm_elementwise(
            "s_waitcnt vmcnt($1)\n v_mov_b32 $0, 0",
            constraints="=v,n,~{memory}",
            args=[4 * min(STAGES - 1, chunks - i - 1)],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        if not EARLY_A:
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_view, gl.DotOperandLayout(0, mma, 8)
            )
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_view, gl.DotOperandLayout(1, mma, 8)
        )
        gl.inline_asm_elementwise(
            "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
            constraints="=v,~{memory}",
            args=[],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        if i + STAGES < chunks:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                b_stages[i % STAGES], wb, bw + (i + STAGES) * BK, cache_modifier=".ca"
            )
            gl.amd.cdna4.async_copy.commit_group()
        acc = gl.amd.cdna4.mfma(a, b, acc)
    for i in gl.static_range(chunks):
        a_panels[i]._keep_alive()
    for i in gl.static_range(STAGES):
        b_stages[i]._keep_alive()
    om = gl.arange(0, BM, gl.SliceLayout(1, mma))
    on = gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.amd.cdna4.buffer_store(
        ptr=P + split * M * PITCH + tile * BN,
        offsets=om[:, None] * PITCH + on[None, :],
        stored_value=acc,
        mask=om[:, None] < M,
        cache=".wt",
    )


@gluon.jit
def _m8_project_resident_row(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    PITCH: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    SPLITS: gl.constexpr,
    STRIPES: gl.constexpr,
    STAGES: gl.constexpr,
    EARLY_A: gl.constexpr,
    NW: gl.constexpr = 1,
):
    gl.static_assert(M == 1 and BM == 1 and (NW == 1))
    gl.static_assert(BN == 16 and BK == 128 and EARLY_A)
    gl.static_assert(K % (BK * SPLITS) == 0)
    total: gl.constexpr = N // BN * SPLITS
    pid = _m8_stripe_program(gl.program_id(0).to(gl.uint32), total, STRIPES)
    tile = pid // SPLITS
    split = pid % SPLITS
    chunks: gl.constexpr = K // (BK * SPLITS)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, NW]
    )
    acopy: gl.constexpr = gl.BlockedLayout([8], [64], [NW], [0])
    bcopy: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1], [2], [4], [512], [1024]],
        lane_bases=[[8], [16], [32], [64], [128], [256]],
        warp_bases=[],
        block_bases=[],
        shape=[BN * BK],
    )
    ai = gl.arange(0, chunks * BM * BK, acopy)
    bi = gl.arange(0, BN * BK, bcopy)
    ar = ai // (chunks * BK)
    br = bi // BK
    ak = (ai % (chunks * BK) // 8 ^ ar % (BK // 8)) * 8
    bk = (bi % BK // 8 ^ br % (BK // 8)) * 8
    ax = gl.max_contiguous(gl.multiple_of(ar * SX + ak, 8) + ai % 8, 8)
    bw = gl.max_contiguous(gl.multiple_of(br * SW + bk, 8) + bi % 8, 8)
    ax = gl.max_contiguous(gl.multiple_of(ax, 8), 8)
    bw = gl.max_contiguous(gl.multiple_of(bw, 8), 8)
    xb = X + split * chunks * BK
    wb = W + tile * BN * SW + split * chunks * BK
    flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    a_all = gl.allocate_shared_memory(X.dtype.element_ty, [chunks * BM * BK], flat)
    b_stages = ()
    for i in gl.static_range(STAGES):
        b_stages += (gl.allocate_shared_memory(W.dtype.element_ty, [BN * BK], flat),)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a_all,
        xb,
        ax,
        ar < M,
        gl.full((chunks * BM * BK,), 0, X.dtype.element_ty, acopy),
        cache_modifier=".ca",
    )
    gl.amd.cdna4.async_copy.commit_group()
    for i in gl.static_range(STAGES):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            b_stages[i], wb, bw + i * BK, cache_modifier=".ca"
        )
        gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.wait_group(STAGES)
    gl.barrier()
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for i in gl.static_range(chunks):
        a_view = a_all._reinterpret(
            shape=[BM, chunks * BK], layout=gl.SwizzledSharedLayout(8, 1, 1, [1, 0])
        ).slice(i * BK, BK, dim=1)
        b_view = b_stages[i % STAGES]._reinterpret(
            shape=[BK, BN], layout=gl.SwizzledSharedLayout(8, 1, BK // 8, [0, 1])
        )
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_view, gl.DotOperandLayout(0, mma, 8)
        )
        gl.inline_asm_elementwise(
            "s_waitcnt vmcnt($1)\n v_mov_b32 $0, 0",
            constraints="=v,n,~{memory}",
            args=[4 * min(STAGES - 1, chunks - i - 1)],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_view, gl.DotOperandLayout(1, mma, 8)
        )
        gl.inline_asm_elementwise(
            "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
            constraints="=v,~{memory}",
            args=[],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        if i + STAGES < chunks:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                b_stages[i % STAGES], wb, bw + (i + STAGES) * BK, cache_modifier=".ca"
            )
            gl.amd.cdna4.async_copy.commit_group()
        acc = gl.amd.cdna4.mfma(a, b, acc)
    a_all._keep_alive()
    for i in gl.static_range(STAGES):
        b_stages[i]._keep_alive()
    om = gl.arange(0, BM, gl.SliceLayout(1, mma))
    on = gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.amd.cdna4.buffer_store(
        ptr=P + split * M * PITCH + tile * BN,
        offsets=om[:, None] * PITCH + on[None, :],
        stored_value=acc,
        mask=om[:, None] < M,
        cache=".wt",
    )


@gluon.jit
def _m8_project_mfma(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    PITCH: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    SPLITS: gl.constexpr,
    GROUP: gl.constexpr,
    STRIPES: gl.constexpr,
    UNROLL: gl.constexpr,
):
    total: gl.constexpr = gl.cdiv(N, BN * GROUP) * GROUP * SPLITS
    pid = _m8_stripe_program(gl.program_id(0).to(gl.uint32), total, STRIPES)
    tile = pid // (GROUP * SPLITS) * GROUP + pid % GROUP
    split = pid // GROUP % SPLITS
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1]
    )
    al: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 1], [0, 1])
    am = gl.arange(0, BM, gl.SliceLayout(1, al))
    ak = gl.arange(0, BK, gl.SliceLayout(0, al))
    bk = gl.arange(0, BK, gl.SliceLayout(1, bl))
    bn = gl.arange(0, BN, gl.SliceLayout(0, bl))
    CHUNKS: gl.constexpr = gl.cdiv(K, BK * SPLITS)
    xb = X + split * CHUNKS * BK
    wb = W + tile * BN * SW + split * CHUNKS * BK
    ma = gl.full((BM, BK), True, gl.int1, al)
    mb = gl.full((BK, BN), True, gl.int1, bl)
    if BM > M:
        ma = ma & (am[:, None] < M)
    if N % (BN * GROUP) != 0:
        mb = mb & (tile * BN + bn[None, :] < N)
    first_ma = ma
    first_mb = mb
    if K % (BK * SPLITS) != 0:
        first_ma = first_ma & (split * CHUNKS * BK + ak[None, :] < K)
        first_mb = first_mb & (split * CHUNKS * BK + bk[:, None] < K)
    a = gl.amd.cdna4.buffer_load(xb, am[:, None] * SX + ak[None, :], first_ma, 0)
    b = gl.amd.cdna4.buffer_load(wb, bn[None, :] * SW + bk[:, None], first_mb, 0)
    next_a = a
    next_b = b
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for block in range(gl.cdiv(CHUNKS, UNROLL)):
        for u in gl.static_range(UNROLL):
            i = block * UNROLL + u
            if i < CHUNKS:
                if i + 1 < CHUNKS:
                    start = (i + 1) * BK
                    next_ma = ma
                    next_mb = mb
                    if K % (BK * SPLITS) != 0:
                        next_ma = next_ma & (
                            split * CHUNKS * BK + start + ak[None, :] < K
                        )
                        next_mb = next_mb & (
                            split * CHUNKS * BK + start + bk[:, None] < K
                        )
                    next_a = gl.amd.cdna4.buffer_load(
                        xb, am[:, None] * SX + start + ak[None, :], next_ma, 0
                    )
                    next_b = gl.amd.cdna4.buffer_load(
                        wb, bn[None, :] * SW + start + bk[:, None], next_mb, 0
                    )
                ad = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
                bd = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))
                acc = gl.amd.cdna4.mfma(ad, bd, acc)
                if i + 1 < CHUNKS:
                    a = next_a
                    b = next_b
    om = gl.arange(0, BM, gl.SliceLayout(1, mma))
    on = gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.amd.cdna4.buffer_store(
        ptr=P + split * M * PITCH + tile * BN,
        offsets=om[:, None] * PITCH + on[None, :],
        stored_value=acc,
        mask=(om[:, None] < M) & (tile * BN + on[None, :] < N),
        cache=".wt",
    )


@gluon.jit
def _m8_finish_part(
    P,
    G,
    O,
    eps,
    row,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    WIDTH: gl.constexpr,
    OFFSET: gl.constexpr,
    SPLITS: gl.constexpr,
    B: gl.constexpr,
    NORM: gl.constexpr,
    VEC: gl.constexpr,
    BUFFER: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [4], [0])
    d = gl.arange(0, B, layout)
    base = P + row * PITCH + OFFSET
    if M == 1:
        value = gl.amd.cdna4.buffer_load(base, d, d < WIDTH, 0)
    else:
        value = gl.full((B,), 0, gl.float32, layout)
    for s in gl.static_range(1 if M == 1 else 0, SPLITS):
        if M == 2:
            value += gl.amd.cdna4.buffer_load(
                P, row * PITCH + OFFSET + s * M * PITCH + d, d < WIDTH, 0
            )
        elif M == 1 or BUFFER:
            value += gl.amd.cdna4.buffer_load(base + s * M * PITCH, d, d < WIDTH, 0)
        else:
            value += gl.load(P + s * M * PITCH + row * PITCH + OFFSET + d, d < WIDTH, 0)
    if NORM:
        value = value.to(gl.bfloat16).to(gl.float32)
        gamma = gl.load(G + d, d < WIDTH, 0).to(gl.float32)
        inv_rms = gl.rsqrt(gl.sum(value * value, 0) / WIDTH + eps)
        value = value * inv_rms * gamma
    gl.store(O + row * WIDTH + d, value, d < WIDTH)


@gluon.jit
def _m8_finish(
    P,
    QG,
    KG,
    QO,
    KO,
    RO,
    eps,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    Q: gl.constexpr,
    KV: gl.constexpr,
    R: gl.constexpr,
    SPLITS: gl.constexpr,
    BQ: gl.constexpr,
    BKV: gl.constexpr,
    BR: gl.constexpr,
    VEC: gl.constexpr,
    BUFFER: gl.constexpr,
):
    row = gl.program_id(0)
    if M == 2:
        row = row.to(gl.uint32)
    part = gl.program_id(1)
    if M == 1 or M == 2:
        part = 2 - part
    if part == 0:
        _m8_finish_part(
            P, QG, QO, eps, row, M, PITCH, Q, 0, SPLITS, BQ, True, VEC, BUFFER
        )
    elif part == 1:
        _m8_finish_part(
            P, KG, KO, eps, row, M, PITCH, KV, Q, SPLITS, BKV, True, VEC, BUFFER
        )
    else:
        _m8_finish_part(
            P, QG, RO, eps, row, M, PITCH, R, Q + KV, SPLITS, BR, False, VEC, BUFFER
        )


@gluon.jit
def _m8_finish_sharded_part(
    P,
    G,
    O,
    eps,
    row,
    shard,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    WIDTH: gl.constexpr,
    OFFSET: gl.constexpr,
    SPLITS: gl.constexpr,
    B: gl.constexpr,
    NORM: gl.constexpr,
    SHARDS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [4], [0])
    d = gl.arange(0, B, layout)
    value = gl.full((B,), 0, gl.float32, layout)
    base = P + row * PITCH + OFFSET
    for s in gl.static_range(SPLITS):
        value += gl.amd.cdna4.buffer_load(base + s * M * PITCH, d, d < WIDTH, 0)
    out_mask = (d < WIDTH) & (d // (B // SHARDS) == shard)
    if NORM:
        value = value.to(gl.bfloat16).to(gl.float32)
        gamma = gl.load(G + d, d < WIDTH, 0).to(gl.float32)
        inv_rms = gl.rsqrt(gl.sum(value * value, 0) / WIDTH + eps)
        value = value * inv_rms * gamma
    gl.store(O + row * WIDTH + d, value, out_mask)


@gluon.jit
def _m8_finish_sharded(
    P,
    QG,
    KG,
    QO,
    KO,
    RO,
    eps,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    Q: gl.constexpr,
    KV: gl.constexpr,
    R: gl.constexpr,
    SPLITS: gl.constexpr,
    BQ: gl.constexpr,
    BKV: gl.constexpr,
    BR: gl.constexpr,
    SHARDS: gl.constexpr,
):
    row = gl.program_id(0).to(gl.uint32)
    part = gl.program_id(1)
    if part < SHARDS:
        _m8_finish_sharded_part(
            P, QG, QO, eps, row, part, M, PITCH, Q, 0, SPLITS, BQ, True, SHARDS
        )
    elif part == SHARDS:
        _m8_finish_sharded_part(
            P, KG, KO, eps, row, 0, M, PITCH, KV, Q, SPLITS, BKV, True, 1
        )
    else:
        _m8_finish_sharded_part(
            P, QG, RO, eps, row, 0, M, PITCH, R, Q + KV, SPLITS, BR, False, 1
        )


def _m8_mla_qkv_a_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    q_gamma: torch.Tensor,
    kv_gamma: torch.Tensor,
    *,
    rope_dim: int = 64,
    eps: float = 1e-05,
):
    m, k = hidden.shape
    q, kv = (q_gamma.numel(), kv_gamma.numel())
    n = q + kv + rope_dim
    use_async = m in (1, 2, 4, 8, 16)
    splits = 6
    if m <= 4 and use_async:
        pitch = 4096
    elif m == 8:
        pitch = 3072
    elif m == 16:
        pitch = 2816
    else:
        pitch = n
    partial = torch.empty((splits, m, pitch), device=hidden.device, dtype=torch.float32)
    qo = torch.empty((m, q), device=hidden.device, dtype=torch.bfloat16)
    ko = torch.empty((m, kv), device=hidden.device, dtype=torch.bfloat16)
    ro = torch.empty((m, rope_dim), device=hidden.device, dtype=torch.bfloat16)
    if m == 1:
        _m8_project_resident_row[n // 16 * splits,](
            hidden,
            weight,
            partial,
            m,
            n,
            k,
            hidden.stride(0),
            weight.stride(0),
            pitch,
            BM=1,
            BN=16,
            BK=128,
            SPLITS=splits,
            STRIPES=8,
            STAGES=4,
            EARLY_A=True,
            NW=1,
            num_warps=1,
        )
    elif m in (2, 4, 8):
        _m8_project_private_weights[n // 64 * splits,](
            hidden,
            weight,
            partial,
            m,
            n,
            k,
            hidden.stride(0),
            weight.stride(0),
            pitch,
            BM=16,
            BN=64,
            BK=128,
            SPLITS=splits,
            STRIPES=4,
            STAGES=4,
            EARLY_A=True,
            num_warps=4,
        )
    elif use_async:
        bn = 16 if m == 1 else 64
        nw = bn // 16
        _m8_project_async[n // bn * splits,](
            hidden,
            weight,
            partial,
            m,
            n,
            k,
            hidden.stride(0),
            weight.stride(0),
            pitch,
            BM=4 if m == 1 else m,
            BN=bn,
            BK=128,
            SPLITS=splits,
            NW=nw,
            STRIPES=8 if m == 1 else 4,
            STAGES=4 if m == 1 else 3,
            TRANSPOSED=m != 16,
            num_warps=nw,
        )
    else:
        group = 4 if m <= 8 else 1
        _m8_project_mfma[triton.cdiv(n, 16 * group) * group * splits,](
            hidden,
            weight,
            partial,
            m,
            n,
            k,
            hidden.stride(0),
            weight.stride(0),
            pitch,
            BM=triton.next_power_of_2(m),
            BN=16,
            BK=128,
            SPLITS=splits,
            GROUP=group,
            STRIPES=1 if m == 7 else 8,
            UNROLL=triton.cdiv(k, 128 * splits),
            num_warps=1,
        )
    vec = 4 if m in (1, 2, 4) else 8
    if m == 16:
        _m8_finish_sharded[m, 4](
            partial,
            q_gamma,
            kv_gamma,
            qo,
            ko,
            ro,
            eps,
            m,
            pitch,
            q,
            kv,
            rope_dim,
            splits,
            BQ=triton.next_power_of_2(q),
            BKV=triton.next_power_of_2(kv),
            BR=triton.next_power_of_2(rope_dim),
            SHARDS=2,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        _m8_finish[m, 3](
            partial,
            q_gamma,
            kv_gamma,
            qo,
            ko,
            ro,
            eps,
            m,
            pitch,
            q,
            kv,
            rope_dim,
            splits,
            BQ=triton.next_power_of_2(q),
            BKV=triton.next_power_of_2(kv),
            BR=triton.next_power_of_2(rope_dim),
            VEC=vec,
            BUFFER=vec == 8,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return (qo, ko, ro)


# ---- variant M <= 16: qkv_a_norm_tp8_m16.py ----


@gluon.jit
def _m16_project_packed(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    SPLITS: gl.constexpr,
    BK: gl.constexpr,
    PACK: gl.constexpr,
    BN: gl.constexpr,
    NW: gl.constexpr,
):
    pn, split = (gl.program_id(0), gl.program_id(1))
    BM: gl.constexpr = _artifact_next_power_of_2(M)
    BATCH: gl.constexpr = BK // PACK
    wl: gl.constexpr = gl.BlockedLayout([1, PACK], [4, 16], [NW, 1], [1, 0])
    if M == 2:
        xl: gl.constexpr = gl.BlockedLayout([1, PACK], [1, 64], [NW, 1], [1, 0])
    else:
        xl: gl.constexpr = gl.BlockedLayout([1, PACK], [1, 64], [1, NW], [1, 0])
    dl: gl.constexpr = gl.BlockedLayout([1, 1, 1], [16, 1, 4], [1, 1, NW], [0, 2, 1])
    ns = pn * BN + gl.arange(0, BN, gl.SliceLayout(1, wl))
    kw = gl.arange(0, BK, gl.SliceLayout(0, wl))
    ms = gl.arange(0, BM, gl.SliceLayout(1, xl))
    kx = gl.arange(0, BK, gl.SliceLayout(0, xl))
    acc = gl.zeros((BATCH, BM, BN), gl.float32, dl)
    STEPS: gl.constexpr = gl.cdiv(K, SPLITS * BK)
    for step in gl.static_range(STEPS):
        start = (split * STEPS + step) * BK
        xm = (ms[:, None] < M) & (start + kx[None, :] < K)
        wm = (ns[:, None] < N) & (start + kw[None, :] < K)
        if K % (SPLITS * BK) == 0:
            xm = (ms[:, None] < M) & (kx[None, :] < BK)
            wm = (ns[:, None] < N) & (kw[None, :] < BK)
            if N % BN == 0:
                wm = gl.full((BN, BK), True, gl.int1, wl)
        x = gl.load(X + ms[:, None] * SX + start + kx[None, :], xm, 0)
        w = gl.load(W + ns[:, None] * SW + start + kw[None, :], wm, 0)
        a = gl.convert_layout(
            x.reshape((BM, BATCH, PACK)).permute((1, 0, 2)),
            gl.DotOperandLayout(0, dl, 0),
        )
        b = gl.convert_layout(
            w.reshape((BN, BATCH, PACK)).permute((1, 2, 0)),
            gl.DotOperandLayout(1, dl, 0),
        )
        acc = gl.dot_fma(a, b, acc)
    values = gl.sum(acc, 0)
    ol: gl.constexpr = gl.SliceLayout(0, dl)
    rows = gl.arange(0, BM, gl.SliceLayout(1, ol))
    cols = pn * BN + gl.arange(0, BN, gl.SliceLayout(0, ol))
    offset = (rows[:, None] * SPLITS + split) * N + cols[None, :]
    gl.store(P + offset, values, (rows[:, None] < M) & (cols[None, :] < N))


@gluon.jit
def _m16_project_mfma(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    SPLITS: gl.constexpr,
    BK: gl.constexpr,
    CHAINS: gl.constexpr,
    ROTATE: gl.constexpr,
    PITCH: gl.constexpr,
):
    pid = gl.program_id(0)
    pn = pid // (SPLITS * 4) * 4 + pid % 4
    split = pid // 4 % SPLITS
    if ROTATE != 0:
        split = (split + ROTATE * pn) % SPLITS
    BM: gl.constexpr = _artifact_next_power_of_2(M)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1]
    )
    la: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
    lb: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 1], [0, 1])
    mi = gl.arange(0, BM, gl.SliceLayout(1, la))
    ni = pn * 16 + gl.arange(0, 16, gl.SliceLayout(0, lb))
    ka = gl.arange(0, BK, gl.SliceLayout(0, la))
    kb = gl.arange(0, BK, gl.SliceLayout(1, lb))
    acc0 = gl.zeros((BM, 16), gl.float32, mma)
    if CHAINS == 2:
        acc1 = gl.zeros((BM, 16), gl.float32, mma)
    STEPS: gl.constexpr = gl.cdiv(K, SPLITS * BK)
    for step in gl.static_range(STEPS):
        start = (split * STEPS + step) * BK
        am = (mi[:, None] < M) & (start + ka[None, :] < K)
        bm = (ni[None, :] < N) & (start + kb[:, None] < K)
        if K % (SPLITS * BK) == 0:
            am = (mi[:, None] < M) & (ka[None, :] < BK)
            if N % 16 == 0:
                bm = gl.full((BK, 16), True, gl.int1, lb)
        a = gl.load(X + mi[:, None] * SX + start + ka[None, :], am, 0)
        b = gl.load(W + ni[None, :] * SW + start + kb[:, None], bm, 0)
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))
        if CHAINS == 1 or step % CHAINS == 0:
            acc0 = gl.amd.cdna4.mfma(a, b, acc0)
        else:
            acc1 = gl.amd.cdna4.mfma(a, b, acc1)
    if CHAINS == 2:
        acc0 = acc0 + acc1
    mr = gl.arange(0, BM, gl.SliceLayout(1, mma))
    nr = pn * 16 + gl.arange(0, 16, gl.SliceLayout(0, mma))
    gl.static_assert(N % 64 == 0)
    offset = (mr[:, None] * SPLITS + split) * PITCH + nr[None, :]
    cache: gl.constexpr = ".wt" if M == 4 else ""
    gl.store(P + offset, acc0, mr[:, None] < M, cache_modifier=cache)


@gluon.jit
def _m16_project_mfma_dual(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    SPLITS: gl.constexpr,
    BK: gl.constexpr,
    ROTATE: gl.constexpr,
    PITCH: gl.constexpr,
):
    pid = gl.program_id(0)
    pn = pid // (SPLITS * 4) * 4 + pid % 4
    split = (pid // 4 % SPLITS + ROTATE * pn) % SPLITS
    BM: gl.constexpr = _artifact_next_power_of_2(M)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1, 1]
    )
    la: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
    lb: gl.constexpr = gl.BlockedLayout([8, 1], [16, 4], [1, 1], [0, 1])
    shared_a_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[128, 8]], [2, BM, BK // 2], [2, 0, 1]
    )
    shared_a = gl.allocate_shared_memory(gl.bfloat16, [2, BM, BK // 2], shared_a_layout)
    mi = gl.arange(0, BM, gl.SliceLayout(1, la))
    ni = pn * 16 + gl.arange(0, 16, gl.SliceLayout(0, lb))
    ka = gl.arange(0, BK, gl.SliceLayout(0, la))
    kb = gl.arange(0, BK, gl.SliceLayout(1, lb))
    acc = gl.zeros((2, BM, 16), gl.float32, mma)
    STEPS: gl.constexpr = gl.cdiv(K, SPLITS * BK)
    for step in gl.static_range(STEPS):
        start = (split * STEPS + step) * BK
        am = (mi[:, None] < M) & (start + ka[None, :] < K)
        bm = (ni[None, :] < N) & (start + kb[:, None] < K)
        if K % (SPLITS * BK) == 0:
            am = (mi[:, None] < M) & (ka[None, :] < BK)
            if N % 16 == 0:
                bm = gl.full((BK, 16), True, gl.int1, lb)
        a = gl.load(X + mi[:, None] * SX + start + ka[None, :], am, 0)
        b = gl.load(W + ni[None, :] * SW + start + kb[:, None], bm, 0)
        a = a.reshape((BM, 2, BK // 2)).permute((1, 0, 2))
        b = b.reshape((2, BK // 2, 16))
        shared_a.store(a)
        a = shared_a.load(gl.DotOperandLayout(0, mma, 8))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))
        acc = gl.amd.cdna4.mfma(a, b, acc)
    values = gl.sum(acc, 0)
    output_layout: gl.constexpr = gl.SliceLayout(0, mma)
    mr = gl.arange(0, BM, gl.SliceLayout(1, output_layout))
    nr = pn * 16 + gl.arange(0, 16, gl.SliceLayout(0, output_layout))
    gl.static_assert(N % 64 == 0)
    offset = (mr[:, None] * SPLITS + split) * PITCH + nr[None, :]
    gl.store(P + offset, values, mr[:, None] < M)


@gluon.jit
def _m16_project_async(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    SPLITS: gl.constexpr,
    PITCH: gl.constexpr,
):
    STAGES: gl.constexpr = 5 if M == 8 else 4
    ROTATE: gl.constexpr = 4 if M == 8 else 1
    SLAB: gl.constexpr = 1024
    pid = gl.program_id(0).to(gl.uint32)
    pn = pid // (SPLITS * 4) * 4 + pid % 4
    split = (pid // 4 % SPLITS + ROTATE * pn) % SPLITS
    BM: gl.constexpr = _artifact_next_power_of_2(M)
    STEPS: gl.constexpr = K // (SPLITS * 64)
    gl.static_assert(K % (SPLITS * 64) == 0)
    gl.static_assert(N % 64 == 0)
    gl.static_assert(M == 8 or M == 16)
    gl.static_assert(STEPS >= STAGES)
    la: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 1], [1, 0])
    lb: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, 1], [0, 1])
    sa: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    sb: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0, 1])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 1]
    )
    da: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    db: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    a0 = gl.allocate_shared_memory(gl.bfloat16, [BM, 64], sa)
    a1 = gl.allocate_shared_memory(gl.bfloat16, [BM, 64], sa)
    a2 = gl.allocate_shared_memory(gl.bfloat16, [BM, 64], sa)
    a3 = gl.allocate_shared_memory(gl.bfloat16, [BM, 64], sa)
    b0 = gl.allocate_shared_memory(gl.bfloat16, [64, 16], sb)
    b1 = gl.allocate_shared_memory(gl.bfloat16, [64, 16], sb)
    b2 = gl.allocate_shared_memory(gl.bfloat16, [64, 16], sb)
    b3 = gl.allocate_shared_memory(gl.bfloat16, [64, 16], sb)
    aa = (a0, a1, a2, a3)
    bb = (b0, b1, b2, b3)
    if M == 8:
        a4 = gl.allocate_shared_memory(gl.bfloat16, [BM, 64], sa)
        b4 = gl.allocate_shared_memory(gl.bfloat16, [64, 16], sb)
        aa = aa + (a4,)
        bb = bb + (b4,)
    mi = gl.arange(0, BM, gl.SliceLayout(1, la))
    ka = gl.arange(0, 64, gl.SliceLayout(0, la))
    kb = gl.arange(0, 64, gl.SliceLayout(1, lb))
    ni = gl.arange(0, 16, gl.SliceLayout(0, lb))
    ao = mi[:, None] * SX + ka[None, :]
    bo = kb[:, None] + ni[None, :] * SW
    xp = X + split * STEPS * 64
    wp = W + pn * 16 * SW + split * STEPS * 64
    for stage in gl.static_range(STAGES):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(aa[stage], xp, ao + stage * 64)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(bb[stage], wp, bo + stage * 64)
        gl.amd.cdna4.async_copy.commit_group()
    acc = gl.zeros((BM, 16), gl.float32, mma)
    for step in gl.static_range(STEPS):
        gl.amd.cdna4.async_copy.wait_group(min(STAGES - 1, STEPS - 1 - step))
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(aa[step % STAGES], da)
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(bb[step % STAGES], db)
        if M == 16:
            acc = gl.amd.cdna4.mfma(a, b, acc)
        if step + STAGES < STEPS:
            gl.inline_asm_elementwise(
                "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
                constraints="=v,~{memory}",
                args=[],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                aa[step % STAGES], xp, ao + (step + STAGES) * 64
            )
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bb[step % STAGES], wp, bo + (step + STAGES) * 64
            )
            gl.amd.cdna4.async_copy.commit_group()
        if M == 8:
            acc = gl.amd.cdna4.mfma(a, b, acc)
    mr = gl.arange(0, BM, gl.SliceLayout(1, mma))
    nr = pn * 16 + gl.arange(0, 16, gl.SliceLayout(0, mma))
    if M == 16:
        offset = (
            (split * (PITCH // SLAB) + nr[None, :] // SLAB) * M + mr[:, None]
        ) * SLAB + nr[None, :] % SLAB
    else:
        offset = (split * M + mr[:, None]) * PITCH + nr[None, :]
    gl.store(P + offset, acc, cache_modifier=".wt")


@gluon.jit
def _m16_finish_q(
    P,
    G,
    Y,
    row,
    eps,
    M: gl.constexpr,
    WIDTH: gl.constexpr,
    OFFSET: gl.constexpr,
    SPLITS: gl.constexpr,
    B: gl.constexpr,
    NW: gl.constexpr,
    VEC: gl.constexpr,
    PITCH: gl.constexpr,
    ROW_MAJOR: gl.constexpr,
    SLAB: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, VEC], [1, 64], [1, NW], [1, 0])
    s = gl.arange(0, _artifact_next_power_of_2(SPLITS), gl.SliceLayout(1, layout))
    d = gl.arange(0, B, gl.SliceLayout(0, layout))
    if M == 16:
        col = OFFSET + d[None, :]
        offset = (
            (s[:, None] * (PITCH // SLAB) + col // SLAB) * M + row
        ) * SLAB + col % SLAB
    elif ROW_MAJOR:
        offset = (row * SPLITS + s[:, None]) * PITCH + OFFSET + d[None, :]
    else:
        offset = (s[:, None] * M + row) * PITCH + OFFSET + d[None, :]
    cache: gl.constexpr = ".cs" if M == 8 or M == 16 else ""
    partial = gl.load(
        P + offset,
        (s[:, None] < SPLITS) & (d[None, :] < WIDTH),
        0,
        cache_modifier=cache,
    )
    x = gl.sum(partial, 0).to(gl.bfloat16).to(gl.float32)
    gamma = gl.load(G + d, d < WIDTH, 0).to(gl.float32)
    x = x * gl.rsqrt(gl.sum(x * x, 0) / WIDTH + eps) * gamma
    gl.store(Y + row * WIDTH + d, x.to(gl.bfloat16), d < WIDTH)


@gluon.jit
def _m16_finish(
    P,
    QG,
    KG,
    QO,
    KO,
    RO,
    eps,
    M: gl.constexpr,
    Q: gl.constexpr,
    KV: gl.constexpr,
    R: gl.constexpr,
    SPLITS: gl.constexpr,
    NW: gl.constexpr,
    VEC: gl.constexpr,
    PITCH: gl.constexpr,
    ROW_MAJOR: gl.constexpr,
    SLAB: gl.constexpr,
    INTERLEAVE: gl.constexpr = False,
):
    if INTERLEAVE:
        row = gl.program_id(0) // 2
        group = gl.program_id(0) % 2
    else:
        row = gl.program_id(0)
        group = gl.program_id(1)
    if group == 0:
        _m16_finish_q(
            P,
            QG,
            QO,
            row,
            eps,
            M,
            Q,
            0,
            SPLITS,
            _artifact_next_power_of_2(Q),
            NW,
            VEC,
            PITCH,
            ROW_MAJOR,
            SLAB,
        )
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, VEC], [1, 64], [1, NW], [1, 0])
        s = gl.arange(0, _artifact_next_power_of_2(SPLITS), gl.SliceLayout(1, layout))
        d = gl.arange(0, _artifact_next_power_of_2(KV + R), gl.SliceLayout(0, layout))
        if M == 16:
            col = Q + d[None, :]
            offset = (
                (s[:, None] * (PITCH // SLAB) + col // SLAB) * M + row
            ) * SLAB + col % SLAB
        elif ROW_MAJOR:
            offset = (row * SPLITS + s[:, None]) * PITCH + Q + d[None, :]
        else:
            offset = (s[:, None] * M + row) * PITCH + Q + d[None, :]
        cache: gl.constexpr = ".cs" if M == 8 or M == 16 else ""
        part = gl.load(
            P + offset,
            (s[:, None] < SPLITS) & (d[None, :] < KV + R),
            0,
            cache_modifier=cache,
        )
        x = gl.sum(part, 0).to(gl.bfloat16).to(gl.float32)
        xv = gl.where(d < KV, x, 0.0)
        gamma = gl.load(KG + d, d < KV, 0).to(gl.float32)
        y = xv * gl.rsqrt(gl.sum(xv * xv, 0) / KV + eps) * gamma
        gl.store(KO + row * KV + d, y.to(gl.bfloat16), d < KV)
        gl.store(RO + row * R + d - KV, x.to(gl.bfloat16), (d >= KV) & (d < KV + R))


@gluon.jit
def _m16_finish_m4(
    P,
    QG,
    KG,
    QO,
    KO,
    RO,
    eps,
    Q: gl.constexpr,
    KV: gl.constexpr,
    R: gl.constexpr,
    SPLITS: gl.constexpr,
    PITCH: gl.constexpr,
):
    row = gl.program_id(0)
    group = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [1, 8], [1, 0])
    s = gl.arange(0, _artifact_next_power_of_2(SPLITS), gl.SliceLayout(1, layout))
    if group == 0:
        q_d = gl.arange(0, _artifact_next_power_of_2(Q), gl.SliceLayout(0, layout))
        q_offset = (row * SPLITS + s[:, None]) * PITCH + q_d[None, :]
        q_part = gl.load(
            P + q_offset,
            (s[:, None] < SPLITS) & (q_d[None, :] < Q),
            0,
            cache_modifier=".cs",
        )
        q_x = gl.sum(q_part, 0).to(gl.bfloat16).to(gl.float32)
        q_gamma = gl.load(QG + q_d, q_d < Q, 0).to(gl.float32)
        q_y = q_x * gl.rsqrt(gl.sum(q_x * q_x, 0) / Q + eps) * q_gamma
        gl.store(QO + row * Q + q_d, q_y.to(gl.bfloat16), q_d < Q)
    else:
        d = gl.arange(0, _artifact_next_power_of_2(KV + R), gl.SliceLayout(0, layout))
        offset = (row * SPLITS + s[:, None]) * PITCH + Q + d[None, :]
        part = gl.load(
            P + offset,
            (s[:, None] < SPLITS) & (d[None, :] < KV + R),
            0,
            cache_modifier=".cs",
        )
        x = gl.sum(part, 0).to(gl.bfloat16).to(gl.float32)
        xv = gl.where(d < KV, x, 0.0)
        gamma = gl.load(KG + d, d < KV, 0).to(gl.float32)
        y = xv * gl.rsqrt(gl.sum(xv * xv, 0) / KV + eps) * gamma
        gl.store(KO + row * KV + d, y.to(gl.bfloat16), d < KV)
        gl.store(RO + row * R + d - KV, x.to(gl.bfloat16), (d >= KV) & (d < KV + R))


def _m16_mla_qkv_a_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    q_gamma: torch.Tensor,
    kv_gamma: torch.Tensor,
    *,
    rope_dim: int = 64,
    eps: float = 1e-05,
):
    m, k = hidden.shape
    q, kv = (q_gamma.numel(), kv_gamma.numel())
    n = q + kv + rope_dim
    splits = 3 if m <= 2 else 6
    use_async = m in (8, 16)
    slab = 1024 if use_async and m == 16 else 512
    pitch = triton.cdiv(n, slab) * slab if m in (8, 16) else 4096 if m == 4 else n
    row_major = m < 16 and m != 8
    partials = torch.empty(
        (m * splits * pitch,), device=hidden.device, dtype=torch.float32
    )
    qo = torch.empty((m, q), device=hidden.device, dtype=torch.bfloat16)
    ko = torch.empty((m, kv), device=hidden.device, dtype=torch.bfloat16)
    ro = torch.empty((m, rope_dim), device=hidden.device, dtype=torch.bfloat16)
    if m <= 2:
        nw, bk = (2, 1024) if m == 1 else (2, 512)
        _m16_project_packed[triton.cdiv(n, 8), splits](
            hidden,
            weight,
            partials,
            m,
            n,
            k,
            hidden.stride(0),
            weight.stride(0),
            splits,
            BK=bk,
            PACK=8,
            BN=8,
            NW=nw,
            num_warps=nw,
        )
    elif use_async:
        _m16_project_async[splits * triton.cdiv(n, 16),](
            hidden,
            weight,
            partials,
            m,
            n,
            k,
            hidden.stride(0),
            weight.stride(0),
            splits,
            PITCH=pitch,
            num_warps=1,
        )
    elif m == 15:
        _m16_project_mfma_dual[splits * triton.cdiv(n, 16),](
            hidden,
            weight,
            partials,
            m,
            n,
            k,
            hidden.stride(0),
            weight.stride(0),
            splits,
            BK=256,
            ROTATE=1,
            PITCH=pitch,
            num_warps=1,
        )
    else:
        _m16_project_mfma[splits * triton.cdiv(n, 16),](
            hidden,
            weight,
            partials,
            m,
            n,
            k,
            hidden.stride(0),
            weight.stride(0),
            splits,
            BK=256,
            CHAINS=2 if m == 4 else 1,
            ROTATE=4 if m == 7 else 0,
            PITCH=pitch,
            num_warps=1,
        )
    if m == 4:
        _m16_finish_m4[m, 2](
            partials,
            q_gamma,
            kv_gamma,
            qo,
            ko,
            ro,
            eps,
            q,
            kv,
            rope_dim,
            splits,
            pitch,
            num_warps=8,
            enable_fp_fusion=False,
        )
    else:
        _m16_finish[(m * 2,) if use_async else (m, 2)](
            partials,
            q_gamma,
            kv_gamma,
            qo,
            ko,
            ro,
            eps,
            m,
            q,
            kv,
            rope_dim,
            splits,
            8,
            4,
            pitch,
            row_major,
            SLAB=slab,
            INTERLEAVE=use_async,
            num_warps=8,
            enable_fp_fusion=False,
        )
    return (qo, ko, ro)


MAX_M = 16
_VARIANT_BY_MAX_M = (
    (1, _m1_mla_qkv_a_norm),
    (2, _m2_mla_qkv_a_norm),
    (4, _m4_mla_qkv_a_norm),
    (8, _m8_mla_qkv_a_norm),
    (16, _m16_mla_qkv_a_norm),
)


def mla_qkv_a_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    q_gamma: torch.Tensor,
    kv_gamma: torch.Tensor,
    *,
    rope_dim: int,
    eps: float,
):
    """Return contiguous BF16 ``(q_lora, k_nope, k_rope)`` for ``1 <= hidden.shape[0] <= MAX_M``."""
    m = hidden.shape[0]
    if m >= 1:
        for max_m, variant in _VARIANT_BY_MAX_M:
            if m <= max_m:
                return variant(
                    hidden, weight, q_gamma, kv_gamma, rope_dim=rope_dim, eps=eps
                )
    raise ValueError(f"mla_qkv_a_norm supports 1 <= M <= {MAX_M}, got M={m}")
