#ifndef HVX_FA_KERNELS_H
#define HVX_FA_KERNELS_H

#include <assert.h>
#include <math.h>
#include "hvx-utils.h"

// Little inner kernels for HVX

#if __HVX_ARCH__ < 79
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#else
#define HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

// This is a bit of a hack because the compiler is struggling to properly inline
// the default hvx_vec_f32_to_f16 with output into the local array.
static __attribute__((unused)) __attribute__((noinline)) void hvx_vec_f32_to_f16_a(void *ptr, HVX_Vector v0, HVX_Vector v1)
{
    *(HVX_Vector *) ptr = hvx_vec_f32_to_f16(v0, v1);
}

// Dot product of two F16 vectors, accumulating to float
static inline void hvx_dot_f16_f16_aa(float * restrict r, const void * restrict x, const void * restrict y, unsigned int n, float s) {
    const HVX_Vector * restrict vx = (const HVX_Vector * restrict) x; // fp16
    const HVX_Vector * restrict vy = (const HVX_Vector * restrict) y; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_VectorPair rsum_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, vx[i], vy[i]);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector y_hf = Q6_V_vand_QV(bmask, vy[i]);
        HVX_Vector x_hf = Q6_V_vand_QV(bmask, vx[i]);

        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_Vector rsum = HVX_OP_ADD_F32(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p));
    rsum = HVX_OP_MUL_F32(hvx_vec_splat_f32(s), hvx_vec_reduce_sum_f32(rsum));
    hvx_vec_store_u(r, 4, rsum);
}

static inline HVX_Vector hvx_dot_f16_f16_aa_rx4(const void * restrict y,
                                                const uint8_t * restrict x,
                                                const size_t stride_x,
                                                const size_t nvec,
                                                const size_t nloe) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector * restrict) x;                   // fp16
    const HVX_Vector * restrict vx1 = (const HVX_Vector * restrict) (x + stride_x);      // fp16
    const HVX_Vector * restrict vx2 = (const HVX_Vector * restrict) (x + stride_x * 2);  // fp16
    const HVX_Vector * restrict vx3 = (const HVX_Vector * restrict) (x + stride_x * 3);  // fp16
    const HVX_Vector * restrict vy  = (const HVX_Vector * restrict) y;                   // fp16

    HVX_VectorPair rsum0_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum1_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum2_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum3_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    for (i = 0; i < nvec; i++) {
        HVX_Vector y_hf  = vy[i];
        HVX_Vector x0_hf = vx0[i];
        HVX_Vector x1_hf = vx1[i];
        HVX_Vector x2_hf = vx2[i];
        HVX_Vector x3_hf = vx3[i];

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    if (nloe) {
        // Load x (fp16) and zero-out unused elements
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector     y_hf  = Q6_V_vand_QV(bmask, vy[i]);
        HVX_Vector     x0_hf = Q6_V_vand_QV(bmask, vx0[i]);
        HVX_Vector     x1_hf = Q6_V_vand_QV(bmask, vx1[i]);
        HVX_Vector     x2_hf = Q6_V_vand_QV(bmask, vx2[i]);
        HVX_Vector     x3_hf = Q6_V_vand_QV(bmask, vx3[i]);

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    HVX_Vector rsum0 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p));
    HVX_Vector rsum1 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p));
    HVX_Vector rsum2 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum2_p), Q6_V_hi_W(rsum2_p));
    HVX_Vector rsum3 = HVX_OP_ADD_F32(Q6_V_lo_W(rsum3_p), Q6_V_hi_W(rsum3_p));

    HVX_Vector_x4 rsum0123 = { .v = { rsum0, rsum1, rsum2, rsum3 } };
    return hvx_vec_reduce_sum_f32x4(rsum0123);
}

static inline HVX_Vector hvx_dot_f16_f16_aa_rx32(const void * restrict y,
                                                 const uint8_t * restrict x,
                                                 const size_t stride_x,
                                                 const size_t n,
                                                 float        s) {

    const size_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    const size_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector   sums = Q6_V_vzero();
    const size_t stride_x_4 = stride_x * 4;
    for (uint32_t j = 0; j < VLEN_FP32; j += 4) {
        HVX_Vector     sums_x4 = hvx_dot_f16_f16_aa_rx4(y, x, stride_x, nvec, nloe);
        HVX_VectorPred pred    = Q6_Q_vsetq_R(j * SIZEOF_FP32);
        sums                   = Q6_V_vmux_QVV(pred, sums, sums_x4);
        x += stride_x_4;
    }

    return HVX_OP_MUL_F32(hvx_vec_splat_f32(s), sums);
}

// MAD: y (F32) += x (F16) * s (F16)
static inline void hvx_mad_f32_f16_aa(float * restrict y, const void * restrict x, const __fp16 * restrict s, uint32_t n) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x;

    HVX_VectorPair * restrict vy_p = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector S0 = hvx_vec_splat_f16(*s);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    if (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        if (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        if (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

// MAD: y (F32) += x0 (F16) * s0 (F16) + x1 (F16) * s1 (F16)
static inline void hvx_mad_f32_f16_aa_rx2(float * restrict y, const void * restrict x0, const void * restrict x1,
                                          const __fp16 * restrict s0, const __fp16 * restrict s1, uint32_t n) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x0;
    const HVX_Vector * restrict vx1 = (const HVX_Vector *) x1;

    HVX_VectorPair * restrict vy_p  = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy        = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    HVX_Vector S0 = hvx_vec_splat_f16(*s0);
    HVX_Vector S1 = hvx_vec_splat_f16(*s1);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    if (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        if (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        if (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}
static inline void hvx_mad_f32_f16_aa_vec(float * restrict y, const void * restrict x, HVX_Vector S0, uint32_t n) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x;

    HVX_VectorPair * restrict vy_p = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    if (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        if (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        if (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

static inline void hvx_mad_f32_f16_aa_rx2_vec(float * restrict y, const void * restrict x0, const void * restrict x1,
                                          HVX_Vector S0, HVX_Vector S1, uint32_t n) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x0;
    const HVX_Vector * restrict vx1 = (const HVX_Vector *) x1;

    HVX_VectorPair * restrict vy_p  = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy        = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    if (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        if (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        if (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

// PV for one K/V block with the f32 accumulator held in registers, 4 vector pairs (256 floats) at a time.
// The stream worker's loop calls hvx_mad_f32_f16_aa_rx2_vec per key pair, which loads and stores the whole
// accumulator from VTCM every two keys. Here each element still receives exactly the same sequence of
// hvx_vec_mpyacc_f32_f16 updates (key j, then j+1, ..., odd tail last), so the result is bit-identical;
// only the accumulator round trips go away. Caller guarantees DV % 256 == 0.
static inline void hvx_pv_block_regacc(float * restrict y, const uint8_t * restrict v_base, size_t v_stride,
                                       HVX_Vector P, uint32_t nkeys, uint32_t DV) {
    HVX_VectorPair * restrict vy_p = (HVX_VectorPair *) y;
    const uint32_t npairs = DV / VLEN_FP16;  // one fp16 V vector <-> one f32 accumulator pair

    for (uint32_t i0 = 0; i0 < npairs; i0 += 4) {
        HVX_VectorPair a0 = vy_p[i0 + 0];
        HVX_VectorPair a1 = vy_p[i0 + 1];
        HVX_VectorPair a2 = vy_p[i0 + 2];
        HVX_VectorPair a3 = vy_p[i0 + 3];

        const uint8_t * v_ptr = v_base;
        uint32_t j = 0;
        for (; j + 1 < nkeys; j += 2) {
            const HVX_Vector S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
            const HVX_Vector S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
            const HVX_Vector * vx0 = (const HVX_Vector *) v_ptr + i0;
            const HVX_Vector * vx1 = (const HVX_Vector *) (v_ptr + v_stride) + i0;

            a0 = hvx_vec_mpyacc_f32_f16(a0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
            a0 = hvx_vec_mpyacc_f32_f16(a0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
            a1 = hvx_vec_mpyacc_f32_f16(a1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
            a1 = hvx_vec_mpyacc_f32_f16(a1, Q6_Vh_vshuff_Vh(vx1[1]), S1);
            a2 = hvx_vec_mpyacc_f32_f16(a2, Q6_Vh_vshuff_Vh(vx0[2]), S0);
            a2 = hvx_vec_mpyacc_f32_f16(a2, Q6_Vh_vshuff_Vh(vx1[2]), S1);
            a3 = hvx_vec_mpyacc_f32_f16(a3, Q6_Vh_vshuff_Vh(vx0[3]), S0);
            a3 = hvx_vec_mpyacc_f32_f16(a3, Q6_Vh_vshuff_Vh(vx1[3]), S1);

            v_ptr += 2 * v_stride;
        }
        if (j < nkeys) {  // odd block size: the last key alone, as hvx_mad_f32_f16_aa_vec does
            const HVX_Vector S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
            const HVX_Vector * vx0 = (const HVX_Vector *) v_ptr + i0;
            a0 = hvx_vec_mpyacc_f32_f16(a0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
            a1 = hvx_vec_mpyacc_f32_f16(a1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
            a2 = hvx_vec_mpyacc_f32_f16(a2, Q6_Vh_vshuff_Vh(vx0[2]), S0);
            a3 = hvx_vec_mpyacc_f32_f16(a3, Q6_Vh_vshuff_Vh(vx0[3]), S0);
        }

        vy_p[i0 + 0] = a0;
        vy_p[i0 + 1] = a1;
        vy_p[i0 + 2] = a2;
        vy_p[i0 + 3] = a3;
    }
}

static inline void hvx_scale_vec_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const uint32_t n, HVX_Vector vs) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);

    const HVX_Vector * restrict vsrc = (const HVX_Vector * restrict) src;
    HVX_Vector * restrict vdst       = (HVX_Vector * restrict) dst;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    #pragma unroll(4)
    for (; i < nvec; ++i) {
        vdst[i] = HVX_OP_MUL_F32(vsrc[i], vs);
    }
    if (nloe) {
        hvx_vec_store_a(&vdst[i], nloe * sizeof(float), HVX_OP_MUL_F32(vsrc[i], vs));
    }
}

// ---- qf32-accumulating variants (HTP_FA_FLAG_QF32) --------------------------------------------------------------
// On < v79 hvx_vec_mpyacc_f32_f16 does, for every multiply, vmpy -> two qf32 adds -> two conversions back to IEEE
// f32, and each conversion waits for its add: the QK loop ran ~1 instruction per packet. These keep the running sums
// in qf32 and convert once at the end (per dot product for QK, per 64-key block for PV). One rounding instead of one
// per multiply, so results are not bit-identical to the functions above.
// Simulator, one thread, one 64-key block, DK = DV = 256 (2026-09-24): QK 4602 -> 2121 cycles, PV 2616 -> 1803.
// On >= v79 they fall back to the functions above (native IEEE vmpyacc, nothing to gain).

#if __HVX_ARCH__ < 79

static inline HVX_Vector hvx_qf32_pair_sum(HVX_VectorPair m) {
    return Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(m), Q6_V_hi_W(m));
}

// One level of a transpose-add tree: interleave a and b at w bytes and add the halves (qf32 in, qf32 out).
static inline HVX_Vector hvx_qf32_tree_level(HVX_Vector a, HVX_Vector b, int w) {
    const HVX_VectorPair p = Q6_W_vshuff_VVR(b, a, w);
    return Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p));
}

// Dot products of y with 4 rows of x, left as a qf32 vector where lane 4k + r holds a partial sum of row r
// (the first two levels of the 32-row tree below).
static inline HVX_Vector hvx_dot_f16_f16_aa_rx4_qf_partial(const void * restrict y,
                                                           const uint8_t * restrict x,
                                                           const size_t stride_x,
                                                           const size_t nvec,
                                                           const size_t nloe) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector * restrict) x;
    const HVX_Vector * restrict vx1 = (const HVX_Vector * restrict) (x + stride_x);
    const HVX_Vector * restrict vx2 = (const HVX_Vector * restrict) (x + stride_x * 2);
    const HVX_Vector * restrict vx3 = (const HVX_Vector * restrict) (x + stride_x * 3);
    const HVX_Vector * restrict vy  = (const HVX_Vector * restrict) y;

    HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero(), a2 = Q6_V_vzero(), a3 = Q6_V_vzero();  // qf32 zero

    uint32_t i = 0;
    if (nvec > 0) {
        const HVX_Vector y_hf = vy[0];
        a0 = hvx_qf32_pair_sum(Q6_Wqf32_vmpy_VhfVhf(vx0[0], y_hf));
        a1 = hvx_qf32_pair_sum(Q6_Wqf32_vmpy_VhfVhf(vx1[0], y_hf));
        a2 = hvx_qf32_pair_sum(Q6_Wqf32_vmpy_VhfVhf(vx2[0], y_hf));
        a3 = hvx_qf32_pair_sum(Q6_Wqf32_vmpy_VhfVhf(vx3[0], y_hf));
        for (i = 1; i < nvec; i++) {
            const HVX_Vector yv = vy[i];
            a0 = Q6_Vqf32_vadd_Vqf32Vqf32(a0, hvx_qf32_pair_sum(Q6_Wqf32_vmpy_VhfVhf(vx0[i], yv)));
            a1 = Q6_Vqf32_vadd_Vqf32Vqf32(a1, hvx_qf32_pair_sum(Q6_Wqf32_vmpy_VhfVhf(vx1[i], yv)));
            a2 = Q6_Vqf32_vadd_Vqf32Vqf32(a2, hvx_qf32_pair_sum(Q6_Wqf32_vmpy_VhfVhf(vx2[i], yv)));
            a3 = Q6_Vqf32_vadd_Vqf32Vqf32(a3, hvx_qf32_pair_sum(Q6_Wqf32_vmpy_VhfVhf(vx3[i], yv)));
        }
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector     y_hf  = Q6_V_vand_QV(bmask, vy[i]);
        a0 = Q6_Vqf32_vadd_Vqf32Vqf32(a0, hvx_qf32_pair_sum(Q6_Wqf32_vmpy_VhfVhf(Q6_V_vand_QV(bmask, vx0[i]), y_hf)));
        a1 = Q6_Vqf32_vadd_Vqf32Vqf32(a1, hvx_qf32_pair_sum(Q6_Wqf32_vmpy_VhfVhf(Q6_V_vand_QV(bmask, vx1[i]), y_hf)));
        a2 = Q6_Vqf32_vadd_Vqf32Vqf32(a2, hvx_qf32_pair_sum(Q6_Wqf32_vmpy_VhfVhf(Q6_V_vand_QV(bmask, vx2[i]), y_hf)));
        a3 = Q6_Vqf32_vadd_Vqf32Vqf32(a3, hvx_qf32_pair_sum(Q6_Wqf32_vmpy_VhfVhf(Q6_V_vand_QV(bmask, vx3[i]), y_hf)));
    }

    return hvx_qf32_tree_level(hvx_qf32_tree_level(a0, a1, 4), hvx_qf32_tree_level(a2, a3, 4), 8);
}

// 32 dot products (lane j = row j), finished by one 5-level shuffle/add tree instead of a reduce per 4 rows plus
// a vmux: half the reduction instructions and every level is independent work. Sim, 64-key block, DK = 256:
// 2121 cycles (reduce-per-4 qf32 2703, as shipped 4602); max |error| vs a double reference 6.3e-7, same as both.
static inline HVX_Vector hvx_dot_f16_f16_aa_rx32_qf(const void * restrict y,
                                                    const uint8_t * restrict x,
                                                    const size_t stride_x,
                                                    const size_t n,
                                                    float        s) {
    const size_t nvec = n / VLEN_FP16;
    const size_t nloe = n % VLEN_FP16;
    const size_t s4   = stride_x * 4;

    const HVX_Vector t0 = hvx_dot_f16_f16_aa_rx4_qf_partial(y, x + 0 * s4, stride_x, nvec, nloe);
    const HVX_Vector t1 = hvx_dot_f16_f16_aa_rx4_qf_partial(y, x + 1 * s4, stride_x, nvec, nloe);
    const HVX_Vector t2 = hvx_dot_f16_f16_aa_rx4_qf_partial(y, x + 2 * s4, stride_x, nvec, nloe);
    const HVX_Vector t3 = hvx_dot_f16_f16_aa_rx4_qf_partial(y, x + 3 * s4, stride_x, nvec, nloe);
    const HVX_Vector t4 = hvx_dot_f16_f16_aa_rx4_qf_partial(y, x + 4 * s4, stride_x, nvec, nloe);
    const HVX_Vector t5 = hvx_dot_f16_f16_aa_rx4_qf_partial(y, x + 5 * s4, stride_x, nvec, nloe);
    const HVX_Vector t6 = hvx_dot_f16_f16_aa_rx4_qf_partial(y, x + 6 * s4, stride_x, nvec, nloe);
    const HVX_Vector t7 = hvx_dot_f16_f16_aa_rx4_qf_partial(y, x + 7 * s4, stride_x, nvec, nloe);

    const HVX_Vector u0 = hvx_qf32_tree_level(t0, t1, 16), u1 = hvx_qf32_tree_level(t2, t3, 16);
    const HVX_Vector u2 = hvx_qf32_tree_level(t4, t5, 16), u3 = hvx_qf32_tree_level(t6, t7, 16);
    const HVX_Vector r  = hvx_qf32_tree_level(hvx_qf32_tree_level(u0, u1, 32), hvx_qf32_tree_level(u2, u3, 32), 64);

    return HVX_OP_MUL_F32(hvx_vec_splat_f32(s), Q6_Vsf_equals_Vqf32(r));
}

#define HVX_PV_QF_ACC(k, xv, S)                                         \
    do {                                                                \
        const HVX_VectorPair m_ = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(xv), S); \
        l##k = Q6_Vqf32_vadd_Vqf32Vqf32(l##k, Q6_V_lo_W(m_));           \
        h##k = Q6_Vqf32_vadd_Vqf32Vqf32(h##k, Q6_V_hi_W(m_));           \
    } while (0)

// Same traversal as hvx_pv_block_regacc; the four accumulator pairs enter as f32 (one qf32 add each), stay qf32
// through the block and are converted back once. Caller guarantees DV % 256 == 0.
static inline void hvx_pv_block_regacc_qf(float * restrict y, const uint8_t * restrict v_base, size_t v_stride,
                                          HVX_Vector P, uint32_t nkeys, uint32_t DV) {
    HVX_VectorPair * restrict vy_p = (HVX_VectorPair *) y;
    const uint32_t npairs = DV / VLEN_FP16;
    const HVX_Vector z = Q6_V_vzero();

    for (uint32_t i0 = 0; i0 < npairs; i0 += 4) {
        HVX_Vector l0 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vy_p[i0 + 0]), z), h0 = Q6_Vqf32_vadd_VsfVsf(Q6_V_hi_W(vy_p[i0 + 0]), z);
        HVX_Vector l1 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vy_p[i0 + 1]), z), h1 = Q6_Vqf32_vadd_VsfVsf(Q6_V_hi_W(vy_p[i0 + 1]), z);
        HVX_Vector l2 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vy_p[i0 + 2]), z), h2 = Q6_Vqf32_vadd_VsfVsf(Q6_V_hi_W(vy_p[i0 + 2]), z);
        HVX_Vector l3 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vy_p[i0 + 3]), z), h3 = Q6_Vqf32_vadd_VsfVsf(Q6_V_hi_W(vy_p[i0 + 3]), z);

        const uint8_t * v_ptr = v_base;
        uint32_t j = 0;
        for (; j + 1 < nkeys; j += 2) {
            const HVX_Vector S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
            const HVX_Vector S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
            const HVX_Vector * vx0 = (const HVX_Vector *) v_ptr + i0;
            const HVX_Vector * vx1 = (const HVX_Vector *) (v_ptr + v_stride) + i0;
            HVX_PV_QF_ACC(0, vx0[0], S0); HVX_PV_QF_ACC(0, vx1[0], S1);
            HVX_PV_QF_ACC(1, vx0[1], S0); HVX_PV_QF_ACC(1, vx1[1], S1);
            HVX_PV_QF_ACC(2, vx0[2], S0); HVX_PV_QF_ACC(2, vx1[2], S1);
            HVX_PV_QF_ACC(3, vx0[3], S0); HVX_PV_QF_ACC(3, vx1[3], S1);
            v_ptr += 2 * v_stride;
        }
        if (j < nkeys) {
            const HVX_Vector S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
            const HVX_Vector * vx0 = (const HVX_Vector *) v_ptr + i0;
            HVX_PV_QF_ACC(0, vx0[0], S0);
            HVX_PV_QF_ACC(1, vx0[1], S0);
            HVX_PV_QF_ACC(2, vx0[2], S0);
            HVX_PV_QF_ACC(3, vx0[3], S0);
        }

        vy_p[i0 + 0] = Q6_W_vcombine_VV(Q6_Vsf_equals_Vqf32(h0), Q6_Vsf_equals_Vqf32(l0));
        vy_p[i0 + 1] = Q6_W_vcombine_VV(Q6_Vsf_equals_Vqf32(h1), Q6_Vsf_equals_Vqf32(l1));
        vy_p[i0 + 2] = Q6_W_vcombine_VV(Q6_Vsf_equals_Vqf32(h2), Q6_Vsf_equals_Vqf32(l2));
        vy_p[i0 + 3] = Q6_W_vcombine_VV(Q6_Vsf_equals_Vqf32(h3), Q6_Vsf_equals_Vqf32(l3));
    }
}

#undef HVX_PV_QF_ACC

#else

#define hvx_dot_f16_f16_aa_rx32_qf hvx_dot_f16_f16_aa_rx32
#define hvx_pv_block_regacc_qf     hvx_pv_block_regacc

#endif

// True when every one of the first nkeys mask entries is -inf, i.e. the whole K/V block contributes exactly zero:
// its scores become -65504 + s after the -inf -> -65504 substitution, the running max does not move (it starts at
// -10000), so the rescale factor is exp2(0) = 1 and P = exp2(<= -80000) = 0 (HTP_FA_FLAG_SKIPMASK).
static inline bool hvx_fa_block_all_masked(const __fp16 * restrict m, uint32_t nkeys) {
    const HVX_Vector     vinf = Q6_Vh_vsplat_R(0xFC00);
    const HVX_VectorPred keep = Q6_Q_vsetq2_R(nkeys * sizeof(__fp16));
    // lanes past nkeys count as masked
    const HVX_Vector     mv   = Q6_V_vmux_QVV(keep, *(const HVX_UVector *) m, vinf);
    const HVX_VectorPred inf  = Q6_Q_vcmp_eq_VhVh(mv, vinf);
    // halfword ones: the predicate selects per 16-bit lane. A word splat of 1 (0x00000001) left the odd lanes 0,
    // so a block whose only live key sat at an odd index was taken for fully masked and skipped (first device run
    // 2026-09-24: FLASH_ATTN_EXT 2462 -> 2455 passed, perplexity 12.7498 -> 12.7381).
    HVX_Vector live = Q6_V_vmux_QVV(inf, Q6_V_vzero(), Q6_Vh_vsplat_R(1));
    for (int s = 64; s >= 4; s >>= 1) {
        live = Q6_V_vor_VV(live, Q6_V_vror_VR(live, s));
    }
    union { HVX_Vector v; int32_t w[32]; } u = { live };
    return u.w[0] == 0;
}

#endif /* HVX_FA_KERNELS_H */
