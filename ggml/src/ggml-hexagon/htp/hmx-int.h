// Integer-HMX Q4_0 x F32 -> F32 matmul, for Hexagon parts whose fp16 HMX path does not compute.
//
// On SM8735 (Hexagon v73) fp16 HMX stores exactly zero, so llama.cpp's HMX kernels cannot run there,
// but the integer path computes an exact matmul (measured on the device against scalar references and
// Qualcomm's v73 simulator, same bits). This file is that path, built only from pieces measured first
// (probe headers hmxver/hmxconv/hmxq4qt/hmxcomb/hmxpipe/hmxq4cv, HTP_DEBUG builds):
//
//   tiles    A[m][k] u16 at 128*(m>>1) + 4k + 2(m&1)   (32 x 32, 2 KB; the unit reads 2048 bytes)
//            W[k][c] int8 at 128*(k>>2) + 4c + (k&3)   (1 KB)
//            Y int16 at out[64*(m>>1) + 2c + (m&1)]      (row pairs interleaved per column)
//   store    out = floor(Q * scale_c / 2), Q = floor(sum a*w / 256); record scale 2.0 gives Q mod 2^16
//   Q4_0     activations u = q + 32768 with q clamped to [-32767, 32767] and s_a = max|x| / 32767 per row;
//            weights w = nibble - 8. Then QT = sext16(u_out - 128 * sum_k w) exactly, and
//            y = s_a * (sum_b 256 d_b QT_b + 128 sum_b d_b)   (the +0.5 quantum hoisted per column).
//   convert  int16 -> float by sign-extending in the integer domain and vconv (the qf32 magic-number
//            trick is NOT exact on v73: 16,383 of 65,536 values off by one).
//   weights  the host's repack_q4_0_tiled 576-byte tiles, converted on HVX (v73 has no vcvt hf->sf, so
//            the fp16 scales are converted by bits).
//   threads  one HMX producer (the HMX queue thread in the op path; it holds the lock) and the work
//            queue's HVX workers as consumers, per-consumer SPSC rings with release/acquire counters.
//            In session start only the calling thread can take the HMX lock - running HMX without it
//            crashed the cDSP once (precise exception 0x1801 at `bias = mxmem`).
//
// K is split into segments of at most HMXI_SEG blocks so one ring slot stays small at any K.

#ifndef HMX_INT_H
#define HMX_INT_H

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

#include "hmx-int-plan.h"

// ------------------------------------------------------------------------------------------------
// HMX instructions (straight-line inline asm in noinline functions: an HMX pair behind an `if` inside
// a loop made hexagon-clang spin for 12+ minutes)

static void __attribute__((noinline)) hmxi_bias(const void * rec) { asm volatile("bias = mxmem(%0)\n" :: "r"(rec) : "memory"); }
static void __attribute__((noinline)) hmxi_clr(void)              { asm volatile("mxclracc\n" ::: "memory"); }
static void __attribute__((noinline)) hmxi_mac(const void * a, const void * w) {
    asm volatile("{\n"
                 "  activation.ub = mxmem(%0, %2):deep\n"
                 "  weight.b      = mxmem(%1, %3)\n"
                 "}\n" :: "r"(a), "r"(w), "r"(2047), "r"(1023) : "memory");
}
static void __attribute__((noinline)) hmxi_store(void * o) { asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n" :: "r"(o), "r"(0) : "memory"); }

// nb blocks: act tiles 2 KB apart, weight tiles 1 KB apart, one store per block into out (2 KB apart)
static void __attribute__((noinline)) hmxi_blocks(const uint8_t * act, const uint8_t * wgt, uint8_t * out, const void * rec, int nb) {
    for (int b = 0; b < nb; b++) {
        hmxi_bias(rec); hmxi_clr();
        hmxi_mac(act + b * 2048, wgt + b * 1024);
        hmxi_store(out + b * 2048);
    }
}

// the conversion record: fp16 2.0 in the low 32 words, zero high half (store returns Q mod 2^16)
static inline void hmxi_init_record(uint32_t * rec) {
    for (int i = 0; i < 64; i++) rec[i] = i < 32 ? 0x00004000u : 0u;
}

// ------------------------------------------------------------------------------------------------
// Activations: rows in pairs -> u16 tiles. x0/x1 may be NULL (padding row -> quantised zero = 32768).
// sa gets max|x| / 32767, or 0 for an all-zero (or padding) row so it contributes exactly 0.

// Four independent max chains, then a rotate-and-max horizontal reduction instead of spilling the vector and
// comparing 32 words one by one. max is associative and commutative, and the words are |x| bit patterns (sign
// cleared), so signed-word max orders them exactly as the old scalar loop did: the result is bit-identical
// (checked in hexagon-sim against the old version, scratchpad gemvsim/pref_sim2.c / pref_sim3.c).
static inline float hmxi_absmax_row(const float * restrict x, int K) {
    const HVX_Vector mask = Q6_V_vsplat_R(0x7fffffff);
    HVX_Vector m0 = Q6_V_vzero(), m1 = Q6_V_vzero(), m2 = Q6_V_vzero(), m3 = Q6_V_vzero();
    int i = 0;
    for (; i + 128 <= K; i += 128) {
        m0 = Q6_Vw_vmax_VwVw(m0, Q6_V_vand_VV(*(const HVX_UVector *) (x + i),      mask));
        m1 = Q6_Vw_vmax_VwVw(m1, Q6_V_vand_VV(*(const HVX_UVector *) (x + i + 32), mask));
        m2 = Q6_Vw_vmax_VwVw(m2, Q6_V_vand_VV(*(const HVX_UVector *) (x + i + 64), mask));
        m3 = Q6_Vw_vmax_VwVw(m3, Q6_V_vand_VV(*(const HVX_UVector *) (x + i + 96), mask));
    }
    for (; i < K; i += 32) m0 = Q6_Vw_vmax_VwVw(m0, Q6_V_vand_VV(*(const HVX_UVector *) (x + i), mask));
    HVX_Vector m = Q6_Vw_vmax_VwVw(Q6_Vw_vmax_VwVw(m0, m1), Q6_Vw_vmax_VwVw(m2, m3));
    for (int s = 64; s >= 4; s >>= 1) m = Q6_Vw_vmax_VwVw(m, Q6_V_vror_VR(m, s));
    union { HVX_Vector v; int32_t w[32]; } u = { m };
    union { int32_t i; float f; } r = { u.w[0] > 0 ? u.w[0] : 0 };
    return r.f;
}

static inline HVX_Vector hmxi_quant_block(const float * x, HVX_Vector vinv) {
    if (!x) return Q6_V_vsplat_R(32768);
    HVX_Vector q = Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *) x, vinv)));
    q = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(q, Q6_V_vsplat_R(-32767)), Q6_V_vsplat_R(32767));
    return Q6_Vw_vadd_VwVw(q, Q6_V_vsplat_R(32768));
}

// Both rows present - every pair except the last one of an odd row count. The general loop below issued ONE HVX op
// per packet and started row 1's load only after row 0's whole vmpy -> sf -> w -> max -> min -> add chain (seen in
// the disassembly): the NULL test inside hmxi_quant_block splits the loop into blocks and float* x vs uint8_t* act_rt
// may alias, so the scheduler could not overlap anything. Here there is no branch, the pointers are restrict, and two
// blocks x two rows = four independent chains are written side by side. Same instructions per value, so the output
// is bit-identical. hexagon-sim, K=1536: 3159 -> 1533 cycles per row pair (-51.5%), 2.14 -> 2.84 ops per packet.
#define HMXI_QCONV(xv, vinv) Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf((xv), (vinv))))
#define HMXI_QCLAMP(q)       Q6_Vw_vadd_VwVw(Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw((q), qlo), qhi), qoff)
static void __attribute__((noinline)) hmxi_quant_pair2(const float * restrict x0, const float * restrict x1, int B,
                                                       float * restrict sa0, float * restrict sa1,
                                                       uint8_t * restrict act_rt, int pr) {
    const float m0 = hmxi_absmax_row(x0, B * 32);
    const float m1 = hmxi_absmax_row(x1, B * 32);
    *sa0 = m0 > 0.0f ? m0 / 32767.0f : 0.0f;
    *sa1 = m1 > 0.0f ? m1 / 32767.0f : 0.0f;
    union { float f; int32_t i; } i0 = { m0 > 0.0f ? 32767.0f / m0 : 0.0f }, i1 = { m1 > 0.0f ? 32767.0f / m1 : 0.0f };
    const HVX_Vector v0 = Q6_V_vsplat_R(i0.i), v1 = Q6_V_vsplat_R(i1.i);
    const HVX_Vector qlo = Q6_V_vsplat_R(-32767), qhi = Q6_V_vsplat_R(32767), qoff = Q6_V_vsplat_R(32768);
    int b = 0;
    for (; b + 2 <= B; b += 2) {
        const HVX_Vector a0 = *(const HVX_UVector *) (x0 + b * 32);
        const HVX_Vector a1 = *(const HVX_UVector *) (x1 + b * 32);
        const HVX_Vector c0 = *(const HVX_UVector *) (x0 + b * 32 + 32);
        const HVX_Vector c1 = *(const HVX_UVector *) (x1 + b * 32 + 32);
        HVX_Vector qa0 = HMXI_QCONV(a0, v0), qa1 = HMXI_QCONV(a1, v1), qc0 = HMXI_QCONV(c0, v0), qc1 = HMXI_QCONV(c1, v1);
        qa0 = HMXI_QCLAMP(qa0); qa1 = HMXI_QCLAMP(qa1); qc0 = HMXI_QCLAMP(qc0); qc1 = HMXI_QCLAMP(qc1);
        *(HVX_Vector *) (act_rt + (size_t) b * 2048 + pr * 128)       = Q6_Vh_vshuff_Vh(Q6_Vh_vpacke_VwVw(qa1, qa0));
        *(HVX_Vector *) (act_rt + (size_t) (b + 1) * 2048 + pr * 128) = Q6_Vh_vshuff_Vh(Q6_Vh_vpacke_VwVw(qc1, qc0));
    }
    for (; b < B; b++) {
        const HVX_Vector qa0 = HMXI_QCLAMP(HMXI_QCONV(*(const HVX_UVector *) (x0 + b * 32), v0));
        const HVX_Vector qa1 = HMXI_QCLAMP(HMXI_QCONV(*(const HVX_UVector *) (x1 + b * 32), v1));
        *(HVX_Vector *) (act_rt + (size_t) b * 2048 + pr * 128) = Q6_Vh_vshuff_Vh(Q6_Vh_vpacke_VwVw(qa1, qa0));
    }
}
#undef HMXI_QCONV
#undef HMXI_QCLAMP

// one row pair (pr = 0..15 inside row tile at act_rt), all B blocks
static void __attribute__((noinline)) hmxi_quant_pair(const float * x0, const float * x1, int B, float * sa0, float * sa1,
                                                      uint8_t * act_rt, int pr) {
    if (x0 && x1 && sa1) {
        hmxi_quant_pair2(x0, x1, B, sa0, sa1, act_rt, pr);
        return;
    }
    const float m0 = x0 ? hmxi_absmax_row(x0, B * 32) : 0.0f;
    const float m1 = x1 ? hmxi_absmax_row(x1, B * 32) : 0.0f;
    *sa0 = m0 > 0.0f ? m0 / 32767.0f : 0.0f;
    if (sa1) *sa1 = m1 > 0.0f ? m1 / 32767.0f : 0.0f;
    union { float f; int32_t i; } i0 = { m0 > 0.0f ? 32767.0f / m0 : 0.0f }, i1 = { m1 > 0.0f ? 32767.0f / m1 : 0.0f };
    const HVX_Vector v0 = Q6_V_vsplat_R(i0.i), v1 = Q6_V_vsplat_R(i1.i);
    for (int b = 0; b < B; b++) {
        HVX_Vector u0 = hmxi_quant_block(x0 ? x0 + b * 32 : NULL, v0);
        HVX_Vector u1 = hmxi_quant_block(x1 ? x1 + b * 32 : NULL, v1);
        // [u0 low halves (32), u1 low halves (32)] then interleave: halfword 2k + (row & 1)
        HVX_Vector h = Q6_Vh_vpacke_VwVw(u1, u0);
        *(HVX_Vector *) (act_rt + (size_t) b * 2048 + pr * 128) = Q6_Vh_vshuff_Vh(h);
    }
}

// ------------------------------------------------------------------------------------------------
// Weights: one host-repacked Q4_0 tile (576 B: byte[cp*32+n] = q[n][2cp] | q[n][2cp+1] << 4, then
// 32 fp16 d) -> int8 HMX tile (q - 8), cv (word c = C | C<<16, C = 128*sum w mod 2^16), dv (256*d).
// Every other tile is only 64-byte aligned in DDR: unaligned loads; the scale load starts at byte 448
// so nothing past the tile is read.

static inline __attribute__((always_inline)) void hmxi_cvt_body(const uint8_t * restrict src, HVX_Vector * restrict wt,
                                                                HVX_Vector * restrict cv, HVX_Vector * restrict dv) {
    const HVX_Vector k0f  = Q6_V_vsplat_R(0x0f0f0f0f);
    const HVX_Vector k08  = Q6_V_vsplat_R(0x08080808);
    HVX_Vector       sumq = Q6_V_vzero();
#pragma unroll
    for (int v = 0; v < 4; v++) {
        HVX_Vector     x  = *(const HVX_UVector *) (src + v * 128);
        HVX_Vector     lo = Q6_V_vand_VV(x, k0f);
        HVX_Vector     hi = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(x, 4), k0f);
        HVX_VectorPair p  = Q6_W_vshuff_VVR(hi, lo, -1);
        HVX_Vector     t0 = Q6_Vh_vshuff_Vh(Q6_V_lo_W(p));
        HVX_Vector     t1 = Q6_Vh_vshuff_Vh(Q6_V_hi_W(p));
        sumq = Q6_Vuw_vrmpyacc_VuwVubRub(sumq, t0, 0x01010101);
        sumq = Q6_Vuw_vrmpyacc_VuwVubRub(sumq, t1, 0x01010101);
        wt[2 * v]     = Q6_Vb_vsub_VbVb(t0, k08);
        wt[2 * v + 1] = Q6_Vb_vsub_VbVb(t1, k08);
    }
    HVX_Vector c = Q6_V_vand_VV(Q6_Vw_vasl_VwR(Q6_Vw_vsub_VwVw(sumq, Q6_V_vsplat_R(256)), 7), Q6_V_vsplat_R(0xffff));
    *cv = Q6_V_vor_VV(c, Q6_Vw_vasl_VwR(c, 16));

    HVX_Vector     xs = *(const HVX_UVector *) (src + 448);
    HVX_VectorPair w  = Q6_Wuw_vzxt_Vuh(xs);
    HVX_VectorPair s  = Q6_W_vshuff_VVR(Q6_V_hi_W(w), Q6_V_lo_W(w), -4);
    HVX_Vector     hb = Q6_V_hi_W(s);                        // word c = fp16 bits of d[c]
    const HVX_Vector sign = Q6_Vw_vasl_VwR(Q6_V_vand_VV(hb, Q6_V_vsplat_R(0x8000)), 16);
    const HVX_Vector mag  = Q6_V_vand_VV(hb, Q6_V_vsplat_R(0x7fff));
    const HVX_Vector nrm  = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(mag, 13), Q6_V_vsplat_R(0x3c000000));
    const HVX_Vector sub  = Q6_Vw_vsub_VwVw(Q6_Vsf_equals_Vw(mag), Q6_V_vsplat_R(16 << 23));
    HVX_Vector r = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(mag, Q6_V_vsplat_R(0x3ff)), nrm, sub);
    r = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(mag, Q6_V_vzero()), Q6_V_vzero(), r);
    *dv = Q6_V_vor_VV(r, sign);
}

static void __attribute__((noinline)) hmxi_cvt_tile(const uint8_t * src, HVX_Vector * wt, HVX_Vector * cv, HVX_Vector * dv) {
    hmxi_cvt_body(src, wt, cv, dv);
}

// Two tiles in one call. On its own a tile runs 53 packets in 168 cycles - mostly waiting on the unaligned loads and
// the shuffle / vrmpy latencies; with two independent tiles scheduled together the stalls of one are filled with the
// other's work. Outputs are disjoint (restrict holds) and bit-identical to two hmxi_cvt_tile calls.
// hexagon-sim: 168 -> 117 cycles per tile (-30.4%), 2.04 -> 2.89 ops per packet.
static void __attribute__((noinline)) hmxi_cvt_tile2(const uint8_t * restrict s0, const uint8_t * restrict s1,
                                                     HVX_Vector * restrict w0, HVX_Vector * restrict c0, HVX_Vector * restrict d0,
                                                     HVX_Vector * restrict w1, HVX_Vector * restrict c1, HVX_Vector * restrict d1) {
    hmxi_cvt_body(s0, w0, c0, d0);
    hmxi_cvt_body(s1, w1, c1, d1);
}

// tiles b0 .. b1-1 of one column tile, two at a time
static inline void hmxi_cvt_run(const uint8_t * src, int b0, int b1, uint8_t * wt, HVX_Vector * cv, HVX_Vector * dv) {
    int b = b0;
    for (; b + 2 <= b1; b += 2)
        hmxi_cvt_tile2(src + (size_t) b * 576, src + (size_t) (b + 1) * 576,
                       (HVX_Vector *) (wt + (size_t) b * 1024), cv + b, dv + b,
                       (HVX_Vector *) (wt + (size_t) (b + 1) * 1024), cv + b + 1, dv + b + 1);
    if (b < b1) hmxi_cvt_tile(src + (size_t) b * 576, (HVX_Vector *) (wt + (size_t) b * 1024), cv + b, dv + b);
}

// a whole column tile (B blocks) plus mid = 128 * sum_b d_b = 0.5 * sum_b dv_b.
// Converts two tiles at a time first, then sums dv in the original b order: the qf32 additions happen in the same
// order as before, so mid is bit-identical.
static void __attribute__((noinline)) hmxi_cvt_coltile(const uint8_t * src, int B, uint8_t * wt, HVX_Vector * cv, HVX_Vector * dv, HVX_Vector * mid) {
    hmxi_cvt_run(src, 0, B, wt, cv, dv);
    HVX_Vector acc = Q6_V_vzero();
    for (int b = 0; b < B; b++) acc = Q6_Vqf32_vadd_Vqf32Vsf(acc, dv[b]);
    *mid = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(acc), Q6_V_vsplat_R(0x3f000000)));
}

// ------------------------------------------------------------------------------------------------
// Combine: nb block stores (2 KB apart) of one output tile, rows 2*p0 .. 2*p0+15, into 16 qf32
// accumulators; `accumulate` continues from acc_out (a previous K segment).

static void __attribute__((noinline)) hmxi_combine(const uint8_t * st0, const HVX_Vector * cv, const HVX_Vector * dv,
                                                   HVX_Vector * acc_out, int p0, int nb, int accumulate) {
    HVX_Vector a[16];
#pragma unroll
    for (int i = 0; i < 16; i++) a[i] = accumulate ? Q6_Vqf32_vadd_VsfVsf(acc_out[2 * p0 + i], Q6_V_vzero()) : Q6_V_vzero();
    for (int b = 0; b < nb; b++) {
        const HVX_Vector * st   = (const HVX_Vector *) (st0 + (size_t) b * 2048) + p0;
        const HVX_Vector   cvec = cv[b];
        const HVX_Vector   dvec = dv[b];
#pragma unroll
        for (int q = 0; q < 8; q++) {
            HVX_VectorPair w  = Q6_Ww_vsxt_Vh(Q6_Vh_vsub_VhVh(st[q], cvec));
            HVX_Vector     f0 = Q6_Vsf_equals_Vw(Q6_V_lo_W(w));
            HVX_Vector     f1 = Q6_Vsf_equals_Vw(Q6_V_hi_W(w));
            a[2 * q]     = Q6_Vqf32_vadd_Vqf32Vqf32(a[2 * q], Q6_Vqf32_vmpy_VsfVsf(f0, dvec));
            a[2 * q + 1] = Q6_Vqf32_vadd_Vqf32Vqf32(a[2 * q + 1], Q6_Vqf32_vmpy_VsfVsf(f1, dvec));
        }
    }
#pragma unroll
    for (int i = 0; i < 16; i++) acc_out[2 * p0 + i] = Q6_Vsf_equals_Vqf32(a[i]);
}

// y = s_a * (acc + mid) for rows < rows, columns < cols, into dst (row stride in floats)
static void __attribute__((noinline)) hmxi_epilogue(const HVX_Vector * acc, HVX_Vector mid, const float * sa, float * dst,
                                                    int dst_stride, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        union { float f; int32_t i; } s = { sa[r] };
        HVX_Vector t = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(acc[r], mid));
        HVX_Vector y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(t, Q6_V_vsplat_R(s.i)));
        float * d = dst + (size_t) r * dst_stride;
        if (cols == 32 && (((uintptr_t) d) & 127) == 0) {
            *(HVX_Vector *) d = y;
        } else if (cols == 32) {
            *(HVX_UVector *) d = y;
        } else {
            union { HVX_Vector v; float f[32]; } u = { y };
            for (int c = 0; c < cols; c++) d[c] = u.f[c];
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Serial driver: the same pieces, one thread, producer and consumer interleaved. Used by the simulator
// harness; also the reference the threaded kernel must match bit for bit.

static int hmxi_gemm_serial(uint8_t * vtcm, size_t vtcm_size, float * dst, int dst_stride, int dst_cols,
                            const float * act, int act_stride, const uint8_t * weight, int m, int k, int n) {
    struct hmxi_layout L;
    if (!hmxi_plan(k, m, n, 1, vtcm_size, &L)) return -1;
    const int B = L.B;
    uint8_t *    ring = vtcm + L.off_ring;
    HVX_Vector * acc  = (HVX_Vector *) (vtcm + L.off_accw);
    uint32_t *   rec  = (uint32_t *) (vtcm + L.off_rec);
    float *      sa   = (float *) (vtcm + L.off_sa);
    uint8_t *    at   = vtcm + L.off_act;
    uint8_t *    wt   = vtcm + L.off_wt;
    HVX_Vector * cv   = (HVX_Vector *) (vtcm + L.off_cv);
    HVX_Vector * dv   = (HVX_Vector *) (vtcm + L.off_dv);
    HVX_Vector * mid  = (HVX_Vector *) (vtcm + L.off_mid);
    const int    n_ct_all = (n + 31) / 32;
    hmxi_init_record(rec);
    for (int m0 = 0; m0 < m; m0 += L.mc) {
        const int mr = m - m0 < L.mc ? m - m0 : L.mc, n_rt = (mr + 31) / 32;
        for (int p = 0; p < n_rt * 16; p++) {
            const int rt = p / 16, pr = p % 16, r0 = rt * 32 + 2 * pr;
            const float * x0 = r0 < mr ? act + (size_t) (m0 + r0) * act_stride : NULL;
            const float * x1 = r0 + 1 < mr ? act + (size_t) (m0 + r0 + 1) * act_stride : NULL;
            hmxi_quant_pair(x0, x1, B, &sa[r0], &sa[r0 + 1], at + (size_t) rt * B * 2048, pr);
        }
        for (int ct0 = 0; ct0 < n_ct_all; ct0 += L.nct) {
            const int nct = n_ct_all - ct0 < L.nct ? n_ct_all - ct0 : L.nct;
            for (int c = 0; c < nct; c++)
                hmxi_cvt_coltile(weight + (size_t) (ct0 + c) * B * 576, B, wt + (size_t) c * B * 1024, cv + c * B, dv + c * B, mid + c);
            for (int c = 0; c < nct; c++) for (int rt = 0; rt < n_rt; rt++) {
                for (int sg = 0; sg < L.nseg; sg++) {
                    const int b0 = sg * L.seg, nb = B - b0 < L.seg ? B - b0 : L.seg;
                    hmxi_blocks(at + ((size_t) rt * B + b0) * 2048, wt + ((size_t) c * B + b0) * 1024, ring, rec, nb);
                    hmxi_combine(ring, cv + c * B + b0, dv + c * B + b0, acc, 0, nb, sg > 0);
                    hmxi_combine(ring, cv + c * B + b0, dv + c * B + b0, acc, 8, nb, sg > 0);
                }
                const int rows = mr - rt * 32 < 32 ? mr - rt * 32 : 32;
                const int col0 = (ct0 + c) * 32, cols = dst_cols - col0 < 32 ? dst_cols - col0 : 32;
                hmxi_epilogue(acc, mid[c], sa + rt * 32, dst + (size_t) (m0 + rt * 32) * dst_stride + col0, dst_stride, rows, cols);
            }
        }
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------
// Production self test: one random 32x32x32 integer matmul through the exact store ABI (scale 2.0,
// u = floor(sum/256) mod 2^16), outputs poisoned first. Needs the HMX lock held by the caller.

static int hmxi_selftest(uint8_t * vtcm) {
    uint8_t *  a   = vtcm;            // 2 KB
    int8_t *   w   = (int8_t *) (vtcm + 2048);
    uint16_t * o   = (uint16_t *) (vtcm + 4096);
    uint32_t * rec = (uint32_t *) (vtcm + 6144);
    static uint16_t A[32][32];
    static int8_t   W[32][32];
    uint32_t s = 12345u;
    for (int mm = 0; mm < 32; mm++) for (int kk = 0; kk < 32; kk++) {
        s = s * 1103515245u + 12345u;
        A[mm][kk] = (uint16_t) ((mm == 0) ? 65535u : (mm == 1) ? 0u : (s >> 8) & 0xffff);
    }
    for (int kk = 0; kk < 32; kk++) for (int c = 0; c < 32; c++) { s = s * 1103515245u + 12345u; W[kk][c] = (int8_t) (((s >> 12) & 15) - 8); }
    memset(a, 0, 2048);
    for (int mm = 0; mm < 32; mm++) for (int kk = 0; kk < 32; kk++) {
        uint8_t * p = a + 128 * (mm >> 1) + 4 * kk + 2 * (mm & 1);
        p[0] = (uint8_t) (A[mm][kk] & 0xff); p[1] = (uint8_t) (A[mm][kk] >> 8);
    }
    for (int kk = 0; kk < 32; kk++) for (int c = 0; c < 32; c++) w[128 * (kk >> 2) + 4 * c + (kk & 3)] = W[kk][c];
    for (int i = 0; i < 1024; i++) o[i] = 0xa5a5;
    hmxi_init_record(rec);
    hmxi_blocks(a, (const uint8_t *) w, (uint8_t *) o, rec, 1);
    int bad = 0;
    for (int mm = 0; mm < 32; mm++) for (int c = 0; c < 32; c++) {
        long long sum = 0;
        for (int kk = 0; kk < 32; kk++) sum += (long long) A[mm][kk] * W[kk][c];
        long long q = sum / 256; if ((sum % 256) && sum < 0) q--;
        if (o[64 * (mm >> 1) + 2 * c + (mm & 1)] != (uint16_t) (q & 0xffff)) bad++;
    }
    return bad;
}

#endif /* HMX_INT_H */
