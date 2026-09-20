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

#include "hvx-base.h"   // hvx_q4_0_tile_to_legacy(): q4_0 tiles are stored in vrmpy order

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

static inline float hmxi_absmax_row(const float * x, int K) {
    HVX_Vector m = Q6_V_vzero();
    const HVX_Vector mask = Q6_V_vsplat_R(0x7fffffff);
    for (int i = 0; i < K; i += 32) m = Q6_Vw_vmax_VwVw(m, Q6_V_vand_VV(*(const HVX_UVector *) (x + i), mask));
    union { HVX_Vector v; int32_t w[32]; } u = { m };
    int32_t best = 0;
    for (int i = 0; i < 32; i++) if (u.w[i] > best) best = u.w[i];
    union { int32_t i; float f; } r = { best };
    return r.f;
}

static inline HVX_Vector hmxi_quant_block(const float * x, HVX_Vector vinv) {
    if (!x) return Q6_V_vsplat_R(32768);
    HVX_Vector q = Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *) x, vinv)));
    q = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(q, Q6_V_vsplat_R(-32767)), Q6_V_vsplat_R(32767));
    return Q6_Vw_vadd_VwVw(q, Q6_V_vsplat_R(32768));
}

// one row pair (pr = 0..15 inside row tile at act_rt), all B blocks
static void __attribute__((noinline)) hmxi_quant_pair(const float * x0, const float * x1, int B, float * sa0, float * sa1,
                                                      uint8_t * act_rt, int pr) {
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
// Weights: one host-repacked Q4_0 tile (576 B; since 2026-09 the 512 quant bytes are stored in vrmpy order,
// so hvx_q4_0_tile_to_legacy() restores the old byte[cp*32+n] = q[n][2cp] | q[n][2cp+1] << 4 order first), then
// 32 fp16 d) -> int8 HMX tile (q - 8), cv (word c = C | C<<16, C = 128*sum w mod 2^16), dv (256*d).
// Every other tile is only 64-byte aligned in DDR: unaligned loads; the scale load starts at byte 448
// so nothing past the tile is read.

static void __attribute__((noinline)) hmxi_cvt_tile(const uint8_t * src, HVX_Vector * wt, HVX_Vector * cv, HVX_Vector * dv) {
    const HVX_Vector k0f  = Q6_V_vsplat_R(0x0f0f0f0f);
    const HVX_Vector k08  = Q6_V_vsplat_R(0x08080808);
    HVX_Vector       sumq = Q6_V_vzero();
#pragma unroll
    for (int v = 0; v < 4; v++) {
        HVX_Vector     x  = hvx_q4_0_tile_to_legacy(*(const HVX_UVector *) (src + v * 128));
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

// a whole column tile (B blocks) plus mid = 128 * sum_b d_b = 0.5 * sum_b dv_b
static void __attribute__((noinline)) hmxi_cvt_coltile(const uint8_t * src, int B, uint8_t * wt, HVX_Vector * cv, HVX_Vector * dv, HVX_Vector * mid) {
    HVX_Vector acc = Q6_V_vzero();
    for (int b = 0; b < B; b++) {
        hmxi_cvt_tile(src + (size_t) b * 576, (HVX_Vector *) (wt + (size_t) b * 1024), cv + b, dv + b);
        acc = Q6_Vqf32_vadd_Vqf32Vsf(acc, dv[b]);
    }
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
