// Codex step 2: the resident HVX combine for the Q4_0 integer-HMX GEMM, correctness and speed - v3.
//
// One HMX block-tile store (32 rows x 32 columns, int16 in the :2x1 layout) is one 2 KB span =
// 16 vectors; vector p holds row pair (2p, 2p+1) interleaved per column: halfword 2c+(m&1).
// Per vector, with C = 128*sum(w) mod 2^16 prepacked as C0,C0,C1,C1,... :
//     h   = u - C                               (= QT, int16, exact mod 2^16)
//     widen h to 32 bits: even halfwords (row 2p) / odd halfwords (row 2p+1), 32 columns each
//     f   = (float) QT                          (IEEE int -> sf convert, exact for |QT| < 2^24)
//     acc_row += f * D_b                        (D_b[c] = 256 * d_b[c], one vector per block)
// Final: y = s_a[m] * (acc_row + 128 * sum_b d_b[c]) - the hoisted midpoint.
//
// v2 measured (simulator) that the magic-number conversion  bitcast(0x4b000000|z) - 8421376.0  in
// qf32 is NOT exact: 16,383 of the 65,536 z values came out wrong, and the combine drifted up to
// 0.14% of sum|terms| from the double sum of the recovered QT products (Codex q19 predicted this:
// qf32 rounds differently from IEEE). It is kept only as path P0 of the exhaustive test.
// v3 widens in the integer domain and converts with vconv (sf <- w):
//   P1  vzxt(h ^ 0x8000) - 32768 (vsub.w), then vconv - layout already proven (vzxt even/odd)
//   P2  vsxt(h) then vconv, when the toolchain defines Q6_Ww_vsxt_Vh - one op fewer
// X  checks P0/P1/P2 over all 65,536 z; the combine forms use P1 (mem) and P1/P2 (reg).
// Checks: loose (quantum bound vs int64 reference) and tight (vs the double sum of the recovered QT
// products); B poisons the store span first, then HMX store + HVX combine per block, bit-compared.
//
// Q4G_PRINT(fmt, ...) must be defined by the includer.

#ifndef HMXCOMB_H
#define HMXCOMB_H

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
#include "hmxq4gemm.h"

#define HC_K  1536
#define HC_NB (HC_K / 32)

#define HC_CONV_P1(h, f0, f1) do {                                                               \
        HVX_VectorPair w_ = Q6_Wuw_vzxt_Vuh(Q6_V_vxor_VV((h), Q6_V_vsplat_R(0x80008000)));       \
        (f0) = Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(Q6_V_lo_W(w_), Q6_V_vsplat_R(32768)));           \
        (f1) = Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(Q6_V_hi_W(w_), Q6_V_vsplat_R(32768)));           \
    } while (0)

#if defined(Q6_Ww_vsxt_Vh)
#define HC_HAVE_P2 1
#define HC_CONV_P2(h, f0, f1) do {                                                               \
        HVX_VectorPair w_ = Q6_Ww_vsxt_Vh(h);                                                    \
        (f0) = Q6_Vsf_equals_Vw(Q6_V_lo_W(w_));                                                  \
        (f1) = Q6_Vsf_equals_Vw(Q6_V_hi_W(w_));                                                  \
    } while (0)
#else
#define HC_HAVE_P2 0
#define HC_CONV_P2 HC_CONV_P1
#endif

// memory-resident form: one block store (16 vectors) into 32 row accumulators (sf, in VTCM)
static inline void hc_combine(const HVX_Vector * st, const HVX_Vector cvec, const HVX_Vector dvec, HVX_Vector * acc) {
#pragma unroll 4
    for (int p = 0; p < 16; p++) {
        HVX_Vector h = Q6_Vh_vsub_VhVh(st[p], cvec), f0, f1;
        HC_CONV_P1(h, f0, f1);
        acc[2 * p]     = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(f0, dvec), acc[2 * p]));
        acc[2 * p + 1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(f1, dvec), acc[2 * p + 1]));
    }
}

__attribute__((noinline)) static void hc_combine_n(const unsigned char * st, const HVX_Vector * cv, const HVX_Vector * dv,
                                                   HVX_Vector * acc, int nb) {
    for (int b = 0; b < nb; b++) hc_combine((const HVX_Vector *) (st + b * 2048), cv[b], dv[b], acc);
}

// register-resident form: one pass owns 8 row pairs (16 rows), 16 qf32 accumulators in registers
// across all nb blocks. Two passes (p0 = 0, 8) = the split two workers would use.
#define HC_DEF_REG(NAME, CONV)                                                                                   \
    __attribute__((noinline)) static void NAME(const unsigned char * st0, const HVX_Vector * cv,                  \
                                               const HVX_Vector * dv, HVX_Vector * acc_out, int p0, int nb) {     \
        HVX_Vector a[16];                                                                                        \
        _Pragma("unroll") for (int i = 0; i < 16; i++) a[i] = Q6_V_vzero();                                      \
        for (int b = 0; b < nb; b++) {                                                                           \
            const HVX_Vector * st   = (const HVX_Vector *) (st0 + b * 2048) + p0;                                \
            const HVX_Vector   cvec = cv[b];                                                                     \
            const HVX_Vector   dvec = dv[b];                                                                     \
            _Pragma("unroll") for (int q = 0; q < 8; q++) {                                                      \
                HVX_Vector h = Q6_Vh_vsub_VhVh(st[q], cvec), f0, f1;                                             \
                CONV(h, f0, f1);                                                                                 \
                a[2 * q]     = Q6_Vqf32_vadd_Vqf32Vqf32(a[2 * q], Q6_Vqf32_vmpy_VsfVsf(f0, dvec));               \
                a[2 * q + 1] = Q6_Vqf32_vadd_Vqf32Vqf32(a[2 * q + 1], Q6_Vqf32_vmpy_VsfVsf(f1, dvec));           \
            }                                                                                                    \
        }                                                                                                        \
        _Pragma("unroll") for (int i = 0; i < 16; i++) acc_out[2 * p0 + i] = Q6_Vsf_equals_Vqf32(a[i]);          \
    }
HC_DEF_REG(hc_combine_reg1, HC_CONV_P1)
HC_DEF_REG(hc_combine_reg2, HC_CONV_P2)

// HMX mac + store for nb blocks
__attribute__((noinline)) static void hc_hmx_n(unsigned char * act, signed char * wgt, unsigned char * sts, unsigned * rec, int nb) {
    for (int b = 0; b < nb; b++) {
        q4g_bias(rec); q4g_clr(); q4g_mac(act + b * 2048, wgt + b * 1024); q4g_store((unsigned short *) (sts + b * 2048));
    }
}

// HMX mac + store of block b, then the memory-form combine of that same block - one thread
__attribute__((noinline)) static void hc_both_n(unsigned char * act, signed char * wgt, unsigned char * sts, unsigned * rec,
                                                const HVX_Vector * cv, const HVX_Vector * dv, HVX_Vector * acc, int nb) {
    for (int b = 0; b < nb; b++) {
        q4g_bias(rec); q4g_clr(); q4g_mac(act + b * 2048, wgt + b * 1024); q4g_store((unsigned short *) (sts + b * 2048));
        hc_combine((const HVX_Vector *) (sts + b * 2048), cv[b], dv[b], acc);
    }
}

// X: one vector of 64 z values (QT = z - 32768) through P0, P1, P2; out[0..1], [2..3], [4..5]
__attribute__((noinline)) static void hc_conv_vec(HVX_Vector z, HVX_Vector * out) {
    const HVX_Vector kexp = Q6_V_vsplat_R(0x4b000000);
    const HVX_Vector kmag = Q6_V_vsplat_R(0x4b008000);
    HVX_VectorPair   w    = Q6_Wuw_vzxt_Vuh(z);
    out[0] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(Q6_V_vor_VV(Q6_V_lo_W(w), kexp), kmag));
    out[1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(Q6_V_vor_VV(Q6_V_hi_W(w), kexp), kmag));
    HVX_Vector h = Q6_V_vxor_VV(z, Q6_V_vsplat_R(0x80008000)), f0, f1;
    HC_CONV_P1(h, f0, f1); out[2] = f0; out[3] = f1;
    HC_CONV_P2(h, f0, f1); out[4] = f0; out[5] = f1;
}

__attribute__((noinline, optnone)) static void hc_conv_exhaustive(unsigned char * scratch) {
    HVX_Vector *     zin = (HVX_Vector *) scratch;
    HVX_Vector *     out = (HVX_Vector *) (scratch + 128);
    unsigned short * zh  = (unsigned short *) zin;
    int bad[3] = { 0, 0, 0 }, badv[3] = { 0, 0, 0 }, first[3] = { -1, -1, -1 };
    float maxd[3] = { 0, 0, 0 };
    for (int base = 0; base < 65536; base += 64) {
        for (int i = 0; i < 64; i++) zh[i] = (unsigned short) (base + i);
        hc_conv_vec(zin[0], out);
        const float * f = (const float *) out;
        for (int i = 0; i < 64; i++) {
            const int   c = i >> 1, hi = i & 1, z = base + i;
            const float want = (float) (z - 32768);
            for (int p = 0; p < 3; p++) {
                const float got = f[p * 64 + hi * 32 + c];
                if (got != want) {
                    bad[p]++; if (z >= 1) badv[p]++;
                    if (first[p] < 0) first[p] = z;
                    float dd = got - want; if (dd < 0) dd = -dd;
                    if (dd > maxd[p]) maxd[p] = dd;
                }
            }
        }
    }
    Q4G_PRINT("COMB X all 65536 z: P0 magic qf32 %d bad (%d in the clamped range, first z=%d, max |err| %g) | "
              "P1 vzxt-vsub.w-vconv %d bad (first z=%d) | P2 vsxt-vconv%s %d bad (first z=%d)",
              bad[0], badv[0], first[0], maxd[0], bad[1], first[1], HC_HAVE_P2 ? "" : " (unavailable, = P1)", bad[2], first[2]);
}

// loose (quantum bound vs int64 reference) and tight (vs double sum of the recovered QT products) checks
__attribute__((noinline, optnone)) static void hc_check(const HVX_Vector * acc, const char * tag, const unsigned char * sts,
                                                        const unsigned short * c16all, float * sa, float * dsum,
                                                        q4g_block (*W)[HC_NB], int (*qa)[HC_K]) {
    int within = 0, tight = 0; double maxe = 0, maxrel = 0;
    for (int m = 0; m < 32; m++) for (int n = 0; n < 32; n++) {
        const float * ar = (const float *) &acc[m];
        double y = (double) sa[m] * ((double) ar[n] + 128.0 * (double) dsum[n]);
        double rq = 0, bnd = 0, tsum = 0, tabs = 0;
        for (int b = 0; b < HC_NB; b++) {
            float d = q4g_h2f(W[n][b].d);
            long long sq = 0;
            for (int k = 0; k < 32; k++) sq += (long long) qa[m][b * 32 + k] * (q4g_nib(&W[n][b], k) - 8);
            rq += (double) d * (double) sq;
            bnd += 128.0 * (double) (d < 0 ? -d : d);
            const unsigned short u  = ((const unsigned short *) (sts + b * 2048))[q4g_io(m, n)];
            const short          qt = (short) (unsigned short) ((u - c16all[b * 64 + 2 * n]) & 0xffff);
            const double         t  = 256.0 * (double) d * (double) qt;
            tsum += t; tabs += t < 0 ? -t : t;
        }
        rq *= sa[m]; bnd *= sa[m];
        double e = y - rq; if (e < 0) e = -e;
        if (e > maxe) maxe = e;
        if (e <= bnd * 1.0001 + 1e-5 * (rq < 0 ? -rq : rq) + 1e-6) within++;
        double r = (double) ar[n] - tsum; if (r < 0) r = -r;
        r = tabs > 0 ? r / tabs : r;
        if (r > maxrel) maxrel = r;
        if (r <= 4e-6) tight++;
    }
    Q4G_PRINT("COMB check %s: loose %d/1024 (max err %.3g) | tight vs double sum of QT products %d/1024 within 4e-6 of sum|terms| (max %.2g)",
              tag, within, maxe, tight, maxrel);
}

__attribute__((noinline, optnone)) static void hmxcomb_run(unsigned char * base) {
    enum { K = HC_K, NB = HC_NB };
    unsigned char *  act  = base;                                  // NB x 2048 = 96 KB
    signed char *    wgt  = (signed char *) (base + 98304);        // NB x 1024 = 48 KB
    unsigned char *  sts  = base + 147456;                         // NB stores x 2048 = 96 KB
    unsigned *       rec  = (unsigned *) (base + 245760);
    HVX_Vector *     cv   = (HVX_Vector *) (base + 246784);        // NB x 128
    HVX_Vector *     dv   = (HVX_Vector *) (base + 246784 + NB * 128);
    HVX_Vector *     acc  = (HVX_Vector *) (base + 262144);        // 32 x 128 = 4 KB
    HVX_Vector *     accC = (HVX_Vector *) (base + 266240);        // copy of C's result, 4 KB
    unsigned char *  scr  = base + 270336;                         // 1 KB scratch for X
    static float     A[32][K];
    static q4g_block W[32][NB];
    static int       qa[32][K];
    static float     sa[32], dsum[32];
    const double     mhz = 1497.6;

    hc_conv_exhaustive(scr);

    for (int i = 0; i < 64; i++) rec[i] = i < 32 ? 0x00004000u : 0u;
    q4g_lcg = 4242u;
    for (int m = 0; m < 32; m++) for (int k = 0; k < K; k++) {
        float v = q4g_frand(-1.0f, 1.0f);
        if (((m * 131 + k * 7) % 97) == 0) v *= 8.0f;
        A[m][k] = v;
    }
    for (int n = 0; n < 32; n++) { dsum[n] = 0; for (int b = 0; b < NB; b++) {
        float d = q4g_frand(0.005f, 0.05f) * ((n + b) % 3 == 0 ? -1.0f : 1.0f);
        W[n][b].d = q4g_f2h(d);
        dsum[n] += q4g_h2f(W[n][b].d);
        for (int j = 0; j < 16; j++) {
            int lo = (int) q4g_frand(0.0f, 15.999f), hi = (int) q4g_frand(0.0f, 15.999f);
            W[n][b].qs[j] = (unsigned char) (lo | (hi << 4));
        }
    } }
    memset(act, 0, NB * 2048);
    for (int m = 0; m < 32; m++) {
        float mx = 0.0f;
        for (int k = 0; k < K; k++) { float t = A[m][k] < 0 ? -A[m][k] : A[m][k]; if (t > mx) mx = t; }
        sa[m] = mx > 0 ? mx / 32767.0f : 1.0f;
        for (int k = 0; k < K; k++) {
            float t = A[m][k] / sa[m];
            int q = (int) (t >= 0 ? t + 0.5f : t - 0.5f);
            if (q > 32767) q = 32767;
            if (q < -32767) q = -32767;
            qa[m][k] = q;
            unsigned u = (unsigned) (q + 32768);
            unsigned char * p = act + (k >> 5) * 2048 + q4g_ia(m, k & 31);
            p[0] = (unsigned char) (u & 0xff); p[1] = (unsigned char) (u >> 8);
        }
    }
    for (int b = 0; b < NB; b++) {
        signed char * t = wgt + b * 1024;
        unsigned short * c16 = (unsigned short *) &cv[b];
        float * df = (float *) &dv[b];
        for (int c = 0; c < 32; c++) {
            int sw = 0;
            for (int k = 0; k < 32; k++) { int w = q4g_nib(&W[c][b], k) - 8; t[q4g_iw(k, c)] = (signed char) w; sw += w; }
            c16[2 * c] = c16[2 * c + 1] = (unsigned short) ((128 * sw) & 0xffff);
            df[c] = 256.0f * q4g_h2f(W[c][b].d);
        }
    }
    const unsigned short * c16all = (const unsigned short *) cv;

    unsigned long long t0, t1, tH = ~0ull, tC = ~0ull, tR1 = ~0ull, tR2 = ~0ull, tB = ~0ull;
    // H: HMX mac + store, all NB blocks, best of 3
    for (int r = 0; r < 3; r++) {
        t0 = qurt_get_core_pcycles(); hc_hmx_n(act, wgt, sts, rec, NB); t1 = qurt_get_core_pcycles();
        if (t1 - t0 < tH) tH = t1 - t0;
    }
    // C: memory form alone over the resident stores, best of 3
    for (int r = 0; r < 3; r++) {
        memset(acc, 0, 4096);
        t0 = qurt_get_core_pcycles(); hc_combine_n(sts, cv, dv, acc, NB); t1 = qurt_get_core_pcycles();
        if (t1 - t0 < tC) tC = t1 - t0;
    }
    hc_check(acc, "mem P1", sts, c16all, sa, dsum, W, qa);
    memcpy(accC, acc, 4096);

    // R1 / R2: register form, two 16-row passes, best of 3 each
    for (int r = 0; r < 3; r++) {
        memset(acc, 0, 4096);
        t0 = qurt_get_core_pcycles();
        hc_combine_reg1(sts, cv, dv, acc, 0, NB);
        hc_combine_reg1(sts, cv, dv, acc, 8, NB);
        t1 = qurt_get_core_pcycles();
        if (t1 - t0 < tR1) tR1 = t1 - t0;
    }
    hc_check(acc, "reg P1", sts, c16all, sa, dsum, W, qa);
    for (int r = 0; r < 3; r++) {
        memset(acc, 0, 4096);
        t0 = qurt_get_core_pcycles();
        hc_combine_reg2(sts, cv, dv, acc, 0, NB);
        hc_combine_reg2(sts, cv, dv, acc, 8, NB);
        t1 = qurt_get_core_pcycles();
        if (t1 - t0 < tR2) tR2 = t1 - t0;
    }
    hc_check(acc, HC_HAVE_P2 ? "reg P2" : "reg P2(=P1)", sts, c16all, sa, dsum, W, qa);

    // B: poison the store span, then HMX store + combine per block; must equal C bit for bit
    int bsame = 0;
    for (int r = 0; r < 3; r++) {
        memset(sts, 0xa5, NB * 2048);
        memset(acc, 0, 4096);
        t0 = qurt_get_core_pcycles(); hc_both_n(act, wgt, sts, rec, cv, dv, acc, NB); t1 = qurt_get_core_pcycles();
        if (t1 - t0 < tB) tB = t1 - t0;
        if (r == 0) { const unsigned * x = (const unsigned *) acc, * y = (const unsigned *) accC; for (int i = 0; i < 1024; i++) bsame += x[i] == y[i]; }
    }
    Q4G_PRINT("COMB check B (poisoned stores, HMX store then HVX load of the same block): %d/1024 bit-identical to C", bsame);

    const unsigned long long tR = tR2 < tR1 ? tR2 : tR1;
    Q4G_PRINT("COMB K=%d NB=%d: HMX mac+store %.1f cyc/block | combine mem %.1f, reg P1 %.1f, reg P2 %.1f cyc/block | HMX+mem serial %.1f cyc/block",
              K, NB, (double) tH / NB, (double) tC / NB, (double) tR1 / NB, (double) tR2 / NB, (double) tB / NB);
    Q4G_PRINT("COMB resident kernel throughput, one thread, 32x32xK tile, no packing/DMA/epilogue: HMX+mem serial %.1f GOP/s, "
              "HMX+best reg (sum of separate timings) %.1f GOP/s  (at %.1f MHz)",
              2.0 * 32 * 32 * K / ((double) tB / (mhz * 1e6)) / 1e9,
              2.0 * 32 * 32 * K / ((double) (tH + tR) / (mhz * 1e6)) / 1e9, mhz);
}

#endif
