// Put the Q4_0 block scale into HMX's own per-column converter scale, so the HVX combine becomes an
// int16 -> int32 add.
//
// The accumulator holds 256*Q_b (the raw sum with its low 8 bits gone), Q_b = floor(S_u/256),
// S_u = sum (q+32768)*w over one 32-K block, and the store converts out = floor(256*Q * scale_c / 512)
// = floor(Q * scale_c / 2) with scale_c the fp16 in record word c (scale 512 returns the raw sum,
// scale 2.0 returns Q - both measured). v1 of this file used Q * scale / 512 and every output wrapped.
// Q4_0's d_b is itself fp16, so scale_c = 2 * d_b[c] * 2^s_c is exact, and
//     out_b = floor(Q_b * d_b * 2^s_c)                       (int16; must not wrap)
// With |Q_b| <= 65535 no wrap needs |d_b * 2^s_c| < 0.5: s_c = floor(log2(0.49 / max_b |d_b[c]|)).
// Then, from sum q*w = 256*Q_b + r_b - 32768*sum(w), r in [0,256), and Q_b*d_b = 2^-s (out_b + e_b):
//     y[m][c] = s_a[m] * (256 * 2^-s_c * sum_b out_b + K_c)
//     K_c     = N_nz * 128 * 2^-s_c + sum_b d_b * (128 - 32768 * sum(w_b))  (both midpoints hoisted)
// N_nz = blocks with d_b != 0 (a zero block stores exactly 0, so it gets no converter midpoint - Codex q20)
// error <= s_a * sum_{b: d_b != 0} (128 * 2^-s_c + 128 |d_b|).
//
// What this measures (same data, K=1536, 32 rows x 32 columns, 48 blocks):
//   E  every stored element against floor(Q * scale / 2) in exact arithmetic, split by the sign of
//      d_b (a negative converter scale is not proven to work)
//   A  accuracy: this path vs the exact-QT path vs the f32 reference, and the bound
//   V  the HVX int16 -> int32 sum (vaddhw accumulate, two blocks per op) against the scalar sum
//   T  cycles: HMX per block with a per-block record, and the HVX sum per block
//
// Q4G_PRINT(fmt, ...) must be defined by the includer.

#ifndef HMXSCL_H
#define HMXSCL_H

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
#include "hmxq4gemm.h"

#define HS_K  1536
#define HS_NB (HS_K / 32)

// HMX mac + store per block, each block with its own conversion record (256 B apart)
__attribute__((noinline)) static void hs_hmx_n(unsigned char * act, signed char * wgt, unsigned char * sts,
                                               unsigned char * recs, int nb) {
    for (int b = 0; b < nb; b++) {
        q4g_bias((unsigned *) (recs + b * 256)); q4g_clr();
        q4g_mac(act + b * 2048, wgt + b * 1024);
        q4g_store((unsigned short *) (sts + b * 2048));
    }
}

// sum of nb int16 block stores into int32, row pair by row pair: lo = row 2p, hi = row 2p+1
__attribute__((noinline)) static void hs_sum_n(const unsigned char * sts, HVX_Vector * acc32, int nb) {
    for (int p = 0; p < 16; p++) {
        HVX_VectorPair a = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
        const HVX_Vector * s = (const HVX_Vector *) sts + p;
#if defined(Q6_Ww_vaddacc_WwVhVh)
        for (int b = 0; b + 1 < nb; b += 2) a = Q6_Ww_vaddacc_WwVhVh(a, s[b * 16], s[(b + 1) * 16]);
        if (nb & 1) a = Q6_Ww_vaddacc_WwVhVh(a, s[(nb - 1) * 16], Q6_V_vzero());
#else
        for (int b = 0; b < nb; b++) {
            HVX_VectorPair w = Q6_Ww_vsxt_Vh(s[b * 16]);
            a = Q6_W_vcombine_VV(Q6_Vw_vadd_VwVw(Q6_V_hi_W(a), Q6_V_hi_W(w)), Q6_Vw_vadd_VwVw(Q6_V_lo_W(a), Q6_V_lo_W(w)));
        }
#endif
        acc32[2 * p]     = Q6_V_lo_W(a);
        acc32[2 * p + 1] = Q6_V_hi_W(a);
    }
}

static inline long long hs_floordiv(long long a, long long b) { long long q = a / b; if ((a % b) && ((a < 0) != (b < 0))) q--; return q; }
// no libm on the simulator link: power of two and square root by hand
static inline double hs_pow2(int s) { double r = 1.0; if (s >= 0) { while (s--) r *= 2.0; } else { while (s++) r *= 0.5; } return r; }
static inline double hs_sqrt(double x) { if (x <= 0) return 0; double r = x > 1 ? x : 1; for (int i = 0; i < 60; i++) r = 0.5 * (r + x / r); return r; }

__attribute__((noinline, optnone)) static void hmxscl_run(unsigned char * base) {
    enum { K = HS_K, NB = HS_NB };
    unsigned char *  act  = base;                                  // NB x 2048 = 96 KB
    signed char *    wgt  = (signed char *) (base + 98304);        // NB x 1024 = 48 KB
    unsigned char *  sts  = base + 147456;                         // NB x 2048 = 96 KB
    unsigned char *  recs = base + 245760;                         // NB x 256  = 12 KB
    HVX_Vector *     acc  = (HVX_Vector *) (base + 262144);        // 32 x 128  = 4 KB
    static float     A[32][K];
    static q4g_block W[32][NB];
    static int       qa[32][K];
    static float     sa[32];
    static int       sc[32], swb[32][NB];
    static double    Kc[32];
    const double     mhz = 1497.6;

    q4g_lcg = 9090u;
    for (int m = 0; m < 32; m++) for (int k = 0; k < K; k++) {
        float v = q4g_frand(-1.0f, 1.0f);
        if (((m * 131 + k * 7) % 97) == 0) v *= 8.0f;
        A[m][k] = v;
    }
    for (int n = 0; n < 32; n++) for (int b = 0; b < NB; b++) {
        float d = q4g_frand(0.005f, 0.05f) * ((n + b) % 3 == 0 ? -1.0f : 1.0f);
        if ((b % 11) == 5) d = 0.0f;
        W[n][b].d = q4g_f2h(d);
        for (int j = 0; j < 16; j++) {
            int lo = (int) q4g_frand(0.0f, 15.999f), hi = (int) q4g_frand(0.0f, 15.999f);
            W[n][b].qs[j] = (unsigned char) (lo | (hi << 4));
        }
    }
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
    // per column shift s_c, per block record scale d_b * 512 * 2^s_c, hoisted constant K_c
    memset(recs, 0, NB * 256);
    for (int c = 0; c < 32; c++) {
        float mx = 0.0f;
        for (int b = 0; b < NB; b++) { float d = q4g_h2f(W[c][b].d); if (d < 0) d = -d; if (d > mx) mx = d; }
        int s = 0;
        if (mx > 0) { while (mx * (float) (1 << (s + 1)) < 0.49f && s < 20) s++; while (mx * (float) hs_pow2(s) >= 0.49f) s--; }
        sc[c] = s;
        Kc[c] = 0.0;
        for (int b = 0; b < NB; b++) {
            const float d = q4g_h2f(W[c][b].d);
            int sw = 0;
            for (int k = 0; k < 32; k++) { int w = q4g_nib(&W[c][b], k) - 8; wgt[b * 1024 + q4g_iw(k, c)] = (signed char) w; sw += w; }
            swb[c][b] = sw;
            Kc[c] += (double) d * (128.0 - 32768.0 * (double) sw) + (d != 0.0f ? 128.0 * hs_pow2(-s) : 0.0);
            ((unsigned *) (recs + b * 256))[c] = (unsigned) q4g_f2h(d * 2.0f * (float) hs_pow2(s));
        }
    }

    unsigned long long t0, t1, tH = ~0ull, tV = ~0ull;
    for (int r = 0; r < 3; r++) {
        t0 = qurt_get_core_pcycles(); hs_hmx_n(act, wgt, sts, recs, NB); t1 = qurt_get_core_pcycles();
        if (t1 - t0 < tH) tH = t1 - t0;
    }

    // E: each stored element vs floor(Q * scale / 512), exact
    int bad_pos = 0, n_pos = 0, bad_neg = 0, n_neg = 0, bad_zero = 0, n_zero = 0, off1 = 0, first_bad = -1;
    long long ref_first = 0, got_first = 0;
    int zb = -1, zc = -1, zm = -1, zgot = 0; long long zQ = 0;
    int nb_ = -1, ngot = 0; long long nQ = 0, nref = 0; unsigned nsc = 0; int nexact = 0, nexact_bad = 0;
    static int badblk[HS_NB];
    memset(badblk, 0, sizeof badblk);
    static int outsum[32][32];
    memset(outsum, 0, sizeof outsum);
    for (int b = 0; b < NB; b++) for (int c = 0; c < 32; c++) {
        const unsigned short hs = (unsigned short) ((const unsigned *) (recs + b * 256))[c];
        const double scale = (double) q4g_h2f(hs);
        for (int m = 0; m < 32; m++) {
            long long su = 0;
            for (int k = 0; k < 32; k++) su += (long long) (qa[m][b * 32 + k] + 32768) * (q4g_nib(&W[c][b], k) - 8);
            const long long Q   = hs_floordiv(su, 256);
            const double    v   = (double) Q * scale / 2.0;
            long long       ref = (long long) v; if ((double) ref > v) ref--;
            const short     got = (short) ((const unsigned short *) (sts + b * 2048))[q4g_io(m, c)];
            outsum[m][c] += got;
            const int bad = (long long) got != ref;
            if (scale < 0 && (double) ref == v) { nexact++; nexact_bad += bad; }
            if (scale > 0) { n_pos++; bad_pos += bad; } else if (scale < 0) { n_neg++; bad_neg += bad; } else { n_zero++; bad_zero += bad; }
            if (bad) {
                badblk[b]++;
                if (scale == 0 && zb < 0) { zb = b; zc = c; zm = m; zgot = got; zQ = Q; }
                if (scale < 0 && nb_ < 0) { nb_ = b; ngot = got; nQ = Q; nref = ref; nsc = hs; }
                long long dd = (long long) got - ref; if (dd == 1 || dd == -1) off1++;
                if (first_bad < 0) { first_bad = (b * 32 + c) * 32 + m; ref_first = ref; got_first = got; }
            }
        }
    }
    Q4G_PRINT("SCL E stored vs floor(Q*scale/2): positive scale %d/%d bad, negative %d/%d bad, zero %d/%d bad "
              "(off by one %d; first bad #%d ref %lld got %lld)",
              bad_pos, n_pos, bad_neg, n_neg, bad_zero, n_zero, off1, first_bad, ref_first, got_first);
    Q4G_PRINT("SCL E negative: first bad block %d Q %lld scale %04x ref %lld got %d | negative-scale products that are exact integers: %d, of them bad %d",
              nb_, nQ, nsc, nref, ngot, nexact, nexact_bad);
    Q4G_PRINT("SCL E detail: first bad zero-scale element block %d col %d row %d got %d (Q %lld, prev block %d scale %04x) | bad per block 0..7: %d %d %d %d %d %d %d %d",
              zb, zc, zm, zgot, zQ, zb - 1, zb > 0 ? (unsigned) (unsigned short) ((const unsigned *) (recs + (zb - 1) * 256))[zc] : 0u,
              badblk[0], badblk[1], badblk[2], badblk[3], badblk[4], badblk[5], badblk[6], badblk[7]);

    // V: HVX sum vs scalar sum
    for (int r = 0; r < 3; r++) {
        t0 = qurt_get_core_pcycles(); hs_sum_n(sts, acc, NB); t1 = qurt_get_core_pcycles();
        if (t1 - t0 < tV) tV = t1 - t0;
    }
    int vmatch = 0;
    for (int m = 0; m < 32; m++) for (int c = 0; c < 32; c++) vmatch += ((const int *) &acc[m])[c] == outsum[m][c];

    // A: accuracy of this path, of the exact-QT path, against f32 and against the int64 quantised reference
    int within = 0;
    double maxe = 0, se_s = 0, se_t = 0, me_s = 0, me_t = 0;
    for (int m = 0; m < 32; m++) for (int c = 0; c < 32; c++) {
        double yf = 0, rq = 0, bnd = 0, yt = 0;
        for (int b = 0; b < NB; b++) {
            const double d = (double) q4g_h2f(W[c][b].d);
            long long sq = 0, su = 0;
            for (int k = 0; k < 32; k++) {
                const int w = q4g_nib(&W[c][b], k) - 8;
                yf += (double) A[m][b * 32 + k] * d * (double) w;
                sq += (long long) qa[m][b * 32 + k] * w;
                su += (long long) (qa[m][b * 32 + k] + 32768) * w;
            }
            rq  += d * (double) sq;
            bnd += d != 0.0 ? 128.0 * hs_pow2(-sc[c]) + 128.0 * (d < 0 ? -d : d) : 0.0;
            const long long QT = hs_floordiv(su, 256) - 128LL * swb[c][b];
            yt  += 256.0 * d * (double) QT + 128.0 * d;
        }
        const double ys = (double) sa[m] * (256.0 * hs_pow2(-sc[c]) * (double) ((const int *) &acc[m])[c] + Kc[c]);
        rq *= sa[m]; bnd *= sa[m]; yt *= sa[m];
        double e = ys - rq; if (e < 0) e = -e;
        if (e > maxe) maxe = e;
        if (e <= bnd * 1.0001 + 1e-6) within++;
        double es = ys - yf, et = yt - yf;
        se_s += es * es; se_t += et * et;
        if ((es < 0 ? -es : es) > me_s) me_s = es < 0 ? -es : es;
        if ((et < 0 ? -et : et) > me_t) me_t = et < 0 ? -et : et;
    }
    Q4G_PRINT("SCL V HVX int16->int32 sum vs scalar: %d/1024 equal%s", vmatch,
#if defined(Q6_Ww_vaddacc_WwVhVh)
              " (vaddhw acc)"
#else
              " (vsxt + vadd.w fallback)"
#endif
              );
    Q4G_PRINT("SCL A K=%d vs f32 ref: scale-in-HMX max %.4g rms %.4g | exact-QT path max %.4g rms %.4g | within bound of int64 ref %d/1024 (max err %.3g) | s_c range [%d..%d]",
              K, me_s, hs_sqrt(se_s / 1024.0), me_t, hs_sqrt(se_t / 1024.0), within, maxe, sc[0], sc[31]);
    Q4G_PRINT("SCL T per block: HMX (own record) mac+store %.1f cyc, HVX int16 sum %.1f cyc -> %.1f cyc/block one thread = %.1f GOP/s resident",
              (double) tH / NB, (double) tV / NB, (double) (tH + tV) / NB,
              2.0 * 32 * 32 * K / ((double) (tH + tV) / (mhz * 1e6)) / 1e9);
}

#endif
