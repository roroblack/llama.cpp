// Codex step 1: the short QT formula, a tighter oracle, and the edge cases, on the integer HMX path.
//
// With activations clamped to [-32767, 32767] (never -32768) and Q4_0 weights, the corrected block
// value QT = floor(S_q/256) always fits int16, so
//     QT = sign_extend_16((u - 128 * sum(w)) mod 2^16)
// replaces both the lower-bound unwrap and the offset correction. Checked element by element against
// the unwrap method, on four data sets:
//   0 random                       1 all activations +32767, all weights -8  (QT = -32767)
//   2 all activations -32767, all weights -8 (QT = +32767)
//   3 random, with zero and negative block scales
// The midpoint is hoisted: y = s_a * (sum_b 256*d_b*QT + 128 * sum_b d_b), which bounds the error
// against the int64 quantised reference by 128 * s_a * sum_b |d_b| - half the earlier bound.
//
// Q4G_PRINT(fmt, ...) must be defined by the includer.

#ifndef HMXQ4QT_H
#define HMXQ4QT_H

#include "hmxq4gemm.h"

__attribute__((noinline, optnone))
static void hmxq4qt_run(unsigned char * base) {
    const int K = 256, NB = 8;
    unsigned char *  act = base;
    signed char *    wgt = (signed char *) (base + 98304);
    unsigned short * out = (unsigned short *) (base + 196608);
    unsigned *       rec = (unsigned *) (base + 200704);
    static float     A[32][256];
    static q4g_block W[64][8];
    static int       qa[32][256];
    static float     sa[32];
    static float     Yf[32][64], dsum[64];

    for (int i = 0; i < 64; i++) rec[i] = i < 32 ? 0x00004000u : 0u;

    for (int set = 0; set < 4; set++) {
        q4g_lcg = 777u + (unsigned) set;
        for (int m = 0; m < 32; m++) for (int k = 0; k < K; k++) {
            float v = set == 1 ? 1.0f : (set == 2 ? -1.0f : q4g_frand(-1.0f, 1.0f));
            if ((set == 0 || set == 3) && ((m * 131 + k * 7) % 97) == 0) v *= 8.0f;
            A[m][k] = v;
        }
        for (int n = 0; n < 64; n++) for (int b = 0; b < NB; b++) {
            float d = q4g_frand(0.005f, 0.05f) * ((n + b) % 3 == 0 ? -1.0f : 1.0f);
            if (set == 3 && (b % 4) == 0) d = 0.0f;
            W[n][b].d = q4g_f2h(d);
            for (int j = 0; j < 16; j++) {
                int lo = (set == 1 || set == 2) ? 0 : (int) q4g_frand(0.0f, 15.999f);
                int hi = (set == 1 || set == 2) ? 0 : (int) q4g_frand(0.0f, 15.999f);
                W[n][b].qs[j] = (unsigned char) (lo | (hi << 4));
            }
        }
        memset(act, 0, 16384);
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
        for (int ct = 0; ct < 2; ct++) for (int b = 0; b < NB; b++) {
            signed char * t = wgt + (ct * NB + b) * 1024;
            for (int c = 0; c < 32; c++) for (int k = 0; k < 32; k++)
                t[q4g_iw(k, c)] = (signed char) (q4g_nib(&W[ct * 32 + c][b], k) - 8);
        }

        memset(Yf, 0, sizeof Yf); memset(dsum, 0, sizeof dsum);
        int mism = 0, total = 0, qmin = 99999, qmax = -99999;
        for (int ct = 0; ct < 2; ct++) for (int b = 0; b < NB; b++) {
            q4g_bias(rec); q4g_clr();
            q4g_mac(act + b * 2048, wgt + (ct * NB + b) * 1024);
            q4g_store(out);
            for (int c = 0; c < 32; c++) {
                const int n = ct * 32 + c;
                long long neg = 0, sw = 0;
                for (int k = 0; k < 32; k++) { int w = q4g_nib(&W[n][b], k) - 8; sw += w; if (w < 0) neg += w; }
                const long long L = q4g_floordiv(65535LL * neg, 256);
                const float     d = q4g_h2f(W[n][b].d);
                dsum[n] += d;
                for (int m = 0; m < 32; m++) {
                    long long u   = (long long) out[q4g_io(m, c)];
                    long long Q   = L + (((u - L) % 65536) + 65536) % 65536;
                    long long qt1 = Q - 128 * sw;
                    short     qt2 = (short) (unsigned short) ((u - 128 * sw) & 0xffff);
                    mism += (qt1 != (long long) qt2);
                    total++;
                    if (qt2 < qmin) qmin = qt2;
                    if (qt2 > qmax) qmax = qt2;
                    Yf[m][n] += 256.0f * d * (float) qt2;
                }
            }
        }
        int within = 0;
        double maxe = 0;
        for (int m = 0; m < 32; m++) for (int n = 0; n < 64; n++) {
            double y = (double) sa[m] * ((double) Yf[m][n] + 128.0 * (double) dsum[n]);
            double rq = 0, bnd = 0;
            for (int b = 0; b < NB; b++) {
                float d = q4g_h2f(W[n][b].d);
                long long sq = 0;
                for (int k = 0; k < 32; k++) sq += (long long) qa[m][b * 32 + k] * (q4g_nib(&W[n][b], k) - 8);
                rq += (double) d * (double) sq;
                bnd += 128.0 * (double) (d < 0 ? -d : d);
            }
            rq *= sa[m];
            bnd *= sa[m];
            double e = y - rq;
            if (e < 0) e = -e;
            if (e > maxe) maxe = e;
            if (e <= bnd * 1.0001 + 1e-6) within++;
        }
        Q4G_PRINT("Q4QT set=%d (%s): short formula vs unwrap %d/%d mismatches, QT range [%d, %d], "
                  "hoisted-midpoint result within 128*s_a*sum|d| of int64 ref %d/2048 (max err %.3g)",
                  set, set == 0 ? "random" : set == 1 ? "a=+32767 w=-8" : set == 2 ? "a=-32767 w=-8" : "zero/neg d",
                  mism, total, qmin, qmax, within, maxe);
    }
}

#endif
