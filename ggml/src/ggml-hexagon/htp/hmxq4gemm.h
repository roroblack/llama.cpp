// Prototype: Q4_0 x f32 matrix multiply on the INTEGER HMX path, correctness first.
//
// Everything this rests on was measured, device == simulator:
//   tile geometry     A[m][k] u16 LE at 128*(m>>1)+4k+2(m&1); W[k][c] int8 at 128*(k>>2)+4c+(k&3);
//                     Y[m][c] at out[64*(m>>1)+2c+(m&1)]
//   accumulator       holds floor(sum(a16*w8)/256), wider than 32 bits
//   conversion        out = floor(acc_sum * scale_c / 512), scale_c = fp16 in the low 32 record words;
//                     scale 2.0 gives u = Q mod 2^16 with Q = floor(sum/256) (wraps, no saturation)
//   one-store unwrap  for a Q4_0 block every possible Q lies in [L, L+65535], so
//                     Q = L + ((u - L) mod 65536) with L = floor(65535 * sum(w<0) / 256)   (Codex)
//
// The algorithm, per output element y[m][n] = sum_k a[m][k] * w[k][n]:
//   activations  per-row symmetric int16 q_a = round(a / s_a), stored unsigned u = q_a + 32768
//   weights      Q4_0 unchanged: w = d_b * (q - 8), q - 8 in -8..7 used directly as the int8 weight
//   per block b  S_u = sum_k u*(q-8) = S_q + 32768*sum(q-8); 32768 = 128*256, so
//                floor(S_u/256) - 128*sum(q-8) = floor(S_q/256) exactly (no extra rounding)
//   combine      y += s_a[m] * d_b[n] * 256 * floor(S_q/256)   (fp32, scalar here - correctness only)
//
// Two references, so kernel bugs, HMX rounding and activation quantization stay separate (Codex):
//   R_f  f32 activations x dequantised Q4_0 weights (what the model means)
//   R_q  int64 quantised product: s_a * sum_b d_b * S_q (what an exact integer kernel would give)
// Bound from the 2^8 quantum: |Y - R_q| <= 255 * s_a * sum_b |d_b| per element.
//
// Q4G_PRINT(fmt, ...) must be defined by the includer.

#ifndef HMXQ4GEMM_H
#define HMXQ4GEMM_H

#include <string.h>

#define Q4G_M   32          // one row tile in this prototype
#define Q4G_KMAX 1536       // up to 48 Q4_0 blocks
#define Q4G_N   64          // two column tiles
#define Q4G_NBMAX (Q4G_KMAX / 32)
#define Q4G_NT  (Q4G_N / 32)

typedef struct { unsigned short d; unsigned char qs[16]; } q4g_block;   // ggml block_q4_0 layout

static inline int q4g_ia(int m, int k) { return 128 * (m >> 1) + 4 * k + 2 * (m & 1); }
static inline int q4g_iw(int k, int c) { return 128 * (k >> 2) + 4 * c + (k & 3); }
static inline int q4g_io(int m, int c) { return 64 * (m >> 1) + 2 * c + (m & 1); }

static void __attribute__((noinline)) q4g_mac(unsigned char * a, signed char * w) {
    asm volatile("{\n"
                 "  activation.ub = mxmem(%0, %2):deep\n"
                 "  weight.b      = mxmem(%1, %3)\n"
                 "}\n" :: "r"(a), "r"(w), "r"(2047), "r"(1023) : "memory");
}
static void __attribute__((noinline)) q4g_bias(unsigned * b) { asm volatile("bias = mxmem(%0)\n" :: "r"(b) : "memory"); }
static void __attribute__((noinline)) q4g_clr(void)          { asm volatile("mxclracc\n" ::: "memory"); }
static void __attribute__((noinline)) q4g_store(unsigned short * o) {
    asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n" :: "r"(o), "r"(0) : "memory");
}

static inline float q4g_h2f(unsigned short h) {
    union { unsigned short u; __fp16 f; } x; x.u = h; return (float) x.f;
}
static inline unsigned short q4g_f2h(float f) {
    union { unsigned short u; __fp16 h; } x; x.h = (__fp16) f; return x.u;
}
static inline int q4g_nib(const q4g_block * b, int j) {     // ggml order: low nibbles 0..15, high 16..31
    return j < 16 ? (b->qs[j] & 0x0f) : (b->qs[j - 16] >> 4);
}
static inline long long q4g_floordiv(long long a, long long b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

static unsigned q4g_lcg = 20260910u;
static float q4g_frand(float lo, float hi) {
    q4g_lcg = q4g_lcg * 1103515245u + 12345u;
    return lo + (hi - lo) * (float) ((q4g_lcg >> 8) & 0xffffff) / 16777216.0f;
}

// One run for a given K. Two changes from the first version, both measured against references:
//   * the store floors every block (acc = floor(S/256)), always downward, so the bias grows with
//     the number of blocks. Adding half a quantum per block (QT + 0.5) makes it zero-mean.
//   * a third reference, R_8: activations quantised per 32-block to int8 (Q8_0), which is what the
//     existing HVX kernel does - so "is this better or worse than the NPU path we have" has an answer.
// base: 2 KB aligned VTCM, at least 208 KB (K = 1536: 48 x 2048 act + 2 x 48 x 1024 wgt)
__attribute__((noinline, optnone))
static void hmxq4gemm_k(unsigned char * base, const int Q4G_K, const int unbias) {
    const int Q4G_NB = Q4G_K / 32;
    unsigned char *  act  = base;                                        // NB x 2048  (<= 96 KB)
    signed char *    wgt  = (signed char *) (base + 98304);              // NT x NB x 1024 (<= 96 KB)
    unsigned short * out  = (unsigned short *) (base + 196608);          // 2 KB
    unsigned *       rec  = (unsigned *) (base + 200704);                // 256 B

    static float     A[Q4G_M][Q4G_KMAX];
    static q4g_block W[Q4G_N][Q4G_NBMAX];                           // ggml: row n of the weight, blocks along K
    static float     Y[Q4G_M][Q4G_N], RF[Q4G_M][Q4G_N], RQ[Q4G_M][Q4G_N], BND[Q4G_M][Q4G_N], R8[Q4G_M][Q4G_N];
    static int       qa[Q4G_M][Q4G_KMAX];
    static float     sa[Q4G_M];

    // data: activations in [-1, 1] with a few outliers; block scales of both signs
    for (int m = 0; m < Q4G_M; m++) for (int k = 0; k < Q4G_K; k++) {
        float v = q4g_frand(-1.0f, 1.0f);
        if (((m * 131 + k * 7) % 97) == 0) v *= 8.0f;
        A[m][k] = v;
    }
    for (int n = 0; n < Q4G_N; n++) for (int b = 0; b < Q4G_NB; b++) {
        float d = q4g_frand(0.005f, 0.05f) * ((n + b) % 3 == 0 ? -1.0f : 1.0f);
        W[n][b].d = q4g_f2h(d);
        for (int j = 0; j < 16; j++) {
            int lo = (int) q4g_frand(0.0f, 15.999f), hi = (int) q4g_frand(0.0f, 15.999f);
            W[n][b].qs[j] = (unsigned char) (lo | (hi << 4));
        }
    }

    // activation quantisation: per-row symmetric int16, stored unsigned with +32768
    q4g_lcg = 20260910u;
    memset(act, 0, 98304);
    for (int m = 0; m < Q4G_M; m++) {
        float mx = 0.0f;
        for (int k = 0; k < Q4G_K; k++) { float t = A[m][k] < 0 ? -A[m][k] : A[m][k]; if (t > mx) mx = t; }
        sa[m] = mx > 0 ? mx / 32767.0f : 1.0f;
        for (int k = 0; k < Q4G_K; k++) {
            float t = A[m][k] / sa[m];
            int q = (int) (t >= 0 ? t + 0.5f : t - 0.5f);
            if (q > 32767) q = 32767; if (q < -32767) q = -32767;
            qa[m][k] = q;
            unsigned u = (unsigned) (q + 32768);
            unsigned char * p = act + (k >> 5) * 2048 + q4g_ia(m, k & 31);
            p[0] = (unsigned char) (u & 0xff); p[1] = (unsigned char) (u >> 8);
        }
    }

    // weight repack: tile (ct, b) holds q-8 for columns ct*32.. and the block's 32 K values
    for (int ct = 0; ct < Q4G_NT; ct++) for (int b = 0; b < Q4G_NB; b++) {
        signed char * t = wgt + (ct * Q4G_NB + b) * 1024;
        for (int c = 0; c < 32; c++) for (int k = 0; k < 32; k++)
            t[q4g_iw(k, c)] = (signed char) (q4g_nib(&W[ct * 32 + c][b], k) - 8);
    }

    // references
    for (int m = 0; m < Q4G_M; m++) for (int n = 0; n < Q4G_N; n++) {
        double rf = 0.0, rq = 0.0, bnd = 0.0;
        for (int b = 0; b < Q4G_NB; b++) {
            float d = q4g_h2f(W[n][b].d);
            long long sq = 0;
            for (int k = 0; k < 32; k++) {
                int w = q4g_nib(&W[n][b], k) - 8;
                rf += (double) A[m][b * 32 + k] * (double) d * (double) w;
                sq += (long long) qa[m][b * 32 + k] * w;
            }
            rq  += (double) d * (double) sq;
            bnd += 255.0 * (double) (d < 0 ? -d : d);
        }
        // R_8: Q8_0 activations (per 32-block int8, scale = max/127) - the existing HVX kernel's scheme
        double r8 = 0.0;
        for (int b = 0; b < Q4G_NB; b++) {
            float amx = 0.0f;
            for (int k = 0; k < 32; k++) { float t = A[m][b * 32 + k]; if (t < 0) t = -t; if (t > amx) amx = t; }
            float da = amx > 0 ? amx / 127.0f : 1.0f;
            float d  = q4g_h2f(W[n][b].d);
            long long s8 = 0;
            for (int k = 0; k < 32; k++) {
                float t = A[m][b * 32 + k] / da;
                int q8 = (int) (t >= 0 ? t + 0.5f : t - 0.5f);
                s8 += (long long) q8 * (q4g_nib(&W[n][b], k) - 8);
            }
            r8 += (double) da * (double) d * (double) s8;
        }
        R8[m][n]  = (float) r8;
        RF[m][n]  = (float) rf;
        RQ[m][n]  = (float) (rq * sa[m]);
        BND[m][n] = (float) (bnd * sa[m]);
    }

    // the HMX kernel: one store per block-tile at scale 2.0, unwrap, correct, combine
    for (int i = 0; i < 64; i++) rec[i] = i < 32 ? 0x00004000u : 0u;
    memset(Y, 0, sizeof Y);
    for (int ct = 0; ct < Q4G_NT; ct++) {
        for (int b = 0; b < Q4G_NB; b++) {
            q4g_bias(rec); q4g_clr();
            q4g_mac(act + b * 2048, wgt + (ct * Q4G_NB + b) * 1024);
            q4g_store(out);
            for (int c = 0; c < 32; c++) {
                const int n = ct * 32 + c;
                long long neg = 0, sw = 0;
                for (int k = 0; k < 32; k++) { int w = q4g_nib(&W[n][b], k) - 8; sw += w; if (w < 0) neg += w; }
                const long long L = q4g_floordiv(65535LL * neg, 256);
                const float     d = q4g_h2f(W[n][b].d);
                for (int m = 0; m < 32; m++) {
                    long long u  = (long long) out[q4g_io(m, c)];
                    long long Q  = L + (((u - L) % 65536) + 65536) % 65536;
                    long long QT = Q - 128 * sw;                    // = floor(S_q / 256)
                    Y[m][n] += d * 256.0f * ((float) QT + (unbias ? 0.5f : 0.0f));
                }
            }
        }
    }
    for (int m = 0; m < Q4G_M; m++) for (int n = 0; n < Q4G_N; n++) Y[m][n] *= sa[m];

    // report
    double max_rf = 0, err_f = 0, err_q = 0, err_8 = 0, rms_f = 0, rms_8 = 0; int within = 0;
    for (int m = 0; m < Q4G_M; m++) for (int n = 0; n < Q4G_N; n++) {
        double ef = Y[m][n] - RF[m][n];  double e8 = R8[m][n] - RF[m][n];  double eq = Y[m][n] - RQ[m][n];
        rms_f += ef * ef; rms_8 += e8 * e8;
        if (ef < 0) ef = -ef; if (e8 < 0) e8 = -e8; if (eq < 0) eq = -eq;
        double af = RF[m][n] < 0 ? -RF[m][n] : RF[m][n];
        if (af > max_rf) max_rf = af;
        if (ef > err_f) err_f = ef;
        if (e8 > err_8) err_8 = e8;
        if (eq > err_q) err_q = eq;
        if (eq <= BND[m][n] * 1.0001 + 1e-6) within++;
    }
    const double cnt = (double) (Q4G_M * Q4G_N);
    Q4G_PRINT("Q4G K=%d unbias=%d: int-HMX within 2^8 bound of int64 ref %d/%d | vs f32 ref: int-HMX max %.3g rms %.3g"
              "  |  Q8_0-activation (existing HVX scheme) max %.3g rms %.3g  (max|y| %.3g)",
              Q4G_K, unbias, within, Q4G_M * Q4G_N, err_f, __builtin_sqrt(rms_f / cnt),
              err_8, __builtin_sqrt(rms_8 / cnt), max_rf);
}

__attribute__((noinline, optnone))
static void hmxq4gemm_run(unsigned char * base) {
    hmxq4gemm_k(base, 256, 0);
    hmxq4gemm_k(base, 256, 1);
    hmxq4gemm_k(base, 1536, 0);
    hmxq4gemm_k(base, 1536, 1);
}

#endif
