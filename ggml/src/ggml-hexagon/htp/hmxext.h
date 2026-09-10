// Integer HMX: read a WIDE sum out of a 16-bit store, exactly.
//
// The store only emits int16 = floor(sum * scale / 512), but two measured facts make a wider
// read possible:
//   * scale 512.0 (fp16 0x6000) returns the raw sum, and the int16 output WRAPS mod 2^16
//     (no saturation) -> that store is exactly the LOW 16 bits of the sum;
//   * scale 2^-7 (fp16 0x2000) returns floor(sum / 65536) -> the HIGH 16 bits, for |sum| < 2^31.
// If a ":retain" store keeps the accumulator, both can be read from ONE accumulation:
//   sum = hi * 65536 + (unsigned) lo
// which would let a kernel keep Q4_0's per-32 block scales exactly instead of requantizing.
//
// Checked here, against a scalar reference, element by element:
//   R1  retain store twice with the same record -> identical tiles (retain really retains)
//   E1  K=32  (one Q4 block), A u16 0..65535, W -8..7 : lo (retain) then hi -> exact 32-bit sum
//   E2  same data, hi (retain) first then lo
//   E3  K=256 (8 tiles), A u16 0..65535, W -127..127 : |sum| up to ~2.13e9 < 2^31
//
// EXT_PRINT(fmt, ...) must be defined by the includer.

#ifndef HMXEXT_H
#define HMXEXT_H

#include <string.h>

static inline int ex_ia(int m, int k) { return 128 * (m >> 1) + 4 * k + 2 * (m & 1); }
static inline int ex_iw(int k, int c) { return 128 * (k >> 2) + 4 * c + (k & 3); }
static inline int ex_io(int m, int c) { return 64 * (m >> 1) + 2 * c + (m & 1); }

static void __attribute__((noinline)) ex_mac(unsigned char * a, signed char * w, unsigned rta, unsigned rtw) {
    asm volatile("{\n"
                 "  activation.ub = mxmem(%0, %2):deep\n"
                 "  weight.b      = mxmem(%1, %3)\n"
                 "}\n" :: "r"(a), "r"(w), "r"(rta), "r"(rtw) : "memory");
}
static void __attribute__((noinline)) ex_bias(unsigned * b) { asm volatile("bias = mxmem(%0)\n" :: "r"(b) : "memory"); }
static void __attribute__((noinline)) ex_clr(void)          { asm volatile("mxclracc\n" ::: "memory"); }
static void __attribute__((noinline)) ex_store(unsigned short * o) {
    asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n" :: "r"(o), "r"(0) : "memory");
}
static void __attribute__((noinline)) ex_store_retain(unsigned short * o) {
    asm volatile("mxmem(%0, %1):after:retain.uh = acc:2x1\n" :: "r"(o), "r"(0) : "memory");
}

static unsigned ex_lcg = 4242u;
static int ex_rand(int lo, int hi) {
    ex_lcg = ex_lcg * 1103515245u + 12345u;
    return lo + (int) ((ex_lcg >> 8) % (unsigned) (hi - lo + 1));
}

// optnone: with -flto -fvectorize, hexagon-link 19.0.07 crashed on this function inside
// HexagonVectorCombine (HvxIdioms::processFxpMulChopped, assertion "Unshifted mul should have been
// skipped") - it tried to turn the 64-bit reference loops into an HVX fixed-point multiply. This is a
// debug probe, so keeping the optimiser away from it costs nothing.
// base: 2 KB aligned VTCM, at least 64 KB
__attribute__((noinline, optnone))
static void hmxext_run(unsigned char * base) {
    unsigned char *  act   = base;                               // 8 tiles x 2048 = 16 KB
    signed char *    wgt   = (signed char *) (base + 16384);     // 8 tiles x 1024 = 8 KB
    unsigned short * lo    = (unsigned short *) (base + 24576);
    unsigned short * hi    = (unsigned short *) (base + 28672);
    unsigned *       r_lo  = (unsigned *) (base + 32768);
    unsigned *       r_hi  = (unsigned *) (base + 33024);
    static unsigned short A[32][256];
    static signed char    W[256][32];

    for (int i = 0; i < 64; i++) { r_lo[i] = i < 32 ? 0x00006000u : 0u; r_hi[i] = i < 32 ? 0x00002000u : 0u; }

    for (int t = 0; t < 4; t++) {
        const int K    = (t == 3) ? 256 : 32;
        const int wmax = (t == 3) ? 127 : 7;
        const int wmin = (t == 3) ? -127 : -8;
        memset(act, 0, 16384); memset(wgt, 0, 8192);
        for (int m = 0; m < 32; m++) for (int k = 0; k < K; k++) {
            int v = ex_rand(0, 65535); A[m][k] = (unsigned short) v;
            unsigned char * p = act + (k >> 5) * 2048 + ex_ia(m, k & 31);
            p[0] = (unsigned char) (v & 0xff); p[1] = (unsigned char) (v >> 8);
        }
        for (int k = 0; k < K; k++) for (int c = 0; c < 32; c++) {
            int v = ex_rand(wmin, wmax); W[k][c] = (signed char) v;
            wgt[(k >> 5) * 1024 + ex_iw(k & 31, c)] = (signed char) v;
        }
        for (int i = 0; i < 1024; i++) { lo[i] = 0xa5a5u; hi[i] = 0x5a5au; }

        const unsigned nt = (unsigned) (K / 32);
        if (t == 0) {                  // R1: retain twice, same record
            ex_bias(r_lo); ex_clr(); ex_mac(act, wgt, nt * 2048 - 1, nt * 1024 - 1);
            ex_store_retain(lo); ex_store(hi);
            int same = 0; for (int i = 0; i < 1024; i++) same += (lo[i] == hi[i]);
            EXT_PRINT("EXT R1 retain then store, same record: identical %d/1024", same);
            continue;
        }
        ex_bias(r_lo); ex_clr(); ex_mac(act, wgt, nt * 2048 - 1, nt * 1024 - 1);
        if (t == 2) { ex_bias(r_hi); ex_store_retain(hi); ex_bias(r_lo); ex_store(lo); }
        else        { ex_store_retain(lo); ex_bias(r_hi); ex_store(hi); }

        int ok = 0, first_bad = -1; long long bref = 0, bgot = 0;
        for (int m = 0; m < 32; m++) for (int c = 0; c < 32; c++) {
            long long s = 0;
            for (int k = 0; k < K; k++) s += (long long) A[m][k] * W[k][c];
            long long got = (long long) (short) hi[ex_io(m, c)] * 65536LL + (long long) lo[ex_io(m, c)];
            if (got == s) ok++;
            else if (first_bad < 0) { first_bad = m * 32 + c; bref = s; bgot = got; }
        }
        EXT_PRINT("EXT E%d K=%d %s: exact %d/1024 first_bad=%d (ref %lld got %lld)",
                  t, K, t == 2 ? "hi-then-lo" : "lo-then-hi", ok, first_bad, bref, bgot);
        int ok256 = 0;
        for (int m = 0; m < 32; m++) for (int c = 0; c < 32; c++) {
            long long s = 0;
            for (int k = 0; k < K; k++) s += (long long) A[m][k] * W[k][c];
            long long f = (s >= 0) ? (s / 256) * 256 : -((-s + 255) / 256) * 256;     // floor(s/256)*256
            long long got = (long long) (short) hi[ex_io(m, c)] * 65536LL + (long long) lo[ex_io(m, c)];
            ok256 += (got == f);
        }
        EXT_PRINT("EXT E%d K=%d: matches floor(sum/256)*256 in %d/1024", t, K, ok256);
    }

    // E4: is the 2^8 truncation in the ACCUMULATOR or in the CONVERSION?  Scale 32768.0 (0x7800)
    // makes the store floor(sum * 64) mod 2^16: bits 6..15 of the output are bits 0..9 of the sum.
    // If the accumulator kept every bit they come out; if it holds floor(sum/256) only bits 8,9 of
    // the sum survive (output is a multiple of 16384).
    {
        for (int i = 0; i < 64; i++) r_hi[i] = i < 32 ? 0x00007800u : 0u;
        memset(act, 0, 16384); memset(wgt, 0, 8192);
        for (int m = 0; m < 32; m++) for (int k = 0; k < 32; k++) {
            int v = ex_rand(0, 65535); A[m][k] = (unsigned short) v;
            unsigned char * p = act + ex_ia(m, k); p[0] = (unsigned char) (v & 0xff); p[1] = (unsigned char) (v >> 8);
        }
        for (int k = 0; k < 32; k++) for (int c = 0; c < 32; c++) {
            int v = ex_rand(-8, 7); W[k][c] = (signed char) v; wgt[ex_iw(k, c)] = (signed char) v;
        }
        for (int i = 0; i < 1024; i++) lo[i] = 0xa5a5u;
        ex_bias(r_hi); ex_clr(); ex_mac(act, wgt, 2047, 1023); ex_store(lo);
        int full = 0, trunc = 0, m16384 = 0;
        for (int m = 0; m < 32; m++) for (int c = 0; c < 32; c++) {
            long long s = 0;
            for (int k = 0; k < 32; k++) s += (long long) A[m][k] * W[k][c];
            unsigned short o = lo[ex_io(m, c)];
            long long f = (s >= 0) ? (s / 256) * 256 : -((-s + 255) / 256) * 256;
            full   += (o == (unsigned short) ((unsigned long long) (s * 64) & 0xffffu));
            trunc  += (o == (unsigned short) ((unsigned long long) (f * 64) & 0xffffu));
            m16384 += ((o & 0x3fffu) == 0);
        }
        EXT_PRINT("EXT E4 scale 32768: full-precision model %d/1024, accumulator-floor(s/256) model %d/1024, multiples of 16384 %d/1024",
                  full, trunc, m16384);
    }

    // E5 (Codex): ONE store at scale 2.0 (0x4000) gives u = Q mod 2^16 with Q = floor(sum/256).
    // For a Q4_0 block (K=32, w in -8..7, a in 0..65535) every possible Q lies in [L, U] with
    // U - L <= 65535, so Q = L + ((u - L) mod 65536) recovers it exactly from one store.
    // Three data sets: random, all w = -8 with a = 65535 (the most negative block), all w = +7.
    {
        for (int i = 0; i < 64; i++) r_hi[i] = i < 32 ? 0x00004000u : 0u;
        for (int set = 0; set < 3; set++) {
            memset(act, 0, 16384); memset(wgt, 0, 8192);
            for (int m = 0; m < 32; m++) for (int k = 0; k < 32; k++) {
                int v = set == 0 ? ex_rand(0, 65535) : 65535;
                A[m][k] = (unsigned short) v;
                unsigned char * p = act + ex_ia(m, k); p[0] = (unsigned char) (v & 0xff); p[1] = (unsigned char) (v >> 8);
            }
            for (int k = 0; k < 32; k++) for (int c = 0; c < 32; c++) {
                int v = set == 0 ? ex_rand(-8, 7) : (set == 1 ? -8 : 7);
                W[k][c] = (signed char) v; wgt[ex_iw(k, c)] = (signed char) v;
            }
            for (int i = 0; i < 1024; i++) lo[i] = 0xa5a5u;
            ex_bias(r_hi); ex_clr(); ex_mac(act, wgt, 2047, 1023); ex_store(lo);
            int ok = 0, first_bad = -1; long long bq = 0, bg = 0;
            for (int m = 0; m < 32; m++) for (int c = 0; c < 32; c++) {
                long long s = 0, neg = 0, pos = 0;
                for (int k = 0; k < 32; k++) {
                    s += (long long) A[m][k] * W[k][c];
                    if (W[k][c] < 0) neg += W[k][c]; else pos += W[k][c];
                }
                long long q  = (s >= 0) ? s / 256 : -((-s + 255) / 256);
                long long nL = 65535LL * neg;
                long long L  = (nL >= 0) ? nL / 256 : -((-nL + 255) / 256);
                long long u  = (long long) lo[ex_io(m, c)];
                long long d  = ((u - L) % 65536 + 65536) % 65536;
                long long rq = L + d;
                if (rq == q) ok++;
                else if (first_bad < 0) { first_bad = m * 32 + c; bq = q; bg = rq; }
                (void) pos;
            }
            EXT_PRINT("EXT E5 one-store scale 2.0, set %s: recovered floor(sum/256) %d/1024 first_bad=%d (ref %lld got %lld)",
                      set == 0 ? "random" : (set == 1 ? "all -8, a=65535" : "all +7, a=65535"), ok, first_bad, bq, bg);
        }
    }
}

#endif
