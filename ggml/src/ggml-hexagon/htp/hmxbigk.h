// Integer HMX: is a long K in ONE packet still exact?
//
// The throughput probe put up to 32 K-tiles in one packet (Rt spanning 64 KB of activations)
// and the cycle count scaled exactly with the tile count. Speed is worthless if the answer is
// wrong, and K accumulation was only verified for two tiles. So: K = 1024 (32 tiles), random
// data, one packet versus 32 single-tile packets, both against a scalar reference.
//
// Tile t of K sits at act + t*2048 and wgt + t*1024, with the decoded in-tile layout
// (hmxver.h). Sums reach ~1024*4095*8 = 33.5M, so the column scale is fp16 0.125 (0x3000):
// out = floor(sum * 0.125 / 512) = floor(sum / 4096), which keeps every result in int16.
//
// BIGK_PRINT(fmt, ...) must be defined by the includer.

#ifndef HMXBIGK_H
#define HMXBIGK_H

#include <string.h>

static inline int bk_ia(int m, int k) { return 128 * (m >> 1) + 4 * k + 2 * (m & 1); }
static inline int bk_iw(int k, int c) { return 128 * (k >> 2) + 4 * c + (k & 3); }
static inline int bk_io(int m, int c) { return 64 * (m >> 1) + 2 * c + (m & 1); }

static void __attribute__((noinline)) bk_mac(unsigned char * a, signed char * w, unsigned rta, unsigned rtw) {
    asm volatile("{\n"
                 "  activation.ub = mxmem(%0, %2):deep\n"
                 "  weight.b      = mxmem(%1, %3)\n"
                 "}\n" :: "r"(a), "r"(w), "r"(rta), "r"(rtw) : "memory");
}
static void __attribute__((noinline)) bk_bias(unsigned * b) { asm volatile("bias = mxmem(%0)\n" :: "r"(b) : "memory"); }
static void __attribute__((noinline)) bk_clr(void)          { asm volatile("mxclracc\n" ::: "memory"); }
static void __attribute__((noinline)) bk_store(unsigned short * o) {
    asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n" :: "r"(o), "r"(0) : "memory");
}

static unsigned bk_lcg = 777u;
static int bk_rand(int lo, int hi) {
    bk_lcg = bk_lcg * 1103515245u + 12345u;
    return lo + (int) ((bk_lcg >> 8) % (unsigned) (hi - lo + 1));
}
static unsigned bk_hash(const unsigned short * o, int n) {
    unsigned h = 2166136261u;
    for (int i = 0; i < n; i++) { h ^= o[i]; h *= 16777619u; }
    return h;
}

#define BK_K   1024
#define BK_NT  (BK_K / 32)

// base: 2 KB aligned VTCM, at least 112 KB
static void hmxbigk_run(unsigned char * base) {
    unsigned char *  act  = base;                              // 32 tiles x 2048 = 64 KB
    signed char *    wgt  = (signed char *) (base + 65536);    // 32 tiles x 1024 = 32 KB
    unsigned short * out  = (unsigned short *) (base + 98304); // 2 KB used
    unsigned *       bias = (unsigned *) (base + 106496);
    static unsigned short A[32][BK_K];
    static signed char    W[BK_K][32];

    memset(base, 0, 112 * 1024);
    for (int m = 0; m < 32; m++) for (int k = 0; k < BK_K; k++) {
        int v = bk_rand(0, 4095); A[m][k] = (unsigned short) v;
        unsigned char * p = act + (k >> 5) * 2048 + bk_ia(m, k & 31);
        p[0] = (unsigned char) (v & 0xff); p[1] = (unsigned char) (v >> 8);
    }
    for (int k = 0; k < BK_K; k++) for (int c = 0; c < 32; c++) {
        int v = bk_rand(-8, 7); W[k][c] = (signed char) v;
        wgt[(k >> 5) * 1024 + bk_iw(k & 31, c)] = (signed char) v;
    }
    for (int i = 0; i < 64; i++) bias[i] = (i < 32) ? 0x00003000u : 0u;   // fp16 0.125 per column

    for (int mode = 0; mode < 2; mode++) {
        for (int i = 0; i < 1024; i++) out[i] = 0xa5a5u;
        bk_bias(bias); bk_clr();
        if (mode == 0) {
            bk_mac(act, wgt, BK_NT * 2048 - 1, BK_NT * 1024 - 1);           // one packet, K = 1024
        } else {
            for (int t = 0; t < BK_NT; t++) bk_mac(act + t * 2048, wgt + t * 1024, 2047, 1023);
        }
        bk_store(out);

        int ok = 0, first_bad = -1; long bad_ref = 0; int bad_got = 0;
        for (int m = 0; m < 32; m++) for (int c = 0; c < 32; c++) {
            long s = 0;
            for (int k = 0; k < BK_K; k++) s += (long) A[m][k] * W[k][c];
            long ref = s >> 12;                      // floor(s * 0.125 / 512)
            short got = (short) out[bk_io(m, c)];
            if (got == (short) ref) ok++;
            else if (first_bad < 0) { first_bad = m * 32 + c; bad_ref = ref; bad_got = got; }
        }
        BIGK_PRINT("BIGK %s K=%d: match %d/1024 first_bad=%d (ref %ld got %d) hash=%08x",
                   mode == 0 ? "one-packet" : "32-packets", BK_K, ok, first_bad, bad_ref, bad_got,
                   bk_hash(out, 1024));
    }
}

#endif
