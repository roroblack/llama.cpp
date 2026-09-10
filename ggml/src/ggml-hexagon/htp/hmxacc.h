// Integer HMX: how wide is the accumulator?
//
// Long K in one packet is exact up to K=1024 with moderate data (device == simulator). Codex:
// if the accumulator is signed 32-bit, u16 x int8 over the model's K (up to 12288) can overflow -
// 12288 * 65535 * 127 ~ 1.0e11 >> 2^31. So drive the sum across 2^31 and watch.
//
// The int16 output only shows the low 16 bits of floor(sum * s / 512), so use a tiny column
// scale to bring the HIGH bits into view: s = fp16 2^-14 (0x0400) -> out = floor(sum / 2^23).
// A = 65535, W = +127 (and -128 for the negative side), K = 32 * nt:
//   nt=4  (K=128)  sum = 1.07e9  -> 126        nt=8  (K=256)  sum = 2.13e9 -> 253
//   nt=12 (K=384)  sum = 3.20e9  -> 380 if the accumulator is wider than 32 bits,
//                                   about -132 if it wraps at 32 bits signed
//   nt=16 (K=512), nt=32 (K=1024) likewise.
//
// ACC_PRINT(fmt, ...) must be defined by the includer.

#ifndef HMXACC_H
#define HMXACC_H

#include <string.h>

static inline int ac_ia(int m, int k) { return 128 * (m >> 1) + 4 * k + 2 * (m & 1); }
static inline int ac_iw(int k, int c) { return 128 * (k >> 2) + 4 * c + (k & 3); }
static inline int ac_io(int m, int c) { return 64 * (m >> 1) + 2 * c + (m & 1); }

static void __attribute__((noinline)) ac_mac(unsigned char * a, signed char * w, unsigned rta, unsigned rtw) {
    asm volatile("{\n"
                 "  activation.ub = mxmem(%0, %2):deep\n"
                 "  weight.b      = mxmem(%1, %3)\n"
                 "}\n" :: "r"(a), "r"(w), "r"(rta), "r"(rtw) : "memory");
}
static void __attribute__((noinline)) ac_bias(unsigned * b) { asm volatile("bias = mxmem(%0)\n" :: "r"(b) : "memory"); }
static void __attribute__((noinline)) ac_clr(void)          { asm volatile("mxclracc\n" ::: "memory"); }
static void __attribute__((noinline)) ac_store(unsigned short * o) {
    asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n" :: "r"(o), "r"(0) : "memory");
}

// base: 2 KB aligned VTCM, at least 112 KB
static void hmxacc_run(unsigned char * base) {
    unsigned char *  act  = base;
    signed char *    wgt  = (signed char *) (base + 65536);
    unsigned short * out  = (unsigned short *) (base + 98304);
    unsigned *       bias = (unsigned *) (base + 106496);
    static const int nts[] = { 4, 8, 12, 16, 32 };
    static const int wv[2] = { 127, -128 };

    for (int i = 0; i < 64; i++) bias[i] = (i < 32) ? 0x00000400u : 0u;   // fp16 2^-14
    for (int s = 0; s < 2; s++) {
        memset(base, 0, 112 * 1024);
        for (int i = 0; i < 64; i++) bias[i] = (i < 32) ? 0x00000400u : 0u;
        for (int t = 0; t < 32; t++) {
            for (int m = 0; m < 32; m++) for (int k = 0; k < 32; k++) {
                unsigned char * p = act + t * 2048 + ac_ia(m, k); p[0] = 0xff; p[1] = 0xff;
            }
            for (int k = 0; k < 32; k++) for (int c = 0; c < 32; c++) wgt[t * 1024 + ac_iw(k, c)] = (signed char) wv[s];
        }
        for (unsigned j = 0; j < sizeof nts / sizeof nts[0]; j++) {
            const int nt = nts[j];
            for (int i = 0; i < 1024; i++) out[i] = 0xa5a5u;
            ac_bias(bias); ac_clr(); ac_mac(act, wgt, nt * 2048 - 1, nt * 1024 - 1); ac_store(out);
            long long sum = 65535LL * wv[s] * 32 * nt;
            ACC_PRINT("ACC w=%d K=%d sum=%lld (vs 2^31=2147483648) out=%d (wide acc predicts %lld)",
                      wv[s], 32 * nt, sum, (short) out[ac_io(0, 0)],
                      sum >= 0 ? sum >> 23 : -((-sum + (1LL << 23) - 1) >> 23));
        }
    }
}

#endif
