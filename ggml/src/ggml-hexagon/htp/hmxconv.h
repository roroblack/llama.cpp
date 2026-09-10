// Integer HMX: conversion record and K accumulation - shared by the hexagon-sim harness and
// the device debug build, so the two can be compared line for line.
//
// Simulator findings this re-checks on the device:
//   * only the LOW 32 words of the record matter - one word per output column
//   * that word is an fp16 scale: out = floor(sum * scale / 512)
//   * the integer record is loaded with "bias = mxmem" (128 B); mxmem2 gives garbage
//   * consecutive MAC packets accumulate and are converted once; a larger Rt covers more K
//
// CONV_PRINT(fmt, ...) must be defined by the includer.

#ifndef HMXCONV_H
#define HMXCONV_H

#include <string.h>

static inline int cv_ia(int m, int k) { return 128 * (m >> 1) + 4 * k + 2 * (m & 1); }
static inline int cv_iw(int k, int c) { return 128 * (k >> 2) + 4 * c + (k & 3); }
static inline int cv_io(int m, int c) { return 64 * (m >> 1) + 2 * c + (m & 1); }

static void __attribute__((noinline)) cv_mac(unsigned char * a, signed char * w, unsigned rta, unsigned rtw) {
    asm volatile("{\n"
                 "  activation.ub = mxmem(%0, %2):deep\n"
                 "  weight.b      = mxmem(%1, %3)\n"
                 "}\n" :: "r"(a), "r"(w), "r"(rta), "r"(rtw) : "memory");
}
static void __attribute__((noinline)) cv_bias(unsigned * b)  { asm volatile("bias = mxmem(%0)\n"  :: "r"(b) : "memory"); }
static void __attribute__((noinline)) cv_bias2(unsigned * b) { asm volatile("bias = mxmem2(%0)\n" :: "r"(b) : "memory"); }
static void __attribute__((noinline)) cv_clr(void)           { asm volatile("mxclracc\n" ::: "memory"); }
static void __attribute__((noinline)) cv_store(unsigned short * o) {
    asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n" :: "r"(o), "r"(0) : "memory");
}

static void cv_fill(unsigned char * act, signed char * wgt, int toff_a, int toff_w, int a, int w) {
    for (int m = 0; m < 32; m++) for (int k = 0; k < 32; k++) {
        act[toff_a + cv_ia(m, k)]     = (unsigned char) (a & 0xff);
        act[toff_a + cv_ia(m, k) + 1] = (unsigned char) ((a >> 8) & 0xff);
    }
    for (int k = 0; k < 32; k++) for (int c = 0; c < 32; c++) wgt[toff_w + cv_iw(k, c)] = (signed char) w;
}

// base: 2 KB aligned VTCM, at least 64 KB
static void hmxconv_run(unsigned char * base) {
    unsigned char *  act  = base + 0;
    signed char *    wgt  = (signed char *) (base + 16384);
    unsigned short * out  = (unsigned short *) (base + 32768);
    unsigned *       bias = (unsigned *) (base + 49152);

    static const int wv[3] = { 1, 3, -2 };      // sums 8192, 24576, -16384 at A=256, K=32
    static const unsigned rec[] = { 0x00003c00u, 0x00000000u, 0x00003000u, 0x00003800u,
                                    0x00003e00u, 0x00004000u, 0x00004400u, 0x00004800u, 0x00006000u };
    for (unsigned r = 0; r < sizeof rec / sizeof rec[0]; r++) {
        for (int half = 0; half < 3; half++) {
            if (half && rec[r] != 0x00003c00u) continue;
            short y[3];
            for (int s = 0; s < 3; s++) {
                memset(act, 0, 16384); memset(wgt, 0, 8192);
                for (int i = 0; i < 4096; i++) out[i] = 0xa5a5u;
                for (int i = 0; i < 64; i++) {
                    unsigned v = rec[r];
                    if (half == 1 && i >= 32) v = 0;
                    if (half == 2 && i <  32) v = 0;
                    bias[i] = v;
                }
                cv_fill(act, wgt, 0, 0, 256, wv[s]);
                cv_bias(bias); cv_clr(); cv_mac(act, wgt, 1023, 1023); cv_store(out);
                y[s] = (short) out[cv_io(0, 0)];
            }
            CONV_PRINT("CONV C rec=%08x half=%d S8192=%d S24576=%d Sm16384=%d", rec[r], half, y[0], y[1], y[2]);
        }
    }

    // per-column scales: column c gets fp16 scale 1.0 (c even) or 2.0 (c odd)
    memset(act, 0, 16384); memset(wgt, 0, 8192);
    for (int i = 0; i < 4096; i++) out[i] = 0xa5a5u;
    for (int i = 0; i < 64; i++) bias[i] = (i < 32) ? ((i & 1) ? 0x00004000u : 0x00003c00u) : 0u;
    cv_fill(act, wgt, 0, 0, 256, 1);
    cv_bias(bias); cv_clr(); cv_mac(act, wgt, 1023, 1023); cv_store(out);
    CONV_PRINT("CONV P per-column scale: Y[0][0]=%d Y[0][1]=%d Y[5][30]=%d Y[5][31]=%d (expect 16 32 16 32)",
               (short) out[cv_io(0, 0)], (short) out[cv_io(0, 1)], (short) out[cv_io(5, 30)], (short) out[cv_io(5, 31)]);

    // mxmem2 on the integer path
    memset(act, 0, 16384); memset(wgt, 0, 8192);
    for (int i = 0; i < 4096; i++) out[i] = 0xa5a5u;
    for (int i = 0; i < 64; i++) bias[i] = 0x00003c00u;
    cv_fill(act, wgt, 0, 0, 256, 1);
    cv_bias2(bias); cv_clr(); cv_mac(act, wgt, 1023, 1023); cv_store(out);
    CONV_PRINT("CONV M mxmem2 record on int path: Y[0][0]=%d", (short) out[cv_io(0, 0)]);

    // Activation signedness: every probe so far used 0..4095. A real kernel needs to know
    // whether the 16-bit activation is signed. 0xffff is -1 or 65535; 0x8000 is -32768 or 32768.
    for (int i = 0; i < 64; i++) bias[i] = 0x00003c00u;
    {
        short sg[2];
        static const int av[2] = { 0xffff, 0x8000 };
        for (int t = 0; t < 2; t++) {
            memset(act, 0, 16384); memset(wgt, 0, 8192);
            for (int i = 0; i < 4096; i++) out[i] = 0xa5a5u;
            cv_fill(act, wgt, 0, 0, av[t], 1);
            cv_bias(bias); cv_clr(); cv_mac(act, wgt, 1023, 1023); cv_store(out);
            sg[t] = (short) out[cv_io(0, 0)];
        }
        CONV_PRINT("CONV S act 0xffff x1 -> %d (signed -1, unsigned 4095)  act 0x8000 x1 -> %d (signed -2048, unsigned 2048)",
                   sg[0], sg[1]);
    }

    // Output range: a sum whose converted value exceeds int16. Saturate, wrap, or clip?
    //   A = 0x3fff (16383), W = 127, K = 32 -> 66,580,512 ; /512 = 130,040.
    {
        memset(act, 0, 16384); memset(wgt, 0, 8192);
        for (int i = 0; i < 4096; i++) out[i] = 0xa5a5u;
        cv_fill(act, wgt, 0, 0, 0x3fff, 127);
        cv_bias(bias); cv_clr(); cv_mac(act, wgt, 1023, 1023); cv_store(out);
        CONV_PRINT("CONV O big sum: raw %04x as int16 %d (true value 130040; 32767 = saturates)",
                   out[cv_io(0, 0)], (short) out[cv_io(0, 0)]);
        memset(act, 0, 16384); memset(wgt, 0, 8192);
        for (int i = 0; i < 4096; i++) out[i] = 0xa5a5u;
        cv_fill(act, wgt, 0, 0, 0x3fff, -128);
        cv_bias(bias); cv_clr(); cv_mac(act, wgt, 1023, 1023); cv_store(out);
        CONV_PRINT("CONV O big negative: raw %04x as int16 %d (true value -131064)",
                   out[cv_io(0, 0)], (short) out[cv_io(0, 0)]);
    }

    // K accumulation
    for (int i = 0; i < 64; i++) bias[i] = 0x00003c00u;
    memset(act, 0, 16384); memset(wgt, 0, 8192); cv_fill(act, wgt, 0, 0, 256, 1);
    cv_bias(bias); cv_clr(); cv_mac(act, wgt, 1023, 1023); cv_mac(act, wgt, 1023, 1023); cv_store(out);
    short k1 = (short) out[cv_io(0, 0)];
    memset(act, 0, 16384); memset(wgt, 0, 8192);
    cv_fill(act, wgt, 0, 0, 256, 1); cv_fill(act, wgt, 2048, 1024, 256, 3);
    cv_bias(bias); cv_clr(); cv_mac(act, wgt, 1023, 1023); cv_mac(act + 2048, wgt + 1024, 1023, 1023); cv_store(out);
    short k2 = (short) out[cv_io(0, 0)];
    memset(act, 0, 16384); memset(wgt, 0, 8192);
    cv_fill(act, wgt, 0, 0, 256, 1); cv_fill(act, wgt, 2048, 1024, 256, 3);
    cv_bias(bias); cv_clr(); cv_mac(act, wgt, 4095, 2047); cv_store(out);
    short k3 = (short) out[cv_io(0, 0)];
    memset(act, 0, 16384); memset(wgt, 0, 8192); cv_fill(act, wgt, 0, 0, 1, 1);
    cv_bias(bias); cv_clr();
    for (int i = 0; i < 32; i++) cv_mac(act, wgt, 1023, 1023);
    cv_store(out);
    short k4 = (short) out[cv_io(0, 0)];
    CONV_PRINT("CONV K same-tile x2=%d two-tiles=%d one-packet-Rt4095=%d 32xS32=%d (expect 32 64 64 2)", k1, k2, k3, k4);
}

#endif
