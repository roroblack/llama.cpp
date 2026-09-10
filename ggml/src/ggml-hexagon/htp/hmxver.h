// Integer HMX: verify the decoded tile geometry against a scalar reference.
//
// Decoded from the simulator impulse sweep (device == simulator bit for bit on the
// linearity probe, so the model stands in for the hardware on this datapath):
//   output Y[m][c]   -> out[64*(m>>1) + 2*c + (m&1)]          (":2x1" = row pairs per column)
//   activation A[m][k]-> 16-bit little-endian at byte 128*(m>>1) + 4*k + 2*(m&1)   (+1 = high)
//   weight   W[k][c] -> int8 at byte 128*(k>>2) + 4*c + (k&3)   (k = 4g + b is a HYPOTHESIS:
//                       the impulse grid only covered g = 0)
//   conversion       -> sum(A*W) >> 9 under a bias record of 0x00003c00 (rounding unknown)
//
// Two tests, decided before looking:
//   E  even byte 255 x weight 127. 16-bit activations predict (255*127)>>9 = 63 at Y[0][*];
//      "only the high byte is read" predicts 0.
//   R  random A (0..4095, both bytes live) x random W (-8..7), every element compared with
//      three candidate conversions: floor, round-half-up, truncate toward zero. A layout or
//      K-map error cannot survive this: a wrong pairing gives wrong sums nearly everywhere.
//
// VER_PRINT(fmt, ...) must be defined by the includer.

#ifndef HMXVER_H
#define HMXVER_H

#include "intlin.h"   // intlin_mac_mul: straight-line asm, the form that compiles

static inline int ver_act(int m, int k)  { return 128 * (m >> 1) + 4 * k + 2 * (m & 1); }
static inline int ver_wgt(int k, int c)  { return 128 * (k >> 2) + 4 * c + (k & 3); }
static inline int ver_out(int m, int c)  { return 64 * (m >> 1) + 2 * c + (m & 1); }

static unsigned ver_lcg = 12345u;
static int ver_rand(int lo, int hi) {
    ver_lcg = ver_lcg * 1103515245u + 12345u;
    return lo + (int) ((ver_lcg >> 8) % (unsigned) (hi - lo + 1));
}

static void hmxver_run(unsigned char * base) {
    unsigned char *  act  = base + 0;
    signed char *    wgt  = (signed char *) (base + 8192);
    unsigned short * out  = (unsigned short *) (base + 16384);
    unsigned *       bias = (unsigned *) (base + 32768);

    // ---- E: even (low) byte only ----
    memset(base, 0, INTLIN_AREA);
    for (int i = 0; i < 64; i++) bias[i] = 0x00003c00u;
    for (int i = 0; i < 4096; i++) out[i] = 0xa5a5u;
    for (int k = 0; k < 32; k++) for (int c = 0; c < 32; c++) wgt[ver_wgt(k, c)] = 127;
    act[ver_act(0, 0) + 0] = 255;                 // low byte of A[0][0]
    intlin_mac_mul(act, wgt, out, bias);
    VER_PRINT("VER E low-byte 255 x 127: Y[0][0]=%04x Y[0][31]=%04x Y[1][0]=%04x (16-bit predicts 003f)",
              out[ver_out(0, 0)], out[ver_out(0, 31)], out[ver_out(1, 0)]);

    // ---- R: random matmul against a scalar reference ----
    static int A[32][32], Wt[32][32];
    memset(base, 0, INTLIN_AREA);
    for (int i = 0; i < 64; i++) bias[i] = 0x00003c00u;
    for (int i = 0; i < 4096; i++) out[i] = 0xa5a5u;
    for (int m = 0; m < 32; m++) for (int k = 0; k < 32; k++) {
        int v = ver_rand(0, 4095); A[m][k] = v;
        act[ver_act(m, k) + 0] = (unsigned char) (v & 0xff);
        act[ver_act(m, k) + 1] = (unsigned char) (v >> 8);
    }
    for (int k = 0; k < 32; k++) for (int c = 0; c < 32; c++) {
        int v = ver_rand(-8, 7); Wt[k][c] = v; wgt[ver_wgt(k, c)] = (signed char) v;
    }
    intlin_mac_mul(act, wgt, out, bias);

    int ok_floor = 0, ok_round = 0, ok_trunc = 0, first_bad = -1;
    long bad_sum = 0; int bad_got = 0;
    for (int m = 0; m < 32; m++) for (int c = 0; c < 32; c++) {
        long s = 0;
        for (int k = 0; k < 32; k++) s += (long) A[m][k] * Wt[k][c];
        short got = (short) out[ver_out(m, c)];
        long fl = s >> 9;                               // arithmetic shift = floor
        long rd = (s + 256) >> 9;                       // round half up
        long tr = s / 512;                              // toward zero
        if (got == (short) fl) ok_floor++;
        if (got == (short) rd) ok_round++;
        if (got == (short) tr) ok_trunc++;
        if (got != (short) fl && got != (short) rd && got != (short) tr && first_bad < 0) {
            first_bad = m * 32 + c; bad_sum = s; bad_got = got;
        }
    }
    VER_PRINT("VER R random 32x32x32: match floor=%d round=%d trunc=%d /1024  first_bad=%d (sum=%ld got=%d) hash=%08x",
              ok_floor, ok_round, ok_trunc, first_bad, bad_sum, bad_got, intlin_hash(out, 1024));
}

#endif
