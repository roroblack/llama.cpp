// fp16 HMX, round 3: sweep the output-conversion SCALE that rounds 1/2 never varied.
//
// Rounds 1-2 always passed 0 as the cvt scale: `cvt.hf = acc(0..3)`. Disassembling the phone's own
// libQnnHtpV73Skel.so (2026-09-14) shows Qualcomm's split fp16 kernel does `r31 = memw(r5++#4); cvt.hf = acc(r31)`
// with r31 a nonzero value from a table, then a plain `mxmem(out, Rt) = cvt` whose Rt is the store RANGE (r11
// from the descriptor), not a scale. So the only scale knob is cvt's Rs, and rounds 1-2 only ever tried 0..3.
// For a float store a zero/near-zero scale could yield exactly the all-zeros seen on the device; the simulator
// accepted 0 (rounds 1-2 gave 32.0), so only the device decides. This probe holds the accumulator at 32.0
// (1.0 x 1.0, one MAC) and sweeps cvt Rs, with the store range fixed at 0 (rounds 1-2 showed range 0 writes all
// 1024). Each case prints its label BEFORE issuing HMX, so a fault names the culprit.
//
// FP16_PRINT(fmt, ...) must be defined by the includer; hmx-int.h must be included first.

#ifndef HMXFP16V3_H
#define HMXFP16V3_H

#include <string.h>

static void __attribute__((noinline)) f3_clr(void)              { asm volatile("mxclracc.hf\n" ::: "memory"); }
static void __attribute__((noinline)) f3_bias2(const void * r)  { asm volatile("bias = mxmem2(%0)\n" :: "r"(r) : "memory"); }
static void __attribute__((noinline)) f3_mac(const void * a, const void * w) {
    asm volatile("{\n activation.hf = mxmem(%0, %2)\n weight.hf = mxmem(%1, %2)\n}\n"
                 :: "r"(a), "r"(w), "r"(2047) : "memory");
}
// split convert+store: cvt.hf = acc(Rs) ; mxmem(out, 0) = cvt   -- Rs is the swept scale, store range fixed 0
static void __attribute__((noinline)) f3_cvt_store(void * o, unsigned rs) {
    asm volatile("cvt.hf = acc(%1)\n mxmem(%0, %2) = cvt\n" :: "r"(o), "r"(rs), "r"(0) : "memory");
}

__attribute__((noinline, optnone)) static void hmxfp16v3_run(uint8_t * base) {
    uint16_t * A = (uint16_t *) (base);
    uint16_t * W = (uint16_t *) (base + 8192);
    uint16_t * O = (uint16_t *) (base + 16384);
    uint32_t * R = (uint32_t *) (base + 32768);

    FP16_PRINT("FP16v3 start; integer self test before: %d/1024 wrong", hmxi_selftest(base + 49152));

    static const struct { const char * n; unsigned v; } S[] = {
        { "0",            0x00000000u },  // rounds 1-2 (control: expect all-zero on device)
        { "1",            0x00000001u },
        { "2",            0x00000002u },
        { "4",            0x00000004u },
        { "8",            0x00000008u },
        { "15",           0x0000000fu },
        { "16",           0x00000010u },
        { "hf1.0=3c00",   0x00003c00u },  // fp16 1.0 in the low half
        { "hf1.0x2=3c003c00", 0x3c003c00u },
        { "hf2.0=4000",   0x00004000u },
        { "512.0=6000",   0x00006000u },  // the integer path's "raw sum" scale
        { "0x0000000a",   0x0000000au },
        { "0x00000100",   0x00000100u },
        { "0x00010001",   0x00010001u },
        { "0x0000ffff",   0x0000ffffu },
    };
    const int NS = (int) (sizeof(S) / sizeof(S[0]));

    for (int si = 0; si < NS; si++) {
        FP16_PRINT("FP16v3 cvt scale %-16s: issuing", S[si].n);
        memset(base, 0, 49152);
        for (int i = 0; i < 4096; i++) { A[i] = 0x3c00; W[i] = 0x3c00; }   // 1.0
        for (int i = 0; i < 1024; i++) O[i] = 0xa5a5;                       // sentinel
        for (int i = 0; i < 64; i++) R[i] = i < 32 ? 0x00003c00u : 0u;      // scale-1.0 record (as rounds 1-2)
        f3_bias2(R);
        f3_clr();
        f3_mac(A, W);                                                       // accumulator = 32.0 per output
        f3_cvt_store(O, S[si].v);
        int e32 = 0, other = 0, zero = 0, sent = 0;
        for (int i = 0; i < 1024; i++) {
            if      (O[i] == 0x5000) e32++;
            else if (O[i] == 0xa5a5) sent++;
            else if (O[i] == 0x0000) zero++;
            else                     other++;
        }
        FP16_PRINT("FP16v3 cvt scale %-16s: 32.0 %4d  other-nonzero %4d  zero %4d  untouched %4d | %04x %04x %04x %04x",
                   S[si].n, e32, other, zero, sent, O[0], O[1], O[2], O[3]);
    }
    FP16_PRINT("FP16v3 integer self test after: %d/1024 wrong", hmxi_selftest(base + 49152));
}

#endif
