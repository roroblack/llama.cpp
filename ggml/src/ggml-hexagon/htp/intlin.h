// Integer HMX linearity probe. Shared, byte for byte, by the hexagon-sim harness and the
// device self test, so the two outputs can be compared element by element.
//
// Why this and not the earlier probe: the earlier integer probe filled 1024-byte tiles in
// a VTCM area that was memset in the simulator but held stale data on the device, and the
// two disagreed in exactly the upper half of the output. If the unit reads more than the
// 1024 bytes that were initialised, stale memory alone explains that. So here the whole
// 64 KB work area is written deterministically before every test.
//
// Pass criteria (decided before looking at any result):
//   * T1 != T0            - the multiply contributes something the clear+store does not
//   * T2 and T3 follow T1  - doubling one operand changes the output consistently
//   * device == simulator  - bit-identical for every test
//
// T6..T9 test the over-read hypothesis the way Codex asked: keep the intended 1024 input
// bytes fixed and change ONLY the padding after them. If the output moves, the unit reads
// past 1024 bytes. (Codex's caveat stands: padding sensitivity shows a dependency outside
// the intended region, it does not measure every physical read.)
//
// The HMX sequence is inline asm in its own noinline function. The first version used the
// Q6_ intrinsics inside the test loop behind an `if`, and hexagon-clang sat at 100% CPU for
// over twelve minutes on it without finishing - the paired activation/weight packet under
// control flow appears to defeat its packetizer. Straight-line asm, which is what the fp16
// self test already uses, compiles immediately.
//
// INTLIN_PRINT(fmt, ...) must be defined by the includer (printf in the sim, FARF on DSP).

#ifndef INTLIN_H
#define INTLIN_H

#include <string.h>

#define INTLIN_AREA (64 * 1024)
#define INTLIN_NTEST 10

static unsigned intlin_hash(const unsigned short * o, int n) {
    unsigned h = 2166136261u;
    for (int i = 0; i < n; i++) { h ^= o[i]; h *= 16777619u; }
    return h;
}

static void __attribute__((noinline)) intlin_mac_mul(unsigned char * act, signed char * wgt,
                                                     unsigned short * out, unsigned * bias) {
    asm volatile("bias = mxmem(%0)\n" :: "r"(bias) : "memory");
    asm volatile("mxclracc\n" ::: "memory");
    asm volatile("{\n"
                 "  activation.ub = mxmem(%0, %2):deep\n"
                 "  weight.b      = mxmem(%1, %2)\n"
                 "}\n" :: "r"(act), "r"(wgt), "r"(1023) : "memory");
    asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n" :: "r"(out), "r"(0) : "memory");
}

static void __attribute__((noinline)) intlin_mac_nomul(unsigned short * out, unsigned * bias) {
    asm volatile("bias = mxmem(%0)\n" :: "r"(bias) : "memory");
    asm volatile("mxclracc\n" ::: "memory");
    asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n" :: "r"(out), "r"(0) : "memory");
}

// base must be 2 KB aligned VTCM with at least INTLIN_AREA bytes
static void intlin_run(unsigned char * base) {
    unsigned char *  act  = base + 0;                         // 8 KB reserved
    signed char *    wgt  = (signed char *) (base + 8192);    // 8 KB reserved
    unsigned short * out  = (unsigned short *) (base + 16384);// 8 KB reserved
    unsigned *       bias = (unsigned *) (base + 32768);      // 256-byte aligned

    for (int t = 0; t < INTLIN_NTEST; t++) {
        memset(base, 0, INTLIN_AREA);                         // deterministic everywhere
        for (int i = 0; i < 64; i++) bias[i] = 0x00003c00u;
        for (int i = 0; i < 4096; i++) {
            int a, w;
            const int head = i < 1024;                        // the intended 32x32 tile
            switch (t) {
                case 1:  a = 1;     w = 1;           break;
                case 2:  a = 2;     w = 1;           break;
                case 3:  a = 1;     w = 2;           break;
                case 4:  a = 0;     w = 1;           break;
                case 5:  a = i % 7; w = (i % 5) - 2; break;
                case 6:  a = head ? 1 : 0;    w = 1;               break;  // act padding 0
                case 7:  a = head ? 1 : 0x55; w = 1;               break;  // act padding 0x55
                case 8:  a = 1;               w = head ? 1 : 0;    break;  // wgt padding 0
                case 9:  a = 1;               w = head ? 1 : 0x33; break;  // wgt padding 0x33
                default: a = 1;     w = 1;           break;   // T0: data present, no multiply
            }
            act[i] = (unsigned char) a;
            wgt[i] = (signed char) w;
        }
        for (int i = 0; i < 4096; i++) out[i] = 0xffffu;

        if (t == 0) { intlin_mac_nomul(out, bias); }
        else        { intlin_mac_mul(act, wgt, out, bias); }

        int written = 0, nz = 0;
        for (int i = 0; i < 4096; i++) {
            if (out[i] != 0xffffu) { written++; if (out[i]) nz++; }
        }
        INTLIN_PRINT("INTLIN T%d written=%d nonzero=%d hash=%08x o0-7=%04x %04x %04x %04x %04x %04x %04x %04x"
                     " o512-515=%04x %04x %04x %04x o1023=%04x",
                     t, written, nz, intlin_hash(out, 4096),
                     out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
                     out[512], out[513], out[514], out[515], out[1023]);
    }
}

#endif
