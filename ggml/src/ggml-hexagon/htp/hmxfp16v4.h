// fp16 HMX, round 4: is the fp16 MULTIPLIER alive on the device, or only the fp16 CONVERT/store dead?
//
// Every device check so far read the accumulator through an fp16 store (:after.hf / cvt.hf) - the part that may
// be the dead one. The accumulator register file is shared with the integer path, which computes exactly on this
// device. So: clear with mxclracc.hf, do an fp16 MAC (1.0 x 1.0), then read the accumulator through the INTEGER
// store (cvt.ub / cvt.uh / :after.uh) - never touching fp16 conversion. If those bytes are nonzero, the fp16
// multiply-accumulate ran and only the fp16 output path is dead (then we can convert in HVX). If they are zero,
// the fp16 MAC itself produces nothing. A control reads the accumulator after mxclracc.hf with NO MAC.
// The simulator (fp16 fully working) gives the reference bit pattern of "32.0 in the accumulator".
//
// FP16_PRINT(fmt, ...) must be defined by the includer; hmx-int.h must be included first.

#ifndef HMXFP16V4_H
#define HMXFP16V4_H

#include <string.h>

static void __attribute__((noinline)) f4_clr_hf(void)  { asm volatile("mxclracc.hf\n" ::: "memory"); }
static void __attribute__((noinline)) f4_clr_i(void)   { asm volatile("mxclracc\n" ::: "memory"); }
static void __attribute__((noinline)) f4_mac_hf(const void * a, const void * w) {
    asm volatile("{\n activation.hf = mxmem(%0, %2)\n weight.hf = mxmem(%1, %2)\n}\n"
                 :: "r"(a), "r"(w), "r"(2047) : "memory");
}
static void __attribute__((noinline)) f4_mac_i(const void * a, const void * w) {
    asm volatile("{\n activation.ub = mxmem(%0, %2):deep\n weight.b = mxmem(%1, %2)\n}\n"
                 :: "r"(a), "r"(w), "r"(2047) : "memory");
}
// integer reads of the accumulator (never fp16)
static void __attribute__((noinline)) f4_rd_after_uh(void * o) { asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n" :: "r"(o), "r"(0) : "memory"); }
static void __attribute__((noinline)) f4_rd_cvt_uh(void * o)   { asm volatile("cvt.uh = acc(%1):2x2\n mxmem(%0, %2) = cvt\n" :: "r"(o), "r"(0x6000), "r"(0) : "memory"); }
static void __attribute__((noinline)) f4_rd_cvt_ub(void * o)   { asm volatile("cvt.ub = acc(%1)\n mxmem(%0, %2) = cvt\n" :: "r"(o), "r"(0x6000), "r"(0) : "memory"); }
// fp16 read, for comparison
static void __attribute__((noinline)) f4_rd_hf(void * o)       { asm volatile("mxmem(%0, %1):after.hf = acc\n" :: "r"(o), "r"(0) : "memory"); }

static int f4_count_nz(const uint8_t * p, int n) { int c = 0; for (int i = 0; i < n; i++) c += p[i] != 0; return c; }

__attribute__((noinline, optnone)) static void hmxfp16v4_run(uint8_t * base) {
    uint16_t * A = (uint16_t *) (base);
    uint16_t * W = (uint16_t *) (base + 8192);
    uint8_t  * O = (uint8_t  *) (base + 16384);
    uint32_t * R = (uint32_t *) (base + 32768);

    FP16_PRINT("FP16v4 integer self test before: %d/1024 wrong", hmxi_selftest(base + 49152));

    // reference: the integer path, exact on this device, read via the same integer stores
    for (int rd = 0; rd < 3; rd++) {
        memset(base, 0, 49152);
        for (int i = 0; i < 4096; i++) { ((uint16_t*)A)[i] = 0; }
        uint8_t * Ai = (uint8_t *) base; uint8_t * Wi = (uint8_t *) (base + 8192);
        for (int i = 0; i < 8192; i++) { Ai[i] = 1; Wi[i] = 1; }   // int 1 x 1
        for (int i = 0; i < 64; i++) R[i] = i < 32 ? 0x00006000u : 0u;   // raw-sum scale
        for (int i = 0; i < 2048; i++) O[i] = 0xa5;
        f4_clr_i();
        asm volatile("bias = mxmem(%0)\n" :: "r"(R) : "memory");
        f4_mac_i(Ai, Wi);
        if (rd == 0) f4_rd_after_uh(O); else if (rd == 1) f4_rd_cvt_uh(O); else f4_rd_cvt_ub(O);
        FP16_PRINT("FP16v4 INT   mac + int-read[%d]: nonzero bytes %d/2048 | %02x %02x %02x %02x %02x %02x %02x %02x",
                   rd, f4_count_nz(O, 2048), O[0], O[1], O[2], O[3], O[4], O[5], O[6], O[7]);
    }

    // the question: fp16 clear + fp16 MAC, read the accumulator through the integer store
    for (int rd = 0; rd < 4; rd++) {
        memset(base, 0, 49152);
        for (int i = 0; i < 4096; i++) { A[i] = 0x3c00; W[i] = 0x3c00; }   // fp16 1.0
        for (int i = 0; i < 64; i++) R[i] = i < 32 ? 0x00006000u : 0u;
        for (int i = 0; i < 2048; i++) O[i] = 0xa5;
        f4_clr_hf();
        asm volatile("bias = mxmem(%0)\n" :: "r"(R) : "memory");
        f4_mac_hf(A, W);
        if (rd == 0) f4_rd_after_uh(O); else if (rd == 1) f4_rd_cvt_uh(O); else if (rd == 2) f4_rd_cvt_ub(O); else f4_rd_hf(O);
        FP16_PRINT("FP16v4 HFMAC + %s-read[%d]: nonzero bytes %d/2048 | %02x %02x %02x %02x %02x %02x %02x %02x",
                   rd == 3 ? "hf " : "int", rd, f4_count_nz(O, 2048), O[0], O[1], O[2], O[3], O[4], O[5], O[6], O[7]);
    }

    // control: fp16 clear, NO mac, integer read (the cleared-accumulator pattern)
    {
        memset(base, 0, 49152);
        for (int i = 0; i < 64; i++) R[i] = i < 32 ? 0x00006000u : 0u;
        for (int i = 0; i < 2048; i++) O[i] = 0xa5;
        f4_clr_hf();
        asm volatile("bias = mxmem(%0)\n" :: "r"(R) : "memory");
        f4_rd_after_uh(O);
        FP16_PRINT("FP16v4 HFCLR no-mac + int-read: nonzero bytes %d/2048 | %02x %02x %02x %02x",
                   f4_count_nz(O, 2048), O[0], O[1], O[2], O[3]);
    }

    FP16_PRINT("FP16v4 integer self test after: %d/1024 wrong", hmxi_selftest(base + 49152));
}

#endif
