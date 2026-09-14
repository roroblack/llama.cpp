// Decode the fp16 HMX conversion RECORD in the simulator, so a device test can use the real format
// instead of my guess. Qualcomm loads it with bias = mxmem2(256-byte record) per output tile; I have only
// ever tried "0x00003c00 in words 0..31". Here: 1.0 x 1.0 MAC (accumulator 32.0 per output column c), then
// set exactly one record word to an fp16 value and watch which output column changes and to what, for both
// bias=mxmem2 and bias=mxmem. That maps record-word -> column and the scale encoding.
//
// FP16_PRINT must be defined; hmx-int.h included first.
#ifndef HMXFP16REC_H
#define HMXFP16REC_H
#include <string.h>

static void __attribute__((noinline)) fr_clr(void) { asm volatile("mxclracc.hf\n" ::: "memory"); }
static void __attribute__((noinline)) fr_b2(const void * r) { asm volatile("bias = mxmem2(%0)\n" :: "r"(r) : "memory"); }
static void __attribute__((noinline)) fr_b1(const void * r) { asm volatile("bias = mxmem(%0)\n" :: "r"(r) : "memory"); }
static void __attribute__((noinline)) fr_mac(const void * a, const void * w) {
    asm volatile("{\n activation.hf = mxmem(%0, %2)\n weight.hf = mxmem(%1, %2)\n}\n" :: "r"(a), "r"(w), "r"(2047) : "memory"); }
static void __attribute__((noinline)) fr_after(void * o) { asm volatile("mxmem(%0, %1):after.hf = acc\n" :: "r"(o), "r"(0) : "memory"); }

__attribute__((noinline, optnone)) static void hmxfp16rec_run(uint8_t * base) {
    uint16_t * A = (uint16_t *) base;
    uint16_t * W = (uint16_t *) (base + 8192);
    uint16_t * O = (uint16_t *) (base + 16384);
    uint32_t * R = (uint32_t *) (base + 32768);

    // baseline: record all 0x00003c00 -> output should be 32.0 (0x5000) everywhere
    for (int load2 = 1; load2 >= 0; load2--) {
        memset(base, 0, 49152);
        for (int i = 0; i < 4096; i++) { A[i] = 0x3c00; W[i] = 0x3c00; }
        for (int i = 0; i < 1024; i++) O[i] = 0xa5a5;
        for (int i = 0; i < 64; i++) R[i] = 0x00003c00u;
        if (load2) fr_b2(R); else fr_b1(R);
        fr_clr(); fr_mac(A, W); fr_after(O);
        int e = 0; for (int i = 0; i < 1024; i++) e += O[i] == 0x5000;
        FP16_PRINT("REC baseline all-3c00 load %s: out[0..3] %04x %04x %04x %04x  (==5000 count %d)",
                   load2 ? "mxmem2" : "mxmem", O[0], O[1], O[2], O[3], e);
    }

    // impulse: set record word k to 0x00004000 (fp16 2.0) or a marker, all else 0x00003c00; report the
    // first four output columns of the first row so we see which column word k drives
    for (int variant = 0; variant < 2; variant++) {
        const uint32_t mark = variant == 0 ? 0x00004000u : 0x47003c00u;  // fp16 2.0 ; or "output bias 7" style
        for (int k = 0; k < 8; k++) {
            memset(base, 0, 49152);
            for (int i = 0; i < 4096; i++) { A[i] = 0x3c00; W[i] = 0x3c00; }
            for (int i = 0; i < 1024; i++) O[i] = 0xa5a5;
            for (int i = 0; i < 64; i++) R[i] = 0x00003c00u;
            R[k] = mark;
            fr_b2(R);
            fr_clr(); fr_mac(A, W); fr_after(O);
            FP16_PRINT("REC mark %08x at word %2d: row0 cols %04x %04x %04x %04x  col%d %04x col%d %04x",
                       mark, k, O[0], O[1], O[2], O[3], k, O[k < 32 ? k : 0], k + 32, O[(k + 32) < 1024 ? k + 32 : 0]);
        }
    }
    FP16_PRINT("REC done");
}
#endif
