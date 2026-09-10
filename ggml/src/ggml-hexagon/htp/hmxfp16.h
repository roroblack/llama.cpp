// fp16 HMX, settled by experiment - designed by Codex 6.0 (q23), run by Claude. Debug-only probe.
//
// The claim "fp16 HMX does not compute on SM8735" rests on one self test that printed only the bad
// count and the first two outputs, so "1024 wrong, first 0000 0000" never proved all 1024 were zero.
// This probe runs, in one VTCM region (2 KB aligned; A +0, W +8192, O +16384, record +32768,
// record readback +40960):
//   integer self test -> every fp16 case -> integer self test again   (same lock, same region)
// fp16 cases (uniform tiles, so layout cannot turn a right answer into zero):
//   C0 clear, no MAC, store       expect 0x0000
//   C1 A=1 W=1, one MAC           expect 0x5000 (32)
//   C2 A=2 W=1                    expect 0x5400 (64)
//   C3 A=1 W=2                    expect 0x5400
//   C4 A=-1 W=1                   expect 0xd000 (-32)
//   C5 A=1 W=1, two MACs          expect 0x5400
// records: R1 0x00003c00 (scale 1), R2 0x00004000 (scale 2: C1 -> 64), R3 0x47003c00 (scale 1 +
//   output bias 7?: C0 -> 0x4700, C1 -> 0x50e0; only meaningful if the simulator agrees)
// record load: bias=mxmem2 (what the self test used) and bias=mxmem
// store (v73 has only fused convert+store): :after.hf, :before.hf, :after:pos.hf
// Every case prints zero / expected / untouched-sentinel counts over all 1024 outputs and the
// first four values; the loaded record is stored back and compared byte for byte.
//
// FP16_PRINT(fmt, ...) must be defined by the includer; hmx-int.h must be included first.

#ifndef HMXFP16_H
#define HMXFP16_H

#include <string.h>

static void __attribute__((noinline)) f16_clr(void) { asm volatile("mxclracc.hf\n" ::: "memory"); }
static void __attribute__((noinline)) f16_mac(const void * a, const void * w) {
    asm volatile("{\n"
                 "  activation.hf = mxmem(%0, %2):deep\n"
                 "  weight.hf     = mxmem(%1, %2)\n"
                 "}\n" :: "r"(a), "r"(w), "r"(2047) : "memory");
}
static void __attribute__((noinline)) f16_bias2(const void * r)  { asm volatile("bias = mxmem2(%0)\n" :: "r"(r) : "memory"); }
static void __attribute__((noinline)) f16_bias1(const void * r)  { asm volatile("bias = mxmem(%0)\n" :: "r"(r) : "memory"); }
static void __attribute__((noinline)) f16_rb2(void * r)          { asm volatile("mxmem2(%0) = bias\n" :: "r"(r) : "memory"); }
static void __attribute__((noinline)) f16_rb1(void * r)          { asm volatile("mxmem(%0) = bias\n" :: "r"(r) : "memory"); }
static void __attribute__((noinline)) f16_st_after(void * o)     { asm volatile("mxmem(%0, %1):after.hf = acc\n" :: "r"(o), "r"(0) : "memory"); }
static void __attribute__((noinline)) f16_st_before(void * o)    { asm volatile("mxmem(%0, %1):before.hf = acc\n" :: "r"(o), "r"(0) : "memory"); }
static void __attribute__((noinline)) f16_st_pos(void * o)       { asm volatile("mxmem(%0, %1):after:pos.hf = acc\n" :: "r"(o), "r"(0) : "memory"); }

static void f16_fill(uint16_t * p, uint16_t v) { for (int i = 0; i < 1024; i++) p[i] = v; }

// one case: returns number of outputs equal to `want`
static int f16_case(uint8_t * base, const char * cname, uint16_t av, uint16_t wv, int nmac, uint16_t want,
                    const char * rname, uint32_t rec_lo, int load2, int store) {
    uint16_t * A  = (uint16_t *) (base);
    uint16_t * W  = (uint16_t *) (base + 8192);
    uint16_t * O  = (uint16_t *) (base + 16384);
    uint32_t * R  = (uint32_t *) (base + 32768);
    uint8_t *  RB = base + 40960;
    memset(base, 0, 49152);
    f16_fill(A, av); f16_fill(W, wv);
    for (int i = 0; i < 1024; i++) O[i] = 0xa5a5;
    for (int i = 0; i < 64; i++) R[i] = i < 32 ? rec_lo : 0u;
    memset(RB, 0x5a, 256);
    if (load2) { f16_bias2(R); f16_rb2(RB); } else { f16_bias1(R); f16_rb1(RB); }
    f16_clr();
    for (int i = 0; i < nmac; i++) f16_mac(A, W);
    if (store == 0) f16_st_after(O); else if (store == 1) f16_st_before(O); else f16_st_pos(O);
    int zero = 0, exp = 0, sent = 0;
    for (int i = 0; i < 1024; i++) { zero += O[i] == 0; exp += O[i] == want; sent += O[i] == 0xa5a5; }
    int rbdiff = 0; const int rblen = load2 ? 256 : 128;
    for (int i = 0; i < rblen; i++) rbdiff += RB[i] != ((const uint8_t *) R)[i];
    FP16_PRINT("FP16 %s rec %s load %s store %s: want %04x -> exp %d zero %d untouched %d | first %04x %04x %04x %04x | record readback %d/%d bytes differ",
               cname, rname, load2 ? "mxmem2" : "mxmem", store == 0 ? "after" : store == 1 ? "before" : "pos",
               want, exp, zero, sent, O[0], O[1], O[2], O[3], rbdiff, rblen);
    return exp;
}

__attribute__((noinline, optnone)) static void hmxfp16_run(uint8_t * base) {
    FP16_PRINT("FP16 integer self test before: %d/1024 wrong", hmxi_selftest(base + 49152));
    static const struct { const char * n; uint16_t a, w; int mac; uint16_t want; } C[] = {
        { "C0 no-mac", 0x3c00, 0x3c00, 0, 0x0000 },
        { "C1 1x1",    0x3c00, 0x3c00, 1, 0x5000 },
        { "C2 2x1",    0x4000, 0x3c00, 1, 0x5400 },
        { "C3 1x2",    0x3c00, 0x4000, 1, 0x5400 },
        { "C4 -1x1",   0xbc00, 0x3c00, 1, 0xd000 },
        { "C5 1x1 x2", 0x3c00, 0x3c00, 2, 0x5400 },
    };
    int total_ok = 0, total = 0;
    for (int load2 = 1; load2 >= 0; load2--) for (int st = 0; st < 3; st++) for (unsigned c = 0; c < 6; c++) {
        total_ok += f16_case(base, C[c].n, C[c].a, C[c].w, C[c].mac, C[c].want, "R1", 0x00003c00u, load2, st) == 1024;
        total++;
    }
    // scale 2 and the "bias 7" record, after store only, both loads
    for (int load2 = 1; load2 >= 0; load2--) {
        f16_case(base, "C1 1x1", 0x3c00, 0x3c00, 1, 0x5400, "R2", 0x00004000u, load2, 0);
        f16_case(base, "C0 no-mac", 0x3c00, 0x3c00, 0, 0x4700, "R3", 0x47003c00u, load2, 0);
        f16_case(base, "C1 1x1", 0x3c00, 0x3c00, 1, 0x50e0, "R3", 0x47003c00u, load2, 0);
    }
    FP16_PRINT("FP16 summary: R1 cases with all 1024 outputs as expected: %d/%d", total_ok, total);
    FP16_PRINT("FP16 integer self test after: %d/1024 wrong", hmxi_selftest(base + 49152));
}

#endif
