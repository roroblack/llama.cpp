// fp16 HMX, second round: Qualcomm's own v73 instruction sequence. Debug-only probe.
//
// Round 1 (hmxfp16.h) used `activation.hf = mxmem(A, Rt):deep` for every MAC and only the fused
// store `mxmem(O, 0):after/before.hf = acc`. Every case stored zero on SM8735. Disassembling the
// phone's own libQnnHtpV73Skel.so (2026-09-14) shows Qualcomm's fp16 HMX code NEVER uses the
// `:deep` activation form on v73. Its fp16 kernels use:
//   hmx_v73_convf16_1x1_stride1:  { activation.hf = mxmem(A,Rt) ; weight.hf = mxmem(W,Rt) }
//                                 then per output slice: bias = mxmem2(rec) ; cvt.hf = acc(Rs) ;
//                                 mxmem(O,Rt) = cvt
//   hmx_convf16_1x1_stride1_unaligned: bias = mxmem(rec) ; mxclracc.hf ;
//                                 { activation.hf = mxmem(A,Rt):single ; weight.hf = mxmem(W,Rt):single }
//                                 then mxmem(O,Rt):after.hf / :after:retain.hf / :before.hf = acc
//   other fp16 kernels:           :above + plain activation pairs with mxswapacc.hf between them.
// This probe runs the same uniform-tile cases (A=1, W=1 -> 32.0 = 0x5000 everywhere) through
// each MAC form x each output form, in one 2 KB-aligned VTCM region, bracketed by the integer
// self test. Nothing here is used by the kernels.
//
// FP16_PRINT(fmt, ...) must be defined by the includer; hmx-int.h must be included first.

#ifndef HMXFP16V2_H
#define HMXFP16V2_H

#include <string.h>

// ---- MAC forms (one packet each, straight-line asm in noinline functions: see memory note on
// hexagon-clang spinning when HMX pairs sit in loops behind branches) ----
static void __attribute__((noinline)) f2_mac_plain(const void * a, const void * w) {
    asm volatile("{\n activation.hf = mxmem(%0, %2)\n weight.hf = mxmem(%1, %2)\n}\n"
                 :: "r"(a), "r"(w), "r"(2047) : "memory");
}
static void __attribute__((noinline)) f2_mac_single(const void * a, const void * w) {
    asm volatile("{\n activation.hf = mxmem(%0, %2):single\n weight.hf = mxmem(%1, %2):single\n}\n"
                 :: "r"(a), "r"(w), "r"(2047) : "memory");
}
static void __attribute__((noinline)) f2_mac_deep(const void * a, const void * w) {
    asm volatile("{\n activation.hf = mxmem(%0, %2):deep\n weight.hf = mxmem(%1, %2)\n}\n"
                 :: "r"(a), "r"(w), "r"(2047) : "memory");
}
static void __attribute__((noinline)) f2_mac_deep2(const void * a, const void * w) {
    asm volatile("{\n activation.hf = mxmem(%0, %2):deep\n weight.hf = mxmem(%1, %2):deep\n}\n"
                 :: "r"(a), "r"(w), "r"(2047) : "memory");
}
static void __attribute__((noinline)) f2_mac_above(const void * a, const void * w) {
    asm volatile("{\n activation.hf = mxmem(%0, %2):above\n weight.hf = mxmem(%1, %2)\n}\n"
                 :: "r"(a), "r"(w), "r"(2047) : "memory");
}

static void __attribute__((noinline)) f2_clr(void)               { asm volatile("mxclracc.hf\n" ::: "memory"); }
static void __attribute__((noinline)) f2_swap(void)              { asm volatile("mxswapacc.hf\n" ::: "memory"); }
static void __attribute__((noinline)) f2_bias2(const void * r)   { asm volatile("bias = mxmem2(%0)\n" :: "r"(r) : "memory"); }
static void __attribute__((noinline)) f2_bias1(const void * r)   { asm volatile("bias = mxmem(%0)\n" :: "r"(r) : "memory"); }

// ---- output forms ----
static void __attribute__((noinline)) f2_st_after(void * o)      { asm volatile("mxmem(%0, %1):after.hf = acc\n" :: "r"(o), "r"(0) : "memory"); }
static void __attribute__((noinline)) f2_st_before(void * o)     { asm volatile("mxmem(%0, %1):before.hf = acc\n" :: "r"(o), "r"(0) : "memory"); }
static void __attribute__((noinline)) f2_st_retain(void * o)     { asm volatile("mxmem(%0, %1):after:retain.hf = acc\n" :: "r"(o), "r"(0) : "memory"); }
static void __attribute__((noinline)) f2_cvt_st(void * o, int rs) {
    asm volatile("cvt.hf = acc(%1)\n mxmem(%0, %2) = cvt\n" :: "r"(o), "r"(rs), "r"(0) : "memory");
}

enum { F2_PLAIN, F2_SINGLE, F2_DEEP, F2_DEEP2, F2_ABOVE, F2_NMAC };
static const char * const f2_mac_name[F2_NMAC] = { "plain", "single", "deep", "deep2", "above" };

enum { F2_ST_AFTER, F2_ST_BEFORE, F2_ST_RETAIN, F2_ST_CVT0, F2_ST_CVT1, F2_ST_CVT2, F2_ST_CVT3, F2_NST };
static const char * const f2_st_name[F2_NST] = { "after", "before", "after:retain", "cvt0", "cvt1", "cvt2", "cvt3" };

static void f2_mac(int form, const void * a, const void * w) {
    switch (form) {
        case F2_PLAIN:  f2_mac_plain(a, w);  break;
        case F2_SINGLE: f2_mac_single(a, w); break;
        case F2_DEEP:   f2_mac_deep(a, w);   break;
        case F2_DEEP2:  f2_mac_deep2(a, w);  break;
        default:        f2_mac_above(a, w);  break;
    }
}
static void f2_store(int st, void * o) {
    switch (st) {
        case F2_ST_AFTER:  f2_st_after(o);  break;
        case F2_ST_BEFORE: f2_st_before(o); break;
        case F2_ST_RETAIN: f2_st_retain(o); break;
        default:           f2_cvt_st(o, st - F2_ST_CVT0); break;
    }
}

static void f2_fill(uint16_t * p, uint16_t v) { for (int i = 0; i < 1024; i++) p[i] = v; }

// One case. seq: 0 = bias, clear, MAC, store (round-1 order)
//                1 = clear, bias, MAC, store (Qualcomm unaligned kernel loads bias first, then clears;
//                    this variant flips it to rule order in or out)
//                2 = bias, clear, MAC, mxswapacc.hf, store
//                3 = bias, clear, MAC, mxswapacc.hf, MAC, mxswapacc.hf, store (Qualcomm pair pattern)
// Returns the number of outputs equal to 0x5000 (32.0) - or 0x5400 (64.0) where two MACs land
// in one accumulator; both counts are printed.
static int f2_case(uint8_t * base, int mac, int st, int load2, int seq) {
    uint16_t * A  = (uint16_t *) (base);
    uint16_t * W  = (uint16_t *) (base + 8192);
    uint16_t * O  = (uint16_t *) (base + 16384);
    uint32_t * R  = (uint32_t *) (base + 32768);
    memset(base, 0, 49152);
    // deep/single forms may read past one tile: fill 4 tiles of A and W so any over-read is 1.0 too
    for (int i = 0; i < 4096; i++) { A[i] = 0x3c00; W[i] = 0x3c00; }
    for (int i = 0; i < 1024; i++) O[i] = 0xa5a5;
    for (int i = 0; i < 64; i++) R[i] = i < 32 ? 0x00003c00u : 0u;     // scale 1.0, no output bias
    if (seq == 1) { f2_clr(); if (load2) f2_bias2(R); else f2_bias1(R); }
    else          { if (load2) f2_bias2(R); else f2_bias1(R); f2_clr(); }
    f2_mac(mac, A, W);
    if (seq >= 2) f2_swap();
    if (seq == 3) { f2_mac(mac, A, W); f2_swap(); }
    f2_store(st, O);
    int e32 = 0, e64 = 0, zero = 0, sent = 0;
    for (int i = 0; i < 1024; i++) {
        e32 += O[i] == 0x5000; e64 += O[i] == 0x5400; zero += O[i] == 0; sent += O[i] == 0xa5a5;
    }
    const int interesting = (e32 + e64) > 0 || (zero != 1024 && sent != 1024);
    if (interesting || (mac == F2_PLAIN && seq == 0)) {
        FP16_PRINT("FP16v2 mac %-6s st %-12s bias %s seq %d: 32.0 %4d  64.0 %4d  zero %4d  untouched %4d | %04x %04x %04x %04x",
                   f2_mac_name[mac], f2_st_name[st], load2 ? "mxmem2" : "mxmem", seq,
                   e32, e64, zero, sent, O[0], O[1], O[2], O[3]);
    }
    return e32 + e64;
}

__attribute__((noinline, optnone)) static void hmxfp16v2_run(uint8_t * base) {
    FP16_PRINT("FP16v2 integer self test before: %d/1024 wrong", hmxi_selftest(base + 49152));
    int n = 0, hit = 0, full = 0;
    for (int seq = 0; seq < 4; seq++)
        for (int load2 = 1; load2 >= 0; load2--)
            for (int mac = 0; mac < F2_NMAC; mac++)
                for (int st = 0; st < F2_NST; st++) {
                    const int ok = f2_case(base, mac, st, load2, seq);
                    n++; hit += ok > 0; full += ok == 1024;
                }
    FP16_PRINT("FP16v2 summary: %d cases, %d with any nonzero expected output, %d with all 1024", n, hit, full);
    FP16_PRINT("FP16v2 integer self test after: %d/1024 wrong", hmxi_selftest(base + 49152));
}

#endif
