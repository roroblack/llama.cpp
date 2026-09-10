// Integer HMX throughput on the device - the go/no-go number for writing a kernel at all.
//
// The HVX tiled matmul runs at ~170 GFLOP/s for every shape (profile, pp512). Codex's budget:
// at a hypothetical 1 TOP/s of sustained HMX MACs there is room for the packing and partial
// output work; at much less there is not. So measure it instead of assuming it.
//
// What is timed (core pcycles, core clock measured earlier at 1497.6 MHz):
//   P[nk]  1000 MAC packets, each spanning nk K-tiles through a larger Rt (verified to
//          accumulate: two tiles in one packet == two packets). Tiles resident in VTCM, no
//          clear between packets, one store at the end so the work has to complete.
//   S      1000 convert+store operations alone.
//   E      empty loop, the call overhead floor.
// Each 32x32 output tile x one K-tile is 32*32*32 MACs = 65,536 ops.
//
// PERF_PRINT(fmt, ...) must be defined by the includer.

#ifndef HMXPERF_H
#define HMXPERF_H

#include <string.h>

static void __attribute__((noinline)) pf_mac(unsigned char * a, signed char * w, unsigned rta, unsigned rtw) {
    asm volatile("{\n"
                 "  activation.ub = mxmem(%0, %2):deep\n"
                 "  weight.b      = mxmem(%1, %3)\n"
                 "}\n" :: "r"(a), "r"(w), "r"(rta), "r"(rtw) : "memory");
}
static void __attribute__((noinline)) pf_bias(unsigned * b) { asm volatile("bias = mxmem(%0)\n" :: "r"(b) : "memory"); }
static void __attribute__((noinline)) pf_clr(void)          { asm volatile("mxclracc\n" ::: "memory"); }
static void __attribute__((noinline)) pf_store(unsigned short * o) {
    asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n" :: "r"(o), "r"(0) : "memory");
}
static void __attribute__((noinline)) pf_nop(void) { asm volatile("" ::: "memory"); }

// base: 2 KB aligned VTCM, at least 256 KB
static void hmxperf_run(unsigned char * base) {
    unsigned char *  act  = base;                               // up to 32 K-tiles x 2048 = 64 KB
    signed char *    wgt  = (signed char *) (base + 131072);    // up to 32 K-tiles x 1024 = 32 KB
    unsigned short * out  = (unsigned short *) (base + 196608);
    unsigned *       bias = (unsigned *) (base + 204800);
    const int N = 1000;
    const double mhz = 1497.6;

    for (int i = 0; i < 65536; i++) act[i] = (unsigned char) (i * 7);
    for (int i = 0; i < 32768; i++) wgt[i] = (signed char) ((i * 5) - 64);
    for (int i = 0; i < 64; i++) bias[i] = 0x00003c00u;

    unsigned long long t0, t1;

    t0 = qurt_get_core_pcycles();
    for (int i = 0; i < N; i++) pf_nop();
    t1 = qurt_get_core_pcycles();
    unsigned long long e = t1 - t0;
    PERF_PRINT("PERF E empty call x%d: %llu cycles (%.1f/iter)", N, e, (double) e / N);

    static const int nks[] = { 1, 2, 4, 8, 16, 32 };
    for (unsigned j = 0; j < sizeof nks / sizeof nks[0]; j++) {
        const int nk = nks[j];
        pf_bias(bias); pf_clr();
        t0 = qurt_get_core_pcycles();
        for (int i = 0; i < N; i++) pf_mac(act, wgt, (unsigned) (nk * 2048 - 1), (unsigned) (nk * 1024 - 1));
        pf_store(out);
        volatile unsigned short sink = out[0]; (void) sink;
        t1 = qurt_get_core_pcycles();
        unsigned long long c = t1 - t0;
        double ops = 2.0 * 32 * 32 * 32 * nk * (double) N;
        double cyc_per_tile = (double) c / ((double) N * nk);
        double gops = ops / ((double) c / (mhz * 1e6)) / 1e9;
        PERF_PRINT("PERF P nk=%d x%d: %llu cycles, %.1f cycles per 32x32x32 tile-K, %.1f GOP/s",
                   nk, N, c, cyc_per_tile, gops);
    }

    pf_bias(bias); pf_clr();
    pf_mac(act, wgt, 1023, 1023);
    t0 = qurt_get_core_pcycles();
    for (int i = 0; i < N; i++) pf_store(out);
    t1 = qurt_get_core_pcycles();
    PERF_PRINT("PERF S convert+store x%d: %llu cycles (%.1f/store)", N, t1 - t0, (double) (t1 - t0) / N);
}

#endif
