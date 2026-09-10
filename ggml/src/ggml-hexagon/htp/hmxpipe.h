// Codex step 3: producer / consumer on the device, exact-QT path.
//
// v2: the calling (RPC) thread takes the HMX lock itself - exactly as the fp16 self test does - and is
// the single producer, as job 0 of the work queue; workers 1..n-1 are the consumers. v1 used the HMX
// queue thread and its reference path ran HMX with no lock held: the DSP faulted on the first
// `bias = mxmem` (precise exception 0x1801, PC in q4g_bias). main.c already notes the queue thread
// cannot take the lock this early in session start.
// The producer: for each output tile
// (32 rows x 32 columns, K=1536) it runs all 48 block MACs + stores into a slot of the ring that
// belongs to the worker owning that tile. Tiles go round-robin to the n HVX workers of the work queue;
// each worker runs the register-resident combine (two 16-row passes, vsxt path) and the epilogue
// y = s_a * (acc + 128 * sum_b d_b) into DDR, then frees the slot. Per-worker SPSC rings, publication
// with release/acquire sequence counters on separate cache lines, polling with hex_pause (Codex q20).
//
// Inputs are resident (packed act panel, NWT weight tiles cycled over T tiles), so this measures the
// pipeline, not packing or DMA. Checks: one run with the ring poisoned after every consume, every
// output tile bit-compared against a single-thread reference built from the same functions; the
// reference tile against the int64/double reconstruction.
//
// Q4G_PRINT(fmt, ...) must be defined by the includer; hmxcomb.h must be included first.

#ifndef HMXPIPE_H
#define HMXPIPE_H

#include <stdatomic.h>
#include <stdlib.h>
#include <HAP_compute_res.h>
#include "work-queue.h"

#define HP_NWT   4
#define HP_T     64
#define HP_SLOT  (HC_NB * 2048)
#define HP_MAXW  8
#define HP_MAXD  3

struct hp_seq { atomic_uint v; unsigned char pad[124]; };

struct hp_ctx {
    unsigned char *    act;
    signed char *      wgt;                  // HP_NWT x HC_NB x 1024
    const HVX_Vector * cv;                   // HP_NWT x HC_NB
    const HVX_Vector * dv;                   // HP_NWT x HC_NB
    const HVX_Vector * mid;                  // HP_NWT: 128 * sum_b d_b per column
    unsigned *         rec;
    unsigned char *    ring;                 // nw x depth x HP_SLOT
    HVX_Vector *       accw;                 // nw x 32 vectors
    HVX_Vector *       out;                  // HP_T x 32 vectors, DDR
    const float *      sa;
    int                nw, depth, poison;
    struct hp_seq      ready[HP_MAXW], freed[HP_MAXW];
    unsigned long long prod_wait, prod_busy, cons_wait[HP_MAXW], cons_busy[HP_MAXW];
};

__attribute__((noinline)) static void hp_epilogue(const HVX_Vector * acc, HVX_Vector mid, const float * sa, HVX_Vector * out) {
    for (int m = 0; m < 32; m++) {
        union { float f; int i; } s = { sa[m] };
        HVX_Vector t = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(acc[m], mid));
        out[m] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(t, Q6_V_vsplat_R(s.i)));
    }
}

static void hp_producer(void * data) {
    struct hp_ctx * c = (struct hp_ctx *) data;
    unsigned long long wait = 0, t0 = qurt_get_core_pcycles();
    for (int t = 0; t < HP_T; t++) {
        const int w = t % c->nw, j = t / c->nw, slot = j % c->depth;
        unsigned long long w0 = qurt_get_core_pcycles();
        while ((int) atomic_load_explicit(&c->freed[w].v, memory_order_acquire) + c->depth < j + 1) hex_pause();
        wait += qurt_get_core_pcycles() - w0;
        unsigned char *     dst = c->ring + (size_t) (w * c->depth + slot) * HP_SLOT;
        const signed char * wt  = c->wgt + (size_t) (t % HP_NWT) * HC_NB * 1024;
        for (int b = 0; b < HC_NB; b++) {
            q4g_bias(c->rec); q4g_clr();
            q4g_mac(c->act + b * 2048, (signed char *) wt + b * 1024);
            q4g_store((unsigned short *) (dst + b * 2048));
        }
        atomic_store_explicit(&c->ready[w].v, (unsigned) (j + 1), memory_order_release);
    }
    c->prod_busy = qurt_get_core_pcycles() - t0 - wait;
    c->prod_wait = wait;
}

static void hp_consumer(unsigned int i, void * data) {
    struct hp_ctx * c = (struct hp_ctx *) data;
    unsigned long long wait = 0, t0 = qurt_get_core_pcycles();
    HVX_Vector * acc = c->accw + i * 32;
    for (int t = (int) i; t < HP_T; t += c->nw) {
        const int j = t / c->nw, slot = j % c->depth, wt = t % HP_NWT;
        unsigned long long w0 = qurt_get_core_pcycles();
        while ((int) atomic_load_explicit(&c->ready[i].v, memory_order_acquire) < j + 1) hex_pause();
        wait += qurt_get_core_pcycles() - w0;
        unsigned char * src = c->ring + (size_t) (i * c->depth + slot) * HP_SLOT;
        hc_combine_reg2(src, c->cv + wt * HC_NB, c->dv + wt * HC_NB, acc, 0, HC_NB);
        hc_combine_reg2(src, c->cv + wt * HC_NB, c->dv + wt * HC_NB, acc, 8, HC_NB);
        if (c->poison) memset(src, 0xa5, HP_SLOT);
        atomic_store_explicit(&c->freed[i].v, (unsigned) (j + 1), memory_order_release);
        hp_epilogue(acc, c->mid[wt], c->sa, c->out + t * 32);
    }
    c->cons_busy[i] = qurt_get_core_pcycles() - t0 - wait;
    c->cons_wait[i] = wait;
}

// job 0 = producer on the calling thread (holds the HMX lock), jobs 1..nw = consumers
static void hp_role(unsigned int n, unsigned int i, void * data) {
    (void) n;
    if (i == 0) hp_producer(data); else hp_consumer(i - 1, data);
}

// one pipelined run; returns wall cycles on the calling thread
static unsigned long long hp_run_once(struct hp_ctx * c, work_queue_t wq) {
    for (int w = 0; w < HP_MAXW; w++) {
        atomic_store(&c->ready[w].v, 0); atomic_store(&c->freed[w].v, 0);
        c->cons_wait[w] = c->cons_busy[w] = 0;
    }
    unsigned long long t0 = qurt_get_core_pcycles();
    work_queue_run(wq, hp_role, c, (unsigned) c->nw + 1);
    return qurt_get_core_pcycles() - t0;
}

__attribute__((noinline, optnone)) static void hmxpipe_run(unsigned rctx, work_queue_t wq, unsigned n_threads, unsigned char * vtcm) {
    if (!wq || n_threads < 2) { Q4G_PRINT("PIPE skipped: work_queue %p n_threads %u", (void *) wq, n_threads); return; }
    const int lrc = HAP_compute_res_hmx_lock(rctx);
    if (lrc != 0) { Q4G_PRINT("PIPE skipped: HMX lock rc %d", lrc); return; }
    enum { K = HC_K, NB = HC_NB };
    const double mhz = 1497.6;

    unsigned char * act  = vtcm;                                   // 96 KB
    signed char *   wgt  = (signed char *) (vtcm + (96u << 10));   // 4 x 48 KB
    HVX_Vector *    cv   = (HVX_Vector *) (vtcm + (288u << 10));   // 4 x 48 x 128 = 24 KB
    HVX_Vector *    dv   = (HVX_Vector *) (vtcm + (312u << 10));   // 24 KB
    HVX_Vector *    mid  = (HVX_Vector *) (vtcm + (336u << 10));   // 4 x 128
    unsigned *      rec  = (unsigned *) (vtcm + (340u << 10));
    HVX_Vector *    accw = (HVX_Vector *) (vtcm + (344u << 10));   // 8 x 4 KB
    unsigned char * ring = vtcm + (384u << 10);                    // up to 8 x 3 x 96 KB = 2.25 MB
    unsigned char * sts  = vtcm + (3u << 20);                      // reference stores, 96 KB
    HVX_Vector *    racc = (HVX_Vector *) (vtcm + (3u << 20) + (96u << 10));

    static float     A[32][K];
    static q4g_block W[HP_NWT][32][NB];
    static int       qa[32][K];
    static float     sa[32];
    static struct hp_ctx c;
    HVX_Vector * out  = (HVX_Vector *) memalign(128, (size_t) HP_T * 32 * 128);
    HVX_Vector * rout = (HVX_Vector *) memalign(128, (size_t) HP_NWT * 32 * 128);
    if (!out || !rout) { Q4G_PRINT("PIPE: out of DDR"); free(out); free(rout); HAP_compute_res_hmx_unlock(rctx); return; }

    // data
    for (int i = 0; i < 64; i++) rec[i] = i < 32 ? 0x00004000u : 0u;
    q4g_lcg = 31337u;
    for (int m = 0; m < 32; m++) for (int k = 0; k < K; k++) {
        float v = q4g_frand(-1.0f, 1.0f);
        if (((m * 131 + k * 7) % 97) == 0) v *= 8.0f;
        A[m][k] = v;
    }
    memset(act, 0, NB * 2048);
    for (int m = 0; m < 32; m++) {
        float mx = 0.0f;
        for (int k = 0; k < K; k++) { float t = A[m][k] < 0 ? -A[m][k] : A[m][k]; if (t > mx) mx = t; }
        sa[m] = mx > 0 ? mx / 32767.0f : 1.0f;
        for (int k = 0; k < K; k++) {
            float t = A[m][k] / sa[m];
            int q = (int) (t >= 0 ? t + 0.5f : t - 0.5f);
            if (q > 32767) q = 32767;
            if (q < -32767) q = -32767;
            qa[m][k] = q;
            unsigned u = (unsigned) (q + 32768);
            unsigned char * p = act + (k >> 5) * 2048 + q4g_ia(m, k & 31);
            p[0] = (unsigned char) (u & 0xff); p[1] = (unsigned char) (u >> 8);
        }
    }
    for (int x = 0; x < HP_NWT; x++) {
        float * md = (float *) &mid[x];
        for (int n = 0; n < 32; n++) {
            md[n] = 0.0f;
            for (int b = 0; b < NB; b++) {
                float d = q4g_frand(0.005f, 0.05f) * ((n + b + x) % 3 == 0 ? -1.0f : 1.0f);
                W[x][n][b].d = q4g_f2h(d);
                md[n] += 128.0f * q4g_h2f(W[x][n][b].d);
                for (int j = 0; j < 16; j++) {
                    int lo = (int) q4g_frand(0.0f, 15.999f), hi = (int) q4g_frand(0.0f, 15.999f);
                    W[x][n][b].qs[j] = (unsigned char) (lo | (hi << 4));
                }
            }
        }
        for (int b = 0; b < NB; b++) {
            signed char *    t   = wgt + ((size_t) x * NB + b) * 1024;
            unsigned short * c16 = (unsigned short *) &cv[x * NB + b];
            float *          df  = (float *) &dv[x * NB + b];
            for (int n = 0; n < 32; n++) {
                int sw = 0;
                for (int k = 0; k < 32; k++) { int w = q4g_nib(&W[x][n][b], k) - 8; t[q4g_iw(k, n)] = (signed char) w; sw += w; }
                c16[2 * n] = c16[2 * n + 1] = (unsigned short) ((128 * sw) & 0xffff);
                df[n] = 256.0f * q4g_h2f(W[x][n][b].d);
            }
        }
    }

    // single-thread reference per weight tile, from the same functions; tile 0 against int64/double
    for (int x = 0; x < HP_NWT; x++) {
        hc_hmx_n(act, wgt + (size_t) x * NB * 1024, sts, rec, NB);
        hc_combine_reg2(sts, cv + x * NB, dv + x * NB, racc, 0, NB);
        hc_combine_reg2(sts, cv + x * NB, dv + x * NB, racc, 8, NB);
        hp_epilogue(racc, mid[x], sa, rout + x * 32);
    }
    {
        int tight = 0; double maxrel = 0;
        for (int m = 0; m < 32; m++) for (int n = 0; n < 32; n++) {
            double ys = 0, yabs = 0;
            for (int b = 0; b < NB; b++) {
                const double d = (double) q4g_h2f(W[0][n][b].d);
                long long su = 0; int sw = 0;
                for (int k = 0; k < 32; k++) { int w = q4g_nib(&W[0][n][b], k) - 8; su += (long long) (qa[m][b * 32 + k] + 32768) * w; sw += w; }
                long long Q = su / 256; if ((su % 256) && su < 0) Q--;
                const double term = 256.0 * d * (double) (Q - 128LL * sw) + 128.0 * d;
                ys += term; yabs += term < 0 ? -term : term;
            }
            ys *= sa[m]; yabs *= sa[m];
            double r = (double) ((const float *) &rout[m])[n] - ys; if (r < 0) r = -r;
            r = yabs > 0 ? r / yabs : r;
            if (r > maxrel) maxrel = r;
            if (r <= 4e-6) tight++;
        }
        Q4G_PRINT("PIPE reference tile vs int64/double reconstruction: %d/1024 within 4e-6 of sum|terms| (max %.2g)", tight, maxrel);
    }

    c.act = act; c.wgt = wgt; c.cv = cv; c.dv = dv; c.mid = mid; c.rec = rec; c.ring = ring; c.accw = accw;
    c.out = out; c.sa = sa;

    work_queue_wakeup(wq);

    // correctness: all workers, depth 2, ring poisoned after every consume
    unsigned nwmax = n_threads - 1 < HP_MAXW ? n_threads - 1 : HP_MAXW;
    c.nw = (int) nwmax; c.depth = 2; c.poison = 1;
    memset(out, 0, (size_t) HP_T * 32 * 128);
    hp_run_once(&c, wq);
    int same = 0;
    for (int t = 0; t < HP_T; t++) {
        const unsigned * a = (const unsigned *) (out + t * 32), * r = (const unsigned *) (rout + (t % HP_NWT) * 32);
        int ok = 1;
        for (int e = 0; e < 1024; e++) if (a[e] != r[e]) { ok = 0; break; }
        same += ok;
    }
    Q4G_PRINT("PIPE check: consumers=%u depth=2, ring poisoned after each consume: %d/%d output tiles bit-identical to the single-thread reference",
              nwmax, same, HP_T);

    // timing sweep, no poisoning, best of 3
    c.poison = 0;
    for (unsigned nw = 1; nw <= nwmax; nw++) for (int depth = 2; depth <= HP_MAXD; depth++) {
        c.nw = (int) nw; c.depth = depth;
        unsigned long long best = ~0ull, pw = 0, pb = 0, cw = 0, cb = 0;
        for (int r = 0; r < 3; r++) {
            unsigned long long tt = hp_run_once(&c, wq);
            if (tt < best) {
                best = tt; pw = c.prod_wait; pb = c.prod_busy; cw = 0; cb = 0;
                for (unsigned w = 0; w < nw; w++) { cw += c.cons_wait[w]; cb += c.cons_busy[w]; }
            }
        }
        const double blocks = (double) HP_T * NB;
        Q4G_PRINT("PIPE consumers=%u depth=%d: %.1f cyc/block wall, %.1f GOP/s | producer busy %.1f wait %.1f cyc/block | consumers busy %.1f wait %.1f cyc/block/worker",
                  nw, depth, (double) best / blocks, 2.0 * 32 * 32 * K * HP_T / ((double) best / (mhz * 1e6)) / 1e9,
                  (double) pb / blocks, (double) pw / blocks, (double) cb / blocks, (double) cw / blocks);
    }

    work_queue_suspend(wq);
    HAP_compute_res_hmx_unlock(rctx);
    free(out); free(rout);
}

#endif
