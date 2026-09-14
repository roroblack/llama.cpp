// Op-path driver of the integer-HMX Q4_0 x F32 matmul (pieces and their provenance: hmx-int.h).
//
// Per chunk of up to mc activation rows:
//   A  the work queue quantises the rows into u16 tiles (+ one scale per row)
//   per group of up to nct column tiles:
//   W  the work queue converts the group's host-repacked Q4_0 tiles (wt / cv / dv / mid)
//   P  one producer job on the HMX queue (its thread holds the HMX lock for the batch) runs every
//      block MAC + store of the group's output tiles into per-consumer rings; the work queue's
//      threads (the caller is consumer 0) combine each K segment and write y straight into dst.
//      The producer is pushed first and popped only after the consumers return: popping first
//      would deadlock on a full ring.
// If the producer finds the HMX lock not held it raises `abort` instead of issuing HMX (which is
// what crashed the cDSP once); consumers stop waiting and the op fails loudly.
//
// Needs htp-ctx.h (context, queues) and hmx-int.h included first.

#ifndef HMX_INT_MM_H
#define HMX_INT_MM_H

#include <stdatomic.h>
#include <HAP_farf.h>

struct hmxi_seq { atomic_uint v; uint8_t pad[124]; };

struct hmxi_job {
    hmx_queue_t                hq;
    struct hmxi_layout         L;
    uint8_t *                  at;
    uint8_t *                  wt;
    HVX_Vector *               cv;
    HVX_Vector *               dv;
    HVX_Vector *               mid;
    const uint32_t *           rec;
    uint8_t *                  ring;
    HVX_Vector *               accw;
    float *                    sa;
    // A
    const float *              act;
    int                        act_stride, m0, mr, n_rt;
    // W
    const uint8_t *            weight;
    int                        ct0, nct;
    // P
    float *                    dst;
    int                        dst_stride, dst_cols;
    struct hmxi_seq            ready[HMXI_MAXW], freed[HMXI_MAXW];
    atomic_int                 abort;
    atomic_int                 why;      // HMXI_WHY_*: what raised abort
    int                        fi_lock;  // fault injection armed for this group (producer side)
    int                        fi_cancel;
    int                        fi_stall_ready, fi_stall_freed, fi_stall_worker, fi_stall_main;
};

#define HMXI_WHY_LOCK    1
#define HMXI_WHY_CANCEL  2
#define HMXI_WHY_TIMEOUT 3

// Codex q33a: a ring wait gives up after 2 s from its first check (checked every 256 spins, so the normal
// path never reads the clock); it raises abort so the other side stops too.
#define HMXI_RING_DEADLINE_US 2000000ull
static inline int hmxi_wait_tick(struct hmxi_job * j, unsigned * spins, unsigned long long * t0) {
    if (atomic_load_explicit(&j->abort, memory_order_relaxed)) return 1;
    if ((++*spins & 255) == 0) {
        const unsigned long long now = HAP_perf_get_time_us();
        if (*t0 == 0) {
            *t0 = now;
        } else if (now - *t0 > HMXI_RING_DEADLINE_US) {
            atomic_store(&j->why, HMXI_WHY_TIMEOUT);
            atomic_store(&j->abort, 1);
            return 1;
        }
    }
    hex_pause();
    return 0;
}

// Fault injection (Codex q31): fires at most once per DSP session, at the point it names, so a case
// that never reaches that point reports no FI_HIT instead of a false pass.
static int hmxi_fi_take(struct htp_context * ctx, int fi, int want, const char * stage) {
    if (fi != want || ctx->hmxi_fi_fired) return 0;
    ctx->hmxi_fi_fired = 1;
    FARF(ALWAYS, "FI_HIT int-hmx stage %s", stage);
    return 1;
}

// Codex q33a: every work-queue stage and the HMX pop have a 10 s deadline from submission. A timed-out
// stage is left outstanding (its workers may still run), so the session is poisoned with
// SESSION_POISONED and nothing - J, VTCM, the queues - may be reused; the host ends the process.
#define HMXI_STAGE_DEADLINE_US 10000000ull
static int hmxi_wq(struct htp_context * ctx, work_queue_func_t f, void * data, unsigned n, int not_submitted_rc) {
    const int r = work_queue_run_timed(ctx->work_queue, f, data, n, HMXI_STAGE_DEADLINE_US);
    if (r == WORK_QUEUE_OK) return 0;
    if (r == WORK_QUEUE_TIMEOUT) {
        atomic_store(&ctx->poisoned, HTP_STATUS_SESSION_POISONED);
        FARF(ERROR, "int-hmx: a work-queue stage passed its deadline; workers not confirmed stopped, session poisoned");
        return -12;
    }
    return not_submitted_rc;
}

static void hmxi_job_quant(unsigned int n, unsigned int i, void * data) {
    struct hmxi_job * j = (struct hmxi_job *) data;
    const int B = j->L.B;
    for (int p = (int) i; p < j->n_rt * 16; p += (int) n) {
        const int rt = p / 16, pr = p % 16, r0 = rt * 32 + 2 * pr;
        const float * x0 = r0 < j->mr ? j->act + (size_t) (j->m0 + r0) * j->act_stride : NULL;
        const float * x1 = r0 + 1 < j->mr ? j->act + (size_t) (j->m0 + r0 + 1) * j->act_stride : NULL;
        hmxi_quant_pair(x0, x1, B, &j->sa[r0], &j->sa[r0 + 1], j->at + (size_t) rt * B * 2048, pr);
    }
}

static void hmxi_job_cvt(unsigned int n, unsigned int i, void * data) {
    struct hmxi_job * j = (struct hmxi_job *) data;
    const int B = j->L.B;
    for (int c = (int) i; c < j->nct; c += (int) n)
        hmxi_cvt_coltile(j->weight + (size_t) (j->ct0 + c) * B * 576, B, j->wt + (size_t) c * B * 1024, j->cv + c * B, j->dv + c * B, j->mid + c);
}

// Codex q25: when a group has fewer column tiles than workers (K=12288 -> nct=2), splitting by column
// tile leaves workers idle. Split into (column tile, run of up to 32 K blocks) tasks instead; every
// task writes its own wt/cv/dv slots, and mid is summed afterwards in the same b order as before.
#define HMXI_CVT_KRUN 32
static void hmxi_job_cvt_k(unsigned int n, unsigned int i, void * data) {
    struct hmxi_job * j = (struct hmxi_job *) data;
    const int B = j->L.B, nkr = (B + HMXI_CVT_KRUN - 1) / HMXI_CVT_KRUN;
    for (int t = (int) i; t < j->nct * nkr; t += (int) n) {
        const int c = t / nkr, b0 = (t % nkr) * HMXI_CVT_KRUN;
        const int b1 = B - b0 < HMXI_CVT_KRUN ? B : b0 + HMXI_CVT_KRUN;
        const uint8_t * src = j->weight + (size_t) (j->ct0 + c) * B * 576;
        for (int b = b0; b < b1; b++)
            hmxi_cvt_tile(src + (size_t) b * 576, (HVX_Vector *) (j->wt + ((size_t) c * B + b) * 1024), j->cv + c * B + b, j->dv + c * B + b);
    }
}

// mid = 128 * sum_b d_b = 0.5 * sum_b dv_b, same order and arithmetic as hmxi_cvt_coltile
static void hmxi_mid_from_dv(const HVX_Vector * dv, int B, HVX_Vector * mid) {
    HVX_Vector acc = Q6_V_vzero();
    for (int b = 0; b < B; b++) acc = Q6_Vqf32_vadd_Vqf32Vsf(acc, dv[b]);
    *mid = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(acc), Q6_V_vsplat_R(0x3f000000)));
}

static void hmxi_job_produce(void * data) {
    struct hmxi_job * j = (struct hmxi_job *) data;
    if (!j->hq->hmx_locked) { atomic_store(&j->why, HMXI_WHY_LOCK); atomic_store(&j->abort, 1); return; }
    const int B = j->L.B, nc = j->L.nc, nseg = j->L.nseg, seg = j->L.seg;
    const int U = j->nct * j->n_rt;
    for (int u = 0; u < U; u++) {
        const int w = u % nc, jw = u / nc, c = u / j->n_rt, rt = u % j->n_rt;
        for (int sg = 0; sg < nseg; sg++) {
            const int s = jw * nseg + sg, slot = s % HMXI_DEPTH;
            if (atomic_load_explicit(&j->abort, memory_order_relaxed)) return;  // segment boundary
            {
                unsigned spins = 0; unsigned long long t0 = 0;
                while ((int) atomic_load_explicit(&j->freed[w].v, memory_order_acquire) + HMXI_DEPTH <= s) {
                    if (hmxi_wait_tick(j, &spins, &t0)) return;
                }
            }
            const int b0 = sg * seg, nb = B - b0 < seg ? B - b0 : seg;
            hmxi_blocks(j->at + ((size_t) rt * B + b0) * 2048, j->wt + ((size_t) c * B + b0) * 1024,
                        j->ring + ((size_t) w * HMXI_DEPTH + slot) * seg * 2048, j->rec, nb);
            if (j->fi_stall_ready) {  // fault injection: never publish; only an abort (consumers' deadline) ends it
                while (!atomic_load_explicit(&j->abort, memory_order_relaxed)) hex_pause();
                return;
            }
            atomic_store_explicit(&j->ready[w].v, (unsigned) (s + 1), memory_order_release);
            if (j->fi_cancel) {  // fault injection "cancel": stop right after the first published segment
                atomic_store(&j->why, HMXI_WHY_CANCEL);
                atomic_store(&j->abort, 1);
                return;
            }
        }
    }
}

static void hmxi_job_consume(unsigned int n, unsigned int i, void * data) {
    struct hmxi_job * j = (struct hmxi_job *) data;
    (void) n;
    const int B = j->L.B, nc = j->L.nc, nseg = j->L.nseg, seg = j->L.seg, w = (int) i;
    const int U = j->nct * j->n_rt;
    HVX_Vector * acc = j->accw + w * 32;
    if ((j->fi_stall_worker && i == 1) || (j->fi_stall_main && i == 0)) {
        // fault injection: a worker (stall_worker) or the calling thread itself (stall_main - job 0 runs on
        // it, so no DSP deadline can see it; only the host's batch deadline ends it) that never returns
        for (;;) hex_pause();
    }
    for (int u = w; u < U; u += nc) {
        const int jw = u / nc, c = u / j->n_rt, rt = u % j->n_rt;
        for (int sg = 0; sg < nseg; sg++) {
            const int s = jw * nseg + sg, slot = s % HMXI_DEPTH;
            {
                unsigned spins = 0; unsigned long long t0 = 0;
                while ((int) atomic_load_explicit(&j->ready[w].v, memory_order_acquire) <= s) {
                    if (hmxi_wait_tick(j, &spins, &t0)) return;
                }
            }
            // an abort can arrive after this segment was already published: stop at the boundary
            if (atomic_load_explicit(&j->abort, memory_order_relaxed)) return;
            const int b0 = sg * seg, nb = B - b0 < seg ? B - b0 : seg;
            const uint8_t * src = j->ring + ((size_t) w * HMXI_DEPTH + slot) * seg * 2048;
            hmxi_combine(src, j->cv + c * B + b0, j->dv + c * B + b0, acc, 0, nb, sg > 0);
            hmxi_combine(src, j->cv + c * B + b0, j->dv + c * B + b0, acc, 8, nb, sg > 0);
            if (j->fi_stall_freed && w == 0) {  // fault injection: never release; only the producer's deadline ends it
                while (!atomic_load_explicit(&j->abort, memory_order_relaxed)) hex_pause();
                return;
            }
            atomic_store_explicit(&j->freed[w].v, (unsigned) (s + 1), memory_order_release);
        }
        if (atomic_load_explicit(&j->abort, memory_order_relaxed)) return;  // no epilogue after an abort
        const int rows = j->mr - rt * 32 < 32 ? j->mr - rt * 32 : 32;
        const int col0 = (j->ct0 + c) * 32, cols = j->dst_cols - col0 < 32 ? j->dst_cols - col0 : 32;
        if (rows > 0 && cols > 0)
            hmxi_epilogue(acc, j->mid[c], j->sa + rt * 32, j->dst + (size_t) (j->m0 + rt * 32) * j->dst_stride + col0, j->dst_stride, rows, cols);
    }
}

// dst[m][dst_cols] (row stride dst_stride floats) = act[m][k] (row stride act_stride) x W^T,
// W = host-repacked Q4_0 with n (padded to 32) output rows. Returns 0 on success.
static int hmx_mm_q4int_2d_f32(struct htp_context * ctx, float * dst, int dst_stride, int dst_cols,
                               const float * act, int act_stride, const uint8_t * weight,
                               int m, int k, int n, int n_threads, int vtcm_size, int seg_cap, int fi) {
    if (!ctx->hmx_queue || !ctx->work_queue) return -2;
    size_t V = (size_t) vtcm_size;
    if (V == 0 || V > ctx->vtcm_size) V = ctx->vtcm_size;
    if (hmxi_fi_take(ctx, fi, HMXI_FI_VTCM_PLAN, "vtcm_plan")) V = 1;  // the planner must refuse a 1-byte budget
    int nc = n_threads;
    if (nc > (int) ctx->n_threads) nc = (int) ctx->n_threads;
    // One job per context: a function static would be shared by every session in the DSP process.
    if (!ctx->hmxi_job) {
        ctx->hmxi_job = memalign(128, sizeof(struct hmxi_job));
        if (!ctx->hmxi_job) return -6;
    }
    struct hmxi_job * const Jp = (struct hmxi_job *) ctx->hmxi_job;
#define J (*Jp)
    memset(&J, 0, sizeof(J));
    if (!hmxi_plan_seg(k, m, n, nc, V, seg_cap, &J.L)) return -3;
    // the layout must fit what we may use; "vtcm_short" pretends one byte less than the plan needs
    {
        size_t usable = V;
        if (hmxi_fi_take(ctx, fi, HMXI_FI_VTCM_SHORT, "vtcm_short")) usable = J.L.total - 1;
        if (J.L.total > usable) return -3;
    }

    uint8_t * vtcm = ctx->vtcm_base;
    J.hq   = ctx->hmx_queue;
    J.ring = vtcm + J.L.off_ring;
    J.accw = (HVX_Vector *) (vtcm + J.L.off_accw);
    J.rec  = (const uint32_t *) (vtcm + J.L.off_rec);
    J.sa   = (float *) (vtcm + J.L.off_sa);
    J.at   = vtcm + J.L.off_act;
    J.wt   = vtcm + J.L.off_wt;
    J.cv   = (HVX_Vector *) (vtcm + J.L.off_cv);
    J.dv   = (HVX_Vector *) (vtcm + J.L.off_dv);
    J.mid  = (HVX_Vector *) (vtcm + J.L.off_mid);
    J.act = act; J.act_stride = act_stride; J.weight = weight;
    J.dst = dst; J.dst_stride = dst_stride; J.dst_cols = dst_cols;
    hmxi_init_record((uint32_t *) J.rec);

    const int n_ct_all = (n + 31) / 32;
    unsigned long long tA = 0, tW = 0, tP = 0, t0;
    for (int m0 = 0; m0 < m; m0 += J.L.mc) {
        J.m0 = m0; J.mr = m - m0 < J.L.mc ? m - m0 : J.L.mc; J.n_rt = (J.mr + 31) / 32;
        t0 = HAP_perf_get_pcycles();
        // fault injection replaces a submission by the "not submitted" result without running it
        {
            const int r = hmxi_fi_take(ctx, fi, HMXI_FI_QUANT, "quant") ? -7 : hmxi_wq(ctx, hmxi_job_quant, &J, ctx->n_threads, -7);
            if (r) return r;
        }
        tA += HAP_perf_get_pcycles() - t0;
        for (int ct0 = 0; ct0 < n_ct_all; ct0 += J.L.nct) {
            J.ct0 = ct0; J.nct = n_ct_all - ct0 < J.L.nct ? n_ct_all - ct0 : J.L.nct;
            t0 = HAP_perf_get_pcycles();
            if (J.nct < (int) ctx->n_threads) {
                const int r = hmxi_fi_take(ctx, fi, HMXI_FI_CVT_K, "cvt_k") ? -8 : hmxi_wq(ctx, hmxi_job_cvt_k, &J, ctx->n_threads, -8);
                if (r) return r;
                for (int c = 0; c < J.nct; c++) hmxi_mid_from_dv(J.dv + c * J.L.B, J.L.B, J.mid + c);
            } else {
                const int r = hmxi_fi_take(ctx, fi, HMXI_FI_CVT, "cvt") ? -8 : hmxi_wq(ctx, hmxi_job_cvt, &J, ctx->n_threads, -8);
                if (r) return r;
            }
            tW += HAP_perf_get_pcycles() - t0;
            for (int w = 0; w < HMXI_MAXW; w++) { atomic_store(&J.ready[w].v, 0); atomic_store(&J.freed[w].v, 0); }
            atomic_store(&J.abort, 0);
            atomic_store(&J.why, 0);
            J.fi_cancel      = hmxi_fi_take(ctx, fi, HMXI_FI_CANCEL, "cancel");
            J.fi_stall_ready = hmxi_fi_take(ctx, fi, HMXI_FI_STALL_READY, "stall_ready");
            J.fi_stall_freed = hmxi_fi_take(ctx, fi, HMXI_FI_STALL_FREED, "stall_freed");
            J.fi_stall_worker = hmxi_fi_take(ctx, fi, HMXI_FI_STALL_WORKER, "stall_worker");
            J.fi_stall_main   = hmxi_fi_take(ctx, fi, HMXI_FI_STALL_MAIN, "stall_main");
            if (hmxi_fi_take(ctx, fi, HMXI_FI_STALL_RSP, "stall_rsp")) {
                ctx->fi_skip_rsp = 1;   // this op completes, but its batch is never answered
            }
            if (hmxi_fi_take(ctx, fi, HMXI_FI_STALL_HMX_DONE, "stall_hmx_done")) {
                ctx->hmx_queue->fi_stall_done = 1;   // the producer descriptor will never be marked done
            }
            if (hmxi_fi_take(ctx, fi, HMXI_FI_LOCK, "lock")) {
                // a real lock failure (Codex q32): the queue thread releases the lock (SUSPEND), and its
                // next lock attempt - for this producer - returns an error without taking it
                hmx_queue_suspend(ctx->hmx_queue);
                ctx->hmx_queue->fi_lock_fail = 1;
            }
            t0 = HAP_perf_get_pcycles();
            if (!hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmxi_job_produce, &J))) return -4;
            // If the consumers cannot be started the producer would fill the rings and spin: raise
            // abort so it returns, and still pop it before anything reuses this VTCM.
            const int crc = hmxi_fi_take(ctx, fi, HMXI_FI_CONSUME, "consume") ? -9
                          : hmxi_wq(ctx, hmxi_job_consume, &J, (unsigned) J.L.nc, -9);
            if (crc != 0) atomic_store(&J.abort, 1);
            if (!hmx_queue_pop_timed(ctx->hmx_queue, HMXI_STAGE_DEADLINE_US)) {
                atomic_store(&ctx->poisoned, HTP_STATUS_SESSION_POISONED);
                FARF(ERROR, "int-hmx: the producer descriptor passed its deadline; not confirmed done, session poisoned");
                return -12;
            }
            if (crc == -12) return -12;
            if (crc != 0) {
                FARF(ERROR, "int-hmx: consumer launch failed; producer aborted and drained");
                return -9;
            }
            tP += HAP_perf_get_pcycles() - t0;
            if (atomic_load(&J.abort)) {
                if (atomic_load(&J.why) == HMXI_WHY_CANCEL) {
                    FARF(ERROR, "int-hmx: cancelled; producer and consumers stopped at a segment boundary");
                    return -10;
                }
                if (atomic_load(&J.why) == HMXI_WHY_TIMEOUT) {
                    // every consumer has returned (work_queue_run) and the producer was popped: drained.
                    // The session stays poisoned; later batches are refused before touching J or VTCM.
                    atomic_store(&ctx->poisoned, HTP_STATUS_TIMEOUT_DRAINED);
                    FARF(ERROR, "int-hmx: ring wait passed its deadline; producer and consumers stopped, session poisoned");
                    return -11;
                }
                FARF(ERROR, "int-hmx: the HMX queue thread does not hold the HMX lock; not issuing HMX");
                return -5;
            }
        }
    }
    FARF(HIGH, "int-hmx m %d k %d n %d: mc %d nct %d nseg %d nc %d | quant %llu cvt %llu pipe %llu kcycles",
         m, k, n, J.L.mc, J.L.nct, J.L.nseg, J.L.nc, tA / 1000, tW / 1000, tP / 1000);
    return 0;
#undef J
}

#endif /* HMX_INT_MM_H */
