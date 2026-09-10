// VTCM layout of the integer-HMX Q4_0 matmul (hmx-int.h). Plain C with no HVX, so the host includes
// it (through matmul-ops.h) to decide eligibility with the same numbers the DSP will use.

#ifndef HMX_INT_PLAN_H
#define HMX_INT_PLAN_H

#include <stddef.h>

#ifndef HMXI_SEG
#define HMXI_SEG   64      // blocks per K segment (one ring slot = HMXI_SEG * 2 KB); a test may lower it
#endif
#define HMXI_DEPTH 2       // ring slots per consumer
#define HMXI_MAXW  8       // max consumers
#define HMXI_MAXMC 256     // max activation rows per chunk

struct hmxi_layout {
    int    B, seg, nseg, nc, mc, nct;
    size_t off_ring, off_accw, off_rec, off_sa, off_act, off_wt, off_cv, off_dv, off_mid, total;
};

// k = K (multiple of 32), m = activation rows, n = output columns, nc = consumers, vtcm = bytes.
// Returns 0 when not even one 32-row chunk and one column tile fit.
static inline int hmxi_plan(int k, int m, int n, int nc, size_t vtcm, struct hmxi_layout * L) {
    if (k <= 0 || (k % 32) != 0 || m <= 0 || n <= 0 || nc < 1) return 0;
    if (nc > HMXI_MAXW) nc = HMXI_MAXW;
    L->B    = k / 32;
    L->seg  = L->B < HMXI_SEG ? L->B : HMXI_SEG;
    L->nseg = (L->B + L->seg - 1) / L->seg;
    L->nc   = nc;
    // Order keeps every tile area 2 KB aligned from an aligned base (ring slots and activation tiles
    // are read by HMX as 2 KB tiles, weight tiles as 1 KB): ring, accumulators, activations, weights,
    // then the 128-byte vectors, the record and the row scales. v1 put the record and the scales
    // first, activations landed at 38,144 (128- but not 2 KB-aligned) and the simulator gave
    // 420/2800 right.
    const size_t tail   = 256 + HMXI_MAXMC * 4;                                // record + row scales
    size_t off = 0;
    L->off_ring = off; off += (size_t) nc * HMXI_DEPTH * L->seg * 2048;
    L->off_accw = off; off += (size_t) nc * 4096;
    const size_t fixed  = off + tail;
    const size_t per_ct = (size_t) L->B * 1024 + (size_t) L->B * 256 + 128;   // wt + cv + dv + mid
    const int    n_ct_all = (n + 31) / 32;
    int mc = (m + 31) / 32 * 32;
    if (mc > HMXI_MAXMC) mc = HMXI_MAXMC;
    for (; mc >= 32; mc -= 32) {
        const size_t act = (size_t) (mc / 32) * L->B * 2048;
        if (fixed + act + per_ct <= vtcm) break;
    }
    if (mc < 32) return 0;
    const size_t act  = (size_t) (mc / 32) * L->B * 2048;
    const size_t left = vtcm - fixed - act;
    int nct = (int) (left / per_ct);
    if (nct > n_ct_all) nct = n_ct_all;
    if (nct > 32) nct = 32;
    L->mc  = mc;
    L->nct = nct;
    L->off_act = off; off += act;
    L->off_wt  = off; off += (size_t) nct * L->B * 1024;
    L->off_cv  = off; off += (size_t) nct * L->B * 128;
    L->off_dv  = off; off += (size_t) nct * L->B * 128;
    L->off_mid = off; off += (size_t) nct * 128;
    L->off_rec = off; off += 256;
    L->off_sa  = off; off += HMXI_MAXMC * 4;
    L->total = off;
    return off <= vtcm;
}

#endif /* HMX_INT_PLAN_H */
