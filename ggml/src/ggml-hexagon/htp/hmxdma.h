// DDR <-> VTCM DMA bandwidth and coherence on the device, through the backend's DMA queue - v3.
//
// v2 (distinct addresses over 32 MB): DDR->VTCM 55.65 GB/s at 64 KB (0/64 bad), 49.93 at 16 KB but
// with 55/256 chunks NOT matching their source; VTCM->DDR ~42 GB/s (never verified). Leading
// explanation: the CPU filled the buffer through its data cache, the queue's DMA bypasses the cache,
// so it read DDR before the dirty lines were written back. A real integer kernel meets exactly this
// when HVX writes activations to DDR and DMA reads them, so it is settled here rather than assumed:
//   A  16 KB stream WITHOUT a cache clean   - should reproduce the mismatches
//   B  clean the whole buffer (qurt_mem_cache_clean FLUSH), then 16 KB and 64 KB streams
//      - mismatches should go to zero; these are the bandwidth numbers to use
//   C  VTCM->DDR, then INVALIDATE the CPU's cached copy and verify what landed in DDR
//   D  a single 1 MB 1-D transfer after a clean
//
// DMA_PRINT(fmt, ...) must be defined by the includer.

#ifndef HMXDMA_H
#define HMXDMA_H

#include <stdlib.h>
#include <string.h>

static inline uint8_t hd_pat(size_t i, unsigned salt) { return (uint8_t) (i * 131u + (i >> 16) + salt); }

static void hmxdma_stream(dma_queue * q, uint8_t * vtcm, size_t vtcm_bytes, uint8_t * ddr, size_t ddr_bytes,
                          size_t chunk, int to_vtcm, const char * tag, unsigned salt) {
    const double   mhz   = 1497.6;
    const unsigned n     = (unsigned) (ddr_bytes / chunk);
    const unsigned slots = (unsigned) (vtcm_bytes / chunk);
    const unsigned batch = 16;

    unsigned long long t0 = qurt_get_core_pcycles();
    unsigned done = 0;
    while (done < n) {
        unsigned b = n - done < batch ? n - done : batch;
        for (unsigned i = 0; i < b; i++) {
            unsigned  idx = done + i;
            uint8_t * v   = vtcm + (size_t) (idx % slots) * chunk;
            uint8_t * d   = ddr + (size_t) idx * chunk;
            if (to_vtcm) dma_queue_push_ddr_to_vtcm(q, dma_make_ptr(v, d), chunk, chunk, 1);
            else         dma_queue_push_vtcm_to_ddr(q, dma_make_ptr(d, v), chunk, chunk, 1);
        }
        for (unsigned i = 0; i < b; i++) dma_queue_pop(q);
        done += b;
    }
    unsigned long long t1 = qurt_get_core_pcycles();

    unsigned checked = 0, bad = 0;
    if (to_vtcm) {
        // the chunks still resident in VTCM, against the pattern the CPU wrote
        unsigned first = n > slots ? n - slots : 0;
        for (unsigned idx = first; idx < n; idx++) {
            uint8_t * v = vtcm + (size_t) (idx % slots) * chunk;
            size_t    o = (size_t) idx * chunk;
            checked++;
            if (v[0] != hd_pat(o, salt) || v[chunk / 2] != hd_pat(o + chunk / 2, salt) ||
                v[chunk - 1] != hd_pat(o + chunk - 1, salt)) bad++;
        }
    } else {
        // DMA wrote DDR behind the cache: drop the CPU's copy first, then read what really landed
        qurt_mem_cache_clean((qurt_addr_t) (uintptr_t) ddr, (qurt_size_t) ddr_bytes, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
        for (unsigned idx = 0; idx < n; idx++) {
            uint8_t * d = ddr + (size_t) idx * chunk;
            uint8_t * v = vtcm + (size_t) (idx % slots) * chunk;
            checked++;
            if (d[0] != v[0] || d[chunk - 1] != v[chunk - 1]) bad++;
        }
    }
    double sec = (double) (t1 - t0) / (mhz * 1e6);
    double gbs = (double) n * (double) chunk / sec / 1e9;
    DMA_PRINT("DMA %s %s chunk=%u n=%u: %.2f GB/s  landed %u/%u bad",
              to_vtcm ? "ddr->vtcm" : "vtcm->ddr", tag, (unsigned) chunk, n, gbs, bad, checked);
}

static void hmxdma_run(dma_queue * q, uint8_t * vtcm, size_t vtcm_bytes) {
    size_t    ddr_bytes = (size_t) 32 << 20;
    uint8_t * ddr       = NULL;
    while (ddr_bytes >= ((size_t) 4 << 20) && !(ddr = (uint8_t *) memalign(4096, ddr_bytes))) ddr_bytes >>= 1;
    if (!ddr) { DMA_PRINT("DMA probe: could not allocate even 4 MB of DDR"); return; }
    DMA_PRINT("DMA probe v3: DDR buffer %u MB, VTCM window %u MB", (unsigned) (ddr_bytes >> 20), (unsigned) (vtcm_bytes >> 20));

    // A: no clean
    for (size_t i = 0; i < ddr_bytes; i++) ddr[i] = hd_pat(i, 7u);
    hmxdma_stream(q, vtcm, vtcm_bytes, ddr, ddr_bytes, 16 * 1024, 1, "no-clean", 7u);

    // B: clean, then stream
    for (size_t i = 0; i < ddr_bytes; i++) ddr[i] = hd_pat(i, 11u);
    int rc = qurt_mem_cache_clean((qurt_addr_t) (uintptr_t) ddr, (qurt_size_t) ddr_bytes, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    DMA_PRINT("DMA clean FLUSH rc=%d", rc);
    hmxdma_stream(q, vtcm, vtcm_bytes, ddr, ddr_bytes, 16 * 1024, 1, "cleaned ", 11u);
    hmxdma_stream(q, vtcm, vtcm_bytes, ddr, ddr_bytes, 64 * 1024, 1, "cleaned ", 11u);

    // C: VTCM -> DDR, verified after invalidating the CPU's cached copy
    for (size_t i = 0; i < vtcm_bytes; i++) vtcm[i] = (uint8_t) (i * 29u + 3u);
    hmxdma_stream(q, vtcm, vtcm_bytes, ddr, ddr_bytes, 16 * 1024, 0, "verified", 0u);
    hmxdma_stream(q, vtcm, vtcm_bytes, ddr, ddr_bytes, 64 * 1024, 0, "verified", 0u);

    // D: single 1 MB 1-D transfer after a clean
    {
        size_t big = (size_t) 1 << 20;
        for (size_t i = 0; i < big; i++) ddr[i] = hd_pat(i, 5u);
        qurt_mem_cache_clean((qurt_addr_t) (uintptr_t) ddr, (qurt_size_t) big, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        memset(vtcm, 0, big);
        dma_queue_push_ddr_to_vtcm(q, dma_make_ptr(vtcm, ddr), big, big, 1);
        dma_queue_pop(q);
        unsigned bad_pages = 0;
        for (size_t i = 0; i < big; i += 4096) if (vtcm[i] != hd_pat(i, 5u) || vtcm[i + 4095] != hd_pat(i + 4095, 5u)) bad_pages++;
        DMA_PRINT("DMA single 1 MB 1-D after clean: %u/256 pages bad", bad_pages);
    }
    free(ddr);
}

#endif
