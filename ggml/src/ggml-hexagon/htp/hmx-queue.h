#ifndef HMX_QUEUE_H
#define HMX_QUEUE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdatomic.h>

#include <hexagon_types.h>
#include <qurt_thread.h>
#include <qurt_futex.h>
#include <HAP_farf.h>
#include <HAP_perf.h>

#include "hex-utils.h"
#include "hex-profile.h"

#ifdef __cplusplus
extern "C" {
#endif

#if __HVX_ARCH__ > 79
#define HMX_QUEUE_POLL_COUNT 2000
#else
#define HMX_QUEUE_POLL_COUNT 1
#endif

typedef void (*hmx_queue_func)(void *);

// Dummy funcs used as signals
enum hmx_queue_signal {
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_WAKEUP,
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

struct hmx_queue_desc {
    hmx_queue_func   func;
    void *           data;
    atomic_uint      done;
    uint64_t         t_us;      // push time (HAP_perf_get_time_us), for hmx_queue_pop_timed
};

struct hmx_queue_s {
    struct hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned int     idx_pop;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    void *           stack;
    uint32_t         hap_rctx;
    bool             hmx_locked;
    volatile int     fi_lock_fail; // fault injection (test only): the next lock attempt fails, one shot
    volatile int     fi_stall_done; // fault injection (test only): the next descriptor is never marked done
    struct htp_thread_trace * trace;
    bool             external_mem; // memory owned externally
};

typedef struct hmx_queue_s * hmx_queue_t;

size_t      hmx_queue_sizeof(size_t capacity, uint32_t stack_size);
size_t      hmx_queue_alignof(void);
hmx_queue_t hmx_queue_init(void * ptr, size_t capacity, uint32_t stack_size, uint32_t hap_rctx, struct htp_thread_trace * trace);
void        hmx_queue_free(hmx_queue_t q);

static inline struct hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, void * data) {
    struct hmx_queue_desc d = { func, data };
    return d;
}

static inline bool hmx_queue_push(hmx_queue_t q, struct hmx_queue_desc d) {
    unsigned int ir = atomic_load(&q->idx_read);
    unsigned int iw = atomic_load(&q->idx_write);

    if (((iw + 1) & q->idx_mask) == ir) {
        FARF(HIGH, "hmx-queue-push: queue is full\n");
        return false;
    }

    atomic_store(&d.done, 0);
    d.t_us = HAP_perf_get_time_us();

    FARF(HIGH, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    return true;
}

static inline bool hmx_queue_signal(hmx_queue_t q, enum hmx_queue_signal sig) {
    return hmx_queue_push(q, hmx_queue_make_desc((hmx_queue_func) sig, NULL));
}

static inline bool hmx_queue_empty(hmx_queue_t q) {
    return q->idx_pop == atomic_load(&q->idx_write);
}

static inline uint32_t hmx_queue_depth(hmx_queue_t q) {
    return (atomic_load(&q->idx_write) - atomic_load(&q->idx_read)) & q->idx_mask;
}

static inline uint32_t hmx_queue_capacity(hmx_queue_t q) {
    return q->capacity;
}

static inline struct hmx_queue_desc hmx_queue_pop_one(hmx_queue_t q) {
    unsigned int ip = q->idx_pop;
    unsigned int iw = atomic_load(&q->idx_write);

    struct hmx_queue_desc rd = { NULL, NULL };
    if (ip == iw) {
        return rd;
    }

    // Wait for desc to complete
    struct hmx_queue_desc * d = &q->desc[ip];
    while (!atomic_load(&d->done)) {
        FARF(HIGH, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_pop = (ip + 1) & q->idx_mask;

    FARF(HIGH, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    return rd;
}

static inline struct hmx_queue_desc hmx_queue_pop(hmx_queue_t q) {
    while (1) {
        struct hmx_queue_desc d = hmx_queue_pop_one(q);

        uint32_t sig = (uint32_t) d.func;
        if (sig && sig <= HMX_QUEUE_KILL)
            continue;

        return d;
    }
}

// Codex q33a: like hmx_queue_pop (signals are skipped), but gives up when a descriptor is still not done
// deadline_us after it was pushed. The descriptor is then left in place (idx_pop not advanced) and false is
// returned: whatever it uses may still be touched by the HMX queue thread.
static inline bool hmx_queue_pop_timed(hmx_queue_t q, uint64_t deadline_us) {
    for (;;) {
        unsigned int ip = q->idx_pop;
        if (ip == atomic_load(&q->idx_write)) {
            return true;
        }
        struct hmx_queue_desc * d = &q->desc[ip];
        unsigned int spins = 0;
        while (!atomic_load(&d->done)) {
            if ((++spins & 255) == 0 && HAP_perf_get_time_us() - d->t_us > deadline_us) {
                FARF(ERROR, "hmx-queue-pop: descriptor %u still not done %llu us after push; left in place",
                     ip, (unsigned long long) deadline_us);
                return false;
            }
            hex_pause();
        }
        const uint32_t sig = (uint32_t) d->func;
        q->idx_pop = (ip + 1) & q->idx_mask;
        if (sig && sig <= HMX_QUEUE_KILL) {
            continue;
        }
        return true;
    }
}

static inline void hmx_queue_flush(hmx_queue_t q) {
    while (hmx_queue_pop_one(q).func != NULL) ;
}

static inline void hmx_queue_wakeup(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_WAKEUP);
}

static inline void hmx_queue_suspend(hmx_queue_t q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* HMX_QUEUE_H */
