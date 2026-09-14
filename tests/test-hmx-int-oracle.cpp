// Integer-HMX Q4_0 x F32 matmul against an independent CPU integer reconstruction (Codex q32 item 1).
//
// The DSP integer path computes, per activation row r and output column c (hmx-int.h):
//   s_a = max|x| / 32767,  q_k = conv(x_k * (32767 / max|x|)) clamped to [-32767, 32767]
//   per 32-block b:  QT_b = floor(sum_k q_k * w_k / 256)          (w = Q4_0 nibble - 8, exact integers)
//   y = s_a * ( sum_b 256 d_b QT_b  +  128 sum_b d_b )           (d_b = the block's fp16 scale)
// Rounding happens only in the integers; floats enter in the final sums. So the device output should
// match this reconstruction to float-summation accuracy - far tighter than test-backend-ops' 5e-4 NMSE
// against the CPU's Q4_0 x Q8_0 matmul. The verdict is the interval check in compare_interval(); the
// round-to-nearest and truncation point reconstructions are printed as diagnostics only.
//
// Shapes below the integer path's row minimum (one activation row) run on the HVX path, which quantizes the
// activations to Q8_0. Their single output is checked against a deterministic error bound instead of an NMSE
// (Codex q34: a one-value NMSE fails whenever the dot product lands near zero).
//
// Finally, padded-row layouts (k = 2048 inside rows of 2080), which the Hexagon backend refuses, are run end
// to end through a CPU+HTP scheduler: they must be placed on the CPU, succeed, and equal a CPU-only run of the
// same graph (Codex q36).
//
// usage: test-hmx-int-oracle [device, default HTP0]   (set GGML_HEXAGON_INT_HMX=1 to use the integer path)
//   ORACLE_BARRIER_DIR / ORACLE_ID / ORACLE_PEERS   start barrier for concurrent sessions (timeout = failure)
//   ORACLE_INJECT=nan|inf|ninf|bound|ref            self-test: corrupt one output (nan/inf/ninf), one error
//                                                   bound or one reference interval (bound/ref); every device
//                                                   case must then fail
// Exit code 0 when every case passes.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>

// ORACLE_INJECT=bound / ref: poison the first bound / reference interval inside the verdict functions
static bool g_inject_bound = false;
static bool g_inject_ref   = false;

// Optional start barrier for concurrent-session runs (Codex q32 item 2): with ORACLE_BARRIER_DIR, ORACLE_ID
// and ORACLE_PEERS set, every instance drops "<case>_<id>" into the directory right before computing a case
// and waits (up to 60 s) until all peers have done the same. It aligns the submissions; it does not prove the
// DSP executions overlapped. Use a fresh directory per run (files are not removed). A timeout is returned as
// false and fails the case (Codex q34).
static bool barrier_wait(const char * dir, const char * id, int case_idx, int peers) {
    const std::string mine = std::string(dir) + "/" + std::to_string(case_idx) + "_" + id;
    if (FILE * f = fopen(mine.c_str(), "w")) fclose(f);
    const std::string prefix = std::to_string(case_idx) + "_";
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        int n = 0;
        if (DIR * d = opendir(dir)) {
            while (dirent * e = readdir(d)) {
                if (strncmp(e->d_name, prefix.c_str(), prefix.size()) == 0) n++;
            }
            closedir(d);
        }
        if (n >= peers) return true;
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60)) {
            fprintf(stderr, "barrier: case %d waited 60 s for %d peers (saw %d)\n", case_idx, peers, n);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

// hvx = 1: one activation row, below the integer path's row minimum -> HVX path, Q8_0 bound check
struct shape { int64_t m_out, n_act, k; int hvx; };

static float fp16_to_f32(uint16_t h) { return ggml_fp16_to_fp32(h); }

struct result { double max_abs, max_rel_row, nmse_exact; int64_t over; };

// device output y[n_act][m_out]; weights as Q4_0 blocks [m_out][k/32]; activations x[n_act][k]
static result compare(const std::vector<float> & y, const std::vector<uint8_t> & wq, const std::vector<float> & x,
                      const shape & s, bool round_nearest, double tol) {
    const int64_t B = s.k / 32;
    const size_t bsz = ggml_row_size(GGML_TYPE_Q4_0, 32);   // 18 bytes: fp16 d + 16 bytes of nibbles
    result r = { 0.0, 0.0, 0.0, 0 };
    double se = 0.0, sref = 0.0;
    std::vector<int32_t> q(s.k);
    for (int64_t row = 0; row < s.n_act; row++) {
        const float * xr = x.data() + row * s.k;
        float m = 0.0f;
        for (int64_t i = 0; i < s.k; i++) m = std::max(m, std::fabs(xr[i]));
        const float sa  = m > 0.0f ? m / 32767.0f : 0.0f;
        const float inv = m > 0.0f ? 32767.0f / m : 0.0f;
        for (int64_t i = 0; i < s.k; i++) {
            const float v = xr[i] * inv;
            long t = round_nearest ? std::lrint(v) : (long) v;   // lrint: current mode = nearest-even
            q[i] = (int32_t) std::min(32767L, std::max(-32767L, t));
        }
        double row_max = 0.0, row_err = 0.0;
        for (int64_t c = 0; c < s.m_out; c++) {
            const uint8_t * wrow = wq.data() + (size_t) c * B * bsz;
            double acc = 0.0, dsum = 0.0, exact = 0.0;
            for (int64_t b = 0; b < B; b++) {
                const uint8_t * blk = wrow + b * bsz;
                uint16_t dh; memcpy(&dh, blk, 2);
                const float d = fp16_to_f32(dh);
                const uint8_t * qs = blk + 2;
                int64_t S = 0;
                for (int j = 0; j < 16; j++) {
                    const int w0 = (qs[j] & 0x0f) - 8, w1 = (qs[j] >> 4) - 8;   // elements j and j + 16
                    S += (int64_t) q[b * 32 + j] * w0 + (int64_t) q[b * 32 + j + 16] * w1;
                    exact += (double) xr[b * 32 + j] * w0 * d + (double) xr[b * 32 + j + 16] * w1 * d;
                }
                const int64_t QT = (S >= 0) ? S / 256 : -((-S + 255) / 256);   // floor division
                acc  += 256.0 * d * (double) QT;
                dsum += d;
            }
            const double yo = (double) sa * (acc + 128.0 * dsum);
            const double yd = y[(size_t) row * s.m_out + c];
            const double e = std::fabs(yd - yo);
            r.max_abs = std::max(r.max_abs, e);
            row_err = std::max(row_err, e);
            row_max = std::max(row_max, std::fabs(yo));
            se += (yd - exact) * (yd - exact);
            sref += exact * exact;
            if (e > tol * (std::fabs(yo) + (double) sa * 256.0 * std::fabs(dsum) + 1e-30)) r.over++;
        }
        if (row_max > 0.0) r.max_rel_row = std::max(r.max_rel_row, row_err / row_max);
    }
    r.nmse_exact = sref > 0.0 ? se / sref : 0.0;
    return r;
}

// Interval check of the integer semantics. Measured on the device (2026-09-14): the DSP truncates x * (32767 /
// max|x|) toward zero (the truncation hypothesis fits ~5x better than nearest), but it forms that product in
// qf32, which can land on the other side of an integer when the exact product is within a few ulp of it.
// For such elements both integers are allowed; each block sum S_b - and QT_b = floor(S_b / 256) - then has a
// small range, and y an interval. The device must land inside it, plus float-summation slack (tol relative to
// the magnitude of the terms). Anything else - a wrong tile, block, segment, scale or row - lands far outside.
// The 4e-7 ambiguity window is EMPIRICAL (Codex q34): it is not a proven bound on the qf32 product error, so
// this is a tight interval oracle, not a proof. *n_amb counts ambiguous ACTIVATION elements (n_act * k), not
// outputs.
static result compare_interval(const std::vector<float> & y, const std::vector<uint8_t> & wq, const std::vector<float> & x,
                               const shape & s, double tol, int64_t * n_amb) {
    const int64_t B = s.k / 32;
    const size_t bsz = ggml_row_size(GGML_TYPE_Q4_0, 32);
    result r = { 0.0, 0.0, 0.0, 0 };
    std::vector<int32_t> qa(s.k), qb(s.k);   // the two candidate integers (equal when unambiguous)
    int64_t amb = 0;
    bool injected = false;
    for (int64_t row = 0; row < s.n_act; row++) {
        const float * xr = x.data() + row * s.k;
        float m = 0.0f;
        for (int64_t i = 0; i < s.k; i++) m = std::max(m, std::fabs(xr[i]));
        const float sa  = m > 0.0f ? m / 32767.0f : 0.0f;
        const float inv = m > 0.0f ? 32767.0f / m : 0.0f;
        for (int64_t i = 0; i < s.k; i++) {
            const double v = (double) xr[i] * (double) inv;
            long t = (long) v, alt = (long) v;                       // truncation toward zero
            const double n = std::nearbyint(v);
            if (n != 0.0 && std::fabs(v - n) <= 4e-7 * std::max(1.0, std::fabs(v))) {
                t   = (long) n;                                       // exact product on or above |n|
                alt = v > 0 ? (long) n - 1 : (long) n + 1;            // qf32 product just below |n|
                amb++;
            }
            qa[i] = (int32_t) std::min(32767L, std::max(-32767L, t));
            qb[i] = (int32_t) std::min(32767L, std::max(-32767L, alt));
        }
        for (int64_t c = 0; c < s.m_out; c++) {
            const uint8_t * wrow = wq.data() + (size_t) c * B * bsz;
            double lo = 0.0, hi = 0.0, dsum = 0.0, mag = 0.0;
            for (int64_t b = 0; b < B; b++) {
                const uint8_t * blk = wrow + b * bsz;
                uint16_t dh; memcpy(&dh, blk, 2);
                const double d = fp16_to_f32(dh);
                const uint8_t * qs = blk + 2;
                int64_t smin = 0, smax = 0;
                for (int j = 0; j < 32; j++) {
                    const int w = (j < 16 ? (qs[j] & 0x0f) : (qs[j - 16] >> 4)) - 8;   // element j of the block
                    const int64_t p0 = (int64_t) qa[b * 32 + j] * w, p1 = (int64_t) qb[b * 32 + j] * w;
                    smin += std::min(p0, p1);
                    smax += std::max(p0, p1);
                }
                auto fl = [](int64_t S) { return S >= 0 ? S / 256 : -((-S + 255) / 256); };
                const double c0 = 256.0 * d * (double) fl(smin), c1 = 256.0 * d * (double) fl(smax);
                lo  += std::min(c0, c1);
                hi  += std::max(c0, c1);
                mag += std::max(std::fabs(c0), std::fabs(c1));
                dsum += d;
            }
            double ylo = (double) sa * (lo + 128.0 * dsum), yhi = (double) sa * (hi + 128.0 * dsum);
            const double slack = tol * (double) sa * (mag + 128.0 * std::fabs(dsum)) + 1e-30;
            if (g_inject_ref && !injected) { ylo = NAN; injected = true; }
            const double yd = y[(size_t) row * s.m_out + c];
            // Codex q34: every comparison with NaN is false, so a NaN output would otherwise count as inside
            if (!std::isfinite(yd) || !std::isfinite(ylo) || !std::isfinite(yhi) || !std::isfinite(slack)) {
                r.over++;
                r.max_abs = INFINITY;
                continue;
            }
            const double out = yd < ylo - slack ? (ylo - slack) - yd : yd > yhi + slack ? yd - (yhi + slack) : 0.0;
            if (out > 0.0) r.over++;
            r.max_abs = std::max(r.max_abs, out);
        }
    }
    *n_amb = amb;
    return r;
}

// Deterministic bound for the HVX path (Q8_0 activations: per 32-block scale max|x_b| / 127). Whatever the
// rounding of the activation, each element is off by at most one step, so
//   |y - exact| <= sum_b (max|x_b| / 127) * sum_j |w_j d_b|   +  1e-3 * sum |x_j w_j d_b|
// where exact = sum x_j w_j d_b in double. The 1e-3 term covers the fp16 activation scale (relative rounding
// 2^-11) and float summation; it is a chosen allowance, not a derived bound (Codex q36). max_abs reports the worst
// |y - exact| / bound (<= 1 passes). Every quantity entering the verdict must be finite (Codex q36).
static result compare_q8bound(const std::vector<float> & y, const std::vector<uint8_t> & wq, const std::vector<float> & x,
                              const shape & s) {
    const int64_t B = s.k / 32;
    const size_t bsz = ggml_row_size(GGML_TYPE_Q4_0, 32);
    result r = { 0.0, 0.0, 0.0, 0 };
    bool injected = false;
    for (int64_t row = 0; row < s.n_act; row++) {
        const float * xr = x.data() + row * s.k;
        for (int64_t c = 0; c < s.m_out; c++) {
            const uint8_t * wrow = wq.data() + (size_t) c * B * bsz;
            double exact = 0.0, bound = 0.0, mag = 0.0;
            for (int64_t b = 0; b < B; b++) {
                const uint8_t * blk = wrow + b * bsz;
                uint16_t dh; memcpy(&dh, blk, 2);
                const double d = fp16_to_f32(dh);
                const uint8_t * qs = blk + 2;
                double mb = 0.0;
                for (int j = 0; j < 32; j++) mb = std::max(mb, (double) std::fabs(xr[b * 32 + j]));
                for (int j = 0; j < 32; j++) {
                    const int w = (j < 16 ? (qs[j] & 0x0f) : (qs[j - 16] >> 4)) - 8;
                    const double t = (double) xr[b * 32 + j] * w * d;
                    exact += t;
                    mag   += std::fabs(t);
                    bound += (mb / 127.0) * std::fabs(w * d);
                }
            }
            bound += 1e-3 * mag + 1e-9;
            if (g_inject_bound && !injected) { bound = NAN; injected = true; }
            if (g_inject_ref && !injected)   { exact = NAN; injected = true; }
            const double yd = y[(size_t) row * s.m_out + c];
            const double ratio = std::fabs(yd - exact) / bound;
            if (!std::isfinite(yd) || !std::isfinite(exact) || !std::isfinite(bound) || !(bound > 0.0) || !std::isfinite(ratio)) {
                r.over++;
                r.max_abs = INFINITY;
                continue;
            }
            if (ratio > 1.0) r.over++;
            r.max_abs = std::max(r.max_abs, ratio);
        }
    }
    return r;
}

// Codex q36: padded-row layouts end to end. Weights of physical row length kv (2080) viewed as k = 2048, and the
// activation likewise, in CPU buffers; computed through a scheduler with the HTP backend first and the CPU second,
// and on the CPU alone. The node must be placed on the CPU (the HTP backend refuses the layout), the computation
// must succeed, and both results must be identical. A contiguous control (kv == k) reports where it was placed.
static int run_fallback(ggml_backend_t htp, std::mt19937 & gen) {
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) { printf("[fallback] no CPU backend | MISMATCH\n"); return 1; }
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const int64_t m = 64, k = 2048;
    int bad = 0;
    for (int64_t kv : { (int64_t) 2080, (int64_t) 2048 }) {
        for (int64_t n : { (int64_t) 32, (int64_t) 64, (int64_t) 256 }) {
            std::vector<float> wf((size_t) (m * kv)), x((size_t) (n * kv));
            for (auto & v : wf) v = u(gen);
            for (auto & v : x) v = u(gen);
            std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_Q4_0, kv) * m);
            ggml_quantize_chunk(GGML_TYPE_Q4_0, wf.data(), wq.data(), 0, m, kv, nullptr);

            std::vector<float> y[2];
            ggml_status st[2] = { GGML_STATUS_FAILED, GGML_STATUS_FAILED };
            std::string placed = "?";
            for (int run = 0; run < 2; run++) {   // 0: scheduler HTP + CPU, 1: CPU only
                ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
                ggml_context * ctx = ggml_init(ip);
                ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, kv, m);
                ggml_tensor * Bt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv, n);
                ggml_tensor * a = kv == k ? A  : ggml_view_2d(ctx, A,  k, m, A->nb[1],  0);
                ggml_tensor * b = kv == k ? Bt : ggml_view_2d(ctx, Bt, k, n, Bt->nb[1], 0);
                ggml_tensor * out = ggml_mul_mat(ctx, a, b);
                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, out);
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
                ggml_backend_tensor_set(A, wq.data(), 0, wq.size());
                ggml_backend_tensor_set(Bt, x.data(), 0, x.size() * sizeof(float));
                if (run == 0) {
                    ggml_backend_t backs[2] = { htp, cpu };
                    ggml_backend_sched_t sched = ggml_backend_sched_new(backs, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
                    if (ggml_backend_sched_alloc_graph(sched, gf)) {
                        st[run] = ggml_backend_sched_graph_compute(sched, gf);
                        ggml_backend_t pb = ggml_backend_sched_get_tensor_backend(sched, out);
                        placed = pb ? ggml_backend_name(pb) : "none";
                    }
                    ggml_backend_sched_free(sched);
                } else {
                    st[run] = ggml_backend_graph_compute(cpu, gf);
                }
                y[run].resize((size_t) (m * n));
                ggml_backend_tensor_get(out, y[run].data(), 0, y[run].size() * sizeof(float));
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
            }
            double max_abs = 0.0;
            bool finite = true;
            for (size_t i = 0; i < y[0].size(); i++) {
                if (!std::isfinite(y[0][i]) || !std::isfinite(y[1][i])) finite = false;
                max_abs = std::max(max_abs, (double) std::fabs(y[0][i] - y[1][i]));
            }
            const bool on_cpu = placed == ggml_backend_name(cpu);
            const bool ok = kv == k ? (st[0] == GGML_STATUS_SUCCESS && finite)      // control: placement reported only
                                    : (st[0] == GGML_STATUS_SUCCESS && st[1] == GGML_STATUS_SUCCESS && on_cpu && finite && max_abs == 0.0);
            bad += ok ? 0 : 1;
            printf("[fallback] m %lld n %lld k %lld rows %lld (%s) | placed on %s | status %d/%d | max |sched - cpu| %.3g | %s\n",
                   (long long) m, (long long) n, (long long) k, (long long) kv, kv == k ? "contiguous control" : "padded",
                   placed.c_str(), (int) st[0], (int) st[1], max_abs, ok ? "OK" : "MISMATCH");
        }
    }
    ggml_backend_free(cpu);
    return bad;
}

int main(int argc, char ** argv) {
    const char * dev_name = argc > 1 ? argv[1] : "HTP0";
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name);
    if (!dev) { fprintf(stderr, "device %s not found\n", dev_name); return 2; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { fprintf(stderr, "cannot init %s\n", dev_name); return 2; }

    const shape shapes[] = {
        {  32,  32,   32, 0 }, { 1536,  33, 1536, 0 }, {  64, 256, 2048, 0 }, {  96, 257, 12288, 0 },
        { 1024, 256, 2048, 0 }, { 1025, 256, 2048, 0 }, { 2560, 256, 10240, 0 }, {  33, 255, 2080, 0 },
        // single-output shapes (HVX path), kept with a deterministic bound instead of test-backend-ops' NMSE
        {    1,   1, 2016, 1 }, {    1,   1, 2080, 1 }, {    1,   1, 12288, 1 },
    };
    const double tol = 1e-5;   // relative to |y| + one block quantum (s_a * 256 * |sum d|)
    const char * barrier_dir = getenv("ORACLE_BARRIER_DIR");
    const char * id          = getenv("ORACLE_ID") ? getenv("ORACLE_ID") : "a";
    const int    peers       = getenv("ORACLE_PEERS") ? atoi(getenv("ORACLE_PEERS")) : 1;
    const char * inject      = getenv("ORACLE_INJECT");
    const bool   inject_out  = inject && (!strcmp(inject, "nan") || !strcmp(inject, "inf") || !strcmp(inject, "ninf"));
    g_inject_bound = inject && !strcmp(inject, "bound");
    g_inject_ref   = inject && !strcmp(inject, "ref");
    // different inputs per instance, so a result that leaked from another session cannot pass
    std::mt19937 gen(42 + (unsigned) id[0]);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    int bad = 0, case_idx = 0;

    for (const shape & s : shapes) {
        std::vector<float> wf((size_t) (s.m_out * s.k)), x((size_t) (s.n_act * s.k));
        for (auto & v : wf) v = u(gen);
        for (auto & v : x) v = u(gen);
        std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_Q4_0, s.k) * s.m_out);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, wf.data(), wq.data(), 0, s.m_out, s.k, nullptr);

        // The weights get their own buffer marked as weights, as test-backend-ops and llama do: the Hexagon
        // backend repacks Q4_0 into its tile layout only for tensors in such a buffer. (The first version put
        // everything in one unmarked buffer; both the integer and the HVX path then read un-repacked blocks
        // and every case "mismatched" by 1e7.)
        ggml_init_params ipw = { ggml_tensor_overhead() * 2, nullptr, true };
        ggml_context * ctx_w = ggml_init(ipw);
        ggml_tensor * a = ggml_new_tensor_2d(ctx_w, GGML_TYPE_Q4_0, s.k, s.m_out);
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, s.n_act);
        ggml_tensor * out = ggml_mul_mat(ctx, a, b);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
        ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(a, wq.data(), 0, wq.size());
        ggml_backend_tensor_set(b, x.data(), 0, x.size() * sizeof(float));
        bool barrier_ok = true;
        if (barrier_dir && peers > 1) barrier_ok = barrier_wait(barrier_dir, id, case_idx, peers);
        case_idx++;
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        std::vector<float> y((size_t) (s.m_out * s.n_act));
        ggml_backend_tensor_get(out, y.data(), 0, y.size() * sizeof(float));
        if (inject_out) {
            // self-test of the verdicts (Codex q34): a non-finite output must fail every check
            y[y.size() / 2] = !strcmp(inject, "inf") ? INFINITY : !strcmp(inject, "ninf") ? -INFINITY : NAN;
        }

        bool ok;
        if (s.hvx) {
            const result qb = compare_q8bound(y, wq, x, s);
            ok = st == GGML_STATUS_SUCCESS && qb.over == 0 && barrier_ok;
            printf("[%s] m_out %5lld n_act %4lld k %5lld status %d | hvx q8 bound: over %lld, worst err/bound %.3g%s | %s\n",
                   id, (long long) s.m_out, (long long) s.n_act, (long long) s.k, (int) st,
                   (long long) qb.over, qb.max_abs, barrier_ok ? "" : " | barrier timeout", ok ? "OK" : "MISMATCH");
        } else {
            const result rn = compare(y, wq, x, s, true, tol);
            const result tr = compare(y, wq, x, s, false, tol);
            int64_t amb = 0;
            const result iv = compare_interval(y, wq, x, s, tol, &amb);
            // the verdict is the interval check; the two point hypotheses are kept as diagnostics
            ok = st == GGML_STATUS_SUCCESS && iv.over == 0 && barrier_ok;
            printf("[%s] m_out %5lld n_act %4lld k %5lld status %d | interval: outside %lld/%lld outputs (by up to %.3g), "
                   "ambiguous activation elements %lld/%lld | nearest: max_abs %.3g over %lld | trunc: max_abs %.3g over %lld | "
                   "NMSE vs exact %.3g%s | %s\n",
                   id, (long long) s.m_out, (long long) s.n_act, (long long) s.k, (int) st,
                   (long long) iv.over, (long long) (s.m_out * s.n_act), iv.max_abs, (long long) amb, (long long) (s.n_act * s.k),
                   rn.max_abs, (long long) rn.over, tr.max_abs, (long long) tr.over,
                   rn.nmse_exact, barrier_ok ? "" : " | barrier timeout", ok ? "OK" : "MISMATCH");
        }
        bad += ok ? 0 : 1;
        ggml_backend_buffer_free(buf);
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx);
        ggml_free(ctx_w);
    }
    // the fallback section is skipped in concurrent and injection runs (its own verdict does not use them)
    if (!barrier_dir && !inject) {
        bad += run_fallback(backend, gen);
    }
    ggml_backend_free(backend);
    printf("%s: %d case(s) mismatched\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
