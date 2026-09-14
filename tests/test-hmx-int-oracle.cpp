// Integer-HMX Q4_0 x F32 matmul against an independent CPU integer reconstruction (Codex q32 item 1).
//
// The DSP integer path computes, per activation row r and output column c (hmx-int.h):
//   s_a = max|x| / 32767,  q_k = conv(x_k * (32767 / max|x|)) clamped to [-32767, 32767]
//   per 32-block b:  QT_b = floor(sum_k q_k * w_k / 256)          (w = Q4_0 nibble - 8, exact integers)
//   y = s_a * ( sum_b 256 d_b QT_b  +  128 sum_b d_b )           (d_b = the block's fp16 scale)
// Rounding happens only in the integers; floats enter in the final sums. So the device output should
// match this reconstruction to float-summation accuracy - far tighter than test-backend-ops' 5e-4 NMSE
// against the CPU's Q4_0 x Q8_0 matmul. The float->int conversion rule of the DSP is not documented
// here, so the reconstruction is done twice (round-to-nearest and truncation) and both are reported.
//
// usage: test-hmx-int-oracle [device, default HTP0]   (set GGML_HEXAGON_INT_HMX=1 to use the integer path)
// Exit code 0 when, for every case, one rounding hypothesis matches within the tolerance.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct shape { int64_t m_out, n_act, k; };

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

int main(int argc, char ** argv) {
    const char * dev_name = argc > 1 ? argv[1] : "HTP0";
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name);
    if (!dev) { fprintf(stderr, "device %s not found\n", dev_name); return 2; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { fprintf(stderr, "cannot init %s\n", dev_name); return 2; }

    const shape shapes[] = {
        {  32,  32,   32 }, { 1536,  33, 1536 }, {  64, 256, 2048 }, {  96, 257, 12288 },
        { 1024, 256, 2048 }, { 1025, 256, 2048 }, { 2560, 256, 10240 }, {  33, 255, 2080 },
    };
    const double tol = 1e-5;   // relative to |y| + one block quantum (s_a * 256 * |sum d|)
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    int bad = 0;

    for (const shape & s : shapes) {
        std::vector<float> wf((size_t) (s.m_out * s.k)), x((size_t) (s.n_act * s.k));
        for (auto & v : wf) v = u(gen);
        for (auto & v : x) v = u(gen);
        std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_Q4_0, s.k) * s.m_out);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, wf.data(), wq.data(), 0, s.m_out, s.k, nullptr);

        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, s.k, s.m_out);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, s.n_act);
        ggml_tensor * out = ggml_mul_mat(ctx, a, b);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(a, wq.data(), 0, wq.size());
        ggml_backend_tensor_set(b, x.data(), 0, x.size() * sizeof(float));
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        std::vector<float> y((size_t) (s.m_out * s.n_act));
        ggml_backend_tensor_get(out, y.data(), 0, y.size() * sizeof(float));

        const result rn = compare(y, wq, x, s, true, tol);
        const result tr = compare(y, wq, x, s, false, tol);
        const bool ok = st == GGML_STATUS_SUCCESS && (rn.over == 0 || tr.over == 0);
        bad += ok ? 0 : 1;
        printf("m_out %5lld n_act %4lld k %5lld status %d | nearest: max_abs %.3g max_rel_row %.3g over %lld | "
               "trunc: max_abs %.3g max_rel_row %.3g over %lld | NMSE vs exact %.3g | %s\n",
               (long long) s.m_out, (long long) s.n_act, (long long) s.k, (int) st,
               rn.max_abs, rn.max_rel_row, (long long) rn.over, tr.max_abs, tr.max_rel_row, (long long) tr.over,
               rn.nmse_exact, ok ? "OK" : "MISMATCH");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    printf("%s: %d case(s) mismatched\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
