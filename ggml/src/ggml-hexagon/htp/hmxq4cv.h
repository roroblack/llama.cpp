// Step 4 building block: convert one host-repacked Q4_0 weight tile (576 B, repack_q4_0_tiled in
// ggml-hexagon.cpp) into what the integer HMX path consumes, on HVX, and check it exhaustively.
//
// Source tile (32 output rows n x one Q4_0 block of 32 K values):
//   bytes 0..511   byte[cp*32 + n] = (q[n][2cp+1] << 4) | q[n][2cp], cp 0..15   (q unsigned 0..15)
//   bytes 512..575 32 x fp16 d[n]
// Tiles sit at (ct*n_k_tiles + kt) * 576 in DDR, so every other tile is only 64-byte aligned:
// all source loads are unaligned.
// Targets:
//   wt   int8 HMX weight tile W[k][c] = q[c][k] - 8 at 128*(k>>2) + 4c + (k&3)      (8 vectors)
//   cv   word c = C | C << 16, C = (128 * sum_k w[c][k]) mod 2^16  (the C,C halfword pair per column)
//   dv   word c = 256 * d[c] as fp32
// Target vector g holds k = 4g..4g+3 = (cp 2g lo, cp 2g hi, cp 2g+1 lo, cp 2g+1 hi), i.e. source
// bytes 64g..64g+63. Per source vector v: split nibbles, vshuff(hi, lo, -1) interleaves them per byte,
// then vshuffh interleaves the two 32-halfword halves -> bytes 4c+{lo(a), hi(a), lo(b), hi(b)}.
//
// Q4G_PRINT(fmt, ...) must be defined by the includer.

#ifndef HMXQ4CV_H
#define HMXQ4CV_H

#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
#include "hmxq4gemm.h"

// fp16 -> fp32 by bits, exact for every fp16 including zero, subnormals, inf/nan (scalar reference)
static inline float hq_h2f(unsigned short h) {
    unsigned s = (unsigned) (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else { int ee = -1; do { ee++; m <<= 1; } while (!(m & 0x400)); m &= 0x3ff; bits = s | ((unsigned) (127 - 15 - ee) << 23) | (m << 13); }
    } else if (e == 31) bits = s | 0x7f800000u | (m << 13);
    else bits = s | ((e + 112) << 23) | (m << 13);
    union { unsigned u; float f; } x = { bits };
    return x.f;
}

// the nibble part: 4 unaligned source vectors -> 8 int8 target vectors, and cv
__attribute__((noinline)) static void hq_cvt_nibbles(const unsigned char * src, HVX_Vector * wt, HVX_Vector * cv) {
    const HVX_Vector k0f  = Q6_V_vsplat_R(0x0f0f0f0f);
    const HVX_Vector k08  = Q6_V_vsplat_R(0x08080808);
    HVX_Vector       sumq = Q6_V_vzero();
#pragma unroll
    for (int v = 0; v < 4; v++) {
        HVX_Vector     x  = *(const HVX_UVector *) (src + v * 128);
        HVX_Vector     lo = Q6_V_vand_VV(x, k0f);
        HVX_Vector     hi = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(x, 4), k0f);
        HVX_VectorPair p  = Q6_W_vshuff_VVR(hi, lo, -1);
        HVX_Vector     t0 = Q6_Vh_vshuff_Vh(Q6_V_lo_W(p));
        HVX_Vector     t1 = Q6_Vh_vshuff_Vh(Q6_V_hi_W(p));
        sumq = Q6_Vuw_vrmpyacc_VuwVubRub(sumq, t0, 0x01010101);
        sumq = Q6_Vuw_vrmpyacc_VuwVubRub(sumq, t1, 0x01010101);
        wt[2 * v]     = Q6_Vb_vsub_VbVb(t0, k08);
        wt[2 * v + 1] = Q6_Vb_vsub_VbVb(t1, k08);
    }
    // sum w = sum q - 256; C = 128 * sum w mod 2^16, duplicated into both halfwords of word c
    HVX_Vector c = Q6_V_vand_VV(Q6_Vw_vasl_VwR(Q6_Vw_vsub_VwVw(sumq, Q6_V_vsplat_R(256)), 7), Q6_V_vsplat_R(0xffff));
    *cv = Q6_V_vor_VV(c, Q6_Vw_vasl_VwR(c, 16));
}

// the scale part, scalar (32 conversions): word c = 256 * d[c]
__attribute__((noinline)) static void hq_cvt_scales(const unsigned char * src, HVX_Vector * dv) {
    const unsigned short * d  = (const unsigned short *) (src + 512);
    float *                df = (float *) dv;
    for (int c = 0; c < 32; c++) df[c] = 256.0f * hq_h2f(d[c]);
}

// HVX scale conversion by bits. v73 has no vcvt(hf -> sf) ("Cannot select V6.vcvt.sf.hf" - the
// intrinsic is in the header but the v73 back end rejects it), so:
//   load bytes 448..575 (halfwords 32..63 = d[0..31], nothing past the tile), vzxt to words
//   (even/odd halfwords), vshuff(-4) back to element order -> hi vector word c = d[c]
//   normal     (mag >= 0x400): bits = sign | ((mag << 13) + 0x3c000000)   = 256*d exactly
//   subnormal  (0 < mag < 0x400): float(mag) via vconv, minus 16 in the exponent = mag * 2^-16 = 256*d
//   zero / -0: sign only
// inf/nan (mag >= 0x7c00) are not valid Q4_0 scales; they come out as huge finite values, not checked.
static inline HVX_Vector hq_h2f256_words(HVX_Vector x) {           // x: word c = fp16 bits of d[c]
    const HVX_Vector sign = Q6_Vw_vasl_VwR(Q6_V_vand_VV(x, Q6_V_vsplat_R(0x8000)), 16);
    const HVX_Vector mag  = Q6_V_vand_VV(x, Q6_V_vsplat_R(0x7fff));
    const HVX_Vector nrm  = Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(mag, 13), Q6_V_vsplat_R(0x3c000000));
    const HVX_Vector sub  = Q6_Vw_vsub_VwVw(Q6_Vsf_equals_Vw(mag), Q6_V_vsplat_R(16 << 23));
    const HVX_VectorPred is_nrm  = Q6_Q_vcmp_gt_VwVw(mag, Q6_V_vsplat_R(0x3ff));
    const HVX_VectorPred is_zero = Q6_Q_vcmp_eq_VwVw(mag, Q6_V_vzero());
    HVX_Vector r = Q6_V_vmux_QVV(is_nrm, nrm, sub);
    r = Q6_V_vmux_QVV(is_zero, Q6_V_vzero(), r);
    return Q6_V_vor_VV(r, sign);
}

__attribute__((noinline)) static void hq_cvt_scales_hvx(const unsigned char * src, HVX_Vector * dv) {
    HVX_Vector     x = *(const HVX_UVector *) (src + 448);
    HVX_VectorPair w = Q6_Wuw_vzxt_Vuh(x);
    HVX_VectorPair s = Q6_W_vshuff_VVR(Q6_V_hi_W(w), Q6_V_lo_W(w), -4);
    *dv = hq_h2f256_words(Q6_V_hi_W(s));
}

__attribute__((noinline)) static void hq_h2f256_raw(HVX_Vector x, HVX_Vector * out) { *out = hq_h2f256_words(x); }

// build one source tile exactly as repack_q4_0_tiled would, from q[n][j] (0..15) and fp16 d[n]
static void hq_repack_ref(const unsigned char q[32][32], const unsigned short * d, unsigned char * t) {
    for (int cp = 0; cp < 16; cp++) for (int n = 0; n < 32; n++)
        t[cp * 32 + n] = (unsigned char) ((q[n][2 * cp + 1] << 4) | q[n][2 * cp]);
    for (int n = 0; n < 32; n++) { t[512 + 2 * n] = (unsigned char) (d[n] & 0xff); t[513 + 2 * n] = (unsigned char) (d[n] >> 8); }
}

__attribute__((noinline, optnone)) static void hmxq4cv_run(unsigned char * base) {
    unsigned char * srcbuf = base;                          // 2 source tiles, one at +64 (misaligned)
    HVX_Vector *    wt     = (HVX_Vector *) (base + 4096);  // 8 vectors
    HVX_Vector *    cv     = (HVX_Vector *) (base + 5120);
    HVX_Vector *    dv     = (HVX_Vector *) (base + 5248);
    static unsigned char  q[32][32];
    static unsigned short d[32];
    int bad_w = 0, bad_c = 0, bad_d = 0, tiles = 0;

    // 1) nibbles and cv over 200 random tiles (incl. all-0, all-15, pad rows = 8), at both alignments
    q4g_lcg = 555u;
    for (int it = 0; it < 200; it++) {
        for (int n = 0; n < 32; n++) for (int j = 0; j < 32; j++) {
            int v = (int) q4g_frand(0.0f, 15.999f);
            if (it == 0) v = 0;
            if (it == 1) v = 15;
            if (it == 2 && n >= 20) v = 8;
            q[n][j] = (unsigned char) v;
        }
        for (int n = 0; n < 32; n++) d[n] = q4g_f2h(q4g_frand(-0.06f, 0.06f));
        unsigned char * t = srcbuf + ((it & 1) ? 64 : 0);
        hq_repack_ref(q, d, t);
        hq_cvt_nibbles(t, wt, cv);
        hq_cvt_scales(t, dv);
        const signed char *    w8  = (const signed char *) wt;
        const unsigned *       c32 = (const unsigned *) cv;
        const float *          d32 = (const float *) dv;
        for (int c = 0; c < 32; c++) {
            int sw = 0;
            for (int k = 0; k < 32; k++) {
                int w = q[c][k] - 8; sw += w;
                if (w8[q4g_iw(k, c)] != w) bad_w++;
            }
            unsigned C = (unsigned) ((128 * sw) & 0xffff);
            if (c32[c] != (C | (C << 16))) bad_c++;
            if (d32[c] != 256.0f * hq_h2f(d[c])) bad_d++;
        }
        tiles++;
    }
    Q4G_PRINT("Q4CV tiles %d (half at +64 B): int8 weights %d/%d bad, cv %d/%d bad, dv %d/%d bad",
              tiles, bad_w, tiles * 1024, bad_c, tiles * 32, bad_d, tiles * 32);

    // 2) the scalar fp16 -> fp32 reference against the compiler's own __fp16 conversion, all 65,536 values
    int bad_h = 0, first_h = -1;
    for (int h = 0; h < 65536; h++) {
        union { unsigned short u; __fp16 f; } a = { (unsigned short) h };
        float ref = (float) a.f, got = hq_h2f((unsigned short) h);
        union { float f; unsigned u; } r = { ref }, g = { got };
        const int nan = ((h >> 10) & 0x1f) == 31 && (h & 0x3ff);
        if (nan ? !(got != got) : (r.u != g.u)) { bad_h++; if (first_h < 0) first_h = h; }
    }
    Q4G_PRINT("Q4CV fp16->fp32 by bits vs __fp16 cast, all 65536: %d bad (first 0x%04x)", bad_h, first_h < 0 ? 0 : first_h);

    // X2: the HVX bit conversion over every finite fp16 value, one word per value
    {
        HVX_Vector * xin = (HVX_Vector *) (base + 6144);
        HVX_Vector * xo  = (HVX_Vector *) (base + 6272);
        unsigned *   xw  = (unsigned *) xin;
        int bad = 0, checked = 0, first = -1;
        for (int b0 = 0; b0 < 65536; b0 += 32) {
            for (int i = 0; i < 32; i++) xw[i] = (unsigned) (b0 + i);
            hq_h2f256_raw(xin[0], xo);
            const float * f = (const float *) xo;
            for (int i = 0; i < 32; i++) {
                const int h = b0 + i;
                if (((h >> 10) & 0x1f) == 31) continue;
                const float want = 256.0f * hq_h2f((unsigned short) h);
                union { float f; unsigned u; } a = { want }, g = { f[i] };
                checked++;
                if (a.u != g.u) { bad++; if (first < 0) first = h; }
            }
        }
        Q4G_PRINT("Q4CV X2 HVX fp16 -> 256*fp32 by bits, %d finite values (incl. +-0, subnormals): %d bad (first 0x%04x)",
                  checked, bad, first < 0 ? 0 : first);
    }
    // tiles again, through the HVX scale conversion
    {
        HVX_Vector * dA = (HVX_Vector *) (base + 6528);
        int badA = 0;
        q4g_lcg = 777u;
        for (int it = 0; it < 200; it++) {
            for (int n = 0; n < 32; n++) for (int j = 0; j < 32; j++) q[n][j] = (unsigned char) (int) q4g_frand(0.0f, 15.999f);
            for (int n = 0; n < 32; n++) d[n] = (it == 0 && n < 8) ? (unsigned short) (n == 0 ? 0 : n == 1 ? 0x8000 : n == 2 ? 0x0001 : n == 3 ? 0x03ff : n == 4 ? 0x0400 : n == 5 ? 0x7bff : n == 6 ? 0xfbff : 0x8001)
                                                                      : q4g_f2h(q4g_frand(-0.06f, 0.06f));
            unsigned char * t = srcbuf + ((it & 1) ? 64 : 0);
            hq_repack_ref(q, d, t);
            hq_cvt_scales_hvx(t, dA);
            const float * fa = (const float *) dA;
            for (int c = 0; c < 32; c++) { const float want = 256.0f * hq_h2f(d[c]); badA += fa[c] != want; }
        }
        Q4G_PRINT("Q4CV HVX scales over 200 tiles (incl. +-0, subnormals, +-max): %d/6400 bad", badA);
    }

    // 3) cycles: the nibble part per tile, and the scalar scale part per tile
    unsigned long long t0, t1, tn = ~0ull, ts = ~0ull, tv = ~0ull;
    for (int r = 0; r < 5; r++) {
        t0 = qurt_get_core_pcycles();
        for (int i = 0; i < 100; i++) hq_cvt_nibbles(srcbuf + ((i & 1) ? 64 : 0), wt, cv);
        t1 = qurt_get_core_pcycles(); if (t1 - t0 < tn) tn = t1 - t0;
        t0 = qurt_get_core_pcycles();
        for (int i = 0; i < 100; i++) hq_cvt_scales(srcbuf, dv);
        t1 = qurt_get_core_pcycles(); if (t1 - t0 < ts) ts = t1 - t0;
        t0 = qurt_get_core_pcycles();
        for (int i = 0; i < 100; i++) hq_cvt_scales_hvx(srcbuf + ((i & 1) ? 64 : 0), dv);
        t1 = qurt_get_core_pcycles(); if (t1 - t0 < tv) tv = t1 - t0;
    }
    Q4G_PRINT("Q4CV cycles per tile: nibbles+cv %.1f, scales scalar %.1f, scales HVX %.1f",
              (double) tn / 100, (double) ts / 100, (double) tv / 100);
}

#endif
