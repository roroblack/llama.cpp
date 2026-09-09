# SM8735 (Adreno 825 / Hexagon v73) — measurements

Working notes from running llama.cpp on a POCO F7 (Snapdragon 8s Gen 4, SM8735).
Everything here was measured on that device. This is a scratch branch on a personal
fork, not a proposed upstream change.

Device: Adreno 825, Hexagon arch v73, 11 GiB RAM, Android/Termux.
Build: `cmake --preset arm64-android-snapdragon-release`, `ghcr.io/snapdragon-toolchain/arm64-android:v0.7`.

---

## 1. Hexagon: fp16 HMX produces no arithmetic result on v73

`test-backend-ops test -b HTP0 -o MUL_MAT` → **514 pass / 44 fail**, every failure `ERR = inf`.

Correlating the kernel path from `GGML_HEXAGON_VERBOSE=1` against pass/fail over the
whole sweep gives an exact 1:1:

| | count |
|---|---:|
| ops that took `hmx-tiled` | **42** |
| ops that failed with `ERR = inf` | **42** |
| any other failure | 0 |

The 42 include two `f16` ops, so this is not specific to the repacked quantized types —
**everything that reaches HMX is wrong.** The passing `f16`/`f32` ops all took `hvx-tiled`.

`GGML_HEXAGON_NHMX=0` (or `GGML_HEXAGON_MM_SELECT<=2`) → **554/554 pass**, and real
model output becomes correct.

### The decisive experiment

Patched `core_mma_chunk_fp16` / `core_mma_chunk_fp16_short` to zero **all** HMX inputs
(activations `a`, weights `b`, bias `col_scales`) before the asm block, then reran the
single failing case:

```
type_a=q4_0,type_b=f32,m=16,n=5,k=256   ->   ERR = inf
```

`0 x 0 + 0` must be `0`. Getting `inf` means the fp16 HMX sequence
(`mxclracc.hf` / `activation.hf = mxmem(...)` / `mxmem(...):after.hf = acc`)
does not compute on this silicon.

The assembler is not a guard here — `hexagon-clang -mv68 -mhmx` through `-mv81 -mhmx`
all accept the `.hf` HMX encodings, including v68 which predates fp16 HMX entirely.

Also note `htp/main.c: htp_iface_hwinfo()` sets `*n_hmx = 1` unconditionally without
querying anything.

### Threshold

Failures begin exactly at `m > HTP_MM_HMX_MIN_NROWS` (= 4), i.e. as soon as the HMX
path is selected. `m <= 4` (decode) always passes, which is why 1B models appear to work
while larger ones emit garbage.

### Change in this branch

`ggml-hexagon.cpp`: keep HMX off below v75. Not proposed as the right upstream fix —
a runtime self-test would be better, and v75+ was never verified here (no such device).

---

## 2. Hexagon speed after the fix — where the time goes

`gemma-4-E2B-it-Q4_0`, `llama-bench -p 64 -n 32 -r 2`:

| device | pp64 | tg32 |
|---|---:|---:|
| GPUOpenCL (Adreno 825) | 185.57 ± 1.04 | **18.29 ± 0.02** |
| HTP0 (v73, HMX off) | 30.10 ± 0.04 | 1.82 ± 0.01 |

`GGML_HEXAGON_PROFILE=1` over a short run, aggregated by op:

| op | count | ms | us/op | share |
|---|---:|---:|---:|---:|
| MUL_MAT | 990 | 240.4 | 243 | 24.9% |
| MUL_MAT_NX | 300 | 193.9 | 646 | 20.1% |
| RMS_NORM+MUL | 1589 | 188.0 | 118 | 19.5% |
| ADD | 742 | 73.8 | 100 | 7.7% |
| FLASH_ATTN_EXT | 245 | 71.9 | 294 | 7.5% |
| ROPE | 350 | 59.9 | 171 | 6.2% |
| MUL / GEGLU / GELU / SET_ROWS / … | 1190 | ~123 | ~100 | 12.8% |

**965 ms over 5551 ops, mean 174 us/op.** MUL_MAT family is 45%; everything else 55%.

The non-matmul ops operate on tiny tensors (e.g. `1536:2` f32 = 12 KB) and still cost
~100 us each, so more than half of DSP time is per-op fixed cost rather than arithmetic.

Tuning knobs made no difference (1B, pp64/tg32):

| setting | pp64 | tg32 |
|---|---:|---:|
| default | 63.37 | 4.07 |
| `MM_SELECT=2` | 64.78 | 4.04 |
| `MM_SELECT=1` | 40.51 | 3.96 |
| `NHVX=2` | 39.86 | 3.99 |
| `OPPOLL=0` | 62.61 | 4.09 |

### CPU fallback

`supports_op` rejects, from one graph build:

| count | shape | type |
|---:|---|---|
| 40 | `12288:1536 x 12288:512` | q4_0 |
| 30 | `6144:1536 x 6144:512` | q4_0 |
| 20 | `1536:8960 x 1536:N` | **bf16** |
| 2 | `1536:262144 x 1536:512` | q8_0 |

Accepted shapes are all `n = 1`. So the large-`n` (prefill) matmuls fall back to the CPU,
and one tensor falls back at every `n`: `per_layer_model_proj.weight [1536, 8960]` is the
model's **only bf16 tensor** and the backend has no bf16 matmul.

Summed DSP time is ~44% of wall time in HTP0 mode, so roughly half of it is spent off
the DSP (CPU ops plus the tensor copies across the backend boundary).

---

## 3. OpenCL: FA prefill512 program built with empty compile options

`ggml_opencl_supports_op()` compiles the `dk == 512` flash-attention prefill program before
`load_cl_kernels()` has filled in `backend_ctx->kernel_compile_opts`, so no `-cl-std=`
reaches `clBuildProgram`. Instrumented build shows:

```
opts=[ -D DK=512 -D DV=512 -D BLOCK_M=8 -D BLOCK_N=16 -D cl_qcom_subgroup_shuffle=1 -D FA_PREFILL_ONLY]
```

Without `-cl-std` this driver compiles as OpenCL C 1.2, where `cl_khr_subgroups` does not
exist, so the build fails with `unsupported OpenCL extension 'cl_khr_subgroups'` even though
the device advertises 11 subgroup extensions. `ggml_opencl_ensure_fa_f32_f16_prefill_512()`
latches the failure in `static bool failed[2]`, so it is never retried and the op silently
runs on the CPU.

Isolating the compile options on the real FA kernel header, only `-cl-std=CL1.2` reproduces
the diagnostic; `CL2.0`, `CL3.0`, no options and options-without-`-cl-std` all build.

Fixing it does not help this device — flash attention is a net loss here at every context
depth (tg64, patched build, r=3):

| depth | `-fa 0` | `-fa 1` |
|---:|---:|---:|
| 0 | 18.61 | 16.21 |
| 1024 | 16.94 | 11.48 |
| 3072 | 15.62 | 8.47 |

---

## 4. Android setup notes (not bugs, but they cost time)

1. SELinux blocks `/dev/fastrpc-cdsp` for an untrusted app; the session fails with
   `0x80000406` and `dmesg` shows the denial. On a rooted device `magiskpolicy --live`
   rules for `vendor_qdsp_device` / `vendor_xdsp_device` (chr_file) and `adsprpcd_file`
   (dir + file) clear it.
2. `ADSP_LIBRARY_PATH` must contain the directory holding `libggml-htp-v73.so`, otherwise
   the session still fails with the same `0x80000406`.
3. Running the binary under plain `su` does not work — the root linker namespace cannot
   resolve `libcdsprpc.so`. The process has to stay in the app context.
