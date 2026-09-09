# SM8735 (Adreno 825 / Hexagon v73) — measurements

Working notes from running llama.cpp on a POCO F7 (Snapdragon 8s Gen 4, SM8735).
Everything here was measured on that device. This is a scratch branch on a personal
fork, not a proposed upstream change.

Device: Adreno 825, Hexagon arch v73, 11 GiB RAM, Android/Termux.
Build: `cmake --preset arm64-android-snapdragon-release`, `ghcr.io/snapdragon-toolchain/arm64-android:v0.7`.

---

## 1. Hexagon: HMX returns zero on this device, and the silicon is not the reason

`test-backend-ops test -b HTP0 -o MUL_MAT`: **514 pass / 556** with HMX on, **556/556**
with it off. Every extra failure is an op that took the `hmx-tiled` path.

### The failures are zeros, not infinities

The test reports `ERR = inf`, which is not the same as the tensor containing infinities.
With DSP-side logging enabled (see below), the actual values are:

```
hmx-dbg in : act 350c bb4f 356b b5e8 | wgt 37f7 b3f7 8000 b3fd | scale 3c00 0000
hmx-dbg out: vtcm 0000 0000 0000 0000 0000 0000
hmx-dbg dst: rows 5 cols 32 dstcols 16 stride 16 | 00000000 00000000 00000000 00000000
```

**1857 of 1857 HMX outputs are exactly zero.** The operands arriving at the kernel are
ordinary fp16 values and the bias is the documented identity (32 words of `0x3c00`, then
32 zero words). The `inf` comes from the error metric dividing by an all-zero result.

### The instruction sequence is correct on v73

Booting the SDK's QuRT image for v73 under `hexagon-sim` and running the kernel's exact
sequence -- `bias = mxmem2` / `mxclracc.hf` / `{activation.hf ; weight.hf}` /
`mxmem(...):after.hf = acc` -- on a tile of 1.0 times a tile of 1.0:

| dot tiles | expected | simulated v73 |
|---:|---:|---|
| 1 | 32 | 32.00, 1024/1024 |
| 2 | 64 | 64.00, 1024/1024 |
| 4 | 128 | 128.00, 1024/1024 |
| 8 (what the device uses) | 256 | 256.00, 1024/1024 |

Qualcomm's `hmx_hexagon_protos.h` agrees: every `.hf` HMX intrinsic sits outside all
`__HMX_ARCH__` guards, so from v68 up. What `__HMX_ARCH__ >= 73` adds is the split store
`cvt.hf=acc` / `mxmem(Rs,Rt)=cvt`.

Feeding that same known-good data (1.0 x 1.0) through the *device's* kernel still returns
zero, so it is not the operands or their layout either.

### What has been ruled out, by measurement

| hypothesis | how it died |
|---|---|
| silicon / architecture generation | simulator computes correctly on v73 |
| resource lock not taken | `HAP_compute_res_hmx_lock` returns 0, 29,265 times |
| weak symbol resolving to a false success | the wrapper returns `0x80000404` when absent; we get 0 |
| wrong lock generation | `hmx_lock3`, `hmx_lock4`, `query_capability` all NOT_SUPPORTED here |
| HMX clock never voted | `HAP_power_set_HMX_v2` is rejected (`0x80000414`); v1 succeeds |
| VTCM pressure | 43 KB used of 8 MB |

Also measured: `qurt_hmx_lock` is not exported into this process domain at all (the backend
requests an unsigned PD), and a hard reference to it stops the skel from loading.

**Not yet tried**: testsig / a signed PD, PMU counters to see whether the unit executes at
all, the ADSP domain, variations of the v1 `HAP_power_set_HMX` parameters.

### Two earlier readings in this file were wrong

Both are kept because a conclusion that closes a door is worse than no conclusion.

**"v73 cannot execute fp16 HMX."** See above.

**"Zeroing every HMX input still yields inf, so the unit does not compute."** That run
zeroed the bias too, and a zero scale is not the documented identity, so the output was
undefined by construction. The load asm also carried no `"memory"` clobber, so the compiler
was free to move the zeroing past the HMX reads.

### Change in this branch

`ggml-hexagon.cpp` + `htp/main.c`: decide by running it. During `htp_iface_start` the DSP
multiplies a tile of 1.0 by a tile of 1.0 and only reports HMX as available if the answer
is 32.0; `htp_iface_hwinfo()` returns that verdict and the host re-reads it after start.
Parts where HMX works keep it, including v73 ones.

Two placement mistakes worth knowing: the test cannot run on the HMX queue thread at
session start (the lock is not available yet, the instructions fault, and the session dies
with `0x8000040d`), and it must be wrapped in `vtcm_acquire()`/`vtcm_release()` or the lock
fails with `failed 1, holder 0x0`. A test that could not run must not be reported as a test
that failed.

The `opt_arch >= 75` gate stays as a fallback for when the self test cannot run.

### Getting DSP logs (both halves are needed)

1. build with `-DGGML_HEXAGON_HTP_DEBUG=ON` (defines `FARF_HIGH=1`)
2. put a `<process-name>.farf` file containing `0x001f001f001f001f` on `ADSP_LIBRARY_PATH`

FastRPC's `log_config` then enables `adspmsgd` and DSP messages appear in `logcat`. Trying
either half alone produces nothing, which is why this looked impossible.

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
