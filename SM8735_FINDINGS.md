# SM8735 (Adreno 825 / Hexagon v73) — measurements

Working notes from running llama.cpp on a POCO F7 (Snapdragon 8s Gen 4, SM8735).
Everything here was measured on that device. This is a scratch branch on a personal
fork, not a proposed upstream change.

Device: Adreno 825, Hexagon arch v73, 11 GiB RAM, Android/Termux.
Build: `cmake --preset arm64-android-snapdragon-release`, `ghcr.io/snapdragon-toolchain/arm64-android:v0.7`.

---

## 1. Hexagon: the fp16 HMX path is wrong on this v73 part (cause still open)

`test-backend-ops test -b HTP0 -o MUL_MAT` -> **514 pass / 44 fail**, every failure `ERR = inf`.

Correlating the kernel path from `GGML_HEXAGON_VERBOSE=1` against pass/fail over the
whole sweep gives an exact 1:1:

| | count |
|---|---:|
| ops that took `hmx-tiled` | **42** |
| ops that failed with `ERR = inf` | **42** |
| any other failure | 0 |

The 42 include two `f16` ops, so this is not specific to the repacked quantized types.

`GGML_HEXAGON_NHMX=0` (or `GGML_HEXAGON_MM_SELECT<=2`) -> **554/554 pass**, and real
model output becomes correct. Failures begin exactly at `m > HTP_MM_HMX_MIN_NROWS` (= 4),
i.e. as soon as the HMX path is selected, which is why 1B models appear to work while
larger ones emit garbage.

### Two readings of this that were wrong

Both are recorded because both would close the door on a fixable bug.

**"v73 cannot execute fp16 HMX."** Qualcomm's own `hmx_hexagon_protos.h` (Hexagon SDK
6.6.0.0, tools 19.0.07) places every `.hf` HMX intrinsic -- `mxclracc.hf`,
`activation.hf=mxmem`, `weight.hf=mxmem`, `mxmem(Rs,Rt):after.hf=acc`, `mxswapacc.hf` --
**outside every `__HMX_ARCH__` guard**, i.e. available from v68 up. The intrinsics that
are gated at `__HMX_ARCH__ >= 73` are the *split* output path (`cvt.hf=acc(Rs)` then
`mxmem(Rs,Rt)=cvt`) and the 4-bit `weight.n` loads. So fp16 HMX is not a v75 feature, and
v73 is the version that *adds* the split store.

**"Zeroing every HMX input still yields inf, so the unit does not compute."** That
experiment zeroed the bias along with the activations and weights, but the documented
identity bias is 32 words of `0x3c00` followed by 32 zero words (which is exactly what
`hmx_init_column_scales(..., Q6_V_vsplat_R(0x3c00))` writes in normal operation). A zero
scale is not identity, so the output was undefined by construction. Separately, the
activation/weight/bias load asm blocks carry no `"memory"` clobber, so the compiler was
free to move the zeroing past the HMX reads; it is not established that the inputs were
zero when the unit read them. The experiment proves nothing either way.

`ERR = inf` from `test-backend-ops` is also not proof that the output tensor contains
infinities -- the error metric itself can overflow. The raw output values have not been
inspected yet.

### What does still hold

Executing `mxclracc.hf` with HMX not enabled raises exception `0x18`, identically on
simulated v73, v75 and v79, while HVX instructions in the same program execute fine
(hexagon-sim, standalone). The device does not raise `0x18`; it returns wrong values and
the DSP thread survives. `htp/hmx-queue.c` ignores the return of
`HAP_compute_res_hmx_lock()`, so a failed lock would have hit `0x18`. That points to the
unit being present and enabled -- but the lock's return value should be recorded rather
than inferred, and that has not been done yet.

Also note `htp/main.c: htp_iface_hwinfo()` sets `*n_hmx = 1` unconditionally without
querying anything.

### Change in this branch

`ggml-hexagon.cpp`: keep HMX off below v75. This is a workaround that is measured to
work, not a claim about the hardware; the comment at the gate says so. An explicit
`GGML_HEXAGON_NHMX` overrides it.

`htp/hmx-utils.h` + `hmx-mm-kernels-tiled.h`: `GGML_HEXAGON_NHMX` 2..6 select fp16 HMX
diagnostics on the DSP (the value already travels there as the `n_hmx` argument of
`htp_iface_start`). Mode 2 runs the real kernel with the v73 split store instead of the
combined `:after.hf` form -- they are separate encodings, so a part can implement one and
not the other. Modes 3..6 store the accumulator with no multiply at all to isolate the
clear and the store. The missing `"memory"` clobbers on the load asm are fixed.

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
