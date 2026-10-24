# Uniform-grid RoPE measurements

GPU: NVIDIA GeForce RTX 4060 Laptop (8 GiB).
Seed 123; deterministic random block weights; no checkpoint or generated-video benchmark.
Every entry is a median of 12 shuffled rounds, two calls per sample, after three warmup calls.
Both original and patched compiled variants use default torch.compile with dynamic=False.

## WSL2 Linux / FlashAttention

PyTorch 2.8.0+cu128, CUDA 12.8. FlashAttention: 2.8.3.post2.

| Case | Original eager, ms | Patched eager, ms | Original compiled, ms | Patched compiled, ms | Compiled speedup |
|---|---:|---:|---:|---:|---:|
| RoPE, 3×30×52, FP16 | 2.651 | 2.680 | 1.619 | 1.281 | 1.264x |
| RoPE, 3×30×52, BF16 | 2.314 | 2.356 | 1.409 | 0.992 | 1.421x |
| RoPE, 3×30×52, FP32 | 2.392 | 2.429 | 1.494 | 1.137 | 1.314x |
| RoPE, 21×30×52, FP16 | 15.885 | 15.937 | 8.977 | 5.376 | 1.670x |
| RoPE, 21×30×52, BF16 | 15.880 | 15.880 | 8.916 | 5.193 | 1.717x |
| RoPE, 21×30×52, FP32 | 16.345 | 16.348 | 9.434 | 5.682 | 1.660x |
| Upstream DiT block, 3×30×52 | 31.799 | 31.912 | 24.888 | 23.766 | 1.047x |

Maximum absolute error, patched compiled vs original compiled: 0.

## Windows / SDPA

PyTorch 2.10.0+cu128, CUDA 12.8. FlashAttention: not used.

| Case | Original eager, ms | Patched eager, ms | Original compiled, ms | Patched compiled, ms | Compiled speedup |
|---|---:|---:|---:|---:|---:|
| RoPE, 3×30×52, FP16 | 2.285 | 2.288 | 1.369 | 0.991 | 1.381x |
| RoPE, 3×30×52, BF16 | 2.338 | 2.356 | 1.487 | 1.121 | 1.327x |
| RoPE, 3×30×52, FP32 | 2.343 | 2.334 | 1.393 | 1.006 | 1.384x |
| RoPE, 21×30×52, FP16 | 15.847 | 15.867 | 8.887 | 5.101 | 1.742x |
| RoPE, 21×30×52, BF16 | 15.853 | 15.871 | 8.894 | 5.116 | 1.738x |
| RoPE, 21×30×52, FP32 | 16.277 | 16.290 | 9.343 | 5.620 | 1.662x |
| Upstream DiT block, 3×30×52 | 34.384 | 34.418 | 27.278 | 26.195 | 1.041x |

Maximum absolute error, patched compiled vs original compiled: 0.

The eager path retains the original implementation. The new real-valued path is selected only during compilation for uniform grids.
Heterogeneous batches retain the original per-sample path. Complex frequency tables, padding and FP32 outputs are preserved.

Both numerical variants use the same disabled host-metadata read so PyTorch 2.8 can compile the tensor work. The unmodified Tensor.tolist path failed with aten._local_scalar_dense and fell back to eager; those preliminary runs are not presented as compiled timings.

The block test instantiates the real upstream WanAttentionBlock: 1.3B dimensions for Wan2.1, 5B dimensions for Wan2.2. Grid (3,30,52) is a 3-frame latent chunk. Isolated RoPE also tests grid (21,30,52).
Windows block results explicitly substitute SDPA for FlashAttention in BOTH variants; the production attention source is unchanged.

FP32-output differences between eager and compiled blocks are recorded separately in raw JSON. They must not be confused with differences caused by this patch.
CUDA peaks in raw JSON include all live inputs, two blocks and compiler allocations, not just temporary RoPE workspace.
Single GPU only. No end-to-end video, pretrained-quality, distributed or older-PyTorch claim is made.

[Raw samples, errors, graph counts and memory](https://gist.github.com/TieBeTie/21d8034aff8791232c221061b7cd8a6e#file-wan21_benchmark-json)

```bash
python -m pytest tests/test_rope.py -q
python tests/benchmark_rope.py --attention flash --output /tmp/rope_benchmark.json
```

## Why it can be faster

The rotation is unchanged: `(a + ib)(c + is) = (ac - bs) + i(bc + as)`. Real arithmetic exposes the promotion, phase indexing and multiply/add/subtract to Inductor fusion. In this capture the separate complex phase concatenation and complex multiply calls disappear, and the input promotion is fused into the arithmetic kernel instead of materializing a full FP64 input buffer. There are still **two Triton kernels**, not one fully fused kernel.

<details>
<summary>Actual generated CUDA launch/Triton code, before and after</summary>

PyTorch 2.8.0+cu128, RTX 4060 Laptop, FP32 input `(1,32760,12,128)`. These are selected lines from the real compiler output; intervening views, checks and allocations are omitted. They are generated CUDA launch wrappers and Triton, not handwritten CUDA C++.

**Before, CUDA launch wrapper:**

```python
triton_poi_fused_view_as_complex_0.run(arg0_1, buf0, 50319360, stream=stream0)
buf21 = torch.ops.aten.cat.default([buf8, buf14, buf20], -1)
buf25 = torch.ops.aten.mul.Tensor(buf2, buf24)
triton_poi_fused__to_copy_1.run(buf28, buf29, 50319360, stream=stream0)
```

**After, CUDA launch wrapper:**

```python
triton_poi_fused_add_mul_sub_0.run(arg0_1, buf7, buf15, buf23, buf25, buf27, buf29, buf30, buf31, 25159680, stream=stream0)
triton_poi_fused__to_copy_1.run(buf30, buf31, buf32, 50319360, stream=stream0)
```

**Fused real arithmetic excerpt:**

```python
tmp19 = tmp1 * tmp18
tmp27 = tmp21 * tmp26
tmp28 = tmp19 - tmp27
tmp29 = tmp21 * tmp18
tmp30 = tmp1 * tmp26
tmp31 = tmp29 + tmp30
tl.store(out_ptr0 + (x3), tmp28, xmask)
tl.store(out_ptr1 + (x3), tmp31, xmask)
tl.store(out_ptr0 + (x2), tmp11, None)
```

[Full generated source and capture metadata](https://gist.github.com/TieBeTie/21d8034aff8791232c221061b7cd8a6e). Reproduce with `python tests/dump_rope_codegen.py --output /tmp/rope_codegen`. Names and schedules may differ across versions/shapes.

</details>

