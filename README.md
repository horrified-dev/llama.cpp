# This fork: Qwen3.6-35B-A3B Q8_0 on a 12 GB GPU

This fork runs **Qwen3.6-35B-A3B at Q8_0** (35.8 GiB of weights, plus its built-in MTP draft head) with a **128K or full
262K context** on an ordinary desktop, as the backend of an agentic coding harness:

- AMD Ryzen 9 5950X (16 cores / 32 threads)
- 32 GB DDR4-3200 (about 45 GB/s of read bandwidth)
- NVIDIA RTX 3060 12 GB on PCIe 4.0
- NVMe SSD

The model does not fit in VRAM + RAM with room to spare, so every byte and every millisecond is placed deliberately.
The changes are specific to this model (the `qwen35moe` architecture) and this kind of machine. Everything else is
upstream llama.cpp; its README follows [further below](#llamacpp).

| | Stock llama.cpp, best flags | This fork |
|---|---|---|
| Decode, writing new code (greedy) | 30.6 tok/s | **42.1 tok/s** (+38%) |
| Decode, writing new code (default sampling) | 30.6 tok/s | **39.0 tok/s** (+27%) |
| Decode, rewriting existing code | 30.6 tok/s | **108.3 tok/s** (3.5x) |
| Attention per layer and token at 128K context | 2034 us | **389 us** (5.2x) |
| Prefill, 11.2K-token agentic turn | 608 tok/s (`-ub 1024`)* | **2029 tok/s** (3.3x: expert streaming, `-ub 4096`, new kernels) |
| Prefill, 112K-token prompt | 511 tok/s* | **1260 tok/s** |
| Prefill, 64-token agentic turn | 0.49 s | **0.34 s** |
| Context | 128K | **128K** (default profile) or **262,144** |
| VRAM used | ~11.0 of 12 GB, from the start | **10.1 GB at start**, grows with the conversation to ~11.3 GB at 112K |

\* prefill "before" is this fork as of 2026-10-13 with `-ub 1024`; stock llama.cpp at its default `-ub 512` is slower.

"Stock, best flags" is upstream llama.cpp with the best settings found for this machine (`-ncmoe 33 -fa on -ctk q8_0
-ctv q8_0 -lm dio -t 16`, 128K context) and no speculative decoding. Decode numbers are measured through `llama-server`
at 128K context; they vary by about 1-2 tok/s between runs.

## Benchmarks

### Decode

Writing new text is limited by how fast the CPU can read the routed experts from RAM. The MTP draft head (shipped in
the GGUF) proposes 2 tokens per step and the main model verifies them in one batch; expert prefetch then hides part
of the remaining memory traffic behind GPU work.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/qwen36/decode-new-code-dark.svg">
  <img alt="Decode speed writing new code: stock 30.6 tok/s; with MTP 39.1 greedy, 36.7 sampled; with n-gram drafts 39.0 / 37.4; with expert prefetch 42.1 / 39.0" src="docs/qwen36/decode-new-code-light.svg">
</picture>

Agentic coding re-emits a lot of existing code (edits, rewrites, quoting). n-gram drafts find the continuation in
the context and propose up to 30 tokens at once, which the GPU verifies nearly for free.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/qwen36/decode-rewrite-dark.svg">
  <img alt="Decode speed rewriting existing code: stock 30.6, MTP 40.3, n-gram 106.9, expert prefetch 108.3 tok/s" src="docs/qwen36/decode-rewrite-light.svg">
</picture>

### Attention at long context

Stock llama.cpp runs single-token attention over a quantized KV cache on CUDA cores. This fork runs it on the tensor
cores, which keeps decode fast deep into the context.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/qwen36/fa-kernel-dark.svg">
  <img alt="Attention time per layer for one token: stock 273, 1027, 2034 us at 16K, 64K, 128K; this fork 63, 202, 389 us" src="docs/qwen36/fa-kernel-light.svg">
</picture>

| KV cache depth | Stock, 1 token | Fork, 1 token | Stock, 3 tokens (MTP verify) | Fork, 3 tokens |
|---|---|---|---|---|
| 16K | 273 us | 63 us | 332 us | 90 us |
| 64K | 1027 us | 202 us | 1267 us | 308 us |
| 128K | 2034 us | 389 us | 2325 us | 600 us |

### Prefill

With 34 of 40 layers' experts in system RAM, a large prompt is processed by streaming those experts over PCIe to the
GPU, once per micro-batch: about 1.1 s at 25.8 GB/s for each pass over the 34 layers. Profiling showed that the GPU
used to alternate between uploading (63% of the time) and computing (30%), never both at once. The fork now uploads
the experts of the next layers on a second CUDA stream while the current layer computes, takes VRAM only when it is
needed (KV cache, vision projector), and spends the rest on micro-batches of 4096 tokens. With the uploads hidden, the
GPU's own kernels became the limit, and four of them were rewritten (below). Measured with Huihui-Qwen3.6-35B-A3B
Q8_0, the vision projector and MTP loaded:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/qwen36/prefill-before-after-dark.svg">
  <img alt="Prefill in three rounds: 3.5K turn 538, 1085, 1678; 11.2K turn 608, 1128, 2029; 16.8K prompt 613, 1309, 1942; 112K prompt 511, 914, 1260 tok/s" src="docs/qwen36/prefill-before-after-light.svg">
</picture>

| Same prompts, same session | 2026-10-13 (`-ub 1024`) | Now (`-ub 4096`) |
|---|---|---|
| VRAM at start / peak at 112K context | 11.53 / 11.62 GB | **10.11 / 11.25 GB** |
| Lowest free RAM during the runs | 442 MiB | **475 MiB** (with `--ctx-checkpoints 2`) |
| Output | | KL divergence 0.0058 vs. before, the noise floor of any change in rounding on this model; perplexity unchanged |

Where the GPU time of a 4096-token micro-batch went once the uploads were hidden (1.6 s of compute against 1.0 s of
uploads): about 16 ms per layer in the expert matmuls, and in the rest of a Gated DeltaNet layer, four kernels that
were far from the hardware limits:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/qwen36/prefill-kernels-dark.svg">
  <img alt="Kernel time before vs now: Gated DeltaNet 7.92 vs 4.17 ms, conv input concatenation 1.61 vs 0.88 ms, RMS norm 0.40 vs 0.22 ms, attention 6.79 vs 6.11 ms" src="docs/qwen36/prefill-kernels-light.svg">
</picture>

How the micro-batch size scales, before the streaming change:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/qwen36/prefill-ubatch-dark.svg">
  <img alt="Prefill speed: -ub 512 487 tok/s, -ub 1024 761 tok/s, -ub 2048 1135 tok/s" src="docs/qwen36/prefill-ubatch-light.svg">
</picture>

Short turns make up most of an agentic session (a tool result of a few dozen to a few hundred tokens). Only the
experts a chunk routes to are uploaded, but even a short chunk routes to many different experts in each of the 33
layers, 3.3 MB each, so the GPU path never gets much faster than ~0.35 s, and below 192 tokens running the experts on
the CPU was faster:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/qwen36/prompt-offload-dark.svg">
  <img alt="Prompt time GPU vs CPU experts: 32 tokens 0.354 vs 0.229 s, 64 0.489 vs 0.344, 128 0.641 vs 0.534, 192 0.722 vs 0.704, 256 0.808 vs 0.867, 512 1.010 vs 1.694, 1024 1.315 vs 3.172 s" src="docs/qwen36/prompt-offload-light.svg">
</picture>

Neither path uses the whole machine: the GPU path waits for PCIe while the CPU idles, and the CPU path is bound by
DDR4 while the GPU idles. For chunks of 64 to 768 tokens the fork now splits the experts of each RAM-resident layer:
the CPU computes the experts [0, k0) while the GPU uploads and computes the rest, with the CPU share 465/(465 + tokens):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/qwen36/prompt-chunks-dark.svg">
  <img alt="Chunk time GPU path vs CPU+GPU split: 0.57 vs 0.48 s at about 115 tokens, 0.92 vs 0.69 s at about 220, 1.25 vs 0.97 s at about 500, equal at 1400 tokens" src="docs/qwen36/prompt-chunks-light.svg">
</picture>

Agentic multi-turn test (`scripts/qwen36/bench_multiturn.py`: two large tool outputs, then a short question):

| Profile | Turn 1 (11.2K new tokens) | Turn 2 (3.5K new tokens) | Turn 3 |
|---|---|---|---|
| 128K, `-ub 4096`, CPU+GPU experts, new kernels | 2029 tok/s | 1678 tok/s | ~25 new tokens, rest from cache |
| 128K, `-ub 2048`, expert streaming | 1128 tok/s | 1085 tok/s | ~25 new tokens, rest from cache |
| 128K, `-ub 1024` (before) | 608 tok/s | 538 tok/s | ~25 new tokens, rest from cache |
| 262K, `-ub 512` (before) | 377 tok/s | 351 tok/s | ~25 new tokens, rest from cache |

## What this fork does differently

### Where everything lives

| Tier | Contents |
|---|---|
| VRAM (12 GB, ~11.3 GB used) | all non-expert weights of the 40 layers (attention, Gated DeltaNet, shared experts, routers), the routed experts of the last 7 layers, the MTP head, KV cache and recurrent state, compute buffers |
| RAM (32 GB) | the routed experts of the first 33 layers (~28 GiB), read once at startup with direct I/O and kept resident |
| SSD | the token embedding table (0.5 GB), read on demand |

Per decoded token, the CPU reads 8 of 256 experts in each of 33 layers (about 0.9 GB), which makes DDR4 bandwidth the
limit. A decode step with MTP (about 66 ms for 2-3 tokens) spends ~46 ms in the CPU experts, ~16 ms on the GPU,
~4.6 ms drafting and ~1.3 ms in copies.

### Code changes

1. **Expert prefetch** (`src/llama-moe-prefetch.{h,cpp}`, hooks in `src/llama-graph.{h,cpp}`,
   `src/models/qwen35moe.cpp`). While the GPU computes the attention of layer L+1, the CPU cores would sit idle. The
   fork predicts which experts layer L+1 will route to, by applying layer L+1's router (folded with the ratio of the
   two layers' norm weights, kept as a Q8_0 copy in RAM) to layer L's expert input. On average 84% of the 8 predicted
   experts are among the ones actually chosen. Helper threads pinned on the SMT siblings of the compute threads then load exactly the
   rows each compute thread will multiply, and stop as soon as the real routing is known. The output is bit-identical
   with prefetch on and off (KL divergence 0). Single-token decode goes from 30.6 to 34.4 tok/s (+12%); with
   speculative decoding the gain is +8% greedy, +4% sampled and +1% when rewriting code.
2. **Tensor-core flash attention for a quantized KV cache** (`ggml/src/ggml-cuda/fattn-mma-f16.cuh`,
   `ggml/src/ggml-cuda/fattn.cu`, two new template instances). The MMA kernel can now read q8_0 K and q8_0/q4_0 V
   directly, dequantizing each tile to f16 as it loads it into shared memory. Stock llama.cpp either uses the slower
   vector kernel or first converts the whole cache to f16 in a scratch buffer. Enabled for Turing and newer, head size
   256 and a GQA ratio that is a multiple of 8 (Qwen3.6 has 16 query heads and 2 KV heads). Perplexity is unchanged
   and the `test-backend-ops` flash attention suite passes.
3. **Sliding-window KV for the MTP draft head** (`src/llama-model.cpp`, `LLAMA_MTP_SWA=<tokens>`). The draft head has
   its own KV cache, which at 128K-262K context costs VRAM that is better spent on the main model. The 16K window only
   affects the drafts, never the output, because the main model verifies every drafted token.
4. **Lazy token embedding** (`src/models/qwen35moe.cpp`). With `--lazy-mode on` the 0.5 GB embedding table stays on
   the SSD and only the rows of tokens that occur are read, leaving that RAM for experts.
5. **Static work split in the CPU `mul_mat_id` for batches of up to 4 tokens** (`ggml/src/ggml-cpu/ggml-cpu.c`). Each
   thread always processes the same rows of an expert, so the rows a prefetch helper loaded are used by the core next
   to it. This alone is ~4% faster for the 3-token batches of MTP verification.
6. **Expert streaming** (`ggml/src/ggml-backend.cpp`, `GGML_SCHED_STREAM_WEIGHTS=<slots>`). The scheduler places the
   device copies of host weights in a ring of slots instead of the compute buffer, and for batches of at least 512
   tokens a second instance of the CUDA backend, with its own stream, uploads the experts of the next splits while the
   current split computes. CUDA events order the uploads and the compute. Batches of 512+ tokens use almost every
   expert anyway, so the whole expert tensor is uploaded without waiting for the router.
7. **Positional causal mask** (`ggml/src/ggml.c`, `ggml/src/ggml-cuda/fattn*`, `ggml/src/ggml-cpu/ops.cpp`,
   `LLAMA_FA_POS_MASK=1`). Flash attention normally reads an F16 mask of n_kv x n_batch, built on the CPU in pinned
   memory, uploaded and kept in VRAM: 512 MiB each at 128K context and `-ub 2048`. With an I32 "mask" that holds the
   positions of the KV cells and of the queries (and the 2D positions of image tokens), the quantized-KV MMA kernel
   computes the mask itself, and the CPU implementation does the same for testing. This memory is what pays for the
   bigger micro-batches.
8. **VRAM as you go** (`src/llama-kv-cache.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu`, `LLAMA_KV_GROW=1`). The KV cache
   reserves only CUDA virtual addresses at startup and maps zeroed device memory in 2 MiB pieces as the conversation
   grows (and releases it when the cache is cleared). The tensors keep their addresses, so no kernel changes. If
   another program takes the VRAM in between, a long conversation can run out of memory later; the request then fails
   instead of the server.
9. **Spare VRAM for speed** (`src/llama-context.cpp`). While the KV cache is small, a third streaming slot lets the
   uploads run further ahead (+13% prefill); it is released when less than 512 MiB of VRAM is free. The MTP draft
   context processes prompts in chunks of 512 tokens (`LLAMA_MTP_UBATCH`) to keep its compute buffers small.
10. **Vision projector on demand** (`tools/mtmd/clip.cpp`, `tools/mtmd/mtmd.{h,cpp}`,
   `tools/server/server-context.cpp`, `LLAMA_MMPROJ_LAZY=1`). The projector's 1.1 GB of VRAM is loaded with the first
   image instead of at startup; before encoding, the server asks the context (`llama_make_device_memory`) to make room.
11. **Memory ladder** (`src/llama-context.cpp`, `LLAMA_UBATCH_MIN`). Micro-batches of 4096 tokens while VRAM is spare.
   Before each micro-batch the context checks the free VRAM: below 512 MiB it re-reserves the graph for 2048-token
   micro-batches, then releases the third streaming slot, and it climbs back when the memory is free again.
12. **CPU and GPU on the same layer** (`ggml_mul_mat_id_set_range` in `ggml/src/ggml.c`, `ggml/src/ggml-cpu/ggml-cpu.c`,
   `ggml/src/ggml-cuda/mmq.cu`, `mmid.cu`, `quantize.cu`; `src/llama-graph.cpp`; `LLAMA_MOE_HYBRID`). `mul_mat_id` can
   be restricted to a range of experts. For chunks of 64-768 tokens, the graph runs the experts [0, k0) of each
   RAM-resident layer on the CPU and [k0, 256) on the GPU at the same time and adds the two halves: 7-18% faster for
   typical tool outputs (16-23% together with the kernels of item 13), with the same KL divergence as the stock CPU
   path.
13. **Prompt-processing kernels** (`ggml/src/ggml-cuda/gated_delta_net.cu`, `concat.cu`, `norm.cu`,
   `fattn-mma-f16.cuh`). The Gated DeltaNet recurrence stages 8 tokens at a time in shared memory, keeps two state
   columns per lane and computes S^T k and S^T q in one reduction (1.9x). The concatenation of the conv state with the
   transposed conv input transposes in 32 x 32 tiles (1.8x). RMS norms of short rows use one warp per row in large
   batches (1.8x). The quantized-KV attention dequantizes by placing the integers in the mantissa of 1024.0h, which is
   bit-identical and 11% faster. Optionally (`GGML_CUDA_FA_F16_MIN_BATCH`), a long quantized KV range is converted to
   f16 in pool memory for the faster f16 kernel: -4% time on a 112K prompt for up to ~250 MB of VRAM, off by default.

### Configuration (upstream features, tuned for this machine)

- **`-ncmoe 33`** (34 in the current profile, see below): keep the routed experts of 33 of the 40 layers in RAM;
  everything else fills the GPU to ~94%.
- **`-lm dio`** instead of mmap: with mmap, at 128K context the mapped weights no longer fit next to everything else
  and decode collapses to 4.5 tok/s from disk thrashing. Direct I/O also cuts tensor loading from 146 s to 20 s
  (about 34 s from start to ready).
- **Speculative decoding, two drafters at once**: the MTP head with 2 draft tokens for new text (1 token: 36.3, 2:
  38.7, 3: 36.3 tok/s greedy) and `ngram-mod` (match 24 tokens, draft 4-30) for text that repeats the context.
- **`-ub 4096 -b 4096 -ncmoe 34 -ctv q4_0`**: the 4096-token micro-batches were measured with one more layer of
  experts in RAM and the smaller V cache, which leave them room; the previous `-ub 2048` profile ran with
  `-ncmoe 33 -ctv q8_0`.
- **`GGML_OP_OFFLOAD_MIN_BATCH=64`, `LLAMA_MOE_HYBRID=465`, `LLAMA_MOE_HYBRID_MAX=768`,
  `GGML_SCHED_STREAM_MIN_BATCH=768`**: below 64 tokens the experts run on the CPU, from 64 to 768 tokens the CPU and the
  GPU share them, above that whole layers are streamed to the GPU (see the charts above).
- **Thread pinning**: 16 compute threads on the physical cores 0-15 (`-C 0xFFFF --cpu-strict 1`), the prefetch
  helpers on their SMT siblings 16-31.
- **Quantized KV cache** (`-ctk q8_0 -ctv q4_0`), `--cache-ram 0` (RAM is the scarcest resource) and
  `--ctx-checkpoints 2`, so follow-up turns reuse the cache: each checkpoint keeps 60-90 MB of recurrent state in RAM,
  and with 4 of them a long benchmark session left less than 400 MiB of RAM free.

Tried and dropped: confidence-gated MTP drafting (higher acceptance, but fewer tokens per second) and longer MTP
drafts (3 to 6 tokens).

## Running it

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q8_0-q4_0"
cmake --build build -j --target llama-server

# put Qwen3.6-35B-A3B-Q8_0-MTP.gguf (unsloth) into models/, then:
LLAMA_API_KEY=... scripts/qwen36/start-server.sh

# reproduce the decode and prefill numbers against the running server
scripts/qwen36/bench.sh
```

`scripts/qwen36/start-server.sh` is the 128K profile. For the full 262K context use
`-c 262144 -ub 512 -ncmoe 34 -ctv q4_0`. The charts are generated by `scripts/qwen36/make_charts.py`. For the full story, see the write-up
[Bytes, Not FLOPs](docs/qwen36/paper.md).

| Environment variable | Meaning |
|---|---|
| `LLAMA_MOE_PREFETCH=<n>` | prefetch the `n` most likely experts of the next layer (off by default; 8 recommended) |
| `LLAMA_MOE_PREFETCH_CPUS=<a-b>` | CPUs for the prefetch helper threads (the SMT siblings of the compute threads) |
| `LLAMA_MOE_PREFETCH_US=<us>` | upper bound for one prefetch, default 1000 |
| `LLAMA_MOE_PREFETCH_MAX_TOKENS=<n>` | only prefetch for batches of up to `n` tokens, default 4 |
| `LLAMA_MTP_SWA=<tokens>` | sliding window of the MTP draft head's KV cache |
| `LLAMA_MTP_UBATCH=<n>` | micro-batch size of the MTP draft context (prompt processing), saves its compute buffers |
| `GGML_SCHED_STREAM_WEIGHTS=<slots>` | stream host experts to the GPU on a second CUDA stream through `slots` slots (2-4, 3 recommended); more than 2 only while VRAM is spare |
| `GGML_SCHED_STREAM_MIN_BATCH=<n>` | smallest batch that streams whole expert tensors, default 512 (768 with `LLAMA_MOE_HYBRID`) |
| `LLAMA_FA_POS_MASK=1` | positional causal mask instead of the F16 KQ mask (q8_0 K, q8_0/q4_0 V, one sequence) |
| `LLAMA_KV_GROW=1` | the KV cache takes VRAM as the conversation grows |
| `LLAMA_MMPROJ_LAZY=1` | load the vision projector with the first image |
| `LLAMA_UBATCH_MIN=<n>` | micro-batch size to fall back to when VRAM runs short (`-ub` is the size while VRAM is spare) |
| `LLAMA_MOE_HYBRID=<c>` | split the experts of RAM-resident layers between CPU and GPU, CPU share c/(c + tokens) (465 recommended) |
| `LLAMA_MOE_HYBRID_MAX=<n>` | largest batch that is split, default 2048 (768 recommended); the smallest is `GGML_OP_OFFLOAD_MIN_BATCH` |
| `GGML_CUDA_GDN_PREFILL=0` | disable the shared-memory Gated DeltaNet kernel for prompt processing |
| `GGML_CUDA_FA_F16_MIN_BATCH=<n>` | batches of `n`+ tokens over at least `GGML_CUDA_FA_F16_MIN_KV` (32768) cells convert a quantized KV range to f16 in pool memory if `GGML_CUDA_FA_F16_MIN_FREE_MIB` (512) stay free (off by default) |
| `GGML_OP_OFFLOAD_MIN_BATCH=<n>` | (upstream) smallest batch for which CPU weights are streamed to the GPU |

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
