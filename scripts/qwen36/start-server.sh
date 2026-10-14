#!/bin/bash
# Qwen3.6-35B-A3B Q8_0 (MTP) on RTX 3060 12 GB + 32 GB RAM, tuned for agentic coding.
# 128K context, fast prompt processing (ubatch 4096/2048, expert streaming, CPU+GPU experts for mid-size chunks),
# n-gram + MTP speculative decoding, expert prefetch during GPU work, KV cache and vision projector memory taken as
# they are needed.
#
# For the full 262144 context use instead:  -c 262144 -ub 512 -b 2048 -ncmoe 34  and  GGML_SCHED_STREAM_MIN_BATCH=512
# (measured before the CPU+GPU expert split: prompt processing ~40% slower, less free RAM)
#
# env: LLAMA_MODELS_DIR (default: models/ in the repo), LLAMA_API_KEY (optional)

cd "$(dirname "$0")/../.." || exit 1

# the MTP draft head only attends to the last 16K tokens, saves VRAM, does not change the output
export LLAMA_MTP_SWA=16384

# batches shorter than this many tokens run the experts of the RAM-resident layers on the CPU only
export GGML_OP_OFFLOAD_MIN_BATCH=64

# batches of 64-768 tokens split the experts of each RAM-resident layer: the CPU computes the share
# 465/(465 + tokens) of them while the GPU uploads and computes the rest at the same time
export LLAMA_MOE_HYBRID=465
export LLAMA_MOE_HYBRID_MAX=768

# while the GPU computes the attention of a layer, helper threads on the SMT siblings (CPUs 16-31) of the
# compute threads (pinned to CPUs 0-15 below) load the 8 experts the router will most likely pick next
export LLAMA_MOE_PREFETCH=8
export LLAMA_MOE_PREFETCH_CPUS=16-31

# prompt processing: a second CUDA stream uploads the experts of the next layers while the GPU computes,
# into 3 slots of 272 MiB while VRAM is spare, 2 once the KV cache needs the memory
# (batches of 768 tokens or more, smaller ones upload only the experts they use)
export GGML_SCHED_STREAM_WEIGHTS=3
export GGML_SCHED_STREAM_MIN_BATCH=768

# the attention computes the causal mask from the token positions instead of reading a mask of
# n_ctx x ubatch (512 MiB of VRAM and 512 MiB of pinned RAM at 128K context and ubatch 2048)
export LLAMA_FA_POS_MASK=1

# the KV cache only reserves addresses at start, VRAM is taken as the conversation grows
export LLAMA_KV_GROW=1

# the MTP draft head processes prompts in chunks of 512 tokens, saves VRAM
export LLAMA_MTP_UBATCH=512

# a vision projector (--mmproj) takes its VRAM with the first image, not at start
export LLAMA_MMPROJ_LAZY=1

# micro-batches of 4096 tokens (-ub below) while VRAM is spare, 2048 once the KV cache or an image needs it
export LLAMA_UBATCH_MIN=2048

args=()
[ -n "$LLAMA_API_KEY" ] && args+=(--api-key "$LLAMA_API_KEY")

exec ./build/bin/llama-server \
    --models-dir "${LLAMA_MODELS_DIR:-models}" \
    -c 131072 -ub 4096 -b 4096 -ncmoe 34 \
    -ctk q8_0 -ctv q4_0 -ctkd f16 -ctvd f16 \
    -fa on -lm dio --lazy-mode on -np 1 \
    -t 16 -C 0xFFFF --cpu-strict 1 -Cb 0xFFFF --cpu-strict-batch 1 \
    --cache-ram 0 --ctx-checkpoints 2 \
    --no-models-autoload --jinja \
    --spec-type ngram-mod,draft-mtp --spec-draft-n-max 2 \
    --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 4 --spec-ngram-mod-n-max 30 \
    --host 0.0.0.0 --port 8080 "${args[@]}" "$@"
