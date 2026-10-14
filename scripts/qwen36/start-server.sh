#!/bin/bash
# Qwen3.6-35B-A3B Q8_0 (MTP) on RTX 3060 12 GB + 32 GB RAM, tuned for agentic coding.
# 128K context, fast prompt processing (ubatch 2048, expert streaming), n-gram + MTP speculative decoding,
# expert prefetch during GPU work, KV cache memory taken as the conversation grows.
#
# For the full 262144 context use instead:  -c 262144 -ub 512 -ncmoe 34 -ctv q4_0
# (prompt processing ~40% slower, less free RAM)
#
# env: LLAMA_MODELS_DIR (default: models/ in the repo), LLAMA_API_KEY (optional)

cd "$(dirname "$0")/../.." || exit 1

# the MTP draft head only attends to the last 16K tokens, saves VRAM, does not change the output
export LLAMA_MTP_SWA=16384

# prompts shorter than this many tokens run their experts on the CPU, faster than copying them to the GPU
export GGML_OP_OFFLOAD_MIN_BATCH=192

# while the GPU computes the attention of a layer, helper threads on the SMT siblings (CPUs 16-31) of the
# compute threads (pinned to CPUs 0-15 below) load the 8 experts the router will most likely pick next
export LLAMA_MOE_PREFETCH=8
export LLAMA_MOE_PREFETCH_CPUS=16-31

# prompt processing: a second CUDA stream uploads the experts of the next layers while the GPU computes,
# into 3 slots of 272 MiB while VRAM is spare, 2 once the KV cache needs the memory
export GGML_SCHED_STREAM_WEIGHTS=3

# the attention computes the causal mask from the token positions instead of reading a mask of
# n_ctx x ubatch (512 MiB of VRAM and 512 MiB of pinned RAM at 128K context and ubatch 2048)
export LLAMA_FA_POS_MASK=1

# the KV cache only reserves addresses at start, VRAM is taken as the conversation grows
export LLAMA_KV_GROW=1

# the MTP draft head processes prompts in chunks of 512 tokens, saves VRAM
export LLAMA_MTP_UBATCH=512

args=()
[ -n "$LLAMA_API_KEY" ] && args+=(--api-key "$LLAMA_API_KEY")

exec ./build/bin/llama-server \
    --models-dir "${LLAMA_MODELS_DIR:-models}" \
    -c 131072 -ub 2048 -ncmoe 33 \
    -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 \
    -fa on -lm dio --lazy-mode on -np 1 \
    -t 16 -C 0xFFFF --cpu-strict 1 -Cb 0xFFFF --cpu-strict-batch 1 \
    --cache-ram 0 --ctx-checkpoints 4 \
    --no-models-autoload --jinja \
    --spec-type ngram-mod,draft-mtp --spec-draft-n-max 2 \
    --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 4 --spec-ngram-mod-n-max 30 \
    --host 0.0.0.0 --port 8080 "${args[@]}" "$@"
