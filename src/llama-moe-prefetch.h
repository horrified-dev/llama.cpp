#pragma once

#include "ggml.h"

#include <cstdint>
#include <vector>

// prefetch of the experts of host-resident MoE layers
//
// while the GPU computes the attention of a layer, the CPU threads that will compute its experts are idle and so is the memory bus
// helper threads, pinned next to the compute threads, use this time to load the rows of the experts that the router will likely select
// into the caches of the compute threads
//
// the experts of layer il+1 are predicted on the CPU right after the experts of layer il are computed, by applying the router of
// layer il+1 to the input of the experts of layer il (the post-attention residual of layer il, rescaled to the norm of layer il+1)
//
// enabled with LLAMA_MOE_PREFETCH=<number of experts per layer>, the helper threads are pinned to the CPUs in LLAMA_MOE_PREFETCH_CPUS,
// helper i prefetches the rows that compute thread i reads, so it should run on a CPU that shares the caches of compute thread i

struct llama_moe_prefetch_layer {
    std::vector<const ggml_tensor *> experts; // host expert tensors of the predicted layer
    int32_t n_threads;                        // number of CPU threads that compute the experts
    ggml_tensor * router;                     // [n_embd, n_expert] in host memory, router of the predicted layer times the norm ratio
};

// number of experts to prefetch per layer, 0 if prefetching is disabled
int32_t llama_moe_prefetch_n_expert();

// largest batch for which the experts are prefetched (LLAMA_MOE_PREFETCH_MAX_TOKENS, default 4)
int32_t llama_moe_prefetch_max_tokens();

// stable state to prefetch the experts of the next layer, nullptr if not possible
// router, norm_cur and norm_next are the router of the next layer and the post-attention norms of the current and next layers
llama_moe_prefetch_layer * llama_moe_prefetch_get_layer(
        const std::vector<const ggml_tensor *> & experts,
        int32_t n_threads,
        const ggml_tensor * router,
        const ggml_tensor * norm_cur,
        const ggml_tensor * norm_next);

// custom op: src[0] holds the router logits [n_expert, n_tokens]; starts the prefetch of the most likely experts and returns
void llama_moe_prefetch_kick(ggml_tensor * dst, int ith, int nth, void * userdata);

// custom op: the experts of the layer are about to be computed, stop prefetching
void llama_moe_prefetch_stop(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);
