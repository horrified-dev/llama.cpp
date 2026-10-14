#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

constexpr int gdn_cols_per_warp = 4;

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// prefill kernel for S_v = 128 and a scalar gate: the inputs of gdn_pf_tc tokens are staged in shared memory,
// each lane keeps 16 rows of gdn_pf_cpl state columns and S^T k, S^T q are reduced together from the old state:
//   delta = beta * (v - g * S^T k), S' = g * S + k delta^T, S'^T q = g * S^T q + delta * (k.q)
constexpr int gdn_pf_tc   = 8;  // tokens per chunk
constexpr int gdn_pf_cpl  = 2;  // state columns per lane
constexpr int gdn_pf_nw   = 4;  // warps per CTA
constexpr int gdn_pf_lpc  = 8;  // lanes per state column
constexpr int gdn_pf_cols = (32 / gdn_pf_lpc) * gdn_pf_cpl * gdn_pf_nw; // state columns per CTA

template <bool keep_rs_t>
__global__ void __launch_bounds__(32 * gdn_pf_nw, 5)
gated_delta_net_prefill_cuda(const float * __restrict__ q,
                             const float * __restrict__ k,
                             const float * __restrict__ v,
                             const float * __restrict__ g,
                             const float * __restrict__ beta,
                             const float * __restrict__ curr_state,
                             float *       __restrict__ dst,
                             float *       __restrict__ state,
                             int64_t       H,
                             int64_t       n_tokens,
                             int64_t       sq1,
                             int64_t       sq2,
                             int64_t       sq3,
                             int64_t       sv1,
                             int64_t       sv2,
                             int64_t       sv3,
                             int64_t       sb1,
                             int64_t       sb2,
                             int64_t       sb3,
                             const uint3   neqk1_magic,
                             const uint3   rq3_magic,
                             float         scale,
                             int64_t       state_slot_stride,
                             int           K) {
    constexpr int S_v  = 128;
    constexpr int TC   = gdn_pf_tc;
    constexpr int CPL  = gdn_pf_cpl;
    constexpr int LPC  = gdn_pf_lpc;
    constexpr int COLS = gdn_pf_cols;
    constexpr int RPL  = S_v / LPC; // rows per lane, the float4 at rows 32*m + 4*l for m < RPL/4
    constexpr int NT   = 32 * gdn_pf_nw;

    __shared__ float4 ks[TC][S_v/4];
    __shared__ float4 qs[TC][S_v/4];
    __shared__ float  vs[TC][COLS];
    __shared__ float  os[TC][COLS];
    __shared__ float  gs[TC];
    __shared__ float  bs[TC];
    __shared__ float  kqs[TC];

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      col0     = blockIdx.z * COLS;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int l    = lane % LPC;                             // lane within the column group
    const int lc   = (warp * (32 / LPC) + lane / LPC) * CPL; // first column of the lane within the CTA

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const float * q_h = q + iq3 * sq3 + iq1 * sq1;
    const float * k_h = k + iq3 * sq3 + iq1 * sq1;
    const float * v_h = v + sequence * sv3 + h_idx * sv1 + col0;
    const float * g_h = g    + sequence * sb3 + h_idx * sb1;
    const float * b_h = beta + sequence * sb3 + h_idx * sb1;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v + col0;

    state      += (sequence * H + h_idx) * S_v * S_v;
    curr_state += (sequence * H + h_idx) * S_v * S_v;

    ggml_cuda_pdl_sync();

    // the state is stored transposed: row col of M holds S[:, col]
    float s[CPL][RPL];
#pragma unroll
    for (int c = 0; c < CPL; ++c) {
        const float4 * src = (const float4 *) (curr_state + (col0 + lc + c) * S_v);
#pragma unroll
        for (int m = 0; m < RPL/4; ++m) {
            const float4 x = src[8*m + l];
            s[c][4*m + 0] = x.x;
            s[c][4*m + 1] = x.y;
            s[c][4*m + 2] = x.z;
            s[c][4*m + 3] = x.w;
        }
    }

    for (int64_t t0 = 0; t0 < n_tokens; t0 += TC) {
        const int nt = (int) min((int64_t) TC, n_tokens - t0);

        __syncthreads(); // the previous chunk is consumed

        for (int idx = tid; idx < nt * (S_v/4); idx += NT) {
            const int tt = idx / (S_v/4);
            const int i4 = idx % (S_v/4);
            ks[tt][i4] = ((const float4 *) (k_h + (t0 + tt) * sq2))[i4];
            qs[tt][i4] = ((const float4 *) (q_h + (t0 + tt) * sq2))[i4];
        }
        for (int idx = tid; idx < nt * COLS; idx += NT) {
            const int tt = idx / COLS;
            const int c  = idx % COLS;
            vs[tt][c] = v_h[(t0 + tt) * sv2 + c];
        }
        if (tid < nt) {
            gs[tid] = expf(g_h[(t0 + tid) * sb2]);
            bs[tid] = b_h[(t0 + tid) * sb2];
        }
        __syncthreads();

        for (int tt = warp; tt < nt; tt += gdn_pf_nw) {
            const float4 a = ks[tt][lane];
            const float4 b = qs[tt][lane];
            const float  p = warp_reduce_sum(a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w);
            if (lane == 0) {
                kqs[tt] = p;
            }
        }
        __syncthreads();

        for (int tt = 0; tt < nt; ++tt) {
            float kr[RPL];
            float qr[RPL];
#pragma unroll
            for (int m = 0; m < RPL/4; ++m) {
                const float4 a = ks[tt][8*m + l];
                const float4 b = qs[tt][8*m + l];
                kr[4*m + 0] = a.x; kr[4*m + 1] = a.y; kr[4*m + 2] = a.z; kr[4*m + 3] = a.w;
                qr[4*m + 0] = b.x; qr[4*m + 1] = b.y; qr[4*m + 2] = b.z; qr[4*m + 3] = b.w;
            }
            const float g_val  = gs[tt];
            const float b_val  = bs[tt];
            const float kq_val = kqs[tt];

            // (S^T k, S^T q) of the columns of the lane
            float2 r[CPL];
#pragma unroll
            for (int c = 0; c < CPL; ++c) {
                float a0 = 0.0f, a1 = 0.0f, b0 = 0.0f, b1 = 0.0f;
#pragma unroll
                for (int i = 0; i < RPL; i += 2) {
                    a0 += s[c][i + 0] * kr[i + 0];
                    a1 += s[c][i + 1] * kr[i + 1];
                    b0 += s[c][i + 0] * qr[i + 0];
                    b1 += s[c][i + 1] * qr[i + 1];
                }
                r[c] = warp_reduce_sum<LPC>(make_float2(a0 + a1, b0 + b1));
            }

#pragma unroll
            for (int c = 0; c < CPL; ++c) {
                const float delta = (vs[tt][lc + c] - g_val * r[c].x) * b_val;
#pragma unroll
                for (int i = 0; i < RPL; ++i) {
                    s[c][i] = g_val * s[c][i] + kr[i] * delta;
                }
                if (l == 0) {
                    os[tt][lc + c] = (g_val * r[c].y + delta * kq_val) * scale;
                }
            }

            if constexpr (keep_rs_t) {
                // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back
                const int64_t target_slot = n_tokens - 1 - (t0 + tt);
                if (target_slot < K) {
                    float * snap = state + target_slot * state_slot_stride;
#pragma unroll
                    for (int c = 0; c < CPL; ++c) {
                        float4 * d4 = (float4 *) (snap + (col0 + lc + c) * S_v);
#pragma unroll
                        for (int m = 0; m < RPL/4; ++m) {
                            d4[8*m + l] = make_float4(s[c][4*m + 0], s[c][4*m + 1], s[c][4*m + 2], s[c][4*m + 3]);
                        }
                    }
                }
            }
        }
        __syncthreads();

        for (int idx = tid; idx < nt * COLS; idx += NT) {
            const int tt = idx / COLS;
            const int c  = idx % COLS;
            attn_data[(t0 + tt) * S_v * H + c] = os[tt][c];
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < CPL; ++c) {
            float4 * d4 = (float4 *) (state + (col0 + lc + c) * S_v);
#pragma unroll
            for (int m = 0; m < RPL/4; ++m) {
                d4[8*m + l] = make_float4(s[c][4*m + 0], s[c][4*m + 1], s[c][4*m + 2], s[c][4*m + 3]);
            }
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    // prompt processing with 128 wide heads and a scalar gate: inputs staged in shared memory
    if constexpr (!KDA) {
        static const bool prefill_enabled = getenv("GGML_CUDA_GDN_PREFILL") == nullptr || atoi(getenv("GGML_CUDA_GDN_PREFILL")) != 0;
        const bool aligned =
            (uintptr_t) q_d % 16 == 0 && (uintptr_t) k_d % 16 == 0 && sq1 % 4 == 0 && sq2 % 4 == 0 && sq3 % 4 == 0 &&
            (uintptr_t) s_d % 16 == 0 && (uintptr_t) state_d % 16 == 0 && state_slot_stride % 4 == 0;
        if (prefill_enabled && S_v == 128 && n_tokens >= 32 && warp_size == 32 && aligned) {
            const uint3 neqk1_magic = init_fastdiv_values(neqk1);
            const uint3 rq3_magic   = init_fastdiv_values(rq3);
            const dim3 grid_dims(H, n_seqs, S_v / gdn_pf_cols);
            const dim3 block_dims(32, gdn_pf_nw, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_prefill_cuda<keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            return;
        }
    }
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
