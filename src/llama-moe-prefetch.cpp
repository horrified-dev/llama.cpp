#include "llama-moe-prefetch.h"

#include "llama-impl.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace {

// "16-31" or "0,2,4-7"
std::vector<int> parse_cpu_list(const char * str) {
    std::vector<int> res;
    if (str == nullptr) {
        return res;
    }
    const std::string s(str);
    size_t pos = 0;
    while (pos < s.size()) {
        size_t end = s.find(',', pos);
        if (end == std::string::npos) {
            end = s.size();
        }
        const std::string item = s.substr(pos, end - pos);
        const size_t dash = item.find('-');
        if (dash == std::string::npos) {
            res.push_back(std::atoi(item.c_str()));
        } else {
            const int lo = std::atoi(item.substr(0, dash).c_str());
            const int hi = std::atoi(item.substr(dash + 1).c_str());
            for (int c = lo; c <= hi; ++c) {
                res.push_back(c);
            }
        }
        pos = end + 1;
    }
    return res;
}

struct prefetch_layer_entry {
    llama_moe_prefetch_layer layer;
    const ggml_tensor * router_src;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;
};

struct prefetch_job {
    const llama_moe_prefetch_layer * layer = nullptr;
    std::vector<int32_t> ids;
    int64_t t_submit_us = 0;
};

class prefetch_engine {
public:
    prefetch_engine(int32_t n_expert, int64_t budget_us, std::vector<int> cpus) :
        n_expert(n_expert), budget_us(budget_us), cpus(std::move(cpus)) {}

    ~prefetch_engine() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        for (auto & t : threads) {
            t.join();
        }
    }

    llama_moe_prefetch_layer * get_layer(
            const std::vector<const ggml_tensor *> & experts, int32_t n_threads,
            const ggml_tensor * router, const ggml_tensor * norm_cur, const ggml_tensor * norm_next) {
        std::lock_guard<std::mutex> lock(mtx);
        for (auto & e : layers) {
            if (e.layer.experts == experts && e.layer.n_threads == n_threads && e.router_src == router) {
                return &e.layer;
            }
        }

        const int64_t n_embd    = router->ne[0];
        const int64_t n_experts = router->ne[1];
        for (const ggml_tensor * t : { router, norm_cur, norm_next }) {
            if (t->type != GGML_TYPE_F32 || t->buffer == nullptr || t->data == nullptr || t->ne[0] != n_embd) {
                return nullptr;
            }
        }

        // fold the norm ratio into the router: the input of the experts of this layer is rms_norm(x)*norm_cur,
        // the input of the router of the next layer would be rms_norm(x)*norm_next
        std::vector<float> w(n_embd*n_experts);
        std::vector<float> g_cur(n_embd);
        std::vector<float> g_next(n_embd);
        ggml_backend_tensor_get(router,    w.data(),      0, w.size()*sizeof(float));
        ggml_backend_tensor_get(norm_cur,  g_cur.data(),  0, n_embd*sizeof(float));
        ggml_backend_tensor_get(norm_next, g_next.data(), 0, n_embd*sizeof(float));
        for (int64_t i = 0; i < n_embd; ++i) {
            const float ratio = g_cur[i] != 0.0f ? g_next[i]/g_cur[i] : 0.0f;
            for (int64_t e = 0; e < n_experts; ++e) {
                w[e*n_embd + i] *= ratio;
            }
        }

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx(ggml_init(params));
        // the router is read for every token on the critical path, Q8_0 is precise enough to rank the experts
        const ggml_type type = n_embd % ggml_blck_size(GGML_TYPE_Q8_0) == 0 ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
        ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), type, n_embd, n_experts);
        ggml_format_name(t, "moe_prefetch.router.%s", router->name);
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
        if (!buf) {
            return nullptr;
        }
        // a weight, so that the scheduler computes the router on the CPU
        ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(t, w.data(), 0, w.size()*sizeof(float));
        } else {
            ggml_quantize_chunk(type, w.data(), t->data, 0, n_experts, n_embd, nullptr);
        }

        if (threads.empty()) {
            LLAMA_LOG_INFO("llama_moe_prefetch: prefetching %d experts per layer for up to %lld us, %zu helper CPUs given\n",
                n_expert, (long long) budget_us, cpus.size());
        }
        while ((int32_t) threads.size() < n_threads) {
            const int i = (int) threads.size();
            threads.emplace_back([this, i]() { worker(i); });
        }

        prefetch_layer_entry entry;
        entry.layer.experts   = experts;
        entry.layer.n_threads = n_threads;
        entry.layer.router    = t;
        entry.router_src      = router;
        entry.ctx             = std::move(ctx);
        entry.buf             = std::move(buf);
        layers.push_back(std::move(entry));
        return &layers.back().layer;
    }

    void submit(const llama_moe_prefetch_layer * layer, std::vector<int32_t> ids) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            job.layer       = layer;
            job.ids         = std::move(ids);
            job.t_submit_us = ggml_time_us();
            gen.fetch_add(1, std::memory_order_release);
        }
        cv.notify_all();
    }

    void cancel() {
        std::lock_guard<std::mutex> lock(mtx);
        job.layer = nullptr;
        job.ids.clear();
        gen.fetch_add(1, std::memory_order_release);
    }

    const int32_t n_expert;
    const int64_t budget_us; // upper bound of the prefetch time, in case the experts are not computed

private:
    void worker(int ith) {
#if defined(__linux__)
        if (ith < (int) cpus.size()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpus[ith], &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        }
#endif
        uint64_t seen = 0;
        prefetch_job cur;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return stop || gen.load(std::memory_order_acquire) != seen; });
                if (stop) {
                    return;
                }
                seen = gen.load(std::memory_order_acquire);
                cur  = job;
            }
            run(cur, ith, seen);
        }
    }

    // load the rows of the experts that compute thread ith will read, until a newer job arrives or the time is up
    void run(const prefetch_job & j, int ith, uint64_t g) {
        if (j.layer == nullptr || ith >= j.layer->n_threads) {
            return;
        }
        const int nth = j.layer->n_threads;
        const int64_t t_end = j.t_submit_us + budget_us;

        uint64_t sink = 0;
        for (const int32_t id : j.ids) {
            for (const ggml_tensor * t : j.layer->experts) {
                if (id < 0 || id >= t->ne[2]) {
                    continue;
                }
                // for a few tokens, the CPU mul_mat_id gives each thread an equal slice of the rows of each expert
                const int64_t nr0 = t->ne[1];
                const int64_t dr0 = (nr0 + nth - 1)/nth;
                const int64_t ir0 = std::min(nr0, dr0*ith);
                const int64_t ir1 = std::min(nr0, ir0 + dr0);

                const char * p    = (const char *) t->data + id*t->nb[2] + ir0*t->nb[1];
                const size_t size = (ir1 - ir0)*t->nb[1];

                // demand loads: software prefetches are dropped when too many are in flight
                for (size_t off = 0; off < size; off += 64) {
                    sink += *(const volatile uint64_t *) (p + off);
                    if ((off & 4095) == 0 && (gen.load(std::memory_order_relaxed) != g || ggml_time_us() > t_end)) {
                        sink_total.fetch_add(sink, std::memory_order_relaxed);
                        return;
                    }
                }
            }
        }
        sink_total.fetch_add(sink, std::memory_order_relaxed);
    }

    std::vector<int> cpus;
    std::list<prefetch_layer_entry> layers;
    std::vector<std::thread> threads;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<uint64_t> gen{0};
    std::atomic<uint64_t> sink_total{0};
    prefetch_job job;
    bool stop = false;
};

prefetch_engine * get_engine() {
    static std::unique_ptr<prefetch_engine> engine = []() -> std::unique_ptr<prefetch_engine> {
        const char * env = getenv("LLAMA_MOE_PREFETCH");
        const int n_expert = env ? std::atoi(env) : 0;
        if (n_expert <= 0) {
            return nullptr;
        }
        const char * env_us = getenv("LLAMA_MOE_PREFETCH_US");
        const int64_t budget_us = env_us ? std::atoi(env_us) : 1000;
        std::vector<int> cpus = parse_cpu_list(getenv("LLAMA_MOE_PREFETCH_CPUS"));
        return std::make_unique<prefetch_engine>(n_expert, budget_us, std::move(cpus));
    }();
    return engine.get();
}

}

int32_t llama_moe_prefetch_n_expert() {
    const prefetch_engine * engine = get_engine();
    return engine ? engine->n_expert : 0;
}

int32_t llama_moe_prefetch_max_tokens() {
    static const int32_t n = getenv("LLAMA_MOE_PREFETCH_MAX_TOKENS") ? std::atoi(getenv("LLAMA_MOE_PREFETCH_MAX_TOKENS")) : 4;
    return n;
}

llama_moe_prefetch_layer * llama_moe_prefetch_get_layer(
        const std::vector<const ggml_tensor *> & experts,
        int32_t n_threads,
        const ggml_tensor * router,
        const ggml_tensor * norm_cur,
        const ggml_tensor * norm_next) {
    prefetch_engine * engine = get_engine();
    return engine ? engine->get_layer(experts, n_threads, router, norm_cur, norm_next) : nullptr;
}

void llama_moe_prefetch_kick(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    prefetch_engine * engine = get_engine();
    if (engine == nullptr) {
        return;
    }

    const ggml_tensor * logits = dst->src[0];
    GGML_ASSERT(logits->type == GGML_TYPE_F32);

    const int64_t n_experts = logits->ne[0];
    const int64_t n_tokens  = logits->ne[1];

    // rank the experts by their total probability over the tokens
    std::vector<float> score(n_experts, 0.0f);
    for (int64_t it = 0; it < n_tokens; ++it) {
        const float * l = (const float *) ((const char *) logits->data + it*logits->nb[1]);
        const float max = *std::max_element(l, l + n_experts);
        float sum = 0.0f;
        for (int64_t e = 0; e < n_experts; ++e) {
            sum += std::exp(l[e] - max);
        }
        for (int64_t e = 0; e < n_experts; ++e) {
            score[e] += std::exp(l[e] - max)/sum;
        }
    }

    std::vector<int32_t> ids(n_experts);
    for (int64_t e = 0; e < n_experts; ++e) {
        ids[e] = (int32_t) e;
    }
    const int64_t n = std::min<int64_t>(engine->n_expert, n_experts);
    std::partial_sort(ids.begin(), ids.begin() + n, ids.end(), [&](int32_t a, int32_t b) { return score[a] > score[b]; });
    ids.resize(n);

    engine->submit((const llama_moe_prefetch_layer *) userdata, std::move(ids));
}

void llama_moe_prefetch_stop(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(dst);
    GGML_UNUSED(a);
    GGML_UNUSED(nth);
    GGML_UNUSED(userdata);
    if (ith != 0) {
        return;
    }
    prefetch_engine * engine = get_engine();
    if (engine != nullptr) {
        engine->cancel();
    }
}
