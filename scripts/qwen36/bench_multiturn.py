# usage: bench_multiturn.py <port>   (run from the repository root)
# three agentic turns with large tool outputs: prompt processing speed and prompt cache reuse
import sys

from common import chat

port = int(sys.argv[1])
f1 = open('src/llama-context.cpp').read()[:40000]
f2 = open('src/llama-kv-cache.cpp').read()[:12000]


def turn(messages, n):
    r = chat(port, {"messages": messages, "max_tokens": n, "chat_template_kwargs": {"enable_thinking": False}})
    return r["choices"][0]["message"]["content"], r["timings"]


def show(k, t):
    print(f"turn {k}: processed {t['prompt_n']} new prompt tok in {t['prompt_ms']/1000:.1f}s "
          f"({t['prompt_per_second']:.0f} tok/s) | gen {t['predicted_n']} @ {t['predicted_per_second']:.1f} tok/s")


msgs = [{"role": "system", "content": "You are a coding agent. Use the tool outputs to answer."},
        {"role": "user", "content": "Tool output (read_file src/llama-context.cpp):\n```cpp\n" + f1 +
         "\n```\nSummarize what llama_context::decode does in 5 bullet points."}]
a1, t1 = turn(msgs, 150); show(1, t1)
msgs += [{"role": "assistant", "content": a1},
         {"role": "user", "content": "Tool output (read_file src/llama-kv-cache.cpp):\n```cpp\n" + f2 +
          "\n```\nHow does this file relate to the previous one? Two sentences."}]
a2, t2 = turn(msgs, 100); show(2, t2)
msgs += [{"role": "assistant", "content": a2}, {"role": "user", "content": "Thanks. One more sentence: which file is longer?"}]
a3, t3 = turn(msgs, 60); show(3, t3)
