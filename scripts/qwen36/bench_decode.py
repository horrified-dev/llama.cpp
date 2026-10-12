# usage: bench_decode.py <port> <greedy|default> [max tokens per prompt]
# decode speed (and draft acceptance) over the prompts in prompts.txt
import os
import sys

from common import chat

port = int(sys.argv[1]); mode = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
prompts = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.txt")).read().split('\n%%%\n')
tot_tok = tot_ms = dn = da = 0
for p in prompts:
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": n, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    if mode == "greedy":
        body.update({"temperature": 0.0, "top_k": 1})
    t = chat(port, body, 3600)["timings"]
    tot_tok += t["predicted_n"]; tot_ms += t["predicted_ms"]; dn += t.get("draft_n", 0); da += t.get("draft_n_accepted", 0)
print(f"{mode}: {tot_tok/tot_ms*1000:.2f} tok/s (acc {da/max(dn,1)*100:.0f}%)")
