# usage: bench_rewrite.py <port> <source file> <max tokens>
# decode speed when the model re-emits a file it was given with one rename (copy-heavy agentic edit)
import sys

from common import chat

port = int(sys.argv[1]); src = open(sys.argv[2]).read(); n = int(sys.argv[3])
q = ("Here is a Python file:\n\n```python\n" + src + "\n```\n\nRename the function `parse_args` to `parse_cli_args` everywhere "
     "and output the COMPLETE updated file in a single code block, with no other changes and no commentary.")
body = {"messages": [{"role": "user", "content": q}], "max_tokens": n, "chat_template_kwargs": {"enable_thinking": False}}
t = chat(port, body)["timings"]
print(f"rewrite: {t['predicted_per_second']:.2f} tok/s (draft {t.get('draft_n_accepted',0)}/{t.get('draft_n',0)})")
