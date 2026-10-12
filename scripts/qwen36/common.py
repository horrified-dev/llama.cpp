import json
import os
import urllib.request


def chat(port, body, timeout=7200):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("LLAMA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["LLAMA_API_KEY"]
    if os.environ.get("LLAMA_MODEL"):
        body = dict(body, model=os.environ["LLAMA_MODEL"])
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=timeout))
