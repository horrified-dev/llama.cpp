#!/bin/bash
# runs the decode and prompt benchmarks of the README against a running server (start-server.sh)
# usage: scripts/qwen36/bench.sh [port]   env: LLAMA_API_KEY, LLAMA_MODEL (when the server hosts several models)
cd "$(dirname "$0")/../.." || exit 1
D=scripts/qwen36
PORT=${1:-8080}
python3 $D/bench_decode.py "$PORT" greedy 300
python3 $D/bench_decode.py "$PORT" default 300
python3 $D/bench_rewrite.py "$PORT" scripts/compare-logprobs.py 1500
python3 $D/bench_multiturn.py "$PORT"
