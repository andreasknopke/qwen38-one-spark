#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next NVFP4 on one DGX Spark.
#
# The checkpoint is 126.0 GiB and the Spark has 121.63 GiB of unified memory.
# It fits because of the two patches applied by prepare.sh -- see ../README.md.
#
# Env: MEMFRAC CTX PREFILL ATTN_PREFILL ATTN_DECODE SPEC LOADFMT NAME PORT PLE_DIR
set -euo pipefail

IMG=${IMG:-lmsysorg/sglang:qwen38flashnext}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$ROOT/build"

PLE_DIR=${PLE_DIR:-$HOME/flashnext-ple}   # mmap backing store; sparse, persists
MEMFRAC=${MEMFRAC:-0.85}                  # 0.72 gives total_rest_memory < 0
CTX=${CTX:-32768}
PREFILL=${PREFILL:-2048}
# flashinfer hits a CUTLASS SM120 kernel that won't compile, so prefill is triton.
# Decode is a different story: --attention-backend trtllm_mha is refused because it
# sets BOTH phases and prefill is gated to SM100, but the decode-side check in
# server_args.py lists is_sm120_supported() explicitly. Splitting them is worth
# +32% on code (31.5 -> 41.5 tok/s).
ATTN_PREFILL=${ATTN_PREFILL:-triton}
ATTN_DECODE=${ATTN_DECODE:-trtllm_mha}
SPEC=${SPEC:-1}                           # 1 = MTP via NEXTN
LOADFMT=${LOADFMT:-auto}                  # NB: "dummy" OOMs with PLE offload, see README
NAME=${NAME:-flashnext}
PORT=${PORT:-30000}

for f in qwen4_exp.py qwen_sparse_attn_backend.py qsa_kv_pool.py qsa_metadata.py \
         qsa_graph_metadata.py sm121_varlen.py kda_kernels batch_result_processor.py \
         path_qwen4_exp.txt path_qsa.txt path_pool.txt path_meta.txt path_qsa_graph.txt \
         path_sm121_varlen.txt path_kda_kernels.txt path_degen.txt \
         mem_cache/cache_init_params.py mem_cache/kv_cache_builder.py mem_cache/mamba_radix_cache.py \
         server_args.py \
         path_cache_init.txt path_kv_cache_builder.txt path_mamba_radix_cache.txt path_server_args.txt; do
  [ -e "$BUILD/$f" ] || { echo "missing $BUILD/$f — run ./scripts/prepare.sh first"; exit 1; }
done
mkdir -p "$PLE_DIR"
PATH_MODEL=$(cat "$BUILD/path_qwen4_exp.txt")
PATH_QSA=$(cat "$BUILD/path_qsa.txt")
PATH_POOL=$(cat "$BUILD/path_pool.txt")
PATH_META=$(cat "$BUILD/path_meta.txt")
PATH_QSA_GRAPH=$(cat "$BUILD/path_qsa_graph.txt")
PATH_SM121=$(cat "$BUILD/path_sm121_varlen.txt")
PATH_KDA=$(cat "$BUILD/path_kda_kernels.txt")
PATH_DEGEN=$(cat "$BUILD/path_degen.txt")
PATH_CACHE_INIT=$(cat "$BUILD/path_cache_init.txt")
PATH_KV_CACHE_BUILDER=$(cat "$BUILD/path_kv_cache_builder.txt")
PATH_MAMBA_RADIX=$(cat "$BUILD/path_mamba_radix_cache.txt")
PATH_SERVER_ARGS=$(cat "$BUILD/path_server_args.txt")

# MTP: 1 layer, already inside the checkpoint. Its 31 tensors are still BF16
# (RadixArk quantized only the routed experts), hence "unquant" for the draft:
# with modelopt_fp4 inherited from the body it would read them as NVFP4.
# PLE requires topk=1, which is exactly what NEXTN does.
SPEC_ARGS=()
if [ "$SPEC" = "adaptive" ]; then
  # Adaptive steps: controller switches 3<->7 based on acceptance. Ring must be
  # wide enough for the max (8 draft tokens) -> SGLANG_QSA_RING_WIDTH=8 required.
  # NOTE: EAGLE (not NEXTN) — adaptive_unsupported_reason checks the raw string,
  # and NEXTN is only resolved to EAGLE later in the arg hooks.
  RING_WIDTH=${RING_WIDTH:-8}
  SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 7
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 8
    --speculative-draft-model-quantization unquant
    --speculative-adaptive
    --speculative-adaptive-config /adaptive/adaptive.json
  )
elif [ "$SPEC" = "ring8" ]; then
  # Fixed code-mode: wide ring + 7 steps, NO adaptive (adaptive+ring crashes, see README)
  RING_WIDTH=${RING_WIDTH:-8}
  SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 7
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 8
    --speculative-draft-model-quantization unquant
  )
elif [ "$SPEC" = "1" ]; then
  SPEC_ARGS=(
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --speculative-draft-model-quantization unquant
  )
fi

docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" \
  --gpus all --ipc host \
  -p "$PORT":30000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PLE_DIR":/ple \
  -v "$BUILD/qwen4_exp.py":"$PATH_MODEL":ro \
  -v "$BUILD/qwen_sparse_attn_backend.py":"$PATH_QSA":ro \
  -v "$BUILD/qsa_kv_pool.py":"$PATH_POOL":ro \
  -v "$BUILD/qsa_metadata.py":"$PATH_META":ro \
  -v "$BUILD/qsa_graph_metadata.py":"$PATH_QSA_GRAPH":ro \
  -v "$BUILD/sm121_varlen.py":"$PATH_SM121":ro \
  -v "$BUILD/kda_kernels":"$PATH_KDA":ro \
  -v "$BUILD/batch_result_processor.py":"$PATH_DEGEN":ro \
  -v "$BUILD/mem_cache/cache_init_params.py":"$PATH_CACHE_INIT":ro \
  -v "$BUILD/mem_cache/kv_cache_builder.py":"$PATH_KV_CACHE_BUILDER":ro \
  -v "$BUILD/mem_cache/mamba_radix_cache.py":"$PATH_MAMBA_RADIX":ro \
  -v "$BUILD/server_args.py":"$PATH_SERVER_ARGS":ro \
  -e SGLANG_QWEN4_PLE_MMAP_DIR=/ple \
  -e SGLANG_QSA_RING_WIDTH=${RING_WIDTH:-4} \
  -e SGLANG_DEGEN_GUARD=1 \
  -e SGLANG_DEGEN_FORENSIC=${DEGEN_FORENSIC:-1} \
  -e SGLANG_DEGEN_WINDOW=999999 \
  -e SGLANG_DEGEN_IMPOSSIBLE_TOKEN=${DEGEN_IMPOSSIBLE_TOKEN:-1} \
  -e SGLANG_IMPOSSIBLE_TOKEN_MAX=${IMPOSSIBLE_TOKEN_MAX:-248077} \
  -v "$(pwd)/config":/adaptive:ro \
  "$IMG" \
  python3 -m sglang.launch_server \
    --model-path RadixArk/Qwen3.8-Flash-Next-NVFP4 \
    --served-model-name qwen38-flash-next \
    --host 0.0.0.0 --port 30000 \
    --load-format "$LOADFMT" \
    --tp-size 1 \
    --prefill-attention-backend "$ATTN_PREFILL" \
    --decode-attention-backend "$ATTN_DECODE" \
    --quantization modelopt_fp4 \
    --ple-offload-embedding \
    --language-only \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "low"}' \
    --tool-call-parser qwen3_coder \
    --mamba-radix-cache-strategy extra_buffer \
    --mem-fraction-static "$MEMFRAC" \
    ${NO_CHUNKED_PREFIX:+--disable-chunked-prefix-cache} \
    ${NO_RADIX:+--disable-radix-cache} \
    --context-length "$CTX" \
    --chunked-prefill-size "$PREFILL" \
    --disable-chunked-radix-insert \
    --max-running-requests 4 \
    --allow-auto-truncate \
    --enable-metrics \
    --kv-cache-dtype "${KVDTYPE:-auto}" \
    "${SPEC_ARGS[@]}"

echo "started ($NAME, port $PORT, spec=$SPEC, load=$LOADFMT)"
echo "first boot takes ~9 min. logs: docker logs -f $NAME"
echo
echo "NOTE: --host 0.0.0.0 with no auth in front. Do not expose this to a network you don't trust."
