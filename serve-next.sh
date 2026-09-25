#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next NVFP4 on one DGX Spark — NEUES Image (Spark-Branch).
#
# rollback: docker rm -f flashnext-next && docker start flashnext
#           (oder bash ~/qwen38-one-spark/serve.sh — altes Image + alter Layer)
#
# Unterschiede zu serve.sh (altes Image qwen38flashnext + voller Patch-Layer):
#   * IMG = lmsysorg/sglang:dev-qwen38-next-local (Branch qwen4-main-squashed, 4ccff141db)
#     -> enthält die GB10-NEXTN-NaN-Fixes (#36811 via #38308, #38290), den
#        ModelOpt-MIXED_PRECISION-Loader (#38121) und das file-backed PLE
#        Backend (#37068) upstream.
#   * Layer = NUR batch_result_processor.py (Degeneration Guard v3/v5/v6 + Auto-Flush).
#     Der Radix-Race-Patch ist NICHT dabei (mamba_radix_cache.py ist im Branch
#     umgebaut, 483 vs 1427 Zeilen -> nicht portierbar; siehe Session-Analyse).
#     Die alten sm121/KDA- und PLE-mmap-Patches sind entfallen: das neue Image
#     bringt eigene sm120-Pfade + eigenes file-backed PLE.
#   * PLE: --ple-offload-backend file mit FRISCHEM --ple-offload-dir, weil der
#     neue Backend ein bereits gefülltes File boot-langsam (17 MB/s, ~55 min)
#     neu durchschreibt. Das alte Verzeichnis bleibt für den Rollback liegen.
#
# Env: MEMFRAC CTX PREFILL ATTN_PREFILL ATTN_DECODE SPEC LOADFMT NAME PORT PLE_DIR
set -euo pipefail

IMG=${IMG:-lmsysorg/sglang:dev-qwen38-next-local}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$ROOT/build-next"

PLE_DIR=${PLE_DIR:-$HOME/flashnext-ple-next}   # frisches File! siehe Kopf-Kommentar
MEMFRAC=${MEMFRAC:-0.79}
CTX=${CTX:-262144}
PREFILL=${PREFILL:-2048}
ATTN_PREFILL=${ATTN_PREFILL:-triton}
ATTN_DECODE=${ATTN_DECODE:-trtllm_mha}
SPEC=${SPEC:-1}
LOADFMT=${LOADFMT:-auto}
NAME=${NAME:-flashnext-next}
PORT=${PORT:-30000}
RSS_BUDGET_GB=${RSS_BUDGET_GB:-8}

for f in batch_result_processor.py path_degen.txt mamba_radix_cache.py path_radix.txt tokenizer_manager.py path_tm.txt; do
  [ -e "$BUILD/$f" ] || { echo "missing $BUILD/$f"; exit 1; }
done
mkdir -p "$PLE_DIR"
PATH_DEGEN=$(cat "$BUILD/path_degen.txt")
PATH_RADIX=$(cat "$BUILD/path_radix.txt")
PATH_TM=$(cat "$BUILD/path_tm.txt")

SPEC_ARGS=()
if [ "$SPEC" = "1" ]; then
  SPEC_ARGS=(
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --speculative-draft-model-quantization unquant
  )
fi

# NB 24.09.: --disable-chunked-radix-insert existiert im qwen4-Branch nicht
# mehr als CLI-Flag, aber die Race-Ursache ist weiterhin real (Loop-Stall
# 24.09., 4 DEGEN-Onsets/4 rids in 2 min). Unser PR#38355-Fix ist deshalb
# als Layer build-next/mamba_radix_cache.py portiert und wird via
# SGLANG_DISABLE_CHUNKED_RADIX_INSERT=1 aktiviert (siehe mount unten).
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" \
  --gpus all --ipc host \
  -p "$PORT":30000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PLE_DIR":/ple \
  -v "$BUILD/batch_result_processor.py":"$PATH_DEGEN":ro \
  -v "$BUILD/mamba_radix_cache.py":"$PATH_RADIX":ro \
  -v "$BUILD/tokenizer_manager.py":"$PATH_TM":ro \
  -e SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB="$RSS_BUDGET_GB" \
  -e SGLANG_DEGEN_GUARD=1 \
  -e SGLANG_DEGEN_FORENSIC=${DEGEN_FORENSIC:-1} \
  -e SGLANG_DEGEN_WINDOW=999999 \
  -e SGLANG_DEGEN_IMPOSSIBLE_TOKEN=${DEGEN_IMPOSSIBLE_TOKEN:-1} \
  -e SGLANG_IMPOSSIBLE_TOKEN_MAX=${IMPOSSIBLE_TOKEN_MAX:-248077} \
  -e SGLANG_DISABLE_CHUNKED_RADIX_INSERT=1 \
  -e SGLANG_ORPHAN_ABORT=1 \
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
    --ple-offload-backend file \
    --ple-offload-dir /ple \
    --language-only \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "low"}' \
    --tool-call-parser qwen3_coder \
    --mamba-radix-cache-strategy extra_buffer \
    --mem-fraction-static "$MEMFRAC" \
    --context-length "$CTX" \
    --chunked-prefill-size "$PREFILL" \
    --max-running-requests 4 \
    --allow-auto-truncate \
    --enable-metrics \
    --kv-cache-dtype "${KVDTYPE:-auto}" \
    "${SPEC_ARGS[@]}"

echo "started ($NAME, port $PORT, spec=$SPEC, load=$LOADFMT, img=$IMG)"
echo "boot ~10 min (frisches PLE-File). logs: docker logs -f $NAME"
echo "ROLLBACK: docker rm -f $NAME && docker start flashnext"