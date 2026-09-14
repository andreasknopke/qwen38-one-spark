# Qwen3.8-Flash-Next — One DGX Spark Recipe

A battle-tested recipe for running **Qwen3.8-Flash-Next 180B MoE (NVFP4)** on a single **NVIDIA DGX Spark (GB10)** — with all patches that permanently fix the `"!" KV loop`.

## Status

- ✅ **The "!" loop is dead** — 5h+ stable VSCode sessions with zero KV-corruption loops (pre-fix: every 1-2h at ~33k or ~145k context)
- ✅ ~35 tok/s single-stream, 85-95 tok/s aggregate
- ✅ 262K context, chunked prefill 2048
- ✅ Speculative decoding (EAGLE NEXTN, 3 steps)
- ✅ KDA decode kernel + SM121 Triton fallback
- ✅ Auto flush on KV corruption detection

## The "!!!!" Loop

Qwen3.8-Flash-Next on DGX Spark has a stochastic bug in the interaction between chunked prefill and the radix cache. A race condition produces a corrupted KV page; the radix cache then perpetuates that corruption — every subsequent request matching the same prefix reads the corrupt attention data and produces **token ID 248319** (the never-trained last row of `lm_head`, which decodes as `''` and appears as `"!"` in the client).

**Symptoms:**
- Sudden `"!!!!"` output mid-session
- Request terminates after 1-2 tokens with ID 248319
- Every retry with the same history crashes identically
- `flush_cache` resolves it temporarily

**Evidence:** 4 consecutive forensic incidents show **identical** KV slot addresses with varying mamba pool indices — the bug lives in the KV cache, not the model state.

## The Fix: `disable_chunked_radix_insert`

The root cause is a race between `cache_unfinished_req` (inserts KV pages into the radix tree mid-prefill) and `cache_finished_req` (frees pages on retract/abort). The radix node survives with dangling references to freed pages.

**Solution:** Skip the radix insert during chunked prefill entirely — defer it to `cache_finished_req`, which runs after the request is fully committed and no longer subject to retraction. The radix tree never references pages while they can still be freed.

Auto-enabled for QSA compressed-attention models (no manual flag needed).

**Fallback:** If corruption somehow still occurs, the v6 guard fires *before* the corrupt tokens are streamed: it drops the junk output (`del req.output_ids[:]`) and aborts the request with a retryable `FINISH_ABORT("kv_cache_corruption: ... safe to retry")`, then triggers a background `flush_cache`. The client never sees `!!` — it receives a transient-error abort it can retry automatically (silent correction).

## Patches in This Repo

| Patch | File | Purpose |
|---|---|---|
| `disable_chunked_radix_insert` | `server_args.py`, `mem_cache/*.py` | **The "!" loop fix** — eliminates the chunked-prefill radix-insert race |
| v6 impossible-token guard | `batch_result_processor.py` | Detects token ID `248319` on the very first decode step; drops junk output and aborts with a retryable reason (silent correction, no `!!` streamed) |
| Auto-flush | `batch_result_processor.py` | Post-v6-trigger background `flush_cache` to clear the corrupt radix prefix |
| KDA decode kernel + SM121 attention | `qwen_sparse_attn_backend.py`, `qsa_*.py`, `sm121_varlen.py`, `kda_kernels/` | SM121-optimized QSA sparse attention |
| Mamba extra_buffer | `qwen4_exp.py` | Hybrid Mamba-attention radix cache for Flash-Next |
| Identical-run guard disabled | `batch_result_processor.py` | Window=999999 — prevents false positives on legitimate number-list output |

## Requirements

- 1× NVIDIA DGX Spark / GB10 (128 GB unified memory)
- Docker installed
- ~126 GB free disk for model cache + PLE offload
- Internet for one-time model download

## Quick Start

```bash
git clone https://github.com/andreasknopke/qwen38-one-spark.git
cd qwen38-one-spark

# Start (approx. 9 min boot time)
MEMFRAC=0.79 PREFILL=2048 CTX=262144 bash serve.sh
```

Configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MEMFRAC` | `0.79` | KV cache fraction of GPU memory |
| `CTX` | `262144` | Context length |
| `PREFILL` | `2048` | Chunked prefill size |
| `SPEC` | `1` | Spec mode (`1`=NEXTN, `ring8`, `adaptive`) |
| `PORT` | `30000` | Server port |
| `KVDTYPE` | `auto` | KV cache data type |
| `DEGEN_FORENSIC` | `1` | Debug logging for degeneration |

## Details

### Model

- **Base**: `RadixArk/Qwen3.8-Flash-Next-NVFP4` (NVIDIA ModelOpt NVFP4)
- **Architecture**: 180B MoE, 48 layers, QSA sparse attention + hybrid mamba
- **Quantization**: NVFP4 (experts), BF16 (remaining layers)
- **Context**: 262144 tokens (maximum)

### Serving Stack

Uses the `lmsysorg/sglang:qwen38flashnext` image. All patches are applied via bind-mount — no custom Docker build needed.

### Attention Backend

| Phase | Backend |
|---|---|
| Prefill | Triton (QSA sparse) |
| Decode | `trtllm_mha` + KDA kernel |

## Repository Structure

```
.
├── serve.sh                    # Docker launch script
├── batch_result_processor.py   # Scheduler guards (v6 + auto-flush)
├── eagle_worker_v2.py          # EAGLE spec-decode
├── kda_kernels/                # KDA decode kernel
├── mem_cache/                  # cache_init_params, kv_cache_builder, mamba_radix_cache
├── qsa_graph_metadata.py       # QSA graph metadata
├── qsa_kv_pool.py              # QSA KV pool
├── qsa_metadata.py             # QSA metadata
├── qwen4_exp.py                # Model architecture (Qwen4-Exp)
├── qwen_sparse_attn_backend.py # Sparse attention backend
├── server_args.py              # Server args (+ disable_chunked_radix_insert)
└── sm121_varlen.py             # SM121 Triton varlen kernel
```

## History

This repo grew out of production work with Qwen3.8-Flash-Next on a single DGX Spark. The original `"!"` KV loop (issue [#38319](https://github.com/sgl-project/sglang/issues/38319)) was caused by a chunked-prefill radix-insert race. The fix (`disable_chunked_radix_insert`) was submitted as PR [#38355](https://github.com/sgl-project/sglang/pull/38355) but was not merged upstream. This repo carries the fix forward as a standalone fork.