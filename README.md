# Qwen3.8-Flash-Next — One DGX Spark Recipe

A battle-tested recipe for running **Qwen3.8-Flash-Next 180B MoE (NVFP4)** on a single
**NVIDIA DGX Spark (GB10)**.

The repo carries **two variants** of the deployment — the current one and the previous one,
kept intact so switching back is a single command:

| Variant | Launch | Base image | Patch layer |
|---|---|---|---|
| **current** (since 2026-09-20) | `bash serve-next.sh` | `lmsysorg/sglang:dev-qwen38-next-local` (commit `4ccff141db`) | `build-next/` — **Degeneration Guard only** |
| rollback (2026-08-29 … 2026-09-20) | `MEMFRAC=0.79 CTX=262144 bash serve.sh` | `lmsysorg/sglang:qwen38flashnext` (nightly `d91c3682` + RadixArk overlay) | `build/` — guard **+ our `disable_chunked_radix_insert` loop fix** + SM121/KDA kernels + PLE-mmap |

Both image digests, the rollback commands and the notes that go with them:
[`ROLLBACK/README.txt`](ROLLBACK/README.txt).

## Status

- ✅ Stable in daily agentic use (VSCode Copilot / Claude Code) on the current variant
- ✅ ~35 tok/s single-stream, 85–95 tok/s aggregate — **no speed difference** to the previous variant
- ✅ 262K context, chunked prefill 2048, EAGLE NEXTN spec decode (3 steps)
- ✅ Degeneration Guard active as the safety net (impossible-token detection + acceptance-stall + auto-flush)

## Quick Start

```bash
git clone https://github.com/andreasknopke/qwen38-one-spark.git
cd qwen38-one-spark

# Current variant (approx. 10-11 min boot time)
bash serve-next.sh
```

Configurable via environment variables (defaults are the tuned, verified values):

| Variable | Default | Description |
|---|---|---|
| `IMG` | `lmsysorg/sglang:dev-qwen38-next-local` | Base image |
| `MEMFRAC` | `0.79` | KV cache fraction of GPU memory |
| `CTX` | `262144` | Context length |
| `PREFILL` | `2048` | Chunked prefill size |
| `SPEC` | `1` | Spec mode (`1` = NEXTN, `0` = off) |
| `ATTN_PREFILL` / `ATTN_DECODE` | `triton` / `trtllm_mha` | Attention backends — split because flashinfer prefill hits a CUTLASS SM120 kernel that will not compile |
| `PLE_DIR` | `$HOME/flashnext-ple-next` | File-backed PLE table (see *Disk*) |
| `RSS_BUDGET_GB` | `8` | Resident-set cap for the mmapped PLE table |
| `DEGEN_FORENSIC` | `1` | Debug logging for degeneration |
| `PORT` | `30000` | Server port |

## Why This Variant Drops Our Own Loop Fix

The previous variant stacked a **local patch layer over an old nightly image**: our
`disable_chunked_radix_insert` guard for the chunked-prefill radix-insert race
(PR [#38355](https://github.com/sgl-project/sglang/pull/38355), issue
[#38319](https://github.com/sgl-project/sglang/issues/38319)), local SM121/KDA attention
kernels and a PLE-mmap patch.

The current base image (`lmsysorg/sglang:dev-qwen38-next-local`, branch
`qwen4-main-squashed`, commit `4ccff141db`, built 2026-09-07) carries that whole area
**upstream in the tree itself**:

- [#37068](https://github.com/sgl-project/sglang/pull/37068) — file-backed PLE table backend for unified-memory devices (replaces our PLE-mmap patch)
- [#38121](https://github.com/sgl-project/sglang/pull/38121) — ModelOpt MIXED_PRECISION loader for `nvidia/*-Flash-Next-NVFP4`
- [#38290](https://github.com/sgl-project/sglang/pull/38290) / [#38308](https://github.com/sgl-project/sglang/pull/38308) / [#36811](https://github.com/sgl-project/sglang/pull/36811) — GB10 NEXTN NaN fixes and the router-bias/PDL kernel fix (Triton + radix), i.e. the image's own head commit

So the parts of our layer that existed to compensate for missing GB10 kernel work are obsolete,
and the corruption class we worked around is being fixed **in the same code path upstream**:
[#40075](https://github.com/sgl-project/sglang/pull/40075) *“Release up to `owned_kv_len` on
radix cache insert”* (merged 2026-09-18) lands exactly in the radix-insert bookkeeping our guard
policed. Our #38355 and the report #38319 are still open.

**Consequence for this variant:** we run the new tree *without* our own patch, as shipped, and
let the in-server Degeneration Guard be the safety net instead of pre-emptively changing upstream
cache semantics. The measured delta of our patch against the new tree is small
(`mamba_radix_cache.py` 10 lines, `cache_init_params.py` 7 lines), so it can be re-applied if the
loop ever shows up again — the intact version stays in `build/` + `serve.sh`, with notes in
[`build-next/_radix-nicht-portiert/README.txt`](build-next/_radix-nicht-portiert/README.txt).

## Background: the “!!!!” Loop (previous variant)

Qwen3.8-Flash-Next on DGX Spark had a stochastic bug in the interaction between chunked prefill
and the radix cache. A race condition produces a corrupted KV page; the radix cache then
perpetuates that corruption — every subsequent request matching the same prefix reads the corrupt
attention data and produces **token ID 248319** (the never-trained last row of `lm_head`, which
appears as `"!"` in the client).

- **Symptoms:** sudden `"!!!!"` mid-session, request terminates after 1–2 tokens with ID 248319, every retry with the same history crashes identically, `flush_cache` resolves it temporarily
- **Evidence:** 4 consecutive forensic incidents show **identical** KV slot addresses with varying mamba pool indices — the bug lives in the KV cache, not in the model state
- **Root cause:** a race between `cache_unfinished_req` (inserts KV pages into the radix tree mid-prefill) and `cache_finished_req` (frees pages on retract/abort) — the radix node survives with dangling references to freed pages
- **Old fix:** skip the radix insert during chunked prefill and defer it to `cache_finished_req` (PR #38355, `--disable-chunked-radix-insert`, auto-enabled for QSA compressed-attention models)

A second, still-unexplained corruption path appeared in the old variant on 2026-09-14 (different
slot addresses, fresh prefill) — that is issue #1 of this repo.

## Degeneration Guard (part of both variants)

`build-next/batch_result_processor.py` is the only file mounted over the new image:

| Guard | Behaviour |
|---|---|
| v6 impossible-token detection | Token id ≥ 248077 (tokenizer length) on the first decode steps → drop the junk output (`del req.output_ids[:]`) and abort with a retryable reason, so the client never sees `!!` |
| Acceptance-stall detector | `num_correct_drafts == 0` for 32 consecutive verify steps → clean finish, no radix insert (the Token-0 loop is invisible in `output_ids`, only in the accept rate) |
| Auto-flush | After a v6 trigger, a background `POST /flush_cache` purges the corrupt radix prefix (fires only when the scheduler is idle) |
| Identical-run detector | **Disabled** (`SGLANG_DEGEN_WINDOW=999999`) — it produced false positives on legitimate number/test-vector output |

The guard fires *before* corrupt tokens are streamed: it drops the junk output and aborts the
request with a retryable `FINISH_ABORT("kv_cache_corruption: ... safe to retry")`, so the client
gets a transient error it can retry instead of `!!`.

## Requirements

- 1× NVIDIA DGX Spark / GB10 (128 GB unified memory)
- Docker with the NVIDIA container toolkit
- Disk: ~126 GB model cache **plus** ~48 GiB for the PLE table
- Internet for the one-time model download

### Disk: the PLE table is a boot artefact

The file-backed PLE offload creates a **sparse 47.7 GiB file**
(`ple_table_<rows>x160_float8_e4m3fn_51200245760B.bin`) under `PLE_DIR`. The name contains dtype
and shape, so a table written by the other variant is *not* reused (the old layer's file was
`ple_table_51200245760_51200245760.bin`). Both backends only create the file when it is missing
or the size does not match, and it is rewritten on every boot — it is a derived artefact and may
be deleted to free space (delete the other variant's file first when switching; this box runs at
~97 % disk).

## Details

### Model

- **Base**: `RadixArk/Qwen3.8-Flash-Next-NVFP4` (NVIDIA ModelOpt NVFP4, MIXED_PRECISION)
- **Architecture**: 180B MoE, QSA sparse attention + hybrid mamba, PLE n-gram embeddings
- **Context**: 262144 tokens (maximum), chunked prefill 2048

### Serving Stack

No custom Docker build — everything is a bind-mount over one pinned image. That is also why a
newer image must never be adopted without diffing the guard layer against it (see
`build-next/_radix-nicht-portiert/README.txt` for the measured deltas).

### Attention Backend

| Phase | Backend |
|---|---|
| Prefill | Triton (QSA sparse) |
| Decode | `trtllm_mha` + KDA kernel |

## Repository Structure

```
.
├── serve-next.sh                       # CURRENT variant: new image + guard-only layer
├── build-next/
│   ├── batch_result_processor.py       # Degeneration Guard (v3/v5/v6 + auto-flush)
│   ├── path_degen.txt                  # mount target inside the new image
│   └── _radix-nicht-portiert/          # notes: why the radix patch is not mounted
├── ROLLBACK/README.txt                 # both image digests + rollback commands
├── serve.sh                            # ROLLBACK variant: old image + full patch layer
├── build/                              # ROLLBACK layer (same file names, pinned per-file paths)
├── batch_result_processor.py           # scheduler guards (v6 + auto-flush)
├── eagle_worker_v2.py                  # EAGLE spec-decode
├── kda_kernels/                        # KDA decode kernel
├── mem_cache/                          # cache_init_params, kv_cache_builder, mamba_radix_cache
├── qsa_graph_metadata.py               # QSA graph metadata
├── qsa_kv_pool.py                      # QSA KV pool
├── qsa_metadata.py                     # QSA metadata
├── qwen4_exp.py                        # model architecture (Qwen4-Exp)
├── qwen_sparse_attn_backend.py         # sparse attention backend
├── server_args.py                      # server args (+ disable_chunked_radix_insert)
└── sm121_varlen.py                     # SM121 Triton varlen kernel
```

## History

This repo grew out of production work with Qwen3.8-Flash-Next on a single DGX Spark. The original
`"!"` KV loop ([#38319](https://github.com/sgl-project/sglang/issues/38319)) was traced to a
chunked-prefill radix-insert race; the fix
([#38355](https://github.com/sgl-project/sglang/pull/38355)) was not merged upstream and was
carried here as a patch layer over an old nightly image.

**2026-09-20 — switch to the upstream tree.** We moved to the newer `dev-qwen38-next-local` image
(commit `4ccff141db`), which brings the GB10 kernel work, the MIXED_PRECISION loader and the
file-backed PLE backend upstream, and reduced our patch layer to the Degeneration Guard. The
previous deployment stays fully reproducible (`serve.sh` + `build/` + the pinned image tag) — see
`ROLLBACK/README.txt`.
