# Qwen3.8-Flash-Next — One DGX Spark Recipe

Ein fertiges, ausführlich getestetes Rezept zum Betreiben von **Qwen3.8-Flash-Next 180B MoE (NVFP4)** auf einem einzelnen **NVIDIA DGX Spark (GB10)** — mit allen Patches, die den `"!" KV-Loop` endgültig besiegen.

## Status

- ✅ **Der "!"-Loop ist tot** — 5h+ stabile VSCode-Sessions ohne einen einzigen KV-Korruptions-Loop (vor dem Fix: alle 1-2h bei ~33k oder ~145k Kontext)
- ✅ ~35 tok/s Single-Stream, 85-95 tok/s aggregate
- ✅ 262K Kontext, Chunked Prefill 2048
- ✅ Speculative Decoding (EAGLE NEXTN, 3 Steps)
- ✅ KDA Decode-Kernel + SM121 Triton Fallback
- ✅ Automatischer Cache-Flush bei Detektion von KV-Korruption

## Problem: Der "!!!!" Loop

Qwen3.8-Flash-Next hat auf dem DGX Spark einen stochastischen Bug, der bei Chunked-Prefill + Radix-Cache eine korrupte KV-Seite produziert. Der Radix-Cache verewigt diese Korruption: jeder neue Request mit demselben Prefix kriegt auf Anhieb die korrupte Attention → das Modell produziert **Token 248319** (die letzte nie-trainierte Zeile des lm_head, dekodiert als `''` → vom Client als `"!"` dargestellt).

**Symptome:**
- Plötzliche "!!!!"-Ausgabe mitten in einer Session
- Request terminiert nach 1-2 Tokens mit Token 248319
- Alle nachfolgenden Requests mit gleicher History crashen identisch
- `flush_cache` hilft temporär

**Beweiskette:** Forensik-Dumps zeigen 4 aufeinanderfolgende Incidents mit **identischen** KV-Slot-Adressen und unterschiedlichen Mamba-Pool-Indizes — der Bug sitzt im KV-Cache, nicht im Modellzustand.

## Der Fix: `disable_chunked_radix_insert`

Die Wurzel ist ein Race zwischen Chunked-Prefill KV-Page-Insert in den Radix-Baum (`cache_unfinished_req`) und Pool-Deallocation bei Retract/Abort (`cache_finished_req`). Der Radix-Knoten überlebt mit hängenden Referenzen auf freigegebene Pages.

**Lösung:** Der Radix-Insert wird während Chunked-Prefill übersprungen und erst nach Request-Commit in `cache_finished_req` ausgeführt. Damit referenziert der Radix-Baum nie Pages, während sie noch freigegeben werden können.

Der Patch ist auto-enabled für QSA compressed-attention Modelle — kein manuelles Flag nötig.

**Fallback:** Sollte dennoch eine Korruption auftreten, triggert der v6 Guard automatisch einen `flush_cache` im Hintergrund — unsichtbar für den laufenden Request.

## Patches in diesem Repo

| Patch | Datei | Zweck |
|---|---|---|
| `disable_chunked_radix_insert` | `server_args.py`, `mem_cache/*.py` | **Der "!"-Loop-Fix** — verhindert die Race-Condition zwischen Chunked-Prefill-Insert und Radix-Retract, die korrupte KV-Seiten erzeugt |
| v6 impossible-token Guard | `batch_result_processor.py` | Erkennt Token-ID `248319` (out-of-vocab) sofort beim ersten Decode und terminiert den Request sauber |
| Auto-Flush | `batch_result_processor.py` | Nach einer v6-Alarm-Auslösung wird automatisch `flush_cache` im Hintergrund getriggert — der korrupte Radix-Prefix wird geräumt |
| Spezialisierte Attention-Overlays | `qwen_sparse_attn_backend.py`, `qsa_*.py`, `sm121_varlen.py`, `kda_kernels/` | SM121-optimierte QSA Sparse Attention + KDA Decode-Kernel (Triton + FlashInfer hybrid) |
| Mamba extra_buffer | `qwen4_exp.py` | Hybrid Mamba-Attention-Radix-Cache für das Flash-Next-Modell |
| Identical-run Guard deaktiviert | `batch_result_processor.py` | (Window=999999) — verhindert False Positives bei legitimen Testvektor-Ausgaben |

## Voraussetzungen

- 1× NVIDIA DGX Spark / GB10 (128 GB unified memory)
- Docker installiert
- ~126 GB freier Plattenplatz für Modell-Cache + PLE-Offload
- Internet für Model-Download (einmalig)

## Schnellstart

```bash
git clone https://github.com/andreasknopke/qwen38-one-spark.git
cd qwen38-one-spark

# Start (ca. 9 Min Bootzeit)
MEMFRAC=0.79 PREFILL=2048 CTX=262144 bash serve.sh
```

Die Umgebungsvariablen sind anpassbar:

| Variable | Default | Beschreibung |
|---|---|---|
| `MEMFRAC` | `0.79` | KV-Cache-Anteil am GPU-Speicher |
| `CTX` | `262144` | Kontextlänge |
| `PREFILL` | `2048` | Chunked-Prefill-Größe |
| `SPEC` | `1` | Spec-Modus (`1`=NEXTN, `ring8`, `adaptive`) |
| `PORT` | `30000` | Server-Port |
| `KVDTYPE` | `auto` | KV-Cache-Datentyp |
| `DEGEN_FORENSIC` | `1` | Debug-Logging für Degeneration |

## Details

### Modell

- **Base**: `RadixArk/Qwen3.8-Flash-Next-NVFP4` (NVIDIA ModelOpt NVFP4)
- **Architektur**: 180B MoE, 48 Layer, QSA Sparse Attention + Hybrid Mamba
- **Quantisierung**: NVFP4 (experts), BF16 (restliche Layer)
- **Context**: 262144 Tokens (maximal)

### Serving Stack

Das Image `lmsysorg/sglang:qwen38flashnext` wird verwendet. Alle Patches werden per Bind-Mount eingespielt — kein eigener Docker-Build nötig.

### Attention Backend

| Phase | Backend |
|---|---|
| Prefill | Triton (QSA sparse) |
| Decode | `trtllm_mha` + KDA-Kernel |

## Benchmark-Ergebnisse

Siehe `~/forensics/`-Verzeichnis im laufenden Betrieb für aktuelle Forensik-Snapshots.

## Repo-Struktur

```
.
├── serve.sh                    # Start-Script (Docker)
├── batch_result_processor.py   # Scheduler Guards (v6 + Auto-Flush)
├── eagle_worker_v2.py          # EAGLE Spec-Decode
├── kda_kernels/                # KDA Decode-Kernel
├── mem_cache/                  # cache_init_params, kv_cache_builder, mamba_radix_cache
├── qsa_graph_metadata.py       # QSA Graph Metadata
├── qsa_kv_pool.py              # QSA KV Pool
├── qsa_metadata.py             # QSA Metadata
├── qwen4_exp.py                # Model Architecture (Qwen4-Exp)
├── qwen_sparse_attn_backend.py # Sparse Attention Backend
├── server_args.py              # Server Args (+ disable_chunked_radix_insert)
└── sm121_varlen.py             # SM121 Triton Varlen Kernel
```

## History

Dieses Repo entstand aus der praktischen Arbeit mit Qwen3.8-Flash-Next auf dem DGX Spark. Der ursprüngliche "!" KV-Loop (Issue [#38319](https://github.com/sgl-project/sglang/issues/38319)) wurde durch einen Chunked-Prefill-Radix-Insert-Race verursacht. Der Fix (`disable_chunked_radix_insert`) wurde als PR [#38355](https://github.com/sgl-project/sglang/pull/38355) eingereicht, aber nicht gemerged. Dieses Repo führt den Fix als Fork fort.