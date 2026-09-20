Rollback-Stand 2026-09-19T20:44:12+02:00
AKTUALISIERT 2026-09-20T12:20:00+02:00 (Deploy des neuen Images erledigt + verifiziert)

=== AKTUELL AKTIV (seit 2026-09-20 12:12 CEST) ===

NEUES Image:
 sha256:cdd9649ba1cf472344fd1e11e7cbaa7161a0624329b522931646537cc1c15701  [lmsysorg/sglang@sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6]
 Tag: lmsysorg/sglang:dev-qwen38-next-local  (Branch-Commit 4ccff141db, CI-Run 34109466898)

Neuer Layer: build-next/ (NUR Degeneration Guard v3/v5/v6 + Auto-Flush)
Neues Serve-Script: serve-next.sh
Container: flashnext-next  (Port 30000, MEMFRAC=0.79 CTX=262144 PREFILL=2048 SPEC=1)
PLE: ~/flashnext-ple-next/ple_table_320001536x160_float8_e4m3fn_51200245760B_rows0-320001536.bin (47.7 GiB, FP8)
Boot verifiziert: /health 200 nach 10 min 24 s (10:01:25 -> 10:12:49 UTC); server_args:
 context_length=262144, mem_fraction_static=0.79, ple_offload_backend=file, page_size=64.
Smoke-Tests: /v1/models ok, greedy Chat ok, Tool-Call (qwen3_coder) ok, 2x greedy identisch.
Guard aktiv im Container: batch_result_processor.py md5 ce8ca9e9a66bbaedfe140aef5eee1cff
 (identisch zu build-next/), SGLANG_DEGEN_GUARD=1 WINDOW=999999 IMPOSSIBLE_TOKEN=1 MAX=248077.

=== ROLLBACK-ZIEL (unverändert vorhanden) ===

ALTEST Image:
 sha256:64c58f100438fa5f036bdfbeb3edd3136fb12c5d22d8ae52786c4a701263c55d  [lmsysorg/sglang@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1]
 Tag: lmsysorg/sglang:qwen38flashnext  (nightly-dev-20260817-d91c3682 + RadixArk-Overlay)

Alter Layer: build/ (vollständig, unverändert: Radix-Race-Patch PR #38355,
 --disable-chunked-radix-insert in serve.sh, sm121/KDA-Patches, PLE-mmap-Patch)
Container flashnext: gestoppt, NICHT gelöscht.

Rollback-Kommandos:
  # A) schnellster Weg (alte CLI + alte Mounts):
  docker rm -f flashnext-next && docker start flashnext
  # B) sauberer Weg (recreate aus dem alten Layer):
  docker rm -f flashnext-next flashnext && cd ~/qwen38-one-spark && MEMFRAC=0.79 CTX=262144 PREFILL=2048 SPEC=1 bash serve.sh

Hinweise für den Rollback:
 * Die alte PLE-Datei ~/flashnext-ple/ple_table_51200245760_51200245760.bin wurde am
   2026-09-20 gelöscht (48 GiB, User-Freigabe). Sie ist reines Boot-Artefakt und wird beim
   Alten-Boot identisch neu erzeugt (identische Tabellen-Shape, FP8 47.7 GiB).
   Vor dem Rollback ggf. ~/flashnext-ple-next (48 GiB) löschen, sonst kein Platz.
 * Der Radix-Race-Patch ist NUR im alten Layer; im neuen Branch ist mamba_radix_cache.py
   umgebaut (483 vs 1427 Zeilen) -> nicht portiert, siehe build-next/_radix-nicht-portiert/.
 * Admin-Panel (Spark Admin :9001, Service qwen-flash) zeigt seit 2026-09-20 auf Container
   flashnext-next + start_script serve-next.sh. Für einen Rollback dort Container und
   start_script auf flashnext / serve.sh zurückstellen.
 * serve.sh ist gegenüber git HEAD um +1 Zeile geändert (--disable-chunked-radix-insert);
   build/ ist committet, der alte Layer ist also aus git reproduzierbar.
