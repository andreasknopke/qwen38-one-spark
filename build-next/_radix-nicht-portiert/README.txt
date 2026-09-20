# Merge-Versuche für den Radix-Race-Patch (PR #38355) auf den neuen Branch

NICHT MOUNTEN — der neue Layer (serve-next.sh) mountet bewusst nur
../batch_result_processor.py (Degeneration Guard).

KORREKTUR 2026-09-20: Die frühere Notiz ("mamba_radix_cache.py im neuen Branch
umgebaut, 483 statt 1427 Zeilen, nicht portierbar") war FALSCH. Gemessen
(neues Image lmsysorg/sglang:dev-qwen38-next-local gegen den lokalen
Patch-Layer ../build/):

| Datei (sglang/srt/...)          | Delta neues Image vs. Patch-Layer |
|---|---|
| mem_cache/mamba_radix_cache.py  | 10 Zeilen (nur unser Skip-Block + Flag-Attribut) |
| mem_cache/cache_init_params.py  | 7 Zeilen (nur unser Feld) |
| mem_cache/kv_cache_builder.py   | 103 Zeilen (unser Auto-Enable + Upstream-Änderungen) |
| server_args.py                  | 39 Zeilen (u.a. unsere Flag-Definition) |

Der Patch ist also im Prinzip portierbar. Er ist hier trotzdem NICHT im Einsatz:
Das neue Image (Commit 4ccff141db, 2026-09-07) fährt die betroffenen Pfade mit
eigenen GB10-/QSA-Kerneln, und die Upstream-Fixes an genau dieser Race-Klasse
sind unterwegs bzw. noch offen (#40075 "Release up to owned_kv_len on radix
cache insert", merged 2026-09-18 — noch nicht in diesem Image; #38355 von uns
und #38319 bleiben offen). Wir testen den neuen Baum ohne eigenen Eingriff, mit
dem Degeneration Guard als Sicherheitsnetz.

Diese .py-Dateien sind die alten Merge-Versuche (eine mit Konfliktmarkern) und
werden per .gitignore nicht committet. Der intakte Radix-Patch liegt unverändert
in ../build/ + ../serve.sh (altes Image, Rollback-Variante).
