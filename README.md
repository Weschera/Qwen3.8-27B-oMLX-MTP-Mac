# Qwen 3.8 27B on Apple Silicon: 53–72 tok/s with oMLX ANE + native MTP

Measured recipe for the fastest single-stream Qwen 3.8 27B serving we've
achieved on a Mac — **53.3 tok/s prose / 72.1 tok/s code** on an M4 Max,
**+11% over our previous best** (48.0 / 65.5) by adding ANE prefill to the
oMLX 0.6.3rc2 stack, re-benchmarked the same day on the same machine with
the same prompts.

Stock model. No cloud. Every number below is from a real run on
2026-08-21; raw JSON is in this repo.

## Measured ladder (M4 Max, 128 GB, macOS 26)

Single stream, temperature 0, thinking off, 320 generated tokens,
~2.1–2.4K-token prompts, 3 reps each (mean, with min–max in the raw JSON):

| Config | Prose tok/s | Code tok/s | Prefill 4K tok/s |
|---|---|---|---|
| **oMLX 0.6.3rc2 + ANE prefill + native MTP, k=3** | **53.3** | **72.1** | **273.7** |
| oMLX 0.6.1 + native MTP, k=3 (previous best, same recipe) | 48.0 | 65.5 | — |
| oMLX 0.6.3rc2 + MTP k=3, ANE off | 47.9 | 47.9 | — |
| oMLX + MTP k=4, ANE on | 47.9 | 73.3 | — |
| oMLX 0.6.3rc2 + ANE + CPU sharing + GDN, MTP k=3 | 52.3 | 72.0 | 277.4 |
| vllm-metal 0.3.0 (Qwen3.8 8-bit, no MTP) | — | 13.2 | — |
| DFlash2 (DeepSeek V4 Flash 2-bit, oMLX 0.6.3rc2) | 36.7 | — | — |

Code accelerates more than prose because MTP draft acceptance is higher on
code (99.6% on code vs 78% on prose). ANE prefill helps decode by reducing
prefill/decode interference in shared engine steps.

## What changed vs the previous recipe

The previous recipe (oMLX 0.6.1 + MTP k=3) hit 48/65.5. This update adds:

1. **oMLX 0.6.3rc2** — fixes hybrid-model KV-cache sizing (4× overestimate
   that caused false OOM and prefill throttling on Qwen3.5/3.6/3.8)
2. **ANE prefill enabled** — 64 MLP layers accelerated on the Neural Engine
   via drowzeys' prebuilt kernel. Prefill throughput jumped from ~83 to ~274
   tok/s at 4K, and decode improved ~10% because faster prefill reduces
   interference with decode steps in shared engine steps
3. **FP16 clone checkpoint** — created with `tools/clone_mlx_model_fp16.py`
   from the oMLX repo. Enables optional CPU sharing (not used in the winning
   config, but available for experimentation)

## Recipe

```bash
# 1. install oMLX 0.6.3rc2
brew install omlx

# 2. pull the oQ4e quant (4-bit affine g64 with the 166 most sensitive
#    tensors kept at 5-bit — this is a STOCK Qwen 3.8 27B conversion)
hf download Jundot/Qwen3.8-27B-oQ4e-mtp

# 3. install the ANE prefill kernel (prebuilt, from drowzeys' DualANE recipe)
curl -sSLO https://github.com/drowzeys/keys-MAC-oMLX-0.6.1-DualANE-Qwen3.8-27B-Abliterated-oQ4e-MTP/releases/download/v1.0/qwen35_prefill-ane-kernel-omlx0.6.1-py311-mlx0.32.0.tar.gz
tar xzf qwen35_prefill-ane-kernel-omlx0.6.1-py311-mlx0.32.0.tar.gz
SITE=$(python3 -c "import omlx, os; print(os.path.dirname(omlx.__file__))")/custom_kernels/qwen35_prefill
cp qwen35_prefill/_ext.cpython-311-darwin.so "$SITE/"
cp qwen35_prefill/*.dylib "$SITE/"
cp qwen35_prefill/*.metallib "$SITE/"

# 4. enable ANE prefill + MTP in ~/.omlx/model_settings.json:
{
  "version": "1.0",
  "models": {
    "Jundot--Qwen3.8-27B-oQ4e-mtp": {
      "qwen35_ane_prefill_enabled": true,
      "qwen35_ane_prefill_sequence_length": 2048,
      "qwen35_ane_prefill_fraction": 0.5,
      "qwen35_ane_prefill_max_layers": 64,
      "qwen35_ane_prefill_dual_ane": true,
      "qwen35_ane_prefill_gdn": false,
      "mtp_enabled": true,
      "mtp_num_draft_tokens": 3,
      "context_window": 262144
    }
  }
}

# 5. restart — settings apply ONLY at startup
omlx restart

# 6. it serves an OpenAI-compatible API (check the port with lsof —
#    ours landed on 127.0.0.1:8083)
curl http://127.0.0.1:8083/v1/models
```

For conversational use, disable thinking per request:

```json
{"chat_template_kwargs": {"thinking": false, "enable_thinking": false}}
```

## Verify it yourself

`bench.py` is the exact harness behind the table (completions endpoint);
`bench-chat.py` is the same prompts adapted for chat-only servers.

```bash
BENCH_URL=http://127.0.0.1:8083/v1 BENCH_MODEL=Jundot--Qwen3.8-27B-oQ4e-mtp \
python bench.py --tag mine \
  --model-path <local tokenizer dir> --out results.json --warmup
```

Raw results from our runs: `baseline.json`, `mtp2.json`, `mtp3.json`,
`mlxvlm.json` (previous recipe), `ane-mtp3-results.json` (this recipe).

## Gotchas we hit

- **Settings apply only at server start.** Editing `model_settings.json`
  does nothing until `omlx restart`.
- **Find the real port** with
  `lsof -nP -iTCP -sTCP:LISTEN -a -p $(pgrep -f omlx-server)` — don't assume.
- **The ANE kernel is not in the brew bottle.** You must install it
  separately from drowzeys' prebuilt release. Without it, oMLX 0.6.3rc2
  reports `qwen35_prefill: available: false` (circular import) and ANE
  prefill stays off — decode falls back to the GPU-only path.
- **CPU sharing requires an FP16 clone.** Create one with
  `tools/clone_mlx_model_fp16.py` from the oMLX repo. CPU fractions must be
  ≤ 0.25 (the engine rejects higher values). In our testing, CPU sharing +
  GDN did not improve decode over the simpler ANE-MLP-only config.
- **MTP k=4 helps code slightly (+1.2 tok/s) but hurts prose (-5.4 tok/s).**
  k=3 is the best all-rounder.
- **The managed brew service auto-starts at boot** and respawns on crash.
  It holds ~17 GB (or ~38 GB with the FP16 clone loaded) whenever the Mac
  is on. `omlx stop` to reclaim.
- **Memory-cap warning** about `iogpu.wired_limit_mb` is benign for this
  model size on 64 GB+ machines.

## Why ANE prefill helps decode

oMLX shares a single engine between prefill and decode. When a prefill is in
flight, decode steps wait. ANE prefill offloads MLP computation to the
Neural Engine, finishing prefills faster and reducing the window where
decode is blocked. The result: ~10% higher decode throughput with no change
to the decode path itself. Prefill throughput at 4K jumped from ~83 to
~274 tok/s (3.3×).

MTP acceptance rates from the logs:
- Code: 99.6% (tok/cycle = 3.76, near-theoretical max for k=3)
- Prose: 75–80% (tok/cycle = 2.5–2.8)

Raw decode (MTP off) is ~24.6 tok/s. MTP multiplies that by the tok/cycle
ratio. The decode ceiling is `raw_decode × tok/cycle`, not prefill.

## What we tested that didn't help

- **vllm-metal 0.3.0** — 13 tok/s on the 8-bit checkpoint, ~3.6× slower
  than oMLX. Easy install, auto-detects Apple Silicon, but no MTP and no
  ANE prefill. Does not support DeepSeek V4 Flash.
- **DFlash2 (DeepSeek V4 Flash 2-bit)** — 37 tok/s on oMLX, slower than
  Qwen3.8 oQ4e + MTP. Uses 94 GB of 128 GB (1M context KV cache). Required
  capping context to 32K to avoid hanging oMLX. Not on vllm-metal.
- **DFlash speculative decoding** — the Inkling-Small DSpark draft model
  failed with "Received 4 parameters not in model" (incompatible config).
  Needs a compatible Qwen3.5 draft model.
- **CPU sharing + GDN on ANE** — prefill improved slightly (277 vs 274
  tok/s) but decode was flat (52.3 vs 53.3). Not worth the 21 GB extra
  memory for the FP16 clone.
- **Higher ANE fraction (0.75)** — worse prefill (257 vs 274) and same
  decode. fraction=0.5 is the sweet spot.

## Watch list

**DFlash speculative decoding** with a proper Qwen3.5 draft model could
theoretically beat MTP on prose (where MTP acceptance is only 78%). Needs
a compatible small checkpoint — we haven't found one yet.

**oMLX 0.6.3 stable** (when it drops from rc2) may fix the custom kernel
circular import so the prebuilt ANE kernel isn't needed separately.

## Credits

- [oMLX](https://github.com/jundot/omlx) — the engine, native MTP, and ANE
  prefill framework.
- [drowzeys' DualANE recipe](https://github.com/drowzeys/keys-MAC-oMLX-0.6.1-DualANE-Qwen3.8-27B-Abliterated-oQ4e-MTP)
  — the prebuilt ANE kernel and the config this is derived from (theirs:
  abliterated checkpoint, M3 Ultra, dual-ANE prefill; ours: stock
  checkpoint, M4 Max, decode-focused).
- [Jundot's oQ4e quant](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp).
- Previous recipe for comparison: oMLX 0.6.1 + MTP k=3 (this repo's
  original release).

Part of a series of measured local-serving recipes:
[github.com/Weschera](https://github.com/Weschera?tab=repositories).
