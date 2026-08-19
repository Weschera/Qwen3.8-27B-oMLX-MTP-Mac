# Qwen 3.8 27B on Apple Silicon: 48–65 tok/s with oMLX native MTP

Measured recipe for the fastest single-stream Qwen 3.8 27B serving we've
achieved on a Mac — **48.0 tok/s prose / 65.5 tok/s code** on an M4 Max,
roughly **2.6× the engine's own baseline** and **+11–16% over the previous
best Mac stack** (mlx-vlm + MTP sidecar), re-benchmarked the same day on the
same machine with the same prompts.

Stock model. No cloud. Every number below is from a real run on
2026-08-18; raw JSON is in this repo.

## Measured ladder (M4 Max, 128 GB, macOS 26)

Single stream, temperature 0, thinking off, 320 generated tokens,
~2.1–2.4K-token prompts, 3 reps each (mean, with min–max in the raw JSON):

| Config | Prose tok/s | Code tok/s |
|---|---|---|
| **oMLX 0.6.1 + native MTP, k=3** | **48.0** | **65.5** |
| oMLX + MTP k=2 | 43.5 | 56.3 |
| mlx-vlm + MTP sidecar (prior Mac champion, same-day rerun) | 43.4 | 56.7 |
| oMLX baseline (MTP off) | 24.9 | 25.9 |

Code accelerates more than prose because MTP draft acceptance is higher on
code. k=3 beat k=2 on this chip in both domains — measure on yours.

## Recipe

```bash
# 1. install oMLX (production MLX server with native MTP)
brew install omlx

# 2. pull the oQ4e quant (4-bit affine g64 with the 166 most sensitive
#    tensors kept at 5-bit — this is a STOCK Qwen 3.8 27B conversion)
hf download Jundot/Qwen3.8-27B-oQ4e-mtp

# 3. enable MTP in ~/.omlx/model_settings.json:
{
  "version": "1.0",
  "models": {
    "Jundot--Qwen3.8-27B-oQ4e-mtp": {
      "qwen35_ane_prefill_enabled": false,
      "mtp_enabled": true,
      "mtp_num_draft_tokens": 3
    }
  }
}

# 4. restart — settings apply ONLY at startup
omlx restart

# 5. it serves an OpenAI-compatible API (check the port with lsof —
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
`mlxvlm.json` (the prior-champion rerun).

## Gotchas we hit

- **Settings apply only at server start.** Editing `model_settings.json`
  does nothing until `omlx restart`.
- **Find the real port** with
  `lsof -nP -iTCP -sTCP:LISTEN -a -p $(pgrep -f omlx-server)` — don't assume.
- The managed brew service auto-starts at boot and respawns on crash — nice
  for appliances (ours powers a desk robot), just remember it's holding
  ~17 GB whenever the Mac is on. `omlx stop` to reclaim.
- `bench.py` needs `transformers` for tokenizer-length padding; point
  `--model-path` at a local snapshot dir (repo ids with `--` are rejected).
- Memory-cap warning about `iogpu.wired_limit_mb` is benign for this model
  size on 64 GB+ machines.

## What we didn't test (yet)

The recipe this builds on also ships an ANE (Neural Engine) prefill offload
that reportedly helps long-prompt TTFT by ~20%. It only affects prefill —
decode (the numbers above) is untouched — and needs a from-source kernel
build, so we skipped it for this round.

## Credits

- [oMLX](https://github.com/omlx) — the engine and its native MTP
  implementation.
- [drowzeys' DualANE recipe](https://github.com/drowzeys/keys-MAC-oMLX-0.6.1-DualANE-Qwen3.8-27B-Abliterated-oQ4e-MTP)
  — the config this is derived from (theirs: abliterated checkpoint, M3
  Ultra, ANE prefill; ours: stock checkpoint, M4 Max, decode-focused).
- [Jundot's oQ4e quant](https://huggingface.co/Jundot/Qwen3.8-27B-oQ4e-mtp).
- Prior champion for comparison: mlx-vlm with an MTP draft sidecar.

Part of a series of measured local-serving recipes:
[github.com/Weschera](https://github.com/Weschera?tab=repositories).
