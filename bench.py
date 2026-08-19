#!/usr/bin/env python3
"""Self-authored single-stream decode + prefill benchmark for a local oMLX server.

Modeled on the recipe's bench_mtp.py / bench_pp.py (both read and verified) but
written locally. Raw /v1/completions (no chat template -> no thinking), temp 0.
"""
import argparse, json, os, statistics, time, urllib.request

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8083/v1")
MODEL = os.environ.get("BENCH_MODEL", "qwen38-27b-oq4e")
API_KEY = os.environ.get("BENCH_KEY", "")

FILLER = (
    "Unified memory on Apple silicon removes the discrete host-to-device copy, so prefill and decode "
    "share one allocator. Prefill is compute bound; decode is bandwidth bound and rarely saturates the "
    "matrix units. Schedulers therefore chunk prompts and interleave decode steps to keep both busy. "
)

CODE_BODY = '''class InventoryLedger:
    def __init__(self, warehouse_id, clock):
        self.warehouse_id = warehouse_id
        self.clock = clock
        self._items = {}
        self._audit = []

    def add_item(self, sku, quantity, unit_cost):
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        entry = self._items.setdefault(sku, {"quantity": 0, "unit_cost": unit_cost})
        entry["quantity"] += quantity
        self._audit.append((self.clock.now(), "add", sku, quantity))
        return entry["quantity"]

    def remove_item(self, sku, quantity):
        entry = self._items.get(sku)
        if entry is None or entry["quantity"] < quantity:
            raise KeyError(sku)
        entry["quantity"] -= quantity
        self._audit.append((self.clock.now(), "remove", sku, quantity))
        return entry["quantity"]

    def total_value(self):
        return sum(e["quantity"] * e["unit_cost"] for e in self._items.values())

    def audit_trail(self):
        return list(self._audit)
'''

TASKS = {
    "prose": "Explain, in about 400 words, why token generation on a large language model is bound by "
             "memory bandwidth rather than arithmetic throughput, and what that implies for batching.",
    "code": "Here is a Python class:\n\n" + CODE_BODY +
            "\nRe-emit the entire class unchanged, adding one new method `low_stock(threshold)` that "
            "returns a sorted list of SKUs whose quantity is below the threshold, and one method "
            "`restock_report()` returning a dict of sku -> quantity. Output only code.",
}


def post_stream(prompt, max_tokens):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    hdr = {"Content-Type": "application/json"}
    if API_KEY:
        hdr["Authorization"] = "Bearer " + API_KEY
    req = urllib.request.Request(URL + "/completions", data=json.dumps(body).encode(), headers=hdr)
    t0 = time.perf_counter()
    ttft, out, usage, ntok = None, [], None, 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            piece = (chunk.get("choices") or [{}])[0].get("text") or ""
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                out.append(piece)
                ntok += 1
            if chunk.get("usage"):
                usage = chunk["usage"]
    return {"ttft": ttft, "total": time.perf_counter() - t0,
            "text": "".join(out), "usage": usage or {}, "ntok": ntok}


def pad_prompt(tok, task_text, marker, ctx):
    pre = f"[{marker}]\n" + FILLER * 400
    ids = tok.encode(pre)
    return tok.decode(ids[:ctx]) + "\n\n" + task_text


def build_prefill_prompt(tok, n_tokens, marker):
    text = marker + "\n" + FILLER * (n_tokens // 40 + 8)
    ids = tok.encode(text)
    while len(ids) < n_tokens:
        text += FILLER
        ids = tok.encode(text)
    return tok.decode(ids[:n_tokens])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=320)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--prefill", type=int, default=0, help="also run N-token prefill reps")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path)

    if a.warmup:
        post_stream("Warmup: say hi.", 16)

    results = []
    for name, task in TASKS.items():
        tps = []
        for i in range(a.repeats):
            p = pad_prompt(tok, task, f"{a.tag}-{name}-{i}-{i*7919}", a.ctx)
            r = post_stream(p, a.gen)
            gen_t = r["total"] - (r["ttft"] or 0)
            n = r["usage"].get("completion_tokens") or r["ntok"]
            d = (n - 1) / gen_t if gen_t > 0 and n else None
            tps.append(d)
            rec = {"tag": a.tag, "task": name, "rep": i, "decode_tps": d,
                   "gen_tokens": n, "ttft": r["ttft"], "total": r["total"],
                   "prompt_tokens": r["usage"].get("prompt_tokens"),
                   "cached": (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"),
                   "text_head": r["text"][:80]}
            results.append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "text_head"}), flush=True)
        good = [t for t in tps if t]
        if good:
            print(f"== {a.tag} {name}: mean {statistics.mean(good):.1f} tok/s "
                  f"(min {min(good):.1f} / max {max(good):.1f}, n={len(good)})", flush=True)

    if a.prefill:
        pps = []
        for i in range(a.repeats):
            p = build_prefill_prompt(tok, a.prefill, f"PF-{a.tag}-{i}-{i*7919}")
            r = post_stream(p, 64)
            npt = r["usage"].get("prompt_tokens") or a.prefill
            pp = npt / r["ttft"] if r["ttft"] else None
            pps.append(pp)
            rec = {"tag": a.tag, "kind": "prefill", "size": a.prefill, "rep": i,
                   "ttft": r["ttft"], "prompt_tokens": npt, "pp": pp,
                   "cached": (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")}
            results.append(rec)
            print(json.dumps(rec), flush=True)
        good = [x for x in pps if x]
        if good:
            print(f"== {a.tag} prefill{a.prefill}: mean {statistics.mean(good):.1f} tok/s PP "
                  f"(min {min(good):.1f} / max {max(good):.1f})", flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
