"""nlaB2_run_nla.py -- run the pre-trained NLA AV (kitft/nla-gemma3-12b-L32-av)
on the activations extracted by nlaB1_extract_L32.py.

Assumes SGLang has been launched in another process:

    HF_TOKEN=... python -m sglang.launch_server \\
        --model-path kitft/nla-gemma3-12b-L32-av \\
        --port 30000 --disable-radix-cache --context-length 512 \\
        --mem-fraction-static 0.85 --trust-remote-code \\
        --attention-backend fa3 &

The fa3 attention backend is REQUIRED for Gemma-3 (head_dim=256; default
flashinfer OOMs). See the NLA docs/inference.md.

This script reads results/nlaB_L32_activations.parquet, calls the AV
once per row, and saves descriptions to results/nlaB_descriptions.json.

We use deterministic decoding (temperature=0) for reproducibility; one
sample per record. If you want exploratory variance, set --temperature 0.7
and --n-samples 3.

Run remotely:
    python paper/scripts/nlaB2_run_nla.py \\
        --sglang-url http://localhost:30000 \\
        --checkpoint kitft/nla-gemma3-12b-L32-av
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import numpy as np

# Import NLAClient from the cloned external repo. Assumes the repo is at
# paper/scripts/nla_external/natural_language_autoencoders/.
HERE = Path(__file__).resolve().parent
NLA_REPO_DIR = HERE / "nla_external" / "natural_language_autoencoders"
sys.path.insert(0, str(NLA_REPO_DIR))
from nla_inference import NLAClient  # noqa: E402

ROOT = HERE.parent.parent
ACT_PARQUET = ROOT / "results" / "nlaB_L32_activations.parquet"
OUT_JSON = ROOT / "results" / "nlaB_descriptions.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="kitft/nla-gemma3-12b-L32-av",
                        help="HF repo id OR local path to NLA AV checkpoint dir.")
    parser.add_argument("--sglang-url", default="http://localhost:30000")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy. NLA AV was RL-trained against a "
                             "fixed MSE objective; greedy is reproducible.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Number of NLA generations per activation (>1 "
                             "useful only if temperature>0).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of records to process (debugging).")
    args = parser.parse_args()

    print(f"[B2] loading NLA AV from {args.checkpoint!r}")
    client = NLAClient(args.checkpoint, sglang_url=args.sglang_url)

    print(f"[B2] reading activations from {ACT_PARQUET}")
    table = pq.read_table(ACT_PARQUET)
    n_records = len(table)
    print(f"     {n_records} records to process")

    if args.limit:
        n_records = min(n_records, args.limit)
        print(f"     limited to first {n_records} records")

    results = []
    t0 = time.time()
    sampling = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }

    # Process one at a time. SGLang handles batching server-side; we
    # keep the client-side flow simple. For higher throughput, batch
    # via NLAClient.generate_batch() — but our 420 records at ~1s each
    # is fine sequentially.
    for i in range(n_records):
        row = {k: table.column(k)[i].as_py() for k in table.column_names}
        act = torch.tensor(row["activation_vector"], dtype=torch.float32)

        samples = []
        for s in range(args.n_samples):
            try:
                text = client.generate(act, **sampling)
            except Exception as e:
                text = f"<ERROR: {e}>"
            samples.append(text)

        results.append({
            "record_id": row["record_id"],
            "case_id":   row["case_id"],
            "format":    row["format"],
            "kind":      row["kind"],
            "token_id":  row["token_id"],
            "token_str": row["token_str"],
            "chat_tok_idx": row["chat_tok_idx"],
            "samples":   samples,
        })

        if (i + 1) % 30 == 0 or (i + 1) == n_records:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_records - i - 1) / rate if rate > 0 else float("inf")
            print(f"  [{i+1}/{n_records}] {elapsed/60:.1f} min; "
                  f"~{rate:.2f} rec/s; ETA {eta/60:.1f} min")

    # ─── Save ────────────────────────────────────────────────────────────
    print(f"\n[B2] writing descriptions to {OUT_JSON}")
    OUT_JSON.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "sglang_url": args.sglang_url,
        "temperature": args.temperature,
        "n_samples": args.n_samples,
        "max_new_tokens": args.max_new_tokens,
        "n_records": len(results),
        "results": results,
    }, indent=2))
    print(f"[B2] DONE. {len(results)} records, elapsed {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
