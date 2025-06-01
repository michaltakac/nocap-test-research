import numpy as np
import glob
import os
import tqdm
import collections
import argparse
import torch

# -----------------------------------------------------------------------------
# Build top-K continuation tables from .bin shards.
# Optional GPU acceleration: set --device cuda
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--bin_pattern", default="data/fineweb10B/fineweb_train_*.bin")
parser.add_argument("--vocab_size", type=int, default=50257)
parser.add_argument("--k", type=int, default=16)
parser.add_argument("--out", default="data/fineweb10B/bigram_topk16.pt")
parser.add_argument("--format", choices=["npz", "pt"], default="pt", help="Output file format: npz or pt (torch)")
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Counting device (cpu/cuda)")
args = parser.parse_args()

device = torch.device(args.device)

# We accumulate counts row-wise to avoid huge (V×V) memory usage.
counts = [collections.Counter() for _ in range(args.vocab_size)]

for fname in tqdm.tqdm(sorted(glob.glob(args.bin_pattern))):
    with open(fname, "rb") as f:
        _ = f.read(1024)  # skip header
        buf = np.frombuffer(f.read(), np.uint16)

    # cast once to int32 for torch ops (argsort unsupported on uint16)
    buf_i32 = buf.astype(np.int32, copy=False)

    if args.device == "cuda":
        t = torch.from_numpy(buf_i32).to(device, non_blocking=True)
        prev = t[:-1]
        nxt  = t[1:]

        # build histogram of prev tokens in 2^16 buckets
        B = 2 ** 16
        buckets = (prev & (B - 1)).long()          # low-16 bits
        seg_sizes = torch.bincount(buckets, minlength=B)

        # cumulative sum gives start index per bucket
        starts = torch.cumsum(seg_sizes, 0, dtype=torch.int64)
        slots  = torch.empty_like(prev)

        # scatter prev's indices into "slots" so that
        # same-bucket items sit contiguously (linear-time radix pass)
        torch.cumsum(seg_sizes, 0, out=starts)
        slots.index_copy_(0, starts[buckets] - 1, nxt)

        # now iterate bucket-wise (each ≤ 65 536) and update Counters
        offset = 0
        for b, sz in enumerate(seg_sizes.tolist()):
            if sz == 0: continue
            segment_prev = (b + (prev[offset] & ~ (B - 1))).item()
            counts[segment_prev].update(slots[offset:offset + sz].tolist())
            offset += sz
    else:  # cpu path (original)
        prev = buf_i32[:-1]
        nxt  = buf_i32[1:]
        for p, n in zip(prev, nxt):
            counts[int(p)].update([int(n)])

K = args.k
tokens = np.full((args.vocab_size, K), -1, np.int32)
probs  = np.zeros((args.vocab_size, K),  np.float32)
for i, c in enumerate(counts):
    top = c.most_common(K)
    if not top: continue
    tok, cnt = zip(*top)
    s = sum(cnt)
    tokens[i, :len(tok)] = tok
    probs [i, :len(tok)] = np.asarray(cnt, np.float32) / s
os.makedirs(os.path.dirname(args.out), exist_ok=True)

if args.format == "npz":
    np.savez_compressed(args.out, tokens=tokens, probs=probs)
else:  # pt
    torch_tokens = torch.from_numpy(tokens).to(torch.uint16)
    torch_probs  = torch.from_numpy(probs ).to(torch.float16)
    torch.save({"tokens": torch_tokens, "probs": torch_probs}, args.out)

print("saved", args.out)