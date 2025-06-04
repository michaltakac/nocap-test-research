import os
import sys
import uuid
import math
import glob
from dataclasses import dataclass

# Reduce VRAM usage by reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import nn
from torch.nn import AdaptiveLogSoftmaxWithLoss
torch.set_float32_matmul_precision('high')
# For BF16 performance
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
import torch.distributed as dist
import torch.nn.functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

with open(sys.argv[0]) as f:
    code = f.read()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = 1 / math.sqrt(2 * config.n_layer)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x


# -----------------------------------------------------------------------------
# Frequency counting and negative sampling utilities

def count_token_frequencies(filename_pattern, vocab_size):
    """Count token frequencies across all data shards for negative sampling."""
    files = sorted(glob.glob(filename_pattern))
    freqs = np.zeros(vocab_size, dtype=np.int64)
    
    print0(f"Counting token frequencies across {len(files)} files...")
    for fname in files:
        tokens = _load_data_shard(fname)
        # Use numpy bincount for efficient counting
        shard_freqs = np.bincount(tokens, minlength=vocab_size)
        freqs[:len(shard_freqs)] += shard_freqs
    
    print0(f"Token frequency counting complete. Total tokens: {freqs.sum():,}")
    return freqs


def build_alias_table(probs, table_size):
    """Build alias table for O(1) sampling using Walker's alias method."""
    n = len(probs)
    # For simplicity, make table_size = n and just use the probabilities directly
    table_size = n
    
    # Normalize probabilities
    probs = np.array(probs, dtype=np.float64)
    probs = probs / probs.sum()
    
    # Scale to table size
    scaled = probs * table_size
    
    alias = np.zeros(table_size, dtype=np.int32)
    prob = np.zeros(table_size, dtype=np.float32)
    
    # Separate indices into small and large based on scaled probabilities
    small = []
    large = []
    
    for i, p in enumerate(scaled):
        if p < 1.0:
            small.append(i)
        else:
            large.append(i)
    
    # Build the alias table
    for i in range(table_size):
        prob[i] = scaled[i]
        alias[i] = i
        
        if scaled[i] < 1.0 and len(large) > 0:
            # Pair this small probability with a large one
            j = large[0]
            alias[i] = j
            scaled[j] -= (1.0 - scaled[i])
            
            # Update large/small categorization
            if scaled[j] < 1.0:
                large.pop(0)
                small.append(j)
    
    return alias, prob


class UnigramSampler:
    """Efficient unigram sampler using alias method."""
    
    def __init__(self, freqs, power=0.75, table_size=1_000_000, device='cuda'):
        self.device = device
        self.vocab_size = len(freqs)
        self.power = power  # Store power for annealing
        
        # Convert frequencies to probabilities with power scaling
        probs = np.power(freqs + 1e-10, power)  # Add small epsilon to avoid zeros
        probs = probs / probs.sum()
        
        # Build alias table and move to GPU
        alias, prob = build_alias_table(probs, self.vocab_size)
        self.alias_table = torch.from_numpy(alias).to(device)
        self.prob_table = torch.from_numpy(prob).to(device)
        
        print0(f"Built unigram sampler with vocab size {self.vocab_size:,}")
    
    def sample(self, n):
        """Sample n tokens using the alias table."""
        # Generate random indices and probabilities
        indices = torch.randint(0, self.vocab_size, (n,), device=self.device)
        uniform = torch.rand(n, device=self.device)
        
        # Use alias table for sampling
        use_primary = uniform < self.prob_table[indices]
        primary_samples = indices
        alias_samples = self.alias_table[indices]
        
        return torch.where(use_primary, primary_samples, alias_samples)


class NegSamplingLoss(nn.Module):
    """Negative Sampling Loss module implementing NCE for language modeling."""
    
    def __init__(self, weight, k, sampler, shared_batch=False):
        super().__init__()
        self.k = k
        self.sampler = sampler
        self.shared = shared_batch
        # Register weight as buffer to avoid it being treated as a parameter
        self.register_buffer('weight', weight)
    
    def forward(self, h, target):
        """
        Compute negative sampling loss.
        
        Args:
            h: Hidden states (B*T, d)
            target: Target tokens (B*T,)
        """
        batch_size = h.shape[0]
        
        if self.shared:
            # Sample k negatives shared across the batch (memory-efficient)
            neg = self.sampler.sample(self.k)  # (k,)
            w_neg = self.weight[neg]           # (k, d)

            # Compute scores efficiently without expanding w_neg for every token
            noise_logprob = torch.log(self.sampler.prob_table[neg])  # (k,)
            # Positive correction still depends on each token
            noise_logprob_full = torch.log(self.sampler.prob_table)  # (V,)

            pos_corr = noise_logprob_full[target] + math.log(self.k)  # (B*T,)
            # h: (B*T,d)  w_neg.T: (d,k)  -> (B*T,k)
            s_pos = (h * self.weight[target]).sum(-1) - pos_corr
            s_neg = torch.matmul(h, w_neg.t()) - noise_logprob.unsqueeze(0)
        else:
            # Sample k negatives per token (fallback path, potentially high mem)
            neg = self.sampler.sample(batch_size * self.k).view(batch_size, self.k)

            w_pos = self.weight[target]  # (B*T, d)
            w_neg = self.weight[neg]     # (B*T, k, d)

            noise_logprob = torch.log(self.sampler.prob_table)   # (V,)
            pos_corr = noise_logprob[target] + math.log(self.k)
            neg_corr = noise_logprob[neg]                        # (B*T,k)

            s_pos = (h * w_pos).sum(-1) - pos_corr
            s_neg = torch.einsum('bd,bkd->bk', h, w_neg) - neg_corr
        
        loss = -F.logsigmoid(s_pos).mean() - F.logsigmoid(-s_neg).mean()
        return loss


# -----------------------------------------------------------------------------
# The main GPT-2 model


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):

    def __init__(self, config, use_asoft=False, cutoffs=None, div_value=4.0, 
                 use_neg_sampling=False, ns_k=20, sampler=None, shared_neg=False):
        super().__init__()
        self.config = config
        self.use_asoft = use_asoft
        self.use_neg_sampling = use_neg_sampling

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        
        if use_neg_sampling:
            assert sampler is not None, "sampler must be provided for negative sampling"
            # For negative sampling, we only need the weight matrix, no bias
            self.lm_head = nn.Parameter(torch.randn(config.vocab_size, config.n_embd))
            # Initialize with small random values
            nn.init.normal_(self.lm_head, mean=0.0, std=0.02)
            # Set up negative sampling loss
            self.neg_loss = NegSamplingLoss(self.lm_head, ns_k, sampler, shared_neg)
            # Weight tying with embedding
            self.transformer.wte.weight = self.lm_head
        elif use_asoft:
            assert cutoffs is not None, "cutoffs must be provided for adaptive softmax"
            self.asoft = AdaptiveLogSoftmaxWithLoss(
                in_features=config.n_embd,
                n_classes=config.vocab_size,
                cutoffs=cutoffs,
                div_value=div_value,
                head_bias=False, # Consistent with original lm_head
            )
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.wte.weight = (
                self.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        # pos = torch.arange(0, t, dtype=torch.long, device=idx.device)  # shape (t) # not used with RoPE

        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if self.use_neg_sampling:
            if targets is not None:
                # training/validation path
                h = x.view(-1, x.size(-1))  # (B*T, d)
                targets_flat = targets.view(-1)  # (B*T,)
                loss = self.neg_loss(h, targets_flat)
                # For negative sampling, we don't typically return logits during training
                # as they're not needed and would be expensive to compute
                logits = None
            else:
                # inference-time: fallback to full softmax on the very last position
                # This is rare and only used for generation
                logits = F.linear(x[:, [-1], :], self.lm_head)  # (B, 1, V)
                loss = None
        elif self.use_asoft:
            if targets is not None:
                # training/validation path - keep asoft in FP32 using simple cast
                x_fp32 = x.view(-1, x.size(-1)).float()
                targets_flat = targets.view(-1)
                
                # asoft operates in FP32
                output_asoft = self.asoft(x_fp32, targets_flat)
                loss = output_asoft.loss
                
                if return_logits:
                    logits = self.asoft.log_prob(x.float()).to(x.dtype)
                else:
                    logits = None
            else:
                # inference-time: only forward on the very last position
                logits = self.asoft.log_prob(x[:, [-1], :].float()).to(x.dtype)
                loss = None
        else:
            # original dense lm_head path
            if targets is not None:
                # if we are given some desired targets also calculate the loss
                logits = self.lm_head(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )
            else:
                # inference-time mini-optimization: only forward the lm_head on the very last position
                logits = self.lm_head(
                    x[:, [-1], :]
                )  # note: using list [-1] to preserve the time dim
                loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits and not self.use_asoft and not self.use_neg_sampling:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )
        return optimizer


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = np.int64(0)
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(
            f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files"
        )

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


# -----------------------------------------------------------------------------
# int main

VAL_TOKENS = 1_048_576  # how many tokens of validation data. It's important to keep this fixed for consistent comparisons


def compute_full_cross_entropy_loss(model, x_val, y_val):
    """
    Compute full cross-entropy loss for negative sampling models during validation.
    This is needed to get the proper benchmark metric.
    """
    # Forward through transformer layers
    b, t = x_val.size()
    x = model.module.transformer.wte(x_val)
    for block in model.module.transformer.h:
        x = block(x)
    x = rmsnorm(x)
    
    # Compute full softmax using the (tied) embedding weights. Using wte avoids
    # any potential drift if lm_head were to become de-tied in future edits.
    logits = F.linear(x, model.module.transformer.wte.weight)  # (B, T, V)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), y_val.view(-1), ignore_index=-1
    )
    return loss


def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


if __name__ == "__main__":
    import time
    import argparse

    print0(f"Running pytorch {torch.version.__version__}")

    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument(
        "--input_bin",
        type=str,
        default="data/fineweb10B/fineweb_train_*.bin",
        help="input .bin to train on",
    )
    parser.add_argument(
        "--input_val_bin",
        type=str,
        default="data/fineweb10B/fineweb_val_*.bin",
        help="input .bin to eval validation loss on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="output directory to which to write logs and checkpoints",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="d12",
        help="d12|d24|d36|d48",
    )
    # token layout for each step of the optimization
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size, in units of #batch dimensions",
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=1,
        help="number of gradient accumulation steps",
    )
    parser.add_argument(
        "--sequence_length", type=int, default=64, help="sequence length"
    )
    # workload (number of steps)
    parser.add_argument(
        "--num_iterations", type=int, default=10, help="number of iterations to run"
    )
    # optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="learning rate warmup iterations",
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=0, help="learning rate warmup iterations"
    )
    parser.add_argument(
        "--warmdown_iters",
        type=int,
        default=0,
        help="learning rate warmdown iterations",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    # evaluation
    parser.add_argument(
        "--val_loss_every",
        type=int,
        default=0,
        help="every how mant steps to evaluate val loss?",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=16,
        help="how many batches of val to average?",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5000,
        help="every how many steps to save the checkpoint",
    )
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="log to wandb",
    )
    parser.add_argument(
        "--precision", 
        type=str,
        choices=["fp32", "fp16", "bf16"], 
        default="bf16",
        help="Computation precision for forward/backward pass (fp32, fp16, bf16)"
    )
    # Adaptive Softmax arguments
    parser.add_argument(
        "--adaptive_softmax",
        action="store_true",
        help="Use Adaptive Softmax instead of a full softmax layer."
    )
    parser.add_argument(
        "--asoft_cutoffs",
        type=str,
        default="500,4000,20000", # Optimization 2a: Better cutoff pattern - smaller head cluster for faster GEMM
        help="Comma-separated string of cutoff values for adaptive softmax, e.g., '500,4000,20000'. The vocab size is implicitly the last cutoff."
    )
    parser.add_argument(
        "--asoft_div_value",
        type=float,
        default=4.0,
        help="Divisor value for adaptive softmax projection sizes."
    )
    parser.add_argument(
        "--target_val_loss",
        type=float,
        default=3.3821,
        help="Target validation loss to stop training."
    )
    # Negative Sampling arguments
    parser.add_argument(
        "--negative_sampling",
        action="store_true",
        help="Use Negative Sampling / NCE instead of a full softmax layer."
    )
    parser.add_argument(
        "--ns_k",
        type=int,
        default=20,
        help="Number of negative samples per positive sample for negative sampling."
    )
    parser.add_argument(
        "--ns_power",
        type=float,
        default=0.75,
        help="Exponent for the unigram distribution in negative sampling (0.75 is Mikolov's default)."
    )
    parser.add_argument(
        "--ns_shared_negatives",
        action="store_true",
        help="If set, sample k negatives per batch instead of per token."
    )
    parser.add_argument(
        "--ns_table_size",
        type=int,
        default=1_000_000,
        help="Size of pre-built alias table for negative sampling."
    )
    parser.add_argument(
        "--ns_k_schedule",
        type=str,
        default="20",
        help="Comma-separated k values for annealing (e.g., '24,12,8'). Single value means no annealing."
    )
    parser.add_argument(
        "--ns_power_schedule",
        type=str,
        default="0.75",
        help="Comma-separated power values for annealing (e.g., '0.75,0.6,0.5'). Single value means no annealing."
    )
    parser.add_argument(
        "--val_sequence_length",
        type=int,
        default=None,
        help="Sequence length used for validation. If None, defaults to --sequence_length." 
    )
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert args.model in {"d12", "d24", "d36", "d48"}
    
    # Mutual exclusion between adaptive softmax and negative sampling
    if args.adaptive_softmax and args.negative_sampling:
        print0("ERROR: Cannot use both --adaptive_softmax and --negative_sampling simultaneously.")
        print0("Please choose one output layer type.")
        exit(1)
    
    # set up DDP (distributed data parallel). torchrun sets this env variable
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    assert (
        args.grad_accumulation_steps % ddp_world_size == 0
    ), "grad_accumulation_steps must be divisible by world size"
    args.grad_accumulation_steps //= (
        ddp_world_size  # each gpu does its fraction of the work
    )
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = 0  # each process gets the exact same seed
    print(f"using device: {device}")

    if args.log_wandb and master_process:
        import wandb
        import datetime

        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wandb.init(project="benchmark_gpt2", name=f"gpt2-{args.model} {start_time}")
        wandb.config.update(args)
        wandb.save("train_gpt2_rtx4090_optim2.py")
        wandb.save("run.sh")

    tokens_per_iter = B * T * ddp_world_size * args.grad_accumulation_steps
    print0(f"tokens per iteration: {tokens_per_iter:,}")

    # Decide mixed-precision mode
    if args.precision == "fp16":
        autocast_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
        print0("Using mixed precision: FP16 with gradient scaling")
    elif args.precision == "bf16":
        autocast_dtype = torch.bfloat16
        scaler = None
        print0("Using mixed precision: BF16 (no gradient scaling needed)")
    else:  # fp32
        autocast_dtype = torch.float32
        scaler = None
        print0("Using full precision: FP32")

    # set up a context manager following the desired dtype and device
    ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype!=torch.float32)

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    
    # Validation sequence length can differ
    T_val = args.val_sequence_length if args.val_sequence_length is not None else T
    
    tokens_per_iter_val = args.val_batch_size * T_val * ddp_world_size
    assert VAL_TOKENS % tokens_per_iter_val == 0, (
        f"VAL_TOKENS ({VAL_TOKENS}) must be divisible by val_batch_size*val_sequence_length*world_size "
        f"({tokens_per_iter_val}). Choose different --val_sequence_length or --val_batch_size.")
    val_steps = VAL_TOKENS // tokens_per_iter_val

    val_loader = DistributedDataLoader(
        args.input_val_bin, args.val_batch_size, T_val, ddp_rank, ddp_world_size
    )
    x, y = train_loader.next_batch()

    # init the model from scratch
    num_vocab = train_loader.tokens.max().item() + 1 # Determine vocab from data, can be slightly different than 50257
    # If using adaptive softmax, vocab_size for GPTConfig should be actual num_vocab
    # For AdaptiveLogSoftmaxWithLoss, n_classes is this num_vocab.
    # GPTConfig's vocab_size is mostly for embedding table size if not tied or for reference.
    # Let's ensure consistency.
    effective_vocab_size = num_vocab
    print0(f"Effective vocabulary size: {effective_vocab_size}")

    # Set up negative sampling if enabled
    sampler = None
    k_schedule = []
    power_schedule = []
    k_transition_steps = []
    freqs = None
    # Provide defaults for initial_k and initial_power even when negative sampling is disabled
    initial_k = args.ns_k
    initial_power = args.ns_power
    if args.negative_sampling:
        print0(f"Setting up negative sampling with k={args.ns_k}, power={args.ns_power}")
        
        # Parse k and power schedules for annealing
        k_schedule = [int(k.strip()) for k in args.ns_k_schedule.split(',')]
        power_schedule = [float(p.strip()) for p in args.ns_power_schedule.split(',')]
        
        # Validate schedules
        if len(k_schedule) != len(power_schedule):
            print0("ERROR: ns_k_schedule and ns_power_schedule must have the same number of values")
            exit(1)
        
        # Validate edge case: short training with 3-value schedule
        if len(k_schedule) == 3 and args.num_iterations < 4:
            print0(f"WARNING: Using 3-value schedule with only {args.num_iterations} iterations may cause transition steps to overlap.")
            print0("Consider using a 2-value schedule or increasing num_iterations.")
            
        # Set initial values and create annealing schedule
        initial_k = k_schedule[0]
        initial_power = power_schedule[0]
        
        if len(k_schedule) > 1:
            # Calculate transition points (evenly spaced)
            total_steps = args.num_iterations
            if len(k_schedule) == 3:
                # For 3 values: first 30% use k[0], next 40% use k[1], final 30% use k[2]
                k_transition_steps = [max(1, int(0.30 * total_steps)), max(2, int(0.70 * total_steps))]
            elif len(k_schedule) == 2:
                # For 2 values: first 50% use k[0], second 50% use k[1]
                k_transition_steps = [max(1, int(0.5 * total_steps))]
            else:
                # For other cases, evenly distribute
                step_size = total_steps // len(k_schedule)
                k_transition_steps = [max(i + 1, (i + 1) * step_size) for i in range(len(k_schedule) - 1)]
            
            print0(f"K annealing schedule: {k_schedule} at steps {[0] + k_transition_steps}")
            print0(f"Power annealing schedule: {power_schedule} at steps {[0] + k_transition_steps}")
        else:
            k_transition_steps = []
            print0(f"No annealing - using fixed k={initial_k}, power={initial_power}")
        
        print0(f"Setting up negative sampling with initial k={initial_k}, power={initial_power}")
        
        # Check if frequency file exists to avoid recomputing
        freq_file = f"data/fineweb10B/token_freqs_vocab{effective_vocab_size}.npy"
        if os.path.exists(freq_file) and master_process:
            print0(f"Loading cached token frequencies from {freq_file}")
            freqs = np.load(freq_file)
        else:
            if master_process:
                # Only master process counts frequencies to avoid conflicts
                freqs = count_token_frequencies(args.input_bin, effective_vocab_size)
                # Save frequencies for future runs
                os.makedirs(os.path.dirname(freq_file), exist_ok=True)
                np.save(freq_file, freqs)
                print0(f"Saved token frequencies to {freq_file}")
            else:
                # Wait for master process to finish and load the file
                while not os.path.exists(freq_file):
                    time.sleep(1)
                freqs = np.load(freq_file)
        
        # Create unigram sampler on all processes
        sampler = UnigramSampler(freqs, power=initial_power, 
                               table_size=args.ns_table_size, device=device)

    model_config = {
        "d12": GPTConfig(
            vocab_size=effective_vocab_size, n_layer=12, n_head=12, n_embd=768
        ),  # 124M GPT-2
        "d24": GPTConfig(vocab_size=effective_vocab_size, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(vocab_size=effective_vocab_size, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(vocab_size=effective_vocab_size, n_layer=48, n_head=25, n_embd=1600),
    }[args.model]

    # decide parameter / activation dtype
    dtype = (
        torch.bfloat16 if args.precision == "bf16"
        else torch.float32 if args.precision == "fp32"
        else torch.float16
    )
    # enable Flash-Attention v2 & memory-efficient SDPA kernels ----
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Set up output layer configuration
    use_asoft = args.adaptive_softmax
    use_neg_sampling = args.negative_sampling
    asoft_cutoffs_parsed = []
    
    if use_asoft:
        if args.asoft_cutoffs.strip():
            asoft_cutoffs_parsed = [int(c.strip()) for c in args.asoft_cutoffs.split(',')]
        # AdaptiveLogSoftmaxWithLoss expects cutoffs to not include vocab_size itself
        # and must be strictly increasing.
        # Also, ensure cutoffs are less than vocab_size.
        asoft_cutoffs_parsed = sorted(list(set(c for c in asoft_cutoffs_parsed if c < effective_vocab_size)))
        if not asoft_cutoffs_parsed: # If empty or all cutoffs were >= vocab_size
             print0("Warning: Adaptive softmax enabled but no valid cutoffs provided or all cutoffs >= vocab_size. Consider providing valid cutoffs like '2000,10000'. Disabling adaptive softmax.")
             use_asoft = False # Fallback to default softmax
        else:
            print0(f"Using Adaptive Softmax with cutoffs: {asoft_cutoffs_parsed} and div_value: {args.asoft_div_value}")

    if use_neg_sampling:
        print0(f"Using Negative Sampling with initial k={initial_k}, shared_negatives={args.ns_shared_negatives}")

    model = GPT(model_config, 
                use_asoft=use_asoft, cutoffs=asoft_cutoffs_parsed, div_value=args.asoft_div_value,
                use_neg_sampling=use_neg_sampling, ns_k=initial_k, 
                sampler=sampler, shared_neg=args.ns_shared_negatives)
    # Move to device first, then cast dtypes
    model.to(device)
    
    # Cast model parameters to the specified precision (dtype)
    if dtype != torch.float32: # For bf16 or fp16
        # First, cast the entire model to the target precision (e.g., bfloat16)
        model.to(dtype=dtype)
        # Then, if adaptive softmax is used, cast the asoft module back to float32
        # This ensures all parameters and buffers within asoft are float32
        if use_asoft and hasattr(model, 'asoft'):
            model.asoft.to(torch.float32)
            print0(f"Casted model.asoft and its submodules to torch.float32. Rest of model is {args.precision}.")
    else: # For fp32
        model.to(torch.float32) # Ensure it is float32 if that's the target

    model = model.train() #.cuda() is redundant due to .to(device)
    
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    print0("compiling the model...")
    
    # Torch.compile still causes large fused kernels for NegSampling; skip when enabled
    if not args.negative_sampling:
        model = torch.compile(
            model,
            options={"triton.cudagraphs": False},
        )
    else:
        print0("Skipping torch.compile for negative sampling to reduce memory usage.")

    # If adaptive softmax is active, wrap it so torch.compile skips it
    if use_asoft and hasattr(model, "asoft"):
        class NonCompiledWrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                # Signal to the compiler to leave this module untouched
                self._is_compiling = True

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        # Replace asoft with non-compiled wrapper to avoid expensive kernel autotune
        model.asoft = NonCompiledWrapper(model.asoft)

    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module  # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=device,
    )

    # learning rate decay scheduler (linear warmup and warmdown)
    def get_lr(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        # 2) constant lr for a while
        elif it < args.num_iterations - args.warmdown_iters:
            return args.learning_rate
        # 3) linear warmdown
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return args.learning_rate * decay_ratio

    run_id = str(uuid.uuid4())

    # create the logging directory if it does not exist
    logfile = None
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "%s.log" % run_id)
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

    training_time_ms = 0.0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    def _maybe_anneal_negative_sampling(step, args, k_schedule, power_schedule, k_transition_steps, 
                                       raw_model, sampler, freqs, device):
        """Apply k and power annealing for negative sampling if conditions are met."""
        if not args.negative_sampling or len(k_schedule) <= 1:
            return sampler
            
        # Determine current schedule index
        current_schedule_idx = 0
        for i, transition_step in enumerate(k_transition_steps):
            if step >= transition_step:
                current_schedule_idx = i + 1
        
        # Check if we need to update k or power
        target_k = k_schedule[current_schedule_idx]
        target_power = power_schedule[current_schedule_idx]
        
        # Update k if it changed
        if hasattr(raw_model, 'neg_loss') and raw_model.neg_loss.k != target_k:
            old_k = raw_model.neg_loss.k
            raw_model.neg_loss.k = target_k
            print0(f"Step {step}: Annealing k from {old_k} to {target_k}")
        
        # Update power if it changed (rebuild sampler)
        if sampler is not None and abs(sampler.power - target_power) > 1e-6:
            old_power = sampler.power
            # Rebuild sampler with new power
            new_sampler = UnigramSampler(freqs, power=target_power, 
                                       table_size=args.ns_table_size, device=device)
            raw_model.neg_loss.sampler = new_sampler
            print0(f"Step {step}: Annealing power from {old_power:.3f} to {target_power:.3f}")
            return new_sampler
            
        return sampler

    # begin training
    for step in range(args.num_iterations + 1):
        last_step = step == args.num_iterations

        # once in a while evaluate the validation dataset
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_loader.reset()  # reset the val loader so that it starts from the beginning
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(val_steps):  # always fiexed number of validation steps
                    x_val, y_val = val_loader.next_batch()
                    if args.negative_sampling:
                        # For negative sampling, compute full cross-entropy for benchmark comparison
                        loss = compute_full_cross_entropy_loss(model, x_val, y_val)
                    else:
                        # Standard validation path for dense and adaptive softmax
                        _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss /= val_steps
            # log to console and to file
            print0(f"step:{step}/{args.num_iterations} | val loss {val_loss:.6f}")
            if master_process:
                if args.log_wandb:
                    wandb.log({"val_loss": val_loss}, step=step * tokens_per_iter)
                    wandb.log({"time": training_time_ms}, step=step * tokens_per_iter)
                if logfile is not None:
                    with open(logfile, "a") as f:
                        f.write("s:%d val:%f\n" % (step, val_loss))
            
            # Check for early stopping based on target validation loss
            if val_loss <= args.target_val_loss:
                print0(f"Target validation loss {args.target_val_loss} reached at step {step}. Stopping training after {training_time_ms/1000:.2f} seconds ({training_time_ms/3600000:.2f} hours).")
                last_step = True # This will break the main loop after this block

            # restart the clock
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        train_loss = torch.zeros(1, device=device)
        for micro_step in range(args.grad_accumulation_steps):
            model.require_backward_grad_sync = (
                micro_step == args.grad_accumulation_steps - 1
            )  # sync only on last micro step to avoid overhead
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                loss = (
                    loss / args.grad_accumulation_steps
                )  # scale loss for gradient accumulation
                train_loss += loss.detach()
            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
            # backward pass
            loss.backward()

        train_loss /= (
            args.grad_accumulation_steps
        )  # average the loss over all micro steps

        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # step the optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Apply annealing for negative sampling if enabled
        if args.negative_sampling and len(k_schedule) > 1:
            sampler = _maybe_anneal_negative_sampling(step, args, k_schedule, power_schedule, k_transition_steps, 
                                                      raw_model, sampler, freqs, device)
        
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        torch.cuda.synchronize()
        # Calculate detailed timing and throughput metrics
        step_time_ms = 1000 * (time.perf_counter() - t0)
        approx_training_time_ms = training_time_ms + step_time_ms
        
        # Calculate tokens per second for this step (skip step 0 as it's often an outlier)
        if step > 0:
            tokens_this_step = tokens_per_iter
            tokens_per_second = tokens_this_step / (step_time_ms / 1000.0)
            # Calculate average tokens per second across all completed steps
            avg_tokens_per_second = ((step + 1) * tokens_per_iter) / (approx_training_time_ms / 1000.0)
        else:
            tokens_per_second = 0
            avg_tokens_per_second = 0

        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()  # keep track of the mean loss
        
        # Enhanced logging with more useful metrics
        if step > 0:  # Skip step 0 for cleaner logs
            current_k = getattr(raw_model.neg_loss, 'k', 'N/A') if args.negative_sampling and hasattr(raw_model, 'neg_loss') else 'N/A'
            current_power = getattr(sampler, 'power', 'N/A') if args.negative_sampling and sampler is not None else 'N/A'
            
            log_msg = (f"step:{step}/{args.num_iterations} | "
                      f"loss {lossf:.6f} | "
                      f"lr {lr:.2e} | "
                      f"step_time:{step_time_ms:.1f}ms | "
                      f"step_avg:{approx_training_time_ms/(step+1):.2f}ms | "
                      f"tok/s:{tokens_per_second/1e6:.2f}M | "
                      f"avg_tok/s:{avg_tokens_per_second/1e6:.2f}M")
            
            if args.negative_sampling:
                log_msg += f" | k:{current_k} | pow:{current_power:.2f}" if current_power != 'N/A' else f" | k:{current_k}"
                
            print0(log_msg)
        else:
            print0(f"step:{step}/{args.num_iterations} | loss {lossf:.6f} | warmup step")
        
        # log to logile
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trn:%f\n" % (step, lossf))

        if master_process and (step + 1) % args.save_every == 0:
            log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
            os.makedirs("logs/%s" % run_id, exist_ok=True)
            torch.save(log, "logs/%s/model_step%06d.pt" % (run_id, step))

    print0(
        f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )

    # -------------------------------------------------------------------------

    if master_process:
        log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
        os.makedirs("logs/%s" % run_id, exist_ok=True)
        torch.save(log, "logs/%s/final.pt" % run_id)

    # -------------------------------------------------------------------------
    # clean up nice
    destroy_process_group()