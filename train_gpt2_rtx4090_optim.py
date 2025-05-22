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
        self.dim = dim

    def forward(self, x):
        seq_len = x.shape[1]
        head_dim = x.shape[-1]
        # Ensure the rotary dimension matches the head_dim (half of it)
        effective_dim = min(self.dim, head_dim // 2)
        
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            # Use only as many frequency components as needed for the effective dimension
            inv_freq_used = self.inv_freq[:effective_dim]
            freqs = torch.outer(t, inv_freq_used).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    # Ensure cos and sin last dimension matches x
    if cos.shape[-1] != d:
        # Trim or pad cos/sin tensors to match x's dimensions
        if cos.shape[-1] > d:
            cos = cos[..., :d]
            sin = sin[..., :d]
        else:
            # This shouldn't normally happen, but handle it anyway
            pad_size = d - cos.shape[-1]
            cos = torch.nn.functional.pad(cos, (0, pad_size))
            sin = torch.nn.functional.pad(sin, (0, pad_size))
    
    x1 = x[..., :d]
    x2 = x[..., d:2*d]  # Only use the second half up to 2*d to match first half
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
        self.n_head_q = config.n_head
        self.n_head_kv = config.n_kv_head if config.n_kv_head is not None else config.n_head
        self.n_embd = config.n_embd
        
        # Verify dimensions are compatible
        assert self.n_embd % self.n_head_q == 0, f"n_embd ({self.n_embd}) must be divisible by n_head_q ({self.n_head_q})"
        assert self.n_head_q % self.n_head_kv == 0, f"n_head_q ({self.n_head_q}) must be divisible by n_head_kv ({self.n_head_kv})"
        
        self.head_dim = self.n_embd // self.n_head_q
        
        # Separate projections for query and key/value
        self.q_proj = nn.Linear(self.n_embd, self.n_head_q * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(self.n_embd, 2 * self.n_head_kv * self.head_dim, bias=False)
        
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # Make sure rotary dimension is capped to head_dim
        self.rotary = Rotary(min(self.head_dim, 64))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(B, T, self.n_head_q, self.head_dim)
        kv = self.kv_proj(x).view(B, T, self.n_head_kv, 2, self.head_dim)
        k, v = kv.unbind(dim=3)  # shapes: (B, T, n_head_kv, head_dim)
        
        # Apply rotary embeddings
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # Rearrange and process attention.
        if self.n_head_q != self.n_head_kv:
            repeats = self.n_head_q // self.n_head_kv  # e.g. 4 for 12/3
            k = k.repeat_interleave(repeats, dim=2)
            v = v.repeat_interleave(repeats, dim=2)

        # Standard attention path with heads now matching
        q_t = q.transpose(1,2)
        k_t = k.transpose(1,2)
        v_t = v.transpose(1,2)
        y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        
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
# The main GPT-2 model


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    # MTP parameters
    mtp_enabled: bool = False
    mtp_max_steps: int = 1 # If 1, only standard next-token. If >1, predicts 1..N and aux loss for 2..N
    mtp_weight: float = 0.1
    mtp_rampup_steps: int = 256 # Number of steps to linearly ramp up MTP weight from 0 to mtp_weight
    # GQA parameters
    gqa_enabled: bool = False
    n_kv_head: int = None  # None defaults to n_head (i.e., MHSA); otherwise n_head > n_kv_head for GQA
    # Precision parameter
    precision: str = "bf16"


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying
        # For torch.compile compatibility with dynamic MTP weight
        self.register_buffer("mtp_weight_buffer", torch.tensor(config.mtp_weight, dtype=torch.float32))

    def forward(self, idx, targets_dict=None, return_logits=True, return_loss_components=False):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)  # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        loss = None
        if targets_dict is not None:
            logits = self.lm_head(x) # (b, t, vocab_size)
            # Standard next-token prediction loss (target_1)
            main_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets_dict['target_1'].view(-1), ignore_index=-1
            )
            total_loss = main_loss

            # Multi-Token Prediction (MTP) auxiliary loss
            aux_loss = None
            # Use the mtp_weight_buffer for compile-time stability
            effective_weight = self.mtp_weight_buffer
            if self.config.mtp_enabled and self.config.mtp_max_steps > 1:
                aux_loss_sum = torch.tensor(0.0, device=idx.device, dtype=x.dtype) # Match dtype
                num_aux_losses = 0
                # Auxiliary losses for predictions from 2 steps ahead up to mtp_max_steps
                for k in range(2, self.config.mtp_max_steps + 1):
                    target_key = f'target_{k}'
                    if target_key in targets_dict:
                        aux_target_k = targets_dict[target_key]
                        # Reuse the same logits from the single forward pass
                        aux_loss_k = F.cross_entropy(
                            logits.view(-1, logits.size(-1)), aux_target_k.view(-1), ignore_index=-1
                        )
                        aux_loss_sum += aux_loss_k
                        num_aux_losses += 1
                
                if num_aux_losses > 0:
                    aux_loss = aux_loss_sum / num_aux_losses
                    total_loss = main_loss + effective_weight * aux_loss
            loss = total_loss

            if return_loss_components:
                return logits, (main_loss, aux_loss) if aux_loss is not None else (main_loss, None)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
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
    def __init__(self, filename_pattern, B, T, process_rank, num_processes, mtp_enabled=False, mtp_max_steps=1):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.mtp_enabled = mtp_enabled
        # Effective max prediction steps: 1 for standard, mtp_max_steps if MTP is on.
        self.actual_max_pred_steps = mtp_max_steps if mtp_enabled else 1

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = np.int64(0)
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            # Ensure shard has enough tokens for all processes, including the max lookahead for targets
            assert shard_ntok >= num_processes * B * T + self.actual_max_pred_steps
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
        # Determine the total number of tokens needed from the buffer for inputs and all targets
        buf_len = B * T + self.actual_max_pred_steps
        buf = self.tokens[self.current_position : self.current_position + buf_len]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)

        x = (buf[:B*T]).view(B, T)  # inputs are the first B*T tokens
        targets_dict = {}
        for k in range(1, self.actual_max_pred_steps + 1):
            # target_k is offset by k tokens from the start of the input sequence slice
            targets_dict[f'target_{k}'] = (buf[k : B*T + k]).view(B, T)

        # Advance current_position to the start of the *next* batch for this process
        self.current_position += B * T * self.num_processes

        # Check if the current shard has enough data for the *next* batch for this process.
        # The next batch would start at self.current_position (offset in current shard)
        # and would require (B*T + self.actual_max_pred_steps) tokens from the shard.
        if self.current_position + B*T + self.actual_max_pred_steps > len(self.tokens):
            self.advance() # Loads new shard and resets self.current_position for that shard
        
        # Move all tensors in targets_dict to CUDA
        targets_dict_cuda = {key: val.cuda() for key, val in targets_dict.items()}
        return x.cuda(), targets_dict_cuda


# -----------------------------------------------------------------------------
# int main

VAL_TOKENS = 1_048_576  # how many tokens of validation data. It's important to keep this fixed for consistent comparisons


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
    # MTP arguments
    parser.add_argument(
        "--mtp_enabled",
        action="store_true",
        help="Enable Multi-Token Prediction auxiliary loss"
    )
    parser.add_argument(
        "--mtp_max_steps",
        type=int,
        default=2, # Default to predicting t+1 and t+2 if mtp_enabled
        help="Max steps ahead for MTP (e.g., 2 means predict t+1 and t+2, aux loss for t+2)"
    )
    parser.add_argument(
        "--mtp_weight",
        type=float,
        default=0.1,
        help="Weight for the MTP auxiliary loss"
    )
    parser.add_argument(
        "--mtp_rampup_steps",
        type=int,
        default=256,
        help="Number of steps to linearly ramp up MTP weight from 0"
    )
    # GQA arguments
    parser.add_argument(
        "--gqa_enabled",
        action="store_true",
        help="Enable Grouped-Query Attention (GQA)"
    )
    parser.add_argument(
        "--n_kv_head",
        type=int,
        default=None,
        help="Number of key/value heads for GQA (must be divisor of n_head)"
    )
    parser.add_argument(
        "--kv_head_ratio",
        type=float,
        default=0.25,
        help="Ratio of KV heads to query heads (e.g., 0.25 = 4x sharing, only used when n_kv_head is None)"
    )
    parser.add_argument(
        "--embd_scale",
        type=float,
        default=1.12,
        help="Scale factor for embedding dimension when using GQA (reinvests saved memory)"
    )
    # Add mixed precision argument
    parser.add_argument(
        "--precision", 
        type=str,
        choices=["fp32", "fp16", "bf16"], 
        default="bf16",
        help="Computation precision for forward/backward pass (fp32, fp16, bf16)"
    )
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert args.model in {"d12", "d24", "d36", "d48"}
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
        wandb.save("train_gpt2_rtx4090_optim.py")
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
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size, args.mtp_enabled, args.mtp_max_steps)
    val_loader = None
    tokens_per_iter_val = args.val_batch_size * T * ddp_world_size
    # Effective max prediction steps for val_loader
    val_actual_max_pred_steps = args.mtp_max_steps if args.mtp_enabled else 1
    assert VAL_TOKENS % tokens_per_iter_val == 0, f"VAL_TOKENS ({VAL_TOKENS}) must be divisible by tokens_per_iter_val ({tokens_per_iter_val})"

    val_steps = VAL_TOKENS // tokens_per_iter_val

    val_loader = DistributedDataLoader(
        args.input_val_bin, args.val_batch_size, T, ddp_rank, ddp_world_size, args.mtp_enabled, args.mtp_max_steps
    )
    x, targets_dict = train_loader.next_batch() # Initial batch load

    # init the model from scratch
    num_vocab = 50257
    model_config_params = {
        "d12": dict(n_layer=12, n_head=12, n_embd=768),  # 124M GPT-2
        "d24": dict(n_layer=24, n_head=16, n_embd=1024),
        "d36": dict(n_layer=36, n_head=20, n_embd=1280),
        "d48": dict(n_layer=48, n_head=25, n_embd=1600),
    }[args.model]
    
    print0(f"Initial model configuration: {model_config_params}")
    
    # Calculate n_kv_head based on ratio if not explicitly provided
    n_kv_head = args.n_kv_head
    if args.gqa_enabled and n_kv_head is None:
        n_head = model_config_params["n_head"]
        # Calculate target KV heads based on ratio, ensure at least 1
        target_kv_heads = max(1, int(n_head * args.kv_head_ratio))
        
        # Find largest divisor of n_head that is <= target_kv_heads
        n_kv_head = target_kv_heads
        while n_head % n_kv_head != 0:
            n_kv_head -= 1
            # Safeguard against going below 1
            if n_kv_head < 1:
                n_kv_head = 1
                break
        
        print0(f"Using n_kv_head={n_kv_head} (ratio={n_kv_head/n_head:.2f}) with n_head={n_head}")
    
    # Scale model size if GQA is enabled to reinvest memory savings
    if args.gqa_enabled:
        original_embd = model_config_params["n_embd"]
        n_head = model_config_params["n_head"]
        # Scale embedding dimension to reinvest memory savings
        scaled_embd = int(original_embd * args.embd_scale)
        # Ensure head_dim stays <=64 and div by 8
        candidate_embd = (scaled_embd // n_head) * n_head
        head_dim_candidate = candidate_embd // n_head
        if head_dim_candidate > 64:
            candidate_embd = 64 * n_head  # cap head_dim at 64
            head_dim_candidate = 64
        # round head_dim_candidate to multiple of 8 if needed
        if head_dim_candidate % 8 != 0:
            head_dim_candidate = (head_dim_candidate // 8) * 8
            candidate_embd = head_dim_candidate * n_head
        model_config_params["n_embd"] = int(candidate_embd)
        print0(f"Scaling embedding dimension from {original_embd} to {model_config_params['n_embd']} with GQA enabled (head_dim={head_dim_candidate})")
    
    # Verify final configuration
    n_head = model_config_params["n_head"]
    n_embd = model_config_params["n_embd"]
    
    if args.gqa_enabled:
        # Verify the dimensions are compatible
        if n_kv_head is not None and n_head % n_kv_head != 0:
            print0(f"Warning: n_head ({n_head}) is not divisible by n_kv_head ({n_kv_head})")
            # Find the largest divisor of n_head that is <= n_kv_head
            new_n_kv_head = n_kv_head
            while n_head % new_n_kv_head != 0 and new_n_kv_head > 1:
                new_n_kv_head -= 1
            print0(f"Adjusting n_kv_head from {n_kv_head} to {new_n_kv_head}")
            n_kv_head = new_n_kv_head
    
    # Verify that embedding dimension is divisible by n_head
    if n_embd % n_head != 0:
        print0(f"Warning: n_embd ({n_embd}) is not divisible by n_head ({n_head})")
        # Adjust embedding dimension to nearest multiple of n_head
        n_embd = (n_embd // n_head) * n_head
        model_config_params["n_embd"] = n_embd
        print0(f"Adjusted n_embd to {n_embd}")
    
    print0(f"Final model configuration: {model_config_params}, n_kv_head={n_kv_head if args.gqa_enabled else n_head}")
    
    model_config = GPTConfig(
        vocab_size=num_vocab, 
        **model_config_params,
        mtp_enabled=args.mtp_enabled,
        mtp_max_steps=args.mtp_max_steps if args.mtp_enabled else 1,
        mtp_weight=args.mtp_weight,
        mtp_rampup_steps=args.mtp_rampup_steps,
        gqa_enabled=args.gqa_enabled,
        n_kv_head=n_kv_head if args.gqa_enabled else None,
        precision=args.precision
    )
    # decide parameter / activation dtype
    dtype = (
        torch.bfloat16 if args.precision == "bf16"
        else torch.float32 if args.precision == "fp32"
        else torch.float16
    )
    # enable Flash-Attention v2 & memory-efficient SDPA kernels ----
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    model = GPT(model_config).to(device, dtype=dtype)
    model = model.train().cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    print0("compiling the model...")
    model = torch.compile(
        model
    )  # NOTE: this might cause issues depending on your GPU, consider turning it off

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

    # begin training
    for step in range(args.num_iterations + 1):
        last_step = step == args.num_iterations

        # Calculate current MTP weight based on ramp-up schedule
        if args.mtp_enabled and step < args.mtp_rampup_steps:
            current_mtp_weight = (step / args.mtp_rampup_steps) * args.mtp_weight
            raw_model.mtp_weight_buffer.fill_(current_mtp_weight) # Update buffer
        elif args.mtp_enabled:
            current_mtp_weight = args.mtp_weight
            raw_model.mtp_weight_buffer.fill_(current_mtp_weight) # Ensure buffer has final weight

        # once in a while evaluate the validation dataset
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_loader.reset()  # reset the val loader so that it starts from the beginning
            with torch.no_grad():
                val_loss = torch.zeros(1, device=device)
                val_ntp_loss = torch.zeros(1, device=device)
                val_mtp_loss = torch.zeros(1, device=device)
                for _ in range(val_steps):  # always fixed number of validation steps
                    x_val, targets_dict_val = val_loader.next_batch()
                    # Get both main and auxiliary losses
                    logits, (main_loss, aux_loss) = model(x_val, targets_dict_val, return_logits=False, return_loss_components=True)
                    val_ntp_loss += main_loss.detach()
                    if aux_loss is not None:
                        val_mtp_loss += (main_loss + aux_loss * raw_model.mtp_weight_buffer).detach()
                    # For val_loss (the main metric), use only next-token prediction loss
                    val_loss += main_loss.detach()
                
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(val_ntp_loss, op=dist.ReduceOp.AVG)
                if args.mtp_enabled:
                    dist.all_reduce(val_mtp_loss, op=dist.ReduceOp.AVG)
                
                val_loss = val_loss.item() / val_steps
                val_ntp_loss = val_ntp_loss.item() / val_steps
                if args.mtp_enabled:
                    val_mtp_loss = val_mtp_loss.item() / val_steps

            # log to console and to file
            if args.mtp_enabled:
                print0(f"step:{step}/{args.num_iterations} | val loss {val_loss:.6f} | mtp loss {val_mtp_loss:.6f}")
            else:
                print0(f"step:{step}/{args.num_iterations} | val loss {val_loss:.6f}")
            if master_process:
                if args.log_wandb:
                    wandb.log({"val_loss": val_loss}, step=step * tokens_per_iter)  # Main metric - next-token only
                    if args.mtp_enabled:
                        wandb.log({"val_ntp_loss": val_ntp_loss}, step=step * tokens_per_iter)  # Same as val_loss, for clarity
                        wandb.log({"val_mtp_loss": val_mtp_loss}, step=step * tokens_per_iter)  # Combined loss if MTP enabled
                    wandb.log({"time": training_time_ms}, step=step * tokens_per_iter)
                if logfile is not None:
                    with open(logfile, "a") as f:
                        if args.mtp_enabled:
                            f.write("s:%d val:%f ntp:%f mtp:%f\n" % (
                                step, val_loss, val_ntp_loss, 
                                val_mtp_loss
                            ))
                        else:
                            f.write("s:%d val:%f\n" % (
                                step, val_loss
                            ))

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
                logits, (main_loss, aux_loss) = model(x, targets_dict, return_logits=False, return_loss_components=True)
                loss = main_loss
                if aux_loss is not None:
                    # Training loss uses the current_mtp_weight by referencing the buffer
                    loss = main_loss + aux_loss * raw_model.mtp_weight_buffer
                loss = (
                    loss / args.grad_accumulation_steps
                )  # scale loss for gradient accumulation
                train_loss += loss.detach()
            # advance the dataset for the next batch
            x, targets_dict = train_loader.next_batch()
            # backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        train_loss /= (
            args.grad_accumulation_steps
        )  # average the loss over all micro steps

        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # step the optimizer with scaler if needed
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            
        optimizer.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        torch.cuda.synchronize()
        # time and print
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        # the 0th iteration is often an outlier (much slower) => skip logging it
        # tokens_per_second = ddp_world_size * B * T / (t1-t0)
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        lossf = train_loss.item()  # keep track of the mean loss
        print0(
            f"step:{step}/{args.num_iterations} | loss {lossf:.6f} | train_time:{approx_training_time_ms/1000:.2f}s | step_avg:{approx_training_time_ms/(step+1):.2f}ms"
        )
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

    peak_mem = torch.cuda.max_memory_allocated() // (1024 * 1024)
    print0(f"Peak memory consumption: {peak_mem} MiB with precision={args.precision}")
    
    if args.gqa_enabled:
        mem_details = {
            "peak_memory_mb": peak_mem,
            "n_head": model_config.n_head,
            "n_kv_head": model_config.n_kv_head,
            "n_embd": model_config.n_embd,
            "n_layer": model_config.n_layer,
            "kv_sharing_ratio": model_config.n_head / model_config.n_kv_head,
            "embd_scale_factor": args.embd_scale
        }
        print0(f"GQA Configuration: {mem_details}")

    if master_process:
        log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
        os.makedirs("logs/%s" % run_id, exist_ok=True)
        torch.save(log, "logs/%s/final.pt" % run_id)

    # -------------------------------------------------------------------------
    # clean up nice
    destroy_process_group()