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
# For BF16 performance
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
# Set TF32 precision for matmul for better performance on Ampere GPUs
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # choose fastest algorithm
# enable efficient FFT plan cache for AFNO layers
if hasattr(torch.backends.cuda, "enable_mem_efficient_fft"):
    torch.backends.cuda.enable_mem_efficient_fft(True)
if hasattr(torch.backends.cuda, "enable_fft_planning_cache"):
    torch.backends.cuda.enable_fft_planning_cache(True)
import torch.distributed as dist
import torch.nn.functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from afno import AFNO1D

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

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        # Alternate mixer based on layer_idx
        if self.layer_idx % 2 == 0: # Even layers use AFNO
            if config.mixer == "afno": # Check if AFNO is the intended primary mixer type
                self.mixer = AFNO1D(
                    hidden_size=config.n_embd, 
                    num_blocks=config.afno_num_blocks, 
                    sparsity_threshold=config.afno_sparsity_threshold,
                    hard_thresholding_fraction=config.afno_hard_thresholding_fraction,
                    hidden_size_factor=config.afno_hidden_size_factor
                )
            elif config.mixer == "attn": # Fallback or if explicitly set to attn for even layer (unusual for hybrid)
                print(f"Warning: Layer {self.layer_idx} (even) is using CausalSelfAttention, but config.mixer is '{config.mixer}'. Ensure this is intended for hybrid setup.")
                self.mixer = CausalSelfAttention(config)
            else:
                raise ValueError(f"Unknown mixer type for even layer: {config.mixer}")
        else: # Odd layers use CausalSelfAttention
            self.mixer = CausalSelfAttention(config)
        
        self.mlp = MLP(config)
        self.mixer_scale = 1 / math.sqrt(2 * config.n_layer) # Renamed from attn_scale

    def forward(self, x):
        x = x + self.mixer_scale * self.mixer(rmsnorm(x)) # Use self.mixer instead of old self.attn
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
    bias: bool = False
    mixer: str = "attn"
    afno_num_blocks: int = 8
    afno_sparsity_threshold: float = 0.01
    afno_hard_thresholding_fraction: float = 1.0
    afno_hidden_size_factor: int = 1
    sequence_length: int = 1024


class GPT(nn.Module):

    def __init__(self, config, use_asoft=False, cutoffs=None, div_value=4.0):
        super().__init__()
        self.config = config
        self.use_asoft = use_asoft

        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        )
        if self.config.mixer == "afno":
            modules['wpe'] = nn.Embedding(config.sequence_length, config.n_embd)
        
        self.transformer = nn.ModuleDict(modules)

        if use_asoft:
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
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        # Add positional embeddings if 'wpe' exists (i.e., if AFNO is part of the model)
        # CausalSelfAttention layers will use their internal RoPE regardless.
        # Channel-wise AFNO layers don't inherently use sequence position but might benefit from global pos_emb.
        if 'wpe' in self.transformer:
            assert t <= self.config.sequence_length, \
                f"Sequence length {t} exceeds model configured sequence_length {self.config.sequence_length}"
            pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = tok_emb + pos_emb
        else:
            # If no 'wpe' (e.g. pure attention model), just use token embeddings
            x = tok_emb
            
        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if self.use_asoft:
            original_x_dtype = x.dtype # Store original dtype, e.g., bfloat16
            x_for_asoft = x.to(torch.float32) # Cast input to asoft to float32

            if targets is not None:
                # training/validation path
                reshaped_x_for_asoft = x_for_asoft.view(-1, x_for_asoft.size(-1))
                reshaped_targets = targets.view(-1) # targets are long, no cast needed

                # self.asoft parameters are float32, input is float32
                output_asoft = self.asoft(reshaped_x_for_asoft, reshaped_targets)
                loss = output_asoft.loss.to(original_x_dtype) # Cast loss back

                if return_logits:
                    # self.asoft.log_prob expects float32 input, will produce float32 output
                    logits_float = self.asoft.log_prob(x_for_asoft) 
                    logits = logits_float.to(original_x_dtype) # Cast logits back
                else:
                    logits = None
            else:
                # inference-time: only forward on the very last position
                # self.asoft.log_prob expects float32 input
                logits_float = self.asoft.log_prob(x_for_asoft[:, [-1], :])
                logits = logits_float.to(original_x_dtype) # Cast logits back
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
        if not return_logits and not self.use_asoft: # asoft handles its own logit return
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
        "--sequence_length", type=int, default=1024, help="sequence length"
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
        default="2000,10000", # Default from common practice, e.g. 2k, 10k, vocab_size (implicit last cutoff)
        help="Comma-separated string of cutoff values for adaptive softmax, e.g., '2000,10000'. The vocab size is implicitly the last cutoff."
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
    # AFNO specific arguments
    parser.add_argument("--mixer", type=str, default="attn", choices=["attn", "afno"], help="Type of mixer to use: attn or afno")
    parser.add_argument("--afno_num_blocks", type=int, default=8, help="Number of blocks for AFNO1D")
    parser.add_argument("--afno_sparsity_threshold", type=float, default=0.01, help="Sparsity threshold for AFNO1D")
    parser.add_argument("--afno_hard_thresholding_fraction", type=float, default=1.0, help="Hard thresholding fraction for AFNO1D")
    parser.add_argument("--afno_hidden_size_factor", type=int, default=1, help="Hidden size factor for AFNO1D MLP")
    parser.add_argument("--bias", action="store_true", help="Enable bias in Linear layers")

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
        wandb.save("train_gpt2_afno_asoft_clean.py")
        wandb.save("run_afno_asoft.sh")

    tokens_per_iter = B * T * ddp_world_size * args.grad_accumulation_steps
    print0(f"tokens per iteration: {tokens_per_iter:,}")

    # Decide mixed-precision mode
    if args.precision == "fp16":
        autocast_dtype = torch.float16
        # scaler = torch.cuda.amp.GradScaler() # Not used for now, as ASoft is FP32
        print0("Using mixed precision: FP16 (ASoft will remain FP32)")
    elif args.precision == "bf16":
        autocast_dtype = torch.bfloat16
        # scaler = None
        print0("Using mixed precision: BF16 (ASoft will remain FP32)")
    else:  # fp32
        autocast_dtype = torch.float32
        # scaler = None
        print0("Using full precision: FP32")

    # set up a context manager following the desired dtype and device
    ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype!=torch.float32)

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    tokens_per_iter_val = args.val_batch_size * T * ddp_world_size
    assert VAL_TOKENS % tokens_per_iter_val == 0
    val_steps = VAL_TOKENS // tokens_per_iter_val

    val_loader = DistributedDataLoader(
        args.input_val_bin, args.val_batch_size, T, ddp_rank, ddp_world_size
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

    model_config_params = {
        "vocab_size": effective_vocab_size, # Use actual vocab size from data
        # Add other relevant params from GPTConfig defaults if they are not in model_size_configs
        "n_layer": 12, "n_head": 12, "n_embd": 768, # Default d12, will be updated by model_size_configs
        "bias": args.bias, # Added for AFNO
        "mixer": args.mixer, # Added for AFNO
        "afno_num_blocks": args.afno_num_blocks, # Added for AFNO
        "afno_sparsity_threshold": args.afno_sparsity_threshold, # Added for AFNO
        "afno_hard_thresholding_fraction": args.afno_hard_thresholding_fraction, # Added for AFNO
        "afno_hidden_size_factor": args.afno_hidden_size_factor, # Added for AFNO
        "sequence_length": args.sequence_length # Added for AFNO
    }
    model_size_configs = {
        "d12": dict(n_layer=12, n_head=12, n_embd=768),
        "d24": dict(n_layer=24, n_head=16, n_embd=1024),
        "d36": dict(n_layer=36, n_head=20, n_embd=1280),
        "d48": dict(n_layer=48, n_head=25, n_embd=1600),
    }
    model_config_params.update(model_size_configs[args.model])
    model_config = GPTConfig(**model_config_params)
    
    # decide parameter / activation dtype
    param_dtype = (
        torch.bfloat16 if args.precision == "bf16"
        else torch.float32 if args.precision == "fp32"
        else torch.float16
    )
    # enable Flash-Attention v2 & memory-efficient SDPA kernels ----
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Set up output layer configuration
    use_asoft = args.adaptive_softmax
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

    model = GPT(model_config, 
                use_asoft=use_asoft, cutoffs=asoft_cutoffs_parsed, div_value=args.asoft_div_value)
    
    # Apply precision casting:
    # 1. Move model to device
    model.to(device) 
    # 2. Cast general model parameters if not FP32
    if param_dtype != torch.float32:
        model.to(dtype=param_dtype)
    # 3. If using Adaptive Softmax, ensure its parameters are FP32
    if use_asoft and hasattr(model, 'asoft'):
        model.asoft.to(torch.float32)
        print0(f"Casted model.asoft to torch.float32. Rest of model is {args.precision}.")

    model = model.train()
    
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

    if master_process:
        log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
        os.makedirs("logs/%s" % run_id, exist_ok=True)
        torch.save(log, "logs/%s/final.pt" % run_id)

    # -------------------------------------------------------------------------
    # clean up nice
    destroy_process_group()