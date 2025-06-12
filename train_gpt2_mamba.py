import os
import sys
import uuid
import math
import glob
from dataclasses import dataclass, field

# Reduce VRAM usage by reducing fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
# For BF16 performance
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
# Set TF32 precision for matmul for better performance on Ampere GPUs
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # choose fastest algorithm
from torch import nn
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
import torch.distributed as dist
import torch.nn.functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Import optimizers
from tensorgrad import TensorGRaD

# Optional Lion optimizer (install with `pip install lion-pytorch`)
try:
    from lion_pytorch import Lion
    LION_AVAILABLE = True
except ImportError:
    LION_AVAILABLE = False

# For memory-saving checkpointing
import torch.utils.checkpoint as ckpt

with open(sys.argv[0]) as f:
    code = f.read()

# -----------------------------------------------------------------------------
# Mamba SSM implementation

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("Warning: mamba_ssm not available. Install with: pip install mamba-ssm")
    MAMBA_AVAILABLE = False
    
    # Fallback implementation for testing
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
            super().__init__()
            self.d_model = d_model
            self.linear = nn.Linear(d_model, d_model)
            
        def forward(self, x):
            return self.linear(x)

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
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
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


class MambaBlock(nn.Module):
    """Mamba SSM block for sequence modeling"""
    
    def __init__(self, config):
        super().__init__()
        self.mamba = Mamba(
            d_model=config.n_embd,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
        )

    def forward(self, x):
        # Mamba handles causality natively
        return self.mamba(x)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.use_checkpoint = getattr(config, "use_checkpoint", False)

    def _forward_impl(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return ckpt.checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.mixer == "mamba":
            self.mixer = MambaBlock(config)
        elif config.mixer == "attn":
            self.mixer = CausalSelfAttention(config)
        else:
            raise ValueError(f"Unknown mixer type: {config.mixer}")
        
        self.mlp = MLP(config)
        self.mixer_scale = 1 / math.sqrt(2 * config.n_layer)
        self.dropout = nn.Dropout(config.dropout)
        self.use_checkpoint = getattr(config, "use_checkpoint", False)

    def forward(self, x):
        x = x + self.mixer_scale * self.mixer(rmsnorm(x)) 
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
    mixer: str = "mamba"  # "mamba" or "attn"
    sequence_length: int = 2048
    # Mamba-specific parameters
    mamba_d_state: int = 16      # SSM state expansion factor
    mamba_d_conv: int = 4        # Local convolution width  
    mamba_expand: int = 2        # Block expansion factor
    dropout: float = 0.0         # Dropout probability
    use_checkpoint: bool = False # Enable gradient checkpointing per Block


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        )
        
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        
        # For Mamba, no explicit positional embeddings - handled through state
        x = tok_emb

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

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
        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay,
        learning_rate,
        betas,
        device_type,
        optimizer_type="adamw",
        # TensorGRaD specific params
        rank=4,
        sparsity=0.01,
        lambda_sparse=1.0,
        update_freq=1,
    ):
        """Configure optimizer based on type"""
        
        if optimizer_type == "tensorgrad":
            optimizer = TensorGRaD(
                self.parameters(),
                lr=learning_rate,
                betas=betas,
                weight_decay=weight_decay,
                rank=rank,
                sparsity=sparsity,
                lambda_sparse=lambda_sparse,
                update_freq=update_freq,
            )
        elif optimizer_type == "adamw":
            # Standard fused AdamW
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_rate,
                betas=betas,
                weight_decay=weight_decay,
                fused=True if device_type == "cuda" else False,
            )
        elif optimizer_type == "lion":
            if not LION_AVAILABLE:
                raise ImportError("lion-pytorch not installed. install with `pip install lion-pytorch`")
            optimizer = Lion(
                self.parameters(),
                lr=learning_rate,
                betas=betas,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
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
    print0(f"Mamba SSM available: {MAMBA_AVAILABLE}")

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
        default=8,
        help="batch size, in units of #batch dimensions",
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=64,
        help="number of gradient accumulation steps",
    )
    parser.add_argument(
        "--sequence_length", type=int, default=2048, help="sequence length for the model"
    )
    # workload (number of steps)
    parser.add_argument(
        "--num_iterations", type=int, default=3000, help="number of iterations to run"
    )
    # optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=256, help="learning rate warmup iterations"
    )
    parser.add_argument(
        "--warmdown_iters",
        type=int,
        default=1024,
        help="learning rate warmdown iterations",
    )
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
    parser.add_argument("--bias", action="store_true", help="Enable bias in Linear layers")
    
    # Optimizer selection
    parser.add_argument(
        "--optimizer", 
        type=str, 
        default="adamw", 
        choices=["adamw", "tensorgrad", "lion"],
        help="Optimizer type: adamw (fused), tensorgrad, or lion"
    )
    
    # evaluation
    parser.add_argument(
        "--val_loss_every",
        type=int,
        default=128,
        help="every how many steps to evaluate val loss?",
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
        "--target_val_loss",
        type=float,
        default=3.3821,
        help="Target validation loss to stop training."
    )
    
    # Mamba specific arguments
    parser.add_argument("--mixer", type=str, default="mamba", choices=["mamba", "attn"], help="Type of mixer to use: mamba or attn")
    parser.add_argument("--mamba_d_state", type=int, default=16, help="Mamba SSM state expansion factor")
    parser.add_argument("--mamba_d_conv", type=int, default=4, help="Mamba local convolution width")
    parser.add_argument("--mamba_expand", type=int, default=2, help="Mamba block expansion factor")

    # TensorGRaD specific arguments (only used if --optimizer tensorgrad)
    parser.add_argument("--tg_rank", type=int, default=4, help="Rank for low-rank approximation in TensorGRaD")
    parser.add_argument("--tg_sparsity", type=float, default=0.02, help="Fraction of gradient elements kept in sparse component")
    parser.add_argument("--tg_lambda", type=float, default=1.0, help="Scaling factor for sparse update in TensorGRaD")
    parser.add_argument("--tg_update_freq", type=int, default=4, help="Frequency (in steps) for refreshing sparse indices")

    # Optimizer hyperparams
    parser.add_argument("--beta1", type=float, default=0.9, help="optimizer beta1")
    parser.add_argument("--beta2", type=float, default=0.99, help="optimizer beta2")

    # Dropout hyperparam
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability in MLP layers")

    # LR schedule options
    parser.add_argument("--lr_schedule", type=str, default="linear", choices=["linear", "cosine"], help="Learning rate schedule: linear or cosine")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR ratio relative to peak when using cosine schedule")

    # Memory checkpoint flag
    parser.add_argument("--use_checkpoint", action="store_true", help="Enable gradient checkpointing (saves memory, extra compute)")

    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert args.model in {"d12", "d24", "d36", "d48"}
    
    if args.mixer == "mamba" and not MAMBA_AVAILABLE:
        print("ERROR: Mamba not available. Install with: pip install mamba-ssm")
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
        wandb.init(project="benchmark_gpt2", name=f"gpt2-{args.model} {start_time} mamba-{args.optimizer}")
        wandb.config.update(args)

    tokens_per_iter = B * T * ddp_world_size * args.grad_accumulation_steps
    print0(f"tokens per iteration: {tokens_per_iter:,}")

    # set up a context manager following the desired dtype and device
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

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
    num_vocab = 50257
    model_config_params = {
        "vocab_size": num_vocab,
        "bias": args.bias,
        "mixer": args.mixer,
        "mamba_d_state": args.mamba_d_state,
        "mamba_d_conv": args.mamba_d_conv,
        "mamba_expand": args.mamba_expand,
        "sequence_length": args.sequence_length,
        "dropout": args.dropout,
        "use_checkpoint": args.use_checkpoint
    }
    model_size_configs = {
        "d12": dict(n_layer=12, n_head=12, n_embd=768),
        "d24": dict(n_layer=24, n_head=16, n_embd=1024),
        "d36": dict(n_layer=36, n_head=20, n_embd=1280),
        "d48": dict(n_layer=48, n_head=25, n_embd=1600),
    }
    model_config_params.update(model_size_configs[args.model])
    model_config = GPTConfig(**model_config_params)
    
    model = GPT(model_config)
    model = model.train().cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    print0("compiling the model...")
    model = torch.compile(
        model,
        # options={"triton.cudagraphs": False}
    )  # NOTE: this might cause issues depending on your GPU, consider turning it off

    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module  # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        device_type=device,
        optimizer_type=args.optimizer,
        rank=args.tg_rank,
        sparsity=args.tg_sparsity,
        lambda_sparse=args.tg_lambda,
        update_freq=args.tg_update_freq,
    )

    print0(f"Using optimizer: {args.optimizer}")

    # learning-rate scheduler helper
    min_lr = args.learning_rate * args.min_lr_ratio

    def lr_linear(it):
        effective_max_iter = original_num_iterations
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        elif it < effective_max_iter - args.warmdown_iters:
            return args.learning_rate
        elif it < effective_max_iter:
            decay_ratio = (effective_max_iter - it) / args.warmdown_iters
            return args.learning_rate * decay_ratio
        else:
            return min_lr

    def lr_cosine(it):
        effective_max_iter = original_num_iterations
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        elif it >= effective_max_iter:
            return min_lr
        else:
            progress = (it - args.warmup_iters) / max(1, effective_max_iter - args.warmup_iters)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr + (args.learning_rate - min_lr) * cosine

    get_lr = lr_cosine if args.lr_schedule == "cosine" else lr_linear

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

    # Extension mechanism for target validation loss
    extension_count = 0
    max_extensions = 10
    original_num_iterations = args.num_iterations

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
                for _ in range(val_steps):  # always fixed number of validation steps
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss /= val_steps
            # log to console and to file
            print0(f"step:{step}/{args.num_iterations} | val loss {val_loss:.6f}")
            
            # Check for early stopping based on target validation loss
            if val_loss.item() <= args.target_val_loss:
                print0(f"Target validation loss {args.target_val_loss} reached at step {step}. Stopping training after {training_time_ms/1000:.2f} seconds ({training_time_ms/3600000:.2f} hours).")
                last_step = True # This will break the main loop after this block
            
            if master_process:
                if args.log_wandb:
                    wandb.log({"val_loss": val_loss}, step=step * tokens_per_iter)
                    wandb.log({"time": training_time_ms}, step=step * tokens_per_iter)
                if logfile is not None:
                    with open(logfile, "a") as f:
                        f.write("s:%d val:%f\n" % (step, val_loss))

            # restart the clock
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # Check if we've reached the end of planned iterations without hitting target
        if last_step:
            # If we reached num_iterations without hitting target, try extensions
            if step >= args.num_iterations and extension_count < max_extensions:
                # Check if we should extend training
                if 'val_loss' in locals() and val_loss.item() > args.target_val_loss:
                    extension_count += 1
                    args.num_iterations += args.val_loss_every
                    print0(f"Extension {extension_count}/{max_extensions}: Target not reached, extending by {args.val_loss_every} steps to {args.num_iterations}")
                    last_step = False  # Continue training
                else:
                    break  # Target reached or no more extensions
            else:
                # Check if we exhausted all extensions without reaching target
                if extension_count >= max_extensions and 'val_loss' in locals() and val_loss.item() > args.target_val_loss:
                    print0(f"❌ TRAINING FAILED: Could not reach target validation loss {args.target_val_loss} after {max_extensions} extensions. Final loss: {val_loss.item():.6f}")
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
        # log to logfile
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