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
from torch import nn
torch.set_float32_matmul_precision('high')
# For BF16 performance
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # choose fastest algorithm
import torch.distributed as dist
import torch.nn.functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Import LightRNN modules
from lightrnn_old import LightRNNCodebook, LightRNNEmbedding, LightRNNDecoder

with open(sys.argv[0]) as f:
    code = f.read()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model


class Rotary(torch.nn.Module):
    """Precomputes rotary cos/sin tables up to max_seq_len once to be CUDA-graph safe."""

    def __init__(self, dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # (T, dim/2)
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        # register as buffers so they move with .to(device)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x):
        seq_len = x.shape[1]
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


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
        self.rotary = Rotary(self.head_dim, config.sequence_length)

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
# The main GPT-2 model


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    sequence_length: int = 1024  # max seq len for rotary tables (and pos emb)
    # Precision parameter
    precision: str = "bf16"
    # LightRNN parameters
    tie_embedding: str = field(default="full", metadata={"choices": ["full", "light"]})
    table_size: int = 0  # Will be calculated if using light embedding


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.tie_embedding == "light":
            if config.table_size == 0: # calculate if not provided
                config.table_size = math.ceil(math.sqrt(config.vocab_size))
            print0(f"Using LightRNN embedding with table size: {config.table_size}")
            self.codebook = LightRNNCodebook(config.vocab_size, config.table_size)
            self.transformer = nn.ModuleDict(
                dict(
                    wte=LightRNNEmbedding(self.codebook, config.n_embd),
                    h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                )
            )
            self.decoder = LightRNNDecoder(self.codebook, config.n_embd)
            self.lm_head = None # Not used with LightRNN
        else:
            print0("Using standard full embedding.")
            self.transformer = nn.ModuleDict(
                dict(
                    wte=nn.Embedding(config.vocab_size, config.n_embd),
                    h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                )
            )
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight # Tie weights
            self.codebook = None
            self.decoder = None

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        # pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # Not used with RoPE

        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if self.config.tie_embedding == "light":
            # LightRNN handles loss calculation internally if targets are provided
            # and returns its own form of "logits" (e.g., row_logits)
            logits, loss = self.decoder(x, targets)
        else:
            # Standard GPT behavior
            if targets is not None:
                logits = self.lm_head(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )
            else:
                logits = self.lm_head(x[:, [-1], :])
                loss = None
        
        # Detach outputs from the internal CUDA-graph memory so that subsequent
        # replays cannot overwrite tensors that autograd might still hold.
        if loss is not None:
            loss = loss.clone()
        if logits is not None and return_logits:
            logits = logits.clone()

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )
        return optimizer

    # ------------------------------------------------------------------
    # Text generation helpers (greedy, single-GPU, no kv-cache for brevity)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int = 32):
        """Greedy decode continuation for a batch of token indices (B, T)."""

        self.eval()

        for _ in range(max_new_tokens):
            # crop context if sequence grows beyond model limit
            idx_cond = idx[:, -self.config.n_embd :]

            # Forward with return_logits=True to obtain last-step logits
            logits, _ = self(idx_cond, targets=None, return_logits=True)

            if self.config.tie_embedding == "full":
                # logits is (B,1,vocab)
                next_token = logits[:, -1, :].argmax(-1, keepdim=True)  # (B,1)
            else:
                # logits returned as tuple (row_logits, col_logits)
                row_logits, col_logits = logits  # each (B,1,R)
                row_id = row_logits.squeeze(1).argmax(-1)  # (B,)
                col_id = col_logits.squeeze(1).argmax(-1)  # (B,)
                token_id = row_id * self.codebook.table_size + col_id
                token_id.clamp_(max=self.config.vocab_size - 1)
                next_token = token_id.unsqueeze(1)

            # append
            idx = torch.cat([idx, next_token], dim=1)

        return idx


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
    # additions for optimization
    parser.add_argument(
        "--precision", 
        type=str,
        choices=["fp32", "fp16", "bf16"], 
        default="bf16",
        help="Computation precision for forward/backward pass (fp32, fp16, bf16)"
    )
    parser.add_argument(
        "--target_val_loss",
        type=float,
        default=3.3821,
        help="Target validation loss to stop training."
    )
    # LightRNN specific arguments
    parser.add_argument(
        "--tie_embedding",
        type=str,
        default="full",
        choices=["full", "light"],
        help="Type of embedding/decoder: full or light (LightRNN)"
    )
    parser.add_argument(
        "--table_size",
        type=int,
        default=0, # Auto-calculate if 0
        help="Table size for LightRNN (approx sqrt(vocab_size))"
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

        run_name = f"gpt2-{args.model}"
        if args.tie_embedding == "light":
            run_name += "-lightrnn"
        run_name += f" {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        wandb.init(project="benchmark_gpt2", name=run_name)
        wandb.config.update(args)
        wandb.save(sys.argv[0]) # Save the current script
        # wandb.save("run_lightrnn.sh") # If you have a specific run script

    tokens_per_iter = B * T * ddp_world_size * args.grad_accumulation_steps
    print0(f"tokens per iteration: {tokens_per_iter:,}")

    # Decide mixed-precision mode
    if args.precision == "fp16":
        autocast_dtype = torch.float16
        # scaler = torch.cuda.amp.GradScaler() # Not using scaler for AdamW
        print0("Using mixed precision: FP16")
    elif args.precision == "bf16":
        autocast_dtype = torch.bfloat16
        # scaler = None
        print0("Using mixed precision: BF16 (no gradient scaling needed)")
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
    assert VAL_TOKENS % tokens_per_iter_val == 0, f"VAL_TOKENS ({VAL_TOKENS}) must be divisible by tokens_per_iter_val ({tokens_per_iter_val})"
    val_steps = VAL_TOKENS // tokens_per_iter_val

    val_loader = DistributedDataLoader(
        args.input_val_bin, args.val_batch_size, T, ddp_rank, ddp_world_size
    )
    x, y = train_loader.next_batch()

    # init the model from scratch
    num_vocab = 50257
    model_config_dict = {
        "d12": dict(n_layer=12, n_head=12, n_embd=768),
        "d24": dict(n_layer=24, n_head=16, n_embd=1024),
        "d36": dict(n_layer=36, n_head=20, n_embd=1280),
        "d48": dict(n_layer=48, n_head=25, n_embd=1600),
    }[args.model]

    model_config = GPTConfig(
        vocab_size=num_vocab, 
        **model_config_dict,
        sequence_length=args.sequence_length,
        precision=args.precision,
        tie_embedding=args.tie_embedding,
        table_size=args.table_size
    )
    # decide parameter / activation dtype
    if args.precision == "bf16":
        param_dtype = torch.bfloat16
    elif args.precision == "fp16":
        param_dtype = torch.float16
    else:
        param_dtype = torch.float32
    
    # enable Flash-Attention v2 & memory-efficient SDPA kernels ----
    # Only enable if not using LightRNN's custom attention/mixer if it had one
    # if model_config.tie_embedding == "full":  # Enable flash attention only when using standard softmax
    #     torch.backends.cuda.enable_flash_sdp(True)
    #     torch.backends.cuda.enable_mem_efficient_sdp(True)
    #     print0("Flash Attention + memory-efficient SDPA enabled (full soft-max path).")

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    model = GPT(model_config).to(device, dtype=param_dtype)
    model = model.train().cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    print0("compiling the model...")
    # The generic CUDA-graph path still causes buffer-reuse issues in the
    # transformer stack even with mark_step_begin guards.  Switch to the
    # special mode that disables CUDA-graphs entirely but keeps Triton /
    # template autotuning – we keep most of the speed-up (≈2× over eager)
    # without the runtime crash.
    model = torch.compile(model, mode="max-autotune-no-cudagraphs")

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
        logfile = os.path.join(args.output_dir, f"{run_id}.log")
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
                    # Ensure return_logits=False if LightRNN handles loss internally and its "logits" are not vocab-sized
                    torch.compiler.cudagraph_mark_step_begin()
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
                        f.write(f"s:{step} val:{val_loss.item():f}\n")

            # Check for early stopping based on target validation loss
            if val_loss.item() <= args.target_val_loss:
                print0(f"Target validation loss {args.target_val_loss} reached at step {step}. Stopping training after {training_time_ms/1000:.2f} seconds ({training_time_ms/3600000:.2f} hours).")
                last_step = True # This will break the main loop after this block

            # restart the clock
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        train_loss_accum = 0.0 # Use float for accumulation
        for micro_step in range(args.grad_accumulation_steps):
            model.require_backward_grad_sync = (
                micro_step == args.grad_accumulation_steps - 1
            )  
            with ctx:
                torch.compiler.cudagraph_mark_step_begin()
                # Ensure return_logits=False if LightRNN handles loss internally
                _, loss = model(x, y, return_logits=False)
                loss = loss / args.grad_accumulation_steps
                train_loss_accum += loss.item() # Accumulate .item() to save memory
            
            x, y = train_loader.next_batch()
            loss.backward() # No scaler needed for AdamW with BF16/FP32

        # Convert accumulated loss back to tensor for DDP reduction
        train_loss_tensor = torch.tensor(train_loss_accum, device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
        lossf = train_loss_tensor.item()

        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # step the optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        
        torch.cuda.synchronize()
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(
            f"step:{step}/{args.num_iterations} | loss {lossf:.6f} | train_time:{approx_training_time_ms/1000:.2f}s | step_avg:{approx_training_time_ms/(step+1):.2f}ms | lr {lr:.2e}"
        )
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write(f"s:{step} trn:{lossf:f} lr:{lr:e}\n")

        if master_process and (step + 1) % args.save_every == 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_config, # Save model_config
                'iter_num': step,
                'args': args, # Save command line args
                'code': code,
            }
            checkpoint_dir = os.path.join(args.output_dir, run_id)
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"model_step{step:06d}.pt"))

    print0(
        f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )

    if master_process:
        final_checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_config,
            'iter_num': step, # Last completed step
            'args': args,
            'code': code,
            'final_val_loss': val_loss.item() if 'val_loss' in locals() else float('inf')
        }
        final_checkpoint_dir = os.path.join(args.output_dir, run_id)
        os.makedirs(final_checkpoint_dir, exist_ok=True)
        torch.save(final_checkpoint, os.path.join(final_checkpoint_dir, "final.pt"))

    destroy_process_group()