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
from torch import nn
torch.set_float32_matmul_precision('high')
# For BF16 performance
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
# Enable NVIDIA Ampere-level TF32 for matmul and conv for additional speedup (RTX 30/40 series)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import torch.nn.functional as F
# Import config from torch._inductor if available for coordinate_descent tuning (helps torch.compile performance)
try:
    import torch._inductor.config as config
except ImportError:
    config = None  # torch._inductor might not be available in this build; we guard its usage accordingly
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
# The main GPT-2 model


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
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

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)  # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )
        return optimizer


# -----------------------------------------------------------------------------
# Echo State Network (ESN) definitions

@dataclass
class ESNConfig:
    vocab_size: int = 50257
    embed_dim: int = 256       # Dimension of token embeddings
    reservoir_size: int = 2048 # Number of neurons in the reservoir
    # ESN specific
    spectral_radius: float = 0.99 # Scales recurrent weights
    input_scaling: float = 0.1    # Scales input weights
    # sparsity: float = 0.0 # Sparsity of reservoir weights (0.0 means dense)
    precision: str = "bf16"
    use_light_rnn_output: bool = True

    # Derived for LightRNN
    sqrt_vocab_size: int = 0
    def __post_init__(self):
        if self.use_light_rnn_output:
            self.sqrt_vocab_size = int(math.ceil(math.sqrt(self.vocab_size)))

class ESN(nn.Module):
    def __init__(self, config: ESNConfig):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.embed_dim)
        # Linear layer to project embeddings to reservoir_size, acting as W_in
        self.input_proj = nn.Linear(config.embed_dim, config.reservoir_size, bias=False)
        
        # The reservoir itself using nn.RNN
        self.reservoir_rnn = nn.RNN(
            input_size=config.reservoir_size, # Input features to RNN cell
            hidden_size=config.reservoir_size, # Size of reservoir state
            num_layers=1,                      # A single recurrent layer
            nonlinearity='tanh',               # Standard ESN activation
            batch_first=True,
            bias=True                          # Reservoir bias b_x
        )
        
        # Output layer (readout) - only this is trained
        if config.use_light_rnn_output:
            self.lm_head1 = nn.Linear(config.reservoir_size, config.sqrt_vocab_size, bias=False)
            self.lm_head2 = nn.Linear(config.reservoir_size, config.sqrt_vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(config.reservoir_size, config.vocab_size, bias=False)

        self._initialize_and_freeze_weights()

    def _map_targets_to_2d(self, targets_1d):
        # targets_1d shape: (N_valid_targets)
        # returns (targets_row, targets_col) each of shape (N_valid_targets)
        factor = self.config.sqrt_vocab_size
        targets_row = targets_1d // factor
        targets_col = targets_1d % factor
        return targets_row, targets_col

    def _initialize_and_freeze_weights(self):
        # 1. Initialize wte and input_proj (randomly, then freeze)
        nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
        self.wte.weight.requires_grad = False
        
        nn.init.uniform_(self.input_proj.weight, -self.config.input_scaling, self.config.input_scaling)
        self.input_proj.weight.requires_grad = False

        # 2. Initialize reservoir_rnn weights (W_res and its bias b_x) and freeze them
        for name, param in self.reservoir_rnn.named_parameters():
            if 'weight_ih' in name: # Corresponds to part of W_in projected from input_proj output
                # These weights connect the projected input to the hidden state update.
                # Initialize uniformly, scale might be implicitly handled by input_scaling on input_proj
                nn.init.uniform_(param, -1.0, 1.0) 
            elif 'weight_hh' in name: # Recurrent weights (W_res)
                nn.init.uniform_(param.data, -1.0, 1.0)
                W_matrix = param.data
                # Calculate eigenvalues for square matrix
                if W_matrix.size(0) == W_matrix.size(1):
                    eigenvalues = torch.linalg.eigvals(W_matrix).abs()
                    current_spectral_radius = torch.max(eigenvalues)
                    if current_spectral_radius > 1e-9:
                        scaling_factor = self.config.spectral_radius / current_spectral_radius
                        param.data *= scaling_factor
                    else:
                         param.data *= self.config.spectral_radius # Scale to be small if SR is ~0
            elif 'bias' in name: # b_x (reservoir bias)
                 nn.init.uniform_(param.data, -0.1, 0.1) # Small random bias for reservoir dynamics
            param.requires_grad = False
        
        # lm_head weights (lm_head1, lm_head2 or lm_head) are trainable 
        # and will be initialized by PyTorch default (Kaiming Uniform for Linear).

    def forward(self, idx, targets=None, return_logits=True, prev_state=None):
        b, t = idx.size()
        
        emb = self.wte(idx) # (b, t, embed_dim)
        # Project embeddings to the dimension expected by the RNN reservoir
        projected_emb = self.input_proj(emb) # (b, t, reservoir_size)
        
        if prev_state is None:
            # h_0 shape: (num_layers * num_directions, batch, hidden_size)
            h_0 = torch.zeros(1, b, self.config.reservoir_size, device=idx.device, dtype=projected_emb.dtype)
        else:
            h_0 = prev_state # Should be (1, b, reservoir_size)
            
        # output_rnn: (b, t, reservoir_size) - features from all RNN time steps
        # h_n: (1, b, reservoir_size) - final hidden state
        output_rnn, h_n = self.reservoir_rnn(projected_emb, h_0)
        current_state_for_next_batch = h_n 
        
        # Determine which states to use for the LM head
        states_for_lm_head = output_rnn if targets is not None else output_rnn[:, [-1], :]
        
        loss = None
        if self.config.use_light_rnn_output:
            logits1 = self.lm_head1(states_for_lm_head) # (B, T_out, sqrt_V)
            logits2 = self.lm_head2(states_for_lm_head) # (B, T_out, sqrt_V)
            
            if targets is not None:
                # Flatten targets and corresponding logits for loss calculation
                # T_out is t if targets is not None, else 1
                T_out_actual = states_for_lm_head.size(1)
                flat_logits1 = logits1.reshape(-1, self.config.sqrt_vocab_size) # (B*T_out, sqrt_V)
                flat_logits2 = logits2.reshape(-1, self.config.sqrt_vocab_size) # (B*T_out, sqrt_V)
                flat_targets = targets.reshape(-1) # (B*T_out)

                # Filter out ignore_index targets (-1)
                valid_targets_mask = (flat_targets != -1)
                valid_flat_targets = flat_targets[valid_targets_mask]
                
                if valid_flat_targets.numel() > 0: # Proceed only if there are valid targets
                    targets_row, targets_col = self._map_targets_to_2d(valid_flat_targets)
                    
                    # Select logits corresponding to valid targets
                    valid_flat_logits1 = flat_logits1[valid_targets_mask]
                    valid_flat_logits2 = flat_logits2[valid_targets_mask]

                    loss1 = F.cross_entropy(valid_flat_logits1, targets_row)
                    loss2 = F.cross_entropy(valid_flat_logits2, targets_col)
                    loss = loss1 + loss2
                else: # No valid targets, loss is 0 or some other appropriate value
                    loss = torch.tensor(0.0, device=idx.device, dtype=logits1.dtype) # Ensure correct dtype and device
            
            # Combine logits for inference/evaluation if needed (for perplexity, generation)
            # log P(token_rc) = log P(row_r) + log P(col_c)
            # For full logits, we create a sqrt_V x sqrt_V grid of log_probs
            if return_logits:
                log_probs1 = F.log_softmax(logits1, dim=-1) # (B, T_out, sqrt_V)
                log_probs2 = F.log_softmax(logits2, dim=-1) # (B, T_out, sqrt_V)
                # Outer sum: (B, T_out, sqrt_V, 1) + (B, T_out, 1, sqrt_V) -> (B, T_out, sqrt_V, sqrt_V)
                combined_log_probs = log_probs1.unsqueeze(-1) + log_probs2.unsqueeze(-2)
                # Reshape to (B, T_out, sqrt_V*sqrt_V) and trim to vocab_size
                logits = combined_log_probs.view(b, states_for_lm_head.size(1), -1)
                logits = logits[:, :, :self.config.vocab_size]
            else:
                logits = None

        else: # Standard lm_head
            logits = self.lm_head(states_for_lm_head) # (B, T_out, vocab_size)
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        if not return_logits and loss is None : # Edge case if targets is None and return_logits is False
             pass # logits is already None
        elif not return_logits:
            logits = None # Ensure logits is None if not requested

        return logits, loss, current_state_for_next_batch

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        trainable_params = []
        if self.config.use_light_rnn_output:
            trainable_params.extend(list(self.lm_head1.parameters()))
            trainable_params.extend(list(self.lm_head2.parameters()))
        else:
            trainable_params.extend(list(self.lm_head.parameters()))
        
        # Ensure there are trainable parameters before creating optimizer
        if not trainable_params:
            print0("Warning: No trainable parameters found for the ESN model.")
            # Return a dummy optimizer or handle appropriately
            # For now, let's allow it to proceed, Pytorch might handle empty param list gracefully for some ops.
            # Or, more robustly:
            # return None # Or raise an error if this state is unexpected
            # For now, creating optimizer with empty list might error out, let's assume it's caught if no params.
            # Actually, AdamW will error on empty list.
            # If no trainable params, it implies an issue or a fully fixed model.
            # In ESN, output layer *must* be trainable.
            if master_process: # only print from master
                print("ESN configure_optimizers: No trainable parameters specified. This is likely an error.")
            # Fallback to a dummy parameter if list is empty to avoid crash, though this indicates a setup issue
            if not trainable_params:
                 dummy_param = nn.Parameter(torch.empty(0)) # Or handle error appropriately
                 trainable_params.append(dummy_param)


        optimizer = torch.optim.AdamW(
            trainable_params, lr=learning_rate, weight_decay=weight_decay, betas=betas
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
        default="d12", # This will be ignored for ESN, but kept for now to avoid breaking arg parsing for other uses.
        help="d12|d24|d36|d48 (ignored for ESN, ESN params are separate)",
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
    parser.add_argument(
        "--val_sequence_length",
        type=int,
        default=1024,
        help="sequence length used for validation (kept constant for fair metric)",
    )
    # additions for ESN
    parser.add_argument("--esn_embed_dim", type=int, default=256, help="ESN embedding dimension")
    parser.add_argument("--esn_reservoir_size", type=int, default=2048, help="ESN reservoir size")
    parser.add_argument("--esn_spectral_radius", type=float, default=0.99, help="ESN spectral radius")
    parser.add_argument("--esn_input_scaling", type=float, default=0.1, help="ESN input scaling")
    parser.add_argument("--esn_no_light_rnn_output", action="store_true", help="Disable LightRNN style output layer for ESN")
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
        wandb.save("train_reservoir.py")
        wandb.save("run_reservoir.sh")

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
    train_loader = DistributedDataLoader(args.input_bin, B, args.val_sequence_length, ddp_rank, ddp_world_size)
    val_loader = None
    tokens_per_iter_val = args.val_batch_size * args.val_sequence_length * ddp_world_size
    assert VAL_TOKENS % tokens_per_iter_val == 0, f"VAL_TOKENS ({VAL_TOKENS}) must be divisible by tokens_per_iter_val ({tokens_per_iter_val})"
    val_steps = VAL_TOKENS // tokens_per_iter_val

    val_loader = DistributedDataLoader(
        args.input_val_bin, args.val_batch_size, args.val_sequence_length, ddp_rank, ddp_world_size
    )
    x, y = train_loader.next_batch()

    # init the model from scratch
    num_vocab = 50257

    # --- ESN Model Configuration ---
    esn_config = ESNConfig(
        vocab_size=num_vocab,
        embed_dim=args.esn_embed_dim,
        reservoir_size=args.esn_reservoir_size,
        spectral_radius=args.esn_spectral_radius,
        input_scaling=args.esn_input_scaling,
        precision=args.precision,
        use_light_rnn_output=not args.esn_no_light_rnn_output
    )
    print0(f"Using ESN configuration: {esn_config}")
    # --- End ESN Model Configuration ---

    # decide parameter / activation dtype
    dtype = (
        torch.bfloat16 if args.precision == "bf16"
        else torch.float32 if args.precision == "fp32"
        else torch.float16
    )
    # enable Flash-Attention v2 & memory-efficient SDPA kernels ----
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # model = GPT(model_config).to(device, dtype=dtype) # Old GPT model
    model = ESN(esn_config).to(device, dtype=dtype) # New ESN model

    model = model.train().cuda()
    if config is not None and hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    print0("compiling the model (torch.compile)...")
    try:
        model = torch.compile(model)
    except Exception as compile_err:
        print0(f"torch.compile failed or not supported: {compile_err}. Proceeding without compilation.")

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

    # If warmdown_iters not provided (0), default to one quarter of total iterations
    if args.warmdown_iters == 0:
        args.warmdown_iters = max(args.num_iterations // 4, 1)

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
                    _, loss, _ = model(x_val, y_val, return_logits=False, prev_state=None) # ESN forward
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
        raw_loss = 0.0
        for micro_step in range(args.grad_accumulation_steps):
            model.require_backward_grad_sync = (
                micro_step == args.grad_accumulation_steps - 1
            )  # sync only on last micro step to avoid overhead
            # forward pass
            with ctx:
                _, ce_loss, _ = model(x, y, return_logits=False, prev_state=None)  # ESN forward
                raw_loss += ce_loss.detach()  # accumulate the *real* loss for logging
                loss = ce_loss / args.grad_accumulation_steps  # scaled for grad-accum
            # advance the dataset for the next batch
            x, y = train_loader.next_batch()
            # backward pass
            loss.backward()

        # clip before the optimizer step to avoid exploding updates
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)

        train_loss = raw_loss / args.grad_accumulation_steps  # mean over micro-steps

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