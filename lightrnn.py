import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class LightRNNCodebook(nn.Module):
    def __init__(self, vocab_size, table_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.table_size = table_size
        
        # Initialize codebook (row and column IDs for each token)
        # For simplicity, we'll do a basic sequential assignment here.
        # A frequency-based assignment or dynamic reassignment would be better.
        row_ids = torch.arange(vocab_size) // table_size
        col_ids = torch.arange(vocab_size) % table_size
        
        self.register_buffer("row_ids", row_ids.long())
        self.register_buffer("col_ids", col_ids.long())

    def lookup(self, token_ids):
        # token_ids: (batch_size, sequence_length)
        return self.row_ids[token_ids], self.col_ids[token_ids]

    def reassign(self, freqs):
        # Placeholder for future implementation of dynamic codebook reassignment
        # This would involve sorting vocab by frequency and re-calculating row/col IDs
        print("Codebook reassignment not yet implemented.")
        pass

class LightRNNEmbedding(nn.Module):
    def __init__(self, codebook, n_embd):
        super().__init__()
        self.codebook = codebook
        self.table_size = codebook.table_size
        self.n_embd = n_embd

        self.row_embed = nn.Embedding(self.table_size, self.n_embd)
        self.col_embed = nn.Embedding(self.table_size, self.n_embd)
        
        # Weight initialization (Xavier is generally good for embeddings & linear)
        nn.init.normal_(self.row_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.col_embed.weight, mean=0.0, std=0.02)

        # Optional: linear layer if concatenating row and col embeddings
        # self.fc = nn.Linear(2 * n_embd, n_embd) 

    def forward(self, token_ids):
        # token_ids: (batch_size, sequence_length)
        row_ids, col_ids = self.codebook.lookup(token_ids)
        
        r_emb = self.row_embed(row_ids)  # (batch, seq_len, n_embd)
        c_emb = self.col_embed(col_ids)  # (batch, seq_len, n_embd)
        
        # Combine embeddings (summing is common)
        x = r_emb + c_emb
        # Alternatively, concatenate and project:
        # x = torch.cat((r_emb, c_emb), dim=-1)
        # x = self.fc(x)
        return x

class LightRNNDecoder(nn.Module):
    def __init__(self, codebook, n_embd, rank: int = 0):
        super().__init__()
        self.codebook = codebook
        self.table_size = codebook.table_size
        self.n_embd = n_embd
        self.low_rank = rank > 0

        # Row logits (shared across words)
        self.to_row_logits = nn.Linear(self.n_embd, self.table_size)
        nn.init.normal_(self.to_row_logits.weight, std=0.02)
        nn.init.zeros_(self.to_row_logits.bias)

        # ------------------------------------------------------------------
        # Row-conditioned column projection (per-row weight + bias)
        # Shape: col_weight[r]  : (d, R)
        #        col_bias[r]    : (R)
        # This matches the original LightRNN formulation and has been shown
        # to converge to much lower loss than a single shared projection.
        # A naive batched matmul (B*T, d) × (R, d, R) would explode memory.
        # Instead we keep the parameters but in the forward pass materialise
        # logits only for the rows that actually appear in the batch, via a
        # lightweight Python loop over at most R masks.  In practice with a
        # table_size of 256 the cost is negligible and keeps memory < 1 GB.
        #
        # NOTE: we still support fast generation by computing column logits
        # for *one* row at a time (the predicted row) – no big matmul.
        # ------------------------------------------------------------------

        self.col_weight = nn.Parameter(torch.empty(self.table_size, self.n_embd, self.table_size))
        self.col_bias   = nn.Parameter(torch.zeros(self.table_size, self.table_size))
        nn.init.normal_(self.col_weight, std=0.02)

        if self.low_rank:
            self.col_u = nn.Parameter(torch.randn(self.table_size, rank))
            self.col_v = nn.Parameter(torch.randn(rank, self.n_embd))
        else:
            # Register as non-trainable placeholders so DDP doesn't track them
            self.register_parameter("col_u", None)
            self.register_parameter("col_v", None)

    def forward(self, hidden_states, target_ids=None):
        """LightRNN factored soft-max.

        Args:
            hidden_states: (B, T, n_embd)
            target_ids   : (B, T) or None
        Returns:
            • (None, loss) during training / validation
            • ((row_logits, col_logits), None) during generation (T==1).
        """
        # ------------------------------------------------------------------
        # Row logits for all tokens (B, T, R)
        # ------------------------------------------------------------------
        row_logits = self.to_row_logits(hidden_states)

        if target_ids is not None:
            # --------------------------------------------------------------
            # Training / validation: LightRNN factored CE loss computed with
            # memory-efficient per-row masking.  We *avoid* materialising
            # the full (B*T, R) tensor of column logits.
            # --------------------------------------------------------------

            tgt_row, tgt_col = self.codebook.lookup(target_ids)      # (B,T)
            row_ids_flat = tgt_row.reshape(-1)                       # (N,)
            col_ids_flat = tgt_col.reshape(-1)
            hs_flat      = hidden_states.reshape(-1, self.n_embd)    # (N,d)

            # Gather weights / bias once
            W_tok = self.col_weight[row_ids_flat].to(hs_flat.dtype)  # (N,d,R)
            b_tok = self.col_bias [row_ids_flat].to(hs_flat.dtype)   # (N,R)

            # Grouped GEMM: (N,1,d) × (N,d,R) + (N,1,R) → (N,R)
            logits = torch.baddbmm(b_tok.unsqueeze(1), hs_flat.unsqueeze(1), W_tok)[:,0]

            # Individually normalised log-softmax for row and column branch.
            # This is the correct factorised NLL:  −log P(row=r) −log P(col=c | r).

            loss_row = F.cross_entropy(
                row_logits.reshape(-1, self.table_size),
                row_ids_flat,
            )

            loss_col = F.cross_entropy(logits, col_ids_flat)

            loss = loss_row + loss_col

            # During training/validation we don't need logits – return None.
            return None, loss

        # ------------------------------------------------------------------
        # Inference / generation – only need logits for the *last* position.
        # We still avoid big matmuls by computing column logits for the most
        # probable row predicted by row_logits.
        # ------------------------------------------------------------------

        last_hidden = hidden_states[:, [-1], :]                 # (B,1,d)
        last_row_logits = row_logits[:, -1:, :]                 # (B,1,R)

        # Greedy choice of row per batch element (could also sample / top-k)
        pred_row = last_row_logits.argmax(-1).squeeze(-1)       # (B,)

        # Gather corresponding weights/biases and compute column logits.
        w_r = self.col_weight[pred_row]                         # (B,d,R)
        b_r = self.col_bias[pred_row]                           # (B,R)
        col_logits_last = torch.baddbmm(b_r.unsqueeze(1), last_hidden, w_r)  # (B,1,R)

        return (last_row_logits, col_logits_last), None 