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
    def __init__(self, codebook, n_embd):
        super().__init__()
        self.codebook = codebook
        self.table_size = codebook.table_size
        self.n_embd = n_embd

        # Row logits (shared across words)
        self.to_row_logits = nn.Linear(self.n_embd, self.table_size)
        nn.init.normal_(self.to_row_logits.weight, std=0.02)
        nn.init.zeros_(self.to_row_logits.bias)

        # Shared column logits projection (LightRNN original formulation)
        # Much cheaper than per-row weight matrices and avoids Python loops.
        self.to_col_logits = nn.Linear(self.n_embd, self.table_size)
        nn.init.normal_(self.to_col_logits.weight, std=0.02)
        nn.init.zeros_(self.to_col_logits.bias)

    @torch._dynamo.disable
    def forward(self, hidden_states, target_ids=None):
        """LightRNN factored soft-max.

        Args:
            hidden_states: (B, T, n_embd)
            target_ids   : (B, T) or None
        Returns:
            • (None, loss) during training / validation
            • ((row_logits, col_logits), None) during generation (T==1).
        """
        # Shared projections → (B, T, R)
        row_logits = self.to_row_logits(hidden_states)
        col_logits = self.to_col_logits(hidden_states)

        if target_ids is not None:
            # ---------- training / validation ----------
            tgt_row, tgt_col = self.codebook.lookup(target_ids)  # (B, T)

            loss_row = F.cross_entropy(
                row_logits.reshape(-1, self.table_size),
                tgt_row.reshape(-1),
            )
            loss_col = F.cross_entropy(
                col_logits.reshape(-1, self.table_size),
                tgt_col.reshape(-1),
            )
            loss = loss_row + loss_col  # correct NLL
            return None, loss

        # ------------ inference (last-token logits) -------------
        return (row_logits[:, -1:, :], col_logits[:, -1:, :]), None 