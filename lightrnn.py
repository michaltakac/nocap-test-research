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
        nn.init.normal_(self.row_embed.weight, mean=0.0, std=self.n_embd ** -0.5)
        nn.init.normal_(self.col_embed.weight, mean=0.0, std=self.n_embd ** -0.5)

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

        # Row-conditioned column projection: per-row weight + bias
        # weight: (R, d, R) , bias: (R, R)
        self.col_weight = nn.Parameter(torch.empty(self.table_size, self.n_embd, self.table_size))
        self.col_bias = nn.Parameter(torch.zeros(self.table_size, self.table_size))
        nn.init.normal_(self.col_weight, std=0.02)

    def forward(self, hidden_states, target_ids=None):
        # hidden_states: (batch_size, sequence_length, n_embd)
        # target_ids: (batch_size, sequence_length) - ground truth token ids

        # Row logits for all tokens (B,T,R)
        row_logits = self.to_row_logits(hidden_states)

        if target_ids is not None:
            # Training / validation: compute LightRNN factored CE loss.
            # No need to materialise full-vocab logits, which saves both compute
            # and memory.  Two small matmuls + two CE calls are enough.

            target_row_ids, target_col_ids = self.codebook.lookup(target_ids)

            # Memory-efficient row-conditioned column CE.
            hs_flat = hidden_states.reshape(-1, self.n_embd)               # (N,d)
            row_idx_flat = target_row_ids.reshape(-1)                       # (N,)
            col_idx_flat = target_col_ids.reshape(-1)

            total_n = hs_flat.size(0)
            loss_col_accum = 0.0
            for r in range(self.table_size):
                mask = row_idx_flat == r
                if mask.any():
                    hs_r = hs_flat[mask]                                    # (m,d)
                    logits_r = torch.addmm(self.col_bias[r], hs_r, self.col_weight[r])  # (m,R)
                    loss_col_accum += F.cross_entropy(logits_r, col_idx_flat[mask], reduction='sum')

            loss_col = loss_col_accum / total_n

            loss_row = F.cross_entropy(
                row_logits.reshape(-1, self.table_size),
                target_row_ids.reshape(-1),
            )

            loss = loss_row + loss_col

            # For training/validation we don't need logits, return None.
            return None, loss

        else: # Inference / generation (targets is None)
            # We are only interested in the last time step for generation.
            last_hidden_state = hidden_states[:, [-1], :] # (B, 1, n_embd)
            
            # Predict row and column logits for the last step
            last_row_logits = self.to_row_logits(last_hidden_state)  # (B,1,R)
            # Choose most probable row for each batch element then compute its column logits
            pred_row = last_row_logits.argmax(-1).squeeze(-1)        # (B,1)->(B,)
            w_r = self.col_weight[pred_row]  # (B,d,R)
            b_r = self.col_bias[pred_row]    # (B,R)
            col_logits_last = torch.baddbmm(b_r.unsqueeze(1), last_hidden_state, w_r).squeeze(1)  # (B,R)
            
            # For generation, the caller needs both row and column distributions.
            # We can return them packed, or the raw logits.
            # Returning raw logits for rows and columns separately seems cleaner.
            # Let's conform to the expectation of "logits" being (B, T, Vocab-like_dim)
            # We can't directly produce Vocab_size logits cheaply.
            # The original GPT returns lm_head(x[:, [-1], :]) which is (B, 1, Vocab_size)
            # For LightRNN, we'll return a tuple perhaps? Or a dict?
            # Or, for now, just row_logits, and the generation logic has to be LightRNN-aware.
            # The paper (Sec 4.1) describes generation:
            # P(w|h) = P_row(r_w|h) * P_col(c_w|h, r_w)
            # To find argmax_w P(w|h) efficiently:
            # 1. Calculate P_row(r|h) for all r.
            # 2. For each r, calculate P_col(c|h,r) for all c.
            # 3. For each word w', look up r_w', c_w'. P(w'|h) = P_row(r_w'|h)P_col(c_w'|h,r_w').
            # This is still effectively iterating over V words unless some approximation is used.
            
            # A common approximation for generation:
            # 1. Find top-k rows: r_top_k based on P_row(r|h).
            # 2. For each r in r_top_k, find top-k cols: c_top_k based on P_col(c|h,r).
            # 3. Combine these (r,c) pairs to form candidate words and pick the best.

            # If the CrossEntropyLoss is calculated outside, it needs full vocab logits.
            # The current `train_lightrnn.py` structure implies internal loss calculation for training,
            # and potentially external for validation if `return_logits=True`.
            # Let's return row_logits as the primary "logits" output, and loss=None.
            # The training loop will use the internally computed loss.
            # The validation loop (if return_logits=False) will use the internal loss.
            # Generation code will need to be LightRNN-specific.
            
            # For inference (logits for the last token only):
            # Return concatenated row and column logits as a stand-in for full vocab logits.
            # This is not ideal but fits the (B, 1, Dim) shape. Generation logic needs to handle this.
            # Or, make GPT.forward more flexible.
            # For now, let's make it clear these are not vocab_size logits.
            # The `lm_head` in original GPT produces (B, T, vocab_size).
            # We will return concatenated row and col logits for the last token.
            # This means the "vocab_size" dim will be 2 * table_size.
            # This is a placeholder and generation logic must be LightRNN-aware.
            # A better solution is to have specific generation methods.
            
            # The simplest for now for the inference path: return row_logits for the last token.
            # This makes the output shape (B, 1, table_size), not vocab_size.
            # This will break downstream if it expects vocab_size.
            # The task is focused on training speed to a validation loss target.
            # The validation loss is computed internally by LightRNNDecoder.
            
            # Let's return (row_logits, col_logits) for the last token for generation.
            # This requires changing the GPT class's forward method's return signature for this case.
            # For now, to keep signature, we can only return one tensor.
            # The paper's main contribution is training efficiency.
            # We will return `last_row_logits` as a placeholder.
            # Actual generation would need a dedicated method in the `GPT` class.
            
            # Return both row & column logits for downstream selection.
            return (last_row_logits, col_logits_last.unsqueeze(1)), None 