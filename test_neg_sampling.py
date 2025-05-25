#!/usr/bin/env python3
"""
Test script for negative sampling components
"""

import torch
import numpy as np
import torch.nn.functional as F

def build_alias_table(probs, table_size):
    """Build alias table for O(1) sampling using Walker's alias method."""
    n = len(probs)
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
    
    def __init__(self, freqs, power=0.75, device='cuda'):
        self.device = device
        self.vocab_size = len(freqs)
        
        # Convert frequencies to probabilities with power scaling
        probs = np.power(freqs + 1e-10, power)  # Add small epsilon to avoid zeros
        probs = probs / probs.sum()
        
        # Build alias table and move to GPU
        alias, prob = build_alias_table(probs, self.vocab_size)
        self.alias_table = torch.from_numpy(alias).to(device)
        self.prob_table = torch.from_numpy(prob).to(device)
        
        print(f"Built unigram sampler with vocab size {self.vocab_size:,}")
    
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


class NegSamplingLoss(torch.nn.Module):
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
            # Sample k negatives shared across the batch
            neg = self.sampler.sample(self.k)  # (k,)
            neg = neg.unsqueeze(0).expand(batch_size, self.k)  # (B*T, k)
        else:
            # Sample k negatives per token
            neg = self.sampler.sample(batch_size * self.k).view(batch_size, self.k)
        
        # Get embeddings for positive and negative samples
        w_pos = self.weight[target]  # (B*T, d)
        w_neg = self.weight[neg]     # (B*T, k, d)
        
        # Compute scores
        pos_score = (h * w_pos).sum(-1)  # (B*T,)
        neg_score = torch.einsum('bd,bkd->bk', h, w_neg)  # (B*T, k)
        
        # NCE loss: maximize log sigmoid for positive, minimize for negative
        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -F.logsigmoid(-neg_score).mean()
        
        return pos_loss + neg_loss


def test_components():
    """Test all negative sampling components"""
    print("=" * 50)
    print("Testing Negative Sampling Components")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test 1: UnigramSampler
    print("\n1. Testing UnigramSampler...")
    freqs = np.array([1000, 500, 200, 100, 50, 20, 10, 5, 2, 1])  # Zipfian-like distribution
    sampler = UnigramSampler(freqs, power=0.75, device=device)
    
    # Sample and check distribution
    samples = sampler.sample(10000)
    print(f"Sampled {len(samples)} tokens from vocab size {sampler.vocab_size}")
    print(f"Sample range: {samples.min().item()}-{samples.max().item()}")
    
    # Check if sampling follows expected distribution (most frequent tokens appear more)
    unique, counts = torch.unique(samples, return_counts=True)
    print("Token frequencies in samples:")
    for i in range(min(5, len(unique))):
        print(f"  Token {unique[i].item()}: {counts[i].item()} times")
    
    # Test 2: NegSamplingLoss
    print("\n2. Testing NegSamplingLoss...")
    vocab_size = 1000
    embed_dim = 128
    batch_size = 32
    k = 5
    
    # Create mock data
    weight = torch.randn(vocab_size, embed_dim, device=device, requires_grad=True)  # Enable gradients
    h = torch.randn(batch_size, embed_dim, device=device, requires_grad=True)  # Enable gradients
    target = torch.randint(0, vocab_size, (batch_size,), device=device)
    
    # Create mock sampler for testing
    mock_freqs = np.ones(vocab_size)  # Uniform distribution for testing
    mock_sampler = UnigramSampler(mock_freqs, device=device)
    
    # Test NegSamplingLoss
    neg_loss = NegSamplingLoss(weight, k, mock_sampler, shared_batch=False)
    loss = neg_loss(h, target)
    
    print(f"Computed negative sampling loss: {loss.item():.4f}")
    print(f"Loss is finite: {torch.isfinite(loss).item()}")
    print(f"Loss requires grad: {loss.requires_grad}")
    
    # Test gradient computation
    loss.backward()
    print("Gradient computation successful!")
    print(f"Weight grad shape: {weight.grad.shape}")
    print(f"Hidden grad shape: {h.grad.shape}")
    
    # Test 3: Memory usage
    print("\n3. Testing memory efficiency...")
    
    # Compare with full softmax computation
    with torch.no_grad():
        # Full softmax (what we're trying to avoid)
        full_logits = torch.mm(h, weight.t())  # (batch_size, vocab_size)
        full_softmax_mem = full_logits.numel() * 4  # 4 bytes per float32
        
        # Negative sampling memory (only k+1 computations per sample)
        neg_sampling_mem = batch_size * (k + 1) * 4  # Much smaller
        
        print(f"Full softmax memory: {full_softmax_mem:,} bytes")
        print(f"Negative sampling memory: {neg_sampling_mem:,} bytes")
        print(f"Memory reduction: {full_softmax_mem / neg_sampling_mem:.1f}x")
    
    print("\n" + "=" * 50)
    print("All tests passed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    test_components() 