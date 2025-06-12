import torch
import torch.nn as nn
import torch.nn.functional as F
from afno_seq import AFNO1DSeq
from tensorgrad import TensorGRaD

# Simple test to verify AFNO training works
def test_afno_training():
    print("Testing AFNO1DSeq training...")
    
    # Small model for testing
    hidden_size = 768
    seq_len = 64
    batch_size = 2
    
    # Create AFNO layer
    afno = AFNO1DSeq(hidden_size=hidden_size, num_blocks=8)
    afno = afno.cuda()
    
    # Create simple loss function
    linear = nn.Linear(hidden_size, hidden_size).cuda()
    
    # Create optimizer
    params = list(afno.parameters()) + list(linear.parameters())
    optimizer = TensorGRaD(params, lr=1e-4)
    
    # Test data
    x = torch.randn(batch_size, seq_len, hidden_size).cuda()
    target = torch.randn(batch_size, seq_len, hidden_size).cuda()
    
    print("Forward pass...")
    y = afno(x)
    output = linear(y)
    
    print("Computing loss...")
    loss = F.mse_loss(output, target)
    print(f"Loss: {loss.item():.6f}")
    
    print("Backward pass...")
    loss.backward()
    
    print("Optimizer step...")
    optimizer.step()
    optimizer.zero_grad()
    
    print("✓ Training test passed!")
    return True

if __name__ == "__main__":
    test_afno_training()