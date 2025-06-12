import torch
import torch.nn as nn
from afno_seq import AFNO1DSeq
from tensorgrad import TensorGRaD

def test_sequence_length(seq_len, batch_size=1):
    """Test if a given sequence length fits in memory"""
    try:
        print(f"Testing seq_len={seq_len}, batch_size={batch_size}...")
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Create model
        afno = AFNO1DSeq(hidden_size=768, num_blocks=8).cuda()
        linear = nn.Linear(768, 768).cuda()
        
        # Create optimizer
        params = list(afno.parameters()) + list(linear.parameters())
        optimizer = TensorGRaD(params, lr=1e-4)
        
        # Test data
        x = torch.randn(batch_size, seq_len, 768).cuda()
        target = torch.randn(batch_size, seq_len, 768).cuda()
        
        # Forward pass
        y = afno(x)
        output = linear(y)
        loss = nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Get memory usage
        memory_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
        print(f"✓ SUCCESS: seq_len={seq_len}, memory={memory_mb} MB")
        return True, memory_mb
        
    except torch.cuda.OutOfMemoryError:
        print(f"✗ OOM: seq_len={seq_len}")
        return False, None
    except Exception as e:
        print(f"✗ ERROR: seq_len={seq_len}, error={e}")
        return False, None
    finally:
        # Cleanup
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Testing memory limits for causal AFNO...")
    
    # Binary search for maximum sequence length
    low, high = 64, 512
    max_working = 64
    
    while low <= high:
        mid = (low + high) // 2
        success, memory = test_sequence_length(mid)
        
        if success:
            max_working = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(f"\nMaximum working sequence length: {max_working}")
    
    # Test with original batch size
    if max_working >= 128:
        print(f"\nTesting with batch_size=8 (original config)...")
        test_sequence_length(max_working // 4, batch_size=8)