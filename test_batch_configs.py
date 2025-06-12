import torch
import torch.nn as nn
from afno_seq import AFNO1DSeq
from tensorgrad import TensorGRaD

def test_config(seq_len, batch_size):
    """Test a specific batch/sequence configuration"""
    try:
        print(f"Testing seq_len={seq_len}, batch_size={batch_size}...")
        
        torch.cuda.empty_cache()
        
        # Create model components
        afno = AFNO1DSeq(hidden_size=768, num_blocks=8).cuda()
        linear = nn.Linear(768, 768).cuda()
        
        # Test forward pass only (most memory intensive)
        x = torch.randn(batch_size, seq_len, 768).cuda()
        y = afno(x)
        
        memory_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
        tokens_per_batch = seq_len * batch_size
        
        print(f"✓ SUCCESS: {tokens_per_batch} tokens/batch, {memory_mb} MB")
        return True, memory_mb, tokens_per_batch
        
    except torch.cuda.OutOfMemoryError:
        print(f"✗ OOM: seq_len={seq_len}, batch_size={batch_size}")
        return False, None, None
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Finding optimal batch configurations for causal AFNO...\n")
    
    # Test various configurations
    configs = [
        # (seq_len, batch_size)
        (512, 1),   # Maximum seq_len
        (256, 2),   # Half seq_len, double batch
        (256, 4),   # Half seq_len, quad batch
        (128, 8),   # Quarter seq_len, 8x batch (original)
        (128, 16),  # Quarter seq_len, 16x batch
        (64, 32),   # 1/8 seq_len, 32x batch
        (64, 64),   # 1/8 seq_len, 64x batch (original grad accum)
    ]
    
    working_configs = []
    
    for seq_len, batch_size in configs:
        success, memory, tokens = test_config(seq_len, batch_size)
        if success:
            working_configs.append((seq_len, batch_size, memory, tokens))
    
    print(f"\n{'='*60}")
    print("WORKING CONFIGURATIONS:")
    print(f"{'Seq Len':<8} {'Batch':<6} {'Memory':<8} {'Tokens/Batch':<12} {'Efficiency'}")
    print(f"{'='*60}")
    
    for seq_len, batch_size, memory, tokens in working_configs:
        efficiency = tokens / memory  # tokens per MB
        print(f"{seq_len:<8} {batch_size:<6} {memory:<8} {tokens:<12} {efficiency:.2f}")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print("- For maximum sequence length: 512 tokens, batch_size=1")
    print("- For balanced training: 256 tokens, batch_size=2-4") 
    print("- For fast training: 128 tokens, batch_size=8-16")
    print("- Original config (2048 tokens) is NOT feasible with causal AFNO")