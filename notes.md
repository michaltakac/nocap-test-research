# Optimizatons for BottleCap AI GPT-2 Benchmark

> ### Objective
> Train a langauge model on a subset of the FineWeb dataset to reach a validation loss of ≤ 3.3821 as fast as possible using 1 GPU.
>
> You can achieve this by:
>
> making your model faster (so that it sees more data in shorter time)
making your training more efficient (so that in less steps your model makes better progress).

### What to try:

1. Mixed precison training

RTX 3090/4090 (currently using RTX 3090 at home) can do FP16 tensor core operations, and also BF16... from their "[NVIDIA Ampere GA102 GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf)" PDF, page 24:

> "The third-generation Tensor Core design in GA10x GPUs further increases raw performance and brings new precision modes such as TF32 and BFloat16 \[...\]"

2. Different attention mechanisms

FlashAttention2 or 3 would be good to implement, but I'm looking at FlashMLA (multi-latent attention), introduced by DeepSeek, which compresses the KV cache (compressing key and value matrices into latent vectors). This should allow for longer contexts. Plus, paged KV cache with a fixed block size (64) ensures efficient and predictable memory access patterns, which should minimize latency and unlock leveraging higher memory bandwidth of the GPU.

For memory-bound optimization, they recommend looking at this [diff](https://github.com/deepseek-ai/FlashMLA/tree/b31bfe72a83ea205467b3271a5845440a03ed7cb).

Although, since I'm running on RTX 3090 which is an Ampere architecture, I found this [FlashMLA implementation adapted to Ampere architectures](https://github.com/pzhao-eng/FlashMLA) by [@pzhao-eng](https://github.com/pzhao-eng). Here's a test benchmark results I ran on my GPU:

- FlashMLA Benchmark TFLOPS Per Run (Performance (TFLOPS) by Run Index)
![FlashMLA Benchmark TFLOPS Per Run](img/tflops.png)

- FlashMLA Benchmark Bandwidth Per Run (Bandwidth (GB/s) by Run Index)
![FlashMLA Benchmark Bandwidth Per Run](img/bandwidth.png)

There's further possible experimentation with:
• other pooling (max, reservoir-sampling, learned “latent tokens”);
• different score_mod for ALiBi/relative-pos biases;
• cached / paged inference via the included flash_mla_with_kvcache functions (already imported).

Note to self: Paged attention is a technique used to handle long sequences in transformer models efficiently by splitting attention computation into smaller, managable “pages” or “blocks”. This approach reduces memory comsumption and computational complexity, making it feasible to process sequences that would otherwise be too large to fit in memory [1](https://medium.com/my-musings-with-llms/understanding-kv-cache-and-paged-attention-in-llms-a-deep-dive-into-efficient-inference-62fa372432ce#:~:text=KV%20cache%20and%20paged%20attention%20are%20powerful%20techniques%20that%20make,constraints%20of%20processing%20long%20sequences.).

3. Increase sequence length

After introducing FlashAttention or FlashMLA, try increasing the context length. 

Test various context lengths on RTX 3090.

- Base: 1024
- a\) 2048
- b\) 4096
- c\) 8192
- d\) 16384

4. Use fused optimizer

By fusing the dropout, layernorm, and linear operations, it's possible to reduce kernel launch overhead.

5. Multi-token prediction auxiliary loss

Introduce a multi-token prediction objective to the training loop. For instance, in addition to the standard next-token loss, have the model also predict two steps ahead (next-next-token) using its current hidden state. Concretely, given a training sequence, you can add an auxiliary loss where at position i the model tries to predict token i+2 (perhaps by using the ground-truth i+1 as input in a second forward pass, or by a tailored two-step decoding within the model).

6. Mixture-of-Experts (MoE) layer

I'll try replacing some feed-forward layers with two or more expert variants and a gating network - this should be doable on single GPU hopefully. 

One expert can specialize in technical text, another in narrative text.

Gating network will be just a learned switchbased on token or context, routing each token through one expert per forward pass.


---

References

1. [Understanding KV Cache and Paged Attention in LLMs: A Deep Dive into Efficient Inference](https://medium.com/my-musings-with-llms/understanding-kv-cache-and-paged-attention-in-llms-a-deep-dive-into-efficient-inference-62fa372432ce#:~:text=KV%20cache%20and%20paged%20attention%20are%20powerful%20techniques%20that%20make,constraints%20of%20processing%20long%20sequences.), 2024-10-23, retrieved 2025-04-29