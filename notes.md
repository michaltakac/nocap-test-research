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

> The third-generation Tensor Core design in GA10x GPUs further increases raw performance
and brings new precision modes such as TF32 and BFloat16

