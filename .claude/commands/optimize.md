---
description: Analyze code for performance optimizations
---

Perform a performance analysis of the codebase:

1. **Identify bottlenecks**
   - Look for inefficient loops
   - Check for unnecessary computations
   - Find redundant operations
   - Identify memory-intensive operations

2. **PyTorch-specific optimizations**
   - Check for operations that could be vectorized
   - Look for CPU/GPU transfer inefficiencies
   - Verify proper use of torch.no_grad() for inference
   - Check DataLoader configuration (num_workers, pin_memory)

3. **Audio processing efficiency**
   - Review spectrogram computation
   - Check for redundant audio transformations
   - Optimize librosa/scipy operations

4. **Training loop optimization**
   - Verify gradient accumulation if needed
   - Check mixed precision training possibilities
   - Review checkpoint saving strategy

Provide specific code improvements with examples.
