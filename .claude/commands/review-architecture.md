---
description: Review neural network architecture implementation
---

Perform an architectural review of the neural network components:

1. **Examine model architecture files** in `src/traincw/models/`
   - Check CNN encoder structure
   - Review LSTM/RNN layers
   - Validate CTC loss implementation
   - Verify input/output dimensions

2. **Assess design alignment** with DESIGN.md specifications
   - Compare implemented architecture with planned design
   - Check if spectrogram processing matches specs
   - Verify sequence-to-sequence approach

3. **Performance considerations**
   - Identify potential bottlenecks
   - Check for unnecessary computations
   - Verify batch processing efficiency
   - Assess memory usage patterns

4. **Code quality**
   - Check for proper error handling
   - Verify type hints and documentation
   - Look for potential bugs or edge cases

Provide specific recommendations for improvements.
