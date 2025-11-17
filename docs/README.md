# BERT-pytorch Documentation

This directory contains comprehensive documentation for the BERT-pytorch protein language model.

## Files

### 0. LATEX_TROUBLESHOOTING.md

**If you encounter LaTeX compilation errors, read this first!**
- Solutions for UTF-8 encoding errors
- Multiple compilation options
- How to use the simplified LaTeX file
- Overleaf online option
- Markdown alternative (no compilation)

### 1. BERT_ARCHITECTURE.md

Complete mathematical description of the BERT model architecture with detailed explanations of:

- **Transformer Architecture Overview**: High-level structure and forward pass
- **Embedding Layer**: Token, positional, and segment embeddings
- **Positional Encoding (Sinusoidal)**:
  - Mathematical formulation
  - Why sinusoidal encoding is chosen
  - Properties and advantages
  - Comparison with alternatives
- **Multi-Head Attention**: Scaled dot-product attention, scaling factors, multi-head benefits
- **Transformer Blocks**: Feed-forward networks, layer normalization, residual connections
- **Pre-training Tasks**: MLM (Masked Language Model) and NSP (Next Sentence Prediction)
- **Computational Complexity**: Time and memory analysis
- **Implementation Details**: Hyperparameters, code examples, forward pass
- **Performance Considerations**: Numerical stability, gradient flow
- **Protein Sequence Applications**: How BERT adapts to protein modeling

### 2. BERT_Architecture.tex

Complete LaTeX source document that can be compiled to PDF. The document includes:

- Professional mathematical typesetting
- Comprehensive table of contents
- Detailed equations with proper numbering
- Code listings for Python implementation
- Multiple sections with subsections
- Bibliography with key references
- Mathematical notation reference appendix

#### Compiling to PDF

**IMPORTANT: If you get "UTF-8.def not found" error, see LATEX_TROUBLESHOOTING.md**

**Option 1: Use Simplified LaTeX (No Dependencies)**
```bash
cd docs
pdflatex BERT_Architecture_Simple.tex
```

**Option 2: Use Original Full-Featured Version**
```bash
cd docs
pdflatex BERT_Architecture.tex
```

**Installation (if needed):**
```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-base texlive-fonts-recommended texlive-latex-extra

# macOS (with Homebrew)
brew install basictex
```

**Option 3: Use Overleaf Online (No Installation)**
1. Visit https://www.overleaf.com/
2. Create → "Upload Project"
3. Upload `docs/BERT_Architecture.tex`
4. Automatic PDF generation

**Output:**
The compilation generates:
- `BERT_Architecture.pdf` - The final document
- `BERT_Architecture.aux` - Auxiliary file
- `BERT_Architecture.log` - Compilation log
- `BERT_Architecture.toc` - Table of contents data

#### Windows

Use a TeX distribution such as:
- [MiKTeX](https://miktex.org/)
- [TeX Live](https://www.tug.org/texlive/)
- [Overleaf](https://www.overleaf.com/) (online, no installation needed)

Then compile using `pdflatex` command.

---

## Document Structure

Both documents follow the same logical structure:

```
1. Introduction
2. Transformer Architecture Overview
3. Embedding Layer
   - Token Embedding
   - Positional Embedding (detailed mathematical treatment)
   - Segment Embedding
   - Combined Embedding
4. Multi-Head Attention Mechanism
   - Scaled Dot-Product Attention
   - Scaling Factor Analysis
   - Multi-Head Attention
   - Attention Masking
5. Transformer Block
   - Architecture
   - Feed-Forward Network
   - GELU Activation
   - Layer Normalization
   - Residual Connections
6. Pre-training Tasks
   - Masked Language Model (MLM)
   - Next Sentence Prediction (NSP)
   - Joint Training Objective
7. Computational Complexity
   - Time Complexity
   - Memory Complexity
8. Implementation Details
   - Key Hyperparameters
   - Positional Encoding Implementation
   - Forward Pass Pseudocode
9. Mathematical Properties
   - Why Sinusoidal Encoding
   - Bidirectionality Advantages
10. Performance Considerations
    - Numerical Stability
    - Gradient Flow
11. Protein Sequence Applications
12. Conclusion
13. References
```

---

## Key Topics Covered

### Positional Encoding (Emphasized)

The documents provide in-depth mathematical treatment of positional encodings:

**Why Sinusoidal?**
- ✓ Invariance to sequence length (extrapolation capability)
- ✓ Distance representation (relative position learning)
- ✓ Unique position representation
- ✓ Linear transformation property (rotation matrices)

**Mathematical Properties:**
- Equations for even/odd dimensions with sine/cosine
- Distance sensitivity analysis
- Extrapolation guarantees
- Differentiability for gradient flow
- Comparison with learnable embeddings and other periodic functions

**Implementation:**
- Complete Python code for computing encodings
- Wavelength analysis (λₖ = 2π · 10000^(2k/d_model))
- Registration as non-trainable buffer in PyTorch

### Attention Mechanism

Detailed coverage of:
- Scaled dot-product attention formula
- Scaling factor (1/√d_k) necessity and effects
- Multi-head parallel attention
- Attention mask creation and application
- Properties of multi-head attention

### Transformer Architecture

Mathematical formulations for:
- Feed-forward networks with GELU activation
- Layer normalization with learnable parameters
- Residual connections and gradient flow
- Complete transformer block architecture

### Pre-training Objectives

Comprehensive treatment of:
- MLM loss computation and masking strategy (80-10-10)
- NSP loss and head architecture
- Joint training with equal loss weights
- Why mixed masking helps learning

---

## Mathematical Notation

Key mathematical symbols used throughout:

| Symbol | Meaning |
|--------|---------|
| d_model | Model hidden dimension |
| d_k, d_v | Key/value dimensions per head |
| h | Number of attention heads |
| L | Number of transformer blocks |
| n, m | Sequence lengths |
| Q, K, V | Query, Key, Value matrices |
| PE | Positional encoding |
| W | Weight matrix |
| σ | Activation function (GELU) |
| ℝ | Real numbers |
| ⊙ | Element-wise multiplication |

---

## Code Examples

Both documents include practical Python code examples:

```python
# Positional encoding computation
position = torch.arange(0, max_len).unsqueeze(1).float()
div_term = torch.exp(
    torch.arange(0, d_model, 2).float() *
    -(math.log(10000.0) / d_model)
)
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

# Attention mask creation
mask = (x > 0).unsqueeze(1).unsqueeze(1)
scores = scores.masked_fill(mask == 0, float('-inf'))
```

---

## References

All documents include comprehensive references to:

1. Devlin et al. (2018) - BERT paper
2. Vaswani et al. (2017) - Attention Is All You Need
3. Hendrycks & Gimpel (2016) - GELU activation
4. Ba et al. (2016) - Layer Normalization
5. Hochreiter et al. (2001) - Gradient flow in RNNs

---

## Viewing the Documents

### Option 1: Read Markdown (Recommended)
- View `BERT_ARCHITECTURE.md` directly in GitHub, VS Code, or any text editor
- No compilation needed
- Contains all mathematical content in readable format

### Option 2: View Compiled PDF
1. Compile `BERT_Architecture.tex` on your system (see instructions above)
2. Open the generated `BERT_Architecture.pdf`
3. Professional formatting with proper mathematical typography

### Option 3: Use Online Editor
- Upload `BERT_Architecture.tex` to [Overleaf](https://www.overleaf.com/)
- View/edit online without local installation
- Automatic PDF generation

---

## For Researchers and Students

These documents are designed for:

✓ **Understanding BERT internals**: Complete mathematical foundations
✓ **Implementation**: Code examples aligned with BERT-pytorch source
✓ **Research**: Detailed analysis of design choices and mathematical properties
✓ **Teaching**: Comprehensive reference material for courses
✓ **Protein modeling**: Application-specific guidance

---

## Questions or Issues?

If you find errors, have questions about the mathematical content, or want clarifications:

1. Check the [main README](../README.md) for general information
2. Review the source code in `bert_pytorch/` for implementation details
3. Refer to original papers in References section
4. Check the example scripts in `examples/` for practical usage

---

## License

These documentation files are part of BERT-pytorch and are licensed under the Apache 2.0 License, same as the main repository.

---

**Last Updated**: November 2025
**Version**: 1.0
**Document Scope**: BERT-pytorch protein language model (November 2025 release)
