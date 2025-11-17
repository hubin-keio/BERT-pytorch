# BERT Model Architecture: Mathematical Foundations

## Executive Summary

This document provides comprehensive mathematical descriptions of the BERT (Bidirectional Encoder Representations from Transformers) model architecture, with particular emphasis on positional encoding mechanisms and their mathematical properties. The document is tailored to the BERT-pytorch implementation for protein language modeling.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Transformer Architecture Overview](#transformer-architecture-overview)
3. [Embedding Layer](#embedding-layer)
4. [Multi-Head Attention Mechanism](#multi-head-attention-mechanism)
5. [Transformer Block](#transformer-block)
6. [Pre-training Tasks](#pre-training-tasks)
7. [Computational Complexity](#computational-complexity)
8. [Implementation Details](#implementation-details)
9. [Performance Considerations](#performance-considerations)
10. [Applications to Protein Sequences](#applications-to-protein-sequences)

---

## Introduction

BERT is a transformer-based bidirectional encoder introduced by Devlin et al. (2018) that has revolutionized natural language processing and can be applied to protein sequence analysis. Unlike traditional left-to-right language models, BERT uses masked language modeling to train a bidirectional representation of text.

### Key Innovations:

- **Bidirectionality**: Processes entire sequences at once, not left-to-right
- **Masked Language Model (MLM)**: Randomly masks tokens and predicts them from context
- **Next Sentence Prediction (NSP)**: Predicts if two sentences are consecutive
- **Pre-training + Fine-tuning**: Transfers learned representations to downstream tasks

---

## Transformer Architecture Overview

### High-Level Architecture

The BERT model consists of:

1. **Embedding Layer**: Combines token, positional, and segment embeddings
2. **Transformer Encoder**: Stack of L identical transformer blocks
3. **Task-Specific Heads**: MLM and NSP heads for pre-training

### Forward Pass

```
H₀ = Embedding(X, S)
Hᵢ = TransformerBlock_i(Hᵢ₋₁)  for i = 1, 2, ..., L

where:
  X = input token sequence
  S = segment information
  Hᵢ = hidden representation at layer i
  L = number of transformer blocks
```

---

## Embedding Layer

### 1. Token Embedding

Maps vocabulary indices to learned dense vectors:

```
eₜ = E_token[xₜ]

where:
  E_token ∈ ℝ^(|V| × d_model) = token embedding matrix
  |V| = vocabulary size
  d_model = hidden dimension (e.g., 256)
  xₜ = token index at position t
```

### 2. Positional Embedding (Sinusoidal)

#### Mathematical Formulation

For each position `pos` and dimension `i`:

```
PE₍ₚₒₛ, ₂ᵢ₎ = sin(pos / 10000^(2i/d_model))
PE₍ₚₒₛ, ₂ᵢ₊₁₎ = cos(pos / 10000^(2i/d_model))

where i ∈ {0, 1, ..., ⌊d_model/2⌋ - 1}
```

#### Why Sinusoidal Encoding?

**1. Invariance to Sequence Length**
- Unlike learnable embeddings, sinusoidal encodings allow extrapolation to longer sequences
- Periodic nature ensures smooth representation for unseen positions

**2. Distance Representation**
- Encodes relative positions naturally
- Dot product between positions depends only on their distance Δ = |pos₁ - pos₂|
- Allows attention mechanism to learn position-relative patterns

**3. Unique Position Representation**
- Each position has a unique encoding
- Wavelengths increase geometrically: λₖ = 2π · 10000^(2k/d_model)

**4. Linear Transformation Property**
- Positional encoding for position (pos + Δ) can be expressed as linear transformation:
  ```
  PE₍ₚₒₛ₊Δ₎ = M(Δ) · PE₍ₚₒₛ₎
  ```
  where M(Δ) is a rotation matrix depending only on distance

#### Comparison with Alternatives

**Learnable Embeddings (NOT used):**
- Cannot extrapolate to longer sequences
- Requires position-specific parameters
- Less interpretable for relative positions
- Computationally expensive

**Sinusoidal vs Other Periodic Functions:**
- Smooth and differentiable ✓
- Natural frequency spectrum ✓
- Efficient computation (no lookup tables) ✓
- Well-suited for attention mechanisms ✓

### 3. Segment Embedding

Distinguishes between two input sequences:

```
eˢᵉᵍₜ = E_segment[sₜ]

where:
  E_segment ∈ ℝ^(3 × d_model)
  sₜ ∈ {0, 1, 2} (pad, segment A, segment B)
```

### 4. Combined Embedding

Final embedding at position t:

```
zₜ = eₜ + PEₜ + eˢᵉᵍₜ

Followed by:
x₀⁽ᵗ⁾ = Dropout(LayerNorm(zₜ))
```

---

## Multi-Head Attention Mechanism

### Scaled Dot-Product Attention

The foundation of the transformer:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

where:
  Q ∈ ℝ^(n × d_k) = query matrix
  K ∈ ℝ^(m × d_k) = key matrix
  V ∈ ℝ^(m × d_v) = value matrix
  n = sequence length (queries)
  m = sequence length (keys/values)
  d_k = key dimension
```

### Scaling Factor: 1/√d_k

**Why is scaling necessary?**

Without scaling, dot products grow large with dimensionality:
- Large logits push softmax into saturation
- Vanishing gradients during backpropagation
- Scaling ensures: Var(QK^T / √d_k) ≈ 1

### Multi-Head Attention

Instead of single attention, perform h parallel operations:

```
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

where i = 1, 2, ..., h (typically h = 8)
W_i^Q, W_i^K, W_i^V ∈ ℝ^(d_model × d_k)

Output:
MultiHead(Q, K, V) = Concat(head₁, ..., head_h) W^O
where W^O ∈ ℝ^(d_model × d_model)
```

### Advantages of Multi-Head Attention

1. **Multiple Representation Subspaces**: Different heads attend to different feature subsets
2. **Reduced Dimensionality**: d_k = d_model/h allows cheaper computation
3. **Diverse Context**: Heads capture different relationship types:
   - Adjacent positions
   - Long-range dependencies
   - Semantic relationships

### Attention Masking for Padding

Prevent attention to padding tokens:

```
Attention_masked(Q, K, V) = softmax(QK^T / √d_k + M) V

where mask M:
  M_ij = 0    if position j is valid
  M_ij = -∞   if position j is padding

Effect: Attention weights for padding become 0 after softmax
```

**Implementation in BERT-pytorch:**

```python
mask = (x > 0).unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
scores = scores.masked_fill(mask == 0, float('-inf'))
```

---

## Transformer Block

### Architecture

Each transformer block has two sub-layers with residual connections:

**Layer 1: Multi-Head Attention**
```
z⁽ˡ⁾ = x⁽ˡ⁾ + MultiHeadAttn(LayerNorm(x⁽ˡ⁾))
```

**Layer 2: Feed-Forward Network**
```
x⁽ˡ⁺¹⁾ = z⁽ˡ⁾ + FFN(LayerNorm(z⁽ˡ⁾))
```

### Feed-Forward Network

Two linear transformations with non-linearity:

```
FFN(x) = W₂(σ(W₁x + b₁)) + b₂

where:
  W₁ ∈ ℝ^(d_model × d_ff), d_ff = 4 · d_model
  W₂ ∈ ℝ^(d_ff × d_model)
  σ = GELU activation
```

### GELU Activation

GELU (Gaussian Error Linear Unit):

```
GELU(x) = x · Φ(x)

where Φ(x) is the CDF of standard normal distribution.

Approximation used:
GELU(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

### Layer Normalization

Stabilizes training by normalizing across feature dimension:

```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β

where:
  μ = mean across features
  σ² = variance across features
  γ, β = learnable affine parameters
  ε = 1e-6 (numerical stability)
```

### Residual Connections

Enable deeper networks by allowing gradient flow:

```
y = x + Sublayer(x)

Gradient flow:
∂L/∂x⁽ˡ⁾ = ∂L/∂x⁽ˡ⁺¹⁾ (I + ∂Sublayer/∂x⁽ˡ⁾)

Identity term ensures direct gradient propagation
```

---

## Pre-training Tasks

### Masked Language Model (MLM)

**Objective**: Predict masked tokens from context

```
L_MLM = -Σ log P(xₜ | x₋ₜ)  for t ∈ M

where M = set of masked positions
```

**Masking Strategy** (15% of tokens):
- 80%: Replace with [MASK] token
- 10%: Replace with random token
- 10%: Keep unchanged

**Why mixed strategy?**
- Prevents model from learning to recognize [MASK]
- Forces robust contextual understanding
- Balances memorization and generalization

**MLM Loss Computation**:

```
P(xₜ | hₜ) = softmax(W_mlm · hₜ + b_mlm)

L_MLM = -Σ log P(xₜ^true | hₜ)  for t ∈ M

where:
  hₜ = hidden representation at position t
  W_mlm ∈ ℝ^(|V| × d_model)
```

### Next Sentence Prediction (NSP)

**Objective**: Determine if two sentences are consecutive

```
L_NSP = -log P(IsNext | h₀)

where h₀ = [CLS] token representation
```

**NSP Head**:

```
P(IsNext | h₀) = softmax(W_nsp · h₀ + b_nsp)

where W_nsp ∈ ℝ^(2 × d_model)
```

**Training Data**:
- 50%: Two consecutive sentences (label = 1)
- 50%: Two random sentences (label = 0)

### Joint Training Objective

```
L_total = L_MLM + L_NSP

Both losses contribute equally to optimization
```

---

## Computational Complexity

### Time Complexity Analysis

```
O(Complexity) = O(L · n · (d_model² + n · d_model))

where:
  L = number of transformer blocks
  n = sequence length
  d_model = hidden dimension
```

**Breakdown**:
- O(L · n · d_model²): FFN layers (dominates for large d_model)
- O(L · n²· d_model): Attention mechanism (dominates for long sequences)

### Memory Complexity

```
O(Memory) = O(L · n · d_model + n²)

where:
  L · n · d_model = storing hidden states
  n² = storing attention weights
```

**For sequence length 512, hidden=256**:
- Hidden states: 512 × 256 = 131K values per layer
- Attention weights: 512² = 262K values
- Total: ~393K values per layer

---

## Implementation Details

### Key Hyperparameters

| Parameter | Protein Default | BERT-Base | BERT-Large |
|-----------|-----------------|-----------|-----------|
| Hidden Size (d_model) | 256 | 768 | 1024 |
| Layers (L) | 8 | 12 | 24 |
| Attention Heads (h) | 8 | 12 | 16 |
| Feed-Forward (d_ff) | 1024 | 3072 | 4096 |
| Dropout Rate | 0.1 | 0.1 | 0.1 |
| Max Sequence | 512 | 512 | 512 |
| Vocab Size | 26 (amino acids) | 30,522 | 30,522 |

### Positional Encoding Implementation

```python
# Compute sinusoidal positional encodings
max_len = 2048
d_model = 256
pe = torch.zeros(max_len, d_model)

position = torch.arange(0, max_len).unsqueeze(1).float()
div_term = torch.exp(
    torch.arange(0, d_model, 2).float() *
    -(math.log(10000.0) / d_model)
)

# Even indices: sine
pe[:, 0::2] = torch.sin(position * div_term)

# Odd indices: cosine
pe[:, 1::2] = torch.cos(position * div_term)

# Register as non-trainable buffer
self.register_buffer('pe', pe.unsqueeze(0))
```

### Forward Pass Pseudocode

```python
def forward(self, x, segment_info):
    # Create attention mask for padding
    mask = (x > 0).unsqueeze(1).unsqueeze(1)

    # Embedding layer
    x = self.embedding(x, segment_info)

    # Stack of transformer blocks
    for transformer in self.transformer_blocks:
        x = transformer.forward(x, mask)

    return x
```

---

## Mathematical Properties and Advantages

### Why Sinusoidal Positional Encoding?

#### Property 1: Distance Sensitivity

The distance between two positions can be computed as:

```
d(i,j) = ||PE_i - PE_j||²_2

This distance:
- Is symmetric: d(i,j) = d(j,i)
- Depends only on relative distance: Δ = |i - j|
- Allows learning position-relative patterns
```

#### Property 2: Sequence Length Extrapolation

```
lim(pos→∞) PE_pos = periodic with bounded magnitude

Max magnitude of PE = √(d_model)

This allows extrapolation to arbitrary lengths
```

#### Property 3: Differentiability

Both sine and cosine are infinitely differentiable:

```
∂PE₍ₚₒₛ, ₂ᵢ₎/∂pos = (2i/10000^(2i/d_model)) · cos(...)

Enables smooth gradient flow during training
```

### Bidirectionality Advantage

Unlike RNNs or autoregressive models, BERT processes entire sequences bidirectionally:

```
hₜ⁽ˡ⁾ = f(x₁, ..., xₙ)  for all t

Each token's representation uses:
- Left context: x₁, ..., xₜ₋₁
- Token itself: xₜ
- Right context: xₜ₊₁, ..., xₙ

Result: Richer contextual understanding
```

---

## Performance Considerations

### Numerical Stability in Softmax

```
stable_softmax(z) = exp(z - max(z)) / Σ exp(z - max(z))

Subtracting max prevents overflow/underflow
```

### Scaling Attention Scores

Prevents attention logits from having inappropriate variance:

```
Attention Score Variance Analysis:
- Without scaling: Var(QK^T) = d_k
- With scaling: Var(QK^T / √d_k) = 1

Ensures stable softmax and good gradient flow
```

### Gradient Flow Through Residuals

Residual connections preserve gradient signal:

```
∂L/∂x⁽ˡ⁾ = ∂L/∂x⁽ˡ⁺¹⁾ · (I + ∂Sublayer/∂x⁽ˡ⁾)

Identity matrix term:
- Direct gradient path through layers
- Mitigates vanishing gradient problem
- Enables training of deep networks (L=100+)
```

---

## Applications to Protein Sequences

BERT extends naturally to protein sequences with minimal modifications:

### Adaptations for Proteins

| Aspect | Text BERT | Protein BERT |
|--------|-----------|-------------|
| **Vocabulary** | ~30K tokens | 26 amino acids |
| **Sequence Length** | Up to 512 | Up to 2048 |
| **Tokenization** | Subword units | Individual amino acids |
| **MLM Task** | Predict masked words | Predict masked residues |
| **NSP Task** | Consecutive sentences | Protein fragments/regions |

### Why BERT Works for Proteins

1. **Contextual Understanding**: Amino acid properties depend on surrounding residues
2. **Bidirectionality**: Proteins fold based on all residues, not just left-to-right
3. **Long-Range Dependencies**: Secondary structure involves distant residues
4. **Transfer Learning**: Pre-trained weights transfer to downstream tasks (classification, binding prediction, etc.)

---

## Conclusion

The BERT architecture combines several elegant mathematical principles:

1. **Sinusoidal Positional Encoding**
   - Provides position information without learnable parameters
   - Enables extrapolation to longer sequences
   - Captures relative distances naturally

2. **Multi-Head Attention**
   - Allows parallel attention operations
   - Multiple representation subspaces
   - Efficient computation through dimension reduction

3. **Residual Connections**
   - Facilitate training of deep networks
   - Enable direct gradient flow
   - Allow feature reuse across layers

4. **Bidirectional Processing**
   - Captures rich contextual information
   - Both left and right context for each token
   - Superior to unidirectional approaches

5. **Joint Pre-training Objectives**
   - MLM + NSP enable unsupervised learning
   - Robust representations applicable to many tasks
   - Strong transfer learning capabilities

These design choices make BERT effective for both natural language and protein sequence modeling, achieving state-of-the-art results across diverse downstream tasks.

---

## References

1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805.

2. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. NeurIPS.

3. Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). arXiv:1606.08415.

4. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. arXiv:1607.06450.

5. Hochreiter, S., Bengio, Y., Frasconi, P., & Schmidhuber, J. (2001). Gradient Flow in Recurrent Nets: The Difficulty of Learning Long-Term Dependencies. In A Field Guide to Dynamical Recurrent Networks.
