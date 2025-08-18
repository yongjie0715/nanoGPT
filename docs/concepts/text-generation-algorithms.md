# Text Generation Algorithms in nanoGPT

## Overview

This document provides a comprehensive explanation of the text generation algorithms implemented in nanoGPT, focusing on autoregressive generation, sampling strategies, and the mathematical foundations behind each approach. Understanding these algorithms is crucial for effectively using and modifying the text generation capabilities.

## Autoregressive Generation Process

### Fundamental Concept

Autoregressive generation is the core algorithm used by GPT models to produce text. The process generates one token at a time, using previously generated tokens as context for predicting the next token.

**Mathematical Foundation:**
```
P(x₁, x₂, ..., xₙ) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ... × P(xₙ|x₁,...,xₙ₋₁)
```

Where each conditional probability P(xᵢ|x₁,...,xᵢ₋₁) is modeled by the transformer network.

### Step-by-Step Generation Algorithm

#### Phase 1: Initialization
```python
# Input: prompt tokens [t₁, t₂, ..., tₖ]
sequence = encode(prompt)  # Convert text to token IDs
```

**Process:**
1. Convert input prompt to token sequence using appropriate tokenizer
2. Create tensor with batch dimension for model compatibility
3. Move tensor to appropriate device (GPU/CPU)

#### Phase 2: Iterative Token Generation
```python
for step in range(max_new_tokens):
    # 1. Context Window Management
    context = sequence[-block_size:] if len(sequence) > block_size else sequence
    
    # 2. Forward Pass
    logits = model(context)  # Shape: [batch_size, sequence_length, vocab_size]
    
    # 3. Next Token Prediction
    next_token_logits = logits[:, -1, :]  # Extract final position
    
    # 4. Apply Sampling Strategy
    next_token = sample(next_token_logits, temperature, top_k)
    
    # 5. Extend Sequence
    sequence = torch.cat([sequence, next_token], dim=1)
```

**Detailed Breakdown:**

##### 1. Context Window Management
```python
idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
```

**Purpose**: Ensures input doesn't exceed model's maximum context length
**Implementation**: 
- If sequence ≤ block_size: use entire sequence
- If sequence > block_size: use last block_size tokens
**Rationale**: Maintains computational efficiency while preserving recent context

##### 2. Forward Pass Through Transformer
```python
logits, _ = self(idx_cond)
```

**Process**:
1. Input tokens pass through embedding layer
2. Positional encodings added to token embeddings
3. Sequence processed through transformer blocks
4. Final layer norm and output projection to vocabulary size

**Output**: Raw logits for each vocabulary token at each position

##### 3. Next Token Logit Extraction
```python
logits = logits[:, -1, :] / temperature
```

**Focus**: Only the final position contains next-token predictions
**Shape Transformation**: [batch_size, sequence_length, vocab_size] → [batch_size, vocab_size]
**Temperature Application**: Applied immediately to control randomness

##### 4. Sampling Strategy Application
Multiple strategies can be applied to convert logits to token selection:
- Temperature scaling
- Top-k filtering  
- Top-p (nucleus) sampling
- Deterministic selection (argmax)

##### 5. Sequence Extension
```python
idx = torch.cat((idx, idx_next), dim=1)
```

**Operation**: Appends sampled token to existing sequence
**Growth Pattern**: Sequence length increases by 1 each iteration
**Memory Consideration**: Sequence grows linearly with generation length

## Temperature Scaling Algorithm

### Mathematical Foundation

Temperature scaling modifies the probability distribution before sampling:

```
P'(xᵢ) = exp(logits[i] / T) / Σⱼ exp(logits[j] / T)
```

Where T is the temperature parameter.

### Temperature Effects

#### Low Temperature (T < 1.0)
```python
# Example: T = 0.5
scaled_logits = logits / 0.5  # Amplifies differences
probs = softmax(scaled_logits)
```

**Mathematical Effect**: Amplifies logit differences, creating sharper distribution
**Behavioral Impact**:
- Higher probability mass on top tokens
- More deterministic, predictable output
- Reduced creativity and diversity
- Better for factual or structured generation

**Example Distribution:**
```
Original: [0.4, 0.3, 0.2, 0.1]
T=0.5:    [0.6, 0.25, 0.12, 0.03]  # More peaked
```

#### High Temperature (T > 1.0)
```python
# Example: T = 2.0
scaled_logits = logits / 2.0  # Reduces differences
probs = softmax(scaled_logits)
```

**Mathematical Effect**: Reduces logit differences, creating flatter distribution
**Behavioral Impact**:
- More uniform probability distribution
- Increased randomness and creativity
- Higher chance of unexpected tokens
- Better for creative or exploratory generation

**Example Distribution:**
```
Original: [0.4, 0.3, 0.2, 0.1]
T=2.0:    [0.32, 0.28, 0.24, 0.16]  # More uniform
```

#### Neutral Temperature (T = 1.0)
```python
# No scaling applied
probs = softmax(logits)
```

**Effect**: Uses model's natural probability distribution without modification

### Implementation Details

```python
def apply_temperature(logits, temperature):
    """Apply temperature scaling to logits"""
    if temperature == 0.0:
        # Deterministic selection (argmax)
        return torch.zeros_like(logits).scatter_(1, logits.argmax(dim=1, keepdim=True), 1.0)
    else:
        # Scale logits by temperature
        return logits / temperature
```

## Top-k Filtering Algorithm

### Concept and Purpose

Top-k filtering restricts sampling to the k most probable tokens, setting all other probabilities to zero (or negative infinity in logit space).

### Mathematical Implementation

```python
def apply_top_k(logits, k):
    """Apply top-k filtering to logits"""
    if k is None or k == 0:
        return logits
    
    # Find k-th largest value
    values, indices = torch.topk(logits, min(k, logits.size(-1)))
    
    # Set all values below k-th largest to -inf
    logits_filtered = logits.clone()
    logits_filtered[logits < values[:, [-1]]] = -float('inf')
    
    return logits_filtered
```

### Step-by-Step Process

#### 1. Top-k Value Identification
```python
v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
```

**Process**: Finds the k highest logit values
**Edge Case**: Uses `min(top_k, vocab_size)` to handle k > vocabulary size
**Output**: Values and indices of top-k tokens

#### 2. Probability Masking
```python
logits[logits < v[:, [-1]]] = -float('Inf')
```

**Threshold**: Uses k-th highest value as cutoff
**Masking**: Sets sub-threshold logits to negative infinity
**Effect**: These tokens will have zero probability after softmax

#### 3. Renormalization
```python
probs = F.softmax(logits, dim=-1)
```

**Automatic**: Softmax automatically renormalizes remaining probabilities
**Result**: Top-k tokens share 100% of probability mass

### Top-k Parameter Effects

#### Small k (k = 1-10)
- **Behavior**: Very focused, deterministic generation
- **Quality**: High coherence, low diversity
- **Use Cases**: Factual responses, structured output

#### Medium k (k = 20-100)  
- **Behavior**: Balanced coherence and creativity
- **Quality**: Good trade-off between quality and diversity
- **Use Cases**: General text generation, dialogue

#### Large k (k = 200+)
- **Behavior**: More exploratory, creative generation
- **Quality**: Higher diversity, potential quality reduction
- **Use Cases**: Creative writing, brainstorming

### Adaptive Top-k Strategy

```python
def adaptive_top_k(logits, base_k, confidence_threshold=0.9):
    """Dynamically adjust k based on prediction confidence"""
    probs = F.softmax(logits, dim=-1)
    max_prob = torch.max(probs, dim=-1)[0]
    
    if max_prob > confidence_threshold:
        # High confidence: use smaller k
        return min(base_k // 2, 10)
    else:
        # Low confidence: use larger k
        return base_k
```

## Advanced Sampling Strategies

### Top-p (Nucleus) Sampling

While not implemented in the base nanoGPT, top-p sampling is a common alternative:

```python
def top_p_sampling(logits, p=0.9):
    """Sample from smallest set of tokens with cumulative probability >= p"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Find cutoff index
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Set filtered logits to -inf
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('inf')
    
    return logits
```

### Repetition Penalty

```python
def apply_repetition_penalty(logits, input_ids, penalty=1.1):
    """Penalize tokens that appear in the input sequence"""
    for token_id in set(input_ids.tolist()):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty
    return logits
```

## Token Decoding Process

### Character-Level Decoding (Custom Datasets)
```python
def decode_character_level(token_ids, itos):
    """Decode token IDs to text using character mapping"""
    characters = [itos[token_id] for token_id in token_ids]
    return ''.join(characters)
```

**Process**:
1. Map each token ID to corresponding character using itos dictionary
2. Concatenate characters to form final text
3. Handle special tokens (padding, unknown) appropriately

### Subword Decoding (GPT-2 BPE)
```python
def decode_bpe(token_ids, tokenizer):
    """Decode token IDs using BPE tokenizer"""
    return tokenizer.decode(token_ids)
```

**Process**:
1. Use tiktoken decoder to convert token IDs to text
2. Handle byte-pair merging automatically
3. Process special tokens like "<|endoftext|>"

### Decoding Considerations

#### Special Token Handling
```python
def clean_decoded_text(text):
    """Clean up decoded text"""
    # Remove padding tokens
    text = text.replace('<|pad|>', '')
    
    # Handle end-of-text tokens
    if '<|endoftext|>' in text:
        text = text.split('<|endoftext|>')[0]
    
    return text.strip()
```

#### Unicode and Encoding Issues
- **BPE Tokenizers**: Handle Unicode properly through byte-level encoding
- **Character Tokenizers**: May need special handling for non-ASCII characters
- **Whitespace**: Preserve original spacing and formatting when possible

## Generation Quality Optimization

### Balancing Parameters

#### For Coherent, Factual Generation:
```python
temperature = 0.7
top_k = 40
max_new_tokens = 200
```

#### For Creative, Diverse Generation:
```python
temperature = 1.0
top_k = 200
max_new_tokens = 500
```

#### For Deterministic Generation:
```python
temperature = 0.0  # or very low like 0.1
top_k = 1
```

### Dynamic Parameter Adjustment

```python
def dynamic_generation_params(step, total_steps):
    """Adjust parameters during generation"""
    progress = step / total_steps
    
    # Start focused, become more creative
    temperature = 0.5 + 0.5 * progress
    top_k = int(20 + 180 * progress)
    
    return temperature, top_k
```

### Quality Metrics and Evaluation

#### Perplexity-Based Quality
```python
def calculate_generation_quality(model, generated_sequence):
    """Estimate generation quality using perplexity"""
    with torch.no_grad():
        logits, _ = model(generated_sequence[:-1])
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(2, generated_sequence[1:].unsqueeze(-1))
        perplexity = torch.exp(-target_log_probs.mean())
    return perplexity.item()
```

This comprehensive analysis covers the mathematical foundations, implementation details, and practical considerations for text generation algorithms in nanoGPT, providing both theoretical understanding and practical guidance for effective use and modification.