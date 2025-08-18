# sample.py Analysis: Inference and Text Generation Pipeline

> **Related Documentation**: [Text Generation Algorithms](../concepts/text-generation-algorithms.md) | [GPT Architecture](../concepts/gpt-architecture.md) | [Model Implementation](model-py-analysis.md) | [Concept Index](../concept-index.md#text-generation)

## File Overview

The `sample.py` script implements the inference pipeline for generating text using trained nanoGPT models. It handles model loading from checkpoints or pretrained GPT-2 models, configures the generation environment, and executes autoregressive text generation with various sampling strategies.

**Key Responsibilities:**
- Load trained models from checkpoints or pretrained GPT-2 variants → *Uses [GPT Model](model-py-analysis.md#pretrained-model-loading)*
- Configure inference environment (device, precision, compilation) → *See [Model Compilation](model-py-analysis.md#model-compilation)*
- Handle tokenization and encoding for different datasets → *See [Data Pipeline](../concepts/data-pipeline.md#tokenization-concepts)*
- Execute text generation with temperature and top-k sampling → *See [Sampling Strategies](../concepts/text-generation-algorithms.md#sampling-strategies)*
- Decode and display generated text samples → *See [Autoregressive Generation](../concepts/text-generation-algorithms.md#autoregressive-generation)*

## Import Analysis

```python
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
```

**Import Breakdown:**
- `os`: File system operations for checkpoint and meta file paths
- `pickle`: Loading tokenizer metadata from dataset preparation
- `nullcontext`: Context manager for CPU inference (no autocast needed)
- `torch`: Core PyTorch functionality for tensors and model operations
- `tiktoken`: OpenAI's BPE tokenizer for GPT-2 compatible encoding
- `model`: Custom GPT implementation with configuration and generation methods

## Configuration Parameters

### Model Loading Configuration
```python
init_from = 'resume'  # 'resume' or gpt2 variant ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
out_dir = 'out'       # Directory containing checkpoint files (ckpt.pt)
```

**Parameter Explanation:**
- `init_from`: Determines model source - 'resume' loads from local checkpoint, GPT-2 variants load pretrained models
- `out_dir`: Path to directory containing saved checkpoint when using 'resume' mode

### Generation Parameters
```python
start = "\n"              # Initial prompt text or "FILE:path" to load from file
num_samples = 10          # Number of independent text samples to generate
max_new_tokens = 500      # Maximum tokens to generate per sample
temperature = 0.8         # Sampling temperature (0.0 = deterministic, >1.0 = more random)
top_k = 200              # Top-k filtering (retain only k most likely tokens)
```

**Parameter Details:**
- `start`: Prompt initialization - can be direct text, special tokens like "<|endoftext|>", or file reference
- `num_samples`: Controls batch generation for multiple independent outputs
- `max_new_tokens`: Generation length limit to prevent infinite generation
- `temperature`: Controls randomness - lower values make output more focused and deterministic
- `top_k`: Nucleus sampling parameter - limits vocabulary to most probable tokens

### System Configuration
```python
seed = 1337               # Random seed for reproducible generation
device = 'cuda'           # Computation device ('cpu', 'cuda', 'cuda:0', etc.)
dtype = 'bfloat16'        # Precision type for inference optimization
compile = False           # PyTorch 2.0 compilation for performance
```

**System Setup:**
- `seed`: Ensures reproducible outputs across runs for debugging and comparison
- `device`: GPU utilization for faster inference, falls back to CPU if unavailable
- `dtype`: Mixed precision for memory efficiency - bfloat16 preferred on modern GPUs
- `compile`: JIT compilation for optimized inference (requires PyTorch 2.0+)

## Model Loading Pipeline

### Checkpoint Loading (Resume Mode)
```python
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    # Handle compiled model state dict cleanup
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
```

**Checkpoint Loading Process:**
1. **Path Construction**: Builds checkpoint path from output directory
2. **Checkpoint Loading**: Loads entire checkpoint dictionary with model args and weights
3. **Config Reconstruction**: Recreates GPTConfig from saved model arguments
4. **Model Instantiation**: Creates fresh GPT model with loaded configuration
5. **State Dict Cleanup**: Removes PyTorch compilation prefixes from parameter names
6. **Weight Loading**: Loads pretrained weights into model architecture

**Key Insight**: The `_orig_mod.` prefix removal handles models that were compiled during training, ensuring compatibility between compiled and non-compiled inference.

### Pretrained Model Loading
```python
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
```

**Pretrained Loading Process:**
- **Direct Loading**: Uses custom `from_pretrained` method to load OpenAI GPT-2 weights
- **Dropout Disable**: Sets dropout to 0.0 for deterministic inference behavior
- **Automatic Configuration**: Model configuration is inferred from pretrained variant

## Device and Optimization Setup

### Random Seed Configuration
```python
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
```

**Seed Setup**: Ensures reproducible generation across CPU and GPU operations for consistent debugging and evaluation.

### Performance Optimizations
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
```

**Optimization Details:**
- **TF32 Acceleration**: Enables Tensor Float-32 for faster matrix operations on Ampere GPUs
- **Device Type Detection**: Extracts device type for autocast context configuration
- **Precision Mapping**: Converts string dtype to PyTorch tensor type
- **Autocast Context**: Sets up automatic mixed precision for GPU inference, uses nullcontext for CPU

### Model Preparation
```python
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)
```

**Model Setup Process:**
1. **Evaluation Mode**: Disables dropout and batch normalization training behavior
2. **Device Transfer**: Moves model parameters to specified device (GPU/CPU)
3. **Optional Compilation**: Applies PyTorch 2.0 JIT compilation for optimized inference

## Tokenization System

### Metadata Loading
```python
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
```

**Custom Tokenizer Loading:**
- **Metadata Detection**: Checks for dataset-specific tokenizer metadata from training
- **Character-Level Encoding**: Loads string-to-index (stoi) and index-to-string (itos) mappings
- **Encode Function**: Converts text strings to token ID lists using character mapping
- **Decode Function**: Converts token ID lists back to text strings

### Fallback Tokenization
```python
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
```

**GPT-2 Tokenizer Fallback:**
- **Default Encoding**: Uses OpenAI's GPT-2 BPE tokenizer when no custom metadata exists
- **Special Token Handling**: Allows "<|endoftext|>" token for proper sequence termination
- **Subword Encoding**: Provides more efficient encoding than character-level for natural language

## Prompt Processing

### Prompt Initialization
```python
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
```

**Prompt Processing Pipeline:**
1. **File Loading**: Handles "FILE:" prefix to load prompts from external files
2. **Text Encoding**: Converts prompt text to token IDs using appropriate tokenizer
3. **Tensor Creation**: Creates PyTorch tensor with proper dtype and device placement
4. **Batch Dimension**: Adds batch dimension with `[None, ...]` for model compatibility

**Key Design**: The `[None, ...]` indexing adds a batch dimension, creating shape `(1, sequence_length)` required by the model's forward pass.#
# Text Generation Execution

### Generation Loop
```python
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
```

**Generation Process:**
1. **Gradient Disable**: `torch.no_grad()` disables gradient computation for inference efficiency
2. **Precision Context**: Uses autocast context for mixed precision inference
3. **Sample Loop**: Generates multiple independent samples from the same prompt
4. **Model Generation**: Calls model's generate method with sampling parameters
5. **Decoding**: Converts generated token IDs back to human-readable text
6. **Output Display**: Prints each sample with separator for clarity

**Memory Optimization**: The `torch.no_grad()` context prevents PyTorch from building computation graphs, significantly reducing memory usage during inference.

## Autoregressive Generation Algorithm

### Core Generation Method (from model.py)
```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # Context window management
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        
        # Forward pass for next token prediction
        logits, _ = self(idx_cond)
        
        # Temperature scaling
        logits = logits[:, -1, :] / temperature
        
        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Probability conversion and sampling
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Sequence extension
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
```

### Step-by-Step Generation Process

#### 1. Context Window Management
```python
idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
```
**Purpose**: Ensures input sequence doesn't exceed model's maximum context length
**Implementation**: Crops sequence to last `block_size` tokens when necessary
**Rationale**: Maintains computational efficiency while preserving recent context

#### 2. Forward Pass Prediction
```python
logits, _ = self(idx_cond)
```
**Process**: Runs conditioned sequence through transformer to get next-token predictions
**Output**: Raw logits for entire vocabulary at each position
**Focus**: Only the final position logits are used for next token prediction

#### 3. Temperature Scaling
```python
logits = logits[:, -1, :] / temperature
```
**Mathematical Effect**: 
- `temperature < 1.0`: Sharpens distribution (more deterministic)
- `temperature = 1.0`: No change to distribution
- `temperature > 1.0`: Flattens distribution (more random)

**Implementation**: Divides logits by temperature before softmax conversion

#### 4. Top-k Filtering
```python
if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
```
**Process**:
1. Find top-k highest logit values using `torch.topk`
2. Set all other logits to negative infinity
3. Effectively removes low-probability tokens from consideration

**Effect**: Prevents sampling from unlikely tokens, improving generation quality

#### 5. Probability Conversion
```python
probs = F.softmax(logits, dim=-1)
```
**Transformation**: Converts logits to normalized probability distribution
**Mathematical**: Applies softmax function: `softmax(x_i) = exp(x_i) / sum(exp(x_j))`

#### 6. Sampling
```python
idx_next = torch.multinomial(probs, num_samples=1)
```
**Method**: Multinomial sampling from probability distribution
**Output**: Single token ID sampled according to computed probabilities
**Randomness**: Introduces controlled randomness based on temperature and top-k settings

#### 7. Sequence Extension
```python
idx = torch.cat((idx, idx_next), dim=1)
```
**Operation**: Appends newly sampled token to existing sequence
**Growth**: Sequence length increases by 1 each iteration
**Continuation**: Extended sequence becomes input for next iteration

## Sampling Strategies Explained

### Temperature Scaling Effects

**Low Temperature (0.1 - 0.7):**
- More deterministic output
- Focuses on high-probability tokens
- Reduces creativity but improves coherence
- Suitable for factual or structured generation

**Medium Temperature (0.7 - 1.2):**
- Balanced randomness and coherence
- Good for creative writing and dialogue
- Default range for most applications

**High Temperature (1.2+):**
- Highly random output
- Explores low-probability tokens
- Increases creativity but may reduce coherence
- Useful for brainstorming or experimental generation

### Top-k Filtering Benefits

**Quality Control**: Prevents sampling from clearly inappropriate tokens
**Computational Efficiency**: Reduces effective vocabulary size during sampling
**Stability**: Provides consistent generation quality across different prompts
**Flexibility**: Can be adjusted based on desired output diversity

## Integration with Training Pipeline

### Checkpoint Compatibility
- **Model Architecture**: Must match training configuration exactly
- **Tokenizer Consistency**: Uses same encoding scheme as training data
- **Parameter Loading**: Handles both compiled and non-compiled model states

### Configuration Inheritance
- **Hyperparameters**: Inherits model configuration from training checkpoint
- **Dataset Metadata**: Uses tokenizer metadata from data preparation
- **Device Settings**: Adapts to available hardware for optimal performance

## Performance Considerations

### Memory Optimization
- **No Gradient Computation**: Saves significant memory during inference
- **Mixed Precision**: Reduces memory footprint with minimal quality impact
- **Context Cropping**: Prevents memory growth with very long sequences

### Speed Optimization
- **Model Compilation**: PyTorch 2.0 compilation for faster execution
- **TF32 Acceleration**: Hardware-specific optimizations for modern GPUs
- **Batch Processing**: Efficient tensor operations for multiple samples

### Quality vs Speed Tradeoffs
- **Temperature**: Lower values faster (more deterministic sampling)
- **Top-k**: Smaller values faster (reduced vocabulary consideration)
- **Sequence Length**: Shorter generations faster (fewer autoregressive steps)

## Error Handling and Edge Cases

### Missing Files
- **Checkpoint Not Found**: Clear error when resume checkpoint missing
- **Meta File Missing**: Graceful fallback to GPT-2 tokenizer
- **Prompt File Missing**: File loading error with clear message

### Device Compatibility
- **CUDA Unavailable**: Automatic fallback to CPU inference
- **Memory Insufficient**: Context cropping prevents out-of-memory errors
- **Precision Unsupported**: Fallback precision selection based on hardware

### Generation Edge Cases
- **Empty Prompt**: Handles zero-length initial sequences
- **Very Long Prompts**: Context window cropping maintains functionality
- **Special Tokens**: Proper handling of end-of-text and padding tokens

This comprehensive analysis covers the complete inference pipeline from model loading through text generation, providing detailed explanations of each component's purpose, implementation, and integration within the broader nanoGPT system.