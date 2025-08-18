# Customization and Extension Guide

This guide shows you how to modify nanoGPT for research purposes, add new datasets, and implement common architectural changes. Whether you're experimenting with new transformer variants or adapting the code for specific domains, this guide provides practical examples and implementation patterns.

## Model Architecture Modifications

### 1. Changing Model Size and Configuration

The easiest way to experiment is by modifying the model configuration parameters.

#### Creating Custom Model Configs

Create a new config file for your experiments:

```python
# config/my_experiment.py

# Model architecture
n_layer = 8        # Number of transformer blocks
n_head = 8         # Number of attention heads  
n_embd = 512       # Embedding dimension
dropout = 0.1      # Dropout rate

# Training parameters
batch_size = 8
block_size = 512   # Context length
max_iters = 5000
learning_rate = 3e-4

# Dataset
dataset = 'openwebtext'
out_dir = 'out-my-experiment'

# Evaluation
eval_interval = 250
eval_iters = 200
```

#### Model Size Guidelines

**Micro Model (Fast experimentation):**
```python
n_layer = 4
n_head = 4
n_embd = 256
# Parameters: ~3M, Memory: ~1GB
```

**Small Model (Balanced):**
```python
n_layer = 8
n_head = 8
n_embd = 512
# Parameters: ~25M, Memory: ~4GB
```

**Medium Model (GPT-2 Small):**
```python
n_layer = 12
n_head = 12
n_embd = 768
# Parameters: ~117M, Memory: ~8GB
```

### 2. Adding New Attention Mechanisms

#### Implementing Rotary Position Embedding (RoPE)

Add RoPE to replace standard positional embeddings:

```python
# In model.py, add this class
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
            
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to queries and keys"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Modify CausalSelfAttention class
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... existing code ...
        
        # Add RoPE
        self.rotary_emb = RotaryEmbedding(config.n_embd // config.n_head)
        
    def forward(self, x):
        B, T, C = x.size()
        
        # ... existing q, k, v computation ...
        
        # Apply RoPE
        cos, sin = self.rotary_emb(x, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # ... rest of attention computation ...
```

#### Adding Sliding Window Attention

Implement local attention for longer sequences:

```python
class SlidingWindowAttention(nn.Module):
    def __init__(self, config, window_size=256):
        super().__init__()
        self.window_size = window_size
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate q, k, v
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Create sliding window mask
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        window_mask = torch.triu(torch.ones(T, T), diagonal=-self.window_size).bool()
        mask = mask | window_mask
        
        # Apply attention with sliding window
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
```

### 3. Implementing Alternative Architectures

#### Adding Layer Normalization Variants

Replace LayerNorm with RMSNorm:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)

# Replace LayerNorm in Block class
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)  # Changed from LayerNorm
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)  # Changed from LayerNorm
        self.mlp = MLP(config)
```

#### Adding Mixture of Experts (MoE)

Replace standard MLP with MoE:

```python
class MoEMLP(nn.Module):
    def __init__(self, config, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router network
        self.router = nn.Linear(config.n_embd, num_experts, bias=False)
        
        # Expert networks
        self.experts = nn.ModuleList([
            MLP(config) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Route to experts
        router_logits = self.router(x)  # (B, T, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # Compute expert outputs
        final_output = torch.zeros_like(x)
        
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]
            expert_weight = top_k_weights[:, :, i].unsqueeze(-1)
            
            # Process each expert
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    final_output[mask] += expert_weight[mask] * expert_output
                    
        return final_output
```

## Adding New Datasets

### 1. Text Dataset Integration

#### Creating a Custom Dataset Loader

```python
# data/my_dataset/prepare.py
import os
import pickle
import requests
import numpy as np
import tiktoken
from datasets import load_dataset

def prepare_custom_dataset():
    """Prepare your custom text dataset"""
    
    # Option 1: Load from HuggingFace
    dataset = load_dataset("your_username/your_dataset")
    text_data = "\n".join(dataset['train']['text'])
    
    # Option 2: Load from local files
    # text_files = glob.glob("*.txt")
    # text_data = ""
    # for file in text_files:
    #     with open(file, 'r', encoding='utf-8') as f:
    #         text_data += f.read() + "\n"
    
    # Option 3: Load from API/web scraping
    # text_data = scrape_your_data_source()
    
    # Tokenize using GPT-2 BPE
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text_data)
    
    print(f"Dataset has {len(tokens):,} tokens")
    print(f"Vocabulary size: {enc.n_vocab}")
    
    # Split train/validation
    train_tokens = tokens[:int(len(tokens) * 0.9)]
    val_tokens = tokens[int(len(tokens) * 0.9):]
    
    # Save as binary files
    train_ids = np.array(train_tokens, dtype=np.uint16)
    val_ids = np.array(val_tokens, dtype=np.uint16)
    
    train_ids.tofile('train.bin')
    val_ids.tofile('val.bin')
    
    # Save metadata
    meta = {
        'vocab_size': enc.n_vocab,
        'train_size': len(train_tokens),
        'val_size': len(val_tokens),
    }
    
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Saved {len(train_tokens):,} train tokens")
    print(f"Saved {len(val_tokens):,} validation tokens")

if __name__ == "__main__":
    prepare_custom_dataset()
```

#### Domain-Specific Preprocessing

For code datasets:

```python
# data/code_dataset/prepare.py
import ast
import tokenize
import io

def preprocess_code(code_text):
    """Clean and preprocess code text"""
    try:
        # Parse to ensure valid Python
        ast.parse(code_text)
        
        # Remove comments (optional)
        tokens = tokenize.generate_tokens(io.StringIO(code_text).readline)
        clean_code = tokenize.untokenize(
            token for token in tokens 
            if token.type != tokenize.COMMENT
        )
        
        return clean_code.decode('utf-8')
    except:
        return None  # Skip invalid code

def prepare_code_dataset():
    # Load code files
    code_files = glob.glob("**/*.py", recursive=True)
    
    clean_code_texts = []
    for file_path in code_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            
        clean_code = preprocess_code(code)
        if clean_code:
            clean_code_texts.append(clean_code)
    
    # Join with special separator
    full_text = "\n<|endoffile|>\n".join(clean_code_texts)
    
    # Continue with standard tokenization...
```

### 2. Multimodal Dataset Support

#### Adding Image-Text Pairs

```python
# For vision-language models (requires architecture changes)
class MultimodalDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_processor = self.setup_image_processor()
        self.text_tokenizer = tiktoken.get_encoding("gpt2")
        
    def setup_image_processor(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def process_item(self, image_path, caption):
        # Process image
        image = Image.open(image_path)
        image_tensor = self.image_processor(image)
        
        # Process text
        text_tokens = self.text_tokenizer.encode(caption)
        
        return {
            'image': image_tensor,
            'text': torch.tensor(text_tokens, dtype=torch.long),
            'length': len(text_tokens)
        }
```

## Training Configuration Modifications

### 1. Custom Learning Rate Schedules

#### Implementing Cosine Annealing

```python
# Add to train.py
import math

def get_lr_cosine(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    """Cosine annealing learning rate schedule"""
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    
    # Cosine decay
    if it > lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# In training loop
lr = get_lr_cosine(iter_num, learning_rate, warmup_iters, lr_decay_iters, min_lr)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

#### Adding Learning Rate Finder

```python
def find_learning_rate(model, train_loader, init_lr=1e-8, final_lr=10, beta=0.98):
    """Find optimal learning rate using the method from fastai"""
    num_batches = len(train_loader)
    mult = (final_lr / init_lr) ** (1/num_batches)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)
    avg_loss = 0.0
    best_loss = 0.0
    batch_num = 0
    losses = []
    lrs = []
    
    for batch in train_loader:
        batch_num += 1
        
        # Forward pass
        optimizer.zero_grad()
        loss = model(batch)
        
        # Compute smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        
        # Stop if loss explodes
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break
            
        # Record best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
            
        # Store values
        losses.append(smoothed_loss)
        lrs.append(optimizer.param_groups[0]['lr'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        optimizer.param_groups[0]['lr'] *= mult
        
    return lrs, losses
```

### 2. Advanced Optimization Techniques

#### Implementing Gradient Clipping Variants

```python
def clip_grad_norm_per_layer(parameters, max_norm, norm_type=2):
    """Clip gradients per layer instead of globally"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        clip_coef = max_norm / (param_norm + 1e-6)
        if clip_coef < 1:
            p.grad.data.mul_(clip_coef)

# In training loop
clip_grad_norm_per_layer(model.parameters(), grad_clip)
```

#### Adding Lookahead Optimizer

```python
class Lookahead:
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        
        for group in self.param_groups:
            group["counter"] = 0
            
    def update_slow(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            
            slow = param_state["slow_param"]
            slow += self.alpha * (fast.data - slow)
            fast.data.copy_(slow)
            
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update_slow(group)
            group["counter"] += 1
            
            if group["counter"] >= self.k:
                group["counter"] = 0
                
        return loss

# Usage
base_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
```

## Evaluation and Analysis Extensions

### 1. Custom Evaluation Metrics

#### Implementing Perplexity Tracking

```python
def evaluate_perplexity(model, eval_loader, device):
    """Calculate perplexity on evaluation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            logits, loss = model(x, y)
            
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss
```

#### Adding BLEU Score for Generation

```python
from nltk.translate.bleu_score import sentence_bleu

def evaluate_generation_quality(model, test_prompts, tokenizer, device):
    """Evaluate generation quality using BLEU scores"""
    model.eval()
    bleu_scores = []
    
    for prompt, reference in test_prompts:
        # Generate text
        prompt_tokens = tokenizer.encode(prompt)
        generated = generate_text(model, prompt_tokens, max_new_tokens=50)
        generated_text = tokenizer.decode(generated)
        
        # Calculate BLEU
        reference_tokens = reference.split()
        generated_tokens = generated_text.split()
        
        bleu = sentence_bleu([reference_tokens], generated_tokens)
        bleu_scores.append(bleu)
    
    return sum(bleu_scores) / len(bleu_scores)
```

### 2. Model Analysis Tools

#### Attention Visualization

```python
def visualize_attention(model, text, layer_idx=0, head_idx=0):
    """Extract and visualize attention patterns"""
    model.eval()
    
    # Hook to capture attention weights
    attention_weights = []
    
    def attention_hook(module, input, output):
        # Capture attention weights from CausalSelfAttention
        if hasattr(module, 'att'):
            attention_weights.append(module.att.detach().cpu())
    
    # Register hook
    hook = model.transformer.h[layer_idx].attn.register_forward_hook(attention_hook)
    
    # Forward pass
    tokens = tiktoken.get_encoding("gpt2").encode(text)
    x = torch.tensor(tokens).unsqueeze(0)
    
    with torch.no_grad():
        model(x)
    
    # Remove hook
    hook.remove()
    
    # Extract attention for specific head
    att_matrix = attention_weights[0][0, head_idx].numpy()
    
    return att_matrix, tokens
```

#### Parameter Analysis

```python
def analyze_model_parameters(model):
    """Analyze model parameter distribution and statistics"""
    param_stats = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_data = param.data.cpu().numpy()
            
            param_stats[name] = {
                'shape': param_data.shape,
                'total_params': param_data.size,
                'mean': np.mean(param_data),
                'std': np.std(param_data),
                'min': np.min(param_data),
                'max': np.max(param_data),
                'zero_fraction': np.mean(param_data == 0),
            }
    
    return param_stats

def print_parameter_summary(model):
    """Print a summary of model parameters"""
    stats = analyze_model_parameters(model)
    
    total_params = sum(s['total_params'] for s in stats.values())
    print(f"Total parameters: {total_params:,}")
    
    for name, stat in stats.items():
        print(f"{name:30} | {str(stat['shape']):15} | "
              f"{stat['total_params']:8,} | "
              f"μ={stat['mean']:6.3f} σ={stat['std']:6.3f}")
```

## Research Extensions

### 1. Implementing Recent Techniques

#### Adding Flash Attention 2

```python
# Requires flash-attn package: pip install flash-attn
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False

class FlashCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate q, k, v
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)
        
        if FLASH_AVAILABLE and x.device.type == 'cuda':
            # Use Flash Attention
            y = flash_attn_func(q, k, v, dropout_p=self.dropout, causal=True)
        else:
            # Fallback to standard attention
            y = self.standard_attention(q, k, v)
        
        y = y.view(B, T, C)
        return self.c_proj(y)
```

#### Adding Mamba/State Space Models

```python
class MambaBlock(nn.Module):
    """Simplified Mamba-style state space block"""
    def __init__(self, config):
        super().__init__()
        self.d_model = config.n_embd
        self.d_state = 16  # State dimension
        self.expand = 2    # Expansion factor
        
        self.in_proj = nn.Linear(self.d_model, self.expand * self.d_model * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.expand * self.d_model,
            out_channels=self.expand * self.d_model,
            kernel_size=3,
            padding=1,
            groups=self.expand * self.d_model,
        )
        
        self.x_proj = nn.Linear(self.expand * self.d_model, self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.expand * self.d_model, self.expand * self.d_model, bias=True)
        
        self.out_proj = nn.Linear(self.expand * self.d_model, self.d_model, bias=False)
        
    def forward(self, x):
        B, L, D = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, L, 2 * expand * D)
        x, z = xz.chunk(2, dim=-1)  # Each: (B, L, expand * D)
        
        # Convolution
        x = x.transpose(1, 2)  # (B, expand * D, L)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # (B, L, expand * D)
        
        # State space computation (simplified)
        x = F.silu(x)
        
        # Delta projection
        dt = self.dt_proj(x)  # (B, L, expand * D)
        dt = F.softplus(dt)
        
        # State projection
        BC = self.x_proj(x)  # (B, L, 2 * d_state)
        B, C = BC.chunk(2, dim=-1)  # Each: (B, L, d_state)
        
        # Simplified state space operation
        # In practice, this would involve more complex recurrent computation
        y = x * torch.sigmoid(B @ C.transpose(-1, -2))
        
        # Gate and output projection
        y = y * F.silu(z)
        return self.out_proj(y)
```

### 2. Experimental Training Techniques

#### Implementing Curriculum Learning

```python
class CurriculumScheduler:
    def __init__(self, start_length=64, end_length=1024, total_steps=10000):
        self.start_length = start_length
        self.end_length = end_length
        self.total_steps = total_steps
        
    def get_sequence_length(self, step):
        """Get current sequence length based on training step"""
        if step >= self.total_steps:
            return self.end_length
            
        progress = step / self.total_steps
        length = self.start_length + (self.end_length - self.start_length) * progress
        return int(length)

# In training loop
curriculum = CurriculumScheduler(64, 1024, 50000)
current_block_size = curriculum.get_sequence_length(iter_num)

# Modify data loading to use current_block_size
def get_batch_curriculum(split, block_size):
    # Modified get_batch function that uses variable block_size
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
```

## Deployment and Production

### 1. Model Optimization for Inference

#### Quantization

```python
def quantize_model(model, calibration_data):
    """Apply dynamic quantization to model"""
    model.eval()
    
    # Dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    return quantized_model

def benchmark_model(model, test_input, num_runs=100):
    """Benchmark model inference speed"""
    model.eval()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_input)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(test_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time
```

#### Model Compilation

```python
def optimize_for_inference(model):
    """Optimize model for inference"""
    model.eval()
    
    # Compile with PyTorch 2.0
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode='max-autotune')
    
    # Enable optimized attention
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False
    ):
        return model
```

### 2. Serving Infrastructure

#### Simple API Server

```python
# serve.py
from flask import Flask, request, jsonify
import torch
import tiktoken

app = Flask(__name__)

# Load model once at startup
model = GPT.from_pretrained('gpt2')
model.eval()
tokenizer = tiktoken.get_encoding('gpt2')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 50)
    temperature = data.get('temperature', 0.8)
    
    # Tokenize
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long)[None, ...]
    
    # Generate
    with torch.no_grad():
        generated = model.generate(x, max_new_tokens=max_tokens, temperature=temperature)
    
    # Decode
    generated_text = tokenizer.decode(generated[0].tolist())
    
    return jsonify({
        'generated_text': generated_text,
        'prompt': prompt
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

This comprehensive customization guide provides practical examples for extending nanoGPT in various directions. Whether you're implementing cutting-edge research techniques or adapting the code for production use, these patterns will help you build on the solid foundation that nanoGPT provides.

For more detailed explanations of the underlying concepts, refer to our [architecture documentation](../concepts/gpt-architecture.md) and [code analysis](../code-analysis/README.md).