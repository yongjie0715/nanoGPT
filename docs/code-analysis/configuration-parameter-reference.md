# Configuration Parameter Reference

## Overview

This document provides a comprehensive reference for all configuration parameters available in nanoGPT. Parameters are organized by category and include default values, valid ranges, and detailed explanations of their effects on training and model behavior.

## Parameter Categories

### I/O Parameters

#### `out_dir`
- **Type**: `str`
- **Default**: `'out'`
- **Description**: Directory where model checkpoints and training outputs are saved
- **Usage**: Specify a custom output directory for organizing different experiments
- **Example**: `--out_dir=experiments/gpt2-large`

#### `eval_interval`
- **Type**: `int`
- **Default**: `2000`
- **Description**: Number of training iterations between model evaluations
- **Range**: `1` to `max_iters`
- **Impact**: Lower values provide more frequent validation feedback but slow training
- **Recommended**: `1000-5000` for most experiments

#### `log_interval`
- **Type**: `int`
- **Default**: `1`
- **Description**: Number of training iterations between logging training metrics
- **Range**: `1` to `eval_interval`
- **Impact**: Higher values reduce logging overhead but provide less granular monitoring
- **Recommended**: `1-10` for detailed monitoring, `50-100` for production runs

#### `eval_iters`
- **Type**: `int`
- **Default**: `200`
- **Description**: Number of validation batches to evaluate during each evaluation
- **Range**: `10` to `1000`
- **Impact**: Higher values provide more accurate validation loss estimates but take longer
- **Recommended**: `100-500` depending on dataset size

#### `eval_only`
- **Type**: `bool`
- **Default**: `False`
- **Description**: If True, performs only evaluation and exits (no training)
- **Usage**: Useful for testing trained models or debugging evaluation pipeline
- **Example**: `--eval_only=True`

#### `always_save_checkpoint`
- **Type**: `bool`
- **Default**: `True`
- **Description**: If True, saves checkpoint after every evaluation regardless of validation loss
- **Impact**: Ensures no progress is lost but uses more disk space
- **Recommended**: `True` for experiments, `False` for production with good validation loss

#### `init_from`
- **Type**: `str`
- **Default**: `'scratch'`
- **Valid Values**: `'scratch'`, `'resume'`, `'gpt2'`, `'gpt2-medium'`, `'gpt2-large'`, `'gpt2-xl'`
- **Description**: Initialization strategy for model weights
  - `'scratch'`: Random initialization
  - `'resume'`: Resume from checkpoint in `out_dir`
  - `'gpt2*'`: Load pretrained OpenAI GPT-2 weights
- **Example**: `--init_from=gpt2-medium`

### Weights & Biases Logging

#### `wandb_log`
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable Weights & Biases experiment tracking
- **Requirements**: Requires `wandb` package installation and login
- **Example**: `--wandb_log=True`

#### `wandb_project`
- **Type**: `str`
- **Default**: `'owt'`
- **Description**: W&B project name for organizing experiments
- **Usage**: Group related experiments under the same project
- **Example**: `--wandb_project=gpt2-experiments`

#### `wandb_run_name`
- **Type**: `str`
- **Default**: `'gpt2'`
- **Description**: Specific run name within the W&B project
- **Usage**: Identify individual experiment runs
- **Example**: `--wandb_run_name=gpt2-124M-lr6e4`

### Data Parameters

#### `dataset`
- **Type**: `str`
- **Default**: `'openwebtext'`
- **Valid Values**: `'openwebtext'`, `'shakespeare'`, `'shakespeare_char'`, or custom dataset name
- **Description**: Dataset to use for training
- **Requirements**: Dataset must be prepared in `data/{dataset}/` directory
- **Example**: `--dataset=shakespeare`

#### `gradient_accumulation_steps`
- **Type**: `int`
- **Default**: `40` (5 * 8)
- **Description**: Number of micro-batches to accumulate before updating weights
- **Purpose**: Simulates larger batch sizes when GPU memory is limited
- **Effective Batch Size**: `gradient_accumulation_steps * batch_size * num_gpus`
- **Range**: `1` to `1000`
- **Example**: `--gradient_accumulation_steps=16`

#### `batch_size`
- **Type**: `int`
- **Default**: `12`
- **Description**: Micro-batch size per GPU (number of sequences processed simultaneously)
- **Memory Impact**: Higher values require more GPU memory
- **Range**: `1` to GPU memory limit
- **Recommended**: `8-64` depending on model size and GPU memory
- **Example**: `--batch_size=32`

#### `block_size`
- **Type**: `int`
- **Default**: `1024`
- **Description**: Maximum sequence length (context window) in tokens
- **Impact**: Longer sequences require quadratically more memory for attention
- **Valid Values**: Powers of 2, typically `256`, `512`, `1024`, `2048`, `4096`
- **Constraints**: Must match or be smaller than pretrained model's context length
- **Example**: `--block_size=2048`

### Model Architecture Parameters

#### `n_layer`
- **Type**: `int`
- **Default**: `12`
- **Description**: Number of transformer layers (depth of the model)
- **Impact**: More layers increase model capacity but require more memory and computation
- **GPT-2 Variants**:
  - GPT-2 Small: `12`
  - GPT-2 Medium: `24`
  - GPT-2 Large: `36`
  - GPT-2 XL: `48`
- **Range**: `1` to `100+`
- **Example**: `--n_layer=24`

#### `n_head`
- **Type**: `int`
- **Default**: `12`
- **Description**: Number of attention heads in each transformer layer
- **Constraint**: `n_embd` must be divisible by `n_head`
- **Impact**: More heads allow the model to attend to different types of relationships
- **GPT-2 Variants**:
  - GPT-2 Small: `12`
  - GPT-2 Medium: `16`
  - GPT-2 Large: `20`
  - GPT-2 XL: `25`
- **Example**: `--n_head=16`

#### `n_embd`
- **Type**: `int`
- **Default**: `768`
- **Description**: Embedding dimension (model width)
- **Impact**: Larger embeddings increase model capacity and memory usage
- **Constraint**: Must be divisible by `n_head`
- **GPT-2 Variants**:
  - GPT-2 Small: `768`
  - GPT-2 Medium: `1024`
  - GPT-2 Large: `1280`
  - GPT-2 XL: `1600`
- **Example**: `--n_embd=1024`

#### `dropout`
- **Type**: `float`
- **Default**: `0.0`
- **Description**: Dropout probability for regularization
- **Range**: `0.0` to `0.5`
- **Usage**:
  - Pretraining: `0.0` (no dropout)
  - Fine-tuning: `0.1-0.3` (prevents overfitting)
- **Example**: `--dropout=0.1`

#### `bias`
- **Type**: `bool`
- **Default**: `False`
- **Description**: Whether to use bias terms in LayerNorm and Linear layers
- **Impact**: 
  - `True`: Slightly more parameters, may improve performance
  - `False`: Fewer parameters, matches modern practices
- **Recommendation**: `False` for most use cases
- **Example**: `--bias=True`

### Optimization Parameters

#### `learning_rate`
- **Type**: `float`
- **Default**: `6e-4`
- **Description**: Maximum learning rate for the optimizer
- **Range**: `1e-6` to `1e-2`
- **Impact**: Higher rates speed up training but may cause instability
- **Scaling**: Should scale with batch size (larger batches → higher LR)
- **Recommended Ranges**:
  - Small models: `1e-4` to `1e-3`
  - Large models: `1e-5` to `6e-4`
- **Example**: `--learning_rate=3e-4`

#### `max_iters`
- **Type**: `int`
- **Default**: `600000`
- **Description**: Total number of training iterations
- **Calculation**: Depends on desired number of tokens and batch size
- **GPT-2 Training**: ~300B tokens for full pretraining
- **Fine-tuning**: `100-10000` iterations typically sufficient
- **Example**: `--max_iters=100000`

#### `weight_decay`
- **Type**: `float`
- **Default**: `1e-1`
- **Description**: L2 regularization strength for AdamW optimizer
- **Range**: `0.0` to `1.0`
- **Impact**: Higher values prevent overfitting but may hurt performance
- **Recommended**: `1e-2` to `1e-1` for most models
- **Example**: `--weight_decay=1e-2`

#### `beta1`
- **Type**: `float`
- **Default**: `0.9`
- **Description**: First momentum parameter for AdamW optimizer
- **Range**: `0.0` to `0.999`
- **Standard**: `0.9` works well for most cases
- **Impact**: Controls exponential decay rate of first moment estimates
- **Example**: `--beta1=0.95`

#### `beta2`
- **Type**: `float`
- **Default**: `0.95`
- **Description**: Second momentum parameter for AdamW optimizer
- **Range**: `0.9` to `0.999`
- **Standard**: `0.95-0.999` for transformer models
- **Impact**: Controls exponential decay rate of second moment estimates
- **Example**: `--beta2=0.999`

#### `grad_clip`
- **Type**: `float`
- **Default**: `1.0`
- **Description**: Gradient clipping threshold (0.0 disables clipping)
- **Range**: `0.0` to `10.0`
- **Purpose**: Prevents gradient explosion during training
- **Recommended**: `0.5-2.0` for most models
- **Example**: `--grad_clip=0.5`

### Learning Rate Schedule Parameters

#### `decay_lr`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Whether to use cosine learning rate decay
- **Impact**: 
  - `True`: LR decays from `learning_rate` to `min_lr`
  - `False`: Constant learning rate throughout training
- **Recommendation**: `True` for pretraining, varies for fine-tuning
- **Example**: `--decay_lr=False`

#### `warmup_iters`
- **Type**: `int`
- **Default**: `2000`
- **Description**: Number of iterations for learning rate warmup
- **Purpose**: Gradually increases LR from 0 to `learning_rate`
- **Range**: `0` to `max_iters/10`
- **Recommended**: `1000-5000` for large models, `100-1000` for small models
- **Example**: `--warmup_iters=1000`

#### `lr_decay_iters`
- **Type**: `int`
- **Default**: `600000`
- **Description**: Number of iterations over which to decay learning rate
- **Recommendation**: Should equal `max_iters` for full training
- **Impact**: Longer decay provides more stable training
- **Example**: `--lr_decay_iters=100000`

#### `min_lr`
- **Type**: `float`
- **Default**: `6e-5`
- **Description**: Minimum learning rate at end of decay schedule
- **Recommendation**: ~10% of `learning_rate` (following Chinchilla paper)
- **Range**: `1e-7` to `learning_rate/2`
- **Example**: `--min_lr=3e-5`

### Distributed Training Parameters

#### `backend`
- **Type**: `str`
- **Default**: `'nccl'`
- **Valid Values**: `'nccl'`, `'gloo'`, `'mpi'`
- **Description**: Backend for distributed training communication
- **Recommendations**:
  - `'nccl'`: Best for NVIDIA GPUs
  - `'gloo'`: CPU or mixed environments
  - `'mpi'`: HPC environments
- **Example**: `--backend=gloo`

### System Parameters

#### `device`
- **Type**: `str`
- **Default**: `'cuda'`
- **Valid Values**: `'cpu'`, `'cuda'`, `'cuda:0'`, `'cuda:1'`, `'mps'`
- **Description**: Device to use for training
- **Auto-detection**: Automatically selects best available device
- **Multi-GPU**: Use `'cuda'` for automatic GPU selection in DDP
- **Example**: `--device=cuda:1`

#### `dtype`
- **Type**: `str`
- **Default**: `'bfloat16'` (if supported) or `'float16'`
- **Valid Values**: `'float32'`, `'bfloat16'`, `'float16'`
- **Description**: Floating point precision for training
- **Trade-offs**:
  - `'float32'`: Highest precision, most memory
  - `'bfloat16'`: Good precision, moderate memory, stable training
  - `'float16'`: Lowest memory, may need gradient scaling
- **Example**: `--dtype=float32`

#### `compile`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Use PyTorch 2.0 compilation for faster training
- **Requirements**: PyTorch 2.0+
- **Impact**: 10-20% speedup but longer startup time
- **Compatibility**: May not work with all model modifications
- **Example**: `--compile=False`

#### `dashboard`
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable web dashboard for real-time training visualization
- **Requirements**: Dashboard module must be available
- **Usage**: Provides web interface for monitoring training progress
- **Example**: `--dashboard=True`

## Parameter Interactions and Dependencies

### Model Size Relationships
The model parameters must maintain specific relationships:
- `n_embd` must be divisible by `n_head`
- Larger `n_layer`, `n_head`, or `n_embd` increase memory usage quadratically
- `block_size` affects memory usage quadratically due to attention mechanism

### Batch Size Calculations
Effective batch size = `batch_size` × `gradient_accumulation_steps` × `num_gpus`
- Target effective batch sizes: 0.5M-2M tokens for pretraining
- Memory usage scales linearly with `batch_size` and quadratically with `block_size`

### Learning Rate Scaling
- Learning rate should scale with effective batch size
- Rule of thumb: LR ∝ √(effective_batch_size)
- Warmup becomes more important with higher learning rates

### Memory Optimization
- Use `gradient_accumulation_steps` to simulate larger batches
- Reduce `batch_size` or `block_size` if running out of memory
- `dtype='float16'` or `dtype='bfloat16'` can halve memory usage

## Common Configuration Patterns

### Small Scale Experimentation
```python
# Quick testing configuration
batch_size = 4
gradient_accumulation_steps = 1
max_iters = 1000
eval_interval = 100
compile = False
```

### Production Pretraining
```python
# Large scale pretraining
batch_size = 12
gradient_accumulation_steps = 40
max_iters = 600000
learning_rate = 6e-4
weight_decay = 1e-1
```

### Fine-tuning Setup
```python
# Fine-tuning configuration
init_from = 'gpt2'
learning_rate = 3e-5
max_iters = 1000
decay_lr = False
dropout = 0.1
```

### Multi-GPU Training
```bash
# Launch with torchrun for distributed training
torchrun --standalone --nproc_per_node=8 train.py \
  --batch_size=6 \
  --gradient_accumulation_steps=5 \
  --backend=nccl
```

## Troubleshooting Common Issues

### Out of Memory Errors
1. Reduce `batch_size`
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `block_size`
4. Use `dtype='float16'` or `dtype='bfloat16'`

### Training Instability
1. Lower `learning_rate`
2. Increase `warmup_iters`
3. Enable `grad_clip` (try 1.0)
4. Check `dtype` compatibility

### Slow Training
1. Enable `compile=True`
2. Increase `batch_size` if memory allows
3. Use appropriate `dtype` for your hardware
4. Optimize `log_interval` and `eval_interval`

### Convergence Issues
1. Verify dataset preparation
2. Check learning rate schedule parameters
3. Ensure sufficient `max_iters`
4. Validate model architecture parameters