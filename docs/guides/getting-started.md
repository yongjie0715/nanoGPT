# Getting Started with nanoGPT

This guide provides a step-by-step tutorial for running nanoGPT, from initial setup through training your first model and generating text. Whether you're new to language models or looking to understand GPT implementation details, this guide will get you up and running quickly.

## Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (recommended for training)
- At least 8GB of GPU memory for small models
- Basic familiarity with command-line operations

## Quick Setup

### 1. Environment Setup

First, set up your Python environment:

```bash
# Clone the repository
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT

# Create virtual environment (recommended)
python -m venv nanogpt-env
source nanogpt-env/bin/activate  # On Windows: nanogpt-env\Scripts\activate

# Install dependencies
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

### 2. Test Your Environment

Run the environment test to ensure everything is working:

```bash
python test_environment.py
```

Expected output:
```
✓ Python version: 3.x.x
✓ PyTorch version: 2.x.x
✓ CUDA available: True
✓ GPU device: [Your GPU Name]
✓ All dependencies installed correctly
```

## Your First Model: Shakespeare Character-Level

The fastest way to see nanoGPT in action is training a small character-level model on Shakespeare text.

### Step 1: Prepare the Data

```bash
cd data/shakespeare_char
python prepare.py
```

This script:
- Downloads the complete works of Shakespeare (~1MB text)
- Creates character-level vocabulary (65 unique characters)
- Splits into train/validation sets (90%/10%)
- Saves binary files for efficient loading

Expected output:
```
Downloading Shakespeare dataset...
Dataset size: 1,115,394 characters
Vocabulary size: 65 unique characters
Train split: 1,003,854 characters
Validation split: 111,540 characters
Saved train.bin and val.bin
```

### Step 2: Start Training

```bash
cd ../..  # Back to root directory
python train.py config/train_shakespeare_char.py
```

### Understanding Training Output

The training script will display output like this:

```
step 0: train loss 4.2674, val loss 4.2632
step 100: train loss 2.4856, val loss 2.5012
step 200: train loss 2.0234, val loss 2.1456
step 300: train loss 1.8123, val loss 1.9876
...
step 5000: train loss 1.2345, val loss 1.3456
```

**What these numbers mean:**

- **Step**: Training iteration number (batch processed)
- **Train loss**: Cross-entropy loss on training data (lower = better)
- **Val loss**: Cross-entropy loss on validation data (measures generalization)

**Good training indicators:**
- Both losses decrease over time
- Validation loss stays close to training loss (no overfitting)
- Loss typically starts around 4.0-4.5 and drops to 1.0-1.5

**Warning signs:**
- Validation loss increases while training loss decreases (overfitting)
- Loss stops decreasing (learning rate too low or model converged)
- Loss explodes to very high values (learning rate too high)

### Step 3: Generate Text

After training (or even during), generate text from your model:

```bash
python sample.py --out_dir=out-shakespeare-char
```

Expected output:
```
Loading model from out-shakespeare-char/ckpt.pt
Generating text...

ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief...
```

## Training a Real GPT Model

For a more serious model, let's train on OpenWebText (a reproduction of GPT-2's training data).

### Step 1: Prepare OpenWebText Data

```bash
cd data/openwebtext
python prepare.py
```

**Warning**: This downloads ~54GB of data and takes several hours to process.

The script will:
- Download OpenWebText dataset from HuggingFace
- Tokenize using GPT-2's BPE tokenizer (50,257 vocabulary)
- Create train/validation splits
- Save as memory-mapped binary files

### Step 2: Configure Training

Create a custom config file or use the provided one:

```bash
# Use the provided GPT-2 config
python train.py config/train_gpt2.py
```

**Key parameters in the config:**
- `n_layer=12`: Number of transformer blocks
- `n_head=12`: Number of attention heads
- `n_embd=768`: Embedding dimension
- `batch_size=12`: Sequences per batch
- `block_size=1024`: Maximum sequence length

### Step 3: Monitor Training

Training a real GPT model takes much longer. Monitor progress:

```bash
# Training output every 10 steps
step 0: train loss 10.9856, val loss 10.9823
step 10: train loss 9.2341, val loss 9.2456
step 20: train loss 8.1234, val loss 8.1567
...
step 1000: train loss 4.5678, val loss 4.6123
step 2000: train loss 3.8901, val loss 3.9234
```

**Expected loss progression:**
- Initial: ~11.0 (random predictions)
- After 1K steps: ~4.5-5.0
- After 10K steps: ~3.5-4.0
- After 100K steps: ~3.0-3.5

## Common Use Cases

### 1. Fine-tuning on Custom Data

To train on your own text data:

```bash
# 1. Create your dataset directory
mkdir data/my_dataset
cd data/my_dataset

# 2. Create prepare.py script
cat > prepare.py << 'EOF'
import os
import pickle
import requests
import numpy as np
import tiktoken

# Your text data (replace with your data loading)
with open('my_text.txt', 'r', encoding='utf-8') as f:
    data = f.read()

# Tokenize using GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(data)

# Split train/val
n = len(tokens)
train_tokens = tokens[:int(n*0.9)]
val_tokens = tokens[int(n*0.9):]

# Save as binary files
train_tokens = np.array(train_tokens, dtype=np.uint16)
val_tokens = np.array(val_tokens, dtype=np.uint16)
train_tokens.tofile('train.bin')
val_tokens.tofile('val.bin')

print(f"Saved {len(train_tokens)} train tokens, {len(val_tokens)} val tokens")
EOF

# 3. Run preparation
python prepare.py

# 4. Train with custom config
cd ../..
python train.py --dataset=my_dataset --batch_size=4 --max_iters=1000
```

### 2. Resuming Training

To resume from a checkpoint:

```bash
python train.py config/train_gpt2.py --resume
```

The script automatically finds the latest checkpoint in `out_dir` and continues training.

### 3. Evaluating Model Performance

Generate samples to evaluate your model:

```bash
# Generate multiple samples
python sample.py --num_samples=5 --max_new_tokens=200

# Use different sampling strategies
python sample.py --temperature=0.8 --top_k=40  # More creative
python sample.py --temperature=0.1 --top_k=1   # More deterministic
```

### 4. Model Comparison

Compare different model sizes:

```bash
# Small model (faster training)
python train.py --n_layer=6 --n_head=6 --n_embd=384

# Medium model (balanced)
python train.py config/train_gpt2.py

# Large model (better quality, slower)
python train.py --n_layer=24 --n_head=16 --n_embd=1024
```

## Interpreting Training Metrics

### Loss Curves

Monitor these patterns in your loss curves:

**Healthy Training:**
```
step 0: train loss 4.267, val loss 4.263
step 100: train loss 2.485, val loss 2.501
step 200: train loss 2.023, val loss 2.145
step 300: train loss 1.812, val loss 1.987
```
- Both losses decrease steadily
- Validation loss slightly higher than training loss
- Gap between train/val loss remains small

**Overfitting:**
```
step 0: train loss 4.267, val loss 4.263
step 100: train loss 2.485, val loss 2.501
step 200: train loss 1.823, val loss 2.345
step 300: train loss 1.412, val loss 2.687
```
- Training loss continues decreasing
- Validation loss starts increasing
- Growing gap indicates overfitting

**Underfitting:**
```
step 0: train loss 4.267, val loss 4.263
step 100: train loss 4.185, val loss 4.201
step 200: train loss 4.123, val loss 4.145
step 300: train loss 4.112, val loss 4.187
```
- Both losses decrease very slowly
- Model may need more capacity or longer training

### Performance Benchmarks

**Character-level Shakespeare:**
- Good model: Final loss ~1.0-1.5
- Training time: 5-10 minutes on modern GPU
- Sample quality: Coherent Shakespeare-style text

**Word-level OpenWebText:**
- Good model: Final loss ~3.0-3.5
- Training time: Hours to days depending on model size
- Sample quality: Coherent paragraphs on various topics

## Troubleshooting Common Issues

### Out of Memory Errors

```bash
# Reduce batch size
python train.py --batch_size=4

# Reduce model size
python train.py --n_layer=6 --n_embd=384

# Use gradient accumulation
python train.py --batch_size=1 --gradient_accumulation_steps=8
```

### Slow Training

```bash
# Enable compilation (PyTorch 2.0+)
python train.py --compile=True

# Use mixed precision
python train.py --dtype=bfloat16

# Increase batch size if memory allows
python train.py --batch_size=16
```

### Poor Sample Quality

```bash
# Train longer
python train.py --max_iters=10000

# Adjust sampling parameters
python sample.py --temperature=0.7 --top_k=50

# Use larger model
python train.py --n_layer=12 --n_embd=768
```

## Next Steps

Once you're comfortable with basic training:

1. **Experiment with architectures**: Try different model sizes and configurations
2. **Custom datasets**: Train on domain-specific text (code, poetry, technical docs)
3. **Advanced techniques**: Implement learning rate scheduling, different optimizers
4. **Evaluation**: Set up proper evaluation metrics beyond loss
5. **Deployment**: Learn to serve your models for inference

For detailed implementation explanations, see our [code analysis documentation](../code-analysis/README.md) and [concept guides](../concepts/README.md).