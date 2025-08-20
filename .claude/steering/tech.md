# Technology Stack - nanoGPT

## Core Technologies
- **Language**: Python 3.x
- **Deep Learning Framework**: PyTorch (with PyTorch 2.0+ compile support)
- **Computation**: CUDA for GPU acceleration, MPS for Apple Silicon

## Dependencies
### Required
- `torch` - Core PyTorch framework
- `numpy` - Numerical computations
- `transformers` - HuggingFace transformers (for loading GPT-2 checkpoints)
- `datasets` - HuggingFace datasets (for data preprocessing)
- `tiktoken` - OpenAI's BPE tokenizer
- `wandb` - Experiment tracking and logging
- `tqdm` - Progress bars

### Development/Testing
- Standard Python testing libraries
- Integration test framework for model validation

## Architecture Decisions

### Model Implementation
- **Pure PyTorch**: No high-level abstractions, direct nn.Module implementations
- **Flash Attention**: Automatic detection and use of PyTorch 2.0's scaled_dot_product_attention
- **Mixed precision**: Support for automatic mixed precision training
- **Gradient checkpointing**: Memory-efficient training for large models

### Training Infrastructure
- **Distributed Data Parallel (DDP)**: Multi-GPU training support
- **Gradient accumulation**: Simulate larger batch sizes on limited hardware
- **Checkpointing**: Automatic model saving and resumption
- **Configuration**: Python-based config files for maximum flexibility

### Data Pipeline
- **Binary format**: Efficient storage as uint16 arrays for fast loading
- **Memory mapping**: Efficient data access for large datasets
- **Tokenization**: Support for both character-level and BPE tokenization
- **Custom datasets**: Simple prepare.py pattern for new data sources

## Performance Considerations
- **PyTorch compile**: Automatic graph optimization when available
- **Hardware compatibility**: CPU, single GPU, multi-GPU, Apple Silicon (MPS)
- **Memory efficiency**: Configurable block size and batch size for various hardware
- **Training speed**: Optimized for reasonable training times on academic hardware

## Technical Constraints
- **Simplicity first**: Avoid complex abstractions that obscure the core logic
- **Educational focus**: Code must remain readable and hackable
- **Hardware accessibility**: Support training on laptops to large clusters
- **Reproducibility**: Deterministic training with proper seeding

## Integration Points
- **Weights & Biases**: Optional experiment tracking and visualization
- **HuggingFace Hub**: Loading pretrained GPT-2 checkpoints
- **OpenAI compatibility**: Can initialize from OpenAI GPT-2 weights
- **Custom tokenizers**: Extensible tokenization system

## Development Tools
- **Benchmarking**: bench.py for performance profiling
- **Testing**: Integration tests for model correctness
- **Validation**: Documentation validation system
- **Dashboard**: Web-based training monitoring