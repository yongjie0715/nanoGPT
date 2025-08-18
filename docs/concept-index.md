# Concept Index

> **Navigation**: [Documentation Home](README.md) | [Glossary](glossary.md) | [Concept Map](concept-map.md)

This index provides quick access to key concepts and their locations across the nanoGPT documentation. Use this to find related information and navigate between different sections.

## How to Use This Index

- **Find by Concept**: Look up specific topics in the sections below
- **Find by File**: Use the [Cross-File Relationships](#cross-file-relationships) section
- **Find Definitions**: Check the [Glossary](glossary.md) for term definitions
- **Understand Relationships**: See the [Concept Map](concept-map.md) for visual connections

## Architecture Concepts

### Transformer Architecture
- **Core Concepts**: [GPT Architecture](concepts/gpt-architecture.md#transformer-architecture-foundation)
- **Implementation**: [model.py Analysis - GPT Class](code-analysis/model-py-analysis.md#gpt-model-class-structure)
- **Configuration**: [GPTConfig Parameters](code-analysis/model-py-analysis.md#gptconfig-dataclass)

### Attention Mechanisms
- **Theory**: [Multi-Head Self-Attention](concepts/gpt-architecture.md#multi-head-self-attention)
- **Implementation**: [CausalSelfAttention Class](code-analysis/model-py-analysis.md#causalselfattention-implementation)
- **Causal Masking**: [Causal Attention Concepts](concepts/gpt-architecture.md#causal-self-attention)
- **Flash Attention**: [Attention Optimization](code-analysis/model-py-analysis.md#flash-attention-optimization)

### Model Components
- **Embeddings**: 
  - Theory: [Positional Encoding](concepts/gpt-architecture.md#positional-encoding)
  - Implementation: [Token & Position Embeddings](code-analysis/model-py-analysis.md#embedding-layers)
- **MLP Blocks**:
  - Theory: [Feed-Forward Networks](concepts/gpt-architecture.md#feed-forward-networks)
  - Implementation: [MLP Class Analysis](code-analysis/model-py-analysis.md#mlp-feed-forward-block)
- **Layer Normalization**:
  - Implementation: [LayerNorm Usage](code-analysis/model-py-analysis.md#layer-normalization)

## Training Concepts

### Language Modeling
- **Objective**: [Next-Token Prediction](concepts/training-process.md#next-token-prediction)
- **Loss Function**: [Cross-Entropy Loss](concepts/training-process.md#cross-entropy-loss)
- **Implementation**: [Training Loop](code-analysis/train-py-analysis.md#training-loop-implementation)

### Optimization
- **Theory**: [Gradient Descent](concepts/training-process.md#gradient-descent-and-optimization)
- **AdamW Optimizer**: [Optimizer Configuration](code-analysis/train-py-analysis.md#optimizer-setup)
- **Learning Rate Scheduling**: [LR Decay Implementation](code-analysis/train-py-analysis.md#learning-rate-scheduling)
- **Gradient Accumulation**: [Batch Processing](code-analysis/train-py-analysis.md#gradient-accumulation)

### Distributed Training
- **Concepts**: [Multi-GPU Training](concepts/training-process.md#distributed-training)
- **DDP Implementation**: [Distributed Data Parallel](code-analysis/train-py-analysis.md#distributed-training-setup)
- **Process Coordination**: [Master Process Logic](code-analysis/train-py-analysis.md#master-process-coordination)

## Data Processing

### Tokenization
- **BPE Encoding**: [Byte-Pair Encoding](concepts/data-pipeline.md#tokenization-concepts)
- **Implementation**: [Tokenizer Usage](code-analysis/prepare-py-analysis.md#tokenization-pipeline)
- **Vocabulary**: [Token Management](code-analysis/prepare-py-analysis.md#vocabulary-handling)

### Data Pipeline
- **Memory Mapping**: [Efficient Data Loading](concepts/data-pipeline.md#memory-optimization)
- **Implementation**: [Binary File Creation](code-analysis/prepare-py-analysis.md#binary-file-generation)
- **Batch Generation**: [Data Loading](code-analysis/prepare-py-analysis.md#batch-processing)

### Dataset Preparation
- **OpenWebText**: [Dataset Processing](code-analysis/prepare-py-analysis.md#openwebtext-preparation)
- **Train/Val Split**: [Data Splitting](code-analysis/prepare-py-analysis.md#dataset-splitting)

## Text Generation

### Sampling Strategies
- **Autoregressive Generation**: [Generation Process](concepts/text-generation-algorithms.md#autoregressive-generation)
- **Temperature Scaling**: [Sampling Control](concepts/text-generation-algorithms.md#temperature-scaling)
- **Top-k Filtering**: [Sampling Strategies](concepts/text-generation-algorithms.md#top-k-sampling)
- **Implementation**: [sample.py Analysis](code-analysis/sample-py-analysis.md#text-generation-pipeline)

### Inference Optimization
- **Model Compilation**: [Inference Setup](code-analysis/sample-py-analysis.md#model-compilation)
- **Device Management**: [GPU Optimization](code-analysis/sample-py-analysis.md#device-selection)

## Configuration System

### Parameter Management
- **Config Files**: [Configuration System](code-analysis/configurator-py-analysis.md#configuration-system)
- **Parameter Override**: [Command Line Args](code-analysis/configurator-py-analysis.md#parameter-override-system)
- **Validation**: [Type Checking](code-analysis/configurator-py-analysis.md#parameter-validation)

### Training Parameters
- **Model Config**: [Architecture Parameters](code-analysis/configuration-parameter-reference.md#model-parameters)
- **Training Config**: [Optimization Parameters](code-analysis/configuration-parameter-reference.md#training-parameters)
- **System Config**: [Hardware Parameters](code-analysis/configuration-parameter-reference.md#system-parameters)

## Implementation Details

### PyTorch Patterns
- **Module Design**: [nn.Module Usage](code-analysis/model-py-analysis.md#pytorch-module-patterns)
- **Parameter Initialization**: [Weight Initialization](code-analysis/model-py-analysis.md#parameter-initialization)
- **Gradient Management**: [Autograd Usage](code-analysis/train-py-analysis.md#gradient-computation)

### Performance Optimization
- **Memory Management**: [GPU Memory](code-analysis/train-py-analysis.md#memory-optimization)
- **Compilation**: [torch.compile Usage](code-analysis/train-py-analysis.md#model-compilation)
- **Mixed Precision**: [AMP Training](code-analysis/train-py-analysis.md#automatic-mixed-precision)

### Checkpointing
- **Model Saving**: [Checkpoint Creation](code-analysis/train-py-analysis.md#checkpoint-management)
- **Resume Training**: [State Restoration](code-analysis/train-py-analysis.md#training-resumption)
- **Pretrained Loading**: [Model Loading](code-analysis/model-py-analysis.md#pretrained-model-loading)

## Cross-File Relationships

### model.py ↔ train.py
- Model instantiation: [GPT Creation](code-analysis/train-py-analysis.md#model-initialization) → [GPT Class](code-analysis/model-py-analysis.md#gpt-model-class-structure)
- Forward pass: [Training Loop](code-analysis/train-py-analysis.md#forward-pass) → [Model Forward](code-analysis/model-py-analysis.md#forward-pass-implementation)

### train.py ↔ configurator.py
- Configuration loading: [Config Setup](code-analysis/train-py-analysis.md#configuration-loading) → [Configurator System](code-analysis/configurator-py-analysis.md#configuration-system)
- Parameter override: [CLI Args](code-analysis/configurator-py-analysis.md#command-line-processing) → [Training Setup](code-analysis/train-py-analysis.md#parameter-initialization)

### prepare.py ↔ train.py
- Data loading: [Batch Generation](code-analysis/train-py-analysis.md#data-loading) → [Data Preparation](code-analysis/prepare-py-analysis.md#batch-processing)
- Memory mapping: [Efficient Loading](code-analysis/prepare-py-analysis.md#memory-mapped-files) → [Training Data](code-analysis/train-py-analysis.md#dataset-handling)

### model.py ↔ sample.py
- Model loading: [Inference Setup](code-analysis/sample-py-analysis.md#model-loading) → [GPT Class](code-analysis/model-py-analysis.md#gpt-model-class-structure)
- Generation: [Text Generation](code-analysis/sample-py-analysis.md#generation-loop) → [Model Forward](code-analysis/model-py-analysis.md#forward-pass-implementation)

## Quick Reference

### Key Files
- **[model.py](code-analysis/model-py-analysis.md)**: GPT architecture implementation
- **[train.py](code-analysis/train-py-analysis.md)**: Training pipeline and optimization
- **[prepare.py](code-analysis/prepare-py-analysis.md)**: Data preprocessing and tokenization
- **[sample.py](code-analysis/sample-py-analysis.md)**: Text generation and inference
- **[configurator.py](code-analysis/configurator-py-analysis.md)**: Configuration management

### Key Concepts
- **[GPT Architecture](concepts/gpt-architecture.md)**: Transformer theory and design
- **[Training Process](concepts/training-process.md)**: Optimization and learning
- **[Data Pipeline](concepts/data-pipeline.md)**: Text processing and tokenization
- **[Text Generation](concepts/text-generation-algorithms.md)**: Sampling and inference

---

*Use Ctrl+F (Cmd+F on Mac) to search for specific terms within this index.*