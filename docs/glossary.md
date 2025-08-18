# nanoGPT Glossary

This glossary provides definitions for technical terms used throughout the nanoGPT documentation. Terms are organized alphabetically with cross-references to relevant documentation sections.

## A

### Attention Mechanism
A neural network component that allows the model to focus on different parts of the input sequence when processing each token. In GPT, this is implemented as causal self-attention.
- **Implementation**: [CausalSelfAttention Class](code-analysis/model-py-analysis.md#causalselfattention-implementation)
- **Theory**: [Attention Concepts](concepts/gpt-architecture.md#attention-mechanisms-deep-dive)

### Autoregressive Generation
A text generation method where each new token is predicted based on all previously generated tokens in the sequence.
- **Theory**: [Autoregressive Generation](concepts/text-generation-algorithms.md#autoregressive-generation)
- **Implementation**: [Generation Loop](code-analysis/sample-py-analysis.md#generation-loop)

### Automatic Mixed Precision (AMP)
A training technique that uses both 16-bit and 32-bit floating-point representations to speed up training while maintaining model accuracy.
- **Implementation**: [AMP Training](code-analysis/train-py-analysis.md#automatic-mixed-precision)

## B

### Batch Size
The number of training examples processed simultaneously during one forward/backward pass.
- **Configuration**: [Training Parameters](code-analysis/configuration-parameter-reference.md#training-parameters)
- **Implementation**: [Batch Processing](code-analysis/train-py-analysis.md#batch-processing)

### Block Size
The maximum sequence length (context window) that the model can process at once. Also called context length.
- **Configuration**: [Model Parameters](code-analysis/configuration-parameter-reference.md#model-parameters)
- **Implementation**: [GPTConfig](code-analysis/model-py-analysis.md#gptconfig-dataclass)

### BPE (Byte-Pair Encoding)
A tokenization algorithm that breaks text into subword units, balancing vocabulary size with meaningful token representation.
- **Theory**: [Tokenization Concepts](concepts/data-pipeline.md#tokenization-concepts)
- **Implementation**: [Tokenization Pipeline](code-analysis/prepare-py-analysis.md#tokenization-pipeline)

## C

### Causal Masking
A technique that prevents the model from attending to future tokens during training, ensuring the model only uses past context for predictions.
- **Theory**: [Causal Self-Attention](concepts/gpt-architecture.md#causal-self-attention)
- **Implementation**: [Attention Masking](code-analysis/model-py-analysis.md#causal-masking-implementation)

### Checkpoint
A saved snapshot of the model's state during training, including weights, optimizer state, and training metadata.
- **Implementation**: [Checkpoint Management](code-analysis/train-py-analysis.md#checkpoint-management)
- **Usage**: [Model Loading](code-analysis/sample-py-analysis.md#model-loading)

### Cross-Entropy Loss
The loss function used for training language models, measuring the difference between predicted and actual token probabilities.
- **Theory**: [Cross-Entropy Loss](concepts/training-process.md#cross-entropy-loss)
- **Implementation**: [Loss Calculation](code-analysis/train-py-analysis.md#loss-calculation)

## D

### DDP (Distributed Data Parallel)
PyTorch's method for training models across multiple GPUs or nodes by distributing batches and synchronizing gradients.
- **Theory**: [Distributed Training](concepts/training-process.md#distributed-training)
- **Implementation**: [DDP Setup](code-analysis/train-py-analysis.md#distributed-training-setup)

### Decoder-Only Architecture
A transformer variant that uses only the decoder portion, making it suitable for autoregressive text generation.
- **Theory**: [Decoder-Only Design](concepts/gpt-architecture.md#decoder-only-architecture)
- **Implementation**: [GPT Class](code-analysis/model-py-analysis.md#gpt-model-class-structure)

### Dropout
A regularization technique that randomly sets some neurons to zero during training to prevent overfitting.
- **Configuration**: [Model Parameters](code-analysis/configuration-parameter-reference.md#model-parameters)
- **Implementation**: [Dropout Usage](code-analysis/model-py-analysis.md#dropout-implementation)

## E

### Embedding
A learned vector representation that maps discrete tokens (words/subwords) to continuous vector space.
- **Theory**: [Token Embeddings](concepts/gpt-architecture.md#token-embeddings)
- **Implementation**: [Embedding Layers](code-analysis/model-py-analysis.md#embedding-layers)

## F

### Flash Attention
An optimized attention implementation that reduces memory usage and increases speed for long sequences.
- **Implementation**: [Flash Attention](code-analysis/model-py-analysis.md#flash-attention-optimization)

### Forward Pass
The process of computing model outputs from inputs by passing data through all layers sequentially.
- **Theory**: [Forward Pass Flow](concepts/training-process.md#forward-pass-flow)
- **Implementation**: [Model Forward](code-analysis/model-py-analysis.md#forward-pass-implementation)

## G

### GPT (Generative Pre-trained Transformer)
A family of autoregressive language models based on the transformer architecture, designed for text generation.
- **Theory**: [GPT Architecture](concepts/gpt-architecture.md)
- **Implementation**: [GPT Class](code-analysis/model-py-analysis.md#gpt-model-class-structure)

### Gradient Accumulation
A technique that simulates larger batch sizes by accumulating gradients over multiple smaller batches before updating parameters.
- **Theory**: [Gradient Accumulation](concepts/training-process.md#gradient-accumulation)
- **Implementation**: [Accumulation Logic](code-analysis/train-py-analysis.md#gradient-accumulation)

### Gradient Clipping
A technique that prevents exploding gradients by limiting the magnitude of gradient updates.
- **Implementation**: [Gradient Clipping](code-analysis/train-py-analysis.md#gradient-clipping)

## H

### Head Dimension
The size of each attention head, calculated as embedding dimension divided by number of heads.
- **Implementation**: [Attention Dimensions](code-analysis/model-py-analysis.md#attention-head-dimensions)

### Hyperparameters
Configuration values that control model architecture and training behavior (learning rate, batch size, etc.).
- **Reference**: [Parameter Guide](code-analysis/configuration-parameter-reference.md)
- **Management**: [Configuration System](code-analysis/configurator-py-analysis.md)

## L

### Language Modeling
The task of predicting the next token in a sequence, which is the primary training objective for GPT models.
- **Theory**: [Language Modeling Objective](concepts/training-process.md#language-modeling-objective)
- **Implementation**: [Training Loop](code-analysis/train-py-analysis.md#training-loop-implementation)

### Layer Normalization
A normalization technique applied to layer inputs to stabilize training and improve convergence.
- **Theory**: [Normalization](concepts/gpt-architecture.md#layer-normalization)
- **Implementation**: [LayerNorm Usage](code-analysis/model-py-analysis.md#layer-normalization)

### Learning Rate
The step size used in gradient descent optimization, controlling how much parameters change during each update.
- **Configuration**: [Optimization Parameters](code-analysis/configuration-parameter-reference.md#optimization-parameters)
- **Scheduling**: [LR Scheduling](code-analysis/train-py-analysis.md#learning-rate-scheduling)

## M

### Memory Mapping
A technique for efficiently loading large datasets by mapping files directly into memory without loading everything at once.
- **Theory**: [Memory Optimization](concepts/data-pipeline.md#memory-optimization)
- **Implementation**: [Memory-Mapped Files](code-analysis/prepare-py-analysis.md#memory-mapped-files)

### MLP (Multi-Layer Perceptron)
The feed-forward network component in each transformer block, consisting of linear layers with activation functions.
- **Theory**: [Feed-Forward Networks](concepts/gpt-architecture.md#feed-forward-networks)
- **Implementation**: [MLP Block](code-analysis/model-py-analysis.md#mlp-feed-forward-block)

### Multi-Head Attention
An attention mechanism that uses multiple parallel attention "heads" to capture different types of relationships.
- **Theory**: [Multi-Head Mechanism](concepts/gpt-architecture.md#multi-head-mechanism)
- **Implementation**: [Multi-Head Implementation](code-analysis/model-py-analysis.md#multi-head-attention-implementation)

## N

### Next-Token Prediction
The fundamental task of language models: predicting the most likely next token given previous context.
- **Theory**: [Next-Token Prediction](concepts/training-process.md#next-token-prediction)
- **Implementation**: [Prediction Logic](code-analysis/model-py-analysis.md#output-projection)

## P

### Perplexity
A metric measuring how well a language model predicts text, with lower values indicating better performance.
- **Theory**: [Perplexity Metric](concepts/training-process.md#perplexity-metric)
- **Implementation**: [Evaluation Metrics](code-analysis/train-py-analysis.md#evaluation-metrics)

### Positional Encoding
A method to inject information about token positions into the model, since attention is position-agnostic.
- **Theory**: [Positional Encoding](concepts/gpt-architecture.md#positional-encoding)
- **Implementation**: [Position Embeddings](code-analysis/model-py-analysis.md#position-embeddings)

## Q

### Query, Key, Value (QKV)
The three components of attention computation: Query (what to look for), Key (what's available), Value (actual content).
- **Theory**: [QKV Mechanism](concepts/gpt-architecture.md#mathematical-foundation)
- **Implementation**: [QKV Projections](code-analysis/model-py-analysis.md#qkv-projections)

## R

### Residual Connection
A connection that adds the input of a layer to its output, helping with gradient flow in deep networks.
- **Theory**: [Residual Connections](concepts/gpt-architecture.md#residual-connections)
- **Implementation**: [Residual Implementation](code-analysis/model-py-analysis.md#residual-connections)

## S

### Self-Attention
An attention mechanism where queries, keys, and values all come from the same sequence.
- **Theory**: [Self-Attention](concepts/gpt-architecture.md#self-attention-mechanism)
- **Implementation**: [Attention Implementation](code-analysis/model-py-analysis.md#self-attention-computation)

### Softmax
An activation function that converts logits to probabilities, ensuring outputs sum to 1.
- **Usage**: [Attention Softmax](code-analysis/model-py-analysis.md#attention-softmax)
- **Theory**: [Probability Distributions](concepts/text-generation-algorithms.md#probability-distributions)

## T

### Temperature Scaling
A technique for controlling randomness in text generation by scaling logits before applying softmax.
- **Theory**: [Temperature Scaling](concepts/text-generation-algorithms.md#temperature-scaling)
- **Implementation**: [Sampling Control](code-analysis/sample-py-analysis.md#temperature-sampling)

### Tokenization
The process of converting raw text into discrete tokens that can be processed by the model.
- **Theory**: [Tokenization Process](concepts/data-pipeline.md#tokenization-concepts)
- **Implementation**: [Tokenizer Usage](code-analysis/prepare-py-analysis.md#tokenization-pipeline)

### Top-k Sampling
A sampling strategy that only considers the k most likely next tokens when generating text.
- **Theory**: [Top-k Sampling](concepts/text-generation-algorithms.md#top-k-sampling)
- **Implementation**: [Sampling Strategies](code-analysis/sample-py-analysis.md#sampling-strategies)

### Transformer Block
A single layer of the transformer architecture, containing self-attention and feed-forward components.
- **Theory**: [Transformer Layers](concepts/gpt-architecture.md#transformer-blocks)
- **Implementation**: [Block Implementation](code-analysis/model-py-analysis.md#transformer-block)

## V

### Vocabulary Size
The number of unique tokens in the model's vocabulary, determining the size of embedding and output layers.
- **Configuration**: [Model Parameters](code-analysis/configuration-parameter-reference.md#model-parameters)
- **Implementation**: [Vocabulary Handling](code-analysis/prepare-py-analysis.md#vocabulary-handling)

## W

### Weight Initialization
The process of setting initial values for model parameters before training begins.
- **Theory**: [Initialization Strategies](concepts/training-process.md#weight-initialization)
- **Implementation**: [Parameter Initialization](code-analysis/model-py-analysis.md#parameter-initialization)

---

## Cross-Reference Tags

### By Category
- **Architecture**: Attention, Transformer Block, MLP, Embedding, Positional Encoding
- **Training**: Language Modeling, Cross-Entropy Loss, Gradient Accumulation, Learning Rate
- **Generation**: Autoregressive, Temperature, Top-k, Sampling
- **Optimization**: DDP, AMP, Gradient Clipping, Checkpoint
- **Data**: Tokenization, BPE, Memory Mapping, Batch Size

### By Implementation File
- **model.py**: GPT, Attention, MLP, Embedding, Layer Normalization
- **train.py**: DDP, Gradient Accumulation, Learning Rate, Checkpoint, AMP
- **sample.py**: Temperature, Top-k, Autoregressive Generation
- **prepare.py**: Tokenization, BPE, Memory Mapping, Vocabulary

### By Difficulty Level
- **Beginner**: Tokenization, Batch Size, Checkpoint, Temperature
- **Intermediate**: Attention, MLP, Gradient Accumulation, Top-k
- **Advanced**: DDP, Flash Attention, AMP, Causal Masking

---

*Use Ctrl+F (Cmd+F on Mac) to search for specific terms. Each term includes links to both theoretical explanations and code implementations.*