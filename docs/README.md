# nanoGPT Documentation

Welcome to the comprehensive documentation for nanoGPT - a minimal, educational implementation of GPT (Generative Pre-trained Transformer) that prioritizes simplicity and readability. This documentation provides detailed line-by-line explanations of every component in the codebase to help developers understand how modern language models work.

## What is nanoGPT?

nanoGPT is the simplest, fastest repository for training/finetuning medium-sized GPTs. It's a rewrite of minGPT that prioritizes practical implementation over educational abstractions. The entire codebase consists of just a few hundred lines of clean, readable Python code that can reproduce GPT-2 results.

## Documentation Structure

This documentation is organized into three main sections, with a comprehensive [Concept Index](concept-index.md) for quick navigation between related topics:

### 📋 Code Analysis
Detailed line-by-line breakdowns of each source file in the nanoGPT codebase:

- **[train.py Analysis](code-analysis/train-py-analysis.md)** - Complete breakdown of the training pipeline, including distributed training, optimization, and checkpointing → *Related: [Training Concepts](concepts/training-process.md)*
- **[model.py Analysis](code-analysis/model-py-analysis.md)** - Detailed explanation of the GPT model architecture, attention mechanisms, and forward pass → *Related: [GPT Architecture](concepts/gpt-architecture.md)*
- **[Data Preparation Analysis](code-analysis/prepare-py-analysis.md)** - How raw text is processed into training-ready format using tokenization and memory mapping → *Related: [Data Pipeline](concepts/data-pipeline.md)*
- **[Configuration System Analysis](code-analysis/configurator-py-analysis.md)** - Understanding the parameter override system and configuration management → *Related: [Parameter Reference](code-analysis/configuration-parameter-reference.md)*
- **[Sampling & Inference Analysis](code-analysis/sample-py-analysis.md)** - Text generation pipeline and autoregressive sampling strategies → *Related: [Text Generation](concepts/text-generation-algorithms.md)*

### 🧠 Concepts
High-level explanations of the key concepts and algorithms:

- **[GPT Architecture](concepts/gpt-architecture.md)** - Understanding transformer architecture, attention mechanisms, and causal modeling
- **[Training Process](concepts/training-process.md)** - Language modeling objective, gradient descent, and optimization strategies
- **[Data Pipeline](concepts/data-pipeline.md)** - Tokenization, BPE encoding, and efficient data loading
- **[Distributed Training](concepts/distributed-training.md)** - Multi-GPU coordination and distributed data parallel training

### 📚 Guides
Practical guides for using and extending nanoGPT:

- **[Getting Started](guides/getting-started.md)** - Step-by-step tutorial for running the code and understanding outputs
- **[Customization Guide](guides/customization.md)** - How to modify model architecture and add new datasets
- **[Troubleshooting](guides/troubleshooting.md)** - Common issues and solutions

## Learning Paths

### For Beginners
If you're new to language models and transformers:
1. Start with [GPT Architecture Concepts](concepts/gpt-architecture.md)
2. Read the [Getting Started Guide](guides/getting-started.md)
3. Follow the [model.py Analysis](code-analysis/model-py-analysis.md)
4. Explore [Training Process Concepts](concepts/training-process.md)

### For ML Engineers
If you have machine learning experience but are new to transformers:
1. Review [GPT Architecture Concepts](concepts/gpt-architecture.md)
2. Dive into [train.py Analysis](code-analysis/train-py-analysis.md)
3. Study [Data Pipeline Concepts](concepts/data-pipeline.md)
4. Explore [Distributed Training](concepts/distributed-training.md)

### For Researchers
If you want to understand implementation details for research purposes:
1. Start with [model.py Analysis](code-analysis/model-py-analysis.md)
2. Study [train.py Analysis](code-analysis/train-py-analysis.md)
3. Review [Configuration System Analysis](code-analysis/configurator-py-analysis.md)
4. Read the [Customization Guide](guides/customization.md)

## Key Features of This Documentation

### Line-by-Line Analysis
Every major code section includes detailed explanations of what each line does and why it's implemented that way.

### Conceptual Connections
Code explanations are linked to underlying machine learning and deep learning concepts, helping you understand both the "what" and the "why."

### Practical Examples
Real code snippets and execution examples help you see how concepts translate to working implementations.

### Cross-References
Extensive linking between related concepts across different files helps you understand how components work together. Use the [Concept Index](concept-index.md) to quickly find related information across all documentation.

### Multiple Difficulty Levels
Content is organized to support different experience levels, from beginners to advanced practitioners.

## Prerequisites

To get the most out of this documentation, you should have:

- **Basic Python knowledge** - Understanding of classes, functions, and common libraries
- **PyTorch familiarity** - Basic understanding of tensors, autograd, and neural network modules
- **Machine Learning basics** - Understanding of neural networks, gradient descent, and training loops
- **Optional: Transformer knowledge** - Familiarity with attention mechanisms helps but isn't required

## How to Use This Documentation

### Reading Online
All documentation is written in Markdown and can be read directly in your browser or text editor.

### Following Along with Code
For the best learning experience, have the nanoGPT source code open alongside the documentation. Each analysis section references specific line numbers and code blocks.

### Navigation Tools
- **[Glossary](glossary.md)**: Look up technical terms and their definitions
- **[Concept Map](concept-map.md)**: Understand how components relate to each other
- **[Concept Index](concept-index.md)**: Find concepts across all documentation files

### Hands-On Learning
Try running the code examples and experiments suggested throughout the documentation. The [Getting Started Guide](guides/getting-started.md) provides setup instructions.

## Contributing

This documentation is designed to be a living resource that grows and improves over time. If you find errors, have suggestions for improvements, or want to contribute additional explanations, please see our contribution guidelines.

## Quick Navigation

| Component | Purpose | Difficulty | Time to Read | Related Concepts |
|-----------|---------|------------|--------------|------------------|
| [model.py](code-analysis/model-py-analysis.md) | GPT architecture | Intermediate | 45 min | [Architecture](concepts/gpt-architecture.md) |
| [train.py](code-analysis/train-py-analysis.md) | Training pipeline | Advanced | 60 min | [Training Process](concepts/training-process.md) |
| [Data prep](code-analysis/prepare-py-analysis.md) | Data processing | Beginner | 30 min | [Data Pipeline](concepts/data-pipeline.md) |
| [sample.py](code-analysis/sample-py-analysis.md) | Text generation | Intermediate | 30 min | [Text Generation](concepts/text-generation-algorithms.md) |
| [configurator.py](code-analysis/configurator-py-analysis.md) | Configuration | Beginner | 20 min | [Config Reference](code-analysis/configuration-parameter-reference.md) |

**Quick Access**: [Concept Index](concept-index.md) | [Glossary](glossary.md) | [Concept Map](concept-map.md) | [Cross-Reference Map](concept-index.md#cross-file-relationships)

## Additional Resources

- **Original nanoGPT Repository**: [github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- **Andrej Karpathy's GPT Video**: [Zero to Hero GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- **Attention Is All You Need Paper**: [Original Transformer Paper](https://arxiv.org/abs/1706.03762)
- **GPT-2 Paper**: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

*This documentation was created to make the elegant simplicity of nanoGPT accessible to developers at all levels. Whether you're learning about transformers for the first time or implementing your own language model, we hope this resource helps you understand not just what the code does, but why it works.*