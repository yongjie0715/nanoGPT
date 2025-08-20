# Project Structure - nanoGPT

## File Organization

### Core Scripts (Root Level)
- `train.py` - Main training script (~300 lines)
- `model.py` - GPT model definition (~300 lines) 
- `sample.py` - Text generation/inference script
- `bench.py` - Performance benchmarking utilities
- `configurator.py` - Configuration management utilities

### Configuration System
- **Location**: `config/` directory
- **Format**: Python files with variable assignments
- **Naming**: `train_<dataset>.py` for training configs, `eval_<model>.py` for evaluation
- **Examples**: `train_shakespeare_char.py`, `finetune_shakespeare.py`, `eval_gpt2.py`
- **Pattern**: Override default values defined in main scripts

### Data Organization
- **Location**: `data/` directory with subdirectories per dataset
- **Structure**: `data/<dataset_name>/`
  - `prepare.py` - Data preprocessing script
  - `readme.md` - Dataset documentation
  - `train.bin`, `val.bin` - Processed binary data files
  - `meta.pkl` - Metadata (vocab size, tokenizer info)

### Output Management
- **Checkpoints**: `out-<experiment-name>/` directories
- **Default**: `out/` for standard training runs
- **Contents**: `ckpt.pt` (model checkpoint), logs, config snapshots

### Documentation
- **Location**: `docs/` directory
- **Structure**:
  - `guides/` - User-facing tutorials and how-tos
  - `concepts/` - Technical explanations of algorithms/architecture
  - `code-analysis/` - Detailed code documentation
  - `validation/` - Documentation validation and QA system

### Testing
- **Integration**: `test_integration.py` for end-to-end testing
- **Quick tests**: `quick_test.py` for rapid validation
- **Validation system**: `docs/validation/` for documentation testing

## Naming Conventions

### Files
- **Scripts**: Descriptive verbs (train.py, sample.py, bench.py)
- **Configs**: `<action>_<dataset>.py` pattern
- **Data**: Standard names (prepare.py, train.bin, val.bin, meta.pkl)

### Variables/Functions
- **Configuration**: Snake_case matching command-line args
- **Classes**: PascalCase (GPT, CausalSelfAttention, LayerNorm)
- **Functions**: Snake_case, descriptive names
- **Constants**: UPPER_CASE for true constants

### Directories
- **Datasets**: Lowercase with underscores (`shakespeare_char`)
- **Output**: Descriptive with hyphens (`out-shakespeare-char`)
- **Code**: Lowercase (`config`, `docs`, `data`)

## Code Patterns

### Configuration Management
- Default values defined in main scripts
- Config files override defaults via variable assignment
- Command-line args take precedence over config files
- Use `configurator.py` for complex config merging

### Model Checkpointing
- Single checkpoint file: `ckpt.pt`
- Contains: model state, optimizer state, config, iteration count
- Resume training: `init_from='resume'` + specify `out_dir`
- Load pretrained: `init_from='gpt2'` or specific checkpoint path

### Data Processing
- Each dataset gets its own subdirectory
- Standard `prepare.py` script for preprocessing
- Binary format (uint16) for efficient loading
- Metadata in pickle files for tokenizer info

### Logging and Monitoring
- Console logging with configurable intervals
- Optional Weights & Biases integration
- Training metrics: loss, learning rate, tokens/sec
- Validation runs at specified intervals

## Extension Patterns

### Adding New Datasets
1. Create `data/<dataset_name>/` directory
2. Implement `prepare.py` following existing patterns
3. Create corresponding config file in `config/`
4. Test with small-scale training run

### Adding New Features
- Extend existing scripts rather than creating new ones
- Add configuration options with sensible defaults
- Maintain backward compatibility
- Follow existing code style and patterns

### Model Modifications
- Modify `model.py` directly (keep it monolithic)
- Add configuration parameters to GPTConfig class
- Ensure compatibility with checkpoint loading/saving
- Test with both training and inference

## Development Guidelines
- **Keep it simple**: Favor clarity over clever abstractions
- **One file, one purpose**: Core functionality should remain in single files
- **Configuration-driven**: Make behavior configurable rather than hardcoded
- **Test compatibility**: Ensure changes work across hardware configurations
- **Document changes**: Update relevant documentation when modifying behavior