# Documentation Validation System

This directory contains a comprehensive validation system for the nanoGPT documentation. The system ensures code examples are syntactically correct, links work properly, and terminology is used consistently across all documentation files.

## Features

### Code Validation
- **Syntax Checking**: Validates Python code blocks for syntax errors
- **Shell Command Safety**: Warns about potentially dangerous shell commands
- **Language Support**: Supports Python, Bash, and Shell code blocks

### Link Validation
- **Internal Links**: Verifies relative links to other documentation files
- **Anchor Links**: Validates internal document anchors and headers
- **External Links**: Checks HTTP/HTTPS links for accessibility
- **Cross-References**: Ensures proper linking between documentation sections

### Structure Validation
- **Heading Hierarchy**: Checks for proper heading level progression
- **Code Block Matching**: Ensures code block delimiters are properly matched
- **Document Structure**: Validates overall document organization

### Consistency Checks
- **Terminology**: Ensures consistent use of technical terms across files
- **Cross-References**: Identifies orphaned files and missing references
- **Style Guidelines**: Checks adherence to documentation standards

## Quick Start

### Prerequisites
- Python 3.6 or higher
- Standard library modules (no external dependencies required)

### Running Validation

```bash
# Navigate to the validation directory
cd docs/validation

# Run full validation with HTML report
make validate

# Run quick validation (JSON only)
make validate-quick

# Run in strict mode (fail on warnings)
make validate-strict

# Run validation system tests
make test
```

### Manual Execution

```bash
# Basic validation
python3 run_validation.py --docs-root ../..

# With custom configuration
python3 run_validation.py --docs-root ../.. --config validation_config.json

# Generate only JSON report
python3 run_validation.py --docs-root ../.. --no-html

# Strict mode (fail on warnings)
python3 run_validation.py --docs-root ../.. --strict
```

## Configuration

The validation system is configured through `validation_config.json`:

```json
{
  "validation_rules": {
    "code_syntax": {
      "enabled": true,
      "languages": ["python", "bash", "shell", "sh"],
      "strict_mode": true
    },
    "link_checking": {
      "enabled": true,
      "check_external_links": true,
      "external_link_timeout": 10
    },
    "terminology_consistency": {
      "enabled": true,
      "standard_terms": {
        "gpt": "GPT",
        "pytorch": "PyTorch"
      }
    }
  },
  "quality_thresholds": {
    "max_failures": 0,
    "max_warnings": 10,
    "min_pass_rate": 0.95
  }
}
```

### Configuration Options

#### Validation Rules
- `code_syntax`: Enable/disable code syntax checking
- `link_checking`: Configure link validation behavior
- `structure_validation`: Control document structure checks
- `terminology_consistency`: Set standard terminology
- `cross_reference_validation`: Configure cross-reference checking

#### Quality Thresholds
- `max_failures`: Maximum allowed failed checks
- `max_warnings`: Maximum allowed warnings
- `min_pass_rate`: Minimum percentage of checks that must pass

## Output Reports

### JSON Report (`validation_report.json`)
Structured data suitable for programmatic processing:

```json
{
  "summary": {
    "total_checks": 150,
    "passed": 145,
    "failed": 2,
    "warnings": 3
  },
  "results_by_file": {
    "docs/README.md": [
      {
        "check_type": "code_syntax",
        "status": "pass",
        "message": "Python code block is syntactically correct",
        "line_number": 25
      }
    ]
  }
}
```

### HTML Report (`validation_report.html`)
Human-readable report with:
- Visual summary dashboard
- Categorized results (failures, warnings, by file)
- Clickable navigation
- Responsive design for mobile/desktop

## Validation Checks

### Code Syntax Validation
Checks all code blocks for syntax correctness:

```markdown
```python
def example():
    return "This will be validated"
```
```

### Link Validation
Validates all types of links:

```markdown
[Internal link](../concepts/gpt-architecture.md)
[Anchor link](#section-header)
[External link](https://pytorch.org)
```

### Structure Validation
Ensures proper document structure:

```markdown
# Main Title (H1)
## Section (H2)
### Subsection (H3)
#### Details (H4) - Warning: skips H3 level
```

### Terminology Consistency
Checks for consistent term usage:

```markdown
<!-- Consistent -->
PyTorch is used for training.
The PyTorch framework provides...

<!-- Inconsistent - generates warning -->
PyTorch is used for training.
The pytorch framework provides...
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Documentation Validation
on: [push, pull_request]

jobs:
  validate-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Validate Documentation
        run: |
          cd docs/validation
          make ci
```

### Pre-commit Hook

```bash
#!/bin/sh
# .git/hooks/pre-commit
cd docs/validation
make validate-quick
```

## Extending the Validation System

### Adding New Check Types

1. Extend the `DocumentationValidator` class:

```python
def _validate_custom_check(self, file_path: Path, content: str):
    """Add your custom validation logic"""
    # Implementation here
    pass
```

2. Call your method in `_validate_file()`:

```python
def _validate_file(self, file_path: Path):
    # ... existing code ...
    self._validate_custom_check(file_path, content)
```

3. Update configuration schema in `validation_config.json`

### Custom Terminology Rules

Add terms to the configuration:

```json
{
  "validation_rules": {
    "terminology_consistency": {
      "standard_terms": {
        "your_term": "Preferred Format",
        "another_term": "Another Format"
      }
    }
  }
}
```

## Troubleshooting

### Common Issues

**External link timeouts**
- Increase `external_link_timeout` in configuration
- Add problematic domains to `ignore_external_domains`

**False positive syntax errors**
- Check code block language specification
- Ensure proper indentation in code blocks

**Missing anchor links**
- Verify header text matches anchor format
- Check for special characters in headers

### Debug Mode

Run with verbose output:

```bash
python3 doc_validator.py --docs-root ../.. 2>&1 | tee debug.log
```

### Performance Optimization

For large documentation sets:
- Disable external link checking for faster runs
- Use `--no-html` for CI environments
- Configure file exclusion patterns

## Contributing

1. Run tests before submitting changes:
   ```bash
   make test
   ```

2. Validate your changes:
   ```bash
   make validate-strict
   ```

3. Update configuration schema if adding new features

4. Add test cases for new validation rules

## License

This validation system is part of the nanoGPT project and follows the same license terms.