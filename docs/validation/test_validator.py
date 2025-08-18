#!/usr/bin/env python3
"""
Test suite for the documentation validation system
"""

import os
import tempfile
import unittest
from pathlib import Path
from doc_validator import DocumentationValidator, ValidationResult


class TestDocumentationValidator(unittest.TestCase):
    """Test cases for the DocumentationValidator class"""
    
    def setUp(self):
        """Set up test environment with temporary directory"""
        self.test_dir = tempfile.mkdtemp()
        self.docs_path = Path(self.test_dir) / "docs"
        self.docs_path.mkdir()
        
        # Create a test glossary
        glossary_content = """# Glossary

**GPT**: Generative Pre-trained Transformer
**PyTorch**: Deep learning framework
**Attention**: Mechanism for focusing on relevant parts of input
"""
        with open(self.docs_path / "glossary.md", 'w') as f:
            f.write(glossary_content)
        
        self.validator = DocumentationValidator(str(self.docs_path))
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_valid_python_code(self):
        """Test validation of syntactically correct Python code"""
        content = """# Test Document

```python
def hello_world():
    print("Hello, World!")
    return 42
```
"""
        test_file = self.docs_path / "test.md"
        with open(test_file, 'w') as f:
            f.write(content)
        
        self.validator._validate_file(test_file)
        
        # Should have one passing code syntax check
        code_results = [r for r in self.validator.results if r.check_type == "code_syntax"]
        self.assertEqual(len(code_results), 1)
        self.assertEqual(code_results[0].status, "pass")
    
    def test_invalid_python_code(self):
        """Test validation of syntactically incorrect Python code"""
        content = """# Test Document

```python
def broken_function(
    print("Missing closing parenthesis"
    return invalid syntax
```
"""
        test_file = self.docs_path / "test.md"
        with open(test_file, 'w') as f:
            f.write(content)
        
        self.validator._validate_file(test_file)
        
        # Should have one failing code syntax check
        code_results = [r for r in self.validator.results if r.check_type == "code_syntax"]
        self.assertEqual(len(code_results), 1)
        self.assertEqual(code_results[0].status, "fail")
    
    def test_valid_internal_link(self):
        """Test validation of valid internal links"""
        # Create target file
        target_content = "# Target Document\n\nThis is the target."
        target_file = self.docs_path / "target.md"
        with open(target_file, 'w') as f:
            f.write(target_content)
        
        # Create source file with link
        source_content = """# Source Document

See [target document](target.md) for more info.
"""
        source_file = self.docs_path / "source.md"
        with open(source_file, 'w') as f:
            f.write(source_content)
        
        self.validator._validate_file(source_file)
        
        # Should have one passing internal link check
        link_results = [r for r in self.validator.results if r.check_type == "internal_link"]
        self.assertEqual(len(link_results), 1)
        self.assertEqual(link_results[0].status, "pass")
    
    def test_broken_internal_link(self):
        """Test validation of broken internal links"""
        content = """# Test Document

See [missing document](nonexistent.md) for more info.
"""
        test_file = self.docs_path / "test.md"
        with open(test_file, 'w') as f:
            f.write(content)
        
        self.validator._validate_file(test_file)
        
        # Should have one failing internal link check
        link_results = [r for r in self.validator.results if r.check_type == "internal_link"]
        self.assertEqual(len(link_results), 1)
        self.assertEqual(link_results[0].status, "fail")
    
    def test_valid_anchor_link(self):
        """Test validation of valid anchor links"""
        content = """# Test Document

## Section One

Some content here.

## Section Two

See [Section One](#section-one) above.
"""
        test_file = self.docs_path / "test.md"
        with open(test_file, 'w') as f:
            f.write(content)
        
        self.validator._validate_file(test_file)
        
        # Should have one passing anchor link check
        anchor_results = [r for r in self.validator.results if r.check_type == "anchor_link"]
        self.assertEqual(len(anchor_results), 1)
        self.assertEqual(anchor_results[0].status, "pass")
    
    def test_broken_anchor_link(self):
        """Test validation of broken anchor links"""
        content = """# Test Document

## Section One

See [nonexistent section](#nonexistent-section).
"""
        test_file = self.docs_path / "test.md"
        with open(test_file, 'w') as f:
            f.write(content)
        
        self.validator._validate_file(test_file)
        
        # Should have one failing anchor link check
        anchor_results = [r for r in self.validator.results if r.check_type == "anchor_link"]
        self.assertEqual(len(anchor_results), 1)
        self.assertEqual(anchor_results[0].status, "fail")
    
    def test_heading_structure_validation(self):
        """Test validation of heading hierarchy"""
        content = """# Main Title

## Section One

#### Subsection (skips h3)

This should generate a warning.
"""
        test_file = self.docs_path / "test.md"
        with open(test_file, 'w') as f:
            f.write(content)
        
        self.validator._validate_file(test_file)
        
        # Should have one warning about heading structure
        heading_results = [r for r in self.validator.results if r.check_type == "heading_structure"]
        self.assertEqual(len(heading_results), 1)
        self.assertEqual(heading_results[0].status, "warning")
    
    def test_unmatched_code_blocks(self):
        """Test detection of unmatched code block delimiters"""
        content = """# Test Document

```python
def test():
    pass

Missing closing delimiter.
"""
        test_file = self.docs_path / "test.md"
        with open(test_file, 'w') as f:
            f.write(content)
        
        self.validator._validate_file(test_file)
        
        # Should have one failing code block check
        code_block_results = [r for r in self.validator.results if r.check_type == "code_blocks"]
        self.assertEqual(len(code_block_results), 1)
        self.assertEqual(code_block_results[0].status, "fail")
    
    def test_terminology_consistency(self):
        """Test terminology consistency checking"""
        content1 = """# Document One

This uses PyTorch for training.
"""
        content2 = """# Document Two

This uses pytorch for inference.
"""
        
        file1 = self.docs_path / "doc1.md"
        file2 = self.docs_path / "doc2.md"
        
        with open(file1, 'w') as f:
            f.write(content1)
        with open(file2, 'w') as f:
            f.write(content2)
        
        md_files = [file1, file2]
        self.validator._validate_terminology_consistency(md_files)
        
        # Should have warnings about inconsistent terminology
        term_results = [r for r in self.validator.results if r.check_type == "terminology"]
        self.assertTrue(len(term_results) > 0)
    
    def test_report_generation(self):
        """Test generation of validation report"""
        # Add some test results
        self.validator.results = [
            ValidationResult("test.md", "code_syntax", "pass", "Valid code"),
            ValidationResult("test.md", "internal_link", "fail", "Broken link"),
            ValidationResult("test.md", "terminology", "warning", "Inconsistent term")
        ]
        
        report = self.validator.generate_report("test_report.json")
        
        # Check report structure
        self.assertIn("summary", report)
        self.assertIn("results_by_file", report)
        self.assertIn("results_by_type", report)
        
        # Check summary counts
        self.assertEqual(report["summary"]["total_checks"], 3)
        self.assertEqual(report["summary"]["passed"], 1)
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(report["summary"]["warnings"], 1)


class TestValidationIntegration(unittest.TestCase):
    """Integration tests for the complete validation system"""
    
    def test_full_validation_run(self):
        """Test a complete validation run on sample documentation"""
        # This test would run against the actual docs directory
        # Skip if docs directory doesn't exist
        if not Path("docs").exists():
            self.skipTest("docs directory not found")
        
        validator = DocumentationValidator("docs")
        results = validator.validate_all()
        
        # Should have some results
        self.assertGreater(len(results), 0)
        
        # Generate report
        report = validator.generate_report("integration_test_report.json")
        
        # Report should have expected structure
        self.assertIn("summary", report)
        self.assertIn("results_by_file", report)
        self.assertIn("results_by_type", report)


if __name__ == "__main__":
    unittest.main()