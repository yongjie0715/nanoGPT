#!/usr/bin/env python3
"""
Test suite for the comprehensive review and refinement system
"""

import os
import tempfile
import unittest
import json
from pathlib import Path
from review_system import DocumentationReviewer, ReviewItem, ComprehensionTest


class TestDocumentationReviewer(unittest.TestCase):
    """Test cases for the DocumentationReviewer class"""
    
    def setUp(self):
        """Set up test environment with temporary directory"""
        self.test_dir = tempfile.mkdtemp()
        self.docs_path = Path(self.test_dir) / "docs"
        self.docs_path.mkdir()
        
        self.reviewer = DocumentationReviewer(str(self.docs_path))
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_technical_accuracy_review(self):
        """Test technical accuracy review functionality"""
        content = """# Test Document

This document uses PyTorch for training:

```python
import torch
model.cuda()  # Deprecated pattern
optimizer = torch.optim.Adam(params)  # Missing parameters
```

The softmax function is calculated as: softmax(x) = exp(x) / sum(exp(x))
"""
        test_file = self.docs_path / "test.md"
        with open(test_file, 'w') as f:
            f.write(content)
        
        technical_items = self.reviewer.perform_technical_accuracy_review()
        
        # Should find issues with deprecated patterns and mathematical formulas
        self.assertGreater(len(technical_items), 0)
        
        # Check for specific issue types
        issue_types = [item.issue_type for item in technical_items]
        self.assertIn("technical_accuracy", issue_types)
    
    def test_clarity_review(self):
        """Test clarity and readability review"""
        content = """# Test Document

This is a very long sentence that contains multiple clauses, several technical terms like autoregressive generation and causal masking, and uses complex grammatical structures that might be difficult for beginners to understand, especially when combined with jargon and technical concepts that are not explained.

## Complex Section

The transformer architecture utilizes multi-head self-attention mechanisms with causal masking for autoregressive generation.
"""
        test_file = self.docs_path / "test.md"
        with open(test_file, 'w') as f:
            f.write(content)
        
        clarity_items = self.reviewer.perform_clarity_review()
        
        # Should find clarity issues
        self.assertGreater(len(clarity_items), 0)
        
        # Check for specific clarity issues
        clarity_issues = [item for item in clarity_items if item.issue_type == "clarity"]
        self.assertGreater(len(clarity_issues), 0)
    
    def test_comprehension_test_generation(self):
        """Test comprehension test generation"""
        content = """# Test Document

Here's a simple Python function:

```python
def calculate_loss(predictions, targets):
    return torch.nn.functional.cross_entropy(predictions, targets)
```

And here's a more complex class:

```python
class GPTModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT(config)
    
    def forward(self, x):
        return self.transformer(x)
```
"""
        test_file = self.docs_path / "test.md"
        with open(test_file, 'w') as f:
            f.write(content)
        
        tests = self.reviewer.generate_comprehension_tests()
        
        # Should generate tests for different complexity levels
        self.assertGreater(len(tests), 0)
        
        # Check for different audience levels
        audience_levels = [test.audience_level for test in tests]
        self.assertTrue(any(level in audience_levels for level in ["beginner", "intermediate", "advanced"]))
    
    def test_feedback_integration_system(self):
        """Test feedback integration system setup"""
        self.reviewer.create_feedback_integration_system()
        
        feedback_dir = self.docs_path / "feedback"
        
        # Check that feedback directory was created
        self.assertTrue(feedback_dir.exists())
        
        # Check that templates were created
        templates = [
            "technical_accuracy_template.md",
            "clarity_template.md", 
            "general_template.md"
        ]
        
        for template in templates:
            self.assertTrue((feedback_dir / template).exists())
        
        # Check that processor script was created
        self.assertTrue((feedback_dir / "process_feedback.py").exists())
    
    def test_comprehensive_report_generation(self):
        """Test comprehensive report generation"""
        # Add some test review items
        self.reviewer.review_items = [
            ReviewItem("test.md", "Section 1", "technical_accuracy", "critical", "Test issue 1"),
            ReviewItem("test.md", "Section 2", "clarity", "major", "Test issue 2"),
            ReviewItem("test.md", "Section 3", "technical_accuracy", "minor", "Test issue 3")
        ]
        
        # Add some test comprehension tests
        self.reviewer.comprehension_tests = [
            ComprehensionTest("test.md", "beginner", ["Q1"], ["A1"], 2.0, ["prereq1"]),
            ComprehensionTest("test.md", "advanced", ["Q2"], ["A2"], 4.5, ["prereq2"])
        ]
        
        report_file = self.test_dir + "/test_report.json"
        report = self.reviewer.generate_comprehensive_report(report_file)
        
        # Check report structure
        self.assertIn("summary", report)
        self.assertIn("comprehension_tests", report)
        self.assertIn("review_items", report)
        self.assertIn("recommendations", report)
        
        # Check summary statistics
        self.assertEqual(report["summary"]["total_review_items"], 3)
        self.assertEqual(report["summary"]["by_severity"]["critical"], 1)
        self.assertEqual(report["summary"]["by_severity"]["major"], 1)
        self.assertEqual(report["summary"]["by_severity"]["minor"], 1)
        
        # Check comprehension test statistics
        self.assertEqual(report["comprehension_tests"]["total_tests"], 2)
        self.assertEqual(report["comprehension_tests"]["by_audience"]["beginner"], 1)
        self.assertEqual(report["comprehension_tests"]["by_audience"]["advanced"], 1)
        
        # Check that report file was created
        self.assertTrue(Path(report_file).exists())


class TestReviewItemAndComprehensionTest(unittest.TestCase):
    """Test the data classes used in the review system"""
    
    def test_review_item_creation(self):
        """Test ReviewItem creation and attributes"""
        item = ReviewItem(
            file_path="test.md",
            section="Section 1", 
            issue_type="technical_accuracy",
            severity="major",
            description="Test description",
            suggested_fix="Test fix",
            reviewer="Test Reviewer"
        )
        
        self.assertEqual(item.file_path, "test.md")
        self.assertEqual(item.section, "Section 1")
        self.assertEqual(item.issue_type, "technical_accuracy")
        self.assertEqual(item.severity, "major")
        self.assertEqual(item.description, "Test description")
        self.assertEqual(item.suggested_fix, "Test fix")
        self.assertEqual(item.reviewer, "Test Reviewer")
        self.assertEqual(item.status, "open")  # Default value
    
    def test_comprehension_test_creation(self):
        """Test ComprehensionTest creation and attributes"""
        test = ComprehensionTest(
            file_path="test.md",
            audience_level="intermediate",
            test_questions=["Question 1", "Question 2"],
            expected_answers=["Answer 1", "Answer 2"],
            difficulty_score=3.5,
            prerequisites=["Prereq 1", "Prereq 2"]
        )
        
        self.assertEqual(test.file_path, "test.md")
        self.assertEqual(test.audience_level, "intermediate")
        self.assertEqual(len(test.test_questions), 2)
        self.assertEqual(len(test.expected_answers), 2)
        self.assertEqual(test.difficulty_score, 3.5)
        self.assertEqual(len(test.prerequisites), 2)


class TestReviewSystemIntegration(unittest.TestCase):
    """Integration tests for the complete review system"""
    
    def test_full_review_workflow(self):
        """Test the complete review workflow"""
        # Skip if docs directory doesn't exist
        if not Path("docs").exists():
            self.skipTest("docs directory not found")
        
        reviewer = DocumentationReviewer("docs")
        
        # Perform technical accuracy review
        technical_items = reviewer.perform_technical_accuracy_review()
        self.assertIsInstance(technical_items, list)
        
        # Perform clarity review
        clarity_items = reviewer.perform_clarity_review()
        self.assertIsInstance(clarity_items, list)
        
        # Generate comprehension tests
        tests = reviewer.generate_comprehension_tests()
        self.assertIsInstance(tests, list)
        
        # Generate comprehensive report
        report = reviewer.generate_comprehensive_report("integration_test_review_report.json")
        
        # Verify report structure
        required_keys = ["summary", "comprehension_tests", "review_items", "recommendations"]
        for key in required_keys:
            self.assertIn(key, report)
        
        # Verify summary structure
        summary_keys = ["total_review_items", "by_severity", "by_type", "by_status"]
        for key in summary_keys:
            self.assertIn(key, report["summary"])


if __name__ == "__main__":
    unittest.main()