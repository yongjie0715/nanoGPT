#!/usr/bin/env python3
"""
Comprehensive Review and Refinement System for nanoGPT Documentation

This module provides tools for technical accuracy review, audience comprehension testing,
and feedback integration to ensure documentation quality and effectiveness.
"""

import os
import re
import json
import ast
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime


@dataclass
class ReviewItem:
    """Represents a single review item or issue"""
    file_path: str
    section: str
    issue_type: str  # "technical_accuracy", "clarity", "completeness", "consistency"
    severity: str    # "critical", "major", "minor", "suggestion"
    description: str
    suggested_fix: Optional[str] = None
    reviewer: Optional[str] = None
    timestamp: Optional[str] = None
    status: str = "open"  # "open", "in_progress", "resolved", "wont_fix"


@dataclass
class ComprehensionTest:
    """Represents a comprehension test for different audience levels"""
    file_path: str
    audience_level: str  # "beginner", "intermediate", "advanced"
    test_questions: List[str]
    expected_answers: List[str]
    difficulty_score: float  # 1.0 (easy) to 5.0 (very hard)
    prerequisites: List[str]


class DocumentationReviewer:
    """Main class for comprehensive documentation review and refinement"""
    
    def __init__(self, docs_root: str = "docs"):
        self.docs_root = Path(docs_root)
        self.review_items: List[ReviewItem] = []
        self.comprehension_tests: List[ComprehensionTest] = []
        
    def perform_technical_accuracy_review(self) -> List[ReviewItem]:
        """Review documentation for technical accuracy"""
        print("🔍 Performing technical accuracy review...")
        
        # Find all markdown files
        md_files = list(self.docs_root.rglob("*.md"))
        
        for md_file in md_files:
            self._review_file_technical_accuracy(md_file)
        
        return [item for item in self.review_items if item.issue_type == "technical_accuracy"]
    
    def _review_file_technical_accuracy(self, file_path: Path):
        """Review a single file for technical accuracy"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common technical inaccuracies
            self._check_pytorch_api_usage(file_path, content)
            self._check_mathematical_formulas(file_path, content)
            self._check_code_examples_accuracy(file_path, content)
            self._check_parameter_descriptions(file_path, content)
            
        except Exception as e:
            self.review_items.append(ReviewItem(
                str(file_path), "file_access", "technical_accuracy", "critical",
                f"Could not read file for technical review: {e}"
            ))
    
    def _check_pytorch_api_usage(self, file_path: Path, content: str):
        """Check for correct PyTorch API usage in documentation"""
        # Common PyTorch API patterns to verify
        pytorch_patterns = [
            (r'torch\.nn\.functional\.([a-zA-Z_]+)', "PyTorch functional API usage"),
            (r'torch\.optim\.([a-zA-Z_]+)', "PyTorch optimizer usage"),
            (r'\.cuda\(\)', "CUDA device placement"),
            (r'\.to\(device\)', "Device placement pattern"),
            (r'torch\.compile', "Model compilation"),
            (r'@torch\.no_grad\(\)', "Gradient context manager")
        ]
        
        for pattern, description in pytorch_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                
                # Check if the usage looks correct in context
                line_start = content.rfind('\n', 0, match.start()) + 1
                line_end = content.find('\n', match.end())
                if line_end == -1:
                    line_end = len(content)
                line_content = content[line_start:line_end]
                
                # Flag potential issues
                if 'deprecated' in line_content.lower():
                    self.review_items.append(ReviewItem(
                        str(file_path), f"Line {line_number}", "technical_accuracy", "major",
                        f"Potentially deprecated PyTorch API usage: {match.group(0)}",
                        f"Verify if {match.group(0)} is still the recommended approach"
                    ))
    
    def _check_mathematical_formulas(self, file_path: Path, content: str):
        """Check mathematical formulas for accuracy"""
        # Look for mathematical expressions
        math_patterns = [
            r'softmax\([^)]+\)',
            r'attention\s*=\s*[^\\n]+',
            r'loss\s*=\s*[^\\n]+',
            r'gradient\s*=\s*[^\\n]+',
            r'\\frac\{[^}]+\}\{[^}]+\}',  # LaTeX fractions
            r'\\sum_\{[^}]+\}',           # LaTeX summations
        ]
        
        for pattern in math_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                
                # Flag for manual review
                self.review_items.append(ReviewItem(
                    str(file_path), f"Line {line_number}", "technical_accuracy", "minor",
                    f"Mathematical formula requires verification: {match.group(0)}",
                    "Verify mathematical accuracy against research papers or PyTorch documentation"
                ))
    
    def _check_code_examples_accuracy(self, file_path: Path, content: str):
        """Check code examples for accuracy and best practices"""
        # Find Python code blocks
        python_blocks = re.finditer(r'```python\n(.*?)\n```', content, re.DOTALL)
        
        for match in python_blocks:
            code = match.group(1)
            line_start = content[:match.start()].count('\n') + 1
            
            # Check for common issues
            self._check_code_best_practices(file_path, code, line_start)
    
    def _check_code_best_practices(self, file_path: Path, code: str, line_start: int):
        """Check code for best practices and common issues"""
        issues = []
        
        # Check for hardcoded values
        if re.search(r'\b\d{3,}\b', code):  # Numbers with 3+ digits
            issues.append("Consider using named constants for large numeric values")
        
        # Check for missing error handling
        if 'open(' in code and 'with' not in code:
            issues.append("Consider using context managers (with statement) for file operations")
        
        # Check for deprecated patterns
        if '.cuda()' in code and 'device' not in code:
            issues.append("Consider using device-agnostic code with .to(device)")
        
        # Check for missing type hints in function definitions
        if re.search(r'def\s+\w+\([^)]*\):', code) and '->' not in code:
            issues.append("Consider adding type hints for better code documentation")
        
        for issue in issues:
            self.review_items.append(ReviewItem(
                str(file_path), f"Line {line_start}", "technical_accuracy", "minor",
                f"Code best practice: {issue}"
            ))
    
    def _check_parameter_descriptions(self, file_path: Path, content: str):
        """Check parameter descriptions for accuracy"""
        # Look for parameter documentation patterns
        param_patterns = [
            r'`([a-zA-Z_][a-zA-Z0-9_]*)`\s*:\s*([^\\n]+)',  # `param`: description
            r'\*\*([a-zA-Z_][a-zA-Z0-9_]*)\*\*\s*:\s*([^\\n]+)',  # **param**: description
        ]
        
        for pattern in param_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                param_name = match.group(1)
                description = match.group(2)
                line_number = content[:match.start()].count('\n') + 1
                
                # Check for vague descriptions
                vague_words = ['various', 'different', 'some', 'certain', 'appropriate']
                if any(word in description.lower() for word in vague_words):
                    self.review_items.append(ReviewItem(
                        str(file_path), f"Line {line_number}", "technical_accuracy", "minor",
                        f"Parameter description for '{param_name}' could be more specific",
                        f"Consider providing more concrete details instead of: {description}"
                    ))
    
    def perform_clarity_review(self) -> List[ReviewItem]:
        """Review documentation for clarity and readability"""
        print("📖 Performing clarity and readability review...")
        
        md_files = list(self.docs_root.rglob("*.md"))
        
        for md_file in md_files:
            self._review_file_clarity(md_file)
        
        return [item for item in self.review_items if item.issue_type == "clarity"]
    
    def _review_file_clarity(self, file_path: Path):
        """Review a single file for clarity"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self._check_sentence_complexity(file_path, content)
            self._check_jargon_usage(file_path, content)
            self._check_explanation_flow(file_path, content)
            
        except Exception as e:
            self.review_items.append(ReviewItem(
                str(file_path), "file_access", "clarity", "critical",
                f"Could not read file for clarity review: {e}"
            ))
    
    def _check_sentence_complexity(self, file_path: Path, content: str):
        """Check for overly complex sentences"""
        sentences = re.split(r'[.!?]+', content)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Count words and complexity indicators
            words = len(sentence.split())
            commas = sentence.count(',')
            semicolons = sentence.count(';')
            
            # Flag very long or complex sentences
            if words > 30 or commas > 3 or semicolons > 1:
                self.review_items.append(ReviewItem(
                    str(file_path), f"Sentence {i+1}", "clarity", "minor",
                    f"Sentence may be too complex ({words} words, {commas} commas)",
                    "Consider breaking into shorter sentences for better readability"
                ))
    
    def _check_jargon_usage(self, file_path: Path, content: str):
        """Check for unexplained jargon or technical terms"""
        # Common ML/AI terms that should be explained for beginners
        technical_terms = [
            'autoregressive', 'causal masking', 'attention mechanism', 'transformer',
            'embedding', 'tokenization', 'gradient descent', 'backpropagation',
            'softmax', 'cross-entropy', 'perplexity', 'beam search'
        ]
        
        for term in technical_terms:
            if term.lower() in content.lower():
                # Check if term is explained nearby (within 200 characters)
                pattern = re.compile(f'{re.escape(term)}', re.IGNORECASE)
                matches = pattern.finditer(content)
                
                for match in matches:
                    line_number = content[:match.start()].count('\n') + 1
                    
                    # Look for explanation patterns nearby
                    context_start = max(0, match.start() - 200)
                    context_end = min(len(content), match.end() + 200)
                    context = content[context_start:context_end]
                    
                    explanation_patterns = [
                        r'is\s+(?:a|an)\s+',
                        r'refers\s+to',
                        r'means\s+',
                        r':\s*[A-Z]',  # Colon followed by explanation
                    ]
                    
                    has_explanation = any(re.search(pattern, context, re.IGNORECASE) 
                                        for pattern in explanation_patterns)
                    
                    if not has_explanation:
                        self.review_items.append(ReviewItem(
                            str(file_path), f"Line {line_number}", "clarity", "minor",
                            f"Technical term '{term}' may need explanation for beginners",
                            f"Consider adding a brief explanation or link to glossary"
                        ))
    
    def _check_explanation_flow(self, file_path: Path, content: str):
        """Check logical flow of explanations"""
        # Look for sections that jump between complexity levels
        headers = re.finditer(r'^#+\s+(.+)$', content, re.MULTILINE)
        
        prev_level = 0
        for header_match in headers:
            level = len(header_match.group(0)) - len(header_match.group(0).lstrip('#'))
            header_text = header_match.group(1)
            line_number = content[:header_match.start()].count('\n') + 1
            
            # Check for logical progression
            if level > prev_level + 1:
                self.review_items.append(ReviewItem(
                    str(file_path), f"Line {line_number}", "clarity", "minor",
                    f"Heading structure jumps from level {prev_level} to {level}",
                    "Consider adding intermediate sections for better flow"
                ))
            
            prev_level = level
    
    def generate_comprehension_tests(self) -> List[ComprehensionTest]:
        """Generate comprehension tests for different audience levels"""
        print("🧪 Generating comprehension tests...")
        
        md_files = list(self.docs_root.rglob("*.md"))
        
        for md_file in md_files:
            self._generate_file_comprehension_tests(md_file)
        
        return self.comprehension_tests
    
    def _generate_file_comprehension_tests(self, file_path: Path):
        """Generate comprehension tests for a specific file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract key concepts and generate questions
            self._extract_key_concepts_and_questions(file_path, content)
            
        except Exception as e:
            print(f"Warning: Could not generate tests for {file_path}: {e}")
    
    def _extract_key_concepts_and_questions(self, file_path: Path, content: str):
        """Extract key concepts and generate appropriate questions"""
        # Find code examples
        code_blocks = re.finditer(r'```python\n(.*?)\n```', content, re.DOTALL)
        
        for match in code_blocks:
            code = match.group(1)
            
            # Generate questions based on code complexity
            if 'class' in code:
                # Advanced level - class definitions
                test = ComprehensionTest(
                    str(file_path), "advanced",
                    [
                        "What design patterns are demonstrated in this class?",
                        "How would you extend this class for a different use case?",
                        "What are the performance implications of this implementation?"
                    ],
                    [
                        "Look for patterns like inheritance, composition, or factory methods",
                        "Consider inheritance or composition approaches",
                        "Analyze memory usage, computational complexity, and scalability"
                    ],
                    4.0,
                    ["Object-oriented programming", "Python classes", "Design patterns"]
                )
                self.comprehension_tests.append(test)
            
            elif 'def' in code:
                # Intermediate level - function definitions
                test = ComprehensionTest(
                    str(file_path), "intermediate",
                    [
                        "What is the purpose of this function?",
                        "What are the input and output types?",
                        "How would you test this function?"
                    ],
                    [
                        "Analyze the function name and implementation",
                        "Look at parameter types and return statements",
                        "Consider edge cases and expected behaviors"
                    ],
                    3.0,
                    ["Python functions", "Type annotations", "Testing concepts"]
                )
                self.comprehension_tests.append(test)
            
            else:
                # Beginner level - simple code snippets
                test = ComprehensionTest(
                    str(file_path), "beginner",
                    [
                        "What does this code do?",
                        "What would happen if you run this code?",
                        "Can you identify the key variables?"
                    ],
                    [
                        "Read through the code line by line",
                        "Trace the execution flow",
                        "Look for variable assignments and usage"
                    ],
                    2.0,
                    ["Basic Python syntax", "Variable assignment", "Code reading"]
                )
                self.comprehension_tests.append(test)
    
    def create_feedback_integration_system(self):
        """Create a system for integrating user feedback"""
        print("💬 Setting up feedback integration system...")
        
        feedback_dir = self.docs_root / "feedback"
        feedback_dir.mkdir(exist_ok=True)
        
        # Create feedback templates
        self._create_feedback_templates(feedback_dir)
        
        # Create feedback processing script
        self._create_feedback_processor(feedback_dir)
    
    def _create_feedback_templates(self, feedback_dir: Path):
        """Create templates for different types of feedback"""
        
        # Technical accuracy feedback template
        technical_template = """# Technical Accuracy Feedback

**File:** [file_path]
**Section:** [section_name]
**Issue Type:** Technical Accuracy

## Description
[Describe the technical issue or inaccuracy]

## Current Content
```
[Copy the current content that has issues]
```

## Suggested Fix
[Provide the corrected content or explanation]

## References
[Include links to documentation, papers, or other authoritative sources]

## Severity
- [ ] Critical (Incorrect information that could cause errors)
- [ ] Major (Misleading or outdated information)
- [ ] Minor (Unclear or imprecise information)

## Additional Notes
[Any additional context or comments]
"""
        
        with open(feedback_dir / "technical_accuracy_template.md", 'w') as f:
            f.write(technical_template)
        
        # Clarity feedback template
        clarity_template = """# Clarity and Readability Feedback

**File:** [file_path]
**Section:** [section_name]
**Issue Type:** Clarity

## Description
[Describe what makes this section unclear or hard to understand]

## Current Content
```
[Copy the current content that needs improvement]
```

## Suggested Improvement
[Provide clearer wording or restructuring suggestions]

## Target Audience
- [ ] Beginner
- [ ] Intermediate
- [ ] Advanced

## Specific Issues
- [ ] Too technical/jargon-heavy
- [ ] Poor organization/flow
- [ ] Missing examples
- [ ] Confusing explanations
- [ ] Other: [specify]

## Additional Notes
[Any additional context or suggestions]
"""
        
        with open(feedback_dir / "clarity_template.md", 'w') as f:
            f.write(clarity_template)
        
        # General feedback template
        general_template = """# General Documentation Feedback

**File:** [file_path]
**Section:** [section_name or "General"]

## Feedback Type
- [ ] Content suggestion
- [ ] Structural improvement
- [ ] Missing information
- [ ] Outdated information
- [ ] Other: [specify]

## Description
[Describe your feedback or suggestion]

## Suggested Changes
[Provide specific suggestions for improvement]

## Priority
- [ ] High (Important for user understanding)
- [ ] Medium (Would improve user experience)
- [ ] Low (Nice to have)

## Additional Context
[Any additional information that might be helpful]
"""
        
        with open(feedback_dir / "general_template.md", 'w') as f:
            f.write(general_template)
    
    def _create_feedback_processor(self, feedback_dir: Path):
        """Create a script to process and integrate feedback"""
        
        processor_script = '''#!/usr/bin/env python3
"""
Feedback Processing Script

This script processes feedback files and integrates them into the review system.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

def process_feedback_files(feedback_dir="docs/feedback"):
    """Process all feedback files in the feedback directory"""
    feedback_path = Path(feedback_dir)
    
    if not feedback_path.exists():
        print(f"Feedback directory {feedback_dir} does not exist")
        return
    
    feedback_files = list(feedback_path.glob("*.md"))
    
    if not feedback_files:
        print("No feedback files found")
        return
    
    processed_feedback = []
    
    for feedback_file in feedback_files:
        if feedback_file.name.endswith("_template.md"):
            continue  # Skip templates
        
        try:
            feedback_data = parse_feedback_file(feedback_file)
            processed_feedback.append(feedback_data)
            print(f"✅ Processed: {feedback_file.name}")
        except Exception as e:
            print(f"❌ Error processing {feedback_file.name}: {e}")
    
    # Generate feedback report
    generate_feedback_report(processed_feedback, feedback_path / "feedback_report.json")
    
    return processed_feedback

def parse_feedback_file(feedback_file):
    """Parse a feedback file and extract structured data"""
    with open(feedback_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract metadata
    file_match = re.search(r'\\*\\*File:\\*\\*\\s*(.+)', content)
    section_match = re.search(r'\\*\\*Section:\\*\\*\\s*(.+)', content)
    issue_type_match = re.search(r'\\*\\*Issue Type:\\*\\*\\s*(.+)', content)
    
    # Extract description
    desc_match = re.search(r'## Description\\n(.+?)\\n##', content, re.DOTALL)
    
    # Extract severity/priority
    severity_matches = re.findall(r'- \\[x\\] (.+)', content)
    
    return {
        'file': feedback_file.name,
        'target_file': file_match.group(1).strip() if file_match else 'Unknown',
        'section': section_match.group(1).strip() if section_match else 'Unknown',
        'issue_type': issue_type_match.group(1).strip() if issue_type_match else 'General',
        'description': desc_match.group(1).strip() if desc_match else 'No description',
        'severity_priority': severity_matches,
        'timestamp': datetime.now().isoformat(),
        'status': 'pending'
    }

def generate_feedback_report(feedback_data, output_file):
    """Generate a comprehensive feedback report"""
    report = {
        'generated_at': datetime.now().isoformat(),
        'total_feedback_items': len(feedback_data),
        'feedback_by_type': {},
        'feedback_by_file': {},
        'items': feedback_data
    }
    
    # Categorize feedback
    for item in feedback_data:
        issue_type = item['issue_type']
        target_file = item['target_file']
        
        if issue_type not in report['feedback_by_type']:
            report['feedback_by_type'][issue_type] = 0
        report['feedback_by_type'][issue_type] += 1
        
        if target_file not in report['feedback_by_file']:
            report['feedback_by_file'][target_file] = 0
        report['feedback_by_file'][target_file] += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Feedback report generated: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process documentation feedback")
    parser.add_argument("--feedback-dir", default="docs/feedback", 
                       help="Directory containing feedback files")
    
    args = parser.parse_args()
    
    process_feedback_files(args.feedback_dir)
'''
        
        with open(feedback_dir / "process_feedback.py", 'w') as f:
            f.write(processor_script)
        
        # Make it executable
        os.chmod(feedback_dir / "process_feedback.py", 0o755)
    
    def generate_comprehensive_report(self, output_file: str = "comprehensive_review_report.json"):
        """Generate a comprehensive review report"""
        print("📊 Generating comprehensive review report...")
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_review_items": len(self.review_items),
                "by_severity": defaultdict(int),
                "by_type": defaultdict(int),
                "by_status": defaultdict(int)
            },
            "comprehension_tests": {
                "total_tests": len(self.comprehension_tests),
                "by_audience": defaultdict(int),
                "average_difficulty": 0.0
            },
            "review_items": [asdict(item) for item in self.review_items],
            "comprehension_tests_data": [asdict(test) for test in self.comprehension_tests],
            "recommendations": self._generate_recommendations()
        }
        
        # Calculate summary statistics
        for item in self.review_items:
            report["summary"]["by_severity"][item.severity] += 1
            report["summary"]["by_type"][item.issue_type] += 1
            report["summary"]["by_status"][item.status] += 1
        
        for test in self.comprehension_tests:
            report["comprehension_tests"]["by_audience"][test.audience_level] += 1
        
        if self.comprehension_tests:
            avg_difficulty = sum(test.difficulty_score for test in self.comprehension_tests) / len(self.comprehension_tests)
            report["comprehension_tests"]["average_difficulty"] = avg_difficulty
        
        # Convert defaultdicts to regular dicts
        report["summary"]["by_severity"] = dict(report["summary"]["by_severity"])
        report["summary"]["by_type"] = dict(report["summary"]["by_type"])
        report["summary"]["by_status"] = dict(report["summary"]["by_status"])
        report["comprehension_tests"]["by_audience"] = dict(report["comprehension_tests"]["by_audience"])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on review results"""
        recommendations = []
        
        # Analyze review items for patterns
        critical_items = [item for item in self.review_items if item.severity == "critical"]
        major_items = [item for item in self.review_items if item.severity == "major"]
        
        if critical_items:
            recommendations.append(f"Address {len(critical_items)} critical issues immediately - these may prevent users from successfully following the documentation")
        
        if major_items:
            recommendations.append(f"Review {len(major_items)} major issues that could significantly impact user experience")
        
        # Check for patterns in issue types
        issue_counts = defaultdict(int)
        for item in self.review_items:
            issue_counts[item.issue_type] += 1
        
        if issue_counts["technical_accuracy"] > 10:
            recommendations.append("Consider having technical content reviewed by domain experts")
        
        if issue_counts["clarity"] > 15:
            recommendations.append("Consider restructuring content for better readability and flow")
        
        # Analyze comprehension test difficulty
        if self.comprehension_tests:
            avg_difficulty = sum(test.difficulty_score for test in self.comprehension_tests) / len(self.comprehension_tests)
            
            if avg_difficulty > 4.0:
                recommendations.append("Content may be too advanced - consider adding more beginner-friendly explanations")
            elif avg_difficulty < 2.0:
                recommendations.append("Content may be too basic - consider adding more advanced examples and concepts")
        
        return recommendations


def main():
    """Main entry point for the review system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive documentation review and refinement")
    parser.add_argument("--docs-root", default="docs", help="Root directory of documentation")
    parser.add_argument("--output", default="comprehensive_review_report.json", 
                       help="Output file for review report")
    parser.add_argument("--setup-feedback", action="store_true", 
                       help="Set up feedback integration system")
    
    args = parser.parse_args()
    
    reviewer = DocumentationReviewer(args.docs_root)
    
    if args.setup_feedback:
        reviewer.create_feedback_integration_system()
        print("✅ Feedback integration system set up")
        return
    
    # Perform comprehensive review
    print("🔍 Starting comprehensive documentation review...")
    
    # Technical accuracy review
    technical_items = reviewer.perform_technical_accuracy_review()
    print(f"✅ Technical accuracy review complete: {len(technical_items)} items")
    
    # Clarity review
    clarity_items = reviewer.perform_clarity_review()
    print(f"✅ Clarity review complete: {len(clarity_items)} items")
    
    # Generate comprehension tests
    tests = reviewer.generate_comprehension_tests()
    print(f"✅ Comprehension tests generated: {len(tests)} tests")
    
    # Generate comprehensive report
    report = reviewer.generate_comprehensive_report(args.output)
    print(f"📊 Comprehensive report generated: {args.output}")
    
    # Print summary
    print(f"\\n=== Review Summary ===")
    print(f"Total review items: {len(reviewer.review_items)}")
    print(f"Critical issues: {len([i for i in reviewer.review_items if i.severity == 'critical'])}")
    print(f"Major issues: {len([i for i in reviewer.review_items if i.severity == 'major'])}")
    print(f"Minor issues: {len([i for i in reviewer.review_items if i.severity == 'minor'])}")
    print(f"Comprehension tests: {len(tests)}")


if __name__ == "__main__":
    main()