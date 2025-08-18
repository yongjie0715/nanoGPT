#!/usr/bin/env python3
"""
Documentation Validation System for nanoGPT Documentation

This module provides comprehensive validation for the nanoGPT documentation,
including syntax checking of code examples, link validation, and consistency checks.
"""

import os
import re
import ast
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ValidationResult:
    """Result of a validation check"""
    file_path: str
    check_type: str
    status: str  # "pass", "fail", "warning"
    message: str
    line_number: Optional[int] = None


class DocumentationValidator:
    """Main validation class for documentation quality assurance"""
    
    def __init__(self, docs_root: str = "docs"):
        self.docs_root = Path(docs_root)
        self.results: List[ValidationResult] = []
        self.terminology_dict = self._load_terminology()
        
    def _load_terminology(self) -> Dict[str, str]:
        """Load standard terminology from glossary"""
        terminology = {}
        glossary_path = self.docs_root / "glossary.md"
        
        if glossary_path.exists():
            with open(glossary_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract terms and definitions from glossary
                term_pattern = r'\*\*([^*]+)\*\*:\s*([^\n]+)'
                matches = re.findall(term_pattern, content)
                terminology = {term.lower(): definition for term, definition in matches}
        
        return terminology
    
    def validate_all(self) -> List[ValidationResult]:
        """Run all validation checks"""
        self.results = []
        
        # Find all markdown files
        md_files = list(self.docs_root.rglob("*.md"))
        
        for md_file in md_files:
            self._validate_file(md_file)
        
        # Run cross-file validations
        self._validate_cross_references(md_files)
        self._validate_terminology_consistency(md_files)
        
        return self.results
    
    def _validate_file(self, file_path: Path):
        """Validate a single markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Validate code examples
            self._validate_code_examples(file_path, content)
            
            # Validate internal links
            self._validate_internal_links(file_path, content)
            
            # Validate external links
            self._validate_external_links(file_path, content)
            
            # Validate structure and formatting
            self._validate_structure(file_path, content)
            
        except Exception as e:
            self.results.append(ValidationResult(
                str(file_path), "file_access", "fail",
                f"Could not read file: {e}"
            ))
    
    def _validate_code_examples(self, file_path: Path, content: str):
        """Validate Python code examples for syntax correctness"""
        # Find Python code blocks
        python_blocks = re.finditer(r'```python\n(.*?)\n```', content, re.DOTALL)
        
        for match in python_blocks:
            code = match.group(1)
            line_start = content[:match.start()].count('\n') + 1
            
            try:
                # Try to parse the code
                ast.parse(code)
                self.results.append(ValidationResult(
                    str(file_path), "code_syntax", "pass",
                    "Python code block is syntactically correct",
                    line_start
                ))
            except SyntaxError as e:
                self.results.append(ValidationResult(
                    str(file_path), "code_syntax", "fail",
                    f"Syntax error in Python code: {e}",
                    line_start + (e.lineno or 1) - 1
                ))
        
        # Find shell/bash code blocks
        shell_blocks = re.finditer(r'```(?:bash|shell|sh)\n(.*?)\n```', content, re.DOTALL)
        
        for match in shell_blocks:
            code = match.group(1)
            line_start = content[:match.start()].count('\n') + 1
            
            # Basic shell command validation
            lines = code.strip().split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Check for dangerous commands
                    dangerous_patterns = [r'rm\s+-rf\s+/', r'sudo\s+rm', r'>\s*/dev/']
                    for pattern in dangerous_patterns:
                        if re.search(pattern, line):
                            self.results.append(ValidationResult(
                                str(file_path), "code_safety", "warning",
                                f"Potentially dangerous command: {line}",
                                line_start + i
                            ))
    
    def _validate_internal_links(self, file_path: Path, content: str):
        """Validate internal markdown links and references"""
        # Find markdown links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = re.finditer(link_pattern, content)
        
        for match in links:
            link_text = match.group(1)
            link_url = match.group(2)
            line_number = content[:match.start()].count('\n') + 1
            
            # Skip external links (http/https)
            if link_url.startswith(('http://', 'https://')):
                continue
            
            # Skip anchors within the same file
            if link_url.startswith('#'):
                self._validate_anchor_link(file_path, content, link_url, line_number)
                continue
            
            # Validate relative file links
            if not link_url.startswith('#'):
                target_path = (file_path.parent / link_url).resolve()
                
                if not target_path.exists():
                    self.results.append(ValidationResult(
                        str(file_path), "internal_link", "fail",
                        f"Broken internal link: {link_url}",
                        line_number
                    ))
                else:
                    self.results.append(ValidationResult(
                        str(file_path), "internal_link", "pass",
                        f"Valid internal link: {link_url}",
                        line_number
                    ))
    
    def _validate_anchor_link(self, file_path: Path, content: str, anchor: str, line_number: int):
        """Validate anchor links within the document"""
        # Remove the # prefix
        anchor_name = anchor[1:].lower().replace(' ', '-').replace('_', '-')
        
        # Find headers in the document
        header_pattern = r'^#+\s+(.+)$'
        headers = re.finditer(header_pattern, content, re.MULTILINE)
        
        valid_anchors = set()
        for header_match in headers:
            header_text = header_match.group(1)
            # Convert to anchor format
            header_anchor = re.sub(r'[^\w\s-]', '', header_text.lower())
            header_anchor = re.sub(r'[\s_]+', '-', header_anchor)
            valid_anchors.add(header_anchor)
        
        if anchor_name in valid_anchors:
            self.results.append(ValidationResult(
                str(file_path), "anchor_link", "pass",
                f"Valid anchor link: {anchor}",
                line_number
            ))
        else:
            self.results.append(ValidationResult(
                str(file_path), "anchor_link", "fail",
                f"Broken anchor link: {anchor}. Available anchors: {', '.join(sorted(valid_anchors))}",
                line_number
            ))
    
    def _validate_external_links(self, file_path: Path, content: str):
        """Validate external HTTP/HTTPS links"""
        # Find external links
        link_pattern = r'\[([^\]]+)\]\((https?://[^)]+)\)'
        links = re.finditer(link_pattern, content)
        
        for match in links:
            link_text = match.group(1)
            link_url = match.group(2)
            line_number = content[:match.start()].count('\n') + 1
            
            try:
                # Create request with user agent to avoid blocking
                req = urllib.request.Request(
                    link_url,
                    headers={'User-Agent': 'Documentation-Validator/1.0'}
                )
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.getcode() == 200:
                        self.results.append(ValidationResult(
                            str(file_path), "external_link", "pass",
                            f"Valid external link: {link_url}",
                            line_number
                        ))
                    else:
                        self.results.append(ValidationResult(
                            str(file_path), "external_link", "warning",
                            f"External link returned status {response.getcode()}: {link_url}",
                            line_number
                        ))
            
            except Exception as e:
                self.results.append(ValidationResult(
                    str(file_path), "external_link", "fail",
                    f"External link failed: {link_url} - {e}",
                    line_number
                ))
    
    def _validate_structure(self, file_path: Path, content: str):
        """Validate document structure and formatting"""
        lines = content.split('\n')
        
        # Check for proper heading hierarchy
        heading_levels = []
        for i, line in enumerate(lines, 1):
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                heading_levels.append((level, i))
        
        # Validate heading progression
        for i in range(1, len(heading_levels)):
            prev_level, prev_line = heading_levels[i-1]
            curr_level, curr_line = heading_levels[i]
            
            # Check for skipped heading levels
            if curr_level > prev_level + 1:
                self.results.append(ValidationResult(
                    str(file_path), "heading_structure", "warning",
                    f"Heading level jumps from {prev_level} to {curr_level}",
                    curr_line
                ))
        
        # Check for consistent code block formatting
        code_block_pattern = r'```(\w+)?'
        code_blocks = list(re.finditer(code_block_pattern, content))
        
        if len(code_blocks) % 2 != 0:
            self.results.append(ValidationResult(
                str(file_path), "code_blocks", "fail",
                "Unmatched code block delimiters",
                None
            ))
    
    def _validate_cross_references(self, md_files: List[Path]):
        """Validate cross-references between documentation files"""
        # Build a map of all available sections
        all_sections = {}
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract headers
                header_pattern = r'^#+\s+(.+)$'
                headers = re.findall(header_pattern, content, re.MULTILINE)
                
                relative_path = md_file.relative_to(self.docs_root)
                all_sections[str(relative_path)] = headers
                
            except Exception:
                continue
        
        # Check for orphaned files (files not referenced by others)
        referenced_files = set()
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find file references
                link_pattern = r'\[([^\]]+)\]\(([^)#]+)(?:#[^)]*)?\)'
                links = re.findall(link_pattern, content)
                
                for link_text, link_path in links:
                    if not link_path.startswith(('http://', 'https://')):
                        referenced_files.add(link_path)
                        
            except Exception:
                continue
        
        # Report orphaned files (excluding main README)
        for md_file in md_files:
            relative_path = str(md_file.relative_to(self.docs_root))
            if (relative_path not in referenced_files and 
                relative_path != "README.md" and 
                not relative_path.startswith('.gitkeep')):
                
                self.results.append(ValidationResult(
                    str(md_file), "cross_reference", "warning",
                    f"File may be orphaned (not referenced by other documentation)",
                    None
                ))
    
    def _validate_terminology_consistency(self, md_files: List[Path]):
        """Check for consistent use of terminology across documentation"""
        if not self.terminology_dict:
            return
        
        # Common variations that should be standardized
        term_variations = {
            'gpt': ['GPT', 'gpt', 'Gpt'],
            'transformer': ['Transformer', 'transformer', 'TRANSFORMER'],
            'pytorch': ['PyTorch', 'pytorch', 'Pytorch', 'PYTORCH'],
            'attention': ['Attention', 'attention', 'ATTENTION'],
            'embedding': ['Embedding', 'embedding', 'embeddings', 'Embeddings']
        }
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for inconsistent terminology
                for standard_term, variations in term_variations.items():
                    used_variations = set()
                    
                    for variation in variations:
                        if variation in content:
                            used_variations.add(variation)
                    
                    if len(used_variations) > 1:
                        self.results.append(ValidationResult(
                            str(md_file), "terminology", "warning",
                            f"Inconsistent terminology for '{standard_term}': {', '.join(used_variations)}",
                            None
                        ))
                        
            except Exception:
                continue
    
    def generate_report(self, output_file: str = "validation_report.json"):
        """Generate a comprehensive validation report"""
        # Categorize results
        report = {
            "summary": {
                "total_checks": len(self.results),
                "passed": len([r for r in self.results if r.status == "pass"]),
                "failed": len([r for r in self.results if r.status == "fail"]),
                "warnings": len([r for r in self.results if r.status == "warning"])
            },
            "results_by_file": defaultdict(list),
            "results_by_type": defaultdict(list)
        }
        
        for result in self.results:
            report["results_by_file"][result.file_path].append({
                "check_type": result.check_type,
                "status": result.status,
                "message": result.message,
                "line_number": result.line_number
            })
            
            report["results_by_type"][result.check_type].append({
                "file_path": result.file_path,
                "status": result.status,
                "message": result.message,
                "line_number": result.line_number
            })
        
        # Convert defaultdict to regular dict for JSON serialization
        report["results_by_file"] = dict(report["results_by_file"])
        report["results_by_type"] = dict(report["results_by_type"])
        
        # Write report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def print_summary(self):
        """Print a summary of validation results"""
        total = len(self.results)
        passed = len([r for r in self.results if r.status == "pass"])
        failed = len([r for r in self.results if r.status == "fail"])
        warnings = len([r for r in self.results if r.status == "warning"])
        
        print(f"\n=== Documentation Validation Summary ===")
        print(f"Total checks: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Warnings: {warnings}")
        
        if failed > 0:
            print(f"\n=== Failed Checks ===")
            for result in self.results:
                if result.status == "fail":
                    line_info = f" (line {result.line_number})" if result.line_number else ""
                    print(f"❌ {result.file_path}{line_info}: {result.message}")
        
        if warnings > 0:
            print(f"\n=== Warnings ===")
            for result in self.results:
                if result.status == "warning":
                    line_info = f" (line {result.line_number})" if result.line_number else ""
                    print(f"⚠️  {result.file_path}{line_info}: {result.message}")


def main():
    """Main entry point for the validation system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate nanoGPT documentation")
    parser.add_argument("--docs-root", default="docs", help="Root directory of documentation")
    parser.add_argument("--output", default="validation_report.json", help="Output file for detailed report")
    parser.add_argument("--quiet", action="store_true", help="Only show summary")
    
    args = parser.parse_args()
    
    validator = DocumentationValidator(args.docs_root)
    results = validator.validate_all()
    
    # Generate detailed report
    report = validator.generate_report(args.output)
    
    if not args.quiet:
        validator.print_summary()
    
    # Exit with error code if there are failures
    failed_count = len([r for r in results if r.status == "fail"])
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()