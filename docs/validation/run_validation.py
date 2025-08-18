#!/usr/bin/env python3
"""
Comprehensive validation runner for nanoGPT documentation

This script runs all validation checks and generates detailed reports
with configurable thresholds and output formats.
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from doc_validator import DocumentationValidator


def load_config(config_path: str) -> dict:
    """Load validation configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)


def generate_html_report(validator: DocumentationValidator, output_path: str):
    """Generate an HTML report for better readability"""
    results = validator.results
    
    # Categorize results
    passed = [r for r in results if r.status == "pass"]
    failed = [r for r in results if r.status == "fail"]
    warnings = [r for r in results if r.status == "warning"]
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentation Validation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            padding: 20px;
            border-radius: 6px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .passed {{ background-color: #d4edda; color: #155724; }}
        .failed {{ background-color: #f8d7da; color: #721c24; }}
        .warnings {{ background-color: #fff3cd; color: #856404; }}
        .total {{ background-color: #e2e3e5; color: #383d41; }}
        .results-section {{
            margin-bottom: 30px;
        }}
        .results-section h2 {{
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }}
        .result-item {{
            margin-bottom: 15px;
            padding: 15px;
            border-left: 4px solid;
            background-color: #f8f9fa;
        }}
        .result-item.pass {{ border-left-color: #28a745; }}
        .result-item.fail {{ border-left-color: #dc3545; }}
        .result-item.warning {{ border-left-color: #ffc107; }}
        .result-meta {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .result-message {{
            font-weight: 500;
        }}
        .file-group {{
            margin-bottom: 25px;
            border: 1px solid #ddd;
            border-radius: 6px;
            overflow: hidden;
        }}
        .file-header {{
            background-color: #f1f3f4;
            padding: 15px;
            font-weight: bold;
            border-bottom: 1px solid #ddd;
        }}
        .file-results {{
            padding: 15px;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 Documentation Validation Report</h1>
            <p>Comprehensive quality assurance for nanoGPT documentation</p>
        </div>
        
        <div class="summary">
            <div class="summary-card total">
                <h3>{len(results)}</h3>
                <p>Total Checks</p>
            </div>
            <div class="summary-card passed">
                <h3>{len(passed)}</h3>
                <p>Passed</p>
            </div>
            <div class="summary-card failed">
                <h3>{len(failed)}</h3>
                <p>Failed</p>
            </div>
            <div class="summary-card warnings">
                <h3>{len(warnings)}</h3>
                <p>Warnings</p>
            </div>
        </div>
"""
    
    if failed:
        html_content += """
        <div class="results-section">
            <h2>❌ Failed Checks</h2>
"""
        for result in failed:
            line_info = f" (line {result.line_number})" if result.line_number else ""
            html_content += f"""
            <div class="result-item fail">
                <div class="result-meta">{result.file_path}{line_info} • {result.check_type}</div>
                <div class="result-message">{result.message}</div>
            </div>
"""
        html_content += "</div>"
    
    if warnings:
        html_content += """
        <div class="results-section">
            <h2>⚠️ Warnings</h2>
"""
        for result in warnings:
            line_info = f" (line {result.line_number})" if result.line_number else ""
            html_content += f"""
            <div class="result-item warning">
                <div class="result-meta">{result.file_path}{line_info} • {result.check_type}</div>
                <div class="result-message">{result.message}</div>
            </div>
"""
        html_content += "</div>"
    
    # Group results by file
    from collections import defaultdict
    results_by_file = defaultdict(list)
    for result in results:
        results_by_file[result.file_path].append(result)
    
    if results_by_file:
        html_content += """
        <div class="results-section">
            <h2>📁 Results by File</h2>
"""
        for file_path, file_results in sorted(results_by_file.items()):
            file_passed = len([r for r in file_results if r.status == "pass"])
            file_failed = len([r for r in file_results if r.status == "fail"])
            file_warnings = len([r for r in file_results if r.status == "warning"])
            
            html_content += f"""
            <div class="file-group">
                <div class="file-header">
                    {file_path} 
                    <span style="float: right; font-weight: normal;">
                        ✅ {file_passed} | ❌ {file_failed} | ⚠️ {file_warnings}
                    </span>
                </div>
                <div class="file-results">
"""
            for result in file_results:
                if result.status != "pass":  # Only show failures and warnings
                    line_info = f" (line {result.line_number})" if result.line_number else ""
                    status_class = result.status
                    html_content += f"""
                    <div class="result-item {status_class}">
                        <div class="result-meta">{result.check_type}{line_info}</div>
                        <div class="result-message">{result.message}</div>
                    </div>
"""
            html_content += """
                </div>
            </div>
"""
        html_content += "</div>"
    
    html_content += f"""
        <div class="timestamp">
            Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def check_quality_thresholds(validator: DocumentationValidator, config: dict) -> bool:
    """Check if validation results meet quality thresholds"""
    thresholds = config.get("quality_thresholds", {})
    
    failed_count = len([r for r in validator.results if r.status == "fail"])
    warning_count = len([r for r in validator.results if r.status == "warning"])
    passed_count = len([r for r in validator.results if r.status == "pass"])
    total_count = len(validator.results)
    
    # Check maximum failures
    max_failures = thresholds.get("max_failures", 0)
    if failed_count > max_failures:
        print(f"❌ Quality check failed: {failed_count} failures exceed threshold of {max_failures}")
        return False
    
    # Check maximum warnings
    max_warnings = thresholds.get("max_warnings", 10)
    if warning_count > max_warnings:
        print(f"⚠️ Quality check failed: {warning_count} warnings exceed threshold of {max_warnings}")
        return False
    
    # Check minimum pass rate
    min_pass_rate = thresholds.get("min_pass_rate", 0.95)
    if total_count > 0:
        pass_rate = passed_count / total_count
        if pass_rate < min_pass_rate:
            print(f"📊 Quality check failed: Pass rate {pass_rate:.2%} below threshold of {min_pass_rate:.2%}")
            return False
    
    print("✅ All quality thresholds met")
    return True


def main():
    """Main entry point for validation runner"""
    parser = argparse.ArgumentParser(description="Run comprehensive documentation validation")
    parser.add_argument("--docs-root", default="docs", help="Root directory of documentation")
    parser.add_argument("--config", default="docs/validation/validation_config.json", 
                       help="Path to validation configuration file")
    parser.add_argument("--output-json", default="validation_report.json", 
                       help="Output file for JSON report")
    parser.add_argument("--output-html", default="validation_report.html", 
                       help="Output file for HTML report")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report generation")
    parser.add_argument("--quiet", action="store_true", help="Only show summary and errors")
    parser.add_argument("--strict", action="store_true", help="Fail on any warnings")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize validator
    validator = DocumentationValidator(args.docs_root)
    
    if not args.quiet:
        print("🔍 Starting documentation validation...")
        print(f"📁 Docs root: {args.docs_root}")
        print(f"⚙️ Config: {args.config}")
    
    # Run validation
    results = validator.validate_all()
    
    if not args.quiet:
        print(f"✅ Validation complete. Processed {len(results)} checks.")
    
    # Generate reports
    json_report = validator.generate_report(args.output_json)
    
    if not args.no_html:
        generate_html_report(validator, args.output_html)
        if not args.quiet:
            print(f"📊 HTML report generated: {args.output_html}")
    
    # Print summary
    if not args.quiet:
        validator.print_summary()
    
    # Check quality thresholds
    quality_passed = check_quality_thresholds(validator, config)
    
    # Determine exit code
    failed_count = len([r for r in results if r.status == "fail"])
    warning_count = len([r for r in results if r.status == "warning"])
    
    if failed_count > 0:
        print(f"\n❌ Validation failed with {failed_count} errors")
        sys.exit(1)
    elif args.strict and warning_count > 0:
        print(f"\n⚠️ Validation failed in strict mode with {warning_count} warnings")
        sys.exit(1)
    elif not quality_passed:
        print(f"\n📊 Validation failed quality thresholds")
        sys.exit(1)
    else:
        if not args.quiet:
            print(f"\n✅ All validation checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()