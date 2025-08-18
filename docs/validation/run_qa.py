#!/usr/bin/env python3
"""
Comprehensive Quality Assurance Runner for nanoGPT Documentation

This script runs both validation and review systems to provide complete
quality assurance for the documentation.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from doc_validator import DocumentationValidator
from review_system import DocumentationReviewer


def run_comprehensive_qa(docs_root: str, config_file: str, output_dir: str, strict: bool = False):
    """Run comprehensive quality assurance including validation and review"""
    
    print("🚀 Starting Comprehensive Documentation Quality Assurance")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize systems
    validator = DocumentationValidator(docs_root)
    reviewer = DocumentationReviewer(docs_root)
    
    qa_results = {
        "timestamp": datetime.now().isoformat(),
        "docs_root": docs_root,
        "validation": {},
        "review": {},
        "summary": {},
        "recommendations": []
    }
    
    # Phase 1: Validation
    print("\\n📋 Phase 1: Documentation Validation")
    print("-" * 40)
    
    try:
        validation_results = validator.validate_all()
        validation_report = validator.generate_report(str(output_path / "validation_report.json"))
        
        qa_results["validation"] = {
            "total_checks": len(validation_results),
            "passed": len([r for r in validation_results if r.status == "pass"]),
            "failed": len([r for r in validation_results if r.status == "fail"]),
            "warnings": len([r for r in validation_results if r.status == "warning"]),
            "details": validation_report
        }
        
        print(f"✅ Validation complete: {qa_results['validation']['total_checks']} checks")
        print(f"   Passed: {qa_results['validation']['passed']}")
        print(f"   Failed: {qa_results['validation']['failed']}")
        print(f"   Warnings: {qa_results['validation']['warnings']}")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        qa_results["validation"]["error"] = str(e)
    
    # Phase 2: Technical Review
    print("\\n🔍 Phase 2: Technical Accuracy Review")
    print("-" * 40)
    
    try:
        technical_items = reviewer.perform_technical_accuracy_review()
        
        qa_results["review"]["technical_accuracy"] = {
            "total_items": len(technical_items),
            "critical": len([i for i in technical_items if i.severity == "critical"]),
            "major": len([i for i in technical_items if i.severity == "major"]),
            "minor": len([i for i in technical_items if i.severity == "minor"])
        }
        
        print(f"✅ Technical review complete: {len(technical_items)} items")
        print(f"   Critical: {qa_results['review']['technical_accuracy']['critical']}")
        print(f"   Major: {qa_results['review']['technical_accuracy']['major']}")
        print(f"   Minor: {qa_results['review']['technical_accuracy']['minor']}")
        
    except Exception as e:
        print(f"❌ Technical review failed: {e}")
        qa_results["review"]["technical_accuracy"] = {"error": str(e)}
    
    # Phase 3: Clarity Review
    print("\\n📖 Phase 3: Clarity and Readability Review")
    print("-" * 40)
    
    try:
        clarity_items = reviewer.perform_clarity_review()
        
        qa_results["review"]["clarity"] = {
            "total_items": len(clarity_items),
            "critical": len([i for i in clarity_items if i.severity == "critical"]),
            "major": len([i for i in clarity_items if i.severity == "major"]),
            "minor": len([i for i in clarity_items if i.severity == "minor"])
        }
        
        print(f"✅ Clarity review complete: {len(clarity_items)} items")
        print(f"   Critical: {qa_results['review']['clarity']['critical']}")
        print(f"   Major: {qa_results['review']['clarity']['major']}")
        print(f"   Minor: {qa_results['review']['clarity']['minor']}")
        
    except Exception as e:
        print(f"❌ Clarity review failed: {e}")
        qa_results["review"]["clarity"] = {"error": str(e)}
    
    # Phase 4: Comprehension Testing
    print("\\n🧪 Phase 4: Comprehension Test Generation")
    print("-" * 40)
    
    try:
        comprehension_tests = reviewer.generate_comprehension_tests()
        
        qa_results["review"]["comprehension_tests"] = {
            "total_tests": len(comprehension_tests),
            "beginner": len([t for t in comprehension_tests if t.audience_level == "beginner"]),
            "intermediate": len([t for t in comprehension_tests if t.audience_level == "intermediate"]),
            "advanced": len([t for t in comprehension_tests if t.audience_level == "advanced"]),
            "average_difficulty": sum(t.difficulty_score for t in comprehension_tests) / len(comprehension_tests) if comprehension_tests else 0
        }
        
        print(f"✅ Comprehension tests generated: {len(comprehension_tests)} tests")
        print(f"   Beginner: {qa_results['review']['comprehension_tests']['beginner']}")
        print(f"   Intermediate: {qa_results['review']['comprehension_tests']['intermediate']}")
        print(f"   Advanced: {qa_results['review']['comprehension_tests']['advanced']}")
        print(f"   Average difficulty: {qa_results['review']['comprehension_tests']['average_difficulty']:.1f}/5.0")
        
    except Exception as e:
        print(f"❌ Comprehension test generation failed: {e}")
        qa_results["review"]["comprehension_tests"] = {"error": str(e)}
    
    # Phase 5: Generate Comprehensive Reports
    print("\\n📊 Phase 5: Report Generation")
    print("-" * 40)
    
    try:
        # Generate detailed review report
        review_report = reviewer.generate_comprehensive_report(str(output_path / "review_report.json"))
        
        # Generate HTML validation report if validator succeeded
        if "error" not in qa_results["validation"]:
            from run_validation import generate_html_report
            generate_html_report(validator, str(output_path / "validation_report.html"))
            print("✅ HTML validation report generated")
        
        print("✅ Comprehensive reports generated")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
    
    # Phase 6: Analysis and Recommendations
    print("\\n🎯 Phase 6: Analysis and Recommendations")
    print("-" * 40)
    
    # Calculate overall quality metrics
    total_issues = 0
    critical_issues = 0
    
    if "error" not in qa_results["validation"]:
        total_issues += qa_results["validation"]["failed"]
        # Validation failures are considered critical
        critical_issues += qa_results["validation"]["failed"]
    
    if "error" not in qa_results["review"].get("technical_accuracy", {}):
        tech_review = qa_results["review"]["technical_accuracy"]
        total_issues += tech_review["total_items"]
        critical_issues += tech_review["critical"]
    
    if "error" not in qa_results["review"].get("clarity", {}):
        clarity_review = qa_results["review"]["clarity"]
        total_issues += clarity_review["total_items"]
        critical_issues += clarity_review["critical"]
    
    qa_results["summary"] = {
        "total_issues": total_issues,
        "critical_issues": critical_issues,
        "quality_score": max(0, 100 - (critical_issues * 10) - ((total_issues - critical_issues) * 2)),
        "overall_status": "pass" if critical_issues == 0 else "fail"
    }
    
    # Generate recommendations
    recommendations = []
    
    if critical_issues > 0:
        recommendations.append(f"🚨 URGENT: Address {critical_issues} critical issues immediately")
    
    if "error" not in qa_results["validation"] and qa_results["validation"]["failed"] > 0:
        recommendations.append(f"🔧 Fix {qa_results['validation']['failed']} validation failures (broken links, syntax errors)")
    
    if "error" not in qa_results["review"].get("technical_accuracy", {}):
        tech_major = qa_results["review"]["technical_accuracy"]["major"]
        if tech_major > 5:
            recommendations.append(f"📚 Review {tech_major} major technical accuracy issues")
    
    if "error" not in qa_results["review"].get("clarity", {}):
        clarity_major = qa_results["review"]["clarity"]["major"]
        if clarity_major > 10:
            recommendations.append(f"✏️ Improve clarity for {clarity_major} sections")
    
    if "error" not in qa_results["review"].get("comprehension_tests", {}):
        avg_difficulty = qa_results["review"]["comprehension_tests"]["average_difficulty"]
        if avg_difficulty > 4.0:
            recommendations.append("📖 Content may be too advanced - add beginner-friendly explanations")
        elif avg_difficulty < 2.0:
            recommendations.append("🎓 Content may be too basic - add advanced examples")
    
    if not recommendations:
        recommendations.append("✅ Documentation quality is excellent!")
    
    qa_results["recommendations"] = recommendations
    
    # Print summary
    print(f"\\n📈 Quality Score: {qa_results['summary']['quality_score']}/100")
    print(f"🎯 Overall Status: {qa_results['summary']['overall_status'].upper()}")
    print(f"📊 Total Issues: {total_issues} (Critical: {critical_issues})")
    
    print("\\n💡 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Save comprehensive QA report
    qa_report_file = output_path / "qa_comprehensive_report.json"
    with open(qa_report_file, 'w', encoding='utf-8') as f:
        json.dump(qa_results, f, indent=2, ensure_ascii=False)
    
    print(f"\\n📋 Comprehensive QA report saved: {qa_report_file}")
    
    # Determine exit code
    if strict and total_issues > 0:
        print("\\n❌ QA failed in strict mode due to issues found")
        return 1
    elif critical_issues > 0:
        print("\\n❌ QA failed due to critical issues")
        return 1
    else:
        print("\\n✅ QA completed successfully!")
        return 0


def main():
    """Main entry point for comprehensive QA"""
    parser = argparse.ArgumentParser(description="Comprehensive Documentation Quality Assurance")
    parser.add_argument("--docs-root", default="docs", help="Root directory of documentation")
    parser.add_argument("--config", default="docs/validation/validation_config.json", 
                       help="Path to validation configuration file")
    parser.add_argument("--output-dir", default="qa_reports", 
                       help="Directory for output reports")
    parser.add_argument("--strict", action="store_true", 
                       help="Fail on any issues (not just critical ones)")
    parser.add_argument("--setup-feedback", action="store_true",
                       help="Set up feedback integration system")
    
    args = parser.parse_args()
    
    if args.setup_feedback:
        reviewer = DocumentationReviewer(args.docs_root)
        reviewer.create_feedback_integration_system()
        print("✅ Feedback integration system set up")
        return 0
    
    # Run comprehensive QA
    exit_code = run_comprehensive_qa(
        args.docs_root, 
        args.config, 
        args.output_dir, 
        args.strict
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()