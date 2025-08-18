# Makefile for nanoGPT Documentation Validation

.PHONY: help validate test clean install-deps validate-strict validate-quick qa qa-strict review setup-feedback test-all

# Default target
help:
	@echo "nanoGPT Documentation Quality Assurance"
	@echo "======================================="
	@echo ""
	@echo "Available targets:"
	@echo "  qa             - Run comprehensive quality assurance (validation + review)"
	@echo "  qa-strict      - Run QA in strict mode (fail on any issues)"
	@echo "  validate       - Run validation only with HTML report"
	@echo "  validate-strict - Run validation in strict mode (fail on warnings)"
	@echo "  validate-quick  - Run validation without HTML report"
	@echo "  review         - Run comprehensive review and refinement"
	@echo "  setup-feedback - Set up feedback integration system"
	@echo "  test           - Run validation system tests"
	@echo "  test-all       - Run all tests (validation + review systems)"
	@echo "  install-deps   - Install Python dependencies"
	@echo "  clean          - Remove generated reports and cache files"
	@echo "  help           - Show this help message"

# Install Python dependencies
install-deps:
	@echo "Installing Python dependencies..."
	@python3 -m pip install --upgrade pip
	@python3 -c "import ast, urllib.request, json, pathlib" || \
		(echo "Required Python modules not available. Please ensure Python 3.6+ is installed." && exit 1)
	@echo "✅ Dependencies verified"

# Run full validation
validate: install-deps
	@echo "🔍 Running comprehensive documentation validation..."
	@python3 run_validation.py --docs-root ../.. --config validation_config.json
	@echo ""
	@echo "📊 Reports generated:"
	@echo "  - JSON: validation_report.json"
	@echo "  - HTML: validation_report.html"

# Run validation in strict mode
validate-strict: install-deps
	@echo "🔍 Running strict documentation validation..."
	@python3 run_validation.py --docs-root ../.. --config validation_config.json --strict

# Run quick validation without HTML report
validate-quick: install-deps
	@echo "🔍 Running quick documentation validation..."
	@python3 run_validation.py --docs-root ../.. --config validation_config.json --no-html --quiet

# Run tests for the validation system
test: install-deps
	@echo "🧪 Running validation system tests..."
	@python3 -m unittest test_validator.py -v

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	@rm -f validation_report.json validation_report.html
	@rm -f comprehensive_review_report.json
	@rm -f integration_test_report.json test_report.json
	@rm -f integration_test_review_report.json
	@rm -rf qa_reports/
	@rm -rf __pycache__/
	@rm -rf .pytest_cache/
	@echo "✅ Cleanup complete"

# Continuous integration target
ci: install-deps test-all qa-strict
	@echo "✅ All CI checks passed"

# Comprehensive Quality Assurance
qa: install-deps
	@echo "🚀 Running comprehensive quality assurance..."
	@python3 run_qa.py --docs-root ../.. --output-dir qa_reports
	@echo ""
	@echo "📊 QA Reports generated in qa_reports/"

# Comprehensive QA in strict mode
qa-strict: install-deps
	@echo "🚀 Running comprehensive QA in strict mode..."
	@python3 run_qa.py --docs-root ../.. --output-dir qa_reports --strict

# Review and refinement only
review: install-deps
	@echo "🔍 Running comprehensive review and refinement..."
	@python3 review_system.py --docs-root ../..

# Set up feedback integration system
setup-feedback: install-deps
	@echo "💬 Setting up feedback integration system..."
	@python3 run_qa.py --setup-feedback --docs-root ../..

# Run all tests
test-all: install-deps
	@echo "🧪 Running all system tests..."
	@python3 -m unittest test_validator.py -v
	@python3 -m unittest test_review_system.py -v

# Development target - run QA and open HTML report
dev: qa
	@echo "🌐 Opening QA reports..."
	@if command -v open >/dev/null 2>&1; then \
		open qa_reports/validation_report.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open qa_reports/validation_report.html; \
	else \
		echo "Please open qa_reports/validation_report.html in your browser"; \
	fi