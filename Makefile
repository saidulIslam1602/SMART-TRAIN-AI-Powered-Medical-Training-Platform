# SMART-TRAIN Medical AI Platform Makefile
# Industry-standard build automation and development workflow

.PHONY: help install install-dev test lint format clean validate demo docs docker deploy

# Default target
help: ## Show this help message
	@echo "SMART-TRAIN Medical AI Platform"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install production dependencies
	python3 -m pip install --upgrade pip
	python3 -m pip install -r requirements_core.txt

install-dev: ## Install development dependencies
	python3 -m pip install --upgrade pip
	python3 -m pip install -r requirements_core.txt
	python3 -m pip install -r requirements.txt

setup: ## Run initial project setup
	python3 scripts/setup.py

setup-no-deps: ## Run setup without installing dependencies
	python3 scripts/setup.py --no-deps

# Development targets
test: ## Run test suite
	python3 -m pytest tests/ -v --cov=src/smart_train --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	python3 -m pytest tests/unit/ -v

test-integration: ## Run integration tests only
	python3 -m pytest tests/integration/ -v

test-compliance: ## Run medical compliance tests
	python3 -m pytest tests/medical_compliance/ -v

lint: ## Run code linting
	python3 -m flake8 src/ scripts/ examples/
	python3 -m mypy src/smart_train/

format: ## Format code with black
	python3 -m black src/ scripts/ examples/
	python3 -m isort src/ scripts/ examples/

format-check: ## Check code formatting
	python3 -m black --check src/ scripts/ examples/
	python3 -m isort --check-only src/ scripts/ examples/

# Validation and demonstration
validate: ## Validate system integrity and readiness
	python3 scripts/validate_system.py

validate-verbose: ## Validate system with verbose output
	python3 scripts/validate_system.py --verbose

validate-report: ## Generate validation report
	python3 scripts/validate_system.py --output validation_report.json

demo: ## Run system demonstration
	python3 examples/system_demonstration.py

# Data and model targets
collect-datasets: ## Collect real medical datasets
	python3 -c "import sys; sys.path.insert(0, 'src'); from smart_train.data.collection import RealDatasetCollector; collector = RealDatasetCollector(); collector.process()"

preprocess-data: ## Run data preprocessing pipeline
	python3 -c "import sys; sys.path.insert(0, 'src'); from smart_train.data.preprocessing import MedicalDataPreprocessor; processor = MedicalDataPreprocessor(); print('Data preprocessing pipeline ready')"

# Documentation targets
docs: ## Generate documentation
	@echo "Generating documentation..."
	@mkdir -p docs/build
	@echo "Documentation generation not yet implemented"

docs-serve: ## Serve documentation locally
	@echo "Serving documentation on http://localhost:8000"
	@echo "Documentation server not yet implemented"

# Docker targets
docker-build: ## Build Docker image
	docker build -t smart-train-ai:latest .

docker-run: ## Run Docker container
	docker run -it --rm -p 8000:8000 smart-train-ai:latest

docker-compose-up: ## Start services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop services with docker-compose
	docker-compose down

# Deployment targets
deploy-dev: ## Deploy to development environment
	@echo "Deploying to development environment..."
	@echo "Development deployment not yet implemented"

deploy-staging: ## Deploy to staging environment
	@echo "Deploying to staging environment..."
	@echo "Staging deployment not yet implemented"

deploy-prod: ## Deploy to production environment
	@echo "Deploying to production environment..."
	@echo "Production deployment not yet implemented"

# Azure targets
azure-setup: ## Setup Azure ML workspace
	python3 scripts/setup_azure_ml.py

azure-deploy: ## Deploy to Azure
	@echo "Deploying to Azure..."
	@echo "Azure deployment not yet implemented"

# Utility targets
clean: ## Clean up temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf build/
	rm -rf dist/

clean-data: ## Clean up data directories (use with caution)
	@echo "This will remove all processed data. Are you sure? [y/N]"
	@read -r response; if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		rm -rf datasets/processed/; \
		rm -rf datasets/raw/; \
		echo "Data directories cleaned"; \
	else \
		echo "Cancelled"; \
	fi

clean-logs: ## Clean up log files
	rm -rf logs/
	rm -rf compliance/audit_logs/

# Security and compliance
security-scan: ## Run security vulnerability scan
	python3 -m safety check
	python3 -m bandit -r src/

compliance-check: ## Run medical compliance validation
	python3 scripts/validate_system.py --verbose | grep -i compliance

audit-trail-report: ## Generate audit trail report
	python3 -c "import sys; sys.path.insert(0, 'src'); from smart_train.compliance.audit_trail import AuditTrailManager; from datetime import datetime, timedelta; am = AuditTrailManager(); report = am.generate_compliance_report(datetime.now() - timedelta(days=30), datetime.now()); print('Audit trail report generated')"

# Development workflow
dev-setup: install-dev setup validate ## Complete development environment setup
	@echo "✅ Development environment ready!"
	@echo "Run 'make demo' to see the system in action"

ci-test: format-check lint test ## Run CI/CD test suite
	@echo "✅ CI/CD tests passed!"

pre-commit: format lint test validate ## Run pre-commit checks
	@echo "✅ Pre-commit checks passed!"

# Information targets
info: ## Show system information
	@echo "SMART-TRAIN Medical AI Platform"
	@echo "==============================="
	@echo "Python version: $(shell python3 --version)"
	@echo "Project root: $(shell pwd)"
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Git commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'Not a git repository')"
	@echo ""
	@echo "Key directories:"
	@echo "  Source code: src/smart_train/"
	@echo "  Configuration: config/"
	@echo "  Scripts: scripts/"
	@echo "  Examples: examples/"
	@echo "  Tests: tests/"

version: ## Show version information
	@python3 -c "import sys; sys.path.insert(0, 'src'); from smart_train import __version__; print(f'SMART-TRAIN v{__version__}')"

# Quick start
quickstart: dev-setup demo ## Quick start for new developers
	@echo ""
	@echo "🎉 SMART-TRAIN quickstart completed!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Review the demonstration output above"
	@echo "  2. Read PHASE1_COMPLETION_REPORT.md"
	@echo "  3. Explore the codebase in src/smart_train/"
	@echo "  4. Run 'make test' to verify everything works"
	@echo "  5. Start developing Phase 2 features!"
