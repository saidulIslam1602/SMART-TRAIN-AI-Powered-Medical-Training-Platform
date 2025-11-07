# Contributing to SMART-TRAIN AI Platform

We welcome contributions to the SMART-TRAIN AI Platform! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our code of conduct:
- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a welcoming environment for all contributors

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Git
- Docker (for containerized development)

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/SMART-TRAIN-AI-Powered-Medical-Training-Platform.git
   cd SMART-TRAIN-AI-Powered-Medical-Training-Platform
   ```

3. Set up the development environment:
   ```bash
   make dev-setup
   ```

4. Create a branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings using Google style
- Maximum line length of 88 characters (Black formatter)

### Code Quality
Before submitting code, ensure it passes all quality checks:

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Run tests
make test

# Security scan
make security-scan
```

### Testing
- Write unit tests for all new functionality
- Include integration tests for API endpoints
- Add medical compliance tests for healthcare-related features
- Maintain test coverage above 80%

### Documentation
- Update README.md for new features
- Add docstrings to all public functions and classes
- Include code examples for new API endpoints
- Update CHANGELOG.md with your changes

## Submitting Changes

### Pull Request Process

1. Ensure your code follows the development guidelines above
2. Update documentation as needed
3. Add or update tests for your changes
4. Ensure all tests pass locally
5. Commit your changes with descriptive messages:
   ```bash
   git commit -m "Add real-time analysis endpoint for CPR assessment"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a pull request on GitHub with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots for UI changes
   - Test results

### Commit Message Guidelines
- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## Types of Contributions

### Bug Reports
When filing bug reports, please include:
- Clear description of the issue
- Steps to reproduce the problem
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant log files or error messages

### Feature Requests
For new features, please provide:
- Clear description of the proposed feature
- Use case and business justification
- Proposed implementation approach
- Any potential breaking changes

### Medical Compliance
For healthcare-related contributions:
- Ensure compliance with relevant medical standards
- Include appropriate audit trail logging
- Consider data privacy and security implications
- Test with medical compliance validation suite

## Development Environment

### Local Development
```bash
# Install development dependencies
make install-dev

# Run the application locally
make run-dev

# Run with hot reload
make run-watch
```

### Docker Development
```bash
# Build development container
make docker-build-dev

# Run in container
make docker-run-dev
```

### Testing Environment
```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-compliance

# Run with coverage
make test-coverage
```

## Project Structure

Understanding the project structure helps with contributions:

```
src/smart_train/
├── core/           # Core framework components
├── data/           # Data processing modules  
├── models/         # AI/ML model implementations
├── api/            # REST API endpoints
├── compliance/     # Medical compliance framework
└── utils/          # Utility functions

tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── medical_compliance/  # Compliance tests
└── fixtures/       # Test data and fixtures
```

## Medical Compliance Considerations

When contributing to medical-related functionality:

1. **Data Privacy**: Ensure all patient data is properly anonymized
2. **Audit Trails**: Log all data access and modifications
3. **Validation**: Include medical expert validation where appropriate
4. **Standards**: Follow ISO 13485 and IEC 62304 guidelines
5. **Testing**: Include compliance-specific test cases

## Release Process

The project follows semantic versioning:
- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features, backward compatible
- **Patch** (0.0.1): Bug fixes, backward compatible

## Getting Help

If you need help with contributions:
- Check existing issues and discussions
- Review the documentation in `docs/`
- Ask questions in GitHub Discussions
- Contact maintainers for complex issues

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for significant contributions
- README.md acknowledgments section
- GitHub contributors page

Thank you for contributing to SMART-TRAIN AI Platform!
