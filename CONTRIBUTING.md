# Contributing to DPR Question Answering System

Thank you for your interest in contributing to the DPR Question Answering System! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)

## Getting Started

1. **Fork the repository** on GitLab
2. **Clone your fork** locally:
   ```bash
   git clone https://gitlab.com/your-username/dpr-qa-system.git
   cd dpr-qa-system
   ```

3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://gitlab.com/original-username/dpr-qa-system.git
   ```

## Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Verify installation**:
   ```bash
   python -m pytest tests/
   ```

## Making Changes

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-description
   ```

2. **Make your changes** following the project conventions

3. **Write tests** for new functionality

4. **Update documentation** if necessary

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dpr_qa_system

# Run specific test file
pytest tests/test_dpr_qa_system.py

# Run tests with specific markers
pytest -m "not slow"  # Skip slow tests
```

### Test Categories

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **Slow Tests**: Tests that require model downloads (marked with `@pytest.mark.slow`)

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test names
- Mock external dependencies when possible
- Include both positive and negative test cases

## Code Style

This project follows Python coding standards with specific configurations:

### Formatting Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting

### Running Code Quality Checks

```bash
# Format code
black .
isort .

# Check linting
flake8

# Type checking
mypy dpr_qa_system.py

# Security scanning
bandit -r .
```

### Pre-commit Hooks

Pre-commit hooks automatically run these checks before each commit. If you haven't installed them:

```bash
pre-commit install
```

## Submitting Changes

1. **Ensure all tests pass**:
   ```bash
   pytest
   ```

2. **Ensure code quality checks pass**:
   ```bash
   pre-commit run --all-files
   ```

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Merge Request** on GitLab:
   - Use a descriptive title
   - Provide detailed description of changes
   - Reference any related issues
   - Ensure CI pipeline passes

### Commit Message Format

Follow conventional commit format:

```
type: Brief description

Detailed explanation of changes (if needed)

Fixes #issue-number
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

## Issue Reporting

When reporting issues, please include:

1. **Clear description** of the problem
2. **Steps to reproduce** the issue
3. **Expected behavior** vs actual behavior
4. **Environment information**:
   - Python version
   - Operating system
   - Package versions
5. **Error messages** and stack traces
6. **Minimal code example** if applicable

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed

## Development Guidelines

### Code Organization

- Keep functions focused and small
- Use descriptive variable and function names
- Add docstrings to all public functions and classes
- Follow PEP 8 style guidelines

### Documentation

- Update README.md for significant changes
- Add docstrings for new functions/classes
- Include type hints where appropriate
- Update CHANGELOG.md for releases

### Performance Considerations

- Profile code for performance bottlenecks
- Use appropriate data structures
- Consider memory usage for large datasets
- Cache expensive operations when possible

## Getting Help

- **GitLab Issues**: For bug reports and feature requests
- **Merge Request Discussions**: For code review discussions
- **Documentation**: Check the README.md and code comments

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- GitLab project contributors page

Thank you for contributing to the DPR Question Answering System! 