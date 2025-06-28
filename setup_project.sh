#!/bin/bash

# DPR Question Answering System - Project Setup Script
# This script sets up the development environment for the project

set -e  # Exit on any error

echo "🚀 Setting up DPR Question Answering System..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version found"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "ℹ️  Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
if [ "$1" = "--dev" ] || [ "$1" = "-d" ]; then
    echo "Installing development dependencies..."
    pip install -e ".[dev]"
    
    # Install pre-commit hooks
    echo "🪝 Installing pre-commit hooks..."
    pre-commit install
    echo "✅ Pre-commit hooks installed"
else
    echo "Installing production dependencies..."
    pip install -e .
fi

# Run initial tests to verify installation
echo "🧪 Running initial tests..."
if command -v pytest &> /dev/null; then
    pytest tests/ -v --tb=short || echo "⚠️  Some tests failed, but installation completed"
else
    echo "ℹ️  Pytest not available, skipping tests"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the virtual environment: source venv/bin/activate"
if [ "$1" = "--dev" ] || [ "$1" = "-d" ]; then
    echo "   2. Run tests: pytest"
    echo "   3. Check code quality: pre-commit run --all-files"
    echo "   4. Start development!"
else
    echo "   2. Run the example: python example_usage.py"
    echo "   3. Try interactive mode: python dpr_qa_system.py --interactive"
fi
echo ""
echo "📚 For more information, see README.md"
echo "🤝 For contributing guidelines, see CONTRIBUTING.md" 