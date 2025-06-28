# DPR Question Answering System

[![pipeline status](https://gitlab.com/your-username/dpr-qa-system/badges/main/pipeline.svg)](https://gitlab.com/your-username/dpr-qa-system/-/commits/main)
[![coverage report](https://gitlab.com/your-username/dpr-qa-system/badges/main/coverage.svg)](https://gitlab.com/your-username/dpr-qa-system/-/commits/main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python implementation of a Question Answering system using Dense Passage Retrieval (DPR) encoders from Facebook AI Research. This system embeds documents using `DPRContextEncoder` and questions using `DPRQuestionEncoder`, then uses cosine similarity to find relevant passages and provide answers.

## 🚀 Features

- **Document Embedding**: Uses DPRContextEncoder to create dense vector representations of documents
- **Question Embedding**: Uses DPRQuestionEncoder to create dense vector representations of questions  
- **Similarity Search**: Uses cosine similarity (without FAISS) to find most relevant documents
- **Answer Extraction**: Simple answer extraction from the most relevant context
- **Knowledge Base Management**: Save and load document embeddings for reuse
- **Multiple Usage Modes**: Demo mode, interactive mode, and single question mode
- **GitLab CI/CD**: Automated testing, linting, and deployment
- **Code Quality**: Pre-commit hooks, type checking, and security scanning

## 📦 Installation

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://gitlab.com/your-username/dpr-qa-system.git
   cd dpr-qa-system
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package**:
   ```bash
   pip install -e .
   ```

### Development Installation

For development with all tools and dependencies:

```bash
pip install -e ".[dev]"
pre-commit install
```

### Docker Installation (Coming Soon)

```bash
docker build -t dpr-qa-system .
docker run -it dpr-qa-system
```

## 🔧 Usage

### 1. Basic Demo Mode
Run the system with sample documents and questions:
```bash
python dpr_qa_system.py
```

### 2. Interactive Mode
Run in interactive mode to ask your own questions:
```bash
python dpr_qa_system.py --interactive
```

### 3. Single Question Mode
Answer a specific question:
```bash
python dpr_qa_system.py --question "What is machine learning?"
```

### 4. Custom Documents
Add your own documents:
```bash
python dpr_qa_system.py --documents "Your first document text here" "Your second document text here" --interactive
```

### 5. Example Usage Script
Run the example script with predefined documents and questions:
```bash
python example_usage.py
```

### 6. As a Python Package
```python
from dpr_qa_system import DPRQuestionAnsweringSystem

# Initialize the system
qa_system = DPRQuestionAnsweringSystem()

# Add documents
documents = ["Document 1 text", "Document 2 text"]
qa_system.add_documents(documents)

# Ask questions
result = qa_system.answer_question("What is the topic?")
print(result['answer'])
```

## 🏗️ Project Structure

```
dpr-qa-system/
├── .gitlab-ci.yml              # GitLab CI/CD configuration
├── .gitignore                  # Git ignore patterns
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── pyproject.toml              # Modern Python project configuration
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
├── LICENSE                     # MIT License
├── README.md                   # This file
├── CONTRIBUTING.md             # Contribution guidelines
├── dpr_qa_system.py           # Main DPR QA system implementation
├── example_usage.py           # Usage examples
└── tests/                     # Test suite
    ├── __init__.py
    └── test_dpr_qa_system.py  # Unit and integration tests
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dpr_qa_system --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Run only integration tests
```

### Code Quality
```bash
# Format code
black .
isort .

# Lint code
flake8

# Type checking
mypy dpr_qa_system.py

# Security scan
bandit -r .

# Run all pre-commit hooks
pre-commit run --all-files
```

## 🔄 CI/CD Pipeline

The GitLab CI/CD pipeline includes:

- **Testing**: Unit tests, integration tests, coverage reporting
- **Code Quality**: Linting with flake8, formatting with black, import sorting with isort
- **Security**: Security scanning with bandit and safety
- **Documentation**: Automatic documentation generation with Sphinx
- **Deployment**: GitLab Pages deployment for documentation

### Pipeline Stages

1. **Test**: Run pytest with coverage reporting
2. **Lint**: Code quality and style checks
3. **Security**: Security vulnerability scanning
4. **Build**: Documentation building
5. **Deploy**: Deploy to GitLab Pages

## 🤖 How It Works

1. **Document Processing**: 
   - Documents are tokenized using DPRContextEncoderTokenizer
   - Embedded using DPRContextEncoder to create dense vectors
   - Stored in memory for similarity search

2. **Question Processing**:
   - Questions are tokenized using DPRQuestionEncoderTokenizer  
   - Embedded using DPRQuestionEncoder to create dense vectors

3. **Retrieval**:
   - Cosine similarity is calculated between question and document embeddings
   - Top-k most similar documents are retrieved

4. **Answer Generation**:
   - Simple extraction from the most relevant document
   - Returns the most relevant context as the answer

## 📊 Example Output

```
Question: What is machine learning?
Answer: Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.
Confidence: 0.8542

Relevant Documents:
1. (Score: 0.8542) Machine learning is a method of data analysis that automates analytical model building...
2. (Score: 0.6234) Artificial Intelligence (AI) refers to the simulation of human intelligence in machines...
```

## 🚧 Limitations

- **Simple Answer Extraction**: Uses basic heuristics rather than sophisticated span extraction
- **No Reader Model**: Doesn't use a dedicated reading comprehension model for answer extraction
- **Memory Storage**: Stores embeddings in memory (no persistent vector database)
- **Single Language**: Optimized for English text

## 🔮 Roadmap

- [ ] **Reader Model Integration**: Add BERT-based reader for better answer extraction
- [ ] **Vector Database**: Add support for persistent vector storage (ChromaDB, Weaviate)
- [ ] **Multi-language Support**: Use multilingual DPR models
- [ ] **Web Interface**: FastAPI-based web service
- [ ] **Docker Support**: Containerized deployment
- [ ] **Batch Processing**: Support for processing large document collections
- [ ] **Advanced Chunking**: Better document chunking strategies
- [ ] **Answer Ranking**: More sophisticated answer scoring

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Quick Contribution Steps

1. Fork the repository on GitLab
2. Create a feature branch
3. Make your changes with tests
4. Ensure all checks pass
5. Submit a merge request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Facebook AI Research for the DPR models
- Hugging Face for the transformers library
- The open-source community for various tools and libraries

## 📚 Dependencies

### Production Dependencies
- `torch>=1.9.0`: PyTorch for deep learning models
- `transformers>=4.21.0`: Hugging Face transformers for DPR models
- `numpy>=1.21.0`: Numerical computations
- `scikit-learn>=1.0.0`: Cosine similarity calculation
- `datasets>=2.0.0`: Dataset utilities

### Development Dependencies
- Testing: `pytest`, `pytest-cov`, `pytest-mock`
- Code Quality: `black`, `flake8`, `isort`, `mypy`
- Security: `bandit`, `safety`
- Documentation: `sphinx`, `sphinx-rtd-theme`
- Development: `pre-commit`, `jupyter`, `ipython`

## 📞 Support

- **Issues**: [GitLab Issues](https://gitlab.com/your-username/dpr-qa-system/-/issues)
- **Discussions**: [GitLab Discussions](https://gitlab.com/your-username/dpr-qa-system/-/issues)
- **Documentation**: [GitLab Pages](https://your-username.gitlab.io/dpr-qa-system)

---

**Note**: Replace `your-username` with your actual GitLab username in the URLs above. 