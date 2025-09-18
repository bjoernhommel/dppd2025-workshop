# Introduction to Large Language Modeling

This repository contains material for the DPPD2025 conference workshop [Introduction to Large Language Modeling](https://dppd2025.de/programm/pre-conference-workshops/).

## Abstract
> Recent progress in the field of natural language processing (NLP) has had a transformative impact on research practice and methodology in the behavioral and social sciences. Most prominently, the transformer model architecture proposed by Vaswani et al. (2017) has led to the rise of large language models (LLMs) and drastically advanced both natural language generation and understanding. While numerous studies employ methods that rely on accessible but rather limited means of interacting with modern LLMs (e.g., prompt engineering), this workshop offers a conceptual and practical deep dive into the technical foundations of transformer models. The workshop will cover a) conceptual and historical foundations (e.g., distributional semantics), b) the anatomy of the transformer model architecture (e.g., the attention mechanism), c) model training and inference, d) types of transformer models and their individual use cases, e) applications of transformer models in behavioral science (e.g., decoder models for automatic generation of personality items, encoder models for estimating social desirability at the item level, predicting survey response patterns with sentence transformers, and extracting trait information from natural language), and f) working with the Hugging Face ecosystem. The workshop combines theoretical foundations with hands-on coding examples, allowing participants to bridge conceptual understanding with practical implementation. The goal of this workshop is to empower researchers at all career stages to use state-of-the-art NLP techniques in their own research and help them make informed methodological decisions when employing LLMs.

**Instructor(s)**: [Björn E. Hommel](mailto:bjorn.hommel@gmail.com)

## Setup
### 1. Enable WSL2 (Windows only)

**Windows:** Open PowerShell as Administrator and run:
```bash
wsl --install
```
Restart your computer, then open WSL2 by running the following command in PowerShell or Command Prompt:
```bash
wsl
```

Follow Linux instructions for all subsequent steps.

### 2. Install Python ≥3.12

Check if python already is installed:

```bash
python3 --version
python --version
```

If Python 3.12+ is not installed:

**Linux (Ubuntu/Debian) & WSL2 on Windows:**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

**macOS:**
```bash
# Using Homebrew
brew install python@3.12
```

### 3. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Add Poetry to PATH:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify installation:
```bash
poetry --version
```

### 4. Clone Repository

Click the `<> Code` dropdown and copy the HTTPS repository url to clipboard.

```bash
git clone <repository-url>
cd <repository-name>
```

### 5. Install Dependencies

```bash
poetry install
```

Installation may take several minutes to complete.

### 6. Launch Tutorial

```bash
poetry run marimo edit ./notebooks/01-neural-language-model.py
```

Your browser should open with the marimo notebook interface.