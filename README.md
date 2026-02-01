# Optimized Indexing and Search with Homomorphic Encryption

Privacy-preserving vector similarity search using Homomorphic Encryption.

## Overview

This project implements encrypted vector similarity search using Microsoft SEAL library. It allows performing nearest neighbor searches on encrypted data without decryption.

## Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
samsung-prism/
├── src/
│   ├── core/           # Core HE operations
│   ├── indexing/       # Search algorithms
│   └── utils/          # Utilities
├── experiments/        # Experiment scripts
├── tests/              # Unit tests
└── docs/               # Documentation
```

## Quick Start

```python
from src.core.he_engine import HEEngine
from src.core.vector_ops import EncryptedVectorOps

# Initialize HE engine
engine = HEEngine()

# Encrypt vectors
encrypted_vec = engine.encrypt_vector([1.0, 2.0, 3.0, 4.0])

# Perform encrypted operations
ops = EncryptedVectorOps(engine)
result = ops.dot_product(encrypted_vec1, encrypted_vec2)
```

## Author

Ghanshyam Sharma
