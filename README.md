# PunyGrad

A tiny autograd engine for learning purposes, inspired by micrograd. Implements reverse-mode autodifferentiation (backpropagation) in pure Python.

## Features

- Scalar-valued autograd engine
- Neural network library with modular design
- Pure Python implementation for learning purposes
- Supports basic arithmetic operations and common activation functions

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/punygrad.git
cd punygrad
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from punygrad.engine import Scalar
from punygrad.nn import MLP

# Create a simple neural network
model = MLP(nin=3, nouts=[4, 4, 1])

# Forward pass
x = [2.0, 3.0, -1.0]
output = model(x)

# Backward pass
output.backprop()
```

## Development

Run tests:
```bash
pytest
```

Format code:
```bash
black .
ruff .
mypy .
```

## Project Structure
```
punygrad/
├── src/
│   └── punygrad/          # Main package
│       ├── __init__.py    # Package initialization
│       ├── engine.py      # Autograd engine
│       └── nn.py          # Neural network components
├── tests/                 # Test directory
├── examples/             # Example usage
└── requirements.txt      # Project dependencies
```

## License

MIT License - see LICENSE file for details.
