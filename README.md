# punygrad

A tiny autograd engine for learning purposes, inspired by [micrograd](https://github.com/karpathy/micrograd). Implements reverse-mode autodifferentiation (backpropagation) in pure Python.

## Features

- Scalar-valued autograd engine
- Pure Python Tensor class with broadcasting capabilities
- Neural network library with modular design
- No external dependencies for core functionality
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

### Scalar Operations

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

### Tensor Operations

```python
from punygrad.engine import Tensor

# Create tensors from nested lists
t1 = Tensor([[1, 2], [3, 4]])
t2 = Tensor([[5], [6]])

# Broadcasting operations
bt1, bt2 = t1.broadcast_with(t2)
# bt1 shape: (2, 2), bt2 shape: (2, 2)

# Convert to Python list for operations
data_list = t1.to_list()
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
│       ├── engine.py      # Autograd engine with Scalar and Tensor classes
│       └── nn.py          # Neural network components
├── tests/                 # Test directory
├── examples/             # Example usage
└── requirements.txt      # Project dependencies
```

## Components

### Scalar

The `Scalar` class is the core of the autograd engine, supporting:
- Automatic differentiation
- Basic arithmetic operations (+, -, *, /, **)
- Common activation functions (exp, tanh)

### Tensor

The `Tensor` class provides multi-dimensional array support:
- Creation from nested lists
- Shape information and manipulation
- Broadcasting capabilities (similar to NumPy/PyTorch)
- Pure Python implementation with no external dependencies

### Neural Network

The `nn` module provides building blocks for neural networks:
- Modules with parameters
- Linear layers
- Activation functions
- Multi-layer perceptrons (MLPs)

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This project is heavily inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd), a tiny scalar-valued autograd engine. punygrad extends the concepts with additional features like the Tensor class with broadcasting capabilities while maintaining the educational focus of the original project.
