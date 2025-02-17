"""Benchmarks comparing PunyGrad with PyTorch."""

import time
import numpy as np
import torch
import torch.nn as nn
from punygrad.engine import Scalar
from punygrad.nn import MLP


def benchmark_scalar_ops(n_iterations=1000):
    """Benchmark scalar operations."""
    # PunyGrad timing
    start = time.perf_counter()
    for _ in range(n_iterations):
        x = Scalar(2.0)
        y = Scalar(3.0)
        z = (x * y + x ** 2).tanh()
        z.backprop()
    puny_time = time.perf_counter() - start

    # PyTorch timing
    start = time.perf_counter()
    for _ in range(n_iterations):
        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(3.0, requires_grad=True)
        z = torch.tanh(x * y + x ** 2)
        z.backward()
    torch_time = time.perf_counter() - start

    return {
        "operation": "scalar_ops",
        "iterations": n_iterations,
        "punygrad_time": puny_time,
        "pytorch_time": torch_time,
        "speedup": max(puny_time / torch_time, torch_time / puny_time),
        "faster": "PunyGrad" if puny_time < torch_time else "PyTorch"
    }


def benchmark_mlp_forward(batch_size=32, n_iterations=100):
    """Benchmark MLP forward pass."""
    # Setup
    nin = 10
    nouts = [32, 32, 1]
    X = np.random.randn(batch_size, nin).tolist()

    # PunyGrad MLP
    mlp = MLP(nin, nouts)

    # PyTorch equivalent
    torch_mlp = nn.Sequential(
        nn.Linear(nin, nouts[0]),
        nn.Tanh(),
        nn.Linear(nouts[0], nouts[1]),
        nn.Tanh(),
        nn.Linear(nouts[1], nouts[2]),
        nn.Tanh()
    )

    # Copy weights to make fair comparison
    layer_idx = 0
    for layer in mlp.network:
        weights = []
        biases = []
        for n in layer.layer:
            weights.append([w.data for w in n.weights])
            biases.append(n.bias.data)
        
        with torch.no_grad():
            torch_mlp[layer_idx*2].weight.copy_(torch.tensor(weights))
            torch_mlp[layer_idx*2].bias.copy_(torch.tensor(biases))
        layer_idx += 1

    # PunyGrad timing
    start = time.perf_counter()
    for _ in range(n_iterations):
        for x in X:
            _ = mlp(x)
    puny_time = time.perf_counter() - start

    # PyTorch timing
    X_torch = torch.tensor(X, dtype=torch.float32)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = torch_mlp(X_torch)
    torch_time = time.perf_counter() - start

    return {
        "operation": "mlp_forward",
        "batch_size": batch_size,
        "iterations": n_iterations,
        "punygrad_time": puny_time,
        "pytorch_time": torch_time,
        "speedup": max(puny_time / torch_time, torch_time / puny_time),
        "faster": "PunyGrad" if puny_time < torch_time else "PyTorch"
    }


def benchmark_mlp_training(batch_size=32, n_iterations=100):
    """Benchmark MLP training step."""
    # Setup
    nin = 10
    nouts = [32, 32, 1]
    X = np.random.randn(batch_size, nin).tolist()
    y = np.random.randn(batch_size).tolist()
    learning_rate = 0.1

    # PunyGrad MLP
    mlp = MLP(nin, nouts)

    # PyTorch equivalent
    torch_mlp = nn.Sequential(
        nn.Linear(nin, nouts[0]),
        nn.Tanh(),
        nn.Linear(nouts[0], nouts[1]),
        nn.Tanh(),
        nn.Linear(nouts[1], nouts[2]),
        nn.Tanh()
    )
    optimizer = torch.optim.SGD(torch_mlp.parameters(), lr=learning_rate)

    # Copy weights
    layer_idx = 0
    for layer in mlp.network:
        weights = []
        biases = []
        for n in layer.layer:
            weights.append([w.data for w in n.weights])
            biases.append(n.bias.data)
        
        with torch.no_grad():
            torch_mlp[layer_idx*2].weight.copy_(torch.tensor(weights))
            torch_mlp[layer_idx*2].bias.copy_(torch.tensor(biases))
        layer_idx += 1

    # PunyGrad timing
    start = time.perf_counter()
    for _ in range(n_iterations):
        total_loss = 0
        for xi, yi in zip(X, y):
            pred = mlp(xi)
            loss = (pred.data - yi) ** 2
            total_loss += loss
            pred.backprop()
        
        for p in mlp.parameters():
            p.data -= learning_rate * p.grad
            p.grad = 0
    puny_time = time.perf_counter() - start

    # PyTorch timing
    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32)
    start = time.perf_counter()
    for _ in range(n_iterations):
        optimizer.zero_grad()
        pred = torch_mlp(X_torch)
        loss = ((pred.squeeze() - y_torch) ** 2).mean()
        loss.backward()
        optimizer.step()
    torch_time = time.perf_counter() - start

    return {
        "operation": "mlp_training",
        "batch_size": batch_size,
        "iterations": n_iterations,
        "punygrad_time": puny_time,
        "pytorch_time": torch_time,
        "speedup": max(puny_time / torch_time, torch_time / puny_time),
        "faster": "PunyGrad" if puny_time < torch_time else "PyTorch"
    }


if __name__ == "__main__":
    # Run benchmarks
    print("\nRunning benchmarks...\n")
    
    results = [
        benchmark_scalar_ops(n_iterations=100),
        benchmark_mlp_forward(batch_size=16, n_iterations=10),
        benchmark_mlp_training(batch_size=16, n_iterations=10)
    ]
    
    # Print results
    print("Benchmark Results:")
    print("-----------------")
    for r in results:
        print(f"\n{r['operation'].replace('_', ' ').title()}:")
        print(f"  Iterations: {r['iterations']}")
        if 'batch_size' in r:
            print(f"  Batch Size: {r['batch_size']}")
        print(f"  PunyGrad Time: {r['punygrad_time']:.4f}s")
        print(f"  PyTorch Time:  {r['pytorch_time']:.4f}s")
        print(f"  {r['faster']} is {r['speedup']:.1f}x faster") 