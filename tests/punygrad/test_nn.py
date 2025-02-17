"""Tests for neural network components."""

import pytest
import torch
import torch.nn as nn
from punygrad.nn import Neuron, Layer, MLP


def test_neuron_forward():
    """Test single neuron forward pass against PyTorch."""
    # PunyGrad neuron
    nin = 3
    neuron = Neuron(nin)
    x = [2.0, 3.0, -1.0]
    
    # Get weights and bias for PyTorch
    weights = [w.data for w in neuron.weights]
    bias = neuron.bias.data
    
    # PyTorch equivalent
    torch_neuron = nn.Sequential(
        nn.Linear(nin, 1, bias=True),
        nn.Tanh()
    )
    with torch.no_grad():
        torch_neuron[0].weight.copy_(torch.tensor(weights).reshape(1, nin))
        torch_neuron[0].bias.copy_(torch.tensor([bias]))
    
    # Compare outputs
    our_output = neuron(x)
    torch_output = torch_neuron(torch.tensor(x, dtype=torch.float32))
    
    assert abs(our_output.data - torch_output.item()) < 1e-6


def test_layer_forward():
    """Test layer forward pass against PyTorch."""
    nin, nout = 3, 4
    layer = Layer(nin, nout)
    x = [2.0, 3.0, -1.0]
    
    # Get weights and biases for PyTorch
    weights = []
    biases = []
    for n in layer.layer:
        weights.append([w.data for w in n.weights])
        biases.append(n.bias.data)
    
    # PyTorch equivalent
    torch_layer = nn.Sequential(
        nn.Linear(nin, nout, bias=True),
        nn.Tanh()
    )
    with torch.no_grad():
        torch_layer[0].weight.copy_(torch.tensor(weights))
        torch_layer[0].bias.copy_(torch.tensor(biases))
    
    # Compare outputs
    our_output = layer(x)
    torch_output = torch_layer(torch.tensor(x, dtype=torch.float32))
    
    for ours, theirs in zip(our_output, torch_output):
        assert abs(ours.data - theirs.item()) < 1e-6


def test_mlp_forward_backward():
    """Test MLP forward and backward pass against PyTorch."""
    # PunyGrad MLP
    nin = 3
    nouts = [4, 4, 1]
    mlp = MLP(nin, nouts)
    x = [2.0, 3.0, -1.0]
    
    # Create equivalent PyTorch model
    torch_mlp = nn.Sequential(
        nn.Linear(nin, nouts[0]),
        nn.Tanh(),
        nn.Linear(nouts[0], nouts[1]),
        nn.Tanh(),
        nn.Linear(nouts[1], nouts[2]),
        nn.Tanh()
    )
    
    # Copy weights and biases
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
    
    # Forward pass
    our_output = mlp(x)
    x_torch = torch.tensor(x, dtype=torch.float32)
    torch_output = torch_mlp(x_torch)
    
    # Compare forward results
    assert abs(our_output.data - torch_output.item()) < 1e-6
    
    # Backward pass
    our_output.backprop()
    torch_output.backward()
    
    # Compare gradients of first layer parameters
    first_layer = mlp.network[0].layer[0]
    torch_first_layer = torch_mlp[0]
    
    # Compare weights
    for i, our_w in enumerate(first_layer.weights):
        torch_w = torch_first_layer.weight.grad[0][i]
        assert abs(our_w.grad - torch_w.item()) < 1e-6
    
    # Compare bias
    assert abs(first_layer.bias.grad - torch_first_layer.bias.grad[0].item()) < 1e-6


def test_mlp_training_step():
    """Test MLP training step matches PyTorch."""
    # Training data
    X = [[2.0, 3.0, -1.0], [1.0, -1.0, 2.0], [0.5, 1.0, 1.0]]
    y = [1.0, -1.0, 0.5]
    
    # PunyGrad MLP
    mlp = MLP(3, [4, 4, 1])
    learning_rate = 0.1
    
    # PyTorch equivalent
    torch_mlp = nn.Sequential(
        nn.Linear(3, 4),
        nn.Tanh(),
        nn.Linear(4, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
        nn.Tanh()
    )
    optimizer = torch.optim.SGD(torch_mlp.parameters(), lr=learning_rate)
    
    # Copy initial weights
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
    
    # Single training step
    # PunyGrad
    total_loss = 0
    for xi, yi in zip(X, y):
        pred = mlp(xi)
        loss = (pred.data - yi) ** 2
        total_loss += loss
        pred.backprop()
    
    # Update weights
    for p in mlp.parameters():
        p.data -= learning_rate * p.grad
        p.grad = 0
    
    our_loss = total_loss / len(X)
    
    # PyTorch
    optimizer.zero_grad()
    torch_total_loss = 0
    for xi, yi in zip(X, y):
        pred = torch_mlp(torch.tensor(xi, dtype=torch.float32))
        loss = (pred - yi) ** 2
        torch_total_loss += loss
        loss.backward()
    
    optimizer.step()
    torch_loss = (torch_total_loss / len(X)).item()
    
    # Compare losses
    assert abs(our_loss - torch_loss) < 1e-5 