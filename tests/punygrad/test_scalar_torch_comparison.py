"""Validation tests comparing punygrad's Scalar class against PyTorch's autograd."""

import math
import pytest
import torch
from punygrad.engine import Scalar


def test_addition_against_torch():
    """Compare addition and its gradients with PyTorch."""
    # PunyGrad
    x1 = Scalar(2.0)
    y1 = Scalar(3.0)
    z1 = x1 + y1
    z1.backprop()

    # PyTorch
    x2 = torch.tensor(2.0, requires_grad=True)
    y2 = torch.tensor(3.0, requires_grad=True)
    z2 = x2 + y2
    z2.backward()

    # Compare results
    assert z1.data == z2.item()
    assert x1.grad == x2.grad.item()
    assert y1.grad == y2.grad.item()


def test_multiplication_against_torch():
    """Compare multiplication and its gradients with PyTorch."""
    # PunyGrad
    x1 = Scalar(2.0)
    y1 = Scalar(3.0)
    z1 = x1 * y1
    z1.backprop()

    # PyTorch
    x2 = torch.tensor(2.0, requires_grad=True)
    y2 = torch.tensor(3.0, requires_grad=True)
    z2 = x2 * y2
    z2.backward()

    # Compare results
    assert z1.data == z2.item()
    assert x1.grad == x2.grad.item()
    assert y1.grad == y2.grad.item()


def test_power_against_torch():
    """Compare power operation and its gradients with PyTorch."""
    # PunyGrad
    x1 = Scalar(2.0)
    z1 = x1 ** 3
    z1.backprop()

    # PyTorch
    x2 = torch.tensor(2.0, requires_grad=True)
    z2 = x2 ** 3
    z2.backward()

    # Compare results
    assert z1.data == z2.item()
    assert x1.grad == x2.grad.item()


def test_tanh_against_torch():
    """Compare tanh and its gradients with PyTorch."""
    # PunyGrad
    x1 = Scalar(0.5)
    z1 = x1.tanh()
    z1.backprop()

    # PyTorch
    x2 = torch.tensor(0.5, requires_grad=True)
    z2 = torch.tanh(x2)
    z2.backward()

    # Compare results
    assert abs(z1.data - z2.item()) < 1e-6
    assert abs(x1.grad - x2.grad.item()) < 1e-6


def test_complex_expression_against_torch():
    """Compare a more complex expression with PyTorch."""
    # PunyGrad
    x1 = Scalar(2.0)
    y1 = Scalar(3.0)
    z1 = (x1 * y1 + x1 ** 2).tanh()
    z1.backprop()

    # PyTorch
    x2 = torch.tensor(2.0, requires_grad=True)
    y2 = torch.tensor(3.0, requires_grad=True)
    z2 = torch.tanh(x2 * y2 + x2 ** 2)
    z2.backward()

    # Compare results
    assert abs(z1.data - z2.item()) < 1e-6
    assert abs(x1.grad - x2.grad.item()) < 1e-6
    assert abs(y1.grad - y2.grad.item()) < 1e-6 