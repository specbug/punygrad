"""Tests for the Scalar autograd engine."""

import pytest
from punygrad.engine import Scalar


def test_scalar_creation():
    """Test basic Scalar creation and properties."""
    x = Scalar(2.0)
    assert x.data == 2.0
    assert x.grad == 0.0
    assert x.op == ""
    assert len(x.prev) == 0


def test_scalar_addition():
    """Test Scalar addition and gradient computation."""
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = x + y
    z.backprop()

    assert z.data == 5.0
    assert x.grad == 1.0
    assert y.grad == 1.0
    assert z.op == "+"


def test_scalar_multiplication():
    """Test Scalar multiplication and gradient computation."""
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = x * y
    z.backprop()

    assert z.data == 6.0
    assert x.grad == 3.0
    assert y.grad == 2.0
    assert z.op == "*"


def test_scalar_power():
    """Test Scalar power operation and gradient computation."""
    x = Scalar(2.0)
    z = x**3
    z.backprop()

    assert z.data == 8.0
    assert x.grad == 12.0
    assert z.op == "**3"


def test_scalar_tanh():
    """Test tanh activation and its gradient."""
    x = Scalar(0.0)
    z = x.tanh()
    z.backprop()

    assert abs(z.data) < 1e-6  # tanh(0) = 0
    assert abs(z.grad - 1.0) < 1e-6
    assert z.op == "tanh"


def test_invalid_operations():
    """Test invalid operations raise appropriate errors."""
    x = Scalar(2.0)

    with pytest.raises(TypeError):
        x.data = "invalid"

    with pytest.raises(TypeError):
        x.grad = "invalid"

    with pytest.raises(TypeError):
        x.label = 123
