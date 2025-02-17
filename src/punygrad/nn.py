import random
from typing import List, Sequence, Union
from punygrad.engine import Scalar


class Neuron:
    """A single neuron with tanh activation.

    Implements a neuron that performs a weighted sum of inputs
    followed by a tanh activation function.
    """

    def __init__(self, nin: int) -> None:
        """Initialize neuron with random weights and bias.

        Args:
            nin: Number of input features
        """
        self.weights = [Scalar(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Scalar(random.uniform(-1, 1))

    def __call__(self, inputs: Sequence[Union[Scalar, float]]) -> Scalar:
        """Forward pass of the neuron.

        Args:
            inputs: Input values to process

        Returns:
            Output after weighted sum and tanh activation
        """
        res = sum((xi * wi for xi, wi in zip(inputs, self.weights)), self.bias)
        return res.tanh()

    def parameters(self) -> List[Scalar]:
        """Get all trainable parameters of the neuron."""
        return self.weights + [self.bias]


class Layer:
    """A layer of neurons with parallel processing."""

    def __init__(self, nin: int, nout: int) -> None:
        """Initialize a layer with specified input and output dimensions.

        Args:
            nin: Number of input features
            nout: Number of neurons in this layer
        """
        self.layer = [Neuron(nin) for _ in range(nout)]

    def __call__(
        self, inputs: Sequence[Union[Scalar, float]]
    ) -> Union[Scalar, List[Scalar]]:
        """Forward pass of the layer.

        Args:
            inputs: Input values to process

        Returns:
            Single Scalar if nout=1, else list of outputs from each neuron
        """
        out = [n(inputs) for n in self.layer]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> List[Scalar]:
        """Get all trainable parameters of the layer."""
        return [p for n in self.layer for p in n.parameters()]


class MLP:
    """Multi-layer perceptron with tanh activations."""

    def __init__(self, nin: int, nouts: List[int]) -> None:
        """Initialize MLP with given architecture.

        Args:
            nin: Number of input features
            nouts: List of neuron counts for each layer
        """
        layers = [nin] + nouts
        self.network = [Layer(layers[i - 1], layers[i]) for i in range(1, len(layers))]

    def __call__(
        self, inputs: Sequence[Union[Scalar, float]]
    ) -> Union[Scalar, List[Scalar]]:
        """Forward pass of the network.

        Args:
            inputs: Input values to process

        Returns:
            Network output after processing through all layers
        """
        for layer in self.network:
            inputs = layer(inputs)
        return inputs

    def parameters(self) -> List[Scalar]:
        """Get all trainable parameters of the network."""
        return [p for l in self.network for p in l.parameters()]
