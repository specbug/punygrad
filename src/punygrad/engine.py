import math
from typing import Union, Set, List, Tuple, Callable


class Scalar:
    """A scalar value with automatic differentiation capabilities.

    This class implements a scalar value that tracks its computational history
    and supports automatic differentiation through various mathematical operations.

    Attributes:
            data: The scalar value
            label: Optional label for debugging
            grad: The gradient value (initialized to 0.0)
            op: The operation that created this scalar (read-only)
            prev: The set of parent scalars (read-only)
    """

    def __init__(
        self,
        data: Union[int, float],
        _children: Tuple["Scalar", ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        """Initialize a new Scalar instance.

        Args:
                data: The scalar value
                _children: Parent scalars that created this scalar (for backprop)
                _op: The operation that created this scalar
                label: Optional label for the scalar
        """
        self.__data = float(data)
        self.__label = label
        self.__grad = 0.0
        self.__op = _op
        self.__prev = set(_children)
        self.__backward_fn = lambda: None

    @property
    def _backward_fn(self) -> Callable[[], None]:
        """Get the backward function for gradient computation."""
        return self.__backward_fn

    @_backward_fn.setter
    def _backward_fn(self, fn: Callable[[], None]) -> None:
        """Set the backward function for gradient computation.

        Args:
                fn: The backward function to set

        Raises:
                TypeError: If fn is not callable
        """
        if not callable(fn):
            raise TypeError("Backward function must be callable")
        self.__backward_fn = fn

    @property
    def data(self) -> float:
        """Get the scalar value."""
        return self.__data

    @data.setter
    def data(self, value: Union[int, float]) -> None:
        """Set the scalar value.

        Args:
                value: The new scalar value

        Raises:
                TypeError: If value is not a number
                ValueError: If value is not finite
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Data must be a number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Data must be finite")
        self.__data = value

    @property
    def label(self) -> str:
        """Get the scalar's label."""
        return self.__label

    @label.setter
    def label(self, value: str) -> None:
        """Set the scalar's label.

        Args:
                value: The new label

        Raises:
                TypeError: If value is not a string
        """
        if not isinstance(value, str):
            raise TypeError("Label must be a string")
        self.__label = value

    @property
    def grad(self) -> float:
        """Get the gradient value."""
        return self.__grad

    @grad.setter
    def grad(self, value: Union[int, float]) -> None:
        """Set the gradient value.

        Args:
                value: The new gradient value

        Raises:
                TypeError: If value is not a number
                ValueError: If value is not finite
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Gradient must be a number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Gradient must be finite")
        self.__grad = value

    @property
    def op(self) -> str:
        """Get the operation that created this scalar (read-only)."""
        return self.__op

    @property
    def prev(self) -> Set["Scalar"]:
        """Get the set of parent scalars (read-only)."""
        return self.__prev.copy()  # Return a copy to prevent modification

    def __repr__(self) -> str:
        """Return a string representation of the scalar."""
        return f"Scalar({self.data})"

    # Basic arithmetic operations
    def __add__(self, other: Union["Scalar", int, float]) -> "Scalar":
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward_fn = _backward
        return out

    def __radd__(self, other: Union[int, float]) -> "Scalar":
        return self + other

    def __neg__(self) -> "Scalar":
        return self * -1

    def __sub__(self, other: Union["Scalar", int, float]) -> "Scalar":
        return self + (-other)

    def __rsub__(self, other: Union[int, float]) -> "Scalar":
        return other + (-self)

    def __mul__(self, other: Union["Scalar", int, float]) -> "Scalar":
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward_fn = _backward
        return out

    def __rmul__(self, other: Union[int, float]) -> "Scalar":
        return self * other

    def __truediv__(self, other: Union["Scalar", int, float]) -> "Scalar":
        return self * other**-1

    def __rtruediv__(self, other: Union[int, float]) -> "Scalar":
        return other * self**-1

    def __pow__(self, other: Union[int, float]) -> "Scalar":
        assert isinstance(other, (int, float)), "only supporting int or float powers"
        out = Scalar(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward_fn = _backward
        return out

    def exp(self) -> "Scalar":
        """Compute the exponential of this scalar."""
        out = Scalar(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward_fn = _backward
        return out

    def tanh(self) -> "Scalar":
        """Compute the hyperbolic tangent of this scalar."""
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Scalar(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward_fn = _backward

        return out

    def backprop(self) -> None:
        """Perform backpropagation starting from this scalar.

        This method computes gradients for all scalars that contributed
        to the computation of this scalar's value.
        """
        topo: List[Scalar] = []
        visited: Set[Scalar] = set()

        def _sort(node: "Scalar") -> None:
            if node not in visited:
                visited.add(node)
                for child in node.prev:
                    _sort(child)
                topo.append(node)

        _sort(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward_fn()
