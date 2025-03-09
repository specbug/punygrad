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


class Tensor:
    """A multi-dimensional array of Scalar objects with broadcasting capabilities.
    
    This class implements a tensor (n-dimensional array) of Scalar objects,
    providing functionality similar to NumPy arrays or PyTorch tensors but
    with automatic differentiation capabilities through the Scalar class.
    
    The Tensor class supports:
    - Creation from nested lists
    - Shape information
    - Broadcasting operations (similar to NumPy broadcasting)
    
    Attributes:
        shape: A tuple representing the dimensions of the tensor
        tensor: The underlying nested list of Scalar objects
    """

    def __init__(self, arr):
        """Initialize a new Tensor instance.
        
        Args:
            arr: The array-like object (nested lists) to convert to a tensor
                 Each element will be wrapped in a Scalar object
        """
        self.__shape = self.__construct_shape(arr)
        self.__tensor = self.__construct_tensor(arr)

    def __construct_shape(self, arr):
        """Determine the shape of the input array.
        
        Recursively traverses the nested list structure to determine
        the dimensions of the tensor.
        
        Args:
            arr: The array-like object
            
        Returns:
            A tuple representing the shape of the array
        """
        # Handle empty array case
        if arr == []:
            return (0,)  # Match numpy's behavior for empty arrays
            
        shape = []

        def helper(a):
            nonlocal shape
            if not isinstance(a, list):
                return
            shape.append(len(a))
            if a and isinstance(a[0], list):
                helper(a[0])

        helper(arr)
        return tuple(shape)

    def __construct_tensor(self, arr):
        """Convert the input array to a tensor of Scalar objects.
        
        Recursively converts each element in the nested list to a Scalar object.
        
        Args:
            arr: The array-like object
            
        Returns:
            A nested list of Scalar objects
        """
        # Handle empty array case
        if arr == []:
            return []
            
        def helper(a):
            if not isinstance(a, list):
                return Scalar(a)
            return [helper(a[i]) for i in range(len(a))]

        return helper(arr)

    def __repr__(self):
        """Return a string representation of the tensor.
        
        Returns:
            A string representation showing the tensor's structure and values
        """
        if not self.shape:
            return "Tensor([])"
            
        def format_array(arr, indent=0):
            if not isinstance(arr, list):
                return str(arr)
                
            if not arr:
                return "[]"
                
            if not isinstance(arr[0], list):
                elements = ", ".join(str(x) for x in arr)
                return f"[{elements}]"
                
            result = "["
            for i, item in enumerate(arr):
                if i > 0:
                    result += " "
                result += "\n" + " " * (indent + 1) + format_array(item, indent + 1)
                if i < len(arr) - 1:
                    result += ","
            result += "\n" + " " * indent + "]"
            return result
            
        formatted = format_array(self.tensor)
        return f"Tensor({formatted})"

    @staticmethod
    def broadcast_shape(shape1, shape2):
        """Compute the broadcast shape of two shapes.
        
        Follows NumPy/PyTorch broadcasting rules:
        1. If shapes have different ranks, prepend 1s to the shorter shape
        2. For each dimension, the output dimension is the max of the input dimensions
        3. Broadcasting is valid when for each dimension, either they're equal or one is 1
        
        Args:
            shape1: First shape tuple
            shape2: Second shape tuple
            
        Returns:
            The resulting broadcast shape
            
        Raises:
            RuntimeError: If shapes cannot be broadcast together
        """
        # Reverse shapes for easier processing
        s1 = list(shape1)[::-1]
        s2 = list(shape2)[::-1]
        result = []
        
        # Process dimensions
        for i in range(max(len(s1), len(s2))):
            dim1 = s1[i] if i < len(s1) else 1
            dim2 = s2[i] if i < len(s2) else 1
            
            if dim1 == dim2 or dim1 == 1 or dim2 == 1:
                result.append(max(dim1, dim2))
            else:
                raise RuntimeError(
                    f"Tensors cannot be broadcast together, incompatible dimension at {max(len(s1), len(s2))-i-1}: "
                    f"({dim1} vs {dim2})"
                )
                
        # Return the shape in correct order
        return tuple(result[::-1])

    def broadcast_to(self, target_shape):
        """Broadcast this tensor to the target shape.
        
        Creates a new tensor with the target shape by repeating values
        according to NumPy/PyTorch broadcasting rules.
        
        Args:
            target_shape: The shape to broadcast to
            
        Returns:
            A new tensor broadcast to the target shape
            
        Raises:
            RuntimeError: If the tensor cannot be broadcast to the target shape
        """
        # If shapes are the same, return self
        if self.shape == target_shape:
            return self
            
        # Check if broadcasting is possible
        try:
            broadcast_shape = self.broadcast_shape(self.shape, target_shape)
            if broadcast_shape != target_shape:
                raise RuntimeError(f"Cannot broadcast tensor of shape {self.shape} to {target_shape}")
        except RuntimeError as e:
            raise RuntimeError(f"Cannot broadcast tensor of shape {self.shape} to {target_shape}: {str(e)}")
            
        # Create a new tensor with the result
        new_tensor = Tensor.__new__(Tensor)
        new_tensor.__shape = target_shape
        
        # Implement broadcasting in pure Python
        new_tensor.__tensor = self.__broadcast_tensor(self.tensor, self.shape, target_shape)
        return new_tensor
        
    def __broadcast_tensor(self, tensor_data, current_shape, target_shape):
        """Broadcast tensor data to target shape in pure Python.
        
        Args:
            tensor_data: The tensor data to broadcast
            current_shape: The current shape of the tensor data
            target_shape: The target shape to broadcast to
            
        Returns:
            The broadcast tensor data
        """
        # Handle broadcasting from lower to higher dimensions
        if len(target_shape) > len(current_shape):
            # Prepend 1s to the current shape
            padding = len(target_shape) - len(current_shape)
            padded_shape = (1,) * padding + current_shape
            
            # Wrap the tensor data in lists to add dimensions
            padded_data = tensor_data
            for _ in range(padding):
                padded_data = [padded_data]
                
            # Now broadcast the padded tensor
            return self.__broadcast_tensor(padded_data, padded_shape, target_shape)
            
        # Handle broadcasting within the same number of dimensions
        result = tensor_data
        
        # Process each dimension
        for dim in range(len(target_shape)):
            if current_shape[dim] == target_shape[dim]:
                # Dimension sizes match, no broadcasting needed
                continue
            elif current_shape[dim] == 1:
                # Broadcast this dimension
                result = self.__broadcast_dimension(result, dim, target_shape[dim])
            else:
                # Cannot broadcast
                raise RuntimeError(f"Cannot broadcast dimension {current_shape[dim]} to {target_shape[dim]}")
                
        return result
        
    def __broadcast_dimension(self, tensor_data, dim, size):
        """Broadcast a specific dimension of the tensor data.
        
        Args:
            tensor_data: The tensor data to broadcast
            dim: The dimension to broadcast
            size: The target size of the dimension
            
        Returns:
            The broadcast tensor data
        """
        if dim == 0:
            # Broadcast the first dimension
            return [tensor_data[0] for _ in range(size)]
        else:
            # Recursively broadcast inner dimensions
            return [self.__broadcast_dimension(item, dim - 1, size) for item in tensor_data]

    def broadcast_with(self, other):
        """Broadcast this tensor with another tensor.
        
        Broadcasts both tensors to a common shape following NumPy/PyTorch
        broadcasting rules.
        
        Args:
            other: The other tensor
            
        Returns:
            A tuple of (broadcast_self, broadcast_other)
            
        Raises:
            RuntimeError: If the tensors cannot be broadcast together
        """
        if self.shape == other.shape:
            return self, other
            
        try:
            # Compute the broadcast shape
            broadcast_shape = self.broadcast_shape(self.shape, other.shape)
            
            # Broadcast both tensors to this shape
            return self.broadcast_to(broadcast_shape), other.broadcast_to(broadcast_shape)
        except RuntimeError as e:
            raise RuntimeError(f"Cannot broadcast tensors of shapes {self.shape} and {other.shape}: {str(e)}")

    def to_list(self):
        """Convert the tensor to a nested list of Python values.
        
        Returns:
            A nested list with the same values as the tensor
        """
        def extract_data(tensor_data):
            if not isinstance(tensor_data, list):
                return tensor_data.data
            return [extract_data(x) for x in tensor_data]
        
        # Handle empty tensor
        if not self.shape:
            return []
            
        # Extract data from Scalar objects
        return extract_data(self.tensor)

    @property
    def shape(self):
        """Get the shape of the tensor.
        
        Returns:
            A tuple representing the dimensions of the tensor
        """
        return self.__shape

    @property
    def tensor(self):
        """Get the underlying tensor data.
        
        Returns:
            The nested list of Scalar objects
        """
        return self.__tensor