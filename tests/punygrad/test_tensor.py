"""Tests for the Tensor class basic functionality."""

import pytest
from punygrad.engine import Tensor, Scalar


def test_tensor_init():
    """Test tensor initialization and shape calculation."""
    test_arrays = [
        [[1, 2], [3, 4]],  # 2D array
        [1, 2, 3],  # 1D array
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],  # 3D array
        [],  # Empty array
        [[]],  # Empty row
    ]
    
    for array in test_arrays:
        t = Tensor(array)
        
        # Check shape calculation
        expected_shape = get_shape(array)
        assert t.shape == expected_shape, f"Shape mismatch for {array}: expected {expected_shape}, got {t.shape}"
        
        # Check tensor data structure
        check_tensor_structure(t.tensor, array)


def test_tensor_repr():
    """Test the string representation of tensors."""
    test_arrays = [
        [[1, 2], [3, 4]],  # 2D array
        [1, 2, 3],  # 1D array
        [],  # Empty array
    ]
    
    for array in test_arrays:
        t = Tensor(array)
        repr_str = repr(t)
        
        # Check that repr starts with "Tensor("
        assert repr_str.startswith("Tensor("), f"Repr should start with 'Tensor(', got {repr_str}"
        
        # Check that repr ends with ")"
        assert repr_str.endswith(")"), f"Repr should end with ')', got {repr_str}"


def test_broadcast_shape():
    """Test the broadcast_shape static method."""
    test_cases = [
        ((2, 3), (2, 3), (2, 3)),  # Same shapes
        ((), (2, 3), (2, 3)),  # Broadcasting scalar to array
        ((1, 3), (2, 3), (2, 3)),  # Broadcasting with dimension 1
        ((2, 1), (1, 3), (2, 3)),  # Broadcasting in multiple dimensions
        ((5, 1, 4), (3, 4), (5, 3, 4)),  # Broadcasting with different ranks
    ]
    
    for shape1, shape2, expected in test_cases:
        result = Tensor.broadcast_shape(shape1, shape2)
        assert result == expected, f"Expected {expected}, got {result}"


def test_broadcast_shape_errors():
    """Test that broadcast_shape raises errors for incompatible shapes."""
    incompatible_shapes = [
        ((2, 3), (2, 4)),  # Incompatible dimensions
        ((3, 2), (2, 2)),  # Incompatible dimensions
        ((2, 3, 4), (3, 4, 5)),  # Incompatible dimensions in 3D
    ]
    
    for shape1, shape2 in incompatible_shapes:
        with pytest.raises(RuntimeError):
            Tensor.broadcast_shape(shape1, shape2)


def test_broadcast_to():
    """Test broadcasting a tensor to a new shape."""
    test_cases = [
        ([[1, 2]], (3, 2)),  # Broadcast rows
        ([[1], [2]], (2, 3)),  # Broadcast columns
        ([1, 2], (2, 2)),  # Broadcast 1D to 2D
        ([[[1]]], (2, 3, 1)),  # Broadcast 3D
    ]
    
    for array, target_shape in test_cases:
        t = Tensor(array)
        bt = t.broadcast_to(target_shape)
        
        # Check shape
        assert bt.shape == target_shape, f"Shape mismatch: expected {target_shape}, got {bt.shape}"
        
        # Check values manually
        check_broadcast_values(bt.to_list(), array, target_shape)


def test_broadcast_with():
    """Test broadcasting two tensors together."""
    test_cases = [
        ([[1, 2]], [[3], [4]], (2, 2)),  # (1, 2) with (2, 1) -> (2, 2)
        ([1, 2, 3], [[4], [5]], (2, 3)),  # (3,) with (2, 1) -> (2, 3)
        ([[1]], [2, 3, 4], (1, 3)),  # (1, 1) with (3,) -> (1, 3)
    ]
    
    for array1, array2, expected_shape in test_cases:
        t1 = Tensor(array1)
        t2 = Tensor(array2)
        
        # Broadcast them
        bt1, bt2 = t1.broadcast_with(t2)
        
        # Check they have the same shape
        assert bt1.shape == bt2.shape, f"Broadcast tensors should have the same shape, got {bt1.shape} and {bt2.shape}"
        assert bt1.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {bt1.shape}"


def test_to_list():
    """Test converting a tensor to a list."""
    test_arrays = [
        [[1, 2], [3, 4]],  # 2D array
        [1, 2, 3],  # 1D array
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],  # 3D array
        [],  # Empty array
    ]
    
    for array in test_arrays:
        t = Tensor(array)
        result = t.to_list()
        assert result == array, f"Expected {array}, got {result}"


# Helper functions

def get_shape(array):
    """Calculate the shape of a nested list."""
    if not isinstance(array, list):
        return ()
    if not array:
        return (0,)
    
    shape = [len(array)]
    if isinstance(array[0], list):
        shape.extend(get_shape(array[0]))
    
    return tuple(shape)


def check_tensor_structure(tensor_data, array_data):
    """Check that the tensor data structure matches the input array."""
    if not isinstance(array_data, list):
        assert isinstance(tensor_data, Scalar), f"Expected Scalar, got {type(tensor_data)}"
        assert tensor_data.data == array_data, f"Expected {array_data}, got {tensor_data.data}"
        return
    
    assert isinstance(tensor_data, list), f"Expected list, got {type(tensor_data)}"
    assert len(tensor_data) == len(array_data), f"Expected length {len(array_data)}, got {len(tensor_data)}"
    
    for t, a in zip(tensor_data, array_data):
        check_tensor_structure(t, a)


def check_broadcast_values(broadcast_data, original_data, target_shape):
    """Check that the broadcast values are correct."""
    # This is a simplified check for common broadcasting patterns
    if not isinstance(original_data, list):
        # Scalar case
        for i in range(target_shape[0]):
            assert broadcast_data[i] == original_data, f"Expected {original_data}, got {broadcast_data[i]}"
        return
    
    if len(original_data) == 1:
        # Broadcasting a single element
        for i in range(target_shape[0]):
            if isinstance(original_data[0], list):
                check_broadcast_values(broadcast_data[i], original_data[0], target_shape[1:])
            else:
                assert broadcast_data[i] == original_data[0], f"Expected {original_data[0]}, got {broadcast_data[i]}"
    elif len(target_shape) > len(get_shape(original_data)):
        # Broadcasting to higher dimensions
        for i in range(target_shape[0]):
            check_broadcast_values(broadcast_data[i], original_data, target_shape[1:])
    else:
        # Other cases
        for i in range(len(original_data)):
            if i < len(broadcast_data):
                if isinstance(original_data[i], list):
                    check_broadcast_values(broadcast_data[i], original_data[i], target_shape[1:])
                else:
                    assert broadcast_data[i] == original_data[i], f"Expected {original_data[i]}, got {broadcast_data[i]}" 