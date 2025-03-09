"""Tests for the Tensor class with broadcasting capabilities."""

import pytest
import numpy as np
import torch
from punygrad.engine import Tensor, Scalar


def test_tensor_init():
    """Test tensor initialization and shape calculation using numpy arrays for validation."""
    # Generate test cases using numpy arrays of different shapes
    test_arrays = [
        np.array([[1, 2], [3, 4]]),  # 2D array
        np.array([1, 2, 3]),  # 1D array
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),  # 3D array
        np.array([]),  # Empty array
        np.array([[]]),  # Empty row
        np.ones((2, 3, 4, 5)),  # 4D array
    ]
    
    for np_array in test_arrays:
        # Convert numpy array to nested list
        nested_list = np_array.tolist()
        
        # Create tensor
        t = Tensor(nested_list)
        
        # Check shape matches numpy
        assert t.shape == np_array.shape, f"Shape mismatch for {np_array}: expected {np_array.shape}, got {t.shape}"
        
        # Convert tensor back to list for value comparison
        if np_array.size > 0:  # Skip empty arrays
            t_list = t.to_list()
            np_list = np_array.tolist()
            assert t_list == np_list, f"Value mismatch: expected {np_list}, got {t_list}"


def test_broadcast_shape():
    """Test the broadcast_shape static method against numpy's broadcasting."""
    # Generate test cases using numpy arrays of different shapes
    shape_pairs = [
        ((2, 3), (2, 3)),  # Same shapes
        ((), (2, 3)),  # Broadcasting scalar to array
        ((1, 3), (2, 3)),  # Broadcasting with dimension 1
        ((2, 1), (1, 3)),  # Broadcasting in multiple dimensions
        ((5, 1, 4), (3, 4)),  # Broadcasting with different ranks
        ((1, 1, 1), (3, 4, 5)),  # Broadcasting all dimensions
    ]
    
    for shape1, shape2 in shape_pairs:
        # Create numpy arrays
        a = np.ones(shape1)
        b = np.ones(shape2)
        
        try:
            # Get numpy's broadcast shape
            np_broadcast_shape = np.broadcast(a, b).shape
            
            # Test our implementation
            result = Tensor.broadcast_shape(shape1, shape2)
            
            # Check result matches numpy
            assert result == np_broadcast_shape, f"Broadcast shape mismatch: expected {np_broadcast_shape}, got {result}"
        except ValueError:
            # If numpy can't broadcast, our implementation should raise an error too
            with pytest.raises(RuntimeError):
                Tensor.broadcast_shape(shape1, shape2)


def test_broadcast_shape_errors():
    """Test that broadcast_shape raises errors for incompatible shapes."""
    # Generate test cases that should fail
    incompatible_shapes = [
        ((2, 3), (2, 4)),  # Incompatible dimensions
        ((3, 2), (2, 2)),  # Incompatible dimensions
        ((2, 3, 4), (3, 4, 5)),  # Incompatible dimensions in 3D
    ]
    
    for shape1, shape2 in incompatible_shapes:
        # Create numpy arrays
        a = np.ones(shape1)
        b = np.ones(shape2)
        
        # Check numpy raises ValueError
        with pytest.raises(ValueError):
            c = a + b
        
        # Check our implementation raises RuntimeError
        with pytest.raises(RuntimeError):
            Tensor.broadcast_shape(shape1, shape2)


def test_broadcast_to():
    """Test broadcasting a tensor to a new shape against numpy's broadcasting."""
    # Generate test cases
    test_cases = [
        (np.array([[1, 2]]), (3, 2)),  # Broadcast rows
        (np.array([[1], [2]]), (2, 3)),  # Broadcast columns
        (np.array([1, 2]), (2, 2)),  # Broadcast 1D to 2D
        (np.array([[[1]]]), (2, 3, 1)),  # Broadcast 3D
        (np.ones((1, 1, 1)), (2, 3, 4)),  # Broadcast all dimensions
    ]
    
    for np_array, target_shape in test_cases:
        # Create our tensor from numpy array
        t = Tensor(np_array.tolist())
        
        # Broadcast using our implementation
        bt = t.broadcast_to(target_shape)
        
        # Check shape
        assert bt.shape == target_shape, f"Shape mismatch: expected {target_shape}, got {bt.shape}"
        
        # Broadcast using numpy
        np_broadcast = np.broadcast_to(np_array, target_shape)
        
        # Convert tensor back to list for comparison
        bt_list = bt.to_list()
        np_list = np_broadcast.tolist()
        
        # Check values match numpy's broadcast
        assert bt_list == np_list, f"Value mismatch: expected {np_list}, got {bt_list}"


def test_broadcast_with():
    """Test broadcasting two tensors together against numpy's broadcasting."""
    # Generate test cases
    test_cases = [
        (np.array([[1, 2]]), np.array([[3], [4]])),  # (1, 2) with (2, 1) -> (2, 2)
        (np.array([1, 2, 3]), np.array([[4], [5]])),  # (3,) with (2, 1) -> (2, 3)
        (np.array([[1]]), np.array([2, 3, 4])),  # (1, 1) with (3,) -> (1, 3)
        (np.ones((2, 1, 3)), np.ones((1, 4, 1))),  # (2, 1, 3) with (1, 4, 1) -> (2, 4, 3)
    ]
    
    for np_array1, np_array2 in test_cases:
        # Create our tensors
        t1 = Tensor(np_array1.tolist())
        t2 = Tensor(np_array2.tolist())
        
        # Broadcast them
        bt1, bt2 = t1.broadcast_with(t2)
        
        # Check they have the same shape
        assert bt1.shape == bt2.shape
        
        # Get the expected broadcast shape from numpy
        expected_shape = np.broadcast(np_array1, np_array2).shape
        
        # Check our broadcast shape matches numpy's
        assert bt1.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {bt1.shape}"
        
        # Broadcast using numpy
        np_broadcast1 = np.broadcast_to(np_array1, expected_shape)
        np_broadcast2 = np.broadcast_to(np_array2, expected_shape)
        
        # Convert tensors back to list for comparison
        bt1_list = bt1.to_list()
        bt2_list = bt2.to_list()
        np_list1 = np_broadcast1.tolist()
        np_list2 = np_broadcast2.tolist()
        
        # Check values match numpy's broadcast
        assert bt1_list == np_list1, f"Value mismatch for tensor 1: expected {np_list1}, got {bt1_list}"
        assert bt2_list == np_list2, f"Value mismatch for tensor 2: expected {np_list2}, got {bt2_list}"


def test_broadcast_with_torch():
    """Test broadcasting against PyTorch's broadcasting."""
    # Generate test cases
    test_cases = [
        (torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0], [4.0]])),  # (1, 2) with (2, 1) -> (2, 2)
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([[4.0], [5.0]])),  # (3,) with (2, 1) -> (2, 3)
        (torch.tensor([[1.0]]), torch.tensor([2.0, 3.0, 4.0])),  # (1, 1) with (3,) -> (1, 3)
    ]
    
    for torch_t1, torch_t2 in test_cases:
        # Create our tensors from torch tensors
        t1 = Tensor(torch_t1.tolist())
        t2 = Tensor(torch_t2.tolist())
        
        # Broadcast them
        bt1, bt2 = t1.broadcast_with(t2)
        
        # PyTorch broadcasting happens implicitly during operations
        torch_result = torch_t1 + torch_t2
        
        # Check shapes match
        assert bt1.shape == tuple(torch_result.shape), f"Shape mismatch: expected {tuple(torch_result.shape)}, got {bt1.shape}"
        
        # Check values match torch's broadcast
        bt1_list = bt1.to_list()
        bt2_list = bt2.to_list()
        
        for i in range(bt1.shape[0]):
            for j in range(bt1.shape[1] if len(bt1.shape) > 1 else 1):
                if len(bt1.shape) > 1:
                    assert bt1_list[i][j] == torch_t1.expand(bt1.shape)[i, j].item()
                    assert bt2_list[i][j] == torch_t2.expand(bt2.shape)[i, j].item()
                else:
                    assert bt1_list[i] == torch_t1.expand(bt1.shape)[i].item()
                    assert bt2_list[i] == torch_t2.expand(bt2.shape)[i].item() 