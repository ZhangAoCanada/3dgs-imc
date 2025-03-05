import torch
import torch.nn.functional as F
from torch import nn

def solve_linear_equation(X, Y):
    """
    Solve the equation aX + b = Y using a direct least squares approach
    
    Args:
    X (torch.Tensor): Input tensor of shape (N, 3)
    Y (torch.Tensor): Target tensor of shape (N, 3)
    
    Returns:
    tuple: (a, b) - scalar values that minimize the least squares error
    """
    # Ensure X and Y are float tensors
    X = X.float()
    Y = Y.float()

    X = X.mean(dim=1, keepdim=True)
    Y = Y.mean(dim=1, keepdim=True)
    
    # Augment X with a column of ones for the bias term
    X_augmented = torch.cat([X, torch.ones(X.size(0), 1, dtype=X.dtype, device=X.device)], dim=1)
    
    # Use torch.lstsq for solving the linear least squares problem
    # We want to solve the equation: Y = aX + b
    # This is equivalent to solving: [a, b] = argmin ||aX + b - Y||^2
    solution = torch.linalg.lstsq(X_augmented, Y).solution
    
    # Extract a and b
    a = solution[0].item()  # Last values are for bias
    b = solution[1].item()
    
    return a, b
