import torch
import numpy as np
import matplotlib.pyplot as plt

class RANSAC:
    def __init__(self, model_class, max_iterations=100, threshold=1.0, min_samples=None):
        """
        RANSAC implementation for robust model fitting
        
        Args:
            model_class: A class with fit and predict methods
            max_iterations: Maximum number of RANSAC iterations
            threshold: Maximum distance for a point to be considered an inlier
            min_samples: Minimum number of samples to fit the model (defaults to model's minimum)
        """
        self.model_class = model_class
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.min_samples = min_samples
        self.best_model = None
        self.best_inliers = None
    
    def fit(self, X, y):
        """
        Perform RANSAC to find the best model
        
        Args:
            X: Input features (torch.Tensor)
            y: Target values (torch.Tensor)
        """
        n_samples = X.shape[0]
        
        # If min_samples not specified, use model's default
        if self.min_samples is None:
            self.min_samples = self.model_class.min_samples
        elif self.min_samples > 1:
            self.min_samples = int(self.min_samples)
        elif self.min_samples < 1 and self.min_samples > 0:
            self.min_samples = int(self.min_samples * n_samples)
        else:
            raise ValueError("min_samples must be a positive number")
        
        max_inliers = 0
        
        for _ in range(self.max_iterations):
            # Randomly sample minimal subset
            subset_indices = torch.randperm(n_samples)[:self.min_samples]
            X_subset = X[subset_indices]
            y_subset = y[subset_indices]
            
            # Fit model to subset
            current_model = self.model_class()
            current_model.fit(X_subset, y_subset)
            
            # Predict for all points
            y_pred = current_model.predict(X)
            
            # Calculate residuals
            residuals = torch.abs(y_pred - y)
            
            # Find inliers
            inliers = residuals <= self.threshold
            n_inliers = inliers.sum().item()
            
            # Update best model if more inliers found
            if n_inliers > max_inliers:
                max_inliers = n_inliers
                self.best_model = current_model
                self.best_inliers = inliers
        
        return self
    
    def predict(self, X):
        """
        Predict using the best model found
        
        Args:
            X: Input features (torch.Tensor)
        
        Returns:
            Predictions from the best model
        """
        if self.best_model is None:
            raise ValueError("RANSAC has not been fit yet")
        return self.best_model.predict(X)

class LinearRegressionModel:
    """
    Simple linear regression model for RANSAC
    """
    min_samples = 2
    
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y):
        """
        Fit linear regression using least squares
        
        Args:
            X: Input features (torch.Tensor)
            y: Target values (torch.Tensor)
        """
        # Add bias term
        X_with_bias = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], dim=1)
        
        # Least squares solution
        self.weights = torch.linalg.lstsq(X_with_bias, y).solution
        return self
    
    def predict(self, X):
        """
        Predict using learned weights
        
        Args:
            X: Input features (torch.Tensor)
        
        Returns:
            Predictions
        """
        # Add bias term
        X_with_bias = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], dim=1)
        return X_with_bias @ self.weights

def generate_noisy_data(n_samples=100, noise_ratio=0.3):
    """
    Generate synthetic data with outliers
    
    Args:
        n_samples: Total number of data points
        noise_ratio: Proportion of outlier points
    
    Returns:
        X, y tensors with ground truth model
    """
    # True line: y = 2x + 1
    torch.manual_seed(42)
    
    # Clean data points
    clean_samples = int(n_samples * (1 - noise_ratio))
    X_clean = torch.rand(clean_samples, 1) * 10
    y_clean = 2 * X_clean + 1 + torch.randn(clean_samples, 1) * 0.5
    
    # Outlier points (random noise)
    outlier_samples = n_samples - clean_samples
    X_outliers = torch.rand(outlier_samples, 1) * 10
    y_outliers = torch.rand(outlier_samples, 1) * 20
    
    # Combine data
    X = torch.cat([X_clean, X_outliers])
    y = torch.cat([y_clean, y_outliers])
    
    return X, y

def main():
    # Generate noisy data
    X, y = generate_noisy_data()
    X = X.cuda()
    y = y.cuda()
    
    # Perform RANSAC
    ransac = RANSAC(
        model_class=LinearRegressionModel, 
        max_iterations=100, 
        threshold=1.0, 
        min_samples=0.5
    )
    ransac.fit(X, y)
    
    # Predict using RANSAC model
    y_pred = ransac.predict(X)
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, c='blue', label='Data Points', alpha=0.5)
    plt.scatter(X[~ransac.best_inliers], y[~ransac.best_inliers], 
                c='red', label='Outliers', alpha=0.5)
    
    # Plot true line
    x_line = torch.tensor([[0], [10]])
    y_line = ransac.best_model.predict(x_line)
    plt.plot(x_line, y_line, c='green', label='RANSAC Line')
    
    plt.title('RANSAC Linear Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    # save the plot
    plt.savefig("tmp/ransac_linear_regression.png")

if __name__ == '__main__':
    main()