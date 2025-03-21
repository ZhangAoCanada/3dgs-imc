import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import numpy as np

class GPUDBSCAN:
    def __init__(self, eps, min_samples, device=None, chunk_size=10000):
        """
        Memory-efficient GPU-accelerated DBSCAN clustering implementation
        
        Args:
            eps (float): Maximum distance between two samples
            min_samples (int): Minimum samples to form a dense region
            device (torch.device, optional): GPU device to use
            chunk_size (int): Size of chunks for memory-efficient computation
        """
        self.eps = eps
        self.min_samples = min_samples
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chunk_size = chunk_size
    
    def _memory_efficient_pairwise_distances(self, X):
        """
        Compute pairwise distances with minimal memory usage
        
        Args:
            X (torch.Tensor): Input data points
        
        Returns:
            torch.Tensor: Sparse neighborhood matrix
        """
        n_points = X.shape[0]
        neighborhoods = []
        
        # Process in chunks to reduce memory consumption
        for i in range(0, n_points, self.chunk_size):
            chunk = X[i:i+self.chunk_size]
            chunk_neighborhoods = []
            
            # Compare chunk with full dataset in smaller batches
            for j in range(0, n_points, self.chunk_size):
                comparison_chunk = X[j:j+self.chunk_size]
                
                # Compute distances for this subset
                dist_matrix = torch.cdist(chunk, comparison_chunk)
                
                # Find neighborhoods efficiently
                neighborhood_mask = dist_matrix <= self.eps
                
                # Convert to list of indices for each point in chunk
                for k in range(neighborhood_mask.shape[0]):
                    local_neighbors = (neighborhood_mask[k].nonzero(as_tuple=False) + j).squeeze()
                    chunk_neighborhoods.append(local_neighbors.tolist())
            
            neighborhoods.extend(chunk_neighborhoods)
        
        return neighborhoods
    
    def fit(self, X):
        """
        Perform DBSCAN clustering with memory-efficient approach
        
        Args:
            X (torch.Tensor or np.ndarray): Input data points
        
        Returns:
            torch.Tensor: Cluster labels for each point
        """
        # Find neighborhoods with minimal memory usage
        neighborhoods = self._memory_efficient_pairwise_distances(X)
        
        # Initialize cluster labels
        labels = torch.full((X.shape[0],), -1, dtype=torch.long, device=X.device)
        current_cluster = 0
        
        # DBSCAN clustering algorithm
        for point in range(X.shape[0]):
            if labels[point] != -1:
                continue
            
            # Check if point is a core point
            if len(neighborhoods[point]) >= self.min_samples:
                labels[point] = current_cluster
                
                # Expand cluster
                seed_points = set(neighborhoods[point])
                while seed_points:
                    current_seed = seed_points.pop()
                    
                    if labels[current_seed] == -1:
                        labels[current_seed] = current_cluster
                    
                    # Expand neighborhood if core point
                    if len(neighborhoods[current_seed]) >= self.min_samples:
                        for neighbor in neighborhoods[current_seed]:
                            if labels[neighbor] == -1:
                                seed_points.add(neighbor)
                
                current_cluster += 1
            else:
                # Mark as noise
                labels[point] = -1
        
        return labels.cpu()

# Example usage
def main():
    # Set up a large dataset
    torch.manual_seed(42)
    X = torch.randn(100000, 3, device='cuda')
    
    # Initialize DBSCAN with lower memory footprint
    dbscan = GPUDBSCAN(eps=0.5, min_samples=5, device='cuda', chunk_size=1024)
    
    # Perform clustering
    clusters = dbscan.fit(X)
    
    # Print cluster statistics
    unique_clusters = torch.unique(clusters)
    print(f"Number of clusters found: {len(unique_clusters)}")
    for cluster in unique_clusters:
        if cluster != -1:
            print(f"Cluster {cluster}: {(clusters == cluster).sum().item()} points")
    print(f"Noise points: {(clusters == -1).sum().item()} points")

if __name__ == "__main__":
    main()