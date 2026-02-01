"""
Data Generation Utilities

Generate synthetic vector datasets for testing and benchmarking.
"""

import numpy as np
from typing import Tuple, Optional


def generate_random_vectors(
    n_vectors: int,
    dimension: int,
    normalize: bool = False,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random vectors from standard normal distribution.
    
    Args:
        n_vectors: Number of vectors to generate
        dimension: Vector dimension
        normalize: If True, normalize to unit length
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_vectors, dimension)
    """
    if seed is not None:
        np.random.seed(seed)
    
    vectors = np.random.randn(n_vectors, dimension)
    
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
    
    return vectors


def generate_clustered_vectors(
    n_vectors: int,
    dimension: int,
    n_clusters: int = 10,
    cluster_std: float = 0.1,
    normalize: bool = False,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate vectors clustered around random centroids.
    
    More realistic for testing nearest neighbor search.
    
    Args:
        n_vectors: Total number of vectors
        dimension: Vector dimension
        n_clusters: Number of clusters
        cluster_std: Standard deviation within clusters
        normalize: Normalize vectors
        seed: Random seed
        
    Returns:
        Tuple of (vectors, cluster_labels)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate cluster centroids
    centroids = np.random.randn(n_clusters, dimension)
    if normalize:
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    
    # Assign vectors to clusters
    labels = np.random.randint(0, n_clusters, n_vectors)
    
    # Generate vectors around centroids
    vectors = np.zeros((n_vectors, dimension))
    for i in range(n_vectors):
        cluster = labels[i]
        vectors[i] = centroids[cluster] + np.random.randn(dimension) * cluster_std
    
    if normalize:
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    return vectors, labels


def generate_query_with_known_neighbors(
    database: np.ndarray,
    n_neighbors: int = 5,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a query vector with known ground truth neighbors.
    
    Useful for testing search accuracy.
    
    Args:
        database: Database vectors
        n_neighbors: Number of neighbors to base query on
        noise_level: Amount of noise to add
        
    Returns:
        Tuple of (query_vector, expected_neighbor_indices)
    """
    # Pick random neighbors
    n_vectors = database.shape[0]
    neighbor_indices = np.random.choice(n_vectors, n_neighbors, replace=False)
    
    # Create query as weighted average of neighbors + noise
    weights = np.random.dirichlet(np.ones(n_neighbors))
    query = np.sum(database[neighbor_indices] * weights[:, np.newaxis], axis=0)
    query += np.random.randn(database.shape[1]) * noise_level
    
    return query, neighbor_indices


def split_database(
    vectors: np.ndarray,
    train_ratio: float = 0.9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split vectors into database and query sets.
    
    Args:
        vectors: All vectors
        train_ratio: Fraction for database
        
    Returns:
        (database, queries)
    """
    n_train = int(len(vectors) * train_ratio)
    indices = np.random.permutation(len(vectors))
    
    return vectors[indices[:n_train]], vectors[indices[n_train:]]
