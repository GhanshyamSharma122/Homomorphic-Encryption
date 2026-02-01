"""
Encrypted Vector Operations

Implements vector similarity operations on encrypted data.
These operations are performed WITHOUT decryption.
"""

import numpy as np
from typing import List, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .he_engine import HEEngine, SimulatedCiphertext


class EncryptedVectorOps:
    """
    Operations on encrypted vectors for similarity search.
    
    Supports:
    - Dot product (encrypted)
    - Euclidean distance squared (encrypted)  
    - Cosine similarity (requires normalization)
    """
    
    def __init__(self, engine: 'HEEngine'):
        """
        Initialize with HE engine.
        
        Args:
            engine: Initialized HEEngine instance
        """
        self.engine = engine
    
    def dot_product(
        self, 
        enc_vec1: 'SimulatedCiphertext', 
        enc_vec2: 'SimulatedCiphertext'
    ) -> 'SimulatedCiphertext':
        """
        Compute encrypted dot product of two encrypted vectors.
        
        Args:
            enc_vec1: First encrypted vector
            enc_vec2: Second encrypted vector
            
        Returns:
            Encrypted scalar (as vector with single element)
        """
        # Element-wise multiplication
        result = enc_vec1 * enc_vec2
        # Sum all elements to get dot product
        result = result.sum()
        return result
    
    def dot_product_plain(
        self,
        enc_vec: 'SimulatedCiphertext',
        plain_vec: Union[List[float], np.ndarray]
    ) -> 'SimulatedCiphertext':
        """
        Compute dot product of encrypted vector with plaintext vector.
        More efficient than encrypted-encrypted operation.
        
        Args:
            enc_vec: Encrypted vector
            plain_vec: Plaintext vector
            
        Returns:
            Encrypted scalar result
        """
        if isinstance(plain_vec, np.ndarray):
            plain_vec = plain_vec.tolist()
        
        result = enc_vec * plain_vec
        result = result.sum()
        return result
    
    def euclidean_distance_squared(
        self,
        enc_vec1: 'SimulatedCiphertext',
        enc_vec2: 'SimulatedCiphertext'
    ) -> 'SimulatedCiphertext':
        """
        Compute squared Euclidean distance between encrypted vectors.
        
        ||a - b||^2 = sum((a_i - b_i)^2)
        
        Note: Square root is expensive in HE, so we return squared distance.
        For nearest neighbor search, this preserves ordering.
        
        Args:
            enc_vec1: First encrypted vector
            enc_vec2: Second encrypted vector
            
        Returns:
            Encrypted squared distance
        """
        diff = enc_vec1 - enc_vec2
        diff_squared = diff * diff
        return diff_squared.sum()
    
    def euclidean_distance_squared_plain(
        self,
        enc_vec: 'SimulatedCiphertext',
        plain_vec: Union[List[float], np.ndarray]
    ) -> 'SimulatedCiphertext':
        """
        Compute squared Euclidean distance between encrypted and plaintext vector.
        
        Args:
            enc_vec: Encrypted vector
            plain_vec: Plaintext vector
            
        Returns:
            Encrypted squared distance
        """
        if isinstance(plain_vec, np.ndarray):
            plain_vec = plain_vec.tolist()
        
        diff = enc_vec - plain_vec
        diff_squared = diff * diff
        return diff_squared.sum()
    
    def batch_distances(
        self,
        enc_query: 'SimulatedCiphertext',
        database_vectors: List['SimulatedCiphertext']
    ) -> List['SimulatedCiphertext']:
        """
        Compute distances from query to all database vectors.
        
        Args:
            enc_query: Encrypted query vector
            database_vectors: List of encrypted database vectors
            
        Returns:
            List of encrypted distances
        """
        return [
            self.euclidean_distance_squared(enc_query, db_vec)
            for db_vec in database_vectors
        ]
    
    def batch_distances_plain(
        self,
        enc_query: 'SimulatedCiphertext',
        database_vectors: np.ndarray
    ) -> List['SimulatedCiphertext']:
        """
        Compute distances from encrypted query to plaintext database.
        
        This is the common scenario: encrypted query searches plaintext database.
        
        Args:
            enc_query: Encrypted query vector
            database_vectors: Plaintext database (2D numpy array)
            
        Returns:
            List of encrypted distances
        """
        return [
            self.euclidean_distance_squared_plain(enc_query, db_vec)
            for db_vec in database_vectors
        ]


class EncryptedNearestNeighbor:
    """
    Encrypted Nearest Neighbor Search.
    
    Scenario: Client encrypts query, server searches without seeing query.
    """
    
    def __init__(self, engine: 'HEEngine', database: np.ndarray):
        """
        Initialize with database.
        
        Args:
            engine: HE Engine
            database: Plaintext database vectors (N x D)
        """
        self.engine = engine
        self.ops = EncryptedVectorOps(engine)
        self.database = database
        self.n_vectors = database.shape[0]
        self.dimension = database.shape[1]
        
        print(f"[EncryptedNN] Database: {self.n_vectors} vectors, {self.dimension}D")
    
    def search(
        self, 
        encrypted_query: 'SimulatedCiphertext', 
        k: int = 5,
        decrypt_distances: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors of encrypted query.
        
        Args:
            encrypted_query: Encrypted query vector
            k: Number of neighbors to return
            decrypt_distances: If True, decrypt and return actual distances
            
        Returns:
            Tuple of (indices, distances) arrays
        """
        # Compute all distances (encrypted)
        encrypted_distances = self.ops.batch_distances_plain(
            encrypted_query, 
            self.database
        )
        
        # Decrypt distances for comparison
        # Note: In a real privacy-preserving scenario, this would be done
        # differently (secure comparison protocols)
        distances = np.array([
            self.engine.decrypt_vector(d)[0] 
            for d in encrypted_distances
        ])
        
        # Get top-k indices
        indices = np.argsort(distances)[:k]
        
        if decrypt_distances:
            return indices, distances[indices]
        else:
            return indices, None
    
    def search_encrypted_db(
        self,
        encrypted_query: 'SimulatedCiphertext',
        encrypted_database: List['SimulatedCiphertext'],
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search when both query AND database are encrypted.
        
        This is the fully encrypted scenario.
        
        Args:
            encrypted_query: Encrypted query vector
            encrypted_database: List of encrypted database vectors
            k: Number of neighbors
            
        Returns:
            Tuple of (indices, distances)
        """
        encrypted_distances = self.ops.batch_distances(
            encrypted_query,
            encrypted_database
        )
        
        # Decrypt for comparison
        distances = np.array([
            self.engine.decrypt_vector(d)[0]
            for d in encrypted_distances
        ])
        
        indices = np.argsort(distances)[:k]
        return indices, distances[indices]
