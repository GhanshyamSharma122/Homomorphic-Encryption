"""
Naive Encrypted Search

Brute-force approach: compute distance to every vector in database.
This serves as the baseline for comparison with optimized approaches.
"""

import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING
from tqdm import tqdm
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.he_engine import HEEngine
from src.core.vector_ops import EncryptedVectorOps

if TYPE_CHECKING:
    from src.core.he_engine import SimulatedCiphertext


class NaiveEncryptedSearch:
    """
    Brute-force encrypted nearest neighbor search.
    
    Computes distance between query and every database vector.
    Time complexity: O(N * D) where N = database size, D = dimension
    """
    
    def __init__(
        self, 
        engine: HEEngine,
        database: Optional[np.ndarray] = None,
        encrypt_database: bool = False
    ):
        """
        Initialize search index.
        
        Args:
            engine: HE Engine instance
            database: Optional database to index (N x D array)
            encrypt_database: If True, encrypt all database vectors
        """
        self.engine = engine
        self.ops = EncryptedVectorOps(engine)
        self.database = database
        self.encrypted_database = None
        self.encrypt_database = encrypt_database
        
        if database is not None:
            self._index_database(database, encrypt_database)
    
    def _index_database(self, database: np.ndarray, encrypt: bool):
        """Index the database."""
        self.database = database
        self.n_vectors = database.shape[0]
        self.dimension = database.shape[1]
        
        print(f"[NaiveSearch] Indexing {self.n_vectors} vectors ({self.dimension}D)")
        
        if encrypt:
            print(f"[NaiveSearch] Encrypting database (this may take a while)...")
            self.encrypted_database = []
            for i, vec in enumerate(tqdm(database, desc="Encrypting")):
                self.encrypted_database.append(self.engine.encrypt_vector(vec))
            print(f"[NaiveSearch] Database encrypted")
    
    def add_vectors(self, vectors: np.ndarray, encrypt: bool = None):
        """
        Add vectors to the database.
        
        Args:
            vectors: Vectors to add (M x D array)
            encrypt: Override encryption setting
        """
        if encrypt is None:
            encrypt = self.encrypt_database
            
        if self.database is None:
            self._index_database(vectors, encrypt)
        else:
            self.database = np.vstack([self.database, vectors])
            self.n_vectors = self.database.shape[0]
            
            if encrypt:
                for vec in vectors:
                    self.encrypted_database.append(self.engine.encrypt_vector(vec))
    
    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        encrypt_query: bool = True,
        return_distances: bool = True,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector (1D array)
            k: Number of neighbors
            encrypt_query: Whether to encrypt the query
            return_distances: Whether to return distance values
            verbose: Print timing info
            
        Returns:
            Tuple of (indices, distances, timing_info)
        """
        timing = {}
        
        # Encrypt query
        t0 = time.time()
        if encrypt_query:
            enc_query = self.engine.encrypt_vector(query)
        timing['query_encryption'] = time.time() - t0
        
        # Compute distances
        t0 = time.time()
        if self.encrypted_database is not None:
            # Both encrypted
            enc_distances = self.ops.batch_distances(enc_query, self.encrypted_database)
        else:
            # Query encrypted, database plaintext
            enc_distances = self.ops.batch_distances_plain(enc_query, self.database)
        timing['distance_computation'] = time.time() - t0
        
        # Decrypt distances
        t0 = time.time()
        distances = np.array([
            self.engine.decrypt_vector(d)[0] for d in enc_distances
        ])
        timing['decryption'] = time.time() - t0
        
        # Sort and get top-k
        t0 = time.time()
        indices = np.argsort(distances)[:k]
        timing['sorting'] = time.time() - t0
        
        timing['total'] = sum(timing.values())
        
        if verbose:
            print(f"\n[NaiveSearch] Timing breakdown:")
            for key, val in timing.items():
                print(f"  {key}: {val:.4f}s")
        
        if return_distances:
            return indices, distances[indices], timing
        return indices, None, timing
    
    def benchmark(
        self,
        n_queries: int = 10,
        k: int = 5
    ) -> dict:
        """
        Run benchmark with random queries.
        
        Args:
            n_queries: Number of queries to run
            k: Number of neighbors per query
            
        Returns:
            Benchmark statistics
        """
        print(f"\n[NaiveSearch] Running benchmark: {n_queries} queries, k={k}")
        
        timings = []
        for i in tqdm(range(n_queries), desc="Benchmarking"):
            query = np.random.randn(self.dimension)
            _, _, timing = self.search(query, k=k)
            timings.append(timing)
        
        # Aggregate
        stats = {}
        for key in timings[0].keys():
            values = [t[key] for t in timings]
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        print(f"\n[NaiveSearch] Benchmark results (mean ± std):")
        for key, val in stats.items():
            print(f"  {key}: {val['mean']:.4f} ± {val['std']:.4f}s")
        
        return stats


def compare_with_plaintext(
    database: np.ndarray,
    queries: np.ndarray,
    k: int = 5
) -> dict:
    """
    Compare encrypted search accuracy with plaintext scipy/numpy.
    
    Args:
        database: Database vectors
        queries: Query vectors
        k: Number of neighbors
        
    Returns:
        Comparison results
    """
    from scipy.spatial.distance import cdist
    
    print("\n[Comparison] Encrypted vs Plaintext Search")
    
    # Plaintext search using scipy
    print("[Comparison] Running plaintext search...")
    t0 = time.time()
    distances_matrix = cdist(queries, database, 'sqeuclidean')
    plain_indices = np.argsort(distances_matrix, axis=1)[:, :k]
    plain_time = time.time() - t0
    print(f"[Comparison] Plaintext search time: {plain_time:.4f}s")
    
    # Encrypted search
    print("[Comparison] Running encrypted search...")
    engine = HEEngine()
    searcher = NaiveEncryptedSearch(engine, database)
    
    encrypted_indices = []
    encrypted_times = []
    for query in tqdm(queries, desc="Encrypted search"):
        idx, _, timing = searcher.search(query, k=k)
        encrypted_indices.append(idx)
        encrypted_times.append(timing['total'])
    
    encrypted_indices = np.array(encrypted_indices)
    
    # Compare results
    matches = (plain_indices == encrypted_indices).mean()
    
    results = {
        'plain_time': plain_time,
        'encrypted_time_mean': np.mean(encrypted_times),
        'encrypted_time_std': np.std(encrypted_times),
        'accuracy': matches,
        'slowdown_factor': np.mean(encrypted_times) / (plain_time / len(queries))
    }
    
    print(f"\n[Comparison] Results:")
    print(f"  Accuracy (index match): {matches:.2%}")
    print(f"  Plaintext time per query: {plain_time/len(queries)*1000:.2f}ms")
    print(f"  Encrypted time per query: {np.mean(encrypted_times)*1000:.2f}ms")
    print(f"  Slowdown factor: {results['slowdown_factor']:.1f}x")
    
    return results
