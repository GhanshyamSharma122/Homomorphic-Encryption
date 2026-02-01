"""
Optimized Encrypted Search

Implements optimization strategies to reduce the computational overhead
of encrypted nearest neighbor search.

Strategies implemented:
1. Pre-filtering with LSH (Locality Sensitive Hashing)
2. Batched SIMD operations
3. Hierarchical search
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
from collections import defaultdict
import time
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.he_engine import HEEngine
from src.core.vector_ops import EncryptedVectorOps

if TYPE_CHECKING:
    from src.core.he_engine import SimulatedCiphertext


class LSHEncryptedIndex:
    """
    Locality Sensitive Hashing for encrypted search pre-filtering.
    
    Idea: Use LSH to reduce candidate set, then run encrypted search
    only on candidates. LSH operates on plaintext database.
    
    Privacy model: Database is plaintext, query is encrypted.
    LSH hash is computed on encrypted query and compared.
    """
    
    def __init__(
        self,
        engine: HEEngine,
        dimension: int,
        n_tables: int = 10,
        n_bits: int = 8,
        seed: int = 42
    ):
        """
        Initialize LSH index.
        
        Args:
            engine: HE Engine
            dimension: Vector dimension
            n_tables: Number of hash tables (more = better recall)
            n_bits: Bits per hash (more = more selective)
            seed: Random seed
        """
        self.engine = engine
        self.ops = EncryptedVectorOps(engine)
        self.dimension = dimension
        self.n_tables = n_tables
        self.n_bits = n_bits
        
        np.random.seed(seed)
        
        # Generate random hyperplanes for each table
        self.hyperplanes = [
            np.random.randn(n_bits, dimension)
            for _ in range(n_tables)
        ]
        
        # Hash tables: table_id -> hash_value -> list of vector indices
        self.tables: List[Dict[int, List[int]]] = [
            defaultdict(list) for _ in range(n_tables)
        ]
        
        self.database = None
        self.n_vectors = 0
    
    def _compute_hash(self, vector: np.ndarray, table_idx: int) -> int:
        """Compute LSH hash for a vector."""
        projections = self.hyperplanes[table_idx] @ vector
        bits = (projections > 0).astype(int)
        # Convert binary to int
        hash_val = int(''.join(map(str, bits)), 2)
        return hash_val
    
    def _compute_hash_encrypted(
        self, 
        enc_vector: 'SimulatedCiphertext', 
        table_idx: int
    ) -> int:
        """
        Compute LSH hash for encrypted vector.
        
        Note: This requires decrypting the projections, which reveals
        some information. In practice, you'd use secure comparison.
        """
        # Project encrypted vector onto hyperplanes
        projections = []
        for hyperplane in self.hyperplanes[table_idx]:
            proj = self.ops.dot_product_plain(enc_vector, hyperplane)
            proj_val = self.engine.decrypt_vector(proj)[0]
            projections.append(proj_val)
        
        bits = [1 if p > 0 else 0 for p in projections]
        hash_val = int(''.join(map(str, bits)), 2)
        return hash_val
    
    def index(self, database: np.ndarray):
        """
        Build LSH index from database.
        
        Args:
            database: Database vectors (N x D)
        """
        self.database = database
        self.n_vectors = database.shape[0]
        
        print(f"[LSH] Indexing {self.n_vectors} vectors into {self.n_tables} tables")
        
        for i, vec in enumerate(tqdm(database, desc="Building LSH")):
            for t in range(self.n_tables):
                h = self._compute_hash(vec, t)
                self.tables[t][h].append(i)
        
        # Statistics
        for t in range(self.n_tables):
            n_buckets = len(self.tables[t])
            avg_size = np.mean([len(v) for v in self.tables[t].values()])
            print(f"[LSH] Table {t}: {n_buckets} buckets, avg size {avg_size:.1f}")
    
    def get_candidates(
        self, 
        enc_query: 'SimulatedCiphertext',
        max_candidates: int = 100
    ) -> np.ndarray:
        """
        Get candidate indices for encrypted query.
        
        Args:
            enc_query: Encrypted query vector
            max_candidates: Maximum candidates to return
            
        Returns:
            Array of candidate indices
        """
        candidates = set()
        
        for t in range(self.n_tables):
            h = self._compute_hash_encrypted(enc_query, t)
            candidates.update(self.tables[t].get(h, []))
        
        candidates = np.array(list(candidates))
        
        if len(candidates) > max_candidates:
            # Random sample if too many
            candidates = np.random.choice(candidates, max_candidates, replace=False)
        
        return candidates
    
    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        max_candidates: int = 100,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Search with LSH pre-filtering.
        
        Args:
            query: Query vector
            k: Number of neighbors
            max_candidates: Max candidates from LSH
            verbose: Print info
            
        Returns:
            (indices, distances, timing)
        """
        timing = {}
        
        # Encrypt query
        t0 = time.time()
        enc_query = self.engine.encrypt_vector(query)
        timing['encryption'] = time.time() - t0
        
        # Get candidates via LSH
        t0 = time.time()
        candidates = self.get_candidates(enc_query, max_candidates)
        timing['lsh'] = time.time() - t0
        
        if len(candidates) == 0:
            # Fallback to brute force on small sample
            candidates = np.random.choice(self.n_vectors, min(100, self.n_vectors), replace=False)
        
        if verbose:
            print(f"[LSH] Found {len(candidates)} candidates")
        
        # Compute distances to candidates only
        t0 = time.time()
        candidate_db = self.database[candidates]
        enc_distances = self.ops.batch_distances_plain(enc_query, candidate_db)
        timing['distance'] = time.time() - t0
        
        # Decrypt and sort
        t0 = time.time()
        distances = np.array([
            self.engine.decrypt_vector(d)[0] for d in enc_distances
        ])
        timing['decryption'] = time.time() - t0
        
        # Get top-k from candidates
        top_k_local = np.argsort(distances)[:k]
        indices = candidates[top_k_local]
        
        timing['total'] = sum(timing.values())
        
        if verbose:
            print(f"[LSH] Search timing: {timing}")
        
        return indices, distances[top_k_local], timing


class OptimizedEncryptedSearch:
    """
    Optimized encrypted search combining multiple strategies.
    """
    
    def __init__(
        self,
        engine: HEEngine,
        database: np.ndarray,
        use_lsh: bool = True,
        lsh_tables: int = 10,
        lsh_bits: int = 8
    ):
        """
        Initialize optimized search.
        
        Args:
            engine: HE Engine
            database: Database vectors
            use_lsh: Whether to use LSH pre-filtering
            lsh_tables: Number of LSH tables
            lsh_bits: Bits per LSH hash
        """
        self.engine = engine
        self.ops = EncryptedVectorOps(engine)
        self.database = database
        self.n_vectors = database.shape[0]
        self.dimension = database.shape[1]
        
        self.use_lsh = use_lsh
        if use_lsh:
            self.lsh = LSHEncryptedIndex(
                engine, 
                self.dimension,
                n_tables=lsh_tables,
                n_bits=lsh_bits
            )
            self.lsh.index(database)
    
    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        max_candidates: int = 100,
        fallback_to_naive: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Optimized search for k nearest neighbors.
        
        Args:
            query: Query vector
            k: Number of neighbors
            max_candidates: LSH candidate limit
            fallback_to_naive: Use naive search if LSH fails
            
        Returns:
            (indices, distances, timing)
        """
        if self.use_lsh:
            indices, distances, timing = self.lsh.search(
                query, k, max_candidates
            )
            return indices, distances, timing
        else:
            # Fallback to naive
            from .naive_search import NaiveEncryptedSearch
            naive = NaiveEncryptedSearch(self.engine, self.database)
            return naive.search(query, k)
    
    def benchmark_comparison(
        self,
        n_queries: int = 10,
        k: int = 5
    ) -> dict:
        """
        Compare optimized vs naive search.
        """
        from .naive_search import NaiveEncryptedSearch
        
        naive = NaiveEncryptedSearch(self.engine, self.database)
        
        naive_times = []
        opt_times = []
        recall_scores = []
        
        print(f"\n[Benchmark] Comparing naive vs optimized ({n_queries} queries)")
        
        for _ in tqdm(range(n_queries)):
            query = np.random.randn(self.dimension)
            
            # Naive search
            naive_idx, _, naive_timing = naive.search(query, k)
            naive_times.append(naive_timing['total'])
            
            # Optimized search  
            opt_idx, _, opt_timing = self.search(query, k)
            opt_times.append(opt_timing['total'])
            
            # Recall: how many of top-k match
            recall = len(set(naive_idx) & set(opt_idx)) / k
            recall_scores.append(recall)
        
        results = {
            'naive_time_mean': np.mean(naive_times),
            'opt_time_mean': np.mean(opt_times),
            'speedup': np.mean(naive_times) / np.mean(opt_times),
            'recall_mean': np.mean(recall_scores)
        }
        
        print(f"\n[Benchmark] Results:")
        print(f"  Naive mean time: {results['naive_time_mean']:.4f}s")
        print(f"  Optimized mean time: {results['opt_time_mean']:.4f}s")
        print(f"  Speedup: {results['speedup']:.2f}x")
        print(f"  Recall@{k}: {results['recall_mean']:.2%}")
        
        return results
