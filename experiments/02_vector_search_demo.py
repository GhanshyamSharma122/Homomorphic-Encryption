"""
Experiment 2: Encrypted Vector Search Demo

Full demonstration of encrypted nearest neighbor search.
Compares naive vs optimized approaches.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from src.core.he_engine import HEEngine
from src.core.vector_ops import EncryptedVectorOps
from src.indexing.naive_search import NaiveEncryptedSearch, compare_with_plaintext
from src.indexing.optimized_search import OptimizedEncryptedSearch
from src.utils.data_generator import generate_random_vectors, generate_clustered_vectors


def demo_encrypted_search():
    """
    Demonstrate encrypted nearest neighbor search.
    """
    print("\n" + "=" * 60)
    print("ENCRYPTED VECTOR SEARCH DEMO")
    print("=" * 60)
    
    # Parameters
    n_vectors = 100  # Start small for demo
    dimension = 64
    k = 5
    
    print(f"\nConfiguration:")
    print(f"  Database size: {n_vectors} vectors")
    print(f"  Dimension: {dimension}")
    print(f"  k (neighbors): {k}")
    
    # Generate data
    print("\n[1] Generating synthetic database...")
    database, labels = generate_clustered_vectors(
        n_vectors, dimension, n_clusters=10, seed=42
    )
    print(f"    Generated {n_vectors} vectors in {10} clusters")
    
    # Initialize HE engine
    print("\n[2] Initializing Homomorphic Encryption engine...")
    engine = HEEngine()
    
    # Create search index
    print("\n[3] Building search index...")
    searcher = NaiveEncryptedSearch(engine, database)
    
    # Generate query
    query = np.random.randn(dimension)
    print(f"\n[4] Searching for {k} nearest neighbors...")
    
    # Search
    indices, distances, timing = searcher.search(query, k=k, verbose=True)
    
    print(f"\n[5] Results:")
    print(f"    Nearest neighbor indices: {indices}")
    print(f"    Distances: {[round(d, 4) for d in distances]}")
    
    # Verify with plaintext search
    print("\n[6] Verifying with plaintext search...")
    from scipy.spatial.distance import cdist
    plain_distances = cdist([query], database, 'sqeuclidean')[0]
    plain_indices = np.argsort(plain_distances)[:k]
    
    print(f"    Plaintext nearest neighbors: {plain_indices}")
    print(f"    Match: {set(indices) == set(plain_indices)}")
    
    return timing


def benchmark_scalability():
    """
    Benchmark search time vs database size.
    """
    print("\n" + "=" * 60)
    print("SCALABILITY BENCHMARK")
    print("=" * 60)
    
    dimension = 32
    k = 5
    sizes = [10, 25, 50, 100]
    
    results = []
    engine = HEEngine()
    
    for n in sizes:
        print(f"\n[Database size: {n}]")
        database = generate_random_vectors(n, dimension, seed=42)
        searcher = NaiveEncryptedSearch(engine, database)
        
        # Run multiple queries
        times = []
        for _ in range(3):
            query = np.random.randn(dimension)
            _, _, timing = searcher.search(query, k=k)
            times.append(timing['total'])
        
        avg_time = np.mean(times)
        results.append({'n': n, 'time': avg_time})
        print(f"  Average search time: {avg_time:.4f}s")
    
    print("\n[Summary]")
    print("-" * 30)
    for r in results:
        print(f"  N={r['n']:4d}: {r['time']:.4f}s")
    
    return results


def demo_optimized_search():
    """
    Compare naive vs LSH-optimized search.
    """
    print("\n" + "=" * 60)
    print("OPTIMIZED (LSH) SEARCH DEMO")
    print("=" * 60)
    
    n_vectors = 200
    dimension = 32
    k = 5
    
    print(f"\nConfiguration:")
    print(f"  Database: {n_vectors} vectors, {dimension}D")
    print(f"  k: {k}")
    
    # Generate data
    database, _ = generate_clustered_vectors(n_vectors, dimension, n_clusters=10, seed=42)
    
    # Initialize
    engine = HEEngine()
    
    # Naive search baseline
    print("\n[1] Running naive search benchmark...")
    naive = NaiveEncryptedSearch(engine, database)
    
    naive_times = []
    for _ in tqdm(range(5), desc="Naive search"):
        query = np.random.randn(dimension)
        _, _, timing = naive.search(query, k=k)
        naive_times.append(timing['total'])
    
    print(f"    Naive avg time: {np.mean(naive_times):.4f}s")
    
    # Optimized search
    print("\n[2] Building optimized (LSH) index...")
    optimized = OptimizedEncryptedSearch(
        engine, database, 
        use_lsh=True, 
        lsh_tables=5, 
        lsh_bits=6
    )
    
    print("\n[3] Running optimized search benchmark...")
    opt_times = []
    recalls = []
    
    for _ in tqdm(range(5), desc="Optimized search"):
        query = np.random.randn(dimension)
        
        # Get ground truth
        naive_idx, _, _ = naive.search(query, k=k)
        
        # Optimized search
        opt_idx, _, timing = optimized.search(query, k=k, max_candidates=50)
        opt_times.append(timing['total'])
        
        # Recall
        recall = len(set(naive_idx) & set(opt_idx)) / k
        recalls.append(recall)
    
    print(f"\n[Results]")
    print(f"  Naive search:     {np.mean(naive_times):.4f}s")
    print(f"  Optimized search: {np.mean(opt_times):.4f}s")
    print(f"  Speedup:          {np.mean(naive_times)/np.mean(opt_times):.2f}x")
    print(f"  Recall@{k}:         {np.mean(recalls):.2%}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ENCRYPTED VECTOR SEARCH - FULL DEMO")
    print("=" * 60)
    print("\nThis script demonstrates privacy-preserving vector search")
    print("using homomorphic encryption (CKKS scheme).\n")
    
    # Run demos
    demo_encrypted_search()
    benchmark_scalability()
    demo_optimized_search()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
