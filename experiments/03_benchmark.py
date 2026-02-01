"""
Experiment 3: Comprehensive Benchmarking

Compare encrypted vs plaintext search at various scales.
Generate publication-ready performance results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

from src.core.he_engine import HEEngine
from src.indexing.naive_search import NaiveEncryptedSearch
from src.indexing.optimized_search import OptimizedEncryptedSearch
from src.utils.data_generator import generate_clustered_vectors


def run_comprehensive_benchmark():
    """
    Run comprehensive benchmarks and save results.
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 60)
    
    # Configuration
    dimensions = [16, 32, 64]
    db_sizes = [50, 100, 200]
    n_queries = 5
    k = 5
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'experiments': []
    }
    
    engine = HEEngine()
    
    for dim in dimensions:
        for n in db_sizes:
            print(f"\n[Dim={dim}, N={n}]")
            
            # Generate data
            database, _ = generate_clustered_vectors(n, dim, seed=42)
            
            # Build indices
            naive = NaiveEncryptedSearch(engine, database)
            optimized = OptimizedEncryptedSearch(
                engine, database, use_lsh=True, lsh_tables=5, lsh_bits=6
            )
            
            # Run queries
            naive_times = []
            opt_times = []
            
            for _ in range(n_queries):
                query = np.random.randn(dim)
                
                _, _, t1 = naive.search(query, k=k)
                naive_times.append(t1['total'])
                
                _, _, t2 = optimized.search(query, k=k)
                opt_times.append(t2['total'])
            
            experiment = {
                'dimension': dim,
                'database_size': n,
                'k': k,
                'naive_time_mean': float(np.mean(naive_times)),
                'naive_time_std': float(np.std(naive_times)),
                'optimized_time_mean': float(np.mean(opt_times)),
                'optimized_time_std': float(np.std(opt_times)),
                'speedup': float(np.mean(naive_times) / np.mean(opt_times))
            }
            results['experiments'].append(experiment)
            
            print(f"  Naive: {experiment['naive_time_mean']:.4f}s")
            print(f"  Optimized: {experiment['optimized_time_mean']:.4f}s")
            print(f"  Speedup: {experiment['speedup']:.2f}x")
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results


def plot_results(results: dict):
    """
    Generate visualization of benchmark results.
    """
    experiments = results['experiments']
    
    # Group by dimension
    dims = sorted(set(e['dimension'] for e in experiments))
    
    fig, axes = plt.subplots(1, len(dims), figsize=(4*len(dims), 4))
    if len(dims) == 1:
        axes = [axes]
    
    for ax, dim in zip(axes, dims):
        dim_data = [e for e in experiments if e['dimension'] == dim]
        sizes = [e['database_size'] for e in dim_data]
        naive_times = [e['naive_time_mean'] for e in dim_data]
        opt_times = [e['optimized_time_mean'] for e in dim_data]
        
        x = np.arange(len(sizes))
        width = 0.35
        
        ax.bar(x - width/2, naive_times, width, label='Naive', color='#e74c3c')
        ax.bar(x + width/2, opt_times, width, label='Optimized', color='#2ecc71')
        
        ax.set_xlabel('Database Size')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Dimension = {dim}')
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.legend()
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), 'benchmark_plot.png')
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.show()


def compare_with_faiss():
    """
    Compare encrypted search with FAISS plaintext search.
    """
    print("\n" + "=" * 60)
    print("COMPARISON WITH FAISS (PLAINTEXT)")
    print("=" * 60)
    
    try:
        import faiss
    except ImportError:
        print("FAISS not installed. Run: pip install faiss-cpu")
        return
    
    n_vectors = 1000
    dimension = 64
    n_queries = 10
    k = 5
    
    print(f"\nConfiguration:")
    print(f"  Database: {n_vectors} vectors, {dimension}D")
    print(f"  Queries: {n_queries}")
    print(f"  k: {k}")
    
    # Generate data
    database, _ = generate_clustered_vectors(n_vectors, dimension, seed=42)
    queries = np.random.randn(n_queries, dimension)
    
    # FAISS baseline
    print("\n[FAISS Plaintext Search]")
    index = faiss.IndexFlatL2(dimension)
    index.add(database.astype('float32'))
    
    import time
    start = time.time()
    faiss_distances, faiss_indices = index.search(queries.astype('float32'), k)
    faiss_time = time.time() - start
    print(f"  Total time: {faiss_time:.4f}s")
    print(f"  Per query: {faiss_time/n_queries*1000:.2f}ms")
    
    # Encrypted search (small sample due to speed)
    print("\n[Encrypted Search (on subset)]")
    engine = HEEngine()
    
    # Use smaller database for demo
    small_n = 100
    small_db = database[:small_n]
    searcher = NaiveEncryptedSearch(engine, small_db)
    
    enc_times = []
    for query in tqdm(queries[:3], desc="Encrypted search"):  # Only 3 queries
        _, _, timing = searcher.search(query, k=k)
        enc_times.append(timing['total'])
    
    avg_enc_time = np.mean(enc_times)
    print(f"  Per query (N={small_n}): {avg_enc_time*1000:.2f}ms")
    
    # Estimate for full database
    estimated_full = avg_enc_time * (n_vectors / small_n)
    print(f"  Estimated for N={n_vectors}: {estimated_full*1000:.2f}ms")
    
    slowdown = estimated_full / (faiss_time / n_queries)
    print(f"\n[Comparison]")
    print(f"  Slowdown factor: ~{slowdown:.0f}x")
    print(f"  (This is expected! HE trades speed for privacy)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("COMPREHENSIVE BENCHMARKING")
    print("=" * 60)
    
    results = run_comprehensive_benchmark()
    
    try:
        plot_results(results)
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    compare_with_faiss()
    
    print("\n" + "=" * 60)
    print("BENCHMARKING COMPLETE!")
    print("=" * 60)
