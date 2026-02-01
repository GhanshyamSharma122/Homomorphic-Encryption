"""
Performance Metrics and Timing Utilities
"""

import time
from typing import Dict, List, Optional, Callable
from contextlib import contextmanager
import numpy as np


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start = None
        self.end = None
        self.elapsed = None
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
    
    def __str__(self):
        if self.elapsed is not None:
            return f"{self.name}: {self.elapsed:.4f}s"
        return f"{self.name}: not measured"


@contextmanager
def timer(name: str = ""):
    """Simple timer context manager."""
    t = Timer(name)
    try:
        yield t
        t.__enter__()
    finally:
        t.__exit__()
        if name:
            print(t)


class PerformanceMetrics:
    """
    Track and aggregate performance metrics across multiple runs.
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def record(self, name: str, value: float):
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if name not in self.metrics:
            return {}
        
        values = self.metrics[name]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {name: self.get_stats(name) for name in self.metrics}
    
    def print_summary(self):
        """Print summary of all metrics."""
        print("\n" + "=" * 50)
        print("PERFORMANCE METRICS SUMMARY")
        print("=" * 50)
        
        for name, values in self.metrics.items():
            stats = self.get_stats(name)
            print(f"\n{name}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")
            print(f"  N:    {stats['count']}")
        
        print("\n" + "=" * 50)
    
    def clear(self):
        """Clear all recorded metrics."""
        self.metrics.clear()


def compute_recall(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    k: Optional[int] = None
) -> float:
    """
    Compute recall@k between predicted and ground truth indices.
    
    Args:
        predicted: Predicted neighbor indices
        ground_truth: Ground truth neighbor indices
        k: Consider only top-k (default: all)
        
    Returns:
        Recall score (0-1)
    """
    if k is not None:
        predicted = predicted[:k]
        ground_truth = ground_truth[:k]
    
    return len(set(predicted) & set(ground_truth)) / len(ground_truth)


def compute_precision(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    k: Optional[int] = None
) -> float:
    """
    Compute precision@k.
    
    Args:
        predicted: Predicted neighbor indices
        ground_truth: Ground truth neighbor indices
        k: Consider only top-k
        
    Returns:
        Precision score (0-1)
    """
    if k is not None:
        predicted = predicted[:k]
    
    return len(set(predicted) & set(ground_truth)) / len(predicted)


def benchmark_function(
    func: Callable,
    n_runs: int = 10,
    warmup: int = 2,
    **kwargs
) -> Dict[str, float]:
    """
    Benchmark a function with multiple runs.
    
    Args:
        func: Function to benchmark
        n_runs: Number of timed runs
        warmup: Number of warmup runs (not timed)
        **kwargs: Arguments to pass to function
        
    Returns:
        Timing statistics
    """
    # Warmup
    for _ in range(warmup):
        func(**kwargs)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(**kwargs)
        times.append(time.perf_counter() - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }
