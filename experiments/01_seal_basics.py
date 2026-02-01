"""
Experiment 1: Basic Homomorphic Encryption Examples

Learn the fundamentals of homomorphic encryption.
Run this first to verify your installation works!

Uses simulation mode for Python 3.14 compatibility.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.core.he_engine import HEEngine


def demo_basic_encryption():
    """
    Demonstrate basic encryption and decryption.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Encryption/Decryption")
    print("=" * 60)
    
    # Create HE engine
    engine = HEEngine()
    
    # Original data
    plain_vector = [1.5, 2.3, 3.7, 4.1, 5.9]
    print(f"\nOriginal vector: {plain_vector}")
    
    # Encrypt
    encrypted_vector = engine.encrypt_vector(plain_vector)
    print(f"Encrypted! (simulated ciphertext)")
    
    # Decrypt
    decrypted_vector = engine.decrypt_vector(encrypted_vector)
    print(f"Decrypted vector: {[round(x, 4) for x in decrypted_vector]}")
    
    # Check error
    error = np.abs(np.array(plain_vector) - decrypted_vector[:len(plain_vector)])
    print(f"Decryption error (max): {error.max():.2e}")


def demo_encrypted_arithmetic():
    """
    Demonstrate arithmetic operations on encrypted data.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Encrypted Arithmetic")
    print("=" * 60)
    
    engine = HEEngine()
    
    # Two vectors
    vec1 = [1.0, 2.0, 3.0, 4.0]
    vec2 = [4.0, 3.0, 2.0, 1.0]
    
    print(f"\nVector 1: {vec1}")
    print(f"Vector 2: {vec2}")
    
    # Encrypt both
    enc1 = engine.encrypt_vector(vec1)
    enc2 = engine.encrypt_vector(vec2)
    
    # Addition (encrypted + encrypted)
    enc_sum = enc1 + enc2
    decrypted_sum = engine.decrypt_vector(enc_sum)[:4]
    expected_sum = np.array(vec1) + np.array(vec2)
    print(f"\nAddition (enc + enc):")
    print(f"  Result:   {[round(x, 4) for x in decrypted_sum]}")
    print(f"  Expected: {expected_sum.tolist()}")
    
    # Multiplication (encrypted * encrypted)
    enc_prod = enc1 * enc2
    decrypted_prod = engine.decrypt_vector(enc_prod)[:4]
    expected_prod = np.array(vec1) * np.array(vec2)
    print(f"\nMultiplication (enc * enc):")
    print(f"  Result:   {[round(x, 4) for x in decrypted_prod]}")
    print(f"  Expected: {expected_prod.tolist()}")
    
    # Add/multiply with plaintext (more efficient!)
    plain_scalar = 2.5
    enc_scaled = enc1 * plain_scalar
    decrypted_scaled = engine.decrypt_vector(enc_scaled)[:4]
    expected_scaled = np.array(vec1) * plain_scalar
    print(f"\nScalar multiply (enc * 2.5):")
    print(f"  Result:   {[round(x, 4) for x in decrypted_scaled]}")
    print(f"  Expected: {expected_scaled.tolist()}")


def demo_dot_product():
    """
    Demonstrate encrypted dot product - key for vector similarity!
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Encrypted Dot Product")
    print("=" * 60)
    
    engine = HEEngine()
    
    vec1 = [1.0, 2.0, 3.0, 4.0]
    vec2 = [4.0, 3.0, 2.0, 1.0]
    
    print(f"\nVector 1: {vec1}")
    print(f"Vector 2: {vec2}")
    
    # Expected dot product
    expected = np.dot(vec1, vec2)
    print(f"Expected dot product: {expected}")
    
    # Encrypted dot product
    enc1 = engine.encrypt_vector(vec1)
    enc2 = engine.encrypt_vector(vec2)
    
    # Element-wise multiply then sum
    enc_prod = enc1 * enc2
    enc_dot = enc_prod.sum()
    
    result = engine.decrypt_vector(enc_dot)[0]
    print(f"Encrypted dot product: {round(result, 4)}")
    print(f"Error: {abs(result - expected):.2e}")


def demo_euclidean_distance():
    """
    Demonstrate encrypted Euclidean distance calculation.
    """
    print("\n" + "=" * 60)
    print("DEMO 4: Encrypted Euclidean Distance")
    print("=" * 60)
    
    engine = HEEngine()
    
    vec1 = [1.0, 2.0, 3.0, 4.0]
    vec2 = [4.0, 3.0, 2.0, 1.0]
    
    print(f"\nVector 1: {vec1}")
    print(f"Vector 2: {vec2}")
    
    # Expected squared distance
    expected = np.sum((np.array(vec1) - np.array(vec2))**2)
    print(f"Expected squared distance: {expected}")
    
    # Encrypted distance
    enc1 = engine.encrypt_vector(vec1)
    enc2 = engine.encrypt_vector(vec2)
    
    # (enc1 - enc2)^2
    enc_diff = enc1 - enc2
    enc_diff_sq = enc_diff * enc_diff
    enc_dist = enc_diff_sq.sum()
    
    result = engine.decrypt_vector(enc_dist)[0]
    print(f"Encrypted squared distance: {round(result, 4)}")
    print(f"Error: {abs(result - expected):.2e}")


def demo_plaintext_operations():
    """
    Compare encrypted-plaintext vs encrypted-encrypted operations.
    """
    print("\n" + "=" * 60)
    print("DEMO 5: Plaintext vs Encrypted Operations")
    print("=" * 60)
    
    import time
    
    engine = HEEngine()
    
    dimension = 128
    vec1 = np.random.randn(dimension).tolist()
    vec2 = np.random.randn(dimension).tolist()
    
    enc1 = engine.encrypt_vector(vec1)
    enc2 = engine.encrypt_vector(vec2)
    
    # Time encrypted * encrypted
    n_runs = 100
    
    start = time.time()
    for _ in range(n_runs):
        result = enc1 * enc2
    enc_enc_time = (time.time() - start) / n_runs
    
    # Time encrypted * plaintext
    start = time.time()
    for _ in range(n_runs):
        result = enc1 * vec2
    enc_plain_time = (time.time() - start) / n_runs
    
    print(f"\nMultiply {dimension}D vectors ({n_runs} runs each):")
    print(f"  Encrypted × Encrypted: {enc_enc_time*1000:.3f} ms")
    print(f"  Encrypted × Plaintext: {enc_plain_time*1000:.3f} ms")
    
    if enc_enc_time > enc_plain_time:
        print(f"  Note: Plaintext operations are faster in simulation too!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HOMOMORPHIC ENCRYPTION FUNDAMENTALS")
    print("=" * 60)
    print("\nThis script demonstrates the basics of homomorphic encryption")
    print("using the CKKS-like scheme for approximate real number arithmetic.")
    
    demo_basic_encryption()
    demo_encrypted_arithmetic()
    demo_dot_product()
    demo_euclidean_distance()
    demo_plaintext_operations()
    
    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. CKKS allows arithmetic on encrypted real numbers")
    print("2. Results have small approximation error (acceptable)")
    print("3. Operations with plaintext are more efficient")
    print("4. Can compute dot product and distances without decryption!")
    print("\nNOTE: This is running in SIMULATION mode for Python 3.14")
    print("      For real encryption, use Python 3.10 with TenSEAL")
