"""
Homomorphic Encryption Engine

This module provides a simulation/implementation of CKKS-like encryption
for vector operations. For production use with Microsoft SEAL, you'll need
Python 3.10 or earlier.

This implementation provides:
1. A working simulation for learning and development
2. The same API as the real TenSEAL implementation
3. Functional encrypted vector operations
"""

import numpy as np
from typing import List, Union, Optional, Tuple
import hashlib
import secrets


class SimulatedCiphertext:
    """
    Simulated encrypted vector for development.
    
    In a real implementation, this would be a TenSEAL CKKSVector.
    This simulation allows developing and testing the search algorithms
    without requiring the actual HE library.
    """
    
    def __init__(self, values: np.ndarray, context: 'HEEngine', noise_scale: float = 1e-6):
        """
        Simulate encryption by storing the values with added noise.
        
        Args:
            values: Original plaintext values
            context: HE context (for parameters)
            noise_scale: Simulated encryption noise
        """
        self._values = np.array(values, dtype=np.float64)
        self._noise = np.random.randn(len(values)) * noise_scale
        self._context = context
        self._operation_count = 0
        self._max_operations = 10  # Simulates noise budget
    
    def _check_budget(self):
        """Simulate noise budget exhaustion."""
        self._operation_count += 1
        if self._operation_count > self._max_operations:
            print("[WARNING] Noise budget exhausted - results may be inaccurate")
    
    def decrypt(self) -> np.ndarray:
        """Decrypt to get approximate original values."""
        # Add accumulated noise to simulate CKKS behavior
        accumulated_noise = self._noise * (1 + self._operation_count * 0.1)
        return self._values + accumulated_noise
    
    def __add__(self, other: Union['SimulatedCiphertext', np.ndarray, list, float]) -> 'SimulatedCiphertext':
        """Encrypted addition."""
        self._check_budget()
        
        if isinstance(other, SimulatedCiphertext):
            new_values = self._values + other._values
            result = SimulatedCiphertext(new_values, self._context)
            result._noise = self._noise + other._noise
        else:
            if isinstance(other, (list, float, int)):
                other = np.array(other) if isinstance(other, list) else other
            new_values = self._values + other
            result = SimulatedCiphertext(new_values, self._context)
            result._noise = self._noise.copy()
        
        result._operation_count = self._operation_count + 1
        return result
    
    def __sub__(self, other: Union['SimulatedCiphertext', np.ndarray, list, float]) -> 'SimulatedCiphertext':
        """Encrypted subtraction."""
        self._check_budget()
        
        if isinstance(other, SimulatedCiphertext):
            new_values = self._values - other._values
            result = SimulatedCiphertext(new_values, self._context)
            result._noise = self._noise + other._noise
        else:
            if isinstance(other, (list, float, int)):
                other = np.array(other) if isinstance(other, list) else other
            new_values = self._values - other
            result = SimulatedCiphertext(new_values, self._context)
            result._noise = self._noise.copy()
        
        result._operation_count = self._operation_count + 1
        return result
    
    def __mul__(self, other: Union['SimulatedCiphertext', np.ndarray, list, float]) -> 'SimulatedCiphertext':
        """Encrypted multiplication."""
        self._check_budget()
        
        if isinstance(other, SimulatedCiphertext):
            new_values = self._values * other._values
            result = SimulatedCiphertext(new_values, self._context)
            # Noise grows with multiplication
            result._noise = np.abs(self._values) * other._noise + np.abs(other._values) * self._noise
        else:
            if isinstance(other, (list, float, int)):
                other = np.array(other) if isinstance(other, list) else other
            new_values = self._values * other
            result = SimulatedCiphertext(new_values, self._context)
            result._noise = self._noise * np.abs(other) if isinstance(other, np.ndarray) else self._noise * abs(other)
        
        result._operation_count = self._operation_count + 2  # Multiplication costs more
        return result
    
    def __neg__(self) -> 'SimulatedCiphertext':
        """Negation."""
        result = SimulatedCiphertext(-self._values, self._context)
        result._noise = self._noise.copy()
        result._operation_count = self._operation_count
        return result
    
    def sum(self) -> 'SimulatedCiphertext':
        """Sum all elements (returns single-element ciphertext)."""
        self._check_budget()
        total = np.sum(self._values)
        result = SimulatedCiphertext(np.array([total]), self._context)
        result._noise = np.array([np.sum(np.abs(self._noise))])
        result._operation_count = self._operation_count + 1
        return result
    
    def serialize(self) -> bytes:
        """Serialize for storage/transmission."""
        return self._values.tobytes()
    
    def __len__(self) -> int:
        return len(self._values)


class HEEngine:
    """
    Homomorphic Encryption Engine.
    
    Uses CKKS-like scheme for approximate arithmetic on real numbers.
    This implementation provides a simulation for development on Python 3.14.
    For production with real encryption, use Python 3.10 with TenSEAL.
    """
    
    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] = None,
        global_scale: float = 2**40,
        simulation_mode: bool = True
    ):
        """
        Initialize the HE context.
        
        Args:
            poly_modulus_degree: Polynomial modulus degree (power of 2)
            coeff_mod_bit_sizes: Coefficient modulus bit sizes
            global_scale: Scale for encoding real numbers
            simulation_mode: If True, use simulation (default for Python 3.14+)
        """
        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]
        
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = global_scale
        self.max_vector_size = poly_modulus_degree // 2
        self.simulation_mode = simulation_mode
        
        # Generate keys (simulated)
        self._secret_key = secrets.token_bytes(32)
        self._public_key = hashlib.sha256(self._secret_key).digest()
        
        print(f"[HEEngine] Initialized in {'SIMULATION' if simulation_mode else 'REAL'} mode")
        print(f"[HEEngine] Poly modulus degree: {poly_modulus_degree}")
        print(f"[HEEngine] Max vector size: {self.max_vector_size}")
        
        if simulation_mode:
            print(f"[HEEngine] NOTE: Using simulation mode for Python 3.14 compatibility")
            print(f"[HEEngine]       For real encryption, use Python 3.10 with TenSEAL")
    
    def encrypt_vector(self, vector: Union[List[float], np.ndarray]) -> SimulatedCiphertext:
        """
        Encrypt a vector of real numbers.
        
        Args:
            vector: List or numpy array of floats
            
        Returns:
            Encrypted vector (SimulatedCiphertext or real ciphertext)
        """
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float64)
        
        if len(vector) > self.max_vector_size:
            raise ValueError(
                f"Vector size {len(vector)} exceeds maximum {self.max_vector_size}"
            )
        
        return SimulatedCiphertext(vector, self)
    
    def decrypt_vector(self, encrypted_vector: SimulatedCiphertext) -> np.ndarray:
        """
        Decrypt an encrypted vector.
        
        Args:
            encrypted_vector: Encrypted vector
            
        Returns:
            Decrypted numpy array
        """
        return encrypted_vector.decrypt()
    
    def encrypt_matrix(self, matrix: np.ndarray) -> List[SimulatedCiphertext]:
        """
        Encrypt a matrix row by row.
        
        Args:
            matrix: 2D numpy array
            
        Returns:
            List of encrypted row vectors
        """
        return [self.encrypt_vector(row) for row in matrix]
    
    def decrypt_matrix(self, encrypted_rows: List[SimulatedCiphertext]) -> np.ndarray:
        """
        Decrypt encrypted matrix rows.
        
        Args:
            encrypted_rows: List of encrypted row vectors
            
        Returns:
            Decrypted 2D numpy array
        """
        return np.array([self.decrypt_vector(row) for row in encrypted_rows])
    
    def get_public_key_bytes(self) -> bytes:
        """Get public key for sharing."""
        return self._public_key
    
    def is_simulation(self) -> bool:
        """Check if running in simulation mode."""
        return self.simulation_mode


class HEEngineLight:
    """
    Lightweight HE Engine for client-side (public key only).
    """
    
    def __init__(self, public_key_bytes: bytes, engine_params: dict = None):
        """
        Initialize with public key only.
        
        Args:
            public_key_bytes: Public key from HEEngine
            engine_params: Optional parameters
        """
        self._public_key = public_key_bytes
        self.max_vector_size = 4096
        
        if engine_params:
            self.max_vector_size = engine_params.get('max_vector_size', 4096)
    
    def encrypt_vector(self, vector: Union[List[float], np.ndarray]) -> SimulatedCiphertext:
        """Encrypt using public key only."""
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float64)
        
        # Create a minimal engine for encryption
        engine = HEEngine(simulation_mode=True)
        return SimulatedCiphertext(vector, engine)


# Attempt to use real TenSEAL if available
_REAL_HE_AVAILABLE = False

try:
    import tenseal as ts
    _REAL_HE_AVAILABLE = True
    
    class RealHEEngine:
        """Real HE Engine using TenSEAL (requires Python 3.10 or earlier)."""
        
        def __init__(self, poly_modulus_degree: int = 8192):
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            self.context.global_scale = 2**40
            self.context.generate_galois_keys()
            self.max_vector_size = poly_modulus_degree // 2
            print("[RealHEEngine] Initialized with TenSEAL")
        
        def encrypt_vector(self, vector):
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            return ts.ckks_vector(self.context, vector)
        
        def decrypt_vector(self, encrypted_vector):
            return np.array(encrypted_vector.decrypt())
    
except ImportError:
    pass


def get_he_engine(prefer_real: bool = True, **kwargs) -> HEEngine:
    """
    Get the best available HE engine.
    
    Args:
        prefer_real: If True and TenSEAL is available, use real HE
        **kwargs: Arguments to pass to engine constructor
        
    Returns:
        HEEngine instance
    """
    if prefer_real and _REAL_HE_AVAILABLE:
        return RealHEEngine(**kwargs)
    return HEEngine(simulation_mode=True, **kwargs)
