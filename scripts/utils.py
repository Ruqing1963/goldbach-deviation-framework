"""
Utility functions for Goldbach deviation analysis.
"""

import numpy as np
from functools import lru_cache


# Twin prime constant C_2
C2 = 0.6601618158468696

# Coupling constant for fine structure
KAPPA_LOCAL = 1/24


def sieve_of_eratosthenes(n):
    """
    Generate all primes up to n using Sieve of Eratosthenes.
    
    Parameters
    ----------
    n : int
        Upper limit for prime generation.
    
    Returns
    -------
    numpy.ndarray
        Array of primes up to n.
    """
    if n < 2:
        return np.array([], dtype=int)
    
    is_prime = np.ones(n + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    return np.nonzero(is_prime)[0]


@lru_cache(maxsize=1)
def get_prime_set(n):
    """Get set of primes for fast lookup."""
    primes = sieve_of_eratosthenes(n)
    return set(primes)


def is_prime(n, prime_set=None):
    """Check if n is prime."""
    if prime_set is not None:
        return n in prime_set
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def goldbach_count(N, primes=None, prime_set=None):
    """
    Count the number of ways to write N as sum of two primes.
    
    Parameters
    ----------
    N : int
        Even integer to decompose.
    primes : array-like, optional
        Pre-computed array of primes.
    prime_set : set, optional
        Pre-computed set of primes for fast lookup.
    
    Returns
    -------
    int
        Number of Goldbach representations G(N).
    """
    if N < 4 or N % 2 != 0:
        return 0
    
    if primes is None:
        primes = sieve_of_eratosthenes(N)
    if prime_set is None:
        prime_set = set(primes)
    
    count = 0
    for p in primes:
        if p > N // 2:
            break
        if (N - p) in prime_set:
            count += 1
    
    return count


def singular_series(N, num_primes=100):
    """
    Compute the singular series S(N) for Goldbach's problem.
    
    S(N) = ∏_{p|N, p>2} (p-1)/(p-2) * ∏_{p∤N, p>2} (1 - 1/(p-1)²)
    
    Parameters
    ----------
    N : int
        Even integer.
    num_primes : int
        Number of primes to use in product approximation.
    
    Returns
    -------
    float
        Approximate value of S(N).
    """
    primes = sieve_of_eratosthenes(num_primes * 10)[:num_primes]
    
    S = 1.0
    for p in primes[1:]:  # Skip p=2
        if N % p == 0:
            S *= (p - 1) / (p - 2)
        else:
            S *= 1 - 1 / (p - 1)**2
    
    # Normalize by C_2 inverse to get S(N)/C_2 factor
    return S / C2 * C2  # Just return S


def hardy_littlewood(N, S_N=None):
    """
    Compute Hardy-Littlewood prediction for G(N).
    
    HL(N) = 2 * C_2 * S(N) * N / (ln N)²
    
    Parameters
    ----------
    N : int
        Even integer.
    S_N : float, optional
        Pre-computed singular series value.
    
    Returns
    -------
    float
        Hardy-Littlewood prediction.
    """
    if N < 4:
        return 0.0
    
    if S_N is None:
        S_N = singular_series(N)
    
    ln_N = np.log(N)
    return 2 * C2 * S_N * N / ln_N**2


def relative_deviation(G_N, HL_N):
    """
    Compute relative deviation δ(N) = (G(N) - HL(N)) / HL(N).
    """
    if HL_N == 0:
        return 0.0
    return (G_N - HL_N) / HL_N


def dirichlet_character_mod3(n):
    """Quadratic character mod 3 (Legendre symbol)."""
    n = n % 3
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:  # n == 2
        return -1


def dirichlet_character_mod5(n):
    """Quadratic character mod 5 (Legendre symbol)."""
    n = n % 5
    if n == 0:
        return 0
    elif n in [1, 4]:
        return 1
    else:  # n in [2, 3]
        return -1


def L_value_chi3():
    """L(1, χ_3) ≈ 0.6046."""
    return 0.6046


def L_value_chi5():
    """L(1, χ_5) for quadratic character ≈ 0.4304."""
    return 0.4304


def L_value_chi7():
    """L(1, χ_7) for quadratic character ≈ 1.1874."""
    return 1.1874


def theoretical_coefficient(p, L_value):
    """
    Compute theoretical coefficient c_p = (1/24) * L(1, χ_p) / (p - 2).
    
    Parameters
    ----------
    p : int
        Prime modulus.
    L_value : float
        Value of L(1, χ_p).
    
    Returns
    -------
    float
        Theoretical coefficient.
    """
    return KAPPA_LOCAL * L_value / (p - 2)


if __name__ == "__main__":
    # Test basic functionality
    print("Twin prime constant C_2 =", C2)
    print("Coupling constant 1/24 =", KAPPA_LOCAL)
    
    # Test prime generation
    primes = sieve_of_eratosthenes(100)
    print(f"\nPrimes up to 100: {primes}")
    
    # Test Goldbach count
    N = 100
    G = goldbach_count(N)
    HL = hardy_littlewood(N)
    delta = relative_deviation(G, HL)
    print(f"\nG({N}) = {G}")
    print(f"HL({N}) = {HL:.2f}")
    print(f"δ({N}) = {delta:.4f}")
    
    # Test theoretical coefficients
    print("\nTheoretical coefficients c_p = (1/24) * L(1,χ_p) / (p-2):")
    print(f"  c_3 = {theoretical_coefficient(3, L_value_chi3()):.4f}")
    print(f"  c_5 = {theoretical_coefficient(5, L_value_chi5()):.4f}")
    print(f"  c_7 = {theoretical_coefficient(7, L_value_chi7()):.4f}")
