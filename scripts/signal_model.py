"""Modelización de señales digitales como vectores en R^n."""

from __future__ import annotations

from typing import List

from matrix_ops import add_vectors, mat_vec_mul


SignalVector = List[float]
BinaryVector = List[int]


def bits_to_baseband_signal(bits: BinaryVector, zero_level: float = -1.0, one_level: float = 1.0) -> SignalVector:
    """Mapea bits a niveles de señal (ej. BPSK simplificado)."""
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("El vector de entrada debe ser binario")
    return [one_level if bit == 1 else zero_level for bit in bits]


def linear_signal_transform(a: List[List[float]], x: SignalVector) -> SignalVector:
    """Transformación lineal de señal: y = A*x."""
    return mat_vec_mul(a, x)


def add_signal_noise(signal: SignalVector, noise: SignalVector) -> SignalVector:
    """Modelo de canal aditivo: r = s + n."""
    return add_vectors(signal, noise)
