"""Operaciones de modulación y demodulación lineales."""

from __future__ import annotations

from typing import List

from matrix_ops import mat_vec_mul


Vector = List[float]
Matrix = List[List[float]]


def modulate(m: Matrix, x: Vector) -> Vector:
    """Modelo lineal de modulación: s = M*x."""
    return mat_vec_mul(m, x)


def demodulate(a: Matrix, r: Vector) -> Vector:
    """Modelo lineal de demodulación/estimación: y = A*r."""
    return mat_vec_mul(a, r)


def hard_decision(signal: Vector, threshold: float = 0.0) -> List[int]:
    """Decisión binaria por umbral para recuperar bits estimados."""
    return [1 if sample >= threshold else 0 for sample in signal]
