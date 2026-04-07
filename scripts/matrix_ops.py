"""Operaciones básicas de álgebra lineal para vectores y matrices."""

from __future__ import annotations

from typing import List

Vector = List[float]
Matrix = List[List[float]]


def _ensure_non_empty_matrix(a: Matrix) -> None:
    if not a or not a[0]:
        raise ValueError("La matriz no puede estar vacía")


def _ensure_rectangular_matrix(a: Matrix) -> None:
    _ensure_non_empty_matrix(a)
    cols = len(a[0])
    for row in a:
        if len(row) != cols:
            raise ValueError("La matriz debe ser rectangular")


def _ensure_same_length(x: List[float], y: List[float]) -> None:
    if len(x) != len(y):
        raise ValueError("Los vectores deben tener la misma dimensión")


def dot_product(x: Vector, y: Vector) -> float:
    """Producto escalar estándar en R^n."""
    _ensure_same_length(x, y)
    return sum(xi * yi for xi, yi in zip(x, y))


def dot_product_mod2(x: List[int], y: List[int]) -> int:
    """Producto escalar sobre F2."""
    _ensure_same_length(x, y)
    return sum((xi & 1) * (yi & 1) for xi, yi in zip(x, y)) % 2


def add_vectors(x: Vector, y: Vector) -> Vector:
    """Suma de vectores en R^n."""
    _ensure_same_length(x, y)
    return [xi + yi for xi, yi in zip(x, y)]


def add_vectors_mod2(x: List[int], y: List[int]) -> List[int]:
    """Suma de vectores en F2 (XOR componente a componente)."""
    _ensure_same_length(x, y)
    return [(xi ^ yi) for xi, yi in zip(x, y)]


def mat_vec_mul(a: Matrix, x: Vector) -> Vector:
    """Multiplicación matriz-vector en R^n."""
    _ensure_rectangular_matrix(a)
    if len(a[0]) != len(x):
        raise ValueError("Dimensiones incompatibles para A*x")
    return [dot_product(row, x) for row in a]


def mat_vec_mul_mod2(a: List[List[int]], x: List[int]) -> List[int]:
    """Multiplicación matriz-vector sobre F2."""
    _ensure_rectangular_matrix(a)
    if len(a[0]) != len(x):
        raise ValueError("Dimensiones incompatibles para A*x en F2")
    return [dot_product_mod2(row, x) for row in a]


def transpose(a: Matrix) -> Matrix:
    """Transpuesta de una matriz."""
    _ensure_rectangular_matrix(a)
    return [list(col) for col in zip(*a)]


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    """Multiplicación de matrices en R^n."""
    _ensure_rectangular_matrix(a)
    _ensure_rectangular_matrix(b)
    if len(a[0]) != len(b):
        raise ValueError("Dimensiones incompatibles para A*B")

    bt = transpose(b)
    return [[dot_product(row, col) for col in bt] for row in a]
