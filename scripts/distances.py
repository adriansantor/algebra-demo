"""Métricas para comparar vectores binarios y reales."""

from __future__ import annotations

import math
from typing import List


Vector = List[float]
BinaryVector = List[int]


def hamming_distance(a: BinaryVector, b: BinaryVector) -> int:
    """Número de posiciones en las que dos vectores binarios difieren."""
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")
    if any(bit not in (0, 1) for bit in a + b):
        raise ValueError("La distancia de Hamming requiere vectores binarios")
    return sum(x != y for x, y in zip(a, b))


def euclidean_distance(a: Vector, b: Vector) -> float:
    """Distancia euclídea entre dos vectores en R^n."""
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma dimensión")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
