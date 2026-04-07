"""Representación de información digital como vectores en F2^n."""

from __future__ import annotations

from typing import List

from matrix_ops import add_vectors_mod2, mat_vec_mul_mod2


BinaryVector = List[int]


def bits_from_string(binary_word: str) -> BinaryVector:
    """Convierte una cadena binaria en un vector de bits."""
    if not binary_word:
        raise ValueError("La palabra binaria no puede estar vacía")
    if any(ch not in {"0", "1"} for ch in binary_word):
        raise ValueError("La palabra debe contener solo 0 y 1")
    return [int(ch) for ch in binary_word]


def string_from_bits(bits: BinaryVector) -> str:
    """Convierte un vector de bits en cadena binaria."""
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("El vector contiene valores no binarios")
    return "".join(str(bit) for bit in bits)


def vector_sum_mod2(a: BinaryVector, b: BinaryVector) -> BinaryVector:
    """Suma vectorial en F2."""
    return add_vectors_mod2(a, b)


def linear_transform_mod2(a: List[List[int]], x: BinaryVector) -> BinaryVector:
    """Aplica una transformación lineal y = A*x sobre F2."""
    return mat_vec_mul_mod2(a, x)
