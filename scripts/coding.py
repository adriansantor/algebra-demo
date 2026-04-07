"""Codificación lineal y control de paridad sobre F2."""

from __future__ import annotations

from typing import List, Optional

from matrix_ops import mat_vec_mul_mod2, transpose


BinaryVector = List[int]
BinaryMatrix = List[List[int]]


def encode_with_generator(g: BinaryMatrix, m: BinaryVector) -> BinaryVector:
    """Codifica un mensaje binario usando c = G*m."""
    return mat_vec_mul_mod2(g, m)


def syndrome(h: BinaryMatrix, r: BinaryVector) -> BinaryVector:
    """Calcula el síndrome s = H*r."""
    return mat_vec_mul_mod2(h, r)


def is_valid_codeword(h: BinaryMatrix, c: BinaryVector) -> bool:
    """Verifica si una palabra cumple H*c = 0."""
    return all(value == 0 for value in syndrome(h, c))


def correct_single_bit_error(h: BinaryMatrix, r: BinaryVector) -> Optional[BinaryVector]:
    """Corrige un solo error comparando el síndrome con columnas de H."""
    s = syndrome(h, r)
    if all(v == 0 for v in s):
        return r[:]

    h_columns = transpose(h)
    for idx, column in enumerate(h_columns):
        if [int(v) & 1 for v in column] == s:
            corrected = r[:]
            corrected[idx] ^= 1
            return corrected

    return None


def apply_binary_noise(bits: BinaryVector, flip_positions: List[int]) -> BinaryVector:
    """Invierte bits en posiciones específicas para simular errores de canal."""
    noisy = bits[:]
    for pos in flip_positions:
        if pos < 0 or pos >= len(noisy):
            raise ValueError(f"Posición de error inválida: {pos}")
        noisy[pos] ^= 1
    return noisy
