"""Cifrado y descifrado binario por XOR repetido."""

from __future__ import annotations

from typing import List


BinaryVector = List[int]


def _validate_bits(bits: BinaryVector, field_name: str) -> None:
    if not bits:
        raise ValueError(f"{field_name} no puede estar vacío")
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError(f"{field_name} debe contener solo bits")


def _expand_key(key: BinaryVector, length: int) -> BinaryVector:
    if len(key) > length:
        return key[:length]
    repeats = (length + len(key) - 1) // len(key)
    return (key * repeats)[:length]


def xor_cipher(bits: BinaryVector, key: BinaryVector) -> BinaryVector:
    """Cifra/descifra usando XOR con clave periódica."""
    _validate_bits(bits, "bits")
    _validate_bits(key, "key")
    long_key = _expand_key(key, len(bits))
    return [b ^ k for b, k in zip(bits, long_key)]
