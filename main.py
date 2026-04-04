"""Aplicación principal para modelización vectorial y codificación en comunicaciones digitales."""

from __future__ import annotations

from typing import List

from coding import (
    apply_binary_noise,
    correct_single_bit_error,
    encode_with_generator,
    is_valid_codeword,
    syndrome,
)
from crypto import xor_cipher
from distances import euclidean_distance, hamming_distance
from modulation import demodulate, hard_decision, modulate
from signal_model import add_signal_noise, bits_to_baseband_signal, linear_signal_transform
from vector_binary import bits_from_string, string_from_bits


# Código lineal (7,4) en forma sistemática: c = G*m
G_HAMMING_7_4 = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1],
]

# Matriz de control: H*c = 0 para palabras válidas
H_HAMMING_7_4 = [
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
]


def _read_bits(prompt: str) -> List[int]:
    return bits_from_string(input(prompt).strip())


def _read_flip_positions(prompt: str) -> List[int]:
    raw = input(prompt).strip()
    if not raw:
        return []
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def option_encode() -> None:
    m = _read_bits("Mensaje binario de 4 bits (ej: 1011): ")
    if len(m) != 4:
        raise ValueError("El código (7,4) requiere exactamente 4 bits de mensaje")
    c = encode_with_generator(G_HAMMING_7_4, m)
    print(f"m = {string_from_bits(m)}")
    print(f"c = Gm = {string_from_bits(c)}")


def option_cipher() -> None:
    bits = _read_bits("Palabra binaria a cifrar: ")
    key = _read_bits("Clave binaria: ")
    encrypted = xor_cipher(bits, key)
    print(f"Cifrado XOR: {string_from_bits(encrypted)}")


def option_decipher() -> None:
    bits = _read_bits("Palabra binaria cifrada: ")
    key = _read_bits("Clave binaria: ")
    decrypted = xor_cipher(bits, key)
    print(f"Descifrado XOR: {string_from_bits(decrypted)}")


def option_full_pipeline() -> None:
    print("Flujo: m -> c=Gm -> cifrado XOR -> ruido -> descifrado -> síndrome/corrección")
    m = _read_bits("Mensaje binario de 4 bits: ")
    key = _read_bits("Clave binaria para XOR: ")
    if len(m) != 4:
        raise ValueError("El código (7,4) requiere exactamente 4 bits de mensaje")

    c = encode_with_generator(G_HAMMING_7_4, m)
    encrypted = xor_cipher(c, key)

    flips = _read_flip_positions(
        "Posiciones de error en el canal sobre el cifrado (0-index, separadas por coma, vacio=sin error): "
    )
    received_encrypted = apply_binary_noise(encrypted, flips)
    received = xor_cipher(received_encrypted, key)

    s = syndrome(H_HAMMING_7_4, received)
    corrected = correct_single_bit_error(H_HAMMING_7_4, received)

    print(f"Mensaje original m:               {string_from_bits(m)}")
    print(f"Codificado c=Gm:                 {string_from_bits(c)}")
    print(f"Cifrado:                         {string_from_bits(encrypted)}")
    print(f"Recibido cifrado con ruido:      {string_from_bits(received_encrypted)}")
    print(f"Descifrado recibido:             {string_from_bits(received)}")
    print(f"Sindrome H*r:                    {string_from_bits(s)}")
    print(f"H*r == 0 ?                       {all(v == 0 for v in s)}")

    if corrected is None:
        print("No se pudo corregir automáticamente con corrección de 1 bit.")
    else:
        print(f"Corregido (si aplica):           {string_from_bits(corrected)}")
        print(f"Palabra valida tras correccion?  {is_valid_codeword(H_HAMMING_7_4, corrected)}")


def option_metrics_and_signals() -> None:
    a = _read_bits("Vector binario A: ")
    b = _read_bits("Vector binario B (misma longitud): ")
    hd = hamming_distance(a, b)
    print(f"Distancia de Hamming(A, B) = {hd}")

    sa = bits_to_baseband_signal(a)
    sb = bits_to_baseband_signal(b)
    ed = euclidean_distance(sa, sb)
    print(f"Distancia euclidea entre señales baseband = {ed:.4f}")

    # Ejemplo de transformaciones lineales en R^n para modulación/demodulación.
    m = [[1.0 if i == j else 0.0 for j in range(len(sa))] for i in range(len(sa))]
    a_demod = [[1.0 if i == j else 0.0 for j in range(len(sa))] for i in range(len(sa))]
    noise = [0.2 if i % 2 == 0 else -0.2 for i in range(len(sa))]

    s = modulate(m, sa)
    r = add_signal_noise(s, noise)
    y = demodulate(a_demod, r)
    hard_bits = hard_decision(y)

    print(f"s = Mx:                          {s}")
    print(f"r = s + n:                       {r}")
    print(f"y = A*r:                         {y}")
    print(f"Decision dura(y):                {string_from_bits(hard_bits)}")

    # Transformación lineal adicional del módulo de señal.
    transformed = linear_signal_transform(m, sa)
    print(f"Transformacion lineal extra A*x: {transformed}")


def main() -> None:
    options = {
        "1": option_encode,
        "2": option_cipher,
        "3": option_decipher,
        "4": option_full_pipeline,
        "5": option_metrics_and_signals,
    }

    while True:
        print("\n=== MENU PRINCIPAL ===")
        print("1) Codificar palabra binaria con matriz generadora (c = Gm)")
        print("2) Cifrar palabra binaria (XOR)")
        print("3) Descifrar palabra binaria (XOR)")
        print("4) Flujo completo: codificar + cifrar + ruido + descifrar + syndrome")
        print("5) Distancias y modelo de señales (Hamming, Euclidea, s=Mx, r=s+n, y=Ar)")
        print("6) Salir")

        choice = input("Selecciona una opcion: ").strip()
        if choice == "6":
            print("Fin del programa.")
            break

        action = options.get(choice)
        if action is None:
            print("Opcion invalida.")
            continue

        try:
            action()
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
