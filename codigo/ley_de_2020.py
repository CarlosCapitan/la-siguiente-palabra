#!/usr/bin/env python3
"""
Capítulo 11 — la ley de 2020 y la receta de 2022, con las cuentas hechas (L24, D14 y D15).

Dos cosas que el capítulo contaba con palabras:

  1. «Doblar el tamaño reduce el error un cinco por ciento», y que eso es fiable a lo largo de
     muchos tamaños. Aquí está la escalera: cuánto queda del error del primero al doblar, al
     multiplicar por diez, y así hasta multiplicar por diez millones (el tramo del que habla el
     resumen del artículo: «more than seven orders of magnitude»).
  2. El modelo «bien alimentado» y el «mal alimentado» de 2022: cuántos números tiene cada uno y
     cuánto texto leyó.

Las cifras de partida son las de los artículos, copiadas aquí con su fuente; el programa solo
hace las cuentas. Nada se mide.

Fuentes (consultadas el 27 de septiembre de 2026):
  - Kaplan y otros, «Scaling Laws for Neural Language Models», arXiv 2001.08361. Los modelos que
    midieron: «ranging in size from 768 to 1.5 billion non-embedding parameters»; y: «These
    relations hold across eight orders of magnitude in Cmin, six orders of magnitude in N, and
    over two orders of magnitude in D» (Cmin es el cálculo; N, el tamaño). Ecuación 1.1:
    «L(N) = (Nc/N)^αN ; αN ∼ 0.076». Y en el texto: «doubling the number of parameters yields
    a loss that is smaller by a factor 2^−αN = 0.95». Resumen: «with some trends spanning more
    than seven orders of magnitude».
  - Hoffmann y otros, «Training Compute-Optimal Large Language Models», arXiv 2203.15556.
    Tabla 1: Gopher, 280 Billion, 300 Billion; Chinchilla, 70 Billion, 1.4 Trillion. Resumen:
    «uses the same compute budget as Gopher but with 70B parameters and 4× more data».

Uso:
    python ley_de_2020.py --selftest
    python ley_de_2020.py
"""

# ======================= CONSTANTES =======================

EXPONENTE = 0.076                    # αN de Kaplan y otros (2020), ecuación 1.1
FACTOR_DOBLAR_ARTICULO = 0.95        # lo que el propio artículo escribe para 2^−αN
TAMANOS = [1, 2, 4, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
MEDIDO_MENOR = 768                   # números del modelo más pequeño que midieron (sin las listas
MEDIDO_MAYOR = 1_500_000_000         # de cada trozo), y del más grande
ORDENES_CALCULO = 8                  # «eight orders of magnitude in Cmin»: por diez, ocho veces

MODELOS_2022 = [                     # (nombre, números, trozos de texto leídos), tabla 1
    ("280.000M", 280_000_000_000, 300_000_000_000),
    ("70.000M", 70_000_000_000, 1_400_000_000_000),
]

# ==========================================================

import argparse
import sys

from formato import ANCHO_CAJA_CITA, coma, comprobar_ancho, miles


def queda(veces, exponente=EXPONENTE):
    """Lo mal que lo hace un modelo `veces` más grande, si el primero hace 100."""
    return 100 * veces ** (-exponente)


def en_palabras(n):
    """280.000.000.000 -> «280.000 millones»; 1.400.000.000.000 -> «1,4 billones»."""
    if n >= 10 ** 12:
        return f"{coma(n / 10 ** 12, 1)} billones"
    return f"{miles(n // 10 ** 6)} millones"


def bloque_ley():
    lin = ["--- 1. LA LEY DE 2020: CUÁNTO BAJA EL ERROR AL CRECER ---", "",
           f"{'tamaño, en veces el primero':>28}{'lo mal que lo hace':>24}",
           f"{'':>28}{'(el primero: 100)':>24}"]
    for t in TAMANOS:
        lin.append(f"{miles(t):>28}{coma(queda(t), 0):>24}")
    tramo = MEDIDO_MAYOR // MEDIDO_MENOR
    lin += ["",
            f"doblar: de 100 a {coma(queda(2), 1)}; se le quita un {coma(100 - queda(2), 0)} %.",
            f"por diez: de 100 a {coma(queda(10), 1)}; se le quita un {coma(100 - queda(10), 0)} %.",
            "",
            "lo que midieron en tamaño:",
            f"de {miles(MEDIDO_MENOR)} números a {miles(MEDIDO_MAYOR // 10 ** 6)} millones, {miles(tramo)} veces",
            f"más; en ese tramo, de 100 a {coma(queda(tramo), 1)}.",
            f"en cálculo, el tramo medido es de {miles(10 ** ORDENES_CALCULO)} de veces."]
    return comprobar_ancho(lin, ANCHO_CAJA_CITA)


def bloque_2022():
    lin = ["--- 2. LA RECETA DE 2022: TAMAÑO Y TEXTO ---", "",
           f"{'modelo':<11}{'números':>18}{'trozos de texto':>18}{'trozos':>11}",
           f"{'':<11}{'':>18}{'leídos':>18}{'por número':>11}"]
    for n, num, texto in MODELOS_2022:
        lin.append(f"{n:<11}{en_palabras(num):>18}{en_palabras(texto):>18}{coma(texto / num, 1):>11}")
    (g, ng, tg), (c, nc, tc) = MODELOS_2022
    coste = (nc * tc) / (ng * tg)
    lin += ["",
            f"el de {c} tiene {coma(ng / nc, 0)} veces menos números que el de",
            f"{g}, y leyó {coma(tc / tg, 1)} veces más texto.",
            "lo que cuesta entrenar va como números por texto leído:",
            f"una cuarta parte por {coma(tc / tg, 1)} veces da {coma(coste, 2)}: más o menos lo",
            "mismo (el artículo dice que gastaron el mismo cálculo).",
            "«billones», en castellano: millones de millones.",
            "«M»: millones de números."]
    return comprobar_ancho(lin, ANCHO_CAJA_CITA)


def selftest():
    fallos = []
    # 1. TEST NULO — con exponente cero, crecer no cambia nada.
    ok = all(abs(queda(t, 0) - 100) < 1e-12 for t in TAMANOS)
    print(f"[1] test nulo         con exponente cero, todo se queda en 100: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("test nulo: con exponente cero la cuenta se mueve")
    # 2. SEÑAL — la cuenta de doblar reproduce la cifra que escribe el propio artículo (0,95).
    v = queda(2) / 100
    print(f"[2] señal             doblar: {coma(v, 4)}; el artículo escribe {coma(FACTOR_DOBLAR_ARTICULO, 2)}")
    if abs(v - FACTOR_DOBLAR_ARTICULO) > 0.005:
        fallos.append("señal: doblar no da el 0,95 del artículo")
    # 3. INVARIANTE — multiplicar por diez dos veces es multiplicar por cien; y Chinchilla es
    #    cuatro veces más pequeño que Gopher (lo que dice el resumen de 2022).
    ok = abs(queda(10) * queda(10) / 100 - queda(100)) < 1e-9
    (_, ng, _), (_, nc, _) = MODELOS_2022
    ok2 = ng / nc == 4
    print(f"[3] invariante        por diez, dos veces, es por cien: {'sí' if ok else 'NO'}; "
          f"cuatro veces más pequeño: {'sí' if ok2 else 'NO'}")
    if not (ok and ok2):
        fallos.append("invariante: las cuentas no cuadran")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    print()
    print("\n".join(bloque_ley()))
    print()
    print("\n".join(bloque_2022()))


if __name__ == "__main__":
    main()
