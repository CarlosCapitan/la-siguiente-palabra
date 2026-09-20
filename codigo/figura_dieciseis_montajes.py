#!/usr/bin/env python3
"""
Capítulo 2 — la figura de los dieciséis montajes.

Los dieciséis montajes de la lámpara del pasillo, dibujados en vez de listados: la misma
tabla que imprime perceptron.py, con una fila por montaje y una columna por posición de los
interruptores, pero con los interruptores dibujados en la cabecera (grandes, una sola vez,
con los números de la figura de las cuatro posiciones) y cada casilla en amarillo si con
esa posición la luz se enciende. Solo hay una lámpara: no se dibuja ninguna bombilla. El
lector cuenta las filas con el dedo, ve que son todas distintas, y ve marcadas las dos con
las que el perceptrón se estrella.

Cuáles son esos dos no se decide aquí: se entrena el perceptrón con los dieciséis, como en
perceptron.py, y se marcan los que no aprende. Si un día el perceptrón aprendiera otros, la
figura cambiaría sola; si no, la figura mentiría.

Uso:
    python figura_dieciseis_montajes.py --selftest
    python figura_dieciseis_montajes.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../figuras/dieciseis_montajes.png"
ANCHO_ALTO = (6.2, 8.06)          # pulgadas, para una página de 6 por 9
PUNTOS = 200                     # puntos por pulgada

# El orden de las filas es el de la tabla de perceptron.py: por sus cuatro casillas, de
# «nunca» (no no no no) a «siempre» (sí sí sí sí).

# Las cuatro posiciones (columnas), en el orden de perceptron.TABLA_DOS y con los números
# de la figura de las cuatro posiciones: 1 los dos bajados, 2 solo el de arriba subido,
# 3 solo el de abajo subido, 4 los dos subidos (con los nombres de perceptron.NOMBRES).
POSICIONES = [(0, 0), (0, 1), (1, 0), (1, 1)]
PALABRA = {0: "bajado", 1: "subido"}

COLOR_ENCENDIDA = "#d9b84a"
COLOR_APAGADA = "#d4d4d4"
COLOR_FICHA_FALLIDA = "#fbe9e4"
COLOR_BORDE_FALLIDA = "#b5533c"
COLOR_TINTA = "#1f2a30"

# ==========================================================

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from perceptron import NOMBRES, reglas_de_dos_interruptores, hay_raya_que_separa, \
    ESQUINAS_PERCEPTRON
from figura_cuatro_posiciones import interruptor      # el mismo dibujo que en esa figura

ORDEN = sorted(NOMBRES)           # los dieciséis, en el orden de la tabla del libro


def montajes_fallidos():
    """Los que el perceptrón no aprende, entrenándolo de verdad: los mismos que imprime
    perceptron.py con la marca «NO PUEDE»."""
    _, fallidas, _ = reglas_de_dos_interruptores()
    return set(fallidas)


def dibujar():
    """La tabla de los dieciséis, dibujada: una fila por montaje, una columna por posición
    de los interruptores. Los interruptores se dibujan UNA vez, en la cabecera —una fila
    para el de abajo y otra para el de arriba, con la palabra debajo de cada uno— y cada
    casilla dice si con esa posición la luz se enciende. Solo hay una lámpara, y por eso no
    se dibuja ninguna bombilla: la casilla es lo que le pasa a la luz."""
    plt.rcParams["font.family"] = "serif"
    fallidas = montajes_fallidos()
    fig = plt.figure(figsize=ANCHO_ALTO)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 52)
    ax.set_aspect("equal")
    ax.axis("off")

    X_NOMBRE = 1.8                     # borde izquierdo de la columna de nombres
    X_COL = [20.6, 25.8, 31.0, 36.2]   # centro de las cuatro columnas
    ANCHO_CASILLA, ALTO_FILA = 4.6, 1.72
    Y0 = 31.4                          # centro de la primera fila

    # Título y cómo leerla.
    ax.text(20, 50.4, "Los dieciséis montajes de la lámpara del pasillo", ha="center",
            va="center", fontsize=13, fontweight="bold", color=COLOR_TINTA)
    ax.text(20, 48.35, "Cada columna es una posición de los dos interruptores.\n"
            "Cada fila es un montaje: cuándo se enciende la luz.",
            ha="center", va="center", fontsize=8.8, color="0.3", linespacing=1.35)

    # Cabecera: número de la posición, y los dos interruptores en dos filas, con la palabra.
    for k, (x, (abajo, arriba)) in enumerate(zip(X_COL, POSICIONES)):
        ax.text(x, 46.0, str(k + 1), ha="center", va="center", fontsize=11,
                fontweight="bold", color="0.35")
        for y, valor in ((43.6, abajo), (39.3, arriba)):
            interruptor(ax, x, y, bool(valor), escala=0.95)
            ax.text(x, y - 1.35, PALABRA[valor], ha="center", va="top", fontsize=7.6,
                    color="0.3")
    ax.text(X_NOMBRE, 43.6, "el interruptor de abajo", ha="left", va="center", fontsize=9,
            fontweight="bold", color=COLOR_TINTA)
    ax.text(X_NOMBRE, 39.3, "el interruptor de arriba", ha="left", va="center", fontsize=9,
            fontweight="bold", color=COLOR_TINTA)
    ax.plot([1.0, 39.0], [36.4, 36.4], color="0.6", linewidth=0.8)
    ax.text(X_NOMBRE, 34.9, "la luz se enciende…", ha="left", va="center", fontsize=9.5,
            fontweight="bold", color=COLOR_TINTA)
    ax.text((X_COL[0] + X_COL[3]) / 2, 34.9, "¿se enciende?", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=COLOR_TINTA)
    ax.plot([1.0, 39.0], [33.5, 33.5], color="0.6", linewidth=0.8)

    for fila, bits in enumerate(ORDEN):
        y = Y0 - fila * ALTO_FILA
        fallida = bits in fallidas
        if fallida:
            ax.add_patch(FancyBboxPatch((0.9, y - ALTO_FILA / 2 + 0.1), 38.2, ALTO_FILA - 0.2,
                                        boxstyle="round,pad=0.02,rounding_size=0.4",
                                        facecolor=COLOR_FICHA_FALLIDA,
                                        edgecolor=COLOR_BORDE_FALLIDA, linewidth=1.2))
        ax.text(X_NOMBRE, y, NOMBRES[bits] + ("  *" if fallida else ""), ha="left",
                va="center", fontsize=8.6, color=COLOR_BORDE_FALLIDA if fallida else COLOR_TINTA)
        for x, encendida in zip(X_COL, bits):
            ax.add_patch(FancyBboxPatch((x - ANCHO_CASILLA / 2, y - 0.72), ANCHO_CASILLA, 1.44,
                                        boxstyle="round,pad=0.02,rounding_size=0.3",
                                        facecolor=COLOR_ENCENDIDA if encendida else COLOR_APAGADA,
                                        edgecolor="none"))
            ax.text(x, y, "sí" if encendida else "no", ha="center", va="center",
                    fontsize=8.5, fontweight="bold" if encendida else "normal",
                    color=COLOR_TINTA if encendida else "0.4")

    # Leyenda y nota al pie.
    for x, texto, color, palabra in ((11.5, "sí", COLOR_ENCENDIDA, "la luz se enciende"),
                                     (23.5, "no", COLOR_APAGADA, "no se enciende")):
        ax.add_patch(FancyBboxPatch((x - 1.6, 2.1), 3.2, 1.4,
                                    boxstyle="round,pad=0.02,rounding_size=0.3",
                                    facecolor=color, edgecolor="none"))
        ax.text(x, 2.8, texto, ha="center", va="center", fontsize=8.5,
                fontweight="bold" if texto == "sí" else "normal", color=COLOR_TINTA)
        ax.text(x + 2.1, 2.8, palabra, ha="left", va="center", fontsize=8.5, color="0.3")
    ax.text(20, 0.8, "*  un solo perceptrón no puede aprender los dos montajes señalados",
            ha="center", va="center", fontsize=8.5, color=COLOR_BORDE_FALLIDA)
    fig.savefig(SALIDA, dpi=PUNTOS)
    print(f"Escrito {SALIDA}")
    print(f"montajes dibujados: {len(ORDEN)}")
    print("marcados con asterisco: " + ", ".join(f"«{NOMBRES[b]}»" for b in ORDEN
                                                 if b in fallidas))


def selftest():
    fallos = []
    # 1. TEST NULO — sin entrenar nada, los dieciséis son dieciséis y todos distintos: si la
    #    lista de nombres repitiera casillas, la cuadrícula tendría dos fichas iguales.
    distintos = len(set(ORDEN))
    print(f"[1] test nulo         {distintos} montajes distintos de {len(ORDEN)}")
    if distintos != 16 or len(ORDEN) != 16:
        fallos.append(f"test nulo: {distintos} distintos de {len(ORDEN)}; tenían que ser 16 y 16")

    # 2. SEÑAL IMPLANTADA — los marcados tienen que ser exactamente los dos que ninguna raya
    #    separa, y esos dos son «en posiciones distintas» y «en la misma posición».
    fallidas = montajes_fallidos()
    inseparables = set()
    for bits in ORDEN:
        encienden = [e for e, b in zip(ESQUINAS_PERCEPTRON, bits) if b]
        if encienden and len(encienden) < 4 and not hay_raya_que_separa(encienden):
            inseparables.add(bits)
    nombres = sorted(NOMBRES[b] for b in fallidas)
    print(f"[2] señal implantada  marcados: {nombres}; inseparables por una raya: "
          f"{sorted(NOMBRES[b] for b in inseparables)}")
    if fallidas != inseparables or nombres != ["en la misma posición", "en posiciones distintas"]:
        fallos.append(f"señal implantada: marcados {nombres}, inseparables "
                      f"{sorted(NOMBRES[b] for b in inseparables)}")

    # 3. INVARIANTE DEL DOMINIO — cada posición está encendida en exactamente ocho de los
    #    dieciséis montajes: es lo que significa «todas las maneras de rellenar cuatro
    #    casillas», y si faltara o sobrara un montaje, alguna columna no sumaría ocho.
    sumas = [sum(bits[i] for bits in ORDEN) for i in range(4)]
    print(f"[3] invariante        cada posición encendida en {sumas} de 16 montajes")
    if sumas != [8, 8, 8, 8]:
        fallos.append(f"invariante: las sumas por posición son {sumas}, no [8, 8, 8, 8]")

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
    if args.selftest:
        sys.exit(selftest())
    dibujar()


if __name__ == "__main__":
    main()
