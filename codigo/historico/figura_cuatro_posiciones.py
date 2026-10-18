#!/usr/bin/env python3
"""
Capítulo 2 — la figura de las cuatro posiciones.

La lámpara del pasillo tiene dos interruptores, y dos interruptores solo se pueden dejar de
cuatro maneras. La figura las enseña las cuatro, dibujadas: cada ficha es una posición, con
los dos interruptores tal como están en la pared y la bombilla encendida o apagada.

Es el montaje de tu pasillo, «en posiciones distintas»: la luz encendida cuando uno está
subido y el otro bajado. Cada interruptor es un marco con una tecla: la tecla arriba es
subido y la tecla abajo es bajado, que es como se ve en la pared.

Qué bombilla se enciende en cada ficha no se decide aquí: sale de NOMBRES, la lista de los
dieciséis montajes de perceptron.py, para que la figura no pueda contradecir a la tabla.

Uso:
    python figura_cuatro_posiciones.py --selftest
    python figura_cuatro_posiciones.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../figuras/cuatro_posiciones.png"
ANCHO_ALTO = (6.2, 5.4)          # pulgadas, para una página de 6 por 9
PUNTOS = 200                     # puntos por pulgada

# El montaje que se dibuja, por el nombre que tiene en perceptron.NOMBRES.
MONTAJE = "en posiciones distintas"            # el de tu pasillo

# Las cuatro posiciones, en el orden de perceptron.TABLA_DOS y de la tabla del libro:
# (el interruptor de abajo, el interruptor de arriba); 0 es bajado y 1 es subido.
POSICIONES = [(0, 0), (0, 1), (1, 0), (1, 1)]
ROTULO_ABAJO = "de abajo"        # el interruptor de abajo, como lo llama el libro
ROTULO_ARRIBA = "de arriba"
PALABRA = {0: "bajado", 1: "subido"}

# Colores: una bombilla encendida es amarilla y una apagada es gris. Nada más.
COLOR_ENCENDIDA = "#d9b84a"
COLOR_APAGADA = "#c9c9c9"
COLOR_FICHA = "#f0f1f2"
COLOR_TINTA = "#1f2a30"

# ==========================================================

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

from perceptron import NOMBRES

BITS_DE = {nombre: bits for bits, nombre in NOMBRES.items()}   # nombre -> sus cuatro casillas


def enciende(montaje, posicion):
    """¿Con este montaje, en esta posición, la luz está encendida? Lo dice la tabla de los
    dieciséis: la casilla de esa posición, en el orden de POSICIONES."""
    return bool(BITS_DE[montaje][POSICIONES.index(posicion)])


def interruptor(ax, x, y, subido, escala=1.0):
    """Un interruptor de pared: un marco y una tecla horizontal, pegada al borde de arriba si
    está subido y al de abajo si está bajado. Es el mismo dibujo en todas las fichas, solo
    cambia dónde está la tecla: así el lector ve la posición sin leerla. La figura de los
    dieciséis montajes lo usa en miniatura (escala < 1)."""
    e = escala
    ax.add_patch(FancyBboxPatch((x - 0.7 * e, y - 1.15 * e), 1.4 * e, 2.3 * e,
                                boxstyle=f"round,pad=0.02,rounding_size={0.18 * e}",
                                facecolor="white", edgecolor="0.55", linewidth=1.0 * e))
    dy = 0.62 * e if subido else -0.62 * e
    ax.plot([x - 0.38 * e, x + 0.38 * e], [y + dy, y + dy], color=COLOR_TINTA,
            linewidth=4.0 * e, solid_capstyle="round")


def ficha(ax, numero, montaje, posicion):
    """Una posición: los dos interruptores con su palabra debajo, y la bombilla."""
    ax.add_patch(FancyBboxPatch((0, 0), 12, 10, boxstyle="round,pad=0.02,rounding_size=0.5",
                                facecolor=COLOR_FICHA, edgecolor="none"))
    ax.text(0.7, 9.0, str(numero), fontsize=12, fontweight="bold", color="0.35")
    abajo, arriba = posicion
    for x, valor, rotulo in ((3.2, abajo, ROTULO_ABAJO), (8.8, arriba, ROTULO_ARRIBA)):
        interruptor(ax, x, 6.5, bool(valor), escala=1.15)
        ax.text(x, 4.75, rotulo, ha="center", va="top", fontsize=10, color="0.3")
        ax.text(x, 3.75, PALABRA[valor], ha="center", va="top", fontsize=10.5,
                fontweight="bold", color=COLOR_TINTA)
    luz = enciende(montaje, posicion)
    ax.add_patch(Circle((3.2, 1.45), 0.75, facecolor=COLOR_ENCENDIDA if luz else COLOR_APAGADA,
                        edgecolor="0.45", linewidth=1.0))
    ax.text(4.6, 1.45, "encendida" if luz else "apagada", va="center", fontsize=12,
            fontweight="bold", color=COLOR_TINTA)
    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(-0.2, 10.2)
    ax.set_aspect("equal")
    ax.axis("off")


def dibujar():
    plt.rcParams["font.family"] = "serif"
    fig, ejes = plt.subplots(2, 2, figsize=ANCHO_ALTO)
    for k, posicion in enumerate(POSICIONES):
        ficha(ejes[k // 2][k % 2], k + 1, MONTAJE, posicion)
    fig.text(0.5, 0.965, f"la lámpara del pasillo con el montaje «{MONTAJE}»",
             ha="center", va="center", fontsize=11, color=COLOR_TINTA)
    fig.text(0.5, 0.012, "la tecla del interruptor arriba: subido    ·    la tecla abajo: bajado",
             ha="center", va="bottom", fontsize=9, color="0.3")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.07, hspace=0.1, wspace=0.06)
    fig.savefig(SALIDA, dpi=PUNTOS)
    print(f"Escrito {SALIDA}")
    print(f"«{MONTAJE}»: la luz se enciende en las posiciones "
          f"{[i + 1 for i, p in enumerate(POSICIONES) if enciende(MONTAJE, p)]}")


def selftest():
    fallos = []
    # 1. TEST NULO — el montaje «nunca» no enciende la luz en ninguna posición. Si aquí
    #    saliera una bombilla encendida, la figura no estaría leyendo la tabla.
    encendidas = [p for p in POSICIONES if enciende("nunca", p)]
    print(f"[1] test nulo         «nunca» enciende {len(encendidas)} bombillas")
    if encendidas:
        fallos.append(f"test nulo: «nunca» enciende la luz en {encendidas}")

    # 2. SEÑAL IMPLANTADA — el montaje de tu pasillo se enciende exactamente en las dos
    #    posiciones con un interruptor subido y el otro bajado, y en ninguna más.
    xor = [p for p in POSICIONES if enciende(MONTAJE, p)]
    print(f"[2] señal implantada  «{MONTAJE}» se enciende en {xor}")
    if xor != [(0, 1), (1, 0)]:
        fallos.append(f"señal implantada: «{MONTAJE}» se enciende en {xor}, "
                      "y tenía que ser en (0, 1) y (1, 0)")

    # 3. INVARIANTE DEL DOMINIO — en cada uno de los dieciséis montajes, encender la luz en
    #    una posición es exactamente lo que dice su casilla: la figura nunca puede
    #    contradecir a la tabla porque lee de ella. Y las posiciones son cuatro, distintas.
    contradice = [(n, p) for n, bits in BITS_DE.items() for p in POSICIONES
                  if enciende(n, p) != bool(bits[POSICIONES.index(p)])]
    print(f"[3] invariante        {len(POSICIONES)} posiciones distintas y "
          f"{len(contradice)} contradicciones con la tabla de los dieciséis")
    if len(set(POSICIONES)) != 4 or contradice:
        fallos.append(f"invariante: posiciones {POSICIONES}, contradicciones {contradice}")

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
