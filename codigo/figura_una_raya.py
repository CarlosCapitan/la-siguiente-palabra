#!/usr/bin/env python3
"""
Capítulo 2 — la figura de la raya.

Las cuatro posiciones de los interruptores, en las cuatro esquinas de un cuadrado. A la
izquierda, un montaje que el perceptrón sí aprende: hay una raya que deja las esquinas donde
la luz se enciende a un lado. A la derecha, el o exclusivo: se dibujan tres intentos y ninguno
separa.

El libro no puede pedirle al lector que se dibuje esto en un papel; se lo damos dibujado.

Uso:
    python figura_una_raya.py --selftest
    python figura_una_raya.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../figuras/una_raya.png"
ANCHO_ALTO = (6.2, 3.9)          # pulgadas, para una página de 6 por 9
PUNTOS = 200                     # puntos por pulgada

ESQUINAS = [(0, 0), (0, 1), (1, 0), (1, 1)]
ENCIENDE_FACIL = {(1, 1)}                    # «los dos encendidos»: una raya basta
ENCIENDE_XOR = {(0, 1), (1, 0)}              # «en posiciones distintas»: ninguna raya basta

# tres intentos de raya para el panel derecho, como (pendiente, altura)
INTENTOS = [(-1.0, 0.5), (-1.0, 1.5), (1.0, -0.4)]

LIMITE = (-0.45, 1.45)           # lo que se ve del cuadrado, en las dos direcciones

# Lo que va escrito debajo de cada panel. Es la parte que el lector lee primero y la que
# le dice qué está mirando; por eso está aquí arriba y no enterrada en una llamada.
PIE_IZQUIERDA = "el negro a un lado, los blancos al otro"
PIE_DERECHA = "tres intentos, y en los tres falla uno"

# ==========================================================

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perceptron import hay_raya_que_separa, ESQUINAS_PERCEPTRON as ESQUINAS_P


def panel(ax, encienden, titulo, rayas, sombrear=False):
    """Un cuadrado con las cuatro posiciones de los interruptores y las rayas que se prueban.

    Si `sombrear`, se pinta de gris claro el lado de la raya donde la luz se enciende.
    Sin el sombreado, «a un lado y al otro» es una frase; con él, se ve. Y el lector
    puede comprobar con el dedo que a un lado están todos los negros y al otro todos
    los blancos, que es exactamente lo que dice el texto."""
    if sombrear and rayas:
        m, h = rayas[0]
        xs = [LIMITE[0], LIMITE[1]]
        ax.fill_between(xs, [m * x + h for x in xs], LIMITE[1],
                        color="0.88", zorder=0)
        ax.text(1.42, 1.38, "de este lado\nla luz se enciende",
                fontsize=7.5, color="0.25", ha="right", va="top", zorder=4)
        ax.text(-0.40, -0.40, "de este lado, no",
                fontsize=7.5, color="0.25", ha="left", va="bottom", zorder=4)
    for (x, y) in ESQUINAS:
        lleno = (x, y) in encienden
        ax.plot(x, y, "o", markersize=15,
                markerfacecolor="black" if lleno else "white",
                markeredgecolor="black", markeredgewidth=1.6, zorder=3)
    for (m, h) in rayas:
        xs = [LIMITE[0], LIMITE[1]]
        ax.plot(xs, [m * x + h for x in xs], "--", color="0.35", linewidth=1.2, zorder=2)
    ax.set_xlim(*LIMITE)
    ax.set_ylim(*LIMITE)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["bajado", "subido"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["bajado", "subido"])
    ax.set_xlabel("el interruptor de abajo")
    ax.set_ylabel("el interruptor de arriba")
    ax.set_title(titulo, fontsize=10.5, pad=10)
    ax.set_aspect("equal")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(length=0)


def dibujar():
    plt.rcParams["font.size"] = 9
    fig, (izq, der) = plt.subplots(1, 2, figsize=ANCHO_ALTO)
    panel(izq, ENCIENDE_FACIL, "«los dos subidos»\nuna raya basta", [(-1.0, 1.5)],
          sombrear=True)
    panel(der, ENCIENDE_XOR, "«en posiciones distintas»\nninguna raya basta", INTENTOS)
    fig.tight_layout(rect=[0, 0.18, 1, 0.98])
    for ax, pie in ((izq, PIE_IZQUIERDA), (der, PIE_DERECHA)):
        ax.text(0.5, -0.30, pie, transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color="0.25")
    fig.text(0.5, 0.015,
             "negro: la luz se enciende    ·    blanco: la luz no se enciende",
             ha="center", fontsize=8.5, color="0.3")
    fig.savefig(SALIDA, dpi=PUNTOS)
    print(f"Escrito {SALIDA}")


def selftest():
    fallos = []
    # 1. TEST NULO — un montaje sin nada que separar no puede fallar.
    #    Si «no se enciende nunca» saliera inseparable, el comprobador estaría roto.
    if not hay_raya_que_separa(set()):
        fallos.append("test nulo: dice que el montaje vacío no se puede separar")
    print("[1] test nulo         el montaje vacío sí se separa")

    # 2. SEÑAL IMPLANTADA — el montaje fácil del panel izquierdo tiene que ser separable.
    facil = hay_raya_que_separa(ENCIENDE_FACIL)
    print(f"[2] señal implantada  «los dos subidos» separable: {'sí' if facil else 'NO'}")
    if not facil:
        fallos.append("señal implantada: «los dos subidos» debería ser separable")

    # 3. INVARIANTE DEL DOMINIO — el o exclusivo NO puede separarse con una raya.
    #    Es lo que afirma la figura; si esto saliera que sí, la figura mentiría.
    xor = hay_raya_que_separa(ENCIENDE_XOR)
    print(f"[3] invariante        «en posiciones distintas» separable: {'sí' if xor else 'no'}")
    if xor:
        fallos.append("invariante: encontró una raya para el o exclusivo; la figura sería falsa")

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
