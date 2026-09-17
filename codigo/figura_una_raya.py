#!/usr/bin/env python3
"""
Capítulo 2 — la figura de la raya.

Cuatro casos en las cuatro esquinas de un cuadrado. A la izquierda, una regla que el perceptrón
sí aprende: hay una raya que deja las encendidas a un lado. A la derecha, el o exclusivo: se
dibujan tres intentos y ninguno separa.

El libro no puede pedirle al lector que se dibuje esto en un papel; se lo damos dibujado.

Uso:
    python figura_una_raya.py --selftest
    python figura_una_raya.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../figuras/una_raya.png"
ANCHO_ALTO = (6.2, 3.3)          # pulgadas, para una página de 6 por 9
PUNTOS = 200                     # puntos por pulgada

ESQUINAS = [(0, 0), (0, 1), (1, 0), (1, 1)]
ENCIENDE_FACIL = {(1, 1)}                    # «los dos encendidos»: una raya basta
ENCIENDE_XOR = {(0, 1), (1, 0)}              # «en posiciones distintas»: ninguna raya basta

# tres intentos de raya para el panel derecho, como (pendiente, altura)
INTENTOS = [(-1.0, 0.5), (-1.0, 1.5), (1.0, -0.4)]

# ==========================================================

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perceptron import hay_raya_que_separa, ESQUINAS_PERCEPTRON as ESQUINAS_P


def panel(ax, encienden, titulo, rayas):
    for (x, y) in ESQUINAS:
        lleno = (x, y) in encienden
        ax.plot(x, y, "o", markersize=15,
                markerfacecolor="black" if lleno else "white",
                markeredgecolor="black", markeredgewidth=1.6, zorder=3)
    for (m, h) in rayas:
        xs = [-0.45, 1.45]
        ax.plot(xs, [m * x + h for x in xs], "--", color="0.35", linewidth=1.2, zorder=2)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(-0.45, 1.45)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["bajado", "subido"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["bajado", "subido"])
    ax.set_xlabel("el interruptor de abajo")
    ax.set_ylabel("el de arriba")
    ax.set_title(titulo, fontsize=10.5, pad=10)
    ax.set_aspect("equal")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(length=0)


def dibujar():
    plt.rcParams["font.size"] = 9
    fig, (izq, der) = plt.subplots(1, 2, figsize=ANCHO_ALTO)
    panel(izq, ENCIENDE_FACIL, "«los dos subidos»\nuna raya basta", [(-1.0, 1.5)])
    panel(der, ENCIENDE_XOR, "«en posiciones distintas»\nninguna raya basta", INTENTOS)
    fig.text(0.5, 0.015,
             "negro: la luz se enciende    ·    blanco: la luz no se enciende",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(SALIDA, dpi=PUNTOS)
    print(f"Escrito {SALIDA}")


def selftest():
    fallos = []
    # 1. TEST NULO — una regla sin nada que separar no puede fallar.
    #    Si «no se enciende nunca» saliera inseparable, el comprobador estaría roto.
    if not hay_raya_que_separa(set()):
        fallos.append("test nulo: dice que la regla vacía no se puede separar")
    print("[1] test nulo         la regla vacía sí se separa")

    # 2. SEÑAL IMPLANTADA — la regla fácil del panel izquierdo tiene que ser separable.
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
