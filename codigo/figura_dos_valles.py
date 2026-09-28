#!/usr/bin/env python3
"""
Capítulo 15 — el terreno con dos números: un valle hondo, uno poco hondo y dos bolas (L24, E26).

Un mapa de relieve visto desde arriba: cada punto es una combinación de los dos números, y el gris
dice la altura (lo mal que lo hace): cuanto más oscuro, más abajo. Las dos bolas bajan siempre
cuesta abajo desde donde se sueltan; una acaba en el valle poco hondo y otra en el hondo.

Los caminos NO se calculan aquí: se leen de `datos/salidas/dos_valles.csv`, que escribe
`dos_valles.py`. El relieve de fondo se dibuja con la misma función de altura de ese programa.

Uso:
    python figura_dos_valles.py --selftest
    python figura_dos_valles.py
"""

# ======================= CONSTANTES =======================

CAMINOS = "../datos/salidas/dos_valles.csv"
DESTINO = "../figuras/dos_valles.png"
ALTO = 4.3                 # pulgadas
REJILLA = 160              # puntos por lado para dibujar el relieve
NIVELES = 12               # cuántos tonos de gris

# ==========================================================

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from dos_valles import LIMITE, VALLES, altura, en_que_valle
from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def leer(ruta=AQUI / CAMINOS):
    caminos = {}
    with open(ruta, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            caminos.setdefault(int(f["bola"]), []).append((float(f["x"]), float(f["y"]), float(f["altura"])))
    assert sorted(caminos) == [1, 2], f"se esperaban dos bolas; hay {sorted(caminos)}"
    return caminos


def dibujar(caminos, paleta, ruta):
    L = Lienzo("Dos bolas, dos valles",
               "El terreno del error con dos pesos: cada punto del mapa da un valor a cada\n"
               "uno, y el gris dice lo mal que lo hace ahí. Cuanto más oscuro, más abajo.",
               paleta, alto=ALTO)
    p = L.p
    ancho_u, alto_u = 100, L.alto_u
    arriba = L.y - 1
    lado = 60
    x0 = (100 - lado) / 2 + 2
    ax = L.fig.add_axes([x0 / ancho_u, (arriba - lado) / alto_u, lado / ancho_u, lado / alto_u])
    g = np.linspace(-LIMITE, LIMITE, REJILLA)
    X, Y = np.meshgrid(g, g)
    Z = np.vectorize(altura)(X, Y)
    ax.contourf(X, Y, Z, levels=NIVELES, cmap="Greys_r", vmin=Z.min() - 1.5, vmax=Z.max() + 0.8)
    ax.contour(X, Y, Z, levels=NIVELES, colors=["white"], linewidths=0.5)
    nombres = {0: "valle hondo", 1: "valle poco hondo"}
    for i, (cx, cy, _, _) in enumerate(VALLES):
        ax.annotate(nombres[i], (cx, cy), xytext=(cx + 0.9, cy - 0.35) if i == 0 else (cx - 0.9, cy + 0.35),
                    fontsize=7.6, color=p.tinta, fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9))
    finales = {}
    for k, c in caminos.items():
        xs, ys = [a for a, _, _ in c], [b for _, b, _ in c]
        ax.plot(xs, ys, color="white", linewidth=2.8)
        ax.plot(xs, ys, color=p.tinta, linewidth=1.3, linestyle=(0, (2, 1.5)))
        ax.plot(xs[0], ys[0], "o", markersize=7, markerfacecolor="white", markeredgecolor=p.tinta)
        ax.plot(xs[-1], ys[-1], "o", markersize=7, markerfacecolor=p.tinta, markeredgecolor="white")
        ax.annotate(f"bola {k}", (xs[0], ys[0]), xytext=(xs[0] + 0.35, ys[0] + (0.35 if ys[0] < 0 else -0.55)),
                    fontsize=7.6, color=p.tinta, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))
        finales[k] = en_que_valle(xs[-1], ys[-1])
    ax.set_xlim(-LIMITE, LIMITE)
    ax.set_ylim(-LIMITE, LIMITE)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("el primer peso", fontsize=7.6, color=p.tinta)
    ax.set_ylabel("el segundo peso", fontsize=7.6, color=p.tinta)
    for s in ax.spines.values():
        s.set_color(p.marco)
    # la clave, en una fila debajo del mapa
    yc = arriba - lado - 9
    L.ax.plot([6], [yc], "o", markersize=6, markerfacecolor="white", markeredgecolor=p.tinta)
    L.texto(8, yc, "donde se suelta", tam=7.2, color=p.suave)
    L.ax.plot([33], [yc], "o", markersize=6, markerfacecolor=p.tinta, markeredgecolor=p.tinta)
    L.texto(35, yc, "donde se para", tam=7.2, color=p.suave)
    L.ax.plot([58, 63], [yc, yc], color=p.tinta, linewidth=1.3, linestyle=(0, (2, 1.5)))
    L.texto(64.5, yc, "el camino, cuesta abajo", tam=7.2, color=p.suave)
    L.pie("Terreno de juguete, con dos pesos para poder dibujarlo. Una máquina de verdad tiene\n"
          "millones: el terreno no se puede dibujar, pero el problema es el mismo.")
    L.guardar(ruta)
    return finales


def selftest():
    fallos = []
    caminos = leer()
    # 1. TEST NULO — una bola que se suelta ya en el fondo del valle hondo acaba en el hondo: la
    #    figura no la manda a otro sitio.
    cx, cy = VALLES[0][:2]
    fin = dibujar({1: [(cx, cy, altura(cx, cy))], 2: [(cx, cy, altura(cx, cy))]}, GRIS, "/dev/null")
    print(f"[1] test nulo         dos bolas en el fondo del hondo se quedan en: {fin}")
    if fin != {1: 0, 2: 0}:
        fallos.append("test nulo: la figura cambia de valle una bola que no se mueve")
    # 2. SEÑAL IMPLANTADA — lo que dice el capítulo: una bola acaba en el poco hondo y la otra en
    #    el hondo.
    fin = dibujar(caminos, GRIS, "/dev/null")
    print(f"[2] señal implantada  valle donde acaba cada bola: {fin}")
    if sorted(fin.values()) != [0, 1]:
        fallos.append("señal: las dos bolas no acaban en valles distintos")
    # 3. INVARIANTE DEL DOMINIO — la altura que trae la salida es la de la función del terreno en
    #    cada punto del camino, y la figura sale en color y en gris.
    casa = all(abs(h - altura(x, y)) < 1e-4 for c in caminos.values() for x, y, h in c)
    dibujar(caminos, COLOR, "/dev/null")
    print(f"[3] invariante        las alturas de la salida son las del terreno: {'sí' if casa else 'NO'}")
    if not casa:
        fallos.append("invariante: la salida y el terreno no casan")
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
    dibujar(leer(), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
