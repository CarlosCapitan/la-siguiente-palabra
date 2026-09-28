#!/usr/bin/env python3
"""
Capítulo 14 — la retina de la salamandra: la imagen va en cuál llega antes (L24, E15).

Gollisch y Meister (2008, «Rapid neural coding in the retina with relative spike latencies»,
Science 319, pp. 1108-1111; resumen en https://europepmc.org/article/MED/18292344) midieron que
ciertas células de la retina de la salamandra llevan la forma de una imagen que se enseña un
instante en CUÁNDO llega su primer pulso. Con menos contraste, cada célula tarda más; pero las dos
tardan más a la vez, y la distancia entre sus dos primeros pulsos casi no cambia. Otra imagen, en
cambio, sí la cambia.

ESTO ES UN ESQUEMA, NO UNA MEDICIÓN: no lleva ningún número, y las distancias no están a escala.
Las posiciones de las marcas (ESQUEMA, abajo) solo tienen que cumplir lo que dice el artículo:
con poco contraste, las dos marcas se retrasan lo mismo; con otra imagen, la distancia cambia.
El selftest comprueba justo eso.

Uso:
    python figura_la_salamandra.py --selftest
    python figura_la_salamandra.py
"""

# ======================= CONSTANTES =======================

DESTINO = "../figuras/la_salamandra.png"
ALTO = 4.75                # pulgadas
ALTO_PANEL = 24            # unidades del lienzo
X0, X1 = 26, 92            # dónde empieza y acaba la línea del tiempo
# (rótulo, primer pulso de la célula 1, primer pulso de la célula 2), en unidades del lienzo
ESQUEMA = (
    ("La imagen A, con mucho contraste", 40, 54),
    ("La imagen A, con poco contraste", 52, 66),
    ("Otra imagen, B, con mucho contraste", 48, 42),
)

# ==========================================================

import argparse
import sys
from pathlib import Path

from matplotlib.patches import FancyArrowPatch

from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def dibujar(esquema, paleta, ruta):
    L = Lienzo("Lo que lleva la imagen es cuál llega antes",
               "Dos células de la retina de una salamandra. Cada raya es el primer pulso\n"
               "de una célula después de que aparezca la imagen.",
               paleta, alto=ALTO)
    p, ax = L.p, L.ax
    distancias = []
    for n, (titulo, t1, t2) in enumerate(esquema, 1):
        _, ytop, _ = L.panel(n, titulo, ALTO_PANEL)
        filas = (ytop - 3.0, ytop - 9.5)
        # la línea vertical: aparece la imagen
        ax.plot([X0, X0], [filas[1] - 2.8, filas[0] + 2.4], color=p.suave, linewidth=0.8,
                linestyle=(0, (2, 1.5)))
        for y, t, nombre in ((filas[0], t1, "célula 1"), (filas[1], t2, "célula 2")):
            L.texto(8, y, nombre, tam=7.6, negrita=True)
            ax.add_patch(FancyArrowPatch((X0, y), (X1, y), arrowstyle="-|>", mutation_scale=8,
                                         color=p.marco, linewidth=0.9))
            ax.plot([t, t], [y - 2.0, y + 2.0], color=p.tinta, linewidth=2.6)
        a, b = sorted((t1, t2))
        yd = (filas[0] + filas[1]) / 2
        ax.add_patch(FancyArrowPatch((a, yd), (b, yd), arrowstyle="<|-|>", mutation_scale=7,
                                     color=p.acento, linewidth=1.3))
        L.texto(b + 2, yd, "la distancia", tam=7.2, color=p.suave)
        if n == 1:
            L.texto(X0, filas[1] - 4.6, "aparece la imagen", tam=7.0, color=p.suave, ha="center")
            L.texto(X1, filas[1] - 4.6, "tiempo", tam=7.0, color=p.suave, ha="right")
        distancias.append(t2 - t1)
    L.pie("Esquema, sin escala. Lo medido (Gollisch y Meister, 2008): con menos contraste las dos\n"
          "células tardan más, pero la distancia entre sus primeros pulsos casi no cambia.")
    L.guardar(ruta)
    return distancias


def selftest():
    fallos = []
    # 1. TEST NULO — si las dos células disparan a la vez, la distancia es cero en la figura.
    d = dibujar((("prueba", 50, 50),) * 3, GRIS, "/dev/null")
    print(f"[1] test nulo         dos pulsos a la vez, distancia dibujada: {d[0]}")
    if d[0] != 0:
        fallos.append("test nulo: aparece una distancia entre dos pulsos simultáneos")
    # 2. SEÑAL IMPLANTADA — lo que dice el artículo y el esquema tiene que enseñar: con poco
    #    contraste las dos marcas van más tarde, y otra imagen cambia la distancia.
    (_, a1, a2), (_, b1, b2), (_, c1, c2) = ESQUEMA
    ok = b1 > a1 and b2 > a2 and (c2 - c1) != (a2 - a1)
    print(f"[2] señal implantada  con poco contraste, más tarde: {'sí' if b1 > a1 and b2 > a2 else 'NO'}; "
          f"otra imagen cambia la distancia: {'sí' if (c2 - c1) != (a2 - a1) else 'NO'}")
    if not ok:
        fallos.append("señal: el esquema no enseña lo que dice el artículo")
    # 3. INVARIANTE DEL DOMINIO — con la misma imagen, la distancia no cambia con el contraste; y
    #    todas las marcas caen dentro de la línea del tiempo, después de que aparezca la imagen.
    d = dibujar(ESQUEMA, GRIS, "/dev/null")
    dibujar(ESQUEMA, COLOR, "/dev/null")
    dentro = all(X0 < t < X1 for _, t1, t2 in ESQUEMA for t in (t1, t2))
    print(f"[3] invariante        distancias {d}; misma imagen, misma distancia: "
          f"{'sí' if d[0] == d[1] else 'NO'}; marcas dentro: {'sí' if dentro else 'NO'}")
    if d[0] != d[1] or not dentro:
        fallos.append("invariante: el esquema cambia la distancia con el contraste o se sale")
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
    dibujar(ESQUEMA, GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
