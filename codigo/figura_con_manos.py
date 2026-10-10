#!/usr/bin/env python3
"""
Capítulo 14 — la línea de metro del capítulo 7, con una estación fuera de la máquina.

Figura de mecanismo (regla 9; AUDITORIA.md, paso 7): no lleva ninguna cifra, así que no hay nada
que contrastar con una salida. Las cinco estaciones son las de `figura_linea_circular.py`, con
los mismos nombres, para que el lector reconozca el plano. Lo nuevo es un ramal: si lo que la
máquina escribe es una orden, el texto sale de la línea, un programa la ejecuta y devuelve el
resultado como texto, que entra por la primera estación como cualquier otro texto.

Uso:
    python figura_con_manos.py --selftest
    python figura_con_manos.py
"""

# ======================= CONSTANTES =======================

DESTINO = "../figuras/con_manos.png"
ALTO = 3.3
CENTRO, RADIO = (40.0, 34.0), 14.0
CAJA = (70.0, 23.0, 27.0, 22.0)          # x, y, ancho, alto de la estación de fuera
# Lo que se dice al lado de cada estación (los nombres son los del capítulo 7)
QUE = ["el texto, en piezas", "una lista por pieza", "de mirar y mezclar",
       "de probabilidades", ""]
FUERA = ["Fuera de la máquina", "un programa lee", "la orden, la ejecuta", "y devuelve texto"]

# ==========================================================

import argparse
import math
import re
import sys
from pathlib import Path

from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from figura_linea_circular import ESTACIONES
from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent
NOMBRES = [n for n, _, _ in ESTACIONES]


def posiciones():
    """La 5 a la derecha, mirando a la estación de fuera; en el sentido del reloj, como en el 7."""
    cx, cy = CENTRO
    n = len(NOMBRES)
    out = []
    for k in range(n):
        ang = 2 * math.pi * (n - 1) / n - 2 * math.pi * k / n
        out.append((cx + RADIO * math.cos(ang), cy + RADIO * math.sin(ang), ang))
    return out


def rotulos():
    return list(zip(NOMBRES, QUE)) + [("", f) for f in FUERA] + [
        ("si escribe", "una orden"), ("el resultado,", "como texto nuevo")]


def dibujar(paleta, ruta):
    L = Lienzo("La misma línea, con una estación fuera",
               "Si lo que escribe la máquina es una orden, un programa la ejecuta y le\n"
               "devuelve el resultado como texto. La máquina no cambia.", paleta, alto=ALTO)
    p, ax = L.p, L.ax
    cx, cy = CENTRO
    ax.add_patch(Circle((cx, cy), RADIO, facecolor="none", edgecolor=p.tinta, linewidth=3.0))
    pos = posiciones()
    for k, ((x, y, ang), nombre, que) in enumerate(zip(pos, NOMBRES, QUE)):
        ax.add_patch(Circle((x, y), 2.8, facecolor="white", edgecolor=p.tinta, linewidth=1.6,
                            zorder=3))
        L.texto(x, y, str(k + 1), tam=8.6, ha="center", negrita=True)
        if k == len(pos) - 1:                     # la 5: su rótulo, dentro del círculo
            L.texto(x - 4.5, y + 1.6, nombre, tam=7.8, ha="right", negrita=True)
            continue
        dx, dy = math.cos(ang), math.sin(ang)
        ha = "left" if dx > 0.3 else ("right" if dx < -0.3 else "center")
        if k == 0:                                # la 1: a la izquierda, para dejar sitio a la
            ha = "right"                          # flecha que vuelve desde fuera
        tx = x + (4.5 if ha == "left" else -4.5 if ha == "right" else 0)
        ty = y if (dx < -0.3) else y + (5.0 if dy > 0 else -4.6)
        L.texto(tx, ty + 1.4, nombre, tam=7.8, ha=ha, negrita=True)
        L.texto(tx, ty - 1.4, que, tam=6.9, ha=ha, color=p.suave)
    # el sentido de la marcha
    n = len(pos)
    for k in range(n):
        a1 = 2 * math.pi * (n - 1) / n - 2 * math.pi * (k + 0.42) / n
        a2 = 2 * math.pi * (n - 1) / n - 2 * math.pi * (k + 0.58) / n
        ax.add_patch(FancyArrowPatch((cx + RADIO * math.cos(a1), cy + RADIO * math.sin(a1)),
                                     (cx + RADIO * math.cos(a2), cy + RADIO * math.sin(a2)),
                                     arrowstyle="-|>", mutation_scale=10, color=p.tinta,
                                     zorder=4))
    # la estación de fuera
    x0, y0, w, h = CAJA
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                facecolor=p.fondo, edgecolor=p.tinta, linewidth=1.4))
    for i, s in enumerate(FUERA):
        L.texto(x0 + w / 2, y0 + h - 4.0 - 4.6 * i, s, tam=7.8 if i == 0 else 7.0,
                ha="center", negrita=i == 0, color=p.tinta if i == 0 else p.suave)
    # el ramal: de la 5 a la caja, y de la caja a la 1
    x5, y5, _ = pos[-1]
    ax.add_patch(FancyArrowPatch((x5 + 2.9, y5), (x0, y5), arrowstyle="-|>", mutation_scale=11,
                                 color=p.tinta, linewidth=1.6))
    L.texto((x5 + x0) / 2 + 1.2, y5 + 5.2, "si escribe", tam=6.9, ha="center", color=p.suave)
    L.texto((x5 + x0) / 2 + 1.2, y5 + 2.4, "una orden", tam=6.9, ha="center", color=p.suave)
    x1, y1, _ = pos[0]
    ax.add_patch(FancyArrowPatch((x0 + w / 2, y0), (x1 + 2.2, y1 - 1.6),
                                 connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                                 mutation_scale=11, color=p.tinta, linewidth=1.6))
    L.texto(x0 + w / 2 - 6, y0 - 9.5, "el resultado,", tam=6.9, ha="center", color=p.suave)
    L.texto(x0 + w / 2 - 6, y0 - 12.3, "como texto nuevo", tam=6.9, ha="center", color=p.suave)
    L.guardar(ruta)
    return pos


def selftest():
    fallos = []
    # 1. TEST NULO — una figura de mecanismo no lleva cifras: ningún rótulo tiene un dígito
    #    (los números de las estaciones son su orden, no un dato).
    con_cifras = [s for par in rotulos() for s in par if re.search(r"\d", s)]
    print(f"[1] test nulo         rótulos con cifras: {con_cifras}")
    if con_cifras:
        fallos.append("test nulo: la figura de mecanismo lleva cifras")
    # 2. SEÑAL — las estaciones se llaman como en el capítulo 7, en el mismo orden
    ok = NOMBRES == [n for n, _, _ in ESTACIONES] and len(NOMBRES) == 5
    print(f"[2] señal             estaciones del capítulo 7: {NOMBRES}: {ok}")
    if not ok:
        fallos.append("señal: las estaciones no son las del capítulo 7")
    # 3. INVARIANTE — cinco estaciones a la misma distancia del centro, en el sentido del reloj;
    #    la 5 es la más cercana a la estación de fuera. En color y en gris.
    pos = dibujar(GRIS, "/dev/null")
    dibujar(COLOR, "/dev/null")
    cx, cy = CENTRO
    dist = {round(math.hypot(x - cx, y - cy), 6) for x, y, _ in pos}
    horario = all((pos[k][0] - cx) * (pos[k + 1][1] - cy) - (pos[k][1] - cy) * (pos[k + 1][0] - cx)
                  < 0 for k in range(len(pos) - 1))
    caja = (CAJA[0], CAJA[1] + CAJA[3] / 2)
    cerca = min(range(len(pos)), key=lambda k: math.hypot(pos[k][0] - caja[0],
                                                          pos[k][1] - caja[1])) == len(pos) - 1
    print(f"[3] invariante        misma distancia: {len(dist) == 1}; sentido del reloj: {horario}; "
          f"la 5 junto a la de fuera: {cerca}")
    if len(dist) != 1 or not horario or not cerca:
        fallos.append("invariante: la línea no es la del capítulo 7 o el ramal no sale de la 5")
    print()
    if fallos:
        for x in fallos:
            print("FALLA:", x)
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
    dibujar(GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
