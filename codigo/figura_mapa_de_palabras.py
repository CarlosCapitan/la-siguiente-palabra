#!/usr/bin/env python3
"""
Capítulo 5 — el mapa de palabras: la distancia en el papel es el parecido (L24).

Cada palabra del modelo del capítulo es una lista de cien números. Aquí veintidós palabras de los
bloques del capítulo, aplanadas de cien números a dos para poder dibujarlas: cuanto más cerca en
el papel, más se parecen sus listas. Aplanar pierde algo, y la figura lo dice con la cifra que
imprime el programa.

Nada se calcula aquí: las posiciones, los grupos y la cifra de lo que se pierde se leen del
bloque 8 de `datos/salidas/palabras_de_cerca.txt`.

Uso:
    python figura_mapa_de_palabras.py --selftest
    python figura_mapa_de_palabras.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/palabras_de_cerca.txt"
DESTINO = "../figuras/mapa_de_palabras.png"
ALTO = 4.7
DESTACADA = "banco"
MARGEN = 0.06          # en unidades del aplanado, alrededor de cada grupo

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def leer(ruta):
    texto = Path(ruta).read_text(encoding="utf-8")
    assert "--- 8. EL MAPA" in texto, f"se esperaba el bloque 8 en {ruta}"
    b = texto.split("--- 8. EL MAPA", 1)[1].split("\n--- ", 1)[0]
    puntos, nombres = [], {}
    for l in b.splitlines():
        m = re.match(r"^(\S+)\s+(-?\d+,\d+)\s+(-?\d+,\d+)\s+(\S+)\s+(\d+)$", l)
        if m:
            puntos.append((m.group(1), float(m.group(2).replace(",", ".")),
                           float(m.group(3).replace(",", ".")), m.group(4), int(m.group(5))))
        m = re.match(r"^\s+(\d+): (.+)$", l)
        if m:
            nombres[int(m.group(1))] = m.group(2)
    m1 = re.search(r"en el papel: (\d+); y está entre las tres más\s+cercanas en el papel: (\d+)", b)
    assert puntos and nombres and m1, "no se pudo leer el bloque 8"
    assert {p[4] for p in puntos} == set(nombres), "grupos sin nombre"
    return puntos, nombres, int(m1.group(1)), int(m1.group(2))


# Una forma por grupo, bien distinta en gris: rellena o hueca, y seis siluetas.
FORMAS = [("o", False), ("s", True), ("^", True), ("D", False), ("P", True), ("X", True),
          ("*", True), ("v", False)]


def colocar_rotulos(pos, anchos, alto=2.6):
    """Para cada punto, dónde va su rótulo: el primero de una lista de sitios alrededor del punto,
    cada vez más lejos, que no pise otro rótulo ni otro punto. Sin azar. Devuelve el centro
    izquierdo del rótulo y si hace falta una raya que lo una a su punto."""
    ocupados = [(x - 1.2, y - 1.2, x + 1.2, y + 1.2) for x, y in pos.values()]
    sitio = {}
    for pal, (x, y) in sorted(pos.items(), key=lambda kv: (-kv[1][1], kv[1][0])):
        w = anchos[pal]
        candidatos = []
        for d in (1.6, 3.2, 5.0, 7.0):
            candidatos += [(x + d, y - alto / 2), (x - d - w, y - alto / 2),
                           (x - w / 2, y + d * 0.8), (x - w / 2, y - d * 0.8 - alto),
                           (x + d * 0.7, y + d * 0.7), (x + d * 0.7, y - d * 0.7 - alto),
                           (x - d * 0.7 - w, y + d * 0.7), (x - d * 0.7 - w, y - d * 0.7 - alto)]
        elegido, lejos = candidatos[0], False
        for k, (cx, cy) in enumerate(candidatos):
            caja = (cx - 0.3, cy - 0.2, cx + w + 0.3, cy + alto + 0.2)
            if all(caja[2] <= o[0] or caja[0] >= o[2] or caja[3] <= o[1] or caja[1] >= o[3]
                   for o in ocupados):
                elegido, lejos = (cx, cy), k >= 8
                break
        ocupados.append((elegido[0], elegido[1], elegido[0] + w, elegido[1] + alto))
        sitio[pal] = (elegido[0], elegido[1] + alto / 2, lejos, w)
    return sitio


def marca(ax, x, y, g, p, grande=False, tam=5.2):
    forma, llena = FORMAS[(g - 1) % len(FORMAS)]
    ax.plot(x, y, forma, ms=(tam + 3.5) if grande else tam,
            markerfacecolor=(p.tinta if grande else p.suave) if llena else "white",
            markeredgecolor=p.tinta, markeredgewidth=0.8, zorder=3)


def dibujar(puntos, nombres, igual1, igual3, paleta, ruta):
    L = Lienzo("Un mapa de palabras",
               "Cada palabra es una lista de cien números. Cuanto más cerca en el\n"
               "papel, más se parecen las dos listas.", paleta, alto=ALTO)
    p, ax = L.p, L.ax
    xs = np.array([q[1] for q in puntos]); ys = np.array([q[2] for q in puntos])
    x0, x1, y0, y1 = 12, 84, 27, L.y - 3
    esc = min((x1 - x0) / max(np.ptp(xs), 1e-9), (y1 - y0) / max(np.ptp(ys), 1e-9))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X = lambda v: cx + (v - (xs.max() + xs.min()) / 2) * esc
    Y = lambda v: cy + (v - (ys.max() + ys.min()) / 2) * esc
    pos = {pal: (X(x), Y(y)) for pal, x, y, _, _ in puntos}
    anchos = {pal: 1.22 * len(pal) + (1.2 if pal == DESTACADA else 0) for pal in pos}
    sitio = colocar_rotulos(pos, anchos)
    for pal, x, y, _, g in puntos:
        d = pal == DESTACADA
        marca(ax, *pos[pal], g, p, grande=d)
        tx, ty, lejos, w = sitio[pal]
        if lejos:
            px, py = pos[pal]
            ax.plot([px, min(max(px, tx), tx + w)], [py, ty], color=p.suave, linewidth=0.5, zorder=2)
        L.texto(tx, ty, pal, tam=8.2 if d else 7.4, negrita=d)
    yk = 22.0
    L.texto(4, yk, "Forma del punto: de qué bloque del capítulo sale la palabra", tam=7.0,
            color=p.suave)
    for k, (g, nombre) in enumerate(sorted(nombres.items())):
        col, fila = k % 3, k // 3
        xk, yy = 5.5 + col * 31, yk - 3.6 - fila * 3.3
        marca(ax, xk, yy, g, p, grande=False)
        L.texto(xk + 1.8, yy, nombre, tam=7.0)
    L.pie(f"Aplanar cien números a dos pierde algo. De estas {len(puntos)} palabras, "
          f"la más parecida a cada una\nes también la más cercana en el papel en {igual1}, "
          f"y está entre sus tres más cercanas en {igual3}.")
    L.guardar(ruta)
    return pos


def selftest():
    fallos = []
    puntos, nombres, i1, i3 = leer(AQUI / SALIDA)
    # 1. TEST NULO — si todas las palabras estuvieran en el mismo sitio, la figura no puede
    #    inventarse distancias: todas caen en el mismo punto del papel.
    iguales = [(p, 0.1, 0.1, v, g) for p, _, _, v, g in puntos]
    try:
        pos = dibujar(iguales, nombres, i1, i3, GRIS, "/dev/null")
        juntos = len({(round(a, 6), round(b, 6)) for a, b in pos.values()}) == 1
    except (ZeroDivisionError, FloatingPointError):
        juntos = True
    print(f"[1] test nulo         palabras en el mismo sitio -> un solo punto: {juntos}")
    if not juntos:
        fallos.append("test nulo: separa en el papel palabras que están en el mismo sitio")
    # 2. SEÑAL — la palabra más cercana en el papel a «banco» es la que dice la salida.
    pos = dibujar(puntos, nombres, i1, i3, GRIS, "/dev/null")
    b = pos[DESTACADA]
    cerca = min((np.hypot(v[0] - b[0], v[1] - b[1]), k) for k, v in pos.items() if k != DESTACADA)[1]
    mas_cerca_salida = min(((q[1] - [r for r in puntos if r[0] == DESTACADA][0][1]) ** 2 +
                            (q[2] - [r for r in puntos if r[0] == DESTACADA][0][2]) ** 2, q[0])
                           for q in puntos if q[0] != DESTACADA)[1]
    print(f"[2] señal             más cerca de «{DESTACADA}» en la figura: «{cerca}»; en la salida: «{mas_cerca_salida}»")
    if cerca != mas_cerca_salida:
        fallos.append("señal: la figura no respeta las distancias de la salida")
    # 3. INVARIANTE — la escala es la misma en los dos ejes (si no, el mapa deformaría las
    #    distancias), y se dibuja en color y en gris.
    xs = [q[1] for q in puntos]; ys = [q[2] for q in puntos]
    a, bb = puntos[0], puntos[1]
    d_sal = np.hypot(a[1] - bb[1], a[2] - bb[2]); d_fig = np.hypot(pos[a[0]][0] - pos[bb[0]][0], pos[a[0]][1] - pos[bb[0]][1])
    c, d = puntos[2], puntos[3]
    d_sal2 = np.hypot(c[1] - d[1], c[2] - d[2]); d_fig2 = np.hypot(pos[c[0]][0] - pos[d[0]][0], pos[c[0]][1] - pos[d[0]][1])
    ok = abs(d_fig / d_sal - d_fig2 / d_sal2) < 1e-6
    for pal in (COLOR, GRIS):
        dibujar(puntos, nombres, i1, i3, pal, "/dev/null")
    print(f"[3] invariante        misma escala en los dos ejes: {ok}")
    if not ok:
        fallos.append("invariante: la escala del papel no es la misma para todas las distancias")
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
    dibujar(*leer(AQUI / SALIDA), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
