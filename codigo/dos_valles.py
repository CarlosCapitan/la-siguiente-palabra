#!/usr/bin/env python3
"""
Capítulo 15 — el terreno del capítulo 4 con solo dos números, y dos bolas que bajan (L24, E26).

El capítulo 4 pidió imaginar el error como un terreno: cada combinación de números es un punto, y
la altura es lo mal que lo hace. Aprender es bajar. El capítulo 15 dice que ese terreno «está lleno
de valles» y que no hay garantía de acabar en uno bueno. Aquí se hace con un terreno de juguete de
DOS números, que sí se puede dibujar: tiene un valle hondo y uno poco hondo. Se sueltan dos bolas
en dos sitios de salida y cada una baja siempre cuesta abajo, a pasos cortos, como en el capítulo
4. Cada una acaba en el valle que tiene más cerca, que no siempre es el hondo.

El terreno es inventado (dos números no hacen una máquina de verdad): lo que se mide es solo por
dónde baja cada bola. La figura (`figura_dos_valles.py`) lee de aquí los caminos.

Uso:
    python dos_valles.py --selftest
    python dos_valles.py > ../datos/salidas/dos_valles.txt
"""

# ======================= CONSTANTES =======================

# Cada valle: (centro en el primer número, centro en el segundo, hondura, anchura).
VALLES = ((1.9, 1.5, 3.0, 1.0),      # el hondo
          (-1.9, -1.5, 1.5, 0.9))    # el poco hondo
CUENCO = 0.06                        # un cuenco suave alrededor, para que todo baje hacia dentro
BASE = 3.0                           # se suma a todo, para que ninguna altura sea negativa
SALIDAS = ((-0.3, -3.3), (-0.2, 3.2)) # dónde se sueltan las dos bolas
PASO = 0.05                          # cuánto se mueve la bola en cada paso (como en el capítulo 4)
PASOS = 4000                         # cuántos pasos da, como mucho
QUIETA = 1e-6                        # si en un paso baja menos que esto, se da por parada
LIMITE = 4.0                         # el trozo de terreno que se dibuja: de -4 a 4 en cada número
SALIDA_CSV = "../datos/salidas/dos_valles.csv"

# ==========================================================

import argparse
import csv
import math
import sys
from pathlib import Path

from formato import ANCHO_CAJA_CITA, coma, comprobar_ancho

AQUI = Path(__file__).resolve().parent


def altura(x, y, valles=VALLES, cuenco=CUENCO):
    h = BASE + cuenco * (x * x + y * y)
    for cx, cy, hondo, ancho in valles:
        h -= hondo * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * ancho ** 2))
    return h


def pendiente(x, y, valles=VALLES, cuenco=CUENCO, e=1e-5):
    return ((altura(x + e, y, valles, cuenco) - altura(x - e, y, valles, cuenco)) / (2 * e),
            (altura(x, y + e, valles, cuenco) - altura(x, y - e, valles, cuenco)) / (2 * e))


def bajar(x, y, valles=VALLES, cuenco=CUENCO):
    """Siempre cuesta abajo, a pasos cortos: el paso es la pendiente por PASO."""
    camino = [(x, y, altura(x, y, valles, cuenco))]
    for _ in range(PASOS):
        gx, gy = pendiente(x, y, valles, cuenco)
        x, y = x - PASO * gx, y - PASO * gy
        h = altura(x, y, valles, cuenco)
        camino.append((x, y, h))
        if camino[-2][2] - h < QUIETA:
            break
    return camino


def en_que_valle(x, y):
    d = [math.hypot(x - cx, y - cy) for cx, cy, _, _ in VALLES]
    return d.index(min(d))


def informe():
    nombres = ("el valle hondo", "el valle poco hondo")
    lineas = ["el terreno: dos números, y la altura es lo mal que lo hace.",
              ""]
    filas = []
    for k, (x0, y0) in enumerate(SALIDAS, 1):
        c = bajar(x0, y0)
        x, y, h = c[-1]
        v = en_que_valle(x, y)
        lineas += [f"bola {k}: sale de la altura {coma(c[0][2], 2)}; tras {len(c) - 1} pasos",
                   f"  se para en {nombres[v]}, a la altura {coma(h, 2)}."]
        for i, (a, b, hh) in enumerate(c):
            filas.append([k, i, f"{a:.5f}", f"{b:.5f}", f"{hh:.5f}"])
    lineas += ["",
               "cada bola baja siempre cuesta abajo y acaba en el valle",
               "que tiene más cerca, no en el más hondo."]
    for l in comprobar_ancho(["  " + l if l else l for l in lineas], ANCHO_CAJA_CITA):
        print(l)
    with open(AQUI / SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["bola", "paso", "x", "y", "altura"]] + filas)


def selftest():
    fallos = []
    # 1. TEST NULO — en un terreno llano la bola no se mueve.
    c = bajar(1.0, 1.0, valles=(), cuenco=0.0)
    print(f"[1] test nulo         en terreno llano, pasos dados: {len(c) - 1}")
    if len(c) - 1 > 1 or (c[-1][0], c[-1][1]) != (1.0, 1.0):
        fallos.append("test nulo: la bola se mueve en un terreno llano")
    # 2. SEÑAL IMPLANTADA — con un solo valle, la bola llega a su fondo desde cualquier sitio.
    uno = ((0.5, -0.5, 2.0, 1.2),)
    c = bajar(-2.0, 2.0, valles=uno, cuenco=0.0)
    lejos = math.hypot(c[-1][0] - 0.5, c[-1][1] + 0.5)
    print(f"[2] señal implantada  con un solo valle, acaba a {lejos:.3f} de su fondo")
    if lejos > 0.05:
        fallos.append("señal: la bola no llega al fondo del único valle")
    # 3. INVARIANTE DEL DOMINIO — la altura no sube nunca a lo largo del camino, y las dos bolas
    #    del capítulo acaban en valles distintos.
    caminos = [bajar(*s) for s in SALIDAS]
    baja = all(b[2] <= a[2] + 1e-12 for c in caminos for a, b in zip(c, c[1:]))
    valles = [en_que_valle(c[-1][0], c[-1][1]) for c in caminos]
    print(f"[3] invariante        la altura nunca sube: {'sí' if baja else 'NO'}; "
          f"valles donde acaban: {valles}")
    if not baja or sorted(valles) != [0, 1]:
        fallos.append("invariante: una bola sube, o las dos acaban en el mismo valle")
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
    if ap.parse_args().selftest:
        sys.exit(selftest())
    informe()


if __name__ == "__main__":
    main()
