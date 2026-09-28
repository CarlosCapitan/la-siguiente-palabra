#!/usr/bin/env python3
"""
Capítulo 11 — las dos curvas del espejismo: la suave y la del acantilado (L24, D13).

La misma máquina imaginaria de `todo_o_nada.py`, que acierta cada parte suelta de una respuesta
con un tanto por ciento que sube poco a poco. Si se puntúa cada parte, la línea sube derecha. Si
solo cuenta la respuesta con las cinco partes bien, la misma máquina se queda pegada al suelo y
luego se dispara. Por debajo no ha cambiado nada: solo la regla con que se corrige.

Los valores se leen de `datos/salidas/todo_o_nada.csv`.

Uso:
    python figura_todo_o_nada.py --selftest
    python figura_todo_o_nada.py
"""

# ======================= CONSTANTES =======================

CSV = "../datos/salidas/todo_o_nada.csv"
DESTINO = "../figuras/todo_o_nada.png"
ALTO = 3.3
MARCAS = [50, 80]           # dos puntos rotulados en la curva del acantilado

# ==========================================================

import argparse
import csv
import sys

from infografia import COLOR, GRIS, Lienzo
from formato import coma


def leer(ruta=CSV):
    with open(ruta, newline="", encoding="utf-8") as fh:
        f = list(csv.reader(fh))
    assert f[0] == ["cada_parte", "las_cinco"], "cabecera inesperada"
    return [(int(a), float(b)) for a, b in f[1:]]


def dibujar(datos, paleta, ruta):
    L = Lienzo("La misma máquina, corregida de dos maneras",
               "Una respuesta de cinco partes. Por debajo, cada parte se acierta un poco\n"
               "más cada vez; lo único que cambia entre las dos líneas es la regla.",
               paleta, alto=ALTO)
    p = paleta
    arriba = L.y / L.alto_u
    ax = L.fig.add_axes([0.13, 0.15, 0.82, arriba - 0.21])
    x = [a for a, _ in datos]
    ax.plot(x, x, color="0.55", linewidth=1.6, label="corrigiendo cada parte: sube recta")
    ax.plot(x, [b for _, b in datos], color=p.tinta, linewidth=1.9,
            label="corrigiendo todo o nada: pegada al suelo,\ny luego se dispara")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.8, length=2.5)
    ax.set_xlabel("acierto en cada parte suelta (de cada cien)", fontsize=7.4)
    ax.set_ylabel("lo que dice la tabla (de cada cien)", fontsize=7.4)
    ax.legend(loc="upper left", frameon=False, fontsize=6.9, handlelength=2.0)
    marcados = []
    v = dict(datos)
    for m in MARCAS:
        ax.plot([m], [v[m]], marker="o", markersize=3.5, color=p.tinta)
        sitio = {50: (50, 18, "left"), 80: (99, 8, "right")}[m]
        ax.annotate(f"{m} en cada parte:\n{coma(v[m], 0)} las cinco", (m, v[m]),
                    xytext=sitio[:2], ha=sitio[2], fontsize=6.6, color=p.tinta,
                    arrowprops=dict(arrowstyle="-", color=p.suave, linewidth=0.6))
        marcados.append((m, round(v[m])))
    L.guardar(ruta)
    return marcados


def selftest():
    fallos = []
    datos = leer()
    # 1. TEST NULO — si las dos maneras de corregir dieran lo mismo, las líneas coincidirían:
    #    con la diagonal como segunda línea, los puntos marcados están sobre la diagonal.
    m = dibujar([(a, float(a)) for a, _ in datos], GRIS, "/dev/null")
    print(f"[1] test nulo         con una sola parte: {m}")
    if any(a != b for a, b in m):
        fallos.append("test nulo: con una sola parte los puntos no caen en la diagonal")
    # 2. SEÑAL — los dos puntos rotulados son los de la tabla del libro (3 y 33).
    m = dibujar(datos, GRIS, "/dev/null")
    print(f"[2] señal             marcados: {m}")
    if m != [(50, 3), (80, 33)]:
        fallos.append("señal: los puntos marcados no son los de la tabla")
    # 3. INVARIANTE — la curva de todo o nada nunca pasa por encima de la diagonal.
    for pal in (COLOR, GRIS):
        dibujar(datos, pal, "/dev/null")
    ok = all(b <= a + 1e-9 for a, b in datos)
    print(f"[3] invariante        todo o nada siempre por debajo: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: la curva de todo o nada pasa por encima de la diagonal")
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
    dibujar(leer(), GRIS, DESTINO)
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
