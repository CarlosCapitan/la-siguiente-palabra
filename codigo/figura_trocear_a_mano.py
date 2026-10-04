#!/usr/bin/env python3
"""
Capítulo 7 — la escalera de «_murciélago» (Carlos, 4 de octubre de 2026).

La misma palabra, renglón a renglón, troceada con 0, 10, 100, 1.000 y 10.000 juntas en la caja
(las de `trocear_a_mano.py`, aprendidas del Quijote). Cada pieza es una caja; donde hay un corte,
hay un hueco. Las juntas se leen de `trocear_a_mano.csv`, y el selftest comprueba que cada renglón
es el que imprime `datos/salidas/trocear_a_mano.txt`.

Uso:
    python figura_trocear_a_mano.py --selftest
    python figura_trocear_a_mano.py
"""

# ======================= CONSTANTES =======================

CSV_JUNTAS = "trocear_a_mano.csv"
SALIDA = "../datos/salidas/trocear_a_mano.txt"
DESTINO = "../figuras/trocear_a_mano.png"
PALABRA = " murciélago"
ALTO = 2.5                   # pulgadas
X0, CELDA, HUECO = 30.0, 5.9, 0.9    # en unidades del lienzo (100 de ancho)
FILA = 6.6

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

from matplotlib.patches import FancyBboxPatch

from infografia import COLOR, GRIS, Lienzo
from trocear_a_mano import trocear, leer, a_la_vista, miles, MOMENTOS

AQUI = Path(__file__).resolve().parent


def renglones(juntas):
    return [(m, trocear(PALABRA, juntas[:m])) for m in MOMENTOS]


def de_la_salida(ruta):
    """Los renglones de «_murciélago» tal como los imprimió trocear_a_mano.py."""
    texto = Path(ruta).read_text(encoding="utf-8")
    bloque = texto.split("\n" + a_la_vista(PALABRA) + "\n", 1)[1].split("\n\n", 1)[0]
    fuera = []
    for l in bloque.splitlines():
        m = re.match(r"^\s*([\d.]+)\s{3}(.*)$", l)
        assert m, f"no entiendo el renglón {l!r} de {ruta}"
        fuera.append((int(m.group(1).replace(".", "")), m.group(2).split(" | ")))
    assert len(fuera) == len(MOMENTOS), f"se esperaban {len(MOMENTOS)} renglones; hay {len(fuera)}"
    return fuera


def dibujar(filas, paleta, ruta):
    L = Lienzo("Cómo se corta una palabra, según la caja",
               "Cómo quedaría cortada «_murciélago» con una caja de 0, 10, 100, 1.000 y 10.000\n"
               "juntas, aprendidas del Quijote. Cada recuadro es un trozo; cada hueco, un corte.",
               paleta, alto=ALTO)
    p, ax = L.p, L.ax
    y = L.y - 3.2
    for m, trozos in filas:
        ax.text(4, y, f"{miles(m)} juntas" if m != 1 else "1 junta", ha="left", va="center",
                fontsize=8.4, color=p.tinta)
        x = X0
        for t in trozos:
            ancho = CELDA * len(t)
            ax.add_patch(FancyBboxPatch((x + HUECO / 2, y - 2.3), ancho - HUECO, 4.6,
                                        boxstyle="round,pad=0,rounding_size=0.7",
                                        facecolor=p.fondo, edgecolor=p.tinta, linewidth=0.9))
            for k, ch in enumerate(a_la_vista(t)):
                ax.text(x + CELDA * (k + 0.5), y, ch, ha="center", va="center", fontsize=9.6,
                        color=p.tinta, family="DejaVu Sans Mono")
            x += ancho
        assert x <= 97, f"la palabra se sale del lienzo: llega a {x:.1f}"
        y -= FILA
    assert y > 0, "los renglones se salen por abajo: sube ALTO"
    L.guardar(ruta)


def selftest():
    fallos = []
    juntas = leer(AQUI / CSV_JUNTAS)
    filas = renglones(juntas)
    largo = len(PALABRA)

    # [1] TEST NULO — sin juntas en la caja, cada letra es un trozo: tantos cortes como huecos
    #     entre letras.
    cero = filas[0][1]
    print(f"[1] test nulo         0 juntas: {len(cero)} trozos para {largo} letras")
    if len(cero) != largo:
        fallos.append("test nulo: sin juntas, la palabra no sale letra a letra")

    # [2] SEÑAL — los renglones dibujados son los que imprimió trocear_a_mano.py.
    salida = de_la_salida(AQUI / SALIDA)
    iguales = [(m, [a_la_vista(t) for t in s]) for m, s in filas] == salida
    print(f"[2] señal             los cinco renglones, como en la salida: {iguales}")
    if not iguales:
        fallos.append("señal: lo dibujado no es lo que imprimió trocear_a_mano.py")

    # [3] INVARIANTE — cada renglón pega la palabra exacta y no tiene más cortes que el de arriba;
    #     y la figura se dibuja en las dos paletas.
    pega = all("".join(s) == PALABRA for _, s in filas)
    baja = all(len(a[1]) >= len(b[1]) for a, b in zip(filas, filas[1:]))
    for pal in (COLOR, GRIS):
        dibujar(filas, pal, "/dev/null")
    print(f"[3] invariante        cada renglón es la palabra entera: {pega}; "
          f"los cortes nunca aumentan: {baja}")
    if not (pega and baja):
        fallos.append("invariante: un renglón rompe la palabra o gana cortes")
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
    dibujar(renglones(leer(AQUI / CSV_JUNTAS)), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
