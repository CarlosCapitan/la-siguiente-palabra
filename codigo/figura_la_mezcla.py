#!/usr/bin/env python3
"""
Capítulo 8 — la mezcla, dibujada (L24).

La «e» de «quiero acordarme» se lleva una mezcla de los contenidos de las letras que tiene detrás, en
la proporción del reparto. Cada fila es una letra: en gris claro, los tres primeros números de su
contenido; en oscuro, esos mismos números multiplicados por su porción. La fila de abajo es la suma
de los oscuros: la mezcla. Se ve que la «m», con 42 de cada cien, pone casi toda la mezcla, y que el
contenido grande de la «a» se queda en poco porque su porción es pequeña.

Los números NO se escriben aquí: se leen del bloque 9 de `datos/salidas/la_e_de_acordarme.txt`.
En gris: el libro se imprime en negro.

Uso:
    python figura_la_mezcla.py --selftest
    python figura_la_mezcla.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/la_e_de_acordarme.txt"
DESTINO = "../figuras/la_mezcla.png"
ALTO = 4.3                   # pulgadas
ALTO_FILA = 7.0              # unidades del lienzo por fila
ESCALA = 3.2                 # unidades del lienzo por cada 1 de valor

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

from matplotlib.patches import Rectangle

from formato import leer_tablas
from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent
NUM = r"(-?\d+,\d+)"


def leer(ruta):
    """Del bloque 9: [(letra, porción, contenido[3], por su porción[3])], la fila de las otras y la
    mezcla, de su tabla editorial (L24, 9 de octubre)."""
    tablas = leer_tablas(Path(ruta).read_text(encoding="utf-8"))
    titulo = "La mezcla: cada contenido, por su porción, y todo sumado"
    assert titulo in tablas, f"se esperaba el bloque 9 en {ruta}"
    f = lambda v: float(v.replace(",", "."))
    filas, otras, mezcla = [], None, None
    for c in tablas[titulo][1]:
        if re.match(r"^\d+ \S$", c[0]):
            filas.append((c[0], f(c[1].replace(" %", "")), [f(x) for x in c[2:5]], [f(x) for x in c[5:8]]))
        elif c[0].startswith("otras "):
            otras = (c[0], f(c[1].replace(" %", "")), None, [f(x) for x in c[5:8]])
        elif c[0] == "la mezcla (la suma)":
            mezcla = [f(x) for x in c[5:8]]
    assert len(filas) >= 2 and otras and mezcla, f"se esperaban varias letras en la mezcla; hay {len(filas)}"
    for j in range(3):
        suma = sum(r[3][j] for r in filas) + otras[3][j]
        assert abs(suma - mezcla[j]) <= 0.005 * (len(filas) + 2), \
            f"la columna {j + 1} no suma la mezcla: {suma} frente a {mezcla[j]}"
    return filas, otras, mezcla


def barras(L, x0, y0, vals, color, ancho):
    for j, v in enumerate(vals):
        x = x0 + j * 7.0
        L.ax.add_patch(Rectangle((x, min(y0, y0 + v * ESCALA)), ancho, abs(v * ESCALA),
                                 facecolor=color, edgecolor=L.p.tinta, linewidth=0.35))


def dibujar(filas, otras, mezcla, paleta, ruta):
    L = Lienzo("La mezcla de la «e»",
               "En claro, los tres primeros números del contenido de cada letra. En oscuro,\n"
               "esos números por la porción de la letra. Abajo, la suma de los oscuros.",
               paleta, alto=ALTO)
    p, ax = L.p, L.ax
    y = L.y - 4.0
    L.texto(34, y, "contenido", ha="center", tam=7.4, negrita=True)
    L.texto(70, y, "por su porción", ha="center", tam=7.4, negrita=True)
    y -= 5.0
    dibujadas = 0
    todas = filas + [otras]
    for nombre, porcion, cont, por in todas:
        L.texto(4, y, f"{nombre.replace(' ', ' «', 1)}»" if not nombre.startswith("otras") else nombre,
                tam=7.4, negrita=True)
        L.texto(14.5, y, f"{porcion:.1f} %".replace(".", ","), tam=7.0, color=p.suave)
        ax.plot([24, 46], [y, y], color=p.marco, linewidth=0.5)
        ax.plot([60, 82], [y, y], color=p.marco, linewidth=0.5)
        if cont:
            barras(L, 25, y, cont, p.neutro, 5.0)
        barras(L, 61, y, por, p.acento, 5.0)
        dibujadas += 1
        y -= ALTO_FILA
    ax.plot([58, 84], [y + ALTO_FILA * 0.8, y + ALTO_FILA * 0.8], color=p.tinta, linewidth=0.8)
    y -= 1.5
    L.texto(4, y, "la mezcla", tam=7.6, negrita=True)
    ax.plot([60, 82], [y, y], color=p.marco, linewidth=0.5)
    barras(L, 61, y, mezcla, p.tinta, 5.0)
    for j, v in enumerate(mezcla):
        L.texto(63.5 + j * 7.0, y + v * ESCALA + (1.4 if v >= 0 else -1.4),
                f"{v:.2f}".replace(".", ","), ha="center", tam=6.2)
    L.pie("Cada barra es un número: hacia arriba si es positivo, hacia abajo si es negativo.\n"
          "Las cifras exactas están en la tabla de la mezcla, en el texto.")
    L.guardar(ruta)
    return dibujadas


def selftest():
    fallos = []
    filas, otras, mezcla = leer(AQUI / SALIDA)
    import os, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("nada\n")
    try:
        leer(fh.name)
        fallos.append("test nulo: leyó una mezcla de un texto que no la tiene")
    except (AssertionError, AttributeError):
        pass
    finally:
        os.unlink(fh.name)
    print("[1] test nulo         un texto sin el bloque 9 no da figura")
    # 2. SEÑAL — la letra con más porción es la que más pone en la mezcla (en el primer número).
    mayor = max(filas, key=lambda r: r[1])
    pone = max(filas, key=lambda r: abs(r[3][0]))
    print(f"[2] señal             más porción: «{mayor[0]}»; más pone en el primer número: «{pone[0]}»")
    if mayor[0] != pone[0]:
        fallos.append("señal: la letra con más porción no es la que más pone")
    # 3. INVARIANTE — cada columna de oscuros suma la mezcla (lo comprueba leer), las porciones
    #    suman cien, y se dibuja en las dos paletas.
    suma = sum(r[1] for r in filas) + otras[1]
    for pal in (COLOR, GRIS):
        n = dibujar(filas, otras, mezcla, pal, "/dev/null")
    print(f"[3] invariante        porciones que suman {suma:.1f}; filas dibujadas {n}")
    if abs(suma - 100) > 0.1 * (len(filas) + 1):
        fallos.append("invariante: las porciones no suman cien")
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
