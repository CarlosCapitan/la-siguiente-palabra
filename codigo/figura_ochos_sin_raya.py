#!/usr/bin/env python3
"""
Capítulo 2 — los dibujos que impiden la raya del ocho.

`el_ocho_y_la_raya.py` comprueba que, con los 1.257 dibujos con los que aprende el comité, no
existe ninguna raya que deje los ochos a un lado y los demás dígitos al otro, y que quitando unos
pocos dibujos concretos sí existe. Esta figura enseña esos dibujos, tal cual, con lo que son.

Los dibujos no se eligen aquí: se leen de `datos/salidas/el_ocho_y_la_raya.txt`. Si la salida
cambia, la figura cambia.

Uso:
    python figura_ochos_sin_raya.py --selftest
    python figura_ochos_sin_raya.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/el_ocho_y_la_raya.txt"
DESTINO = "../figuras/ochos_sin_raya.png"
POR_FILA = 6
LADO = 1.8          # lado de un punto del dibujo, en unidades del lienzo
TINTA_MAXIMA = 16   # el conjunto de dígitos va de 0 (papel) a 16 (tinta)

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from infografia import COLOR, GRIS, Lienzo
from perceptron import cargar_digitos

AQUI = Path(__file__).resolve().parent


def leer_estorban(ruta):
    """(índice, dígito) de cada dibujo de la tabla «LOS DIBUJOS QUE IMPIDEN LA RAYA DEL OCHO», y
    las cuentas de dibujos que dice la salida."""
    texto = Path(ruta).read_text(encoding="utf-8")
    assert "LOS DIBUJOS QUE IMPIDEN LA RAYA DEL OCHO" in texto, \
        f"se esperaba la tabla de los dibujos que impiden la raya en {ruta}"
    tramo = texto.split("LOS DIBUJOS QUE IMPIDEN LA RAYA DEL OCHO", 1)[1]
    filas = [(int(i), int(d)) for i, d in re.findall(r"^\s+(\d+)\s+un (\d)\s*$", tramo, re.M)]
    assert filas, f"se esperaba al menos un dibujo en la tabla de {ruta}"
    return filas


def dibujar(filas, paleta, ruta):
    from matplotlib.patches import Rectangle
    X, t = cargar_digitos()
    n = len(filas)
    filas_de_dibujos = (n + POR_FILA - 1) // POR_FILA
    alto = (24 + filas_de_dibujos * (8 * LADO + 9)) * 4.45 / 100   # unidades del lienzo a pulgadas
    ochos = sum(1 for _, d in filas if d == 8)
    quienes = ("Todos son ochos." if ochos == n else
               f"{ochos} son ochos y {n - ochos}, otros dígitos.")
    L = Lienzo("Los dibujos que impiden la raya",
               f"Sin estos {n}, de los 1.257, ya hay una raya que deja los ochos a un\n"
               f"lado y los demás dígitos al otro. {quienes}",
               paleta, alto=alto)
    ancho_dibujo = 8 * LADO
    hueco = (92 - POR_FILA * ancho_dibujo) / (POR_FILA - 1)
    y_tope = L.y - 1.0
    for k, (i, d) in enumerate(filas):
        fila, col = divmod(k, POR_FILA)
        x0 = 4 + col * (ancho_dibujo + hueco)
        y0 = y_tope - fila * (ancho_dibujo + 8.0)
        img = X[i].reshape(8, 8)            # 0 a 1 (cargar_digitos ya divide por 16)
        assert img.min() >= 0 and img.max() <= 1, f"el dibujo {i} no está entre 0 y 1"
        for f in range(8):
            for c in range(8):
                g = 1.0 - img[f, c]
                L.ax.add_patch(Rectangle((x0 + c * LADO, y0 - (f + 1) * LADO), LADO, LADO,
                                         facecolor=(g, g, g), edgecolor=paleta.marco,
                                         linewidth=0.3))
        L.texto(x0 + ancho_dibujo / 2, y0 - ancho_dibujo - 3.0, f"es un {d}", tam=7.6, ha="center",
                negrita=(d == 8))
    L.guardar(ruta)


def selftest():
    fallos = []
    filas = leer_estorban(AQUI / SALIDA)
    X, t = cargar_digitos()
    # 1. TEST NULO: un texto sin la tabla no puede dar dibujos.
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("nada que ver aquí\n")
    try:
        leer_estorban(fh.name)
        fallos.append("test nulo: leyó dibujos de un texto que no tiene la tabla")
    except AssertionError:
        pass
    finally:
        os.unlink(fh.name)
    print("[1] test nulo         un texto sin la tabla no da ningún dibujo")
    # 2. SEÑAL: lo que la salida dice que es cada dibujo coincide con el conjunto de dígitos.
    distintos = [(i, d) for i, d in filas if int(t[i]) != d]
    print(f"[2] señal             {len(filas)} dibujos leídos; etiquetas que no cuadran: {len(distintos)}")
    if distintos:
        fallos.append(f"señal: la salida dice un dígito distinto del conjunto en {distintos}")
    # 3. INVARIANTE: la figura cabe en la página y se dibuja entera, en las dos paletas.
    for p in (COLOR, GRIS):
        dibujar(filas, p, "/dev/null")
    print("[3] invariante        la figura se dibuja en color y en gris sin salirse de la página")
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
    dibujar(leer_estorban(AQUI / SALIDA), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
