#!/usr/bin/env python3
"""
Capítulo 14 — qué cifras salen bien cuando la máquina multiplica sin calculadora.

Figura de medición. Una fila por cada una de las 40 respuestas del modelo de 32.000 millones sin
herramienta (20 con la pregunta libre y 20 con restricción), una casilla por cifra, alineadas por
la derecha como en una cuenta escrita. Gris: la cifra es la misma que en el producto exacto, en
el mismo sitio. Blanco: es otra. Las respuestas de siete cifras dejan vacía la primera columna.

No lleva ninguna cifra dibujada: lo que enseña es la forma (los extremos, grises; el medio,
blanco). Los recuentos que la acompañan los imprime `ordenes_a_la_vista.py` en la tabla «Qué
cifras salen bien sin calculadora…», y el selftest comprueba que la figura y esa tabla cuentan lo
mismo.

Auditoría del capítulo 14, 10 de octubre de 2026 (F2, P1).

Uso (desde codigo/):
    python figura_cifra_a_cifra.py --selftest
    python figura_cifra_a_cifra.py
"""

# ======================= CONSTANTES =======================

DESTINO = "../figuras/cifra_a_cifra.png"
TABLA = "../datos/salidas/ordenes_a_la_vista.txt"
TITULO_TABLA = "Qué cifras salen bien sin calculadora, de izquierda a derecha (32.000M)"
ALTO = 5.4
COLUMNAS = 8                # cifras del producto más largo
ANCHO_CASILLA = 4.5         # en unidades del lienzo (100 de ancho)
ALTO_FILA = 2.0
HUECO_BLOQUES = 2.6         # entre las 20 con restricción y las 20 libres
X0 = 30.0                   # borde izquierdo de la rejilla
SEMILLA = 20261010          # para el test nulo

# ==========================================================

import argparse
import random
import re
import sys

from matplotlib.patches import Rectangle

from infografia import COLOR, GRIS, Lienzo
from ordenes_a_la_vista import GRANDE, SIN, leer

BLOQUES = [("con restricción", SIN[0]), ("libre", SIN[1])]


def filas(fs):
    """Por bloque, una lista por respuesta: COLUMNAS casillas alineadas por la derecha, cada una
    True (misma cifra), False (otra) o None (no hay cifra)."""
    out = []
    for _, cond in BLOQUES:
        bloque = []
        for f in fs:
            if f["modelo"] != GRANDE or f["condicion"] != cond:
                continue
            p, e = f["producto"], f["respuesta_numero"]
            assert len(p) == len(e), (f"Se esperaba la misma cantidad de cifras en {p} y {e}; "
                                      "la figura no sabe alinear números de largo distinto")
            assert len(p) <= COLUMNAS, f"Se esperaban como mucho {COLUMNAS} cifras; {p} tiene más"
            bloque.append([None] * (COLUMNAS - len(p)) + [x == y for x, y in zip(p, e)])
        out.append(bloque)
    return out


def por_columna_ocho(bloques):
    """Aciertos por columna en las respuestas de ocho cifras: lo que cuenta la tabla."""
    ocho = [r for b in bloques for r in b if r[0] is not None]
    return [sum(r[i] for r in ocho) for i in range(COLUMNAS)], len(ocho)


def dibujar(bloques, paleta, ruta):
    L = Lienzo("Qué cifras salen bien sin calculadora",
               "Cada fila es una de las cuarenta respuestas del modelo de 32.000M sin\n"
               "calculadora; cada casilla, una cifra, alineadas por la derecha como en una\n"
               "cuenta escrita. Gris: la misma cifra que en el producto exacto.",
               paleta, alto=ALTO)
    p, ax = L.p, L.ax
    ancho = COLUMNAS * ANCHO_CASILLA
    y = L.y - 2.0
    L.texto(X0, y, "las primeras cifras", tam=7.0, color=p.suave)
    L.texto(X0 + ancho, y, "las últimas", tam=7.0, color=p.suave, ha="right")
    y -= 3.4
    casillas = 0
    for (nombre, _), bloque in zip(BLOQUES, bloques):
        tope = y
        for fila in bloque:
            for c, bien in enumerate(fila):
                if bien is None:
                    continue
                ax.add_patch(Rectangle((X0 + c * ANCHO_CASILLA, y - ALTO_FILA),
                                       ANCHO_CASILLA, ALTO_FILA,
                                       facecolor=p.contra if bien else p.papel,
                                       edgecolor=p.marco, linewidth=0.4))
                casillas += 1
            y -= ALTO_FILA
        L.texto(X0 - 3.0, (tope + y) / 2, nombre, tam=7.4, ha="right")
        y -= HUECO_BLOQUES
    # la clave, a la derecha
    xc, yc = X0 + ancho + 6.0, L.y - 10.0
    for k, (relleno, s) in enumerate([(p.contra, "la misma cifra"), (p.papel, "otra cifra")]):
        ax.add_patch(Rectangle((xc, yc - 6.0 * k - ALTO_FILA / 2), ANCHO_CASILLA, ALTO_FILA,
                               facecolor=relleno, edgecolor=p.marco, linewidth=0.4))
        L.texto(xc + ANCHO_CASILLA + 2.0, yc - 6.0 * k, s, tam=7.0, color=p.suave)
    assert y > 8.0, f"la rejilla baja hasta {y:.1f} y pisa el pie"
    L.pie("Cada respuesta, comparada cifra a cifra con el producto exacto. Los datos salen de\n"
          "datos/salidas/ordenes_escritas.csv.")
    L.guardar(ruta)
    return casillas


def tabla_impresa(ruta=TABLA):
    """Los aciertos por posición, y de cuántas, tal como los imprime ordenes_a_la_vista.py."""
    t = open(ruta, encoding="utf-8").read()
    i = t.find(TITULO_TABLA)
    assert i >= 0, f"Se esperaba la tabla «{TITULO_TABLA}» en {ruta}; no está"
    bloque = t[i:t.index(":::", i)]
    de = int(re.search(r"\| la cifra \| bien, de (\d+) \|", bloque).group(1))
    cuentas = [int(x) for x in re.findall(r"^\| [a-zé]+ \| (\d+) \|$", bloque, re.M)]
    assert len(cuentas) == COLUMNAS, f"Se esperaban {COLUMNAS} filas en la tabla; hay {len(cuentas)}"
    return cuentas, de


def selftest():
    ok = True
    fs = leer()
    # [1] test nulo: con respuestas al azar del mismo largo, la forma desaparece (ninguna columna
    #     pasa de la mitad de las de ocho cifras)
    rnd = random.Random(SEMILLA)
    azar = [dict(f, respuesta_numero="".join(rnd.choice("123456789") if k == 0 else
                                             rnd.choice("0123456789")
                                             for k in range(len(f["producto"]))))
            for f in fs]
    cols, n = por_columna_ocho(filas(azar))
    p1 = max(cols) <= n // 2
    ok &= p1
    print(f"[1] test nulo         respuestas al azar, aciertos por columna: {cols} de {n}: "
          f"{'bien' if p1 else 'MAL'}")
    # [2] señal implantada: con el producto exacto como respuesta, todas las casillas grises
    exactas = [dict(f, respuesta_numero=f["producto"]) for f in fs]
    cols2, n2 = por_columna_ocho(filas(exactas))
    p2 = cols2 == [n2] * COLUMNAS
    ok &= p2
    print(f"[2] señal implantada  respuestas exactas: {cols2} de {n2}: {'bien' if p2 else 'MAL'}")
    # [3] invariante: la figura cuenta lo mismo que la tabla impresa, y dibuja una casilla por
    #     cifra de las 40 respuestas, en color y en gris
    bloques = filas(fs)
    cols3, n3 = por_columna_ocho(bloques)
    tabla, de = tabla_impresa()
    cifras = sum(len(f["producto"]) for f in fs if f["modelo"] == GRANDE and f["condicion"] in SIN)
    dib = [dibujar(bloques, GRIS, "/dev/null"), dibujar(bloques, COLOR, "/dev/null")]
    p3 = cols3 == tabla and n3 == de and dib == [cifras, cifras]
    ok &= p3
    print(f"[3] invariante        figura {cols3} de {n3}; tabla {tabla} de {de}; casillas "
          f"dibujadas {dib[0]} de {cifras} cifras: {'bien' if p3 else 'MAL'}")
    print()
    print("SELFTEST: las tres pruebas pasan." if ok else "SELFTEST: FALLA.")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if not selftest():
        sys.exit(1)
    if args.selftest:
        sys.exit(0)
    dibujar(filas(leer()), GRIS, DESTINO)
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
