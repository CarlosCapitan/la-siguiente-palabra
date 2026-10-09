#!/usr/bin/env python3
"""
Capítulos 8 y 9 — dos cuentas que el libro afirma y que aquí se hacen casilla a casilla (L24).

1. «El trabajo crece con el cuadrado de la longitud del texto» (capítulo 8, «Lo que cuesta»).
   Cada palabra compara su pregunta con la etiqueta de ella misma y de todas las ANTERIORES (estas
   máquinas solo miran hacia atrás). Se dibuja la cuadrícula de una frase de cuatro palabras y de
   otra de ocho, se cuentan las casillas, y se cuentan también para textos más largos.
2. «Cuántos pasos hay que dar esperando al anterior» (capítulo 9). Se construye, para una frase
   corta, de qué depende cada cuenta en la máquina que lee en orden (el resumen de cada palabra
   necesita el de la anterior, capítulo 6) y en la que mira todo a la vez (la mirada de cada
   palabra solo necesita las listas de las palabras), y se cuenta la cadena más larga de cuentas
   que tienen que esperarse una a otra.

Nada de esto es una medición de tiempo: sale del diseño y da lo mismo en cualquier ordenador.
Las cuadrículas y las cadenas se CONSTRUYEN (no se escribe la fórmula del resultado) y el
selftest comprueba que salen lo que dice la cuenta.

Uso:
    python cuadricula_y_fila.py --selftest
    python cuadricula_y_fila.py > ../datos/salidas/cuadricula_y_fila.txt
"""

# ======================= CONSTANTES =======================

FRASE = "El vaso no cabía en el cajón porque"        # la de la cuadrícula: sus 4 y sus 8 primeras
CORTAS = [4, 8]                                      # las dos cuadrículas que se dibujan
LONGITUDES = [4, 8, 16, 32, 64, 128, 256, 512, 1024] # las que se cuentan
FRASE_FILA = "el perro mordió al hombre"             # la de los pasos en fila
LONGITUDES_FILA = [64, 1024]                         # las de la tabla del capítulo 9

# ==========================================================

import argparse
import sys

from formato import comprobar_ancho, miles, coma, tabla_editorial

ANCHO = 64


def cuadricula(n):
    """Qué casillas se comparan: fila = la palabra que pregunta, columna = la que tiene la etiqueta.
    Se compara con ella misma y con las de antes; nunca con las que vienen (no existen todavía)."""
    return [[columna <= fila for columna in range(n)] for fila in range(n)]


def casillas(n, solo_atras=True):
    c = cuadricula(n)
    return sum(sum(1 for v in fila if v or not solo_atras) for fila in c)


def dependencias(n, en_orden):
    """De qué cuentas depende cada cuenta. Cada palabra tiene su lista (lista i, que ya está) y su
    cuenta (cuenta i). En orden: la cuenta i necesita la lista i y la cuenta i-1 (el resumen de antes).
    A la vez: la cuenta i necesita las listas de 0..i, y ninguna otra cuenta."""
    dep = {}
    for i in range(n):
        if en_orden:
            dep[i] = [i - 1] if i > 0 else []
        else:
            dep[i] = []
    return dep


def pasos_en_fila(dep):
    """La cadena más larga de cuentas que se esperan una a otra."""
    hondo = {}
    for i in sorted(dep):
        hondo[i] = 1 + max((hondo[j] for j in dep[i]), default=0)
    return max(hondo.values())


def selftest():
    fallos = []
    # 1. TEST NULO — una sola palabra: una casilla y un paso, en las dos máquinas.
    nulo = (casillas(1), pasos_en_fila(dependencias(1, True)), pasos_en_fila(dependencias(1, False)))
    print(f"[1] test nulo         una sola palabra: casillas {nulo[0]}, pasos en fila {nulo[1]} y {nulo[2]}")
    if nulo != (1, 1, 1):
        fallos.append("test nulo: con una palabra no sale una casilla y un paso")
    # 2. SEÑAL — las cuentas hechas a mano: 4 palabras, 1+2+3+4 = 10 casillas; 8 palabras, 36; y la
    #    que lee en orden espera tantos pasos como palabras.
    senal = (casillas(4), casillas(8), pasos_en_fila(dependencias(5, True)))
    print(f"[2] señal             casillas con 4 y con 8 palabras: {senal[0]} y {senal[1]}; "
          f"pasos en orden con 5: {senal[2]}")
    if senal != (10, 36, 5):
        fallos.append("señal: las cuentas no dan 10, 36 y 5")
    # 3. INVARIANTE — ninguna palabra mira hacia delante (la casilla de arriba a la derecha está
    #    vacía), y al doblar un texto largo las casillas se multiplican casi por cuatro.
    c = cuadricula(8)
    delante = any(c[f][k] for f in range(8) for k in range(f + 1, 8))
    razon = casillas(1024) / casillas(512)
    print(f"[3] invariante        mira hacia delante: {'sí' if delante else 'no'}; "
          f"de 512 a 1.024 palabras las casillas se multiplican por {coma(razon, 2)}")
    if delante or not 3.9 < razon < 4.0:
        fallos.append("invariante: mira hacia delante, o doblar no multiplica casi por cuatro")
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
    print("--- selftest ---")
    codigo = selftest()
    if codigo or args.selftest:
        return codigo
    palabras = FRASE.split()
    assert len(palabras) >= max(CORTAS)
    L = ["", "########## capítulos 8 y 9: la cuadrícula y la fila ##########",
         "No es una medición: sale del diseño y da lo mismo en cualquier",
         "ordenador.", ""]
    L += ["1. CADA PALABRA, CON ELLA MISMA Y CON LAS DE ANTES",
          "   (fila: la palabra que pregunta; columna: la de la etiqueta;",
          "   «x»: se comparan; «·»: no, porque esa aún no se ha escrito)", ""]
    for n in CORTAS:
        pal = palabras[:n]
        w = max(len(f"{k + 1} {p}") for k, p in enumerate(pal)) + 1
        L.append(f"  {'':<{w}}" + "".join(f"{str(k + 1):>3}" for k in range(n)))
        for f, fila in enumerate(cuadricula(n)):
            L.append(f"  {str(f + 1) + ' ' + pal[f]:<{w}}" + "".join(f"{'x' if v else '·':>3}" for v in fila))
        L += [f"  {n} palabras: {casillas(n)} casillas de {n * n}.", ""]
    # L24 (9 de octubre): el apartado 2, en tabla editorial (regla 6 ter). El 1 se queda como
    # estaba: lo lee la figura de la cuadrícula.
    filas, antes = [], None
    for n in LONGITUDES:
        v = casillas(n)
        filas.append([miles(n), miles(v), coma(v / antes, 2) if antes else ""])
        antes = v
    L += ["2. CUÁNTAS CASILLAS, SEGÚN LO LARGO QUE SEA EL TEXTO", ""] + tabla_editorial(
        "Cuántas casillas, según lo largo que sea el texto",
        ["palabras", "casillas", "veces las de la fila de arriba"], filas, "ddd",
        ["Doble de texto, casi cuatro veces más casillas: el cuadrado."]) + [""]
    fp = FRASE_FILA.split()
    n = len(fp)
    L += ["3. CUÁNTAS TANDAS HAY QUE HACER UNA DETRÁS DE OTRA",
          "   (una tanda: las cuentas que se pueden hacer a la vez)",
          f"   (la frase «{FRASE_FILA}», {n} palabras)", "",
          "  La que lee en orden: la cuenta de cada palabra necesita el",
          "  resumen que dejó la anterior.", ""]
    for i, p in enumerate(fp):
        L.append(f"    tanda {i + 1}: «{p}»" + (f", con el resumen de «{fp[i - 1]}»" if i else ""))
    L += ["", "  La que mira todo a la vez: la cuenta de cada palabra solo",
          "  necesita las listas de las palabras, que ya están todas.", "",
          "    tanda 1: «" + "», «".join(fp) + "», a la vez", ""]
    # L24 (9 de octubre, capítulo 9): la cuenta de tandas, en tabla editorial. Las líneas de
    # «tanda» de arriba se quedan como estaban: las lee figura_en_fila.py.
    L += tabla_editorial(
        "Cuántas tandas, una detrás de otra",
        ["palabras", "en orden", "a la vez"],
        [[miles(m), miles(pasos_en_fila(dependencias(m, True))),
          miles(pasos_en_fila(dependencias(m, False)))] for m in [n] + LONGITUDES_FILA],
        "ddd",
        ["En orden: la máquina que lee en orden. A la vez: la que lo mira todo de golpe. "
         "Una tanda: las cuentas que se pueden hacer a la vez."]) + [""]
    # El ancho se comprueba fuera de las tablas editoriales: ésas las compone el libro, no se
    # copian tal cual, y sus notas son párrafos.
    dentro = False
    for l in [l.rstrip() for l in L]:
        if l in ("::: tabla", "::: muestra"):
            dentro = True
        elif l == ":::":
            dentro = False
        elif not dentro:
            comprobar_ancho([l], ANCHO)
        print(l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
