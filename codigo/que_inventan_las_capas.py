#!/usr/bin/env python3
"""
Capítulo 3 — qué se inventan de verdad las neuronas de en medio.

El libro afirmaba que, al resolver el o exclusivo con dos neuronas intermedias, una
se encarga de «al menos uno está encendido» y la otra de «los dos están encendidos».
Eso estaba escrito y no estaba medido, y depende del arranque aleatorio. Esto lo mide.

Para cada arranque: se entrena una red de dos entradas, dos neuronas en medio y una
salida sobre las cuatro situaciones del o exclusivo; se mira qué contesta cada neurona
de en medio en esas cuatro situaciones; y ese sí/no de cuatro casillas se traduce a la
regla que esa neurona ha acabado calculando, de las dieciséis que hay.

Uso:
    python que_inventan_las_capas.py
    python que_inventan_las_capas.py --selftest
"""

# ======================= CONSTANTES =======================

ARRANQUES = 200            # semillas distintas; cada una es una red entrenada desde cero
SEMILLA_BASE = 20260916    # de aquí salen las ARRANQUES semillas, de forma reproducible
OCULTAS = 2                # neuronas en la capa de en medio
OCULTAS_NULO = 1           # test nulo: con una sola no puede salir señal
TASA = 0.5
PASOS = 20_000
UMBRAL = 0.5               # por encima, la neurona dice que sí
MINIMO_RESUELTAS = 1       # si no resuelve ninguna, algo está roto y hay que enterarse

SALIDA_CSV = "que_inventan_las_capas.csv"

# Las dos que ninguna neurona suelta puede calcular, porque una neurona es una raya
# (capítulo 2). Si alguna vez aparecen aquí, la medición está mal.
# Se apuntan por su patrón de cuatro respuestas, NO por su nombre: los nombres son texto
# del libro y el libro se reescribe. Una vez ya se quedaron apuntando a dos nombres que
# habían dejado de existir, y entonces este invariante comprobaba el vacío: no podía
# fallar nunca, que es la peor manera de pasar.
CLAVES_IMPOSIBLES = [(0, 1, 1, 0), (1, 0, 0, 1)]

# ==========================================================

import argparse
import csv
import sys
from collections import Counter

from formato import comprobar_ancho
import numpy as np

from retropropagacion import Red, TABLA_XOR, Y_XOR
from perceptron import NOMBRES   # las dieciséis reglas viven en el capítulo 2

IMPOSIBLES_PARA_UNA_SOLA = {NOMBRES[c] for c in CLAVES_IMPOSIBLES}


def comprobar_entrada():
    """La entrada de este programa es la tabla del o exclusivo. No viene de fuera, pero
    si alguien la toca, todo lo de abajo deja de significar lo que dice que significa."""
    assert TABLA_XOR.shape == (4, 2), \
        f"Se esperaba una tabla de 4 situaciones x 2 interruptores; se encontró {TABLA_XOR.shape}"
    assert set(np.unique(TABLA_XOR)) == {0.0, 1.0}, \
        f"Se esperaban solo ceros y unos en la tabla; se encontró {sorted(set(np.unique(TABLA_XOR)))}"
    esperado = [(0., 0.), (0., 1.), (1., 0.), (1., 1.)]
    encontrado = [tuple(f) for f in TABLA_XOR]
    assert encontrado == esperado, \
        f"Se esperaba el orden {esperado} (los nombres de las reglas dependen de él); se encontró {encontrado}"
    assert list(Y_XOR) == [0., 1., 1., 0.], \
        f"Se esperaba el o exclusivo como respuesta correcta; se encontró {list(Y_XOR)}"
    assert len(NOMBRES) == 16, \
        f"Se esperaban las dieciséis reglas nombradas; se encontraron {len(NOMBRES)}"
    assert len(IMPOSIBLES_PARA_UNA_SOLA) == 2, \
        f"Se esperaban 2 reglas imposibles para una neurona sola; se encontraron {IMPOSIBLES_PARA_UNA_SOLA}"


def regla_de(activaciones):
    """Traduce las cuatro respuestas de una neurona a la regla que ha acabado calculando."""
    assert activaciones.shape == (4,), \
        f"Se esperaban 4 respuestas, una por situación; se encontraron {activaciones.shape}"
    clave = tuple(int(a > UMBRAL) for a in activaciones)
    return NOMBRES[clave]


def entrenar_una(semilla, ocultas=OCULTAS):
    """Una red entrenada desde cero. Devuelve si resolvió, y qué calculó cada neurona de en medio."""
    red = Red([2, ocultas, 1], semilla=semilla).entrenar(TABLA_XOR, Y_XOR, TASA, PASOS)
    activaciones = red.adelante(TABLA_XOR)
    salida = activaciones[-1].ravel()
    resuelve = bool(((salida > UMBRAL) == (Y_XOR > UMBRAL)).all())
    medio = activaciones[1]                      # 4 situaciones x ocultas
    reglas = tuple(sorted(regla_de(medio[:, j]) for j in range(ocultas)))
    return resuelve, reglas


# Los rótulos de la tabla. Están aquí arriba, con nombre, y no escondidos dentro de un
# f-string: son texto del libro, y el libro se lee entero desde este bloque.
CABECERA_UNA = "lo que mira una de las dos"
CABECERA_OTRA = "lo que mira la otra"
CABECERA_CUENTA_1 = "de cada"
CABECERA_CUENTA_2 = "100"
ANCHO_CUENTA = 7   # lo que mide «de cada», para que las dos líneas de cabecera cuadren


def medir(arranques=ARRANQUES, ocultas=OCULTAS):
    semillas = np.random.default_rng(SEMILLA_BASE).integers(1, 2**31 - 1, size=arranques)
    resueltas, cuenta = 0, Counter()
    for s in semillas:
        resuelve, reglas = entrenar_una(int(s), ocultas)
        if resuelve:
            resueltas += 1
            cuenta[reglas] += 1
    return {"arranques": arranques, "resueltas": resueltas, "cuenta": cuenta}


def imprimir(r):
    """La tabla que cita el capítulo. Las divisiones las hace aquí el programa: en el
    libro no se calcula nada en prosa (regla 1 bis).

    La tabla lleva sus propios rótulos. Sin ellos son tres columnas de palabras sueltas
    y un número, y el lector que se la encuentre al volver la página no tiene manera de
    saber qué es cada cosa. Un rótulo no es adorno: es lo que convierte una rejilla de
    palabras en una tabla."""
    ancho = max(len(x) for reglas in r["cuenta"] for x in reglas)
    ancho = max(ancho, len(CABECERA_UNA), len(CABECERA_OTRA))
    hueco = 2
    cuenta = ANCHO_CUENTA

    def fila(a, b, n):
        return f"{a:<{ancho + hueco}}{b:<{ancho + hueco}}{n:>{cuenta}}"

    lineas = [
        f"de {r['arranques']} arranques distintos, {r['resueltas']} resolvieron el o exclusivo",
        f"y se repartieron el trabajo de {len(r['cuenta'])} maneras distintas:",
        "",
        fila("", "", CABECERA_CUENTA_1),
        fila(CABECERA_UNA, CABECERA_OTRA, CABECERA_CUENTA_2),
        fila("-" * ancho, "-" * ancho, "-" * cuenta),
    ]
    for reglas, n in r["cuenta"].most_common():
        por_cien = round(100 * n / r["resueltas"])
        lineas.append(fila(reglas[0], reglas[1], por_cien))
    lineas = [l.rstrip() for l in lineas]
    comprobar_ancho(lineas)
    for l in lineas:
        print(l)


def guardar(r):
    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["primera", "segunda", "veces", "de_resueltas", "de_arranques"])
        for reglas, n in r["cuenta"].most_common():
            w.writerow([reglas[0], reglas[1], n, r["resueltas"], r["arranques"]])


# ============================ SELFTEST ============================

def selftest():
    fallos = []
    comprobar_entrada()

    # [1] Test nulo: con UNA sola neurona en medio no hay red que valga, porque una
    #     neurona es una raya. Si sale señal aquí, la medición se la está inventando.
    nulo = medir(arranques=20, ocultas=OCULTAS_NULO)
    print(f"[1] test nulo         con una sola neurona en medio resuelven "
          f"{nulo['resueltas']} de {nulo['arranques']}")
    if nulo["resueltas"] != 0:
        fallos.append(f"test nulo: con una sola neurona en medio resolvieron {nulo['resueltas']}, "
                      f"y no debería resolver ninguna")

    # [2] Señal implantada: se construye a mano una neurona que es «al menos uno» y otra
    #     que es «los dos», y se comprueba que el traductor las llama por su nombre.
    al_menos_uno = np.array([0.02, 0.98, 0.98, 0.99])
    los_dos      = np.array([0.01, 0.03, 0.03, 0.97])
    nombres = (regla_de(al_menos_uno), regla_de(los_dos))
    print(f"[2] señal implantada  las reconoce como: {nombres[0]} / {nombres[1]}")
    esperados = (NOMBRES[(0, 1, 1, 1)], NOMBRES[(0, 0, 0, 1)])
    if nombres != esperados:
        fallos.append(f"señal implantada: esperaba {esperados}; encontré {nombres}")

    # [3] Invariante del dominio: ninguna neurona suelta puede calcular «exactamente uno»
    #     ni «los dos iguales» —es el resultado del capítulo 2—, así que no pueden salir
    #     nunca como trabajo de una neurona de en medio.
    r = medir(arranques=30)
    salidas = {x for reglas in r["cuenta"] for x in reglas}
    prohibidas = salidas & IMPOSIBLES_PARA_UNA_SOLA
    print(f"[3] invariante        {len(salidas)} reglas distintas, "
          f"{len(prohibidas)} de las imposibles para una sola neurona")
    if prohibidas:
        fallos.append(f"invariante: una neurona suelta no puede calcular {sorted(prohibidas)}, "
                      f"y la medición dice que lo hace")
    if r["resueltas"] < MINIMO_RESUELTAS:
        fallos.append(f"invariante: esperaba al menos {MINIMO_RESUELTAS} red resuelta de "
                      f"{r['arranques']}; encontré {r['resueltas']}")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        return selftest()
    comprobar_entrada()
    r = medir()
    imprimir(r)
    guardar(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
