#!/usr/bin/env python3
"""
Capítulo 3 — por qué «¿es par?» es difícil escrito a mano, y fácil en un reloj digital.

El capítulo necesita justificar que «¿este dígito es par?» no se resuelve con una raya.
La justificación que estaba escrita era una intuición: «no hay ninguna forma que compartan
el cero, el dos, el cuatro, el seis y el ocho». Esto la pone a prueba en el sitio donde la
intuición es más fuerte —un reloj digital, donde los dígitos están hechos de siete piezas
fijas— y la intuición FALLA: ahí una sola raya sí separa los pares de los impares.

Ese fallo es el contenido del apartado. En un reloj hay siete piezas siempre en el mismo
sitio, y con eso casi cualquier división de los diez dígitos se resuelve con una raya.
Escritos a mano no hay piezas: hay tinta en sitios distintos cada vez, y ahí una sola raya
se queda en el 90 %. La dificultad no viene de «par»: viene de la escritura a mano.

Uso:
    python siete_segmentos.py
    python siete_segmentos.py --selftest
"""

# ======================= CONSTANTES =======================

SEMILLA = 20260917
PASOS_MAX = 20_000          # correcciones antes de dar una partición por no separable

# Los siete segmentos de un reloj digital, con el nombre que les pondría cualquiera.
# El orden de esta lista es el orden de las filas de la tabla.
SEGMENTOS = [
    "el de arriba",
    "el de arriba izquierda",
    "el de arriba derecha",
    "el del medio",
    "el de abajo izquierda",
    "el de abajo derecha",
    "el de abajo",
]

# Qué segmentos enciende cada dígito. Es el dibujo de toda la vida de un reloj de horno,
# de microondas o de despertador; se puede comprobar mirando uno.
ENCENDIDOS = {
    0: "el de arriba, el de arriba izquierda, el de arriba derecha, "
       "el de abajo izquierda, el de abajo derecha, el de abajo",
    1: "el de arriba derecha, el de abajo derecha",
    2: "el de arriba, el de arriba derecha, el del medio, "
       "el de abajo izquierda, el de abajo",
    3: "el de arriba, el de arriba derecha, el del medio, el de abajo derecha, "
       "el de abajo",
    4: "el de arriba izquierda, el de arriba derecha, el del medio, "
       "el de abajo derecha",
    5: "el de arriba, el de arriba izquierda, el del medio, el de abajo derecha, "
       "el de abajo",
    6: "el de arriba, el de arriba izquierda, el del medio, el de abajo izquierda, "
       "el de abajo derecha, el de abajo",
    7: "el de arriba, el de arriba derecha, el de abajo derecha",
    8: "el de arriba, el de arriba izquierda, el de arriba derecha, el del medio, "
       "el de abajo izquierda, el de abajo derecha, el de abajo",
    9: "el de arriba, el de arriba izquierda, el de arriba derecha, el del medio, "
       "el de abajo derecha, el de abajo",
}

# Una partición de los diez dígitos que NINGUNA raya separa en este espacio: el seis y el
# nueve a un lado y los otros ocho al otro. No está elegida a ojo —la primera que probé,
# «solo el cero a un lado», resultó SER separable y el test nulo la cazó—: sale del
# recuento de las 1.024 particiones, de las que 160 no se separan con una raya.
PARTICION_IMPOSIBLE = (0, 0, 0, 0, 0, 0, 1, 0, 0, 1)   # el 6 y el 9 contra los demás

SALIDA_CSV = "siete_segmentos.csv"

# ==========================================================

import argparse
import csv
import itertools
import sys

from formato import comprobar_ancho
import numpy as np


def tabla_de_segmentos():
    """Los diez dígitos como filas de ceros y unos, un número por segmento."""
    filas = []
    for d in range(10):
        enciende = [s.strip() for s in ENCENDIDOS[d].split(",")]
        for s in enciende:
            assert s in SEGMENTOS, f"«{s}» no es uno de los siete segmentos"
        filas.append([1.0 if s in enciende else 0.0 for s in SEGMENTOS])
    return np.array(filas)


def entrenar(X, y, pasos_max=PASOS_MAX, semilla=SEMILLA):
    """El perceptrón del capítulo 2, sin un solo añadido. Devuelve (pesos, listón) si
    encuentra la raya, y (None, None) si se rinde."""
    rng = np.random.default_rng(semilla)
    w = np.zeros(X.shape[1])
    b = 0.0
    orden = np.arange(len(X))
    for _ in range(pasos_max):
        rng.shuffle(orden)
        errores = 0
        for i in orden:
            if y[i] * (X[i] @ w + b) <= 0:
                w += y[i] * X[i]
                b += y[i]
                errores += 1
        if errores == 0:
            return w, b
    return None, None


def separable(X, bits):
    y = np.array([1.0 if b else -1.0 for b in bits])
    if len(set(bits)) == 1:            # todos al mismo lado: no hay nada que separar
        return True
    w, _ = entrenar(X, y)
    return w is not None


def un_solo_segmento_basta(X):
    """¿Hay UN segmento encendido en los cinco pares y apagado en los cinco impares (o al
    revés)? Es la versión ingenua de la pregunta, y la respuesta es que no."""
    for j, nombre in enumerate(SEGMENTOS):
        pares, impares = X[::2, j], X[1::2, j]
        if (pares.all() and not impares.any()) or (impares.all() and not pares.any()):
            return nombre
    return None


# ---- Los bloques que cita el capítulo ----------------------------------------------

ANCHO_NOMBRE = 24
COL = 4


def bloque_tabla(X):
    lineas = ["LOS DIEZ DÍGITOS DE UN RELOJ DIGITAL, SEGMENTO A SEGMENTO", ""]
    lineas.append(f"{'':<{ANCHO_NOMBRE}}" + "".join(f"{d:>{COL}}" for d in range(10)))
    lineas.append(f"{'':<{ANCHO_NOMBRE}}" +
                  "".join(f"{('par' if d % 2 == 0 else 'imp'):>{COL}}" for d in range(10)))
    lineas.append(f"{'':<{ANCHO_NOMBRE}}" + "".join(f"{'--':>{COL}}" for _ in range(10)))
    for j, nombre in enumerate(SEGMENTOS):
        # «sí» y «no», no un puntito: el libro no usa símbolos, y un punto en una tabla
        # hay que ir a buscar a qué se refiere (regla 1 y regla 9).
        celdas = "".join(f"{('sí' if X[d, j] else 'no'):>{COL}}" for d in range(10))
        lineas.append(f"{nombre:<{ANCHO_NOMBRE}}{celdas}")
    return comprobar_ancho(lineas)


def bloque_comite(X, w, b):
    lineas = ["EL COMITÉ QUE SÍ LO CONSIGUE, CON LOS SIETE SEGMENTOS", ""]
    for nombre, peso in zip(SEGMENTOS, w):
        lineas.append(f"{nombre:<{ANCHO_NOMBRE}}{peso:>+6.0f}")
    lineas.append(f"{'':<{ANCHO_NOMBRE}}{'-' * 6:>6}")
    # El listón puede salir cero, y «-0» no lo escribe nadie.
    liston = -b
    texto = f"{liston:+.0f}" if abs(liston) > 0.5 else "0"
    lineas.append(f"para decir «par», el total tiene que pasar de {texto}")
    lineas.append("")
    lineas.append(f"{'dígito':<10}{'total':>8}{'dice':>8}")
    for d in range(10):
        total = X[d] @ w + b
        lineas.append(f"{d:<10}{total:>+8.0f}{('par' if total > 0 else 'impar'):>8}"
                      f"   {'correcto' if (total > 0) == (d % 2 == 0) else 'MAL'}")
    return comprobar_ancho(lineas)


def selftest():
    fallos = []
    X = tabla_de_segmentos()

    # 1. TEST NULO — una partición que ninguna raya separa. Si el montaje dijera que sí,
    #    estaría diciendo que sí a todo y la medición no valdría nada.
    imposible = separable(X, PARTICION_IMPOSIBLE)
    print(f"[1] test nulo         una partición que no se puede separar: "
          f"{'la separa, MAL' if imposible else 'no la separa, bien'}")
    if imposible:
        fallos.append("test nulo: dice separar una partición que no es separable")

    # 2. SEÑAL IMPLANTADA — par contra impar tiene que salir, y con los diez bien.
    y = np.array([1.0 if d % 2 == 0 else -1.0 for d in range(10)])
    w, b = entrenar(X, y)
    aciertos = 0 if w is None else int(sum((X[d] @ w + b > 0) == (d % 2 == 0)
                                          for d in range(10)))
    print(f"[2] señal implantada  par contra impar: "
          f"{'no converge' if w is None else f'converge y acierta {aciertos} de 10'}")
    if w is None or aciertos != 10:
        fallos.append("señal implantada: par contra impar debería separarse con los diez bien")

    # 3. INVARIANTE DEL DOMINIO — una raya separa dos grupos; si se cambia de lado a los
    #    dos grupos, la misma raya sirve girada. Si una partición se separa y su contraria
    #    no, el comprobador no está midiendo una raya.
    malas = []
    for bits in [(0,1,0,1,0,1,0,1,0,1), (1,1,0,0,0,0,0,0,0,0), PARTICION_IMPOSIBLE]:
        contraria = tuple(1 - x for x in bits)
        if separable(X, bits) != separable(X, contraria):
            malas.append(bits)
    print(f"[3] invariante        una partición y su contraria salen igual: "
          f"{'sí' if not malas else f'NO en {malas}'}")
    if malas:
        fallos.append(f"invariante: {malas} no coincide con su contraria")

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
    ap.add_argument("--recuento", action="store_true",
                    help="cuenta cuántas de las 1.024 particiones separa una raya (tarda)")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    X = tabla_de_segmentos()
    for l in bloque_tabla(X):
        print(l)

    print()
    nombre = un_solo_segmento_basta(X)
    print("¿hay un solo segmento encendido en los cinco pares y apagado en los")
    print(f"cinco impares?   {nombre if nombre else 'no, ninguno'}")

    y = np.array([1.0 if d % 2 == 0 else -1.0 for d in range(10)])
    w, b = entrenar(X, y)
    assert w is not None, "no encontró la raya; con estos siete segmentos debería encontrarla"
    print()
    for l in bloque_comite(X, w, b):
        print(l)

    filas = [["segmento", "peso"]] + [[n, f"{p:.0f}"] for n, p in zip(SEGMENTOS, w)]
    filas.append(["liston", f"{-b:.0f}"])

    if args.recuento:
        print()
        n = sum(1 for bits in itertools.product([0, 1], repeat=10) if separable(X, bits))
        print(f"de las 1.024 maneras de partir los diez dígitos en dos grupos,")
        print(f"una sola raya separa {n}.")
        filas.append(["particiones_separables", str(n)])

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(filas)
    print(f"\nEscrito {SALIDA_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
