#!/usr/bin/env python3
"""
Capítulo 15 — se refuerza lo que se premia: una tabla de números que aprende a hacer trampa.

Pedido por Carlos el 11 de octubre de 2026, para el apartado sobre las máquinas que «se saltan las
reglas». La idea que tiene que enseñar: para que aparezca la trampa no hace falta nadie dentro. Basta
con una nota que se pueda subir por un camino que nadie quería.

No hay modelo de lenguaje. Hay un tablero de 7 × 7 casillas, una casilla de salida, una meta y un
letrero que pone «meta conseguida». Quien se mueve por el tablero es una tabla: para cada casilla y
cada uno de los cuatro movimientos, un número que dice cuánto premio cabe esperar si se hace ese
movimiento ahí. Al principio son todos cero. Tras cada movimiento, el número se ajusta hacia el premio
que vino después (aprendizaje por refuerzo con tabla, Q-learning, Watkins 1989). Es la tercera fase
del adiestramiento del capítulo 12, sin texto: «estudiar para el juez».

La partida acaba al pisar la meta o el letrero, o tras PASOS_MAX movimientos. Al acabar, un juez pone
la nota (1 o 0). Hay dos jueces:

  - el que solo lee: pone un 1 si al final se lee «meta conseguida», lo diga la meta o el letrero;
  - el que comprueba: pone un 1 solo si se ha llegado a la meta.

Tres montajes:

  A. juez que solo lee, letrero cerca de la salida (a 2 pasos; la meta está a 6);
  B. juez que comprueba, el mismo tablero (test nulo: el atajo no da nota);
  C. juez que solo lee, letrero lejos (a 12 pasos, más lejos que la meta);
  D. juez que solo lee, letrero lejos, y la meta rodeada de muros: no se puede alcanzar. Es la
     «tarea imposible» del informe de OpenAI (el 93 % de lo que se discutía en el tablón de los
     agentes salía de las tareas que ningún modelo había resuelto nunca). Añadido el mismo día,
     antes de ejecutarlo.

Se mide, para SEMILLAS tablas que aprenden cada una con su sorteo, qué hace la tabla en distintos
momentos del aprendizaje si se la deja elegir siempre el número más alto: llegar al letrero, llegar
a la meta o ninguna de las dos. El aprendizaje se acota por partidas, no por reloj (regla 10).

Predicciones, escritas ANTES de ejecutar:
  - A: al final, 18 o más de 20 tablas van al letrero.
  - B: ninguna va al letrero.
  - C: 18 o más de 20 van a la meta (el letrero solo es «trampa» si es el camino más corto).
  - D: 18 o más de 20 van al letrero, aunque esté a 12 pasos.

Uso:
    python aprender_la_trampa.py > ../datos/salidas/aprender_la_trampa.txt
    python aprender_la_trampa.py --selftest
"""

# ======================= CONSTANTES =======================

LADO = 7                          # el tablero es de LADO × LADO casillas
SALIDA = (0, 0)
META = (3, 3)                     # a 6 pasos de la salida
LETRERO_CERCA = (0, 2)            # a 2 pasos
LETRERO_LEJOS = (6, 6)            # a 12 pasos
MUROS_META = {(2, 3), (4, 3), (3, 2), (3, 4)}      # rodean la meta en el montaje D
MOVIMIENTOS = [(0, 1), (1, 0), (0, -1), (-1, 0)]   # derecha, abajo, izquierda, arriba
NOMBRES_MOV = ["derecha", "abajo", "izquierda", "arriba"]

PARTIDAS = 2000                   # partidas de aprendizaje por tabla
PASOS_MAX = 40                    # movimientos por partida, como mucho
ALFA = 0.5                        # cuánto se mueve un número hacia lo que vino después
GAMMA = 0.95                      # cuánto vale un premio que llega un paso más tarde
EPSILON = 0.2                     # una de cada cinco veces, la tabla prueba un movimiento al azar
SEMILLAS = list(range(20))        # veinte tablas, cada una con su sorteo
MOMENTOS = [0, 10, 25, 50, 100, 200, 400, 1000, 2000]   # partidas aprendidas al mirar

# selftest
UMBRAL_SENAL = 0.9                # fracción de tablas que tienen que aprender el atajo en A
TOL = 1e-9

# ==========================================================

from formato import coma, tabla_editorial

import argparse
import random
import sys
from collections import deque

LEE, COMPRUEBA = "solo lee", "comprueba"
SIN_MUROS = frozenset()
MONTAJES = {
    "A": ("juez que solo lee, letrero cerca", LEE, LETRERO_CERCA, SIN_MUROS),
    "B": ("juez que comprueba, letrero cerca", COMPRUEBA, LETRERO_CERCA, SIN_MUROS),
    "C": ("juez que solo lee, letrero lejos", LEE, LETRERO_LEJOS, SIN_MUROS),
    "D": ("juez que solo lee, letrero lejos, meta tapiada", LEE, LETRERO_LEJOS,
          frozenset(MUROS_META)),
}


def dentro(c):
    return 0 <= c[0] < LADO and 0 <= c[1] < LADO


def validar():
    for nombre, c in [("salida", SALIDA), ("meta", META), ("letrero cerca", LETRERO_CERCA),
                      ("letrero lejos", LETRERO_LEJOS)]:
        assert dentro(c), f"Se esperaba la casilla {nombre} dentro del tablero; es {c}"
    assert len({SALIDA, META, LETRERO_CERCA, LETRERO_LEJOS}) == 4, \
        "Se esperaban cuatro casillas distintas"
    assert distancia(SALIDA, LETRERO_CERCA) < distancia(SALIDA, META) < \
        distancia(SALIDA, LETRERO_LEJOS), (
        "Se esperaba el letrero cerca más cerca que la meta, y el lejos más lejos; distancias "
        f"{distancia(SALIDA, LETRERO_CERCA)}, {distancia(SALIDA, META)}, "
        f"{distancia(SALIDA, LETRERO_LEJOS)}")
    assert not MUROS_META & {SALIDA, META, LETRERO_CERCA, LETRERO_LEJOS}, \
        "Se esperaban los muros fuera de la salida, la meta y los letreros"
    assert distancia(SALIDA, META, MUROS_META) is None, \
        "Se esperaba la meta inalcanzable con los muros del montaje D"
    assert distancia(SALIDA, LETRERO_LEJOS, MUROS_META) == distancia(SALIDA, LETRERO_LEJOS), \
        "Se esperaba que los muros no alargaran el camino al letrero lejos"
    assert 0 < ALFA <= 1 and 0 < GAMMA < 1 and 0 <= EPSILON < 1, "Parámetros fuera de rango"
    assert MOMENTOS == sorted(MOMENTOS) and MOMENTOS[-1] == PARTIDAS, \
        "Se esperaban los momentos en orden y el último igual a PARTIDAS"


def mover(c, m, muros=SIN_MUROS):
    """Un movimiento; contra el borde o contra un muro, la ficha se queda donde está."""
    n = (c[0] + MOVIMIENTOS[m][0], c[1] + MOVIMIENTOS[m][1])
    return n if dentro(n) and n not in muros else c


def distancia(a, b, muros=SIN_MUROS):
    """Pasos del camino más corto (en este tablero sin muros, la distancia en cuadrícula, pero se
    calcula recorriendo el tablero para no suponerlo)."""
    vistos, cola = {a: 0}, deque([a])
    while cola:
        c = cola.popleft()
        if c == b:
            return vistos[c]
        for m in range(len(MOVIMIENTOS)):
            n = mover(c, m, muros)
            if n not in vistos:
                vistos[n] = vistos[c] + 1
                cola.append(n)
    return None                     # no se puede llegar


def nota(juez, final):
    """La nota del juez al acabar la partida: `final` es «meta», «letrero» o «nada»."""
    if juez == LEE:
        return 1.0 if final in ("meta", "letrero") else 0.0
    return 1.0 if final == "meta" else 0.0


def donde(c, letrero):
    return "meta" if c == META else "letrero" if c == letrero else None


def mejores(tabla, c):
    """Los movimientos empatados en el número más alto de la casilla."""
    fila = tabla[c]
    tope = max(fila)
    return [m for m in range(len(fila)) if fila[m] >= tope - TOL]


def mejor(tabla, c, rng):
    """El movimiento de número más alto; un empate se decide a sorteo. Sin sorteo, con la tabla
    a cero se elegiría siempre «derecha», y el letrero cerca está a la derecha de la salida: el
    atajo saldría del orden de la lista, no de lo aprendido (visto antes de ejecutar)."""
    return rng.choice(mejores(tabla, c))


def tabla_nueva():
    return {(i, j): [0.0] * len(MOVIMIENTOS) for i in range(LADO) for j in range(LADO)}


def partida(tabla, rng, juez, letrero, muros=SIN_MUROS):
    """Una partida aprendiendo. Devuelve cómo acabó y la nota."""
    c = SALIDA
    for _ in range(PASOS_MAX):
        m = rng.randrange(len(MOVIMIENTOS)) if rng.random() < EPSILON else mejor(tabla, c, rng)
        n = mover(c, m, muros)
        final = donde(n, letrero)
        if final:
            r, siguiente = nota(juez, final), 0.0          # se acaba: no hay nada después
        else:
            r, siguiente = 0.0, max(tabla[n])
        tabla[c][m] += ALFA * (r + GAMMA * siguiente - tabla[c][m])
        if final:
            return final, r
        c = n
    return "nada", nota(juez, "nada")


def a_la_vista(tabla, letrero, muros=SIN_MUROS):
    """Qué hace la tabla si se la deja elegir siempre el número más alto, sin sorteo. Si en
    alguna casilla del camino todo sigue a cero, la tabla todavía no ha aprendido nada ahí: se
    devuelve «sin decidir» en vez de desempatar por el orden de la lista. Un empate por encima de
    cero es otra cosa: hay varios caminos igual de cortos hacia lo mismo (a la meta se llega de
    muchas maneras en 6 pasos), y cualquiera da el mismo final; se coge el primero. (La primera
    versión trataba todo empate como «sin decidir», y daba por indecisas tablas que llegaban a la
    meta siempre: lo cazó la nota media, 0,94 y 1, contra «ni una cosa ni otra» en 19 de 20.)
    Si vuelve a una casilla por la que ya pasó, da vueltas: «nada»."""
    c, pasos, vistos = SALIDA, 0, {SALIDA}
    while pasos < PASOS_MAX:
        ms = mejores(tabla, c)
        if max(tabla[c]) <= TOL and len(ms) > 1:
            return "sin decidir", pasos
        n = mover(c, ms[0], muros)
        pasos += 1
        final = donde(n, letrero)
        if final:
            return final, pasos
        if n in vistos:
            return "nada", pasos
        vistos.add(n)
        c = n
    return "nada", pasos


def aprender(semilla, juez, letrero, muros=SIN_MUROS):
    """Una tabla aprendiendo PARTIDAS partidas; qué hace a la vista en cada momento."""
    rng = random.Random(semilla)
    tabla, fotos, notas = tabla_nueva(), {}, []
    if 0 in MOMENTOS:
        fotos[0] = a_la_vista(tabla, letrero, muros)
    for k in range(1, PARTIDAS + 1):
        _, r = partida(tabla, rng, juez, letrero, muros)
        notas.append(r)
        if k in MOMENTOS:
            fotos[k] = a_la_vista(tabla, letrero, muros)
    return tabla, fotos, notas


def ejecutar(clave):
    _, juez, letrero, muros = MONTAJES[clave]
    return [aprender(s, juez, letrero, muros) for s in SEMILLAS]


def recuento(resultados, momento, final):
    return sum(1 for _, fotos, _ in resultados if fotos[momento][0] == final)


# ------------------------------------------------------------------------------ la salida

def bloque_montaje(clave, resultados):
    titulo, _, letrero, muros = MONTAJES[clave]
    d_meta = distancia(SALIDA, META, muros)
    meta = "no se puede llegar" if d_meta is None else f"a {d_meta}"
    filas = [[f"{k:,}".replace(",", "."), str(recuento(resultados, k, "letrero")),
              str(recuento(resultados, k, "meta")),
              str(len(SEMILLAS) - recuento(resultados, k, "letrero")
                  - recuento(resultados, k, "meta"))]
             for k in MOMENTOS]
    return tabla_editorial(
        f"Montaje {clave}: {titulo}",
        ["partidas aprendidas", "al letrero", "a la meta", "ni una cosa ni otra"], filas, "dddd",
        [f"De {len(SEMILLAS)} tablas, qué hace cada una si elige siempre el número más alto. "
         f"El letrero está a {distancia(SALIDA, letrero, muros)} pasos de la salida; a la meta, "
         f"{meta}."])


def bloque_dentro(resultados_a):
    tabla, fotos, _ = resultados_a[0]
    fila = tabla[SALIDA]
    filas = [[NOMBRES_MOV[m], coma(fila[m], 3)] for m in range(len(MOVIMIENTOS))]
    return tabla_editorial(
        "Lo que hay dentro, en la casilla de salida (montaje A, primera tabla)",
        ["movimiento", "número"], filas, "id",
        [f"Toda la tabla son {LADO * LADO * len(MOVIMIENTOS)} números como estos: "
         f"{LADO * LADO} casillas por {len(MOVIMIENTOS)} movimientos. Al acabar, esta tabla va "
         f"{'al letrero' if fotos[PARTIDAS][0] == 'letrero' else 'a la meta'} en "
         f"{fotos[PARTIDAS][1]} pasos."])


def nota_final(resultados, ultimas=100):
    notas = [r for _, _, ns in resultados for r in ns[-ultimas:]]
    return sum(notas) / len(notas)


# ------------------------------------------------------------------------------ selftest

def selftest():
    fallos = []
    a, b = ejecutar("A"), ejecutar("B")
    n = len(SEMILLAS)

    # 1. TEST NULO — con el juez que comprueba, el letrero no da nota: ninguna tabla puede
    #    acabar yendo a él. Y para que este nulo pueda fallar en la dirección del resultado, el
    #    mismo tablero con el juez que solo lee tiene que dar letreros (lo mira la prueba 2).
    b_letrero = recuento(b, PARTIDAS, "letrero")
    print(f"[1] test nulo         juez que comprueba: {b_letrero} de {n} tablas al letrero")
    if b_letrero != 0:
        fallos.append(f"test nulo: con el juez que comprueba, {b_letrero} tablas van al letrero")

    # 2. SEÑAL IMPLANTADA — con el juez que solo lee y el letrero a 2 pasos, el atajo tiene que
    #    aprenderse en la mayoría, y por el camino más corto: en el sitio y con la medida justas.
    a_letrero = recuento(a, PARTIDAS, "letrero")
    corto = distancia(SALIDA, LETRERO_CERCA)
    largos = [f[PARTIDAS][1] for _, f, _ in a if f[PARTIDAS][0] == "letrero"]
    print(f"[2] señal implantada  juez que solo lee: {a_letrero} de {n} al letrero; pasos del "
          f"camino aprendido: {sorted(set(largos))} (el más corto es {corto})")
    if a_letrero < UMBRAL_SENAL * n:
        fallos.append(f"señal implantada: se esperaban al menos {UMBRAL_SENAL * n:.0f} tablas "
                      f"al letrero; van {a_letrero}")
    if any(x != corto for x in largos):
        fallos.append(f"señal implantada: algún camino al letrero no es el más corto: {largos}")

    # 3. INVARIANTE DEL DOMINIO — la nota es 0 o 1 y nada se descuenta, así que ningún número de
    #    ninguna tabla puede salir de [0, 1]; y la nota de cada partida es 0 o 1.
    fuera = [v for res in (a, b) for t, _, _ in res for fila in t.values() for v in fila
             if v < -TOL or v > 1 + TOL]
    notas_raras = [r for res in (a, b) for _, _, ns in res for r in ns if r not in (0.0, 1.0)]
    print(f"[3] invariante        números fuera de [0, 1]: {len(fuera)}; "
          f"notas que no son 0 ni 1: {len(notas_raras)}")
    if fuera:
        fallos.append(f"invariante: {len(fuera)} números fuera de [0, 1]")
    if notas_raras:
        fallos.append(f"invariante: {len(notas_raras)} notas que no son 0 ni 1")

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
    validar()
    if args.selftest:
        sys.exit(selftest())

    print(f"tablero {LADO} × {LADO}; {PARTIDAS} partidas por tabla; {len(SEMILLAS)} tablas por "
          f"montaje; alfa {coma(ALFA, 2)}, gamma {coma(GAMMA, 2)}, epsilon {coma(EPSILON, 2)}")
    print()
    resultados = {}
    for clave in MONTAJES:
        resultados[clave] = ejecutar(clave)
        print("\n".join(bloque_montaje(clave, resultados[clave])))
        print(f"nota media en las 100 últimas partidas: "
              f"{coma(nota_final(resultados[clave]), 3)}")
        print()
    print("\n".join(bloque_dentro(resultados["A"])))


if __name__ == "__main__":
    main()
