#!/usr/bin/env python3
"""
Capítulo 2 — ¿existe la raya del ocho?

El capítulo enseña que el perceptrón, con la pregunta «¿es un ocho?», es el único de los diez que
no se queda quieto (`es_un_cuatro.txt`), y dejaba la pregunta abierta: ¿le faltaba plazo, o le
pasa lo que a la lámpara del pasillo, que no existe ninguna raya? (L15 de PENDIENTES.)

Eso se puede saber sin entrenar nada. Que exista una raya que deje todos los dibujos de un dígito
a un lado y los demás al otro es lo mismo que exista un caso para cada punto y un listón que
cumplan, dibujo a dibujo, «llega» o «no llega». Son 1.257 desigualdades con 65 incógnitas, y la
programación lineal dice con certeza si tienen solución. No hay plazo ni azar: la respuesta es sí
o no.

Además, para el ocho, comprueba un conjunto concreto de dibujos (ESTORBAN) que, quitados, dejan
que la raya exista. El conjunto lo encontró una búsqueda aparte (programación lineal entera, en
el Mac el 26 de septiembre de 2026, ver el docstring de ESTORBAN); aquí solo se VERIFICA, que es
rápido y exacto.

Uso:
    python el_ocho_y_la_raya.py --selftest
    python el_ocho_y_la_raya.py > ../datos/salidas/el_ocho_y_la_raya.txt
"""

# ======================= CONSTANTES =======================

DIGITO = 8
TABLA_DEL_PERCEPTRON = "../datos/salidas/es_un_cuatro.txt"   # quién se queda quieto

# Los dibujos que, quitados, dejan existir la raya del ocho: índices en el conjunto de dígitos
# (el que devuelve perceptron.cargar_digitos). Cómo se encontraron, el 26 de septiembre de 2026:
# una búsqueda de programación lineal entera (quitar los menos dibujos posibles), cortada a los
# 160 segundos, dio siete; al comprobarlos uno a uno, uno sobraba (el 814). Quedan seis, y aquí se
# verifica que sin ellos la raya existe y que ninguno sobra. Lo que NO está demostrado: que no
# baste con menos de seis quitando otros; la búsqueda solo acotó que hacen falta al menos tres.
# Por eso el libro dice «quitando estos seis», no «hacen falta seis».
ESTORBAN = [129, 899, 1149, 1529, 1553, 1781]

REPETICIONES_NULO = 5

# ==========================================================

import argparse
import platform
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from es_un_cuatro import reparto
from formato import ANCHO_CAJA, comprobar_ancho, miles
from perceptron import SEMILLA, cargar_digitos

AQUI = Path(__file__).resolve().parent


def existe_la_raya(X, y):
    """¿Hay un caso para cada punto (w) y un listón (b) con y·(X·w + b) >= 1 en todos los dibujos?
    El 1 no quita nada: si hay una raya que deja cada dibujo en su lado, se puede estirar hasta que
    el más justo quede a distancia 1. Devuelve True o False; revienta si el cálculo no termina."""
    assert X.ndim == 2 and len(X) == len(y), "se esperaba una matriz de dibujos y una etiqueta por dibujo"
    assert set(np.unique(y)) <= {-1, 1}, f"se esperaban etiquetas -1 y 1; hay {sorted(set(y))}"
    n, d = X.shape
    A = -y[:, None] * np.hstack([X, np.ones((n, 1))])
    r = linprog(np.zeros(d + 1), A_ub=A, b_ub=-np.ones(n), bounds=[(None, None)] * (d + 1),
                method="highs")
    assert r.status in (0, 2), f"se esperaba «hay solución» o «no la hay»; el cálculo dijo: {r.message}"
    return r.status == 0


def datos_de_aprendizaje():
    """Los mismos 1.257 dibujos con los que aprende el comité en es_un_cuatro.py."""
    X, t = cargar_digitos()
    tr, _ = reparto(len(X), np.random.default_rng(SEMILLA))
    return X, t, tr


def quien_se_queda_quieto(ruta):
    """De la tabla del capítulo (es_un_cuatro.txt): para cada dígito, si el perceptrón se quedó
    quieto. Se lee la salida, no se vuelve a entrenar: lo que se compara es lo que el libro dice."""
    texto = Path(ruta).read_text(encoding="utf-8")
    quietos = {}
    for d, s in re.findall(r"¿es un (\d)\?\s+[\d,]+ %\s+(sí|NO)", texto):
        quietos[int(d)] = (s == "sí")
    assert sorted(quietos) == list(range(10)), \
        f"se esperaban las diez filas de la tabla en {ruta}; se encontraron {sorted(quietos)}"
    return quietos


def selftest():
    fallos = []
    X, t, tr = datos_de_aprendizaje()
    Xs, y8 = X[tr], np.where(t[tr] == DIGITO, 1, -1)

    # 1. TEST NULO — las mismas respuestas, barajadas: tantos «sí» como ochos, pero repartidos al
    #    azar. Una pregunta sin sentido no puede tener raya; si la cuenta dijera que sí, diría que
    #    sí a todo.
    nulos = [existe_la_raya(Xs, np.random.default_rng(s).permutation(y8))
             for s in range(REPETICIONES_NULO)]
    print(f"[1] test nulo         respuestas barajadas: raya en {sum(nulos)} de {len(nulos)}")
    if any(nulos):
        fallos.append("test nulo: con las respuestas barajadas la cuenta encuentra raya")

    # 2. SEÑAL IMPLANTADA — respuestas fabricadas con una raya conocida: la cuenta tiene que
    #    encontrarla. Y si se le da la vuelta a un solo dibujo muy dentro de su lado, tiene que
    #    dejar de existir.
    rng = np.random.default_rng(SEMILLA)
    w0 = rng.normal(size=X.shape[1])
    total = Xs @ w0
    b0 = -np.median(total)
    y0 = np.where(total + b0 > 0, 1, -1)
    hay = existe_la_raya(Xs, y0)
    lejos = int(np.argmax(np.abs(total + b0)))
    y1 = y0.copy()
    y1[lejos] *= -1
    rota = existe_la_raya(Xs, y1)
    print(f"[2] señal implantada  raya fabricada: encontrada={hay}; con un dibujo al revés: {rota}")
    if not hay:
        fallos.append("señal implantada: no encuentra una raya que existe por construcción")
    if rota:
        fallos.append("señal implantada: encuentra raya con un dibujo puesto al otro lado")

    # 3. INVARIANTE — donde no hay raya, el perceptrón no puede quedarse quieto nunca (quedarse
    #    quieto es no fallar ningún dibujo, y eso ya es una raya). Donde la hay, el teorema de
    #    Rosenblatt dice que acaba quedándose quieto, aunque no dice cuándo. Se comprueba que la
    #    tabla del libro cuadra con eso, dígito a dígito.
    quietos = quien_se_queda_quieto(AQUI / TABLA_DEL_PERCEPTRON)
    rayas = {d: existe_la_raya(Xs, np.where(t[tr] == d, 1, -1)) for d in range(10)}
    distintos = [d for d in range(10) if rayas[d] != quietos[d]]
    print(f"[3] invariante        dígitos donde raya y quietud no coinciden: {distintos or 'ninguno'}")
    if distintos:
        fallos.append(f"invariante: la raya y la tabla del perceptrón no coinciden en {distintos}")

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
        sys.exit(codigo)

    X, t, tr = datos_de_aprendizaje()
    Xs = X[tr]
    quietos = quien_se_queda_quieto(AQUI / TABLA_DEL_PERCEPTRON)
    L = [
        "",
        "########## capítulo 2: ¿existe la raya del ocho? ##########",
        f"máquina: {platform.system()} {platform.machine()}. Medido el {date.today().isoformat()}.",
        "La respuesta no depende de la máquina: es una cuenta exacta.",
        f"Dibujos: los {miles(len(tr))} con los que aprende el comité del capítulo 2.",
        "",
        "¿EXISTE UNA RAYA QUE DEJE CADA DÍGITO A UN LADO Y LOS DEMÁS AL OTRO?",
        "",
        f"  {'la pregunta':<16}{'¿existe la raya?':>18}{'¿se queda quieto?':>22}",
        f"  {'-----------':<16}{'----------------':>18}{'-----------------':>22}",
    ]
    for d in range(10):
        raya = existe_la_raya(Xs, np.where(t[tr] == d, 1, -1))
        L.append(f"  {'¿es un ' + str(d) + '?':<16}{'sí' if raya else 'NO':>18}"
                 f"{'sí' if quietos[d] else 'NO':>22}")
    L += ["", "  «¿se queda quieto?» es la columna de la tabla del capítulo",
          "  (datos/salidas/es_un_cuatro.txt)."]

    assert ESTORBAN, "ESTORBAN está vacío: falta pegar el resultado de la búsqueda"
    y8 = np.where(t[tr] == DIGITO, 1, -1)
    dentro = np.isin(tr, ESTORBAN)
    assert dentro.sum() == len(ESTORBAN), \
        f"se esperaban {len(ESTORBAN)} dibujos de ESTORBAN entre los de aprendizaje; hay {dentro.sum()}"
    sin_ellos = existe_la_raya(Xs[~dentro], y8[~dentro])
    L += [
        "",
        "LOS DIBUJOS QUE IMPIDEN LA RAYA DEL OCHO",
        "",
        f"  {'con los ' + miles(len(tr)) + ' dibujos:':<30}raya: "
        f"{'sí' if existe_la_raya(Xs, y8) else 'NO'}",
        f"  {'quitando estos ' + str(len(ESTORBAN)) + ':':<30}raya: "
        f"{'sí' if sin_ellos else 'NO'}",
        "",
        f"  {'dibujo':>8}   {'qué es':<10}",
        f"  {'------':>8}   {'------':<10}",
    ]
    for i in ESTORBAN:
        L.append(f"  {i:>8}   {'un ' + str(int(t[i])):<10}")
    sobran = [i for i in ESTORBAN
              if existe_la_raya(Xs[~np.isin(tr, [j for j in ESTORBAN if j != i])],
                                y8[~np.isin(tr, [j for j in ESTORBAN if j != i])])]
    L += ["", f"  devolviendo cualquiera de ellos, la raya deja de existir: "
              f"{'sí' if not sobran else 'NO'}"]
    assert sin_ellos, "sin los dibujos de ESTORBAN la raya tendría que existir"
    assert not sobran, f"sobran dibujos en ESTORBAN: {sobran}"
    for l in comprobar_ancho([l.rstrip() for l in L], ANCHO_CAJA):
        print(l)


if __name__ == "__main__":
    main()
