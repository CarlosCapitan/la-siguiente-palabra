#!/usr/bin/env python3
"""
Capítulo 2 — «¿es un cuatro?», que es la pregunta con la que abre el capítulo.

El capítulo empieza pidiéndole al lector que imagine un comité que decide UNA sola cosa:
si el dibujo que le ponen delante es un cuatro. Sí o no. Y más adelante afirma que
«reconocer un cuatro manuscrito resulta que es un problema de un solo lado de la raya».

Lo que `perceptron.py` mide no es esa pregunta: mide parejas de dígitos (el 4 contra el 9,
el 3 contra el 5). Distinguir un 4 de un 9 y reconocer un 4 entre los diez dígitos no son
la misma tarea, y la segunda es la que promete el capítulo. Este programa la mide.

Y mide de paso la misma pregunta para los diez dígitos, porque la afirmación del capítulo
es sobre el cuatro y conviene saber si vale para todos o solo para ése.

Uso:
    python es_un_cuatro.py
    python es_un_cuatro.py --selftest
"""

# ======================= CONSTANTES =======================

DIGITO_DEL_CAPITULO = 4      # el cuatro, que es el que sale en el capítulo
FRACCION_PRUEBA = 0.3        # parte de los datos reservada para evaluar, como en perceptron.py
PASOS_MAX = 100_000          # correcciones antes de rendirse, como en perceptron.py

UMBRAL_NULO = 0.65           # acierto equilibrado máximo admisible con etiquetas al azar
UMBRAL_IMPLANTADA = 0.95     # acierto mínimo exigido en la señal implantada

# ==========================================================

import argparse
import platform
import sys
from datetime import date

import numpy as np

from formato import comprobar_ancho, coma, miles, pct
from perceptron import SEMILLA, acierto, cargar_digitos, entrenar


def acierto_equilibrado(y, pred):
    """La media del acierto en cada uno de los dos grupos.

    Hace falta porque aquí los dos grupos son de tamaños muy distintos: de cada diez
    dibujos, uno es un cuatro y nueve no lo son. Un aparato que dijera «no» siempre
    acertaría el noventa por ciento de las veces sin haber aprendido nada, y el acierto
    a secas no lo delataría. Éste sí: se queda en la mitad, que es lo que vale el azar."""
    grupos = [float((pred[y == c] == c).mean()) for c in (1, -1)]
    return float(np.mean(grupos)), grupos[0], grupos[1]


def reparto(n, rng):
    """Los mismos ejemplos para aprender y para evaluar que usa perceptron.py."""
    idx = rng.permutation(n)
    corte = int(n * (1 - FRACCION_PRUEBA))
    return idx[:corte], idx[corte:]


def es_este_digito(X, t, digito, rng):
    """Un solo comité contra la pregunta «¿es este dígito?», con los otros nueve enfrente."""
    y = np.where(t == digito, 1, -1)
    tr, te = reparto(len(X), rng)
    w, sesgo, pasos = entrenar(X[tr], y[tr], pasos_max=PASOS_MAX)
    pred = np.where(X[te] @ w + sesgo > 0, 1, -1)
    equilibrado, en_los_que_si, en_los_que_no = acierto_equilibrado(y[te], pred)
    return {
        "ejemplos_entrenamiento": len(tr),
        "de_esos_son_el_digito": int((y[tr] == 1).sum()),
        "ejemplos_prueba": len(te),
        "acierto_prueba": acierto(X[te], y[te], w, sesgo),
        "acierto_en_los_que_si": en_los_que_si,
        "acierto_en_los_que_no": en_los_que_no,
        "acierto_equilibrado": equilibrado,
        "converge": pasos is not None,
        "correcciones": pasos,
    }


def pareja(X, t, a, b, rng):
    """El 4 contra el 9 tal como lo mide perceptron.py: sirve de comprobación cruzada."""
    mask = (t == a) | (t == b)
    Xs, ys = X[mask], np.where(t[mask] == a, 1, -1)
    tr, te = reparto(len(Xs), rng)
    w, sesgo, pasos = entrenar(Xs[tr], ys[tr], pasos_max=PASOS_MAX)
    return acierto(Xs[te], ys[te], w, sesgo), pasos is not None


# ---- El aspecto de los bloques ---------------------------------------------------
ANCHO_ROTULO = 44
# ---------------------------------------------------------------------------------


def bloque_del_cuatro(r):
    """El resultado de la pregunta del capítulo, con cada número rotulado entero.

    Un rótulo como «acierto» no dice de qué: si es sobre los dibujos que ha visto o
    sobre los que no, y si es sobre los cuatros o sobre todo. Aquí cada línea lo dice."""
    lineas = [
        "¿ES UN CUATRO? UN SOLO COMITÉ CONTRA LOS OTROS NUEVE DÍGITOS",
        "",
        f"{'dibujos que se le enseñan para aprender':<{ANCHO_ROTULO}}{miles(r['ejemplos_entrenamiento']):>8}",
        f"{'de ésos, los que sí son un cuatro':<{ANCHO_ROTULO}}{miles(r['de_esos_son_el_digito']):>8}",
        f"{'dibujos que no ve nunca, para evaluarlo':<{ANCHO_ROTULO}}{miles(r['ejemplos_prueba']):>8}",
        "",
        f"{'acierta sobre los dibujos que no vio':<{ANCHO_ROTULO}}{pct(r['acierto_prueba']):>8}",
        f"{'de ésos, sobre los que sí son un cuatro':<{ANCHO_ROTULO}}{pct(r['acierto_en_los_que_si']):>8}",
        f"{'de ésos, sobre los que no son un cuatro':<{ANCHO_ROTULO}}{pct(r['acierto_en_los_que_no']):>8}",
        "",
        f"{'deja de corregirse solo':<{ANCHO_ROTULO}}{('sí' if r['converge'] else 'NO'):>8}",
        f"{'correcciones hasta quedarse quieto':<{ANCHO_ROTULO}}{miles(r['correcciones']) if r['converge'] else '-':>8}",
    ]
    return comprobar_ancho(lineas)


def bloque_de_los_diez(filas):
    """La misma pregunta para los diez dígitos, uno por uno."""
    lineas = [
        "LA MISMA PREGUNTA PARA CADA UNO DE LOS DIEZ DÍGITOS",
        "",
        f"{'la pregunta':<20}{'acierta sobre los':>20}{'deja de':>14}",
        f"{'':<20}{'que no vio':>20}{'corregirse':>14}",
        f"{'-' * 18:<20}{'-' * 18:>20}{'-' * 12:>14}",
    ]
    for d, r in filas:
        lineas.append(
            f"{f'¿es un {d}?':<20}{pct(r['acierto_prueba']):>20}"
            f"{('sí' if r['converge'] else 'NO'):>14}"
        )
    return comprobar_ancho(lineas)


def selftest():
    fallos = []
    rng = np.random.default_rng(SEMILLA)
    X, t = cargar_digitos()

    # 1. TEST NULO — etiquetas permutadas: no hay nada que aprender. Se mira el acierto
    #    EQUILIBRADO, porque el acierto a secas se queda en el 90 % diciendo «no» siempre.
    y = rng.permutation(np.where(t == DIGITO_DEL_CAPITULO, 1, -1))
    tr, te = reparto(len(X), rng)
    w, b, _ = entrenar(X[tr], y[tr], pasos_max=20_000)
    pred = np.where(X[te] @ w + b > 0, 1, -1)
    eq, _, _ = acierto_equilibrado(y[te], pred)
    print(f"[1] test nulo         etiquetas permutadas: acierto equilibrado {coma(eq, 3)} (azar = 0,5)")
    if eq > UMBRAL_NULO:
        fallos.append(f"test nulo: {coma(eq, 3)} con etiquetas al azar; algo filtra información")

    # 2. SEÑAL IMPLANTADA — una columna que decide por sí sola: hay que recuperarla entera.
    Xi = rng.normal(size=(400, 5))
    yi = np.where(Xi[:, 2] > 0, 1, -1)
    w, b, _ = entrenar(Xi, yi, pasos_max=20_000)
    a_imp = acierto(Xi, yi, w, b)
    dominante = int(np.argmax(np.abs(w)))
    print(f"[2] señal implantada  acierto {coma(a_imp, 3)}; la entrada que más pesa es la {dominante} (implantada: 2)")
    if a_imp < UMBRAL_IMPLANTADA or dominante != 2:
        fallos.append(
            f"señal implantada: se esperaba acierto >= {UMBRAL_IMPLANTADA} con la entrada 2 dominante; "
            f"se obtuvo {coma(a_imp, 3)} con la {dominante}"
        )

    # 3. INVARIANTE DEL DOMINIO — la tabla del capítulo dice que el 4 contra el 9 sale al
    #    100 % y converge. Este programa usa la misma regla de Rosenblatt: tiene que salirle
    #    lo mismo. Si no, es que una de las dos mediciones no mide lo que dice medir.
    a_49, conv_49 = pareja(X, t, 4, 9, np.random.default_rng(SEMILLA))
    print(f"[3] invariante        el 4 contra el 9: acierto {pct(a_49, 0)} y converge {'sí' if conv_49 else 'NO'}"
          f" (la tabla del capítulo: 100 % y sí)")
    if round(100 * a_49) != 100 or not conv_49:
        fallos.append(f"invariante: el 4 contra el 9 debe dar 100 % y converger; dio {pct(a_49)} y {conv_49}")

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
    if args.selftest:
        sys.exit(selftest())

    print(f"Máquina: {platform.machine()}, {platform.system()} {platform.release()}.")
    print(f"Medido el {date.today().isoformat()}. Semilla: {SEMILLA}.")
    X, t = cargar_digitos()
    print(f"Dígitos manuscritos: {miles(len(X))} imágenes de 8 por 8 puntos.\n")

    print("--- 1. LA PREGUNTA CON LA QUE ABRE EL CAPÍTULO ---")
    rng = np.random.default_rng(SEMILLA)
    r4 = es_este_digito(X, t, DIGITO_DEL_CAPITULO, rng)
    for linea in bloque_del_cuatro(r4):
        print(linea)

    print("\n--- 2. LA MISMA PREGUNTA PARA LOS DIEZ DÍGITOS ---")
    filas = []
    for d in range(10):
        filas.append((d, es_este_digito(X, t, d, np.random.default_rng(SEMILLA))))
    for linea in bloque_de_los_diez(filas):
        print(linea)
    cuantos = sum(1 for _, r in filas if r["converge"])
    print(f"\ndejan de corregirse solos {cuantos} de los diez dígitos.")

    print("\n--- 3. COMPROBACIÓN CRUZADA CON LA TABLA DEL CAPÍTULO ---")
    a_49, conv_49 = pareja(X, t, 4, 9, np.random.default_rng(SEMILLA))
    print(f"el 4 contra el 9 (la tabla del capítulo): {pct(a_49, 0)} y converge "
          f"{'sí' if conv_49 else 'NO'}")


if __name__ == "__main__":
    main()
