#!/usr/bin/env python3
"""
Capítulo 3 — par o impar en un reloj digital, con tres pesos puestos a mano (L17, T16).

El capítulo enseña la tabla de los siete segmentos y pide buscar uno que esté encendido en los
cinco pares y apagado en los cinco impares. No lo hay. Luego dice que un comité, mirando varios
a la vez, sí lo consigue. Este programa hace ese paso a mano, con pasos que el lector puede
seguir con el dedo sobre la tabla:

  1. Qué dígitos se saltan, segmento a segmento, la regla «encendido en los cinco pares y apagado
     en los cinco impares»: los pares que lo tienen apagado y los impares que lo tienen encendido.
     Al de abajo izquierda solo se la salta uno, el cuatro, que es par y lo tiene apagado.
     (L23: hasta el 27 esto se contaba como «cuántos dígitos acierta cada segmento, en la mejor de
     sus dos lecturas», y que un segmento apagado «acierte» con el uno no lo entendía nadie.)
  2. Qué tiene el cuatro que no tenga nadie más: es el único con el de arriba izquierda encendido
     y el de arriba apagado.
  3. Tres pesos que dicen eso mismo en números (el de arriba −1, el de arriba izquierda +1, el de
     abajo izquierda +2), un listón de 0,5, y la cuenta de los diez dígitos.

Los pesos son los de la maqueta de Carlos (L9, `notas/maquetas/L9-pares-e-impares-reloj.png`).
Allí el listón era «llega a 1»; aquí es «pasa de 0,5», que es la forma en que el libro dice el
listón desde el capítulo 2 y deja a todos los dígitos lejos de él. Con totales enteros, las dos
dicen lo mismo, y el selftest lo comprueba.

Estos pesos NO los ha encontrado una máquina: están puestos a mano. El comité entrenado, con los
siete segmentos, es el de `siete_segmentos.py`, y el libro los presenta como dos cosas distintas.

Uso:
    python reloj_a_mano.py --selftest
    python reloj_a_mano.py > ../datos/salidas/reloj_a_mano.txt
"""

# ======================= CONSTANTES =======================

# Los pesos puestos a mano. Los segmentos que no están aquí pesan 0.
PESOS = {
    "el de arriba": -1,
    "el de arriba izquierda": +1,
    "el de abajo izquierda": +2,
}
LISTON = 0.5                     # para decir «par», el total tiene que pasar de aquí
LISTON_DE_LA_MAQUETA = 1         # la maqueta decía «si el total llega a 1»

# ==========================================================

import argparse
import itertools
import sys

import numpy as np

from formato import ANCHO_CAJA, coma, comprobar_ancho
from siete_segmentos import SEGMENTOS, tabla_de_segmentos

PARES = [d for d in range(10) if d % 2 == 0]
COL = 4
ANCHO_NOMBRE = 24


def vector_de_pesos(pesos):
    for s in pesos:
        assert s in SEGMENTOS, f"«{s}» no es uno de los siete segmentos: {SEGMENTOS}"
    return np.array([float(pesos.get(s, 0)) for s in SEGMENTOS])


def se_la_saltan(X, es_par, j):
    """Los dígitos que se saltan la regla «encendido en los pares, apagado en los impares» con el
    segmento j: los pares que lo tienen apagado y los impares que lo tienen encendido."""
    encendido = X[:, j] > 0
    pares_apagados = [d for d in range(len(es_par)) if es_par[d] and not encendido[d]]
    impares_encendidos = [d for d in range(len(es_par)) if not es_par[d] and encendido[d]]
    return pares_apagados, impares_encendidos


def lista(ds):
    return " ".join(str(d) for d in ds) if ds else "ninguno"


def decide(X, w, liston):
    return X @ w > liston


def signo(v):
    v = int(round(v))
    return "0" if v == 0 else f"{v:+d}"


def cabecera_digitos():
    return [f"{'':<{ANCHO_NOMBRE}}" + "".join(f"{d:>{COL}}" for d in range(10)),
            f"{'':<{ANCHO_NOMBRE}}" + "".join(f"{('par' if d % 2 == 0 else 'imp'):>{COL}}"
                                              for d in range(10)),
            f"{'':<{ANCHO_NOMBRE}}" + "".join(f"{'--':>{COL}}" for _ in range(10))]


def fila(nombre, valores):
    return f"{nombre:<{ANCHO_NOMBRE}}" + "".join(f"{v:>{COL}}" for v in valores)


def bloques(X):
    es_par = np.array([d % 2 == 0 for d in range(10)])
    w = vector_de_pesos(PESOS)
    L = []

    # ---- 1. cada segmento, él solo
    ANCHO_PA, ANCHO_IE = 12, 15
    L += ["1. QUÉ DÍGITOS SE SALTAN LA REGLA, SEGMENTO A SEGMENTO",
          "   (la regla: encendido en los cinco pares",
          "   y apagado en los cinco impares)", "",
          f"{'':<{ANCHO_NOMBRE}}{'pares':<{ANCHO_PA}}{'impares':<{ANCHO_IE}}",
          f"{'el segmento':<{ANCHO_NOMBRE}}{'apagados':<{ANCHO_PA}}{'encendidos':<{ANCHO_IE}}cuántos"]
    saltan = [se_la_saltan(X, es_par, j) for j in range(len(SEGMENTOS))]
    cuentas = [len(pa) + len(ie) for pa, ie in saltan]
    for nombre, (pa, ie), c in zip(SEGMENTOS, saltan, cuentas):
        L.append(f"{nombre:<{ANCHO_NOMBRE}}{lista(pa):<{ANCHO_PA}}{lista(ie):<{ANCHO_IE}}{c:>7}")
    mejor = int(np.argmin(cuentas))
    assert SEGMENTOS[mejor] == "el de abajo izquierda", \
        f"se esperaba que al que menos se la saltan fuera el de abajo izquierda; es {SEGMENTOS[mejor]}"
    assert min(cuentas) > 0, "a un segmento no se la salta nadie: el capítulo diría lo contrario"
    j = mejor
    L += ["", "EL DE ABAJO IZQUIERDA, DÍGITO A DÍGITO", ""] + cabecera_digitos()
    enc = X[:, j] > 0
    L.append(fila("debería estar encendido", ["sí" if p else "no" for p in es_par]))
    L.append(fila("está encendido", ["sí" if e else "no" for e in enc]))
    fallan = [d for d in range(10) if enc[d] != es_par[d]]
    assert fallan == saltan[j][0] + saltan[j][1], "el dígito a dígito no dice lo mismo que la tabla"
    L += ["", f"no coinciden: {', '.join(f'el {d}' for d in fallan)}"]

    # ---- 2. lo que solo tiene el cuatro
    a = SEGMENTOS.index("el de arriba izquierda")
    b = SEGMENTOS.index("el de arriba")
    solo = [d for d in range(10) if X[d, a] > 0 and X[d, b] == 0]
    L += ["", "", "2. LO QUE TIENE EL CUATRO Y NO TIENE NINGÚN OTRO", ""] + cabecera_digitos()
    L.append(fila("el de arriba izquierda", ["sí" if X[d, a] else "no" for d in range(10)]))
    L.append(fila("el de arriba", ["sí" if X[d, b] else "no" for d in range(10)]))
    L += ["", "encendido el de arriba izquierda y apagado el de arriba: "
          + ", ".join(f"el {d}" for d in solo)]

    # ---- 3. tres pesos y un listón
    L += ["", "", "3. TRES PESOS PUESTOS A MANO, Y LA CUENTA DE LOS DIEZ", "",
          f"{'el segmento':<{ANCHO_NOMBRE}}{'puntos que suma':>19}",
          f"{'-' * 23:<{ANCHO_NOMBRE}}{'-' * 15:>19}"]
    for s in SEGMENTOS:
        if s in PESOS:
            L.append(f"{s:<{ANCHO_NOMBRE}}{signo(PESOS[s]):>19}")
    L.append(f"{'los otros cuatro':<{ANCHO_NOMBRE}}{'0':>19}")
    L.append(f"{'':<{ANCHO_NOMBRE}}{'-' * 6:>19}")
    L.append(f"para decir «par», el total tiene que pasar de {coma(LISTON)}")

    cols = [s for s in SEGMENTOS if s in PESOS]
    idx = [SEGMENTOS.index(s) for s in cols]
    cortos = {"el de arriba": ("el de", "arriba"),
              "el de arriba izquierda": ("el de arriba", "izquierda"),
              "el de abajo izquierda": ("el de abajo", "izquierda")}
    ANCHO_D, ANCHO_C, ANCHO_T, ANCHO_DICE = 8, 14, 8, 8
    L += ["",
          f"{'':<{ANCHO_D}}" + "".join(f"{cortos[s][0]:>{ANCHO_C}}" for s in cols),
          f"{'dígito':<{ANCHO_D}}" + "".join(f"{cortos[s][1]:>{ANCHO_C}}" for s in cols)
          + f"{'total':>{ANCHO_T}}{'dice':>{ANCHO_DICE}}"]
    total = X @ w
    dice = decide(X, w, LISTON)
    for d in range(10):
        partes = [signo(X[d, k] * w[k]) for k in idx]
        L.append(f"{d:<{ANCHO_D}}" + "".join(f"{p:>{ANCHO_C}}" for p in partes)
                 + f"{signo(total[d]):>{ANCHO_T}}{('par' if dice[d] else 'impar'):>{ANCHO_DICE}}")
    bien = int(np.sum(dice == es_par))
    L += ["", f"los mismos pesos y el mismo listón aciertan {bien} de 10"]
    return comprobar_ancho([l.rstrip() for l in L], ANCHO_CAJA)


def selftest():
    fallos = []
    X = tabla_de_segmentos()
    w = vector_de_pesos(PESOS)
    es_par = np.array([d % 2 == 0 for d in range(10)])

    # 1. TEST NULO — de las 252 maneras de llamar «par» a cinco dígitos cualesquiera, los mismos
    #    pesos tienen que acertar los diez solo en una: la de verdad. Si acertaran muchas, los
    #    pesos no estarían diciendo nada de «par».
    dice = decide(X, w, LISTON)
    aciertan = [c for c in itertools.combinations(range(10), 5)
                if all(dice[d] == (d in c) for d in range(10))]
    print(f"[1] test nulo         de 252 maneras de elegir cinco «pares», aciertan los diez en "
          f"{len(aciertan)}: {aciertan}")
    if aciertan != [tuple(PARES)]:
        fallos.append(f"test nulo: aciertan los diez con {aciertan}, no solo con los pares")

    # 2. SEÑAL IMPLANTADA — se fabrica una tabla en la que el del medio está encendido justo en
    #    los pares. La cuenta de «qué dígitos se saltan la regla» tiene que encontrarlo, en su sitio,
    #    sin ninguno; y en la tabla de verdad, al de abajo izquierda se la tiene que saltar solo el 4,
    #    que es par y lo tiene apagado (lo que dice el capítulo).
    Xf = X.copy()
    m = SEGMENTOS.index("el del medio")
    Xf[:, m] = es_par.astype(float)
    limpios = [SEGMENTOS[j] for j in range(len(SEGMENTOS))
               if se_la_saltan(Xf, es_par, j) == ([], [])]
    ai = se_la_saltan(X, es_par, SEGMENTOS.index("el de abajo izquierda"))
    print(f"[2] señal implantada  en la tabla fabricada no se saltan la regla con: {limpios}; "
          f"en la de verdad, con el de abajo izquierda: {ai}")
    if limpios != ["el del medio"] or ai != ([4], []):
        fallos.append("señal implantada: no encuentra el segmento fabricado, o el de abajo "
                      "izquierda no falla solo con el 4")

    # 3. INVARIANTE DEL DOMINIO — el ocho enciende los siete segmentos, así que su total es la
    #    suma de todos los pesos. Y con totales enteros, «pasa de 0,5» y «llega a 1» (la maqueta)
    #    deciden lo mismo en los diez dígitos.
    total = X @ w
    ocho_ok = X[8].all() and total[8] == w.sum()
    iguales = bool(np.all((total > LISTON) == (total >= LISTON_DE_LA_MAQUETA)))
    enteros = bool(np.all(total == np.round(total)))
    print(f"[3] invariante        el ocho suma todos los pesos: {'sí' if ocho_ok else 'NO'}; "
          f"«pasa de 0,5» y «llega a 1» coinciden: {'sí' if iguales and enteros else 'NO'}")
    if not ocho_ok:
        fallos.append("invariante: el total del ocho no es la suma de todos los pesos")
    if not (iguales and enteros):
        fallos.append("invariante: el listón de 0,5 y el de la maqueta no deciden lo mismo")

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
    X = tabla_de_segmentos()
    print()
    print("########## capítulo 3: par o impar en un reloj, con tres pesos a mano ##########")
    print("Pesos puestos a mano, no entrenados. La cuenta no depende de la máquina.")
    print()
    for l in bloques(X):
        print(l)


if __name__ == "__main__":
    main()
