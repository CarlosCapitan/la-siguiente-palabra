#!/usr/bin/env python3
"""
Capítulo 5 — ¿es «la misma dirección»?

El capítulo afirma, después de la resta de «rey», que en ese espacio de números «la
diferencia entre "rey" y "reina" resultó ser aproximadamente LA MISMA DIRECCIÓN que la
diferencia entre "padre" y "madre", y que la de "hermano" y "hermana"». Es la frase de la
que cuelga el remate del capítulo («lo más cerca que hemos estado de algo que uno llamaría
un concepto»), y hasta hoy no había ninguna línea en `datos/salidas/` detrás de ella:
`palabras_numeros.py` mide si la resta ACIERTA la palabra esperada, que es otra cosa. Que
«rey - hombre + mujer» caiga cerca de «reina» no dice cuánto se parecen entre sí las
cuatro diferencias.

Esto lo mide. Para cada pareja masculino->femenino se toma su diferencia, y se compara
cada diferencia con cada otra: 1 es exactamente la misma dirección, 0 es ninguna relación,
-1 es la contraria. Al lado, la misma cuenta para parejas de palabras cualesquiera, que es
contra lo que hay que compararlo: si dos diferencias al azar ya salieran a 0,6, un 0,6 en
las del género no significaría nada.

El corpus, los ajustes y la semilla son los de `palabras_numeros.py`, importados de allí y
no copiados, para que no puedan separarse con el tiempo.

Uso:
    python la_misma_direccion.py
    python la_misma_direccion.py --selftest
"""

# ======================= CONSTANTES =======================

PAREJAS_GENERO = [("rey", "reina"),
                  ("padre", "madre"),
                  ("hermano", "hermana"),
                  ("hijo", "hija")]

PAREJAS_AZAR = 400           # parejas de palabras cualesquiera para la comparación de referencia
TOL_INVARIANTE = 1e-5

# Umbrales del selftest. No se tocan para que salga un número más bonito: si alguno falla,
# el número no vale (encargo de auditoría, apartado 3).
NULO_MAXIMO = 0.10           # dos diferencias al azar no pueden parecerse más que esto, de media
IMPLANTADA_MINIMA = 0.99     # una dirección metida a mano tiene que salir reconocida

# ==========================================================

import argparse
import glob
import os
import platform
import random
import sys
from datetime import date

import numpy as np

from formato import coma, comprobar_ancho, miles
from palabras_numeros import (CORPUS_BIBLIOTECA, LIBROS_SELFTEST, SEMILLA,
                              cargar_corpus, entrenar, frases_de_fichero)


def coseno(u, v):
    """Cuánto apuntan dos listas de números en la misma dirección: 1 la misma, 0 ninguna
    relación, -1 la contraria. Es la misma cuenta con la que se mide quién está cerca de
    quién, aplicada aquí a DIFERENCIAS entre dos palabras, no a las palabras."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def diferencias(vectores, parejas):
    """La diferencia de cada pareja, y las parejas que se han podido usar."""
    usables = [(a, b) for a, b in parejas if a in vectores and b in vectores]
    return [vectores[b] - vectores[a] for a, b in usables], usables


def parecidos_entre_si(difs):
    """Todos los pares de diferencias, comparados uno con otro."""
    return [(i, j, coseno(difs[i], difs[j]))
            for i in range(len(difs)) for j in range(i + 1, len(difs))]


def referencia_al_azar(vectores, semilla=SEMILLA, n=PAREJAS_AZAR):
    """La misma cuenta para parejas de palabras cualesquiera. Sin esto no hay manera de
    saber si un 0,4 entre las diferencias del género es mucho o es lo normal."""
    rng = random.Random(semilla)
    vocab = list(vectores.index_to_key)
    difs = []
    for _ in range(n):
        a, b = rng.choice(vocab), rng.choice(vocab)
        if a != b:
            difs.append(vectores[b] - vectores[a])
    valores = [coseno(difs[i], difs[i + 1]) for i in range(0, len(difs) - 1, 2)]
    return valores


def medir(vectores):
    difs, usables = diferencias(vectores, PAREJAS_GENERO)
    pares = parecidos_entre_si(difs)
    azar = referencia_al_azar(vectores)
    return usables, pares, azar


def selftest():
    fallos = []
    rng = np.random.default_rng(SEMILLA)

    # 1. TEST NULO — dos diferencias entre palabras cualesquiera no pueden parecerse. Se
    #    hace sobre el modelo de verdad (60 libros de la biblioteca), no sobre ruido: lo
    #    que hay que descartar es que en ESTE espacio todo se parezca a todo, que es
    #    exactamente el error que ya se coló una vez en `palabras_numeros.py`.
    ficheros = sorted(glob.glob(os.path.join(CORPUS_BIBLIOTECA, "*.txt")))[:LIBROS_SELFTEST]
    assert ficheros, f"Se esperaban libros en «{CORPUS_BIBLIOTECA}»; no se encontró ninguno"
    frases = [f for ruta in ficheros for f in frases_de_fichero(ruta)]
    print(f"Corpus de prueba: {len(ficheros)} libros, {miles(sum(len(f) for f in frases))} palabras.\n")
    vectores = entrenar(frases, epocas=3).wv

    azar = referencia_al_azar(vectores)
    media_azar = float(np.mean(azar))
    print(f"[1] test nulo         dos diferencias entre palabras cualesquiera se parecen, "
          f"de media, {media_azar:+.3f} ({len(azar)} comparaciones)")
    if abs(media_azar) > NULO_MAXIMO:
        fallos.append(f"test nulo: dos diferencias al azar se parecen {media_azar:+.3f}; "
                      f"se esperaba algo menor que {NULO_MAXIMO} en valor absoluto")

    # 2. SEÑAL IMPLANTADA — se fabrican dos parejas cuya diferencia es, por construcción,
    #    la misma. La cuenta tiene que reconocerla; y con dos desplazamientos distintos,
    #    no.
    base_a, base_b = rng.normal(size=100), rng.normal(size=100)
    desplazamiento = rng.normal(size=100)
    otro = rng.normal(size=100)
    implantada = coseno((base_a + desplazamiento) - base_a,
                        (base_b + desplazamiento) - base_b)
    distinta = coseno((base_a + desplazamiento) - base_a, (base_b + otro) - base_b)
    print(f"[2] señal implantada  el mismo desplazamiento en dos parejas: {implantada:+.3f}; "
          f"dos desplazamientos distintos: {distinta:+.3f}")
    if implantada < IMPLANTADA_MINIMA or abs(distinta) > 0.5:
        fallos.append(f"señal implantada: {implantada:+.3f} con el mismo desplazamiento y "
                      f"{distinta:+.3f} con dos distintos")

    # 3. INVARIANTE DEL DOMINIO — la cuenta es simétrica, vale 1 consigo misma y nunca se
    #    sale de -1 a 1.
    palabras = list(vectores.index_to_key)[:80]
    propios = [coseno(vectores[p], vectores[p]) for p in palabras]
    todos = [coseno(vectores[a], vectores[b]) for a in palabras[:40] for b in palabras[:40]]
    simetria = max(abs(coseno(vectores[a], vectores[b]) - coseno(vectores[b], vectores[a]))
                   for a in palabras[:20] for b in palabras[:20])
    peor = max(abs(v - 1.0) for v in propios)
    fuera = [v for v in todos if v < -1 - TOL_INVARIANTE or v > 1 + TOL_INVARIANTE]
    print(f"[3] invariante        consigo misma: error máximo {peor:.2e}; asimetría máxima "
          f"{simetria:.2e}; valores fuera de [-1, 1]: {len(fuera)}")
    if peor > 1e-6 or simetria > 1e-6 or fuera:
        fallos.append(f"invariante: error {peor:.2e} consigo misma, asimetría {simetria:.2e}, "
                      f"{len(fuera)} fuera de rango")

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

    frases, total = cargar_corpus(CORPUS_BIBLIOTECA)
    vectores = entrenar(frases).wv
    print(f"Biblioteca: {miles(total)} palabras, {miles(len(vectores.index_to_key))} distintas.\n")

    usables, pares, azar = medir(vectores)

    medias = float(np.mean([c for _, _, c in pares]))
    media_azar = float(np.mean(azar))

    lineas = ["¿APUNTAN EN LA MISMA DIRECCIÓN LAS DIFERENCIAS DE GÉNERO?",
              "1 es exactamente la misma dirección; 0, ninguna relación;",
              "-1, la contraria",
              "",
              "la diferencia de     y la diferencia de   se parecen",
              "------------------   ------------------   ----------"]
    for i, j, c in pares:
        izq = f"{usables[i][0]} -> {usables[i][1]}"
        der = f"{usables[j][0]} -> {usables[j][1]}"
        lineas.append(f"{izq:<18}   {der:<18}   {coma(c, 2):>10}")
    lineas += ["",
               f"media de las {len(pares)} comparaciones de género:            "
               f"{coma(medias, 2):>10}",
               f"media de {len(azar)} comparaciones entre dos parejas",
               f"de palabras cualesquiera del mismo modelo:   {coma(media_azar, 2):>10}"]
    for l in comprobar_ancho(lineas):
        print(l)


if __name__ == "__main__":
    main()
