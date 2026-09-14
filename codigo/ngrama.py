#!/usr/bin/env python3
"""
Capítulo 1 — la máquina de Shannon: predecir el siguiente carácter, y luego la siguiente palabra.

Construye modelos de n-gramas de órdenes crecientes sobre un texto en español y genera
muestras con cada uno. Es el experimento de Shannon de 1948, en castellano.

Uso:
    python ngrama.py
    python ngrama.py --selftest
"""

# ======================= CONSTANTES =======================

CORPUS = "../datos/quijote.txt"
MARCA_INICIO = "*** START OF"      # Gutenberg: recortar cabecera y pie
MARCA_FIN = "*** END OF"

ALFABETO = "abcdefghijklmnñopqrstuvwxyzáéíóúü ,.;:¿?¡!"
ORDENES_LETRA = [0, 1, 2, 3, 5]    # 0 = azar puro; 5 = cinco letras de contexto
ORDENES_PALABRA = [1, 2]           # 1 = solo frecuencias; 2 = pares de palabras

LARGO_MUESTRA_LETRAS = 220
LARGO_MUESTRA_PALABRAS = 35
SEMILLA = 20260914

MIN_CARACTERES_CORPUS = 500_000
TOL_SUMA = 1e-9

# ==========================================================

import argparse
import random
import re
import sys
import unicodedata
from collections import defaultdict, Counter


def cargar_corpus(ruta):
    with open(ruta, encoding="utf-8") as fh:
        texto = fh.read()
    if MARCA_INICIO in texto:
        texto = texto.split(MARCA_INICIO, 1)[1].split("\n", 1)[1]
    if MARCA_FIN in texto:
        texto = texto.split(MARCA_FIN, 1)[0]
    assert len(texto) >= MIN_CARACTERES_CORPUS, (
        f"Se esperaba un corpus de al menos {MIN_CARACTERES_CORPUS:,} caracteres; "
        f"se encontraron {len(texto):,}"
    )
    return texto


def normalizar(texto, alfabeto):
    texto = texto.lower().replace("\n", " ").replace("\r", " ")
    texto = unicodedata.normalize("NFC", texto)
    permitido = set(alfabeto)
    limpio = "".join(c if c in permitido else " " for c in texto)
    limpio = re.sub(r" {2,}", " ", limpio)
    assert set(limpio) <= permitido, (
        f"Se esperaba que el texto limpio solo usara el alfabeto declarado; "
        f"sobran {sorted(set(limpio) - permitido)[:10]}"
    )
    return limpio


def construir(secuencia, orden):
    """Tabla contexto -> contador de continuaciones. Sin suavizado: si un contexto no se
    ha visto, no se inventa nada; se corta y se reinicia (ver generar)."""
    assert orden >= 0, f"Se esperaba un orden no negativo; se encontró {orden}"
    tabla = defaultdict(Counter)
    if orden == 0:
        tabla[()] = Counter(secuencia)
        return tabla
    for i in range(len(secuencia) - orden):
        contexto = tuple(secuencia[i:i + orden])
        tabla[contexto][secuencia[i + orden]] += 1
    assert tabla, f"Se esperaba al menos un contexto para el orden {orden}; la tabla salió vacía"
    return tabla


def elegir(contador, rng):
    total = sum(contador.values())
    assert total > 0, "Se esperaba un contador no vacío; se encontró uno con total 0"
    umbral = rng.random() * total
    acumulado = 0
    for simbolo, n in contador.items():
        acumulado += n
        if acumulado > umbral:
            return simbolo
    return simbolo


def generar(tabla, orden, largo, rng, arranque=None):
    if orden == 0:
        return "".join(elegir(tabla[()], rng) for _ in range(largo)) if isinstance(
            next(iter(tabla[()])), str) else [elegir(tabla[()], rng) for _ in range(largo)]
    contextos = list(tabla.keys())
    estado = list(arranque) if arranque else list(rng.choice(contextos))
    salida = list(estado)
    for _ in range(largo - orden):
        clave = tuple(estado[-orden:])
        if clave not in tabla:                 # contexto nunca visto: reinicia, no inventa
            clave = rng.choice(contextos)
            estado = list(clave)
            salida.extend(clave)
        siguiente = elegir(tabla[clave], rng)
        salida.append(siguiente)
        estado.append(siguiente)
    return salida


def muestra_letras(secuencia, orden, rng):
    return "".join(generar(construir(secuencia, orden), orden, LARGO_MUESTRA_LETRAS, rng))


def muestra_palabras(palabras, orden, rng):
    return " ".join(generar(construir(palabras, orden), orden, LARGO_MUESTRA_PALABRAS, rng))


def fraccion_palabras_reales(texto, vocabulario):
    trozos = [t for t in texto.split() if t]
    if not trozos:
        return 0.0
    return sum(1 for t in trozos if t.strip(",.;:¿?¡!") in vocabulario) / len(trozos)


def selftest(secuencia, palabras, vocabulario):
    fallos = []
    rng = random.Random(SEMILLA)

    # 1. TEST NULO — corpus con las letras barajadas: destruye toda estructura. Un modelo de
    #    orden 3 entrenado sobre él no puede producir palabras reales.
    barajado = list(secuencia[:400_000])
    random.Random(SEMILLA).shuffle(barajado)
    texto_nulo = "".join(generar(construir(barajado, 3), 3, 2000, rng))
    frac_nula = fraccion_palabras_reales(texto_nulo, vocabulario)
    texto_real = "".join(generar(construir(secuencia[:400_000], 3), 3, 2000, rng))
    frac_real = fraccion_palabras_reales(texto_real, vocabulario)
    print(f"[1] test nulo         palabras reales: barajado={frac_nula:.3f}  real={frac_real:.3f}")
    if frac_nula > 0.25 or frac_nula >= frac_real:
        fallos.append(
            f"test nulo: el corpus barajado produce {frac_nula:.3f} de palabras reales "
            f"frente a {frac_real:.3f} del real; el montaje no distingue estructura"
        )

    # 2. SEÑAL IMPLANTADA — una cadena rara repetida debe recuperarse literalmente.
    marca = "zqzqzq"
    implantado = (secuencia[:200_000] + (" " + marca) * 400)
    tabla = construir(implantado, 4)
    salida = "".join(generar(tabla, 4, 4000, random.Random(SEMILLA), arranque="zqzq"))
    print(f"[2] señal implantada  «{marca}» aparece {salida.count(marca)} veces en la muestra")
    if marca not in salida:
        fallos.append(f"señal implantada: «{marca}» no aparece en la muestra generada")

    # 3. INVARIANTE DEL DOMINIO — toda letra generada pertenece al alfabeto, y las
    #    probabilidades de cada contexto suman uno.
    tabla3 = construir(secuencia[:200_000], 3)
    fuera = set(texto_real) - set(ALFABETO)
    sumas_mal = [c for c, cnt in list(tabla3.items())[:5000]
                 if abs(sum(n / sum(cnt.values()) for n in cnt.values()) - 1.0) > TOL_SUMA]
    print(f"[3] invariante        letras fuera del alfabeto: {len(fuera)}; contextos mal normalizados: {len(sumas_mal)}")
    if fuera:
        fallos.append(f"invariante: aparecen letras fuera del alfabeto: {sorted(fuera)[:5]}")
    if sumas_mal:
        fallos.append(f"invariante: {len(sumas_mal)} contextos cuyas probabilidades no suman 1")

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

    crudo = cargar_corpus(CORPUS)
    secuencia = normalizar(crudo, ALFABETO)
    palabras = [p for p in secuencia.split() if p]
    vocabulario = set(palabras)
    print(f"Corpus: {len(secuencia):,} caracteres, {len(palabras):,} palabras, "
          f"{len(vocabulario):,} palabras distintas.\n")

    if args.selftest:
        sys.exit(selftest(secuencia, palabras, vocabulario))

    rng = random.Random(SEMILLA)
    for orden in ORDENES_LETRA:
        etiqueta = "azar puro" if orden == 0 else f"{orden} letra{'s' if orden > 1 else ''} de contexto"
        print(f"--- LETRAS, {etiqueta} ---")
        print(muestra_letras(secuencia, orden, rng))
        print()
    for orden in ORDENES_PALABRA:
        etiqueta = "frecuencias sueltas" if orden == 1 else f"{orden} palabras de contexto"
        print(f"--- PALABRAS, {etiqueta} ---")
        print(muestra_palabras(palabras, orden, rng))
        print()


if __name__ == "__main__":
    main()
