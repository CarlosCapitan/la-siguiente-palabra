#!/usr/bin/env python3
"""
Capítulo 6 — ¿lo escribe la red o lo copia?

El capítulo 6 enseña tres muestras de texto escritas por una red recurrente y dice de la
tercera que la red «ha aprendido que existen géneros». Antes de afirmar eso hay que
descartar lo aburrido: que la muestra sea un trozo del corpus repetido de memoria.

Esto mide exactamente eso. Para cada muestra busca la racha más larga de palabras
seguidas que aparece TAL CUAL en el texto con el que se entrenó, y qué parte de la
muestra está cubierta por rachas largas.

El texto se carga con `cargar_texto()` de `memoria_recurrente.py`, o sea con el mismo
recorte y la misma limpieza que vio la red: si se cargara de otra manera, la medida no
diría nada sobre esta red.

Uso:
    python cuanto_copia.py
    python cuanto_copia.py --selftest
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/memoria_recurrente.txt"
SALIDA_NGRAMA = "../datos/salidas/ngrama.txt"
# El capítulo 6 compara su texto con el de la máquina de contar del capítulo 1, así que la
# medida hay que hacérsela a las dos: si sólo se mide una, la comparación sigue sin datos.
MUESTRAS_NGRAMA = ("PALABRAS, frecuencias sueltas", "PALABRAS, 2 palabras de contexto")
RACHA_LARGA = 5        # a partir de cuántas palabras seguidas se considera copia
TOPE_NULO = 4          # lo que un texto ajeno puede coincidir por pura frecuencia
PALABRAS_IMPLANTADAS = 30

# ==========================================================

import argparse
import glob
import platform
import os
import re
import sys
import unicodedata
from datetime import date

from formato import coma, comprobar_ancho, miles
from memoria_recurrente import (ALFABETO, CORPUS, MAX_CARACTERES,
                                 cargar_texto)


def aplanar(t):
    """Una sola racha de espacios entre palabra y palabra.

    El corpus conserva los saltos de línea del libro original, y un salto de línea entre
    dos palabras es imprenta, no dato: si no se aplanan, «de la misma conformidad» deja de
    encontrarse solo porque el original partía ahí el renglón.
    """
    return re.sub(r"\s+", " ", t)


def racha_mas_larga(texto_generado, corpus):
    """Devuelve (racha más larga en palabras, palabras cubiertas por rachas largas).

    Una «racha» es un tramo de palabras seguidas de la muestra que aparece literal en el
    corpus. Se mide desde cada posición y se alarga mientras siga apareciendo.
    """
    palabras = texto_generado.split()
    mejor, cubiertas = 0, [False] * len(palabras)
    for i in range(len(palabras)):
        k = 0
        while i + k < len(palabras):
            trozo = " ".join(palabras[i:i + k + 1])
            if trozo not in corpus:
                break
            k += 1
        if k > mejor:
            mejor = k
        if k >= RACHA_LARGA:
            for j in range(i, i + k):
                cubiertas[j] = True
    return mejor, sum(cubiertas), len(palabras)


def muestras_del_fichero(ruta):
    """Las tres muestras que el capítulo imprime, leídas de la salida guardada."""
    texto = open(ruta, encoding="utf-8").read()
    pares = re.findall(r"\[arranque: «([^»]+)»\]\n(.+?)\n", texto)
    assert pares, f"No se encontró ninguna muestra en «{ruta}»"
    return pares


def muestras_del_ngrama(ruta):
    """Las muestras de PALABRAS de la máquina de contar del capítulo 1."""
    texto = open(ruta, encoding="utf-8").read()
    pares = []
    for rotulo in MUESTRAS_NGRAMA:
        m = re.search(r"--- " + re.escape(rotulo) + r" ---\n(.+?)\n", texto)
        assert m, f"No se encontró «{rotulo}» en «{ruta}»"
        pares.append((rotulo, m.group(1)))
    return pares


def libros_leidos():
    """Cuántos de los libros de la carpeta entran de verdad en el entrenamiento.

    `cargar_texto()` deja de leer al llegar al tope de caracteres, así que el texto del que
    se entrena NO es la carpeta entera. Esto repite su misma limpieza —y tiene que ser la
    misma, o el recuento sería de otro corpus— y cuenta los ficheros que llega a abrir.
    """
    todos = sorted(glob.glob(os.path.join(CORPUS, "*.txt")))
    texto = cargar_texto()
    total, leidos = 0, 0
    for ruta in todos:
        t = open(ruta, encoding="utf-8", errors="replace").read()
        if "*** START OF" in t:
            t = t.split("*** START OF", 1)[1].split("\n", 1)[-1]
        if "*** END OF" in t:
            t = t.split("*** END OF", 1)[0]
        t = unicodedata.normalize("NFC", t.lower())
        t = "".join(c if c in ALFABETO else " " for c in t)
        t = re.sub(r" {2,}", " ", t)
        total += len(t)
        leidos += 1
        if total >= MAX_CARACTERES:
            break
    return leidos, len(todos), texto


def titulo(ruta):
    for l in open(ruta, encoding="utf-8", errors="replace"):
        if l.startswith("Title:"):
            return l.split(":", 1)[1].strip()
    return "(sin título)"


# --------------------------- pruebas ---------------------------

def selftest():
    fallos = []
    texto = aplanar(cargar_texto())

    # 1. TEST NULO — una frase que no está en esos libros. Si la medida marcara aquí una
    #    racha larga, estaría contando coincidencias de palabras corrientes, no copias.
    ajena = ("el algoritmo descargó el fichero comprimido desde el servidor de la nube "
             "y actualizó el controlador de la tarjeta gráfica")
    n_ajena, _, _ = racha_mas_larga(ajena, texto)
    print(f"[1] test nulo         frase ajena al corpus: racha de {n_ajena} palabras "
          f"(tope {TOPE_NULO})")
    if n_ajena > TOPE_NULO:
        fallos.append(f"test nulo: racha de {n_ajena} en un texto que no es del corpus")

    # 2. SEÑAL IMPLANTADA — treinta palabras sacadas del propio corpus. La medida tiene que
    #    devolver esas treinta exactas: ni menos (se le escapa) ni más (no puede haber más).
    implantada = " ".join(texto.split()[500_000:500_000 + PALABRAS_IMPLANTADAS])
    n_impl, _, _ = racha_mas_larga(implantada, texto)
    print(f"[2] señal implantada  {PALABRAS_IMPLANTADAS} palabras copiadas del corpus: "
          f"racha de {n_impl}")
    if n_impl != PALABRAS_IMPLANTADAS:
        fallos.append(f"señal implantada: se esperaba {PALABRAS_IMPLANTADAS} y salió {n_impl}")

    # 3. INVARIANTE DEL DOMINIO — el orden es lo que se está midiendo. Las mismas treinta
    #    palabras del revés tienen que dejar de ser una copia.
    del_reves = " ".join(reversed(implantada.split()))
    n_rev, _, _ = racha_mas_larga(del_reves, texto)
    print(f"[3] invariante        las mismas {PALABRAS_IMPLANTADAS} del revés: "
          f"racha de {n_rev} (tope {TOPE_NULO})")
    if n_rev > TOPE_NULO:
        fallos.append(f"invariante: del revés sigue dando {n_rev}")

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
    print(f"Medido el {date.today().isoformat()}.")
    print("No depende de la máquina: no entrena nada; busca texto en texto.")
    print()
    leidos, hay, crudo = libros_leidos()
    texto = aplanar(crudo)
    print("--- DE CUÁNTOS LIBROS SE ENTRENA DE VERDAD ---")
    print(f"libros en la carpeta del corpus: {miles(hay)}")
    print(f"libros que entran antes del tope de caracteres: {miles(leidos)}")
    print(f"palabras de entrenamiento: {miles(len(crudo.split()))}")
    print()
    for ruta in sorted(glob.glob(os.path.join(CORPUS, "*.txt")))[:leidos]:
        t = titulo(ruta)
        if len(t) > 52:
            t = t[:51].rstrip(" ,.") + "…"
        print(f"  {os.path.basename(ruta):<12} {t}")

    print("\n--- CUÁNTO DE CADA MUESTRA ESTÁ COPIADO LITERAL ---")
    print("Racha = palabras seguidas que aparecen tal cual en el texto de")
    print(f"entrenamiento. «Copiado» = cubierto por rachas de {RACHA_LARGA} o más.\n")
    cabecera = [
        f"{'arranque de la muestra':<24}{'racha más larga,':>18}{'copiado,':>12}",
        f"{'':<24}{'en palabras':>18}{'en % de la':>12}",
        f"{'':<24}{'':>18}{'muestra':>12}",
        f"{'-' * 24}{'-' * 18}{'-' * 12}",
    ]
    for l in comprobar_ancho(cabecera):
        print(l)
    for arranque, muestra in muestras_del_fichero(SALIDA):
        mejor, cub, total = racha_mas_larga(muestra, texto)
        fila = (f"{'«' + arranque + '»':<24}{mejor:>18}"
                f"{coma(100 * cub / total) + ' %':>12}")
        print(*comprobar_ancho([fila]), sep="")

    print("\n--- LA MISMA MEDIDA, A LA MÁQUINA DE CONTAR DEL CAPÍTULO 1 ---")
    print("Otro corpus (el Quijote) y otra máquina, medidos igual.\n")
    import ngrama
    quijote = aplanar(ngrama.normalizar(ngrama.cargar_corpus(ngrama.CORPUS),
                                        ngrama.ALFABETO))
    cab = [
        f"{'muestra de la máquina de contar':<38}{'racha':>8}{'copiado':>10}",
        f"{'-' * 38}{'-' * 8}{'-' * 10}",
    ]
    for l in comprobar_ancho(cab):
        print(l)
    for rotulo, muestra in muestras_del_ngrama(SALIDA_NGRAMA):
        mejor, cub, total = racha_mas_larga(muestra, quijote)
        fila = f"{rotulo:<38}{mejor:>8}{coma(100 * cub / total) + ' %':>10}"
        print(*comprobar_ancho([fila]), sep="")

    print("\n--- LA RACHA MÁS LARGA, ENTERA ---")
    for arranque, muestra in muestras_del_fichero(SALIDA):
        palabras = muestra.split()
        mejor, ini = 0, 0
        for i in range(len(palabras)):
            k = 0
            while i + k < len(palabras) and " ".join(palabras[i:i + k + 1]) in texto:
                k += 1
            if k > mejor:
                mejor, ini = k, i
        print(f"[arranque: «{arranque}»] {mejor} palabras seguidas:")
        trozo = " ".join(palabras[ini:ini + mejor])
        for i in range(0, len(trozo), 64):
            print("   ", trozo[i:i + 64])
        print()


if __name__ == "__main__":
    main()
