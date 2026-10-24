#!/usr/bin/env python3
"""
Comprueba que cada título del libro está separado del texto de antes y del de después.

Sale de cuatro títulos que se quedaron pegados al párrafo anterior, sin línea en blanco
(uno en el cap. 3 y tres en el 4), por culpa de unos guiones que reescribían los
capítulos: en Markdown un «##» que no tiene una línea en blanco delante no es un título
sino texto, y el PDF y el Word lo enseñaron tal cual, con sus «##». Ningún otro
verificador lo veía, porque todos miran lo que dice el texto y no cómo está montado.

Mira los `manuscript/*.md`. Un título es una línea que empieza por «#», «##», «###» o
«####» y un espacio. Tiene que llevar una línea en blanco delante (salvo si es la primera
línea del fichero) y otra detrás (o el final del fichero): un título pegado al párrafo de
después tampoco lo sería a ojos de un lector que mira el PDF. No cuentan, porque no son
títulos del libro, las líneas de bloques de código (``` o cuatro espacios), de citas en
bloque (`>`) ni las de los bloques `::: tabla` y `::: muestra`.

Uso:
    python verificar_titulos.py ../../libro-ia-libro
    python verificar_titulos.py ../../libro-ia-libro --selftest
"""

# ======================= CONSTANTES =======================

MANUSCRITO = "manuscript"
TITULO = r"#{1,4} "                         # «#», «##», «###» o «####» y un espacio
BLOQUES_QUE_SE_SALTAN = ("tabla", "muestra")   # ::: tabla ... :::
MIN_TITULOS = 100                           # menos que esto en el libro, algo va mal en el recorrido

# ==========================================================

import argparse
import glob
import os
import re
import sys
import tempfile


def revisar_texto(nombre, texto):
    """Devuelve (cuántos títulos ha mirado, lista de fallos) de un fichero."""
    lineas = texto.split("\n")
    es_titulo = re.compile(TITULO)
    vacia = lambda i: lineas[i].strip() == ""
    mirados, fallos = 0, []
    en_codigo = False       # dentro de un bloque ``` ... ```
    en_bloque = False       # dentro de ::: tabla o ::: muestra
    for i, l in enumerate(lineas):
        if l.lstrip().startswith("```"):
            en_codigo = not en_codigo
            continue
        if en_codigo:
            continue
        if en_bloque:
            if l.strip() == ":::":
                en_bloque = False
            continue
        m = re.match(r"^:::\s*(\w+)", l)
        if m and m.group(1) in BLOQUES_QUE_SE_SALTAN:
            en_bloque = True
            continue
        if l.startswith("    ") or l.startswith(">") or not es_titulo.match(l):
            continue
        mirados += 1
        if i > 0 and not vacia(i - 1):
            fallos.append(f"{nombre} línea {i + 1}: «{l.strip()}» está pegado al texto de antes "
                          f"(le falta una línea en blanco delante)")
        if i + 1 < len(lineas) and not vacia(i + 1):
            fallos.append(f"{nombre} línea {i + 1}: «{l.strip()}» está pegado al texto de después "
                          f"(le falta una línea en blanco detrás)")
    return mirados, fallos


def revisar(raiz):
    ficheros = sorted(glob.glob(os.path.join(raiz, MANUSCRITO, "*.md")))
    assert ficheros, f"No hay ningún .md en {os.path.join(raiz, MANUSCRITO)}"
    total, fallos = 0, []
    for f in ficheros:
        m, fl = revisar_texto(os.path.basename(f), open(f, encoding="utf-8").read())
        total += m
        fallos += fl
    assert total >= MIN_TITULOS, \
        f"Solo he mirado {total} títulos en {len(ficheros)} ficheros; algo va mal en el recorrido"
    return len(ficheros), total, fallos


def imprimir(ficheros, titulos, fallos):
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        print(f"\nFALLA: {len(fallos)} título(s) pegado(s) al texto, de {titulos} en {ficheros} capítulos")
        return 1
    print(f"PASA: {titulos} títulos en {ficheros} capítulos, todos con su línea en blanco delante y detrás.")
    return 0


# ============================ SELFTEST ============================

BIEN = """# El título del capítulo

Un párrafo.

## Un apartado

Otro párrafo,
partido en dos líneas.

### Uno más pequeño

Texto.
"""


def carpeta_con(t, capitulos):
    """Monta en t un manuscript/ con tantos capítulos como textos se den."""
    os.makedirs(os.path.join(t, MANUSCRITO))
    for n, texto in enumerate(capitulos):
        open(os.path.join(t, MANUSCRITO, f"c{n:03d}.md"), "w", encoding="utf-8").write(texto)


def selftest(raiz):
    fallos = []
    cien = [BIEN] * MIN_TITULOS     # 300 títulos: pasa el mínimo que pide revisar()

    # [1] Test nulo: un capítulo bien formado (con el título en la primera línea) no da aviso,
    #     ni suelto ni dentro de una carpeta.
    m1, f1 = revisar_texto("bien.md", BIEN)
    with tempfile.TemporaryDirectory() as t:
        carpeta_con(t, cien)
        _, t1, f1b = revisar(t)
    print(f"[1] test nulo         un capítulo bien formado: {m1} títulos, {len(f1)} avisos; "
          f"{MIN_TITULOS} en una carpeta: {t1} títulos, {len(f1b)} avisos")
    if f1 or m1 != 3 or f1b:
        fallos.append("test nulo: un capítulo bien formado da avisos, o no cuenta sus títulos")

    # [2] Señal implantada: un título pegado al párrafo de antes se detecta en su línea exacta
    #     (la 4), y otro pegado al de después, en la suya (la 9); y lo mismo en una carpeta.
    mal = BIEN.replace("Un párrafo.\n\n## Un apartado\n", "Un párrafo.\n## Un apartado\n")
    mal = mal.replace("### Uno más pequeño\n\nTexto.", "### Uno más pequeño\nTexto.")
    _, f2 = revisar_texto("mal.md", mal)
    donde = sorted(int(re.search(r"línea (\d+)", f).group(1)) for f in f2)
    with tempfile.TemporaryDirectory() as t:
        carpeta_con(t, [mal] + cien[1:])
        _, _, f2b = revisar(t)
    print(f"[2] señal implantada  dos títulos pegados: avisos en las líneas {donde}; "
          f"en una carpeta: {len(f2b)} avisos")
    if donde != [4, 9] or not all("mal.md" in f for f in f2) or len(f2b) != 2 \
            or not all("c000.md" in f for f in f2b):
        fallos.append(f"señal implantada: esperaba avisos en las líneas [4, 9] y salen {donde}")

    # [3] Invariante del dominio: un «##» dentro de una cita en bloque, de una tabla, de una
    #     muestra o de un bloque de código no es un título y no cuenta, aunque esté pegado.
    ajeno = ("# Capítulo\n\nUn párrafo.\n\n> ## Dentro de una cita\n> ## y otra\n\n"
             "::: tabla\n## no es un título\n| a | b |\n:::\n\n"
             "::: muestra\n## ni éste\n:::\n\n"
             "    ## salida de un programa\n\n```\n## ni éste\n```\n\nFin.\n")
    m3, f3 = revisar_texto("ajeno.md", ajeno)
    print(f"[3] invariante        «##» en cita, tabla, muestra y código: {m3} títulos, {len(f3)} avisos")
    if f3 or m3 != 1:
        fallos.append(f"invariante: lo que no es título cuenta como título ({m3} títulos, {f3[:1]})")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raiz")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest(a.raiz)
    return imprimir(*revisar(a.raiz))


if __name__ == "__main__":
    sys.exit(main())
