#!/usr/bin/env python3
"""Comprueba la regla 9 bis: detrás de cada bloque y de cada figura va prosa.

    verificar_conclusiones.py capitulo.md [capitulo2.md ...]
    verificar_conclusiones.py --selftest

La regla 9 bis dice que los números son para quien quiera seguirlos y la conclusión para todos:
después de cada bloque de salida y de cada figura, un párrafo que diga en palabras qué muestran,
para que quien se salte la tabla no pierda el concepto. Un programa no puede saber si ese párrafo
dice de verdad la conclusión. Lo que sí puede saber es lo que viene detrás: si detrás de un bloque
o de una figura llega un título, o el final del fichero, sin una sola línea de prosa en medio, la
conclusión no está. Eso es lo que se comprueba aquí, y nada más.

Qué es cada cosa, mirando el texto crudo:
  - bloque: líneas sangradas con cuatro espacios, o citas que empiezan por «>»;
  - figura: una línea que empieza por «![»;
  - prosa: cualquier otra línea con texto que no sea un título («#»), el comienzo o el final de un
    destacado («:::»), una tabla de barras («|») ni otro bloque o figura. El texto de dentro de un
    destacado sí cuenta como prosa: es una frase del autor (regla 6 bis).

Las tablas de barras del capítulo 15 no son salida de máquina: son la lista de nombres, y no
entran en la regla.
"""

# ======================= CONSTANTES =======================

SANGRIA = "    "
CITA = ">"
FIGURA = "!["
TITULO = "#"
DESTACADO = ":::"
TABLA = "|"
# Las figuras cuya conclusión va en el párrafo de justo antes, declaradas una a una con su motivo
# (regla 9 bis). Solo figuras: a un bloque de salida se le exige siempre prosa detrás.
EXCEPCIONES = "../../libro-ia-libro/notas/CONCLUSION-ANTES.md"   # relativo a este programa

# ==========================================================

import argparse
import os
import re
import sys


def tipo(linea):
    if not linea.strip():
        return "blanca"
    if linea.startswith(SANGRIA) or linea.lstrip().startswith(CITA):
        return "bloque"
    if linea.startswith(FIGURA):
        return "figura"
    if linea.startswith(TITULO):
        return "titulo"
    if linea.startswith(DESTACADO):
        return "marca"
    if linea.startswith(TABLA):
        return "tabla"
    return "prosa"


def figura_de(linea):
    """El nombre del fichero de una línea de figura: «![…](figuras/x.png)» -> «x.png»."""
    m = re.search(r"\]\(([^)]+)\)", linea)
    assert m, f"Se esperaba una figura con su fichero entre paréntesis; se encontró {linea!r}"
    return os.path.basename(m.group(1))


def leer_excepciones(ruta=None):
    """Las figuras declaradas, de las líneas «figura.png | motivo». Revienta si alguna no trae
    motivo: declarar una excepción es un acto consciente, y el motivo es la prueba."""
    if ruta is None:
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), EXCEPCIONES)
    if not os.path.exists(ruta):
        return frozenset()
    figs = set()
    for l in open(ruta, encoding="utf-8"):
        # las líneas sangradas son el ejemplo del formato, no una excepción
        if (not l.startswith(" ") and l.count("|") == 1
                and l.split("|")[0].strip().endswith(".png")):
            fig, motivo = (x.strip() for x in l.split("|"))
            assert motivo, f"Se esperaba un motivo para la excepción de {fig}; no lo hay"
            figs.add(fig)
    return frozenset(figs)


def huecos(texto, excepciones=frozenset()):
    """Devuelve [(línea, qué, qué viene detrás)] de cada bloque o figura tras el que no llega prosa
    antes de un título o del final. Un bloque o una figura seguidos de otro bloque o figura no son
    un fallo por sí mismos: se mira el último de la racha."""
    lineas = texto.split("\n")
    tipos = [tipo(l) for l in lineas]
    fallos = []
    i, n = 0, len(lineas)
    while i < n:
        if tipos[i] not in ("bloque", "figura"):
            i += 1
            continue
        inicio, que = i, tipos[i]
        # la racha de bloques, figuras, blancas, marcas y tablas hasta la primera prosa o título
        j = i
        while j < n and tipos[j] in ("bloque", "figura", "blanca", "marca", "tabla"):
            j += 1
        racha = [k for k in range(i, j) if tipos[k] in ("bloque", "figura")]
        if (len(racha) == 1 and tipos[racha[0]] == "figura"
                and figura_de(lineas[racha[0]]) in excepciones):
            i = j
            continue
        if j == n:
            fallos.append((inicio + 1, que, "el final del fichero"))
        elif tipos[j] == "titulo":
            fallos.append((inicio + 1, que, f"el título «{lineas[j][:50]}»"))
        i = j
    return fallos


def selftest():
    fallos = []

    # 1. TEST NULO — un capítulo bien hecho: bloque, prosa; figura, prosa; dos bloques con una
    #    línea de enlace y prosa detrás del último. No puede dar ningún hueco.
    bien = ("# 1. Uno\n\nTexto.\n\n    TABLA\n    1  2\n\nLa tabla dice que uno es menos.\n\n"
            "![Una figura.](f.png)\n\nLa figura enseña lo mismo.\n\n    A\n\nY la otra:\n\n"
            "    B\n\nLas dos dicen lo mismo.\n\n## Otro\n\nFin.\n")
    h = huecos(bien)
    print(f"[1] test nulo         capítulo bien hecho: {len(h)} huecos")
    if h:
        fallos.append(f"test nulo: encontró huecos donde no los hay: {h}")

    # 2. SEÑAL IMPLANTADA — tres huecos puestos a propósito: un bloque seguido de un título, una
    #    figura seguida de un título y un bloque al final del fichero. Tiene que dar los tres, en
    #    sus líneas.
    mal = ("# 1. Uno\n\nTexto.\n\n    TABLA\n\n## Dos\n\n![F.](f.png)\n\n## Tres\n\nAlgo.\n\n"
           "    ULTIMA\n")
    h = huecos(mal)
    lineas = [x[0] for x in h]
    print(f"[2] señal implantada  tres huecos puestos: encontrados {len(h)}, en las líneas "
          f"{lineas}")
    if lineas != [5, 9, 15]:
        fallos.append(f"señal implantada: se esperaban huecos en 5, 9 y 15; salen {lineas}")

    # 3. INVARIANTE — bloque seguido de figura y de la prosa detrás vale (la prosa cubre las dos);
    #    un destacado cuenta como prosa; y una tabla de barras sola no es un bloque.
    decl = frozenset({"f.png"})
    casos = {
        "bloque, figura y prosa": ("    T\n\n![F.](f.png)\n\nLo que muestran.\n", 0),
        "bloque y destacado": ("    T\n\n::: destacado\nLa frase.\n:::\n\n## Dos\n", 0),
        "tabla de barras y título": ("| a | b |\n|---|---|\n\n## Dos\n", 0),
        "figura declarada y título": ("Antes.\n\n![F.](figuras/f.png)\n\n## Dos\n", 0),
        "figura sin declarar y título": ("Antes.\n\n![G.](figuras/g.png)\n\n## Dos\n", 1),
        "bloque, figura declarada y título": ("    T\n\n![F.](figuras/f.png)\n\n## Dos\n", 1),
    }
    malos = [k for k, (t, esperado) in casos.items() if len(huecos(t, decl)) != esperado]
    print(f"[3] invariante        {len(casos)} casos de forma: "
          f"{'todos bien' if not malos else 'MAL en ' + ', '.join(malos)}")
    if malos:
        fallos.append(f"invariante: {malos}")

    print()
    if fallos:
        print("SELFTEST: FALLA.")
        for f in fallos:
            print("  " + f)
        return False
    print("SELFTEST: las tres pruebas pasan.")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capitulos", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    assert args.capitulos, "Se esperaba al menos un capítulo; no se ha dado ninguno"
    total = 0
    excepciones = leer_excepciones()
    for ruta in args.capitulos:
        with open(ruta, encoding="utf-8") as fh:
            h = huecos(fh.read(), excepciones)
        for linea, que, detras in h:
            print(f"FALLA: {ruta.split('/')[-1]} línea {linea}: un{'a' if que == 'figura' else ''} "
                  f"{que} sin prosa detrás; llega {detras}")
        total += len(h)
    if total:
        print(f"\n{total} hueco(s): falta la conclusión en palabras (regla 9 bis).")
        sys.exit(1)
    print(f"PASA: {len(args.capitulos)} fichero(s); detrás de cada bloque y de cada figura hay "
          f"prosa (regla 9 bis), salvo {len(excepciones)} figura(s) declarada(s) con la "
          f"conclusión justo antes. Que diga la conclusión se mira al leer.")


if __name__ == "__main__":
    main()
