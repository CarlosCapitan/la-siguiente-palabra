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

# ==========================================================

import argparse
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


def huecos(texto):
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
    casos = {
        "bloque, figura y prosa": ("    T\n\n![F.](f.png)\n\nLo que muestran.\n", 0),
        "bloque y destacado": ("    T\n\n::: destacado\nLa frase.\n:::\n\n## Dos\n", 0),
        "tabla de barras y título": ("| a | b |\n|---|---|\n\n## Dos\n", 0),
    }
    malos = [k for k, (t, esperado) in casos.items() if len(huecos(t)) != esperado]
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
    for ruta in args.capitulos:
        with open(ruta, encoding="utf-8") as fh:
            h = huecos(fh.read())
        for linea, que, detras in h:
            print(f"FALLA: {ruta.split('/')[-1]} línea {linea}: un{'a' if que == 'figura' else ''} "
                  f"{que} sin prosa detrás; llega {detras}")
        total += len(h)
    if total:
        print(f"\n{total} hueco(s): falta la conclusión en palabras (regla 9 bis).")
        sys.exit(1)
    print(f"PASA: {len(args.capitulos)} fichero(s); detrás de cada bloque y de cada figura hay "
          f"prosa (regla 9 bis). Que diga la conclusión se mira al leer.")


if __name__ == "__main__":
    main()
