#!/usr/bin/env python3
"""
Capítulo 13 — las dos tablas de la batería, en preguntas (L24).

Los capítulos 11 y 12 dan la batería de treinta preguntas en preguntas acertadas («2 de 5»,
«19 de 30»), no en porcentajes. El 13 tiene que hablar igual. Este programa no vuelve a pasar
ningún modelo: lee lo que ya midieron `comprimir.py` (datos/salidas/comprimir.csv) y
`crudo_o_adiestrado_una_a_una.py` (datos/salidas/crudo_o_adiestrado_una_a_una.txt, bloque del
modelo de 7.000M) y lo imprime en preguntas.

  1. La batería con tres máquinas: 7.000M sin comprimir, 7.000M comprimido, 32.000M comprimido.
  2. La columna «adiestrado» del capítulo 12 al lado de «7.000M sin comprimir» del 13: el mismo
     modelo, la misma batería, la misma regla, y otro programa que lo pone en marcha.

Uso:
    python comprimir_en_preguntas.py --selftest
    python comprimir_en_preguntas.py > ../datos/salidas/comprimir_en_preguntas.txt
"""

# ======================= CONSTANTES =======================

CSV_COMPRIMIR = "../datos/salidas/comprimir.csv"
TXT_CAP12 = "../datos/salidas/crudo_o_adiestrado_una_a_una.txt"
BLOQUE_CAP12 = "--- 3. LA BATERÍA EN TRES COLUMNAS, EN PREGUNTAS (7.000M) ---"
POR_TAREA = 5
TAREAS = ["sumar dos cifras", "plurales", "traducir del inglés", "saber cosas del mundo",
          "seguir un patrón inventado", "razonar sobre una frase"]
COLUMNAS = ["7B sin comprimir", "7B comprimido", "32B comprimido"]

# ==========================================================

import argparse
import csv
import re
import sys
from pathlib import Path

from formato import ANCHO_CAJA, comprobar_ancho

AQUI = Path(__file__).resolve().parent


def leer_comprimir(ruta):
    filas = {}
    with open(ruta, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["clave"] != "bateria" or r["tarea"] == "media":
                continue
            filas[r["tarea"]] = [float(r[c]) for c in COLUMNAS]
    assert list(filas) == TAREAS, f"se esperaban las seis tareas en orden; hay {list(filas)}"
    aciertos = {}
    for t, fr in filas.items():
        n = [f * POR_TAREA for f in fr]
        assert all(abs(x - round(x)) < 1e-6 and 0 <= round(x) <= POR_TAREA for x in n), \
            f"«{t}»: {fr} no son aciertos sobre {POR_TAREA} preguntas"
        aciertos[t] = [int(round(x)) for x in n]
    return aciertos


def leer_cap12(ruta):
    texto = Path(ruta).read_text(encoding="utf-8")
    assert texto.count(BLOQUE_CAP12) == 1, f"se esperaba el bloque «{BLOQUE_CAP12}» una vez en {ruta}"
    tramo = texto.split(BLOQUE_CAP12, 1)[1]
    adiestrado = {}
    for t in TAREAS:
        m = re.search(rf"^{re.escape(t)}\s+(\d) de 5\s+(\d) de 5\s+(\d) de 5\s*$", tramo, re.M)
        assert m, f"no encuentro la fila «{t}» en el bloque del 7.000M de {ruta}"
        adiestrado[t] = int(m.group(2))            # columnas: crudo, adiestrado, con formato
    return adiestrado


# Rótulos de las columnas tal como los escribe el libro (segunda vuelta, 28 de septiembre de 2026):
# las tablas del capítulo 13 son tablas de texto, no bloques de consola, y sus rótulos y su clave
# tienen que salir de aquí, no de la salida vieja en porcentajes de `comprimir_a_la_vista.py`.
ROTULOS_1 = ["7.000M sin comprimir", "7.000M comprimido", "32.000M comprimido"]
ROTULOS_2 = ["cap. 12, adiestrado", "cap. 13, sin comprimir", "¿cambia?"]


def bloques(a, c12):
    """Las dos tablas, con los rótulos del libro, una fila por línea, y la clave de cada una
    (esta sí, en líneas que caben en un bloque del libro)."""
    L = ["1. LA BATERÍA CON TRES MÁQUINAS, EN PREGUNTAS", "",
         f"{'tarea':<28}" + "".join(f"{r:>24}" for r in ROTULOS_1)]
    for t in TAREAS:
        L.append(f"{t:<28}" + "".join(f"{f'{x} de 5':>24}" for x in a[t]))
    tot = [sum(a[t][k] for t in TAREAS) for k in range(3)]
    L.append(f"{'en total':<28}" + "".join(f"{f'{x} de 30':>24}" for x in tot))
    L += [""] + comprobar_ancho([
        "    «7.000M», «32.000M»: millones de números (nominal).",
        "    «comprimido»: guardado con menos detalle en cada número;",
        "    «sin comprimir»: tal como salió del entrenamiento.",
        "    las tres máquinas están adiestradas (capítulo 12)."], ANCHO_CAJA)
    L += ["", "2. LA MISMA MÁQUINA EN LOS CAPÍTULOS 12 Y 13", "",
          f"{'tarea':<28}" + "".join(f"{r:>24}" for r in ROTULOS_2)]
    for t in TAREAS:
        x, y = c12[t], a[t][0]
        L.append(f"{t:<28}{f'{x} de 5':>24}{f'{y} de 5':>24}{('sí' if x != y else 'no'):>24}")
    t12, t13 = sum(c12.values()), tot[0]
    L.append(f"{'en total':<28}{f'{t12} de 30':>24}{f'{t13} de 30':>24}")
    L += [""] + comprobar_ancho([
        "    cap. 12: 7.000M adiestrado, con el programa de uso general.",
        "    cap. 13: 7.000M sin comprimir, con el programa de Apple."], ANCHO_CAJA)
    return L, tot, (t12, t13)


def selftest():
    fallos = []
    a = leer_comprimir(AQUI / CSV_COMPRIMIR)
    c12 = leer_cap12(AQUI / TXT_CAP12)
    # 1. TEST NULO — un csv con una tarea que no es un número entero de aciertos de 5 revienta.
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fh:
        fh.write("clave,tarea," + ",".join(COLUMNAS) + "\n")
        for t in TAREAS:
            fh.write(f"bateria,{t},0.500,0.600,0.600\n")
    try:
        leer_comprimir(fh.name)
        fallos.append("test nulo: acepta 0,5 como aciertos sobre 5")
    except AssertionError:
        pass
    finally:
        os.unlink(fh.name)
    print("[1] test nulo         un 0,5 no se deja convertir en aciertos de 5")
    # 2. SEÑAL — la casilla que el capítulo comenta: el 32.000M acierta 0 de 5 en «saber cosas
    #    del mundo», y el 7.000M sin comprimir y el adiestrado del 12 difieren en dos filas.
    _, tot, (t12, t13) = bloques(a, c12)
    cambian = [t for t in TAREAS if c12[t] != a[t][0]]
    ok = a["saber cosas del mundo"][2] == 0 and cambian == ["sumar dos cifras", "razonar sobre una frase"]
    print(f"[2] señal             32.000M en «saber cosas del mundo»: {a['saber cosas del mundo'][2]} de 5; "
          f"cambian: {cambian}")
    if not ok:
        fallos.append("señal: no salen la casilla del cero ni las dos filas que cambian")
    # 3. INVARIANTE — el total en preguntas es la media del csv por treinta.
    medias = {}
    with open(AQUI / CSV_COMPRIMIR, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["clave"] == "bateria" and r["tarea"] == "media":
                medias = [float(r[c]) for c in COLUMNAS]
    ok = all(abs(m * 30 - x) < 0.02 for m, x in zip(medias, tot))
    print(f"[3] invariante        totales {tot} = medias del csv {medias} por 30: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: los totales no son la media por treinta")
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
    print()
    L, _, _ = bloques(leer_comprimir(AQUI / CSV_COMPRIMIR), leer_cap12(AQUI / TXT_CAP12))
    for l in L:
        print(l)


if __name__ == "__main__":
    main()
