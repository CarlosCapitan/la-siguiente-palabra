#!/usr/bin/env python3
"""
Capítulos 14 y 15 — dos cuentas que el libro anunciaba y no hacía (L24, E16 y E25).

  1. EL EMPUJÓN, EN LOS DOS SENTIDOS (capítulo 14). La propuesta de Lillicrap y otros (2020) que
     cuenta el capítulo: cada unión multiplica la actividad de sus dos neuronas antes del empujón y
     después, y cambia según la diferencia. El capítulo hacía el caso en que la respuesta correcta
     estaba más arriba (0,5 y 0,4 antes; 0,6 y 0,6 después); aquí está también el contrario, con
     la respuesta correcta más abajo. Los números de actividad son de ejemplo, como los del
     capítulo: lo que se enseña es la cuenta, no una medición.
  2. EL PRODUCTO ESCALAR, HECHO UNA VEZ (capítulo 15). La comparación entre la pregunta de una
     palabra y la etiqueta de otra, con dos listas de tres números: se multiplica casilla a
     casilla y se suma. Una etiqueta que encaja con la pregunta y otra que no. Números de ejemplo.

Uso:
    python cuentas_a_mano.py --selftest
    python cuentas_a_mano.py > ../datos/salidas/cuentas_a_mano.txt
"""

# ======================= CONSTANTES =======================

# (en medio, salida) antes del empujón y después; «arriba» es el caso del capítulo 14.
EMPUJON = {
    "la respuesta correcta estaba más arriba": ((0.5, 0.4), (0.6, 0.6)),
    "la respuesta correcta estaba más abajo":  ((0.5, 0.4), (0.4, 0.2)),
}
PREGUNTA = (3, 1, 0)
ETIQUETAS = {"una etiqueta que encaja": (2, 1, 0), "una que no encaja": (0, 1, 3)}
SALIDA_CSV = "../datos/salidas/cuentas_a_mano.csv"

# ==========================================================

import argparse
import csv
import sys
from pathlib import Path

from formato import ANCHO_CAJA_CITA, coma, comprobar_ancho

AQUI = Path(__file__).resolve().parent


def cambio(antes, despues):
    return despues[0] * despues[1] - antes[0] * antes[1]


def escalar(a, b):
    return sum(x * y for x, y in zip(a, b))


def informe():
    filas = []
    print("--- 1. EL EMPUJÓN, EN LOS DOS SENTIDOS ---")
    lineas = []
    for caso, (antes, despues) in EMPUJON.items():
        d = cambio(antes, despues)
        lineas += [caso + ":",
                   f"  antes:   en medio {coma(antes[0], 1)}, salida {coma(antes[1], 1)}; "
                   f"producto {coma(antes[0] * antes[1], 2)}",
                   f"  después: en medio {coma(despues[0], 1)}, salida {coma(despues[1], 1)}; "
                   f"producto {coma(despues[0] * despues[1], 2)}",
                   f"  el producto {'sube' if d > 0 else 'baja'} {coma(abs(d), 2)}: la unión se "
                   f"{'refuerza' if d > 0 else 'debilita'}.",
                   ""]
        filas.append([caso, antes[0], antes[1], despues[0], despues[1], f"{d:.4f}"])
    lineas += ["«en medio», «salida»: la actividad de las dos neuronas",
               "que une la unión. «producto»: una multiplicada por otra."]
    for l in comprobar_ancho(["  " + l if l else l for l in lineas], ANCHO_CAJA_CITA):
        print(l)

    print("\n--- 2. EL PRODUCTO ESCALAR, HECHO UNA VEZ ---")
    lineas = [f"la pregunta de una palabra: {', '.join(str(x) for x in PREGUNTA)}", ""]
    for nombre, e in ETIQUETAS.items():
        productos = [x * y for x, y in zip(PREGUNTA, e)]
        lineas += [f"{nombre}: {', '.join(str(x) for x in e)}",
                   "  casilla a casilla: " + "; ".join(f"{x} por {y}: {x * y}"
                                                        for x, y in zip(PREGUNTA, e)),
                   f"  sumado: {' más '.join(str(p) for p in productos)} son {escalar(PREGUNTA, e)}",
                   ""]
        filas.append(["escalar " + nombre, *e, escalar(PREGUNTA, e), ""])
    lineas += ["cuanto mayor el total, más encajan la pregunta y la",
               "etiqueta. Son números de ejemplo."]
    for l in comprobar_ancho(["  " + l if l else l for l in lineas], ANCHO_CAJA_CITA):
        print(l)
    with open(AQUI / SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["caso", "a", "b", "c", "d", "e"]] + filas)


def selftest():
    fallos = []
    # 1. TEST NULO — si el empujón no mueve nada, la unión no cambia.
    nulo = cambio((0.5, 0.4), (0.5, 0.4))
    print(f"[1] test nulo         sin empujón, el producto cambia {coma(nulo, 2)}")
    if nulo != 0:
        fallos.append("test nulo: la unión cambia sin empujón")
    # 2. SEÑAL IMPLANTADA — el caso del capítulo 14 da 0,20 antes, 0,36 después, y refuerza; el
    #    contrario debilita.
    (a, d), (b, e) = EMPUJON.values()
    ok = (abs(a[0] * a[1] - 0.20) < 1e-9 and abs(d[0] * d[1] - 0.36) < 1e-9
          and cambio(a, d) > 0 and cambio(b, e) < 0)
    print(f"[2] señal implantada  el caso del capítulo: {coma(a[0] * a[1], 2)} y "
          f"{coma(d[0] * d[1], 2)}; arriba refuerza y abajo debilita: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("señal: los casos no dan lo que dice el capítulo")
    # 3. INVARIANTE DEL DOMINIO — el producto escalar no depende del orden de las dos listas, y
    #    la etiqueta que encaja da más que la que no.
    enc, no = ETIQUETAS.values()
    ok = (escalar(PREGUNTA, enc) == escalar(enc, PREGUNTA)
          and escalar(PREGUNTA, enc) > escalar(PREGUNTA, no))
    print(f"[3] invariante        el orden no importa y la que encaja da más: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: el producto escalar no se comporta como debe")
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
    if ap.parse_args().selftest:
        sys.exit(selftest())
    informe()


if __name__ == "__main__":
    main()
