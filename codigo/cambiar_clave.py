#!/usr/bin/env python3
"""
Cambia una clave de tabla en las salidas guardadas, sin volver a ejecutar nada, y demuestra que
no se ha tocado nada más.

Nace de la excepción de `REGLAS.md` (regla 6, 17 de octubre de 2026): un cambio de rótulo o de
clave que no toca ninguna cifra puede aplicarse a la vez al programa y a la salida guardada. Este
programa hace la parte de la salida: sustituye UNA cadena en cada `datos/salidas/*.txt` que la
contenga (hoy, «Cada casilla:» por «Cada número:»), y para cada fichero

  - cuenta las apariciones antes y después;
  - comprueba línea a línea que las únicas líneas que cambian son las que contenían la cadena, y
    que cada una es exactamente la anterior con la cadena sustituida; si cambia cualquier otra,
    revienta y no escribe nada;
  - imprime cuántas líneas ha cambiado.

También sirve para comprobar un cambio ya hecho: con `--contra CARPETA`, no escribe nada y mira si
la carpeta de salidas actual es la de CARPETA con la sustitución aplicada (por ejemplo, una copia
de las salidas de un commit anterior).

Uso:
    python cambiar_clave.py --selftest
    python cambiar_clave.py                       # aplica en ../datos/salidas
    python cambiar_clave.py --contra /ruta/vieja  # solo comprueba, no escribe
    python cambiar_clave.py --carpeta /ruta       # aplica en otra carpeta
"""

# ======================= CONSTANTES =======================

ANTES = "Cada casilla:"
DESPUES = "Cada número:"
CARPETA = "../datos/salidas"

# ==========================================================

import argparse
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def sustituir(texto, antes=ANTES, despues=DESPUES):
    """El texto con la cadena sustituida y cuántas líneas la contenían."""
    lineas = texto.split("\n")
    return texto.replace(antes, despues), sum(1 for l in lineas if antes in l)


def comprobar(viejo, nuevo, antes=ANTES, despues=DESPUES):
    """Revienta si `nuevo` no es `viejo` con la cadena sustituida, línea a línea: el mismo número
    de líneas, y solo difieren las que contenían la cadena. Devuelve cuántas difieren."""
    a, b = viejo.split("\n"), nuevo.split("\n")
    assert len(a) == len(b), f"Se esperaban {len(a)} líneas y hay {len(b)}"
    cambiadas = 0
    for i, (x, y) in enumerate(zip(a, b), 1):
        if x == y:
            assert antes not in x, f"línea {i}: tiene «{antes}» y no se ha cambiado"
            continue
        assert antes in x and y == x.replace(antes, despues), (
            f"línea {i}: ha cambiado sin contener «{antes}», o no solo por la sustitución:\n"
            f"  antes:   {x[:100]}\n  después: {y[:100]}")
        cambiadas += 1
    return cambiadas


def aplicar(carpeta):
    """Sustituye en cada .txt de la carpeta; escribe solo si la comprobación pasa."""
    total = 0
    for ruta in sorted(Path(carpeta).glob("*.txt")):
        viejo = ruta.read_text(encoding="utf-8")
        antes = viejo.count(ANTES)
        if not antes:
            continue
        nuevo, lineas = sustituir(viejo)
        cambiadas = comprobar(viejo, nuevo)
        assert cambiadas == lineas and nuevo.count(ANTES) == 0
        ruta.write_text(nuevo, encoding="utf-8")
        print(f"{ruta.name:<44}{antes} apariciones, {cambiadas} líneas cambiadas")
        total += cambiadas
    print(f"{'en total':<44}{total} líneas cambiadas")
    return total


def contra(actual, vieja):
    """Comprueba que `actual` es `vieja` con la sustitución: cada fichero, y que no sobra ni falta
    ninguno. No escribe nada."""
    a = {p.name: p for p in Path(actual).glob("*.txt")}
    v = {p.name: p for p in Path(vieja).glob("*.txt")}
    assert a.keys() >= v.keys(), f"faltan ficheros: {sorted(v.keys() - a.keys())}"
    total = distintos = 0
    for nombre in sorted(v):
        viejo = v[nombre].read_text(encoding="utf-8")
        nuevo = a[nombre].read_text(encoding="utf-8")
        if viejo == nuevo:
            continue
        distintos += 1
        esperado, _ = sustituir(viejo)
        cambiadas = comprobar(viejo, nuevo)
        assert nuevo == esperado, f"{nombre}: no es la versión vieja con la sustitución"
        print(f"{nombre:<44}{cambiadas} líneas cambiadas, solo esas")
        total += cambiadas
    print(f"{'en total':<44}{total} líneas cambiadas en {distintos} ficheros; "
          f"los otros {len(v) - distintos}, idénticos")
    return total


def selftest():
    fallos = []
    base = f"uno\n{ANTES} dos 3,5\ntres\n{ANTES} cuatro\n"

    # 1. TEST NULO — un texto sin la cadena no cambia, ni se cuenta nada.
    limpio = "uno\ndos 3,5\n"
    texto, n = sustituir(limpio)
    igual = "sí" if texto == limpio else "NO"
    print(f"[1] test nulo         sin la cadena: {n} líneas cambiadas, texto igual: {igual}")
    if n or texto != limpio:
        fallos.append("test nulo: cambia un texto que no tiene la cadena")

    # 2. SEÑAL IMPLANTADA — en un texto con la cadena en dos líneas, cambian esas dos y solo ellas,
    #    y las cifras quedan como estaban.
    nuevo, n = sustituir(base)
    cambiadas = comprobar(base, nuevo)
    print(f"[2] señal implantada  dos líneas con la cadena: {n} contadas, {cambiadas} cambiadas")
    if (n, cambiadas) != (2, 2) or ANTES in nuevo or "3,5" not in nuevo:
        fallos.append(f"señal implantada: se esperaban 2 y 2; salieron {n} y {cambiadas}")

    # 3. INVARIANTE — si cambia cualquier otra línea, o una cifra, comprobar revienta.
    malos = {"una línea de más": nuevo + "extra\n",
             "una cifra tocada": nuevo.replace("3,5", "3,6"),
             "una línea sin cadena cambiada": nuevo.replace("tres", "TRES"),
             "una con la cadena sin cambiar": base}
    sin_reventar = []
    for nombre, texto in malos.items():
        try:
            comprobar(base, texto)
            sin_reventar.append(nombre)
        except AssertionError:
            pass
    print(f"[3] invariante        {len(malos) - len(sin_reventar)} de {len(malos)} cambios indebidos "
          f"hacen reventar")
    if sin_reventar:
        fallos.append(f"invariante: no revienta con {sin_reventar}")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--contra", help="carpeta de salidas anterior: solo comprueba, no escribe")
    ap.add_argument("--carpeta", default=str(AQUI / CARPETA), help="dónde aplicar (por defecto, las salidas)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.contra:
        contra(args.carpeta, args.contra)
    else:
        aplicar(args.carpeta)


if __name__ == "__main__":
    main()
