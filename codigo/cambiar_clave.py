#!/usr/bin/env python3
"""
Cambia una o varias cadenas en las salidas guardadas, sin volver a ejecutar nada, y demuestra que
no se ha tocado nada más.

Nace de la excepción de `REGLAS.md` (regla 6, 17 de octubre de 2026): un cambio de rótulo o de
clave que no toca ninguna cifra puede aplicarse a la vez al programa y a la salida guardada. Este
programa hace la parte de la salida: sustituye las cadenas que se le dan (por defecto, una:
«Cada casilla:» por «Cada número:»; con `--cambio ANTES=>DESPUES`, las que se quiera, en ese
orden) en cada `datos/salidas/*.txt` que las contenga (o solo en los que se nombren con
`--fichero`), y para cada fichero

  - cuenta las apariciones antes y después;
  - comprueba línea a línea que las únicas líneas que cambian son las que contenían alguna de las
    cadenas, y que cada una es exactamente la anterior con las cadenas sustituidas; si cambia
    cualquier otra, revienta y no escribe nada;
  - imprime cuántas líneas ha cambiado.

También sirve para comprobar un cambio ya hecho: con `--contra CARPETA`, no escribe nada y mira si
la carpeta de salidas actual es la de CARPETA con la sustitución aplicada (por ejemplo, una copia
de las salidas de un commit anterior).

Uso:
    python cambiar_clave.py --selftest
    python cambiar_clave.py                       # aplica en ../datos/salidas
    python cambiar_clave.py --contra /ruta/vieja  # solo comprueba, no escribe
    python cambiar_clave.py --carpeta /ruta       # aplica en otra carpeta
    python cambiar_clave.py --fichero mirada_a_mano.txt \
        --cambio 'uno=>UNO' --cambio 'dos=>DOS'    # varias cadenas, en un fichero
"""

# ======================= CONSTANTES =======================

ANTES = "Cada casilla:"
DESPUES = "Cada número:"
CAMBIOS = [(ANTES, DESPUES)]       # lo que se cambia si no se dice otra cosa
CARPETA = "../datos/salidas"

# ==========================================================

import argparse
import sys
import tempfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def aplica(linea, cambios):
    """Una línea con todas las cadenas sustituidas, en orden."""
    for antes, despues in cambios:
        linea = linea.replace(antes, despues)
    return linea


def sustituir(texto, cambios=CAMBIOS):
    """El texto con las cadenas sustituidas y cuántas líneas tenían alguna."""
    lineas = texto.split("\n")
    return aplica(texto, cambios), sum(1 for l in lineas if any(a in l for a, _ in cambios))


def comprobar(viejo, nuevo, cambios=CAMBIOS):
    """Revienta si `nuevo` no es `viejo` con las cadenas sustituidas, línea a línea: el mismo
    número de líneas, y solo difieren las que contenían alguna. Devuelve cuántas difieren."""
    a, b = viejo.split("\n"), nuevo.split("\n")
    assert len(a) == len(b), f"Se esperaban {len(a)} líneas y hay {len(b)}"
    cambiadas = 0
    for i, (x, y) in enumerate(zip(a, b), 1):
        if x == y:
            assert not any(c in x for c, _ in cambios), f"línea {i}: tiene una cadena y no se ha cambiado"
            continue
        assert any(c in x for c, _ in cambios) and y == aplica(x, cambios), (
            f"línea {i}: ha cambiado sin contener ninguna de las cadenas, o no solo por la sustitución:\n"
            f"  antes:   {x[:100]}\n  después: {y[:100]}")
        cambiadas += 1
    return cambiadas


def aplicar(carpeta, cambios=CAMBIOS, ficheros=None):
    """Sustituye en cada .txt de la carpeta (o solo en `ficheros`); escribe solo si la
    comprobación pasa."""
    total = 0
    rutas = [Path(carpeta) / f for f in ficheros] if ficheros else sorted(Path(carpeta).glob("*.txt"))
    for ruta in rutas:
        viejo = ruta.read_text(encoding="utf-8")
        antes = sum(viejo.count(a) for a, _ in cambios)
        if not antes:
            assert not ficheros, f"{ruta.name}: no tiene ninguna de las cadenas"
            continue
        nuevo, lineas = sustituir(viejo, cambios)
        cambiadas = comprobar(viejo, nuevo, cambios)
        assert cambiadas == lineas and not any(a in nuevo for a, _ in cambios)
        ruta.write_text(nuevo, encoding="utf-8")
        print(f"{ruta.name:<44}{antes} apariciones, {cambiadas} líneas cambiadas")
        total += cambiadas
    print(f"{'en total':<44}{total} líneas cambiadas")
    return total


def contra(actual, vieja, cambios=CAMBIOS):
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
        esperado, _ = sustituir(viejo, cambios)
        cambiadas = comprobar(viejo, nuevo, cambios)
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
    # y con varias cadenas a la vez, en una línea y en otra
    varios = [("uno", "UNO"), ("cuatro", "CUATRO")]
    mult, n2 = sustituir(base, varios)
    if (n2, comprobar(base, mult, varios)) != (2, 2) or "UNO" not in mult or "CUATRO" not in mult:
        fallos.append(f"señal implantada: con varias cadenas salió {n2} líneas y {mult!r}")
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
    # con varias cadenas, tocar una línea que no contiene ninguna también revienta
    try:
        comprobar(base, mult.replace("tres", "TRES"), varios)
        sin_reventar.append("varias cadenas y una línea ajena tocada")
    except AssertionError:
        pass
    malos["varias cadenas y una línea ajena tocada"] = None
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
    ap.add_argument("--cambio", action="append", metavar="ANTES=>DESPUES",
                    help="una cadena y su sustitución; se puede repetir")
    ap.add_argument("--fichero", action="append", help="solo este fichero de la carpeta; se puede repetir")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    cambios = CAMBIOS
    if args.cambio:
        assert all("=>" in c for c in args.cambio), "cada --cambio es ANTES=>DESPUES"
        cambios = [tuple(c.split("=>", 1)) for c in args.cambio]
    if args.contra:
        contra(args.carpeta, args.contra, cambios)
    else:
        aplicar(args.carpeta, cambios, args.fichero)


if __name__ == "__main__":
    main()
