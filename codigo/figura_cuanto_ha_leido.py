#!/usr/bin/env python3
"""
Capítulo 10 — cuánto texto lleva leído la máquina en cada retrato (L24, D03).

«Tras 30 pasos» no se puede poner a escala si no se sabe qué es un paso. En un paso la máquina
lee un puñado de trozos de texto, adivina cada letra y se corrige una vez. Esta figura enseña,
para cada retrato, cuánto lleva leído, medido en «el texto entero»: los ocho millones de letras
de los libros. Un cuadrado es el texto entero leído una vez. Sin cifras en el dibujo (regla 4
del capítulo): se ve.

Las cantidades NO se escriben aquí: se leen del bloque 1 de `datos/salidas/cuatro_retratos.txt`.

Uso:
    python figura_cuanto_ha_leido.py --selftest
    python figura_cuanto_ha_leido.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/cuatro_retratos.txt"
DESTINO = "../figuras/cuanto_ha_leido.png"
ALTO = 2.3                  # pulgadas
CASI_NADA = 0.1             # por debajo de esto, el relleno no se ve y se dice
LADO = 2.3                  # lado de un cuadrado, en unidades del lienzo
HUECO = 0.55
POR_FILA = 24
X0 = 27.0                   # donde empiezan los cuadrados

# ==========================================================

import argparse
import re
import sys

from matplotlib.patches import Rectangle

from infografia import COLOR, GRIS, Lienzo


def leer(ruta=SALIDA):
    t = open(ruta, encoding="utf-8").read()
    assert "--- 1. LO QUE LEE UN PASO DE APRENDIZAJE ---" in t, f"falta el bloque 1 en {ruta}"
    b = t.split("--- 1. LO QUE LEE UN PASO DE APRENDIZAJE ---", 1)[1].split("--- 2.", 1)[0]
    filas = re.findall(r"^\s*([\d.]+) pasos\s+([\d.]+)\s+([\d,]+)\s*$", b, re.M)
    assert len(filas) == 4, f"se esperaban cuatro retratos; hay {len(filas)}"
    m = re.search(r"\(el texto entero: ([\d.]+) letras\)", b)
    assert m, "falta el tamaño del texto entero"
    entero = int(m.group(1).replace(".", ""))
    fuera = []
    for p, letras, v in filas:
        veces = int(letras.replace(".", "")) / entero
        assert abs(veces - float(v.replace(",", "."))) < 0.006, f"las dos columnas no cuadran en {p}"
        fuera.append((p, veces))
    return fuera


def dibujar(filas, paleta, ruta):
    L = Lienzo("Cuánto lleva leído en cada foto",
               "Un cuadrado: el texto entero de los libros, leído una vez. En oscuro, lo que\nlleva leído; un cuadrado a medio llenar es una parte del texto.",
               paleta, alto=ALTO)
    p, ax = L.p, L.ax
    y = L.y - 2.0
    dibujados = {}
    for pasos, veces in filas:
        L.texto(4, y, f"tras {pasos} pasos", tam=7.8, negrita=True)
        llenos, parte = int(veces), veces - int(veces)
        n = llenos + (1 if parte > 0 else 0)
        filas_c = (n + POR_FILA - 1) // POR_FILA
        for i in range(n):
            fx, fy = i % POR_FILA, i // POR_FILA
            x = X0 + fx * (LADO + HUECO)
            yy = y - LADO / 2 - fy * (LADO + HUECO)
            frac = 1.0 if i < llenos else parte
            ax.add_patch(Rectangle((x, yy), LADO, LADO, facecolor="white", edgecolor=p.suave, linewidth=0.5))
            ax.add_patch(Rectangle((x, yy), LADO * frac, LADO, facecolor=p.acento, edgecolor="none"))
        if veces < CASI_NADA:
            L.texto(X0 + LADO + 2.0, y, "casi nada: la parte oscura no llega a verse", tam=7.0, color=p.suave)
        dibujados[pasos] = round(llenos + parte, 2)
        y -= max(1, filas_c) * (LADO + HUECO) + 2.6
    assert y > 1.0, f"no cabe: sobran {1 - y:.1f} unidades"
    L.guardar(ruta)
    return dibujados


def selftest():
    fallos = []
    filas = leer()
    # 1. TEST NULO — con cero veces leído no se rellena nada.
    rec = dibujar([(p, 0.0) for p, _ in filas], GRIS, "/dev/null")
    print(f"[1] test nulo         con nada leído, cuadrados rellenos: {sum(rec.values())}")
    if sum(rec.values()) != 0:
        fallos.append("test nulo: rellena cuadrados sin haber leído nada")
    # 2. SEÑAL — lo dibujado es lo que dice la salida.
    rec = dibujar(filas, GRIS, "/dev/null")
    ok = all(abs(rec[p] - round(v, 2)) < 1e-9 for p, v in filas)
    print(f"[2] señal             dibujado {rec}")
    if not ok:
        fallos.append("señal: lo dibujado no es lo que dice la salida")
    # 3. INVARIANTE — lo leído crece tantas veces como los pasos: en cada paso lee lo mismo.
    pasos = [int(p.replace(".", "")) for p, _ in filas]
    razones = [(filas[i + 1][1] / filas[i][1]) / (pasos[i + 1] / pasos[i]) for i in range(3)]
    for pal in (COLOR, GRIS):
        dibujar(filas, pal, "/dev/null")
    print(f"[3] invariante        lo leído crece igual que los pasos: {['%.3f' % r for r in razones]}")
    if any(abs(r - 1) > 1e-9 for r in razones):
        fallos.append("invariante: lo leído no crece igual que los pasos")
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
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    dibujar(leer(), GRIS, DESTINO)
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
