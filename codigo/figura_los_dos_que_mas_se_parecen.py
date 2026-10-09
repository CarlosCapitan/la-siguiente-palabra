#!/usr/bin/env python3
"""
Capítulo 3 — los dos comités de en medio que más se parecen, uno al lado del otro (L24, A03).

La pareja y su parecido (−0,44, con la cuenta del cuatro y el nueve) se leen del último bloque de
`datos/salidas/que_mira_cada_una.txt`. Los pesos salen de la misma red que dibuja la figura de los
ocho comités (`que_mira_cada_una.entrenar_con_capa`, misma semilla). Tres cuadros: el primero de la
pareja, el segundo, y el segundo en negativo (blanco donde era negro), que es con el que hay que
comparar al primero cuando el parecido sale negativo.

Uso:
    python figura_los_dos_que_mas_se_parecen.py --selftest
    python figura_los_dos_que_mas_se_parecen.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/que_mira_cada_una.txt"
DESTINO = "../figuras/los_dos_que_mas_se_parecen.png"
ALTO = 2.6                    # pulgadas

# ==========================================================

import argparse
from formato import coma, leer_tablas
import re
import sys
from pathlib import Path

import numpy as np

from infografia import COLOR, GRIS, Lienzo
from que_mira_cada_una import entrenar_con_capa, LADO

AQUI = Path(__file__).resolve().parent


def leer(ruta):
    # Desde el 9 de octubre de 2026, de la tabla editorial, por su título.
    titulo = "Cuánto se parecen, con la cuenta del cuatro y el nueve"
    T = leer_tablas(Path(ruta).read_text(encoding="utf-8"))
    assert titulo in T, f"no encuentro la tabla «{titulo}» en {ruta}"
    _, filas, _ = T[titulo]
    m = re.fullmatch(r"el comité (\d+) y el comité (\d+)", filas[0][0])
    assert m and len(filas) == 1, f"no encuentro la pareja que más se parece en {ruta}"
    return int(m.group(1)), int(m.group(2)), float(filas[0][1].replace(",", "."))


def dibujar(W, a, b, r, paleta, ruta):
    L = Lienzo(f"Los comités {a} y {b}",
               f"Los dos de en medio que más se parecen: {str(r).replace('.', ',').replace('-', '−')}. En medio, el {b} en\n"
               f"negativo, al lado del {a}. Negro: tinta ahí sube el número de ese comité (en el\n"
               f"cuadro del medio, lo baja); blanco: al revés.",
               paleta, alto=ALTO)
    lim = float(np.abs(W).max())
    cuadros = [(W[:, a - 1], f"el comité {a}"), (-W[:, b - 1], f"el comité {b},\nen negativo"),
               (W[:, b - 1], f"el comité {b}")]
    for k, (w, rotulo) in enumerate(cuadros):
        ax = L.fig.add_axes([0.08 + 0.31 * k, 0.19, 0.22, 0.22 * 4.45 / ALTO])
        ax.imshow(w.reshape(LADO, LADO), cmap="gray_r", vmin=-lim, vmax=lim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(rotulo, fontsize=7.8, color=L.p.tinta, linespacing=1.05)
    L.guardar(ruta)


def selftest():
    fallos = []
    a, b, r = leer(AQUI / SALIDA)
    W = entrenar_con_capa()["red"].W[0]

    # 1. TEST NULO — un comité comparado con su propio negativo da exactamente −1.
    nulo = float(np.corrcoef(W[:, a - 1], -W[:, a - 1])[0, 1])
    print(f"[1] test nulo         un comité contra su negativo: {coma(nulo, 3)} (tiene que ser -1)")
    if abs(nulo + 1) > 1e-9:
        fallos.append("test nulo: un comité contra su negativo no da -1")

    # 2. SEÑAL — la pareja de la salida, con los pesos de esta red, da el parecido de la salida.
    rr = float(np.corrcoef(W[:, a - 1], W[:, b - 1])[0, 1])
    print(f"[2] señal             comités {a} y {b}: {coma(rr, 2)} (la salida dice {coma(r, 2)})")
    if abs(rr - r) > 0.006:
        fallos.append("señal: la pareja dibujada no tiene el parecido que dice la salida")

    # 3. INVARIANTE — ninguna otra pareja se parece más (sin mirar el signo), y se dibuja en las
    #    dos paletas.
    C = np.abs(np.corrcoef(W.T)); np.fill_diagonal(C, 0)
    mayor = float(C.max())
    for pal in (COLOR, GRIS):
        dibujar(W, a, b, r, pal, "/dev/null")
    print(f"[3] invariante        el parecido más grande entre dos comités: {coma(mayor, 2)}")
    if abs(mayor - abs(r)) > 0.006:
        fallos.append("invariante: hay otra pareja que se parece más")
    print()
    if fallos:
        for x in fallos:
            print("FALLA:", x)
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
    a, b, r = leer(AQUI / SALIDA)
    dibujar(entrenar_con_capa()["red"].W[0], a, b, r, GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
