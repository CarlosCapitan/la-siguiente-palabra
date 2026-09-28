#!/usr/bin/env python3
"""
Capítulo 4 — la rampa corta, con las tres neuronas del ejemplo del pasillo encima (L24, segunda
vuelta, problema 5 de la verificación).

En el ejemplo (paso 2.500, los dos interruptores subidos) cada neurona pasa lo que le llega por su
rampa, y hacia atrás la culpa se multiplica por lo que esa rampa deja pasar en ese punto. La figura
pone las tres en su sitio de la rampa: la primera, en lo plano de arriba (deja pasar casi nada); la
segunda, subiendo; la final, cerca del centro. La inclinación marcada en cada punto es lo que deja
pasar.

Los números se leen del bloque 4 de `datos/salidas/culpa_hacia_atras.txt` (tabla «HACIA DELANTE» y
las líneas de «lo que deja pasar su rampa»); la curva es la función del programa, y el selftest
comprueba que cada punto está en ella y que la inclinación marcada es lo que dice por lo que le
falta para 1.

Uso:
    python figura_la_rampa_del_ejemplo.py --selftest
    python figura_la_rampa_del_ejemplo.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/culpa_hacia_atras.txt"
DESTINO = "../figuras/la_rampa_del_ejemplo.png"
ALTO = 3.0                     # pulgadas
LADO_TANGENTE = 0.9

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from infografia import COLOR, GRIS, Lienzo
from retropropagacion import sigmoide

AQUI = Path(__file__).resolve().parent
NOMBRES = ["la primera", "la segunda", "la final"]


def num(s):
    return float(s.replace("+", "").replace(",", "."))


def leer(ruta):
    texto = Path(ruta).read_text(encoding="utf-8")
    bloque = texto.split("4. UN EJEMPLO A MEDIO ENTRENAR", 1)[1].split("LA MISMA CULPA", 1)[0]
    puntos = {}
    for n in NOMBRES:
        m = re.search(rf"^{n}\s+(-?\d+,\d+)\s+(-?\d+,\d+)\s+(-?\d+,\d+)\s+(\d+,\d+)\s*$", bloque, re.M)
        assert m, f"no encuentro la fila de «{n}» en la tabla hacia delante"
        puntos[n] = [num(m.group(3)), num(m.group(4))]
    m = re.search(r"le falta para 1: \d+,\d+ por \d+,\d+ da (\d+,\d+)", bloque)
    puntos["la final"].append(num(m.group(1)))
    m = re.search(r"^   lo que deja pasar su rampa\s+(\d+,\d+)\s+(\d+,\d+)", bloque, re.M)
    puntos["la primera"].append(num(m.group(1)))
    puntos["la segunda"].append(num(m.group(2)))
    return puntos           # nombre: [pasa del listón por, dice, deja pasar]


def coma(x):
    return f"{x:.4g}".replace(".", ",")


def dibujar(puntos, paleta, ruta):
    L = Lienzo("La rampa, en el ejemplo",
               "Las tres neuronas del ejemplo, cada una en su sitio de la rampa. La inclinación\n"
               "en cada punto es lo que deja pasar: lo que dice, por lo que le falta para 1.",
               paleta, alto=ALTO)
    p = L.p
    ax = L.fig.add_axes([0.11, 0.17, 0.84, 0.50])
    xs = np.linspace(-7.5, 7.5, 400)
    ax.plot(xs, sigmoide(xs), color=p.tinta, linewidth=1.6)
    textos = {"la primera": (4.6, 0.55), "la segunda": (-2.7, 0.93), "la final": (-4.3, 0.55)}
    for n, (x, y, pendiente) in puntos.items():
        t = np.array([x - LADO_TANGENTE, x + LADO_TANGENTE])
        ax.plot(t, y + pendiente * (t - x), color=p.acento, linewidth=3.0, alpha=0.55,
                solid_capstyle="round")
        ax.plot([x], [y], "o", color=p.tinta, markersize=4)
        ax.annotate(f"{n}: dice {coma(y)}\ndeja pasar {coma(pendiente)}", xy=(x, y), xytext=textos[n],
                    fontsize=7.0, ha="center", va="center", color=p.tinta, linespacing=1.15,
                    arrowprops=dict(arrowstyle="-", color=p.suave, lw=0.6))
    ax.set_xlabel("cuánto pasa del listón lo que le llega", fontsize=7.4, color=p.tinta)
    ax.set_ylabel("lo que dice", fontsize=7.4, color=p.tinta)
    ax.set_xticks([-6, -4, -2, 0, 2, 4, 6]); ax.set_xticklabels(["−6", "−4", "−2", "0", "+2", "+4", "+6"])
    ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "0,5", "1"])
    ax.set_ylim(-0.08, 1.12)
    ax.tick_params(labelsize=7.0)
    ax.axvline(0, color=p.marco, linewidth=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    L.guardar(ruta)


def selftest():
    fallos = []
    puntos = leer(AQUI / SALIDA)

    # 1. TEST NULO — la regla «lo que dice, por lo que le falta para 1» da cero en los extremos de
    #    la rampa: una neurona que dijera 0 o 1 no dejaría pasar nada.
    nulo = [d * (1 - d) for d in (0.0, 1.0)]
    print(f"[1] test nulo         con 0 y con 1, la regla da {nulo}")
    if any(nulo):
        fallos.append("test nulo")

    # 2. SEÑAL — cada punto está en la rampa (lo que dice es la rampa de lo que le llega).
    peor = max(abs(float(sigmoide(x)) - y) for x, y, _ in puntos.values())
    print(f"[2] señal             puntos contra la curva: diferencia máxima {peor:.4f}")
    if peor > 0.006:
        fallos.append("señal: algún punto no está en la rampa")

    # 3. INVARIANTE — la inclinación marcada es lo que dice por lo que le falta para 1 (con los
    #    números impresos), y se dibuja en las dos paletas.
    peor2 = max(abs(y * (1 - y) - s) / s for _, y, s in puntos.values())
    for pal in (COLOR, GRIS):
        dibujar(puntos, pal, "/dev/null")
    print(f"[3] invariante        inclinación contra la regla: {100 * peor2:.2f} % como mucho")
    if peor2 > 0.01:
        fallos.append("invariante: la inclinación no es lo que dice por lo que le falta para 1")
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
    dibujar(leer(AQUI / SALIDA), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
