#!/usr/bin/env python3
"""
Capítulo 4 — la rampa corta, con las tres neuronas del ejemplo del colegio encima (Orden 20).

En el ejemplo (paso 1.000, la edad de 13 años) cada neurona pasa lo que le llega por su rampa, y
hacia atrás la culpa se multiplica por lo que esa rampa deja pasar en ese punto. La figura pone las
tres en su sitio de la rampa: la primera, en lo plano de arriba (deja pasar casi nada); la segunda,
subiendo; la final, cerca del centro. La inclinación marcada en cada punto es lo que deja pasar.

Los números se leen de `datos/salidas/culpa_en_el_colegio_datos.csv`; la curva es la función del
programa, y el selftest comprueba que cada punto está en ella y que la inclinación marcada es lo que
dice por lo que le falta para 1.

Uso:
    python figura_la_rampa_del_colegio.py --selftest
    python figura_la_rampa_del_colegio.py
"""

# ======================= CONSTANTES =======================

DATOS = "../datos/salidas/culpa_en_el_colegio_datos.csv"   # los números, que escribe culpa_en_el_colegio.py
DESTINO = "../figuras/la_rampa_del_colegio.png"
ALTO = 3.0                     # pulgadas
LADO_TANGENTE = 0.4            # media longitud de cada tangente: con 0,9 la de la segunda y la de la final se montaban
NOMBRES = ["la primera", "la segunda", "la final"]

# ==========================================================

import argparse
import sys
from pathlib import Path

import numpy as np

from figura_la_red_del_colegio import leer_datos, v
from formato import coma as coma_fija
from infografia import COLOR, GRIS, Lienzo
from retropropagacion import sigmoide

AQUI = Path(__file__).resolve().parent


def puntos(d):
    """nombre: [pasa del listón por, dice, deja pasar] del ejemplo; lo que deja pasar, con las cuatro
    cifras que imprime la salida (mano_pasa_medio y mano_pasa_final)."""
    p = {}
    for j, n in enumerate(NOMBRES[:2]):
        p[n] = [v(d, "caso_pasa_medio", j), v(d, "caso_dice_medio", j), v(d, "mano_pasa_medio", j)]
    p["la final"] = [v(d, "caso_pasa_final"), v(d, "caso_dice_final"), v(d, "mano_pasa_final")]
    return p


def coma(x):
    return f"{x:.4g}".replace(".", ",")


def dibujar(d, paleta, ruta):
    puntos_ = puntos(d)
    L = Lienzo("La rampa, en el ejemplo",
               "Las tres neuronas del ejemplo, cada una en su sitio de la rampa. La inclinación\n"
               "en cada punto es lo que deja pasar: lo que dice, por lo que le falta para 1.",
               paleta, alto=ALTO)
    p = L.p
    ax = L.fig.add_axes([0.11, 0.17, 0.84, 0.50])
    xs = np.linspace(-7.5, 7.5, 400)
    ax.plot(xs, sigmoide(xs), color=p.tinta, linewidth=1.6)
    # la segunda arriba a la izquierda; la final y la primera debajo de la curva, a distinta altura
    textos = {"la primera": (5.4, 0.50), "la segunda": (-3.4, 0.93), "la final": (2.4, 0.20)}
    for n, (x, y, pendiente) in puntos_.items():
        t = np.array([x - LADO_TANGENTE, x + LADO_TANGENTE])
        ax.plot(t, y + pendiente * (t - x), color=p.acento, linewidth=3.0, alpha=0.55, solid_capstyle="round")
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
    d = leer_datos(AQUI / DATOS)
    puntos_ = puntos(d)

    # 1. TEST NULO — la regla «lo que dice, por lo que le falta para 1» da cero en los extremos de la
    #    rampa: una neurona que dijera 0 o 1 no dejaría pasar nada.
    nulo = [x * (1 - x) for x in (0.0, 1.0)]
    print(f"[1] test nulo         con 0 y con 1, la regla da {' y '.join(coma_fija(x, 1) for x in nulo)}")
    if any(nulo):
        fallos.append("test nulo")

    # 2. SEÑAL — cada punto está en la rampa (lo que dice es la rampa de lo que le llega).
    peor = max(abs(float(sigmoide(x)) - y) for x, y, _ in puntos_.values())
    print(f"[2] señal             puntos contra la curva: diferencia máxima {coma_fija(peor, 6)}")
    if peor > 1e-9:
        fallos.append("señal: algún punto no está en la rampa")

    # 3. INVARIANTE — la inclinación marcada es lo que dice por lo que le falta para 1 (con las cuatro
    #    cifras impresas), la primera deja pasar mucho menos que la segunda, y se dibuja en las dos
    #    paletas.
    peor2 = max(abs(y * (1 - y) - s) / s for _, y, s in puntos_.values())
    for pal in (COLOR, GRIS):
        dibujar(d, pal, "/dev/null")
    ordenadas = puntos_["la primera"][2] < puntos_["la segunda"][2] < 0.25 + 1e-9
    print(f"[3] invariante        inclinación contra la regla: {coma_fija(100 * peor2, 2)} % como mucho; "
          f"la primera deja pasar menos que la segunda, y ninguna pasa de 0,25: {'sí' if ordenadas else 'NO'}")
    if peor2 > 0.01 or not ordenadas:
        fallos.append("invariante: la inclinación no es lo que dice por lo que le falta para 1")
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
    dibujar(leer_datos(AQUI / DATOS), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
