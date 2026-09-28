#!/usr/bin/env python3
"""
Capítulo 4 — el terreno del error, con un peso (L24, A18).

La red entrenada del pasillo, moviendo solo la línea de la primera a la final. En horizontal, lo
que vale ese peso; en vertical, el error (la media de las cuatro posiciones). Encima, los pasos
cuesta abajo desde 2, moviendo solo ese peso, marcados cada 25. (Hubo un segundo cuadro con el
mapa de dos pesos; se quitó en la segunda vuelta de L24: el texto no lo usaba y no se leía solo.)

Todos los números se leen de `datos/salidas/culpa_hacia_atras_datos.csv` (bloque 8 de
`culpa_hacia_atras.txt`).

Uso:
    python figura_el_terreno.py --selftest
    python figura_el_terreno.py
"""

# ======================= CONSTANTES =======================

DATOS = "../datos/salidas/culpa_hacia_atras_datos.csv"
SALIDA = "../datos/salidas/culpa_hacia_atras.txt"
DESTINO = "../figuras/el_terreno.png"
ALTO = 3.3                     # pulgadas
CADA_CUANTOS = 25              # el mismo que culpa_hacia_atras.CADA_CUANTOS
NIVELES = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# ==========================================================

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def leer(ruta):
    terreno, camino, mapa = [], [], []
    peso_final = {}
    with open(ruta, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            if f["que"] == "terreno":
                terreno.append((float(f["a"]), float(f["valor"])))
            elif f["que"] == "camino":
                camino.append((int(f["a"]), float(f["b"]), float(f["valor"])))
            elif f["que"] == "mapa":
                mapa.append((float(f["a"]), float(f["b"]), float(f["valor"])))
            elif f["que"] == "final_peso_final":
                peso_final[int(f["a"])] = float(f["valor"])
    assert terreno and camino and mapa and len(peso_final) == 2, "faltan datos del terreno"
    t = np.array(terreno)
    c = np.array(sorted(camino))
    m = np.array(mapa)
    ejes = np.unique(m[:, 0])
    z = np.zeros((len(ejes), len(ejes)))
    idx = {v: k for k, v in enumerate(ejes)}
    for a, b, e in m:
        z[idx[b], idx[a]] = e
    return t, c, ejes, z, peso_final


def coma(x, d=1):
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def pie_abajo(L, s):
    """El pie, apoyado en el borde de abajo: con tres renglones, el de Lienzo.pie se sale."""
    L.ax.text(4, 1.0, s, ha="left", va="bottom", fontsize=6.8, color=L.p.suave, linespacing=1.4)


def dibujar(t, c, ejes, z, peso_final, paleta, ruta):
    L = Lienzo("El terreno del error, con un peso",
               "La red del pasillo, ya entrenada. Se mueve solo la línea de la primera a la final;\n"
               "los demás pesos se quedan quietos. La altura es el error: la media de las cuatro\n"
               "posiciones. Los puntos, los pasos cuesta abajo moviendo solo ese peso, uno cada 25.",
               paleta, alto=ALTO)
    p = L.p
    a = L.fig.add_axes([0.12, 0.17, 0.84, 0.52])
    a.plot(t[:, 0], t[:, 1], color=p.tinta, linewidth=1.5)
    marcas = c[::CADA_CUANTOS]
    a.plot(marcas[:, 1], marcas[:, 2], "o", markersize=4.2, markerfacecolor=p.acento,
           markeredgecolor=p.tinta, markeredgewidth=0.6, zorder=4)
    a.annotate("empieza aquí", xy=(marcas[0, 1], marcas[0, 2]), xytext=(marcas[0, 1] + 3.5, 0.47),
               fontsize=7.0, color=p.tinta, arrowprops=dict(arrowstyle="-", color=p.suave, lw=0.6))
    k = int(np.argmin(t[:, 1]))
    a.annotate("el fondo", xy=(t[k, 0], t[k, 1]), xytext=(t[k, 0] + 0.5, 0.30), fontsize=7.0,
               color=p.tinta, arrowprops=dict(arrowstyle="-", color=p.suave, lw=0.6))
    a.text(-5.6, 0.43, "llano: la culpa\nes casi cero", fontsize=7.0, color=p.suave, va="top",
           linespacing=1.1)
    a.set_xlabel("lo que vale el peso de la línea de la primera a la final", fontsize=7.4, color=p.tinta)
    a.set_ylabel("error", fontsize=7.4, color=p.tinta)
    a.set_ylim(-0.02, 0.53)
    a.set_yticks([0, 0.25, 0.5]); a.set_yticklabels(["0", "0,25", "0,5"])
    xt = [-5, 0, 5, 10, 15]
    a.set_xticks(xt); a.set_xticklabels([coma(x, 0) for x in xt])
    a.tick_params(labelsize=7.0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    L.guardar(ruta)


def selftest():
    fallos = []
    t, c, ejes, z, peso_final = leer(AQUI / DATOS)
    texto = (AQUI / SALIDA).read_text(encoding="utf-8")

    # 1. TEST NULO — en el tramo llano de la izquierda (pesos muy negativos) el error no cambia:
    #    ahí la culpa es casi cero, que es lo que dice el rótulo.
    llano = t[t[:, 0] < -4, 1]
    print(f"[1] test nulo         en el llano, el error va de {llano.min():.4f} a {llano.max():.4f}")
    if llano.max() - llano.min() > 0.005:
        fallos.append("test nulo: el tramo que la figura llama llano no lo es")

    # 2. SEÑAL — los pasos cuesta abajo bajan siempre, y lo que la figura marca es lo que imprime
    #    el bloque 8 de la salida.
    baja = bool(np.all(np.diff(c[:, 2]) <= 1e-15))
    marcas = [f"{x:.4f}".replace(".", ",") for x in c[::CADA_CUANTOS, 2]]
    faltan = [m for m in marcas if m not in texto]
    print(f"[2] señal             el camino siempre baja: {baja}; marcas que no están en la salida: {faltan or 'ninguna'}")
    if not baja or faltan:
        fallos.append("señal: el camino sube en algún paso, o sus marcas no son las de la salida")

    # 3. INVARIANTE — el punto donde lo dejó el entrenamiento está en la zona más baja del mapa, y
    #    la figura se dibuja en las dos paletas.
    i = int(np.argmin(np.abs(ejes - peso_final[1]))); j = int(np.argmin(np.abs(ejes - peso_final[0])))
    bajo = z[i, j] <= np.quantile(z, 0.10)
    for pal in (COLOR, GRIS):
        dibujar(t, c, ejes, z, peso_final, pal, "/dev/null")
    print(f"[3] invariante        error del mapa donde lo dejó el entrenamiento: {z[i, j]:.4f} (entre el 10 % más bajo: {bajo})")
    if not bajo:
        fallos.append("invariante: el punto entrenado no está en lo más bajo del mapa")
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
    dibujar(*leer(AQUI / DATOS), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
