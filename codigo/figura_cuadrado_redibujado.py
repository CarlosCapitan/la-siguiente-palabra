#!/usr/bin/env python3
"""
Capítulos 3 y 4 — el cuadrado del pasillo, antes y después de la capa de en medio (L24, A04,
A10 y A11).

Dos figuras con el mismo dibujo:

  cuadrado_redibujado.png (capítulo 4). A la izquierda, el cuadrado del capítulo 2: los ejes son
      los dos interruptores y las dos esquinas que encienden la luz están en diagonal, sin raya
      posible. A la derecha, las mismas cuatro posiciones colocadas según lo que dicen las dos
      neuronas de en medio de la red entrenada, con la raya de la neurona final y su regla escrita.
      Las coordenadas se leen de `datos/salidas/culpa_hacia_atras_datos.csv` (las mismas cifras
      que el bloque 3 de `culpa_hacia_atras.txt`).

  cuadrado_a_mano.png (capítulo 3). Lo mismo con los tres comités puestos a mano al final del
      capítulo 2: «con al menos uno subido» y «con al menos uno bajado», y el tercero, que
      enciende la luz si los dos dicen que sí. Los síes y noes salen de los montajes de
      `perceptron.NOMBRES`, no se escriben aquí.

Uso:
    python figura_cuadrado_redibujado.py --selftest
    python figura_cuadrado_redibujado.py
"""

# ======================= CONSTANTES =======================

DATOS = "../datos/salidas/culpa_hacia_atras_datos.csv"
DESTINO_4 = "../figuras/cuadrado_redibujado.png"
DESTINO_3 = "../figuras/cuadrado_a_mano.png"
ALTO = 3.45                     # pulgadas
MONTAJES_A_MANO = ["con al menos uno subido", "con al menos uno bajado"]
LISTON_A_MANO = 1.5            # el tercero: caso 1 a cada uno y listón 1,5 (capítulo 2)
POSICIONES = ["ninguno subido", "solo el de arriba", "solo el de abajo", "los dos subidos"]
ENCIENDE = [False, True, True, False]
SOLAPE = 0.08                  # dos puntos más cerca que esto están en el mismo sitio

# ==========================================================

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from infografia import COLOR, GRIS, Lienzo
from perceptron import NOMBRES

AQUI = Path(__file__).resolve().parent
INTERRUPTORES = [(0, 0), (0, 1), (1, 0), (1, 1)]    # (el de abajo, el de arriba), orden de TABLA_XOR


def leer(ruta):
    d = {}
    with open(ruta, encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            if fila["que"].startswith("final_"):
                d[(fila["que"], fila["a"], fila["b"])] = float(fila["valor"])
    medio = np.array([[d[("final_dice_medio", str(p), str(j))] for j in range(2)] for p in range(4)])
    raya = (d[("final_peso_final", "0", "")], d[("final_peso_final", "1", "")],
            d[("final_liston_final", "", "")])
    return medio, raya


def a_mano():
    """Lo que dicen los dos comités del capítulo 2 en cada posición (0 o 1), según su montaje."""
    por_nombre = {v: k for k, v in NOMBRES.items()}
    return np.array([[por_nombre[m][p] for m in MONTAJES_A_MANO] for p in range(4)], dtype=float)


def en_el_mismo_sitio(puntos):
    return [(a, b) for a in range(4) for b in range(a + 1, 4)
            if np.hypot(*(puntos[a] - puntos[b])) < SOLAPE]


def ejes(L, rect, xlab, ylab, titulo):
    ax = L.fig.add_axes(rect)
    ax.set_xlim(-0.35, 1.35); ax.set_ylim(-0.35, 1.35)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.tick_params(labelsize=7.2)
    ax.set_xlabel(xlab, fontsize=7.3, color=L.p.tinta, linespacing=1.1)
    ax.set_ylabel(ylab, fontsize=7.3, color=L.p.tinta, linespacing=1.1)
    ax.set_title(titulo, fontsize=8.6, fontweight="bold", color=L.p.tinta, pad=4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_aspect("equal")
    return ax


def puntos(ax, xy, p):
    """Cada posición, con su número (el del capítulo 2: 1 ninguno subido, 2 solo el de arriba,
    3 solo el de abajo, 4 los dos subidos). Las que caen juntas llevan los dos números."""
    juntos = en_el_mismo_sitio(xy)
    hechos = set()
    for k in range(4):
        ax.plot(xy[k, 0], xy[k, 1], "o", markersize=10, markerfacecolor=p.tinta if ENCIENDE[k] else "white",
                markeredgecolor=p.tinta, markeredgewidth=1.4, zorder=4)
    for k in range(4):
        if k in hechos:
            continue
        par = [b for a, b in juntos if a == k]
        texto = str(k + 1) if not par else f"{k + 1} y {par[0] + 1}"
        hechos.update(par)
        x, y = xy[k]
        dx = 0.13 if x > 0.5 else -0.13
        dy = 0.13 if y > 0.5 else -0.13
        ax.text(x + dx, y + dy, texto, ha="left" if dx > 0 else "right", va="center",
                fontsize=7.6, fontweight="bold", color=p.tinta, zorder=5)
    return juntos


CLAVE_POSICIONES = ("1: ninguno subido  ·  2: solo el de arriba  ·  3: solo el de abajo  ·  "
                    "4: los dos subidos")


def dibujar_4(medio, raya, paleta, ruta):
    L = Lienzo("El mismo cuadrado, antes y después",
               "Negro: la luz tiene que encenderse. Blanco: tiene que quedarse apagada.\n"
               + CLAVE_POSICIONES + "\nA la derecha, en gris, donde la final enciende la luz: "
               "la primera dice sí y la segunda no.", paleta, alto=ALTO)
    p = L.p
    si_no = {0: "bajado", 1: "subido"}
    a = ejes(L, [0.14, 0.20, 0.30, 0.50], "el interruptor de abajo", "el interruptor de arriba",
             "antes: los interruptores\n(ninguna raya deja solas las negras)")
    a.set_xticklabels([si_no[0], si_no[1]]); a.set_yticklabels([si_no[0], si_no[1]])
    puntos(a, np.array(INTERRUPTORES, dtype=float), p)
    b = ejes(L, [0.63, 0.20, 0.30, 0.50], "la primera:\n«¿hay al menos uno subido?»",
             "la segunda:\n«¿están los dos subidos?»", "después: las de en medio\n(una raya basta)")
    b.set_xticklabels(["no", "sí"]); b.set_yticklabels(["no", "sí"])
    w1, w2, liston = raya
    xs = np.linspace(-0.35, 1.35, 50)
    ys = raya_y(xs, w1, w2, liston)
    b.fill_between(xs, ys, -0.35, color=p.neutro, zorder=0)
    b.plot(xs, ys, "--", color=p.suave, linewidth=1.2, zorder=1)
    puntos(b, medio, p)
    L.pie("Izquierda, el cuadrado del capítulo 2. Derecha, cada posición según lo que dicen las dos de\n"
          "en medio de la red entrenada: las dos negras caen juntas, y la raya de la final las deja solas.")
    L.guardar(ruta)
    return en_el_mismo_sitio(medio)


def dibujar_3(xy, paleta, ruta):
    L = Lienzo("Los tres comités del pasillo",
               "Las cuatro posiciones, colocadas según lo que dicen los dos comités de abajo.\n"
               "Negro: la luz tiene que encenderse. Blanco: tiene que quedarse apagada.\n"
               + CLAVE_POSICIONES, paleta, alto=3.3)
    p = L.p
    b = ejes(L, [0.33, 0.28, 0.38, 0.40], f"el primero:\n«{MONTAJES_A_MANO[0]}»",
             f"el segundo:\n«{MONTAJES_A_MANO[1]}»", "")
    b.set_xticklabels(["no", "sí"]); b.set_yticklabels(["no", "sí"])
    xs = np.linspace(-0.35, 1.35, 50)
    ys = LISTON_A_MANO - xs                # los dos con caso 1: el total vale el listón en la raya
    b.fill_between(xs, ys, 1.35, color=p.neutro, zorder=0)
    b.plot(xs, ys, "--", color=p.suave, linewidth=1.2, zorder=1)
    puntos(b, xy, p)
    L.pie("Gris: el tercero enciende la luz, «si los dos dicen que sí». Las dos negras caen en la misma\n"
          "esquina, la de «sí, sí», y la raya de trazos las deja solas.")
    L.guardar(ruta)
    return en_el_mismo_sitio(xy)


def raya_y(xs, w1, w2, liston):
    """La raya de la final: los puntos donde lo que le llega vale justo su listón."""
    return (liston - w1 * xs) / w2


def lado(xy, w1, w2, liston):
    return [bool(w1 * x + w2 * y > liston) for x, y in xy]


def selftest():
    fallos = []
    medio, raya = leer(AQUI / DATOS)
    mano = a_mano()

    # 1. TEST NULO — en el cuadrado de los interruptores ninguna posición cae encima de otra, y la
    #    raya de la final, aplicada a él, no deja las dos negras solas (ahí no hay raya).
    juntos = en_el_mismo_sitio(np.array(INTERRUPTORES, dtype=float))
    print(f"[1] test nulo         en el cuadrado de los interruptores, posiciones juntas: {juntos or 'ninguna'}")
    if juntos:
        fallos.append("test nulo: en el cuadrado de siempre dos posiciones caen en el mismo sitio")

    # 2. SEÑAL — con las de en medio como ejes, las dos negras caen juntas y la raya de la final
    #    las separa de las blancas; y lo mismo con los comités a mano.
    j4, j3 = en_el_mismo_sitio(medio), en_el_mismo_sitio(mano)
    ok4 = lado(medio, *raya) == ENCIENDE
    ok3 = lado(mano, 1.0, 1.0, LISTON_A_MANO) == ENCIENDE
    print(f"[2] señal             juntas: red {j4}, a mano {j3}; la raya separa: red {ok4}, a mano {ok3}")
    if j4 != [(1, 2)] or j3 != [(1, 2)] or not (ok4 and ok3):
        fallos.append("señal: las negras no caen juntas o la raya no las separa")

    # 3. INVARIANTE — los comités a mano dicen lo de la tabla del capítulo 2 (ninguno: no, sí; uno:
    #    sí, sí; los dos: sí, no), y las dos figuras se dibujan en las dos paletas.
    tabla = [[0, 1], [1, 1], [1, 1], [1, 0]]
    xs = np.linspace(0, 1, 5)
    en_la_raya = max(abs(raya[0] * x + raya[1] * y - raya[2]) for x, y in zip(xs, raya_y(xs, *raya)))
    print(f"[3] invariante        la raya dibujada está donde la final vale su listón: {en_la_raya:.1e}")
    if en_la_raya > 1e-9:
        fallos.append("invariante: la raya dibujada no es la de la final")
    for pal in (COLOR, GRIS):
        dibujar_4(medio, raya, pal, "/dev/null"); dibujar_3(mano, pal, "/dev/null")
    print(f"[3] invariante        los comités a mano: {mano.astype(int).tolist()}")
    if mano.astype(int).tolist() != tabla:
        fallos.append("invariante: los comités a mano no dicen lo de la tabla del capítulo 2")
    print()
    if fallos:
        for x in fallos:
            print("FALLA:", x)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def tabla_a_mano():
    """La tabla de los tres comités del final del capítulo 2, calculada con sus montajes. Desde
    el 9 oct 2026, tabla de libro (formato.py; REGLAS 6 ter)."""
    from formato import tabla_editorial
    mano = a_mano()
    si_no = {0: "no", 1: "sí"}
    filas = [("ninguno subido", 0), ("uno subido, cualquiera", 1), ("los dos subidos", 3)]
    assert (mano[1] == mano[2]).all(), "las dos posiciones de un solo interruptor deberían coincidir"
    T = []
    for nombre, k in filas:
        a, b = int(mano[k, 0]), int(mano[k, 1])
        luz = "encendida" if (a + b) > LISTON_A_MANO else "apagada"
        assert (luz == "encendida") == ENCIENDE[k], "el tercero no hace el o exclusivo"
        T.append([nombre, si_no[a], si_no[b], luz])
    print("\n".join(tabla_editorial(
        "Los tres comités del final del capítulo 2",
        ["los interruptores", "el primero", "el segundo", "la luz"], T, "iccc",
        [f"El primero: «{MONTAJES_A_MANO[0]}». El segundo: «{MONTAJES_A_MANO[1]}». El tercero "
         "enciende la luz si los dos dicen que sí."])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    medio, raya = leer(AQUI / DATOS)
    dibujar_4(medio, raya, GRIS, str(AQUI / DESTINO_4))
    dibujar_3(a_mano(), GRIS, str(AQUI / DESTINO_3))
    print(f"escritas {DESTINO_4} y {DESTINO_3}")
    print()
    tabla_a_mano()


if __name__ == "__main__":
    main()
