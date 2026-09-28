#!/usr/bin/env python3
"""
Capítulo 13 — cien papeletas a cuatro temperaturas (L24, E01).

El sorteo del capítulo 7 («La capital de Francia es», con la máquina de quinientos millones): de
cada cien papeletas, cuántas llevan «la», «una», «un» y cuántas cualquier otro trozo, a
temperatura 0; 1; 1,3 y 1,8. Una cuadrícula de diez por diez por temperatura: cada casilla es una
papeleta.

Los números NO se calculan aquí: se leen de las filas «urna» de
`datos/salidas/temperatura_a_mano.csv`, que escribe `temperatura_a_mano.py`. En gris: el libro se
imprime en negro; las cuatro clases se distinguen por tono y trama.

Uso:
    python figura_cien_papeletas.py --selftest
    python figura_cien_papeletas.py
"""

# ======================= CONSTANTES =======================

DATOS = "../datos/salidas/temperatura_a_mano.csv"
DESTINO = "../figuras/cien_papeletas.png"
ALTO = 2.85                # pulgadas
LADO = 1.75                # lado de una papeleta, en unidades del lienzo
HUECO = 0.25               # separación entre papeletas
CLASES = ("«la»", "«una»", "«un»", "otro trozo")

# ==========================================================

import argparse
import csv
import sys
from pathlib import Path

from matplotlib.patches import Rectangle

from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def leer(ruta=AQUI / DATOS):
    filas = []
    with open(ruta, encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            if f["parte"] == "urna":
                filas.append((float(f["temperatura"]), [int(f[k]) for k in "efgh"]))
    assert len(filas) == 4, f"se esperaban cuatro temperaturas; hay {len(filas)}"
    assert all(sum(p) == 100 for _, p in filas), "cada temperatura tiene que repartir cien papeletas"
    return filas


def estilos(p):
    # (relleno, trama) de cada clase: negro, gris medio, rayas, blanco
    return ((p.tinta, None), (p.contra, None), ("white", "////"), ("white", None))


def coma(t):
    return f"{t:.1f}".replace(".", ",").replace(",0", "") if t == int(t) else f"{t:.1f}".replace(".", ",")


def dibujar(filas, paleta, ruta):
    L = Lienzo("Cien papeletas a cuatro temperaturas",
               "«La capital de Francia es»: el sorteo del capítulo 7. Cada casilla es\n"
               "una papeleta; lo que lleva escrito, según la clave.",
               paleta, alto=ALTO)
    p, ax = L.p, L.ax
    est = estilos(p)
    # la clave
    yc = L.y - 0.8
    xs = (4, 18, 34, 48)
    for x, (rel, tr), nombre in zip(xs, est, CLASES):
        ax.add_patch(Rectangle((x, yc - 1.3), 2.6, 2.6, facecolor=rel, hatch=tr,
                               edgecolor=p.suave, linewidth=0.5))
        L.texto(x + 3.6, yc, nombre, tam=7.4, color=p.suave)
    top = yc - 5.5
    ancho_rej = 10 * (LADO + HUECO)
    sep = (92 - 4 * ancho_rej) / 3
    contadas = []
    for i, (t, pap) in enumerate(filas):
        x0 = 4 + i * (ancho_rej + sep)
        clases = [k for k, n in enumerate(pap) for _ in range(n)]
        contadas.append([clases.count(k) for k in range(4)])
        for j, k in enumerate(clases):
            fila, col = divmod(j, 10)
            rel, tr = est[k]
            ax.add_patch(Rectangle((x0 + col * (LADO + HUECO), top - (fila + 1) * (LADO + HUECO)),
                                   LADO, LADO, facecolor=rel, hatch=tr, edgecolor=p.suave,
                                   linewidth=0.35))
        yb = top - 10 * (LADO + HUECO) - 3.0
        L.texto(x0 + ancho_rej / 2, yb, f"temperatura {coma(t)}", tam=8.0, ha="center", negrita=True)
        L.texto(x0 + ancho_rej / 2, yb - 3.4, " · ".join(str(n) for n in pap), tam=7.2,
                ha="center", color=p.suave)
    L.pie("Debajo de cada cuadrícula, las papeletas de «la», «una», «un» y otro trozo, en ese orden.\n"
          "A temperatura 0 no hay sorteo; a 1,8, casi todas son de trozos que a 1 casi no tenían.")
    L.guardar(ruta)
    return contadas


def selftest():
    fallos = []
    filas = leer()
    # 1. TEST NULO — si una temperatura no tuviera más que papeletas de «otro trozo», la figura no
    #    dibuja ni una de las otras tres clases.
    c = dibujar([(t, [0, 0, 0, 100]) for t, _ in filas], GRIS, "/dev/null")
    print(f"[1] test nulo         cien papeletas de otro trozo, dibujadas de cada clase: {c[0]}")
    if c[0] != [0, 0, 0, 100]:
        fallos.append("test nulo: la figura dibuja clases que no están")
    # 2. SEÑAL IMPLANTADA — lo que cuenta el capítulo: a temperatura 0, las cien para «la»; y
    #    según sube, «la» pierde papeletas y «otro trozo» las gana.
    la = [p[0] for _, p in filas]
    otros = [p[3] for _, p in filas]
    ok = la[0] == 100 and all(a > b for a, b in zip(la, la[1:])) and all(a < b for a, b in zip(otros, otros[1:]))
    print(f"[2] señal implantada  «la»: {la}; otro trozo: {otros}")
    if not ok:
        fallos.append("señal: las papeletas no se mueven como cuenta el capítulo")
    # 3. INVARIANTE DEL DOMINIO — lo dibujado es lo leído, casilla a casilla, y cada cuadrícula
    #    tiene cien; en color y en gris.
    for pal in (COLOR, GRIS):
        c = dibujar(filas, pal, "/dev/null")
    igual = c == [p for _, p in filas]
    print(f"[3] invariante        lo dibujado es lo leído: {'sí' if igual else 'NO'}")
    if not igual:
        fallos.append("invariante: la figura no dibuja lo que trae la salida")
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
    dibujar(leer(), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
