#!/usr/bin/env python3
"""
Capítulo 3 — diez dígitos, siete segmentos (L22).

Arriba, un dígito de reloj con los siete segmentos encendidos y el nombre que les da el libro.
Abajo, los diez dígitos: los cinco pares en una fila y los cinco impares en otra, para que la
pregunta que el capítulo hace justo después —¿hay algún segmento encendido en los cinco pares y
apagado en los cinco impares?— se pueda contestar con la vista.

Es la maqueta de Carlos del 27 de septiembre (`notas/maquetas/L22-reloj-siete-segmentos.png`), en
gris, con los nombres del libro y con los pares y los impares separados.

Qué segmentos enciende cada dígito NO está escrito aquí: sale de `siete_segmentos.py`, la misma
tabla que imprime el libro. Aquí solo se dibuja. `dibuja_digito` la usa también
`figura_reloj_a_mano.py`, para que los dos relojes del capítulo sean el mismo reloj.

Uso:
    python figura_diez_digitos.py --selftest
    python figura_diez_digitos.py
"""

# ======================= CONSTANTES =======================

DESTINO = "../figuras/diez_digitos.png"
ALTO = 4.12                   # pulgadas: así cabe en la misma página que el texto que la presenta

# La forma de un segmento, en proporción al ancho del dígito.
GROSOR = 0.17                 # grueso de cada segmento
HUECO = 0.07                  # separación entre segmentos vecinos
APAGADO_RELLENO = "#eeeeee"   # un segmento apagado se ve, pero apenas: se sabe que está ahí
APAGADO_BORDE = "#c4c4c4"

PARES = [0, 2, 4, 6, 8]
IMPARES = [1, 3, 5, 7, 9]

# ==========================================================

import argparse
import sys
from pathlib import Path

from matplotlib.patches import Polygon

from infografia import COLOR, GRIS, Lienzo
from siete_segmentos import SEGMENTOS, tabla_de_segmentos

AQUI = Path(__file__).resolve().parent

# Dónde va cada segmento en un dígito de 1 de ancho por 2 de alto: extremos del eje del segmento.
EJES = {
    "el de arriba": ((0, 2), (1, 2)),
    "el de arriba izquierda": ((0, 1), (0, 2)),
    "el de arriba derecha": ((1, 1), (1, 2)),
    "el del medio": ((0, 1), (1, 1)),
    "el de abajo izquierda": ((0, 0), (0, 1)),
    "el de abajo derecha": ((1, 0), (1, 1)),
    "el de abajo": ((0, 0), (1, 0)),
}


def hexagono(x1, y1, x2, y2, grueso, hueco):
    """El segmento como un hexágono alargado de (x1, y1) a (x2, y2), con las puntas en bisel."""
    g, h = grueso / 2, hueco
    if y1 == y2:                                   # horizontal
        a, b = min(x1, x2) + h, max(x1, x2) - h
        return [(a, y1), (a + g, y1 + g), (b - g, y1 + g), (b, y1), (b - g, y1 - g), (a + g, y1 - g)]
    a, b = min(y1, y2) + h, max(y1, y2) - h        # vertical
    return [(x1, a), (x1 + g, a + g), (x1 + g, b - g), (x1, b), (x1 - g, b - g), (x1 - g, a + g)]


def dibuja_digito(ax, x, y, ancho, encendidos, tinta, zorder=3):
    """Un dígito de reloj con la esquina de abajo izquierda del eje en (x, y). Devuelve cuántos
    segmentos ha dibujado encendidos y cuántos apagados, para que el selftest lo compruebe."""
    n_enc = n_apa = 0
    for nombre, ((x1, y1), (x2, y2)) in EJES.items():
        pts = hexagono(x + x1 * ancho, y + y1 * ancho, x + x2 * ancho, y + y2 * ancho,
                       GROSOR * ancho, HUECO * ancho)
        if nombre in encendidos:
            ax.add_patch(Polygon(pts, closed=True, facecolor=tinta, edgecolor="none", zorder=zorder))
            n_enc += 1
        else:
            ax.add_patch(Polygon(pts, closed=True, facecolor=APAGADO_RELLENO, edgecolor=APAGADO_BORDE,
                                 linewidth=0.5, zorder=zorder))
            n_apa += 1
    return n_enc, n_apa


def encendidos_de(X, d):
    return {s for j, s in enumerate(SEGMENTOS) if X[d, j]}


def dibujar(paleta, ruta):
    X = tabla_de_segmentos()
    # Sin subtítulo: lo dice el título de abajo, y así la figura cabe en la página que la presenta.
    L = Lienzo("Diez dígitos, siete segmentos", "", paleta, alto=ALTO)
    p, ax = L.p, L.ax
    cuentas = {}

    # --- arriba: el ocho, con el nombre de cada segmento
    w = 9.0
    x0, ytop = 50 - w / 2, L.y - 3.5
    y0 = ytop - 2 * w
    dibuja_digito(ax, x0, y0, w, set(SEGMENTOS), p.acento)
    etiquetas = {   # dónde se engancha la etiqueta (punto del segmento) y dónde va el texto
        "el de arriba": ((x0 + w / 2, y0 + 2 * w), (x0 + w / 2, y0 + 2 * w + 3.6), "center"),
        "el de arriba izquierda": ((x0, y0 + 1.5 * w), (x0 - 12, y0 + 1.5 * w), "right"),
        "el de arriba derecha": ((x0 + w, y0 + 1.5 * w), (x0 + w + 12, y0 + 1.5 * w), "left"),
        "el del medio": ((x0 + w * 0.3, y0 + w), (x0 - 12, y0 + w), "right"),
        "el de abajo izquierda": ((x0, y0 + 0.5 * w), (x0 - 12, y0 + 0.5 * w), "right"),
        "el de abajo derecha": ((x0 + w, y0 + 0.5 * w), (x0 + w + 12, y0 + 0.5 * w), "left"),
        "el de abajo": ((x0 + w / 2, y0), (x0 + w / 2, y0 - 3.6), "center"),
    }
    assert set(etiquetas) == set(SEGMENTOS), "la figura tiene que nombrar los siete segmentos"
    for nombre, ((ax_, ay), (tx, ty), ha) in etiquetas.items():
        fin = tx + (1.2 if ha == "right" else -1.2 if ha == "left" else 0)
        fy = ty + (-1.4 if ha == "center" and ty > ay else 1.4 if ha == "center" else 0)
        ax.plot([ax_, fin], [ay, fy], color=p.suave, linewidth=0.6, zorder=4)
        ax.plot([ax_], [ay], "o", markersize=2.2, color=p.suave, zorder=5)
        L.texto(tx, ty, nombre, ha=ha, tam=7.6)

    # la clave
    yc = y0 - 9.0
    for xc, (enc, rotulo) in zip((30, 60), ((True, "encendido"), (False, "apagado"))):
        pts = hexagono(xc - 4, yc, xc + 4, yc, GROSOR * w * 0.7, 0)
        ax.add_patch(Polygon(pts, closed=True, facecolor=p.acento if enc else APAGADO_RELLENO,
                             edgecolor="none" if enc else APAGADO_BORDE, linewidth=0.5))
        L.texto(xc + 6, yc, rotulo, tam=7.6)

    # --- abajo: pares y los impares
    ysep = yc - 5.0
    ax.plot([4, 96], [ysep, ysep], color=p.marco, linewidth=0.6)
    L.texto(4, ysep - 4.5, "Solo cambia qué segmentos están encendidos", tam=9.6, negrita=True)
    wd = 5.6
    cols = [30 + 15 * i for i in range(5)]
    for fila, (rotulo, digs) in enumerate((("pares", PARES), ("impares", IMPARES))):
        yb = ysep - 19.5 - fila * 18.5
        L.texto(4, yb + wd, rotulo, tam=8.6, negrita=True)
        for c, d in zip(cols, digs):
            cuentas[d] = dibuja_digito(ax, c - wd / 2, yb, wd, encendidos_de(X, d), p.acento)
            L.texto(c, yb - 3.6, str(d), ha="center", tam=9.0, negrita=True)
    L.guardar(ruta)
    return cuentas


def selftest():
    fallos = []
    X = tabla_de_segmentos()
    # 1. TEST NULO — los ejes dibujados son los siete segmentos del libro, ni uno más ni uno menos, y
    #    un dígito sin nada encendido se dibuja con los siete apagados.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    vacio = dibuja_digito(ax, 0, 0, 1, set(), "black")
    plt.close(fig)
    print(f"[1] test nulo         segmentos dibujados = segmentos del libro: "
          f"{'sí' if set(EJES) == set(SEGMENTOS) else 'NO'}; sin nada encendido: {vacio}")
    if set(EJES) != set(SEGMENTOS) or vacio != (0, 7):
        fallos.append("test nulo: los segmentos dibujados no son los del libro")
    # 2. SEÑAL — el uno enciende los dos de la derecha y nada más; el ocho, los siete.
    uno, ocho = encendidos_de(X, 1), encendidos_de(X, 8)
    ok = uno == {"el de arriba derecha", "el de abajo derecha"} and ocho == set(SEGMENTOS)
    print(f"[2] señal             el uno: {sorted(uno)}; el ocho: {len(ocho)} de 7")
    if not ok:
        fallos.append("señal: el uno o el ocho no encienden lo que enciende un reloj")
    # 3. INVARIANTE — cada dígito de la figura enciende tantos segmentos como dice la tabla, y la
    #    figura cabe en la página en las dos paletas.
    for pal in (COLOR, GRIS):
        cuentas = dibujar(pal, "/dev/null")
    malos = [d for d in range(10) if cuentas[d][0] != int(X[d].sum()) or sum(cuentas[d]) != 7]
    print(f"[3] invariante        dígitos dibujados con otros segmentos que la tabla: {malos or 'ninguno'}")
    if malos:
        fallos.append(f"invariante: {malos} no se dibujan como dice la tabla")
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
    dibujar(GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
