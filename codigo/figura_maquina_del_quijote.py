#!/usr/bin/env python3
"""
Capítulo 8 — la máquina del Quijote por dentro, dibujada (L24).

El camino de la última letra de un tramo, de arriba abajo: su lista de números (con la marca de su
sitio), las tres tablas que sacan la pregunta, la etiqueta y el contenido, las puntuaciones, el
reparto y la mezcla, la cuarta tabla, la SUMA con la propia lista (que baja por el lado sin pasar
por la mirada), mezclar, y los porcentajes de las 42 letras. El recuadro de trazos es lo que se
quita en «sin mirar atrás»: la propia lista sigue llegando por el lado.

Cuántos números tiene cada pieza NO se escribe aquí: se lee del apartado 1 de
`datos/salidas/la_e_de_acordarme.txt`. En gris: el libro se imprime en negro.

Uso:
    python figura_maquina_del_quijote.py --selftest
    python figura_maquina_del_quijote.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/la_e_de_acordarme.txt"
DESTINO = "../figuras/maquina_del_quijote.png"
ALTO = 5.0                  # pulgadas
TABLAS_DE_MIRAR = ["la tabla de las preguntas", "la tabla de las etiquetas",
                   "la tabla de los contenidos", "la tabla que devuelve lo traído"]

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def leer(ruta):
    """Del apartado 1: {pieza: números}, el total y lo que se quita sin mirar atrás."""
    texto = Path(ruta).read_text(encoding="utf-8")
    assert "1. LA MÁQUINA, PIEZA A PIEZA" in texto, f"se esperaba el apartado 1 en {ruta}"
    uno = texto.split("1. LA MÁQUINA, PIEZA A PIEZA", 1)[1].split("\n2.", 1)[0]
    piezas = {}
    for m in re.finditer(r"^  (\S.*?)\s{2,}([\d.]+)$", uno, re.M):
        piezas[m.group(1)] = int(m.group(2).replace(".", ""))
    total = piezas.pop("total")
    piezas.pop("la pieza", None)
    quita = int(re.search(r"se quitan las cuatro tablas: ([\d.]+) números", uno).group(1).replace(".", ""))
    assert sum(piezas.values()) == total, "las piezas no suman el total"
    assert sum(piezas[t] for t in TABLAS_DE_MIRAR) == quita, "las cuatro tablas no suman lo que se quita"
    return piezas, total, quita


def miles(n):
    return f"{n:,}".replace(",", ".")


def dibujar(piezas, total, quita, paleta, ruta):
    L = Lienzo("La máquina del Quijote por dentro",
               "El camino de la última letra del fragmento, de arriba abajo. Entre corchetes,\n"
               "cuántos números tiene cada pieza.", paleta, alto=ALTO)
    p, ax = L.p, L.ax
    busca = lambda prefijo: next(v for k, v in piezas.items() if k.startswith(prefijo))
    cx, w = 47.0, 56.0          # la columna central
    y = L.y - 3.5
    h = 5.6
    paso = 9.6
    cajas = []

    def caja(y, texto, x=cx, ancho=w, relleno="white", negrita=False, tam=7.4, alto=h):
        L.ficha(x - ancho / 2, y, texto, ancho, alto=alto, relleno=relleno, negrita=negrita, tam=tam)
        cajas.append((x, y))

    def flecha(y0, y1, x=cx):
        ax.add_patch(FancyArrowPatch((x, y0 - h / 2), (x, y1 + h / 2), arrowstyle="-|>",
                                     mutation_scale=7, color=p.acento, linewidth=1.0))

    ys = [y - i * paso for i in range(9)]
    caja(ys[0], "las 16 letras del fragmento: «quiero acordarme»", negrita=True)
    caja(ys[1], f"cada letra, su lista de 32 números   [{miles(busca('las listas'))}]\n"
                f"más la marca de su sitio   [{miles(busca('las marcas'))}]", tam=7.0)
    tercio = w / 3
    nombres = [("la tabla de las preguntas", "tabla de las preguntas"),
               ("la tabla de las etiquetas", "tabla de las etiquetas"),
               ("la tabla de los contenidos", "tabla de los contenidos")]
    for i, (clave, rotulo) in enumerate(nombres):
        x = cx - w / 2 + tercio * (i + 0.5)
        caja(ys[2], f"{rotulo.replace(' de l', chr(10) + 'de l')}\n[{miles(piezas[clave])}]", x=x,
             ancho=tercio - 1.4, tam=6.6, alto=h + 2.6)
        ax.add_patch(FancyArrowPatch((cx, ys[1] - h / 2), (x, ys[2] + h / 2 + 1.3), arrowstyle="-|>",
                                     mutation_scale=6, color=p.acento, linewidth=0.9))
    caja(ys[3], "puntuaciones: la pregunta contra cada etiqueta")
    caja(ys[4], "reparto, de cada cien; mezcla de los contenidos")
    caja(ys[5], f"la tabla que devuelve lo traído   [{miles(piezas['la tabla que devuelve lo traído'])}]")
    caja(ys[6], "SUMA: la propia lista más lo traído", negrita=True)
    caja(ys[7], f"los comités de después   [{miles(piezas['los comités de después'])}]")
    caja(ys[8], f"porcentajes de las 42 letras   [{miles(busca('de la lista a'))}]", negrita=True)
    flecha(ys[0], ys[1])
    ax.add_patch(FancyArrowPatch((cx, ys[2] - h / 2 - 1.3), (cx, ys[3] + h / 2), arrowstyle="-|>",
                                 mutation_scale=7, color=p.acento, linewidth=1.0))
    for a, b in zip(ys[3:8], ys[4:9]):
        flecha(a, b)
    # el camino del lado: la propia lista baja sin pasar por la mirada
    xl = cx - w / 2 - 6.0
    ax.plot([cx - w / 2, xl, xl], [ys[1], ys[1], ys[6]], color=p.tinta, linewidth=1.1)
    ax.add_patch(FancyArrowPatch((xl, ys[6]), (cx - w / 2, ys[6]), arrowstyle="-|>",
                                 mutation_scale=7, color=p.tinta, linewidth=1.1))
    ax.text(xl - 1.4, (ys[1] + ys[6]) / 2, "la propia lista\nsigue adelante", rotation=90,
            ha="right", va="center", fontsize=7.0, color=p.tinta, family="Carlito", linespacing=1.2)
    # lo que se quita sin mirar atrás
    top, bot = ys[2] + h / 2 + 2.4, ys[5] - h / 2 - 1.2
    ax.add_patch(FancyBboxPatch((cx - w / 2 - 2.0, bot), w + 4.0, top - bot,
                                boxstyle="round,pad=0,rounding_size=1.0", facecolor="none",
                                edgecolor=p.tinta, linewidth=1.0, linestyle=(0, (4, 3))))
    ax.text(cx + w / 2 + 3.0, (top + bot) / 2, f"sin mirar atrás\nse quita esto:\n{miles(quita)} números",
            ha="left", va="center", fontsize=7.0, color=p.tinta, family="Carlito", linespacing=1.25)
    L.pie(f"En total, {miles(total)} números; sin mirar atrás, {miles(total - quita)}.")
    L.guardar(ruta)
    return cajas


def selftest():
    fallos = []
    piezas, total, quita = leer(AQUI / SALIDA)
    import os, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("nada\n")
    try:
        leer(fh.name)
        fallos.append("test nulo: leyó piezas de un texto que no las tiene")
    except (AssertionError, AttributeError, StopIteration):
        pass
    finally:
        os.unlink(fh.name)
    print("[1] test nulo         un texto sin el apartado 1 no da figura")
    # 2. SEÑAL — las cifras del libro: 15.690 en total, 4.096 que se quitan, 11.594 que quedan.
    ok = (total, quita, total - quita) == (15690, 4096, 11594)
    print(f"[2] señal             total {miles(total)}, se quitan {miles(quita)}, quedan {miles(total - quita)}")
    if not ok:
        fallos.append("señal: las cifras no son las del capítulo (15.690, 4.096, 11.594)")
    # 3. INVARIANTE — ninguna caja se sale de la página, en las dos paletas.
    for pal in (COLOR, GRIS):
        cajas = dibujar(piezas, total, quita, pal, "/dev/null")
    dentro = all(0 < x < 100 and 8 < y for x, y in cajas)
    print(f"[3] invariante        {len(cajas)} cajas, todas dentro de la página: {'sí' if dentro else 'NO'}")
    if not dentro:
        fallos.append("invariante: una caja se sale de la página o pisa el pie")
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
    dibujar(*leer(AQUI / SALIDA), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
