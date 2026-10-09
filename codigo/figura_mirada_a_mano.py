#!/usr/bin/env python3
"""
Capítulo 8 — la figura de una mirada que nadie programa.

La última letra de un trozo del Quijote reparte su mirada entre las dieciséis que tiene detrás
(ella incluida). Arriba, antes de aprender nada; abajo, al final del entrenamiento. Cada casilla
lleva la letra y, debajo, de cada cien, cuánto de la mirada va a ella; cuanto más oscura, más.

Los números NO se calculan aquí: se leen del apartado 5 de `datos/salidas/mirada_a_mano.txt`. Si la
salida cambia, la figura cambia. En gris: el libro se imprime en negro.

Uso:
    python figura_mirada_a_mano.py --selftest
    python figura_mirada_a_mano.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/mirada_a_mano.txt"
DESTINO = "../figuras/mirada_a_mano.png"
ALTO = 3.7                  # pulgadas
CONTEXTO = 16               # tiene que coincidir con las casillas que trae la salida

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

from formato import leer_tablas
from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def leer(ruta):
    """El reparto de la frase, de su tabla editorial (L24, 9 de octubre)."""
    tablas = leer_tablas(Path(ruta).read_text(encoding="utf-8"))
    titulos = [t for t in tablas if t.startswith("El reparto de la última letra de «")]
    assert len(titulos) == 1, f"se esperaba la tabla del reparto en {ruta}; hay {titulos}"
    frase = re.search(r"«(.+?)»", titulos[0]).group(1)
    rotulos, filas, notas = tablas[titulos[0]]
    letras = [f[1] for f in filas]
    repartos = [(re.search(r"tras ([\d.]+) pasos", rotulos[k]).group(1), [int(f[k]) for f in filas])
                for k in (2, 3)]
    assert len(letras) == CONTEXTO, f"se esperaban {CONTEXTO} casillas; hay {len(letras)}"
    nota = " ".join(notas)
    viene = re.search(r"En el Quijote viene detrás: «(.)»", nota).group(1)
    m = re.search(r"Reparte entre: al empezar ([\d,]+) letras; al final ([\d,]+)\.", nota)
    apuesta = re.search(r"Lo más probable detrás: al empezar «(.)», al final «(.)»", nota)
    extra = dict(viene=viene, reparte=(m.group(1), m.group(2)),
                 apuesta=tuple("_" if a == " " else a for a in apuesta.groups()))
    return frase, letras, repartos, extra


def dibujar(frase, letras, repartos, extra, paleta, ruta):
    from matplotlib.patches import Rectangle
    L = Lienzo("Una mirada que nadie programa",
               f"La última letra de «{frase}» reparte su mirada entre las {CONTEXTO} que tiene\n"
               "detrás. Debajo de cada una, de cada cien, cuánto va a ella.",
               paleta, alto=ALTO)
    p = L.p
    maximo = max(max(r) for _, r in repartos)
    for n, (paso, r) in enumerate(repartos, 1):
        titulo = "Antes de aprender nada" if paso == "0" else f"Tras {paso} pasos de aprender"
        titulo += f": reparte entre {extra['reparte'][n - 1]} letras"
        x0, y, ancho = L.panel(n, titulo, 24.0)
        lado = ancho / (CONTEXTO + 1.4)
        for i, (ch, v) in enumerate(zip(letras, r)):
            g = 1 - 0.85 * v / maximo
            x = x0 + i * lado
            ultima = i == CONTEXTO - 1
            L.ax.add_patch(Rectangle((x + 0.3, y - 7.6), lado - 0.6, 6.0,
                                     facecolor=(g, g, g), edgecolor=p.tinta if ultima else p.marco,
                                     linewidth=1.3 if ultima else 0.6))
            L.texto(x + lado / 2, y - 4.6, "_" if ch == "_" else ch, ha="center", tam=8.6,
                    negrita=True, color="white" if g < 0.5 else p.tinta)
            L.texto(x + lado / 2, y - 10.0, str(v), ha="center", tam=7.0)
        # la letra que viene de verdad detrás del tramo, y a cuál apuesta la máquina
        x = x0 + (CONTEXTO + 0.4) * lado
        L.ax.add_patch(Rectangle((x + 0.3, y - 7.6), lado - 0.6, 6.0, facecolor="white",
                                 edgecolor=p.tinta, linewidth=0.9, linestyle=(0, (2, 1.5))))
        L.texto(x + lado / 2, y - 4.6, extra["viene"], ha="center", tam=8.6, negrita=True)
        L.texto(x + lado / 2, y - 10.0, "viene", ha="center", tam=6.2, color=p.suave)
        L.texto(x0 + ancho, y - 13.0, f"lo más probable: «{extra['apuesta'][n - 1]}»", ha="right", tam=6.6,
                color=p.suave)
        if n == 1:
            L.flecha()
    igual = f"{100 / CONTEXTO:.2f}".replace(".", ",")
    L.pie("«_» es un espacio. Borde grueso: la letra que mira. De trazos: la que viene de verdad detrás\n"
          f"en el Quijote, que tiene que acertar. Antes de aprender, cada una recibe {igual}, que "
          f"redondeado es {round(100 / CONTEXTO)}.")
    L.guardar(ruta)


def selftest():
    fallos = []
    frase, letras, repartos, extra = leer(AQUI / SALIDA)
    import os, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("nada\n")
    try:
        leer(fh.name)
        fallos.append("test nulo: leyó un reparto de un texto que no lo tiene")
    except (AssertionError, AttributeError):
        pass
    finally:
        os.unlink(fh.name)
    print("[1] test nulo         un texto sin el apartado 5 no da figura")
    # 2. SEÑAL: las letras leídas son las últimas de la frase, y cada reparto suma cien (redondeos
    #    aparte: cada casilla puede perder medio punto).
    esperadas = [("_" if c == " " else c) for c in frase[-CONTEXTO:]]
    sumas = [sum(r) for _, r in repartos]
    ok = letras == esperadas and all(abs(s - 100) <= CONTEXTO / 2 for s in sumas)
    print(f"[2] señal             letras de la frase: {'sí' if letras == esperadas else 'NO'}; "
          f"los repartos suman {sumas}")
    if not ok:
        fallos.append("señal: las letras no son las de la frase, o un reparto no suma cien")
    for pal in (COLOR, GRIS):
        dibujar(frase, letras, repartos, extra, pal, "/dev/null")
    print("[3] invariante        la figura se dibuja en color y en gris sin salirse de la página")
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
