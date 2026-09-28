#!/usr/bin/env python3
"""
Capítulo 8 — la cuadrícula de comparaciones, dibujada (L24).

«Si cada palabra tiene que comparar su pregunta con la etiqueta de todas las anteriores, el trabajo
crece con el cuadrado de la longitud del texto.» Esta figura lo enseña con dos frases: cuatro
palabras y ocho. Cada casilla oscura es una comparación: la palabra de la fila (la que pregunta)
con la de la columna (la de la etiqueta). Las de arriba a la derecha quedan en blanco porque esas
palabras todavía no se han escrito. Al doblar la frase, las casillas pasan de 10 a 36.

Las casillas NO se deciden aquí: se leen del apartado 1 de `datos/salidas/cuadricula_y_fila.txt`.
En gris: el libro se imprime en negro.

Uso:
    python figura_cuadricula.py --selftest
    python figura_cuadricula.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/cuadricula_y_fila.txt"
DESTINO = "../figuras/cuadricula.png"
ALTO = 2.8                  # pulgadas
LADO = 4.3                   # lado de una casilla, en unidades del lienzo

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

from matplotlib.patches import Rectangle

from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def leer(ruta):
    """Del apartado 1: para cada cuadrícula, (palabras, filas de x/·, casillas, total)."""
    texto = Path(ruta).read_text(encoding="utf-8")
    assert "1. CADA PALABRA, CON ELLA MISMA" in texto, f"se esperaba el apartado 1 en {ruta}"
    uno = texto.split("1. CADA PALABRA, CON ELLA MISMA", 1)[1].split("\n2.", 1)[0]
    rejillas, palabras, filas = [], [], []
    for linea in uno.splitlines():
        m = re.match(r"^\s+(\d+) (\S+)\s+((?:[x·]\s*)+)$", linea)
        if m:
            palabras.append(m.group(2))
            filas.append([c == "x" for c in m.group(3).split()])
            continue
        m = re.match(r"^\s+(\d+) palabras: (\d+) casillas de (\d+)\.$", linea)
        if m:
            n = int(m.group(1))
            assert len(palabras) == n and all(len(f) == n for f in filas), \
                f"se esperaba una cuadrícula de {n} por {n}"
            assert sum(map(sum, filas)) == int(m.group(2)), "las casillas contadas no son las de la salida"
            rejillas.append((palabras, filas, int(m.group(2)), int(m.group(3))))
            palabras, filas = [], []
    assert len(rejillas) == 2, f"se esperaban dos cuadrículas; hay {len(rejillas)}"
    return rejillas


def dibujar(rejillas, paleta, ruta):
    L = Lienzo("Cada palabra, con las de antes",
               "Cada casilla oscura es una comparación: la pregunta de la palabra de la\n"
               "fila con la etiqueta de la palabra de la columna.", paleta, alto=ALTO)
    p, ax = L.p, L.ax
    ytop = L.y - 3.0
    x0s = [14.0, 58.0]
    oscuras = []
    for (pal, filas, n_cas, total), x0 in zip(rejillas, x0s):
        n = len(pal)
        lado = LADO if n <= 4 else LADO * 0.78
        for f in range(n):
            L.texto(x0 - 1.0, ytop - (f + 0.5) * lado, pal[f], tam=6.6, ha="right")
            for c in range(n):
                on = filas[f][c]
                oscuras.append(on)
                ax.add_patch(Rectangle((x0 + c * lado, ytop - (f + 1) * lado), lado, lado,
                                       facecolor=p.acento if on else "white",
                                       edgecolor=p.marco, linewidth=0.6))
        for c in range(n):
            ax.text(x0 + (c + 0.5) * lado, ytop + 1.2, pal[c], rotation=60, ha="left", va="bottom",
                    fontsize=6.0, color=p.suave, family="Carlito")
        L.texto(x0 + n * lado / 2, ytop - n * lado - 3.2, f"{n} palabras: {n_cas} casillas",
                ha="center", tam=8.4, negrita=True)
    L.pie("Fila: la palabra que pregunta. Columna: la palabra de la etiqueta. En blanco, las palabras\n"
          "que aún no se han escrito: estas máquinas solo miran hacia atrás.")
    L.guardar(ruta)
    return oscuras


def selftest():
    fallos = []
    rej = leer(AQUI / SALIDA)
    # 1. TEST NULO — una salida sin el apartado 1 no da figura.
    import os, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("nada\n")
    try:
        leer(fh.name)
        fallos.append("test nulo: leyó cuadrículas de un texto que no las tiene")
    except (AssertionError, AttributeError):
        pass
    finally:
        os.unlink(fh.name)
    print("[1] test nulo         un texto sin el apartado 1 no da figura")
    # 2. SEÑAL — las casillas oscuras dibujadas son las contadas: 10 y 36.
    osc = dibujar(rej, GRIS, "/dev/null")
    dibujadas = sum(osc)
    print(f"[2] señal             casillas oscuras dibujadas: {dibujadas} (salida: "
          f"{' + '.join(str(r[2]) for r in rej)})")
    if dibujadas != sum(r[2] for r in rej):
        fallos.append("señal: no se dibujan las casillas que cuenta la salida")
    # 3. INVARIANTE — ninguna casilla por encima de la diagonal está oscura, y se dibuja en las dos
    #    paletas.
    arriba = any(f[c] for pal, filas, _, _ in rej for i, f in enumerate(filas) for c in range(i + 1, len(f)))
    for pal in (COLOR, GRIS):
        dibujar(rej, pal, "/dev/null")
    print(f"[3] invariante        casillas oscuras por encima de la diagonal: {'sí' if arriba else 'ninguna'}")
    if arriba:
        fallos.append("invariante: una palabra mira hacia delante")
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
