#!/usr/bin/env python3
"""
Capítulo 10 — lo que la máquina se encuentra en un paso de aprendizaje, peldaño a peldaño
(L24, D06).

«Lo que falla más veces pesa más» es la bisagra del capítulo, y se decía sin la cuenta. Esta
figura la enseña: en un paso, la máquina adivina todas las letras que lee y se corrige por todas a
la vez; cada punto es una vez que, en ese paso, se ha encontrado con lo que enseña cada peldaño
de la escalera. Los espacios son una mancha; el ejemplo de concordancia ni siquiera sale una vez
en un paso: sale una vez cada muchos. Sin cifras en el dibujo (regla 4 del capítulo).

Las cantidades se leen del bloque 4 de `datos/salidas/cuatro_retratos.txt`.

Uso:
    python figura_lo_que_encuentra_en_un_paso.py --selftest
    python figura_lo_que_encuentra_en_un_paso.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/cuatro_retratos.txt"
DESTINO = "../figuras/lo_que_encuentra_en_un_paso.png"
ALTO = 3.8
X0 = 6.0                   # donde empiezan los puntos
ANCHO_PUNTOS = 88.0
PASO_PUNTO = 1.05           # separación entre puntos
RADIO = 0.34
LADO_PASO = 0.86            # cuadrito de un paso, en la última fila

# ==========================================================

import argparse
import re
import sys

from matplotlib.patches import Circle, Rectangle

from infografia import COLOR, GRIS, Lienzo


def leer(ruta=SALIDA):
    t = open(ruta, encoding="utf-8").read()
    marca = "--- 4. LO QUE SE ENCUENTRA EN UN PASO, DE MEDIA ---"
    assert marca in t, f"falta el bloque 4 en {ruta}"
    b = t.split(marca, 1)[1].split("--- 5.", 1)[0]
    filas = []
    lineas = b.splitlines()
    for i, l in enumerate(lineas):
        m = re.match(r"^(\S.*?)\s{2,}([\d,]+)$", l)
        if m and i + 1 < len(lineas) and lineas[i + 1].startswith("  "):
            filas.append((m.group(1).strip(), lineas[i + 1].strip(), float(m.group(2).replace(",", "."))))
    assert len(filas) == 4, f"se esperaban cuatro peldaños; hay {len(filas)}"
    m = re.search(r"el último, una vez cada (\d+) pasos", b)
    assert m, "falta la línea «el último, una vez cada N pasos»"
    cada = int(m.group(1))
    assert abs(1 / filas[-1][2] - cada) < 2, "el último peldaño y su «una vez cada» no cuadran"
    return filas, cada


def dibujar(filas, cada_ultimo, paleta, ruta):
    L = Lienzo("Lo que se encuentra en un paso",
               "En un paso adivina todas las letras de lo que lee y se corrige por todas a la vez.\n"
               "Cada punto: una vez que, en ese paso, se encuentra con lo que enseña el peldaño.",
               paleta, alto=ALTO)
    p, ax = L.p, L.ax
    por_fila = int(ANCHO_PUNTOS / PASO_PUNTO)
    y = L.y - 1.5
    puntos = {}
    for peldano, que, veces in filas:
        L.texto(4, y, f"{peldano[0].upper()}{peldano[1:]}", tam=7.8, negrita=True)
        L.texto(4, y - 3.0, que, tam=6.8, color=p.suave)
        y -= 6.4
        if veces >= 1:
            n = round(veces)
            for i in range(n):
                cx = X0 + (i % por_fila) * PASO_PUNTO
                cy = y - (i // por_fila) * PASO_PUNTO
                ax.add_patch(Circle((cx, cy), RADIO, facecolor=p.tinta, edgecolor="none"))
            alto = ((n - 1) // por_fila + 1) * PASO_PUNTO
            puntos[peldano] = n
        else:
            cada = cada_ultimo
            for i in range(cada):
                x = X0 + (i % por_fila) * PASO_PUNTO - LADO_PASO / 2
                yy = y - (i // por_fila) * PASO_PUNTO - LADO_PASO / 2
                ax.add_patch(Rectangle((x, yy), LADO_PASO, LADO_PASO, facecolor="white",
                                       edgecolor=p.suave, linewidth=0.4))
            ax.add_patch(Circle((X0 + (cada - 1) % por_fila * PASO_PUNTO,
                                 y - ((cada - 1) // por_fila) * PASO_PUNTO), RADIO,
                                facecolor=p.tinta, edgecolor="none"))
            filas_c = (cada - 1) // por_fila + 1
            L.texto(X0 - 0.5, y - filas_c * PASO_PUNTO - 1.8,
                    "aquí cada cuadrito es un paso: sale una vez cada todos estos", tam=6.8, color=p.suave)
            alto = filas_c * PASO_PUNTO + 2.6
            puntos[peldano] = -cada
        y -= alto + 3.4
    assert y > 1.0, f"no cabe: sobran {1 - y:.1f} unidades"
    L.guardar(ruta)
    return puntos


def selftest():
    fallos = []
    filas, cada = leer()
    # 1. TEST NULO — un peldaño que no sale nunca en un paso no pinta puntos... salvo el que
    #    marca la vez que sale; con una vez cada un paso, un solo punto.
    rec = dibujar([(a, b, 1.0) for a, b, _ in filas], cada, GRIS, "/dev/null")
    print(f"[1] test nulo         con una vez por paso en todos: {rec}")
    if any(v != 1 for v in rec.values()):
        fallos.append("test nulo: con una vez por paso no sale un punto por peldaño")
    # 2. SEÑAL — lo dibujado es lo que dice la salida.
    rec = dibujar(filas, cada, GRIS, "/dev/null")
    esperado = {a: (round(v) if v >= 1 else -cada) for a, _, v in filas}
    print(f"[2] señal             dibujado {rec}")
    if rec != esperado:
        fallos.append(f"señal: se esperaba {esperado}")
    # 3. INVARIANTE — la escalera va de mucho a poco: cada peldaño sale menos que el anterior.
    v = [x for _, _, x in filas]
    for pal in (COLOR, GRIS):
        dibujar(filas, cada, pal, "/dev/null")
    from figura_cuatro_retratos import PELDANO
    ok = all(a > b for a, b in zip(v, v[1:])) and [f[0] for f in filas] == list(PELDANO.values())
    print(f"[3] invariante        de más a menos, y con los nombres de los peldaños de la otra figura: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: los peldaños no van de más a menos")
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
    filas, cada = leer()
    dibujar(filas, cada, GRIS, DESTINO)
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
