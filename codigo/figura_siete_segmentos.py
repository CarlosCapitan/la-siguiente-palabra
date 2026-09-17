#!/usr/bin/env python3
"""
Capítulo 3 — la figura de por qué «¿es par?» es fácil en un reloj y difícil a mano.

Arriba: los diez dígitos de un reloj digital, colocados en el total que les da el comité
de siete miembros que encontró `siete_segmentos.py`. Todos los pares caen a un lado del
listón y todos los impares al otro: ahí una sola raya basta.

Abajo: los mismos diez dígitos escritos a mano, de los que usa la medición del capítulo.
No hay segmentos. No hay nada que compartan los cinco de arriba y no tenga ninguno de los
de abajo, y por eso ahí una sola raya se queda en el 90 %.

Nada está dibujado a mano: los totales salen del comité entrenado y los dígitos, del mismo
conjunto que mide el capítulo. Lo que la figura afirma, el selftest lo comprueba.

Uso:
    python figura_siete_segmentos.py
    python figura_siete_segmentos.py --selftest
"""

# ======================= CONSTANTES =======================

SALIDA = "../figuras/siete_segmentos.png"
ANCHO_ALTO = (6.2, 4.9)          # pulgadas, para una página de 6 por 9
PUNTOS = 200
# xelatex encoge la figura para meterla en la caja de texto; las letras se dibujan más
# grandes en esa proporción para que en el papel salgan del tamaño que se quiere.
ESCALA = 1.0 / 0.72

LADO = 8                         # los dígitos manuscritos son de ocho puntos por ocho
SEPARACION = 0.42                # cuánto se apartan, a los lados, dos dígitos con el mismo
                                 # total. Apilarlos los dejaba montados unos encima de otros.

TITULO_ARRIBA = "en un reloj digital, una sola raya basta"
TITULO_ABAJO = "escritos a mano, no"
PIE_ARRIBA = "cada dígito, puesto en el total que le da el comité de siete miembros"
PIE_ABAJO = ("busca algo que tengan los cinco de arriba\n"
             "y no tenga ninguno de los cinco de abajo")
CLAVE = "negro: par    ·    blanco: impar"

# ==========================================================

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from siete_segmentos import tabla_de_segmentos, entrenar, SEGMENTOS
from que_mira_cada_una import cargar_digitos


# Cada segmento, como un trazo (x1, y1) -> (x2, y2) en una cajita de 1 de ancho por 2 de
# alto. El orden es el de SEGMENTOS, y se comprueba en el selftest.
TRAZOS = [
    ((0.0, 2.0), (1.0, 2.0)),    # el de arriba
    ((0.0, 1.0), (0.0, 2.0)),    # el de arriba izquierda
    ((1.0, 1.0), (1.0, 2.0)),    # el de arriba derecha
    ((0.0, 1.0), (1.0, 1.0)),    # el del medio
    ((0.0, 0.0), (0.0, 1.0)),    # el de abajo izquierda
    ((1.0, 0.0), (1.0, 1.0)),    # el de abajo derecha
    ((0.0, 0.0), (1.0, 0.0)),    # el de abajo
]


def dibuja_reloj(ax, encendidos, x, y, alto, par):
    """Un dígito de reloj, con los segmentos apagados en gris muy claro para que se vea
    que están ahí y no encendidos: es lo que hace reconocible la forma."""
    ancho = alto / 2.4
    for enc, ((x1, y1), (x2, y2)) in zip(encendidos, TRAZOS):
        ax.plot([x + x1 * ancho, x + x2 * ancho],
                [y + (y1 - 1) * alto / 2, y + (y2 - 1) * alto / 2],
                color="black" if enc else "0.88",
                linewidth=3.0 if enc else 2.0,
                solid_capstyle="round", zorder=3)
    ax.plot(x + ancho / 2, y - alto * 0.78, "o", markersize=6,
            markerfacecolor="black" if par else "white",
            markeredgecolor="black", markeredgewidth=1.2, zorder=3)


def totales(X, w, b):
    return np.array([X[d] @ w + b for d in range(10)])


def colocados(t):
    """Dónde se dibuja cada dígito. Los que caen en el mismo total se apartan a los lados
    para que se vean los diez; el apartado es siempre menor que la distancia al listón, y
    el selftest lo comprueba, porque si un dígito cruzara la raya el dibujo mentiría."""
    fuera = []
    for valor in sorted(set(round(float(x), 3) for x in t)):
        iguales = [d for d in range(10) if round(float(t[d]), 3) == valor]
        for k, d in enumerate(iguales):
            fuera.append((valor + (k - (len(iguales) - 1) / 2) * SEPARACION, d))
    return fuera


def panel_reloj(ax, X, w, b):
    t = totales(X, w, b)
    lim = (t.min() - 1.6, t.max() + 1.6)
    # El lado del listón donde el comité dice «par», sombreado como en el capítulo 2.
    ax.axvspan(-b, lim[1], color="0.90", zorder=0)
    ax.axvline(-b, linestyle="--", color="0.35", linewidth=1.4, zorder=2)
    for x, d in colocados(t):
        dibuja_reloj(ax, X[d] > 0, x, 0.0, 0.44, d % 2 == 0)
    ax.text(lim[1] - 0.1, 0.95, "de este lado, «par»", ha="right", va="top",
            fontsize=8.5 * ESCALA * 0.72, color="0.3")
    ax.set_xlim(*lim)
    ax.set_ylim(-0.95, 1.0)
    ax.set_xlabel("el total que suma el comité")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)


def panel_mano(ax, imagenes):
    rejilla = np.ones((2 * LADO + 2, 5 * (LADO + 1) + 1))
    for d in range(10):
        fila, col = d % 2, d // 2
        y0 = 1 + fila * (LADO + 1)
        x0 = 1 + col * (LADO + 1)
        rejilla[y0:y0 + LADO, x0:x0 + LADO] = 1.0 - imagenes[d]
    ax.imshow(rejilla, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.text(-1.5, 1 + LADO / 2, "pares", ha="right", va="center",
            fontsize=9.5 * ESCALA * 0.72)
    ax.text(-1.5, 2 + LADO + LADO / 2, "impares", ha="right", va="center",
            fontsize=9.5 * ESCALA * 0.72)


def un_ejemplo_de_cada(X_dig, t_dig):
    """El primer ejemplo de cada dígito, en el orden del conjunto. Sin elegir el que queda
    bonito: el primero."""
    # cargar_digitos ya devuelve la tinta de 0 a 1. Dividir otra vez por dieciséis los
    # dejaba casi blancos, y la figura enseñaba diez manchas invisibles.
    assert X_dig.max() <= 1.0, f"Se esperaba tinta de 0 a 1; se encontró hasta {X_dig.max()}"
    return [X_dig[np.argmax(t_dig == d)].reshape(LADO, LADO) for d in range(10)]


def dibujar():
    X = tabla_de_segmentos()
    y = np.array([1.0 if d % 2 == 0 else -1.0 for d in range(10)])
    w, b = entrenar(X, y)
    assert w is not None, "el comité de siete segmentos debería converger"
    X_dig, t_dig = cargar_digitos()
    imagenes = un_ejemplo_de_cada(X_dig, t_dig)

    plt.rcParams["font.size"] = 9 * ESCALA * 0.72
    fig, (arriba, abajo) = plt.subplots(2, 1, figsize=ANCHO_ALTO,
                                        gridspec_kw={"height_ratios": [1.35, 1]})
    panel_reloj(arriba, X, w, b)
    panel_mano(abajo, imagenes)
    arriba.set_title(TITULO_ARRIBA, fontsize=10.5)
    abajo.set_title(TITULO_ABAJO, fontsize=10.5, pad=10)
    fig.text(0.5, 0.505, PIE_ARRIBA, ha="center", va="center",
             fontsize=8.5, color="0.35")
    fig.text(0.5, 0.045, PIE_ABAJO, ha="center", va="bottom",
             fontsize=8.5, color="0.35")
    fig.subplots_adjust(left=0.15, right=0.97, top=0.92, bottom=0.18, hspace=0.75)
    fig.savefig(SALIDA, dpi=PUNTOS)
    plt.close(fig)
    print(f"escrito {SALIDA}")
    return X, w, b


def selftest():
    fallos = []
    X = tabla_de_segmentos()

    # 1. TEST NULO — los trazos del dibujo tienen que ser los siete segmentos y en el
    #    mismo orden que la tabla. Si el dibujo pintara otra cosa, la figura enseñaría un
    #    reloj que no es el que se ha medido, y nadie lo notaría.
    print(f"[1] test nulo         trazos dibujados: {len(TRAZOS)}; segmentos medidos: "
          f"{len(SEGMENTOS)}")
    if len(TRAZOS) != len(SEGMENTOS):
        fallos.append("test nulo: el dibujo no tiene un trazo por cada segmento medido")
    ocho = X[8] > 0
    if not ocho.all():
        fallos.append("test nulo: el ocho debería encender los siete segmentos")

    # 2. SEÑAL IMPLANTADA — el uno enciende dos segmentos y solo dos, y son los dos de la
    #    derecha. Es la comprobación que cualquiera puede hacer mirando un microondas.
    uno = X[1] > 0
    derechas = [SEGMENTOS.index("el de arriba derecha"), SEGMENTOS.index("el de abajo derecha")]
    ok = uno.sum() == 2 and all(uno[i] for i in derechas)
    print(f"[2] señal implantada  el uno enciende {int(uno.sum())} segmentos, "
          f"y son los dos de la derecha: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("señal implantada: el uno debería encender solo los dos de la derecha")

    # 3. INVARIANTE DEL DOMINIO — la figura afirma que todos los pares caen a un lado del
    #    listón y todos los impares al otro. Si no fuera verdad, el dibujo mentiría.
    y = np.array([1.0 if d % 2 == 0 else -1.0 for d in range(10)])
    w, b = entrenar(X, y)
    t = totales(X, w, b)
    bien = all((t[d] > 0) == (d % 2 == 0) for d in range(10))
    # Y que el apartado de los empates no cruce el listón: donde se DIBUJA cada dígito
    # tiene que seguir estando del lado que le toca.
    dibujado = all((x > -b) == (d % 2 == 0) for x, d in colocados(t))
    print(f"[3] invariante        los cinco pares a un lado del listón y los cinco "
          f"impares al otro: {'sí' if bien else 'NO'}; y donde se dibujan, "
          f"{'también' if dibujado else 'NO'}")
    if not bien or not dibujado:
        fallos.append("invariante: el dibujo colocaría algún dígito en el lado equivocado")

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
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    dibujar()
    return 0


if __name__ == "__main__":
    sys.exit(main())
