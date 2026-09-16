#!/usr/bin/env python3
"""
Capítulo de los conceptos — la figura de la capa: ocho preguntas que nadie repartió.

Dibuja, una al lado de otra, lo que mira cada una de las ocho neuronas de la capa de
en medio de la red de `que_mira_cada_una.py`, entrenada para decir si un dígito
manuscrito es par. Debajo de cada una, lo que acierta ella sola.

Lo que la figura afirma es que son ocho preguntas DISTINTAS y que NINGUNA es la
respuesta. Las dos cosas las comprueba el selftest con números, no a ojo.

Uso:
    python figura_una_capa.py
    python figura_una_capa.py --selftest
"""

# ======================= CONSTANTES =======================

SALIDA = "../figuras/una_capa.png"
ANCHO_ALTO = (6.2, 3.3)          # pulgadas, para una página de 6 por 9
PUNTOS = 200
# xelatex encoge la figura al 72 % para meterla en la caja de texto; las letras se
# dibujan más grandes en esa proporción para que en el papel salgan como se quiere.
ESCALA = 1.0 / 0.72
FILAS, COLUMNAS = 2, 4
PARECIDO_MAXIMO_ADMISIBLE = 0.90   # por encima de esto, dos neuronas son la misma

# ==========================================================

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from que_mira_cada_una import (entrenar_con_capa, acierto_de_cada_una, parecido_maximo,
                               una_raya_sola, EN_MEDIO, LADO)


def dibujar():
    m = entrenar_con_capa()
    aciertos = acierto_de_cada_una(m)
    W = m["red"].W[0]
    assert W.shape == (LADO * LADO, EN_MEDIO), \
        f"Se esperaban {LADO*LADO} pesos para cada una de las {EN_MEDIO}; se encontró {W.shape}"

    fig, ejes = plt.subplots(FILAS, COLUMNAS, figsize=ANCHO_ALTO)
    lim = np.abs(W).max()
    for j, ax in enumerate(ejes.ravel()):
        ax.imshow(W[:, j].reshape(LADO, LADO), cmap="gray_r", vmin=-lim, vmax=lim)
        ax.set_title(f"la número {j + 1}", fontsize=10.4)
        ax.set_xlabel(f"acierta ella sola\n{100 * aciertos[j]:.0f} %", fontsize=9.7)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.8)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.14, wspace=0.25, hspace=0.75)
    fig.savefig(SALIDA, dpi=PUNTOS)
    plt.close(fig)
    return m, aciertos


# ============================ SELFTEST ============================

def selftest():
    fallos = []
    m = entrenar_con_capa()
    aciertos = acierto_de_cada_una(m)

    # [1] Test nulo: con las etiquetas barajadas la red se cae al azar, así que los
    #     ocho cuadros que dibujaríamos no dibujarían nada.
    nulo = entrenar_con_capa(permutar=True)
    print(f"[1] test nulo         etiquetas barajadas: la red acierta "
          f"{100 * nulo['entera']:.1f} % (azar = 50 %)")
    if nulo["entera"] > 0.75:
        fallos.append(f"test nulo: con etiquetas barajadas acierta {100*nulo['entera']:.1f} %")

    # [2] Señal implantada: la figura dice que son ocho preguntas DISTINTAS. Se mide
    #     cuánto se parecen entre sí; si dos fueran casi la misma, sería mentira.
    parecido = parecido_maximo(m["red"])
    print(f"[2] señal implantada  lo más que se parecen dos de las ocho: {100*parecido:.1f} %")
    if parecido > PARECIDO_MAXIMO_ADMISIBLE:
        fallos.append(f"señal implantada: dos de las ocho se parecen un {100*parecido:.1f} %, "
                      f"así que no son ocho preguntas distintas sino menos")

    # [3] Invariante del dominio: ninguna de las ocho es la respuesta, y juntas ganan a
    #     la raya sola. Es exactamente lo que dicen los pies de la figura.
    mejor, sola = float(aciertos.max()), una_raya_sola()
    print(f"[3] invariante        entera {100*m['entera']:.1f} %; mejor de las ocho "
          f"{100*mejor:.1f} %; sin capa {100*sola:.1f} %")
    if not (m["entera"] > mejor and m["entera"] > sola):
        fallos.append(f"invariante: la figura enseña ocho especialistas mediocres que juntos "
                      f"ganan, y los números no lo sostienen "
                      f"(entera {100*m['entera']:.1f}, mejor {100*mejor:.1f}, sola {100*sola:.1f})")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    m, aciertos = dibujar()
    print(f"escrito {SALIDA}")
    print(f"ocho neuronas de en medio; la mejor sola acierta {100*aciertos.max():.1f} % "
          f"y la red entera {100*m['entera']:.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
