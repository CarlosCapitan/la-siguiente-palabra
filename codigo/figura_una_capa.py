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
ANCHO_ALTO = (6.2, 3.9)          # pulgadas, para una página de 6 por 9
PUNTOS = 200
# xelatex encoge la figura al 72 % para meterla en la caja de texto; las letras se
# dibujan más grandes en esa proporción para que en el papel salgan como se quiere.
ESCALA = 1.0 / 0.72
FILAS, COLUMNAS = 2, 4
PARECIDO_MAXIMO_ADMISIBLE = 0.90   # por encima de esto, dos neuronas son la misma

# El título y la clave. Ocho cuadros grises sin rótulo no dicen nada: hay que decir qué
# es un cuadro (los 64 puntos del dígito), qué es el negro y qué es el blanco. La figura
# tiene que poder entenderse sin el párrafo que la presenta.
TITULO = "LO QUE MIRA CADA UNO DE LOS OCHO COMITÉS DE EN MEDIO"
SUBTITULO = "cada cuadro son los 64 puntos del dígito, como los ve ese comité"
# L24 (A02): la clave decía «negro: este punto empuja hacia «es par»», pero qué extremo de un comité
# es el «par» depende de la lectura que se le deje elegir (acierto_de_cada_una). La clave dice ahora
# lo que el dibujo enseña de verdad, y debajo de cada cuadro va qué lectura se le ha contado.
CLAVE = ("negro: tinta ahí sube el número de ese comité  ·  blanco: lo baja"
         "  ·  gris: no cuenta")

# ==========================================================

import argparse
import sys

from formato import pct
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
        ax.set_title(f"el comité {j + 1}", fontsize=10.4)
        dice_par = float(((m["medio"][:, j] > 0.5) == (m["yte"] > 0.5)).mean()) >= 0.5
        ax.set_xlabel(f"acierta él solo {pct(aciertos[j], 0)}\nsi pasa de la mitad:\n«{'par' if dice_par else 'impar'}»",
                      fontsize=9.2, linespacing=1.05)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.8)
    fig.suptitle(TITULO, fontsize=11.0, y=0.985)
    fig.text(0.5, 0.925, SUBTITULO, ha="center", va="top", fontsize=8.8, color="0.35")
    fig.text(0.5, 0.015, CLAVE, ha="center", va="bottom", fontsize=8.8, color="0.35")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.79, bottom=0.20, wspace=0.25, hspace=1.05)
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
    print(f"ocho neuronas de en medio; la mejor sola acierta {pct(aciertos.max())} "
          f"y la red entera {pct(m['entera'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
