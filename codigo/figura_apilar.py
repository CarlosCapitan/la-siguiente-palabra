#!/usr/bin/env python3
"""
Capítulos 2 y 3 — la figura de apilar: la segunda capa no ve la foto, ve lo que la
primera dijo sobre la foto.

Dos paneles:
  Izquierda  el montaje real de la red que resuelve el o exclusivo: dos interruptores,
             dos neuronas en medio, una luz. El grosor de cada flecha es el peso que
             de verdad aprendió la red; la línea de puntos es un peso que frena.
  Derecha    las mismas cuatro situaciones, dibujadas ya no según los interruptores sino
             según lo que dijeron las dos neuronas de en medio. En ese cuadrado nuevo, una
             sola raya basta, y la raya dibujada es la que de verdad usa la última neurona.

Nada de esto está dibujado a mano: los pesos, las posiciones y la raya salen de la red
entrenada. Lo que la figura afirma, el selftest lo comprueba por fuerza bruta.

Uso:
    python figura_apilar.py
    python figura_apilar.py --selftest
"""

# ======================= CONSTANTES =======================

SALIDA = "../figuras/apilar.png"
ANCHO_ALTO = (6.2, 3.2)          # pulgadas, para una página de 6 por 9
PUNTOS = 200                     # puntos por pulgada

PAREJA_BUSCADA = ("al menos uno subido", "los dos subidos")   # el reparto más repetido, medido aparte
MAX_SEMILLAS = 60                # cuántas probar hasta dar con ese reparto
UMBRAL = 0.5
GROSOR_MAXIMO = 4.5              # grosor de la flecha del peso más grande
REJILLA = 61                     # finura de la búsqueda a lo bruto de rayas
SOLAPE = 0.08                    # dos puntos más cerca que esto son el mismo sitio
SEPARACION = 0.045               # cuánto se separan para que se vean los dos

ETIQUETAS_ENTRADA = ["el de\nabajo", "el de\narriba"]
ETIQUETA_SALIDA = "la luz"

# ==========================================================

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from retropropagacion import Red, TABLA_XOR, Y_XOR
from que_inventan_las_capas import regla_de, SEMILLA_BASE, TASA, PASOS, OCULTAS
from figura_una_raya import hay_raya_que_separa, ENCIENDE_XOR


def red_representativa():
    """La primera red, de una lista fija de semillas, que resuelve el o exclusivo
    repartiéndose el trabajo de la manera más repetida. Fija y reproducible: no se
    elige la que queda bonita, se elige la primera que cumple."""
    semillas = np.random.default_rng(SEMILLA_BASE).integers(1, 2**31 - 1, size=MAX_SEMILLAS)
    for s in semillas:
        red = Red([2, OCULTAS, 1], semilla=int(s)).entrenar(TABLA_XOR, Y_XOR, TASA, PASOS)
        act = red.adelante(TABLA_XOR)
        if not ((act[-1].ravel() > UMBRAL) == (Y_XOR > UMBRAL)).all():
            continue
        reglas = [regla_de(act[1][:, j]) for j in range(OCULTAS)]
        if tuple(sorted(reglas)) == tuple(sorted(PAREJA_BUSCADA)):
            return red, act[1], reglas, int(s)
    raise SystemExit(
        f"Se esperaba encontrar en {MAX_SEMILLAS} semillas una red con el reparto "
        f"{PAREJA_BUSCADA}; no apareció ninguna. Mire que_inventan_las_capas.py antes de tocar esto.")


def hay_raya_en(puntos, encienden):
    """Lo mismo que hay_raya_que_separa, pero para puntos cualesquiera, no para las
    cuatro esquinas del cuadrado."""
    assert puntos.shape == (4, 2), \
        f"Se esperaban 4 puntos de 2 coordenadas; se encontró {puntos.shape}"
    lim = float(np.abs(puntos).max()) + 1.0
    rejilla = np.linspace(-6, 6, REJILLA)
    for a in rejilla:
        for b in rejilla:
            for c in np.linspace(-6 * lim, 6 * lim, REJILLA):
                lados = (a * puntos[:, 0] + b * puntos[:, 1] + c) > 0
                if (lados == encienden).all() or (lados == ~encienden).all():
                    return True
    return False


def panel_montaje(ax, red, reglas):
    columnas = {"entrada": 0.0, "medio": 1.0, "salida": 2.0}
    filas = {"entrada": [0.75, -0.75], "medio": [0.75, -0.75], "salida": [0.0]}
    maximo = max(np.abs(red.W[0]).max(), np.abs(red.W[1]).max())

    def dibuja_flechas(W, col_a, col_b):
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                peso = W[i, j]
                ax.plot([columnas[col_a] + 0.16, columnas[col_b] - 0.16],
                        [filas[col_a][i], filas[col_b][j]],
                        linestyle="-" if peso > 0 else ":",
                        linewidth=0.5 + GROSOR_MAXIMO * abs(peso) / maximo,
                        color="0.35", zorder=1)

    dibuja_flechas(red.W[0], "entrada", "medio")
    dibuja_flechas(red.W[1], "medio", "salida")

    for col, etiquetas in [("entrada", ETIQUETAS_ENTRADA),
                           ("medio", reglas),
                           ("salida", [ETIQUETA_SALIDA])]:
        for k, texto in enumerate(etiquetas):
            x, y = columnas[col], filas[col][k]
            ax.plot(x, y, "o", markersize=17, markerfacecolor="white",
                    markeredgecolor="black", markeredgewidth=1.6, zorder=2)
            if col == "medio":
                arriba = filas[col][k] > 0
                ax.text(x, y + (0.33 if arriba else -0.33), f"«{texto}»", ha="center",
                        va="bottom" if arriba else "top", fontsize=8.5)
            elif col == "entrada":
                ax.text(x - 0.24, y, texto, ha="right", va="center", fontsize=8.5)
            else:
                ax.text(x + 0.24, y, texto, ha="left", va="center", fontsize=8.5)

    ax.set_title("el montaje\ndos neuronas de más", fontsize=10)
    ax.set_xlim(-0.95, 2.75)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")


def panel_cuadrado_nuevo(ax, red, medio):
    enciende = Y_XOR > UMBRAL
    # Dos de las cuatro situaciones acaban en el mismo punto: las dos que encienden la
    # luz. Eso no es un defecto del dibujo, es el trabajo que ha hecho la capa de en
    # medio, así que se separan un pelo para que se vean las dos y se dice en voz alta.
    juntos = [k for k in range(4) if any(
        j != k and np.hypot(*(medio[k] - medio[j])) < SOLAPE for j in range(4))]
    for k in range(4):
        dx = 0.0
        if k in juntos:
            dx = SEPARACION * (-1 if juntos.index(k) == 0 else 1)
        ax.plot(medio[k, 0] + dx, medio[k, 1], "o", markersize=13,
                markerfacecolor="black" if enciende[k] else "white",
                markeredgecolor="black", markeredgewidth=1.8, zorder=3)
    if len(juntos) == 2:
        x, y = medio[juntos[0], 0], medio[juntos[0], 1]
        ax.text(x, y - 0.13, "las dos que encienden\nla luz, en el mismo sitio",
                fontsize=7.5, ha="center", va="top", color="0.3")

    # La raya que de verdad usa la última neurona: donde deja de decir no y empieza a decir sí.
    w, b = red.W[1].ravel(), red.b[1][0]
    xs = np.linspace(-0.15, 1.15, 50)
    if abs(w[1]) > 1e-9:
        ax.plot(xs, -(w[0] * xs + b) / w[1], "--", color="0.35", linewidth=1.6, zorder=2)

    ax.set_title("el cuadrado, redibujado\npor las dos de en medio", fontsize=10)
    ax.set_xlabel("lo que dijo la primera", fontsize=9)
    ax.set_ylabel("lo que dijo la segunda", fontsize=9)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.45, 1.2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no", "sí"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "sí"], fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def dibujar():
    red, medio, reglas, semilla = red_representativa()
    fig, (izq, der) = plt.subplots(1, 2, figsize=ANCHO_ALTO)
    panel_montaje(izq, red, reglas)
    panel_cuadrado_nuevo(der, red, medio)
    fig.text(0.5, 0.015, "negro: la luz se enciende  ·  blanco: la luz no se enciende",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(SALIDA, dpi=PUNTOS)
    plt.close(fig)
    return red, medio, reglas, semilla


# ============================ SELFTEST ============================

def selftest():
    fallos = []
    red, medio, reglas, semilla = red_representativa()
    enciende = Y_XOR > UMBRAL

    # [1] Test nulo: en el cuadrado de siempre —el de los interruptores— no hay raya.
    #     Es el resultado del capítulo 2, y si aquí saliera que sí, todo esto sobra.
    nulo = hay_raya_que_separa(ENCIENDE_XOR)
    print(f"[1] test nulo         con los interruptores como ejes, ¿hay raya?: "
          f"{'sí' if nulo else 'no'}")
    if nulo:
        fallos.append("test nulo: dice que una raya separa el o exclusivo en el cuadrado "
                      "de los interruptores, y no la hay")

    # [2] Señal implantada: las etiquetas del dibujo tienen que ser lo que las neuronas
    #     calculan de verdad, no lo que a uno le gustaría que calcularan.
    print(f"[2] señal implantada  la red (semilla {semilla}) se repartió el trabajo en: "
          f"{reglas[0]} / {reglas[1]}")
    if tuple(sorted(reglas)) != tuple(sorted(PAREJA_BUSCADA)):
        fallos.append(f"señal implantada: esperaba {PAREJA_BUSCADA}; encontré {tuple(reglas)}")

    # [3] Invariante del dominio: en el cuadrado nuevo SÍ hay raya. Ésta es exactamente
    #     la afirmación que hace la figura, comprobada a lo bruto.
    hay = hay_raya_en(medio, enciende)
    juntos = sum(1 for k in range(4) for j in range(k + 1, 4)
                 if np.hypot(*(medio[k] - medio[j])) < SOLAPE)
    print(f"[3] invariante        ¿hay raya en el cuadrado nuevo?: {'sí' if hay else 'no'}; "
          f"situaciones que caen en el mismo sitio: {juntos}")
    if not hay:
        fallos.append("invariante: la figura dice que en el cuadrado nuevo una raya basta, "
                      "y la búsqueda a lo bruto no encuentra ninguna")
    if juntos != 1:
        fallos.append(f"invariante: esperaba exactamente un par de situaciones aplastadas en "
                      f"el mismo punto (las dos que encienden la luz); encontré {juntos} pares")

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
    args = p.parse_args()
    if args.selftest:
        return selftest()
    red, medio, reglas, semilla = dibujar()
    print(f"escrito {SALIDA}")
    print(f"red de la semilla {semilla}; las de en medio acabaron siendo "
          f"«{reglas[0]}» y «{reglas[1]}»")
    return 0


if __name__ == "__main__":
    sys.exit(main())
