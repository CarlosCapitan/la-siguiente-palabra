#!/usr/bin/env python3
"""
Capítulo 2 — la figura del bautizo: qué hay dentro de las palabras «neurona
artificial» y «red neuronal».

Cuatro paneles, y los tres primeros salen de un perceptrón entrenado de verdad
sobre dígitos manuscritos, con la regla de Rosenblatt de `perceptron.py`:

  1. Un cuatro, tal como lo ve la máquina: sesenta y cuatro puntos, uno por jurado.
  2. Cuánto se le hace caso a cada punto: los pesos que aprendió, dibujados en el
     mismo cuadrado de ocho por ocho. Negro empuja hacia «es un cuatro», blanco
     hacia «no lo es», gris medio es un punto que no cuenta.
  3. El total de puntos de dos ejemplos reales contra el listón. Uno lo pasa y
     otro no, y eso es la decisión entera.
  4. Y una red neuronal es muchas de éstas, conectadas unas a la salida de otras.

Aquí no hay números inventados: las alturas de las barras y la altura del listón
son las que tiene la máquina entrenada, y el selftest lo comprueba recalculándolas
por su cuenta.

Uso:
    python figura_una_neurona.py
    python figura_una_neurona.py --selftest
"""

# ======================= CONSTANTES =======================

SALIDA = "../figuras/una_neurona.png"
ANCHO_ALTO = (6.2, 4.6)          # pulgadas, para una página de 6 por 9
PUNTOS = 200
# La caja de texto de una página de 6 por 9 mide 4,45 pulgadas, así que xelatex
# encoge esta figura a un 72 %. Las letras se dibujan más grandes en esa proporción
# para que en el papel salgan del tamaño que se quiere, y no más pequeñas.
ESCALA = 1.0 / 0.72

DIGITO_SI = 4                    # la pregunta del tribunal en el libro: ¿esto es un cuatro?
DIGITO_NO = 9                    # el que más se le parece escrito a mano
LADO = 8                         # los dibujos son de ocho puntos por ocho
SEMILLA_FIGURA = 20260916
TOLERANCIA = 1e-9                # al recalcular el total a mano

# El rótulo de cada cuadro y la letra pequeña que va debajo.
# El negro y el blanco NO significan lo mismo en el primer cuadro que en el segundo: en
# el primero es tinta y en el segundo es cuánto empuja ese punto. Antes había una sola
# línea al pie de toda la figura explicando el segundo, y desde el primero se leía como
# si hablara de él. Cada cuadro lleva ahora su propia clave, debajo y pegada a él.
TITULOS = [
    "el dibujo, punto a punto",
    "cuánto cuenta cada punto",
    "el total contra el listón",
    "y una red es muchas de éstas",
]
# Cuánto hay que bajar el pie de cada cuadro: el tercero lleva debajo los nombres de las
# dos barras, y si no se baja, el pie se le echa encima.
BAJADA_PIE = [0.022, 0.022, 0.090, 0.022]
PIES = [
    "negro: donde hay tinta",
    "negro: empuja hacia el «sí»\nblanco: empuja hacia el «no»",
    "cada barra es el total de sumar los 64 puntos",
    "cada círculo es un aparato como el de arriba",
]

RED_COLUMNAS = [3, 4, 4, 2]      # el esquema de la derecha: solo para enseñar la forma

# ==========================================================

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from perceptron import cargar_digitos, entrenar, acierto


def entrenar_el_tribunal(semilla=SEMILLA_FIGURA, permutar=False):
    """Un perceptrón que separa DIGITO_SI de DIGITO_NO, con la regla de Rosenblatt.

    Con permutar=True se le barajan las etiquetas: es el test nulo."""
    X, t = cargar_digitos()
    mask = (t == DIGITO_SI) | (t == DIGITO_NO)
    Xs = X[mask]
    ys = np.where(t[mask] == DIGITO_SI, 1, -1)
    rng = np.random.default_rng(semilla)
    if permutar:
        ys = rng.permutation(ys)
    idx = rng.permutation(len(Xs))
    corte = int(len(Xs) * 0.7)
    tr, te = idx[:corte], idx[corte:]
    w, b, _ = entrenar(Xs[tr], ys[tr])
    return {"w": w, "b": b, "X": Xs, "y": ys, "prueba": (Xs[te], ys[te]),
            "acierto": acierto(Xs[te], ys[te], w, b)}


def ejemplos(m):
    """Un dibujo de cada clase que la máquina acierta, para que el panel 3 no haga trampa."""
    total = m["X"] @ m["w"] + m["b"]
    i_si = next(i for i in range(len(m["y"])) if m["y"][i] == 1 and total[i] > 0)
    i_no = next(i for i in range(len(m["y"])) if m["y"][i] == -1 and total[i] <= 0)
    return i_si, i_no


def puntos_y_liston(m, indices):
    """Los puntos de cada ejemplo y la altura del listón, en las mismas unidades.

    La máquina decide con (puntos + sesgo) > 0. Dicho como lo dice el libro: los
    puntos tienen que llegar al listón, y el listón está en menos el sesgo."""
    puntos = m["X"][list(indices)] @ m["w"]
    return puntos, -m["b"]


def panel_dibujo(ax, imagen):
    ax.imshow(imagen.reshape(LADO, LADO), cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.8)


def panel_pesos(ax, w):
    lim = np.abs(w).max()
    ax.imshow(w.reshape(LADO, LADO), cmap="gray_r", vmin=-lim, vmax=lim)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.8)


def panel_liston(ax, puntos, liston):
    etiquetas = ["un cuatro", "un nueve"]
    colores = ["0.15", "0.75"]
    barras = ax.bar([0, 1], puntos, width=0.55, color=colores,
                    edgecolor="black", linewidth=1.0)
    ax.axhline(liston, linestyle="--", color="black", linewidth=1.3)
    alto = max(puntos.max(), liston)
    bajo = min(puntos.min(), liston, 0)
    margen = 0.22 * (alto - bajo)
    ax.text(-0.62, liston - 0.04 * (alto - bajo), "el listón", fontsize=10.4,
            va="top", ha="left")
    for x, p in enumerate(puntos):
        dice = "dice sí" if p > liston else "dice no"
        ax.text(x, p + (0.05 if p >= 0 else -0.05) * (alto - bajo), dice,
                ha="center", fontsize=10.4,
                va="bottom" if p >= 0 else "top")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["un cuatro", "un nueve"], fontsize=10.4)
    ax.set_xlim(-0.85, 1.7)
    ax.set_ylim(bajo - margen * 1.7, alto + margen * 1.9)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)


def panel_red(ax):
    """Muchas de éstas, conectadas. Una va rellena: es la de los paneles de arriba,
    para que se vea que cada círculo del dibujo es el aparato entero, no un punto."""
    pos = {}
    for c, n in enumerate(RED_COLUMNAS):
        pos[c] = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)
    for c in range(len(RED_COLUMNAS) - 1):
        for y0 in pos[c]:
            for y1 in pos[c + 1]:
                ax.plot([c + 0.16, c + 1 - 0.16], [y0, y1],
                        color="0.65", linewidth=0.7, zorder=1)
    marcada = (1, len(pos[1]) - 1)        # columna, índice dentro de la columna (la de arriba)
    for c in range(len(RED_COLUMNAS)):
        for k, y in enumerate(pos[c]):
            destacada = (c, k) == marcada
            ax.plot(c, y, "o", markersize=13,
                    markerfacecolor="black" if destacada else "white",
                    markeredgecolor="black", markeredgewidth=1.3, zorder=2)
    ym = pos[marcada[0]][marcada[1]]
    ax.annotate("ésta es la de arriba, entera", xy=(marcada[0], ym + 0.2),
                xytext=(1.4, 3.05), fontsize=9.7, ha="center", va="bottom",
                color="0.25", arrowprops=dict(arrowstyle="-", color="0.5", lw=0.9))
    ax.set_xlim(-0.75, len(RED_COLUMNAS) - 0.25)
    ax.set_ylim(-2.3, 4.0)
    ax.axis("off")


def dibujar():
    m = entrenar_el_tribunal()
    i_si, i_no = ejemplos(m)
    puntos, liston = puntos_y_liston(m, (i_si, i_no))

    fig = plt.figure(figsize=ANCHO_ALTO)
    gs = GridSpec(2, 2, height_ratios=[1, 1.05], hspace=0.52, wspace=0.28, figure=fig)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.89, bottom=0.14)
    ejes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    panel_dibujo(ejes[0], m["X"][i_si])
    panel_pesos(ejes[1], m["w"])
    panel_liston(ejes[2], puntos, liston)
    panel_red(ejes[3])
    for ax, titulo, pie, bajada in zip(ejes, TITULOS, PIES, BAJADA_PIE):
        c = ax.get_position()
        fig.text(c.x0 + c.width / 2, c.y1 + 0.030, titulo, ha="center", va="bottom",
                 fontsize=11.0)
        if pie:
            fig.text(c.x0 + c.width / 2, c.y0 - bajada, pie, ha="center", va="top",
                     fontsize=8.6, color="0.35")
    fig.savefig(SALIDA, dpi=PUNTOS)
    plt.close(fig)
    return m, puntos, liston


# ============================ SELFTEST ============================

def selftest():
    fallos = []
    m = entrenar_el_tribunal()
    i_si, i_no = ejemplos(m)
    puntos, liston = puntos_y_liston(m, (i_si, i_no))

    # [1] Test nulo: con las etiquetas barajadas no hay nada que aprender, y el
    #     acierto tiene que caerse al azar. Si no se cae, la figura estaría
    #     dibujando pesos que no significan nada.
    nulo = entrenar_el_tribunal(permutar=True)
    print(f"[1] test nulo         etiquetas barajadas: acierto {nulo['acierto']:.3f} "
          f"(azar = 0,5); de verdad: {m['acierto']:.3f}")
    if nulo["acierto"] > 0.7:
        fallos.append(f"test nulo: con etiquetas barajadas acierta {nulo['acierto']:.3f}, "
                      f"y no debería pasar del azar")

    # [2] Señal implantada: las alturas que dibuja la figura tienen que ser las que
    #     de verdad usa la máquina. Se recalculan a mano, punto por punto, sin usar
    #     la misma operación.
    a_mano = np.array([sum(float(x) * float(p) for x, p in zip(m["X"][i], m["w"]))
                       for i in (i_si, i_no)])
    dif = float(np.abs(a_mano - puntos).max())
    print(f"[2] señal implantada  totales recalculados a mano: diferencia máxima {dif:.2e}")
    if dif > TOLERANCIA:
        fallos.append(f"señal implantada: la figura dibuja unos totales y a mano salen otros, "
                      f"se diferencian en {dif:.2e}")

    # [3] Invariante del dominio: lo que dice el dibujo tiene que ser lo que decide
    #     la máquina. El cuatro pasa el listón y el nueve no; y si no fuera así, la
    #     figura estaría enseñando una decisión que la máquina no toma.
    veredicto = (puntos > liston)
    real = ((m["X"][[i_si, i_no]] @ m["w"] + m["b"]) > 0)
    print(f"[3] invariante        el cuatro pasa el listón: {'sí' if veredicto[0] else 'no'}; "
          f"el nueve lo pasa: {'sí' if veredicto[1] else 'no'}")
    if not (veredicto[0] and not veredicto[1]):
        fallos.append("invariante: la figura tiene que enseñar un ejemplo que pasa el listón "
                      "y otro que no, y no es el caso")
    if not (veredicto == real).all():
        fallos.append("invariante: el listón dibujado no separa igual que la máquina")

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
    m, puntos, liston = dibujar()
    print(f"escrito {SALIDA}")
    print(f"perceptrón que separa el {DIGITO_SI} del {DIGITO_NO}: "
          f"acierto {m['acierto']:.3f} sobre los ejemplos de prueba")
    return 0


if __name__ == "__main__":
    sys.exit(main())
