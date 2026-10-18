#!/usr/bin/env python3
"""
Capítulo 2 — la figura del colegio (Orden 16, 18 oct 2026).

Dos reglas de edades, una encima de otra, de 3 a 15 años, con una marca por año y su edad escrita
debajo. Arriba, «sí de seis en adelante»: entran de 6 a 15 y un solo corte, entre el 5 y el 6,
deja a cada lado lo que toca. Abajo, el colegio, «sí de seis a doce»: entran de 6 a 12 y hacen
falta dos cortes, entre el 5 y el 6 y entre el 12 y el 13. Un comité tiene un solo listón, y un
listón es un solo corte.

Qué entra y qué no sale de `el_colegio.py` (las edades y las respuestas de las dos reglas), no
está escrito a mano; los cortes se cuentan de dónde cambia la respuesta de una edad a la
siguiente. La figura no lleva más cifras que las edades ni título (el pie lo pone el libro).

Uso:
    python figura_el_colegio.py --selftest
    python figura_el_colegio.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../figuras/el_colegio.png"
ANCHO_ALTO = (6.2, 3.4)          # pulgadas, para una página de 6 por 9
PUNTOS = 200                     # puntos por pulgada

REGLAS = ["de 6 en adelante", "de 6 a 12"]                # las dos que se dibujan, como las nombra el_colegio
ROTULOS = ["sí de seis en adelante", "el colegio: sí de seis a doce"]
CORTES_ESPERADOS = [1, 2]
ROTULO_DOS = "hacen falta dos"

TAMANO_MARCA = 15
ALTURA_CORTE = 0.62              # hasta dónde sube cada corte; la línea de «hacen falta dos» cuelga de ahí

# ==========================================================

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import el_colegio as C


def entran(regla):
    """Para cada edad de C.EDADES, True si la regla dice que entra."""
    return [bool(v > 0) for v in C.etiquetas(regla)]


def cortes(entra):
    """Los sitios donde cambia la respuesta de una edad a la siguiente: [(edad antes, edad después)]."""
    return [(int(C.EDADES[i]), int(C.EDADES[i + 1])) for i in range(len(entra) - 1)
            if entra[i] != entra[i + 1]]


def regla(ax, entra, rotulo, doble):
    edades = [int(e) for e in C.EDADES]
    for e, si in zip(edades, entra):
        ax.plot(e, 0, "o", markersize=TAMANO_MARCA, markerfacecolor="black" if si else "white",
                markeredgecolor="black", markeredgewidth=1.6, zorder=3)
    ax.set_xticks(edades)
    ax.set_xticklabels([str(e) for e in edades])
    ax.set_yticks([])
    ax.set_xlim(edades[0] - 0.7, edades[-1] + 0.7)
    ax.set_ylim(-0.55, 0.95)
    for lado in ("top", "right", "left", "bottom"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(length=0, pad=8)
    ax.text(edades[0] - 0.6, 0.86, rotulo, fontsize=9.5, ha="left", va="top")
    cs = cortes(entra)
    for antes, despues in cs:
        x = (antes + despues) / 2
        ax.plot([x, x], [-0.36, ALTURA_CORTE], "--", color="0.30", linewidth=1.3, zorder=2)
    if doble:
        (a0, d0), (a1, d1) = cs
        x0, x1 = (a0 + d0) / 2, (a1 + d1) / 2
        ax.plot([x0, x1], [ALTURA_CORTE, ALTURA_CORTE], "-", color="0.30", linewidth=1.0, zorder=2)
        ax.text((x0 + x1) / 2, ALTURA_CORTE + 0.04, ROTULO_DOS, fontsize=8.5, color="0.25",
                ha="center", va="bottom")
    return cs


def dibujar():
    plt.rcParams["font.size"] = 9
    fig, ejes = plt.subplots(2, 1, figsize=ANCHO_ALTO)
    for ax, r, rot, esperados in zip(ejes, REGLAS, ROTULOS, CORTES_ESPERADOS):
        cs = regla(ax, entran(r), rot, doble=esperados == 2)
        assert len(cs) == esperados, f"Se esperaban {esperados} cortes en «sí {r}»; hay {len(cs)}"
    fig.tight_layout(rect=[0, 0.08, 1, 1], h_pad=1.2)
    fig.text(0.5, 0.015, "relleno: entra en el colegio    ·    vacío: no entra",
             ha="center", fontsize=8.5, color="0.3")
    fig.savefig(SALIDA, dpi=PUNTOS)
    print(f"Escrito {SALIDA}")


def selftest():
    fallos = []
    # 1. TEST NULO — «sí de seis en adelante» tiene un solo corte, no dos: donde no hay
    #    imposibilidad, la figura no dibuja dos cortes.
    n_nulo = len(cortes(entran("de 6 en adelante")))
    print(f"[1] test nulo         «sí de seis en adelante» tiene {n_nulo} corte")
    if n_nulo != 1:
        fallos.append(f"test nulo: «de 6 en adelante» tiene {n_nulo} cortes, y debería tener uno")

    # 2. SEÑAL IMPLANTADA — el colegio, «sí de seis a doce», tiene dos cortes, entre el 5 y el 6 y
    #    entre el 12 y el 13; y una regla fabricada con tres tramos tiene tres (contar cortes
    #    funciona).
    cs = cortes(entran("de 6 a 12"))
    fabricada = cortes([False, True, True, False, False, True])
    print(f"[2] señal implantada  el colegio tiene los cortes {cs}; una regla fabricada con tres "
          f"tramos tiene {len(fabricada)} cortes")
    if cs != [(5, 6), (12, 13)] or len(fabricada) != 3:
        fallos.append(f"señal implantada: cortes {cs} (se esperaba [(5, 6), (12, 13)]) y {len(fabricada)} "
                      f"en la fabricada (se esperaban 3)")

    # 3. INVARIANTE — lo que entra sale de el_colegio.py: las edades son las 13 de 3 a 15, entran
    #    exactamente las de DEBERIA (de 6 a 12), y «de 6 en adelante» entra de 6 a 15.
    edades = [int(e) for e in C.EDADES]
    dentro = [e for e, s in zip(edades, entran("de 6 a 12")) if s]
    dentro_arriba = [e for e, s in zip(edades, entran("de 6 en adelante")) if s]
    igual = dentro == [int(e) for e, d in zip(C.EDADES, C.DEBERIA) if d == 1]
    print(f"[3] invariante        {len(edades)} edades de {edades[0]} a {edades[-1]}; entran de "
          f"{dentro[0]} a {dentro[-1]} (las de el_colegio.DEBERIA: {'sí' if igual else 'NO'}) y, "
          f"arriba, de {dentro_arriba[0]} a {dentro_arriba[-1]}")
    if len(edades) != 13 or not igual or dentro_arriba != list(range(6, 16)):
        fallos.append("invariante: las edades o lo que entra no son los de el_colegio.py")

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
    C.comprobar_entrada()
    if args.selftest:
        sys.exit(selftest())
    dibujar()


if __name__ == "__main__":
    main()
