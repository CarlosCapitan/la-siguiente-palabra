#!/usr/bin/env python3
"""
Capítulo 3 — el cuadrado del colegio (Orden 18, 18 oct 2026).

Las trece edades colocadas no por los años que tienen sino por lo que dicen de ellas los dos
primeros comités de los tres a mano de `el_colegio.py`: a lo ancho, «¿al menos seis?» (no a la
izquierda, sí a la derecha); a lo alto, «¿como mucho doce?» (no abajo, sí arriba). Las de tres a
cinco caen en «no, sí»; las de trece a quince, en «sí, no»; las siete de primaria, todas juntas,
en «sí, sí»; y «no, no» queda vacía, con el rótulo «nadie». Una raya recta deja sola la esquina
de «sí, sí».

Las edades y las respuestas salen de los comités de `el_colegio.py` (importado), no están escritas
a mano. La figura no lleva más cifras que las edades ni título (el pie lo pone el libro).

Uso:
    python figura_cuadrado_del_colegio.py --selftest
    python figura_cuadrado_del_colegio.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../figuras/cuadrado_del_colegio.png"
ANCHO_ALTO = (6.2, 4.0)          # pulgadas, para una página de 6 por 9
PUNTOS = 200                     # puntos por pulgada

LIMITE_X = (-0.55, 1.55)         # lo que se ve del cuadrado, a lo ancho (las siete de «sí, sí» caben)
LIMITE = (-0.45, 1.45)           # y a lo alto
RAYA = (-1.0, 1.35)              # (pendiente, altura): y = -x + 1,35 deja sola la esquina de «sí, sí»
SEPARACION = 0.063               # entre marcas de una misma esquina, en unidades del cuadrado
TAMANO_MARCA = 9
ROTULO_VACIA = "nadie"

# ==========================================================

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import el_colegio as C


def votos():
    """Para cada edad, (lo que dice el primer comité, lo que dice el segundo): 0 o 1, de los comités
    a mano de el_colegio.COMITES."""
    (_, _, p1, l1), (_, _, p2, l2), _ = C.COMITES
    return [(C.dice_comite(p1, l1, (e,))[1], C.dice_comite(p2, l2, (e,))[1]) for e in C.EDADES]


def esquinas():
    """{(primero, segundo): [(edad, entra)]} con las cuatro esquinas, la vacía incluida."""
    r = {(a, b): [] for a in (0, 1) for b in (0, 1)}
    for e, (a, b), debe in zip(C.EDADES, votos(), C.DEBERIA):
        r[(a, b)].append((int(e), bool(debe == 1)))
    return r


def lado_de_la_raya(x, y):
    m, h = RAYA
    return y - (m * x + h) > 0


def dibujar():
    plt.rcParams["font.size"] = 9
    fig, ax = plt.subplots(figsize=ANCHO_ALTO)
    m, h = RAYA
    xs = [LIMITE_X[0], LIMITE_X[1]]
    ax.fill_between(xs, [m * x + h for x in xs], LIMITE[1], color="0.90", zorder=0)
    ax.plot(xs, [m * x + h for x in xs], "--", color="0.35", linewidth=1.3, zorder=2)
    ax.text(1.52, 1.40, "de este lado,\nlos que entran", fontsize=8, color="0.25", ha="right",
            va="top", zorder=4)
    for (a, b), edades in esquinas().items():
        if not edades:
            ax.text(a, b, ROTULO_VACIA, fontsize=9, color="0.4", style="italic", ha="center",
                    va="center", zorder=4)
            continue
        n = len(edades)
        for k, (e, entra) in enumerate(edades):
            x = a + (k - (n - 1) / 2) * SEPARACION * 1.9
            ax.plot(x, b, "o", markersize=TAMANO_MARCA, markerfacecolor="black" if entra else "white",
                    markeredgecolor="black", markeredgewidth=1.4, zorder=3)
            ax.text(x, b - 0.075, str(e), fontsize=8, ha="center", va="top", zorder=4)
    ax.set_xlim(*LIMITE_X)
    ax.set_ylim(*LIMITE)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no", "sí"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "sí"])
    ax.set_xlabel("lo que dice el primero: «¿al menos seis?»")
    ax.set_ylabel("lo que dice el segundo: «¿como mucho doce?»")
    ax.set_aspect("equal")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.5, 0.012, "relleno: entra en el colegio    ·    vacío: no entra",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(SALIDA, dpi=PUNTOS)
    print(f"Escrito {SALIDA}")


def selftest():
    fallos = []
    C.comprobar_entrada()
    es = esquinas()

    # 1. TEST NULO — la esquina de «no, no» queda vacía (nadie tiene menos de seis y más de doce a la
    #    vez), y con un solo comité (el primero) no hay raya que deje solas a las siete: de las
    #    que dicen «sí» al primero, hay seis que no entran.
    vacia = es[(0, 0)]
    dicen_si_primero = sum(len(v) for (a, _), v in es.items() if a == 1)
    print(f"[1] test nulo         la esquina «no, no» tiene {len(vacia)} edades; con solo el primero "
          f"dicen que sí {dicen_si_primero}, y entran 7")
    if vacia or dicen_si_primero == 7:
        fallos.append("test nulo: la esquina vacía no está vacía, o un solo comité ya separa a las siete")

    # 2. SEÑAL IMPLANTADA — las 13 edades caen en tres esquinas, y son las que dice la tabla de
    #    el_colegio: de 3 a 5 en «no, sí», de 6 a 12 en «sí, sí» y de 13 a 15 en «sí, no».
    ocupadas = {k: [e for e, _ in v] for k, v in es.items() if v}
    esperado = {(0, 1): [3, 4, 5], (1, 1): list(range(6, 13)), (1, 0): [13, 14, 15]}
    print(f"[2] señal implantada  esquinas ocupadas: {len(ocupadas)}; {ocupadas}")
    if ocupadas != esperado or sum(len(v) for v in ocupadas.values()) != 13:
        fallos.append(f"señal implantada: las esquinas son {ocupadas}, y se esperaba {esperado}")

    # 3. INVARIANTE — la raya deja a un lado exactamente las siete que entran, y la marca rellena
    #    es la de las que entran.
    puestos = [(a, b, e) for (a, b), v in es.items() for e, _ in v]
    al_lado = sorted(e for a, b, e in puestos if lado_de_la_raya(a, b))
    entran = sorted(int(e) for e, d in zip(C.EDADES, C.DEBERIA) if d == 1)
    rellenas = sorted(e for v in es.values() for e, s in v if s)
    print(f"[3] invariante        la raya deja a un lado {al_lado}; entran {entran}; rellenas {rellenas}")
    if al_lado != entran or rellenas != entran or len(entran) != 7:
        fallos.append("invariante: la raya no deja solas a las siete que entran")

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
    if args.selftest:
        sys.exit(selftest())
    C.comprobar_entrada()
    dibujar()


if __name__ == "__main__":
    main()
