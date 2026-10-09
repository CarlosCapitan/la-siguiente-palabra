#!/usr/bin/env python3
"""
Capítulo 3 — el sorteo del arranque, dibujado (Carlos, 1 de octubre de 2026).

Los 64 pesos de un solo comité, en su cuadrícula de ocho por ocho, en dos de las cinco tiradas de
`el_sorteo_del_arranque.py`: la que menos acierta y la que más. A la izquierda, como salen del
sorteo; a la derecha, después de las 20.000 correcciones. Se marcan los tres puntos de la tabla.

Qué tirada es la peor y la mejor, cuánto aciertan y los pesos de arranque de los tres puntos se
leen de `datos/salidas/el_sorteo_del_arranque.txt`; los 64 pesos salen de la misma cuenta
(`el_sorteo_del_arranque.una_tirada`, misma semilla), y el selftest comprueba que coinciden.

Uso:
    python figura_sorteo_del_arranque.py --selftest
    python figura_sorteo_del_arranque.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/el_sorteo_del_arranque.txt"
DESTINO = "../figuras/sorteo_del_arranque.png"
ALTO = 4.95                   # pulgadas
LADO_CUADRO = 1.25            # pulgadas de cada cuadrícula
X_CUADROS = (1.55, 3.05)      # pulgadas desde la izquierda: el sorteo, y después de entrenar
Y_CUADROS = (2.0, 0.3)        # pulgadas desde abajo: la peor tirada, y la mejor
PARECIDO_ARRANQUE_MAX = 0.35  # dos sorteos no se parecen: por encima de esto, algo los ata
PARECIDO_FINAL_MIN = 0.5      # dos entrenamientos de lo mismo sí se parecen al final

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

from infografia import COLOR, GRIS, Lienzo, ANCHO_PAGINA
from el_sorteo_del_arranque import una_tirada, indice, PUNTOS, PUNTO_SIN_TINTA, LADO

AQUI = Path(__file__).resolve().parent


def leer(ruta):
    """Las cinco filas de la tabla: tirada, semilla, los tres pesos de arranque y el acierto.
    Desde el 9 de octubre de 2026, de la tabla editorial de la salida, por su título (la
    salida dejó de imprimir renglones sangrados y este lector se había quedado atrás)."""
    from formato import leer_tablas
    titulo = "Lo que pone al azar la semilla: los pesos con los que arranca"
    T = leer_tablas(Path(ruta).read_text(encoding="utf-8"))
    assert titulo in T, f"Se esperaba la tabla «{titulo}» en {ruta}"
    rot, tabla, _ = T[titulo]
    assert rot[0] == "tirada" and rot[-1] == "acierta" and len(rot) == 6, rot
    num = lambda s: float(s.replace("+", "").replace(".", "").replace(",", ".")) if "," in s \
        else float(s.replace(".", ""))
    filas = []
    for t in tabla:
        assert t[-1].endswith(" %"), f"acierto inesperado: {t}"
        filas.append({"tirada": int(t[0]), "semilla": int(t[1].replace(".", "")),
                      "pesos": [num(x) for x in t[2:5]], "acierta": num(t[5][:-2])})
    assert len(filas) == 5, f"Se esperaban 5 tiradas en {ruta}; se encontraron {len(filas)}"
    return filas


def peor_y_mejor(filas):
    peor = min(filas, key=lambda f: f["acierta"])
    mejor = max(filas, key=lambda f: f["acierta"])
    return peor, mejor


def coma(x):
    return f"{x:.1f}".replace(".", ",")


def dibujar(peor, mejor, tiradas, paleta, ruta):
    lim_final = max(float(np.abs(t["final"]).max()) for t in tiradas)
    lim_arranque = max(float(np.abs(t["arranque"]).max()) for t in tiradas)
    assert lim_arranque <= 1.0, f"Se esperaba un sorteo entre -1 y 1; se encontró {lim_arranque}"
    L = Lienzo("El sorteo del arranque",
               "Los 64 pesos de un solo comité, en la peor y en la mejor de las cinco tiradas.\n"
               "Negro: tinta en ese punto empuja hacia «par»; blanco, hacia «impar»; gris\n"
               "medio, ni una cosa ni otra. Recuadrados, los tres puntos de la tabla. Cada\n"
               f"columna, con su escala: al arrancar no pasan de 1, y al final llegan a {coma(lim_final)}.",
               paleta, alto=ALTO)
    W, H = ANCHO_PAGINA, ALTO
    for x, rotulo in zip(X_CUADROS, ("como salen\ndel sorteo", "después de 20.000\ncorrecciones")):
        L.fig.text((x + LADO_CUADRO / 2) / W, (Y_CUADROS[0] + LADO_CUADRO + 0.08) / H, rotulo,
                   ha="center", va="bottom", fontsize=8.4, color=L.p.tinta, linespacing=1.1)
    for y, fila, t, cual in zip(Y_CUADROS, (peor, mejor), tiradas, ("la peor", "la mejor")):
        L.fig.text(0.18 / W, (y + LADO_CUADRO / 2) / H,
                   f"tirada {fila['tirada']},\n{cual}:\nacierta\n{coma(fila['acierta'])} %",
                   ha="left", va="center", fontsize=8.4, color=L.p.tinta, linespacing=1.15)
        for x, pesos, lim in ((X_CUADROS[0], t["arranque"], 1.0),
                              (X_CUADROS[1], t["final"], lim_final)):
            ax = L.fig.add_axes([x / W, y / H, LADO_CUADRO / W, LADO_CUADRO / H])
            ax.imshow(pesos.reshape(LADO, LADO), cmap="gray_r", vmin=-lim, vmax=lim)
            ax.set_xticks([]); ax.set_yticks([])
            for f, c in PUNTOS:
                for grosor, color in ((2.4, "black"), (1.0, "white")):
                    ax.add_patch(Rectangle((c - 1.5, f - 1.5), 1, 1, fill=False,
                                           edgecolor=color, linewidth=grosor))
    L.guardar(ruta)


def cargar(ruta):
    filas = leer(ruta)
    peor, mejor = peor_y_mejor(filas)
    tiradas = [una_tirada(peor["semilla"]), una_tirada(mejor["semilla"])]
    return filas, peor, mejor, tiradas


def selftest():
    fallos = []
    filas, peor, mejor, tiradas = cargar(AQUI / SALIDA)
    parecido = lambda a, b: float(np.corrcoef(a, b)[0, 1])

    # [1] TEST NULO — los dos sorteos no tienen nada que ver entre sí: el arranque no sabe nada
    #     de los dígitos. Si se parecieran, el dibujo de la izquierda diría algo que no es.
    r0 = parecido(tiradas[0]["arranque"], tiradas[1]["arranque"])
    print(f"[1] test nulo         parecido entre los dos sorteos: {r0:+.2f} "
          f"(tiene que quedar por debajo de {PARECIDO_ARRANQUE_MAX} en valor absoluto)")
    if abs(r0) > PARECIDO_ARRANQUE_MAX:
        fallos.append("test nulo: los dos sorteos se parecen; el arranque no sería al azar")

    # [2] SEÑAL — lo dibujado es lo de la tabla: los tres pesos de arranque y el acierto de las
    #     dos tiradas; y después de entrenar, las dos sí se parecen, porque aprendieron lo mismo.
    iguales = all(abs(round(t["arranque"][indice(f, c)], 2) - p) < 1e-9
                  for t, fila in zip(tiradas, (peor, mejor))
                  for (f, c), p in zip(PUNTOS, fila["pesos"]))
    aciertos = all(abs(round(100 * t["acierta"], 1) - fila["acierta"]) < 1e-9
                   for t, fila in zip(tiradas, (peor, mejor)))
    r1 = parecido(tiradas[0]["final"], tiradas[1]["final"])
    print(f"[2] señal             pesos y aciertos como en la tabla: {iguales and aciertos}; "
          f"parecido después de entrenar: {r1:+.2f} (al menos {PARECIDO_FINAL_MIN})")
    if not (iguales and aciertos):
        fallos.append("señal: lo dibujado no es lo de la tabla")
    if r1 < PARECIDO_FINAL_MIN:
        fallos.append("señal: dos entrenamientos de lo mismo no se parecen al final")

    # [3] INVARIANTE — el punto sin tinta en ningún dígito acaba con el peso del sorteo, en las
    #     dos tiradas; y la figura se dibuja en las dos paletas.
    p0 = indice(*PUNTO_SIN_TINTA)
    quieto = all(t["final"][p0] == t["arranque"][p0] for t in tiradas)
    for pal in (COLOR, GRIS):
        dibujar(peor, mejor, tiradas, pal, "/dev/null")
    print(f"[3] invariante        fila 1, columna 1 acaba con su peso del sorteo: {quieto}")
    if not quieto:
        fallos.append("invariante: el punto sin tinta ha cambiado de peso al entrenar")
    print()
    if fallos:
        for x in fallos:
            print("FALLA:", x)
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
    _, peor, mejor, tiradas = cargar(AQUI / SALIDA)
    dibujar(peor, mejor, tiradas, GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
