#!/usr/bin/env python3
"""
Capítulo 4 — la red del pasillo, con sus números (L24, hallazgos A07 y A12).

La red que resuelve el o exclusivo, dibujada con lo que aprendió escrito encima: el peso de
cada línea y el listón de cada neurona. Con los números a la vista se ve por qué dos neuronas
que reciben los mismos dos interruptores, con líneas parecidas, preguntan cosas distintas: la
diferencia está en el listón (lo mismo que el capítulo 2 enseñó con el listón en 0,5 y en 1,5).

Los números NO se deciden aquí: se leen del bloque 2 de `datos/salidas/culpa_hacia_atras.txt`.

Uso:
    python figura_la_red_del_pasillo.py --selftest
    python figura_la_red_del_pasillo.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/culpa_hacia_atras.txt"
DESTINO = "../figuras/la_red_del_pasillo.png"
ALTO = 3.95                    # pulgadas
GROSOR_MAXIMO = 3.2            # grosor de la línea del peso más grande
RADIO = 5.2                    # radio de cada neurona, en unidades del lienzo
X_ENTRADA, X_MEDIO, X_FINAL = 13.0, 50.0, 85.0
NOMBRES_ENTRADA = ["el de abajo", "el de arriba"]
NOMBRES_MEDIO = ["la primera", "la segunda"]
PREGUNTAS = ["«¿hay al menos\nuno subido?»", "«¿están los dos\nsubidos?»"]

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

from matplotlib.patches import Circle, FancyBboxPatch

from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def numero(s):
    return float(s.replace("+", "").replace(",", "."))


def leer(ruta, bloque="2. LOS PESOS Y LOS LISTONES QUE APRENDIÓ"):
    """Del bloque 2: pesos[neurona][de] y listones[neurona], con neurona en
    («la primera», «la segunda», «la final»)."""
    texto = Path(ruta).read_text(encoding="utf-8")
    assert bloque in texto, f"se esperaba el bloque «{bloque}» en {ruta}"
    trozo = texto.split(bloque, 1)[1].split("\n\n3.", 1)[0]
    pesos, listones, actual = {}, {}, None
    for linea in trozo.splitlines():
        m = re.match(r"^(la primera|la segunda|la final)?\s+(el de abajo|el de arriba|la primera|la segunda)"
                     r"\s+([+-]\d+,\d+)(?:\s+(\d+,\d+))?\s*$", linea)
        if not m:
            continue
        if m.group(1):
            actual = m.group(1)
        pesos.setdefault(actual, {})[m.group(2)] = numero(m.group(3))
        if m.group(4):
            listones[actual] = numero(m.group(4))
    assert sorted(pesos) == ["la final", "la primera", "la segunda"], f"neuronas leídas: {sorted(pesos)}"
    assert all(len(v) == 2 for v in pesos.values()), f"se esperaban dos líneas por neurona: {pesos}"
    assert sorted(listones) == sorted(pesos), f"se esperaba un listón por neurona: {listones}"
    return pesos, listones


def que_pregunta(pesos, listones, neurona):
    """Qué montaje hace la neurona con sus pesos y su listón, con los interruptores a 0 y 1:
    en qué posiciones pasa del listón. Sale de los números, no de la etiqueta."""
    p = pesos[neurona]
    return tuple(int(a * p["el de abajo"] + b * p["el de arriba"] > listones[neurona])
                 for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)))


def dibujar_red(L, pesos, listones, y_arriba, y_abajo, rotulos_linea=None, texto_neurona=None,
                flechas_atras=False):
    """La red: entradas, dos de en medio y la final. rotulos_linea[(neurona, de)] es el texto
    de cada línea; texto_neurona[neurona] lo que va debajo de cada neurona."""
    p, ax = L.p, L.ax
    y_medio = (y_arriba + y_abajo) / 2
    pos = {"el de arriba": (X_ENTRADA, y_arriba), "el de abajo": (X_ENTRADA, y_abajo),
           "la primera": (X_MEDIO, y_arriba), "la segunda": (X_MEDIO, y_abajo),
           "la final": (X_FINAL, y_medio)}
    maximo = max(abs(v) for d in pesos.values() for v in d.values())
    for destino, entradas in pesos.items():
        for origen, peso in entradas.items():
            (x0, y0), (x1, y1) = pos[origen], pos[destino]
            ax.plot([x0 + RADIO, x1 - RADIO], [y0, y1], color=p.acento,
                    linestyle="-" if peso > 0 else (0, (1.2, 1.4)),
                    linewidth=0.6 + GROSOR_MAXIMO * abs(peso) / maximo, zorder=1,
                    solid_capstyle="butt")
            if flechas_atras:
                t = 0.42
                xa, ya = x0 + RADIO + (x1 - x0 - 2 * RADIO) * t, y0 + (y1 - y0) * t
                ax.annotate("", xy=(xa - 2.2, ya - (y1 - y0) / (x1 - x0) * 2.2), xytext=(xa, ya),
                            arrowprops=dict(arrowstyle="-|>", color=p.tinta, lw=1.0), zorder=5)
            if rotulos_linea:
                texto = rotulos_linea[(destino, origen)]
                # el rótulo va cerca del destino, en el lado de fuera, para que las dos líneas
                # que se cruzan en medio no se pisen
                t = 0.70
                xr = x0 + RADIO + (x1 - x0 - 2 * RADIO) * t
                yr = y0 + (y1 - y0) * t
                recta = abs(y1 - y0) < 1e-6
                if recta:
                    yr += 2.6 if y0 > y_medio else -2.6
                else:
                    yr += 2.4 if y1 > y0 else -2.4
                    xr += 1.0
                ax.text(xr, yr, texto, ha="center", va="center", fontsize=7.6, color=p.tinta,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"),
                        zorder=4)
    for nombre, (x, y) in pos.items():
        if nombre in NOMBRES_ENTRADA:
            # los interruptores no son comités: van en un cuadrado, no en un círculo
            ax.add_patch(FancyBboxPatch((x - RADIO, y - RADIO), 2 * RADIO, 2 * RADIO,
                                        boxstyle="round,pad=0,rounding_size=1.0",
                                        facecolor=p.fondo, edgecolor=p.tinta, linewidth=1.2, zorder=3))
        else:
            ax.add_patch(Circle((x, y), RADIO, facecolor="white", edgecolor=p.tinta, linewidth=1.2,
                                zorder=3))
        if nombre in NOMBRES_ENTRADA:
            ax.text(x, y, nombre.replace("el de ", "el de\n"), ha="center", va="center",
                    fontsize=6.4, color=p.tinta, zorder=4, linespacing=1.0)
        else:
            corto = {"la primera": "la\nprimera", "la segunda": "la\nsegunda", "la final": "la\nfinal"}[nombre]
            ax.text(x, y, corto, ha="center", va="center", fontsize=6.6, color=p.tinta,
                    fontweight="bold", zorder=4, linespacing=1.0)
        if texto_neurona and nombre in texto_neurona:
            debajo = y < y_medio or nombre == "la final"
            ax.text(x, y - RADIO - 1.2 if debajo else y + RADIO + 1.2, texto_neurona[nombre],
                    ha="center", va="top" if debajo else "bottom", fontsize=7.2,
                    color=p.tinta, linespacing=1.15, zorder=4)
    return pos


def dibujar(pesos, listones, paleta, ruta):
    L = Lienzo("La red del pasillo, entrenada",
               "Los dos interruptores (en cuadrados), una capa de en medio con dos neuronas, y\n"
               "la final. Cada círculo es un comité: suma lo que le llega por sus líneas, cada\n"
               "una por su peso, y lo compara con su listón. Éstos son los números que aprendió.",
               paleta, alto=ALTO)
    p = L.p
    y_arriba, y_abajo = L.y - 12.0, L.y - 42.0
    rot = {(d, o): ("+" if v > 0 else "−") + f"{abs(v):.2f}".replace(".", ",")
           for d, e in pesos.items() for o, v in e.items()}
    txt = {"la primera": "listón " + f"{listones['la primera']:.2f}".replace(".", ",")
                         + "\n" + PREGUNTAS[0],
           "la segunda": "listón " + f"{listones['la segunda']:.2f}".replace(".", ",")
                         + "\n" + PREGUNTAS[1],
           "la final": "listón " + f"{listones['la final']:.2f}".replace(".", ",")
                       + "\n«¿enciendo\nla luz?»"}
    dibujar_red(L, pesos, listones, y_arriba, y_abajo, rot, txt)
    L.pie("Línea continua: suma; de puntos: resta; cuanto más gruesa, más pesa. El número de cada línea\n"
          "es su peso; el de cada círculo, su listón. Un interruptor subido trae un 1; bajado, un 0.")
    L.guardar(ruta)


def selftest():
    fallos = []
    pesos, listones = leer(AQUI / SALIDA)

    # 1. TEST NULO — con un listón altísimo, una neurona no pasa en ninguna posición: el montaje
    #    «nunca». La cuenta que usa la figura para decir qué pregunta cada una no se inventa nada.
    alto = dict(listones, **{"la primera": 1e9})
    nulo = que_pregunta(pesos, alto, "la primera")
    print(f"[1] test nulo         con el listón por las nubes, la primera pasa en: {nulo}")
    if nulo != (0, 0, 0, 0):
        fallos.append("test nulo: con el listón por las nubes la neurona sigue pasando")

    # 2. SEÑAL — con los números leídos, la primera hace «al menos uno subido» y la segunda
    #    «los dos subidos»: lo que dicen los rótulos de la figura.
    a, b = que_pregunta(pesos, listones, "la primera"), que_pregunta(pesos, listones, "la segunda")
    print(f"[2] señal             la primera pasa en {a}; la segunda en {b}")
    if a != (0, 1, 1, 1) or b != (0, 0, 0, 1):
        fallos.append("señal: los rótulos de la figura no son lo que hacen sus números")

    # 3. INVARIANTE — las dos de en medio tienen líneas del mismo signo y parecidas (lo que el
    #    capítulo dice), y es el listón lo que las separa; y la figura se dibuja en las dos paletas.
    mismos = all(v > 0 for n in NOMBRES_MEDIO for v in pesos[n].values())
    for pal in (COLOR, GRIS):
        dibujar(pesos, listones, pal, "/dev/null")
    print(f"[3] invariante        las cuatro líneas de la primera capa suman: {'sí' if mismos else 'no'}; "
          f"listones {listones['la primera']} y {listones['la segunda']}")
    if not mismos or not listones["la segunda"] > listones["la primera"]:
        fallos.append("invariante: las dos de en medio no se distinguen por el listón como dice el texto")
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
    pesos, listones = leer(AQUI / SALIDA)
    dibujar(pesos, listones, GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
