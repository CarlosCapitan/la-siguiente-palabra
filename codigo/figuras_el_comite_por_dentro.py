#!/usr/bin/env python3
"""
Capítulo 3 — el comité real, en cinco figuras de una idea cada una (L16, T12).

Sustituyen a `una_neurona.png`, que metía cuatro ideas en cuatro cuadros sin un número a la vista.
Cada figura cuenta una sola cosa, en el orden en que trabaja el comité:

  comite_la_tinta.png        lo que recibe: un cuatro, con la tinta de cada punto
  comite_los_pesos.png       lo que aprendió: el peso de cada punto
  comite_el_liston.png       la decisión: el total del cuatro y el del nueve contra el listón
  comite_la_diferencia.png   por qué: los casos se parecen a «cuatro menos nueve», no al cuatro
  de_un_comite_a_una_red.png y una red es muchos de éstos (mecanismo, sin datos; la maqueta de
                             Carlos de L8)

Los números NO se calculan aquí: se leen de `datos/salidas/el_comite_por_dentro.txt`. Si la
salida cambia, las figuras cambian. Se dibujan en gris: el libro se imprime en negro.

Uso:
    python figuras_el_comite_por_dentro.py --selftest
    python figuras_el_comite_por_dentro.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/el_comite_por_dentro.txt"
DESTINOS = {
    "tinta": "../figuras/comite_la_tinta.png",
    "casos": "../figuras/comite_los_pesos.png",
    "liston": "../figuras/comite_el_liston.png",
    "diferencia": "../figuras/comite_la_diferencia.png",
    "red": "../figuras/de_un_comite_a_una_red.png",
}
LADO = 8
COMITES_EN_EL_DIBUJO = 3        # los que caben; el pie dice que en el capítulo son ocho
COMITES_EN_EL_CAPITULO = 8

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from infografia import GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


# ---- leer la salida ----------------------------------------------------------------------

def _cuadricula(lineas, desde):
    """Las ocho filas de números que siguen a la cabecera de columnas «1 2 … 8»."""
    i = desde
    while not re.match(r"^\s+1\s+2\s+3\s+4\s+5\s+6\s+7\s+8\s*$", lineas[i]):
        i += 1
    filas = []
    for f in range(LADO):
        partes = lineas[i + 1 + f].split()
        assert partes and int(partes[0]) == f + 1, \
            f"se esperaba la fila {f + 1} de una cuadrícula en la línea {i + 2 + f}"
        filas.append([int(x) for x in partes[1:]])
    m = np.array(filas)
    assert m.shape == (LADO, LADO), f"se esperaba una cuadrícula de 8 por 8; salió {m.shape}"
    return m


def _numero(texto):
    return int(texto.replace(".", "").replace("−", "-"))


def leer(ruta):
    texto = Path(ruta).read_text(encoding="utf-8")
    L = texto.split("\n")

    def linea(prefijo):
        k = [i for i, l in enumerate(L) if l.startswith(prefijo)]
        assert k, f"se esperaba una línea que empiece por «{prefijo}» en {ruta}"
        return k[0]

    d = {}
    d["tinta"] = _cuadricula(L, linea("1. EL DIBUJO DE UN"))
    d["casos"] = _cuadricula(L, linea("2. LO QUE APRENDIÓ"))
    d["liston"] = _numero(re.search(r"el listón: (-?[\d.]+)\.", texto).group(1))
    d["si"], d["no"] = [int(x) for x in re.search(r"¿es un (\d) o un (\d)\?", texto).groups()]
    # Desde el 9 de octubre de 2026, los totales y los parecidos se leen de las tablas
    # editoriales de la salida, por su título (la salida dejó de imprimir renglones sangrados y
    # este lector se había quedado atrás).
    from formato import leer_tablas
    T = leer_tablas(texto)
    t_liston = f"El {d['si']} y el {d['no']}, contra el listón"
    assert t_liston in T, f"se esperaba la tabla «{t_liston}» en {ruta}"
    rot, filas, _ = T[t_liston]
    assert rot[:2] == ["dibujo", "total"], rot
    tot = {f[0]: _numero(f[1]) for f in filas}
    d["total_si"], d["total_no"] = tot[f"un {d['si']}"], tot[f"un {d['no']}"]
    d["media_si"] = _cuadricula(L, linea(f"  el {d['si']} medio"))
    d["media_no"] = _cuadricula(L, linea(f"  el {d['no']} medio"))
    d["diferencia"] = _cuadricula(L, linea("  la diferencia:"))
    par = [f for rot, fs, _ in T.values() if rot and rot[0] == "los pesos, comparados con"
           for f in fs]
    par = {f[0]: f[1] for f in par}
    assert f"el {d['si']} medio" in par and "la diferencia" in par, f"parecidos en {ruta}: {par}"
    d["parecido_media"] = par[f"el {d['si']} medio"]
    d["parecido_dif"] = par["la diferencia"]
    return d


# ---- dibujo ------------------------------------------------------------------------------

def _rejilla(L, x0, y_tope, lado, valores, tono, texto=None, tam=6.4, numerar=True):
    """Una cuadrícula de 8 por 8 con su tono por casilla (0 papel, 1 negro) y, si se pide, el
    número escrito dentro, en blanco sobre oscuro y en negro sobre claro."""
    from matplotlib.patches import Rectangle
    p = L.p
    for f in range(LADO):
        for c in range(LADO):
            t = float(tono[f, c])
            g = 1.0 - t
            L.ax.add_patch(Rectangle((x0 + c * lado, y_tope - (f + 1) * lado), lado, lado,
                                     facecolor=(g, g, g), edgecolor=p.marco, linewidth=0.4))
            if texto is not None:
                L.ax.text(x0 + (c + 0.5) * lado, y_tope - (f + 0.5) * lado,
                          str(texto[f, c]).replace("-", "\u2212"),
                          ha="center", va="center", fontsize=tam,
                          color="white" if t > 0.55 else p.tinta)
    if numerar:
        for k in range(LADO):
            L.ax.text(x0 + (k + 0.5) * lado, y_tope + 1.4, str(k + 1), ha="center", va="bottom",
                      fontsize=5.8, color=p.suave)
            L.ax.text(x0 - 1.2, y_tope - (k + 0.5) * lado, str(k + 1), ha="right", va="center",
                      fontsize=5.8, color=p.suave)


def figura_tinta(d, paleta, ruta):
    L = Lienzo("Lo que recibe el comité",
               f"Un {d['si']} de verdad, del conjunto de dígitos escritos a mano: 64 puntos, y en\n"
               "cada uno, cuánta tinta hay, de 0 (papel) a 16 (negro del todo).",
               paleta, alto=3.95)
    lado = 7.2
    _rejilla(L, 21, L.y - 3.0, lado, d["tinta"], d["tinta"] / 16.0, d["tinta"], tam=7.0)
    L.pie("Cada miembro del comité mira uno de estos puntos y solo sabe su número de tinta.")
    L.guardar(ruta)


def figura_casos(d, paleta, ruta):
    L = Lienzo("Lo que aprendió el comité",
               "Un número por punto, su peso: cuánto se le hace caso. Positivo, empuja hacia «es un "
               f"{d['si']}»;\nnegativo, hacia «es un {d['no']}». Nadie los escribió: salen de "
               "corregirse.",
               paleta, alto=3.95)
    lado = 7.2
    c = d["casos"]
    tope = np.abs(c).max()
    # Oscuro cuanto más empuja hacia el sí; los que empujan hacia el no, en blanco con el número
    # entre paréntesis sería otra convención: aquí se distinguen por el tono, de gris medio (no
    # empuja) a negro (sí) y a blanco (no).
    tono = 0.5 + 0.5 * c / tope
    _rejilla(L, 21, L.y - 3.0, lado, c, tono, c, tam=6.2)
    L.pie(f"Oscuro: empuja hacia el {d['si']}. Claro: hacia el {d['no']}. Gris medio: casi no "
          "cuenta.")
    L.guardar(ruta)


def figura_liston(d, paleta, ruta):
    from matplotlib.patches import Rectangle
    L = Lienzo("La decisión: el total contra el listón",
               "Para cada dibujo: tinta por peso en los 64 puntos, y se suma. Si el total pasa\n"
               f"del listón, el comité dice «es un {d['si']}».",
               paleta, alto=2.1)
    p = L.p
    minimo = min(d["total_no"], d["liston"])
    maximo = max(d["total_si"], d["liston"])
    x_izq, x_der = 14.0, 72.0

    def x(v):
        return x_izq + (v - minimo) / (maximo - minimo) * (x_der - x_izq)

    def cifra(v):
        return f"{v:,}".replace(",", ".").replace("-", "\u2212")

    y1, y2, alto = L.y - 5.0, L.y - 13.0, 5.0
    for yy, total, nombre in ((y1, d["total_si"], f"el {d['si']}"),
                              (y2, d["total_no"], f"el {d['no']}")):
        pasa = total > d["liston"]
        L.ax.add_patch(Rectangle((min(x(0), x(total)), yy - alto / 2), abs(x(total) - x(0)), alto,
                                 facecolor=p.acento if pasa else p.neutro, edgecolor="none"))
        L.texto(4, yy, nombre, negrita=True)
        L.texto(76, yy, f"{cifra(total)}: {'pasa' if pasa else 'no pasa'}", tam=7.6,
                negrita=pasa)
    L.ax.plot([x(0), x(0)], [y2 - 4, y1 + 4], color=p.tinta, linewidth=0.8)
    L.texto(x(0), y1 + 5.6, "0", ha="center", tam=6.6, color=p.suave)
    L.ax.plot([x(d["liston"])] * 2, [y2 - 5, y1 + 4], color=p.tinta, linewidth=1.0,
              linestyle=(0, (3, 2)))
    L.texto(x(d["liston"]), y2 - 7.4, "el listón: " + cifra(d["liston"]), ha="center", tam=7.0)
    L.guardar(ruta)


def figura_diferencia(d, paleta, ruta):
    L = Lienzo("Lo que dibujan los pesos no es un cuatro",
               f"El {d['si']} medio, el {d['no']} medio, lo que va del uno al otro, y los pesos "
               "que aprendió\nel comité. Los pesos se parecen a la tercera, no a la primera.",
               paleta, alto=2.2)
    lado = 2.3
    ancho = LADO * lado
    hueco = (92 - 4 * ancho) / 3
    y = L.y - 4.0
    paneles = [
        (f"el {d['si']} medio", d["media_si"] / 16.0),
        (f"el {d['no']} medio", d["media_no"] / 16.0),
        (f"{d['si']} menos {d['no']}", 0.5 + 0.5 * d["diferencia"] / np.abs(d["diferencia"]).max()),
        ("los pesos", 0.5 + 0.5 * d["casos"] / np.abs(d["casos"]).max()),
    ]
    for k, (titulo, tono) in enumerate(paneles):
        x0 = 4 + k * (ancho + hueco)
        L.texto(x0 + ancho / 2, y + 2.4, titulo, ha="center", tam=7.4, negrita=True)
        _rejilla(L, x0, y, lado, None, tono, None, numerar=False)
    L.pie(f"Parecido de los pesos con el {d['si']} medio: {d['parecido_media']}. "
          f"Con la diferencia: {d['parecido_dif']} (1 sería iguales).")
    L.guardar(ruta)


def figura_red(d, paleta, ruta):
    from matplotlib.patches import Circle, FancyBboxPatch
    L = Lienzo("De un comité a una red",
               "Varios comités miran el mismo dibujo, y otro comité mira lo que dicen ellos.",
               paleta, alto=3.2)
    p = L.p
    lado = 1.9
    y0 = L.y - 9.0
    _rejilla(L, 5, y0, lado, None, d["tinta"] / 16.0, None, numerar=False)
    L.texto(5 + 4 * lado, y0 - 8 * lado - 2.6, "el dibujo", ha="center", tam=7.0, color=p.suave)
    xc, xo, xr = 47.0, 74.0, 92.0
    ys = [y0 + 1.0, y0 - 7.6, y0 - 16.2]
    ym = ys[1]
    for k, yy in enumerate(ys[:COMITES_EN_EL_DIBUJO]):
        L.ax.annotate("", xy=(xc - 4.4, yy), xytext=(5 + 8 * lado + 1.0, y0 - 4 * lado),
                      arrowprops=dict(arrowstyle="->", color=p.tinta, linewidth=0.8))
        L.ax.add_patch(Circle((xc, yy), 3.9, facecolor=p.fondo if k else p.neutro,
                              edgecolor=p.tinta, linewidth=0.9))
        L.texto(xc, yy, f"comité {k + 1}", ha="center", tam=6.0)
        L.ax.annotate("", xy=(xo - 5.4, ym), xytext=(xc + 4.2, yy),
                      arrowprops=dict(arrowstyle="->", color=p.tinta, linewidth=0.8))
    L.texto(xc, ys[0] + 6.4, "primera capa", ha="center", tam=7.2, negrita=True)
    L.ax.add_patch(Circle((xo, ym), 5.0, facecolor=p.fondo, edgecolor=p.tinta, linewidth=0.9))
    L.texto(xo, ym + 1.3, "otro", ha="center", tam=6.6)
    L.texto(xo, ym - 1.6, "comité", ha="center", tam=6.6)
    L.texto(xo, ys[0] + 6.4, "la siguiente", ha="center", tam=7.2, negrita=True)
    L.ax.annotate("", xy=(xo + 12.0, ym), xytext=(xo + 5.2, ym),
                  arrowprops=dict(arrowstyle="->", color=p.tinta, linewidth=0.8))
    L.texto(xo + 13.0, ym, "sí o no", tam=6.8, negrita=True)
    # el recuadro: qué es un círculo
    yb = ys[-1] - 8.0
    L.ax.add_patch(FancyBboxPatch((4, yb - 13.0), 92, 13.0,
                                  boxstyle="round,pad=0,rounding_size=1.2",
                                  facecolor=p.fondo, edgecolor=p.marco, linewidth=0.9))
    L.texto(7, yb - 3.2, "Cada círculo es un comité entero, como el de las figuras anteriores:",
            tam=7.4, negrita=True)
    pasos = ["recibe números", "los multiplica\npor sus pesos", "los suma", "compara con\nsu listón"]
    for k, s in enumerate(pasos):
        xk = 8 + k * 22.5
        L.ficha(xk, yb - 8.8, s, 18.0, alto=6.4, tam=6.6)
        if k < 3:
            L.ax.annotate("", xy=(xk + 22.3, yb - 8.8), xytext=(xk + 18.3, yb - 8.8),
                          arrowprops=dict(arrowstyle="->", color=p.tinta, linewidth=0.8))
    L.pie(f"Tres comités, para que quepan. Las redes de este capítulo tienen "
          f"{COMITES_EN_EL_CAPITULO} en la primera capa.")
    L.guardar(ruta)


FIGURAS = {"tinta": figura_tinta, "casos": figura_casos, "liston": figura_liston,
           "diferencia": figura_diferencia, "red": figura_red}


def selftest():
    fallos = []
    d = leer(AQUI / SALIDA)
    # 1. TEST NULO: un texto sin las secciones no da figuras.
    import os, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("nada\n")
    try:
        leer(fh.name)
        fallos.append("test nulo: leyó números de un texto que no tiene ninguno")
    except (AssertionError, AttributeError, IndexError):
        pass
    finally:
        os.unlink(fh.name)
    print("[1] test nulo         un texto sin las secciones no da ninguna figura")
    # 2. SEÑAL: con la tinta y los casos leídos, la cuenta da el total que dice la salida.
    rehecho = int((d["tinta"] * d["casos"]).sum())
    print(f"[2] señal             tinta × peso, sumado: {rehecho}; la salida dice {d['total_si']}")
    if rehecho != d["total_si"]:
        fallos.append(f"señal: la cuenta rehecha da {rehecho} y la salida {d['total_si']}")
    # 3. INVARIANTE: las cinco caben en la página y se dibujan.
    for nombre, f in FIGURAS.items():
        f(d, GRIS, "/dev/null")
    print("[3] invariante        las cinco figuras se dibujan sin salirse de la página")
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
    d = leer(AQUI / SALIDA)
    for nombre, f in FIGURAS.items():
        f(d, GRIS, str(AQUI / DESTINOS[nombre]))
        print("escrita", DESTINOS[nombre])


if __name__ == "__main__":
    main()
