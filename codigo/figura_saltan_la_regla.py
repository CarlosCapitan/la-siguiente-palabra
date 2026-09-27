#!/usr/bin/env python3
"""
Capítulo 3 — el paso «Primero» del reloj, dibujado (L24).

La regla del capítulo: un segmento que distinguiera los pares estaría encendido en los cinco
pares y apagado en los cinco impares. La tabla del paso «Primero» dice, para cada segmento, qué
dígitos se la saltan. Esta figura lo enseña con los dígitos delante, para dos segmentos: el de
arriba, al que se la saltan cinco, y el de abajo izquierda, al que se la salta uno. El lector no
tiene que recorrer un segmento con la vista a lo largo de diez dígitos: lo ve (criterio de Carlos
del 27 de septiembre: «no tiene que imaginar, tiene que verlo escrito y dibujado»).

Qué dígitos se saltan la regla NO se decide aquí: se lee del bloque 1 de
`datos/salidas/reloj_a_mano.txt`, y el selftest comprueba que coincide con la tabla de
`siete_segmentos.py`, que es la que imprime el libro. El dígito se dibuja con la misma forma que
la figura de los diez dígitos (`figura_diez_digitos.py`).

Uso:
    python figura_saltan_la_regla.py --selftest
    python figura_saltan_la_regla.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/reloj_a_mano.txt"
DESTINO = "../figuras/saltan_la_regla.png"
ALTO = 4.05                    # pulgadas
ALTO_PANEL = 29.5              # unidades del lienzo
ANCHO_DIGITO = 3.8             # unidades del lienzo: diez dígitos en una fila
MIRADOS = ["el de arriba", "el de abajo izquierda"]   # los dos que cuenta el texto
OTRO_ENCENDIDO = "#c2c2c2"     # los demás segmentos encendidos: se ve el dígito, pero en segundo plano
BORDE_MIRADO = 1.1             # grueso del borde del segmento mirado cuando está apagado
BORDE_RECUADRO = 1.3           # grueso del recuadro de los dígitos que se saltan la regla

PARES = [0, 2, 4, 6, 8]
IMPARES = [1, 3, 5, 7, 9]

# ==========================================================

import argparse
import sys
from pathlib import Path

from matplotlib.patches import FancyBboxPatch, Polygon

from figura_diez_digitos import APAGADO_BORDE, APAGADO_RELLENO, EJES, GROSOR, HUECO, hexagono
from infografia import COLOR, GRIS, Lienzo
from siete_segmentos import SEGMENTOS, tabla_de_segmentos

AQUI = Path(__file__).resolve().parent
NUMEROS = {1: "uno", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 7: "siete"}


def leer(ruta):
    """Del bloque 1 de la salida: para cada segmento, (pares apagados, impares encendidos)."""
    texto = Path(ruta).read_text(encoding="utf-8")
    assert "1. QUÉ DÍGITOS SE SALTAN LA REGLA" in texto, f"se esperaba el bloque 1 en {ruta}"
    uno = texto.split("1. QUÉ DÍGITOS SE SALTAN LA REGLA", 1)[1].split("EL DE ABAJO", 1)[0]
    import re
    saltan = {}
    for linea in uno.splitlines():
        if not linea.startswith("el ") or linea.startswith("el segmento"):
            continue
        campos = re.split(r"\s{2,}", linea.strip())
        assert len(campos) == 4, f"se esperaban cuatro columnas; hay {campos} en «{linea}»"
        nombre, pa, ie, n = campos
        lee = lambda t: [] if t == "ninguno" else [int(x) for x in t.split()]
        saltan[nombre] = (lee(pa), lee(ie))
        assert len(saltan[nombre][0]) + len(saltan[nombre][1]) == int(n), \
            f"«{nombre}»: la columna «cuántos» dice {n} y las listas suman otra cosa"
    assert sorted(saltan) == sorted(SEGMENTOS), f"se esperaban los siete segmentos; hay {sorted(saltan)}"
    return saltan


def de_la_tabla(X, nombre):
    """Lo mismo, contado sobre la tabla de segmentos del libro."""
    j = SEGMENTOS.index(nombre)
    return ([d for d in PARES if not X[d, j]], [d for d in IMPARES if X[d, j]])


def digito(ax, x, y, ancho, encendidos, mirado, p):
    """El dígito entero, con el segmento mirado destacado: negro si está encendido, hueco con
    borde negro si está apagado. Los demás encendidos, en gris claro; los apagados, casi blancos."""
    for nombre, ((x1, y1), (x2, y2)) in EJES.items():
        pts = hexagono(x + x1 * ancho, y + y1 * ancho, x + x2 * ancho, y + y2 * ancho,
                       GROSOR * ancho, HUECO * ancho)
        if nombre == mirado and nombre in encendidos:
            ax.add_patch(Polygon(pts, closed=True, facecolor=p.tinta, edgecolor="none", zorder=4))
        elif nombre == mirado:
            ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor=p.tinta,
                                 linewidth=BORDE_MIRADO, zorder=4))
        elif nombre in encendidos:
            ax.add_patch(Polygon(pts, closed=True, facecolor=OTRO_ENCENDIDO, edgecolor="none", zorder=3))
        else:
            ax.add_patch(Polygon(pts, closed=True, facecolor=APAGADO_RELLENO, edgecolor=APAGADO_BORDE,
                                 linewidth=0.5, zorder=3))


def dibujar(saltan, paleta, ruta):
    X = tabla_de_segmentos()
    L = Lienzo("¿Quién se salta la regla?",
               "La regla: encendido en los cinco pares y apagado en los cinco impares.",
               paleta, alto=ALTO)
    p, ax = L.p, L.ax
    recuadrados = {}
    # la clave, antes de los paneles
    yc = L.y - 1.2
    L.texto(4, yc, "El segmento que se mira:", tam=7.4, color=p.suave)
    for x, enc, rotulo in ((29.5, True, "encendido"), (47.5, False, "apagado")):
        pts = hexagono(x, yc, x + 5, yc, GROSOR * 4.2, 0)
        ax.add_patch(Polygon(pts, closed=True, facecolor=p.tinta if enc else "white",
                             edgecolor="none" if enc else p.tinta,
                             linewidth=0 if enc else BORDE_MIRADO))
        L.texto(x + 6.3, yc, rotulo, tam=7.4, color=p.suave)
    ax.add_patch(FancyBboxPatch((67, yc - 2.2), 4.4, 4.4, boxstyle="round,pad=0,rounding_size=0.7",
                                facecolor="none", edgecolor=p.tinta, linewidth=BORDE_RECUADRO))
    L.texto(73.2, yc, "se salta la regla", tam=7.4, color=p.suave)
    L.y -= 5.5
    # los diez dígitos en una fila: los cinco pares a la izquierda y los cinco impares a la derecha
    wd = ANCHO_DIGITO
    paso = 7.9
    x_pares = [12.0 + paso * i for i in range(5)]
    x_impares = [x_pares[-1] + 11.5 + paso * i for i in range(5)]
    for n, mirado in enumerate(MIRADOS, 1):
        pa, ie = saltan[mirado]
        cuantos = len(pa) + len(ie)
        cuenta = ("no se la salta ninguno" if cuantos == 0 else
                  f"se la {'salta' if cuantos == 1 else 'saltan'} {NUMEROS[cuantos]}")
        titulo = f"{mirado[0].upper()}{mirado[1:]}: {cuenta}"
        x0, ytop, _ = L.panel(n, titulo, ALTO_PANEL)
        recuadrados[mirado] = []
        yb = ytop - 13.5
        for xs_, rotulo, digs in ((x_pares, "pares: deben tenerlo encendido", PARES),
                                  (x_impares, "impares: deben tenerlo apagado", IMPARES)):
            L.texto((xs_[0] + xs_[-1]) / 2, ytop - 1.6, rotulo, ha="center", tam=7.4, negrita=True)
            for c, d in zip(xs_, digs):
                enc = {s for j, s in enumerate(SEGMENTOS) if X[d, j]}
                digito(ax, c - wd / 2, yb, wd, enc, mirado, p)
                L.texto(c, yb - 3.0, str(d), ha="center", tam=8.2, negrita=True)
                if d in pa or d in ie:
                    recuadrados[mirado].append(d)
                    ax.add_patch(FancyBboxPatch((c - 3.5, yb - 5.3), 7.0, 2 * wd + 7.4,
                                                boxstyle="round,pad=0,rounding_size=0.9",
                                                facecolor="none", edgecolor=p.tinta,
                                                linewidth=BORDE_RECUADRO, zorder=5))
        ax.plot([x_pares[-1] + 5.75] * 2, [yb - 5.5, yb + 2 * wd + 2.5], color=p.marco,
                linewidth=0.8)
    L.guardar(ruta)
    return recuadrados


def selftest():
    fallos = []
    X = tabla_de_segmentos()
    saltan = leer(AQUI / SALIDA)

    # 1. TEST NULO — un segmento fabricado que está encendido justo en los pares no se lo salta
    #    nadie: la figura no tiene que recuadrar ningún dígito.
    rec = dibujar(_fabricado(X), GRIS, "/dev/null")
    vacio = all(v == [] for v in rec.values())
    print(f"[1] test nulo         con un segmento que cumple la regla, dígitos recuadrados: "
          f"{'ninguno' if vacio else rec}")
    if not vacio:
        fallos.append("test nulo: recuadra dígitos de un segmento que cumple la regla")

    # 2. SEÑAL — lo que dice el capítulo: al de arriba se la saltan el 4 (par, apagado) y el 3, el
    #    5, el 7 y el 9 (impares, encendidos); al de abajo izquierda, solo el 4.
    rec = dibujar(saltan, GRIS, "/dev/null")
    ok = (sorted(rec["el de arriba"]) == [3, 4, 5, 7, 9] and rec["el de abajo izquierda"] == [4])
    print(f"[2] señal             recuadrados: el de arriba {sorted(rec['el de arriba'])}, "
          f"el de abajo izquierda {rec['el de abajo izquierda']}")
    if not ok:
        fallos.append("señal: los dígitos recuadrados no son los que dice el capítulo")

    # 3. INVARIANTE — lo que la salida dice de los siete segmentos es lo que sale de contar sobre
    #    la tabla del libro, y la figura se dibuja en color y en gris.
    malos = [s for s in SEGMENTOS if saltan[s] != de_la_tabla(X, s)]
    for pal in (COLOR, GRIS):
        dibujar(saltan, pal, "/dev/null")
    print(f"[3] invariante        segmentos en que la salida y la tabla no coinciden: {malos or 'ninguno'}")
    if malos:
        fallos.append(f"invariante: la salida y la tabla no coinciden en {malos}")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def _fabricado(X):
    """Los dos segmentos mirados, como si estuvieran encendidos justo en los pares."""
    import numpy as np
    es_par = np.array([d % 2 == 0 for d in range(10)])
    Xf = X.copy()
    for m in MIRADOS:
        Xf[:, SEGMENTOS.index(m)] = es_par.astype(float)
    return {s: ([d for d in PARES if not Xf[d, SEGMENTOS.index(s)]],
                [d for d in IMPARES if Xf[d, SEGMENTOS.index(s)]]) for s in SEGMENTOS}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    dibujar(leer(AQUI / SALIDA), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
