#!/usr/bin/env python3
"""
Capítulo 15 — el tablero de la tabla que aprende a hacer trampa, en sus cuatro variantes.

Figura de medición. Cuatro tableros de 7 × 7, uno por variante de aprender_la_trampa.py, con la
salida (S), la meta (M), el letrero que pone «meta conseguida» (L) y, en la D, los muros que tapian
la meta. Encima de cada uno, el camino que sigue la primera tabla de esa variante después de
aprender, eligiendo siempre el número más alto: lo calcula el mismo programa (camino()), no se
dibuja a mano.

Pedido por Carlos el 11 de octubre de 2026: «el ejemplo hay que explicarlo más detalladamente,
paso a paso».

Uso (desde codigo/):
    python figura_tablero.py --selftest
    python figura_tablero.py
"""

# ======================= CONSTANTES =======================

DESTINO = "../figuras/tablero_trampa.png"
ALTO = 5.2                        # pulgadas
LADO_CASILLA = 4.6                # en unidades del lienzo (100 de ancho)
X_TABLEROS = (8.0, 54.0)          # borde izquierdo de las dos columnas de tableros
HUECO_FILAS = 12.0                # entre la fila de arriba y la de abajo (títulos de dos renglones)
SEMILLA_DIBUJADA = 0              # la primera tabla de cada variante

# ==========================================================

import argparse
import sys

from matplotlib.patches import Rectangle

from infografia import GRIS, Lienzo
import aprender_la_trampa as A

ORDEN = ["A", "B", "C", "D"]
# Los títulos, en dos renglones cortos: con el de aprender_la_trampa.py en uno solo, el de la D
# se salía por la derecha (visto en la primera versión de la figura).
TITULOS = {"A": ("A. El juez solo lee", "letrero cerca"),
           "B": ("B. El juez comprueba", "letrero cerca"),
           "C": ("C. El juez solo lee", "letrero lejos"),
           "D": ("D. El juez solo lee", "letrero lejos, meta tapiada")}


def caminos():
    """Para cada variante: (título, letrero, muros, casillas del camino aprendido)."""
    out = {}
    for clave in ORDEN:
        titulo, juez, letrero, muros = A.MONTAJES[clave]
        tabla, _, _ = A.aprender(SEMILLA_DIBUJADA, juez, letrero, muros)
        out[clave] = (titulo, letrero, muros, A.camino(tabla, letrero, muros))
    return out


def comprobar(clave, letrero, muros, cs):
    """Un camino que se puede dibujar: empieza en la salida, cada paso es a una casilla vecina,
    no pisa muros y acaba en la meta o en el letrero."""
    assert cs[0] == A.SALIDA, f"{clave}: el camino no empieza en la salida"
    for x, y in zip(cs, cs[1:]):
        assert abs(x[0] - y[0]) + abs(x[1] - y[1]) == 1, f"{clave}: salto de {x} a {y}"
        assert y not in muros, f"{clave}: el camino pisa un muro en {y}"
    assert A.donde(cs[-1], letrero), (
        f"{clave}: se esperaba un camino que acabe en la meta o el letrero; acaba en {cs[-1]}")


def dibujar(datos, paleta, ruta):
    L = Lienzo("El tablero, en sus cuatro variantes",
               "S, la salida. M, la meta. L, el letrero que pone «meta conseguida». La línea es\n"
               "el camino que sigue la primera tabla de cada variante cuando ya ha aprendido.",
               paleta, alto=ALTO)
    p, ax = L.p, L.ax
    lado = A.LADO * LADO_CASILLA
    y_arriba = L.y - 8.0
    for i, clave in enumerate(ORDEN):
        titulo, letrero, muros, cs = datos[clave]
        comprobar(clave, letrero, muros, cs)
        x0 = X_TABLEROS[i % 2]
        tope = y_arriba - (i // 2) * (lado + HUECO_FILAS)
        L.texto(x0, tope + 5.6, TITULOS[clave][0], tam=7.4, negrita=True)
        L.texto(x0, tope + 2.2, TITULOS[clave][1], tam=7.0, color=p.suave)

        def esquina(c):                       # fila 0 arriba, columna 0 a la izquierda
            return x0 + c[1] * LADO_CASILLA, tope - (c[0] + 1) * LADO_CASILLA

        def centro(c):
            x, y = esquina(c)
            return x + LADO_CASILLA / 2, y + LADO_CASILLA / 2

        for f in range(A.LADO):
            for c in range(A.LADO):
                x, y = esquina((f, c))
                ax.add_patch(Rectangle((x, y), LADO_CASILLA, LADO_CASILLA,
                                       facecolor=p.contra if (f, c) in muros else p.papel,
                                       edgecolor=p.marco, linewidth=0.5))
        xs, ys = zip(*[centro(c) for c in cs])
        ax.plot(xs, ys, color=p.acento, linewidth=2.0, solid_capstyle="round", zorder=3)
        ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=p.acento, lw=2.0,
                                    mutation_scale=9, shrinkA=0, shrinkB=4))
        for letra, c in [("S", A.SALIDA), ("M", A.META), ("L", letrero)]:
            x, y = centro(c)
            # La letra va encima del camino, sobre un fondo blanco: en la primera versión la
            # línea la tachaba.
            ax.add_patch(Rectangle((x - 1.5, y - 1.6), 3.0, 3.2, facecolor=p.papel,
                                   edgecolor="none", zorder=5))
            ax.text(x, y, letra, ha="center", va="center", fontsize=7.6, fontweight="bold",
                    color=p.tinta, zorder=6)
        L.texto(x0 + lado, tope - lado - 2.6, f"{len(cs) - 1} pasos", tam=6.8,
                color=p.suave, ha="right")
    abajo = y_arriba - 2 * lado - HUECO_FILAS - 4.0
    # HUECO_FILAS deja sitio a los dos renglones del título de la fila de abajo.
    assert abajo > 8.0, f"los tableros bajan hasta {abajo:.1f} y pisan el pie"
    L.pie("En gris, los muros de la variante D: a la meta no se puede llegar. Los caminos los\n"
          "calcula aprender_la_trampa.py; la figura no decide ninguno.")
    L.guardar(ruta)


def selftest():
    ok = True
    datos = caminos()
    # 1. NULO — una tabla sin aprender (todo a cero) no tiene camino: se queda en la salida, y la
    #    figura se niega a dibujarlo.
    cs0 = A.camino(A.tabla_nueva(), A.LETRERO_CERCA)
    try:
        comprobar("nulo", A.LETRERO_CERCA, A.SIN_MUROS, cs0)
        nulo = False
    except AssertionError:
        nulo = True
    print(f"[1] test nulo         tabla a cero: camino de {len(cs0) - 1} pasos; la figura lo "
          f"rechaza: {nulo}")
    ok &= nulo
    # 2. SEÑAL — los caminos dibujados tienen la longitud y el final de la tabla resumen: A y D al
    #    letrero (2 y 12 pasos), B y C a la meta (6 pasos).
    esperado = {"A": ("letrero", 2), "B": ("meta", 6), "C": ("meta", 6), "D": ("letrero", 12)}
    vistos = {k: (A.donde(v[3][-1], v[1]), len(v[3]) - 1) for k, v in datos.items()}
    senal = vistos == esperado
    print(f"[2] señal implantada  caminos dibujados: {vistos}; los esperados: {senal}")
    ok &= senal
    # 3. INVARIANTE — todos los caminos son dibujables (vecinos, sin muros) y la figura cabe.
    try:
        for k, (_, letrero, muros, cs) in datos.items():
            comprobar(k, letrero, muros, cs)
        dibujar(datos, GRIS, "/dev/null")
        inv = True
    except AssertionError as e:
        print("   ", e)
        inv = False
    print(f"[3] invariante        los cuatro caminos se pueden dibujar y la figura cabe: {inv}")
    ok &= inv
    print()
    print("SELFTEST: las tres pruebas pasan." if ok else "SELFTEST: FALLA.")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if not selftest():
        sys.exit(1)
    if args.selftest:
        sys.exit(0)
    dibujar(caminos(), GRIS, DESTINO)
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
