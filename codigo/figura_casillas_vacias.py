#!/usr/bin/env python3
"""
Capítulo 1 — la tabla de parejas de palabras, dibujada a escala de verdad.

El capítulo dice que la tabla está casi vacía y da el número: una casilla con algo dentro de cada
tantas (3.609 desde el 25 de septiembre de 2026, cuando los signos pasaron a contarse aparte, T24;
antes eran 7.337). Un número así se lee y se pasa de largo. Esta figura lo enseña: dibuja esas
casillas, una a una, y ennegrece la única que tiene algo dentro. El número, el título y la rejilla
salen de la salida: no hay ninguno escrito aquí. La lupa de abajo está porque a tamaño de
impresión cada casilla mide un milímetro escaso, y el lector tiene que poder ver que son casillas
y no una trama gris.

Los cuadraditos no son decorado: se dibujan de uno en uno a partir del número que hay en la
salida. Si el número cambia, cambia el dibujo. Todos los números salen de
`datos/salidas/casillas_vacias.txt`.

Uso:
    python figura_casillas_vacias.py --selftest
    python figura_casillas_vacias.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/casillas_vacias.txt"
DESTINO = "../figuras/casillas_vacias.png"

COLUMNAS = 67               # casillas por fila; con 3.609 salen 67 x 53 + 58, que ocupa casi el
                            # mismo cuadrado que ocupaban las 7.337 con 96 columnas; cada casilla
                            # sale de milímetro y medio en el papel
FILA_LLENA, COLUMNA_LLENA = 40, 14      # dónde va la única casilla con algo dentro; da igual el
                                        # sitio, y la rejilla comprueba que caiga dentro
LADO_LUPA = 5               # cuántas casillas de ancho enseña la lupa

# ==========================================================

import argparse, re, sys
from infografia import Lienzo, GRIS, COLOR


def numero(s):
    """«1.312.685.361» -> 1312685361; «0,0136» -> 0.0136. El libro escribe en español."""
    return float(s.replace(".", "").replace(",", ".")) if "," in s else int(s.replace(".", ""))


def buscar(texto, patron, que):
    m = re.search(patron, texto)
    assert m, f"no encuentro {que} en la salida"
    return m.group(1)


def leer(ruta):
    t = open(ruta, encoding="utf-8").read()
    corte = t.find("LA TABLA DE PAREJAS DE TROZOS")
    assert corte > 0, "no encuentro la sección de la tabla de parejas en la salida"
    s = t[corte:]
    return {
        "distintas":   buscar(t, r"trozos distintos \(palabras y \d+ signos\)\s+([\d.]+)",
                              "los trozos distintos"),
        "posibles":    buscar(s, r"casillas posibles \(trozos distintos al cuadrado\)\s+([\d.]+)",
                              "las casillas posibles"),
        "ocupadas":    buscar(s, r"casillas con algo dentro \(parejas vistas\)\s+([\d.]+)",
                              "las casillas ocupadas"),
        "por_ciento":  buscar(s, r"casillas con algo dentro, por cada cien posibles\s+([\d,]+) %",
                              "el porcentaje de casillas ocupadas"),
        "una_de_cada": buscar(s, r"una casilla con algo dentro de cada\s+([\d.]+)",
                              "la proporción «una de cada»"),
    }


def dibujar(d, paleta, ruta):
    from matplotlib.patches import Rectangle, Circle
    p = paleta
    total = numero(d["una_de_cada"])
    filas = -(-total // COLUMNAS)

    L = Lienzo(f"Una casilla de cada {d['una_de_cada']}",
               "Una casilla por cada pareja de palabras que el Quijote podría llegar a decir:\n"
               f"{d['distintas']} palabras y signos distintos dan {d['posibles']} casillas. Esto es\n"
               "lo que hay escrito en ellas después de leer el libro entero.",
               paleta, alto=6.50)

    y = L.y - 1.0
    L.texto(4, y, f"Cada cuadradito es una casilla. Aquí hay {d['una_de_cada']}: las que caben "
                  f"entre dos", tam=7.4, color=p.suave)
    L.texto(4, y - 3.4, "que tengan algo dentro. Solo una lo tiene, y es la negra.", tam=7.4,
            color=p.suave)

    # --- las casillas, dibujadas de una en una -----------------------------------------
    x0, y_tope, ancho = 4.0, y - 7.6, 92.0
    paso = ancho / COLUMNAS
    lado = paso * 0.84
    dibujadas, marca = 0, None
    for f in range(filas):
        for c in range(COLUMNAS):
            if dibujadas >= total:
                break
            x = x0 + c * paso + (paso - lado) / 2
            yy = y_tope - f * paso - lado
            llena = (f == FILA_LLENA and c == COLUMNA_LLENA)
            L.ax.add_patch(Rectangle((x, yy), lado, lado,
                                     facecolor=p.tinta if llena else p.papel,
                                     edgecolor=p.tinta if llena else p.marco, linewidth=0.3))
            if llena:
                marca = (x + lado / 2, yy + lado / 2)
            dibujadas += 1
    assert marca, "la casilla llena cae fuera de la rejilla"
    assert dibujadas == total, f"dibujadas {dibujadas} casillas y hacían falta {total}"
    L.ax.add_patch(Circle(marca, 2.6, facecolor="none", edgecolor=p.tinta, linewidth=1.0))
    y_rejilla = y_tope - filas * paso

    # --- la lupa: las mismas casillas de cerca -----------------------------------------------
    # A tamaño de impresión cada casilla mide un milímetro escaso. Sin esto, el lector ve una
    # trama gris y se cree que el cuadrado grande es un dibujo, no miles de casillas.
    lado_lupa = 4.4
    x_lupa, y_lupa = 4.0, y_rejilla - 6.0
    centro = LADO_LUPA // 2
    for f in range(LADO_LUPA):
        for c in range(LADO_LUPA):
            llena = (f == centro and c == centro)
            L.ax.add_patch(Rectangle((x_lupa + c * lado_lupa, y_lupa - (f + 1) * lado_lupa),
                                     lado_lupa, lado_lupa,
                                     facecolor=p.tinta if llena else p.papel,
                                     edgecolor=p.tinta if llena else p.marco, linewidth=0.6))
    ancho_lupa = LADO_LUPA * lado_lupa
    L.ax.annotate("", xy=(x_lupa + ancho_lupa / 2, y_lupa + 0.4), xytext=marca,
                  arrowprops=dict(arrowstyle="-", color=p.tinta, linewidth=0.7,
                                  linestyle=(0, (2, 2))))
    L.texto(x_lupa, y_lupa - ancho_lupa - 3.4, "las mismas casillas, de cerca", tam=7.0,
            color=p.suave)

    x_texto = x_lupa + ancho_lupa + 6.0
    L.texto(x_texto, y_lupa - 4.0, f"La tabla entera tiene {d['posibles']}", tam=7.4)
    L.texto(x_texto, y_lupa - 7.4, f"casillas y solo {d['ocupadas']} tienen algo", tam=7.4)
    L.texto(x_texto, y_lupa - 10.8, f"dentro: {d['por_ciento']} %. El resto está en", tam=7.4)
    L.texto(x_texto, y_lupa - 14.2, "blanco, exactamente como aquí.", tam=7.4)

    L.pie("Los números salen de datos/salidas/casillas_vacias.txt, medidos sobre el Quijote. Los\n"
          f"{d['una_de_cada']} cuadraditos se dibujan de uno en uno a partir de ese número.")
    L.guardar(ruta)
    return ruta


def selftest():
    fallos, d = [], leer(SALIDA)
    crudo = open(SALIDA, encoding="utf-8").read()

    # 1. TEST NULO — una salida sin la sección de la tabla de parejas tiene que reventar. Si no,
    #    «una de cada» podría pescarse de otro sitio y la rejilla dibujaría otro número de
    #    casillas con el mismo rótulo puesto.
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(crudo[:crudo.find("LA TABLA DE PAREJAS DE TROZOS")]); tmp.close()
    try:
        leer(tmp.name); revienta = False
    except AssertionError:
        revienta = True
    os.unlink(tmp.name)
    print(f"[1] test nulo         sin la sección de la tabla, "
          f"{'revienta' if revienta else 'NO revienta'}")
    if not revienta:
        fallos.append("test nulo: leer() no revienta sin la sección de la tabla")

    # 2. SEÑAL IMPLANTADA — cada número que se dibuja está, escrito igual, en la salida.
    fuera = [v for v in d.values() if v not in crudo]
    print(f"[2] señal implantada  {len(d) - len(fuera)} de {len(d)} números aparecen literales "
          f"en la salida")
    if fuera:
        fallos.append(f"señal implantada: estos números no están literales: {fuera}")

    # 3. INVARIANTE DEL DOMINIO — se dibujan tantos cuadraditos como dice el rótulo (lo comprueba
    #    el propio dibujo con un assert), y cada casilla mide en el papel al menos un milímetro
    #    escaso; por debajo de eso la rejilla se imprime como una mancha gris y la figura deja de
    #    decir lo que dice que dice.
    from infografia import ANCHO_PAGINA
    total = numero(d["una_de_cada"])
    filas = -(-total // COLUMNAS)
    mm = (92.0 / COLUMNAS) / 100 * ANCHO_PAGINA * 25.4
    dibujar(d, GRIS, "/dev/null")       # sus asserts cuentan los cuadraditos
    print(f"[3] invariante        {COLUMNAS} x {filas - 1} + {total - COLUMNAS * (filas - 1)} = "
          f"{total} cuadraditos, de {mm:.2f} mm en el papel")
    if mm < 0.90:
        fallos.append(f"invariante: cada casilla mide {mm:.2f} mm en el papel y no se va a ver")

    print()
    if fallos:
        for f in fallos: print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    print("Escrito", dibujar(leer(SALIDA), GRIS, DESTINO))


if __name__ == "__main__":
    main()
