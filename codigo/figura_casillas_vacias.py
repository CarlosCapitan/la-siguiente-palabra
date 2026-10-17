#!/usr/bin/env python3
"""
Capítulo 1 — la libreta de parejas de palabras, dibujada a escala de verdad.

El capítulo dice que la libreta está casi vacía y da el número: una página con algo apuntado de
cada tantas (3.609 desde el 25 de septiembre de 2026, cuando los signos pasaron a contarse aparte,
T24; antes eran 7.337). Un número así se lee y se pasa de largo. Esta figura lo enseña: dibuja esas
páginas, una a una, y ennegrece la única que tiene algo apuntado. El número, el título y las filas
salen de la salida: no hay ninguno escrito aquí. El detalle de abajo está porque a tamaño de
impresión cada página mide un milímetro escaso de ancho, y el lector tiene que poder ver que son
páginas y no una trama gris.

Las páginas no son decorado: se dibujan de una en una a partir del número que hay en la salida. Si
el número cambia, cambia el dibujo. Todos los números salen de `datos/salidas/casillas_vacias.txt`.

Uso:
    python figura_casillas_vacias.py --selftest
    python figura_casillas_vacias.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/casillas_vacias.txt"
DESTINO = "../figuras/casillas_vacias.png"

COLUMNAS = 80               # páginas por fila; con 3.609 salen 80 x 45 + 9. Las páginas son más
                            # altas que anchas, así que hacen falta más columnas que con los
                            # cuadraditos de antes (67) para que la rejilla no se coma la hoja
PROPORCION_PAGINA = 1.5     # alto / ancho de una página, como una hoja de libro
HUECO_ANCHO = 0.72          # lo que mide de ancho cada página respecto de su paso en la fila
FILA_LLENA, COLUMNA_LLENA = 40, 14      # dónde va la única página con algo apuntado; da igual el
                                        # sitio, y la rejilla comprueba que caiga dentro
PAGINAS_DETALLE = 7         # cuántas páginas de ancho enseña el detalle ampliado
ANCHO_MINIMO_MM = 0.90      # una página más estrecha que esto en el papel se imprime como mancha

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
    corte = t.find("LA LIBRETA DE PAREJAS DE TROZOS")
    assert corte > 0, "no encuentro la sección de la libreta de parejas en la salida"
    s = t[corte:]
    return {
        "distintas":   buscar(t, r"trozos distintos \(palabras y \d+ signos\)\s+([\d.]+)",
                              "los trozos distintos"),
        "posibles":    buscar(s, r"páginas posibles \(trozos distintos al cuadrado\)\s+([\d.]+)",
                              "las páginas posibles"),
        "ocupadas":    buscar(s, r"páginas con algo apuntado \(parejas vistas\)\s+([\d.]+)",
                              "las páginas con algo apuntado"),
        "por_ciento":  buscar(s, r"páginas con algo apuntado, por cada cien posibles\s*([\d,]+) %",
                              "el porcentaje de páginas con algo apuntado"),
        "una_de_cada": buscar(s, r"una página con algo apuntado de cada\s+([\d.]+)",
                              "la proporción «una de cada»"),
    }


def pagina(L, x, y, ancho, alto, llena=False, escrita=False, grosor=0.3):
    """Una página de la libreta, con su esquina inferior izquierda en (x, y). `llena`: la pinta
    de negro (la de la rejilla grande). `escrita`: línea de título y rayitas de texto (el
    detalle). Sin ninguna de las dos, está en blanco."""
    from matplotlib.patches import Rectangle
    p = L.p
    L.ax.add_patch(Rectangle((x, y), ancho, alto,
                             facecolor=p.tinta if llena else p.papel,
                             edgecolor=p.tinta if (llena or escrita) else p.marco,
                             linewidth=grosor))
    if escrita:
        # el título, más grueso y más corto; debajo, las rayitas de lo apuntado
        L.ax.plot([x + ancho * 0.18, x + ancho * 0.60], [y + alto * 0.84] * 2,
                  color=p.tinta, linewidth=1.5, solid_capstyle="butt")
        for k in range(5):
            largo = 0.82 if k < 4 else 0.45
            L.ax.plot([x + ancho * 0.18, x + ancho * largo], [y + alto * (0.68 - 0.115 * k)] * 2,
                      color=p.suave, linewidth=0.6, solid_capstyle="butt")


def dibujar(d, paleta, ruta):
    from matplotlib.patches import Circle
    p = paleta
    total = numero(d["una_de_cada"])
    filas = -(-total // COLUMNAS)

    L = Lienzo(f"Una página de cada {d['una_de_cada']}",
               "Una página por cada pareja de palabras que el Quijote podría llegar a decir:\n"
               f"{d['distintas']} palabras y signos distintos dan {d['posibles']} páginas. Esto es\n"
               "lo que hay apuntado en ellas después de leer el libro entero.",
               paleta, alto=5.96)  # sin el hueco del pie (Orden 4; era 6,25)

    y = L.y - 1.0
    L.texto(4, y, f"Cada rectangulito es una página de la libreta. Aquí hay {d['una_de_cada']}: "
                  f"las que", tam=7.4, color=p.suave)
    L.texto(4, y - 3.4, "caben entre dos que tengan algo apuntado. Solo una lo tiene, y es la negra.",
            tam=7.4, color=p.suave)

    # --- las páginas, dibujadas de una en una ------------------------------------------
    x0, y_tope, ancho = 4.0, y - 7.6, 92.0
    paso = ancho / COLUMNAS
    w = paso * HUECO_ANCHO
    h = w * PROPORCION_PAGINA
    paso_y = h * 1.12
    dibujadas, marca = 0, None
    for f in range(filas):
        for c in range(COLUMNAS):
            if dibujadas >= total:
                break
            x = x0 + c * paso + (paso - w) / 2
            yy = y_tope - f * paso_y - h
            llena = (f == FILA_LLENA and c == COLUMNA_LLENA)
            pagina(L, x, yy, w, h, llena=llena)
            if llena:
                marca = (x + w / 2, yy + h / 2)
            dibujadas += 1
    assert marca, "la página llena cae fuera de la rejilla"
    assert dibujadas == total, f"dibujadas {dibujadas} páginas y hacían falta {total}"
    L.ax.add_patch(Circle(marca, 2.6, facecolor="none", edgecolor=p.tinta, linewidth=1.0))
    y_rejilla = y_tope - filas * paso_y

    # --- el detalle: las mismas páginas de cerca ---------------------------------------------
    # A tamaño de impresión cada página mide un milímetro de ancho. Sin esto, el lector ve una
    # trama gris y se cree que el dibujo grande no son miles de páginas.
    w_d = 9.0
    h_d = w_d * PROPORCION_PAGINA
    sep = (ancho - PAGINAS_DETALLE * w_d) / (PAGINAS_DETALLE - 1)
    y_d = y_rejilla - 5.0 - h_d
    centro = PAGINAS_DETALLE // 2
    for c in range(PAGINAS_DETALLE):
        pagina(L, x0 + c * (w_d + sep), y_d, w_d, h_d, escrita=(c == centro), grosor=0.7)
    x_centro = x0 + centro * (w_d + sep) + w_d / 2
    L.ax.annotate("", xy=(x_centro, y_d + h_d + 0.4), xytext=marca,
                  arrowprops=dict(arrowstyle="-", color=p.tinta, linewidth=0.7,
                                  linestyle=(0, (2, 2))))
    L.texto(x0, y_d - 3.0, "las mismas páginas, de cerca: solo una tiene algo apuntado", tam=7.0,
            color=p.suave)

    L.texto(x0, y_d - 8.4, f"La libreta entera tiene {d['posibles']} páginas y solo {d['ocupadas']}",
            tam=7.4)
    L.texto(x0, y_d - 11.8, f"tienen algo apuntado: {d['por_ciento']} %. El resto está en blanco,",
            tam=7.4)
    L.texto(x0, y_d - 15.2, "exactamente como aquí.", tam=7.4)

    L.guardar(ruta)
    return ruta


def selftest():
    fallos, d = [], leer(SALIDA)
    crudo = open(SALIDA, encoding="utf-8").read()

    # 1. TEST NULO — una salida sin la sección de la libreta de parejas tiene que reventar. Si no,
    #    «una de cada» podría pescarse de otro sitio y la rejilla dibujaría otro número de
    #    páginas con el mismo rótulo puesto.
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(crudo[:crudo.find("LA LIBRETA DE PAREJAS DE TROZOS")]); tmp.close()
    try:
        leer(tmp.name); revienta = False
    except AssertionError:
        revienta = True
    os.unlink(tmp.name)
    print(f"[1] test nulo         sin la sección de la libreta, "
          f"{'revienta' if revienta else 'NO revienta'}")
    if not revienta:
        fallos.append("test nulo: leer() no revienta sin la sección de la libreta")

    # 2. SEÑAL IMPLANTADA — cada número que se dibuja está, escrito igual, en la salida.
    fuera = [v for v in d.values() if v not in crudo]
    print(f"[2] señal implantada  {len(d) - len(fuera)} de {len(d)} números aparecen literales "
          f"en la salida")
    if fuera:
        fallos.append(f"señal implantada: estos números no están literales: {fuera}")

    # 3. INVARIANTE DEL DOMINIO — se dibujan tantas páginas como dice el rótulo (lo comprueba el
    #    propio dibujo con un assert), y cada página mide en el papel al menos un milímetro
    #    escaso de ancho; por debajo de eso la rejilla se imprime como una mancha gris y la
    #    figura deja de decir lo que dice que dice.
    from infografia import ANCHO_PAGINA
    total = numero(d["una_de_cada"])
    filas = -(-total // COLUMNAS)
    mm = (92.0 / COLUMNAS) * HUECO_ANCHO / 100 * ANCHO_PAGINA * 25.4
    dibujar(d, GRIS, "/dev/null")       # sus asserts cuentan las páginas
    mm_txt = f"{mm:.2f}".replace(".", ",")
    print(f"[3] invariante        {COLUMNAS} x {filas - 1} + {total - COLUMNAS * (filas - 1)} = "
          f"{total} páginas, de {mm_txt} mm de ancho en el papel")
    if mm < ANCHO_MINIMO_MM:
        fallos.append(f"invariante: cada página mide {mm_txt} mm de ancho en el papel y no se va a ver")

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
