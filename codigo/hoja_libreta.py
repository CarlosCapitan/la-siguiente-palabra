#!/usr/bin/env python3
"""
Las páginas de la libreta, dibujadas como hojas de libreta con sus palitos.

La máquina de contar del capítulo 1 es una libreta: una página por cada contexto, con el contexto
de título y, debajo, un palito por cada vez que vino detrás cada letra (o palabra). Este módulo
dibuja eso y nada más: la hoja (con su margen, sus agujeros y sus renglones) y los palitos
agrupados de cinco en cinco, cuatro rayas y la quinta cruzada. Aquí no hay ningún dato: cada
figura trae los suyos, contados del Quijote, y este módulo dibuja exactamente tantos palitos como
se le piden (el selftest lo cuenta).

Lo usan `figura_paginas_libreta.py` y `figura_pagina_el_caballero.py`.

Uso:
    from hoja_libreta import palitos, hoja
    python hoja_libreta.py --selftest
"""

# ======================= CONSTANTES =======================

PASO_PALITO = 1.0       # distancia entre dos palitos de un mismo grupo, en unidades del lienzo
HUECO_GRUPO = 1.7       # espacio entre un grupo de cinco y el siguiente
ALTO_PALITO = 2.8       # lo que mide de alto cada palito
GROSOR_PALITO = 0.9     # puntos; con menos los palitos se pierden en el papel, con más se pegan
POR_GRUPO = 5           # cuatro rayas y la quinta, cruzada

# Selftest
MAXIMO_PROBADO = 60     # se prueban todos los recuentos de 0 a este número

# ==========================================================

import sys


def ancho_palitos(n, paso=PASO_PALITO, hueco=HUECO_GRUPO):
    """Lo que mide de ancho, en unidades del lienzo, `n` palitos agrupados de cinco en cinco."""
    assert n >= 0, f"Se esperaba un número de palitos no negativo; se encontró {n}"
    if n == 0:
        return 0.0
    completos, resto = divmod(n, POR_GRUPO)
    grupos = [(POR_GRUPO - 2) * paso] * completos + ([(resto - 1) * paso] if resto else [])
    return sum(grupos) + hueco * (len(grupos) - 1)


def palitos(L, x, y, n, paso=PASO_PALITO, hueco=HUECO_GRUPO, alto=ALTO_PALITO):
    """Dibuja `n` palitos en el lienzo `L`, empezando en `x` y centrados en la altura `y`. Cada
    grupo completo son cuatro rayas verticales y una quinta que las cruza en diagonal; el resto,
    solo rayas. Devuelve la `x` donde acaba el último. Cada palito es una línea del lienzo: el
    selftest cuenta que salen exactamente `n`."""
    assert n >= 0, f"Se esperaba un número de palitos no negativo; se encontró {n}"
    tinta = L.p.tinta
    rayas = dict(color=tinta, linewidth=GROSOR_PALITO, solid_capstyle="butt")
    pintados, x_fin = 0, x
    while pintados < n:
        en_grupo = min(POR_GRUPO, n - pintados)
        verticales = min(en_grupo, POR_GRUPO - 1)
        for k in range(verticales):
            L.ax.plot([x + k * paso] * 2, [y - alto / 2, y + alto / 2], **rayas)
        if en_grupo == POR_GRUPO:
            L.ax.plot([x - 0.45 * paso, x + (POR_GRUPO - 2) * paso + 0.45 * paso],
                      [y - alto * 0.38, y + alto * 0.38], **rayas)
        x_fin = x + (verticales - 1) * paso
        pintados += en_grupo
        x += (verticales - 1) * paso + hueco
    return x_fin


def hoja(L, x, y_tope, ancho, titulo, renglones, alto_renglon, tam_titulo=12.5, alto_titulo=9.5):
    """Dibuja una hoja de libreta con su esquina superior izquierda en (x, y_tope): el título
    arriba (con una doble raya debajo), un margen a la izquierda con sus agujeros, y `renglones`
    renglones con su línea. Devuelve (ys, x_texto, x_fin, y_base): la altura del centro de cada
    renglón, dónde empieza el texto de cada renglón, dónde acaba, y el borde de abajo de la hoja."""
    from matplotlib.patches import Rectangle, Circle
    p = L.p
    alto = alto_titulo + renglones * alto_renglon + 1.5
    y_base = y_tope - alto
    L.ax.add_patch(Rectangle((x, y_base), ancho, alto, facecolor=p.papel, edgecolor=p.tinta,
                             linewidth=0.9))
    margen = x + 6.0
    L.ax.plot([margen] * 2, [y_base, y_tope], color=p.suave, linewidth=0.6)
    k = 0
    while k * 3 + 1.5 < renglones:                      # los agujeros de la libreta, dentro de ella
        L.ax.add_patch(Circle((x + 2.2, y_tope - alto_titulo - (k * 3 + 1.5) * alto_renglon), 0.8,
                              facecolor=p.papel, edgecolor=p.suave, linewidth=0.6))
        k += 1
    L.texto(margen + 2.2, y_tope - alto_titulo * 0.45, titulo, tam=tam_titulo, negrita=True)
    y_raya = y_tope - alto_titulo + 1.2
    for dy in (0.0, 0.7):
        L.ax.plot([x + 1.0, x + ancho - 1.0], [y_raya - dy] * 2, color=p.tinta, linewidth=0.7)
    ys = []
    for k in range(renglones):
        arriba = y_tope - alto_titulo - k * alto_renglon
        ys.append(arriba - alto_renglon / 2)
        L.ax.plot([x + 1.0, x + ancho - 1.0], [arriba - alto_renglon] * 2, color=p.marco,
                  linewidth=0.5)
    return ys, margen + 2.2, x + ancho - 2.0, y_base


# ---- selftest ------------------------------------------------------------------------

def selftest():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from infografia import Lienzo, GRIS
    fallos = []

    def lineas_nuevas(n, **k):
        L = Lienzo("prueba", "", GRIS, alto=2.0)
        antes = len(L.ax.lines)
        x_fin = palitos(L, 10.0, 20.0, n, **k)
        hechas = L.ax.lines[antes:]
        plt.close(L.fig)
        return hechas, x_fin

    # 1. TEST NULO — cero palitos no dibujan nada, y un recuento negativo revienta: una página
    #    sin nada apuntado se queda en blanco, y no hay palitos «de menos uno».
    hechas, _ = lineas_nuevas(0)
    try:
        palitos(None, 0, 0, -1); revienta = False
    except AssertionError:
        revienta = True
    print(f"[1] test nulo         0 palitos dibujan {len(hechas)} líneas; "
          f"-1 palitos {'revienta' if revienta else 'NO revienta'}")
    if hechas or not revienta:
        fallos.append("test nulo: cero palitos dibujan algo, o un recuento negativo se deja dibujar")

    # 2. SEÑAL IMPLANTADA — para cada recuento de 1 a MAXIMO_PROBADO, se dibujan exactamente
    #    tantas líneas como palitos, y de ellas cruzadas (diagonales) tantas como grupos de cinco.
    mal = []
    for n in range(1, MAXIMO_PROBADO + 1):
        hechas, _ = lineas_nuevas(n)
        cruzadas = sum(1 for l in hechas if len(set(l.get_xdata())) > 1)
        if len(hechas) != n or cruzadas != n // POR_GRUPO:
            mal.append((n, len(hechas), cruzadas))
    print(f"[2] señal implantada  de 1 a {MAXIMO_PROBADO} palitos, las líneas dibujadas son "
          f"tantas como palitos y las cruzadas, una por cada cinco: "
          f"{MAXIMO_PROBADO - len(mal)} de {MAXIMO_PROBADO} recuentos bien")
    if mal:
        fallos.append(f"señal implantada: recuentos mal dibujados (n, líneas, cruzadas): {mal[:5]}")

    # 3. INVARIANTE DEL DOMINIO — lo que se dibuja cabe en el ancho que se promete, y ese ancho
    #    no decrece al subir el recuento; y una hoja da tantos renglones como se piden, todos
    #    dentro de ella.
    sale = []
    anterior = -1.0
    for n in range(0, MAXIMO_PROBADO + 1):
        hechas, x_fin = lineas_nuevas(n)
        previsto = ancho_palitos(n)
        if hechas:
            xs = [v for l in hechas for v in l.get_xdata()]
            dibujado = max(xs) - min(xs)
            # la diagonal sobresale medio paso por cada lado de su grupo
            if dibujado > previsto + 0.9 * PASO_PALITO + 1e-9 or abs(x_fin - 10.0 - previsto) > 1e-9:
                sale.append(n)
        if previsto < anterior:
            sale.append(n)
        anterior = previsto
    L = Lienzo("prueba", "", GRIS, alto=3.0)
    ys, x_texto, x_fin, y_base = hoja(L, 4, 60, 40, "xt", 6, 7.0)
    plt.close(L.fig)
    dentro = len(ys) == 6 and all(y_base < y < 60 for y in ys) and x_texto < x_fin
    print(f"[3] invariante        el ancho de los palitos crece con el recuento y lo dibujado "
          f"cabe en él: {'sí' if not sale else 'NO'}; la hoja da 6 renglones dentro de ella: "
          f"{'sí' if dentro else 'NO'}")
    if sale:
        fallos.append(f"invariante: los palitos se salen de su ancho o este decrece, en {sale[:5]}")
    if not dentro:
        fallos.append("invariante: la hoja no da los renglones pedidos dentro de ella")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        sys.exit(selftest())
    print("Esto es un módulo de dibujo: no dibuja nada por su cuenta. Usa --selftest.")
