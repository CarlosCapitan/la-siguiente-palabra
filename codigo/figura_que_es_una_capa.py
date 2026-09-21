#!/usr/bin/env python3
"""
Capítulo 3 — qué es una capa.

El capítulo dice que una capa son varias unidades que reciben lo mismo, cada una con sus propios
pesos, y que lo que sale de todas ellas pasa a la capa siguiente. Dicho con palabras, el lector
tiene que imaginarse el reparto; dibujado, lo ve. Esta figura no sustituye a la explicación del
capítulo: la acompaña.

Esta figura no lleva ningún dato: no hay nada que medir en ella. Es la forma de la cosa, no un
resultado. El selftest lo comprueba: si alguien mete aquí un número que parezca medido, revienta.
El dibujo de la entrada es el mismo del capítulo 2. importado de su figura para que no haya dos
copias que puedan separarse.

Uso:
    python figura_que_es_una_capa.py --selftest
    python figura_que_es_una_capa.py
"""

# ======================= CONSTANTES =======================

DESTINO = "../figuras/que_es_una_capa.png"

COMITES = 3                 # tres es lo que cabe y se lee; la figura dice que es un ejemplo
TITULO_COMITE = "Comité"
LINEAS_COMITE = ["sus propios pesos", "suma y compara"]
TITULO_RESULTADO = "Resultado"

# ==========================================================

import argparse, sys
from infografia import Lienzo, GRIS, COLOR

# El mismo dibujo que entra en la unidad del capítulo 2, sin copiarlo: si allí cambia, aquí
# cambia. Que sea el mismo es parte de lo que la figura dice.
from figura_el_comite import DIBUJO, TINTA_MAXIMA


def dibujar(paleta, ruta):
    from matplotlib.patches import Rectangle, FancyBboxPatch
    # Una capa con un solo comité no es una capa: es la unidad del capítulo anterior otra vez.
    assert COMITES >= 2, f"una capa necesita al menos dos comités, y se han pedido {COMITES}"
    p = paleta
    rotulos = []                    # todo lo que se escribe, para que el selftest lo revise

    def escribir(x, y, s, **kw):
        rotulos.append(s)
        L.texto(x, y, s, **kw)

    def pastilla(x, y, s, ancho, **kw):
        rotulos.append(s)
        L.ficha(x, y, s, ancho, **kw)

    L = Lienzo("Qué es una capa",
               "Varios comités reciben exactamente lo mismo, y cada uno lo mira con sus\n"
               "propios pesos. Una capa es eso: varios comités en paralelo.",
               paleta, alto=5.44)

    # --- la entrada: el mismo dibujo del capítulo anterior -----------------------------------
    y = L.y - 1.0
    escribir(50, y, "El mismo dibujo para todos", tam=8.4, ha="center", negrita=True)
    lado = 2.6
    ancho_dibujo = 8 * lado
    x0, ytop = 50 - ancho_dibujo / 2, y - 3.6
    for f, fila in enumerate(DIBUJO):
        for c, v in enumerate(fila):
            gris = 1.0 - v / TINTA_MAXIMA
            L.ax.add_patch(Rectangle((x0 + c * lado, ytop - (f + 1) * lado), lado, lado,
                                     facecolor=(gris, gris, gris), edgecolor=p.marco,
                                     linewidth=0.4))
    y_dibujo = ytop - 8 * lado
    escribir(50, y_dibujo - 3.0, "los mismos 64 puntos", tam=7.4, ha="center", color=p.suave)

    # --- el reparto: una flecha a cada comité ------------------------------------------------
    ancho_caja, hueco = 26.0, 5.0
    x_izq = 50 - (COMITES * ancho_caja + (COMITES - 1) * hueco) / 2
    centros = [x_izq + i * (ancho_caja + hueco) + ancho_caja / 2 for i in range(COMITES)]
    y_capa_tope = y_dibujo - 12.0
    y_caja_tope = y_capa_tope - 6.0
    for xc in centros:
        L.ax.annotate("", xy=(xc, y_caja_tope - 0.6), xytext=(50, y_dibujo - 5.2),
                      arrowprops=dict(arrowstyle="->", color=p.tinta, linewidth=1.0,
                                      connectionstyle="arc3,rad=0"))

    # --- la capa: los comités, dentro de una caja que es la capa ------------------------------
    alto_caja = 17.0
    L.ax.add_patch(FancyBboxPatch((x_izq - 4.0, y_caja_tope - alto_caja - 4.0),
                                  COMITES * ancho_caja + (COMITES - 1) * hueco + 8.0,
                                  alto_caja + 10.0,
                                  boxstyle="round,pad=0,rounding_size=1.6",
                                  facecolor=p.fondo, edgecolor=p.tinta, linewidth=1.1))
    # El rótulo de la caja va a la izquierda: en el centro lo cruzaría la flecha del
    # comité de en medio.
    escribir(x_izq - 1.0, y_caja_tope + 3.4, "UNA CAPA", tam=9.2, negrita=True)
    for i, xc in enumerate(centros, start=1):
        L.ax.add_patch(FancyBboxPatch((xc - ancho_caja / 2, y_caja_tope - alto_caja),
                                      ancho_caja, alto_caja,
                                      boxstyle="round,pad=0,rounding_size=1.0",
                                      facecolor=p.papel, edgecolor=p.tinta, linewidth=0.8))
        escribir(xc, y_caja_tope - 4.2, f"{TITULO_COMITE} {i}", tam=8.6, ha="center", negrita=True)
        for k, linea in enumerate(LINEAS_COMITE):
            escribir(xc, y_caja_tope - 9.0 - k * 4.0, linea, tam=7.2, ha="center", color=p.suave)

    # --- lo que sale de cada comité -----------------------------------------------------------
    y_res = y_caja_tope - alto_caja - 13.0
    for i, xc in enumerate(centros, start=1):
        L.ax.annotate("", xy=(xc, y_res + 3.0), xytext=(xc, y_caja_tope - alto_caja - 4.6),
                      arrowprops=dict(arrowstyle="->", color=p.tinta, linewidth=1.0))
        pastilla(xc - ancho_caja / 2, y_res, f"{TITULO_RESULTADO} {i}", ancho_caja, alto=5.6,
                 relleno=p.acento, tinta="white", negrita=True, tam=8.0)
    escribir(50, y_res - 6.4, "y estos resultados son lo que recibe la capa siguiente", tam=7.6,
             ha="center")

    # --- lo que hay que entender ---------------------------------------------------------------
    escribir(4, y_res - 13.0, "Cada comité es una unidad completa: la del capítulo anterior, "
                              "entera.", tam=7.6)
    escribir(4, y_res - 16.6, "Tres comités es solo un ejemplo: una capa puede tener los que sea.",
             tam=7.6)

    L.pie("Esta figura no lleva ningún dato medido: dice cómo está montada una capa, no cuánto\n"
          "acierta. El dibujo de la entrada es el mismo del capítulo 2.")
    L.guardar(ruta)
    return ruta, rotulos


def selftest():
    fallos = []

    # 1. TEST NULO — una capa sin comités no es una capa. Si COMITES fuera 0 o negativo, el
    #    dibujo saldría vacío y con su título puesto, que es peor que no salir.
    global COMITES
    guardado = COMITES
    revienta = True
    for malo in (0, -1):
        COMITES = malo
        try:
            dibujar(GRIS, "/dev/null")
            revienta = False
        except (AssertionError, ValueError, ZeroDivisionError):
            pass
    COMITES = guardado
    print(f"[1] test nulo         con cero comités, "
          f"{'revienta' if revienta else 'NO revienta: dibuja una capa vacía'}")
    if not revienta:
        fallos.append("test nulo: con cero comités la figura se dibuja igual")

    # 2. SEÑAL IMPLANTADA — cada comité tiene su rótulo y su resultado, y son tantos como dice
    #    COMITES. Si el bucle del reparto y el de los resultados se separaran, esto lo canta.
    _, rotulos = dibujar(GRIS, "/dev/null")
    comites = [r for r in rotulos if r.startswith(TITULO_COMITE)]
    resultados = [r for r in rotulos if r.startswith(TITULO_RESULTADO)]
    print(f"[2] señal implantada  {len(comites)} comités y {len(resultados)} resultados, "
          f"para {COMITES} comités pedidos")
    if not (len(comites) == len(resultados) == COMITES):
        fallos.append(f"señal implantada: {len(comites)} comités y {len(resultados)} resultados "
                      f"para {COMITES} pedidos")

    # 3. INVARIANTE DEL DOMINIO — en esta figura no hay nada medido. Los únicos números que
    #    pueden aparecer son los que numeran los comités y sus resultados, y el 64 de los puntos
    #    del dibujo. Cualquier otro número sería un dato colado sin salida que lo respalde.
    permitidos = {str(i) for i in range(1, COMITES + 1)} | {"64"}
    import re
    colados = sorted({n for r in rotulos for n in re.findall(r"\d+", r)} - permitidos)
    print(f"[3] invariante        números en la figura: "
          f"{sorted({n for r in rotulos for n in re.findall(r'[0-9]+', r)})}; "
          f"colados sin salida que los respalde: {colados if colados else 'ninguno'}")
    if colados:
        fallos.append(f"invariante: números sin respaldo en una figura sin datos: {colados}")

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
    print("Escrito", dibujar(GRIS, DESTINO)[0])


if __name__ == "__main__":
    main()
