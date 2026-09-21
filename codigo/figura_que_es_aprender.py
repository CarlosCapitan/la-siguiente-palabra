#!/usr/bin/env python3
"""
Capítulo 2 — qué es aprender, aquí.

El capítulo enuncia la regla de Rosenblatt en una cita de tres líneas y sigue adelante. Pero esa
regla es el bucle que sostiene todo lo que viene después, y un bucle contado en prosa hay que
montarlo en la cabeza. Esta figura lo monta. No sustituye a la cita del capítulo: la acompaña.

Aquí no hay ningún dato: no hay nada que medir en un bucle. El selftest lo comprueba, igual que
en la figura de la capa: si alguien cuela aquí un número que parezca medido, revienta.

Uso:
    python figura_que_es_aprender.py --selftest
    python figura_que_es_aprender.py
"""

# ======================= CONSTANTES =======================

DESTINO = "../figuras/que_es_aprender.png"

PASOS = ["Se le enseña un dibujo, y se le dice qué era.",
         "El comité suma sus puntos y dice que sí o que no."]
PREGUNTA = "¿Ha acertado?"
RAMAS = [("si acierta", ["No se toca nada. Ni un peso, ni el listón."]),
         ("si se equivoca", ["A cada miembro que informó algo se le hace un poco más de",
                             "caso, o un poco menos, en la dirección que habría hecho",
                             "falta para acertar ese dibujo.",
                             "A los que tenían su punto en blanco no se les toca:",
                             "no han dicho nada."])]
VUELTA = "y otra vez, con el siguiente dibujo"

# ==========================================================

import argparse, re, sys
from infografia import Lienzo, GRIS, COLOR


def dibujar(paleta, ruta):
    from matplotlib.patches import FancyBboxPatch
    p = paleta
    # Un bucle de aprendizaje sin la rama del error no es un bucle de aprendizaje: es una
    # máquina que mira ejemplos y no cambia nunca.
    ramas = dict(RAMAS)
    assert ramas.get("si se equivoca"), "sin la rama del fallo no hay nada que aprender"
    assert "si acierta" in ramas, "sin la rama del acierto no se ve que al acertar no se toca nada"

    rotulos, cajas = [], []
    IZQ, ANCHO, TAG = 12.0, 84.0, 22.0

    L = Lienzo("Qué es aprender, aquí",
               "Nadie escribe una regla y nadie le dice a la máquina qué es un cuatro. Esto es\n"
               "todo lo que pasa, repetido unos cientos de veces.",
               paleta, alto=5.30)

    def caja(x, y, ancho, alto, lineas, tag=None, relleno=None, tinta=None, negrita=False,
             tam=8.0):
        L.ax.add_patch(FancyBboxPatch((x, y - alto), ancho, alto,
                                      boxstyle="round,pad=0,rounding_size=1.2",
                                      facecolor=relleno or p.papel, edgecolor=p.tinta,
                                      linewidth=0.9))
        cajas.append(lineas)
        x_texto = x + (TAG if tag else 0)
        if tag:
            rotulos.append(tag)
            L.texto(x + 3.0, y - alto / 2, tag, tam=8.0, negrita=True,
                    color=tinta or p.tinta)
            L.ax.plot([x_texto - 2.0, x_texto - 2.0], [y - alto + 2.0, y - 2.0],
                      color=tinta or p.marco, linewidth=0.7)
        for i, linea in enumerate(lineas):
            rotulos.append(linea)
            yy = y - alto / 2 + (len(lineas) - 1 - 2 * i) * 2.2
            if tag:
                L.texto(x_texto + 1.5, yy, linea, tam=tam, color=tinta or p.tinta)
            else:
                L.texto(x + ancho / 2, yy, linea, tam=tam, ha="center", negrita=negrita,
                        color=tinta or p.tinta)

    def flecha(x, y0, y1):
        L.ax.annotate("", xy=(x, y1), xytext=(x, y0),
                      arrowprops=dict(arrowstyle="->", color=p.tinta, linewidth=1.1))

    y = L.y - 1.0
    y_entrada = y - 4.5
    for paso in PASOS:
        caja(IZQ, y, ANCHO, 9.0, [paso])
        flecha(IZQ + ANCHO / 2, y - 9.0, y - 13.4)
        y -= 13.8

    caja(IZQ + ANCHO / 4, y, ANCHO / 2, 9.0, [PREGUNTA], relleno=p.fondo, negrita=True, tam=9.0)
    flecha(IZQ + ANCHO / 2, y - 9.0, y - 13.4)
    y -= 13.8

    for i, (tag, lineas) in enumerate(RAMAS):
        alto = 5.0 + 4.4 * len(lineas)
        caja(IZQ, y, ANCHO, alto, lineas, tag=tag, tam=7.4,
             relleno=p.acento if i else None, tinta="white" if i else None)
        y -= alto + 4.0

    # La vuelta: por la izquierda, hasta el primer paso. Es lo que convierte esto en un bucle.
    y_fondo = y - 1.0
    L.ax.plot([IZQ + ANCHO / 2, IZQ + ANCHO / 2], [y + 3.0, y_fondo], color=p.tinta,
              linewidth=1.1)
    L.ax.plot([4.0, IZQ + ANCHO / 2], [y_fondo, y_fondo], color=p.tinta, linewidth=1.1)
    L.ax.plot([4.0, 4.0], [y_fondo, y_entrada], color=p.tinta, linewidth=1.1)
    L.ax.annotate("", xy=(IZQ - 0.5, y_entrada), xytext=(4.0, y_entrada),
                  arrowprops=dict(arrowstyle="->", color=p.tinta, linewidth=1.1))
    rotulos.append(VUELTA)
    L.texto(IZQ + ANCHO / 2 - 2.0, y_fondo + 2.4, VUELTA, tam=7.4, ha="right", color=p.suave)

    L.pie("Esta figura no lleva ningún dato: dice qué pasa, no cuánto acierta. Lo único que cambia\n"
          "en toda la vuelta son los pesos de los miembros y el listón.")
    L.guardar(ruta)
    return ruta, rotulos, cajas


def selftest():
    fallos = []

    # 1. TEST NULO — un bucle sin la rama del fallo no aprende nada, y dibujado con su título
    #    puesto diría que sí. Tiene que reventar.
    global RAMAS
    guardado = list(RAMAS)
    revienta = True
    for malo in ([guardado[0]], [guardado[0], ("si se equivoca", [])]):
        RAMAS = malo
        try:
            dibujar(GRIS, "/dev/null")
            revienta = False
        except AssertionError:
            pass
    RAMAS = guardado
    print(f"[1] test nulo         sin la rama del fallo, "
          f"{'revienta' if revienta else 'NO revienta'}")
    if not revienta:
        fallos.append("test nulo: la figura se dibuja sin la rama del fallo")

    # 2. SEÑAL IMPLANTADA — se dibujan tantas cajas como pasos hay declarados, más la pregunta,
    #    más una por rama. Si un bucle se quedara a medias, aquí se ve.
    _, rotulos, cajas = dibujar(GRIS, "/dev/null")
    esperadas = len(PASOS) + 1 + len(RAMAS)
    print(f"[2] señal implantada  {len(cajas)} cajas dibujadas, {esperadas} declaradas "
          f"({len(PASOS)} pasos + la pregunta + {len(RAMAS)} ramas)")
    if len(cajas) != esperadas:
        fallos.append(f"señal implantada: {len(cajas)} cajas para {esperadas} declaradas")

    # 3. INVARIANTE DEL DOMINIO — aquí no hay nada medido, así que no puede haber ningún número.
    #    Cualquiera que aparezca sería un dato colado sin salida que lo respalde.
    colados = sorted({n for r in rotulos for n in re.findall(r"\d+", r)})
    print(f"[3] invariante        números en la figura: "
          f"{colados if colados else 'ninguno'}")
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
