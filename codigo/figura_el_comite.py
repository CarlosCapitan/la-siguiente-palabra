#!/usr/bin/env python3
"""
Capítulo 2 — el comité de sesenta y cuatro, dibujado.

La sección «Qué significa aprender, por primera vez» explica el mecanismo entero en cinco
párrafos seguidos sin un solo dibujo: sesenta y cuatro miembros, cada uno mirando un punto, cada
uno con su peso, los informes convertidos en puntos, un total y un listón. Es el pasaje que
sostiene el resto del libro y es el más abstracto que hay en él. Esta figura lo acompaña; no lo
sustituye.

Aquí no hay ningún dato medido, y es a propósito. Los pesos de verdad de un comité entrenado ya
salen en el capítulo 3 (figuras/una_neurona.png) con sus números reales; lo que falta aquí, antes
de eso, es el mecanismo. Así que el ejemplo de la cuenta está puesto a mano, es un comité completo
de cuatro miembros en vez de sesenta y cuatro, y el lector puede sumarlo con el dedo y comprobar
que el veredicto sale de ahí. La figura lo dice en su propio pie.

El dibujo del panel 1 sí es real: es la imagen número 134 del conjunto de dígitos manuscritos que
usan los programas del libro, con su tinta de 0 a 16 tal cual.

Uso:
    python figura_el_comite.py --selftest
    python figura_el_comite.py
"""

# ======================= CONSTANTES =======================

DESTINO = "../figuras/el_comite.png"

# Imagen número 134 del conjunto de dígitos manuscritos: un cuatro. Tinta de 0 (blanco) a 16
# (negro), tal como la entrega el conjunto. 64 puntos, ocho filas de ocho.
DIBUJO = [[ 0,  0,  0,  3, 16,  3,  0, 0],
          [ 0,  0,  0, 12, 16,  2,  0, 0],
          [ 0,  0,  8, 16, 16,  4,  0, 0],
          [ 0,  7, 16, 15, 16, 12, 11, 0],
          [ 0,  8, 16, 16, 16, 13,  3, 0],
          [ 0,  0,  0,  7, 14,  1,  0, 0],
          [ 0,  0,  0,  6, 16,  0,  0, 0],
          [ 0,  0,  0,  4, 14,  0,  0, 0]]
TINTA_MAXIMA = 16
FILA_SENALADA, COLUMNA_SENALADA = 3, 2      # el miembro del que habla el rótulo

# El comité del ejemplo: completo, de cuatro miembros, para que la cuenta se pueda hacer entera.
# «pintado» va de 0 (en blanco) a 2 (del todo); «caso» es cuánto se le hace caso a ese miembro.
# Puestos a mano: el pie de la figura lo dice.
EJEMPLO = [("punto 1", 2, +3), ("punto 2", 1, -4), ("punto 3", 0, -5), ("punto 4", 2, +2)]
LISTON = 5
CUANTO_PINTADO = {0: "en blanco", 1: "a medias", 2: "del todo"}

# ==========================================================

import argparse, sys
from infografia import Lienzo, GRIS, COLOR


def signo(n):
    return f"+{n}" if n > 0 else str(n).replace("-", "\u2212")


def dibujar(paleta, ruta):
    from matplotlib.patches import Rectangle, Circle
    p = paleta
    assert len(EJEMPLO) >= 2, "un comité de menos de dos miembros no enseña nada"
    L = Lienzo("El comité de sesenta y cuatro",
               "Un miembro por cada punto del dibujo. Ninguno ve el dibujo entero, ninguno sabe\n"
               "qué es un cuatro y ninguno manda: la decisión sale de sumar y comparar.",
               paleta, alto=6.44)

    # --- 1. un miembro por punto --------------------------------------------------------------
    x, y, w = L.panel(1, "Cada miembro mira un punto, y nada más", 40.0)
    lado = 3.0
    x0, ytop = x + 1.0, y - 0.6
    for f, fila in enumerate(DIBUJO):
        for c, v in enumerate(fila):
            gris = 1.0 - v / TINTA_MAXIMA
            L.ax.add_patch(Rectangle((x0 + c * lado, ytop - (f + 1) * lado), lado, lado,
                                     facecolor=(gris, gris, gris), edgecolor=p.marco,
                                     linewidth=0.5))
    cx = x0 + (COLUMNA_SENALADA + 0.5) * lado
    cy = ytop - (FILA_SENALADA + 0.5) * lado
    L.ax.add_patch(Circle((cx, cy), lado * 0.9, facecolor="none", edgecolor=p.tinta,
                          linewidth=1.2))
    xt = x0 + 8 * lado + 5.0
    L.ax.annotate("", xy=(cx + lado * 1.0, cy), xytext=(xt - 1.0, cy),
                  arrowprops=dict(arrowstyle="->", color=p.tinta, linewidth=0.9))
    L.texto(xt, cy + 5.0, "Este miembro mira este", tam=7.4)
    L.texto(xt, cy + 1.6, "punto. Lo único que puede", tam=7.4)
    L.texto(xt, cy - 1.8, "informar es cuánto de", tam=7.4)
    L.texto(xt, cy - 5.2, "pintado está.", tam=7.4)
    L.texto(x, ytop - 8 * lado - 3.6,
            "Ocho filas de ocho: 64 puntos, 64 miembros. Ninguno habla con los demás.",
            tam=7.2, color=p.suave)
    L.flecha()

    # --- 2. la cuenta entera, con un comité que cabe en la cabeza -----------------------------
    x, y, w = L.panel(2, "Cada informe se convierte en puntos", 73.0)
    L.texto(x, y + 0.4, "Con un comité de cuatro miembros, en vez de sesenta y cuatro, la cuenta",
            tam=7.2, color=p.suave)
    L.texto(x, y - 2.8, "entera cabe aquí y la puedes hacer tú:", tam=7.2, color=p.suave)

    c1, c2, c3 = x + 20.0, x + 48.0, x + w
    yc = y - 9.0
    for xx, rot, al in ((x, "el miembro", "left"), (c1, "lo que ve", "left"),
                        (c2, "cuánto se le hace caso", "left"), (c3, "puntos", "right")):
        L.texto(xx, yc, rot, tam=6.8, color=p.suave, ha=al)
    L.ax.plot([x, x + w], [yc - 2.6, yc - 2.6], color=p.marco, linewidth=0.8)

    total = 0
    for i, (rotulo, pintado, caso) in enumerate(EJEMPLO):
        yr = yc - 7.0 - i * 5.2
        puntos = pintado * caso
        total += puntos
        L.texto(x, yr, rotulo, tam=7.6)
        gris = 1.0 - pintado / max(CUANTO_PINTADO)
        L.ax.add_patch(Rectangle((c1, yr - 1.7), 3.4, 3.4, facecolor=(gris, gris, gris),
                                 edgecolor=p.marco, linewidth=0.6))
        L.texto(c1 + 4.8, yr, CUANTO_PINTADO[pintado], tam=7.6)
        L.texto(c2, yr, signo(caso), tam=7.6)
        L.texto(c3, yr, signo(puntos) if pintado else "0", tam=7.6, ha="right",
                negrita=True, color=p.tinta if pintado else p.suave)

    yt = yc - 7.0 - len(EJEMPLO) * 5.2 - 1.0
    L.ax.plot([x, x + w], [yt + 2.6, yt + 2.6], color=p.tinta, linewidth=0.8)
    L.texto(x, yt - 1.4, "el total", tam=8.0, negrita=True)
    L.texto(c3, yt - 1.4, signo(total), tam=8.0, negrita=True, ha="right")
    L.texto(x, yt - 7.2, "el listón", tam=8.0, negrita=True)
    L.texto(c3, yt - 7.2, str(LISTON), tam=8.0, negrita=True, ha="right")
    L.ficha(x, yt - 13.4, f"el total llega al listón: el comité dice que sí" if total >= LISTON
            else "el total no llega al listón: el comité dice que no", w, alto=5.6,
            relleno=p.acento, tinta="white", negrita=True, tam=8.0)
    L.texto(x, yt - 19.8, "Con sesenta y cuatro miembros es esto mismo, con sesenta y cuatro "
                          "sumandos.", tam=7.2, color=p.suave)
    L.texto(x, yt - 23.0, "El punto en blanco no aporta nada, tenga el peso que tenga.", tam=7.2,
            color=p.suave)

    L.pie("Los cuatro miembros y sus números están puestos a mano para que puedas sumarlos: no son\n"
          "medidas. Los pesos de verdad de un comité entrenado salen en el capítulo siguiente.")
    L.guardar(ruta)
    return ruta, total


def selftest():
    fallos = []

    # 1. TEST NULO — un comité de menos de dos miembros no enseña ninguna cuenta; si la figura
    #    se dibujara igual, saldría una tabla con su título y sin nada que sumar.
    global EJEMPLO
    guardado = EJEMPLO
    revienta = True
    for malo in ([], [("punto 1", 2, +3)]):
        EJEMPLO = malo
        try:
            dibujar(GRIS, "/dev/null")
            revienta = False
        except AssertionError:
            pass
    EJEMPLO = guardado
    print(f"[1] test nulo         con un comité de menos de dos, "
          f"{'revienta' if revienta else 'NO revienta'}")
    if not revienta:
        fallos.append("test nulo: la figura se dibuja con un comité de menos de dos miembros")

    # 2. SEÑAL IMPLANTADA — el total que la figura escribe es el que sale de sumar sus propias
    #    filas, y no otro. Es lo único comprobable que hay aquí, y es justo lo que se le pide al
    #    lector que haga.
    _, total = dibujar(GRIS, "/dev/null")
    a_mano = sum(pintado * caso for _, pintado, caso in EJEMPLO)
    cuenta = " ".join(f"{signo(pintado * caso)}" for _, pintado, caso in EJEMPLO)
    print(f"[2] señal implantada  {cuenta} = {a_mano}, y la figura escribe {signo(total)}")
    if total != a_mano:
        fallos.append(f"señal implantada: la figura escribe {total} y la suma da {a_mano}")

    # 3. INVARIANTE DEL DOMINIO — el ejemplo tiene que enseñar las dos reglas de las que habla el
    #    capítulo: que un punto en blanco no aporta nada aunque su peso sea grande, y que hay
    #    pesos a favor y en contra. Un ejemplo sin las dos cosas ilustraría media regla.
    blanco = [(r, c) for r, pin, c in EJEMPLO if pin == 0]
    hay_blanco_con_peso = any(c != 0 for _, c in blanco)
    hay_favor = any(c > 0 for _, _, c in EJEMPLO)
    hay_contra = any(c < 0 for _, _, c in EJEMPLO)
    print(f"[3] invariante        un punto en blanco con peso: "
          f"{'sí' if hay_blanco_con_peso else 'NO'}; pesos a favor: "
          f"{'sí' if hay_favor else 'NO'}; pesos en contra: {'sí' if hay_contra else 'NO'}")
    if not hay_blanco_con_peso:
        fallos.append("invariante: sin un punto en blanco con peso, no se ve que no aporte nada")
    if not (hay_favor and hay_contra):
        fallos.append("invariante: el ejemplo no tiene pesos a favor y en contra")

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
