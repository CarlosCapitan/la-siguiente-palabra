#!/usr/bin/env python3
"""
Capítulo 4 — la red del pasillo recorrida al revés, con la culpa escrita en cada línea (L24, A14).

El ejemplo del capítulo: la red a medio entrenar (paso 2.500), con los dos interruptores subidos.
La final dice 0,593 cuando tendría que decir 0. La figura sigue la culpa desde la final hasta las
cuatro líneas de la primera capa: en cada neurona, lo que le toca y lo que deja pasar su rampa; en
cada línea, la culpa de su peso.

Los números se leen de `datos/salidas/culpa_hacia_atras_datos.csv`, que escribe
`culpa_hacia_atras.py` junto con el bloque 4 de su salida; el selftest comprueba que la figura y el
bloque 4 dicen lo mismo.

Uso:
    python figura_culpa_hacia_atras.py --selftest
    python figura_culpa_hacia_atras.py
"""

# ======================= CONSTANTES =======================

DATOS = "../datos/salidas/culpa_hacia_atras_datos.csv"
SALIDA = "../datos/salidas/culpa_hacia_atras.txt"
DESTINO = "../figuras/culpa_hacia_atras.png"
ALTO = 4.6                     # pulgadas

# ==========================================================

import argparse
import csv
import sys
from pathlib import Path

from infografia import COLOR, GRIS, Lienzo
from figura_la_red_del_pasillo import dibujar_red, RADIO, X_FINAL, X_MEDIO, X_ENTRADA

AQUI = Path(__file__).resolve().parent
ENTRADAS = ["el de abajo", "el de arriba"]
MEDIO = ["la primera", "la segunda"]


def leer(ruta):
    d = {}
    with open(ruta, encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            if fila["que"].startswith(("caso_", "mano_")):
                d[(fila["que"], fila["a"], fila["b"])] = float(fila["valor"])
    assert d, f"no hay datos del caso en {ruta}"
    return d


def v(d, que, a="", b=""):
    return d[(que, str(a), str(b))]


def f(x, n=3, signo=True):
    s = f"{abs(x):.{n}f}".replace(".", ",")
    return (("+" if x >= 0 else "−") if signo else ("" if x >= 0 else "−")) + s


def pie_abajo(L, s):
    """El pie, apoyado en el borde de abajo: con tres renglones, el de Lienzo.pie se sale."""
    L.ax.text(4, 1.0, s, ha="left", va="bottom", fontsize=6.8, color=L.p.suave, linespacing=1.4)


def dibujar(d, paleta, ruta):
    L = Lienzo("La culpa, hacia atrás",
               "La red a medio entrenar (paso 2.500), con los dos interruptores subidos. La luz\n"
               f"tiene que estar apagada: la final debería decir 0, y dice {f(v(d, 'mano_dice_final'), 4, False)}.",
               paleta, alto=ALTO)
    p = L.p
    pesos = {"la primera": {e: v(d, "caso_peso_medio", i, 0) for i, e in enumerate(ENTRADAS)},
             "la segunda": {e: v(d, "caso_peso_medio", i, 1) for i, e in enumerate(ENTRADAS)},
             "la final": {m: v(d, "caso_peso_final", j) for j, m in enumerate(MEDIO)}}
    listones = {"la primera": v(d, "caso_liston_medio", 0), "la segunda": v(d, "caso_liston_medio", 1),
                "la final": v(d, "caso_liston_final")}
    rot = {}
    for j, m in enumerate(MEDIO):
        for i, e in enumerate(ENTRADAS):
            rot[(m, e)] = "culpa " + g(v(d, "mano_lineas_primera", i, j))
        rot[("la final", m)] = "culpa " + g(v(d, "mano_lineas_final", j))
    txt = {"la final": (f"falla por {g(v(d, 'mano_falla'))}\n"
                        f"el doble, por lo que deja\npasar su rampa ({g(v(d, 'mano_pasa_final'), False)}):\n"
                        f"culpa {g(v(d, 'mano_culpa_final'))}")}
    for j, m in enumerate(MEDIO):
        txt[m] = (f"le toca {g(v(d, 'mano_le_toca', j))}\n"
                  f"su rampa deja pasar {g(v(d, 'mano_pasa_medio', j), False)}\n"
                  f"culpa {g(v(d, 'mano_culpa_medio', j))}")
    y_arriba, y_abajo = L.y - 22.0, L.y - 55.0
    dibujar_red(L, pesos, listones, y_arriba, y_abajo, rot, txt, flechas_atras=True)
    for i, e in enumerate(ENTRADAS):
        y = y_arriba if e == "el de arriba" else y_abajo
        L.texto(X_ENTRADA, y + (RADIO + 2.2 if e == "el de arriba" else -RADIO - 2.2),
                "subido: trae 1", ha="center", tam=7.0, color=p.suave)
    # los cuatro tramos, numerados como en el texto
    for x, s in ((X_FINAL, "1. la final"), ((X_MEDIO + X_FINAL) / 2 + 2, "2. sus dos\nlíneas"),
                 (X_MEDIO, "3. las de\nen medio"), (X_ENTRADA + 9.0, "4. las líneas de la\nprimera capa")):
        L.texto(x, L.y - 3.0, s, ha="center", tam=7.6, negrita=True)
    pie_abajo(L, "La culpa va de derecha a izquierda, en los cuatro tramos del texto. El grosor de cada línea\n"
          "es su peso, como en la figura de la red. A cada una de en medio le toca la culpa de la final por el\n"
          "peso de su línea; la de cada línea es la culpa de su neurona por lo que trajo esa línea. Culpa\n"
          "positiva: si ese peso sube, el error sube; negativa: si sube, el error baja.")
    L.guardar(ruta)


def g(x, signo=True):
    """Como en la salida: cuatro cifras significativas."""
    s = f"{abs(x):.4g}".replace(".", ",")
    return (("+" if x >= 0 else "−") if signo else ("" if x >= 0 else "−")) + s


def selftest():
    fallos = []
    d = leer(AQUI / DATOS)
    texto = (AQUI / SALIDA).read_text(encoding="utf-8")

    # 1. TEST NULO — una línea que trae un 0 (interruptor bajado) no tiene culpa. En el caso los
    #    dos están subidos, así que ninguna de las cuatro culpas de la primera capa puede ser 0,
    #    y cada una tiene que ser exactamente la culpa de su neurona (por 1).
    dif = max(abs(v(d, "caso_culpa_pesos_medio", i, j) - v(d, "caso_culpa_medio", j))
              for i in range(2) for j in range(2))
    print(f"[1] test nulo         culpa de cada línea menos la de su neurona (trae un 1): {dif:.1e}")
    if dif > 1e-12:
        fallos.append("test nulo: con los interruptores subidos, la culpa de la línea no es la de su neurona")

    # 2. SEÑAL — lo que la figura escribe es lo que imprime el bloque 4 de la salida.
    buscados = [g(v(d, "mano_culpa_final"), False), g(v(d, "mano_culpa_medio", 0), False),
                g(v(d, "mano_culpa_medio", 1), False).replace("−", "-"), g(v(d, "mano_pasa_medio", 0), False),
                g(v(d, "mano_dice_final"), False), g(v(d, "mano_lineas_final", 1), False)]
    faltan = [b for b in buscados if b not in texto]
    print(f"[2] señal             números de la figura que no están en la salida: {faltan or 'ninguno'}")
    if faltan:
        fallos.append(f"señal: la figura escribe {faltan} y la salida no")

    # 3. INVARIANTE — la culpa de cada una de en medio es la de la final por el peso de su línea
    #    por lo que deja pasar su rampa; y la figura se dibuja en las dos paletas.
    peor = max(abs(v(d, "caso_culpa_final") * v(d, "caso_peso_final", j) * v(d, "caso_pasa_rampa_medio", j)
                   - v(d, "caso_culpa_medio", j)) for j in range(2))
    for pal in (COLOR, GRIS):
        dibujar(d, pal, "/dev/null")
    print(f"[3] invariante        la culpa repartida hacia atrás cuadra: diferencia {peor:.1e}")
    if peor > 1e-9:
        fallos.append("invariante: la culpa de en medio no es la de la final por el peso por la rampa")
    print()
    if fallos:
        for x in fallos:
            print("FALLA:", x)
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
    dibujar(leer(AQUI / DATOS), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
