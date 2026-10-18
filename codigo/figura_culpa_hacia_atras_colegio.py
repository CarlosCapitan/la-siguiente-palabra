#!/usr/bin/env python3
"""
Capítulo 4 — la red del colegio recorrida al revés, con la culpa escrita en cada línea (Orden 20).

El ejemplo del capítulo: la red a medio entrenar (paso 1.000), con la edad de 13 años. El colegio no
la admite: la final debería decir 0 y dice 0,6068. La figura sigue la culpa desde la final hasta las
dos líneas de la primera capa: en cada neurona, lo que le toca y lo que deja pasar su rampa; en cada
línea, la culpa de su peso.

Los números se leen de `datos/salidas/culpa_en_el_colegio_datos.csv`, que escribe
`culpa_en_el_colegio.py` junto con su salida; el selftest comprueba que la figura y la salida dicen
lo mismo.

Uso:
    python figura_culpa_hacia_atras_colegio.py --selftest
    python figura_culpa_hacia_atras_colegio.py
"""

# ======================= CONSTANTES =======================

DATOS = "../datos/salidas/culpa_en_el_colegio_datos.csv"   # los números, que escribe culpa_en_el_colegio.py
SALIDA = "../datos/salidas/culpa_en_el_colegio.txt"
DESTINO = "../figuras/culpa_hacia_atras_colegio.png"
ALTO = 4.6                     # pulgadas

# ==========================================================

import argparse
import sys
from pathlib import Path

from figura_la_red_del_colegio import RADIO, X_ENTRADA, X_FINAL, X_MEDIO, dibujar_red, leer_datos, red_de, v
from formato import leer_tablas
from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent
MEDIO = ["la primera", "la segunda"]


def g(x, signo=True):
    """Como en la salida: cuatro cifras significativas, sin exponente."""
    from decimal import Decimal
    t = format(Decimal(repr(round(abs(float(f"{x:.4g}")), 12))), "f")
    t = (t.rstrip("0").rstrip(".") if "." in t else t).replace(".", ",")
    return (("+" if x >= 0 else "−") if signo else ("" if x >= 0 else "−")) + t


def pie_abajo(L, s):
    L.ax.text(4, 1.0, s, ha="left", va="bottom", fontsize=6.8, color=L.p.suave, linespacing=1.4)


def dibujar(d, paleta, ruta):
    edad = int(v(d, "edad_del_caso"))
    paso = int(v(d, "paso_del_caso"))
    L = Lienzo("La culpa, hacia atrás",
               f"La red a medio entrenar (paso {paso:,}".replace(",", ".") + f"), con la edad de {edad} años. "
               "El colegio\n" f"no la admite: la final debería decir 0, y dice {g(v(d, 'mano_dice_final'), False)}.",
               paleta, alto=ALTO)
    p = L.p
    pesos, listones = red_de(d, "caso")
    rot = {}
    for j, m in enumerate(MEDIO):
        rot[(m, "la edad")] = "culpa " + g(v(d, "mano_lineas_primera", 0, j))
        rot[("la final", m)] = "culpa " + g(v(d, "mano_lineas_final", j))
    txt = {"la final": (f"falla por {g(v(d, 'mano_falla'))}\n"
                        f"el doble, por lo que deja\npasar su rampa ({g(v(d, 'mano_pasa_final'), False)}):\n"
                        f"culpa {g(v(d, 'mano_culpa_final'))}")}
    for j, m in enumerate(MEDIO):
        txt[m] = (f"le toca {g(v(d, 'mano_le_toca', j))}\n"
                  f"su rampa deja pasar {g(v(d, 'mano_pasa_medio', j), False)}\n"
                  f"culpa {g(v(d, 'mano_culpa_medio', j))}")
    y_arriba, y_abajo = L.y - 24.0, L.y - 52.0
    dibujar_red(L, pesos, listones, y_arriba, y_abajo, rot, txt, flechas_atras=True,
                texto_entrada=f"trae {edad}")
    for x, s in ((X_FINAL, "1. la final"), ((X_MEDIO + X_FINAL) / 2 + 2, "2. sus dos\nlíneas"),
                 (X_MEDIO, "3. las de\nen medio"), (X_ENTRADA + 9.0, "4. las líneas de la\nprimera capa")):
        L.texto(x, L.y - 3.0, s, ha="center", tam=7.6, negrita=True)
    pie_abajo(L, "La culpa va de derecha a izquierda, en los cuatro tramos del texto. El grosor de cada línea\n"
              "es su peso, como en la figura de la red. A cada una de en medio le toca la culpa de la final por el\n"
              "peso de su línea; la de cada línea es la culpa de su neurona por lo que trajo esa línea (la edad).\n"
              "Culpa positiva: si ese peso sube, el error sube; negativa: si sube, el error baja.")
    L.guardar(ruta)


def selftest():
    fallos = []
    d = leer_datos(AQUI / DATOS)
    texto = (AQUI / SALIDA).read_text(encoding="utf-8")
    T = leer_tablas(texto)
    edad = v(d, "edad_del_caso")

    # 1. TEST NULO — la línea de la edad trae la edad (13), no un 1: la culpa de cada línea de la
    #    primera capa es exactamente la culpa de su neurona por la edad, y no es la de su neurona.
    dif = max(abs(v(d, "caso_culpa_peso_medio", 0, j) - v(d, "caso_culpa_medio", j) * edad) for j in range(2))
    igual_que_neurona = max(abs(v(d, "caso_culpa_peso_medio", 0, j) - v(d, "caso_culpa_medio", j)) for j in range(2))
    print(f"[1] test nulo         culpa de cada línea menos la de su neurona por la edad ({int(edad)}): "
          f"{dif:.1e}; y menos la de su neurona sola: {igual_que_neurona:.2f} (no es cero)")
    if dif > 1e-12 or igual_que_neurona < 1e-3:
        fallos.append("test nulo: la línea de la edad no trae la edad")

    # 2. SEÑAL — lo que la figura escribe es lo que imprime la salida: tramos 1, 3 y 4 (las celdas
    #    de `leer_tablas` vienen sin la negrita).
    t1 = {f[0]: f[1] for f in T["Tramo 1: la culpa de la final"][1]}
    t3 = {f[0]: f[1:] for f in T["Tramo 3: cuánta le toca a cada una de las de en medio"][1]}
    t4 = T["Tramo 4: las dos líneas de la primera capa"][1]
    hay_final = [x for k, x in t1.items() if k.startswith("la culpa de la final")][0]
    hay_culpa = [x for k, x in t3.items() if k.startswith("su culpa")][0]
    h = lambda x: g(x, False).replace("−", "-")      # la salida escribe el menos corto
    ok_final = h(v(d, "mano_culpa_final")) == hay_final
    ok_toca = [h(v(d, "mano_le_toca", j)) for j in range(2)] == t3["le toca"]
    ok_culpa = [h(v(d, "mano_culpa_medio", j)) for j in range(2)] == hay_culpa
    ok_da = [h(v(d, "mano_lineas_primera", 0, j)) for j in range(2)] == [f[3] for f in t4]
    print(f"[2] señal             la figura dice lo que la salida: culpa de la final {'sí' if ok_final else 'NO'}, "
          f"lo que le toca {'sí' if ok_toca else 'NO'}, culpa de las de en medio {'sí' if ok_culpa else 'NO'}, "
          f"culpa de las líneas {'sí' if ok_da else 'NO'}")
    if not (ok_final and ok_toca and ok_culpa and ok_da):
        fallos.append("señal: la figura y la salida dicen cosas distintas")

    # 3. INVARIANTE — a cada una de en medio le toca la culpa de la final por el peso de su línea, y
    #    su culpa es lo que le toca por lo que deja pasar su rampa (con las cifras impresas, a menos del
    #    1 %); y se dibuja en las dos paletas.
    peor = 0.0
    for j in range(2):
        peso = v(d, "caso_peso_final", j)
        peor = max(peor, abs(v(d, "mano_le_toca", j) - v(d, "mano_culpa_final") * round(peso, 2))
                   / abs(v(d, "mano_le_toca", j)))
        peor = max(peor, abs(v(d, "mano_culpa_medio", j) - v(d, "mano_le_toca", j) * v(d, "mano_pasa_medio", j))
                   / abs(v(d, "mano_culpa_medio", j)))
    for pal in (COLOR, GRIS):
        dibujar(d, pal, "/dev/null")
    print(f"[3] invariante        la cadena de la figura se aleja de la regla un {100 * peor:.2f} % como mucho")
    if peor > 0.01:
        fallos.append("invariante: la cadena de la figura no cuadra")
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
    dibujar(leer_datos(AQUI / DATOS), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
