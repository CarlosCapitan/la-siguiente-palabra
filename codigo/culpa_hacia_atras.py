#!/usr/bin/env python3
"""
Capítulo 4 — la culpa, hecha a mano, sobre la red de la figura del pasillo (L24).

El capítulo contaba el reparto de la culpa hacia atrás con palabras y sin un solo número
(hallazgos A11 a A22 de `notas/auditoria/L24-IMAGINAR.md`). Este programa hace el caso
entero, con números, sobre la misma red que dibuja la figura del capítulo: la que resuelve
el o exclusivo con una neurona que pregunta «¿hay al menos uno subido?» y otra que pregunta
«¿están los dos subidos?» (`figura_apilar.red_representativa`, semilla y pasos en sus
constantes y en las de `que_inventan_las_capas.py`).

Imprime, por bloques:
  1. lo que dice la red entrenada en las cuatro posiciones, y lo que falla;
  2. los pesos y los listones que aprendió;
  3. la cuenta de cada neurona en cada posición (lo que le llega, su listón, lo que dice);
  4. un ejemplo a medio entrenar, hacia delante y hacia atrás: lo que falla la final, su
     culpa, cuánta le toca a cada una de las dos de en medio y por qué, la culpa de cada
     peso, y la comprobación a lo bruto de uno de ellos;
  5. el paso que aprende: la culpa de cada peso con las cuatro posiciones, el poquito, y el
     peso antes y después (comprobado contra el entrenamiento de verdad);
  6. lo que deja pasar la rampa corta y lo que deja pasar el codo (L24: antes «la que corta»);
  7. la culpa capa a capa en la red de cinco capas en medio (la de «como en los 80»);
  8. el terreno del error: un peso movido solo, y las redes que se atascan.

Las palabras, en el libro:
  - «lo que falla»: lo que dice la red menos lo que tendría que decir;
  - «el error» de un ejemplo: lo que falla, multiplicado por sí mismo; el de la red, la media
    de las cuatro posiciones (es el que minimiza `retropropagacion.Red.entrenar`);
  - «la culpa» de un peso: cuánto sube el error por cada unidad que sube ese peso; se mide
    moviéndolo un pelín, y el atajo la da sin moverlo;
  - «lo que deja pasar» la rampa: cuánto se mueve lo que dice una neurona por cada unidad que
    se mueve lo que le llega.

Uso:
    python culpa_hacia_atras.py
    python culpa_hacia_atras.py --selftest
"""

# ======================= CONSTANTES =======================

SEMILLA_RED = 318265456      # la que elige figura_apilar.red_representativa (el selftest lo comprueba)
PASO_DEL_CASO = 2_500        # a media bajada: el error ya ha empezado a caer y la red aún falla las cuatro
PELIN = 1e-6                 # lo que se sube lo que entra para medir lo que deja pasar la rampa
PELIN_BRUTO = 1e-5           # lo que se mueve un peso a lo bruto: el epsilon de retropropagacion.py
PUNTOS_RAMPA = [-6, -4, -2, -0.5, 0, 0.5, 2, 4, 6]   # «cuánto pasa del listón», para la tabla de las dos
PROFUNDIDAD_CADENA = 5       # capas en medio de la red en la que se sigue la culpa capa a capa
KILOMETRO_EN_MM = 1_000_000  # para poner a escala el factor de veinte capas
# El terreno se dibuja sobre la red ENTRENADA: moviendo un solo peso, el fondo es el sitio
# donde el entrenamiento lo dejó, y se ve entero. (En la red del paso 2.500 la curva de
# cualquier peso suelto es casi plana alrededor de donde está: no enseña nada.)
PESO_DEL_TERRENO = (1, 0, 0) # la línea de la primera a la final: (capa, de, a) en Red.W
TERRENO_DESDE, TERRENO_HASTA, TERRENO_PUNTOS = -6.0, 16.0, 221
INICIO_CUESTA_ABAJO = 2.0    # se saca el peso de su sitio y se pone aquí, en la ladera
PASOS_CUESTA_ABAJO = 200     # pasos moviendo solo ese peso
CADA_CUANTOS = 25            # cada cuántos pasos se imprime (y se dibuja) dónde va
PESOS_DEL_MAPA = ((1, 0, 0), (1, 1, 0))   # las dos líneas que llegan a la final
MAPA_DESDE, MAPA_HASTA, MAPA_PUNTOS = -16.0, 16.0, 129
PASOS_EXTRA_ATASCADAS = 100_000   # a las que se atascan, cuántos pasos más se les dan
TOL_BRUTA = 1e-6             # culpa por el atajo y a lo bruto: tienen que coincidir hasta aquí

POSICIONES = ["ninguno subido", "solo el de arriba", "solo el de abajo", "los dos subidos"]
ENTRADAS = ["el de abajo", "el de arriba"]          # el orden de las columnas de TABLA_XOR
NEURONAS = ["la primera", "la segunda"]
PREGUNTAS = ["¿hay al menos uno subido?", "¿están los dos subidos?"]

SALIDA_DATOS = "../datos/salidas/culpa_hacia_atras_datos.csv"   # para las figuras

# ==========================================================

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from formato import coma, comprobar_ancho, miles, ANCHO_CAJA_CITA
from retropropagacion import Red, TABLA_XOR, Y_XOR, sigmoide, SEMILLA, ANCHO
from que_inventan_las_capas import SEMILLA_BASE, TASA, PASOS, OCULTAS, ARRANQUES

AQUI = Path(__file__).resolve().parent
POQUITO = TASA               # lo que el entrenamiento resta de cada peso por cada unidad de culpa


def comprobar_entrada():
    assert [tuple(f) for f in TABLA_XOR] == [(0., 0.), (0., 1.), (1., 0.), (1., 1.)], \
        f"Se esperaba el orden ninguno / solo el de arriba / solo el de abajo / los dos; hay {TABLA_XOR}"
    assert list(Y_XOR) == [0., 1., 1., 0.], f"Se esperaba el o exclusivo; hay {list(Y_XOR)}"
    assert OCULTAS == 2, f"Se esperaban dos neuronas en medio; hay {OCULTAS}"
    assert 0 < PASO_DEL_CASO < PASOS, f"El paso del caso tiene que estar dentro del entrenamiento"


# ---------------------------------------------------------------- la red y su entrenamiento

def un_paso(red):
    """Un paso de aprendizaje, exactamente el de Red.entrenar."""
    gW, gb = red.gradiente_retro(TABLA_XOR, Y_XOR)
    for k in range(len(red.W)):
        red.W[k] -= TASA * gW[k]
        red.b[k] -= TASA * gb[k]


def red_en_el_paso(n, semilla=SEMILLA_RED):
    red = Red([2, OCULTAS, 1], semilla=semilla)
    for _ in range(n):
        un_paso(red)
    return red


def copia(red):
    otra = Red(list(red.tamanos), semilla=0)
    otra.W = [w.copy() for w in red.W]
    otra.b = [b.copy() for b in red.b]
    return otra


def rampa(x):
    return sigmoide(x)


def el_codo(x):
    """Lo que pasa del listón, tal cual; si no llega, cero. (Se llamaba «la que corta»; chocaba
    con «la rampa corta», y se cambió en la segunda vuelta de L24.)"""
    return np.maximum(0.0, x)


def deja_pasar(f, x, pelin=PELIN):
    """Cuánto se mueve lo que sale por cada unidad que se mueve lo que entra: se sube lo que
    entra un pelín y se mira cuánto sube lo que sale. A lo bruto, sin ninguna cuenta."""
    return float((f(x + pelin) - f(x)) / pelin)


# ---------------------------------------------------------------- el caso, a mano

def caso(red, p):
    """Un ejemplo, hacia delante y hacia atrás, con cada número con su nombre."""
    x = TABLA_XOR[p]
    W0, b0, W1, b1 = red.W[0], red.b[0], red.W[1][:, 0], red.b[1][0]
    c = {"x": x, "deberia": Y_XOR[p]}
    c["llega_medio"] = x @ W0                   # lo que le llega a cada una de en medio
    c["liston_medio"] = -b0                     # su listón: el total tiene que pasar de ahí
    c["pasa_medio"] = c["llega_medio"] - c["liston_medio"]
    c["dice_medio"] = rampa(c["pasa_medio"])
    c["llega_final"] = c["dice_medio"] @ W1
    c["liston_final"] = -b1
    c["pasa_final"] = c["llega_final"] - c["liston_final"]
    c["dice_final"] = float(rampa(c["pasa_final"]))
    c["falla"] = c["dice_final"] - c["deberia"]
    c["error"] = c["falla"] ** 2
    # hacia atrás
    c["error_por_salida"] = 2 * c["falla"]      # si lo que dice sube un pelín, el error sube esto por pelín
    c["pasa_rampa_final"] = deja_pasar(rampa, c["pasa_final"])
    c["culpa_final"] = c["error_por_salida"] * c["pasa_rampa_final"]
    c["culpa_lineas_final"] = c["culpa_final"] * c["dice_medio"]
    c["le_toca_medio"] = c["culpa_final"] * W1                    # por el peso de su línea
    c["pasa_rampa_medio"] = np.array([deja_pasar(rampa, z) for z in c["pasa_medio"]])
    c["culpa_medio"] = c["le_toca_medio"] * c["pasa_rampa_medio"]
    c["culpa_pesos_medio"] = np.outer(x, c["culpa_medio"])       # [entrada, neurona]
    return c


def error_de_un_ejemplo(red, p):
    return float((red.adelante(TABLA_XOR[p:p + 1])[-1].ravel()[0] - Y_XOR[p]) ** 2)


def culpa_a_lo_bruto(red, p, capa, i, j, pelin=PELIN_BRUTO):
    """Mover un peso un pelín hacia arriba y hacia abajo y ver cuánto cambia el error de ese
    ejemplo: lo mismo que hace `retropropagacion.Red.gradiente_fuerza_bruta`, peso a peso."""
    otra = copia(red)
    otra.W[capa][i, j] += pelin
    mas = error_de_un_ejemplo(otra, p)
    otra.W[capa][i, j] -= 2 * pelin
    menos = error_de_un_ejemplo(otra, p)
    return mas, menos, (mas - menos) / (2 * pelin)


def culpas_por_posicion(red):
    """La culpa de cada peso en cada una de las cuatro posiciones, por el atajo."""
    salida = []
    for p in range(4):
        gW, _ = red.gradiente_retro(TABLA_XOR[p:p + 1], Y_XOR[p:p + 1])
        salida.append(gW)
    return salida


# ---------------------------------------------------------------- impresión

def tabla(lineas):
    comprobar_ancho(lineas, ANCHO_CAJA_CITA)
    for l in lineas:
        print(l.rstrip())
    print()


def c3(x):
    """Tres decimales, con signo si es negativo."""
    return coma(x, 3)


def firmado(x, d=2):
    return ("+" if x >= 0 else "-") + coma(abs(x), d)


def bloque_1(red):
    act = red.adelante(TABLA_XOR)
    dice = act[-1].ravel()
    L = ["1. LO QUE DICE LA RED ENTRENADA EN LAS CUATRO POSICIONES",
         "(de 0 a 1: cerca de 0 es «no» y cerca de 1 es «sí»; lo que",
         "falla es lo que dice menos lo que tendría que decir)",
         "",
         f"{'posición':<19}{'la luz tiene':<14}{'la red':>8}{'lo que':>10}",
         f"{'':<19}{'que estar':<14}{'dice':>8}{'falla':>10}",
         f"{'-'*17:<19}{'-'*12:<14}{'-'*6:>8}{'-'*6:>10}"]
    for p in range(4):
        luz = "encendida" if Y_XOR[p] else "apagada"
        L.append(f"{POSICIONES[p]:<19}{luz:<14}{c3(dice[p]):>8}{firmado(dice[p]-Y_XOR[p], 3):>10}")
    tabla(L)
    return dice


def bloque_2(red):
    W0, b0, W1, b1 = red.W[0], red.b[0], red.W[1][:, 0], red.b[1][0]
    L = ["2. LOS PESOS Y LOS LISTONES QUE APRENDIÓ",
         "(peso de cada línea: positivo suma, negativo resta)",
         "",
         f"{'neurona':<13}{'la línea que viene de':<24}{'peso':>8}{'listón':>9}",
         f"{'-'*11:<13}{'-'*21:<24}{'-'*6:>8}{'-'*6:>9}"]
    for j in range(2):
        for i in range(2):
            L.append(f"{(NEURONAS[j] if i == 0 else ''):<13}{ENTRADAS[i]:<24}"
                     f"{firmado(W0[i, j]):>8}{(coma(-b0[j], 2) if i == 0 else ''):>9}")
    for j in range(2):
        L.append(f"{('la final' if j == 0 else ''):<13}{NEURONAS[j]:<24}"
                 f"{firmado(W1[j]):>8}{(coma(-b1, 2) if j == 0 else ''):>9}")
    tabla(L)


def bloque_3(red):
    L = ["3. LA CUENTA DE CADA NEURONA, POSICIÓN A POSICIÓN",
         "(le llega: la suma de lo que trae cada línea por su peso;",
         "dice: cerca de 1 si le llega más que su listón, cerca de 0",
         "si le llega menos)",
         ""]
    for j in range(2):
        L += [f"{NEURONAS[j].upper()}, «{PREGUNTAS[j]}»",
              f"listón: {coma(-red.b[0][j], 2)}",
              f"{'posición':<19}{'le llega':>10}{'¿pasa del listón?':>20}{'dice':>8}"]
        for p in range(4):
            c = caso(red, p)
            L.append(f"{POSICIONES[p]:<19}{coma(c['llega_medio'][j], 2):>10}"
                     f"{('sí' if c['pasa_medio'][j] > 0 else 'no'):>20}{coma(c['dice_medio'][j], 2):>8}")
        L.append("")
    L += [f"LA FINAL, que escucha a las dos",
          f"listón: {coma(-red.b[1][0], 2)}",
          f"{'posición':<19}{'le llega':>10}{'¿pasa del listón?':>20}{'dice':>8}"]
    for p in range(4):
        c = caso(red, p)
        L.append(f"{POSICIONES[p]:<19}{coma(c['llega_final'], 2):>10}"
                 f"{('sí' if c['pasa_final'] > 0 else 'no'):>20}{coma(c['dice_final'], 2):>8}")
    tabla(L)
    # la regla de la final, en palabras, comprobada en las cuatro esquinas del cuadrado nuevo
    W1, b1 = red.W[1][:, 0], red.b[1][0]
    esquinas = {(a, b): (a * W1[0] + b * W1[1] + b1) > 0 for a in (0, 1) for b in (0, 1)}
    regla = [k for k, v in esquinas.items() if v]
    nombre = {0: "no", 1: "sí"}
    L = ["LA FINAL, SI LAS DE EN MEDIO DIJERAN UN SÍ O UN NO SECOS",
         f"{'la primera':<13}{'la segunda':<13}{'le llega':>10}{'¿pasa?':>9}"]
    for (a, b), v in esquinas.items():
        L.append(f"{nombre[a]:<13}{nombre[b]:<13}{firmado(a*W1[0]+b*W1[1], 2):>10}{('sí' if v else 'no'):>9}")
    tabla(L)
    return regla


def r4(x):
    """Cuatro cifras significativas: el número tal como se imprime."""
    s = f"{x:.4g}"
    assert "e" not in s, f"número demasiado pequeño para imprimirlo sin exponente: {s}"
    return float(s)


def s4(x, signo=False):
    s = f"{abs(r4(x)):.4g}".replace(".", ",")
    if r4(x) < 0:
        return "-" + s
    return ("+" + s) if signo else s


def cadena_a_mano(red, c):
    """El caso hacia atrás, hecho como lo haría el lector con los números que se imprimen.

    Cada cuenta se hace con los números ya redondeados a cuatro cifras, y su resultado se
    redondea igual: así cualquier multiplicación del libro se puede repetir con el dedo y da lo
    mismo (hallazgo 19 de la verificación de L24). El selftest y los asserts de aquí comprueban
    que cada número de la cadena está a menos del 1 % del que da el programa sin redondear.
    «Lo que deja pasar la rampa» se calcula con la regla de la rampa, que es de multiplicar:
    lo que dice, por lo que le falta para 1."""
    W1 = [round(float(w), 2) for w in red.W[1][:, 0]]          # como en la tabla hacia delante
    m = {"pesos_final": W1}
    m["dice_final"] = r4(c["dice_final"])
    m["falla"] = r4(m["dice_final"] - c["deberia"])
    m["doble"] = r4(2 * m["falla"])
    m["falta_final"] = r4(1 - m["dice_final"])
    m["pasa_final"] = r4(m["dice_final"] * m["falta_final"])
    m["culpa_final"] = r4(m["doble"] * m["pasa_final"])
    m["dice_medio"] = [r4(v) for v in c["dice_medio"]]
    m["lineas_final"] = [r4(m["culpa_final"] * d) for d in m["dice_medio"]]
    m["le_toca"] = [r4(m["culpa_final"] * w) for w in W1]
    m["falta_medio"] = [r4(1 - d) for d in m["dice_medio"]]
    m["pasa_medio"] = [r4(d * f) for d, f in zip(m["dice_medio"], m["falta_medio"])]
    m["culpa_medio"] = [r4(a * b) for a, b in zip(m["le_toca"], m["pasa_medio"])]
    m["lineas_primera"] = [[r4(m["culpa_medio"][j] * c["x"][i]) for j in range(2)] for i in range(2)]
    exactos = [(m["culpa_final"], c["culpa_final"]), (m["pasa_final"], c["pasa_rampa_final"])]
    exactos += list(zip(m["lineas_final"], c["culpa_lineas_final"]))
    exactos += list(zip(m["le_toca"], c["le_toca_medio"]))
    exactos += list(zip(m["pasa_medio"], c["pasa_rampa_medio"]))
    exactos += list(zip(m["culpa_medio"], c["culpa_medio"]))
    for a, b in exactos:
        assert abs(a - b) <= 0.01 * abs(b) + 1e-9, f"la cadena a mano se aleja del programa: {a} y {b}"
    return m


def bloque_4(red, p):
    c = caso(red, p)
    m = cadena_a_mano(red, c)
    W1 = m["pesos_final"]
    L = [f"4. UN EJEMPLO A MEDIO ENTRENAR: EL PASO {miles(PASO_DEL_CASO)} DE {miles(PASOS)}",
         f"posición: {POSICIONES[p]}; la luz tiene que estar "
         f"{'encendida' if c['deberia'] else 'apagada'},",
         f"así que la final tendría que decir {int(c['deberia'])}",
         "",
         "HACIA DELANTE",
         f"{'':<12}{'le llega':>10}{'listón':>9}{'pasa del':>11}{'dice':>9}",
         f"{'':<12}{'':>10}{'':>9}{'listón por':>11}{'':>9}"]
    for j in range(2):
        L.append(f"{NEURONAS[j]:<12}{coma(c['llega_medio'][j], 2):>10}{coma(c['liston_medio'][j], 2):>9}"
                 f"{coma(c['pasa_medio'][j], 2):>11}{s4(m['dice_medio'][j]):>9}")
    L.append(f"{'la final':<12}{coma(c['llega_final'], 2):>10}{coma(c['liston_final'], 2):>9}"
             f"{coma(c['pasa_final'], 2):>11}{s4(m['dice_final']):>9}")
    L += ["",
          f"lo que le llega a la final: {s4(m['dice_medio'][0])} por {coma(W1[0], 2)},",
          f"más {s4(m['dice_medio'][1])} por {coma(W1[1], 2)}: {coma(c['llega_final'], 2)}",
          f"lo que falla: dice {s4(m['dice_final'])} y tendría que decir {int(c['deberia'])}: "
          f"{s4(m['falla'], True)}",
          f"el error: lo que falla por sí mismo, {s4(r4(m['falla']) ** 2)}",
          ""]
    tabla(L)

    s = c["dice_final"]
    e_mas = (s + PELIN - c["deberia"]) ** 2
    L = ["HACIA ATRÁS (los números, redondeados a cuatro cifras, y",
         "cada cuenta hecha con los números redondeados)",
         "",
         "TRAMO 1: LA CULPA DE LA FINAL",
         "   cuánto se mueve el error si se mueve lo que dice:",
         f"   si dijera una millonésima más, el error pasaría",
         f"   de {coma(c['error'], 9)} a {coma(e_mas, 9)}: sube {coma((e_mas - c['error']) / PELIN, 3)}",
         f"   millonésimas; el doble de lo que falla, {s4(m['falla'])}",
         "",
         "   lo que deja pasar su rampa: lo que dice, por lo que",
         f"   le falta para 1: {s4(m['dice_final'])} por {s4(m['falta_final'])} da {s4(m['pasa_final'])}",
         "",
         f"   el doble de lo que falla                 {s4(m['doble']):>7}",
         f"   por lo que deja pasar su rampa           {s4(m['pasa_final']):>7}",
         f"   da la culpa de la final                  {s4(m['culpa_final']):>7}",
         "",
         "TRAMO 2: LAS DOS LÍNEAS QUE LLEGAN A LA FINAL",
         "   (la culpa de la final, por lo que trajo esa línea)",
         f"   la de la primera: {s4(m['culpa_final'])} por {s4(m['dice_medio'][0])}"
         f" da {s4(m['lineas_final'][0])}",
         f"   la de la segunda: {s4(m['culpa_final'])} por {s4(m['dice_medio'][1])}"
         f" da {s4(m['lineas_final'][1])}",
         "",
         "TRAMO 3: CUÁNTA LE TOCA A CADA UNA DE LAS DE EN MEDIO",
         f"{'':<31}{'la primera':>12}{'la segunda':>12}",
         f"{'   culpa de la final':<31}{s4(m['culpa_final']):>12}{s4(m['culpa_final']):>12}",
         f"{'   por el peso de su línea':<31}{firmado(W1[0], 2):>12}{firmado(W1[1], 2):>12}",
         f"{'   le toca':<31}{s4(m['le_toca'][0]):>12}{s4(m['le_toca'][1]):>12}",
         "",
         f"{'   lo que dice':<31}{s4(m['dice_medio'][0]):>12}{s4(m['dice_medio'][1]):>12}",
         f"{'   lo que le falta para 1':<31}{s4(m['falta_medio'][0]):>12}{s4(m['falta_medio'][1]):>12}",
         f"{'   lo que deja pasar su rampa':<31}{s4(m['pasa_medio'][0]):>12}{s4(m['pasa_medio'][1]):>12}",
         "",
         f"{'   su culpa: lo que le toca,':<31}{s4(m['culpa_medio'][0]):>12}{s4(m['culpa_medio'][1]):>12}",
         "   por lo que deja pasar su rampa",
         "",
         "TRAMO 4: LAS CUATRO LÍNEAS DE LA PRIMERA CAPA",
         "   (la culpa de su neurona, por lo que trajo la línea:",
         "   1 si el interruptor está subido, 0 si está bajado)"]
    for j in range(2):
        for i in range(2):
            L.append(f"   {NEURONAS[j]}, del {ENTRADAS[i][3:]:<10}{s4(m['culpa_medio'][j]):>9} por "
                     f"{int(c['x'][i])} da {s4(m['lineas_primera'][i][j])}")
    tabla(L)
    c["mano"] = m
    return c


def bloque_4_bruto(red, p, c):
    """La comprobación a lo bruto de los pesos: moverlos un pelín y ver cuánto cambia el error."""
    gW, _ = red.gradiente_retro(TABLA_XOR[p:p + 1], Y_XOR[p:p + 1])
    L = ["LA MISMA CULPA, A LO BRUTO",
         "   se sube el peso una cienmilésima y se mira el error; se",
         "   baja una cienmilésima y se mira otra vez; la culpa es lo",
         "   que va de uno a otro, entre lo que se ha movido",
         ""]
    filas = []
    for (capa, i, j, nombre) in ((0, 1, 1, "la segunda, del de arriba"),
                                 (1, 1, 0, "la final, de la segunda")):
        mas, menos, bruta = culpa_a_lo_bruto(red, p, capa, i, j)
        atajo = float(gW[capa][i, j])
        filas.append((nombre, atajo, bruta, mas, menos))
        # Diez decimales y no más: de la undécima cifra en adelante, lo que sale a lo bruto
        # cambia de un ordenador a otro (redondeo), y el libro no imprime nada que dependa
        # de la máquina (regla 10).
        assert abs(atajo - bruta) < 1e-10, f"atajo y fuerza bruta difieren en {abs(atajo - bruta)}"
        L += [f"   {nombre}",
              f"      por el atajo    {coma(atajo, 10)}",
              f"      a lo bruto      {coma(bruta, 10)}"]
    nombre, atajo, bruta, mas, menos = filas[0]
    L += ["",
          "   las dos, iguales en sus diez primeros decimales; lo que",
          "   va de una a otra no llega a una diezmilmillonésima",
          "",
          f"   ({nombre}: con el peso una cienmilésima",
          f"   más arriba, el error es {coma(mas, 10)};",
          f"   una cienmilésima más abajo, {coma(menos, 10)})",
          "",
          f"   el redondeo, en una calculadora de seis decimales: un",
          f"   tercio es {coma(round(1/3, 6), 6)}, y por tres da {coma(3 * round(1/3, 6), 6)}"]
    tabla(L)
    return filas


def bloque_5(red):
    """El paso de aprendizaje: con las cuatro posiciones, lo que hace de verdad el programa."""
    por_pos = culpas_por_posicion(red)
    gW, gb = red.gradiente_retro(TABLA_XOR, Y_XOR)
    siguiente = copia(red)
    un_paso(siguiente)
    L = ["5. EL PASO: CADA PESO SE MUEVE CONTRA SU CULPA",
         f"(la culpa de cada peso es la media de su culpa en las",
         f"cuatro posiciones; se mueve {coma(POQUITO, 1)} veces esa culpa, en",
         "contra: si la culpa es negativa, sube; si es positiva, baja)",
         "",
         "LA CULPA DE LA SEGUNDA, DEL DE ARRIBA, POSICIÓN A POSICIÓN",
         f"{'posición':<19}{'culpa':>10}"]
    for p in range(4):
        L.append(f"{POSICIONES[p]:<19}{coma(por_pos[p][0][1, 1], 4):>10}")
    media = float(np.mean([por_pos[p][0][1, 1] for p in range(4)]))
    L += [f"{'-'*17:<19}{'-'*7:>10}", f"{'la media':<19}{coma(media, 4):>10}", ""]
    assert abs(media - gW[0][1, 1]) < 1e-12, "la media de las cuatro no es la culpa del programa"
    L += [f"{'peso':<26}{'culpa':>8}{'se mueve':>10}{'antes':>9}{'después':>9}",
          f"{'-'*24:<26}{'-'*7:>8}{'-'*8:>10}{'-'*6:>9}{'-'*7:>9}"]
    filas = []
    for j in range(2):
        for i in range(2):
            filas.append((f"{NEURONAS[j]}, del {ENTRADAS[i][3:]}", 0, i, j))
    for j in range(2):
        filas.append((f"la final, de {NEURONAS[j]}", 1, j, 0))
    for nombre, capa, i, j in filas:
        culpa = gW[capa][i, j]
        antes, despues = red.W[capa][i, j], siguiente.W[capa][i, j]
        assert abs((antes - POQUITO * culpa) - despues) < 1e-12, "el paso no es el del programa"
        L.append(f"{nombre:<26}{firmado(culpa, 4):>8}{firmado(-POQUITO * culpa, 4):>10}"
                 f"{coma(antes, 4):>9}{coma(despues, 4):>9}")
    tabla(L)
    return gW


def bloque_6():
    L = ["6. LO QUE DEJA PASAR CADA UNA",
         "(cuánto se mueve lo que dice la neurona por cada unidad",
         "que se mueve lo que le llega; medido subiendo lo que le",
         "llega una millonésima)",
         "",
         f"{'pasa del':>9}{'la rampa corta':>22}{'el codo':>22}",
         f"{'listón':>9}{'dice':>10}{'deja pasar':>12}{'dice':>10}{'deja pasar':>12}",
         f"{'por':>9}",
         f"{'-'*7:>9}{'-'*6:>10}{'-'*10:>12}{'-'*6:>10}{'-'*10:>12}"]
    for x in PUNTOS_RAMPA:
        etiqueta = "0" if x == 0 else ("+" if x > 0 else "-") + coma(abs(x), 1 if x % 1 else 0)
        if x == 0:
            # justo en el listón el codo se dobla: por debajo no deja pasar nada y por encima,
            # todo. No se imprime un número que dependa de hacia qué lado se mide.
            codo_dice, codo_pasa = coma(0.0, 1), "(el codo)"
        else:
            codo_dice, codo_pasa = coma(float(el_codo(x)), 1), coma(deja_pasar(el_codo, x), 3)
        L.append(f"{etiqueta:>9}{coma(float(rampa(x)), 3):>10}{coma(deja_pasar(rampa, x), 3):>12}"
                 f"{codo_dice:>10}{codo_pasa:>12}")
    xs = np.linspace(-10, 10, 20001)
    maximo = max(deja_pasar(rampa, float(x)) for x in xs)
    L += ["", f"lo más que deja pasar la rampa corta, en cualquier sitio:",
          f"{coma(maximo, 3)} (en el centro, donde pasa del listón por 0)"]
    tabla(L)
    return maximo


def bloque_7():
    """La culpa capa a capa, en la red de cinco capas en medio que imprime retropropagacion.py."""
    rng = np.random.default_rng(SEMILLA)
    red = None
    for prof in [2, 5, 10, 20]:            # el mismo orden de tiradas que medir_desvanecimiento
        r = Red([ANCHO] + [ANCHO] * prof + [1])
        X = rng.normal(size=(64, ANCHO)); y = (rng.random(64) > 0.5).astype(float)
        if prof == PROFUNDIDAD_CADENA:
            red, Xc, yc = r, X, y
    gW, _ = red.gradiente_retro(Xc, yc)
    m = [float(np.abs(g).mean()) for g in gW]
    n = len(m)
    L = [f"7. LA CULPA, CAPA A CAPA, EN LA RED DE {PROFUNDIDAD_CADENA} CAPAS EN MEDIO",
         f"(como en los 80: {ANCHO} neuronas por capa, recién arrancada,",
         f"sin entrenar; la culpa se mide con {len(Xc)} ejemplos de {ANCHO}",
         "números puestos al azar, cada uno con un sí o un no al azar)",
         "",
         f"entre la entrada, las {PROFUNDIDAD_CADENA} capas de en medio y la final hay",
         f"{n} capas de líneas, cada una con sus pesos; se cuentan desde",
         "la entrada: la 1.ª es la que sale de la entrada, y la de",
         "encima de cada una es la siguiente, más cerca de la final",
         "",
         "la culpa de una capa de líneas es la media de la culpa de",
         "sus pesos, sin mirar el signo",
         "",
         f"{'capa de líneas':<24}{'culpa':>12}{'de cada 1 de':>16}",
         f"{'(de la final hacia':<24}{'':>12}{'la de encima,':>16}",
         f"{'la entrada)':<24}{'':>12}{'le llega':>16}",
         f"{'-'*22:<24}{'-'*9:>12}{'-'*14:>16}"]
    for k in range(n - 1, -1, -1):
        nombre = (f"la {k + 1}.ª, la última" if k == n - 1 else
                  "la 1.ª, la primera" if k == 0 else f"la {k + 1}.ª")
        factor = "" if k == n - 1 else coma(m[k] / m[k + 1], 3)
        L.append(f"{nombre:<24}{coma(m[k], 7):>12}{factor:>16}")
    veces = m[-1] / m[0]
    L += ["",
          f"de la última a la primera: {coma(m[-1], 7)} entre {coma(m[0], 7)};",
          f"la culpa se queda en una {miles(round(veces))}.ª parte: {miles(round(veces))} veces menos"]
    tabla(L)
    return m


def bloque_7_escala(factor_20):
    mm = KILOMETRO_EN_MM / factor_20
    L = ["A ESCALA: SI LA CULPA DE LA ÚLTIMA CAPA FUERA UN KILÓMETRO",
         f"con 20 capas en medio, a la primera le llegarían {coma(mm, 3)} mm"]
    tabla(L)


def terreno(red):
    capa, i, j = PESO_DEL_TERRENO
    xs = np.linspace(TERRENO_DESDE, TERRENO_HASTA, TERRENO_PUNTOS)
    otra = copia(red)
    ys = []
    for v in xs:
        otra.W[capa][i, j] = v
        ys.append(otra.error(TABLA_XOR, Y_XOR))
    # los pasos cuesta abajo moviendo SOLO ese peso, desde un sitio de la ladera
    otra = copia(red)
    otra.W[capa][i, j] = INICIO_CUESTA_ABAJO
    camino = [(float(otra.W[capa][i, j]), otra.error(TABLA_XOR, Y_XOR))]
    for n in range(1, PASOS_CUESTA_ABAJO + 1):
        gW, _ = otra.gradiente_retro(TABLA_XOR, Y_XOR)
        otra.W[capa][i, j] -= POQUITO * gW[capa][i, j]
        camino.append((float(otra.W[capa][i, j]), otra.error(TABLA_XOR, Y_XOR)))
    # el mapa de dos pesos
    (c1, i1, j1), (c2, i2, j2) = PESOS_DEL_MAPA
    malla = np.linspace(MAPA_DESDE, MAPA_HASTA, MAPA_PUNTOS)
    mapa = np.zeros((MAPA_PUNTOS, MAPA_PUNTOS))
    otra = copia(red)
    for a, va in enumerate(malla):
        for b, vb in enumerate(malla):
            otra.W[c1][i1, j1] = va
            otra.W[c2][i2, j2] = vb
            mapa[b, a] = otra.error(TABLA_XOR, Y_XOR)
    return xs, np.array(ys), camino, malla, mapa


def atascadas():
    """Las redes de los 200 arranques que no resuelven el o exclusivo: qué error tienen, y qué
    pasa si se les dan PASOS_EXTRA_ATASCADAS pasos más."""
    semillas = [int(s) for s in
                np.random.default_rng(SEMILLA_BASE).integers(1, 2**31 - 1, size=ARRANQUES)]
    # Cada arranque es independiente y tiene su semilla: repartirlos entre los núcleos del
    # ordenador no cambia ningún número, solo lo que se tarda.
    from multiprocessing import Pool
    with Pool() as pool:
        filas = pool.map(un_arranque, semillas)
    buenas = [f[1] for f in filas if f[0] == "buena"]
    malas = [f[1:] for f in filas if f[0] == "mala"]
    return buenas, malas


def un_arranque(s):
    red = Red([2, OCULTAS, 1], semilla=s).entrenar(TABLA_XOR, Y_XOR, TASA, PASOS)
    sal = red.adelante(TABLA_XOR)[-1].ravel()
    if ((sal > 0.5) == (Y_XOR > 0.5)).all():
        return ("buena", red.error(TABLA_XOR, Y_XOR))
    antes = red.error(TABLA_XOR, Y_XOR)
    red.entrenar(TABLA_XOR, Y_XOR, TASA, PASOS_EXTRA_ATASCADAS)
    sal2 = red.adelante(TABLA_XOR)[-1].ravel()
    resuelta = bool(((sal2 > 0.5) == (Y_XOR > 0.5)).all())
    dudas = int((np.abs(sal2 - 0.5) < 0.05).sum())
    return ("mala", antes, red.error(TABLA_XOR, Y_XOR), resuelta, dudas)


def bloque_8(red, xs, ys, camino, malas, buenas):
    capa, i, j = PESO_DEL_TERRENO
    k = int(np.argmin(ys))
    L = ["8. EL TERRENO DEL ERROR, CON UN SOLO PESO",
         f"(la red entrenada; se mueve solo la línea de la primera a",
         "la final, y los demás pesos se quedan quietos; el error es",
         "la media de las cuatro posiciones)",
         "",
         f"donde lo dejó el entrenamiento: peso {coma(red.W[capa][i, j], 2)}, "
         f"error {coma(red.error(TABLA_XOR, Y_XOR), 4)}",
         f"el fondo de esta curva: peso {coma(xs[k], 1)}, error {coma(ys[k], 4)}",
         f"con el peso en {firmado(xs[0], 0)}: error {coma(ys[0], 4)}; "
         f"en {firmado(xs[-1], 0)}: {coma(ys[-1], 4)}",
         "",
         f"cuesta abajo, moviendo solo ese peso, desde {coma(INICIO_CUESTA_ABAJO, 1)}:",
         f"{'paso':>6}{'peso':>10}{'error':>10}{'lo que se movió':>20}",
         f"{'':>6}{'':>10}{'':>10}{'en esos pasos':>20}"]
    for n in range(0, len(camino), CADA_CUANTOS):
        v, e = camino[n]
        movido = "" if n == 0 else coma(v - camino[n - CADA_CUANTOS][0], 2)
        L.append(f"{n:>6}{coma(v, 2):>10}{coma(e, 4):>10}{movido:>20}")
    tabla(L)
    siguen = [m for m in malas if not m[2]]
    salen = [m for m in malas if m[2]]
    rango = lambda xs: (coma(min(xs), 4) if abs(min(xs) - max(xs)) < 5e-5
                        else f"{coma(min(xs), 4)} a {coma(max(xs), 4)}")
    L = [f"LAS QUE SE ATASCAN: {len(malas)} DE {len(malas) + len(buenas)} ARRANQUES",
         f"(error tras {miles(PASOS)} pasos, y después de darles",
         f"{miles(PASOS_EXTRA_ATASCADAS)} pasos más)",
         "",
         f"{'':<22}{'tras':>18}{'con los pasos':>20}",
         f"{'':<22}{miles(PASOS) + ' pasos':>18}{'de más':>20}",
         f"{'las ' + str(len(siguen)) + ' que siguen':<22}{rango([m[0] for m in siguen]):>18}"
         f"{rango([m[1] for m in siguen]):>20}",
         f"{('la que sale' if len(salen) == 1 else 'las ' + str(len(salen)) + ' que salen'):<22}"
         f"{rango([m[0] for m in salen]):>18}{rango([m[1] for m in salen]):>20}",
         f"{'las que resuelven':<22}{rango(buenas):>18}",
         "",
         f"de las {len(siguen)} que siguen, dicen entre 0,45 y 0,55, «no lo",
         f"tengo claro», en dos posiciones: {sum(1 for m in siguen if m[3] == 2)}",
         "",
         "el error de decir 0,5 en dos posiciones y acertar las otras",
         f"dos: 0,5 por 0,5 es {coma(0.25, 2)}; dos veces, {coma(0.5, 1)}; entre cuatro, {coma(0.125, 3)}"]
    assert salen, "se esperaba que alguna saliera con los pasos de más"
    tabla(L)


def guardar_datos(red_final, red_caso, c, xs, ys, camino, malla, mapa):
    """Lo que necesitan las figuras, en un fichero de datos al lado de la salida."""
    with open(AQUI / SALIDA_DATOS, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["que", "a", "b", "valor"])
        for nombre, red in (("final", red_final), ("caso", red_caso)):
            for i in range(2):
                for j in range(2):
                    w.writerow([f"{nombre}_peso_medio", i, j, repr(float(red.W[0][i, j]))])
            for j in range(2):
                w.writerow([f"{nombre}_liston_medio", j, "", repr(float(-red.b[0][j]))])
                w.writerow([f"{nombre}_peso_final", j, "", repr(float(red.W[1][j, 0]))])
            w.writerow([f"{nombre}_liston_final", "", "", repr(float(-red.b[1][0]))])
        act = red_final.adelante(TABLA_XOR)
        for p in range(4):
            for j in range(2):
                w.writerow(["final_dice_medio", p, j, repr(float(act[1][p, j]))])
            w.writerow(["final_dice", p, "", repr(float(act[2][p, 0]))])
        for clave in ("culpa_final", "dice_final", "falla"):
            w.writerow([f"caso_{clave}", "", "", repr(float(c[clave]))])
        for j in range(2):
            for clave in ("dice_medio", "le_toca_medio", "culpa_medio", "pasa_rampa_medio"):
                w.writerow([f"caso_{clave}", j, "", repr(float(c[clave][j]))])
            w.writerow(["caso_culpa_lineas_final", j, "", repr(float(c["culpa_lineas_final"][j]))])
            for i in range(2):
                w.writerow(["caso_culpa_pesos_medio", i, j, repr(float(c["culpa_pesos_medio"][i, j]))])
        w.writerow(["caso_pasa_rampa_final", "", "", repr(float(c["pasa_rampa_final"]))])
        # la cadena tal como se imprime (cuatro cifras): es la que dibuja la figura
        m = c["mano"]
        for clave in ("dice_final", "falla", "culpa_final", "pasa_final"):
            w.writerow([f"mano_{clave}", "", "", repr(m[clave])])
        for j in range(2):
            for clave in ("dice_medio", "le_toca", "pasa_medio", "culpa_medio", "lineas_final"):
                w.writerow([f"mano_{clave}", j, "", repr(m[clave][j])])
            for i in range(2):
                w.writerow(["mano_lineas_primera", i, j, repr(m["lineas_primera"][i][j])])
        for x, y in zip(xs, ys):
            w.writerow(["terreno", repr(float(x)), "", repr(float(y))])
        for n, (v, e) in enumerate(camino):
            w.writerow(["camino", n, repr(v), repr(e)])
        for a, va in enumerate(malla):
            for b, vb in enumerate(malla):
                w.writerow(["mapa", repr(float(va)), repr(float(vb)), repr(float(mapa[b, a]))])


# ============================ SELFTEST ============================

def selftest():
    fallos = []
    comprobar_entrada()

    # [1] TEST NULO — en una posición que la red ya acierta del todo (lo que falla es cero), no
    #     hay culpa que repartir: todos los pesos tienen que salir con culpa cero. Se fabrica
    #     cambiando lo que «tendría que decir» por lo que ya dice.
    red = red_en_el_paso(PASO_DEL_CASO)
    s = red.adelante(TABLA_XOR[3:4])[-1].ravel()
    gW, gb = red.gradiente_retro(TABLA_XOR[3:4], s)
    nulo = max(float(np.abs(g).max()) for g in gW + gb)
    print(f"[1] test nulo         si no falla, la mayor culpa de un peso es {nulo:.1e} (tiene que ser 0)")
    if nulo != 0.0:
        fallos.append(f"test nulo: sin fallo sale culpa {nulo}")

    # [2] SEÑAL IMPLANTADA — la red es la de la figura: la que figura_apilar elige, con su
    #     reparto, y la cuenta a mano del caso da lo mismo que el atajo del programa.
    from figura_apilar import red_representativa
    rr, _, reglas, semilla = red_representativa()
    print(f"[2] señal implantada  la red de la figura es la de la semilla {semilla}: "
          f"{'sí' if semilla == SEMILLA_RED else 'NO'}")
    if semilla != SEMILLA_RED:
        fallos.append(f"señal: la figura usa la semilla {semilla} y este programa {SEMILLA_RED}")
    final = red_en_el_paso(PASOS)
    if not all(np.allclose(a, b) for a, b in zip(final.W + final.b, rr.W + rr.b)):
        fallos.append("señal: la red entrenada aquí no es la de la figura")
    p = int(np.argmax(np.abs(red.adelante(TABLA_XOR)[-1].ravel() - Y_XOR)))
    c = caso(red, p)
    gW, _ = red.gradiente_retro(TABLA_XOR[p:p + 1], Y_XOR[p:p + 1])
    dif = max(float(np.abs(gW[0] - c["culpa_pesos_medio"]).max()),
              float(np.abs(gW[1][:, 0] - c["culpa_lineas_final"]).max()))
    print(f"[2] señal implantada  la cuenta a mano del caso y el atajo del programa difieren en "
          f"{dif:.1e}")
    if dif > TOL_BRUTA:
        fallos.append(f"señal: la cuenta a mano y el atajo difieren en {dif}")

    # [3] INVARIANTE DEL DOMINIO — la culpa por el atajo es la de mover el peso un pelín, en
    #     los seis pesos del caso; y lo que deja pasar la rampa no pasa nunca de una cuarta parte.
    peor = 0.0
    for capa, i, j in [(0, a, b) for a in range(2) for b in range(2)] + [(1, 0, 0), (1, 1, 0)]:
        otra = copia(red)
        otra.W[capa][i, j] += PELIN
        mas = error_de_un_ejemplo(otra, p)
        otra.W[capa][i, j] -= 2 * PELIN
        menos = error_de_un_ejemplo(otra, p)
        peor = max(peor, abs((mas - menos) / (2 * PELIN) - gW[capa][i, j]))
    maximo = max(deja_pasar(rampa, float(x)) for x in np.linspace(-10, 10, 2001))
    print(f"[3] invariante        atajo contra fuerza bruta, diferencia máxima {peor:.1e}; "
          f"la rampa deja pasar como mucho {maximo:.4f}")
    if peor > TOL_BRUTA:
        fallos.append(f"invariante: atajo y fuerza bruta difieren en {peor}")
    if maximo > 0.25 + 1e-6:
        fallos.append(f"invariante: la rampa deja pasar {maximo}, más de una cuarta parte")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    comprobar_entrada()
    if args.selftest:
        return selftest()
    print("########## capítulo 4: la culpa, hecha a mano (L24) ##########")
    print(f"la red de la figura del pasillo: semilla {SEMILLA_RED}, {miles(PASOS)} pasos,")
    print(f"poquito {coma(POQUITO, 1)}; nada de lo que sigue depende del ordenador")
    print()
    final = red_en_el_paso(PASOS)
    bloque_1(final)
    bloque_2(final)
    regla = bloque_3(final)
    assert regla == [(1, 0)], f"la final debería encender solo con «la primera sí, la segunda no»: {regla}"
    red = red_en_el_paso(PASO_DEL_CASO)
    p = int(np.argmax(np.abs(red.adelante(TABLA_XOR)[-1].ravel() - Y_XOR)))
    print(f"(la posición del ejemplo es la que más falla en ese paso)")
    print()
    c = bloque_4(red, p)
    bloque_4_bruto(red, p, c)
    bloque_5(red)
    bloque_6()
    bloque_7()
    from culpa_que_se_desvanece import factores, MONTAJES
    f20 = [f for f in factores(MONTAJES[0]) if f["capas"] == 20][0]["veces"]
    bloque_7_escala(f20)
    xs, ys, camino, malla, mapa = terreno(final)
    buenas, malas = atascadas()
    bloque_8(final, xs, ys, camino, malas, buenas)
    guardar_datos(final, red, c, xs, ys, camino, malla, mapa)
    print(f"escrito {SALIDA_DATOS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
