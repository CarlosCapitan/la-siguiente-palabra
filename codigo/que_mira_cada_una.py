#!/usr/bin/env python3
"""
Capítulo de los conceptos — qué mira cada neurona de una capa.

El libro dice que una capa son varios comités a la vez, cada uno con una
pregunta distinta sobre lo mismo, y que nadie les reparte las preguntas. Esto lo
mide: entrena una red con una capa de en medio sobre dígitos manuscritos de verdad,
y luego mira, una por una, qué ha acabado mirando cada neurona de esa capa y cuánto
acierta ella sola.

La afirmación que sostiene el capítulo es que **ninguna de ellas es la respuesta**:
la red entera acierta más que cualquiera de sus neuronas de en medio por separado.
Eso lo comprueba el selftest.

Uso:
    python que_mira_cada_una.py
    python que_mira_cada_una.py --selftest
"""

# ======================= CONSTANTES =======================

# La pregunta es «¿este dígito es par?». Se elige a propósito porque una raya sola se
# queda corta con ella: aquí sale 91,9 %, y con una capa de en medio, 97,9 %, las dos
# como media de cinco entrenamientos desde cero.
# OJO con la justificación vieja, que estaba escrita aquí y era FALSA: decía que no hay
# ninguna forma que compartan el 0, el 2, el 4, el 6 y el 8 y que no tenga ninguno de
# los impares, y que esto «es el o exclusivo del capítulo anterior escrito a mano».
# `siete_segmentos.py` la pone a prueba en un reloj digital y la tumba: allí una sola
# raya separa par de impar, diez de diez. Y el o exclusivo es IMPOSIBLE para una raya,
# mientras que aquí una raya llega al 91,9 %: no es lo mismo imposible que difícil.
EN_MEDIO = 8                   # neuronas en la capa de en medio
LADO = 8                       # los dibujos son de ocho puntos por ocho
SEMILLA = 20260916
TASA = 0.5
PASOS = 20_000
FRACCION_PRUEBA = 0.3
UMBRAL = 0.5

ANCHURAS = [1, 2, 4, 8, 16]    # cuántas neuronas de en medio, para ver de dónde sale la mejora
REPETICIONES = 5               # arranques distintos por tamaño: una sola tirada oscila
# Los rótulos de las tablas. Son texto del libro: van aquí arriba, con nombre, y no
# escondidos dentro de un print a mitad del fichero.
TITULO_1 = "¿ES PAR ESTE DÍGITO ESCRITO A MANO?"
SUBTITULO_1A = "de cada 100 dígitos que nunca había visto, cuántos acierta"
SUBTITULO_1B = "media de {repeticiones} entrenamientos desde cero; entre paréntesis,"
SUBTITULO_1C = "el peor y el mejor de los {repeticiones}"
TITULO_2 = "LO QUE ACIERTA, ÉL SOLO, CADA UNO DE LOS OCHO COMITÉS DE EN MEDIO"
TITULO_3 = "CUÁNTO SE PARECEN ENTRE SÍ ESOS OCHO COMITÉS"
SUBTITULO_3 = "(0 % serían ocho preguntas sin nada en común; 100 %, ocho copias)"
TITULO_4 = "¿Y SI EN MEDIO HAY MÁS COMITÉS, O MENOS?"
SUBTITULO_4A = "media de {repeticiones} entrenamientos desde cero; entre paréntesis,"
SUBTITULO_4B = "el peor y el mejor de los {repeticiones}"

SALIDA_CSV = "que_mira_cada_una.csv"

# ==========================================================

import argparse
import csv
import sys

from formato import comprobar_ancho, pct, coma
import numpy as np

from perceptron import cargar_digitos
from retropropagacion import Red


def datos(semilla=SEMILLA, permutar=False):
    Xs, t = cargar_digitos()
    ys = np.where(t % 2 == 0, 1.0, 0.0)
    rng = np.random.default_rng(semilla)
    if permutar:
        ys = rng.permutation(ys)
    idx = rng.permutation(len(Xs))
    corte = int(len(Xs) * (1 - FRACCION_PRUEBA))
    return Xs[idx[:corte]], ys[idx[:corte]], Xs[idx[corte:]], ys[idx[corte:]]


def comprobar(Xtr, ytr):
    assert Xtr.shape[1] == LADO * LADO, \
        f"Se esperaban {LADO*LADO} puntos por dibujo; se encontraron {Xtr.shape[1]}"
    assert set(np.unique(ytr)) <= {0.0, 1.0}, \
        f"Se esperaban respuestas de sí o no; se encontró {sorted(set(np.unique(ytr)))}"
    assert Xtr.min() >= 0.0 and Xtr.max() <= 1.0, \
        f"Se esperaban tonos entre 0 y 1; se encontró de {Xtr.min():.2f} a {Xtr.max():.2f}"


def entrenar_con_capa(semilla=SEMILLA, en_medio=EN_MEDIO, permutar=False):
    Xtr, ytr, Xte, yte = datos(semilla, permutar)
    comprobar(Xtr, ytr)
    red = Red([LADO * LADO, en_medio, 1], semilla=semilla).entrenar(Xtr, ytr, TASA, PASOS)
    act = red.adelante(Xte)
    entera = float((((act[-1].ravel() > UMBRAL) == (yte > UMBRAL)).mean()))
    return {"red": red, "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
            "medio": act[1], "entera": entera}


def una_raya_sola(semilla=SEMILLA):
    """La misma pregunta con un solo comité: sin capa de en medio. Es el capítulo
    anterior aplicado a esto, y es la comparación que da sentido a la capa."""
    Xtr, ytr, Xte, yte = datos(semilla)
    red = Red([LADO * LADO, 1], semilla=semilla).entrenar(Xtr, ytr, TASA, PASOS)
    dice = red.adelante(Xte)[-1].ravel() > UMBRAL
    return float((dice == (yte > UMBRAL)).mean())


def acierto_de_cada_una(m):
    """Cuánto acierta cada neurona de en medio ELLA SOLA, mirando solo su respuesta.

    El sentido de cada neurona es arbitrario —nadie le dijo cuál es el sí—, así que se
    le concede el mejor de los dos sentidos. Es concederle ventaja a propósito: si aun
    así acierta menos que la red entera, la afirmación del capítulo aguanta."""
    aciertos = []
    for j in range(m["medio"].shape[1]):
        dice = m["medio"][:, j] > UMBRAL
        a = float((dice == (m["yte"] > UMBRAL)).mean())
        aciertos.append(max(a, 1.0 - a))
    return np.array(aciertos)


def parecido_maximo(red):
    """Lo más que se parecen dos de las neuronas de en medio entre sí, de 0 a 1.

    Si salieran todas iguales, la capa sería una sola neurona repetida ocho veces."""
    W = red.W[0]
    n = W / np.linalg.norm(W, axis=0, keepdims=True)
    s = np.abs(n.T @ n)
    np.fill_diagonal(s, 0.0)
    return float(s.max())


def barrido_de_anchura(semilla=SEMILLA):
    """La misma pregunta con capas de en medio de distinto tamaño. Sirve para ver que
    la mejora no viene de «tener una capa» sino de tener VARIAS neuronas en ella: con
    una sola no se gana nada respecto a no tener capa.

    Cada tamaño se entrena varias veces desde cero, porque una sola tirada oscila lo
    bastante como para dibujar una curva que no existe. Se da la media y el margen."""
    fila = []
    for n in ANCHURAS:
        a = [entrenar_con_capa(semilla + 1000 * k, n)["entera"] for k in range(REPETICIONES)]
        fila.append((n, float(np.mean(a)), float(np.min(a)), float(np.max(a))))
    return fila


def media_de_varias(funcion, semilla=SEMILLA):
    """La media de REPETICIONES entrenamientos desde cero, y el peor y el mejor.

    El titular del capítulo era una sola tirada, y resultó ser LA PEOR de las cinco que ya
    calculaba el barrido de más abajo: el margen de la fila de ocho empezaba justo en ese número.
    La mejora parecía de seis puntos por casualidad: comparaba una peor con otra peor. Un entrenamiento no da
    un número, da un margen, y las dos cifras que el capítulo compara tienen que medir lo mismo.

    Se usan las mismas semillas que barrido_de_anchura, para que las dos tablas del capítulo
    hablen de los mismos cinco entrenamientos y no de dos conjuntos parecidos."""
    a = [funcion(semilla + 1000 * k) for k in range(REPETICIONES)]
    return float(np.mean(a)), float(np.min(a)), float(np.max(a))


def imprimir(m, aciertos, parecido, sola, con_capa):
    """Las tablas que cita el capítulo 3.

    Cada tabla lleva su rótulo y dice de qué son sus números. Antes iban las tres
    seguidas, con la misma columna de porcentajes para tres cosas distintas: un acierto,
    otro acierto y un parecido. Puestos en la misma columna, parecían lo mismo. Un
    porcentaje sin rótulo no es un dato: es una cifra."""
    ANCHO = 34
    lineas = []

    def fila(etiqueta, valor, cola=""):
        lineas.append(f"{etiqueta:<{ANCHO}}{valor:>7}{cola}".rstrip())

    lineas.append(TITULO_1)
    lineas.append(SUBTITULO_1A)
    lineas.append(SUBTITULO_1B.format(repeticiones=REPETICIONES))
    lineas.append(SUBTITULO_1C.format(repeticiones=REPETICIONES))
    lineas.append("")
    for etiqueta, (media, peor, mejor) in (("un solo comité, sin capa", sola),
                                           ("una capa de ocho comités en medio", con_capa)):
        fila(etiqueta, pct(media), f"   (de {pct(peor)} a {pct(mejor)})")
    lineas.append("")
    lineas.append("")

    lineas.append(TITULO_2)
    lineas.append("")
    for j, a in enumerate(aciertos, 1):
        fila(f"el comité número {j}", pct(a))
    lineas.append(f"{'-' * ANCHO}{'-' * 7}")
    fila("el mejor de los ocho", pct(aciertos.max()))
    lineas.append("")
    lineas.append("")

    lineas.append(TITULO_3)
    lineas.append(SUBTITULO_3)
    lineas.append("")
    fila("los dos que más se parecen", pct(parecido))
    lineas.append("")
    lineas.append("")

    lineas.append(TITULO_4)
    lineas.append(SUBTITULO_4A.format(repeticiones=REPETICIONES))
    lineas.append(SUBTITULO_4B.format(repeticiones=REPETICIONES))
    lineas.append("")
    fila("ninguno, un solo comité", pct(sola[0]),
         f"   (de {coma(100 * sola[1])} a {coma(100 * sola[2])})")
    for n, med, lo, hi in barrido_de_anchura():
        # «1 comité» y «2 comités»: el plural se dice bien o no se dice. Una tabla que
        # pone «1 comités» delata que el rótulo se escribió pensando solo en el número.
        cuantos = f"{n:>2} comité" + ("s" if n > 1 else "") + " en medio"
        fila(cuantos, pct(med), f"   (de {coma(100 * lo)} a {coma(100 * hi)})")

    comprobar_ancho(lineas)
    for l in lineas:
        print(l)


def guardar(m, aciertos, parecido, sola, con_capa):
    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["medicion", "cual", "valor"])
        # Las dos cifras del titular son (media, peor, mejor) de REPETICIONES tiradas: se guardan
        # las tres, porque guardar solo la media vuelve a esconder que un entrenamiento es un
        # margen y no un número, que es justo lo que este capítulo acaba de aprender.
        for nombre, (media, peor, mejor) in (("una_raya_sola", sola),
                                             ("con_capa", con_capa)):
            w.writerow([nombre, "media", f"{media:.4f}"])
            w.writerow([nombre, "peor", f"{peor:.4f}"])
            w.writerow([nombre, "mejor", f"{mejor:.4f}"])
        w.writerow(["red_entera", "", f"{m['entera']:.4f}"])
        for j, a in enumerate(aciertos, 1):
            w.writerow(["una_sola", j, f"{a:.4f}"])
        w.writerow(["parecido_maximo", "", f"{parecido:.4f}"])


# ============================ SELFTEST ============================

def selftest():
    fallos = []
    m = entrenar_con_capa()
    aciertos = acierto_de_cada_una(m)

    # [1] Test nulo: con las etiquetas barajadas no hay nada que aprender y la red
    #     entera tiene que caerse al azar. Si no se cae, los mapas que dibujaríamos
    #     no significarían nada.
    nulo = entrenar_con_capa(permutar=True)
    print(f"[1] test nulo         etiquetas barajadas: la red acierta "
          f"{100 * nulo['entera']:.1f} % (azar = 50 %); de verdad: {100 * m['entera']:.1f} %")
    if nulo["entera"] > 0.75:
        fallos.append(f"test nulo: con etiquetas barajadas acierta {100*nulo['entera']:.1f} %, "
                      f"y no debería pasar del azar")

    # [2] Señal implantada: se añade un punto que delata la respuesta siempre. Alguna
    #     neurona de en medio tiene que acabar mirándolo más que a ningún otro punto.
    Xtr, ytr, Xte, yte = datos()
    chivato = np.zeros((len(Xtr), 1)); chivato[ytr > UMBRAL] = 1.0
    Xi = np.hstack([Xtr, chivato])
    red = Red([Xi.shape[1], EN_MEDIO, 1], semilla=SEMILLA).entrenar(Xi, ytr, TASA, PASOS)
    peso_chivato = np.abs(red.W[0][-1, :])
    mayor = np.abs(red.W[0]).max(axis=0)
    cuantas = int((peso_chivato >= mayor - 1e-12).sum())
    print(f"[2] señal implantada  un punto que delata la respuesta: "
          f"{cuantas} de {EN_MEDIO} neuronas lo miran más que a ningún otro")
    if cuantas == 0:
        fallos.append("señal implantada: puse un punto que delata la respuesta y ninguna "
                      "neurona de en medio acabó mirándolo por encima del resto")

    # [3] Invariante del dominio: la red entera tiene que acertar más que cualquiera de
    #     sus neuronas de en medio por separado. Ésta es la afirmación del capítulo: que
    #     ninguna de ellas es la respuesta, y que juntas sí.
    mejor = float(aciertos.max())
    sola = una_raya_sola()
    print(f"[3] invariante        la red entera {100 * m['entera']:.1f} %; la mejor de sus "
          f"neuronas sola {100 * mejor:.1f} %; sin capa {100 * sola:.1f} %")
    if m["entera"] <= mejor:
        fallos.append(f"invariante: la red entera acierta {100*m['entera']:.1f} % y una sola de "
                      f"sus neuronas {100*mejor:.1f} %; entonces la capa no está aportando nada")
    if m["entera"] <= sola:
        fallos.append(f"invariante: con capa {100*m['entera']:.1f} % y sin capa {100*sola:.1f} %; "
                      f"entonces esta tarea no necesita capa y el capítulo no se sostiene")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    m = entrenar_con_capa()
    aciertos = acierto_de_cada_una(m)
    parecido = parecido_maximo(m["red"])
    sola = media_de_varias(una_raya_sola)
    con_capa = media_de_varias(lambda sem: entrenar_con_capa(sem, EN_MEDIO)["entera"])
    imprimir(m, aciertos, parecido, sola, con_capa)
    guardar(m, aciertos, parecido, sola, con_capa)
    return 0


if __name__ == "__main__":
    sys.exit(main())
