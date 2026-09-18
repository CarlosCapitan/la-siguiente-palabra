#!/usr/bin/env python3
"""¿Qué le pasa al ocho cuando se le pone una capa de comités en medio?

    es_un_ocho_con_capa.py            la medición
    es_un_ocho_con_capa.py --selftest las tres pruebas

El capítulo 2 termina midiendo los diez dígitos con un solo comité y encontrando que **el ocho es
el único que no se queda quieto**: da vueltas hasta que se le acaba el plazo y se queda en el peor
acierto de los diez. El capítulo 2 le pide al lector que guarde esa fila.

El capítulo 3 es donde se arregla justamente eso —poniendo comités en medio—, y hasta hoy no
volvía al ocho ni una vez. Esta medición cobra esa promesa.

Predicción escrita ANTES de medir, para que pueda salir mal: con una capa de ocho comités en
medio, el ocho sube del 95,9 % y deja de dar vueltas. Confianza media. Qué la tumbaría: que con
capa el ocho no mejore, o que mejore menos que lo que mejora «¿es par?», que es la pregunta que el
capítulo usa para enseñar que la capa sirve."""
DIGITO = 8                     # la pregunta: ¿es este dígito un ocho?
EN_MEDIO = 8                   # comités en la capa de en medio, los mismos que el capítulo 3
LADO = 8                       # los dibujos son de ocho puntos por ocho
SEMILLA = 20260916             # la misma que que_mira_cada_una.py, para poder comparar
TASA = 0.5
PASOS = 20_000
FRACCION_PRUEBA = 0.3
UMBRAL = 0.5
REPETICIONES = 5               # una sola tirada oscila: se da la media y el margen

TITULO = "¿ES UN OCHO ESTE DÍGITO ESCRITO A MANO?"
SUBTITULO_A = "de cada 100 dígitos que nunca había visto, cuántos acierta"
SUBTITULO_B = "media de {repeticiones} entrenamientos desde cero; entre paréntesis,"
SUBTITULO_C = "el peor y el mejor de los {repeticiones}"

# ==========================================================

import platform, sys, time
import numpy as np

from formato import comprobar_ancho, pct
from perceptron import cargar_digitos
from retropropagacion import Red


def datos(semilla=SEMILLA, permutar=False, chivato=False):
    """Los dibujos, partidos en los que se enseñan y los que no se ven nunca.

    La etiqueta es «¿es un ocho?», no «¿es par?»: ésa es la única diferencia con
    que_mira_cada_una.py, y es a propósito, para que las dos tablas se puedan comparar."""
    Xs, t = cargar_digitos()
    ys = np.where(t == DIGITO, 1.0, 0.0)
    rng = np.random.default_rng(semilla)
    if permutar:
        ys = rng.permutation(ys)
    if chivato:
        delator = np.zeros((len(Xs), 1))
        delator[ys > UMBRAL] = 1.0
        Xs = np.hstack([Xs, delator])
    idx = rng.permutation(len(Xs))
    corte = int(len(Xs) * (1 - FRACCION_PRUEBA))
    return Xs[idx[:corte]], ys[idx[:corte]], Xs[idx[corte:]], ys[idx[corte:]]


def comprobar(Xtr, ytr, puntos=LADO * LADO):
    assert Xtr.shape[1] == puntos, \
        f"Se esperaban {puntos} puntos por dibujo; se encontraron {Xtr.shape[1]}"
    assert set(np.unique(ytr)) <= {0.0, 1.0}, \
        f"Se esperaban respuestas de sí o no; se encontró {sorted(set(np.unique(ytr)))}"
    assert Xtr.min() >= 0.0 and Xtr.max() <= 1.0, \
        f"Se esperaban tonos entre 0 y 1; se encontró de {Xtr.min():.2f} a {Xtr.max():.2f}"
    assert 0.0 < ytr.mean() < 0.5, \
        f"Se esperaba que los ochos fueran minoría y no una rareza; son el {100*ytr.mean():.1f} %"


def entrenar(semilla=SEMILLA, en_medio=EN_MEDIO, permutar=False, chivato=False):
    """Entrena y devuelve qué acierta sobre los dibujos que no vio.

    `en_medio=0` es el capítulo 2: un solo comité, sin capa. Cualquier otro número es el
    capítulo 3: esa cantidad de comités en medio."""
    Xtr, ytr, Xte, yte = datos(semilla, permutar, chivato)
    comprobar(Xtr, ytr, Xtr.shape[1])
    capas = [Xtr.shape[1]] + ([en_medio] if en_medio else []) + [1]
    red = Red(capas, semilla=semilla).entrenar(Xtr, ytr, TASA, PASOS)
    act = red.adelante(Xte)
    dice = act[-1].ravel() > UMBRAL
    return {"red": red, "acierto": float((dice == (yte > UMBRAL)).mean()),
            "medio": act[1] if en_medio else None, "Xte": Xte, "yte": yte}


def acierto_de_cada_uno(m):
    """Cuánto acierta cada comité de en medio ÉL SOLO. El sentido de cada uno es arbitrario
    —nadie le dijo cuál es el sí—, así que se toma el mejor de los dos sentidos."""
    yte = m["yte"] > UMBRAL
    out = []
    for j in range(m["medio"].shape[1]):
        dice = m["medio"][:, j] > UMBRAL
        out.append(max((dice == yte).mean(), (~dice == yte).mean()))
    return np.array(out)


def media_de_varias(en_medio, semilla=SEMILLA):
    a = [entrenar(semilla + 1000 * k, en_medio)["acierto"] for k in range(REPETICIONES)]
    return float(np.mean(a)), float(np.min(a)), float(np.max(a))


def selftest():
    """Tres pruebas, calcadas de las de que_mira_cada_una.py, sobre lo único que esto decide:
    si una capa de comités en medio le sirve de algo al ocho."""
    fallos = []
    m = entrenar()

    # 1. TEST NULO — con las etiquetas barajadas no hay nada que aprender. Pero ojo: aquí el
    #    «no» es el 90 % de los dibujos, así que una máquina que diga siempre «no» ya acierta
    #    el 90 %. El azar de esta pregunta NO es el 50 %: es la proporción de los que no son
    #    ocho, y contra eso hay que compararlo, o el test nulo no prueba nada.
    nulo = entrenar(permutar=True)
    _, ytr, _, yte = datos()
    siempre_no = float((yte <= UMBRAL).mean())
    print(f"[1] test nulo         etiquetas barajadas: acierta {pct(nulo['acierto'])}; "
          f"decir siempre «no» acierta {pct(siempre_no)}; de verdad: {pct(m['acierto'])}")
    if nulo["acierto"] > siempre_no + 0.03:
        fallos.append(f"test nulo: con etiquetas barajadas acierta {pct(nulo['acierto'])}, "
                      f"por encima de decir siempre «no» ({pct(siempre_no)})")

    # 2. SEÑAL IMPLANTADA — un punto de más que delata la respuesta siempre. Algún comité de
    #    en medio tiene que acabar mirándolo más que a ningún otro punto.
    con = entrenar(chivato=True)
    peso = np.abs(con["red"].W[0][-1, :])
    mayor = np.abs(con["red"].W[0]).max(axis=0)
    cuantos = int((peso >= mayor - 1e-12).sum())
    print(f"[2] señal implantada  un punto que delata la respuesta: {cuantos} de {EN_MEDIO} "
          f"comités lo miran más que a ningún otro; acierto {pct(con['acierto'])}")
    if cuantos == 0:
        fallos.append("señal implantada: ningún comité de en medio acabó mirando el punto "
                      "que delata la respuesta por encima del resto")

    # 3. INVARIANTE DEL DOMINIO — la red entera tiene que acertar más que cualquiera de sus
    #    comités de en medio por su cuenta. Es la afirmación del capítulo 3: ninguno de ellos
    #    es la respuesta, y juntos sí.
    mejor = float(acierto_de_cada_uno(m).max())
    print(f"[3] invariante        la red entera {pct(m['acierto'])}; "
          f"el mejor de sus {EN_MEDIO} comités, él solo, {pct(mejor)}")
    if m["acierto"] <= mejor:
        fallos.append(f"invariante: la red entera ({pct(m['acierto'])}) no supera al mejor de "
                      f"sus comités por separado ({pct(mejor)})")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    print(f"Máquina: {platform.machine()}, {platform.system()} {platform.release()}.")
    print(f"Medido el {time.strftime('%Y-%m-%d')}. Semilla: {SEMILLA}.")
    print("La pregunta que el capítulo 2 dejó abierta: el ocho era el único de los diez")
    print("que no se quedaba quieto con un solo comité.")
    print()
    if selftest():
        print("\nEl selftest falla: el número no vale.")
        return 1
    print()

    sin_capa = media_de_varias(0)
    con_capa = media_de_varias(EN_MEDIO)
    lineas = [TITULO,
              SUBTITULO_A,
              SUBTITULO_B.format(repeticiones=REPETICIONES),
              SUBTITULO_C.format(repeticiones=REPETICIONES),
              ""]
    ANCHO = 34
    for etiqueta, (media, peor, mejor) in (("un solo comité, sin capa", sin_capa),
                                           ("una capa de ocho comités en medio", con_capa)):
        lineas.append(f"{etiqueta:<{ANCHO}}{pct(media):>7}   (de {pct(peor)} a {pct(mejor)})")
    comprobar_ancho(lineas)
    print()
    for l in lineas:
        print(l)
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
