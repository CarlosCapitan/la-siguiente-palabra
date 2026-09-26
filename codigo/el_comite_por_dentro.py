#!/usr/bin/env python3
"""
Capítulo 3 — el comité real, por dentro, con todas sus cuentas (L16, T12).

El apartado «Y ahora uno de verdad» enseñaba un comité entrenado de verdad en una sola figura de
cuatro cuadros, sin un solo número a la vista, y afirmaba sin enseñarlo que sus pesos «no dibujan
un cuatro, sino la diferencia entre un cuatro y un nueve». Este programa imprime todo lo que hace
falta para hacerlo con el dedo:

  1. el dibujo de un cuatro, punto a punto, con su tinta (de 0 a 16);
  2. los 64 casos que aprendió el comité, uno por punto;
  3. la cuenta de tres puntos, la de los que más suman y más restan, el total y el listón, y el
     total de un nueve;
  4. el cuatro medio, el nueve medio y su diferencia, y cuánto se parecen los casos a cada cosa.

El comité es el de `figura_una_neurona.py` (el mismo entrenamiento, la misma semilla).

LOS NÚMEROS ENTEROS. El comité trabaja con la tinta de 0 a 1 y con pesos que son múltiplos de
1/32 (la regla de Rosenblatt suma medio dibujo cada vez que corrige, y la tinta va en dieciseisavos).
Multiplicar la tinta por 16, los pesos por 32 y el listón por 16 × 32 = 512 no cambia ninguna
decisión: todo queda multiplicado por lo mismo. Y así todas las cuentas salen enteras, como las del
comité del capítulo 2. El selftest comprueba que decide igual en todos los dibujos.

Uso:
    python el_comite_por_dentro.py --selftest
    python el_comite_por_dentro.py > ../datos/salidas/el_comite_por_dentro.txt
"""

# ======================= CONSTANTES =======================

POR_TINTA = 16                 # la tinta del conjunto va de 0 a 16
POR_PESO = 32                  # los pesos de la regla son múltiplos de 1/32
LADO = 8
PUNTOS_EN_TABLA = 5            # cuántos de los que más suman y de los que más restan se enseñan

# Selftest
ACIERTO_NULO_MAX = 0.75        # con las respuestas barajadas no puede acertar mucho más que el azar
PARECIDO_NULO_MAX = 0.3        # ni sus casos parecerse a la diferencia de verdad
PUNTO_IMPLANTADO = 27          # fila 4, columna 4
TOLERANCIA = 1e-9

# ==========================================================

import argparse
import platform
import sys
from datetime import date

import numpy as np

from formato import ANCHO_CAJA, coma, comprobar_ancho, miles, pct
from figura_una_neurona import DIGITO_NO, DIGITO_SI, ejemplos, entrenar_el_comite
from perceptron import entrenar


def en_enteros(m):
    """Los casos, la tinta y el listón del comité, en números enteros (ver el docstring)."""
    casos = POR_PESO * m["w"]
    assert np.allclose(casos, np.round(casos), atol=TOLERANCIA), \
        "se esperaban pesos múltiplos de 1/32; la regla de perceptron.py ha cambiado"
    tinta = POR_TINTA * m["X"]
    assert np.allclose(tinta, np.round(tinta), atol=TOLERANCIA), \
        "se esperaba tinta en dieciseisavos; el conjunto de dígitos ha cambiado"
    liston = -POR_TINTA * POR_PESO * m["b"]
    assert abs(liston - round(liston)) < TOLERANCIA, "se esperaba un listón entero"
    return np.round(casos).astype(int), np.round(tinta).astype(int), int(round(liston))


def lugar(k):
    """El punto k (de 0 a 63), dicho como se busca en la cuadrícula: fila y columna, desde 1."""
    return f"fila {k // LADO + 1}, columna {k % LADO + 1}"


def cuadricula(valores, ancho=5):
    """Ocho filas de ocho números, con la fila y la columna numeradas."""
    lineas = ["      " + "".join(f"{c:>{ancho}}" for c in range(1, LADO + 1))]
    for f in range(LADO):
        fila = valores[f * LADO:(f + 1) * LADO]
        lineas.append(f"  {f + 1:>2}  " + "".join(f"{int(v):>{ancho}}" for v in fila))
    return lineas


def parecido(a, b):
    """Cuánto se parecen dos cuadrículas: la correlación, de −1 a 1 (1: iguales salvo escala)."""
    return float(np.corrcoef(a, b)[0, 1])


def mismo_lado(casos, diferencia):
    """En cuántos puntos, de los que tienen caso y diferencia distintos de cero, empujan los dos
    hacia el mismo lado."""
    vivos = (casos != 0) & (diferencia != 0)
    return int((np.sign(casos[vivos]) == np.sign(diferencia[vivos])).sum()), int(vivos.sum())


def selftest():
    fallos = []
    m = entrenar_el_comite()
    casos, tinta, liston = en_enteros(m)
    y = m["y"]
    diferencia = tinta[y == 1].mean(0) - tinta[y == -1].mean(0)

    # 1. TEST NULO — con las respuestas barajadas, el comité no aprende nada que se parezca a la
    #    diferencia entre un cuatro y un nueve.
    nulo = entrenar_el_comite(permutar=True)
    casos_nulo = POR_PESO * nulo["w"]
    p_nulo = parecido(casos_nulo, diferencia)
    print(f"[1] test nulo         respuestas barajadas: acierta {pct(nulo['acierto'], 0)}; "
          f"parecido con la diferencia {coma(p_nulo, 2)}")
    if nulo["acierto"] > ACIERTO_NULO_MAX or abs(p_nulo) > PARECIDO_NULO_MAX:
        fallos.append(f"test nulo: con respuestas barajadas acierta {nulo['acierto']:.2f} y se "
                      f"parece {p_nulo:.2f} a la diferencia")

    # 2. SEÑAL IMPLANTADA — dos grupos de dibujos que solo se distinguen en un punto: el caso más
    #    grande tiene que caer justo ahí.
    rng = np.random.default_rng(20260926)
    A = rng.uniform(0, 0.3, size=(200, LADO * LADO))
    etiquetas = np.where(np.arange(200) < 100, 1, -1)
    A[etiquetas == 1, PUNTO_IMPLANTADO] += 0.7
    w, b, _ = entrenar(A, etiquetas)
    mayor = int(np.argmax(np.abs(w)))
    print(f"[2] señal implantada  la diferencia está en el punto {PUNTO_IMPLANTADO}; "
          f"el peso más grande, en el {mayor}")
    if mayor != PUNTO_IMPLANTADO:
        fallos.append(f"señal implantada: el peso más grande cae en {mayor}, no en {PUNTO_IMPLANTADO}")

    # 3. INVARIANTE — en enteros decide lo mismo que con decimales, en todos los dibujos; y la
    #    tabla de la cuenta suma exactamente el total.
    total_ent = tinta @ casos
    decide_ent = total_ent > liston
    decide_dec = (m["X"] @ m["w"] + m["b"]) > 0
    i_si, _ = ejemplos(m)
    aporta = tinta[i_si] * casos
    orden = np.argsort(aporta)
    arriba, abajo = orden[::-1][:PUNTOS_EN_TABLA], orden[:PUNTOS_EN_TABLA]
    resto = np.setdiff1d(np.arange(LADO * LADO), np.concatenate([arriba, abajo]))
    cuadra = aporta[arriba].sum() + aporta[abajo].sum() + aporta[resto].sum() == total_ent[i_si]
    print(f"[3] invariante        decisiones iguales en enteros y en decimales: "
          f"{int((decide_ent == decide_dec).sum())} de {len(y)}; la tabla suma el total: {cuadra}")
    if not (decide_ent == decide_dec).all():
        fallos.append("invariante: en enteros no decide lo mismo que con decimales")
    if not cuadra:
        fallos.append("invariante: la tabla de la cuenta no suma el total")

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
    print("--- selftest ---")
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)

    m = entrenar_el_comite()
    casos, tinta, liston = en_enteros(m)
    y = m["y"]
    i_si, i_no = ejemplos(m)
    total = tinta @ casos
    L = [
        "",
        "########## capítulo 3: el comité real, por dentro ##########",
        f"máquina: {platform.system()} {platform.machine()}. Medido el {date.today().isoformat()}.",
        f"La pregunta: ¿es un {DIGITO_SI} o un {DIGITO_NO}? Aprende con "
        f"{miles(int(len(y) * 0.7))} dibujos",
        f"y acierta {pct(m['acierto'], 1)} de los que no vio.",
        f"Todo en enteros: tinta × {POR_TINTA}, pesos × {POR_PESO}, listón × "
        f"{POR_TINTA * POR_PESO}; decide igual.",
        "",
        f"1. EL DIBUJO DE UN {DIGITO_SI}: LA TINTA DE CADA PUNTO (de 0 a {POR_TINTA})",
        "",
    ]
    L += cuadricula(tinta[i_si], 4)
    L += ["", "2. LO QUE APRENDIÓ: EL PESO DE CADA PUNTO", ""]
    L += cuadricula(casos, 6)
    L += ["", f"  el listón: {miles(liston)}. Dice «es un {DIGITO_SI}» si el total pasa de ahí.",
          "", f"3. LA CUENTA DEL {DIGITO_SI}: TINTA × PESO, PUNTO A PUNTO", ""]
    aporta = tinta[i_si] * casos
    orden = np.argsort(aporta)
    mas, menos = orden[::-1][0], orden[0]
    sin_tinta = [k for k in np.argsort(-np.abs(casos)) if tinta[i_si][k] == 0][0]
    for k in (mas, menos, sin_tinta):
        L.append(f"  {lugar(k):<20} tinta {tinta[i_si][k]:>2} × peso {casos[k]:>5} = "
                 f"{miles(int(aporta[k])):>6}")
    arriba, abajo = orden[::-1][:PUNTOS_EN_TABLA], orden[:PUNTOS_EN_TABLA]
    resto = np.setdiff1d(np.arange(LADO * LADO), np.concatenate([arriba, abajo]))
    L += ["", f"  {'punto':<20}{'tinta':>6}{'peso':>8}{'aporta':>10}",
          f"  {'-----':<20}{'-----':>6}{'----':>8}{'------':>10}"]
    for k in list(arriba) + list(abajo[::-1]):
        L.append(f"  {lugar(k):<20}{tinta[i_si][k]:>6}{casos[k]:>8}{miles(int(aporta[k])):>10}")
    L += [f"  {'los otros ' + str(len(resto)) + ' puntos':<34}{miles(int(aporta[resto].sum())):>10}",
          f"  {'':<34}{'------':>10}",
          f"  {'total del ' + str(DIGITO_SI):<34}{miles(int(total[i_si])):>10}",
          f"  {'listón':<34}{miles(liston):>10}",
          f"  {'¿pasa del listón?':<34}{'sí' if total[i_si] > liston else 'no':>10}",
          "",
          f"  un {DIGITO_NO}, con la misma cuenta: total {miles(int(total[i_no]))}; "
          f"¿pasa del listón? {'sí' if total[i_no] > liston else 'no'}"]

    media_si = tinta[y == 1].mean(0)
    media_no = tinta[y == -1].mean(0)
    diferencia = media_si - media_no
    iguales, vivos = mismo_lado(casos, np.round(diferencia))
    L += ["", f"4. EL {DIGITO_SI} MEDIO, EL {DIGITO_NO} MEDIO Y SU DIFERENCIA (tinta media, redondeada)",
          "", f"  el {DIGITO_SI} medio ({(y == 1).sum()} dibujos)"]
    L += cuadricula(np.round(media_si), 4)
    L += ["", f"  el {DIGITO_NO} medio ({(y == -1).sum()} dibujos)"]
    L += cuadricula(np.round(media_no), 4)
    L += ["", f"  la diferencia: el {DIGITO_SI} medio menos el {DIGITO_NO} medio"]
    L += cuadricula(np.round(diferencia), 4)
    L += ["", "  cuánto se parecen los pesos a cada cosa (de −1 a 1; 1, iguales):",
          f"    {'al ' + str(DIGITO_SI) + ' medio':<24}{coma(parecido(casos, media_si), 2):>6}",
          f"    {'a la diferencia':<24}{coma(parecido(casos, diferencia), 2):>6}",
          f"  puntos donde peso y diferencia empujan al mismo lado: {iguales} de {vivos}",
          "  (de los que no valen cero ni el uno ni la otra)"]

    for l in comprobar_ancho([l.rstrip() for l in L], ANCHO_CAJA):
        print(l)


if __name__ == "__main__":
    main()
