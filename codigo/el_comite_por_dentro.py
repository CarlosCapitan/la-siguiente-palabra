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
  4. el cuatro medio, el nueve medio y su diferencia, y cuánto se parecen los casos a cada cosa;
  5. de dónde salen los pesos: las correcciones una a una, y cómo cada peso y el listón son una
     suma de tinta.

El comité es el de `figura_una_neurona.py` (el mismo entrenamiento, la misma semilla).

LA TINTA, TAL COMO VIENE (Orden 38, 23 oct 2026). El comité se entrena con la tinta tal como la da el
conjunto de dígitos, enteros de 0 a 16, sin dividir entre 16. Así la regla de Rosenblatt es
literalmente la del capítulo 2: si el comité no reconoce un cuatro, se suma a cada miembro la tinta
que ve y el listón baja un punto; si toma un nueve por un cuatro, se resta y el listón sube un punto.
Todo sale entero sin multiplicar nada (antes, con la tinta de 0 a 1, los pesos salían en múltiplos de
1/32 y había que multiplicar la tinta por 16, los pesos por 32 y el listón por 512). Y cada peso es,
por construcción, la tinta de ese punto en los cuatros corregidos menos la de los nueves corregidos:
el apartado 5 lo cuenta y el selftest lo comprueba en los 64 puntos.

Uso:
    python el_comite_por_dentro.py --selftest
    python el_comite_por_dentro.py > ../datos/salidas/el_comite_por_dentro.txt
"""

# ======================= CONSTANTES =======================

LADO = 8
PUNTOS_EN_TABLA = 5            # cuántos de los que más suman y de los que más restan se enseñan
TINTA_ALTA = 8                 # «mucha tinta»: de 8 a 16, la mitad de arriba (apartado 5)
PALABRA = {4: "cuatro", 9: "nueve"}   # cómo se dicen los dos dígitos del comité, en las tablas del apartado 5

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

from formato import ANCHO_CAJA, coma, comprobar_ancho, miles, pct, tabla_editorial
from figura_una_neurona import DIGITO_NO, DIGITO_SI, ejemplos, entrenar_el_comite
from perceptron import PASOS_MAX, SEMILLA, TASA, entrenar


def en_enteros(m):
    """Los casos (pesos), la tinta y el listón del comité, que salen enteros sin tocarlos: la
    tinta es entera y la regla suma o resta tinta entera y un punto de listón cada vez."""
    casos = m["w"]
    assert np.array_equal(casos, np.round(casos)), \
        "se esperaban pesos enteros; la regla de perceptron.py o la tinta han cambiado"
    tinta = m["X"]
    assert np.array_equal(tinta, np.round(tinta)) and tinta.min() == 0 and tinta.max() <= 16, \
        "se esperaba tinta entera de 0 a 16; el conjunto de dígitos ha cambiado"
    liston = -m["b"]
    assert liston == round(liston), "se esperaba un listón entero"
    return np.round(casos).astype(int), np.round(tinta).astype(int), int(round(liston))


def correcciones(m):
    """Las correcciones del entrenamiento, una a una: se repite el aprendizaje de
    `perceptron.entrenar` (la misma semilla, el mismo orden) apuntando cada vez que corrige. Devuelve
    la lista de (vuelta, dibujo, +1 si era un cuatro y -1 si era un nueve) y cuántas vueltas dio,
    contando la última, en la que ya no hubo nada que corregir. Se comprueba que acaba en los mismos
    pesos y el mismo listón que el comité de verdad."""
    X, y = m["entrenamiento"]
    X = np.round(X).astype(int)
    rng = np.random.default_rng(SEMILLA)
    w, b = np.zeros(X.shape[1], dtype=int), 0
    orden = np.arange(len(X))
    apuntes = []
    for vuelta in range(1, PASOS_MAX + 1):
        rng.shuffle(orden)
        errores = 0
        for i in orden:
            if y[i] * (X[i] @ w + b) <= 0:
                assert TASA == 1.0, "se esperaba la tasa de la regla del capítulo 2 (1)"
                w += y[i] * X[i]
                b += y[i]
                apuntes.append((vuelta, int(i), int(y[i])))
                errores += 1
        if errores == 0:
            break
    assert errores == 0, "se esperaba que el comité dejara de corregirse"
    assert np.array_equal(w, m["w"]) and b == m["b"], \
        "las correcciones repetidas no dan los pesos y el listón del comité"
    return apuntes, vuelta, X, y


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
    casos_nulo = nulo["w"]
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

    # 3. INVARIANTE — cada peso es la tinta de ese punto en los cuatros corregidos menos la de los
    #    nueves corregidos, en los 64 puntos, y el listón es los nueves corregidos menos los cuatros
    #    (las correcciones repetidas dan los mismos pesos que el comité); el comité decide como dice
    #    su suma en todos los dibujos; y la tabla de la cuenta suma exactamente el total.
    apuntes, vueltas, Xtr, ytr = correcciones(m)
    de_cuatros = [i for _, i, s in apuntes if s == 1]
    de_nueves = [i for _, i, s in apuntes if s == -1]
    pesos_rehechos = Xtr[de_cuatros].sum(0) - Xtr[de_nueves].sum(0)
    pesos_bien = int((pesos_rehechos == casos).sum())
    liston_rehecho = len(de_nueves) - len(de_cuatros)
    total_ent = tinta @ casos
    decide_ent = total_ent > liston
    decide_dec = (m["X"] @ m["w"] + m["b"]) > 0
    i_si, _ = ejemplos(m)
    aporta = tinta[i_si] * casos
    orden = np.argsort(aporta)
    arriba, abajo = orden[::-1][:PUNTOS_EN_TABLA], orden[:PUNTOS_EN_TABLA]
    resto = np.setdiff1d(np.arange(LADO * LADO), np.concatenate([arriba, abajo]))
    cuadra = aporta[arriba].sum() + aporta[abajo].sum() + aporta[resto].sum() == total_ent[i_si]
    print(f"[3] invariante        peso = tinta de los cuatros corregidos menos la de los nueves: "
          f"{pesos_bien} de {LADO * LADO} puntos; listón = nueves menos cuatros corregidos: "
          f"{liston_rehecho} (el del comité, {liston})")
    print(f"                      decide como dice la suma: {int((decide_ent == decide_dec).sum())} "
          f"de {len(y)} dibujos; la tabla suma el total: {cuadra}")
    if pesos_bien != LADO * LADO:
        fallos.append(f"invariante: solo {pesos_bien} de 64 pesos son la tinta de los cuatros "
                      "corregidos menos la de los nueves")
    if liston_rehecho != liston:
        fallos.append(f"invariante: el listón sale {liston_rehecho} de las correcciones y es {liston}")
    if not (decide_ent == decide_dec).all():
        fallos.append("invariante: no decide lo que dice la suma de tinta por peso más el listón")
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
        "La tinta va de 0 a 16, tal como viene en el conjunto de dígitos;",
        "los pesos y el listón salen enteros, sin multiplicar nada.",
        "",
        f"1. EL DIBUJO DE UN {DIGITO_SI}: LA TINTA DE CADA PUNTO (de 0 a 16)",
        "",
    ]
    L += cuadricula(tinta[i_si], 4)
    L += ["", "2. LO QUE APRENDIÓ: EL PESO DE CADA PUNTO", ""]
    L += cuadricula(casos, 6)
    L += ["", f"  el listón: {miles(liston)}. Dice «es un {DIGITO_SI}» si el total pasa de ahí.",
          "", f"3. LA CUENTA DEL {DIGITO_SI}: TINTA POR PESO, PUNTO A PUNTO"]
    # Desde el 9 oct 2026 lo que va al libro (la cuenta y el parecido) sale en tablas de libro
    # (formato.py; REGLAS 6 ter); las cuadrículas, que el libro enseña como figuras, siguen igual.
    for l in comprobar_ancho([l.rstrip() for l in L], ANCHO_CAJA):
        print(l)
    T = []
    aporta = tinta[i_si] * casos
    orden = np.argsort(aporta)
    mas, menos = orden[::-1][0], orden[0]
    sin_tinta = [k for k in np.argsort(-np.abs(casos)) if tinta[i_si][k] == 0][0]
    T.append(tabla_editorial(
        "Tres puntos, para ver cómo va", ["punto", "tinta", "peso", "aporta"],
        [[lugar(k), str(tinta[i_si][k]), miles(int(casos[k])), miles(int(aporta[k]))]
         for k in (mas, menos, sin_tinta)], "iddd",
        ["Aporta: la tinta por el peso."]))
    arriba, abajo = orden[::-1][:PUNTOS_EN_TABLA], orden[:PUNTOS_EN_TABLA]
    resto = np.setdiff1d(np.arange(LADO * LADO), np.concatenate([arriba, abajo]))
    filas = [[lugar(k), str(tinta[i_si][k]), miles(int(casos[k])), miles(int(aporta[k]))]
             for k in list(arriba) + list(abajo[::-1])]
    filas.append([f"los otros {len(resto)} puntos", "", "", miles(int(aporta[resto].sum()))])
    filas.append([f"**total del {DIGITO_SI}**", "", "", f"**{miles(int(total[i_si]))}**"])
    T.append(tabla_editorial(
        f"La cuenta del {DIGITO_SI}, punto a punto", ["punto", "tinta", "peso", "aporta"], filas,
        "iddd",
        [f"Los {PUNTOS_EN_TABLA} puntos que más suman, los {PUNTOS_EN_TABLA} que más restan y lo "
         "que suman juntos todos los demás. Aporta: la tinta por el peso.",
         f"El listón es {miles(liston)}: dice «es un {DIGITO_SI}» si el total pasa de ahí."]))
    T.append(tabla_editorial(
        f"El {DIGITO_SI} y el {DIGITO_NO}, contra el listón",
        ["dibujo", "total", "listón", "¿pasa del listón?"],
        [[f"un {d}", miles(int(total[i])), miles(liston), "sí" if total[i] > liston else "no"]
         for d, i in ((DIGITO_SI, i_si), (DIGITO_NO, i_no))], "iddc",
        [f"La misma cuenta, tinta por peso en los {LADO * LADO} puntos, sumada."]))
    for t in T:
        print()
        print("\n".join(t))

    media_si = tinta[y == 1].mean(0)
    media_no = tinta[y == -1].mean(0)
    diferencia = media_si - media_no
    iguales, vivos = mismo_lado(casos, np.round(diferencia))
    L = ["", f"4. EL {DIGITO_SI} MEDIO, EL {DIGITO_NO} MEDIO Y SU DIFERENCIA (tinta media, redondeada)",
         "", f"  el {DIGITO_SI} medio ({(y == 1).sum()} dibujos)"]
    L += cuadricula(np.round(media_si), 4)
    L += ["", f"  el {DIGITO_NO} medio ({(y == -1).sum()} dibujos)"]
    L += cuadricula(np.round(media_no), 4)
    L += ["", f"  la diferencia: el {DIGITO_SI} medio menos el {DIGITO_NO} medio"]
    L += cuadricula(np.round(diferencia), 4)
    for l in comprobar_ancho([l.rstrip() for l in L], ANCHO_CAJA):
        print(l)
    print()
    print("\n".join(tabla_editorial(
        "Cuánto se parecen los pesos a cada cosa",
        ["los pesos, comparados con", "cuánto se parecen"],
        [[f"el {DIGITO_SI} medio", coma(parecido(casos, media_si), 2)],
         ["la diferencia", coma(parecido(casos, diferencia), 2)],
         ["puntos donde peso y diferencia empujan al mismo lado", f"{iguales} de {vivos}"]],
         "id",
        ["Cuánto se parecen: de −1 a 1; 1 quiere decir iguales salvo el tamaño de los números, y "
         "0, que no tienen nada que ver.",
         "Mismo lado: se cuentan solo los puntos donde ni el peso ni la diferencia valen cero."])))
    de_donde_salen_los_pesos(m, casos, tinta, liston)


def de_donde_salen_los_pesos(m, casos, tinta, liston):
    """El apartado 5: cada peso y el listón, contados corrección a corrección."""
    apuntes, vueltas, Xtr, ytr = correcciones(m)
    de_cuatros = [i for _, i, s in apuntes if s == 1]
    de_nueves = [i for _, i, s in apuntes if s == -1]
    n4, n9 = len(de_cuatros), len(de_nueves)
    si, no = PALABRA[DIGITO_SI], PALABRA[DIGITO_NO]       # «cuatro», «nueve»
    con_correccion = len({v for v, _, _ in apuntes})
    print()
    print(f"5. DE DÓNDE SALEN LOS PESOS")
    print()
    print("\n".join(tabla_editorial(
        "Las correcciones que hizo el comité al aprender",
        ["", "cuántas"],
        [["correcciones en total", str(n4 + n9)],
         [f"de ellas, con un {si}", str(n4)],
         [f"de ellas, con un {no}", str(n9)],
         ["vueltas por todos los dibujos", str(vueltas)]], "id",
        [f"Con un {si}: el comité no lo reconocía y se sumó a cada peso la tinta del "
         f"dibujo. Con un {no}: lo tomó por un {si} y se restó.",
         f"Vueltas: en las {con_correccion} primeras corrigió algo; en la última ya no tuvo que "
         "corregir nada y se quedó quieto."])))
    mas, menos = int(np.argmax(casos)), int(np.argmin(casos))
    filas = []
    for k in (mas, menos):
        suma4 = int(Xtr[de_cuatros][:, k].sum())
        suma9 = int(Xtr[de_nueves][:, k].sum())
        assert suma4 - suma9 == casos[k], (k, suma4, suma9, casos[k])
        filas.append([lugar(k), miles(suma4), miles(suma9), miles(int(casos[k]))])
    print()
    print("\n".join(tabla_editorial(
        "De dónde sale el peso de los dos puntos que más cuentan",
        ["punto", f"tinta sumada con los {si}s", f"tinta restada con los {no}s",
         "peso"], filas, "iddd",
        [f"Tinta sumada con los {si}s: la suma de la tinta de ese punto en las {n4} "
         f"correcciones con un {si}. Tinta restada con los {no}s: lo mismo en las "
         f"{n9} correcciones con un {no}. El peso es la primera menos la segunda.",
         "Son el punto de peso más positivo y el de peso más negativo."])))
    baja, sube = n4, n9
    assert 0 - baja + sube == liston, (baja, sube, liston)
    print()
    print("\n".join(tabla_editorial(
        "De dónde sale el listón",
        ["", "listón"],
        [["empieza en", "0"],
         [f"baja con los {n4} {si}s corregidos, un punto cada vez", miles(-baja)],
         [f"sube con los {n9} {no}s corregidos, un punto cada vez", f"+{miles(sube)}"],
         ["**acaba en**", f"**{miles(liston)}**"]], "id",
        [f"Dice «es un {si}» si el total pasa del listón."])))
    Xp, yp = m["prueba"]
    pred = np.where(Xp @ m["w"] + m["b"] > 0, 1, -1)
    mal = pred != yp
    fallos4, fallos9 = int((mal & (yp == 1)).sum()), int((mal & (yp == -1)).sum())
    print()
    print("\n".join(tabla_editorial(
        "Lo que acierta de los dibujos que no vio",
        ["", "dibujos"],
        [["dibujos que no vio", str(len(yp))],
         ["los que acierta", str(int((~mal).sum()))],
         [f"los que falla, que son un {si}", str(fallos4)],
         [f"los que falla, que son un {no}", str(fallos9)]], "id",
        [f"Acierto: {pct(float((~mal).mean()), 1)}."])))
    cuatros, nueves = Xtr[ytr == 1], Xtr[ytr == -1]
    filas = []
    for k in (mas, menos):
        a4 = int((cuatros[:, k] >= TINTA_ALTA).sum())
        a9 = int((nueves[:, k] >= TINTA_ALTA).sum())
        filas.append([lugar(k), f"{a4} de {len(cuatros)} ({pct(a4 / len(cuatros), 0)})",
                      f"{a9} de {len(nueves)} ({pct(a9 / len(nueves), 0)})"])
    print()
    print("\n".join(tabla_editorial(
        f"Cuántos dibujos tienen tinta {TINTA_ALTA} o más en esos dos puntos",
        ["punto", f"de los {si}s para aprender", f"de los {no}s para aprender"],
        filas, "idd",
        [f"Los dibujos con los que aprendió: {len(cuatros)} {si}s y {len(nueves)} {no}s. "
         f"Tinta {TINTA_ALTA} o más: de la mitad de arriba de la escala, que va de 0 a 16."])))


if __name__ == "__main__":
    main()
