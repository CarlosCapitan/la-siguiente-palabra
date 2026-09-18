#!/usr/bin/env python3
"""
Capítulo 10 — ¿son de verdad «las más frecuentes» las palabras que el capítulo enseña?

El capítulo 10 sostiene su bisagra —«lo que se aprende primero es lo que más abunda»— con
dos observaciones sobre las muestras que imprime, y las dos se pueden comprobar sin volver
a entrenar nada, porque el texto de entrenamiento y las cuatro muestras ya están guardados:

  1. «Han empezado a aparecer palabras reales, y son exactamente las que esperarías: "y",
     "de", "es", "ten", las más cortas y las más frecuentes.»
  2. «espacios repartidos de forma razonable, grupos de letras de longitud plausible».

Este programa NO entrena y NO necesita tarjeta gráfica: cuenta palabras sobre el mismo
texto que se le dio al modelo —el mismo corpus, el mismo recorte y la misma limpieza, por
eso importa el guion del capítulo en vez de copiarle el cargador— y mide las muestras ya
impresas. Da lo mismo en cualquier ordenador y no hay máquina que declarar para las cifras.

Uso:
    python lo_que_mas_abunda.py
    python lo_que_mas_abunda.py --selftest
"""

# ======================= CONSTANTES =======================

SALIDA_CAP10 = "../datos/salidas/en_que_orden_aprende.txt"

# Las cuatro palabras que el capítulo nombra, en el orden en que las nombra. Están aquí y no
# dentro del código para que se vea de un vistazo qué se está comprobando y contra qué frase.
PALABRAS_DEL_CAPITULO = ["y", "de", "es", "ten"]

# Por debajo de este puesto una palabra es «de las más frecuentes» sin discusión; el corte
# está escrito aquí y en la clave impresa, para que sea el lector quien juzgue y no el autor.
PUESTO_FRECUENTE = 100
LARGO_TROZO = 3              # letras por racha, en la segunda tabla

MUESTRAS_ESPERADAS = 4       # las que imprime el capítulo; ni una más (ver el aserto de abajo)
MINIMO_PALABRAS = 50_000     # menos que esto y contar frecuencias no significa nada

# ==========================================================

import argparse
import collections
import datetime
import os
import platform
import re
import sys

from formato import ANCHO_CAJA_CITA, coma, comprobar_ancho, miles, pct


def texto_de_entrenamiento(maximo=None):
    """El mismo texto que vio el modelo, cargado por el mismo guion.

    Se importa `en_que_orden_aprende` en vez de repetir aquí el cargador a propósito: dos
    copias del mismo recorte se separan en cuanto alguien toca una, y entonces este programa
    estaría contando palabras de un texto que el modelo no vio nunca."""
    import en_que_orden_aprende as cap10
    return cap10.cargar_texto(maximo if maximo is not None else cap10.MAX_CARACTERES)


def cuenta(texto):
    """Cuántas veces sale cada palabra y en qué puesto queda, de más a menos frecuente.

    El puesto se calcula una sola vez y se devuelve ya hecho: pedir el puesto de una palabra
    buscándola en una lista ordenada, cada vez, es lo que convierte una comprobación en una
    cuenta que tarda minutos y que nadie vuelve a pasar."""
    veces = collections.Counter(texto.split())
    puesto = {p: i + 1 for i, (p, _) in enumerate(veces.most_common())}
    return veces, puesto


def cuenta_trozos(texto, largo):
    """Cuántas veces sale cada trozo de `largo` letras, y en qué puesto queda.

    Esto es la otra mitad de la historia, y es la que explica el capítulo. El modelo NO ve
    palabras: ve letras, una detrás de otra. Así que lo que aprende primero no son las
    palabras más frecuentes, son las SECUENCIAS DE LETRAS más frecuentes, y algunas de ellas
    resultan ser palabras. «ten» es el ejemplo perfecto: como palabra es rarísima, y como
    trozo de tres letras está entre los treinta y tantos más comunes del castellano —está
    dentro de «tiene», «tener», «intento», «atención», «contento»—.

    Los trozos que cruzan un espacio no se cuentan: el espacio también es un símbolo que el
    modelo predice, pero aquí lo que se busca es qué racha de letras abunda, y «a d» no es
    una racha de letras."""
    veces = collections.Counter()
    for palabra in texto.split():
        for i in range(len(palabra) - largo + 1):
            veces[palabra[i:i + largo]] += 1
    puesto = {t: i + 1 for i, (t, _) in enumerate(veces.most_common())}
    return veces, puesto


def muestras_guardadas(ruta=SALIDA_CAP10):
    """Las cuatro muestras del capítulo, leídas del fichero de salida.

    Valida la entrada ANTES de usarla, y dice cuántas esperaba. Si algún día ese fichero
    lleva dos ejecuciones pegadas —le pasó al del capítulo 9 en cuanto la medición se
    repitió tres veces—, aquí revienta en vez de mezclar las muestras de dos modelos
    distintos en una sola escalera."""
    assert os.path.exists(ruta), f"Se esperaba el fichero de salida «{ruta}»; no existe"
    lineas = open(ruta, encoding="utf-8").read().splitlines()
    fuera = []
    for i, l in enumerate(lineas):
        m = re.match(r"^\[tras ([\d.,]+) pasos", l)
        if m and i + 1 < len(lineas):
            fuera.append((int(m.group(1).replace(",", "").replace(".", "")),
                          lineas[i + 1].rstrip()))
    assert len(fuera) == MUESTRAS_ESPERADAS, (
        f"Se esperaban {MUESTRAS_ESPERADAS} muestras en «{ruta}»; "
        f"se encontraron {len(fuera)}. Dos ejecuciones pegadas en el mismo fichero dan más "
        f"de cuatro, y entonces la escalera que sale no es la de ningún modelo.")
    pasos = [p for p, _ in fuera]
    assert pasos == sorted(set(pasos)), (
        f"Se esperaban cuatro momentos distintos y crecientes; se encontraron {pasos}. "
        f"Un paso repetido es la señal de que hay dos ejecuciones en el fichero.")
    return fuera


def forma(texto):
    """Qué proporción del texto son espacios y cuántas letras tiene el grupo medio.

    Son las dos cosas que el capítulo afirma de la primera muestra —«espacios repartidos de
    forma razonable, grupos de letras de longitud plausible»— y las dos únicas que se pueden
    medir sin decidir antes qué es una palabra, que es justo lo que la máquina no sabe."""
    grupos = texto.split()
    assert grupos, "Se esperaba un texto con algún grupo de letras; se encontró vacío"
    return texto.count(" ") / len(texto), sum(len(g) for g in grupos) / len(grupos)


def tabla_palabras(veces, puesto, total):
    """Las palabras que el capítulo nombra, con su puesto y su peso en el texto.

    Rótulo corto en la cabecera y la clave debajo de la tabla, en una línea, impresa por el
    programa: apilar el rótulo en dos renglones se probó y no se puede verificar, porque los
    dos renglones leídos seguidos entrelazan palabras de columnas distintas."""
    lineas = [
        f"{'palabra':>10}   {'veces':>9}   {'puesto':>7}   {'del texto':>9}",
        f"{'-'*10}   {'-'*9}   {'-'*7}   {'-'*9}",
    ]
    for p in PALABRAS_DEL_CAPITULO:
        n = veces.get(p, 0)
        pu = puesto.get(p)
        lineas.append(f"{p:>10}   {miles(n):>9}   "
                      f"{(miles(pu) if pu else '—'):>7}   {pct(n/total, 3):>9}")
    lineas.append("")
    lineas.append("«puesto» es el lugar en la lista de palabras ordenada de más")
    lineas.append(f"a menos frecuente, entre las {miles(len(veces))} distintas del texto")
    lineas.append("de entrenamiento; «del texto», qué parte de todas las palabras")
    lineas.append("del texto es ésa.")
    return comprobar_ancho(lineas, ANCHO_CAJA_CITA)


def tabla_trozos(texto):
    """Las mismas cadenas del capítulo, contadas como rachas de letras.

    Cada una se cuenta y se ordena entre las rachas de SU MISMO largo: «de» entre las de dos
    letras y «ten» entre las de tres. Compararlas entre sí sería comparar peras con manzanas,
    y la primera versión de esta tabla lo hacía: buscaba «de» en la lista de rachas de tres
    letras, no lo encontraba, y escribía un cero. Un cero que no significaba «no sale nunca»,
    sino «he preguntado mal»."""
    lineas = [f"{'racha':>10}{'letras':>9}{'veces':>12}{'puesto':>10}",
              f"{'-' * 10:>10}{'-' * 6:>9}{'-' * 11:>12}{'-' * 7:>10}"]
    cache = {}
    for p in PALABRAS_DEL_CAPITULO:
        if len(p) not in cache:
            cache[len(p)] = cuenta_trozos(texto, len(p))
        veces, puesto = cache[len(p)]
        assert p in veces, (
            f"Se esperaba encontrar la racha «{p}» entre las de {len(p)} letras del texto de "
            "entrenamiento; no está, así que se está contando sobre otro texto")
        lineas.append(f"{p:>10}{len(p):>9}{miles(veces[p]):>12}"
                      f"{miles(puesto[p]):>10}")
    return comprobar_ancho(lineas, ANCHO_CAJA_CITA), cache


def tabla_forma(muestras, esp_real, largo_real):
    """La forma de cada muestra contra la del texto real."""
    lineas = [
        f"{'momento':>16}   {'espacios':>9}   {'grupo':>7}",
        f"{'-'*16}   {'-'*9}   {'-'*7}",
    ]
    for pasos, texto in muestras:
        e, l = forma(texto)
        lineas.append(f"{miles(pasos) + ' pasos':>16}   {pct(e, 1):>9}   {coma(l, 2):>7}")
    lineas.append(f"{'texto real':>16}   {pct(esp_real, 1):>9}   {coma(largo_real, 2):>7}")
    lineas.append("")
    lineas.append("«espacios» es qué parte de la muestra son espacios; «grupo»,")
    lineas.append("cuántas letras tiene de media lo que va entre dos espacios. La")
    lineas.append("última fila es el castellano que se le dio, medido igual.")
    return comprobar_ancho(lineas, ANCHO_CAJA_CITA)


def selftest():
    fallos = []

    # 1. TEST NULO — en un texto donde cada palabra sale una sola vez no hay ninguna que sea
    #    «la más frecuente»: si el recuento dice que alguna sale más de una vez, está
    #    contando mal, y todo lo demás que imprima este programa es ruido.
    nulo = " ".join(f"pal{i}" for i in range(5_000))
    v_nulo, p_nulo = cuenta(nulo)
    tope = max(v_nulo.values())
    print(f"[1] test nulo         {miles(len(v_nulo))} palabras distintas, la que más sale "
          f"aparece {tope} vez")
    if tope != 1 or len(v_nulo) != 5_000:
        fallos.append(f"test nulo: se esperaban 5.000 palabras distintas con una aparición "
                      f"cada una; la más frecuente aparece {tope} y hay {len(v_nulo)}")

    # 2. SEÑAL IMPLANTADA — una palabra que no existe, metida muchas veces, tiene que subir
    #    al primer puesto. Si no sube, el orden que imprime la tabla no es un orden.
    marca = "qxqxqx"
    implantado = nulo + (" " + marca) * 6_000
    v_imp, p_imp = cuenta(implantado)
    print(f"[2] señal implantada  «{marca}» sale {miles(v_imp[marca])} veces y queda en el "
          f"puesto {p_imp[marca]}")
    if p_imp[marca] != 1:
        fallos.append(f"señal implantada: «{marca}» queda en el puesto {p_imp[marca]}, "
                      f"no en el primero")

    # 3. INVARIANTE DEL DOMINIO — los puestos son una numeración de 1 a N sin huecos ni
    #    repeticiones, las cuentas suman el total de palabras, y una proporción de espacios
    #    está entre 0 y 1. Tres cosas que tienen que cumplirse siempre, mida lo que mida.
    texto = "de la casa   de piedra  y la casa de al lado "
    v, p = cuenta(texto)
    e, l = forma(texto)
    suma_ok = sum(v.values()) == len(texto.split())
    puestos_ok = sorted(p.values()) == list(range(1, len(v) + 1))
    forma_ok = 0.0 <= e <= 1.0 and l > 0
    # Y lo mismo para las rachas de letras, con dos cosas que no pueden fallar nunca: una
    # racha no puede salir menos veces que la palabra entera en la que vive —«casa» sale dos
    # veces, así que «cas» sale dos o más—, y ninguna racha puede cruzar un espacio.
    t, pt = cuenta_trozos(texto, 3)
    dentro_ok = t["cas"] >= v["casa"]
    sin_espacios_ok = not any(" " in x for x in t)
    trozos_ok = sorted(pt.values()) == list(range(1, len(t) + 1))
    print(f"[3] invariante        {len(v)} palabras distintas; los puestos van de 1 a "
          f"{len(v)} sin huecos: {puestos_ok}; suman el total: {suma_ok}; "
          f"espacios {pct(e, 1)}")
    print(f"                      {len(t)} rachas de 3 letras; «cas» sale al menos lo que "
          f"«casa»: {dentro_ok}; ninguna cruza un espacio: {sin_espacios_ok}")
    if not (suma_ok and puestos_ok and forma_ok and dentro_ok and sin_espacios_ok
            and trozos_ok):
        fallos.append(f"invariante: suman el total {suma_ok}, puestos sin huecos "
                      f"{puestos_ok}, forma en rango {forma_ok}, racha dentro de su palabra "
                      f"{dentro_ok}, sin cruzar espacios {sin_espacios_ok}, puestos de las "
                      f"rachas sin huecos {trozos_ok}")

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
    if args.selftest:
        sys.exit(selftest())

    texto = texto_de_entrenamiento()
    veces, puesto = cuenta(texto)
    total = sum(veces.values())
    assert total >= MINIMO_PALABRAS, (
        f"Se esperaban al menos {miles(MINIMO_PALABRAS)} palabras de entrenamiento; "
        f"se encontraron {miles(total)}")
    esp_real, largo_real = forma(texto)
    muestras = muestras_guardadas()

    cabecera = [
        "##### capítulo 10: lo que más abunda, contado #####",
        f"fecha: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"máquina: {platform.system()} {platform.machine()}. No entrena ni",
        "cronometra: cuenta palabras del mismo texto que se le dio al",
        "modelo y mide las cuatro muestras ya guardadas, así que da lo",
        "mismo en cualquier ordenador.",
        "",
        f"texto de entrenamiento: {miles(len(texto))} letras, {miles(total)} palabras,",
        f"{miles(len(veces))} distintas.",
        "",
        "--- LAS PALABRAS QUE EL CAPÍTULO LLAMA «LAS MÁS FRECUENTES» ---",
        "",
    ]
    for l in comprobar_ancho(cabecera, ANCHO_CAJA_CITA):
        print(l)
    for l in tabla_palabras(veces, puesto, total):
        print(l)

    fuera = [p for p in PALABRAS_DEL_CAPITULO
             if puesto.get(p, 10**9) > PUESTO_FRECUENTE]
    print()
    if fuera:
        print(f"De las {len(PALABRAS_DEL_CAPITULO)} palabras que el capítulo nombra, "
              f"{len(fuera)} NO está")
        print(f"entre las {miles(PUESTO_FRECUENTE)} más frecuentes: "
              f"{', '.join('«' + p + '»' for p in fuera)}.")
    else:
        print(f"Las {len(PALABRAS_DEL_CAPITULO)} palabras que el capítulo nombra están "
              f"entre las {miles(PUESTO_FRECUENTE)} más")
        print("frecuentes del texto de entrenamiento.")

    # Y ahora lo mismo contado como lo cuenta el modelo, que no ve palabras sino letras.
    print()
    for l in comprobar_ancho(["--- LAS MISMAS CADENAS, COMO RACHAS DE LETRAS ---", ""],
                             ANCHO_CAJA_CITA):
        print(l)
    lineas, cache = tabla_trozos(texto)
    for l in lineas:
        print(l)
    print()
    for l in comprobar_ancho([
            "Aquí «puesto» es el lugar entre las rachas de letras del MISMO",
            "largo que hay en el texto, contando solo las que caben dentro",
            "de una palabra. Rachas distintas que hay, por largo:",
            *[f"    de {n} letra{'s' if n > 1 else ''}: {miles(len(cache[n][0]))}"
              for n in sorted(cache)],
            "",
            "El modelo no ve palabras: ve letras, una detrás de otra. Así que",
            "ésta es la lista que le importa y no la de arriba, y es la que",
            "explica por qué sale «ten» sin ser una palabra frecuente.",
            ""], ANCHO_CAJA_CITA):
        print(l)

    print("--- LA FORMA DE CADA MUESTRA, CONTRA LA DEL TEXTO REAL ---")
    print()
    for l in tabla_forma(muestras, esp_real, largo_real):
        print(l)


if __name__ == "__main__":
    main()
