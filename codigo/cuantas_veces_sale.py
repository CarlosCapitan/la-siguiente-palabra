#!/usr/bin/env python3
"""¿Cuántas de cinco tiradas enseñan la avería de las palabras con dos significados?

    cuantas_veces_sale.py            la medición (entrena cinco veces: tarda)
    cuantas_veces_sale.py --selftest las tres pruebas

El capítulo 5 enseña las vecinas de «banco» en UNA tirada y sostiene sobre ellas la avería que
abre el camino a los capítulos 6 y 8: que el método mete los dos significados de una palabra en el
mismo sitio. La tirada del libro sale bien. Pero entrenar no da un número, da un margen —el
capítulo 3 lo enseña con todas las letras—, y una avería demostrada con una tirada afortunada no
está demostrada.

Esto entrena cinco veces desde cero, cambiando solo la semilla, y cuenta **en cuántas de las cinco
aparece al menos una vecina de cada uno de los dos significados**. Si sale en las cinco, el
ejemplo del libro vale. Si sale en dos, el libro tiene que decirlo o cambiar de ejemplo.

Las listas de qué vecina pertenece a qué significado están abajo, con nombre, y son el único
juicio humano de este programa: se escriben ANTES de mirar ninguna salida, y se dejan a la vista
para que cualquiera pueda discutirlas. Una vecina que no esté en ninguna de las dos listas no
cuenta para ningún significado."""
SEMILLAS = [20260914, 20260915, 20260916, 20260917, 20260918]
VECINOS = 5

# Las palabras con dos significados que se ponen a prueba, y qué vecinas delatan cada significado.
# Escrito antes de mirar ninguna salida. Una vecina fuera de las dos listas no cuenta.
PALABRAS = [
    {"palabra": "banco",
     "sentidos": ("el mueble", "el del dinero"),
     "delatan": (["escritorio", "baul", "armario", "sillon", "silla", "mesa", "asiento",
                  "taburete", "arca", "cofre", "mueble", "banqueta", "sofa"],
                 ["almacen", "ferrocarril", "empresa", "credito", "deuda", "hipoteca",
                  "capital", "bolsa", "comercio", "negocio", "sociedad", "compania",
                  "prestamo", "acciones", "bancos", "letra", "giro"])},
    {"palabra": "cura",
     "sentidos": ("el sacerdote", "el remedio"),
     "delatan": (["sacerdote", "parroco", "obispo", "fraile", "clerigo", "vicario",
                  "capellan", "sacristan", "abad", "canonigo", "barbero"],
                 ["boticario", "medico", "remedio", "medicina", "enfermedad", "cirujano",
                  "herida", "salud", "curacion", "sangria", "dolencia", "medicamento"])},
    {"palabra": "lengua",
     "sentidos": ("el idioma", "la del cuerpo"),
     "delatan": (["idioma", "castellana", "castellano", "latina", "griega", "habla",
                  "lenguaje", "palabra", "gramatica", "dialecto", "idiomas"],
                 ["boca", "labios", "dientes", "garganta", "paladar", "saliva",
                  "diente", "labio", "mandibula"])},
]

# ==========================================================

import sys, unicodedata

from formato import comprobar_ancho, miles
from palabras_numeros import (CORPUS_BIBLIOTECA, cargar_corpus, entrenar, vecinos)


def pelada(p):
    """Sin tildes y en minúscula: las listas de arriba están escritas así y el corpus no."""
    return ''.join(c for c in unicodedata.normalize('NFD', p.lower())
                   if unicodedata.category(c) != 'Mn')


def sentidos_presentes(vecinas, delatan):
    """Cuáles de los dos significados aparecen entre las vecinas. Devuelve (bool, bool)."""
    v = {pelada(x) for x in vecinas}
    return tuple(bool(v & {pelada(x) for x in lista}) for lista in delatan)


def selftest():
    """Tres pruebas sobre lo único que este programa decide: si unas vecinas dadas enseñan uno
    de los dos significados, los dos, o ninguno. No entrena nada: eso tarda minutos y lo que hay
    que comprobar aquí es el juicio, no el entrenamiento."""
    fallos = []
    delatan = PALABRAS[0]["delatan"]

    # 1. TEST NULO — cinco vecinas que no son de ninguno de los dos significados. No puede
    #    encontrar nada. Si encontrara algo, la medición de abajo contaría aciertos inventados.
    nada = sentidos_presentes(["montar", "galope", "estribo", "noche", "camino"], delatan)
    print(f"[1] test nulo         vecinas de ningún sentido: {nada} (tiene que ser (False, False))")
    if any(nada):
        fallos.append(f"test nulo: encontró un sentido donde no hay ninguno: {nada}")

    # 2. SEÑAL IMPLANTADA — las vecinas exactas que imprime el libro. Tienen que dar los dos
    #    sentidos: si no los dieran, este programa no estaría midiendo lo que dice el capítulo.
    #    Y una lista de solo muebles tiene que dar uno solo.
    libro = sentidos_presentes(["escritorio", "almacén", "baúl", "ferrocarril", "armario"], delatan)
    solo_muebles = sentidos_presentes(["escritorio", "baúl", "armario", "silla", "mesa"], delatan)
    print(f"[2] señal implantada  las del libro: {libro} (los dos); solo muebles: {solo_muebles}")
    if libro != (True, True) or solo_muebles != (True, False):
        fallos.append(f"señal implantada: las del libro tenían que dar los dos sentidos y dieron "
                      f"{libro}; solo muebles tenía que dar uno y dio {solo_muebles}")

    # 3. INVARIANTE DEL DOMINIO — las tildes no pueden cambiar el veredicto, porque el corpus
    #    las lleva y las listas de arriba no. Y ninguna palabra puede estar en las dos listas
    #    de la misma palabra, o «los dos sentidos» no querría decir nada.
    con = sentidos_presentes(["almacén"], delatan)
    sin = sentidos_presentes(["almacen"], delatan)
    solapes = [p["palabra"] for p in PALABRAS
               if {pelada(x) for x in p["delatan"][0]} & {pelada(x) for x in p["delatan"][1]}]
    print(f"[3] invariante        con tilde {con} igual que sin tilde {sin}; "
          f"palabras en las dos listas a la vez: {solapes or 'ninguna'}")
    if con != sin or solapes:
        fallos.append(f"invariante: tildes {con} contra {sin}; solapes {solapes}")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    import platform, time
    print(f"Máquina: {platform.machine()}, {platform.system()} {platform.release()}.")
    print(f"Medido el {time.strftime('%Y-%m-%d')}. Semillas: {', '.join(str(s) for s in SEMILLAS)}.")
    if selftest():
        print("\nEl selftest falla: el número no vale.")
        return 1

    frases, total = cargar_corpus(CORPUS_BIBLIOTECA)
    print(f"\nBiblioteca: {miles(total)} palabras.\n")

    salida = {p["palabra"]: [] for p in PALABRAS}
    for semilla in SEMILLAS:
        modelo = entrenar(frases, semilla=semilla)
        for p in PALABRAS:
            v = vecinos(modelo, p["palabra"], VECINOS)
            salida[p["palabra"]].append((semilla, v,
                                         sentidos_presentes(v or [], p["delatan"])))

    lineas = ["¿SALEN LOS DOS SIGNIFICADOS, O SOLO UNO?",
              f"{len(SEMILLAS)} entrenamientos desde cero; solo cambia la semilla",
              "las " + str(VECINOS) + " vecinas más próximas, en cada uno", ""]
    for p in PALABRAS:
        a, b = p["sentidos"]
        lineas.append(f"«{p['palabra']}»  —  {a} / {b}")
        for semilla, v, pres in salida[p["palabra"]]:
            marca = {(True, True): "los dos", (True, False): a,
                     (False, True): b, (False, False): "ninguno"}[pres]
            lineas.append(f"  {', '.join(v) if v else '(no aparece bastante)'}")
            lineas.append(f"      -> {marca}")
        cuantas = sum(1 for _, _, pres in salida[p["palabra"]] if all(pres))
        lineas.append(f"  los dos significados salen en {cuantas} de {len(SEMILLAS)}")
        lineas.append("")
    comprobar_ancho(lineas, 64)
    for l in lineas:
        print(l)
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
