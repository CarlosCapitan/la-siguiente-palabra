#!/usr/bin/env python3
"""¿Cuántas de cinco tiradas enseñan la avería de las palabras con dos significados?

    cuantas_veces_sale.py            la medición (entrena cinco veces: tarda)
    cuantas_veces_sale.py --selftest las tres pruebas
    cuantas_veces_sale.py --a-la-vez los cinco entrenamientos en paralelo: mismo resultado

El capítulo 5 enseña las vecinas de «banco» en UNA tirada y sostiene sobre ellas la avería que
abre el camino a los capítulos 6 y 8: que el método mete los dos significados de una palabra en el
mismo sitio. La tirada del libro sale bien. Pero entrenar no da un número, da un margen —el
capítulo 3 lo enseña con todas las letras—, y una avería demostrada con una tirada afortunada no
está demostrada.

Esto entrena cinco veces desde cero, cambiando solo la semilla, y cuenta **en cuántas de las cinco
aparece al menos una vecina de cada uno de los dos significados**. Si sale en las cinco, el
ejemplo del libro vale. Si sale en dos, el libro tiene que decirlo o cambiar de ejemplo.

Las listas de qué vecina pertenece a qué significado están abajo, con nombre, y son el único
juicio humano de este programa. Aviso (segunda vuelta de L24, 28 de septiembre): NO se
escribieron antes de mirar ninguna salida. Se escribieron el 18 de septiembre, cuando el capítulo
ya imprimía la primera tirada de «banco» (escritorio, almacén, baúl, ferrocarril, armario), y la
del dinero empieza justo por dos de esas vecinas. Por eso el capítulo 5 ya no usa este recuento:
usa `banco_segun_el_diccionario.py`, con listas sacadas del Diccionario de la lengua española.
Una vecina que no esté en ninguna de las dos listas no cuenta para ningún significado."""
SEMILLAS = [20260914, 20260915, 20260916, 20260917, 20260918]
VECINOS = 5

# Las palabras con dos significados que se ponen a prueba, y qué vecinas delatan cada significado.
# Escritas el 18 de septiembre, DESPUÉS de ver la primera tirada (ver el aviso de arriba).
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

import os, sys, unicodedata

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


_FRASES = None
_CACHE = None


def _dir_cache():
    """--cache DIR: guarda las vecinas de cada semilla en cuanto se entrena, y no vuelve a
    entrenar las que ya están. Sirve para repartir los cinco entrenamientos en varias sesiones
    cortas; el resultado es el mismo, porque lo guardado es lo que salió del entrenamiento."""
    if "--cache" not in sys.argv:
        return None
    d = sys.argv[sys.argv.index("--cache") + 1]
    os.makedirs(d, exist_ok=True)
    return d


def _leer_cache(d):
    import json
    hechas = {}
    if d:
        for s_ in SEMILLAS:
            ruta = os.path.join(d, f"{s_}.json")
            if os.path.exists(ruta):
                datos = json.load(open(ruta, encoding="utf-8"))
                assert [p["palabra"] for p in PALABRAS] == datos["palabras"], \
                    f"la caché {ruta} es de otras palabras"
                hechas[s_] = datos["vecinas"]
    return hechas


def _vecinas_de_una_semilla(semilla):
    import json
    modelo = entrenar(_FRASES, semilla=semilla)
    vs = [vecinos(modelo, p["palabra"], VECINOS) for p in PALABRAS]
    if _CACHE:
        with open(os.path.join(_CACHE, f"{semilla}.json"), "w", encoding="utf-8") as fh:
            json.dump({"palabras": [p["palabra"] for p in PALABRAS], "vecinas": vs}, fh,
                      ensure_ascii=False)
    return vs


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
    cache = _dir_cache()
    hechas = _leer_cache(cache)
    if "--a-la-vez" in sys.argv:
        # Los cinco entrenamientos a la vez, cada uno en su proceso (añadido el 28 de
        # septiembre, L24): el resultado es el mismo, porque cada entrenamiento usa un solo
        # hilo y su semilla, y no depende de en qué proceso corra; solo cambia lo que se tarda.
        # Las palabras se guardan una sola vez en memoria (sys.intern) para que quepan cinco
        # procesos en un ordenador pequeño.
        import multiprocessing
        global _FRASES
        _FRASES = [[sys.intern(w) for w in f] for f in frases]
        del frases
        global _CACHE
        _CACHE = cache
        faltan = [s_ for s_ in SEMILLAS if s_ not in hechas]
        if faltan:
            with multiprocessing.get_context("fork").Pool(min(len(faltan), 4)) as pool:
                for s_, vs in zip(faltan, pool.map(_vecinas_de_una_semilla, faltan)):
                    hechas[s_] = vs
        por_semilla = [hechas[s_] for s_ in SEMILLAS]
    else:
        por_semilla = []
        for semilla in SEMILLAS:
            modelo = entrenar(frases, semilla=semilla)
            por_semilla.append([vecinos(modelo, p["palabra"], VECINOS) for p in PALABRAS])
    for semilla, vs in zip(SEMILLAS, por_semilla):
        for p, v in zip(PALABRAS, vs):
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
