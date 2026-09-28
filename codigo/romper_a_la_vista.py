#!/usr/bin/env python3
"""
Capítulo 13 — lo que las mediciones de `romper_la_maquina.py` usaron y no enseñaban (L24, E06,
E09 y E10).

No ejecuta ningún modelo: lee lo que el programa de las mediciones lleva escrito (la frase
escondida, la pregunta, el enunciado que avisa de que fallar resta) y lo que dejó en sus salidas,
y lo pone a la vista del lector:

  1. LA AGUJA. La frase que se escondió y la pregunta que se le hizo, tal cual.
  2. EL PAJAR, A ESCALA. Las palabras del pajar más largo, contadas contra los capítulos del
     Quijote, que está en el repositorio.
  3. EL ENUNCIADO QUE AVISA. Las dos líneas que se pusieron delante de las veinte preguntas, y la
     tabla de abstenciones con un rótulo del libro en lugar de «con umbral explícito».
  4. EL EXAMEN TIPO TEST. La cuenta que el capítulo anuncia («el argumento es de contabilidad»):
     cuántos puntos da, de media, contestar a ciegas una pregunta de cuatro opciones, con la regla
     de casi todas las pruebas (acierto 1, fallo 0, en blanco 0) y con la del enunciado que avisa
     (la que imprime el apartado 3; el programa la lee de ahí).

Uso:
    python romper_a_la_vista.py --selftest
    python romper_a_la_vista.py > ../datos/salidas/romper_a_la_vista.txt
"""

# ======================= CONSTANTES =======================

PROGRAMA = "romper_la_maquina.py"                       # de donde se leen la aguja y el enunciado
SALIDA_CSV = "../datos/salidas/romper_la_maquina.csv"
RESPUESTAS = "../datos/salidas/romper_la_maquina_respuestas.txt"
QUIJOTE = "../datos/quijote.txt"
FIN_QUIJOTE = "*** END"                                  # donde acaba el texto y empieza el aviso legal
CAPITULO = r"(?m)^Capítulo "                             # cómo empieza cada capítulo en el fichero
CAPITULOS_QUIJOTE = 126                                  # 52 de la primera parte y 74 de la segunda
OPCIONES = 4                                             # un examen tipo test de cuatro respuestas
PREGUNTAS_EXAMEN = 4                                     # la cuenta se hace sobre cuatro preguntas
ROTULOS = {                                              # los del libro, en lugar de los del programa
    "enunciado normal": "enunciado normal",
    "con umbral explícito": "avisando de que fallar resta",
    "control con respuesta": "con respuesta, avisando",
}

# ==========================================================

import argparse
import ast
import csv
import re
import sys
from pathlib import Path

from formato import ANCHO_CAJA_CITA, coma, comprobar_ancho, miles, pct

AQUI = Path(__file__).resolve().parent


def constantes(ruta=AQUI / PROGRAMA):
    """Las constantes de arriba del programa, leídas sin ejecutarlo (importarlo cargaría la
    biblioteca de la tarjeta gráfica, que aquí no hace falta)."""
    arbol = ast.parse(Path(ruta).read_text(encoding="utf-8"))
    out = {}
    for n in arbol.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    return out


def partido(texto, ancho=ANCHO_CAJA_CITA, sangria="  "):
    """Un texto largo, partido por palabras para que quepa en la caja de cita."""
    lineas = []
    for parrafo in texto.split("\n"):
        linea = sangria
        for p in parrafo.split():
            if len(linea) + len(p) + (0 if linea.strip() == "" else 1) > ancho:
                lineas.append(linea.rstrip())
                linea = sangria
            linea += ("" if linea.strip() == "" else " ") + p
        lineas.append(linea.rstrip())
    return comprobar_ancho(lineas, ancho)


def leer_csv(ruta=AQUI / SALIDA_CSV):
    with open(ruta, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def palabras_por_capitulo(ruta=AQUI / QUIJOTE):
    t = Path(ruta).read_text(encoding="utf-8")
    inicios = [m.start() for m in re.finditer(CAPITULO, t)]
    fin = t.find(FIN_QUIJOTE)
    assert len(inicios) == CAPITULOS_QUIJOTE, \
        f"se esperaban {CAPITULOS_QUIJOTE} capítulos en el Quijote; hay {len(inicios)}"
    assert fin > inicios[-1], "se esperaba el aviso final del Proyecto Gutenberg tras el último capítulo"
    palabras = len(t[inicios[0]:fin].split())
    return palabras, palabras / len(inicios)


def regla_del_enunciado(enunciado):
    """Lo que vale acertar, fallar y no contestar, leído del propio enunciado."""
    resta = re.search(r"equivocada resta (\d+) punto", enunciado)
    suma = re.search(r"correcta suma (\d+) punto", enunciado)
    cero = re.search(r"no lo sabes vale (\d+) punto", enunciado)
    assert resta and suma and cero, "no encuentro en el enunciado lo que vale cada cosa"
    return int(suma.group(1)), -int(resta.group(1)), int(cero.group(1))


def a_ciegas(regla, opciones=OPCIONES, n=PREGUNTAS_EXAMEN):
    """Puntos de media en n preguntas contestadas a ciegas: de cada `opciones` se acierta una."""
    acierto, fallo, _ = regla
    aciertos = n / opciones
    return aciertos, n - aciertos, aciertos * acierto + (n - aciertos) * fallo


def cuantas_abstenciones(ruta=AQUI / RESPUESTAS):
    """Por apartado del fichero de respuestas: (cuántas «[se abstiene]», cuántas preguntas)."""
    t = Path(ruta).read_text(encoding="utf-8")
    partes = re.split(r"(?m)^=== (.+?) ===$", t)
    out = {}
    for i in range(1, len(partes), 2):
        clave = "control con respuesta" if partes[i].startswith("control") else partes[i]
        out[clave] = (partes[i + 1].count("[se abstiene]"), len(re.findall(r"(?m)^\[", partes[i + 1])))
    return out


def informe(C, filas, por_cap, n_pajar):
    print("--- 1. LA AGUJA Y LA PREGUNTA ---")
    print("la frase que se escondió en el texto:")
    for l in partido(C["AGUJA"]):
        print(l)
    print("la pregunta que se le hizo después del texto:")
    for l in partido(C["PREGUNTA_AGUJA"]):
        print(l)

    print("\n--- 2. EL PAJAR MÁS LARGO, A ESCALA ---")
    palabras = int(next(f["a"] for f in filas
                        if f["medicion"] == "C_aguja" and f["clave"] == "palabras"))
    lineas = [f"el pajar más largo: {miles(n_pajar)} trozos, {miles(palabras)} palabras",
              f"un capítulo del Quijote, de media: {miles(round(por_cap))} palabras",
              f"el pajar más largo, en capítulos del Quijote: {coma(palabras / por_cap, 1)}"]
    for l in comprobar_ancho(["  " + l for l in lineas], ANCHO_CAJA_CITA):
        print(l)

    print("\n--- 3. EL ENUNCIADO QUE AVISA DE QUE FALLAR RESTA ---")
    print("lo que se puso delante de cada una de las veinte preguntas:")
    for l in partido(C["ENUNCIADO_KALAI"].replace("{p}", "").strip()):
        print(l)
    print()
    cuenta = cuantas_abstenciones()
    tabla = []
    for f in filas:
        if f["medicion"] != "D_abstencion":
            continue
        total = cuenta[f["clave"]][1]
        k = round(float(f["a"]) * total)
        assert k == cuenta[f["clave"]][0], f"«{f['clave']}»: el CSV y las respuestas no casan"
        tabla.append(f"  {ROTULOS[f['clave']]:<30}se calla en {k} de {total} ({pct(k / total, 0)})")
    tabla += ["",
              "  «se calla»: contesta que no lo sabe, o que no existe.",
              "  las dos primeras filas: las veinte preguntas sin respuesta.",
              "  la tercera: cinco que sí la tienen, del capítulo 11."]
    for l in comprobar_ancho(tabla, ANCHO_CAJA_CITA):
        print(l)

    print("\n--- 4. EL EXAMEN TIPO TEST, CONTESTADO A CIEGAS ---")
    normal = (1, 0, 0)
    aviso = regla_del_enunciado(C["ENUNCIADO_KALAI"])
    n, o = PREGUNTAS_EXAMEN, OPCIONES
    lineas = [f"{n} preguntas de {o} respuestas cada una; a ciegas se acierta",
              f"1 de cada {o}: en las {n}, {n // o} acierto y {n - n // o} fallos.",
              ""]
    for nombre, r in (("con la regla de casi todas las pruebas:", normal),
                      ("con la regla del enunciado que avisa:", aviso)):
        acierto, fallo, blanco = r
        _, _, pts = a_ciegas(r)
        lineas += [nombre,
                   f"  un acierto suma {acierto}; un fallo "
                   + (f"resta {-fallo}" if fallo else "no resta") + f"; en blanco, {blanco}.",
                   f"  a ciegas, en las {n}: " + (f"gana {pts:.0f} punto" + ("s" if pts != 1 else "")
                                                   if pts >= 0 else f"pierde {-pts:.0f} puntos") + "."]
    lineas += [f"en blanco, las {n} dan 0 puntos con las dos reglas."]
    for l in comprobar_ancho(["  " + l if l else l for l in lineas], ANCHO_CAJA_CITA):
        print(l)
    return aviso


def selftest():
    fallos = []
    C = constantes()
    filas = leer_csv()

    # 1. TEST NULO — con la regla de casi todas las pruebas (fallar no resta), contestar a ciegas
    #    nunca da menos que dejarlo en blanco, con cualquier número de opciones.
    nunca_pierde = all(a_ciegas((1, 0, 0), o)[2] >= 0 for o in (2, 3, 4, 5, 10))
    print(f"[1] test nulo         sin castigo, contestar a ciegas da menos que en blanco: "
          f"{'nunca' if nunca_pierde else 'A VECES'}")
    if not nunca_pierde:
        fallos.append("test nulo: sin castigo, contestar a ciegas sale peor que en blanco")

    # 2. SEÑAL IMPLANTADA — la frase escondida lleva la respuesta que se busca, y la regla del
    #    enunciado que avisa castiga el fallo: a ciegas, pierde puntos.
    lleva = C["RESPUESTA_AGUJA"] in C["AGUJA"] and C["RESPUESTA_AGUJA"] not in C["PREGUNTA_AGUJA"]
    aviso = regla_del_enunciado(C["ENUNCIADO_KALAI"])
    pierde = a_ciegas(aviso)[2] < 0
    print(f"[2] señal implantada  la frase lleva la respuesta y la pregunta no: "
          f"{'sí' if lleva else 'NO'}; con la regla {aviso}, a ciegas pierde: "
          f"{'sí' if pierde else 'NO'}")
    if not lleva:
        fallos.append("señal: la frase escondida no lleva la respuesta, o la pregunta la regala")
    if not pierde:
        fallos.append("señal: la regla del enunciado no castiga contestar a ciegas")

    # 3. INVARIANTE DEL DOMINIO — las abstenciones de la tabla (del CSV) son las que se cuentan
    #    una a una en el fichero de respuestas, y el Quijote tiene sus 126 capítulos.
    cuenta = cuantas_abstenciones()
    del_csv = {f["clave"]: float(f["a"]) for f in filas if f["medicion"] == "D_abstencion"}
    n = len(C["SIN_RESPUESTA"])
    casan = all(cuenta[k][0] == round(del_csv[k] * cuenta[k][1]) for k in del_csv) \
        and cuenta["enunciado normal"][1] == n
    _, por_cap = palabras_por_capitulo()
    print(f"[3] invariante        abstenciones contadas una a una: {cuenta}; casan con la tabla: "
          f"{'sí' if casan else 'NO'}; palabras por capítulo del Quijote: {por_cap:.0f}")
    if not casan:
        fallos.append("invariante: la tabla y las respuestas una a una no cuentan lo mismo")
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
    if ap.parse_args().selftest:
        sys.exit(selftest())
    C = constantes()
    filas = leer_csv()
    _, por_cap = palabras_por_capitulo()
    informe(C, filas, por_cap, C["LONGITUDES"][-1])


if __name__ == "__main__":
    main()
