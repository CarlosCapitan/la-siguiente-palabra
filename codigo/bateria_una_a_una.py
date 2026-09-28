#!/usr/bin/env python3
"""
Capítulo 11 — la batería de treinta preguntas, pregunta a pregunta (L24, hallazgos D10, D11,
D12, D13 y D15).

`crecer.py` imprime la tabla del capítulo: cuántas preguntas de cada tarea acierta cada tamaño.
Lo que no imprime es lo que el lector necesita para leerla sin imaginar nada: qué se le
escribió a la máquina, qué contestó y por qué eso cuenta como acierto o como fallo. Este
programa vuelve a pasar la MISMA batería, con los MISMOS modelos y la MISMA regla (importa
`crecer.py`, no lo copia), guarda todas las respuestas y comprueba que salen exactamente los
aciertos de `crecer.csv`. Después imprime:

  1. la tabla del capítulo contada en preguntas («2 de 5») en vez de en porcentajes;
  2. una pregunta de cada tarea tal como la recibe la máquina, con lo que contesta cada tamaño;
  3. cómo le llega a la máquina, en trozos, el patrón inventado (la palabra al revés);
  4. la misma tabla puntuada con otra regla, por letras, para ver que la forma de una fila
     depende de cómo se corrige;
  5. cuántos de los números de cada modelo son la tabla de la primera estación (la lista de
     números de cada trozo), que es la parte que el estudio de 2020 dejó fuera de su recuento.

No toca `crecer.py` ni su salida, que el libro ya cita.

Uso:
    python bateria_una_a_una.py --selftest
    python bateria_una_a_una.py
"""

# ======================= CONSTANTES =======================

CSV_TABLA = "../datos/salidas/crecer.csv"            # lo que tiene que salir, pregunta a pregunta
SALIDA_TODAS = "../datos/salidas/bateria_una_a_una_todas.txt"
TAREA_PATRON = "seguir un patrón inventado"
PALABRAS_TABLA = ["libro", "sol"]    # de las cinco del patrón, las que enseña la tabla del troceado
ANCHO_RESPUESTA = 30         # caracteres de la respuesta que caben en la columna
ANCHO_ENUNCIADO = 54         # renglón del enunciado, antes de partirlo
TOLERANCIA = 1e-9

# ==========================================================

import argparse
import csv
import datetime
import platform
import sys
import textwrap

import torch

import crecer
from crecer import ETIQUETA_TAMANO, MODELOS, TAREAS, TAREA_CONTROL, acierta, normalizar
from formato import ANCHO_CAJA_CITA, comprobar_ancho, miles, pct


def primera_linea(respuesta):
    """Lo que la regla corrige: el primer renglón de lo que escribe (crecer.acierta)."""
    return respuesta.split("\n")[0]


def por_letras(respuesta, esperada):
    """Otra regla, para comparar: qué parte de la palabra correcta escribe bien, letra a letra y
    desde el principio, antes de equivocarse. «libros» para «libros» vale 1; «lib» vale 0,5."""
    r, e = normalizar(primera_linea(respuesta)), normalizar(esperada)
    n = 0
    while n < min(len(r), len(e)) and r[n] == e[n]:
        n += 1
    return n / len(e)


def responder_todo(tok, modelo):
    """Todas las respuestas del modelo a la batería, en el orden de TAREAS."""
    return {t: [crecer.continuar(tok, modelo, p) for p, _ in items] for t, items in TAREAS.items()}


def leer_tabla(ruta=CSV_TABLA):
    with open(ruta, newline="", encoding="utf-8") as fh:
        filas = list(csv.reader(fh))
    assert filas[0] == ["modelo", "numeros", "tarea", "acierto"], f"cabecera inesperada en {ruta}"
    return {(m, t): float(a) for m, _, t, a in filas[1:]}


def comprobar_contra_tabla(nombre, respuestas, tabla):
    for t, items in TAREAS.items():
        a = sum(acierta(r, e) for r, (_, e) in zip(respuestas[t], items)) / len(items)
        assert abs(a - tabla[(nombre, t)]) < TOLERANCIA, (
            f"Se esperaba reproducir crecer.csv: {nombre}, «{t}» da {a:.3f} y la tabla "
            f"dice {tabla[(nombre, t)]:.3f}")


def trozos(tok, texto):
    return [tok.decode([i]) for i in tok(texto)["input_ids"]]


def visible(t):
    return t.replace(" ", "_")


def enunciado_en_renglones(p):
    """El enunciado tal cual, renglón a renglón; los largos, partidos para que quepan."""
    lin = []
    for r in p.split("\n"):
        partes = textwrap.wrap(r, ANCHO_ENUNCIADO) or [""]
        lin.append("    " + partes[0])
        lin += ["      " + x for x in partes[1:]]
    return lin


def recorte(s, n=ANCHO_RESPUESTA):
    s = s.replace("\t", " ")
    return s if len(s) <= n else s[:n - 1] + "…"


def elegida(t, veredictos):
    """La pregunta que se enseña de cada tarea: la primera en que no aciertan todos los tamaños
    por igual; si en todas aciertan o fallan todos a la vez, la primera."""
    for i in range(len(TAREAS[t])):
        if len({veredictos[m][t][i] for m in veredictos}) > 1:
            return i
    return 0


def bloque_tabla(usados, resp):
    etiquetas = [ETIQUETA_TAMANO[m] for m in usados]
    lin = ["--- 1. LA TABLA DEL CAPÍTULO, EN PREGUNTAS ---", "",
           f"{'tarea':<26}" + "".join(f"{e:>9}" for e in etiquetas)]
    totales = {m: 0 for m in usados}
    n_total = 0
    for t, items in TAREAS.items():
        fila = f"{t:<26}"
        for m in usados:
            a = sum(acierta(r, e) for r, (_, e) in zip(resp[m][t], items))
            totales[m] += a
            fila += f"{f'{a} de {len(items)}':>9}"
        n_total += len(items)
        lin.append(fila)
    lin.append(f"{'en total':<26}" + "".join(f"{f'{totales[m]} de {n_total}':>9}" for m in usados))
    lin += ["",
            f"«{etiquetas[0]}» quiere decir {etiquetas[0][:-1]} millones de números",
            "(pesos). Es el nombre redondo con que se publicó cada",
            "modelo; el recuento exacto va debajo.",
            "cada casilla: cuántas de las 5 preguntas de la tarea acierta.",
            "acierta: si lo primero que escribe es la respuesta correcta."]
    return comprobar_ancho(lin, ANCHO_CAJA_CITA)


def bloque_ejemplos(usados, resp):
    lin = ["--- 2. UNA PREGUNTA DE CADA TAREA, Y LO QUE CONTESTA ---"]
    ver = {m: {t: [acierta(r, e) for r, (_, e) in zip(resp[m][t], TAREAS[t])] for t in TAREAS}
           for m in usados}
    for t, items in TAREAS.items():
        i = elegida(t, ver)
        p, e = items[i]
        lin += ["", f"[{t}]", "lo que recibe la máquina:"] + enunciado_en_renglones(p)
        lin.append(f"respuesta correcta: «{e}»")
        for m in usados:
            r = primera_linea(resp[m][t][i]).strip()
            nota = "acierta" if ver[m][t][i] else "falla"
            lin.append(f"  {ETIQUETA_TAMANO[m]:>6}  {'«' + recorte(r) + '»':<34}{nota:>8}")
    lin += ["", "a la derecha de cada tamaño, el primer renglón de lo que",
            "escribe: es lo único que mira la regla."]
    return comprobar_ancho(lin, ANCHO_CAJA_CITA)


def bloque_troceado(tok, usados, resp):
    items = TAREAS[TAREA_PATRON]
    p, e = items[0]
    lin = ["--- 3. CÓMO LE LLEGA EL PATRÓN INVENTADO ---",
           "el primer enunciado, en trozos («|» separa dos trozos;",
           "«_», un espacio):"]
    t = trozos(tok, p)
    renglon = []
    for x in t:
        assert "\n" not in x or x == "\n", f"se esperaba el salto de línea como trozo aparte: {x!r}"
        if x == "\n":
            lin.append("    " + "|".join(visible(y) for y in renglon))
            renglon = []
        else:
            renglon.append(x)
    lin.append("    " + "|".join(visible(y) for y in renglon))
    lin += ["cada renglón acaba en un trozo más, el salto de línea.", "",
            f"{'palabra':<10}{'en trozos':<14}{'al revés':<10}{'en trozos':<14}"]
    for pal, rev in [("casa", "asac"), ("mesa", "asem")] + \
            [(pp.split("\n")[-1].split(" ")[0], ee) for pp, ee in items
             if pp.split("\n")[-1].split(" ")[0] in PALABRAS_TABLA]:
        assert pal[::-1] == rev, f"se esperaba que «{rev}» fuera «{pal}» al revés"
        lin.append(f"{pal:<10}{'|'.join(trozos(tok, pal)):<14}{rev:<10}{'|'.join(trozos(tok, rev)):<14}")
    # En el enunciado, lo que va detrás de «->» lleva un espacio delante, y el espacio cambia el
    # troceado: «asem» sola es «as|em», y « asem» es «_a|sem». Se dice, para que el lector no vea
    # dos troceados de la misma palabra sin saber por qué.
    for rev in ("asac", "asem"):
        solo, con = trozos(tok, rev), trozos(tok, " " + rev)
        if [visible(x) for x in con] != ["_" + solo[0]] + solo[1:]:
            lin.append(f"«{rev}» con el espacio de delante, como va en el enunciado,")
            lin.append(f"  se parte de otra manera: {'|'.join(visible(x) for x in con)}")
    lin += ["", f"lo que contesta cada tamaño a «{p.split(chr(10))[-1].strip()}»:"]
    for m in usados:
        lin.append(f"  {ETIQUETA_TAMANO[m]:>6}  «{recorte(primera_linea(resp[m][TAREA_PATRON][0]).strip())}»")
    return comprobar_ancho([l.rstrip() for l in lin], ANCHO_CAJA_CITA)


def bloque_por_letras(usados, resp):
    etiquetas = [ETIQUETA_TAMANO[m] for m in usados]
    lin = ["--- 4. LAS MISMAS RESPUESTAS, CORREGIDAS POR LETRAS ---", "",
           f"{'tarea':<26}" + "".join(f"{e:>9}" for e in etiquetas)]
    for t, items in TAREAS.items():
        fila = f"{t:<26}"
        for m in usados:
            v = sum(por_letras(r, e) for r, (_, e) in zip(resp[m][t], items)) / len(items)
            fila += f"{pct(v, 0):>9}"
        lin.append(fila)
    lin += ["",
            "cada casilla: de las letras de la respuesta correcta, qué",
            "parte escribe bien y en su sitio, desde la primera, antes",
            "de equivocarse; en promedio de las 5 preguntas. «libros»",
            "para «libros» vale 100 %; «lib», 50 %. Las mayúsculas no",
            "cuentan: a «¿cuál es la capital de Italia?», «rome» por",
            "«Roma» vale 75 %."]
    return comprobar_ancho(lin, ANCHO_CAJA_CITA)


def bloque_primera_estacion(usados, cuentas):
    lin = ["--- 5. LAS LISTAS DE LA SEGUNDA ESTACIÓN ---",
           "la segunda estación del capítulo 7 cambia cada trozo por su",
           "lista de números. Cuántos números suman todas esas listas,",
           "y qué parte son de todos los números de la máquina:", "",
           f"{'tamaño':>6}{'números en total':>19}{'en las listas':>16}{'parte':>9}"]
    for m in usados:
        total, tabla = cuentas[m]
        lin.append(f"{ETIQUETA_TAMANO[m]:>6}{miles(total):>19}{miles(tabla):>16}{pct(tabla / total, 1):>9}")
    return comprobar_ancho(lin, ANCHO_CAJA_CITA)


def selftest():
    fallos = []
    nombre = MODELOS[0]
    tok, modelo = crecer.cargar(nombre)
    resp = responder_todo(tok, modelo)

    # 1. TEST NULO — corregir cada respuesta contra la respuesta de OTRA pregunta de la misma
    #    tarea (girando la lista una posición): la regla no puede regalar aciertos.
    girado = {t: [e for _, e in items[1:] + items[:1]] for t, items in TAREAS.items()}
    nulo = sum(acierta(r, e) for t in TAREAS for r, e in zip(resp[t], girado[t]))
    real = sum(acierta(r, e) for t, items in TAREAS.items() for r, (_, e) in zip(resp[t], items))
    print(f"[1] test nulo         aciertos con las respuestas giradas {nulo}, con las de verdad {real}")
    if nulo >= real or nulo > 2:
        fallos.append(f"test nulo: con las respuestas giradas acierta {nulo}")

    # 2. SEÑAL IMPLANTADA — copiar una palabra que está delante: la regla la da por buena y
    #    la regla por letras le da la nota entera.
    ok = [acierta(crecer.continuar(tok, modelo, p), e) for p, e in TAREA_CONTROL]
    letras = [por_letras(crecer.continuar(tok, modelo, p), e) for p, e in TAREA_CONTROL]
    print(f"[2] señal implantada  copiar la palabra anterior: {sum(ok)} de {len(ok)}; por letras "
          f"{pct(sum(letras) / len(letras), 0)}")
    if not all(ok) or min(letras) < 1:
        fallos.append("señal implantada: copiar una palabra no sale como acierto")

    # 3. INVARIANTE — las respuestas de aquí dan exactamente los aciertos de crecer.csv.
    try:
        comprobar_contra_tabla(nombre, resp, leer_tabla())
        igual = True
    except AssertionError as e:
        igual = False
        fallos.append(f"invariante: {e}")
    print(f"[3] invariante        {ETIQUETA_TAMANO[nombre]} reproduce crecer.csv: {'sí' if igual else 'NO'}")
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

    print(f"Medido el {datetime.date.today()} en {platform.platform()}; generación determinista "
          "(do_sample=False) en procesador, como crecer.py.")
    tabla = leer_tabla()
    resp, cuentas, usados, tok_patron = {}, {}, [], None
    for nombre in MODELOS:
        tok, modelo = crecer.cargar(nombre)
        resp[nombre] = responder_todo(tok, modelo)
        comprobar_contra_tabla(nombre, resp[nombre], tabla)
        cuentas[nombre] = (sum(p.numel() for p in modelo.parameters()),
                           modelo.get_input_embeddings().weight.numel())
        usados.append(nombre)
        if tok_patron is None:
            tok_patron = tok
        else:
            assert trozos(tok, TAREAS[TAREA_PATRON][0][0]) == trozos(tok_patron, TAREAS[TAREA_PATRON][0][0]), \
                "Se esperaba el mismo troceador en los cuatro tamaños"
        del modelo
    print("Las respuestas reproducen crecer.csv en los cuatro tamaños.\n")

    for bloque in (bloque_tabla(usados, resp), bloque_ejemplos(usados, resp),
                   bloque_troceado(tok_patron, usados, resp), bloque_por_letras(usados, resp),
                   bloque_primera_estacion(usados, cuentas)):
        print("\n".join(bloque))
        print()
    print("números de cada modelo (sus pesos):")
    for m in usados:
        print(f"  {ETIQUETA_TAMANO[m]}: {miles(cuentas[m][0])}")

    with open(SALIDA_TODAS, "w", encoding="utf-8") as fh:
        for t, items in TAREAS.items():
            for i, (p, e) in enumerate(items):
                fh.write(f"[{t}, pregunta {i + 1}] correcta: {e!r}\n{p!r}\n")
                for m in usados:
                    r = resp[m][t][i]
                    fh.write(f"  {ETIQUETA_TAMANO[m]:>6} {'acierta' if acierta(r, e) else 'falla  '} {r!r}\n")
                fh.write("\n")
    print(f"\nTodas las respuestas, en {SALIDA_TODAS}")


if __name__ == "__main__":
    main()
