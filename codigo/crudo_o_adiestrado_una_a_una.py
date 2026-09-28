#!/usr/bin/env python3
"""
Capítulo 12 — crudo o adiestrado, pregunta a pregunta (L24, hallazgos D17, D18 y D21).

`crudo_o_adiestrado.py` imprime lo que cita el capítulo: las dos listas de «La capital de
Francia es», la pregunta directa y la batería en tres columnas. Faltan tres cosas para que el
lector no tenga que imaginar nada, y las mide este programa con los MISMOS modelos, las MISMAS
funciones (las importa, no las copia) y la MISMA regla:

  1. París en las dos listas: cuánto tiene la palabra entera, en crudo y adiestrado, y qué es
     la fila «___».
  2. Nápoles: qué lista tiene la máquina pequeña justo antes de escribir «Nápoles».
  3. Una pregunta de la batería tal como la recibe la máquina en cada columna (con su formato
     de conversación entero), lo que contesta y si la regla la aprueba; y la tabla de las tres
     columnas contada en preguntas, comprobada contra la del capítulo.

No toca `crudo_o_adiestrado.py` ni su salida.

Uso:
    python crudo_o_adiestrado_una_a_una.py --selftest
    python crudo_o_adiestrado_una_a_una.py
"""

# ======================= CONSTANTES =======================

CSV_TABLA = "crudo_o_adiestrado.csv"      # donde lo deja crudo_o_adiestrado.py
PALABRAS = [" París", " Paris"]           # con tilde (dos trozos) y sin tilde (uno)
BUSCADA_NAPOLES = "Nápoles"
EJEMPLOS = [("saber cosas del mundo", 0), ("sumar dos cifras", 0)]   # (tarea, pregunta)
ANCHO_RENGLON = 54
ANCHO_RESPUESTA = 34
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
import crudo_o_adiestrado as cap12
from crecer import ETIQUETA_TAMANO, TAREAS, TAREA_CONTROL, acierta
from formato import ANCHO_CAJA_CITA, comprobar_ancho, miles, pct, tabla_de_probabilidades


def lista(modelo, ids):
    with torch.no_grad():
        logits = modelo(torch.tensor([ids])).logits[0, -1]
    p = torch.softmax(logits.float(), dim=-1)
    assert abs(float(p.sum()) - 1.0) < 1e-3, f"la lista no suma uno: {float(p.sum())}"
    return p


def puesto(p, i):
    return int((p > p[i]).sum()) + 1


def prob_palabra(tok, modelo, ids, palabra):
    """La palabra entera detrás de `ids`: la del primer trozo por la del segundo, etc."""
    total, primero = 1.0, None
    for i in tok(palabra)["input_ids"]:
        p = lista(modelo, ids)
        total *= float(p[i])
        if primero is None:
            primero = (tok.decode([i]), puesto(p, i))
        ids = ids + [i]
    return total, primero


def visible(t):
    return t.replace(" ", "_")


def clave_de_rayas(pares):
    """Una línea por cada trozo de la lista que lleva rayas bajas de verdad: sin ella, el
    lector lee «___» como tres espacios, porque «_» es como el libro dibuja un espacio."""
    lin = []
    for w, _ in pares:
        n = w.count("_")
        if n:
            ante = "un espacio y " if w.startswith(" ") else ""
            lin.append(f"«{visible(w)}»: {ante}{n} raya{'s' if n > 1 else ''} baja{'s' if n > 1 else ''}, «{'_' * n}».")
    return lin


def lineas_paris(titulo, tok, mod):
    """La lista de «La capital de Francia es» de un modelo, con París entera debajo."""
    ids = tok(cap12.FRASE_CAP7)["input_ids"]
    p = lista(mod, ids)
    top = torch.topk(p, cap12.TOP_N)
    pares = [(tok.decode([int(i)]), float(v)) for v, i in zip(top.values, top.indices)]
    lin = ["", f"{titulo}:"] + tabla_de_probabilidades([(visible(w), v) for w, v in pares], decimales=1)
    suma = 0.0
    for pal in PALABRAS:
        total, (t1, pu) = prob_palabra(tok, mod, ids, pal)
        n = len(tok(pal)["input_ids"])
        suma += total
        lin.append(f"«{pal.strip()}» entera ({n} trozo{'s' if n > 1 else ''}): {pct(total, 2)}; "
                   f"«{visible(t1)}», puesto {miles(pu)}")
    lin.append(f"las dos maneras juntas: {pct(suma, 1)}")
    return lin + clave_de_rayas(pares)


def bloque_paris(partes):
    lin = [f"--- 1. «{cap12.FRASE_CAP7}»: DÓNDE ESTÁ PARÍS ---"]
    for x in partes:
        lin += x
    lin += ["", "«_» delante de un trozo es un espacio. El puesto es el del",
            "primer trozo en la lista entera, de más a menos probable."]
    return comprobar_ancho(lin, ANCHO_CAJA_CITA)


def bloque_napoles(tok, mod, etiqueta):
    """Genera la pregunta directa y se para justo antes del trozo que empieza la ciudad."""
    ids = tok(cap12.PREGUNTA)["input_ids"]
    with torch.no_grad():
        salida = mod.generate(torch.tensor([ids]), max_new_tokens=cap12.MAX_NUEVOS,
                              do_sample=False, pad_token_id=tok.eos_token_id)
    nuevos = salida[0, len(ids):].tolist()
    texto_completo = tok.decode(nuevos, skip_special_tokens=True)
    corte = None
    for k in range(1, len(nuevos)):
        if tok.decode(nuevos[:k]).strip().endswith("Francia es"):
            corte = k
            break
    assert corte is not None, f"no se encontró la ciudad en {texto_completo!r}"
    p = lista(mod, ids + nuevos[:corte])
    top = torch.topk(p, cap12.TOP_N)
    pares = [(tok.decode([int(i)]), float(v)) for v, i in zip(top.values, top.indices)]
    lin = [f"--- 2. {etiqueta}, EN CRUDO, JUSTO ANTES DE LA CIUDAD ---",
           f"pregunta: «{cap12.PREGUNTA}»",
           f"lo que lleva escrito: «{tok.decode(nuevos[:corte]).strip()}»",
           "lo que puede venir ahora:"]
    lin += tabla_de_probabilidades([(visible(w), v) for w, v in pares], decimales=1)
    for pal in (" París", " Nápoles", " Roma"):
        total, (t1, pu) = prob_palabra(tok, mod, ids + nuevos[:corte], pal)
        lin.append(f"«{pal.strip()}» entera: {pct(total, 2)}; «{visible(t1)}», puesto {miles(pu)}")
    lin.append(f"lo que escribe entero: «{texto_completo.strip()}»")
    return comprobar_ancho(lin, ANCHO_CAJA_CITA)


def responder(tok, mod, con_formato):
    """Las respuestas a la batería, exactamente como las da crudo_o_adiestrado.py."""
    if con_formato:
        return {t: [cap12.continuar(tok, mod, cap12.con_formato(tok, p), 12) for p, _ in items]
                for t, items in TAREAS.items()}
    return {t: [crecer.continuar(tok, mod, p) for p, _ in items] for t, items in TAREAS.items()}


def leer_tabla(ruta=CSV_TABLA):
    with open(ruta, newline="", encoding="utf-8") as fh:
        filas = list(csv.reader(fh))
    return {(f[0], f[2]): [float(x) for x in f[3:6]] for f in filas[1:] if f[1] == "bateria"}


def aciertos(resp):
    return {t: sum(acierta(r, e) for r, (_, e) in zip(resp[t], items)) for t, items in TAREAS.items()}


def comprobar(etiqueta, columnas, tabla):
    for j, resp in enumerate(columnas):
        for t, a in aciertos(resp).items():
            esperado = tabla[(etiqueta, t)][j]
            assert abs(a / len(TAREAS[t]) - esperado) < TOLERANCIA, (
                f"Se esperaba reproducir {CSV_TABLA}: {etiqueta}, «{t}», columna {j + 1}: "
                f"{a} de {len(TAREAS[t])} frente a {esperado:.3f}")


def bloque_tabla(etiqueta, columnas):
    lin = [f"--- 3. LA BATERÍA EN TRES COLUMNAS, EN PREGUNTAS ({etiqueta}) ---", "",
           f"{'tarea':<26}{'crudo':>9}{'adiestrado':>12}{'con formato':>13}"]
    tot = [0, 0, 0]
    for t in TAREAS:
        fila = f"{t:<26}"
        for j, (resp, w) in enumerate(zip(columnas, (9, 12, 13))):
            a = aciertos(resp)[t]
            tot[j] += a
            fila += f"{f'{a} de {len(TAREAS[t])}':>{w}}"
        lin.append(fila)
    n = sum(len(v) for v in TAREAS.values())
    lin.append(f"{'en total':<26}" + "".join(f"{f'{a} de {n}':>{w}}" for a, w in zip(tot, (9, 12, 13))))
    lin += ["",
            "cada casilla: cuántas de las 5 preguntas de la tarea acierta.",
            f"las tres columnas, el mismo tamaño: {etiqueta}.",
            "crudo: sin adiestrar, con el enunciado tal cual.",
            "adiestrado: el mismo ya adiestrado, enunciado tal cual.",
            "con formato: el adiestrado, con su formato de conversación.",
            "acierta: si lo primero que escribe es la respuesta correcta."]
    return comprobar_ancho(lin, ANCHO_CAJA_CITA)


def renglones(texto, sangria="    "):
    lin = []
    for r in texto.split("\n"):
        partes = textwrap.wrap(r, ANCHO_RENGLON) or [""]
        lin.append(sangria + partes[0])
        lin += [sangria + "  " + x for x in partes[1:]]
    return lin


def bloque_tres_maneras(tok_b, tok_i, columnas, etiqueta):
    lin = [f"--- 4. LA MISMA PREGUNTA EN LAS TRES COLUMNAS ({etiqueta}) ---"]
    for tarea, i in EJEMPLOS:
        p, e = TAREAS[tarea][i]
        lin += ["", f"[{tarea}, pregunta {i + 1}]", f"respuesta correcta: «{e}»"]
        for j, (titulo, texto) in enumerate((
                ("crudo: recibe el enunciado tal cual", p),
                ("adiestrado: recibe el mismo enunciado tal cual", p),
                ("con formato: recibe esto", cap12.con_formato(tok_i, p)))):
            if j == 1:
                lin.append(f"{titulo}")
            else:
                lin.append(f"{titulo}:")
                lin += renglones(texto.rstrip("\n"))
            r = columnas[j][tarea][i]
            primera = r.split("\n")[0].strip()
            if len(primera) > ANCHO_RESPUESTA:
                primera = primera[:ANCHO_RESPUESTA - 1] + "…"
            nota = "acierta" if acierta(r, e) else "falla"
            lin.append(f"  contesta «{primera}»: {nota}")
    lin += ["", "La regla mira el primer renglón de lo que contesta y",
            "acierta si empieza por la respuesta correcta."]
    return comprobar_ancho(lin, ANCHO_CAJA_CITA)


def cargar_pareja(base, instruido):
    tok_b, mod_b = cap12.cargar(base)
    tok_i, mod_i = cap12.cargar(instruido)
    return tok_b, mod_b, tok_i, mod_i


def selftest():
    fallos = []
    base, instruido = cap12.PAREJAS[0]
    tok_b, mod_b, tok_i, mod_i = cargar_pareja(base, instruido)

    # 1. TEST NULO — una palabra que no tiene nada que ver con la frase tiene una probabilidad
    #    minúscula, y la suma de las dos Parises no pasa del cien por cien.
    ids = tok_b(cap12.FRASE_CAP7)["input_ids"]
    nada, _ = prob_palabra(tok_b, mod_b, ids, " hipopótamo")
    paris = sum(prob_palabra(tok_b, mod_b, ids, w)[0] for w in PALABRAS)
    print(f"[1] test nulo         «hipopótamo» detrás de la frase: {pct(nada, 4)}; París, las dos: {pct(paris, 2)}")
    if nada > 0.001 or paris > 1:
        fallos.append("test nulo: una palabra ajena sale probable, o las probabilidades pasan de uno")

    # 2. SEÑAL IMPLANTADA — detrás de «La capital de Francia es la ciudad de», el crudo pone
    #    París la primera (capítulo 7, M7-2): la cuenta de la palabra entera tiene que verlo.
    ids2 = tok_b(cap12.FRASE_CAP7 + " la ciudad de")["input_ids"]
    v, (_, pu) = prob_palabra(tok_b, mod_b, ids2, " París")
    print(f"[2] señal implantada  «París» tras «…la ciudad de»: {pct(v, 2)}, puesto {pu}")
    if pu != 1:
        fallos.append(f"señal implantada: París no sale la primera tras «la ciudad de» (puesto {pu})")

    # 3. INVARIANTE — las respuestas de aquí reproducen la batería de crudo_o_adiestrado.csv.
    cols = [responder(tok_b, mod_b, False), responder(tok_i, mod_i, False), responder(tok_i, mod_i, True)]
    try:
        comprobar(ETIQUETA_TAMANO[base], cols, leer_tabla())
        igual = True
    except AssertionError as e:
        igual = False
        fallos.append(f"invariante: {e}")
    print(f"[3] invariante        {ETIQUETA_TAMANO[base]} reproduce {CSV_TABLA}: {'sí' if igual else 'NO'}")
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
          "(do_sample=False), como crudo_o_adiestrado.py.")
    tabla = leer_tabla()
    for base, instruido in cap12.PAREJAS:
        etiqueta = ETIQUETA_TAMANO[base]
        print(f"\n{'=' * 60}\nTAMAÑO {etiqueta}\n{'=' * 60}\n")
        # Uno detrás de otro, no los dos a la vez: dos de siete mil millones no caben juntos
        # en la memoria del portátil con lo demás que hay en marcha.
        tok_b, mod_b = cap12.cargar(base)
        paris = [lineas_paris("en crudo", tok_b, mod_b)]
        napoles = bloque_napoles(tok_b, mod_b, etiqueta)
        cols = [responder(tok_b, mod_b, False)]
        del mod_b
        tok_i, mod_i = cap12.cargar(instruido)
        paris.append(lineas_paris("adiestrado, enunciado tal cual", tok_i, mod_i))
        cols += [responder(tok_i, mod_i, False), responder(tok_i, mod_i, True)]
        del mod_i
        comprobar(etiqueta, cols, tabla)
        if etiqueta == "7.000M":
            print("\n".join(bloque_paris(paris)) + "\n")
        print("\n".join(napoles) + "\n")
        print(f"(las respuestas reproducen la batería de {CSV_TABLA})\n")
        print("\n".join(bloque_tabla(etiqueta, cols)) + "\n")
        print("\n".join(bloque_tres_maneras(tok_b, tok_i, cols, etiqueta)))


if __name__ == "__main__":
    main()
