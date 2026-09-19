#!/usr/bin/env python3
"""
Capítulo 11 — qué aparece al crecer.

Pasa la misma batería de tareas a tres modelos de la MISMA familia y distinto tamaño, para
ver qué sabe hacer cada uno. Todos son modelos en crudo, sin adiestramiento de instrucciones:
lo único que cambia entre ellos es el tamaño.

Cada tarea se puntúa por coincidencia exacta del principio de la continuación, eligiendo
siempre la opción más probable. Sin trucos y sin interpretación.

Uso:
    python crecer.py
    python crecer.py --selftest
"""

# ======================= CONSTANTES =======================

# Misma familia, mismo adiestramiento (ninguno), solo cambia el tamaño. El de 7.000 millones
# ocupa unos 15 GB de descarga; si no cabe o no se quiere bajar, el guion lo salta y lo dice.
MODELOS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]
DTYPE = "bfloat16"
MAX_NUEVOS = 8
SEMILLA = 20260914

TAREAS = {
    "sumar dos cifras": [
        ("12 + 25 = ", "37"), ("31 + 46 = ", "77"), ("54 + 23 = ", "77"),
        ("18 + 41 = ", "59"), ("62 + 27 = ", "89"),
    ],
    "plurales": [
        ("perro -> perros\ngato -> gatos\nmesa -> ", "mesas"),
        ("perro -> perros\ngato -> gatos\nlibro -> ", "libros"),
        ("perro -> perros\ngato -> gatos\nluz -> ", "luces"),
        ("perro -> perros\ngato -> gatos\npared -> ", "paredes"),
        ("perro -> perros\ngato -> gatos\nlápiz -> ", "lápices"),
    ],
    "traducir del inglés": [
        ("dog = perro\nhouse = casa\nwater = ", "agua"),
        ("dog = perro\nhouse = casa\nbook = ", "libro"),
        ("dog = perro\nhouse = casa\nnight = ", "noche"),
        ("dog = perro\nhouse = casa\nbread = ", "pan"),
        ("dog = perro\nhouse = casa\nfriend = ", "amigo"),
    ],
    "saber cosas del mundo": [
        ("Pregunta: ¿cuál es la capital de Francia?\nRespuesta: ", "París"),
        ("Pregunta: ¿cuál es la capital de Italia?\nRespuesta: ", "Roma"),
        ("Pregunta: ¿cuál es la capital de Portugal?\nRespuesta: ", "Lisboa"),
        ("Pregunta: ¿en qué continente está Egipto?\nRespuesta: ", "África"),
        ("Pregunta: ¿cuántas patas tiene una araña?\nRespuesta: ", "8"),
    ],
    "seguir un patrón inventado": [
        ("casa -> asac\nmesa -> asem\nlibro -> ", "orbil"),
        ("casa -> asac\nmesa -> asem\ngato -> ", "otag"),
        ("casa -> asac\nmesa -> asem\nsol -> ", "los"),
        ("casa -> asac\nmesa -> asem\npan -> ", "nap"),
        ("casa -> asac\nmesa -> asem\nmar -> ", "ram"),
    ],
    "razonar sobre una frase": [
        ("El vaso no cabía en el cajón porque el vaso era demasiado alto.\n"
         "Pregunta: ¿qué era demasiado alto?\nRespuesta: el ", "vaso"),
        ("El vaso no cabía en el cajón porque el cajón era demasiado bajo.\n"
         "Pregunta: ¿qué era demasiado bajo?\nRespuesta: el ", "cajón"),
        ("Ana le dio un libro a Luis. Luis lo leyó.\n"
         "Pregunta: ¿quién leyó el libro?\nRespuesta: ", "Luis"),
        ("Hoy es martes.\nPregunta: ¿qué día será mañana?\nRespuesta: ", "miércoles"),
        ("Tengo 5 manzanas y me como 2.\nPregunta: ¿cuántas me quedan?\nRespuesta: ", "3"),
    ],
}

TAREA_CONTROL = [("Repite: perro\nperro\nRepite: gato\ngato\nRepite: mesa\n", "mesa"),
                 ("Repite: sol\nsol\nRepite: mar\nmar\nRepite: pan\n", "pan")]
UMBRAL_CONTROL = 0.99

SALIDA_CSV = "crecer.csv"

# ==========================================================

import argparse
import csv
import random
import re
import sys
import unicodedata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from formato import comprobar_ancho, miles, pct

# Rótulo corto para cada tamaño (rule 9: "500 millones" no cabe en la columna). "M" y no
# "B": el nombre en inglés del modelo dice "B" por "billion" (mil millones), y en español
# "billón" es otra cosa (un millón de millones). Traducir esa letra tal cual sería el
# mismo fallo que castellanizar un punto que es parte de un nombre.
ETIQUETA_TAMANO = {
    "Qwen/Qwen2.5-0.5B": "500M",
    "Qwen/Qwen2.5-1.5B": "1.500M",
    "Qwen/Qwen2.5-3B": "3.000M",
    "Qwen/Qwen2.5-7B": "7.000M",
}


def normalizar(t):
    t = unicodedata.normalize("NFC", t.strip().lower())
    return re.sub(r"[^\wáéíóúüñ]+", "", t)


def cargar(nombre):
    tok = AutoTokenizer.from_pretrained(nombre)
    modelo = AutoModelForCausalLM.from_pretrained(nombre, dtype=getattr(torch, DTYPE))
    modelo.eval()
    return tok, modelo


def continuar(tok, modelo, prompt, maximo=MAX_NUEVOS):
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        salida = modelo.generate(ids, max_new_tokens=maximo, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
    return tok.decode(salida[0, ids.shape[1]:], skip_special_tokens=True)


def acierta(respuesta, esperada):
    r = normalizar(respuesta.split("\n")[0])
    return r.startswith(normalizar(esperada))


def evaluar(tok, modelo, tareas=TAREAS):
    resultados = {}
    for nombre, items in tareas.items():
        aciertos = sum(acierta(continuar(tok, modelo, p), e) for p, e in items)
        resultados[nombre] = aciertos / len(items)
    return resultados


def selftest():
    fallos = []
    nombre = MODELOS[0]
    tok, modelo = cargar(nombre)

    # 1. TEST NULO — se barajan las palabras de cada enunciado, así que la tarea deja de
    #    tener sentido. El acierto debe caer a casi nada.
    rng = random.Random(SEMILLA)
    revueltas = {}
    for n, items in TAREAS.items():
        nuevos = []
        for p, e in items:
            palabras = p.split()
            rng.shuffle(palabras)
            nuevos.append((" ".join(palabras) + " ", e))
        revueltas[n] = nuevos
    r_nulo = evaluar(tok, modelo, revueltas)
    media_nula = sum(r_nulo.values()) / len(r_nulo)
    r_real = evaluar(tok, modelo)
    media_real = sum(r_real.values()) / len(r_real)
    print(f"[1] test nulo         enunciados barajados {media_nula:.3f}  enunciados reales {media_real:.3f}")
    if media_nula >= media_real or media_nula > 0.20:
        fallos.append(f"test nulo: barajado {media_nula:.3f} frente a real {media_real:.3f}; "
                      "el montaje da puntos sin que haya tarea")

    # 2. SEÑAL IMPLANTADA — copiar una palabra que está delante. Si el más pequeño falla
    #    esto, el problema es el montaje y no el tamaño.
    ok = sum(acierta(continuar(tok, modelo, p), e) for p, e in TAREA_CONTROL) / len(TAREA_CONTROL)
    print(f"[2] señal implantada  copiar la palabra anterior: {ok:.3f}")
    if ok < UMBRAL_CONTROL:
        fallos.append(f"señal implantada: el modelo más pequeño no copia una palabra ({ok:.3f}); "
                      "el montaje es defectuoso")

    # 3. INVARIANTE DEL DOMINIO — se elige siempre la opción más probable, así que dos
    #    ejecuciones tienen que dar exactamente lo mismo.
    p = TAREAS["plurales"][0][0]
    a, b = continuar(tok, modelo, p), continuar(tok, modelo, p)
    print(f"[3] invariante        dos ejecuciones idénticas: {'sí' if a == b else 'NO'}")
    if a != b:
        fallos.append(f"invariante: dos ejecuciones dan resultados distintos: {a!r} y {b!r}")

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

    filas, tabla, usados = [], {}, []
    for nombre in MODELOS:
        try:
            tok, modelo = cargar(nombre)
        except Exception as e:                      # sin memoria, sin disco o sin red
            print(f"SALTADO {nombre}: {type(e).__name__}: {str(e)[:120]}")
            continue
        total = sum(p.numel() for p in modelo.parameters())
        r = evaluar(tok, modelo)
        tabla[nombre] = (total, r)
        usados.append(nombre)
        filas += [[nombre, total, t, f"{v:.3f}"] for t, v in r.items()]
        del modelo

    assert len(usados) >= 2, \
        f"Se esperaban al menos dos modelos para poder comparar; solo se pudo cargar {len(usados)}"
    etiquetas = [ETIQUETA_TAMANO[n] for n in usados]
    print(f"\n{'tarea':<28}" + "".join(f"{e:>10}" for e in etiquetas))
    print("-" * (28 + 10 * len(etiquetas)))
    for tarea in TAREAS:
        fila = "".join(f"{pct(tabla[m][1][tarea], 0):>10}" for m in usados)
        print(f"{tarea:<28}{fila}")
    print("-" * (28 + 10 * len(etiquetas)))
    medias = "".join(f"{pct(sum(tabla[m][1].values()) / len(TAREAS), 0):>10}" for m in usados)
    print(f"{'media':<28}{medias}")
    print()
    # La clave debajo de la tabla, impresa por el programa (regla 9): el rótulo corto de
    # la columna ("500M") no dice lo que mide, y la unidad ("N intentos por tarea") tampoco
    # cabe en la cabecera. Las dos van aquí, en líneas que el libro copia tal cual (regla 6).
    # Una línea por columna diciendo todas lo mismo no es una clave, es ruido: en la página
    # impresa salían cuatro renglones idénticos debajo de la tabla, y eso no lo caza ningún
    # verificador porque las cuatro líneas son literales y correctas. Se dice una vez.
    clave = [f"«{etiquetas[0]}», «{etiquetas[1]}»…: números ajustables del modelo, "
             "redondeados a millones."]
    clave.append(f"las cifras de la tabla son aciertos sobre {len(next(iter(TAREAS.values())))} "
                  "intentos por tarea.")
    for l in comprobar_ancho(clave):
        print(l)
    print("\nnúmeros ajustables del modelo:")
    for e, m in zip(etiquetas, usados):
        for l in comprobar_ancho([f"  {e}: {miles(tabla[m][0])}"]):
            print(l)

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["modelo", "numeros", "tarea", "acierto"]] + filas)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
