#!/usr/bin/env python3
"""
Capítulo 7 — la máquina entera: seguir una frase de punta a punta.

Enseña, sobre un modelo real y pequeño, los cuatro momentos del recorrido:
  1. El texto se parte en trozos.
  2. Cada trozo se convierte en una lista de números.
  3. Los números atraviesan muchas rondas de mirar y mezclar.
  4. Sale una lista de probabilidades sobre TODAS las palabras posibles, se elige una,
     y se vuelve a empezar.

Uso:
    python maquina_entera.py
    python maquina_entera.py --selftest
"""

# ======================= CONSTANTES =======================

MODELO = "Qwen/Qwen2.5-0.5B"
DTYPE = "float32"          # float32 para que el resultado sea el mismo en cualquier máquina

FRASE = "La capital de Francia es"
FRASES_TROCEADO = [
    "La capital de Francia es",
    "El murciélago hambriento vigilaba",
    "anticonstitucionalmente",
]
TOP_N = 10
PASOS_BUCLE = 8

# selftest
PROMPT_FACIL = "uno, dos, tres, cuatro,"
CONTINUACION_FACIL = " cinco"
UMBRAL_FACIL = 0.30
TOL_SUMA = 1e-4

SALIDA_CSV = "maquina_entera.csv"

# ==========================================================

from formato import (ANCHO_CAJA_CITA, coma, comprobar_ancho, pct, miles, tabla_de_probabilidades,
                     ANCHO_TROZO, ANCHO_PROB)

import argparse
import csv
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def cargar():
    tok = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO, dtype=getattr(torch, DTYPE))
    modelo.eval()
    return tok, modelo


def trozos(tok, texto):
    ids = tok(texto)["input_ids"]
    return [tok.decode([i]) for i in ids]


def siguientes(tok, modelo, texto, n=TOP_N):
    ids = tok(texto, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        logits = modelo(ids).logits[0, -1]
    p = torch.softmax(logits.float(), dim=-1)
    suma = float(p.sum())
    assert abs(suma - 1.0) < TOL_SUMA, \
        f"Invariante roto: se esperaba que las probabilidades sumaran 1; se encontró {suma:.6f}"
    assert float(p.min()) >= 0, \
        f"Invariante roto: se esperaban probabilidades no negativas; la menor es {float(p.min()):.2e}"
    top = torch.topk(p, n)
    return [(tok.decode([int(i)]), float(v)) for v, i in zip(top.values, top.indices)], p


def dispersion(p):
    """Entropía normalizada: 0 = seguro de una sola opción, 1 = todas igual de probables."""
    q = p[p > 0]
    return float(-(q * q.log()).sum() / math.log(len(p)))


def selftest(tok, modelo):
    fallos = []

    # 1. TEST NULO — una frase sin estructura debe dejar al modelo mucho menos decidido
    #    que una frase con una continuación evidente.
    _, p_facil = siguientes(tok, modelo, PROMPT_FACIL, 3)
    _, p_nula = siguientes(tok, modelo, "qx zr vb kk pl", 3)
    d_facil, d_nula = dispersion(p_facil), dispersion(p_nula)
    print(f"[1] test nulo         dispersión: frase con sentido {coma(d_facil, 3)}  "
          f"sin sentido {coma(d_nula, 3)}")
    if not d_nula > d_facil:
        fallos.append(f"test nulo: la frase sin sentido ({coma(d_nula, 3)}) no sale más dispersa "
                      f"que la que tiene sentido ({coma(d_facil, 3)})")

    # 2. SEÑAL IMPLANTADA — tras «uno, dos, tres, cuatro,» la continuación es obvia.
    top, _ = siguientes(tok, modelo, PROMPT_FACIL, 5)
    prob_cinco = dict(top).get(CONTINUACION_FACIL, 0.0)
    print(f"[2] señal implantada  «{PROMPT_FACIL}» -> «{top[0][0]}» con {coma(top[0][1], 3)}; "
          f"«{CONTINUACION_FACIL.strip()}» tiene {coma(prob_cinco, 3)}")
    if prob_cinco < UMBRAL_FACIL:
        fallos.append(f"señal implantada: se esperaba «cinco» con al menos "
                      f"{coma(UMBRAL_FACIL, 2)}; se obtuvo {coma(prob_cinco, 3)}")

    # 3. INVARIANTE DEL DOMINIO — la lista de probabilidades cubre TODO el vocabulario y
    #    suma uno. (Los asserts de `siguientes` ya lo comprueban en cada llamada.)
    _, p = siguientes(tok, modelo, FRASE)
    print(f"[3] invariante        {miles(len(p))} palabras posibles, "
          f"suman {coma(float(p.sum()), 6)}, mínima {float(p.min()):.2e}")
    if len(p) < 1000:
        fallos.append(f"invariante: vocabulario sospechosamente pequeño ({miles(len(p))})")

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

    tok, modelo = cargar()
    if args.selftest:
        sys.exit(selftest(tok, modelo))

    filas = []

    print("--- 1. EL TEXTO SE PARTE EN TROZOS ---")
    for t in FRASES_TROCEADO:
        piezas = trozos(tok, t)
        print(f"«{t}»")
        print(f"   {len(piezas)} trozos: " + " | ".join(p.replace(' ', '_') for p in piezas))
    print()

    print("--- 2. EL TAMAÑO DE LA MÁQUINA ---")
    cfg = modelo.config
    total = sum(p.numel() for p in modelo.parameters())
    print(f"modelo: {MODELO}")
    print(f"números por trozo: {cfg.hidden_size}")
    print(f"rondas de mirar y mezclar: {cfg.num_hidden_layers} capas x "
          f"{cfg.num_attention_heads} cabezas = {cfg.num_hidden_layers * cfg.num_attention_heads}")
    print(f"palabras posibles en la salida: {miles(cfg.vocab_size)}")
    print(f"números ajustables en total: {miles(total)}")
    print()

    print("--- 3. LA LISTA DE PROBABILIDADES ---")
    top, p = siguientes(tok, modelo, FRASE)
    # La tabla lleva rótulo de columna y una barra a escala fija: el número dice cuánto,
    # y la barra deja verlo sin leerlo. El rótulo no es adorno; sin él son dos columnas
    # de cifras y el lector que se encuentre la tabla al volver la página no sabe de qué.
    resto = 1.0 - sum(v for _, v in top)
    pares = [(w.replace(" ", "_"), v) for w, v in top] + [("(el resto)", resto)]
    lineas = [f"«{FRASE}» -> ¿qué viene después?", ""]
    lineas += tabla_de_probabilidades(pares)
    lineas += ["", f"el resto se reparte entre las otras {miles(len(p)-TOP_N)} posibilidades"]
    comprobar_ancho(lineas)
    for l in lineas:
        print(l)
    filas += [["probabilidad", FRASE, w, f"{v:.6f}"] for w, v in top]
    print()

    print("--- 4. Y SE VUELVE A EMPEZAR ---")
    texto = FRASE
    # Dos líneas por paso, y no una. Con el texto entero en la misma fila la línea llegaba a 91
    # caracteres: en el libro eso no cabe, se convierte en tabla, y una tabla de Markdown no la
    # mide ningún verificador —ni se desborda en el PDF: se dobla—. Acabó impresa partida en
    # cuatro renglones por fila, ocupando una página entera e ilegible (fallo 4.27). Así se lee
    # como una escalera, que además es lo que la tabla quiere enseñar.
    lineas = [f"{'paso':>4}  {'trozo':>{ANCHO_TROZO}}  {'probabilidad':>{ANCHO_PROB}}"]
    for i in range(1, PASOS_BUCLE + 1):
        top, _ = siguientes(tok, modelo, texto, 1)
        palabra, prob = top[0]
        texto += palabra
        lineas.append(f"{i:>4}  {palabra.replace(' ', '_'):>{ANCHO_TROZO}}  "
                      f"{pct(prob, 2):>{ANCHO_PROB}}")
        # Si el texto ya no cabe, se corta por la IZQUIERDA y se marca con puntos suspensivos:
        # un corte que no se ve es una mentira pequeña, y aquí el lector tiene que poder saber
        # que lo que ve es el final de algo más largo.
        visible = texto if len(texto) <= 56 else "…" + texto[-55:]
        lineas.append(f"      {visible}")
        filas.append(["bucle", str(i), palabra, f"{prob:.6f}"])
    comprobar_ancho(lineas, ANCHO_CAJA_CITA)
    for l in lineas:
        print(l)

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["medicion", "contexto", "palabra", "probabilidad"]] + filas)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
