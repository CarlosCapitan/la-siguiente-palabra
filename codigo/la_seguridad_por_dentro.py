#!/usr/bin/env python3
"""
Capítulo 8 — de dónde sale «la seguridad» con que responde el modelo (L24).

`respuesta_modelo.py` obliga al modelo a elegir entre «vaso» y «cajón» y da su «confianza». El
capítulo la llama «seguridad» y concluye con ella («en la única pregunta que no tiene respuesta, la
máquina está más segura que en las otras dos»). Este programa enseña la cuenta entera, sin cambiar
nada de aquel:

  - el texto exacto que se le pasa al modelo;
  - de cada cien veces, cuántas escribiría «vaso» y cuántas «cajón» justo ahí, en su lista entera
    de trozos posibles (capítulo 7): el resto de su lista se va a otras cosas;
  - la seguridad: lo que se lleva la ganadora de lo que suman las dos, de cada cien;
  - y cuántos números tiene cada modelo («tres veces mayor»).

El selftest comprueba que la seguridad rehecha aquí es la «confianza» de `respuesta_modelo.py`.

Uso:
    python la_seguridad_por_dentro.py --selftest
    python la_seguridad_por_dentro.py > ../datos/salidas/la_seguridad_por_dentro.txt
"""

# ======================= CONSTANTES =======================

from respuesta_modelo import (CASOS, DTYPE, MODELOS, OPCIONES, PLANTILLA, SEMILLA,
                              eleccion_forzada, log_prob_continuacion)

TOL = 1e-6              # diferencia máxima con la confianza de respuesta_modelo.py
NOMBRES = {"Qwen/Qwen2.5-0.5B": "el de quinientos millones",      # como los llama el capítulo 11
           "Qwen/Qwen2.5-1.5B": "el de mil quinientos millones"}

# ==========================================================

import argparse
import math
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from formato import coma, comprobar_ancho, miles

ANCHO = 64


def cargar(nombre):
    tok = AutoTokenizer.from_pretrained(nombre)
    modelo = AutoModelForCausalLM.from_pretrained(nombre, dtype=getattr(torch, DTYPE))
    modelo.eval()
    return tok, modelo


def cuenta(tok, modelo, frase, adj, espacio_en_su_sitio=False):
    """De cada uno: cuánto va a «vaso» y a «cajón» en la lista entera, y la seguridad.

    respuesta_modelo.py deja el espacio al final del texto («Respuesta: El ») y pide «vaso» sin
    espacio delante. El troceador del modelo lleva el espacio pegado a la palabra que sigue
    («_vaso»), así que ese corte es uno que el modelo casi nunca ve: da a las dos opciones unas
    pocas millonésimas. Con espacio_en_su_sitio=True, el texto acaba en «Respuesta: El» y las
    opciones son « vaso» y « cajón», que es como el modelo las escribiría."""
    prompt = PLANTILLA.format(frase=frase, adj=adj)
    opciones = {o: o for o in OPCIONES}
    if espacio_en_su_sitio:
        assert prompt.endswith(" "), "se esperaba que la plantilla acabara en un espacio"
        prompt = prompt[:-1]
        opciones = {o: " " + o for o in OPCIONES}
    p = {o: math.exp(log_prob_continuacion(tok, modelo, prompt, c)) for o, c in opciones.items()}
    ganadora = max(p, key=p.get)
    return p, ganadora, p[ganadora] / sum(p.values())


def selftest(tok, modelo):
    fallos = []
    frase, _ = CASOS["alto"]
    # 1. TEST NULO — si las dos opciones fueran la misma palabra, la seguridad tendría que ser
    #    exactamente 50 de cada cien: la cuenta no prefiere a nadie por construcción.
    prompt = PLANTILLA.format(frase=frase, adj="alto")
    lp = log_prob_continuacion(tok, modelo, prompt, "vaso")
    nula = math.exp(lp) / (2 * math.exp(lp))
    print(f"[1] test nulo         «vaso» contra «vaso»: seguridad {coma(100 * nula)} %")
    if abs(nula - 0.5) > 1e-12:
        fallos.append("test nulo: con dos opciones iguales la seguridad no es 50")
    # 2. SEÑAL — la seguridad rehecha es la «confianza» de respuesta_modelo.py en los tres casos.
    peor = 0.0
    for adj, (fr, _) in CASOS.items():
        _, gan, seg = cuenta(tok, modelo, fr, adj)
        gan2, conf, _ = eleccion_forzada(tok, modelo, fr, adj)
        peor = max(peor, abs(seg - conf) + (0 if gan == gan2 else 1))
    print(f"[2] señal             contra la confianza de respuesta_modelo.py: diferencia {peor:.1e}")
    if peor > TOL:
        fallos.append("señal: la seguridad rehecha no es la de respuesta_modelo.py")
    # 3. INVARIANTE — lo que va a las dos opciones no pasa de cien, y la seguridad de la ganadora
    #    está entre 50 y 100.
    ok = True
    for adj, (fr, _) in CASOS.items():
        for sitio in (False, True):
            p, _, seg = cuenta(tok, modelo, fr, adj, sitio)
            ok &= sum(p.values()) <= 1 and 0.5 <= seg <= 1
    print(f"[3] invariante        las dos no pasan de cien y la seguridad está entre 50 y 100: "
          f"{'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: porcentajes fuera de rango")
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
    random.seed(SEMILLA); np.random.seed(SEMILLA); torch.manual_seed(SEMILLA)
    tok, modelo = cargar(MODELOS[0])
    print("--- selftest ---")
    codigo = selftest(tok, modelo)
    if codigo or args.selftest:
        return codigo
    L = ["", "####### capítulo 8: de dónde sale la seguridad #######", ""]
    frase, _ = CASOS["caro"]
    L += ["1. EL TEXTO QUE SE LE PASA (la frase del «caro»)", ""]
    L += ["  " + l for l in PLANTILLA.format(frase=frase, adj="caro").split("\n")]
    L += ["", "  El modelo continúa detrás de «Respuesta: El». Lo mismo con",
          "  «alto» y con «bajo».", ""]
    tamanos = {}
    modelos = {}
    for nombre in MODELOS:
        tok, modelo = cargar(nombre)
        tamanos[nombre] = sum(t.numel() for t in modelo.parameters())
        modelos[nombre] = {adj: (cuenta(tok, modelo, fr, adj), cuenta(tok, modelo, fr, adj, True))
                           for adj, (fr, _) in CASOS.items()}
    L += ["2. TAL COMO LO MIDE respuesta_modelo.py", "",
          "  El texto acaba en «Respuesta: El », con el espacio, y las",
          "  opciones son «vaso» y «cajón» sin espacio delante: un corte",
          "  que el modelo casi nunca ve.",
          "  «vaso», «cajón»: de cada MILLÓN de veces, cuántas escribiría",
          "  esa palabra justo ahí, en su lista entera de trozos posibles.",
          "  «seguridad»: lo que se lleva la ganadora de lo que suman",
          "  las dos, de cada cien.", ""]
    for nombre in MODELOS:
        L += [f"  {NOMBRES[nombre]}",
              f"  {'frase':<8}{'«vaso»':>10}{'«cajón»':>10}{'responde':>11}{'seguridad':>12}",
              f"  {'-----':<8}{'------':>10}{'-------':>10}{'--------':>11}{'---------':>12}"]
        for adj, ((p, gan, seg), _) in modelos[nombre].items():
            L.append(f"  {adj:<8}{coma(1e6 * p['vaso'], 2):>10}{coma(1e6 * p['cajón'], 2):>10}"
                     f"{gan:>11}{coma(100 * seg) + ' %':>12}")
        L.append("")
    L += ["3. CON EL ESPACIO EN SU SITIO", "",
          "  El texto acaba en «Respuesta: El» y las opciones son «_vaso»",
          "  y «_cajón», con el espacio delante, como las escribe el",
          "  modelo («_» es un espacio).",
          "  «vaso», «cajón»: de cada CIEN veces, cuántas escribiría esa",
          "  palabra justo ahí, en su lista entera de trozos posibles.",
          "  «las dos»: la suma; el resto va a otras palabras.",
          "  «seguridad»: lo que se lleva la ganadora de lo que suman",
          "  las dos, de cada cien.", ""]
    for nombre in MODELOS:
        L += [f"  {NOMBRES[nombre]}",
              f"  {'frase':<8}{'«vaso»':>10}{'«cajón»':>10}{'las dos':>10}{'responde':>11}{'seguridad':>12}",
              f"  {'-----':<8}{'------':>10}{'-------':>10}{'-------':>10}{'--------':>11}{'---------':>12}"]
        for adj, (_, (p, gan, seg)) in modelos[nombre].items():
            L.append(f"  {adj:<8}{coma(100 * p['vaso']) + ' %':>10}{coma(100 * p['cajón']) + ' %':>10}"
                     f"{coma(100 * sum(p.values())) + ' %':>10}{gan:>11}{coma(100 * seg) + ' %':>12}")
        L.append("")
    a, b = MODELOS
    L += ["4. EL TAMAÑO DE LOS DOS MODELOS", ""]
    for n in MODELOS:
        L.append(f"  {NOMBRES[n]:<32}{miles(tamanos[n]):>16} números")
    L += [f"  el segundo tiene {coma(tamanos[b] / tamanos[a])} veces los números del primero", ""]
    for l in comprobar_ancho([l.rstrip() for l in L], ANCHO):
        print(l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
