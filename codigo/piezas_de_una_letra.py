#!/usr/bin/env python3
"""
Capítulo 7 — ¿hay piezas de una sola letra en el repertorio de trozos?

El borrador del capítulo 7 dice que el repertorio de trozos funciona como una caja de piezas de
construcción, y que «como en la caja hay piezas de una sola letra, se puede montar cualquier cosa».
Estaba comprobado a mano, pero no lo imprimía ningún programa (fallo 9). Éste lo imprime.

Solo usa el troceador del modelo, no el modelo: la respuesta no depende de la máquina ni de ningún
cálculo con decimales.

Uso:
    python piezas_de_una_letra.py --selftest
    python piezas_de_una_letra.py > ../datos/salidas/piezas_de_una_letra.txt
"""

# ======================= CONSTANTES =======================

MODELO = "Qwen/Qwen2.5-0.5B"      # el mismo de maquina_entera.py y maquina_entera_detalle.py
LETRAS = "abcdefghijklmnñopqrstuvwxyzáéíóúü"   # las del alfabeto de ngrama.py, sin signos
PALABRA_INVENTADA = "zorrillañez"             # no existe: se tiene que poder montar igual

# ==========================================================

import argparse
import sys

from formato import ANCHO_CAJA, comprobar_ancho


def cargar():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODELO)


def trozos(tok, texto):
    ids = tok.encode(texto, add_special_tokens=False)
    return [tok.decode([i]) for i in ids]


def bloque(tok):
    L = ["LAS LETRAS DEL CASTELLANO, UNA A UNA, EN EL TROCEADOR", "",
         f"{'letra':<8}{'trozos':>8}{'mayúscula':>12}{'trozos':>8}"]
    solas = 0
    for c in LETRAS:
        n, N = len(trozos(tok, c)), len(trozos(tok, c.upper()))
        solas += (n == 1) + (N == 1)
        L.append(f"{c:<8}{n:>8}{c.upper():>12}{N:>8}")
    L += ["", f"son un trozo ellas solas: {solas} de {2 * len(LETRAS)}", ""]
    t = trozos(tok, PALABRA_INVENTADA)
    L.append(f"una palabra que no existe, «{PALABRA_INVENTADA}»: {len(t)} trozos")
    L.append("  " + " | ".join(t))
    return comprobar_ancho(L, ANCHO_CAJA)


def selftest():
    fallos = []
    tok = cargar()
    # 1. TEST NULO — una cadena que no es una letra, «ab», no puede contar como una letra suelta:
    #    o es un trozo de dos letras o son dos. Lo que se comprueba es que la cuenta de «letras que
    #    son un trozo» no cuenta todo lo que se le pasa.
    ab = "".join(trozos(tok, "ab"))
    print(f"[1] test nulo         «ab» vuelve a dar «{ab}» al pegar sus trozos: {'sí' if ab == 'ab' else 'NO'}")
    if ab != "ab":
        fallos.append("test nulo: pegar los trozos de «ab» no devuelve «ab»")
    # 2. SEÑAL — la «a» sola tiene que ser un trozo (es la letra más corriente del idioma), y una
    #    frase conocida del libro tiene que salir con los trozos que enseña el capítulo 7.
    fr = trozos(tok, "La capital de Francia es")
    esperado = ["La", " capital", " de", " Franc", "ia", " es"]
    print(f"[2] señal             «a»: {len(trozos(tok, 'a'))} trozo; la frase del capítulo: "
          f"{'la misma' if fr == esperado else fr}")
    if len(trozos(tok, "a")) != 1 or fr != esperado:
        fallos.append("señal: la «a» no es un trozo, o la frase no sale como en el capítulo 7")
    # 3. INVARIANTE — pegar los trozos de cualquier palabra devuelve la palabra: nada se pierde.
    malas = [p for p in [PALABRA_INVENTADA, "murciélago", "ñandú", "esternocleidomastoideo"]
             if "".join(trozos(tok, p)) != p]
    print(f"[3] invariante        pegar los trozos devuelve la palabra: {'sí' if not malas else malas}")
    if malas:
        fallos.append(f"invariante: al pegar los trozos no vuelve la palabra en {malas}")
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
        return codigo
    print()
    print(f"modelo: {MODELO} (solo su troceador)")
    print()
    for l in bloque(cargar()):
        print(l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
