#!/usr/bin/env python3
"""
Capítulo 7 — de dónde salen los trozos (Carlos, 4 de octubre de 2026: «no está muy claro de
dónde salió esa forma de trocear»).

La máquina del capítulo 7 parte el texto en trozos con un procedimiento que se puede hacer a mano:

  1. se empieza con las letras sueltas; cada palabra lleva pegado su espacio de delante;
  2. se cuentan, en todo el texto, las parejas de piezas que van seguidas dentro de una palabra;
  3. la pareja que más se repite se suelda en una pieza nueva, que entra en la caja;
  4. y vuelta al paso 2.

Esto lo hace con el Quijote, y enseña las primeras juntas y cómo queda troceada una palabra con
0, 10, 100, 1.000 y 10.000 juntas en la caja.

Supuestos (declarados): letras, no bytes —la máquina grande empieza por los bytes, y una letra con
tilde es ahí dos piezas al principio—; los saltos de línea cuentan como espacios; el texto se
parte antes en palabras con su espacio delante, números y signos aparte, como hace la máquina
grande; si dos parejas empatan, gana la primera por orden alfabético; se respetan mayúsculas y
tildes.

Uso:
    python trocear_a_mano.py --selftest
    python trocear_a_mano.py > ../datos/salidas/trocear_a_mano.txt
"""

# ======================= CONSTANTES =======================

JUNTAS = 10_000                       # cuántas piezas nuevas se sueldan
PRIMERAS = 12                         # cuántas juntas se enseñan una a una
MOMENTOS = [0, 10, 100, 1_000, 10_000]
PALABRAS = [" murciélago", " Francia"]   # con su espacio delante (" caballero" salió el 4 oct.: no cabía en la página)
SALIDA_CSV = "trocear_a_mano.csv"
SEMILLA = 20261004

# ==========================================================

import argparse
import csv
import heapq
import random
import re
import sys
from collections import Counter, defaultdict

from formato import comprobar_ancho, ANCHO_CAJA
from ngrama import cargar_corpus, CORPUS

PARTIR = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+")


def a_la_vista(t):
    return t.replace(" ", "_")


def palabras(texto):
    """Las palabras del texto, cada una con su espacio de delante, contadas."""
    texto = re.sub(r"\s+", " ", texto)
    trozos = PARTIR.findall(texto)
    assert trozos, "Se esperaba algún texto que partir; no hay nada"
    assert "".join(trozos).replace(" ", "") == texto.replace(" ", ""), \
        "Se esperaba que las palabras, juntas, devolvieran el texto entero"
    return Counter(trozos)


def aprender(cuenta, juntas):
    """Las juntas, en orden: [(a, b, veces)]. Cuenta las parejas una vez y después solo actualiza
    las palabras que cambian."""
    tipos = [list(w) for w in cuenta]
    peso = [cuenta[w] for w in cuenta]
    parejas = Counter()
    donde = defaultdict(set)
    for i, s in enumerate(tipos):
        for p in zip(s, s[1:]):
            parejas[p] += peso[i]
            donde[p].add(i)
    monton = [(-n, p) for p, n in parejas.items()]
    heapq.heapify(monton)
    fuera = []
    while len(fuera) < juntas and monton:
        n, p = heapq.heappop(monton)
        if parejas.get(p, 0) != -n or n == 0:
            continue                                   # entrada vieja del montón
        a, b = p
        fuera.append((a, b, -n))
        cambiadas = set()
        for i in list(donde[p]):
            s = tipos[i]
            for q in zip(s, s[1:]):
                parejas[q] -= peso[i]
                donde[q].discard(i)
                cambiadas.add(q)
            nuevo, k = [], 0
            while k < len(s):
                if k + 1 < len(s) and s[k] == a and s[k + 1] == b:
                    nuevo.append(a + b); k += 2
                else:
                    nuevo.append(s[k]); k += 1
            tipos[i] = nuevo
            for q in zip(nuevo, nuevo[1:]):
                parejas[q] += peso[i]
                donde[q].add(i)
                cambiadas.add(q)
        for q in cambiadas:
            if parejas[q] > 0:
                heapq.heappush(monton, (-parejas[q], q))
        parejas.pop(p, None)
    return fuera


def trocear(palabra, juntas):
    """Trocea una palabra con las juntas que haya en la caja, en su orden."""
    rango = {(a, b): k for k, (a, b, _) in enumerate(juntas)}
    s = list(palabra)
    while len(s) > 1:
        cand = [(rango[p], i) for i, p in enumerate(zip(s, s[1:])) if p in rango]
        if not cand:
            break
        r, _ = min(cand)
        a, b, _ = juntas[r]
        nuevo, k = [], 0
        while k < len(s):
            if k + 1 < len(s) and s[k] == a and s[k + 1] == b:
                nuevo.append(a + b); k += 2
            else:
                nuevo.append(s[k]); k += 1
        s = nuevo
    return s


def miles(n):
    return f"{n:,}".replace(",", ".")


def tabla(juntas, letras):
    lin = ["LAS PRIMERAS JUNTAS, CON EL QUIJOTE",
           "cada junta suelda la pareja que más se repite; «_» es un espacio", "",
           f"{'junta':>5}   {'pareja':<16}{'pieza nueva':<14}{'veces':>9}"]
    for k, (a, b, n) in enumerate(juntas[:PRIMERAS], 1):
        par = f"«{a_la_vista(a)}» + «{a_la_vista(b)}»"
        lin.append(f"{k:>5}   {par:<16}{'«' + a_la_vista(a + b) + '»':<14}{miles(n):>9}")
    lin += ["", f"se empieza con {letras} piezas de una letra; con "
                f"{miles(len(juntas))} juntas, la caja", f"tiene {miles(letras + len(juntas))} piezas"]
    lin += ["", "", "UNA PALABRA, SEGÚN CUÁNTAS JUNTAS HAY EN LA CAJA"]
    for w in PALABRAS:
        lin += ["", a_la_vista(w)]
        for m in MOMENTOS:
            s = trocear(w, juntas[:m])
            lin.append(f"{miles(m):>8}   " + " | ".join(a_la_vista(t) for t in s))
    return comprobar_ancho(lin, ANCHO_CAJA)


def guardar(juntas, ruta):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["junta", "a", "b", "veces"])
        for k, (a, b, n) in enumerate(juntas, 1):
            w.writerow([k, a, b, n])


def leer(ruta):
    with open(ruta, encoding="utf-8", newline="") as f:
        return [(r["a"], r["b"], int(r["veces"])) for r in csv.DictReader(f)]


# ============================ SELFTEST ============================

def selftest():
    fallos = []
    texto = cargar_corpus(CORPUS)
    cuenta = palabras(texto)

    # [1] Test nulo: palabras de letras al azar. Ninguna pareja de letras puede destacar: la que
    #     más se repite no llega a vez y media la media de todas.
    rng = random.Random(SEMILLA)
    abc = "abcdefghijklmnopqrst"
    azar = " ".join("".join(rng.choice(abc) for _ in range(rng.randint(3, 8))) for _ in range(60_000))
    c = Counter()
    for w, n in palabras(azar).items():
        for p in zip(w, w[1:]):
            if " " not in p:
                c[p] += n
    razon = max(c.values()) / (sum(c.values()) / len(c))
    print(f"[1] test nulo         letras al azar: la pareja que más se repite, "
          f"{razon:.2f} veces la media (tiene que quedar por debajo de 1,5)")
    if razon >= 1.5:
        fallos.append("test nulo: con letras al azar hay una pareja que destaca")

    # [2] Señal implantada: una palabra inventada, repetida doscientas mil veces, sale soldada
    #     entera entre las primeras juntas.
    con = cuenta.copy()
    con[" zxqvk"] += 200_000
    j = aprender(con, 15)
    entera = trocear(" zxqvk", j) == [" zxqvk"]
    print(f"[2] señal implantada  «_zxqvk» doscientas mil veces: entera con 15 juntas: {entera}")
    if not entera:
        fallos.append("señal: la palabra implantada no sale soldada entre las primeras juntas")

    # [3] Invariante: pegar los trozos devuelve la palabra exacta, y con más juntas en la caja una
    #     palabra nunca tiene más cortes que con menos.
    juntas = aprender(cuenta, 2_000)
    muestra = rng.sample(sorted(cuenta), 2_000) + PALABRAS
    exacta = all("".join(trocear(w, juntas)) == w for w in muestra)
    mono = all(len(trocear(w, juntas[:a])) >= len(trocear(w, juntas[:b]))
               for w in PALABRAS for a, b in zip([0, 10, 100, 1000], [10, 100, 1000, 2000]))
    print(f"[3] invariante        los trozos pegados dan la palabra: {exacta}; "
          f"más juntas, nunca más cortes: {mono}")
    if not (exacta and mono):
        fallos.append("invariante: trocear rompe la palabra o le añade cortes")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    if p.parse_args().selftest:
        return selftest()
    cuenta = palabras(cargar_corpus(CORPUS))
    letras = len({ch for w in cuenta for ch in w})
    juntas = aprender(cuenta, JUNTAS)
    assert len(juntas) == JUNTAS, f"Se esperaban {JUNTAS} juntas; salieron {len(juntas)}"
    for l in tabla(juntas, letras):
        print(l)
    guardar(juntas, SALIDA_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
