#!/usr/bin/env python3
"""
Capítulo 7 — cuatro pasos más de la máquina entera, y qué es «población» (Orden 11, 17 oct 2026).

El capítulo 7 sigue ocho pasos de la máquina de 500 millones detrás de «La capital de Francia es»
(la tabla «Las tres primeras candidatas de cada paso» de `maquina_entera_detalle.txt`) y, al contar
cuántos de los pasos terminan una palabra, dice de uno de los que no: ««población» (93 %) detrás de
«la ciudad más grande de Francia, con una», tres pasos después de donde se paró la tabla». Esa cifra
no estaba en ninguna salida, y en la caja hay un trozo `_pobl`: si «población» fuera en dos trozos,
y el segundo («ación») terminara la palabra, el ejemplo estaría en el grupo equivocado.

Este programa sigue la misma máquina con el mismo procedimiento de aquella tabla (siempre el trozo
más probable) **cuatro pasos más**, y da para cada uno las tres primeras candidatas con su
porcentaje, como allí. Luego mira «población» de cerca: en cuántos trozos la parte el cortador, qué
probabilidad lleva cada trozo en su sitio y si alguno termina palabra, y qué pasa si en vez de
«población» entera la máquina empezara por el trozo `_pobl`.

No toca `maquina_entera_detalle.txt`: importa sus funciones (el mismo modelo, el mismo tipo de
número, la lista de probabilidades en float64) y repite sus ocho primeros pasos, que tienen que
dar, fila a fila, los de su tabla guardada; si no, el selftest revienta.

Uso:
    python maquina_entera_pasos.py --selftest
    python maquina_entera_pasos.py > ../datos/salidas/maquina_entera_pasos.txt
"""

# ======================= CONSTANTES =======================

DETALLE = "../datos/salidas/maquina_entera_detalle.txt"
TABLA_OCHO_PASOS = "Las tres primeras candidatas de cada paso"

PASOS_DE_LA_TABLA = 8              # los de maquina_entera_detalle
PASOS_MAS = 4                      # los que se siguen ahora (9 a 12)
CANDIDATAS = 3                     # como en la tabla de los ocho pasos
PALABRA = " población"             # la del capítulo
INICIO_PALABRA = " pobl"           # el trozo que hay en la caja y que podría empezar «población»
FINAL_ESPERADO = "ación"           # lo que terminaría una «pobl»+«ación»
UMBRAL_AZAR = 1e-3                 # un trozo cualquiera, al azar, no pasa de esto en la lista

# ==========================================================

import argparse
import datetime
import platform
import sys
from pathlib import Path

import torch
import transformers

import maquina_entera_detalle as D
from formato import coma, leer_tablas, pct, tabla_editorial, trozo

AQUI = Path(__file__).resolve().parent


def seguir(tok, modelo, pasos):
    """Sigue `pasos` pasos desde FRASE siempre con el trozo más probable. Devuelve, por paso, las
    CANDIDATAS primeras [(trozo, probabilidad)] y la lista de ids hasta ese momento (antes de
    añadir el elegido)."""
    ids = D.ids_de(tok, D.FRASE)
    filas = []
    for _ in range(pasos):
        p = D.lista(modelo, ids)
        top = torch.topk(p, CANDIDATAS)
        filas.append(([(tok.decode([int(i)]), float(v)) for v, i in zip(top.values, top.indices)],
                      list(ids)))
        ids = ids + [int(top.indices[0])]
    return filas, ids


def celda(par):
    t, v = par
    return f"{trozo(D.visible(t))} {pct(v, 1)}"


# ---- el informe ------------------------------------------------------------------------

def informe(tok, modelo):
    ocho, ids8 = seguir(tok, modelo, PASOS_DE_LA_TABLA)
    todos, ids12 = seguir(tok, modelo, PASOS_DE_LA_TABLA + PASOS_MAS)
    mas = todos[PASOS_DE_LA_TABLA:]
    L = []
    L += [f"--- 1. CUATRO PASOS MÁS, DETRÁS DE «{tok.decode(ids8)}» ---", ""]
    filas = [[str(PASOS_DE_LA_TABLA + k)] + [celda(c) for c in cands]
             for k, (cands, _) in enumerate(mas, 1)]
    L += tabla_editorial(
        "Cuatro pasos más de la máquina entera",
        ["paso"] + [f"{k}.ª opción" for k in range(1, CANDIDATAS + 1)], filas,
        "c" + "i" * CANDIDATAS,
        ["Sigue la tabla de los ocho pasos de maquina_entera_detalle.txt: se elige siempre la 1.ª; "
         "las otras dos son las que perdieron. Porcentajes con una cifra decimal.",
         f"Texto al acabar: «{tok.decode(ids12)}»."])
    L += ["", "--- 2. «POBLACIÓN», DE CERCA ---", ""]
    # el sitio donde la máquina podría escribir «población»: el paso en que el elegido la contiene
    ids_antes = None
    paso_poblacion = None
    for k, (cands, previos) in enumerate(mas, PASOS_DE_LA_TABLA + 1):
        if cands[0][0].startswith(INICIO_PALABRA):          # «población» entera o `_pobl`
            ids_antes, paso_poblacion = previos, k
            break
    assert ids_antes is not None, (f"Se esperaba «{PALABRA.strip()}» entre las elegidas de los pasos "
                                   f"9 a 12; no está")
    contexto = tok.decode(ids_antes)
    cortes = [tok.decode([i]) for i in D.ids_de(tok, PALABRA)]
    total, desglose = D.prob_palabra(tok, modelo, contexto, PALABRA)
    L += tabla_editorial(
        f"«{PALABRA.strip()}» detrás de «{contexto.replace(D.FRASE, '…', 1)}»",
        ["trozo", "probabilidad", "puesto"],
        [[trozo(D.visible(t)), pct(v, 1), str(pu)] for t, v, pu in desglose]
        + [["**«" + PALABRA.strip() + "» entera**", "**" + pct(total, 1) + "**", ""]], "idd",
        [f"El cortador parte «{PALABRA.strip()}» en {len(cortes)} trozo{'s' if len(cortes) > 1 else ''}: "
         + " ".join(trozo(D.visible(c)) for c in cortes) + ".",
         f"Es el paso {paso_poblacion} desde «{D.FRASE}»; el contexto es «{contexto}»."])
    # ¿y si empezara por el trozo `_pobl`, que también está en la caja?
    p = D.lista(modelo, D.ids_de(tok, contexto))
    ini = D.ids_de(tok, INICIO_PALABRA)
    assert len(ini) == 1, f"Se esperaba un solo trozo para {INICIO_PALABRA!r}; hay {len(ini)}"
    p_ini = float(p[ini[0]])
    top1 = torch.topk(p, CANDIDATAS)
    p2 = D.lista(modelo, D.ids_de(tok, contexto) + ini)
    top2 = torch.topk(p2, CANDIDATAS)
    fin = D.ids_de(tok, FINAL_ESPERADO)
    p_fin = float(p2[fin[0]]) if len(fin) == 1 else None
    L += [""]
    L += tabla_editorial(
        f"Y si empezara por {trozo(D.visible(INICIO_PALABRA))}",
        ["detrás de…", "1.ª opción", "2.ª opción", "3.ª opción"],
        [[f"«{contexto}»", *[celda((tok.decode([int(i)]), float(v)))
                              for v, i in zip(top1.values, top1.indices)]],
         [f"…{trozo(D.visible(INICIO_PALABRA))}", *[celda((tok.decode([int(i)]), float(v)))
                                                     for v, i in zip(top2.values, top2.indices)]]],
        "iiii",
        [f"{trozo(D.visible(INICIO_PALABRA))} saca {pct(p_ini, 1)} en el primer sitio"
         + (f"; y {trozo(FINAL_ESPERADO)}, el trozo que terminaría «población» en dos, saca "
            f"{pct(p_fin, 1)} detrás de él." if p_fin is not None else ".")])
    return L, dict(cortes=cortes, desglose=desglose, total=total, p_ini=p_ini, p_fin=p_fin,
                   mas=mas, paso=paso_poblacion)


# ---- selftest --------------------------------------------------------------------------

def selftest(tok, modelo):
    fallos = []

    # 1. TEST NULO — un trozo al azar no se parece al elegido: en cada paso, la 1.ª opción pesa
    #    mucho más que el trozo que sale de una tirada al azar de la caja.
    gen = torch.Generator().manual_seed(0)
    azar = torch.randint(0, len(tok), (50,), generator=gen).tolist()
    p = D.lista(modelo, D.ids_de(tok, D.FRASE))
    medio_azar = float(p[azar].mean())
    primera = float(p.max())
    print(f"[1] test nulo         trozos al azar: {medio_azar:.1e} de media; la 1.ª opción: {primera:.3f}")
    if not (medio_azar < UMBRAL_AZAR and primera > 100 * medio_azar):
        fallos.append("test nulo: un trozo al azar compite con la 1.ª opción")

    # 2. SEÑAL IMPLANTADA — «París» sale en 2 trozos y 14,9237 %, como en maquina_entera_detalle.txt.
    total, desglose = D.prob_palabra(tok, modelo, D.FRASES_PARIS[0], D.PALABRA_BUSCADA)
    leida = leer_tablas((AQUI / DETALLE).read_text(encoding="utf-8"))
    guardada = next(f for t, (_, filas, _) in leida.items() for f in filas
                    if f and f[0] == "«París» entera")[1]
    print(f"[2] señal implantada  «París»: {len(desglose)} trozos, {pct(total, 4)}; guardada: {guardada}")
    if len(desglose) != 2 or pct(total, 4) != guardada:
        fallos.append(f"señal implantada: «París» da {len(desglose)} trozos y {pct(total, 4)}; "
                      f"la salida guardada, {guardada}")

    # 3. INVARIANTE — los ocho primeros pasos que se repiten aquí son, fila a fila, los de la tabla
    #    guardada de maquina_entera_detalle.txt, y los doce son los ocho más cuatro.
    ocho, _ = seguir(tok, modelo, PASOS_DE_LA_TABLA)
    doce, _ = seguir(tok, modelo, PASOS_DE_LA_TABLA + PASOS_MAS)
    mias = [[str(k)] + [celda(c) for c in cands] for k, (cands, _) in enumerate(ocho, 1)]
    suyas = leida[TABLA_OCHO_PASOS][1]
    igual = mias == suyas
    prefijo = [c for c, _ in doce[:PASOS_DE_LA_TABLA]] == [c for c, _ in ocho]
    print(f"[3] invariante        los {PASOS_DE_LA_TABLA} pasos repetidos son los de la tabla guardada: "
          f"{'sí' if igual else 'NO'}; los {PASOS_DE_LA_TABLA + PASOS_MAS} empiezan por esos ocho: "
          f"{'sí' if prefijo else 'NO'}")
    if not igual:
        fallos.append(f"invariante: los ocho pasos no son los de la tabla guardada:\n{mias}\n{suyas}")
    if not prefijo:
        fallos.append("invariante: los doce pasos no empiezan por los ocho")

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
    tok, modelo = D.cargar()
    D.validar_entrada(tok)
    if args.selftest:
        sys.exit(selftest(tok, modelo))
    print(f"máquina: {platform.machine()}, {platform.system()} {platform.release()}, procesador")
    print(f"modelo: {D.MODELO} en {D.DTYPE}")
    print(f"torch {torch.__version__}   transformers {transformers.__version__}")
    print(f"fecha: {datetime.date.today().isoformat()}")
    print()
    lineas, _ = informe(tok, modelo)
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
