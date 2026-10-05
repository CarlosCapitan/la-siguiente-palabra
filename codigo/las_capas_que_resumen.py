#!/usr/bin/env python3
"""
Cuántas capas de un modelo híbrido miran a todo el texto y cuántas llevan un resumen.

En la máquina del capítulo 8 cada capa mira, para cada trozo, a él mismo y a todos los
anteriores, y deja guardadas la etiqueta y el contenido de cada trozo. Algunos modelos recientes
no hacen eso en todas las capas: en unas mantienen esa mirada completa y en otras llevan un
resumen de tamaño fijo, como el del capítulo 6, que no crece con el texto. El artículo que lo
cuenta (Shao y otros, 2026, nota 3) lo dice así de Qwen3.6-27B: «48 of 64 layers use linear
attention», y esas capas guardan «a fixed-size recurrent state».

Este programa no ejecuta ningún modelo: lee el fichero de configuración (config.json) de uno y
cuenta qué tipo de capa es cada una. Se le pasa la ruta del fichero, porque cada cual lo tiene en
un sitio distinto. En el portátil del autor:

    python las_capas_que_resumen.py --selftest
    python las_capas_que_resumen.py \\
        ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-6bit/config.json \\
        > ../datos/salidas/las_capas_que_resumen.txt

No depende de la máquina: es leer un fichero.
"""

# ======================= CONSTANTES =======================

COMPLETA = "full_attention"     # cada trozo mira a él mismo y a todos los anteriores
RESUMEN = "linear_attention"    # resumen de tamaño fijo (la nota 3 del artículo)
LETRA = {COMPLETA: "M", RESUMEN: "R"}
PRIMERAS = 8                    # cuántas capas se pintan una a una

# ==========================================================

import argparse
import json
import os
import sys

from formato import ANCHO_CAJA, comprobar_ancho, miles


def leer(config, origen):
    """Del config.json (ya leído) a (número de capas, tipos, intervalo, largo máximo).
    Los modelos que también leen imágenes guardan lo del texto en «text_config»."""
    t = config.get("text_config", config)
    for clave in ("num_hidden_layers", "layer_types"):
        assert clave in t, f"Se esperaba la clave «{clave}» en {origen}; no está"
    n, tipos = t["num_hidden_layers"], t["layer_types"]
    assert isinstance(n, int) and n > 0, f"Se esperaba un número de capas positivo; hay {n!r}"
    assert len(tipos) == n, (
        f"Se esperaban {n} tipos de capa en «layer_types» de {origen}; hay {len(tipos)}")
    raros = sorted(set(tipos) - set(LETRA))
    assert not raros, f"Se esperaban solo los tipos {sorted(LETRA)} en {origen}; hay {raros}"
    return n, tipos, t.get("full_attention_interval"), t.get("max_position_embeddings")


def contar(n, tipos, intervalo):
    """Cuántas de cada tipo, y si las completas caen exactamente una cada `intervalo` capas
    (la última de cada grupo), que es lo que el fichero dice que hace."""
    completas = sum(1 for x in tipos if x == COMPLETA)
    resumen = n - completas
    cuadra = None
    if intervalo is not None:
        cuadra = all((x == COMPLETA) == ((i + 1) % intervalo == 0) for i, x in enumerate(tipos))
    return completas, resumen, cuadra


def imprimir(nombre, n, tipos, intervalo, largo):
    completas, resumen, cuadra = contar(n, tipos, intervalo)
    out = [f"LAS {n} CAPAS DE UN MODELO HÍBRIDO",
           f"({nombre}, leído de su config.json)",
           "",
           f"  {'mirada completa (cada trozo con él y los anteriores):':<54} {completas:>3} capas",
           f"  {'resumen de tamaño fijo:':<54} {resumen:>3} capas"]
    if intervalo is not None:
        out.append(f"  una de mirada completa cada {intervalo} capas, la última de cada "
                   f"grupo: {'sí' if cuadra else 'no'}")
    out += ["",
            f"  las {PRIMERAS} primeras: " + " ".join(LETRA[x] for x in tipos[:PRIMERAS]),
            "  M: mirada completa.  R: resumen de tamaño fijo."]
    if largo is not None:
        out.append(f"  texto más largo que admite: {miles(largo)} trozos")
    print("\n".join(comprobar_ancho(out, ANCHO_CAJA)))


def selftest():
    ok = True

    # [1] test nulo: un modelo normal, con mirada completa en todas sus capas, no tiene ninguna
    #     capa de resumen y no tiene por qué cuadrar con ningún intervalo.
    normal = {"num_hidden_layers": 24, "layer_types": [COMPLETA] * 24}
    n, tipos, intervalo, _ = leer(normal, "normal")
    c, r, cuadra = contar(n, tipos, intervalo)
    p1 = (c, r, cuadra) == (24, 0, None)
    ok &= p1
    print(f"[1] test nulo         24 capas de mirada completa: {c} completas, {r} de resumen: "
          f"{'bien' if p1 else 'MAL'}")

    # [2] señal implantada: 64 capas con una completa cada 4 (la última de cada grupo) dan 16 y
    #     48, y cuadran; si se mueve UNA completa de sitio, los recuentos no cambian pero tiene
    #     que dejar de cuadrar.
    tipos = [COMPLETA if (i + 1) % 4 == 0 else RESUMEN for i in range(64)]
    hecho = {"text_config": {"num_hidden_layers": 64, "layer_types": tipos,
                             "full_attention_interval": 4}}
    n, t, iv, _ = leer(hecho, "implantado")
    a = contar(n, t, iv)
    movido = list(tipos)
    movido[3], movido[4] = movido[4], movido[3]
    b = contar(64, movido, 4)
    p2 = a == (16, 48, True) and b == (16, 48, False)
    ok &= p2
    print(f"[2] señal implantada  una de cada 4: {a[0]} y {a[1]}, cuadra {a[2]}; con una movida: "
          f"{b[0]} y {b[1]}, cuadra {b[2]}: {'bien' if p2 else 'MAL'}")

    # [3] invariante: lo que no cuadra con el fichero revienta en vez de contarse mal: más tipos
    #     que capas, y un tipo que no se conoce.
    p3 = True
    for malo in ({"num_hidden_layers": 3, "layer_types": [COMPLETA] * 4},
                 {"num_hidden_layers": 2, "layer_types": [COMPLETA, "sliding_attention"]}):
        try:
            leer(malo, "malo")
            p3 = False
        except AssertionError:
            pass
    ok &= p3
    print(f"[3] invariante        tipos de más y tipo desconocido revientan: "
          f"{'bien' if p3 else 'MAL'}")

    print()
    print("SELFTEST: las tres pruebas pasan." if ok else "SELFTEST: FALLA.")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("config", nargs="?", help="ruta del config.json del modelo")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    assert args.config, "Se esperaba la ruta de un config.json; no se ha dado ninguna"
    ruta = os.path.expanduser(args.config)
    with open(ruta, encoding="utf-8") as fh:
        config = json.load(fh)
    nombre = "/".join(os.path.normpath(ruta).split(os.sep)[-3:-1])   # autor/modelo
    imprimir(nombre, *leer(config, ruta))


if __name__ == "__main__":
    main()
