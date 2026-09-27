#!/usr/bin/env python3
"""
Capítulo 3 — pares e impares en un reloj digital, con tres pesos puestos a mano (L17, T16).

La maqueta de Carlos (L9, `notas/maquetas/L9-pares-e-impares-reloj.png`), en gris y con dos
cambios: el listón es «pasa de 0,5», como lo dice el libro desde el capítulo 2, y el paso 2 dice
«los mismos pesos», no «el mismo baremo».

  1. Tres pesos puestos a mano y el listón.
  2. Los mismos pesos aciertan los diez: el total de cada dígito.

Los diez dígitos, uno a uno, van en la figura anterior (`figura_diez_digitos.py`, L22), y por eso
esta ya no los repite: tenía un primer paso con ellos que se quitó el 27 de septiembre.

Los pesos, el listón y los totales NO se calculan aquí: se leen de
`datos/salidas/reloj_a_mano.txt`. Qué segmentos enciende cada dígito sale de la tabla de
`siete_segmentos.py`, la misma que imprime el libro.

Uso:
    python figura_reloj_a_mano.py --selftest
    python figura_reloj_a_mano.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/reloj_a_mano.txt"
DESTINO = "../figuras/reloj_a_mano.png"
ALTO = 4.15                      # pulgadas (caben 6,55)

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from infografia import COLOR, GRIS, Lienzo
from siete_segmentos import SEGMENTOS, tabla_de_segmentos
from figura_diez_digitos import dibuja_digito   # el mismo reloj que la figura de los diez dígitos

AQUI = Path(__file__).resolve().parent

# Cada segmento como un trazo en una cajita de 1 de ancho por 2 de alto, en el orden de SEGMENTOS.
TRAZOS = {
    "el de arriba": ((0, 2), (1, 2)),
    "el de arriba izquierda": ((0, 1), (0, 2)),
    "el de arriba derecha": ((1, 1), (1, 2)),
    "el del medio": ((0, 1), (1, 1)),
    "el de abajo izquierda": ((0, 0), (0, 1)),
    "el de abajo derecha": ((1, 0), (1, 1)),
    "el de abajo": ((0, 0), (1, 0)),
}


def _numero(s):
    return float(s.replace("−", "-").replace(",", "."))


def leer(ruta):
    """Pesos, listón, cuántos dígitos se saltan la regla con cada segmento, y los totales."""
    texto = Path(ruta).read_text(encoding="utf-8")
    assert "3. TRES PESOS PUESTOS A MANO" in texto, f"se esperaba la sección 3 en {ruta}"
    assert "1. QUÉ DÍGITOS SE SALTAN LA REGLA" in texto, f"se esperaba la sección 1 en {ruta}"
    uno = texto.split("1. QUÉ DÍGITOS SE SALTAN LA REGLA", 1)[1].split("EL DE ABAJO", 1)[0]
    solos = {m.group(1): int(m.group(2))
             for m in re.finditer(r"^(el (?:\S+ )*?\S+)\s{2,}.*?(\d+)$", uno, re.M)}
    assert sorted(solos) == sorted(SEGMENTOS), f"se esperaban los siete segmentos; hay {sorted(solos)}"
    tres = texto.split("3. TRES PESOS PUESTOS A MANO", 1)[1]
    cabeza = tres.split("los otros cuatro", 1)[0]
    pesos = {n.strip(): int(v) for n, v in re.findall(r"^(el [^\n]*?)\s+([+-]\d+)$", cabeza, re.M)}
    assert pesos, f"se esperaban los pesos en la sección 3 de {ruta}"
    m = re.search(r"el total tiene que pasar de ([\d,]+)", tres)
    assert m, f"se esperaba el listón en {ruta}"
    liston = _numero(m.group(1))
    filas = re.findall(r"^(\d)\s+\S+\s+\S+\s+\S+\s+(\S+)\s+(par|impar)$", tres, re.M)
    assert [int(d) for d, _, _ in filas] == list(range(10)), \
        f"se esperaban los diez dígitos en la cuenta de {ruta}; hay {[d for d, _, _ in filas]}"
    totales = [int(_numero(t)) for _, t, _ in filas]
    dice = [s == "par" for _, _, s in filas]
    return {"pesos": pesos, "liston": liston, "totales": totales, "dice": dice, "solos": solos}


def digito(L, x, y, alto, encendidos, destacar=(), grueso=2.6):
    """Un dígito de reloj con su esquina de abajo izquierda en (x, y). Los apagados, en gris muy
    claro, para que se vea que están ahí."""
    p = L.p
    ancho = alto / 2.0
    for s, ((x1, y1), (x2, y2)) in TRAZOS.items():
        enc = s in encendidos
        color = p.tinta if enc else p.neutro
        if destacar:
            color = p.tinta if s in destacar else p.neutro
        L.ax.plot([x + x1 * ancho, x + x2 * ancho], [y + y1 * alto / 2, y + y2 * alto / 2],
                  color=color, linewidth=grueso, solid_capstyle="round", zorder=3)


def cifra(v):
    return str(v).replace("-", "−")


def puntos(v):
    return f"{cifra(v)} punto" + ("" if abs(v) == 1 else "s")


def dibujar(d, paleta, ruta):
    X = tabla_de_segmentos()
    enc = {k: {s for j, s in enumerate(SEGMENTOS) if X[k, j]} for k in range(10)}
    L = Lienzo("Pares e impares en un reloj digital",
               "Un solo comité los distingue sumando lo que aporta cada segmento encendido.",
               paleta, alto=ALTO)
    p = L.p
    cols = [34 + 13.5 * i for i in range(5)]

    mejor = min(d["solos"].values())
    assert mejor > 0, "la salida dice que con un segmento no se salta la regla ningún dígito"

    # 1. tres pesos
    x0, y, _ = L.panel(1, "Tres pesos puestos a mano", 34)
    xd, yd, h = 36, y - 24, 17
    dibuja_digito(L.ax, xd, yd, h / 2, set(d["pesos"]), p.acento)
    ancho = h / 2
    etiquetas = {"el de arriba": (xd + ancho / 2, yd + h, xd + ancho + 4, yd + h + 1.0, "left"),
                 "el de arriba izquierda": (xd, yd + 3 * h / 4, xd - 3, yd + 3 * h / 4, "right"),
                 "el de abajo izquierda": (xd, yd + h / 4, xd - 3, yd + h / 4, "right")}
    for s, v in d["pesos"].items():
        assert s in etiquetas, f"la figura no sabe dónde poner la etiqueta de «{s}»"
        ax_, ay, tx, ty, ha = etiquetas[s]
        L.ax.plot([ax_, tx], [ay, ty], color=p.suave,
                  linewidth=0.6, zorder=2)
        verbo = "suma" if v > 0 else "resta"
        L.texto(tx, ty + 1.6, s, ha=ha, tam=7.2)
        L.texto(tx, ty - 1.6, f"{verbo} {abs(v)}", ha=ha, tam=8.0, negrita=True)
    L.texto(xd + ancho + 2.5, yd + h / 2 - 3.5, "los otros cuatro", tam=7.0, color=p.suave)
    L.texto(xd + ancho + 2.5, yd + h / 2 - 6.5, "no suman nada", tam=7.0, color=p.suave)
    from matplotlib.patches import FancyBboxPatch
    bx, by, bw, bh = 64, y - 17, 30, 16
    L.ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=1.0",
                                  facecolor="white", edgecolor=p.tinta, linewidth=1.0))
    lis = f"{d['liston']:g}".replace(".", ",")
    L.texto(bx + bw / 2, by + bh - 3.8, f"Listón: {lis}", ha="center", tam=10, negrita=True)
    L.texto(bx + bw / 2, by + bh - 8.8, f"si el total pasa de {lis}: par", ha="center", tam=7.4)
    L.texto(bx + bw / 2, by + bh - 12.4, "si no pasa: impar", ha="center", tam=7.4)
    L.texto(bx + bw / 2, y - 20.5, "un segmento solo suma", ha="center", tam=7.0, color=p.suave)
    L.texto(bx + bw / 2, y - 23.3, "si está encendido", ha="center", tam=7.0, color=p.suave)
    L.flecha()

    # 2. los diez totales
    x0, y, _ = L.panel(2, "Los mismos pesos aciertan los diez", 31)
    for fila, (nombre, digs) in enumerate((("pares", [0, 2, 4, 6, 8]),
                                          ("impares", [1, 3, 5, 7, 9]))):
        yc = y - 4.5 - fila * 11
        L.texto(x0 + 1, yc, nombre, negrita=True, tam=8.4)
        for c, k in zip(cols, digs):
            t = d["totales"][k]
            pasa = t > d["liston"]
            assert pasa == d["dice"][k], f"el {k}: el total y lo que dice la salida no cuadran"
            assert pasa == (k % 2 == 0), f"el {k}: la figura diría que falla"
            L.ax.add_patch(FancyBboxPatch((c - 6, yc - 5.0), 12, 10.0,
                                          boxstyle="round,pad=0,rounding_size=0.8",
                                          facecolor=p.fondo if pasa else "white",
                                          edgecolor=p.tinta if pasa else p.marco,
                                          linewidth=1.0 if pasa else 0.8))
            L.texto(c, yc + 2.5, str(k), ha="center", tam=10, negrita=True)
            L.texto(c, yc - 1.0, puntos(t), ha="center", tam=6.8)
            L.texto(c, yc - 3.4, "pasa" if pasa else "no pasa", ha="center", tam=6.4,
                    color=p.tinta if pasa else p.suave, negrita=pasa)
    L.pie("Pesos puestos a mano para enseñar la cuenta, comprobados con los diez dígitos del reloj.\n"
          "No los ha encontrado una máquina, y no son una medición con dígitos escritos a mano.")
    L.guardar(ruta)


def selftest():
    fallos = []
    d = leer(AQUI / SALIDA)
    # 1. TEST NULO: un texto sin la sección no da figura.
    import os, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("nada que ver aquí\n")
    try:
        leer(fh.name)
        fallos.append("test nulo: leyó pesos de un texto que no los tiene")
    except (AssertionError, IndexError):
        pass
    finally:
        os.unlink(fh.name)
    print("[1] test nulo         un texto sin la cuenta no da ninguna figura")
    # 2. SEÑAL: con los pesos leídos y la tabla de segmentos, la cuenta rehecha da los diez
    #    totales que dice la salida, y los diez deciden bien.
    X = tabla_de_segmentos()
    w = np.array([d["pesos"].get(s, 0) for s in SEGMENTOS], dtype=float)
    rehechos = [int(v) for v in X @ w]
    bien = sum((t > d["liston"]) == (k % 2 == 0) for k, t in enumerate(rehechos))
    print(f"[2] señal             totales rehechos {rehechos}; iguales a la salida: "
          f"{'sí' if rehechos == d['totales'] else 'NO'}; aciertan {bien} de 10")
    if rehechos != d["totales"] or bien != 10:
        fallos.append("señal: la cuenta rehecha no da los totales de la salida, o no acierta los diez")
    # 3. INVARIANTE: cabe en la página y se dibuja en las dos paletas.
    for pal in (COLOR, GRIS):
        dibujar(d, pal, "/dev/null")
    print("[3] invariante        la figura se dibuja en color y en gris sin salirse de la página")
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
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    dibujar(leer(AQUI / SALIDA), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
