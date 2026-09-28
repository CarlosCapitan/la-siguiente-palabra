#!/usr/bin/env python3
"""
Capítulo 12 — las tres fases del adiestramiento, siguiendo un solo enunciado (L24, D19).

El capítulo cuenta el procedimiento de 2022 con verbos —«se ajusta para imitarlas», «se entrena
un segundo modelo», «se sigue ajustando para sacar nota alta»— y sin un solo enunciado a la
vista. Aquí se sigue uno, «¿Cuál es la capital de Francia?», por las tres fases. Las respuestas
de la máquina son de verdad: las que dieron las máquinas del capítulo a esa misma pregunta, y se
leen de `datos/salidas/crudo_o_adiestrado.txt`. El orden que les pone la persona y las notas del
juez son un ejemplo, y la figura lo dice: no se ha entrenado ningún juez para este libro.

Uso:
    python figura_tres_fases.py --selftest
    python figura_tres_fases.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/crudo_o_adiestrado.txt"
DESTINO = "../figuras/tres_fases.png"
ALTO = 5.15
PREGUNTA = "¿Cuál es la capital de Francia?"
LARGO_RESPUESTA = 58        # letras de cada respuesta que caben en su renglón
# Qué respuestas de la salida se usan, por su rótulo en el bloque 2 de cada tamaño, y en qué
# orden las pondría una persona (ejemplo, no medición): la corta y buena, la buena que se va por
# las ramas, la falsa.
ELEGIDAS = [("7.000M", "adiestrado, con su formato"), ("500M", "adiestrado, enunciado tal cual"),
            ("500M", "en crudo")]
NOTAS_EJEMPLO = [0.9, 0.55, 0.15]   # largo de la barra de la nota, de ejemplo

# ==========================================================

import argparse
import ast
import re
import sys

from matplotlib.patches import FancyArrowPatch, Rectangle

from infografia import COLOR, GRIS, Lienzo


def leer(ruta=SALIDA):
    t = open(ruta, encoding="utf-8").read()
    respuestas = {}
    for bloque in t.split("TAMAÑO ")[1:]:
        tam = bloque.split("\n", 1)[0].strip()
        for rot, r in re.findall(r"^  (en crudo|adiestrado, enunciado tal cual|adiestrado, con su formato) -> (.*)$",
                                 bloque, re.M):
            respuestas[(tam, rot)] = ast.literal_eval(r)
    for k in ELEGIDAS:
        assert k in respuestas, f"falta la respuesta {k} en {ruta}"
    return [respuestas[k] for k in ELEGIDAS]


def renglon(r, n=LARGO_RESPUESTA):
    r = " ".join(r.split())
    return "«" + (r if len(r) <= n else r[:n - 1].rstrip() + "…") + "»"


def dibujar(resp, paleta, ruta):
    L = Lienzo("Un enunciado, tres fases",
               f"El enunciado: «{PREGUNTA}». Las respuestas que salen son de verdad:\n"
               "las de tres máquinas de este capítulo. El orden y las notas, un ejemplo sin medir.",
               paleta, alto=ALTO)
    p, ax = L.p, L.ax
    notas = {}
    # 1
    x, y, w = L.panel(1, "Demostraciones: una persona escribe la respuesta", 22.5)
    L.texto(x, y - 0.5, "Una persona escribe a mano una respuesta ejemplar, como esta:", tam=7.2, color=p.suave)
    L.texto(x + 2, y - 4.3, renglon(resp[0]), tam=7.4)
    L.texto(x, y - 8.4, "La máquina se ajusta para escribir como ella: el ajuste del capítulo 4,",
            tam=7.2, color=p.suave)
    L.texto(x, y - 11.4, "con esa respuesta como la buena.", tam=7.2, color=p.suave)
    L.flecha()
    # 2
    x, y, w = L.panel(2, "Un juez: la persona ordena; se entrena otro modelo", 38.5)
    L.texto(x, y - 0.5, "Hay varias respuestas al mismo enunciado; la persona las ordena, de mejor a peor:",
            tam=7.2, color=p.suave)
    for i, r in enumerate(resp):
        L.texto(x + 1, y - 4.3 - i * 3.4, f"{i + 1}.º", tam=7.4, negrita=True)
        L.texto(x + 5, y - 4.3 - i * 3.4, renglon(r), tam=7.4)
    y2 = y - 16.0
    L.texto(x, y2, "Con miles de ordenaciones así se entrena el juez: recibe una respuesta y da",
            tam=7.2, color=p.suave)
    L.texto(x, y2 - 3.0, "un solo número, su nota, como el total del comité del capítulo 3:",
            tam=7.2, color=p.suave)
    for i, (r, v) in enumerate(zip(resp, NOTAS_EJEMPLO)):
        yy = y2 - 7.0 - i * 2.6
        L.texto(x + 1, yy, f"{i + 1}.º", tam=7.0)
        ax.add_patch(Rectangle((x + 5, yy - 0.9), 40 * v, 1.8, facecolor=p.acento, edgecolor="none"))
        notas[i] = v
    L.texto(x + 48, y2 - 9.6, "nota del juez\n(largo de la barra)", tam=6.8, color=p.suave)
    L.flecha()
    # 3
    x, y, w = L.panel(3, "Estudiar para el juez: ya sin personas", 21.0)
    L.texto(x, y - 0.5, "La máquina escribe; el juez pone nota; y sus pesos se mueven un poco para",
            tam=7.2, color=p.suave)
    L.texto(x, y - 3.5, "que la próxima vez salga más lo que sacó más nota, y menos lo que sacó menos.",
            tam=7.2, color=p.suave)
    L.texto(x, y - 7.3, "Miles de veces. Nadie comprueba si la respuesta es verdad:", tam=7.2, color=p.suave)
    L.texto(x, y - 10.3, "solo si se parece a lo que prefirieron aquellas personas.", tam=7.2, negrita=True)
    assert L.y > 0.5, f"no cabe: sobran {0.5 - L.y:.1f} unidades"
    L.guardar(ruta)
    return notas


def selftest():
    fallos = []
    resp = leer()
    # 1. TEST NULO — una salida sin el bloque de la pregunta directa tiene que reventar.
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write("TAMAÑO 500M\nnada\n"); tmp.close()
    try:
        leer(tmp.name); revienta = False
    except AssertionError:
        revienta = True
    os.unlink(tmp.name)
    print(f"[1] test nulo         salida sin respuestas: {'revienta' if revienta else 'NO revienta'}")
    if not revienta:
        fallos.append("test nulo: dibuja sin respuestas")
    # 2. SEÑAL — las respuestas son las que cuenta el capítulo: París la primera, Nápoles la última.
    ok = "París" in resp[0] and "Nápoles" in resp[2]
    print(f"[2] señal             primera {resp[0]!r}; última {resp[2]!r}")
    if not ok:
        fallos.append("señal: las respuestas no son las esperadas")
    # 3. INVARIANTE — el orden de ejemplo y las barras van juntos: la primera, la barra más larga.
    n = dibujar(resp, GRIS, "/dev/null")
    dibujar(resp, COLOR, "/dev/null")
    ok = n[0] > n[1] > n[2]
    print(f"[3] invariante        barras de mayor a menor: {'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: las barras no siguen el orden")
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
    dibujar(leer(), GRIS, DESTINO)
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
